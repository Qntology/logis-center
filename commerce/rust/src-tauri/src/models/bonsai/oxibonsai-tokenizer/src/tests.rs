//! Integration tests for `oxibonsai-tokenizer`.
//!
//! These tests exercise the public API at a higher level than the unit tests
//! embedded in each module, verifying end-to-end encode/decode behaviour and
//! cross-module interactions (e.g., chat templates + batch encoding).

#[cfg(test)]
mod integration {
    use crate::{
        tokenizer::{OxiTokenizer, TokenizerConfig},
        utils::{BatchEncoder, BatchEncoding, ChatTemplate, PaddingStrategy, TruncationSide},
    };

    // ── OxiTokenizer: encode/decode roundtrip ────────────────────────────────

    /// Verify that ASCII text encodes to a non-empty sequence and decodes back
    /// to a string that contains the original characters.
    #[test]
    fn test_oxi_tokenizer_encode_decode_ascii() {
        let tok = OxiTokenizer::char_level_stub(256);

        // Encode
        let ids = tok.encode("hello").expect("encode must succeed");
        assert!(!ids.is_empty(), "encoded sequence must be non-empty");

        // Decode — the char-level stub assigns each printable ASCII char its own
        // token, so decoding should reproduce the original string.
        let decoded = tok.decode(&ids).expect("decode must succeed");
        // The decoded text should at minimum contain the characters.
        for ch in "hello".chars() {
            assert!(
                decoded.contains(ch),
                "decoded text should contain '{ch}': got {decoded:?}"
            );
        }
    }

    /// Single-character encoding produces exactly one token in char-level mode.
    #[test]
    fn test_oxi_tokenizer_encode_single_char() {
        let tok = OxiTokenizer::char_level_stub(256);
        let ids_a = tok.encode("a").expect("encode a");
        let ids_b = tok.encode("b").expect("encode b");
        assert_eq!(ids_a.len(), 1, "single char → single token");
        assert_eq!(ids_b.len(), 1, "single char → single token");
        assert_ne!(ids_a[0], ids_b[0], "'a' and 'b' must have different IDs");
    }

    // ── OxiTokenizer: char-level stub vocabulary size ────────────────────────

    /// The stub vocab never exceeds the requested size and always has the four
    /// reserved special tokens.
    #[test]
    fn test_oxi_tokenizer_char_level_stub_vocab_size() {
        for requested in [4usize, 10, 50, 100, 256] {
            let tok = OxiTokenizer::char_level_stub(requested);
            let sz = tok.vocab_size();
            assert!(
                sz <= requested,
                "vocab_size {sz} must be <= requested {requested}"
            );
            assert!(sz >= 4, "vocab_size must include at least 4 special tokens");
        }
    }

    // ── OxiTokenizer: special token IDs ─────────────────────────────────────

    /// All four reserved special tokens (UNK/BOS/EOS/PAD) are correctly
    /// recognised by `is_special()`.
    #[test]
    fn test_oxi_tokenizer_special_tokens() {
        let tok = OxiTokenizer::char_level_stub(256);

        assert!(tok.is_special(0), "ID 0 must be UNK — a special token");
        assert!(tok.is_special(1), "ID 1 must be BOS — a special token");
        assert!(tok.is_special(2), "ID 2 must be EOS — a special token");
        assert!(tok.is_special(3), "ID 3 must be PAD — a special token");

        // IDs 4+ are regular tokens.
        assert!(
            !tok.is_special(4),
            "ID 4 is a regular printable ASCII token"
        );
        assert!(!tok.is_special(10), "ID 10 is a regular token");
    }

    /// BOS and EOS are inserted at the correct positions when enabled via `from_json`.
    #[test]
    fn test_oxi_tokenizer_bos_eos_injection() {
        // Build via from_json so we can pass a config with BOS/EOS enabled.
        let vocab_json = r#"{
            "a":10,"b":11,"ab":20,
            "<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3
        }"#;
        let merges_json = r#"[["a","b"]]"#;
        let config = TokenizerConfig {
            add_bos: true,
            add_eos: true,
            ..TokenizerConfig::default()
        };
        let tok = OxiTokenizer::from_json(vocab_json, merges_json, config)
            .expect("from_json must succeed");

        let ids = tok.encode("ab").expect("encode must succeed");
        assert_eq!(
            ids.first().copied(),
            Some(1u32),
            "first token must be BOS (id=1)"
        );
        assert_eq!(
            ids.last().copied(),
            Some(2u32),
            "last token must be EOS (id=2)"
        );
    }

    // ── BatchEncoding: sizes match across batch ──────────────────────────────

    /// After padding to `Longest`, all sequences in the batch share the same
    /// padded length, and `lengths` reflects the original (pre-pad) lengths.
    #[test]
    fn test_batch_encoding_sizes_match() {
        let tok = OxiTokenizer::char_level_stub(256);
        let enc = BatchEncoder::new(&tok).with_padding(PaddingStrategy::Longest);

        let texts = ["a", "hello", "hi there"];
        let result: BatchEncoding = enc.encode_batch(&texts).expect("batch encode must succeed");

        assert_eq!(
            result.batch_size(),
            3,
            "batch size must equal number of inputs"
        );

        // All padded sequences must be the same length.
        let padded_len = result.max_seq_len();
        for ids in &result.input_ids {
            assert_eq!(
                ids.len(),
                padded_len,
                "every padded sequence must have length {padded_len}"
            );
        }

        // The `lengths` field must hold the pre-padding token counts.
        for (i, &len) in result.lengths.iter().enumerate() {
            let real_tokens: Vec<u32> = result.input_ids[i][..len].to_vec();
            // Real tokens must not be the pad token (3).
            for &id in &real_tokens {
                assert_ne!(
                    id, 3u32,
                    "position within `length` must not be the pad token"
                );
            }
        }

        // Attention mask length must equal the padded sequence length.
        for mask in &result.attention_mask {
            assert_eq!(
                mask.len(),
                padded_len,
                "mask length must equal padded seq len"
            );
        }
    }

    /// Truncation to a fixed length produces sequences of exactly that length.
    #[test]
    fn test_batch_encoding_truncation_length() {
        let tok = OxiTokenizer::char_level_stub(256);
        let limit = 3usize;
        let enc = BatchEncoder::new(&tok)
            .with_max_length(limit)
            .with_truncation(TruncationSide::Right);

        let texts = ["abcde", "hello world", "x"];
        let result = enc.encode_batch(&texts).expect("encode must succeed");

        for (i, len) in result.lengths.iter().enumerate() {
            assert!(
                *len <= limit,
                "sequence {i} length {len} exceeds max_length {limit}"
            );
        }
    }

    // ── ChatTemplate: system + user + assistant ──────────────────────────────

    /// A full system / user / assistant conversation formats correctly under
    /// the ChatML template and can be parsed back by `extract_user_message`.
    #[test]
    fn test_chat_template_system_user_assistant() {
        let tmpl = ChatTemplate::chatml();

        let messages = [
            ("system", "You are a helpful coding assistant."),
            ("user", "How do I reverse a string in Rust?"),
            ("assistant", "Use `.chars().rev().collect::<String>()`."),
        ];

        let formatted = tmpl.format(&messages);

        // All roles and their content must appear in the output.
        assert!(
            formatted.contains("<|im_start|>system"),
            "system block must be present"
        );
        assert!(
            formatted.contains("You are a helpful coding assistant."),
            "system content must be present"
        );
        assert!(
            formatted.contains("<|im_start|>user"),
            "user block must be present"
        );
        assert!(
            formatted.contains("How do I reverse a string in Rust?"),
            "user content must be present"
        );
        assert!(
            formatted.contains("<|im_start|>assistant"),
            "assistant block must be present"
        );
        assert!(
            formatted.contains(".chars().rev().collect"),
            "assistant content must be present"
        );

        // All blocks must be properly terminated.
        let end_count = formatted.matches("<|im_end|>").count();
        assert_eq!(
            end_count, 3,
            "each of the 3 messages must have an <|im_end|>"
        );

        // Extracting the last user message must work.
        let user_msg = ChatTemplate::extract_user_message(&formatted);
        assert_eq!(
            user_msg.as_deref(),
            Some("How do I reverse a string in Rust?"),
            "extract_user_message must return the last user message"
        );
    }

    /// A two-turn conversation where the last message is from the user.
    #[test]
    fn test_chat_template_multi_turn_last_user() {
        let tmpl = ChatTemplate::chatml();
        let messages = [
            ("user", "First question"),
            ("assistant", "First answer"),
            ("user", "Follow-up question"),
        ];
        let formatted = tmpl.format(&messages);

        let extracted = ChatTemplate::extract_user_message(&formatted);
        assert_eq!(
            extracted.as_deref(),
            Some("Follow-up question"),
            "must extract the very last user message from a multi-turn conversation"
        );
    }

    // ── from_json roundtrip integration ──────────────────────────────────────

    /// Build a tokenizer from JSON and verify encode→decode produces the
    /// original characters.
    #[test]
    fn test_from_json_encode_decode_roundtrip() {
        let vocab_json = r#"{"a":10,"b":11,"ab":20,"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3}"#;
        let merges_json = r#"[["a","b"]]"#;
        let tok = OxiTokenizer::from_json(vocab_json, merges_json, TokenizerConfig::default())
            .expect("from_json must succeed");

        // "ab" should encode to the single merged token 20.
        let ids = tok.encode("ab").expect("encode ab");
        assert!(ids.contains(&20), "merged token 20 expected in {ids:?}");
    }
}
