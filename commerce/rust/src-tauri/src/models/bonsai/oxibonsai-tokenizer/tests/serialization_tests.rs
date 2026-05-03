use oxibonsai_tokenizer::{
    base64_decode, base64_encode,
    trainer::{BpeTrainer, TrainerConfig},
    SerializationError, TokenizerState,
};
use std::io::{BufReader, Cursor};

// ── 1. base64_roundtrip_ascii ─────────────────────────────────────────────────

#[test]
fn base64_roundtrip_ascii() {
    let original = b"Hello, World!";
    let encoded = base64_encode(original);
    let decoded = base64_decode(&encoded).expect("decode should succeed");
    assert_eq!(decoded, original);
}

// ── 2. base64_roundtrip_binary ────────────────────────────────────────────────

#[test]
fn base64_roundtrip_binary() {
    // Include non-ASCII bytes (all 256 values)
    let original: Vec<u8> = (0u8..=255u8).collect();
    let encoded = base64_encode(&original);
    let decoded = base64_decode(&encoded).expect("decode should succeed");
    assert_eq!(decoded, original);
}

// ── 3. base64_decode_invalid ──────────────────────────────────────────────────

#[test]
fn base64_decode_invalid() {
    // '!' is not a valid base64 character
    let result = base64_decode("SGVsbG8!!");
    assert!(result.is_err(), "expected error for invalid base64");
}

// ── 4. tokenizer_state_new_empty ─────────────────────────────────────────────

#[test]
fn tokenizer_state_new_empty() {
    let state = TokenizerState::new();
    assert_eq!(state.vocab_size(), 0);
    assert!(state.vocab.is_empty());
    assert!(state.merges.is_empty());
    assert!(state.special_tokens.is_empty());
}

// ── 5. tokenizer_state_save_load_empty ───────────────────────────────────────

#[test]
fn tokenizer_state_save_load_empty() {
    let state = TokenizerState::new();
    let mut buf: Vec<u8> = Vec::new();
    state.save_to(&mut buf).expect("save_to should succeed");

    let mut reader = BufReader::new(buf.as_slice());
    let loaded = TokenizerState::load_from(&mut reader).expect("load_from should succeed");

    assert_eq!(loaded.vocab_size(), 0);
    assert!(loaded.merges.is_empty());
    assert!(loaded.special_tokens.is_empty());
}

// ── 6. tokenizer_state_save_load_with_vocab ──────────────────────────────────

#[test]
fn tokenizer_state_save_load_with_vocab() {
    let mut state = TokenizerState::new();
    state.vocab.insert(10, "hello".to_string());
    state.vocab.insert(11, "world".to_string());
    state.vocab.insert(12, "foo bar".to_string()); // with space
    state.vocab.insert(13, "<unk>".to_string());

    let mut buf: Vec<u8> = Vec::new();
    state.save_to(&mut buf).expect("save should succeed");

    let mut reader = BufReader::new(buf.as_slice());
    let loaded = TokenizerState::load_from(&mut reader).expect("load should succeed");

    assert_eq!(loaded.vocab_size(), 4);
    assert_eq!(loaded.vocab.get(&10), Some(&"hello".to_string()));
    assert_eq!(loaded.vocab.get(&11), Some(&"world".to_string()));
    assert_eq!(loaded.vocab.get(&12), Some(&"foo bar".to_string()));
    assert_eq!(loaded.vocab.get(&13), Some(&"<unk>".to_string()));
}

// ── 7. tokenizer_state_save_load_with_merges ─────────────────────────────────

#[test]
fn tokenizer_state_save_load_with_merges() {
    let mut state = TokenizerState::new();
    state.vocab.insert(0, "a".to_string());
    state.vocab.insert(1, "b".to_string());
    state.vocab.insert(2, "ab".to_string());
    state.merges.push((0, 1, 2));
    state.merges.push((2, 0, 3));

    let mut buf: Vec<u8> = Vec::new();
    state.save_to(&mut buf).expect("save should succeed");

    let mut reader = BufReader::new(buf.as_slice());
    let loaded = TokenizerState::load_from(&mut reader).expect("load should succeed");

    assert_eq!(loaded.merges.len(), 2);
    assert_eq!(loaded.merges[0], (0, 1, 2));
    assert_eq!(loaded.merges[1], (2, 0, 3));
}

// ── 8. tokenizer_state_save_load_special_tokens ──────────────────────────────

#[test]
fn tokenizer_state_save_load_special_tokens() {
    let mut state = TokenizerState::new();
    state.vocab.insert(0, "<bos>".to_string());
    state.vocab.insert(1, "<eos>".to_string());
    state.special_tokens.insert("<bos>".to_string(), 0);
    state.special_tokens.insert("<eos>".to_string(), 1);

    let mut buf: Vec<u8> = Vec::new();
    state.save_to(&mut buf).expect("save should succeed");

    let mut reader = BufReader::new(buf.as_slice());
    let loaded = TokenizerState::load_from(&mut reader).expect("load should succeed");

    assert_eq!(loaded.special_tokens.get("<bos>"), Some(&0));
    assert_eq!(loaded.special_tokens.get("<eos>"), Some(&1));
}

// ── 9. tokenizer_state_save_tempfile ─────────────────────────────────────────

#[test]
fn tokenizer_state_save_tempfile() {
    let mut state = TokenizerState::new();
    state.vocab.insert(42, "rust".to_string());

    let mut tmp = std::env::temp_dir();
    tmp.push("oxibonsai_serialization_test.txt");

    state.save(&tmp).expect("save to file should succeed");
    let loaded = TokenizerState::load(&tmp).expect("load from file should succeed");

    assert_eq!(loaded.vocab.get(&42), Some(&"rust".to_string()));

    // Clean up
    let _ = std::fs::remove_file(&tmp);
}

// ── 10. tokenizer_state_invalid_magic ────────────────────────────────────────

#[test]
fn tokenizer_state_invalid_magic() {
    let bad_data = b"not a tokenizer\nvocab_size 0\nmerges 0\n";
    let mut reader = BufReader::new(bad_data.as_ref());
    let result = TokenizerState::load_from(&mut reader);

    match result {
        Err(SerializationError::InvalidMagic { .. }) => {} // expected
        other => panic!("expected InvalidMagic, got: {other:?}"),
    }
}

// ── 11. tokenizer_state_from_trained ─────────────────────────────────────────

#[test]
fn tokenizer_state_from_trained() {
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let corpus = ["hello world", "hello rust", "world rust"];
    let trained = trainer.train(&corpus).expect("training should succeed");

    let state = TokenizerState::from_trained(&trained);

    // State must have at least as many vocab entries as the trained tokenizer
    assert!(state.vocab_size() > 0);
    assert_eq!(state.vocab_size(), trained.vocab_size());
    assert_eq!(state.merges.len(), trained.merges.len());
}

// ── 12. tokenizer_state_to_oxi_tokenizer ─────────────────────────────────────

#[test]
fn tokenizer_state_to_oxi_tokenizer() {
    let mut state = TokenizerState::new();
    // Add byte tokens a–z (IDs 0–25)
    for (i, c) in (b'a'..=b'z').enumerate() {
        state.vocab.insert(i as u32, (c as char).to_string());
    }

    // Converting to OxiTokenizer should not panic
    let tokenizer = state.to_oxi_tokenizer();
    assert_eq!(tokenizer.vocab_size(), 26);

    // We should be able to encode ASCII text using char-level fallback
    let result = tokenizer.encode("abc");
    // Even if it returns an error (unk), it must not panic
    let _ = result;
}

// ── Additional edge cases ─────────────────────────────────────────────────────

#[test]
fn base64_roundtrip_unicode() {
    let text = "日本語テスト🦀";
    let encoded = base64_encode(text.as_bytes());
    let decoded = base64_decode(&encoded).expect("decode should succeed");
    let restored = String::from_utf8(decoded).expect("must be valid UTF-8");
    assert_eq!(restored, text);
}

#[test]
fn tokenizer_state_default_is_empty() {
    let state = TokenizerState::default();
    assert_eq!(state.vocab_size(), 0);
}

#[test]
fn save_load_roundtrip_via_cursor() {
    let mut state = TokenizerState::new();
    state.vocab.insert(0, "a".to_string());
    state.vocab.insert(1, "b".to_string());
    state.merges.push((0, 1, 2));
    state.special_tokens.insert("<pad>".to_string(), 3);
    state.vocab.insert(3, "<pad>".to_string());

    let mut buf: Vec<u8> = Vec::new();
    state.save_to(&mut buf).expect("save");

    let cursor = Cursor::new(buf);
    let mut reader = BufReader::new(cursor);
    let loaded = TokenizerState::load_from(&mut reader).expect("load");

    assert_eq!(loaded.vocab.get(&0), Some(&"a".to_string()));
    assert_eq!(loaded.vocab.get(&1), Some(&"b".to_string()));
    assert_eq!(loaded.merges, vec![(0, 1, 2)]);
    assert_eq!(loaded.special_tokens.get("<pad>"), Some(&3));
}
