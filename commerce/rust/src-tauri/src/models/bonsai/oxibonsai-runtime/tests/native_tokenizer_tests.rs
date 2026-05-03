//! Integration tests for [`NativeTokenizerBridge`].
//!
//! These tests exercise the bridge through the public `oxibonsai_runtime` API
//! and verify encode/decode roundtrips, error paths, and chat formatting.

use oxibonsai_runtime::native_tokenizer::{NativeTokenizerBridge, NativeTokenizerError};

// ── 1. encode → decode roundtrip for "hello" ─────────────────────────────────

#[test]
fn char_level_fallback_encode_decode() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    let ids = bridge.encode("hello").expect("encode should succeed");
    let text = bridge.decode(&ids).expect("decode should succeed");
    assert_eq!(text, "hello");
}

// ── 2. encode returns non-empty ids ──────────────────────────────────────────

#[test]
fn char_level_fallback_nonempty() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    let ids = bridge.encode("hello").expect("encode should succeed");
    assert!(
        !ids.is_empty(),
        "encoding a non-empty string must yield tokens"
    );
}

// ── 3. roundtrip for a longer string ─────────────────────────────────────────
//
// Note: the char-level stub uses GPT-2 pretokenization.  Spaces between words
// become a Ġ (U+0120) prefix on the next token rather than a literal space
// token.  The decoder strips Ġ and re-inserts a space before each non-first
// word.  A single unbroken word (no spaces) round-trips perfectly.

#[test]
fn char_level_fallback_roundtrip_long() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    // Use a single word so the GPT-2 space-prefix handling does not apply.
    let original = "thequickbrownfox";
    let ids = bridge.encode(original).expect("encode should succeed");
    assert!(
        ids.len() >= original.len(),
        "must encode at least one token per char"
    );
    let decoded = bridge.decode(&ids).expect("decode should succeed");
    assert_eq!(decoded, original);
}

// ── 4. vocab_size() > 0 ───────────────────────────────────────────────────────

#[test]
fn native_tokenizer_vocab_size_positive() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    assert!(
        bridge.vocab_size() > 0,
        "vocab_size must be positive, got {}",
        bridge.vocab_size()
    );
}

// ── 5. same input yields same output ─────────────────────────────────────────

#[test]
fn char_level_encode_consistent() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    let ids1 = bridge.encode("consistent").expect("first encode");
    let ids2 = bridge.encode("consistent").expect("second encode");
    assert_eq!(
        ids1, ids2,
        "encoding the same text twice must yield equal IDs"
    );
}

// ── 6. handles spaces, newlines, and unicode ──────────────────────────────────

#[test]
fn char_level_special_chars() {
    let bridge = NativeTokenizerBridge::char_level_fallback();

    // ASCII with space and newline
    let ids_ascii = bridge
        .encode("hello world\nhow are you")
        .expect("encode ascii with space+newline");
    assert!(!ids_ascii.is_empty());

    // Basic Unicode (café — includes multi-byte é)
    let ids_unicode = bridge.encode("café").expect("encode unicode");
    assert!(!ids_unicode.is_empty());
}

// ── 7. decoding empty slice returns empty string ──────────────────────────────

#[test]
fn native_tokenizer_decode_empty() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    let text = bridge
        .decode(&[])
        .expect("decoding empty slice should succeed");
    assert_eq!(text, "", "decoding [] must return an empty string");
}

// ── 8. format_chat without template returns NoChatTemplate ───────────────────

#[test]
fn native_tokenizer_format_chat_no_template() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    let result = bridge.format_chat(&[("user", "Hello!")]);
    match result {
        Err(NativeTokenizerError::NoChatTemplate) => { /* expected */ }
        other => panic!("expected NoChatTemplate, got {:?}", other),
    }
}

// ── 9. chatml template formats messages correctly ────────────────────────────

#[test]
fn native_tokenizer_with_chatml_format() {
    let bridge = NativeTokenizerBridge::char_level_fallback_with_chatml();

    // Single user message
    let prompt = bridge
        .format_chat(&[("user", "Hello!")])
        .expect("format_chat should succeed");
    assert!(
        prompt.contains("<|im_start|>user"),
        "prompt must contain im_start+role; got: {prompt:?}"
    );
    assert!(
        prompt.contains("Hello!"),
        "prompt must contain the message content; got: {prompt:?}"
    );
    assert!(
        prompt.contains("<|im_end|>"),
        "prompt must contain im_end; got: {prompt:?}"
    );

    // Multi-turn: system + user
    let multi = bridge
        .format_chat(&[
            ("system", "You are a helpful assistant."),
            ("user", "What is 2+2?"),
        ])
        .expect("multi-turn format_chat should succeed");
    assert!(multi.contains("<|im_start|>system"));
    assert!(multi.contains("You are a helpful assistant."));
    assert!(multi.contains("<|im_start|>user"));
    assert!(multi.contains("What is 2+2?"));
}

// ── 10. encoding empty string returns empty or BOS ───────────────────────────

#[test]
fn native_tokenizer_encode_empty() {
    let bridge = NativeTokenizerBridge::char_level_fallback();
    // Encoding an empty string must not fail.
    let ids = bridge
        .encode("")
        .expect("encoding empty string should succeed");
    // The default config has add_bos = false, so we expect an empty result.
    // If the config is changed in the future to prepend BOS, 0 or 1 tokens
    // are both acceptable.
    assert!(
        ids.len() <= 1,
        "encoding empty string should yield 0 or 1 tokens (BOS), got {}",
        ids.len()
    );
}
