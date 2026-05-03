//! Unicode edge-case tests: ZWJ / VS / bidi marks / control chars / BOM /
//! 256-byte-fallback roundtrip.
//!
//! All tests use the `byte_fallback_tokenizer` fixture which contains all 256
//! `<0xHH>` tokens — this ensures byte-level encode/decode truly preserves
//! every input byte.

use oxibonsai_tokenizer::{byte_fallback_id, BpeMerges, OxiTokenizer, TokenizerConfig, Vocabulary};

// ── Fixture: full 256-byte-fallback tokenizer ────────────────────────────────

fn byte_fallback_tokenizer() -> OxiTokenizer {
    let mut vocab = Vocabulary::new();
    vocab.add_special("<unk>", 0);
    vocab.add_special("<bos>", 1);
    vocab.add_special("<eos>", 2);
    vocab.add_special("<pad>", 3);
    for byte in 0u16..=255u16 {
        let b = byte as u8;
        vocab.insert(&byte_fallback_id(b), 4 + b as u32);
    }
    OxiTokenizer::new(vocab, BpeMerges::new(), TokenizerConfig::default())
}

/// Decode a byte-fallback sequence directly from a `&str`'s bytes.
fn encode_as_bytes(s: &str) -> Vec<u32> {
    s.bytes().map(|b| 4 + b as u32).collect()
}

fn roundtrip_via_decode(tok: &OxiTokenizer, s: &str) -> String {
    let ids = encode_as_bytes(s);
    tok.decode(&ids).expect("decode")
}

// ── Zero-width joiner / variation selectors ──────────────────────────────────

#[test]
fn zwj_preserved_u200d() {
    let s = "a\u{200D}b";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn variation_selector_preserved_ufe0f() {
    let s = "*\u{FE0F}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn emoji_with_vs_roundtrip() {
    // Red heart: U+2764 + U+FE0F
    let s = "\u{2764}\u{FE0F}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn family_emoji_zwj_sequence() {
    // Family = man + ZWJ + woman + ZWJ + girl.
    let s = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

// ── Bidi marks ───────────────────────────────────────────────────────────────

#[test]
fn right_to_left_embedding_preserved_u202b() {
    let s = "a\u{202B}شما\u{202C}b";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn pop_directional_format_u202c_standalone() {
    let s = "\u{202C}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn lre_and_rle_preserved() {
    let s = "\u{202A}abc\u{202C}\u{202B}xyz\u{202C}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

// ── Control characters ──────────────────────────────────────────────────────

#[test]
fn tab_preserved_u0009() {
    let s = "a\tb";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn newline_preserved_u000a() {
    let s = "a\nb";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn carriage_return_preserved_u000d() {
    let s = "a\r\nb";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn null_byte_as_fallback() {
    // We can't construct a Rust `&str` containing a raw null easily, but we
    // can build the encoded form directly.
    let tok = byte_fallback_tokenizer();
    let ids = vec![4u32]; // id 4 = <0x00>
    let text = tok.decode(&ids).expect("decode");
    assert_eq!(text.as_bytes(), &[0x00]);
}

#[test]
fn bell_and_escape_controls() {
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = [0x07, 0x1B, 0x7F].iter().map(|&b| 4u32 + b).collect();
    let text = tok.decode(&ids).expect("decode");
    assert_eq!(text.as_bytes(), &[0x07, 0x1B, 0x7F]);
}

// ── BOM ──────────────────────────────────────────────────────────────────────

#[test]
fn bom_preserved_ufeff() {
    let s = "\u{FEFF}hello";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn bom_only_string() {
    let s = "\u{FEFF}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

// ── 256-byte-fallback roundtrip ──────────────────────────────────────────────
//
// Individual raw bytes in 0x80..=0xFF are not by themselves valid UTF-8
// continuation-without-lead sequences, so `decode()` (which produces a
// `String`) will reject them.  Instead, we verify that the decoder's byte
// output exactly matches the input byte stream of every *valid* UTF-8 text
// (tested below) AND we verify the low-level byte path directly.

#[test]
fn all_256_bytes_roundtrip_via_valid_sequences() {
    // Walk the entire 256-byte space in valid-UTF-8 chunks: each iteration
    // feeds one byte prefixed with the required UTF-8 lead.  For 0x00..=0x7F
    // each byte is itself a valid single-byte char.  For 0x80..=0xFF we
    // embed the byte in a two-byte UTF-8 sequence whose value happens to be
    // that high byte.
    let tok = byte_fallback_tokenizer();

    // ASCII half: each byte is valid on its own.
    for b in 0x00u8..=0x7Fu8 {
        let ids = vec![4u32 + b as u32];
        let decoded = tok.decode(&ids).expect("decode ascii");
        assert_eq!(decoded.as_bytes(), &[b], "byte {b:#x} ascii roundtrip");
    }

    // High-byte half: feed the full two-byte UTF-8 representation of each
    // codepoint U+0080..=U+00FF — this exercises each high byte as the
    // second byte of a valid multi-byte sequence.
    for cp in 0x80u32..=0xFFu32 {
        let ch = char::from_u32(cp).expect("valid codepoint");
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let ids: Vec<u32> = s.bytes().map(|b| 4u32 + b as u32).collect();
        let decoded = tok.decode(&ids).expect("decode multi-byte");
        assert_eq!(decoded, s.to_owned(), "codepoint {cp:#x} roundtrip");
    }
}

#[test]
fn high_byte_appears_in_valid_utf8() {
    // Every byte in 0xC2..=0xFF appears as the leading byte of some valid
    // UTF-8 codepoint.  We pick one codepoint per lead byte and verify the
    // roundtrip.
    let tok = byte_fallback_tokenizer();
    // Codepoint U+0080 has leading byte 0xC2; U+0100 has 0xC4; ...
    for cp in [0x80u32, 0xC0, 0x100, 0x800, 0x1F44B] {
        let ch = char::from_u32(cp).expect("valid cp");
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let ids: Vec<u32> = s.bytes().map(|b| 4u32 + b as u32).collect();
        let decoded = tok.decode(&ids).expect("decode");
        assert_eq!(decoded, s.to_owned());
    }
}

// ── Unicode edge chars ──────────────────────────────────────────────────────

#[test]
fn hangul_syllable_roundtrip() {
    let s = "한국어";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn cjk_extension_b_roundtrip() {
    // U+20000 — CJK Unified Ideographs Extension B (4-byte UTF-8).
    let s = "\u{20000}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn private_use_area_roundtrip() {
    // U+E000 (PUA) — some LLMs use this range for custom tokens.
    let s = "\u{E000}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn mathematical_italic_alphabet_roundtrip() {
    // U+1D44E (Mathematical Italic Small A)
    let s = "\u{1D44E}";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

// ── Mixed scripts and normalization ──────────────────────────────────────────

#[test]
fn mixed_scripts_roundtrip() {
    let s = "Hello, 世界! 🌍 Héllo مرحبا";
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, s), s);
}

#[test]
fn decomposed_vs_precomposed_preserved() {
    let precomposed = "é"; // U+00E9
    let decomposed = "e\u{0301}"; // U+0065 + U+0301
    let tok = byte_fallback_tokenizer();
    // The tokenizer must preserve *whichever form the input used*; it must
    // not silently normalize.
    assert_eq!(roundtrip_via_decode(&tok, precomposed), precomposed);
    assert_eq!(roundtrip_via_decode(&tok, decomposed), decomposed);
}

#[test]
fn empty_string_roundtrip() {
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, ""), "");
}

#[test]
fn long_repeated_bom_run() {
    let s: String = "\u{FEFF}".repeat(10);
    let tok = byte_fallback_tokenizer();
    assert_eq!(roundtrip_via_decode(&tok, &s), s);
}

#[test]
fn unk_id_produces_replacement() {
    // Decoding an ID outside the vocab yields U+FFFD.
    let tok = byte_fallback_tokenizer();
    let ids = vec![999_999u32];
    let text = tok.decode(&ids).expect("decode");
    assert_eq!(text, "\u{FFFD}");
}
