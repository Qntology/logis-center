//! Byte-Pair Encoding (BPE) merge table and encoding routines.
//!
//! This module implements:
//! - [`BpeMerges`] — a merge table mapping symbol pairs to merged-token IDs
//! - [`bpe_encode`] — greedy BPE encoding of a pre-tokenized word
//! - [`pretokenize`] — GPT-2–style whitespace/punctuation split
//! - [`byte_fallback_id`] — produce a `<0xHH>` token name for unknown bytes

use std::collections::HashMap;

use crate::vocab::Vocabulary;

// ── BpeMerges ────────────────────────────────────────────────────────────────

/// BPE merge table: a set of (A, B) → merged-ID rules ordered by priority.
///
/// Lower priority index = earlier merge (higher priority).
#[derive(Debug, Clone, Default)]
pub struct BpeMerges {
    /// Maps a symbol pair to the ID of the merged token.
    merges: HashMap<(String, String), u32>,
    /// Insertion-ordered list of merge pairs (defines priority).
    merge_order: Vec<(String, String)>,
}

impl BpeMerges {
    /// Create an empty merge table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a merge rule: `a` + `b` → token with ID `result_id`.
    ///
    /// Duplicate entries (same pair) are silently overwritten; the existing
    /// order-slot is preserved.
    pub fn add_merge(&mut self, a: &str, b: &str, result_id: u32) {
        let key = (a.to_owned(), b.to_owned());
        if !self.merges.contains_key(&key) {
            self.merge_order.push(key.clone());
        }
        self.merges.insert(key, result_id);
    }

    /// Return the 0-based priority index for a merge pair, if it exists.
    ///
    /// Lower index = higher priority (applied first during encoding).
    pub fn get_merge_priority(&self, a: &str, b: &str) -> Option<usize> {
        let key = (a.to_owned(), b.to_owned());
        if self.merges.contains_key(&key) {
            self.merge_order.iter().position(|p| *p == key)
        } else {
            None
        }
    }

    /// Return the merged token ID for a pair, if a rule exists.
    pub fn get_merge_result(&self, a: &str, b: &str) -> Option<u32> {
        self.merges.get(&(a.to_owned(), b.to_owned())).copied()
    }

    /// Number of merge rules in the table.
    pub fn len(&self) -> usize {
        self.merges.len()
    }

    /// Returns `true` if the merge table is empty.
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }
}

// ── Pre-tokenizer ─────────────────────────────────────────────────────────────

/// Split text into pre-tokens using a GPT-2–style rule:
/// words (optionally preceded by a space) are separated from punctuation and
/// standalone whitespace runs.
///
/// The returned strings are the raw Unicode chunks to be BPE-encoded
/// individually.
pub fn pretokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut last_was_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            last_was_space = true;
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            // Punctuation gets its own token; add the leading space prefix if
            // the previous character was whitespace (GPT-2 convention: Ġ prefix).
            let mut tok = String::new();
            if last_was_space {
                tok.push('\u{0120}'); // Ġ — GPT-2 space prefix
            }
            tok.push(ch);
            tokens.push(tok);
            last_was_space = false;
        } else {
            if last_was_space && !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if last_was_space && current.is_empty() {
                current.push('\u{0120}'); // Leading Ġ prefix
            }
            current.push(ch);
            last_was_space = false;
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

// ── BPE encoder ──────────────────────────────────────────────────────────────

/// Greedy BPE encode a single pre-tokenized word.
///
/// Algorithm:
/// 1. Split the word into individual Unicode characters as the initial symbol
///    sequence.
/// 2. Repeatedly find the pair with the lowest priority index in `merges` and
///    merge it.
/// 3. Continue until no more merges apply.
/// 4. Map each remaining symbol to its vocabulary ID; use byte-fallback for
///    any symbol not found in the vocabulary.
pub fn bpe_encode(word: &str, vocab: &Vocabulary, merges: &BpeMerges) -> Vec<u32> {
    if word.is_empty() {
        return Vec::new();
    }

    // Initialise the symbol sequence as individual characters.
    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    // Iteratively apply the highest-priority merge.
    loop {
        if symbols.len() < 2 {
            break;
        }

        // Find the pair with the best (lowest index) priority.
        let best = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(pos, pair)| {
                merges
                    .get_merge_priority(&pair[0], &pair[1])
                    .map(|priority| (priority, pos))
            })
            .min_by_key(|&(priority, _)| priority);

        match best {
            None => break, // No more applicable merges.
            Some((_, pos)) => {
                // Merge symbols[pos] and symbols[pos+1].
                let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
                symbols[pos] = merged;
                symbols.remove(pos + 1);
            }
        }
    }

    // Map symbols → token IDs.
    symbols
        .iter()
        .flat_map(|sym| symbol_to_ids(sym, vocab))
        .collect()
}

/// Convert a symbol to one or more token IDs.
///
/// If the symbol is directly in the vocabulary, return its single ID.
/// Otherwise attempt UTF-8 byte fallback: each byte is encoded as `<0xHH>`.
fn symbol_to_ids(sym: &str, vocab: &Vocabulary) -> Vec<u32> {
    if let Some(id) = vocab.get_id(sym) {
        return vec![id];
    }

    // Byte fallback.
    sym.as_bytes()
        .iter()
        .filter_map(|&b| {
            let fallback = byte_fallback_id(b);
            vocab.get_id(&fallback)
        })
        .collect()
}

// ── Byte fallback ─────────────────────────────────────────────────────────────

/// Return the byte-fallback token name for a single byte value.
///
/// Format: `<0xHH>` where `HH` is the uppercase hexadecimal byte value.
///
/// # Example
/// ```
/// use oxibonsai_tokenizer::bpe::byte_fallback_id;
/// assert_eq!(byte_fallback_id(0x20), "<0x20>");
/// assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
/// ```
pub fn byte_fallback_id(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocabulary;

    fn make_vocab_with_merges() -> (Vocabulary, BpeMerges) {
        let mut vocab = Vocabulary::new();
        // Individual characters
        vocab.insert("h", 10);
        vocab.insert("e", 11);
        vocab.insert("l", 12);
        vocab.insert("o", 13);
        // Merged tokens
        vocab.insert("he", 20);
        vocab.insert("hel", 21);
        vocab.insert("hell", 22);
        vocab.insert("hello", 23);
        vocab.insert("lo", 24);

        let mut merges = BpeMerges::new();
        merges.add_merge("h", "e", 20);
        merges.add_merge("he", "l", 21);
        merges.add_merge("hel", "l", 22);
        merges.add_merge("hell", "o", 23);
        merges.add_merge("l", "o", 24);

        (vocab, merges)
    }

    #[test]
    fn byte_fallback_format() {
        assert_eq!(byte_fallback_id(0x00), "<0x00>");
        assert_eq!(byte_fallback_id(0x20), "<0x20>");
        assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
        assert_eq!(byte_fallback_id(0x0A), "<0x0A>");
    }

    #[test]
    fn bpe_merges_priority() {
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 1);
        m.add_merge("b", "c", 2);
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_priority("b", "c"), Some(1));
        assert_eq!(m.get_merge_priority("x", "y"), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn bpe_encode_hello() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("hello", &vocab, &merges);
        // Should merge all the way to "hello" → id 23
        assert_eq!(ids, vec![23]);
    }

    #[test]
    fn bpe_encode_empty() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("", &vocab, &merges);
        assert!(ids.is_empty());
    }

    #[test]
    fn pretokenize_simple_sentence() {
        let tokens = pretokenize("hello world");
        assert!(!tokens.is_empty());
        // Should split into at least "hello" and "Ġworld"
        assert!(tokens.iter().any(|t| t.contains("hello") || t == "hello"));
    }

    #[test]
    fn pretokenize_empty() {
        assert!(pretokenize("").is_empty());
    }

    #[test]
    fn pretokenize_punctuation_splits() {
        let tokens = pretokenize("hi,there");
        // Should split around the comma
        assert!(tokens.len() >= 2);
    }
}
