//! BPE tokenizer trainer: learn merge rules from a text corpus.
//!
//! Algorithm (Sennrich et al. 2016):
//! 1. Initialize vocabulary with byte-level characters (0–255).
//! 2. Encode corpus as sequences of byte token IDs.
//! 3. Repeat for `num_merges` iterations:
//!    a. Count all adjacent symbol-pair frequencies.
//!    b. Find the most frequent pair.
//!    c. Merge that pair everywhere in the corpus.
//!    d. Add the merged token to vocabulary.
//! 4. Return trained vocabulary + merge rules.

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    bpe::BpeMerges,
    tokenizer::{OxiTokenizer, TokenizerConfig},
    vocab::Vocabulary,
};

// ── TrainerConfig ─────────────────────────────────────────────────────────────

/// Configuration for the BPE trainer.
///
/// Marked `#[non_exhaustive]` so that new training knobs can be added in
/// future minor releases without a breaking change.  Downstream callers must
/// construct it via [`TrainerConfig::new`] or [`TrainerConfig::default`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrainerConfig {
    /// Target vocabulary size (base 256 byte tokens + num_merges merged tokens).
    pub vocab_size: usize,
    /// Minimum pair frequency required to perform a merge.
    pub min_frequency: usize,
    /// Whether to add special tokens (BOS=0, EOS=1, PAD=2, UNK=3) at IDs 0–3.
    /// When `true`, byte tokens start at ID 4 instead of ID 0.
    pub add_special_tokens: bool,
    /// When `true`, pre-tokenize on whitespace boundaries (GPT-2 style) before BPE.
    pub byte_level: bool,
    /// If `Some(n)`, log progress every `n` merges.
    pub progress_interval: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            min_frequency: 2,
            add_special_tokens: true,
            byte_level: true,
            progress_interval: None,
        }
    }
}

impl TrainerConfig {
    /// Create a config targeting `vocab_size` tokens with all other fields at
    /// their defaults.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            ..Default::default()
        }
    }

    /// Override the minimum pair frequency threshold.
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Enable or disable automatic special-token insertion.
    pub fn with_special_tokens(mut self, add: bool) -> Self {
        self.add_special_tokens = add;
        self
    }
}

// ── SymbolPair ────────────────────────────────────────────────────────────────

/// A pair of adjacent symbol IDs (left, right).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolPair(pub u32, pub u32);

impl SymbolPair {
    /// Construct a pair from two token IDs.
    pub fn new(a: u32, b: u32) -> Self {
        Self(a, b)
    }

    /// Produce the [`MergeRule`] that results from merging this pair into `new_id`.
    pub fn merged_symbol(&self, new_id: u32, merged_text: String) -> MergeRule {
        MergeRule {
            left: self.0,
            right: self.1,
            merged: new_id,
            merged_text,
        }
    }
}

// ── MergeRule ─────────────────────────────────────────────────────────────────

/// A single BPE merge rule: (left, right) → merged token.
#[derive(Debug, Clone)]
pub struct MergeRule {
    /// ID of the left symbol in the pair.
    pub left: u32,
    /// ID of the right symbol in the pair.
    pub right: u32,
    /// ID assigned to the merged token.
    pub merged: u32,
    /// String representation of the merged token.
    pub merged_text: String,
}

// ── Word ──────────────────────────────────────────────────────────────────────

/// A word (pre-token) in the training corpus represented as an ordered sequence
/// of symbol IDs together with its frequency.
#[derive(Debug, Clone)]
struct Word {
    /// Current symbol sequence (may shrink as merges are applied).
    symbols: Vec<u32>,
    /// Number of times this word appears in the corpus.
    freq: usize,
}

impl Word {
    fn new(symbols: Vec<u32>, freq: usize) -> Self {
        Self { symbols, freq }
    }
}

// ── TrainingStats ─────────────────────────────────────────────────────────────

/// Statistics gathered during a training run.
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Vocabulary size before any merges (256 byte tokens + optional specials).
    pub initial_vocab_size: usize,
    /// Vocabulary size at the end of training.
    pub final_vocab_size: usize,
    /// Number of merge operations successfully applied.
    pub num_merges_performed: usize,
    /// Number of candidate pairs rejected because they fell below `min_frequency`.
    pub num_merges_skipped: usize,
    /// Total character count across the entire corpus (sum of `str::len()`).
    pub corpus_size_chars: usize,
    /// Number of distinct pre-tokenized word types.
    pub unique_words: usize,
}

impl TrainingStats {
    /// Human-readable one-line summary of the training run.
    pub fn summary(&self) -> String {
        format!(
            "BPE training: {init} → {fin} tokens | \
             {merges} merges applied, {skipped} skipped | \
             corpus {chars} bytes, {words} unique words",
            init = self.initial_vocab_size,
            fin = self.final_vocab_size,
            merges = self.num_merges_performed,
            skipped = self.num_merges_skipped,
            chars = self.corpus_size_chars,
            words = self.unique_words,
        )
    }
}

// ── TrainedTokenizer ──────────────────────────────────────────────────────────

/// The result returned by [`BpeTrainer::train`].
#[derive(Debug)]
pub struct TrainedTokenizer {
    /// Full ID → token-string mapping (byte tokens + merged tokens + specials).
    pub vocab: HashMap<u32, String>,
    /// Merge rules in the order they were learned (first learned = highest priority).
    pub merges: Vec<MergeRule>,
    /// Diagnostic information about the training run.
    pub stats: TrainingStats,
}

impl TrainedTokenizer {
    /// Convert this trained result into a ready-to-use [`OxiTokenizer`].
    ///
    /// The [`TokenizerConfig`] is set to defaults; callers may rebuild from the
    /// raw `vocab` / `merges` fields if a custom config is needed.
    pub fn to_oxi_tokenizer(&self) -> OxiTokenizer {
        let mut vocabulary = Vocabulary::new();
        // Determine whether special-token slots are present by checking IDs 0-3.
        // Special tokens are identified by their angle-bracket names.
        for (&id, token) in &self.vocab {
            if token.starts_with('<') && token.ends_with('>') {
                vocabulary.add_special(token, id);
            } else {
                vocabulary.insert(token, id);
            }
        }

        let mut bpe_merges = BpeMerges::new();
        for rule in &self.merges {
            // Reconstruct the left and right token strings from the vocab map.
            let left_str = self.vocab.get(&rule.left).map(|s| s.as_str()).unwrap_or("");
            let right_str = self
                .vocab
                .get(&rule.right)
                .map(|s| s.as_str())
                .unwrap_or("");
            bpe_merges.add_merge(left_str, right_str, rule.merged);
        }

        let config = TokenizerConfig::default();
        OxiTokenizer::new(vocabulary, bpe_merges, config)
    }

    /// Serialize merge rules as plain text (one rule per line).
    ///
    /// Format: `<left_token> <right_token>`
    /// (matching the HuggingFace `merges.txt` convention).
    pub fn merges_to_text(&self) -> String {
        let mut out = String::new();
        for rule in &self.merges {
            let left = self.vocab.get(&rule.left).map(|s| s.as_str()).unwrap_or("");
            let right = self
                .vocab
                .get(&rule.right)
                .map(|s| s.as_str())
                .unwrap_or("");
            out.push_str(left);
            out.push(' ');
            out.push_str(right);
            out.push('\n');
        }
        out
    }

    /// Total number of tokens in the trained vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── TrainerError ──────────────────────────────────────────────────────────────

/// Errors that can occur during BPE training.
#[derive(Debug, Error)]
pub enum TrainerError {
    /// The corpus slice was empty.
    #[error("empty corpus")]
    EmptyCorpus,
    /// Requested `vocab_size` is too small to hold even the base byte vocabulary.
    #[error("vocab_size {0} must be > 256 (base byte vocabulary)")]
    VocabSizeTooSmall(usize),
    /// Pre-tokenization produced no usable words.
    #[error("corpus has no valid words after pre-tokenization")]
    NoValidWords,
}

// ── BpeTrainer ────────────────────────────────────────────────────────────────

/// BPE trainer that learns merge rules from a raw text corpus.
///
/// # Example
///
/// ```rust
/// use oxibonsai_tokenizer::trainer::{BpeTrainer, TrainerConfig};
///
/// let mut trainer = BpeTrainer::new(TrainerConfig::new(512));
/// let corpus = ["the quick brown fox", "the fox jumped"];
/// let trained = trainer.train(&corpus).expect("training should succeed");
/// println!("{}", trained.stats.summary());
/// ```
pub struct BpeTrainer {
    config: TrainerConfig,
    /// Byte value → initial token ID (256 entries when `add_special_tokens` is
    /// false; otherwise IDs are offset by 4 to leave room for specials).
    char_vocab: HashMap<u8, u32>,
    /// The next token ID to assign to a newly merged token.
    next_id: u32,
}

impl BpeTrainer {
    /// Create a new trainer with the supplied configuration.
    pub fn new(config: TrainerConfig) -> Self {
        let char_vocab = HashMap::new(); // populated lazily in `train`
        let next_id = 0;
        Self {
            config,
            char_vocab,
            next_id,
        }
    }

    /// Convenience constructor with default configuration.
    pub fn default_config() -> Self {
        Self::new(TrainerConfig::default())
    }

    // ── Public entry point ────────────────────────────────────────────────

    /// Train a BPE tokenizer on the supplied corpus.
    ///
    /// Each element of `corpus` is treated as an independent document.
    /// The function is deterministic: given the same corpus and config it always
    /// produces the same output.
    pub fn train(&mut self, corpus: &[&str]) -> Result<TrainedTokenizer, TrainerError> {
        // ── Validate inputs ───────────────────────────────────────────────
        if corpus.is_empty() {
            return Err(TrainerError::EmptyCorpus);
        }

        // We always need room for at least 256 byte tokens.
        let min_size: usize = if self.config.add_special_tokens {
            256 + 4
        } else {
            256
        };
        if self.config.vocab_size <= min_size.saturating_sub(1) {
            return Err(TrainerError::VocabSizeTooSmall(self.config.vocab_size));
        }

        // ── Build initial byte vocabulary ─────────────────────────────────
        let mut id_to_token: HashMap<u32, String> = HashMap::new();

        // Reserve IDs 0-3 for special tokens when requested.
        let byte_id_offset: u32 = if self.config.add_special_tokens { 4 } else { 0 };

        if self.config.add_special_tokens {
            id_to_token.insert(0, "<unk>".to_owned());
            id_to_token.insert(1, "<bos>".to_owned());
            id_to_token.insert(2, "<eos>".to_owned());
            id_to_token.insert(3, "<pad>".to_owned());
        }

        self.char_vocab.clear();
        for byte in 0u8..=255u8 {
            let id = byte as u32 + byte_id_offset;
            // Token string for a byte is the raw UTF-8 character if it is
            // printable ASCII; otherwise use the `<0xHH>` byte-fallback form.
            let token = byte_token_string(byte);
            self.char_vocab.insert(byte, id);
            id_to_token.insert(id, token);
        }

        self.next_id = 256 + byte_id_offset;

        let initial_vocab_size = id_to_token.len();

        // ── Pre-tokenize corpus ───────────────────────────────────────────
        let corpus_size_chars: usize = corpus.iter().map(|s| s.len()).sum();
        let word_freqs = self.pretokenize(corpus);

        if word_freqs.is_empty() {
            return Err(TrainerError::NoValidWords);
        }

        let unique_words = word_freqs.len();

        // Convert word-frequency map to a Vec<Word> of symbol sequences.
        let mut words: Vec<Word> = word_freqs
            .into_iter()
            .map(|(text, freq)| {
                let symbols = self.encode_word(&text);
                Word::new(symbols, freq)
            })
            .collect();

        // ── BPE training loop ─────────────────────────────────────────────
        let num_merges = self.config.vocab_size.saturating_sub(self.next_id as usize);
        let mut merge_rules: Vec<MergeRule> = Vec::with_capacity(num_merges);
        let mut num_merges_skipped: usize = 0;

        for merge_idx in 0..num_merges {
            // Log progress if requested.
            if let Some(interval) = self.config.progress_interval {
                if interval > 0 && merge_idx % interval == 0 {
                    tracing::debug!(
                        merge = merge_idx,
                        total = num_merges,
                        vocab = self.next_id,
                        "BPE training progress",
                    );
                }
            }

            // Count pair frequencies.
            let pair_counts = self.count_pairs(&words);
            if pair_counts.is_empty() {
                // No more pairs — corpus has been fully merged.
                break;
            }

            // Select the best pair.
            let best = match self.best_pair(&pair_counts) {
                Some(b) => b,
                None => {
                    // All remaining pairs are below min_frequency.
                    num_merges_skipped += num_merges - merge_idx;
                    break;
                }
            };

            let (pair, _freq) = best;

            // Build the merged token string.
            let left_str = id_to_token.get(&pair.0).cloned().unwrap_or_default();
            let right_str = id_to_token.get(&pair.1).cloned().unwrap_or_default();
            let merged_text = format!("{left_str}{right_str}");

            // Assign a new ID to the merged token.
            let new_id = self.next_id;
            self.next_id += 1;
            id_to_token.insert(new_id, merged_text.clone());

            // Record the merge rule.
            let rule = pair.merged_symbol(new_id, merged_text);
            merge_rules.push(rule);

            // Apply the merge throughout the corpus.
            self.apply_merge(&mut words, &pair, new_id);
        }

        let final_vocab_size = id_to_token.len();
        let num_merges_performed = merge_rules.len();

        let stats = TrainingStats {
            initial_vocab_size,
            final_vocab_size,
            num_merges_performed,
            num_merges_skipped,
            corpus_size_chars,
            unique_words,
        };

        Ok(TrainedTokenizer {
            vocab: id_to_token,
            merges: merge_rules,
            stats,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Count the frequency of every adjacent symbol pair across all words.
    ///
    /// Each pair's count is weighted by the frequency of the word it appears in.
    fn count_pairs(&self, words: &[Word]) -> HashMap<SymbolPair, usize> {
        let mut counts: HashMap<SymbolPair, usize> = HashMap::new();
        for word in words {
            if word.symbols.len() < 2 {
                continue;
            }
            for window in word.symbols.windows(2) {
                let pair = SymbolPair::new(window[0], window[1]);
                *counts.entry(pair).or_insert(0) += word.freq;
            }
        }
        counts
    }

    /// Find the most frequent pair whose count meets or exceeds `min_frequency`.
    ///
    /// Ties are broken deterministically by preferring the pair with the smallest
    /// (left, right) ID values so that training is fully reproducible.
    fn best_pair(&self, pair_counts: &HashMap<SymbolPair, usize>) -> Option<(SymbolPair, usize)> {
        pair_counts
            .iter()
            .filter(|(_, &count)| count >= self.config.min_frequency)
            .max_by(|(pair_a, &cnt_a), (pair_b, &cnt_b)| {
                // Primary: higher frequency wins.
                // Secondary (tiebreak): lower IDs win (deterministic).
                cnt_a
                    .cmp(&cnt_b)
                    .then_with(|| pair_b.0.cmp(&pair_a.0))
                    .then_with(|| pair_b.1.cmp(&pair_a.1))
            })
            .map(|(pair, &count)| (pair.clone(), count))
    }

    /// Apply a merge rule to every occurrence of `pair` in all words in-place.
    ///
    /// When a match is found at position `i`, `symbols[i]` is replaced with
    /// `new_id` and `symbols[i+1]` is removed.  The scan continues from
    /// position `i` (not `i+1`) to handle non-overlapping matches correctly.
    fn apply_merge(&self, words: &mut [Word], pair: &SymbolPair, new_id: u32) {
        for word in words.iter_mut() {
            if word.symbols.len() < 2 {
                continue;
            }
            let mut i = 0;
            while i + 1 < word.symbols.len() {
                if word.symbols[i] == pair.0 && word.symbols[i + 1] == pair.1 {
                    word.symbols[i] = new_id;
                    word.symbols.remove(i + 1);
                    // Do NOT advance `i`: the newly merged token at position `i`
                    // may form another valid pair with the next symbol.
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Pre-tokenize the corpus into a map from word-string → frequency.
    ///
    /// When `byte_level` is set, text is split on whitespace boundaries so that
    /// BPE operates on words rather than the full document.  Otherwise the
    /// entire document is treated as one unit.
    fn pretokenize(&self, corpus: &[&str]) -> HashMap<String, usize> {
        let mut freq_map: HashMap<String, usize> = HashMap::new();
        for &doc in corpus {
            if self.config.byte_level {
                // Split on whitespace; keep non-empty parts only.
                for word in doc.split_whitespace() {
                    if !word.is_empty() {
                        *freq_map.entry(word.to_owned()).or_insert(0) += 1;
                    }
                }
            } else {
                // Treat the entire document as a single unit.
                if !doc.is_empty() {
                    *freq_map.entry(doc.to_owned()).or_insert(0) += 1;
                }
            }
        }
        freq_map
    }

    /// Encode a word string into its initial byte-level token ID sequence.
    ///
    /// Each byte of the UTF-8 representation becomes one symbol ID.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        word.as_bytes()
            .iter()
            .filter_map(|b| self.char_vocab.get(b).copied())
            .collect()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the canonical string representation for a byte token.
///
/// - Printable ASCII (0x20–0x7E): the character itself.
/// - Everything else: `<0xHH>` byte-fallback form.
fn byte_token_string(byte: u8) -> String {
    if byte.is_ascii() && !byte.is_ascii_control() {
        // Printable ASCII.
        (byte as char).to_string()
    } else {
        format!("<0x{byte:02X}>")
    }
}

// ── Tests (inline sanity checks) ──────────────────────────────────────────────

#[cfg(test)]
mod inline_tests {
    use super::*;

    #[test]
    fn byte_token_string_printable() {
        assert_eq!(byte_token_string(b'a'), "a");
        assert_eq!(byte_token_string(b' '), " ");
        assert_eq!(byte_token_string(b'~'), "~");
    }

    #[test]
    fn byte_token_string_control() {
        assert_eq!(byte_token_string(0x00), "<0x00>");
        assert_eq!(byte_token_string(0x0A), "<0x0A>");
        assert_eq!(byte_token_string(0xFF), "<0xFF>");
    }

    #[test]
    fn count_pairs_basic() {
        let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
        trainer.char_vocab.insert(b'a', 0);
        trainer.char_vocab.insert(b'b', 1);
        let words = vec![Word::new(vec![0, 1, 0, 1], 3)];
        let counts = trainer.count_pairs(&words);
        assert_eq!(counts.get(&SymbolPair::new(0, 1)), Some(&6)); // appears twice × freq 3
    }

    #[test]
    fn apply_merge_replaces_pair() {
        let trainer = BpeTrainer::new(TrainerConfig::new(300));
        let mut words = vec![Word::new(vec![0, 1, 0, 1], 1)];
        trainer.apply_merge(&mut words, &SymbolPair::new(0, 1), 99);
        assert_eq!(words[0].symbols, vec![99, 99]);
    }
}
