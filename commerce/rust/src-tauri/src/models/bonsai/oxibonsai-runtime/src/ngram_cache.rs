//! N-gram cache for zero-cost speculative decoding draft generation.
//!
//! Maintains a frequency-based cache of token patterns observed during
//! generation. When a trigram pattern (a, b) → c has been seen before,
//! it can predict c as the likely next token after seeing (a, b).

use std::collections::HashMap;

/// Token-level n-gram cache for speculative draft generation.
///
/// Records bigram and trigram patterns from generated text and
/// predicts likely next tokens based on observed frequencies.
pub struct NgramCache {
    /// Bigram: single token → (next_token, count) sorted by count desc
    bigrams: HashMap<u32, Vec<(u32, u32)>>,
    /// Trigram: (token_a, token_b) → (next_token, count) sorted by count desc
    trigrams: HashMap<(u32, u32), Vec<(u32, u32)>>,
    /// Maximum entries per n-gram key (prevents unbounded growth)
    max_entries_per_key: usize,
}

impl NgramCache {
    /// Create a new empty n-gram cache.
    pub fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            max_entries_per_key: 8,
        }
    }

    /// Record a sequence of tokens into the cache.
    ///
    /// Updates both bigram and trigram frequency tables.
    pub fn record(&mut self, tokens: &[u32]) {
        // Record bigrams
        for window in tokens.windows(2) {
            self.record_bigram(window[0], window[1]);
        }
        // Record trigrams
        for window in tokens.windows(3) {
            self.record_trigram(window[0], window[1], window[2]);
        }
    }

    /// Record a single bigram observation.
    fn record_bigram(&mut self, a: u32, next: u32) {
        let entries = self.bigrams.entry(a).or_default();
        if let Some(entry) = entries.iter_mut().find(|(tok, _)| *tok == next) {
            entry.1 += 1;
        } else if entries.len() < self.max_entries_per_key {
            entries.push((next, 1));
        }
        // Keep sorted by count descending for fast top-1 lookup
        entries.sort_unstable_by_key(|e| std::cmp::Reverse(e.1));
    }

    /// Record a single trigram observation.
    fn record_trigram(&mut self, a: u32, b: u32, next: u32) {
        let entries = self.trigrams.entry((a, b)).or_default();
        if let Some(entry) = entries.iter_mut().find(|(tok, _)| *tok == next) {
            entry.1 += 1;
        } else if entries.len() < self.max_entries_per_key {
            entries.push((next, 1));
        }
        entries.sort_unstable_by_key(|e| std::cmp::Reverse(e.1));
    }

    /// Predict the most likely next token given the context.
    ///
    /// Tries trigram first (higher accuracy), falls back to bigram.
    /// Returns `None` if no matching pattern is found.
    pub fn predict_one(&self, context: &[u32]) -> Option<u32> {
        // Try trigram: use last 2 tokens
        if context.len() >= 2 {
            let a = context[context.len() - 2];
            let b = context[context.len() - 1];
            if let Some(entries) = self.trigrams.get(&(a, b)) {
                if let Some(&(next, _count)) = entries.first() {
                    return Some(next);
                }
            }
        }

        // Fallback: bigram using last token
        if let Some(&last) = context.last() {
            if let Some(entries) = self.bigrams.get(&last) {
                if let Some(&(next, _count)) = entries.first() {
                    return Some(next);
                }
            }
        }

        None
    }

    /// Predict up to `lookahead` tokens by chaining predictions.
    ///
    /// Each predicted token is appended to the context for the next prediction.
    /// Stops early if no prediction is available.
    pub fn draft(&self, context: &[u32], lookahead: usize) -> Vec<u32> {
        let mut draft = Vec::with_capacity(lookahead);
        let mut ctx: Vec<u32> = context.to_vec();

        for _ in 0..lookahead {
            match self.predict_one(&ctx) {
                Some(token) => {
                    draft.push(token);
                    ctx.push(token);
                }
                None => break,
            }
        }

        draft
    }

    /// Number of unique trigram keys stored.
    pub fn trigram_count(&self) -> usize {
        self.trigrams.len()
    }

    /// Number of unique bigram keys stored.
    pub fn bigram_count(&self) -> usize {
        self.bigrams.len()
    }

    /// Returns true if the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.bigrams.is_empty() && self.trigrams.is_empty()
    }
}

impl Default for NgramCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_returns_no_prediction() {
        let cache = NgramCache::new();
        assert_eq!(cache.predict_one(&[1, 2, 3]), None);
        assert!(cache.is_empty());
    }

    #[test]
    fn bigram_prediction() {
        let mut cache = NgramCache::new();
        cache.record(&[10, 20, 30]);
        // Bigram: 10→20, 20→30
        assert_eq!(cache.predict_one(&[10]), Some(20));
        assert_eq!(cache.predict_one(&[20]), Some(30));
    }

    #[test]
    fn trigram_preferred_over_bigram() {
        let mut cache = NgramCache::new();
        cache.record(&[10, 20, 30]);
        cache.record(&[10, 20, 40]); // second trigram (10,20)→40
        cache.record(&[10, 20, 40]); // now (10,20)→40 has count=2 > (10,20)→30 count=1
                                     // Trigram (10,20) predicts 40 (higher count)
        assert_eq!(cache.predict_one(&[10, 20]), Some(40));
    }

    #[test]
    fn draft_chains_predictions() {
        let mut cache = NgramCache::new();
        // Record a repeating pattern: 1, 2, 3, 1, 2, 3, 1, 2, 3
        cache.record(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);

        let draft = cache.draft(&[1, 2], 4);
        // Should predict: 3, 1, 2, 3 (repeating pattern)
        assert_eq!(draft, vec![3, 1, 2, 3]);
    }

    #[test]
    fn draft_stops_on_no_prediction() {
        let mut cache = NgramCache::new();
        cache.record(&[1, 2, 3]);

        // Context [99] has no match
        let draft = cache.draft(&[99], 4);
        assert!(draft.is_empty());
    }

    #[test]
    fn frequency_tracking() {
        let mut cache = NgramCache::new();
        cache.record(&[1, 2, 3]);
        cache.record(&[1, 2, 3]);
        cache.record(&[1, 2, 3]);
        cache.record(&[1, 2, 5]);

        // (1,2)→3 has count 3, (1,2)→5 has count 1
        assert_eq!(cache.predict_one(&[1, 2]), Some(3));
    }
}
