//! Sliding window attention for efficient long-context inference.
//!
//! Only attends to the most recent `window_size` tokens, reducing
//! compute from O(seq_len) to O(window_size) per layer. Implements
//! "attention sink" tokens (the first N tokens are always retained)
//! to maintain generation quality even when the window slides past
//! the beginning of the context.

/// Configuration for sliding window attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindowConfig {
    /// Maximum number of tokens to attend to within the window.
    pub window_size: usize,
    /// Number of "attention sink" tokens at the beginning of the sequence
    /// that are always retained in the attention window, regardless of
    /// the current position. Typically 4.
    pub sink_tokens: usize,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            sink_tokens: 4,
        }
    }
}

impl SlidingWindowConfig {
    /// Create a new sliding window config.
    pub fn new(window_size: usize, sink_tokens: usize) -> Self {
        Self {
            window_size,
            sink_tokens,
        }
    }

    /// Whether sliding window attention is effectively disabled
    /// (window larger than any practical sequence).
    pub fn is_disabled(&self) -> bool {
        self.window_size == 0
    }
}

/// Compute the effective attention range for a given position.
///
/// Returns a list of positions to attend to and the total count.
/// The range includes:
/// 1. Sink tokens (always positions 0..sink_tokens if they exist)
/// 2. Recent tokens within the window (max of `window_size - sink_tokens` recent positions)
///
/// Positions are returned in ascending order and deduplicated.
pub fn attention_range(
    pos: usize,
    seq_len: usize,
    config: &SlidingWindowConfig,
) -> (Vec<usize>, usize) {
    if config.window_size == 0 || seq_len == 0 {
        return (Vec::new(), 0);
    }

    let effective_seq_len = seq_len.min(pos + 1);
    let sink_count = config.sink_tokens.min(effective_seq_len);

    // If the window covers the entire sequence, return all positions
    if config.window_size >= effective_seq_len {
        let positions: Vec<usize> = (0..effective_seq_len).collect();
        let count = positions.len();
        return (positions, count);
    }

    // Budget for recent tokens (after reserving sink slots)
    let recent_budget = config.window_size.saturating_sub(sink_count);

    // Recent window: attend to [pos - recent_budget + 1 .. pos] inclusive
    // But start no earlier than sink_count (to avoid overlap with sink tokens)
    let recent_start = if pos + 1 > recent_budget {
        (pos + 1 - recent_budget).max(sink_count)
    } else {
        sink_count
    };
    let recent_end = pos + 1; // exclusive

    let mut positions: Vec<usize> = Vec::with_capacity(config.window_size);

    // Add sink tokens
    for i in 0..sink_count {
        positions.push(i);
    }

    // Add recent tokens (non-overlapping with sinks)
    for i in recent_start..recent_end {
        positions.push(i);
    }

    let count = positions.len();
    (positions, count)
}

/// Apply sliding window mask to attention scores.
///
/// Sets scores to `f32::NEG_INFINITY` for positions outside the
/// attention window, effectively zeroing them after softmax.
///
/// - `scores`: Mutable slice of attention scores, one per key position.
/// - `query_pos`: The position of the query token.
/// - `key_positions`: The positions corresponding to each score entry.
/// - `config`: Sliding window configuration.
pub fn apply_sliding_window_mask(
    scores: &mut [f32],
    query_pos: usize,
    key_positions: &[usize],
    config: &SlidingWindowConfig,
) {
    debug_assert_eq!(scores.len(), key_positions.len());

    if config.window_size == 0 {
        for score in scores.iter_mut() {
            *score = f32::NEG_INFINITY;
        }
        return;
    }

    let (valid_positions, _) = attention_range(
        query_pos,
        query_pos + 1, // seq_len is at least query_pos + 1
        config,
    );

    for (score, &key_pos) in scores.iter_mut().zip(key_positions.iter()) {
        if !valid_positions.contains(&key_pos) {
            *score = f32::NEG_INFINITY;
        }
    }
}

/// Evict KV cache entries outside the sliding window.
///
/// Resets entries in the cache slice that fall outside the current
/// attention window to their default value, freeing them for reuse.
///
/// Returns the number of entries evicted.
///
/// - `cache`: Mutable slice of cache entries (one per position).
/// - `current_pos`: The current token position.
/// - `config`: Sliding window configuration.
pub fn evict_outside_window<T: Default + Clone>(
    cache: &mut [T],
    current_pos: usize,
    config: &SlidingWindowConfig,
) -> usize {
    if config.window_size == 0 || cache.is_empty() {
        return 0;
    }

    let seq_len = cache.len().min(current_pos + 1);
    let (valid_positions, _) = attention_range(current_pos, seq_len, config);

    let mut evicted = 0;
    for (pos, entry) in cache.iter_mut().enumerate().take(seq_len) {
        if !valid_positions.contains(&pos) {
            *entry = T::default();
            evicted += 1;
        }
    }

    evicted
}

/// Check whether a given key position should be included in attention
/// for a query at `query_pos`.
#[inline]
pub fn is_in_window(key_pos: usize, query_pos: usize, config: &SlidingWindowConfig) -> bool {
    if config.window_size == 0 {
        return false;
    }
    // Sink tokens are always included
    if key_pos < config.sink_tokens {
        return true;
    }
    // Check if within the recent window
    if query_pos >= key_pos {
        let distance = query_pos - key_pos;
        let recent_budget = config.window_size.saturating_sub(config.sink_tokens);
        distance < recent_budget
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SlidingWindowConfig::default();
        assert_eq!(config.window_size, 4096);
        assert_eq!(config.sink_tokens, 4);
        assert!(!config.is_disabled());
    }

    #[test]
    fn disabled_config() {
        let config = SlidingWindowConfig::new(0, 0);
        assert!(config.is_disabled());
    }

    #[test]
    fn small_sequence_within_window() {
        let config = SlidingWindowConfig::new(8, 2);
        // Sequence of 5 tokens, all within window of 8
        let (positions, count) = attention_range(4, 5, &config);
        assert_eq!(count, 5);
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn window_slides_past_beginning() {
        let config = SlidingWindowConfig::new(4, 2);
        // At position 10, with seq_len=11, window=4, sink=2:
        // Sink: [0, 1], Recent budget: 4-2=2, so recent: [9, 10]
        let (positions, count) = attention_range(10, 11, &config);
        assert_eq!(count, 4);
        assert_eq!(positions, vec![0, 1, 9, 10]);
    }

    #[test]
    fn sink_tokens_always_included() {
        let config = SlidingWindowConfig::new(4, 2);
        // Even at a far position, sink tokens are retained
        let (positions, _) = attention_range(100, 101, &config);
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
    }

    #[test]
    fn window_with_no_sinks() {
        let config = SlidingWindowConfig::new(3, 0);
        // At position 10, window=3, no sinks: recent [8, 9, 10]
        let (positions, count) = attention_range(10, 11, &config);
        assert_eq!(count, 3);
        assert_eq!(positions, vec![8, 9, 10]);
    }

    #[test]
    fn empty_sequence() {
        let config = SlidingWindowConfig::default();
        let (positions, count) = attention_range(0, 0, &config);
        assert_eq!(count, 0);
        assert!(positions.is_empty());
    }

    #[test]
    fn mask_application() {
        let config = SlidingWindowConfig::new(3, 1);
        // Position 5, keys at positions [0, 1, 2, 3, 4, 5]
        // Window=3, sink=1 -> sink: [0], recent budget=2: [4, 5]
        // Valid: [0, 4, 5]
        let key_positions: Vec<usize> = (0..6).collect();
        let mut scores = vec![1.0; 6];

        apply_sliding_window_mask(&mut scores, 5, &key_positions, &config);

        assert!(scores[0].is_finite(), "sink token should be kept");
        assert!(scores[1] == f32::NEG_INFINITY, "pos 1 should be masked");
        assert!(scores[2] == f32::NEG_INFINITY, "pos 2 should be masked");
        assert!(scores[3] == f32::NEG_INFINITY, "pos 3 should be masked");
        assert!(scores[4].is_finite(), "recent pos 4 should be kept");
        assert!(scores[5].is_finite(), "recent pos 5 should be kept");
    }

    #[test]
    fn eviction() {
        let config = SlidingWindowConfig::new(3, 1);
        let mut cache: Vec<i32> = vec![10, 20, 30, 40, 50, 60];

        // At position 5 with window=3 and sink=1:
        // Valid: [0, 4, 5], evict: [1, 2, 3]
        let evicted = evict_outside_window(&mut cache, 5, &config);

        assert_eq!(evicted, 3);
        assert_eq!(cache[0], 10, "sink should be preserved");
        assert_eq!(cache[1], 0, "pos 1 should be evicted");
        assert_eq!(cache[2], 0, "pos 2 should be evicted");
        assert_eq!(cache[3], 0, "pos 3 should be evicted");
        assert_eq!(cache[4], 50, "pos 4 should be preserved");
        assert_eq!(cache[5], 60, "pos 5 should be preserved");
    }

    #[test]
    fn is_in_window_basic() {
        let config = SlidingWindowConfig::new(4, 2);

        // Sink tokens always in window
        assert!(is_in_window(0, 100, &config));
        assert!(is_in_window(1, 100, &config));

        // Recent positions in window (budget = 4 - 2 = 2)
        assert!(is_in_window(100, 100, &config)); // distance = 0
        assert!(is_in_window(99, 100, &config)); // distance = 1

        // Outside window
        assert!(!is_in_window(98, 100, &config)); // distance = 2, budget = 2
    }

    #[test]
    fn zero_window_masks_everything() {
        let config = SlidingWindowConfig::new(0, 0);
        let mut scores = vec![1.0, 2.0, 3.0];
        let key_positions = vec![0, 1, 2];

        apply_sliding_window_mask(&mut scores, 2, &key_positions, &config);

        for score in &scores {
            assert_eq!(*score, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn window_total_count_bounded() {
        let config = SlidingWindowConfig::new(4, 2);
        // Even with long sequence, window bounds the attended positions
        let (positions, count) = attention_range(1000, 1001, &config);
        assert!(count <= config.window_size);
        assert_eq!(positions.len(), count);
    }
}
