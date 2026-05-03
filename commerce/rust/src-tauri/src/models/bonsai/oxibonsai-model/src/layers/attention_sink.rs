//! Attention Sink for StreamingLLM: enables infinite-length text generation
//! by reserving initial "sink" tokens in the KV cache.
//!
//! # Algorithm
//!
//! Based on Xiao et al. 2023 — "Efficient Streaming Language Models with
//! Attention Sinks" (<https://arxiv.org/abs/2309.17453>).
//!
//! 1. Always keep the first `num_sink_tokens` KV pairs in the cache (sinks).
//! 2. Keep the most recent `window_size` non-sink tokens in a circular buffer.
//! 3. When at capacity: evict the oldest non-sink token (FIFO via VecDeque).
//! 4. Remap positions so the kept tokens have *contiguous* positions for RoPE
//!    re-application: sinks → 0..num_sink_tokens, recent → num_sink_tokens..
//!
//! # Layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  Sink slots [0..num_sink_tokens]   │  Recent circular buf │
//! │  (permanent, never evicted)        │  (FIFO, window_size) │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! The KV data is stored head-major: each head owns its own `Vec<SinkSlot>`
//! (sinks) and `VecDeque<SinkSlot>` (recent), so per-head retrieval is a
//! single contiguous copy.

use std::collections::VecDeque;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────

/// Errors produced by the attention sink cache.
#[derive(Debug, Error)]
pub enum SinkError {
    /// Requested head index is out of range.
    #[error("head {head} out of range (num_heads = {num_heads})")]
    HeadOutOfRange { head: usize, num_heads: usize },

    /// Requested layer index is out of range.
    #[error("layer {layer} out of range (num_layers = {num_layers})")]
    LayerOutOfRange { layer: usize, num_layers: usize },

    /// Input slice has wrong number of elements.
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// Attempted to read sink slots before they have all been filled.
    #[error("sink slots not yet filled (only {filled}/{total} sink tokens pushed)")]
    SinkNotFilled { filled: usize, total: usize },
}

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for the attention sink window.
///
/// The total cache capacity is `num_sink_tokens + window_size` KV pairs
/// per head per layer. Once the capacity is reached, new tokens evict the
/// oldest non-sink entry from `recent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionSinkConfig {
    /// Number of sink tokens to keep at all times (typically 4).
    pub num_sink_tokens: usize,

    /// Size of the sliding window for recent (non-sink) tokens.
    ///
    /// Common values: 512, 1024, 2048.
    pub window_size: usize,
}

impl AttentionSinkConfig {
    /// Create a new configuration.
    ///
    /// # Panics
    ///
    /// Does not panic. Both values may be zero (degenerate but valid config).
    pub fn new(num_sink_tokens: usize, window_size: usize) -> Self {
        Self {
            num_sink_tokens,
            window_size,
        }
    }

    /// Total cache capacity: `num_sink_tokens + window_size`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.num_sink_tokens + self.window_size
    }

    /// Effective maximum sequence length before eviction starts.
    ///
    /// Equals `num_sink_tokens + window_size` — identical to `capacity()`.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.capacity()
    }
}

impl Default for AttentionSinkConfig {
    fn default() -> Self {
        Self::new(4, 512)
    }
}

// ─────────────────────────────────────────────────────────────
// Single cache slot
// ─────────────────────────────────────────────────────────────

/// One cached KV entry for a single head at a single sequence position.
///
/// `key` and `value` both hold `head_dim` f32 values.
#[derive(Debug, Clone)]
pub struct SinkSlot {
    /// The token's position in the *original* (untruncated) sequence.
    pub original_position: usize,
    /// Key vector: `head_dim` elements.
    pub key: Vec<f32>,
    /// Value vector: `head_dim` elements.
    pub value: Vec<f32>,
}

impl SinkSlot {
    fn new(original_position: usize, key: Vec<f32>, value: Vec<f32>) -> Self {
        Self {
            original_position,
            key,
            value,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Per-layer attention sink cache
// ─────────────────────────────────────────────────────────────

/// The attention sink KV cache for a **single transformer layer**.
///
/// Maintains:
/// - `sinks`: permanent storage for the first `num_sink_tokens` tokens.
/// - `recent`: circular FIFO buffer of the most recent `window_size` tokens.
///
/// Layout is *head-major*: `sinks[head][sink_idx]` and `recent[head]`.
pub struct AttentionSinkLayer {
    config: AttentionSinkConfig,
    head_dim: usize,
    num_heads: usize,
    /// Permanent sink slots: `[num_heads][num_sink_tokens]`.
    ///
    /// A head's inner `Vec` grows from 0 to `num_sink_tokens` as the first
    /// tokens are pushed. After that it is frozen.
    sinks: Vec<Vec<SinkSlot>>,
    /// Recent token slots: `[num_heads]`, each a FIFO of ≤ `window_size` entries.
    recent: Vec<VecDeque<SinkSlot>>,
    /// Total tokens ever pushed into this layer (including evicted ones).
    pub total_tokens: usize,
    /// Tokens that have been evicted from `recent` so far.
    evicted: usize,
}

impl AttentionSinkLayer {
    /// Create an empty layer cache.
    pub fn new(config: AttentionSinkConfig, num_heads: usize, head_dim: usize) -> Self {
        let sinks = (0..num_heads).map(|_| Vec::new()).collect();
        let recent = (0..num_heads)
            .map(|_| VecDeque::with_capacity(config.window_size))
            .collect();
        Self {
            config,
            head_dim,
            num_heads,
            sinks,
            recent,
            total_tokens: 0,
            evicted: 0,
        }
    }

    // ── Internal helpers ────────────────────────────────────────

    /// Slice of key data for head `h` at flat index `flat`.
    ///
    /// `flat` is an offset into the concatenated `keys` slice
    /// (`num_heads * head_dim` elements).
    #[inline]
    fn head_key_slice(keys: &[f32], h: usize, head_dim: usize) -> &[f32] {
        let start = h * head_dim;
        &keys[start..start + head_dim]
    }

    #[inline]
    fn head_value_slice(values: &[f32], h: usize, head_dim: usize) -> &[f32] {
        let start = h * head_dim;
        &values[start..start + head_dim]
    }

    // ── Public API ──────────────────────────────────────────────

    /// Push one token's KV data into the cache.
    ///
    /// `keys` and `values` must each have exactly `num_heads * head_dim`
    /// elements laid out in head-major order.
    ///
    /// Routing:
    /// - If `total_tokens < num_sink_tokens` → appended to sink slots.
    /// - Otherwise → appended to the `recent` circular buffer.
    ///   If the buffer is full, the oldest entry is evicted first.
    pub fn push(&mut self, keys: &[f32], values: &[f32]) -> Result<(), SinkError> {
        let expected = self.num_heads * self.head_dim;
        if keys.len() != expected {
            return Err(SinkError::ShapeMismatch {
                expected,
                actual: keys.len(),
            });
        }
        if values.len() != expected {
            return Err(SinkError::ShapeMismatch {
                expected,
                actual: values.len(),
            });
        }

        let pos = self.total_tokens;
        let is_sink = pos < self.config.num_sink_tokens;

        for h in 0..self.num_heads {
            let k = Self::head_key_slice(keys, h, self.head_dim).to_vec();
            let v = Self::head_value_slice(values, h, self.head_dim).to_vec();
            let slot = SinkSlot::new(pos, k, v);

            if is_sink {
                // Sink region: simply append (fills up to num_sink_tokens).
                self.sinks[h].push(slot);
            } else {
                // Recent region: evict oldest if at window capacity.
                if self.recent[h].len() >= self.config.window_size {
                    // Only count evictions on head 0 to avoid N-fold counting.
                    if h == 0 {
                        self.evicted += 1;
                    }
                    self.recent[h].pop_front();
                }
                self.recent[h].push_back(slot);
            }
        }

        self.total_tokens += 1;
        Ok(())
    }

    /// Return the remapped position vector for all cached tokens.
    ///
    /// Sink tokens receive positions `0..num_sink_tokens`.
    /// Recent tokens receive contiguous positions starting from
    /// `num_sink_tokens`, preserving their relative order.
    ///
    /// This remapping lets RoPE embeddings be applied correctly even after
    /// evictions have created gaps in the original position sequence.
    pub fn get_remapped_positions(&self) -> Vec<usize> {
        let sink_count = self.sinks.first().map(|s| s.len()).unwrap_or(0);
        let recent_count = self.recent.first().map(|r| r.len()).unwrap_or(0);
        let total = sink_count + recent_count;
        let mut positions = Vec::with_capacity(total);
        for i in 0..sink_count {
            positions.push(i);
        }
        for j in 0..recent_count {
            positions.push(sink_count + j);
        }
        positions
    }

    /// Total number of tokens currently in cache (sinks + recent).
    #[inline]
    pub fn cache_len(&self) -> usize {
        let sink_count = self.sinks.first().map(|s| s.len()).unwrap_or(0);
        let recent_count = self.recent.first().map(|r| r.len()).unwrap_or(0);
        sink_count + recent_count
    }

    /// Number of recent (non-sink) tokens currently cached.
    #[inline]
    pub fn recent_len(&self) -> usize {
        self.recent.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Returns `true` once the cache is in streaming mode — i.e., the recent
    /// buffer has been filled at least once and evictions have begun.
    #[inline]
    pub fn is_streaming(&self) -> bool {
        self.evicted > 0
    }

    /// Get all cached key vectors for a given head, concatenated in order:
    /// sink keys first, then recent keys.
    ///
    /// Returns a flat `Vec<f32>` of length `cache_len() * head_dim`.
    pub fn get_keys_for_head(&self, head: usize) -> Result<Vec<f32>, SinkError> {
        if head >= self.num_heads {
            return Err(SinkError::HeadOutOfRange {
                head,
                num_heads: self.num_heads,
            });
        }
        let cap = self.cache_len() * self.head_dim;
        let mut out = Vec::with_capacity(cap);
        for slot in &self.sinks[head] {
            out.extend_from_slice(&slot.key);
        }
        for slot in &self.recent[head] {
            out.extend_from_slice(&slot.key);
        }
        Ok(out)
    }

    /// Get all cached value vectors for a given head.
    ///
    /// Returns a flat `Vec<f32>` of length `cache_len() * head_dim`.
    pub fn get_values_for_head(&self, head: usize) -> Result<Vec<f32>, SinkError> {
        if head >= self.num_heads {
            return Err(SinkError::HeadOutOfRange {
                head,
                num_heads: self.num_heads,
            });
        }
        let cap = self.cache_len() * self.head_dim;
        let mut out = Vec::with_capacity(cap);
        for slot in &self.sinks[head] {
            out.extend_from_slice(&slot.value);
        }
        for slot in &self.recent[head] {
            out.extend_from_slice(&slot.value);
        }
        Ok(out)
    }

    /// Number of non-sink tokens that have been evicted from the cache.
    #[inline]
    pub fn evicted_count(&self) -> usize {
        self.evicted
    }

    /// Approximate memory used by this layer's cache, in bytes.
    ///
    /// Accounts for all `SinkSlot` key and value vectors across every head.
    pub fn memory_bytes(&self) -> usize {
        let bytes_per_slot = self.head_dim * std::mem::size_of::<f32>() * 2; // key + value
        let sink_slots: usize = self.sinks.iter().map(|s| s.len()).sum();
        let recent_slots: usize = self.recent.iter().map(|r| r.len()).sum();
        (sink_slots + recent_slots) * bytes_per_slot
    }
}

// ─────────────────────────────────────────────────────────────
// Multi-layer cache
// ─────────────────────────────────────────────────────────────

/// Multi-layer attention sink KV cache.
///
/// Wraps one [`AttentionSinkLayer`] per transformer layer, presenting a
/// unified interface for the decode loop.
///
/// # Typical usage
///
/// ```rust,ignore
/// let mut cache = AttentionSinkCache::new(32, 32, 128, AttentionSinkConfig::default());
///
/// // Each decode step:
/// cache.push_step(&all_keys, &all_values)?;
///
/// // Retrieve keys/values for attention:
/// let keys = cache.get_keys_for_head(layer, head)?;
/// let positions = cache.get_remapped_positions(layer)?;
/// ```
pub struct AttentionSinkCache {
    layers: Vec<AttentionSinkLayer>,
    config: AttentionSinkConfig,
    /// Number of transformer layers.
    pub num_layers: usize,
}

impl AttentionSinkCache {
    /// Create a new multi-layer cache.
    ///
    /// - `num_layers`: number of transformer layers.
    /// - `num_heads`: number of attention heads per layer.
    /// - `head_dim`: dimension of each attention head.
    /// - `config`: attention sink configuration.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        config: AttentionSinkConfig,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| AttentionSinkLayer::new(config.clone(), num_heads, head_dim))
            .collect();
        Self {
            layers,
            config,
            num_layers,
        }
    }

    /// Push one decode step's KV data across all layers simultaneously.
    ///
    /// - `all_keys[layer]`: flat `Vec<f32>` of shape `[num_heads * head_dim]`.
    /// - `all_values[layer]`: same shape.
    ///
    /// Returns an error if lengths don't match `num_layers` or if any
    /// individual layer's push fails.
    pub fn push_step(
        &mut self,
        all_keys: &[Vec<f32>],
        all_values: &[Vec<f32>],
    ) -> Result<(), SinkError> {
        if all_keys.len() != self.num_layers {
            return Err(SinkError::ShapeMismatch {
                expected: self.num_layers,
                actual: all_keys.len(),
            });
        }
        if all_values.len() != self.num_layers {
            return Err(SinkError::ShapeMismatch {
                expected: self.num_layers,
                actual: all_values.len(),
            });
        }
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.push(&all_keys[layer_idx], &all_values[layer_idx])?;
        }
        Ok(())
    }

    /// Get all cached keys for a specific layer and head.
    ///
    /// Returns a flat `Vec<f32>` of length `cache_len() * head_dim`.
    pub fn get_keys_for_head(&self, layer: usize, head: usize) -> Result<Vec<f32>, SinkError> {
        self.layer(layer)?.get_keys_for_head(head)
    }

    /// Get all cached values for a specific layer and head.
    pub fn get_values_for_head(&self, layer: usize, head: usize) -> Result<Vec<f32>, SinkError> {
        self.layer(layer)?.get_values_for_head(head)
    }

    /// Get remapped positions for a specific layer.
    pub fn get_remapped_positions(&self, layer: usize) -> Result<Vec<usize>, SinkError> {
        Ok(self.layer(layer)?.get_remapped_positions())
    }

    /// Total tokens currently cached (using layer 0 as reference).
    ///
    /// All layers are kept in sync, so any layer gives the same answer.
    pub fn cache_len(&self) -> usize {
        self.layers.first().map(|l| l.cache_len()).unwrap_or(0)
    }

    /// Whether the cache is in streaming mode (evictions have begun).
    pub fn is_streaming(&self) -> bool {
        self.layers
            .first()
            .map(|l| l.is_streaming())
            .unwrap_or(false)
    }

    /// Total tokens evicted across **all** layers (sum over all layers).
    ///
    /// Because each layer is an independent cache, evictions accumulate
    /// independently. Divide by `num_layers` to get per-layer evictions.
    pub fn total_evicted(&self) -> usize {
        self.layers.iter().map(|l| l.evicted_count()).sum()
    }

    /// Reference to the config used when constructing this cache.
    pub fn config(&self) -> &AttentionSinkConfig {
        &self.config
    }

    // ── Private ─────────────────────────────────────────────────

    #[inline]
    fn layer(&self, layer: usize) -> Result<&AttentionSinkLayer, SinkError> {
        self.layers.get(layer).ok_or(SinkError::LayerOutOfRange {
            layer,
            num_layers: self.num_layers,
        })
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests (basic smoke tests; exhaustive tests in tests/ dir)
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kv(num_heads: usize, head_dim: usize, val: f32) -> Vec<f32> {
        vec![val; num_heads * head_dim]
    }

    #[test]
    fn config_default_values() {
        let cfg = AttentionSinkConfig::default();
        assert_eq!(cfg.num_sink_tokens, 4);
        assert_eq!(cfg.window_size, 512);
        assert_eq!(cfg.capacity(), 516);
        assert_eq!(cfg.max_seq_len(), 516);
    }

    #[test]
    fn push_sink_and_recent() {
        let cfg = AttentionSinkConfig::new(2, 3);
        let mut layer = AttentionSinkLayer::new(cfg, 1, 4);
        // Push 2 sink tokens
        layer
            .push(&make_kv(1, 4, 1.0), &make_kv(1, 4, 1.0))
            .expect("push sink 0");
        layer
            .push(&make_kv(1, 4, 2.0), &make_kv(1, 4, 2.0))
            .expect("push sink 1");
        assert_eq!(layer.cache_len(), 2);
        assert_eq!(layer.recent_len(), 0);
        assert!(!layer.is_streaming());

        // Push into recent
        layer
            .push(&make_kv(1, 4, 3.0), &make_kv(1, 4, 3.0))
            .expect("push recent 0");
        assert_eq!(layer.cache_len(), 3);
        assert_eq!(layer.recent_len(), 1);
    }

    #[test]
    fn eviction_and_streaming_flag() {
        let cfg = AttentionSinkConfig::new(1, 2);
        let mut layer = AttentionSinkLayer::new(cfg, 1, 2);
        // Fill: 1 sink + 2 recent = capacity 3
        for i in 0..3u32 {
            layer
                .push(&[i as f32, i as f32], &[i as f32, i as f32])
                .expect("push");
        }
        assert!(!layer.is_streaming());
        assert_eq!(layer.cache_len(), 3);

        // One more — triggers eviction
        layer.push(&[9.0, 9.0], &[9.0, 9.0]).expect("evicting push");
        assert!(layer.is_streaming());
        assert_eq!(layer.evicted_count(), 1);
        // Cache length stays at capacity
        assert_eq!(layer.cache_len(), 3);
    }

    #[test]
    fn remapped_positions_contiguous() {
        let cfg = AttentionSinkConfig::new(2, 3);
        let mut layer = AttentionSinkLayer::new(cfg, 1, 2);
        for i in 0..4u32 {
            layer
                .push(&[i as f32, i as f32], &[i as f32, i as f32])
                .expect("push");
        }
        let positions = layer.get_remapped_positions();
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn multi_layer_cache_push_step() {
        let cfg = AttentionSinkConfig::new(2, 4);
        let mut cache = AttentionSinkCache::new(3, 2, 8, cfg);
        let keys: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0f32; 16]).collect();
        let values: Vec<Vec<f32>> = (0..3).map(|_| vec![2.0f32; 16]).collect();
        cache.push_step(&keys, &values).expect("push step");
        assert_eq!(cache.cache_len(), 1);
        assert!(!cache.is_streaming());
    }
}
