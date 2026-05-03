//! Per-layer attention configuration for mixed-strategy transformer models.
//!
//! Different layers in a model can use different attention strategies:
//! - Full attention (standard quadratic attention over the whole context)
//! - Sliding window attention (attend only to recent tokens + sinks)
//! - Different positional encodings (RoPE, ALiBi, or none)
//!
//! [`LayerAttentionConfig`] describes a single layer's attention strategy.
//! [`ModelAttentionConfig`] collects configs for all layers in a model and
//! provides convenient factory methods for standard architectures.

// ─── PositionalEncoding ──────────────────────────────────────────────────────

/// Positional encoding scheme used by an attention layer.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionalEncoding {
    /// Rotary Position Embedding (the default for Qwen3/Bonsai).
    ///
    /// `freq_base` controls the angular frequency base; typical values are
    /// 10 000 (original RoPE) or 1 000 000 (Qwen3 extended context).
    RoPE {
        /// Base frequency for the rotation angles.
        freq_base: f32,
    },

    /// ALiBi: Attention with Linear Biases.
    ///
    /// No learned parameters — a fixed slope per head penalises distance.
    AliBi,

    /// No positional encoding (e.g. layers after a sliding window that
    /// already encodes recency through its mask).
    None,
}

// ─── LayerAttentionConfig ─────────────────────────────────────────────────────

/// Attention configuration for a single transformer layer.
///
/// Constructed via the builder helpers (`with_rope`, `with_alibi`, …) for
/// ergonomic chaining:
///
/// ```rust
/// use oxibonsai_model::layers::attention_config::{LayerAttentionConfig, PositionalEncoding};
///
/// let cfg = LayerAttentionConfig::new(0, 32, 8, 128)
///     .with_rope(1_000_000.0)
///     .with_sliding_window(4096)
///     .with_sink_tokens(4);
///
/// assert!(cfg.sliding_window.is_some());
/// assert_eq!(cfg.positional_encoding, PositionalEncoding::RoPE { freq_base: 1_000_000.0 });
/// ```
#[derive(Debug, Clone)]
pub struct LayerAttentionConfig {
    /// Zero-based index of this layer in the transformer stack.
    pub layer_idx: usize,
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key-value heads (GQA; `num_q_heads / num_kv_heads` is the
    /// query-per-kv ratio).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Positional encoding strategy.
    pub positional_encoding: PositionalEncoding,
    /// Sliding window size. `None` means full (unbounded) attention.
    pub sliding_window: Option<usize>,
    /// Number of "attention sink" tokens at sequence position 0 that are
    /// always retained inside the sliding window. Only meaningful when
    /// `sliding_window.is_some()`.
    pub sink_tokens: usize,
    /// Attention scale factor applied to raw dot products.
    ///
    /// Defaults to `1 / sqrt(head_dim)` as per the original Transformer paper.
    pub scale: f32,
}

impl LayerAttentionConfig {
    /// Create a new layer config with sensible defaults.
    ///
    /// - Positional encoding: `PositionalEncoding::None`
    /// - Sliding window: disabled (`None`)
    /// - Sink tokens: 0
    /// - Scale: `1 / sqrt(head_dim)`
    pub fn new(layer_idx: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let scale = if head_dim > 0 {
            1.0_f32 / (head_dim as f32).sqrt()
        } else {
            1.0_f32
        };
        Self {
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
            positional_encoding: PositionalEncoding::None,
            sliding_window: None,
            sink_tokens: 0,
            scale,
        }
    }

    /// Set positional encoding to RoPE with the given frequency base.
    #[must_use]
    pub fn with_rope(mut self, freq_base: f32) -> Self {
        self.positional_encoding = PositionalEncoding::RoPE { freq_base };
        self
    }

    /// Set positional encoding to ALiBi.
    #[must_use]
    pub fn with_alibi(mut self) -> Self {
        self.positional_encoding = PositionalEncoding::AliBi;
        self
    }

    /// Enable sliding window attention with the given window size (in tokens).
    #[must_use]
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Set the number of attention sink tokens to always retain in the window.
    #[must_use]
    pub fn with_sink_tokens(mut self, n: usize) -> Self {
        self.sink_tokens = n;
        self
    }

    /// Returns `true` when this layer uses full (unbounded) attention.
    #[inline]
    pub fn is_full_attention(&self) -> bool {
        self.sliding_window.is_none()
    }

    /// Compute the effective KV-cache length visible to this layer.
    ///
    /// - For full attention: `total_len` (all tokens up to and including
    ///   `q_pos`).
    /// - For sliding window: `min(window + sink_tokens, total_len)`.
    ///
    /// `total_len` is the number of tokens currently in the KV cache.
    /// `q_pos` is the absolute position of the query (0-based).
    pub fn effective_kv_len(&self, total_len: usize, _q_pos: usize) -> usize {
        match self.sliding_window {
            None => total_len,
            Some(window) => (window + self.sink_tokens).min(total_len),
        }
    }
}

// ─── ModelAttentionConfig ─────────────────────────────────────────────────────

/// Attention configuration for every layer in a model.
///
/// Provides factory methods for common architectures and utilities for
/// analysing and planning KV cache memory usage.
pub struct ModelAttentionConfig {
    /// Per-layer configs, indexed by layer index.
    pub layers: Vec<LayerAttentionConfig>,
}

impl ModelAttentionConfig {
    /// Create a model config from a pre-built `Vec` of layer configs.
    pub fn new(layers: Vec<LayerAttentionConfig>) -> Self {
        Self { layers }
    }

    /// Standard Bonsai-8B configuration.
    ///
    /// 36 layers, all using full attention with RoPE (freq_base = 1 000 000),
    /// 32 query heads, 8 KV heads, head_dim = 128.
    pub fn bonsai_8b() -> Self {
        let layers = (0..36)
            .map(|i| LayerAttentionConfig::new(i, 32, 8, 128).with_rope(1_000_000.0))
            .collect();
        Self { layers }
    }

    /// Hypothetical mixed-window configuration.
    ///
    /// Even-indexed layers use full attention; odd-indexed layers use
    /// sliding window attention with `window_size = 1024` and 4 sink tokens.
    /// Useful for experimenting with hybrid long-context strategies.
    pub fn mixed_window_config(
        num_layers: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let head_dim = hidden_size.checked_div(num_q_heads).unwrap_or(hidden_size);
        let layers = (0..num_layers)
            .map(|i| {
                let base = LayerAttentionConfig::new(i, num_q_heads, num_kv_heads, head_dim)
                    .with_rope(1_000_000.0);
                if i % 2 == 0 {
                    // Even layer: full attention
                    base
                } else {
                    // Odd layer: sliding window
                    base.with_sliding_window(1024).with_sink_tokens(4)
                }
            })
            .collect();
        Self { layers }
    }

    /// Return the config for `layer_idx`, or `None` if out of range.
    pub fn get(&self, layer_idx: usize) -> Option<&LayerAttentionConfig> {
        self.layers.get(layer_idx)
    }

    /// Total number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of layers that use full (unbounded) attention.
    pub fn full_attention_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.is_full_attention()).count()
    }

    /// Number of layers that use sliding window attention.
    pub fn sliding_window_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.sliding_window.is_some())
            .count()
    }

    /// Estimate the peak KV cache memory in bytes for a given context length
    /// and batch size.
    ///
    /// For each layer the visible context is `effective_kv_len(context_len, context_len - 1)`.
    /// Each KV pair is stored as `f32` (4 bytes) and we store both K and V:
    ///
    /// ```text
    /// per_layer = num_kv_heads * head_dim * eff_kv_len * 4 (bytes) * 2 (K+V)
    /// total     = sum_over_layers(per_layer) * num_batches
    /// ```
    pub fn memory_estimate_kv_cache(&self, context_len: usize, num_batches: usize) -> usize {
        let q_pos = context_len.saturating_sub(1);
        self.layers
            .iter()
            .map(|l| {
                let eff = l.effective_kv_len(context_len, q_pos);
                // K + V, f32 = 4 bytes
                l.num_kv_heads * l.head_dim * eff * 4 * 2
            })
            .sum::<usize>()
            * num_batches
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_attention_config_defaults() {
        let cfg = LayerAttentionConfig::new(3, 16, 4, 64);
        assert_eq!(cfg.layer_idx, 3);
        assert_eq!(cfg.num_q_heads, 16);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.positional_encoding, PositionalEncoding::None);
        assert!(cfg.sliding_window.is_none());
        assert_eq!(cfg.sink_tokens, 0);
        // scale = 1/sqrt(64) = 0.125
        let expected_scale = 1.0_f32 / 64.0_f32.sqrt();
        assert!((cfg.scale - expected_scale).abs() < 1e-7, "scale mismatch");
    }

    #[test]
    fn test_layer_attention_config_with_rope() {
        let cfg = LayerAttentionConfig::new(0, 32, 8, 128).with_rope(1_000_000.0);
        assert_eq!(
            cfg.positional_encoding,
            PositionalEncoding::RoPE {
                freq_base: 1_000_000.0
            }
        );
    }

    #[test]
    fn test_layer_attention_config_with_alibi() {
        let cfg = LayerAttentionConfig::new(0, 8, 8, 64).with_alibi();
        assert_eq!(cfg.positional_encoding, PositionalEncoding::AliBi);
    }

    #[test]
    fn test_layer_attention_config_sliding_window() {
        let cfg = LayerAttentionConfig::new(1, 8, 2, 64)
            .with_sliding_window(512)
            .with_sink_tokens(4);
        assert_eq!(cfg.sliding_window, Some(512));
        assert_eq!(cfg.sink_tokens, 4);
        assert!(!cfg.is_full_attention());
    }

    #[test]
    fn test_effective_kv_len_full() {
        let cfg = LayerAttentionConfig::new(0, 8, 2, 64);
        // Full attention: effective = total_len regardless of q_pos
        assert_eq!(cfg.effective_kv_len(100, 99), 100);
        assert_eq!(cfg.effective_kv_len(1, 0), 1);
        assert_eq!(cfg.effective_kv_len(0, 0), 0);
    }

    #[test]
    fn test_effective_kv_len_sliding() {
        let cfg = LayerAttentionConfig::new(0, 8, 2, 64)
            .with_sliding_window(8)
            .with_sink_tokens(2);
        // window + sinks = 10; total_len=100 → min(10, 100) = 10
        assert_eq!(cfg.effective_kv_len(100, 99), 10);
        // total_len=5 < 10 → capped at 5
        assert_eq!(cfg.effective_kv_len(5, 4), 5);
        // total_len=10 == 10 → 10
        assert_eq!(cfg.effective_kv_len(10, 9), 10);
    }

    #[test]
    fn test_model_attention_config_bonsai_8b() {
        let cfg = ModelAttentionConfig::bonsai_8b();
        assert_eq!(cfg.num_layers(), 36);
        assert_eq!(cfg.full_attention_layers(), 36);
        assert_eq!(cfg.sliding_window_layers(), 0);

        let l0 = cfg.get(0).expect("layer 0 must exist");
        assert_eq!(l0.num_q_heads, 32);
        assert_eq!(l0.num_kv_heads, 8);
        assert_eq!(l0.head_dim, 128);
        assert_eq!(
            l0.positional_encoding,
            PositionalEncoding::RoPE {
                freq_base: 1_000_000.0
            }
        );
    }

    #[test]
    fn test_model_attention_config_mixed() {
        let cfg = ModelAttentionConfig::mixed_window_config(8, 1024, 8, 2);
        assert_eq!(cfg.num_layers(), 8);
        // Even layers (0, 2, 4, 6) → full attention = 4
        assert_eq!(cfg.full_attention_layers(), 4);
        // Odd layers (1, 3, 5, 7) → sliding window = 4
        assert_eq!(cfg.sliding_window_layers(), 4);

        let even = cfg.get(0).expect("layer 0 must exist");
        assert!(even.is_full_attention());

        let odd = cfg.get(1).expect("layer 1 must exist");
        assert_eq!(odd.sliding_window, Some(1024));
        assert_eq!(odd.sink_tokens, 4);
    }

    #[test]
    fn test_memory_estimate_kv_cache() {
        // Single layer, 2 kv heads, head_dim=4, context=10, batch=1
        // Full attention: eff=10
        // memory = 2 * 4 * 10 * 4 * 2 * 1 = 640 bytes
        let cfg = ModelAttentionConfig::new(vec![LayerAttentionConfig::new(0, 4, 2, 4)]);
        let mem = cfg.memory_estimate_kv_cache(10, 1);
        assert_eq!(mem, 640, "expected 640 bytes, got {mem}");

        // With sliding window (window=5, sinks=1 → eff=min(6,10)=6)
        // memory = 2 * 4 * 6 * 4 * 2 * 1 = 384 bytes
        let cfg2 = ModelAttentionConfig::new(vec![LayerAttentionConfig::new(0, 4, 2, 4)
            .with_sliding_window(5)
            .with_sink_tokens(1)]);
        let mem2 = cfg2.memory_estimate_kv_cache(10, 1);
        assert_eq!(mem2, 384, "expected 384 bytes, got {mem2}");
    }
}
