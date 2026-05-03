//! Grouped Query Attention (GQA) for Qwen3.
//!
//! Qwen3-8B uses 32 query heads and 8 key-value heads (ratio 4:1).
//! Each head has dimension 128.
//!
//! This module provides:
//! - Basic single-head attention
//! - Causal masking support
//! - Multi-head attention with optional sliding window
//! - Numerically stable softmax

use crate::error::{ModelError, ModelResult};
use crate::layers::sliding_window::SlidingWindowConfig;

/// Compute softmax over a slice in-place.
///
/// Delegates to the SIMD-accelerated implementation in `oxibonsai_kernels`.
pub fn softmax(values: &mut [f32]) {
    oxibonsai_kernels::softmax_simd(values);
}

/// Compute dot product of two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Single-head attention: `softmax(Q @ K^T / sqrt(d)) @ V`.
///
/// - `query`: Query vector \[head_dim\].
/// - `keys`: Cached key vectors \[seq_len x head_dim\] (row-major).
/// - `values`: Cached value vectors \[seq_len x head_dim\] (row-major).
/// - `output`: Result vector \[head_dim\].
/// - `seq_len`: Number of tokens in KV cache.
/// - `head_dim`: Dimension per head.
pub fn attention_head(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) -> ModelResult<()> {
    debug_assert_eq!(query.len(), head_dim);
    debug_assert!(keys.len() >= seq_len * head_dim);
    debug_assert!(values.len() >= seq_len * head_dim);
    debug_assert!(output.len() >= head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: Q @ K^T (scaled)
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let key = &keys[t * head_dim..(t + 1) * head_dim];
        scores[t] = dot(query, key) * scale;
    }

    // Softmax
    softmax(&mut scores);

    // Weighted sum of values
    for d in 0..head_dim {
        let mut sum = 0.0f32;
        for t in 0..seq_len {
            sum += scores[t] * values[t * head_dim + d];
        }
        output[d] = sum;
    }

    Ok(())
}

// ─── Causal Mask ──────────────────────────────────────────────────

/// Precomputed causal attention mask.
///
/// For a sequence of length N, position `i` can only attend to
/// positions `j <= i`. This struct generates the mask efficiently
/// and optionally incorporates sliding window constraints.
#[derive(Debug, Clone)]
pub struct CausalMask {
    /// Maximum sequence length this mask was created for.
    max_seq_len: usize,
    /// Optional sliding window configuration.
    sliding_window: Option<SlidingWindowConfig>,
}

impl CausalMask {
    /// Create a new causal mask for up to `max_seq_len` positions.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            sliding_window: None,
        }
    }

    /// Create a causal mask with sliding window attention.
    pub fn with_sliding_window(max_seq_len: usize, config: SlidingWindowConfig) -> Self {
        Self {
            max_seq_len,
            sliding_window: Some(config),
        }
    }

    /// Check whether query position `q_pos` can attend to key position `k_pos`.
    #[inline]
    pub fn is_allowed(&self, q_pos: usize, k_pos: usize) -> bool {
        // Basic causal constraint: can only attend to past and current positions
        if k_pos > q_pos {
            return false;
        }
        // Sliding window constraint
        if let Some(ref sw) = self.sliding_window {
            return crate::layers::sliding_window::is_in_window(k_pos, q_pos, sw);
        }
        true
    }

    /// Apply the causal mask to a row of attention scores.
    ///
    /// Sets scores for disallowed positions to `f32::NEG_INFINITY`.
    /// `scores` has length `key_len`, one score per key position `0..key_len`.
    pub fn apply(&self, scores: &mut [f32], query_pos: usize) {
        for (k_pos, score) in scores.iter_mut().enumerate() {
            if !self.is_allowed(query_pos, k_pos) {
                *score = f32::NEG_INFINITY;
            }
        }
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Single-head attention with causal mask support.
///
/// Like `attention_head` but applies a causal mask (and optionally
/// a sliding window mask) to the attention scores before softmax.
#[allow(clippy::too_many_arguments)]
pub fn attention_head_with_mask(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    query_pos: usize,
    mask: &CausalMask,
) -> ModelResult<()> {
    if query.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query".to_string(),
            expected: vec![head_dim],
            actual: vec![query.len()],
        });
    }
    if keys.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "keys".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![keys.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: Q @ K^T (scaled)
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let key = &keys[t * head_dim..(t + 1) * head_dim];
        scores[t] = dot(query, key) * scale;
    }

    // Apply causal mask
    mask.apply(&mut scores, query_pos);

    // Numerically stable softmax
    softmax(&mut scores);

    // Weighted sum of values
    for d in 0..head_dim {
        let mut sum = 0.0f32;
        for t in 0..seq_len {
            sum += scores[t] * values[t * head_dim + d];
        }
        output[d] = sum;
    }

    Ok(())
}

/// Multi-head attention convenience function.
///
/// Splits query into `num_heads` heads, performs attention for each head
/// (with GQA head mapping), and concatenates the results.
///
/// - `query_all`: All query heads concatenated `[num_heads * head_dim]`.
/// - `keys`: Per-head key cache slices (one per KV head).
/// - `values`: Per-head value cache slices (one per KV head).
/// - `output`: Output buffer `[num_heads * head_dim]`.
/// - `num_heads`: Number of query heads.
/// - `num_kv_heads`: Number of KV heads (for GQA mapping).
/// - `head_dim`: Dimension per head.
/// - `seq_len`: Current sequence length for attention.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    query_all: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> ModelResult<()> {
    if query_all.len() < num_heads * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query_all".to_string(),
            expected: vec![num_heads * head_dim],
            actual: vec![query_all.len()],
        });
    }
    if output.len() < num_heads * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![num_heads * head_dim],
            actual: vec![output.len()],
        });
    }
    if keys.len() != num_kv_heads || values.len() != num_kv_heads {
        return Err(ModelError::ShapeMismatch {
            name: "keys/values kv_heads".to_string(),
            expected: vec![num_kv_heads],
            actual: vec![keys.len()],
        });
    }

    if num_kv_heads == 0 {
        return Err(ModelError::ShapeMismatch {
            name: "num_kv_heads".to_string(),
            expected: vec![1],
            actual: vec![0],
        });
    }
    let heads_per_group = num_heads / num_kv_heads;

    let mut head_output = vec![0.0f32; head_dim];

    for q_head in 0..num_heads {
        let kv_head = q_head / heads_per_group;
        let q_start = q_head * head_dim;

        attention_head(
            &query_all[q_start..q_start + head_dim],
            keys[kv_head],
            values[kv_head],
            &mut head_output,
            seq_len,
            head_dim,
        )?;

        output[q_start..q_start + head_dim].copy_from_slice(&head_output);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_basic() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be in increasing order
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
    }

    #[test]
    fn softmax_single() {
        let mut values = vec![5.0];
        softmax(&mut values);
        assert!((values[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn attention_single_token() {
        let head_dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![1.0, 0.0, 0.0, 0.0]; // 1 token
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let mut output = vec![0.0; 4];

        attention_head(&query, &keys, &values, &mut output, 1, head_dim).expect("should succeed");

        // With single token, softmax is 1.0, output = values
        for i in 0..4 {
            assert!(
                (output[i] - values[i]).abs() < 1e-4,
                "at {i}: expected {}, got {}",
                values[i],
                output[i]
            );
        }
    }

    // ── CausalMask tests ──

    #[test]
    fn causal_mask_basic() {
        let mask = CausalMask::new(16);
        // Position 5 can attend to 0..5
        assert!(mask.is_allowed(5, 0));
        assert!(mask.is_allowed(5, 5));
        // Cannot attend to future positions
        assert!(!mask.is_allowed(5, 6));
        assert!(!mask.is_allowed(5, 10));
    }

    #[test]
    fn causal_mask_with_sliding_window() {
        let sw = SlidingWindowConfig::new(3, 1);
        let mask = CausalMask::with_sliding_window(16, sw);

        // Position 10: sink=[0], recent budget=2: [9, 10]
        assert!(mask.is_allowed(10, 0)); // sink
        assert!(!mask.is_allowed(10, 5)); // outside window
        assert!(mask.is_allowed(10, 9)); // in window
        assert!(mask.is_allowed(10, 10)); // current position
        assert!(!mask.is_allowed(10, 11)); // future
    }

    #[test]
    fn causal_mask_apply_scores() {
        let mask = CausalMask::new(8);
        let mut scores = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // Query at position 2: can attend to 0, 1, 2 but not 3, 4
        mask.apply(&mut scores, 2);

        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        assert!(scores[2].is_finite());
        assert_eq!(scores[3], f32::NEG_INFINITY);
        assert_eq!(scores[4], f32::NEG_INFINITY);
    }

    #[test]
    fn attention_head_with_mask_single_token() {
        let head_dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let mut output = vec![0.0; 4];
        let mask = CausalMask::new(16);

        attention_head_with_mask(&query, &keys, &values, &mut output, 1, head_dim, 0, &mask)
            .expect("should succeed");

        for i in 0..4 {
            assert!(
                (output[i] - values[i]).abs() < 1e-4,
                "at {i}: expected {}, got {}",
                values[i],
                output[i]
            );
        }
    }

    #[test]
    fn attention_head_with_mask_causal() {
        let head_dim = 4;
        // 3 tokens
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];
        let values = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];

        let mask = CausalMask::new(16);

        // Query at position 0: should only attend to token 0
        let mut output = vec![0.0; 4];
        attention_head_with_mask(&query, &keys, &values, &mut output, 3, head_dim, 0, &mask)
            .expect("should succeed");

        // Output should be close to values[0] since tokens 1,2 are masked
        assert!((output[0] - 1.0).abs() < 1e-4);
    }

    // ── Multi-head attention tests ──

    #[test]
    fn multi_head_attention_basic() {
        let head_dim = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let seq_len = 1;

        let query_all = vec![
            1.0, 0.0, 0.0, 0.0, // head 0
            0.0, 1.0, 0.0, 0.0, // head 1
        ];
        let keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // 1 KV head, 1 token
        let values: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let keys_refs: Vec<&[f32]> = vec![&keys];
        let values_refs: Vec<&[f32]> = vec![&values];

        let mut output = vec![0.0; num_heads * head_dim];

        multi_head_attention(
            &query_all,
            &keys_refs,
            &values_refs,
            &mut output,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        )
        .expect("should succeed");

        // Both heads attend to the same single KV, so output = values for each head
        for (i, &val) in output[..head_dim].iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 1e-4,
                "head 0 dim {i}: expected 0.5, got {}",
                val
            );
        }
        for i in 0..head_dim {
            assert!(
                (output[head_dim + i] - 0.5).abs() < 1e-4,
                "head 1 dim {i}: expected 0.5, got {}",
                output[head_dim + i]
            );
        }
    }

    #[test]
    fn multi_head_attention_error_on_zero_kv_heads() {
        let result = multi_head_attention(&[1.0; 4], &[], &[], &mut [0.0; 4], 1, 0, 4, 1);
        assert!(result.is_err());
    }
}
