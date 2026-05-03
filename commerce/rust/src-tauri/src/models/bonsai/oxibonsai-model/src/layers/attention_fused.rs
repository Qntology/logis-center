//! Fused attention: memory-efficient single-pass attention computation.
//!
//! Standard attention materializes the full `seq_len x seq_len` attention
//! matrix, requiring O(seq_len^2) memory. For long sequences this dominates
//! memory usage and thrashes caches.
//!
//! This module implements **online softmax** (inspired by Flash Attention v2)
//! to compute attention output in a single pass over KV positions, using
//! only O(block_size) working memory regardless of sequence length.
//!
//! **Algorithm:**
//! ```text
//! For each block of KV positions:
//!   1. Compute QK^T scores for the block
//!   2. Update running max and sum_exp (online softmax)
//!   3. Rescale previous output accumulator
//!   4. Accumulate weighted V for this block
//! ```
//!
//! The final output is mathematically identical to standard attention
//! (within floating-point tolerance), but uses constant working memory.

use crate::error::{ModelError, ModelResult};
use crate::layers::attention::CausalMask;

/// Number of KV positions processed per attention block.
/// 32 positions x head_dim floats fits comfortably in L1 cache.
pub const ATTENTION_BLOCK_SIZE: usize = 32;

// ─── SIMD dot product ─────────────────────────────────────────────────────

/// Compute `dot(a, b)` with SIMD acceleration where available.
///
/// Dispatches at runtime to:
/// - NEON (aarch64): dual-accumulator 8-wide vfmaq_f32
/// - AVX2+FMA (x86_64): dual-accumulator 16-wide _mm256_fmadd_ps
/// - Scalar fallback for all other targets
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: neon feature confirmed above
            return unsafe { dot_f32_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
            // SAFETY: avx+fma features confirmed above
            return unsafe { dot_f32_avx2_fma(a, b) };
        }
    }

    dot_f32_scalar(a, b)
}

/// Scalar dot product with 4-way ILP accumulation.
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for i in (len - remainder)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON dot product: dual 128-bit accumulators = 8 floats/iter.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let b0 = vld1q_f32(b.as_ptr().add(i));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        i += 8;
    }
    let mut tail = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

/// AVX2+FMA dot product: dual 256-bit accumulators = 16 floats/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    // Horizontal sum of acc0 + acc1
    let combined = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(combined);
    let hi = _mm256_extractf128_ps(combined, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sum2);
    let sum1 = _mm_add_ss(sum2, shuf2);
    let mut tail = _mm_cvtss_f32(sum1);
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

// ─── Online Softmax State ──────────────────────────────────────────────

/// Running state for numerically stable online softmax computation.
///
/// Maintains the running maximum, exponential sum, and weighted output
/// accumulator across blocks of KV positions. The key insight is that
/// when we encounter a new maximum, we can rescale the previous
/// accumulator without needing to revisit past positions.
struct OnlineSoftmaxState {
    /// Running maximum of attention scores seen so far.
    max_val: f32,
    /// Running sum of exp(score - max_val) for all scores seen so far.
    sum_exp: f32,
    /// Running weighted sum of V vectors: sum(softmax_weight * V).
    /// Needs rescaling when max_val changes.
    output: Vec<f32>,
}

impl OnlineSoftmaxState {
    /// Create a new state for the given head dimension.
    fn new(head_dim: usize) -> Self {
        Self {
            max_val: f32::NEG_INFINITY,
            sum_exp: 0.0,
            output: vec![0.0f32; head_dim],
        }
    }

    /// Process a block of attention scores and corresponding V vectors.
    ///
    /// Updates the running softmax state and output accumulator.
    ///
    /// - `scores`: QK^T scores for this block (already scaled by 1/sqrt(d)).
    /// - `values`: Corresponding V vectors, each of length `head_dim`.
    /// - `head_dim`: Dimension of each V vector.
    fn update(&mut self, scores: &[f32], values: &[&[f32]], head_dim: usize) {
        debug_assert_eq!(scores.len(), values.len());

        for (idx, &score) in scores.iter().enumerate() {
            let v = values[idx];
            debug_assert_eq!(v.len(), head_dim);

            if score > self.max_val {
                // New maximum found: rescale previous accumulation
                let rescale = if self.max_val == f32::NEG_INFINITY {
                    0.0 // No previous accumulation to rescale
                } else {
                    (self.max_val - score).exp()
                };

                self.sum_exp *= rescale;
                for d in 0..head_dim {
                    self.output[d] *= rescale;
                }
                self.max_val = score;
            }

            let exp_score = (score - self.max_val).exp();
            self.sum_exp += exp_score;

            for (out_d, &v_d) in self.output[..head_dim].iter_mut().zip(v.iter()) {
                *out_d += exp_score * v_d;
            }
        }
    }

    /// Finalize the output by dividing by the total softmax denominator.
    ///
    /// After this call, `self.output` contains the correct attention output.
    fn finalize(&mut self) {
        if self.sum_exp > 0.0 {
            let inv_sum = 1.0 / self.sum_exp;
            for d in self.output.iter_mut() {
                *d *= inv_sum;
            }
        }
    }
}

// ─── Fused attention ───────────────────────────────────────────────────

/// Fused attention for a single query head against KV cache.
///
/// Computes `output = softmax(Q @ K^T / sqrt(d)) @ V` without
/// materializing the full attention matrix.
///
/// This is a drop-in replacement for [`super::attention::attention_head`]
/// with the same semantics but O(ATTENTION_BLOCK_SIZE) working memory
/// instead of O(seq_len).
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Slice of key vector references, one per sequence position.
/// - `values`: Slice of value vector references, one per sequence position.
/// - `head_dim`: Dimension of each head.
/// - `output`: Output buffer `[head_dim]`.
///
/// # Errors
/// Returns `ModelError` if dimensions are inconsistent.
pub fn fused_attention_head(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    head_dim: usize,
    output: &mut [f32],
) -> ModelResult<()> {
    if query.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query".to_string(),
            expected: vec![head_dim],
            actual: vec![query.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }
    if keys.len() != values.len() {
        return Err(ModelError::ShapeMismatch {
            name: "keys/values length".to_string(),
            expected: vec![keys.len()],
            actual: vec![values.len()],
        });
    }

    let seq_len = keys.len();
    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process KV positions in blocks
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;

        // Compute scaled dot products for this block
        let mut block_scores = Vec::with_capacity(block_len);
        let mut block_values = Vec::with_capacity(block_len);

        for t in pos..block_end {
            let score = dot_f32(query, keys[t]) * scale;
            block_scores.push(score);
            block_values.push(values[t]);
        }

        state.update(&block_scores, &block_values, head_dim);
        pos = block_end;
    }

    state.finalize();

    // Copy result to output
    output[..head_dim].copy_from_slice(&state.output[..head_dim]);

    Ok(())
}

/// Fused attention with contiguous KV buffers (matching existing API).
///
/// This variant accepts contiguous row-major key/value buffers
/// `[seq_len x head_dim]`, matching the layout used by
/// [`super::attention::attention_head`].
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Contiguous key buffer `[seq_len x head_dim]` (row-major).
/// - `values`: Contiguous value buffer `[seq_len x head_dim]` (row-major).
/// - `output`: Output buffer `[head_dim]`.
/// - `seq_len`: Number of KV positions.
/// - `head_dim`: Dimension per head.
pub fn fused_attention_head_contiguous(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
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
    if values.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "values".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![values.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }

    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process in blocks, building slice references on the fly
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;

        let mut block_scores = Vec::with_capacity(block_len);
        let mut block_values: Vec<&[f32]> = Vec::with_capacity(block_len);

        for t in pos..block_end {
            let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
            let score = dot_f32(query, k_slice) * scale;
            block_scores.push(score);
            block_values.push(&values[t * head_dim..(t + 1) * head_dim]);
        }

        state.update(&block_scores, &block_values, head_dim);
        pos = block_end;
    }

    state.finalize();
    output[..head_dim].copy_from_slice(&state.output[..head_dim]);

    Ok(())
}

/// Fused attention with contiguous KV buffers and causal masking.
///
/// Like [`fused_attention_head_contiguous`] but applies a [`CausalMask`]
/// (optionally with sliding window) to skip masked positions in the
/// online softmax accumulation.
///
/// Masked positions (where `!mask.is_allowed(query_pos, t)`) are silently
/// skipped — they contribute neither a score nor a V accumulation — giving
/// numerically identical results to applying `f32::NEG_INFINITY` before
/// softmax, but without allocating a score buffer.
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Contiguous key buffer `[seq_len x head_dim]` (row-major).
/// - `values`: Contiguous value buffer `[seq_len x head_dim]` (row-major).
/// - `output`: Output buffer `[head_dim]`.
/// - `seq_len`: Number of KV positions.
/// - `head_dim`: Dimension per head.
/// - `query_pos`: Position of the current query token in the full sequence.
/// - `mask`: Causal mask (with optional sliding window).
///
/// # Errors
/// Returns `ModelError` if dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_head_contiguous_with_mask(
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
    if values.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "values".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![values.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }

    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process in blocks; skip masked positions per block
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;

        let mut block_scores = Vec::with_capacity(block_len);
        let mut block_values: Vec<&[f32]> = Vec::with_capacity(block_len);

        for t in pos..block_end {
            if !mask.is_allowed(query_pos, t) {
                continue;
            }
            let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
            let score = dot_f32(query, k_slice) * scale;
            block_scores.push(score);
            block_values.push(&values[t * head_dim..(t + 1) * head_dim]);
        }

        if !block_scores.is_empty() {
            state.update(&block_scores, &block_values, head_dim);
        }
        pos = block_end;
    }

    state.finalize();
    output[..head_dim].copy_from_slice(&state.output[..head_dim]);

    Ok(())
}

// ─── Vectorized utilities ──────────────────────────────────────────────

/// In-place softmax with numerical stability.
///
/// Uses the max-subtraction trick to prevent overflow:
/// `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
///
/// This is a standalone utility that can be used outside fused attention.
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    // Find maximum for numerical stability
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Exponentiate and sum
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in logits.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Fused scaled dot product: `dot(q, k) * scale`.
///
/// Computes the dot product of `q` and `k`, then multiplies by `scale`.
/// This is the QK^T / sqrt(d) operation that forms attention scores.
///
/// Uses SIMD via [`dot_f32`] where available.
#[inline]
pub fn scaled_dot_product(q: &[f32], k: &[f32], scale: f32) -> f32 {
    dot_f32(q, k) * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference standard attention for comparison.
    fn reference_attention(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) {
        use super::super::attention::attention_head;
        attention_head(query, keys, values, output, seq_len, head_dim)
            .expect("reference attention should succeed");
    }

    #[test]
    fn fused_matches_standard_single_token() {
        let head_dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let values = vec![0.0, 1.0, 2.0, 3.0];

        let mut out_std = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        reference_attention(&query, &keys, &values, &mut out_std, 1, head_dim);
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, 1, head_dim)
            .expect("fused attention should succeed");

        for i in 0..head_dim {
            assert!(
                (out_std[i] - out_fused[i]).abs() < 1e-5,
                "dim {i}: std={}, fused={}",
                out_std[i],
                out_fused[i]
            );
        }
    }

    #[test]
    fn fused_matches_standard_multiple_tokens() {
        let head_dim = 8;
        let seq_len = 10;

        // Generate deterministic test data
        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        let mut out_std = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
            .expect("fused attention should succeed");

        for i in 0..head_dim {
            assert!(
                (out_std[i] - out_fused[i]).abs() < 1e-4,
                "dim {i}: std={}, fused={}",
                out_std[i],
                out_fused[i]
            );
        }
    }

    #[test]
    fn fused_matches_standard_large_seq() {
        // Sequence longer than ATTENTION_BLOCK_SIZE to test multi-block
        let head_dim = 16;
        let seq_len = 100; // > ATTENTION_BLOCK_SIZE (32)

        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.2) - 1.0).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.04 - 0.5)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.06 - 0.6)
            .collect();

        let mut out_std = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
            .expect("fused attention should succeed");

        for i in 0..head_dim {
            assert!(
                (out_std[i] - out_fused[i]).abs() < 1e-3,
                "dim {i}: std={}, fused={}",
                out_std[i],
                out_fused[i]
            );
        }
    }

    #[test]
    fn fused_with_slice_api() {
        let head_dim = 4;
        let seq_len = 3;

        let query = vec![1.0, 0.5, -0.5, 0.0];
        let k0 = vec![0.5, 0.5, 0.0, 0.0];
        let k1 = vec![0.0, 1.0, 0.0, 0.0];
        let k2 = vec![-0.5, 0.0, 1.0, 0.0];
        let v0 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.0, 1.0, 0.0, 0.0];
        let v2 = vec![0.0, 0.0, 1.0, 0.0];

        let keys_refs: Vec<&[f32]> = vec![&k0, &k1, &k2];
        let values_refs: Vec<&[f32]> = vec![&v0, &v1, &v2];

        let mut output = vec![0.0f32; head_dim];
        fused_attention_head(&query, &keys_refs, &values_refs, head_dim, &mut output)
            .expect("fused attention should succeed");

        // Build contiguous buffers for standard attention comparison
        let mut keys_flat = vec![0.0f32; seq_len * head_dim];
        let mut values_flat = vec![0.0f32; seq_len * head_dim];
        for (t, (k, v)) in keys_refs.iter().zip(values_refs.iter()).enumerate() {
            keys_flat[t * head_dim..(t + 1) * head_dim].copy_from_slice(k);
            values_flat[t * head_dim..(t + 1) * head_dim].copy_from_slice(v);
        }

        let mut out_std = vec![0.0f32; head_dim];
        reference_attention(
            &query,
            &keys_flat,
            &values_flat,
            &mut out_std,
            seq_len,
            head_dim,
        );

        for i in 0..head_dim {
            assert!(
                (out_std[i] - output[i]).abs() < 1e-4,
                "dim {i}: std={}, fused={}",
                out_std[i],
                output[i]
            );
        }
    }

    #[test]
    fn fused_empty_sequence() {
        let head_dim = 4;
        let keys: Vec<&[f32]> = vec![];
        let values: Vec<&[f32]> = vec![];
        let query = vec![1.0; head_dim];
        let mut output = vec![99.0f32; head_dim];

        fused_attention_head(&query, &keys, &values, head_dim, &mut output)
            .expect("fused attention should handle empty seq");

        for &v in &output {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn softmax_inplace_basic() {
        let mut vals = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut vals);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
    }

    #[test]
    fn softmax_inplace_single() {
        let mut vals = vec![5.0];
        softmax_inplace(&mut vals);
        assert!((vals[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_inplace_empty() {
        let mut vals: Vec<f32> = vec![];
        softmax_inplace(&mut vals); // Should not panic
    }

    #[test]
    fn scaled_dot_product_basic() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![4.0, 3.0, 2.0, 1.0];
        let scale = 0.5;
        let result = scaled_dot_product(&q, &k, scale);
        // dot = 4+6+6+4 = 20, scaled = 10.0
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn scaled_dot_product_non_multiple_of_4() {
        let q = vec![1.0, 2.0, 3.0];
        let k = vec![4.0, 5.0, 6.0];
        let scale = 1.0;
        let result = scaled_dot_product(&q, &k, scale);
        // dot = 4+10+18 = 32
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn fused_validation_errors() {
        let head_dim = 4;
        let query = vec![1.0; 2]; // Too short
        let keys: Vec<&[f32]> = vec![];
        let values: Vec<&[f32]> = vec![];
        let mut output = vec![0.0f32; head_dim];

        let result = fused_attention_head(&query, &keys, &values, head_dim, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn fused_contiguous_validation_errors() {
        let head_dim = 4;
        let query = vec![1.0; head_dim];
        let keys = vec![1.0; 4]; // seq_len=1 matches
        let values = vec![1.0; 2]; // Too short for seq_len=1
        let mut output = vec![0.0f32; head_dim];

        let result =
            fused_attention_head_contiguous(&query, &keys, &values, &mut output, 1, head_dim);
        assert!(result.is_err());
    }

    #[test]
    fn fused_head_dim_128() {
        // Realistic head_dim matching Qwen3-8B
        let head_dim = 128;
        let seq_len = 50;

        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.03) - 2.0).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 31) as f32 * 0.02 - 0.3)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 13 + 7) % 23) as f32 * 0.04 - 0.5)
            .collect();

        let mut out_std = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
            .expect("fused attention should succeed");

        let max_diff = out_std
            .iter()
            .zip(out_fused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "max difference between standard and fused: {max_diff}"
        );
    }

    // ── P11.1 Step 6: New tests ────────────────────────────────────────────

    /// Test masked fused attention matches naive masked attention within 1e-3.
    #[test]
    fn fused_masked_matches_naive_masked() {
        use super::super::attention::attention_head_with_mask;
        let head_dim = 8;
        let seq_len = 12;
        let query_pos = 6; // causal: can attend to 0..=6

        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.15) - 0.5).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 5 + 2) % 17) as f32 * 0.05 - 0.4)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.08 - 0.5)
            .collect();

        let mask = CausalMask::new(32);

        let mut out_naive = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        attention_head_with_mask(
            &query,
            &keys,
            &values,
            &mut out_naive,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("naive masked attention should succeed");

        fused_attention_head_contiguous_with_mask(
            &query,
            &keys,
            &values,
            &mut out_fused,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("fused masked attention should succeed");

        let max_diff = out_naive
            .iter()
            .zip(out_fused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "masked: max diff = {max_diff}; naive={out_naive:?}, fused={out_fused:?}"
        );
    }

    /// Long-context test at S=4096 exercising many block boundaries.
    #[test]
    fn fused_long_context_4096() {
        let head_dim = 16;
        let seq_len = 4096;

        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1) - 0.8).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 3 + 1) % 29) as f32 * 0.02 - 0.3)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 5) % 17) as f32 * 0.03 - 0.25)
            .collect();

        let mut out_std = vec![0.0f32; head_dim];
        let mut out_fused = vec![0.0f32; head_dim];

        reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
            .expect("fused long-context attention should succeed");

        let max_diff = out_std
            .iter()
            .zip(out_fused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "long-context S=4096: max diff = {max_diff}"
        );
    }

    /// Multi-head test: 4 Q heads, 2 KV heads (GQA 2:1), head_dim=32.
    #[test]
    fn fused_multi_head_gqa() {
        let head_dim = 32;
        let seq_len = 40;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_heads_per_kv = num_q_heads / num_kv_heads;

        let query_all: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.05) - 1.0)
            .collect();
        let keys: Vec<Vec<f32>> = (0..num_kv_heads)
            .map(|kv| {
                (0..seq_len * head_dim)
                    .map(|i| ((i * (kv + 3) + 1) % 23) as f32 * 0.04 - 0.45)
                    .collect()
            })
            .collect();
        let values: Vec<Vec<f32>> = (0..num_kv_heads)
            .map(|kv| {
                (0..seq_len * head_dim)
                    .map(|i| ((i * (kv + 5) + 2) % 19) as f32 * 0.06 - 0.55)
                    .collect()
            })
            .collect();

        let mut out_fused = vec![0.0f32; num_q_heads * head_dim];
        let mut out_ref = vec![0.0f32; num_q_heads * head_dim];

        // Reference: naive per-head
        for q_head in 0..num_q_heads {
            let kv_head = q_head / q_heads_per_kv;
            let q_start = q_head * head_dim;
            let mut head_out = vec![0.0f32; head_dim];
            reference_attention(
                &query_all[q_start..q_start + head_dim],
                &keys[kv_head],
                &values[kv_head],
                &mut head_out,
                seq_len,
                head_dim,
            );
            out_ref[q_start..q_start + head_dim].copy_from_slice(&head_out);
        }

        // Fused: per-head
        for q_head in 0..num_q_heads {
            let kv_head = q_head / q_heads_per_kv;
            let q_start = q_head * head_dim;
            fused_attention_head_contiguous(
                &query_all[q_start..q_start + head_dim],
                &keys[kv_head],
                &values[kv_head],
                &mut out_fused[q_start..q_start + head_dim],
                seq_len,
                head_dim,
            )
            .expect("fused multi-head attention should succeed");
        }

        let max_diff = out_ref
            .iter()
            .zip(out_fused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "multi-head GQA: max diff = {max_diff}");
    }

    // ── P11.2: SIMD dot correctness ────────────────────────────────────────

    /// Verify SIMD dot_f32 matches scalar for various lengths.
    #[test]
    fn dot_f32_matches_scalar() {
        let test_lengths = [128usize, 127, 513, 1024];

        for &n in &test_lengths {
            // Deterministic pseudo-random vectors
            let a: Vec<f32> = (0..n)
                .map(|i| ((i * 7 + 3) % 31) as f32 * 0.1 - 1.5)
                .collect();
            let b: Vec<f32> = (0..n)
                .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.2)
                .collect();

            let scalar_val = dot_f32_scalar(&a, &b);
            let simd_val = dot_f32(&a, &b);

            let denom = scalar_val.abs().max(1.0);
            let rel_err = (simd_val - scalar_val).abs() / denom;
            assert!(
                rel_err < 1e-5,
                "n={n}: scalar={scalar_val}, simd={simd_val}, rel_err={rel_err}"
            );
        }
    }
}
