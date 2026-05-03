//! Cross-attention: decoder queries attend to encoder context.
//!
//! Standard self-attention: Q, K, V all come from the same sequence.
//! Cross-attention:         Q from decoder, K/V from encoder.
//!
//! This is the attention pattern used in encoder-decoder architectures such as
//! T5 and BART.  The decoder generates queries against its own hidden state,
//! while keys and values come from the fixed encoder output, allowing every
//! decoder position to attend over the full source sequence.
//!
//! # Layout convention
//!
//! All hidden-state buffers use a **flat row-major** layout:
//!
//! ```text
//! hidden[pos * num_heads * head_dim + head * head_dim + d]
//!      = element d of head `head` at position `pos`
//! ```
//!
//! Attention weight buffers use:
//!
//! ```text
//! weights[dec_pos * enc_seq + enc_pos]
//!      = weight that decoder position `dec_pos` assigns to encoder position `enc_pos`
//! ```

use thiserror::Error;

// ─── Error types ─────────────────────────────────────────────────────────────

/// Errors returned by cross-attention functions.
#[derive(Debug, Error)]
pub enum CrossAttnError {
    /// The `decoder_hidden` slice has the wrong number of elements.
    #[error("decoder_hidden length {got} != decoder_seq_len * num_heads * head_dim = {expected}")]
    DecoderDimMismatch { expected: usize, got: usize },

    /// The `encoder_hidden` slice has the wrong number of elements.
    #[error("encoder_hidden length {got} != encoder_seq_len * num_heads * head_dim = {expected}")]
    EncoderDimMismatch { expected: usize, got: usize },

    /// The encoder mask has the wrong length.
    #[error("encoder mask length {got} != encoder_seq_len {expected}")]
    MaskLengthMismatch { expected: usize, got: usize },

    /// `head_dim` was supplied as zero.
    #[error("head_dim must be > 0")]
    InvalidHeadDim,

    /// `num_heads` was supplied as zero.
    #[error("num_heads must be > 0")]
    InvalidNumHeads,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a cross-attention layer.
///
/// The attention scale is pre-computed as `1 / sqrt(head_dim)` so that it does
/// not have to be recomputed on every call.
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Number of parallel attention heads.
    pub num_heads: usize,
    /// Dimension of each head's key/query/value vectors.
    pub head_dim: usize,
    /// Dropout rate applied to attention weights during training.
    /// Set to `0.0` at inference time.
    pub dropout_rate: f32,
    /// Scaling factor: `1.0 / sqrt(head_dim)`.
    pub scale: f32,
}

impl CrossAttentionConfig {
    /// Create a new configuration, computing the scale automatically.
    ///
    /// `dropout_rate` defaults to `0.0` (no dropout); override via the public
    /// field after construction if required.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };
        Self {
            num_heads,
            head_dim,
            dropout_rate: 0.0,
            scale,
        }
    }

    /// Total hidden dimension: `num_heads * head_dim`.
    pub fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

// ─── Core attention primitives ────────────────────────────────────────────────

/// Compute attention weights for a single query against a set of keys.
///
/// Returns a `[decoder_seq * encoder_seq]` flat buffer.  Each row (stride
/// `encoder_seq`) is a probability distribution over encoder positions (sums to
/// 1.0) after softmax normalisation.
///
/// # Arguments
///
/// - `queries`      — `[decoder_seq * head_dim]`
/// - `keys`         — `[encoder_seq * head_dim]`
/// - `decoder_seq`  — number of decoder positions
/// - `encoder_seq`  — number of encoder positions
/// - `head_dim`     — per-head dimension
/// - `scale`        — pre-computed `1/sqrt(head_dim)`
pub fn compute_attention_weights(
    queries: &[f32],
    keys: &[f32],
    decoder_seq: usize,
    encoder_seq: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, CrossAttnError> {
    if head_dim == 0 {
        return Err(CrossAttnError::InvalidHeadDim);
    }

    let mut weights = vec![0.0f32; decoder_seq * encoder_seq];

    for dq in 0..decoder_seq {
        let q_slice = &queries[dq * head_dim..(dq + 1) * head_dim];

        // Compute raw scores and track maximum for numerical stability.
        let row_start = dq * encoder_seq;
        let mut max_score = f32::NEG_INFINITY;

        for ek in 0..encoder_seq {
            let k_slice = &keys[ek * head_dim..(ek + 1) * head_dim];
            let score = dot_scaled(q_slice, k_slice, scale);
            weights[row_start + ek] = score;
            if score > max_score {
                max_score = score;
            }
        }

        // Softmax with numerical stability (subtract max before exp).
        let mut sum_exp = 0.0f32;
        for ek in 0..encoder_seq {
            let e = (weights[row_start + ek] - max_score).exp();
            weights[row_start + ek] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            for ek in 0..encoder_seq {
                weights[row_start + ek] *= inv;
            }
        }
    }

    Ok(weights)
}

/// Single-head cross-attention (building block for multi-head).
///
/// Computes `output = softmax(Q K^T * scale) V` for one attention head.
///
/// # Arguments
///
/// - `queries`      — `[decoder_seq * head_dim]`
/// - `keys`         — `[encoder_seq * head_dim]`
/// - `values`       — `[encoder_seq * head_dim]`
/// - `decoder_seq`  — number of decoder positions
/// - `encoder_seq`  — number of encoder positions
/// - `head_dim`     — per-head dimension
/// - `scale`        — pre-computed `1/sqrt(head_dim)`
/// - `mask`         — optional `[encoder_seq]` bool mask; `false` = ignore position
///
/// Returns `[decoder_seq * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn single_head_cross_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    decoder_seq: usize,
    encoder_seq: usize,
    head_dim: usize,
    scale: f32,
    mask: Option<&[bool]>,
) -> Result<Vec<f32>, CrossAttnError> {
    if head_dim == 0 {
        return Err(CrossAttnError::InvalidHeadDim);
    }
    if let Some(m) = mask {
        if m.len() != encoder_seq {
            return Err(CrossAttnError::MaskLengthMismatch {
                expected: encoder_seq,
                got: m.len(),
            });
        }
    }

    let mut output = vec![0.0f32; decoder_seq * head_dim];

    for dq in 0..decoder_seq {
        let q_slice = &queries[dq * head_dim..(dq + 1) * head_dim];

        // ── Compute raw attention scores ──────────────────────────────────
        let mut scores = vec![0.0f32; encoder_seq];
        let mut max_score = f32::NEG_INFINITY;

        for ek in 0..encoder_seq {
            // Apply mask: ignored positions get −∞ so softmax gives them 0.
            let score = if mask.is_none_or(|m| m[ek]) {
                dot_scaled(q_slice, &keys[ek * head_dim..(ek + 1) * head_dim], scale)
            } else {
                f32::NEG_INFINITY
            };
            scores[ek] = score;
            if score > max_score {
                max_score = score;
            }
        }

        // ── Softmax ───────────────────────────────────────────────────────
        // Guard against all-masked case (max = -∞ → treat as uniform zero).
        if max_score == f32::NEG_INFINITY {
            max_score = 0.0;
        }

        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            let e = (*s - max_score).exp();
            *s = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv;
            }
        }

        // ── Weighted sum of values ────────────────────────────────────────
        let out_slice = &mut output[dq * head_dim..(dq + 1) * head_dim];
        for ek in 0..encoder_seq {
            let w = scores[ek];
            let v_slice = &values[ek * head_dim..(ek + 1) * head_dim];
            for d in 0..head_dim {
                out_slice[d] += w * v_slice[d];
            }
        }
    }

    Ok(output)
}

// ─── Multi-head cross-attention ───────────────────────────────────────────────

/// Compute multi-head cross-attention.
///
/// Decoder queries attend to encoder keys and values.  Each head operates
/// independently; results are concatenated along the head dimension.
///
/// # Arguments
///
/// - `decoder_hidden` — `[decoder_seq_len * num_heads * head_dim]`
/// - `encoder_hidden` — `[encoder_seq_len * num_heads * head_dim]`
/// - `decoder_seq_len` — number of decoder positions
/// - `encoder_seq_len` — number of encoder positions
/// - `config`          — attention configuration
/// - `encoder_mask`    — optional `[encoder_seq_len]` bool mask
///
/// Returns `[decoder_seq_len * num_heads * head_dim]`.
pub fn cross_attention_forward(
    decoder_hidden: &[f32],
    encoder_hidden: &[f32],
    decoder_seq_len: usize,
    encoder_seq_len: usize,
    config: &CrossAttentionConfig,
    encoder_mask: Option<&[bool]>,
) -> Result<Vec<f32>, CrossAttnError> {
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    if num_heads == 0 {
        return Err(CrossAttnError::InvalidNumHeads);
    }
    if head_dim == 0 {
        return Err(CrossAttnError::InvalidHeadDim);
    }

    let dec_expected = decoder_seq_len * num_heads * head_dim;
    if decoder_hidden.len() != dec_expected {
        return Err(CrossAttnError::DecoderDimMismatch {
            expected: dec_expected,
            got: decoder_hidden.len(),
        });
    }

    let enc_expected = encoder_seq_len * num_heads * head_dim;
    if encoder_hidden.len() != enc_expected {
        return Err(CrossAttnError::EncoderDimMismatch {
            expected: enc_expected,
            got: encoder_hidden.len(),
        });
    }

    if let Some(m) = encoder_mask {
        if m.len() != encoder_seq_len {
            return Err(CrossAttnError::MaskLengthMismatch {
                expected: encoder_seq_len,
                got: m.len(),
            });
        }
    }

    let mut output = vec![0.0f32; decoder_seq_len * num_heads * head_dim];

    // Process each head independently.
    for h in 0..num_heads {
        // Extract contiguous head slices: [seq_len * head_dim]
        let dec_queries = extract_head(decoder_hidden, decoder_seq_len, num_heads, head_dim, h);
        let enc_keys = extract_head(encoder_hidden, encoder_seq_len, num_heads, head_dim, h);
        let enc_values = enc_keys.clone(); // same layout; K/V from encoder

        // For cross-attention K and V both come from encoder_hidden.
        // In a full model the projection matrices Q/K/V would differ,
        // but here we use encoder_hidden directly for both K and V.
        let head_out = single_head_cross_attention(
            &dec_queries,
            &enc_keys,
            &enc_values,
            decoder_seq_len,
            encoder_seq_len,
            head_dim,
            config.scale,
            encoder_mask,
        )?;

        // Write back into the interleaved output buffer.
        scatter_head(
            &mut output,
            &head_out,
            decoder_seq_len,
            num_heads,
            head_dim,
            h,
        );
    }

    Ok(output)
}

/// Causal cross-attention: each decoder position `dq` only attends to encoder
/// positions `0..=dq` (monotonic alignment).
///
/// Useful for models that enforce a strict left-to-right alignment between
/// source and target (e.g., monotonic attention in speech synthesis).
///
/// Returns `[decoder_seq_len * num_heads * head_dim]`.
pub fn causal_cross_attention(
    decoder_hidden: &[f32],
    encoder_hidden: &[f32],
    decoder_seq_len: usize,
    encoder_seq_len: usize,
    config: &CrossAttentionConfig,
) -> Result<Vec<f32>, CrossAttnError> {
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    if num_heads == 0 {
        return Err(CrossAttnError::InvalidNumHeads);
    }
    if head_dim == 0 {
        return Err(CrossAttnError::InvalidHeadDim);
    }

    let dec_expected = decoder_seq_len * num_heads * head_dim;
    if decoder_hidden.len() != dec_expected {
        return Err(CrossAttnError::DecoderDimMismatch {
            expected: dec_expected,
            got: decoder_hidden.len(),
        });
    }

    let enc_expected = encoder_seq_len * num_heads * head_dim;
    if encoder_hidden.len() != enc_expected {
        return Err(CrossAttnError::EncoderDimMismatch {
            expected: enc_expected,
            got: encoder_hidden.len(),
        });
    }

    let mut output = vec![0.0f32; decoder_seq_len * num_heads * head_dim];

    for h in 0..num_heads {
        let dec_queries = extract_head(decoder_hidden, decoder_seq_len, num_heads, head_dim, h);
        let enc_keys = extract_head(encoder_hidden, encoder_seq_len, num_heads, head_dim, h);

        // For each decoder position, construct a causal mask that allows
        // only encoder positions <= dq (capped at encoder_seq_len).
        let mut head_out = vec![0.0f32; decoder_seq_len * head_dim];

        for dq in 0..decoder_seq_len {
            // Causal limit: only attend to encoder positions 0..=dq
            let allowed = (dq + 1).min(encoder_seq_len);

            let q_slice = &dec_queries[dq * head_dim..(dq + 1) * head_dim];

            let mut scores = vec![0.0f32; encoder_seq_len];
            let mut max_score = f32::NEG_INFINITY;

            for ek in 0..encoder_seq_len {
                let score = if ek < allowed {
                    dot_scaled(
                        q_slice,
                        &enc_keys[ek * head_dim..(ek + 1) * head_dim],
                        config.scale,
                    )
                } else {
                    f32::NEG_INFINITY
                };
                scores[ek] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            if max_score == f32::NEG_INFINITY {
                max_score = 0.0;
            }

            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                let e = (*s - max_score).exp();
                *s = e;
                sum_exp += e;
            }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
            }

            let out_slice = &mut head_out[dq * head_dim..(dq + 1) * head_dim];
            for ek in 0..encoder_seq_len {
                let w = scores[ek];
                let v_slice = &enc_keys[ek * head_dim..(ek + 1) * head_dim];
                for d in 0..head_dim {
                    out_slice[d] += w * v_slice[d];
                }
            }
        }

        scatter_head(
            &mut output,
            &head_out,
            decoder_seq_len,
            num_heads,
            head_dim,
            h,
        );
    }

    Ok(output)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Extract a single head's data into a contiguous `[seq_len * head_dim]` buffer.
///
/// Input layout: `hidden[pos * num_heads * head_dim + head * head_dim + d]`
fn extract_head(
    hidden: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    head: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * head_dim];
    for pos in 0..seq_len {
        let src_start = pos * num_heads * head_dim + head * head_dim;
        let dst_start = pos * head_dim;
        out[dst_start..dst_start + head_dim]
            .copy_from_slice(&hidden[src_start..src_start + head_dim]);
    }
    out
}

/// Scatter a single head's `[seq_len * head_dim]` result back into the
/// interleaved `[seq_len * num_heads * head_dim]` output buffer.
fn scatter_head(
    output: &mut [f32],
    head_data: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    head: usize,
) {
    for pos in 0..seq_len {
        let dst_start = pos * num_heads * head_dim + head * head_dim;
        let src_start = pos * head_dim;
        output[dst_start..dst_start + head_dim]
            .copy_from_slice(&head_data[src_start..src_start + head_dim]);
    }
}

/// Scaled dot product of two equal-length vectors.
///
/// Computes `dot(a, b) * scale` in a single pass.
#[inline]
fn dot_scaled(a: &[f32], b: &[f32], scale: f32) -> f32 {
    let len = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..len {
        acc += a[i] * b[i];
    }
    acc * scale
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn make_hidden(seq: usize, num_heads: usize, head_dim: usize, fill: f32) -> Vec<f32> {
        vec![fill; seq * num_heads * head_dim]
    }

    #[test]
    fn cross_attn_config_hidden_dim() {
        let cfg = CrossAttentionConfig::new(4, 8);
        assert_eq!(
            cfg.hidden_dim(),
            32,
            "hidden_dim should be num_heads * head_dim"
        );
    }

    #[test]
    fn cross_attention_output_shape() {
        let num_heads = 2;
        let head_dim = 4;
        let dec_seq = 3;
        let enc_seq = 5;
        let cfg = CrossAttentionConfig::new(num_heads, head_dim);
        let dec = make_hidden(dec_seq, num_heads, head_dim, 0.1);
        let enc = make_hidden(enc_seq, num_heads, head_dim, 0.2);
        let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, None)
            .expect("cross_attention_forward should succeed");
        assert_eq!(out.len(), dec_seq * num_heads * head_dim);
    }

    #[test]
    fn cross_attention_identity_query() {
        // With a single head and single encoder/decoder position, output == value.
        let head_dim = 4;
        let cfg = CrossAttentionConfig::new(1, head_dim);
        // query = encoder key → attention weight = 1.0 → output ≈ value
        let dec = vec![1.0f32, 0.0, 0.0, 0.0]; // [dec_seq=1, heads=1, head_dim=4]
        let enc = vec![1.0f32, 0.0, 0.0, 0.0]; // [enc_seq=1, heads=1, head_dim=4]
        let out = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None).expect("should succeed");
        // With a single encoder position, softmax gives weight 1.0, so output = value = enc.
        for i in 0..head_dim {
            assert!(
                (out[i] - enc[i]).abs() < EPS,
                "output[{i}] = {} expected {}",
                out[i],
                enc[i]
            );
        }
    }

    #[test]
    fn cross_attention_with_mask() {
        let num_heads = 1;
        let head_dim = 2;
        let dec_seq = 1;
        let enc_seq = 3;
        let cfg = CrossAttentionConfig::new(num_heads, head_dim);

        // Encoder: three positions, only position 0 is unmasked.
        let dec = vec![1.0f32, 0.0];
        // Encoder position 0: value [2.0, 3.0]
        // Encoder position 1: value [5.0, 5.0] — masked (should not contribute)
        // Encoder position 2: value [9.0, 9.0] — masked
        let enc = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mask = vec![true, false, false];

        let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, Some(&mask))
            .expect("masked cross attention should succeed");

        // Output should be entirely from position 0 of encoder.
        // Weight on positions 1,2 is 0.0 (masked out via -inf).
        // Position 0's key = [1.0, 0.0], value = [1.0, 0.0]
        // With only one valid position, softmax gives 1.0, so output = enc[0..head_dim].
        assert!(
            (out[0] - 1.0).abs() < EPS,
            "output[0] = {} expected 1.0",
            out[0]
        );
        assert!(
            (out[1] - 0.0).abs() < EPS,
            "output[1] = {} expected 0.0",
            out[1]
        );
    }

    #[test]
    fn cross_attention_uniform_encoder() {
        // Uniform attention over identical encoder positions → output = that value.
        let num_heads = 1;
        let head_dim = 2;
        let dec_seq = 1;
        let enc_seq = 4;
        let cfg = CrossAttentionConfig::new(num_heads, head_dim);

        // Query = zero vector → all dot products are 0 → uniform softmax.
        let dec = vec![0.0f32; dec_seq * num_heads * head_dim];
        // All encoder positions have the same value [1.0, 2.0].
        let enc: Vec<f32> = (0..enc_seq * num_heads * head_dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { 2.0 })
            .collect();

        let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, None)
            .expect("uniform encoder cross attention should succeed");

        // Weighted avg of identical values = those values.
        assert!((out[0] - 1.0).abs() < EPS, "expected 1.0 got {}", out[0]);
        assert!((out[1] - 2.0).abs() < EPS, "expected 2.0 got {}", out[1]);
    }

    #[test]
    fn single_head_output_shape() {
        let dec_seq = 3;
        let enc_seq = 5;
        let head_dim = 4;
        let q = vec![0.1f32; dec_seq * head_dim];
        let k = vec![0.2f32; enc_seq * head_dim];
        let v = vec![0.3f32; enc_seq * head_dim];
        let out = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
            .expect("single head should succeed");
        assert_eq!(out.len(), dec_seq * head_dim);
    }

    #[test]
    fn single_head_deterministic() {
        let dec_seq = 2;
        let enc_seq = 3;
        let head_dim = 4;
        let q: Vec<f32> = (0..dec_seq * head_dim).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..enc_seq * head_dim).map(|i| i as f32 * 0.05).collect();
        let v: Vec<f32> = (0..enc_seq * head_dim).map(|i| (i as f32).sin()).collect();
        let out1 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
            .expect("first call should succeed");
        let out2 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
            .expect("second call should succeed");
        assert_eq!(out1, out2, "single_head must be deterministic");
    }

    #[test]
    fn single_head_scale_effect() {
        let dec_seq = 1;
        let enc_seq = 2;
        let head_dim = 2;
        let q = vec![1.0f32, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = vec![1.0f32, 0.0, 0.0, 1.0];
        let out1 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 1.0, None)
            .expect("scale=1.0 should succeed");
        let out2 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.01, None)
            .expect("scale=0.01 should succeed");
        assert_ne!(out1, out2, "different scale must produce different output");
    }

    #[test]
    fn causal_cross_attention_shape() {
        let num_heads = 2;
        let head_dim = 4;
        let dec_seq = 3;
        let enc_seq = 5;
        let cfg = CrossAttentionConfig::new(num_heads, head_dim);
        let dec = make_hidden(dec_seq, num_heads, head_dim, 0.1);
        let enc = make_hidden(enc_seq, num_heads, head_dim, 0.2);
        let out = causal_cross_attention(&dec, &enc, dec_seq, enc_seq, &cfg)
            .expect("causal cross attention should succeed");
        assert_eq!(out.len(), dec_seq * num_heads * head_dim);
    }

    #[test]
    fn attention_weights_shape() {
        let dec_seq = 3;
        let enc_seq = 5;
        let head_dim = 4;
        let q = vec![0.1f32; dec_seq * head_dim];
        let k = vec![0.2f32; enc_seq * head_dim];
        let weights = compute_attention_weights(&q, &k, dec_seq, enc_seq, head_dim, 0.5)
            .expect("compute_attention_weights should succeed");
        assert_eq!(weights.len(), dec_seq * enc_seq);
    }

    #[test]
    fn attention_weights_sum_to_one() {
        let dec_seq = 4;
        let enc_seq = 6;
        let head_dim = 8;
        let q: Vec<f32> = (0..dec_seq * head_dim)
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();
        let k: Vec<f32> = (0..enc_seq * head_dim).map(|i| (i as f32) * 0.05).collect();
        let weights = compute_attention_weights(&q, &k, dec_seq, enc_seq, head_dim, 0.5)
            .expect("compute_attention_weights should succeed");

        for dq in 0..dec_seq {
            let row_sum: f32 = weights[dq * enc_seq..(dq + 1) * enc_seq].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {dq} sums to {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn cross_attn_invalid_head_dim_error() {
        let cfg = CrossAttentionConfig::new(2, 0);
        let dec = vec![0.0f32; 0]; // empty
        let enc = vec![0.0f32; 0];
        let result = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None);
        assert!(
            matches!(result, Err(CrossAttnError::InvalidHeadDim)),
            "head_dim=0 should return InvalidHeadDim"
        );
    }

    #[test]
    fn cross_attn_dim_mismatch_error() {
        let cfg = CrossAttentionConfig::new(2, 4);
        // Provide wrong-sized decoder_hidden (too short).
        let dec = vec![0.0f32; 3]; // should be dec_seq * 2 * 4 = 1 * 2 * 4 = 8
        let enc = vec![0.0f32; 8];
        let result = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None);
        assert!(
            matches!(result, Err(CrossAttnError::DecoderDimMismatch { .. })),
            "wrong decoder_hidden size should return DecoderDimMismatch"
        );
    }
}
