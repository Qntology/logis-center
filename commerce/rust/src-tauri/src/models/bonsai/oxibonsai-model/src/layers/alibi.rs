//! ALiBi (Attention with Linear Biases) positional encoding.
//!
//! ALiBi adds a fixed, non-learned linear bias to attention scores based on the
//! query-key distance. Unlike RoPE, it generalises to longer sequences than seen
//! during training without any learned parameters.
//!
//! Reference: Press et al. "Train Short, Test Long: Attention with Linear Biases
//! Enables Input Length Extrapolation" (ICLR 2022).
//!
//! ## Slope schedule
//!
//! For `n` heads, the slopes follow a geometric sequence:
//! ```text
//! start = 2^(-8/n)
//! slopes[i] = start^(i+1)   for i in 0..n
//! ```
//! For non-power-of-2 head counts an "extrapolated" variant fills in missing
//! slopes by interpolating between adjacent powers of 2.
//!
//! ## Bias formula
//!
//! For causal attention, the bias added to score `(q, k)` for head `h` is:
//! ```text
//! bias[h][q][k] = -slope_h * (q_pos - k_pos)   for k_pos <= q_pos
//! ```
//! This is always ≤ 0 and equals 0 only when `k_pos == q_pos`.

use crate::layers::attention::{dot, softmax};

// ─── AliBiSlopes ─────────────────────────────────────────────────────────────

/// Pre-computed ALiBi slope for each attention head.
///
/// Slopes follow a geometric sequence derived from the number of heads so
/// that each head focuses on a different recency scale.
#[derive(Debug, Clone)]
pub struct AliBiSlopes {
    slopes: Vec<f32>,
    num_heads: usize,
}

impl AliBiSlopes {
    /// Compute slopes for exactly `num_heads` heads (power-of-2 preferred).
    ///
    /// Formula: `slopes[i] = (2^(-8/n))^(i+1)` for `i` in `0..n`, where
    /// `n = num_heads`.
    pub fn new(num_heads: usize) -> Self {
        assert!(num_heads > 0, "num_heads must be > 0");
        let start = 2.0_f32.powf(-8.0 / num_heads as f32);
        let slopes: Vec<f32> = (1..=num_heads).map(|i| start.powi(i as i32)).collect();
        Self { slopes, num_heads }
    }

    /// Extrapolated variant for non-power-of-2 head counts.
    ///
    /// The paper recommends computing slopes for the nearest smaller and
    /// larger powers of 2, then interleaving them to fill `num_heads` slots.
    /// This gives better extrapolation behaviour than the basic formula when
    /// `num_heads` is not a power of 2.
    pub fn new_extrapolated(num_heads: usize) -> Self {
        assert!(num_heads > 0, "num_heads must be > 0");

        // Find nearest power of 2 >= num_heads
        let mut p = 1usize;
        while p < num_heads {
            p <<= 1;
        }

        // Slopes for the full power-of-2 count
        let start_p = 2.0_f32.powf(-8.0 / p as f32);
        let full_slopes: Vec<f32> = (1..=p).map(|i| start_p.powi(i as i32)).collect();

        if p == num_heads {
            // Exact power of 2 — no extrapolation needed
            return Self {
                slopes: full_slopes,
                num_heads,
            };
        }

        // Half count (nearest smaller power of 2)
        let half = p / 2;
        let start_half = 2.0_f32.powf(-8.0 / half as f32);
        let half_slopes: Vec<f32> = (1..=half).map(|i| start_half.powi(i as i32)).collect();

        // Interleave half_slopes and full_slopes (even positions from half,
        // odd positions from full) and take the first num_heads entries.
        // This mirrors the implementation in the original paper repo.
        let mut slopes: Vec<f32> = Vec::with_capacity(num_heads);
        let mut hi = half_slopes.iter();
        let mut fi = full_slopes.iter();
        for idx in 0..num_heads {
            if idx % 2 == 0 {
                // Take from half-count slopes when available, otherwise full
                if let Some(&s) = hi.next() {
                    slopes.push(s);
                } else if let Some(&s) = fi.next() {
                    slopes.push(s);
                }
            } else {
                // Take from full slopes when available, otherwise half
                if let Some(&s) = fi.next() {
                    slopes.push(s);
                } else if let Some(&s) = hi.next() {
                    slopes.push(s);
                }
            }
        }

        // Pad if needed (unlikely but safe)
        while slopes.len() < num_heads {
            let last = *slopes.last().expect("at least one slope computed");
            slopes.push(last * 0.5);
        }

        Self { slopes, num_heads }
    }

    /// Return all slopes as a slice.
    #[inline]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Return the slope for head `head`.
    ///
    /// # Panics
    /// Panics if `head >= num_heads`.
    #[inline]
    pub fn get(&self, head: usize) -> f32 {
        self.slopes[head]
    }

    /// Number of heads these slopes were computed for.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ─── AliBiBias ───────────────────────────────────────────────────────────────

/// Computes and applies ALiBi bias matrices for a set of attention heads.
///
/// The bias for head `h`, query position `q_pos`, and key position `k` is:
/// ```text
/// bias = -slope_h * (q_pos - k)   for k in 0..kv_len
/// ```
/// All values are ≤ 0 (zero at `k == q_pos`), penalising earlier tokens
/// proportionally to their distance from the current query.
pub struct AliBiBias {
    /// Pre-computed slopes, one per head.
    pub slopes: AliBiSlopes,
}

impl AliBiBias {
    /// Create a new `AliBiBias` for `num_heads` heads using the standard
    /// slope schedule.
    pub fn new(num_heads: usize) -> Self {
        Self {
            slopes: AliBiSlopes::new(num_heads),
        }
    }

    /// Compute the ALiBi bias vector for a single head at query position
    /// `q_pos` over `kv_len` key positions.
    ///
    /// Returns a `Vec<f32>` of length `kv_len` where:
    /// ```text
    /// result[k] = -slope * (q_pos - k)
    /// ```
    pub fn bias_for_head(&self, head: usize, q_pos: usize, kv_len: usize) -> Vec<f32> {
        let slope = self.slopes.get(head);
        (0..kv_len)
            .map(|k| {
                let distance = q_pos as f32 - k as f32;
                -slope * distance
            })
            .collect()
    }

    /// Compute bias vectors for all heads at query position `q_pos`.
    ///
    /// Returns shape `[num_heads][kv_len]`.
    pub fn biases_all_heads(&self, q_pos: usize, kv_len: usize) -> Vec<Vec<f32>> {
        (0..self.slopes.num_heads())
            .map(|head| self.bias_for_head(head, q_pos, kv_len))
            .collect()
    }

    /// Add ALiBi biases to attention scores in-place.
    ///
    /// `scores` must have shape `[num_heads][kv_len]` where `kv_len =
    /// scores[0].len()`.
    pub fn apply(&self, scores: &mut [Vec<f32>], q_pos: usize) {
        let kv_len = scores.first().map(|s| s.len()).unwrap_or(0);
        let biases = self.biases_all_heads(q_pos, kv_len);
        for (head_scores, head_biases) in scores.iter_mut().zip(biases.iter()) {
            for (s, b) in head_scores.iter_mut().zip(head_biases.iter()) {
                *s += b;
            }
        }
    }

    /// Compute biases for an entire sequence of query positions.
    ///
    /// - `q_len`: number of query positions (starting at `q_offset`).
    /// - `kv_len`: number of key/value positions.
    /// - `q_offset`: absolute position of the first query token.
    ///
    /// Returns shape `[q_len][num_heads][kv_len]`.
    pub fn biases_for_sequence(
        &self,
        q_len: usize,
        kv_len: usize,
        q_offset: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        (0..q_len)
            .map(|qi| self.biases_all_heads(q_offset + qi, kv_len))
            .collect()
    }
}

// ─── AliBiConfig ─────────────────────────────────────────────────────────────

/// Configuration for ALiBi-enhanced attention.
#[derive(Debug, Clone)]
pub struct AliBiConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Whether to use the extrapolated slope schedule for non-power-of-2
    /// head counts.
    pub use_extrapolated_slopes: bool,
    /// Whether to apply a causal mask (key positions after the query
    /// position receive `-∞` attention score). Almost always `true`.
    pub causal: bool,
}

impl Default for AliBiConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            use_extrapolated_slopes: false,
            causal: true,
        }
    }
}

// ─── attention_with_alibi ────────────────────────────────────────────────────

/// Compute grouped-query attention augmented with ALiBi positional biases.
///
/// This is a single-token (decode-step) attention kernel. It reads one
/// query vector per head, attends over all `kv_len` key/value positions,
/// adds the ALiBi bias, applies the causal mask when `config.causal`, and
/// returns the concatenated head outputs.
///
/// # Arguments
///
/// - `query`:       flattened query tensor `[num_heads * head_dim]`
/// - `keys`:        flattened key cache `[kv_len * num_kv_heads * head_dim]`
///   (tokens in outermost dimension)
/// - `values`:      flattened value cache `[kv_len * num_kv_heads * head_dim]`
/// - `config`:      ALiBi attention configuration
/// - `head_dim`:    dimension of each attention head
/// - `num_kv_heads`: number of key-value heads (GQA)
/// - `q_pos`:       absolute position of the query token
///
/// # Returns
///
/// Flattened output `[num_heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_alibi(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    config: &AliBiConfig,
    head_dim: usize,
    num_kv_heads: usize,
    q_pos: usize,
) -> Vec<f32> {
    let num_heads = config.num_heads;
    debug_assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    debug_assert_eq!(query.len(), num_heads * head_dim);
    let kv_len = if num_kv_heads > 0 && head_dim > 0 {
        keys.len() / (num_kv_heads * head_dim)
    } else {
        0
    };

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let alibi = if config.use_extrapolated_slopes {
        AliBiBias {
            slopes: AliBiSlopes::new_extrapolated(num_heads),
        }
    } else {
        AliBiBias::new(num_heads)
    };

    let mut output = vec![0.0_f32; num_heads * head_dim];

    for q_head in 0..num_heads {
        let kv_head = q_head / heads_per_kv;
        let q_start = q_head * head_dim;
        let q_vec = &query[q_start..q_start + head_dim];

        // Compute raw dot-product scores
        let mut scores: Vec<f32> = (0..kv_len)
            .map(|t| {
                // keys layout: [kv_len * num_kv_heads * head_dim], token-major
                let k_start = t * num_kv_heads * head_dim + kv_head * head_dim;
                let k_vec = &keys[k_start..k_start + head_dim];
                dot(q_vec, k_vec) * scale
            })
            .collect();

        // Add ALiBi bias
        let biases = alibi.bias_for_head(q_head, q_pos, kv_len);
        for (s, b) in scores.iter_mut().zip(biases.iter()) {
            *s += b;
        }

        // Apply causal mask: future tokens → -∞
        if config.causal {
            for (k, s) in scores.iter_mut().enumerate() {
                if k > q_pos {
                    *s = f32::NEG_INFINITY;
                }
            }
        }

        softmax(&mut scores);

        // Weighted sum of values
        let out_start = q_head * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (t, &score_t) in scores.iter().enumerate().take(kv_len) {
                let v_start = t * num_kv_heads * head_dim + kv_head * head_dim;
                acc += score_t * values[v_start + d];
            }
            output[out_start + d] = acc;
        }
    }

    output
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── slope tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_alibi_slopes_power_of_2() {
        // 8 heads is a power of 2 — both constructors should give the same result
        let s1 = AliBiSlopes::new(8);
        let s2 = AliBiSlopes::new_extrapolated(8);
        assert_eq!(s1.num_heads(), 8);
        assert_eq!(s2.num_heads(), 8);
        for i in 0..8 {
            assert!(
                (s1.get(i) - s2.get(i)).abs() < 1e-6,
                "slope mismatch at head {i}: {} vs {}",
                s1.get(i),
                s2.get(i)
            );
        }
    }

    #[test]
    fn test_alibi_slopes_8_heads() {
        let slopes = AliBiSlopes::new(8);
        assert_eq!(slopes.num_heads(), 8);
        assert_eq!(slopes.slopes().len(), 8);
        // start = 2^(-8/8) = 2^(-1) = 0.5
        // slopes[0] = 0.5^1 = 0.5
        let expected_first = 0.5_f32;
        assert!(
            (slopes.get(0) - expected_first).abs() < 1e-5,
            "first slope: got {}, expected {}",
            slopes.get(0),
            expected_first
        );
        // slopes[7] = 0.5^8 ≈ 0.00390625
        let expected_last = 0.5_f32.powi(8);
        assert!(
            (slopes.get(7) - expected_last).abs() < 1e-7,
            "last slope: got {}, expected {}",
            slopes.get(7),
            expected_last
        );
    }

    #[test]
    fn test_alibi_slopes_decreasing() {
        // Each successive slope must be strictly smaller
        let slopes = AliBiSlopes::new(16);
        for i in 1..16 {
            assert!(
                slopes.get(i) < slopes.get(i - 1),
                "slopes not strictly decreasing at index {i}: {} >= {}",
                slopes.get(i),
                slopes.get(i - 1)
            );
        }
    }

    // ── bias tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_alibi_bias_zero_distance() {
        // At k == q_pos the distance is 0, so bias must be exactly 0
        let bias = AliBiBias::new(4);
        for head in 0..4 {
            let q_pos = 7;
            let biases = bias.bias_for_head(head, q_pos, q_pos + 1);
            let at_q = biases[q_pos];
            assert!(
                at_q.abs() < 1e-8,
                "head {head}: expected zero bias at q_pos={q_pos}, got {at_q}"
            );
        }
    }

    #[test]
    fn test_alibi_bias_increases_with_distance() {
        // Bias is -slope * (q - k), so bias[k] < bias[k+1] (further away = more negative)
        let bias = AliBiBias::new(4);
        let q_pos = 10;
        let kv_len = 11;
        for head in 0..4 {
            let biases = bias.bias_for_head(head, q_pos, kv_len);
            // biases[0] < biases[1] < ... < biases[q_pos] == 0
            for k in 1..=q_pos {
                assert!(
                    biases[k] > biases[k - 1],
                    "head {head}: bias should increase with k, but biases[{k}]={} <= biases[{}]={}",
                    biases[k],
                    k - 1,
                    biases[k - 1]
                );
            }
        }
    }

    #[test]
    fn test_alibi_biases_all_heads_shape() {
        let num_heads = 6;
        let kv_len = 20;
        let bias = AliBiBias::new(num_heads);
        let all = bias.biases_all_heads(15, kv_len);
        assert_eq!(all.len(), num_heads, "outer dim must equal num_heads");
        for (h, row) in all.iter().enumerate() {
            assert_eq!(row.len(), kv_len, "head {h}: inner dim must equal kv_len");
        }
    }

    #[test]
    fn test_alibi_apply_modifies_scores() {
        let num_heads = 4;
        let kv_len = 5;
        let bias = AliBiBias::new(num_heads);

        // Start with all-zero scores
        let mut scores: Vec<Vec<f32>> = vec![vec![0.0_f32; kv_len]; num_heads];
        let q_pos = 4; // last position
        bias.apply(&mut scores, q_pos);

        // After apply, position q_pos should have bias == 0
        for (head, scores_head) in scores.iter().enumerate() {
            assert!(
                scores_head[q_pos].abs() < 1e-8,
                "head {head}: score at q_pos should be 0 after ALiBi, got {}",
                scores_head[q_pos]
            );
            // Positions before q_pos should be negative
            for (k, &score_k) in scores_head[..q_pos].iter().enumerate() {
                assert!(
                    score_k < 0.0,
                    "head {head}: score at k={k} should be negative, got {}",
                    score_k
                );
            }
        }
    }

    #[test]
    fn test_alibi_biases_for_sequence_shape() {
        let num_heads = 4;
        let q_len = 3;
        let kv_len = 8;
        let q_offset = 5;
        let bias = AliBiBias::new(num_heads);
        let seq_biases = bias.biases_for_sequence(q_len, kv_len, q_offset);

        assert_eq!(seq_biases.len(), q_len, "outer dim must equal q_len");
        for (qi, head_biases) in seq_biases.iter().enumerate() {
            assert_eq!(
                head_biases.len(),
                num_heads,
                "q={qi}: second dim must equal num_heads"
            );
            for (h, kv_biases) in head_biases.iter().enumerate() {
                assert_eq!(
                    kv_biases.len(),
                    kv_len,
                    "q={qi} h={h}: inner dim must equal kv_len"
                );
            }
        }
    }

    // ── attention_with_alibi tests ────────────────────────────────────────────

    #[test]
    fn test_attention_with_alibi_output_shape() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let kv_len = 5;
        let q_pos = 4;

        let query = vec![0.1_f32; num_heads * head_dim];
        // layout: [kv_len][num_kv_heads][head_dim]
        let keys = vec![0.05_f32; kv_len * num_kv_heads * head_dim];
        let values = vec![0.2_f32; kv_len * num_kv_heads * head_dim];

        let config = AliBiConfig {
            num_heads,
            use_extrapolated_slopes: false,
            causal: true,
        };

        let output = attention_with_alibi(
            &query,
            &keys,
            &values,
            &config,
            head_dim,
            num_kv_heads,
            q_pos,
        );

        assert_eq!(
            output.len(),
            num_heads * head_dim,
            "output length must be num_heads * head_dim"
        );
        // All outputs should be finite
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_alibi_extrapolated_slopes() {
        // 12 is not a power of 2 — extrapolated variant must produce 12 slopes
        let slopes = AliBiSlopes::new_extrapolated(12);
        assert_eq!(slopes.num_heads(), 12);
        assert_eq!(slopes.slopes().len(), 12);
        // All slopes must be in (0, 1)
        for (i, &s) in slopes.slopes().iter().enumerate() {
            assert!(
                s > 0.0 && s < 1.0,
                "extrapolated slope[{i}] = {s} out of (0,1)"
            );
        }
    }
}
