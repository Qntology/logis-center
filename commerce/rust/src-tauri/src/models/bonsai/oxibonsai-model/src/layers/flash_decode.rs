//! Flash Decoding: parallelized decode-phase attention.
//!
//! During inference decoding, we have Q=[1, h, d] and K/V=[S, h, d] where S can be large.
//! Flash Decoding splits the KV sequence into tiles and computes partial softmax in parallel,
//! then combines using the log-sum-exp trick.
//!
//! References: Dao et al. 2023 — "FlashDecoding++"

use rayon::prelude::*;

// ─── FlashDecodeConfig ────────────────────────────────────────────────────────

/// Configuration for flash decoding.
#[derive(Debug, Clone)]
pub struct FlashDecodeConfig {
    /// Number of tiles to split the KV sequence into.
    pub num_tiles: usize,
    /// Scale factor for attention scores: 1/sqrt(head_dim).
    pub scale: f32,
}

impl FlashDecodeConfig {
    /// Create a config with default num_tiles=4 and scale=1/sqrt(head_dim).
    pub fn new(head_dim: usize) -> Self {
        let scale = if head_dim > 0 {
            1.0_f32 / (head_dim as f32).sqrt()
        } else {
            1.0_f32
        };
        Self {
            num_tiles: 4,
            scale,
        }
    }

    /// Set the number of tiles.
    #[must_use]
    pub fn with_num_tiles(mut self, n: usize) -> Self {
        self.num_tiles = n;
        self
    }
}

// ─── flash_decode_tile ───────────────────────────────────────────────────────

/// Compute partial attention output for a single tile.
///
/// Returns `(output_tile, max_score, log_sum_exp)`.
///
/// The log-sum-exp trick:
/// ```text
/// m  = max(scores)
/// sum = Σ exp(score_i - m)
/// lse = m + ln(sum)
/// output = Σ (exp(score_i - m) / sum) * v_i
/// ```
fn flash_decode_tile(
    query: &[f32],
    keys_tile: &[f32],
    values_tile: &[f32],
    tile_len: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, f32, f32) {
    // Compute dot-product scores
    let mut scores: Vec<f32> = (0..tile_len)
        .map(|t| {
            let k_start = t * head_dim;
            let k_vec = &keys_tile[k_start..k_start + head_dim];
            query
                .iter()
                .zip(k_vec.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale
        })
        .collect();

    // Find max for numerical stability (log-sum-exp trick)
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if !max_score.is_finite() {
        // All -inf: return zero output
        return (
            vec![0.0_f32; head_dim],
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
    }

    // Shift scores and compute softmax weights
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
    }
    let sum: f32 = scores.iter().sum();
    let log_sum_exp = max_score + sum.ln();

    // Weighted sum of values
    let mut output = vec![0.0_f32; head_dim];
    for (t, &w) in scores.iter().enumerate() {
        let v_start = t * head_dim;
        let v_vec = &values_tile[v_start..v_start + head_dim];
        for d in 0..head_dim {
            output[d] += w * v_vec[d];
        }
    }

    // Normalize by sum
    if sum > 0.0 {
        for o in output.iter_mut() {
            *o /= sum;
        }
    }

    (output, max_score, log_sum_exp)
}

// ─── combine_tile_outputs ────────────────────────────────────────────────────

/// Combine tile outputs using log-sum-exp reduction.
///
/// Each tile has a partial output, max score, and log-sum-exp value.
/// The final output is the weighted combination where each tile is weighted
/// by `exp(lse_i - global_lse)`.
fn combine_tile_outputs(
    tile_outputs: &[Vec<f32>],
    tile_max_scores: &[f32],
    tile_lse: &[f32],
    head_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(tile_outputs.len(), tile_lse.len());
    debug_assert_eq!(tile_outputs.len(), tile_max_scores.len());

    if tile_outputs.is_empty() {
        return vec![0.0_f32; head_dim];
    }
    if tile_outputs.len() == 1 {
        return tile_outputs[0].clone();
    }

    // Filter out tiles with -inf lse (empty or all-masked tiles)
    let valid: Vec<usize> = (0..tile_lse.len())
        .filter(|&i| tile_lse[i].is_finite())
        .collect();

    if valid.is_empty() {
        return vec![0.0_f32; head_dim];
    }
    if valid.len() == 1 {
        return tile_outputs[valid[0]].clone();
    }

    // Global log-sum-exp across tile LSEs
    let global_lse_max = valid
        .iter()
        .map(|&i| tile_lse[i])
        .fold(f32::NEG_INFINITY, f32::max);

    let global_sum: f32 = valid
        .iter()
        .map(|&i| (tile_lse[i] - global_lse_max).exp())
        .sum();
    let global_lse = global_lse_max + global_sum.ln();

    // Combine: output = Σ_tile exp(lse_tile - global_lse) * output_tile
    let mut combined = vec![0.0_f32; head_dim];
    for &i in &valid {
        let weight = (tile_lse[i] - global_lse).exp();
        for d in 0..head_dim {
            combined[d] += weight * tile_outputs[i][d];
        }
    }

    combined
}

// ─── flash_decode_single_head ─────────────────────────────────────────────────

/// Compute attention for a single query token against full KV cache.
///
/// - `query`:   shape `[head_dim]`
/// - `keys`:    shape `[seq_len * head_dim]` (row-major: token is outer dim)
/// - `values`:  shape `[seq_len * head_dim]`
/// - Returns:   shape `[head_dim]`
pub fn flash_decode_single_head(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
    config: &FlashDecodeConfig,
) -> Result<Vec<f32>, FlashDecodeError> {
    if seq_len == 0 {
        return Err(FlashDecodeError::EmptyKv);
    }
    if query.len() != head_dim {
        return Err(FlashDecodeError::DimMismatch {
            q_dim: query.len(),
            k_dim: head_dim,
        });
    }
    if keys.len() != seq_len * head_dim {
        return Err(FlashDecodeError::DimMismatch {
            q_dim: query.len(),
            k_dim: keys.len() / seq_len.max(1),
        });
    }
    if values.len() != seq_len * head_dim {
        return Err(FlashDecodeError::DimMismatch {
            q_dim: query.len(),
            k_dim: values.len() / seq_len.max(1),
        });
    }

    // Clamp num_tiles to seq_len
    let num_tiles = config.num_tiles.min(seq_len).max(1);

    let tile_size_base = seq_len / num_tiles;
    let remainder = seq_len % num_tiles;

    let mut tile_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_tiles);
    let mut tile_max_scores: Vec<f32> = Vec::with_capacity(num_tiles);
    let mut tile_lse: Vec<f32> = Vec::with_capacity(num_tiles);

    let mut offset = 0usize;
    for tile_idx in 0..num_tiles {
        // Distribute remainder tokens among the first `remainder` tiles
        let tile_len = tile_size_base + if tile_idx < remainder { 1 } else { 0 };
        if tile_len == 0 {
            break;
        }

        let k_start = offset * head_dim;
        let k_end = k_start + tile_len * head_dim;
        let v_start = offset * head_dim;
        let v_end = v_start + tile_len * head_dim;

        let (out, max_s, lse) = flash_decode_tile(
            query,
            &keys[k_start..k_end],
            &values[v_start..v_end],
            tile_len,
            head_dim,
            config.scale,
        );
        tile_outputs.push(out);
        tile_max_scores.push(max_s);
        tile_lse.push(lse);

        offset += tile_len;
    }

    Ok(combine_tile_outputs(
        &tile_outputs,
        &tile_max_scores,
        &tile_lse,
        head_dim,
    ))
}

// ─── flash_decode_multi_head ─────────────────────────────────────────────────

/// Multi-head flash decode: compute attention across all heads in parallel (via rayon).
///
/// - `queries`: shape `[num_heads * head_dim]`
/// - `keys`:    shape `[seq_len * num_heads * head_dim]` (token-major)
/// - `values`:  same shape as `keys`
///
/// Returns flattened `[num_heads * head_dim]`.
pub fn flash_decode_multi_head(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    config: &FlashDecodeConfig,
) -> Result<Vec<f32>, FlashDecodeError> {
    if seq_len == 0 {
        return Err(FlashDecodeError::EmptyKv);
    }
    if queries.len() != num_heads * head_dim {
        return Err(FlashDecodeError::DimMismatch {
            q_dim: queries.len(),
            k_dim: head_dim,
        });
    }

    // Re-index keys/values from [seq_len, num_heads, head_dim] to per-head
    // [seq_len, head_dim] slices for each head.
    // We build per-head key and value buffers.
    let per_head_keys: Vec<Vec<f32>> = (0..num_heads)
        .map(|h| {
            let mut buf = vec![0.0_f32; seq_len * head_dim];
            for t in 0..seq_len {
                let src_start = t * num_heads * head_dim + h * head_dim;
                let dst_start = t * head_dim;
                buf[dst_start..dst_start + head_dim]
                    .copy_from_slice(&keys[src_start..src_start + head_dim]);
            }
            buf
        })
        .collect();

    let per_head_values: Vec<Vec<f32>> = (0..num_heads)
        .map(|h| {
            let mut buf = vec![0.0_f32; seq_len * head_dim];
            for t in 0..seq_len {
                let src_start = t * num_heads * head_dim + h * head_dim;
                let dst_start = t * head_dim;
                buf[dst_start..dst_start + head_dim]
                    .copy_from_slice(&values[src_start..src_start + head_dim]);
            }
            buf
        })
        .collect();

    // Process each head in parallel using rayon
    let results: Vec<Result<Vec<f32>, FlashDecodeError>> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            let q_start = h * head_dim;
            let q_vec = &queries[q_start..q_start + head_dim];
            flash_decode_single_head(
                q_vec,
                &per_head_keys[h],
                &per_head_values[h],
                seq_len,
                head_dim,
                config,
            )
        })
        .collect();

    // Flatten results
    let mut output = vec![0.0_f32; num_heads * head_dim];
    for (h, res) in results.into_iter().enumerate() {
        let head_out = res?;
        let start = h * head_dim;
        output[start..start + head_dim].copy_from_slice(&head_out);
    }

    Ok(output)
}

// ─── flash_vs_naive_error ─────────────────────────────────────────────────────

/// Compare flash decode vs naive attention (for testing).
///
/// Returns the mean absolute error between the two implementations.
pub fn flash_vs_naive_error(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Result<f32, FlashDecodeError> {
    if seq_len == 0 {
        return Err(FlashDecodeError::EmptyKv);
    }
    if query.len() != head_dim {
        return Err(FlashDecodeError::DimMismatch {
            q_dim: query.len(),
            k_dim: head_dim,
        });
    }

    // Flash decode output
    let config = FlashDecodeConfig::new(head_dim);
    let flash_out = flash_decode_single_head(query, keys, values, seq_len, head_dim, &config)?;

    // Naive attention output
    let scale = config.scale;
    let mut scores: Vec<f32> = (0..seq_len)
        .map(|t| {
            let k_start = t * head_dim;
            let k_vec = &keys[k_start..k_start + head_dim];
            query
                .iter()
                .zip(k_vec.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale
        })
        .collect();

    // Softmax
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for s in scores.iter_mut() {
        *s = (*s - max_s).exp();
    }
    let sum: f32 = scores.iter().sum();
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }

    // Weighted sum of values
    let mut naive_out = vec![0.0_f32; head_dim];
    for (t, &w) in scores.iter().enumerate() {
        let v_start = t * head_dim;
        for d in 0..head_dim {
            naive_out[d] += w * values[v_start + d];
        }
    }

    // Mean absolute error
    let mae = flash_out
        .iter()
        .zip(naive_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / head_dim as f32;

    Ok(mae)
}

// ─── FlashDecodeError ────────────────────────────────────────────────────────

/// Errors from flash decode operations.
#[derive(Debug, thiserror::Error)]
pub enum FlashDecodeError {
    #[error("empty KV sequence")]
    EmptyKv,

    #[error("dimension mismatch: query has {q_dim}, keys have {k_dim}")]
    DimMismatch { q_dim: usize, k_dim: usize },

    #[error("num_tiles ({0}) exceeds seq_len ({1})")]
    TooManyTiles(usize, usize),

    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_deterministic_data(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let query: Vec<f32> = (0..head_dim).map(|i| 0.1 * i as f32).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| 0.05 * i as f32 + 0.01)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| 0.02 * i as f32 + 0.1)
            .collect();
        (query, keys, values)
    }

    #[test]
    fn flash_decode_config_default() {
        let head_dim = 64usize;
        let cfg = FlashDecodeConfig::new(head_dim);
        let expected_scale = 1.0_f32 / (head_dim as f32).sqrt();
        assert!(
            (cfg.scale - expected_scale).abs() < 1e-6,
            "scale mismatch: {} vs {}",
            cfg.scale,
            expected_scale
        );
        assert_eq!(cfg.num_tiles, 4);
    }

    #[test]
    fn flash_decode_single_head_matches_naive() {
        let head_dim = 16;
        let seq_len = 32;
        let (q, k, v) = make_deterministic_data(seq_len, head_dim);
        let mae = flash_vs_naive_error(&q, &k, &v, seq_len, head_dim)
            .expect("flash_vs_naive_error failed");
        assert!(
            mae < 1e-5,
            "MAE between flash and naive exceeds threshold: {mae}"
        );
    }

    #[test]
    fn flash_decode_empty_kv_error() {
        let head_dim = 8;
        let config = FlashDecodeConfig::new(head_dim);
        let q = vec![0.1f32; head_dim];
        let result = flash_decode_single_head(&q, &[], &[], 0, head_dim, &config);
        assert!(
            matches!(result, Err(FlashDecodeError::EmptyKv)),
            "expected EmptyKv, got {result:?}"
        );
    }

    #[test]
    fn flash_decode_dim_mismatch_error() {
        let head_dim = 8;
        let config = FlashDecodeConfig::new(head_dim);
        // query has wrong length
        let q = vec![0.1f32; head_dim + 2];
        let k = vec![0.1f32; head_dim];
        let v = vec![0.1f32; head_dim];
        let result = flash_decode_single_head(&q, &k, &v, 1, head_dim, &config);
        assert!(
            matches!(result, Err(FlashDecodeError::DimMismatch { .. })),
            "expected DimMismatch, got {result:?}"
        );
    }

    #[test]
    fn flash_decode_single_token() {
        // seq_len=1: output should equal value[0] (since softmax of single element = 1.0)
        let head_dim = 4;
        let config = FlashDecodeConfig::new(head_dim);
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let k = vec![0.5f32, 0.5, 0.5, 0.5]; // single key
        let v = vec![3.0f32, 1.0, 2.0, 4.0]; // single value

        let out = flash_decode_single_head(&q, &k, &v, 1, head_dim, &config)
            .expect("flash_decode_single_head failed");

        for (i, (&o, &expected)) in out.iter().zip(v.iter()).enumerate() {
            assert!(
                (o - expected).abs() < 1e-5,
                "output[{i}] = {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn flash_decode_uniform_keys() {
        // When all keys are identical and uniform queries, output = average of values
        // Actually: uniform attention weights → output = mean of values per dimension
        let head_dim = 4;
        let seq_len = 4;
        let config = FlashDecodeConfig::new(head_dim);
        let q = vec![0.1f32; head_dim];
        let k = vec![0.1f32; seq_len * head_dim]; // identical keys

        // Values: row t has all elements = (t+1) as f32
        let v: Vec<f32> = (0..seq_len)
            .flat_map(|t| vec![(t + 1) as f32; head_dim])
            .collect();

        let out = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config)
            .expect("flash_decode_single_head failed");

        // With uniform keys, all attention weights equal 1/seq_len
        // Expected output per dim = mean of [1, 2, 3, 4] = 2.5
        let expected = 2.5_f32;
        for (i, &o) in out.iter().enumerate() {
            assert!(
                (o - expected).abs() < 1e-4,
                "output[{i}] = {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn flash_decode_tile_count_1() {
        let head_dim = 8;
        let seq_len = 16;
        let config = FlashDecodeConfig::new(head_dim).with_num_tiles(1);
        let (q, k, v) = make_deterministic_data(seq_len, head_dim);
        let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
        assert!(result.is_ok(), "num_tiles=1 should be valid: {result:?}");
    }

    #[test]
    fn flash_decode_tile_count_many() {
        let head_dim = 8;
        let seq_len = 16;
        let config = FlashDecodeConfig::new(head_dim).with_num_tiles(8);
        let (q, k, v) = make_deterministic_data(seq_len, head_dim);
        let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
        assert!(
            result.is_ok(),
            "num_tiles=8 with seq_len=16 failed: {result:?}"
        );
    }

    #[test]
    fn flash_vs_naive_error_small() {
        let head_dim = 32;
        let seq_len = 64;
        let (q, k, v) = make_deterministic_data(seq_len, head_dim);
        let mae = flash_vs_naive_error(&q, &k, &v, seq_len, head_dim)
            .expect("flash_vs_naive_error failed");
        assert!(mae < 1e-4, "MAE too large: {mae}");
    }

    #[test]
    fn flash_decode_multi_head_shape() {
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let config = FlashDecodeConfig::new(head_dim);

        let queries = vec![0.1f32; num_heads * head_dim];
        let keys = vec![0.05f32; seq_len * num_heads * head_dim];
        let values = vec![0.2f32; seq_len * num_heads * head_dim];

        let out = flash_decode_multi_head(
            &queries, &keys, &values, num_heads, seq_len, head_dim, &config,
        )
        .expect("multi_head flash decode failed");

        assert_eq!(
            out.len(),
            num_heads * head_dim,
            "output shape mismatch: {} vs {}",
            out.len(),
            num_heads * head_dim
        );
    }

    #[test]
    fn flash_decode_multi_head_matches_naive_per_head() {
        let num_heads = 2;
        let head_dim = 8;
        let seq_len = 16;
        let config = FlashDecodeConfig::new(head_dim);

        // Deterministic data
        let queries: Vec<f32> = (0..num_heads * head_dim).map(|i| 0.1 * i as f32).collect();
        let keys: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| 0.05 * (i % 17) as f32 + 0.01)
            .collect();
        let values: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| 0.02 * (i % 13) as f32 + 0.1)
            .collect();

        let flash_out = flash_decode_multi_head(
            &queries, &keys, &values, num_heads, seq_len, head_dim, &config,
        )
        .expect("multi_head flash decode failed");

        // Check each head individually against naive attention
        for h in 0..num_heads {
            let q_vec = &queries[h * head_dim..(h + 1) * head_dim];

            // Extract per-head K/V
            let mut k_head = vec![0.0f32; seq_len * head_dim];
            let mut v_head = vec![0.0f32; seq_len * head_dim];
            for t in 0..seq_len {
                let src_k = t * num_heads * head_dim + h * head_dim;
                let src_v = t * num_heads * head_dim + h * head_dim;
                let dst = t * head_dim;
                k_head[dst..dst + head_dim].copy_from_slice(&keys[src_k..src_k + head_dim]);
                v_head[dst..dst + head_dim].copy_from_slice(&values[src_v..src_v + head_dim]);
            }

            let naive_config = FlashDecodeConfig::new(head_dim).with_num_tiles(1);
            let naive_out =
                flash_decode_single_head(q_vec, &k_head, &v_head, seq_len, head_dim, &naive_config)
                    .expect("naive single head failed");

            let head_flash = &flash_out[h * head_dim..(h + 1) * head_dim];
            let mae: f32 = head_flash
                .iter()
                .zip(naive_out.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / head_dim as f32;
            assert!(
                mae < 1e-4,
                "head {h}: MAE between multi_head flash and single-head naive = {mae}"
            );
        }
    }

    #[test]
    fn combine_tiles_single_tile() {
        let head_dim = 4;
        let tile_out = vec![1.0f32, 2.0, 3.0, 4.0];
        let combined = combine_tile_outputs(
            std::slice::from_ref(&tile_out),
            &[0.5_f32],
            &[1.0_f32],
            head_dim,
        );
        // Single tile: output == tile output
        for (i, (&c, &t)) in combined.iter().zip(tile_out.iter()).enumerate() {
            assert!((c - t).abs() < 1e-5, "combined[{i}] = {c}, expected {t}");
        }
    }

    #[test]
    fn flash_decode_long_sequence() {
        let head_dim = 16;
        let seq_len = 128;
        let config = FlashDecodeConfig::new(head_dim).with_num_tiles(8);
        let (q, k, v) = make_deterministic_data(seq_len, head_dim);
        let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
        assert!(
            result.is_ok(),
            "long sequence (seq_len=128) failed: {result:?}"
        );
        let out = result.expect("already checked");
        assert_eq!(out.len(), head_dim);
        for (i, &o) in out.iter().enumerate() {
            assert!(o.is_finite(), "output[{i}] = {o} is not finite");
        }
    }
}
