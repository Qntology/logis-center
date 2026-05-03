//! Dynamic activation quantization for W8A8 / W4A8 inference.
//!
//! Unlike static quantization (which uses pre-computed scales), dynamic
//! quantization computes the quantization scale from the current activation
//! values at inference time. This is slower than static but more accurate.
//!
//! # Supported formats
//! - `DynamicInt8`: Per-tensor symmetric INT8 (1 scale per tensor)
//! - `DynamicInt8PerRow`: Per-row symmetric INT8 (1 scale per row in a 2D tensor)
//! - `DynamicInt4`: Per-tensor symmetric INT4 (values in [-7, 7], using i8 storage)
//! - `SmoothQuant`: Activation-weight smoothing to reduce quantization error

/// How to compute the dynamic quantization scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicScaleMode {
    /// Use the max absolute value: scale = max(|x|) / clip_val
    MaxAbs,
    /// Use a percentile of absolute values (more robust to outliers).
    /// `percentile` in (0, 1] — e.g. 0.99 = 99th percentile
    Percentile(f32),
}

/// Format of a dynamically quantized tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynQuantFormat {
    /// Per-tensor INT8: one scale for the whole tensor.
    Int8PerTensor,
    /// Per-row INT8: one scale per row in a 2D tensor.
    Int8PerRow,
    /// Per-tensor INT4: packed 2 values per byte, stored as i8 in [-7, 7].
    Int4PerTensor,
}

/// A dynamically quantized tensor.
#[derive(Debug, Clone)]
pub struct DynQuantTensor {
    /// Quantized values (i8 storage for both INT8 and INT4).
    pub data: Vec<i8>,
    /// Scales, one per quantization group.
    pub scales: Vec<f32>,
    /// Shape of the original tensor.
    pub shape: Vec<usize>,
    /// Quantization format.
    pub format: DynQuantFormat,
}

impl DynQuantTensor {
    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        match self.format {
            DynQuantFormat::Int8PerTensor => {
                let scale = self.scales.first().copied().unwrap_or(0.0);
                self.data.iter().map(|&q| q as f32 * scale).collect()
            }
            DynQuantFormat::Int8PerRow => {
                if self.scales.is_empty() || self.data.is_empty() {
                    return Vec::new();
                }
                let rows = self.scales.len();
                let cols = self.data.len() / rows.max(1);
                let mut out = Vec::with_capacity(self.data.len());
                for (r, &scale) in self.scales.iter().enumerate() {
                    let start = r * cols;
                    let end = (start + cols).min(self.data.len());
                    for &q in &self.data[start..end] {
                        out.push(q as f32 * scale);
                    }
                }
                out
            }
            DynQuantFormat::Int4PerTensor => {
                let scale = self.scales.first().copied().unwrap_or(0.0);
                self.data.iter().map(|&q| q as f32 * scale).collect()
            }
        }
    }

    /// Memory in bytes (data + scales).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * core::mem::size_of::<f32>()
    }

    /// Compression ratio vs f32 (data only, excluding scales).
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.data.len() * core::mem::size_of::<f32>();
        let quantized_bytes = self.memory_bytes();
        if quantized_bytes == 0 {
            return 1.0;
        }
        original_bytes as f32 / quantized_bytes as f32
    }

    /// Number of elements.
    pub fn element_count(&self) -> usize {
        self.data.len()
    }
}

// ─── Scale computation ────────────────────────────────────────────────────────

/// Compute the quantization scale for a slice.
///
/// - `MaxAbs`: `scale = max(|x|) / clip_val`
/// - `Percentile(p)`: sort absolute values, use p-th percentile value / clip_val
pub fn compute_scale(data: &[f32], clip_val: f32, mode: DynamicScaleMode) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let abs_max = match mode {
        DynamicScaleMode::MaxAbs => data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max),
        DynamicScaleMode::Percentile(p) => {
            let p_clamped = p.clamp(0.0, 1.0);
            let mut abs_vals: Vec<f32> = data.iter().map(|x| x.abs()).collect();
            // Sort ascending
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let len = abs_vals.len();
            // Compute index: ceiling of p * len, then subtract 1, clamped
            let idx = ((p_clamped * len as f32).ceil() as usize)
                .saturating_sub(1)
                .min(len - 1);
            abs_vals[idx]
        }
    };

    if abs_max == 0.0 {
        return 0.0;
    }

    abs_max / clip_val
}

// ─── INT8 per-tensor ──────────────────────────────────────────────────────────

/// Dynamically quantize a 1D activation tensor to INT8 (per-tensor).
pub fn dynamic_quantize_int8(data: &[f32], mode: DynamicScaleMode) -> DynQuantTensor {
    const CLIP_VAL: f32 = 127.0;

    if data.is_empty() {
        return DynQuantTensor {
            data: Vec::new(),
            scales: vec![0.0],
            shape: vec![0],
            format: DynQuantFormat::Int8PerTensor,
        };
    }

    let scale = compute_scale(data, CLIP_VAL, mode);

    let quantized: Vec<i8> = if scale == 0.0 {
        vec![0i8; data.len()]
    } else {
        data.iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect()
    };

    DynQuantTensor {
        data: quantized,
        scales: vec![scale],
        shape: vec![data.len()],
        format: DynQuantFormat::Int8PerTensor,
    }
}

// ─── INT8 per-row ─────────────────────────────────────────────────────────────

/// Dynamically quantize a 2D activation tensor to INT8, one scale per row.
///
/// `data` is row-major with shape `[rows, cols]`.
pub fn dynamic_quantize_int8_per_row(
    data: &[f32],
    rows: usize,
    cols: usize,
    mode: DynamicScaleMode,
) -> DynQuantTensor {
    const CLIP_VAL: f32 = 127.0;

    if data.is_empty() || rows == 0 || cols == 0 {
        return DynQuantTensor {
            data: Vec::new(),
            scales: Vec::new(),
            shape: vec![rows, cols],
            format: DynQuantFormat::Int8PerRow,
        };
    }

    let total = rows * cols;
    let actual_len = data.len().min(total);

    let mut quantized = Vec::with_capacity(actual_len);
    let mut scales = Vec::with_capacity(rows);

    for r in 0..rows {
        let start = r * cols;
        let end = (start + cols).min(data.len());
        if start >= data.len() {
            // Pad with zeros if row is out of bounds
            quantized.extend(vec![0i8; cols]);
            scales.push(0.0_f32);
            continue;
        }
        let row = &data[start..end];
        let scale = compute_scale(row, CLIP_VAL, mode);
        scales.push(scale);
        if scale == 0.0 {
            quantized.extend(vec![0i8; row.len()]);
        } else {
            for &x in row {
                quantized.push((x / scale).round().clamp(-127.0, 127.0) as i8);
            }
        }
    }

    DynQuantTensor {
        data: quantized,
        scales,
        shape: vec![rows, cols],
        format: DynQuantFormat::Int8PerRow,
    }
}

// ─── INT4 per-tensor ──────────────────────────────────────────────────────────

/// INT4 quantization: clamp to [-7, 7], stored as i8.
///
/// `scale = max(|x|) / 7.0`
pub fn dynamic_quantize_int4(data: &[f32], mode: DynamicScaleMode) -> DynQuantTensor {
    const CLIP_VAL: f32 = 7.0;

    if data.is_empty() {
        return DynQuantTensor {
            data: Vec::new(),
            scales: vec![0.0],
            shape: vec![0],
            format: DynQuantFormat::Int4PerTensor,
        };
    }

    let scale = compute_scale(data, CLIP_VAL, mode);

    let quantized: Vec<i8> = if scale == 0.0 {
        vec![0i8; data.len()]
    } else {
        data.iter()
            .map(|&x| (x / scale).round().clamp(-7.0, 7.0) as i8)
            .collect()
    };

    DynQuantTensor {
        data: quantized,
        scales: vec![scale],
        shape: vec![data.len()],
        format: DynQuantFormat::Int4PerTensor,
    }
}

// ─── Error metrics ────────────────────────────────────────────────────────────

/// Mean absolute quantization error between original f32 data and a quantized tensor.
pub fn quantization_mae(original: &[f32], quantized: &DynQuantTensor) -> f32 {
    let reconstructed = quantized.dequantize();
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let sum_abs_err: f32 = original[..n]
        .iter()
        .zip(reconstructed[..n].iter())
        .map(|(&o, &r)| (o - r).abs())
        .sum();
    sum_abs_err / n as f32
}

// ─── SmoothQuant ─────────────────────────────────────────────────────────────

/// SmoothQuant configuration: redistribute quantization difficulty from activations to weights.
///
/// Smoothing factor: `s_j = max(|A_j|)^α / max(|W_j|)^(1-α)`
/// Then: `Ã = A / s`, `W̃ = W * s`
#[derive(Debug, Clone)]
pub struct SmoothQuantConfig {
    /// Balance factor in [0, 1]. Typically 0.5.
    pub alpha: f32,
    /// Floor for scale values to avoid division by zero.
    pub epsilon: f32,
}

impl SmoothQuantConfig {
    /// Create a new config with the given alpha (must be in [0, 1]).
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            epsilon: 1e-5,
        }
    }

    /// Default config with alpha = 0.5.
    pub fn default_alpha() -> Self {
        Self::new(0.5)
    }
}

/// Compute SmoothQuant smoothing factors (one per input feature).
///
/// - `activations`: shape `[tokens, in_features]` (row-major)
/// - `weights`: shape `[out_features, in_features]` (row-major)
/// - Returns: smoothing factors of length `in_features`
pub fn compute_smooth_factors(
    activations: &[f32],
    weights: &[f32],
    in_features: usize,
    tokens: usize,
    out_features: usize,
    config: &SmoothQuantConfig,
) -> Vec<f32> {
    if in_features == 0 {
        return Vec::new();
    }

    let alpha = config.alpha.clamp(0.0, 1.0);
    let epsilon = config.epsilon.max(1e-10);

    // Compute per-column max abs of activations: shape [tokens, in_features]
    let mut act_max = vec![0.0_f32; in_features];
    for t in 0..tokens {
        for (j, slot) in act_max.iter_mut().enumerate() {
            let idx = t * in_features + j;
            if idx < activations.len() {
                let v = activations[idx].abs();
                if v > *slot {
                    *slot = v;
                }
            }
        }
    }

    // Compute per-column max abs of weights: shape [out_features, in_features]
    let mut w_max = vec![0.0_f32; in_features];
    for o in 0..out_features {
        for (j, slot) in w_max.iter_mut().enumerate() {
            let idx = o * in_features + j;
            if idx < weights.len() {
                let v = weights[idx].abs();
                if v > *slot {
                    *slot = v;
                }
            }
        }
    }

    // s_j = max(|A_j|)^alpha / max(|W_j|)^(1 - alpha)
    (0..in_features)
        .map(|j| {
            let a = (act_max[j] + epsilon).powf(alpha);
            let w = (w_max[j] + epsilon).powf(1.0 - alpha);
            (a / w).max(epsilon)
        })
        .collect()
}

/// Apply smoothing factors to activations in-place: `A_smooth[i,j] = A[i,j] / s[j]`.
pub fn smooth_activations(
    activations: &mut [f32],
    smooth_factors: &[f32],
    tokens: usize,
    in_features: usize,
) -> Result<(), DynQuantError> {
    if smooth_factors.len() != in_features {
        return Err(DynQuantError::FeatureDimMismatch {
            in_features,
            sf_len: smooth_factors.len(),
        });
    }
    let expected = tokens * in_features;
    if activations.len() != expected {
        return Err(DynQuantError::ShapeMismatch {
            expected,
            actual: activations.len(),
        });
    }
    for t in 0..tokens {
        for (j, &sf) in smooth_factors.iter().enumerate() {
            let idx = t * in_features + j;
            activations[idx] /= sf;
        }
    }
    Ok(())
}

/// Apply smoothing factors to weights in-place: `W_smooth[i,j] = W[i,j] * s[j]`.
pub fn smooth_weights(
    weights: &mut [f32],
    smooth_factors: &[f32],
    out_features: usize,
    in_features: usize,
) -> Result<(), DynQuantError> {
    if smooth_factors.len() != in_features {
        return Err(DynQuantError::FeatureDimMismatch {
            in_features,
            sf_len: smooth_factors.len(),
        });
    }
    let expected = out_features * in_features;
    if weights.len() != expected {
        return Err(DynQuantError::ShapeMismatch {
            expected,
            actual: weights.len(),
        });
    }
    for o in 0..out_features {
        for (j, &sf) in smooth_factors.iter().enumerate() {
            let idx = o * in_features + j;
            weights[idx] *= sf;
        }
    }
    Ok(())
}

// ─── W8A8 GEMV ────────────────────────────────────────────────────────────────

/// W8A8 matrix-vector multiply: quantize activation on-the-fly, then perform INT8 GEMV.
///
/// - `weight_i8`: shape `[out_size, in_size]` pre-quantized INT8 (row-major)
/// - `weight_scales`: shape `[out_size]` per-row dequant scales
/// - `activation`: shape `[in_size]` — dynamically quantized per-tensor
/// - Returns: shape `[out_size]` as f32
pub fn w8a8_matvec(
    weight_i8: &[i8],
    weight_scales: &[f32],
    activation: &[f32],
    out_size: usize,
    in_size: usize,
) -> Result<Vec<f32>, DynQuantError> {
    if activation.is_empty() {
        return Err(DynQuantError::EmptyInput);
    }
    if activation.len() != in_size {
        return Err(DynQuantError::ShapeMismatch {
            expected: in_size,
            actual: activation.len(),
        });
    }
    let expected_w = out_size * in_size;
    if weight_i8.len() != expected_w {
        return Err(DynQuantError::ShapeMismatch {
            expected: expected_w,
            actual: weight_i8.len(),
        });
    }
    if weight_scales.len() != out_size {
        return Err(DynQuantError::ShapeMismatch {
            expected: out_size,
            actual: weight_scales.len(),
        });
    }

    // Dynamically quantize activation per-tensor
    let act_quant = dynamic_quantize_int8(activation, DynamicScaleMode::MaxAbs);
    let act_scale = act_quant.scales.first().copied().unwrap_or(0.0);
    let act_i8 = &act_quant.data;

    let mut output = vec![0.0_f32; out_size];

    for o in 0..out_size {
        let row_start = o * in_size;
        let row_end = row_start + in_size;
        let row = &weight_i8[row_start..row_end];

        let mut acc = 0_i32;
        for (&w, &a) in row.iter().zip(act_i8.iter()) {
            acc += w as i32 * a as i32;
        }

        // Dequantize: result = acc * w_scale * act_scale
        output[o] = acc as f32 * weight_scales[o] * act_scale;
    }

    Ok(output)
}

// ─── Calibration statistics ───────────────────────────────────────────────────

/// Calibration statistics for choosing static quantization scales.
#[derive(Debug, Clone)]
pub struct CalibStats {
    /// Minimum value across all batches.
    pub min: f32,
    /// Maximum value across all batches.
    pub max: f32,
    /// Mean value across all batches.
    pub mean: f32,
    /// Standard deviation across all batches.
    pub std_dev: f32,
    /// 99th percentile of absolute values.
    pub p99: f32,
    /// Suggested quantization scale (p99 / 127.0 for INT8).
    pub suggested_scale: f32,
}

impl CalibStats {
    /// Collect calibration statistics from a batch of activation vectors.
    pub fn collect(batches: &[Vec<f32>]) -> Self {
        let all_values: Vec<f32> = batches.iter().flat_map(|b| b.iter().copied()).collect();

        if all_values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                p99: 0.0,
                suggested_scale: 0.0,
            };
        }

        let n = all_values.len();

        // min and max
        let min_val = all_values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // mean
        let sum: f32 = all_values.iter().sum();
        let mean_val = sum / n as f32;

        // std dev
        let variance: f32 = all_values
            .iter()
            .map(|&x| {
                let d = x - mean_val;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        let std_dev_val = variance.sqrt();

        // p99 of absolute values
        let mut abs_vals: Vec<f32> = all_values.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let p99_idx = ((0.99_f32 * n as f32).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let p99_val = abs_vals[p99_idx];

        let suggested = if p99_val > 0.0 {
            p99_val / 127.0
        } else {
            // Fallback: use max abs
            let max_abs = abs_vals.last().copied().unwrap_or(0.0);
            if max_abs > 0.0 {
                max_abs / 127.0
            } else {
                1.0 / 127.0
            }
        };

        Self {
            min: min_val,
            max: max_val,
            mean: mean_val,
            std_dev: std_dev_val,
            p99: p99_val,
            suggested_scale: suggested,
        }
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors from dynamic quantization operations.
#[derive(Debug, thiserror::Error)]
pub enum DynQuantError {
    /// Shape mismatch between expected and actual sizes.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// Input tensor is empty.
    #[error("empty input")]
    EmptyInput,

    /// Alpha value is out of the valid [0, 1] range.
    #[error("invalid alpha {0}: must be in [0, 1]")]
    InvalidAlpha(f32),

    /// Input feature dimension doesn't match smooth factors length.
    #[error("dimension mismatch: in_features {in_features}, smooth_factors {sf_len}")]
    FeatureDimMismatch { in_features: usize, sf_len: usize },
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_scale_max_abs_basic() {
        let data = [1.0_f32, -2.0, 0.5];
        let scale = compute_scale(&data, 127.0, DynamicScaleMode::MaxAbs);
        let expected = 2.0 / 127.0;
        assert!(
            (scale - expected).abs() < 1e-6,
            "scale={scale}, expected={expected}"
        );
    }

    #[test]
    fn test_compute_scale_zeros() {
        let data = [0.0_f32; 8];
        let scale = compute_scale(&data, 127.0, DynamicScaleMode::MaxAbs);
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn test_dequantize_roundtrip_int8() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
        let recon = qt.dequantize();
        let mae = quantization_mae(&data, &qt);
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(
            mae < 0.005 * max_abs,
            "MAE {mae} >= 0.5% of max_abs {max_abs}"
        );
        assert_eq!(recon.len(), data.len());
    }

    #[test]
    fn test_int4_range() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.3).collect();
        let qt = dynamic_quantize_int4(&data, DynamicScaleMode::MaxAbs);
        for &q in &qt.data {
            assert!((-7..=7).contains(&q), "INT4 value {q} out of range [-7, 7]");
        }
    }

    #[test]
    fn test_smooth_quant_config_new() {
        let cfg = SmoothQuantConfig::new(0.7);
        assert!((cfg.alpha - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_quant_config_default_alpha() {
        let cfg = SmoothQuantConfig::default_alpha();
        assert!((cfg.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calib_stats_basic() {
        let batches = vec![vec![1.0_f32, 2.0, 3.0], vec![-1.0_f32, 0.0, 4.0]];
        let stats = CalibStats::collect(&batches);
        assert!(stats.min <= stats.mean);
        assert!(stats.mean <= stats.max);
        assert!(stats.suggested_scale > 0.0);
    }
}
