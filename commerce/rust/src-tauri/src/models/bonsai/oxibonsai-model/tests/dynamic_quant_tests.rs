//! Tests for dynamic activation quantization (dynamic_quant module).

use oxibonsai_model::dynamic_quant::{
    compute_scale, compute_smooth_factors, dynamic_quantize_int4, dynamic_quantize_int8,
    dynamic_quantize_int8_per_row, quantization_mae, smooth_activations, smooth_weights,
    w8a8_matvec, CalibStats, DynQuantFormat, DynamicScaleMode, SmoothQuantConfig,
};

// ─── 1. compute_scale_max_abs ─────────────────────────────────────────────────

#[test]
fn compute_scale_max_abs() {
    let data = [1.0_f32, -2.0, 0.5, -0.1];
    let scale = compute_scale(&data, 127.0, DynamicScaleMode::MaxAbs);
    let expected = 2.0_f32 / 127.0;
    assert!(
        (scale - expected).abs() < 1e-6,
        "scale={scale:.8}, expected={expected:.8}"
    );
}

// ─── 2. compute_scale_percentile ─────────────────────────────────────────────

#[test]
fn compute_scale_percentile() {
    // With an outlier, Percentile(0.99) should give a scale <= MaxAbs scale
    let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    data[99] = 1000.0; // extreme outlier

    let scale_max = compute_scale(&data, 127.0, DynamicScaleMode::MaxAbs);
    let scale_pct = compute_scale(&data, 127.0, DynamicScaleMode::Percentile(0.99));

    assert!(
        scale_pct <= scale_max,
        "Percentile scale {scale_pct} should be <= MaxAbs scale {scale_max}"
    );
}

// ─── 3. dynamic_quant_int8_dequantize_roundtrip ───────────────────────────────

#[test]
fn dynamic_quant_int8_dequantize_roundtrip() {
    // Linspace-like: 256 evenly spaced values in [-12.8, 12.8)
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
    let mae = quantization_mae(&data, &qt);
    let max_val = data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    assert_eq!(qt.data.len(), data.len(), "quantized length mismatch");
    assert!(
        mae < 0.005 * max_val,
        "MAE {mae:.6} should be < 0.5% of max_val {:.6}",
        0.005 * max_val
    );
}

// ─── 4. dynamic_quant_int8_memory ─────────────────────────────────────────────

#[test]
fn dynamic_quant_int8_memory() {
    let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
    let mem = qt.memory_bytes();
    // Must be less than len * 4 (original f32 size)
    assert!(
        mem < data.len() * 4,
        "memory_bytes {mem} should be < {} (f32 size)",
        data.len() * 4
    );
}

// ─── 5. dynamic_quant_int8_compression_ratio ──────────────────────────────────

#[test]
fn dynamic_quant_int8_compression_ratio() {
    // Large tensor: ratio approaches 4.0 as overhead of scale vector becomes negligible
    let data: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.001).collect();
    let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
    let ratio = qt.compression_ratio();
    // For 4096 elements: original=16384 bytes, quantized=4096+4=4100 bytes => ~3.995
    assert!(
        ratio > 3.9,
        "compression_ratio {ratio} should be approximately 4.0"
    );
}

// ─── 6. dynamic_quant_int8_all_zeros ─────────────────────────────────────────

#[test]
fn dynamic_quant_int8_all_zeros() {
    let data = vec![0.0_f32; 64];
    let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
    assert_eq!(qt.scales[0], 0.0, "scale should be 0 for all-zero input");
    for &q in &qt.data {
        assert_eq!(q, 0i8, "quantized value should be 0 for zero input");
    }
}

// ─── 7. dynamic_quant_int8_per_row_shape ─────────────────────────────────────

#[test]
fn dynamic_quant_int8_per_row_shape() {
    let rows = 8_usize;
    let cols = 16_usize;
    let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 * 0.01).collect();
    let qt = dynamic_quantize_int8_per_row(&data, rows, cols, DynamicScaleMode::MaxAbs);

    assert_eq!(
        qt.scales.len(),
        rows,
        "scale count should equal number of rows"
    );
    assert_eq!(
        qt.data.len(),
        rows * cols,
        "data length should be rows*cols"
    );
    assert_eq!(qt.format, DynQuantFormat::Int8PerRow);
}

// ─── 8. dynamic_quant_int8_per_row_dequantize ─────────────────────────────────

#[test]
fn dynamic_quant_int8_per_row_dequantize() {
    // Heterogeneous data: rows have very different magnitudes
    let rows = 4_usize;
    let cols = 32_usize;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        // Each row has magnitude 10^r: row 0 ~1.0, row 1 ~10.0, etc.
        let scale_factor = 10_f32.powi(r as i32);
        for c in 0..cols {
            data.push((c as f32 - 16.0) * 0.1 * scale_factor);
        }
    }

    let qt_per_row = dynamic_quantize_int8_per_row(&data, rows, cols, DynamicScaleMode::MaxAbs);
    let qt_per_tensor = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);

    let mae_per_row = quantization_mae(&data, &qt_per_row);
    let mae_per_tensor = quantization_mae(&data, &qt_per_tensor);

    assert!(
        mae_per_row <= mae_per_tensor + 1e-3,
        "per-row MAE {mae_per_row:.6} should be <= per-tensor MAE {mae_per_tensor:.6} for heterogeneous data"
    );
}

// ─── 9. dynamic_quant_int4_range ─────────────────────────────────────────────

#[test]
fn dynamic_quant_int4_range() {
    let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.5).collect();
    let qt = dynamic_quantize_int4(&data, DynamicScaleMode::MaxAbs);

    for &q in &qt.data {
        assert!(
            (-7..=7).contains(&q),
            "INT4 quantized value {q} out of range [-7, 7]"
        );
    }
    assert_eq!(qt.format, DynQuantFormat::Int4PerTensor);
}

// ─── 10. dynamic_quant_int4_dequantize ───────────────────────────────────────

#[test]
fn dynamic_quant_int4_dequantize() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
    let qt = dynamic_quantize_int4(&data, DynamicScaleMode::MaxAbs);
    let recon = qt.dequantize();

    assert_eq!(recon.len(), data.len(), "dequantized length mismatch");

    // MAE should be < 20% of (scale * 7), i.e. within one quantization step
    let scale = qt.scales[0];
    let tolerance = 0.20 * scale * 7.0;
    let mae = quantization_mae(&data, &qt);
    assert!(
        mae <= tolerance + 1e-5,
        "INT4 MAE {mae:.6} exceeds tolerance {tolerance:.6}"
    );
}

// ─── 11. quantization_mae_bounds ─────────────────────────────────────────────

#[test]
fn quantization_mae_bounds() {
    let data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.05).collect();
    let qt = dynamic_quantize_int8(&data, DynamicScaleMode::MaxAbs);
    let mae = quantization_mae(&data, &qt);
    let scale = qt.scales[0];

    // For INT8, quantization error is at most 0.5 * scale
    // MAE should be well below 1% of scale * 127
    let bound = 0.01 * scale * 127.0;
    assert!(
        mae < bound,
        "MAE {mae:.6} should be < 1% of scale*127 = {bound:.6}"
    );
}

// ─── 12. smooth_quant_config_alpha ───────────────────────────────────────────

#[test]
fn smooth_quant_config_alpha() {
    let cfg = SmoothQuantConfig::new(0.3);
    assert!(
        (cfg.alpha - 0.3).abs() < 1e-6,
        "alpha should be stored as 0.3, got {}",
        cfg.alpha
    );

    let cfg_default = SmoothQuantConfig::default_alpha();
    assert!(
        (cfg_default.alpha - 0.5).abs() < 1e-6,
        "default alpha should be 0.5, got {}",
        cfg_default.alpha
    );
}

// ─── 13. compute_smooth_factors_length ───────────────────────────────────────

#[test]
fn compute_smooth_factors_length() {
    let in_features = 8_usize;
    let tokens = 4_usize;
    let out_features = 6_usize;

    let activations: Vec<f32> = (0..(tokens * in_features))
        .map(|i| i as f32 * 0.1 + 0.1)
        .collect();
    let weights: Vec<f32> = (0..(out_features * in_features))
        .map(|i| i as f32 * 0.05 + 0.05)
        .collect();

    let config = SmoothQuantConfig::default_alpha();
    let factors = compute_smooth_factors(
        &activations,
        &weights,
        in_features,
        tokens,
        out_features,
        &config,
    );

    assert_eq!(
        factors.len(),
        in_features,
        "smooth factors length should equal in_features"
    );
}

// ─── 14. compute_smooth_factors_positive ─────────────────────────────────────

#[test]
fn compute_smooth_factors_positive() {
    let in_features = 4_usize;
    let tokens = 3_usize;
    let out_features = 5_usize;

    let activations: Vec<f32> = (0..(tokens * in_features))
        .map(|i| (i as f32 + 1.0) * 0.2)
        .collect();
    let weights: Vec<f32> = (0..(out_features * in_features))
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    let config = SmoothQuantConfig::default_alpha();
    let factors = compute_smooth_factors(
        &activations,
        &weights,
        in_features,
        tokens,
        out_features,
        &config,
    );

    for (j, &f) in factors.iter().enumerate() {
        assert!(f > 0.0, "smooth factor[{j}] = {f} should be positive");
    }
}

// ─── 15. smooth_activations_divides ──────────────────────────────────────────

#[test]
fn smooth_activations_divides() {
    let tokens = 2_usize;
    let in_features = 3_usize;
    let smooth_factors = [2.0_f32, 4.0, 8.0];
    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut activations = original.to_vec();

    smooth_activations(&mut activations, &smooth_factors, tokens, in_features)
        .expect("smooth_activations should succeed");

    for t in 0..tokens {
        for (j, &sf) in smooth_factors.iter().enumerate() {
            let idx = t * in_features + j;
            let expected = original[idx] / sf;
            assert!(
                (activations[idx] - expected).abs() < 1e-6,
                "activations[{idx}] = {} expected {} (divided by {})",
                activations[idx],
                expected,
                sf
            );
        }
    }
}

// ─── 16. smooth_weights_multiplies ───────────────────────────────────────────

#[test]
fn smooth_weights_multiplies() {
    let out_features = 2_usize;
    let in_features = 3_usize;
    let smooth_factors = [2.0_f32, 0.5, 3.0];
    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut weights = original.to_vec();

    smooth_weights(&mut weights, &smooth_factors, out_features, in_features)
        .expect("smooth_weights should succeed");

    for o in 0..out_features {
        for (j, &sf) in smooth_factors.iter().enumerate() {
            let idx = o * in_features + j;
            let expected = original[idx] * sf;
            assert!(
                (weights[idx] - expected).abs() < 1e-6,
                "weights[{idx}] = {} expected {} (multiplied by {})",
                weights[idx],
                expected,
                sf
            );
        }
    }
}

// ─── 17. w8a8_matvec_identity ────────────────────────────────────────────────

#[test]
fn w8a8_matvec_identity() {
    // Identity weight matrix 4x4 with scale 1.0/127 per row (quantize identity)
    let n = 4_usize;
    // Build identity as f32, quantize it
    let identity_f32: Vec<f32> = (0..n * n)
        .map(|i| if i / n == i % n { 1.0_f32 } else { 0.0 })
        .collect();

    // Pre-quantize the identity row by row: each row is a basis vector
    // q = round(w / scale), scale = max(|w|) / 127 = 1/127 per row
    let scale_per_row = 1.0_f32 / 127.0;
    let weight_i8: Vec<i8> = identity_f32
        .iter()
        .map(|&w| (w / scale_per_row).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let weight_scales = vec![scale_per_row; n];

    let activation = vec![1.0_f32, 2.0, 3.0, 4.0];
    let result = w8a8_matvec(&weight_i8, &weight_scales, &activation, n, n)
        .expect("w8a8_matvec should succeed");

    assert_eq!(result.len(), n, "output length should be out_size");
    for (i, (&r, &a)) in result.iter().zip(activation.iter()).enumerate() {
        assert!(
            (r - a).abs() < 0.2,
            "w8a8_matvec identity: result[{i}]={r:.4} should ≈ activation[{i}]={a:.4}"
        );
    }
}

// ─── 18. w8a8_matvec_shape ───────────────────────────────────────────────────

#[test]
fn w8a8_matvec_shape() {
    let out_size = 6_usize;
    let in_size = 4_usize;

    // Zero weight matrix
    let weight_i8 = vec![0_i8; out_size * in_size];
    let weight_scales = vec![0.01_f32; out_size];
    let activation = vec![1.0_f32; in_size];

    let result = w8a8_matvec(&weight_i8, &weight_scales, &activation, out_size, in_size)
        .expect("w8a8_matvec should succeed");

    assert_eq!(
        result.len(),
        out_size,
        "output length should equal out_size"
    );
}

// ─── 19. calib_stats_collect_nonempty ────────────────────────────────────────

#[test]
fn calib_stats_collect_nonempty() {
    let batches = vec![
        vec![0.5_f32, -0.5, 1.0],
        vec![2.0_f32, -2.0, 0.0],
        vec![0.1_f32, 0.9, -0.3],
    ];
    let stats = CalibStats::collect(&batches);

    // Basic sanity: stats are finite
    assert!(stats.min.is_finite(), "min should be finite");
    assert!(stats.max.is_finite(), "max should be finite");
    assert!(stats.mean.is_finite(), "mean should be finite");
    assert!(stats.std_dev >= 0.0, "std_dev should be non-negative");
    assert!(stats.p99 >= 0.0, "p99 should be non-negative");
    assert!(
        stats.suggested_scale > 0.0,
        "suggested_scale should be positive"
    );
}

// ─── 20. calib_stats_min_max ──────────────────────────────────────────────────

#[test]
fn calib_stats_min_max() {
    let batches = vec![
        vec![-5.0_f32, 0.0, 3.0, 1.5, -2.0],
        vec![4.0_f32, -1.0, 2.0],
    ];
    let stats = CalibStats::collect(&batches);

    assert!(
        stats.min <= stats.mean,
        "min ({}) should be <= mean ({})",
        stats.min,
        stats.mean
    );
    assert!(
        stats.mean <= stats.max,
        "mean ({}) should be <= max ({})",
        stats.mean,
        stats.max
    );
    assert_eq!(stats.min, -5.0, "min should be -5.0");
    assert_eq!(stats.max, 4.0, "max should be 4.0");
}

// ─── 21. calib_stats_suggested_scale_positive ────────────────────────────────

#[test]
fn calib_stats_suggested_scale_positive() {
    let batches = vec![vec![1.0_f32, -2.0, 0.5], vec![3.0_f32, -3.0, 1.5]];
    let stats = CalibStats::collect(&batches);
    assert!(
        stats.suggested_scale > 0.0,
        "suggested_scale should be positive, got {}",
        stats.suggested_scale
    );
    // Should be roughly p99 / 127
    let expected = stats.p99 / 127.0;
    assert!(
        (stats.suggested_scale - expected).abs() < 1e-4,
        "suggested_scale {} should be ~p99/127 = {}",
        stats.suggested_scale,
        expected
    );
}
