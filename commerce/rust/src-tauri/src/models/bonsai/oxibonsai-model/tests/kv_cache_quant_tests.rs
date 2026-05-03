//! Integration tests for the INT8-quantized KV cache.
//!
//! All values are deterministic — no random number generator is used.
//! Test vectors are constructed with simple arithmetic expressions such as
//! `(i as f32) * 0.1 - 5.0`.

use oxibonsai_model::{
    dequantize_row_i8, quant_error_mae, quantize_row_i8, QuantKvError, QuantizedKvCache,
    QuantizedKvLayer,
};

// ─── Primitive helpers ────────────────────────────────────────────────────────

/// Test 1: exact range [127, -127, 0] → scale=1.0, i8=[127,-127,0]
#[test]
fn quantize_row_exact_range() {
    let row = vec![127.0_f32, -127.0_f32, 0.0_f32];
    let (q, scale) = quantize_row_i8(&row);
    assert!(
        (scale - 1.0_f32).abs() < 1e-5,
        "expected scale≈1.0, got {scale}"
    );
    assert_eq!(q[0], 127i8, "127 should quantize to 127");
    assert_eq!(q[1], -127i8, "-127 should quantize to -127");
    assert_eq!(q[2], 0i8, "0 should quantize to 0");
}

/// Test 2: typical activation values, MAE < 0.5% of max absolute value.
#[test]
fn quantize_row_roundtrip_small_error() {
    // Deterministic "random-ish" values in [-5, +5]
    let row: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let (q, scale) = quantize_row_i8(&row);
    let deq = dequantize_row_i8(&q, scale);

    let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let mae: f32 = row
        .iter()
        .zip(deq.iter())
        .map(|(o, r)| (o - r).abs())
        .sum::<f32>()
        / row.len() as f32;

    assert!(
        mae < max_abs * 0.005,
        "MAE {mae} exceeds 0.5% of max_abs {max_abs}"
    );
}

/// Test 3: scale near 0 (all-zero row) doesn't panic, returns zeros.
#[test]
fn dequantize_row_zero_scale() {
    let row = vec![0.0_f32; 8];
    let (q, scale) = quantize_row_i8(&row);
    // scale should be EPSILON, not 0 — must not panic
    let deq = dequantize_row_i8(&q, scale);
    assert_eq!(deq.len(), 8);
    for (i, &v) in deq.iter().enumerate() {
        assert!(v.abs() < 1e-5, "dequantized[{i}] should be ~0.0, got {v}");
    }
}

/// Test 4: quantizing an all-zero row has 0 MAE.
#[test]
fn quant_error_mae_perfect() {
    let row = vec![0.0_f32; 32];
    let (q, scale) = quantize_row_i8(&row);
    let mae = quant_error_mae(&row, &q, scale);
    assert_eq!(mae, 0.0, "all-zero row must have zero MAE");
}

/// Test 5: MAE < max(|x|) / 100 for typical activation values.
#[test]
fn quant_error_mae_bounded() {
    let row: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.2).collect();
    let (q, scale) = quantize_row_i8(&row);
    let mae = quant_error_mae(&row, &q, scale);
    let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(
        mae < max_abs / 100.0,
        "MAE {mae} must be < max_abs/100 = {}",
        max_abs / 100.0
    );
}

// ─── QuantizedKvLayer ─────────────────────────────────────────────────────────

/// Test 6: newly created layer has correct capacity and len=0.
#[test]
fn quant_layer_new() {
    let layer = QuantizedKvLayer::new(32, 4, 64);
    assert_eq!(layer.capacity, 32);
    assert_eq!(layer.num_kv_heads, 4);
    assert_eq!(layer.head_dim, 64);
    assert_eq!(layer.len, 0);
}

/// Test 7: push one token → len becomes 1.
#[test]
fn quant_layer_push_one_token() {
    let mut layer = QuantizedKvLayer::new(8, 2, 16);
    let keys: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let values: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.1).collect();
    layer.push(&keys, &values).expect("push should succeed");
    assert_eq!(layer.len, 1);
}

/// Test 8: push then get_key roundtrip, relative error < 1%.
#[test]
fn quant_layer_get_key_roundtrip() {
    let num_kv_heads = 4;
    let head_dim = 64;
    let mut layer = QuantizedKvLayer::new(16, num_kv_heads, head_dim);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| (i as f32) * 0.05 - 6.4)
        .collect();
    let values = vec![0.0_f32; num_kv_heads * head_dim];

    layer.push(&keys, &values).expect("push");

    for head in 0..num_kv_heads {
        let retrieved = layer.get_key(0, head).expect("get_key");
        assert_eq!(retrieved.len(), head_dim);

        let row_start = head * head_dim;
        let original = &keys[row_start..row_start + head_dim];
        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

        if max_abs < f32::EPSILON {
            continue; // trivially zero
        }

        let mae: f32 = original
            .iter()
            .zip(retrieved.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < max_abs * 0.01,
            "head {head}: MAE {mae} exceeds 1% of max_abs {max_abs}"
        );
    }
}

/// Test 9: push then get_value roundtrip, relative error < 1%.
#[test]
fn quant_layer_get_value_roundtrip() {
    let num_kv_heads = 4;
    let head_dim = 64;
    let mut layer = QuantizedKvLayer::new(16, num_kv_heads, head_dim);

    let keys = vec![0.0_f32; num_kv_heads * head_dim];
    let values: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| (i as f32) * 0.07 - 8.0)
        .collect();

    layer.push(&keys, &values).expect("push");

    for head in 0..num_kv_heads {
        let retrieved = layer.get_value(0, head).expect("get_value");
        let row_start = head * head_dim;
        let original = &values[row_start..row_start + head_dim];
        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

        if max_abs < f32::EPSILON {
            continue;
        }

        let mae: f32 = original
            .iter()
            .zip(retrieved.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < max_abs * 0.01,
            "value head {head}: MAE {mae} exceeds 1% of max_abs {max_abs}"
        );
    }
}

/// Test 10: get_keys_at returns all heads for a position.
#[test]
fn quant_layer_get_keys_at() {
    let num_kv_heads = 3;
    let head_dim = 8;
    let mut layer = QuantizedKvLayer::new(4, num_kv_heads, head_dim);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| i as f32 + 1.0)
        .collect();
    let values = vec![0.0_f32; num_kv_heads * head_dim];

    layer.push(&keys, &values).expect("push");

    let all_keys = layer.get_keys_at(0).expect("get_keys_at");
    assert_eq!(
        all_keys.len(),
        num_kv_heads * head_dim,
        "get_keys_at should return num_kv_heads * head_dim floats"
    );
}

/// Test 11: pushing beyond capacity returns CapacityExceeded.
#[test]
fn quant_layer_capacity_error() {
    let mut layer = QuantizedKvLayer::new(2, 1, 4);
    let keys = vec![1.0_f32; 4];
    let values = vec![2.0_f32; 4];

    layer.push(&keys, &values).expect("push 0");
    layer.push(&keys, &values).expect("push 1");

    let result = layer.push(&keys, &values);
    assert!(
        matches!(
            result,
            Err(QuantKvError::CapacityExceeded {
                capacity: 2,
                pos: 2
            })
        ),
        "expected CapacityExceeded, got {result:?}"
    );
}

/// Test 12: get at position >= len returns PositionOutOfRange.
#[test]
fn quant_layer_oob_error() {
    let layer = QuantizedKvLayer::new(4, 2, 8);
    // len is 0, so any position is out of range
    let result = layer.get_key(0, 0);
    assert!(
        matches!(result, Err(QuantKvError::PositionOutOfRange(0))),
        "expected PositionOutOfRange(0), got {result:?}"
    );
}

/// Test 13: INT8 memory is strictly less than FP32 memory (for non-trivial dims).
#[test]
fn quant_layer_memory_bytes() {
    let layer = QuantizedKvLayer::new(64, 8, 128);
    assert!(
        layer.memory_bytes() < layer.fp32_memory_bytes(),
        "INT8 memory {} should be less than FP32 memory {}",
        layer.memory_bytes(),
        layer.fp32_memory_bytes()
    );
}

/// Test 14: compression ratio is approximately 4× vs FP32.
///
/// For large enough dimensions the per-head scale overhead is negligible and
/// the ratio converges toward 4.0. We check it is in [3.5, 4.1].
#[test]
fn quant_layer_compression_ratio() {
    // Use large dimensions so scale overhead is < 12% of total.
    let layer = QuantizedKvLayer::new(512, 8, 128);
    let ratio = layer.compression_ratio();
    assert!(
        (3.5..=4.1_f32).contains(&ratio),
        "compression ratio {ratio} should be ≈ 4.0"
    );
}

// ─── QuantizedKvCache ─────────────────────────────────────────────────────────

/// Test 15: new cache has the correct layer count.
#[test]
fn quant_cache_new() {
    let cache = QuantizedKvCache::new(6, 32, 4, 64);
    assert_eq!(cache.num_layers, 6);
    assert_eq!(cache.num_kv_heads, 4);
    assert_eq!(cache.head_dim, 64);
}

/// Test 16: push one decode step → seq_len becomes 1.
#[test]
fn quant_cache_push_step() {
    let num_layers = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let mut cache = QuantizedKvCache::new(num_layers, 8, num_kv_heads, head_dim);

    let kv_size = num_kv_heads * head_dim;
    let all_keys: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| (0..kv_size).map(|i| i as f32 * 0.01 + l as f32).collect())
        .collect();
    let all_values: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| {
            (0..kv_size)
                .map(|i| -(i as f32) * 0.01 - l as f32)
                .collect()
        })
        .collect();

    cache.push_step(&all_keys, &all_values).expect("push_step");
    assert_eq!(cache.seq_len(), 1);
}

/// Test 17: total quantized memory is less than total FP32 memory.
#[test]
fn quant_cache_total_memory_compressed() {
    let cache = QuantizedKvCache::new(12, 256, 8, 128);
    assert!(
        cache.total_memory_bytes() < cache.total_fp32_memory_bytes(),
        "quantized total {} should be < FP32 total {}",
        cache.total_memory_bytes(),
        cache.total_fp32_memory_bytes()
    );
}

/// Test 18: write to all layers, read each layer correctly.
#[test]
fn quant_cache_get_across_layers() {
    let num_layers = 3;
    let num_kv_heads = 2;
    let head_dim = 8;
    let kv_size = num_kv_heads * head_dim;
    let mut cache = QuantizedKvCache::new(num_layers, 8, num_kv_heads, head_dim);

    // Each layer gets a distinctive signal: layer_idx * 10 + element_index
    let all_keys: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| (0..kv_size).map(|i| l as f32 * 10.0 + i as f32).collect())
        .collect();
    let all_values: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| {
            (0..kv_size)
                .map(|i| -(l as f32 * 10.0 + i as f32))
                .collect()
        })
        .collect();

    cache.push_step(&all_keys, &all_values).expect("push_step");

    for l in 0..num_layers {
        for h in 0..num_kv_heads {
            let key = cache.get_key(l, 0, h).expect("get_key");
            let val = cache.get_value(l, 0, h).expect("get_value");
            assert_eq!(key.len(), head_dim);
            assert_eq!(val.len(), head_dim);

            // Verify the first element is in the right ballpark for this layer
            let expected_first_key = l as f32 * 10.0 + (h * head_dim) as f32;
            assert!(
                (key[0] - expected_first_key).abs() < expected_first_key.abs() * 0.02 + 0.5,
                "layer {l} head {h}: key[0]={} expected≈{expected_first_key}",
                key[0]
            );
            // Values should be negated keys
            assert!(
                (val[0] + key[0]).abs() < key[0].abs() * 0.04 + 0.5,
                "layer {l} head {h}: val[0]+key[0] should be ≈0, got {}",
                val[0] + key[0]
            );
        }
    }
}

/// Test 19: accessing layer >= num_layers returns LayerOutOfRange.
#[test]
fn quant_cache_layer_oob_error() {
    let cache = QuantizedKvCache::new(4, 8, 2, 16);
    let result = cache.get_key(4, 0, 0);
    assert!(
        matches!(
            result,
            Err(QuantKvError::LayerOutOfRange {
                layer: 4,
                num_layers: 4
            })
        ),
        "expected LayerOutOfRange, got {result:?}"
    );
}

/// Test 20: push 10 tokens, all are readable.
#[test]
fn quant_cache_multiple_tokens() {
    let num_layers = 2;
    let num_kv_heads = 2;
    let head_dim = 16;
    let kv_size = num_kv_heads * head_dim;
    let num_tokens = 10;
    let mut cache = QuantizedKvCache::new(num_layers, 32, num_kv_heads, head_dim);

    for t in 0..num_tokens {
        let all_keys: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                (0..kv_size)
                    .map(|i| t as f32 * 100.0 + l as f32 * 10.0 + i as f32)
                    .collect()
            })
            .collect();
        let all_values: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                (0..kv_size)
                    .map(|i| -(t as f32 * 100.0 + l as f32 * 10.0 + i as f32))
                    .collect()
            })
            .collect();
        cache.push_step(&all_keys, &all_values).expect("push_step");
    }

    assert_eq!(
        cache.seq_len(),
        num_tokens,
        "seq_len should equal num_tokens"
    );

    // Verify every stored token/layer/head is accessible.
    for t in 0..num_tokens {
        for l in 0..num_layers {
            for h in 0..num_kv_heads {
                let key = cache
                    .get_key(l, t, h)
                    .expect("get_key should succeed for all stored positions");
                let val = cache
                    .get_value(l, t, h)
                    .expect("get_value should succeed for all stored positions");
                assert_eq!(key.len(), head_dim);
                assert_eq!(val.len(), head_dim);

                // Crude sanity check: key[0] should be positive, val[0] should be negative
                // (for t,l>0 the dominant term is t*100 > 0)
                if t > 0 {
                    assert!(
                        key[0] > 0.0,
                        "t={t} l={l} h={h}: key[0]={} should be positive",
                        key[0]
                    );
                    assert!(
                        val[0] < 0.0,
                        "t={t} l={l} h={h}: val[0]={} should be negative",
                        val[0]
                    );
                }
            }
        }
    }
}
