//! Integration tests for Flash Decoding.

use oxibonsai_model::layers::flash_decode::{
    flash_decode_multi_head, flash_decode_single_head, flash_vs_naive_error, FlashDecodeConfig,
    FlashDecodeError,
};

fn make_data(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let query: Vec<f32> = (0..head_dim).map(|i| 0.1 * i as f32).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| 0.05 * (i % 31) as f32 + 0.01)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| 0.02 * (i % 17) as f32 + 0.1)
        .collect();
    (query, keys, values)
}

#[test]
fn flash_decode_config_default() {
    let head_dim = 64usize;
    let cfg = FlashDecodeConfig::new(head_dim);
    let expected = 1.0_f32 / (head_dim as f32).sqrt();
    assert!(
        (cfg.scale - expected).abs() < 1e-6,
        "scale mismatch: {} vs {expected}",
        cfg.scale
    );
    assert_eq!(cfg.num_tiles, 4);
}

#[test]
fn flash_decode_single_head_matches_naive() {
    let head_dim = 16;
    let seq_len = 32;
    let (q, k, v) = make_data(seq_len, head_dim);
    let mae =
        flash_vs_naive_error(&q, &k, &v, seq_len, head_dim).expect("flash_vs_naive_error failed");
    assert!(
        mae < 1e-5,
        "MAE between flash and naive should be < 1e-5, got {mae}"
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
        "expected EmptyKv error, got {result:?}"
    );
}

#[test]
fn flash_decode_dim_mismatch_error() {
    let head_dim = 8;
    let config = FlashDecodeConfig::new(head_dim);
    // Query has extra elements
    let q = vec![0.1f32; head_dim + 4];
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
    // seq_len=1: softmax of single score = 1.0, so output == value
    let head_dim = 4;
    let config = FlashDecodeConfig::new(head_dim);
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let k = vec![0.5f32, 0.5, 0.5, 0.5];
    let v = vec![3.0f32, 1.0, 2.0, 4.0];

    let out = flash_decode_single_head(&q, &k, &v, 1, head_dim, &config)
        .expect("single token decode failed");

    for (i, (&o, &expected)) in out.iter().zip(v.iter()).enumerate() {
        assert!(
            (o - expected).abs() < 1e-5,
            "output[{i}] = {o}, expected {expected} (single token)"
        );
    }
}

#[test]
fn flash_decode_uniform_keys() {
    // With uniform keys, attention weights are equal → output = mean of values
    let head_dim = 4;
    let seq_len = 4;
    let config = FlashDecodeConfig::new(head_dim);
    let q = vec![0.1f32; head_dim];
    let k = vec![0.1f32; seq_len * head_dim]; // identical keys

    // Token t has all value dimensions = (t+1) as f32 → mean = 2.5
    let v: Vec<f32> = (0..seq_len)
        .flat_map(|t| vec![(t + 1) as f32; head_dim])
        .collect();

    let out = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config)
        .expect("uniform keys decode failed");

    let expected = 2.5_f32;
    for (i, &o) in out.iter().enumerate() {
        assert!(
            (o - expected).abs() < 1e-4,
            "output[{i}] = {o}, expected {expected} (uniform keys)"
        );
    }
}

#[test]
fn flash_decode_tile_count_1() {
    let head_dim = 8;
    let seq_len = 16;
    let config = FlashDecodeConfig::new(head_dim).with_num_tiles(1);
    let (q, k, v) = make_data(seq_len, head_dim);
    let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
    assert!(result.is_ok(), "num_tiles=1 should succeed: {result:?}");
}

#[test]
fn flash_decode_tile_count_many() {
    let head_dim = 8;
    let seq_len = 16;
    let config = FlashDecodeConfig::new(head_dim).with_num_tiles(8);
    let (q, k, v) = make_data(seq_len, head_dim);
    let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
    assert!(
        result.is_ok(),
        "num_tiles=8 with seq_len=16 should succeed: {result:?}"
    );
}

#[test]
fn flash_vs_naive_error_small() {
    let head_dim = 32;
    let seq_len = 64;
    let (q, k, v) = make_data(seq_len, head_dim);
    let mae =
        flash_vs_naive_error(&q, &k, &v, seq_len, head_dim).expect("flash_vs_naive_error failed");
    assert!(mae < 1e-4, "MAE too large: {mae}, expected < 1e-4");
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
    .expect("multi-head flash decode failed");

    assert_eq!(
        out.len(),
        num_heads * head_dim,
        "output has wrong shape: {} vs expected {}",
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
    .expect("multi-head flash decode failed");

    // Verify each head against a single-head decode (num_tiles=1 = naive)
    for h in 0..num_heads {
        let q_vec = &queries[h * head_dim..(h + 1) * head_dim];

        // Gather per-head K/V
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_head = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            let src = t * num_heads * head_dim + h * head_dim;
            let dst = t * head_dim;
            k_head[dst..dst + head_dim].copy_from_slice(&keys[src..src + head_dim]);
            v_head[dst..dst + head_dim].copy_from_slice(&values[src..src + head_dim]);
        }

        let naive_config = FlashDecodeConfig::new(head_dim).with_num_tiles(1);
        let naive_out =
            flash_decode_single_head(q_vec, &k_head, &v_head, seq_len, head_dim, &naive_config)
                .expect("naive single-head failed");

        let head_flash = &flash_out[h * head_dim..(h + 1) * head_dim];
        let mae: f32 = head_flash
            .iter()
            .zip(naive_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < 1e-4,
            "head {h}: multi-head vs single-head naive MAE = {mae}"
        );
    }
}

#[test]
fn combine_tiles_single_tile() {
    // Single tile: output should be exactly the tile's output
    // (test via 1-tile flash decode)
    let head_dim = 4;
    let seq_len = 8;
    let config_one = FlashDecodeConfig::new(head_dim).with_num_tiles(1);
    let config_many = FlashDecodeConfig::new(head_dim).with_num_tiles(4);
    let (q, k, v) = make_data(seq_len, head_dim);

    let out_one = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config_one)
        .expect("1-tile decode failed");
    let out_many = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config_many)
        .expect("4-tile decode failed");

    // Both should match closely (same mathematical result)
    let mae: f32 = out_one
        .iter()
        .zip(out_many.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / head_dim as f32;
    assert!(
        mae < 1e-4,
        "single vs multi tile outputs diverged: MAE={mae}"
    );
}

#[test]
fn flash_decode_long_sequence() {
    let head_dim = 16;
    let seq_len = 128;
    let config = FlashDecodeConfig::new(head_dim).with_num_tiles(8);
    let (q, k, v) = make_data(seq_len, head_dim);

    let result = flash_decode_single_head(&q, &k, &v, seq_len, head_dim, &config);
    assert!(result.is_ok(), "seq_len=128 should not panic: {result:?}");
    let out = result.expect("already checked");
    assert_eq!(out.len(), head_dim);
    for (i, &o) in out.iter().enumerate() {
        assert!(o.is_finite(), "output[{i}] = {o} is not finite");
    }
}
