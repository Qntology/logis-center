//! Tests for sparse attention patterns.

use oxibonsai_model::layers::sparse_attention::{
    memory_reduction, sparse_attention_forward, sparse_vs_dense_error, SparseAttentionMask,
    SparsePattern,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = seq_len * head_dim;
    let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) - 0.5).collect();
    let k: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) % 17) as f32 * 0.04 - 0.3)
        .collect();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i * 11 + 5) % 13) as f32 * 0.05 - 0.3)
        .collect();
    (q, k, v)
}

// ─── Mask construction tests ──────────────────────────────────────────────────

#[test]
fn dense_mask_full() {
    let seq_len = 8;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .expect("dense mask should build");
    assert_eq!(
        mask.nnz(),
        seq_len * seq_len,
        "dense mask must have seq_len^2 pairs"
    );
}

#[test]
fn local_window_nnz() {
    // window_size=3: interior tokens attend to 3, boundary tokens attend to 2.
    let seq_len = 10;
    let window_size = 3;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size })
        .expect("local window mask should build");
    // Exact count: 2 boundary * 2 + (seq_len - 2) * window_size
    let expected_nnz = 2 * 2 + (seq_len - 2) * window_size;
    assert_eq!(mask.nnz(), expected_nnz, "nnz mismatch for local window");
}

#[test]
fn local_window_boundary() {
    let seq_len = 6;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");

    // First token (q=0): attends only to 0, 1 (not negative)
    let first = mask.keys_for_query(0);
    for &k in first {
        assert!(k < seq_len, "boundary key out of range: {k}");
    }
    assert!(!first.is_empty(), "first token must attend to something");

    // Last token (q = seq_len-1): attends only to existing positions
    let last = mask.keys_for_query(seq_len - 1);
    for &k in last {
        assert!(k < seq_len, "boundary key out of range: {k}");
    }
    assert!(!last.is_empty(), "last token must attend to something");
}

#[test]
fn local_window_density() {
    let seq_len = 20;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");
    assert!(
        mask.density() < 1.0,
        "local window density must be < 1.0, got {}",
        mask.density()
    );
}

#[test]
fn bigbird_mask_builds() {
    let seq_len = 32;
    let result = SparseAttentionMask::build(
        seq_len,
        &SparsePattern::BigBird {
            window_size: 3,
            num_global_tokens: 2,
            num_random_connections: 2,
            seed: 12345,
        },
    );
    assert!(result.is_ok(), "BigBird mask should build without error");
}

#[test]
fn bigbird_global_attends_all() {
    let seq_len = 16;
    let num_global = 2;
    let mask = SparseAttentionMask::build(
        seq_len,
        &SparsePattern::BigBird {
            window_size: 3,
            num_global_tokens: num_global,
            num_random_connections: 0,
            seed: 99,
        },
    )
    .expect("BigBird mask should build");

    // Each of the first `num_global` tokens should attend to all positions
    for g in 0..num_global {
        let keys = mask.keys_for_query(g);
        assert_eq!(
            keys.len(),
            seq_len,
            "global token {g} must attend to all {seq_len} positions, got {}",
            keys.len()
        );
    }
}

#[test]
fn strided_mask_has_global_stride() {
    let seq_len = 12;
    let stride = 3;
    let mask = SparseAttentionMask::build(
        seq_len,
        &SparsePattern::Strided {
            window_size: 3,
            stride,
        },
    )
    .expect("strided mask should build");

    // Position 0, 3, 6, 9 should attend to all positions
    for q in (0..seq_len).step_by(stride) {
        let keys = mask.keys_for_query(q);
        assert_eq!(
            keys.len(),
            seq_len,
            "stride position {q} must attend to all {seq_len} positions, got {}",
            keys.len()
        );
    }
}

#[test]
fn mask_can_attend_local() {
    let seq_len = 10;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");

    // Each token attends to itself
    for q in 0..seq_len {
        assert!(mask.can_attend(q, q), "token {q} must attend to itself");
    }
}

#[test]
fn mask_cannot_attend_far() {
    let seq_len = 10;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");

    // Token 0 with window_size=3 attends to [0,1,2] only
    // Token 0 should NOT attend to token 5
    assert!(
        !mask.can_attend(0, 5),
        "token 0 should not attend to token 5 with window_size=3"
    );
    // Token 0 should NOT attend to token seq_len-1
    assert!(
        !mask.can_attend(0, seq_len - 1),
        "token 0 should not attend to token {} with window_size=3",
        seq_len - 1
    );
}

#[test]
fn mask_to_dense_shape() {
    let seq_len = 6;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .expect("dense mask should build");
    let dense = mask.to_dense();
    assert_eq!(dense.len(), seq_len, "to_dense must have seq_len rows");
    for row in &dense {
        assert_eq!(row.len(), seq_len, "each row must have seq_len columns");
    }
}

// ─── Forward pass tests ───────────────────────────────────────────────────────

#[test]
fn sparse_forward_dense_matches_naive() {
    // With a dense mask, sparse_attention_forward should produce the same
    // output as standard softmax attention.
    let seq_len = 4;
    let head_dim = 4;
    let (q, k, v) = make_qkv(seq_len, head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .expect("dense mask should build");
    let sparse_out = sparse_attention_forward(&q, &k, &v, seq_len, head_dim, &mask, scale)
        .expect("sparse forward should succeed");

    // Compute naive reference: for each query, full softmax attention
    let mut ref_out = vec![0.0f32; seq_len * head_dim];
    for qi in 0..seq_len {
        let q_vec = &q[qi * head_dim..(qi + 1) * head_dim];
        let mut scores: Vec<f32> = (0..seq_len)
            .map(|ki| {
                let k_vec = &k[ki * head_dim..(ki + 1) * head_dim];
                q_vec
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        // softmax
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_e = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_e += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum_e;
        }
        let out_row = &mut ref_out[qi * head_dim..(qi + 1) * head_dim];
        for (wi, &w) in scores.iter().enumerate() {
            let v_vec = &v[wi * head_dim..(wi + 1) * head_dim];
            for (o, &vv) in out_row.iter_mut().zip(v_vec.iter()) {
                *o += w * vv;
            }
        }
    }

    let mae: f32 = sparse_out
        .iter()
        .zip(ref_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (seq_len * head_dim) as f32;

    assert!(
        mae < 1e-5,
        "dense sparse attention should match naive attention, MAE={mae}"
    );
}

#[test]
fn sparse_forward_local_output_shape() {
    let seq_len = 8;
    let head_dim = 4;
    let (q, k, v) = make_qkv(seq_len, head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");
    let out = sparse_attention_forward(&q, &k, &v, seq_len, head_dim, &mask, scale)
        .expect("sparse forward should succeed");
    assert_eq!(out.len(), seq_len * head_dim, "output shape mismatch");
}

#[test]
fn sparse_vs_dense_error_dense_zero() {
    let seq_len = 6;
    let head_dim = 4;
    let (q, k, v) = make_qkv(seq_len, head_dim);
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .expect("dense mask should build");
    let err = sparse_vs_dense_error(&q, &k, &v, seq_len, head_dim, &mask)
        .expect("dense error computation should succeed");
    assert!(
        err < 1e-6,
        "dense mask vs dense reference error should be ~0, got {err}"
    );
}

// ─── Memory reduction tests ───────────────────────────────────────────────────

#[test]
fn memory_reduction_local() {
    let seq_len = 20;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
        .expect("local window mask should build");
    let reduction = memory_reduction(seq_len, &mask);
    assert!(
        reduction > 0.0,
        "local window should save some memory, got {reduction}"
    );
}

#[test]
fn memory_reduction_dense() {
    let seq_len = 10;
    let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .expect("dense mask should build");
    let reduction = memory_reduction(seq_len, &mask);
    assert!(
        reduction.abs() < 1e-6,
        "dense mask should have ~0 memory reduction, got {reduction}"
    );
}
