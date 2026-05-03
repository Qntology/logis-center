//! Integration tests for Mixture of Depths (MoD).

use oxibonsai_model::layers::mixture_of_depths::{
    mixture_of_depths_forward, ModConfig, ModError, ModRouter, ModStats,
};

// ─── ModConfig ────────────────────────────────────────────────────────────────

#[test]
fn mod_config_default() {
    let cfg = ModConfig::default();
    assert!((cfg.capacity_factor - 0.5).abs() < 1e-6);
    assert_eq!(cfg.hidden_dim, 128);
    assert!(!cfg.normalize_router);
}

// ─── ModRouter: score_tokens ──────────────────────────────────────────────────

#[test]
fn mod_router_score_tokens_shape() {
    let cfg = ModConfig::new(0.5, 8);
    let router = ModRouter::new(cfg, 42);
    let seq_len = 16;
    let tokens: Vec<f32> = (0..seq_len * 8).map(|i| i as f32 * 0.01).collect();
    let scores = router
        .score_tokens(&tokens, seq_len)
        .expect("score_tokens failed");
    assert_eq!(scores.len(), seq_len);
}

// ─── ModRouter: select_tokens count ──────────────────────────────────────────

#[test]
fn mod_router_select_tokens_count() {
    let cf = 0.5_f32;
    let cfg = ModConfig::new(cf, 8);
    let router = ModRouter::new(cfg, 7);
    let seq_len = 10;
    let tokens: Vec<f32> = (0..seq_len * 8).map(|i| i as f32).collect();
    let scores = router.score_tokens(&tokens, seq_len).expect("score");
    let selected = router.select_tokens(&scores, seq_len);
    let expected_k = (cf * seq_len as f32).round() as usize;
    assert_eq!(selected.len(), expected_k);
}

// ─── ModRouter: capacity fraction ────────────────────────────────────────────

#[test]
fn mod_router_capacity_fraction() {
    let cfg = ModConfig::new(0.25, 16);
    let router = ModRouter::new(cfg, 1);
    assert_eq!(router.capacity(8), 2); // 0.25 * 8 = 2
    assert_eq!(router.capacity(10), 3); // 0.25 * 10 = 2.5 → rounds to 3
    assert_eq!(router.capacity(0), 0); // degenerate
}

// ─── ModRouter: selected tokens have higher scores than skipped ───────────────

#[test]
fn mod_router_select_top_scores() {
    let cfg = ModConfig::new(0.4, 4);
    let router = ModRouter::new(cfg, 99);
    let seq_len = 10;
    // Construct tokens so each token has a distinct clear score difference.
    let mut tokens = vec![0.0_f32; seq_len * 4];
    for i in 0..seq_len {
        // Make token i have a score proportional to i (last row highest).
        for j in 0..4 {
            tokens[i * 4 + j] = (i as f32 + 1.0) * 10.0;
        }
    }
    let scores = router.score_tokens(&tokens, seq_len).expect("score");
    let selected = router.select_tokens(&scores, seq_len);
    let k = router.capacity(seq_len);

    // Minimum score among selected must be >= maximum score among skipped.
    let all_indices: Vec<usize> = (0..seq_len).collect();
    let skipped: Vec<usize> = all_indices
        .into_iter()
        .filter(|i| !selected.contains(i))
        .collect();

    if !skipped.is_empty() && k < seq_len {
        let min_selected = selected
            .iter()
            .map(|&i| scores[i])
            .fold(f32::INFINITY, f32::min);
        let max_skipped = skipped
            .iter()
            .map(|&i| scores[i])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_selected >= max_skipped,
            "selected min={min_selected} < skipped max={max_skipped}"
        );
    }
}

// ─── mixture_of_depths_forward: output shape ─────────────────────────────────

#[test]
fn mixture_forward_output_shape() {
    let hidden_dim = 8;
    let seq_len = 12;
    let cfg = ModConfig::new(0.5, hidden_dim);
    let router = ModRouter::new(cfg, 3);
    let hidden: Vec<f32> = (0..seq_len * hidden_dim).map(|i| i as f32).collect();
    let out = mixture_of_depths_forward(&hidden, seq_len, hidden_dim, &router, |buf, count| {
        buf[..count * hidden_dim].to_vec()
    })
    .expect("forward failed");
    assert_eq!(out.len(), seq_len * hidden_dim);
}

// ─── mixture_of_depths_forward: identity layer → same as input ───────────────

#[test]
fn mixture_forward_identity_layer() {
    let hidden_dim = 4;
    let seq_len = 8;
    let cfg = ModConfig::new(1.0, hidden_dim); // all tokens processed
    let router = ModRouter::new(cfg, 5);
    let hidden: Vec<f32> = (0..seq_len * hidden_dim).map(|i| i as f32).collect();
    let out = mixture_of_depths_forward(&hidden, seq_len, hidden_dim, &router, |buf, _count| {
        buf.to_vec()
    })
    .expect("forward");
    assert_eq!(out, hidden);
}

// ─── mixture_of_depths_forward: non-identity changes selected tokens ──────────

#[test]
fn mixture_forward_nonidentity() {
    let hidden_dim = 4;
    let seq_len = 6;
    let cfg = ModConfig::new(0.5, hidden_dim);
    let router = ModRouter::new(cfg, 17);
    let hidden: Vec<f32> = vec![1.0; seq_len * hidden_dim];
    // Layer that negates every element.
    let out = mixture_of_depths_forward(&hidden, seq_len, hidden_dim, &router, |buf, _count| {
        buf.iter().map(|x| -x).collect()
    })
    .expect("forward");

    // At least one token must have been changed.
    let any_changed = out
        .iter()
        .zip(hidden.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_changed, "expected at least one token to change");
}

// ─── mixture_of_depths_forward: non-selected tokens equal input ───────────────

#[test]
fn mixture_forward_skipped_unchanged() {
    let hidden_dim = 4;
    let seq_len = 8;
    let cfg = ModConfig::new(0.25, hidden_dim); // only ~2 tokens processed
    let router = ModRouter::new(cfg, 21);
    let hidden: Vec<f32> = (0..seq_len * hidden_dim).map(|i| i as f32 + 1.0).collect();

    // Compute which indices will be selected.
    let scores = router.score_tokens(&hidden, seq_len).expect("score");
    let selected = router.select_tokens(&scores, seq_len);

    // Layer that negates (so we can detect changed tokens).
    let out = mixture_of_depths_forward(&hidden, seq_len, hidden_dim, &router, |buf, _count| {
        buf.iter().map(|x| -x).collect()
    })
    .expect("forward");

    // Non-selected tokens must be unchanged.
    for tok_idx in 0..seq_len {
        if !selected.contains(&tok_idx) {
            let start = tok_idx * hidden_dim;
            let end = start + hidden_dim;
            assert_eq!(
                &out[start..end],
                &hidden[start..end],
                "token {tok_idx} should be unchanged"
            );
        }
    }
}

// ─── ModStats ────────────────────────────────────────────────────────────────

#[test]
fn mod_stats_compute() {
    let stats = ModStats::compute(20, 10);
    assert_eq!(stats.seq_len, 20);
    assert_eq!(stats.tokens_processed, 10);
    assert_eq!(stats.tokens_skipped, 10);
    assert_eq!(stats.tokens_processed + stats.tokens_skipped, stats.seq_len);
}

#[test]
fn mod_stats_compute_reduction() {
    let stats = ModStats::compute(100, 50);
    assert!(
        (stats.compute_reduction - 0.5).abs() < 1e-5,
        "{}",
        stats.compute_reduction
    );
}

#[test]
fn mod_stats_summary_nonempty() {
    let stats = ModStats::compute(16, 8);
    let s = stats.summary();
    assert!(!s.is_empty());
    // Should contain key numbers.
    assert!(s.contains("16"), "summary should mention seq_len: {s}");
    assert!(
        s.contains('8'),
        "summary should mention processed count: {s}"
    );
}

// ─── Error paths ──────────────────────────────────────────────────────────────

#[test]
fn mod_error_empty_seq() {
    let cfg = ModConfig::new(0.5, 4);
    let router = ModRouter::new(cfg, 0);
    let result = router.score_tokens(&[], 0);
    assert!(
        matches!(result, Err(ModError::EmptySequence)),
        "expected EmptySequence, got: {result:?}"
    );
}

#[test]
fn mod_error_invalid_capacity() {
    let hidden_dim = 4;
    let seq_len = 4;
    // capacity_factor > 1.0 is invalid.
    let cfg = ModConfig::new(1.5, hidden_dim);
    let router = ModRouter::new(cfg, 0);
    let hidden = vec![1.0_f32; seq_len * hidden_dim];
    let result =
        mixture_of_depths_forward(&hidden, seq_len, hidden_dim, &router, |buf, _| buf.to_vec());
    assert!(
        matches!(result, Err(ModError::InvalidCapacity(_))),
        "expected InvalidCapacity, got: {result:?}"
    );
}
