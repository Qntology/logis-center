//! Integration tests for YaRN extended RoPE.

use oxibonsai_model::layers::yarn_rope::{
    apply_rope, apply_yarn_rope, LongRopeConfig, YarnConfig, YarnError, YarnFreqTable,
};

fn default_config() -> YarnConfig {
    YarnConfig::new(4096, 32768, 10000.0, 64)
}

#[test]
fn yarn_config_scale() {
    let cfg = default_config();
    let s = cfg.scale();
    assert!(
        (s - 8.0).abs() < 1e-5,
        "scale should be 32768/4096=8.0, got {s}"
    );
}

#[test]
fn yarn_config_attention_scale() {
    let cfg = default_config();
    let a = cfg.attention_scale();
    // s = 8 > e, so attention_scale < 1
    assert!(a < 1.0, "attention_scale must be < 1.0 for s=8, got {a}");
    // Expected: sqrt(1 / ln(8))
    let expected = (1.0_f32 / 8.0_f32.ln()).sqrt();
    assert!(
        (a - expected).abs() < 1e-5,
        "attention_scale = {a}, expected {expected}"
    );
}

#[test]
fn yarn_config_interpolation_factors_bounds() {
    let cfg = default_config();
    let factors = cfg.interpolation_factors();
    for (i, &f) in factors.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&f),
            "factor[{i}] = {f} is out of [0.0, 1.0]"
        );
    }
}

#[test]
fn yarn_config_interpolation_factors_length() {
    let cfg = default_config();
    let factors = cfg.interpolation_factors();
    assert_eq!(
        factors.len(),
        cfg.head_dim / 2,
        "factors length should be head_dim/2"
    );
}

#[test]
fn yarn_scaled_frequencies_positive() {
    let cfg = default_config();
    let freqs = cfg.scaled_frequencies();
    for (i, &f) in freqs.iter().enumerate() {
        assert!(f > 0.0, "freq[{i}] = {f} should be positive");
    }
}

#[test]
fn yarn_scaled_frequencies_monotone_decreasing() {
    let cfg = default_config();
    let freqs = cfg.scaled_frequencies();
    // The first frequency should be >= the last (overall trend: lower freq for higher dims)
    let first = freqs[0];
    let last = *freqs.last().expect("non-empty frequencies");
    assert!(
        first >= last,
        "frequencies should not increase overall: first={first}, last={last}"
    );
}

#[test]
fn apply_rope_identity_zero_pos() {
    let freqs = vec![0.01f32, 0.001f32];
    let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut k = vec![5.0f32, 6.0, 7.0, 8.0];
    let q_orig = q.clone();
    let k_orig = k.clone();

    apply_rope(&mut q, &mut k, 0, &freqs);

    // At pos=0: angle = 0*freq = 0 → cos(0)=1, sin(0)=0 → identity
    for i in 0..q.len() {
        assert!(
            (q[i] - q_orig[i]).abs() < 1e-5,
            "q[{i}] changed at pos=0: {} → {}",
            q_orig[i],
            q[i]
        );
        assert!(
            (k[i] - k_orig[i]).abs() < 1e-5,
            "k[{i}] changed at pos=0: {} → {}",
            k_orig[i],
            k[i]
        );
    }
}

#[test]
fn apply_rope_changes_values() {
    let freqs = vec![0.5f32, 0.1f32];
    let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut k = vec![1.0f32, 1.0, 1.0, 1.0];
    let q_before = q.clone();

    apply_rope(&mut q, &mut k, 5, &freqs);

    let changed = q
        .iter()
        .zip(q_before.iter())
        .any(|(after, before)| (after - before).abs() > 1e-5);
    assert!(changed, "apply_rope at pos=5 should change vector values");
}

#[test]
fn apply_yarn_rope_basic() {
    let cfg = default_config();
    let head_dim = cfg.head_dim;
    let mut q = vec![0.1f32; head_dim];
    let mut k = vec![0.2f32; head_dim];
    let result = apply_yarn_rope(&mut q, &mut k, 100, &cfg);
    assert!(result.is_ok(), "apply_yarn_rope returned error: {result:?}");
}

#[test]
fn yarn_freq_table_new() {
    let cfg = default_config();
    let table = YarnFreqTable::new(cfg);
    assert!(table.num_frequencies() > 0, "table should have frequencies");
}

#[test]
fn yarn_freq_table_apply_basic() {
    let cfg = default_config();
    let head_dim = cfg.head_dim;
    let table = YarnFreqTable::new(cfg);

    // At pos=0, rotation should be identity
    let mut q = vec![1.0f32; head_dim];
    let mut k = vec![2.0f32; head_dim];
    let q_orig = q.clone();
    let k_orig = k.clone();

    table
        .apply(&mut q, &mut k, 0)
        .expect("apply at pos=0 failed");

    for i in 0..head_dim {
        assert!((q[i] - q_orig[i]).abs() < 1e-5, "q[{i}] changed at pos=0");
        assert!((k[i] - k_orig[i]).abs() < 1e-5, "k[{i}] changed at pos=0");
    }
}

#[test]
fn yarn_freq_table_num_frequencies() {
    let cfg = default_config();
    let head_dim = cfg.head_dim;
    let table = YarnFreqTable::new(cfg);
    assert_eq!(
        table.num_frequencies(),
        head_dim / 2,
        "num_frequencies should be head_dim/2"
    );
}

#[test]
fn yarn_freq_table_effective_context() {
    let cfg = default_config();
    let extended = cfg.extended_max_position;
    let table = YarnFreqTable::new(cfg);
    assert_eq!(
        table.effective_context(),
        extended,
        "effective_context should equal extended_max_position"
    );
}

#[test]
fn yarn_freq_table_apply_batch() {
    let cfg = default_config();
    let head_dim = cfg.head_dim;
    let table = YarnFreqTable::new(cfg);

    let num_tokens = 4;
    let mut queries = vec![0.1f32; num_tokens * head_dim];
    let mut keys = vec![0.2f32; num_tokens * head_dim];
    let positions = vec![0usize, 10, 100, 1000];

    let result = table.apply_batch(&mut queries, &mut keys, &positions, head_dim);
    assert!(
        result.is_ok(),
        "apply_batch failed for multiple positions: {result:?}"
    );
}

#[test]
fn longrope_remap_start() {
    let cfg = LongRopeConfig::new(4096, 32768);
    let remapped = cfg.remap_position(0);
    assert!(
        remapped.abs() < 1e-5,
        "pos=0 should remap to 0.0, got {remapped}"
    );
}

#[test]
fn longrope_remap_end() {
    let cfg = LongRopeConfig::new(4096, 32768);
    // pos = extended_max_pos → should map to ≈ original_max_pos
    let remapped = cfg.remap_position(32768);
    assert!(
        (remapped - 4096.0).abs() < 1.0,
        "pos=extended_max_pos should remap to ~4096.0, got {remapped}"
    );
}

#[test]
fn longrope_effective_pos_bounded() {
    let cfg = LongRopeConfig::new(4096, 32768);
    // Every effective_pos must be strictly less than original_max_pos
    for pos in [0usize, 1000, 10000, 32768, 50000] {
        let ep = cfg.effective_pos(pos);
        assert!(
            ep < cfg.original_max_pos,
            "effective_pos({pos}) = {ep} should be < original_max_pos={}",
            cfg.original_max_pos
        );
    }
}

#[test]
fn yarn_error_odd_head_dim() {
    // head_dim=3 (odd) should return OddHeadDim error
    let cfg = YarnConfig::new(4096, 32768, 10000.0, 3);
    let mut q = vec![0.1f32; 3];
    let mut k = vec![0.1f32; 3];
    let result = apply_yarn_rope(&mut q, &mut k, 0, &cfg);
    assert!(
        matches!(result, Err(YarnError::OddHeadDim(3))),
        "expected OddHeadDim(3), got {result:?}"
    );
}

#[test]
fn yarn_error_position_exceeds_context() {
    let cfg = default_config();
    let mut q = vec![0.1f32; cfg.head_dim];
    let mut k = vec![0.1f32; cfg.head_dim];
    // Position exactly at extended_max_position should fail
    let result = apply_yarn_rope(&mut q, &mut k, cfg.extended_max_position, &cfg);
    assert!(
        matches!(result, Err(YarnError::PositionExceedsContext { .. })),
        "expected PositionExceedsContext, got {result:?}"
    );
}
