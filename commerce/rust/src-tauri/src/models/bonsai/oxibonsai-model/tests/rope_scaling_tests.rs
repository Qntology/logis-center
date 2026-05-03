//! Integration tests for RoPE scaling strategies.
//!
//! Covers all 16 required test cases:
//!  1.  no_scaling_standard_freqs
//!  2.  linear_scaling_divides_freqs
//!  3.  linear_scaling_scale_1_unchanged
//!  4.  dynamic_ntk_longer_seq_higher_base
//!  5.  dynamic_ntk_at_orig_len_unchanged
//!  6.  llama31_freqs_length
//!  7.  llama31_freqs_positive
//!  8.  llama31_scale_1_unchanged
//!  9.  longrope_freqs_uses_factors
//! 10.  longrope_wrong_factor_count_error
//! 11.  apply_rope_zero_pos_identity
//! 12.  apply_rope_changes_at_pos1
//! 13.  apply_rope_invalid_head_dim_error
//! 14.  freq_stats_min_max_ordering
//! 15.  freq_stats_effective_context_positive
//! 16.  compute_freqs_invalid_dim_error

use oxibonsai_model::{
    apply_rope_with_freqs, compute_rope_frequencies, dynamic_ntk_base, llama31_frequencies,
    FreqStats, RopeScalingError, RopeScalingStrategy,
};

const BASE: f32 = 10_000.0;
const HEAD_DIM: usize = 64;
const ORIG_MAX_POS: usize = 4_096;

// Helper: reference implementation of standard RoPE frequencies.
fn standard_freqs(head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| 1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

// ── Test 1: no_scaling_standard_freqs ────────────────────────────────────────

#[test]
fn no_scaling_standard_freqs() {
    let freqs = compute_rope_frequencies(HEAD_DIM, BASE, &RopeScalingStrategy::None, ORIG_MAX_POS)
        .expect("None strategy must succeed");

    let expected = standard_freqs(HEAD_DIM, BASE);
    assert_eq!(freqs.len(), expected.len(), "length mismatch");
    for (i, (got, exp)) in freqs.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "freq[{i}]: got {got}, expected {exp}"
        );
    }
}

// ── Test 2: linear_scaling_divides_freqs ─────────────────────────────────────

#[test]
fn linear_scaling_divides_freqs() {
    let scale = 4.0_f32;
    let freqs = compute_rope_frequencies(
        HEAD_DIM,
        BASE,
        &RopeScalingStrategy::Linear {
            scale_factor: scale,
        },
        ORIG_MAX_POS,
    )
    .expect("Linear strategy must succeed");

    let standard = standard_freqs(HEAD_DIM, BASE);
    for (i, (got, std_f)) in freqs.iter().zip(standard.iter()).enumerate() {
        let expected = std_f / scale;
        assert!(
            (got - expected).abs() < 1e-6,
            "freq[{i}]: got {got}, expected {expected}"
        );
    }
}

// ── Test 3: linear_scaling_scale_1_unchanged ──────────────────────────────────

#[test]
fn linear_scaling_scale_1_unchanged() {
    let freqs_linear = compute_rope_frequencies(
        HEAD_DIM,
        BASE,
        &RopeScalingStrategy::Linear { scale_factor: 1.0 },
        ORIG_MAX_POS,
    )
    .expect("Linear scale=1 must succeed");

    let freqs_none = standard_freqs(HEAD_DIM, BASE);
    assert_eq!(freqs_linear.len(), freqs_none.len(), "length mismatch");
    for (i, (a, b)) in freqs_linear.iter().zip(freqs_none.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "freq[{i}]: linear scale=1 got {a}, standard got {b}"
        );
    }
}

// ── Test 4: dynamic_ntk_longer_seq_higher_base ────────────────────────────────

#[test]
fn dynamic_ntk_longer_seq_higher_base() {
    let base_orig = dynamic_ntk_base(BASE, HEAD_DIM, ORIG_MAX_POS, ORIG_MAX_POS);
    let base_2x = dynamic_ntk_base(BASE, HEAD_DIM, ORIG_MAX_POS, ORIG_MAX_POS * 2);
    let base_4x = dynamic_ntk_base(BASE, HEAD_DIM, ORIG_MAX_POS, ORIG_MAX_POS * 4);

    assert!(
        base_2x > base_orig,
        "2× seq should give higher base than original: {base_2x} vs {base_orig}"
    );
    assert!(
        base_4x > base_2x,
        "4× seq should give higher base than 2× seq: {base_4x} vs {base_2x}"
    );
}

// ── Test 5: dynamic_ntk_at_orig_len_unchanged ────────────────────────────────

#[test]
fn dynamic_ntk_at_orig_len_unchanged() {
    let effective = dynamic_ntk_base(BASE, HEAD_DIM, ORIG_MAX_POS, ORIG_MAX_POS);
    assert!(
        (effective - BASE).abs() < 1e-3,
        "at original length, effective base must equal base: {effective} vs {BASE}"
    );

    let freqs_ntk = compute_rope_frequencies(
        HEAD_DIM,
        BASE,
        &RopeScalingStrategy::DynamicNtk {
            original_max_position: ORIG_MAX_POS,
            base: BASE,
        },
        ORIG_MAX_POS,
    )
    .expect("DynamicNtk at orig len must succeed");

    let freqs_none = standard_freqs(HEAD_DIM, BASE);
    for (i, (a, b)) in freqs_ntk.iter().zip(freqs_none.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "freq[{i}]: NTK at orig len got {a}, standard got {b}"
        );
    }
}

// ── Test 6: llama31_freqs_length ──────────────────────────────────────────────

#[test]
fn llama31_freqs_length() {
    let freqs = llama31_frequencies(HEAD_DIM, BASE, ORIG_MAX_POS * 2, 8.0, 1.0, 4.0);
    assert_eq!(
        freqs.len(),
        HEAD_DIM / 2,
        "llama31_frequencies must return head_dim/2 values, got {}",
        freqs.len()
    );
}

// ── Test 7: llama31_freqs_positive ────────────────────────────────────────────

#[test]
fn llama31_freqs_positive() {
    let freqs = llama31_frequencies(HEAD_DIM, BASE, ORIG_MAX_POS * 2, 8.0, 1.0, 4.0);
    for (i, &f) in freqs.iter().enumerate() {
        assert!(f > 0.0, "freq[{i}] = {f} is not positive");
    }
}

// ── Test 8: llama31_scale_1_unchanged ─────────────────────────────────────────

#[test]
fn llama31_scale_1_unchanged() {
    let freqs_scaled = llama31_frequencies(HEAD_DIM, BASE, ORIG_MAX_POS, 1.0, 1.0, 4.0);
    let freqs_standard = standard_freqs(HEAD_DIM, BASE);

    for (i, (got, exp)) in freqs_scaled.iter().zip(freqs_standard.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "freq[{i}]: scale=1 got {got}, standard got {exp}"
        );
    }
}

// ── Test 9: longrope_freqs_uses_factors ───────────────────────────────────────

#[test]
fn longrope_freqs_uses_factors() {
    let half = HEAD_DIM / 2;
    let factors: Vec<f32> = (0..half).map(|i| 1.0 + i as f32 * 0.1).collect();

    let freqs = compute_rope_frequencies(
        HEAD_DIM,
        BASE,
        &RopeScalingStrategy::LongRope {
            rescale_factors: factors.clone(),
            original_max_position: ORIG_MAX_POS,
        },
        ORIG_MAX_POS * 2,
    )
    .expect("LongRope with valid factors must succeed");

    let standard = standard_freqs(HEAD_DIM, BASE);
    for (i, ((got, std_f), &r)) in freqs
        .iter()
        .zip(standard.iter())
        .zip(factors.iter())
        .enumerate()
    {
        let expected = std_f / r;
        assert!(
            (got - expected).abs() < 1e-6,
            "freq[{i}]: got {got}, expected {expected} (std={std_f}, factor={r})"
        );
    }
}

// ── Test 10: longrope_wrong_factor_count_error ────────────────────────────────

#[test]
fn longrope_wrong_factor_count_error() {
    let wrong_factors = vec![1.0_f32; 5]; // need head_dim/2 = 32, not 5
    let result = compute_rope_frequencies(
        HEAD_DIM,
        BASE,
        &RopeScalingStrategy::LongRope {
            rescale_factors: wrong_factors,
            original_max_position: ORIG_MAX_POS,
        },
        ORIG_MAX_POS * 2,
    );
    assert!(
        matches!(
            result,
            Err(RopeScalingError::RescaleFactorLengthMismatch {
                got: 5,
                expected: 32
            })
        ),
        "expected RescaleFactorLengthMismatch {{got:5, expected:32}}, got: {result:?}"
    );
}

// ── Test 11: apply_rope_zero_pos_identity ─────────────────────────────────────

#[test]
fn apply_rope_zero_pos_identity() {
    let freqs = standard_freqs(HEAD_DIM, BASE);
    let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.1).collect();
    let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.2).collect();
    let q_orig = q.clone();
    let k_orig = k.clone();

    apply_rope_with_freqs(&mut q, &mut k, 0, &freqs).expect("apply at pos=0 must succeed");

    for i in 0..HEAD_DIM {
        assert!(
            (q[i] - q_orig[i]).abs() < 1e-5,
            "q[{i}] must be unchanged at pos=0: {} → {}",
            q_orig[i],
            q[i]
        );
        assert!(
            (k[i] - k_orig[i]).abs() < 1e-5,
            "k[{i}] must be unchanged at pos=0: {} → {}",
            k_orig[i],
            k[i]
        );
    }
}

// ── Test 12: apply_rope_changes_at_pos1 ───────────────────────────────────────

#[test]
fn apply_rope_changes_at_pos1() {
    let freqs = standard_freqs(HEAD_DIM, BASE);
    let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.5).collect();
    let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.3).collect();
    let q_orig = q.clone();

    apply_rope_with_freqs(&mut q, &mut k, 1, &freqs).expect("apply at pos=1 must succeed");

    let changed = q
        .iter()
        .zip(q_orig.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(
        changed,
        "apply_rope_with_freqs at pos=1 must change at least one value"
    );
}

// ── Test 13: apply_rope_invalid_head_dim_error ────────────────────────────────

#[test]
fn apply_rope_invalid_head_dim_error() {
    // Empty freqs → half_dim == 0 → InvalidHeadDim(0)
    let freqs: Vec<f32> = vec![];
    let mut q = vec![1.0_f32];
    let mut k = vec![1.0_f32];
    let result = apply_rope_with_freqs(&mut q, &mut k, 0, &freqs);
    assert!(
        matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
        "empty freqs must return InvalidHeadDim(0), got: {result:?}"
    );
}

// ── Test 14: freq_stats_min_max_ordering ──────────────────────────────────────

#[test]
fn freq_stats_min_max_ordering() {
    let freqs = standard_freqs(HEAD_DIM, BASE);
    let stats = FreqStats::compute(&freqs);
    assert!(
        stats.min_freq <= stats.mean_freq,
        "min ({}) must be <= mean ({})",
        stats.min_freq,
        stats.mean_freq
    );
    assert!(
        stats.mean_freq <= stats.max_freq,
        "mean ({}) must be <= max ({})",
        stats.mean_freq,
        stats.max_freq
    );
    assert!(
        stats.min_freq <= stats.max_freq,
        "min ({}) must be <= max ({})",
        stats.min_freq,
        stats.max_freq
    );
}

// ── Test 15: freq_stats_effective_context_positive ───────────────────────────

#[test]
fn freq_stats_effective_context_positive() {
    let freqs = standard_freqs(HEAD_DIM, BASE);
    let stats = FreqStats::compute(&freqs);
    assert!(
        stats.effective_context > 0.0,
        "effective_context must be positive, got {}",
        stats.effective_context
    );
    // Sanity: effective context must be 1/min_freq, which for base=10000 dim=64
    // gives a large number.
    assert!(
        stats.effective_context > 1.0,
        "effective_context should be >> 1 for typical RoPE params, got {}",
        stats.effective_context
    );
}

// ── Test 16: compute_freqs_invalid_dim_error ──────────────────────────────────

#[test]
fn compute_freqs_invalid_dim_error() {
    // head_dim = 0
    let result = compute_rope_frequencies(0, BASE, &RopeScalingStrategy::None, ORIG_MAX_POS);
    assert!(
        matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
        "head_dim=0 must return InvalidHeadDim(0), got: {result:?}"
    );

    // head_dim = 3 (odd)
    let result_odd = compute_rope_frequencies(3, BASE, &RopeScalingStrategy::None, ORIG_MAX_POS);
    assert!(
        matches!(result_odd, Err(RopeScalingError::InvalidHeadDim(3))),
        "head_dim=3 must return InvalidHeadDim(3), got: {result_odd:?}"
    );
}

// ── Bonus: summary string is non-empty ────────────────────────────────────────

#[test]
fn freq_stats_summary_non_empty() {
    let freqs = standard_freqs(HEAD_DIM, BASE);
    let stats = FreqStats::compute(&freqs);
    let s = stats.summary();
    assert!(!s.is_empty(), "summary() must return a non-empty string");
}
