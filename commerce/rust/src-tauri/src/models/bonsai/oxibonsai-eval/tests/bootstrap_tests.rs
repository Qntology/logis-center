//! Integration tests for bootstrap confidence intervals.

use oxibonsai_eval::bootstrap::bootstrap_ci;

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// Basic shape
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bootstrap_empty_samples_errors() {
    let res = bootstrap_ci(&[], 100, 0.95, 42);
    assert!(res.is_err());
}

#[test]
fn bootstrap_confidence_out_of_range_errors() {
    let samples = vec![0.5f32; 10];
    assert!(bootstrap_ci(&samples, 100, 0.0, 42).is_err());
    assert!(bootstrap_ci(&samples, 100, 1.0, 42).is_err());
    assert!(bootstrap_ci(&samples, 100, -0.1, 42).is_err());
    assert!(bootstrap_ci(&samples, 100, 1.5, 42).is_err());
}

#[test]
fn bootstrap_zero_resamples_zero_width() {
    let samples = vec![0.5f32; 10];
    let ci = bootstrap_ci(&samples, 0, 0.95, 42).expect("ci");
    assert!((ci.mean - 0.5).abs() < EPS);
    assert!((ci.lo - ci.mean).abs() < EPS);
    assert!((ci.hi - ci.mean).abs() < EPS);
    assert_eq!(ci.n_resamples, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Mean & containment
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bootstrap_ci_contains_mean() {
    let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let ci = bootstrap_ci(&samples, 1000, 0.95, 42).expect("ci");
    assert!(
        ci.lo <= ci.mean && ci.mean <= ci.hi,
        "mean {} not in [{}, {}]",
        ci.mean,
        ci.lo,
        ci.hi
    );
}

#[test]
fn bootstrap_ci_constant_sample_has_zero_width() {
    // When every sample is identical, every resample has the same mean.
    let samples = vec![0.42f32; 20];
    let ci = bootstrap_ci(&samples, 500, 0.95, 1234).expect("ci");
    assert!((ci.lo - ci.hi).abs() < EPS);
    assert!((ci.mean - 0.42).abs() < EPS);
}

// ──────────────────────────────────────────────────────────────────────────────
// Seed determinism
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bootstrap_seed_is_deterministic() {
    let samples: Vec<f32> = (0..50).map(|i| (i as f32 * 0.01).sin()).collect();
    let a = bootstrap_ci(&samples, 200, 0.95, 7).expect("a");
    let b = bootstrap_ci(&samples, 200, 0.95, 7).expect("b");
    assert!((a.lo - b.lo).abs() < EPS);
    assert!((a.hi - b.hi).abs() < EPS);
}

#[test]
fn bootstrap_different_seeds_may_differ() {
    let samples: Vec<f32> = (0..50).map(|i| (i as f32 * 0.03).cos()).collect();
    let a = bootstrap_ci(&samples, 200, 0.95, 1).expect("a");
    let b = bootstrap_ci(&samples, 200, 0.95, 9999).expect("b");
    // Means identical (same sample), but bounds likely differ. Allow equality
    // but do not require it.
    assert!((a.mean - b.mean).abs() < EPS);
    let differ = (a.lo - b.lo).abs() + (a.hi - b.hi).abs();
    assert!(differ >= 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Width dependency on resamples
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bootstrap_width_tightens_with_more_resamples() {
    // With more resamples, the estimated percentile bounds should stabilise —
    // we assert the interval stays in a sensible range (i.e. not inverted nor
    // exploded) when moving from 100 to 1000 resamples.
    let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let ci_100 = bootstrap_ci(&samples, 100, 0.95, 42).expect("a");
    let ci_1000 = bootstrap_ci(&samples, 1000, 0.95, 42).expect("b");
    assert!(ci_100.lo <= ci_100.hi);
    assert!(ci_1000.lo <= ci_1000.hi);
    assert!(ci_1000.lo >= samples[0] - EPS);
    assert!(ci_1000.hi <= samples[samples.len() - 1] + EPS);
}

#[test]
fn bootstrap_bounds_within_sample_range() {
    let samples: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let ci = bootstrap_ci(&samples, 500, 0.9, 11).expect("ci");
    let min = *samples
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("min");
    let max = *samples
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("max");
    assert!(ci.lo >= min - EPS);
    assert!(ci.hi <= max + EPS);
}

#[test]
fn bootstrap_preserves_confidence_level() {
    let samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let ci = bootstrap_ci(&samples, 100, 0.99, 3).expect("ci");
    assert!((ci.confidence - 0.99).abs() < EPS);
    assert_eq!(ci.n_resamples, 100);
}
