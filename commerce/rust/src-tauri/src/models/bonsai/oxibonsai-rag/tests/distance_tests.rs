//! Distance / similarity metric integration tests.
//!
//! Covers every [`Distance`] variant with the same axis of tests:
//!
//! - identity (`d(x, x)` behaviour),
//! - symmetry (`d(a, b) == d(b, a)`),
//! - unit-norm behaviour,
//! - triangle inequality (Euclidean only),
//! - zero-vector edge cases,
//! - NaN / Inf rejection,
//! - dimension-mismatch rejection.

use oxibonsai_rag::{Distance, RagError};

// ── helpers ────────────────────────────────────────────────────────────────

fn near(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// ── Cosine ─────────────────────────────────────────────────────────────────

#[test]
fn cosine_identity_unit_vector() {
    let a = vec![0.6, 0.8];
    let score = Distance::Cosine.compute(&a, &a).expect("compute");
    assert!(near(score, 1.0, 1e-5));
}

#[test]
fn cosine_symmetry() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let ab = Distance::Cosine.compute(&a, &b).expect("ab");
    let ba = Distance::Cosine.compute(&b, &a).expect("ba");
    assert!(near(ab, ba, 1e-6));
}

#[test]
fn cosine_zero_vector_returns_zero() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 1.0, 1.0];
    let c = Distance::Cosine.compute(&a, &b).expect("compute");
    assert!(near(c, 0.0, 1e-6));
}

#[test]
fn cosine_rejects_nan() {
    let a = vec![f32::NAN, 0.0];
    let b = vec![1.0, 0.0];
    assert!(matches!(
        Distance::Cosine.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

#[test]
fn cosine_rejects_inf() {
    let a = vec![f32::INFINITY, 0.0];
    let b = vec![1.0, 0.0];
    assert!(matches!(
        Distance::Cosine.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

// ── Euclidean ───────────────────────────────────────────────────────────────

#[test]
fn euclidean_identity_is_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let d = Distance::Euclidean.compute(&a, &a).expect("compute");
    assert!(near(d, 0.0, 1e-6));
}

#[test]
fn euclidean_symmetry() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let ab = Distance::Euclidean.compute(&a, &b).expect("ab");
    let ba = Distance::Euclidean.compute(&b, &a).expect("ba");
    assert!(near(ab, ba, 1e-6));
    assert!(near(ab, 5.0, 1e-5));
}

#[test]
fn euclidean_triangle_inequality() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 0.0];
    let c = vec![0.0, 4.0];
    let ab = Distance::Euclidean.compute(&a, &b).expect("ab");
    let bc = Distance::Euclidean.compute(&b, &c).expect("bc");
    let ac = Distance::Euclidean.compute(&a, &c).expect("ac");
    assert!(ac <= ab + bc + 1e-5);
}

#[test]
fn euclidean_zero_vector_is_norm() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0];
    let d = Distance::Euclidean.compute(&a, &b).expect("compute");
    assert!(near(d, 5.0, 1e-5));
}

#[test]
fn euclidean_rejects_nan() {
    let a = vec![f32::NAN];
    let b = vec![0.0];
    assert!(matches!(
        Distance::Euclidean.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

#[test]
fn euclidean_rejects_inf() {
    let a = vec![f32::NEG_INFINITY];
    let b = vec![0.0];
    assert!(matches!(
        Distance::Euclidean.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

// ── DotProduct ───────────────────────────────────────────────────────────────

#[test]
fn dot_product_identity_is_squared_norm() {
    let a = vec![2.0, 0.0];
    let d = Distance::DotProduct.compute(&a, &a).expect("compute");
    assert!(near(d, 4.0, 1e-5));
}

#[test]
fn dot_product_symmetry() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let ab = Distance::DotProduct.compute(&a, &b).expect("ab");
    let ba = Distance::DotProduct.compute(&b, &a).expect("ba");
    assert!(near(ab, ba, 1e-6));
}

#[test]
fn dot_product_zero_vector() {
    let a = vec![0.0, 0.0];
    let b = vec![5.0, 5.0];
    let d = Distance::DotProduct.compute(&a, &b).expect("compute");
    assert!(near(d, 0.0, 1e-6));
}

#[test]
fn dot_product_rejects_nan() {
    let a = vec![f32::NAN];
    let b = vec![1.0];
    assert!(matches!(
        Distance::DotProduct.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

#[test]
fn dot_product_rejects_inf() {
    let a = vec![f32::INFINITY];
    let b = vec![1.0];
    assert!(matches!(
        Distance::DotProduct.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

// ── Angular ──────────────────────────────────────────────────────────────────

#[test]
fn angular_identity_is_zero() {
    let a = vec![1.0, 1.0];
    let d = Distance::Angular.compute(&a, &a).expect("compute");
    assert!(d <= 1e-5);
}

#[test]
fn angular_opposite_is_one() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let d = Distance::Angular.compute(&a, &b).expect("compute");
    assert!(near(d, 1.0, 1e-5));
}

#[test]
fn angular_symmetry() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let ab = Distance::Angular.compute(&a, &b).expect("ab");
    let ba = Distance::Angular.compute(&b, &a).expect("ba");
    assert!(near(ab, ba, 1e-6));
}

#[test]
fn angular_range_within_unit_interval() {
    let a = vec![1.0, 0.0];
    let b = vec![0.5, 0.5];
    let d = Distance::Angular.compute(&a, &b).expect("compute");
    assert!((0.0..=1.0).contains(&d), "got {d}");
}

#[test]
fn angular_rejects_nan() {
    let a = vec![f32::NAN];
    let b = vec![1.0];
    assert!(matches!(
        Distance::Angular.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

#[test]
fn angular_rejects_inf() {
    let a = vec![f32::INFINITY];
    let b = vec![1.0];
    assert!(matches!(
        Distance::Angular.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

// ── Hamming ──────────────────────────────────────────────────────────────────

#[test]
fn hamming_identity_is_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let d = Distance::Hamming.compute(&a, &a).expect("compute");
    assert_eq!(d, 0.0);
}

#[test]
fn hamming_symmetry() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let ab = Distance::Hamming.compute(&a, &b).expect("ab");
    let ba = Distance::Hamming.compute(&b, &a).expect("ba");
    assert!(near(ab, ba, 1e-6));
}

#[test]
fn hamming_bound() {
    // Each f32 contributes at most 32 bit differences.  Use negative zero
    // vs positive zero as a tiny-but-non-zero bit difference that is
    // guaranteed to be finite.
    let a = vec![0.0_f32, 0.0];
    let b = vec![-0.0_f32, -0.0];
    let d = Distance::Hamming.compute(&a, &b).expect("compute");
    assert!((0.0..=32.0).contains(&d), "got {d}");
}

#[test]
fn hamming_rejects_nan() {
    let a = vec![f32::NAN];
    let b = vec![1.0];
    assert!(matches!(
        Distance::Hamming.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

#[test]
fn hamming_rejects_inf() {
    let a = vec![f32::INFINITY];
    let b = vec![1.0];
    assert!(matches!(
        Distance::Hamming.compute(&a, &b),
        Err(RagError::NonFinite)
    ));
}

// ── Dim mismatch / zero-length ──────────────────────────────────────────────

#[test]
fn dim_mismatch_rejected_for_each_metric() {
    let a = vec![1.0];
    let b = vec![1.0, 2.0];
    for metric in [
        Distance::Cosine,
        Distance::Euclidean,
        Distance::DotProduct,
        Distance::Angular,
        Distance::Hamming,
    ] {
        let err = metric.compute(&a, &b);
        assert!(
            matches!(err, Err(RagError::DimensionMismatch { .. })),
            "metric {metric:?}"
        );
    }
}

#[test]
fn empty_inputs_rejected_for_each_metric() {
    let empty: Vec<f32> = Vec::new();
    for metric in [
        Distance::Cosine,
        Distance::Euclidean,
        Distance::DotProduct,
        Distance::Angular,
        Distance::Hamming,
    ] {
        let err = metric.compute(&empty, &empty);
        assert!(
            matches!(err, Err(RagError::DimensionMismatch { .. })),
            "metric {metric:?}"
        );
    }
}

// ── to_score polarity ────────────────────────────────────────────────────────

#[test]
fn similarity_metrics_preserve_sign() {
    assert!(near(Distance::Cosine.to_score(0.5), 0.5, 1e-6));
    assert!(near(Distance::DotProduct.to_score(7.0), 7.0, 1e-6));
}

#[test]
fn distance_metrics_negate_score() {
    assert!(near(Distance::Euclidean.to_score(2.0), -2.0, 1e-6));
    assert!(near(Distance::Angular.to_score(0.4), -0.4, 1e-6));
    assert!(near(Distance::Hamming.to_score(3.0), -3.0, 1e-6));
}
