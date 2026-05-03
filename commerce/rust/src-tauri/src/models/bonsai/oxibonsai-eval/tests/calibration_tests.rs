//! Integration tests for calibration metrics (ECE, Brier, NLL).

use oxibonsai_eval::calibration::{
    brier_score, calibration_all, expected_calibration_error, nll_from_logits,
};

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// ECE
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ece_perfect_calibration_high_confidence() {
    // Confidence=1, always correct → ECE = 0.
    let conf = vec![1.0f32; 10];
    let correct = vec![1u8; 10];
    let (ece, stats) = expected_calibration_error(&conf, &correct, 10).expect("ece should succeed");
    assert!(ece.abs() < EPS, "got {}", ece);
    assert_eq!(stats.len(), 10);
}

#[test]
fn ece_length_mismatch_errors() {
    let conf = vec![0.5f32; 3];
    let correct = vec![1u8; 4];
    let err = expected_calibration_error(&conf, &correct, 5);
    assert!(err.is_err());
}

#[test]
fn ece_empty_is_zero() {
    let conf: Vec<f32> = vec![];
    let correct: Vec<u8> = vec![];
    let (ece, stats) = expected_calibration_error(&conf, &correct, 10).expect("ece");
    assert_eq!(ece, 0.0);
    assert!(stats.is_empty());
}

#[test]
fn ece_uniform_known_value() {
    // Confidence = 0.5 everywhere, accuracy = 0 → |0 - 0.5| = 0.5.
    let conf = vec![0.5f32; 8];
    let correct = vec![0u8; 8];
    let (ece, _) = expected_calibration_error(&conf, &correct, 10).expect("ece");
    assert!((ece - 0.5).abs() < EPS, "expected ECE=0.5, got {}", ece);
}

#[test]
fn ece_overconfident_model() {
    // Confidence 0.9 everywhere, accuracy 0.5 → |acc - conf| = 0.4.
    // 10 samples, 5 correct, 5 wrong, all in bin [0.9, 1.0].
    let conf = vec![0.9f32; 10];
    let mut correct = vec![0u8; 10];
    for c in correct.iter_mut().take(5) {
        *c = 1;
    }
    let (ece, _) = expected_calibration_error(&conf, &correct, 10).expect("ece");
    assert!((ece - 0.4).abs() < 1e-3, "expected ~0.4, got {}", ece);
}

#[test]
fn ece_bin_boundary_last_bin_inclusive() {
    // Confidence=1.0 must land in the last bin [0.9, 1.0], not out-of-range.
    let conf = vec![1.0f32, 1.0];
    let correct = vec![1u8, 1];
    let (ece, stats) = expected_calibration_error(&conf, &correct, 10).expect("ece");
    assert!(ece.abs() < EPS);
    // Last bin should contain both samples.
    assert_eq!(stats.last().expect("bin").count, 2);
    // Earlier bins must be empty.
    for s in stats.iter().take(9) {
        assert_eq!(s.count, 0);
    }
}

#[test]
fn ece_in_unit_interval() {
    let conf = vec![0.2, 0.4, 0.6, 0.8, 0.9];
    let correct = vec![0u8, 1, 0, 1, 1];
    let (ece, _) = expected_calibration_error(&conf, &correct, 5).expect("ece");
    assert!((0.0..=1.0).contains(&ece));
}

#[test]
fn ece_n_bins_one_behaves_as_global_gap() {
    // With 1 bin, ECE = |mean(acc) - mean(conf)|.
    let conf = vec![0.2f32, 0.4, 0.6, 0.8];
    let correct = vec![1u8, 0, 1, 0];
    let (ece, _) = expected_calibration_error(&conf, &correct, 1).expect("ece");
    let expected = (0.5f32 - 0.5).abs();
    assert!((ece - expected).abs() < EPS, "got {}", ece);
}

// ──────────────────────────────────────────────────────────────────────────────
// Brier
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn brier_perfect_is_zero() {
    let probs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let labels = vec![0usize, 1];
    let b = brier_score(&probs, &labels).expect("brier");
    assert!(b.abs() < EPS);
}

#[test]
fn brier_is_in_unit_interval_two_class() {
    // Binary [0,1] prob vectors: sum-of-squares of two residuals ≤ 2 but we expect
    // realistic values within [0, 1].
    let probs = vec![vec![0.5f32, 0.5], vec![0.3, 0.7]];
    let labels = vec![0usize, 1];
    let b = brier_score(&probs, &labels).expect("brier");
    assert!((0.0..=1.0).contains(&b));
}

#[test]
fn brier_length_mismatch_errors() {
    let probs = vec![vec![0.5f32, 0.5]];
    let labels = vec![0usize, 1];
    assert!(brier_score(&probs, &labels).is_err());
}

#[test]
fn brier_label_out_of_range_errors() {
    let probs = vec![vec![0.5f32, 0.5]];
    let labels = vec![5usize];
    assert!(brier_score(&probs, &labels).is_err());
}

// ──────────────────────────────────────────────────────────────────────────────
// NLL
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn nll_perfectly_confident_is_near_zero() {
    // Extremely large logit on the correct class → NLL ≈ 0.
    let logits = vec![vec![1000.0f32, 0.0], vec![0.0, 1000.0]];
    let labels = vec![0usize, 1];
    let n = nll_from_logits(&logits, &labels).expect("nll");
    assert!(n.abs() < 1e-3, "got {}", n);
}

#[test]
fn nll_length_mismatch_errors() {
    let logits = vec![vec![1.0f32, 2.0]];
    let labels = vec![0usize, 1];
    assert!(nll_from_logits(&logits, &labels).is_err());
}

// ──────────────────────────────────────────────────────────────────────────────
// Combined
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn calibration_all_perfect() {
    let probs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let logits = vec![vec![1000.0f32, 0.0], vec![0.0, 1000.0]];
    let labels = vec![0usize, 1];
    let res = calibration_all(&probs, &logits, &labels, 5).expect("calibration_all");
    assert!(res.ece.abs() < EPS);
    assert!(res.brier.abs() < EPS);
    assert!(res.nll.abs() < 1e-3);
}
