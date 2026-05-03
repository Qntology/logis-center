//! Integration tests for streaming / online evaluation.

use oxibonsai_eval::streaming::{OnlineAccuracy, OnlinePerplexity};

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// OnlinePerplexity
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn online_perplexity_empty_is_infinity() {
    let p = OnlinePerplexity::new();
    assert_eq!(p.current(), f32::INFINITY);
    assert_eq!(p.tokens(), 0);
}

#[test]
fn online_perplexity_perfect_prediction_one() {
    // log_p = 0 (probability = 1) → PPL = exp(0) = 1.
    let mut p = OnlinePerplexity::new();
    for _ in 0..10 {
        p.push(0.0);
    }
    assert!((p.current() - 1.0).abs() < EPS, "got {}", p.current());
    assert_eq!(p.tokens(), 10);
}

#[test]
fn online_perplexity_batch_equivalent_to_streaming() {
    // Feeding one by one must equal feeding as a chunk (f32 epsilon).
    let log_ps = vec![-0.1f32, -0.2, -0.5, -0.3, -0.8];
    let mut a = OnlinePerplexity::new();
    for l in &log_ps {
        a.push(*l);
    }
    let mut b = OnlinePerplexity::new();
    b.push_chunk(&log_ps);
    assert!(
        (a.current() - b.current()).abs() < 1e-5,
        "a={} b={}",
        a.current(),
        b.current()
    );
}

#[test]
fn online_perplexity_reset() {
    let mut p = OnlinePerplexity::new();
    p.push(-1.0);
    p.push(-0.5);
    assert_eq!(p.tokens(), 2);
    p.reset();
    assert_eq!(p.tokens(), 0);
    assert_eq!(p.current(), f32::INFINITY);
}

#[test]
fn online_perplexity_partial_early_stop_stable() {
    // After partial feed, current() must be well-defined and match the formula.
    let mut p = OnlinePerplexity::new();
    p.push(-(2.0f32.ln())); // p = 0.5
                            // Mean neg log = ln(2) → PPL = 2.
    assert!((p.current() - 2.0).abs() < EPS, "got {}", p.current());
}

// ──────────────────────────────────────────────────────────────────────────────
// OnlineAccuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn online_accuracy_empty_is_zero() {
    let a = OnlineAccuracy::new();
    assert_eq!(a.current(), 0.0);
    assert_eq!(a.counts(), (0, 0));
}

#[test]
fn online_accuracy_all_correct() {
    let mut a = OnlineAccuracy::new();
    for _ in 0..5 {
        a.push(true);
    }
    assert_eq!(a.current(), 1.0);
    assert_eq!(a.counts(), (5, 5));
}

#[test]
fn online_accuracy_mixed_outcomes() {
    let mut a = OnlineAccuracy::new();
    a.push_many(&[true, false, true, false, true]);
    assert!((a.current() - 0.6).abs() < EPS);
    assert_eq!(a.counts(), (3, 5));
}

#[test]
fn online_accuracy_reset() {
    let mut a = OnlineAccuracy::new();
    a.push(true);
    a.push(false);
    a.reset();
    assert_eq!(a.counts(), (0, 0));
    assert_eq!(a.current(), 0.0);
}

#[test]
fn online_accuracy_in_unit_interval() {
    let mut a = OnlineAccuracy::new();
    a.push_many(&[true, false, false, true]);
    let c = a.current();
    assert!((0.0..=1.0).contains(&c));
}
