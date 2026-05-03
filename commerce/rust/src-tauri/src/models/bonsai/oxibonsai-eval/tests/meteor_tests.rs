//! Integration tests for METEOR (lexical, exact-match only).

use oxibonsai_eval::meteor::{align_tokens, meteor, meteor_multi, MeteorConfig};
use oxibonsai_eval::rouge::tokenize;

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// Identical / empty
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn meteor_identical_is_one() {
    let cfg = MeteorConfig::default();
    let s = meteor("the cat sat on the mat", "the cat sat on the mat", &cfg);
    // METEOR applies the fragmentation penalty `γ · (chunks/matches)^β` even
    // on a single-chunk identical alignment, so the score is
    // `(1 - 0.5·(1/6)^3) · 1.0 ≈ 0.9977`. Assert it's essentially perfect.
    assert!(s.score > 0.99, "score={}", s.score);
    assert!((s.precision - 1.0).abs() < EPS);
    assert!((s.recall - 1.0).abs() < EPS);
    // Fragmentation is recorded as the penalty factor itself (see meteor.rs);
    // ensure it is small but non-negative.
    assert!(s.fragmentation >= 0.0);
    assert!(s.fragmentation < 0.01);
}

#[test]
fn meteor_both_empty_is_one() {
    let cfg = MeteorConfig::default();
    let s = meteor("", "", &cfg);
    assert!((s.score - 1.0).abs() < EPS);
}

#[test]
fn meteor_one_empty_is_zero() {
    let cfg = MeteorConfig::default();
    assert_eq!(meteor("", "hello", &cfg).score, 0.0);
    assert_eq!(meteor("hello", "", &cfg).score, 0.0);
}

#[test]
fn meteor_disjoint_is_zero() {
    let cfg = MeteorConfig::default();
    let s = meteor("abc def ghi", "jkl mno pqr", &cfg);
    assert_eq!(s.score, 0.0);
    assert_eq!(s.precision, 0.0);
    assert_eq!(s.recall, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Default α = 0.9
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn meteor_default_alpha_is_09() {
    let cfg = MeteorConfig::default();
    assert!((cfg.alpha - 0.9).abs() < EPS);
}

#[test]
fn meteor_default_gamma_beta() {
    let cfg = MeteorConfig::default();
    assert!((cfg.gamma - 0.5).abs() < EPS);
    assert!((cfg.beta - 3.0).abs() < EPS);
}

// ──────────────────────────────────────────────────────────────────────────────
// Fragmentation penalty ordering
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn meteor_fragmentation_penalises_shuffled() {
    let cfg = MeteorConfig::default();
    // Candidate preserves reference word order → 1 chunk, no penalty.
    let in_order = meteor("the cat sat on the mat", "the cat sat on the mat", &cfg);
    // Candidate reverses word order → more chunks → larger fragmentation.
    let shuffled = meteor("mat the on sat cat the", "the cat sat on the mat", &cfg);
    assert!(
        in_order.fragmentation <= shuffled.fragmentation,
        "expected in-order fragmentation <= shuffled, got {} vs {}",
        in_order.fragmentation,
        shuffled.fragmentation
    );
    assert!(
        in_order.score >= shuffled.score,
        "in-order score should >= shuffled"
    );
}

#[test]
fn meteor_partial_overlap_positive() {
    let cfg = MeteorConfig::default();
    let s = meteor("the cat ran", "the cat sat", &cfg);
    assert!(s.score > 0.0);
    assert!(s.score < 1.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-reference
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn meteor_multi_picks_max() {
    let cfg = MeteorConfig::default();
    let s = meteor_multi(
        "the cat sat on the mat",
        &["something unrelated", "the cat sat on the mat"],
        &cfg,
    );
    // Identical ref → near-1 after tiny fragmentation correction (see
    // `meteor_identical_is_one`).
    assert!(s.score > 0.99, "score={}", s.score);
}

#[test]
fn meteor_multi_empty_refs_is_zero() {
    let cfg = MeteorConfig::default();
    let s = meteor_multi("anything", &[], &cfg);
    assert_eq!(s.score, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Alignment / chunks
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn align_tokens_full_match() {
    let c = tokenize("a b c");
    let r = tokenize("a b c");
    let alignment = align_tokens(&c, &r);
    assert_eq!(alignment, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn align_tokens_each_ref_used_once() {
    let c = tokenize("the the the");
    let r = tokenize("the cat sat");
    let alignment = align_tokens(&c, &r);
    // Only one "the" on the reference side → only one match.
    assert_eq!(alignment.len(), 1);
}

#[test]
fn align_tokens_no_overlap() {
    let c = tokenize("x y z");
    let r = tokenize("a b c");
    assert!(align_tokens(&c, &r).is_empty());
}

#[test]
fn meteor_range_is_unit_interval() {
    let cfg = MeteorConfig::default();
    let s = meteor("some tokens here", "different tokens entirely", &cfg);
    assert!((0.0..=1.0).contains(&s.score));
}

#[test]
fn meteor_recall_precision_bounds() {
    let cfg = MeteorConfig::default();
    let s = meteor("the cat", "the cat sat on the mat", &cfg);
    assert!((0.0..=1.0).contains(&s.precision));
    assert!((0.0..=1.0).contains(&s.recall));
    // Candidate ⊂ reference tokens → precision > recall.
    assert!(s.precision > s.recall);
}
