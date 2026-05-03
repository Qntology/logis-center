//! Integration tests for BLEU.
//!
//! Covers:
//! - identical → 1.0
//! - disjoint → 0.0
//! - brevity penalty on shorter candidate
//! - smoothing: `None` (collapses to 0) vs `AddOne` (strictly positive) on sparse
//! - corpus vs sentence aggregation
//! - multi-reference: best (closest-length) reference drives score

use oxibonsai_eval::bleu::{corpus_bleu, sentence_bleu, BleuConfig, SmoothingMethod};

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// Identical / disjoint
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bleu_identical_is_one() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu("the cat sat on the mat", &["the cat sat on the mat"], &cfg);
    assert!((s.bleu - 1.0).abs() < EPS, "expected 1.0, got {}", s.bleu);
    assert!((s.brevity_penalty - 1.0).abs() < EPS);
    assert!((s.length_ratio - 1.0).abs() < EPS);
}

#[test]
fn bleu_disjoint_is_zero() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu(
        "the quick brown fox jumps",
        &["pneumonoultramicroscopic silicovolcanoconiosis lorem ipsum dolor sit"],
        &cfg,
    );
    assert!(s.bleu == 0.0, "expected 0, got {}", s.bleu);
}

#[test]
fn bleu_tokens_must_overlap() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu("x y z w", &["a b c d"], &cfg);
    assert_eq!(s.bleu, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Empty cases
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bleu_empty_candidate_zero() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu("", &["abc def"], &cfg);
    assert_eq!(s.bleu, 0.0);
    assert_eq!(s.brevity_penalty, 0.0);
}

#[test]
fn bleu_no_references_zero() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu("abc def", &[], &cfg);
    assert_eq!(s.bleu, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Brevity penalty
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn brevity_penalty_applied_when_short() {
    // Candidate shorter than reference → BP < 1.
    let cfg = BleuConfig::default();
    let s = sentence_bleu(
        "the cat",
        &["the cat sat on the mat very comfortably"],
        &cfg,
    );
    assert!(s.brevity_penalty < 1.0, "BP={}", s.brevity_penalty);
    assert!(s.brevity_penalty > 0.0);
}

#[test]
fn brevity_penalty_one_when_longer() {
    let cfg = BleuConfig::default();
    let s = sentence_bleu("the cat sat on the mat with joy", &["the cat sat"], &cfg);
    assert!((s.brevity_penalty - 1.0).abs() < EPS);
}

#[test]
fn brevity_penalty_equal_length() {
    // Equal c and r with all matching → BP = 1
    let cfg = BleuConfig::default();
    let s = sentence_bleu("a b c d", &["a b c d"], &cfg);
    assert!((s.brevity_penalty - 1.0).abs() < EPS);
    assert!((s.bleu - 1.0).abs() < EPS);
}

// ──────────────────────────────────────────────────────────────────────────────
// Smoothing
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn smoothing_none_collapses_when_missing_higher_order() {
    let cfg = BleuConfig::new(4, SmoothingMethod::None);
    // Short candidate: no 4-gram match possible → unsmoothed BLEU = 0.
    let s = sentence_bleu(
        "the cat sat",
        &["the dog ran fast over the river yesterday"],
        &cfg,
    );
    assert_eq!(s.bleu, 0.0);
}

#[test]
fn smoothing_add_one_recovers_positive_sparse() {
    let cfg = BleuConfig::new(4, SmoothingMethod::AddOne);
    // A mostly-matching short sentence where 4-gram matches are sparse.
    let s = sentence_bleu("the cat sat on", &["the cat sat on the mat"], &cfg);
    // Add-one smoothing must yield > 0 even with partial n-gram coverage.
    assert!(
        s.bleu > 0.0,
        "add-one should smooth sparse n-grams but got {}",
        s.bleu
    );
    assert!(s.bleu <= 1.0);
}

#[test]
fn smoothing_none_vs_add_one_ordering() {
    let cand = "the cat sat";
    let refs = ["the dog ran"];
    let none = sentence_bleu(cand, &refs, &BleuConfig::new(4, SmoothingMethod::None));
    let add_one = sentence_bleu(cand, &refs, &BleuConfig::new(4, SmoothingMethod::AddOne));
    // Add-one must never be strictly less than no smoothing on this kind of sparse input.
    assert!(
        add_one.bleu >= none.bleu,
        "add_one={} none={}",
        add_one.bleu,
        none.bleu
    );
}

#[test]
fn smoothing_exp_decay_strictly_positive() {
    let cfg = BleuConfig::new(4, SmoothingMethod::ExpDecay);
    let s = sentence_bleu("the cat sat on", &["the cat sat on the mat"], &cfg);
    assert!(s.bleu > 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Corpus vs sentence
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn corpus_bleu_identical_is_one() {
    let cfg = BleuConfig::default();
    let cands = vec!["the cat sat", "the dog ran fast"];
    let refs = vec![vec!["the cat sat"], vec!["the dog ran fast"]];
    let s = corpus_bleu(&cands, &refs, &cfg);
    assert!((s.bleu - 1.0).abs() < EPS, "got {}", s.bleu);
}

#[test]
fn corpus_bleu_accumulates() {
    // Corpus aggregates counts across sentences; individual sentences may have
    // fewer 4-grams, but corpus can still be > sentence average for mismatched
    // micro cases. Main property: both paths ∈ [0,1].
    let cfg = BleuConfig::default();
    let cands = vec!["the quick brown fox", "jumps over the lazy dog"];
    let refs = vec![vec!["the quick brown fox"], vec!["jumps over the lazy dog"]];
    let s_corpus = corpus_bleu(&cands, &refs, &cfg);
    assert!((0.0..=1.0).contains(&s_corpus.bleu));
    assert!((s_corpus.bleu - 1.0).abs() < EPS);
}

#[test]
fn corpus_bleu_empty_input() {
    let cfg = BleuConfig::default();
    let cands: Vec<&str> = vec![];
    let refs: Vec<Vec<&str>> = vec![];
    let s = corpus_bleu(&cands, &refs, &cfg);
    assert_eq!(s.bleu, 0.0);
}

#[test]
fn corpus_greater_or_equal_than_some_sentence() {
    // Corpus aggregation tends to dampen per-sentence zeros. Here with
    // matching sentences, corpus BLEU should equal 1 while a zero-overlap
    // pair alone would be 0.
    let cfg = BleuConfig::default();
    let good_s = sentence_bleu("a b c d", &["a b c d"], &cfg);
    let corpus_s = corpus_bleu(&["a b c d"], &[vec!["a b c d"]], &cfg);
    assert!((good_s.bleu - corpus_s.bleu).abs() < EPS);
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-reference — best ref drives score
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn multi_reference_picks_best_match() {
    let cfg = BleuConfig::default();
    let cand = "the cat sat on the mat";
    // One good reference, one unrelated → score must reflect the good reference.
    let s = sentence_bleu(
        cand,
        &["the cat sat on the mat", "something entirely different"],
        &cfg,
    );
    assert!(
        s.bleu > 0.5,
        "multi-ref BLEU should be high, got {}",
        s.bleu
    );
}

#[test]
fn multi_reference_length_closest_used() {
    let cfg = BleuConfig::default();
    // Use a candidate of ≥ max_n (=4) tokens so all n-gram orders are populated.
    let cand = "a b c d e";
    // Two references, one length-matching, one much longer.
    let s = sentence_bleu(cand, &["a b c d e", "a b c d e f g h i j k l"], &cfg);
    // Closest-length reference → no brevity penalty, full score.
    assert!(s.brevity_penalty >= 1.0 - EPS, "BP={}", s.brevity_penalty);
    assert!((s.bleu - 1.0).abs() < EPS, "BLEU={}", s.bleu);
}

// ──────────────────────────────────────────────────────────────────────────────
// Range & shape
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bleu_in_unit_interval() {
    let cfg = BleuConfig::default();
    let cand = "an interesting string with many tokens to test";
    let refs = ["a wildly different string of tokens to measure"];
    let s = sentence_bleu(cand, &refs, &cfg);
    assert!((0.0..=1.0).contains(&s.bleu));
    assert_eq!(s.precisions.len(), cfg.max_n);
}
