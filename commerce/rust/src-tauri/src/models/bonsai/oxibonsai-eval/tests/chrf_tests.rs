//! Integration tests for chrF / chrF++.

use oxibonsai_eval::chrf::{chrf, chrf_plus_plus, chrf_with};

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// Identical & disjoint
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chrf_identical_is_one() {
    let s = chrf("the quick brown fox", "the quick brown fox");
    assert!((s.score - 1.0).abs() < EPS, "score={}", s.score);
}

#[test]
fn chrf_both_empty_is_one() {
    let s = chrf("", "");
    assert!((s.score - 1.0).abs() < EPS);
}

#[test]
fn chrf_one_empty_is_zero() {
    assert_eq!(chrf("", "abc").score, 0.0);
    assert_eq!(chrf("abc", "").score, 0.0);
}

#[test]
fn chrf_disjoint_is_zero_or_tiny() {
    let s = chrf("xyz", "abc");
    assert!(
        s.score < 0.05,
        "disjoint should be near zero, got {}",
        s.score
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Order & sensitivity
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chrf_score_in_unit_interval() {
    let s = chrf("hello world", "world hello");
    assert!((0.0..=1.0).contains(&s.score));
}

#[test]
fn chrf_order_sensitivity_char() {
    // Completely reversed string still shares many low-order char n-grams,
    // but higher-order n-grams diverge → score < 1.
    let s = chrf("abcdef", "fedcba");
    assert!(s.score < 1.0);
    assert!(s.score > 0.0);
}

#[test]
fn chrf_partial_match_between_disjoint_and_identical() {
    let identical = chrf("the cat sat on the mat", "the cat sat on the mat").score;
    let partial = chrf("the cat sat on the mat", "the cat ran on the mat").score;
    let disjoint = chrf("the cat sat on the mat", "zzz yyy xxx www").score;
    assert!(partial < identical);
    assert!(partial > disjoint);
}

// ──────────────────────────────────────────────────────────────────────────────
// Config & fields
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chrf_default_order_is_six() {
    let s = chrf("hi", "hi");
    assert_eq!(s.order, 6);
}

#[test]
fn chrf_default_beta_is_two() {
    let s = chrf("hi", "hi");
    assert!((s.beta - 2.0).abs() < EPS);
}

#[test]
fn chrf_plus_plus_has_word_order_two() {
    let s = chrf_plus_plus("the cat sat", "the cat sat");
    assert_eq!(s.word_order, 2);
    assert!((s.score - 1.0).abs() < EPS);
}

#[test]
fn chrf_with_lower_beta_weights_precision() {
    // β=0.5 weights precision. Compare two candidates vs the same reference:
    // cand_a is a prefix of reference (high recall loss, okay precision),
    // cand_b has many extra characters (precision loss).
    let reference = "the cat sat on the mat today";
    let cand_a = "the cat sat"; // short — lower recall
    let cand_b = "the cat sat on the mat today extra extras added"; // noisy — lower precision
    let s_a = chrf_with(cand_a, reference, 6, 0.5, 0).score;
    let s_b = chrf_with(cand_b, reference, 6, 0.5, 0).score;
    // Low-beta: precision-heavy → cand_a (shorter, all chars in ref) should beat
    // cand_b (precision diluted by extra chars).
    assert!(
        s_a > s_b,
        "β=0.5 should prefer precision-clean output: s_a={} s_b={}",
        s_a,
        s_b
    );
}

#[test]
fn chrf_with_custom_order() {
    let s = chrf_with("hello", "hello", 3, 2.0, 0);
    assert_eq!(s.order, 3);
    assert!((s.score - 1.0).abs() < EPS);
}

// ──────────────────────────────────────────────────────────────────────────────
// UTF-8 handling
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chrf_handles_multibyte_utf8() {
    // Iteration must be over chars, not bytes.
    let s = chrf("日本語のテスト", "日本語のテスト");
    assert!((s.score - 1.0).abs() < EPS);
}

#[test]
fn chrf_utf8_vs_ascii_are_independent() {
    // Two texts with no shared chars.
    let s = chrf("日本語", "abc");
    assert!(s.score < 0.05);
}

#[test]
fn chrf_whitespace_matters_for_char_ngrams() {
    // Spaces are characters under chrF, so they influence n-grams.
    let s = chrf("a b c", "abc");
    assert!(s.score < 1.0);
    assert!(s.score > 0.0);
}
