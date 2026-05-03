//! Integration tests for SQuAD-style QA evaluation.

use oxibonsai_eval::qa::{
    corpus_em_f1, exact_match, f1_score, normalize_answer, normalize_tokens, score_multi,
};

const EPS: f32 = 1e-4;

// ──────────────────────────────────────────────────────────────────────────────
// normalize_answer
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn normalize_lowercases() {
    assert_eq!(normalize_answer("HELLO World"), "hello world");
}

#[test]
fn normalize_strips_punctuation() {
    assert_eq!(normalize_answer("hello, world!"), "hello world");
}

#[test]
fn normalize_removes_articles() {
    // All three articles must vanish as standalone tokens.
    assert_eq!(normalize_answer("a banana"), "banana");
    assert_eq!(normalize_answer("an apple"), "apple");
    assert_eq!(normalize_answer("the cat"), "cat");
    assert_eq!(normalize_answer("the an a cat"), "cat");
}

#[test]
fn normalize_keeps_article_inside_token() {
    // "the" as a substring of another word must not be stripped.
    assert_eq!(normalize_answer("theatre"), "theatre");
}

#[test]
fn normalize_collapses_whitespace() {
    assert_eq!(normalize_answer("  hello   \t world  \n"), "hello world");
}

#[test]
fn normalize_tokens_split() {
    let tokens = normalize_tokens("The quick, brown fox!");
    assert_eq!(tokens, vec!["quick", "brown", "fox"]);
}

#[test]
fn normalize_empty_is_empty() {
    assert_eq!(normalize_answer(""), "");
    assert!(normalize_tokens("").is_empty());
    assert!(normalize_tokens("the").is_empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// Exact match
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn exact_match_identical_after_normalisation() {
    assert_eq!(exact_match("The Cat", "the cat"), 1.0);
    assert_eq!(exact_match("hello!", "hello"), 1.0);
}

#[test]
fn exact_match_distinct_strings() {
    assert_eq!(exact_match("dog", "cat"), 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// F1 score
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn f1_identical_is_one() {
    assert!((f1_score("the cat sat", "the cat sat") - 1.0).abs() < EPS);
}

#[test]
fn f1_partial_overlap() {
    // pred tokens {cat, sat}, ref tokens {cat, ran} → common=1 → P=R=0.5 → F1=0.5
    let f = f1_score("the cat sat", "the cat ran");
    assert!((f - 0.5).abs() < EPS, "got {}", f);
}

#[test]
fn em_vs_f1_divergence() {
    // Partial overlap: EM=0 but F1 > 0.
    let em = exact_match("the cat sat", "the cat ran");
    let f = f1_score("the cat sat", "the cat ran");
    assert_eq!(em, 0.0);
    assert!(f > 0.0);
}

#[test]
fn f1_empty_prediction_and_nonempty_ref_zero() {
    assert_eq!(f1_score("", "the cat"), 0.0);
}

#[test]
fn f1_both_empty_one() {
    assert!((f1_score("", "") - 1.0).abs() < EPS);
    assert!((f1_score("the", "a") - 1.0).abs() < EPS); // both normalise to empty
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-reference
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn score_multi_picks_max() {
    let s = score_multi("the cat", &["the dog", "the cat"]);
    assert_eq!(s.exact_match, 1.0);
    assert!((s.f1 - 1.0).abs() < EPS);
}

#[test]
fn score_multi_empty_refs_zero() {
    let s = score_multi("anything", &[]);
    assert_eq!(s.exact_match, 0.0);
    assert_eq!(s.f1, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Corpus aggregation
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn corpus_em_f1_empty() {
    let (em, f) = corpus_em_f1(&[]);
    assert_eq!(em, 0.0);
    assert_eq!(f, 0.0);
}

#[test]
fn corpus_em_f1_averages() {
    let examples: Vec<(String, Vec<String>)> = vec![
        ("the cat".into(), vec!["the cat".into()]),
        ("dog".into(), vec!["fish".into()]),
    ];
    let (em, f) = corpus_em_f1(&examples);
    // First is perfect (EM=1, F1=1), second is zero → averages 0.5 / 0.5.
    assert!((em - 0.5).abs() < EPS);
    assert!((f - 0.5).abs() < EPS);
}
