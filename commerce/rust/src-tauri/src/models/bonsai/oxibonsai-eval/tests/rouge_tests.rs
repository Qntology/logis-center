//! Integration tests for ROUGE evaluation metrics.

use oxibonsai_eval::rouge::{
    ngram_counts, tokenize, CorpusRouge, RougeLScore, RougeNScore, RougeSScore,
};

// ──────────────────────────────────────────────────────────────────────────────
// Tokenization tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn tokenize_basic() {
    let tokens = tokenize("the cat sat on the mat");
    assert_eq!(tokens, vec!["the", "cat", "sat", "on", "the", "mat"]);
}

#[test]
fn tokenize_punctuation() {
    let tokens = tokenize("Hello, world! How are you?");
    // Punctuation is stripped; tokens are lowercased.
    assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
}

// ──────────────────────────────────────────────────────────────────────────────
// N-gram count tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ngram_counts_unigrams() {
    let tokens = tokenize("the cat the");
    let counts = ngram_counts(&tokens, 1);
    assert_eq!(counts.get(&vec!["the".to_string()]).copied(), Some(2));
    assert_eq!(counts.get(&vec!["cat".to_string()]).copied(), Some(1));
}

#[test]
fn ngram_counts_bigrams() {
    let tokens = tokenize("a b c a b");
    let counts = ngram_counts(&tokens, 2);
    let ab = vec!["a".to_string(), "b".to_string()];
    let bc = vec!["b".to_string(), "c".to_string()];
    let ca = vec!["c".to_string(), "a".to_string()];
    assert_eq!(counts.get(&ab).copied(), Some(2));
    assert_eq!(counts.get(&bc).copied(), Some(1));
    assert_eq!(counts.get(&ca).copied(), Some(1));
}

// ──────────────────────────────────────────────────────────────────────────────
// ROUGE-1 tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn rouge_1_identical() {
    let s = "the quick brown fox";
    let score = RougeNScore::compute(s, s, 1);
    assert!(
        (score.precision - 1.0).abs() < 1e-5,
        "precision={}",
        score.precision
    );
    assert!((score.recall - 1.0).abs() < 1e-5, "recall={}", score.recall);
    assert!((score.f1 - 1.0).abs() < 1e-5, "f1={}", score.f1);
}

#[test]
fn rouge_1_disjoint() {
    let cand = "alpha beta gamma";
    let reference = "delta epsilon zeta";
    let score = RougeNScore::compute(cand, reference, 1);
    assert!((score.precision).abs() < 1e-5);
    assert!((score.recall).abs() < 1e-5);
    assert!((score.f1).abs() < 1e-5);
}

#[test]
fn rouge_1_partial() {
    let cand = "the cat sat";
    let reference = "the cat on the mat";
    let score = RougeNScore::compute(cand, reference, 1);
    // Some overlap ("the", "cat") but not full.
    assert!(
        score.precision > 0.0 && score.precision < 1.0,
        "precision={}",
        score.precision
    );
    assert!(
        score.recall > 0.0 && score.recall < 1.0,
        "recall={}",
        score.recall
    );
    assert!(score.f1 > 0.0 && score.f1 < 1.0, "f1={}", score.f1);
}

#[test]
fn rouge_1_recall_focus() {
    // Candidate is a subset of reference.
    // Recall = shared unigrams / reference unigrams.
    let cand = "the cat";
    let reference = "the cat sat on the mat";
    let score = RougeNScore::compute(cand, reference, 1);
    // cand: {the:1, cat:1} → 2 unigrams
    // reference: {the:2, cat:1, sat:1, on:1, mat:1} → 6 unigrams
    // overlap (clipped): the=min(1,2)=1 + cat=min(1,1)=1 = 2
    // recall = 2/6 ≈ 0.333
    assert!(
        (score.recall - 2.0 / 6.0).abs() < 1e-4,
        "expected recall≈{:.4} got {:.4}",
        2.0 / 6.0,
        score.recall
    );
    // precision = 2/2 = 1.0
    assert!(
        (score.precision - 1.0).abs() < 1e-4,
        "expected precision=1.0 got {:.4}",
        score.precision
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// ROUGE-2 tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn rouge_2_identical() {
    let s = "the quick brown fox jumps";
    let score = RougeNScore::compute(s, s, 2);
    assert!((score.f1 - 1.0).abs() < 1e-5, "f1={}", score.f1);
}

#[test]
fn rouge_2_disjoint() {
    let cand = "alpha beta gamma delta";
    let reference = "epsilon zeta eta theta";
    let score = RougeNScore::compute(cand, reference, 2);
    assert!(score.f1 < 1e-5, "f1should be 0, got {}", score.f1);
}

// ──────────────────────────────────────────────────────────────────────────────
// ROUGE-L tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn rouge_l_identical() {
    let s = "the quick brown fox";
    let score = RougeLScore::compute(s, s);
    assert!((score.f1 - 1.0).abs() < 1e-5, "f1={}", score.f1);
    // LCS length equals the token count.
    let token_len = tokenize(s).len();
    assert_eq!(score.lcs_length, token_len);
}

#[test]
fn rouge_l_lcs_length_correct() {
    // Known LCS: "a b c" in both sequences.
    let a_tokens: Vec<String> = ["a", "x", "b", "y", "c"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b_tokens: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let lcs = RougeLScore::lcs_length(&a_tokens, &b_tokens);
    assert_eq!(lcs, 3, "LCS of (a x b y c) and (a b c) should be 3");
}

// ──────────────────────────────────────────────────────────────────────────────
// ROUGE-S tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn rouge_s_identical() {
    let s = "the quick brown fox";
    let score = RougeSScore::compute(s, s);
    assert!((score.f1 - 1.0).abs() < 1e-5, "f1={}", score.f1);
}

// ──────────────────────────────────────────────────────────────────────────────
// CorpusRouge tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn corpus_rouge_single_pair() {
    let cand = "the quick brown fox";
    let reference = "the quick brown fox";
    let corpus = CorpusRouge::compute(&[(cand, reference)]);
    assert_eq!(corpus.num_samples, 1);
    // Single identical pair — same as per-pair score.
    let r1_f1 = corpus.rouge_1.as_ref().map(|s| s.f1).unwrap_or(0.0);
    assert!((r1_f1 - 1.0).abs() < 1e-4, "r1 f1={}", r1_f1);
    let rl_f1 = corpus.rouge_l.as_ref().map(|s| s.f1).unwrap_or(0.0);
    assert!((rl_f1 - 1.0).abs() < 1e-4, "rl f1={}", rl_f1);
}

#[test]
fn corpus_rouge_summary_nonempty() {
    let pairs: Vec<(&str, &str)> = vec![
        ("hello world", "hello world"),
        ("foo bar baz", "bar baz qux"),
    ];
    let corpus = CorpusRouge::compute(&pairs);
    let summary = corpus.summary();
    assert!(!summary.is_empty(), "summary should not be empty");
    assert!(summary.contains("ROUGE"), "summary should mention ROUGE");
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-reference test
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn rouge_multi_ref_takes_max() {
    let cand = "the cat sat on the mat";
    let good_ref = "the cat sat on the mat";
    let bad_ref = "completely unrelated text here";

    let multi = RougeNScore::compute_multi_ref(cand, &[bad_ref, good_ref], 1);
    let single_good = RougeNScore::compute(cand, good_ref, 1);
    let single_bad = RougeNScore::compute(cand, bad_ref, 1);

    // Multi-ref should be >= single-ref against the bad reference.
    assert!(
        multi.recall >= single_bad.recall,
        "multi-ref recall {} should >= bad ref recall {}",
        multi.recall,
        single_bad.recall
    );
    // Multi-ref should match the good reference (highest recall).
    assert!(
        (multi.recall - single_good.recall).abs() < 1e-4,
        "multi-ref recall {} should ≈ good ref recall {}",
        multi.recall,
        single_good.recall
    );
}
