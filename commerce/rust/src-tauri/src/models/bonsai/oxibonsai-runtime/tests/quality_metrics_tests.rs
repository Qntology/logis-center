//! Integration tests for `quality_metrics` module.

use oxibonsai_runtime::quality_metrics::{
    extract_ngrams, perplexity_from_logprobs, self_bleu, token_entropy, BatchQualityAnalyzer,
    BleuScore, DiversityMetrics, GenerationQualityReport, RepetitionMetrics,
};

// ── 1. extract_ngrams: unigrams ──────────────────────────────────────────────

#[test]
fn extract_ngrams_unigrams() {
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
    let ngrams = extract_ngrams(&tokens, 1);
    assert_eq!(ngrams.len(), 5, "5 tokens → 5 unigrams");
    assert_eq!(ngrams[0], vec![10]);
    assert_eq!(ngrams[4], vec![50]);
}

// ── 2. extract_ngrams: bigrams ───────────────────────────────────────────────

#[test]
fn extract_ngrams_bigrams() {
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let ngrams = extract_ngrams(&tokens, 2);
    assert_eq!(ngrams.len(), 4, "5 tokens → 4 bigrams");
    assert_eq!(ngrams[0], vec![1, 2]);
    assert_eq!(ngrams[3], vec![4, 5]);
}

// ── 3. extract_ngrams: edge cases ────────────────────────────────────────────

#[test]
fn extract_ngrams_empty() {
    // Empty token list → always empty
    let empty: Vec<u32> = vec![];
    assert!(
        extract_ngrams(&empty, 2).is_empty(),
        "empty tokens → empty bigrams"
    );

    // Single token, n=2 → impossible
    let single: Vec<u32> = vec![42];
    assert!(
        extract_ngrams(&single, 2).is_empty(),
        "single token → no bigrams"
    );

    // n=0 → always empty
    let tokens: Vec<u32> = vec![1, 2, 3];
    assert!(extract_ngrams(&tokens, 0).is_empty(), "n=0 → empty");
}

// ── 4. RepetitionMetrics: no repetition ──────────────────────────────────────

#[test]
fn repetition_metrics_no_repetition() {
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let m = RepetitionMetrics::compute(&tokens);
    assert_eq!(m.token_count, 5);
    assert_eq!(m.repeated_unigrams, 0.0, "all distinct → rep=0");
    assert_eq!(m.unique_tokens, 5);
}

// ── 5. RepetitionMetrics: all same ───────────────────────────────────────────

#[test]
fn repetition_metrics_all_same() {
    let tokens: Vec<u32> = vec![7, 7, 7, 7, 7];
    let m = RepetitionMetrics::compute(&tokens);
    // Positions 1..4 all repeat the first token → repeated_unigrams = 4/4 = 1.0
    assert!(
        m.repeated_unigrams > 0.9,
        "all-same → very high repetition, got {}",
        m.repeated_unigrams
    );
    assert_eq!(m.unique_tokens, 1);
}

// ── 6. RepetitionMetrics: is_degenerate ──────────────────────────────────────

#[test]
fn repetition_metrics_is_degenerate() {
    // 8 repeats of 1, then 2 → repeated_unigrams = 7/8 = 0.875 > 0.7
    let tokens: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 2];
    let m = RepetitionMetrics::compute(&tokens);
    assert!(m.is_degenerate(), "high repetition → degenerate");

    // All distinct → not degenerate
    let ok: Vec<u32> = (0..10).collect();
    let m2 = RepetitionMetrics::compute(&ok);
    assert!(!m2.is_degenerate(), "no repetition → not degenerate");
}

// ── 7. RepetitionMetrics: max_consecutive_repeat ─────────────────────────────

#[test]
fn repetition_metrics_max_consecutive() {
    let tokens: Vec<u32> = vec![1, 1, 1, 2, 3];
    let m = RepetitionMetrics::compute(&tokens);
    assert_eq!(m.max_consecutive_repeat, 3, "longest run is [1,1,1] = 3");
}

// ── 8. DiversityMetrics: all unique ──────────────────────────────────────────

#[test]
fn diversity_metrics_all_unique() {
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
    let d = DiversityMetrics::compute(&tokens);
    assert!(
        (d.distinct_1 - 1.0).abs() < 1e-5,
        "all distinct unigrams → distinct_1=1.0, got {}",
        d.distinct_1
    );
    assert_eq!(d.vocab_coverage, d.distinct_1);
}

// ── 9. DiversityMetrics: all same ────────────────────────────────────────────

#[test]
fn diversity_metrics_all_same() {
    let n = 6u32;
    let tokens: Vec<u32> = vec![5; n as usize];
    let d = DiversityMetrics::compute(&tokens);
    // 1 unique / 6 total = 1/6 ≈ 0.1667
    let expected = 1.0 / n as f32;
    assert!(
        (d.distinct_1 - expected).abs() < 1e-5,
        "all-same distinct_1 expected {}, got {}",
        expected,
        d.distinct_1
    );
}

// ── 10. DiversityMetrics: overall_diversity ──────────────────────────────────

#[test]
fn diversity_overall() {
    let tokens: Vec<u32> = vec![1, 2, 3, 1, 2, 4];
    let d = DiversityMetrics::compute(&tokens);
    let expected = (d.distinct_1 + d.distinct_2 + d.distinct_3) / 3.0;
    assert!(
        (d.overall_diversity() - expected).abs() < 1e-5,
        "overall_diversity should be average of D1..D3"
    );
}

// ── 11. BleuScore: identical sequences ───────────────────────────────────────

#[test]
fn bleu_identical() {
    let seq: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let score = BleuScore::compute(&seq, &seq);
    assert!(
        (score.bleu - 1.0).abs() < 1e-4,
        "identical sequences → BLEU≈1.0, got {}",
        score.bleu
    );
    assert!(
        (score.brevity_penalty - 1.0).abs() < 1e-5,
        "same length → BP=1.0"
    );
}

// ── 12. BleuScore: disjoint sequences ────────────────────────────────────────

#[test]
fn bleu_disjoint() {
    let cand: Vec<u32> = vec![1, 2, 3, 4];
    let reference: Vec<u32> = vec![5, 6, 7, 8];
    let score = BleuScore::compute(&cand, &reference);
    assert_eq!(score.bleu, 0.0, "no overlap → BLEU=0");
}

// ── 13. BleuScore: partial overlap ───────────────────────────────────────────

#[test]
fn bleu_partial_overlap() {
    // Sequences long enough that all 4-gram orders can have overlap.
    // cand and ref share a 5-token prefix [1,2,3,4,5], differ afterwards.
    let cand: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 101, 102];
    let reference: Vec<u32> = vec![1, 2, 3, 4, 5, 200, 201, 202];
    let score = BleuScore::compute(&cand, &reference);
    assert!(
        score.bleu > 0.0 && score.bleu < 1.0,
        "partial overlap → 0 < BLEU < 1, got {}",
        score.bleu
    );
    // Also verify individual precisions are sensible
    assert!(
        score.precision_1 > 0.0,
        "P1 should be > 0 with shared prefix"
    );
    assert!(
        score.precision_4 > 0.0,
        "P4 should be > 0 with 5-token shared run"
    );
}

// ── 14. BleuScore: brevity penalty ───────────────────────────────────────────

#[test]
fn bleu_brevity_penalty() {
    // Very short candidate vs long reference
    let cand: Vec<u32> = vec![1, 2];
    let reference: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let score = BleuScore::compute(&cand, &reference);
    assert!(
        score.brevity_penalty < 1.0,
        "short candidate → BP < 1.0, got {}",
        score.brevity_penalty
    );
    // BP = exp(1 - 10/2) = exp(-4) ≈ 0.0183
    let expected_bp = (1.0f32 - 10.0 / 2.0).exp();
    assert!(
        (score.brevity_penalty - expected_bp).abs() < 1e-4,
        "BP expected {}, got {}",
        expected_bp,
        score.brevity_penalty
    );
}

// ── 15. self_bleu: identical samples ─────────────────────────────────────────

#[test]
fn self_bleu_identical() {
    let seq: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let samples = vec![seq.clone(), seq.clone(), seq.clone()];
    let sb = self_bleu(&samples);
    assert!(
        (sb - 1.0).abs() < 1e-4,
        "all identical → self_bleu≈1.0, got {}",
        sb
    );
}

// ── 16. self_bleu: diverse samples ───────────────────────────────────────────

#[test]
fn self_bleu_all_different() {
    // Completely disjoint token sets
    let samples = vec![
        vec![1u32, 2, 3, 4, 5, 6, 7, 8],
        vec![11, 12, 13, 14, 15, 16, 17, 18],
        vec![21, 22, 23, 24, 25, 26, 27, 28],
        vec![31, 32, 33, 34, 35, 36, 37, 38],
    ];
    let sb = self_bleu(&samples);
    assert!(
        sb < 0.5,
        "diverse (disjoint) samples → self_bleu < 0.5, got {}",
        sb
    );
}

// ── 17. token_entropy: uniform distribution ───────────────────────────────────

#[test]
fn token_entropy_uniform() {
    // Equal logits → uniform distribution → maximum entropy = log2(n)
    let n = 8usize;
    let logits = vec![1.0f32; n];
    let h = token_entropy(&logits);
    let expected = (n as f32).log2();
    assert!(
        (h - expected).abs() < 1e-4,
        "uniform → entropy=log2({})={}, got {}",
        n,
        expected,
        h
    );
}

// ── 18. token_entropy: peaked distribution ───────────────────────────────────

#[test]
fn token_entropy_peaked() {
    // One dominant logit → low entropy
    let mut logits = vec![-100.0f32; 100];
    logits[0] = 100.0;
    let h = token_entropy(&logits);
    assert!(h < 0.01, "peaked distribution → entropy ≈ 0, got {}", h);
}

// ── 19. perplexity_from_logprobs ─────────────────────────────────────────────

#[test]
fn perplexity_from_logprobs_ones() {
    // log_probs = [-1, -1, -1] → PPL = exp(1) ≈ 2.718
    let log_probs = vec![-1.0f32, -1.0, -1.0];
    let ppl = perplexity_from_logprobs(&log_probs);
    let expected = std::f32::consts::E;
    assert!(
        (ppl - expected).abs() < 1e-4,
        "log_probs all -1 → PPL=e={}, got {}",
        expected,
        ppl
    );
}

// ── 20. GenerationQualityReport: smoke test ───────────────────────────────────

#[test]
fn quality_report_compute() {
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2, 6, 7, 8];
    let logits = vec![0.1f32; 32];
    let report = GenerationQualityReport::compute(&tokens, Some(&logits));
    // Smoke test: should not panic, quality_score in [0,1]
    assert!(
        report.quality_score >= 0.0 && report.quality_score <= 1.0,
        "quality_score should be in [0,1], got {}",
        report.quality_score
    );
    let summary = report.summary();
    assert!(
        summary.contains("GenerationQualityReport"),
        "summary should mention struct name"
    );
}

// ── 21. BatchQualityAnalyzer: mean_quality ────────────────────────────────────

#[test]
fn batch_analyzer_mean_quality() {
    let mut analyzer = BatchQualityAnalyzer::new();
    analyzer.add_generation(&[1u32, 2, 3, 4, 5]);
    analyzer.add_generation(&[6u32, 7, 8, 9, 10]);

    let mean_q = analyzer.mean_quality_score().expect("should have mean");
    assert!(
        (0.0..=1.0).contains(&mean_q),
        "mean quality in [0,1], got {}",
        mean_q
    );
    assert_eq!(analyzer.num_reports(), 2);
}

// ── 22. BatchQualityAnalyzer: degenerate_fraction ────────────────────────────

#[test]
fn batch_analyzer_degenerate_fraction() {
    let mut analyzer = BatchQualityAnalyzer::new();
    // Degenerate: 8× same token then 1 different
    analyzer.add_generation(&[1u32, 1, 1, 1, 1, 1, 1, 1, 2]);
    // Healthy: all distinct
    analyzer.add_generation(&[10u32, 20, 30, 40, 50]);

    let frac = analyzer.degenerate_fraction();
    // 1 out of 2 → 0.5
    assert!(
        (frac - 0.5).abs() < 1e-5,
        "1/2 degenerate → fraction=0.5, got {}",
        frac
    );
}

// ── 23. BatchQualityAnalyzer: self_bleu_score ────────────────────────────────

#[test]
fn batch_analyzer_self_bleu() {
    let mut analyzer = BatchQualityAnalyzer::new();
    analyzer.add_generation(&[1u32, 2, 3, 4, 5]);
    analyzer.add_generation(&[6u32, 7, 8, 9, 10]);
    analyzer.add_generation(&[11u32, 12, 13, 14, 15]);

    // Should run without panic; disjoint → low self-BLEU
    let sb = analyzer.self_bleu_score();
    assert!(sb >= 0.0, "self_bleu >= 0, got {}", sb);
    assert!(sb < 0.5, "disjoint samples → self_bleu < 0.5, got {}", sb);

    let report_str = analyzer.report();
    assert!(
        report_str.contains("BatchQualityAnalyzer"),
        "report should contain struct name"
    );
}
