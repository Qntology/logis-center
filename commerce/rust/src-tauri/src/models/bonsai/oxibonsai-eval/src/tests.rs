//! Integration tests for the evaluation harness.

use crate::accuracy::{AccuracyResult, ExactMatchEvaluator, McEvaluator};
use crate::dataset::{EvalDataset, EvalExample, McDataset, MultipleChoiceQuestion};
use crate::perplexity::PerplexityEvaluator;
use crate::report::EvalReport;
use crate::throughput::{percentile, ThroughputBenchmark};

// ──────────────────────────────────────────────────────────────────────────────
// Dataset tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_eval_dataset_from_jsonl() {
    let jsonl = r#"{"id":"1","input":"hello world","expected_output":"hello"}
{"id":"2","input":"foo bar"}"#;
    let ds = EvalDataset::from_jsonl("test", jsonl).expect("should parse");
    assert_eq!(ds.len(), 2);
    assert_eq!(ds.examples[0].id, "1");
    assert_eq!(ds.examples[0].input, "hello world");
    assert_eq!(ds.examples[0].expected_output.as_deref(), Some("hello"));
    assert_eq!(ds.examples[1].id, "2");
    assert!(ds.examples[1].expected_output.is_none());
}

#[test]
fn test_eval_dataset_sample_deterministic() {
    let mut ds = EvalDataset::new("test");
    for i in 0..20 {
        ds.add(EvalExample {
            id: i.to_string(),
            input: format!("input {}", i),
            expected_output: None,
            metadata: Default::default(),
        });
    }
    let s1 = ds.sample(5, 42);
    let s2 = ds.sample(5, 42);
    // Same seed → same order
    let ids1: Vec<&str> = s1.examples.iter().map(|e| e.id.as_str()).collect();
    let ids2: Vec<&str> = s2.examples.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids1, ids2);
    assert_eq!(s1.len(), 5);
}

#[test]
fn test_eval_dataset_split() {
    let mut ds = EvalDataset::new("test");
    for i in 0..10 {
        ds.add(EvalExample {
            id: i.to_string(),
            input: format!("input {}", i),
            expected_output: None,
            metadata: Default::default(),
        });
    }
    let (train, test) = ds.split(0.8);
    assert_eq!(train.len(), 8);
    assert_eq!(test.len(), 2);
    // Order is preserved
    assert_eq!(train.examples[0].id, "0");
    assert_eq!(test.examples[0].id, "8");
}

#[test]
fn test_mc_dataset_filter_by_subject() {
    let mut ds = McDataset::new("mmlu");
    ds.add(MultipleChoiceQuestion {
        id: "1".to_string(),
        question: "Q1".to_string(),
        choices: vec![
            "A: x".to_string(),
            "B: y".to_string(),
            "C: z".to_string(),
            "D: w".to_string(),
        ],
        correct_answer: 0,
        subject: Some("biology".to_string()),
        difficulty: None,
    });
    ds.add(MultipleChoiceQuestion {
        id: "2".to_string(),
        question: "Q2".to_string(),
        choices: vec![
            "A: a".to_string(),
            "B: b".to_string(),
            "C: c".to_string(),
            "D: d".to_string(),
        ],
        correct_answer: 1,
        subject: Some("chemistry".to_string()),
        difficulty: None,
    });
    let bio = ds.filter_by_subject("biology");
    assert_eq!(bio.len(), 1);
    assert_eq!(bio.questions[0].id, "1");
}

#[test]
fn test_mc_dataset_subjects() {
    let mut ds = McDataset::new("mmlu");
    for subj in &["chemistry", "biology", "biology", "physics"] {
        ds.add(MultipleChoiceQuestion {
            id: "x".to_string(),
            question: "Q".to_string(),
            choices: vec![
                "A: a".to_string(),
                "B: b".to_string(),
                "C: c".to_string(),
                "D: d".to_string(),
            ],
            correct_answer: 0,
            subject: Some(subj.to_string()),
            difficulty: None,
        });
    }
    let subjects = ds.subjects();
    // Sorted and deduplicated
    assert_eq!(subjects, vec!["biology", "chemistry", "physics"]);
}

// ──────────────────────────────────────────────────────────────────────────────
// Perplexity tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_perplexity_perfect_predictions() {
    // log(1) = 0.0 for every token → PPL = exp(0) = 1.0
    let eval = PerplexityEvaluator::new();
    let log_probs = vec![0.0f32; 10];
    let ppl = eval.compute(&log_probs);
    assert!((ppl - 1.0).abs() < 1e-5, "expected PPL ≈ 1.0, got {}", ppl);
}

#[test]
fn test_perplexity_uniform_predictions() {
    // Uniform over V=100 → log(1/100) = -ln(100) per token → PPL = 100
    let eval = PerplexityEvaluator::new();
    let lp = -(100.0f32).ln();
    let log_probs = vec![lp; 20];
    let ppl = eval.compute(&log_probs);
    assert!(
        (ppl - 100.0).abs() < 1e-3,
        "expected PPL ≈ 100, got {}",
        ppl
    );
}

#[test]
fn test_perplexity_batch() {
    let eval = PerplexityEvaluator::new();
    // Two sequences: perfect (PPL=1) and uniform-100 (PPL=100)
    let s1 = vec![0.0f32; 10];
    let s2 = vec![-(100.0f32).ln(); 10];
    let result = eval.compute_batch(&[s1, s2]);
    assert_eq!(result.n_samples, 2);
    assert_eq!(result.total_tokens, 20);
    // mean of 1 and 100 = 50.5
    assert!(
        (result.mean_ppl - 50.5).abs() < 1.0,
        "mean_ppl={}",
        result.mean_ppl
    );
    assert!(result.min_ppl < result.max_ppl);
}

#[test]
fn test_bits_per_byte() {
    let eval = PerplexityEvaluator::new();
    // log(1) = 0.0 → BPB = 0
    let log_probs = vec![0.0f32; 10];
    let bpb = eval.bits_per_byte(&log_probs, 10);
    assert!(bpb.abs() < 1e-5, "expected BPB ≈ 0, got {}", bpb);

    // Uniform over 2 → log(0.5) = -ln2 → BPB = 1.0 bit/byte (1 byte per token)
    let lp = -(2.0f32).ln();
    let log_probs2 = vec![lp; 8];
    let bpb2 = eval.bits_per_byte(&log_probs2, 8);
    assert!(
        (bpb2 - 1.0).abs() < 1e-5,
        "expected BPB ≈ 1.0, got {}",
        bpb2
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Accuracy / MC tests
// ──────────────────────────────────────────────────────────────────────────────

fn make_mc_question(id: &str, correct: usize, subject: Option<&str>) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: id.to_string(),
        question: "What is 2+2?".to_string(),
        choices: vec![
            "A: 3".to_string(),
            "B: 4".to_string(),
            "C: 5".to_string(),
            "D: 6".to_string(),
        ],
        correct_answer: correct,
        subject: subject.map(str::to_string),
        difficulty: None,
    }
}

#[test]
fn test_mc_evaluator_format_question() {
    let eval = McEvaluator::new();
    let q = make_mc_question("1", 1, None);
    let formatted = eval.format_question(&q);
    assert!(
        formatted.contains("What is 2+2?"),
        "formatted: {}",
        formatted
    );
    assert!(formatted.contains("Answer:"));
}

#[test]
fn test_mc_evaluator_extract_answer() {
    let eval = McEvaluator::new();
    assert_eq!(eval.extract_answer("A"), Some(0));
    assert_eq!(eval.extract_answer("B is correct"), Some(1));
    assert_eq!(eval.extract_answer("c"), Some(2));
    assert_eq!(eval.extract_answer("D."), Some(3));
    assert_eq!(eval.extract_answer("  B"), Some(1));
    assert_eq!(eval.extract_answer("X"), None);
    assert_eq!(eval.extract_answer(""), None);
}

#[test]
fn test_mc_evaluator_score_correct() {
    let eval = McEvaluator::new();
    // correct_answer = 1 (B), completion starts with "B"
    assert!(eval.score_completion("B", 1));
    assert!(eval.score_completion("B: four", 1));
}

#[test]
fn test_mc_evaluator_score_incorrect() {
    let eval = McEvaluator::new();
    assert!(!eval.score_completion("A", 1));
    assert!(!eval.score_completion("C is correct", 1));
    assert!(!eval.score_completion("", 0));
}

#[test]
fn test_exact_match_evaluator_basic() {
    let eval = ExactMatchEvaluator::new();
    assert!(eval.score("hello", "hello"));
    assert!(!eval.score("hello", "Hello"));
    assert!(!eval.score("hello world", "hello"));
}

#[test]
fn test_exact_match_evaluator_normalized() {
    let eval = ExactMatchEvaluator {
        normalize: true,
        partial_match: false,
    };
    assert!(eval.score("Hello ", "hello"));
    assert!(eval.score(" FOO ", "foo"));
    assert!(!eval.score("hello world", "hello"));
}

#[test]
fn test_accuracy_result_pct() {
    let result = AccuracyResult {
        correct: 3,
        total: 4,
        accuracy: 0.75,
        by_subject: Default::default(),
    };
    assert!((result.accuracy_pct() - 75.0).abs() < 1e-5);
}

// ──────────────────────────────────────────────────────────────────────────────
// Throughput tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_percentile_median() {
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let median = percentile(values, 50.0);
    assert!((median - 3.0).abs() < 1e-5, "median={}", median);
}

#[test]
fn test_percentile_p95() {
    // 20 evenly spaced values 1..=20
    let values: Vec<f32> = (1..=20).map(|x| x as f32).collect();
    let p95 = percentile(values, 95.0);
    // 95th percentile of 1..20 ≈ 19.05
    assert!(p95 > 18.0 && p95 <= 20.0, "p95={}", p95);
}

#[test]
fn test_throughput_result_meets_target() {
    let bench = ThroughputBenchmark::new("hello", 128);
    // 10 runs: each 10 ms prefill, 90 ms decode, 100 tokens → ~909 t/s
    let timings: Vec<(f32, f32, usize)> = (0..10).map(|_| (10.0, 90.0, 100)).collect();
    let result = bench.from_timings(&timings);
    assert!(
        result.meets_target(500.0),
        "tps={}",
        result.tokens_per_second
    );
    assert!(!result.meets_target(10_000.0));
}

// ──────────────────────────────────────────────────────────────────────────────
// Report tests
// ──────────────────────────────────────────────────────────────────────────────

fn make_report() -> EvalReport {
    use crate::perplexity::PerplexityResult;
    use crate::throughput::ThroughputResult;

    let mut report = EvalReport::new("test-model");

    let ppl_result = PerplexityResult {
        mean_ppl: 12.5,
        min_ppl: 10.0,
        max_ppl: 15.0,
        std_ppl: 1.5,
        n_samples: 100,
        total_tokens: 5000,
    };
    report.add_perplexity("wikitext-2", &ppl_result);

    let acc_result = AccuracyResult {
        correct: 75,
        total: 100,
        accuracy: 0.75,
        by_subject: Default::default(),
    };
    report.add_accuracy("mmlu", &acc_result);

    let tps_result = ThroughputResult {
        tokens_per_second: 120.0,
        prefill_ms: 5.0,
        decode_ms_per_token: 3.5,
        total_tokens: 1200,
        runs: 10,
        min_tps: 100.0,
        max_tps: 140.0,
        p50_tps: 118.0,
        p95_tps: 138.0,
    };
    report.add_throughput(&tps_result);

    report
}

#[test]
fn test_eval_report_to_json() {
    let report = make_report();
    let json = report.to_json();
    assert!(json.contains("\"model_name\""), "json: {}", json);
    assert!(json.contains("test-model"), "json: {}", json);
    assert!(json.contains("perplexity"), "json: {}", json);
    assert!(json.contains("accuracy"), "json: {}", json);
    // Valid JSON
    let _: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");
}

#[test]
fn test_eval_report_to_markdown() {
    let report = make_report();
    let md = report.to_markdown();
    assert!(md.contains("# Evaluation Report"), "md: {}", md);
    assert!(md.contains("test-model"), "md: {}", md);
    assert!(md.contains("| Task |"), "md: {}", md);
    assert!(md.contains("perplexity"), "md: {}", md);
    assert!(md.contains("accuracy"), "md: {}", md);
}

#[test]
fn test_eval_report_summary() {
    let report = make_report();
    let summary = report.summary();
    assert!(summary.contains("test-model"), "summary: {}", summary);
    assert!(summary.contains("PPL:"), "summary: {}", summary);
    assert!(summary.contains("Acc:"), "summary: {}", summary);
    assert!(summary.contains("TPS:"), "summary: {}", summary);
}
