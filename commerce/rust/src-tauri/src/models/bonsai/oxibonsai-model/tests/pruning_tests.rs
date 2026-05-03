//! Integration tests for the pruning module.

use oxibonsai_model::{
    compute_importance, model_sparsity_report, prune_model, prune_tensor, prune_tensor_inplace,
    ImportanceMetric, ModelSparsitySummary, PruningConfig, PruningGranularity, SparsityReport,
    WeightTensor,
};

// ──────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────

fn make_tensor(name: &str, data: Vec<f32>, shape: Vec<usize>) -> WeightTensor {
    WeightTensor::new(name, data, shape)
}

fn ascending_tensor(name: &str, n: usize) -> WeightTensor {
    let data: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    make_tensor(name, data, vec![n])
}

fn checkerboard_2d(name: &str, rows: usize, cols: usize) -> WeightTensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| if i % 2 == 0 { 1.0 } else { -2.0 })
        .collect();
    make_tensor(name, data, vec![rows, cols])
}

// ──────────────────────────────────────────────────────────────────
// 1. compute_importance_l1_values
// ──────────────────────────────────────────────────────────────────
#[test]
fn compute_importance_l1_values() {
    let t = make_tensor("w", vec![-3.0, 2.0, -1.0, 0.5], vec![4]);
    let scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
    assert_eq!(scores.scores.len(), 4);
    assert!(
        (scores.scores[0] - 3.0).abs() < 1e-6,
        "expected 3.0 got {}",
        scores.scores[0]
    );
    assert!((scores.scores[1] - 2.0).abs() < 1e-6);
    assert!((scores.scores[2] - 1.0).abs() < 1e-6);
    assert!((scores.scores[3] - 0.5).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────
// 2. compute_importance_l2_values
// ──────────────────────────────────────────────────────────────────
#[test]
fn compute_importance_l2_values() {
    let t = make_tensor("w", vec![2.0, -3.0, 0.5], vec![3]);
    let scores = compute_importance(&t, ImportanceMetric::L2Magnitude);
    assert!(
        (scores.scores[0] - 4.0).abs() < 1e-6,
        "expected 4.0 got {}",
        scores.scores[0]
    );
    assert!((scores.scores[1] - 9.0).abs() < 1e-6);
    assert!((scores.scores[2] - 0.25).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────
// 3. compute_importance_random_deterministic
// ──────────────────────────────────────────────────────────────────
#[test]
fn compute_importance_random_deterministic() {
    let t = make_tensor("w", vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let s1 = compute_importance(&t, ImportanceMetric::Random { seed: 42 });
    let s2 = compute_importance(&t, ImportanceMetric::Random { seed: 42 });
    assert_eq!(s1.scores, s2.scores, "same seed must yield same scores");

    // Different seed => different scores
    let s3 = compute_importance(&t, ImportanceMetric::Random { seed: 99 });
    assert_ne!(s1.scores, s3.scores, "different seeds should differ");
}

// ──────────────────────────────────────────────────────────────────
// 4. importance_scores_sparsity
// ──────────────────────────────────────────────────────────────────
#[test]
fn importance_scores_sparsity() {
    let t = ascending_tensor("w", 10);
    let mut scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
    // Set threshold so 3 of 10 scores (1.0, 2.0, 3.0) are below/at it
    scores.threshold = 3.0;
    let s = scores.sparsity();
    assert!((s - 0.3).abs() < 1e-5, "expected ~0.3 got {s}");
}

// ──────────────────────────────────────────────────────────────────
// 5. importance_scores_top_k
// ──────────────────────────────────────────────────────────────────
#[test]
fn importance_scores_top_k() {
    let t = make_tensor("w", vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]);
    let scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
    let top3 = scores.top_k(3);
    assert_eq!(top3.len(), 3);
    assert!((top3[0] - 5.0).abs() < 1e-6, "largest should be first");
    assert!((top3[1] - 4.0).abs() < 1e-6);
    assert!((top3[2] - 3.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────
// 6. score_stats_min_max
// ──────────────────────────────────────────────────────────────────
#[test]
fn score_stats_min_max() {
    let t = make_tensor("w", vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]);
    let scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
    let stats = scores.stats();
    assert!(stats.min <= stats.mean, "min <= mean");
    assert!(stats.mean <= stats.max, "mean <= max");
    assert!((stats.min - 1.0).abs() < 1e-6);
    assert!((stats.max - 5.0).abs() < 1e-6);
    assert!((stats.mean - 3.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────
// 7. prune_tensor_sparsity_50pct
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_sparsity_50pct() {
    let t = ascending_tensor("w", 10);
    let config = PruningConfig::unstructured_l1(0.5);
    let (pruned, _mask) = prune_tensor(&t, &config).expect("prune ok");
    let zeros = pruned.data.iter().filter(|&&x| x == 0.0).count();
    assert_eq!(zeros, 5, "expected 5 zeros, got {zeros}");
}

// ──────────────────────────────────────────────────────────────────
// 8. prune_tensor_sparsity_zero
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_sparsity_zero() {
    let t = ascending_tensor("w", 8);
    let config = PruningConfig::unstructured_l1(0.0);
    let (pruned, mask) = prune_tensor(&t, &config).expect("prune ok");
    assert_eq!(pruned.data, t.data, "0% sparsity must leave data unchanged");
    assert!(
        mask.iter().all(|&m| m == 1.0),
        "all mask entries should be 1.0"
    );
}

// ──────────────────────────────────────────────────────────────────
// 9. prune_tensor_mask_ones_and_zeros
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_mask_ones_and_zeros() {
    let t = ascending_tensor("w", 12);
    let config = PruningConfig::unstructured_l1(0.4);
    let (_pruned, mask) = prune_tensor(&t, &config).expect("prune ok");
    for &m in &mask {
        assert!(m == 0.0 || m == 1.0, "mask must be binary, got {m}");
    }
}

// ──────────────────────────────────────────────────────────────────
// 10. prune_tensor_min_nonzero_respected
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_min_nonzero_respected() {
    let t = ascending_tensor("w", 10);
    let mut config = PruningConfig::unstructured_l1(0.9); // wants to prune 90%
    config.min_nonzero = 5; // but must keep at least 5
    let (pruned, _mask) = prune_tensor(&t, &config).expect("prune ok");
    let nonzero = pruned.data.iter().filter(|&&x| x != 0.0).count();
    assert!(nonzero >= 5, "must keep at least 5 nonzero, got {nonzero}");
}

// ──────────────────────────────────────────────────────────────────
// 11. prune_tensor_keeps_largest
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_keeps_largest() {
    // values 1..10; pruning 70% should keep 7,8,9,10 (top 4 roughly, since 30%)
    // Actually 70% of 10 = 7 pruned, so top 3 kept: 8, 9, 10
    let t = ascending_tensor("w", 10);
    let config = PruningConfig::unstructured_l1(0.7);
    let (pruned, _mask) = prune_tensor(&t, &config).expect("prune ok");
    // The largest weights (indices 7, 8, 9 = values 8, 9, 10) must survive
    assert_ne!(pruned.data[9], 0.0, "value 10 must be kept");
    assert_ne!(pruned.data[8], 0.0, "value 9 must be kept");
    assert_ne!(pruned.data[7], 0.0, "value 8 must be kept");
}

// ──────────────────────────────────────────────────────────────────
// 12. prune_tensor_inplace_same_result
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_inplace_same_result() {
    let t = ascending_tensor("w", 8);
    let config = PruningConfig::unstructured_l1(0.5);

    let (out_copy, mask_copy) = prune_tensor(&t, &config).expect("prune ok");

    let mut t_inplace = t.clone();
    let mask_inplace = prune_tensor_inplace(&mut t_inplace, &config).expect("inplace ok");

    assert_eq!(out_copy.data, t_inplace.data, "data should match");
    assert_eq!(mask_copy, mask_inplace, "masks should match");
}

// ──────────────────────────────────────────────────────────────────
// 13. prune_model_all_tensors
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_model_all_tensors() {
    let tensors = vec![
        ascending_tensor("layer0.weight", 10),
        ascending_tensor("layer1.weight", 8),
        ascending_tensor("layer2.weight", 6),
    ];
    let config = PruningConfig::unstructured_l1(0.3);
    let pruned = prune_model(&tensors, &config).expect("prune model ok");
    assert_eq!(
        pruned.len(),
        tensors.len(),
        "output count must match input count"
    );
}

// ──────────────────────────────────────────────────────────────────
// 14. prune_model_reduces_nonzero
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_model_reduces_nonzero() {
    let tensors = vec![
        ascending_tensor("layer0.weight", 10),
        ascending_tensor("layer1.weight", 10),
    ];
    let before: usize = tensors
        .iter()
        .flat_map(|t| t.data.iter())
        .filter(|&&x| x != 0.0)
        .count();
    let config = PruningConfig::unstructured_l1(0.4);
    let pruned = prune_model(&tensors, &config).expect("prune model ok");
    let after: usize = pruned
        .iter()
        .flat_map(|t| t.data.iter())
        .filter(|&&x| x != 0.0)
        .count();
    assert!(
        after < before,
        "pruning must reduce nonzero count: before={before} after={after}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 15. sparsity_report_compute
// ──────────────────────────────────────────────────────────────────
#[test]
fn sparsity_report_compute() {
    let data = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    let t = make_tensor("w", data, vec![5]);
    let report = SparsityReport::compute(&t);
    assert_eq!(report.total_params, 5);
    assert_eq!(report.nonzero_params, 3);
}

// ──────────────────────────────────────────────────────────────────
// 16. sparsity_report_zero_fraction
// ──────────────────────────────────────────────────────────────────
#[test]
fn sparsity_report_zero_fraction() {
    let data = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    let t = make_tensor("w", data, vec![5]);
    let report = SparsityReport::compute(&t);
    assert!((report.zero_fraction() - report.sparsity).abs() < 1e-6);
    assert!((report.zero_fraction() - 0.4).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────
// 17. sparsity_report_density
// ──────────────────────────────────────────────────────────────────
#[test]
fn sparsity_report_density() {
    let data = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    let t = make_tensor("w", data, vec![5]);
    let report = SparsityReport::compute(&t);
    let sum = report.sparsity + report.density();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "sparsity + density must equal 1.0, got {sum}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 18. sparsity_report_summary_nonempty
// ──────────────────────────────────────────────────────────────────
#[test]
fn sparsity_report_summary_nonempty() {
    let t = ascending_tensor("layer.weight", 6);
    let report = SparsityReport::compute(&t);
    let s = report.summary();
    assert!(!s.is_empty(), "summary must not be empty");
    assert!(
        s.contains("layer.weight"),
        "summary should include tensor name"
    );
}

// ──────────────────────────────────────────────────────────────────
// 19. model_sparsity_report_count
// ──────────────────────────────────────────────────────────────────
#[test]
fn model_sparsity_report_count() {
    let tensors = vec![
        ascending_tensor("a", 4),
        ascending_tensor("b", 8),
        ascending_tensor("c", 6),
    ];
    let reports = model_sparsity_report(&tensors);
    assert_eq!(reports.len(), 3, "one report per tensor");
}

// ──────────────────────────────────────────────────────────────────
// 20. model_sparsity_summary_overall
// ──────────────────────────────────────────────────────────────────
#[test]
fn model_sparsity_summary_overall() {
    // Layer A: 50% sparse (half zeros)
    let data_a = vec![0.0, 1.0, 0.0, 1.0];
    // Layer B: 25% sparse (one zero)
    let data_b = vec![0.0, 1.0, 1.0, 1.0];
    let tensors = vec![
        make_tensor("a", data_a, vec![4]),
        make_tensor("b", data_b, vec![4]),
    ];
    let summary = ModelSparsitySummary::from_model(&tensors);
    // Overall: 3 zeros / 8 total = 0.375
    assert!(
        (summary.overall_sparsity - 0.375).abs() < 1e-5,
        "expected 0.375 got {}",
        summary.overall_sparsity
    );

    let layer_min = summary
        .layer_reports
        .iter()
        .map(|r| r.sparsity)
        .fold(f32::INFINITY, f32::min);
    let layer_max = summary
        .layer_reports
        .iter()
        .map(|r| r.sparsity)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(summary.overall_sparsity >= layer_min - 1e-5);
    assert!(summary.overall_sparsity <= layer_max + 1e-5);
}

// ──────────────────────────────────────────────────────────────────
// 21. model_sparsity_summary_nonempty
// ──────────────────────────────────────────────────────────────────
#[test]
fn model_sparsity_summary_nonempty() {
    let tensors = vec![ascending_tensor("w", 10)];
    let summary = ModelSparsitySummary::from_model(&tensors);
    let s = summary.summary();
    assert!(!s.is_empty());
    assert!(s.contains("layers=1"));
}

// ──────────────────────────────────────────────────────────────────
// 22. pruning_config_unstructured_l1
// ──────────────────────────────────────────────────────────────────
#[test]
fn pruning_config_unstructured_l1() {
    let config = PruningConfig::unstructured_l1(0.6);
    assert!((config.sparsity - 0.6).abs() < 1e-6);
    assert_eq!(config.metric, ImportanceMetric::L1Magnitude);
    assert_eq!(config.granularity, PruningGranularity::Unstructured);
    assert!(config.min_nonzero >= 1);
}

// ──────────────────────────────────────────────────────────────────
// 23. prune_tensor_invalid_sparsity
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_tensor_invalid_sparsity() {
    let t = ascending_tensor("w", 5);
    let config = PruningConfig::unstructured_l1(1.0); // invalid: must be < 1.0
    let result = prune_tensor(&t, &config);
    assert!(result.is_err(), "sparsity=1.0 must return an error");
}

// ──────────────────────────────────────────────────────────────────
// 24. structured_row_prune
// ──────────────────────────────────────────────────────────────────
#[test]
fn structured_row_prune() {
    // 4 rows x 3 cols; rows have L2 norms 1, 2, 3, 4 (ascending)
    // Prune 50% of rows → prune rows 0 and 1 (lowest norms)
    let data: Vec<f32> = vec![
        1.0, 0.0, 0.0, // row 0: norm = 1.0
        2.0, 0.0, 0.0, // row 1: norm = 2.0
        3.0, 0.0, 0.0, // row 2: norm = 3.0
        4.0, 0.0, 0.0, // row 3: norm = 4.0
    ];
    let t = make_tensor("w", data, vec![4, 3]);
    let config = PruningConfig::structured_row_l2(0.5);
    let (pruned, mask) = prune_tensor(&t, &config).expect("prune ok");

    // Rows 0 and 1 (first 6 elements) must be all zeros
    for (i, (&pruned_val, &mask_val)) in pruned.data[..6].iter().zip(mask[..6].iter()).enumerate() {
        assert_eq!(pruned_val, 0.0, "row 0/1 element {i} must be zero");
        assert_eq!(mask_val, 0.0, "mask row 0/1 element {i} must be 0.0");
    }
    // Rows 2 and 3 must still contain original non-zero values
    assert_ne!(pruned.data[6], 0.0, "row 2 col 0 must survive");
    assert_ne!(pruned.data[9], 0.0, "row 3 col 0 must survive");
}

// ──────────────────────────────────────────────────────────────────
// Bonus: taylor proxy same as l2
// ──────────────────────────────────────────────────────────────────
#[test]
fn taylor_proxy_equals_l2_scores() {
    let t = make_tensor("w", vec![1.0, -2.0, 3.0], vec![3]);
    let l2 = compute_importance(&t, ImportanceMetric::L2Magnitude);
    let tp = compute_importance(&t, ImportanceMetric::TaylorProxy);
    for (a, b) in l2.scores.iter().zip(tp.scores.iter()) {
        assert!((a - b).abs() < 1e-6, "TaylorProxy must match L2 scores");
    }
}

// ──────────────────────────────────────────────────────────────────
// structured pruning: non-2D tensor returns error
// ──────────────────────────────────────────────────────────────────
#[test]
fn structured_row_prune_non_2d_error() {
    let t = ascending_tensor("w", 8); // shape=[8], 1D
    let config = PruningConfig::structured_row_l2(0.5);
    let result = prune_tensor(&t, &config);
    assert!(
        result.is_err(),
        "1D tensor with structured pruning must error"
    );
}

// ──────────────────────────────────────────────────────────────────
// score stats: std_dev non-negative
// ──────────────────────────────────────────────────────────────────
#[test]
fn score_stats_std_dev_nonneg() {
    let t = make_tensor("w", vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
    let stats = scores.stats();
    assert!(stats.std_dev >= 0.0, "std_dev must be non-negative");
}

// ──────────────────────────────────────────────────────────────────
// checkerboard 2D unstructured prune
// ──────────────────────────────────────────────────────────────────
#[test]
fn prune_checkerboard_unstructured() {
    let t = checkerboard_2d("w", 4, 4); // values alternating 1.0 and -2.0
    let config = PruningConfig::unstructured_l1(0.5);
    let (pruned, _mask) = prune_tensor(&t, &config).expect("prune ok");
    let zeros = pruned.data.iter().filter(|&&x| x == 0.0).count();
    assert_eq!(zeros, 8, "expected 8 zeros from 50% prune of 16 elements");
}
