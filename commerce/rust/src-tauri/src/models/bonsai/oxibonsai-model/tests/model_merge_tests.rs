//! Integration tests for the model_merge module.
//!
//! Covers WeightTensor primitives, low-level merge functions, and the
//! high-level `merge_models` / `merge_models_with_stats` APIs.

use oxibonsai_model::model_merge::{
    dare_merge, linear_merge, merge_models, merge_models_with_stats, merge_tensors, slerp,
    task_vector_merge, ties_merge, MergeConfig, MergeError, MergeMethod, WeightTensor,
};

// ──────────────────────────────────────────────────────────────────
// Helper constructors
// ──────────────────────────────────────────────────────────────────

fn tensor(name: &str, data: Vec<f32>) -> WeightTensor {
    let n = data.len();
    WeightTensor::new(name, data, vec![n])
}

fn tensor_2d(name: &str, data: Vec<f32>, rows: usize, cols: usize) -> WeightTensor {
    WeightTensor::new(name, data, vec![rows, cols])
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

fn slice_approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
}

// ──────────────────────────────────────────────────────────────────
// 1. weight_tensor_element_count
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_element_count() {
    let t = tensor_2d("w", vec![1.0; 6], 2, 3);
    assert_eq!(t.element_count(), 6, "shape [2,3] should have 6 elements");
}

// ──────────────────────────────────────────────────────────────────
// 2. weight_tensor_l2_norm
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_l2_norm() {
    // [3, 4] → sqrt(9 + 16) = 5
    let t = tensor("w", vec![3.0, 4.0]);
    assert!(approx_eq(t.l2_norm(), 5.0), "l2_norm([3,4]) should be 5.0");
}

// ──────────────────────────────────────────────────────────────────
// 3. weight_tensor_cosine_similarity_parallel
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_cosine_similarity_parallel() {
    let a = tensor("a", vec![1.0, 2.0, 3.0]);
    let b = tensor("b", vec![1.0, 2.0, 3.0]);
    let sim = a
        .cosine_similarity(&b)
        .expect("should not fail for parallel vectors");
    assert!(
        approx_eq(sim, 1.0),
        "identical vectors should have cosine sim 1.0, got {sim}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 4. weight_tensor_cosine_similarity_orthogonal
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_cosine_similarity_orthogonal() {
    let a = tensor("a", vec![1.0, 0.0]);
    let b = tensor("b", vec![0.0, 1.0]);
    let sim = a
        .cosine_similarity(&b)
        .expect("should not fail for orthogonal vectors");
    assert!(
        approx_eq(sim, 0.0),
        "orthogonal vectors should have cosine sim 0.0, got {sim}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 5. weight_tensor_add_correct
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_add_correct() {
    let a = tensor("a", vec![1.0, 2.0]);
    let b = tensor("b", vec![3.0, 4.0]);
    let result = a.add(&b).expect("add should succeed");
    assert!(
        slice_approx_eq(&result.data, &[4.0, 6.0]),
        "expected [4,6], got {:?}",
        result.data
    );
}

// ──────────────────────────────────────────────────────────────────
// 6. weight_tensor_sub_correct
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_sub_correct() {
    let a = tensor("a", vec![3.0, 4.0]);
    let b = tensor("b", vec![1.0, 2.0]);
    let result = a.sub(&b).expect("sub should succeed");
    assert!(
        slice_approx_eq(&result.data, &[2.0, 2.0]),
        "expected [2,2], got {:?}",
        result.data
    );
}

// ──────────────────────────────────────────────────────────────────
// 7. weight_tensor_scale
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_scale() {
    let a = tensor("a", vec![1.0, 2.0]);
    let result = a.scale(2.0);
    assert!(
        slice_approx_eq(&result.data, &[2.0, 4.0]),
        "expected [2,4], got {:?}",
        result.data
    );
}

// ──────────────────────────────────────────────────────────────────
// 8. weight_tensor_lerp_endpoints
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_lerp_endpoints() {
    let a = tensor("a", vec![1.0, 2.0, 3.0]);
    let b = tensor("b", vec![7.0, 8.0, 9.0]);

    // t=0.0 → must match a
    let at_zero = a.lerp(&b, 0.0).expect("lerp(0) should succeed");
    assert!(
        slice_approx_eq(&at_zero.data, &a.data),
        "lerp at t=0 should equal a, got {:?}",
        at_zero.data
    );

    // t=1.0 → must match b
    let at_one = a.lerp(&b, 1.0).expect("lerp(1) should succeed");
    assert!(
        slice_approx_eq(&at_one.data, &b.data),
        "lerp at t=1 should equal b, got {:?}",
        at_one.data
    );
}

// ──────────────────────────────────────────────────────────────────
// 9. linear_merge_midpoint
// ──────────────────────────────────────────────────────────────────

#[test]
fn linear_merge_midpoint() {
    let a = vec![0.0, 0.0];
    let b = vec![2.0, 2.0];
    let result = linear_merge(&a, &b, 0.5);
    assert!(
        slice_approx_eq(&result, &[1.0, 1.0]),
        "midpoint of [0,0] and [2,2] should be [1,1]"
    );
}

// ──────────────────────────────────────────────────────────────────
// 10. linear_merge_alpha_zero
// ──────────────────────────────────────────────────────────────────

#[test]
fn linear_merge_alpha_zero() {
    let a = vec![3.0, 5.0, 7.0];
    let b = vec![100.0, 200.0, 300.0];
    let result = linear_merge(&a, &b, 0.0);
    assert!(
        slice_approx_eq(&result, &a),
        "alpha=0 should return a unchanged"
    );
}

// ──────────────────────────────────────────────────────────────────
// 11. linear_merge_alpha_one
// ──────────────────────────────────────────────────────────────────

#[test]
fn linear_merge_alpha_one() {
    let a = vec![3.0, 5.0, 7.0];
    let b = vec![100.0, 200.0, 300.0];
    let result = linear_merge(&a, &b, 1.0);
    assert!(
        slice_approx_eq(&result, &b),
        "alpha=1 should return b unchanged"
    );
}

// ──────────────────────────────────────────────────────────────────
// 12. slerp_t_zero_returns_a
// ──────────────────────────────────────────────────────────────────

#[test]
fn slerp_t_zero_returns_a() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = slerp(&a, &b, 0.0);
    assert!(
        slice_approx_eq(&result, &a),
        "slerp at t=0 should return a direction, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 13. slerp_t_one_returns_b
// ──────────────────────────────────────────────────────────────────

#[test]
fn slerp_t_one_returns_b() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = slerp(&a, &b, 1.0);
    assert!(
        slice_approx_eq(&result, &b),
        "slerp at t=1 should return b direction, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 14. slerp_parallel_falls_back_to_linear
// ──────────────────────────────────────────────────────────────────

#[test]
fn slerp_parallel_falls_back_to_linear() {
    // Nearly parallel: a ≈ 1.0001 * b (cos_theta > 0.9995)
    let a = vec![1.0, 0.0];
    let b = vec![1.0001, 0.0]; // extremely close direction
    let t = 0.5;
    let slerp_result = slerp(&a, &b, t);
    let linear_result = linear_merge(&a, &b, t);
    // Both should give the same answer when falling back
    assert!(
        slice_approx_eq(&slerp_result, &linear_result),
        "nearly-parallel slerp should equal linear; slerp={slerp_result:?}, linear={linear_result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 15. ties_merge_density_one
// ──────────────────────────────────────────────────────────────────

#[test]
fn ties_merge_density_one() {
    // density=1.0 means no trimming; with alpha=0.5 and same-sign vectors,
    // result should be close to linear average.
    let a = vec![2.0, 4.0, 6.0];
    let b = vec![4.0, 8.0, 12.0];
    let result = ties_merge(&a, &b, 0.5, 1.0);
    let linear = linear_merge(&a, &b, 0.5);
    assert!(
        slice_approx_eq(&result, &linear),
        "ties with density=1.0 and same-sign should equal linear; ties={result:?}, linear={linear:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 16. ties_merge_trims_small_weights
// ──────────────────────────────────────────────────────────────────

#[test]
fn ties_merge_trims_small_weights() {
    // Only keep the top 20% by magnitude → only the largest element survives
    let a = vec![0.01, 0.02, 10.0, 0.03, 0.04];
    let b = vec![0.01, 0.02, 10.0, 0.03, 0.04];
    let result = ties_merge(&a, &b, 0.5, 0.2);
    // The four small elements should be zeroed out
    for &v in &result[..2] {
        assert_eq!(v, 0.0, "small elements should be trimmed to zero");
    }
    for &v in &result[3..] {
        assert_eq!(v, 0.0, "small elements should be trimmed to zero");
    }
    // The large element (10.0) should survive
    assert!(result[2] != 0.0, "large element should survive trimming");
}

// ──────────────────────────────────────────────────────────────────
// 17. task_vector_alpha_zero
// ──────────────────────────────────────────────────────────────────

#[test]
fn task_vector_alpha_zero() {
    let base = vec![1.0, 2.0, 3.0];
    let finetuned = vec![10.0, 20.0, 30.0];
    let result = task_vector_merge(&base, &finetuned, 0.0);
    assert!(
        slice_approx_eq(&result, &base),
        "task_vector with alpha=0 should return base, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 18. task_vector_alpha_one
// ──────────────────────────────────────────────────────────────────

#[test]
fn task_vector_alpha_one() {
    let base = vec![1.0, 2.0, 3.0];
    let finetuned = vec![10.0, 20.0, 30.0];
    let result = task_vector_merge(&base, &finetuned, 1.0);
    assert!(
        slice_approx_eq(&result, &finetuned),
        "task_vector with alpha=1 should return finetuned, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 19. dare_merge_deterministic
// ──────────────────────────────────────────────────────────────────

#[test]
fn dare_merge_deterministic() {
    let base = vec![0.0; 100];
    let finetuned: Vec<f32> = (1..=100).map(|i| i as f32 * 0.1).collect();
    let r1 = dare_merge(&base, &finetuned, 0.5, 0.3, 12345);
    let r2 = dare_merge(&base, &finetuned, 0.5, 0.3, 12345);
    assert_eq!(
        r1, r2,
        "dare_merge should be deterministic with the same seed"
    );

    // Different seed → different result (with high probability on 100 elements)
    let r3 = dare_merge(&base, &finetuned, 0.5, 0.3, 99999);
    assert_ne!(
        r1, r3,
        "different seeds should (almost certainly) produce different results"
    );
}

// ──────────────────────────────────────────────────────────────────
// 20. merge_tensors_linear
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_tensors_linear() {
    let a = tensor("w", vec![0.0, 0.0, 0.0]);
    let b = tensor("w", vec![2.0, 4.0, 6.0]);
    let config = MergeConfig {
        method: MergeMethod::Linear,
        alpha: 0.5,
        ..Default::default()
    };
    let result = merge_tensors(&a, &b, &config).expect("merge_tensors should succeed");
    assert!(
        slice_approx_eq(&result.data, &[1.0, 2.0, 3.0]),
        "linear midpoint should be [1,2,3], got {:?}",
        result.data
    );
}

// ──────────────────────────────────────────────────────────────────
// 21. merge_tensors_slerp
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_tensors_slerp() {
    // Use orthogonal unit vectors for a clean slerp test
    let a = tensor("w", vec![1.0, 0.0]);
    let b = tensor("w", vec![0.0, 1.0]);
    let config = MergeConfig {
        method: MergeMethod::Slerp,
        alpha: 0.5,
        ..Default::default()
    };
    let result = merge_tensors(&a, &b, &config).expect("slerp should succeed for unit vectors");
    // At t=0.5, SLERP of two orthogonal unit vectors gives [1/√2, 1/√2]
    let expected = 1.0_f32 / 2.0_f32.sqrt();
    assert!(
        approx_eq(result.data[0], expected),
        "slerp midpoint x should be 1/√2 ≈ {expected}, got {}",
        result.data[0]
    );
    assert!(
        approx_eq(result.data[1], expected),
        "slerp midpoint y should be 1/√2 ≈ {expected}, got {}",
        result.data[1]
    );
}

// ──────────────────────────────────────────────────────────────────
// 22. merge_models_basic
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_models_basic() {
    let base = vec![
        tensor("layer.0.weight", vec![1.0, 2.0]),
        tensor("layer.1.weight", vec![3.0, 4.0]),
    ];
    let other = vec![tensor("layer.0.weight", vec![3.0, 4.0])];
    let config = MergeConfig {
        method: MergeMethod::Linear,
        alpha: 0.5,
        ..Default::default()
    };
    let merged = merge_models(&base, &other, &config).expect("merge_models should succeed");
    assert_eq!(
        merged.len(),
        2,
        "output should have same tensor count as base"
    );

    // Overlapping tensor: should be averaged
    assert!(
        slice_approx_eq(&merged[0].data, &[2.0, 3.0]),
        "overlapping tensor should be averaged, got {:?}",
        merged[0].data
    );
    // Non-overlapping tensor: should be copied from base unchanged
    assert!(
        slice_approx_eq(&merged[1].data, &[3.0, 4.0]),
        "non-overlapping tensor should be copied from base, got {:?}",
        merged[1].data
    );
}

// ──────────────────────────────────────────────────────────────────
// 23. merge_models_only_base
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_models_only_base() {
    let base = vec![
        tensor("a.weight", vec![5.0, 6.0]),
        tensor("b.weight", vec![7.0, 8.0]),
    ];
    let other: Vec<WeightTensor> = vec![]; // empty: no tensors to merge
    let config = MergeConfig::default();
    let merged =
        merge_models(&base, &other, &config).expect("merge with empty other should succeed");
    assert_eq!(merged.len(), 2);
    assert!(
        slice_approx_eq(&merged[0].data, &[5.0, 6.0]),
        "base-only tensor should be unchanged"
    );
    assert!(
        slice_approx_eq(&merged[1].data, &[7.0, 8.0]),
        "base-only tensor should be unchanged"
    );
}

// ──────────────────────────────────────────────────────────────────
// 24. merge_models_with_stats_counts
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_models_with_stats_counts() {
    let base = vec![
        tensor("shared.weight", vec![1.0, 2.0, 3.0]),
        tensor("base_only.weight", vec![9.0, 9.0]),
    ];
    let other = vec![tensor("shared.weight", vec![4.0, 5.0, 6.0])];
    let config = MergeConfig::default();
    let (_merged, stats) = merge_models_with_stats(&base, &other, &config)
        .expect("merge_models_with_stats should succeed");

    assert_eq!(stats.tensors_merged, 1, "exactly 1 tensor should be merged");
    assert_eq!(stats.tensors_copied, 1, "exactly 1 tensor should be copied");
    assert_eq!(
        stats.total_params,
        3 + 2,
        "total_params should sum element counts of all base tensors"
    );
}

// ──────────────────────────────────────────────────────────────────
// 25. merge_stats_summary_nonempty
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_stats_summary_nonempty() {
    let base = vec![tensor("w", vec![1.0, 2.0])];
    let other = vec![tensor("w", vec![3.0, 4.0])];
    let config = MergeConfig::default();
    let (_merged, stats) = merge_models_with_stats(&base, &other, &config).expect("should succeed");
    let summary = stats.summary();
    assert!(!summary.is_empty(), "summary string should not be empty");
    assert!(
        summary.contains("merged="),
        "summary should contain 'merged=', got: {summary}"
    );
}

// ──────────────────────────────────────────────────────────────────
// 26. merge_error_invalid_alpha
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_error_invalid_alpha() {
    let a = tensor("w", vec![1.0, 2.0]);
    let b = tensor("w", vec![3.0, 4.0]);
    let config = MergeConfig {
        alpha: 1.5, // invalid: > 1.0
        ..Default::default()
    };
    let result = merge_tensors(&a, &b, &config);
    assert!(
        matches!(result, Err(MergeError::InvalidAlpha(_))),
        "alpha > 1.0 should return InvalidAlpha, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: shape mismatch error
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_tensors_shape_mismatch_error() {
    let a = tensor_2d("w", vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = tensor_2d("w", vec![1.0, 2.0, 3.0], 3, 1);
    let config = MergeConfig::default();
    let result = merge_tensors(&a, &b, &config);
    assert!(
        matches!(result, Err(MergeError::ShapeMismatch { .. })),
        "incompatible shapes should return ShapeMismatch, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: DARE merge preserves expected sparsity
// ──────────────────────────────────────────────────────────────────

#[test]
fn dare_merge_sparsity_is_approximately_dropout_rate() {
    let n = 10_000usize;
    let base = vec![0.0f32; n];
    let finetuned = vec![1.0f32; n];
    let dropout_rate = 0.3;
    let result = dare_merge(&base, &finetuned, 1.0, dropout_rate, 777);

    // After dare_merge: kept elements are rescaled by 1/(1-dropout_rate) = 1/0.7 ≈ 1.4286
    // Zeroed elements = 0.0
    let zeros = result.iter().filter(|&&v| v == 0.0).count();
    let actual_rate = zeros as f32 / n as f32;
    // Allow ±3% deviation from the expected dropout_rate
    assert!(
        (actual_rate - dropout_rate).abs() < 0.03,
        "actual dropout rate {actual_rate} should be close to {dropout_rate}"
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: task vector midpoint is correct
// ──────────────────────────────────────────────────────────────────

#[test]
fn task_vector_midpoint() {
    let base = vec![0.0, 0.0, 0.0];
    let finetuned = vec![2.0, 4.0, 6.0];
    let result = task_vector_merge(&base, &finetuned, 0.5);
    assert!(
        slice_approx_eq(&result, &[1.0, 2.0, 3.0]),
        "task_vector at alpha=0.5 should give midpoint, got {result:?}"
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: WeightTensor::zeros constructor
// ──────────────────────────────────────────────────────────────────

#[test]
fn weight_tensor_zeros_constructor() {
    let z = WeightTensor::zeros("zero_w", vec![3, 4]);
    assert_eq!(z.element_count(), 12);
    assert!(
        z.data.iter().all(|&v| v == 0.0),
        "zeros tensor should have all-zero data"
    );
    assert_eq!(z.name, "zero_w");
}

// ──────────────────────────────────────────────────────────────────
// Bonus: MergeMethod equality
// ──────────────────────────────────────────────────────────────────

#[test]
fn merge_method_equality() {
    assert_eq!(MergeMethod::Linear, MergeMethod::Linear);
    assert_ne!(MergeMethod::Linear, MergeMethod::Slerp);
    assert_eq!(
        MergeMethod::Dare {
            seed: 42,
            dropout_rate: 0.3
        },
        MergeMethod::Dare {
            seed: 42,
            dropout_rate: 0.3
        }
    );
    assert_ne!(
        MergeMethod::Dare {
            seed: 42,
            dropout_rate: 0.3
        },
        MergeMethod::Dare {
            seed: 99,
            dropout_rate: 0.3
        }
    );
}
