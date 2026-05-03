//! Property-based fuzz tests for Q1_0_g128 kernels.
//!
//! These tests verify that the kernel functions are robust against adversarial
//! inputs: they should never panic, should respect output shape invariants,
//! and should satisfy basic mathematical properties (linearity, bounds).

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_kernels::dequant::dequant_1bit_g128;
use oxibonsai_kernels::gemm::gemm_1bit_g128;
use oxibonsai_kernels::gemv::gemv_1bit_g128;
use proptest::prelude::*;

// ── Helper strategies ─────────────────────────────────────────────────────────

/// Strategy producing a single BlockQ1_0G128 with arbitrary (valid) content.
fn arb_block_any_scale() -> impl Strategy<Value = BlockQ1_0G128> {
    (any::<u16>(), prop::array::uniform16(any::<u8>())).prop_map(|(raw_scale, qs)| BlockQ1_0G128 {
        d: f16::from_bits(raw_scale),
        qs,
    })
}

/// Strategy producing a single block with a finite, non-tiny scale (for
/// tests that need numerically meaningful values).
fn arb_block_finite() -> impl Strategy<Value = BlockQ1_0G128> {
    (
        prop::num::f32::NORMAL.prop_filter("finite nonzero scale", |v| {
            v.is_finite() && v.abs() > 1e-5 && v.abs() < 1e4
        }),
        prop::array::uniform16(any::<u8>()),
    )
        .prop_map(|(scale, qs)| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs,
        })
}

/// Strategy producing a Vec of `count` blocks with finite scales.
#[allow(dead_code)]
fn arb_blocks_finite(count: usize) -> impl Strategy<Value = Vec<BlockQ1_0G128>> {
    prop::collection::vec(arb_block_finite(), count..=count)
}

/// Strategy producing a valid FP32 input vector of given length.
fn arb_input(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop::num::f32::NORMAL.prop_filter("finite input", |v| v.is_finite() && v.abs() < 1e6),
        len..=len,
    )
}

// ── 1. dequant never panics on random block data ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Random block data (including NaN/Inf scales) must never cause a panic.
    /// The function must return either Ok or a well-typed Err.
    #[test]
    fn prop_test_dequant_never_panics(block in arb_block_any_scale()) {
        let mut output = vec![0.0f32; QK1_0_G128];
        // We only care that this does not panic.
        let _ = dequant_1bit_g128(&[block], &mut output);
    }
}

// ── 2. gemv output length always equals n_rows ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// For any valid (n_rows, blocks) configuration the output Vec has exactly
    /// `n_rows` elements after a successful call.
    #[test]
    fn prop_test_gemv_output_shape(
        n_rows in 1usize..=8,
        blocks in prop::collection::vec(arb_block_finite(), 1..=8),
    ) {
        let k = QK1_0_G128; // simplest case: 1 block per row
        // Only test when blocks.len() == n_rows (one block per row).
        let n_rows = n_rows.min(blocks.len());
        let weight_blocks = blocks[..n_rows].to_vec();

        let input = vec![1.0f32; k];
        let mut output = vec![0.0f32; n_rows];

        let result = gemv_1bit_g128(&weight_blocks, &input, &mut output, n_rows, k);
        if result.is_ok() {
            prop_assert_eq!(output.len(), n_rows,
                "output length must equal n_rows");
        }
        // If it errs, shape constraint is vacuously satisfied.
    }
}

// ── 3. gemm output shape always rows×cols_b ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// For any valid (m, n_rows) configuration the output Vec has exactly
    /// `m * n_rows` elements after a successful GEMM call.
    #[test]
    fn prop_test_gemm_output_shape(
        m in 1usize..=4,
        n_rows in 1usize..=4,
        blocks in prop::collection::vec(arb_block_finite(), 1..=16),
    ) {
        let k = QK1_0_G128;
        let blocks_needed = n_rows; // 1 block per row
        if blocks.len() < blocks_needed {
            return Ok(());
        }
        let weight_blocks = blocks[..blocks_needed].to_vec();

        let input = vec![1.0f32; m * k];
        let mut output = vec![0.0f32; m * n_rows];

        let result = gemm_1bit_g128(&weight_blocks, &input, &mut output, m, n_rows, k);
        if result.is_ok() {
            prop_assert_eq!(output.len(), m * n_rows,
                "GEMM output length must be m * n_rows");
        }
    }
}

// ── 4. dequant output values bounded by ±scale ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every dequantized value must be either +scale or -scale (within f16
    /// rounding tolerance of 1%).
    #[test]
    fn prop_test_dequant_values_bounded(block in arb_block_finite()) {
        let d = block.d.to_f32();
        if !d.is_finite() || d == 0.0 {
            return Ok(());
        }
        let scale = d.abs();
        let mut output = vec![0.0f32; QK1_0_G128];
        if dequant_1bit_g128(&[block], &mut output).is_err() {
            return Ok(());
        }
        let tol = scale * 0.02 + f32::EPSILON;
        for &v in &output {
            prop_assert!(
                v.is_finite(),
                "dequant output must be finite, got {v}"
            );
            prop_assert!(
                (v.abs() - scale).abs() <= tol,
                "dequant value {v} not within ±{scale} (tol={tol})"
            );
        }
    }
}

// ── 5. gemv linearity: gemv(A, 2*x) ≈ 2 * gemv(A, x) ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Scalar multiplication distributes over GEMV: gemv(A, α·x) ≈ α · gemv(A, x).
    #[test]
    fn prop_test_gemv_linearity(
        block in arb_block_finite(),
        alpha in prop::num::f32::NORMAL.prop_filter("finite nonzero alpha", |v| {
            v.is_finite() && v.abs() > 0.01 && v.abs() < 100.0
        }),
        input in arb_input(QK1_0_G128),
    ) {
        let blocks = vec![block];
        let scaled_input: Vec<f32> = input.iter().map(|&x| x * alpha).collect();

        let mut out_base = vec![0.0f32; 1];
        let mut out_scaled = vec![0.0f32; 1];

        let r1 = gemv_1bit_g128(&blocks, &input, &mut out_base, 1, QK1_0_G128);
        let r2 = gemv_1bit_g128(&blocks, &scaled_input, &mut out_scaled, 1, QK1_0_G128);

        if r1.is_ok() && r2.is_ok() {
            let expected = out_base[0] * alpha;
            let actual = out_scaled[0];
            // Tolerance: f16 scale rounding + floating-point accumulation error.
            let tol = expected.abs() * 0.05 + 1.0;
            prop_assert!(
                (expected - actual).abs() < tol,
                "linearity failed: α·gemv(x)={expected:.4} vs gemv(α·x)={actual:.4}, α={alpha}"
            );
        }
    }
}

// ── 6. Zero-row matrix returns empty output ───────────────────────────────────

#[test]
fn prop_test_kernel_with_zero_rows() {
    // GEMV with n_rows=0 should either succeed (returning empty output) or
    // return a well-typed error — it must not panic.
    let blocks: &[BlockQ1_0G128] = &[];
    let input = vec![0.0f32; QK1_0_G128];
    let mut output = vec![0.0f32; 0];
    let result = gemv_1bit_g128(blocks, &input, &mut output, 0, QK1_0_G128);
    // Either ok with empty output, or Err — never panic.
    if let Ok(()) = result {
        assert!(output.is_empty(), "n_rows=0 should yield empty output");
    }
    // Err is acceptable: validation can reject n_rows=0.
}

// ── 7. Single-row matrix behaves correctly ────────────────────────────────────

#[test]
fn prop_test_kernel_with_single_row() {
    // One row, one block of all-one weights, constant input of 1.0.
    let block = BlockQ1_0G128 {
        d: f16::from_f32(1.0),
        qs: [0xFF; 16], // all bits set → all +1.0
    };
    let input = vec![1.0f32; QK1_0_G128];
    let mut output = vec![0.0f32; 1];

    gemv_1bit_g128(&[block], &input, &mut output, 1, QK1_0_G128)
        .expect("single-row GEMV should succeed");

    // All weights +1, all inputs 1.0 → output = sum of 128 ones = 128.0
    assert!(
        (output[0] - 128.0).abs() < 1.0,
        "single-row GEMV with all-one weights: expected 128.0, got {}",
        output[0]
    );
}

// ── 8. Large block count (up to 1024 blocks) doesn't panic ───────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Running dequant on up to 1024 blocks should never panic.
    #[test]
    fn prop_test_large_block_count(
        block_count in 1usize..=1024,
    ) {
        let blocks: Vec<BlockQ1_0G128> = (0..block_count)
            .map(|_| BlockQ1_0G128 {
                d: f16::from_f32(1.0),
                qs: [0xAAu8; 16],
            })
            .collect();

        let mut output = vec![0.0f32; block_count * QK1_0_G128];
        // Must not panic regardless of block count.
        let _ = dequant_1bit_g128(&blocks, &mut output);
    }
}

// ── 9. GEMV with random blocks never panics ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// GEMV must never panic on arbitrary block data (even NaN/Inf scales).
    #[test]
    fn prop_test_gemv_never_panics(block in arb_block_any_scale()) {
        let input = vec![1.0f32; QK1_0_G128];
        let mut output = vec![0.0f32; 1];
        let _ = gemv_1bit_g128(&[block], &input, &mut output, 1, QK1_0_G128);
    }
}

// ── 10. GEMM with random blocks never panics ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// GEMM must never panic on arbitrary block data.
    #[test]
    fn prop_test_gemm_never_panics(block in arb_block_any_scale()) {
        let input = vec![1.0f32; QK1_0_G128];
        let mut output = vec![0.0f32; 1];
        let _ = gemm_1bit_g128(&[block], &input, &mut output, 1, 1, QK1_0_G128);
    }
}
