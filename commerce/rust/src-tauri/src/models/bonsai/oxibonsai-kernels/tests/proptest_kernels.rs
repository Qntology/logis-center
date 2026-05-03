//! Property-based tests for Q1_0_g128 kernel correctness.

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_kernels::dequant::dequant_1bit_g128;
use oxibonsai_kernels::gemm::gemm_1bit_g128;
use oxibonsai_kernels::gemv::gemv_1bit_g128;
use proptest::prelude::*;

/// Strategy to generate a single BlockQ1_0G128 with a finite, non-zero scale.
fn arb_block() -> impl Strategy<Value = BlockQ1_0G128> {
    (
        prop::num::f32::NORMAL.prop_filter("non-zero finite scale", |v| {
            v.is_finite() && v.abs() > 1e-6 && v.abs() < 100.0
        }),
        prop::array::uniform16(any::<u8>()),
    )
        .prop_map(|(scale, qs)| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs,
        })
}

/// Strategy for a vector of blocks (1..=4 blocks).
fn arb_blocks(count: usize) -> impl Strategy<Value = Vec<BlockQ1_0G128>> {
    prop::collection::vec(arb_block(), count..=count)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every dequantized value is either +d or -d (for finite, non-zero scales).
    #[test]
    fn dequant_outputs_are_pm_scale(block in arb_block()) {
        let d = block.d.to_f32();
        if !d.is_finite() || d.abs() < 1e-6 {
            return Ok(());
        }
        let mut output = vec![0.0f32; QK1_0_G128];
        dequant_1bit_g128(&[block], &mut output)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        for (i, &v) in output.iter().enumerate() {
            let abs_diff_pos = (v - d).abs();
            let abs_diff_neg = (v + d).abs();
            let tol = d.abs() * 0.02; // f16 rounding tolerance
            prop_assert!(
                abs_diff_pos < tol || abs_diff_neg < tol,
                "output[{i}]={v} is not close to +d={d} or -d={neg_d}",
                neg_d = -d,
            );
        }
    }

    /// gemv(A, alpha*x) ~= alpha * gemv(A, x) (linearity in input).
    #[test]
    fn gemv_linearity(
        blocks in arb_blocks(1),
        alpha in prop::num::f32::NORMAL.prop_filter("finite nonzero", |v| v.is_finite() && v.abs() > 0.01 && v.abs() < 10.0),
    ) {
        let k = QK1_0_G128;
        let n_rows = 1;
        // Deterministic input
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();
        let scaled_input: Vec<f32> = input.iter().map(|&v| v * alpha).collect();

        let mut out_base = vec![0.0f32; n_rows];
        let mut out_scaled = vec![0.0f32; n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_base, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        gemv_1bit_g128(&blocks, &scaled_input, &mut out_scaled, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;

        let expected = out_base[0] * alpha;
        let actual = out_scaled[0];
        let tol = expected.abs() * 0.05 + 1.0; // tolerance for f16 scale rounding
        prop_assert!(
            (expected - actual).abs() < tol,
            "linearity: alpha*gemv(x)={expected} vs gemv(alpha*x)={actual}, alpha={alpha}"
        );
    }

    /// gemm with m=1 matches gemv output.
    #[test]
    fn gemm_m1_equals_gemv(blocks in arb_blocks(2)) {
        let k = QK1_0_G128;
        let n_rows = 2;
        let m = 1;
        // Rebuild blocks for 2 rows, 1 block per row
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();

        let mut out_gemv = vec![0.0f32; n_rows];
        let mut out_gemm = vec![0.0f32; m * n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_gemv, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        gemm_1bit_g128(&blocks, &input, &mut out_gemm, m, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;

        for i in 0..n_rows {
            let diff = (out_gemv[i] - out_gemm[i]).abs();
            prop_assert!(
                diff < 0.01,
                "gemm(m=1) vs gemv mismatch at row {i}: gemv={}, gemm={}",
                out_gemv[i],
                out_gemm[i],
            );
        }
    }

    /// NEON dequant output matches reference exactly.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_reference_dequant(block in arb_block()) {
        let mut out_ref = vec![0.0f32; QK1_0_G128];
        let mut out_neon = vec![0.0f32; QK1_0_G128];

        dequant_1bit_g128(&[block], &mut out_ref)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::dequant_1bit_g128_neon(&[block], &mut out_neon)
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..QK1_0_G128 {
            let diff = (out_ref[i] - out_neon[i]).abs();
            prop_assert!(
                diff < 0.01,
                "dequant mismatch at {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i],
            );
        }
    }

    /// NEON gemv matches reference within tolerance.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_reference_gemv(blocks in arb_blocks(1)) {
        let k = QK1_0_G128;
        let n_rows = 1;
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon = vec![0.0f32; n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::gemv_1bit_g128_neon(
                &blocks, &input, &mut out_neon, n_rows, k,
            )
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..n_rows {
            let diff = (out_ref[i] - out_neon[i]).abs();
            prop_assert!(
                diff < 0.5,
                "gemv mismatch at row {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i],
            );
        }
    }
}
