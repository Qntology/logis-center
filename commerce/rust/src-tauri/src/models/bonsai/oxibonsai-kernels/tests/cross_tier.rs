//! Cross-tier correctness tests.
//!
//! Verifies that all available kernel tiers produce identical (within tolerance)
//! output for the same inputs, ensuring SIMD kernels match the reference implementation.

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_kernels::{KernelDispatcher, OneBitKernel};

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

fn make_deterministic_blocks(n_blocks: usize) -> Vec<BlockQ1_0G128> {
    (0..n_blocks)
        .map(|i| {
            let scale = 0.3 + (i as f32) * 0.02;
            let bits: [u8; 16] = core::array::from_fn(|j| ((i * 37 + j * 13 + 7) & 0xFF) as u8);
            make_block(scale, bits)
        })
        .collect()
}

#[test]
fn test_cross_tier_dequant_large() {
    let n_blocks = 32; // 4096 elements
    let n_elements = n_blocks * QK1_0_G128;
    let blocks = make_deterministic_blocks(n_blocks);

    // Reference output
    let ref_dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut ref_output = vec![0.0f32; n_elements];
    ref_dispatcher
        .dequant(&blocks, &mut ref_output)
        .expect("reference dequant should succeed");

    // Auto-detected tier output
    let auto_dispatcher = KernelDispatcher::auto_detect();
    let mut auto_output = vec![0.0f32; n_elements];
    auto_dispatcher
        .dequant(&blocks, &mut auto_output)
        .expect("auto dequant should succeed");

    for i in 0..n_elements {
        assert!(
            (ref_output[i] - auto_output[i]).abs() < 0.01,
            "dequant mismatch at element {i}: ref={}, auto({})={}",
            ref_output[i],
            auto_dispatcher.tier(),
            auto_output[i]
        );
    }
}

#[test]
fn test_cross_tier_gemv_large() {
    let n_rows = 128;
    let k = 1024;
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;
    let blocks = make_deterministic_blocks(total_blocks);
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.007) - 3.5).collect();

    // Reference
    let ref_dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut ref_output = vec![0.0f32; n_rows];
    ref_dispatcher
        .gemv(&blocks, &input, &mut ref_output, n_rows, k)
        .expect("reference gemv should succeed");

    // Auto-detected
    let auto_dispatcher = KernelDispatcher::auto_detect();
    let mut auto_output = vec![0.0f32; n_rows];
    auto_dispatcher
        .gemv(&blocks, &input, &mut auto_output, n_rows, k)
        .expect("auto gemv should succeed");

    for i in 0..n_rows {
        assert!(
            (ref_output[i] - auto_output[i]).abs() < 1.0,
            "gemv mismatch at row {i}: ref={}, auto({})={}",
            ref_output[i],
            auto_dispatcher.tier(),
            auto_output[i]
        );
    }
}

#[test]
fn test_cross_tier_gemm_batch() {
    let m = 4;
    let n_rows = 64;
    let k = 512;
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;
    let blocks = make_deterministic_blocks(total_blocks);
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.003) - 0.6).collect();

    // Reference
    let ref_dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut ref_output = vec![0.0f32; m * n_rows];
    ref_dispatcher
        .gemm(&blocks, &input, &mut ref_output, m, n_rows, k)
        .expect("reference gemm should succeed");

    // Auto-detected
    let auto_dispatcher = KernelDispatcher::auto_detect();
    let mut auto_output = vec![0.0f32; m * n_rows];
    auto_dispatcher
        .gemm(&blocks, &input, &mut auto_output, m, n_rows, k)
        .expect("auto gemm should succeed");

    for i in 0..(m * n_rows) {
        assert!(
            (ref_output[i] - auto_output[i]).abs() < 1.0,
            "gemm mismatch at idx {i}: ref={}, auto({})={}",
            ref_output[i],
            auto_dispatcher.tier(),
            auto_output[i]
        );
    }
}
