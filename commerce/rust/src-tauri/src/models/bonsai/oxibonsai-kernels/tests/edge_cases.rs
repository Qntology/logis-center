//! Edge case tests for kernel operations.
//!
//! Tests boundary conditions for dequant, GEMV, and GEMM kernels.

use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::{dequant, gemm, gemv};

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

fn ref_dispatcher() -> KernelDispatcher {
    KernelDispatcher::with_tier(KernelTier::Reference)
}

// ──────────────────────────────────────────────────────────────
// Minimum k=128 (single block per row)
// ──────────────────────────────────────────────────────────────

#[test]
fn gemv_minimum_k_128_single_block() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("k=128 should work");
    // All +1 weights, all 1.0 input -> sum = 128
    assert!((output[0] - 128.0).abs() < 1.0);
}

#[test]
fn gemm_minimum_k_128_single_block() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, 128).expect("k=128 should work");
    assert!((output[0] - 128.0).abs() < 1.0);
}

// ──────────────────────────────────────────────────────────────
// Large k=12800 (100 blocks per row)
// ──────────────────────────────────────────────────────────────

#[test]
fn gemv_large_k_12800() {
    let k = 12800;
    let blocks_per_row = k / 128;
    let blocks: Vec<BlockQ1_0G128> = (0..blocks_per_row)
        .map(|_| make_block(1.0, [0xFF; 16]))
        .collect();
    let input = vec![1.0f32; k];
    let mut output = vec![0.0f32; 1];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, k).expect("large k should work");
    assert!(
        (output[0] - k as f32).abs() < 2.0,
        "expected ~{k}, got {}",
        output[0]
    );
}

#[test]
fn gemm_large_k_12800() {
    let k = 12800;
    let blocks_per_row = k / 128;
    let blocks: Vec<BlockQ1_0G128> = (0..blocks_per_row)
        .map(|_| make_block(1.0, [0xFF; 16]))
        .collect();
    let input = vec![1.0f32; k];
    let mut output = vec![0.0f32; 1];
    gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, k).expect("large k should work");
    assert!((output[0] - k as f32).abs() < 2.0);
}

// ──────────────────────────────────────────────────────────────
// n_rows boundary cases
// ──────────────────────────────────────────────────────────────

#[test]
fn gemv_single_row() {
    let blocks = vec![make_block(2.0, [0xFF; 16])];
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("single row");
    assert!((output[0] - 256.0).abs() < 2.0);
}

#[test]
fn gemv_many_rows() {
    let n_rows = 100;
    let blocks: Vec<BlockQ1_0G128> = (0..n_rows).map(|_| make_block(0.5, [0xFF; 16])).collect();
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; n_rows];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, n_rows, 128).expect("many rows");
    for (i, &v) in output.iter().enumerate() {
        assert!((v - 64.0).abs() < 1.0, "row {i}: expected ~64.0, got {v}");
    }
}

// ──────────────────────────────────────────────────────────────
// Zero-scale blocks
// ──────────────────────────────────────────────────────────────

#[test]
fn dequant_zero_scale_output_all_zeros() {
    let blocks = vec![make_block(0.0, [0xFF; 16])];
    let mut output = vec![999.0f32; 128];
    dequant::dequant_1bit_g128(&blocks, &mut output).expect("should succeed");
    for (i, &v) in output.iter().enumerate() {
        assert!(v.abs() < f32::EPSILON, "index {i}: expected 0, got {v}");
    }
}

#[test]
fn gemv_zero_scale_output_zeros() {
    let blocks = vec![make_block(0.0, [0xFF; 16])];
    let input = vec![100.0f32; 128];
    let mut output = vec![999.0f32; 1];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("should succeed");
    assert!(
        output[0].abs() < 0.01,
        "zero scale should give zero output, got {}",
        output[0]
    );
}

#[test]
fn gemm_zero_scale_output_zeros() {
    let blocks = vec![make_block(0.0, [0xFF; 16])];
    let input = vec![100.0f32; 128];
    let mut output = vec![999.0f32; 1];
    gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, 128).expect("should succeed");
    assert!(output[0].abs() < 0.01);
}

// ──────────────────────────────────────────────────────────────
// Bit patterns
// ──────────────────────────────────────────────────────────────

#[test]
fn dequant_all_ones_bits() {
    let blocks = vec![make_block(3.0, [0xFF; 16])];
    let mut output = vec![0.0f32; 128];
    dequant::dequant_1bit_g128(&blocks, &mut output).expect("should succeed");
    for &v in &output {
        assert!((v - 3.0).abs() < 0.01, "all-ones should give +scale");
    }
}

#[test]
fn dequant_all_zeros_bits() {
    let blocks = vec![make_block(3.0, [0x00; 16])];
    let mut output = vec![0.0f32; 128];
    dequant::dequant_1bit_g128(&blocks, &mut output).expect("should succeed");
    for &v in &output {
        assert!((v + 3.0).abs() < 0.01, "all-zeros should give -scale");
    }
}

#[test]
fn dequant_alternating_0xaa_pattern() {
    // 0xAA = 10101010: bits 1,3,5,7 set; 0,2,4,6 clear
    let blocks = vec![make_block(1.0, [0xAA; 16])];
    let mut output = vec![0.0f32; 128];
    dequant::dequant_1bit_g128(&blocks, &mut output).expect("should succeed");
    for (i, &val) in output.iter().enumerate().take(128) {
        let expected = if i % 2 == 0 { -1.0 } else { 1.0 };
        assert!(
            (val - expected).abs() < 0.01,
            "at {i}: expected {expected}, got {val}",
        );
    }
}

#[test]
fn dequant_alternating_0x55_pattern() {
    // 0x55 = 01010101: bits 0,2,4,6 set; 1,3,5,7 clear
    let blocks = vec![make_block(1.0, [0x55; 16])];
    let mut output = vec![0.0f32; 128];
    dequant::dequant_1bit_g128(&blocks, &mut output).expect("should succeed");
    for (i, &val) in output.iter().enumerate().take(128) {
        let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (val - expected).abs() < 0.01,
            "at {i}: expected {expected}, got {val}",
        );
    }
}

// ──────────────────────────────────────────────────────────────
// GEMV with all-zero input
// ──────────────────────────────────────────────────────────────

#[test]
fn gemv_zero_input_produces_zero_output() {
    let blocks = vec![make_block(5.0, [0xFF; 16])];
    let input = vec![0.0f32; 128];
    let mut output = vec![999.0f32; 1];
    gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("should succeed");
    assert!(
        output[0].abs() < f32::EPSILON,
        "zero input should produce zero output, got {}",
        output[0]
    );
}

// ──────────────────────────────────────────────────────────────
// GEMM with m=1 (degenerates to GEMV-like)
// ──────────────────────────────────────────────────────────────

#[test]
fn gemm_m1_matches_gemv() {
    let blocks = vec![make_block(1.0, [0xFF; 16]), make_block(1.0, [0x00; 16])];
    let input = vec![1.0f32; 128];

    let mut gemv_output = vec![0.0f32; 2];
    gemv::gemv_1bit_g128(&blocks, &input, &mut gemv_output, 2, 128).expect("gemv");

    let mut gemm_output = vec![0.0f32; 2];
    gemm::gemm_1bit_g128(&blocks, &input, &mut gemm_output, 1, 2, 128).expect("gemm m=1");

    for i in 0..2 {
        assert!(
            (gemv_output[i] - gemm_output[i]).abs() < 0.01,
            "gemv[{i}]={} != gemm[{i}]={}",
            gemv_output[i],
            gemm_output[i]
        );
    }
}

// ──────────────────────────────────────────────────────────────
// GEMM with large m
// ──────────────────────────────────────────────────────────────

#[test]
fn gemm_large_m_16() {
    let m = 16;
    let n_rows = 2;
    let k = 128;
    let blocks = vec![
        make_block(1.0, [0xFF; 16]), // row 0: all +1
        make_block(1.0, [0x00; 16]), // row 1: all -1
    ];
    // Input: each of 16 rows has all 1.0
    let input = vec![1.0f32; m * k];
    let mut output = vec![0.0f32; m * n_rows];

    gemm::gemm_1bit_g128(&blocks, &input, &mut output, m, n_rows, k).expect("large m should work");

    for mi in 0..m {
        let o0 = output[mi * n_rows];
        let o1 = output[mi * n_rows + 1];
        assert!(
            (o0 - 128.0).abs() < 1.0,
            "row {mi}, col 0: expected ~128, got {o0}"
        );
        assert!(
            (o1 + 128.0).abs() < 1.0,
            "row {mi}, col 1: expected ~-128, got {o1}"
        );
    }
}

// ──────────────────────────────────────────────────────────────
// Dispatcher-based tests (ensure dispatch path works)
// ──────────────────────────────────────────────────────────────

#[test]
fn dispatcher_dequant_correct() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(2.5, [0xFF; 16])];
    let mut output = vec![0.0f32; 128];
    dispatcher
        .dequant(&blocks, &mut output)
        .expect("dispatch dequant");
    for &v in &output {
        assert!((v - 2.5).abs() < 0.01);
    }
}

#[test]
fn dispatcher_gemv_correct() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![2.0f32; 128];
    let mut output = vec![0.0f32; 1];
    dispatcher
        .gemv(&blocks, &input, &mut output, 1, 128)
        .expect("dispatch gemv");
    assert!((output[0] - 256.0).abs() < 2.0);
}

#[test]
fn dispatcher_gemm_correct() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![2.0f32; 128];
    let mut output = vec![0.0f32; 1];
    dispatcher
        .gemm(&blocks, &input, &mut output, 1, 1, 128)
        .expect("dispatch gemm");
    assert!((output[0] - 256.0).abs() < 2.0);
}

#[test]
fn dispatcher_name_is_nonempty() {
    let dispatcher = ref_dispatcher();
    let name = dispatcher.name();
    assert!(!name.is_empty());
    assert!(name.contains("reference"));
}

#[test]
fn auto_detect_dispatcher_works() {
    let dispatcher = KernelDispatcher::auto_detect();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let mut output = vec![0.0f32; 128];
    dispatcher
        .dequant(&blocks, &mut output)
        .expect("auto-detect dequant");
    for &v in &output {
        assert!((v - 1.0).abs() < 0.01);
    }
}
