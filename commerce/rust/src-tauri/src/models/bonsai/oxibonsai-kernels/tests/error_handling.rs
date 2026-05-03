//! Tests that every KernelError variant can be triggered through
//! both the reference kernels and the dispatcher.

use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::error::KernelError;
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
// Dequant errors
// ──────────────────────────────────────────────────────────────

#[test]
fn dequant_buffer_too_small_reference() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let mut output = vec![0.0f32; 64]; // need 128
    let result = dequant::dequant_1bit_g128(&blocks, &mut output);
    match result.expect_err("should fail") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 128);
            assert_eq!(available, 64);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

#[test]
fn dequant_buffer_too_small_dispatcher() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let mut output = vec![0.0f32; 10]; // need 128
    let result = dispatcher.dequant(&blocks, &mut output);
    match result.expect_err("should fail") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 128);
            assert_eq!(available, 10);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

#[test]
fn dequant_empty_blocks_succeeds() {
    let blocks: Vec<BlockQ1_0G128> = vec![];
    let mut output: Vec<f32> = vec![];
    let result = dequant::dequant_1bit_g128(&blocks, &mut output);
    assert!(result.is_ok(), "empty blocks should succeed");
}

// ──────────────────────────────────────────────────────────────
// GEMV errors
// ──────────────────────────────────────────────────────────────

#[test]
fn gemv_not_block_aligned_reference() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 1];
    let result = gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 100);
    match result.expect_err("k=100 not aligned") {
        KernelError::NotBlockAligned { count, block_size } => {
            assert_eq!(count, 100);
            assert_eq!(block_size, 128);
        }
        other => panic!("expected NotBlockAligned, got: {other}"),
    }
}

#[test]
fn gemv_not_block_aligned_dispatcher() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 1];
    let result = dispatcher.gemv(&blocks, &input, &mut output, 1, 100);
    assert!(result.is_err());
}

#[test]
fn gemv_dimension_mismatch_input_too_small() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 64]; // need 128
    let mut output = vec![0.0f32; 1];
    let result = gemv::gemv_1bit_g128(&blocks, &input, &mut output, 1, 128);
    match result.expect_err("input too small") {
        KernelError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 128);
            assert_eq!(got, 64);
        }
        other => panic!("expected DimensionMismatch, got: {other}"),
    }
}

#[test]
fn gemv_dimension_mismatch_dispatcher() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];
    let result = dispatcher.gemv(&blocks, &input, &mut output, 1, 128);
    assert!(result.is_err());
}

#[test]
fn gemv_output_buffer_too_small() {
    let blocks = vec![make_block(1.0, [0xFF; 16]); 2]; // 2 rows
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1]; // need 2
    let result = gemv::gemv_1bit_g128(&blocks, &input, &mut output, 2, 128);
    match result.expect_err("output too small") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 2);
            assert_eq!(available, 1);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

#[test]
fn gemv_blocks_too_few_for_rows() {
    let blocks = vec![make_block(1.0, [0xFF; 16])]; // only 1 block
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 3]; // requesting 3 rows
    let result = gemv::gemv_1bit_g128(&blocks, &input, &mut output, 3, 128);
    match result.expect_err("not enough blocks") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 3); // 3 rows * 1 block_per_row
            assert_eq!(available, 1);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

// ──────────────────────────────────────────────────────────────
// GEMM errors
// ──────────────────────────────────────────────────────────────

#[test]
fn gemm_not_block_aligned() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 1];
    let result = gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, 100);
    match result.expect_err("k=100 not aligned") {
        KernelError::NotBlockAligned { count, block_size } => {
            assert_eq!(count, 100);
            assert_eq!(block_size, 128);
        }
        other => panic!("expected NotBlockAligned, got: {other}"),
    }
}

#[test]
fn gemm_output_buffer_too_small() {
    let blocks = vec![make_block(1.0, [0xFF; 16]); 2]; // 2 rows
    let input = vec![1.0f32; 256]; // m=2, k=128
    let mut output = vec![0.0f32; 2]; // need m*n_rows = 2*2 = 4
    let result = gemm::gemm_1bit_g128(&blocks, &input, &mut output, 2, 2, 128);
    match result.expect_err("output too small") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 4); // 2*2
            assert_eq!(available, 2);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

#[test]
fn gemm_input_buffer_too_small() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 64]; // need m*k = 1*128 = 128
    let mut output = vec![0.0f32; 1];
    let result = gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, 128);
    match result.expect_err("input too small") {
        KernelError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 128);
            assert_eq!(got, 64);
        }
        other => panic!("expected DimensionMismatch, got: {other}"),
    }
}

#[test]
fn gemm_input_too_small_dispatcher() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];
    let result = dispatcher.gemm(&blocks, &input, &mut output, 1, 1, 128);
    assert!(result.is_err());
}

#[test]
fn gemm_output_too_small_dispatcher() {
    let dispatcher = ref_dispatcher();
    let blocks = vec![make_block(1.0, [0xFF; 16]); 2];
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2]; // need 4
    let result = dispatcher.gemm(&blocks, &input, &mut output, 2, 2, 128);
    assert!(result.is_err());
}

#[test]
fn gemm_blocks_too_few() {
    let blocks = vec![make_block(1.0, [0xFF; 16])]; // 1 block, need 4
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 4];
    let result = gemm::gemm_1bit_g128(&blocks, &input, &mut output, 1, 4, 128);
    match result.expect_err("not enough blocks") {
        KernelError::BufferTooSmall { needed, available } => {
            assert_eq!(needed, 4);
            assert_eq!(available, 1);
        }
        other => panic!("expected BufferTooSmall, got: {other}"),
    }
}

// ──────────────────────────────────────────────────────────────
// Error display
// ──────────────────────────────────────────────────────────────

#[test]
fn error_display_messages_are_descriptive() {
    let e1 = KernelError::DimensionMismatch {
        expected: 128,
        got: 64,
    };
    let msg = format!("{e1}");
    assert!(msg.contains("128"), "should mention expected");
    assert!(msg.contains("64"), "should mention got");

    let e2 = KernelError::BufferTooSmall {
        needed: 256,
        available: 100,
    };
    let msg2 = format!("{e2}");
    assert!(msg2.contains("256"));
    assert!(msg2.contains("100"));

    let e3 = KernelError::NotBlockAligned {
        count: 100,
        block_size: 128,
    };
    let msg3 = format!("{e3}");
    assert!(msg3.contains("100"));
    assert!(msg3.contains("128"));
}
