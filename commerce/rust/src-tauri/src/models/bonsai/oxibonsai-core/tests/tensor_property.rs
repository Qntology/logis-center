//! Property-based tests for Q1_0_g128 tensor operations.
//!
//! Uses proptest to verify invariants of the 1-bit block format.

use half::f16;
use proptest::prelude::*;

use oxibonsai_core::tensor::{BlockQ1_0G128, OneBitTensor, BLOCK_SIZE_BYTES, QK1_0_G128};

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

fn block_to_bytes(block: &BlockQ1_0G128) -> Vec<u8> {
    let ptr = block as *const BlockQ1_0G128 as *const u8;
    // SAFETY: BlockQ1_0G128 is repr(C) with known size
    unsafe { std::slice::from_raw_parts(ptr, BLOCK_SIZE_BYTES).to_vec() }
}

// ──────────────────────────────────────────────────────────────
// Deterministic tests
// ──────────────────────────────────────────────────────────────

#[test]
fn block_size_is_always_18_bytes() {
    assert_eq!(std::mem::size_of::<BlockQ1_0G128>(), 18);
    assert_eq!(BLOCK_SIZE_BYTES, 18);
}

#[test]
fn qk_constant_is_128() {
    assert_eq!(QK1_0_G128, 128);
}

#[test]
fn zero_scale_produces_all_zero_output() {
    let block = make_block(0.0, [0xFF; 16]);
    for i in 0..128 {
        assert!(
            block.weight(i).abs() < f32::EPSILON,
            "zero scale should give zero weight at index {i}"
        );
    }
}

#[test]
fn zero_scale_negative_bits_also_zero() {
    let block = make_block(0.0, [0x00; 16]);
    for i in 0..128 {
        // -0.0 == 0.0 in float comparison
        assert!(
            block.weight(i).abs() < f32::EPSILON,
            "zero scale should give zero weight at index {i}"
        );
    }
}

#[test]
fn from_bytes_roundtrip_preserves_fields() {
    let block = make_block(1.5, [0xAB; 16]);
    let bytes = block_to_bytes(&block);
    let parsed = BlockQ1_0G128::from_bytes(&bytes).expect("should parse valid bytes");
    assert_eq!(parsed.d, block.d);
    assert_eq!(parsed.qs, block.qs);
}

#[test]
fn from_bytes_too_small_returns_error() {
    let bytes = [0u8; 17]; // 17 < 18
    let result = BlockQ1_0G128::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn slice_from_bytes_two_blocks() {
    let b1 = make_block(1.0, [0xFF; 16]);
    let b2 = make_block(2.0, [0x00; 16]);
    let mut bytes = block_to_bytes(&b1);
    bytes.extend_from_slice(&block_to_bytes(&b2));
    let blocks = BlockQ1_0G128::slice_from_bytes(&bytes).expect("should parse 2 blocks");
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].d.to_f32(), 1.0);
    assert_eq!(blocks[1].d.to_f32(), 2.0);
}

#[test]
fn slice_from_bytes_not_aligned_returns_error() {
    let bytes = [0u8; 19]; // 19 % 18 != 0
    let result = BlockQ1_0G128::slice_from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn one_bit_tensor_dequantize_multiple_blocks() {
    let b1 = make_block(1.0, [0xFF; 16]);
    let b2 = make_block(2.0, [0x00; 16]);
    let mut bytes = block_to_bytes(&b1);
    bytes.extend_from_slice(&block_to_bytes(&b2));

    let tensor = OneBitTensor::from_raw("test".to_string(), vec![256], &bytes)
        .expect("should create tensor");
    assert_eq!(tensor.num_blocks(), 2);
    assert_eq!(tensor.element_count(), 256);

    let values = tensor.dequantize_all();
    assert_eq!(values.len(), 256);
    // First 128: all +1.0
    for &v in &values[..128] {
        assert!((v - 1.0).abs() < 0.01);
    }
    // Next 128: all -2.0
    for &v in &values[128..] {
        assert!((v + 2.0).abs() < 0.01);
    }
}

#[test]
fn maximum_f16_scale() {
    let max_f16 = f16::MAX;
    let block = BlockQ1_0G128 {
        d: max_f16,
        qs: [0xFF; 16],
    };
    let scale = max_f16.to_f32();
    assert!(scale.is_finite(), "f16::MAX should convert to finite f32");
    for i in 0..128 {
        let w = block.weight(i);
        assert!(w.is_finite(), "weight at {i} should be finite");
        assert!((w - scale).abs() < 1.0, "weight should be close to scale");
    }
}

#[test]
fn minimum_positive_f16_scale() {
    let min_f16 = f16::MIN_POSITIVE;
    let block = BlockQ1_0G128 {
        d: min_f16,
        qs: [0xFF; 16],
    };
    let scale = min_f16.to_f32();
    assert!(scale > 0.0);
    for i in 0..128 {
        let w = block.weight(i);
        assert!(
            w > 0.0,
            "positive bit with positive scale should be positive"
        );
    }
}

// ──────────────────────────────────────────────────────────────
// Property-based tests
// ──────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn sign_bit_and_weight_are_consistent(
        scale in -100.0f32..100.0f32,
        bits in prop::array::uniform16(any::<u8>()),
    ) {
        let block = make_block(scale, bits);
        let d = f16::from_f32(scale).to_f32();

        for i in 0..128 {
            let w = block.weight(i);
            let sign = block.sign_bit(i);

            if d.abs() < f32::EPSILON {
                // Zero scale: weight should be ~0 regardless of sign bit
                prop_assert!(w.abs() < 0.01, "zero scale but weight={w} at {i}");
            } else if sign {
                // Positive sign: weight should be +d
                prop_assert!(
                    (w - d).abs() < 0.01,
                    "sign=true but weight={w} != d={d} at {i}"
                );
            } else {
                // Negative sign: weight should be -d
                prop_assert!(
                    (w + d).abs() < 0.01,
                    "sign=false but weight={w} != -d={} at {i}",
                    -d
                );
            }
        }
    }

    #[test]
    fn dequantized_values_are_plus_minus_scale(
        scale in 0.001f32..50.0f32,
        bits in prop::array::uniform16(any::<u8>()),
    ) {
        let block = make_block(scale, bits);
        let d = f16::from_f32(scale).to_f32();

        for i in 0..128 {
            let w = block.weight(i);
            // Weight should be either +d or -d
            let is_plus = (w - d).abs() < 0.01;
            let is_minus = (w + d).abs() < 0.01;
            prop_assert!(
                is_plus || is_minus,
                "weight={w} should be +/-{d} at index {i}"
            );
        }
    }

    #[test]
    fn from_bytes_roundtrip_property(
        scale in -50.0f32..50.0f32,
        bits in prop::array::uniform16(any::<u8>()),
    ) {
        let block = make_block(scale, bits);
        let bytes = block_to_bytes(&block);
        let parsed = BlockQ1_0G128::from_bytes(&bytes).expect("should parse");
        prop_assert_eq!(parsed.d, block.d);
        prop_assert_eq!(parsed.qs, block.qs);
    }

    #[test]
    fn dequantize_output_length_equals_blocks_times_128(
        num_blocks in 1usize..10,
        scale in 0.1f32..10.0f32,
    ) {
        let block = make_block(scale, [0xAA; 16]);
        let mut bytes = Vec::with_capacity(num_blocks * BLOCK_SIZE_BYTES);
        for _ in 0..num_blocks {
            bytes.extend_from_slice(&block_to_bytes(&block));
        }

        let tensor = OneBitTensor::from_raw(
            "prop_test".to_string(),
            vec![(num_blocks * 128) as u64],
            &bytes,
        ).expect("should create tensor");

        let values = tensor.dequantize_all();
        prop_assert_eq!(values.len(), num_blocks * 128);
    }

    #[test]
    fn sign_bit_matches_raw_bit_extraction(
        bits in prop::array::uniform16(any::<u8>()),
        idx in 0usize..128,
    ) {
        let block = make_block(1.0, bits);
        let byte_index = idx / 8;
        let bit_offset = idx % 8;
        let expected = (bits[byte_index] >> bit_offset) & 1 != 0;
        prop_assert_eq!(block.sign_bit(idx), expected);
    }
}
