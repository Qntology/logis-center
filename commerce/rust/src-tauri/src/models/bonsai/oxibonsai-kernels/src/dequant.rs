//! Reference (naive) dequantization kernel for Q1\_0\_g128.
//!
//! This is the correctness reference implementation — pure scalar Rust,
//! no SIMD, no unsafe.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::error::{KernelError, KernelResult};

/// Dequantize Q1\_0\_g128 blocks to FP32.
///
/// For each block: `output[i] = bit[i] ? +d : -d`
///
/// This is the reference implementation for correctness verification.
pub fn dequant_1bit_g128(blocks: &[BlockQ1_0G128], output: &mut [f32]) -> KernelResult<()> {
    let expected_len = blocks.len() * QK1_0_G128;
    if output.len() < expected_len {
        return Err(KernelError::BufferTooSmall {
            needed: expected_len,
            available: output.len(),
        });
    }

    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let base = i * QK1_0_G128;

        for j in 0..QK1_0_G128 {
            let byte_index = j / 8;
            let bit_offset = j % 8;
            let bit = (block.qs[byte_index] >> bit_offset) & 1;
            output[base + j] = if bit != 0 { d } else { -d };
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn dequant_all_positive() {
        let block = make_block(2.0, [0xFF; 16]);
        let mut output = vec![0.0f32; 128];
        dequant_1bit_g128(&[block], &mut output).expect("dequant should succeed");
        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn dequant_all_negative() {
        let block = make_block(3.0, [0x00; 16]);
        let mut output = vec![0.0f32; 128];
        dequant_1bit_g128(&[block], &mut output).expect("dequant should succeed");
        for &v in &output {
            assert!((v + 3.0).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn dequant_alternating() {
        let block = make_block(1.0, [0xAA; 16]); // 10101010
        let mut output = vec![0.0f32; 128];
        dequant_1bit_g128(&[block], &mut output).expect("dequant should succeed");
        for (i, &val) in output.iter().enumerate().take(128) {
            let expected = if i % 2 == 0 { -1.0 } else { 1.0 };
            assert!(
                (val - expected).abs() < 0.01,
                "at {i}: expected {expected}, got {val}",
            );
        }
    }

    #[test]
    fn dequant_multiple_blocks() {
        let blocks = vec![make_block(1.0, [0xFF; 16]), make_block(2.0, [0x00; 16])];
        let mut output = vec![0.0f32; 256];
        dequant_1bit_g128(&blocks, &mut output).expect("dequant should succeed");

        // First 128: all +1.0
        for &v in &output[..128] {
            assert!((v - 1.0).abs() < 0.01);
        }
        // Next 128: all -2.0
        for &v in &output[128..] {
            assert!((v + 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn dequant_buffer_too_small() {
        let block = make_block(1.0, [0xFF; 16]);
        let mut output = vec![0.0f32; 64]; // too small
        let result = dequant_1bit_g128(&[block], &mut output);
        assert!(result.is_err());
    }
}
