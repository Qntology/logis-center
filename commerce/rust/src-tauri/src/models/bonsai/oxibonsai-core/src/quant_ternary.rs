//! Ternary quantization block types for TQ2_0_g128 and TQ2_0 formats.
//!
//! Two ternary formats: `BlockTQ2_0_g128` (128 weights, 34 bytes, PrismML)
//! and `BlockTQ2_0` (256 weights, 66 bytes, llama.cpp compat).
//! Both use 2-bit coding: `00→-1`, `01→0`, `10→+1`, 4 weights per byte LSB-first.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per TQ2_0_g128 block.
pub const QK_TQ2_0_G128: usize = 128;

/// Number of weights per TQ2_0 block.
pub const QK_TQ2_0: usize = 256;

/// Number of bytes per TQ2_0_g128 block.
pub const BLOCK_TQ2_0_G128_BYTES: usize = 34;

/// Number of bytes per TQ2_0 block.
pub const BLOCK_TQ2_0_BYTES: usize = 66;

// ---------------------------------------------------------------------------
// TernaryCode
// ---------------------------------------------------------------------------

/// Ternary weight code for 2-bit encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TernaryCode {
    /// Negative weight (-1): bit pattern `0b00`.
    Neg = 0b00,
    /// Zero weight (0): bit pattern `0b01`.
    Zero = 0b01,
    /// Positive weight (+1): bit pattern `0b10`.
    Pos = 0b10,
}

impl TernaryCode {
    /// Convert to integer representation: Neg→-1, Zero→0, Pos→+1.
    pub fn to_i8(self) -> i8 {
        match self {
            Self::Neg => -1,
            Self::Zero => 0,
            Self::Pos => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockTQ2_0_g128
// ---------------------------------------------------------------------------

/// TQ2_0_g128 block: 128 weights at 2 bits each, PrismML format.
///
/// Layout (34 bytes): `qs[32]` packed codes + `d` FP16 scale.
/// Bit coding: `00→-1`, `01→0`, `10→+1`, 4 weights per byte LSB-first.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockTQ2_0_g128 {
    /// 128 × 2-bit quantized weights, 4 per byte, LSB-first.
    pub qs: [u8; 32],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockTQ2_0_g128>() == BLOCK_TQ2_0_G128_BYTES);

impl BlockTQ2_0_g128 {
    /// Dequantize a slice of TQ2_0_g128 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * 128`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_TQ2_0_G128;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_TQ2_0_G128;
            for j in 0..QK_TQ2_0_G128 {
                let byte_idx = j / 4;
                let lane = j % 4;
                let code_val = Self::ternary_decode(block.qs[byte_idx], lane);
                output[base + j] = d * (code_val as f32);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into TQ2_0_g128 blocks.
    ///
    /// Input length must be a multiple of 128.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_TQ2_0_G128 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_TQ2_0_G128
                ),
            });
        }
        let num_blocks = input.len() / QK_TQ2_0_G128;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_TQ2_0_G128;
            let chunk = &input[base..base + QK_TQ2_0_G128];

            let absmax = chunk
                .iter()
                .copied()
                .fold(0.0f32, |acc, x| acc.max(x.abs()));

            let mut qs = [0u8; 32];

            if absmax == 0.0 {
                // All zero: code = 0b01 (Zero), qs bytes = 0b01_01_01_01 = 0x55
                for b in qs.iter_mut() {
                    *b = 0x55;
                }
                blocks.push(BlockTQ2_0_g128 { qs, d: f16::ZERO });
                continue;
            }

            let threshold = 0.5 * absmax;
            for (j, &x) in chunk.iter().enumerate() {
                let code: u8 = if x >= threshold {
                    TernaryCode::Pos as u8 // 0b10
                } else if x <= -threshold {
                    TernaryCode::Neg as u8 // 0b00
                } else {
                    TernaryCode::Zero as u8 // 0b01
                };
                let byte_idx = j / 4;
                let shift = (j % 4) * 2;
                qs[byte_idx] |= code << shift;
            }

            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(absmax),
            });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of TQ2_0_g128 blocks.
    ///
    /// Returns error if length is not a multiple of 34 or pointer is misaligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_TQ2_0_G128_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_TQ2_0_G128_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 slice_from_bytes: pointer not {}-byte aligned",
                    align
                ),
            });
        }
        let count = data.len() / BLOCK_TQ2_0_G128_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Decode a 2-bit code at `lane` (0..4) from `byte`, returning the weight as i8.
    ///
    /// Code map: `0b00→-1`, `0b01→0`, `0b10→+1`, `0b11→0` (reserved treated as zero).
    pub fn ternary_decode(byte: u8, lane: usize) -> i8 {
        let shift = lane * 2;
        let code = (byte >> shift) & 0x03;
        match code {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0, // 0b11 reserved → zero
        }
    }
}

// ---------------------------------------------------------------------------
// BlockTQ2_0
// ---------------------------------------------------------------------------

/// TQ2_0 block: 256 weights at 2 bits each, llama.cpp compat format.
///
/// Layout (66 bytes): `qs[64]` packed codes + `d` FP16 scale.
/// Same 2-bit coding as TQ2_0_g128.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockTQ2_0 {
    /// 256 × 2-bit quantized weights, 4 per byte, LSB-first.
    pub qs: [u8; 64],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockTQ2_0>() == BLOCK_TQ2_0_BYTES);

impl BlockTQ2_0 {
    /// Dequantize a slice of TQ2_0 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * 256`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_TQ2_0;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_TQ2_0;
            for j in 0..QK_TQ2_0 {
                let byte_idx = j / 4;
                let lane = j % 4;
                let code_val = ternary_decode_g256(block.qs[byte_idx], lane);
                output[base + j] = d * (code_val as f32);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into TQ2_0 blocks.
    ///
    /// Input length must be a multiple of 256.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_TQ2_0 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_TQ2_0
                ),
            });
        }
        let num_blocks = input.len() / QK_TQ2_0;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_TQ2_0;
            let chunk = &input[base..base + QK_TQ2_0];

            let absmax = chunk
                .iter()
                .copied()
                .fold(0.0f32, |acc, x| acc.max(x.abs()));

            let mut qs = [0u8; 64];

            if absmax == 0.0 {
                for b in qs.iter_mut() {
                    *b = 0x55;
                }
                blocks.push(BlockTQ2_0 { qs, d: f16::ZERO });
                continue;
            }

            let threshold = 0.5 * absmax;
            for (j, &x) in chunk.iter().enumerate() {
                let code: u8 = if x >= threshold {
                    TernaryCode::Pos as u8
                } else if x <= -threshold {
                    TernaryCode::Neg as u8
                } else {
                    TernaryCode::Zero as u8
                };
                let byte_idx = j / 4;
                let shift = (j % 4) * 2;
                qs[byte_idx] |= code << shift;
            }

            blocks.push(BlockTQ2_0 {
                qs,
                d: f16::from_f32(absmax),
            });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of TQ2_0 blocks.
    ///
    /// Returns error if length is not a multiple of 66 or pointer is misaligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_TQ2_0_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_TQ2_0_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("TQ2_0 slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_TQ2_0_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

/// Decode a 2-bit code at `lane` (0..4) from `byte` for BlockTQ2_0.
///
/// Code map: `0b00→-1`, `0b01→0`, `0b10→+1`, `0b11→0` (reserved treated as zero).
fn ternary_decode_g256(byte: u8, lane: usize) -> i8 {
    let shift = lane * 2;
    let code = (byte >> shift) & 0x03;
    match code {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq2_0_g128_block_size_correct() {
        assert_eq!(
            std::mem::size_of::<BlockTQ2_0_g128>(),
            BLOCK_TQ2_0_G128_BYTES
        );
        assert_eq!(BLOCK_TQ2_0_G128_BYTES, 34);
    }

    #[test]
    fn tq2_0_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockTQ2_0>(), BLOCK_TQ2_0_BYTES);
        assert_eq!(BLOCK_TQ2_0_BYTES, 66);
    }

    #[test]
    fn tq2_0_g128_roundtrip_uniform() {
        // Alternating 0.5, -0.5, 0.0 pattern for 128 values.
        let mut input = [0.0f32; 128];
        for (i, x) in input.iter_mut().enumerate() {
            *x = match i % 3 {
                0 => 0.5,
                1 => -0.5,
                _ => 0.0,
            };
        }
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");
        let mse: f32 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 128.0;
        assert!(mse < 1e-3, "MSE {mse} too high");
    }

    #[test]
    fn tq2_0_g128_all_zero_input() {
        let input = [0.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].d, f16::ZERO);
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");
        for &v in &output {
            assert_eq!(v, 0.0, "all outputs should be zero");
        }
    }

    #[test]
    fn tq2_0_g128_all_positive() {
        let input = [1.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        // absmax = 1.0 → d = f16(1.0)
        assert!(
            (blocks[0].d.to_f32() - 1.0).abs() < 1e-3,
            "d should be ~1.0"
        );
        // All codes should be Pos (0b10), so each byte = 0b10101010 = 0xAA
        for &b in &blocks[0].qs {
            assert_eq!(b, 0xAA, "all bytes should be 0xAA for all-positive");
        }
    }

    #[test]
    fn tq2_0_g128_all_negative() {
        let input = [-1.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        // absmax = 1.0 → d = f16(1.0)
        assert!(
            (blocks[0].d.to_f32() - 1.0).abs() < 1e-3,
            "d should be ~1.0"
        );
        // All codes should be Neg (0b00), so each byte = 0b00000000 = 0x00
        for &b in &blocks[0].qs {
            assert_eq!(b, 0x00, "all bytes should be 0x00 for all-negative");
        }
    }

    #[test]
    fn tq2_0_g128_mixed_threshold() {
        // Pattern: [2.0, 0.9, 0.0, -0.9, -2.0] repeating to fill 128 elements.
        // absmax=2.0, threshold=1.0:
        //   2.0 ≥ 1.0  → Pos (+d = 2.0)
        //   0.9 < 1.0  → Zero (0.0)
        //   0.0 < 1.0  → Zero (0.0)
        //  -0.9: abs=0.9 < 1.0 → Zero (0.0)
        //  -2.0 ≤ -1.0 → Neg (-d = -2.0)
        let mut input = [0.0f32; 128];
        let pattern = [2.0f32, 0.9, 0.0, -0.9, -2.0];
        for i in 0..128 {
            input[i] = pattern[i % 5];
        }
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

        let expected_pattern = [2.0f32, 0.0, 0.0, 0.0, -2.0];
        for i in 0..128 {
            let expected = expected_pattern[i % 5];
            assert!(
                (output[i] - expected).abs() < 1e-3,
                "index {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn tq2_0_g128_slice_from_bytes_misaligned() {
        // 35 bytes is not a multiple of 34 → should return Err.
        let data = vec![0u8; 35];
        let result = BlockTQ2_0_g128::slice_from_bytes(&data);
        assert!(result.is_err(), "35-byte slice should fail");
    }

    #[test]
    fn tq2_0_g128_slice_from_bytes_aligned() {
        // Build a real block and reinterpret as bytes (guaranteed alignment).
        let block = BlockTQ2_0_g128 {
            qs: [0u8; 32],
            d: f16::from_f32(1.0),
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockTQ2_0_g128 as *const u8,
                BLOCK_TQ2_0_G128_BYTES,
            )
        };
        let result =
            BlockTQ2_0_g128::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(1.0));
    }

    #[test]
    fn tq2_0_roundtrip_random() {
        // 256 values oscillating in [-1, 1].
        let mut input = [0.0f32; 256];
        for (i, x) in input.iter_mut().enumerate() {
            *x = ((i as f32) / 128.0 - 1.0).clamp(-1.0, 1.0);
        }
        let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 256];
        BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");
        let mse: f32 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 256.0;
        // TQ2_0 is a 3-level ternary quantizer; on a continuous ramp in [-1,1]
        // a large fraction of values are zeroed (|x| < 0.5 * absmax), so MSE
        // around 0.08–0.10 is expected.  Require < 0.15 to catch regressions.
        assert!(mse < 0.15, "MSE {mse} too high for TQ2_0 roundtrip");
    }

    #[test]
    fn ternary_decode_all_lanes() {
        // Construct a byte to test all four lanes:
        //   lane 0 (bits 1:0): 0b00 → -1
        //   lane 1 (bits 3:2): 0b11 → 0 (reserved)
        //   lane 2 (bits 5:4): 0b01 → 0
        //   lane 3 (bits 7:6): 0b10 → +1
        // Byte = 0b10_01_11_00 = 0b10011100 = 0x9C
        let byte: u8 = 0b10011100;
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 0),
            -1,
            "lane 0: 0b00 → -1"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 1),
            0,
            "lane 1: 0b11 → 0 (reserved)"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 2),
            0,
            "lane 2: 0b01 → 0"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 3),
            1,
            "lane 3: 0b10 → +1"
        );
    }
}
