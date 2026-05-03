//! K-quant block types for Q2_K and Q4_K quantization formats.
//!
//! These follow the GGML K-quant specification:
//! - **Q2_K**: 2-bit quantization with 4-bit scales, super-block of 256 weights (84 bytes)
//! - **Q4_K**: 4-bit quantization with 6-bit scales, super-block of 256 weights (144 bytes)
//!
//! Each super-block stores a global `d` (scale) and `dmin` (minimum) in FP16,
//! plus per-sub-block scales and quantized weight nibbles/pairs.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per K-quant super-block.
pub const QK_K: usize = 256;

/// Number of bytes per Q2_K block.
pub const BLOCK_Q2_K_BYTES: usize = 84;

/// Number of bytes per Q4_K block.
pub const BLOCK_Q4_K_BYTES: usize = 144;

// ---------------------------------------------------------------------------
// BlockQ2K
// ---------------------------------------------------------------------------

/// Q2_K super-block: 256 weights quantized to 2 bits each.
///
/// Layout (84 bytes):
/// - `scales`: 16 bytes — packed 4-bit scale/min pairs for 16 sub-blocks of 16 weights.
///   Each byte holds two 4-bit values: low nibble = scale, high nibble = min.
/// - `qs`: 64 bytes — 256 x 2-bit quantized weights (4 per byte, LSB first).
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
///
/// Dequant: `w[i] = d * sub_scale * q[i] - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    /// Packed 4-bit scale/min pairs for 16 sub-blocks.
    pub scales: [u8; 16],
    /// 256 x 2-bit quantized weights, 4 per byte.
    pub qs: [u8; 64],
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
}

const _: () = assert!(std::mem::size_of::<BlockQ2K>() == BLOCK_Q2_K_BYTES);

impl BlockQ2K {
    /// Dequantize a slice of Q2_K blocks into f32 output.
    ///
    /// `output` must have length `blocks.len() * QK_K`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let base = block_idx * QK_K;

            // 16 sub-blocks of 16 weights each
            for sub in 0..16 {
                let scale_byte = block.scales[sub];
                let sc = (scale_byte & 0x0F) as f32; // low nibble = scale
                let mn = ((scale_byte >> 4) & 0x0F) as f32; // high nibble = min

                let sub_offset = sub * 16;
                for j in 0..16 {
                    let global_idx = sub_offset + j;
                    // Each byte holds 4 x 2-bit values
                    let byte_idx = global_idx / 4;
                    let shift = (global_idx % 4) * 2;
                    let q = ((block.qs[byte_idx] >> shift) & 0x03) as f32;
                    output[base + global_idx] = d * sc * q - dmin * mn;
                }
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q2_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_K;
            let chunk = &input[base..base + QK_K];

            // Pass 1: find global max absolute value and min value across
            // all sub-blocks to set d and dmin.
            // For each sub-block of 16 weights, we find the range [min, max].
            let mut sub_scales = [0.0f32; 16];
            let mut sub_mins = [0.0f32; 16];

            for sub in 0..16 {
                let sub_offset = sub * 16;
                let sub_chunk = &chunk[sub_offset..sub_offset + 16];

                let mut smin = f32::MAX;
                let mut smax = f32::MIN;
                for &v in sub_chunk {
                    if v < smin {
                        smin = v;
                    }
                    if v > smax {
                        smax = v;
                    }
                }

                // The offset (min) removes the minimum, then scale maps remainder to 0..3
                sub_mins[sub] = if smin < 0.0 { -smin } else { 0.0 };
                let range = smax + sub_mins[sub];
                sub_scales[sub] = if range > 0.0 { range / 3.0 } else { 0.0 };
            }

            // Find the global maximum scale and minimum across sub-blocks
            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

            // Compute d and dmin so that 4-bit sub-block factors (0..15) can represent
            // the per-sub-block scales and mins.
            let d = if max_scale > 0.0 {
                max_scale / 15.0
            } else {
                0.0
            };
            let dmin = if max_min > 0.0 { max_min / 15.0 } else { 0.0 };

            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
            let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

            // Quantize per-sub-block scales and mins to 4 bits
            let mut scales = [0u8; 16];
            let mut quant_sc = [0u8; 16];
            let mut quant_mn = [0u8; 16];

            for sub in 0..16 {
                let sc = (sub_scales[sub] * inv_d + 0.5).min(15.0) as u8;
                let mn = (sub_mins[sub] * inv_dmin + 0.5).min(15.0) as u8;
                quant_sc[sub] = sc;
                quant_mn[sub] = mn;
                scales[sub] = sc | (mn << 4);
            }

            // Quantize weights to 2 bits
            let mut qs = [0u8; 64];
            for sub in 0..16 {
                let sub_offset = sub * 16;
                let sc_f = d * (quant_sc[sub] as f32);
                let mn_f = dmin * (quant_mn[sub] as f32);
                let inv_sc = if sc_f > 0.0 { 1.0 / sc_f } else { 0.0 };

                for j in 0..16 {
                    let global_idx = sub_offset + j;
                    let val = chunk[global_idx] + mn_f;
                    let q = (val * inv_sc + 0.5).clamp(0.0, 3.0) as u8;
                    let byte_idx = global_idx / 4;
                    let shift = (global_idx % 4) * 2;
                    qs[byte_idx] |= q << shift;
                }
            }

            blocks.push(BlockQ2K {
                scales,
                qs,
                d: f16::from_f32(d),
                dmin: f16::from_f32(dmin),
            });
        }

        Ok(blocks)
    }
}

// ---------------------------------------------------------------------------
// BlockQ4K
// ---------------------------------------------------------------------------

/// Q4_K super-block: 256 weights quantized to 4 bits each.
///
/// Layout (144 bytes):
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
/// - `scales`: 12 bytes — packed 6-bit scale/min values for 8 sub-blocks of 32 weights.
///   Encoding: bytes 0..3 hold low 4 bits of scale[0..7], bytes 4..7 hold low 4 bits
///   of min[0..7], bytes 8..11 hold the upper 2 bits of scales and mins packed.
/// - `qs`: 128 bytes — 256 x 4-bit quantized weights (2 per byte).
///
/// Dequant: `w[i] = d * sub_scale * q[i] - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ4K {
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
    /// Packed 6-bit scales for 8 sub-blocks.
    pub scales: [u8; 12],
    /// 256 x 4-bit quantized weights, 2 per byte.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == BLOCK_Q4_K_BYTES);

/// Decode the 8 six-bit scale values and 8 six-bit min values from the
/// 12-byte packed `scales` array in a Q4_K block.
///
/// Layout of the 12 bytes:
/// - bytes 0..3:  low 4 bits of scale[0..7] (two per byte, 4 bits each)
/// - bytes 4..7:  low 4 bits of min[0..7]   (two per byte, 4 bits each)
/// - bytes 8..11: upper 2 bits of scale and min, packed
///
/// Specifically for bytes 8..11:
/// - byte  8: bits 0..1 = scale[0] hi, bits 2..3 = scale[1] hi, bits 4..5 = scale[2] hi, bits 6..7 = scale[3] hi
/// - byte  9: bits 0..1 = scale[4] hi, bits 2..3 = scale[5] hi, bits 4..5 = scale[6] hi, bits 6..7 = scale[7] hi
/// - byte 10: bits 0..1 = min[0] hi,   bits 2..3 = min[1] hi,   bits 4..5 = min[2] hi,   bits 6..7 = min[3] hi
/// - byte 11: bits 0..1 = min[4] hi,   bits 2..3 = min[5] hi,   bits 4..5 = min[6] hi,   bits 6..7 = min[7] hi
fn decode_q4k_scales(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // Low 4 bits of scales (2 per byte in bytes 0..3)
    for i in 0..4 {
        sc[2 * i] = scales_raw[i] & 0x0F;
        sc[2 * i + 1] = (scales_raw[i] >> 4) & 0x0F;
    }

    // Low 4 bits of mins (2 per byte in bytes 4..7)
    for i in 0..4 {
        mn[2 * i] = scales_raw[4 + i] & 0x0F;
        mn[2 * i + 1] = (scales_raw[4 + i] >> 4) & 0x0F;
    }

    // Upper 2 bits of scales from bytes 8..9
    for i in 0..4 {
        sc[i] |= ((scales_raw[8] >> (2 * i)) & 0x03) << 4;
        sc[4 + i] |= ((scales_raw[9] >> (2 * i)) & 0x03) << 4;
    }

    // Upper 2 bits of mins from bytes 10..11
    for i in 0..4 {
        mn[i] |= ((scales_raw[10] >> (2 * i)) & 0x03) << 4;
        mn[4 + i] |= ((scales_raw[11] >> (2 * i)) & 0x03) << 4;
    }

    (sc, mn)
}

/// Encode 8 six-bit scale values and 8 six-bit min values into the 12-byte
/// packed format used by Q4_K blocks.
fn encode_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];

    // Low 4 bits of scales into bytes 0..3
    for i in 0..4 {
        out[i] = (sc[2 * i] & 0x0F) | ((sc[2 * i + 1] & 0x0F) << 4);
    }

    // Low 4 bits of mins into bytes 4..7
    for i in 0..4 {
        out[4 + i] = (mn[2 * i] & 0x0F) | ((mn[2 * i + 1] & 0x0F) << 4);
    }

    // Upper 2 bits of scales into bytes 8..9
    for i in 0..4 {
        out[8] |= ((sc[i] >> 4) & 0x03) << (2 * i);
        out[9] |= ((sc[4 + i] >> 4) & 0x03) << (2 * i);
    }

    // Upper 2 bits of mins into bytes 10..11
    for i in 0..4 {
        out[10] |= ((mn[i] >> 4) & 0x03) << (2 * i);
        out[11] |= ((mn[4 + i] >> 4) & 0x03) << (2 * i);
    }

    out
}

impl BlockQ4K {
    /// Dequantize a slice of Q4_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let dmin_val = block.dmin.to_f32();
            let base = block_idx * QK_K;

            let (sc, mn) = decode_q4k_scales(&block.scales);

            // 8 sub-blocks of 32 weights each
            for sub in 0..8 {
                let sub_scale = d * (sc[sub] as f32);
                let sub_min = dmin_val * (mn[sub] as f32);
                let sub_offset = sub * 32;

                for j in 0..32 {
                    let global_idx = sub_offset + j;
                    let byte_idx = global_idx / 2;
                    let q = if global_idx % 2 == 0 {
                        (block.qs[byte_idx] & 0x0F) as f32
                    } else {
                        ((block.qs[byte_idx] >> 4) & 0x0F) as f32
                    };
                    output[base + global_idx] = sub_scale * q - sub_min;
                }
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q4_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_K;
            let chunk = &input[base..base + QK_K];

            // 8 sub-blocks of 32 weights
            let mut sub_scales = [0.0f32; 8];
            let mut sub_mins = [0.0f32; 8];

            for sub in 0..8 {
                let sub_offset = sub * 32;
                let sub_chunk = &chunk[sub_offset..sub_offset + 32];

                let mut smin = f32::MAX;
                let mut smax = f32::MIN;
                for &v in sub_chunk {
                    if v < smin {
                        smin = v;
                    }
                    if v > smax {
                        smax = v;
                    }
                }

                sub_mins[sub] = if smin < 0.0 { -smin } else { 0.0 };
                let range = smax + sub_mins[sub];
                sub_scales[sub] = if range > 0.0 { range / 15.0 } else { 0.0 };
            }

            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

            // 6-bit sub-block factors: 0..63
            let d = if max_scale > 0.0 {
                max_scale / 63.0
            } else {
                0.0
            };
            let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
            let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];

            for sub in 0..8 {
                sc[sub] = (sub_scales[sub] * inv_d + 0.5).min(63.0) as u8;
                mn[sub] = (sub_mins[sub] * inv_dmin + 0.5).min(63.0) as u8;
            }

            let scales = encode_q4k_scales(&sc, &mn);

            // Quantize weights to 4 bits
            let mut qs = [0u8; 128];
            for sub in 0..8 {
                let sub_offset = sub * 32;
                let sc_f = d * (sc[sub] as f32);
                let mn_f = dmin * (mn[sub] as f32);
                let inv_sc = if sc_f > 0.0 { 1.0 / sc_f } else { 0.0 };

                for j in 0..32 {
                    let global_idx = sub_offset + j;
                    let val = chunk[global_idx] + mn_f;
                    let q = (val * inv_sc + 0.5).clamp(0.0, 15.0) as u8;
                    let byte_idx = global_idx / 2;
                    if global_idx % 2 == 0 {
                        qs[byte_idx] |= q & 0x0F;
                    } else {
                        qs[byte_idx] |= (q & 0x0F) << 4;
                    }
                }
            }

            blocks.push(BlockQ4K {
                d: f16::from_f32(d),
                dmin: f16::from_f32(dmin),
                scales,
                qs,
            });
        }

        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ2K>(), BLOCK_Q2_K_BYTES);
        assert_eq!(BLOCK_Q2_K_BYTES, 84);
    }

    #[test]
    fn q4k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), BLOCK_Q4_K_BYTES);
        assert_eq!(BLOCK_Q4_K_BYTES, 144);
    }

    #[test]
    fn q4k_scale_encode_decode_roundtrip() {
        let sc = [1, 2, 3, 4, 5, 63, 32, 0];
        let mn = [10, 20, 30, 40, 50, 60, 15, 7];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q4k_scale_encode_decode_all_zeros() {
        let sc = [0u8; 8];
        let mn = [0u8; 8];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q4k_scale_encode_decode_max_values() {
        let sc = [63u8; 8];
        let mn = [63u8; 8];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }
}
