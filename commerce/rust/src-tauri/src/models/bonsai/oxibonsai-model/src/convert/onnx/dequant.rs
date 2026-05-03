//! Microsoft MatMulNBits (bits=2, block_size=128) → f32 dequantization.
//!
//! The MatMulNBits op (domain `com.microsoft`) stores a quantized weight
//! matrix `B` of logical shape `[N, K]` (rows = output features, cols =
//! input features) split into blocks of `block_size` columns along `K`.
//! For `bits = 2`:
//!
//! * Packed bytes: `B.shape = [N, n_blocks_per_row, block_size / 4]`
//!   where `n_blocks_per_row = K.div_ceil(block_size)`. Each byte holds 4
//!   unsigned 2-bit codes `c ∈ {0, 1, 2, 3}`, LSB-first:
//!   `byte = (c0 << 0) | (c1 << 2) | (c2 << 4) | (c3 << 6)`.
//! * Scales: one f32 per (row, block) in row-major
//!   `[N * n_blocks_per_row]` order.
//! * Zero points (optional): one unsigned 2-bit code per (row, block),
//!   packed LSB-first into `ceil(N * n_blocks_per_row / 4)` bytes.
//!   Default zero point when absent is `(1 << bits) / 2 = 2`.
//!
//! Each dequantized element is computed as
//!
//! ```text
//! f = (code - zero_point) as f32 * scale
//! ```
//!
//! yielding an output buffer of logical shape `[N, K]` (row-major, same as
//! the conceptual HuggingFace `nn.Linear.weight`). Callers are expected to
//! reverse the dimensions when emitting GGUF shape (GGUF uses `[K, N]`).
//!
//! This module handles only the `bits = 2, block_size = 128` case, which is
//! the single configuration used by `onnx-community/Ternary-Bonsai-1.7B-ONNX`
//! and Microsoft's Qwen3 Matmul 2-bit ONNX exports. Other configurations
//! return [`DequantError::Unsupported`].

use super::error::DequantError;

/// Ternary block size hard-coded by `onnx-community/Ternary-Bonsai-1.7B-ONNX`.
pub const EXPECTED_BLOCK_SIZE: usize = 128;

/// The only number of bits currently supported.
pub const EXPECTED_BITS: u32 = 2;

/// Unpack a single byte into four unsigned 2-bit codes, LSB-first.
///
/// Output `out[i] = (byte >> (2 * i)) & 0b11` for `i ∈ 0..4`.
#[inline]
pub fn unpack_2bit_le(byte: u8) -> [u8; 4] {
    [
        byte & 0b11,
        (byte >> 2) & 0b11,
        (byte >> 4) & 0b11,
        (byte >> 6) & 0b11,
    ]
}

/// Dequantize a MatMulNBits weight matrix into a `Vec<f32>` of row-major
/// `[N, K]` data.
///
/// # Arguments
///
/// * `packed` — raw bytes of the `B` initializer (after external-data
///   resolution). Must be exactly `n * n_blocks * (block_size / 4)` bytes.
/// * `scales` — `n * n_blocks` f32 values, one per (row, block).
/// * `zero_points` — optional packed 2-bit zero-point codes (same layout as
///   `packed` for codes but only `n * n_blocks` codes total). When
///   `None`, the default zero point `2` is used uniformly.
/// * `n` — number of output features (rows).
/// * `k` — number of input features (columns); MatMulNBits typically pads K
///   internally to a multiple of `block_size`. Pass the attribute value
///   exactly; padding is handled transparently.
/// * `bits` — must be `EXPECTED_BITS` (2); other values return
///   [`DequantError::Unsupported`].
/// * `block_size` — must be `EXPECTED_BLOCK_SIZE` (128); other values return
///   [`DequantError::Unsupported`].
///
/// # Errors
///
/// * [`DequantError::Unsupported`] — unsupported `bits`/`block_size` combo.
/// * [`DequantError::LengthMismatch`] — any buffer length disagrees with
///   the required shape.
pub fn dequantize_matmul_nbits(
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    n: usize,
    k: usize,
    bits: u32,
    block_size: usize,
) -> Result<Vec<f32>, DequantError> {
    if bits != EXPECTED_BITS {
        return Err(DequantError::Unsupported(format!(
            "unsupported bits={bits} (only 2 is implemented)"
        )));
    }
    if block_size != EXPECTED_BLOCK_SIZE {
        return Err(DequantError::Unsupported(format!(
            "unsupported block_size={block_size} (only 128 is implemented)"
        )));
    }
    if n == 0 || k == 0 {
        return Ok(Vec::new());
    }

    // MatMulNBits conceptually pads K to a multiple of block_size for storage.
    let n_blocks = k.div_ceil(block_size);
    let bytes_per_row = n_blocks * (block_size / 4); // bits=2 → 4 codes/byte
    let expected_packed = n * bytes_per_row;
    if packed.len() != expected_packed {
        return Err(DequantError::LengthMismatch {
            what: "packed B",
            expected: expected_packed,
            got: packed.len(),
        });
    }

    let expected_scales = n * n_blocks;
    if scales.len() != expected_scales {
        return Err(DequantError::LengthMismatch {
            what: "scales",
            expected: expected_scales,
            got: scales.len(),
        });
    }

    if let Some(zp) = zero_points {
        // 4 zero-point codes per byte.
        let expected_zp_bytes = expected_scales.div_ceil(4);
        if zp.len() != expected_zp_bytes {
            return Err(DequantError::LengthMismatch {
                what: "zero_points",
                expected: expected_zp_bytes,
                got: zp.len(),
            });
        }
    }

    let k_padded = n_blocks * block_size;
    let mut out = vec![0.0_f32; n * k];

    for row in 0..n {
        let row_packed_base = row * bytes_per_row;
        let row_scales_base = row * n_blocks;

        for block_idx in 0..n_blocks {
            let scale = scales[row_scales_base + block_idx];

            // Resolve zero point for (row, block).
            let zp_value: u8 = if let Some(zp) = zero_points {
                let global_zp_idx = row_scales_base + block_idx;
                let byte = zp[global_zp_idx / 4];
                (byte >> (2 * (global_zp_idx % 4))) & 0b11
            } else {
                // Default zero-point for unsigned uniform quant is 2^(bits-1) = 2.
                2
            };
            let zp_f32 = zp_value as f32;

            let block_packed_base = row_packed_base + block_idx * (block_size / 4);
            let block_k_base = block_idx * block_size;

            for byte_idx in 0..(block_size / 4) {
                let byte = packed[block_packed_base + byte_idx];
                let codes = unpack_2bit_le(byte);
                let k_base = block_k_base + byte_idx * 4;
                for (lane, code) in codes.iter().enumerate() {
                    let k_pos = k_base + lane;
                    // Drop the padded tail cells beyond the real K.
                    if k_pos >= k_padded || k_pos >= k {
                        continue;
                    }
                    let value = ((*code as f32) - zp_f32) * scale;
                    out[row * k + k_pos] = value;
                }
            }
        }
    }

    Ok(out)
}

/// Re-pack a 4-bit-packed zero-point buffer (two nibbles per byte) into a
/// 2-bit-packed buffer (four codes per byte), LSB-first.
///
/// Used by the `GatherBlockQuantized` embedding path for the 8B Ternary-Bonsai
/// ONNX export. Although the GBQ op carries a `bits=4` attribute, the actual
/// packed data (both weights *and* zero-points) is 2-bit ternary — every
/// nibble in the zero-point buffer is `≤ 3`. We re-pack the ZP buffer so it
/// can be fed directly into [`dequantize_matmul_nbits`] with `bits=2`.
///
/// # Input layout
///
/// Each byte of `zp_4bit` holds two 4-bit nibbles. The low nibble
/// (`b & 0x0F`) comes first, the high nibble (`b >> 4`) second. The flat
/// nibble sequence is:
///
/// ```text
/// nibble_0 = zp_4bit[0] & 0x0F
/// nibble_1 = zp_4bit[0] >> 4
/// nibble_2 = zp_4bit[1] & 0x0F
/// nibble_3 = zp_4bit[1] >> 4
/// ...
/// ```
///
/// # Output layout
///
/// Four 2-bit codes per byte, LSB-first:
///
/// ```text
/// out_byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
/// ```
///
/// # Arguments
///
/// * `zp_4bit` — raw bytes of the GBQ zero-point buffer. Must contain at
///   least `total_codes.div_ceil(2)` bytes.
/// * `total_codes` — exact number of 2-bit codes to produce. The tail of
///   the final output byte is zero-filled when `total_codes` is not a
///   multiple of 4.
///
/// # Errors
///
/// * [`DequantError::LengthMismatch`] — `zp_4bit` is shorter than
///   `total_codes.div_ceil(2)` bytes.
/// * [`DequantError::NibbleOutOfRange`] — any nibble in the range
///   `[0, total_codes)` has a value greater than `3` (which would indicate
///   the input really is 4-bit, contradicting our ternary assumption).
pub fn repack_4bit_zp_to_2bit(zp_4bit: &[u8], total_codes: usize) -> Result<Vec<u8>, DequantError> {
    let required_input_bytes = total_codes.div_ceil(2);
    if zp_4bit.len() < required_input_bytes {
        return Err(DequantError::LengthMismatch {
            what: "4-bit ZP input",
            expected: required_input_bytes,
            got: zp_4bit.len(),
        });
    }

    let out_len = total_codes.div_ceil(4);
    let mut out = vec![0u8; out_len];

    for code_idx in 0..total_codes {
        let byte_idx = code_idx / 2;
        let nibble = if code_idx % 2 == 0 {
            zp_4bit[byte_idx] & 0x0F
        } else {
            zp_4bit[byte_idx] >> 4
        };
        if nibble > 3 {
            return Err(DequantError::NibbleOutOfRange {
                index: code_idx,
                value: nibble,
            });
        }
        let code = nibble & 0b11;
        let out_byte_idx = code_idx / 4;
        let out_lane = code_idx % 4;
        out[out_byte_idx] |= code << (2 * out_lane);
    }

    Ok(out)
}

#[cfg(test)]
// Nibble packing: `(hi << 4) | lo` kept verbose to document both slots.
#[allow(clippy::identity_op)]
mod tests {
    use super::*;

    #[test]
    fn unpack_2bit_le_works() {
        // byte = 0b11_10_01_00 → codes [0, 1, 2, 3] (LSB first).
        assert_eq!(unpack_2bit_le(0b11_10_01_00), [0, 1, 2, 3]);
        // byte = 0xFF → [3, 3, 3, 3].
        assert_eq!(unpack_2bit_le(0xFF), [3, 3, 3, 3]);
        // byte = 0 → all zeros.
        assert_eq!(unpack_2bit_le(0), [0, 0, 0, 0]);
    }

    #[test]
    fn dequant_single_block_zp1_matches_ternary() {
        // Construct one block of K=128 with codes [0,1,2,0,1,2,…] repeating.
        // With scale=1.0 and zp=1, this produces values [-1,0,1,-1,0,1,…].
        let block_size = 128;
        let n = 1;
        let k = 128;
        let n_blocks = 1;
        let bytes_per_row = n_blocks * (block_size / 4); // 32 bytes

        let mut packed = vec![0u8; n * bytes_per_row];
        // Produce codes[i] = i % 3 (0,1,2,0,1,2,…).
        for code_idx in 0..(block_size) {
            let code = (code_idx % 3) as u8;
            let byte_idx = code_idx / 4;
            let lane = code_idx % 4;
            packed[byte_idx] |= code << (2 * lane);
        }

        let scales = vec![1.0_f32; n * n_blocks];
        // Single zero-point code = 1 (binary 01), packed into one byte's
        // first lane. div_ceil(1, 4) = 1 byte total.
        let zero_points = vec![0b01u8];

        let out =
            dequantize_matmul_nbits(&packed, &scales, Some(&zero_points), n, k, 2, block_size)
                .expect("dequantize ok");

        assert_eq!(out.len(), n * k);
        for (i, v) in out.iter().enumerate() {
            let expected = match i % 3 {
                0 => -1.0_f32,
                1 => 0.0,
                _ => 1.0,
            };
            assert!(
                (*v - expected).abs() < 1e-6,
                "mismatch at {i}: got {v}, want {expected}"
            );
        }
    }

    #[test]
    fn dequant_default_zp_is_2() {
        // Without zero_points, the default zero-point is 2 for bits=2.
        // A single code=0 → (0 - 2) * scale = -2.0, code=2 → 0.0, code=3 → 1.0*scale.
        let block_size = 128;
        let n = 1;
        let k = 128;
        let n_blocks = 1;
        let bytes_per_row = n_blocks * (block_size / 4);

        let mut packed = vec![0u8; n * bytes_per_row];
        // All codes = 2 (binary 10) → packed byte = 0b10_10_10_10 = 0xAA.
        for byte in packed.iter_mut() {
            *byte = 0xAA;
        }

        let scales = vec![0.5_f32; n * n_blocks];
        let out = dequantize_matmul_nbits(&packed, &scales, None, n, k, 2, block_size)
            .expect("dequantize ok");

        // (2 - 2) * 0.5 = 0.0 everywhere.
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn dequant_rejects_unsupported_bits() {
        let err = dequantize_matmul_nbits(&[], &[], None, 0, 0, 4, 128).unwrap_err();
        assert!(matches!(err, DequantError::Unsupported(_)));
    }

    #[test]
    fn dequant_rejects_unsupported_block_size() {
        let err = dequantize_matmul_nbits(&[], &[], None, 0, 0, 2, 64).unwrap_err();
        assert!(matches!(err, DequantError::Unsupported(_)));
    }

    #[test]
    fn dequant_rejects_packed_length_mismatch() {
        // n=2, k=128, block_size=128 → 1 block/row × 32 bytes × 2 rows = 64 bytes.
        // Pass only 63 bytes to trigger the error.
        let packed = vec![0u8; 63];
        let scales = vec![1.0_f32; 2];
        let err = dequantize_matmul_nbits(&packed, &scales, None, 2, 128, 2, 128).unwrap_err();
        match err {
            DequantError::LengthMismatch {
                what,
                expected,
                got,
            } => {
                assert_eq!(what, "packed B");
                assert_eq!(expected, 64);
                assert_eq!(got, 63);
            }
            _ => panic!("expected LengthMismatch, got {:?}", err),
        }
    }

    #[test]
    fn dequant_k_padding_truncates_to_real_k() {
        // n=1, block_size=128, k=120 → still 1 block but last 8 cells dropped.
        let block_size = 128;
        let n = 1;
        let k = 120;
        let n_blocks = 1;
        let bytes_per_row = n_blocks * (block_size / 4);

        // All codes = 3 → after (3-2)*scale=1.0*1.0 = 1.0.
        let packed = vec![0xFFu8; n * bytes_per_row];
        let scales = vec![1.0_f32; n * n_blocks];
        let out = dequantize_matmul_nbits(&packed, &scales, None, n, k, 2, block_size).expect("ok");

        assert_eq!(out.len(), n * k);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    // ─── repack_4bit_zp_to_2bit ────────────────────────────────────────────

    #[test]
    fn repack_happy_path_eight_codes() {
        // 4 input bytes carry 8 nibbles; each ≤ 3. Sequence: 0,1,2,3,0,1,2,3.
        // Expected output: two bytes of 4 codes each, LSB-first:
        //   byte 0 = 0 | (1<<2) | (2<<4) | (3<<6) = 0b11_10_01_00 = 0xE4
        //   byte 1 = 0 | (1<<2) | (2<<4) | (3<<6) = 0xE4
        let input: Vec<u8> = vec![
            (1u8 << 4) | 0, // nibbles 0, 1 -> codes 0, 1
            (3u8 << 4) | 2, // nibbles 2, 3 -> codes 2, 3
            (1u8 << 4) | 0, // nibbles 4, 5 -> codes 0, 1
            (3u8 << 4) | 2, // nibbles 6, 7 -> codes 2, 3
        ];
        let out = repack_4bit_zp_to_2bit(&input, 8).expect("happy path ok");
        assert_eq!(out, vec![0xE4, 0xE4]);
    }

    #[test]
    fn repack_trailing_partial_byte() {
        // 6 codes → needs ceil(6/2)=3 input bytes, produces ceil(6/4)=2 output
        // bytes. Codes = 1,2,3,0,1,2. The final output byte carries only two
        // codes in its low 4 bits; the high 4 bits must be zero.
        let input: Vec<u8> = vec![
            (2u8 << 4) | 1, // nibbles 0, 1 -> codes 1, 2
            (0u8 << 4) | 3, // nibbles 2, 3 -> codes 3, 0
            (2u8 << 4) | 1, // nibbles 4, 5 -> codes 1, 2 (nibbles 5 and 6 present in byte)
        ];
        let out = repack_4bit_zp_to_2bit(&input, 6).expect("partial tail ok");
        assert_eq!(out.len(), 2);
        // First byte: c0=1, c1=2, c2=3, c3=0
        //   = 1 | (2 << 2) | (3 << 4) | (0 << 6)
        //   = 0b00_11_10_01 = 0x39
        assert_eq!(out[0], 0x39);
        // Second byte: c4=1, c5=2, c6=0 (unused), c7=0 (unused)
        //   = 1 | (2 << 2) | 0 | 0 = 0b00_00_10_01 = 0x09
        assert_eq!(out[1], 0x09);
        // Explicitly verify the high 4 bits of the tail byte are zero
        // (i.e. the two unused code slots were not populated with junk).
        assert_eq!(out[1] & 0xF0, 0);
    }

    #[test]
    fn repack_rejects_nibble_above_three() {
        // Byte 0 low nibble = 0x4 (> 3) → must be rejected at nibble index 0.
        let input: Vec<u8> = vec![0x04, 0x00];
        let err = repack_4bit_zp_to_2bit(&input, 4).expect_err("should reject");
        match err {
            DequantError::NibbleOutOfRange { index, value } => {
                assert_eq!(index, 0);
                assert_eq!(value, 0x4);
            }
            other => panic!("expected NibbleOutOfRange, got {other:?}"),
        }

        // High-nibble case: byte 0 = 0xF0 (low=0 OK, high=0xF > 3) →
        // error at nibble index 1.
        let input_hi: Vec<u8> = vec![0xF0];
        let err_hi = repack_4bit_zp_to_2bit(&input_hi, 2).expect_err("should reject");
        match err_hi {
            DequantError::NibbleOutOfRange { index, value } => {
                assert_eq!(index, 1);
                assert_eq!(value, 0xF);
            }
            other => panic!("expected NibbleOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn repack_empty_input() {
        // total_codes = 0 → empty output, no error even with empty input.
        let out = repack_4bit_zp_to_2bit(&[], 0).expect("empty ok");
        assert!(out.is_empty());
    }

    #[test]
    fn repack_rejects_short_input() {
        // total_codes = 5 requires ceil(5/2) = 3 input bytes; provide only 2.
        let input: Vec<u8> = vec![0x00, 0x00];
        let err = repack_4bit_zp_to_2bit(&input, 5).expect_err("short input");
        match err {
            DequantError::LengthMismatch {
                what,
                expected,
                got,
            } => {
                assert_eq!(what, "4-bit ZP input");
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }
}
