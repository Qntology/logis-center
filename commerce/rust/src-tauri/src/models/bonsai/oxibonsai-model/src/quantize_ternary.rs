//! Ternary quantization helpers for GGUF export.
//!
//! Serializes `f32` weight data into the TQ2_0_g128 byte format (34 bytes
//! per 128-weight group) for embedding in a GGUF tensor data section.

use oxibonsai_core::{error::BonsaiError, BlockTQ2_0_g128, BLOCK_TQ2_0_G128_BYTES};

/// Number of weights covered by one TQ2_0_g128 block.
pub const TERNARY_GROUP_SIZE: usize = 128;

/// Compute the byte size for a ternary-quantized tensor with `elements` weights.
///
/// Uses ceiling division so tensors whose weight count is not a multiple of
/// `TERNARY_GROUP_SIZE` are still accounted for correctly.
#[inline]
pub fn tq2_0_g128_size_bytes(elements: usize) -> usize {
    elements.div_ceil(TERNARY_GROUP_SIZE) * BLOCK_TQ2_0_G128_BYTES
}

/// Quantize f32 weight data to the TQ2_0_g128 byte representation.
///
/// If `data.len()` is not already a multiple of 128, the slice is zero-padded
/// to the next multiple before quantizing. A [`tracing::warn!`] is emitted when
/// padding is applied; callers should pre-align their tensors to avoid padding.
///
/// Returns raw bytes suitable for embedding directly into a GGUF tensor data
/// section. The returned length always equals
/// `tq2_0_g128_size_bytes(data.len())`.
pub fn quantize_tq2_0_g128(data: &[f32]) -> Result<Vec<u8>, BonsaiError> {
    let len = data.len();

    // Pad to the next multiple of TERNARY_GROUP_SIZE if needed.
    let padded: std::borrow::Cow<[f32]> = if len % TERNARY_GROUP_SIZE == 0 {
        std::borrow::Cow::Borrowed(data)
    } else {
        let pad = TERNARY_GROUP_SIZE - (len % TERNARY_GROUP_SIZE);
        tracing::warn!(
            original_len = len,
            padded_len = len + pad,
            "quantize_tq2_0_g128: padding input to multiple of 128"
        );
        let mut v = data.to_vec();
        v.resize(len + pad, 0.0_f32);
        std::borrow::Cow::Owned(v)
    };

    let blocks = BlockTQ2_0_g128::quantize(&padded)?;

    // Serialize blocks to raw bytes via zero-copy pointer cast.
    // SAFETY: BlockTQ2_0_g128 is #[repr(C)] with a compile-time assert that
    // its size is exactly BLOCK_TQ2_0_G128_BYTES (34) bytes. The struct
    // contains only a [u8; 32] and an f16 (u16 layout), so alignment is
    // trivially satisfied when reading as u8. The lifetime of the source
    // allocation is guaranteed to outlive `block_bytes` since `blocks` is
    // alive for the duration of the copy.
    let byte_len = blocks.len() * BLOCK_TQ2_0_G128_BYTES;
    let block_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
    Ok(block_bytes.to_vec())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq2_0_g128_size_bytes_correct() {
        assert_eq!(tq2_0_g128_size_bytes(128), 34);
        assert_eq!(tq2_0_g128_size_bytes(256), 68);
        assert_eq!(tq2_0_g128_size_bytes(129), 68); // rounds up
        assert_eq!(tq2_0_g128_size_bytes(0), 0);
    }

    #[test]
    fn quantize_roundtrip_uniform() {
        // Pattern [1.0, -1.0, 0.0, …] × 128 — quantize → bytes → reload → dequant → compare.
        let mut data = vec![0.0_f32; 128];
        for (i, v) in data.iter_mut().enumerate() {
            *v = match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            };
        }
        let bytes = quantize_tq2_0_g128(&data).expect("quantize ok");
        assert_eq!(bytes.len(), 34);

        let blocks = BlockTQ2_0_g128::slice_from_bytes(&bytes).expect("slice ok");
        let mut out = vec![0.0_f32; 128];
        BlockTQ2_0_g128::dequant(blocks, &mut out).expect("dequant ok");

        // MSE should be < 1e-3
        let mse: f32 = data
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        assert!(mse < 1e-3, "MSE too high: {mse}");
    }

    #[test]
    fn size_bytes_matches_actual_output() {
        let data = vec![1.0_f32; 128];
        let bytes = quantize_tq2_0_g128(&data).expect("ok");
        assert_eq!(bytes.len(), tq2_0_g128_size_bytes(128));
    }

    #[test]
    fn size_bytes_matches_actual_output_256() {
        let data = vec![-1.0_f32; 256];
        let bytes = quantize_tq2_0_g128(&data).expect("ok");
        assert_eq!(bytes.len(), tq2_0_g128_size_bytes(256));
    }

    #[test]
    fn padding_applied_for_non_aligned_length() {
        // 130 elements → padded to 256 → 2 groups → 68 bytes.
        let data = vec![1.0_f32; 130];
        let bytes = quantize_tq2_0_g128(&data).expect("ok");
        assert_eq!(
            bytes.len(),
            68,
            "130 elements should produce 2 blocks (68 bytes)"
        );
    }
}
