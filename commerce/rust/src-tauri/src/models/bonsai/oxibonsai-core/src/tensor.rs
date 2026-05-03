//! Q1\_0\_g128 tensor types and 1-bit data access.
//!
//! Defines the [`BlockQ1_0G128`] structure matching the PrismML GGUF format
//! and the [`OneBitTensor`] wrapper for efficient tensor access.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

/// Number of weights per Q1\_0\_g128 block.
pub const QK1_0_G128: usize = 128;

/// Size of a Q1\_0\_g128 block in bytes (2-byte FP16 scale + 16 bytes sign bits).
pub const BLOCK_SIZE_BYTES: usize = 18;

/// A single Q1\_0\_g128 quantized block.
///
/// Layout (18 bytes total):
/// - `d`: FP16 scale factor (2 bytes) — shared by all 128 weights
/// - `qs`: 128 sign bits packed into 16 bytes
///
/// Weight reconstruction: `w[i] = bit[i] ? +d : -d`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ1_0G128 {
    /// Scale factor (delta), FP16.
    pub d: f16,
    /// 128 sign bits packed into 16 bytes.
    pub qs: [u8; QK1_0_G128 / 8],
}

const _: () = assert!(std::mem::size_of::<BlockQ1_0G128>() == BLOCK_SIZE_BYTES);

impl BlockQ1_0G128 {
    /// Interpret a raw byte slice as a block reference (zero-copy).
    pub fn from_bytes(data: &[u8]) -> BonsaiResult<&Self> {
        if data.len() < BLOCK_SIZE_BYTES {
            return Err(BonsaiError::InvalidBlockSize { actual: data.len() });
        }
        // SAFETY: BlockQ1_0G128 is repr(C) with known layout, and we've validated
        // the minimum size. The f16 type is repr(transparent) over u16.
        let ptr = data.as_ptr() as *const BlockQ1_0G128;
        Ok(unsafe { &*ptr })
    }

    /// Interpret a raw byte slice as a slice of blocks (zero-copy).
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_SIZE_BYTES != 0 {
            return Err(BonsaiError::InvalidBlockSize { actual: data.len() });
        }
        let count = data.len() / BLOCK_SIZE_BYTES;
        let ptr = data.as_ptr() as *const BlockQ1_0G128;
        // SAFETY: Same as above, plus we've checked alignment to block size.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Get the sign bit for weight at index `i` (0..127).
    /// Returns `true` for +d, `false` for -d.
    #[inline]
    pub fn sign_bit(&self, i: usize) -> bool {
        debug_assert!(i < QK1_0_G128);
        let byte_index = i / 8;
        let bit_offset = i % 8;
        (self.qs[byte_index] >> bit_offset) & 1 != 0
    }

    /// Get the reconstructed weight value at index `i`.
    #[inline]
    pub fn weight(&self, i: usize) -> f32 {
        let d = self.d.to_f32();
        if self.sign_bit(i) {
            d
        } else {
            -d
        }
    }
}

/// A 1-bit tensor backed by Q1\_0\_g128 blocks.
///
/// This wraps raw GGUF tensor data and provides typed access to blocks
/// without copying or dequantizing the entire tensor.
#[derive(Debug)]
pub struct OneBitTensor<'a> {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Raw block data.
    blocks: &'a [BlockQ1_0G128],
}

impl<'a> OneBitTensor<'a> {
    /// Create a 1-bit tensor from raw GGUF tensor data bytes.
    pub fn from_raw(name: String, shape: Vec<u64>, data: &'a [u8]) -> BonsaiResult<Self> {
        let blocks = BlockQ1_0G128::slice_from_bytes(data)?;
        Ok(Self {
            name,
            shape,
            blocks,
        })
    }

    /// Number of blocks in this tensor.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of elements (weights) in this tensor.
    pub fn element_count(&self) -> usize {
        self.blocks.len() * QK1_0_G128
    }

    /// Get a reference to the block at the given index.
    pub fn block(&self, index: usize) -> &BlockQ1_0G128 {
        &self.blocks[index]
    }

    /// Get all blocks as a slice.
    pub fn blocks(&self) -> &[BlockQ1_0G128] {
        self.blocks
    }

    /// Dequantize all blocks to FP32 values.
    ///
    /// For the full tensor, this allocates and fills an output vector.
    /// For per-operation dequantization, use the kernel crate instead.
    pub fn dequantize_all(&self) -> Vec<f32> {
        let n = self.element_count();
        let mut output = vec![0.0f32; n];
        for (i, block) in self.blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = i * QK1_0_G128;
            for j in 0..QK1_0_G128 {
                let byte_index = j / 8;
                let bit_offset = j % 8;
                let bit = (block.qs[byte_index] >> bit_offset) & 1;
                output[base + j] = if bit != 0 { d } else { -d };
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn block_size_is_18_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ1_0G128>(), 18);
    }

    #[test]
    fn all_ones_dequantize_to_positive() {
        let block = make_block(2.0, [0xFF; 16]);
        for i in 0..128 {
            assert!(block.sign_bit(i));
            assert!((block.weight(i) - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn all_zeros_dequantize_to_negative() {
        let block = make_block(3.0, [0x00; 16]);
        for i in 0..128 {
            assert!(!block.sign_bit(i));
            assert!((block.weight(i) + 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn alternating_bits() {
        // 0xAA = 10101010 in binary: bits 1,3,5,7 set; 0,2,4,6 clear
        let block = make_block(1.0, [0xAA; 16]);
        for i in 0..128 {
            if i % 2 == 0 {
                assert!(!block.sign_bit(i), "bit {i} should be 0");
            } else {
                assert!(block.sign_bit(i), "bit {i} should be 1");
            }
        }
    }

    #[test]
    fn from_bytes_roundtrip() {
        let block = make_block(1.5, [0xFF; 16]);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockQ1_0G128 as *const u8,
                BLOCK_SIZE_BYTES,
            )
        };
        let parsed = BlockQ1_0G128::from_bytes(bytes).expect("block parse should succeed");
        assert_eq!(parsed, &block);
    }

    #[test]
    fn one_bit_tensor_dequantize() {
        let block = make_block(2.0, [0xFF; 16]);
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockQ1_0G128 as *const u8,
                BLOCK_SIZE_BYTES,
            )
            .to_vec()
        };
        let tensor = OneBitTensor::from_raw("test".to_string(), vec![128], &bytes)
            .expect("tensor creation should succeed");
        assert_eq!(tensor.num_blocks(), 1);
        assert_eq!(tensor.element_count(), 128);

        let values = tensor.dequantize_all();
        for &v in &values {
            assert!((v - 2.0).abs() < 0.01);
        }
    }
}
