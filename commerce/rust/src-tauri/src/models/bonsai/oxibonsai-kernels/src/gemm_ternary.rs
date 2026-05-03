//! Reference (naive) GEMM kernels for ternary TQ2\_0\_g128 and TQ2\_0 formats.
//!
//! Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` where
//! weights are ternary-quantized. Used for prompt prefill (batch matmul).
//!
//! Each batch row is processed as an independent GEMV call.

use oxibonsai_core::{BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

use crate::error::{KernelError, KernelResult};
use crate::gemv_ternary::{gemv_tq2_0, gemv_tq2_0_g128};

// ---------------------------------------------------------------------------
// TQ2_0_g128 — 128 weights per block
// ---------------------------------------------------------------------------

/// Scalar GEMM for TQ2\_0\_g128-quantized weight matrix.
///
/// Computes `output[batch, row] = dot(weight_row, input[batch])` for all
/// batch rows and weight rows.
///
/// - `blocks`: Weight blocks, row-major [n\_rows × blocks\_per\_row].
/// - `input`: Row-major FP32 input matrix [m × k].
/// - `output`: Row-major FP32 output matrix [m × n\_rows].
/// - `m`: Batch/sequence dimension.
/// - `n_rows`: Number of weight matrix rows (output columns).
/// - `k`: Inner dimension (must be divisible by 128).
///
/// # Errors
///
/// Propagates all errors from [`gemv_tq2_0_g128`]:
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::DimensionMismatch`] if dimensions mismatch.
/// - [`KernelError::BufferTooSmall`] if any buffer is too small.
pub fn gemm_tq2_0_g128(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }

    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_tq2_0_g128(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TQ2_0 — 256 weights per block
// ---------------------------------------------------------------------------

/// Scalar GEMM for TQ2\_0-quantized weight matrix.
///
/// Computes `output[batch, row] = dot(weight_row, input[batch])` for all
/// batch rows and weight rows.
///
/// - `blocks`: Weight blocks, row-major [n\_rows × blocks\_per\_row].
/// - `input`: Row-major FP32 input matrix [m × k].
/// - `output`: Row-major FP32 output matrix [m × n\_rows].
/// - `m`: Batch/sequence dimension.
/// - `n_rows`: Number of weight matrix rows (output columns).
/// - `k`: Inner dimension (must be divisible by 256).
///
/// # Errors
///
/// Propagates all errors from [`gemv_tq2_0`]:
/// - [`KernelError::NotBlockAligned`] if `k % 256 != 0`.
/// - [`KernelError::DimensionMismatch`] if dimensions mismatch.
/// - [`KernelError::BufferTooSmall`] if any buffer is too small.
pub fn gemm_tq2_0(
    blocks: &[BlockTQ2_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_TQ2_0 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }

    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_tq2_0(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_ternary::gemv_tq2_0_g128;
    use half::f16;

    fn make_g128_block(scale: f32, qs: [u8; 32]) -> BlockTQ2_0_g128 {
        BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    /// gemm output must match 4 individual gemv calls for each batch row.
    ///
    /// m=4, n_rows=2, k=128.
    #[test]
    fn gemm_tq2_0_g128_matches_gemv() {
        // 2 rows of weights, k=128 → 2 blocks total
        let blocks = vec![
            make_g128_block(1.0, [0xAA; 32]), // row 0: all +1
            make_g128_block(1.5, [0x00; 32]), // row 1: all -1 (scale 1.5)
        ];
        // 4 different input rows
        let m = 4;
        let n_rows = 2;
        let k = 128;
        let mut input = vec![0.0_f32; m * k];
        for batch in 0..m {
            for j in 0..k {
                input[batch * k + j] = (batch + 1) as f32 * 0.5;
            }
        }

        // Run GEMM
        let mut gemm_out = vec![0.0_f32; m * n_rows];
        gemm_tq2_0_g128(&blocks, &input, &mut gemm_out, m, n_rows, k).expect("gemm should succeed");

        // Run 4 individual GEMVs and compare
        for batch in 0..m {
            let input_row = &input[batch * k..(batch + 1) * k];
            let mut gemv_out = vec![0.0_f32; n_rows];
            gemv_tq2_0_g128(&blocks, input_row, &mut gemv_out, n_rows, k)
                .expect("gemv should succeed");

            for row in 0..n_rows {
                let gemm_val = gemm_out[batch * n_rows + row];
                let gemv_val = gemv_out[row];
                assert!(
                    (gemm_val - gemv_val).abs() < 1e-4,
                    "batch={batch} row={row}: gemm={gemm_val} vs gemv={gemv_val}",
                );
            }
        }
    }

    /// Batch GEMM with all-positive weights: every output row should equal k * d.
    #[test]
    fn gemm_tq2_0_g128_all_positive() {
        let m = 3;
        let n_rows = 4;
        let k = 128;
        let blocks = vec![make_g128_block(1.0, [0xAA; 32]); n_rows];
        let input = vec![1.0_f32; m * k];
        let mut output = vec![0.0_f32; m * n_rows];

        gemm_tq2_0_g128(&blocks, &input, &mut output, m, n_rows, k).expect("gemm should succeed");

        for batch in 0..m {
            for row in 0..n_rows {
                let v = output[batch * n_rows + row];
                assert!(
                    (v - 128.0).abs() < 0.5,
                    "batch={batch} row={row}: expected 128.0, got {v}",
                );
            }
        }
    }

    /// k=100 is not a multiple of 128 → NotBlockAligned error.
    #[test]
    fn gemm_tq2_0_g128_not_block_aligned() {
        let blocks = vec![make_g128_block(1.0, [0xAA; 32])];
        let input = vec![1.0_f32; 100];
        let mut output = vec![0.0_f32; 1];

        let result = gemm_tq2_0_g128(&blocks, &input, &mut output, 1, 1, 100);
        assert!(result.is_err(), "expected NotBlockAligned error");
    }
}
