//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::{CudaGraphError, QkvBuffers};

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Ensure QKV projection buffers are allocated for `(input_len, output_len)`.
    /// Re-allocates if the existing buffers are too small.
    fn acquire_qkv_buffers(
        &self,
        input_len: usize,
        output_len: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<QkvBuffers>>, CudaGraphError> {
        let mut guard = self
            .qkv_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.fits(input_len, output_len),
            None => true,
        };
        if needs_alloc {
            let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros qkv({n}): {e}")))
            };
            *guard = Some(QkvBuffers {
                d_input: alloc(input_len)?,
                d_output: alloc(output_len)?,
                input_capacity: input_len,
                output_capacity: output_len,
            });
        }
        Ok(guard)
    }
    /// Execute a QKV projection using pre-allocated device buffers.
    ///
    /// Eliminates per-call `cuMemAlloc`/`cuMemFree` that penalised the V1 path.
    /// Uses V8 (shared-mem input cache) when `k ≤ 48 KB threshold`, V7 otherwise.
    pub fn encode_qkv_phase(
        &self,
        input: &[f32],
        output: &mut [f32],
        weight_w: &Arc<CudaSlice<u8>>,
        n_rows: usize,
        k: usize,
    ) -> Result<(), CudaGraphError> {
        let mut qkv_guard = self.acquire_qkv_buffers(k, n_rows)?;
        let qkv = qkv_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("qkv buffers not allocated".into()))?;
        self.stream
            .memcpy_htod(&input[..k], &mut qkv.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload qkv_input: {e}")))?;
        unsafe {
            match Self::v8_shared_bytes(k) {
                Some(smem) => self.launch_gemv_v8(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                    smem,
                )?,
                None => self.launch_gemv_v7(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                )?,
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv stream sync: {e}")))?;
        self.stream
            .memcpy_dtoh(&qkv.d_output, &mut output[..n_rows])
            .map_err(|e| CudaGraphError::DriverError(format!("download qkv_output: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv D2H sync: {e}")))?;
        Ok(())
    }
}
