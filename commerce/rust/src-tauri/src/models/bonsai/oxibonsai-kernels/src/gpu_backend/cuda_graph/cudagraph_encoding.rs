//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::{CudaActivationBuffers, CudaGraphError};

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Ensure activation buffers are allocated for `(hidden_size, intermediate_size)`.
    /// Re-allocates if dimensions changed.
    fn acquire_buffers(
        &self,
        h: usize,
        inter: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<CudaActivationBuffers>>, CudaGraphError> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(h, inter),
            None => true,
        };
        if needs_alloc {
            let alloc_f32 = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros({n}): {e}")))
            };
            *guard = Some(CudaActivationBuffers {
                d_hidden: alloc_f32(h)?,
                d_attn_out: alloc_f32(h)?,
                d_norm_weight: alloc_f32(h)?,
                d_scratch: alloc_f32(h)?,
                d_normed: alloc_f32(h)?,
                d_gate_up: alloc_f32(2 * inter)?,
                d_swiglu: alloc_f32(inter)?,
                hidden_size: h,
                intermediate_size: inter,
            });
        }
        Ok(guard)
    }
    /// Execute the optimised FFN phase pipeline (6 kernel launches: fused gate+up+SwiGLU).
    ///
    /// Improvements over the V7 two-step (GEMV → SwiGLU) pipeline:
    /// - Steps 2 uses `gemv_q1_g128_v8` (shared-memory padded input cache) when
    ///   `k = hidden_size ≤ 48 KB threshold` → eliminates non-coalesced global reads.
    /// - Steps 5+6 are **fused** into `fused_gate_up_swiglu_q1` — reads gate and up
    ///   rows simultaneously and applies `SiLU(gate)*up` in the epilogue, halving the
    ///   dispatch count for this step vs. the old GEMV + swiglu_fused pair.
    /// - Hardware fp16 scale decode (`cvt.f32.f16`) in all kernels.
    /// - `d_scratch` reused for both attn_proj and down outputs → 1 fewer GPU buffer.
    ///
    /// | Step | Op                                                              |
    /// |------|-----------------------------------------------------------------|
    /// | 1    | Upload hidden, attn_out, norm_weight → device                   |
    /// | 2    | GEMV_v8(attn_proj, attn_out → scratch)                          |
    /// | 3    | residual_add(hidden += scratch)                                  |
    /// | 4    | rmsnorm(hidden, norm_weight → normed)                            |
    /// | 5    | fused_gate_up_swiglu_q1(gate_up, normed → swiglu_buf)           |
    /// | 6    | GEMV_v7/v8(down, swiglu → scratch)                              |
    /// | 7    | residual_add(hidden += scratch)                                  |
    /// | 8    | Download hidden → host (stream-synchronised)                     |
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        eps: f32,
        attn_proj_w: &Arc<CudaSlice<u8>>,
        gate_up_w: &Arc<CudaSlice<u8>>,
        down_w: &Arc<CudaSlice<u8>>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<(), CudaGraphError> {
        let h = hidden_size;
        let inter = intermediate_size;
        let h_u32 = h as u32;
        let i_u32 = inter as u32;
        let h_v8_smem = Self::v8_shared_bytes(h);
        let mut buf_guard = self.acquire_buffers(h, inter)?;
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("buffers not allocated".into()))?;
        self.stream
            .memcpy_htod(&hidden[..h], &mut bufs.d_hidden)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden: {e}")))?;
        self.stream
            .memcpy_htod(&attn_out[..h], &mut bufs.d_attn_out)
            .map_err(|e| CudaGraphError::DriverError(format!("upload attn_out: {e}")))?;
        self.stream
            .memcpy_htod(&norm_weight[..h], &mut bufs.d_norm_weight)
            .map_err(|e| CudaGraphError::DriverError(format!("upload norm_weight: {e}")))?;
        unsafe {
            match h_v8_smem {
                Some(smem) => self.launch_gemv_v8(
                    attn_proj_w,
                    &bufs.d_attn_out,
                    &mut bufs.d_scratch,
                    h_u32,
                    h_u32,
                    smem,
                )?,
                None => self.launch_gemv_v7(
                    attn_proj_w,
                    &bufs.d_attn_out,
                    &mut bufs.d_scratch,
                    h_u32,
                    h_u32,
                )?,
            }
            self.launch_residual_add(&mut bufs.d_hidden, &bufs.d_scratch, h_u32)?;
            self.launch_rmsnorm(
                &bufs.d_hidden,
                &bufs.d_norm_weight,
                &mut bufs.d_normed,
                h_u32,
                eps,
            )?;
            self.launch_fused_gate_up_swiglu(
                gate_up_w,
                &bufs.d_normed,
                &mut bufs.d_swiglu,
                i_u32,
                h_u32,
            )?;
            match Self::v8_shared_bytes(inter) {
                Some(smem) => self.launch_gemv_v8(
                    down_w,
                    &bufs.d_swiglu,
                    &mut bufs.d_scratch,
                    h_u32,
                    i_u32,
                    smem,
                )?,
                None => {
                    self.launch_gemv_v9(down_w, &bufs.d_swiglu, &mut bufs.d_scratch, h_u32, i_u32)?
                }
            }
            self.launch_residual_add(&mut bufs.d_hidden, &bufs.d_scratch, h_u32)?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync: {e}")))?;
        self.stream
            .memcpy_dtoh(&bufs.d_hidden, &mut hidden[..h])
            .map_err(|e| CudaGraphError::DriverError(format!("download hidden: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync D2H: {e}")))?;
        Ok(())
    }
}
