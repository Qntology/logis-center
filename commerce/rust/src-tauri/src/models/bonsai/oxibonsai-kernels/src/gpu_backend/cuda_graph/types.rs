//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::{CpuBackend, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice};
use std::sync::Arc;

use super::cudagraph_type::CudaGraph;

/// Pre-allocated GPU buffers for the LM-head GEMV.
pub(crate) struct LmHeadBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) hidden_capacity: usize,
    pub(crate) vocab_capacity: usize,
}
impl LmHeadBuffers {
    pub(crate) fn fits(&self, hidden: usize, vocab: usize) -> bool {
        self.hidden_capacity >= hidden && self.vocab_capacity >= vocab
    }
}
/// Errors from the CUDA graph dispatch engine.
#[derive(Debug)]
pub enum CudaGraphError {
    /// No CUDA-capable GPU found or driver not present.
    DeviceNotFound(String),
    /// NVRTC PTX compilation failed.
    CompilationFailed(String),
    /// CUDA driver API error.
    DriverError(String),
    /// A requested weight handle was not found in the cache.
    WeightNotFound(u64),
    /// The internal weight layout conversion failed (malformed bytes).
    WeightLayoutError(String),
    /// A mutex was poisoned.
    LockPoisoned,
}
/// Pre-allocated GPU activation buffers for a single FFN forward pass.
///
/// `d_scratch` is reused for both the attn_proj GEMV output (size h) and the
/// down GEMV output (size h) — they are never needed simultaneously.  This
/// saves one GPU buffer vs. keeping separate `d_proj` and `d_down`.
///
/// Resized lazily when `hidden_size` or `intermediate_size` changes.
#[allow(dead_code)]
pub(crate) struct CudaActivationBuffers {
    pub(crate) d_hidden: CudaSlice<f32>,
    pub(crate) d_attn_out: CudaSlice<f32>,
    pub(crate) d_norm_weight: CudaSlice<f32>,
    pub(crate) d_scratch: CudaSlice<f32>,
    pub(crate) d_normed: CudaSlice<f32>,
    /// Intermediate gate+up GEMV output (2 × inter). Retained as a pre-allocated fallback
    /// buffer; not used in the fused `fused_gate_up_swiglu_q1` path.
    #[allow(dead_code)]
    pub(crate) d_gate_up: CudaSlice<f32>,
    pub(crate) d_swiglu: CudaSlice<f32>,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
}
impl CudaActivationBuffers {
    pub(crate) fn matches(&self, h: usize, inter: usize) -> bool {
        self.hidden_size == h && self.intermediate_size == inter
    }
}
/// Pre-allocated GPU buffers for the QKV projection.
///
/// Eliminates per-call `cuMemAlloc`/`cuMemFree` in `encode_qkv_phase`.
pub(crate) struct QkvBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) input_capacity: usize,
    pub(crate) output_capacity: usize,
}
impl QkvBuffers {
    pub(crate) fn fits(&self, input_len: usize, output_len: usize) -> bool {
        self.input_capacity >= input_len && self.output_capacity >= output_len
    }
}
/// Thin `GpuBackendTrait` wrapper around `CudaGraph`.
///
/// Returned by [`select_backend`](super::select_backend) when a CUDA device
/// is present and the `native-cuda` feature is enabled.
pub struct NativeCudaBackend {
    pub(super) graph: Arc<CudaGraph>,
    pub(super) cpu_fallback: CpuBackend,
}
impl NativeCudaBackend {
    /// Initialise the backend (may fail if no CUDA device).
    pub fn new() -> Result<Self, GpuError> {
        let graph = CudaGraph::global().map_err(|e| GpuError::NotAvailable(e.to_string()))?;
        Ok(Self {
            graph,
            cpu_fallback: CpuBackend::new(),
        })
    }
}
/// Handles to all compiled CUDA kernel functions used by `CudaGraph`.
#[allow(dead_code)]
pub(crate) struct CudaModules {
    pub(crate) gemv_q1_g128_v7: CudaFunction,
    pub(crate) gemv_q1_g128_v7_residual: CudaFunction,
    pub(crate) gemv_q1_g128_v8: CudaFunction,
    pub(crate) gemv_q1_g128_v8_residual: CudaFunction,
    pub(crate) gemv_q1_g128_v9: CudaFunction,
    pub(crate) gemv_q1_g128_v9_residual: CudaFunction,
    pub(crate) rmsnorm_weighted_v2: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    pub(crate) swiglu_fused: CudaFunction,
    /// Fused gate+up Q1 GEMV with SwiGLU epilogue — halves dispatch count for FFN step 5+6.
    pub(crate) fused_gate_up_swiglu: CudaFunction,
    pub(crate) argmax_f32: CudaFunction,
    /// Ternary (TQ2_0_g128) GEMV — SoA weight layout, 8 rows per CTA.
    pub(crate) gemv_tq2_g128_v1: CudaFunction,
}
/// Reusable input/output device buffers for `encode_gemv_tq2_cached`.
///
/// Grows monotonically to fit the largest GEMV seen so far. Eliminates the
/// per-call `cuMemAlloc`/`cuMemFree` round-trip that otherwise dominates
/// dispatch overhead for short kernels.
pub(crate) struct TernaryGemvBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) input_capacity: usize,
    pub(crate) output_capacity: usize,
}
impl TernaryGemvBuffers {
    pub(crate) fn fits(&self, input_len: usize, output_len: usize) -> bool {
        self.input_capacity >= input_len && self.output_capacity >= output_len
    }
}
