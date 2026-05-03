//! # CudaGraph - global_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::cuda_kernels::CUDA_V7_KERNELS_SRC;
use cudarc::driver::{CudaContext, CudaFunction};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::debug;

use super::cudagraph_type::CudaGraph;
use super::functions::{compile_or_load_ptx, GLOBAL_CUDA_GRAPH};
use super::types::{CudaGraphError, CudaModules};

impl CudaGraph {
    /// Access the process-wide `CudaGraph` singleton, initialising on first call.
    ///
    /// Returns `Err` if no CUDA device is present or PTX compilation fails.
    pub fn global() -> Result<Arc<CudaGraph>, CudaGraphError> {
        let mutex = GLOBAL_CUDA_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let graph = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&graph));
        debug!("CudaGraph singleton initialised");
        Ok(graph)
    }
    /// Construct a new `CudaGraph` — heavy operation (device init + NVRTC compile).
    fn new() -> Result<Self, CudaGraphError> {
        let context =
            CudaContext::new(0).map_err(|e| CudaGraphError::DeviceNotFound(format!("{e}")))?;
        unsafe {
            context.disable_event_tracking();
        }
        let stream = context
            .new_stream()
            .map_err(|e| CudaGraphError::DriverError(format!("create stream: {e}")))?;
        let ptx = compile_or_load_ptx(CUDA_V7_KERNELS_SRC, "v7_kernels")?;
        let module = context
            .load_module(ptx)
            .map_err(|e| CudaGraphError::DriverError(format!("load_module: {e}")))?;
        let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            module
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        let modules = CudaModules {
            gemv_q1_g128_v7: load("gemv_q1_g128_v7")?,
            gemv_q1_g128_v7_residual: load("gemv_q1_g128_v7_residual")?,
            gemv_q1_g128_v8: load("gemv_q1_g128_v8")?,
            gemv_q1_g128_v8_residual: load("gemv_q1_g128_v8_residual")?,
            gemv_q1_g128_v9: load("gemv_q1_g128_v9")?,
            gemv_q1_g128_v9_residual: load("gemv_q1_g128_v9_residual")?,
            rmsnorm_weighted_v2: load("rmsnorm_weighted_v2")?,
            residual_add: load("residual_add")?,
            swiglu_fused: load("swiglu_fused")?,
            fused_gate_up_swiglu: load("fused_gate_up_swiglu_q1")?,
            argmax_f32: load("argmax_f32")?,
            gemv_tq2_g128_v1: load("gemv_tq2_g128_v1")?,
        };
        Ok(Self {
            context,
            stream,
            modules,
            buffers: Mutex::new(None),
            qkv_buffers: Mutex::new(None),
            weight_cache: Mutex::new(HashMap::new()),
            f32_weight_cache: Mutex::new(HashMap::new()),
            lm_head_buffers: Mutex::new(None),
            tq2_gemv_buffers: Mutex::new(None),
        })
    }
}
