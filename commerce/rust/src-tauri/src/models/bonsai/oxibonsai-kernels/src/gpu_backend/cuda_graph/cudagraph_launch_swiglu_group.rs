//! # CudaGraph - launch_swiglu_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `swiglu_fused` on the default stream.
    ///
    /// Kept as a fallback / building block; the hot path uses `launch_fused_gate_up_swiglu`
    /// which fuses the GEMV + SwiGLU steps into a single kernel dispatch.
    #[allow(dead_code)]
    unsafe fn launch_swiglu(
        &self,
        d_gate_up: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.swiglu_fused)
            .arg(d_gate_up)
            .arg(d_output)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("swiglu launch: {e}")))
    }
    /// Public wrapper around `launch_swiglu_fused`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_swiglu_pub(
        &self,
        d_gate_up: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_swiglu(d_gate_up, d_output, n)
    }
}
