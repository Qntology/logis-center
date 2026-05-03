//! # CudaGraph - launch_rmsnorm_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `rmsnorm_weighted_v2` on the default stream.
    pub(crate) unsafe fn launch_rmsnorm(
        &self,
        d_input: &CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.rmsnorm_weighted_v2)
            .arg(d_input)
            .arg(d_weight)
            .arg(d_output)
            .arg(&n)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("rmsnorm launch: {e}")))
    }
    /// Public wrapper around `launch_rmsnorm_weighted_v2`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_rmsnorm_pub(
        &self,
        d_input: &CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        self.launch_rmsnorm(d_input, d_weight, d_output, n, eps)
    }
}
