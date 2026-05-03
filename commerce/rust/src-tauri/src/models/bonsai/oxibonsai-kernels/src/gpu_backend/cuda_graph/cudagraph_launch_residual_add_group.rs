//! # CudaGraph - launch_residual_add_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `residual_add` on the default stream.
    pub(crate) unsafe fn launch_residual_add(
        &self,
        d_a: &mut CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.residual_add)
            .arg(d_a)
            .arg(d_b)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("residual_add launch: {e}")))
    }
    /// Public wrapper around `launch_residual_add`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_residual_add_pub(
        &self,
        d_a: &mut CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_residual_add(d_a, d_b, n)
    }
}
