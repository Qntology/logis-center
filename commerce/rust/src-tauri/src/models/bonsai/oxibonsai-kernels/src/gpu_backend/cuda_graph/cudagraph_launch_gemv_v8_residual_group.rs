//! # CudaGraph - launch_gemv_v8_residual_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemv_q1_g128_v8_residual`: V8 GEMV + fused residual add.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_gemv_v8_residual(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_residual: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
        shared_mem_bytes: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v8_residual)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_residual)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v8_residual launch: {e}")))
    }
}
