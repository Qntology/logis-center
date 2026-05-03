//! # CudaGraph - launch_gemv_v9_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemv_q1_g128_v9`: vectorized 128-bit weight loads + `__ldg()` scales.
    ///
    /// No shared memory required; identical grid/block to V7.
    /// Use this for large `k` where V8 shared-mem would exceed 48 KB.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    pub(crate) unsafe fn launch_gemv_v9(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v9)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v9 launch: {e}")))
    }
}
