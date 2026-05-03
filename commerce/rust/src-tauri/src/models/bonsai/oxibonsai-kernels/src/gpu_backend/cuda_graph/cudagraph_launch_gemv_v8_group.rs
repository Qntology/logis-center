//! # CudaGraph - launch_gemv_v8_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemv_q1_g128_v8` (shared-memory input cache, k ≤ 48 KB threshold).
    ///
    /// `shared_mem_bytes` must be `(k/128) * 129 * 4`; caller computes via [`Self::v8_shared_bytes`].
    pub(crate) unsafe fn launch_gemv_v8(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
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
            .launch_builder(&self.modules.gemv_q1_g128_v8)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v8 launch: {e}")))
    }
    /// Public wrapper around `launch_gemv_v8`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_v8_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
        shared_mem_bytes: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_v8(d_weight, d_input, d_output, n_rows, k, shared_mem_bytes)
    }
}
