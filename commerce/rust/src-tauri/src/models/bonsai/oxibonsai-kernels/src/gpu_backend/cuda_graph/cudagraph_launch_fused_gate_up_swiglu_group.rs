//! # CudaGraph - launch_fused_gate_up_swiglu_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `fused_gate_up_swiglu_q1`: fused gate+up Q1 GEMV with SwiGLU epilogue.
    ///
    /// Reads both the gate row and up row from the concatenated SoA weight matrix and
    /// writes `output[row] = SiLU(gate_dot) * up_dot` directly — no intermediate buffer.
    ///
    /// Grid: `(ceil(n_rows/8), 1, 1)`, Block: `(256, 1, 1)` (8 warps × 32 lanes).
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    pub(crate) unsafe fn launch_fused_gate_up_swiglu(
        &self,
        blocks: &CudaSlice<u8>,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
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
            .launch_builder(&self.modules.fused_gate_up_swiglu)
            .arg(blocks)
            .arg(input)
            .arg(output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("fused_gate_up_swiglu launch: {e}")))
    }
    /// Public wrapper around `launch_fused_gate_up_swiglu`.
    ///
    /// # Safety
    /// All slices must be valid device pointers allocated on the graph's stream.
    pub unsafe fn launch_fused_gate_up_swiglu_pub(
        &self,
        blocks: &Arc<CudaSlice<u8>>,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_fused_gate_up_swiglu(blocks, input, output, n_rows, k)
    }
}
