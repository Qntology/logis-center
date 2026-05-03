//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `argmax_f32`: find the index of the maximum f32 value in `input`.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_argmax(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<u32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.argmax_f32)
            .arg(input)
            .arg(output)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("argmax launch: {e}")))
    }
    /// Upload `logits` to device, launch the argmax kernel, and return the index.
    ///
    /// Grid: `(1, 1, 1)`, Block: `(256, 1, 1)` — single-block reduction over 256 threads.
    pub fn encode_argmax(&self, logits: &[f32]) -> Result<u32, CudaGraphError> {
        let n = logits.len() as u32;
        let d_input = self
            .stream
            .clone_htod(logits)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod argmax input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros argmax output: {e}")))?;
        unsafe {
            self.launch_argmax(&d_input, &mut d_output, n)?;
        }
        let result = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh argmax result: {e}")))?;
        Ok(result[0])
    }
}
