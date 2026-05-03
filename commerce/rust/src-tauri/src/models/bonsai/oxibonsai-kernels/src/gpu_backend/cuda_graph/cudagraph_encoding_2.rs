//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemv_q1_g128_v7` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    pub(crate) unsafe fn launch_gemv_v7(
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
            .launch_builder(&self.modules.gemv_q1_g128_v7)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v7 launch: {e}")))
    }
    /// Execute a single Q1 GEMV (`output = weight × input`) and return the result.
    pub fn encode_gemv(
        &self,
        weight_id: u64,
        weight_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = self.get_or_upload_weight_soa(weight_id, weight_bytes)?;
        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros output: {e}")))?;
        unsafe {
            self.launch_gemv_v7(&d_weight, &d_input, &mut d_output, n_rows as u32, k as u32)?;
        }
        self.stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh output: {e}")))
    }
    /// Execute a single GEMV using a pre-cached weight (handle already in cache).
    pub fn encode_gemv_cached(
        &self,
        weight_id: u64,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache
                .get(&weight_id)
                .map(Arc::clone)
                .ok_or(CudaGraphError::WeightNotFound(weight_id))?
        };
        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros output: {e}")))?;
        unsafe {
            self.launch_gemv_v7(&d_weight, &d_input, &mut d_output, n_rows as u32, k as u32)?;
        }
        self.stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh output: {e}")))
    }
    /// Public wrapper around `launch_gemv_v7`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_v7_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_v7(d_weight, d_input, d_output, n_rows, k)
    }
}
