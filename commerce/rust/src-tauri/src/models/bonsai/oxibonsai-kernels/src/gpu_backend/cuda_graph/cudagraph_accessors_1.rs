//! # CudaGraph - accessors Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Upload `f32` weights and cache them under `key`.
    ///
    /// On the first call for `key`, the slice is copied to a device buffer and
    /// stored in `f32_weight_cache`.  Subsequent calls clone the cached `Arc`.
    ///
    /// Unlike [`get_or_upload_weight_soa`], no SoA reformatting is performed;
    /// the data is uploaded verbatim as typed `f32` device memory.
    pub fn get_or_upload_f32_weight(
        &self,
        key: u64,
        data: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
        {
            let cache = self
                .f32_weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&key) {
                return Ok(Arc::clone(existing));
            }
        }
        let d_buf = self
            .stream
            .clone_htod(data)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
        let arc = Arc::new(d_buf);
        let mut cache = self
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(key, Arc::clone(&arc));
        Ok(arc)
    }
}
