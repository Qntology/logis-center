//! # CudaGraph - raw_htod_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{result as cudarc_result, CudaSlice, DevicePtrMut};

use super::cudagraph_type::CudaGraph;
use super::types::CudaGraphError;

impl CudaGraph {
    /// Raw H2D copy via `cuMemcpyHtoDAsync`.
    ///
    /// Event tracking is permanently disabled (see `CudaGraph::new`), so
    /// `device_ptr_mut` is a cheap pointer extraction with no injected waits.
    /// Using the raw driver API directly keeps the `unsafe` contract visible.
    ///
    /// # Safety
    /// - `src` must remain valid until the stream synchronises.
    /// - `dst` must be a valid device allocation on this graph's stream.
    /// - `count` must not exceed `dst.len()`.
    pub unsafe fn raw_htod<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &[T],
        dst: &mut CudaSlice<T>,
        count: usize,
    ) -> Result<(), CudaGraphError> {
        let (dst_ptr, _rec) = dst.device_ptr_mut(&self.stream);
        cudarc_result::memcpy_htod_async(dst_ptr, &src[..count], self.stream.cu_stream())
            .map_err(|e| CudaGraphError::DriverError(format!("raw_htod: {e}")))
    }
}
