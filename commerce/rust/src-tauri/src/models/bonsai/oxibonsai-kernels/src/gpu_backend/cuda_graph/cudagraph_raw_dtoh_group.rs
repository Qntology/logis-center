//! # CudaGraph - raw_dtoh_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{result as cudarc_result, CudaSlice, DevicePtr};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Raw D2H copy via `cuMemcpyDtoHAsync`.
    ///
    /// # Safety
    /// Caller must synchronise the stream before reading `dst`.
    pub unsafe fn raw_dtoh<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut [T],
        count: usize,
    ) -> Result<(), CudaGraphError> {
        let (src_ptr, _rec) = src.device_ptr(&self.stream);
        cudarc_result::memcpy_dtoh_async(&mut dst[..count], src_ptr, self.stream.cu_stream())
            .map_err(|e| CudaGraphError::DriverError(format!("raw_dtoh: {e}")))
    }
}
