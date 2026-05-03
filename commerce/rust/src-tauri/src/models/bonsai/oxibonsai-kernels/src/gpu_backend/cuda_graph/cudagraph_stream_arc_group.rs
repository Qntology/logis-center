//! # CudaGraph - stream_arc_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaStream;
use std::sync::Arc;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Return the underlying `CudaStream` (shared reference).
    ///
    /// Used by `cuda_full_layer.rs` to perform uploads and synchronisation
    /// without going through the FFN-specific `encode_ffn_phase` API.
    pub fn stream_arc(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
