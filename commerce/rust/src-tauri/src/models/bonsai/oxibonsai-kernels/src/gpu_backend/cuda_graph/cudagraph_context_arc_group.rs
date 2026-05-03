//! # CudaGraph - context_arc_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaContext;
use std::sync::Arc;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Return the underlying `CudaContext` (shared reference).
    ///
    /// Used by `cuda_full_layer.rs` to compile the attention PTX module.
    pub fn context_arc(&self) -> &Arc<CudaContext> {
        &self.context
    }
}
