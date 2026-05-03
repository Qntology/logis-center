//! # CudaGraph - upload_weight_soa_new_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Upload raw SoA bytes directly (used by `NativeCudaBackend::upload_weights_raw`).
    pub fn upload_weight_soa_new(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<(), CudaGraphError> {
        let _ = self.get_or_upload_weight_soa(handle_id, aos_bytes)?;
        Ok(())
    }
}
