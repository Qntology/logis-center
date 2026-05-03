//! GPU weight cache — uploads model weights once, reuses across all GEMV/GEMM calls.
//!
//! The [`GpuWeightHandle`] is a lightweight, copyable identifier that references
//! a weight buffer resident on GPU memory. Weights are uploaded once via the
//! [`OneBitKernel::upload_weights`](crate::traits::OneBitKernel::upload_weights)
//! method and reused for all subsequent inference calls, eliminating costly
//! host→device copies on every token.

/// Opaque handle referencing a weight buffer resident on GPU memory.
///
/// Cheap to clone (just a `u64` ID). Works even without the `gpu` feature —
/// in that case it's simply a marker that the non-GPU kernel ignores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuWeightHandle(pub(crate) u64);

impl GpuWeightHandle {
    /// The raw numeric identifier (useful for logging/debugging).
    pub fn id(&self) -> u64 {
        self.0
    }
}
