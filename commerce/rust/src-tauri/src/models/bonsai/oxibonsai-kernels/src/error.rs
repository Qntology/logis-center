//! Error types for kernel operations.

use thiserror::Error;

/// Result type alias for kernel operations.
pub type KernelResult<T> = Result<T, KernelError>;

/// Errors that can occur during 1-bit kernel operations.
#[derive(Error, Debug)]
pub enum KernelError {
    /// Matrix/vector dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Output buffer is too small.
    #[error("output buffer too small: need {needed} elements, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// Number of elements is not a multiple of the block size.
    #[error("{count} elements is not divisible by block size {block_size}")]
    NotBlockAligned { count: usize, block_size: usize },

    /// Underlying core error.
    #[error("core error: {0}")]
    Core(#[from] oxibonsai_core::error::BonsaiError),

    /// Operation is not supported by this kernel tier.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// A GPU backend error propagated to the kernel layer.
    #[error("GPU error: {0}")]
    GpuError(String),
}

impl KernelError {
    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::DimensionMismatch { .. } => "DIMENSION_MISMATCH",
            Self::BufferTooSmall { .. } => "BUFFER_TOO_SMALL",
            Self::NotBlockAligned { .. } => "NOT_BLOCK_ALIGNED",
            Self::Core(_) => "CORE_ERROR",
            Self::UnsupportedOperation(_) => "UNSUPPORTED_OPERATION",
            Self::GpuError(_) => "GPU_ERROR",
        }
    }
}
