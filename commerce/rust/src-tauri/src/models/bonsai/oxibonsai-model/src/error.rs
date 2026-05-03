//! Error types for the model crate.

use thiserror::Error;

/// Result type alias for model operations.
pub type ModelResult<T> = Result<T, ModelError>;

/// Errors that can occur during model construction and forward pass.
#[derive(Error, Debug)]
pub enum ModelError {
    /// A required tensor was not found during model loading.
    #[error("missing tensor: {name}")]
    MissingTensor { name: String },

    /// Tensor shape doesn't match expected dimensions.
    #[error("shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Sequence length exceeds model's maximum context length.
    #[error("sequence length {seq_len} exceeds max context {max_ctx}")]
    SequenceTooLong { seq_len: usize, max_ctx: usize },

    /// Underlying core error.
    #[error("core: {0}")]
    Core(#[from] oxibonsai_core::error::BonsaiError),

    /// Underlying kernel error.
    #[error("kernel: {0}")]
    Kernel(#[from] oxibonsai_kernels::error::KernelError),

    /// Internal error (e.g. poisoned mutex).
    #[error("internal: {0}")]
    Internal(String),
}

impl ModelError {
    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::MissingTensor { .. } => "MISSING_TENSOR",
            Self::ShapeMismatch { .. } => "SHAPE_MISMATCH",
            Self::SequenceTooLong { .. } => "SEQUENCE_TOO_LONG",
            Self::Core(_) => "CORE_ERROR",
            Self::Kernel(_) => "KERNEL_ERROR",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }
}
