//! Error types for OxiBonsai core operations.

use thiserror::Error;

/// Result type alias for OxiBonsai core operations.
pub type BonsaiResult<T> = Result<T, BonsaiError>;

/// Errors that can occur during GGUF parsing, tensor loading, and configuration.
#[derive(Error, Debug)]
pub enum BonsaiError {
    /// Invalid GGUF magic number in file header.
    #[error("invalid GGUF magic number: expected 0x46554747, got 0x{magic:08X}")]
    InvalidMagic { magic: u32 },

    /// Unsupported GGUF format version.
    #[error("unsupported GGUF version: {version} (supported: 2, 3)")]
    UnsupportedVersion { version: u32 },

    /// Invalid or missing metadata entry.
    #[error("invalid metadata for key '{key}': {reason}")]
    InvalidMetadata { key: String, reason: String },

    /// A required tensor was not found in the model file.
    #[error("tensor not found: '{name}'")]
    TensorNotFound { name: String },

    /// Unsupported quantization type encountered.
    #[error("unsupported quantization type id {type_id}: known execution types are Q1_0_g128=41, TQ2_0_g128=42, TQ2_0=35")]
    UnsupportedQuantType { type_id: u32 },

    /// Memory mapping failed.
    #[error("memory mapping failed: {0}")]
    MmapError(#[from] std::io::Error),

    /// Unexpected end of file during parsing.
    #[error("unexpected end of file at offset {offset}")]
    UnexpectedEof { offset: u64 },

    /// Data alignment error.
    #[error("alignment error: expected {expected}-byte alignment at offset {offset}")]
    AlignmentError { expected: usize, offset: u64 },

    /// Invalid string encoding in GGUF data.
    #[error("invalid UTF-8 string at offset {offset}")]
    InvalidString { offset: u64 },

    /// A required configuration key is missing from model metadata.
    #[error("missing config key '{key}' in model metadata")]
    MissingConfigKey { key: String },

    /// Dimension mismatch between expected and actual tensor shape.
    #[error("tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },

    /// Block size validation failed.
    #[error("invalid Q1_0_g128 block: expected 18 bytes, got {actual}")]
    InvalidBlockSize { actual: usize },

    /// K-quant quantization or dequantization error.
    #[error("k-quant error: {reason}")]
    KQuantError { reason: String },
}

impl BonsaiError {
    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::InvalidMagic { .. } => "INVALID_MAGIC",
            Self::UnsupportedVersion { .. } => "UNSUPPORTED_VERSION",
            Self::InvalidMetadata { .. } => "INVALID_METADATA",
            Self::TensorNotFound { .. } => "TENSOR_NOT_FOUND",
            Self::UnsupportedQuantType { .. } => "UNSUPPORTED_QUANT_TYPE",
            Self::MmapError(_) => "MMAP_ERROR",
            Self::UnexpectedEof { .. } => "UNEXPECTED_EOF",
            Self::AlignmentError { .. } => "ALIGNMENT_ERROR",
            Self::InvalidString { .. } => "INVALID_STRING",
            Self::MissingConfigKey { .. } => "MISSING_CONFIG_KEY",
            Self::ShapeMismatch { .. } => "SHAPE_MISMATCH",
            Self::InvalidBlockSize { .. } => "INVALID_BLOCK_SIZE",
            Self::KQuantError { .. } => "K_QUANT_ERROR",
        }
    }

    /// Whether this error is potentially recoverable by retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::MmapError(_))
    }
}
