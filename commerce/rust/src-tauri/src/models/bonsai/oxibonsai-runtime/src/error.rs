//! Runtime error types.

use oxibonsai_core::error::BonsaiError;
use oxibonsai_kernels::error::KernelError;
use oxibonsai_model::error::ModelError;

/// Runtime error type.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("core error: {0}")]
    Core(#[from] BonsaiError),

    #[error("kernel error: {0}")]
    Kernel(#[from] KernelError),

    #[error("model error: {0}")]
    Model(#[from] ModelError),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("generation stopped: {reason}")]
    GenerationStopped { reason: String },

    #[error("server error: {0}")]
    Server(String),

    #[error("GGUF file not found: {path}")]
    FileNotFound { path: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("operation timed out: {operation} after {duration_ms}ms")]
    Timeout {
        /// Description of the operation that timed out.
        operation: String,
        /// Duration in milliseconds before timeout.
        duration_ms: u64,
    },

    #[error("circuit breaker is open, requests are being rejected")]
    CircuitOpen,

    #[error("capacity exhausted: {resource}")]
    CapacityExhausted {
        /// Description of the exhausted resource (e.g. "kv_cache", "memory").
        resource: String,
    },

    #[error("batch error: {} sub-errors", .0.len())]
    BatchError(Vec<RuntimeError>),
}

impl RuntimeError {
    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::Core(_) => "CORE_ERROR",
            Self::Kernel(_) => "KERNEL_ERROR",
            Self::Model(_) => "MODEL_ERROR",
            Self::Tokenizer(_) => "TOKENIZER_ERROR",
            Self::GenerationStopped { .. } => "GENERATION_STOPPED",
            Self::Server(_) => "SERVER_ERROR",
            Self::FileNotFound { .. } => "FILE_NOT_FOUND",
            Self::Io(_) => "IO_ERROR",
            Self::Config(_) => "CONFIG_ERROR",
            Self::Timeout { .. } => "TIMEOUT",
            Self::CircuitOpen => "CIRCUIT_OPEN",
            Self::CapacityExhausted { .. } => "CAPACITY_EXHAUSTED",
            Self::BatchError(_) => "BATCH_ERROR",
        }
    }

    /// Whether this error is potentially recoverable by retrying.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Io(_) => true,
            Self::Timeout { .. } => true,
            Self::Server(_) => true,
            Self::CircuitOpen => true,
            Self::CapacityExhausted { .. } => true,
            Self::BatchError(errors) => errors.iter().any(|e| e.is_retryable()),
            _ => false,
        }
    }
}

/// Result type alias.
pub type RuntimeResult<T> = Result<T, RuntimeError>;
