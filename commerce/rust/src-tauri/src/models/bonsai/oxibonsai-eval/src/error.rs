//! Error types for the evaluation harness.
//!
//! The error enum is marked `#[non_exhaustive]`: callers must include a
//! wildcard arm when matching so that future variant additions are a
//! non-breaking change.

use thiserror::Error;

/// Errors that can occur during model evaluation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvalError {
    /// The provided dataset contains no examples.
    #[error("dataset is empty")]
    DatasetEmpty,

    /// Input data has an unexpected or malformed format.
    #[error("invalid format: {0}")]
    InvalidFormat(String),

    /// The model inference step failed.
    #[error("inference failed: {0}")]
    InferenceFailed(String),

    /// An I/O error with a human-readable message (legacy string form).
    #[error("I/O error: {0}")]
    IoError(String),

    /// Parsing of a value (e.g. JSON field, integer) failed.
    #[error("parse error: {0}")]
    ParseError(String),

    /// A numerical invariant was violated (NaN, divide-by-zero, overflow).
    #[error("numerical error: {0}")]
    Numerical(String),

    /// An I/O error bubbled up from the standard library.
    ///
    /// Preferred over [`EvalError::IoError`] for automatic `?`-propagation
    /// of `std::io::Error` values.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The caller supplied data with the wrong metric shape or type.
    #[error("metric mismatch: expected {expected}, got {got}")]
    MetricMismatch {
        /// Human-readable description of what was expected.
        expected: &'static str,
        /// Human-readable description of what was actually provided.
        got: String,
    },
}
