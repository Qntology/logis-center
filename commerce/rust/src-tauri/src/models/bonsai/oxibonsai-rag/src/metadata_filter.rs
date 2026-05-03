//! Metadata filtering for the vector store.
//!
//! A [`MetadataFilter`] is a boolean predicate over a `HashMap<String,
//! MetadataValue>` that is attached to each indexed [`crate::chunker::Chunk`]
//! via its optional `metadata` map.  The filter tree is serialisable so
//! that it round-trips through the persistence layer along with the index
//! itself.
//!
//! Filters compose algebraically: use [`MetadataFilter::All`] and
//! [`MetadataFilter::Any`] to build boolean trees over the primitive
//! leaves ([`MetadataFilter::Equals`], [`MetadataFilter::NotEquals`],
//! [`MetadataFilter::In`], [`MetadataFilter::Exists`]).
//!
//! Example:
//!
//! ```
//! use oxibonsai_rag::metadata_filter::{MetadataFilter, MetadataValue};
//! use std::collections::HashMap;
//!
//! let mut meta: HashMap<String, MetadataValue> = HashMap::new();
//! meta.insert("lang".into(), MetadataValue::from("rust"));
//! meta.insert("year".into(), MetadataValue::from(2026_i64));
//!
//! let filter = MetadataFilter::All(vec![
//!     MetadataFilter::Equals("lang".into(), MetadataValue::from("rust")),
//!     MetadataFilter::In(
//!         "year".into(),
//!         vec![MetadataValue::from(2025_i64), MetadataValue::from(2026_i64)],
//!     ),
//! ]);
//!
//! assert!(filter.matches(&meta));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// MetadataValue
// ─────────────────────────────────────────────────────────────────────────────

/// A primitive metadata value.
///
/// Kept intentionally small — the goal is filterable *tags* rather than
/// structured records.  For richer payloads users can store JSON strings
/// and parse them post-retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    /// String tag (the most common case).
    String(String),
    /// Signed 64-bit integer.
    Int(i64),
    /// 64-bit float — compared with exact bit equality so do not use for
    /// keys that are the result of arithmetic.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
}

impl MetadataValue {
    /// Structural equality used by filter matching.
    ///
    /// We deliberately use `==` which compares string content, integer
    /// value, exact `f64` bit pattern, and booleans.  `NaN` floats will
    /// therefore *never* match themselves — callers who need NaN handling
    /// should filter at insertion time.
    #[inline]
    pub fn matches_value(&self, other: &MetadataValue) -> bool {
        self == other
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}
impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}
impl From<i64> for MetadataValue {
    fn from(n: i64) -> Self {
        Self::Int(n)
    }
}
impl From<i32> for MetadataValue {
    fn from(n: i32) -> Self {
        Self::Int(n as i64)
    }
}
impl From<f64> for MetadataValue {
    fn from(n: f64) -> Self {
        Self::Float(n)
    }
}
impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataFilter
// ─────────────────────────────────────────────────────────────────────────────

/// A predicate over `HashMap<String, MetadataValue>` used to narrow vector
/// store results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// `metadata[key] == value`.
    Equals(String, MetadataValue),
    /// `metadata[key] != value` (missing keys do not match).
    NotEquals(String, MetadataValue),
    /// `metadata[key] ∈ {values}`.
    In(String, Vec<MetadataValue>),
    /// `key` is present in `metadata`.
    Exists(String),
    /// Logical AND of all children; an empty vector matches every
    /// metadata map.
    All(Vec<MetadataFilter>),
    /// Logical OR of all children; an empty vector matches nothing.
    Any(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Validate the filter structure.  Returns
    /// [`RagError::InvalidFilter`] if any leaf is malformed.
    pub fn validate(&self) -> Result<(), RagError> {
        match self {
            Self::Equals(k, _) | Self::NotEquals(k, _) | Self::Exists(k) => {
                if k.is_empty() {
                    return Err(RagError::InvalidFilter("empty key".into()));
                }
            }
            Self::In(k, values) => {
                if k.is_empty() {
                    return Err(RagError::InvalidFilter("empty key".into()));
                }
                if values.is_empty() {
                    return Err(RagError::InvalidFilter(
                        "`In` filter requires at least one value".into(),
                    ));
                }
            }
            Self::All(children) | Self::Any(children) => {
                for c in children {
                    c.validate()?;
                }
            }
        }
        Ok(())
    }

    /// Evaluate this filter against a metadata map.  Returns `true` if
    /// the entry passes the filter.
    pub fn matches(&self, metadata: &HashMap<String, MetadataValue>) -> bool {
        match self {
            Self::Equals(key, value) => metadata
                .get(key)
                .is_some_and(|actual| actual.matches_value(value)),
            Self::NotEquals(key, value) => metadata
                .get(key)
                .is_some_and(|actual| !actual.matches_value(value)),
            Self::In(key, values) => metadata
                .get(key)
                .is_some_and(|actual| values.iter().any(|v| actual.matches_value(v))),
            Self::Exists(key) => metadata.contains_key(key),
            Self::All(children) => children.iter().all(|c| c.matches(metadata)),
            Self::Any(children) => children.iter().any(|c| c.matches(metadata)),
        }
    }

    /// Convenience constructor for an equality check.
    pub fn eq(key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        Self::Equals(key.into(), value.into())
    }

    /// Convenience constructor for a not-equal check.
    pub fn neq(key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        Self::NotEquals(key.into(), value.into())
    }

    /// Convenience constructor for an existence check.
    pub fn exists(key: impl Into<String>) -> Self {
        Self::Exists(key.into())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> HashMap<String, MetadataValue> {
        let mut m = HashMap::new();
        m.insert("lang".into(), MetadataValue::from("rust"));
        m.insert("year".into(), MetadataValue::from(2026_i64));
        m.insert("stable".into(), MetadataValue::from(true));
        m
    }

    #[test]
    fn equals_matches() {
        let m = sample_metadata();
        assert!(MetadataFilter::eq("lang", "rust").matches(&m));
        assert!(!MetadataFilter::eq("lang", "python").matches(&m));
    }

    #[test]
    fn not_equals_matches() {
        let m = sample_metadata();
        assert!(MetadataFilter::neq("lang", "python").matches(&m));
        assert!(!MetadataFilter::neq("lang", "rust").matches(&m));
    }

    #[test]
    fn exists_matches() {
        let m = sample_metadata();
        assert!(MetadataFilter::exists("lang").matches(&m));
        assert!(!MetadataFilter::exists("missing").matches(&m));
    }

    #[test]
    fn all_any_compose() {
        let m = sample_metadata();
        let f = MetadataFilter::All(vec![
            MetadataFilter::eq("lang", "rust"),
            MetadataFilter::Any(vec![
                MetadataFilter::eq("year", 2025_i64),
                MetadataFilter::eq("year", 2026_i64),
            ]),
        ]);
        assert!(f.matches(&m));
    }

    #[test]
    fn validate_rejects_empty_key() {
        let f = MetadataFilter::Equals(String::new(), MetadataValue::from("x"));
        assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
    }

    #[test]
    fn validate_rejects_empty_in() {
        let f = MetadataFilter::In("k".into(), vec![]);
        assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
    }
}
