//! JSON persistence for the vector store and retriever.
//!
//! This module provides durable snapshots of an indexed corpus so callers
//! can save an index to disk and reload it without re-embedding every
//! document.  Two snapshot types are exposed:
//!
//! - [`IndexSnapshot`] captures a [`crate::vector_store::VectorStore`] — its
//!   dimensionality, distance metric, and every stored entry.
//! - [`RetrieverSnapshot`] wraps an [`IndexSnapshot`] together with the
//!   [`Retriever`]'s document counter so that `add_document` continues to
//!   produce monotonically increasing `doc_id`s after a round-trip.
//!
//! A monotonically-increasing [`SCHEMA_VERSION`] is stored in every
//! snapshot.  [`VectorStore::load_json`] and [`Retriever::load`] refuse to
//! deserialise unknown versions with [`RagError::Persistence`].
//!
//! # Embedder state
//!
//! The [`Retriever`] is generic over its [`Embedder`].  Because the trait
//! does not require `Serialize` we cannot persist embedder internals
//! generically — callers must reconstruct the embedder themselves and pass
//! it to [`Retriever::load`].  The `tfidf_state` field is an optional
//! escape hatch (`serde_json::Value`) that advanced users can populate by
//! hand if they wish to round-trip a TF-IDF vocabulary alongside the index.
//!
//! # Example
//!
//! ```no_run
//! use oxibonsai_rag::embedding::IdentityEmbedder;
//! use oxibonsai_rag::pipeline::RagConfig;
//! use oxibonsai_rag::retriever::{Retriever, RetrieverConfig};
//!
//! let embedder = IdentityEmbedder::new(32).expect("valid dim");
//! let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
//! retriever
//!     .add_document("some text", &RagConfig::default().chunk_config)
//!     .expect("index");
//!
//! let path = std::env::temp_dir().join("rag_snapshot.json");
//! retriever.save(&path).expect("save");
//!
//! let embedder = IdentityEmbedder::new(32).expect("valid dim");
//! let restored = Retriever::load(embedder, &path).expect("load");
//! assert_eq!(restored.chunk_count(), 1);
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::distance::Distance;
use crate::embedding::Embedder;
use crate::error::RagError;
use crate::retriever::Retriever;
use crate::vector_store::{VectorEntry, VectorStore};

// ─────────────────────────────────────────────────────────────────────────────
// Schema version
// ─────────────────────────────────────────────────────────────────────────────

/// Current on-disk snapshot schema version.
///
/// Bump this when the [`IndexSnapshot`] layout changes in a non-backwards-
/// compatible way.  Loaders reject unknown values with
/// [`RagError::Persistence`] so that a stale binary cannot silently
/// misinterpret a newer file.
pub const SCHEMA_VERSION: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// IndexSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Serde-serialisable snapshot of a [`VectorStore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Schema version tag (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Distance metric the store was configured with.
    #[serde(default)]
    pub distance: Distance,
    /// All stored entries, in insertion order.
    pub entries: Vec<VectorEntry>,
    /// Optional serialised TF-IDF state.  Advanced users may populate this
    /// by hand (see module-level docs).  Kept as an opaque
    /// [`serde_json::Value`] so that the store does not need to know the
    /// embedder's internal layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tfidf_state: Option<serde_json::Value>,
}

impl IndexSnapshot {
    /// Ensure the schema version matches the build-time constant.  Returns
    /// [`RagError::Persistence`] for unknown versions.
    pub fn check_version(&self) -> Result<(), RagError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(RagError::Persistence(format!(
                "unsupported schema_version {} (expected {})",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RetrieverSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Serde-serialisable snapshot of a [`Retriever`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieverSnapshot {
    /// Schema version tag (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Number of distinct documents indexed so far.
    pub doc_count: usize,
    /// The underlying vector-store snapshot.
    pub store: IndexSnapshot,
}

impl RetrieverSnapshot {
    fn check_version(&self) -> Result<(), RagError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(RagError::Persistence(format!(
                "unsupported schema_version {} (expected {})",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        self.store.check_version()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore <-> IndexSnapshot conversions
// ─────────────────────────────────────────────────────────────────────────────

impl VectorStore {
    /// Produce an [`IndexSnapshot`] capturing the current store contents.
    pub fn to_snapshot(&self) -> IndexSnapshot {
        IndexSnapshot {
            schema_version: SCHEMA_VERSION,
            dim: self.dim(),
            distance: self.distance(),
            entries: self.entries().to_vec(),
            tfidf_state: None,
        }
    }

    /// Build a [`VectorStore`] from a previously-produced snapshot.
    ///
    /// Returns [`RagError::Persistence`] if the schema version is unknown,
    /// and [`RagError::DimensionMismatch`] if any stored entry has a
    /// vector whose length disagrees with the snapshot's `dim`.
    pub fn from_snapshot(snapshot: IndexSnapshot) -> Result<Self, RagError> {
        snapshot.check_version()?;
        for entry in &snapshot.entries {
            if entry.vector.len() != snapshot.dim {
                return Err(RagError::DimensionMismatch {
                    expected: snapshot.dim,
                    got: entry.vector.len(),
                });
            }
        }
        let mut store = VectorStore::new_with_distance(snapshot.dim, snapshot.distance);
        store.set_entries(snapshot.entries);
        Ok(store)
    }

    /// Serialise this store to `path` as pretty-printed JSON.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.to_snapshot())
            .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
        Ok(())
    }

    /// Deserialise a store previously written by [`VectorStore::save_json`].
    ///
    /// Returns [`RagError::Persistence`] on malformed JSON or unknown
    /// schema version, and [`RagError::DimensionMismatch`] if any stored
    /// entry's vector length disagrees with the snapshot's `dim`.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, RagError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let snapshot: IndexSnapshot = serde_json::from_reader(reader)
            .map_err(|e| RagError::Persistence(format!("parse failed: {e}")))?;
        Self::from_snapshot(snapshot)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retriever persistence
// ─────────────────────────────────────────────────────────────────────────────

impl<E: Embedder> Retriever<E> {
    /// Serialise this retriever's index to `path` (pretty JSON).
    ///
    /// The embedder itself is *not* persisted — callers must provide an
    /// equivalent embedder to [`Retriever::load`].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let snapshot = RetrieverSnapshot {
            schema_version: SCHEMA_VERSION,
            doc_count: self.document_count(),
            store: self.store().to_snapshot(),
        };
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &snapshot)
            .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
        Ok(())
    }

    /// Reconstruct a [`Retriever`] from a previously-saved snapshot.
    ///
    /// `embedder` must produce vectors of the same dimensionality as the
    /// snapshot, otherwise [`RagError::DimensionMismatch`] is returned.
    pub fn load(embedder: E, path: impl AsRef<Path>) -> Result<Self, RagError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let snapshot: RetrieverSnapshot = serde_json::from_reader(reader)
            .map_err(|e| RagError::Persistence(format!("parse failed: {e}")))?;
        snapshot.check_version()?;

        if embedder.embedding_dim() != snapshot.store.dim {
            return Err(RagError::DimensionMismatch {
                expected: snapshot.store.dim,
                got: embedder.embedding_dim(),
            });
        }

        let store = VectorStore::from_snapshot(snapshot.store)?;
        Ok(Self::from_parts(
            embedder,
            store,
            snapshot.doc_count,
            crate::retriever::RetrieverConfig::default(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::Chunk;

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("oxibonsai_rag_persist_{tag}_{pid}_{nanos}.json"))
    }

    #[test]
    fn roundtrip_preserves_entries() {
        let mut store = VectorStore::new(3);
        let chunk = Chunk::new("hello".into(), 0, 0, 0);
        store.insert(vec![1.0, 0.0, 0.0], chunk).expect("insert");

        let path = tmp_path("roundtrip");
        store.save_json(&path).expect("save");
        let loaded = VectorStore::load_json(&path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.dim(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_version_rejected() {
        let snapshot = IndexSnapshot {
            schema_version: 9999,
            dim: 1,
            distance: Distance::Cosine,
            entries: Vec::new(),
            tfidf_state: None,
        };
        let result = VectorStore::from_snapshot(snapshot);
        assert!(matches!(result, Err(RagError::Persistence(_))));
    }
}
