//! In-memory flat vector store with configurable distance metric.
//!
//! The [`VectorStore`] holds a flat list of [`VectorEntry`] items.  Search
//! is performed with a brute-force linear scan over all entries, evaluating
//! the configured [`Distance`] metric against the query vector.  This is
//! appropriate for corpora up to tens of thousands of chunks; larger
//! corpora benefit from approximate-nearest-neighbour indices (out of
//! scope for this crate).
//!
//! Scoring semantics are unified by [`Distance::to_score`]: similarity
//! metrics (Cosine, DotProduct) use their raw value as the score, whereas
//! true distances (Euclidean, Angular, Hamming) are negated so that
//! "higher is better" sorting always yields the closest match first.
//!
//! NaN / Inf guards: any non-finite value in an inserted vector or the
//! query vector is rejected with [`RagError::NonFinite`].

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;
use crate::distance::Distance;
use crate::error::RagError;
use crate::metadata_filter::MetadataFilter;

// ─────────────────────────────────────────────────────────────────────────────
// Math primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
///
/// Returns 0.0 if either slice is empty or they have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalise `v` in place.
///
/// If the Euclidean norm is smaller than `1e-10` the vector is left
/// unchanged to prevent NaN propagation.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Both vectors are assumed to be *unit vectors* (L2-normalised).  Under
/// that assumption, cosine similarity == dot product and the denominator
/// can be skipped.
///
/// Returns a value in `[-1.0, 1.0]`.  Returns `0.0` for empty or mismatched
/// inputs rather than panicking.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    dot_product(a, b).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorEntry & SearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single indexed entry in the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique identifier assigned at insertion time.
    pub id: usize,
    /// Stored embedding vector.  For similarity metrics this is
    /// L2-normalised; for distance metrics it is stored verbatim.
    pub vector: Vec<f32>,
    /// The chunk this entry was derived from.
    pub chunk: Chunk,
}

/// A result returned by a similarity / distance search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unified "higher is better" score (see [`Distance::to_score`]).  For
    /// similarity metrics this equals the raw similarity; for true
    /// distances it equals the negative of the raw distance.
    pub score: f32,
    /// The chunk associated with this result.
    pub chunk: Chunk,
    /// The entry's unique identifier in the store.
    pub id: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory flat vector store backed by a `Vec<VectorEntry>`.
///
/// The configured [`Distance`] controls both how vectors are *stored*
/// (similarity metrics pre-normalise; distance metrics store verbatim) and
/// how queries are scored.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VectorStore {
    entries: Vec<VectorEntry>,
    dim: usize,
    #[serde(default)]
    distance: Distance,
}

impl VectorStore {
    /// Create an empty cosine-similarity store for vectors of dim `dim`.
    pub fn new(dim: usize) -> Self {
        Self::new_with_distance(dim, Distance::default())
    }

    /// Create an empty store with a specific [`Distance`] metric.
    pub fn new_with_distance(dim: usize, distance: Distance) -> Self {
        Self {
            entries: Vec::new(),
            dim,
            distance,
        }
    }

    /// Insert a vector+chunk pair into the store.
    ///
    /// Behaviour depends on the store's [`Distance`]:
    ///
    /// - Similarity metrics (Cosine, DotProduct, Angular) L2-normalise the
    ///   stored vector up-front so that scoring is cheap.
    /// - True distance metrics (Euclidean, Hamming) preserve the vector
    ///   verbatim.
    ///
    /// Returns the assigned entry id.  Errors:
    ///
    /// - [`RagError::DimensionMismatch`] for wrong-size vectors.
    /// - [`RagError::NonFinite`] for `NaN` / `±∞` entries.
    pub fn insert(&mut self, mut vector: Vec<f32>, chunk: Chunk) -> Result<usize, RagError> {
        if vector.len() != self.dim {
            return Err(RagError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(RagError::NonFinite);
        }
        if matches!(
            self.distance,
            Distance::Cosine | Distance::DotProduct | Distance::Angular
        ) {
            l2_normalize(&mut vector);
        }
        let id = self.entries.len();
        self.entries.push(VectorEntry { id, vector, chunk });
        Ok(id)
    }

    /// Return the top-`top_k` entries by score.
    ///
    /// The query vector is normalised internally when the metric is a
    /// similarity; it is not mutated.  Results are returned in descending
    /// score order (see [`SearchResult::score`] for polarity).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        self.search_with_threshold(query, top_k, f32::NEG_INFINITY)
    }

    /// Like [`Self::search`] but discards results whose score is below
    /// `min_score`.
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        self.scored(query, top_k, min_score, None)
    }

    /// Search filtered by a [`MetadataFilter`].
    ///
    /// Filter evaluation is post-scoring; the metric is evaluated against
    /// every entry, then results that fail the filter are discarded.
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>, RagError> {
        filter.validate()?;
        Ok(self.scored(query, top_k, f32::NEG_INFINITY, Some(filter)))
    }

    fn scored(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
        filter: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        if self.entries.is_empty() || top_k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Vec::new();
        }

        // Prepare the query according to metric semantics.
        let prepared: Vec<f32> = if matches!(
            self.distance,
            Distance::Cosine | Distance::DotProduct | Distance::Angular
        ) {
            let mut q = query.to_vec();
            l2_normalize(&mut q);
            q
        } else {
            query.to_vec()
        };

        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if let Some(f) = filter {
                if !f.matches(&entry.chunk.metadata) {
                    continue;
                }
            }
            let raw = match self.distance.compute(&prepared, &entry.vector) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let score = self.distance.to_score(raw);
            if score >= min_score {
                scored.push((score, entry.id));
            }
        }

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(score, id)| SearchResult {
                score,
                chunk: self.entries[id].chunk.clone(),
                id,
            })
            .collect()
    }

    /// Number of entries currently in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the store (preserves the configured
    /// dimension and distance metric).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Approximate heap memory used by the stored vectors and chunk
    /// texts.  This is a lower-bound estimate: it counts vector bytes and
    /// chunk-text bytes but ignores allocator overhead and struct
    /// padding.
    pub fn memory_usage_bytes(&self) -> usize {
        self.entries.iter().fold(0usize, |acc, e| {
            acc + e.vector.len() * std::mem::size_of::<f32>()
                + e.chunk.text.len()
                + std::mem::size_of::<VectorEntry>()
        })
    }

    /// The embedding dimensionality this store was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The active distance metric.
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Borrow the internal entries (used by the persistence layer).
    pub(crate) fn entries(&self) -> &[VectorEntry] {
        &self.entries
    }

    /// Replace the internal entries (used by the persistence layer).
    pub(crate) fn set_entries(&mut self, entries: Vec<VectorEntry>) {
        self.entries = entries;
    }
}
