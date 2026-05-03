//! Embedding backends for the RAG pipeline.
//!
//! This module defines the [`Embedder`] trait and two built-in implementations:
//!
//! - [`IdentityEmbedder`] — deterministic byte-hash embedding for tests.
//! - [`TfIdfEmbedder`] — simple bag-of-words TF-IDF embedding with no external deps.

use std::collections::HashMap;

use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// Embedder trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait that all embedding backends must implement.
///
/// Implementations must be `Send + Sync` so they can be shared across threads.
pub trait Embedder: Send + Sync {
    /// Embed a text string, returning a dense `f32` vector.
    ///
    /// The returned vector always has exactly [`Embedder::embedding_dim`] elements.
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError>;

    /// The fixed number of dimensions produced by this embedder.
    fn embedding_dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// IdentityEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic, hash-based embedder intended for testing.
///
/// Converts a text string into a fixed-dimensional `f32` vector by iterating
/// over the bytes and accumulating them into bins using FNV-1a mixing.  The
/// output is L2-normalised so cosine similarity between identical texts is 1.0.
pub struct IdentityEmbedder {
    dim: usize,
}

impl IdentityEmbedder {
    /// Create an embedder that produces vectors of length `dim`.
    ///
    /// Returns [`RagError::DimensionMismatch`] if `dim` is zero.
    pub fn new(dim: usize) -> Result<Self, RagError> {
        if dim == 0 {
            return Err(RagError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self { dim })
    }

    /// Internal: hash bytes of `text` into a vector of `dim` floats.
    ///
    /// Uses a SplitMix64-inspired per-dimension mixing to produce a
    /// deterministic, well-distributed unit vector.  Each dimension `d` is
    /// seeded from the text bytes using an independent mixing step so that
    /// even very short texts produce a non-zero vector.
    fn hash_to_vec(&self, text: &str) -> Vec<f32> {
        let text_bytes = text.as_bytes();
        // Compute a 64-bit text fingerprint using FNV-1a
        let mut fingerprint: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for &byte in text_bytes {
            fingerprint ^= byte as u64;
            fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01B3);
        }
        // Also fold in the text length so that "" != " "
        fingerprint ^= text_bytes.len() as u64;
        fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01B3);

        // For each dimension, derive a float using SplitMix64 stepping
        (0..self.dim)
            .map(|d| {
                // Mix dimension index with the fingerprint
                let mut z =
                    fingerprint.wrapping_add((d as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                // SplitMix64 finaliser
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                // Map to a float in (-1, 1) with guaranteed non-zero magnitude.
                // Interpret as signed and scale to float range.
                let signed = z as i64;
                // Divide by half of i64::MAX to get a value in [-2, 2], then
                // clamp to (-1, 1) range to stay inside unit-ball territory.
                let f = (signed as f64 / (i64::MAX as f64)).clamp(-1.0, 1.0) as f32;
                // Ensure non-zero: if exactly zero (astronomically unlikely),
                // substitute a small constant derived from the dimension.
                if f == 0.0 {
                    ((d + 1) as f32) * 1e-7
                } else {
                    f
                }
            })
            .collect()
    }
}

impl Embedder for IdentityEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let mut v = self.hash_to_vec(text);
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TfIdfEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Simple bag-of-words TF-IDF embedder with no external dependencies.
///
/// Build the vocabulary from a corpus with [`TfIdfEmbedder::fit`], then embed
/// new texts with [`Embedder::embed`].  The output dimensionality equals the
/// vocabulary size (capped at `max_features`).
pub struct TfIdfEmbedder {
    /// term → column index in the TF-IDF vector
    vocab: HashMap<String, usize>,
    /// Inverse document frequency for each vocabulary term (indexed by column)
    idf: Vec<f32>,
    /// Fixed output dimension == vocab size
    dim: usize,
}

impl TfIdfEmbedder {
    /// Build vocabulary and IDF weights from a corpus.
    ///
    /// Tokenisation is whitespace + punctuation splitting, lowercased.
    /// Stop-words are not removed — the caller may pre-filter if desired.
    ///
    /// The `max_features` parameter caps the vocabulary size; the most frequent
    /// terms are retained.
    pub fn fit(documents: &[&str], max_features: usize) -> Self {
        let max_features = max_features.max(1);
        let n_docs = documents.len().max(1);

        // Count document frequency for each token
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in documents {
            let tokens = tokenize(doc);
            let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
            for tok in unique {
                *df.entry(tok).or_insert(0) += 1;
            }
        }

        // Sort by document frequency descending; take top max_features
        let mut df_vec: Vec<(String, usize)> = df.into_iter().collect();
        df_vec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        df_vec.truncate(max_features);

        let dim = df_vec.len();
        let mut vocab = HashMap::with_capacity(dim);
        let mut idf = vec![0.0f32; dim];

        for (idx, (term, doc_freq)) in df_vec.into_iter().enumerate() {
            vocab.insert(term, idx);
            // Smooth IDF: log((1 + n_docs) / (1 + df)) + 1
            idf[idx] = ((1.0 + n_docs as f32) / (1.0 + doc_freq as f32)).ln() + 1.0;
        }

        Self { vocab, idf, dim }
    }

    /// Compute a raw term-frequency vector for `text` (no IDF weighting).
    ///
    /// Useful for inspecting term counts without the IDF transform.
    pub fn embed_bow(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let n_tokens = tokens.len().max(1) as f32;
        let mut tf = vec![0.0f32; self.dim];
        for tok in &tokens {
            if let Some(&idx) = self.vocab.get(tok) {
                tf[idx] += 1.0;
            }
        }
        // Normalise by document length → term frequency
        for v in tf.iter_mut() {
            *v /= n_tokens;
        }
        tf
    }

    /// The vocabulary size (= output dimension).
    pub fn vocab_size(&self) -> usize {
        self.dim
    }

    /// Retrieve the vocabulary mapping (term → column index).
    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        if self.dim == 0 {
            return Err(RagError::EmbeddingFailed(
                "TfIdfEmbedder has an empty vocabulary".into(),
            ));
        }
        let mut tf = self.embed_bow(text);
        // Apply IDF weighting
        for (i, v) in tf.iter_mut().enumerate() {
            *v *= self.idf[i];
        }
        l2_normalize(&mut tf);
        Ok(tf)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise `v` in place.  If the norm is zero (or very small), the vector
/// is left unchanged to avoid producing NaN.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenise text into lowercase words, splitting on whitespace and common
/// punctuation.  Returns an empty `Vec` for empty input.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}
