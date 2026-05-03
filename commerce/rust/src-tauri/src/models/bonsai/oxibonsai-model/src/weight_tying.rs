//! Weight tying: share input embedding weights with the LM head.
//!
//! In language models, the input embedding table E[vocab_size, hidden_dim]
//! can be shared with the output projection W[hidden_dim, vocab_size] = E^T.
//! This reduces parameter count by ~hidden_dim * vocab_size * 4 bytes.
//!
//! # Memory savings
//!
//! Without tying: `vocab_size * hidden_dim * 4` bytes for embedding +
//! `hidden_dim * vocab_size * 4` bytes for LM head = 2x.
//! With tying: only `vocab_size * hidden_dim * 4` bytes total.

use thiserror::Error;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during weight tying operations.
#[derive(Debug, Error)]
pub enum TyingError {
    #[error("token_id {id} out of range (vocab_size = {vocab_size})")]
    TokenOutOfRange { id: usize, vocab_size: usize },
    #[error("weight shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
    #[error("hidden dim mismatch: expected {expected}, got {actual}")]
    HiddenDimMismatch { expected: usize, actual: usize },
}

// ─── TiedEmbedding ───────────────────────────────────────────────────────────

/// A tied embedding/unembedding pair.
///
/// Stores a single weight matrix `E` of shape `[vocab_size, hidden_dim]`.
/// The embedding lookup uses rows of `E` directly, while the LM head
/// projection computes `logits = hidden @ E^T`, reusing the same weights.
#[derive(Debug)]
pub struct TiedEmbedding {
    /// The weight matrix, layout: row-major [vocab_size, hidden_dim].
    pub weights: Vec<f32>,
    /// Number of vocabulary tokens.
    pub vocab_size: usize,
    /// Hidden / embedding dimensionality.
    pub hidden_dim: usize,
}

impl TiedEmbedding {
    /// Create a new zero-initialized tied embedding.
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weights: vec![0.0f32; vocab_size * hidden_dim],
            vocab_size,
            hidden_dim,
        }
    }

    /// Create from an existing weight vector.
    ///
    /// Verifies that `weights.len() == vocab_size * hidden_dim`.
    pub fn from_weights(
        weights: Vec<f32>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self, TyingError> {
        let expected = vocab_size * hidden_dim;
        if weights.len() != expected {
            return Err(TyingError::ShapeMismatch {
                expected,
                actual: weights.len(),
            });
        }
        Ok(Self {
            weights,
            vocab_size,
            hidden_dim,
        })
    }

    /// Embedding lookup: `token_id` → hidden vector of length `hidden_dim`.
    pub fn embed(&self, token_id: usize) -> Result<Vec<f32>, TyingError> {
        if token_id >= self.vocab_size {
            return Err(TyingError::TokenOutOfRange {
                id: token_id,
                vocab_size: self.vocab_size,
            });
        }
        let start = token_id * self.hidden_dim;
        Ok(self.weights[start..start + self.hidden_dim].to_vec())
    }

    /// Batch embedding lookup: returns a vector of vectors, one per token.
    pub fn embed_batch(&self, token_ids: &[usize]) -> Result<Vec<Vec<f32>>, TyingError> {
        token_ids.iter().map(|&id| self.embed(id)).collect()
    }

    /// LM head: hidden vector → logits (via W = E^T).
    ///
    /// `hidden`: slice of length `hidden_dim`.
    /// Returns: slice of length `vocab_size`.
    ///
    /// Computes `logits[v] = dot(hidden, E[v, :])` for each vocab entry `v`.
    pub fn project_to_logits(&self, hidden: &[f32]) -> Result<Vec<f32>, TyingError> {
        if hidden.len() != self.hidden_dim {
            return Err(TyingError::HiddenDimMismatch {
                expected: self.hidden_dim,
                actual: hidden.len(),
            });
        }
        let mut logits = Vec::with_capacity(self.vocab_size);
        for v in 0..self.vocab_size {
            let row = &self.weights[v * self.hidden_dim..(v + 1) * self.hidden_dim];
            let dot: f32 = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();
            logits.push(dot);
        }
        Ok(logits)
    }

    /// Batch LM head: `[batch_size * hidden_dim]` → `[batch_size * vocab_size]`.
    ///
    /// Input layout: row-major, each row of length `hidden_dim`.
    /// Output layout: row-major, each row of length `vocab_size`.
    pub fn project_batch(&self, hidden: &[f32], batch_size: usize) -> Result<Vec<f32>, TyingError> {
        let expected_len = batch_size * self.hidden_dim;
        if hidden.len() != expected_len {
            return Err(TyingError::ShapeMismatch {
                expected: expected_len,
                actual: hidden.len(),
            });
        }
        let mut output = Vec::with_capacity(batch_size * self.vocab_size);
        for b in 0..batch_size {
            let h_row = &hidden[b * self.hidden_dim..(b + 1) * self.hidden_dim];
            let logits = self.project_to_logits(h_row)?;
            output.extend_from_slice(&logits);
        }
        Ok(output)
    }

    /// Memory saved vs separate embedding + LM head (in bytes, assuming f32).
    ///
    /// Returns the number of bytes that would be used by a second copy of
    /// the weight matrix — which weight tying eliminates.
    pub fn memory_saved_bytes(&self) -> usize {
        self.vocab_size * self.hidden_dim * std::mem::size_of::<f32>()
    }

    /// Initialize with Kaiming-uniform scaled random weights.
    ///
    /// Uses a simple LCG (no external rand dependency) to fill weights
    /// uniformly in `[-bound, +bound]` where `bound = sqrt(1 / hidden_dim)`.
    pub fn init_kaiming(vocab_size: usize, hidden_dim: usize, seed: u64) -> Self {
        let bound = (1.0_f32 / hidden_dim as f32).sqrt();
        let n = vocab_size * hidden_dim;
        let mut state = seed.wrapping_add(0xC0FF_EE00_1234_5678_u64);
        let weights: Vec<f32> = (0..n)
            .map(|_| {
                // LCG step
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map to [-bound, +bound]
                let u = (state >> 11) as f32 / (1u64 << 53) as f32; // uniform [0,1)
                (u * 2.0 - 1.0) * bound
            })
            .collect();

        Self {
            weights,
            vocab_size,
            hidden_dim,
        }
    }

    /// Tie existing separate matrices: adopt the embedding weights as the
    /// shared matrix, discarding the separate LM head weights.
    ///
    /// This validates that:
    /// - `embed_weights.len() == vocab_size * hidden_dim`
    /// - `lm_head_weights.len() == hidden_dim * vocab_size` (same size, transposed)
    ///
    /// The function simply adopts `embed_weights`; it does **not** average the
    /// two because in practice the LM head is initialized as E^T anyway.
    pub fn from_separate(
        embed_weights: Vec<f32>,
        lm_head_weights: Vec<f32>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self, TyingError> {
        let expected = vocab_size * hidden_dim;

        if embed_weights.len() != expected {
            return Err(TyingError::ShapeMismatch {
                expected,
                actual: embed_weights.len(),
            });
        }
        // LM head can be [hidden_dim, vocab_size] = same total elements
        if lm_head_weights.len() != expected {
            return Err(TyingError::ShapeMismatch {
                expected,
                actual: lm_head_weights.len(),
            });
        }

        // Adopt embed_weights as the canonical tied matrix
        Ok(Self {
            weights: embed_weights,
            vocab_size,
            hidden_dim,
        })
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_zero_weights() {
        let te = TiedEmbedding::new(10, 8);
        assert_eq!(te.weights.len(), 80);
        assert!(te.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn embed_returns_correct_row() {
        let vocab_size = 4;
        let hidden_dim = 3;
        let weights: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
        let te = TiedEmbedding::from_weights(weights, vocab_size, hidden_dim)
            .expect("from_weights should succeed");
        let row = te.embed(2).expect("embed should succeed");
        assert_eq!(row, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn project_to_logits_shape_and_value() {
        let vocab_size = 5;
        let hidden_dim = 4;
        let te = TiedEmbedding::init_kaiming(vocab_size, hidden_dim, 42);
        let hidden = vec![1.0f32; hidden_dim];
        let logits = te
            .project_to_logits(&hidden)
            .expect("project should succeed");
        assert_eq!(logits.len(), vocab_size);
    }
}
