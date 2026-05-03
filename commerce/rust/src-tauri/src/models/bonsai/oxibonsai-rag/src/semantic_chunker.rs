//! Semantic (embedding-driven) sentence grouping.
//!
//! The [`SemanticChunker`] splits a document into sentences and then groups
//! consecutive sentences whose embeddings are similar to the *running mean*
//! embedding of the current chunk.  Whenever a new sentence's cosine
//! similarity with the accumulated mean falls below a user-supplied
//! threshold, a new chunk is started.
//!
//! This is a lightweight alternative to neural topic segmentation — it
//! leans entirely on whatever notion of "similarity" the embedder exposes
//! through cosine similarity of the embedding vectors.
//!
//! # Algorithm
//!
//! 1. Split the input text into sentences with the same terminal-punctuation
//!    heuristic used by [`crate::chunker::chunk_by_sentences`].
//! 2. Seed the first chunk with the first sentence and its embedding.
//! 3. For each subsequent sentence:
//!    - embed it and L2-normalise the vector,
//!    - compute cosine similarity between the new embedding and the running
//!      mean embedding of the current chunk,
//!    - if similarity >= `similarity_threshold`: append the sentence to the
//!      current chunk and update the running mean,
//!    - otherwise: flush the current chunk and start a fresh one with this
//!      sentence.
//! 4. Stop when we run out of sentences.
//!
//! ## Running-mean update
//!
//! Given the current mean `m_k` over `k` sentences and a new embedding `e`,
//! the update is:
//!
//! ```text
//! m_{k+1} = m_k + (e - m_k) / (k + 1)
//! ```
//!
//! This is Welford's running-mean formula and is numerically stable for
//! long chunks.

use crate::chunker::Chunk;
use crate::embedding::{l2_normalize, Embedder};
use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// SemanticChunker
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding-driven sentence grouper.
pub struct SemanticChunker<'a, E: Embedder> {
    embedder: &'a E,
    similarity_threshold: f32,
    min_sentences_per_chunk: usize,
}

impl<'a, E: Embedder> SemanticChunker<'a, E> {
    /// Create a new semantic chunker that uses `embedder` to score sentence
    /// similarity and starts a new chunk whenever cosine similarity between
    /// a sentence and the running mean drops below `similarity_threshold`.
    ///
    /// Typical thresholds are in the range `[0.3, 0.8]`; too low produces
    /// one giant chunk, too high produces one sentence per chunk.
    pub fn new(embedder: &'a E, similarity_threshold: f32) -> Self {
        Self {
            embedder,
            similarity_threshold,
            min_sentences_per_chunk: 1,
        }
    }

    /// Require each produced chunk to contain at least `min` sentences
    /// (default `1`).  A chunk that would be flushed while holding fewer
    /// than `min` sentences will be extended anyway, overriding the
    /// similarity threshold.
    #[must_use]
    pub fn with_min_sentences(mut self, min: usize) -> Self {
        self.min_sentences_per_chunk = min.max(1);
        self
    }

    /// Run the chunker over `text` and return a `Vec<Chunk>`.
    ///
    /// The returned chunks are assigned the supplied `doc_id` and a
    /// monotonically-increasing `chunk_idx` starting at zero.  Each chunk's
    /// `char_offset` points to the byte offset of its first sentence in
    /// `text`.
    pub fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let sentences = split_sentences(text);
        if sentences.is_empty() {
            // Fallback: treat the whole document as one chunk.
            return Ok(vec![Chunk::new(text.trim().to_string(), doc_id, 0, 0)]);
        }
        if sentences.len() == 1 {
            let (start, s) = sentences[0];
            return Ok(vec![Chunk::new(s.to_string(), doc_id, 0, start)]);
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current_text = String::new();
        let mut current_start = sentences[0].0;
        let mut current_mean: Vec<f32> = self.embed_unit(sentences[0].1)?;
        let mut current_count: usize = 1;
        current_text.push_str(sentences[0].1);

        for (offset, sentence) in &sentences[1..] {
            let emb = self.embed_unit(sentence)?;
            let similarity = cosine_unit(&current_mean, &emb);
            let must_extend = current_count < self.min_sentences_per_chunk;

            if similarity >= self.similarity_threshold || must_extend {
                if !current_text.is_empty() {
                    current_text.push(' ');
                }
                current_text.push_str(sentence);
                current_count += 1;
                // Welford-style running-mean update
                for (m, e) in current_mean.iter_mut().zip(emb.iter()) {
                    *m += (*e - *m) / current_count as f32;
                }
            } else {
                chunks.push(Chunk::new(
                    std::mem::take(&mut current_text),
                    doc_id,
                    chunks.len(),
                    current_start,
                ));
                current_start = *offset;
                current_mean = emb;
                current_count = 1;
                current_text.push_str(sentence);
            }
        }

        if !current_text.is_empty() {
            chunks.push(Chunk::new(
                current_text,
                doc_id,
                chunks.len(),
                current_start,
            ));
        }

        Ok(chunks)
    }

    /// Embed a sentence and return an L2-normalised vector.
    fn embed_unit(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let mut v = self.embedder.embed(text)?;
        if v.iter().any(|x| !x.is_finite()) {
            return Err(RagError::NonFinite);
        }
        l2_normalize(&mut v);
        Ok(v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into sentences with their starting byte offset.
fn split_sentences(text: &str) -> Vec<(usize, &str)> {
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' || b == b'!' || b == b'?' {
            let mut j = i + 1;
            while j < bytes.len() && matches!(bytes[j], b'.' | b'!' | b'?') {
                j += 1;
            }
            while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | b'\n' | b'\r') {
                j += 1;
            }
            let sentence = text[start..j].trim();
            if !sentence.is_empty() {
                // Anchor the offset to the first non-whitespace byte of the sentence
                let anchor = start
                    + text[start..]
                        .char_indices()
                        .find(|(_, c)| !c.is_whitespace())
                        .map(|(b, _)| b)
                        .unwrap_or(0);
                out.push((anchor, sentence));
            }
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        let anchor = start
            + text[start..]
                .char_indices()
                .find(|(_, c)| !c.is_whitespace())
                .map(|(b, _)| b)
                .unwrap_or(0);
        out.push((anchor, tail));
    }
    out
}

/// Cosine similarity of two unit-normalised vectors, clamped to `[-1, 1]`.
fn cosine_unit(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    // Re-normalise the running mean to guard against drift
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::IdentityEmbedder;

    #[test]
    fn empty_input_yields_no_chunks() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let chunker = SemanticChunker::new(&emb, 0.5);
        let chunks = chunker.chunk("", 0).expect("chunk");
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_sentence_fallback() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let chunker = SemanticChunker::new(&emb, 0.5);
        let chunks = chunker.chunk("Only one sentence here", 0).expect("chunk");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn threshold_zero_merges_everything() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let chunker = SemanticChunker::new(&emb, -2.0);
        let text = "Alpha. Beta. Gamma.";
        let chunks = chunker.chunk(text, 0).expect("chunk");
        assert_eq!(chunks.len(), 1);
    }
}
