//! Perplexity evaluator.
//!
//! Perplexity (PPL) measures how well a probability distribution predicted a
//! text sample. Lower is better.
//!
//! PPL = exp(−(1/N) · Σ log p(xᵢ | x<ᵢ))
//!
//! This module also provides bits-per-byte (BPB), an alternative metric
//! normalised by the number of UTF-8 bytes in the corpus.

use serde::Serialize;

// ──────────────────────────────────────────────────────────────────────────────
// PerplexityResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics from a batch perplexity evaluation.
#[derive(Debug, Serialize)]
pub struct PerplexityResult {
    /// Mean perplexity across all samples.
    pub mean_ppl: f32,
    /// Minimum perplexity across all samples.
    pub min_ppl: f32,
    /// Maximum perplexity across all samples.
    pub max_ppl: f32,
    /// Population standard deviation of perplexity values.
    pub std_ppl: f32,
    /// Number of samples evaluated.
    pub n_samples: usize,
    /// Total number of tokens processed.
    pub total_tokens: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// PerplexityEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator that computes perplexity from model log-probabilities.
pub struct PerplexityEvaluator {
    /// Sliding-window stride used when chunking long sequences (default: 512).
    pub stride: usize,
    /// Optional maximum sequence length to consider.
    pub max_length: Option<usize>,
}

impl Default for PerplexityEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexityEvaluator {
    /// Create a new evaluator with sensible defaults (stride = 512, no max length).
    pub fn new() -> Self {
        Self {
            stride: 512,
            max_length: None,
        }
    }

    /// Create an evaluator with the specified sliding-window stride.
    pub fn with_stride(stride: usize) -> Self {
        Self {
            stride,
            max_length: None,
        }
    }

    /// Compute perplexity for a single sequence of log-probabilities.
    ///
    /// Each element of `log_probs` is the natural log-probability of the token
    /// at that position given all preceding tokens.
    ///
    /// Returns `f32::INFINITY` when `log_probs` is empty (undefined PPL).
    pub fn compute(&self, log_probs: &[f32]) -> f32 {
        let probs = match self.max_length {
            Some(max) => &log_probs[..log_probs.len().min(max)],
            None => log_probs,
        };

        if probs.is_empty() {
            return f32::INFINITY;
        }

        let n = probs.len() as f32;
        let avg_neg_log_prob = -probs.iter().copied().sum::<f32>() / n;
        avg_neg_log_prob.exp()
    }

    /// Compute perplexity statistics for a batch of log-probability sequences.
    ///
    /// Each inner `Vec<f32>` corresponds to one sample.
    /// Empty sequences are silently skipped.
    pub fn compute_batch(&self, log_probs_batch: &[Vec<f32>]) -> PerplexityResult {
        let ppls: Vec<f32> = log_probs_batch
            .iter()
            .filter(|lp| !lp.is_empty())
            .map(|lp| self.compute(lp))
            .collect();

        let total_tokens: usize = log_probs_batch.iter().map(Vec::len).sum();

        if ppls.is_empty() {
            return PerplexityResult {
                mean_ppl: f32::INFINITY,
                min_ppl: f32::INFINITY,
                max_ppl: f32::INFINITY,
                std_ppl: 0.0,
                n_samples: 0,
                total_tokens,
            };
        }

        let n = ppls.len() as f32;
        let mean_ppl = ppls.iter().copied().sum::<f32>() / n;
        let min_ppl = ppls.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_ppl = ppls.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let variance = ppls.iter().map(|p| (p - mean_ppl).powi(2)).sum::<f32>() / n;
        let std_ppl = variance.sqrt();

        PerplexityResult {
            mean_ppl,
            min_ppl,
            max_ppl,
            std_ppl,
            n_samples: ppls.len(),
            total_tokens,
        }
    }

    /// Compute perplexity from raw logits and the ground-truth token IDs.
    ///
    /// `logits[i]` is the vocabulary-wide logit vector at position `i`.
    /// `token_ids[i]` is the token that was actually observed at position `i`.
    ///
    /// The function applies the log-softmax over each logit vector and selects
    /// the log-prob corresponding to the ground-truth token.
    ///
    /// Panics (via bounds check) if `token_ids[i]` is out of range for `logits[i]`.
    pub fn from_logits(&self, logits: &[Vec<f32>], token_ids: &[u32]) -> f32 {
        let len = logits.len().min(token_ids.len());
        if len == 0 {
            return f32::INFINITY;
        }

        let log_probs: Vec<f32> = logits[..len]
            .iter()
            .zip(token_ids[..len].iter())
            .map(|(logit_vec, &token_id)| {
                let max_logit = logit_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logit_vec.iter().map(|&l| (l - max_logit).exp()).sum();
                let log_sum_exp = max_logit + exp_sum.ln();
                let tid = token_id as usize;
                logit_vec[tid] - log_sum_exp
            })
            .collect();

        self.compute(&log_probs)
    }

    /// Compute bits-per-byte (BPB).
    ///
    /// BPB normalises perplexity by the number of bytes in the corpus:
    ///
    /// BPB = (−Σ log₂ p(xᵢ | x<ᵢ)) / n_bytes
    ///
    /// `log_probs` must be natural-log probabilities. Returns `f32::INFINITY`
    /// when `n_bytes == 0` or `log_probs` is empty.
    pub fn bits_per_byte(&self, log_probs: &[f32], n_bytes: usize) -> f32 {
        let probs = match self.max_length {
            Some(max) => &log_probs[..log_probs.len().min(max)],
            None => log_probs,
        };

        if probs.is_empty() || n_bytes == 0 {
            return f32::INFINITY;
        }

        // Convert nats to bits: log₂(x) = ln(x) / ln(2)
        let log2_e: f32 = std::f32::consts::E.log2();
        let neg_sum_log2_prob: f32 = probs.iter().map(|&lp| -lp * log2_e).sum();
        neg_sum_log2_prob / n_bytes as f32
    }
}
