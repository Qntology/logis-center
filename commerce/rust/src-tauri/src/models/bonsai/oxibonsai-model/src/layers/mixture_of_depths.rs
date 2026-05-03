//! Mixture of Depths: adaptive per-token compute allocation.
//!
//! Some tokens are "routed through" a layer (full computation),
//! while others "skip" it (residual connection only).
//!
//! Reference: Raposo et al. 2024 "Mixture of Depths"

use thiserror::Error;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors from MoD operations.
#[derive(Debug, Error)]
pub enum ModError {
    #[error("hidden_dim mismatch: router expects {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("empty token sequence")]
    EmptySequence,
    #[error("capacity_factor {0} must be in (0, 1]")]
    InvalidCapacity(f32),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a MoD layer.
#[derive(Debug, Clone)]
pub struct ModConfig {
    /// Fraction of tokens to process (0.0–1.0). Others skip.
    pub capacity_factor: f32,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Whether to normalize router scores before top-k selection.
    pub normalize_router: bool,
}

impl ModConfig {
    /// Create a new `ModConfig` with the given capacity factor and hidden dim.
    /// Normalisation defaults to `false`.
    pub fn new(capacity_factor: f32, hidden_dim: usize) -> Self {
        Self {
            capacity_factor,
            hidden_dim,
            normalize_router: false,
        }
    }

    /// Enable or disable router score normalisation.
    pub fn with_normalize(mut self, norm: bool) -> Self {
        self.normalize_router = norm;
        self
    }
}

impl Default for ModConfig {
    fn default() -> Self {
        Self::new(0.5, 128)
    }
}

// ─── LCG RNG ─────────────────────────────────────────────────────────────────

/// Minimal 64-bit Linear Congruential Generator (no `rand` crate).
///
/// Parameters follow Knuth TAOCP Vol 2, Table 1.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance state and return next value.
    fn next_u64(&mut self) -> u64 {
        // Multiplier and increment from Knuth / glibc
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a value in `[0.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Router that scores each token's importance.
///
/// In production this would be a learned linear layer; here we use a
/// deterministic scoring function (dot product with fixed weight vector)
/// for a pure-Rust implementation with no external deps.
pub struct ModRouter {
    config: ModConfig,
    /// Router weights: [hidden_dim] → scalar importance score.
    weights: Vec<f32>,
}

impl ModRouter {
    /// Initialise with Xavier-style weights drawn from an LCG.
    pub fn new(config: ModConfig, seed: u64) -> Self {
        let hidden_dim = config.hidden_dim;
        let mut rng = Lcg64::new(seed);
        // Xavier uniform: range ±sqrt(1/hidden_dim)
        let scale = (1.0_f32 / hidden_dim as f32).sqrt();
        let weights: Vec<f32> = (0..hidden_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        Self { config, weights }
    }

    /// Score each token.
    ///
    /// `tokens` must have length `seq_len * hidden_dim` (row-major).
    /// Returns `seq_len` importance scores.
    pub fn score_tokens(&self, tokens: &[f32], seq_len: usize) -> Result<Vec<f32>, ModError> {
        if seq_len == 0 {
            return Err(ModError::EmptySequence);
        }
        let hd = self.config.hidden_dim;
        if tokens.len() != seq_len * hd {
            return Err(ModError::DimMismatch {
                expected: seq_len * hd,
                actual: tokens.len(),
            });
        }

        let mut scores: Vec<f32> = (0..seq_len)
            .map(|i| {
                let row = &tokens[i * hd..(i + 1) * hd];
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(x, w)| x * w)
                    .sum()
            })
            .collect();

        if self.config.normalize_router {
            // Softmax normalisation over the score vector.
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
            if sum_exp > 0.0 {
                for s in &mut scores {
                    *s = (*s - max_s).exp() / sum_exp;
                }
            }
        }

        Ok(scores)
    }

    /// Select the top-k token indices by score, where k = `capacity(seq_len)`.
    ///
    /// Returns indices sorted in ascending order (original token order).
    pub fn select_tokens(&self, scores: &[f32], seq_len: usize) -> Vec<usize> {
        let k = self.capacity(seq_len);
        if k == 0 || seq_len == 0 {
            return vec![];
        }
        // Build (score, index) pairs and partial-sort by descending score.
        let mut indexed: Vec<(f32, usize)> =
            scores.iter().enumerate().map(|(i, &s)| (s, i)).collect();

        // Partial sort: move top-k to the front (selection sort — O(n·k) but k ≤ n).
        for rank in 0..k {
            let mut best = rank;
            for j in (rank + 1)..indexed.len() {
                if indexed[j].0 > indexed[best].0 {
                    best = j;
                }
            }
            indexed.swap(rank, best);
        }

        let mut selected: Vec<usize> = indexed[..k].iter().map(|&(_, idx)| idx).collect();
        selected.sort_unstable();
        selected
    }

    /// Number of tokens that will be processed at `seq_len`.
    ///
    /// k = round(capacity_factor * seq_len), clamped to [1, seq_len].
    pub fn capacity(&self, seq_len: usize) -> usize {
        if seq_len == 0 {
            return 0;
        }
        let k = (self.config.capacity_factor * seq_len as f32).round() as usize;
        k.clamp(1, seq_len)
    }
}

// ─── Forward pass ─────────────────────────────────────────────────────────────

/// Apply a MoD-wrapped computation.
///
/// Selected tokens (by router) go through `layer_fn`; all others are passed
/// through unchanged via the residual connection.
///
/// # Arguments
/// * `hidden`     – flat `[seq_len * hidden_dim]` input buffer (row-major).
/// * `seq_len`    – number of tokens.
/// * `hidden_dim` – width of each token vector.
/// * `router`     – pre-built [`ModRouter`].
/// * `layer_fn`   – closure that receives `[selected_count * hidden_dim]` and
///   returns the same-shape processed tensor.
///
/// # Returns
/// A new `[seq_len * hidden_dim]` buffer with processed tokens substituted
/// back in their original positions.
pub fn mixture_of_depths_forward<F>(
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    router: &ModRouter,
    layer_fn: F,
) -> Result<Vec<f32>, ModError>
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    if seq_len == 0 {
        return Err(ModError::EmptySequence);
    }
    if router.config.capacity_factor <= 0.0 || router.config.capacity_factor > 1.0 {
        return Err(ModError::InvalidCapacity(router.config.capacity_factor));
    }
    if hidden.len() != seq_len * hidden_dim {
        return Err(ModError::DimMismatch {
            expected: seq_len * hidden_dim,
            actual: hidden.len(),
        });
    }

    // 1. Score all tokens.
    let scores = router.score_tokens(hidden, seq_len)?;

    // 2. Select token indices.
    let selected_indices = router.select_tokens(&scores, seq_len);
    let selected_count = selected_indices.len();

    // 3. Gather selected tokens into a contiguous buffer.
    let mut selected_buf: Vec<f32> = Vec::with_capacity(selected_count * hidden_dim);
    for &idx in &selected_indices {
        let row = &hidden[idx * hidden_dim..(idx + 1) * hidden_dim];
        selected_buf.extend_from_slice(row);
    }

    // 4. Apply layer function.
    let processed = layer_fn(&selected_buf, selected_count);

    // 5. Scatter results back; non-selected tokens keep the residual.
    let mut output = hidden.to_vec();
    for (rank, &idx) in selected_indices.iter().enumerate() {
        let src = &processed[rank * hidden_dim..(rank + 1) * hidden_dim];
        let dst = &mut output[idx * hidden_dim..(idx + 1) * hidden_dim];
        dst.copy_from_slice(src);
    }

    Ok(output)
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Statistics collected from one MoD forward pass.
#[derive(Debug, Clone)]
pub struct ModStats {
    /// Total tokens in the sequence.
    pub seq_len: usize,
    /// Number of tokens that went through the full layer.
    pub tokens_processed: usize,
    /// Number of tokens that skipped the layer.
    pub tokens_skipped: usize,
    /// `tokens_processed / capacity`, where capacity = `capacity_factor * seq_len`.
    pub capacity_utilization: f32,
    /// `1.0 - tokens_processed / seq_len`.
    pub compute_reduction: f32,
}

impl ModStats {
    /// Compute stats given `seq_len` and how many tokens were processed.
    pub fn compute(seq_len: usize, tokens_processed: usize) -> Self {
        let tokens_skipped = seq_len.saturating_sub(tokens_processed);
        let compute_reduction = if seq_len == 0 {
            0.0
        } else {
            1.0 - tokens_processed as f32 / seq_len as f32
        };
        // capacity_utilization: fraction of capacity actually used.
        // If tokens_processed == 0 and seq_len == 0, we avoid 0/0.
        let capacity_utilization = if tokens_processed == 0 {
            0.0
        } else {
            // By definition all selected tokens are "used", so utilization = 1.0
            // unless the layer_fn discards some (not tracked here).
            1.0_f32
        };
        Self {
            seq_len,
            tokens_processed,
            tokens_skipped,
            capacity_utilization,
            compute_reduction,
        }
    }

    /// Human-readable one-liner summary.
    pub fn summary(&self) -> String {
        format!(
            "MoD: seq={} processed={} skipped={} reduction={:.1}% utilization={:.1}%",
            self.seq_len,
            self.tokens_processed,
            self.tokens_skipped,
            self.compute_reduction * 100.0,
            self.capacity_utilization * 100.0,
        )
    }
}
