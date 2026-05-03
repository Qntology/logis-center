//! Streaming / online evaluation state machines.
//!
//! These mirror the batch evaluation paths but maintain running state so
//! callers can report a partial score mid-stream. The finalised value is
//! mathematically equivalent to the batch path (property-tested).

use serde::Serialize;

// ──────────────────────────────────────────────────────────────────────────────
// OnlinePerplexity
// ──────────────────────────────────────────────────────────────────────────────

/// Running perplexity estimator: accumulates `Σ log_p` and token count, and
/// reports `exp(-mean_neg_log_p)` on demand.
///
/// Feeding tokens one at a time yields exactly the same value as the batch
/// [`crate::perplexity::PerplexityEvaluator::compute`] call at the end, up to
/// `f32` accumulation order.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OnlinePerplexity {
    /// Sum of log-probabilities seen so far (natural log).
    sum_log_p: f64,
    /// Number of tokens observed.
    n: usize,
}

impl OnlinePerplexity {
    /// Construct a fresh estimator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a single log-probability (natural log).
    pub fn push(&mut self, log_p: f32) {
        self.sum_log_p += log_p as f64;
        self.n += 1;
    }

    /// Feed a chunk of log-probabilities.
    pub fn push_chunk(&mut self, log_ps: &[f32]) {
        for &l in log_ps {
            self.push(l);
        }
    }

    /// Reset the state (zero tokens, zero sum).
    pub fn reset(&mut self) {
        self.sum_log_p = 0.0;
        self.n = 0;
    }

    /// Number of tokens seen.
    pub fn tokens(&self) -> usize {
        self.n
    }

    /// Current perplexity estimate; `f32::INFINITY` if no tokens have been seen.
    pub fn current(&self) -> f32 {
        if self.n == 0 {
            f32::INFINITY
        } else {
            let mean_neg_log = -self.sum_log_p / self.n as f64;
            mean_neg_log.exp() as f32
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OnlineAccuracy
// ──────────────────────────────────────────────────────────────────────────────

/// Running accuracy counter.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OnlineAccuracy {
    /// Correct count.
    correct: usize,
    /// Total count.
    total: usize,
}

impl OnlineAccuracy {
    /// Construct a fresh counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single prediction outcome.
    pub fn push(&mut self, is_correct: bool) {
        if is_correct {
            self.correct += 1;
        }
        self.total += 1;
    }

    /// Record a batch of outcomes.
    pub fn push_many(&mut self, outcomes: &[bool]) {
        for &b in outcomes {
            self.push(b);
        }
    }

    /// Return the current accuracy in `[0, 1]`. Returns 0.0 for empty state.
    pub fn current(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    /// Return `(correct, total)` so callers can build an
    /// [`crate::accuracy::AccuracyResult`] later.
    pub fn counts(&self) -> (usize, usize) {
        (self.correct, self.total)
    }

    /// Reset the counter.
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}
