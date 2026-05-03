//! Bootstrap confidence intervals.
//!
//! Given a sample of per-example metric values, [`bootstrap_ci`] produces a
//! seed-deterministic confidence interval around the sample mean by
//! resampling with replacement `n_resamples` times and reporting the
//! percentile interval `[confidence_lo, confidence_hi]`.
//!
//! Uses a self-contained 64-bit xorshift PRNG (no external `rand` dependency)
//! so results are bit-reproducible across platforms for a given seed.

use serde::Serialize;

use crate::error::EvalError;

/// A bootstrap percentile confidence interval.
#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceInterval {
    /// Lower bound (confidence_lo percentile of resampled means).
    pub lo: f32,
    /// Upper bound (confidence_hi percentile of resampled means).
    pub hi: f32,
    /// Mean of the original sample.
    pub mean: f32,
    /// Confidence level used (e.g. 0.95 for 95% CI).
    pub confidence: f32,
    /// Number of resamples performed.
    pub n_resamples: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Xorshift64*
// ──────────────────────────────────────────────────────────────────────────────

/// Marsaglia xorshift64* step. Avoids a zero seed by mixing in a constant.
#[inline]
fn xorshift64(state: u64) -> u64 {
    // Ensure non-zero state — the transform hits a fixed point at 0.
    let mut x = if state == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        state
    };
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Draw one uniform `usize` in `[0, n)` from a 64-bit state (mutated).
#[inline]
fn next_index(state: &mut u64, n: usize) -> usize {
    *state = xorshift64(*state);
    if n == 0 {
        0
    } else {
        (*state as usize) % n
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Compute a bootstrap percentile confidence interval.
///
/// - `samples`: per-example metric values.
/// - `n_resamples`: number of resamples (≥ 1; 0 yields a zero-width CI at the mean).
/// - `confidence`: confidence level in `(0, 1)`, typically 0.95.
/// - `seed`: PRNG seed for reproducibility.
///
/// Returns `Err(EvalError::DatasetEmpty)` when `samples` is empty, or
/// `Err(EvalError::Numerical)` when `confidence` is out of range.
pub fn bootstrap_ci(
    samples: &[f32],
    n_resamples: usize,
    confidence: f32,
    seed: u64,
) -> Result<ConfidenceInterval, EvalError> {
    if samples.is_empty() {
        return Err(EvalError::DatasetEmpty);
    }
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(EvalError::Numerical(format!(
            "confidence must be in (0,1), got {}",
            confidence
        )));
    }

    let mean = mean_f32(samples);

    if n_resamples == 0 {
        return Ok(ConfidenceInterval {
            lo: mean,
            hi: mean,
            mean,
            confidence,
            n_resamples: 0,
        });
    }

    let mut state = if seed == 0 {
        0xDEAD_BEEF_CAFE_F00D
    } else {
        seed
    };
    // Avoid a degenerate all-zero xorshift state on subsequent calls.
    state = xorshift64(state);

    let n = samples.len();
    let mut resample_means: Vec<f32> = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let mut acc = 0.0f64;
        for _ in 0..n {
            let idx = next_index(&mut state, n);
            acc += samples[idx] as f64;
        }
        resample_means.push((acc / n as f64) as f32);
    }

    resample_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let lo = percentile(&resample_means, lo_q);
    let hi = percentile(&resample_means, hi_q);

    Ok(ConfidenceInterval {
        lo,
        hi,
        mean,
        confidence,
        n_resamples,
    })
}

fn mean_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        let s: f64 = xs.iter().map(|&v| v as f64).sum();
        (s / xs.len() as f64) as f32
    }
}

/// Percentile of a *sorted* slice via linear interpolation on rank.
fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let rank = q * (sorted.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f32;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}
