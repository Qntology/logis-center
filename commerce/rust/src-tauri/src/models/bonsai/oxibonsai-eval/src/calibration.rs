//! Calibration metrics — ECE, Brier score, NLL.
//!
//! ## Expected Calibration Error (ECE)
//!
//! ECE divides the predicted confidence range `[0, 1]` into `n_bins`
//! equal-width bins, then measures the weighted average gap between
//! average confidence and average accuracy in each bin:
//!
//! ```text
//! ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|
//! ```
//!
//! **Bin boundary convention:** `[lo, hi)` except the final bin which is
//! `[lo, 1.0]`. Implemented as
//! `bin = min(floor(p · n_bins), n_bins - 1)`.
//!
//! ## Brier score (multi-class, mean-squared-error form)
//!
//! ```text
//! Brier = (1/N) Σ_i Σ_c (p_{i,c} − y_{i,c})²
//! ```
//!
//! where `y_{i,c} ∈ {0,1}` is the one-hot label. For binary (C=2) this is
//! twice the MSE of the predicted probability of the positive class; we
//! return the multi-class form as defined above.
//!
//! ## Negative log-likelihood (NLL)
//!
//! NLL accepts raw logits for numerical stability:
//!
//! ```text
//! logsumexp(l) = max(l) + log(Σ exp(l − max(l)))
//! NLL_i        = logsumexp(l_i) − l_i[y_i]
//! NLL          = (1/N) Σ_i NLL_i
//! ```

use serde::Serialize;

use crate::error::EvalError;

/// Per-bin calibration statistics.
#[derive(Debug, Clone, Serialize)]
pub struct BinStat {
    /// Bin lower bound (inclusive).
    pub lo: f32,
    /// Bin upper bound (exclusive, except last bin which is inclusive).
    pub hi: f32,
    /// Number of samples in this bin.
    pub count: usize,
    /// Mean predicted confidence inside this bin.
    pub avg_confidence: f32,
    /// Empirical accuracy inside this bin.
    pub avg_accuracy: f32,
}

/// Combined calibration result (ECE + Brier + NLL).
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationResult {
    /// Expected Calibration Error in `[0, 1]`.
    pub ece: f32,
    /// Multi-class Brier score.
    pub brier: f32,
    /// Mean Negative Log-Likelihood (nats).
    pub nll: f32,
    /// Per-bin breakdown.
    pub bin_stats: Vec<BinStat>,
}

// ──────────────────────────────────────────────────────────────────────────────
// ECE
// ──────────────────────────────────────────────────────────────────────────────

/// Compute Expected Calibration Error.
///
/// - `confidences[i]` = predicted probability of the model's top choice.
/// - `correct[i]` = 1 if the top choice was correct, else 0.
/// - `n_bins` = number of bins (≥ 1). Values outside `[0, 1]` are clamped.
pub fn expected_calibration_error(
    confidences: &[f32],
    correct: &[u8],
    n_bins: usize,
) -> Result<(f32, Vec<BinStat>), EvalError> {
    let n_bins = n_bins.max(1);
    if confidences.len() != correct.len() {
        return Err(EvalError::MetricMismatch {
            expected: "equal-length confidences and correct arrays",
            got: format!("{} vs {}", confidences.len(), correct.len()),
        });
    }
    if confidences.is_empty() {
        return Ok((0.0, Vec::new()));
    }

    let total = confidences.len();
    let mut bin_count = vec![0usize; n_bins];
    let mut bin_conf_sum = vec![0.0f32; n_bins];
    let mut bin_acc_sum = vec![0.0f32; n_bins];

    for (i, &p_raw) in confidences.iter().enumerate() {
        let p = p_raw.clamp(0.0, 1.0);
        let mut bin = (p * n_bins as f32) as usize;
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        bin_count[bin] += 1;
        bin_conf_sum[bin] += p;
        bin_acc_sum[bin] += correct[i] as f32;
    }

    let mut ece = 0.0f32;
    let mut stats: Vec<BinStat> = Vec::with_capacity(n_bins);
    let step = 1.0f32 / n_bins as f32;
    for b in 0..n_bins {
        let cnt = bin_count[b];
        let lo = b as f32 * step;
        let hi = if b + 1 == n_bins {
            1.0
        } else {
            (b + 1) as f32 * step
        };
        let (avg_conf, avg_acc) = if cnt == 0 {
            (0.0, 0.0)
        } else {
            (bin_conf_sum[b] / cnt as f32, bin_acc_sum[b] / cnt as f32)
        };
        if cnt > 0 {
            ece += (cnt as f32 / total as f32) * (avg_acc - avg_conf).abs();
        }
        stats.push(BinStat {
            lo,
            hi,
            count: cnt,
            avg_confidence: avg_conf,
            avg_accuracy: avg_acc,
        });
    }
    Ok((ece.clamp(0.0, 1.0), stats))
}

// ──────────────────────────────────────────────────────────────────────────────
// Brier
// ──────────────────────────────────────────────────────────────────────────────

/// Multi-class Brier score.
///
/// `probs[i][c]` is the predicted probability of class `c` for sample `i`.
/// `labels[i]` is the true class index.
pub fn brier_score(probs: &[Vec<f32>], labels: &[usize]) -> Result<f32, EvalError> {
    if probs.len() != labels.len() {
        return Err(EvalError::MetricMismatch {
            expected: "equal-length probs and labels arrays",
            got: format!("{} vs {}", probs.len(), labels.len()),
        });
    }
    if probs.is_empty() {
        return Ok(0.0);
    }
    let c = probs[0].len();
    if c == 0 {
        return Err(EvalError::MetricMismatch {
            expected: "at least one class per sample",
            got: "0".to_string(),
        });
    }
    let mut acc = 0.0f64;
    for (p_vec, &y) in probs.iter().zip(labels.iter()) {
        if p_vec.len() != c {
            return Err(EvalError::MetricMismatch {
                expected: "uniform class dimension across samples",
                got: format!("class count changed to {}", p_vec.len()),
            });
        }
        if y >= c {
            return Err(EvalError::MetricMismatch {
                expected: "label < num_classes",
                got: format!("label={} but only {} classes", y, c),
            });
        }
        let mut s = 0.0f64;
        for (ci, &p) in p_vec.iter().enumerate() {
            let y_ic = if ci == y { 1.0f32 } else { 0.0f32 };
            let diff = (p - y_ic) as f64;
            s += diff * diff;
        }
        acc += s;
    }
    Ok((acc / probs.len() as f64) as f32)
}

// ──────────────────────────────────────────────────────────────────────────────
// NLL
// ──────────────────────────────────────────────────────────────────────────────

/// Negative log-likelihood from raw logits.
///
/// Applies a numerically-stable log-softmax per sample:
/// `logsumexp = max(l) + log(Σ exp(l − max))`, then returns
/// `(1/N) Σ (logsumexp(l_i) − l_i[y_i])`.
pub fn nll_from_logits(logits: &[Vec<f32>], labels: &[usize]) -> Result<f32, EvalError> {
    if logits.len() != labels.len() {
        return Err(EvalError::MetricMismatch {
            expected: "equal-length logits and labels arrays",
            got: format!("{} vs {}", logits.len(), labels.len()),
        });
    }
    if logits.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0f64;
    for (l, &y) in logits.iter().zip(labels.iter()) {
        if l.is_empty() {
            return Err(EvalError::Numerical("empty logit vector".to_string()));
        }
        if y >= l.len() {
            return Err(EvalError::MetricMismatch {
                expected: "label < num_logits",
                got: format!("label={} but only {} logits", y, l.len()),
            });
        }
        let max_l = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max_l.is_finite() {
            return Err(EvalError::Numerical(
                "non-finite max logit encountered".to_string(),
            ));
        }
        let sum_exp: f64 = l.iter().map(|&v| ((v - max_l) as f64).exp()).sum();
        if sum_exp <= 0.0 {
            return Err(EvalError::Numerical(
                "log-sum-exp produced non-positive sum".to_string(),
            ));
        }
        let lse = (max_l as f64) + sum_exp.ln();
        total += lse - l[y] as f64;
    }
    Ok((total / logits.len() as f64) as f32)
}

// ──────────────────────────────────────────────────────────────────────────────
// Combined
// ──────────────────────────────────────────────────────────────────────────────

/// Convenience: compute ECE + Brier + NLL in one pass from probabilities + logits.
///
/// - `probs[i][c]` predicted probability of class `c`.
/// - `logits[i][c]` raw logits (used only for NLL).
/// - `labels[i]` true class index.
/// - `n_bins` number of ECE bins.
pub fn calibration_all(
    probs: &[Vec<f32>],
    logits: &[Vec<f32>],
    labels: &[usize],
    n_bins: usize,
) -> Result<CalibrationResult, EvalError> {
    let confidences: Vec<f32> = probs
        .iter()
        .map(|p| p.iter().cloned().fold(0.0f32, f32::max))
        .collect();
    let correct: Vec<u8> = probs
        .iter()
        .zip(labels.iter())
        .map(|(p, &y)| {
            let (argmax, _) =
                p.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    });
            if argmax == y {
                1
            } else {
                0
            }
        })
        .collect();
    let (ece, bin_stats) = expected_calibration_error(&confidences, &correct, n_bins)?;
    let brier = brier_score(probs, labels)?;
    let nll = nll_from_logits(logits, labels)?;
    Ok(CalibrationResult {
        ece,
        brier,
        nll,
        bin_stats,
    })
}
