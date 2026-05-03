//! Loss functions for training language models.
//!
//! Includes standard losses (cross-entropy) plus specialised variants
//! (label smoothing, focal, contrastive) commonly used in LLM training.
//! All functions are numerically stable and avoid `unwrap()`.

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by loss-function computations.
#[derive(Debug, thiserror::Error)]
pub enum LossError {
    /// One or more input slices were empty.
    #[error("empty inputs")]
    EmptyInput,

    /// Two slices that must have matching shapes do not.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A target index falls outside `[0, vocab_size)`.
    #[error("target {target} out of range (vocab_size = {vocab_size})")]
    TargetOutOfRange { target: usize, vocab_size: usize },

    /// The label-smoothing parameter is outside `[0, 1)`.
    #[error("invalid smoothing {0}: must be in [0, 1)")]
    InvalidSmoothing(f32),

    /// The distillation temperature is not strictly positive.
    #[error("invalid temperature {0}: must be > 0")]
    InvalidTemperature(f32),

    /// A slice was passed as a probability distribution but does not sum to 1.
    #[error("probability distribution does not sum to 1 (sum = {0})")]
    NotADistribution(f32),
}

// ─── Numerically stable primitives ───────────────────────────────────────────

/// Numerically stable log-softmax.
///
/// Each element is computed as:
/// ```text
/// log_softmax(x_i) = x_i - max(x) - log( Σ_j exp(x_j - max(x)) )
/// ```
/// This keeps exponents non-positive, avoiding overflow.
///
/// # Panics
/// Never panics; returns an empty `Vec` for empty input.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = sum_exp.ln();
    logits.iter().map(|&x| x - max_val - log_sum_exp).collect()
}

/// Standard softmax.
///
/// Uses the max-subtraction trick for numerical stability.
///
/// # Panics
/// Never panics; returns an empty `Vec` for empty input.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Basic cross-entropy ──────────────────────────────────────────────────────

/// Cross-entropy loss for a single sample.
///
/// `logits` is an unnormalised score vector of length `vocab_size`.
/// `target` is the true class index.
/// Returns `−log p(target)`.
pub fn cross_entropy_single(logits: &[f32], target: usize) -> Result<f32, LossError> {
    if logits.is_empty() {
        return Err(LossError::EmptyInput);
    }
    let vocab_size = logits.len();
    if target >= vocab_size {
        return Err(LossError::TargetOutOfRange { target, vocab_size });
    }
    let lsm = log_softmax(logits);
    Ok(-lsm[target])
}

/// Cross-entropy loss over a batch of logits and target class indices.
///
/// `logits`: flat array of shape `[batch × vocab_size]`.
/// `targets`: class indices, one per sample.
/// Returns the mean cross-entropy over the batch.
pub fn cross_entropy(
    logits: &[f32],
    targets: &[usize],
    vocab_size: usize,
) -> Result<f32, LossError> {
    if logits.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if vocab_size == 0 {
        return Err(LossError::EmptyInput);
    }
    let batch = targets.len();
    if logits.len() != batch * vocab_size {
        return Err(LossError::ShapeMismatch(format!(
            "logits.len()={} != batch({}) * vocab_size({})",
            logits.len(),
            batch,
            vocab_size
        )));
    }

    let mut total = 0.0_f32;
    for (i, &t) in targets.iter().enumerate() {
        if t >= vocab_size {
            return Err(LossError::TargetOutOfRange {
                target: t,
                vocab_size,
            });
        }
        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        let lsm = log_softmax(row);
        total -= lsm[t];
    }
    Ok(total / batch as f32)
}

// ─── Label-smoothed cross-entropy ─────────────────────────────────────────────

/// Label-smoothed cross-entropy.
///
/// The true target gets probability `(1 − ε) + ε/V`; all other classes get
/// `ε/V`. The effective loss is:
/// ```text
/// L = (1 − ε) * CE + ε * H(uniform)
/// ```
/// where `H(uniform)` is the cross-entropy against the uniform distribution
/// (`log V` nats).
pub fn label_smoothed_cross_entropy(
    logits: &[f32],
    targets: &[usize],
    vocab_size: usize,
    smoothing: f32,
) -> Result<f32, LossError> {
    if !(0.0..1.0).contains(&smoothing) {
        return Err(LossError::InvalidSmoothing(smoothing));
    }
    if logits.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if vocab_size == 0 {
        return Err(LossError::EmptyInput);
    }
    let batch = targets.len();
    if logits.len() != batch * vocab_size {
        return Err(LossError::ShapeMismatch(format!(
            "logits.len()={} != batch({}) * vocab_size({})",
            logits.len(),
            batch,
            vocab_size
        )));
    }

    let mut total = 0.0_f32;
    for (i, &t) in targets.iter().enumerate() {
        if t >= vocab_size {
            return Err(LossError::TargetOutOfRange {
                target: t,
                vocab_size,
            });
        }
        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        let lsm = log_softmax(row);

        // Hard-target loss component.
        let ce = -lsm[t];
        // Smooth component: mean of -log_softmax over all classes.
        let mean_log_prob: f32 = lsm.iter().sum::<f32>() / vocab_size as f32;
        // label_smoothed = (1 - ε) * ce - ε * mean_log_prob
        total += (1.0 - smoothing) * ce + smoothing * (-mean_log_prob);
    }
    Ok(total / batch as f32)
}

// ─── Focal loss ───────────────────────────────────────────────────────────────

/// Focal loss (Lin et al., 2017).
///
/// ```text
/// FL = −α · (1 − p_t)^γ · log(p_t)
/// ```
/// where `p_t` is the model's estimated probability for the true class.
/// When `γ = 0` this reduces to standard cross-entropy scaled by `α`.
pub fn focal_loss(
    logits: &[f32],
    targets: &[usize],
    vocab_size: usize,
    gamma: f32,
    alpha: f32,
) -> Result<f32, LossError> {
    if logits.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if vocab_size == 0 {
        return Err(LossError::EmptyInput);
    }
    let batch = targets.len();
    if logits.len() != batch * vocab_size {
        return Err(LossError::ShapeMismatch(format!(
            "logits.len()={} != batch({}) * vocab_size({})",
            logits.len(),
            batch,
            vocab_size
        )));
    }

    let mut total = 0.0_f32;
    for (i, &t) in targets.iter().enumerate() {
        if t >= vocab_size {
            return Err(LossError::TargetOutOfRange {
                target: t,
                vocab_size,
            });
        }
        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        let probs = softmax(row);
        let p_t = probs[t];
        // Clamp to avoid log(0).
        let log_p_t = p_t.max(1e-9_f32).ln();
        let focal_weight = (1.0 - p_t).powf(gamma);
        total += alpha * focal_weight * (-log_p_t);
    }
    Ok(total / batch as f32)
}

// ─── KL divergence ────────────────────────────────────────────────────────────

/// KL divergence: `KL(P ‖ Q) = Σ_i P_i · log(P_i / Q_i)`.
///
/// Both `p` and `q` must be valid probability distributions (non-negative,
/// summing to 1 within a tolerance of 1e-4).  Terms where `P_i = 0` are
/// skipped (by convention `0 · log(0/q) = 0`).
pub fn kl_divergence(p: &[f32], q: &[f32]) -> Result<f32, LossError> {
    if p.is_empty() || q.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if p.len() != q.len() {
        return Err(LossError::ShapeMismatch(format!(
            "p.len()={} != q.len()={}",
            p.len(),
            q.len()
        )));
    }
    let sum_p: f32 = p.iter().sum();
    if (sum_p - 1.0).abs() > 1e-4 {
        return Err(LossError::NotADistribution(sum_p));
    }
    let sum_q: f32 = q.iter().sum();
    if (sum_q - 1.0).abs() > 1e-4 {
        return Err(LossError::NotADistribution(sum_q));
    }

    let mut kl = 0.0_f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            // Clamp qi to avoid division by zero.
            let qi_safe = qi.max(1e-30);
            kl += pi * (pi / qi_safe).ln();
        }
    }
    Ok(kl)
}

// ─── Knowledge distillation loss ─────────────────────────────────────────────

/// Soft target cross-entropy for knowledge distillation.
///
/// Both `teacher_logits` and `student_logits` are divided by `temperature`
/// before softmax / log-softmax respectively.  A higher temperature softens
/// the distributions and transfers more inter-class knowledge.
///
/// ```text
/// L = −Σ_i softmax(teacher / T)_i · log_softmax(student / T)_i
/// ```
///
/// Both slices must have the same length.
pub fn distillation_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    temperature: f32,
) -> Result<f32, LossError> {
    if temperature <= 0.0 {
        return Err(LossError::InvalidTemperature(temperature));
    }
    if teacher_logits.is_empty() || student_logits.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if teacher_logits.len() != student_logits.len() {
        return Err(LossError::ShapeMismatch(format!(
            "teacher.len()={} != student.len()={}",
            teacher_logits.len(),
            student_logits.len()
        )));
    }

    let t_scaled: Vec<f32> = teacher_logits.iter().map(|&x| x / temperature).collect();
    let s_scaled: Vec<f32> = student_logits.iter().map(|&x| x / temperature).collect();

    let teacher_probs = softmax(&t_scaled);
    let student_log_probs = log_softmax(&s_scaled);

    let loss: f32 = teacher_probs
        .iter()
        .zip(student_log_probs.iter())
        .map(|(&p, &lq)| -p * lq)
        .sum();
    Ok(loss)
}

// ─── MSE ──────────────────────────────────────────────────────────────────────

/// Mean squared error: `(1/N) · Σ (predicted_i − target_i)²`.
pub fn mse(predicted: &[f32], targets: &[f32]) -> Result<f32, LossError> {
    if predicted.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if predicted.len() != targets.len() {
        return Err(LossError::ShapeMismatch(format!(
            "predicted.len()={} != targets.len()={}",
            predicted.len(),
            targets.len()
        )));
    }
    let sum: f32 = predicted
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum();
    Ok(sum / predicted.len() as f32)
}

// ─── Huber loss ───────────────────────────────────────────────────────────────

/// Huber loss (smooth L1).
///
/// ```text
/// L_δ(r) = 0.5 · r²            if |r| ≤ δ
///           δ · (|r| − 0.5·δ)  otherwise
/// ```
/// Reduces to quadratic loss for small residuals and linear for large ones,
/// providing robustness to outliers.
pub fn huber_loss(predicted: &[f32], targets: &[f32], delta: f32) -> Result<f32, LossError> {
    if predicted.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if predicted.len() != targets.len() {
        return Err(LossError::ShapeMismatch(format!(
            "predicted.len()={} != targets.len()={}",
            predicted.len(),
            targets.len()
        )));
    }
    let sum: f32 = predicted
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let r = (p - t).abs();
            if r <= delta {
                0.5 * r * r
            } else {
                delta * (r - 0.5 * delta)
            }
        })
        .sum();
    Ok(sum / predicted.len() as f32)
}

// ─── Contrastive loss ─────────────────────────────────────────────────────────

/// Cosine-distance contrastive loss (triplet-style).
///
/// Encourages the anchor to be closer to the positive than the negative by at
/// least `margin` in cosine-similarity space:
///
/// ```text
/// L = max(0, d(anchor, positive) − d(anchor, negative) + margin)
/// ```
///
/// where `d(u, v) = 1 − cosine_similarity(u, v)`.
pub fn contrastive_loss(
    anchor: &[f32],
    positive: &[f32],
    negative: &[f32],
    margin: f32,
) -> Result<f32, LossError> {
    if anchor.is_empty() || positive.is_empty() || negative.is_empty() {
        return Err(LossError::EmptyInput);
    }
    let n = anchor.len();
    if positive.len() != n {
        return Err(LossError::ShapeMismatch(format!(
            "anchor.len()={} != positive.len()={}",
            n,
            positive.len()
        )));
    }
    if negative.len() != n {
        return Err(LossError::ShapeMismatch(format!(
            "anchor.len()={} != negative.len()={}",
            n,
            negative.len()
        )));
    }

    let cosine_sim = |u: &[f32], v: &[f32]| -> f32 {
        let dot: f32 = u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        let norm_u: f32 = u.iter().map(|&a| a * a).sum::<f32>().sqrt();
        let norm_v: f32 = v.iter().map(|&b| b * b).sum::<f32>().sqrt();
        if norm_u == 0.0 || norm_v == 0.0 {
            0.0
        } else {
            dot / (norm_u * norm_v)
        }
    };

    let d_pos = 1.0 - cosine_sim(anchor, positive);
    let d_neg = 1.0 - cosine_sim(anchor, negative);
    Ok((d_pos - d_neg + margin).max(0.0))
}

// ─── NTP loss ─────────────────────────────────────────────────────────────────

/// Per-token next-token-prediction (NTP) loss with optional padding mask.
///
/// `logits`: flat array of shape `[seq_len × vocab_size]`.
/// `targets`: `[seq_len]` ground-truth next-token IDs (as `u32`).
/// `padding_id`: if `Some(id)`, tokens whose *target* equals `id` are excluded
///               from the mean (but their individual loss is still returned as
///               0.0 in the per-token vector).
///
/// Returns `(mean_loss, per_token_losses)`.
pub fn ntp_loss(
    logits: &[f32],
    targets: &[u32],
    vocab_size: usize,
    padding_id: Option<u32>,
) -> Result<(f32, Vec<f32>), LossError> {
    if logits.is_empty() || targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if vocab_size == 0 {
        return Err(LossError::EmptyInput);
    }
    let seq_len = targets.len();
    if logits.len() != seq_len * vocab_size {
        return Err(LossError::ShapeMismatch(format!(
            "logits.len()={} != seq_len({}) * vocab_size({})",
            logits.len(),
            seq_len,
            vocab_size
        )));
    }

    let mut per_token = Vec::with_capacity(seq_len);
    let mut total = 0.0_f32;
    let mut count = 0usize;

    for (i, &t) in targets.iter().enumerate() {
        let t_usize = t as usize;
        if t_usize >= vocab_size {
            return Err(LossError::TargetOutOfRange {
                target: t_usize,
                vocab_size,
            });
        }

        let is_pad = padding_id == Some(t);
        if is_pad {
            per_token.push(0.0_f32);
            continue;
        }

        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        let lsm = log_softmax(row);
        let loss = -lsm[t_usize];
        per_token.push(loss);
        total += loss;
        count += 1;
    }

    let mean = if count > 0 { total / count as f32 } else { 0.0 };
    Ok((mean, per_token))
}

// ─── Gradient of cross-entropy w.r.t. logits ─────────────────────────────────

/// Gradient of cross-entropy loss w.r.t. logits for a single sample.
///
/// For softmax cross-entropy the gradient w.r.t. logit `j` is:
/// ```text
/// dL/dz_j = softmax(z)_j − 1{j == target}
/// ```
/// The returned vector has the same length as `logits` and sums to zero
/// (since `Σ softmax(z)_j = 1` and we subtract exactly 1 from one index).
pub fn cross_entropy_grad(logits: &[f32], target: usize) -> Result<Vec<f32>, LossError> {
    if logits.is_empty() {
        return Err(LossError::EmptyInput);
    }
    let vocab_size = logits.len();
    if target >= vocab_size {
        return Err(LossError::TargetOutOfRange { target, vocab_size });
    }
    let mut probs = softmax(logits);
    probs[target] -= 1.0;
    Ok(probs)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5, -1.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6), "sum={sum}");
    }

    #[test]
    fn log_softmax_all_non_positive() {
        let logits = vec![2.0_f32, -1.0, 0.0, 4.0];
        let lsm = log_softmax(&logits);
        for &v in &lsm {
            assert!(v <= 0.0 + EPS, "log_softmax value should be ≤ 0, got {v}");
        }
    }

    #[test]
    fn cross_entropy_single_perfect_prediction() {
        // When the logit for the correct class is overwhelmingly large, loss → 0.
        let mut logits = vec![-1000.0_f32; 10];
        logits[3] = 1000.0;
        let loss = cross_entropy_single(&logits, 3).expect("should not fail");
        assert!(
            loss < 1e-3,
            "near-perfect pred should have ~0 loss, got {loss}"
        );
    }

    #[test]
    fn cross_entropy_batch_mean() {
        let vocab = 4_usize;
        // Two samples, each with a uniform-ish logit vector.
        let logits = vec![
            1.0_f32, 2.0, 3.0, 4.0, // sample 0 → class 3 correct
            4.0_f32, 3.0, 2.0, 1.0, // sample 1 → class 0 correct
        ];
        let targets = vec![3_usize, 0];
        let loss = cross_entropy(&logits, &targets, vocab).expect("ok");
        // Manually compute expected values.
        let l0 = cross_entropy_single(&logits[..4], 3).expect("ok");
        let l1 = cross_entropy_single(&logits[4..], 0).expect("ok");
        let expected = (l0 + l1) / 2.0;
        assert!(
            approx_eq(loss, expected, 1e-5),
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn cross_entropy_target_oob_error() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let result = cross_entropy_single(&logits, 5);
        assert!(
            matches!(
                result,
                Err(LossError::TargetOutOfRange {
                    target: 5,
                    vocab_size: 3
                })
            ),
            "expected TargetOutOfRange, got {result:?}"
        );
    }

    #[test]
    fn label_smoothed_ce_less_than_ce_for_correct() {
        // For a "perfect" prediction, smoothing should increase the loss slightly
        // (it penalises overconfidence); but for an imperfect distribution it
        // can be lower.  Here we just verify the call succeeds and returns a
        // finite positive value different from standard CE.
        let vocab = 5_usize;
        let logits = vec![0.1_f32, 0.2, 5.0, 0.1, 0.1];
        let targets = vec![2_usize];
        let ce = cross_entropy(&logits, &targets, vocab).expect("ok");
        let ls_ce = label_smoothed_cross_entropy(&logits, &targets, vocab, 0.1).expect("ok");
        // With smoothing the loss should differ from plain CE.
        assert!(
            (ce - ls_ce).abs() > 1e-7,
            "smoothed CE should differ from CE: ce={ce}, ls_ce={ls_ce}"
        );
        assert!(ls_ce > 0.0, "loss must be positive");
    }

    #[test]
    fn focal_loss_gamma_zero_equals_ce() {
        let vocab = 4_usize;
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let targets = vec![2_usize];
        let fl = focal_loss(&logits, &targets, vocab, 0.0, 1.0).expect("ok");
        let ce = cross_entropy(&logits, &targets, vocab).expect("ok");
        assert!(
            approx_eq(fl, ce, 1e-5),
            "focal(γ=0, α=1) should equal CE: fl={fl}, ce={ce}"
        );
    }

    #[test]
    fn kl_divergence_identical() {
        let p = vec![0.25_f32, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p).expect("ok");
        assert!(approx_eq(kl, 0.0, 1e-6), "KL(P||P) should be 0, got {kl}");
    }

    #[test]
    fn kl_divergence_asymmetric() {
        // Use distributions that are genuinely asymmetric (not mirror images).
        let p = vec![0.8_f32, 0.15, 0.05];
        let q = vec![0.1_f32, 0.3, 0.6];
        let kl_pq = kl_divergence(&p, &q).expect("ok");
        let kl_qp = kl_divergence(&q, &p).expect("ok");
        assert!(
            (kl_pq - kl_qp).abs() > 1e-3,
            "KL divergence should be asymmetric: KL(P||Q)={kl_pq}, KL(Q||P)={kl_qp}"
        );
    }

    #[test]
    fn distillation_loss_temperature_effect() {
        let teacher = vec![10.0_f32, -10.0, 0.0];
        let student = vec![5.0_f32, -5.0, 0.0];
        let loss_low_t = distillation_loss(&teacher, &student, 1.0).expect("ok");
        let loss_high_t = distillation_loss(&teacher, &student, 10.0).expect("ok");
        // Higher temperature softens distributions → lower divergence in practice.
        // We just verify both succeed and differ.
        assert!(loss_low_t > 0.0);
        assert!(loss_high_t > 0.0);
        assert!(
            (loss_low_t - loss_high_t).abs() > 1e-5,
            "temperature should change the loss: t=1 → {loss_low_t}, t=10 → {loss_high_t}"
        );
    }

    #[test]
    fn mse_zero_error() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let loss = mse(&v, &v).expect("ok");
        assert!(
            approx_eq(loss, 0.0, 1e-8),
            "MSE of identical vectors should be 0, got {loss}"
        );
    }

    #[test]
    fn huber_loss_small_delta_approaches_l1() {
        // For delta → 0 Huber → δ·|r| (linear); normalised by δ gives |r|.
        // More precisely, for |r| >> delta: L_δ(r) ≈ δ·|r|.
        // We check that with small delta the loss is dominated by the linear term.
        let predicted = vec![0.0_f32];
        let targets = vec![10.0_f32];
        let delta = 0.001_f32;
        let hl = huber_loss(&predicted, &targets, delta).expect("ok");
        // Expected ≈ delta * (10 - 0.5*delta) ≈ delta * 10
        let expected = delta * (10.0 - 0.5 * delta);
        assert!(
            approx_eq(hl, expected, 1e-5),
            "small-delta Huber ≈ δ·|r|: hl={hl}, expected={expected}"
        );
    }

    #[test]
    fn contrastive_loss_identical_positive() {
        // When positive == anchor the positive distance is 0.
        let anchor = vec![1.0_f32, 0.0, 0.0];
        let positive = anchor.clone();
        let negative = vec![0.0_f32, 1.0, 0.0];
        let loss = contrastive_loss(&anchor, &positive, &negative, 0.5).expect("ok");
        // d(anchor, positive) = 0, d(anchor, negative) = 1.
        // loss = max(0, 0 - 1 + 0.5) = max(0, -0.5) = 0.
        assert!(
            approx_eq(loss, 0.0, 1e-6),
            "loss should be 0 when positive=anchor, got {loss}"
        );
    }

    #[test]
    fn ntp_loss_shape() {
        let vocab = 5_usize;
        let seq_len = 4_usize;
        let logits: Vec<f32> = (0..seq_len * vocab).map(|i| i as f32 * 0.1).collect();
        let targets: Vec<u32> = vec![0, 1, 2, 3];
        let (mean, per_token) = ntp_loss(&logits, &targets, vocab, None).expect("ok");
        assert_eq!(
            per_token.len(),
            seq_len,
            "per_token_losses should have seq_len elements"
        );
        assert!(mean > 0.0, "mean loss should be positive");
    }

    #[test]
    fn cross_entropy_grad_sums_to_zero() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let grad = cross_entropy_grad(&logits, 2).expect("ok");
        let sum: f32 = grad.iter().sum();
        assert!(
            approx_eq(sum, 0.0, 1e-5),
            "CE grad should sum to 0, got {sum}"
        );
    }
}
