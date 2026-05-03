//! Integration tests for `oxibonsai_model::losses`.

use oxibonsai_model::losses::{
    contrastive_loss, cross_entropy, cross_entropy_grad, cross_entropy_single, distillation_loss,
    focal_loss, huber_loss, kl_divergence, label_smoothed_cross_entropy, log_softmax, mse,
    ntp_loss, softmax, LossError,
};

const EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ── 1. softmax sums to one ────────────────────────────────────────────────────

#[test]
fn softmax_sums_to_one() {
    let logits = vec![3.0_f32, 1.0, 0.2, -1.5, 4.0];
    let p = softmax(&logits);
    let sum: f32 = p.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-6), "softmax sum={sum}");
}

// ── 2. log-softmax produces non-positive values ───────────────────────────────

#[test]
fn log_softmax_all_non_positive() {
    let logits = vec![5.0_f32, -2.0, 0.0, 1.5];
    let lsm = log_softmax(&logits);
    for &v in &lsm {
        assert!(v <= EPS, "log_softmax value should be ≤ 0, got {v}");
    }
}

// ── 3. cross-entropy is ~0 for a perfect prediction ───────────────────────────

#[test]
fn cross_entropy_single_correct() {
    let mut logits = vec![-1e5_f32; 8];
    logits[5] = 1e5;
    let loss = cross_entropy_single(&logits, 5).expect("should succeed");
    assert!(
        loss < 1e-3,
        "near-perfect prediction should yield ~0 loss, got {loss}"
    );
}

// ── 4. batch cross-entropy is the mean of per-sample losses ───────────────────

#[test]
fn cross_entropy_batch_mean() {
    let vocab = 3_usize;
    let logits = vec![
        1.0_f32, 2.0, 0.5, // sample 0
        0.5_f32, 1.5, 3.0, // sample 1
        2.0_f32, 0.1, 0.1, // sample 2
    ];
    let targets = vec![1_usize, 2, 0];
    let batch_loss = cross_entropy(&logits, &targets, vocab).expect("ok");
    let l0 = cross_entropy_single(&logits[..3], 1).expect("ok");
    let l1 = cross_entropy_single(&logits[3..6], 2).expect("ok");
    let l2 = cross_entropy_single(&logits[6..], 0).expect("ok");
    let expected = (l0 + l1 + l2) / 3.0;
    assert!(
        approx_eq(batch_loss, expected, 1e-5),
        "batch CE={batch_loss} expected={expected}"
    );
}

// ── 5. target out of range returns error ──────────────────────────────────────

#[test]
fn cross_entropy_target_oob_error() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    match cross_entropy_single(&logits, 10) {
        Err(LossError::TargetOutOfRange {
            target: 10,
            vocab_size: 3,
        }) => {}
        other => panic!("expected TargetOutOfRange, got {other:?}"),
    }
}

// ── 6. label-smoothed CE differs from standard CE ─────────────────────────────

#[test]
fn label_smoothed_ce_differs_from_ce() {
    let vocab = 6_usize;
    let logits: Vec<f32> = vec![0.1, 0.2, 5.0, 0.1, 0.1, 0.1];
    let targets = vec![2_usize];
    let ce = cross_entropy(&logits, &targets, vocab).expect("ok");
    let ls = label_smoothed_cross_entropy(&logits, &targets, vocab, 0.1).expect("ok");
    assert!(
        (ce - ls).abs() > 1e-7,
        "smoothed CE must differ from CE: ce={ce}, ls={ls}"
    );
    assert!(ls > 0.0, "smoothed loss should be positive");
}

// ── 7. focal loss with γ=0 equals standard CE ─────────────────────────────────

#[test]
fn focal_loss_gamma_zero_equals_ce() {
    let vocab = 5_usize;
    let logits = vec![1.0_f32, 2.0, 3.0, 0.5, -0.5];
    let targets = vec![2_usize];
    let fl = focal_loss(&logits, &targets, vocab, 0.0, 1.0).expect("ok");
    let ce = cross_entropy(&logits, &targets, vocab).expect("ok");
    assert!(
        approx_eq(fl, ce, 1e-5),
        "focal(γ=0, α=1) should equal CE: fl={fl}, ce={ce}"
    );
}

// ── 8. KL(P||P) = 0 ───────────────────────────────────────────────────────────

#[test]
fn kl_divergence_identical_distributions() {
    let p = vec![0.1_f32, 0.5, 0.2, 0.2];
    let kl = kl_divergence(&p, &p).expect("ok");
    assert!(approx_eq(kl, 0.0, 1e-6), "KL(P||P) should be 0, got {kl}");
}

// ── 9. KL is asymmetric ───────────────────────────────────────────────────────

#[test]
fn kl_divergence_asymmetric() {
    // Use non-mirror distributions so KL(P||Q) ≠ KL(Q||P).
    let p = vec![0.8_f32, 0.15, 0.05];
    let q = vec![0.1_f32, 0.3, 0.6];
    let kl_pq = kl_divergence(&p, &q).expect("ok");
    let kl_qp = kl_divergence(&q, &p).expect("ok");
    assert!(
        (kl_pq - kl_qp).abs() > 0.01,
        "KL divergence should be asymmetric: KL(P||Q)={kl_pq}, KL(Q||P)={kl_qp}"
    );
}

// ── 10. distillation loss changes with temperature ────────────────────────────

#[test]
fn distillation_loss_temperature_effect() {
    let teacher = vec![10.0_f32, -10.0, 0.5];
    let student = vec![5.0_f32, -5.0, 1.0];
    let loss_t1 = distillation_loss(&teacher, &student, 1.0).expect("ok");
    let loss_t5 = distillation_loss(&teacher, &student, 5.0).expect("ok");
    assert!(loss_t1 > 0.0);
    assert!(loss_t5 > 0.0);
    assert!(
        (loss_t1 - loss_t5).abs() > 1e-5,
        "temperature should change loss: T=1 → {loss_t1}, T=5 → {loss_t5}"
    );
}

// ── 11. MSE of identical vectors is 0 ────────────────────────────────────────

#[test]
fn mse_zero_error() {
    let v = vec![0.5_f32, -1.0, 2.0, std::f32::consts::PI];
    let loss = mse(&v, &v).expect("ok");
    assert!(approx_eq(loss, 0.0, 1e-9), "MSE(x, x)=0, got {loss}");
}

// ── 12. Huber with tiny delta approaches L1 ───────────────────────────────────

#[test]
fn huber_loss_small_delta_is_l1() {
    // For |r| >> delta: L_δ ≈ delta * |r|.
    let predicted = vec![0.0_f32];
    let targets = vec![100.0_f32];
    let delta = 0.001_f32;
    let hl = huber_loss(&predicted, &targets, delta).expect("ok");
    let expected = delta * (100.0 - 0.5 * delta);
    assert!(
        approx_eq(hl, expected, 1e-5),
        "small-delta Huber ≈ linear: hl={hl}, expected={expected}"
    );
}

// ── 13. contrastive loss is 0 when positive equals anchor ─────────────────────

#[test]
fn contrastive_loss_identical_positive_zero() {
    let anchor = vec![1.0_f32, 0.0, 0.0];
    let positive = anchor.clone();
    let negative = vec![0.0_f32, 1.0, 0.0]; // orthogonal → max distance
                                            // d(anchor, positive) = 0, d(anchor, negative) = 1
                                            // loss = max(0, 0 - 1 + margin) with margin=0.5 → max(0, -0.5) = 0
    let loss = contrastive_loss(&anchor, &positive, &negative, 0.5).expect("ok");
    assert!(approx_eq(loss, 0.0, 1e-6), "loss should be 0, got {loss}");
}

// ── 14. ntp_loss returns per_token_losses of correct length ───────────────────

#[test]
fn ntp_loss_shape() {
    let vocab = 8_usize;
    let seq_len = 6_usize;
    let logits: Vec<f32> = (0..seq_len * vocab).map(|i| i as f32 * 0.1).collect();
    let targets: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let (mean, per_token) = ntp_loss(&logits, &targets, vocab, None).expect("ok");
    assert_eq!(
        per_token.len(),
        seq_len,
        "per_token length must equal seq_len"
    );
    assert!(mean > 0.0, "mean loss must be positive");
}

// ── 15. cross_entropy_grad sums to 0 ─────────────────────────────────────────

#[test]
fn cross_entropy_grad_sums_to_zero() {
    let logits = vec![1.0_f32, -0.5, 2.0, 0.3, 1.7];
    let grad = cross_entropy_grad(&logits, 2).expect("ok");
    let sum: f32 = grad.iter().sum();
    assert!(
        approx_eq(sum, 0.0, 1e-5),
        "CE grad must sum to 0 (softmax constraint), got {sum}"
    );
}

// ── 16. softmax is always non-negative ────────────────────────────────────────

#[test]
fn softmax_non_negative() {
    let logits = vec![-100.0_f32, 0.0, 100.0, -50.0, 25.0];
    let p = softmax(&logits);
    for &v in &p {
        assert!(v >= 0.0, "softmax values must be ≥ 0, got {v}");
    }
}

// ── 17. ntp_loss respects padding_id ─────────────────────────────────────────

#[test]
fn ntp_loss_padding_excluded() {
    let vocab = 4_usize;
    let seq_len = 4_usize;
    // Uniform logits so every non-pad token has the same loss.
    let logits: Vec<f32> = vec![1.0; seq_len * vocab];
    // Targets: two real tokens (0, 1) and two padding (3, 3).
    let targets: Vec<u32> = vec![0, 1, 3, 3];
    let padding_id = Some(3_u32);
    let (mean, per_token) = ntp_loss(&logits, &targets, vocab, padding_id).expect("ok");
    // Padded positions should be 0.
    assert!(
        approx_eq(per_token[2], 0.0, EPS),
        "padded token must have loss=0"
    );
    assert!(
        approx_eq(per_token[3], 0.0, EPS),
        "padded token must have loss=0"
    );
    // Mean should be over 2 real tokens only.
    let expected_mean = (per_token[0] + per_token[1]) / 2.0;
    assert!(
        approx_eq(mean, expected_mean, EPS),
        "mean={mean}, expected={expected_mean}"
    );
}

// ── 18. label smoothing with ε=0 equals standard CE ──────────────────────────

#[test]
fn label_smoothed_epsilon_zero_equals_ce() {
    let vocab = 4_usize;
    let logits = vec![2.0_f32, 1.0, 0.5, -1.0];
    let targets = vec![0_usize];
    let ce = cross_entropy(&logits, &targets, vocab).expect("ok");
    let ls = label_smoothed_cross_entropy(&logits, &targets, vocab, 0.0).expect("ok");
    assert!(
        approx_eq(ce, ls, 1e-5),
        "smoothing=0 should equal plain CE: ce={ce}, ls={ls}"
    );
}
