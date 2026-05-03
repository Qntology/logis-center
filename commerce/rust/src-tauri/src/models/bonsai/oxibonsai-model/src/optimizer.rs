//! Optimizer implementations for LoRA fine-tuning.
//!
//! Provides SGD (with momentum and optional Nesterov), Adam, AdamW, and a
//! collection of learning-rate schedulers.  All optimizers operate on slices
//! of parameter vectors (`&mut [&mut Vec<f32>]`) alongside matching gradient
//! slices so that the caller owns the parameter storage.

// ─── Learning-rate schedulers ─────────────────────────────────────────────────

/// Trait for objects that compute a learning rate at a given training step.
pub trait LrScheduler {
    /// Return the learning rate to use at `step` (0-indexed).
    fn get_lr(&self, step: usize) -> f32;
}

/// Constant learning rate — returns the same value at every step.
pub struct ConstantLr {
    /// The fixed learning rate.
    pub lr: f32,
}

impl LrScheduler for ConstantLr {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Linear warmup then constant plateau.
///
/// During `[0, warmup_steps)` the learning rate increases linearly from
/// `0` to `base_lr`.  At and after `warmup_steps` it stays at `base_lr`.
pub struct WarmupConstantLr {
    /// Peak (and plateau) learning rate.
    pub base_lr: f32,
    /// Number of warmup steps.
    pub warmup_steps: usize,
}

impl LrScheduler for WarmupConstantLr {
    fn get_lr(&self, step: usize) -> f32 {
        if self.warmup_steps == 0 || step >= self.warmup_steps {
            self.base_lr
        } else {
            self.base_lr * (step as f32) / (self.warmup_steps as f32)
        }
    }
}

/// Cosine annealing with optional linear warmup.
///
/// During `[0, warmup_steps)`: linear ramp from `0` → `base_lr`.
/// During `[warmup_steps, total_steps]`: cosine decay from `base_lr` → `min_lr`.
/// After `total_steps`: clamped at `min_lr`.
///
/// The cosine formula is:
/// ```text
/// lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
/// ```
/// where `progress = (step - warmup_steps) / (total_steps - warmup_steps)`.
pub struct CosineAnnealingLr {
    /// Peak learning rate (start of cosine phase).
    pub base_lr: f32,
    /// Minimum (floor) learning rate.
    pub min_lr: f32,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Number of warmup steps (may be 0).
    pub warmup_steps: usize,
}

impl LrScheduler for CosineAnnealingLr {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup.
            if self.warmup_steps == 0 {
                return self.base_lr;
            }
            return self.base_lr * (step as f32) / (self.warmup_steps as f32);
        }

        let cosine_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if cosine_steps == 0 {
            return self.min_lr;
        }

        let elapsed = step.saturating_sub(self.warmup_steps);
        let progress = (elapsed as f32 / cosine_steps as f32).min(1.0);
        self.min_lr
            + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

// ─── SGD ─────────────────────────────────────────────────────────────────────

/// SGD with momentum, optional Nesterov update, and optional L2 weight decay.
pub struct Sgd {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (default 0.9).
    pub momentum: f32,
    /// L2 weight-decay coefficient (default 0.0).
    pub weight_decay: f32,
    /// Whether to use Nesterov momentum.
    pub nesterov: bool,
    /// Per-parameter velocity buffers.  Allocated lazily on the first step.
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    /// Create a new SGD optimizer with the given learning rate and sensible
    /// defaults (momentum = 0.9, no weight decay, no Nesterov).
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
            velocity: Vec::new(),
        }
    }

    /// Set the momentum coefficient (builder pattern).
    pub fn with_momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    /// Enable L2 weight decay (builder pattern).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable Nesterov momentum (builder pattern).
    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    /// Apply one SGD step.
    ///
    /// `params` and `grads` must have the same length; each `grads[i]` must
    /// have the same length as `params[i]`.
    pub fn step(&mut self, params: &mut [&mut Vec<f32>], grads: &[Vec<f32>]) {
        // Lazy-initialise velocity buffers.
        if self.velocity.len() != params.len() {
            self.velocity = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let v = &mut self.velocity[i];
            for (j, (p, &g)) in param.iter_mut().zip(grad.iter()).enumerate() {
                // L2 regularization adds wd * p to the gradient.
                let g_eff = g + self.weight_decay * (*p);

                if self.momentum == 0.0 {
                    *p -= self.lr * g_eff;
                } else {
                    v[j] = self.momentum * v[j] + g_eff;
                    if self.nesterov {
                        *p -= self.lr * (self.momentum * v[j] + g_eff);
                    } else {
                        *p -= self.lr * v[j];
                    }
                }
            }
        }
    }
}

// ─── Adam ─────────────────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2015).
///
/// Maintains first-moment (mean) and second-moment (uncentred variance)
/// estimates with bias correction.  Optional L2 weight decay adds `wd * p`
/// to the gradient before the moment updates.
pub struct Adam {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay for the first moment (default 0.9).
    pub beta1: f32,
    /// Exponential decay for the second moment (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant (default 1e-8).
    pub epsilon: f32,
    /// L2 weight-decay coefficient (default 0.0).
    pub weight_decay: f32,
    /// Number of steps taken so far (used for bias correction).
    step_count: usize,
    /// First-moment buffers.
    m: Vec<Vec<f32>>,
    /// Second-moment buffers.
    v: Vec<Vec<f32>>,
}

impl Adam {
    /// Create a new Adam optimizer with the given learning rate and defaults.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Override both beta values (builder pattern).
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Enable L2 weight decay (builder pattern).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Override epsilon (builder pattern).
    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Apply one Adam step.
    ///
    /// `params` and `grads` must have the same length; each `grads[i]` must
    /// have the same length as `params[i]`.
    pub fn step(&mut self, params: &mut [&mut Vec<f32>], grads: &[Vec<f32>]) {
        // Lazy-initialise moment buffers.
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
        }

        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let m_buf = &mut self.m[i];
            let v_buf = &mut self.v[i];
            for (j, (p, &g)) in param.iter_mut().zip(grad.iter()).enumerate() {
                let g_eff = g + self.weight_decay * (*p);

                m_buf[j] = self.beta1 * m_buf[j] + (1.0 - self.beta1) * g_eff;
                v_buf[j] = self.beta2 * v_buf[j] + (1.0 - self.beta2) * g_eff * g_eff;

                let m_hat = m_buf[j] / bc1;
                let v_hat = v_buf[j] / bc2;

                *p -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }

    /// Reset all state (moment buffers and step counter).
    pub fn reset(&mut self) {
        self.step_count = 0;
        for m in self.m.iter_mut() {
            m.iter_mut().for_each(|x| *x = 0.0);
        }
        for v in self.v.iter_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

// ─── AdamW ────────────────────────────────────────────────────────────────────

/// AdamW optimizer — Adam with *decoupled* weight decay (Loshchilov & Hutter,
/// 2019).
///
/// The difference from [`Adam`] with `weight_decay > 0` is that AdamW applies
/// the weight-decay *directly* to the parameters (`param *= 1 - lr * wd`)
/// **before** the gradient update, rather than adding `wd * param` to the
/// gradient.  This prevents weight decay from entering the moment estimates.
pub struct AdamW {
    inner: Adam,
}

impl AdamW {
    /// Create a new AdamW optimizer with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self {
            inner: Adam::new(lr),
        }
    }

    /// Override both beta values (builder pattern).
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner = self.inner.with_betas(b1, b2);
        self
    }

    /// Set decoupled weight-decay coefficient (builder pattern).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        // Store in epsilon temporarily; we'll read it back.
        // Actually store cleanly in weight_decay field.
        self.inner.weight_decay = wd;
        self
    }

    /// Apply one AdamW step.
    ///
    /// Weight decay is applied as `param *= (1 - lr * wd)` before the Adam
    /// gradient update (which is performed with `weight_decay = 0` to keep
    /// the moment estimates clean).
    pub fn step(&mut self, params: &mut [&mut Vec<f32>], grads: &[Vec<f32>]) {
        let wd = self.inner.weight_decay;
        let lr = self.inner.lr;

        // Decoupled weight decay: shrink each parameter first.
        if wd > 0.0 {
            let decay_factor = 1.0 - lr * wd;
            for param in params.iter_mut() {
                for p in param.iter_mut() {
                    *p *= decay_factor;
                }
            }
        }

        // Run Adam update with weight_decay=0 (already applied above).
        let saved_wd = self.inner.weight_decay;
        self.inner.weight_decay = 0.0;
        self.inner.step(params, grads);
        self.inner.weight_decay = saved_wd;
    }
}

// ─── Gradient utilities ───────────────────────────────────────────────────────

/// Compute the total L2 norm across all gradient tensors.
///
/// `||grads||_2 = sqrt(sum_i sum_j grads[i][j]^2)`
pub fn grad_norm(grads: &[Vec<f32>]) -> f32 {
    let sq_sum: f32 = grads.iter().flat_map(|g| g.iter()).map(|&x| x * x).sum();
    sq_sum.sqrt()
}

/// Clip all gradients so that their total L2 norm does not exceed `max_norm`.
///
/// Returns the gradient norm *before* clipping.  If the norm is already
/// ≤ `max_norm` the gradients are left unchanged.
pub fn clip_grad_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let norm = grad_norm(grads);
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= scale;
            }
        }
    }
    norm
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── LR Schedulers ────────────────────────────────────────────────────────

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLr { lr: 0.01 };
        assert!(approx_eq(sched.get_lr(0), 0.01));
        assert!(approx_eq(sched.get_lr(1000), 0.01));
    }

    #[test]
    fn test_warmup_lr_before_warmup() {
        let sched = WarmupConstantLr {
            base_lr: 1.0,
            warmup_steps: 100,
        };
        // At step 50 we should be at 50% of base_lr.
        assert!(approx_eq(sched.get_lr(50), 0.5));
        // Step 0 → 0.
        assert!(approx_eq(sched.get_lr(0), 0.0));
    }

    #[test]
    fn test_warmup_lr_after_warmup() {
        let sched = WarmupConstantLr {
            base_lr: 3e-4,
            warmup_steps: 100,
        };
        assert!(approx_eq(sched.get_lr(100), 3e-4));
        assert!(approx_eq(sched.get_lr(500), 3e-4));
    }

    #[test]
    fn test_cosine_annealing_at_zero() {
        let sched = CosineAnnealingLr {
            base_lr: 1.0,
            min_lr: 0.0,
            total_steps: 100,
            warmup_steps: 0,
        };
        // At step 0 with no warmup, lr should be base_lr.
        assert!(approx_eq(sched.get_lr(0), 1.0));
    }

    #[test]
    fn test_cosine_annealing_at_total() {
        let sched = CosineAnnealingLr {
            base_lr: 1.0,
            min_lr: 0.1,
            total_steps: 100,
            warmup_steps: 0,
        };
        // At the final step the lr should equal min_lr.
        let lr = sched.get_lr(100);
        assert!(
            (lr - 0.1).abs() < 1e-4,
            "expected min_lr=0.1 at total_steps, got {lr}"
        );
    }

    // ── SGD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_sgd_step_basic() {
        let mut p = vec![1.0f32, 2.0, 3.0];
        let g = vec![0.1f32, 0.2, 0.3];
        let mut sgd = Sgd::new(1.0).with_momentum(0.0);
        sgd.step(&mut [&mut p], &[g]);
        // With momentum=0 and no weight decay: p -= lr * g
        assert!(approx_eq(p[0], 0.9));
        assert!(approx_eq(p[1], 1.8));
        assert!(approx_eq(p[2], 2.7));
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut p = vec![1.0f32];
        let g = vec![1.0f32];
        let mut sgd = Sgd::new(0.1).with_momentum(0.9);
        // Step 1: v = 0.9*0 + 1.0 = 1.0; p -= 0.1 * 1.0 → 0.9
        sgd.step(&mut [&mut p], std::slice::from_ref(&g));
        assert!((p[0] - 0.9).abs() < 1e-5, "after step 1: {}", p[0]);
        // Step 2: v = 0.9*1.0 + 1.0 = 1.9; p -= 0.1 * 1.9 → 0.711
        sgd.step(&mut [&mut p], &[g]);
        assert!(
            (p[0] - (0.9 - 0.1 * 1.9)).abs() < 1e-5,
            "after step 2: {}",
            p[0]
        );
    }

    // ── Adam ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_adam_step_basic() {
        let mut p = vec![1.0f32];
        let g = vec![1.0f32];
        let mut adam = Adam::new(0.01);
        adam.step(&mut [&mut p], &[g]);
        // After one step the parameter must have decreased.
        assert!(
            p[0] < 1.0,
            "Adam must decrease parameter on positive gradient"
        );
    }

    #[test]
    fn test_adam_step_reduces_loss() {
        // Simulate minimising x^2: gradient = 2x.
        let mut p = vec![5.0f32];
        let mut adam = Adam::new(0.1);
        for _ in 0..200 {
            let grad = vec![2.0 * p[0]];
            adam.step(&mut [&mut p], &[grad]);
        }
        assert!(
            p[0].abs() < 0.5,
            "Adam should converge x^2 toward 0, got {}",
            p[0]
        );
    }

    #[test]
    fn test_adamw_step_basic() {
        let mut p = vec![1.0f32];
        let g = vec![0.0f32]; // zero gradient → only weight decay applies
        let mut adamw = AdamW::new(0.01).with_weight_decay(0.1);
        adamw.step(&mut [&mut p], &[g]);
        // With wd>0 and zero gradient the parameter must shrink.
        assert!(p[0] < 1.0, "AdamW must shrink parameter via weight decay");
    }

    // ── Gradient utilities ────────────────────────────────────────────────────

    #[test]
    fn test_clip_grad_norm_clips() {
        let mut grads = vec![vec![3.0f32, 4.0]]; // norm = 5.0
        let norm_before = clip_grad_norm(&mut grads, 1.0);
        assert!(approx_eq(norm_before, 5.0));
        // After clipping, norm should be ~1.0.
        let norm_after = grad_norm(&grads);
        assert!(
            (norm_after - 1.0).abs() < 1e-5,
            "clipped norm = {norm_after}"
        );
    }

    #[test]
    fn test_clip_grad_norm_no_clip() {
        let mut grads = vec![vec![0.3f32, 0.4]]; // norm = 0.5
        let norm_before = clip_grad_norm(&mut grads, 1.0);
        // Norm was already below max — values must not change.
        assert!((norm_before - 0.5).abs() < 1e-5);
        assert!(approx_eq(grads[0][0], 0.3));
        assert!(approx_eq(grads[0][1], 0.4));
    }

    #[test]
    fn test_grad_norm_correct() {
        let grads = vec![vec![3.0f32, 4.0]];
        let n = grad_norm(&grads);
        assert!(approx_eq(n, 5.0), "expected norm 5.0, got {n}");
    }
}
