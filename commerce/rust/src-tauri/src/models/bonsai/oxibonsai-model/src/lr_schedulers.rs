//! Advanced learning rate schedulers for LLM training.
//!
//! Supplements the basic schedulers in `optimizer.rs` with:
//! - [`OneCycleLr`] — Smith & Topin (2019) one-cycle policy
//! - [`ReduceOnPlateau`] — halve LR when a metric stops improving
//! - [`LinearWarmupCosineDecay`] — Llama-style warmup + cosine decay
//! - [`PolynomialDecay`] — power-law decay to an end learning rate
//! - [`CyclicLr`] — triangular cyclical LR (Smith, 2017)
//!
//! All schedulers are mutable-state objects driven by explicit calls to
//! their `step()` method, which returns the learning rate for the *current*
//! step and then advances the internal counter.

use std::f32::consts::PI;

// ─── OneCycleLr ───────────────────────────────────────────────────────────────

/// One-Cycle LR policy (Smith & Topin, 2019).
///
/// Phase 1 `[0, warmup_steps)`: linear ramp from `min_lr` → `max_lr`.
/// Phase 2 `[warmup_steps, total_steps]`: cosine decay from `max_lr` → `min_lr`.
///
/// The default warmup fraction is **0.3** (30 % of training for the rising
/// phase).
///
/// # Example
/// ```ignore
/// let mut sched = OneCycleLr::new(3e-4, 1000)
///     .with_warmup_fraction(0.2)
///     .with_min_lr(1e-6);
/// for _ in 0..1000 {
///     let lr = sched.step();
///     // apply lr to optimizer …
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleLr {
    /// Peak learning rate.
    pub max_lr: f32,
    /// Minimum (floor) learning rate used at start and end.
    pub min_lr: f32,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Number of warmup (rising-phase) steps.
    pub warmup_steps: usize,
    /// Current step index (0-based).
    step: usize,
}

impl OneCycleLr {
    /// Create a new `OneCycleLr` with `max_lr` and `total_steps`.
    ///
    /// Defaults: warmup fraction = 0.3, min_lr = `max_lr / 10_000`.
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        let warmup_steps = (total_steps as f32 * 0.3) as usize;
        let min_lr = max_lr / 10_000.0_f32;
        Self {
            max_lr,
            min_lr,
            total_steps,
            warmup_steps,
            step: 0,
        }
    }

    /// Override the warmup fraction (builder pattern).
    ///
    /// `fraction` must be in `[0, 1]`.  Values outside this range are clamped.
    pub fn with_warmup_fraction(mut self, fraction: f32) -> Self {
        let fraction = fraction.clamp(0.0, 1.0);
        self.warmup_steps = (self.total_steps as f32 * fraction) as usize;
        self
    }

    /// Set the minimum learning rate (builder pattern).
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Compute the learning rate for the current internal step.
    pub fn current_lr(&self) -> f32 {
        let s = self.step.min(self.total_steps);
        if s < self.warmup_steps {
            // Linear warmup.
            if self.warmup_steps == 0 {
                return self.max_lr;
            }
            let t = s as f32 / self.warmup_steps as f32;
            self.min_lr + t * (self.max_lr - self.min_lr)
        } else {
            // Cosine decay.
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
            if decay_steps == 0 {
                return self.min_lr;
            }
            let elapsed = s.saturating_sub(self.warmup_steps);
            let progress = (elapsed as f32 / decay_steps as f32).min(1.0);
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    /// Return the current learning rate and advance the internal step counter.
    pub fn step(&mut self) -> f32 {
        let lr = self.current_lr();
        self.step = (self.step + 1).min(self.total_steps);
        lr
    }

    /// Fractional progress through training: `step / total_steps` ∈ `[0, 1]`.
    pub fn progress(&self) -> f32 {
        if self.total_steps == 0 {
            return 1.0;
        }
        (self.step as f32 / self.total_steps as f32).min(1.0)
    }
}

// ─── ReduceOnPlateau ──────────────────────────────────────────────────────────

/// Selects whether the metric should be minimised or maximised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Reduce LR when the metric stops *decreasing* (e.g., validation loss).
    Min,
    /// Reduce LR when the metric stops *increasing* (e.g., accuracy).
    Max,
}

/// Reduce LR on Plateau scheduler.
///
/// Monitors a scalar metric and multiplies the learning rate by `factor`
/// (default 0.5) whenever the metric fails to improve for `patience`
/// consecutive calls.  The LR is never reduced below `min_lr`.
///
/// # Example
/// ```ignore
/// let mut sched = ReduceOnPlateau::new(1e-3, 5, PlateauMode::Min);
/// for epoch in 0..100 {
///     let val_loss = train_epoch();
///     let lr = sched.step(val_loss);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ReduceOnPlateau {
    /// Current learning rate.
    lr: f32,
    /// Multiplicative reduction factor (default 0.5).
    factor: f32,
    /// Steps without improvement before reducing.
    patience: usize,
    /// Hard floor for the learning rate.
    min_lr: f32,
    /// Best metric value seen so far.
    best_metric: f32,
    /// Number of consecutive non-improving steps.
    bad_steps: usize,
    /// Whether lower or higher is better.
    mode: PlateauMode,
    /// How many times the LR has been reduced.
    reduction_count: usize,
}

impl ReduceOnPlateau {
    /// Create a new `ReduceOnPlateau`.
    ///
    /// Defaults: factor = 0.5, min_lr = 1e-8.
    pub fn new(initial_lr: f32, patience: usize, mode: PlateauMode) -> Self {
        let best_metric = match mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };
        Self {
            lr: initial_lr,
            factor: 0.5,
            patience,
            min_lr: 1e-8,
            best_metric,
            bad_steps: 0,
            mode,
            reduction_count: 0,
        }
    }

    /// Override the reduction factor (builder pattern, called before training).
    pub fn with_factor(mut self, factor: f32) -> Self {
        self.factor = factor;
        self
    }

    /// Override the minimum LR floor (builder pattern).
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Record a new metric value and possibly reduce the LR.
    ///
    /// Returns the (possibly reduced) learning rate.
    pub fn step(&mut self, metric: f32) -> f32 {
        let improved = match self.mode {
            PlateauMode::Min => metric < self.best_metric,
            PlateauMode::Max => metric > self.best_metric,
        };
        if improved {
            self.best_metric = metric;
            self.bad_steps = 0;
        } else {
            self.bad_steps += 1;
            if self.bad_steps >= self.patience {
                let new_lr = (self.lr * self.factor).max(self.min_lr);
                if new_lr < self.lr {
                    self.lr = new_lr;
                    self.reduction_count += 1;
                }
                self.bad_steps = 0;
            }
        }
        self.lr
    }

    /// Return the current learning rate without advancing state.
    pub fn current_lr(&self) -> f32 {
        self.lr
    }

    /// How many times the LR has been reduced.
    pub fn times_reduced(&self) -> usize {
        self.reduction_count
    }
}

// ─── LinearWarmupCosineDecay ──────────────────────────────────────────────────

/// Linear warmup followed by cosine decay (Llama training style).
///
/// - **Warmup** `[0, warmup_steps)`: LR increases linearly from 0 → `max_lr`.
/// - **Decay** `[warmup_steps, total_steps]`: cosine annealing from `max_lr` → `min_lr`.
/// - After `total_steps`: clamped at `min_lr`.
#[derive(Debug, Clone)]
pub struct LinearWarmupCosineDecay {
    /// Peak learning rate.
    pub max_lr: f32,
    /// Floor learning rate.
    pub min_lr: f32,
    /// Number of linear-warmup steps.
    pub warmup_steps: usize,
    /// Total training steps (warmup + decay).
    pub total_steps: usize,
    /// Current step index.
    step: usize,
}

impl LinearWarmupCosineDecay {
    /// Create a new `LinearWarmupCosineDecay`.
    ///
    /// `min_lr` defaults to 0.
    pub fn new(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            step: 0,
        }
    }

    /// Override the minimum LR (builder pattern).
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Compute the learning rate for the current internal step.
    pub fn current_lr(&self) -> f32 {
        let s = self.step.min(self.total_steps);
        if s < self.warmup_steps {
            if self.warmup_steps == 0 {
                return self.max_lr;
            }
            // Linear ramp from 0 → max_lr.
            self.max_lr * (s as f32 / self.warmup_steps as f32)
        } else {
            let cosine_steps = self.total_steps.saturating_sub(self.warmup_steps);
            if cosine_steps == 0 {
                return self.min_lr;
            }
            let elapsed = s.saturating_sub(self.warmup_steps);
            let progress = (elapsed as f32 / cosine_steps as f32).min(1.0);
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    /// Return the current LR and advance the step counter.
    pub fn step(&mut self) -> f32 {
        let lr = self.current_lr();
        self.step = (self.step + 1).min(self.total_steps);
        lr
    }
}

// ─── PolynomialDecay ──────────────────────────────────────────────────────────

/// Polynomial learning rate decay.
///
/// ```text
/// lr(t) = (initial_lr − end_lr) · (1 − t / total_steps)^power + end_lr
/// ```
///
/// At `t = 0` the LR equals `initial_lr`; at `t = total_steps` it equals
/// `end_lr`.
#[derive(Debug, Clone)]
pub struct PolynomialDecay {
    /// Starting learning rate.
    pub initial_lr: f32,
    /// Final learning rate.
    pub end_lr: f32,
    /// Total number of decay steps.
    pub total_steps: usize,
    /// Polynomial exponent (e.g., 1.0 = linear, 2.0 = quadratic).
    pub power: f32,
    /// Current step.
    step: usize,
}

impl PolynomialDecay {
    /// Create a new `PolynomialDecay`.
    pub fn new(initial_lr: f32, end_lr: f32, total_steps: usize, power: f32) -> Self {
        Self {
            initial_lr,
            end_lr,
            total_steps,
            power,
            step: 0,
        }
    }

    /// Compute the learning rate for the current internal step.
    pub fn current_lr(&self) -> f32 {
        if self.total_steps == 0 || self.step >= self.total_steps {
            return self.end_lr;
        }
        let t = self.step as f32 / self.total_steps as f32;
        let decay = (1.0 - t).powf(self.power);
        (self.initial_lr - self.end_lr) * decay + self.end_lr
    }

    /// Return the current LR and advance the step counter.
    pub fn step(&mut self) -> f32 {
        let lr = self.current_lr();
        self.step = (self.step + 1).min(self.total_steps);
        lr
    }
}

// ─── CyclicLr ────────────────────────────────────────────────────────────────

/// Cyclical learning rate — triangular oscillation between `base_lr` and
/// `max_lr` (Smith, 2017).
///
/// Each full cycle has length `2 * step_size` steps: the first half linearly
/// increases from `base_lr` → `max_lr`; the second half linearly decreases back.
///
/// # Example
/// ```ignore
/// let mut sched = CyclicLr::new(1e-4, 1e-2, 500);
/// for _ in 0..2000 {
///     let lr = sched.step();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CyclicLr {
    /// Minimum (base) learning rate.
    pub base_lr: f32,
    /// Maximum learning rate.
    pub max_lr: f32,
    /// Number of steps in a *half*-cycle.
    pub step_size: usize,
    /// Global step counter.
    step: usize,
}

impl CyclicLr {
    /// Create a new `CyclicLr`.
    pub fn new(base_lr: f32, max_lr: f32, step_size: usize) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size,
            step: 0,
        }
    }

    /// Position within the current *full* cycle: `[0, 1)`.
    ///
    /// 0.0–0.5 = rising half; 0.5–1.0 = falling half.
    pub fn cycle_position(&self) -> f32 {
        if self.step_size == 0 {
            return 0.0;
        }
        let cycle_len = 2 * self.step_size;
        let pos_in_cycle = self.step % cycle_len;
        pos_in_cycle as f32 / cycle_len as f32
    }

    /// Compute the LR for the current internal step.
    pub fn current_lr(&self) -> f32 {
        if self.step_size == 0 {
            return self.base_lr;
        }
        let cycle_len = 2 * self.step_size;
        let pos_in_cycle = self.step % cycle_len;
        // Triangular: rise for first half, fall for second.
        let t = if pos_in_cycle < self.step_size {
            pos_in_cycle as f32 / self.step_size as f32
        } else {
            1.0 - (pos_in_cycle - self.step_size) as f32 / self.step_size as f32
        };
        self.base_lr + t * (self.max_lr - self.base_lr)
    }

    /// Return the current LR and advance the step counter.
    pub fn step(&mut self) -> f32 {
        let lr = self.current_lr();
        self.step += 1;
        lr
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── OneCycleLr ────────────────────────────────────────────────────────────

    #[test]
    fn onecycle_starts_at_min_lr() {
        let sched = OneCycleLr::new(1.0, 100)
            .with_min_lr(0.01)
            .with_warmup_fraction(0.3);
        // Step 0 is the very first step before any progress.
        let lr = sched.current_lr();
        assert!(
            approx_eq(lr, 0.01, 1e-3),
            "first LR should be ~min_lr, got {lr}"
        );
    }

    #[test]
    fn onecycle_peaks_at_warmup() {
        let total = 100_usize;
        let warmup_frac = 0.3_f32;
        let mut sched = OneCycleLr::new(1.0, total)
            .with_min_lr(0.0)
            .with_warmup_fraction(warmup_frac);
        // Drive to the warmup boundary.
        let warmup_steps = (total as f32 * warmup_frac) as usize;
        let mut lr_at_peak = 0.0_f32;
        for i in 0..=total {
            let lr = sched.step();
            if i + 1 == warmup_steps {
                lr_at_peak = lr;
            }
        }
        assert!(
            approx_eq(lr_at_peak, 1.0, 0.05),
            "LR should peak near max_lr at warmup boundary, got {lr_at_peak}"
        );
    }

    #[test]
    fn onecycle_ends_at_min_lr() {
        let total = 100_usize;
        let max_lr = 1.0_f32;
        let min_lr = 1e-4_f32;
        // After all steps the LR should be at min_lr.
        let sched = {
            let mut s = OneCycleLr::new(max_lr, total)
                .with_min_lr(min_lr)
                .with_warmup_fraction(0.3);
            for _ in 0..total {
                s.step();
            }
            s
        };
        let lr = sched.current_lr();
        assert!(
            approx_eq(lr, min_lr, min_lr * 10.0),
            "final LR should be ~min_lr, got {lr}"
        );
    }

    #[test]
    fn onecycle_progress_monotone() {
        let total = 50_usize;
        let mut sched = OneCycleLr::new(1.0, total);
        let mut prev = sched.progress();
        for _ in 0..total {
            sched.step();
            let p = sched.progress();
            assert!(p >= prev, "progress must be non-decreasing: {prev} → {p}");
            prev = p;
        }
        assert!(approx_eq(prev, 1.0, EPS), "progress must reach 1.0 at end");
    }

    // ── ReduceOnPlateau ───────────────────────────────────────────────────────

    #[test]
    fn reduce_plateau_min_mode_reduces_lr() {
        let patience = 3_usize;
        let mut sched = ReduceOnPlateau::new(1e-2, patience, PlateauMode::Min);
        // First call establishes best = 1.0.  Then `patience` identical calls
        // accumulate bad_steps until the reduction is triggered.
        sched.step(1.0); // sets best_metric = 1.0 (improvement from +inf)
        for _ in 0..patience {
            sched.step(1.0); // non-improving: bad_steps increments each time
        }
        assert_eq!(
            sched.times_reduced(),
            1,
            "should have reduced once after patience steps"
        );
        assert!(sched.current_lr() < 1e-2, "LR should have decreased");
    }

    #[test]
    fn reduce_plateau_improvement_keeps_lr() {
        let mut sched = ReduceOnPlateau::new(1e-2, 3, PlateauMode::Min);
        // Improving metric — should never reduce.
        for i in 0..20_usize {
            sched.step(1.0 / (i + 1) as f32);
        }
        assert_eq!(
            sched.times_reduced(),
            0,
            "should not reduce when metric improves"
        );
        assert!(approx_eq(sched.current_lr(), 1e-2, EPS));
    }

    #[test]
    fn reduce_plateau_min_lr_floor() {
        let min_lr = 1e-5_f32;
        let mut sched = ReduceOnPlateau::new(1e-3, 1, PlateauMode::Min).with_min_lr(min_lr);
        // Feed many bad steps.
        for _ in 0..100 {
            sched.step(1.0);
        }
        assert!(
            sched.current_lr() >= min_lr,
            "LR must never go below min_lr, got {}",
            sched.current_lr()
        );
    }

    // ── LinearWarmupCosineDecay ───────────────────────────────────────────────

    #[test]
    fn linear_warmup_cosine_warmup_phase_increases() {
        let warmup = 10_usize;
        let total = 100_usize;
        let mut sched = LinearWarmupCosineDecay::new(1.0, warmup, total);
        let mut prev = -1.0_f32;
        for _ in 0..warmup {
            let lr = sched.step();
            assert!(lr >= prev, "LR must increase during warmup: {prev} → {lr}");
            prev = lr;
        }
    }

    #[test]
    fn linear_warmup_cosine_decay_phase_decreases() {
        let warmup = 10_usize;
        let total = 100_usize;
        let mut sched = LinearWarmupCosineDecay::new(1.0, warmup, total).with_min_lr(0.0);
        // Skip warmup.
        for _ in 0..warmup {
            sched.step();
        }
        let mut prev = f32::INFINITY;
        for _ in warmup..total {
            let lr = sched.step();
            assert!(
                lr <= prev + EPS,
                "LR must decrease (or stay) during decay: {prev} → {lr}"
            );
            prev = lr;
        }
    }

    // ── PolynomialDecay ───────────────────────────────────────────────────────

    #[test]
    fn polynomial_decay_starts_at_initial_lr() {
        let sched = PolynomialDecay::new(1e-3, 1e-6, 1000, 1.0);
        let first = sched.current_lr();
        assert!(
            approx_eq(first, 1e-3, 1e-7),
            "should start at initial_lr, got {first}"
        );
    }

    #[test]
    fn polynomial_decay_ends_at_end_lr() {
        let end_lr = 1e-6_f32;
        let mut sched = PolynomialDecay::new(1e-3, end_lr, 100, 1.0);
        for _ in 0..100 {
            sched.step();
        }
        let last = sched.current_lr();
        assert!(
            approx_eq(last, end_lr, 1e-9),
            "should end at end_lr, got {last}"
        );
    }

    // ── CyclicLr ──────────────────────────────────────────────────────────────

    #[test]
    fn cyclic_lr_oscillates() {
        let base = 1e-4_f32;
        let max = 1e-2_f32;
        let step_size = 10_usize;
        let mut sched = CyclicLr::new(base, max, step_size);
        // Collect one full cycle.
        let lrs: Vec<f32> = (0..2 * step_size).map(|_| sched.step()).collect();
        // Rising half: LR should increase.
        for i in 1..step_size {
            assert!(
                lrs[i] >= lrs[i - 1] - EPS,
                "should rise in first half: lrs[{i}]={} < lrs[{}]={}",
                lrs[i],
                i - 1,
                lrs[i - 1]
            );
        }
        // Falling half: LR should decrease.
        for i in (step_size + 1)..(2 * step_size) {
            assert!(
                lrs[i] <= lrs[i - 1] + EPS,
                "should fall in second half: lrs[{i}]={} > lrs[{}]={}",
                lrs[i],
                i - 1,
                lrs[i - 1]
            );
        }
    }

    #[test]
    fn cyclic_lr_period_is_two_step_size() {
        let step_size = 20_usize;
        let mut sched = CyclicLr::new(0.0, 1.0, step_size);
        // LR at step k should equal LR at step k + 2*step_size.
        let lrs_first: Vec<f32> = (0..2 * step_size).map(|_| sched.step()).collect();
        let lrs_second: Vec<f32> = (0..2 * step_size).map(|_| sched.step()).collect();
        for (a, b) in lrs_first.iter().zip(lrs_second.iter()) {
            assert!(
                approx_eq(*a, *b, EPS),
                "cyclic LR must repeat with period 2*step_size: {a} vs {b}"
            );
        }
    }
}
