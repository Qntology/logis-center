//! Integration tests for `oxibonsai_model::lr_schedulers`.

use oxibonsai_model::lr_schedulers::{
    CyclicLr, LinearWarmupCosineDecay, OneCycleLr, PlateauMode, PolynomialDecay, ReduceOnPlateau,
};

const EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ── 1. OneCycleLr starts at min_lr ────────────────────────────────────────────

#[test]
fn onecycle_starts_at_min() {
    let min_lr = 1e-5_f32;
    let sched = OneCycleLr::new(1e-3, 200)
        .with_min_lr(min_lr)
        .with_warmup_fraction(0.3);
    // Before any step the position is 0 (= start of warmup).
    let lr = sched.current_lr();
    assert!(
        approx_eq(lr, min_lr, min_lr * 2.0),
        "initial LR should be ~min_lr, got {lr}"
    );
}

// ── 2. OneCycleLr peaks near max_lr at the warmup boundary ───────────────────

#[test]
fn onecycle_peaks_at_warmup() {
    let total = 100_usize;
    let warmup_frac = 0.3_f32;
    let max_lr = 1.0_f32;
    let mut sched = OneCycleLr::new(max_lr, total)
        .with_min_lr(0.0)
        .with_warmup_fraction(warmup_frac);
    let warmup_steps = (total as f32 * warmup_frac) as usize;
    let mut lr_at_warmup_end = 0.0_f32;
    for i in 0..total {
        let lr = sched.step();
        if i + 1 == warmup_steps {
            lr_at_warmup_end = lr;
        }
    }
    assert!(
        approx_eq(lr_at_warmup_end, max_lr, 0.05),
        "LR should peak near max_lr at warmup boundary, got {lr_at_warmup_end}"
    );
}

// ── 3. OneCycleLr ends near min_lr ────────────────────────────────────────────

#[test]
fn onecycle_ends_at_min() {
    let total = 100_usize;
    let min_lr = 1e-5_f32;
    let mut sched = OneCycleLr::new(1e-3, total)
        .with_min_lr(min_lr)
        .with_warmup_fraction(0.3);
    for _ in 0..total {
        sched.step();
    }
    let lr = sched.current_lr();
    assert!(
        approx_eq(lr, min_lr, min_lr * 10.0),
        "final LR should be ~min_lr, got {lr}"
    );
}

// ── 4. OneCycleLr progress is monotone and reaches 1 ─────────────────────────

#[test]
fn onecycle_progress_monotone_to_one() {
    let total = 60_usize;
    let mut sched = OneCycleLr::new(1e-3, total);
    let mut prev = sched.progress();
    for _ in 0..total {
        sched.step();
        let p = sched.progress();
        assert!(p >= prev, "progress must be non-decreasing: {prev} → {p}");
        prev = p;
    }
    assert!(approx_eq(prev, 1.0, EPS), "progress must reach 1.0 at end");
}

// ── 5. ReduceOnPlateau reduces LR after patience bad steps ────────────────────

#[test]
fn reduce_plateau_min_mode_reduces() {
    let patience = 4_usize;
    let init_lr = 1e-2_f32;
    let mut sched = ReduceOnPlateau::new(init_lr, patience, PlateauMode::Min);
    // First call establishes the best metric; subsequent `patience` calls
    // with the same value accumulate bad_steps and trigger one reduction.
    sched.step(1.0); // sets best = 1.0
    for _ in 0..patience {
        sched.step(1.0); // non-improving
    }
    assert_eq!(
        sched.times_reduced(),
        1,
        "should have reduced exactly once after patience={patience} bad steps"
    );
    assert!(
        sched.current_lr() < init_lr,
        "LR must have decreased: {}",
        sched.current_lr()
    );
}

// ── 6. ReduceOnPlateau does not reduce when metric improves ───────────────────

#[test]
fn reduce_plateau_not_reduce_on_improvement() {
    let init_lr = 1e-3_f32;
    let mut sched = ReduceOnPlateau::new(init_lr, 3, PlateauMode::Min);
    for i in 0..30_usize {
        sched.step(1.0 / (i as f32 + 1.0));
    }
    assert_eq!(
        sched.times_reduced(),
        0,
        "must not reduce when metric improves"
    );
    assert!(approx_eq(sched.current_lr(), init_lr, EPS));
}

// ── 7. ReduceOnPlateau respects min_lr floor ──────────────────────────────────

#[test]
fn reduce_plateau_min_lr_floor() {
    let min_lr = 1e-6_f32;
    let mut sched = ReduceOnPlateau::new(1e-2, 1, PlateauMode::Min).with_min_lr(min_lr);
    // Drive many reductions.
    for _ in 0..200 {
        sched.step(1.0);
    }
    assert!(
        sched.current_lr() >= min_lr,
        "LR must never go below min_lr, got {}",
        sched.current_lr()
    );
}

// ── 8. LinearWarmupCosineDecay increases during warmup ────────────────────────

#[test]
fn linear_warmup_cosine_warmup_phase_increases() {
    let warmup = 20_usize;
    let total = 200_usize;
    let mut sched = LinearWarmupCosineDecay::new(1.0, warmup, total);
    let mut prev = -1.0_f32;
    for _ in 0..warmup {
        let lr = sched.step();
        assert!(
            lr >= prev,
            "LR should increase during warmup: {prev} → {lr}"
        );
        prev = lr;
    }
}

// ── 9. LinearWarmupCosineDecay decreases during decay phase ───────────────────

#[test]
fn linear_warmup_cosine_decay_phase_decreases() {
    let warmup = 10_usize;
    let total = 100_usize;
    let mut sched = LinearWarmupCosineDecay::new(1.0, warmup, total).with_min_lr(0.0);
    // Advance past warmup.
    for _ in 0..warmup {
        sched.step();
    }
    let mut prev = f32::INFINITY;
    for _ in warmup..total {
        let lr = sched.step();
        assert!(
            lr <= prev + EPS,
            "LR should decrease during decay: {prev} → {lr}"
        );
        prev = lr;
    }
}

// ── 10. PolynomialDecay starts at initial_lr ──────────────────────────────────

#[test]
fn polynomial_decay_start() {
    let init = 3e-4_f32;
    let sched = PolynomialDecay::new(init, 1e-6, 500, 2.0);
    assert!(
        approx_eq(sched.current_lr(), init, 1e-9),
        "should start at initial_lr={init}, got {}",
        sched.current_lr()
    );
}

// ── 11. PolynomialDecay ends at end_lr ───────────────────────────────────────

#[test]
fn polynomial_decay_end() {
    let end_lr = 1e-7_f32;
    let steps = 200_usize;
    let mut sched = PolynomialDecay::new(1e-3, end_lr, steps, 1.0);
    for _ in 0..steps {
        sched.step();
    }
    assert!(
        approx_eq(sched.current_lr(), end_lr, 1e-11),
        "should end at end_lr={end_lr}, got {}",
        sched.current_lr()
    );
}

// ── 12. CyclicLr oscillates up then down ─────────────────────────────────────

#[test]
fn cyclic_lr_oscillates() {
    let base_lr = 0.0_f32;
    let max_lr = 1.0_f32;
    let step_size = 5_usize;
    let mut sched = CyclicLr::new(base_lr, max_lr, step_size);
    let lrs: Vec<f32> = (0..2 * step_size).map(|_| sched.step()).collect();
    // First half must be non-decreasing.
    for i in 1..step_size {
        assert!(
            lrs[i] >= lrs[i - 1] - EPS,
            "rising half: lrs[{i}]={} < lrs[{}]={}",
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }
    // Second half must be non-increasing.
    for i in (step_size + 1)..(2 * step_size) {
        assert!(
            lrs[i] <= lrs[i - 1] + EPS,
            "falling half: lrs[{i}]={} > lrs[{}]={}",
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }
}

// ── 13. CyclicLr repeats with period 2*step_size ──────────────────────────────

#[test]
fn cyclic_lr_period() {
    let step_size = 15_usize;
    let cycle = 2 * step_size;
    let mut sched = CyclicLr::new(0.0, 1.0, step_size);
    let lrs_c1: Vec<f32> = (0..cycle).map(|_| sched.step()).collect();
    let lrs_c2: Vec<f32> = (0..cycle).map(|_| sched.step()).collect();
    for (k, (a, b)) in lrs_c1.iter().zip(lrs_c2.iter()).enumerate() {
        assert!(
            approx_eq(*a, *b, EPS),
            "cycle[{k}]: first={a}, second={b} — period must be 2*step_size"
        );
    }
}

// ── 14. ReduceOnPlateau Max mode reduces on non-improving (non-increasing) ─────

#[test]
fn reduce_plateau_max_mode_reduces() {
    let patience = 3_usize;
    let init_lr = 5e-3_f32;
    let mut sched = ReduceOnPlateau::new(init_lr, patience, PlateauMode::Max);
    // First call establishes best = 0.5; subsequent `patience` identical calls
    // accumulate bad_steps and trigger a reduction.
    sched.step(0.5); // sets best = 0.5
    for _ in 0..patience {
        sched.step(0.5); // non-improving in Max mode
    }
    assert!(
        sched.times_reduced() >= 1,
        "Max mode: should reduce after constant metric"
    );
}

// ── 15. LinearWarmupCosineDecay final LR is min_lr ────────────────────────────

#[test]
fn linear_warmup_cosine_ends_at_min_lr() {
    let min_lr = 1e-6_f32;
    let total = 100_usize;
    let warmup = 10_usize;
    let mut sched = LinearWarmupCosineDecay::new(1e-3, warmup, total).with_min_lr(min_lr);
    for _ in 0..total {
        sched.step();
    }
    let lr = sched.current_lr();
    assert!(
        approx_eq(lr, min_lr, min_lr * 10.0),
        "final LR should be ~min_lr, got {lr}"
    );
}
