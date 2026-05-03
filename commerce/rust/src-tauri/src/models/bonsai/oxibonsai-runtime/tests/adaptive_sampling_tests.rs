//! Integration tests for adaptive sampling strategies.

use oxibonsai_runtime::adaptive_sampling::AdaptiveStrategy;
use oxibonsai_runtime::adaptive_sampling::{
    AdaptiveSamplerChain, EntropyCooling, GenerationState, RepetitionAdaptation, ScheduledDecay,
};
use oxibonsai_runtime::sampling::SamplingParams;

// ─── GenerationState tests ─────────────────────────────────────────────────────

#[test]
fn generation_state_new_empty() {
    let state = GenerationState::new();
    assert_eq!(state.step, 0);
    assert!(state.recent_tokens.is_empty());
    assert!(state.recent_entropies.is_empty());
    assert_eq!(state.repetition_count, 0);
}

#[test]
fn generation_state_update() {
    let mut state = GenerationState::new();
    state.update(42, 1.5);
    assert_eq!(state.step, 1);
    assert_eq!(state.recent_tokens, vec![42u32]);
    assert!((state.recent_entropies[0] - 1.5).abs() < 1e-6);
}

#[test]
fn generation_state_repetition_rate_no_rep() {
    let mut state = GenerationState::new();
    for tok in [1u32, 2, 3, 4, 5] {
        state.update(tok, 1.0);
    }
    let rate = state.recent_repetition_rate(5);
    assert!(
        (rate - 0.0).abs() < 1e-6,
        "expected 0 repetition rate, got {rate}"
    );
}

#[test]
fn generation_state_repetition_rate_all_same() {
    let mut state = GenerationState::new();
    for _ in 0..5 {
        state.update(7, 1.0);
    }
    let rate = state.recent_repetition_rate(5);
    assert!(rate > 0.5, "expected high repetition rate, got {rate}");
}

#[test]
fn generation_state_mean_entropy() {
    let mut state = GenerationState::new();
    state.update(1, 2.0);
    state.update(2, 4.0);
    state.update(3, 6.0);
    let mean = state.mean_recent_entropy(3);
    assert!((mean - 4.0).abs() < 1e-5, "expected 4.0, got {mean}");
}

// ─── EntropyCooling tests ──────────────────────────────────────────────────────

#[test]
fn entropy_cooling_high_entropy_reduces_temp() {
    let strategy = EntropyCooling::new(1.0);
    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    // Entropy well above target of 1.0
    for _ in 0..8 {
        state.update(1, 3.0);
    }
    let adjusted = strategy.adjust(&state, &base);
    assert!(
        adjusted.temperature < base.temperature,
        "expected temperature decrease, got {}",
        adjusted.temperature
    );
}

#[test]
fn entropy_cooling_low_entropy_no_change() {
    let strategy = EntropyCooling::new(2.0);
    let base = SamplingParams {
        temperature: 0.7,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    // Entropy below target
    for _ in 0..8 {
        state.update(1, 0.5);
    }
    let adjusted = strategy.adjust(&state, &base);
    assert!(
        (adjusted.temperature - base.temperature).abs() < 1e-6,
        "expected no change, got {}",
        adjusted.temperature
    );
}

#[test]
fn entropy_cooling_min_temp_floor() {
    let strategy = EntropyCooling {
        target_entropy: 0.0,
        cooling_rate: 100.0,
        min_temperature: 0.05,
    };
    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    for _ in 0..8 {
        state.update(1, 5.0);
    }
    let adjusted = strategy.adjust(&state, &base);
    assert!(
        adjusted.temperature >= 0.05,
        "temperature below min floor: {}",
        adjusted.temperature
    );
}

// ─── RepetitionAdaptation tests ───────────────────────────────────────────────

#[test]
fn repetition_adaptation_high_rep_cools() {
    let strategy = RepetitionAdaptation::new();
    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    // Same token repeated many times → high rep rate
    for _ in 0..20 {
        state.update(42, 0.1);
    }
    let adjusted = strategy.adjust(&state, &base);
    assert!(
        adjusted.temperature < base.temperature,
        "expected cooling, got {}",
        adjusted.temperature
    );
}

#[test]
fn repetition_adaptation_low_rep_unchanged() {
    let strategy = RepetitionAdaptation::new();
    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    // Unique tokens
    for i in 0..5u32 {
        state.update(i, 1.0);
    }
    let adjusted = strategy.adjust(&state, &base);
    // Should not cool — may heat slightly but must be >= base.
    assert!(
        adjusted.temperature >= base.temperature - 0.01,
        "unexpected cooling to {}",
        adjusted.temperature
    );
}

// ─── ScheduledDecay tests ─────────────────────────────────────────────────────

#[test]
fn scheduled_decay_at_step_zero() {
    let sched = ScheduledDecay::new(1.0, 0.1, 100);
    assert!((sched.temperature_at_step(0) - 1.0).abs() < 1e-6);
}

#[test]
fn scheduled_decay_at_final_step() {
    let sched = ScheduledDecay::new(1.0, 0.1, 100);
    assert!((sched.temperature_at_step(100) - 0.1).abs() < 1e-6);
}

#[test]
fn scheduled_decay_intermediate() {
    let sched = ScheduledDecay::new(1.0, 0.0, 100);
    let mid = sched.temperature_at_step(50);
    assert!((mid - 0.5).abs() < 1e-5, "expected 0.5, got {mid}");
}

// ─── AdaptiveSamplerChain tests ───────────────────────────────────────────────

#[test]
fn adaptive_chain_empty() {
    let chain = AdaptiveSamplerChain::new();
    let base = SamplingParams::default();
    let state = GenerationState::new();
    let adjusted = chain.adjust(&state, &base);
    assert!(
        (adjusted.temperature - base.temperature).abs() < 1e-6,
        "empty chain should leave params unchanged"
    );
}

#[test]
fn adaptive_chain_applies_all() {
    // Two strategies applied sequentially.
    let chain = AdaptiveSamplerChain::new()
        .add(Box::new(ScheduledDecay::new(1.0, 0.0, 100)))
        .add(Box::new(EntropyCooling::new(0.0)));

    assert_eq!(chain.len(), 2);

    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut state = GenerationState::new();
    // Advance to step 50 with high entropy so both strategies lower temperature.
    for _ in 0..50 {
        state.update(1, 5.0);
    }

    let adjusted = chain.adjust(&state, &base);
    // ScheduledDecay at step=50 gives temp=0.5; EntropyCooling lowers further.
    assert!(
        adjusted.temperature <= 0.5 + 1e-3,
        "expected temp <= 0.5, got {}",
        adjusted.temperature
    );
}
