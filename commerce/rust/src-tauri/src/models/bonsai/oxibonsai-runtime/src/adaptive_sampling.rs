//! Adaptive sampling: dynamically adjust temperature/top_p based on generation state.
//!
//! Strategies:
//! - `EntropyCooling`: lower temperature when entropy is too high (reduce randomness)
//! - `RepetitionAdaptation`: lower temp when repeating, raise when stuck
//! - `ScheduledDecay`: gradually decay temperature over the sequence

use crate::sampling::SamplingParams;

// ─── GenerationState ───────────────────────────────────────────────────────────

/// Current generation state for adaptive decisions.
#[derive(Debug, Clone)]
pub struct GenerationState {
    /// Current decoding step (0-indexed).
    pub step: usize,
    /// Last N generated tokens (ring-buffer style; most recent last).
    pub recent_tokens: Vec<u32>,
    /// Shannon entropy (in nats) at each recent step.
    pub recent_entropies: Vec<f32>,
    /// Number of consecutive steps where repeated n-grams were detected.
    pub repetition_count: usize,
}

impl Default for GenerationState {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationState {
    const WINDOW_CAP: usize = 64;

    /// Create a fresh, empty generation state.
    pub fn new() -> Self {
        Self {
            step: 0,
            recent_tokens: Vec::new(),
            recent_entropies: Vec::new(),
            repetition_count: 0,
        }
    }

    /// Record a newly generated token and the entropy at this step.
    pub fn update(&mut self, token: u32, entropy: f32) {
        self.step += 1;

        self.recent_tokens.push(token);
        if self.recent_tokens.len() > Self::WINDOW_CAP {
            self.recent_tokens.remove(0);
        }

        self.recent_entropies.push(entropy);
        if self.recent_entropies.len() > Self::WINDOW_CAP {
            self.recent_entropies.remove(0);
        }

        // Detect bigram repetition in the recent window.
        let len = self.recent_tokens.len();
        if len >= 2 {
            let last = self.recent_tokens[len - 1];
            let prev = self.recent_tokens[len - 2];
            // Check whether the same bigram appeared before in the window.
            let repeated = self.recent_tokens[..len.saturating_sub(2)]
                .windows(2)
                .any(|w| w[0] == prev && w[1] == last);
            if repeated {
                self.repetition_count += 1;
            } else {
                self.repetition_count = 0;
            }
        }
    }

    /// Fraction of the last `window` tokens that are identical to the immediately
    /// preceding token (simple unigram repetition rate).
    pub fn recent_repetition_rate(&self, window: usize) -> f32 {
        if window == 0 || self.recent_tokens.is_empty() {
            return 0.0;
        }
        let tokens = &self.recent_tokens;
        let start = tokens.len().saturating_sub(window);
        let slice = &tokens[start..];
        if slice.len() < 2 {
            return 0.0;
        }
        let repeats = slice.windows(2).filter(|w| w[0] == w[1]).count();
        repeats as f32 / (slice.len() - 1) as f32
    }

    /// Mean entropy over the last `window` steps.
    pub fn mean_recent_entropy(&self, window: usize) -> f32 {
        if window == 0 || self.recent_entropies.is_empty() {
            return 0.0;
        }
        let start = self.recent_entropies.len().saturating_sub(window);
        let slice = &self.recent_entropies[start..];
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().sum::<f32>() / slice.len() as f32
    }
}

// ─── AdaptiveStrategy ──────────────────────────────────────────────────────────

/// Adaptive sampling strategy.
pub trait AdaptiveStrategy: Send + Sync {
    /// Given the current generation state and base params, return adjusted params.
    fn adjust(&self, state: &GenerationState, base: &SamplingParams) -> SamplingParams;
    /// Human-readable name of this strategy.
    fn name(&self) -> &'static str;
}

// ─── EntropyCooling ────────────────────────────────────────────────────────────

/// Lower temperature when entropy is too high (generation is too random).
///
/// When `mean_entropy > target_entropy`, temperature is scaled down by
/// `cooling_rate * excess_ratio`, clamped to `[min_temperature, base_temp]`.
pub struct EntropyCooling {
    /// Entropy level above which cooling begins (in nats).
    pub target_entropy: f32,
    /// Fraction of the excess entropy translated into temperature reduction (0..1).
    pub cooling_rate: f32,
    /// Minimum temperature floor.
    pub min_temperature: f32,
}

impl EntropyCooling {
    /// Create with sensible defaults.
    pub fn new(target_entropy: f32) -> Self {
        Self {
            target_entropy,
            cooling_rate: 0.5,
            min_temperature: 0.1,
        }
    }
}

impl AdaptiveStrategy for EntropyCooling {
    fn adjust(&self, state: &GenerationState, base: &SamplingParams) -> SamplingParams {
        let mut params = base.clone();
        let window = 8.min(state.recent_entropies.len().max(1));
        let mean_entropy = state.mean_recent_entropy(window);

        if mean_entropy > self.target_entropy {
            let excess = mean_entropy - self.target_entropy;
            // Reduce temperature proportionally to excess entropy.
            let reduction = self.cooling_rate * excess;
            let new_temp = (base.temperature - reduction).max(self.min_temperature);
            params.temperature = new_temp;
        }

        params
    }

    fn name(&self) -> &'static str {
        "EntropyCooling"
    }
}

// ─── RepetitionAdaptation ─────────────────────────────────────────────────────

/// Adapt temperature based on repetition rate.
///
/// - High repetition → cool down (reduce temperature) to break out of loops.
/// - Low repetition with high entropy → heat up slightly to encourage diversity.
pub struct RepetitionAdaptation {
    /// Repetition rate above which cooling is applied (0..1).
    pub rep_threshold: f32,
    /// Multiply temperature by this factor when repeating (< 1.0 to cool).
    pub cool_factor: f32,
    /// Multiply temperature by this factor when stuck (> 1.0 to heat).
    pub heat_factor: f32,
}

impl Default for RepetitionAdaptation {
    fn default() -> Self {
        Self::new()
    }
}

impl RepetitionAdaptation {
    /// Create with sensible defaults.
    pub fn new() -> Self {
        Self {
            rep_threshold: 0.3,
            cool_factor: 0.8,
            heat_factor: 1.1,
        }
    }
}

impl AdaptiveStrategy for RepetitionAdaptation {
    fn adjust(&self, state: &GenerationState, base: &SamplingParams) -> SamplingParams {
        let mut params = base.clone();
        let window = 16.min(state.recent_tokens.len().max(1));
        let rep_rate = state.recent_repetition_rate(window);

        if rep_rate > self.rep_threshold {
            params.temperature = (base.temperature * self.cool_factor).max(0.01);
        } else if rep_rate < self.rep_threshold / 2.0 && state.step > 4 {
            // Very low repetition — gentle heating to encourage variety.
            params.temperature = (base.temperature * self.heat_factor).min(2.0);
        }

        params
    }

    fn name(&self) -> &'static str {
        "RepetitionAdaptation"
    }
}

// ─── ScheduledDecay ────────────────────────────────────────────────────────────

/// Linearly decay temperature from `initial_temperature` to `final_temperature`
/// over `total_steps` decoding steps.
pub struct ScheduledDecay {
    /// Starting temperature (at step 0).
    pub initial_temperature: f32,
    /// Ending temperature (at step >= total_steps).
    pub final_temperature: f32,
    /// Number of steps over which to interpolate.
    pub total_steps: usize,
}

impl ScheduledDecay {
    /// Create a new scheduled decay.
    pub fn new(initial: f32, final_temp: f32, steps: usize) -> Self {
        Self {
            initial_temperature: initial,
            final_temperature: final_temp,
            total_steps: steps,
        }
    }

    /// Return the interpolated temperature at the given absolute step.
    pub fn temperature_at_step(&self, step: usize) -> f32 {
        if self.total_steps == 0 {
            return self.final_temperature;
        }
        let t = (step as f32 / self.total_steps as f32).min(1.0);
        self.initial_temperature + t * (self.final_temperature - self.initial_temperature)
    }
}

impl AdaptiveStrategy for ScheduledDecay {
    fn adjust(&self, state: &GenerationState, base: &SamplingParams) -> SamplingParams {
        let mut params = base.clone();
        params.temperature = self.temperature_at_step(state.step);
        params
    }

    fn name(&self) -> &'static str {
        "ScheduledDecay"
    }
}

// ─── AdaptiveSamplerChain ─────────────────────────────────────────────────────

/// Compose multiple adaptive strategies by applying them in sequence.
///
/// Each strategy sees the result of the previous one's adjustment.
pub struct AdaptiveSamplerChain {
    strategies: Vec<Box<dyn AdaptiveStrategy>>,
}

impl Default for AdaptiveSamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveSamplerChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Append a strategy (builder pattern).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, strategy: Box<dyn AdaptiveStrategy>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Apply all strategies in order, threading params through each.
    pub fn adjust(&self, state: &GenerationState, base: &SamplingParams) -> SamplingParams {
        self.strategies
            .iter()
            .fold(base.clone(), |params, strategy| {
                strategy.adjust(state, &params)
            })
    }

    /// Number of strategies in this chain.
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Whether this chain has no strategies.
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(state.recent_tokens, vec![42]);
        assert!((state.recent_entropies[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn generation_state_repetition_rate_no_rep() {
        let mut state = GenerationState::new();
        for tok in [1u32, 2, 3, 4, 5] {
            state.update(tok, 1.0);
        }
        let rate = state.recent_repetition_rate(5);
        assert!((rate - 0.0).abs() < 1e-6);
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

    #[test]
    fn entropy_cooling_high_entropy_reduces_temp() {
        let strategy = EntropyCooling::new(1.0);
        let base = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = GenerationState::new();
        // High entropy — well above target of 1.0
        for _ in 0..8 {
            state.update(1, 3.0);
        }
        let adjusted = strategy.adjust(&state, &base);
        assert!(
            adjusted.temperature < base.temperature,
            "expected temperature to decrease, got {}",
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
        // Low entropy — below target of 2.0
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

    #[test]
    fn repetition_adaptation_high_rep_cools() {
        let strategy = RepetitionAdaptation::new();
        let base = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = GenerationState::new();
        // Repeated same token many times
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
        // Unique tokens only
        for i in 0..5u32 {
            state.update(i, 1.0);
        }
        // rep_rate = 0 < rep_threshold/2 but step=5, heat_factor applies
        // We just verify it doesn't go below base.
        let adjusted = strategy.adjust(&state, &base);
        // Either unchanged or slightly heated — must not cool.
        assert!(
            adjusted.temperature >= base.temperature - 0.01,
            "unexpected cooling: {}",
            adjusted.temperature
        );
    }

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

    #[test]
    fn adaptive_chain_empty() {
        let chain = AdaptiveSamplerChain::new();
        let base = SamplingParams::default();
        let state = GenerationState::new();
        let adjusted = chain.adjust(&state, &base);
        assert!((adjusted.temperature - base.temperature).abs() < 1e-6);
    }

    #[test]
    fn adaptive_chain_applies_all() {
        // ScheduledDecay brings temp to 0.5 at step 50, then EntropyCooling may lower it further.
        let chain = AdaptiveSamplerChain::new()
            .add(Box::new(ScheduledDecay::new(1.0, 0.0, 100)))
            .add(Box::new(EntropyCooling::new(0.0)));

        assert_eq!(chain.len(), 2);

        let base = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = GenerationState::new();
        for _ in 0..50 {
            state.update(1, 5.0); // high entropy
        }

        let adjusted = chain.adjust(&state, &base);
        // After ScheduledDecay at step=50: temp=0.5. EntropyCooling lowers further.
        assert!(
            adjusted.temperature < 0.5 + 1e-3,
            "expected temp <= 0.5, got {}",
            adjusted.temperature
        );
    }
}
