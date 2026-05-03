//! Token healing for partial-token prompt repair.
//!
//! When a prompt ends in the middle of a token boundary, the model is biased
//! toward completing that token rather than exploring alternatives. Token healing
//! backs up `lookback` tokens, regenerates from the valid prefix, and splices
//! the result — producing more natural continuations.
//!
//! ## Algorithm
//!
//! 1. Strip the last `lookback` tokens from the prompt to form a *prefix*.
//! 2. Call a user-supplied `get_logits` closure on the prefix.
//! 3. Select `t*` = argmax of the returned logit vector.
//! 4. If `t*` equals the original next token, no change is needed.
//! 5. Otherwise replace those `lookback` tokens with `[t*]` — the healed sequence.
//!
//! ## Example
//!
//! ```rust
//! use oxibonsai_runtime::token_healing::{TokenHealer, TokenHealingConfig};
//!
//! let healer = TokenHealer::new(TokenHealingConfig::default());
//! let tokens = vec![10u32, 20, 99]; // 99 might be a mid-word continuation
//!
//! let result = healer.heal(&tokens, 128, |prefix| {
//!     // Mock: always prefer token 42 as the next token
//!     let mut logits = vec![0.0f32; 128];
//!     logits[42] = 10.0;
//!     logits
//! });
//!
//! // token 42 != 99, so healing changed the sequence
//! assert!(result.was_healed());
//! assert_eq!(result.healed_tokens.last().copied(), Some(42));
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the token healing pass.
#[derive(Debug, Clone)]
pub struct TokenHealingConfig {
    /// Number of tokens to back up and re-score.
    ///
    /// A value of `1` (the default) is sufficient for the vast majority of
    /// tokenisation schemes. Larger values provide wider context but are slower.
    pub lookback: usize,

    /// Minimum probability that a healed token must have to be accepted.
    ///
    /// If the best candidate falls below `min_prob`, healing is skipped and
    /// the original sequence is returned unchanged.
    pub min_prob: f32,

    /// Master switch. When `false` the healer is a no-op.
    pub enabled: bool,
}

impl Default for TokenHealingConfig {
    fn default() -> Self {
        Self {
            lookback: 1,
            min_prob: 0.0,
            enabled: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HealingResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by [`TokenHealer::heal`].
#[derive(Debug, Clone)]
pub struct HealingResult {
    /// The token sequence supplied to [`TokenHealer::heal`] (before any change).
    pub original_tokens: Vec<u32>,
    /// The token sequence after healing.  Equal to `original_tokens` when unchanged.
    pub healed_tokens: Vec<u32>,
    /// How many trailing tokens were backed up and re-scored.
    pub tokens_healed: usize,
    /// `true` iff the healed sequence differs from the original.
    pub changed: bool,
}

impl HealingResult {
    /// Construct a result that records no change.
    pub fn unchanged(tokens: Vec<u32>) -> Self {
        Self {
            healed_tokens: tokens.clone(),
            original_tokens: tokens,
            tokens_healed: 0,
            changed: false,
        }
    }

    /// Returns `true` when the healer actually changed the sequence.
    pub fn was_healed(&self) -> bool {
        self.changed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenHealer
// ─────────────────────────────────────────────────────────────────────────────

/// Backs up `lookback` tokens and re-scores from the prefix using the
/// caller-supplied logit function.
pub struct TokenHealer {
    config: TokenHealingConfig,
}

impl TokenHealer {
    /// Create a new healer with the supplied configuration.
    pub fn new(config: TokenHealingConfig) -> Self {
        Self { config }
    }

    /// Convenience constructor — use all defaults but override `lookback`.
    pub fn with_lookback(lookback: usize) -> Self {
        Self::new(TokenHealingConfig {
            lookback,
            ..TokenHealingConfig::default()
        })
    }

    /// Apply token healing to `tokens`.
    ///
    /// `get_logits` receives a prefix slice and returns raw (unnormalized) logits
    /// over the vocabulary.  The closure is called at most once.
    ///
    /// Returns a [`HealingResult`] describing what (if anything) changed.
    pub fn heal<F>(&self, tokens: &[u32], vocab_size: usize, mut get_logits: F) -> HealingResult
    where
        F: FnMut(&[u32]) -> Vec<f32>,
    {
        // Short-circuit: disabled or not enough tokens to back up.
        if !self.config.enabled || tokens.len() <= self.config.lookback {
            return HealingResult::unchanged(tokens.to_vec());
        }

        let split = tokens.len() - self.config.lookback;
        let prefix = &tokens[..split];
        let logits = get_logits(prefix);

        if logits.is_empty() || logits.len() < vocab_size {
            // Cannot score — return unchanged rather than panicking.
            return HealingResult::unchanged(tokens.to_vec());
        }

        // Find the highest-scoring token.
        let best_token = argmax_f32(&logits) as u32;

        // Check min_prob gate.
        let prob = Self::token_prob(&logits, best_token);
        if prob < self.config.min_prob {
            return HealingResult::unchanged(tokens.to_vec());
        }

        // If best token already matches what was there, no change needed.
        if best_token == tokens[split] {
            return HealingResult {
                original_tokens: tokens.to_vec(),
                healed_tokens: tokens.to_vec(),
                tokens_healed: self.config.lookback,
                changed: false,
            };
        }

        // Build the healed sequence: prefix + [best_token]
        let mut healed = prefix.to_vec();
        healed.push(best_token);

        HealingResult {
            original_tokens: tokens.to_vec(),
            healed_tokens: healed,
            tokens_healed: self.config.lookback,
            changed: true,
        }
    }

    /// Heuristic: returns `true` when `token_text` looks like a continuation
    /// of `prev_token_text` (i.e., no leading whitespace and `prev_token_text`
    /// ends mid-word).
    ///
    /// This is a lightweight signal used to decide whether healing is semantically
    /// meaningful.  It does not affect the heal algorithm itself.
    pub fn is_continuation_token(prev_token_text: &str, token_text: &str) -> bool {
        if token_text.is_empty() || prev_token_text.is_empty() {
            return false;
        }
        // The next token is a continuation if it does NOT start with whitespace.
        let next_starts_clean = !token_text.starts_with(' ');
        // The previous token ends mid-word (last char is alphanumeric).
        let prev_ends_mid_word = prev_token_text
            .chars()
            .next_back()
            .map(|c| c.is_alphanumeric())
            .unwrap_or(false);
        prev_ends_mid_word && next_starts_clean
    }

    /// Compute the probability of `token_id` under the softmax of `logits`.
    ///
    /// Returns `0.0` when `token_id` is out of range or `logits` is empty.
    pub fn token_prob(logits: &[f32], token_id: u32) -> f32 {
        let idx = token_id as usize;
        if logits.is_empty() || idx >= logits.len() {
            return 0.0;
        }
        // Numerically stable softmax.
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        exps[idx] / sum
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HealingDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// Combines token healing with a simple token-by-token generation loop.
///
/// Healing is applied once to the prompt; then `max_tokens` additional tokens
/// are drawn using the `sample` closure.
pub struct HealingDecoder {
    /// The inner healer driving the healing step.
    pub healer: TokenHealer,
}

impl HealingDecoder {
    /// Create a new decoder with the supplied healing configuration.
    pub fn new(config: TokenHealingConfig) -> Self {
        Self {
            healer: TokenHealer::new(config),
        }
    }

    /// Apply token healing to `prompt_tokens`, then generate up to `max_tokens`
    /// additional tokens.
    ///
    /// # Parameters
    ///
    /// - `get_logits` — called with the current token sequence; returns logits.
    /// - `sample`     — called with the raw logits; returns the next token id.
    ///
    /// # Returns
    ///
    /// A pair `(HealingResult, generated_tokens)`.
    pub fn generate<F, G>(
        &self,
        prompt_tokens: Vec<u32>,
        vocab_size: usize,
        max_tokens: usize,
        mut get_logits: F,
        mut sample: G,
    ) -> (HealingResult, Vec<u32>)
    where
        F: FnMut(&[u32]) -> Vec<f32>,
        G: FnMut(Vec<f32>) -> u32,
    {
        // Phase 1: heal the prompt.
        let healing = self
            .healer
            .heal(&prompt_tokens, vocab_size, &mut get_logits);
        let healed_prompt = healing.healed_tokens.clone();

        // Phase 2: generate up to max_tokens from the (possibly healed) prompt.
        let mut context = healed_prompt.clone();
        let mut generated = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let logits = get_logits(&context);
            if logits.is_empty() {
                break;
            }
            let next_token = sample(logits);
            context.push(next_token);
            generated.push(next_token);
        }

        (healing, generated)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the index of the maximum value in `values`.
/// Returns `0` for empty slices (safe default).
fn argmax_f32(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a logit vector where `winner` has a high score.
    fn logits_prefer(vocab_size: usize, winner: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab_size];
        v[winner] = 100.0;
        v
    }

    #[test]
    fn test_token_healing_disabled_returns_unchanged() {
        let config = TokenHealingConfig {
            enabled: false,
            ..TokenHealingConfig::default()
        };
        let healer = TokenHealer::new(config);
        let tokens = vec![1u32, 2, 3, 4];
        let result = healer.heal(&tokens, 10, |_| logits_prefer(10, 7));
        assert!(!result.changed);
        assert_eq!(result.healed_tokens, tokens);
        assert_eq!(result.original_tokens, tokens);
    }

    #[test]
    fn test_token_healing_empty_input_unchanged() {
        let healer = TokenHealer::new(TokenHealingConfig::default());
        let result = healer.heal(&[], 10, |_| logits_prefer(10, 0));
        assert!(!result.changed);
        assert!(result.healed_tokens.is_empty());
    }

    #[test]
    fn test_token_healing_lookback_1_no_change_when_correct() {
        // The best logit token IS the last token in the sequence → no change.
        let healer = TokenHealer::new(TokenHealingConfig::default());
        let tokens = vec![10u32, 20, 5]; // last token = 5
        let result = healer.heal(&tokens, 30, |_| logits_prefer(30, 5));
        assert!(
            !result.changed,
            "no change expected when prediction matches"
        );
        assert_eq!(result.healed_tokens, tokens);
        assert_eq!(result.tokens_healed, 1);
    }

    #[test]
    fn test_token_healing_lookback_1_changes_wrong_token() {
        // Best logit token (7) differs from last token (99) → healing fires.
        let healer = TokenHealer::new(TokenHealingConfig::default());
        let tokens = vec![10u32, 20, 99];
        let result = healer.heal(&tokens, 128, |_| logits_prefer(128, 7));
        assert!(result.changed);
        assert!(result.was_healed());
        // Healed sequence = prefix [10, 20] + [7]
        assert_eq!(result.healed_tokens, vec![10u32, 20, 7]);
        assert_eq!(result.original_tokens, tokens);
        assert_eq!(result.tokens_healed, 1);
    }

    #[test]
    fn test_token_prob_correct() {
        // With one dominant logit the probability of that token should be ≈ 1.
        let mut logits = vec![0.0f32; 10];
        logits[3] = 100.0;
        let p = TokenHealer::token_prob(&logits, 3);
        assert!(
            (p - 1.0).abs() < 1e-5,
            "dominant token should have prob ≈ 1"
        );

        // Uniform logits → all tokens should have prob ≈ 1/n.
        let uniform = vec![0.0f32; 4];
        let p_uniform = TokenHealer::token_prob(&uniform, 2);
        assert!(
            (p_uniform - 0.25).abs() < 1e-5,
            "uniform prob should be 0.25"
        );
    }

    #[test]
    fn test_healing_result_unchanged() {
        let tokens = vec![1u32, 2, 3];
        let result = HealingResult::unchanged(tokens.clone());
        assert!(!result.changed);
        assert!(!result.was_healed());
        assert_eq!(result.original_tokens, tokens);
        assert_eq!(result.healed_tokens, tokens);
        assert_eq!(result.tokens_healed, 0);
    }

    #[test]
    fn test_healing_decoder_runs() {
        let decoder = HealingDecoder::new(TokenHealingConfig::default());
        let prompt = vec![1u32, 2, 3]; // last token = 3; best = 9 → healing fires
        let vocab_size = 20;
        let max_tokens = 5;

        let call_count = std::cell::Cell::new(0usize);
        let get_logits = |_prefix: &[u32]| {
            call_count.set(call_count.get() + 1);
            logits_prefer(vocab_size, 9)
        };
        // sample always returns token 1
        let sample = |_logits: Vec<f32>| 1u32;

        let (healing, generated) =
            decoder.generate(prompt, vocab_size, max_tokens, get_logits, sample);
        // Healing should have fired (best=9, last was 3).
        assert!(healing.changed);
        // Exactly max_tokens tokens generated.
        assert_eq!(generated.len(), max_tokens);
        // All generated tokens are 1 (from our mock sampler).
        assert!(generated.iter().all(|&t| t == 1));
    }

    #[test]
    fn test_is_continuation_token() {
        // "ing" follows "call" — mid-word continuation.
        assert!(
            TokenHealer::is_continuation_token("call", "ing"),
            "\"calling\" split should be a continuation"
        );
        // " the" after "call" — new word, NOT a continuation.
        assert!(
            !TokenHealer::is_continuation_token("call", " the"),
            "space-prefixed token is not a continuation"
        );
        // Empty inputs → not a continuation.
        assert!(!TokenHealer::is_continuation_token("", "ing"));
        assert!(!TokenHealer::is_continuation_token("call", ""));
        // Punctuation ending the previous token → not mid-word.
        assert!(
            !TokenHealer::is_continuation_token("call.", "ing"),
            "period-ended token is not mid-word"
        );
    }
}
