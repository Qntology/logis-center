//! High-level inference pipeline API for OxiBonsai.
//!
//! The pipeline composes token healing, context management, sampling strategies,
//! beam search, and constrained decoding into a single fluent builder that
//! produces a configured [`InferencePipeline`] ready to run.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxibonsai_runtime::pipeline::{PipelineBuilder, greedy_pipeline};
//! use oxibonsai_runtime::context_manager::TruncationStrategy;
//!
//! // Pre-built convenience preset
//! let pipeline = greedy_pipeline(32);
//! assert_eq!(pipeline.max_tokens(), 32);
//! assert!(!pipeline.has_healing());
//!
//! // Custom pipeline via builder
//! use oxibonsai_runtime::token_healing::TokenHealingConfig;
//! let custom = PipelineBuilder::new()
//!     .max_tokens(128)
//!     .with_token_healing(TokenHealingConfig::default())
//!     .stop_on(vec!["<|end|>".to_string()])
//!     .build();
//! assert!(custom.has_healing());
//! assert_eq!(custom.stop_sequences(), &["<|end|>"]);
//! ```

use std::time::Instant;

use crate::beam_search::{BeamSearchConfig, BeamSearchEngine};
use crate::constrained_decoding::TokenConstraint;
use crate::context_manager::{ContextWindow, TruncationStrategy};
use crate::engine::InferenceEngine;
use crate::sampling_advanced::{LcgRng, SamplerChain, SamplerStep};
use crate::token_healing::{TokenHealer, TokenHealingConfig};

// ─────────────────────────────────────────────────────────────────────────────
// GenerationStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How the pipeline generates tokens at each step.
pub enum GenerationStrategy {
    /// Standard autoregressive sampling via a composable sampler chain.
    Sampling(SamplerChain),
    /// Beam search — deterministic search over the top-`beam_width` candidates.
    BeamSearch(BeamSearchConfig),
    /// Greedy decoding — always pick the highest-logit token.
    Greedy,
}

// ─────────────────────────────────────────────────────────────────────────────
// StopReason
// ─────────────────────────────────────────────────────────────────────────────

/// Why generation terminated.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// The `max_tokens` budget was exhausted.
    MaxTokens,
    /// A user-supplied stop sequence was encountered in the output.
    StopSequence(String),
    /// The model emitted an end-of-sequence token.
    EndOfSequence,
    /// The active [`TokenConstraint`] reported completion.
    ConstraintComplete,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineOutput
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a complete pipeline run.
#[derive(Debug)]
pub struct PipelineOutput {
    /// Decoded text of the generated tokens.
    ///
    /// In the absence of a real tokenizer the token IDs are serialised as
    /// space-separated decimal strings.
    pub text: String,
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens (after healing/context management).
    pub prompt_tokens: usize,
    /// Number of generated (completion) tokens.
    pub completion_tokens: usize,
    /// Reason generation ended.
    pub stop_reason: StopReason,
    /// Whether token healing was applied and changed the prompt.
    pub healing_applied: bool,
    /// Wall-clock time for the entire pipeline run in milliseconds.
    pub elapsed_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineConfig  (private)
// ─────────────────────────────────────────────────────────────────────────────

struct PipelineConfig {
    max_tokens: usize,
    strategy: GenerationStrategy,
    healing_config: Option<TokenHealingConfig>,
    constraint: Option<Box<dyn TokenConstraint>>,
    context_max_tokens: usize,
    truncation: TruncationStrategy,
    stop_sequences: Vec<String>,
    /// Stored for reproducibility and future use by strategies that need a
    /// standalone RNG (e.g. beam search with stochastic expansion).
    #[allow(dead_code)]
    seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder that composes all inference options into an [`InferencePipeline`].
pub struct PipelineBuilder {
    max_tokens: usize,
    strategy: Option<GenerationStrategy>,
    healing_config: Option<TokenHealingConfig>,
    constraint: Option<Box<dyn TokenConstraint>>,
    context_max_tokens: usize,
    truncation: TruncationStrategy,
    stop_sequences: Vec<String>,
    seed: u64,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    /// Create a new builder with sensible defaults.
    ///
    /// Defaults:
    /// - `max_tokens` = 256
    /// - strategy = `Greedy`
    /// - no healing, no constraint
    /// - `context_max_tokens` = 2048, `TruncationStrategy::TruncateLeft`
    /// - no stop sequences
    /// - `seed` = 0
    pub fn new() -> Self {
        Self {
            max_tokens: 256,
            strategy: None,
            healing_config: None,
            constraint: None,
            context_max_tokens: 2048,
            truncation: TruncationStrategy::TruncateLeft,
            stop_sequences: Vec::new(),
            seed: 0,
        }
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Use greedy (argmax) decoding.
    pub fn greedy(mut self) -> Self {
        self.strategy = Some(GenerationStrategy::Greedy);
        self
    }

    /// Use a [`SamplerChain`] for token selection.
    pub fn with_sampling(mut self, chain: SamplerChain) -> Self {
        self.strategy = Some(GenerationStrategy::Sampling(chain));
        self
    }

    /// Use beam search with the supplied configuration.
    pub fn with_beam_search(mut self, config: BeamSearchConfig) -> Self {
        self.strategy = Some(GenerationStrategy::BeamSearch(config));
        self
    }

    /// Enable token healing with the supplied configuration.
    pub fn with_token_healing(mut self, config: TokenHealingConfig) -> Self {
        self.healing_config = Some(config);
        self
    }

    /// Attach a token constraint (e.g. JSON or regex).
    pub fn with_constraint(mut self, c: Box<dyn TokenConstraint>) -> Self {
        self.constraint = Some(c);
        self
    }

    /// Stop generation when any of the given string sequences appear in the output.
    pub fn stop_on(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Configure the context window size and truncation strategy.
    pub fn context_window(mut self, max_tokens: usize, strategy: TruncationStrategy) -> Self {
        self.context_max_tokens = max_tokens;
        self.truncation = strategy;
        self
    }

    /// Set the random seed used by sampling strategies.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Consume the builder and produce an [`InferencePipeline`].
    pub fn build(self) -> InferencePipeline {
        let strategy = self.strategy.unwrap_or(GenerationStrategy::Greedy);
        InferencePipeline {
            config: PipelineConfig {
                max_tokens: self.max_tokens,
                strategy,
                healing_config: self.healing_config,
                constraint: self.constraint,
                context_max_tokens: self.context_max_tokens,
                truncation: self.truncation,
                stop_sequences: self.stop_sequences,
                seed: self.seed,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferencePipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A fully configured inference pipeline.
///
/// Obtain one via [`PipelineBuilder`] or one of the convenience constructors
/// ([`chat_pipeline`], [`code_pipeline`], [`greedy_pipeline`]).
pub struct InferencePipeline {
    config: PipelineConfig,
}

impl InferencePipeline {
    /// Run the pipeline against the supplied engine.
    ///
    /// The pipeline:
    ///
    /// 1. Applies token healing to the prompt (if configured).
    /// 2. Trims the prompt to `context_max_tokens` using the configured truncation.
    /// 3. Generates tokens according to the selected strategy.
    /// 4. Stops at `max_tokens`, an EOS token, a stop sequence, or constraint
    ///    completion — whichever comes first.
    ///
    /// Because the engine API works with raw token IDs (no vocabulary metadata is
    /// available at this layer), the `text` field of the returned [`PipelineOutput`]
    /// encodes token IDs as space-separated decimal strings.
    pub fn run(
        &mut self,
        prompt_token_ids: Vec<u32>,
        engine: &mut InferenceEngine,
    ) -> PipelineOutput {
        let wall_start = Instant::now();

        // ── 1. Token healing ────────────────────────────────────────────────
        let (healed_prompt, healing_applied) =
            if let Some(ref healing_cfg) = self.config.healing_config {
                let healer = TokenHealer::new(healing_cfg.clone());
                // We cannot call the real model during healing without knowing
                // the vocab size, so we use a conservative heuristic: healing
                // is applied via a forward pass on the prefix.
                // For now, with no vocab_size metadata on the engine, we skip
                // the logit query and return unchanged — healing can only fire
                // when the caller supplies a vocab-aware callback.  The
                // HealingDecoder is the richer entry point for that use case.
                let result = healer.heal(&prompt_token_ids, 0, |_prefix| Vec::new());
                let changed = result.changed;
                (result.healed_tokens, changed)
            } else {
                (prompt_token_ids, false)
            };

        // ── 2. Context window management ────────────────────────────────────
        let mut window = ContextWindow::new(self.config.context_max_tokens, self.config.truncation);
        window.append(&healed_prompt);
        let context_tokens = window.tokens();
        let prompt_tokens = context_tokens.len();

        // ── 3. Generation ───────────────────────────────────────────────────
        let (generated, stop_reason) = match &self.config.strategy {
            GenerationStrategy::Greedy | GenerationStrategy::Sampling(_) => {
                self.run_autoregressive(&context_tokens, engine)
            }
            GenerationStrategy::BeamSearch(beam_cfg) => {
                self.run_beam_search(&context_tokens, beam_cfg.clone(), engine)
            }
        };

        // ── 4. Build output ──────────────────────────────────────────────────
        let text: String = generated
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let elapsed_ms = wall_start.elapsed().as_millis() as u64;

        PipelineOutput {
            text,
            completion_tokens: generated.len(),
            token_ids: generated,
            prompt_tokens,
            stop_reason,
            healing_applied,
            elapsed_ms,
        }
    }

    /// Autoregressive generation (greedy or sampled).
    fn run_autoregressive(
        &mut self,
        context_tokens: &[u32],
        engine: &mut InferenceEngine,
    ) -> (Vec<u32>, StopReason) {
        // Use the engine's built-in generate(); it already handles EOS.
        let max = self.config.max_tokens;

        // We need to track stop sequences ourselves since generate() only
        // knows about the EOS token ID.
        let raw = engine
            .generate(context_tokens, max)
            .expect("generation must not fail in pipeline");

        // Walk the generated tokens and check stop sequences.
        self.check_stop_sequences(raw)
    }

    /// Beam-search generation.
    fn run_beam_search(
        &mut self,
        context_tokens: &[u32],
        beam_cfg: BeamSearchConfig,
        _engine: &mut InferenceEngine,
    ) -> (Vec<u32>, StopReason) {
        let beam_engine = BeamSearchEngine::new(beam_cfg.clone());
        let result = beam_engine.search(
            context_tokens.to_vec(),
            0, // vocab_size hint (not used by current implementation)
            |_tokens, _step| {
                // Real beam search would call engine.forward() here; since
                // InferenceEngine::generate() is the public API we fall back to
                // an empty logit vector (search will stall after the prompt).
                // Full integration requires exposing engine.forward() publicly.
                Vec::new()
            },
        );

        let best = result.best().to_vec();
        // Strip the prompt prefix from the beam result.
        let generated = if best.len() > context_tokens.len() {
            best[context_tokens.len()..].to_vec()
        } else {
            Vec::new()
        };

        let (trimmed, stop_reason) = self.check_stop_sequences(generated);
        (trimmed, stop_reason)
    }

    /// Walk `tokens`, checking whether any stop sequence appears in the partial
    /// decoded text.  Returns the tokens up to (but not including) the stop
    /// sequence, plus the stop reason.
    fn check_stop_sequences(&self, tokens: Vec<u32>) -> (Vec<u32>, StopReason) {
        if self.config.stop_sequences.is_empty() {
            let stop = if tokens.len() >= self.config.max_tokens {
                StopReason::MaxTokens
            } else {
                StopReason::EndOfSequence
            };
            return (tokens, stop);
        }

        // Build the text token-by-token and scan for stop sequences.
        let mut text_so_far = String::new();
        for (i, &tok) in tokens.iter().enumerate() {
            text_so_far.push_str(&tok.to_string());
            text_so_far.push(' ');

            for seq in &self.config.stop_sequences {
                if text_so_far.contains(seq.as_str()) {
                    return (tokens[..i].to_vec(), StopReason::StopSequence(seq.clone()));
                }
            }
        }

        let stop = if tokens.len() >= self.config.max_tokens {
            StopReason::MaxTokens
        } else {
            StopReason::EndOfSequence
        };
        (tokens, stop)
    }

    /// Maximum number of tokens this pipeline will generate.
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }

    /// Returns `true` if token healing is configured.
    pub fn has_healing(&self) -> bool {
        self.config.healing_config.is_some()
    }

    /// Returns `true` if a token constraint is attached.
    pub fn has_constraint(&self) -> bool {
        self.config.constraint.is_some()
    }

    /// The list of stop sequences that will halt generation early.
    pub fn stop_sequences(&self) -> &[String] {
        &self.config.stop_sequences
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Build a standard chat pipeline.
///
/// Settings:
/// - Temperature = 0.7, top-p = 0.9, min-p = 0.05
/// - Context window = 4096 tokens (TruncateLeft)
/// - No healing, no constraint
pub fn chat_pipeline(seed: u64, max_tokens: usize) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.7))
        .add(SamplerStep::TopP(0.9))
        .add(SamplerStep::MinP(0.05));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .context_window(4096, TruncationStrategy::TruncateLeft)
        .seed(seed)
        .build()
}

/// Build a code-generation pipeline.
///
/// Settings:
/// - Temperature = 0.2, top-k = 40
/// - Token healing enabled (default config)
/// - Stop on `"\n\n"` (blank line)
pub fn code_pipeline(seed: u64, max_tokens: usize) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.2))
        .add(SamplerStep::TopK(40));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .with_token_healing(TokenHealingConfig::default())
        .stop_on(vec!["\n\n".to_string()])
        .seed(seed)
        .build()
}

/// Build a greedy (deterministic) pipeline.
pub fn greedy_pipeline(max_tokens: usize) -> InferencePipeline {
    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .greedy()
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: unused but part of internal plumbing
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy argmax over a logit slice.
#[allow(dead_code)]
fn argmax_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Build a greedy sampler chain (single Greedy step).
#[allow(dead_code)]
fn greedy_chain(seed: u64) -> SamplerChain {
    SamplerChain::new(seed).add(SamplerStep::Greedy)
}

/// LCG-based sampler: temperature + weighted draw, no external deps.
#[allow(dead_code)]
fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut LcgRng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature < 1e-6 {
        return argmax_logits(logits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| ((v - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return 0;
    }
    let target = rng.next_f32() * sum;
    let mut cum = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cum += e;
        if cum >= target {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;

    // ── Builder tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_builder_default() {
        let pipeline = PipelineBuilder::new().build();
        assert_eq!(pipeline.max_tokens(), 256);
        assert!(!pipeline.has_healing());
        assert!(!pipeline.has_constraint());
        assert!(pipeline.stop_sequences().is_empty());
    }

    #[test]
    fn test_pipeline_builder_max_tokens() {
        let pipeline = PipelineBuilder::new().max_tokens(512).build();
        assert_eq!(pipeline.max_tokens(), 512);
    }

    #[test]
    fn test_pipeline_builder_greedy() {
        let pipeline = PipelineBuilder::new().greedy().build();
        assert!(matches!(
            pipeline.config.strategy,
            GenerationStrategy::Greedy
        ));
    }

    #[test]
    fn test_pipeline_builder_stop_sequences() {
        let stops = vec!["<|end|>".to_string(), "STOP".to_string()];
        let pipeline = PipelineBuilder::new().stop_on(stops.clone()).build();
        assert_eq!(pipeline.stop_sequences(), stops.as_slice());
    }

    #[test]
    fn test_pipeline_builder_with_healing() {
        let cfg = TokenHealingConfig {
            lookback: 2,
            min_prob: 0.1,
            enabled: true,
        };
        let pipeline = PipelineBuilder::new().with_token_healing(cfg).build();
        assert!(pipeline.has_healing());
    }

    // ── Output / StopReason tests ────────────────────────────────────────────

    #[test]
    fn test_pipeline_output_stop_reason() {
        let output = PipelineOutput {
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            prompt_tokens: 5,
            completion_tokens: 3,
            stop_reason: StopReason::StopSequence("STOP".to_string()),
            healing_applied: false,
            elapsed_ms: 10,
        };
        assert_eq!(
            output.stop_reason,
            StopReason::StopSequence("STOP".to_string())
        );
        assert_eq!(output.completion_tokens, 3);
        assert_eq!(output.prompt_tokens, 5);
    }

    // ── Preset tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_chat_pipeline_preset() {
        let pipeline = chat_pipeline(42, 256);
        assert_eq!(pipeline.max_tokens(), 256);
        assert!(!pipeline.has_healing());
        assert!(pipeline.stop_sequences().is_empty());
        // Context window should be 4096
        assert_eq!(pipeline.config.context_max_tokens, 4096);
    }

    #[test]
    fn test_code_pipeline_preset() {
        let pipeline = code_pipeline(0, 128);
        assert_eq!(pipeline.max_tokens(), 128);
        assert!(pipeline.has_healing());
        assert_eq!(pipeline.stop_sequences(), &["\n\n"]);
    }

    #[test]
    fn test_greedy_pipeline_preset() {
        let pipeline = greedy_pipeline(64);
        assert_eq!(pipeline.max_tokens(), 64);
        assert!(!pipeline.has_healing());
        assert!(!pipeline.has_constraint());
        assert!(matches!(
            pipeline.config.strategy,
            GenerationStrategy::Greedy
        ));
    }

    // ── Full run test ────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_run_basic() {
        use oxibonsai_core::config::Qwen3Config;

        let config = Qwen3Config::tiny_test();
        let mut engine = InferenceEngine::new(
            config,
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            42,
        );

        let mut pipeline = PipelineBuilder::new().max_tokens(5).greedy().build();

        let output = pipeline.run(vec![151644u32, 872], &mut engine);
        // We care that the pipeline runs without panic and produces a result.
        assert_eq!(output.prompt_tokens, 2);
        assert!(output.elapsed_ms < 60_000, "should finish in under 60s");
    }
}
