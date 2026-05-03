//! Token budget management: enforce per-request and global token limits.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use thiserror::Error;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors raised by budget enforcement.
#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("prompt tokens {prompt} exceeds max_prompt_tokens {max}")]
    PromptTooLong { prompt: usize, max: usize },
    #[error("completion token budget exhausted (limit = {limit})")]
    CompletionBudgetExhausted { limit: usize },
    #[error("total token budget exhausted (limit = {limit}, used = {used})")]
    TotalBudgetExhausted { limit: usize, used: usize },
}

// ─── Policy ───────────────────────────────────────────────────────────────────

/// Action taken when a budget limit is reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetPolicy {
    /// Stop generation cleanly.
    StopGeneration,
    /// Truncate the oldest context.
    TruncateContext,
    /// Return an error.
    ReturnError,
}

// ─── Budget configuration ─────────────────────────────────────────────────────

/// Configuration for token budget enforcement.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Maximum tokens allowed in the prompt.
    pub max_prompt_tokens: Option<usize>,
    /// Maximum tokens to generate (completion only).
    pub max_completion_tokens: Option<usize>,
    /// Maximum prompt + completion tokens combined.
    pub max_total_tokens: Option<usize>,
    /// Policy to apply when a limit is breached.
    pub policy: BudgetPolicy,
}

impl BudgetConfig {
    /// Create a config with no limits and the default policy (`StopGeneration`).
    pub fn new() -> Self {
        Self {
            max_prompt_tokens: None,
            max_completion_tokens: None,
            max_total_tokens: None,
            policy: BudgetPolicy::StopGeneration,
        }
    }

    /// Set the maximum completion tokens.
    pub fn with_max_completion(mut self, n: usize) -> Self {
        self.max_completion_tokens = Some(n);
        self
    }

    /// Set the maximum total (prompt + completion) tokens.
    pub fn with_max_total(mut self, n: usize) -> Self {
        self.max_total_tokens = Some(n);
        self
    }

    /// Override the enforcement policy.
    pub fn with_policy(mut self, policy: BudgetPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Convenience: no limits whatsoever.
    pub fn unlimited() -> Self {
        Self::new()
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Per-request budget ───────────────────────────────────────────────────────

/// Per-request token budget tracker.
#[derive(Debug)]
pub struct RequestBudget {
    config: BudgetConfig,
    prompt_tokens: usize,
    completion_tokens: usize,
}

impl RequestBudget {
    /// Create a new `RequestBudget`, validating the initial prompt length.
    ///
    /// Returns `Err(BudgetError::PromptTooLong)` if `prompt_tokens` exceeds
    /// `config.max_prompt_tokens`.
    pub fn new(config: BudgetConfig, prompt_tokens: usize) -> Result<Self, BudgetError> {
        if let Some(max) = config.max_prompt_tokens {
            if prompt_tokens > max {
                return Err(BudgetError::PromptTooLong {
                    prompt: prompt_tokens,
                    max,
                });
            }
        }
        Ok(Self {
            config,
            prompt_tokens,
            completion_tokens: 0,
        })
    }

    /// Record one generated token.
    ///
    /// Returns an error (according to the configured policy) if any limit is
    /// exceeded after recording the token.
    pub fn record_token(&mut self) -> Result<(), BudgetError> {
        self.record_tokens(1)
    }

    /// Record `n` generated tokens.
    ///
    /// Limits are checked after adding `n`.
    pub fn record_tokens(&mut self, n: usize) -> Result<(), BudgetError> {
        self.completion_tokens = self.completion_tokens.saturating_add(n);

        // Check completion limit.
        if let Some(limit) = self.config.max_completion_tokens {
            if self.completion_tokens > limit {
                return Err(BudgetError::CompletionBudgetExhausted { limit });
            }
        }

        // Check total limit.
        if let Some(limit) = self.config.max_total_tokens {
            let used = self.total_tokens();
            if used > limit {
                return Err(BudgetError::TotalBudgetExhausted { limit, used });
            }
        }

        Ok(())
    }

    /// Number of tokens in the prompt.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Number of tokens generated so far.
    pub fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    /// Prompt + completion tokens.
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// How many more completion tokens can be generated, or `None` if unlimited.
    pub fn remaining_completion_tokens(&self) -> Option<usize> {
        self.config
            .max_completion_tokens
            .map(|limit| limit.saturating_sub(self.completion_tokens))
    }

    /// Whether any configured budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        if let Some(limit) = self.config.max_completion_tokens {
            if self.completion_tokens >= limit {
                return true;
            }
        }
        if let Some(limit) = self.config.max_total_tokens {
            if self.total_tokens() >= limit {
                return true;
            }
        }
        false
    }

    /// The policy that governs how exhaustion is handled.
    pub fn policy(&self) -> BudgetPolicy {
        self.config.policy
    }
}

// ─── Global token budget ──────────────────────────────────────────────────────

/// Global token budget shared across requests via an `Arc<AtomicU64>`.
pub struct GlobalTokenBudget {
    total_tokens_used: Arc<AtomicU64>,
    max_tokens: Option<u64>,
}

impl GlobalTokenBudget {
    /// Create a global budget with an optional hard cap.
    pub fn new(max_tokens: Option<u64>) -> Self {
        Self {
            total_tokens_used: Arc::new(AtomicU64::new(0)),
            max_tokens,
        }
    }

    /// Convenience: no cap.
    pub fn unlimited() -> Self {
        Self::new(None)
    }

    /// Add `tokens` to the global counter.
    pub fn record(&self, tokens: u64) {
        self.total_tokens_used.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Total tokens consumed so far.
    pub fn total_used(&self) -> u64 {
        self.total_tokens_used.load(Ordering::Relaxed)
    }

    /// How many tokens remain before the cap, or `None` if unlimited.
    pub fn remaining(&self) -> Option<u64> {
        self.max_tokens
            .map(|cap| cap.saturating_sub(self.total_used()))
    }

    /// Whether the global cap has been reached.
    pub fn is_exhausted(&self) -> bool {
        match self.max_tokens {
            None => false,
            Some(cap) => self.total_used() >= cap,
        }
    }

    /// Fraction of the cap consumed (`total_used / max_tokens`), or `None`
    /// if the budget is unlimited.
    pub fn utilization(&self) -> Option<f32> {
        self.max_tokens.map(|cap| {
            if cap == 0 {
                1.0
            } else {
                self.total_used() as f32 / cap as f32
            }
        })
    }
}

// ─── Cost estimator ───────────────────────────────────────────────────────────

/// Estimated monetary cost for a request (for billing / logging).
#[derive(Debug, Clone)]
pub struct TokenCostEstimate {
    /// Tokens in the prompt.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub completion_tokens: usize,
    /// Cost for the prompt portion.
    pub prompt_cost: f64,
    /// Cost for the completion portion.
    pub completion_cost: f64,
    /// `prompt_cost + completion_cost`.
    pub total_cost: f64,
}

impl TokenCostEstimate {
    /// Compute cost given per-1 000-token rates for prompt and completion.
    pub fn compute(
        prompt_tokens: usize,
        completion_tokens: usize,
        prompt_cost_per_1k: f64,
        completion_cost_per_1k: f64,
    ) -> Self {
        let prompt_cost = prompt_tokens as f64 / 1_000.0 * prompt_cost_per_1k;
        let completion_cost = completion_tokens as f64 / 1_000.0 * completion_cost_per_1k;
        let total_cost = prompt_cost + completion_cost;
        Self {
            prompt_tokens,
            completion_tokens,
            prompt_cost,
            completion_cost,
            total_cost,
        }
    }

    /// Human-readable summary of the cost breakdown.
    pub fn summary(&self) -> String {
        format!(
            "tokens: prompt={} completion={} | cost: prompt=${:.6} completion=${:.6} total=${:.6}",
            self.prompt_tokens,
            self.completion_tokens,
            self.prompt_cost,
            self.completion_cost,
            self.total_cost,
        )
    }
}
