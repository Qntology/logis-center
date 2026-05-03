//! Builder pattern for constructing [`Qwen3Config`] values.
//!
//! [`ModelConfigBuilder`] provides a fluent, ergonomic API for assembling
//! custom model configurations without needing to fill every field manually.
//! All fields default to the Bonsai-8B values when not explicitly set so
//! that partial configurations remain valid.
//!
//! # Examples
//!
//! ```rust
//! use oxibonsai_model::model_config_builder::ModelConfigBuilder;
//!
//! // Build a custom tiny config for unit tests
//! let config = ModelConfigBuilder::build_tiny();
//! assert_eq!(config.num_layers, 2);
//!
//! // Build a custom config with the builder API
//! let config = ModelConfigBuilder::new()
//!     .layers(8)
//!     .hidden_size(512)
//!     .num_attention_heads(8)
//!     .num_kv_heads(2)
//!     .intermediate_size(1024)
//!     .vocab_size(1000)
//!     .max_position_embeddings(2048)
//!     .rope_freq_base(10_000.0)
//!     .rms_norm_eps(1e-6)
//!     .build()
//!     .expect("valid config");
//!
//! assert_eq!(config.num_layers, 8);
//! assert_eq!(config.hidden_size, 512);
//! ```

use oxibonsai_core::config::Qwen3Config;

// ─── ConfigError ─────────────────────────────────────────────────────────────

/// Error produced when a [`ModelConfigBuilder`] constraint is violated.
///
/// Carries a human-readable description of the violated constraint so callers
/// can surface actionable diagnostics without needing pattern-matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid model configuration: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl ConfigError {
    /// Construct a new error with the provided message.
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

// ─── ModelConfigBuilder ───────────────────────────────────────────────────────

/// Fluent builder for [`Qwen3Config`].
///
/// Each setter consumes the builder by value to enable method-chaining without
/// requiring mutable references. Call [`build`][ModelConfigBuilder::build] to
/// validate the accumulated settings and produce a [`Qwen3Config`].
///
/// Fields left unset are filled from the Bonsai-8B defaults.
#[derive(Debug, Default, Clone)]
pub struct ModelConfigBuilder {
    num_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_kv_heads: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_freq_base: Option<f32>,
    rms_norm_eps: Option<f32>,
    architecture: Option<String>,
    model_name: Option<String>,
}

impl ModelConfigBuilder {
    /// Create a new builder with all fields unset (will use Bonsai-8B defaults).
    pub fn new() -> Self {
        Self::default()
    }

    // ── Setters ──────────────────────────────────────────────────────────────

    /// Set the number of Transformer layers.
    pub fn layers(mut self, n: usize) -> Self {
        self.num_layers = Some(n);
        self
    }

    /// Set the hidden (embedding) dimension.
    pub fn hidden_size(mut self, n: usize) -> Self {
        self.hidden_size = Some(n);
        self
    }

    /// Set the number of query attention heads.
    pub fn num_attention_heads(mut self, n: usize) -> Self {
        self.num_attention_heads = Some(n);
        self
    }

    /// Set the number of key-value heads (for Grouped Query Attention).
    pub fn num_kv_heads(mut self, n: usize) -> Self {
        self.num_kv_heads = Some(n);
        self
    }

    /// Set the intermediate (FFN / SwiGLU) size.
    pub fn intermediate_size(mut self, n: usize) -> Self {
        self.intermediate_size = Some(n);
        self
    }

    /// Set the vocabulary size.
    pub fn vocab_size(mut self, n: usize) -> Self {
        self.vocab_size = Some(n);
        self
    }

    /// Set the maximum position embedding length (= maximum context length).
    pub fn max_position_embeddings(mut self, n: usize) -> Self {
        self.max_position_embeddings = Some(n);
        self
    }

    /// Set the RoPE frequency base (theta).
    pub fn rope_freq_base(mut self, f: f32) -> Self {
        self.rope_freq_base = Some(f);
        self
    }

    /// Set the RMSNorm epsilon.
    pub fn rms_norm_eps(mut self, f: f32) -> Self {
        self.rms_norm_eps = Some(f);
        self
    }

    /// Override the architecture tag stored in the resulting config.
    pub fn architecture(mut self, s: impl Into<String>) -> Self {
        self.architecture = Some(s.into());
        self
    }

    /// Override the model name stored in the resulting config.
    pub fn model_name(mut self, s: impl Into<String>) -> Self {
        self.model_name = Some(s.into());
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Validate the accumulated settings and produce a [`Qwen3Config`].
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when any of the following constraints are violated:
    ///
    /// | Constraint | Reason |
    /// |------------|--------|
    /// | `num_layers >= 1` | A transformer with zero layers is meaningless |
    /// | `hidden_size >= 1` | Zero-dimensional embeddings are invalid |
    /// | `num_attention_heads >= 1` | At least one query head is required |
    /// | `num_kv_heads >= 1` | At least one KV head is required |
    /// | `hidden_size` divisible by `num_attention_heads` | Needed for equal head_dim split |
    /// | `num_attention_heads` divisible by `num_kv_heads` | GQA requirement |
    /// | `intermediate_size >= 1` | FFN must have positive width |
    /// | `vocab_size >= 2` | At least 2 tokens needed for meaningful output |
    /// | `max_position_embeddings >= 1` | Context must be at least 1 token |
    /// | `rope_freq_base > 0` | Must be a positive real number |
    /// | `rms_norm_eps > 0` | Epsilon must be strictly positive |
    pub fn build(self) -> Result<Qwen3Config, ConfigError> {
        // Merge with defaults
        let defaults = Qwen3Config::bonsai_8b();

        let num_layers = self.num_layers.unwrap_or(defaults.num_layers);
        let hidden_size = self.hidden_size.unwrap_or(defaults.hidden_size);
        let num_attention_heads = self
            .num_attention_heads
            .unwrap_or(defaults.num_attention_heads);
        let num_kv_heads = self.num_kv_heads.unwrap_or(defaults.num_kv_heads);
        let intermediate_size = self.intermediate_size.unwrap_or(defaults.intermediate_size);
        let vocab_size = self.vocab_size.unwrap_or(defaults.vocab_size);
        let max_context_length = self
            .max_position_embeddings
            .unwrap_or(defaults.max_context_length);
        let rope_freq_base = self.rope_freq_base.unwrap_or(defaults.rope_freq_base);
        let rms_norm_eps = self.rms_norm_eps.unwrap_or(defaults.rms_norm_eps);
        let architecture = self
            .architecture
            .unwrap_or_else(|| defaults.architecture.clone());
        let model_name = self
            .model_name
            .unwrap_or_else(|| defaults.model_name.clone());

        // ── Validation ────────────────────────────────────────────────────────

        if num_layers == 0 {
            return Err(ConfigError::new("num_layers must be >= 1"));
        }
        if hidden_size == 0 {
            return Err(ConfigError::new("hidden_size must be >= 1"));
        }
        if num_attention_heads == 0 {
            return Err(ConfigError::new("num_attention_heads must be >= 1"));
        }
        if num_kv_heads == 0 {
            return Err(ConfigError::new("num_kv_heads must be >= 1"));
        }
        if hidden_size % num_attention_heads != 0 {
            return Err(ConfigError::new(format!(
                "hidden_size ({hidden_size}) must be divisible by num_attention_heads \
                 ({num_attention_heads})"
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(ConfigError::new(format!(
                "num_attention_heads ({num_attention_heads}) must be divisible by \
                 num_kv_heads ({num_kv_heads}) for Grouped Query Attention"
            )));
        }
        if intermediate_size == 0 {
            return Err(ConfigError::new("intermediate_size must be >= 1"));
        }
        if vocab_size < 2 {
            return Err(ConfigError::new("vocab_size must be >= 2"));
        }
        if max_context_length == 0 {
            return Err(ConfigError::new("max_position_embeddings must be >= 1"));
        }
        if rope_freq_base <= 0.0 || rope_freq_base.is_nan() || rope_freq_base.is_infinite() {
            return Err(ConfigError::new(format!(
                "rope_freq_base must be a finite positive number, got {rope_freq_base}"
            )));
        }
        if rms_norm_eps <= 0.0 || rms_norm_eps.is_nan() || rms_norm_eps.is_infinite() {
            return Err(ConfigError::new(format!(
                "rms_norm_eps must be a finite positive number, got {rms_norm_eps}"
            )));
        }

        let head_dim = hidden_size / num_attention_heads;

        Ok(Qwen3Config {
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_context_length,
            rms_norm_eps,
            rope_freq_base,
            architecture,
            model_name,
        })
    }

    // ── Convenience constructors ──────────────────────────────────────────────

    /// Produce a minimal valid [`Qwen3Config`] suitable for fast unit tests.
    ///
    /// Uses the same parameters as [`Qwen3Config::tiny_test`] so that tests
    /// written against this builder's output are directly comparable to tests
    /// using the core crate's constant.
    pub fn build_tiny() -> Qwen3Config {
        // Uses exact same values as Qwen3Config::tiny_test() for consistency.
        // We bypass the builder to avoid any possibility of validation failure
        // in a shared test helper.
        Qwen3Config::tiny_test()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Happy-path: valid builds ──────────────────────────────────────────────

    #[test]
    fn build_with_all_defaults_succeeds() {
        let config = ModelConfigBuilder::new()
            .build()
            .expect("default builder should succeed");
        // Defaults match Bonsai-8B
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn build_custom_small_config_succeeds() {
        let config = ModelConfigBuilder::new()
            .layers(4)
            .hidden_size(256)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(512)
            .vocab_size(100)
            .max_position_embeddings(512)
            .rope_freq_base(10_000.0)
            .rms_norm_eps(1e-5)
            .build()
            .expect("small valid config should build");

        assert_eq!(config.num_layers, 4);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.max_context_length, 512);
        assert!((config.rope_freq_base - 10_000.0).abs() < 1.0);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        // Derived field
        assert_eq!(config.head_dim, 64); // 256 / 4
    }

    #[test]
    fn build_tiny_returns_valid_config() {
        let config = ModelConfigBuilder::build_tiny();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        // head_dim derived correctly
        assert_eq!(config.head_dim, 16);
    }

    #[test]
    fn architecture_and_model_name_setters_work() {
        let config = ModelConfigBuilder::new()
            .layers(2)
            .hidden_size(64)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(128)
            .vocab_size(1000)
            .architecture("custom_arch")
            .model_name("My-Model")
            .build()
            .expect("should build");
        assert_eq!(config.architecture, "custom_arch");
        assert_eq!(config.model_name, "My-Model");
    }

    #[test]
    fn partial_override_inherits_defaults() {
        // Only override layers; everything else should come from Bonsai-8B defaults
        let config = ModelConfigBuilder::new()
            .layers(12)
            .build()
            .expect("partial override should succeed");
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.hidden_size, 4096); // default
        assert_eq!(config.vocab_size, 151936); // default
    }

    // ── Error cases: invalid builds ───────────────────────────────────────────

    #[test]
    fn zero_layers_returns_error() {
        let err = ModelConfigBuilder::new()
            .layers(0)
            .build()
            .expect_err("zero layers should fail");
        assert!(
            err.0.contains("num_layers"),
            "error should mention field: {err}"
        );
    }

    #[test]
    fn zero_hidden_size_returns_error() {
        let err = ModelConfigBuilder::new()
            .hidden_size(0)
            .build()
            .expect_err("zero hidden_size should fail");
        assert!(err.0.contains("hidden_size"), "{err}");
    }

    #[test]
    fn zero_attention_heads_returns_error() {
        let err = ModelConfigBuilder::new()
            .num_attention_heads(0)
            .build()
            .expect_err("zero attention heads should fail");
        assert!(err.0.contains("num_attention_heads"), "{err}");
    }

    #[test]
    fn zero_kv_heads_returns_error() {
        let err = ModelConfigBuilder::new()
            .num_kv_heads(0)
            .build()
            .expect_err("zero kv_heads should fail");
        assert!(err.0.contains("num_kv_heads"), "{err}");
    }

    #[test]
    fn hidden_size_not_divisible_by_heads_returns_error() {
        // hidden=100, heads=3 → 100 % 3 ≠ 0
        let err = ModelConfigBuilder::new()
            .hidden_size(100)
            .num_attention_heads(3)
            .num_kv_heads(1)
            .build()
            .expect_err("indivisible hidden/heads should fail");
        assert!(
            err.0.contains("divisible"),
            "error should mention divisibility: {err}"
        );
    }

    #[test]
    fn attention_heads_not_divisible_by_kv_heads_returns_error() {
        // heads=6, kv_heads=4 → 6 % 4 ≠ 0
        let err = ModelConfigBuilder::new()
            .hidden_size(96) // 96 / 6 = 16 (valid head_dim)
            .num_attention_heads(6)
            .num_kv_heads(4)
            .build()
            .expect_err("GQA divisibility violation should fail");
        assert!(
            err.0.contains("divisible"),
            "error should mention divisibility: {err}"
        );
    }

    #[test]
    fn zero_intermediate_size_returns_error() {
        let err = ModelConfigBuilder::new()
            .intermediate_size(0)
            .build()
            .expect_err("zero intermediate_size should fail");
        assert!(err.0.contains("intermediate_size"), "{err}");
    }

    #[test]
    fn vocab_size_one_returns_error() {
        let err = ModelConfigBuilder::new()
            .vocab_size(1)
            .build()
            .expect_err("vocab_size=1 should fail");
        assert!(err.0.contains("vocab_size"), "{err}");
    }

    #[test]
    fn zero_max_position_embeddings_returns_error() {
        let err = ModelConfigBuilder::new()
            .max_position_embeddings(0)
            .build()
            .expect_err("zero max_position_embeddings should fail");
        assert!(err.0.contains("max_position_embeddings"), "{err}");
    }

    #[test]
    fn non_positive_rope_freq_base_returns_error() {
        for bad in [-1.0f32, 0.0, f32::NEG_INFINITY, f32::NAN] {
            let err = ModelConfigBuilder::new()
                .rope_freq_base(bad)
                .build()
                .expect_err(&format!("rope_freq_base={bad} should fail"));
            assert!(err.0.contains("rope_freq_base"), "{err}");
        }
    }

    #[test]
    fn non_positive_rms_norm_eps_returns_error() {
        for bad in [-1e-6f32, 0.0, f32::NEG_INFINITY, f32::NAN] {
            let err = ModelConfigBuilder::new()
                .rms_norm_eps(bad)
                .build()
                .expect_err(&format!("rms_norm_eps={bad} should fail"));
            assert!(err.0.contains("rms_norm_eps"), "{err}");
        }
    }

    // ── ConfigError trait impls ───────────────────────────────────────────────

    #[test]
    fn config_error_display_contains_message() {
        let e = ConfigError::new("test message");
        let s = format!("{e}");
        assert!(s.contains("test message"), "Display should include message");
        assert!(
            s.contains("invalid model configuration"),
            "Display should include prefix"
        );
    }

    #[test]
    fn config_error_is_std_error() {
        let e = ConfigError::new("oops");
        // Verifies that ConfigError implements std::error::Error
        let _: &dyn std::error::Error = &e;
    }
}
