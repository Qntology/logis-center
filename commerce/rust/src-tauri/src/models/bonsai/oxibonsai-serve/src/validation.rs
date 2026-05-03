//! Configuration validation.
//!
//! A [`crate::config::ServerConfig`] is considered well-formed when it passes
//! [`ServerConfig::validate`].  The rules are:
//!
//! | Field                                      | Rule                                   |
//! |--------------------------------------------|----------------------------------------|
//! | `bind.port`                                | `1..=65535`                            |
//! | `sampling.default_max_tokens`              | `1..=8192`                             |
//! | `sampling.default_temperature`             | `0.0..=2.0` (and finite)               |
//! | `sampling.default_top_p`                   | `0.0..=1.0` (and finite)               |
//! | `observability.log_level`                  | ∈ { error, warn, info, debug, trace, off } |
//! | `model.path`                               | Must exist on disk if set              |
//! | `tokenizer.path`                           | Must exist on disk if set              |
//! | `auth.bearer_token`                        | ≥ 16 characters if set                 |
//! | `observability.metrics_path`               | Non-empty; must start with `/`         |
//! | `limits.max_concurrent_requests`           | ≥ 1                                    |
//! | `limits.per_request_timeout_ms`            | ≥ 1                                    |
//! | `limits.max_input_tokens`                  | ≥ 1                                    |

use crate::config::{ConfigError, ServerConfig};

/// Whitelist of accepted `log_level` values.
pub const VALID_LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace", "off"];

/// Minimum bearer-token length, in UTF-8 bytes.
pub const MIN_BEARER_TOKEN_LEN: usize = 16;

/// Upper bound on `default_max_tokens`.
pub const MAX_DEFAULT_MAX_TOKENS: usize = 8192;

impl ServerConfig {
    /// Validate the configuration, returning a [`ConfigError::Validation`] on
    /// the first rule that fails.
    ///
    /// Rules are deliberately conservative — unusual values (e.g. `port=0`)
    /// trip validation and force the operator to think.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // ─── Bind ────────────────────────────────────────────────────────
        if self.bind.port == 0 {
            return Err(ConfigError::Validation(
                "bind.port must be in [1, 65535]".to_string(),
            ));
        }

        // ─── Sampling ────────────────────────────────────────────────────
        if self.sampling.default_max_tokens == 0
            || self.sampling.default_max_tokens > MAX_DEFAULT_MAX_TOKENS
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_max_tokens must be in [1, {MAX_DEFAULT_MAX_TOKENS}], got {}",
                self.sampling.default_max_tokens
            )));
        }
        if !self.sampling.default_temperature.is_finite()
            || self.sampling.default_temperature < 0.0
            || self.sampling.default_temperature > 2.0
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_temperature must be in [0, 2], got {}",
                self.sampling.default_temperature
            )));
        }
        if !self.sampling.default_top_p.is_finite()
            || self.sampling.default_top_p < 0.0
            || self.sampling.default_top_p > 1.0
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_top_p must be in [0, 1], got {}",
                self.sampling.default_top_p
            )));
        }

        // ─── Observability ───────────────────────────────────────────────
        if !VALID_LOG_LEVELS
            .iter()
            .any(|l| l.eq_ignore_ascii_case(&self.observability.log_level))
        {
            return Err(ConfigError::Validation(format!(
                "observability.log_level must be one of {VALID_LOG_LEVELS:?}, got {:?}",
                self.observability.log_level
            )));
        }
        if self.observability.metrics_path.is_empty()
            || !self.observability.metrics_path.starts_with('/')
        {
            return Err(ConfigError::Validation(format!(
                "observability.metrics_path must be an absolute HTTP path, got {:?}",
                self.observability.metrics_path
            )));
        }

        // ─── Limits ──────────────────────────────────────────────────────
        if self.limits.max_concurrent_requests == 0 {
            return Err(ConfigError::Validation(
                "limits.max_concurrent_requests must be ≥ 1".to_string(),
            ));
        }
        if self.limits.per_request_timeout_ms == 0 {
            return Err(ConfigError::Validation(
                "limits.per_request_timeout_ms must be ≥ 1".to_string(),
            ));
        }
        if self.limits.max_input_tokens == 0 {
            return Err(ConfigError::Validation(
                "limits.max_input_tokens must be ≥ 1".to_string(),
            ));
        }

        // ─── Auth ────────────────────────────────────────────────────────
        if let Some(ref tok) = self.auth.bearer_token {
            if tok.len() < MIN_BEARER_TOKEN_LEN {
                return Err(ConfigError::Validation(format!(
                    "auth.bearer_token must be at least {MIN_BEARER_TOKEN_LEN} chars"
                )));
            }
        }

        // ─── Paths (existence) ───────────────────────────────────────────
        if let Some(ref path) = self.model.path {
            if !path.exists() {
                return Err(ConfigError::Validation(format!(
                    "model.path does not exist: {}",
                    path.display()
                )));
            }
        }
        if let Some(ref path) = self.tokenizer.path {
            if !path.exists() {
                return Err(ConfigError::Validation(format!(
                    "tokenizer.path does not exist: {}",
                    path.display()
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = ServerConfig::default();
        cfg.validate().expect("defaults should validate");
    }

    #[test]
    fn port_zero_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.bind.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_log_level_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.observability.log_level = "loud".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_top_p_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.sampling.default_top_p = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn short_bearer_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.auth.bearer_token = Some("short".to_string());
        assert!(cfg.validate().is_err());
    }
}
