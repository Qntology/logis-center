//! Tracing initialization with configurable output format.
//!
//! Provides a structured way to initialize the `tracing` subscriber
//! with either human-readable or JSON output, driven by configuration.
//!
//! On WASM targets, `tracing_subscriber` is not available. The
//! `init_tracing` function is a no-op stub that always succeeds.

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Configuration for the tracing/logging subsystem.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Log level filter string (e.g. "info", "debug", "oxibonsai=trace").
    pub log_level: String,
    /// Whether to emit JSON-formatted log lines.
    pub json_output: bool,
    /// Whether to include the source file path in log output.
    pub with_file: bool,
    /// Whether to include line numbers in log output.
    pub with_line_number: bool,
    /// Whether to include the tracing target in log output.
    pub with_target: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            json_output: false,
            with_file: false,
            with_line_number: false,
            with_target: true,
        }
    }
}

impl TracingConfig {
    /// Create a `TracingConfig` from an `ObservabilityConfig`.
    pub fn from_observability(obs: &crate::config::ObservabilityConfig) -> Self {
        Self {
            log_level: obs.log_level.clone(),
            json_output: obs.json_logs,
            ..Self::default()
        }
    }
}

/// Initialize tracing with the given configuration.
///
/// On non-WASM targets, sets up a `tracing_subscriber` with either
/// human-readable or JSON output format.
///
/// On WASM targets, this is a no-op (tracing_subscriber is unavailable).
///
/// # Errors
///
/// On non-WASM: returns an error if the tracing subscriber cannot be
/// initialized (e.g. if a global subscriber is already set).
///
/// On WASM: always returns `Ok(())`.
#[cfg(not(target_arch = "wasm32"))]
pub fn init_tracing(config: &TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    if config.json_output {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .try_init()
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(format!(
                    "failed to init tracing: {e}"
                )))
            })?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                fmt::layer()
                    .with_file(config.with_file)
                    .with_line_number(config.with_line_number)
                    .with_target(config.with_target),
            )
            .try_init()
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(format!(
                    "failed to init tracing: {e}"
                )))
            })?;
    }

    Ok(())
}

/// Initialize tracing (no-op on WASM targets).
#[cfg(target_arch = "wasm32")]
pub fn init_tracing(_config: &TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = TracingConfig::default();
        assert_eq!(cfg.log_level, "info");
        assert!(!cfg.json_output);
        assert!(!cfg.with_file);
        assert!(!cfg.with_line_number);
        assert!(cfg.with_target);
    }

    #[test]
    fn from_observability_config() {
        let obs = crate::config::ObservabilityConfig {
            log_level: "debug".to_string(),
            json_logs: true,
        };
        let cfg = TracingConfig::from_observability(&obs);
        assert_eq!(cfg.log_level, "debug");
        assert!(cfg.json_output);
    }

    #[test]
    fn tracing_config_clone() {
        let cfg = TracingConfig {
            log_level: "warn".to_string(),
            json_output: true,
            with_file: true,
            with_line_number: true,
            with_target: false,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.log_level, "warn");
        assert!(cloned.json_output);
        assert!(cloned.with_file);
        assert!(cloned.with_line_number);
        assert!(!cloned.with_target);
    }

    #[test]
    fn tracing_config_debug() {
        let cfg = TracingConfig::default();
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("TracingConfig"));
        assert!(debug_str.contains("info"));
    }
}
