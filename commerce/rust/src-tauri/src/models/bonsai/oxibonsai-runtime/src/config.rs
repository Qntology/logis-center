//! Layered configuration system for OxiBonsai.
//!
//! Loading order: defaults → TOML file → CLI argument overrides.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{RuntimeError, RuntimeResult};

/// Top-level OxiBonsai configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OxiBonsaiConfig {
    /// Server configuration.
    pub server: ServerConfig,
    /// Sampling parameters.
    pub sampling: SamplingConfig,
    /// Model paths and limits.
    pub model: ModelConfig,
    /// Observability settings.
    pub observability: ObservabilityConfig,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind host address.
    pub host: String,
    /// Bind port.
    pub port: u16,
}

/// Sampling parameters configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingConfig {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path to the GGUF model file.
    pub model_path: Option<String>,
    /// Path to tokenizer.json file.
    pub tokenizer_path: Option<String>,
    /// Maximum sequence length (prompt + generated).
    pub max_seq_len: usize,
}

/// Observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservabilityConfig {
    /// Log level filter (e.g. "info", "debug", "warn").
    pub log_level: String,
    /// Whether to emit JSON-formatted logs.
    pub json_logs: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_tokens: 512,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            max_seq_len: 4096,
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            json_logs: false,
        }
    }
}

/// Severity level for configuration warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningSeverity {
    /// Informational only.
    Info,
    /// May cause suboptimal behavior.
    Warning,
    /// Will likely cause failures.
    Error,
}

impl std::fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// A warning about a configuration value.
#[derive(Debug, Clone)]
pub struct ConfigWarning {
    /// Which configuration field this warning applies to.
    pub field: String,
    /// Human-readable warning message.
    pub message: String,
    /// Severity of this warning.
    pub severity: WarningSeverity,
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

impl OxiBonsaiConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> RuntimeResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::Config(format!(
                "failed to read config file {}: {e}",
                path.display()
            ))
        })?;
        let config: Self = toml::from_str(&content).map_err(|e| {
            RuntimeError::Config(format!(
                "failed to parse config file {}: {e}",
                path.display()
            ))
        })?;
        Ok(config)
    }

    /// Load configuration from a TOML file if a path is given, otherwise return defaults.
    pub fn load_or_default(path: Option<&Path>) -> Self {
        match path {
            Some(p) => match Self::load(p) {
                Ok(cfg) => cfg,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load config, using defaults");
                    Self::default()
                }
            },
            None => Self::default(),
        }
    }

    /// Validate this configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> RuntimeResult<()> {
        if self.sampling.temperature < 0.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.temperature must be >= 0.0, got {}",
                self.sampling.temperature
            )));
        }
        if self.sampling.top_p < 0.0 || self.sampling.top_p > 1.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.top_p must be in [0.0, 1.0], got {}",
                self.sampling.top_p
            )));
        }
        if self.sampling.repetition_penalty < 1.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.repetition_penalty must be >= 1.0, got {}",
                self.sampling.repetition_penalty
            )));
        }
        if self.sampling.max_tokens == 0 {
            return Err(RuntimeError::Config(
                "sampling.max_tokens must be > 0".to_string(),
            ));
        }
        if self.model.max_seq_len == 0 {
            return Err(RuntimeError::Config(
                "model.max_seq_len must be > 0".to_string(),
            ));
        }
        if self.server.host.is_empty() {
            return Err(RuntimeError::Config(
                "server.host must not be empty".to_string(),
            ));
        }
        // Port 0 is technically valid (OS assigns), so no check needed
        Ok(())
    }

    /// Run a dry-run check of this configuration.
    ///
    /// Returns warnings about potential issues without stopping execution.
    /// Checks for model file existence, tokenizer existence, and
    /// reasonable parameter values.
    pub fn dry_run_check(&self) -> Vec<ConfigWarning> {
        let mut warnings = Vec::new();

        // Check model file
        match &self.model.model_path {
            None => {
                warnings.push(ConfigWarning {
                    field: "model.model_path".to_string(),
                    message: "no model path configured".to_string(),
                    severity: WarningSeverity::Warning,
                });
            }
            Some(path) => {
                if !Path::new(path).exists() {
                    warnings.push(ConfigWarning {
                        field: "model.model_path".to_string(),
                        message: format!("model file does not exist: {}", path),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        // Check tokenizer file
        match &self.model.tokenizer_path {
            None => {
                warnings.push(ConfigWarning {
                    field: "model.tokenizer_path".to_string(),
                    message: "no tokenizer path configured; token IDs will be used".to_string(),
                    severity: WarningSeverity::Info,
                });
            }
            Some(path) => {
                if !Path::new(path).exists() {
                    warnings.push(ConfigWarning {
                        field: "model.tokenizer_path".to_string(),
                        message: format!("tokenizer file does not exist: {}", path),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        // Check sequence length
        if self.model.max_seq_len > 65536 {
            warnings.push(ConfigWarning {
                field: "model.max_seq_len".to_string(),
                message: format!(
                    "very large max_seq_len ({}); may require significant memory",
                    self.model.max_seq_len
                ),
                severity: WarningSeverity::Warning,
            });
        }

        // Check temperature
        if self.sampling.temperature > 2.0 {
            warnings.push(ConfigWarning {
                field: "sampling.temperature".to_string(),
                message: format!(
                    "high temperature ({}) may produce incoherent output",
                    self.sampling.temperature
                ),
                severity: WarningSeverity::Warning,
            });
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = OxiBonsaiConfig::default();
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 8080);
        assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.top_k, 40);
        assert!((cfg.sampling.top_p - 0.9).abs() < f32::EPSILON);
        assert!((cfg.sampling.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.max_tokens, 512);
        assert_eq!(cfg.model.max_seq_len, 4096);
        assert!(cfg.model.model_path.is_none());
        assert!(cfg.model.tokenizer_path.is_none());
        assert_eq!(cfg.observability.log_level, "info");
        assert!(!cfg.observability.json_logs);
    }

    #[test]
    fn toml_parsing() {
        let model_path = std::env::temp_dir().join("model.gguf");
        let tokenizer_path = std::env::temp_dir().join("tokenizer.json");
        let toml_str = format!(
            r#"
[server]
host = "127.0.0.1"
port = 3000

[sampling]
temperature = 0.5
top_k = 50
top_p = 0.95
repetition_penalty = 1.2
max_tokens = 1024

[model]
model_path = "{}"
tokenizer_path = "{}"
max_seq_len = 8192

[observability]
log_level = "debug"
json_logs = true
"#,
            model_path.display(),
            tokenizer_path.display()
        );
        let cfg: OxiBonsaiConfig = toml::from_str(&toml_str).expect("should parse valid TOML");
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 3000);
        assert!((cfg.sampling.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.top_k, 50);
        assert_eq!(cfg.sampling.max_tokens, 1024);
        assert_eq!(
            cfg.model.model_path.as_deref(),
            Some(model_path.to_str().expect("path is valid UTF-8"))
        );
        assert_eq!(cfg.model.max_seq_len, 8192);
        assert_eq!(cfg.observability.log_level, "debug");
        assert!(cfg.observability.json_logs);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"
[server]
port = 9090
"#;
        let cfg: OxiBonsaiConfig = toml::from_str(toml_str).expect("should parse partial TOML");
        assert_eq!(cfg.server.port, 9090);
        // Rest should be defaults
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.model.max_seq_len, 4096);
    }

    #[test]
    fn missing_file_returns_default() {
        let path = std::env::temp_dir().join("nonexistent_oxibonsai_config_12345.toml");
        let cfg = OxiBonsaiConfig::load_or_default(Some(&path));
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn load_or_default_none_returns_default() {
        let cfg = OxiBonsaiConfig::load_or_default(None);
        assert_eq!(cfg.server.host, "0.0.0.0");
    }

    // ── Validation tests ──

    #[test]
    fn validate_defaults_ok() {
        let cfg = OxiBonsaiConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_negative_temperature() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.temperature = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_top_p_out_of_range() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.top_p = 1.5;
        assert!(cfg.validate().is_err());

        cfg.sampling.top_p = -0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_repetition_penalty_too_low() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.repetition_penalty = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_tokens_zero() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.max_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_seq_len_zero() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_empty_host() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.server.host = String::new();
        assert!(cfg.validate().is_err());
    }

    // ── Dry-run check tests ──

    #[test]
    fn dry_run_no_model_path() {
        let cfg = OxiBonsaiConfig::default();
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "model.model_path"));
    }

    #[test]
    fn dry_run_nonexistent_model() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.model_path = Some(
            std::env::temp_dir()
                .join("nonexistent_oxibonsai_test_99999.gguf")
                .display()
                .to_string(),
        );
        let warnings = cfg.dry_run_check();
        let model_warning = warnings
            .iter()
            .find(|w| w.field == "model.model_path")
            .expect("should have model warning");
        assert_eq!(model_warning.severity, WarningSeverity::Error);
    }

    #[test]
    fn dry_run_high_temperature() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.temperature = 3.0;
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "sampling.temperature"));
    }

    #[test]
    fn dry_run_large_seq_len() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_seq_len = 100_000;
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "model.max_seq_len"));
    }

    #[test]
    fn warning_severity_display() {
        assert_eq!(format!("{}", WarningSeverity::Info), "info");
        assert_eq!(format!("{}", WarningSeverity::Warning), "warning");
        assert_eq!(format!("{}", WarningSeverity::Error), "error");
    }

    #[test]
    fn config_warning_display() {
        let w = ConfigWarning {
            field: "test.field".to_string(),
            message: "test message".to_string(),
            severity: WarningSeverity::Warning,
        };
        let s = format!("{}", w);
        assert!(s.contains("warning"));
        assert!(s.contains("test.field"));
        assert!(s.contains("test message"));
    }

    #[test]
    fn load_from_temp_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_test_config.toml");
        std::fs::write(
            &path,
            r#"
[server]
host = "10.0.0.1"
port = 4444
"#,
        )
        .expect("write temp config");

        let cfg = OxiBonsaiConfig::load(&path).expect("should load temp config");
        assert_eq!(cfg.server.host, "10.0.0.1");
        assert_eq!(cfg.server.port, 4444);

        let _ = std::fs::remove_file(&path);
    }
}
