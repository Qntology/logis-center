//! Integration tests for `ServerConfig::validate`.
//!
//! Covers the full rule table documented on the `validate` doc-comment.

use std::path::PathBuf;

use oxibonsai_serve::config::{ConfigError, ServerConfig};
use oxibonsai_serve::validation::{MAX_DEFAULT_MAX_TOKENS, MIN_BEARER_TOKEN_LEN, VALID_LOG_LEVELS};

fn assert_validation(cfg: &ServerConfig, should_pass: bool) {
    let result = cfg.validate();
    if should_pass {
        result.expect("expected validation to pass");
    } else {
        match result {
            Err(ConfigError::Validation(_)) => {}
            other => panic!("expected Validation error, got {other:?}"),
        }
    }
}

// ─── Defaults ─────────────────────────────────────────────────────────────

#[test]
fn defaults_pass_validation() {
    let cfg = ServerConfig::default();
    assert_validation(&cfg, true);
}

// ─── Port ─────────────────────────────────────────────────────────────────

#[test]
fn port_zero_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.bind.port = 0;
    assert_validation(&cfg, false);
}

#[test]
fn port_65535_accepted() {
    let mut cfg = ServerConfig::default();
    cfg.bind.port = 65535;
    assert_validation(&cfg, true);
}

// ─── Sampling ─────────────────────────────────────────────────────────────

#[test]
fn max_tokens_zero_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_max_tokens = 0;
    assert_validation(&cfg, false);
}

#[test]
fn max_tokens_above_limit_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_max_tokens = MAX_DEFAULT_MAX_TOKENS + 1;
    assert_validation(&cfg, false);
}

#[test]
fn max_tokens_at_upper_bound_accepted() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_max_tokens = MAX_DEFAULT_MAX_TOKENS;
    assert_validation(&cfg, true);
}

#[test]
fn temperature_negative_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_temperature = -0.1;
    assert_validation(&cfg, false);
}

#[test]
fn temperature_above_two_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_temperature = 2.1;
    assert_validation(&cfg, false);
}

#[test]
fn temperature_nan_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_temperature = f32::NAN;
    assert_validation(&cfg, false);
}

#[test]
fn temperature_zero_ok() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_temperature = 0.0;
    assert_validation(&cfg, true);
}

#[test]
fn top_p_above_one_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_top_p = 1.5;
    assert_validation(&cfg, false);
}

#[test]
fn top_p_negative_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.sampling.default_top_p = -0.01;
    assert_validation(&cfg, false);
}

// ─── Log level ────────────────────────────────────────────────────────────

#[test]
fn unknown_log_level_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.observability.log_level = "loud".to_string();
    assert_validation(&cfg, false);
}

#[test]
fn all_valid_log_levels_pass() {
    for lvl in VALID_LOG_LEVELS {
        let mut cfg = ServerConfig::default();
        cfg.observability.log_level = (*lvl).to_string();
        assert_validation(&cfg, true);
    }
}

// ─── Metrics path ─────────────────────────────────────────────────────────

#[test]
fn metrics_path_empty_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_path = "".to_string();
    assert_validation(&cfg, false);
}

#[test]
fn metrics_path_must_be_absolute() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_path = "metrics".to_string();
    assert_validation(&cfg, false);
}

#[test]
fn metrics_path_absolute_accepted() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_path = "/prometheus-scrape".to_string();
    assert_validation(&cfg, true);
}

// ─── Auth ─────────────────────────────────────────────────────────────────

#[test]
fn short_bearer_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("short".to_string());
    assert_validation(&cfg, false);
}

#[test]
fn long_bearer_accepted() {
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("x".repeat(MIN_BEARER_TOKEN_LEN));
    assert_validation(&cfg, true);
}

#[test]
fn no_bearer_accepted() {
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = None;
    assert_validation(&cfg, true);
}

// ─── Limits ───────────────────────────────────────────────────────────────

#[test]
fn zero_concurrent_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_concurrent_requests = 0;
    assert_validation(&cfg, false);
}

#[test]
fn zero_timeout_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.limits.per_request_timeout_ms = 0;
    assert_validation(&cfg, false);
}

#[test]
fn zero_input_tokens_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_input_tokens = 0;
    assert_validation(&cfg, false);
}

// ─── File existence ───────────────────────────────────────────────────────

#[test]
fn missing_model_path_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.model.path = Some(PathBuf::from(
        "/definitely/does/not/exist/oxibonsai-test-missing.gguf",
    ));
    assert_validation(&cfg, false);
}

#[test]
fn existing_model_path_accepted() {
    // Create a real temp file and point the config at it.
    let mut p = std::env::temp_dir();
    p.push(format!(
        "oxibonsai-serve-validation-{}.gguf",
        std::process::id()
    ));
    std::fs::write(&p, b"GGUF").expect("write temp");
    let mut cfg = ServerConfig::default();
    cfg.model.path = Some(p.clone());
    assert_validation(&cfg, true);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn missing_tokenizer_path_rejected() {
    let mut cfg = ServerConfig::default();
    cfg.tokenizer.path = Some(PathBuf::from(
        "/definitely/does/not/exist/oxibonsai-test-missing-tok.json",
    ));
    assert_validation(&cfg, false);
}
