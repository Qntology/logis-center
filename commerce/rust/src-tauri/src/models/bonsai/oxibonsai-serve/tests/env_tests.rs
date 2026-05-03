//! Integration tests for environment-variable → `PartialServerConfig` mapping.
//!
//! These do **not** touch the live process environment; instead every test
//! calls `parse_env_map` with an explicit iterator.

use std::path::PathBuf;

use oxibonsai_serve::config::{ConfigError, PartialServerConfig};
use oxibonsai_serve::env::parse_env_map;

fn kv(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

// ─── Recognised variables ─────────────────────────────────────────────────

#[test]
fn empty_yields_empty_partial() {
    let p = parse_env_map(std::iter::empty()).expect("parse");
    assert_eq!(p, PartialServerConfig::default());
}

#[test]
fn host_port_parsed() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_HOST", "127.0.0.1"),
        ("OXIBONSAI_PORT", "3333"),
    ]))
    .expect("parse");
    assert_eq!(p.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(p.port, Some(3333));
}

#[test]
fn model_and_tokenizer_paths() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_MODEL_PATH", "/m.gguf"),
        ("OXIBONSAI_TOKENIZER_PATH", "/t.json"),
        ("OXIBONSAI_TOKENIZER_KIND", "huggingface"),
    ]))
    .expect("parse");
    assert_eq!(p.model_path, Some(PathBuf::from("/m.gguf")));
    assert_eq!(p.tokenizer_path, Some(PathBuf::from("/t.json")));
    assert_eq!(p.tokenizer_kind.as_deref(), Some("huggingface"));
}

#[test]
fn sampling_fields() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_MAX_TOKENS", "1000"),
        ("OXIBONSAI_TEMPERATURE", "0.8"),
        ("OXIBONSAI_TOP_P", "0.95"),
    ]))
    .expect("parse");
    assert_eq!(p.default_max_tokens, Some(1000));
    let t = p.default_temperature.expect("some");
    assert!((t - 0.8).abs() < f32::EPSILON);
    let tp = p.default_top_p.expect("some");
    assert!((tp - 0.95).abs() < f32::EPSILON);
}

#[test]
fn limits_fields() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_MAX_INPUT_TOKENS", "4096"),
        ("OXIBONSAI_MAX_CONCURRENT", "64"),
        ("OXIBONSAI_REQUEST_TIMEOUT_MS", "15000"),
    ]))
    .expect("parse");
    assert_eq!(p.max_input_tokens, Some(4096));
    assert_eq!(p.max_concurrent_requests, Some(64));
    assert_eq!(p.per_request_timeout_ms, Some(15000));
}

#[test]
fn bearer_token_and_seed() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_BEARER_TOKEN", "some-long-secret-token"),
        ("OXIBONSAI_SEED", "1234"),
    ]))
    .expect("parse");
    assert_eq!(p.bearer_token.as_deref(), Some("some-long-secret-token"));
    assert_eq!(p.seed, Some(1234));
}

#[test]
fn observability_fields() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_LOG_LEVEL", "debug"),
        ("OXIBONSAI_METRICS_ENABLED", "true"),
        ("OXIBONSAI_METRICS_PATH", "/prom"),
    ]))
    .expect("parse");
    assert_eq!(p.log_level.as_deref(), Some("debug"));
    assert_eq!(p.metrics_enabled, Some(true));
    assert_eq!(p.metrics_path.as_deref(), Some("/prom"));
}

#[test]
fn bool_accepts_yes_on_1() {
    for v in ["yes", "on", "1", "YES", "True"] {
        let p = parse_env_map(kv(&[("OXIBONSAI_METRICS_ENABLED", v)])).expect("parse");
        assert_eq!(p.metrics_enabled, Some(true), "value={v}");
    }
}

#[test]
fn bool_accepts_no_off_0() {
    for v in ["no", "off", "0", "false", "FALSE"] {
        let p = parse_env_map(kv(&[("OXIBONSAI_METRICS_ENABLED", v)])).expect("parse");
        assert_eq!(p.metrics_enabled, Some(false), "value={v}");
    }
}

// ─── Error cases ──────────────────────────────────────────────────────────

#[test]
fn bad_port_errors() {
    let err = parse_env_map(kv(&[("OXIBONSAI_PORT", "abc")])).expect_err("should fail");
    assert!(matches!(err, ConfigError::EnvParse { ref name, .. } if name == "OXIBONSAI_PORT"));
}

#[test]
fn bad_max_tokens_errors() {
    let err = parse_env_map(kv(&[("OXIBONSAI_MAX_TOKENS", "-1")])).expect_err("should fail");
    assert!(
        matches!(err, ConfigError::EnvParse { ref name, .. } if name == "OXIBONSAI_MAX_TOKENS")
    );
}

#[test]
fn bad_temperature_errors() {
    let err =
        parse_env_map(kv(&[("OXIBONSAI_TEMPERATURE", "not-a-float")])).expect_err("should fail");
    assert!(
        matches!(err, ConfigError::EnvParse { ref name, .. } if name == "OXIBONSAI_TEMPERATURE")
    );
}

#[test]
fn bad_bool_errors() {
    let err =
        parse_env_map(kv(&[("OXIBONSAI_METRICS_ENABLED", "maybe")])).expect_err("should fail");
    assert!(
        matches!(err, ConfigError::EnvParse { ref name, .. } if name == "OXIBONSAI_METRICS_ENABLED")
    );
}

#[test]
fn unrecognised_env_vars_are_ignored() {
    let p = parse_env_map(kv(&[
        ("OXIBONSAI_PORT", "1234"),
        ("OXIBONSAI_SOMETHING_RANDOM", "foo"),
        ("PATH", "/usr/bin"),
    ]))
    .expect("parse");
    assert_eq!(p.port, Some(1234));
}
