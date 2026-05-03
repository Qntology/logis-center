//! Property-based tests using `proptest`.
//!
//! Invariants checked here:
//!
//! 1. TOML round-trip is idempotent for any value in the valid ranges.
//! 2. `ServerConfig::validate` accepts all in-range fields and rejects
//!    all out-of-range fields.
//! 3. `PartialServerConfig::merge` is right-biased: `(a.merge(b)).port` is
//!    `b.port` if `b.port.is_some()`, else `a.port`.
//! 4. `MetricsRegistry::inc_counter` composed N times equals
//!    `add_counter(N)`.
//! 5. Histogram observations always advance the count.

use oxibonsai_serve::config::{PartialServerConfig, ServerConfig};
use oxibonsai_serve::metrics::MetricsRegistry;
use oxibonsai_serve::validation::MAX_DEFAULT_MAX_TOKENS;

use proptest::prelude::*;

// ─── Strategies ───────────────────────────────────────────────────────────

fn valid_port() -> impl Strategy<Value = u16> {
    1u16..=65535
}

fn valid_max_tokens() -> impl Strategy<Value = usize> {
    1usize..=MAX_DEFAULT_MAX_TOKENS
}

fn valid_temperature() -> impl Strategy<Value = f32> {
    (0.0f32..=2.0f32).prop_filter("finite", |v| v.is_finite())
}

fn valid_top_p() -> impl Strategy<Value = f32> {
    (0.0f32..=1.0f32).prop_filter("finite", |v| v.is_finite())
}

fn label_key() -> impl Strategy<Value = String> {
    // Prometheus label keys: ASCII letters/digits/underscore, not starting
    // with digit.
    "[a-zA-Z][a-zA-Z0-9_]{0,7}".prop_map(String::from)
}

// ─── Property 1 + 2: validation bounds ────────────────────────────────────

proptest! {
    #[test]
    fn in_range_config_always_validates(
        port in valid_port(),
        max_tokens in valid_max_tokens(),
        temperature in valid_temperature(),
        top_p in valid_top_p(),
    ) {
        let mut cfg = ServerConfig::default();
        cfg.bind.port = port;
        cfg.sampling.default_max_tokens = max_tokens;
        cfg.sampling.default_temperature = temperature;
        cfg.sampling.default_top_p = top_p;
        prop_assert!(cfg.validate().is_ok(), "in-range config must validate: {cfg:?}");
    }

    #[test]
    fn out_of_range_max_tokens_always_fails(
        too_many in (MAX_DEFAULT_MAX_TOKENS + 1)..=(MAX_DEFAULT_MAX_TOKENS * 2),
    ) {
        let mut cfg = ServerConfig::default();
        cfg.sampling.default_max_tokens = too_many;
        prop_assert!(cfg.validate().is_err());
    }

    #[test]
    fn out_of_range_top_p_always_fails(
        too_high in 1.01f32..=100.0f32,
    ) {
        let mut cfg = ServerConfig::default();
        cfg.sampling.default_top_p = too_high;
        prop_assert!(cfg.validate().is_err());
    }

    #[test]
    fn out_of_range_temperature_always_fails(
        too_high in 2.01f32..=100.0f32,
    ) {
        let mut cfg = ServerConfig::default();
        cfg.sampling.default_temperature = too_high;
        prop_assert!(cfg.validate().is_err());
    }
}

/// Port 0 is the only out-of-range value expressible as `u16`; exercise it as
/// a deterministic test rather than via proptest (which does not accept empty
/// parameter lists).
#[test]
fn port_zero_always_fails_validation() {
    let mut cfg = ServerConfig::default();
    cfg.bind.port = 0;
    assert!(cfg.validate().is_err());
}

// ─── Property 3: merge right-biased ───────────────────────────────────────

proptest! {
    #[test]
    fn merge_right_biased_port(
        a in proptest::option::of(valid_port()),
        b in proptest::option::of(valid_port()),
    ) {
        let pa = PartialServerConfig { port: a, ..Default::default() };
        let pb = PartialServerConfig { port: b, ..Default::default() };
        let merged = pa.merge(pb);
        let expected = b.or(a);
        prop_assert_eq!(merged.port, expected);
    }

    #[test]
    fn merge_right_biased_seed(
        a in proptest::option::of(any::<u64>()),
        b in proptest::option::of(any::<u64>()),
    ) {
        let pa = PartialServerConfig { seed: a, ..Default::default() };
        let pb = PartialServerConfig { seed: b, ..Default::default() };
        let merged = pa.merge(pb);
        let expected = b.or(a);
        prop_assert_eq!(merged.seed, expected);
    }
}

// ─── Property 4 + 5: metrics registry laws ───────────────────────────────

proptest! {
    #[test]
    fn counter_inc_equals_add(
        name in label_key(),
        n in 1u32..200u32,
    ) {
        let r1 = MetricsRegistry::new();
        let r2 = MetricsRegistry::new();
        let full_name = format!("{name}_total");
        for _ in 0..n {
            r1.inc_counter(&full_name, &[]);
        }
        r2.add_counter(&full_name, &[], n as u64);
        prop_assert_eq!(
            r1.counter_value(&full_name, &[]),
            r2.counter_value(&full_name, &[]),
        );
    }

    #[test]
    fn histogram_observe_increments_count(
        values in proptest::collection::vec(0.0f64..100.0f64, 1..20),
    ) {
        let r = MetricsRegistry::new();
        for v in &values {
            r.observe_histogram("x", &[], *v);
        }
        prop_assert_eq!(r.histogram_count("x", &[]), values.len() as u64);
    }

    #[test]
    fn gauge_set_is_last_write_wins(
        values in proptest::collection::vec(-1000i64..1000i64, 1..20),
    ) {
        let r = MetricsRegistry::new();
        for v in &values {
            r.set_gauge("g", &[], *v);
        }
        let last = *values.last().expect("non-empty by strategy");
        prop_assert_eq!(r.gauge_value("g", &[]), last);
    }
}

// ─── Property 6: TOML round-trip (defaults only for structural shape) ─────

proptest! {
    #[test]
    fn toml_roundtrip_for_port(
        port in valid_port(),
    ) {
        let mut cfg = ServerConfig::default();
        cfg.bind.port = port;
        let toml_s = cfg.to_toml_string().expect("serialize");
        let reparsed = ServerConfig::from_toml(&toml_s).expect("parse");
        prop_assert_eq!(cfg.bind.port, reparsed.bind.port);
    }

    #[test]
    fn toml_roundtrip_for_seed(
        seed in any::<u64>(),
    ) {
        let mut cfg = ServerConfig::default();
        cfg.seed = seed;
        let toml_s = cfg.to_toml_string().expect("serialize");
        let reparsed = ServerConfig::from_toml(&toml_s).expect("parse");
        prop_assert_eq!(cfg.seed, reparsed.seed);
    }
}
