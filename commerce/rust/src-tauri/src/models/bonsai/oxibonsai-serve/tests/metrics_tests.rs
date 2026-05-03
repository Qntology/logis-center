//! Integration tests for the hand-rolled Prometheus metrics registry.
//!
//! Exercises counters, gauges, histograms, label-escaping, and the text
//! exposition rendering.

use oxibonsai_serve::metrics::{MetricsRegistry, DEFAULT_HISTOGRAM_BUCKETS};

// ─── Counters ─────────────────────────────────────────────────────────────

#[test]
fn counter_increments_monotonically() {
    let r = MetricsRegistry::new();
    r.inc_counter("hits_total", &[]);
    r.inc_counter("hits_total", &[]);
    r.inc_counter("hits_total", &[]);
    assert_eq!(r.counter_value("hits_total", &[]), 3);
}

#[test]
fn counter_separate_label_sets() {
    let r = MetricsRegistry::new();
    r.inc_counter("hits_total", &[("endpoint", "/a")]);
    r.inc_counter("hits_total", &[("endpoint", "/b")]);
    r.inc_counter("hits_total", &[("endpoint", "/a")]);
    assert_eq!(r.counter_value("hits_total", &[("endpoint", "/a")]), 2);
    assert_eq!(r.counter_value("hits_total", &[("endpoint", "/b")]), 1);
}

#[test]
fn counter_add_by_value() {
    let r = MetricsRegistry::new();
    r.add_counter("tokens_total", &[], 100);
    r.add_counter("tokens_total", &[], 50);
    assert_eq!(r.counter_value("tokens_total", &[]), 150);
}

// ─── Gauges ───────────────────────────────────────────────────────────────

#[test]
fn gauge_last_write_wins() {
    let r = MetricsRegistry::new();
    r.set_gauge("inflight", &[], 5);
    r.set_gauge("inflight", &[], 2);
    assert_eq!(r.gauge_value("inflight", &[]), 2);
}

#[test]
fn gauge_add_delta_handles_negatives() {
    let r = MetricsRegistry::new();
    r.set_gauge("inflight", &[], 10);
    r.add_gauge("inflight", &[], -3);
    assert_eq!(r.gauge_value("inflight", &[]), 7);
}

#[test]
fn gauge_absent_returns_zero() {
    let r = MetricsRegistry::new();
    assert_eq!(r.gauge_value("never_seen", &[]), 0);
}

// ─── Histograms ───────────────────────────────────────────────────────────

#[test]
fn histogram_count_matches_observations() {
    let r = MetricsRegistry::new();
    for _ in 0..7 {
        r.observe_histogram("lat", &[], 0.1);
    }
    assert_eq!(r.histogram_count("lat", &[]), 7);
}

#[test]
fn histogram_uses_default_buckets() {
    let r = MetricsRegistry::new();
    r.observe_histogram("lat", &[], 0.05);
    let counts = r.histogram_bucket_counts("lat", &[]);
    assert_eq!(counts.len(), DEFAULT_HISTOGRAM_BUCKETS.len());
}

#[test]
fn histogram_respects_bucket_boundaries() {
    let r = MetricsRegistry::new();
    // 0.005 falls into every bucket (all ≥ 0.01).
    r.observe_histogram("lat", &[], 0.005);
    let counts = r.histogram_bucket_counts("lat", &[]);
    for c in &counts {
        assert_eq!(*c, 1);
    }
}

#[test]
fn histogram_nan_is_ignored() {
    let r = MetricsRegistry::new();
    r.observe_histogram("lat", &[], f64::NAN);
    r.observe_histogram("lat", &[], 0.1);
    assert_eq!(r.histogram_count("lat", &[]), 1);
}

#[test]
fn custom_buckets_registry() {
    let r = MetricsRegistry::with_buckets(&[0.1, 1.0, 10.0]);
    r.observe_histogram("x", &[], 0.5);
    let counts = r.histogram_bucket_counts("x", &[]);
    assert_eq!(counts.len(), 3);
    assert_eq!(counts[0], 0); // 0.5 > 0.1
    assert_eq!(counts[1], 1);
    assert_eq!(counts[2], 1);
}

// ─── Rendering (Prometheus text exposition) ───────────────────────────────

#[test]
fn render_contains_help_and_type_lines() {
    let r = MetricsRegistry::new();
    r.inc_counter("requests_total", &[]);
    let body = r.render();
    assert!(body.contains("# HELP requests_total"));
    assert!(body.contains("# TYPE requests_total counter"));
    assert!(body.contains("requests_total 1"));
}

#[test]
fn render_counter_with_labels() {
    let r = MetricsRegistry::new();
    r.inc_counter("requests_total", &[("method", "GET"), ("code", "200")]);
    let body = r.render();
    // Labels are lex-sorted; check each key shows up.
    assert!(body.contains("code=\"200\""));
    assert!(body.contains("method=\"GET\""));
}

#[test]
fn render_escapes_label_values() {
    let r = MetricsRegistry::new();
    r.inc_counter("err_total", &[("msg", "has \"quote\"")]);
    let body = r.render();
    assert!(
        body.contains("has \\\"quote\\\""),
        "double-quotes should be escaped in the rendered output: {body}"
    );
}

#[test]
fn render_histogram_has_bucket_sum_count() {
    let r = MetricsRegistry::new();
    r.observe_histogram("lat_seconds", &[], 0.02);
    let body = r.render();
    assert!(body.contains("lat_seconds_bucket{le=\"0.01\"}"));
    assert!(body.contains("lat_seconds_bucket{le=\"+Inf\"}"));
    assert!(body.contains("lat_seconds_sum"));
    assert!(body.contains("lat_seconds_count"));
}

#[test]
fn render_gauge_emits_type_gauge() {
    let r = MetricsRegistry::new();
    r.set_gauge("inflight", &[], 4);
    let body = r.render();
    assert!(body.contains("# TYPE inflight gauge"));
    assert!(body.contains("inflight 4"));
}

#[test]
fn render_sorts_label_keys() {
    let r = MetricsRegistry::new();
    r.inc_counter("x_total", &[("z", "1"), ("a", "2")]);
    let body = r.render();
    let line = body
        .lines()
        .find(|l| l.starts_with("x_total{"))
        .expect("counter line present");
    // "a" must come before "z" in the rendered label set.
    let a_pos = line.find("a=").expect("a= in line");
    let z_pos = line.find("z=").expect("z= in line");
    assert!(a_pos < z_pos, "expected 'a' before 'z' in: {line}");
}

#[test]
fn render_is_deterministic_across_calls() {
    let r = MetricsRegistry::new();
    r.inc_counter("x_total", &[("method", "GET")]);
    r.inc_counter("x_total", &[("method", "POST")]);
    let a = r.render();
    let b = r.render();
    assert_eq!(a, b, "render should be deterministic");
}
