//! Integration tests for the token streaming metrics module.

use oxibonsai_runtime::stream_metrics::{
    RequestStreamMetrics, StreamMetricsSnapshot, StreamingMetricsAggregator,
};
use std::time::Duration;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a snapshot with explicit values for use in aggregator tests.
fn make_snapshot(
    ttft_ms: Option<f64>,
    tps: Option<f64>,
    e2e_ms: Option<f64>,
    completion_tokens: usize,
    prompt_tokens: usize,
) -> StreamMetricsSnapshot {
    StreamMetricsSnapshot {
        ttft_ms,
        mean_tbt_ms: None,
        p99_tbt_ms: None,
        tokens_per_second: tps,
        e2e_latency_ms: e2e_ms,
        completion_tokens,
        prompt_tokens,
    }
}

// ── RequestStreamMetrics tests ────────────────────────────────────────────────

/// 1. A freshly created collector reports no TTFT.
#[test]
fn metrics_new_no_ttft() {
    let m = RequestStreamMetrics::new();
    assert!(
        m.ttft().is_none(),
        "ttft() must be None before any token is recorded"
    );
}

/// 2. After record_first_token(), ttft() becomes Some.
#[test]
fn metrics_record_first_token() {
    let mut m = RequestStreamMetrics::new();
    m.record_first_token();
    assert!(
        m.ttft().is_some(),
        "ttft() must be Some after record_first_token()"
    );
    // The TTFT should be a non-negative duration.
    let ttft = m.ttft().expect("checked");
    assert!(ttft >= Duration::ZERO);
}

/// 3. completion_tokens() reflects the total number of tokens recorded.
#[test]
fn metrics_record_multiple_tokens() {
    let mut m = RequestStreamMetrics::new();
    m.record_first_token();
    m.record_token();
    m.record_token();
    m.record_token();
    assert_eq!(
        m.completion_tokens(),
        4,
        "should count first + 3 subsequent tokens"
    );
}

/// 4. mean_tbt() returns Some when at least 3 tokens have been recorded
///    (which gives at least 2 TBT samples).
#[test]
fn metrics_mean_tbt_with_tokens() {
    let mut m = RequestStreamMetrics::new();
    m.record_first_token();
    std::thread::sleep(Duration::from_micros(200));
    m.record_token();
    std::thread::sleep(Duration::from_micros(200));
    m.record_token();
    assert!(
        m.mean_tbt().is_some(),
        "mean_tbt() must be Some after at least 3 tokens"
    );
}

/// 5. tokens_per_second() is positive after recording multiple tokens.
#[test]
fn metrics_tokens_per_second_positive() {
    let mut m = RequestStreamMetrics::new();
    m.record_first_token();
    std::thread::sleep(Duration::from_micros(300));
    m.record_token();
    let tps = m.tokens_per_second();
    assert!(tps.is_some(), "tps must be Some after >=2 tokens");
    assert!(tps.expect("checked") > 0.0, "tps must be strictly positive");
}

/// 6. e2e_latency() returns Some after finish() is called.
#[test]
fn metrics_e2e_latency() {
    let mut m = RequestStreamMetrics::new();
    m.record_first_token();
    m.record_token();
    m.finish();
    assert!(
        m.e2e_latency().is_some(),
        "e2e_latency() must be Some after finish()"
    );
    assert!(m.e2e_latency().expect("checked") >= Duration::ZERO);
}

/// 7. snapshot().summary() is non-empty.
#[test]
fn metrics_snapshot_summary_nonempty() {
    let mut m = RequestStreamMetrics::new_with_prompt_tokens(64);
    m.record_first_token();
    std::thread::sleep(Duration::from_micros(150));
    m.record_token();
    m.finish();
    let snap = m.snapshot();
    assert!(
        !snap.summary().is_empty(),
        "snapshot summary must not be empty"
    );
}

// ── StreamingMetricsAggregator tests ──────────────────────────────────────────

/// 8. An empty aggregator reports None for avg_ttft_ms().
#[test]
fn aggregator_empty() {
    let agg = StreamingMetricsAggregator::new();
    assert!(
        agg.avg_ttft_ms().is_none(),
        "avg_ttft_ms() must be None for empty aggregator"
    );
    assert_eq!(agg.num_requests(), 0);
}

/// 9. Aggregator with a single snapshot: avg_ttft equals the snapshot's ttft.
#[test]
fn aggregator_single_snapshot() {
    let mut agg = StreamingMetricsAggregator::new();
    agg.record(make_snapshot(Some(55.0), Some(80.0), Some(200.0), 20, 10));
    let avg = agg.avg_ttft_ms().expect("must be Some");
    assert!(
        (avg - 55.0).abs() < 1e-9,
        "avg_ttft should equal the single snapshot value; got {avg}"
    );
}

/// 10. Aggregator with multiple snapshots correctly averages TTFT.
#[test]
fn aggregator_multiple_snapshots() {
    let mut agg = StreamingMetricsAggregator::new();
    agg.record(make_snapshot(Some(10.0), None, Some(100.0), 5, 2));
    agg.record(make_snapshot(Some(20.0), None, Some(200.0), 5, 2));
    agg.record(make_snapshot(Some(30.0), None, Some(300.0), 5, 2));
    let avg = agg.avg_ttft_ms().expect("must be Some");
    assert!(
        (avg - 20.0).abs() < 1e-9,
        "avg_ttft of [10, 20, 30] should be 20; got {avg}"
    );
    assert_eq!(agg.num_requests(), 3);
}

/// 11. p99_e2e_ms() returns Some when at least one snapshot has e2e data.
#[test]
fn aggregator_p99_e2e() {
    let mut agg = StreamingMetricsAggregator::new();
    for ms in [100.0_f64, 200.0, 300.0, 400.0, 500.0] {
        agg.record(make_snapshot(None, None, Some(ms), 10, 5));
    }
    let p99 = agg.p99_e2e_ms();
    assert!(
        p99.is_some(),
        "p99_e2e_ms must be Some when snapshots have e2e"
    );
    // P99 of [100, 200, 300, 400, 500] must be the upper range.
    let v = p99.expect("checked");
    assert!(
        (400.0..=500.0).contains(&v),
        "P99 of [100..500] should be 400–500; got {v}"
    );
}

/// 12. total_completion_tokens() sums correctly across requests.
#[test]
fn aggregator_total_tokens() {
    let mut agg = StreamingMetricsAggregator::new();
    agg.record(make_snapshot(None, None, None, 15, 0));
    agg.record(make_snapshot(None, None, None, 25, 0));
    agg.record(make_snapshot(None, None, None, 35, 0));
    assert_eq!(
        agg.total_completion_tokens(),
        75,
        "total should sum 15+25+35=75"
    );
}

/// 13. report() is non-empty.
#[test]
fn aggregator_report_nonempty() {
    let mut agg = StreamingMetricsAggregator::new();
    agg.record(make_snapshot(
        Some(50.0),
        Some(100.0),
        Some(1000.0),
        50,
        128,
    ));
    let report = agg.report();
    assert!(
        !report.is_empty(),
        "report() must return a non-empty string"
    );
    assert!(
        report.contains("1"),
        "report should mention at least one request"
    );
}
