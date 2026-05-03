//! Token streaming metrics: TTFT, inter-token latency (TBT), throughput.
//!
//! These metrics are essential for production LLM serving SLAs.
//!
//! - **TTFT** — Time To First Token: latency from request start to the first
//!   generated token.
//! - **TBT** — Time Between Tokens: inter-token latency during the decode phase.
//! - **E2E latency** — total request latency from start to generation complete.
//!
//! # Usage
//!
//! ```rust
//! use oxibonsai_runtime::stream_metrics::RequestStreamMetrics;
//!
//! let mut m = RequestStreamMetrics::new_with_prompt_tokens(128);
//! // ... first token arrives ...
//! m.record_first_token();
//! // ... subsequent tokens ...
//! m.record_token();
//! m.record_token();
//! m.finish();
//!
//! let snap = m.snapshot();
//! println!("{}", snap.summary());
//! ```

use std::time::{Duration, Instant};

// ── RequestStreamMetrics ─────────────────────────────────────────────────────

/// Per-request streaming metrics collector.
///
/// Call [`record_first_token`](RequestStreamMetrics::record_first_token) on the
/// first generated token, [`record_token`](RequestStreamMetrics::record_token)
/// for each subsequent token, and
/// [`finish`](RequestStreamMetrics::finish) when generation is complete.
pub struct RequestStreamMetrics {
    request_start: Instant,
    first_token_time: Option<Instant>,
    last_token_time: Option<Instant>,
    /// Wall-clock gaps between consecutive tokens (TBT samples).
    inter_token_gaps: Vec<Duration>,
    token_count: usize,
    prompt_tokens: usize,
}

impl RequestStreamMetrics {
    /// Create a new metrics collector with zero prompt tokens.
    pub fn new() -> Self {
        Self {
            request_start: Instant::now(),
            first_token_time: None,
            last_token_time: None,
            inter_token_gaps: Vec::new(),
            token_count: 0,
            prompt_tokens: 0,
        }
    }

    /// Create a new metrics collector that records the prompt token count.
    pub fn new_with_prompt_tokens(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            ..Self::new()
        }
    }

    /// Record the arrival of the **first** generated token.
    ///
    /// Calling this multiple times is safe — only the first call is recorded.
    pub fn record_first_token(&mut self) {
        if self.first_token_time.is_none() {
            let now = Instant::now();
            self.first_token_time = Some(now);
            self.last_token_time = Some(now);
            self.token_count = 1;
        }
    }

    /// Record the arrival of a **subsequent** generated token (not the first).
    ///
    /// If [`record_first_token`](Self::record_first_token) has not been called
    /// yet, this call is treated as the first token.
    pub fn record_token(&mut self) {
        let now = Instant::now();
        if self.first_token_time.is_none() {
            // Treat as first token.
            self.first_token_time = Some(now);
            self.last_token_time = Some(now);
            self.token_count = 1;
            return;
        }
        if let Some(prev) = self.last_token_time {
            self.inter_token_gaps.push(now.duration_since(prev));
        }
        self.last_token_time = Some(now);
        self.token_count += 1;
    }

    /// Mark the end of the generation pass.
    ///
    /// This records a final timestamp used for end-to-end latency.  It is safe
    /// to call even before any tokens have been recorded.
    pub fn finish(&mut self) {
        // If no last_token_time has been set we still want an e2e measurement.
        if self.last_token_time.is_none() {
            self.last_token_time = Some(Instant::now());
        }
    }

    /// Time to first token (TTFT).
    ///
    /// Returns `None` if [`record_first_token`](Self::record_first_token) has
    /// not been called.
    pub fn ttft(&self) -> Option<Duration> {
        let first = self.first_token_time?;
        Some(first.duration_since(self.request_start))
    }

    /// Median inter-token latency.
    ///
    /// Returns `None` when fewer than two tokens have been recorded (i.e. there
    /// are no TBT samples).
    pub fn median_tbt(&self) -> Option<Duration> {
        percentile_duration(&self.inter_token_gaps, 50)
    }

    /// P99 inter-token latency.
    ///
    /// Returns `None` when there are no TBT samples.
    pub fn p99_tbt(&self) -> Option<Duration> {
        percentile_duration(&self.inter_token_gaps, 99)
    }

    /// Mean (arithmetic average) inter-token latency.
    ///
    /// Returns `None` when there are no TBT samples.
    pub fn mean_tbt(&self) -> Option<Duration> {
        if self.inter_token_gaps.is_empty() {
            return None;
        }
        let total_nanos: u128 = self.inter_token_gaps.iter().map(|d| d.as_nanos()).sum();
        let mean_nanos = total_nanos / self.inter_token_gaps.len() as u128;
        Some(Duration::from_nanos(mean_nanos as u64))
    }

    /// Generation throughput in tokens per second.
    ///
    /// Computed over the decode window (from first token to last token) so that
    /// TTFT does not distort the throughput figure.
    ///
    /// Returns `None` if the decode window cannot be determined.
    pub fn tokens_per_second(&self) -> Option<f64> {
        if self.token_count < 2 {
            // Need at least two tokens to define a decode window.
            return None;
        }
        let first = self.first_token_time?;
        let last = self.last_token_time?;
        let elapsed = last.duration_since(first);
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs <= 0.0 {
            return None;
        }
        // Denominator is (token_count - 1) inter-token intervals.
        Some((self.token_count - 1) as f64 / elapsed_secs)
    }

    /// End-to-end request latency (from request start to generation finish).
    ///
    /// Returns `None` if [`finish`](Self::finish) has not been called.
    pub fn e2e_latency(&self) -> Option<Duration> {
        let last = self.last_token_time?;
        Some(last.duration_since(self.request_start))
    }

    /// Total number of completion tokens generated so far.
    pub fn completion_tokens(&self) -> usize {
        self.token_count
    }

    /// Take a point-in-time snapshot of the current metrics.
    pub fn snapshot(&self) -> StreamMetricsSnapshot {
        StreamMetricsSnapshot {
            ttft_ms: self.ttft().map(duration_to_ms),
            mean_tbt_ms: self.mean_tbt().map(duration_to_ms),
            p99_tbt_ms: self.p99_tbt().map(duration_to_ms),
            tokens_per_second: self.tokens_per_second(),
            e2e_latency_ms: self.e2e_latency().map(duration_to_ms),
            completion_tokens: self.token_count,
            prompt_tokens: self.prompt_tokens,
        }
    }
}

impl Default for RequestStreamMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── StreamMetricsSnapshot ────────────────────────────────────────────────────

/// A point-in-time snapshot of streaming metrics for a single request.
#[derive(Debug, Clone)]
pub struct StreamMetricsSnapshot {
    /// Time to first token in milliseconds, if available.
    pub ttft_ms: Option<f64>,
    /// Mean inter-token latency in milliseconds, if available.
    pub mean_tbt_ms: Option<f64>,
    /// P99 inter-token latency in milliseconds, if available.
    pub p99_tbt_ms: Option<f64>,
    /// Tokens per second during the decode phase, if available.
    pub tokens_per_second: Option<f64>,
    /// End-to-end latency in milliseconds, if available.
    pub e2e_latency_ms: Option<f64>,
    /// Number of completion tokens generated.
    pub completion_tokens: usize,
    /// Number of prompt tokens (prefill).
    pub prompt_tokens: usize,
}

impl StreamMetricsSnapshot {
    /// Format as a one-line human-readable summary.
    pub fn summary(&self) -> String {
        let ttft = opt_ms_str(self.ttft_ms, "TTFT");
        let tbt = opt_ms_str(self.mean_tbt_ms, "mean TBT");
        let tps = self
            .tokens_per_second
            .map(|v| format!("TPS={v:.1}"))
            .unwrap_or_else(|| "TPS=n/a".to_owned());
        let e2e = opt_ms_str(self.e2e_latency_ms, "E2E");
        let tokens = format!("tokens={}/{}", self.completion_tokens, self.prompt_tokens);
        format!("{ttft} | {tbt} | {tps} | {e2e} | {tokens}")
    }
}

// ── StreamingMetricsAggregator ───────────────────────────────────────────────

/// Aggregates [`StreamMetricsSnapshot`]s across multiple requests.
///
/// Useful for computing fleet-wide statistics in a serving system.
pub struct StreamingMetricsAggregator {
    snapshots: Vec<StreamMetricsSnapshot>,
}

impl StreamingMetricsAggregator {
    /// Create an empty aggregator.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    /// Add a snapshot from a completed request.
    pub fn record(&mut self, snapshot: StreamMetricsSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Number of requests recorded.
    pub fn num_requests(&self) -> usize {
        self.snapshots.len()
    }

    /// Average TTFT across all requests that have a TTFT measurement.
    pub fn avg_ttft_ms(&self) -> Option<f64> {
        avg_opt_field(self.snapshots.iter().map(|s| s.ttft_ms))
    }

    /// Average tokens-per-second across all requests that have a TPS measurement.
    pub fn avg_tokens_per_second(&self) -> Option<f64> {
        avg_opt_field(self.snapshots.iter().map(|s| s.tokens_per_second))
    }

    /// P99 end-to-end latency across all recorded requests.
    pub fn p99_e2e_ms(&self) -> Option<f64> {
        let mut values: Vec<f64> = self
            .snapshots
            .iter()
            .filter_map(|s| s.e2e_latency_ms)
            .collect();
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = percentile_index(values.len(), 99);
        Some(values[idx])
    }

    /// Total number of completion tokens across all recorded requests.
    pub fn total_completion_tokens(&self) -> usize {
        self.snapshots.iter().map(|s| s.completion_tokens).sum()
    }

    /// Generate a multi-line human-readable report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Requests recorded : {}\n", self.num_requests()));
        match self.avg_ttft_ms() {
            Some(v) => out.push_str(&format!("Avg TTFT          : {v:.2} ms\n")),
            None => out.push_str("Avg TTFT          : n/a\n"),
        }
        match self.avg_tokens_per_second() {
            Some(v) => out.push_str(&format!("Avg TPS           : {v:.2} tok/s\n")),
            None => out.push_str("Avg TPS           : n/a\n"),
        }
        match self.p99_e2e_ms() {
            Some(v) => out.push_str(&format!("P99 E2E latency   : {v:.2} ms\n")),
            None => out.push_str("P99 E2E latency   : n/a\n"),
        }
        out.push_str(&format!(
            "Total tokens      : {}\n",
            self.total_completion_tokens()
        ));
        out
    }
}

impl Default for StreamingMetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Convert a [`Duration`] to floating-point milliseconds.
fn duration_to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

/// Compute the p-th percentile (0–100) of a slice of [`Duration`]s.
///
/// The slice is copied and sorted internally so the original order is preserved.
/// Returns `None` when the slice is empty.
fn percentile_duration(samples: &[Duration], p: u8) -> Option<Duration> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort_unstable();
    let idx = percentile_index(sorted.len(), p);
    Some(sorted[idx])
}

/// Map a percentile (0–100) to a concrete 0-based index into a sorted slice of
/// length `n`.
fn percentile_index(n: usize, p: u8) -> usize {
    if n == 0 {
        return 0;
    }
    // Ceiling index: ceil((p / 100) * n) - 1, clamped.
    let idx = ((p as usize) * n).div_ceil(100);
    idx.saturating_sub(1).min(n - 1)
}

/// Compute the arithmetic mean of an iterator of `Option<f64>`, ignoring `None`s.
fn avg_opt_field(iter: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values: Vec<f64> = iter.flatten().collect();
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Format an optional millisecond value as `"label=X.Yms"` or `"label=n/a"`.
fn opt_ms_str(v: Option<f64>, label: &str) -> String {
    match v {
        Some(ms) => format!("{label}={ms:.2}ms"),
        None => format!("{label}=n/a"),
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_have_no_ttft() {
        let m = RequestStreamMetrics::new();
        assert!(m.ttft().is_none());
    }

    #[test]
    fn record_first_token_sets_ttft() {
        let mut m = RequestStreamMetrics::new();
        m.record_first_token();
        assert!(m.ttft().is_some());
    }

    #[test]
    fn record_multiple_tokens_counts_correctly() {
        let mut m = RequestStreamMetrics::new();
        m.record_first_token();
        m.record_token();
        m.record_token();
        assert_eq!(m.completion_tokens(), 3);
    }

    #[test]
    fn mean_tbt_available_after_three_tokens() {
        let mut m = RequestStreamMetrics::new();
        m.record_first_token();
        // Spin briefly so there is a measurable gap.
        std::thread::sleep(Duration::from_micros(100));
        m.record_token();
        std::thread::sleep(Duration::from_micros(100));
        m.record_token();
        assert!(m.mean_tbt().is_some());
    }

    #[test]
    fn tokens_per_second_positive_after_tokens() {
        let mut m = RequestStreamMetrics::new();
        m.record_first_token();
        std::thread::sleep(Duration::from_micros(200));
        m.record_token();
        let tps = m.tokens_per_second();
        assert!(tps.is_some(), "tps should be Some after 2 tokens");
        assert!(tps.expect("checked") > 0.0, "tps must be positive");
    }

    #[test]
    fn e2e_latency_available_after_finish() {
        let mut m = RequestStreamMetrics::new();
        m.record_first_token();
        m.record_token();
        m.finish();
        assert!(m.e2e_latency().is_some());
    }

    #[test]
    fn snapshot_summary_is_nonempty() {
        let mut m = RequestStreamMetrics::new_with_prompt_tokens(64);
        m.record_first_token();
        m.record_token();
        m.finish();
        let snap = m.snapshot();
        assert!(!snap.summary().is_empty());
    }

    #[test]
    fn aggregator_empty_returns_none_avg_ttft() {
        let agg = StreamingMetricsAggregator::new();
        assert!(agg.avg_ttft_ms().is_none());
    }

    #[test]
    fn aggregator_single_snapshot_avg_ttft_equals_snapshot() {
        let mut agg = StreamingMetricsAggregator::new();
        let snap = StreamMetricsSnapshot {
            ttft_ms: Some(42.0),
            mean_tbt_ms: None,
            p99_tbt_ms: None,
            tokens_per_second: None,
            e2e_latency_ms: Some(100.0),
            completion_tokens: 10,
            prompt_tokens: 5,
        };
        agg.record(snap);
        let avg = agg.avg_ttft_ms().expect("should have avg");
        assert!((avg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn aggregator_multiple_snapshots_averages_correctly() {
        let mut agg = StreamingMetricsAggregator::new();
        for ttft in [10.0_f64, 20.0, 30.0] {
            agg.record(StreamMetricsSnapshot {
                ttft_ms: Some(ttft),
                mean_tbt_ms: None,
                p99_tbt_ms: None,
                tokens_per_second: None,
                e2e_latency_ms: Some(ttft * 2.0),
                completion_tokens: 5,
                prompt_tokens: 2,
            });
        }
        let avg = agg.avg_ttft_ms().expect("should have avg");
        assert!((avg - 20.0).abs() < 1e-9, "expected avg=20.0, got {avg}");
    }

    #[test]
    fn aggregator_p99_e2e_returns_some_after_records() {
        let mut agg = StreamingMetricsAggregator::new();
        for ms in [100.0_f64, 200.0, 300.0, 400.0, 500.0] {
            agg.record(StreamMetricsSnapshot {
                ttft_ms: None,
                mean_tbt_ms: None,
                p99_tbt_ms: None,
                tokens_per_second: None,
                e2e_latency_ms: Some(ms),
                completion_tokens: 1,
                prompt_tokens: 1,
            });
        }
        assert!(agg.p99_e2e_ms().is_some());
    }

    #[test]
    fn aggregator_total_tokens_sums_correctly() {
        let mut agg = StreamingMetricsAggregator::new();
        for tokens in [10usize, 20, 30] {
            agg.record(StreamMetricsSnapshot {
                ttft_ms: None,
                mean_tbt_ms: None,
                p99_tbt_ms: None,
                tokens_per_second: None,
                e2e_latency_ms: None,
                completion_tokens: tokens,
                prompt_tokens: 0,
            });
        }
        assert_eq!(agg.total_completion_tokens(), 60);
    }

    #[test]
    fn aggregator_report_is_nonempty() {
        let mut agg = StreamingMetricsAggregator::new();
        agg.record(StreamMetricsSnapshot {
            ttft_ms: Some(50.0),
            mean_tbt_ms: Some(10.0),
            p99_tbt_ms: Some(20.0),
            tokens_per_second: Some(100.0),
            e2e_latency_ms: Some(1000.0),
            completion_tokens: 50,
            prompt_tokens: 128,
        });
        assert!(!agg.report().is_empty());
    }
}
