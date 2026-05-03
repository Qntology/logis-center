//! Throughput benchmarking for LLM inference.
//!
//! [`ThroughputBenchmark`] accumulates timing information from repeated
//! generation runs and produces a [`ThroughputResult`] with token-per-second
//! statistics, latency breakdowns, and percentile metrics.

use std::time::{Duration, Instant};

use serde::Serialize;

// ──────────────────────────────────────────────────────────────────────────────
// Timing helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Time the execution of `f`, returning both the result and the elapsed duration.
pub fn time_fn<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Compute the p-th percentile of `values` (0.0 ≤ p ≤ 100.0).
///
/// `values` is sorted in place. Uses linear interpolation between adjacent
/// elements when the index is not an integer. Returns `0.0` for an empty slice.
pub fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p_clamped = p.clamp(0.0, 100.0);
    let index = p_clamped / 100.0 * (values.len() - 1) as f32;
    let lo = index.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let frac = index - lo as f32;
    values[lo] * (1.0 - frac) + values[hi] * frac
}

// ──────────────────────────────────────────────────────────────────────────────
// ThroughputResult
// ──────────────────────────────────────────────────────────────────────────────

/// Statistics from a completed throughput benchmark.
#[derive(Debug, Serialize)]
pub struct ThroughputResult {
    /// Mean tokens per second across all measurement runs.
    pub tokens_per_second: f32,
    /// Mean prefill latency in milliseconds.
    pub prefill_ms: f32,
    /// Mean per-token decode latency in milliseconds.
    pub decode_ms_per_token: f32,
    /// Total number of tokens generated across all runs.
    pub total_tokens: usize,
    /// Number of measurement runs performed.
    pub runs: usize,
    /// Minimum tokens per second observed.
    pub min_tps: f32,
    /// Maximum tokens per second observed.
    pub max_tps: f32,
    /// 50th-percentile tokens per second.
    pub p50_tps: f32,
    /// 95th-percentile tokens per second.
    pub p95_tps: f32,
}

impl ThroughputResult {
    /// One-line human-readable summary of throughput statistics.
    pub fn summary(&self) -> String {
        format!(
            "Throughput: {:.1} t/s (p50: {:.1}, p95: {:.1})",
            self.tokens_per_second, self.p50_tps, self.p95_tps
        )
    }

    /// Return `true` if the mean throughput meets or exceeds `target_tps`.
    pub fn meets_target(&self, target_tps: f32) -> bool {
        self.tokens_per_second >= target_tps
    }

    /// Human-readable latency breakdown string.
    pub fn latency_breakdown(&self) -> String {
        format!(
            "Prefill: {:.2} ms | Decode: {:.3} ms/token",
            self.prefill_ms, self.decode_ms_per_token
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ThroughputBenchmark
// ──────────────────────────────────────────────────────────────────────────────

/// Builder for throughput benchmark runs.
///
/// Collects timing data from caller-supplied closures rather than running the
/// model directly, keeping this crate decoupled from the inference engine.
pub struct ThroughputBenchmark {
    /// Number of warm-up runs (results discarded).
    pub warmup_runs: usize,
    /// Number of measurement runs (results aggregated).
    pub measurement_runs: usize,
    /// The prompt used for benchmarking.
    pub prompt: String,
    /// Maximum tokens to generate per run.
    pub max_tokens: usize,
}

impl ThroughputBenchmark {
    /// Create a benchmark with 3 warm-up runs and 10 measurement runs.
    pub fn new(prompt: &str, max_tokens: usize) -> Self {
        Self {
            warmup_runs: 3,
            measurement_runs: 10,
            prompt: prompt.to_string(),
            max_tokens,
        }
    }

    /// Override the number of warm-up runs.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_runs = warmup;
        self
    }

    /// Override the number of measurement runs.
    pub fn with_runs(mut self, runs: usize) -> Self {
        self.measurement_runs = runs;
        self
    }

    /// Run the benchmark using caller-supplied timing data.
    ///
    /// `run_timings` is a slice of `(prefill_ms, decode_ms, tokens_generated)` tuples,
    /// one per measurement run (warm-up timings should already be excluded by the caller).
    ///
    /// This method computes aggregate statistics from the provided data without
    /// calling the inference engine itself, allowing flexible integration.
    pub fn from_timings(&self, run_timings: &[(f32, f32, usize)]) -> ThroughputResult {
        if run_timings.is_empty() {
            return ThroughputResult {
                tokens_per_second: 0.0,
                prefill_ms: 0.0,
                decode_ms_per_token: 0.0,
                total_tokens: 0,
                runs: 0,
                min_tps: 0.0,
                max_tps: 0.0,
                p50_tps: 0.0,
                p95_tps: 0.0,
            };
        }

        let n = run_timings.len() as f32;
        let mut tps_values: Vec<f32> = Vec::with_capacity(run_timings.len());
        let mut total_prefill_ms = 0.0f32;
        let mut total_decode_ms = 0.0f32;
        let mut total_tokens = 0usize;

        for &(prefill_ms, decode_ms, tokens) in run_timings {
            total_prefill_ms += prefill_ms;
            total_decode_ms += decode_ms;
            total_tokens += tokens;

            let total_ms = prefill_ms + decode_ms;
            let tps = if total_ms > 0.0 {
                tokens as f32 / (total_ms / 1000.0)
            } else {
                0.0
            };
            tps_values.push(tps);
        }

        let mean_tps = tps_values.iter().copied().sum::<f32>() / n;
        let min_tps = tps_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_tps = tps_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let p50_tps = percentile(tps_values.clone(), 50.0);
        let p95_tps = percentile(tps_values, 95.0);

        let mean_prefill_ms = total_prefill_ms / n;
        let mean_decode_ms_per_token = if total_tokens > 0 {
            total_decode_ms / total_tokens as f32
        } else {
            0.0
        };

        ThroughputResult {
            tokens_per_second: mean_tps,
            prefill_ms: mean_prefill_ms,
            decode_ms_per_token: mean_decode_ms_per_token,
            total_tokens,
            runs: run_timings.len(),
            min_tps,
            max_tps,
            p50_tps,
            p95_tps,
        }
    }
}
