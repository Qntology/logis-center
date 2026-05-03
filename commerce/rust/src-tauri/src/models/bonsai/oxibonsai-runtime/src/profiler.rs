//! Inference profiler: per-layer timing, memory, and FLOP accounting.
//!
//! The profiler uses `std::time::Instant` for timing and provides detailed
//! per-layer and per-phase breakdowns suitable for performance analysis.
//!
//! ## Usage
//!
//! ```rust
//! use oxibonsai_runtime::profiler::{Profiler, flop_counter};
//!
//! let mut prof = Profiler::new();
//! prof.begin_trace();
//!
//! let result = prof.profile("attention.layer0", flop_counter::attention(512, 64, 8), || {
//!     42u32
//! });
//!
//! let trace = prof.end_trace().expect("trace should exist");
//! println!("{}", trace.summary());
//! ```

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

// ─── ProfileEvent ────────────────────────────────────────────────────────────

/// A single profiled event (one layer or one phase).
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Human-readable name, e.g. `"attention.layer3"`.
    pub name: String,
    /// Wall-clock duration of the event.
    pub duration: Duration,
    /// Signed memory delta in bytes (positive = allocated, negative = freed).
    pub memory_delta_bytes: i64,
    /// Estimated floating point operations performed.
    pub flops: u64,
    /// Arbitrary key-value metadata attached to this event.
    pub metadata: HashMap<String, String>,
}

impl ProfileEvent {
    /// Create a new event with zero duration, no memory delta, and no FLOPs.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            duration: Duration::ZERO,
            memory_delta_bytes: 0,
            flops: 0,
            metadata: HashMap::new(),
        }
    }

    /// Builder: attach an estimated FLOP count.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = flops;
        self
    }

    /// Builder: attach a key-value metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Duration in milliseconds (floating point).
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000.0
    }

    /// Estimated GFLOPs per second for this event.
    ///
    /// Returns `0.0` if duration is zero or flops is zero.
    pub fn gflops_per_second(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 || self.flops == 0 {
            return 0.0;
        }
        (self.flops as f64) / secs / 1e9
    }
}

// ─── ProfileGuard ─────────────────────────────────────────────────────────────

/// RAII guard that measures wall-clock time for a scope and appends a
/// [`ProfileEvent`] to the owning [`Profiler`] when dropped.
///
/// Obtain via [`Profiler::begin_event`] is the manual pair; for a scoped
/// version use [`Profiler::profile`].  `ProfileGuard` is exposed so callers
/// can adjust FLOP counts mid-scope via [`Self::set_flops`].
pub struct ProfileGuard<'a> {
    profiler: &'a mut Profiler,
    name: String,
    start: Instant,
    flops: u64,
}

impl<'a> ProfileGuard<'a> {
    /// Update the estimated FLOP count before the guard is dropped.
    pub fn set_flops(&mut self, flops: u64) {
        self.flops = flops;
    }
}

impl<'a> Drop for ProfileGuard<'a> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if !self.profiler.enabled {
            return;
        }
        if let Some(trace) = self.profiler.current_trace.as_mut() {
            let event = ProfileEvent {
                name: self.name.clone(),
                duration: elapsed,
                memory_delta_bytes: 0,
                flops: self.flops,
                metadata: HashMap::new(),
            };
            trace.total_flops = trace.total_flops.saturating_add(event.flops);
            trace.events.push(event);
        }
    }
}

// ─── ProfileTrace ─────────────────────────────────────────────────────────────

/// Complete record of one inference pass.
#[derive(Debug, Clone, Default)]
pub struct ProfileTrace {
    /// Ordered list of events that occurred during the pass.
    pub events: Vec<ProfileEvent>,
    /// Total wall-clock duration of the entire trace.
    pub total_duration: Duration,
    /// Peak resident memory observed during the trace (best-effort).
    pub peak_memory_bytes: usize,
    /// Sum of estimated FLOPs across all events.
    pub total_flops: u64,
}

impl ProfileTrace {
    /// Return the `n` events with the longest duration, sorted descending.
    pub fn top_events(&self, n: usize) -> Vec<&ProfileEvent> {
        let mut refs: Vec<&ProfileEvent> = self.events.iter().collect();
        refs.sort_by_key(|b| std::cmp::Reverse(b.duration));
        refs.into_iter().take(n).collect()
    }

    /// Sum the durations of all events whose names start with `prefix`.
    pub fn duration_for_prefix(&self, prefix: &str) -> Duration {
        self.events
            .iter()
            .filter(|e| e.name.starts_with(prefix))
            .map(|e| e.duration)
            .fold(Duration::ZERO, |acc, d| acc + d)
    }

    /// Average duration of events whose names start with `prefix`.
    ///
    /// Returns `None` if no events match.
    pub fn avg_duration_for_prefix(&self, prefix: &str) -> Option<Duration> {
        let matching: Vec<Duration> = self
            .events
            .iter()
            .filter(|e| e.name.starts_with(prefix))
            .map(|e| e.duration)
            .collect();

        if matching.is_empty() {
            return None;
        }

        let total_nanos: u128 = matching.iter().map(|d| d.as_nanos()).sum();
        let avg_nanos = total_nanos / matching.len() as u128;
        Some(Duration::from_nanos(avg_nanos as u64))
    }

    /// Human-readable summary of the trace.
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(
            out,
            "=== ProfileTrace: {:.3} ms total, {} events, {:.2} GFLOPs ===",
            self.total_duration.as_secs_f64() * 1_000.0,
            self.events.len(),
            self.aggregate_gflops(),
        );
        let _ = writeln!(out, "  peak_memory: {} bytes", self.peak_memory_bytes);

        let top = self.top_events(10);
        if !top.is_empty() {
            let _ = writeln!(out, "  Top events by duration:");
            for ev in top {
                let _ = writeln!(
                    out,
                    "    {:40} {:8.3} ms  {:6.2} GFLOPs/s",
                    ev.name,
                    ev.duration_ms(),
                    ev.gflops_per_second(),
                );
            }
        }

        out
    }

    /// Overall GFLOPs/s: total_flops / total_duration.
    ///
    /// Returns `0.0` if duration is zero.
    pub fn aggregate_gflops(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs <= 0.0 || self.total_flops == 0 {
            return 0.0;
        }
        (self.total_flops as f64) / secs / 1e9
    }

    /// Map from event name to duration in milliseconds.
    ///
    /// If multiple events share the same name, their durations are summed.
    pub fn layer_breakdown(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for ev in &self.events {
            *map.entry(ev.name.clone()).or_insert(0.0) += ev.duration_ms();
        }
        map
    }
}

// ─── Profiler ─────────────────────────────────────────────────────────────────

/// Main inference profiler.
///
/// Maintains a stack of completed [`ProfileTrace`]s and an optional
/// in-progress trace.  Use [`Self::begin_trace`] / [`Self::end_trace`] to
/// bracket an inference pass, and [`Self::profile`] (or the
/// `begin_event`/`end_event` pair) to record individual operations.
pub struct Profiler {
    /// All completed traces.
    traces: Vec<ProfileTrace>,
    /// The trace currently being built, if any.
    current_trace: Option<ProfileTrace>,
    /// Wall-clock start of the current trace.
    current_trace_start: Option<Instant>,
    /// When `false`, `profile()` still runs closures but records nothing.
    enabled: bool,
    /// RSS at profiler construction (reserved for future memory-delta tracking).
    #[allow(dead_code)]
    memory_baseline: usize,
}

impl Profiler {
    /// Create an enabled profiler.
    pub fn new() -> Self {
        Self {
            traces: Vec::new(),
            current_trace: None,
            current_trace_start: None,
            enabled: true,
            memory_baseline: crate::memory::get_rss_bytes() as usize,
        }
    }

    /// Create a profiler with an explicit enabled/disabled flag.
    pub fn enabled(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::new()
        }
    }

    /// Whether the profiler is currently recording events.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a new trace.
    ///
    /// Any previous in-progress trace is discarded; call [`Self::end_trace`]
    /// first if you want to keep it.
    pub fn begin_trace(&mut self) {
        if !self.enabled {
            return;
        }
        self.current_trace = Some(ProfileTrace::default());
        self.current_trace_start = Some(Instant::now());
    }

    /// Finalise the current trace, push it to the completed list, and return
    /// a clone.
    ///
    /// Returns `None` if no trace is in progress.
    pub fn end_trace(&mut self) -> Option<ProfileTrace> {
        let trace_start = self.current_trace_start.take()?;
        let mut trace = self.current_trace.take()?;
        trace.total_duration = trace_start.elapsed();
        trace.peak_memory_bytes = crate::memory::get_rss_bytes() as usize;
        self.traces.push(trace.clone());
        Some(trace)
    }

    /// Record the start of an event and return the `Instant`.
    ///
    /// Pair with [`Self::end_event`].
    pub fn begin_event(&mut self, _name: impl Into<String>) -> Instant {
        Instant::now()
    }

    /// Complete an event started at `start_time` and record it in the active
    /// trace (if any).
    pub fn end_event(&mut self, name: impl Into<String>, start_time: Instant, flops: u64) {
        if !self.enabled {
            return;
        }
        let elapsed = start_time.elapsed();
        if let Some(trace) = self.current_trace.as_mut() {
            let event = ProfileEvent {
                name: name.into(),
                duration: elapsed,
                memory_delta_bytes: 0,
                flops,
                metadata: HashMap::new(),
            };
            trace.total_flops = trace.total_flops.saturating_add(event.flops);
            trace.events.push(event);
        }
    }

    /// Time `f`, record the event as `name` with `flops` estimated FLOPs, and
    /// return whatever `f` returns.
    ///
    /// When the profiler is disabled the closure is still executed; only
    /// recording is skipped.
    pub fn profile<F, R>(&mut self, name: impl Into<String>, flops: u64, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        let name_str: String = name.into();
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        if let Some(trace) = self.current_trace.as_mut() {
            let event = ProfileEvent {
                name: name_str,
                duration: elapsed,
                memory_delta_bytes: 0,
                flops,
                metadata: HashMap::new(),
            };
            trace.total_flops = trace.total_flops.saturating_add(event.flops);
            trace.events.push(event);
        }
        result
    }

    /// Return a scoped guard that records an event when dropped.
    ///
    /// This allows recording events that span a `?`-early-return path without
    /// an explicit `end_event` call.
    pub fn scoped<'a>(&'a mut self, name: impl Into<String>) -> ProfileGuard<'a> {
        ProfileGuard {
            profiler: self,
            name: name.into(),
            start: Instant::now(),
            flops: 0,
        }
    }

    /// All completed traces (oldest first).
    pub fn traces(&self) -> &[ProfileTrace] {
        &self.traces
    }

    /// The most recently completed trace, if any.
    pub fn last_trace(&self) -> Option<&ProfileTrace> {
        self.traces.last()
    }

    /// Aggregate statistics across all completed traces.
    pub fn aggregate_stats(&self) -> AggregateStats {
        let num_traces = self.traces.len();
        if num_traces == 0 {
            return AggregateStats {
                num_traces: 0,
                total_duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
                p50_duration: Duration::ZERO,
                p99_duration: Duration::ZERO,
                total_flops: 0,
                avg_tokens_per_second: 0.0,
            };
        }

        let total_duration: Duration = self
            .traces
            .iter()
            .map(|t| t.total_duration)
            .fold(Duration::ZERO, |acc, d| acc + d);

        let avg_nanos = total_duration.as_nanos() / num_traces as u128;
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        let total_flops: u64 = self
            .traces
            .iter()
            .map(|t| t.total_flops)
            .fold(0u64, |acc, f| acc.saturating_add(f));

        // Percentile computation on sorted durations
        let mut sorted_nanos: Vec<u128> = self
            .traces
            .iter()
            .map(|t| t.total_duration.as_nanos())
            .collect();
        sorted_nanos.sort_unstable();

        let p50_idx = (num_traces as f64 * 0.50) as usize;
        let p99_idx = ((num_traces as f64 * 0.99) as usize).min(num_traces - 1);

        let p50_nanos = sorted_nanos.get(p50_idx).copied().unwrap_or(0);
        let p99_nanos = sorted_nanos.get(p99_idx).copied().unwrap_or(0);

        let p50_duration = Duration::from_nanos(p50_nanos as u64);
        let p99_duration = Duration::from_nanos(p99_nanos as u64);

        // avg tokens/s approximation: assume 1 "token" per trace for now
        let avg_tokens_per_second = if avg_duration.as_secs_f64() > 0.0 {
            1.0 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        AggregateStats {
            num_traces,
            total_duration,
            avg_duration,
            p50_duration,
            p99_duration,
            total_flops,
            avg_tokens_per_second,
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── AggregateStats ───────────────────────────────────────────────────────────

/// Aggregate statistics across multiple completed [`ProfileTrace`]s.
#[derive(Debug, Clone)]
pub struct AggregateStats {
    /// Number of traces included in the aggregate.
    pub num_traces: usize,
    /// Sum of all trace durations.
    pub total_duration: Duration,
    /// Mean trace duration.
    pub avg_duration: Duration,
    /// Median (p50) trace duration.
    pub p50_duration: Duration,
    /// 99th-percentile trace duration.
    pub p99_duration: Duration,
    /// Sum of all FLOPs across all traces.
    pub total_flops: u64,
    /// Approximate average tokens per second (1 token per trace).
    pub avg_tokens_per_second: f64,
}

impl AggregateStats {
    /// Human-readable aggregate summary.
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(256);
        let _ = writeln!(out, "=== AggregateStats ({} traces) ===", self.num_traces);
        let _ = writeln!(
            out,
            "  total_duration : {:.3} ms",
            self.total_duration.as_secs_f64() * 1_000.0,
        );
        let _ = writeln!(
            out,
            "  avg_duration   : {:.3} ms",
            self.avg_duration.as_secs_f64() * 1_000.0,
        );
        let _ = writeln!(
            out,
            "  p50_duration   : {:.3} ms",
            self.p50_duration.as_secs_f64() * 1_000.0,
        );
        let _ = writeln!(
            out,
            "  p99_duration   : {:.3} ms",
            self.p99_duration.as_secs_f64() * 1_000.0,
        );
        let _ = writeln!(out, "  total_flops    : {}", self.total_flops);
        let _ = writeln!(out, "  avg_tok/s      : {:.2}", self.avg_tokens_per_second,);
        out
    }
}

// ─── flop_counter ─────────────────────────────────────────────────────────────

/// FLOP estimation helpers for common transformer operations.
///
/// All formulas count multiply-add pairs as **2** FLOPs (the standard
/// "operations" convention used in most ML literature).
pub mod flop_counter {
    /// FLOPs for a general matrix multiplication A\[m,k\] × B\[k,n\].
    ///
    /// Formula: `2 * m * k * n`
    pub fn matmul(m: usize, k: usize, n: usize) -> u64 {
        2u64.saturating_mul(m as u64)
            .saturating_mul(k as u64)
            .saturating_mul(n as u64)
    }

    /// FLOPs for a linear (fully-connected) layer without bias.
    ///
    /// Equivalent to `matmul(batch, in_features, out_features)`.
    pub fn linear(batch: usize, in_features: usize, out_features: usize) -> u64 {
        matmul(batch, in_features, out_features)
    }

    /// FLOPs for scaled dot-product attention.
    ///
    /// Formula: `2 * seq_len^2 * head_dim * num_heads`
    ///
    /// This accounts for the QK^T and softmax(QK^T)V matmuls but not
    /// the projection layers.
    pub fn attention(seq_len: usize, head_dim: usize, num_heads: usize) -> u64 {
        2u64.saturating_mul(seq_len as u64)
            .saturating_mul(seq_len as u64)
            .saturating_mul(head_dim as u64)
            .saturating_mul(num_heads as u64)
    }

    /// FLOPs for RMSNorm over a sequence.
    ///
    /// Formula: `5 * seq_len * hidden`  (square, sum, rsqrt, scale, multiply).
    pub fn rms_norm(seq_len: usize, hidden: usize) -> u64 {
        5u64.saturating_mul(seq_len as u64)
            .saturating_mul(hidden as u64)
    }

    /// FLOPs for a SwiGLU feed-forward network.
    ///
    /// Counts three linear projections:
    /// - gate projection:  `batch × hidden × intermediate`
    /// - up projection:    `batch × hidden × intermediate`
    /// - down projection:  `batch × intermediate × hidden`
    ///
    /// Plus the element-wise SiLU gate (2 ops per element):
    /// `2 * batch * intermediate`
    ///
    /// Total: `2*batch*(2*hidden*intermediate + intermediate*hidden + intermediate)`
    pub fn swiglu_ffn(seq_len: usize, hidden: usize, intermediate: usize) -> u64 {
        // gate + up projections: each is seq_len × hidden × intermediate
        let gate_up = 2u64
            .saturating_mul(seq_len as u64)
            .saturating_mul(hidden as u64)
            .saturating_mul(intermediate as u64);
        // down projection: seq_len × intermediate × hidden
        let down = 2u64
            .saturating_mul(seq_len as u64)
            .saturating_mul(intermediate as u64)
            .saturating_mul(hidden as u64);
        // SiLU element-wise gate (2 ops per element)
        let silu = 2u64
            .saturating_mul(seq_len as u64)
            .saturating_mul(intermediate as u64);

        gate_up
            .saturating_add(gate_up)
            .saturating_add(down)
            .saturating_add(silu)
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_event_new() {
        let ev = ProfileEvent::new("test.layer");
        assert_eq!(ev.name, "test.layer");
        assert_eq!(ev.flops, 0);
        assert_eq!(ev.duration, Duration::ZERO);
    }

    #[test]
    fn profile_event_builders() {
        let ev = ProfileEvent::new("layer")
            .with_flops(1_000_000)
            .with_metadata("dtype", "f16");
        assert_eq!(ev.flops, 1_000_000);
        assert_eq!(ev.metadata["dtype"], "f16");
    }

    #[test]
    fn profile_event_duration_ms() {
        let mut ev = ProfileEvent::new("x");
        ev.duration = Duration::from_millis(250);
        assert!((ev.duration_ms() - 250.0).abs() < 1e-6);
    }

    #[test]
    fn profile_event_gflops_zero_duration() {
        let mut ev = ProfileEvent::new("x");
        ev.flops = 1_000_000_000;
        assert_eq!(ev.gflops_per_second(), 0.0);
    }

    #[test]
    fn flop_counter_matmul_formula() {
        assert_eq!(flop_counter::matmul(2, 3, 4), 48);
    }

    #[test]
    fn flop_counter_linear_formula() {
        assert_eq!(flop_counter::linear(1, 4, 8), 64);
    }

    #[test]
    fn flop_counter_attention_formula() {
        // 2 * seq_len^2 * head_dim * num_heads = 2 * 4 * 4 * 8 * 2 = 512
        assert_eq!(flop_counter::attention(4, 8, 2), 512);
    }
}
