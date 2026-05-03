//! Hand-rolled Prometheus text exposition.
//!
//! Implements just enough of the 0.0.4 spec to emit counters, histograms and
//! gauges without pulling in the `prometheus` crate (which pulls in a
//! thread-local protobuf runtime we don't need).
//!
//! # Model
//!
//! A [`MetricsRegistry`] owns three kinds of metric families:
//!
//! - **counters** — monotonic `u64` counters, one per `(name, labels)` tuple.
//!   Rendered with a `_total` suffix.
//! - **histograms** — observation-based; each family has a fixed set of
//!   cumulative buckets (in seconds, by convention).  Rendered as
//!   `name_bucket{le=...}`, `name_sum` and `name_count`.
//! - **gauges** — last-write-wins `i64`.
//!
//! Metrics are keyed by a `(name, labels)` tuple.  Labels are
//! lexicographically sorted on insert to guarantee stable rendering.
//!
//! # Example
//!
//! ```
//! use oxibonsai_serve::metrics::MetricsRegistry;
//!
//! let reg = MetricsRegistry::new();
//! reg.inc_counter("oxibonsai_requests_total", &[("endpoint", "/health"), ("status", "200")]);
//! reg.observe_histogram(
//!     "oxibonsai_request_duration_seconds",
//!     &[("endpoint", "/health")],
//!     0.012,
//! );
//! reg.set_gauge("oxibonsai_inflight_requests", &[], 3);
//! let body = reg.render();
//! assert!(body.contains("oxibonsai_requests_total"));
//! assert!(body.contains("oxibonsai_request_duration_seconds_bucket"));
//! ```

use std::collections::BTreeMap;
use std::sync::RwLock;

// ─── Constants ────────────────────────────────────────────────────────────

/// Default histogram buckets (in seconds).
pub const DEFAULT_HISTOGRAM_BUCKETS: &[f64] = &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0];

// ─── Key type ─────────────────────────────────────────────────────────────

/// A metric key: `(name, sorted labels)`.
type LabelPairs = Vec<(String, String)>;
type Key = (String, LabelPairs);

fn make_key(name: &str, labels: &[(&str, &str)]) -> Key {
    let mut labels: LabelPairs = labels
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    (name.to_string(), labels)
}

fn format_labels(labels: &[(String, String)]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let inside = labels
        .iter()
        .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{inside}}}")
}

/// Escape backslash, double-quote and newline in a label value per the
/// Prometheus text-format spec.
fn escape_label_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

// ─── Histogram value ──────────────────────────────────────────────────────

/// Internal state of a single histogram time series.
#[derive(Debug, Clone)]
struct HistogramValue {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl HistogramValue {
    fn new(buckets: &[f64]) -> Self {
        let mut b = buckets.to_vec();
        // Ensure sorted ascending and that each bucket is finite.
        b.retain(|x| x.is_finite());
        b.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        let counts = vec![0u64; b.len()];
        Self {
            buckets: b,
            counts,
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.sum += value;
        self.count = self.count.saturating_add(1);
        for (i, b) in self.buckets.iter().enumerate() {
            if value <= *b {
                self.counts[i] = self.counts[i].saturating_add(1);
            }
        }
    }
}

// ─── Registry ─────────────────────────────────────────────────────────────

/// Inner mutable state guarded by a single [`RwLock`].
#[derive(Debug, Default)]
struct Inner {
    counters: BTreeMap<Key, u64>,
    gauges: BTreeMap<Key, i64>,
    histograms: BTreeMap<Key, HistogramValue>,
    /// Default buckets used when a histogram is first observed.
    default_buckets: Vec<f64>,
}

/// Thread-safe Prometheus metrics registry.
#[derive(Debug)]
pub struct MetricsRegistry {
    inner: RwLock<Inner>,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Build a new empty registry using [`DEFAULT_HISTOGRAM_BUCKETS`].
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                counters: BTreeMap::new(),
                gauges: BTreeMap::new(),
                histograms: BTreeMap::new(),
                default_buckets: DEFAULT_HISTOGRAM_BUCKETS.to_vec(),
            }),
        }
    }

    /// Build a registry with a custom default bucket set.
    pub fn with_buckets(buckets: &[f64]) -> Self {
        Self {
            inner: RwLock::new(Inner {
                counters: BTreeMap::new(),
                gauges: BTreeMap::new(),
                histograms: BTreeMap::new(),
                default_buckets: buckets.to_vec(),
            }),
        }
    }

    /// Increment a counter by 1.
    pub fn inc_counter(&self, name: &str, labels: &[(&str, &str)]) {
        self.add_counter(name, labels, 1);
    }

    /// Add `value` to a counter.
    pub fn add_counter(&self, name: &str, labels: &[(&str, &str)], value: u64) {
        let key = make_key(name, labels);
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let entry = guard.counters.entry(key).or_insert(0);
        *entry = entry.saturating_add(value);
    }

    /// Set a gauge to `value`.
    pub fn set_gauge(&self, name: &str, labels: &[(&str, &str)], value: i64) {
        let key = make_key(name, labels);
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.gauges.insert(key, value);
    }

    /// Add `delta` to a gauge (may be negative).
    pub fn add_gauge(&self, name: &str, labels: &[(&str, &str)], delta: i64) {
        let key = make_key(name, labels);
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let entry = guard.gauges.entry(key).or_insert(0);
        *entry = entry.saturating_add(delta);
    }

    /// Observe a histogram value.  The histogram is created lazily with the
    /// registry's default buckets on first observation.
    pub fn observe_histogram(&self, name: &str, labels: &[(&str, &str)], value: f64) {
        let key = make_key(name, labels);
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let buckets = guard.default_buckets.clone();
        let entry = guard
            .histograms
            .entry(key)
            .or_insert_with(|| HistogramValue::new(&buckets));
        entry.observe(value);
    }

    /// Get a counter value (0 if absent).
    pub fn counter_value(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let key = make_key(name, labels);
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.counters.get(&key).copied().unwrap_or(0)
    }

    /// Get a gauge value (0 if absent).
    pub fn gauge_value(&self, name: &str, labels: &[(&str, &str)]) -> i64 {
        let key = make_key(name, labels);
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.gauges.get(&key).copied().unwrap_or(0)
    }

    /// Get histogram count (number of observations).  Returns 0 if the
    /// histogram has not been observed yet.
    pub fn histogram_count(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let key = make_key(name, labels);
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.histograms.get(&key).map(|h| h.count).unwrap_or(0)
    }

    /// Get the per-bucket counts for a histogram, in the same order as the
    /// bucket vector supplied at creation.  Returns an empty vector if the
    /// histogram does not exist.
    pub fn histogram_bucket_counts(&self, name: &str, labels: &[(&str, &str)]) -> Vec<u64> {
        let key = make_key(name, labels);
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard
            .histograms
            .get(&key)
            .map(|h| h.counts.clone())
            .unwrap_or_default()
    }

    /// Render all metrics as a Prometheus 0.0.4 text-format string.
    pub fn render(&self) -> String {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        let mut out = String::new();

        // Group by name so we emit HELP/TYPE once per family.
        let mut counter_names: BTreeMap<String, Vec<(&LabelPairs, &u64)>> = BTreeMap::new();
        for ((n, l), v) in &guard.counters {
            counter_names.entry(n.clone()).or_default().push((l, v));
        }
        for (name, series) in &counter_names {
            out.push_str(&format!(
                "# HELP {name} Counter auto-registered by oxibonsai-serve.\n"
            ));
            out.push_str(&format!("# TYPE {name} counter\n"));
            for (labels, value) in series {
                out.push_str(&format!("{}{} {}\n", name, format_labels(labels), value));
            }
        }

        let mut gauge_names: BTreeMap<String, Vec<(&LabelPairs, &i64)>> = BTreeMap::new();
        for ((n, l), v) in &guard.gauges {
            gauge_names.entry(n.clone()).or_default().push((l, v));
        }
        for (name, series) in &gauge_names {
            out.push_str(&format!(
                "# HELP {name} Gauge auto-registered by oxibonsai-serve.\n"
            ));
            out.push_str(&format!("# TYPE {name} gauge\n"));
            for (labels, value) in series {
                out.push_str(&format!("{}{} {}\n", name, format_labels(labels), value));
            }
        }

        let mut hist_names: BTreeMap<String, Vec<(&LabelPairs, &HistogramValue)>> = BTreeMap::new();
        for ((n, l), v) in &guard.histograms {
            hist_names.entry(n.clone()).or_default().push((l, v));
        }
        for (name, series) in &hist_names {
            out.push_str(&format!(
                "# HELP {name} Histogram auto-registered by oxibonsai-serve.\n"
            ));
            out.push_str(&format!("# TYPE {name} histogram\n"));
            for (labels, hv) in series {
                // Bucket lines.
                for (i, b) in hv.buckets.iter().enumerate() {
                    let mut ls = (*labels).clone();
                    ls.push(("le".to_string(), format_float(*b)));
                    ls.sort_by(|a, c| a.0.cmp(&c.0));
                    out.push_str(&format!(
                        "{}_bucket{} {}\n",
                        name,
                        format_labels(&ls),
                        hv.counts[i]
                    ));
                }
                // +Inf bucket.
                let mut inf_labels = (*labels).clone();
                inf_labels.push(("le".to_string(), "+Inf".to_string()));
                inf_labels.sort_by(|a, c| a.0.cmp(&c.0));
                out.push_str(&format!(
                    "{}_bucket{} {}\n",
                    name,
                    format_labels(&inf_labels),
                    hv.count
                ));
                // _sum
                out.push_str(&format!(
                    "{}_sum{} {}\n",
                    name,
                    format_labels(labels),
                    format_float(hv.sum)
                ));
                // _count
                out.push_str(&format!(
                    "{}_count{} {}\n",
                    name,
                    format_labels(labels),
                    hv.count
                ));
            }
        }

        out
    }
}

/// Format a floating-point value for Prometheus output.
///
/// Integer-valued floats are printed without a trailing `.0` to match what the
/// official Prometheus client libraries emit (e.g. `1`, not `1.0`).
fn format_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }
    if v.fract() == 0.0 && v.abs() < 1.0e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_roundtrip() {
        let r = MetricsRegistry::new();
        r.inc_counter("foo_total", &[("a", "x")]);
        r.inc_counter("foo_total", &[("a", "x")]);
        assert_eq!(r.counter_value("foo_total", &[("a", "x")]), 2);
    }

    #[test]
    fn gauge_set() {
        let r = MetricsRegistry::new();
        r.set_gauge("inflight", &[], 3);
        assert_eq!(r.gauge_value("inflight", &[]), 3);
        r.set_gauge("inflight", &[], 1);
        assert_eq!(r.gauge_value("inflight", &[]), 1);
    }

    #[test]
    fn histogram_observe_count() {
        let r = MetricsRegistry::new();
        r.observe_histogram("lat", &[("endpoint", "/h")], 0.005);
        r.observe_histogram("lat", &[("endpoint", "/h")], 0.20);
        assert_eq!(r.histogram_count("lat", &[("endpoint", "/h")]), 2);
    }

    #[test]
    fn render_includes_type_and_help() {
        let r = MetricsRegistry::new();
        r.inc_counter("x_total", &[]);
        let body = r.render();
        assert!(body.contains("# HELP x_total"));
        assert!(body.contains("# TYPE x_total counter"));
        assert!(body.contains("x_total 1"));
    }

    #[test]
    fn render_histogram_has_buckets_sum_count() {
        let r = MetricsRegistry::new();
        r.observe_histogram("lat_seconds", &[], 0.02);
        let body = r.render();
        assert!(body.contains("lat_seconds_bucket{le=\"0.01\"}"));
        assert!(body.contains("lat_seconds_bucket{le=\"+Inf\"}"));
        assert!(body.contains("lat_seconds_sum"));
        assert!(body.contains("lat_seconds_count"));
    }

    #[test]
    fn label_escape() {
        assert_eq!(escape_label_value("a\"b"), "a\\\"b");
        assert_eq!(escape_label_value("a\\b"), "a\\\\b");
        assert_eq!(escape_label_value("a\nb"), "a\\nb");
    }
}
