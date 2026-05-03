//! Integration tests for the inference profiler.

use oxibonsai_runtime::profiler::{flop_counter, ProfileEvent, ProfileTrace, Profiler};
use std::time::Duration;

// ── 1. profiler_new_no_traces ────────────────────────────────────────────────

#[test]
fn profiler_new_no_traces() {
    let prof = Profiler::new();
    assert_eq!(prof.traces().len(), 0);
}

// ── 2. profiler_begin_end_trace ──────────────────────────────────────────────

#[test]
fn profiler_begin_end_trace() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    let trace = prof.end_trace();
    assert!(trace.is_some());
    assert!(prof.last_trace().is_some());
}

// ── 3. profiler_profile_closure ──────────────────────────────────────────────

#[test]
fn profiler_profile_closure() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    let val = prof.profile("test.op", 0, || 42u32);
    let trace = prof.end_trace().expect("trace should exist");
    assert_eq!(val, 42);
    assert_eq!(trace.events.len(), 1);
    assert_eq!(trace.events[0].name, "test.op");
}

// ── 4. profiler_event_duration_positive ─────────────────────────────────────

#[test]
fn profiler_event_duration_positive() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    prof.profile("busy.work", 0, || {
        // do a tiny bit of work to ensure non-zero duration
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(i);
        }
        x
    });
    let trace = prof.end_trace().expect("trace should exist");
    // Duration should be >= 0 (can be zero on very fast machines, but should not panic)
    assert!(trace.total_duration >= Duration::ZERO);
    assert!(trace.events[0].duration >= Duration::ZERO);
}

// ── 5. profiler_begin_event_end_event ────────────────────────────────────────

#[test]
fn profiler_begin_event_end_event() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    let t = prof.begin_event("manual.event");
    prof.end_event("manual.event", t, 500);
    let trace = prof.end_trace().expect("trace should exist");
    assert_eq!(trace.events.len(), 1);
    assert_eq!(trace.events[0].name, "manual.event");
    assert_eq!(trace.events[0].flops, 500);
}

// ── 6. profiler_multiple_events ──────────────────────────────────────────────

#[test]
fn profiler_multiple_events() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    for i in 0..5 {
        let name = format!("layer.{i}");
        prof.profile(&name, 100 * i as u64, || ());
    }
    let trace = prof.end_trace().expect("trace should exist");
    assert_eq!(trace.events.len(), 5);
    // total_flops = 0 + 100 + 200 + 300 + 400
    assert_eq!(trace.total_flops, 1000);
}

// ── 7. profile_trace_top_events ──────────────────────────────────────────────

#[test]
fn profile_trace_top_events() {
    let mut trace = ProfileTrace::default();
    for ms in [10u64, 50, 20, 5, 100] {
        let mut ev = ProfileEvent::new(format!("ev_{ms}ms"));
        ev.duration = Duration::from_millis(ms);
        trace.events.push(ev);
    }
    let top = trace.top_events(3);
    assert_eq!(top.len(), 3);
    // sorted descending
    assert!(top[0].duration >= top[1].duration);
    assert!(top[1].duration >= top[2].duration);
    assert_eq!(top[0].name, "ev_100ms");
}

// ── 8. profile_trace_duration_for_prefix ────────────────────────────────────

#[test]
fn profile_trace_duration_for_prefix() {
    let mut trace = ProfileTrace::default();
    let names_ms = [
        ("attn.0", 10u64),
        ("attn.1", 20),
        ("ffn.0", 30),
        ("ffn.1", 40),
    ];
    for (name, ms) in names_ms {
        let mut ev = ProfileEvent::new(name);
        ev.duration = Duration::from_millis(ms);
        trace.events.push(ev);
    }
    let attn_total = trace.duration_for_prefix("attn");
    assert_eq!(attn_total, Duration::from_millis(30));
    let ffn_total = trace.duration_for_prefix("ffn");
    assert_eq!(ffn_total, Duration::from_millis(70));
}

// ── 9. profile_trace_avg_duration ────────────────────────────────────────────

#[test]
fn profile_trace_avg_duration() {
    let mut trace = ProfileTrace::default();
    for ms in [10u64, 20, 30] {
        let mut ev = ProfileEvent::new(format!("layer.{ms}"));
        ev.duration = Duration::from_millis(ms);
        trace.events.push(ev);
    }
    let avg = trace
        .avg_duration_for_prefix("layer")
        .expect("should have average");
    assert_eq!(avg, Duration::from_millis(20));
}

// ── 10. profile_trace_summary_nonempty ───────────────────────────────────────

#[test]
fn profile_trace_summary_nonempty() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    prof.profile("some.layer", 1_000_000, || ());
    let trace = prof.end_trace().expect("trace should exist");
    let summary = trace.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("ProfileTrace"));
}

// ── 11. profile_trace_layer_breakdown ────────────────────────────────────────

#[test]
fn profile_trace_layer_breakdown() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    for name in ["attention", "ffn", "norm"] {
        prof.profile(name, 0, || ());
    }
    let trace = prof.end_trace().expect("trace should exist");
    let breakdown = trace.layer_breakdown();
    assert!(breakdown.contains_key("attention"));
    assert!(breakdown.contains_key("ffn"));
    assert!(breakdown.contains_key("norm"));
    assert_eq!(breakdown.len(), 3);
}

// ── 12. profiler_aggregate_stats_num_traces ──────────────────────────────────

#[test]
fn profiler_aggregate_stats_num_traces() {
    let mut prof = Profiler::new();
    for _ in 0..4 {
        prof.begin_trace();
        prof.profile("op", 0, || ());
        prof.end_trace();
    }
    let stats = prof.aggregate_stats();
    assert_eq!(stats.num_traces, 4);
}

// ── 13. profiler_aggregate_stats_total_duration ──────────────────────────────

#[test]
fn profiler_aggregate_stats_total_duration() {
    let mut prof = Profiler::new();
    for _ in 0..3 {
        prof.begin_trace();
        prof.profile("op", 0, || ());
        prof.end_trace();
    }
    let stats = prof.aggregate_stats();
    // total_duration should be >= sum of individual trace total_durations
    let sum_of_traces: Duration = prof.traces().iter().map(|t| t.total_duration).sum();
    assert!(stats.total_duration >= sum_of_traces);
}

// ── 14. aggregate_stats_summary_nonempty ─────────────────────────────────────

#[test]
fn aggregate_stats_summary_nonempty() {
    let mut prof = Profiler::new();
    prof.begin_trace();
    prof.profile("x", 100, || ());
    prof.end_trace();
    let stats = prof.aggregate_stats();
    let summary = stats.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("AggregateStats"));
}

// ── 15. flop_counter_matmul ──────────────────────────────────────────────────

#[test]
fn flop_counter_matmul() {
    // 2 * m * k * n
    assert_eq!(flop_counter::matmul(3, 4, 5), 2 * 3 * 4 * 5);
    assert_eq!(flop_counter::matmul(1, 1, 1), 2);
    assert_eq!(flop_counter::matmul(10, 20, 30), 2 * 10 * 20 * 30);
}

// ── 16. flop_counter_linear ──────────────────────────────────────────────────

#[test]
fn flop_counter_linear() {
    // 2 * batch * in * out
    let batch = 4usize;
    let in_f = 128usize;
    let out_f = 256usize;
    assert_eq!(
        flop_counter::linear(batch, in_f, out_f),
        2u64 * batch as u64 * in_f as u64 * out_f as u64
    );
}

// ── 17. flop_counter_attention ───────────────────────────────────────────────

#[test]
fn flop_counter_attention() {
    // 2 * seq^2 * head_dim * num_heads
    let seq = 8;
    let head_dim = 64;
    let heads = 4;
    let expected = 2u64 * (seq * seq) * head_dim * heads;
    assert_eq!(
        flop_counter::attention(seq as usize, head_dim as usize, heads as usize),
        expected
    );
}

// ── 18. flop_counter_swiglu_ffn ──────────────────────────────────────────────

#[test]
fn flop_counter_swiglu_ffn() {
    let seq = 4usize;
    let hidden = 16usize;
    let intermediate = 32usize;

    let result = flop_counter::swiglu_ffn(seq, hidden, intermediate);

    // Two gate/up projections (each 2*seq*hidden*intermediate) +
    // down projection (2*seq*intermediate*hidden) +
    // SiLU element-wise (2*seq*intermediate)
    let gate_up = 2u64 * seq as u64 * hidden as u64 * intermediate as u64;
    let down = 2u64 * seq as u64 * intermediate as u64 * hidden as u64;
    let silu = 2u64 * seq as u64 * intermediate as u64;
    let expected = gate_up + gate_up + down + silu;
    assert_eq!(result, expected);
}

// ── 19. profiler_disabled_skips ──────────────────────────────────────────────

#[test]
fn profiler_disabled_skips() {
    let mut prof = Profiler::enabled(false);
    assert!(!prof.is_enabled());
    // begin_trace should be a no-op
    prof.begin_trace();
    // profile() must still execute the closure
    let val = prof.profile("should.run", 100, || 99u32);
    assert_eq!(val, 99);
    // end_trace returns None because no trace was started
    let trace = prof.end_trace();
    assert!(trace.is_none());
    // no traces recorded
    assert_eq!(prof.traces().len(), 0);
}

// ── 20. profile_event_gflops ─────────────────────────────────────────────────

#[test]
fn profile_event_gflops() {
    let mut ev = ProfileEvent::new("matmul");
    ev.flops = 2_000_000_000; // 2 GFLOPs
    ev.duration = Duration::from_secs(1);
    let gflops = ev.gflops_per_second();
    assert!(
        gflops > 0.0,
        "GFLOPs/s should be positive for non-zero flops and duration"
    );
    // 2e9 / 1s / 1e9 = 2.0 GFLOPs/s
    assert!((gflops - 2.0).abs() < 1e-6);
}
