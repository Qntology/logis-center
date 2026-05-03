//! Integration tests for the model hot-reload coordinator and reload log.

use std::sync::Arc;
use std::thread;

use oxibonsai_runtime::hot_reload::{HotReloadCoordinator, ModelVersion, ReloadLog};

// ─────────────────────────────────────────────────────────────────────────────
// HotReloadCoordinator tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn coordinator_new_generation_zero() {
    let coord = HotReloadCoordinator::new();
    assert_eq!(
        coord.current_generation(),
        0,
        "fresh coordinator must start at generation 0"
    );
}

#[test]
fn coordinator_record_reload_increments() {
    let coord = HotReloadCoordinator::new();

    let g1 = coord.record_reload("first load", None);
    assert_eq!(g1, 1, "first reload must produce generation 1");

    let g2 = coord.record_reload("second load", None);
    assert_eq!(g2, 2, "second reload must produce generation 2");

    assert_eq!(coord.current_generation(), 2);
}

#[test]
fn coordinator_current_version_some() {
    let coord = HotReloadCoordinator::new();
    assert!(
        coord.current_version().is_none(),
        "no version before any reload"
    );

    coord.record_reload("weights v1", Some("/models/v1.bin".to_string()));

    let ver = coord.current_version();
    assert!(ver.is_some(), "current_version must be Some after a reload");
    let ver = ver.expect("just asserted Some");
    assert_eq!(ver.generation, 1);
    assert_eq!(ver.path.as_deref(), Some("/models/v1.bin"));
}

#[test]
fn coordinator_version_history_ordered() {
    let coord = HotReloadCoordinator::new();
    coord.record_reload("v1", None);
    coord.record_reload("v2", None);
    coord.record_reload("v3", None);

    let history = coord.version_history();
    assert_eq!(history.len(), 3);
    // Most recent first.
    assert_eq!(
        history[0].generation, 3,
        "first element must be most recent"
    );
    assert_eq!(history[1].generation, 2);
    assert_eq!(history[2].generation, 1);
}

#[test]
fn coordinator_reload_count() {
    let coord = HotReloadCoordinator::new();
    assert_eq!(coord.reload_count(), 0);

    coord.record_reload("a", None);
    assert_eq!(coord.reload_count(), 1);

    coord.record_reload("b", None);
    coord.record_reload("c", None);
    assert_eq!(coord.reload_count(), 3);
}

#[test]
fn model_version_age() {
    let ver = ModelVersion::new(1, "test version");
    // Sleep a tiny bit so elapsed > 0.
    std::thread::sleep(std::time::Duration::from_millis(1));
    assert!(
        ver.age_seconds() > 0.0,
        "age_seconds must be > 0 after at least 1 ms"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ReloadLog tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reload_log_new_empty() {
    let log = ReloadLog::new(10);
    assert_eq!(log.total_events(), 0, "new log must have 0 events");
}

#[test]
fn reload_log_record() {
    let mut log = ReloadLog::new(10);
    log.record(0, 1, "first reload");
    assert_eq!(log.total_events(), 1);
    log.record(1, 2, "second reload");
    assert_eq!(log.total_events(), 2);
}

#[test]
fn reload_log_recent_events() {
    let mut log = ReloadLog::new(20);
    for i in 0..5_u64 {
        log.record(i, i + 1, format!("reload {}", i + 1));
    }

    let recent = log.recent_events(3);
    assert_eq!(recent.len(), 3, "must return last 3 events");
    // The last event recorded has new_generation = 5.
    assert_eq!(recent[2].new_generation, 5);
    // The second-to-last has new_generation = 4.
    assert_eq!(recent[1].new_generation, 4);
}

#[test]
fn reload_log_summary_nonempty() {
    let mut log = ReloadLog::new(5);
    log.record(0, 1, "initial");
    let summary = log.summary();
    assert!(!summary.is_empty(), "summary must not be empty");
}

// ─────────────────────────────────────────────────────────────────────────────
// History capacity cap test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn coordinator_max_history() {
    // Only keep the last 3 versions.
    let coord = HotReloadCoordinator::with_max_history(3);

    for i in 0..6_u64 {
        coord.record_reload(format!("load {}", i), None);
    }

    // History must be capped at max_history = 3.
    let history = coord.version_history();
    assert_eq!(
        history.len(),
        3,
        "version history must be capped at max_history"
    );
    // The most recent generation must be 6.
    assert_eq!(history[0].generation, 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent generation read (smoke test)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_generation_read() {
    let coord = Arc::new(HotReloadCoordinator::new());

    // Perform several reloads from the main thread.
    for _ in 0..5 {
        coord.record_reload("bg reload", None);
    }

    let expected_gen = coord.current_generation();

    // Spin up multiple reader threads; all should see the same generation.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let c = Arc::clone(&coord);
            thread::spawn(move || c.current_generation())
        })
        .collect();

    for handle in handles {
        let gen = handle.join().expect("reader thread must not panic");
        assert_eq!(
            gen, expected_gen,
            "all reader threads must see the same generation"
        );
    }
}
