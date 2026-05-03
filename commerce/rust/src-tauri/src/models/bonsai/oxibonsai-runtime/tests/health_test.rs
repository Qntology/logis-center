//! Health check integration tests.
//!
//! Tests individual health checks and aggregated health reports.

use std::time::Duration;

use oxibonsai_runtime::health::{
    check_kernel_tier, check_kv_cache, check_memory_pressure, check_model_loaded,
    run_health_checks, HealthStatus,
};

// ── 1. Model loaded check → Healthy ──────────────────────────────────────

#[test]
fn model_loaded_is_healthy() {
    let check = check_model_loaded(true);
    assert!(check.status.is_healthy());
    assert_eq!(check.name, "model");
    assert!(check.details.is_some());
    let details = check.details.expect("should have details");
    assert!(details.contains("loaded"));
}

// ── 2. Model not loaded → Unhealthy ─────────────────────────────────────

#[test]
fn model_not_loaded_is_unhealthy() {
    let check = check_model_loaded(false);
    assert!(check.status.is_unhealthy());
    assert!(!check.status.is_healthy());
    assert!(!check.status.is_degraded());
}

// ── 3. Memory below 80% → Healthy ───────────────────────────────────────

#[test]
fn memory_below_80_pct_healthy() {
    // 50% usage
    let check = check_memory_pressure(5_000_000, 10_000_000);
    assert!(check.status.is_healthy());
    assert_eq!(check.name, "memory");
}

#[test]
fn memory_at_79_pct_healthy() {
    // ~79% usage
    let check = check_memory_pressure(79_000, 100_000);
    assert!(check.status.is_healthy());
}

// ── 4. Memory above 90% → Unhealthy ─────────────────────────────────────

#[test]
fn memory_above_95_pct_unhealthy() {
    // 96% usage
    let check = check_memory_pressure(96_000, 100_000);
    assert!(check.status.is_unhealthy());
}

#[test]
fn memory_between_80_95_degraded() {
    // 85% usage
    let check = check_memory_pressure(85_000, 100_000);
    assert!(check.status.is_degraded());
}

#[test]
fn memory_zero_limit_unhealthy() {
    let check = check_memory_pressure(1000, 0);
    assert!(check.status.is_unhealthy());
}

// ── 5. KV cache low utilization → Healthy ────────────────────────────────

#[test]
fn kv_cache_low_utilization_healthy() {
    let check = check_kv_cache(0.3);
    assert!(check.status.is_healthy());
    assert_eq!(check.name, "kv_cache");
}

#[test]
fn kv_cache_zero_utilization_healthy() {
    let check = check_kv_cache(0.0);
    assert!(check.status.is_healthy());
}

// ── KV cache degraded and unhealthy ──────────────────────────────────────

#[test]
fn kv_cache_high_utilization_degraded() {
    let check = check_kv_cache(0.8);
    assert!(check.status.is_degraded());
}

#[test]
fn kv_cache_critical_utilization_unhealthy() {
    let check = check_kv_cache(0.95);
    assert!(check.status.is_unhealthy());
}

// ── 6. Aggregated report JSON is valid ───────────────────────────────────

#[test]
fn health_report_json_has_required_fields() {
    let report = run_health_checks(
        true,
        1_000_000,
        10_000_000,
        0.3,
        "avx2+fma",
        Duration::from_secs(60),
    );
    let json = report.to_json();

    assert!(json.get("status").is_some(), "JSON should have status");
    assert!(
        json.get("uptime_seconds").is_some(),
        "JSON should have uptime_seconds"
    );
    assert!(json.get("version").is_some(), "JSON should have version");
    assert!(json.get("checks").is_some(), "JSON should have checks");

    let checks = json
        .get("checks")
        .expect("checks should exist")
        .as_array()
        .expect("checks should be an array");
    assert!(checks.len() >= 4, "should have at least 4 checks");

    // Verify each check has required fields
    for check in checks {
        assert!(check.get("name").is_some(), "check should have name");
        assert!(check.get("status").is_some(), "check should have status");
        assert!(
            check.get("duration_ms").is_some(),
            "check should have duration_ms"
        );
    }
}

#[test]
fn health_report_json_status_matches_overall() {
    let report = run_health_checks(
        true,
        1_000_000,
        10_000_000,
        0.3,
        "neon",
        Duration::from_secs(10),
    );
    let json = report.to_json();
    let status_str = json
        .get("status")
        .expect("should have status")
        .as_str()
        .expect("status should be string");

    // Status string should be one of the valid values
    assert!(
        status_str == "healthy" || status_str == "degraded" || status_str == "unhealthy",
        "unexpected status: {}",
        status_str
    );
}

// ── 7. Uptime and version populated ──────────────────────────────────────

#[test]
fn health_report_uptime_populated() {
    let report = run_health_checks(true, 1_000, 10_000, 0.1, "neon", Duration::from_secs(120));
    let json = report.to_json();
    let uptime = json
        .get("uptime_seconds")
        .expect("should have uptime")
        .as_f64()
        .expect("uptime should be f64");
    assert!(
        (uptime - 120.0).abs() < 0.1,
        "uptime should be ~120s, got {}",
        uptime
    );
}

#[test]
fn health_report_version_non_empty() {
    let report = run_health_checks(true, 1_000, 10_000, 0.1, "neon", Duration::from_secs(1));
    assert!(!report.version.is_empty(), "version should not be empty");
}

// ── Aggregation: unhealthy propagates ────────────────────────────────────

#[test]
fn unhealthy_check_propagates_to_overall() {
    let report = run_health_checks(
        false, // model not loaded → unhealthy
        1_000,
        10_000,
        0.1,
        "avx2+fma",
        Duration::from_secs(10),
    );
    assert!(
        report.overall.is_unhealthy(),
        "overall should be unhealthy when model not loaded"
    );
}

// ── Kernel tier checks ───────────────────────────────────────────────────

#[test]
fn kernel_tier_simd_healthy() {
    let check = check_kernel_tier("avx2+fma");
    assert!(check.status.is_healthy());
    assert_eq!(check.name, "kernel");
}

#[test]
fn kernel_tier_neon_healthy() {
    let check = check_kernel_tier("neon");
    assert!(check.status.is_healthy());
}

#[test]
fn kernel_tier_reference_on_current_arch() {
    let check = check_kernel_tier("reference");
    // On x86_64/aarch64, reference is degraded; otherwise healthy
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(check.status.is_degraded());
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    assert!(check.status.is_healthy());
}

// ── HealthStatus methods ─────────────────────────────────────────────────

#[test]
fn health_status_display() {
    assert_eq!(format!("{}", HealthStatus::Healthy), "healthy");
    assert_eq!(
        format!("{}", HealthStatus::Degraded("slow".to_string())),
        "degraded: slow"
    );
    assert_eq!(
        format!("{}", HealthStatus::Unhealthy("down".to_string())),
        "unhealthy: down"
    );
}

#[test]
fn health_status_methods_comprehensive() {
    let healthy = HealthStatus::Healthy;
    assert!(healthy.is_healthy());
    assert!(!healthy.is_degraded());
    assert!(!healthy.is_unhealthy());

    let degraded = HealthStatus::Degraded("warning".to_string());
    assert!(!degraded.is_healthy());
    assert!(degraded.is_degraded());
    assert!(!degraded.is_unhealthy());

    let unhealthy = HealthStatus::Unhealthy("critical".to_string());
    assert!(!unhealthy.is_healthy());
    assert!(!unhealthy.is_degraded());
    assert!(unhealthy.is_unhealthy());
}

// ── Health check duration is non-negative ────────────────────────────────

#[test]
fn health_check_duration_non_negative() {
    let check = check_model_loaded(true);
    // Duration should be non-negative (it's a duration, so it always is)
    assert!(
        check.duration.as_nanos() < 1_000_000_000,
        "check should be fast"
    );
}
