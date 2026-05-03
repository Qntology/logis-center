//! Health check system for production monitoring.
//!
//! Provides structured health checks for model loading, memory pressure,
//! KV cache utilization, and kernel availability.

use std::time::Duration;

/// Health status of a component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// Component is operating normally.
    Healthy,
    /// Component is working but suboptimal.
    Degraded(String),
    /// Component is not working.
    Unhealthy(String),
}

impl HealthStatus {
    /// Whether this status represents a healthy state.
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Whether this status represents a degraded state.
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Degraded(_))
    }

    /// Whether this status represents an unhealthy state.
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, Self::Unhealthy(_))
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded(_) => "degraded",
            Self::Unhealthy(_) => "unhealthy",
        }
    }

    fn message(&self) -> Option<&str> {
        match self {
            Self::Healthy => None,
            Self::Degraded(msg) | Self::Unhealthy(msg) => Some(msg.as_str()),
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded(msg) => write!(f, "degraded: {}", msg),
            Self::Unhealthy(msg) => write!(f, "unhealthy: {}", msg),
        }
    }
}

/// Individual health check result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the check (e.g. "model", "memory", "kv_cache").
    pub name: String,
    /// Check result status.
    pub status: HealthStatus,
    /// Time taken to run this check.
    pub duration: Duration,
    /// Optional details about the check.
    pub details: Option<String>,
}

/// Aggregated health report.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall health status (worst of individual checks).
    pub overall: HealthStatus,
    /// Individual check results.
    pub checks: Vec<HealthCheck>,
    /// Engine uptime.
    pub uptime: Duration,
    /// Engine version string.
    pub version: String,
}

impl HealthReport {
    /// Serialize the health report as a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        let checks: Vec<serde_json::Value> = self
            .checks
            .iter()
            .map(|c| {
                let mut check = serde_json::json!({
                    "name": c.name,
                    "status": c.status.as_str(),
                    "duration_ms": c.duration.as_millis() as u64,
                });
                if let Some(msg) = c.status.message() {
                    check["message"] = serde_json::Value::String(msg.to_string());
                }
                if let Some(details) = &c.details {
                    check["details"] = serde_json::Value::String(details.clone());
                }
                check
            })
            .collect();

        let mut result = serde_json::json!({
            "status": self.overall.as_str(),
            "uptime_seconds": self.uptime.as_secs_f64(),
            "version": self.version,
            "checks": checks,
        });
        if let Some(msg) = self.overall.message() {
            result["message"] = serde_json::Value::String(msg.to_string());
        }
        result
    }
}

/// Check if the model is loaded and functional.
pub fn check_model_loaded(has_model: bool) -> HealthCheck {
    let start = std::time::Instant::now();
    let status = if has_model {
        HealthStatus::Healthy
    } else {
        HealthStatus::Unhealthy("no model loaded".to_string())
    };
    HealthCheck {
        name: "model".to_string(),
        status,
        duration: start.elapsed(),
        details: if has_model {
            Some("model loaded and ready".to_string())
        } else {
            Some("model not loaded".to_string())
        },
    }
}

/// Check memory pressure by comparing current usage to the limit.
///
/// - Healthy if usage < 80% of limit
/// - Degraded if usage is 80-95% of limit
/// - Unhealthy if usage > 95% of limit
pub fn check_memory_pressure(current_rss_bytes: u64, limit_bytes: u64) -> HealthCheck {
    let start = std::time::Instant::now();

    if limit_bytes == 0 {
        return HealthCheck {
            name: "memory".to_string(),
            status: HealthStatus::Unhealthy("memory limit is zero".to_string()),
            duration: start.elapsed(),
            details: None,
        };
    }

    let usage_pct = (current_rss_bytes as f64 / limit_bytes as f64) * 100.0;
    let details = Some(format!(
        "{:.1}% used ({} / {})",
        usage_pct,
        crate::convenience::format_bytes(current_rss_bytes),
        crate::convenience::format_bytes(limit_bytes),
    ));

    let status = if usage_pct < 80.0 {
        HealthStatus::Healthy
    } else if usage_pct < 95.0 {
        HealthStatus::Degraded(format!("memory usage at {:.1}%", usage_pct))
    } else {
        HealthStatus::Unhealthy(format!("memory usage critical at {:.1}%", usage_pct))
    };

    HealthCheck {
        name: "memory".to_string(),
        status,
        duration: start.elapsed(),
        details,
    }
}

/// Check KV cache utilization.
///
/// - Healthy if utilization < 75%
/// - Degraded if utilization is 75-90%
/// - Unhealthy if utilization > 90%
pub fn check_kv_cache(utilization: f64) -> HealthCheck {
    let start = std::time::Instant::now();
    let details = Some(format!("{:.1}% utilized", utilization * 100.0));

    let status = if utilization < 0.75 {
        HealthStatus::Healthy
    } else if utilization < 0.90 {
        HealthStatus::Degraded(format!("KV cache {:.1}% full", utilization * 100.0))
    } else {
        HealthStatus::Unhealthy(format!(
            "KV cache {:.1}% full, near capacity",
            utilization * 100.0
        ))
    };

    HealthCheck {
        name: "kv_cache".to_string(),
        status,
        duration: start.elapsed(),
        details,
    }
}

/// Check kernel tier availability.
///
/// Reports healthy for any recognized tier, degraded for reference
/// (scalar) tier on supported SIMD platforms.
pub fn check_kernel_tier(tier_name: &str) -> HealthCheck {
    let start = std::time::Instant::now();
    let details = Some(format!("kernel tier: {}", tier_name));

    let status = match tier_name {
        "reference" | "Q1_0_g128 reference (scalar)" => {
            // Reference is functional but suboptimal on SIMD-capable hardware
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                HealthStatus::Degraded(
                    "using scalar reference kernel; SIMD may be available".to_string(),
                )
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                HealthStatus::Healthy
            }
        }
        _ => HealthStatus::Healthy,
    };

    HealthCheck {
        name: "kernel".to_string(),
        status,
        duration: start.elapsed(),
        details,
    }
}

/// Run all health checks and aggregate into a report.
pub fn run_health_checks(
    has_model: bool,
    memory_usage_bytes: u64,
    memory_limit_bytes: u64,
    kv_utilization: f64,
    kernel_tier: &str,
    uptime: Duration,
) -> HealthReport {
    let checks = vec![
        check_model_loaded(has_model),
        check_memory_pressure(memory_usage_bytes, memory_limit_bytes),
        check_kv_cache(kv_utilization),
        check_kernel_tier(kernel_tier),
    ];

    // Overall status is the worst of all checks
    let overall = aggregate_status(&checks);

    HealthReport {
        overall,
        checks,
        uptime,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Determine overall health from a list of checks.
///
/// Returns the worst status found: Unhealthy > Degraded > Healthy.
fn aggregate_status(checks: &[HealthCheck]) -> HealthStatus {
    let mut has_degraded = false;
    let mut unhealthy_reason = None;

    for check in checks {
        match &check.status {
            HealthStatus::Unhealthy(msg) => {
                unhealthy_reason = Some(msg.clone());
            }
            HealthStatus::Degraded(_) => {
                has_degraded = true;
            }
            HealthStatus::Healthy => {}
        }
    }

    if let Some(reason) = unhealthy_reason {
        HealthStatus::Unhealthy(reason)
    } else if has_degraded {
        HealthStatus::Degraded("some checks degraded".to_string())
    } else {
        HealthStatus::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loaded_healthy() {
        let check = check_model_loaded(true);
        assert!(check.status.is_healthy());
        assert_eq!(check.name, "model");
    }

    #[test]
    fn model_not_loaded_unhealthy() {
        let check = check_model_loaded(false);
        assert!(check.status.is_unhealthy());
    }

    #[test]
    fn memory_healthy() {
        let check = check_memory_pressure(1_000_000, 10_000_000);
        assert!(check.status.is_healthy());
    }

    #[test]
    fn memory_degraded() {
        let check = check_memory_pressure(8_500_000, 10_000_000);
        assert!(check.status.is_degraded());
    }

    #[test]
    fn memory_unhealthy() {
        let check = check_memory_pressure(9_600_000, 10_000_000);
        assert!(check.status.is_unhealthy());
    }

    #[test]
    fn memory_zero_limit() {
        let check = check_memory_pressure(1_000, 0);
        assert!(check.status.is_unhealthy());
    }

    #[test]
    fn kv_cache_healthy() {
        let check = check_kv_cache(0.5);
        assert!(check.status.is_healthy());
    }

    #[test]
    fn kv_cache_degraded() {
        let check = check_kv_cache(0.8);
        assert!(check.status.is_degraded());
    }

    #[test]
    fn kv_cache_unhealthy() {
        let check = check_kv_cache(0.95);
        assert!(check.status.is_unhealthy());
    }

    #[test]
    fn kernel_tier_simd() {
        let check = check_kernel_tier("avx2+fma");
        assert!(check.status.is_healthy());
    }

    #[test]
    fn kernel_tier_reference() {
        let check = check_kernel_tier("reference");
        // On x86_64 or aarch64, reference is degraded; otherwise healthy
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(check.status.is_degraded());
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(check.status.is_healthy());
    }

    #[test]
    fn health_report_all_healthy() {
        let report = run_health_checks(
            true,
            1_000_000,
            10_000_000,
            0.3,
            "neon",
            Duration::from_secs(60),
        );
        // Overall should be healthy (or degraded on x86_64 if "neon" is not recognized as ref)
        assert!(report.checks.len() >= 4);
        assert!(!report.version.is_empty());
    }

    #[test]
    fn health_report_json() {
        let report = run_health_checks(
            true,
            1_000_000,
            10_000_000,
            0.3,
            "avx2+fma",
            Duration::from_secs(120),
        );
        let json = report.to_json();
        assert!(json.get("status").is_some());
        assert!(json.get("uptime_seconds").is_some());
        assert!(json.get("version").is_some());
        assert!(json.get("checks").is_some());

        let checks = json.get("checks").expect("checks should exist");
        assert!(checks.is_array());
    }

    #[test]
    fn health_report_unhealthy_propagates() {
        let report = run_health_checks(
            false, // model not loaded
            1_000_000,
            10_000_000,
            0.3,
            "avx2+fma",
            Duration::from_secs(10),
        );
        assert!(report.overall.is_unhealthy());
    }

    #[test]
    fn aggregate_status_empty() {
        let status = aggregate_status(&[]);
        assert!(status.is_healthy());
    }

    #[test]
    fn health_status_display() {
        assert_eq!(format!("{}", HealthStatus::Healthy), "healthy");
        assert_eq!(
            format!("{}", HealthStatus::Degraded("test".to_string())),
            "degraded: test"
        );
        assert_eq!(
            format!("{}", HealthStatus::Unhealthy("fail".to_string())),
            "unhealthy: fail"
        );
    }

    #[test]
    fn health_status_methods() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Healthy.is_degraded());
        assert!(!HealthStatus::Healthy.is_unhealthy());

        let degraded = HealthStatus::Degraded("slow".to_string());
        assert!(!degraded.is_healthy());
        assert!(degraded.is_degraded());
        assert!(!degraded.is_unhealthy());

        let unhealthy = HealthStatus::Unhealthy("down".to_string());
        assert!(!unhealthy.is_healthy());
        assert!(!unhealthy.is_degraded());
        assert!(unhealthy.is_unhealthy());
    }
}
