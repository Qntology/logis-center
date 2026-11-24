//! Four-layer reflection contract (`ReflectionGate`), model trim policy (`TrimModePolicy`),
//! equivalence tiers (`EquivalenceTier`), and runtime rescue profitability gate (`NetGainGate`).
//! (Phase 1 – Part 1 D1, D2, D4)

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayerStatus {
    Pass,
    Conditional,
    Fail,
    Unmeasured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReflectionGate {
    pub artifact: LayerStatus,
    pub load: LayerStatus,
    pub runtime: LayerStatus,
    pub effect: LayerStatus,
}

impl ReflectionGate {
    pub fn all_pass(&self) -> bool {
        self.artifact == LayerStatus::Pass
            && self.load == LayerStatus::Pass
            && self.runtime == LayerStatus::Pass
            && self.effect == LayerStatus::Pass
    }

    pub fn format_log(&self, target: &str) -> String {
        format!(
            "[REFLECT-GATE] target={target} artifact={:?} load={:?} runtime={:?} effect={:?}",
            self.artifact, self.load, self.runtime, self.effect
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrimModePolicy {
    StaticOnly,
    StaticPlusRuntime,
    Forbidden,
    Retire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EquivalenceTier {
    E0,
    E1,
    E2,
    E3,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelTrimPolicyEntry {
    pub policy: TrimModePolicy,
    #[serde(default)]
    pub static_k: Option<usize>,
    #[serde(default)]
    pub runtime_k_ko: Option<usize>,
    #[serde(default)]
    pub artifact_tier: Option<EquivalenceTier>,
    #[serde(default)]
    pub runtime_tier: Option<EquivalenceTier>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Evaluates whether runtime vocabulary trimming + rescue guard remains strictly profitable.
/// Requires rescue penalty <= 20% of projection savings AND total projection time <= 95% of full.
#[derive(Debug, Clone, Copy)]
pub struct NetGainGate {
    pub proj_full_ms: f64,
    pub proj_kept_ms: f64,
    pub rescue_rate: f64,
    pub rescue_host_ms: f64,
}

impl NetGainGate {
    pub fn is_profitable(&self) -> bool {
        if self.proj_full_ms <= 0.0 || self.proj_kept_ms < 0.0 || self.rescue_rate < 0.0 || self.rescue_host_ms < 0.0 {
            return false;
        }
        let saved = (self.proj_full_ms - self.proj_kept_ms).max(0.0);
        let pen = self.rescue_rate * self.rescue_host_ms;
        pen <= 0.20 * saved && (self.proj_kept_ms + pen) <= 0.95 * self.proj_full_ms
    }

    pub fn net_savings_pct(&self) -> f64 {
        if self.proj_full_ms <= 0.0 {
            return 0.0;
        }
        let eff = self.proj_kept_ms + self.rescue_rate * self.rescue_host_ms;
        (1.0 - eff / self.proj_full_ms) * 100.0
    }
}

/// Built-in default policy map matching `apps/core/src/model/model_trim_policy.json`.
pub fn default_model_trim_policy() -> BTreeMap<String, ModelTrimPolicyEntry> {
    let raw = include_str!("../../../../../apps/core/src/model/model_trim_policy.json");
    serde_json::from_str(raw).unwrap_or_default()
}

pub fn net_gain_gate_enabled() -> bool {
    match std::env::var("LOGIS_TRIM_NET_GAIN_GATE")
        .unwrap_or_else(|_| "on".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "0" | "off" | "false" | "no" => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_trim_reflection_policy_and_net_gain_gate() {
        let pol = default_model_trim_policy();
        assert_eq!(pol.len(), 7);
        assert_eq!(pol["Qwen3-0.6B"].policy, TrimModePolicy::StaticPlusRuntime);
        assert_eq!(pol["Qwen3.5-2B"].policy, TrimModePolicy::StaticPlusRuntime);
        assert_eq!(pol["Qwen3.5-4B"].policy, TrimModePolicy::StaticPlusRuntime);
        assert_eq!(pol["granite-embedding-97m-multilingual-r2"].policy, TrimModePolicy::StaticOnly);
        assert_eq!(pol["siglip2-so400m-patch16-naflex"].policy, TrimModePolicy::StaticOnly);
        assert_eq!(pol["embeddinggemma-300m"].policy, TrimModePolicy::Forbidden);
        assert_eq!(pol["granite-4.0-h-350m"].policy, TrimModePolicy::Retire);

        // Profitable case: full 10ms, kept 2ms (saved 8ms), rescue 1% * 5ms = 0.05ms <= 1.6ms
        let good = NetGainGate { proj_full_ms: 10.0, proj_kept_ms: 2.0, rescue_rate: 0.01, rescue_host_ms: 5.0 };
        assert!(good.is_profitable());

        // Penalty > 20% of savings -> fails
        let too_costly = NetGainGate { proj_full_ms: 10.0, proj_kept_ms: 5.0, rescue_rate: 0.25, rescue_host_ms: 5.0 };
        assert!(!too_costly.is_profitable());

        // Savings < 5% -> fails
        let marginal = NetGainGate { proj_full_ms: 10.0, proj_kept_ms: 9.6, rescue_rate: 0.0, rescue_host_ms: 0.0 };
        assert!(!marginal.is_profitable());

        let gate = ReflectionGate {
            artifact: LayerStatus::Pass,
            load: LayerStatus::Pass,
            runtime: LayerStatus::Pass,
            effect: LayerStatus::Pass,
        };
        assert!(gate.all_pass());
        assert!(gate.format_log("Qwen3.5-4B").contains("[REFLECT-GATE]"));
    }
}
