//! Four-layer reflection contract (`ReflectionGate`), model trim policy (`TrimModePolicy`),
//! equivalence tiers (`EquivalenceTier`), and runtime rescue profitability gate (`NetGainGate`).
//! (Phase 1 – Part 1 D1, D2, D4)

use std::collections::BTreeMap;
use std::sync::OnceLock;
use premium_discount_quant::trim::Langs;

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
    pub runtime_k_default: Option<usize>,
    #[serde(default)]
    pub runtime_k_ko: Option<usize>,
    #[serde(default)]
    pub runtime_k_cjk: Option<usize>,
    #[serde(default)]
    pub runtime_k_all: Option<usize>,
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

const POLICY_JSON: &str = include_str!("model_trim_policy.json");

/// Cached canonical trim policy owned directly inside `logis-llm`.
pub fn cached_model_trim_policy() -> &'static BTreeMap<String, ModelTrimPolicyEntry> {
    static POLICY: OnceLock<BTreeMap<String, ModelTrimPolicyEntry>> = OnceLock::new();
    POLICY.get_or_init(|| serde_json::from_str(POLICY_JSON).unwrap_or_default())
}

/// Built-in default policy map (cloned convenience wrapper).
pub fn default_model_trim_policy() -> BTreeMap<String, ModelTrimPolicyEntry> {
    cached_model_trim_policy().clone()
}

/// Canonicalize slot / folder / IPC names to policy JSON keys.
/// Strict exact matching for `granite-4.0-h-350m` prevents ever blocking `granite-embedding-97m-multilingual-r2`.
pub fn canonical_policy_key(raw: &str) -> Option<&'static str> {
    let s = raw.trim().to_ascii_lowercase();
    if s == "granite-4.0-h-350m" || s == "ibm-granite/granite-4.0-h-350m" {
        Some("granite-4.0-h-350m")
    } else if s == "embeddinggemma-300m" || s == "google/embeddinggemma-300m" {
        Some("embeddinggemma-300m")
    } else if s.contains("granite-embedding-97m") || s == "granite_embed_97m_r2" {
        Some("granite-embedding-97m-multilingual-r2")
    } else if s.contains("siglip2") {
        Some("siglip2-so400m-patch16-naflex")
    } else if s.contains("qwen3-0.6b") || s.contains("qwen3_0_6b") {
        Some("Qwen3-0.6B")
    } else if s.contains("qwen3.5-2b") || s.contains("qwen3_5_2b") {
        Some("Qwen3.5-2B")
    } else if s.contains("qwen3.5-4b") || s.contains("qwen3_5_4b") || s == "model-q4km" || s == "model-q8_0" {
        Some("Qwen3.5-4B")
    } else {
        None
    }
}

/// Returns `Some((canonical_key, reason))` if the requested model is retired or forbidden.
pub fn is_retired_or_forbidden_model(model_name: &str) -> Option<(&'static str, &'static str)> {
    let key = canonical_policy_key(model_name)?;
    let entry = cached_model_trim_policy().get(key)?;
    match entry.policy {
        TrimModePolicy::Retire => Some((key, entry.reason.as_deref().unwrap_or("retired"))),
        TrimModePolicy::Forbidden => Some((key, entry.reason.as_deref().unwrap_or("forbidden"))),
        _ => None,
    }
}

/// Resolve Track M runtime active row budget `K` from model hint, vocab size, and document languages.
pub fn policy_runtime_k(model_hint: Option<&str>, vocab: usize, langs: Langs) -> usize {
    let key = model_hint
        .and_then(canonical_policy_key)
        .or(if vocab <= 160_000 && vocab > 70_000 { Some("Qwen3-0.6B") } else { Some("Qwen3.5-4B") })
        .unwrap_or("Qwen3.5-4B");
    let entry = cached_model_trim_policy().get(key);
    if langs.other || (langs.hangul && langs.kana && langs.arabic && langs.cjk) {
        entry.and_then(|e| e.runtime_k_all).unwrap_or(24_576)
    } else if langs.cjk || langs.kana {
        entry.and_then(|e| e.runtime_k_cjk).unwrap_or(12_288)
    } else if langs.hangul || langs.arabic {
        entry.and_then(|e| e.runtime_k_ko).unwrap_or(12_288)
    } else {
        entry.and_then(|e| e.runtime_k_default).unwrap_or(10_240)
    }
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

        // Exact policy key checks: granite-4.0-h-350m blocked, granite-embedding-97m NOT blocked
        assert!(is_retired_or_forbidden_model("granite-4.0-h-350m").is_some());
        assert!(is_retired_or_forbidden_model("embeddinggemma-300m").is_some());
        assert!(is_retired_or_forbidden_model("granite-embedding-97m-multilingual-r2").is_none());

        // Track M policy budgets strictly < 16384 for single-language modes
        assert_eq!(policy_runtime_k(Some("Qwen3-0.6B"), 151936, Langs::of_doc_lang("en")), 10_240);
        assert_eq!(policy_runtime_k(Some("Qwen3-0.6B"), 151936, Langs::of_doc_lang("fr")), 10_240);
        assert_eq!(policy_runtime_k(Some("Qwen3.5-4B"), 248320, Langs::of_doc_lang("ko")), 12_288);
        assert_eq!(policy_runtime_k(Some("Qwen3.5-4B"), 248320, Langs::of_doc_lang("ja")), 12_288);
    }
}
