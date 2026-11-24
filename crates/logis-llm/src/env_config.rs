//! Startup snapshot of runtime `LOGIS_*` environment variables with typo warnings
//! (Phase 1 – Part 2 D3).

use std::collections::BTreeMap;
use once_cell::sync::Lazy;

/// Registry of recognized `LOGIS_*` runtime & evaluation environment variables.
pub const KNOWN_LOGIS_ENV_VARS: &[&str] = &[
    "LOGIS_APP_DIR",
    "LOGIS_STANZA_DIR",
    "LOGIS_VOCAB_TRIM",
    "LOGIS_VOCAB_LANGS",
    "LOGIS_VOCAB_PLAN",
    "LOGIS_VOCAB_GUARD",
    "LOGIS_VOCAB_ACTIVE_K",
    "LOGIS_VOCAB_STATS_LOG",
    "LOGIS_VOCAB_MINE_MANIFEST",
    "LOGIS_GUARD_CLUSTERS",
    "LOGIS_TRIM_NET_GAIN_GATE",
    "LOGIS_USE_TRIMMED_ARTIFACT",
    "LOGIS_VERIFY_ARTIFACT",
    "LOGIS_TRIM_ANCESTOR_MAX",
    "LOGIS_EXACT_LM_HEAD",
    "LOGIS_QWEN3_DENSE",
    "LOGIS_EMBED_Q8",
    "LOGIS_Q8_EXACT",
    "LOGIS_KQUANT_EXACT",
    "LOGIS_FUSED_OPS",
    "LOGIS_ACCEL_CHECK",
    "LOGIS_ACCEL_SUBPROC",
    "LOGIS_ACCEL_ISOLATE",
    "LOGIS_REFLECTION_LOG",
    "LOGIS_REFLECTION_OUT",
    "LOGIS_REFLECT_OUT",
    "LOGIS_RUN_ID",
    "LOGIS_EMBED_TABLE",
    "LOGIS_EMBED_VARIANT",
    "LOGIS_EMBED_FORWARD",
    "LOGIS_EMBED_BATCHED",
    "LOGIS_EMBED_DISK_CACHE",
    "LOGIS_EMBED_CPU_KERNEL",
    "LOGIS_EMBED_CHUNK_OVERLAP",
    "LOGIS_SIGLIP2_TABLE",
    "LOGIS_SIGLIP2_QUANT",
    "LOGIS_SIGLIP2_QUANT_SIM",
    "LOGIS_CROP_DIET",
    "LOGIS_VISION_PRUNE",
    "LOGIS_LANG_VARIANT",
    "LOGIS_LANG_MODEL_DIR",
    "LOGIS_LANG4B_WAIT_SECS",
    "LOGIS_KEEP_SAFETENSORS",
    "LOGIS_LATIN_4B_ROUTE",
    "LOGIS_SEARCH_PURGE",
    "LOGIS_FUSION_MODES",
    "LOGIS_SPARSE_MODE",
    "LOGIS_SPARSE_HEAD_DIR",
    "LOGIS_SPARSE_WEIGHT",
    "LOGIS_SPARSE_DOC_TOPK",
    "LOGIS_SPARSE_QUERY_TOPK",
    "LOGIS_SPARSE_WINDOW_OVERLAP",
    "LOGIS_PERSIST_ONESHOT_KV",
    "LOGIS_EVAL_NOW_MS",
    "LOGIS_EVAL_BACKEND",
    "LOGIS_EVAL_GATE",
    "LOGIS_EVAL_BASELINE",
    "LOGIS_EVAL_LONG_DOC",
    "LOGIS_REQUIRE_GPU",
    "LOGIS_REQUIRE_MODELS",
    "LOGIS_TEST_MODELS",
    "LOGIS_TEST_FAST_EXIT",
    "LOGIS_TRIM_MODELS",
    "LOGIS_TRIM_IMAGES",
    "LOGIS_BENCH_WARMUP",
    "LOGIS_BENCH_REPEATS",
    "LOGIS_GATE_MAX_RESCUE_RATE",
    "LOGIS_SYNTH_PLAN",
    "LOGIS_SYNTH_OUT",
    "LOGIS_SYNTH_TEXTS",
    "LOGIS_SYNTH_MODEL",
    "LOGIS_SYNTH_CASES",
    "LOGIS_SYNTH_BBOX",
    "LOGIS_SYNTH_CPU",
    "LOGIS_SYNTH_PAGES",
    "LOGIS_SYNTH_LANG",
    "LOGIS_SYNTH_DUMP",
];

#[derive(Debug, Clone)]
pub struct RuntimeEnvSnapshot {
    pub overrides: BTreeMap<String, String>,
    pub unknown_keys: Vec<String>,
}

impl RuntimeEnvSnapshot {
    pub fn capture_from<I: IntoIterator<Item = (String, String)>>(vars: I) -> Self {
        let mut overrides = BTreeMap::new();
        let mut unknown_keys = Vec::new();
        for (k, v) in vars {
            if !k.starts_with("LOGIS_") {
                continue;
            }
            if KNOWN_LOGIS_ENV_VARS.contains(&k.as_str()) {
                overrides.insert(k, v);
            } else {
                unknown_keys.push(k.clone());
                overrides.insert(k, v);
            }
        }
        unknown_keys.sort();
        Self { overrides, unknown_keys }
    }

    pub fn capture_process() -> Self {
        let snap = Self::capture_from(std::env::vars());
        for k in &snap.unknown_keys {
            println!("[ENV-CONFIG] ⚠️ 미등록 환경 변수 감지: {k} (오타 여부를 확인하세요)");
        }
        snap
    }

    pub fn as_pairs(&self) -> Vec<(String, String)> {
        self.overrides
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

static SNAPSHOT: Lazy<RuntimeEnvSnapshot> = Lazy::new(RuntimeEnvSnapshot::capture_process);

pub fn snapshot() -> &'static RuntimeEnvSnapshot {
    &SNAPSHOT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_record_roundtrip_and_env_snapshot() {
        let snap = RuntimeEnvSnapshot::capture_from([
            ("LOGIS_VOCAB_TRIM".into(), "compact".into()),
            ("LOGIS_TYPO".into(), "1".into()),
            ("PATH".into(), "/usr/bin".into()),
        ]);
        assert_eq!(snap.overrides.get("LOGIS_VOCAB_TRIM").map(String::as_str), Some("compact"));
        assert_eq!(snap.unknown_keys, vec!["LOGIS_TYPO".to_string()]);
        assert!(!snap.overrides.contains_key("PATH"));
    }
}
