//! Four-layer reflection audit ledger (`ReflectionRecord`, Phase 1 – Part 2 D1).
//!
//! Records every model load and inference execution across the four reflection layers
//! (artifact, load, runtime, effect) to `{app_dir}/tmp/logs/reflection.jsonl` (or
//! `LOGIS_REFLECTION_OUT`), flushing with `File::sync_all()` so fast ROCm process exit
//! (`TerminateProcess`) never drops audit entries.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

const MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;
const MAX_ROTATED_FILES: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteBlockReason {
    CpuMode,
    UnsupportedLang,
    LatinPolicy,
    ScriptCoverageMismatch,
    LowRam,
    LowVram,
    LowDecodeResidentRatio,
    SdsPassGate,
    TokenFitFailed,
    StepDisabled,
    NotReady,
    Other(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactReflection {
    pub slot: String,
    pub bytes: u64,
    pub sha256_16: String,
    pub recipe: String,
    pub vocab_v: u32,
    pub artifact_k: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space_tag: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadReflection {
    pub load_ms: f32,
    pub vram_before_mb: f32,
    pub vram_after_mb: f32,
    pub table_mode: String,
    #[serde(default)]
    pub resident_table_mb: f32,
    #[serde(default)]
    pub mapped_table_mb: f32,
    #[serde(default)]
    pub env_overrides: Vec<(String, String)>,
    #[serde(default)]
    pub accel_fallbacks: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_block_reason: Option<RouteBlockReason>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecReflection {
    pub step: String,
    /// `[doc_lang, ui_lang, resolved_scope_lang]`
    pub langs: [String; 3],
    pub scope_label: String,
    /// `[vocab_v, active_k]`
    pub vocab_k: [u32; 2],
    pub head_path: String,
    pub guard_kind: String,
    /// `[prompt_tokens, gen_tokens]`
    pub tokens: [u32; 2],
    /// `[prefill_ms, decode_ms, rescue_ms]`
    pub timing_ms: [f32; 3],
    /// `[fast_guard, penalty_fb, unguarded, rescued, overrode]`
    pub steps: [u32; 5],
    #[serde(default)]
    pub sampling: bool,
    #[serde(default)]
    pub trim_mode: String,
    #[serde(default)]
    pub clusters_checked: u32,
    #[serde(default)]
    pub rows_scored: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_vocab_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_query_nnz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_latency_ms: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_mode_applied: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionRecord {
    pub ts_ms: i64,
    pub run_id: String,
    pub task_id: String,
    pub git_rev: String,
    pub build_profile: String,
    pub features: Vec<String>,
    pub backend: String,
    pub device_name: String,
    pub artifact: ArtifactReflection,
    pub load: Option<LoadReflection>,
    pub exec: Option<ExecReflection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionLogMode {
    On,
    Summary,
    Off,
}

impl ReflectionLogMode {
    pub fn from_env() -> Self {
        match std::env::var("LOGIS_REFLECTION_LOG")
            .unwrap_or_else(|_| "on".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" | "none" => Self::Off,
            "summary" => Self::Summary,
            _ => Self::On,
        }
    }
}

static WRITE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

pub fn default_reflection_path() -> PathBuf {
    if let Ok(p) = std::env::var("LOGIS_REFLECTION_OUT") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(p) = std::env::var("LOGIS_REFLECT_OUT") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    crate::paths::get_app_dir()
        .join("tmp")
        .join("logs")
        .join("reflection.jsonl")
}

pub fn build_profile_label() -> String {
    let base = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let panic_mode = if cfg!(panic = "abort") {
        "abort"
    } else {
        "unwind"
    };
    format!("{base}-{panic_mode}")
}

pub fn current_ts_ms() -> i64 {
    if let Ok(v) = std::env::var("LOGIS_EVAL_NOW_MS") {
        if let Ok(ms) = v.trim().parse::<i64>() {
            return ms;
        }
    }
    chrono::Utc::now().timestamp_millis()
}

pub fn current_run_id() -> String {
    std::env::var("LOGIS_RUN_ID").unwrap_or_else(|_| format!("pid-{}", std::process::id()))
}

fn rotate_if_needed(path: &Path) {
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if len < MAX_LOG_BYTES {
        return;
    }
    for idx in (1..=MAX_ROTATED_FILES).rev() {
        let dst = path.with_extension(format!("jsonl.{idx}"));
        if idx == MAX_ROTATED_FILES {
            let _ = std::fs::remove_file(&dst);
        } else {
            let src = path.with_extension(format!("jsonl.{idx}"));
            let next = path.with_extension(format!("jsonl.{}", idx + 1));
            let _ = std::fs::rename(&src, &next);
        }
    }
    let first = path.with_extension("jsonl.1");
    let _ = std::fs::rename(path, first);
}

/// Append a `ReflectionRecord` as one JSONL line and `sync_all()` to disk.
pub fn record(rec: &ReflectionRecord) {
    let mode = ReflectionLogMode::from_env();
    if mode == ReflectionLogMode::Off {
        return;
    }
    if mode == ReflectionLogMode::Summary && rec.load.is_none() && rec.exec.is_some() {
        // Summary mode skips per-step exec lines unless explicitly outputting to LOGIS_REFLECTION_OUT
        if std::env::var("LOGIS_REFLECTION_OUT").is_err()
            && std::env::var("LOGIS_REFLECT_OUT").is_err()
        {
            return;
        }
    }
    let path = default_reflection_path();
    let _ = record_to_path(&path, rec);
}

pub fn record_to_path(path: &Path, rec: &ReflectionRecord) -> std::io::Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    rotate_if_needed(path);
    let line = serde_json::to_string(rec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file: File = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_record_roundtrip_and_sync() {
        let rec = ReflectionRecord {
            ts_ms: 1795420800000,
            run_id: "unit-01".into(),
            task_id: "t-1".into(),
            git_rev: "2bc4ebc7".into(),
            build_profile: build_profile_label(),
            features: vec!["rocm".into(), "flash-attn".into()],
            backend: "rocm".into(),
            device_name: "AMD Radeon RX 6600".into(),
            artifact: ArtifactReflection {
                slot: "qwen3_5_4b".into(),
                bytes: 2354733056,
                sha256_16: "bb38bd40c9fde0ae".into(),
                recipe: "Q4_K_M".into(),
                vocab_v: 248320,
                artifact_k: 65536,
                plan_sha256: Some("7e2d51b8".into()),
                space_tag: None,
            },
            load: Some(LoadReflection {
                load_ms: 820.0,
                vram_before_mb: 7800.0,
                vram_after_mb: 5500.0,
                table_mode: "mapped".into(),
                resident_table_mb: 33.4,
                mapped_table_mb: 178.3,
                env_overrides: vec![],
                accel_fallbacks: vec![],
                route_block_reason: None,
            }),
            exec: Some(ExecReflection {
                step: "commerce.detail_synthesis".into(),
                langs: ["ko".into(), "ko".into(), "ko+en".into()],
                scope_label: "compact:hangul".into(),
                vocab_k: [65536, 12288],
                head_path: "quant_mode_a".into(),
                guard_kind: "clustered_64".into(),
                tokens: [256, 64],
                timing_ms: [42.0, 1280.0, 1.4],
                steps: [62, 1, 1, 2, 0],
                sampling: false,
                trim_mode: "compact".into(),
                clusters_checked: 2,
                rows_scored: 320,
                sparse_mode: None,
                sparse_vocab_k: None,
                sparse_query_nnz: None,
                sparse_latency_ms: None,
                fusion_mode_applied: None,
            }),
        };
        let p = std::env::temp_dir().join(format!("logis_reflect_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&p);
        record_to_path(&p, &rec).unwrap();
        let content = std::fs::read_to_string(&p).unwrap();
        let parsed: ReflectionRecord = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed.artifact.artifact_k, 65536);
        assert_eq!(parsed.exec.as_ref().unwrap().vocab_k, [65536, 12288]);
        let _ = std::fs::remove_file(&p);
    }
}
