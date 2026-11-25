//! Four-layer reflection audit ledger (`ReflectionRecord`, Phase 1 – Part 2 D1).
//!
//! Records every model load and inference execution across the four reflection layers
//! (artifact, load, runtime, effect) to `{app_dir}/tmp/logs/reflection.jsonl` (or
//! `LOGIS_REFLECTION_OUT`), using a persistent 64 KiB `BufWriter<File>` in `Batch` mode
//! with RAII (`ReflectionFlushGuard`) + `OsFlush` for fast-exit / eval paths.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;
const BUF_CAPACITY: usize = 64 * 1024;
const BATCH_FLUSH_RECORDS: u32 = 16;
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

static WRITE_LOCK: Lazy<Mutex<Option<ReflectionWriter>>> = Lazy::new(|| Mutex::new(None));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionSyncPolicy {
    /// Default runtime mode: 64 KiB `BufWriter`, flushes on `load` / 16 records / RAII `ReflectionFlushGuard`.
    Batch,
    /// Benchmark / fast-exit mode: flushes to OS page cache on every record (~3 us, survives `TerminateProcess`).
    OsFlush,
    /// Crash-trace debug (`LOGIS_REFLECTION_SYNC=always|sync|fsync|crash`): `File::sync_all()` on every record.
    SyncAll,
}

/// Cached reflection sync policy (avoids repeated `env::var` syscalls on hot path).
pub fn cached_sync_policy() -> ReflectionSyncPolicy {
    static POLICY: OnceLock<ReflectionSyncPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        match std::env::var("LOGIS_REFLECTION_SYNC")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "always" | "sync" | "fsync" | "crash" => ReflectionSyncPolicy::SyncAll,
            "os" | "line" => ReflectionSyncPolicy::OsFlush,
            _ => {
                if std::env::var_os("LOGIS_REFLECTION_OUT").is_some()
                    || std::env::var_os("LOGIS_REFLECT_OUT").is_some()
                    || std::env::var_os("LOGIS_TEST_FAST_EXIT").is_some()
                {
                    ReflectionSyncPolicy::OsFlush
                } else {
                    ReflectionSyncPolicy::Batch
                }
            }
        }
    })
}

/// Persistent buffered writer with handle-releasing forced rotation on Windows.
pub struct ReflectionWriter {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    written_bytes: u64,
    unflushed_records: u32,
    max_log_bytes: u64,
}

impl ReflectionWriter {
    pub fn open(path: PathBuf, max_log_bytes: u64) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path,
            writer: Some(BufWriter::with_capacity(BUF_CAPACITY, file)),
            written_bytes: len,
            unflushed_records: 0,
            max_log_bytes: max_log_bytes.max(1024),
        })
    }

    pub fn append_line(
        &mut self,
        bytes: &[u8],
        is_load: bool,
        policy: ReflectionSyncPolicy,
    ) -> std::io::Result<()> {
        let next_total = self.written_bytes.saturating_add(bytes.len() as u64);
        let prev_bucket = self.written_bytes / self.max_log_bytes;
        let next_bucket = next_total / self.max_log_bytes;
        if self.written_bytes > 0 && next_bucket > prev_bucket {
            if let Some(mut old_w) = self.writer.take() {
                old_w.flush()?;
                let _ = old_w.get_ref().sync_all();
            }
            force_rotate_files(&self.path);
            let reopened = OpenOptions::new().create(true).append(true).open(&self.path)?;
            self.written_bytes = reopened.metadata().map(|m| m.len()).unwrap_or(0);
            self.writer = Some(BufWriter::with_capacity(BUF_CAPACITY, reopened));
            self.unflushed_records = 0;
        }
        if self.writer.is_none() {
            let reopened = OpenOptions::new().create(true).append(true).open(&self.path)?;
            self.written_bytes = reopened.metadata().map(|m| m.len()).unwrap_or(0);
            self.writer = Some(BufWriter::with_capacity(BUF_CAPACITY, reopened));
        }
        let w = self.writer.as_mut().expect("writer initialized");
        w.write_all(bytes)?;
        self.written_bytes = self.written_bytes.saturating_add(bytes.len() as u64);
        self.unflushed_records += 1;
        match policy {
            ReflectionSyncPolicy::SyncAll => {
                w.flush()?;
                w.get_ref().sync_all()?;
                self.unflushed_records = 0;
            }
            ReflectionSyncPolicy::OsFlush => {
                w.flush()?;
                self.unflushed_records = 0;
            }
            ReflectionSyncPolicy::Batch => {
                if is_load || self.unflushed_records >= BATCH_FLUSH_RECORDS {
                    w.flush()?;
                    self.unflushed_records = 0;
                }
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
            self.unflushed_records = 0;
        }
        Ok(())
    }

    pub fn flush_and_sync(&mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
            w.get_ref().sync_all()?;
            self.unflushed_records = 0;
        }
        Ok(())
    }
}

/// RAII scope guard flushing buffered reflection records on task/command exit or early `?` error.
pub struct ReflectionFlushGuard;

impl Drop for ReflectionFlushGuard {
    fn drop(&mut self) {
        flush();
    }
}

/// Flush any buffered records in the active global writer to OS page cache.
pub fn flush() {
    if let Ok(mut g) = WRITE_LOCK.lock() {
        if let Some(w) = g.as_mut() {
            let _ = w.flush();
        }
    }
}

/// Flush and `sync_all()` any buffered records in the active global writer to disk.
pub fn flush_and_sync() {
    if let Ok(mut g) = WRITE_LOCK.lock() {
        if let Some(w) = g.as_mut() {
            let _ = w.flush_and_sync();
        }
    }
}

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

fn force_rotate_files(path: &Path) {
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

/// Append a `ReflectionRecord` as one JSONL line using `cached_sync_policy()`.
pub fn record(rec: &ReflectionRecord) {
    let mode = ReflectionLogMode::from_env();
    if mode == ReflectionLogMode::Off {
        return;
    }
    if mode == ReflectionLogMode::Summary && rec.load.is_none() && rec.exec.is_some() {
        if std::env::var("LOGIS_REFLECTION_OUT").is_err()
            && std::env::var("LOGIS_REFLECT_OUT").is_err()
        {
            return;
        }
    }
    let path = default_reflection_path();
    let policy = cached_sync_policy();
    let Ok(mut line) = serde_json::to_string(rec) else { return };
    line.push('\n');
    let mut guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let need_reopen = guard.as_ref().is_none_or(|w| w.path != path);
    if need_reopen {
        if let Some(prev) = guard.as_mut() {
            let _ = prev.flush();
        }
        if let Ok(w) = ReflectionWriter::open(path, DEFAULT_MAX_LOG_BYTES) {
            *guard = Some(w);
        } else {
            return;
        }
    }
    if let Some(w) = guard.as_mut() {
        let _ = w.append_line(line.as_bytes(), rec.load.is_some(), policy);
    }
}

pub fn record_to_path(path: &Path, rec: &ReflectionRecord) -> std::io::Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = ReflectionWriter::open(path.to_path_buf(), DEFAULT_MAX_LOG_BYTES)?;
    let mut line = serde_json::to_string(rec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    w.append_line(line.as_bytes(), true, ReflectionSyncPolicy::SyncAll)?;
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
                vram_before_mb: 240.0,
                vram_after_mb: 2401.0,
                table_mode: "mapped".into(),
                resident_table_mb: 33.4,
                mapped_table_mb: 178.3,
                env_overrides: vec![],
                accel_fallbacks: vec![],
                route_block_reason: None,
            }),
            exec: Some(ExecReflection {
                step: "commerce.detail_synthesis".into(),
                langs: ["ko".into(), "ko".into(), "hangul".into()],
                scope_label: "compact:hangul".into(),
                vocab_k: [65536, 12288],
                head_path: "quant_mode_a".into(),
                guard_kind: "off".into(),
                tokens: [256, 64],
                timing_ms: [42.0, 1280.0, 0.0],
                steps: [64, 0, 0, 0, 0],
                sampling: false,
                trim_mode: "compact".into(),
                clusters_checked: 0,
                rows_scored: 0,
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
        assert!(parsed.load.unwrap().vram_after_mb > 240.0);
        assert_eq!(parsed.exec.as_ref().unwrap().vocab_k, [65536, 12288]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn reflection_writer_batches_and_rotates() {
        let dir = std::env::temp_dir().join(format!("logis_reflect_rot_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("reflection.jsonl");
        let mut w = ReflectionWriter::open(p.clone(), 1024).unwrap();
        let line = format!("{{\"msg\":\"{}\"}}\n", "x".repeat(280));
        for _ in 0..8 {
            w.append_line(line.as_bytes(), false, ReflectionSyncPolicy::Batch).unwrap();
        }
        w.flush_and_sync().unwrap();
        assert!(p.exists());
        assert!(p.with_extension("jsonl.1").exists(), "forced rotation must create .jsonl.1");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
