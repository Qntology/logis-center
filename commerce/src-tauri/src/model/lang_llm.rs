use once_cell::sync::{Lazy, OnceCell};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::Emitter;

pub const LANG_LLM_OWNER: &str = "alphaedge-ai";
pub const LANG_LLM_FILES: [&str; 9] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors",
];
pub const LANG_LLM_REQUIRED: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];
pub const LANG_LLM_MIN_WEIGHT_BYTES: u64 = 400_000_000;
pub const LANG_LLM_RESIDENT_RATIO: f64 = 0.30;

const WEIGHT_EXTS: [&str; 5] = ["safetensors", "bin", "pt", "pth", "gguf"];
const FAIL_RETRY_SECS: i64 = 600;
const CHUNK_TIMEOUT_SECS: u64 = 60;
const FILE_ATTEMPTS: u32 = 4;
const PROGRESS_STEP: u64 = 5;
const PROGRESS_MIN_BYTES: u64 = 64 * 1024 * 1024;
const CONVERT_LABEL: &str = "Q4_K_M 변환";
const WAIT_TICK_MS: u64 = 1000;
const WAIT_HEARTBEAT_SECS: u64 = 30;
const WAIT_RETRIES: u32 = 3;

const KNOWN_CODES: &[&str] = &[
    "afr", "asm", "ast", "aze", "bak", "bel", "ben", "bos", "bul", "cat", "ceb", "ces",
    "cym", "dan", "deu", "ell", "eng", "est", "eus", "fas", "fin", "fra", "gle", "glg",
    "guj", "hat", "heb", "hin", "hrv", "hun", "hye", "ind", "isl", "ita", "jav", "jpn",
    "kan", "kat", "kaz", "khm", "kor", "lao", "lit", "ltz", "lvs", "mal", "mar", "min",
    "mkd", "mlt", "mya", "nep", "nld", "nno", "nob", "oci", "pan", "pol", "por", "ron",
    "rus", "scn", "sin", "slk", "slv", "snd", "spa", "srp", "sun", "swe", "tam", "tat",
    "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "vie", "war", "ydd", "zho",
];

pub type LangLlmLoader = fn(
    &Path,
    &candle_core::Device,
) -> anyhow::Result<crate::models::qwen3_5::generate::Qwen3_5GenerateModel>;

static LOADER: OnceCell<LangLlmLoader> = OnceCell::new();

pub fn register_loader(f: LangLlmLoader) -> bool {
    LOADER.set(f).is_ok()
}

pub fn loader() -> Option<LangLlmLoader> {
    LOADER.get().copied()
}

const ISO1_TO_ALPHAEDGE: &[(&str, &str)] = &[
    ("ko", "kor"), ("en", "eng"), ("ja", "jpn"), ("zh", "zho"), ("fr", "fra"), ("de", "deu"),
    ("es", "spa"), ("it", "ita"), ("pt", "por"), ("nl", "nld"), ("ru", "rus"), ("uk", "ukr"),
    ("be", "bel"), ("bg", "bul"), ("sr", "srp"), ("mk", "mkd"), ("kk", "kaz"), ("th", "tha"),
    ("el", "ell"), ("ta", "tam"), ("te", "tel"), ("hi", "hin"), ("mr", "mar"), ("ne", "nep"),
    ("bn", "ben"), ("fa", "fas"), ("ur", "urd"), ("vi", "vie"), ("id", "ind"), ("tr", "tur"),
    ("pl", "pol"), ("cs", "ces"), ("sk", "slk"), ("sl", "slv"), ("hr", "hrv"), ("bs", "bos"),
    ("ro", "ron"), ("hu", "hun"), ("fi", "fin"), ("et", "est"), ("lv", "lvs"), ("lt", "lit"),
    ("sv", "swe"), ("da", "dan"), ("no", "nob"), ("nb", "nob"), ("nn", "nno"), ("is", "isl"),
    ("ga", "gle"), ("gl", "glg"), ("eu", "eus"), ("ca", "cat"), ("cy", "cym"), ("mt", "mlt"),
    ("lb", "ltz"), ("oc", "oci"), ("ka", "kat"), ("hy", "hye"), ("az", "aze"), ("ba", "bak"),
    ("tt", "tat"), ("tg", "tgk"), ("tl", "tgl"), ("fil", "tgl"), ("jv", "jav"), ("su", "sun"),
    ("km", "khm"), ("lo", "lao"), ("my", "mya"), ("si", "sin"), ("pa", "pan"), ("gu", "guj"),
    ("kn", "kan"), ("ml", "mal"), ("sd", "snd"), ("he", "heb"), ("iw", "heb"), ("ht", "hat"),
    ("yi", "ydd"), ("af", "afr"), ("as", "asm"),
];

pub fn alphaedge_code(doc_lang: &str) -> Option<&'static str> {
    let lower = doc_lang.trim().to_lowercase();
    if lower.is_empty() {
        return None;
    }
    if let Some(c) = KNOWN_CODES.iter().find(|c| **c == lower.as_str()) {
        return Some(*c);
    }
    let base = lower.split(|c: char| c == '-' || c == '_').next().unwrap_or("");
    if let Some((_, code)) = ISO1_TO_ALPHAEDGE.iter().find(|(iso1, _)| *iso1 == base) {
        return Some(*code);
    }
    let norm = crate::utils::bias_schema::lang_code_of(&lower);
    let norm_base = norm.split(|c: char| c == '-' || c == '_').next().unwrap_or("");
    ISO1_TO_ALPHAEDGE
        .iter()
        .find(|(iso1, _)| *iso1 == norm_base)
        .map(|(_, code)| *code)
}

pub fn needs_lang_engine(doc_lang: &str) -> bool {
    alphaedge_code(doc_lang).is_some()
        && !crate::nl_convert::is_latin_dominant(&crate::nl_convert::native_script_sample(doc_lang, "", ""))
}

pub fn repo_name(code: &str) -> String {
    format!("Qwen3.5-4B-{}-16384", code)
}

pub fn repo_id(code: &str) -> String {
    format!("{}/{}", LANG_LLM_OWNER, repo_name(code))
}

pub fn model_dir(code: &str) -> PathBuf {
    crate::utils::get_app_dir().join("models").join(repo_name(code))
}

fn file_url(code: &str, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo_id(code), file)
}

fn weight_bytes(dir: &Path) -> u64 {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    rd.flatten()
        .filter_map(|e| {
            let p = e.path();
            if !p.is_file() {
                return None;
            }
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !WEIGHT_EXTS.contains(&ext.as_str()) {
                return None;
            }
            e.metadata().ok().map(|m| m.len())
        })
        .sum()
}

pub fn files_ready(code: &str) -> bool {
    let dir = model_dir(code);
    if !dir.is_dir() {
        return false;
    }
    for f in LANG_LLM_REQUIRED.iter() {
        match std::fs::metadata(dir.join(f)) {
            Ok(m) if m.len() > 0 => {}
            _ => return false,
        }
    }
    weight_bytes(&dir) >= LANG_LLM_MIN_WEIGHT_BYTES
}

pub fn is_ready(code: &str) -> bool {
    files_ready(code) && crate::model::lang_gguf::runtime_ready(&model_dir(code))
}

pub fn resident_estimate_mb(code: &str) -> u64 {
    let dir = model_dir(code);
    let runtime = crate::model::lang_gguf::runtime_bytes(&dir);
    let bytes = if runtime > 0 {
        runtime as f64
    } else {
        weight_bytes(&dir) as f64 * LANG_LLM_RESIDENT_RATIO
    };
    (bytes / (1024.0 * 1024.0)).ceil() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DlPhase {
    Absent,
    Queued,
    Downloading,
    Ready,
    Failed,
    Unavailable,
}

#[derive(Clone, Debug)]
pub struct DlState {
    pub phase: DlPhase,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub bytes_per_sec: f64,
    pub error: String,
    pub retry_at_ms: i64,
    pub updated_ms: i64,
}

impl DlState {
    fn new(phase: DlPhase) -> Self {
        Self {
            phase,
            file: String::new(),
            done: 0,
            total: 0,
            bytes_per_sec: 0.0,
            error: String::new(),
            retry_at_ms: 0,
            updated_ms: now_ms(),
        }
    }
}

static STATES: Lazy<Mutex<HashMap<String, DlState>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static RUNTIME_FAIL: Lazy<Mutex<HashMap<String, String>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static QUEUE: Lazy<Mutex<VecDeque<(String, tauri::AppHandle, String)>>> =
    Lazy::new(|| Mutex::new(VecDeque::new()));
static WORKER: AtomicBool = AtomicBool::new(false);
static RESIDENT: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));
static BOUND: Lazy<Mutex<Option<(String, usize)>>> = Lazy::new(|| Mutex::new(None));
static WAITERS: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn set_state<F: FnOnce(&mut DlState)>(code: &str, f: F) {
    if let Ok(mut m) = STATES.lock() {
        let e = m
            .entry(code.to_string())
            .or_insert_with(|| DlState::new(DlPhase::Absent));
        f(e);
        e.updated_ms = now_ms();
    }
}

pub fn state(code: &str) -> DlState {
    let mut s = STATES
        .lock()
        .ok()
        .and_then(|m| m.get(code).cloned())
        .unwrap_or_else(|| DlState::new(DlPhase::Absent));
    let ready = is_ready(code);
    if s.phase != DlPhase::Downloading && s.phase != DlPhase::Queued && ready {
        s.phase = DlPhase::Ready;
    } else if s.phase == DlPhase::Ready && !ready {
        s.phase = DlPhase::Absent;
    }
    s
}

pub fn mark_runtime_failure(code: &str, why: &str) {
    if let Ok(mut m) = RUNTIME_FAIL.lock() {
        m.insert(code.to_string(), why.to_string());
    }
}

pub fn runtime_failure(code: &str) -> Option<String> {
    RUNTIME_FAIL.lock().ok().and_then(|m| m.get(code).cloned())
}

pub fn resident_variant() -> Option<String> {
    RESIDENT.lock().ok().and_then(|g| g.clone())
}

pub fn set_resident_variant(v: Option<String>) {
    if let Ok(mut g) = RESIDENT.lock() {
        *g = v;
    }
}

pub struct TranslitBinding {
    code: Option<String>,
}

impl TranslitBinding {
    pub fn none() -> Self {
        Self { code: None }
    }

    pub fn is_bound(&self) -> bool {
        self.code.is_some()
    }
}

impl Drop for TranslitBinding {
    fn drop(&mut self) {
        let code = match self.code.take() {
            Some(c) => c,
            None => return,
        };
        if let Ok(mut b) = BOUND.lock() {
            let clear = match b.as_mut() {
                Some((c, n)) if *c == code => {
                    *n = n.saturating_sub(1);
                    *n == 0
                }
                _ => false,
            };
            if clear {
                *b = None;
            }
        }
    }
}

pub fn bind_translit(code: &str) -> TranslitBinding {
    if let Ok(mut b) = BOUND.lock() {
        match b.as_mut() {
            None => {
                *b = Some((code.to_string(), 1));
                return TranslitBinding { code: Some(code.to_string()) };
            }
            Some((c, n)) if c.as_str() == code => {
                *n += 1;
                return TranslitBinding { code: Some(code.to_string()) };
            }
            _ => {}
        }
    }
    TranslitBinding::none()
}

pub fn bound_translit_code() -> Option<String> {
    BOUND.lock().ok().and_then(|b| b.as_ref().map(|(c, _)| c.clone()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslitEngine {
    Lang4B { code: String },
    Base2B,
}

impl TranslitEngine {
    pub fn is_lang(&self) -> bool {
        matches!(self, TranslitEngine::Lang4B { .. })
    }

    pub fn code(&self) -> Option<&str> {
        match self {
            TranslitEngine::Lang4B { code } => Some(code.as_str()),
            TranslitEngine::Base2B => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            TranslitEngine::Lang4B { code } => format!("Qwen3.5-4B-{} (alphaedge-ai)", code),
            TranslitEngine::Base2B => "Qwen3.5-2B".to_string(),
        }
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.2}GB", bytes as f64 / 1_000_000_000.0)
}

fn eta(remaining: u64, bps: f64) -> String {
    if bps < 1.0 {
        return "계산 중".to_string();
    }
    let secs = (remaining as f64 / bps).round() as u64;
    if secs >= 3600 {
        format!("{}시간 {}분", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}분 {}초", secs / 60, secs % 60)
    } else {
        format!("{}초", secs)
    }
}

fn percent(done: u64, total: u64) -> u64 {
    if total == 0 {
        0
    } else {
        (done.min(total) * 100) / total
    }
}

fn progress_line(code: &str, st: &DlState) -> String {
    format!(
        "📥 [LANG-LLM] {} · {} {}% ({} / {}) · {:.1}MB/s · 남은 약 {}",
        repo_name(code),
        st.file,
        percent(st.done, st.total),
        gb(st.done),
        gb(st.total),
        st.bytes_per_sec / 1_000_000.0,
        eta(st.total.saturating_sub(st.done), st.bytes_per_sec)
    )
}

pub fn resolve_engine(
    doc_lang: &str,
    app: &tauri::AppHandle,
    task_id: &str,
    fetch: bool,
) -> (TranslitEngine, String) {
    let code = match alphaedge_code(doc_lang) {
        Some(c) => c,
        None => {
            return (
                TranslitEngine::Base2B,
                format!(
                    "🔤 [TRANSLIT ENGINE] 문서 언어 '{}' 에 대응하는 alphaedge-ai Qwen3.5-4B 언어 모델이 없어 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                    doc_lang
                ),
            )
        }
    };
    if let Some(why) = runtime_failure(code) {
        return (
            TranslitEngine::Base2B,
            format!(
                "⚠️ [TRANSLIT ENGINE] Qwen3.5-4B-{} 는 이번 세션에서 쓸 수 없습니다 ({}). 앱을 다시 시작하기 전까지 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                code, why
            ),
        );
    }
    if !is_ready(code) {
        let phase = if fetch {
            request_download(code, app, task_id)
        } else {
            state(code).phase
        };
        let st = state(code);
        let converting = files_ready(code);
        let line = match phase {
            DlPhase::Unavailable => format!(
                "🚫 [TRANSLIT ENGINE] {} 저장소에서 필수 파일을 받을 수 없습니다 ({}). 이 언어는 이번 세션 동안 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                repo_id(code),
                st.error
            ),
            DlPhase::Failed => format!(
                "⚠️ [TRANSLIT ENGINE] {} 다운로드가 실패해 대기 중입니다 ({}). 약 {}분 뒤 다음 태스크에서 받은 곳부터 이어받습니다. 그동안 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                repo_id(code),
                st.error,
                ((st.retry_at_ms - now_ms()).max(0) / 60_000) + 1
            ),
            DlPhase::Downloading if st.file == CONVERT_LABEL && st.total > 0 => format!(
                "🔧 [TRANSLIT ENGINE] {} 실행용 변환 {}% ({} / {}) — 변환하는 동안 음차는 Qwen3.5-2B + 발음 게이트로 진행하며, 끝나면 다음 캐시 미스부터 자동으로 4B 로 전환합니다.",
                repo_id(code),
                percent(st.done, st.total),
                gb(st.done),
                gb(st.total)
            ),
            DlPhase::Downloading if st.total > 0 => format!(
                "{} — 받는 동안 음차는 Qwen3.5-2B + 발음 게이트로 진행하며, 완료되면 다음 캐시 미스부터 자동으로 4B 로 전환합니다.",
                progress_line(code, &st)
            ),
            DlPhase::Ready => format!(
                "✅ [TRANSLIT ENGINE] {} 파일 준비를 방금 마쳤습니다. 다음 태스크부터 4B 를 검토합니다.",
                repo_id(code)
            ),
            DlPhase::Absent if converting => format!(
                "⚪ [TRANSLIT ENGINE] {} 파일은 받아 두었지만 실행용 변환(Q4_K_M GGUF)이 아직입니다. LLM 음차가 실제로 필요해지는 순간 백그라운드로 변환합니다. 지금은 Qwen3.5-2B + 발음 게이트로 진행합니다.",
                repo_id(code)
            ),
            DlPhase::Absent => format!(
                "⚪ [TRANSLIT ENGINE] {} 은 아직 받지 않았습니다. 이 문서 언어는 라틴 문자라 LLM 음차가 드물어, LLM 음차가 실제로 필요해지는 순간 백그라운드로 받습니다. 지금은 Qwen3.5-2B + 발음 게이트로 진행합니다.",
                repo_id(code)
            ),
            _ if converting => format!(
                "🔧 [TRANSLIT ENGINE] {} 파일은 준비되어 있어 실행용 변환(Q4_K_M GGUF, 첫 준비에 한 번)을 백그라운드로 진행합니다. 저장 위치 {}. 그동안 음차는 Qwen3.5-2B + 발음 게이트로 진행하며 작업을 멈출 필요가 없습니다.",
                repo_id(code),
                crate::model::lang_gguf::runtime_dir(&model_dir(code)).display()
            ),
            _ => format!(
                "📥 [TRANSLIT ENGINE] {} 가 설치되어 있지 않아 백그라운드 다운로드를 시작했습니다 (약 7.9GB · 저장 위치 {}). 받는 동안 음차는 Qwen3.5-2B + 발음 게이트로 진행하며 작업을 멈출 필요가 없습니다.",
                repo_id(code),
                model_dir(code).display()
            ),
        };
        return (TranslitEngine::Base2B, line);
    }
    if loader().is_none() {
        return (
            TranslitEngine::Base2B,
            format!(
                "🔌 [TRANSLIT ENGINE] {} 는 준비되었지만 Qwen3.5-4B 로더가 등록되지 않았습니다 (run() 의 register_loader). 이번 태스크는 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                repo_id(code)
            ),
        );
    }
    (
        TranslitEngine::Lang4B { code: code.to_string() },
        format!(
            "🧠 [TRANSLIT ENGINE] 음차 엔진: Qwen3.5-4B-{} (alphaedge-ai · vocab 16384) | 상주 예상 {}MB",
            code,
            resident_estimate_mb(code)
        ),
    )
}

pub fn lang_engine_available(doc_lang: &str) -> bool {
    match alphaedge_code(doc_lang) {
        Some(c) => loader().is_some() && runtime_failure(c).is_none() && is_ready(c),
        None => false,
    }
}

pub fn engine_tag(doc_lang: &str) -> String {
    match alphaedge_code(doc_lang) {
        Some(c) if lang_engine_available(doc_lang) => format!("4B-{}", c),
        _ => "2B".to_string(),
    }
}

pub fn prefetch(doc_lang: &str, app: &tauri::AppHandle, task_id: &str) -> Option<String> {
    if !needs_lang_engine(doc_lang) {
        return None;
    }
    let code = alphaedge_code(doc_lang)?;
    if runtime_failure(code).is_some() {
        return None;
    }
    if is_ready(code) {
        return Some(format!(
            "🧠 [TRANSLIT ENGINE / PREFETCH] 문서 언어 '{}' 의 음차 엔진 {} 이 준비되어 있습니다. 음차 단계에서 바로 씁니다.",
            doc_lang,
            repo_id(code)
        ));
    }
    let phase = request_download(code, app, task_id);
    let st = state(code);
    let line = match phase {
        DlPhase::Queued => format!(
            "📥 [TRANSLIT ENGINE / PREFETCH] 문서 언어 '{}' 의 음차 엔진 {} 이 아직 없어 지금부터 백그라운드로 받습니다 (저장 위치 {}). 추출은 그대로 진행하고, 음차 단계에 이르렀을 때 준비가 안 되어 있으면 그 자리에서 기다립니다.",
            doc_lang,
            repo_id(code),
            model_dir(code).display()
        ),
        DlPhase::Downloading if st.file == CONVERT_LABEL => format!(
            "🔧 [TRANSLIT ENGINE / PREFETCH] {} 실행용 변환이 진행 중입니다 ({}%). 음차 단계에서 끝날 때까지 기다립니다.",
            repo_id(code),
            percent(st.done, st.total)
        ),
        DlPhase::Downloading => format!(
            "{} — 음차 단계에서 끝날 때까지 기다립니다.",
            progress_line(code, &st)
        ),
        DlPhase::Failed => format!(
            "⚠️ [TRANSLIT ENGINE / PREFETCH] {} 이전 다운로드가 실패했습니다 ({}). 음차 단계에서 받은 곳부터 다시 이어받습니다.",
            repo_id(code),
            st.error
        ),
        DlPhase::Unavailable => format!(
            "🚫 [TRANSLIT ENGINE / PREFETCH] {} 저장소에서 필수 파일을 받을 수 없습니다 ({}). 이 언어는 이번 세션 동안 Qwen3.5-2B + 발음 게이트로 음차합니다.",
            repo_id(code),
            st.error
        ),
        DlPhase::Ready | DlPhase::Absent => format!(
            "⚪ [TRANSLIT ENGINE / PREFETCH] {} 상태 {:?}. 음차 단계에서 다시 확인합니다.",
            repo_id(code),
            phase
        ),
    };
    Some(line)
}

pub fn translit_demand(item: &Value) -> usize {
    fn skip_key(k: &str) -> bool {
        let l = k.to_lowercase();
        l == "id" || l == "link" || l == "index" || l == "type" || l == "detail" || l == "digest"
            || l == "text" || l == "masked_text" || l == "mode" || l == "updated_at" || l == "created_at"
            || l.contains("url") || l.contains("link") || l.contains("image")
            || l.starts_with("rel_") || l.starts_with("reference_") || l.starts_with('_')
            || l.contains("insight") || l.contains("summary") || l.contains("analysis")
    }
    fn walk(v: &Value, out: &mut usize) {
        match v {
            Value::Object(m) => {
                for (k, x) in m.iter() {
                    if skip_key(k) {
                        continue;
                    }
                    match x {
                        Value::String(s) => {
                            let t = s.trim();
                            let n = t.chars().count();
                            if n < 2 || n > 150 || t.contains("://") {
                                continue;
                            }
                            let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
                            if digits * 2 >= n {
                                continue;
                            }
                            let (_, latin) = crate::nl_convert::split_words_by_script(t);
                            if latin.iter().any(|w| w.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 2) {
                                *out += 1;
                            }
                        }
                        Value::Array(_) | Value::Object(_) => walk(x, out),
                        _ => {}
                    }
                }
            }
            Value::Array(a) => {
                for x in a.iter() {
                    walk(x, out);
                }
            }
            _ => {}
        }
    }
    let mut n = 0usize;
    walk(item, &mut n);
    n
}

pub async fn await_engine(
    doc_lang: &str,
    app: &tauri::AppHandle,
    task_id: &str,
    cancel: &std::sync::Arc<AtomicBool>,
    demand: usize,
) -> (TranslitEngine, String) {
    if !needs_lang_engine(doc_lang) {
        return resolve_engine(doc_lang, app, task_id, false);
    }
    let code = match alphaedge_code(doc_lang) {
        Some(c) => c,
        None => return resolve_engine(doc_lang, app, task_id, false),
    };
    if is_ready(code) || runtime_failure(code).is_some() {
        return resolve_engine(doc_lang, app, task_id, true);
    }
    if demand == 0 {
        let phase = request_download(code, app, task_id);
        return (
            TranslitEngine::Base2B,
            format!(
                "⚪ [TRANSLIT ENGINE / WAIT SKIP] 이 문서의 값에는 라틴 문자 단어가 없어 LLM 음차가 필요하지 않습니다. {} 은 백그라운드로 계속 받고({:?}) 이번 태스크는 기다리지 않습니다.",
                repo_id(code),
                phase
            ),
        );
    }
    let started = Instant::now();
    let mut retries = 0u32;
    let mut last_beat = Instant::now();
    let _waiting = WaitGuard::new(task_id);
    announce(
        app,
        task_id,
        &format!(
            "⏸️ [TRANSLIT ENGINE / WAIT] 음차 단계에 도달했지만 {} 이 아직 준비되지 않았습니다 (라틴 단어를 가진 값 {}개). 이 태스크는 여기서 멈추고 모델을 먼저 받은 뒤(필요하면 실행용 변환까지) 같은 자리에서 음차를 이어 갑니다. 그동안 Qwen3.5-2B 로 음차하지 않습니다 — 2B 결과가 캐시에 남으면 4B 가 준비된 뒤에도 그 값이 재사용되기 때문입니다.",
            repo_id(code),
            demand
        ),
    );
    progress_card(app, task_id, "Model Download", &format!("Waiting for {}...", repo_name(code)));
    loop {
        if cancel.load(Ordering::Relaxed) {
            return (
                TranslitEngine::Base2B,
                format!(
                    "🛑 [TRANSLIT ENGINE / WAIT] 태스크가 취소되어 {} 대기를 중단합니다. 다운로드는 백그라운드에서 계속됩니다.",
                    repo_id(code)
                ),
            );
        }
        if is_ready(code) {
            break;
        }
        let st = state(code);
        match st.phase {
            DlPhase::Unavailable => {
                return (
                    TranslitEngine::Base2B,
                    format!(
                        "🚫 [TRANSLIT ENGINE / WAIT] {} 저장소에서 필수 파일을 받을 수 없습니다 ({}). 기다려도 해결되지 않으므로 이번 태스크는 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                        repo_id(code),
                        st.error
                    ),
                );
            }
            DlPhase::Failed => {
                if runtime_failure(code).is_some() {
                    break;
                }
                if retries >= WAIT_RETRIES {
                    return (
                        TranslitEngine::Base2B,
                        format!(
                            "⚠️ [TRANSLIT ENGINE / WAIT] {} 다운로드를 {}번 다시 시도했지만 이어받지 못했습니다 ({}). 이번 태스크는 Qwen3.5-2B + 발음 게이트로 음차하고, 받은 부분(.part)은 다음 태스크가 이어받습니다.",
                            repo_id(code),
                            retries,
                            st.error
                        ),
                    );
                }
                retries += 1;
                set_state(code, |s| {
                    s.retry_at_ms = 0;
                });
                announce(
                    app,
                    task_id,
                    &format!(
                        "🔁 [TRANSLIT ENGINE / WAIT] 다운로드가 끊겨({}) 받은 곳부터 바로 이어받습니다 ({}/{}).",
                        st.error, retries, WAIT_RETRIES
                    ),
                );
                request_download(code, app, task_id);
            }
            DlPhase::Absent | DlPhase::Ready => {
                request_download(code, app, task_id);
            }
            DlPhase::Queued | DlPhase::Downloading => {}
        }
        if last_beat.elapsed().as_secs() >= WAIT_HEARTBEAT_SECS {
            last_beat = Instant::now();
            let st = state(code);
            let line = if st.file == CONVERT_LABEL {
                format!(
                    "🔧 [LANG-LLM] {} · {} {}% ({} / {})",
                    repo_name(code),
                    CONVERT_LABEL,
                    percent(st.done, st.total),
                    gb(st.done),
                    gb(st.total)
                )
            } else if st.total > 0 {
                progress_line(code, &st)
            } else {
                format!("📥 [LANG-LLM] {} · 연결 중 ({:?})", repo_name(code), st.phase)
            };
            announce(
                app,
                task_id,
                &format!("⏳ [TRANSLIT ENGINE / WAIT] {} · 대기 {}초", line, started.elapsed().as_secs()),
            );
            progress_card(
                app,
                task_id,
                "Model Download",
                &format!("{} {}%", if st.file == CONVERT_LABEL { CONVERT_LABEL } else { "Downloading" }, percent(st.done, st.total)),
            );
        }
        tokio::time::sleep(Duration::from_millis(WAIT_TICK_MS)).await;
    }
    let waited = started.elapsed().as_secs();
    crate::utils::score_dynamics::record_baseline("indexing.translit_engine_wait_secs", waited as f32);
    let (engine, status) = resolve_engine(doc_lang, app, task_id, true);
    announce(
        app,
        task_id,
        &format!("▶️ [TRANSLIT ENGINE / RESUME] {}초 기다린 뒤 음차를 같은 자리에서 이어 갑니다. {}", waited, status),
    );
    progress_card(app, task_id, "Handover", "Resuming transliteration...");
    (engine, status)
}

fn console_targets(task_id: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if !task_id.is_empty() {
        out.push(task_id.to_string());
    }
    if let Ok(w) = WAITERS.lock() {
        for t in w.iter() {
            if !out.iter().any(|x| x == t) {
                out.push(t.clone());
            }
        }
    }
    out
}

fn announce(app: &tauri::AppHandle, task_id: &str, msg: &str) {
    println!("{}", msg);
    for t in console_targets(task_id) {
        let _ = app.emit(
            "task-console-log",
            json!({"task_id": t, "text": format!("{}\n", msg)}),
        );
    }
}

fn announce_fresh_line(app: &tauri::AppHandle, task_id: &str, msg: &str) {
    println!("\n{}", msg);
    for t in console_targets(task_id) {
        let _ = app.emit(
            "task-console-log",
            json!({"task_id": t, "text": format!("{}\n", msg)}),
        );
    }
}

struct WaitGuard(String);

impl WaitGuard {
    fn new(task_id: &str) -> Self {
        if !task_id.is_empty() {
            if let Ok(mut w) = WAITERS.lock() {
                w.push(task_id.to_string());
            }
        }
        WaitGuard(task_id.to_string())
    }
}

impl Drop for WaitGuard {
    fn drop(&mut self) {
        if let Ok(mut w) = WAITERS.lock() {
            if let Some(i) = w.iter().position(|x| *x == self.0) {
                w.remove(i);
            }
        }
    }
}

fn progress_card(app: &tauri::AppHandle, task_id: &str, category: &str, summary: &str) {
    if task_id.is_empty() {
        return;
    }
    let payload = json!({
        "task_id": task_id,
        "category": category,
        "summary": summary,
        "spinner": "📥"
    });
    let _ = app.emit("extraction-progress", &payload);
    crate::utils::logger::log_task_progress(app, task_id, &payload);
}

fn publish(app: &tauri::AppHandle, code: &str) {
    let _ = app.emit("lang-llm-status", status_json(code));
}

pub fn request_download(code: &str, app: &tauri::AppHandle, task_id: &str) -> DlPhase {
    if is_ready(code) {
        return DlPhase::Ready;
    }
    let st = state(code);
    match st.phase {
        DlPhase::Unavailable | DlPhase::Queued | DlPhase::Downloading => return st.phase,
        DlPhase::Failed if now_ms() < st.retry_at_ms => return DlPhase::Failed,
        _ => {}
    }
    set_state(code, |s| {
        s.phase = DlPhase::Queued;
        s.error.clear();
    });
    if let Ok(mut q) = QUEUE.lock() {
        if !q.iter().any(|(c, _, _)| c.as_str() == code) {
            q.push_back((code.to_string(), app.clone(), task_id.to_string()));
        }
    }
    spawn_worker_if_idle();
    DlPhase::Queued
}

fn spawn_worker_if_idle() {
    if WORKER.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async {
        loop {
            let next = QUEUE.lock().ok().and_then(|mut q| q.pop_front());
            match next {
                Some((code, app, task_id)) => download_repo(&code, &app, &task_id).await,
                None => {
                    WORKER.store(false, Ordering::SeqCst);
                    let pending = QUEUE.lock().map(|q| !q.is_empty()).unwrap_or(false);
                    if pending && !WORKER.swap(true, Ordering::SeqCst) {
                        continue;
                    }
                    break;
                }
            }
        }
    });
}

enum DlError {
    Missing(u16),
    Failed(String),
}

fn content_range_total(res: &reqwest::Response) -> Option<u64> {
    res.headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.rsplit('/').next())
        .and_then(|t| t.trim().parse::<u64>().ok())
}

fn backoff_secs(attempt: u32) -> u64 {
    match attempt {
        0 | 1 => 0,
        2 => 3,
        3 => 10,
        _ => 30,
    }
}

async fn download_file(
    client: &reqwest::Client,
    code: &str,
    file: &str,
    dir: &Path,
    app: &tauri::AppHandle,
    task_id: &str,
) -> Result<u64, DlError> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let dst = dir.join(file);
    if let Ok(m) = std::fs::metadata(&dst) {
        if m.len() > 0 {
            return Ok(m.len());
        }
    }
    let part = dir.join(format!("{}.part", file));
    let url = file_url(code, file);
    let mut last_err = String::new();

    for attempt in 1..=FILE_ATTEMPTS {
        let wait = backoff_secs(attempt);
        if wait > 0 {
            tokio::time::sleep(Duration::from_secs(wait)).await;
        }
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut req = client.get(&url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={}-", have));
        }
        let res = match tokio::time::timeout(Duration::from_secs(CHUNK_TIMEOUT_SECS), req.send()).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                last_err = e.to_string();
                continue;
            }
            Err(_) => {
                last_err = format!("{}초 동안 응답 헤더가 오지 않았습니다", CHUNK_TIMEOUT_SECS);
                continue;
            }
        };
        let status = res.status().as_u16();
        if matches!(status, 401 | 403 | 404) {
            return Err(DlError::Missing(status));
        }
        if status == 416 {
            let remote = match client
                .get(&url)
                .header(reqwest::header::RANGE, "bytes=0-0")
                .send()
                .await
            {
                Ok(r) => content_range_total(&r),
                Err(_) => None,
            };
            if remote == Some(have) {
                if dst.exists() {
                    let _ = std::fs::remove_file(&dst);
                }
                if std::fs::rename(&part, &dst).is_ok() {
                    return Ok(have);
                }
            }
            let _ = std::fs::remove_file(&part);
            last_err = "HTTP 416 (이어받기 범위가 서버 파일과 맞지 않아 처음부터 다시 받습니다)".to_string();
            continue;
        }
        if !(200..300).contains(&status) {
            last_err = format!("HTTP {}", status);
            continue;
        }
        let resumed = status == 206 && have > 0;
        let total = if resumed {
            content_range_total(&res).unwrap_or(0)
        } else {
            res.content_length().unwrap_or(0)
        };
        let loud = total >= PROGRESS_MIN_BYTES;
        let mut done = if resumed { have } else { 0 };
        let opened = if resumed {
            tokio::fs::OpenOptions::new().append(true).open(&part).await
        } else {
            tokio::fs::File::create(&part).await
        };
        let mut f = match opened {
            Ok(f) => f,
            Err(e) => {
                last_err = format!("임시 파일 열기 실패: {}", e);
                continue;
            }
        };
        set_state(code, |s| {
            s.phase = DlPhase::Downloading;
            s.file = file.to_string();
            s.done = done;
            s.total = total;
            s.bytes_per_sec = 0.0;
        });
        if resumed && loud {
            announce(
                app,
                task_id,
                &format!(
                    "↪️ [LANG-LLM] {} 이어받기: 이미 받은 {} 다음부터 계속합니다.",
                    file,
                    gb(have)
                ),
            );
        }
        let started = Instant::now();
        let mut session_bytes: u64 = 0;
        let mut next_mark = if total > 0 {
            (percent(done, total) / PROGRESS_STEP) * PROGRESS_STEP + PROGRESS_STEP
        } else {
            u64::MAX
        };
        let mut stream = res.bytes_stream();
        let mut broken: Option<String> = None;
        loop {
            match tokio::time::timeout(Duration::from_secs(CHUNK_TIMEOUT_SECS), stream.next()).await {
                Err(_) => {
                    broken = Some(format!("{}초 동안 데이터가 오지 않았습니다", CHUNK_TIMEOUT_SECS));
                    break;
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => {
                    broken = Some(e.to_string());
                    break;
                }
                Ok(Some(Ok(chunk))) => {
                    if let Err(e) = f.write_all(&chunk).await {
                        broken = Some(format!("디스크 기록 실패: {}", e));
                        break;
                    }
                    done += chunk.len() as u64;
                    session_bytes += chunk.len() as u64;
                    let bps = session_bytes as f64 / started.elapsed().as_secs_f64().max(0.001);
                    set_state(code, |s| {
                        s.done = done;
                        s.bytes_per_sec = bps;
                    });
                    if loud && percent(done, total) >= next_mark {
                        next_mark = (percent(done, total) / PROGRESS_STEP) * PROGRESS_STEP + PROGRESS_STEP;
                        announce_fresh_line(app, task_id, &progress_line(code, &state(code)));
                        publish(app, code);
                    }
                }
            }
        }
        let _ = f.flush().await;
        drop(f);
        if let Some(e) = broken {
            last_err = e;
            continue;
        }
        let written = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if total > 0 && written != total {
            last_err = format!("크기 불일치 {} / {}", written, total);
            continue;
        }
        if dst.exists() {
            let _ = std::fs::remove_file(&dst);
        }
        if let Err(e) = std::fs::rename(&part, &dst) {
            last_err = format!("파일 이름 확정 실패: {}", e);
            continue;
        }
        return Ok(written);
    }
    Err(DlError::Failed(last_err))
}

fn fail(code: &str, app: &tauri::AppHandle, task_id: &str, why: &str) {
    set_state(code, |s| {
        s.phase = DlPhase::Failed;
        s.error = why.to_string();
        s.retry_at_ms = now_ms() + FAIL_RETRY_SECS * 1000;
    });
    publish(app, code);
    announce(
        app,
        task_id,
        &format!(
            "⚠️ [LANG-LLM] {} 다운로드 실패: {} — 받은 부분은 .part 로 남겨 둡니다. 음차 단계에서 기다리는 태스크는 바로 이어받기를 다시 시도하고, 기다리는 태스크가 없으면 {}분 뒤 다음 태스크에서 이어받습니다.",
            repo_id(code),
            why,
            FAIL_RETRY_SECS / 60
        ),
    );
}

async fn download_repo(code: &str, app: &tauri::AppHandle, task_id: &str) {
    let dir = model_dir(code);
    if files_ready(code) {
        finish_repo(code, app, task_id).await;
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        fail(code, app, task_id, &format!("모델 폴더를 만들 수 없습니다: {}", e));
        return;
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    set_state(code, |s| {
        s.phase = DlPhase::Downloading;
        s.error.clear();
    });
    publish(app, code);
    announce(
        app,
        task_id,
        &format!(
            "📥 [LANG-LLM] {} 백그라운드 다운로드를 시작합니다. 저장 위치: {} | 필수 파일 {:?} | 추출 단계는 그대로 진행되고, 음차 단계에 이른 태스크는 이 다운로드(와 실행용 변환)가 끝날 때까지 그 자리에서 기다렸다가 4B 로 이어 갑니다. 직접 받을 필요는 없습니다.",
            repo_id(code),
            dir.display(),
            LANG_LLM_REQUIRED
        ),
    );
    for file in LANG_LLM_FILES.iter() {
        let required = LANG_LLM_REQUIRED.contains(file);
        match download_file(&client, code, file, &dir, app, task_id).await {
            Ok(_) => {}
            Err(DlError::Missing(status)) if !required => {
                announce(
                    app,
                    task_id,
                    &format!(
                        "⚪ [LANG-LLM] 선택 파일 {} 이 저장소에 없습니다 (HTTP {}). 필수 파일이 아니므로 건너뜁니다.",
                        file, status
                    ),
                );
            }
            Err(DlError::Missing(status)) => {
                set_state(code, |s| {
                    s.phase = DlPhase::Unavailable;
                    s.error = format!("{} HTTP {}", file, status);
                });
                publish(app, code);
                announce(
                    app,
                    task_id,
                    &format!(
                        "🚫 [LANG-LLM] {} 의 필수 파일 {} 을 받을 수 없습니다 (HTTP {}). 이 언어는 이번 세션 동안 Qwen3.5-2B + 발음 게이트로 음차합니다.",
                        repo_id(code),
                        file,
                        status
                    ),
                );
                return;
            }
            Err(DlError::Failed(e)) if !required => {
                announce(
                    app,
                    task_id,
                    &format!(
                        "⚠️ [LANG-LLM] 선택 파일 {} 을 받지 못했습니다 ({}). 필수 파일이 아니므로 계속합니다.",
                        file, e
                    ),
                );
            }
            Err(DlError::Failed(e)) => {
                fail(code, app, task_id, &format!("{}: {}", file, e));
                return;
            }
        }
    }
    if !files_ready(code) {
        fail(code, app, task_id, "필수 파일 또는 가중치 용량 검증에 실패했습니다");
        return;
    }
    finish_repo(code, app, task_id).await;
}

async fn finish_repo(code: &str, app: &tauri::AppHandle, task_id: &str) {
    if !convert_runtime(code, app, task_id).await {
        return;
    }
    let dir = model_dir(code);
    set_state(code, |s| {
        s.phase = DlPhase::Ready;
        s.error.clear();
    });
    publish(app, code);
    let next = if loader().is_some() {
        "다음 음차 캐시 미스부터 이 모델로 전환합니다."
    } else {
        "Qwen3.5-4B 로더가 등록되기 전까지는 Qwen3.5-2B + 발음 게이트로 진행합니다."
    };
    announce(
        app,
        task_id,
        &format!(
            "✅ [LANG-LLM] {} 준비 완료 (원본 가중치 {} → 실행 파일 {}). {}",
            repo_id(code),
            gb(weight_bytes(&dir)),
            gb(crate::model::lang_gguf::runtime_bytes(&dir)),
            next
        ),
    );
}

async fn convert_runtime(code: &str, app: &tauri::AppHandle, task_id: &str) -> bool {
    let dir = model_dir(code);
    if crate::model::lang_gguf::runtime_ready(&dir) {
        return true;
    }
    announce(
        app,
        task_id,
        &format!(
            "🔧 [LANG-LLM] {} 를 Qwen3.5 런타임이 읽는 GGUF(Q4_K_M 규칙)로 변환합니다. 저장 위치: {} | 원본 safetensors 는 그대로 두고 첫 준비에 한 번만 수행합니다. 음차 단계에서 기다리는 태스크는 변환이 끝나는 즉시 4B 로 이어 갑니다.",
            repo_id(code),
            crate::model::lang_gguf::runtime_dir(&dir).display()
        ),
    );
    set_state(code, |s| {
        s.phase = DlPhase::Downloading;
        s.file = CONVERT_LABEL.to_string();
        s.done = 0;
        s.total = 0;
        s.bytes_per_sec = 0.0;
    });
    publish(app, code);
    let (c, a, t) = (code.to_string(), app.clone(), task_id.to_string());
    let started = Instant::now();
    let joined = tokio::task::spawn_blocking(move || {
        let next_mark = std::sync::atomic::AtomicU64::new(PROGRESS_STEP);
        crate::model::lang_gguf::prepare(&model_dir(&c), &|done, total| {
            let bps = done as f64 / started.elapsed().as_secs_f64().max(0.001);
            set_state(&c, |s| {
                s.done = done;
                s.total = total;
                s.bytes_per_sec = bps;
            });
            let pct = percent(done, total);
            if total > 0 && pct >= next_mark.load(Ordering::SeqCst) {
                next_mark.store((pct / PROGRESS_STEP) * PROGRESS_STEP + PROGRESS_STEP, Ordering::SeqCst);
                announce_fresh_line(
                    &a,
                    &t,
                    &format!(
                        "🔧 [LANG-LLM] {} · {} {}% ({} / {}) · 남은 약 {}",
                        repo_name(&c),
                        CONVERT_LABEL,
                        pct,
                        gb(done),
                        gb(total),
                        eta(total.saturating_sub(done), bps)
                    ),
                );
                publish(&a, &c);
            }
        })
    })
    .await;
    let err = match joined {
        Ok(Ok(_)) => {
            announce(
                app,
                task_id,
                &format!(
                    "✅ [LANG-LLM] {} 변환 완료 ({} · {}초).",
                    repo_name(code),
                    gb(crate::model::lang_gguf::runtime_bytes(&dir)),
                    started.elapsed().as_secs()
                ),
            );
            return true;
        }
        Ok(Err(e)) => format!("{:#}", e),
        Err(e) => e.to_string(),
    };
    let why = format!("Q4_K_M 변환 실패: {}", err);
    mark_runtime_failure(code, &why);
    set_state(code, |s| {
        s.phase = DlPhase::Failed;
        s.error = why.clone();
        s.retry_at_ms = now_ms() + FAIL_RETRY_SECS * 1000;
    });
    publish(app, code);
    announce(
        app,
        task_id,
        &format!(
            "⚠️ [LANG-LLM] {} {} — 받은 원본 파일은 그대로 두었습니다. 이번 세션 동안은 Qwen3.5-2B + 발음 게이트로 음차하고, 앱을 다시 시작하면 변환을 다시 시도합니다.",
            repo_id(code),
            why
        ),
    );
    false
}

fn installed_codes() -> Vec<String> {
    let root = crate::utils::get_app_dir().join("models");
    let mut out: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(rest) = name.strip_prefix("Qwen3.5-4B-") {
                if let Some(code) = rest.strip_suffix("-16384") {
                    if !code.is_empty() && !out.iter().any(|c| c == code) {
                        out.push(code.to_string());
                    }
                }
            }
        }
    }
    if let Ok(m) = STATES.lock() {
        for k in m.keys() {
            if !out.iter().any(|c| c == k) {
                out.push(k.clone());
            }
        }
    }
    out.sort();
    out
}

pub fn status_json(code: &str) -> Value {
    let st = state(code);
    json!({
        "code": code,
        "repo": repo_id(code),
        "dir": model_dir(code).to_string_lossy(),
        "phase": format!("{:?}", st.phase),
        "ready": is_ready(code),
        "files_ready": files_ready(code),
        "runtime_file": crate::model::lang_gguf::runtime_gguf(&model_dir(code)).to_string_lossy(),
        "runtime_bytes": crate::model::lang_gguf::runtime_bytes(&model_dir(code)),
        "file": st.file,
        "done": st.done,
        "total": st.total,
        "percent": percent(st.done, st.total),
        "bytes_per_sec": st.bytes_per_sec,
        "error": st.error,
        "loader_linked": loader().is_some(),
        "runtime_failure": runtime_failure(code),
    })
}

pub fn status_report(doc_lang: Option<&str>) -> Value {
    if let Some(code) = doc_lang.and_then(alphaedge_code) {
        return status_json(code);
    }
    let langs: Vec<Value> = installed_codes().iter().map(|c| status_json(c)).collect();
    json!({
        "languages": langs,
        "loader_linked": loader().is_some(),
    })
}