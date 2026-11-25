//! 🌟 [VOCAB TRIM] 어휘 잘라 내기: 생성 모델(Qwen3 0.6B · Qwen3.5 2B · 4B)의 출력 투영에서 남길 행 고르기와 그 행만의 출력 투영.
//!
//! 입력 임베딩은 원본 전체 어휘 그대로 두고(프롬프트 토큰은 어떤 것이든 그대로), 매 디코드 스텝 행렬-벡터 곱의 대부분을 차지하는 출력 투영만
//! 남긴 K 행으로 곱합니다. 잘린 행은 어디에도 남기지 않습니다. 묶인 표라 입력 임베딩 행은 남긴 행의 출력 투영(장치)에서 꺼내고, 남긴 행 밖의
//! 토큰만 매핑에서 읽습니다 (`EmbedTable::forward_with_head` · `rows_f32_with_head`). 남길 행 (blueprint-sandbox `core/src/ai/trim.rs` 와 같은 규칙):
//!
//! - **공통 코어**: 추가 · 특수 토큰, 한 바이트 토큰 256 개(바이트 수준 BPE 의 바이트 폴백), 혼자서는 UTF-8 이 아닌 바이트 조각 전부,
//!   기호 · 단위 · 그리스 문자 토큰, ASCII 전부 (영어는 어느 문서에서나 전부 디코딩됩니다: JSON 키 · 영어 값 · 숫자 · 코드).
//! - **문서 언어의 문자 체계 전부**: 한글(ko) · 라틴 확장(de · fr · es · it · pt · nl · cs) · 가나(ja) · 아랍 문자(ar) · 한자(ja · zh).
//!   한자를 쓰지 않는 문서에 섞인 한자(한국어 문서의 한자 등, [`Langs::hanja`])는 도메인 글에 나온 것 + 병합 순 상위 [`CJK_COMMON`] 개.
//!   (일본어 · 중국어 문서도 상위 4,096 개만 남기면 blueprint 의 Zoro-1(같은 Qwen3.5 어휘)에서 일본어 한자 設 · 場 · 階 가 간체로 바뀌고
//!   중국어 여러 글자 토큰이 빠져 글자가 사라졌습니다.) 12 개 언어 밖(키릴 · 타이 · 데바나가리 문자 등, `other`)은
//!   그 언어의 문서이거나 글에 나왔을 때만 남깁니다.
//! - **글에 실제로 나온 문자 체계**: 판정 언어와 합칩니다 ([`langs_of_document`]). `detect_document_language` 는 라틴 언어를
//!   fr · de · es · it · pt · nl · cs 밖이면 `en` 으로 돌려주므로(폴란드어 ł · ś 등), 판정만 따르면 문서에 있는 글자를 잘라 냅니다.
//! - **도메인 글**: 내장 도메인 사전(`bias.json` · `full_bias.json` 의 모든 글)과 JSON · HTML 표 태그가 쓰는 토큰 전부.
//! - 남긴 수는 64 의 배수.
//!
//! 문서 언어는 `utils::lang_utils::detect_document_language` 가 정합니다 (현지 언어 + 영어 두 코드). 텍스트 문서는 생성 전에
//! (스케줄러의 `[DOC LANG]`), 이미지 문서는 첫 판독 글로 정합니다 ([`DocLangScope`] · [`observe_read`], 실행 흐름마다). 정하지 않았으면 문자 체계를 빼지 않습니다.
//!
//! 출력은 장치에서 전체 어휘 길이로 다시 펼칩니다(역 번호표 `index_select`): 남긴 행의 로짓은 그 자리에, 잘린 행은 −∞. 그래서 생성 루프의
//! 토큰 번호 기반 처리(끝 토큰 · `{` 강제 · 무시 목록 · 반복 벌점 · 선입견 벡터)는 그대로이고, 잘린 토큰은 절대 뽑히지 않습니다.
//! 로짓 수치는 슬롯의 원래 출력 투영과 같은 dtype 입니다 (Qwen3 · Qwen3.5 는 F32, QwenVL 의 `QLinear` 는 BF16 출력).
//! 샘플링(온도 > 0)은 전체 행으로 곱합니다 ([`refresh_for`]).

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor};
use once_cell::sync::Lazy;
use rayon::prelude::*;

// 🌟 [PD-QUANT TRIM]
pub use premium_discount_quant::trim::{
    kind_of, select_compact_track_m, select_from, Kind, Langs, TrimRule, ALIGN, CJK_COMMON,
};

/// 문서 글에서: `detect_document_language` 의 현지 언어 + 영어, 그리고 글에 실제로 나온 문자 체계 (예전 `Langs::of_document`)
pub fn langs_of_document(text: &str) -> Langs {
    Langs::of_doc_lang(&crate::utils::lang_utils::detect_document_language(text)).union(Langs::present(text))
}

/// Normalize language code / name preserving `zh-tw`, `zh-hk`, and Tier-2 fallback codes (`ru`, `th`, etc.).
pub fn normalize_lang_tag(raw: &str) -> Option<String> {
    let lower = raw.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    let canon = match lower.as_str() {
        "zh-tw" | "zh_tw" | "zh-hant" | "zh_hant" | "zho-tw" | "traditional chinese" => "zh-tw",
        "zh-hk" | "zh_hk" => "zh-hk",
        "zh" | "zho" | "cmn" | "chinese" | "zh-cn" | "zh_cn" | "zh-hans" | "zh_hans" => "zh",
        "ko" | "kor" | "korean" => "ko",
        "en" | "eng" | "english" => "en",
        "ja" | "jpn" | "japanese" => "ja",
        "ar" | "ara" | "arabic" => "ar",
        "fr" | "fra" | "fre" | "french" => "fr",
        "de" | "deu" | "ger" | "german" => "de",
        "es" | "spa" | "spanish" => "es",
        "it" | "ita" | "italian" => "it",
        "pt" | "por" | "portuguese" | "pt-br" | "pt_br" => "pt",
        "nl" | "nld" | "dut" | "dutch" => "nl",
        "cs" | "ces" | "cze" | "czech" => "cs",
        "ru" | "rus" | "russian" => "ru",
        "uk" | "ukr" | "ukrainian" => "uk",
        "th" | "tha" | "thai" => "th",
        "hi" | "hin" | "hindi" => "hi",
        "el" | "ell" | "gre" | "greek" => "el",
        "he" | "heb" | "iw" | "hebrew" => "he",
        "vi" | "vie" | "vietnamese" => "vi",
        "pl" | "pol" | "polish" => "pl",
        "tr" | "tur" | "turkish" => "tr",
        "id" | "ind" | "indonesian" => "id",
        other => other.split(['-', '_']).next().unwrap_or(other),
    };
    Some(canon.to_string())
}

/// `LOGIS_VOCAB_TRIM` (기본 Track M `compact`, `full` · `off` 선택). 읽지 못한 값은 알리고 full 로 둡니다.
pub fn rule_from_env() -> TrimRule {
    match std::env::var(TRIM_ENV) {
        Ok(v) => {
            if v.trim().eq_ignore_ascii_case("compact") || v.trim().is_empty() {
                return TrimRule::Full;
            }
            TrimRule::parse(&v).unwrap_or_else(|e| {
                println!("[VOCAB-TRIM] {TRIM_ENV}={v:?}: {e} → full");
                TrimRule::Full
            })
        }
        Err(_) => TrimRule::Full,
    }
}

/// 규칙 (`compact` 기본 · `full` · `off`)
pub const TRIM_ENV: &str = "LOGIS_VOCAB_TRIM";
/// 언어 (`auto` = 문서 언어, `all`, 또는 `ko,en` 처럼)
pub const LANGS_ENV: &str = "LOGIS_VOCAB_LANGS";
/// 가드 모드 (`off` 기본 · `clustered` · `single`)
pub const GUARD_ENV: &str = "LOGIS_VOCAB_GUARD";
/// 컴팩트 모드 목표 활성 행 수 (예: `12288`)
pub const ACTIVE_K_ENV: &str = "LOGIS_VOCAB_ACTIVE_K";
/// 사전 계산된 `VocabPlan` JSON 경로
pub const PLAN_ENV: &str = "LOGIS_VOCAB_PLAN";

/// GPU W8A16 `forward_exact` kernel switch (ON by default when unset or `1|true|on|strict|e1`; CPU/Vulkan skip at call site).
pub fn exact_lm_head_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("LOGIS_EXACT_LM_HEAD")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        !matches!(v.as_str(), "0" | "off" | "false" | "no" | "legacy")
    })
}

/// Whether `temperature > 0` sampling falls back to full rows (`LOGIS_SAMPLE_FULL_VOCAB=1`, default `false` for Track M K).
pub fn sample_full_vocab_env() -> bool {
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FULL.get_or_init(|| {
        matches!(
            std::env::var("LOGIS_SAMPLE_FULL_VOCAB")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabExecMode {
    ArtifactDirect,
    RuntimeSliced,
    FullUntrimmed,
}

impl VocabExecMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ArtifactDirect => "artifact_direct",
            Self::RuntimeSliced => "runtime_sliced",
            Self::FullUntrimmed => "full_untrimmed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardMode {
    Clustered,
    Single,
    Off,
}

impl GuardMode {
    pub fn from_env() -> Self {
        GuardPolicyConfig::from_env().guard_mode
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clustered => "clustered",
            Self::Single => "single",
            Self::Off => "off",
        }
    }
}

/// Immutable per-head guard policy snapshot captured at build time (0 `env::var` syscalls during decode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardPolicyConfig {
    pub guard_mode: GuardMode,
    /// Strict E1 verification mode (`LOGIS_EXACT_LM_HEAD=strict|e1|lossless`), distinct from W8A16 kernel toggle.
    pub strict_e1: bool,
    /// Enable 16-step `NetGainGate` circuit breaker (default `true`).
    pub net_gain_gate: bool,
    /// Whether this head was promoted to `all_rows` due to `strict_e1` circuit trip.
    pub promoted_from_e1: bool,
}

impl Default for GuardPolicyConfig {
    fn default() -> Self {
        Self {
            guard_mode: GuardMode::Off,
            strict_e1: false,
            net_gain_gate: true,
            promoted_from_e1: false,
        }
    }
}

impl GuardPolicyConfig {
    /// Pure parser for thread-safe unit testing without global `std::env::set_var` races.
    pub fn from_raw_vars(
        raw_guard: Option<&str>,
        raw_exact: Option<&str>,
        raw_gate: Option<&str>,
    ) -> Self {
        let strict_e1 = raw_exact.is_some_and(|v| {
            matches!(v.trim().to_ascii_lowercase().as_str(), "strict" | "e1" | "lossless")
        });
        let guard_mode = match raw_guard.map(|s| s.trim().to_ascii_lowercase()) {
            Some(ref s) if matches!(s.as_str(), "" | "off" | "0" | "none" | "false") => {
                if s.is_empty() && strict_e1 {
                    GuardMode::Clustered
                } else {
                    GuardMode::Off
                }
            }
            Some(ref s) if matches!(s.as_str(), "single" | "sphere") => GuardMode::Single,
            Some(ref s) if matches!(s.as_str(), "clustered" | "cold" | "on" | "1" | "true") => {
                GuardMode::Clustered
            }
            Some(_) => GuardMode::Off,
            None => {
                if strict_e1 {
                    GuardMode::Clustered
                } else {
                    GuardMode::Off
                }
            }
        };
        let net_gain_gate = !raw_gate.is_some_and(|v| {
            matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no")
        });
        Self {
            guard_mode,
            strict_e1,
            net_gain_gate,
            promoted_from_e1: false,
        }
    }

    pub fn from_env() -> Self {
        let g = std::env::var(GUARD_ENV).ok();
        let e = std::env::var("LOGIS_EXACT_LM_HEAD").ok();
        let n = std::env::var("LOGIS_TRIM_NET_GAIN_GATE").ok();
        Self::from_raw_vars(g.as_deref(), e.as_deref(), n.as_deref())
    }
}

pub fn guard_clusters_from_env() -> usize {
    std::env::var("LOGIS_GUARD_CLUSTERS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|c| c.clamp(4, 256))
        .unwrap_or(64)
}

/// Model-aware Track M active head row target (`K = 10,240` for `en`/Latin, `12,288` for `ko`/`ja`/`zh`/`ar`).
/// Enabled by default unless `LOGIS_VOCAB_TRIM=full|off`, `LOGIS_VOCAB_EXEC_MODE=artifact_direct|full`,
/// or a matching `LOGIS_VOCAB_PLAN` is provided.
pub fn compact_active_k_for_model(model_hint: Option<&str>, vocab: usize, langs: Langs) -> Option<usize> {
    if matches!(
        exec_mode_override(),
        Some(VocabExecMode::ArtifactDirect | VocabExecMode::FullUntrimmed)
    ) {
        return None;
    }
    if let Ok(v) = std::env::var(ACTIVE_K_ENV) {
        if let Ok(k) = v.trim().parse::<usize>() {
            if k >= ALIGN {
                let aligned = k.next_multiple_of(ALIGN);
                return (aligned < vocab).then_some(aligned);
            }
        }
    }
    let raw_trim = std::env::var(TRIM_ENV).unwrap_or_default();
    if matches!(raw_trim.trim().to_ascii_lowercase().as_str(), "off" | "none" | "0" | "full") {
        return None;
    }
    if load_env_vocab_plan(vocab).is_some() {
        return None;
    }
    let target = crate::models::common::reflection_gate::policy_runtime_k(model_hint, vocab, langs);
    if target >= ALIGN {
        let aligned = target.next_multiple_of(ALIGN);
        (aligned < vocab).then_some(aligned)
    } else {
        None
    }
}

/// Backward-compatible wrapper resolving compact K from vocab size + active languages.
pub fn compact_active_k(vocab: usize, langs: Langs) -> Option<usize> {
    compact_active_k_for_model(None, vocab, langs)
}

/// Unified per-step & cumulative guard/rescue telemetry (Phase 1-1 D1/D4, Phase 1-2 D2, Phase 3-1 D2, Phase 5-2 D1).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StepGuardStats {
    pub steps: usize,
    pub fast_steps: usize,
    pub constrained_steps: usize,
    pub skip_i0: bool,
    pub flagged: usize,
    pub clusters_checked: usize,
    pub rows_scored: usize,
    pub rest_wins: usize,
    pub rescue_us: u64,
    #[serde(default)]
    pub proj_kept_us: u64,
    pub quant_mode_a: bool,
    #[serde(default)]
    pub circuit_tripped: bool,
}

impl StepGuardStats {
    pub fn merge(&mut self, other: &Self) {
        self.steps += other.steps;
        self.fast_steps += other.fast_steps;
        self.constrained_steps += other.constrained_steps;
        self.skip_i0 |= other.skip_i0;
        self.flagged += other.flagged;
        self.clusters_checked += other.clusters_checked;
        self.rows_scored += other.rows_scored;
        self.rest_wins += other.rest_wins;
        self.rescue_us += other.rescue_us;
        self.proj_kept_us += other.proj_kept_us;
        self.quant_mode_a |= other.quant_mode_a;
        self.circuit_tripped |= other.circuit_tripped;
    }

    pub fn rescue_rate(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.flagged as f64 / self.steps as f64
        }
    }

    pub fn rescue_ms(&self) -> f64 {
        self.rescue_us as f64 / 1000.0
    }
}

/// Atomic cumulative counters attached to each `TrimmedHead`.
#[derive(Default)]
pub struct RescueCounters {
    pub steps: std::sync::atomic::AtomicU64,
    pub rescued: std::sync::atomic::AtomicU64,
    pub rest_wins: std::sync::atomic::AtomicU64,
    pub clusters_checked: std::sync::atomic::AtomicU64,
    pub rows_scored: std::sync::atomic::AtomicU64,
    pub rescue_us: std::sync::atomic::AtomicU64,
    pub proj_kept_us: std::sync::atomic::AtomicU64,
    pub tripped: std::sync::atomic::AtomicBool,
    pub promote_all_rows: std::sync::atomic::AtomicBool,
}

impl RescueCounters {
    pub fn record(&self, st: &StepGuardStats) {
        use std::sync::atomic::Ordering::Relaxed;
        self.steps.fetch_add(st.steps as u64, Relaxed);
        self.rescued.fetch_add(st.flagged as u64, Relaxed);
        self.rest_wins.fetch_add(st.rest_wins as u64, Relaxed);
        self.clusters_checked.fetch_add(st.clusters_checked as u64, Relaxed);
        self.rows_scored.fetch_add(st.rows_scored as u64, Relaxed);
        self.rescue_us.fetch_add(st.rescue_us, Relaxed);
        self.proj_kept_us.fetch_add(st.proj_kept_us, Relaxed);
    }

    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(std::sync::atomic::Ordering::Acquire)
    }

    #[inline]
    pub fn should_promote_all_rows(&self) -> bool {
        self.promote_all_rows.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn snapshot(&self) -> StepGuardStats {
        use std::sync::atomic::Ordering::Relaxed;
        StepGuardStats {
            steps: self.steps.load(Relaxed) as usize,
            flagged: self.rescued.load(Relaxed) as usize,
            rest_wins: self.rest_wins.load(Relaxed) as usize,
            clusters_checked: self.clusters_checked.load(Relaxed) as usize,
            rows_scored: self.rows_scored.load(Relaxed) as usize,
            rescue_us: self.rescue_us.load(Relaxed),
            proj_kept_us: self.proj_kept_us.load(Relaxed),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 토크나이저에서
// ---------------------------------------------------------------------------------------------

/// GPT-2 바이트 수준 BPE 의 글자 → 바이트 표
fn byte_decoder() -> &'static HashMap<char, u8> {
    static TABLE: Lazy<HashMap<char, u8>> = Lazy::new(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
        let mut cs = bs.clone();
        let mut n = 0u32;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap(), b as u8)).collect()
    });
    &TABLE
}

/// 토큰의 원래 바이트 (바이트 수준 BPE 를 되돌림, 표에 없는 글자가 있으면 글자 그대로)
pub fn token_bytes(tok: &tokenizers::Tokenizer, id: u32) -> Vec<u8> {
    let Some(t) = tok.id_to_token(id) else { return Vec::new() };
    let table = byte_decoder();
    let mut out = Vec::with_capacity(t.len());
    for c in t.chars() {
        match table.get(&c) {
            Some(b) => out.push(*b),
            None => return t.into_bytes(),
        }
    }
    out
}

/// 도메인 글: 내장 도메인 사전의 모든 글(키 · 값)과 JSON · HTML 표 태그
pub fn corpus() -> &'static Vec<String> {
    static CORPUS: Lazy<Vec<String>> = Lazy::new(|| {
        fn walk(v: &serde_json::Value, out: &mut Vec<String>) {
            match v {
                serde_json::Value::String(s) => out.push(s.clone()),
                serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
                serde_json::Value::Object(o) => logis_schema::json_order::sorted_iter(o).for_each(|(k, x)| {
                    out.push(k.clone());
                    walk(x, out)
                }),
                serde_json::Value::Number(n) => out.push(n.to_string()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(&crate::utils::bias_schema::BIAS_DICT, &mut out);
        for t in ["<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>", "<thead>", "<tbody>", "<br>", "{\"", "\":", "\": \"", "\", \"", "\"}", "[{", "}]", "null", "true", "false"] {
            out.push(t.to_string());
        }
        out
    });
    &CORPUS
}

/// Deterministic 64-bit tokenizer fingerprint isolating distinct vocabularies of identical size (e.g., 65k 0.6B vs 4B).
pub fn tokenizer_fingerprint(tok: &tokenizers::Tokenizer) -> u64 {
    use std::hash::{Hash, Hasher};
    let n = tok.get_vocab_size(true);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    n.hash(&mut hasher);
    let head_end = n.min(64);
    let tail_start = n.saturating_sub(64);
    for id in (0..head_end)
        .chain((head_end..tail_start).step_by(32))
        .chain(tail_start..n)
    {
        id.hash(&mut hasher);
        if let Some(tok_str) = tok.id_to_token(id as u32) {
            tok_str.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// 토큰 종류 · 원본 바이트 조각. 토크나이저 지문마다 한 번만 계산합니다.
fn kinds_and_pieces(tok: &tokenizers::Tokenizer, vocab: usize) -> Arc<(Vec<Kind>, Vec<Vec<u8>>)> {
    static BASE: Lazy<Mutex<HashMap<(usize, u64), Arc<(Vec<Kind>, Vec<Vec<u8>>)>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tokenizer_fingerprint(tok));
    if let Some(b) = BASE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Arc::clone(b);
    }
    let t0 = std::time::Instant::now();
    let added: std::collections::HashSet<u32> = tok.get_added_tokens_decoder().keys().copied().collect();
    let pairs: Vec<(Kind, Vec<u8>)> = (0..vocab as u32)
        .into_par_iter()
        .map(|i| {
            let bytes = token_bytes(tok, i);
            let k = kind_of(&bytes, added.contains(&i));
            (k, bytes)
        })
        .collect();
    let mut kinds = Vec::with_capacity(vocab);
    let mut pieces = Vec::with_capacity(vocab);
    for (k, p) in pairs {
        kinds.push(k);
        pieces.push(p);
    }
    println!("[VOCAB-TRIM] 어휘 {vocab}: 토큰 종류 분류 · {:.2} s (토크나이저마다 한 번)", t0.elapsed().as_secs_f64());
    let b = Arc::new((kinds, pieces));
    BASE.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&b));
    b
}

/// 언어 묶음별 도메인 글 토큰 집합 (번들 격리, 캐시).
fn domain_tokens_for(tok: &tokenizers::Tokenizer, vocab: usize, langs: Langs) -> Arc<BTreeSet<u32>> {
    static DOM: Lazy<Mutex<HashMap<(usize, u64, String), Arc<BTreeSet<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tokenizer_fingerprint(tok), langs.label());
    if let Some(d) = DOM.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Arc::clone(d);
    }
    let texts = corpus_for_langs(langs);
    let domain: BTreeSet<u32> = texts
        .par_iter()
        .filter_map(|text| tok.encode(text.as_str(), false).ok().map(|e| e.get_ids().to_vec()))
        .flatten_iter()
        .filter(|&id| (id as usize) < vocab)
        .collect::<Vec<u32>>()
        .into_iter()
        .collect();
    let d = Arc::new(domain);
    DOM.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&d));
    d
}

/// 5-variant structural ASCII core tokens (`w`, `" w"`, `"\"w\":"`, `"_w"`, `"{\"w\":"`), cached once per tokenizer.
pub fn ascii_core_for(tok: &tokenizers::Tokenizer, vocab: usize) -> Arc<BTreeSet<u32>> {
    static ASCII_CORE: Lazy<Mutex<HashMap<(usize, u64), Arc<BTreeSet<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tokenizer_fingerprint(tok));
    if let Some(c) = ASCII_CORE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Arc::clone(c);
    }
    let mut set = BTreeSet::new();
    for w in premium_discount_quant::mine::default_ascii_struct_corpus() {
        for variant in [
            w.clone(),
            format!(" {w}"),
            format!("\"{w}\":"),
            format!("_{w}"),
            format!("{{\"{w}\":"),
        ] {
            if let Ok(enc) = tok.encode(variant.as_str(), false) {
                for &id in enc.get_ids() {
                    if (id as usize) < vocab {
                        set.insert(id);
                    }
                }
            }
        }
    }
    let arc = Arc::new(set);
    ASCII_CORE.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&arc));
    arc
}

/// Expand prompt tokens with 1-hop spacing/punctuation variants (e.g., `"foo"` <-> `" foo"`) via O(1) vocab lookup.
pub fn expand_prompt_context_ids(
    tok: &tokenizers::Tokenizer,
    pieces: &[Vec<u8>],
    base_ids: &[u32],
    vocab: usize,
) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let mut visited = std::collections::HashSet::with_capacity(base_ids.len().min(512));
    for &id in base_ids {
        if (id as usize) >= vocab || !visited.insert(id) {
            continue;
        }
        out.insert(id);
        if let Some(raw_tok) = tok.id_to_token(id) {
            if let Some(stripped) = raw_tok.strip_prefix('\u{0120}') {
                if let Some(alt) = tok.token_to_id(stripped) {
                    if (alt as usize) < vocab {
                        out.insert(alt);
                    }
                }
            } else {
                let with_g = format!("\u{0120}{raw_tok}");
                if let Some(alt) = tok.token_to_id(&with_g) {
                    if (alt as usize) < vocab {
                        out.insert(alt);
                    }
                }
            }
            if let Some(stripped) = raw_tok.strip_prefix('\u{2581}') {
                if let Some(alt) = tok.token_to_id(stripped) {
                    if (alt as usize) < vocab {
                        out.insert(alt);
                    }
                }
            }
        }
        if let Some(bytes) = pieces.get(id as usize) {
            if let Ok(s) = std::str::from_utf8(bytes) {
                let trimmed = s.trim();
                if !trimmed.is_empty() && trimmed.len() <= 24 {
                    if let Some(alt) = tok.token_to_id(trimmed) {
                        if (alt as usize) < vocab {
                            out.insert(alt);
                        }
                    }
                    let spaced_g = format!("\u{0120}{trimmed}");
                    if let Some(alt) = tok.token_to_id(&spaced_g) {
                        if (alt as usize) < vocab {
                            out.insert(alt);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Optional pre-mined `VocabPlan` loaded from `LOGIS_VOCAB_PLAN`.
pub fn load_env_vocab_plan(vocab: usize) -> Option<Arc<Vec<u32>>> {
    static PLAN_CACHE: Lazy<Mutex<HashMap<(String, usize), Option<Arc<Vec<u32>>>>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    let path = std::env::var(PLAN_ENV).ok()?;
    let path = path.trim().to_string();
    if path.is_empty() {
        return None;
    }
    let key = (path.clone(), vocab);
    if let Some(cached) = PLAN_CACHE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return cached.clone();
    }
    let loaded = match premium_discount_quant::vocab_plan::VocabPlan::load(std::path::Path::new(&path)) {
        Ok(plan) => {
            if let Err(e) = plan.verify() {
                println!("[VOCAB-TRIM] ⚠️ {PLAN_ENV}={path} 검증 실패: {e}");
                None
            } else if plan.orig_vocab_size == vocab && plan.kept_ids.iter().all(|&id| (id as usize) < vocab) {
                println!("[VOCAB-TRIM] {PLAN_ENV} 로드 완료: {} -> {} 행 (bundle={})", vocab, plan.kept_ids.len(), plan.bundle);
                Some(Arc::new(plan.kept_ids))
            } else {
                println!(
                    "[VOCAB-TRIM] {PLAN_ENV}={path} 어휘 크기 불일치 (plan={}, model={}) → 무시",
                    plan.orig_vocab_size, vocab
                );
                None
            }
        }
        Err(e) => {
            println!("[VOCAB-TRIM] ⚠️ {PLAN_ENV}={path} 읽기 실패: {e}");
            None
        }
    };
    PLAN_CACHE.lock().unwrap_or_else(|p| p.into_inner()).insert(key, loaded.clone());
    loaded
}

/// Optional explicit execution mode override (`LOGIS_VOCAB_EXEC_MODE=artifact_direct|runtime_sliced|auto`).
pub fn exec_mode_override() -> Option<VocabExecMode> {
    match std::env::var("LOGIS_VOCAB_EXEC_MODE")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "artifact_direct" | "direct" | "artifact" => Some(VocabExecMode::ArtifactDirect),
        "runtime_sliced" | "sliced" => Some(VocabExecMode::RuntimeSliced),
        "full" | "full_untrimmed" => Some(VocabExecMode::FullUntrimmed),
        _ => None,
    }
}

/// 토크나이저 · 규칙 · 언어로 남길 행 (오름차순). 어휘 크기 · 규칙 · 언어마다 한 번 계산하고 기억합니다.
pub fn ids_for(tok: &tokenizers::Tokenizer, vocab: usize, rule: TrimRule, langs: Langs) -> Result<Arc<Vec<u32>>> {
    ids_for_with_prompt(tok, vocab, rule, langs, "")
}

/// [`ids_for`] + 컴팩트 활성 K / `LOGIS_VOCAB_PLAN` / 프롬프트 토큰 합집합 지원.
pub fn ids_for_with_prompt(
    tok: &tokenizers::Tokenizer,
    vocab: usize,
    rule: TrimRule,
    langs: Langs,
    prompt: &str,
) -> Result<Arc<Vec<u32>>> {
    ids_for_with_prompt_ids(tok, vocab, rule, langs, prompt, &[], None, current_doc_lang_hint().as_deref())
}

/// Track M selection using optional pre-encoded prompt token IDs, model hint, and document language hint.
#[allow(clippy::too_many_arguments)]
pub fn ids_for_with_prompt_ids(
    tok: &tokenizers::Tokenizer,
    vocab: usize,
    rule: TrimRule,
    langs: Langs,
    prompt_text: &str,
    prompt_ids_slice: &[u32],
    model_hint: Option<&str>,
    doc_lang: Option<&str>,
) -> Result<Arc<Vec<u32>>> {
    let compact_k = compact_active_k_for_model(model_hint, vocab, langs);
    if compact_k.is_none() {
        if let Some(plan_ids) = load_env_vocab_plan(vocab) {
            return Ok(plan_ids);
        }
    }
    let base = kinds_and_pieces(tok, vocab);
    let domain = domain_tokens_for(tok, vocab, langs);

    if let Some(target_k) = compact_k {
        let prompt_ids = if !prompt_ids_slice.is_empty() {
            expand_prompt_context_ids(tok, &base.1, prompt_ids_slice, vocab)
        } else if !prompt_text.is_empty() {
            let encoded = tok
                .encode(prompt_text, false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            expand_prompt_context_ids(tok, &base.1, &encoded, vocab)
        } else {
            BTreeSet::new()
        };
        let ascii_core = ascii_core_for(tok, vocab);
        let t0 = std::time::Instant::now();
        let (ids, tiers) = select_compact_track_m(
            &base.0,
            &base.1,
            &domain,
            &prompt_ids,
            &ascii_core,
            langs,
            doc_lang,
            target_k,
            None,
        );
        let t: Vec<String> = tiers.iter().map(|(n, k)| format!("{n} {k}")).collect();
        println!(
            "[VOCAB-TRIM] track_m(k<={target_k},doc={}) · {}: 어휘 {vocab} → {} 행 ({}) · {:.2} s",
            doc_lang.unwrap_or("none"),
            langs.label(),
            ids.len(),
            t.join(" · "),
            t0.elapsed().as_secs_f64()
        );
        return Ok(Arc::new(ids));
    }

    static CHOSEN: Lazy<Mutex<HashMap<(usize, u64, String), Arc<Vec<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tokenizer_fingerprint(tok), format!("{}:{}", rule.label(), langs.label()));
    if let Some(ids) = CHOSEN.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(Arc::clone(ids));
    }
    let t0 = std::time::Instant::now();
    let (ids, tiers) = select_from(&base.0, &domain, rule, langs);
    let t: Vec<String> = tiers.iter().map(|(n, k)| format!("{n} {k}")).collect();
    println!("[VOCAB-TRIM] {} · {}: 어휘 {vocab} → {} 행 ({}) · {:.2} s", rule.label(), langs.label(), ids.len(), t.join(" · "), t0.elapsed().as_secs_f64());
    let ids = Arc::new(ids);
    CHOSEN.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&ids));
    Ok(ids)
}

// ---------------------------------------------------------------------------------------------
// 문서 언어 (실행 흐름마다)
// ---------------------------------------------------------------------------------------------
//
// 예전에는 프로세스 전체에 값 하나(`DOC_LANGS`)를 두고 범위를 열 때 덮어쓰고 닫을 때 되돌렸습니다. 검색(질의 언어 + 앱 언어)과
// 스케줄러의 문서 추출(그 문서의 언어)이 동시에 돌면 나중에 연 쪽이 값을 덮어, 다른 쪽의 생성이 남의 언어로 잘린 행을 썼습니다
// (예: 독일어 문서 추출이 범위를 연 사이 한국어 앱의 검색 답변 → 한글 행이 빠짐). 닫는 순서가 엇갈리면 이미 끝난 범위의
// 값이 되살아나기도 했습니다. 이제 범위는 연 실행 흐름(tokio 작업, 작업 밖이면 스레드)에 묶입니다:
// - 생성은 자기 흐름이 연 가장 안쪽 범위를 씁니다 (한 흐름 안에서 겹쳐 연 범위는 예전처럼 안쪽이 이김).
// - 자기 흐름이 연 범위가 없으면(다른 작업으로 넘겨진 생성) 열린 모든 흐름의 가장 안쪽 범위의 합집합을 씁니다. 행이 더 많을 뿐
//   어느 흐름의 글도 잘리지 않습니다 (off 에 더 가깝기만 함). 열린 범위가 없으면 아무 문자 체계도 빼지 않습니다.

/// 범위를 연 실행 흐름
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Task(tokio::task::Id),
    Thread(std::thread::ThreadId),
}

fn flow() -> Flow {
    match tokio::task::try_id() {
        Some(id) => Flow::Task(id),
        None => Flow::Thread(std::thread::current().id()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDocLang {
    pub doc_lang: String,
    pub langs: Langs,
}

/// Lock-free language context propagated across `tokio::spawn` and `spawn_blocking` boundaries.
#[derive(Clone, Debug)]
pub struct LangContext {
    pub scope_id: u64,
    pub doc_lang: String,
    pub ui_lang: String,
    pub langs: Langs,
    pub pending: bool,
    shared_pending: Option<Arc<std::sync::OnceLock<ResolvedDocLang>>>,
}

impl Default for LangContext {
    fn default() -> Self {
        Self {
            scope_id: 0,
            doc_lang: String::new(),
            ui_lang: String::new(),
            langs: Langs::any(),
            pending: false,
            shared_pending: None,
        }
    }
}

impl LangContext {
    #[inline]
    pub fn resolve_effective(&self) -> Self {
        if self.pending {
            if let Some(cell) = &self.shared_pending {
                if let Some(resolved) = cell.get() {
                    let mut next = self.clone();
                    next.doc_lang = resolved.doc_lang.clone();
                    next.langs = if !next.ui_lang.is_empty() && !matches!(next.ui_lang.as_str(), "en" | "pending" | "unscoped") {
                        resolved.langs.union(Langs::of_doc_lang(&next.ui_lang))
                    } else {
                        resolved.langs
                    };
                    if next.ui_lang.is_empty() || next.ui_lang == "pending" || next.ui_lang == "unscoped" {
                        next.ui_lang = resolved.doc_lang.clone();
                    }
                    next.pending = false;
                    return next;
                }
            }
        }
        self.clone()
    }

    pub fn capture_current() -> Self {
        current_lang_context()
    }

    pub fn capture_or_from_doc(doc_lang: &str) -> Self {
        let cur = current_lang_context();
        if cur.scope_id > 0 && !matches!(cur.doc_lang.as_str(), "" | "unscoped") {
            return cur;
        }
        let norm = normalize_lang_tag(doc_lang).unwrap_or_else(|| "en".to_string());
        let langs = Langs::of_doc_lang(&norm);
        Self {
            scope_id: NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            doc_lang: norm.clone(),
            ui_lang: norm,
            langs,
            pending: false,
            shared_pending: None,
        }
    }

    pub fn attach(self) -> BoundLangGuard {
        THREAD_BOUND_CTX.with(|st| st.borrow_mut().push(self));
        BoundLangGuard
    }

    pub async fn scope_future<F: std::future::Future>(self, fut: F) -> F::Output {
        TASK_BOUND_CTX.scope(std::cell::RefCell::new(self), fut).await
    }
}

tokio::task_local! {
    static TASK_BOUND_CTX: std::cell::RefCell<LangContext>;
}

thread_local! {
    static THREAD_BOUND_CTX: std::cell::RefCell<Vec<LangContext>> = const { std::cell::RefCell::new(Vec::new()) };
    static LOCAL_THREAD_SCOPES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub struct BoundLangGuard;

impl Drop for BoundLangGuard {
    fn drop(&mut self) {
        THREAD_BOUND_CTX.with(|st| {
            st.borrow_mut().pop();
        });
    }
}

/// 열린 범위 하나 (연 순서대로 쌓임)
struct ScopeEntry {
    id: u64,
    flow: Flow,
    doc_lang: String,
    ui_lang: String,
    langs: Langs,
    /// 이미지 문서가 아직 언어를 정하지 않았는가 ([`DocLangScope::image`] → [`observe_read`])
    pending: bool,
    shared_pending: Option<Arc<std::sync::OnceLock<ResolvedDocLang>>>,
}

static SCOPES: Lazy<Mutex<Vec<ScopeEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn scopes() -> std::sync::MutexGuard<'static, Vec<ScopeEntry>> {
    SCOPES.lock().unwrap_or_else(|p| p.into_inner())
}

fn scope_entry_to_ctx(e: &ScopeEntry) -> LangContext {
    let base = LangContext {
        scope_id: e.id,
        doc_lang: e.doc_lang.clone(),
        ui_lang: if e.ui_lang.is_empty() { e.doc_lang.clone() } else { e.ui_lang.clone() },
        langs: e.langs,
        pending: e.pending,
        shared_pending: e.shared_pending.clone(),
    };
    base.resolve_effective()
}

fn union_scopes_fallback(g: &[ScopeEntry]) -> LangContext {
    let mut seen: Vec<Flow> = Vec::new();
    let mut acc: Option<Langs> = None;
    for e in g.iter().rev() {
        if !seen.contains(&e.flow) {
            seen.push(e.flow);
            let eff = scope_entry_to_ctx(e);
            acc = Some(acc.map_or(eff.langs, |a| a.union(eff.langs)));
        }
    }
    LangContext {
        scope_id: 0,
        doc_lang: "unscoped".to_string(),
        ui_lang: "unscoped".to_string(),
        langs: acc.unwrap_or_else(Langs::any),
        pending: false,
        shared_pending: None,
    }
}

/// Resolve current language context without locking `SCOPES` when bound on a worker without local `DocLangScope`.
pub fn current_lang_context() -> LangContext {
    let thread_top = THREAD_BOUND_CTX.with(|st| st.borrow().last().map(LangContext::resolve_effective));
    let task_bound = TASK_BOUND_CTX
        .try_with(|c| c.borrow().resolve_effective())
        .ok();
    let best_bound = match (thread_top, task_bound) {
        (Some(t), Some(a)) => Some(if t.scope_id >= a.scope_id { t } else { a }),
        (Some(t), None) => Some(t),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    let me = flow();
    let need_scopes_check = matches!(me, Flow::Task(_))
        || best_bound.is_none()
        || LOCAL_THREAD_SCOPES.with(|c| c.get() > 0);
    if !need_scopes_check {
        if let Some(b) = best_bound {
            return b;
        }
    }
    let g = scopes();
    let local_scope = g.iter().rev().find(|e| e.flow == me).map(scope_entry_to_ctx);
    match (best_bound, local_scope) {
        (Some(b), Some(l)) => {
            if l.scope_id >= b.scope_id {
                l
            } else {
                b
            }
        }
        (Some(b), None) => b,
        (None, Some(l)) => l,
        (None, None) => union_scopes_fallback(&g),
    }
}

/// Return current concrete document language tag (`Some("ko"|"zh-tw"|"fr"|...)`) when resolved.
pub fn current_doc_lang_hint() -> Option<String> {
    let ctx = current_lang_context();
    if matches!(ctx.doc_lang.as_str(), "" | "pending" | "unscoped" | "all" | "any") {
        None
    } else {
        Some(ctx.doc_lang)
    }
}

///지금 흐름의 문서 언어 (위 규칙)
fn scoped_langs() -> Langs {
    current_lang_context().langs
}

/// Convert engine code (e.g. `"kor"`, `"jpn"`, `"zh-tw"`) to `Langs` using canonical tag normalization.
pub fn langs_for_engine_code(code: &str) -> Langs {
    let norm = normalize_lang_tag(code).unwrap_or_else(|| "en".to_string());
    Langs::of_doc_lang(&norm)
}

/// 문서 하나의 언어 범위. 연 실행 흐름에 묶이고 닫힐 때(드롭) 사라집니다. 문서 처리 함수의 맨 위에 두면
/// 그 문서가 끝난 뒤의 생성(검색 질의 등)은 다시 바깥 범위(없으면 아무 문자 체계도 빼지 않음)로 돌아갑니다.
pub struct DocLangScope {
    id: u64,
    is_thread_flow: bool,
    shared_pending: Option<Arc<std::sync::OnceLock<ResolvedDocLang>>>,
}

impl DocLangScope {
    /// 텍스트 맥락: 생성 전에 판정한 문서 언어(`detect_document_language` 의 현지 언어 + 영어)와 판정에 쓴 글에 나온 문자 체계
    pub fn text(doc_lang: &str, text: &str) -> Self {
        let norm = normalize_lang_tag(doc_lang).unwrap_or_else(|| "en".to_string());
        let langs = Langs::of_doc_lang(&norm).union(Langs::present(text));
        Self::open_full(norm.clone(), norm, langs, false, None)
    }

    /// 이미지 맥락: 언어를 미리 모르므로 아무 문자 체계도 빼지 않고 시작하고, 판독 글이 문자 체계를 가리키면 [`observe_read`] 가 정합니다.
    pub fn image() -> Self {
        Self::open_full(
            "pending".to_string(),
            "en".to_string(),
            Langs::any(),
            true,
            Some(Arc::new(std::sync::OnceLock::new())),
        )
    }

    fn open_full(
        doc_lang: String,
        ui_lang: String,
        langs: Langs,
        pending: bool,
        shared_pending: Option<Arc<std::sync::OnceLock<ResolvedDocLang>>>,
    ) -> Self {
        let id = NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let me = flow();
        let is_thread_flow = matches!(me, Flow::Thread(_));
        if is_thread_flow {
            LOCAL_THREAD_SCOPES.with(|c| c.set(c.get().saturating_add(1)));
        }
        scopes().push(ScopeEntry {
            id,
            flow: me,
            doc_lang,
            ui_lang,
            langs,
            pending,
            shared_pending: shared_pending.clone(),
        });
        Self { id, is_thread_flow, shared_pending }
    }

    fn update(&self, f: impl FnOnce(&mut ScopeEntry)) {
        if let Some(e) = scopes().iter_mut().find(|e| e.id == self.id) {
            f(e);
        }
    }

    /// 같은 문서에서 언어를 다시 판정했을 때 (노이즈 제거 뒤 재확정)
    pub fn set(&self, doc_lang: &str, text: &str) {
        let norm = normalize_lang_tag(doc_lang).unwrap_or_else(|| "en".to_string());
        let langs = Langs::of_doc_lang(&norm).union(Langs::present(text));
        self.update(|e| {
            e.doc_lang = norm.clone();
            if e.ui_lang.is_empty() || e.ui_lang == "pending" {
                e.ui_lang = norm.clone();
            }
            e.langs = langs;
            e.pending = false;
        });
    }

    /// 앱이 출력하라고 지시하는 언어를 더합니다 (검색 답변의 "Write in {LANG}" = 앱 언어 등). 언어 이름(`korean`)도 코드(`ko`)도 받습니다.
    pub fn also(&self, lang: &str) {
        let mapped = crate::utils::bias_schema::lang_code_of(lang);
        let norm = normalize_lang_tag(&mapped)
            .or_else(|| normalize_lang_tag(lang))
            .unwrap_or_else(|| "en".to_string());
        let more = Langs::of_doc_lang(&norm);
        self.update(|e| {
            e.ui_lang = norm.clone();
            e.langs = e.langs.union(more);
        });
    }

    /// If an image scope stayed ASCII-only across all crops, finalize it to `fallback_doc_lang` (typically `"en"`).
    pub fn finalize_pending_if_ascii(&self, fallback_doc_lang: &str) -> Langs {
        let norm = normalize_lang_tag(fallback_doc_lang).unwrap_or_else(|| "en".to_string());
        let fallback_langs = Langs::of_doc_lang(&norm);
        if let Some(cell) = &self.shared_pending {
            let _ = cell.set(ResolvedDocLang {
                doc_lang: norm.clone(),
                langs: fallback_langs,
            });
        }
        let mut final_langs = fallback_langs;
        self.update(|e| {
            if e.pending {
                e.doc_lang = norm.clone();
                if e.ui_lang.is_empty() || e.ui_lang == "pending" {
                    e.ui_lang = norm.clone();
                }
                e.langs = fallback_langs;
                e.pending = false;
            }
            final_langs = e.langs;
        });
        final_langs
    }
}

/// Extract byte ranges of actual document payload sections (`[SOURCE DATA]`, `[TARGET DATA]`, `[SOURCE]: ...`)
/// while ignoring inline mentions inside instruction sentences.
pub fn extract_data_payload_ranges(user_text: &str) -> Vec<(usize, usize)> {
    const DATA_HEADERS: &[&str] = &[
        "[TARGET DATA]",
        "[SOURCE DATA]",
        "[PUG CONTENT]",
        "[TEXT TO ANALYZE]",
        "[CANDIDATES]",
        "[SOURCE]",
    ];
    const ALL_SECTION_HEADERS: &[&str] = &[
        "[TASK]",
        "[CONTEXT]",
        "[RULES]",
        "[WRITING RULES]",
        "[SCHEMA DEFINITIONS]",
        "[OUTPUT FORMAT]",
        "[OUTPUT]",
        "[ACTION]",
        "[TARGET DATA]",
        "[SOURCE DATA]",
        "[PUG CONTENT]",
        "[TEXT TO ANALYZE]",
        "[CANDIDATES]",
        "[SOURCE]",
    ];
    fn match_section_header(line: &str, headers: &[&'static str]) -> Option<(&'static str, usize)> {
        let lead_ws = line.len() - line.trim_start().len();
        let trimmed = &line[lead_ws..];
        for &hdr in headers {
            if let Some(rest) = trimmed.strip_prefix(hdr) {
                let rest_trim = rest.trim_start_matches([' ', '\t']);
                if rest_trim.is_empty() || rest_trim == "\r" || rest_trim == "\n" || rest_trim == "\r\n" {
                    return Some((hdr, lead_ws + hdr.len()));
                }
                if let Some(after_colon) = rest_trim.strip_prefix(':') {
                    let colon_offset = line.len() - after_colon.len();
                    return Some((hdr, colon_offset));
                }
            }
        }
        None
    }
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut offset = 0usize;
    for raw_line in user_text.split_inclusive('\n') {
        lines.push((offset, raw_line));
        offset += raw_line.len();
    }
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < lines.len() {
        let (line_start, raw_line) = lines[idx];
        if let Some((_hdr, inline_body_rel)) = match_section_header(raw_line, DATA_HEADERS) {
            let body_start = line_start + inline_body_rel;
            let mut end_idx = idx + 1;
            while end_idx < lines.len() {
                let (_, next_raw) = lines[end_idx];
                if match_section_header(next_raw, ALL_SECTION_HEADERS).is_some() {
                    break;
                }
                end_idx += 1;
            }
            let body_end = lines.get(end_idx).map_or(user_text.len(), |(s, _)| *s);
            if body_end > body_start && !user_text[body_start..body_end].trim().is_empty() {
                ranges.push((body_start, body_end));
            }
            idx = end_idx;
        } else {
            idx += 1;
        }
    }
    ranges
}

pub fn extract_data_payload_only(user_text: &str) -> String {
    let ranges = extract_data_payload_ranges(user_text);
    if ranges.is_empty() {
        return user_text.trim().to_string();
    }
    let mut out = String::new();
    for (s, e) in ranges {
        let chunk = user_text[s..e].trim();
        if !chunk.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(chunk);
        }
    }
    out
}

/// 언어 묶음별로 격리된 도메인 글 (`en` 문서에 비ASCII 6,050행이 섞이지 않도록 분리, Phase 2-1 D1).
pub fn corpus_for_langs(langs: Langs) -> Vec<String> {
    fn script_allowed(s: &str, langs: Langs) -> bool {
        if langs.is_all() || langs == Langs::any() || langs.other {
            return true;
        }
        let sp = Langs::present(s);
        (!sp.hangul || langs.hangul)
            && (!sp.kana || langs.kana)
            && (!sp.cjk || langs.cjk || langs.hanja)
            && (!sp.arabic || langs.arabic)
            && (!sp.other || langs.other)
    }
    fn walk(v: &serde_json::Value, langs: Langs, out: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => {
                if script_allowed(s, langs) {
                    out.push(s.clone());
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, langs, out)),
            serde_json::Value::Object(o) => logis_schema::json_order::sorted_iter(o).for_each(|(k, x)| {
                if script_allowed(k, langs) {
                    out.push(k.clone());
                }
                walk(x, langs, out);
            }),
            serde_json::Value::Number(n) => out.push(n.to_string()),
            _ => {}
        }
    }
    let mut out = Vec::new();
    let dict = &crate::utils::bias_schema::BIAS_DICT;
    if let Some(root) = dict.as_object() {
        let mut include_keys: Vec<&str> = vec!["en"];
        if langs.hangul { include_keys.push("ko"); }
        if langs.kana { include_keys.push("ja"); }
        if langs.cjk { include_keys.push("zh"); include_keys.push("ja"); }
        if langs.arabic { include_keys.push("ar"); }
        if langs.latin {
            include_keys.extend(["de", "fr", "es", "it", "pt", "nl", "cs"]);
        }
        if langs.other || langs.is_all() || langs == Langs::any() {
            for (k, v) in logis_schema::json_order::sorted_iter(root) {
                out.push(k.clone());
                walk(v, langs, &mut out);
            }
        } else {
            for (k, v) in logis_schema::json_order::sorted_iter(root) {
                // Keep shared structural sections (length > 3) + requested language sections only
                let is_lang_section = k.len() <= 3 || k.contains('-');
                if !is_lang_section || include_keys.contains(&k.as_str()) {
                    walk(v, langs, &mut out);
                }
            }
        }
    }
    for t in [
        "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>",
        "<thead>", "<tbody>", "<br>", "{\"", "\":", "\": \"", "\", \"", "\"}", "[{", "}]",
        "null", "true", "false",
    ] {
        out.push(t.to_string());
    }
    out
}

/// 생성 요청의 사용자·도구 메시지 글만 모읍니다 (시스템 프롬프트 지시어가 문자 체계를 팽창시키지 않게 제외, Phase 3-1 D5).
pub fn user_prompt_text(mes: &crate::openai_types::ChatCompletionParameters) -> String {
    use crate::openai_types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPart,
        ChatCompletionRequestUserMessageContent,
    };
    let mut out = String::new();
    for msg in &mes.messages {
        match msg {
            ChatCompletionRequestMessage::System(_) => {}
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    out.push_str(t);
                    out.push('\n');
                }
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    for p in parts {
                        if let ChatCompletionRequestMessageContentPart::Text(tp) = p {
                            out.push_str(&tp.text);
                            out.push('\n');
                        }
                    }
                }
            },
            ChatCompletionRequestMessage::Assistant(a) => {
                if let Some(t) = &a.content {
                    out.push_str(t);
                    out.push('\n');
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                out.push_str(&t.content);
                out.push('\n');
            }
            ChatCompletionRequestMessage::Function(_) => {}
        }
    }
    out
}

/// 생성 요청의 메시지 글 (호환 별칭 — 시스템 프롬프트 제외 본문 반환)
pub fn prompt_text(mes: &crate::openai_types::ChatCompletionParameters) -> String {
    user_prompt_text(mes)
}

impl Drop for DocLangScope {
    fn drop(&mut self) {
        if self.is_thread_flow {
            LOCAL_THREAD_SCOPES.with(|c| c.set(c.get().saturating_sub(1)));
        }
        scopes().retain(|e| e.id != self.id);
    }
}

/// 판독 글에서 값만 (JSON 이면 문자열 값만 모읍니다: 키 이름은 영어라 언어 판정을 흐립니다)
fn read_values(text: &str) -> String {
    fn walk(v: &serde_json::Value, out: &mut String) {
        match v {
            serde_json::Value::String(s) => {
                out.push_str(s);
                out.push('\n');
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            serde_json::Value::Object(m) => logis_schema::json_order::sorted_values(m).for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let t = text.trim();
    let body = t.find(|c| c == '{' || c == '[').map_or(t, |s| &t[s..]);
    let body = body.rfind(|c| c == '}' || c == ']').map_or(body, |e| &body[..=e]);
    match serde_json::from_str::<serde_json::Value>(body).map(logis_schema::json_order::sorted) {
        Ok(v) => {
            let mut out = String::new();
            walk(&v, &mut out);
            out
        }
        Err(_) => text.to_string(),
    }
}

/// 이미지 문서의 판독 글 하나를 봅니다. 아직 언어를 정하지 않았고, 비-ASCII 문자 체계가 2자 이상 나오면
/// `detect_document_language` 로 문서 언어(현지 언어 + 영어)를 정해 `shared_pending` 및 범위 상태를 확정합니다.
pub fn observe_read(text: &str) -> Option<Langs> {
    let values = read_values(text);
    let sp = Langs::present(&values);
    if !(sp.hangul || sp.kana || sp.cjk || sp.arabic || sp.latin || sp.other) {
        return None;
    }
    let decisive = values.chars().filter(|&c| !c.is_ascii() && c.is_alphabetic()).count();
    if decisive < 2 {
        return None;
    }
    let detected = crate::utils::lang_utils::detect_document_language(&values);
    let norm = normalize_lang_tag(&detected).unwrap_or_else(|| "en".to_string());
    let langs = langs_of_document(&values);

    // Check lock-free bound pending handle first when running in spawned crop workers
    let bound = current_lang_context();
    if bound.pending {
        if let Some(cell) = &bound.shared_pending {
            if cell.set(ResolvedDocLang { doc_lang: norm.clone(), langs }).is_err() {
                return None;
            }
            let mut g = scopes();
            if let Some(e) = g.iter_mut().find(|e| e.id == bound.scope_id && e.pending) {
                e.doc_lang = norm.clone();
                if e.ui_lang.is_empty() || e.ui_lang == "pending" {
                    e.ui_lang = norm.clone();
                }
                e.langs = langs;
                e.pending = false;
            }
            return Some(langs);
        }
    }

    let mut g = scopes();
    let me = flow();
    let at = match g.iter().rposition(|e| e.flow == me && e.pending) {
        Some(i) => i,
        None => {
            let pending: Vec<usize> = g.iter().enumerate().filter(|(_, e)| e.pending).map(|(i, _)| i).collect();
            match pending.as_slice() {
                [i] if !g.iter().any(|e| e.flow == me) => *i,
                _ => return None,
            }
        }
    };
    if let Some(cell) = &g[at].shared_pending {
        let _ = cell.set(ResolvedDocLang { doc_lang: norm.clone(), langs });
    }
    g[at].doc_lang = norm.clone();
    if g[at].ui_lang.is_empty() || g[at].ui_lang == "pending" {
        g[at].ui_lang = norm;
    }
    g[at].langs = langs;
    g[at].pending = false;
    Some(langs)
}

/// 지금 쓸 언어: `LOGIS_VOCAB_LANGS` 가 `all`(12 개 언어) · `any` · 코드면 그것, 아니면(`auto`, 기본) 지금 흐름의 문서 언어
pub fn current_langs() -> Langs {
    match std::env::var(LANGS_ENV) {
        Ok(v) if !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto") => {
            Langs::from_codes(&v.split(',').collect::<Vec<_>>()).unwrap_or_else(Langs::any)
        }
        _ => scoped_langs(),
    }
}

// ---------------------------------------------------------------------------------------------
// 남긴 행만의 출력 투영
// ---------------------------------------------------------------------------------------------

/// 출력 투영 표(묶인 `token_embd.weight`)의 저장된 바이트를 빌려 줄 곳: 매핑된 GGUF 와 텐서 위치
#[derive(Clone)]
pub struct TableSource {
    pub map: Arc<memmap2::Mmap>,
    pub start: usize,
    pub rows: usize,
    pub cols: usize,
    pub dtype: GgmlDType,
}

impl TableSource {
    /// GGUF 머리(`content`)와 매핑에서 텐서 `name` 의 자리
    pub fn from_gguf(map: Arc<memmap2::Mmap>, content: &candle_core::quantized::gguf_file::Content, name: &str) -> Result<Self> {
        let info = content.tensor_infos.get(name).ok_or_else(|| anyhow!("GGUF 에 {name} 이 없습니다"))?;
        let (rows, cols) = info.shape.dims2()?;
        let dtype = info.ggml_dtype;
        if cols % dtype.block_size() != 0 {
            bail!("{name}: 열 {cols} 이 블록 {} 로 나뉘지 않습니다", dtype.block_size());
        }
        let start = (content.tensor_data_offset + info.offset) as usize;
        let row_bytes = cols / dtype.block_size() * dtype.type_size();
        if start + rows * row_bytes > map.len() {
            bail!("{name}: 파일이 잘렸습니다");
        }
        Ok(Self { map, start, rows, cols, dtype })
    }

    fn row_bytes(&self) -> usize {
        self.cols / self.dtype.block_size() * self.dtype.type_size()
    }

    /// `ids` 번 행만 모은 양자화 텐서 (그 순서대로)
    pub fn gather(&self, ids: &[u32], device: &Device) -> Result<candle_core::quantized::QTensor> {
        let rb = self.row_bytes();
        let mut words = vec![0u64; (ids.len() * rb).div_ceil(8)];
        let total = ids.len() * rb;
        let contiguous = ids.first().is_some_and(|&f| ids.iter().enumerate().all(|(k, &r)| r as usize == f as usize + k));
        if contiguous {
            // 이어진 행(전체 행 등)은 한 번에 복사합니다
            let f = ids[0] as usize;
            if f + ids.len() > self.rows {
                bail!("행 {} 이 표 크기 {} 를 넘습니다", f + ids.len() - 1, self.rows);
            }
            let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
            raw[..total].copy_from_slice(&self.map[self.start + f * rb..self.start + f * rb + total]);
        } else {
            let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
            for (k, &r) in ids.iter().enumerate() {
                let r = r as usize;
                if r >= self.rows {
                    bail!("행 {r} 이 표 크기 {} 를 넘습니다", self.rows);
                }
                raw[k * rb..(k + 1) * rb].copy_from_slice(&self.map[self.start + r * rb..self.start + (r + 1) * rb]);
            }
        }
        let (_, raw, _) = unsafe { words.align_to::<u8>() };
        Ok(qtensor_from_ggml(self.dtype, &raw[..total], vec![ids.len(), self.cols], device)?)
    }
}

/// Part 5 (Pack A·C) + Phase 3-1 D2: 잘린 REST 어휘의 평균 중심화 Cauchy-Schwarz(CS) 상한 가드 +
/// 다중 군집(`ColdGuard`, C=64, `.cold_guard.bin` 캐시) 호스트 mmap SIMD 구출기.
pub struct RestRescueGuard {
    src: TableSource,
    pub rest_ids: Arc<Vec<u32>>,
    pub kept_gpu: Tensor,
    pub cs_guard: candle_core::lm_head::CsGuardParams,
    pub clustered: Option<Arc<premium_discount_quant::lmhead::ColdGuard>>,
    pub cache_hit: bool,
}

impl RestRescueGuard {
    /// 환경 설정 기반 가드 생성 (기본 Off 시 0 B 할당으로 `Ok(None)` 반환).
    pub fn build(src: &TableSource, kept: &[u32], device: &Device) -> Result<Option<Self>> {
        Self::build_with_mode(src, kept, GuardMode::from_env(), device)
    }

    /// 명시적 `GuardMode` 로 `RestRescueGuard` 를 생성합니다 (`Off` 또는 `kept/vocab >= 95%` 면 즉시 `Ok(None)`).
    pub fn build_with_mode(
        src: &TableSource,
        kept: &[u32],
        guard_mode: GuardMode,
        device: &Device,
    ) -> Result<Option<Self>> {
        if guard_mode == GuardMode::Off || kept.len() >= src.rows || kept.len() * 100 >= src.rows * 95 {
            return Ok(None);
        }
        let mut is_kept = vec![false; src.rows];
        for &id in kept {
            if (id as usize) < src.rows {
                is_kept[id as usize] = true;
            }
        }
        let rest_ids: Vec<u32> = (0..src.rows as u32)
            .filter(|&id| !is_kept[id as usize])
            .collect();
        if rest_ids.is_empty() {
            return Ok(None);
        }
        let cols = src.cols;

        // Pass 1: mu_rest [cols] (f64 누산 → f32)
        let mut sum_f64 = vec![0.0f64; cols];
        let mut all_rest_vals: Vec<f32> = if guard_mode == GuardMode::Clustered && rest_ids.len() >= 64 {
            Vec::with_capacity(rest_ids.len() * cols)
        } else {
            Vec::new()
        };
        for chunk in rest_ids.chunks(512) {
            let deq = src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
            let vals = deq.flatten_all()?.to_vec1::<f32>()?;
            for row in vals.chunks_exact(cols) {
                for c in 0..cols {
                    sum_f64[c] += row[c] as f64;
                }
            }
            if guard_mode == GuardMode::Clustered && rest_ids.len() >= 64 {
                all_rest_vals.extend_from_slice(&vals);
            }
        }
        let inv_n = 1.0 / (rest_ids.len() as f64);
        let mu_vec: Vec<f32> = sum_f64.into_iter().map(|s| (s * inv_n) as f32).collect();

        // Pass 2: max_{r in REST} ||w_r - mu_rest||_2
        let mut max_norm_sq = 0.0f32;
        if !all_rest_vals.is_empty() {
            max_norm_sq = all_rest_vals
                .par_chunks_exact(cols)
                .map(|row| {
                    let mut sq = 0.0f32;
                    for c in 0..cols {
                        let diff = row[c] - mu_vec[c];
                        sq = diff.mul_add(diff, sq);
                    }
                    sq
                })
                .reduce(|| 0.0f32, f32::max);
        } else {
            for chunk in rest_ids.chunks(512) {
                let deq = src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                for row in vals.chunks_exact(cols) {
                    let mut sq = 0.0f32;
                    for c in 0..cols {
                        let diff = row[c] - mu_vec[c];
                        sq = diff.mul_add(diff, sq);
                    }
                    if sq > max_norm_sq {
                        max_norm_sq = sq;
                    }
                }
            }
        }

        let w_rest_bound = max_norm_sq.sqrt() * (1.0 + 1e-3);
        let mu_rest = Tensor::from_vec(mu_vec, (cols,), device)?;
        let kept_gpu = Tensor::from_vec(kept.to_vec(), (kept.len(),), device)?;
        let cs_guard = candle_core::lm_head::CsGuardParams {
            mu_rest,
            w_rest_bound,
            b_rest_max: 0.0,
        };

        let mut cache_hit = false;
        let clustered = if guard_mode == GuardMode::Clustered && rest_ids.len() >= 64 {
            let clusters = guard_clusters_from_env();
            let cache_path = Self::cold_guard_cache_path(src, kept, clusters);
            if let Some(cg) = cache_path
                .as_ref()
                .and_then(|p| premium_discount_quant::lmhead::ColdGuard::load_from_file(p).ok())
                .filter(|cg| cg.dim == cols && cg.members.len() == rest_ids.len())
            {
                cache_hit = true;
                Some(Arc::new(cg))
            } else {
                let cg = premium_discount_quant::lmhead::build_cold_guard(
                    &all_rest_vals,
                    &rest_ids,
                    cols,
                    clusters,
                    4,
                );
                if let Some(p) = &cache_path {
                    let _ = cg.save_to_file(p);
                }
                Some(Arc::new(cg))
            }
        } else {
            None
        };

        Ok(Some(Self {
            src: src.clone(),
            rest_ids: Arc::new(rest_ids),
            kept_gpu,
            cs_guard,
            clustered,
            cache_hit,
        }))
    }

    fn cold_guard_cache_path(src: &TableSource, kept: &[u32], clusters: usize) -> Option<std::path::PathBuf> {
        if src.rows < 4096 {
            return None;
        }
        let mut h_in = Vec::with_capacity(32 + kept.len() * 4);
        h_in.extend_from_slice(&(src.rows as u64).to_le_bytes());
        h_in.extend_from_slice(&(src.cols as u64).to_le_bytes());
        h_in.extend_from_slice(&(src.start as u64).to_le_bytes());
        h_in.extend_from_slice(&(clusters as u64).to_le_bytes());
        for &id in kept {
            h_in.extend_from_slice(&id.to_le_bytes());
        }
        let sample_end = (src.start + 64).min(src.map.len());
        h_in.extend_from_slice(&src.map[src.start..sample_end]);
        let sha = premium_discount_quant::vocab_plan::sha256_hex(&h_in);
        Some(
            logis_schema::paths::get_app_dir()
                .join("cache")
                .join("cold_guards")
                .join(format!("v{}_k{}_c{}_{}.cold_guard.bin", src.rows, kept.len(), clusters, &sha[..16])),
        )
    }

    /// Bufferless SIMD scoring for Q8_0 and F16 cold rows; lazy scratch allocation for other K-quants.
    #[inline]
    pub fn score_cold_row(
        &self,
        rid: u32,
        h_row: &[f32],
        target_norm_cpu: Option<&[f32]>,
        kquant_scratch: &mut Vec<f32>,
    ) -> Result<f32> {
        let rb = self.src.row_bytes();
        let v = match self.src.dtype {
            GgmlDType::Q8_0 => {
                let off = self.src.start + (rid as usize) * rb;
                let raw = &self.src.map[off..off + rb];
                if let Some(tn) = target_norm_cpu {
                    embed_cpu_kernels::q8::score_q8_0_prejudice_simd(raw, h_row, tn)
                } else {
                    embed_cpu_kernels::q8::dot_q8_0_f32_simd(raw, h_row)
                }
            }
            GgmlDType::F16 => {
                let off = self.src.start + (rid as usize) * rb;
                let raw = &self.src.map[off..off + rb];
                if let Some(tn) = target_norm_cpu {
                    embed_cpu_kernels::q8::score_f16_prejudice_simd(raw, h_row, tn)
                } else {
                    embed_cpu_kernels::q8::dot_f16_f32_simd(raw, h_row)
                }
            }
            _ => {
                let deq = self.src.gather(&[rid], &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                if kquant_scratch.capacity() == 0 {
                    kquant_scratch.reserve(self.src.cols);
                }
                kquant_scratch.clear();
                kquant_scratch.extend_from_slice(&vals);
                let raw = embed_cpu_kernels::dot_f32_simd(h_row, kquant_scratch);
                if let Some(tn) = target_norm_cpu {
                    raw - crate::models::common::prejudice::scalar_prejudice(kquant_scratch, tn)
                } else {
                    raw
                }
            }
        };
        Ok(v)
    }

    /// Stage-1 clustered rescue evaluation without extra cold post-processing.
    pub fn eval_rest_best_clustered(
        &self,
        h_row: &[f32],
        best_id: u32,
        best_val: f32,
        target_norm_cpu: Option<&[f32]>,
    ) -> Result<(u32, f32, usize, usize)> {
        self.eval_rest_best_clustered_with_post(h_row, best_id, best_val, target_norm_cpu, |_, raw| raw)
    }

    /// Stage-1 clustered rescue evaluation with post-processing symmetry for strict E1 constrained greedy decoding.
    pub fn eval_rest_best_clustered_with_post<F>(
        &self,
        h_row: &[f32],
        mut best_id: u32,
        mut best_val: f32,
        target_norm_cpu: Option<&[f32]>,
        post_cold: F,
    ) -> Result<(u32, f32, usize, usize)>
    where
        F: Fn(u32, f32) -> f32 + Sync,
    {
        let cols = self.src.cols;
        anyhow::ensure!(h_row.len() == cols, "RestRescueGuard: hidden len {} != cols {}", h_row.len(), cols);
        let better = |(i1, v1): (u32, f32), (i2, v2): (u32, f32)| -> (u32, f32) {
            if v2 > v1 || (v2 == v1 && i2 < i1) { (i2, v2) } else { (i1, v1) }
        };
        let Some(cg) = &self.clustered else {
            let (rid, rval) = self
                .rest_ids
                .par_chunks(512)
                .map(|chunk| -> Result<(u32, f32)> {
                    let mut bid = u32::MAX;
                    let mut bval = f32::NEG_INFINITY;
                    let mut kquant_scratch: Vec<f32> = Vec::new();
                    for &rid in chunk {
                        let raw = self.score_cold_row(rid, h_row, target_norm_cpu, &mut kquant_scratch)?;
                        let v = post_cold(rid, raw);
                        if v > bval || (v == bval && rid < bid) {
                            bval = v;
                            bid = rid;
                        }
                    }
                    Ok((bid, bval))
                })
                .try_reduce(|| (u32::MAX, f32::NEG_INFINITY), |a, b| Ok(better(a, b)))?;
            if rval > best_val || (rval == best_val && rid < best_id) {
                best_id = rid;
                best_val = rval;
            }
            return Ok((best_id, best_val, 1, self.rest_ids.len()));
        };

        let hn = embed_cpu_kernels::dot_f32_simd(h_row, h_row).sqrt();
        let c = cg.radius.len();
        let mut bounds: Vec<(f32, usize)> = (0..c)
            .map(|j| {
                let ch = embed_cpu_kernels::dot_f32_simd(&cg.centers[j * cols..(j + 1) * cols], h_row);
                let rh = cg.radius[j] * hn;
                let ub = ch + rh + 1e-3 * (ch.abs() + rh) + 1e-4;
                (ub, j)
            })
            .collect();
        bounds.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut clusters_checked = 0usize;
        let mut rows_scored = 0usize;
        for (ub, j) in bounds {
            if ub < best_val {
                break;
            }
            let members = &cg.members[cg.offsets[j] as usize..cg.offsets[j + 1] as usize];
            if members.is_empty() {
                continue;
            }
            clusters_checked += 1;
            rows_scored += members.len();
            let (cid, cval) = if matches!(self.src.dtype, GgmlDType::Q8_0 | GgmlDType::F16) && members.len() >= 128 {
                members
                    .par_chunks(256)
                    .map(|chunk| -> Result<(u32, f32)> {
                        let mut lid = u32::MAX;
                        let mut lval = f32::NEG_INFINITY;
                        let mut kquant_scratch: Vec<f32> = Vec::new();
                        for &rid in chunk {
                            let raw = self.score_cold_row(rid, h_row, target_norm_cpu, &mut kquant_scratch)?;
                            let v = post_cold(rid, raw);
                            if v > lval || (v == lval && rid < lid) {
                                lval = v;
                                lid = rid;
                            }
                        }
                        Ok((lid, lval))
                    })
                    .try_reduce(|| (u32::MAX, f32::NEG_INFINITY), |a, b| Ok(better(a, b)))?
            } else {
                let mut lid = u32::MAX;
                let mut lval = f32::NEG_INFINITY;
                let mut kquant_scratch: Vec<f32> = Vec::new();
                for &rid in members {
                    let raw = self.score_cold_row(rid, h_row, target_norm_cpu, &mut kquant_scratch)?;
                    let v = post_cold(rid, raw);
                    if v > lval || (v == lval && rid < lid) {
                        lval = v;
                        lid = rid;
                    }
                }
                (lid, lval)
            };
            if cval > best_val || (cval == best_val && cid < best_id) {
                best_val = cval;
                best_id = cid;
            }
        }
        Ok((best_id, best_val, clusters_checked, rows_scored))
    }

    /// 희귀 구출(`rescue_mask == 1`): 호스트 mmap 의 REST 행을 bufferless SIMD(`embed_cpu_kernels::q8`)로 평가해
    /// `(best_vocab_id, best_logit)` 을 반환합니다.
    pub fn eval_rest_best(&self, h_row: &[f32]) -> Result<(u32, f32)> {
        let cols = self.src.cols;
        anyhow::ensure!(h_row.len() == cols, "RestRescueGuard: hidden len {} != cols {}", h_row.len(), cols);
        let better = |(i1, v1): (u32, f32), (i2, v2): (u32, f32)| -> (u32, f32) {
            if v2 > v1 || (v2 == v1 && i2 < i1) { (i2, v2) } else { (i1, v1) }
        };
        if matches!(self.src.dtype, GgmlDType::Q8_0 | GgmlDType::F16) {
            self.rest_ids
                .par_chunks(512)
                .map(|chunk| -> Result<(u32, f32)> {
                    let mut best_id = u32::MAX;
                    let mut best_val = f32::NEG_INFINITY;
                    let mut kquant_scratch: Vec<f32> = Vec::new();
                    for &rid in chunk {
                        let v = self.score_cold_row(rid, h_row, None, &mut kquant_scratch)?;
                        if v > best_val || (v == best_val && rid < best_id) {
                            best_val = v;
                            best_id = rid;
                        }
                    }
                    Ok((best_id, best_val))
                })
                .try_reduce(|| (u32::MAX, f32::NEG_INFINITY), |a, b| Ok(better(a, b)))
        } else {
            let mut best = (u32::MAX, f32::NEG_INFINITY);
            for chunk in self.rest_ids.chunks(512) {
                let deq = self.src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                if let Some((local_idx, local_val)) =
                    embed_cpu_kernels::dot_argmax_rows_f32(h_row, &vals, chunk.len(), cols)
                {
                    if local_idx < chunk.len() {
                        best = better(best, (chunk[local_idx], local_val));
                    }
                }
            }
            Ok(best)
        }
    }
}

/// Gate 1-C: Greedy 디코드 스텝의 그림자 Top-3 후보.
#[derive(Debug, Clone, Copy)]
pub struct ShadowCandidate {
    pub token_id: u32,
    pub logit: f32,
    pub prob: f32,
}

/// Gate 1-C: Greedy 디코드 1스텝 결과 (100% 무손실 top-1 + 그림자 top-3 + top-1 확신도).
#[derive(Debug, Clone)]
pub struct GreedyShadowStep {
    pub token_id: u32,
    pub confidence: f32,
    pub top3: [Option<ShadowCandidate>; 3],
    pub rescued: bool,
}

/// 스택 할당 `SmallVec<[u8; 16]>` 기반 경량 바이트 편집 거리 (최대 허용 거리 `max_dist` 초과 시 조기 종료).
fn bounded_edit_distance(a: &[u8], b: &[u8], max_dist: usize) -> usize {
    let n = a.len();
    let m = b.len();
    if n.abs_diff(m) > max_dist {
        return max_dist + 1;
    }
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: smallvec::SmallVec<[u8; 32]> = (0..=m as u8).collect();
    let mut curr: smallvec::SmallVec<[u8; 32]> = smallvec::smallvec![0u8; m + 1];
    for (i, &ac) in a.iter().enumerate() {
        curr[0] = (i + 1) as u8;
        let mut row_min = curr[0] as usize;
        for (j, &bc) in b.iter().enumerate() {
            let cost = u8::from(ac != bc);
            let del = prev[j + 1].saturating_add(1);
            let ins = curr[j].saturating_add(1);
            let sub = prev[j].saturating_add(cost);
            let v = del.min(ins).min(sub);
            curr[j + 1] = v;
            row_min = row_min.min(v as usize);
        }
        if row_min > max_dist {
            return max_dist + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m] as usize
}

/// Gate 1-C: `SmallVec<[u8; 16]>` 무할당 슬라이스 비교로 그림자 Top-3 후보 중
/// 문서 앵커(`anchor_bytes`)와 편집 거리 1~2 이내로 일치하는 대체 토큰을 고릅니다.
///
/// 발동 조건:
/// - `step.confidence < 0.95` (고확신 스텝은 즉시 건너뜀)
/// - top-1 자체가 앵커와 정확히 일치하지 않음
/// - 후보(top-2/top-3) 확률 `>= 0.08`, 바이트 편집 거리 `1..=2`
pub fn match_shadow_anchor_candidate(
    step: &GreedyShadowStep,
    tok: &tokenizers::Tokenizer,
    anchor_bytes: &[u8],
) -> Option<u32> {
    if step.confidence >= 0.95 || anchor_bytes.is_empty() {
        return None;
    }
    let decode_small = |id: u32| -> smallvec::SmallVec<[u8; 16]> {
        let raw = token_bytes(tok, id);
        let mut sv = smallvec::SmallVec::<[u8; 16]>::with_capacity(raw.len());
        sv.extend_from_slice(&raw);
        sv
    };
    let top1_bytes = decode_small(step.token_id);
    let top1_trimmed = top1_bytes.as_slice();
    if top1_trimmed == anchor_bytes {
        return None;
    }
    let top1_dist = bounded_edit_distance(top1_trimmed, anchor_bytes, 2);
    for cand in step.top3.iter().skip(1).flatten() {
        if cand.prob < 0.08 {
            continue;
        }
        let cand_bytes = decode_small(cand.token_id);
        let dist = bounded_edit_distance(cand_bytes.as_slice(), anchor_bytes, 2);
        if dist < top1_dist && (dist == 0 || dist <= 2) {
            return Some(cand.token_id);
        }
    }
    None
}

const NET_GAIN_WARMUP_STEPS: u64 = 16;
const MAX_ALLOWED_RESCUE_RATE: f64 = 0.05;

/// Reusable scratch buffers for zero-heap-allocation single-pass decode post-processing.
#[derive(Default)]
pub struct DecodeStepScratch {
    pub local_seen: Vec<u32>,
    pub banned_ids: Vec<u32>,
}

/// Parameters for `[1, K]` single-pass constraint & penalty decoding.
pub struct StepPostProcess<'a> {
    pub seen_tokens: &'a [u32],
    pub repeat_penalty: f32,
    pub banned_ids: &'a [u32],
    pub prejudice_kept: Option<&'a Tensor>,
    pub target_norm_cpu: Option<&'a [f32]>,
    pub fallback_eos_id: u32,
}

/// 남긴 행만의 출력 투영 (GPU `inv`/`pad` 텐서 0 B).
#[derive(Clone)]
pub struct TrimmedHead {
    w: QMatMul,
    qweight: Arc<candle_core::quantized::QTensor>,
    pub kept: Arc<Vec<u32>>,
    pub kept_hash: u64,
    pub vocab: usize,
    /// 남긴 행을 고른 규칙 · 언어 (같으면 다시 만들지 않음)
    pub label: String,
    /// 호스트 역표 [vocab] (u32): 원본 토큰 ID -> 0..K-1 로컬 인덱스 (미포함 행은 `u32::MAX`)
    host_inv: Arc<Vec<u32>>,
    /// Part 5 (Pack A·C) + Phase 3-1 D2: 무손실 Greedy 디코딩을 위한 REST CS 가드 + 다중 군집 호스트 구출기.
    rescue: Option<Arc<RestRescueGuard>>,
    pub exec_mode: VocabExecMode,
    pub guard_mode: GuardMode,
    pub policy: GuardPolicyConfig,
    pub counters: Arc<RescueCounters>,
}

impl TrimmedHead {
    pub fn build(src: &TableSource, kept: Arc<Vec<u32>>, label: String, device: &Device) -> Result<Self> {
        Self::build_with_policy(src, kept, label, device, GuardPolicyConfig::from_env())
    }

    pub fn build_with_policy(
        src: &TableSource,
        mut kept: Arc<Vec<u32>>,
        mut label: String,
        device: &Device,
        mut policy: GuardPolicyConfig,
    ) -> Result<Self> {
        let t0 = std::time::Instant::now();
        if policy.strict_e1 && kept.len() < src.rows && kept.len() * 100 >= src.rows * 95 {
            kept = Arc::new((0..src.rows as u32).collect());
            label = format!("{label}:promoted_e1");
            policy.promoted_from_e1 = true;
            policy.guard_mode = GuardMode::Off;
        }
        let qweight = Arc::new(src.gather(&kept, device)?);
        let mut inv = vec![u32::MAX; src.rows];
        for (i, &id) in kept.iter().enumerate() {
            if (id as usize) < src.rows {
                inv[id as usize] = i as u32;
            }
        }
        let host_inv = Arc::new(inv);
        let kept_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            src.rows.hash(&mut h);
            kept.len().hash(&mut h);
            for &id in kept.iter() {
                id.hash(&mut h);
            }
            h.finish()
        };
        let is_trimmed = kept.len() < src.rows;
        let rescue = if is_trimmed && policy.guard_mode != GuardMode::Off {
            RestRescueGuard::build_with_mode(src, &kept, policy.guard_mode, device)?.map(Arc::new)
        } else {
            None
        };
        let exec_mode = if is_trimmed {
            VocabExecMode::RuntimeSliced
        } else if src.rows <= 65_536 {
            VocabExecMode::ArtifactDirect
        } else {
            VocabExecMode::FullUntrimmed
        };
        let mb = (kept.len() * src.row_bytes()) as f64 / 1e6;
        let guard_tag = if policy.promoted_from_e1 {
            "promoted_all_rows".to_string()
        } else {
            match &rescue {
                None => if is_trimmed { "off".to_string() } else { "full".to_string() },
                Some(r) => match &r.clustered {
                    Some(cg) => format!("clustered(c={},cache={})", cg.radius.len(), if r.cache_hit { "hit" } else { "build" }),
                    None => "single".to_string(),
                },
            }
        };
        println!(
            "[VOCAB-TRIM] HEAD v_art={} k={} mb={mb:.1} scope={label} mode={} guard={guard_tag} build_ms={:.0}",
            src.rows,
            kept.len(),
            exec_mode.as_str(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(Self {
            w: QMatMul::from_arc(Arc::clone(&qweight))?,
            qweight,
            kept,
            kept_hash,
            vocab: src.rows,
            label,
            host_inv,
            rescue,
            exec_mode,
            guard_mode: policy.guard_mode,
            policy,
            counters: Arc::new(RescueCounters::default()),
        })
    }

    pub fn head_bytes(&self) -> usize {
        self.kept.len() * (self.qweight.shape().dims().last().copied().unwrap_or(0) / self.qweight.dtype().block_size().max(1) * self.qweight.dtype().type_size())
    }

    #[inline]
    pub fn active_rescue(&self) -> Option<&RestRescueGuard> {
        if self.counters.is_tripped() && !self.policy.strict_e1 {
            return None;
        }
        self.rescue.as_deref()
    }

    #[inline]
    pub fn map_local_to_orig_id(&self, raw_id: u32, kernel_mapped: bool) -> u32 {
        if kernel_mapped || self.kept.len() >= self.vocab {
            raw_id
        } else {
            self.kept.get(raw_id as usize).copied().unwrap_or(raw_id)
        }
    }

    pub fn guard_kind_label(&self) -> &'static str {
        if self.policy.promoted_from_e1 {
            return "promoted_all_rows";
        }
        if self.rescue.is_some() && self.counters.is_tripped() {
            return if self.policy.strict_e1 {
                "strict_e1_pending_promote"
            } else {
                "circuit_open"
            };
        }
        match &self.rescue {
            None => if self.kept.len() < self.vocab { "off" } else { "none" },
            Some(r) if r.clustered.is_some() => "clustered_cs",
            Some(_) => "single_cs",
        }
    }

    fn evaluate_circuit_breaker(&self, st: &mut StepGuardStats) {
        use crate::models::common::reflection_gate::NetGainGate;
        use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
        if self.rescue.is_none() || self.counters.is_tripped() || !self.policy.net_gain_gate {
            return;
        }
        let steps = self.counters.steps.load(Relaxed);
        if steps < NET_GAIN_WARMUP_STEPS {
            return;
        }
        let flagged = self.counters.rescued.load(Relaxed);
        let rescue_us = self.counters.rescue_us.load(Relaxed);
        let proj_us = self.counters.proj_kept_us.load(Relaxed);
        let proj_kept_ms = (proj_us as f64 / 1000.0) / (steps as f64);
        let proj_kept_eff = proj_kept_ms.max(1e-3);
        let proj_full_ms = proj_kept_eff * (self.vocab as f64 / self.kept.len().max(1) as f64);
        let rescue_rate = (flagged as f64) / (steps as f64);
        let rescue_host_ms = if flagged > 0 {
            (rescue_us as f64 / 1000.0) / (flagged as f64)
        } else {
            0.0
        };
        let gate = NetGainGate {
            proj_full_ms,
            proj_kept_ms: proj_kept_eff,
            rescue_rate,
            rescue_host_ms,
        };
        if rescue_rate > MAX_ALLOWED_RESCUE_RATE || !gate.is_profitable() {
            if self.counters.tripped.compare_exchange(false, true, AcqRel, Relaxed).is_ok() {
                if self.policy.strict_e1 {
                    self.counters.promote_all_rows.store(true, Release);
                }
                st.circuit_tripped = true;
                println!(
                    "[VOCAB-TRIM-STATS] scope={} steps={} rescued={} ({:.2}%) kept_ms={:.3} rescue_ms_avg={:.3} net_gain_gate=FAIL -> {}",
                    self.label,
                    steps,
                    flagged,
                    rescue_rate * 100.0,
                    proj_kept_ms,
                    rescue_host_ms,
                    if self.policy.strict_e1 { "promote_all_rows(strict_e1)" } else { "circuit_open" }
                );
            }
        }
    }

    /// 입력 임베딩 행: 묶인 표라 남긴 행의 출력 투영과 같은 바이트입니다.
    pub fn embed(&self, ids: &[u32], device: &Device) -> Option<candle_core::Result<Tensor>> {
        let rows: Option<Vec<u32>> = ids.iter().map(|&id| self.row_of(id)).collect();
        rows.map(|r| self.embed_rows(&r, device))
    }

    /// 어휘 번호 `id` 의 남긴 행 자리 (잘린 번호면 `None`)
    pub fn row_of(&self, id: u32) -> Option<u32> {
        self.host_inv.get(id as usize).copied().filter(|&j| (j as usize) < self.kept.len())
    }

    /// 남긴 행 자리 `rows` 의 입력 임베딩 [n, hidden] F32 (장치의 `qembed`)
    pub fn embed_rows(&self, rows: &[u32], device: &Device) -> candle_core::Result<Tensor> {
        self.w.embedding(&Tensor::from_vec(rows.to_vec(), rows.len(), device)?)
    }

    /// 남긴 행 `off..off+take` (`0..kept.len()`) 의 입력 임베딩 [take, hidden] F32 (`PrejudiceCache` 용).
    pub fn kept_rows_f32(&self, off: usize, take: usize, device: &Device) -> candle_core::Result<Tensor> {
        let rows: Vec<u32> = (off as u32..(off + take) as u32).collect();
        self.embed_rows(&rows, device)
    }

    /// Project `[M, hidden]` to `[M, K]` F32 logits without allocating GPU `inv`/`pad` or CPU F32 full dequant.
    pub fn project_rows_f32(&self, h_2d: &Tensor) -> candle_core::Result<Tensor> {
        let dev = h_2d.device();
        let exact_ok = exact_lm_head_enabled()
            && matches!(h_2d.dtype(), candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16)
            && matches!(
                self.qweight.dtype(),
                GgmlDType::Q8_0 | GgmlDType::Q4_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            );
        let out = if exact_ok && (dev.is_cuda() || dev.is_rocm()) {
            self.qweight.forward_exact(h_2d)?
        } else if exact_ok && dev.is_metal() {
            self.qweight.forward_exact(&h_2d.to_dtype(candle_core::DType::F32)?)?
        } else {
            self.w.forward(&h_2d.to_dtype(candle_core::DType::F32)?)?
        };
        out.to_dtype(candle_core::DType::F32)
    }

    /// Legacy compatibility full-vocab `[.., vocab]` logits via `[.., K]` projection + CPU scatter (0 B GPU `inv`/`pad`).
    pub fn logits(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        let mut dims = h.dims().to_vec();
        let k_dim = dims.last().copied().unwrap_or(0);
        let n = h.elem_count() / k_dim.max(1);
        let h_2d = h.reshape((n, k_dim))?.contiguous()?;
        let sub = self.project_rows_f32(&h_2d)?;
        if self.kept.len() == self.vocab {
            return if dims.len() == 2 { Ok(sub) } else {
                if let Some(last) = dims.last_mut() {
                    *last = self.vocab;
                }
                sub.reshape(dims)
            };
        }
        let sub_cpu = sub.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let k_len = self.kept.len();
        let mut full = vec![f32::NEG_INFINITY; n * self.vocab];
        for r in 0..n {
            let src_row = &sub_cpu[r * k_len..(r + 1) * k_len];
            let dst_row = &mut full[r * self.vocab..(r + 1) * self.vocab];
            for (j, &vid) in self.kept.iter().enumerate() {
                if (vid as usize) < self.vocab {
                    dst_row[vid as usize] = src_row[j];
                }
            }
        }
        if let Some(last) = dims.last_mut() {
            *last = self.vocab;
        }
        Tensor::from_vec(full, dims, h.device())
    }

    #[inline]
    fn apply_penalties_on_kept_slice(
        &self,
        sub_k: &mut [f32],
        seen_tokens: &[u32],
        repeat_penalty: f32,
        banned_ids: &[u32],
        local_seen_buf: &mut Vec<u32>,
    ) {
        if (repeat_penalty - 1.0).abs() > f32::EPSILON && !seen_tokens.is_empty() {
            local_seen_buf.clear();
            local_seen_buf.extend(seen_tokens.iter().filter_map(|&tid| self.row_of(tid)));
            local_seen_buf.sort_unstable();
            local_seen_buf.dedup();
            let inv_p = 1.0 / repeat_penalty;
            for &j in local_seen_buf.iter() {
                let slot = &mut sub_k[j as usize];
                if slot.is_finite() {
                    *slot = if *slot > 0.0 { *slot * inv_p } else { *slot * repeat_penalty };
                }
            }
        }
        for &bid in banned_ids {
            if let Some(j) = self.row_of(bid) {
                sub_k[j as usize] = f32::NEG_INFINITY;
            }
        }
    }

    #[inline]
    fn adjust_cold_logit(raw_val: f32, vid: u32, pp: &StepPostProcess<'_>) -> f32 {
        if pp.banned_ids.contains(&vid) || !raw_val.is_finite() {
            return f32::NEG_INFINITY;
        }
        if (pp.repeat_penalty - 1.0).abs() > f32::EPSILON && pp.seen_tokens.contains(&vid) {
            if raw_val > 0.0 {
                raw_val / pp.repeat_penalty
            } else {
                raw_val * pp.repeat_penalty
            }
        } else {
            raw_val
        }
    }

    /// Part 5 (Pack A·C): 100% 무손실 Greedy 토큰 선택 (Mode B `logits_argmax_cs` + 희귀 호스트 SIMD 구출).
    pub fn greedy_next_token(&self, last_hidden: &Tensor) -> Result<Vec<u32>> {
        self.greedy_next_token_with_stats(last_hidden).map(|(ids, _)| ids)
    }

    /// [`Self::greedy_next_token`] + `(token_ids, rescued_row_count)` 반환 (검증/관측용).
    pub fn greedy_next_token_with_stats(&self, last_hidden: &Tensor) -> Result<(Vec<u32>, usize)> {
        let (ids, st) = self.greedy_next_token_detailed(last_hidden)?;
        Ok((ids, st.flagged))
    }

    /// Detailed 1-step greedy selection with full `StepGuardStats` + `NetGainGate` circuit breaker.
    pub fn greedy_next_token_detailed(&self, last_hidden: &Tensor) -> Result<(Vec<u32>, StepGuardStats)> {
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let m = last_hidden.elem_count() / k_dim.max(1);
        let h_2d = last_hidden.reshape((m, k_dim))?.contiguous()?;
        let guard_opt = self.active_rescue();

        let use_fused = crate::models::common::accel_check::fused_op_ok(
            h_2d.device(),
            crate::models::common::accel_check::FusedOpKind::LmHeadArgmaxCs,
            h_2d.dtype(),
            k_dim,
        );
        let quant_mode_a_ok = use_fused
            && (h_2d.device().is_cuda() || h_2d.device().is_rocm())
            && k_dim % self.qweight.dtype().block_size() == 0
            && matches!(
                self.qweight.dtype(),
                GgmlDType::Q8_0 | GgmlDType::Q4_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            );
        let t_proj = std::time::Instant::now();
        let out = if quant_mode_a_ok {
            if let Some(guard) = guard_opt {
                candle_core::lm_head::fused_quant_argmax_cs(
                    &h_2d,
                    &self.qweight,
                    None,
                    Some(&guard.kept_gpu),
                    Some(&guard.cs_guard),
                )?
            } else {
                candle_core::lm_head::fused_quant_argmax_cs(&h_2d, &self.qweight, None, None, None)?
            }
        } else {
            let sub_logits = self.project_rows_f32(&h_2d)?;
            if let Some(guard) = guard_opt {
                if use_fused || h_2d.device().is_cpu() {
                    candle_core::lm_head::logits_argmax_cs(
                        &sub_logits,
                        Some(&h_2d),
                        Some(&guard.kept_gpu),
                        Some(&guard.cs_guard),
                    )?
                } else {
                    let sub_cpu = sub_logits.to_device(&Device::Cpu)?;
                    let h_cpu = h_2d.to_device(&Device::Cpu)?;
                    let kept_cpu = Tensor::from_vec(self.kept.as_ref().clone(), (self.kept.len(),), &Device::Cpu)?;
                    let guard_cpu = candle_core::lm_head::CsGuardParams {
                        mu_rest: guard.cs_guard.mu_rest.to_device(&Device::Cpu)?,
                        w_rest_bound: guard.cs_guard.w_rest_bound,
                        b_rest_max: guard.cs_guard.b_rest_max,
                    };
                    candle_core::lm_head::logits_argmax_cs(
                        &sub_cpu,
                        Some(&h_cpu),
                        Some(&kept_cpu),
                        Some(&guard_cpu),
                    )?
                }
            } else {
                candle_core::lm_head::logits_argmax_cs(&sub_logits, None, None, None)?
            }
        };

        let (mut ids, masks_opt) = out.download_ids_and_mask()?;
        let proj_kept_us = t_proj.elapsed().as_micros() as u64;
        let kernel_mapped = guard_opt.is_some();
        for id in &mut ids {
            *id = self.map_local_to_orig_id(*id, kernel_mapped);
        }

        let mut st = StepGuardStats {
            steps: m,
            fast_steps: m,
            proj_kept_us,
            quant_mode_a: quant_mode_a_ok,
            ..Default::default()
        };
        if let (Some(guard), Some(masks), Some(vals_t)) = (guard_opt, masks_opt, out.max_vals) {
            if masks.iter().any(|&f| f != 0) {
                let t_rescue = std::time::Instant::now();
                let core_vals = vals_t.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
                let h_host = h_2d
                    .to_device(&Device::Cpu)?
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for r in 0..m {
                    if masks[r] != 0 {
                        st.flagged += 1;
                        let h_row = &h_host[r * k_dim..(r + 1) * k_dim];
                        let core_id = ids[r];
                        let core_val = core_vals[r];
                        let (best_id, _best_val, c_chk, r_scr) =
                            guard.eval_rest_best_clustered(h_row, core_id, core_val, None)?;
                        st.clusters_checked += c_chk;
                        st.rows_scored += r_scr;
                        if best_id != core_id {
                            ids[r] = best_id;
                            st.rest_wins += 1;
                        }
                    }
                }
                st.rescue_us = t_rescue.elapsed().as_micros() as u64;
            }
        }
        self.counters.record(&st);
        self.evaluate_circuit_breaker(&mut st);
        Ok((ids, st))
    }

    /// `[1, K]` single-pass constrained greedy selection eliminating double projection (`penalty_fb`) and `[1, V]` expansion.
    pub fn greedy_single_pass_k(
        &self,
        last_hidden: &Tensor,
        pp: &StepPostProcess<'_>,
        scratch: &mut DecodeStepScratch,
    ) -> Result<(u32, StepGuardStats)> {
        let has_constraints = pp.prejudice_kept.is_some()
            || !pp.banned_ids.is_empty()
            || ((pp.repeat_penalty - 1.0).abs() > f32::EPSILON && !pp.seen_tokens.is_empty());
        if !has_constraints {
            let (ids, st) = self.greedy_next_token_detailed(last_hidden)?;
            return Ok((ids[0], st));
        }
        let t_proj = std::time::Instant::now();
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let mut sub_t = self.project_rows_f32(&h_2d)?;
        if let Some(prej) = pp.prejudice_kept {
            let prej_2d = prej.to_dtype(candle_core::DType::F32)?.reshape((1, self.kept.len()))?;
            sub_t = sub_t.broadcast_sub(&prej_2d)?;
        }
        let mut sub_vec = sub_t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let proj_kept_us = t_proj.elapsed().as_micros() as u64;
        self.apply_penalties_on_kept_slice(
            &mut sub_vec,
            pp.seen_tokens,
            pp.repeat_penalty,
            pp.banned_ids,
            &mut scratch.local_seen,
        );
        let mut best_j: Option<usize> = None;
        let mut best_val = f32::NEG_INFINITY;
        for (j, &v) in sub_vec.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_j = Some(j);
            }
        }
        let mut best_id = match best_j {
            Some(j) => self.kept.get(j).copied().unwrap_or(pp.fallback_eos_id),
            None => pp.fallback_eos_id,
        };
        let mut st = StepGuardStats {
            steps: 1,
            constrained_steps: 1,
            proj_kept_us,
            ..Default::default()
        };
        if let Some(guard) = self.active_rescue() {
            let t_rescue = std::time::Instant::now();
            let h_host = h_2d
                .to_device(&Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (rid, rval, c_chk, r_scr) = guard.eval_rest_best_clustered_with_post(
                &h_host,
                best_id,
                best_val,
                pp.target_norm_cpu,
                |vid, raw| Self::adjust_cold_logit(raw, vid, pp),
            )?;
            st.clusters_checked = c_chk;
            st.rows_scored = r_scr;
            if c_chk > 0 {
                st.flagged = 1;
            }
            if rval.is_finite() && rid != best_id {
                best_id = rid;
                st.rest_wins = 1;
            }
            st.rescue_us = t_rescue.elapsed().as_micros() as u64;
        }
        self.counters.record(&st);
        self.evaluate_circuit_breaker(&mut st);
        Ok((best_id, st))
    }

    /// `[1, K]` direct temperature sampling without reloading `all_rows` or expanding `[1, V]`.
    pub fn sample_single_pass_k(
        &self,
        last_hidden: &Tensor,
        logit_processor: &mut candle_transformers::generation::LogitsProcessor,
        pp: &StepPostProcess<'_>,
        scratch: &mut DecodeStepScratch,
    ) -> Result<u32> {
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let mut sub_t = self.project_rows_f32(&h_2d)?;
        if let Some(prej) = pp.prejudice_kept {
            let prej_2d = prej.to_dtype(candle_core::DType::F32)?.reshape((1, self.kept.len()))?;
            sub_t = sub_t.broadcast_sub(&prej_2d)?;
        }
        let mut sub_vec = sub_t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        self.apply_penalties_on_kept_slice(
            &mut sub_vec,
            pp.seen_tokens,
            pp.repeat_penalty,
            pp.banned_ids,
            &mut scratch.local_seen,
        );
        if !sub_vec.iter().any(|v| v.is_finite()) {
            return Ok(pp.fallback_eos_id);
        }
        let k_cpu = Tensor::from_vec(sub_vec, self.kept.len(), &Device::Cpu)?;
        let local_idx = logit_processor.sample(&k_cpu)?;
        Ok(self.kept.get(local_idx as usize).copied().unwrap_or(pp.fallback_eos_id))
    }

    /// Legacy constrained greedy callback wrapper respecting `active_rescue()` + `evaluate_circuit_breaker`.
    pub fn greedy_constrained<F>(
        &self,
        last_hidden: &Tensor,
        prej_kept: Option<&[f32]>,
        target_norm_cpu: Option<&[f32]>,
        mut adjust_kept: F,
    ) -> Result<(u32, StepGuardStats)>
    where
        F: FnMut(u32, f32) -> f32,
    {
        let t_proj = std::time::Instant::now();
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let sub_logits = self.project_rows_f32(&h_2d)?;
        let sub_vec = sub_logits.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let proj_kept_us = t_proj.elapsed().as_micros() as u64;
        let mut best_id = u32::MAX;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &raw_v) in sub_vec.iter().enumerate() {
            let vid = self.kept.get(idx).copied().unwrap_or(idx as u32);
            let mut v = raw_v;
            if let Some(pk) = prej_kept {
                if let Some(&p) = pk.get(idx) {
                    v -= p;
                }
            }
            v = adjust_kept(vid, v);
            if v > best_val || (v == best_val && vid < best_id) {
                best_val = v;
                best_id = vid;
            }
        }
        let mut st = StepGuardStats {
            steps: 1,
            constrained_steps: 1,
            proj_kept_us,
            ..Default::default()
        };
        if let Some(guard) = self.active_rescue() {
            let t_rescue = std::time::Instant::now();
            let h_host = h_2d
                .to_device(&Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (rid, _rval, c_chk, r_scr) =
                guard.eval_rest_best_clustered(&h_host, best_id, best_val, target_norm_cpu)?;
            st.clusters_checked = c_chk;
            st.rows_scored = r_scr;
            if c_chk > 0 {
                st.flagged = 1;
            }
            if rid != best_id {
                best_id = rid;
                st.rest_wins = 1;
            }
            st.rescue_us = t_rescue.elapsed().as_micros() as u64;
        }
        self.counters.record(&st);
        self.evaluate_circuit_breaker(&mut st);
        Ok((best_id, st))
    }

    /// Gate 1-C: 1스텝 무손실 Greedy + 그림자 Top-3 후보 + 확신도(confidence) 추출.
    pub fn greedy_with_shadow_top3(&self, last_hidden: &Tensor) -> Result<GreedyShadowStep> {
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let sub_logits = self.project_rows_f32(&h_2d)?;
        let sub_vec = sub_logits.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

        let mut top: [(u32, f32); 3] = [(u32::MAX, f32::NEG_INFINITY); 3];
        let mut max_l = f32::NEG_INFINITY;
        for (idx, &v) in sub_vec.iter().enumerate() {
            let vid = self.kept.get(idx).copied().unwrap_or(idx as u32);
            if v > max_l {
                max_l = v;
            }
            if v > top[0].1 || (v == top[0].1 && vid < top[0].0) {
                top[2] = top[1];
                top[1] = top[0];
                top[0] = (vid, v);
            } else if v > top[1].1 || (v == top[1].1 && vid < top[1].0) {
                top[2] = top[1];
                top[1] = (vid, v);
            } else if v > top[2].1 || (v == top[2].1 && vid < top[2].0) {
                top[2] = (vid, v);
            }
        }

        let mut rescued = false;
        if let Some(guard) = self.active_rescue() {
            let h_host = h_2d
                .to_device(&Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mu_host = guard.cs_guard.mu_rest.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let xx = embed_cpu_kernels::dot_f32_simd(&h_host, &h_host);
            let xm = embed_cpu_kernels::dot_f32_simd(&h_host, &mu_host);
            let upper = xx.sqrt().mul_add(guard.cs_guard.w_rest_bound, xm) + guard.cs_guard.b_rest_max;
            if top[0].1 < upper {
                rescued = true;
                let (rid, rv) = guard.eval_rest_best(&h_host)?;
                if rv > max_l {
                    max_l = rv;
                }
                if rv > top[0].1 || (rv == top[0].1 && rid < top[0].0) {
                    top[2] = top[1];
                    top[1] = top[0];
                    top[0] = (rid, rv);
                } else if rv > top[1].1 || (rv == top[1].1 && rid < top[1].0) {
                    top[2] = top[1];
                    top[1] = (rid, rv);
                } else if rv > top[2].1 || (rv == top[2].1 && rid < top[2].0) {
                    top[2] = (rid, rv);
                }
            }
        }

        let mut sum_exp = 0.0f64;
        for &v in &sub_vec {
            sum_exp += ((v - max_l) as f64).exp();
        }
        if rescued && top[0].0 != u32::MAX && self.row_of(top[0].0).is_none() {
            sum_exp += ((top[0].1 - max_l) as f64).exp();
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        let mk = |(id, logit): (u32, f32)| -> Option<ShadowCandidate> {
            if id == u32::MAX || !logit.is_finite() {
                None
            } else {
                let prob = (((logit - max_l) as f64).exp() * inv_sum) as f32;
                Some(ShadowCandidate { token_id: id, logit, prob })
            }
        };
        let top3 = [mk(top[0]), mk(top[1]), mk(top[2])];
        let confidence = top3[0].map_or(0.0, |c| c.prob);
        Ok(GreedyShadowStep {
            token_id: top[0].0,
            confidence,
            top3,
            rescued,
        })
    }
}

/// 잘라 내기를 켤 수 있으면(규칙이 `off` 가 아니고 매핑 · Q8_0 류 블록 표 — GPU 3종 + CPU + Metal 전체 지원, Phase 3-1 D4)
/// 입력 임베딩 · 출력 투영 표의 자리.
pub fn mapped_table(device: &Device, map: Option<&Arc<memmap2::Mmap>>, content: Option<&candle_core::quantized::gguf_file::Content>) -> Option<TableSource> {
    if rule_from_env() == TrimRule::Off {
        return None;
    }
    let _ = device;
    let (map, content) = (map?, content?);
    if content.tensor_infos.contains_key("output.weight") {
        println!("[VOCAB-TRIM] 출력 투영이 임베딩 표와 묶여 있지 않아(output.weight) 자르지 않습니다.");
        return None;
    }
    match TableSource::from_gguf(Arc::clone(map), content, "token_embd.weight") {
        Ok(src) if src.dtype.block_size() > 1 => Some(src),
        Ok(_) => None,
        Err(e) => {
            println!("[VOCAB-TRIM] 출력 투영 표를 매핑하지 못해 자르지 않습니다: {e}");
            None
        }
    }
}

/// 어휘 길이의 의미 선입견 벡터(로짓에서 빼는 값)에서 잘린 행 자리를 +∞ 로 정합니다.
pub fn exclude_dropped(head: Option<&TrimmedHead>, prejudice: Tensor) -> candle_core::Result<Tensor> {
    let Some(h) = head else { return Ok(prejudice) };
    if h.kept.len() == h.vocab || prejudice.elem_count() != h.vocab {
        return Ok(prejudice);
    }
    let kept: Vec<u8> = h.host_inv.iter().map(|&j| ((j as usize) < h.kept.len()) as u8).collect();
    let mask = Tensor::from_vec(kept, prejudice.shape(), prejudice.device())?;
    let inf = Tensor::full(f32::INFINITY, prejudice.shape(), prejudice.device())?.to_dtype(prejudice.dtype())?;
    mask.where_cond(&prejudice, &inf)
}

/// 오름차순 `a` 가 오름차순 `b` 안에 모두 있는가
fn sorted_subset(a: &[u32], b: &[u32]) -> bool {
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j == b.len() || b[j] != x {
            return false;
        }
        j += 1;
    }
    true
}

/// 전체 행 출력 투영의 이름표
pub fn all_rows_label() -> String {
    format!("{}:all-rows", rule_from_env().label())
}

/// 전체 행 출력 투영 (`all_rows`)
pub fn all_rows(src: &TableSource, device: &Device) -> Result<TrimmedHead> {
    TrimmedHead::build(src, Arc::new((0..src.rows as u32).collect()), all_rows_label(), device)
}

/// strict_e1 승격으로 전체 행으로 전환된 출력 투영
pub fn all_rows_promoted(src: &TableSource, device: &Device) -> Result<TrimmedHead> {
    let mut policy = GuardPolicyConfig::from_env();
    policy.promoted_from_e1 = true;
    policy.guard_mode = GuardMode::Off;
    TrimmedHead::build_with_policy(
        src,
        Arc::new((0..src.rows as u32).collect()),
        format!("{}:promoted_all_rows", rule_from_env().label()),
        device,
        policy,
    )
}

/// 지금의 문서 언어로 남긴 행을 맞춥니다 (규칙 · 언어가 바뀌었으면 다시 모음). 규칙이 `off` 면 비웁니다.
pub fn refresh(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device) -> Result<()> {
    refresh_for(slot, src, tok, device, "", false)
}

/// [`refresh`] + 프롬프트 텍스트 기반 갱신 (기존 호환 래퍼).
pub fn refresh_for(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device, prompt: &str, sampling: bool) -> Result<()> {
    let fixed = std::env::var(LANGS_ENV).is_ok_and(|v| !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto"));
    let langs = if fixed { current_langs() } else { current_langs().union(Langs::present(prompt)) };
    let compact_k = compact_active_k_for_model(None, src.rows, langs);
    let prompt_ids: Vec<u32> = if compact_k.is_some() && !prompt.is_empty() {
        tok.encode(prompt, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    refresh_for_ids(slot, src, tok, device, prompt, &prompt_ids, None, sampling)
}

/// Zero-re-encode head refresh using pre-encoded prompt payload token IDs & Track M adaptive budgets.
#[allow(clippy::too_many_arguments)]
pub fn refresh_for_ids(
    slot: &mut Option<TrimmedHead>,
    src: &TableSource,
    tok: &tokenizers::Tokenizer,
    device: &Device,
    prompt_text: &str,
    prompt_ids: &[u32],
    model_hint: Option<&str>,
    sampling: bool,
) -> Result<()> {
    let rule = rule_from_env();
    if rule == TrimRule::Off {
        *slot = None;
        return Ok(());
    }
    let promoted = slot.as_ref().is_some_and(|h| h.counters.should_promote_all_rows());
    let fixed = std::env::var(LANGS_ENV).is_ok_and(|v| !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto"));
    let langs = if fixed { current_langs() } else { current_langs().union(Langs::present(prompt_text)) };
    let compact_k = compact_active_k_for_model(model_hint, src.rows, langs);

    let force_all = promoted
        || (sampling && sample_full_vocab_env())
        || exec_mode_override() == Some(VocabExecMode::ArtifactDirect)
        || exec_mode_override() == Some(VocabExecMode::FullUntrimmed)
        || (src.rows <= 16_384 && compact_k.is_none());
    if force_all {
        if slot.as_ref().is_some_and(|h| h.kept.len() == src.rows && (!promoted || h.policy.promoted_from_e1)) {
            return Ok(());
        }
        *slot = None;
        *slot = Some(if promoted {
            all_rows_promoted(src, device)?
        } else {
            all_rows(src, device)?
        });
        return Ok(());
    }

    let doc_lang = current_doc_lang_hint();
    let label = match compact_k {
        Some(k) => match doc_lang.as_deref() {
            Some(dl) => format!("track_m({k},{dl}):{}", langs.label()),
            None => format!("track_m({k}):{}", langs.label()),
        },
        None => format!("{}:{}", rule.label(), langs.label()),
    };

    if let Some(h) = slot.as_ref() {
        if h.label == label
            && (prompt_ids.is_empty() && compact_k.is_none()
                || (!prompt_ids.is_empty()
                    && prompt_ids.iter().all(|&id| (id as usize) >= h.vocab || h.row_of(id).is_some())))
        {
            return Ok(());
        }
    }

    let ids = ids_for_with_prompt_ids(
        tok,
        src.rows,
        rule,
        langs,
        prompt_text,
        prompt_ids,
        model_hint,
        doc_lang.as_deref(),
    )?;
    if let Some(h) = slot.as_ref() {
        if h.kept.len() <= ids.len() + ids.len() / 10 && sorted_subset(&ids, &h.kept) {
            return Ok(());
        }
    }
    *slot = None;
    *slot = Some(TrimmedHead::build(src, ids, label, device)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_languages() {
        assert_eq!(kind_of(b"a", false), Kind::Byte);
        assert_eq!(kind_of(&[0xEC, 0x9E], false), Kind::Partial);
        assert_eq!(kind_of(" ORDER".as_bytes(), false), Kind::Ascii);
        assert!(matches!(kind_of("주문".as_bytes(), false), Kind::Script { hangul: true, .. }));
        assert_eq!(kind_of("заказ".as_bytes(), false), Kind::Other);
        assert_eq!(Langs::of_doc_lang("ko"), Langs::from_codes(&["ko", "en"]).unwrap());
        let ru = Langs::of_doc_lang("ru");
        assert!(ru.other && ru.latin && !ru.hangul && !ru.cjk, "12 개 밖 언어는 라틴 확장 + 그 밖 문자: {ru:?}");
        assert_eq!(langs_of_document("주문 번호 20231"), Langs::from_codes(&["ko"]).unwrap());
        let pl = langs_of_document("Zamówienie nr 20231, łączna wartość");
        assert!(pl.latin && !pl.hangul, "{pl:?}");
        assert_eq!(Langs::from_codes(&["all"]).unwrap(), Langs::all());
        let ko_hanja = langs_of_document("급수 펌프(給水) 압력 주문");
        assert!(ko_hanja.hangul && ko_hanja.hanja && !ko_hanja.cjk, "{ko_hanja:?}");
        let zh = langs_of_document("给水泵压力");
        assert!(zh.cjk && !zh.hanja, "{zh:?}");
    }

    #[test]
    fn image_reads_decide_on_non_latin_values_only() {
        let _scope = DocLangScope::image();
        assert_eq!(current_langs_raw(), Langs::any());
        assert_eq!(observe_read(r#"{"doc_type": "commercial_invoice", "number": "INV-2023-001"}"#), None, "라틴 값만으로는 정하지 않음");
        let got = observe_read(r#"{"seller": "주식회사 한빛", "number": "INV-2023-001"}"#).expect("한글 값으로 정함");
        assert_eq!(got, Langs::from_codes(&["ko", "en"]).unwrap());
        assert_eq!(observe_read(r#"{"seller": "株式会社"}"#), None, "한 번 정하면 그대로");
        let id = _scope.id;
        drop(_scope);
        assert!(!scopes().iter().any(|e| e.id == id), "문서가 끝나면 그 범위는 사라짐");
    }

    fn current_langs_raw() -> Langs {
        scoped_langs()
    }

    #[test]
    fn concurrent_flows_keep_their_own_language() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async {
            let opened = Arc::new(tokio::sync::Barrier::new(3));
            let checked = Arc::new(tokio::sync::Barrier::new(3));
            let run = |lang: &'static str, text: &'static str| {
                let (opened, checked) = (Arc::clone(&opened), Arc::clone(&checked));
                tokio::spawn(async move {
                    let scope = DocLangScope::text(lang, text);
                    opened.wait().await;
                    let mine = current_langs_raw();
                    checked.wait().await;
                    let id = scope.id;
                    drop(scope);
                    (mine, id)
                })
            };
            let ko = run("ko", "주문 번호");
            let de = run("de", "Bestellung Größe");
            opened.wait().await;
            let outside = current_langs_raw();
            checked.wait().await;
            let ((ko, ko_id), (de, de_id)) = (ko.await.unwrap(), de.await.unwrap());
            assert!(ko.hangul && !ko.latin, "검색 흐름은 한국어: {ko:?}");
            assert!(de.latin && !de.hangul, "추출 흐름은 독일어: {de:?}");
            assert!(outside.hangul && outside.latin, "범위 없는 흐름은 합집합: {outside:?}");
            assert!(!scopes().iter().any(|e| e.id == ko_id || e.id == de_id), "닫힌 범위는 사라짐");
        });
    }

    #[test]
    fn selection_keeps_bytes_scripts_and_ascii_and_drops_other_scripts() {
        let pieces: Vec<Vec<u8>> = vec![b"<s>".to_vec(), b"a".to_vec(), vec![0xEC, 0x9E], b" ORDER".to_vec(), b" the".to_vec(), "주문".as_bytes().to_vec(), "заказ".as_bytes().to_vec(), "Größe".as_bytes().to_vec()];
        let kinds: Vec<Kind> = pieces.iter().enumerate().map(|(i, p)| kind_of(p, i == 0)).collect();
        let (ids, tiers) = select_from(&kinds, &BTreeSet::new(), TrimRule::Full, Langs::from_codes(&["ko", "en"]).unwrap());
        let before_pad: usize = tiers.iter().filter(|(n, _)| *n != "pad").map(|(_, k)| k).sum();
        assert_eq!(before_pad, 6, "ko,en: 특수 · 바이트 · 조각 · ASCII 둘 · 한글 (키릴 · 라틴 확장 빠짐) {tiers:?}");
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert!(sorted_subset(&[1, 3], &[0, 1, 2, 3]) && sorted_subset(&[], &[5]) && !sorted_subset(&[1, 4], &[0, 1, 2, 3]) && !sorted_subset(&[2], &[]));
        assert_eq!(TrimRule::parse("full").unwrap(), TrimRule::Full);
        assert_eq!(TrimRule::parse("off").unwrap(), TrimRule::Off);
        assert!(TrimRule::parse("compact").is_err(), "예산을 둔 규칙은 없습니다 (영어 전부 디코딩)");
    }

    #[test]
    fn trimmed_logits_spread_back_to_the_vocabulary_with_minus_infinity() -> Result<()> {
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..4 * 32).map(|i| ((i as f32) * 0.37).sin()).collect();
        let q = candle_core::quantized::QTensor::quantize(&Tensor::from_vec(w, (4, 32), &dev)?, GgmlDType::Q8_0)?;
        let bytes = q.data()?.to_vec();
        let dir = std::env::temp_dir().join(format!("logis-vt-{}", std::process::id()));
        std::fs::write(&dir, &bytes)?;
        let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&dir)?)? });
        let src = TableSource { map, start: 0, rows: 4, cols: 32, dtype: GgmlDType::Q8_0 };
        let head = TrimmedHead::build(&src, Arc::new(vec![1, 3]), "t".into(), &dev)?;
        let h = Tensor::from_vec((0..32).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>(), (1, 32), &dev)?;
        let got: Vec<f32> = head.logits(&h)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = src.gather(&[3, 1], &dev)?.dequantize(&dev)?.flatten_all()?.to_vec1()?;
        let emb: Vec<f32> = head.embed(&[3, 1], &dev).expect("남긴 행")?.flatten_all()?.to_vec1()?;
        assert_eq!(emb, want);
        assert!(head.embed(&[1, 2], &dev).is_none() && head.embed(&[9], &dev).is_none());
        let table = crate::models::qwen3_5::model::EmbedTable::Mapped { src: src.clone(), device: dev.clone() };
        let ids = Tensor::new(&[[3u32, 2, 1, 0]], &dev)?;
        let mixed: Vec<f32> = table.forward_with_head(&ids, Some(&head))?.flatten_all()?.to_vec1()?;
        let plain: Vec<f32> = table.forward(&ids)?.flatten_all()?.to_vec1()?;
        assert_eq!(mixed, plain);
        let via_head = table.rows_f32_with_head(0, 4, Some(&head))?.to_vec2::<f32>()?;
        let mapped = table.rows_f32(0, 4)?.to_vec2::<f32>()?;
        assert_eq!((&via_head[1], &via_head[3]), (&mapped[1], &mapped[3]));
        assert_eq!((&via_head[0], &via_head[2]), (&mapped[1], &mapped[1]));
        let prej = exclude_dropped(Some(&head), Tensor::new(&[0.5f32, 0.25, 0.75, 1.0], &dev)?)?;
        let out: Vec<f32> = Tensor::new(&[5.0f32, 2.0, f32::NEG_INFINITY, 3.0], &dev)?.broadcast_sub(&prej)?.to_vec1()?;
        assert!(out[0] == f32::NEG_INFINITY && out[2] == f32::NEG_INFINITY, "{out:?}");
        assert_eq!((out[1], out[3]), (1.75, 2.0));
        let every = TrimmedHead::build(&src, Arc::new((0..4).collect()), "every".into(), &dev)?;
        assert_eq!(exclude_dropped(Some(&every), Tensor::new(&[0.5f32, 0.25, 0.75, 1.0], &dev)?)?.to_vec1::<f32>()?, vec![0.5, 0.25, 0.75, 1.0], "전체 행이면 그대로");
        let full: Vec<f32> = QMatMul::from_qtensor(q)?.forward(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!(got.len(), 4);
        assert!(got[0].is_infinite() && got[2].is_infinite());
        assert!((got[1] - full[1]).abs() < 1e-5 && (got[3] - full[3]).abs() < 1e-5);
        let all = TrimmedHead::build(&src, Arc::new((0..4).collect()), "all".into(), &dev)?;
        let got_all: Vec<f32> = all.logits(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!(got_all, full);
        let mid = TrimmedHead::build(&src, Arc::new(vec![1, 2]), "mid".into(), &dev)?;
        let got_mid: Vec<f32> = mid.logits(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!((got_mid[1], got_mid[2]), (full[1], full[2]));
        assert!(got_mid[0].is_infinite() && got_mid[3].is_infinite());
        drop(head);
        let _ = std::fs::remove_file(&dir);
        Ok(())
    }

    #[test]
    fn clustered_guard_constrained_lossless_parity() -> Result<()> {
        let dev = Device::Cpu;
        let rows = 512usize;
        let cols = 64usize;
        let mut w = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                w.push(((r * 13 + c * 7) as f32 * 0.19).sin() * 0.6 + ((r % 17) as f32) * 0.03);
            }
        }
        let q = candle_core::quantized::QTensor::quantize(&Tensor::from_vec(w, (rows, cols), &dev)?, GgmlDType::Q8_0)?;
        let bytes = q.data()?.to_vec();
        let path = std::env::temp_dir().join(format!("logis-vt-cg-{}", std::process::id()));
        std::fs::write(&path, &bytes)?;
        let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&path)?)? });
        let src = TableSource { map, start: 0, rows, cols, dtype: GgmlDType::Q8_0 };
        let kept: Vec<u32> = (0..128u32).collect();
        let policy = GuardPolicyConfig::from_raw_vars(Some("clustered"), None, Some("off"));
        let head = TrimmedHead::build_with_policy(&src, Arc::new(kept.clone()), "cg-test".into(), &dev, policy)?;
        let full_w = q.dequantize(&dev)?.to_vec2::<f32>()?;

        let target_norm: Vec<f32> = {
            let mut v: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.31).cos()).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v { *x /= n; }
            v
        };
        let prej_kept: Vec<f32> = kept
            .iter()
            .map(|&r| crate::models::common::prejudice::scalar_prejudice(&full_w[r as usize], &target_norm))
            .collect();

        for trial in 0..12usize {
            let h_vec: Vec<f32> = if trial % 2 == 0 {
                full_w[150 + trial * 25].clone()
            } else {
                (0..cols).map(|c| ((trial * 29 + c * 5) as f32 * 0.23).cos()).collect()
            };
            let h_t = Tensor::from_vec(h_vec.clone(), (1, cols), &dev)?;

            let (got_ids, _st) = head.greedy_next_token_detailed(&h_t)?;
            let mut want_id = 0u32;
            let mut want_val = f32::NEG_INFINITY;
            for (r, row) in full_w.iter().enumerate() {
                let d: f32 = row.iter().zip(&h_vec).map(|(a, b)| a * b).sum();
                if d > want_val {
                    want_val = d;
                    want_id = r as u32;
                }
            }
            assert_eq!(got_ids[0], want_id, "pure greedy mismatch at trial {trial}");

            let (got_c, _stc) = head.greedy_constrained(
                &h_t,
                Some(&prej_kept),
                Some(&target_norm),
                |_vid, v| v,
            )?;
            let mut want_cid = 0u32;
            let mut want_cval = f32::NEG_INFINITY;
            for (r, row) in full_w.iter().enumerate() {
                let d: f32 = row.iter().zip(&h_vec).map(|(a, b)| a * b).sum();
                let p = crate::models::common::prejudice::scalar_prejudice(row, &target_norm);
                let adj = d - p;
                if adj > want_cval {
                    want_cval = adj;
                    want_cid = r as u32;
                }
            }
            assert_eq!(got_c, want_cid, "constrained greedy mismatch at trial {trial}");
        }
        drop(head);
        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn guard_policy_circuit_trip_and_step17_id_remap() -> Result<()> {
        let p_def = GuardPolicyConfig::from_raw_vars(None, None, None);
        assert_eq!(p_def.guard_mode, GuardMode::Off);
        assert!(!p_def.strict_e1);
        let p_one = GuardPolicyConfig::from_raw_vars(None, Some("1"), None);
        assert_eq!(p_one.guard_mode, GuardMode::Off);
        let p_strict = GuardPolicyConfig::from_raw_vars(None, Some("strict"), None);
        assert_eq!(p_strict.guard_mode, GuardMode::Clustered);
        assert!(p_strict.strict_e1);

        let dev = Device::Cpu;
        let rows = 512usize;
        let cols = 64usize;
        let mut w = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let mut val = ((r * 11 + c * 3) as f32 * 0.21).sin() * 0.15;
                if r == 400 && c < 8 {
                    val = 2.0;
                } else if r == 300 && (8..16).contains(&c) {
                    val = 2.0;
                }
                w.push(val);
            }
        }
        let q = candle_core::quantized::QTensor::quantize(&Tensor::from_vec(w, (rows, cols), &dev)?, GgmlDType::Q8_0)?;
        let bytes = q.data()?.to_vec();
        let path = std::env::temp_dir().join(format!("logis-vt-cb-{}", std::process::id()));
        std::fs::write(&path, &bytes)?;
        let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&path)?)? });
        let src = TableSource { map, start: 0, rows, cols, dtype: GgmlDType::Q8_0 };
        let full_w = q.dequantize(&dev)?.to_vec2::<f32>()?;
        // Non-contiguous kept IDs including high indices >= 256 to prove step 17 remap
        let kept: Vec<u32> = (0..64u32).chain(256..320u32).collect();
        let policy = GuardPolicyConfig::from_raw_vars(Some("clustered"), None, Some("on"));
        let head = TrimmedHead::build_with_policy(&src, Arc::new(kept.clone()), "cb".into(), &dev, policy)?;
        let mut tripped_step = None;
        for step in 0..20usize {
            let h_vec = if step < 16 {
                full_w[400].clone() // cold row forces high rescue rate > 5% -> trips at step 15 (16th step)
            } else {
                full_w[300].clone() // non-contiguous kept row (local index 108 -> orig ID 300)
            };
            let h_t = Tensor::from_vec(h_vec, (1, cols), &dev)?;
            let (ids, st) = head.greedy_next_token_detailed(&h_t)?;
            if st.circuit_tripped {
                tripped_step = Some(step);
            }
            if step >= 16 {
                assert_eq!(head.guard_kind_label(), "circuit_open");
                assert_eq!(ids[0], 300, "step {step} after circuit open must still remap local idx -> orig ID 300");
            }
        }
        assert_eq!(tripped_step, Some(15), "circuit breaker must trip on 16th accumulated step");
        drop(head);
        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn single_pass_k_matches_full_vocab_and_all_masked_eos() -> Result<()> {
        let dev = Device::Cpu;
        let rows = 512usize;
        let cols = 64usize;
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.13).cos() * 0.4).collect();
        let q = candle_core::quantized::QTensor::quantize(&Tensor::from_vec(w, (rows, cols), &dev)?, GgmlDType::Q8_0)?;
        let bytes = q.data()?.to_vec();
        let path = std::env::temp_dir().join(format!("logis-vt-spk-{}", std::process::id()));
        std::fs::write(&path, &bytes)?;
        let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&path)?)? });
        let src = TableSource { map, start: 0, rows, cols, dtype: GgmlDType::Q8_0 };
        let kept: Vec<u32> = (10..138u32).collect();
        let head = TrimmedHead::build_with_policy(&src, Arc::new(kept.clone()), "spk".into(), &dev, GuardPolicyConfig::default())?;
        let full_w = q.dequantize(&dev)?.to_vec2::<f32>()?;
        let seen = vec![12u32, 12u32, 15u32, 42u32, 42u32];
        let banned = vec![10u32, 20u32];
        let penalty = 1.25f32;
        let mut scratch = DecodeStepScratch::default();
        let h_vec: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.27).sin()).collect();
        let h_t = Tensor::from_vec(h_vec.clone(), (1, cols), &dev)?;
        let pp = StepPostProcess {
            seen_tokens: &seen,
            repeat_penalty: penalty,
            banned_ids: &banned,
            prejudice_kept: None,
            target_norm_cpu: None,
            fallback_eos_id: 999,
        };
        let (got_id, st) = head.greedy_single_pass_k(&h_t, &pp, &mut scratch)?;
        assert_eq!(st.constrained_steps, 1);

        let mut want_id = u32::MAX;
        let mut want_val = f32::NEG_INFINITY;
        for &vid in &kept {
            if banned.contains(&vid) {
                continue;
            }
            let mut v: f32 = full_w[vid as usize].iter().zip(&h_vec).map(|(a, b)| a * b).sum();
            if seen.contains(&vid) {
                v = if v > 0.0 { v / penalty } else { v * penalty };
            }
            if v > want_val {
                want_val = v;
                want_id = vid;
            }
        }
        assert_eq!(got_id, want_id);

        // All candidates masked -> fallback_eos_id without leaking kept[0] or panicking in sampling
        let pp_masked = StepPostProcess {
            seen_tokens: &[],
            repeat_penalty: 1.0,
            banned_ids: &kept,
            prejudice_kept: None,
            target_norm_cpu: None,
            fallback_eos_id: 777,
        };
        let (masked_greedy, _) = head.greedy_single_pass_k(&h_t, &pp_masked, &mut scratch)?;
        assert_eq!(masked_greedy, 777);
        let mut lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            42,
            candle_transformers::generation::Sampling::TopKThenTopP { k: 40, p: 0.9, temperature: 0.8 },
        );
        let masked_sample = head.sample_single_pass_k(&h_t, &mut lp, &pp_masked, &mut scratch)?;
        assert_eq!(masked_sample, 777);

        // 95%+ ratio guard skip
        let mostly_kept: Vec<u32> = (0..(rows * 96 / 100) as u32).collect();
        let g96 = RestRescueGuard::build_with_mode(&src, &mostly_kept, GuardMode::Clustered, &dev)?;
        assert!(g96.is_none());

        drop(head);
        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn normalize_lang_tag_payload_ranges_and_lang_context() {
        assert_eq!(normalize_lang_tag("zh-tw").as_deref(), Some("zh-tw"));
        assert_eq!(normalize_lang_tag("zh_Hant").as_deref(), Some("zh-tw"));
        assert_eq!(normalize_lang_tag("kor").as_deref(), Some("ko"));
        assert_eq!(normalize_lang_tag("jpn").as_deref(), Some("ja"));
        assert_eq!(normalize_lang_tag("rus").as_deref(), Some("ru"));
        assert_eq!(normalize_lang_tag("tha").as_deref(), Some("th"));

        let prompt = "[RULES]\nRead ALL of [SOURCE DATA] carefully.\n[SOURCE DATA]\n株式会社テスト\n[TARGET DATA]\n[SOURCE]: こもれび商店\n[OUTPUT FORMAT]\n{}";
        let payload = extract_data_payload_only(prompt);
        assert!(payload.contains("株式会社テスト"));
        assert!(payload.contains("こもれび商店"));
        assert!(!payload.contains("Read ALL"));

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let ctx = LangContext::capture_or_from_doc("zh-tw");
            let child = ctx.clone().scope_future(async {
                let c = current_lang_context();
                assert_eq!(c.doc_lang, "zh-tw");
                assert!(c.langs.cjk);
            });
            child.await;
            let _g = ctx.attach();
            assert_eq!(current_doc_lang_hint().as_deref(), Some("zh-tw"));
        });
    }

    #[test]
    fn domain_lexicon_bundle_isolation() {
        let en_corpus = corpus_for_langs(Langs::of_doc_lang("en"));
        let has_hangul = |v: &[String]| v.iter().any(|s| s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c)));
        let has_arabic = |v: &[String]| v.iter().any(|s| s.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)));
        assert!(!has_hangul(&en_corpus), "en bundle must not pull Korean bias entries");
        assert!(!has_arabic(&en_corpus), "en bundle must not pull Arabic bias entries");

        let ko_corpus = corpus_for_langs(Langs::of_doc_lang("ko"));
        assert!(has_hangul(&ko_corpus), "ko bundle must include Korean bias entries");
        assert!(!has_arabic(&ko_corpus), "ko bundle must not pull Arabic bias entries");

        let ar_corpus = corpus_for_langs(Langs::of_doc_lang("ar"));
        if crate::utils::bias_schema::BIAS_DICT.get("ar").is_some() {
            assert!(has_arabic(&ar_corpus), "ar bundle must include Arabic bias entries when full-bias is loaded");
        }
        assert!(!has_hangul(&ar_corpus), "ar bundle must not pull Korean bias entries");
    }

    #[test]
    fn user_prompt_text_excludes_system_prompt() {
        use crate::openai_types::{
            ChatCompletionParameters, ChatCompletionRequestMessage,
            ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
            ChatCompletionRequestUserMessageContent,
        };
        let params = ChatCompletionParameters {
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: "한국어 시스템 안내 العربية".into(),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("Invoice #12345 USD 99.50".into()),
                    name: None,
                }),
            ],
            ..Default::default()
        };
        let u = user_prompt_text(&params);
        assert!(u.contains("Invoice #12345"));
        assert!(!u.contains("한국어"));
        assert!(!u.contains("العربية"));
        let langs = Langs::present(&u);
        assert!(!langs.hangul && !langs.arabic, "system scripts must not leak: {langs:?}");
    }
}
