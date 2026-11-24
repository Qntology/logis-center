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

// 🌟 [PD-QUANT TRIM] 토큰 종류(kind_of) · 언어 묶음(Langs) · 규칙(TrimRule) · 남길 행 고르기(select_from) · CJK_COMMON · ALIGN 은
//    premium-discount-quant::trim 을 씁니다 (이 파일에서 옮겨 간 코드, 주석 · 줄바꿈을 빼면 토큰 단위로 같음). 여기에는 logis 쪽 연결만
//    남습니다: 환경 변수 · 문서 언어 판정 · 토크나이저 · 출력 투영(TrimmedHead) · 실행 흐름별 언어 범위(DocLangScope).
pub use premium_discount_quant::trim::{kind_of, select_from, Kind, Langs, TrimRule, ALIGN, CJK_COMMON};

/// 문서 글에서: `detect_document_language` 의 현지 언어 + 영어, 그리고 글에 실제로 나온 문자 체계 (예전 `Langs::of_document`)
pub fn langs_of_document(text: &str) -> Langs {
    Langs::of_doc_lang(&crate::utils::lang_utils::detect_document_language(text)).union(Langs::present(text))
}

/// `LOGIS_VOCAB_TRIM` (기본 `full`; `compact` 는 `TrimRule::Full` + 컴팩트 활성 K 로 동작). 읽지 못한 값은 알리고 full 로 둡니다.
pub fn rule_from_env() -> TrimRule {
    match std::env::var(TRIM_ENV) {
        Ok(v) => {
            if v.trim().eq_ignore_ascii_case("compact") {
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

/// 규칙 (`full` · `compact` · `off`)
pub const TRIM_ENV: &str = "LOGIS_VOCAB_TRIM";
/// 언어 (`auto` = 문서 언어, `all`, 또는 `ko,en` 처럼)
pub const LANGS_ENV: &str = "LOGIS_VOCAB_LANGS";
/// 가드 모드 (`clustered` · `single` · `off`)
pub const GUARD_ENV: &str = "LOGIS_VOCAB_GUARD";
/// 컴팩트 모드 목표 활성 행 수 (예: `12288`)
pub const ACTIVE_K_ENV: &str = "LOGIS_VOCAB_ACTIVE_K";
/// 사전 계산된 `VocabPlan` JSON 경로
pub const PLAN_ENV: &str = "LOGIS_VOCAB_PLAN";

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
        match std::env::var(GUARD_ENV)
            .unwrap_or_else(|_| "clustered".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "none" | "false" => Self::Off,
            "single" | "sphere" => Self::Single,
            _ => Self::Clustered,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clustered => "clustered",
            Self::Single => "single",
            Self::Off => "off",
        }
    }
}

pub fn guard_clusters_from_env() -> usize {
    std::env::var("LOGIS_GUARD_CLUSTERS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|c| c.clamp(4, 256))
        .unwrap_or(64)
}

/// Whether compact 3-tier ASCII active-head mode is enabled (`LOGIS_VOCAB_TRIM=compact` or `LOGIS_VOCAB_ACTIVE_K`).
pub fn compact_active_k(vocab: usize, langs: Langs) -> Option<usize> {
    if let Ok(v) = std::env::var(ACTIVE_K_ENV) {
        if let Ok(k) = v.trim().parse::<usize>() {
            if k >= ALIGN && k < vocab {
                return Some(k.next_multiple_of(ALIGN).min(vocab));
            }
        }
    }
    let trim_val = std::env::var(TRIM_ENV).unwrap_or_default().trim().to_ascii_lowercase();
    if trim_val == "compact" {
        let base = if langs.cjk {
            16_384
        } else if langs.latin && !langs.hangul && !langs.arabic {
            14_336
        } else if langs.is_all() || langs == Langs::any() {
            24_576
        } else {
            12_288
        };
        if base < vocab {
            return Some(base.next_multiple_of(ALIGN).min(vocab));
        }
    }
    None
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
    pub quant_mode_a: bool,
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
        self.quant_mode_a |= other.quant_mode_a;
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

/// 토큰 종류 · 원본 바이트 조각. 어휘 크기마다 한 번만 계산합니다.
fn kinds_and_pieces(tok: &tokenizers::Tokenizer, vocab: usize) -> Arc<(Vec<Kind>, Vec<Vec<u8>>)> {
    static BASE: Lazy<Mutex<HashMap<(usize, usize), Arc<(Vec<Kind>, Vec<Vec<u8>>)>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tok.get_vocab_size(true));
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
    println!("[VOCAB-TRIM] 어휘 {vocab}: 토큰 종류 분류 · {:.2} s (어휘마다 한 번)", t0.elapsed().as_secs_f64());
    let b = Arc::new((kinds, pieces));
    BASE.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&b));
    b
}

/// 언어 묶음별 도메인 글 토큰 집합 (번들 격리, 캐시).
fn domain_tokens_for(tok: &tokenizers::Tokenizer, vocab: usize, langs: Langs) -> Arc<BTreeSet<u32>> {
    static DOM: Lazy<Mutex<HashMap<(usize, usize, String), Arc<BTreeSet<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tok.get_vocab_size(true), langs.label());
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

/// Optional pre-mined `VocabPlan` loaded from `LOGIS_VOCAB_PLAN`.
fn load_env_vocab_plan(vocab: usize) -> Option<Arc<Vec<u32>>> {
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
    if let Some(plan_ids) = load_env_vocab_plan(vocab) {
        if compact_active_k(vocab, langs).is_none() {
            return Ok(plan_ids);
        }
    }
    let compact_k = compact_active_k(vocab, langs);
    let base = kinds_and_pieces(tok, vocab);
    let domain = domain_tokens_for(tok, vocab, langs);

    if let Some(target_k) = compact_k {
        let prompt_ids: BTreeSet<u32> = if prompt.is_empty() {
            BTreeSet::new()
        } else {
            tok.encode(prompt, false)
                .map(|e| e.get_ids().iter().copied().filter(|&id| (id as usize) < vocab).collect())
                .unwrap_or_default()
        };
        let ascii_core: BTreeSet<u32> = premium_discount_quant::mine::default_ascii_struct_corpus()
            .iter()
            .filter_map(|s| tok.encode(s.as_str(), false).ok())
            .flat_map(|e| e.get_ids().to_vec())
            .filter(|&id| (id as usize) < vocab)
            .collect();
        let t0 = std::time::Instant::now();
        let (ids, tiers) = premium_discount_quant::trim::select_compact_ranked(
            &base.0,
            &domain,
            &prompt_ids,
            &ascii_core,
            langs,
            target_k,
            None,
        );
        let t: Vec<String> = tiers.iter().map(|(n, k)| format!("{n} {k}")).collect();
        println!(
            "[VOCAB-TRIM] compact(k<={target_k}) · {}: 어휘 {vocab} → {} 행 ({}) · {:.2} s",
            langs.label(),
            ids.len(),
            t.join(" · "),
            t0.elapsed().as_secs_f64()
        );
        return Ok(Arc::new(ids));
    }

    static CHOSEN: Lazy<Mutex<HashMap<(usize, usize, String), Arc<Vec<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tok.get_vocab_size(true), format!("{}:{}", rule.label(), langs.label()));
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

/// 열린 범위 하나 (연 순서대로 쌓임)
struct ScopeEntry {
    id: u64,
    flow: Flow,
    langs: Langs,
    /// 이미지 문서가 아직 언어를 정하지 않았는가 ([`DocLangScope::image`] → [`observe_read`])
    pending: bool,
}

static SCOPES: Lazy<Mutex<Vec<ScopeEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn scopes() -> std::sync::MutexGuard<'static, Vec<ScopeEntry>> {
    SCOPES.lock().unwrap_or_else(|p| p.into_inner())
}

/// 지금 흐름의 문서 언어 (위 규칙)
fn scoped_langs() -> Langs {
    let g = scopes();
    let me = flow();
    if let Some(e) = g.iter().rev().find(|e| e.flow == me) {
        return e.langs;
    }
    let mut seen: Vec<Flow> = Vec::new();
    let mut acc: Option<Langs> = None;
    for e in g.iter().rev() {
        if !seen.contains(&e.flow) {
            seen.push(e.flow);
            acc = Some(acc.map_or(e.langs, |a| a.union(e.langs)));
        }
    }
    acc.unwrap_or_else(Langs::any)
}

/// 문서 하나의 언어 범위. 연 실행 흐름에 묶이고 닫힐 때(드롭) 사라집니다. 문서 처리 함수의 맨 위에 두면
/// 그 문서가 끝난 뒤의 생성(검색 질의 등)은 다시 바깥 범위(없으면 아무 문자 체계도 빼지 않음)로 돌아갑니다.
pub struct DocLangScope {
    id: u64,
}

impl DocLangScope {
    /// 텍스트 맥락: 생성 전에 판정한 문서 언어(`detect_document_language` 의 현지 언어 + 영어)와 판정에 쓴 글에 나온 문자 체계
    pub fn text(doc_lang: &str, text: &str) -> Self {
        Self::open(Langs::of_doc_lang(doc_lang).union(Langs::present(text)), false)
    }

    /// 이미지 맥락: 언어를 미리 모르므로 아무 문자 체계도 빼지 않고 시작하고, 판독 글이 문자 체계를 가리키면 [`observe_read`] 가 정합니다.
    pub fn image() -> Self {
        Self::open(Langs::any(), true)
    }

    fn open(langs: Langs, pending: bool) -> Self {
        let id = NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        scopes().push(ScopeEntry { id, flow: flow(), langs, pending });
        Self { id }
    }

    fn update(&self, f: impl FnOnce(&mut ScopeEntry)) {
        if let Some(e) = scopes().iter_mut().find(|e| e.id == self.id) {
            f(e);
        }
    }

    /// 같은 문서에서 언어를 다시 판정했을 때 (노이즈 제거 뒤 재확정)
    pub fn set(&self, doc_lang: &str, text: &str) {
        let langs = Langs::of_doc_lang(doc_lang).union(Langs::present(text));
        self.update(|e| {
            e.langs = langs;
            e.pending = false;
        });
    }

    /// 앱이 출력하라고 지시하는 언어를 더합니다 (검색 답변의 "Write in {LANG}" = 앱 언어 등). 언어 이름(`korean`)도 코드(`ko`)도 받습니다.
    pub fn also(&self, lang: &str) {
        let more = Langs::of_doc_lang(&crate::utils::bias_schema::lang_code_of(lang));
        self.update(|e| e.langs = e.langs.union(more));
    }
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

/// 이미지 문서의 판독 글 하나를 봅니다. 아직 언어를 정하지 않았고, 값에 라틴 밖 문자 체계(한글 · 가나 · 한자 · 아랍 문자 등)가
/// 3 자 이상 나오면 `detect_document_language` 로 문서 언어(현지 언어 + 영어)를 정해 다음 생성부터 그 행만 남깁니다.
/// 라틴 문자만 나온 판독으로는 정하지 않습니다: 한국어 무역 서류도 첫 값(번호 · 영문 상호)은 라틴 문자뿐인 일이 흔해서,
/// 영어로 정하면 뒤의 한글 값을 쓸 수 없게 됩니다. 정했으면 그 언어를 돌려줍니다.
/// 정할 이미지 범위는 지금 흐름이 연 가장 안쪽의 언어 미정 범위이고, 지금 흐름에 없으면 열린 언어 미정 범위가 하나뿐일 때 그것입니다
/// (둘 이상이면 어느 문서의 판독인지 모르므로 정하지 않습니다: 아무 문자 체계도 빼지 않는 채로 둠).
pub fn observe_read(text: &str) -> Option<Langs> {
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
    let values = read_values(text);
    let decisive = values.chars().filter(|&c| c.is_alphabetic() && !c.is_ascii() && !matches!(c as u32, 0x00C0..=0x024F | 0x1E00..=0x1EFF)).count();
    if decisive < 3 {
        return None;
    }
    let langs = langs_of_document(&values);
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

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn decode_q8_0_row(raw: &[u8], out: &mut [f32]) {
    debug_assert_eq!(raw.len(), (out.len() / 32) * 34);
    for (blk, dst) in raw.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..32 {
            dst[i] = (blk[2 + i] as i8) as f32 * d;
        }
    }
}

impl RestRescueGuard {
    /// `TableSource` mmap 에서 잘린 REST 행을 청크 스트리밍해 `mu_rest` 와 `w_rest_bound` 를 1회 사전 계산합니다.
    pub fn build(src: &TableSource, kept: &[u32], device: &Device) -> Result<Option<Self>> {
        if kept.len() >= src.rows {
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
        let rb = src.row_bytes();

        // Pass 1: mu_rest [cols] (f64 누산 → f32)
        let sum_f64: Vec<f64> = if src.dtype == GgmlDType::Q8_0 {
            rest_ids
                .par_chunks(512)
                .map(|chunk| {
                    let mut acc = vec![0.0f64; cols];
                    let mut row_buf = vec![0.0f32; cols];
                    for &rid in chunk {
                        let off = src.start + (rid as usize) * rb;
                        decode_q8_0_row(&src.map[off..off + rb], &mut row_buf);
                        for c in 0..cols {
                            acc[c] += row_buf[c] as f64;
                        }
                    }
                    acc
                })
                .reduce(
                    || vec![0.0f64; cols],
                    |mut a, b| {
                        for c in 0..cols {
                            a[c] += b[c];
                        }
                        a
                    },
                )
        } else {
            let mut acc = vec![0.0f64; cols];
            for chunk in rest_ids.chunks(512) {
                let deq = src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                for row in vals.chunks_exact(cols) {
                    for c in 0..cols {
                        acc[c] += row[c] as f64;
                    }
                }
            }
            acc
        };
        let inv_n = 1.0 / (rest_ids.len() as f64);
        let mu_vec: Vec<f32> = sum_f64.into_iter().map(|s| (s * inv_n) as f32).collect();

        // Pass 2: max_{r in REST} ||w_r - mu_rest||_2
        let max_norm_sq: f32 = if src.dtype == GgmlDType::Q8_0 {
            rest_ids
                .par_chunks(512)
                .map(|chunk| {
                    let mut local_max = 0.0f32;
                    let mut row_buf = vec![0.0f32; cols];
                    for &rid in chunk {
                        let off = src.start + (rid as usize) * rb;
                        decode_q8_0_row(&src.map[off..off + rb], &mut row_buf);
                        let mut sq = 0.0f32;
                        for c in 0..cols {
                            let diff = row_buf[c] - mu_vec[c];
                            sq = diff.mul_add(diff, sq);
                        }
                        if sq > local_max {
                            local_max = sq;
                        }
                    }
                    local_max
                })
                .reduce(|| 0.0f32, f32::max)
        } else {
            let mut msq = 0.0f32;
            for chunk in rest_ids.chunks(512) {
                let deq = src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                for row in vals.chunks_exact(cols) {
                    let mut sq = 0.0f32;
                    for c in 0..cols {
                        let diff = row[c] - mu_vec[c];
                        sq = diff.mul_add(diff, sq);
                    }
                    if sq > msq {
                        msq = sq;
                    }
                }
            }
            msq
        };

        let w_rest_bound = max_norm_sq.sqrt() * (1.0 + 1e-3);
        let mu_rest = Tensor::from_vec(mu_vec, (cols,), device)?;
        let kept_gpu = Tensor::from_vec(kept.to_vec(), (kept.len(),), device)?;
        let cs_guard = candle_core::lm_head::CsGuardParams {
            mu_rest,
            w_rest_bound,
            b_rest_max: 0.0,
        };

        // Phase 3-1 D2: Optional 64-cluster ColdGuard with disk cache (`.cold_guard.bin`).
        let guard_mode = GuardMode::from_env();
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
                // Decode cold rows in parallel chunks for Lloyd k-means (4 iters)
                let mut cold_rows = vec![0.0f32; rest_ids.len() * cols];
                if src.dtype == GgmlDType::Q8_0 {
                    cold_rows
                        .par_chunks_exact_mut(cols)
                        .zip(rest_ids.par_iter())
                        .for_each(|(dst, &rid)| {
                            let off = src.start + (rid as usize) * rb;
                            decode_q8_0_row(&src.map[off..off + rb], dst);
                        });
                } else {
                    for (ci, chunk) in rest_ids.chunks(512).enumerate() {
                        let deq = src.gather(chunk, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                        let vals = deq.flatten_all()?.to_vec1::<f32>()?;
                        let base = ci * 512 * cols;
                        cold_rows[base..base + vals.len()].copy_from_slice(&vals);
                    }
                }
                let cg = premium_discount_quant::lmhead::build_cold_guard(
                    &cold_rows,
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
        // Skip caching for tiny unit-test synthetic tables
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
        // Sample first 64 bytes of weight table to bind cache to exact model weights
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

    /// Score one cold row `rid` directly from mmap (SIMD F32 dot product, optional nonnegative prejudice).
    #[inline]
    fn score_cold_row(&self, rid: u32, h_row: &[f32], target_norm_cpu: Option<&[f32]>, row_buf: &mut [f32]) -> Result<f32> {
        let rb = self.src.row_bytes();
        let v = if self.src.dtype == GgmlDType::Q8_0 {
            let off = self.src.start + (rid as usize) * rb;
            decode_q8_0_row(&self.src.map[off..off + rb], row_buf);
            let raw = embed_cpu_kernels::dot_f32_simd(h_row, row_buf);
            if let Some(tn) = target_norm_cpu {
                raw - crate::models::common::prejudice::scalar_prejudice(row_buf, tn)
            } else {
                raw
            }
        } else {
            let deq = self.src.gather(&[rid], &Device::Cpu)?.dequantize(&Device::Cpu)?;
            let vals = deq.flatten_all()?.to_vec1::<f32>()?;
            let raw = embed_cpu_kernels::dot_f32_simd(h_row, &vals);
            if let Some(tn) = target_norm_cpu {
                raw - crate::models::common::prejudice::scalar_prejudice(&vals, tn)
            } else {
                raw
            }
        };
        Ok(v)
    }

    /// Stage-1 clustered rescue evaluation returning `(best_id, best_val, clusters_checked, rows_scored)`.
    /// Guarantees 0 false negatives even with `target_norm_cpu` (since `prejudice >= 0`, `w·h - prej <= w·h <= ub_j`).
    pub fn eval_rest_best_clustered(
        &self,
        h_row: &[f32],
        mut best_id: u32,
        mut best_val: f32,
        target_norm_cpu: Option<&[f32]>,
    ) -> Result<(u32, f32, usize, usize)> {
        let cols = self.src.cols;
        anyhow::ensure!(h_row.len() == cols, "RestRescueGuard: hidden len {} != cols {}", h_row.len(), cols);
        let Some(cg) = &self.clustered else {
            let (rid, rval) = if target_norm_cpu.is_none() {
                self.eval_rest_best(h_row)?
            } else {
                let mut rb_buf = vec![0.0f32; cols];
                let mut bid = u32::MAX;
                let mut bval = f32::NEG_INFINITY;
                for &rid in self.rest_ids.iter() {
                    let v = self.score_cold_row(rid, h_row, target_norm_cpu, &mut rb_buf)?;
                    if v > bval || (v == bval && rid < bid) {
                        bval = v;
                        bid = rid;
                    }
                }
                (bid, bval)
            };
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
        let rb = self.src.row_bytes();
        let better = |(i1, v1): (u32, f32), (i2, v2): (u32, f32)| -> (u32, f32) {
            if v2 > v1 || (v2 == v1 && i2 < i1) { (i2, v2) } else { (i1, v1) }
        };

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
            let (cid, cval) = if self.src.dtype == GgmlDType::Q8_0 && members.len() >= 128 {
                members
                    .par_chunks(256)
                    .map(|chunk| {
                        let mut lid = u32::MAX;
                        let mut lval = f32::NEG_INFINITY;
                        let mut row_buf = vec![0.0f32; cols];
                        for &rid in chunk {
                            let off = self.src.start + (rid as usize) * rb;
                            decode_q8_0_row(&self.src.map[off..off + rb], &mut row_buf);
                            let mut v = embed_cpu_kernels::dot_f32_simd(h_row, &row_buf);
                            if let Some(tn) = target_norm_cpu {
                                v -= crate::models::common::prejudice::scalar_prejudice(&row_buf, tn);
                            }
                            if v > lval || (v == lval && rid < lid) {
                                lval = v;
                                lid = rid;
                            }
                        }
                        (lid, lval)
                    })
                    .reduce(|| (u32::MAX, f32::NEG_INFINITY), better)
            } else {
                let mut lid = u32::MAX;
                let mut lval = f32::NEG_INFINITY;
                let mut row_buf = vec![0.0f32; cols];
                for &rid in members {
                    let v = self.score_cold_row(rid, h_row, target_norm_cpu, &mut row_buf)?;
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

    /// 희귀 구출(`rescue_mask == 1`): 호스트 mmap 의 REST 행을 SIMD(`embed_cpu_kernels`) 로 평가해
    /// `(best_vocab_id, best_logit)` 을 반환합니다.
    pub fn eval_rest_best(&self, h_row: &[f32]) -> Result<(u32, f32)> {
        let cols = self.src.cols;
        anyhow::ensure!(h_row.len() == cols, "RestRescueGuard: hidden len {} != cols {}", h_row.len(), cols);
        let rb = self.src.row_bytes();
        let better = |(i1, v1): (u32, f32), (i2, v2): (u32, f32)| -> (u32, f32) {
            if v2 > v1 || (v2 == v1 && i2 < i1) { (i2, v2) } else { (i1, v1) }
        };
        if self.src.dtype == GgmlDType::Q8_0 {
            let best = self
                .rest_ids
                .par_chunks(512)
                .map(|chunk| {
                    let mut best_id = u32::MAX;
                    let mut best_val = f32::NEG_INFINITY;
                    let mut row_buf = vec![0.0f32; cols];
                    for &rid in chunk {
                        let off = self.src.start + (rid as usize) * rb;
                        decode_q8_0_row(&self.src.map[off..off + rb], &mut row_buf);
                        let v = embed_cpu_kernels::dot_f32_simd(h_row, &row_buf);
                        if v > best_val || (v == best_val && rid < best_id) {
                            best_val = v;
                            best_id = rid;
                        }
                    }
                    (best_id, best_val)
                })
                .reduce(|| (u32::MAX, f32::NEG_INFINITY), better);
            Ok(best)
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

/// 남긴 행만의 출력 투영. 로짓은 전체 어휘 길이로 펼쳐 CPU 에 돌려줍니다 (잘린 행은 −∞).
/// 복제는 장치의 행렬을 함께 씁니다 (`QMatMul` 의 Arc).
#[derive(Clone)]
pub struct TrimmedHead {
    w: QMatMul,
    qweight: Arc<candle_core::quantized::QTensor>,
    pub kept: Arc<Vec<u32>>,
    pub vocab: usize,
    /// 남긴 행을 고른 규칙 · 언어 (같으면 다시 만들지 않음)
    pub label: String,
    /// 장치의 역표 [vocab] (u32): 어휘 번호 → 남긴 K 행 안의 자리, 잘린 번호는 K (덧붙인 −∞ 칸)
    inv: Tensor,
    /// 같은 역표의 호스트 사본 (입력 임베딩을 이 행에서 꺼낼 때, [`Self::embed`])
    host_inv: Arc<Vec<u32>>,
    /// 덧붙일 −∞ 한 칸 [1, 1]
    pad: Tensor,
    /// Part 5 (Pack A·C) + Phase 3-1 D2: 무손실 Greedy 디코딩을 위한 REST CS 가드 + 다중 군집 호스트 구출기.
    rescue: Option<Arc<RestRescueGuard>>,
    pub exec_mode: VocabExecMode,
    pub guard_mode: GuardMode,
    pub counters: Arc<RescueCounters>,
}

impl TrimmedHead {
    pub fn build(src: &TableSource, kept: Arc<Vec<u32>>, label: String, device: &Device) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let qweight = Arc::new(src.gather(&kept, device)?);
        let mut inv = vec![kept.len() as u32; src.rows];
        for (i, &id) in kept.iter().enumerate() {
            inv[id as usize] = i as u32;
        }
        let host_inv = Arc::new(inv.clone());
        let inv = Tensor::from_vec(inv, src.rows, device)?;
        let pad = Tensor::full(f32::NEG_INFINITY, (1, 1), device)?;
        let guard_mode = GuardMode::from_env();
        let is_trimmed = kept.len() < src.rows;
        let rescue = if is_trimmed && guard_mode != GuardMode::Off {
            RestRescueGuard::build(src, &kept, device)?.map(Arc::new)
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
        let guard_tag = match &rescue {
            None => if is_trimmed { "off".to_string() } else { "full".to_string() },
            Some(r) => match &r.clustered {
                Some(cg) => format!("clustered(c={},cache={})", cg.radius.len(), if r.cache_hit { "hit" } else { "build" }),
                None => "single".to_string(),
            },
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
            vocab: src.rows,
            label,
            inv,
            host_inv,
            pad,
            rescue,
            exec_mode,
            guard_mode,
            counters: Arc::new(RescueCounters::default()),
        })
    }

    pub fn head_bytes(&self) -> usize {
        self.kept.len() * (self.qweight.shape().dims().last().copied().unwrap_or(0) / self.qweight.dtype().block_size().max(1) * self.qweight.dtype().type_size())
    }

    pub fn guard_kind_label(&self) -> &'static str {
        match &self.rescue {
            None => if self.kept.len() < self.vocab { "off" } else { "none" },
            Some(r) if r.clustered.is_some() => "clustered_cs",
            Some(_) => "single_cs",
        }
    }

    /// 입력 임베딩 행: 묶인 표라 남긴 행의 출력 투영과 같은 바이트입니다. `ids` 가 모두 남긴 행에 있을 때만 (아니면 `None`) 장치의
    /// 양자화 임베딩(`qembed`)으로 꺼내므로, 표 전체를 장치에 둔 off 의 입력 임베딩과 비트까지 같습니다. 프롬프트에 나온 문자 체계는
    /// 남긴 행에 들어가므로([`refresh_for`]) 프롬프트 토큰은 거의 언제나 여기서 나옵니다. 매핑에서 행을 읽으면 프리필마다 작업 집합에서
    /// 밀려난 표 페이지를 다시 들여와(0.6B 목록 추출 프리필 토큰당 +0.2 ms) 느렸습니다.
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

    /// `PRC-03`: 출력 헤드 산술 단일화 — CUDA/ROCm 에서 지원 양자화 형식은 활성값 Q8_1 근사
    /// (`QMatMul::forward`) 대신 정확 W8A16(`forward_exact`)을 사용하여 그리디 고속 경로(Mode A)와
    /// 금지 토큰 폴백·제약 디코딩·샘플링 경로의 산술을 일치시킵니다. (`LOGIS_EXACT_LM_HEAD=0` 으로 끄기 가능)
    fn project_rows(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        static EXACT_HEAD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let exact_on = *EXACT_HEAD.get_or_init(|| {
            let v = std::env::var("LOGIS_EXACT_LM_HEAD")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "off" | "false" | "no" | "legacy")
        });
        let use_exact = exact_on
            && (h.device().is_cuda() || h.device().is_rocm())
            && matches!(h.dtype(), candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16)
            && matches!(
                self.qweight.dtype(),
                GgmlDType::Q8_0 | GgmlDType::Q4_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            );
        if use_exact {
            self.w.forward_exact(h)?.to_dtype(candle_core::DType::F32)
        } else {
            self.w.forward(h)?.to_dtype(candle_core::DType::F32)
        }
    }

    /// `h` `[.., hidden]` (F32) → `[.., vocab]` F32, `h` 와 같은 장치. 남긴 행 밖은 −∞.
    /// 펼치기는 장치의 모으기 하나입니다: 남긴 K 행의 로짓 뒤에 −∞ 한 칸을 붙이고 역표로 `index_select` (출력 원소마다 스레드 하나).
    /// (CPU 에서 어휘 길이의 배열을 채우면 dev 빌드 0.6B 디코드가 전체 어휘보다 1 ~ 8 % 느렸고, 포크의 `index_add` 는 이 모양에서
    /// 스레드 하나가 K 행을 모두 돌아 0.6B +138 % · 2B +129 % 였습니다.)
    pub fn logits(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        let mut dims = h.dims().to_vec();
        let k = self.project_rows(h)?;
        if self.kept.len() == self.vocab {
            // 전체 행(번호 순)은 펼칠 것이 없습니다
            return Ok(k);
        }
        let n = k.elem_count() / self.kept.len().max(1);
        let k = k.reshape((n, self.kept.len()))?;
        let pad = if n == 1 { self.pad.clone() } else { self.pad.broadcast_as((n, 1))?.contiguous()? };
        let full = Tensor::cat(&[&k, &pad], 1)?.index_select(&self.inv, 1)?;
        if let Some(last) = dims.last_mut() {
            *last = self.vocab;
        }
        full.reshape(dims)
    }

    /// Part 5 (Pack A·C): 100% 무손실 Greedy 토큰 선택 (Mode B `logits_argmax_cs` + 희귀 호스트 SIMD 구출).
    ///
    /// `[M, V_full]` 로짓 확장 없이 `[M, K]` 서브 로짓에서 즉시 argmax + CS 상한 검증을 수행하고,
    /// `rescue_mask[r] == 1` 인 행에 한해서만 호스트 mmap REST 행을 SIMD 로 평가해
    /// 전체 어휘 Greedy 와 100% 동일한 토큰 ID 를 반환합니다.
    pub fn greedy_next_token(&self, last_hidden: &Tensor) -> Result<Vec<u32>> {
        self.greedy_next_token_with_stats(last_hidden).map(|(ids, _)| ids)
    }

    /// [`Self::greedy_next_token`] + `(token_ids, rescued_row_count)` 반환 (검증/관측용).
    pub fn greedy_next_token_with_stats(&self, last_hidden: &Tensor) -> Result<(Vec<u32>, usize)> {
        let (ids, st) = self.greedy_next_token_detailed(last_hidden)?;
        Ok((ids, st.flagged))
    }

    /// Detailed 1-step greedy selection with full `StepGuardStats` (clusters, rows scored, rescue µs, Mode A).
    pub fn greedy_next_token_detailed(&self, last_hidden: &Tensor) -> Result<(Vec<u32>, StepGuardStats)> {
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let m = last_hidden.elem_count() / k_dim.max(1);
        let h_2d = last_hidden.reshape((m, k_dim))?.contiguous()?;

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
        let out = if quant_mode_a_ok {
            if let Some(guard) = &self.rescue {
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
            let sub_logits = self.project_rows(&h_2d)?;
            if let Some(guard) = &self.rescue {
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
        // Critical: when `self.rescue` is None (e.g. LOGIS_VOCAB_GUARD=off) and kept < vocab,
        // kernel returns local indices 0..K-1; remap back to original vocab IDs!
        if self.rescue.is_none() && self.kept.len() < self.vocab {
            for id in &mut ids {
                if let Some(&orig) = self.kept.get(*id as usize) {
                    *id = orig;
                }
            }
        }

        let mut st = StepGuardStats {
            steps: m,
            fast_steps: m,
            quant_mode_a: quant_mode_a_ok,
            ..Default::default()
        };
        if let (Some(guard), Some(masks), Some(vals_t)) = (&self.rescue, masks_opt, out.max_vals) {
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
        Ok((ids, st))
    }

    /// Phase 3-1 D3: Lossless constrained greedy decoding (supports `semantic_prejudice >= 0`,
    /// `banned_ids`, and `repeat_penalty` without materializing `[1, V]` logits).
    /// Because `prejudice >= 0`, `w_r · h - prej_r <= w_r · h <= ub_j`, preserving 0 false negatives!
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
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let sub_logits = self.project_rows(&h_2d)?;
        let sub_vec = sub_logits.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
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
            ..Default::default()
        };
        if let Some(guard) = &self.rescue {
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
        Ok((best_id, st))
    }

    /// Gate 1-C: 1스텝 무손실 Greedy + 그림자 Top-3 후보 + 확신도(confidence) 추출.
    pub fn greedy_with_shadow_top3(&self, last_hidden: &Tensor) -> Result<GreedyShadowStep> {
        let k_dim = *last_hidden.dims().last().unwrap_or(&0);
        let h_2d = last_hidden.reshape((1, k_dim))?.contiguous()?;
        let sub_logits = self.w.forward(&h_2d)?.to_dtype(candle_core::DType::F32)?;
        let sub_vec = sub_logits.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

        // Top-3 CORE 후보 + softmax 정규화 분모 계산
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

        // CS Guard 검사 및 희귀 구출
        let mut rescued = false;
        if let Some(guard) = &self.rescue {
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

/// 어휘 길이의 의미 선입견 벡터(로짓에서 빼는 값)에서 잘린 행 자리를 +∞ 로 정합니다. 빼면 그 토큰은 −∞ 라, 잘린 토큰은 선입견을
/// 거쳐서도 절대 뽑히지 않습니다 (잘라 내기의 약속을 이 벡터에도 직접 걸어 둠). 선입견 블록은 잘린 행 자리를 남긴 행 하나로 채워
/// 계산하므로(`EmbedTable::rows_f32_with_head`) 그 자리 값은 의미가 없는데, 지금은 잘린 토큰의 로짓이 이미 −∞ 라 쓰이지 않을 뿐이었습니다.
/// 나중에 선입견을 빼기 전에 로짓을 고치는 코드가 생겨도 채움값이 새어 나가지 않습니다. 남긴 행의 값은 그대로입니다.
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

/// 전체 행 출력 투영 (off 와 같은 로짓). 샘플링과, 생성기가 언어를 정하기 전에 불린 계산에 씁니다.
/// 모델을 올릴 때 미리 문서 언어의 행을 모으지 않습니다: 첫 생성이 샘플링이면 곧바로 전체 행으로 바꿔 모으기만 두 번이 되고,
/// 탐욕 생성이면 [`refresh_for`] 가 그때 그 언어로 모읍니다.
pub fn all_rows(src: &TableSource, device: &Device) -> Result<TrimmedHead> {
    TrimmedHead::build(src, Arc::new((0..src.rows as u32).collect()), all_rows_label(), device)
}

/// 지금의 문서 언어로 남긴 행을 맞춥니다 (규칙 · 언어가 바뀌었으면 다시 모음). 규칙이 `off` 면 비웁니다.
pub fn refresh(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device) -> Result<()> {
    refresh_for(slot, src, tok, device, "", false)
}

/// [`refresh`] + 이번 생성의 프롬프트 글에 나온 문자 체계. 읽기는 다국어 전부라 프롬프트(질의 · 검색된 기록 · 문서)에 어떤 글자든
/// 올 수 있고, 모델은 그 값을 그대로 베껴 씁니다 (예: 한국어 질의에 일본 쇼핑몰 기록 → 상품 이름은 가나). 그래서 문서 언어 + 영어에
/// 프롬프트에 나온 문자 체계를 더해야 베낀 값이 잘리지 않습니다. `LOGIS_VOCAB_LANGS` 로 언어를 고정했으면 더하지 않습니다.
///
/// `sampling` (온도 > 0) 이면 전체 행으로 곱합니다. 잘린 행을 −∞ 로 두면 남은 토큰의 확률이 다시 정규화되어, top-p 의 누적 확률 경계에
/// 걸린 후보가 들어가거나 빠지고 같은 난수로도 다른 토큰을 뽑습니다 (N:N 검색에서 0.6B 가 `"percent_total": "20.0"` 대신 `"20.5"`).
/// 탐욕 디코딩은 off 가 고르는 토큰이 남긴 행에 있으면 결과가 같으므로 잘라 냅니다.
pub fn refresh_for(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device, prompt: &str, sampling: bool) -> Result<()> {
    let rule = rule_from_env();
    if rule == TrimRule::Off {
        *slot = None;
        return Ok(());
    }
    let fixed = std::env::var(LANGS_ENV).is_ok_and(|v| !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto"));
    let langs = if fixed { current_langs() } else { current_langs().union(Langs::present(prompt)) };
    let compact_k = compact_active_k(src.rows, langs);

    // Direct artifact execution: if explicitly requested (`LOGIS_VOCAB_EXEC_MODE=artifact_direct`),
    // or if the artifact is already compact (`<= 16,384` rows) and no smaller compact_k is requested,
    // or during sampling (`temperature > 0`), load all artifact rows directly without slicing.
    let force_all = sampling
        || exec_mode_override() == Some(VocabExecMode::ArtifactDirect)
        || exec_mode_override() == Some(VocabExecMode::FullUntrimmed)
        || (src.rows <= 16_384 && compact_k.is_none());
    if force_all {
        if slot.as_ref().is_some_and(|h| h.kept.len() == src.rows) {
            return Ok(());
        }
        *slot = None;
        *slot = Some(all_rows(src, device)?);
        return Ok(());
    }

    let label = match compact_k {
        Some(k) => format!("compact({k}):{}", langs.label()),
        None => format!("{}:{}", rule.label(), langs.label()),
    };
    // When compact_k is None, label match is sufficient; when compact_k is Some, we also ensure
    // any prompt tokens are covered (checked via sorted_subset below).
    if compact_k.is_none() && slot.as_ref().is_some_and(|h| h.label == label) {
        return Ok(());
    }
    let ids = ids_for_with_prompt(tok, src.rows, rule, langs, prompt)?;
    // 지금 올린 행이 이번에 남길 행을 모두 담고 많아야 10 % 더 크면 그대로 씁니다.
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
        // 판정은 en(whatlang 의 폴란드어 → en)이어도 글의 라틴 확장 글자는 남김
        let pl = langs_of_document("Zamówienie nr 20231, łączna wartość");
        assert!(pl.latin && !pl.hangul, "{pl:?}");
        assert_eq!(Langs::from_codes(&["all"]).unwrap(), Langs::all());
        let ko_hanja = langs_of_document("급수 펌프(給水) 압력 주문");
        assert!(ko_hanja.hangul && ko_hanja.hanja && !ko_hanja.cjk, "{ko_hanja:?}");
        let zh = langs_of_document("给水泵压力");
        assert!(zh.cjk && !zh.hanja, "{zh:?}");
        // keeps_script(Kind::Other) 는 premium-discount-quant 안의 비공개 함수라 그쪽 kinds_and_languages 시험이 봅니다.
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

    /// 두 흐름(검색 · 문서 추출)이 동시에 범위를 열어도 각자의 언어를 쓰고, 범위가 없는 흐름은 둘의 합집합을 씁니다
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
        // 어휘 4 × 열 32 Q8_0 표에서 1 · 3 행만
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
        // 입력 임베딩: 남긴 행이면 표 전체의 임베딩과 비트까지 같고, 잘린 행이 섞이면 꺼내지 않음
        let want: Vec<f32> = src.gather(&[3, 1], &dev)?.dequantize(&dev)?.flatten_all()?.to_vec1()?;
        let emb: Vec<f32> = head.embed(&[3, 1], &dev).expect("남긴 행")?.flatten_all()?.to_vec1()?;
        assert_eq!(emb, want);
        assert!(head.embed(&[1, 2], &dev).is_none() && head.embed(&[9], &dev).is_none());
        // 남긴 행 · 잘린 행이 섞인 입력: 남긴 행은 투영에서, 잘린 행은 매핑에서 꺼내 원래 순서로 → 매핑만 쓴 것과 같음
        let table = crate::models::qwen3_5::model::EmbedTable::Mapped { src: src.clone(), device: dev.clone() };
        let ids = Tensor::new(&[[3u32, 2, 1, 0]], &dev)?;
        let mixed: Vec<f32> = table.forward_with_head(&ids, Some(&head))?.flatten_all()?.to_vec1()?;
        let plain: Vec<f32> = table.forward(&ids)?.flatten_all()?.to_vec1()?;
        assert_eq!(mixed, plain);
        // 어휘 블록(의미 선입견): 남긴 행은 같은 값, 잘린 행 자리는 남긴 행 0 번(= 어휘 1)으로 채움 (노름 0 → NaN 방지)
        let via_head = table.rows_f32_with_head(0, 4, Some(&head))?.to_vec2::<f32>()?;
        let mapped = table.rows_f32(0, 4)?.to_vec2::<f32>()?;
        assert_eq!((&via_head[1], &via_head[3]), (&mapped[1], &mapped[3]));
        assert_eq!((&via_head[0], &via_head[2]), (&mapped[1], &mapped[1]));
        // 선입견의 잘린 행 자리는 +∞: 그 자리 로짓이 (앞으로 어떤 코드로) 유한값이 되어 있어도 빼면 −∞, 남긴 행은 그대로
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
        // 전체 행(이어진 행을 한 번에 복사, 펼치기 없음)은 원래 투영과 비트까지 같음
        let all = TrimmedHead::build(&src, Arc::new((0..4).collect()), "all".into(), &dev)?;
        let got_all: Vec<f32> = all.logits(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!(got_all, full);
        // 이어진 일부 행도 한 번에 복사
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
        let head = TrimmedHead::build(&src, Arc::new(kept.clone()), "cg-test".into(), &dev)?;
        let full_w = q.dequantize(&dev)?.to_vec2::<f32>()?;

        // Target vector for nonnegative prejudice test
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

        // Test 12 hidden vectors including vectors aligned directly with cold rows (>128)
        for trial in 0..12usize {
            let h_vec: Vec<f32> = if trial % 2 == 0 {
                // Force cold-row win
                full_w[150 + trial * 25].clone()
            } else {
                (0..cols).map(|c| ((trial * 29 + c * 5) as f32 * 0.23).cos()).collect()
            };
            let h_t = Tensor::from_vec(h_vec.clone(), (1, cols), &dev)?;

            // 1. Pure greedy parity
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

            // 2. Constrained greedy with nonnegative prejudice parity
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
