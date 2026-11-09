//! Stanza 실행기 (stanza-onnx-export format 2 변환본, 12개 언어).
//!
//! 규약은 stanza-onnx-export 의 `verify_onnx.py` RefRuntime 과 같습니다. 그 스크립트가 원문 → 토큰 · 문장 → MWT 펼침 →
//! 품사(upos · xpos · feats) → 원형 → 의존 구문(머리 · 관계)을 PyTorch Stanza 전체 파이프라인과 대조하고,
//! 이 파일은 같은 규약을 Rust 로 옮겼습니다 (`tests/it/stanza_runtime.rs` 가 같은 기준 JSON 으로 대조).
//!
//! 이전 실행기(2026-07 변환본용)는 변환본 결함(사전학습 사전 없음 · 문자 투영 누락 · charlm 순서 · 원형 디코더 길이 고정)
//! 때문에 품사가 무작위 수준이라 `pos_reliable` 로 막혀 있었고, 토크나이저 특징을 0 으로 넣었으며, 질의마다 세션 네 개를
//! 새로 열었습니다. 모델은 HF 에 없는 주소에서 받으려 했습니다.
//!
//! 모델 폴더: `LOGIS_STANZA_DIR` 이 있으면 그 아래 `<lang>/`, 없으면 앱 폴더 `models/stanza/<lang>/`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use ndarray::{ArrayD, IxDyn};
use onnxruntime::mixed::{DynInput, SessionMixedExt};
use onnxruntime::session::Session;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

const PAD: i64 = 0;
const UNK: i64 = 1;
const EMPTY: i64 = 2;
const ROOT: i64 = 3;
/// UD 관계 사전 앞의 특수 칸 (PAD · UNK · EMPTY · ROOT)
const VOCAB_PREFIX_SIZE: usize = 4;
/// 변환본이 지원하는 언어 (stanza-onnx-export LANGS)
pub const LANGS: [&str; 12] = ["en", "de", "es", "fr", "ja", "pt", "ar", "cs", "it", "ko", "nl", "zh-hans"];
/// 한 번에 메모리에 둘 언어 수. 언어 하나가 세션 · 사전으로 0.4~0.9 GB(CPU RAM)라 하나만 둡니다.
/// 같은 언어로 이어 검색하면 다시 열지 않고, 언어가 바뀌면 이전 언어를 내리고 1~4초 걸려 새로 엽니다.
/// (이전 실행기는 질의마다 세션 네 개를 새로 열었습니다.)
const CACHE_LANGS: usize = 1;
const ORT_THREADS: i16 = 4;

pub static STANZA_ENV: Lazy<&'static onnxruntime::environment::Environment> = Lazy::new(|| {
    Box::leak(Box::new(
        onnxruntime::environment::Environment::builder()
            .with_name("stanza_global_env")
            .build()
            .expect("Failed to initialize global ONNX Runtime Environment"),
    ))
});

// ---------------------------------------------------------------------------------------------
// 언어 코드
// ---------------------------------------------------------------------------------------------

/// 앱 언어 이름 · ISO 코드 → 변환본 언어 코드. 12개 언어가 아니면 None.
pub fn lang_code(language: &str) -> Option<&'static str> {
    let l = language.trim().to_lowercase();
    Some(match l.as_str() {
        "korean" | "ko" | "kor" => "ko",
        "english" | "en" | "eng" => "en",
        "japanese" | "ja" | "jpn" => "ja",
        "chinese" | "zh" | "zh-hans" | "zh-cn" | "cmn" | "zh-hant" | "zh-tw" | "zh-hk" => "zh-hans",
        "french" | "fr" | "fra" => "fr",
        "german" | "de" | "deu" => "de",
        "spanish" | "es" | "spa" => "es",
        "italian" | "it" | "ita" => "it",
        "portuguese" | "pt" | "por" => "pt",
        "dutch" | "nl" | "nld" => "nl",
        "arabic" | "ar" | "ara" => "ar",
        "czech" | "cs" | "ces" => "cs",
        _ => return None,
    })
}

/// 라틴 문자 변환본 언어 (whatlang 후보를 이 여덟으로 제한)
const LATIN_LANGS: [(whatlang::Lang, &str); 8] = [
    (whatlang::Lang::Eng, "en"),
    (whatlang::Lang::Deu, "de"),
    (whatlang::Lang::Spa, "es"),
    (whatlang::Lang::Fra, "fr"),
    (whatlang::Lang::Ita, "it"),
    (whatlang::Lang::Por, "pt"),
    (whatlang::Lang::Nld, "nl"),
    (whatlang::Lang::Ces, "cs"),
];

static LATIN_DETECTOR: Lazy<whatlang::Detector> =
    Lazy::new(|| whatlang::Detector::with_allowlist(LATIN_LANGS.iter().map(|(l, _)| *l).collect()));

/// 도메인 사전 언어 구역(필드 라벨 · 뜻 · 편견 문구)의 낱말 → 그 낱말이 든 라틴 언어 구역들.
/// 8개 라틴 언어 구역이 모두 있는 사전(배포 사전 · full-bias)에서만 만듭니다. 개발 사전(ko · en 구역만)에서는 None 이라 투표하지 않습니다.
static LATIN_VOCAB: Lazy<Option<HashMap<String, Vec<&'static str>>>> = Lazy::new(|| {
    let dict: &Value = &crate::parsing::BIAS_DICT;
    let mut map: HashMap<String, Vec<&'static str>> = HashMap::new();
    for (_, code) in LATIN_LANGS.iter() {
        let mut words: HashSet<String> = HashSet::new();
        collect_vocab(dict.get(*code)?, &mut words);
        for w in words {
            map.entry(w).or_default().push(*code);
        }
    }
    Some(map)
});

/// 사전 투표 낱말: 세 글자 이상, 숫자 · 밑줄 없는 낱말 (소문자)
fn vocab_words(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().count() >= 3 && !w.chars().any(|c| c.is_numeric() || c == '_'))
        .map(|w| w.to_lowercase())
}

fn collect_vocab(v: &Value, out: &mut HashSet<String>) {
    match v {
        Value::String(s) => out.extend(vocab_words(s)),
        Value::Array(a) => a.iter().for_each(|x| collect_vocab(x, out)),
        Value::Object(o) => o.values().for_each(|x| collect_vocab(x, out)),
        _ => {}
    }
}

/// 라틴 문자 질의의 사전 투표: 질의 낱말이 든 언어 구역마다 1 / (그 낱말이 든 구역 수)를 더해, 1위 언어의 낱말이 둘 이상이고
/// 2위보다 1 이상 앞설 때만 그 언어입니다. whatlang 은 짧은 질의에서 cs → pt · nl → fr · de → en · fr → es 로 자주 틀렸습니다
/// (v2 질의 라틴 8개 언어 184개 중 16개). 투표가 그중 9개를 바로잡고 새로 틀린 것은 없었습니다 (쇼핑몰 이름 Copperleaf Goods 의
/// goods 처럼 낱말 하나뿐인 근거는 쓰지 않습니다).
fn latin_vocab_vote(text: &str) -> Option<&'static str> {
    let vocab = LATIN_VOCAB.as_ref()?;
    let mut score: Vec<(&'static str, f32, usize)> = LATIN_LANGS.iter().map(|(_, c)| (*c, 0.0, 0)).collect();
    let mut seen: HashSet<String> = HashSet::new();
    for w in vocab_words(text) {
        if !seen.insert(w.clone()) {
            continue;
        }
        if let Some(owners) = vocab.get(&w) {
            for s in score.iter_mut().filter(|s| owners.contains(&s.0)) {
                s.1 += 1.0 / owners.len() as f32;
                s.2 += 1;
            }
        }
    }
    score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (top, first, hits) = score[0];
    (hits >= 2 && first - score[1].1 >= 0.999).then_some(top)
}

fn is_latin_letter(c: char) -> bool {
    c.is_alphabetic() && matches!(c as u32, 0x0041..=0x024F | 0x1E00..=0x1EFF)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Script {
    Hangul,
    Kana,
    Han,
    Arabic,
    Latin,
    Other,
}

impl Script {
    fn of(c: char) -> Option<Script> {
        Some(match c as u32 {
            0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F => Script::Hangul,
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Script::Kana,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => Script::Han,
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
            _ if is_latin_letter(c) => Script::Latin,
            _ if c.is_alphabetic() => Script::Other,
            _ => return None,
        })
    }

    fn cjk(self) -> bool {
        matches!(self, Script::Hangul | Script::Kana | Script::Han)
    }
}

/// 따옴표 짝: 여는 글자 → 닫는 글자들. `'` · `"` 는 낱말 경계에서만 짝으로 봅니다 (l'opération · dell'ordine 의 아포스트로피 제외).
const QUOTE_PAIRS: [(char, &str); 10] = [
    ('\'', "'"),
    ('"', "\""),
    ('‘', "’"),
    ('“', "”"),
    ('„', "“”"),
    ('‚', "‘’"),
    ('«', "»"),
    ('‹', "›"),
    ('「', "」"),
    ('『', "』"),
];

/// 따옴표로 묶은 구의 바이트 범위 (따옴표 글자 포함). 행사 · 쿠폰 · 상품 이름을 그대로 적은 값입니다.
pub fn quoted_ranges(query: &str) -> Vec<std::ops::Range<usize>> {
    let chars: Vec<(usize, char)> = query.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, c) = chars[i];
        if let Some((_, closers)) = QUOTE_PAIRS.iter().find(|(o, _)| *o == c) {
            let ascii = c == '\'' || c == '"';
            if !ascii || i == 0 || !chars[i - 1].1.is_alphanumeric() {
                let close = (i + 2..chars.len()).find(|&j| {
                    closers.contains(chars[j].1) && (!ascii || j + 1 == chars.len() || !chars[j + 1].1.is_alphanumeric())
                });
                if let Some(j) = close {
                    out.push(start..chars[j].0 + chars[j].1.len_utf8());
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// 따옴표로 묶은 구를 공백으로 바꿉니다. 질의 문장의 언어 · 기간은 따옴표 밖 글로 정합니다.
/// 따옴표 밖에 글자가 하나도 없으면(이름만 친 질의) 원문을 그대로 냅니다.
pub fn strip_quoted(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut at = 0;
    for r in quoted_ranges(query) {
        out.push_str(&query[at..r.start]);
        out.push(' ');
        at = r.end;
    }
    out.push_str(&query[at..]);
    if out.chars().any(|c| c.is_alphabetic()) {
        out
    } else {
        query.to_string()
    }
}

/// 질의를 나눈 낱말(Stanza 토큰 · 공백 어절)마다 따옴표 안에 있는지. 낱말을 원문에서 차례로 찾아 맞추고,
/// 원문에 그대로 없는 낱말(MWT 풀이 du → de le 등)은 따옴표 밖으로 봅니다.
pub fn quoted_token_flags(query: &str, tokens: &[String]) -> Vec<bool> {
    let ranges = quoted_ranges(query);
    let mut at = 0;
    tokens
        .iter()
        .map(|t| match (!t.is_empty()).then(|| query[at..].find(t.as_str())).flatten() {
            Some(off) => {
                let start = at + off;
                at = start + t.len();
                ranges.iter().any(|r| r.start <= start && at <= r.end)
            }
            None => false,
        })
        .collect()
}

/// 간체에만 쓰는 한자 (일본어 신자체와 모양이 다른 것만: 単/单 · 発/发 · 記/记 …). 가나가 섞인 글이
/// 일본어 이름(こもれび商店 · 株式会社ひなた工房)이 든 중국어 문장인지 가립니다.
const SIMPLIFIED_ONLY: &str = "单发运购订货车录页选结进领优费详记关实东们这么吗呢给让说话时间门问题对应开买卖务产业场报价钱银账码张网认证书动华汇过还达递邮输转统计设备线级类样历现为处见观视规则质从项阳伞";

/// 라틴 · 아랍 문장에 든 한중일 이름 어절을 뺀 글. 문장 낱말(숫자 · 대문자 약어 · 코드를 뺀 라틴 · 아랍 낱말)이
/// 셋 이상이고 한중일 글자가 든 어절의 두 배 이상일 때만 뺍니다 (products 이예준 wrote a review for →
/// products wrote a review for). 한국어 · 일본어 문장에 든 영어 이름(Felix Wagner 고객이 …)은 그대로 둡니다.
fn drop_cjk_names(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let has_cjk = |t: &str| t.chars().filter_map(Script::of).any(Script::cjk);
    let (mut latin_words, mut arabic_words, mut cjk_tokens) = (0usize, 0usize, 0usize);
    for t in &tokens {
        let scripts: Vec<Script> = t.chars().filter_map(Script::of).collect();
        if scripts.is_empty() {
            continue;
        }
        if scripts.iter().any(|s| s.cjk()) {
            cjk_tokens += 1;
        } else if t.chars().any(|c| c.is_ascii_digit()) {
            continue;
        } else if scripts.iter().all(|s| *s == Script::Latin) {
            let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
            if !(letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase())) {
                latin_words += 1;
            }
        } else if scripts.iter().all(|s| *s == Script::Arabic) {
            arabic_words += 1;
        }
    }
    let sentence_words = latin_words.max(arabic_words);
    if cjk_tokens > 0 && sentence_words >= 3 && sentence_words >= 2 * cjk_tokens {
        tokens.into_iter().filter(|t| !has_cjk(t)).collect::<Vec<_>>().join(" ")
    } else {
        text.to_string()
    }
}

/// 질의 글에서 변환본 언어를 고릅니다. 12개 언어가 아니면(키릴 · 태국 문자 …) None.
///  · 따옴표로 묶은 이름은 빼고, 라틴 · 아랍 문장에 든 한중일 이름 어절도 뺀 글로 정합니다. 데이터 값(행사 「夏休み準備セール」 ·
///    고객 이예준 · 쇼핑몰 こもれび商店)의 문자 체계가 문장 언어를 덮어 영어 질의가 일본어 사전 · 변환본으로 풀리던 것을 막습니다
///    (v2 관계형 합성 데이터 커머스 질의 120개 중 32개).
///  · 문자 체계: 한글 → ko, 가나 → ja, 가나 없는 한자 → zh-hans (앱 언어가 일본어면 ja), 아랍 문자 → ar
///    한글은 글자 수가 한자 · 가나보다 적어도 어절 끝(조사 · 어미 자리)이 더 많이 한글이면 ko (株式会社ひなた工房에서 보낸 선하증권),
///    가나가 있어도 간체에만 쓰는 한자가 둘 이상이면 zh-hans (在こもれび商店商品详情页更改选项的记录)
///  · 라틴 문자: 그 언어에만 쓰는 글자(ě š č ř ž ů ť ď ň → cs, ã õ → pt, ñ ¿ ¡ → es, ß → de)를 먼저 보고, 다음으로 도메인 사전
///    언어 구역 낱말 투표(배포 사전에서만, [`latin_vocab_vote`]), 아니면 whatlang 을 8개 라틴 언어 안에서 돌립니다. 후보를 넓게 두면
///    짧은 질의가 라틴어 · 슬로바키아어 · 아프리칸스어로 판정되어 영어로 떨어졌습니다 (12개 언어 질의 144개 중 26개).
///    판정을 믿을 수 없고 앱 언어가 라틴 문자 언어면 앱 언어.
pub fn detect_query_lang(query: &str, ui_language: &str) -> Option<&'static str> {
    let text = drop_cjk_names(&strip_quoted(query));
    let query = text.as_str();
    let (mut hangul, mut kana, mut han, mut arabic, mut latin, mut other) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for c in query.chars() {
        match Script::of(c) {
            Some(Script::Hangul) => hangul += 1,
            Some(Script::Kana) => kana += 1,
            Some(Script::Han) => han += 1,
            Some(Script::Arabic) => arabic += 1,
            Some(Script::Latin) => latin += 1,
            Some(Script::Other) => other += 1,
            None => {}
        }
    }
    // 어절 끝 한중일 글자의 문자 체계: 한국어는 조사 · 어미가 어절 끝에 붙습니다 (…工房에서 · Goods에서)
    let (mut hangul_tail, mut kanji_tail) = (0usize, 0usize);
    for t in query.split_whitespace() {
        match t.chars().rev().filter_map(Script::of).find(|s| s.cjk()) {
            Some(Script::Hangul) => hangul_tail += 1,
            Some(_) => kanji_tail += 1,
            None => {}
        }
    }
    if hangul > 0 && (hangul >= kana + han || hangul_tail > kanji_tail) {
        return Some("ko");
    }
    if kana > 0 {
        let simplified = query.chars().filter(|c| SIMPLIFIED_ONLY.contains(*c)).count();
        return Some(if simplified >= 2 { "zh-hans" } else { "ja" });
    }
    if han > 0 {
        // 가나 없는 한자만: 앱 언어가 일본어면 일본어, 아니면 중국어
        return Some(if lang_code(ui_language) == Some("ja") { "ja" } else { "zh-hans" });
    }
    if arabic > 0 && arabic >= latin {
        return Some("ar");
    }
    if other > latin {
        return None;
    }
    let lower = query.to_lowercase();
    for (letters, code) in [("ěščřžůťďň", "cs"), ("ãõ", "pt"), ("ñ¿¡", "es"), ("ß", "de")] {
        if lower.chars().any(|c| letters.contains(c)) {
            return Some(code);
        }
    }
    if let Some(code) = latin_vocab_vote(query) {
        return Some(code);
    }
    let ui = lang_code(ui_language).filter(|c| LATIN_LANGS.iter().any(|(_, l)| l == c));
    match LATIN_DETECTOR.detect(query) {
        Some(info) => {
            let code = LATIN_LANGS.iter().find(|(l, _)| *l == info.lang()).map(|(_, c)| *c).unwrap_or("en");
            // 짧은 질의는 whatlang 이 자주 '믿을 수 없음' 입니다. 그때 앱 언어가 라틴 문자 언어면 앱 언어를 따릅니다.
            if info.is_reliable() || ui.is_none() {
                Some(code)
            } else {
                ui
            }
        }
        None => Some(ui.unwrap_or("en")),
    }
}

/// 공백으로 낱말을 나누지 않는 언어 (질의 단위를 Stanza 토큰으로 씀)
pub fn lang_without_spaces(lang: &str) -> bool {
    matches!(lang, "ja" | "zh-hans")
}

/// 앱 모델 목록 이름(`stanza_<이름>`) ↔ 변환본 언어 코드
pub const UI_LANGS: [(&str, &str); 12] = [
    ("korean", "ko"),
    ("english", "en"),
    ("japanese", "ja"),
    ("chinese", "zh-hans"),
    ("french", "fr"),
    ("german", "de"),
    ("spanish", "es"),
    ("italian", "it"),
    ("portuguese", "pt"),
    ("dutch", "nl"),
    ("arabic", "ar"),
    ("czech", "cs"),
];

/// 변환본을 받는 Hugging Face 저장소 (이전 이름 PopupLink/stanza-<lang> 은 여기로 넘어감)
pub const HF_REPO_PREFIX: &str = "https://huggingface.co/Qntology/stanza-";
/// format 2 변환본을 두는 main 의 폴더. 루트의 format 1 파일은 이미 배포된 앱이 PopupLink 주소
/// (→ Qntology 리다이렉트)로 받으므로 그대로 둡니다.
pub const HF_REPO_SUBDIR: &str = "format2";

/// 언어 하나의 변환본 주소. 이 아래의 vocab.json 이 받을 파일 목록입니다.
pub fn remote_base(code: &str) -> String {
    format!("{HF_REPO_PREFIX}{code}/resolve/main/{HF_REPO_SUBDIR}")
}

/// vocab.json 의 `files` (처리기별 파일 목록)을 펼칩니다. vocab.json 자신은 넣지 않습니다.
pub fn manifest_files(vocab: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(o) = vocab.get("files").and_then(|f| f.as_object()) {
        for v in o.values() {
            for f in units(v) {
                if !f.is_empty() && !out.contains(&f) {
                    out.push(f);
                }
            }
        }
    }
    out
}

/// 폴더에 format 2 vocab.json 과 그 파일 목록이 모두 있는가
pub fn is_installed(dir: &Path) -> bool {
    let v = match read_json(&dir.join("vocab.json")) {
        Ok(v) => v,
        Err(_) => return false,
    };
    v.get("format").and_then(|x| x.as_i64()) == Some(2) && manifest_files(&v).iter().all(|f| dir.join(f).exists())
}

/// 원격 저장소의 vocab.json 을 읽어 받을 파일 목록을 냅니다 (vocab.json 은 맨 끝). 이전 판(format 1)이면 Err.
pub async fn remote_manifest(base_url: &str) -> Result<Vec<String>> {
    let url = format!("{base_url}/vocab.json");
    let res = reqwest::get(&url).await.with_context(|| format!("{url} 요청"))?;
    if !res.status().is_success() {
        return Err(anyhow!("{url}: HTTP {}", res.status()));
    }
    let v: Value = serde_json::from_slice(&res.bytes().await?).with_context(|| format!("{url} 해석"))?;
    if v.get("format").and_then(|x| x.as_i64()) != Some(2) {
        return Err(anyhow!(
            "{base_url} 의 Stanza 변환본이 이전 판(format 1)입니다. stanza-onnx-export 의 onnx_final/<lang> (format 2)을 올려야 합니다"
        ));
    }
    let mut files = manifest_files(&v);
    files.push("vocab.json".to_string());
    Ok(files)
}

pub fn models_root() -> PathBuf {
    match std::env::var("LOGIS_STANZA_DIR") {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d),
        _ => crate::utils::get_app_dir().join("models").join("stanza"),
    }
}

// ---------------------------------------------------------------------------------------------
// 사전 조각
// ---------------------------------------------------------------------------------------------

fn units(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default()
}

fn str_index(units: &[String]) -> HashMap<String, i64> {
    units.iter().enumerate().map(|(i, u)| (u.clone(), i as i64)).collect()
}

/// 한 글자 단위만 (특수 단위 `<PAD>` 등은 글자로 찾을 일이 없음)
fn char_index(units: &[String]) -> HashMap<char, i64> {
    let mut m = HashMap::new();
    for (i, u) in units.iter().enumerate() {
        let mut it = u.chars();
        if let (Some(c), None) = (it.next(), it.next()) {
            m.insert(c, i as i64);
        }
    }
    m
}

/// {"xpos": ["_"]} → 이름별 집합
fn ignores(entry: &Value) -> HashMap<String, HashSet<String>> {
    entry
        .get("ignore")
        .and_then(|x| x.as_object())
        .map(|o| o.iter().map(|(k, v)| (k.clone(), units(v).into_iter().collect())).collect())
        .unwrap_or_default()
}

fn as_bool(v: &Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn as_usize(v: &Value, key: &str) -> usize {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0) as usize
}

fn as_string(v: &Value, key: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// stanza CompositeVocab (xpos · feats): 칸마다 따로 예측하고 sep 로 잇습니다.
#[derive(Debug, Clone)]
pub struct Composite {
    sep: String,
    keyed: bool,
    keys: Vec<String>,
    units: Vec<Vec<String>>,
    ids: Vec<HashMap<String, i64>>,
}

impl Composite {
    pub fn from_json(v: &Value) -> Option<Self> {
        let parts = v.get("parts")?.as_array()?;
        let mut keys = Vec::new();
        let mut us = Vec::new();
        for p in parts {
            let p = p.as_array()?;
            keys.push(p.first()?.as_str()?.to_string());
            us.push(units(p.get(1)?));
        }
        let ids = us.iter().map(|u| str_index(u)).collect();
        Some(Self { sep: as_string(v, "sep"), keyed: as_bool(v, "keyed"), keys, units: us, ids })
    }

    fn split(&self, unit: &str) -> Vec<String> {
        if self.sep.is_empty() {
            unit.chars().map(|c| c.to_string()).collect()
        } else {
            unit.split(self.sep.as_str()).map(|s| s.to_string()).collect()
        }
    }

    pub fn unit2id(&self, unit: &str) -> Vec<i64> {
        let parts = self.split(unit);
        if self.keyed {
            let mut map: HashMap<String, String> = HashMap::new();
            if !(parts.len() == 1 && parts[0] == "_") {
                for p in &parts {
                    if let Some((k, v)) = p.split_once('=') {
                        map.insert(k.to_string(), v.to_string());
                    }
                }
            }
            self.keys
                .iter()
                .enumerate()
                .map(|(i, k)| match map.get(k) {
                    Some(v) => *self.ids[i].get(v).unwrap_or(&UNK),
                    None => EMPTY,
                })
                .collect()
        } else {
            let parts = if unit == "_" { Vec::new() } else { parts };
            (0..self.keys.len())
                .map(|i| if i < parts.len() { *self.ids[i].get(&parts[i]).unwrap_or(&UNK) } else { EMPTY })
                .collect()
        }
    }

    pub fn id2unit(&self, ids: &[usize]) -> String {
        let mut items = Vec::new();
        for (i, &v) in ids.iter().enumerate() {
            if v as i64 == EMPTY {
                continue;
            }
            let u = self.units[i].get(v).cloned().unwrap_or_default();
            items.push(if self.keyed { format!("{}={}", self.keys[i], u) } else { u });
        }
        let s = items.join(&self.sep);
        if s.is_empty() {
            "_".to_string()
        } else {
            s
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

#[derive(Debug, Clone)]
enum TagSpec {
    List(Vec<String>),
    Composite(Composite),
}

impl TagSpec {
    fn from_json(v: &Value) -> Self {
        if v.is_object() {
            if let Some(c) = Composite::from_json(v) {
                return TagSpec::Composite(c);
            }
        }
        TagSpec::List(units(v))
    }
}

#[derive(Debug, Clone)]
struct CharLm {
    forward: HashMap<char, i64>,
    backward: HashMap<char, i64>,
    start: char,
    end: char,
}

impl CharLm {
    fn from_json(v: &Value) -> Option<Self> {
        if !v.is_object() {
            return None;
        }
        Some(Self {
            forward: char_index(&units(v.get("forward")?)),
            backward: char_index(&units(v.get("backward")?)),
            start: v.get("start")?.as_str()?.chars().next()?,
            end: v.get("end")?.as_str()?.chars().next()?,
        })
    }

    /// stanza build_char_representation: 문장 전체 글자열에서 낱말 끝 자리. 역방향은 낱말 순서와 글자를 모두 뒤집음.
    fn feed(&self, words: &[String], forward: bool) -> (Vec<i64>, Vec<i64>) {
        let ids = if forward { &self.forward } else { &self.backward };
        let seq: Vec<String> = if forward {
            words.to_vec()
        } else {
            words.iter().rev().map(|w| w.chars().rev().collect()).collect()
        };
        let mut chars = vec![self.start];
        let mut offsets = Vec::with_capacity(words.len());
        for w in &seq {
            chars.extend(w.chars());
            chars.push(self.end);
            offsets.push((chars.len() - 1) as i64);
        }
        if !forward {
            offsets.reverse();
        }
        (chars.iter().map(|c| *ids.get(c).unwrap_or(&UNK)).collect(), offsets)
    }
}

fn char_feed(words: &[String], ids: &HashMap<char, i64>) -> (ArrayD<i64>, ArrayD<i64>) {
    let width = words.iter().map(|w| w.chars().count()).max().unwrap_or(0).max(1);
    let mut out = ArrayD::<i64>::zeros(IxDyn(&[words.len(), width]));
    let mut lens = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        for (j, c) in w.chars().enumerate() {
            out[[i, j]] = *ids.get(&c).unwrap_or(&UNK);
        }
        lens.push(w.chars().count() as i64);
    }
    (out, ArrayD::from_shape_vec(IxDyn(&[words.len()]), lens).unwrap())
}

fn row_i64(v: Vec<i64>) -> ArrayD<i64> {
    let n = v.len();
    ArrayD::from_shape_vec(IxDyn(&[1, n]), v).unwrap()
}

fn vec_i64(v: Vec<i64>) -> ArrayD<i64> {
    let n = v.len();
    ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap()
}

/// np.argmax: 처음 나온 최댓값 자리
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

// ---------------------------------------------------------------------------------------------
// ONNX 세션
// ---------------------------------------------------------------------------------------------

struct Onnx {
    path: PathBuf,
    session: Option<Session<'static>>,
}

impl Onnx {
    fn new(path: PathBuf) -> Self {
        Self { path, session: None }
    }

    fn session(&mut self) -> Result<&mut Session<'static>> {
        if self.session.is_none() {
            // onnxruntime 0.0.14 의 with_model_from_file 은 경로 수명이 세션과 같아야 해서 한 번 고정합니다 (언어 캐시가 재사용).
            let p: &'static str = Box::leak(self.path.to_string_lossy().into_owned().into_boxed_str());
            let s = STANZA_ENV
                .new_session_builder()
                .map_err(|e| anyhow!("ONNX 세션 빌더: {e}"))?
                .with_number_threads(ORT_THREADS)
                .map_err(|e| anyhow!("ONNX 스레드 설정: {e}"))?
                .with_model_from_file(p)
                .map_err(|e| anyhow!("{} 열기 실패: {e}", self.path.display()))?;
            self.session = Some(s);
        }
        Ok(self.session.as_mut().unwrap())
    }

    /// 이름 붙은 입력(i64 · f32 섞음) → f32 출력 (소유 배열로 복사)
    fn run(&mut self, inputs: Vec<(&str, DynInput)>, outputs: &[&str]) -> Result<Vec<ArrayD<f32>>> {
        let path = self.path.clone();
        let s = self.session()?;
        let outs = s
            .run_mixed(inputs, outputs.to_vec())
            .map_err(|e| anyhow!("{} 실행 실패: {e}", path.display()))?;
        Ok(outs
            .iter()
            .map(|t| {
                let v: &ndarray::ArrayViewD<f32> = t;
                v.to_owned()
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------------------------
// 토크나이저 규약 (stanza tokenization/data.py · utils.py)
// ---------------------------------------------------------------------------------------------

static NUMERIC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[\d]+([,\.]+[\d]+)*[,\.]*$").unwrap());
static NEWLINE_WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n[\s\x{80}-\x{9f}]*\n").unwrap());
static STRUCTURAL: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    vec![
        ("labeled_field", Regex::new(r"\b[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}\s*:").unwrap()),
        ("phone_id", Regex::new(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b|\bx\d{3,5}\b").unwrap()),
        (
            "date_pattern",
            Regex::new(
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            )
            .unwrap(),
        ),
        ("currency", Regex::new(r"\$\s?\d[\d,]*(?:\.\d+)?").unwrap()),
    ]
});
/// 이메일 · URL (utils.py MASK_RE). URL 의 `(?:www\.|(?!www))` 앞보기는 regex 크레이트에 없어 `(?:www\.)?` 로 둡니다
/// ('http://wwwx.com' 처럼 www 로 시작하지만 'www.' 이 아닌 호스트만 다릅니다).
static MASK_RE: Lazy<Regex> = Lazy::new(|| {
    let email = r#"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"#;
    let url = r#"(?:https?://(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?://(?:www\.)?[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})|[a-zA-Z0-9]+\.(?:gov|org|edu|net|com|co)(?:\.[^\s"]{2,})"#;
    Regex::new(&format!("(?:{email}|{url})")).unwrap()
});

/// Python `re` 의 `[\s\u0080-\u009f]` (str.isspace 의 정보 구분 문자 \x1c-\x1f 포함)
fn is_stanza_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) || ('\u{80}'..='\u{9f}').contains(&c)
}

/// TokenizationDataset.__init__: 빈 줄로 문단, 오른쪽 공백 제거, 공백류는 ' ', 연속 공백은 하나
fn paragraphs(text: &str, skip_newline: bool) -> Vec<Vec<char>> {
    let mut out = Vec::new();
    for pt in NEWLINE_WHITESPACE_RE.split(text) {
        let pt = pt.trim_end_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
        if pt.is_empty() {
            continue;
        }
        let chars: Vec<char> = pt
            .chars()
            .filter(|c| !(skip_newline && *c == '\n'))
            .map(|c| if is_stanza_space(c) { ' ' } else { c })
            .collect();
        let mut kept = Vec::with_capacity(chars.len());
        for (i, &c) in chars.iter().enumerate() {
            if i > 0 && c == ' ' && chars[i - 1] == ' ' {
                continue;
            }
            kept.push(c);
        }
        out.push(kept);
    }
    out
}

struct Lexicon {
    words: HashSet<String>,
    prefixes: HashSet<String>,
    suffixes: HashSet<String>,
}

impl Lexicon {
    /// create_dictionary: 낱말 · 접두(마지막 글자 뺀) · 접미(첫 글자 뺀)
    fn new(list: Vec<String>) -> Self {
        let mut prefixes = HashSet::new();
        let mut suffixes = HashSet::new();
        for w in &list {
            let cs: Vec<char> = w.chars().collect();
            for i in 1..cs.len() {
                prefixes.insert(cs[..i].iter().collect::<String>());
                suffixes.insert(cs[cs.len() - i..].iter().collect::<String>());
            }
        }
        Self { words: list.into_iter().collect(), prefixes, suffixes }
    }

    /// extract_dict_feat: 첫 글자는 그대로, 붙이는 글자는 소문자
    fn feats(&self, para: &[char], idx: usize, n: usize) -> Vec<f32> {
        let mut fwd = vec![0f32; n];
        let mut bwd = vec![0f32; n];
        let mut fw = para[idx].to_string();
        let mut bw = para[idx].to_string();
        let (mut prefix, mut suffix) = (true, true);
        for win in 1..=n {
            if idx + win <= para.len() - 1 && prefix {
                fw.push_str(&para[idx + win].to_lowercase().to_string());
                fwd[win - 1] = if self.words.contains(&fw) { 1.0 } else { 0.0 };
                if !self.prefixes.contains(&fw) {
                    prefix = false;
                }
            }
            if idx >= win && suffix {
                bw = format!("{}{}", para[idx - win].to_lowercase(), bw);
                bwd[win - 1] = if self.words.contains(&bw) { 1.0 } else { 0.0 };
                if !self.suffixes.contains(&bw) {
                    suffix = false;
                }
            }
            if !prefix && !suffix {
                break;
            }
        }
        fwd.extend(bwd);
        fwd
    }
}

struct Tokenizer {
    onnx: Onnx,
    ids: HashMap<char, i64>,
    lower: bool,
    layout: Vec<String>,
    num_dict_feat: usize,
    lexicon: Option<Lexicon>,
    skip_newline: bool,
    replaces_spaces: bool,
    use_regex_tokens: bool,
    max_seqlen: usize,
}

impl Tokenizer {
    fn load(dir: &Path, e: &Value) -> Result<Self> {
        let lexicon = match e.get("lexicon_file").and_then(|v| v.as_str()) {
            Some(f) => {
                let raw = std::fs::read_to_string(dir.join(f)).with_context(|| format!("{f} 읽기"))?;
                let list: Vec<String> = serde_json::from_str(&raw)?;
                Some(Lexicon::new(list))
            }
            None => None,
        };
        Ok(Self {
            onnx: Onnx::new(dir.join("tokenizer.onnx")),
            ids: char_index(&units(&e["main"])),
            lower: as_bool(e, "lower"),
            layout: units(&e["feat_layout"]),
            num_dict_feat: as_usize(e, "num_dict_feat"),
            lexicon,
            skip_newline: as_bool(e, "skip_newline"),
            replaces_spaces: as_bool(e, "replaces_spaces"),
            use_regex_tokens: e.get("use_regex_tokens").and_then(|v| v.as_bool()).unwrap_or(true),
            max_seqlen: e.get("max_seqlen").and_then(|v| v.as_u64()).unwrap_or(1000) as usize,
        })
    }

    fn features(&self, para: &[char]) -> ArrayD<f32> {
        let n = para.len();
        let raw: String = para.iter().collect();
        // 구조 특징은 문단 글에서 정규식 자리(바이트) → 글자 자리
        let byte_to_char: Vec<usize> = {
            let mut v = vec![0usize; raw.len() + 1];
            let mut ci = 0;
            for (bi, ch) in raw.char_indices() {
                for k in 0..ch.len_utf8() {
                    v[bi + k] = ci;
                }
                ci += 1;
            }
            v[raw.len()] = ci;
            v
        };
        let mut masks: HashMap<&str, Vec<f32>> = HashMap::new();
        for (name, re) in STRUCTURAL.iter() {
            if !self.layout.iter().any(|l| l == name) {
                continue;
            }
            let mut mask = vec![0f32; n];
            for m in re.find_iter(&raw) {
                for p in byte_to_char[m.start()]..byte_to_char[m.end()] {
                    mask[p] = 1.0;
                }
            }
            masks.insert(name, mask);
        }
        let nd = self.num_dict_feat;
        let mut f = ArrayD::<f32>::zeros(IxDyn(&[1, n + 1, self.layout.len()]));
        for (i, &c) in para.iter().enumerate() {
            let dict = match (&self.lexicon, nd) {
                (Some(lex), nd) if nd > 0 => Some(lex.feats(para, i, nd)),
                _ => None,
            };
            for (k, name) in self.layout.iter().enumerate() {
                let v = match name.as_str() {
                    "space_before" => (c == ' ') as u8 as f32,
                    "capitalized" => c.is_uppercase() as u8 as f32,
                    "numeric" => NUMERIC_RE.is_match(&c.to_string()) as u8 as f32,
                    "end_of_para" => (i == n - 1) as u8 as f32,
                    "start_of_para" => (i == 0) as u8 as f32,
                    other => {
                        if let Some(m) = masks.get(other) {
                            m[i]
                        } else if let Some(k1) = other.strip_prefix("dict_forward_") {
                            dict.as_ref().map(|d| d[k1.parse::<usize>().unwrap_or(1) - 1]).unwrap_or(0.0)
                        } else if let Some(k1) = other.strip_prefix("dict_backward_") {
                            dict.as_ref().map(|d| d[nd + k1.parse::<usize>().unwrap_or(1) - 1]).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    }
                };
                f[[0, i, k]] = v;
            }
        }
        f
    }

    /// 문장마다 (토큰, 분류 1~4). 3 · 4 는 MWT 표시.
    fn tokenize(&mut self, text: &str) -> Result<Vec<Vec<(String, u8)>>> {
        let mut sents = Vec::new();
        for para in paragraphs(text, self.skip_newline) {
            let n = para.len();
            if n + 1 > self.max_seqlen {
                return Err(anyhow!("토크나이저 한 문단 최대 길이({})를 넘었습니다", self.max_seqlen));
            }
            let mut x: Vec<i64> = para
                .iter()
                .map(|c| {
                    if self.lower {
                        let l: String = c.to_lowercase().collect();
                        l.chars().next().and_then(|lc| self.ids.get(&lc).copied()).unwrap_or(UNK)
                    } else {
                        *self.ids.get(c).unwrap_or(&UNK)
                    }
                })
                .collect();
            x.push(PAD);
            let f = self.features(&para);
            let out = self.onnx.run(vec![("x", DynInput::I64(row_i64(x))), ("f", DynInput::F32(f))], &["output"])?;
            let o = &out[0];
            let classes = o.shape()[2];
            let mut pred: Vec<u8> = (0..n)
                .map(|i| {
                    let row: Vec<f32> = (0..classes).map(|c| o[[0, i, c]]).collect();
                    argmax(&row) as u8
                })
                .collect();
            let last = pred[n - 1];
            pred[n - 1] = if last < 2 { 2 } else if last > 2 { 4 } else { 2 };
            if self.use_regex_tokens {
                let raw: String = para.iter().collect();
                let byte_to_char: Vec<usize> = {
                    let mut v = vec![0usize; raw.len() + 1];
                    let mut ci = 0;
                    for (bi, ch) in raw.char_indices() {
                        for k in 0..ch.len_utf8() {
                            v[bi + k] = ci;
                        }
                        ci += 1;
                    }
                    v[raw.len()] = ci;
                    v
                };
                for m in MASK_RE.find_iter(&raw) {
                    let (s, e) = (byte_to_char[m.start()], byte_to_char[m.end()]);
                    if e == 0 || e <= s {
                        continue;
                    }
                    for p in s..e - 1 {
                        pred[p] = 0;
                    }
                    if pred[e - 1] == 0 {
                        pred[e - 1] = 1;
                    }
                }
            }
            let mut cur = String::new();
            let mut sent: Vec<(String, u8)> = Vec::new();
            for (c, p) in para.iter().zip(pred.iter()) {
                cur.push(*c);
                if *p >= 1 {
                    // normalize_token: 왼쪽 공백 제거, 공백류는 ' ', 중국어 · 일본어 · 한국어는 토큰 안 공백 제거
                    let t: String = cur.trim_start().chars().map(|ch| if ch.is_whitespace() { ' ' } else { ch }).collect();
                    let t = if self.replaces_spaces { t.replace(' ', "") } else { t };
                    cur.clear();
                    if t.is_empty() {
                        continue;
                    }
                    sent.push((t, *p));
                    if *p == 2 || *p == 4 {
                        sents.push(std::mem::take(&mut sent));
                    }
                }
            }
            if !sent.is_empty() {
                sents.push(sent);
            }
        }
        Ok(sents)
    }
}

// ---------------------------------------------------------------------------------------------
// MWT (토크나이저가 3 · 4 로 표시한 토큰만)
// ---------------------------------------------------------------------------------------------

enum MwtModel {
    Dict,
    Classifier(Onnx),
    Seq2Seq { enc: Onnx, dec: Onnx, max_dec_len: usize, oov_slots: usize },
}

struct Mwt {
    dict: HashMap<String, String>,
    dict_only: bool,
    ensemble_dict: bool,
    never_decode_unk: bool,
    chars: Vec<String>,
    ids: HashMap<char, i64>,
    vocab_size: usize,
    sos: i64,
    eos: i64,
    model: MwtModel,
}

/// mwt/trainer.py dict_expansion: 그대로 → 전부 대문자면 소문자로 찾아 대문자로 → 첫 글자만 대문자면 소문자로 찾아 첫 글자 대문자로
pub fn dict_expansion(d: &HashMap<String, String>, word: &str) -> Option<String> {
    if let Some(h) = d.get(word) {
        return Some(h.clone());
    }
    if py_isupper(word) {
        if let Some(h) = d.get(&lower(word)) {
            return Some(h.to_uppercase());
        }
    }
    let mut cs = word.chars();
    if let Some(first) = cs.next() {
        let rest: String = cs.collect();
        if first.is_uppercase() && py_islower(&rest) {
            if let Some(h) = d.get(&lower(word)) {
                let mut hc = h.chars();
                if let Some(f) = hc.next() {
                    return Some(format!("{}{}", f.to_uppercase(), hc.collect::<String>()));
                }
                return Some(h.clone());
            }
        }
    }
    None
}

/// str.isupper: 대소문자 있는 글자가 하나 이상이고 모두 대문자
fn py_isupper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}

/// str.islower: 대소문자 있는 글자가 하나 이상이고 모두 소문자
fn py_islower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            cased = true;
        }
    }
    cased
}

impl Mwt {
    fn load(dir: &Path, e: &Value) -> Result<Self> {
        let dict: HashMap<String, String> = match e.get("dict_file").and_then(|v| v.as_str()) {
            Some(f) => serde_json::from_str(&std::fs::read_to_string(dir.join(f)).with_context(|| format!("{f} 읽기"))?)?,
            None => HashMap::new(),
        };
        let chars = units(e.get("char").unwrap_or(&Value::Null));
        let model = match as_string(e, "type").as_str() {
            "classifier" => MwtModel::Classifier(Onnx::new(dir.join("mwt_classifier.onnx"))),
            "seq2seq" => MwtModel::Seq2Seq {
                enc: Onnx::new(dir.join("mwt_encoder.onnx")),
                dec: Onnx::new(dir.join("mwt_decoder.onnx")),
                max_dec_len: as_usize(e, "max_dec_len"),
                oov_slots: as_usize(e, "oov_slots"),
            },
            _ => MwtModel::Dict,
        };
        Ok(Self {
            dict,
            dict_only: as_bool(e, "dict_only"),
            ensemble_dict: as_bool(e, "ensemble_dict"),
            never_decode_unk: e.get("never_decode_unk").and_then(|v| v.as_bool()).unwrap_or(true),
            ids: char_index(&chars),
            chars,
            vocab_size: as_usize(e, "vocab_size"),
            sos: e.get("sos").and_then(|v| v.as_i64()).unwrap_or(2),
            eos: e.get("eos").and_then(|v| v.as_i64()).unwrap_or(3),
            model,
        })
    }

    /// [SOS] 글자… [EOS], 학습 사전에 없는 글자는 정렬 순서대로 V, V+1 … (DeltaVocab)
    fn src(&self, tok: &str, slots: usize) -> (Vec<i64>, Vec<char>) {
        let mut delta: Vec<char> = tok.chars().filter(|c| !self.ids.contains_key(c)).collect();
        delta.sort();
        delta.dedup();
        let mut src = vec![self.sos];
        for c in tok.chars() {
            let id = match self.ids.get(&c) {
                Some(i) => *i,
                None => match delta.iter().position(|d| *d == c) {
                    Some(p) if p < slots => (self.vocab_size + p) as i64,
                    _ => UNK,
                },
            };
            src.push(id);
        }
        src.push(self.eos);
        (src, delta)
    }

    fn model_expand(&mut self, tok: &str) -> Result<String> {
        match &mut self.model {
            MwtModel::Dict => Ok(tok.to_string()),
            MwtModel::Classifier(onnx) => {
                let (src, _) = {
                    let ids = &self.ids;
                    let mut s = vec![self.sos];
                    s.extend(tok.chars().map(|c| *ids.get(&c).unwrap_or(&UNK)));
                    s.push(self.eos);
                    (s, ())
                };
                let out = onnx.run(vec![("src_ids", DynInput::I64(row_i64(src)))], &["logits"])?;
                let lg = &out[0];
                let mut pieces = String::new();
                for (i, c) in tok.chars().enumerate() {
                    if lg[[0, i + 1, 1]] > lg[[0, i + 1, 0]] {
                        pieces.push(' ');
                    }
                    pieces.push(c);
                }
                Ok(pieces.trim().to_string())
            }
            MwtModel::Seq2Seq { .. } => self.seq2seq(tok),
        }
    }

    fn seq2seq(&mut self, tok: &str) -> Result<String> {
        let (slots, max_dec_len) = match &self.model {
            MwtModel::Seq2Seq { oov_slots, max_dec_len, .. } => (*oov_slots, *max_dec_len),
            _ => return Ok(tok.to_string()),
        };
        let (src, delta) = self.src(tok, slots);
        let v = self.vocab_size;
        let (sos, eos, never_unk) = (self.sos, self.eos, self.never_decode_unk);
        let MwtModel::Seq2Seq { enc, dec, .. } = &mut self.model else { unreachable!() };
        let e = enc.run(vec![("src_ids", DynInput::I64(row_i64(src.clone())))], &["h_in", "hn", "cn"])?;
        let h_in = e[0].clone();
        let (mut hn, mut cn) = (e[1].clone(), e[2].clone());
        let mut token = sos;
        let mut out: Vec<usize> = Vec::new();
        for _ in 0..max_dec_len {
            let d = dec.run(
                vec![
                    ("token", DynInput::I64(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![token]).unwrap())),
                    ("hn", DynInput::F32(hn.clone())),
                    ("cn", DynInput::F32(cn.clone())),
                    ("h_in", DynInput::F32(h_in.clone())),
                    ("src_ids", DynInput::I64(row_i64(src.clone()))),
                ],
                &["log_probs", "hn_out", "cn_out"],
            )?;
            let mut row: Vec<f32> = d[0].iter().copied().collect();
            if never_unk {
                row[UNK as usize] = f32::NEG_INFINITY;
            }
            let t = argmax(&row);
            hn = d[1].clone();
            cn = d[2].clone();
            if t as i64 == eos {
                break;
            }
            out.push(t);
            token = t as i64;
        }
        let pred: String = out
            .iter()
            .filter(|&&t| t >= 4)
            .filter_map(|&t| if t < v { self.chars.get(t).cloned() } else { delta.get(t - v).map(|c| c.to_string()) })
            .collect();
        // 펼침이 비었거나 공백이 없으면 원문 토큰 (mwt/trainer.py predict)
        Ok(if !pred.trim().is_empty() && pred.contains(' ') { pred } else { tok.to_string() })
    }

    fn expand(&mut self, tok: &str) -> Result<String> {
        if self.dict_only || self.ensemble_dict {
            if let Some(h) = dict_expansion(&self.dict, tok) {
                return Ok(h);
            }
            if self.dict_only {
                return Ok(tok.to_string());
            }
        }
        self.model_expand(tok)
    }
}

// ---------------------------------------------------------------------------------------------
// 품사 · 원형 · 의존 구문
// ---------------------------------------------------------------------------------------------

struct Pos {
    onnx: Onnx,
    word: HashMap<String, i64>,
    word_lower: bool,
    chars: HashMap<char, i64>,
    upos: Vec<String>,
    tag_order: Vec<String>,
    tags: HashMap<String, TagSpec>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    charlm: Option<CharLm>,
    /// WordVocab.ignore: 목록이 있는 태그는 EMPTY 번호를 '_' 로 냅니다 (xpos 를 학습하지 않은 fr · pt)
    ignore: HashMap<String, HashSet<String>>,
}

struct Lemma {
    dict: HashMap<String, HashMap<String, String>>,
    chars: Vec<String>,
    char_ids: HashMap<char, i64>,
    pos_ids: HashMap<String, i64>,
    caseless: bool,
    dict_only: bool,
    use_identity: bool,
    vocab_size: usize,
    oov_slots: usize,
    sos: i64,
    eos: i64,
    max_dec_len: usize,
    edit: bool,
    enc: Option<Onnx>,
    dec: Option<Onnx>,
}

struct Dep {
    onnx: Onnx,
    word: HashMap<String, i64>,
    word_lower: bool,
    lemma: HashMap<String, i64>,
    lemma_lower: bool,
    chars: HashMap<char, i64>,
    upos: HashMap<String, i64>,
    xpos: TagSpec,
    xpos_ids: HashMap<String, i64>,
    deprel: Vec<String>,
    inputs: Vec<String>,
    charlm: Option<CharLm>,
    root_word: String,
    /// WordVocab.ignore: 이 단위는 입력에서 EMPTY 번호 (UNK 가 아님)
    ignore: HashMap<String, HashSet<String>>,
}

/// 한 낱말 분석
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct WordAnalysis {
    pub text: String,
    pub upos: String,
    pub xpos: String,
    pub feats: String,
    pub lemma: String,
    /// 머리 낱말 번호 (1부터, 0 은 ROOT)
    pub head: usize,
    pub deprel: String,
}

/// 한 문장: 토큰(원문 조각)과 펼친 낱말. `token_words[i]` 는 토큰 i 의 낱말 번호 범위.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SentenceAnalysis {
    pub tokens: Vec<String>,
    pub token_words: Vec<(usize, usize)>,
    pub words: Vec<WordAnalysis>,
}

/// 한 언어의 변환본 (세션은 처음 쓸 때 엽니다)
pub struct StanzaLang {
    pub lang: String,
    pub dir: PathBuf,
    tokenizer: Tokenizer,
    mwt: Option<Mwt>,
    pos: Pos,
    pretrain: Option<HashMap<String, i64>>,
    lemma: Lemma,
    dep: Dep,
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("{} 읽기", path.display()))?;
    Ok(serde_json::from_str(&raw).with_context(|| format!("{} 해석", path.display()))?)
}

impl StanzaLang {
    pub fn load(dir: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let v = read_json(&dir.join("vocab.json"))?;
        if v.get("format").and_then(|x| x.as_i64()) != Some(2) {
            return Err(anyhow!(
                "{} 는 이전 변환본(format 1)입니다. stanza-onnx-export 의 final_definitive_conversion.py 로 다시 변환하세요",
                dir.display()
            ));
        }
        for key in ["tokenize", "pos", "lemma", "depparse"] {
            if v.get(key).is_none() {
                return Err(anyhow!("vocab.json 에 {key} 가 없습니다 ({})", dir.display()));
            }
        }
        let tokenizer = Tokenizer::load(dir, &v["tokenize"])?;
        let mwt = match v.get("mwt") {
            Some(e) => Some(Mwt::load(dir, e)?),
            None => None,
        };
        let pretrain = match v.get("pretrain").and_then(|p| p.get("file")).and_then(|f| f.as_str()) {
            Some(f) => {
                let list: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join(f)).with_context(|| format!("{f} 읽기"))?)?;
                Some(str_index(&list))
            }
            None => None,
        };
        let p = &v["pos"];
        let tags = p
            .get("tags")
            .and_then(|t| t.as_object())
            .map(|o| o.iter().map(|(k, x)| (k.clone(), TagSpec::from_json(x))).collect())
            .unwrap_or_default();
        let pos = Pos {
            onnx: Onnx::new(dir.join("pos.onnx")),
            word: str_index(&units(&p["word"])),
            word_lower: as_bool(p, "word_lower"),
            chars: char_index(&units(&p["char"])),
            upos: units(&p["upos"]),
            tag_order: units(&p["tag_order"]),
            tags,
            inputs: units(&p["inputs"]),
            outputs: units(&p["outputs"]),
            charlm: p.get("charlm").and_then(CharLm::from_json),
            ignore: ignores(p),
        };
        let l = &v["lemma"];
        let dict: HashMap<String, HashMap<String, String>> = match l.get("dict_file").and_then(|f| f.as_str()) {
            Some(f) if dir.join(f).exists() => serde_json::from_str(&std::fs::read_to_string(dir.join(f))?)?,
            _ => HashMap::new(),
        };
        let lchars = units(&l["char"]);
        let has_model = l.get("encoder").is_some() && !as_bool(l, "dict_only");
        let lemma = Lemma {
            dict,
            char_ids: char_index(&lchars),
            chars: lchars,
            pos_ids: str_index(&units(&l["pos"])),
            caseless: as_bool(l, "caseless"),
            dict_only: as_bool(l, "dict_only"),
            use_identity: as_bool(l, "use_identity"),
            vocab_size: as_usize(l, "vocab_size"),
            oov_slots: as_usize(l, "oov_slots"),
            sos: l.get("sos").and_then(|x| x.as_i64()).unwrap_or(2),
            eos: l.get("eos").and_then(|x| x.as_i64()).unwrap_or(3),
            max_dec_len: as_usize(l, "max_dec_len"),
            edit: l.get("edit").and_then(|x| x.as_array()).map(|a| !a.is_empty()).unwrap_or(false),
            enc: has_model.then(|| Onnx::new(dir.join("lemma_encoder.onnx"))),
            dec: has_model.then(|| Onnx::new(dir.join("lemma_decoder.onnx"))),
        };
        let d = &v["depparse"];
        let xpos = TagSpec::from_json(&d["xpos"]);
        let xpos_ids = match &xpos {
            TagSpec::List(u) => str_index(u),
            TagSpec::Composite(_) => HashMap::new(),
        };
        let dep = Dep {
            onnx: Onnx::new(dir.join("depparse.onnx")),
            word: str_index(&units(&d["word"])),
            word_lower: as_bool(d, "word_lower"),
            lemma: str_index(&units(&d["lemma"])),
            lemma_lower: as_bool(d, "lemma_lower"),
            chars: char_index(&units(&d["char"])),
            upos: str_index(&units(&d["upos"])),
            xpos,
            xpos_ids,
            deprel: units(&d["deprel"]),
            inputs: units(&d["inputs"]),
            charlm: d.get("charlm").and_then(CharLm::from_json),
            root_word: d.get("root_word").and_then(|x| x.as_str()).unwrap_or("\n").to_string(),
            ignore: ignores(d),
        };
        let lang = v.get("lang").and_then(|x| x.as_str()).unwrap_or("").to_string();
        println!("[STANZA] '{}' 사전 적재 {:.2}s ({})", lang, t0.elapsed().as_secs_f32(), dir.display());
        Ok(Self { lang, dir: dir.to_path_buf(), tokenizer, mwt, pos, pretrain, lemma, dep })
    }

    fn pretrain_ids(&self, words: &[String]) -> Vec<i64> {
        let p = match &self.pretrain {
            Some(p) => p,
            None => return vec![UNK; words.len()],
        };
        words.iter().map(|w| *p.get(&lower(w).replace(' ', "\u{a0}")).unwrap_or(&UNK)).collect()
    }

    /// 품사: (upos, xpos, feats)
    fn tag(&mut self, words: &[String]) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
        let n = words.len();
        let mut feed: Vec<(&str, DynInput)> = Vec::new();
        let pos = &self.pos;
        for name in &pos.inputs {
            match name.as_str() {
                "word_ids" => {
                    let ids = words
                        .iter()
                        .map(|w| *pos.word.get(&if pos.word_lower { lower(w) } else { w.clone() }).unwrap_or(&UNK))
                        .collect();
                    feed.push(("word_ids", DynInput::I64(row_i64(ids))));
                }
                "pretrain_ids" => feed.push(("pretrain_ids", DynInput::I64(row_i64(self.pretrain_ids(words))))),
                "char_ids" | "char_lens" => {}
                "charlm_f_ids" | "charlm_f_offsets" | "charlm_b_ids" | "charlm_b_offsets" => {}
                other => return Err(anyhow!("pos.onnx 의 모르는 입력: {other}")),
            }
        }
        if let Some(lm) = &pos.charlm {
            let (fi, fo) = lm.feed(words, true);
            let (bi, bo) = lm.feed(words, false);
            feed.push(("charlm_f_ids", DynInput::I64(row_i64(fi))));
            feed.push(("charlm_f_offsets", DynInput::I64(vec_i64(fo))));
            feed.push(("charlm_b_ids", DynInput::I64(row_i64(bi))));
            feed.push(("charlm_b_offsets", DynInput::I64(vec_i64(bo))));
        } else if pos.inputs.iter().any(|x| x == "char_ids") {
            let (ids, lens) = char_feed(words, &pos.chars);
            feed.push(("char_ids", DynInput::I64(ids)));
            feed.push(("char_lens", DynInput::I64(lens)));
        }
        let out_names: Vec<String> = pos.outputs.clone();
        let names: Vec<&str> = out_names.iter().map(|s| s.as_str()).collect();
        let outs = self.pos.onnx.run(feed, &names)?;
        let by_name: HashMap<&str, &ArrayD<f32>> = names.iter().copied().zip(outs.iter()).collect();
        let pick = |a: &ArrayD<f32>, i: usize| -> usize {
            let c = a.shape()[2];
            let row: Vec<f32> = (0..c).map(|k| a[[0, i, k]]).collect();
            argmax(&row)
        };
        let up = by_name.get("upos").ok_or_else(|| anyhow!("pos.onnx 에 upos 출력이 없습니다"))?;
        let upos: Vec<String> = (0..n).map(|i| self.pos.upos.get(pick(up, i)).cloned().unwrap_or_default()).collect();
        let mut cols: HashMap<String, Vec<String>> = HashMap::new();
        for name in self.pos.tag_order.iter().skip(1) {
            let spec = match self.pos.tags.get(name) {
                Some(s) => s,
                None => continue,
            };
            let col: Vec<String> = match spec {
                TagSpec::Composite(c) => (0..n)
                    .map(|i| {
                        let ids: Vec<usize> = (0..c.len())
                            .map(|k| by_name.get(format!("{name}_{k}").as_str()).map(|a| pick(a, i)).unwrap_or(EMPTY as usize))
                            .collect();
                        c.id2unit(&ids)
                    })
                    .collect(),
                TagSpec::List(u) => {
                    let ignore = self.pos.ignore.get(name).map(|s| !s.is_empty()).unwrap_or(false);
                    (0..n)
                        .map(|i| match by_name.get(name.as_str()) {
                            Some(a) => {
                                let k = pick(a, i);
                                if ignore && k as i64 == EMPTY { "_".to_string() } else { u.get(k).cloned().unwrap_or_default() }
                            }
                            None => "_".into(),
                        })
                        .collect()
                }
            };
            cols.insert(name.clone(), col);
        }
        let xpos = cols.remove("xpos").unwrap_or_else(|| vec!["_".into(); n]);
        let feats = cols.remove("feats").unwrap_or_else(|| vec!["_".into(); n]);
        Ok((upos, xpos, feats))
    }

    fn lemma_lookup(&self, w: &str, pos: &str) -> Option<String> {
        if let Some(h) = self.lemma.dict.get(pos).and_then(|d| d.get(w)) {
            return Some(h.clone());
        }
        self.lemma.dict.get("*").and_then(|d| d.get(w)).cloned()
    }

    fn lemma_seq2seq(&mut self, word: &str, upos: &str) -> Result<String> {
        let lm = &self.lemma;
        let (v, k) = (lm.vocab_size, lm.oov_slots);
        let mut delta: Vec<char> = word.chars().filter(|c| !lm.char_ids.contains_key(c)).collect();
        delta.sort();
        delta.dedup();
        let mut src = vec![lm.sos];
        for c in word.chars() {
            src.push(match lm.char_ids.get(&c) {
                Some(i) => *i,
                None => match delta.iter().position(|d| *d == c) {
                    Some(p) if p < k => (v + p) as i64,
                    _ => UNK,
                },
            });
        }
        src.push(lm.eos);
        let pos_id = *lm.pos_ids.get(upos).unwrap_or(&UNK);
        let (sos, eos, max_len, edit) = (lm.sos, lm.eos, lm.max_dec_len, lm.edit);
        let enc = self.lemma.enc.as_mut().ok_or_else(|| anyhow!("원형 인코더 없음"))?;
        let e = enc.run(
            vec![("src_ids", DynInput::I64(row_i64(src.clone()))), ("pos_id", DynInput::I64(vec_i64(vec![pos_id])))],
            &["h_in", "hn", "cn", "edit_logits"],
        )?;
        let h_in = e[0].clone();
        let (mut hn, mut cn) = (e[1].clone(), e[2].clone());
        let edit_row: Vec<f32> = e[3].iter().copied().collect();
        let dec = self.lemma.dec.as_mut().ok_or_else(|| anyhow!("원형 디코더 없음"))?;
        let mut token = sos;
        let mut out: Vec<usize> = Vec::new();
        for _ in 0..max_len {
            let d = dec.run(
                vec![
                    ("token", DynInput::I64(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![token]).unwrap())),
                    ("hn", DynInput::F32(hn.clone())),
                    ("cn", DynInput::F32(cn.clone())),
                    ("h_in", DynInput::F32(h_in.clone())),
                    ("src_ids", DynInput::I64(row_i64(src.clone()))),
                ],
                &["log_probs", "hn_out", "cn_out"],
            )?;
            let row: Vec<f32> = d[0].iter().copied().collect();
            let t = argmax(&row);
            hn = d[1].clone();
            cn = d[2].clone();
            if t as i64 == eos {
                break;
            }
            out.push(t);
            token = t as i64;
        }
        let chars = &self.lemma.chars;
        let mut pred = String::new();
        for t in out {
            if t < v {
                pred.push_str(chars.get(t).map(|s| s.as_str()).unwrap_or("<UNK>"));
            } else if t - v < delta.len() {
                pred.push(delta[t - v]);
            } else {
                pred.push_str("<UNK>");
            }
        }
        if edit {
            match argmax(&edit_row) {
                1 => pred = word.to_string(),
                2 => pred = lower(word),
                _ => {}
            }
        }
        if pred.is_empty() || pred.contains("<UNK>") {
            pred = word.to_string();
        }
        Ok(pred)
    }

    fn lemmatize(&mut self, words: &[String], upos: &[String]) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(words.len());
        for (w, p) in words.iter().zip(upos.iter()) {
            if self.lemma.use_identity {
                out.push(w.clone());
                continue;
            }
            let key = if self.lemma.caseless { lower(w) } else { w.clone() };
            let mut hit = self.lemma_lookup(&key, p);
            if hit.is_none() && !self.lemma.dict_only && self.lemma.enc.is_some() {
                hit = Some(self.lemma_seq2seq(&key, p)?);
            }
            let l = match hit {
                Some(h) if !h.is_empty() => h,
                Some(_) => "_".to_string(),
                None => w.clone(),
            };
            out.push(if l.is_empty() { "_".to_string() } else { l });
        }
        Ok(out)
    }

    fn parse(&mut self, words: &[String], upos: &[String], xpos: &[String], lemmas: &[String]) -> Result<(Vec<usize>, Vec<String>)> {
        let n1 = words.len() + 1;
        let mut feed: Vec<(&str, DynInput)> = Vec::new();
        let d = &self.dep;
        let pre = self.pretrain_ids(words);
        for name in &d.inputs {
            match name.as_str() {
                "pretrain_ids" => {
                    let mut v = vec![ROOT];
                    v.extend(pre.iter().copied());
                    feed.push(("pretrain_ids", DynInput::I64(row_i64(v))));
                }
                "word_ids" => {
                    let mut v = vec![ROOT];
                    v.extend(words.iter().map(|w| *d.word.get(&if d.word_lower { lower(w) } else { w.clone() }).unwrap_or(&UNK)));
                    feed.push(("word_ids", DynInput::I64(row_i64(v))));
                }
                "lemma_ids" => {
                    let mut v = vec![ROOT];
                    v.extend(lemmas.iter().map(|l| *d.lemma.get(&if d.lemma_lower { lower(l) } else { l.clone() }).unwrap_or(&UNK)));
                    feed.push(("lemma_ids", DynInput::I64(row_i64(v))));
                }
                "upos_ids" => {
                    let ig = d.ignore.get("upos");
                    let mut v = vec![ROOT];
                    v.extend(upos.iter().map(|u| if ig.map_or(false, |s| s.contains(u)) { EMPTY } else { *d.upos.get(u).unwrap_or(&UNK) }));
                    feed.push(("upos_ids", DynInput::I64(row_i64(v))));
                }
                "xpos_ids" => match &d.xpos {
                    TagSpec::Composite(c) => {
                        let k = c.len();
                        let mut flat = vec![ROOT; k];
                        for x in xpos {
                            flat.extend(c.unit2id(x));
                        }
                        feed.push(("xpos_ids", DynInput::I64(ArrayD::from_shape_vec(IxDyn(&[1, n1, k]), flat).unwrap())));
                    }
                    TagSpec::List(_) => {
                        let ig = d.ignore.get("xpos");
                        let mut v = vec![ROOT];
                        v.extend(xpos.iter().map(|x| if ig.map_or(false, |s| s.contains(x)) { EMPTY } else { *d.xpos_ids.get(x).unwrap_or(&UNK) }));
                        feed.push(("xpos_ids", DynInput::I64(row_i64(v))));
                    }
                },
                _ => {}
            }
        }
        if let Some(lm) = &d.charlm {
            let mut rooted = vec![d.root_word.clone()];
            rooted.extend(words.iter().cloned());
            let (fi, fo) = lm.feed(&rooted, true);
            let (bi, bo) = lm.feed(&rooted, false);
            feed.push(("charlm_f_ids", DynInput::I64(row_i64(fi))));
            feed.push(("charlm_f_offsets", DynInput::I64(vec_i64(fo))));
            feed.push(("charlm_b_ids", DynInput::I64(row_i64(bi))));
            feed.push(("charlm_b_offsets", DynInput::I64(vec_i64(bo))));
        } else if d.inputs.iter().any(|x| x == "char_ids") {
            let (ids, lens) = char_feed(words, &d.chars);
            let width = ids.shape()[1];
            let mut all = ArrayD::<i64>::zeros(IxDyn(&[n1, width]));
            all[[0, 0]] = ROOT;
            for i in 0..words.len() {
                for j in 0..width {
                    all[[i + 1, j]] = ids[[i, j]];
                }
            }
            let mut l = vec![1i64];
            l.extend(lens.iter().copied());
            feed.push(("char_ids", DynInput::I64(all)));
            feed.push(("char_lens", DynInput::I64(vec_i64(l))));
        }
        let outs = self.dep.onnx.run(feed, &["unlabeled", "deprel"])?;
        let (un, rel) = (&outs[0], &outs[1]);
        let r = rel.shape()[3];
        let scores: Vec<Vec<f64>> = (0..n1)
            .map(|i| log_softmax(&(0..n1).map(|j| un[[0, i, j]]).collect::<Vec<f32>>()).into_iter().map(|x| x as f64).collect())
            .collect();
        let label_lp: Vec<Vec<Vec<f64>>> = (0..n1)
            .map(|i| {
                (0..n1)
                    .map(|j| log_softmax(&(0..r).map(|k| rel[[0, i, j, k]]).collect::<Vec<f32>>()).into_iter().map(|x| x as f64).collect())
                    .collect()
            })
            .collect();
        let tree = chuliu_edmonds_one_root(&scores);
        let groups: Vec<Option<&'static str>> = (0..r).map(|k| deprel_group(self.dep.deprel.get(k + VOCAB_PREFIX_SIZE).map(|s| s.as_str()).unwrap_or(""))).collect();
        let (tree, raw) = resolve_head_constraint_violations(&scores, &label_lp, tree, &groups, 5);
        let heads = tree[1..].to_vec();
        let rels = raw.iter().map(|&k| self.dep.deprel.get(k + VOCAB_PREFIX_SIZE).cloned().unwrap_or_else(|| "dep".into())).collect();
        Ok((heads, rels))
    }

    /// 낱말 목록(이미 나눈)에 품사 · 원형 · 의존 구문
    pub fn tag_words(&mut self, words: &[String]) -> Result<Vec<WordAnalysis>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let (upos, xpos, feats) = self.tag(words)?;
        let lemmas = self.lemmatize(words, &upos)?;
        let (heads, rels) = self.parse(words, &upos, &xpos, &lemmas)?;
        Ok((0..words.len())
            .map(|i| WordAnalysis {
                text: words[i].clone(),
                upos: upos[i].clone(),
                xpos: xpos[i].clone(),
                feats: feats[i].clone(),
                lemma: lemmas[i].clone(),
                head: heads[i],
                deprel: rels[i].clone(),
            })
            .collect())
    }

    /// 원문 → 문장마다 토큰 · 낱말 분석 (Stanza tokenize[,mwt],pos,lemma,depparse 와 같은 결과)
    pub fn analyze(&mut self, text: &str) -> Result<Vec<SentenceAnalysis>> {
        let mut res = Vec::new();
        for sent in self.tokenizer.tokenize(text)? {
            let mut tokens = Vec::new();
            let mut spans = Vec::new();
            let mut words: Vec<String> = Vec::new();
            for (tok, p) in sent {
                let parts: Vec<String> = if (p == 3 || p == 4) && self.mwt.is_some() {
                    let exp = self.mwt.as_mut().unwrap().expand(&tok)?;
                    let ps: Vec<String> = exp.split(' ').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
                    if ps.len() > 1 { ps } else { vec![tok.clone()] }
                } else {
                    vec![tok.clone()]
                };
                spans.push((words.len(), words.len() + parts.len()));
                words.extend(parts);
                tokens.push(tok);
            }
            let analysed = self.tag_words(&words)?;
            res.push(SentenceAnalysis { tokens, token_words: spans, words: analysed });
        }
        Ok(res)
    }
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = x.iter().map(|v| (v - m).exp()).sum();
    let ls = s.ln();
    x.iter().map(|v| (v - m) - ls).collect()
}

// ---------------------------------------------------------------------------------------------
// 의존 구문 디코딩 (stanza common/chuliu_edmonds.py · depparse/head_constraints.py 를 그대로)
// ---------------------------------------------------------------------------------------------

fn argmax_f64(row: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

/// tarjan: 순환마다 참여 마디 표시 (재귀 대신 직접 만든 스택, 발견 순서도 원본과 같음)
fn tarjan(tree: &[usize]) -> Vec<Vec<bool>> {
    let n = tree.len();
    let mut indices = vec![-1i64; n];
    let mut lowlinks = vec![-1i64; n];
    let mut onstack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut counter = 0i64;
    let mut cycles = Vec::new();
    let dependents = |i: usize| -> Vec<usize> { (0..n).filter(|&j| tree[j] == i).collect() };
    for root in 0..n {
        if indices[root] != -1 {
            continue;
        }
        // (마디, 의존자 목록, 다음 볼 자리, 방금 돌아온 의존자)
        let mut call: Vec<(usize, Vec<usize>, usize, Option<usize>)> = vec![(root, Vec::new(), usize::MAX, None)];
        while let Some((i, deps, mut pos, back)) = call.pop() {
            let deps = if pos == usize::MAX {
                counter += 1;
                indices[i] = counter - 1;
                lowlinks[i] = counter - 1;
                stack.push(i);
                onstack[i] = true;
                pos = 0;
                dependents(i)
            } else {
                if let Some(j) = back {
                    lowlinks[i] = lowlinks[i].min(lowlinks[j]);
                }
                deps
            };
            let mut descended = false;
            while pos < deps.len() {
                let j = deps[pos];
                pos += 1;
                if indices[j] == -1 {
                    call.push((i, deps.clone(), pos, Some(j)));
                    call.push((j, Vec::new(), usize::MAX, None));
                    descended = true;
                    break;
                } else if onstack[j] {
                    lowlinks[i] = lowlinks[i].min(indices[j]);
                }
            }
            if descended {
                continue;
            }
            if lowlinks[i] == indices[i] {
                let mut cycle = vec![false; n];
                while let Some(&top) = stack.last() {
                    if top == i {
                        break;
                    }
                    let j = stack.pop().unwrap();
                    onstack[j] = false;
                    cycle[j] = true;
                }
                stack.pop();
                onstack[i] = false;
                cycle[i] = true;
                if cycle.iter().filter(|&&b| b).count() > 1 {
                    cycles.push(cycle);
                }
            }
        }
    }
    cycles
}

fn prepare_scores(scores: &mut [Vec<f64>]) {
    let n = scores.len();
    for i in 0..n {
        scores[i][i] = f64::NEG_INFINITY;
    }
    for j in 0..n {
        scores[0][j] = f64::NEG_INFINITY;
    }
    scores[0][0] = 0.0;
}

struct Contracted {
    tree: Vec<usize>,
    cycle_locs: Vec<usize>,
    noncycle_locs: Vec<usize>,
    metanode_heads: Vec<usize>,
    metanode_deps: Vec<usize>,
}

fn process_cycle(tree: &[usize], cycle: &[bool], scores: &[Vec<f64>]) -> (Vec<Vec<f64>>, Contracted) {
    let cycle_locs: Vec<usize> = (0..tree.len()).filter(|&i| cycle[i]).collect();
    let noncycle_locs: Vec<usize> = (0..tree.len()).filter(|&i| !cycle[i]).collect();
    let cycle_scores: Vec<f64> = cycle_locs.iter().map(|&c| scores[c][tree[c]]).collect();
    let cycle_score: f64 = cycle_scores.iter().sum();
    let (nc, nn) = (cycle_locs.len(), noncycle_locs.len());
    // metanode_head_scores (c x n), metanode_dep_scores (n x c)
    let head_scores: Vec<Vec<f64>> = (0..nc)
        .map(|a| (0..nn).map(|b| scores[cycle_locs[a]][noncycle_locs[b]] - cycle_scores[a] + cycle_score).collect())
        .collect();
    let dep_scores: Vec<Vec<f64>> = (0..nn).map(|b| (0..nc).map(|a| scores[noncycle_locs[b]][cycle_locs[a]]).collect()).collect();
    let metanode_heads: Vec<usize> = (0..nn).map(|b| argmax_f64(&(0..nc).map(|a| head_scores[a][b]).collect::<Vec<_>>())).collect();
    let metanode_deps: Vec<usize> = (0..nn).map(|b| argmax_f64(&dep_scores[b])).collect();
    let mut sub = vec![vec![0.0f64; nn + 1]; nn + 1];
    for (bi, &b) in noncycle_locs.iter().enumerate() {
        for (ci, &c) in noncycle_locs.iter().enumerate() {
            sub[bi][ci] = scores[b][c];
        }
    }
    for b in 0..nn {
        sub[nn][b] = head_scores[metanode_heads[b]][b];
        sub[b][nn] = dep_scores[b][metanode_deps[b]];
    }
    (sub, Contracted { tree: tree.to_vec(), cycle_locs, noncycle_locs, metanode_heads, metanode_deps })
}

fn expand_contracted_tree(c: &Contracted, contracted_tree: &[usize]) -> Vec<usize> {
    let nn = c.noncycle_locs.len();
    let cycle_head = contracted_tree[nn];
    let ct = &contracted_tree[..nn];
    let mut new_tree = vec![usize::MAX; c.tree.len()];
    for b in 0..nn {
        if ct[b] < nn {
            new_tree[c.noncycle_locs[b]] = c.noncycle_locs[ct[b]];
        } else {
            new_tree[c.noncycle_locs[b]] = c.cycle_locs[c.metanode_deps[b]];
        }
    }
    for &l in &c.cycle_locs {
        new_tree[l] = c.tree[l];
    }
    let cycle_root = c.metanode_heads[cycle_head];
    new_tree[c.cycle_locs[cycle_root]] = c.noncycle_locs[cycle_head];
    new_tree
}

fn chuliu_edmonds(mut scores: Vec<Vec<f64>>) -> Vec<usize> {
    let mut stack: Vec<Contracted> = Vec::new();
    prepare_scores(&mut scores);
    let mut tree: Vec<usize> = scores.iter().map(|r| argmax_f64(r)).collect();
    let mut cycles = tarjan(&tree);
    while let Some(cycle) = cycles.pop() {
        let (sub, c) = process_cycle(&tree, &cycle, &scores);
        stack.push(c);
        scores = sub;
        prepare_scores(&mut scores);
        tree = scores.iter().map(|r| argmax_f64(r)).collect();
        cycles = tarjan(&tree);
    }
    while let Some(c) = stack.pop() {
        tree = expand_contracted_tree(&c, &tree);
    }
    tree
}

/// 루트 칸 점수를 (가장 작은 점수 × 길이) 만큼 내려 루트가 하나만 붙게 합니다.
pub fn chuliu_edmonds_one_root(scores: &[Vec<f64>]) -> Vec<usize> {
    let n = scores.len();
    let min = scores.iter().flatten().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
    let mut s: Vec<Vec<f64>> = scores.to_vec();
    for row in s.iter_mut() {
        row[0] += min * n as f64;
    }
    chuliu_edmonds(s)
}

/// 한 머리에 둘 이상 붙으면 안 되는 관계 묶음 (depparse/head_constraints.py SINGLETON_DEPREL_GROUPS)
fn deprel_group(rel: &str) -> Option<&'static str> {
    match rel {
        "nsubj" | "csubj" | "nsubj:pass" | "csubj:pass" => Some("subj"),
        "nsubj:outer" | "csubj:outer" => Some("subj_outer"),
        "obj" => Some("obj"),
        _ => None,
    }
}

fn find_violations(tree: &[usize], raw: &[usize], groups: &[Option<&'static str>]) -> Vec<(usize, &'static str, Vec<usize>)> {
    // 머리 · 묶음을 처음 나온 순서대로 (파이썬 dict 삽입 순서)
    let mut order: Vec<(usize, &'static str, Vec<usize>)> = Vec::new();
    for (j, &r) in raw.iter().enumerate() {
        let w = j + 1;
        if let Some(g) = groups.get(r).copied().flatten() {
            let h = tree[w];
            match order.iter_mut().find(|(hh, gg, _)| *hh == h && *gg == g) {
                Some(e) => e.2.push(w),
                None => order.push((h, g, vec![w])),
            }
        }
    }
    // 같은 머리의 묶음들은 머리가 처음 나온 순서 안에서 묶음 순서
    let mut heads: Vec<usize> = Vec::new();
    for (h, _, _) in &order {
        if !heads.contains(h) {
            heads.push(*h);
        }
    }
    let mut out = Vec::new();
    for h in heads {
        for (hh, g, deps) in &order {
            if *hh == h && deps.len() > 1 {
                out.push((h, *g, deps.clone()));
            }
        }
    }
    out
}

/// stanza depparse/head_constraints.py resolve_head_constraint_violations
pub fn resolve_head_constraint_violations(
    scores: &[Vec<f64>],
    label_lp: &[Vec<Vec<f64>>],
    mut tree: Vec<usize>,
    groups: &[Option<&'static str>],
    max_iterations: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut excluded: HashMap<(usize, usize), HashSet<&'static str>> = HashMap::new();
    let label_for = |excluded: &HashMap<(usize, usize), HashSet<&'static str>>, dep: usize, head: usize, extra: Option<&'static str>| -> (usize, f64) {
        let row = &label_lp[dep][head];
        let mut ex: HashSet<&'static str> = excluded.get(&(dep, head)).cloned().unwrap_or_default();
        if let Some(e) = extra {
            ex.insert(e);
        }
        if ex.is_empty() {
            let b = argmax_f64(row);
            return (b, row[b]);
        }
        let (mut best, mut best_score) = (usize::MAX, f64::NEG_INFINITY);
        for (k, &s) in row.iter().enumerate() {
            if let Some(g) = groups[k] {
                if ex.contains(g) {
                    continue;
                }
            }
            if s > best_score {
                best_score = s;
                best = k;
            }
        }
        (if best == usize::MAX { 0 } else { best }, best_score)
    };
    let labels_for_tree = |excluded: &HashMap<(usize, usize), HashSet<&'static str>>, t: &[usize]| -> Vec<usize> {
        (1..t.len()).map(|j| label_for(excluded, j, t[j], None).0).collect()
    };
    let combined = |t: &[usize], raw: &[usize]| -> f64 { (1..t.len()).map(|j| scores[j][t[j]] + label_lp[j][t[j]][raw[j - 1]]).sum() };

    let mut raw = labels_for_tree(&excluded, &tree);
    for _ in 0..max_iterations {
        let violations = find_violations(&tree, &raw, groups);
        let (head_idx, group, deps) = match violations.into_iter().next() {
            Some(v) => v,
            None => break,
        };
        let mut best: Option<(f64, Vec<usize>, Vec<usize>, usize)> = None;
        for keep in 0..deps.len() {
            let mut trial = scores.to_vec();
            for (k, &dep) in deps.iter().enumerate() {
                if k == keep {
                    continue;
                }
                for h in 0..tree.len() {
                    let extra = if h == head_idx { Some(group) } else { None };
                    let (_, s) = label_for(&excluded, dep, h, extra);
                    trial[dep][h] = scores[dep][h] + s;
                }
            }
            let ttree = chuliu_edmonds_one_root(&trial);
            let mut tids = labels_for_tree(&excluded, &ttree);
            for (k, &dep) in deps.iter().enumerate() {
                if k != keep && ttree[dep] == head_idx {
                    tids[dep - 1] = label_for(&excluded, dep, head_idx, Some(group)).0;
                }
            }
            let score = combined(&ttree, &tids);
            if best.as_ref().map_or(true, |b| score > b.0) {
                best = Some((score, ttree, tids, keep));
            }
        }
        let (_, t, ids, keep) = best.unwrap();
        tree = t;
        raw = ids;
        for (k, &dep) in deps.iter().enumerate() {
            if k != keep {
                excluded.entry((dep, head_idx)).or_default().insert(group);
            }
        }
    }
    (tree, raw)
}

// ---------------------------------------------------------------------------------------------
// 언어별 캐시
// ---------------------------------------------------------------------------------------------

static CACHE: Lazy<Mutex<Vec<(String, Arc<Mutex<StanzaLang>>)>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// 언어 변환본을 엽니다 (최근 쓴 CACHE_LANGS 개 언어만 메모리에 둠). 폴더가 없으면 Err.
pub fn pipeline(lang: &str) -> Result<Arc<Mutex<StanzaLang>>> {
    {
        let mut c = CACHE.lock().unwrap();
        if let Some(pos) = c.iter().position(|(l, _)| l == lang) {
            let e = c.remove(pos);
            let p = e.1.clone();
            c.push(e);
            return Ok(p);
        }
    }
    let dir = models_root().join(lang);
    if !dir.join("vocab.json").exists() {
        return Err(anyhow!("Stanza 모델이 없습니다: {} (stanza-onnx-export 의 onnx_final/{lang} 를 넣으세요)", dir.display()));
    }
    let p = Arc::new(Mutex::new(StanzaLang::load(&dir)?));
    let mut c = CACHE.lock().unwrap();
    c.push((lang.to_string(), p.clone()));
    while c.len() > CACHE_LANGS {
        let (l, _) = c.remove(0);
        println!("[STANZA] '{l}' 변환본을 메모리에서 내립니다 (언어 캐시 {CACHE_LANGS}개)");
    }
    Ok(p)
}

/// 모든 언어 변환본을 내립니다.
pub fn release_all() {
    CACHE.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------------------------
// 검색 질의용: 질의 단위(공백 어절, 일본어 · 중국어는 Stanza 토큰)마다 대표 품사 · 어간 · 관계
// ---------------------------------------------------------------------------------------------

/// 내용어가 아닌 품사 (어절 대표 낱말을 고를 때 건너뜀)
const FUNCTION_UPOS: [&str; 9] = ["ADP", "DET", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON", "AUX", "SYM"];

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct QueryUnit {
    /// 질의 단위 표면형 (공백 어절 또는 Stanza 토큰)
    pub surface: String,
    /// 대표(첫 내용어) 낱말의 UPOS. 내용어가 없으면 첫 낱말의 UPOS.
    /// 한국어 UD(KAIST)는 어절 끝 기능으로 UPOS 를 매겨 '따뜻하게' 가 SCONJ 이므로, 한국어는 첫 실질 형태소의 세종 태그로 고칩니다.
    pub upos: String,
    /// 내용어가 하나도 없는 단위 (전치사 · 관사 · 대명사 · 조사 · 구두점만). 검색어에서 뺄지 판정에 씁니다.
    pub functional: bool,
    /// 검색용 어간: 한국어는 xpos 로 실질 형태소만 이은 것(가디건+을 → 가디건), 그 밖의 언어는 대표 낱말의 원형
    /// (아랍어는 모음 부호를 뺌). 숫자가 든 단위(식별자 · 금액)는 표면형 그대로.
    pub stem: String,
    pub deprel: String,
    pub feats: String,
    /// 이 단위에 든 낱말 (MWT 를 펼친 뒤)
    pub words: Vec<WordAnalysis>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct QueryAnalysis {
    pub lang: String,
    pub units: Vec<QueryUnit>,
    /// 단위가 공백 어절과 같은가 (false 면 Stanza 토큰)
    pub whitespace_units: bool,
    pub elapsed_ms: u128,
}

impl QueryAnalysis {
    pub fn surfaces(&self) -> Vec<String> {
        self.units.iter().map(|u| u.surface.clone()).collect()
    }

    /// 로그 한 줄: 표면(품사, 어간, 관계)
    pub fn describe(&self) -> Vec<String> {
        self.units
            .iter()
            .map(|u| {
                let inner = if u.words.len() > 1 {
                    format!(" [{}]", u.words.iter().map(|w| format!("{}/{}", w.text, w.upos)).collect::<Vec<_>>().join(" "))
                } else {
                    String::new()
                };
                let f = if u.functional { ", 기능어" } else { "" };
                format!("{}(tag:{}, lemma:{}, rel:{}{}){}", u.surface, u.upos, u.stem, u.deprel, f, inner)
            })
            .collect()
    }
}

/// KAIST 품사(xpos)로 한국어 어절의 실질 형태소만 앞에서부터 잇습니다. 조사(j) · 어미(e) · 접미사(xs)에서 멈춥니다.
pub fn korean_stem(lemma: &str, xpos: &str) -> String {
    let ls: Vec<&str> = lemma.split('+').collect();
    let xs: Vec<&str> = xpos.split('+').collect();
    if ls.len() != xs.len() || ls.is_empty() {
        return ls.first().map(|s| s.to_string()).unwrap_or_default();
    }
    let mut stem = String::new();
    for (l, x) in ls.iter().zip(xs.iter()) {
        let content = matches!(x.chars().next(), Some('n' | 'f' | 'p' | 'm' | 'i' | 's')) || x.starts_with("xp");
        if !content {
            break;
        }
        stem.push_str(l);
    }
    if stem.is_empty() {
        ls[0].to_string()
    } else {
        stem
    }
}

fn strip_arabic_marks(s: &str) -> String {
    s.chars().filter(|c| !matches!(*c as u32, 0x064B..=0x0652 | 0x0670 | 0x0640)).collect()
}

/// 한국어 낱말의 첫 형태소 세종(KAIST) 태그가 실질 형태소인가 (체언 · 용언 · 수식언 · 외국어 · 감탄사 · 접두사)
fn korean_content(xpos: &str) -> bool {
    let first = xpos.split('+').next().unwrap_or("");
    matches!(first.chars().next(), Some('n' | 'p' | 'm' | 'f' | 'i')) || first.starts_with("xp")
}

/// 첫 실질 형태소 태그 → UPOS (모델 UPOS 가 기능어일 때만 씀)
fn korean_content_upos(xpos: &str) -> Option<&'static str> {
    let first = xpos.split('+').next().unwrap_or("");
    Some(match first {
        t if t.starts_with("nq") => "PROPN",
        t if t.starts_with("np") => "PRON",
        t if t.starts_with('n') => "NOUN",
        t if t.starts_with("pa") => "ADJ",
        t if t.starts_with("px") => "AUX",
        t if t.starts_with('p') => "VERB",
        t if t.starts_with("mm") => "DET",
        t if t.starts_with('m') => "ADV",
        t if t.starts_with('f') => "X",
        t if t.starts_with('i') => "INTJ",
        _ => return None,
    })
}

fn make_unit(surface: String, words: Vec<WordAnalysis>, lang: &str) -> QueryUnit {
    let is_content = |w: &WordAnalysis| {
        if lang == "ko" {
            korean_content(&w.xpos) && w.upos != "PUNCT"
        } else {
            !FUNCTION_UPOS.contains(&w.upos.as_str())
        }
    };
    let head_idx = words.iter().position(|w| is_content(w));
    let functional = head_idx.is_none();
    let head = head_idx.unwrap_or(0);
    let w = words.get(head).cloned().unwrap_or_default();
    let mut upos = w.upos.clone();
    if lang == "ko" && !functional && FUNCTION_UPOS.contains(&upos.as_str()) {
        if let Some(u) = korean_content_upos(&w.xpos) {
            upos = u.to_string();
        }
    }
    let stem = if surface.chars().any(|c| c.is_ascii_digit()) {
        // 식별자 · 금액은 표면형 그대로 ('CI-2026-08001' 의 원형 'CI' 를 어간으로 쓰면 하류에서 잔여가 긴 기능어로 빠짐)
        surface.clone()
    } else {
        match lang {
            // 토크나이저가 어절을 둘로 가르면('물건' → 물 · 건) 이어지는 실질 조각을 잇습니다. 뒤 조각은 원형이 표면의 접두가 아니면
            // ('건' 의 원형 '것+은') 표면형을 씁니다 — 어절을 가른 것이 토크나이저의 실수라 원형보다 원문이 맞습니다.
            "ko" => words[head..]
                .iter()
                .take_while(|x| is_content(x))
                .enumerate()
                .map(|(k, x)| {
                    let s = korean_stem(&x.lemma, &x.xpos);
                    if k == 0 || x.text.starts_with(&s) { s } else { x.text.clone() }
                })
                .collect::<String>(),
            "ar" => strip_arabic_marks(&w.lemma),
            _ => w.lemma.clone(),
        }
    };
    let stem = if stem.is_empty() { w.lemma.clone() } else { stem };
    QueryUnit { surface, upos, functional, stem, deprel: w.deprel.clone(), feats: w.feats.clone(), words }
}

/// 검색 질의 분석. 공백으로 낱말을 나누는 언어는 공백 어절마다 그 안의 Stanza 토큰 · 낱말을 모읍니다
/// (검색 파서의 창 · NMS 좌표계가 공백 어절 기준이라). 일본어 · 중국어는 Stanza 토큰이 단위입니다.
pub fn analyze_query(query: &str, lang: &str) -> Result<QueryAnalysis> {
    if !LANGS.contains(&lang) {
        return Err(anyhow!("Stanza 변환본이 없는 언어입니다 ({lang:?})"));
    }
    let t0 = std::time::Instant::now();
    let p = pipeline(lang)?;
    let mut st = p.lock().map_err(|_| anyhow!("Stanza 파이프라인 잠금 실패"))?;
    let sents = st.analyze(query)?;
    drop(st);
    // 토큰 목록 (문장 경계는 질의에서 의미가 없어 이어 붙임)
    let mut toks: Vec<(String, Vec<WordAnalysis>)> = Vec::new();
    for s in &sents {
        for (t, &(a, b)) in s.tokens.iter().zip(s.token_words.iter()) {
            // 낱말 머리 번호는 문장 안 번호 그대로 둡니다 (단위 판정에는 관계 이름만 씀)
            toks.push((t.clone(), s.words[a..b].to_vec()));
        }
    }
    let ws: Vec<&str> = query.split_whitespace().collect();
    let mut units = Vec::new();
    let mut aligned = !lang_without_spaces(lang);
    if aligned {
        // 공백 어절마다 이어지는 토큰을 글자 수로 맞춥니다 (토큰은 공백을 넘지 않음)
        let mut ti = 0usize;
        'outer: for w in &ws {
            let target: String = w.chars().filter(|c| !is_stanza_space(*c)).collect();
            let mut acc = String::new();
            let mut words = Vec::new();
            while ti < toks.len() && acc.chars().count() < target.chars().count() {
                acc.push_str(&toks[ti].0.replace(' ', ""));
                words.extend(toks[ti].1.iter().cloned());
                ti += 1;
            }
            if acc != target {
                aligned = false;
                break 'outer;
            }
            units.push(make_unit(w.to_string(), words, lang));
        }
        if aligned && ti != toks.len() {
            aligned = false;
        }
    }
    if !aligned {
        units = toks.into_iter().map(|(t, w)| make_unit(t, w, lang)).collect();
    }
    Ok(QueryAnalysis { lang: lang.to_string(), units, whitespace_units: aligned, elapsed_ms: t0.elapsed().as_millis() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn korean_stem_keeps_content_morphemes() {
        assert_eq!(korean_stem("가디건+을", "ncn+jco"), "가디건");
        assert_eq!(korean_stem("취소+되+ㄴ", "ncpa+xsv+etm"), "취소");
        assert_eq!(korean_stem("5만+원", "nnc+nbu"), "5만원");
        assert_eq!(korean_stem("찾+아+주+어", "pvg+ecx+px+ef"), "찾");
        assert_eq!(korean_stem("을", "jco"), "을");
    }

    #[test]
    fn mwt_dictionary_casing() {
        let d: HashMap<String, String> = [("del".to_string(), "de el".to_string())].into_iter().collect();
        assert_eq!(dict_expansion(&d, "del").as_deref(), Some("de el"));
        assert_eq!(dict_expansion(&d, "DEL").as_deref(), Some("DE EL"));
        assert_eq!(dict_expansion(&d, "Del").as_deref(), Some("De el"));
        assert_eq!(dict_expansion(&d, "dEl"), None);
    }

    #[test]
    fn query_lang_by_script() {
        assert_eq!(detect_query_lang("베이지 가디건", "english"), Some("ko"));
        assert_eq!(detect_query_lang("先週キャンセルされた注文", "korean"), Some("ja"));
        assert_eq!(detect_query_lang("上周取消的订单", "korean"), Some("zh-hans"));
        assert_eq!(detect_query_lang("الطلبات الملغاة", "korean"), Some("ar"));
        assert_eq!(detect_query_lang("pedidos con envío pequeño", "korean"), Some("es"));
    }
}
