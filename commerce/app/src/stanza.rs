//! Stanza 실행기 (stanza-onnx-export format 2 변환본, 12개 언어) — premium-discount-nlp 의 `stanza` 를 감싸는 얇은 층.
//!
//! 실행기 본체(토크나이저 · MWT · 품사 · 원형 · 의존 구문 · 질의 단위, 언어 캐시)는 premium-discount-nlp 에 있고, 예전 이 파일의
//! 구현과 같은 코드입니다: stanza-onnx-export 의 기준 원문 278개(12개 언어, 1,924 낱말)에서 토큰 · 원형 · 품사 · 형태 · 머리 ·
//! 관계와 질의 분석이 바이트 단위로 같습니다 (`tests/it/stanza_runtime.rs` 의 `dump_analyses`, premium-discount-nlp 에 같은 시험).
//! 규약은 stanza-onnx-export 의 `verify_onnx.py` RefRuntime 과 같고, 같은 기준 JSON 으로 대조합니다(`real_models_match_pytorch_reference`).
//!
//! 이 앱에 남은 것:
//! - 모델 폴더: `LOGIS_STANZA_DIR` 이 있으면 그 아래 `<lang>/`, 없으면 앱 폴더 `models/stanza/<lang>/` (평가 폴더 `LOGIS_APP_DIR` 포함).
//!   premium-discount-nlp 의 기본 규칙(다른 앱 폴더부터 찾음) 대신 이 규칙을 `set_dir_resolver` 로 겁니다.
//! - 질의 언어 판정: logis-bias 의 `query_lang` (bias 사전의 라틴 언어 구역으로 투표 — premium-discount-nlp 판에는 없음).
//! - 내려받기 목록 (`remote_manifest` · `manifest_files`, 키 순서는 `json_order`).
//!
//! ONNX Runtime 은 링크하지 않고 실행할 때 불러옵니다(premium-discount `runtime-load`): `ORT_DYLIB_PATH`, 실행 파일 옆
//! (배포 앱은 `dlls/*` 가 실행 파일 옆), 앱 모델 폴더의 `onnxruntime/`, 그다음 빌드가 준비한 라이브러리 순서로 전체 경로로만 엽니다
//! (Windows 의 `System32\onnxruntime.dll` 옛 판을 집지 않음).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

pub use premium_discount_nlp::stanza::{
    argmax, chuliu_edmonds_one_root, dict_expansion, korean_stem, release_all, remote_base, resolve_head_constraint_violations,
    Composite, QueryAnalysis, QueryUnit, SentenceAnalysis, StanzaLang, WordAnalysis, HF_REPO_PREFIX, HF_REPO_SUBDIR, LANGS,
};

// ---------------------------------------------------------------------------------------------
// 언어 코드
// ---------------------------------------------------------------------------------------------

pub use logis_bias::query_lang::{lang_code, LATIN_LANGS, LATIN_DETECTOR, LATIN_VOCAB, vocab_words, collect_vocab, latin_vocab_vote, name_words, is_latin_letter, Script, QUOTE_PAIRS, quoted_ranges, strip_quoted, quoted_token_flags, SIMPLIFIED_ONLY, KANJI_NOT_SIMPLIFIED, drop_cjk_names, detect_query_lang, lang_without_spaces, UI_LANGS};

// ---------------------------------------------------------------------------------------------
// 모델 폴더 · 내려받기
// ---------------------------------------------------------------------------------------------

pub fn models_root() -> PathBuf {
    match std::env::var("LOGIS_STANZA_DIR") {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d),
        _ => crate::utils::get_app_dir().join("models").join("stanza"),
    }
}

/// premium-discount-nlp 가 언어 폴더를 이 앱 규칙([`models_root`] 아래 `<lang>`)으로 찾게 합니다. 처음 한 번만 겁니다.
fn use_app_models_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        premium_discount_nlp::stanza::set_dir_resolver(|lang| models_root().join(lang));
    });
}

/// 언어 변환본을 엽니다 (최근 쓴 언어 하나만 메모리에 둠). 폴더가 없으면 Err.
pub fn pipeline(lang: &str) -> Result<Arc<Mutex<StanzaLang>>> {
    use_app_models_root();
    premium_discount_nlp::stanza::pipeline(lang)
}

/// 검색 질의를 질의 단위(공백 어절, 일본어 · 중국어는 Stanza 토큰)로 나눠 대표 품사 · 어간 · 관계를 붙입니다.
pub fn analyze_query(query: &str, lang: &str) -> Result<QueryAnalysis> {
    use_app_models_root();
    premium_discount_nlp::stanza::analyze_query(query, lang)
}

/// vocab.json 의 `files` (처리기별 파일 목록)을 펼칩니다. vocab.json 자신은 넣지 않습니다.
pub fn manifest_files(vocab: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(o) = vocab.get("files").and_then(|f| f.as_object()) {
        for v in logis_schema::json_order::sorted_values(o) {
            for f in v.as_array().into_iter().flatten().map(|x| x.as_str().unwrap_or("").to_string()) {
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
    let v: Value = serde_json::from_slice(&res.bytes().await?).map(logis_schema::json_order::sorted).with_context(|| format!("{url} 해석"))?;
    if v.get("format").and_then(|x| x.as_i64()) != Some(2) {
        return Err(anyhow!(
            "{base_url} 의 Stanza 변환본이 이전 판(format 1)입니다. stanza-onnx-export 의 onnx_final/<lang> (format 2)을 올려야 합니다"
        ));
    }
    let mut files = manifest_files(&v);
    files.push("vocab.json".to_string());
    Ok(files)
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("{} 읽기", path.display()))?;
    serde_json::from_str(&raw).map(logis_schema::json_order::sorted).with_context(|| format!("{} 해석", path.display()))
}
