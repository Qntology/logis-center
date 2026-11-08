//! 내장 도메인 사전 — 개발 빌드는 `bias.json`(공통 구역 + ko · en), 배포 빌드는 `full_bias.json`(공통 구역 + 모든 언어).
//!
//! 두 파일은 손으로 따로 고치므로 어긋나기 쉽습니다. 공통 구역과 ko · en 구역이 같은지, full 에만 있는 구역이 언어 구역인지 봅니다.

use serde_json::{Map, Value};
use tauri_app_lib::utils::bias_schema::{lang_code_of, BIAS_SOURCE};

const DEV: &str = include_str!("../../src/utils/bias.json");
const FULL: &str = include_str!("../../src/utils/full_bias.json");

fn sections(text: &str) -> Map<String, Value> {
    serde_json::from_str::<Value>(text).expect("JSON").as_object().expect("최상위 객체").clone()
}

/// 언어 구역: 페이지 유형(goods · order · tracking …)별 사전
fn is_language_section(v: &Value) -> bool {
    v.get("goods").is_some() && v.get("order").is_some() && v.get("tracking").is_some()
}

#[test]
fn full_bias_is_dev_bias_plus_language_sections() {
    let dev = sections(DEV);
    let full = sections(FULL);
    for (k, v) in &dev {
        assert_eq!(full.get(k), Some(v), "구역 '{k}' 가 bias.json 과 full_bias.json 에서 다릅니다 (한쪽만 고침)");
    }
    let mut dev_langs: Vec<&str> = dev.iter().filter(|(_, v)| is_language_section(v)).map(|(k, _)| k.as_str()).collect();
    dev_langs.sort();
    assert_eq!(dev_langs, vec!["en", "ko"], "개발 사전은 한국어 · 영어 언어 구역만 담습니다");
    let extra: Vec<&String> = full.keys().filter(|k| !dev.contains_key(*k)).collect();
    assert!(!extra.is_empty(), "full_bias.json 에 다른 언어 구역이 없습니다");
    for k in extra {
        assert!(is_language_section(&full[k]), "full_bias.json 에만 있는 '{k}' 는 언어 구역이어야 합니다");
    }
}

/// 배포 사전은 Stanza 가 받는 12개 언어마다 언어 구역을 둡니다. 구역이 없으면 그 언어 페이지 · 서류도
/// 영어 구역 앵커로 필드를 맞춥니다(bias_schema 의 `or_else(|| BIAS_DICT.get("en"))`).
#[test]
fn full_bias_covers_every_stanza_language() {
    let full = sections(FULL);
    let missing: Vec<String> = tauri_app_lib::stanza::LANGS
        .iter()
        .map(|l| lang_code_of(l))
        .filter(|code| !full.get(code).is_some_and(is_language_section))
        .collect();
    assert!(missing.is_empty(), "full_bias.json 에 언어 구역이 없습니다: {missing:?}");
}

#[test]
fn build_kind_picks_bias_source() {
    let want = if cfg!(feature = "full-bias") || !cfg!(debug_assertions) { "full_bias.json" } else { "bias.json" };
    assert_eq!(BIAS_SOURCE.0, want);
    let want_text = if want == "bias.json" { DEV } else { FULL };
    assert_eq!(BIAS_SOURCE.1.len(), want_text.len());
}
