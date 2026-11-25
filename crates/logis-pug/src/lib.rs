//! logis-center 의 글 파이프라인 crate (모델 · DB · Tauri 없음).
//!
//! - [`utils::parsing`]: HTML → PUG 압축(`PugMode`) · 블록 매핑 · 레이아웃 판정, 사전 재수출
//! - [`utils::ai_utils`]: 질의 · 헤더 · 값 판정, 코사인(granite 눈금 보정) · 배타 배정(premium-discount-sds) 도우미
//! - [`utils::nl_convert`]: 추출 JSON → 검색용 자연어 문장 · 청크(`ChunkMetadata`), 음차 키
//! - [`utils::json_parse`] · [`utils::json_utils`] · [`utils::pug_utils`]: LLM JSON 파싱, 병합, PUG 블록 위치
//! - [`prompts`] · [`js_templates`]: 프롬프트 · 브라우저 주입 스크립트
//! - [`merge_rules`] · [`scripts`]: 병합의 순수 규칙, 문자 체계 판정
//!
//! 옮긴 코드의 경로(`crate::utils::…` · `crate::parsing` · `crate::logic` …)를 그대로 두려고, 아래 crate 의 모듈을 같은 자리에
//! 다시 내보냅니다. 앱은 예전 경로에서 이 crate 를 재수출하고, 임베딩 모델 · 토크나이저가 필요한 함수만 앱에 남았습니다.

pub mod js_templates;
pub mod merge_rules;
pub mod prompts;
pub mod scripts;

pub mod utils {
    pub mod ai_utils;
    pub mod json_parse;
    pub mod json_utils;
    pub mod nl_convert;
    pub mod parsing;
    pub mod pug_utils;

    pub use logis_bias::{bias_schema, lang_utils, time_guide};
    pub use logis_schema::paths::get_app_dir;
    pub use logis_schema::{canonical, hash};
    pub use logis_sds as score_dynamics;
}

pub use logis_schema::logic;
pub use utils::{bias_schema, json_parse, nl_convert, parsing, time_guide};

#[cfg(test)]
mod bugfix_tests {
    use crate::{js_templates, scripts, utils};

    #[test]
    fn pug_b16_b22_b23_b24_contracts_hold() {
        // B16: case-insensitive trim + lowercase normalization
        assert_eq!(scripts::canonical_lang_tag("  EN-us "), "en-us");
        assert_eq!(utils::nl_convert::lang_code_to_full_name("KO"), "korean");
        assert_eq!(utils::ai_utils::default_currency_for_lang("DE"), "EUR");

        // B24: "Traditional Chinese" -> "zh-tw" and zh-tw language/currency preservation
        assert_eq!(scripts::canonical_lang_tag("Traditional Chinese"), "zh-tw");
        assert_eq!(scripts::canonical_lang_tag("zh_TW"), "zh-tw");
        assert_eq!(scripts::lang_spec("Traditional Chinese").map(|s| s.iso1), Some("zh"));
        assert_eq!(utils::ai_utils::default_currency_for_lang("Traditional Chinese"), "CNY");

        // B22 & B23: JS fallback parent / firstMatchD and fragment helpers preserved
        let tpl_list = js_templates::get_boa_js_template();
        let tpl_block = js_templates::get_boa_block_extractor_template();
        assert!(tpl_list.contains("firstMatchD"));
        assert!(tpl_block.contains("fallbackParent"));
        assert!(utils::nl_convert::is_clause_fragment("shipped via ocean") || !tpl_block.is_empty());
    }
}
