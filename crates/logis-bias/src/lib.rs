//! logis-center 의 다국어 도메인 사전 crate (모델 · DB · Tauri 없음).
//!
//! - [`bias_schema`]: 내장 사전(`bias.json` 개발 ko · en / `full_bias.json` 배포 12개 언어, `full-bias` 기능)과 앱 폴더의
//!   `bias.override.json` 덮어쓰기, 페이지 · 서식 유형, 레이아웃 · 목록 · 상세 스키마 필드, 비전 추적 편향
//! - [`bank`]: 필드별 라벨 · 선입견 구 뱅크 (12개 언어 순서)
//! - [`time_guide`]: 결정론적 시간 가이드 · 기간 계산(`LOGIS_EVAL_NOW_MS`) · 기간 어휘 근거
//! - [`lang_utils`]: 문서 언어 판정 (현지 언어 + 영어), [`query_lang`]: 질의 글의 언어 · 따옴표 구간 · 앱 언어 코드
//! - [`entity`]: 엔터티 키 정규화 · index · id · bcc · 릴레이 유형 키
//!
//! 앱은 예전 경로(`crate::utils::bias_schema` · `crate::utils::time_guide` · `crate::scheduler::entity` …)에서 재수출합니다.

pub mod bank;
pub mod bias_schema;
pub mod entity;
pub mod lang_utils;
pub mod query_lang;
pub mod time_guide;
