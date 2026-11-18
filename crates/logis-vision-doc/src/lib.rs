//! logis-center 의 문서 비전 crate (Tauri · DB 없음).
//!
//! [`siglip2`] 하나에 예전 앱 `models/siglip2/` 를 그대로 옮겼습니다.
//!
//! - 타워 · 적재: `vision` · `text` · `tokenizer` · `preprocessor`(NaFlex) · `gguf` · `gguf_quant` · `quant`(사용 시 양자화 · 정밀도 지문)
//! - 문서 · 서식 판정: `vision_encoder`(카테고리 히트맵 · 2단계 문서 분류) · `legibility`(잉크 · 선명도 게이트) · `vision_crop`(패치 소유권 ·
//!   크롭 계획) · `nms_arena` · `value_grounding` · `layout_cache` · `phrase_cache`
//!
//! 기획서는 타워(vision · text · tokenizer · quant · preprocessor)를 premium-discount-vision · -sensor 로 바꾸라고 하지만, 이 앱의 ROCm q8 GGUF
//! 경로 · 정밀도 지문과 HF 로더의 차이(열 순서 · antialias · bilinear · 토크나이저 패딩)를 실제 모델로 대조하기 전에는 바꾸지 않습니다.
//! 앱은 예전 경로 `crate::models::siglip2` 에서 이 모듈을 재수출합니다.

pub mod siglip2;
