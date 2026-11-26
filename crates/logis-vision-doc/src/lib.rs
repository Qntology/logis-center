//! logis-center 의 문서 비전 crate (Tauri · DB 없음).
//!
//! [`siglip2`] 하나에 예전 앱 `models/siglip2/` 를 그대로 옮겼습니다.
//!
//! - 타워 · 적재: `vision` · `text` · `tokenizer` · `preprocessor`(NaFlex) · `gguf` · `gguf_quant` · `quant`(사용 시 양자화 · 정밀도 지문)
//! - 문서 · 서식 판정: `vision_encoder`(카테고리 히트맵 · 2단계 문서 분류) · `legibility`(잉크 · 선명도 게이트) · `vision_crop`(패치 소유권 ·
//!   크롭 계획) · `nms_arena` · `value_grounding` · `layout_cache` · `phrase_cache`
//!
//! 타워(vision · text · tokenizer · quant · preprocessor)는 premium-discount-vision · -sensor 로 바꾸지 않고 이 crate 에 둡니다. 실제 가중치
//! (google/siglip2-so400m-patch16-naflex, CPU F32)로 같은 입력을 넣어 대조한 결과(2026-11-18, 문장 60 · 이미지 8, 두 저장소의
//! `siglip2_dump` 시험), premium-discount 판으로 바꾸면 이 앱의 출력이 바뀝니다:
//! - 텍스트: 같은 토큰이면 벡터 코사인 1.000000 (타워 수치는 같음). 토큰 정규화가 달라 60개 중 8개가 다른 토큰 — 이 앱은 영숫자가 아닌
//!   글자를 공백으로 바꾸고 특수 토큰 · 마스크를 쓰며, premium-discount 는 ASCII 구두점을 지우고 EOS 뒤를 0 으로 채움 ("HS-8471.30" 코사인 0.749).
//! - 비전: 전처리 리사이즈가 8비트 한 단계씩 다르고(image Triangle vs PIL 방식), 위치 임베딩 보간이 다름(이 앱은 클램프 bilinear,
//!   premium-discount · transformers 는 antialias). 풀링 벡터 코사인 0.946~0.999, 패치 공유공간 행 평균 0.73~0.89 (보간이 없는 16×16 격자만 0.997).
//! - GPU 기본 경로(ROCm q8): 이 앱은 원본에서 한 번 만든 GGUF Q8_0 사본 · 정밀도 지문(`quant`), premium-discount 는 불러올 때 F32 에서
//!   Q8_0 로 양자화하고 MLP 폭을 4304 → 4320 으로 덧댐.
//! 이 앱의 비전 게이트 · 크롭 계획 · 값 그라운딩 · 합성 평가 기준선은 지금 출력에 맞춰져 있어, 바꾸려면 출력 변화를 받아들이고 다시
//! 맞춰야 합니다. 앱은 예전 경로 `crate::models::siglip2` 에서 이 모듈을 재수출합니다.

pub mod facade;
pub mod siglip2;

pub use facade::{
    check_trade_outbound_clean, device_pref_from_env, heavy4b_q8_profile, preprocess_logis_doc_v1,
    repair_and_verify_container, retry_bbox_with_margin, reverse_table_grid,
    rocm_fast_exit_if_used, verify_trade_ledger_shadow, zh_tw_scope_labels, TradeDocDomainBank,
    TradeGroundingLexicon,
};
