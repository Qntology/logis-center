//! logis-center 의 검색 규칙 crate (저장소 · 모델 · Tauri 없음).
//!
//! - [`fusion`]: 질의 전체 트랙과 문서 트랙의 융합 (`FUSION_K` · `LITERAL_WEIGHT` · 질의 낱말 · 도메인)
//! - [`relation`]: 질의 속 고정점(식별자 · 사람 이름 · 제목 · 상태 · 서식 번호) → 목표 유형으로 건너가는 관계 단계의 규칙,
//!   서식 질의 근거, 음성 고정점, 교차 언어 제목 조각 고르기
//!
//! 저장소(LanceDB) · 임베딩 모델을 쓰는 단계 함수(`relation_join` · `trade_relation_join` · `cross_lingual_title_anchors`)는
//! 앱의 `search_relation.rs` 에 남고, 앱은 예전 경로(`crate::search_fusion` · `crate::search_relation`)에서 이 crate 를 재수출합니다.

pub mod fusion;
pub mod relation;
