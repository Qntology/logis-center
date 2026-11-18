//! logis-center 의 데이터 계약 crate: 모델 · DB · Tauri · tokio 없이 순수 함수만 둡니다.
//!
//! - [`canonical`]: 필드 이름 · 접미사로 저장 타입 확정(`CanonKind`), 릴레이 원장 키, 포장 축 모순 제거
//! - [`logic`]: 무역 6대 그룹 · 28종 서식 · 허브 필드(PO → CI → BL → LC), 질의 · 병합 정보
//! - [`hash`]: hashId(premium-discount-identity 위) · 숫자 동형문자 정규화 · 릴레이 키 · crc32 · 기준 도메인
//! - [`text`]: 숫자 · 월 이름 · 구(phrase) 나누기 (정규화 · 무역 규칙 · 도메인 사전이 함께 씀)
//! - [`embed_space`]: granite 임베딩 차원 · 순전파 모드 · 코사인 눈금 보정 표
//! - [`envelope`]: 저장소 봉투 컬럼, [`paths`]: 앱 데이터 폴더 · tmp(kv · 작업 데이터 · 로그) · 추출 중단 신호
//! - [`json_order`]: JSON 객체 키 순서를 예전(키 정렬)과 같게 — 순회 · 직렬화 도우미 (serde_json `preserve_order` 가 켜져도 같은 출력)
//!
//! 앱(apps/desktop)은 예전 경로(`crate::utils::canonical` · `crate::logic` · `crate::utils::hash` …)에서 이 모듈을
//! 그대로 재수출하므로 호출부는 바뀌지 않았습니다.

pub mod canonical;
pub mod embed_space;
pub mod envelope;
pub mod hash;
pub mod json_order;
pub mod logic;
pub mod paths;
pub mod text;
