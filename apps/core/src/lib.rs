//! logis-center 헤드리스 코어 (crate `logis-app`, 기획서 4단계의 `apps/core`): Tauri 없이 도는 앱 계층.
//!
//! - [`model`]: 모델 상주 · 전환(`residency` · `lifecycle`), 생성 · 비전 진입점, 언어 엔진(AlphaEdge) 관리, 병합 · 리서치 · 도메인 질의
//! - [`store`]: LanceDB 벡터 저장소(봉투 스키마), [`scheduler`]: 추출 · 무역 파이프라인 · 음차 · 색인 · 릴레이 원장 작업자
//! - [`stanza`] · [`search_fusion`] · [`search_relation`]: 질의 구문 분석과 검색 융합 · 관계 단계
//! - [`sink`]: 진행 이벤트 추상화(`EventSink` · `NoopSink`) — Tauri 창으로 보내는 `TauriSink` 는 셸(apps/desktop)에 있습니다
//! - [`utils`] · [`models`] · [`logic`] · [`prompts`] · [`js_templates`] · [`parsers`]: 예전 경로 그대로의 재수출과 모델 · 저장소가 필요한 도우미
//!
//! 예전에는 Tauri 앱 crate(`tauri_app_lib`) 안에 있었습니다. 옮긴 코드의 `crate::…` 경로를 그대로 두려고 예전 lib.rs 루트의 재수출과
//! 코어가 읽는 전역 상태를 이 crate 루트에 둡니다. 셸은 이 모듈들을 같은 이름으로 다시 내보내므로 명령 · 통합 시험의 경로는 바뀌지 않았습니다.

use once_cell::sync::Lazy;
use serde_json::Value;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex as TokioMutex;

use crate::model::LogisModel;
use crate::store::VectorStore;

pub mod model;
pub mod store;
pub mod logic;
pub mod scheduler;
pub mod stanza;
pub mod search_fusion;
pub mod search_relation;
pub mod js_templates;
pub mod prompts;
pub mod parsers;
pub mod models;
pub mod utils;
// 🌟 [EVENT SINK] model · scheduler 는 tauri::AppHandle 대신 Sink 로 이벤트를 내보냅니다 (셸의 명령 진입점에서 TauriSink 로 감쌈).
pub mod sink;
pub mod browser_state;
// 🌟 [QUERY] 자연어 검색 (예전 셸 lib.rs 의 ai_search_complex 본문과 도우미)
pub mod query;
// 🌟 [COMMANDS] 화면 명령의 본문 (예전 셸 lib.rs 의 #[tauri::command] 함수들, 셸은 같은 이름의 껍데기)
pub mod commands;

pub use utils::parsing;
pub use utils::bias_schema;
pub use utils::json_parse;
pub use utils::nl_convert;
pub use utils::time_guide;
// 🌟 [SCORE DYNAMICS] 점수를 신호로 취급하는 관측·통계 계층.
//    bias.json 이 정적 사전이라면 이것은 동적 사전입니다.
//    Phase 0 에서는 계측과 영속화만 수행하고 판정에는 개입하지 않습니다.
pub use utils::score_dynamics;
// 🌟 [LOGIS-LLM] 텍스트 · VL 생성 모델 · 토크나이저 · 채팅 템플릿 · 위치 임베딩 · 장치 도우미는 crates/logis-llm 에 있습니다.
pub use logis_llm::{chat_template, openai_types, position_embed, tokenizer};
// JSON 객체 키 순서를 예전(키 정렬)과 같게 지키는 도우미 (셸도 이 경로로 씀)
pub use logis_schema::json_order;

// ── 코어가 읽고 셸(명령)도 읽는 전역 상태 (예전 lib.rs) ──
/// 지금 처리 중인 작업의 문맥 (스케줄러가 채우고 UI 명령이 읽음)
pub static ACTIVE_TASK_MEM: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));
/// 생성 진행 이벤트에 붙는 화면 분류 (모델이 읽으므로 crates/logis-llm 에 있음)
pub use logis_llm::progress::CURRENT_UI_CATEGORY;
/// 마지막 진행 이벤트 (화면이 늦게 붙어도 다시 보여 줄 수 있게)
pub static LATEST_PROGRESS_PAYLOAD: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));

// ── 셸 명령과 코어 검색이 함께 쓰는 상태 (예전 lib.rs) ──
// 🌟 [PUB] AI 검색이 도는 동안 임베딩 재색인 · 스케줄러가 VRAM 을 양보할 수 있도록 공개합니다.
pub static IS_SEARCHING: AtomicBool = AtomicBool::new(false);

pub struct AppState {
    pub model: Arc<TokioMutex<Option<LogisModel>>>,
    pub store: Arc<TokioMutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

/// Tauri 없이 코어를 쓸 때(통합 시험 · 합성 평가 · CLI) 엔진을 초기화합니다: 언어 엔진 런타임 로더 등록, KV 굽기 작업자 시작.
/// Tauri 앱은 `run()` 이 같은 두 가지를 합니다(로더는 시작할 때, 굽기 작업자는 setup 에서).
pub fn init_headless() {
    crate::model::lang_llm::register_loader(crate::models::qwen4_exp::registry::load_lang_runtime);
    crate::models::qwen::generate::init_bake_worker();
}
