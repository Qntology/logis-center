//! logis-center 의 생성 모델 엔진 crate (Tauri · DB 없음).
//!
//! - [`models`]: Qwen2.5-VL(`qwen`) · Qwen3 · Qwen3.5 · Qwen3-VL · Qwen4-Exp · granite 생성 루프, 공용 층(`common`: GGUF 적재 ·
//!   어휘 잘라 내기 · TurboQuant KV · 가속 점검), 비전 임베딩 캐시
//! - [`lang_gguf`]: Qwen3.5 언어 엔진 safetensors → Q4_K_M GGUF 변환 · 적재 (SigLIP2 양자화가 쓰는 GGUF 쓰기 도우미 포함)
//! - [`tokenizer`] · [`chat_template`] · [`openai_types`] · [`position_embed`]: 토크나이저, Jinja 채팅 템플릿, 메시지 형식, RoPE
//! - [`utils`]: 장치 · VRAM · RAM · KV 배치(`resources`), 텐서 · 이미지 · 직접 적재 · KV 블록 암호화
//! - [`progress`] · [`lang_resident`] · [`rt`]: 앱이 채우는 진행 채널과 화면 분류, 올라 있는 언어 엔진 변형, 비동기 작업 띄우기
//!
//! 옮긴 코드의 경로(`crate::models::…` · `crate::utils::…` · `crate::tokenizer` …)를 그대로 두려고 앱과 같은 모듈 이름을 씁니다.
//! `crate::utils` 는 아래 crate 의 모듈(사전 · 경로 · SDS)도 같은 자리에 다시 내보냅니다. 앱은 예전 경로에서 이 crate 를 재수출합니다.

pub mod chat_template;
pub mod lang_gguf;
pub mod lang_resident;
pub mod models;
pub mod openai_types;
pub mod position_embed;
pub mod progress;
pub mod rt;
pub mod tokenizer;
pub mod utils;
