// 🌟 [LOGIS-LLM] 텍스트 · VL 생성 모델(qwen · qwen3 · qwen3_5 · qwen3vl · qwen4_exp · granite)과 공용 층(common) · 비전 임베딩
//    캐시는 crates/logis-llm 으로 옮겼습니다. 예전 경로 crate::models::… 를 그대로 쓰도록 재수출합니다.
//    임베딩(granite 임베딩 · Stanza 와 함께 PD nlp 대조 대기)과 SigLIP2(문서 비전, logis-vision-doc 예정)는 아직 앱에 있습니다.
pub use logis_llm::models::{common, granite, qwen, qwen3, qwen3_5, qwen3vl, qwen4_exp, vision_cache};
pub mod embedding;
pub mod siglip2;
