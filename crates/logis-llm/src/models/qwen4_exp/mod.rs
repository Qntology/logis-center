//! Qwen4-Exp (transformers 5.16+ `qwen4_exp`, "Qwen3.8-Flash-Next" / Qwen 4 프리뷰) 텍스트 모델의 candle 포트.
//!
//! Qwen3.5 하이브리드(Gated DeltaNet + gated full attention) 에
//!  - GatedResidual 하이퍼커넥션 (hc 개 잔차 스트림 + GatedNorm, RMSNorm/최종 norm 대체) — [`hyper`]
//!  - Qwen Sparse Attention (블록 압축 키 인덱서, budget top-k + 꼬리 블록) — [`qsa`]
//!  - Per-Layer Embedding (해시 n-gram 임베딩 + 팽창 depthwise conv) — [`ple`]
//!  - MoE (라우터 softmax top-k + 시그모이드 게이트 공유 전문가) — [`moe`]
//! 를 더한 구조입니다. 기준 구현: transformers 5.17 `modeling_qwen4_exp.py` (명세: qwen4-spec).
//! 정확도 검증: `tests/it/qwen4_exp.rs` (작은 무작위 체크포인트, HF 기준 logits 대조, CPU + CUDA).
//!
//! 등록: [`registry::detect_text_arch`] 가 config.json 의 `model_type` 으로 판별하고,
//! [`registry::TextGenerator::load_dir`] 가 Qwen3.5 / Qwen4-Exp 생성기를 같은 `generate(..)` 형태로 돌려줍니다.
//! GGUF arch `qwen4exp` 는 규격이 정해지지 않아 명확한 미지원 오류를 냅니다.

pub mod config;
pub mod gdn;
pub mod generate;
pub mod hyper;
pub mod layers;
pub mod model;
pub mod moe;
pub mod ple;
pub mod qsa;
pub mod registry;

pub use config::Qwen4ExpTextConfig;
pub use generate::Qwen4ExpGenerateModel;
pub use model::Qwen4ExpForCausalLM;
pub use registry::{detect_text_arch, TextArch, TextGenerator};
