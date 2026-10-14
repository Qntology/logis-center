//! logis-center 통합 테스트 (src-tauri/tests)
//!
//! 링크 시간을 줄이기 위해 하나의 테스트 바이너리(`it`)로 묶었습니다.
//!
//! 리눅스 실행 예:
//!   ORT_STRATEGY=system ORT_LIB_LOCATION=<onnxruntime-linux-x64-1.28.0> \
//!   cargo test --test it --no-default-features --features vulkan
//!
//! GPU 가 없는 리눅스(클라우드)에서는 Mesa llvmpipe 를 쓰도록 다음도 지정합니다:
//!   CANDLE_VULKAN_ALLOW_CPU=1   (포크는 기본적으로 CPU 타입 Vulkan 디바이스를 건너뜀)
//! 앱 데이터 디렉터리 오염을 막으려면 XDG_DATA_HOME 을 임시 디렉터리로 지정하세요.
//! io_uring 폴백을 실패로 취급하려면 LOGIS_REQUIRE_IO_URING=1.
//! 알려진 버그를 문서화한 테스트는 #[ignore = "BUG(..)"] 이며 `-- --ignored` 로 확인합니다.
//!
//! - `direct_loader` : SSD 오프로딩 I/O (Linux=io_uring, Windows=DirectStorage)
//! - `parsers`       : SyntheticData/trading 합성 PDF 로 문서 파서 검증
//! - `vulkan`        : (feature="vulkan") Vulkan 디바이스 연산을 CPU 결과와 대조
//! - `utils_*`       : utils 계층 순수 로직
//! - `models_cpu`    : 모델 보조 로직 (CPU)
//! - `qwen4_exp`     : Qwen4-Exp 포트 정확도 (tests/fixtures/qwen4_exp_tiny, HF 기준 logits)
//! - `app_*`         : 앱 계층 (logic / store / scheduler / model::merge)

mod common;

/// 앱(src/main.rs)과 같은 전역 할당기. 메모리 반환 테스트(gpu_memory)가 출시 앱의 할당기 동작
/// (mi_collect 로 빈 페이지 반환)을 재도록 합니다.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ── 1단계: 플랫폼 검증 (SSD 오프로딩 I/O · 합성 문서 · GPU) ──
mod direct_loader;
mod parsers;
#[cfg(feature = "vulkan")]
mod vulkan;

// ── 2단계: 계층별 테스트 ──
// utils (순수 로직: 식별자/해시, 텍스트·숫자·날짜, 스키마·파싱·NL)
mod utils_identity;
mod utils_text;
mod utils_schema;
// models (CPU: 위치 임베딩, 텐서 유틸, KV 레지스트리, 비전 캐시, 설정)
mod models_cpu;
// Qwen4-Exp candle 포트 ↔ transformers 기준 logits (작은 무작위 체크포인트, CPU + CUDA)
mod qwen4_exp;
// 앱 계층 (업무 규칙, LanceDB 저장소, 스케줄러, LLM 출력 병합)
mod app_logic;
mod app_store;
mod app_scheduler;
mod app_model;
// 임베딩 처리량 · NMS 배틀 가속 (전역 캐시 · NVML 핸들 · 프리패치 동치 · 배치 순전파 GPU 대조)
mod app_embed;
// 메모리 수명주기 · KV 계획 v2 (증분 가격 · 비례 여유 · 히스테리시스 · F16 원본 KV · 공유 장치 해제)
mod mem_lifecycle;
// P2b 실모델 대조: 융합 GDN(Qwen3.5) · Qwen3-0.6B Q8_0 상주 정확 W8A16 (LOGIS_TEST_MODELS=1, --ignored)
mod p2b_models;
// FlashAttention / TurboQuant 정확도 teacher-forcing (목업 Qwen3, FA on/off × TQ KV on/off, CPU + GPU)
mod accel_accuracy;
// GPU 메모리 보고 · 반환 · KV 계층 강제 (장치 공통: Metal · Vulkan · CUDA · ROCm)
mod gpu_memory;
mod qwen35_kv;
