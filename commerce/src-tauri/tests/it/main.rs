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
//! - `app_*`         : 앱 계층 (logic / store / scheduler / model::merge)

mod common;

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
// 앱 계층 (업무 규칙, LanceDB 저장소, 스케줄러, LLM 출력 병합)
mod app_logic;
mod app_store;
mod app_scheduler;
mod app_model;
