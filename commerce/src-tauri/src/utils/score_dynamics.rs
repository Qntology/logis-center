//! SDS 원장은 crates/logis-sds 로 옮겼습니다. 예전 경로(crate::utils::score_dynamics · crate::score_dynamics)를 그대로 씁니다.
pub use logis_sds::*;

/// 스코프에 들어갑니다. logis-sds 는 모델 계층을 모르므로, 비전 트랙 키의 정밀도 꼬리표(SigLIP2 양자화 지문)를 여기서 먼저 등록합니다.
pub fn enter_scope(team: &str, track: Track, primary: &str, secondary: &str) {
    logis_sds::set_vision_precision(crate::models::siglip2::quant::fingerprint_suffix);
    logis_sds::enter_scope(team, track, primary, secondary)
}
