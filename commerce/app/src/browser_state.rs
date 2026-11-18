//! 브라우저 자동화(셸의 automation.rs)가 감지한 지금 활성 탭의 상태. 자동화가 쓰고, 코어의 URL 도우미(`utils::url_utils`)가
//! 상대 주소를 절대 주소로 합칠 때 읽습니다. 예전에는 automation.rs 에 있었습니다.

use once_cell::sync::Lazy;
use std::sync::Arc;

// Global storage for the last detected browser state
pub struct DetectedState {
    pub url: String,
    pub tab_id: String,
    pub is_client: bool,
    pub is_admin: bool,
}

pub static LAST_DETECTED_STATE: Lazy<Arc<tokio::sync::Mutex<DetectedState>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(DetectedState {
        url: String::new(),
        tab_id: String::new(),
        is_client: false,
        is_admin: false,
    }))
});
