//! 생성 진행 이벤트 채널과 그 이벤트에 붙는 화면 분류 (예전 앱 `scheduler::PROGRESS_TX` · `lib.rs` 의 `CURRENT_UI_CATEGORY`).
//!
//! 스케줄러가 처음 시작할 때 채널을 채우고, Qwen 생성 루프가 토큰 진행을 보냅니다. 화면 분류는 앱 명령이 바꾸고 모델이 읽습니다.
//! 앱은 예전 경로에서 이 둘을 재수출하므로 정의는 여기 하나뿐입니다.

use once_cell::sync::{Lazy, OnceCell};
use std::sync::RwLock;

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();

pub static CURRENT_UI_CATEGORY: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::from("Processing")));
