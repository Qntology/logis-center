//! 🌟 [EVENT SINK] 진행 상태 · 이벤트를 UI(Tauri 창) 또는 헤드리스(시험 · 콘솔 · 합성 평가)로 내보내는 추상화.
//!
//! model · scheduler 는 `tauri::AppHandle` 대신 [`Sink`](= `Arc<dyn EventSink>`)를 받습니다. 이 모듈 밖에서 그 핸들로 하던 일은
//! 이벤트 내보내기(`emit`)와 복제뿐이었으므로, `dyn EventSink` 에 같은 모양의 `emit` 을 두어 호출부는 그대로입니다.
//! Tauri 창에는 셸(apps/desktop/src/sink.rs)의 `TauriSink`, 시험 · CLI 에는 [`NoopSink`] 를 넘깁니다.
//! (기획서의 `apps/core/src/sink.rs`: 코어를 이 crate 로 나누며 `TauriSink` 만 셸에 남았습니다.)

use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

/// UI(Tauri) 또는 헤드리스 시험 · 콘솔로 진행 상태와 이벤트를 전달하는 추상 싱크
pub trait EventSink: Send + Sync + 'static {
    fn emit_json(&self, event: &str, payload: Value);

    fn log_progress(&self, mode: &str, payload: &Value) {
        self.emit_json(&format!("{mode}-progress"), payload.clone());
    }
}

/// model · scheduler 가 들고 다니는 싱크 (복제는 Arc 하나)
pub type Sink = Arc<dyn EventSink>;

impl dyn EventSink {
    /// `tauri::Emitter::emit` 과 같은 모양: 값을 JSON 으로 바꿔 [`EventSink::emit_json`] 으로 보냅니다.
    pub fn emit<S: Serialize>(&self, event: &str, payload: S) -> Result<(), serde_json::Error> {
        let value = serde_json::to_value(payload)?;
        self.emit_json(event, value);
        Ok(())
    }
}

/// 시험 · CLI · 합성 데이터 평가용 무동작 싱크
pub struct NoopSink;

impl EventSink for NoopSink {
    fn emit_json(&self, _event: &str, _payload: Value) {}
}

/// 아무 데도 보내지 않는 싱크
pub fn noop() -> Sink {
    Arc::new(NoopSink)
}

