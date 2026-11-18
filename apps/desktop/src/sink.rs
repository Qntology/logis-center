//! 🌟 [EVENT SINK] Tauri 창으로 내보내는 싱크. 추상화(`EventSink` · `Sink` · `NoopSink` · `noop`)는 헤드리스 코어(apps/core)의
//! `sink` 이고, 여기서 같은 이름으로 다시 내보내 명령 · 통합 시험의 경로(`crate::sink::…` · `tauri_app_lib::sink::…`)가 그대로입니다.

use std::sync::Arc;

use serde_json::Value;

pub use logis_app::sink::*;

/// Tauri 창으로 보내는 싱크 (`AppHandle::emit`)
pub struct TauriSink(pub tauri::AppHandle);

impl EventSink for TauriSink {
    fn emit_json(&self, event: &str, mut payload: Value) {
        use tauri::Emitter;
        // 화면으로 나가는 JSON 의 키 순서를 예전(키 정렬)과 같게 (serde_json preserve_order 대비, logis_app::json_order)
        logis_app::json_order::sort_all(&mut payload);
        let _ = self.0.emit(event, payload);
    }
}

/// 앱 핸들을 싱크로 감쌉니다 (명령 진입점에서 한 번).
pub fn tauri(app: &tauri::AppHandle) -> Sink {
    Arc::new(TauriSink(app.clone()))
}
