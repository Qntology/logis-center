//! Web UI module — serves a minimal HTML chat interface from the Axum server.
//!
//! # Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `GET` | `/ui` | Serves the embedded [`CHAT_UI_HTML`] page |
//! | `GET` | `/ui/health` | Returns `{"status":"ok"}` |
//!
//! The HTML is embedded at compile time via [`include_str!`] from
//! `assets/chat.html`.  No runtime file I/O is required.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxibonsai_runtime::web_ui::create_ui_router;
//!
//! let router = create_ui_router();
//! // Merge into an existing Axum router:
//! // let app = existing_router.merge(router);
//! ```

use axum::{
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

// ─── Embedded HTML ────────────────────────────────────────────────────────────

/// The single-page chat UI, embedded from `assets/chat.html` at compile time.
///
/// This is a pure HTML/CSS/JS application with no external CDN dependencies.
/// It communicates with the inference server via `POST /v1/chat/completions`.
pub const CHAT_UI_HTML: &str = include_str!("../assets/chat.html");

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// Serve the embedded chat UI as `text/html; charset=utf-8`.
async fn serve_chat_ui() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        CHAT_UI_HTML,
    )
        .into_response()
}

/// Liveness probe for the UI sub-router.
///
/// Returns `200 OK` with body `{"status":"ok"}`.
async fn ui_health() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        r#"{"status":"ok"}"#,
    )
        .into_response()
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the UI sub-router.
///
/// Routes:
/// - `GET /ui`        → `serve_chat_ui`
/// - `GET /ui/health` → `ui_health`
///
/// Merge this into an existing [`axum::Router`] with
/// [`Router::merge`](axum::Router::merge).
pub fn create_ui_router() -> Router {
    Router::new()
        .route("/ui", get(serve_chat_ui))
        .route("/ui/health", get(ui_health))
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_ui_html_is_nonempty() {
        assert!(!CHAT_UI_HTML.is_empty(), "CHAT_UI_HTML must not be empty");
    }

    #[test]
    fn chat_ui_html_contains_doctype() {
        assert!(
            CHAT_UI_HTML.contains("<!DOCTYPE html>"),
            "HTML should start with DOCTYPE"
        );
    }

    #[test]
    fn chat_ui_html_contains_fetch() {
        assert!(
            CHAT_UI_HTML.contains("fetch"),
            "HTML should use fetch() for API calls"
        );
    }

    #[test]
    fn create_ui_router_does_not_panic() {
        let _router = create_ui_router();
    }
}
