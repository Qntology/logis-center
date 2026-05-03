//! Integration tests for the admin API.
//!
//! Each test spins up the admin router directly (without a real TCP socket)
//! using `axum::Router::into_service` + `tower::ServiceExt::oneshot`.

#[cfg(feature = "server")]
mod admin_api_tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use oxibonsai_runtime::{
        admin::{create_admin_router, features_enabled, AdminState},
        metrics::InferenceMetrics,
    };
    use std::sync::Arc;
    use tower::ServiceExt; // for `.oneshot()`

    /// Build a fresh router backed by a new AdminState.
    fn build_router() -> (axum::Router, Arc<AdminState>) {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(metrics));
        let router = create_admin_router(Arc::clone(&state)).with_state(Arc::clone(&state));
        (router, state)
    }

    // ── GET /admin/status ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_status_returns_200() {
        let (router, _state) = build_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/admin/status")
                    .method("GET")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    // ── GET /admin/config ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_config_returns_200() {
        let (router, _state) = build_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/admin/config")
                    .method("GET")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    // ── POST /admin/reset-metrics ──────────────────────────────────────────

    #[tokio::test]
    async fn test_reset_metrics_returns_200() {
        let (router, _state) = build_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/admin/reset-metrics")
                    .method("POST")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    // ── GET /admin/cache-stats ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_cache_stats_returns_200() {
        let (router, _state) = build_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/admin/cache-stats")
                    .method("GET")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    // ── AdminState::uptime_secs ────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_state_uptime() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = AdminState::new(metrics);
        let uptime = state.uptime_secs();
        assert!(
            uptime < 5,
            "uptime should be nearly 0 right after construction; got {uptime}"
        );
    }

    // ── features_enabled returns non-empty list ────────────────────────────

    #[tokio::test]
    async fn test_features_enabled_non_empty() {
        let features = features_enabled();
        assert!(
            !features.is_empty(),
            "features_enabled() must return at least one entry"
        );
        assert!(
            features.contains(&"runtime".to_owned()),
            "expected 'runtime' in features list; got {features:?}"
        );
    }

    // ── status body contains expected JSON fields ──────────────────────────

    #[tokio::test]
    async fn test_status_body_has_version_field() {
        let (router, _state) = build_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/admin/status")
                    .method("GET")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("should read body");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body should be valid JSON");

        assert!(
            json.get("version").is_some(),
            "status JSON should include 'version'"
        );
        assert!(
            json.get("uptime_secs").is_some(),
            "status JSON should include 'uptime_secs'"
        );
        assert!(
            json.get("requests_total").is_some(),
            "status JSON should include 'requests_total'"
        );
    }
}
