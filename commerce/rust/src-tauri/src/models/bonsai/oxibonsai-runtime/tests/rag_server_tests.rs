//! Integration tests for the RAG HTTP endpoints.
//!
//! Uses `tower::ServiceExt::oneshot` to exercise handlers in-process,
//! mirroring the pattern used in `server_tests.rs`.

#[cfg(feature = "rag")]
mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::rag_server::create_rag_router;
    use oxibonsai_runtime::sampling::SamplingParams;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn test_rag_router() -> axum::Router {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        create_rag_router(engine)
    }

    fn json_body(value: serde_json::Value) -> Body {
        Body::from(serde_json::to_string(&value).expect("serialize"))
    }

    fn json_request(method: Method, path: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(json_body(body))
            .expect("build request")
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("parse JSON")
    }

    // ── test_index_documents_returns_200 ─────────────────────────────────────

    #[tokio::test]
    async fn test_index_documents_returns_200() {
        let app = test_rag_router();

        let req = json_request(
            Method::POST,
            "/rag/index",
            serde_json::json!({
                "documents": [
                    "Rust is a systems programming language emphasising memory safety.",
                    "Axum is an ergonomic and modular web framework built on Tokio.",
                    "RAG combines retrieval with language model generation."
                ]
            }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK, "expected 200 OK");

        let json = body_json(resp).await;
        assert_eq!(
            json["indexed"], 3,
            "three documents should be indexed; got {json}"
        );
        assert!(
            json["chunks"].as_u64().unwrap_or(0) >= 1,
            "should produce at least one chunk; got {json}"
        );
        let ids = json["document_ids"].as_array().expect("document_ids array");
        assert_eq!(ids.len(), 3, "should have one id per document");
    }

    // ── test_index_empty_documents_handled ───────────────────────────────────

    #[tokio::test]
    async fn test_index_empty_documents_handled() {
        let app = test_rag_router();

        let req = json_request(
            Method::POST,
            "/rag/index",
            serde_json::json!({ "documents": [] }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "empty documents list should return 400"
        );

        let json = body_json(resp).await;
        assert!(
            json["error"].is_string(),
            "response should contain an error field; got {json}"
        );
    }

    // ── test_rag_stats_returns_json ───────────────────────────────────────────

    #[tokio::test]
    async fn test_rag_stats_returns_json() {
        let app = test_rag_router();

        let req = Request::get("/rag/stats")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert!(
            json["documents_indexed"].is_number(),
            "stats must contain documents_indexed; got {json}"
        );
        assert!(
            json["chunks_indexed"].is_number(),
            "stats must contain chunks_indexed; got {json}"
        );
        assert!(
            json["embedding_dim"].is_number(),
            "stats must contain embedding_dim; got {json}"
        );
        assert!(
            json["store_memory_bytes"].is_number(),
            "stats must contain store_memory_bytes; got {json}"
        );
        assert!(
            json["store_memory_human"].is_string(),
            "stats must contain store_memory_human; got {json}"
        );
    }

    // ── test_rag_query_returns_answer ─────────────────────────────────────────

    #[tokio::test]
    async fn test_rag_query_returns_answer() {
        // We need two separate one-shot calls; use a make_service approach or
        // build the router twice.  For simplicity we build two routers backed
        // by independent state: one to index, one to query — but then they
        // share no state.  Instead, we verify the query endpoint works even
        // when the store is empty (pipeline answers with empty context).
        let app = test_rag_router();

        let req = json_request(
            Method::POST,
            "/rag/query",
            serde_json::json!({
                "query": "What is Rust?",
                "max_tokens": 5
            }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "rag query should return 200 even with empty index"
        );

        let json = body_json(resp).await;
        assert!(
            json["answer"].is_string(),
            "response must have answer field; got {json}"
        );
        assert!(
            json["prompt_used"].is_string(),
            "response must have prompt_used; got {json}"
        );
        assert!(
            json["usage"].is_object(),
            "response must have usage; got {json}"
        );
        let usage = &json["usage"];
        assert!(usage["completion_tokens"].is_number());
        assert!(usage["prompt_tokens"].is_number());
        assert!(usage["chunks_retrieved"].is_number());
        assert!(usage["documents_searched"].is_number());
    }

    // ── test_clear_index_resets_stats ─────────────────────────────────────────

    #[tokio::test]
    async fn test_clear_index_resets_stats() {
        // Build a router backed by its own shared state.
        let config = Qwen3Config::tiny_test();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        let router = create_rag_router(engine);

        // We need multiple requests on the same router state.  axum routers
        // implement `Service + Clone`, so we can call `into_make_service` and
        // use `tower_service::Service::call` directly.  Simpler: use the router
        // via `axum::serve` in a test server — but that requires a port.
        //
        // The easiest approach: wrap the router in a `Arc<Mutex<Router>>` or
        // share via `axum::Router::into_make_service` with a memory-listener.
        // For this test we use `hyper`'s in-memory transport via axum helpers.
        //
        // Actually the simplest correct approach: use the router directly via
        // `axum::Router`'s `Service` impl.  Each `oneshot` consumes the router,
        // but we can clone it first (Router: Clone).
        let index_req = json_request(
            Method::POST,
            "/rag/index",
            serde_json::json!({
                "documents": [
                    "Rust memory safety ensures freedom from data races.",
                    "Axum is built on top of Hyper and Tokio."
                ]
            }),
        );

        let resp = router
            .clone()
            .oneshot(index_req)
            .await
            .expect("index response");
        assert_eq!(resp.status(), StatusCode::OK, "indexing should succeed");

        // After indexing, stats should show documents.
        let stats_resp = router
            .clone()
            .oneshot(Request::get("/rag/stats").body(Body::empty()).expect("req"))
            .await
            .expect("stats response");
        let stats = body_json(stats_resp).await;
        assert_eq!(
            stats["documents_indexed"], 2,
            "after indexing 2 docs, stats should show 2; got {stats}"
        );

        // Clear the index.
        let clear_req = Request::builder()
            .method(Method::DELETE)
            .uri("/rag/index")
            .body(Body::empty())
            .expect("delete request");
        let clear_resp = router
            .clone()
            .oneshot(clear_req)
            .await
            .expect("clear response");
        assert_eq!(clear_resp.status(), StatusCode::OK);
        let clear_json = body_json(clear_resp).await;
        assert_eq!(
            clear_json["status"], "cleared",
            "clear response should contain status:cleared; got {clear_json}"
        );

        // Stats after clearing should show 0 documents.
        let stats_after_resp = router
            .clone()
            .oneshot(Request::get("/rag/stats").body(Body::empty()).expect("req"))
            .await
            .expect("stats after clear");
        let stats_after = body_json(stats_after_resp).await;
        assert_eq!(
            stats_after["documents_indexed"], 0,
            "after clear, documents_indexed should be 0; got {stats_after}"
        );
        assert_eq!(
            stats_after["chunks_indexed"], 0,
            "after clear, chunks_indexed should be 0; got {stats_after}"
        );
    }

    // ── test_rag_query_includes_context_when_requested ────────────────────────

    #[tokio::test]
    async fn test_rag_query_includes_context_when_requested() {
        let app = test_rag_router();

        // Request with include_context: true
        let req = json_request(
            Method::POST,
            "/rag/query",
            serde_json::json!({
                "query": "Tell me about memory safety.",
                "max_tokens": 3,
                "include_context": true
            }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        // retrieved_chunks should be present (may be empty if store is empty,
        // but the field itself must not be null).
        assert!(
            json["retrieved_chunks"].is_array(),
            "retrieved_chunks should be an array when include_context=true; got {json}"
        );
    }

    #[tokio::test]
    async fn test_rag_query_omits_context_by_default() {
        let app = test_rag_router();

        let req = json_request(
            Method::POST,
            "/rag/query",
            serde_json::json!({
                "query": "What is Axum?",
                "max_tokens": 3
            }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert!(
            json["retrieved_chunks"].is_null(),
            "retrieved_chunks should be absent when include_context is not set; got {json}"
        );
    }

    // ── test_rag_router_wires_all_routes ──────────────────────────────────────

    #[tokio::test]
    async fn test_rag_router_wires_all_routes() {
        // Verify that all four expected routes are registered by issuing a
        // request to each and confirming we do NOT get 404/405 Method Not
        // Allowed for routes that should exist.

        // GET /rag/stats
        {
            let app = test_rag_router();
            let req = Request::get("/rag/stats").body(Body::empty()).expect("req");
            let resp = app.oneshot(req).await.expect("resp");
            assert_ne!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "GET /rag/stats should be registered"
            );
        }

        // POST /rag/index
        {
            let app = test_rag_router();
            let req = json_request(
                Method::POST,
                "/rag/index",
                serde_json::json!({ "documents": ["hello world this is a test document"] }),
            );
            let resp = app.oneshot(req).await.expect("resp");
            assert_ne!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "POST /rag/index should be registered"
            );
        }

        // DELETE /rag/index
        {
            let app = test_rag_router();
            let req = Request::builder()
                .method(Method::DELETE)
                .uri("/rag/index")
                .body(Body::empty())
                .expect("req");
            let resp = app.oneshot(req).await.expect("resp");
            assert_ne!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "DELETE /rag/index should be registered"
            );
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "DELETE /rag/index should return 200"
            );
        }

        // POST /rag/query
        {
            let app = test_rag_router();
            let req = json_request(
                Method::POST,
                "/rag/query",
                serde_json::json!({ "query": "test query", "max_tokens": 1 }),
            );
            let resp = app.oneshot(req).await.expect("resp");
            assert_ne!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "POST /rag/query should be registered"
            );
        }
    }

    // ── test_rag_query_rejects_empty_query ────────────────────────────────────

    #[tokio::test]
    async fn test_rag_query_rejects_empty_query() {
        let app = test_rag_router();

        let req = json_request(
            Method::POST,
            "/rag/query",
            serde_json::json!({ "query": "   " }),
        );

        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "blank query should return 400"
        );

        let json = body_json(resp).await;
        assert!(
            json["error"].is_string(),
            "error field should be present; got {json}"
        );
    }
}
