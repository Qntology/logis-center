//! Integration tests for multi-model serving and LoRA adapter routing.

use oxibonsai_runtime::multi_model::{
    AdapterStack, EndpointStatus, ModelEndpoint, ModelId, ModelRegistry, ModelRouter, RoutingError,
};

// ─── ModelId ─────────────────────────────────────────────────────────────────

#[test]
fn model_id_new() {
    let id = ModelId::new("bonsai-8b");
    assert_eq!(id.as_str(), "bonsai-8b");
}

#[test]
fn model_id_is_base() {
    assert!(ModelId::new("bonsai-8b").is_base());
    assert!(!ModelId::new("bonsai-8b:lora").is_base());
}

#[test]
fn model_id_adapter_name() {
    assert_eq!(ModelId::new("bonsai-8b").adapter_name(), None);
    assert_eq!(
        ModelId::new("bonsai-8b:code-assist").adapter_name(),
        Some("code-assist")
    );
}

// ─── ModelEndpoint ───────────────────────────────────────────────────────────

#[test]
fn endpoint_new_defaults() {
    let ep = ModelEndpoint::new("bonsai-8b", "qwen3-8b");
    assert_eq!(ep.id.as_str(), "bonsai-8b");
    assert_eq!(ep.base_model, "qwen3-8b");
    assert_eq!(ep.status, EndpointStatus::Ready);
    assert!(ep.adapter.is_none());
    assert!(!ep.is_default);
    assert_eq!(ep.max_context_length, 4096);
}

#[test]
fn endpoint_with_adapter() {
    let ep = ModelEndpoint::new("bonsai-8b:code", "qwen3-8b").with_adapter("code-assist");
    assert_eq!(ep.adapter.as_deref(), Some("code-assist"));
}

#[test]
fn endpoint_status_available() {
    assert!(EndpointStatus::Ready.is_available());
    assert!(!EndpointStatus::Loading.is_available());
    assert!(!EndpointStatus::Error.is_available());
    assert!(!EndpointStatus::Disabled.is_available());
}

// ─── ModelRegistry ───────────────────────────────────────────────────────────

#[test]
fn registry_register() {
    let mut reg = ModelRegistry::new();
    assert_eq!(reg.len(), 0);
    assert!(reg.is_empty());

    reg.register(ModelEndpoint::new("bonsai-8b", "qwen3-8b"));
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
}

#[test]
fn registry_resolve_by_id() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("bonsai-8b", "qwen3-8b").with_description("Base 8B model"));

    let ep = reg.resolve("bonsai-8b");
    assert!(ep.is_some());
    assert_eq!(ep.map(|e| e.base_model.as_str()), Some("qwen3-8b"));
}

#[test]
fn registry_resolve_by_alias() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("bonsai-8b", "qwen3-8b"));
    reg.add_alias("gpt-4", ModelId::new("bonsai-8b"));

    let ep = reg.resolve("gpt-4");
    assert!(ep.is_some());
    assert_eq!(ep.map(|e| e.id.as_str()), Some("bonsai-8b"));
}

#[test]
fn registry_resolve_unknown() {
    let reg = ModelRegistry::new();
    assert!(reg.resolve("nonexistent").is_none());
}

#[test]
fn registry_default_endpoint() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("bonsai-8b", "qwen3-8b").set_default());

    let ep = reg.default_endpoint();
    assert!(ep.is_some());
    assert_eq!(ep.map(|e| e.id.as_str()), Some("bonsai-8b"));
}

#[test]
fn registry_available_endpoints() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("model-a", "base-a"));
    reg.register(ModelEndpoint::new("model-b", "base-b"));

    // Both should be available (default status is Ready).
    assert_eq!(reg.available_endpoints().len(), 2);

    // Disable one.
    reg.set_status(&ModelId::new("model-b"), EndpointStatus::Disabled);
    assert_eq!(reg.available_endpoints().len(), 1);
}

#[test]
fn registry_set_status() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("model-a", "base-a"));

    let changed = reg.set_status(&ModelId::new("model-a"), EndpointStatus::Error);
    assert!(changed);

    let ep = reg.resolve("model-a");
    assert_eq!(ep.map(|e| e.status), Some(EndpointStatus::Error));

    // Non-existent model returns false.
    let not_found = reg.set_status(&ModelId::new("no-such"), EndpointStatus::Ready);
    assert!(!not_found);
}

#[test]
fn registry_unregister() {
    let mut reg = ModelRegistry::new();
    reg.register(ModelEndpoint::new("bonsai-8b", "qwen3-8b").set_default());
    reg.add_alias("gpt-4", ModelId::new("bonsai-8b"));

    let removed = reg.unregister(&ModelId::new("bonsai-8b"));
    assert!(removed.is_some());
    assert_eq!(reg.len(), 0);

    // Default should be cleared.
    assert!(reg.default_endpoint().is_none());
    // Alias should be removed.
    assert!(reg.resolve("gpt-4").is_none());
}

// ─── ModelRouter ─────────────────────────────────────────────────────────────

fn make_test_router() -> ModelRouter {
    let mut reg = ModelRegistry::new();
    reg.register(
        ModelEndpoint::new("bonsai-8b", "qwen3-8b")
            .with_context_length(8192)
            .set_default(),
    );
    reg.register(ModelEndpoint::new("bonsai-70b", "qwen3-70b").with_context_length(32768));
    ModelRouter::new(reg)
}

#[test]
fn router_route_default() {
    let router = make_test_router();
    let ep = router.route(None);
    assert!(ep.is_ok());
    let endpoint = ep.expect("route should succeed");
    assert_eq!(endpoint.id.as_str(), "bonsai-8b");
}

#[test]
fn router_route_specific() {
    let router = make_test_router();
    let ep = router.route(Some("bonsai-70b"));
    assert!(ep.is_ok());
    let endpoint = ep.expect("route should succeed");
    assert_eq!(endpoint.id.as_str(), "bonsai-70b");
}

#[test]
fn router_route_not_found() {
    let router = make_test_router();
    let err = router.route(Some("nonexistent"));
    assert!(err.is_err());
    assert!(matches!(err, Err(RoutingError::ModelNotFound(_))));
}

#[test]
fn router_route_for_context() {
    let router = make_test_router();

    // Default model (8192 ctx) should handle 4096.
    let ep = router.route_for_context(None, 4096);
    assert!(ep.is_ok());

    // Request more context than the specific model can handle.
    let err = router.route_for_context(Some("bonsai-8b"), 16384);
    assert!(matches!(err, Err(RoutingError::ContextTooLong { .. })));
}

#[test]
fn router_models_list() {
    let router = make_test_router();
    let models = router.models_list();
    assert_eq!(models.len(), 2);
    assert!(models.iter().all(|m| m.object == "model"));
    assert!(models.iter().all(|m| m.owned_by == "oxibonsai"));
}

// ─── AdapterStack ────────────────────────────────────────────────────────────

#[test]
fn adapter_stack_add() {
    let stack = AdapterStack::new();
    assert!(stack.is_empty());
    assert_eq!(stack.len(), 0);

    let stack = stack.add("code-assist", 0.8).add("chat-tuned", 0.5);
    assert_eq!(stack.len(), 2);
    assert!(!stack.is_empty());
}

#[test]
fn adapter_stack_normalize_weights() {
    let mut stack = AdapterStack::new()
        .add("adapter-a", 0.6)
        .add("adapter-b", 0.4);

    stack.normalize_weights();
    let total = stack.total_weight();
    assert!((total - 1.0).abs() < f32::EPSILON);
}

#[test]
fn adapter_stack_total_weight() {
    let stack = AdapterStack::new()
        .add("a", 0.3)
        .add("b", 0.5)
        .add("c", 0.2);
    let total = stack.total_weight();
    assert!((total - 1.0).abs() < f32::EPSILON);
}
