//! Integration tests for the token budget module.

use oxibonsai_runtime::token_budget::{
    BudgetConfig, BudgetError, BudgetPolicy, GlobalTokenBudget, RequestBudget, TokenCostEstimate,
};

// ─── BudgetConfig ─────────────────────────────────────────────────────────────

#[test]
fn budget_config_unlimited() {
    let cfg = BudgetConfig::unlimited();
    assert!(cfg.max_prompt_tokens.is_none());
    assert!(cfg.max_completion_tokens.is_none());
    assert!(cfg.max_total_tokens.is_none());
}

#[test]
fn budget_config_with_max_completion() {
    let cfg = BudgetConfig::new().with_max_completion(256);
    assert_eq!(cfg.max_completion_tokens, Some(256));
}

// ─── RequestBudget: construction ─────────────────────────────────────────────

#[test]
fn request_budget_new_within_limit() {
    let cfg = BudgetConfig::new()
        .with_max_completion(100)
        .with_max_total(200);
    let budget = RequestBudget::new(cfg, 50);
    assert!(budget.is_ok());
    let b = budget.expect("should be ok");
    assert_eq!(b.prompt_tokens(), 50);
    assert_eq!(b.completion_tokens(), 0);
}

#[test]
fn request_budget_prompt_too_long() {
    let cfg = BudgetConfig::new()
        .with_max_total(100)
        // max_prompt_tokens set via a direct field assignment is tested below via
        // a custom config.  We use the field directly.
        .with_policy(BudgetPolicy::ReturnError);
    // Build a config with max_prompt_tokens = 10.
    let mut cfg2 = cfg;
    cfg2.max_prompt_tokens = Some(10);
    let result = RequestBudget::new(cfg2, 20);
    assert!(
        matches!(
            result,
            Err(BudgetError::PromptTooLong {
                prompt: 20,
                max: 10
            })
        ),
        "expected PromptTooLong, got: {result:?}"
    );
}

// ─── RequestBudget: record_token ──────────────────────────────────────────────

#[test]
fn request_budget_record_token() {
    let cfg = BudgetConfig::new().with_max_completion(10);
    let mut budget = RequestBudget::new(cfg, 5).expect("new");
    budget.record_token().expect("record");
    assert_eq!(budget.completion_tokens(), 1);
    assert_eq!(budget.total_tokens(), 6);
}

#[test]
fn request_budget_completion_exhausted() {
    let cfg = BudgetConfig::new().with_max_completion(3);
    let mut budget = RequestBudget::new(cfg, 0).expect("new");
    budget.record_token().expect("1");
    budget.record_token().expect("2");
    budget.record_token().expect("3");
    let result = budget.record_token();
    assert!(
        matches!(
            result,
            Err(BudgetError::CompletionBudgetExhausted { limit: 3 })
        ),
        "expected CompletionBudgetExhausted, got: {result:?}"
    );
}

#[test]
fn request_budget_total_exhausted() {
    // prompt=5, max_total=7 → 3 completion tokens allowed
    let cfg = BudgetConfig::new().with_max_total(7);
    let mut budget = RequestBudget::new(cfg, 5).expect("new");
    budget.record_token().expect("1");
    budget.record_token().expect("2");
    let result = budget.record_token();
    assert!(
        matches!(
            result,
            Err(BudgetError::TotalBudgetExhausted { limit: 7, used: 8 })
        ),
        "expected TotalBudgetExhausted, got: {result:?}"
    );
}

// ─── RequestBudget: remaining / exhausted ─────────────────────────────────────

#[test]
fn request_budget_remaining_completion() {
    let cfg = BudgetConfig::new().with_max_completion(10);
    let mut budget = RequestBudget::new(cfg, 0).expect("new");
    budget.record_tokens(3).expect("record");
    assert_eq!(budget.remaining_completion_tokens(), Some(7));
}

#[test]
fn request_budget_is_exhausted() {
    let cfg = BudgetConfig::new().with_max_completion(2);
    let mut budget = RequestBudget::new(cfg, 0).expect("new");
    assert!(!budget.is_exhausted());
    budget.record_tokens(2).expect("record");
    assert!(budget.is_exhausted());
}

// ─── GlobalTokenBudget ───────────────────────────────────────────────────────

#[test]
fn global_budget_record() {
    let gb = GlobalTokenBudget::new(Some(1000));
    gb.record(100);
    gb.record(200);
    assert_eq!(gb.total_used(), 300);
}

#[test]
fn global_budget_unlimited_no_exhaustion() {
    let gb = GlobalTokenBudget::unlimited();
    gb.record(u64::MAX / 2);
    assert!(!gb.is_exhausted());
    assert!(gb.remaining().is_none());
}

#[test]
fn global_budget_limited_exhaustion() {
    let gb = GlobalTokenBudget::new(Some(500));
    assert!(!gb.is_exhausted());
    gb.record(500);
    assert!(gb.is_exhausted());
    gb.record(100); // goes over but atomic keeps counting
    assert!(gb.is_exhausted());
}

#[test]
fn global_budget_utilization() {
    let gb = GlobalTokenBudget::new(Some(200));
    gb.record(50);
    let util = gb.utilization().expect("should have utilization");
    assert!((util - 0.25).abs() < 1e-5, "utilization={util}");
}

// ─── TokenCostEstimate ────────────────────────────────────────────────────────

#[test]
fn cost_estimate_basic() {
    // 1000 prompt tokens @ $0.01/1k, 500 completion tokens @ $0.03/1k
    let est = TokenCostEstimate::compute(1000, 500, 0.01, 0.03);
    assert!((est.prompt_cost - 0.01).abs() < 1e-9, "{}", est.prompt_cost);
    assert!(
        (est.completion_cost - 0.015).abs() < 1e-9,
        "{}",
        est.completion_cost
    );
    assert!((est.total_cost - 0.025).abs() < 1e-9, "{}", est.total_cost);
    assert_eq!(est.prompt_tokens, 1000);
    assert_eq!(est.completion_tokens, 500);
}

#[test]
fn cost_estimate_zero_tokens() {
    let est = TokenCostEstimate::compute(0, 0, 0.01, 0.03);
    assert!((est.prompt_cost - 0.0).abs() < 1e-12);
    assert!((est.completion_cost - 0.0).abs() < 1e-12);
    assert!((est.total_cost - 0.0).abs() < 1e-12);
}
