//! Circuit breaker integration tests.
//!
//! Tests state transitions, execute wrapper, reset, and custom thresholds.

use std::time::Duration;

use oxibonsai_runtime::circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
};

// ── Helper ───────────────────────────────────────────────────────────────

fn fast_config(failure_threshold: u64) -> CircuitBreakerConfig {
    CircuitBreakerConfig {
        failure_threshold,
        recovery_timeout: Duration::from_millis(10),
        half_open_max_requests: 2,
        success_threshold: 2,
    }
}

// ── 1. Starts in Closed state ────────────────────────────────────────────

#[test]
fn starts_closed() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.allow_request());
    assert_eq!(cb.failure_count(), 0);
}

// ── 2. N failures → transitions to Open ──────────────────────────────────

#[test]
fn failures_transition_to_open() {
    let cb = CircuitBreaker::new(fast_config(3));

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 1);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 2);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert_eq!(cb.failure_count(), 3);
}

// ── 3. Open state rejects requests ───────────────────────────────────────

#[test]
fn open_rejects_requests() {
    let config = CircuitBreakerConfig {
        failure_threshold: 1,
        recovery_timeout: Duration::from_secs(60), // long timeout
        half_open_max_requests: 1,
        success_threshold: 1,
    };
    let cb = CircuitBreaker::new(config);
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(!cb.allow_request(), "open circuit should reject requests");
}

// ── 4. After recovery timeout → transitions to HalfOpen ──────────────────

#[test]
fn transitions_to_half_open_after_timeout() {
    let cb = CircuitBreaker::new(fast_config(1));
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);

    std::thread::sleep(Duration::from_millis(20));

    assert!(
        cb.allow_request(),
        "should allow request after recovery timeout"
    );
    assert_eq!(cb.state(), CircuitState::HalfOpen);
}

// ── 5. Success in HalfOpen → transitions to Closed ───────────────────────

#[test]
fn half_open_closes_on_sufficient_success() {
    let cb = CircuitBreaker::new(fast_config(1));
    cb.record_failure();
    std::thread::sleep(Duration::from_millis(20));

    // Transition to half-open
    assert!(cb.allow_request());
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // Need success_threshold=2 successes to close
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 0);
}

// ── 6. Failure in HalfOpen → back to Open ────────────────────────────────

#[test]
fn half_open_reopens_on_failure() {
    let cb = CircuitBreaker::new(fast_config(1));
    cb.record_failure();
    std::thread::sleep(Duration::from_millis(20));

    assert!(cb.allow_request()); // -> half-open
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

// ── 7. Reset clears all state ────────────────────────────────────────────

#[test]
fn reset_clears_all_state() {
    let cb = CircuitBreaker::new(fast_config(1));
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(cb.failure_count() > 0);

    cb.reset();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 0);
    assert!(cb.allow_request());
}

#[test]
fn reset_from_half_open() {
    let cb = CircuitBreaker::new(fast_config(1));
    cb.record_failure();
    std::thread::sleep(Duration::from_millis(20));
    cb.allow_request(); // -> half-open

    cb.reset();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 0);
}

// ── 8. Execute wrapper passes through success ────────────────────────────

#[test]
fn execute_success_passes_through() {
    let cb = CircuitBreaker::new(fast_config(3));
    let result: Result<i32, CircuitBreakerError<String>> = cb.execute(|| Ok(42));
    assert!(result.is_ok());
    assert_eq!(result.expect("should be ok"), 42);
    assert_eq!(cb.failure_count(), 0);
}

#[test]
fn execute_multiple_successes() {
    let cb = CircuitBreaker::new(fast_config(3));
    for i in 0..10 {
        let result: Result<i32, CircuitBreakerError<String>> = cb.execute(|| Ok(i));
        assert!(result.is_ok());
    }
    assert_eq!(cb.state(), CircuitState::Closed);
    assert_eq!(cb.failure_count(), 0);
}

// ── 9. Execute wrapper records failures ──────────────────────────────────

#[test]
fn execute_failure_records() {
    let cb = CircuitBreaker::new(fast_config(3));
    let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Err("oops"));
    assert!(result.is_err());
    assert_eq!(cb.failure_count(), 1);

    match result {
        Err(CircuitBreakerError::Inner(msg)) => assert_eq!(msg, "oops"),
        _ => panic!("expected Inner error"),
    }
}

#[test]
fn execute_opens_after_threshold_failures() {
    let cb = CircuitBreaker::new(fast_config(2));

    let _: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Err("fail1"));
    assert_eq!(cb.state(), CircuitState::Closed);

    let _: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Err("fail2"));
    assert_eq!(cb.state(), CircuitState::Open);

    // Next execute should be rejected
    let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Ok(42));
    match result {
        Err(CircuitBreakerError::CircuitOpen) => {}
        _ => panic!("expected CircuitOpen"),
    }
}

// ── 10. Custom config thresholds respected ───────────────────────────────

#[test]
fn custom_high_failure_threshold() {
    let config = CircuitBreakerConfig {
        failure_threshold: 10,
        recovery_timeout: Duration::from_secs(30),
        half_open_max_requests: 5,
        success_threshold: 3,
    };
    let cb = CircuitBreaker::new(config);

    // 9 failures should not open
    for _ in 0..9 {
        cb.record_failure();
    }
    assert_eq!(cb.state(), CircuitState::Closed);

    // 10th failure opens
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn custom_success_threshold() {
    let config = CircuitBreakerConfig {
        failure_threshold: 1,
        recovery_timeout: Duration::from_millis(5),
        half_open_max_requests: 10,
        success_threshold: 5,
    };
    let cb = CircuitBreaker::new(config);
    cb.record_failure();
    std::thread::sleep(Duration::from_millis(10));
    cb.allow_request(); // -> half-open

    // Need 5 successes to close
    for i in 0..4 {
        cb.record_success();
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen,
            "should still be half-open after {} successes",
            i + 1
        );
    }
    cb.record_success(); // 5th success
    assert_eq!(cb.state(), CircuitState::Closed);
}

// ── Circuit breaker error display ────────────────────────────────────────

#[test]
fn circuit_breaker_error_display() {
    let open_err: CircuitBreakerError<String> = CircuitBreakerError::CircuitOpen;
    assert_eq!(format!("{}", open_err), "circuit breaker is open");

    let inner_err: CircuitBreakerError<String> =
        CircuitBreakerError::Inner("connection refused".to_string());
    assert_eq!(format!("{}", inner_err), "inner error: connection refused");
}

// ── Circuit state display ────────────────────────────────────────────────

#[test]
fn circuit_state_display_all() {
    assert_eq!(format!("{}", CircuitState::Closed), "closed");
    assert_eq!(format!("{}", CircuitState::Open), "open");
    assert_eq!(format!("{}", CircuitState::HalfOpen), "half-open");
}

// ── Config accessor ──────────────────────────────────────────────────────

#[test]
fn config_accessor_returns_correct_values() {
    let config = CircuitBreakerConfig {
        failure_threshold: 7,
        recovery_timeout: Duration::from_secs(45),
        half_open_max_requests: 4,
        success_threshold: 3,
    };
    let cb = CircuitBreaker::new(config);
    assert_eq!(cb.config().failure_threshold, 7);
    assert_eq!(cb.config().recovery_timeout, Duration::from_secs(45));
    assert_eq!(cb.config().half_open_max_requests, 4);
    assert_eq!(cb.config().success_threshold, 3);
}

// ── Success resets failure count ─────────────────────────────────────────

#[test]
fn success_resets_failure_count() {
    let cb = CircuitBreaker::new(fast_config(5));
    cb.record_failure();
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.failure_count(), 3);

    cb.record_success();
    assert_eq!(cb.failure_count(), 0);
    assert_eq!(cb.state(), CircuitState::Closed);
}
