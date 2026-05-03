//! Circuit breaker pattern for production resilience.
//!
//! Prevents cascading failures by short-circuiting when errors exceed a
//! configurable threshold. Supports closed, open, and half-open states
//! with automatic recovery testing.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// State of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation: requests flow through.
    Closed = 0,
    /// Failing: requests are rejected immediately.
    Open = 1,
    /// Testing recovery: a limited number of requests are allowed through.
    HalfOpen = 2,
}

impl CircuitState {
    fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Closed,
            1 => Self::Open,
            2 => Self::HalfOpen,
            _ => Self::Closed,
        }
    }
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "closed"),
            Self::Open => write!(f, "open"),
            Self::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: u64,
    /// How long to stay open before transitioning to half-open.
    pub recovery_timeout: Duration,
    /// Maximum test requests allowed in half-open state.
    pub half_open_max_requests: u64,
    /// Number of successes required to close from half-open.
    pub success_threshold: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            half_open_max_requests: 3,
            success_threshold: 2,
        }
    }
}

/// Circuit breaker for protecting against cascading failures.
///
/// Thread-safe: uses atomics for counters and a mutex for the timestamp.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
///
/// let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
/// assert_eq!(cb.state(), CircuitState::Closed);
///
/// // Execute a fallible operation with circuit-breaker protection
/// let result: Result<i32, _> = cb.execute(|| Ok::<i32, String>(42));
/// assert_eq!(result.expect("should succeed"), 42);
/// ```
pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    half_open_requests: AtomicU64,
    last_failure_time: Mutex<Option<Instant>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            half_open_requests: AtomicU64::new(0),
            last_failure_time: Mutex::new(None),
            config,
        }
    }

    /// Check if a request should be allowed through.
    ///
    /// Returns `true` if the circuit is closed or in half-open testing mode.
    /// Returns `false` if the circuit is open (failing).
    pub fn allow_request(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                let should_transition = {
                    let guard = self
                        .last_failure_time
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    match *guard {
                        Some(last_failure) => {
                            last_failure.elapsed() >= self.config.recovery_timeout
                        }
                        None => true,
                    }
                };

                if should_transition {
                    // Transition to half-open
                    self.state
                        .store(CircuitState::HalfOpen as u8, Ordering::Release);
                    self.half_open_requests.store(0, Ordering::Release);
                    self.success_count.store(0, Ordering::Release);
                    tracing::info!("circuit breaker transitioning to half-open");
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests for testing
                let current = self.half_open_requests.fetch_add(1, Ordering::AcqRel);
                current < self.config.half_open_max_requests
            }
        }
    }

    /// Record a successful operation.
    ///
    /// In half-open state, enough successes will close the circuit.
    pub fn record_success(&self) {
        match self.state() {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                if count >= self.config.success_threshold {
                    self.state
                        .store(CircuitState::Closed as u8, Ordering::Release);
                    self.failure_count.store(0, Ordering::Release);
                    self.success_count.store(0, Ordering::Release);
                    tracing::info!("circuit breaker closed (recovered)");
                }
            }
            CircuitState::Open => {
                // Ignore successes in open state (shouldn't happen normally)
            }
        }
    }

    /// Record a failed operation.
    ///
    /// If the failure count exceeds the threshold, the circuit opens.
    pub fn record_failure(&self) {
        match self.state() {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
                if count >= self.config.failure_threshold {
                    self.state
                        .store(CircuitState::Open as u8, Ordering::Release);
                    {
                        let mut guard = self
                            .last_failure_time
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *guard = Some(Instant::now());
                    }
                    tracing::warn!(
                        failures = count,
                        "circuit breaker opened after {} failures",
                        count
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open immediately re-opens
                self.state
                    .store(CircuitState::Open as u8, Ordering::Release);
                {
                    let mut guard = self
                        .last_failure_time
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    *guard = Some(Instant::now());
                }
                self.half_open_requests.store(0, Ordering::Release);
                self.success_count.store(0, Ordering::Release);
                tracing::warn!("circuit breaker re-opened from half-open state");
            }
            CircuitState::Open => {
                // Update failure time to extend the open period
                let mut guard = self
                    .last_failure_time
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *guard = Some(Instant::now());
            }
        }
    }

    /// Get the current circuit state.
    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.half_open_requests.store(0, Ordering::Release);
        {
            let mut guard = self
                .last_failure_time
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *guard = None;
        }
        tracing::info!("circuit breaker reset to closed");
    }

    /// Execute a closure with circuit breaker protection.
    ///
    /// If the circuit is open, returns `CircuitBreakerError::CircuitOpen`
    /// without calling the closure. Otherwise, executes the closure and
    /// records success or failure.
    pub fn execute<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if !self.allow_request() {
            return Err(CircuitBreakerError::CircuitOpen);
        }

        match f() {
            Ok(val) => {
                self.record_success();
                Ok(val)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitBreakerError::Inner(e))
            }
        }
    }

    /// Get the current failure count.
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Get the configuration.
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }
}

/// Error type for circuit breaker-protected operations.
#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    /// The circuit is open; the operation was not attempted.
    CircuitOpen,
    /// The inner operation failed.
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CircuitOpen => write!(f, "circuit breaker is open"),
            Self::Inner(e) => write!(f, "inner error: {}", e),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for CircuitBreakerError<E> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(50),
            half_open_max_requests: 2,
            success_threshold: 2,
        }
    }

    #[test]
    fn initial_state_is_closed() {
        let cb = CircuitBreaker::new(test_config());
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreaker::new(test_config());
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn success_resets_failure_count() {
        let cb = CircuitBreaker::new(test_config());
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn transitions_to_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(10),
            half_open_max_requests: 2,
            success_threshold: 1,
        };
        let cb = CircuitBreaker::new(config);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(20));

        // allow_request should transition to half-open
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn half_open_limits_requests() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(1),
            half_open_max_requests: 2,
            success_threshold: 2,
        };
        let cb = CircuitBreaker::new(config);
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));

        // First request transitions from open to half-open (not counted)
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        // Next two requests allowed (half_open_max_requests = 2)
        assert!(cb.allow_request());
        assert!(cb.allow_request());
        // Now the limit is reached
        assert!(!cb.allow_request());
    }

    #[test]
    fn half_open_closes_on_success() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(1),
            half_open_max_requests: 3,
            success_threshold: 2,
        };
        let cb = CircuitBreaker::new(config);
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));

        assert!(cb.allow_request()); // -> half-open
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_reopens_on_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(1),
            half_open_max_requests: 3,
            success_threshold: 2,
        };
        let cb = CircuitBreaker::new(config);
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));

        assert!(cb.allow_request()); // -> half-open
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn reset_clears_state() {
        let cb = CircuitBreaker::new(test_config());
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(cb.allow_request());
    }

    #[test]
    fn execute_success() {
        let cb = CircuitBreaker::new(test_config());
        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Ok(42));
        assert!(result.is_ok());
        assert_eq!(result.expect("should be ok"), 42);
    }

    #[test]
    fn execute_failure() {
        let cb = CircuitBreaker::new(test_config());
        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Err("oops"));
        assert!(result.is_err());
        match result {
            Err(CircuitBreakerError::Inner(msg)) => assert_eq!(msg, "oops"),
            _ => panic!("expected Inner error"),
        }
    }

    #[test]
    fn execute_circuit_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Ok(42));
        assert!(result.is_err());
        match result {
            Err(CircuitBreakerError::CircuitOpen) => {}
            _ => panic!("expected CircuitOpen error"),
        }
    }

    #[test]
    fn default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.recovery_timeout, Duration::from_secs(30));
        assert_eq!(config.half_open_max_requests, 3);
        assert_eq!(config.success_threshold, 2);
    }

    #[test]
    fn circuit_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "closed");
        assert_eq!(format!("{}", CircuitState::Open), "open");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "half-open");
    }

    #[test]
    fn circuit_breaker_error_display() {
        let err: CircuitBreakerError<String> = CircuitBreakerError::CircuitOpen;
        assert_eq!(format!("{}", err), "circuit breaker is open");

        let err: CircuitBreakerError<String> = CircuitBreakerError::Inner("test error".to_string());
        assert_eq!(format!("{}", err), "inner error: test error");
    }

    #[test]
    fn config_accessor() {
        let config = test_config();
        let cb = CircuitBreaker::new(config.clone());
        assert_eq!(cb.config().failure_threshold, 3);
    }
}
