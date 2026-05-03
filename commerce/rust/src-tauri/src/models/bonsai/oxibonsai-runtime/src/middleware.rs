//! Request middleware: context injection, logging, CORS, and idempotency caching.
//!
//! This module provides building blocks for production-grade HTTP middleware:
//!
//! - [`RequestContext`] — per-request metadata injected at the entry point
//! - [`RequestIdGen`] — atomic, monotonically increasing request ID generator
//! - [`RequestLogger`] — structured request/response logging with optional body capture
//! - [`CorsConfig`] — configurable CORS policy with header generation helpers
//! - [`IdempotencyCache`] — idempotency-key cache for safe request deduplication
//!
//! # Example
//!
//! ```
//! use oxibonsai_runtime::middleware::{RequestContext, RequestLogger, CorsConfig};
//!
//! let ctx = RequestContext::new("/v1/chat/completions", "POST", "10.0.0.1");
//! let logger = RequestLogger::new();
//! logger.log_request(&ctx);
//! logger.log_response(&ctx, 200, 512);
//!
//! let cors = CorsConfig::default();
//! assert!(cors.is_origin_allowed("*"));
//! ```

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use std::time::{Duration, Instant};

// ─── RequestContext ──────────────────────────────────────────────────────────

/// Per-request context injected by middleware at the entry point.
///
/// Carries metadata needed for logging, tracing, and metrics throughout
/// the request lifetime.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request identifier (e.g. `"oxibonsai-1714000000000-1"`).
    pub request_id: String,
    /// Caller identity — typically an IP address or API key prefix.
    pub client_id: String,
    /// Wall-clock instant when the request was received.
    pub started_at: Instant,
    /// Request path (e.g. `"/v1/chat/completions"`).
    pub path: String,
    /// HTTP method in upper-case (e.g. `"POST"`).
    pub method: String,
}

impl RequestContext {
    /// Create a new context with an auto-generated request ID.
    pub fn new(path: &str, method: &str, client_id: &str) -> Self {
        // Generate a lightweight ID without an external generator so the type
        // is self-contained; callers can supply a [`RequestIdGen`] for prod use.
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let request_id = format!("req-{ts_ms}");
        Self {
            request_id,
            client_id: client_id.to_owned(),
            started_at: Instant::now(),
            path: path.to_owned(),
            method: method.to_uppercase(),
        }
    }

    /// Create a context with an explicit request ID (used with [`RequestIdGen`]).
    pub fn with_id(request_id: String, path: &str, method: &str, client_id: &str) -> Self {
        Self {
            request_id,
            client_id: client_id.to_owned(),
            started_at: Instant::now(),
            path: path.to_owned(),
            method: method.to_uppercase(),
        }
    }

    /// Elapsed time since the request was received, in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    /// Elapsed time since the request was received as a [`Duration`].
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

// ─── RequestIdGen ────────────────────────────────────────────────────────────

/// Atomic, monotonically increasing request ID generator.
///
/// IDs have the form `"{prefix}-{timestamp_ms}-{counter}"`, e.g.
/// `"oxibonsai-1714000000000-42"`. The combination of a millisecond
/// timestamp and a per-process counter makes collisions practically
/// impossible across restarts.
pub struct RequestIdGen {
    counter: AtomicU64,
    prefix: String,
}

impl RequestIdGen {
    /// Create a new generator with the given prefix string.
    pub fn new(prefix: &str) -> Self {
        Self {
            counter: AtomicU64::new(0),
            prefix: prefix.to_owned(),
        }
    }

    /// Generate the next unique request ID.
    ///
    /// Format: `"{prefix}-{timestamp_ms}-{counter}"`
    pub fn next(&self) -> String {
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{}-{ts_ms}-{counter}", self.prefix)
    }
}

// ─── RequestLogger ───────────────────────────────────────────────────────────

/// Middleware that logs each request and its corresponding response.
///
/// Emits structured log lines via [`tracing`]. Body logging is opt-in and
/// truncated at `max_body_log_bytes` to avoid flooding logs with large payloads.
pub struct RequestLogger {
    /// Whether to include body content in log output.
    pub log_bodies: bool,
    /// Maximum number of body bytes to include in a log line.
    pub max_body_log_bytes: usize,
}

impl RequestLogger {
    /// Create a logger that does not log bodies.
    pub fn new() -> Self {
        Self {
            log_bodies: false,
            max_body_log_bytes: 0,
        }
    }

    /// Create a logger that includes up to `max_bytes` of body content.
    pub fn with_body_logging(max_bytes: usize) -> Self {
        Self {
            log_bodies: true,
            max_body_log_bytes: max_bytes,
        }
    }

    /// Log an incoming request.
    pub fn log_request(&self, ctx: &RequestContext) {
        let line = Self::format_request_line(ctx);
        tracing::info!(target: "oxibonsai::middleware", "{line}");
    }

    /// Log an outgoing response.
    pub fn log_response(&self, ctx: &RequestContext, status: u16, body_bytes: usize) {
        let elapsed_ms = ctx.elapsed_ms();
        let line = Self::format_response_line(ctx, status, elapsed_ms);
        if self.log_bodies && body_bytes > 0 {
            tracing::info!(
                target: "oxibonsai::middleware",
                "{line} body_bytes={body_bytes}"
            );
        } else {
            tracing::info!(target: "oxibonsai::middleware", "{line}");
        }
    }

    /// Format an incoming-request log line.
    ///
    /// Output: `"[{request_id}] {method} {path} from {client_id}"`
    pub fn format_request_line(ctx: &RequestContext) -> String {
        format!(
            "[{}] {} {} from {}",
            ctx.request_id, ctx.method, ctx.path, ctx.client_id
        )
    }

    /// Format an outgoing-response log line.
    ///
    /// Output: `"[{request_id}] {status} in {elapsed_ms}ms"`
    pub fn format_response_line(ctx: &RequestContext, status: u16, elapsed_ms: u64) -> String {
        format!("[{}] {} in {}ms", ctx.request_id, status, elapsed_ms)
    }
}

impl Default for RequestLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CorsConfig ──────────────────────────────────────────────────────────────

/// Cross-Origin Resource Sharing (CORS) policy configuration.
///
/// Used to generate `Access-Control-*` headers for preflight and main requests.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins. Use `["*"]` to permit all origins.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers.
    pub allowed_headers: Vec<String>,
    /// `Access-Control-Max-Age` in seconds (how long browsers may cache the preflight).
    pub max_age_secs: u64,
    /// Whether to allow credentials (cookies, auth headers).
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            max_age_secs: 3600,
            allow_credentials: false,
        }
    }
}

impl CorsConfig {
    /// Returns `true` if the given `origin` is permitted by this policy.
    ///
    /// An entry of `"*"` in `allowed_origins` permits all origins.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Generate `Access-Control-*` response headers as `(name, value)` pairs.
    ///
    /// Returns headers suitable for both preflight (`OPTIONS`) and actual responses.
    pub fn access_control_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::with_capacity(5);

        let origin_value = if self.allowed_origins.iter().any(|o| o == "*") {
            "*".to_owned()
        } else {
            self.allowed_origins.join(", ")
        };
        headers.push(("Access-Control-Allow-Origin".to_owned(), origin_value));

        headers.push((
            "Access-Control-Allow-Methods".to_owned(),
            self.allowed_methods.join(", "),
        ));

        headers.push((
            "Access-Control-Allow-Headers".to_owned(),
            self.allowed_headers.join(", "),
        ));

        headers.push((
            "Access-Control-Max-Age".to_owned(),
            self.max_age_secs.to_string(),
        ));

        if self.allow_credentials {
            headers.push((
                "Access-Control-Allow-Credentials".to_owned(),
                "true".to_owned(),
            ));
        }

        headers
    }
}

// ─── IdempotencyCache ────────────────────────────────────────────────────────

/// Cached entry for a previously processed idempotent request.
struct CachedResponse {
    status: u16,
    body: Vec<u8>,
    created_at: Instant,
}

/// Request deduplication cache keyed on client-supplied idempotency keys.
///
/// When a client sends the same idempotency key twice, the second request
/// receives the cached response without re-executing the operation.
/// Entries expire after `ttl` and are lazily evicted.
pub struct IdempotencyCache {
    cache: Mutex<HashMap<String, CachedResponse>>,
    max_entries: usize,
    ttl: Duration,
}

impl IdempotencyCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_entries,
            ttl,
        }
    }

    /// Look up a previously cached response by idempotency key.
    ///
    /// Returns `(status_code, body)` if a fresh entry exists; `None` otherwise.
    pub fn get(&self, key: &str) -> Option<(u16, Vec<u8>)> {
        let cache = self.cache.lock().expect("idempotency cache mutex poisoned");
        if let Some(entry) = cache.get(key) {
            if entry.created_at.elapsed() < self.ttl {
                return Some((entry.status, entry.body.clone()));
            }
        }
        None
    }

    /// Store a response under the given idempotency key.
    ///
    /// If the cache is full, expired entries are evicted first. If still
    /// full after eviction, the insert is silently dropped to prevent
    /// unbounded memory growth.
    pub fn insert(&self, key: &str, status: u16, body: Vec<u8>) {
        let mut cache = self.cache.lock().expect("idempotency cache mutex poisoned");

        // Evict expired entries when approaching capacity.
        if cache.len() >= self.max_entries {
            let ttl = self.ttl;
            cache.retain(|_, v| v.created_at.elapsed() < ttl);
        }

        // After eviction, only insert if we still have room.
        if cache.len() < self.max_entries {
            cache.insert(
                key.to_owned(),
                CachedResponse {
                    status,
                    body,
                    created_at: Instant::now(),
                },
            );
        }
    }

    /// Remove all expired entries from the cache.
    pub fn evict_expired(&self) {
        let ttl = self.ttl;
        let mut cache = self.cache.lock().expect("idempotency cache mutex poisoned");
        cache.retain(|_, v| v.created_at.elapsed() < ttl);
    }

    /// Return the number of entries currently in the cache (including stale ones).
    pub fn len(&self) -> usize {
        self.cache
            .lock()
            .expect("idempotency cache mutex poisoned")
            .len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_request_context_elapsed() {
        let ctx = RequestContext::new("/health", "GET", "10.0.0.1");
        // Elapsed should be very small immediately after creation.
        assert!(
            ctx.elapsed_ms() < 500,
            "elapsed should be <500ms at creation"
        );
        assert!(ctx.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn test_request_id_gen_unique() {
        let gen = RequestIdGen::new("test");
        let ids: Vec<String> = (0..100).map(|_| gen.next()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "all generated IDs must be unique");
    }

    #[test]
    fn test_request_id_gen_prefix() {
        let gen = RequestIdGen::new("oxibonsai");
        let id = gen.next();
        assert!(
            id.starts_with("oxibonsai-"),
            "ID should start with prefix; got {id}"
        );
    }

    #[test]
    fn test_request_logger_format_request_line() {
        let mut ctx = RequestContext::new("/v1/chat/completions", "post", "1.2.3.4");
        ctx.request_id = "req-42".to_owned();
        let line = RequestLogger::format_request_line(&ctx);
        assert_eq!(line, "[req-42] POST /v1/chat/completions from 1.2.3.4");
    }

    #[test]
    fn test_request_logger_format_response_line() {
        let mut ctx = RequestContext::new("/health", "GET", "127.0.0.1");
        ctx.request_id = "req-99".to_owned();
        let line = RequestLogger::format_response_line(&ctx, 200, 15);
        assert_eq!(line, "[req-99] 200 in 15ms");
    }

    #[test]
    fn test_cors_config_default_allows_all() {
        let cors = CorsConfig::default();
        assert!(cors.is_origin_allowed("https://example.com"));
        assert!(cors.is_origin_allowed("null"));
        assert!(cors.is_origin_allowed("*"));
    }

    #[test]
    fn test_cors_config_specific_origin() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://app.example.com".to_string()],
            ..Default::default()
        };
        assert!(cors.is_origin_allowed("https://app.example.com"));
        assert!(!cors.is_origin_allowed("https://evil.example.com"));
    }

    #[test]
    fn test_cors_access_control_headers() {
        let cors = CorsConfig::default();
        let headers = cors.access_control_headers();

        // Should contain Access-Control-Allow-Origin
        let has_origin = headers
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "*");
        assert!(has_origin, "should have wildcard Allow-Origin header");

        // Should contain methods
        let has_methods = headers
            .iter()
            .any(|(k, _)| k == "Access-Control-Allow-Methods");
        assert!(has_methods);

        // allow_credentials is false by default, so no credentials header
        let has_creds = headers
            .iter()
            .any(|(k, _)| k == "Access-Control-Allow-Credentials");
        assert!(
            !has_creds,
            "should not include credentials header by default"
        );
    }

    #[test]
    fn test_idempotency_cache_insert_and_get() {
        let cache = IdempotencyCache::new(100, Duration::from_secs(60));
        cache.insert("key-1", 200, b"hello".to_vec());
        let result = cache.get("key-1");
        assert_eq!(result, Some((200, b"hello".to_vec())));
    }

    #[test]
    fn test_idempotency_cache_miss() {
        let cache = IdempotencyCache::new(100, Duration::from_secs(60));
        assert!(cache.get("nonexistent-key").is_none());
    }

    #[test]
    fn test_idempotency_cache_evicts_expired() {
        // TTL of 10ms so entries expire quickly in tests.
        let cache = IdempotencyCache::new(100, Duration::from_millis(10));
        cache.insert("exp-key", 200, vec![]);
        assert_eq!(cache.len(), 1);

        thread::sleep(Duration::from_millis(20));
        cache.evict_expired();
        assert_eq!(cache.len(), 0, "expired entry should have been evicted");
    }

    #[test]
    fn test_idempotency_cache_expired_returns_none() {
        let cache = IdempotencyCache::new(100, Duration::from_millis(10));
        cache.insert("ttl-key", 201, b"data".to_vec());
        thread::sleep(Duration::from_millis(20));
        // get() should not return stale entries.
        assert!(
            cache.get("ttl-key").is_none(),
            "stale cache entry must not be returned"
        );
    }
}
