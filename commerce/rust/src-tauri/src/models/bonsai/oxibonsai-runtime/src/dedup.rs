//! Request deduplication: cache identical requests to avoid redundant inference.
//!
//! Uses content hashing for exact deduplication and optional
//! fuzzy matching for near-duplicate detection.
//!
//! # Overview
//!
//! The [`DedupCache`] computes an FNV-1a hash over the serialized request content
//! (or a concatenation of role+content pairs for message lists) and uses that as
//! the cache key.  On a hit the cached response is returned immediately, bypassing
//! the inference pipeline entirely.  Entries carry a TTL and are lazily evicted on
//! access as well as via the explicit [`DedupCache::evict_expired`] method.
//!
//! # Example
//!
//! ```rust
//! use std::time::Duration;
//! use oxibonsai_runtime::dedup::{DedupCache, RequestKey};
//!
//! let mut cache = DedupCache::with_capacity(128);
//! let key = RequestKey::from_str("What is Rust?");
//! cache.insert(key.clone(), "Rust is a systems language.".to_string());
//!
//! assert_eq!(cache.get(&key), Some("Rust is a systems language."));
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a helpers (inline, no external crate)
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 1099511628211;

/// Compute an FNV-1a 64-bit hash over arbitrary bytes.
#[inline]
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// RequestKey
// ─────────────────────────────────────────────────────────────────────────────

/// A hashed request key (FNV-1a of serialized request content).
///
/// Two [`RequestKey`] values are equal iff the underlying hash values are equal.
/// Collisions are theoretically possible but extremely rare for the prompt sizes
/// typical in LLM serving.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestKey(pub u64);

impl RequestKey {
    /// Compute the key from a serialized request string.
    ///
    /// # Example
    /// ```
    /// use oxibonsai_runtime::dedup::RequestKey;
    /// let k1 = RequestKey::from_str("hello world");
    /// let k2 = RequestKey::from_str("hello world");
    /// assert_eq!(k1, k2);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self(fnv1a_hash(s.as_bytes()))
    }

    /// Compute from a message list: FNV-1a over the concatenation of
    /// `role + "\x00" + content + "\x01"` for every message pair, in order.
    ///
    /// The sentinel bytes (`\x00` and `\x01`) prevent role/content boundary
    /// collisions (e.g. `("ab", "c")` vs `("a", "bc")`).
    ///
    /// # Example
    /// ```
    /// use oxibonsai_runtime::dedup::RequestKey;
    /// let msgs = [("user", "What is Rust?")];
    /// let k = RequestKey::from_messages(&msgs);
    /// assert_eq!(k, RequestKey::from_messages(&msgs));
    /// ```
    pub fn from_messages(messages: &[(&str, &str)]) -> Self {
        let mut hash = FNV_OFFSET_BASIS;
        for (role, content) in messages {
            for &byte in role.as_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            // Separator between role and content
            hash ^= 0x00;
            hash = hash.wrapping_mul(FNV_PRIME);
            for &byte in content.as_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            // Separator between messages
            hash ^= 0x01;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        Self(hash)
    }

    /// Return the raw 64-bit hash value.
    pub fn value(&self) -> u64 {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CachedResponse
// ─────────────────────────────────────────────────────────────────────────────

/// A cached response entry stored in the [`DedupCache`].
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The cached response text.
    pub content: String,
    /// When this entry was first created.
    pub created_at: Instant,
    /// How many times this entry has been returned as a cache hit.
    pub hit_count: u64,
    /// How long this entry is considered valid.
    pub ttl: Duration,
}

impl CachedResponse {
    /// Create a new entry with `hit_count = 0`.
    pub fn new(content: String, ttl: Duration) -> Self {
        Self {
            content,
            created_at: Instant::now(),
            hit_count: 0,
            ttl,
        }
    }

    /// Returns `true` if this entry's TTL has been exceeded.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    /// Increment the hit counter by one.
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the deduplication cache.
///
/// All counters are monotonically increasing; they are never reset unless
/// [`DedupCache::clear`] is called.
#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    /// Total number of [`DedupCache::get`] calls (hits + misses).
    pub total_requests: u64,
    /// Number of lookups that returned a cached response.
    pub cache_hits: u64,
    /// Number of lookups that found no valid entry.
    pub cache_misses: u64,
    /// Number of entries evicted due to capacity overflow or TTL expiry.
    pub evictions: u64,
}

impl DedupStats {
    /// Fraction of lookups that were cache hits, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no requests have been made yet.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }

    /// Human-readable one-line summary of the statistics.
    pub fn summary(&self) -> String {
        format!(
            "requests={} hits={} misses={} evictions={} hit_rate={:.1}%",
            self.total_requests,
            self.cache_hits,
            self.cache_misses,
            self.evictions,
            self.hit_rate() * 100.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupCache
// ─────────────────────────────────────────────────────────────────────────────

/// Request deduplication cache.
///
/// Stores responses keyed by [`RequestKey`].  When the cache is full, the
/// oldest entry (by insertion order, tracked via a monotonic sequence counter)
/// is evicted to make room for the new one.
///
/// # Thread safety
///
/// `DedupCache` is **not** `Sync` by design; wrap it in a `Mutex` or
/// `RwLock` when sharing across threads.
pub struct DedupCache {
    /// The backing store: key → (entry, insertion_seq).
    cache: HashMap<RequestKey, (CachedResponse, u64)>,
    /// Maximum number of entries before eviction kicks in.
    capacity: usize,
    /// Default TTL used by [`DedupCache::insert`].
    default_ttl: Duration,
    /// Aggregate statistics.
    stats: DedupStats,
    /// Monotonically increasing insertion counter.
    next_seq: u64,
}

impl DedupCache {
    /// Create a new cache with the given `capacity` and `default_ttl`.
    pub fn new(capacity: usize, default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
            default_ttl,
            stats: DedupStats::default(),
            next_seq: 0,
        }
    }

    /// Shorthand constructor with a 60-second default TTL.
    pub fn with_capacity(n: usize) -> Self {
        Self::new(n, Duration::from_secs(60))
    }

    /// Look up a cached response.
    ///
    /// Returns `None` if the key is not present or if the cached entry has
    /// expired.  On a valid hit the `hit_count` of the entry is incremented
    /// and the response string slice is returned.
    pub fn get(&mut self, key: &RequestKey) -> Option<&str> {
        self.stats.total_requests += 1;

        // Check for expiry first without borrowing mutably into the hit path.
        let expired = self
            .cache
            .get(key)
            .map(|(entry, _)| entry.is_expired())
            .unwrap_or(false);

        if expired {
            self.cache.remove(key);
            self.stats.cache_misses += 1;
            self.stats.evictions += 1;
            return None;
        }

        match self.cache.get_mut(key) {
            Some((entry, _seq)) => {
                entry.record_hit();
                self.stats.cache_hits += 1;
                // Return a reference to the content inside the map.
                // SAFETY: The entry lives as long as self.
                Some(self.cache[key].0.content.as_str())
            }
            None => {
                self.stats.cache_misses += 1;
                None
            }
        }
    }

    /// Insert a response with the `DedupCache::default_ttl`.
    ///
    /// If the cache is at capacity, the entry with the smallest insertion
    /// sequence number (i.e. the oldest entry) is evicted first.
    pub fn insert(&mut self, key: RequestKey, response: String) {
        let ttl = self.default_ttl;
        self.insert_with_ttl(key, response, ttl);
    }

    /// Insert a response with a custom `ttl`.
    ///
    /// Evicts the oldest entry when the cache is at capacity before inserting.
    pub fn insert_with_ttl(&mut self, key: RequestKey, response: String, ttl: Duration) {
        // If key already exists, just update it in-place.
        if self.cache.contains_key(&key) {
            let seq = self.next_seq;
            self.next_seq += 1;
            let entry = CachedResponse::new(response, ttl);
            self.cache.insert(key, (entry, seq));
            return;
        }

        // Evict the oldest entry when at capacity.
        if self.cache.len() >= self.capacity {
            self.evict_oldest();
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = CachedResponse::new(response, ttl);
        self.cache.insert(key, (entry, seq));
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&mut self) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, (entry, _)| !entry.is_expired());
        let removed = before - self.cache.len();
        self.stats.evictions += removed as u64;
        removed
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Reference to the current statistics snapshot.
    pub fn stats(&self) -> &DedupStats {
        &self.stats
    }

    /// Remove all entries and reset statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = DedupStats::default();
        self.next_seq = 0;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Evict the entry with the smallest insertion sequence number.
    fn evict_oldest(&mut self) {
        // Find the key with the minimum sequence number.
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|(_, (_, seq))| *seq)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (unit, inline)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_key_deterministic() {
        let k1 = RequestKey::from_str("hello");
        let k2 = RequestKey::from_str("hello");
        assert_eq!(k1, k2);
    }

    #[test]
    fn request_key_different_inputs() {
        let k1 = RequestKey::from_str("foo");
        let k2 = RequestKey::from_str("bar");
        assert_ne!(k1, k2);
    }

    #[test]
    fn cached_response_not_expired_immediately() {
        let r = CachedResponse::new("hi".to_string(), Duration::from_secs(60));
        assert!(!r.is_expired());
    }

    #[test]
    fn dedup_cache_basic_insert_get() {
        let mut cache = DedupCache::with_capacity(10);
        let key = RequestKey::from_str("test");
        cache.insert(key.clone(), "response".to_string());
        assert_eq!(cache.get(&key), Some("response"));
    }

    #[test]
    fn dedup_stats_hit_rate_zero_on_empty() {
        let stats = DedupStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }
}
