//! Integration tests for the request deduplication cache.

use std::time::Duration;

use oxibonsai_runtime::dedup::{DedupCache, DedupStats, RequestKey};

// ─────────────────────────────────────────────────────────────────────────────
// RequestKey tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn request_key_from_str_deterministic() {
    let k1 = RequestKey::from_str("identical request");
    let k2 = RequestKey::from_str("identical request");
    assert_eq!(k1, k2, "same input must produce same key");
    assert_eq!(k1.value(), k2.value());
}

#[test]
fn request_key_different_strs() {
    let k1 = RequestKey::from_str("request A");
    let k2 = RequestKey::from_str("request B");
    assert_ne!(k1, k2, "different inputs must produce different keys");
}

#[test]
fn request_key_from_messages() {
    let msgs = [
        ("user", "What is Rust?"),
        ("assistant", "Rust is a language."),
    ];
    let k1 = RequestKey::from_messages(&msgs);
    let k2 = RequestKey::from_messages(&msgs);
    assert_eq!(k1, k2, "same message list must yield same key");
}

#[test]
fn request_key_from_messages_order_matters() {
    let msgs_a = [("user", "hello"), ("assistant", "world")];
    let msgs_b = [("assistant", "world"), ("user", "hello")];
    let k_a = RequestKey::from_messages(&msgs_a);
    let k_b = RequestKey::from_messages(&msgs_b);
    assert_ne!(
        k_a, k_b,
        "different message order must yield different keys"
    );
}

#[test]
fn request_key_from_messages_role_content_boundary() {
    // ("ab", "c") vs ("a", "bc") must not collide.
    let k1 = RequestKey::from_messages(&[("ab", "c")]);
    let k2 = RequestKey::from_messages(&[("a", "bc")]);
    assert_ne!(
        k1, k2,
        "role/content boundary sentinel must prevent collision"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupCache construction tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_cache_new_empty() {
    let cache = DedupCache::with_capacity(64);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert / get tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_cache_insert_get() {
    let mut cache = DedupCache::with_capacity(16);
    let key = RequestKey::from_str("What is the capital of France?");
    cache.insert(key.clone(), "Paris.".to_string());

    let result = cache.get(&key);
    assert_eq!(result, Some("Paris."), "inserted entry must be retrievable");
}

#[test]
fn dedup_cache_miss_returns_none() {
    let mut cache = DedupCache::with_capacity(16);
    let key = RequestKey::from_str("unknown query");
    assert_eq!(cache.get(&key), None, "unknown key must return None");
}

// ─────────────────────────────────────────────────────────────────────────────
// TTL / expiry tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_cache_expired_returns_none() {
    let mut cache = DedupCache::new(16, Duration::from_millis(30));
    let key = RequestKey::from_str("expiring entry");
    cache.insert(key.clone(), "response before expiry".to_string());

    // Should be present immediately.
    assert!(
        cache.get(&key).is_some(),
        "entry must be present before TTL"
    );

    std::thread::sleep(Duration::from_millis(60));

    // Should be expired now.
    assert_eq!(
        cache.get(&key),
        None,
        "entry must be absent after TTL expires"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hit count tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_cache_hit_count_increases() {
    // Access the entry multiple times and verify stats reflect the hits.
    let mut cache = DedupCache::with_capacity(16);
    let key = RequestKey::from_str("repeated query");
    cache.insert(key.clone(), "cached answer".to_string());

    cache.get(&key); // hit 1
    cache.get(&key); // hit 2
    cache.get(&key); // hit 3

    let stats = cache.stats();
    assert_eq!(stats.cache_hits, 3, "cache_hits must equal number of gets");
    assert_eq!(stats.total_requests, 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Eviction tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_cache_evict_expired() {
    let mut cache = DedupCache::new(32, Duration::from_millis(30));

    for i in 0..5_u32 {
        let key = RequestKey::from_str(&format!("query {}", i));
        cache.insert(key, format!("response {}", i));
    }
    assert_eq!(cache.len(), 5);

    std::thread::sleep(Duration::from_millis(60));

    let removed = cache.evict_expired();
    assert_eq!(removed, 5, "all entries must be evicted after TTL");
    assert_eq!(cache.len(), 0);
}

#[test]
fn dedup_cache_capacity_evicts_oldest() {
    // Capacity of 3 — inserting a 4th entry should evict the first.
    let mut cache = DedupCache::with_capacity(3);

    let k0 = RequestKey::from_str("oldest");
    let k1 = RequestKey::from_str("second");
    let k2 = RequestKey::from_str("third");
    let k3 = RequestKey::from_str("newest");

    cache.insert(k0.clone(), "r0".to_string());
    cache.insert(k1.clone(), "r1".to_string());
    cache.insert(k2.clone(), "r2".to_string());
    assert_eq!(cache.len(), 3);

    // Insert a 4th entry — oldest (k0) should be evicted.
    cache.insert(k3.clone(), "r3".to_string());
    assert_eq!(cache.len(), 3, "length must stay at capacity");

    // k0 must be gone.
    assert_eq!(cache.get(&k0), None, "oldest entry must have been evicted");
    // k3 must be present.
    assert_eq!(cache.get(&k3), Some("r3"), "newest entry must be present");
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupStats tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dedup_stats_hit_rate() {
    let stats = DedupStats {
        total_requests: 10,
        cache_hits: 4,
        cache_misses: 6,
        evictions: 0,
    };

    let rate = stats.hit_rate();
    assert!(
        (rate - 0.4).abs() < 1e-9,
        "hit_rate must be 4/10 = 0.4, got {}",
        rate
    );
}

#[test]
fn dedup_stats_summary_nonempty() {
    let stats = DedupStats {
        total_requests: 5,
        cache_hits: 3,
        cache_misses: 2,
        evictions: 1,
    };

    let summary = stats.summary();
    assert!(!summary.is_empty(), "summary must not be empty");
    // Should mention the key counters.
    assert!(
        summary.contains('5') || summary.contains("requests"),
        "{}",
        summary
    );
}
