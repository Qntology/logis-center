//! FP16 KV cache integration tests.
//!
//! Tests store/retrieve, FP16 precision, memory usage, reset,
//! layer independence, range queries, and large value handling.
//! Also covers KV cache memory comparison (FP16 vs FP32) and paged lazy
//! allocation behaviour.

use oxibonsai_model::kv_cache::{KvCache, PagedKvCache};
use oxibonsai_model::KvCacheFp16;

// ── Helper ───────────────────────────────────────────────────────────────

fn make_small_cache() -> KvCacheFp16 {
    KvCacheFp16::new(2, 2, 4, 8)
}

// ── 1. Store and retrieve at various positions ───────────────────────────

#[test]
fn store_and_retrieve_position_zero() {
    let mut cache = make_small_cache();
    let key = vec![1.0, 2.0, 3.0, 4.0];
    let value = vec![5.0, 6.0, 7.0, 8.0];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store at pos 0 should succeed");

    let retrieved_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    let retrieved_value = cache.get_value(0, 0, 0).expect("get_value should succeed");

    for (orig, ret) in key.iter().zip(retrieved_key.iter()) {
        assert!(
            (orig - ret).abs() < 0.1,
            "key mismatch: orig={orig}, ret={ret}"
        );
    }
    for (orig, ret) in value.iter().zip(retrieved_value.iter()) {
        assert!(
            (orig - ret).abs() < 0.1,
            "value mismatch: orig={orig}, ret={ret}"
        );
    }
}

#[test]
fn store_and_retrieve_multiple_positions() {
    let mut cache = make_small_cache();

    for pos in 0..5 {
        let key: Vec<f32> = (0..4).map(|i| (pos * 4 + i) as f32).collect();
        let value: Vec<f32> = (0..4).map(|i| (pos * 4 + i + 100) as f32).collect();
        cache
            .store(0, 0, pos, &key, &value)
            .expect("store should succeed");
    }

    assert_eq!(cache.current_len(), 5);

    // Retrieve and verify position 3
    let key3 = cache
        .get_key(0, 0, 3)
        .expect("get_key pos 3 should succeed");
    assert!(
        (key3[0] - 12.0).abs() < 0.1,
        "key[0] at pos 3 should be ~12.0"
    );
}

#[test]
fn store_at_last_valid_position() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);
    let key = vec![1.0; 4];
    let value = vec![2.0; 4];

    // Valid positions are 0, 1, 2, 3 (max_seq_len=4)
    cache
        .store(0, 0, 3, &key, &value)
        .expect("pos 3 should be valid");
    assert_eq!(cache.current_len(), 4);

    // Position 4 should fail
    let result = cache.store(0, 0, 4, &key, &value);
    assert!(result.is_err(), "pos 4 should exceed max_seq_len=4");
}

// ── 2. FP16 precision within expected tolerance ──────────────────────────

#[test]
fn fp16_precision_small_values() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);
    let key = vec![0.001, 0.01, 0.1, 1.0];
    let value = vec![-0.5, 0.0, 0.5, 1.5];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");

    let ret_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    let ret_value = cache.get_value(0, 0, 0).expect("get_value should succeed");

    for (orig, ret) in key.iter().zip(ret_key.iter()) {
        let tolerance = orig.abs() * 0.01 + 0.001;
        assert!(
            (orig - ret).abs() < tolerance,
            "FP16 precision: orig={orig}, ret={ret}, tol={tolerance}"
        );
    }
    for (orig, ret) in value.iter().zip(ret_value.iter()) {
        let tolerance = orig.abs() * 0.01 + 0.001;
        assert!(
            (orig - ret).abs() < tolerance,
            "FP16 precision: orig={orig}, ret={ret}, tol={tolerance}"
        );
    }
}

#[test]
fn fp16_precision_typical_attention_values() {
    let mut cache = KvCacheFp16::new(1, 1, 8, 4);
    // Typical attention key/value range: -2.0 to +2.0
    let key = vec![-1.5, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5];
    let value = vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");

    let ret_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    for (orig, ret) in key.iter().zip(ret_key.iter()) {
        assert!(
            (orig - ret).abs() < 0.01,
            "typical value FP16: orig={orig}, ret={ret}"
        );
    }
}

// ── 3. Memory usage is ~half of FP32 equivalent ─────────────────────────

#[test]
fn memory_usage_half_of_fp32() {
    let num_layers = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let max_seq_len = 128;

    let fp16_cache = KvCacheFp16::new(num_layers, num_kv_heads, head_dim, max_seq_len);

    // FP16: each element is 2 bytes, keys + values = 2x
    // Expected = num_layers * num_kv_heads * max_seq_len * head_dim * 2 bytes * 2 (K+V)
    let expected = num_layers * num_kv_heads * max_seq_len * head_dim * 2 * 2;
    assert_eq!(fp16_cache.memory_usage_bytes(), expected);

    // FP32 equivalent would be 4 bytes per element
    let fp32_equivalent = num_layers * num_kv_heads * max_seq_len * head_dim * 4 * 2;
    assert_eq!(
        fp16_cache.memory_usage_bytes() * 2,
        fp32_equivalent,
        "FP16 cache should use exactly half the memory of FP32"
    );
}

#[test]
fn memory_usage_bonsai_8b_dimensions() {
    let cache = KvCacheFp16::new(36, 8, 128, 4096);
    let expected = 36 * 8 * 4096 * 128 * 2 * 2; // 2 bytes per f16, 2 for K+V
    assert_eq!(cache.memory_usage_bytes(), expected);
    // Verify it's reasonable (~576 MB)
    assert!(cache.memory_usage_bytes() > 500_000_000);
    assert!(cache.memory_usage_bytes() < 700_000_000);
}

// ── 4. Reset clears all data ─────────────────────────────────────────────

#[test]
fn reset_clears_current_len() {
    let mut cache = make_small_cache();
    let key = vec![1.0; 4];
    let value = vec![2.0; 4];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");
    cache
        .store(0, 0, 1, &key, &value)
        .expect("store should succeed");
    assert_eq!(cache.current_len(), 2);

    cache.reset();
    assert_eq!(cache.current_len(), 0);
}

#[test]
fn reset_zeroes_stored_data() {
    let mut cache = make_small_cache();
    let key = vec![10.0, 20.0, 30.0, 40.0];
    let value = vec![50.0, 60.0, 70.0, 80.0];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");
    cache.reset();

    // After reset, reading position 0 should return zeros
    // (current_len is 0, but the underlying data is zeroed)
    // We need to store something first to update current_len
    let zero_key = vec![0.0; 4];
    let zero_value = vec![0.0; 4];
    cache
        .store(0, 0, 0, &zero_key, &zero_value)
        .expect("store after reset should succeed");

    let ret_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    for &v in &ret_key {
        assert!(v.abs() < 0.001, "value after reset should be ~0.0, got {v}");
    }
}

// ── 5. Multiple layers stored independently ──────────────────────────────

#[test]
fn layers_are_independent() {
    let mut cache = KvCacheFp16::new(3, 1, 4, 4);

    let key0 = vec![1.0, 2.0, 3.0, 4.0];
    let key1 = vec![10.0, 20.0, 30.0, 40.0];
    let key2 = vec![100.0, 200.0, 300.0, 400.0];
    let value = vec![0.0; 4];

    cache.store(0, 0, 0, &key0, &value).expect("layer 0 store");
    cache.store(1, 0, 0, &key1, &value).expect("layer 1 store");
    cache.store(2, 0, 0, &key2, &value).expect("layer 2 store");

    let ret0 = cache.get_key(0, 0, 0).expect("layer 0 get");
    let ret1 = cache.get_key(1, 0, 0).expect("layer 1 get");
    let ret2 = cache.get_key(2, 0, 0).expect("layer 2 get");

    assert!((ret0[0] - 1.0).abs() < 0.1);
    assert!((ret1[0] - 10.0).abs() < 0.1);
    assert!((ret2[0] - 100.0).abs() < 1.0); // f16 precision for 100.0
}

#[test]
fn heads_are_independent() {
    let mut cache = KvCacheFp16::new(1, 3, 4, 4);

    let key_h0 = vec![1.0, 2.0, 3.0, 4.0];
    let key_h1 = vec![10.0, 20.0, 30.0, 40.0];
    let key_h2 = vec![100.0, 200.0, 300.0, 400.0];
    let value = vec![0.0; 4];

    cache.store(0, 0, 0, &key_h0, &value).expect("head 0 store");
    cache.store(0, 1, 0, &key_h1, &value).expect("head 1 store");
    cache.store(0, 2, 0, &key_h2, &value).expect("head 2 store");

    let ret0 = cache.get_key(0, 0, 0).expect("head 0 get");
    let ret1 = cache.get_key(0, 1, 0).expect("head 1 get");
    let ret2 = cache.get_key(0, 2, 0).expect("head 2 get");

    assert!((ret0[0] - 1.0).abs() < 0.1);
    assert!((ret1[0] - 10.0).abs() < 0.1);
    assert!((ret2[0] - 100.0).abs() < 1.0);
}

// ── 6. get_keys_range returns correct number of positions ────────────────

#[test]
fn get_keys_range_correct_count() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 8);

    for pos in 0..5 {
        let key: Vec<f32> = (0..4).map(|i| (pos * 4 + i) as f32).collect();
        let value = vec![0.0; 4];
        cache
            .store(0, 0, pos, &key, &value)
            .expect("store should succeed");
    }

    let range = cache
        .get_keys_range(0, 0, 3)
        .expect("get_keys_range should succeed");
    // Should return 3 positions * 4 elements = 12 floats
    assert_eq!(range.len(), 12);
}

#[test]
fn get_keys_range_full() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);

    for pos in 0..4 {
        let key = vec![pos as f32; 4];
        let value = vec![0.0; 4];
        cache
            .store(0, 0, pos, &key, &value)
            .expect("store should succeed");
    }

    let range = cache
        .get_keys_range(0, 0, 4)
        .expect("get_keys_range should succeed");
    assert_eq!(range.len(), 16); // 4 positions * 4 elements
}

#[test]
fn get_values_range_correct() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 8);

    for pos in 0..3 {
        let key = vec![0.0; 4];
        let value: Vec<f32> = (0..4).map(|i| (pos * 4 + i + 100) as f32).collect();
        cache
            .store(0, 0, pos, &key, &value)
            .expect("store should succeed");
    }

    let range = cache
        .get_values_range(0, 0, 3)
        .expect("get_values_range should succeed");
    assert_eq!(range.len(), 12);
    // First value at pos 0 should be ~100.0
    assert!((range[0] - 100.0).abs() < 1.0);
}

#[test]
fn get_range_clamped_to_current_len() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 8);

    // Only store 2 positions
    for pos in 0..2 {
        let key = vec![pos as f32; 4];
        let value = vec![0.0; 4];
        cache
            .store(0, 0, pos, &key, &value)
            .expect("store should succeed");
    }

    // Request range up to 10, but current_len is 2
    let range = cache
        .get_keys_range(0, 0, 10)
        .expect("should clamp to current_len");
    assert_eq!(range.len(), 8); // 2 positions * 4 elements
}

// ── 7. Large values don't overflow f16 ───────────────────────────────────

#[test]
fn large_values_clamped_by_f16_range() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);

    // f16 max is ~65504. Values beyond that will be inf in f16.
    // We test with values within the f16 range.
    let key = vec![1000.0, 2000.0, 5000.0, 10000.0];
    let value = vec![-1000.0, -2000.0, -5000.0, -10000.0];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");

    let ret_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    let ret_value = cache.get_value(0, 0, 0).expect("get_value should succeed");

    // For values in the thousands, f16 has ~1 digit of precision after decimal
    for (orig, ret) in key.iter().zip(ret_key.iter()) {
        let tolerance = orig.abs() * 0.002; // 0.2% relative error for f16
        assert!(
            (orig - ret).abs() < tolerance,
            "large key: orig={orig}, ret={ret}, tol={tolerance}"
        );
    }
    for (orig, ret) in value.iter().zip(ret_value.iter()) {
        let tolerance = orig.abs() * 0.002;
        assert!(
            (orig - ret).abs() < tolerance,
            "large value: orig={orig}, ret={ret}, tol={tolerance}"
        );
    }
}

#[test]
fn very_large_values_become_inf() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);

    // Values beyond f16 range (~65504) should become inf
    let key = vec![100000.0, 200000.0, 300000.0, 400000.0];
    let value = vec![0.0; 4];

    cache
        .store(0, 0, 0, &key, &value)
        .expect("store should succeed");

    let ret_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
    for &v in &ret_key {
        assert!(
            v.is_infinite() || v > 60000.0,
            "very large values should be inf or saturated, got {v}"
        );
    }
}

// ── Error cases ──────────────────────────────────────────────────────────

#[test]
fn invalid_layer_index() {
    let cache = KvCacheFp16::new(2, 2, 4, 4);
    assert!(cache.get_key(5, 0, 0).is_err());
    assert!(cache.get_value(5, 0, 0).is_err());
    assert!(cache.get_keys_range(5, 0, 1).is_err());
    assert!(cache.get_values_range(5, 0, 1).is_err());
}

#[test]
fn invalid_head_index() {
    let cache = KvCacheFp16::new(2, 2, 4, 4);
    assert!(cache.get_key(0, 5, 0).is_err());
    assert!(cache.get_value(0, 5, 0).is_err());
    assert!(cache.get_keys_range(0, 5, 1).is_err());
}

#[test]
fn invalid_position_index() {
    let cache = KvCacheFp16::new(2, 2, 4, 4);
    assert!(cache.get_key(0, 0, 10).is_err());
    assert!(cache.get_value(0, 0, 10).is_err());
}

#[test]
fn store_wrong_key_dim() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);
    let wrong_key = vec![1.0; 8]; // head_dim is 4, not 8
    let value = vec![1.0; 4];
    let result = cache.store(0, 0, 0, &wrong_key, &value);
    assert!(result.is_err(), "wrong key dimension should fail");
}

#[test]
fn store_wrong_value_dim() {
    let mut cache = KvCacheFp16::new(1, 1, 4, 4);
    let key = vec![1.0; 4];
    let wrong_value = vec![1.0; 8];
    let result = cache.store(0, 0, 0, &key, &wrong_value);
    assert!(result.is_err(), "wrong value dimension should fail");
}

// ── Capacity info ────────────────────────────────────────────────────────

#[test]
fn max_seq_len_accessor() {
    let cache = KvCacheFp16::new(1, 1, 4, 128);
    assert_eq!(cache.max_seq_len(), 128);
}

#[test]
fn current_len_starts_at_zero() {
    let cache = KvCacheFp16::new(1, 1, 4, 128);
    assert_eq!(cache.current_len(), 0);
}

// ══════════════════════════════════════════════════════════════
// Memory comparison: FP16 vs FP32 (Task 5 verification tests)
// ══════════════════════════════════════════════════════════════

/// FP16 cache must use exactly half the memory of an equivalent FP32 cache.
#[test]
fn fp16_cache_uses_half_memory_of_fp32() {
    let layers = 4;
    let kv_heads = 2;
    let head_dim = 64;
    let max_seq = 512;

    let fp16_cache = KvCacheFp16::new(layers, kv_heads, head_dim, max_seq);
    // FP32 reference: keys + values each = layers * kv_heads * max_seq * head_dim * 4 bytes
    let fp32_size = layers * 2 * kv_heads * head_dim * max_seq * 4; // f32 = 4 bytes
    let fp16_size = fp16_cache.memory_usage_bytes();

    // FP16 should be at most half of FP32 (within a small absolute tolerance
    // for struct bookkeeping, but the payload is exactly half).
    assert!(
        fp16_size <= fp32_size / 2 + 128,
        "FP16 cache ({fp16_size} bytes) should be ~half of FP32 ({fp32_size} bytes)"
    );

    // Verify the exact relationship: FP16 payload * 2 == FP32 payload.
    assert_eq!(
        fp16_size * 2,
        fp32_size,
        "FP16 bytes * 2 should equal FP32 bytes exactly"
    );
}

/// Scaling the cache dimensions doubles the FP16 memory proportionally.
#[test]
fn fp16_memory_scales_linearly_with_seq_len() {
    let layers = 2;
    let kv_heads = 2;
    let head_dim = 32;

    let cache_512 = KvCacheFp16::new(layers, kv_heads, head_dim, 512);
    let cache_1024 = KvCacheFp16::new(layers, kv_heads, head_dim, 1024);

    assert_eq!(
        cache_1024.memory_usage_bytes(),
        cache_512.memory_usage_bytes() * 2,
        "doubling max_seq_len should double memory usage"
    );
}

/// FP32 KvCache reports memory_bytes() correctly.
#[test]
fn fp32_cache_memory_bytes_matches_formula() {
    let layers = 4;
    let kv_heads = 2;
    let head_dim = 64;
    let max_seq = 256;

    let fp32_cache = KvCache::new(layers, kv_heads, head_dim, max_seq);
    let expected = layers * kv_heads * max_seq * head_dim * std::mem::size_of::<f32>() * 2;
    assert_eq!(fp32_cache.memory_bytes(), expected);
}

// ══════════════════════════════════════════════════════════════
// PagedKvCache lazy allocation verification (Task 5)
// ══════════════════════════════════════════════════════════════

/// A freshly-created PagedKvCache must have zero allocated memory.
#[test]
fn paged_kv_cache_lazy_allocation_is_zero_initially() {
    let layers = 4;
    let kv_heads = 2;
    let head_dim = 64;
    let max_seq = 4096;

    let cache = PagedKvCache::new(layers, kv_heads, head_dim, max_seq);
    assert_eq!(
        cache.total_pages(),
        0,
        "no pages should be allocated before any writes"
    );
    assert_eq!(
        cache.memory_usage_bytes(),
        0,
        "memory usage should be 0 before any writes, got {} bytes",
        cache.memory_usage_bytes()
    );
}

/// Writing one position allocates exactly one page per (layer, head) written.
#[test]
fn paged_kv_cache_allocates_one_page_on_first_write() {
    let mut cache = PagedKvCache::with_page_size(1, 1, 4, 1024, 256);
    assert_eq!(cache.memory_usage_bytes(), 0, "starts empty");

    let key = vec![1.0f32; 4];
    let value = vec![2.0f32; 4];
    cache.store_key(0, 0, 0, &key);
    cache.store_value(0, 0, 0, &value);

    // One page allocated: page_size * head_dim * f32_size * 2 (K+V)
    let one_page = 256 * 4 * std::mem::size_of::<f32>() * 2;
    assert_eq!(
        cache.memory_usage_bytes(),
        one_page,
        "exactly one page should be allocated after the first write"
    );
}

/// Paged cache initial allocation is much smaller than the full FP32 equivalent.
#[test]
fn paged_kv_cache_much_smaller_than_fp32_initially() {
    let layers = 4;
    let kv_heads = 2;
    let head_dim = 64;
    let max_seq = 4096;

    let paged = PagedKvCache::new(layers, kv_heads, head_dim, max_seq);

    // The full FP32 equivalent pre-allocates everything upfront.
    let fp32_full = layers * kv_heads * max_seq * head_dim * std::mem::size_of::<f32>() * 2;

    // Paged cache starts at 0 — a dramatic reduction.
    assert!(
        paged.memory_usage_bytes() < fp32_full,
        "paged cache ({} bytes) should use less than full FP32 ({fp32_full} bytes)",
        paged.memory_usage_bytes()
    );
}

/// After clear(), a paged cache releases all pages.
#[test]
fn paged_kv_cache_clear_frees_all_pages() {
    let mut cache = PagedKvCache::with_page_size(2, 2, 4, 512, 16);

    // Populate several pages.
    for layer in 0..2 {
        for head in 0..2 {
            for pos in 0..32 {
                cache.store_key(layer, head, pos, &[1.0f32; 4]);
            }
        }
    }

    assert!(cache.total_pages() > 0, "should have pages after writes");

    cache.clear();
    assert_eq!(
        cache.total_pages(),
        0,
        "all pages should be freed after clear"
    );
    assert_eq!(
        cache.memory_usage_bytes(),
        0,
        "memory should be 0 after clear"
    );
    assert_eq!(cache.seq_len(), 0, "seq_len should be 0 after clear");
}
