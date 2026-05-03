//! Integration tests for the Attention Sink / StreamingLLM KV cache.
//!
//! Covers `AttentionSinkConfig`, `AttentionSinkLayer`, and `AttentionSinkCache`.
//! All test values are deterministic — no random number generation.

use oxibonsai_model::{AttentionSinkCache, AttentionSinkConfig, AttentionSinkLayer};

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/// Build a flat `[num_heads * head_dim]` KV vector filled with `val`.
fn flat_kv(num_heads: usize, head_dim: usize, val: f32) -> Vec<f32> {
    vec![val; num_heads * head_dim]
}

/// Push `n` tokens (each `val` increments by 1.0) into a layer.
fn push_n(layer: &mut AttentionSinkLayer, n: usize, num_heads: usize, head_dim: usize) {
    for i in 0..n {
        let v = i as f32;
        layer
            .push(
                &flat_kv(num_heads, head_dim, v),
                &flat_kv(num_heads, head_dim, v),
            )
            .expect("push should succeed");
    }
}

// ─────────────────────────────────────────────────────────────
// Config tests
// ─────────────────────────────────────────────────────────────

/// 1. Default config has num_sink_tokens=4 and window_size=512.
#[test]
fn sink_config_default() {
    let cfg = AttentionSinkConfig::default();
    assert_eq!(cfg.num_sink_tokens, 4, "default num_sink_tokens");
    assert_eq!(cfg.window_size, 512, "default window_size");
}

/// 2. capacity() == num_sink_tokens + window_size.
#[test]
fn sink_config_capacity() {
    let cfg = AttentionSinkConfig::new(4, 512);
    assert_eq!(cfg.capacity(), 516);
    assert_eq!(cfg.max_seq_len(), 516);

    let cfg2 = AttentionSinkConfig::new(8, 1024);
    assert_eq!(cfg2.capacity(), 1032);
}

// ─────────────────────────────────────────────────────────────
// Layer construction
// ─────────────────────────────────────────────────────────────

/// 3. Freshly constructed layer has cache_len == 0.
#[test]
fn sink_layer_new_empty() {
    let cfg = AttentionSinkConfig::default();
    let layer = AttentionSinkLayer::new(cfg, 4, 64);
    assert_eq!(layer.cache_len(), 0);
    assert_eq!(layer.recent_len(), 0);
    assert_eq!(layer.total_tokens, 0);
    assert_eq!(layer.evicted_count(), 0);
    assert!(!layer.is_streaming());
}

// ─────────────────────────────────────────────────────────────
// Sink filling
// ─────────────────────────────────────────────────────────────

/// 4. The first `num_sink_tokens` pushes fill the sink region.
#[test]
fn sink_layer_push_sink_tokens() {
    let num_sink = 4;
    let cfg = AttentionSinkConfig::new(num_sink, 512);
    let mut layer = AttentionSinkLayer::new(cfg, 2, 8);

    push_n(&mut layer, num_sink, 2, 8);

    assert_eq!(layer.cache_len(), num_sink);
    // No recent tokens yet
    assert_eq!(layer.recent_len(), 0);
    // Total tokens processed equals sink count
    assert_eq!(layer.total_tokens, num_sink);
}

/// 5. is_streaming == false before the recent window is full.
#[test]
fn sink_layer_not_streaming_early() {
    let cfg = AttentionSinkConfig::new(4, 512);
    let mut layer = AttentionSinkLayer::new(cfg, 1, 16);
    push_n(&mut layer, 100, 1, 16);
    assert!(
        !layer.is_streaming(),
        "only 100 tokens; window not yet exhausted"
    );
}

// ─────────────────────────────────────────────────────────────
// Filling the recent window
// ─────────────────────────────────────────────────────────────

/// 6. Push sink + window_size tokens; cache_len == capacity.
#[test]
fn sink_layer_push_fills_recent() {
    let num_sink = 4;
    let window = 16;
    let cfg = AttentionSinkConfig::new(num_sink, window);
    let mut layer = AttentionSinkLayer::new(cfg.clone(), 1, 4);

    // Push exactly capacity tokens
    push_n(&mut layer, cfg.capacity(), 1, 4);

    assert_eq!(layer.cache_len(), cfg.capacity());
    assert_eq!(layer.recent_len(), window);
    assert_eq!(layer.total_tokens, cfg.capacity());
}

// ─────────────────────────────────────────────────────────────
// Eviction
// ─────────────────────────────────────────────────────────────

/// 7. Pushing beyond capacity starts evictions.
#[test]
fn sink_layer_eviction_starts() {
    let cfg = AttentionSinkConfig::new(4, 8);
    let mut layer = AttentionSinkLayer::new(cfg.clone(), 1, 4);

    // Fill to capacity exactly — no evictions yet
    push_n(&mut layer, cfg.capacity(), 1, 4);
    assert_eq!(layer.evicted_count(), 0);

    // One extra token triggers the first eviction
    layer
        .push(&flat_kv(1, 4, 99.0), &flat_kv(1, 4, 99.0))
        .expect("push beyond capacity");

    assert!(layer.evicted_count() > 0, "at least one eviction expected");
    assert_eq!(layer.evicted_count(), 1);
}

/// 8. is_streaming == true after eviction starts.
#[test]
fn sink_layer_streaming_mode() {
    let cfg = AttentionSinkConfig::new(2, 4);
    let mut layer = AttentionSinkLayer::new(cfg.clone(), 1, 4);

    push_n(&mut layer, cfg.capacity() + 1, 1, 4);
    assert!(layer.is_streaming());
}

/// 9. cache_len never exceeds capacity, no matter how many tokens are pushed.
#[test]
fn sink_layer_cache_len_capped() {
    let cfg = AttentionSinkConfig::new(4, 8);
    let cap = cfg.capacity();
    let mut layer = AttentionSinkLayer::new(cfg, 1, 4);

    // Push 5× capacity
    push_n(&mut layer, cap * 5, 1, 4);

    assert_eq!(layer.cache_len(), cap, "cache_len must equal capacity");
}

// ─────────────────────────────────────────────────────────────
// Key / value retrieval
// ─────────────────────────────────────────────────────────────

/// 10. get_keys_for_head returns a non-empty vector after pushes.
#[test]
fn sink_layer_get_keys_for_head() {
    let cfg = AttentionSinkConfig::new(2, 4);
    let mut layer = AttentionSinkLayer::new(cfg, 2, 8);
    push_n(&mut layer, 4, 2, 8);

    let keys = layer.get_keys_for_head(0).expect("head 0 should exist");
    assert!(!keys.is_empty(), "keys must be non-empty");
    // 4 tokens × 8 head_dim = 32 elements
    assert_eq!(keys.len(), 4 * 8);
}

/// 11. get_values_for_head returns non-empty after pushes.
#[test]
fn sink_layer_get_values_for_head() {
    let cfg = AttentionSinkConfig::new(2, 4);
    let mut layer = AttentionSinkLayer::new(cfg, 2, 8);
    push_n(&mut layer, 4, 2, 8);

    let vals = layer.get_values_for_head(1).expect("head 1 should exist");
    assert!(!vals.is_empty());
    assert_eq!(vals.len(), 4 * 8);
}

// ─────────────────────────────────────────────────────────────
// Remapped positions
// ─────────────────────────────────────────────────────────────

/// 12. Remapped positions are strictly monotonically increasing.
#[test]
fn sink_layer_remapped_positions_monotone() {
    let cfg = AttentionSinkConfig::new(4, 8);
    let mut layer = AttentionSinkLayer::new(cfg, 1, 4);
    // Push more than capacity to trigger streaming
    push_n(&mut layer, 20, 1, 4);

    let positions = layer.get_remapped_positions();
    assert!(!positions.is_empty());
    for window in positions.windows(2) {
        assert!(
            window[1] > window[0],
            "positions must be strictly increasing: {:?}",
            positions
        );
    }
}

/// 13. Length of remapped positions equals cache_len.
#[test]
fn sink_layer_remapped_positions_length() {
    let cfg = AttentionSinkConfig::new(4, 8);
    let mut layer = AttentionSinkLayer::new(cfg, 1, 4);
    push_n(&mut layer, 20, 1, 4);

    let positions = layer.get_remapped_positions();
    assert_eq!(positions.len(), layer.cache_len());
}

/// Sink positions always start at 0.
#[test]
fn sink_layer_remapped_positions_sinks_at_zero() {
    let num_sink = 4;
    let cfg = AttentionSinkConfig::new(num_sink, 8);
    let mut layer = AttentionSinkLayer::new(cfg, 1, 4);
    push_n(&mut layer, 20, 1, 4);

    let positions = layer.get_remapped_positions();
    // First num_sink remapped positions must be 0..num_sink
    for (i, &p) in positions.iter().take(num_sink).enumerate() {
        assert_eq!(p, i, "sink position {i} should remap to {i}");
    }
}

// ─────────────────────────────────────────────────────────────
// Recent token count
// ─────────────────────────────────────────────────────────────

/// 14. recent_len tracks the number of non-sink cached tokens.
#[test]
fn sink_layer_recent_len() {
    let num_sink = 4;
    let window = 8;
    let cfg = AttentionSinkConfig::new(num_sink, window);
    let mut layer = AttentionSinkLayer::new(cfg, 1, 4);

    // Push only sink tokens — recent_len should be 0
    push_n(&mut layer, num_sink, 1, 4);
    assert_eq!(layer.recent_len(), 0);

    // Push 3 more into recent
    push_n(&mut layer, 3, 1, 4);
    assert_eq!(layer.recent_len(), 3);

    // Fill the window
    push_n(&mut layer, window - 3, 1, 4);
    assert_eq!(layer.recent_len(), window);

    // Push past window — eviction, recent_len stays at window
    push_n(&mut layer, 5, 1, 4);
    assert_eq!(layer.recent_len(), window);
}

// ─────────────────────────────────────────────────────────────
// Multi-layer cache
// ─────────────────────────────────────────────────────────────

/// 15. Multi-layer cache: push to all layers and read back from each.
#[test]
fn sink_cache_multi_layer() {
    let num_layers = 4;
    let num_heads = 2;
    let head_dim = 8;
    let cfg = AttentionSinkConfig::new(2, 6);
    let mut cache = AttentionSinkCache::new(num_layers, num_heads, head_dim, cfg);

    // Push 4 tokens
    for t in 0..4u32 {
        let k: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![t as f32; num_heads * head_dim])
            .collect();
        let v = k.clone();
        cache.push_step(&k, &v).expect("push_step");
    }

    assert_eq!(cache.cache_len(), 4);

    // Every layer must return the correct key length
    for layer in 0..num_layers {
        let keys = cache.get_keys_for_head(layer, 0).expect("get keys");
        assert_eq!(keys.len(), 4 * head_dim, "layer {layer} keys length");

        let vals = cache.get_values_for_head(layer, 1).expect("get values");
        assert_eq!(vals.len(), 4 * head_dim, "layer {layer} values length");
    }
}

/// 16. total_evicted() sums evictions across all layers.
#[test]
fn sink_cache_total_evicted() {
    let num_layers = 3;
    let num_heads = 1;
    let head_dim = 4;
    let cfg = AttentionSinkConfig::new(2, 4); // capacity = 6
    let mut cache = AttentionSinkCache::new(num_layers, num_heads, head_dim, cfg.clone());

    // Fill to capacity (no evictions)
    for t in 0..cfg.capacity() {
        let k = vec![vec![t as f32; num_heads * head_dim]; num_layers];
        let v = k.clone();
        cache.push_step(&k, &v).expect("push");
    }
    assert_eq!(cache.total_evicted(), 0);

    // Push 3 more tokens — each layer evicts 3 tokens
    for t in 0..3u32 {
        let k = vec![vec![t as f32; num_heads * head_dim]; num_layers];
        let v = k.clone();
        cache.push_step(&k, &v).expect("push evicting");
    }
    // 3 evictions × 3 layers = 9
    assert_eq!(cache.total_evicted(), 9);
}

/// 17. get_remapped_positions via layer-indexed access.
#[test]
fn sink_cache_get_remapped_positions() {
    let num_layers = 2;
    let num_heads = 1;
    let head_dim = 4;
    let cfg = AttentionSinkConfig::new(2, 4);
    let mut cache = AttentionSinkCache::new(num_layers, num_heads, head_dim, cfg);

    // Push more than capacity to trigger streaming
    for t in 0..10u32 {
        let k = vec![vec![t as f32; num_heads * head_dim]; num_layers];
        let v = k.clone();
        cache.push_step(&k, &v).expect("push");
    }

    for layer in 0..num_layers {
        let positions = cache
            .get_remapped_positions(layer)
            .expect("get_remapped_positions");
        assert_eq!(
            positions.len(),
            cache.cache_len(),
            "layer {layer} positions length"
        );
        // Strictly increasing
        for w in positions.windows(2) {
            assert!(w[1] > w[0], "positions not monotone for layer {layer}");
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Memory
// ─────────────────────────────────────────────────────────────

/// 18. memory_bytes() > 0 after at least one push.
#[test]
fn sink_layer_memory_bytes_positive() {
    let cfg = AttentionSinkConfig::new(2, 4);
    let mut layer = AttentionSinkLayer::new(cfg, 2, 8);
    assert_eq!(layer.memory_bytes(), 0, "empty layer uses 0 bytes");

    push_n(&mut layer, 3, 2, 8);
    assert!(
        layer.memory_bytes() > 0,
        "memory_bytes must be > 0 after pushes"
    );

    // Sanity: 3 tokens × 2 heads × 8 head_dim × 4 bytes × 2 (key+value) = 384 bytes
    assert_eq!(layer.memory_bytes(), 3 * 2 * 8 * 4 * 2);
}
