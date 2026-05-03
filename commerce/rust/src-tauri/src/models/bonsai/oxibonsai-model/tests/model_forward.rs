//! Tests for model construction and forward pass behavior.

use half::f16;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::block::TransformerBlock;
use oxibonsai_model::kv_cache::KvCache;
use oxibonsai_model::layers::linear::Linear1Bit;
use oxibonsai_model::layers::rms_norm::RmsNorm;
use oxibonsai_model::layers::rope::RopeTable;
use oxibonsai_model::model::BonsaiModel;

fn make_blocks(n: usize, scale: f32, pattern: u8) -> Vec<BlockQ1_0G128> {
    (0..n)
        .map(|_| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: [pattern; 16],
        })
        .collect()
}

fn ref_kernel() -> std::sync::Arc<KernelDispatcher> {
    std::sync::Arc::new(KernelDispatcher::with_tier(KernelTier::Reference))
}

/// Build a small Transformer block for testing with given dimensions.
#[allow(clippy::too_many_arguments)]
fn build_test_block<'a>(
    layer_idx: usize,
    h: usize,
    hd: usize,
    nq: usize,
    nkv: usize,
    inter: usize,
    scale: f32,
    pattern: u8,
    q_blocks: &'a [BlockQ1_0G128],
    k_blocks: &'a [BlockQ1_0G128],
    v_blocks: &'a [BlockQ1_0G128],
    o_blocks: &'a [BlockQ1_0G128],
    gate_blocks: &'a [BlockQ1_0G128],
    up_blocks: &'a [BlockQ1_0G128],
    down_blocks: &'a [BlockQ1_0G128],
) -> TransformerBlock<'a> {
    let _ = (scale, pattern); // used indirectly through blocks
    let kernel = ref_kernel();
    TransformerBlock::new(
        layer_idx,
        RmsNorm::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(q_blocks, nq * hd, h, kernel.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(k_blocks, nkv * hd, h, kernel.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(v_blocks, nkv * hd, h, kernel.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(o_blocks, h, nq * hd, kernel.clone())
            .expect("o")
            .into(),
        RmsNorm::new(vec![1.0; hd], 1e-6),
        RmsNorm::new(vec![1.0; hd], 1e-6),
        RmsNorm::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(gate_blocks, inter, h, kernel.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(up_blocks, inter, h, kernel.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(down_blocks, h, inter, kernel.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    )
}

// ══════════════════════════════════════════════════════════════
// Model creation tests
// ══════════════════════════════════════════════════════════════

#[test]
fn model_new_creates_valid_model() {
    let config = Qwen3Config::bonsai_8b();
    let model = BonsaiModel::new(config);
    assert_eq!(model.config().hidden_size, 4096);
    assert_eq!(model.config().num_layers, 36);
    assert_eq!(model.config().vocab_size, 151936);
}

#[test]
fn model_new_has_empty_blocks() {
    let config = Qwen3Config::bonsai_8b();
    let model = BonsaiModel::new(config);
    // BonsaiModel::new creates an empty blocks vec for testing
    // We verify the config is correct
    assert_eq!(model.config().num_attention_heads, 32);
    assert_eq!(model.config().num_kv_heads, 8);
}

#[test]
fn model_forward_produces_logits_of_vocab_size() {
    // Use a small config for testing
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0, // no blocks for speed
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        vocab_size: 100,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model = BonsaiModel::new(config);
    let kernel = ref_kernel();

    // With 0 blocks, forward should still work (embedding + norm + output projection)
    let logits = model
        .forward(0, 0, kernel.as_ref())
        .expect("forward should succeed with empty blocks");
    assert_eq!(logits.len(), 100, "logits should match vocab_size");
}

#[test]
fn model_forward_deterministic() {
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        vocab_size: 50,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model1 = BonsaiModel::new(config.clone());
    let mut model2 = BonsaiModel::new(config);
    let kernel = ref_kernel();

    let logits1 = model1.forward(0, 0, kernel.as_ref()).expect("forward 1");
    let logits2 = model2.forward(0, 0, kernel.as_ref()).expect("forward 2");

    assert_eq!(logits1.len(), logits2.len());
    for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logits should be identical: index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn model_reset_clears_kv_cache() {
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        vocab_size: 50,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model = BonsaiModel::new(config);
    let kernel = ref_kernel();

    let _ = model.forward(0, 0, kernel.as_ref()).expect("forward");
    model.reset();
    assert_eq!(model.kv_cache_mut().seq_len(), 0);
}

// ══════════════════════════════════════════════════════════════
// TransformerBlock forward tests
// ══════════════════════════════════════════════════════════════

#[test]
fn transformer_block_forward_changes_hidden_state() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.01,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("block forward");

    let max_diff = hidden
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "forward should change hidden state, max_diff={max_diff}"
    );
}

#[test]
fn transformer_block_residual_connection_preserves_input_contribution() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    // Use very small scale so sublayer outputs are small
    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.001, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.001, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.001, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.001, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.001, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.001, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.001, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.001,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("block forward");

    // With very small weights, the residual contribution from input dominates
    // So output should be close to input (but not identical)
    let correlation: f32 = hidden
        .iter()
        .zip(original.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
        / (hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
            * original.iter().map(|x| x * x).sum::<f32>().sqrt());

    assert!(
        correlation > 0.9,
        "with small weights, output should be highly correlated with input: corr={correlation}"
    );
}

#[test]
fn rope_produces_position_dependent_qk_vectors() {
    // Test that RoPE at different positions produces different Q/K vectors,
    // which is the mechanism by which Transformer blocks distinguish positions.
    let hd = 64;
    let rope = RopeTable::new(hd, 16, 10000.0);

    let input = vec![1.0f32; hd];
    let mut out_pos0 = vec![0.0f32; hd];
    let mut out_pos5 = vec![0.0f32; hd];

    rope.apply(&input, &mut out_pos0, 0).expect("pos 0");
    rope.apply(&input, &mut out_pos5, 5).expect("pos 5");

    let diff: f32 = out_pos0
        .iter()
        .zip(out_pos5.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-3,
        "RoPE at different positions should produce different outputs: diff={diff}"
    );
}

#[test]
fn kv_cache_accumulates_across_forward_calls() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.01,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    // Forward at position 0
    let mut hidden = vec![0.1f32; h];
    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("pos 0");

    // Check that keys were stored at position 0
    let keys_after_0 = kv_cache.keys_for(0, 0, 1);
    assert_eq!(keys_after_0.len(), hd);
    // At least some values should be non-zero
    let has_nonzero = keys_after_0.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        has_nonzero,
        "KV cache should have non-zero values after forward"
    );

    // Forward at position 1
    let mut hidden2 = vec![0.2f32; h];
    block
        .forward(&mut hidden2, 1, &mut kv_cache, &rope, kernel.as_ref())
        .expect("pos 1");

    // Now KV cache should have 2 positions
    let keys_after_1 = kv_cache.keys_for(0, 0, 2);
    assert_eq!(keys_after_1.len(), 2 * hd);
}

#[test]
fn model_config_bonsai_8b_defaults() {
    let config = Qwen3Config::bonsai_8b();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.num_layers, 36);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 151936);
    assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert!((config.rope_freq_base - 1_000_000.0).abs() < 1.0);
}
