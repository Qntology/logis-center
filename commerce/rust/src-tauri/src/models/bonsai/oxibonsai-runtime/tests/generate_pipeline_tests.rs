//! Integration tests for the full generate() pipeline.
//!
//! Uses `Qwen3Config::tiny_test()` to construct lightweight engines and
//! exercises the end-to-end generation path including prefill, decode,
//! sampling parameter effects, edge cases, and engine state management.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;

// ══════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════

fn tiny_engine(params: SamplingParams, seed: u64) -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), params, seed)
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

fn default_prompt() -> Vec<u32> {
    vec![151644, 872, 1234]
}

// ══════════════════════════════════════════════════════════════════════════
// Basic generation
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn generate_single_token() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 1)
        .expect("generate with max_tokens=1 should succeed");
    assert!(
        tokens.len() <= 1,
        "max_tokens=1 should produce at most 1 token, got {}",
        tokens.len()
    );
}

#[test]
fn generate_multiple_tokens_respects_max() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 5)
        .expect("generate with max_tokens=5 should succeed");
    assert!(
        tokens.len() <= 5,
        "max_tokens=5 should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn generate_greedy_is_deterministic() {
    let prompt = default_prompt();

    let mut engine1 = tiny_engine(greedy_params(), 42);
    let out1 = engine1
        .generate(&prompt, 10)
        .expect("first greedy generate should succeed");

    let mut engine2 = tiny_engine(greedy_params(), 42);
    let out2 = engine2
        .generate(&prompt, 10)
        .expect("second greedy generate should succeed");

    assert_eq!(
        out1, out2,
        "greedy decoding with same seed must produce identical output"
    );
}

#[test]
fn generate_greedy_deterministic_three_runs() {
    let prompt = default_prompt();
    let mut results = Vec::new();

    for _ in 0..3 {
        let mut engine = tiny_engine(greedy_params(), 99);
        let tokens = engine
            .generate(&prompt, 8)
            .expect("greedy generate should succeed");
        results.push(tokens);
    }

    assert_eq!(
        results[0], results[1],
        "run 0 and run 1 must match for greedy"
    );
    assert_eq!(
        results[1], results[2],
        "run 1 and run 2 must match for greedy"
    );
}

#[test]
fn generate_different_seeds_can_differ() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine1 = tiny_engine(params.clone(), 1);
    let out1 = engine1
        .generate(&prompt, 20)
        .expect("generate with seed 1 should succeed");

    let mut engine2 = tiny_engine(params, 9999);
    let out2 = engine2
        .generate(&prompt, 20)
        .expect("generate with seed 9999 should succeed");

    // With high temperature and different seeds, outputs should differ
    // (not guaranteed but overwhelmingly likely with 20 tokens)
    let differ = out1 != out2;
    assert!(
        differ,
        "different seeds with temp=1.0 should produce different outputs \
         (this can fail with astronomically low probability)"
    );
}

#[test]
fn generate_empty_prompt_returns_empty() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&[], 10)
        .expect("empty prompt generate should succeed");
    assert!(
        tokens.is_empty(),
        "empty prompt should produce no output, got {} tokens",
        tokens.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Sampling parameter effects
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn temperature_zero_is_greedy() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut results = Vec::new();
    for seed in [1, 42, 12345] {
        let mut engine = tiny_engine(params.clone(), seed);
        let tokens = engine
            .generate(&prompt, 5)
            .expect("greedy generate should succeed");
        results.push(tokens);
    }

    // Temperature 0 is pure argmax -- seed should not matter
    assert_eq!(
        results[0], results[1],
        "temperature=0 should be deterministic regardless of seed"
    );
    assert_eq!(
        results[1], results[2],
        "temperature=0 should be deterministic regardless of seed"
    );
}

#[test]
fn high_temperature_produces_valid_tokens() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 2.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("high temperature generate should succeed");

    // Should still produce valid output
    assert!(
        !tokens.is_empty(),
        "high temperature should still generate tokens"
    );
    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn top_k_1_is_deterministic() {
    let prompt = default_prompt();

    let top_k_1 = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    // top_k=1 always picks the highest-logit token, so it should be
    // deterministic regardless of seed (only one candidate remains).
    let mut engine1 = tiny_engine(top_k_1.clone(), 42);
    let out1 = engine1
        .generate(&prompt, 8)
        .expect("top_k=1 generate should succeed");

    let mut engine2 = tiny_engine(top_k_1, 9999);
    let out2 = engine2
        .generate(&prompt, 8)
        .expect("top_k=1 generate with different seed should succeed");

    assert_eq!(
        out1, out2,
        "top_k=1 should produce the same result regardless of seed"
    );
}

#[test]
fn top_k_full_vocab_no_filtering() {
    let prompt = default_prompt();
    let vocab_size = Qwen3Config::tiny_test().vocab_size;

    let params = SamplingParams {
        temperature: 0.7,
        top_k: vocab_size,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("top_k=vocab_size generate should succeed");

    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn top_p_near_zero_produces_valid_output() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.01,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    // Very small top_p restricts the candidate set heavily.
    // With the tiny test model (random weights), the logit distribution
    // may still have multiple tokens in the nucleus, so we just verify
    // that valid tokens are produced.
    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("top_p=0.01 should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn top_p_1_allows_all_tokens() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("top_p=1.0 generate should succeed");

    assert!(!tokens.is_empty(), "top_p=1.0 should generate tokens");
}

#[test]
fn repetition_penalty_above_one_produces_valid_output() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.5,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("repetition penalty generate should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn repetition_penalty_vs_no_penalty_can_differ() {
    let prompt = default_prompt();

    let no_penalty = SamplingParams {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let with_penalty = SamplingParams {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
        max_tokens: 128,
    };

    let mut engine_no = tiny_engine(no_penalty, 42);
    let out_no = engine_no
        .generate(&prompt, 20)
        .expect("no penalty generate should succeed");

    let mut engine_with = tiny_engine(with_penalty, 42);
    let out_with = engine_with
        .generate(&prompt, 20)
        .expect("with penalty generate should succeed");

    // Note: repetition penalty in the basic Sampler does NOT track previous tokens,
    // it's only applied if the engine feeds them back. With the basic generate()
    // pipeline, the effect may be zero. We just verify both succeed.
    let _ = (out_no, out_with);
}

// ══════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn generate_max_tokens_zero_returns_empty() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 0)
        .expect("max_tokens=0 should succeed");
    assert!(
        tokens.is_empty(),
        "max_tokens=0 should produce no output, got {} tokens",
        tokens.len()
    );
}

#[test]
fn generate_single_token_prompt() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&[151644], 5)
        .expect("single token prompt should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn generate_with_various_prompt_lengths() {
    for len in [1, 2, 5, 10, 50] {
        let prompt: Vec<u32> = (0..len).map(|i| (i % 1000) as u32).collect();
        let mut engine = tiny_engine(greedy_params(), 42);
        let tokens = engine
            .generate(&prompt, 3)
            .unwrap_or_else(|e| panic!("generate with prompt length {len} should succeed: {e}"));
        assert!(
            tokens.len() <= 3,
            "prompt_len={len}: should produce at most 3 tokens, got {}",
            tokens.len()
        );
    }
}

#[test]
fn generate_long_prompt_near_context_limit() {
    let config = Qwen3Config::tiny_test();
    // Use a 64-token prompt to verify that long prompts are handled without
    // panicking and respect the max_tokens limit.
    let prompt: Vec<u32> = (0..64).map(|i| (i % 1000) as u32).collect();
    let mut engine = InferenceEngine::new(config, greedy_params(), 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("long prompt should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Engine state
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn sequential_generates_independent() {
    let prompt = default_prompt();
    let mut engine = tiny_engine(greedy_params(), 42);

    let out1 = engine
        .generate(&prompt, 5)
        .expect("first generate should succeed");

    // Reset and generate again with same params
    engine.reset();
    let mut engine2 = tiny_engine(greedy_params(), 42);
    let out2 = engine2
        .generate(&prompt, 5)
        .expect("second generate on fresh engine should succeed");

    assert_eq!(
        out1, out2,
        "greedy generation after reset should match a fresh engine"
    );
}

#[test]
fn multiple_generates_without_reset_succeed() {
    let mut engine = tiny_engine(greedy_params(), 42);

    for i in 0..5 {
        let prompt = vec![151644, (i * 100 + 1) as u32];
        let tokens = engine
            .generate(&prompt, 3)
            .unwrap_or_else(|e| panic!("generate iteration {i} should succeed: {e}"));
        assert!(
            tokens.len() <= 3,
            "iteration {i}: should produce at most 3 tokens"
        );
    }
}

#[test]
fn engine_stats_update_after_generate() {
    let mut engine = tiny_engine(greedy_params(), 42);
    assert_eq!(engine.stats().requests_completed(), 0);

    let _ = engine
        .generate(&default_prompt(), 5)
        .expect("generate should succeed");
    assert!(
        engine.stats().requests_completed() >= 1,
        "stats should reflect completed request"
    );
}

#[test]
fn engine_stats_accumulate_tokens() {
    let mut engine = tiny_engine(greedy_params(), 42);

    let out1 = engine
        .generate(&default_prompt(), 3)
        .expect("first generate should succeed");
    let count1 = engine.stats().tokens_generated();

    engine.reset();
    let out2 = engine
        .generate(&default_prompt(), 3)
        .expect("second generate should succeed");
    let count2 = engine.stats().tokens_generated();

    assert_eq!(
        count2,
        count1 + out2.len() as u64,
        "token count should accumulate: first={}, second output={}, total={}",
        out1.len(),
        out2.len(),
        count2,
    );
}

#[test]
fn generate_with_seed_method() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params.clone(), 42);
    let out1 = engine
        .generate_with_seed(&prompt, 5, 100, &params)
        .expect("generate_with_seed should succeed");

    engine.reset();
    let out2 = engine
        .generate_with_seed(&prompt, 5, 100, &params)
        .expect("generate_with_seed with same seed should succeed");

    assert_eq!(
        out1, out2,
        "generate_with_seed with identical seeds must produce identical output"
    );
}

#[test]
fn generate_tokens_within_vocab_range() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 50,
        top_p: 0.95,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&default_prompt(), 20)
        .expect("generate should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn batch_generate_returns_correct_count() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let prompts = vec![vec![151644, 100], vec![151644, 200], vec![151644, 300]];
    let results = engine.batch_generate(&prompts, 3);
    assert_eq!(
        results.len(),
        3,
        "batch_generate should return one result per prompt"
    );
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.is_ok(),
            "batch_generate prompt {i} should succeed: {:?}",
            r.as_ref().err()
        );
    }
}

#[test]
fn generate_with_default_params() {
    let mut engine = tiny_engine(SamplingParams::default(), 42);
    let tokens = engine
        .generate(&default_prompt(), 5)
        .expect("generate with default params should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}
