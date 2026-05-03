//! Integration tests for the advanced sampling module.
//!
//! Run with:
//! ```sh
//! cargo test -p oxibonsai-runtime -- sampling
//! ```

use oxibonsai_runtime::sampling_advanced::{
    apply_repetition_penalty, apply_temperature, entropy, softmax_inplace, top_k_indices,
    EtaSampler, LcgRng, MinPSampler, MirostatV2Sampler, SamplerChain, SamplerStep, TypicalSampler,
};

// ─────────────────────────────────────────────────────────────────────────────
// LcgRng
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lcg_rng_deterministic() {
    // Two RNGs with the same seed must produce the same sequence.
    let mut rng1 = LcgRng::new(12345);
    let mut rng2 = LcgRng::new(12345);

    for _ in 0..100 {
        assert_eq!(
            rng1.next_u64(),
            rng2.next_u64(),
            "identical seeds must produce identical u64 sequences"
        );
    }
}

#[test]
fn test_lcg_rng_f32_range() {
    let mut rng = LcgRng::new(0xdeadbeef);
    for _ in 0..10_000 {
        let v = rng.next_f32();
        assert!(
            (0.0..1.0).contains(&v),
            "next_f32 produced value outside [0, 1): {v}"
        );
    }
}

#[test]
fn test_lcg_rng_different_seeds_differ() {
    let mut rng1 = LcgRng::new(1);
    let mut rng2 = LcgRng::new(2);
    // With overwhelming probability the first values differ.
    let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
    let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
    assert_ne!(
        seq1, seq2,
        "different seeds should produce different sequences"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirostat v2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mirostat_v2_basic() {
    let logits = vec![0.1_f32, 5.0, 2.0, 3.0, 0.5];
    let mut sampler = MirostatV2Sampler::new(5.0, 0.1);
    let mut rng = LcgRng::new(42);

    let initial_mu = sampler.mu();

    let idx = sampler.sample(&logits, &mut rng);

    // Must return a valid index.
    assert!(idx < logits.len(), "token index {idx} out of range");

    // mu should have been updated after sampling.
    let new_mu = sampler.mu();
    // mu changes unless surprise exactly equals tau (highly unlikely with real data).
    // We just assert it's a finite value.
    assert!(new_mu.is_finite(), "mu became non-finite: {new_mu}");
    // At least record that mu was accessed before sampling.
    let _ = initial_mu;
}

#[test]
fn test_mirostat_v2_reduces_to_greedy_at_low_tau() {
    // With a very low tau (near 0), the threshold probability 2^{-mu} starts very high,
    // so only the top token survives — effectively greedy.
    let logits = vec![0.01_f32, 10.0, 0.5, 1.0, 0.2];
    let mut rng = LcgRng::new(7);

    // Run several steps; the dominant token (index 1, logit=10) should always win.
    let mut all_top = true;
    for _ in 0..20 {
        // Create a fresh sampler each iteration to keep tau effect clean.
        let mut s2 = MirostatV2Sampler::new(0.001, 0.1);
        let idx = s2.sample(&logits, &mut rng);
        if idx != 1 {
            all_top = false;
        }
    }
    assert!(
        all_top,
        "low-tau mirostat v2 should consistently pick the dominant token"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Typical sampler
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_typical_sampler_basic() {
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let sampler = TypicalSampler::new(0.9, 1);
    let mut rng = LcgRng::new(55);

    for _ in 0..50 {
        let idx = sampler.sample(&logits, &mut rng);
        assert!(
            idx < logits.len(),
            "typical sampler returned out-of-range index {idx}"
        );
    }
}

#[test]
fn test_typical_sampler_min_keep() {
    // Even with p=0.0 (degenerate), min_keep=3 must ensure at least 3 candidates survive.
    // We can't directly observe how many candidates survived, but sampling should not panic
    // and should return a valid index.
    let logits = vec![10.0_f32, 0.0001, 0.0001, 0.0001, 0.0001];
    let sampler = TypicalSampler::new(0.01, 3);
    let mut rng = LcgRng::new(11);

    for _ in 0..20 {
        let idx = sampler.sample(&logits, &mut rng);
        assert!(idx < logits.len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Min-P sampler
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_min_p_sampler_basic() {
    // With min_p=0.05 and a dominant token, only tokens with p >= 5% of max survive.
    let logits = vec![5.0_f32, 0.0, 0.0, 0.0, 0.0];
    let sampler = MinPSampler::new(0.05, 1);
    let mut rng = LcgRng::new(33);

    for _ in 0..30 {
        let idx = sampler.sample(&logits, &mut rng);
        // The distribution is heavily skewed; index 0 should dominate.
        assert!(
            idx < logits.len(),
            "min-p sampler returned index {idx} out of range"
        );
    }
}

#[test]
fn test_min_p_sampler_returns_valid_for_uniform() {
    let logits = vec![1.0_f32; 20];
    let sampler = MinPSampler::new(0.05, 1);
    let mut rng = LcgRng::new(22);

    for _ in 0..100 {
        let idx = sampler.sample(&logits, &mut rng);
        assert!(idx < logits.len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Eta sampler
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_eta_sampler_basic() {
    let logits = vec![0.5_f32, 3.0, 1.0, 2.0, 0.1];
    let sampler = EtaSampler::new(0.0009, 0.07);
    let mut rng = LcgRng::new(77);

    for _ in 0..50 {
        let idx = sampler.sample(&logits, &mut rng);
        assert!(
            idx < logits.len(),
            "eta sampler returned out-of-range index {idx}"
        );
    }
}

#[test]
fn test_eta_sampler_empty_logits() {
    let sampler = EtaSampler::new(0.0009, 0.07);
    let mut rng = LcgRng::new(1);
    let idx = sampler.sample(&[], &mut rng);
    assert_eq!(idx, 0, "empty logits should return 0");
}

// ─────────────────────────────────────────────────────────────────────────────
// SamplerChain — greedy preset
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sampler_chain_greedy() {
    // The greedy chain must always pick the token with the maximum logit.
    let cases: &[(&[f32], usize)] = &[
        (&[0.1, 5.0, 2.0, 3.0], 1),
        (&[9.0, 1.0, 1.0, 1.0], 0),
        (&[1.0, 1.0, 1.0, 7.0], 3),
        (&[0.0, 0.0, 4.0, 0.0], 2),
    ];

    for &(logits, expected) in cases {
        let mut chain = SamplerChain::greedy();
        let mut l = logits.to_vec();
        let tok = chain.sample(&mut l);
        assert_eq!(
            tok, expected,
            "greedy should pick {expected} from {logits:?}, got {tok}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SamplerChain — temperature
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sampler_chain_temperature() {
    // Temperature ~ 0 should collapse to greedy.
    let logits = vec![1.0_f32, 8.0, 2.0, 3.0];
    let mut chain = SamplerChain::new(42)
        .add(SamplerStep::Temperature(1e-7))
        .add(SamplerStep::Greedy);

    let mut l = logits.clone();
    let tok = chain.sample(&mut l);
    // After near-zero temperature, the max-logit token wins.
    assert_eq!(tok, 1, "near-zero temperature should pick argmax (index 1)");
}

// ─────────────────────────────────────────────────────────────────────────────
// SamplerChain — composable
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sampler_chain_composable() {
    // Chain multiple steps; result must be a valid index.
    let logits = vec![0.5_f32, 1.0, 2.5, 0.1, 3.0, 1.5];
    let mut chain = SamplerChain::new(999)
        .add(SamplerStep::Temperature(0.8))
        .add(SamplerStep::TopK(4))
        .add(SamplerStep::TopP(0.95));

    for _ in 0..30 {
        let mut l = logits.clone();
        let tok = chain.sample(&mut l);
        assert!(
            tok < logits.len(),
            "composable chain returned out-of-range index {tok}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: softmax
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_softmax_sums_to_one() {
    let cases: &[&[f32]] = &[
        &[1.0, 2.0, 3.0],
        &[0.0, 0.0, 0.0],
        &[-1.0, 0.0, 1.0, 100.0],
        &[f32::NEG_INFINITY, 1.0, 2.0],
    ];

    for &logits in cases {
        let mut v = logits.to_vec();
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().filter(|&&x| x.is_finite()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum={sum} for input {logits:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: entropy
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_entropy_uniform_distribution() {
    // For a uniform distribution over n events, H = ln(n).
    let n = 8_usize;
    let probs = vec![1.0_f32 / n as f32; n];
    let h = entropy(&probs);
    let expected = (n as f32).ln();
    assert!(
        (h - expected).abs() < 1e-4,
        "entropy of uniform({n}) should be ln({n})={expected:.4}, got {h:.4}"
    );
}

#[test]
fn test_entropy_degenerate_is_zero() {
    // A distribution concentrated on one token has H = 0.
    let mut probs = vec![0.0_f32; 10];
    probs[3] = 1.0;
    let h = entropy(&probs);
    assert!(
        h.abs() < 1e-6,
        "entropy of delta distribution should be 0, got {h}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: top_k_indices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_top_k_indices_correct() {
    let logits = vec![0.1_f32, 5.0, 3.0, 7.0, 2.0, 6.0];
    // Expected descending order: 7.0(3), 6.0(5), 5.0(1)
    let indices = top_k_indices(&logits, 3);
    assert_eq!(indices.len(), 3, "should return exactly 3 indices");
    // The set {1, 3, 5} must be exactly the returned indices.
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![1, 3, 5],
        "top-3 indices should be {{1, 3, 5}}, got {indices:?}"
    );
}

#[test]
fn test_top_k_indices_clamps_to_vocab() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    // Requesting k=10 on a 3-element slice should return all 3.
    let indices = top_k_indices(&logits, 10);
    assert_eq!(indices.len(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: apply_temperature
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_apply_temperature_scales_logits() {
    let logits = vec![2.0_f32, 4.0, 6.0];
    let mut scaled = logits.clone();
    apply_temperature(&mut scaled, 2.0);
    // Each element should be halved.
    for (orig, sc) in logits.iter().zip(scaled.iter()) {
        assert!(
            (sc - orig / 2.0).abs() < 1e-6,
            "expected {}, got {sc}",
            orig / 2.0
        );
    }
}

#[test]
fn test_apply_temperature_zero_is_noop() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    let mut copy = logits.clone();
    apply_temperature(&mut copy, 0.0);
    // Temperature=0 must leave logits unchanged (greedy is handled elsewhere).
    assert_eq!(
        copy, logits,
        "temperature=0 should be a no-op in apply_temperature"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: apply_repetition_penalty
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_repetition_penalty_reduces_seen_tokens() {
    let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let original = logits.clone();

    // Penalise token 2 (logit=3.0) and token 4 (logit=5.0).
    let seen = vec![2_u32, 4];
    apply_repetition_penalty(&mut logits, &seen, 1.5);

    // Penalised tokens should have smaller logits.
    assert!(
        logits[2] < original[2],
        "logit for seen token 2 should decrease: before={}, after={}",
        original[2],
        logits[2]
    );
    assert!(
        logits[4] < original[4],
        "logit for seen token 4 should decrease: before={}, after={}",
        original[4],
        logits[4]
    );

    // Unseen tokens must be untouched.
    assert_eq!(logits[0], original[0]);
    assert_eq!(logits[1], original[1]);
    assert_eq!(logits[3], original[3]);
}

#[test]
fn test_repetition_penalty_unity_is_noop() {
    let mut logits = vec![1.0_f32, 2.0, 3.0];
    let original = logits.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
    assert_eq!(logits, original, "penalty=1.0 should not change logits");
}

// ─────────────────────────────────────────────────────────────────────────────
// SamplerChain presets
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sampler_chain_default_chat_preset() {
    let logits = vec![0.5_f32, 3.0, 1.0, 2.0, 0.1, 4.0, 1.5];
    let mut chain = SamplerChain::default_chat(42);

    // Should produce valid indices across many runs without panicking.
    for _ in 0..100 {
        let mut l = logits.clone();
        let tok = chain.sample(&mut l);
        assert!(
            tok < logits.len(),
            "default_chat preset returned out-of-range index {tok}"
        );
    }
}

#[test]
fn test_sampler_chain_creative_preset() {
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let mut chain = SamplerChain::creative(77);

    for _ in 0..50 {
        let mut l = logits.clone();
        let tok = chain.sample(&mut l);
        assert!(tok < logits.len());
    }
}

#[test]
fn test_sampler_chain_precise_preset() {
    let logits = vec![0.1_f32, 0.2, 8.0, 0.3, 0.4];
    let mut chain = SamplerChain::precise(13);

    for _ in 0..50 {
        let mut l = logits.clone();
        let tok = chain.sample(&mut l);
        assert!(tok < logits.len());
    }
}
