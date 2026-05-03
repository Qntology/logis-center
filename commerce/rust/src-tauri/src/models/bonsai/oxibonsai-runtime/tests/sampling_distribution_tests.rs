//! Statistical distribution tests for the sampling module.
//!
//! These tests verify the statistical properties of sampling by running
//! sampling many times (1000+) and validating that the resulting distributions
//! match expected behaviour for temperature, top-k, top-p, and repetition penalty.

use oxibonsai_runtime::sampling::{Sampler, SamplingParams};
use oxibonsai_runtime::sampling_advanced::{
    apply_repetition_penalty, apply_temperature, softmax_inplace, LcgRng,
};

// ══════════════════════════════════════════════════════════════════════════
// Statistical helpers
// ══════════════════════════════════════════════════════════════════════════

/// Chi-square test: compare observed counts vs expected proportions.
/// Returns the chi-square statistic.
fn chi_square_test(observed: &[usize], expected_probs: &[f64]) -> f64 {
    let total: usize = observed.iter().sum();
    let total_f = total as f64;
    observed
        .iter()
        .zip(expected_probs)
        .map(|(&o, &p)| {
            let expected = p * total_f;
            if expected < 1e-12 {
                return 0.0;
            }
            let diff = o as f64 - expected;
            diff * diff / expected
        })
        .sum()
}

/// Check whether observed frequencies match expected proportions at
/// the given significance level using a chi-square goodness-of-fit test.
fn chi_square_passes(observed: &[usize], expected_probs: &[f64]) -> bool {
    let chi2 = chi_square_test(observed, expected_probs);
    let df = observed.len() - 1;
    // Critical values at p=0.01 significance level
    let critical = match df {
        1 => 6.635,
        2 => 9.210,
        3 => 11.345,
        4 => 13.277,
        5 => 15.086,
        6 => 16.812,
        _ => 3.0 * df as f64, // rough upper bound for large df
    };
    chi2 < critical
}

/// Compute softmax probabilities from logits at a given temperature.
fn softmax_probs(logits: &[f32], temperature: f32) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| ((l - max) as f64 / temperature as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Count how many distinct values appear in a slice.
fn count_distinct(values: &[u32]) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

// ══════════════════════════════════════════════════════════════════════════
// Temperature tests
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn temperature_1_distribution_matches_softmax() {
    let logits = vec![10.0_f32, 5.0, 1.0, 0.0];
    let expected = softmax_probs(&logits, 1.0);
    let n = 5000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut counts = [0_usize; 4];

    for _ in 0..n {
        let token = sampler.sample(&logits).expect("sampling should succeed") as usize;
        assert!(token < 4, "token index out of range: {token}");
        counts[token] += 1;
    }

    assert!(
        chi_square_passes(&counts, &expected),
        "temperature=1.0 distribution should match softmax. \
         counts={counts:?}, expected_probs={expected:?}, \
         chi2={}",
        chi_square_test(&counts, &expected),
    );
}

#[test]
fn temperature_1_token_0_most_frequent() {
    let logits = vec![10.0_f32, 5.0, 1.0, 0.0];
    let n = 2000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut counts = [0_usize; 4];

    for _ in 0..n {
        let token = sampler.sample(&logits).expect("sampling should succeed") as usize;
        counts[token] += 1;
    }

    let ratio_0 = counts[0] as f64 / n as f64;
    let ratio_3 = counts[3] as f64 / n as f64;

    assert!(
        ratio_0 > 0.40,
        "token 0 (logit=10) should appear >40% of the time, got {:.2}%",
        ratio_0 * 100.0
    );
    assert!(
        ratio_3 < 0.10,
        "token 3 (logit=0) should appear <10% of the time, got {:.2}%",
        ratio_3 * 100.0
    );
}

#[test]
fn temperature_zero_always_greedy() {
    let logits = vec![10.0_f32, 5.0, 1.0, 0.0];
    let n = 1000;

    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("greedy sampling should succeed");
        assert_eq!(
            token, 0,
            "temperature=0 must always select token 0 (argmax)"
        );
    }
}

#[test]
fn temperature_very_high_approaches_uniform() {
    let logits = vec![10.0_f32, 5.0, 1.0, 0.0];
    let n = 4000;

    let params = SamplingParams {
        temperature: 100.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut counts = [0_usize; 4];

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("high temp sampling should succeed") as usize;
        counts[token] += 1;
    }

    // With temperature=100, the distribution should be nearly uniform (~25% each)
    for (i, &c) in counts.iter().enumerate() {
        let ratio = c as f64 / n as f64;
        assert!(
            (ratio - 0.25).abs() < 0.10,
            "at temp=100, token {i} should be ~25%, got {:.2}%",
            ratio * 100.0
        );
    }
}

#[test]
fn temperature_chi_square_goodness_of_fit() {
    // Test at multiple temperatures
    for temp in [0.5_f32, 1.0, 2.0] {
        let logits = vec![3.0_f32, 2.0, 1.0, 0.5, 0.1];
        let expected = softmax_probs(&logits, temp);
        let n = 5000;

        let params = SamplingParams {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };

        let mut sampler = Sampler::new(params, 777);
        let mut counts = [0_usize; 5];

        for _ in 0..n {
            let token = sampler.sample(&logits).expect("sampling should succeed") as usize;
            counts[token] += 1;
        }

        assert!(
            chi_square_passes(&counts, &expected),
            "chi-square test failed at temp={temp}: counts={counts:?}, \
             expected={expected:?}, chi2={}",
            chi_square_test(&counts, &expected),
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Top-k tests
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn top_k_1_always_selects_argmax() {
    let logits = vec![1.0_f32, 5.0, 3.0, 2.0];
    let n = 1000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_k=1 sampling should succeed");
        assert_eq!(
            token, 1,
            "top_k=1 must always select the highest logit token (index 1)"
        );
    }
}

#[test]
fn top_k_2_only_selects_top_two() {
    // logits sorted descending: index 1 (5.0), index 2 (3.0), index 3 (2.0), index 0 (1.0)
    let logits = vec![1.0_f32, 5.0, 3.0, 2.0];
    let n = 2000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut counts = [0_usize; 4];

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_k=2 sampling should succeed") as usize;
        counts[token] += 1;
    }

    // Only tokens 1 (5.0) and 2 (3.0) should be selected
    assert_eq!(
        counts[0], 0,
        "token 0 (logit=1.0) should never be selected with top_k=2"
    );
    assert_eq!(
        counts[3], 0,
        "token 3 (logit=2.0) should never be selected with top_k=2"
    );
    assert!(
        counts[1] > 0,
        "token 1 (logit=5.0) should be selected at least once"
    );
    assert!(
        counts[2] > 0,
        "token 2 (logit=3.0) should be selected at least once"
    );
}

#[test]
fn top_k_full_vocab_selects_all() {
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
    let n = 2000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 4, // same as vocab size
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut seen = [false; 4];

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_k=vocab sampling should succeed") as usize;
        seen[token] = true;
    }

    for (i, &s) in seen.iter().enumerate() {
        assert!(
            s,
            "top_k=4 with 4 logits: token {i} should be selected at least once in {n} samples"
        );
    }
}

#[test]
fn top_k_filtered_tokens_never_appear() {
    // logits: indices sorted descending: 3 (10.0), 0 (5.0), 2 (2.0), 1 (0.1)
    let logits = vec![5.0_f32, 0.1, 2.0, 10.0];
    let n = 1000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_k=2 sampling should succeed") as usize;
        assert!(
            token == 0 || token == 3,
            "top_k=2: only tokens 0 (5.0) and 3 (10.0) should appear, got {token}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Top-p (nucleus) tests
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn top_p_small_selects_dominant_token() {
    // logits [10.0, 0.0, 0.0, 0.0] -> softmax: token 0 has ~99.99% of mass
    let logits = vec![10.0_f32, 0.0, 0.0, 0.0];
    let n = 1000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut count_0 = 0_usize;

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_p=0.5 sampling should succeed");
        if token == 0 {
            count_0 += 1;
        }
    }

    let ratio = count_0 as f64 / n as f64;
    assert!(
        ratio > 0.95,
        "with logits [10,0,0,0] and top_p=0.5, token 0 should appear >95%, got {:.2}%",
        ratio * 100.0
    );
}

#[test]
fn top_p_uniform_selects_subset() {
    // Uniform logits: all tokens have equal probability
    let logits = vec![1.0_f32, 1.0, 1.0, 1.0];
    let n = 2000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut tokens_seen = Vec::new();

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_p=0.5 uniform sampling should succeed");
        tokens_seen.push(token);
    }

    let distinct = count_distinct(&tokens_seen);
    // With uniform probabilities and top_p=0.5, the nucleus should contain ~2 tokens
    // (each has p=0.25, so cumsum reaches 0.5 after 2 tokens)
    assert!(
        distinct <= 3,
        "top_p=0.5 with uniform logits should select at most ~2-3 tokens, got {distinct}"
    );
    assert!(distinct >= 1, "top_p=0.5 should select at least 1 token");
}

#[test]
fn top_p_1_allows_all_tokens() {
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let n = 3000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut seen = [false; 5];

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_p=1.0 sampling should succeed") as usize;
        if token < 5 {
            seen[token] = true;
        }
    }

    let num_seen = seen.iter().filter(|&&s| s).count();
    assert!(
        num_seen >= 3,
        "top_p=1.0 should allow most tokens to be selected, saw only {num_seen}/5"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Repetition penalty tests (using sampling_advanced helpers)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn repetition_penalty_1_no_effect() {
    let logits = vec![3.0_f32, 2.0, 1.0, 0.0];
    let seen_tokens = vec![0_u32, 1, 2, 3];

    let mut logits_penalised = logits.clone();
    apply_repetition_penalty(&mut logits_penalised, &seen_tokens, 1.0);

    assert_eq!(
        logits, logits_penalised,
        "penalty=1.0 should not change logits"
    );
}

#[test]
fn repetition_penalty_reduces_seen_token_probability() {
    let logits = vec![3.0_f32, 3.0, 3.0, 3.0];
    let n = 3000;

    // Without penalty
    let params_no = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler_no = Sampler::new(params_no, 42);
    let mut counts_no = [0_usize; 4];

    for _ in 0..n {
        let token = sampler_no
            .sample(&logits)
            .expect("no penalty sampling should succeed") as usize;
        counts_no[token] += 1;
    }

    // With penalty applied to logits for token 0
    let mut logits_penalised = logits.clone();
    apply_repetition_penalty(&mut logits_penalised, &[0], 2.0);

    let params_pen = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler_pen = Sampler::new(params_pen, 42);
    let mut counts_pen = [0_usize; 4];

    for _ in 0..n {
        let token = sampler_pen
            .sample(&logits_penalised)
            .expect("penalised sampling should succeed") as usize;
        counts_pen[token] += 1;
    }

    let ratio_no = counts_no[0] as f64 / n as f64;
    let ratio_pen = counts_pen[0] as f64 / n as f64;

    assert!(
        ratio_pen < ratio_no,
        "penalty should reduce token 0 frequency: without={:.2}%, with={:.2}%",
        ratio_no * 100.0,
        ratio_pen * 100.0
    );
}

#[test]
fn repetition_penalty_statistical_effect() {
    // Create a peaked distribution and penalise the top token
    let base_logits = vec![5.0_f32, 2.0, 1.0, 0.5];
    let seen_tokens = vec![0_u32]; // penalise the dominant token
    let n = 3000;

    // Measure frequency without penalty
    let mut sampler_baseline = Sampler::new(
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        },
        42,
    );
    let mut baseline_count_0 = 0_usize;
    for _ in 0..n {
        let token = sampler_baseline
            .sample(&base_logits)
            .expect("baseline sampling should succeed");
        if token == 0 {
            baseline_count_0 += 1;
        }
    }

    // Measure frequency with penalty
    let mut penalised = base_logits.clone();
    apply_repetition_penalty(&mut penalised, &seen_tokens, 2.0);

    let mut sampler_pen = Sampler::new(
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        },
        42,
    );
    let mut pen_count_0 = 0_usize;
    for _ in 0..n {
        let token = sampler_pen
            .sample(&penalised)
            .expect("penalised sampling should succeed");
        if token == 0 {
            pen_count_0 += 1;
        }
    }

    assert!(
        pen_count_0 < baseline_count_0,
        "repetition penalty should reduce frequency of penalised token: \
         baseline={baseline_count_0}, penalised={pen_count_0}"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Advanced sampling helper tests
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn softmax_inplace_distribution_sums_to_one() {
    for logits in [
        vec![10.0_f32, 5.0, 1.0, 0.0],
        vec![0.0_f32, 0.0, 0.0, 0.0],
        vec![-5.0_f32, -1.0, 0.0, 2.0, 10.0],
    ] {
        let mut probs = logits.clone();
        softmax_inplace(&mut probs);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1.0 for logits {:?}, got sum={sum}",
            logits
        );
    }
}

#[test]
fn apply_temperature_preserves_argmax() {
    let logits = vec![1.0_f32, 5.0, 3.0, 2.0];
    for temp in [0.1_f32, 0.5, 1.0, 2.0, 10.0] {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);

        // Find argmax of both
        let orig_max = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should find argmax of originals");
        let scaled_max = scaled
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should find argmax of scaled");

        assert_eq!(
            orig_max, scaled_max,
            "temperature scaling should preserve argmax at temp={temp}"
        );
    }
}

#[test]
fn lcg_rng_uniform_coverage() {
    // Verify that LcgRng covers all bins roughly uniformly
    let mut rng = LcgRng::new(42);
    let n = 10_000;
    const NUM_BINS: usize = 10;
    let mut bins = [0_usize; NUM_BINS];

    for _ in 0..n {
        let v = rng.next_f32();
        let bin = (v * NUM_BINS as f32).min((NUM_BINS - 1) as f32) as usize;
        bins[bin] += 1;
    }

    let expected = [1.0 / NUM_BINS as f64; NUM_BINS];
    assert!(
        chi_square_passes(&bins, &expected),
        "LcgRng should produce roughly uniform f32 values. bins={bins:?}, chi2={}",
        chi_square_test(&bins, &expected)
    );
}

#[test]
fn lcg_rng_deterministic_across_runs() {
    let mut rng1 = LcgRng::new(12345);
    let mut rng2 = LcgRng::new(12345);

    let seq1: Vec<f32> = (0..100).map(|_| rng1.next_f32()).collect();
    let seq2: Vec<f32> = (0..100).map(|_| rng2.next_f32()).collect();

    assert_eq!(
        seq1, seq2,
        "LcgRng with same seed must produce identical sequences"
    );
}

#[test]
fn softmax_probs_helper_matches_inplace() {
    let logits = vec![3.0_f32, 1.0, 2.0, 0.5];
    let temp = 1.0;
    let computed = softmax_probs(&logits, temp);

    let mut inplace = logits.clone();
    apply_temperature(&mut inplace, temp);
    softmax_inplace(&mut inplace);

    for (i, (&c, &ip)) in computed.iter().zip(inplace.iter()).enumerate() {
        assert!(
            (c - ip as f64).abs() < 1e-4,
            "mismatch at index {i}: softmax_probs={c}, inplace={ip}"
        );
    }
}

#[test]
fn temperature_scaling_flattens_distribution() {
    // At low temperature, the distribution should be more peaked.
    // At high temperature, it should be flatter.
    let logits = vec![5.0_f32, 2.0, 1.0, 0.0];

    let probs_low = softmax_probs(&logits, 0.1);
    let probs_med = softmax_probs(&logits, 1.0);
    let probs_high = softmax_probs(&logits, 10.0);

    // Entropy should increase with temperature
    let entropy_fn = |probs: &[f64]| -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum::<f64>()
    };

    let h_low = entropy_fn(&probs_low);
    let h_med = entropy_fn(&probs_med);
    let h_high = entropy_fn(&probs_high);

    assert!(
        h_low < h_med,
        "lower temperature should have lower entropy: h_low={h_low}, h_med={h_med}"
    );
    assert!(
        h_med < h_high,
        "higher temperature should have higher entropy: h_med={h_med}, h_high={h_high}"
    );
}

#[test]
fn top_k_3_excludes_bottom_tokens_statistically() {
    // 5 logits, top_k=3 -> only the 3 highest should be sampled
    let logits = vec![1.0_f32, 5.0, 3.0, 0.5, 4.0];
    // Sorted desc: index 1 (5.0), index 4 (4.0), index 2 (3.0), index 0 (1.0), index 3 (0.5)
    let n = 2000;

    let params = SamplingParams {
        temperature: 1.0,
        top_k: 3,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut sampler = Sampler::new(params, 42);
    let mut counts = [0_usize; 5];

    for _ in 0..n {
        let token = sampler
            .sample(&logits)
            .expect("top_k=3 sampling should succeed") as usize;
        counts[token] += 1;
    }

    assert_eq!(
        counts[0], 0,
        "token 0 (logit=1.0) should be excluded by top_k=3, count={}",
        counts[0]
    );
    assert_eq!(
        counts[3], 0,
        "token 3 (logit=0.5) should be excluded by top_k=3, count={}",
        counts[3]
    );
    assert!(
        counts[1] > 0 && counts[2] > 0 && counts[4] > 0,
        "all top-3 tokens should appear: counts={counts:?}"
    );
}
