//! Property-based tests for the evaluation metrics.

use oxibonsai_eval::bleu::{sentence_bleu, BleuConfig};
use oxibonsai_eval::calibration::expected_calibration_error;
use oxibonsai_eval::chrf::chrf;
use oxibonsai_eval::qa::f1_score;
use proptest::prelude::*;

const EPS: f32 = 1e-4;

// Simple ASCII-ish strategy: tokens from a small alphabet joined with spaces.
fn text_strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec("[a-z]{1,6}", 1..8).prop_map(|v| v.join(" "))
}

proptest! {
    // ──────────────────────────────────────────────────────────────────────
    // BLEU in [0,1]
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn prop_bleu_in_unit_interval(cand in text_strategy(), r in text_strategy()) {
        let cfg = BleuConfig::default();
        let s = sentence_bleu(&cand, &[&r], &cfg);
        prop_assert!((0.0..=1.0 + EPS).contains(&s.bleu));
        prop_assert!((0.0..=1.0 + EPS).contains(&s.brevity_penalty));
    }

    #[test]
    fn prop_bleu_identical_is_one(text in text_strategy()) {
        let cfg = BleuConfig::default();
        let s = sentence_bleu(&text, &[&text], &cfg);
        // For sufficiently long candidates (≥4 tokens) BLEU = 1.
        let token_count = text.split_whitespace().count();
        if token_count >= cfg.max_n {
            prop_assert!(
                (s.bleu - 1.0).abs() < 1e-3,
                "BLEU(a,a)={} for token_count={}",
                s.bleu,
                token_count
            );
        } else {
            // Shorter than max_n → BLEU collapses to 0 under default (no smoothing).
            prop_assert!(s.bleu >= 0.0);
        }
    }

    #[test]
    fn prop_bleu_precisions_in_unit_interval(
        cand in text_strategy(),
        r in text_strategy()
    ) {
        let cfg = BleuConfig::default();
        let s = sentence_bleu(&cand, &[&r], &cfg);
        for p in &s.precisions {
            prop_assert!((0.0..=1.0 + EPS).contains(p));
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // chrF in [0,1]
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn prop_chrf_in_unit_interval(cand in text_strategy(), r in text_strategy()) {
        let s = chrf(&cand, &r);
        prop_assert!((0.0..=1.0 + EPS).contains(&s.score));
    }

    #[test]
    fn prop_chrf_identical_is_one(text in text_strategy()) {
        let s = chrf(&text, &text);
        // chrF default order is 6; if the full text is shorter than 6 chars,
        // high-order n-grams are undefined and contribute 0 to the average.
        // Only assert perfect score when the text is long enough for all orders.
        if text.chars().count() >= 6 {
            prop_assert!(
                (s.score - 1.0).abs() < 1e-3,
                "chrF(a,a)={} for text_chars={}",
                s.score,
                text.chars().count()
            );
        } else {
            // Shorter inputs still score in [0,1] but may be below 1.
            prop_assert!((0.0..=1.0 + EPS).contains(&s.score));
        }
    }

    #[test]
    fn prop_chrf_symmetric_in_range(a in text_strategy(), b in text_strategy()) {
        // chrF is not strictly symmetric (precision vs recall are swapped) but
        // both directions remain in [0,1].
        let s_ab = chrf(&a, &b);
        let s_ba = chrf(&b, &a);
        prop_assert!((0.0..=1.0 + EPS).contains(&s_ab.score));
        prop_assert!((0.0..=1.0 + EPS).contains(&s_ba.score));
    }

    // ──────────────────────────────────────────────────────────────────────
    // F1 in [0,1]
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn prop_f1_in_unit_interval(pred in text_strategy(), r in text_strategy()) {
        let f = f1_score(&pred, &r);
        prop_assert!((0.0..=1.0 + EPS).contains(&f));
    }

    #[test]
    fn prop_f1_identical_is_one(text in text_strategy()) {
        let f = f1_score(&text, &text);
        prop_assert!((f - 1.0).abs() < 1e-3);
    }

    // ──────────────────────────────────────────────────────────────────────
    // ECE bounded
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn prop_ece_bounded(
        confidences in proptest::collection::vec(0.0f32..=1.0, 1..64),
        bits in proptest::collection::vec(0u8..=1, 1..64),
    ) {
        // Align lengths — zip smaller.
        let n = confidences.len().min(bits.len());
        let conf = &confidences[..n];
        let correct = &bits[..n];
        let (ece, stats) = expected_calibration_error(conf, correct, 10).expect("ece");
        prop_assert!((0.0..=1.0 + EPS).contains(&ece));
        // Per-bin counts must sum to n.
        let total: usize = stats.iter().map(|s| s.count).sum();
        prop_assert_eq!(total, n);
    }

    #[test]
    fn prop_ece_clamped_inputs_ok(
        confidences in proptest::collection::vec(-0.5f32..=1.5, 1..32),
        bits in proptest::collection::vec(0u8..=1, 1..32),
    ) {
        // Out-of-range confidences should be clamped, not crash.
        let n = confidences.len().min(bits.len());
        let (ece, _) = expected_calibration_error(&confidences[..n], &bits[..n], 5)
            .expect("ece");
        prop_assert!((0.0..=1.0 + EPS).contains(&ece));
    }

    // ──────────────────────────────────────────────────────────────────────
    // Range invariance under varied bin counts
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn prop_ece_any_bin_count_in_range(
        confidences in proptest::collection::vec(0.0f32..=1.0, 1..32),
        bits in proptest::collection::vec(0u8..=1, 1..32),
        n_bins in 1usize..20usize,
    ) {
        let n = confidences.len().min(bits.len());
        let (ece, _) = expected_calibration_error(&confidences[..n], &bits[..n], n_bins)
            .expect("ece");
        prop_assert!((0.0..=1.0 + EPS).contains(&ece));
    }
}
