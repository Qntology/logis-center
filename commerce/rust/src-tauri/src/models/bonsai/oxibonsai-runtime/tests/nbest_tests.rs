//! Integration tests for N-best hypothesis tracking.

use oxibonsai_runtime::nbest::{Hypothesis, NBestDecoder, NBestList};

// ─── Hypothesis tests ──────────────────────────────────────────────────────────

#[test]
fn hypothesis_new() {
    let h = Hypothesis::new(vec![1u32, 2, 3], -3.0);
    assert_eq!(h.tokens, vec![1u32, 2, 3]);
    assert!((h.log_prob - -3.0).abs() < f64::EPSILON);
    assert!(!h.is_complete);
}

#[test]
fn hypothesis_extend() {
    let h = Hypothesis::new(vec![1u32, 2], -2.0);
    let h2 = h.extend(3, -1.0f32);
    assert_eq!(h2.tokens, vec![1u32, 2, 3]);
    assert!((h2.log_prob - -3.0).abs() < 1e-6);
    assert!(!h2.is_complete);
}

#[test]
fn hypothesis_complete() {
    let h = Hypothesis::new(vec![1u32, 2], -2.0);
    let h = h.complete(2);
    assert!(h.is_complete);
}

#[test]
fn hypothesis_score_normalized() {
    // A longer sequence with proportionally worse total log_prob has a lower normalised score.
    let short = Hypothesis::new(vec![1u32], -1.0);
    let long_bad = Hypothesis::new(vec![1u32, 2, 3, 4, 5], -10.0);
    // short: -1.0/1 = -1.0; long: -10.0/5 = -2.0
    assert!(
        short.score() > long_bad.score(),
        "short={} long_bad={}",
        short.score(),
        long_bad.score()
    );
}

// ─── NBestList tests ───────────────────────────────────────────────────────────

#[test]
fn nbest_list_new() {
    let list = NBestList::new(5);
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
    assert!(!list.is_full());
}

#[test]
fn nbest_list_push_under_capacity() {
    let mut list = NBestList::new(5);
    list.push(Hypothesis::new(vec![1u32], -1.0));
    list.push(Hypothesis::new(vec![2u32], -2.0));
    list.push(Hypothesis::new(vec![3u32], -3.0));
    assert_eq!(list.len(), 3);
    assert!(!list.is_full());
}

#[test]
fn nbest_list_push_over_capacity() {
    let mut list = NBestList::new(3);
    for i in 0..5u32 {
        // scores: 0.0, -1.0, -2.0, -3.0, -4.0
        list.push(Hypothesis::new(vec![i], -(i as f64)));
    }
    assert_eq!(list.len(), 3);
    let sorted = list.into_sorted();
    // Best three: tokens [0],[1],[2]
    assert_eq!(sorted[0].tokens, vec![0u32]);
    assert_eq!(sorted[1].tokens, vec![1u32]);
    assert_eq!(sorted[2].tokens, vec![2u32]);
}

#[test]
fn nbest_list_worst_score() {
    let mut list = NBestList::new(3);
    list.push(Hypothesis::new(vec![1u32], -1.0));
    list.push(Hypothesis::new(vec![2u32], -2.0));
    list.push(Hypothesis::new(vec![3u32], -3.0));
    let worst = list.worst_score().expect("should have worst score");
    assert!((worst - -3.0).abs() < 1e-9, "worst={worst}");
}

#[test]
fn nbest_list_into_sorted_order() {
    let mut list = NBestList::new(5);
    list.push(Hypothesis::new(vec![1u32], -3.0));
    list.push(Hypothesis::new(vec![2u32], -1.0));
    list.push(Hypothesis::new(vec![3u32], -2.0));
    let sorted = list.into_sorted();
    assert_eq!(sorted.len(), 3);
    // Best first
    assert!(
        (sorted[0].log_prob - -1.0).abs() < 1e-9,
        "best={}",
        sorted[0].log_prob
    );
    assert!(
        (sorted[1].log_prob - -2.0).abs() < 1e-9,
        "second={}",
        sorted[1].log_prob
    );
    assert!(
        (sorted[2].log_prob - -3.0).abs() < 1e-9,
        "third={}",
        sorted[2].log_prob
    );
}

#[test]
fn nbest_list_complete_hypotheses() {
    let mut list = NBestList::new(5);
    list.push(Hypothesis::new(vec![1u32], -1.0).complete(2));
    list.push(Hypothesis::new(vec![2u32], -2.0));
    let complete = list.complete_hypotheses();
    assert_eq!(complete.len(), 1);
    assert!(complete[0].is_complete);
}

// ─── NBestDecoder tests ────────────────────────────────────────────────────────

#[test]
fn nbest_decoder_step_expands() {
    let decoder = NBestDecoder::new(5, 99, 20);
    let hyps = vec![Hypothesis::new(vec![1u32], -0.5)];
    let logits = vec![vec![0.0f32; 10]];
    let expanded = decoder.step(&hyps, &logits, 3);
    // One hypothesis × top_k=3 expansions
    assert!(expanded.len() >= 3, "got {} expansions", expanded.len());
}

#[test]
fn nbest_decoder_partition() {
    let active_h = Hypothesis::new(vec![1u32], -1.0);
    let complete_h = Hypothesis::new(vec![2u32], -2.0).complete(2);
    let (active, complete) = NBestDecoder::partition(vec![active_h, complete_h]);
    assert_eq!(active.len(), 1);
    assert_eq!(complete.len(), 1);
    assert!(!active[0].is_complete);
    assert!(complete[0].is_complete);
}

#[test]
fn nbest_decoder_eos_completes() {
    let eos = 2u32;
    let decoder = NBestDecoder::new(5, eos, 20);
    let hyps = vec![Hypothesis::new(vec![1u32], -0.5)];
    let mut logits = vec![f32::NEG_INFINITY; 5];
    logits[eos as usize] = 10.0;
    let expanded = decoder.step(&hyps, &[logits], 1);
    assert!(!expanded.is_empty());
    // The top-1 expansion should be the EOS token
    let eos_hyps: Vec<_> = expanded.iter().filter(|h| h.is_complete).collect();
    assert!(
        !eos_hyps.is_empty(),
        "expected at least one complete hypothesis"
    );
}

#[test]
fn nbest_decoder_length_penalty() {
    // Longer sequences with proportionally worse log_prob have lower normalised score.
    let h_short = Hypothesis::new(vec![1u32], -1.0);
    let h_long = Hypothesis::new(vec![1u32, 2, 3, 4], -6.0);
    // short: -1.0/1=-1.0; long: -6.0/4=-1.5
    assert!(
        h_short.score() > h_long.score(),
        "short={} long={}",
        h_short.score(),
        h_long.score()
    );
}
