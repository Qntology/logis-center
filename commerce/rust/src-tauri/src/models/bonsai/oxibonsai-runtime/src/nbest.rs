//! N-best hypothesis tracking for diverse generation.
//!
//! Maintains a heap of the N best partial sequences seen during decoding,
//! scored by cumulative log-probability. Used for:
//! - Diverse beam search
//! - Post-generation reranking
//! - Multi-hypothesis output

use std::cmp;
use std::collections::BinaryHeap;

use crate::beam_search::BeamSearchEngine;

// ─── Hypothesis ────────────────────────────────────────────────────────────────

/// A single hypothesis (partial or complete token sequence).
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// All token IDs in this hypothesis.
    pub tokens: Vec<u32>,
    /// Cumulative log probability of the sequence.
    pub log_prob: f64,
    /// Length-normalised score: log_prob / tokens.len().max(1)
    pub normalized_score: f64,
    /// Whether this hypothesis ended with an EOS token.
    pub is_complete: bool,
}

impl Hypothesis {
    /// Create a new hypothesis with the given tokens and cumulative log probability.
    pub fn new(tokens: Vec<u32>, log_prob: f64) -> Self {
        let len = tokens.len().max(1) as f64;
        let normalized_score = log_prob / len;
        Self {
            tokens,
            log_prob,
            normalized_score,
            is_complete: false,
        }
    }

    /// Return the length-normalised score used for ranking.
    pub fn score(&self) -> f64 {
        self.normalized_score
    }

    /// Extend this hypothesis with one more token, accumulating its log probability.
    pub fn extend(&self, token: u32, token_log_prob: f32) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        let log_prob = self.log_prob + token_log_prob as f64;
        let len = tokens.len().max(1) as f64;
        let normalized_score = log_prob / len;
        Self {
            tokens,
            log_prob,
            normalized_score,
            is_complete: false,
        }
    }

    /// Mark this hypothesis as complete (ended with EOS).
    pub fn complete(mut self, _eos_id: u32) -> Self {
        self.is_complete = true;
        self
    }

    /// Number of tokens in this hypothesis.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the hypothesis has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// Ordering by normalized_score (higher is better).
// BinaryHeap is a max-heap, so Reverse<Hypothesis> gives a min-heap suitable for NBestList.

impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.normalized_score.total_cmp(&other.normalized_score) == cmp::Ordering::Equal
    }
}

impl Eq for Hypothesis {}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.normalized_score.total_cmp(&other.normalized_score)
    }
}

// ─── NBestList ─────────────────────────────────────────────────────────────────

/// A fixed-capacity heap of the N best hypotheses.
///
/// Internally uses a min-heap (via `Reverse`) so that the worst hypothesis is
/// always at the top and can be evicted when capacity is exceeded.
pub struct NBestList {
    capacity: usize,
    /// Min-heap: the root is the *worst* kept hypothesis.
    hypotheses: BinaryHeap<cmp::Reverse<Hypothesis>>,
}

impl NBestList {
    /// Create an empty N-best list with the given capacity.
    pub fn new(n: usize) -> Self {
        Self {
            capacity: n,
            hypotheses: BinaryHeap::with_capacity(n + 1),
        }
    }

    /// Push a hypothesis into the list.
    ///
    /// If the list is already at capacity and the new hypothesis scores better
    /// than the current worst, the worst is evicted.
    pub fn push(&mut self, hyp: Hypothesis) {
        if self.capacity == 0 {
            return;
        }
        if self.hypotheses.len() < self.capacity {
            self.hypotheses.push(cmp::Reverse(hyp));
        } else {
            // Only replace worst if the new hypothesis is strictly better.
            let should_insert = self
                .hypotheses
                .peek()
                .map(|cmp::Reverse(worst)| hyp.score() > worst.score())
                .unwrap_or(true);

            if should_insert {
                self.hypotheses.pop();
                self.hypotheses.push(cmp::Reverse(hyp));
            }
        }
    }

    /// Return a reference to the best hypothesis (highest score) in the list.
    pub fn top(&self) -> Option<&Hypothesis> {
        // The min-heap's root is the worst; we need to scan for best.
        self.hypotheses
            .iter()
            .map(|cmp::Reverse(h)| h)
            .max_by(|a, b| a.score().total_cmp(&b.score()))
    }

    /// Number of hypotheses currently held.
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// Whether the list has no hypotheses.
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Whether the list has reached its capacity.
    pub fn is_full(&self) -> bool {
        self.hypotheses.len() >= self.capacity
    }

    /// Score of the worst hypothesis currently kept, or `None` if empty.
    pub fn worst_score(&self) -> Option<f64> {
        self.hypotheses.peek().map(|cmp::Reverse(h)| h.score())
    }

    /// Consume the list and return hypotheses sorted best-first.
    pub fn into_sorted(self) -> Vec<Hypothesis> {
        let mut v: Vec<Hypothesis> = self
            .hypotheses
            .into_iter()
            .map(|cmp::Reverse(h)| h)
            .collect();
        v.sort_by(|a, b| b.score().total_cmp(&a.score()));
        v
    }

    /// Return references to all complete hypotheses.
    pub fn complete_hypotheses(&self) -> Vec<&Hypothesis> {
        self.hypotheses
            .iter()
            .map(|cmp::Reverse(h)| h)
            .filter(|h| h.is_complete)
            .collect()
    }
}

// ─── NBestDecoder ──────────────────────────────────────────────────────────────

/// Decoder that expands hypotheses by one step and maintains an N-best list.
pub struct NBestDecoder {
    /// Maximum number of hypotheses to track.
    pub n: usize,
    /// Token ID that marks end-of-sequence.
    pub eos_id: u32,
    /// Maximum generation length (inclusive).
    pub max_len: usize,
    /// Length-penalty exponent α used for normalised scoring.
    pub length_penalty: f32,
}

impl NBestDecoder {
    /// Create a new decoder.
    pub fn new(n: usize, eos_id: u32, max_len: usize) -> Self {
        Self {
            n,
            eos_id,
            max_len,
            length_penalty: 1.0,
        }
    }

    /// Set the length-penalty exponent (builder pattern).
    pub fn with_length_penalty(mut self, alpha: f32) -> Self {
        self.length_penalty = alpha;
        self
    }

    /// Expand a batch of hypotheses by one step.
    ///
    /// `logits_per_hyp[i]` must be the logit vector for `hypotheses[i]`.
    /// Returns the flat list of expanded hypotheses (up to `top_k` per input).
    pub fn step(
        &self,
        hypotheses: &[Hypothesis],
        logits_per_hyp: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<Hypothesis> {
        let effective_k = top_k.max(1);
        let mut expanded: Vec<Hypothesis> = Vec::new();

        for (hyp, logits) in hypotheses.iter().zip(logits_per_hyp.iter()) {
            if hyp.is_complete {
                expanded.push(hyp.clone());
                continue;
            }

            let top = BeamSearchEngine::top_k_log_probs(logits, effective_k);

            for (token, log_prob) in top {
                let new_hyp = hyp.extend(token, log_prob as f32);
                let new_hyp = if token == self.eos_id {
                    new_hyp.complete(self.eos_id)
                } else {
                    new_hyp
                };
                expanded.push(new_hyp);
            }
        }

        expanded
    }

    /// Return an empty N-best list with this decoder's capacity.
    pub fn init(&self) -> NBestList {
        NBestList::new(self.n)
    }

    /// Partition hypotheses into (active, complete).
    pub fn partition(hyps: Vec<Hypothesis>) -> (Vec<Hypothesis>, Vec<Hypothesis>) {
        let mut active = Vec::new();
        let mut complete = Vec::new();
        for h in hyps {
            if h.is_complete {
                complete.push(h);
            } else {
                active.push(h);
            }
        }
        (active, complete)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypothesis_new() {
        let h = Hypothesis::new(vec![1, 2, 3], -3.0);
        assert_eq!(h.tokens, vec![1, 2, 3]);
        assert!((h.log_prob - -3.0).abs() < f64::EPSILON);
        assert!(!h.is_complete);
    }

    #[test]
    fn hypothesis_extend() {
        let h = Hypothesis::new(vec![1, 2], -2.0);
        let h2 = h.extend(3, -1.0);
        assert_eq!(h2.tokens, vec![1, 2, 3]);
        assert!((h2.log_prob - -3.0).abs() < 1e-6);
    }

    #[test]
    fn hypothesis_complete() {
        let h = Hypothesis::new(vec![1, 2], -2.0);
        let h = h.complete(2);
        assert!(h.is_complete);
    }

    #[test]
    fn hypothesis_score_normalized() {
        let short = Hypothesis::new(vec![1], -1.0);
        let _long = Hypothesis::new(vec![1, 2, 3, 4], -4.0);
        // short: -1.0/1 = -1.0; long: -4.0/4 = -1.0 — same here.
        // But a longer sequence with same total lp has same normalised score.
        // Let's verify that a longer bad sequence scores worse.
        let long_bad = Hypothesis::new(vec![1, 2, 3, 4, 5], -10.0);
        assert!(long_bad.score() < short.score());
    }

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
        for i in 0..3u32 {
            list.push(Hypothesis::new(vec![i], -(i as f64)));
        }
        assert_eq!(list.len(), 3);
        assert!(!list.is_full());
    }

    #[test]
    fn nbest_list_push_over_capacity() {
        let mut list = NBestList::new(3);
        // Push 5 hypotheses with scores -0.0, -1.0, -2.0, -3.0, -4.0
        for i in 0..5u32 {
            list.push(Hypothesis::new(vec![i], -(i as f64)));
        }
        assert_eq!(list.len(), 3);
        // Should keep the three best (i=0,1,2 with scores 0,-1,-2)
        let sorted = list.into_sorted();
        assert_eq!(sorted.len(), 3);
        // Best should be token [0] with score 0.0
        assert_eq!(sorted[0].tokens, vec![0]);
    }

    #[test]
    fn nbest_list_worst_score() {
        let mut list = NBestList::new(3);
        list.push(Hypothesis::new(vec![1], -1.0));
        list.push(Hypothesis::new(vec![2], -2.0));
        list.push(Hypothesis::new(vec![3], -3.0));
        let worst = list.worst_score().expect("should have worst score");
        assert!((worst - -3.0).abs() < 1e-9);
    }

    #[test]
    fn nbest_list_into_sorted_order() {
        let mut list = NBestList::new(5);
        list.push(Hypothesis::new(vec![1], -3.0));
        list.push(Hypothesis::new(vec![2], -1.0));
        list.push(Hypothesis::new(vec![3], -2.0));
        let sorted = list.into_sorted();
        assert_eq!(sorted.len(), 3);
        // Best first
        assert!((sorted[0].log_prob - -1.0).abs() < 1e-9);
        assert!((sorted[1].log_prob - -2.0).abs() < 1e-9);
        assert!((sorted[2].log_prob - -3.0).abs() < 1e-9);
    }

    #[test]
    fn nbest_list_complete_hypotheses() {
        let mut list = NBestList::new(5);
        list.push(Hypothesis::new(vec![1], -1.0).complete(2));
        list.push(Hypothesis::new(vec![3], -2.0));
        let complete = list.complete_hypotheses();
        assert_eq!(complete.len(), 1);
        assert!(complete[0].is_complete);
    }

    #[test]
    fn nbest_decoder_step_expands() {
        let decoder = NBestDecoder::new(5, 99, 20);
        let hyps = vec![Hypothesis::new(vec![1], -0.5)];
        let logits = vec![vec![0.0f32; 10]];
        let expanded = decoder.step(&hyps, &logits, 3);
        assert!(expanded.len() >= 3);
    }

    #[test]
    fn nbest_decoder_partition() {
        let active_h = Hypothesis::new(vec![1], -1.0);
        let complete_h = Hypothesis::new(vec![2], -2.0).complete(2);
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
        let hyps = vec![Hypothesis::new(vec![1], -0.5)];
        // Give EOS token the highest logit
        let mut logits = vec![f32::NEG_INFINITY; 5];
        logits[eos as usize] = 10.0;
        let expanded = decoder.step(&hyps, &[logits], 1);
        assert!(!expanded.is_empty());
        assert!(expanded[0].is_complete);
    }

    #[test]
    fn nbest_decoder_length_penalty() {
        // Longer sequences have lower normalised score when log_prob is proportional.
        let h_short = Hypothesis::new(vec![1], -1.0);
        let h_long = Hypothesis::new(vec![1, 2, 3, 4], -6.0);
        // short: -1.0/1=-1.0; long: -6.0/4=-1.5
        assert!(h_short.score() > h_long.score());
    }
}
