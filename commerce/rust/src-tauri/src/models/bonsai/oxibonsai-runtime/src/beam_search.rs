//! Beam search decoding for OxiBonsai.
//!
//! Beam search maintains `beam_width` candidate sequences simultaneously,
//! expanding each at every step and keeping the top-`beam_width` by
//! cumulative log-probability (with optional length penalty).
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_runtime::beam_search::{BeamSearchConfig, BeamSearchEngine};
//!
//! let config = BeamSearchConfig {
//!     beam_width: 2,
//!     max_tokens: 10,
//!     eos_token_id: 2,
//!     ..Default::default()
//! };
//! let engine = BeamSearchEngine::new(config);
//!
//! // Mock logits: always prefer token 5
//! let result = engine.search(vec![1, 2], 10, |_tokens, _step| {
//!     let mut logits = vec![0.0f32; 10];
//!     logits[5] = 10.0;
//!     logits[2] = -10.0; // EOS gets low score
//!     logits
//! });
//!
//! assert!(!result.best().is_empty());
//! ```

// ─── Config ────────────────────────────────────────────────────────────────

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of parallel beams to maintain (typical: 4–8).
    pub beam_width: usize,
    /// Maximum tokens to generate per beam.
    pub max_tokens: usize,
    /// Length penalty exponent α: `score = log_prob / len^α`.
    ///
    /// Values in [0.6, 1.0] are typical. α = 1.0 is neutral; α < 1.0
    /// rewards longer sequences; α > 1.0 penalises them.
    pub length_penalty: f32,
    /// Block any token that would create a repeated n-gram of this size.
    /// Set to 0 to disable (default).
    pub no_repeat_ngram_size: usize,
    /// Stop as soon as the best beam generates an EOS token.
    pub early_stopping: bool,
    /// Token ID that marks end of sequence.
    pub eos_token_id: u32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_tokens: 256,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: 2,
        }
    }
}

// ─── Beam ──────────────────────────────────────────────────────────────────

/// One candidate sequence in the beam search.
#[derive(Debug, Clone)]
pub struct Beam {
    /// All token IDs in this candidate (prompt + generated so far).
    pub tokens: Vec<u32>,
    /// Cumulative log-probability of this sequence.
    pub log_prob: f64,
    /// Whether this beam has hit an EOS token and is finished.
    pub is_done: bool,
}

impl Beam {
    /// Create a new beam seeded with the given initial tokens.
    pub fn new(initial_tokens: Vec<u32>) -> Self {
        Self {
            tokens: initial_tokens,
            log_prob: 0.0,
            is_done: false,
        }
    }

    /// Length-normalised score used for beam ranking.
    ///
    /// `score = log_prob / (len ^ length_penalty)`
    ///
    /// Avoids division-by-zero by treating a zero-length sequence as length 1.
    pub fn score(&self, length_penalty: f32) -> f64 {
        let len = self.tokens.len().max(1) as f64;
        self.log_prob / len.powf(length_penalty as f64)
    }

    /// Extend the beam with one more token, returning a new beam.
    pub fn extend(&self, token: u32, log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        Self {
            tokens,
            log_prob: self.log_prob + log_prob,
            is_done: false,
        }
    }

    /// Total number of tokens in this beam.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` when the beam contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ─── Result ────────────────────────────────────────────────────────────────

/// Output of a beam search run.
#[derive(Debug)]
pub struct BeamSearchResult {
    /// All completed sequences, ordered best-first.
    pub sequences: Vec<Vec<u32>>,
    /// Length-normalised score for each sequence.
    pub scores: Vec<f64>,
    /// Number of generation steps taken.
    pub num_steps: usize,
}

impl BeamSearchResult {
    /// The highest-scoring token sequence.
    pub fn best(&self) -> &[u32] {
        self.sequences.first().map(|s| s.as_slice()).unwrap_or(&[])
    }

    /// Score of the highest-scoring sequence.
    pub fn best_score(&self) -> f64 {
        self.scores.first().copied().unwrap_or(f64::NEG_INFINITY)
    }
}

// ─── Engine ────────────────────────────────────────────────────────────────

/// Beam search engine.
///
/// Decoupled from the model via a `get_logits` closure so it can be used
/// with any inference backend.
pub struct BeamSearchEngine {
    /// Search configuration.
    pub config: BeamSearchConfig,
}

impl BeamSearchEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Run beam search.
    ///
    /// `get_logits(beam_tokens, step)` is called for every live beam at every
    /// step and must return a logit vector of length `vocab_size`.
    pub fn search<F>(
        &self,
        initial_tokens: Vec<u32>,
        _vocab_size: usize,
        mut get_logits: F,
    ) -> BeamSearchResult
    where
        F: FnMut(&[u32], usize) -> Vec<f32>,
    {
        let cfg = &self.config;
        let bw = cfg.beam_width.max(1);

        // Initialise with a single beam
        let mut beams: Vec<Beam> = vec![Beam::new(initial_tokens)];
        let mut completed: Vec<Beam> = Vec::new();
        let mut steps = 0;

        for step in 0..cfg.max_tokens {
            steps = step + 1;

            // Collect live (non-done) beams
            let live: Vec<Beam> = beams.iter().filter(|b| !b.is_done).cloned().collect();

            if live.is_empty() {
                steps = step;
                break;
            }

            // Expand every live beam
            let mut candidates: Vec<Beam> = Vec::new();

            for beam in &live {
                let mut logits = get_logits(&beam.tokens, step);

                // Apply no-repeat-ngram masking if configured
                if cfg.no_repeat_ngram_size > 0 {
                    Self::apply_no_repeat_ngram(
                        &mut logits,
                        &beam.tokens,
                        cfg.no_repeat_ngram_size,
                    );
                }

                // Get top-k (token, log_prob) candidates from this beam
                let top = Self::top_k_log_probs(&logits, bw);

                for (token, lp) in top {
                    let mut new_beam = beam.extend(token, lp);

                    if token == cfg.eos_token_id {
                        new_beam.is_done = true;
                        if cfg.early_stopping {
                            completed.push(new_beam);
                            continue;
                        }
                    }
                    candidates.push(new_beam);
                }
            }

            // Keep any already-done beams from the previous round
            // Use drain to avoid moving `beams` so we can still use it after break
            let done_indices: Vec<usize> = beams
                .iter()
                .enumerate()
                .filter(|(_, b)| b.is_done)
                .map(|(i, _)| i)
                .collect();
            // Remove done beams in reverse index order to preserve indices
            for &idx in done_indices.iter().rev() {
                completed.push(beams.remove(idx));
            }

            if candidates.is_empty() {
                break;
            }

            // Prune to beam_width
            beams = Self::prune_beams(candidates, bw, cfg.length_penalty);

            // Early-stop when best completed beam outscores every live beam
            if cfg.early_stopping && !completed.is_empty() {
                let best_completed_score = completed
                    .iter()
                    .map(|b| b.score(cfg.length_penalty))
                    .fold(f64::NEG_INFINITY, f64::max);

                let best_live_score = beams
                    .iter()
                    .map(|b| b.score(cfg.length_penalty))
                    .fold(f64::NEG_INFINITY, f64::max);

                if best_completed_score >= best_live_score {
                    steps = step + 1;
                    break;
                }
            }
        }

        // Gather all remaining live beams as completed
        for b in beams {
            completed.push(b);
        }

        // Sort by score descending
        completed.sort_by(|a, b| {
            b.score(cfg.length_penalty)
                .partial_cmp(&a.score(cfg.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep at most beam_width results
        completed.truncate(bw);

        let scores: Vec<f64> = completed
            .iter()
            .map(|b| b.score(cfg.length_penalty))
            .collect();
        let sequences: Vec<Vec<u32>> = completed.into_iter().map(|b| b.tokens).collect();

        BeamSearchResult {
            sequences,
            scores,
            num_steps: steps,
        }
    }

    /// Zero out (set to −∞) any token that would create a repeated n-gram.
    ///
    /// For each position in `tokens` where the last `ngram_size - 1` tokens
    /// match the trailing `ngram_size - 1` tokens of the current sequence,
    /// the following token is forbidden.
    pub fn apply_no_repeat_ngram(logits: &mut [f32], tokens: &[u32], ngram_size: usize) {
        if ngram_size == 0 || tokens.len() < ngram_size {
            return;
        }

        // The suffix we want to avoid repeating is the last (ngram_size - 1) tokens
        let prefix_len = ngram_size - 1;
        let suffix = &tokens[tokens.len() - prefix_len..];

        // Scan all valid n-gram starting positions in the existing token sequence
        for start in 0..tokens.len().saturating_sub(prefix_len) {
            let window = &tokens[start..start + prefix_len];
            if window == suffix {
                // The token that would complete the n-gram is at `start + prefix_len`
                let banned_token = tokens[start + prefix_len] as usize;
                if banned_token < logits.len() {
                    logits[banned_token] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Return the top-`k` `(token_id, log_prob)` pairs from a logit vector.
    ///
    /// Logits are converted to log-probabilities via log-softmax.
    pub fn top_k_log_probs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Numerical stability: subtract max before exp
        let max_logit = logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute log-softmax: log_prob_i = logit_i - max - log(sum(exp(logit_j - max)))
        let shifted: Vec<f32> = logits
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    v - max_logit
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();

        let log_sum_exp = shifted.iter().copied().map(|v| v.exp()).sum::<f32>().ln();

        let mut indexed: Vec<(u32, f64)> = shifted
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, (v - log_sum_exp) as f64))
            .collect();

        // Sort by log-prob descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Keep the top `beam_width` beams by length-normalised score.
    pub fn prune_beams(mut beams: Vec<Beam>, beam_width: usize, length_penalty: f32) -> Vec<Beam> {
        beams.sort_by(|a, b| {
            b.score(length_penalty)
                .partial_cmp(&a.score(length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beams.truncate(beam_width);
        beams
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Beam unit tests ────────────────────────────────────────────────────

    #[test]
    fn test_beam_new_initial() {
        let tokens = vec![1u32, 2, 3];
        let beam = Beam::new(tokens.clone());
        assert_eq!(beam.tokens, tokens);
        assert!((beam.log_prob - 0.0).abs() < f64::EPSILON);
        assert!(!beam.is_done);
        assert_eq!(beam.len(), 3);
        assert!(!beam.is_empty());
    }

    #[test]
    fn test_beam_score_length_penalty() {
        let beam = Beam {
            tokens: vec![1, 2, 3, 4],
            log_prob: -4.0,
            is_done: false,
        };
        // score = -4.0 / 4^0.6
        let expected = -4.0_f64 / (4.0_f64.powf(0.6));
        let score = beam.score(0.6);
        assert!(
            (score - expected).abs() < 1e-6,
            "score={score}, expected={expected}"
        );
    }

    #[test]
    fn test_beam_score_zero_length() {
        // An empty beam should not panic — treated as length 1
        let beam = Beam {
            tokens: vec![],
            log_prob: -1.0,
            is_done: false,
        };
        let score = beam.score(0.6);
        assert!((score - -1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_beam_extend() {
        let beam = Beam {
            tokens: vec![1, 2],
            log_prob: -1.5,
            is_done: false,
        };
        let extended = beam.extend(3, -0.5);
        assert_eq!(extended.tokens, vec![1, 2, 3]);
        assert!((extended.log_prob - -2.0).abs() < 1e-10);
        assert!(!extended.is_done);
    }

    // ── top_k_log_probs tests ──────────────────────────────────────────────

    #[test]
    fn test_top_k_log_probs_returns_k_best() {
        // logits with clear winner at index 3
        let logits = vec![0.0f32, 1.0, 2.0, 10.0, 0.5];
        let result = BeamSearchEngine::top_k_log_probs(&logits, 2);
        assert_eq!(result.len(), 2);
        // Best token should be index 3
        assert_eq!(result[0].0, 3);
        // Log-probs should be in descending order
        assert!(result[0].1 >= result[1].1);
    }

    #[test]
    fn test_top_k_log_probs_k_larger_than_vocab() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let result = BeamSearchEngine::top_k_log_probs(&logits, 10);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_top_k_log_probs_empty() {
        let result = BeamSearchEngine::top_k_log_probs(&[], 4);
        assert!(result.is_empty());
    }

    // ── prune_beams tests ──────────────────────────────────────────────────

    #[test]
    fn test_prune_beams_keeps_best() {
        let beams = vec![
            Beam {
                tokens: vec![1],
                log_prob: -10.0,
                is_done: false,
            },
            Beam {
                tokens: vec![2],
                log_prob: -1.0,
                is_done: false,
            },
            Beam {
                tokens: vec![3],
                log_prob: -5.0,
                is_done: false,
            },
            Beam {
                tokens: vec![4],
                log_prob: -2.0,
                is_done: false,
            },
        ];
        let pruned = BeamSearchEngine::prune_beams(beams, 2, 1.0);
        assert_eq!(pruned.len(), 2);
        // Best beam has log_prob = -1.0 → tokens = [2]
        assert_eq!(pruned[0].tokens, vec![2]);
        // Second-best has log_prob = -2.0 → tokens = [4]
        assert_eq!(pruned[1].tokens, vec![4]);
    }

    #[test]
    fn test_prune_beams_fewer_than_width() {
        let beams = vec![Beam {
            tokens: vec![1],
            log_prob: -3.0,
            is_done: false,
        }];
        let pruned = BeamSearchEngine::prune_beams(beams, 4, 0.6);
        assert_eq!(pruned.len(), 1);
    }

    // ── apply_no_repeat_ngram tests ───────────────────────────────────────

    #[test]
    fn test_apply_no_repeat_ngram_blocks_repeated() {
        // tokens = [1, 2, 3]; ngram_size = 2 → last prefix is [3]
        // If [3] appeared before at position 1 (tokens[1]=2≠3), skip.
        // If [3] appeared before at position 2 (tokens[2]=3), following token is tokens[3] — but
        // tokens only has length 3, so that would be out of bounds. Let's use a longer sequence.
        //
        // tokens = [1, 2, 1, 2]; ngram_size = 2 → suffix = [2]
        // position 1: tokens[1]=2 matches; next token = tokens[2]=1 → ban token 1
        let tokens = vec![1u32, 2, 1, 2];
        let mut logits = vec![0.0f32; 5];
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 2);
        assert_eq!(logits[1], f32::NEG_INFINITY, "token 1 should be banned");
        // token 2 not yet banned (the last occurrence of [2] is at the very end,
        // no following token exists in history)
        assert!(logits[2].is_finite());
    }

    #[test]
    fn test_no_repeat_ngram_no_effect_when_disabled() {
        let tokens = vec![1u32, 2, 1, 2];
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut logits = original.clone();
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 0);
        assert_eq!(
            logits, original,
            "ngram_size=0 should leave logits unchanged"
        );
    }

    #[test]
    fn test_no_repeat_ngram_too_short_sequence() {
        // Sequence shorter than ngram_size → no banning
        let tokens = vec![1u32];
        let mut logits = vec![1.0f32; 5];
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 3);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    // ── BeamSearchEngine::search integration tests ────────────────────────

    #[test]
    fn test_beam_search_greedy_equivalent_width1() {
        // With beam_width=1 and greedy logits, beam search is equivalent to greedy decoding.
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 5,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 99, // Never generated
        };
        let engine = BeamSearchEngine::new(config);

        // Always return token 7 as the best
        let result = engine.search(vec![0u32], 10, |_tokens, _step| {
            let mut logits = vec![0.0f32; 10];
            logits[7] = 100.0;
            logits
        });

        assert_eq!(result.num_steps, 5);
        let best = result.best();
        // First token is initial (0), remaining should all be 7
        assert!(best.iter().skip(1).all(|&t| t == 7));
    }

    #[test]
    fn test_beam_search_with_eos() {
        // Beam search should stop early when EOS is generated (early_stopping=true).
        let eos = 3u32;
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 20,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: eos,
        };
        let engine = BeamSearchEngine::new(config);

        let step_counter = std::cell::Cell::new(0usize);
        let result = engine.search(vec![1u32], 5, |_tokens, _step| {
            step_counter.set(step_counter.get() + 1);
            // After 2 calls produce EOS as best token
            let mut logits = vec![0.0f32; 5];
            if step_counter.get() >= 2 {
                logits[eos as usize] = 100.0;
            } else {
                logits[1] = 5.0;
            }
            logits
        });

        // Should not have run all 20 steps
        assert!(
            result.num_steps < 20,
            "expected early stop, got {} steps",
            result.num_steps
        );
        assert!(!result.sequences.is_empty());
    }

    #[test]
    fn test_beam_search_result_best() {
        let result = BeamSearchResult {
            sequences: vec![vec![1, 2, 3], vec![4, 5, 6]],
            scores: vec![-0.5, -1.0],
            num_steps: 3,
        };
        assert_eq!(result.best(), &[1, 2, 3]);
        assert!((result.best_score() - -0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beam_search_result_empty() {
        let result = BeamSearchResult {
            sequences: vec![],
            scores: vec![],
            num_steps: 0,
        };
        assert_eq!(result.best(), &[] as &[u32]);
        assert_eq!(result.best_score(), f64::NEG_INFINITY);
    }
}
