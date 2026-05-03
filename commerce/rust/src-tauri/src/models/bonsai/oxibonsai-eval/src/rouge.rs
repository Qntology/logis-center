//! ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics.
//!
//! ROUGE-N: n-gram recall/precision/F1 between summary and reference.
//! ROUGE-L: Longest Common Subsequence (LCS) based F1.
//! ROUGE-S: Skip-bigram overlap.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A token sequence (words or characters, caller's choice).
pub type TokenSeq = Vec<String>;

// ──────────────────────────────────────────────────────────────────────────────
// Tokenization
// ──────────────────────────────────────────────────────────────────────────────

/// Tokenize a string into words (lowercase, split on whitespace/punctuation).
///
/// Splits on whitespace, then strips leading/trailing punctuation from each
/// token and lowercases.  Empty tokens are discarded.
pub fn tokenize(text: &str) -> TokenSeq {
    text.split_whitespace()
        .filter_map(|word| {
            // Strip non-alphanumeric characters from both ends.
            let stripped: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            let lower = stripped.to_lowercase();
            if lower.is_empty() {
                None
            } else {
                Some(lower)
            }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// N-gram helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute n-gram counts from a token sequence.
///
/// Returns a map from (n-gram as `Vec<String>`) to its occurrence count.
pub fn ngram_counts(tokens: &TokenSeq, n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    if n == 0 || tokens.len() < n {
        return counts;
    }
    for window in tokens.windows(n) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts
}

/// Compute the clipped n-gram overlap (ROUGE numerator).
///
/// For each n-gram, the overlap count is `min(count_in_candidate, count_in_reference)`.
fn clipped_ngram_overlap(
    cand_counts: &HashMap<Vec<String>, usize>,
    ref_counts: &HashMap<Vec<String>, usize>,
) -> usize {
    let mut overlap = 0usize;
    for (ngram, &cand_count) in cand_counts {
        if let Some(&ref_count) = ref_counts.get(ngram) {
            overlap += cand_count.min(ref_count);
        }
    }
    overlap
}

// ──────────────────────────────────────────────────────────────────────────────
// RougeNScore
// ──────────────────────────────────────────────────────────────────────────────

/// ROUGE-N scores for one (candidate, reference) pair.
#[derive(Debug, Clone)]
pub struct RougeNScore {
    pub n: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

impl RougeNScore {
    /// Compute ROUGE-N between a candidate and a reference text.
    ///
    /// Returns precision, recall, and F1. When either sequence is empty for
    /// the given n, all scores are 0.0.
    pub fn compute(candidate: &str, reference: &str, n: usize) -> Self {
        let cand_tokens = tokenize(candidate);
        let ref_tokens = tokenize(reference);
        Self::from_tokens(&cand_tokens, &ref_tokens, n)
    }

    /// Compute ROUGE-N for a candidate against multiple references (max recall).
    ///
    /// Each reference is scored independently; the one yielding the highest
    /// recall is selected (standard multi-reference ROUGE protocol).
    pub fn compute_multi_ref(candidate: &str, references: &[&str], n: usize) -> Self {
        if references.is_empty() {
            return Self {
                n,
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            };
        }
        let cand_tokens = tokenize(candidate);
        references
            .iter()
            .map(|r| {
                let ref_tokens = tokenize(r);
                Self::from_tokens(&cand_tokens, &ref_tokens, n)
            })
            .max_by(|a, b| {
                a.recall
                    .partial_cmp(&b.recall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(Self {
                n,
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            })
    }

    /// Internal: compute from pre-tokenized sequences.
    fn from_tokens(cand_tokens: &TokenSeq, ref_tokens: &TokenSeq, n: usize) -> Self {
        let cand_counts = ngram_counts(cand_tokens, n);
        let ref_counts = ngram_counts(ref_tokens, n);

        let cand_total: usize = cand_counts.values().sum();
        let ref_total: usize = ref_counts.values().sum();
        let overlap = clipped_ngram_overlap(&cand_counts, &ref_counts);

        let precision = if cand_total == 0 {
            0.0
        } else {
            overlap as f32 / cand_total as f32
        };
        let recall = if ref_total == 0 {
            0.0
        } else {
            overlap as f32 / ref_total as f32
        };
        let f1 = f1_score(precision, recall);

        Self {
            n,
            precision,
            recall,
            f1,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RougeLScore
// ──────────────────────────────────────────────────────────────────────────────

/// ROUGE-L: Longest Common Subsequence based score.
#[derive(Debug, Clone)]
pub struct RougeLScore {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub lcs_length: usize,
}

impl RougeLScore {
    /// Compute ROUGE-L between candidate and reference text.
    pub fn compute(candidate: &str, reference: &str) -> Self {
        let cand_tokens = tokenize(candidate);
        let ref_tokens = tokenize(reference);

        let lcs_len = Self::lcs_length(&cand_tokens, &ref_tokens);
        let cand_len = cand_tokens.len();
        let ref_len = ref_tokens.len();

        let precision = if cand_len == 0 {
            0.0
        } else {
            lcs_len as f32 / cand_len as f32
        };
        let recall = if ref_len == 0 {
            0.0
        } else {
            lcs_len as f32 / ref_len as f32
        };
        let f1 = f1_score(precision, recall);

        Self {
            precision,
            recall,
            f1,
            lcs_length: lcs_len,
        }
    }

    /// LCS length between two token sequences (O(n*m) dynamic programming).
    ///
    /// Allocates a `Vec<Vec<usize>>` DP table of size `(a.len()+1) x (b.len()+1)`.
    pub fn lcs_length(a: &TokenSeq, b: &TokenSeq) -> usize {
        let m = a.len();
        let n = b.len();
        if m == 0 || n == 0 {
            return 0;
        }

        // Allocate full DP table: dp[i][j] = LCS length of a[..i] and b[..j]
        let mut dp: Vec<Vec<usize>> = vec![vec![0usize; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RougeSScore
// ──────────────────────────────────────────────────────────────────────────────

/// ROUGE-S: Skip-bigram overlap.
///
/// A skip-bigram is any pair of words in sentence order, allowing arbitrary
/// gaps between them.  Scores are precision, recall, and F1 of the clipped
/// skip-bigram overlap between candidate and reference.
#[derive(Debug, Clone)]
pub struct RougeSScore {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

impl RougeSScore {
    /// Compute ROUGE-S between candidate and reference text.
    pub fn compute(candidate: &str, reference: &str) -> Self {
        let cand_tokens = tokenize(candidate);
        let ref_tokens = tokenize(reference);

        let cand_bigrams = Self::skip_bigrams(&cand_tokens);
        let ref_bigrams = Self::skip_bigrams(&ref_tokens);

        // Count occurrences of each skip-bigram.
        let cand_counts = bigram_counts(&cand_bigrams);
        let ref_counts = bigram_counts(&ref_bigrams);

        let cand_total = cand_bigrams.len();
        let ref_total = ref_bigrams.len();

        // Clipped overlap.
        let overlap: usize = cand_counts
            .iter()
            .map(|(bg, &cc)| {
                let rc = ref_counts.get(bg).copied().unwrap_or(0);
                cc.min(rc)
            })
            .sum();

        let precision = if cand_total == 0 {
            0.0
        } else {
            overlap as f32 / cand_total as f32
        };
        let recall = if ref_total == 0 {
            0.0
        } else {
            overlap as f32 / ref_total as f32
        };
        let f1 = f1_score(precision, recall);

        Self {
            precision,
            recall,
            f1,
        }
    }

    /// Generate all skip-bigrams (ordered pairs) from a token sequence.
    ///
    /// For a sequence of length n, produces n*(n-1)/2 pairs.
    fn skip_bigrams(tokens: &TokenSeq) -> Vec<(String, String)> {
        let mut bigrams = Vec::new();
        let n = tokens.len();
        for i in 0..n {
            for j in (i + 1)..n {
                bigrams.push((tokens[i].clone(), tokens[j].clone()));
            }
        }
        bigrams
    }
}

/// Count occurrences of each skip-bigram in a slice.
fn bigram_counts(bigrams: &[(String, String)]) -> HashMap<(String, String), usize> {
    let mut counts = HashMap::new();
    for bg in bigrams {
        *counts.entry(bg.clone()).or_insert(0) += 1;
    }
    counts
}

// ──────────────────────────────────────────────────────────────────────────────
// CorpusRouge
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate ROUGE scores across a corpus.
///
/// Averages ROUGE-1, ROUGE-2, and ROUGE-L F1 scores over all (candidate,
/// reference) pairs in the corpus.
#[derive(Debug, Clone, Default)]
pub struct CorpusRouge {
    pub rouge_1: Option<RougeNScore>,
    pub rouge_2: Option<RougeNScore>,
    pub rouge_l: Option<RougeLScore>,
    pub num_samples: usize,
}

impl CorpusRouge {
    /// Compute aggregate ROUGE scores across (candidate, reference) pairs.
    ///
    /// Each field of the result contains macro-averaged scores (average of
    /// per-pair scores) over all pairs in `pairs`.
    pub fn compute(pairs: &[(&str, &str)]) -> Self {
        if pairs.is_empty() {
            return Self::default();
        }

        let n = pairs.len();
        let mut sum_r1 = RougeAccum::default();
        let mut sum_r2 = RougeAccum::default();
        let mut sum_rl_p = 0.0f64;
        let mut sum_rl_r = 0.0f64;
        let mut sum_rl_f1 = 0.0f64;
        let mut sum_rl_lcs = 0usize;

        for &(cand, reference) in pairs {
            let r1 = RougeNScore::compute(cand, reference, 1);
            let r2 = RougeNScore::compute(cand, reference, 2);
            let rl = RougeLScore::compute(cand, reference);

            sum_r1.add(&r1);
            sum_r2.add(&r2);
            sum_rl_p += f64::from(rl.precision);
            sum_rl_r += f64::from(rl.recall);
            sum_rl_f1 += f64::from(rl.f1);
            sum_rl_lcs += rl.lcs_length;
        }

        let nf = n as f64;
        let rouge_1 = Some(RougeNScore {
            n: 1,
            precision: (sum_r1.precision / nf) as f32,
            recall: (sum_r1.recall / nf) as f32,
            f1: (sum_r1.f1 / nf) as f32,
        });
        let rouge_2 = Some(RougeNScore {
            n: 2,
            precision: (sum_r2.precision / nf) as f32,
            recall: (sum_r2.recall / nf) as f32,
            f1: (sum_r2.f1 / nf) as f32,
        });
        let rouge_l = Some(RougeLScore {
            precision: (sum_rl_p / nf) as f32,
            recall: (sum_rl_r / nf) as f32,
            f1: (sum_rl_f1 / nf) as f32,
            lcs_length: sum_rl_lcs / n, // integer average
        });

        Self {
            rouge_1,
            rouge_2,
            rouge_l,
            num_samples: n,
        }
    }

    /// Return a human-readable summary of the corpus-level scores.
    pub fn summary(&self) -> String {
        let r1_f1 = self.rouge_1.as_ref().map_or(0.0, |s| s.f1);
        let r2_f1 = self.rouge_2.as_ref().map_or(0.0, |s| s.f1);
        let rl_f1 = self.rouge_l.as_ref().map_or(0.0, |s| s.f1);
        format!(
            "CorpusROUGE(n={}) ROUGE-1 F1={:.4} | ROUGE-2 F1={:.4} | ROUGE-L F1={:.4}",
            self.num_samples, r1_f1, r2_f1, rl_f1,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulates precision/recall/F1 sums (as f64 to avoid precision loss).
#[derive(Default)]
struct RougeAccum {
    precision: f64,
    recall: f64,
    f1: f64,
}

impl RougeAccum {
    fn add(&mut self, score: &RougeNScore) {
        self.precision += f64::from(score.precision);
        self.recall += f64::from(score.recall);
        self.f1 += f64::from(score.f1);
    }
}

/// Compute F1 from precision and recall.
#[inline]
fn f1_score(precision: f32, recall: f32) -> f32 {
    let denom = precision + recall;
    if denom == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / denom
    }
}
