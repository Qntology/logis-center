//! Generation quality metrics for evaluating LLM outputs.
//!
//! Metrics:
//! - N-gram repetition rate: fraction of n-grams that are repeated
//! - Distinct-N: fraction of unique n-grams (diversity)
//! - Self-BLEU: similarity between generated samples (diversity)
//! - Perplexity proxy: entropy of token distribution
//! - Repetition penalty score: how much repetition penalty is triggered

use std::collections::{HashMap, HashSet};

/// N-gram type (sequence of token IDs).
pub type NGram = Vec<u32>;

/// Compute all n-grams of size `n` from a token sequence.
pub fn extract_ngrams(tokens: &[u32], n: usize) -> Vec<NGram> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}

/// Repetition metrics for a single generated sequence.
#[derive(Debug, Clone)]
pub struct RepetitionMetrics {
    pub token_count: usize,
    /// Fraction of tokens that appeared before (at their position).
    pub repeated_unigrams: f32,
    /// Fraction of bigrams that appeared before.
    pub repeated_bigrams: f32,
    /// Fraction of trigrams that appeared before.
    pub repeated_trigrams: f32,
    /// Longest run of identical consecutive tokens.
    pub max_consecutive_repeat: usize,
    pub unique_tokens: usize,
}

impl RepetitionMetrics {
    /// Compute repetition metrics from a token sequence.
    pub fn compute(tokens: &[u32]) -> Self {
        let token_count = tokens.len();

        if token_count == 0 {
            return Self {
                token_count: 0,
                repeated_unigrams: 0.0,
                repeated_bigrams: 0.0,
                repeated_trigrams: 0.0,
                max_consecutive_repeat: 0,
                unique_tokens: 0,
            };
        }

        // Unigram repetition: for each position i>0, has token[i] appeared in tokens[0..i]?
        let mut seen: HashSet<u32> = HashSet::new();
        let mut repeated_uni_count = 0usize;
        for &tok in tokens {
            if seen.contains(&tok) {
                repeated_uni_count += 1;
            }
            seen.insert(tok);
        }
        let unique_tokens = seen.len();
        let repeated_unigrams = if token_count > 1 {
            repeated_uni_count as f32 / (token_count - 1) as f32
        } else {
            0.0
        };

        // Bigram repetition
        let repeated_bigrams = compute_ngram_repetition_rate(tokens, 2);

        // Trigram repetition
        let repeated_trigrams = compute_ngram_repetition_rate(tokens, 3);

        // Max consecutive repeat
        let max_consecutive_repeat = compute_max_consecutive(tokens);

        Self {
            token_count,
            repeated_unigrams,
            repeated_bigrams,
            repeated_trigrams,
            max_consecutive_repeat,
            unique_tokens,
        }
    }

    /// Returns true if repeated_unigrams > 0.7 (degenerate generation).
    pub fn is_degenerate(&self) -> bool {
        self.repeated_unigrams > 0.7
    }

    /// Human-readable summary of repetition metrics.
    pub fn summary(&self) -> String {
        format!(
            "RepetitionMetrics {{ tokens={}, unique={}, rep1={:.3}, rep2={:.3}, rep3={:.3}, max_consec={}, degenerate={} }}",
            self.token_count,
            self.unique_tokens,
            self.repeated_unigrams,
            self.repeated_bigrams,
            self.repeated_trigrams,
            self.max_consecutive_repeat,
            self.is_degenerate(),
        )
    }
}

/// Fraction of n-grams (n>=2) that have appeared before at their position.
fn compute_ngram_repetition_rate(tokens: &[u32], n: usize) -> f32 {
    if tokens.len() < n {
        return 0.0;
    }
    let ngrams = extract_ngrams(tokens, n);
    let total = ngrams.len();
    if total == 0 {
        return 0.0;
    }
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut repeated = 0usize;
    for gram in &ngrams {
        if seen.contains(gram) {
            repeated += 1;
        }
        seen.insert(gram.clone());
    }
    repeated as f32 / total as f32
}

/// Longest consecutive run of the same token value.
fn compute_max_consecutive(tokens: &[u32]) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let mut max_run = 1usize;
    let mut current_run = 1usize;
    for i in 1..tokens.len() {
        if tokens[i] == tokens[i - 1] {
            current_run += 1;
            if current_run > max_run {
                max_run = current_run;
            }
        } else {
            current_run = 1;
        }
    }
    max_run
}

/// Diversity metrics (Distinct-N).
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// |unique unigrams| / |all unigrams|
    pub distinct_1: f32,
    /// |unique bigrams| / |all bigrams|
    pub distinct_2: f32,
    /// |unique trigrams| / |all trigrams|
    pub distinct_3: f32,
    /// Unique tokens / total tokens (same as distinct_1).
    pub vocab_coverage: f32,
    pub token_count: usize,
}

impl DiversityMetrics {
    /// Compute diversity metrics from a token sequence.
    pub fn compute(tokens: &[u32]) -> Self {
        let token_count = tokens.len();

        if token_count == 0 {
            return Self {
                distinct_1: 0.0,
                distinct_2: 0.0,
                distinct_3: 0.0,
                vocab_coverage: 0.0,
                token_count: 0,
            };
        }

        let distinct_1 = distinct_n(tokens, 1);
        let distinct_2 = distinct_n(tokens, 2);
        let distinct_3 = distinct_n(tokens, 3);

        Self {
            distinct_1,
            distinct_2,
            distinct_3,
            vocab_coverage: distinct_1,
            token_count,
        }
    }

    /// Average of distinct_1, distinct_2, distinct_3.
    pub fn overall_diversity(&self) -> f32 {
        (self.distinct_1 + self.distinct_2 + self.distinct_3) / 3.0
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "DiversityMetrics {{ tokens={}, D1={:.3}, D2={:.3}, D3={:.3}, overall={:.3} }}",
            self.token_count,
            self.distinct_1,
            self.distinct_2,
            self.distinct_3,
            self.overall_diversity(),
        )
    }
}

/// Fraction of unique n-grams over all n-grams.
fn distinct_n(tokens: &[u32], n: usize) -> f32 {
    let ngrams = extract_ngrams(tokens, n);
    let total = ngrams.len();
    if total == 0 {
        return 0.0;
    }
    let unique: HashSet<Vec<u32>> = ngrams.into_iter().collect();
    unique.len() as f32 / total as f32
}

/// BLEU-related scoring (simplified, for reference/diversity measurement).
///
/// Implements clipped n-gram precision between a candidate and reference.
#[derive(Debug, Clone)]
pub struct BleuScore {
    pub precision_1: f32,
    pub precision_2: f32,
    pub precision_3: f32,
    pub precision_4: f32,
    pub brevity_penalty: f32,
    /// Geometric mean of precisions * brevity_penalty.
    pub bleu: f32,
}

impl BleuScore {
    /// Compute BLEU between a candidate and a reference (as token ID sequences).
    pub fn compute(candidate: &[u32], reference: &[u32]) -> Self {
        let bp = brevity_penalty(candidate.len(), reference.len());

        let p1 = clipped_precision(candidate, reference, 1);
        let p2 = clipped_precision(candidate, reference, 2);
        let p3 = clipped_precision(candidate, reference, 3);
        let p4 = clipped_precision(candidate, reference, 4);

        let bleu = geometric_mean_bleu([p1, p2, p3, p4], bp);

        Self {
            precision_1: p1,
            precision_2: p2,
            precision_3: p3,
            precision_4: p4,
            brevity_penalty: bp,
            bleu,
        }
    }

    /// Corpus-level BLEU across multiple (candidate, reference) pairs.
    pub fn corpus_bleu(pairs: &[(&[u32], &[u32])]) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }

        // Accumulate clipped counts and total counts per n-gram order
        let mut clipped_counts = [0usize; 4];
        let mut total_counts = [0usize; 4];
        let mut cand_len_total = 0usize;
        let mut ref_len_total = 0usize;

        for (candidate, reference) in pairs {
            cand_len_total += candidate.len();
            ref_len_total += reference.len();

            for n in 1..=4usize {
                let (clipped, total) = clipped_count_raw(candidate, reference, n);
                clipped_counts[n - 1] += clipped;
                total_counts[n - 1] += total;
            }
        }

        let bp = brevity_penalty(cand_len_total, ref_len_total);

        let mut log_sum = 0.0f32;
        let mut valid = 0usize;
        for n in 0..4 {
            if total_counts[n] == 0 {
                continue;
            }
            let p = clipped_counts[n] as f32 / total_counts[n] as f32;
            if p > 0.0 {
                log_sum += p.ln();
                valid += 1;
            } else {
                // Zero precision for any order => BLEU = 0
                return 0.0;
            }
        }

        if valid == 0 {
            return 0.0;
        }

        bp * (log_sum / valid as f32).exp()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "BleuScore {{ P1={:.3}, P2={:.3}, P3={:.3}, P4={:.3}, BP={:.3}, BLEU={:.4} }}",
            self.precision_1,
            self.precision_2,
            self.precision_3,
            self.precision_4,
            self.brevity_penalty,
            self.bleu,
        )
    }
}

/// BP = 1 if |cand| >= |ref|, else exp(1 - |ref|/|cand|).
fn brevity_penalty(cand_len: usize, ref_len: usize) -> f32 {
    if cand_len == 0 {
        return 0.0;
    }
    if cand_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len as f32 / cand_len as f32).exp()
    }
}

/// Clipped n-gram precision: min(count_in_cand, max_count_in_ref) / count_in_cand_total.
fn clipped_precision(candidate: &[u32], reference: &[u32], n: usize) -> f32 {
    let (clipped, total) = clipped_count_raw(candidate, reference, n);
    if total == 0 {
        return 0.0;
    }
    clipped as f32 / total as f32
}

/// Returns (clipped_matches, total_candidate_ngrams).
fn clipped_count_raw(candidate: &[u32], reference: &[u32], n: usize) -> (usize, usize) {
    let cand_ngrams = extract_ngrams(candidate, n);
    let ref_ngrams = extract_ngrams(reference, n);

    let total = cand_ngrams.len();
    if total == 0 {
        return (0, 0);
    }

    // Build reference ngram counts
    let mut ref_counts: HashMap<Vec<u32>, usize> = HashMap::new();
    for g in &ref_ngrams {
        *ref_counts.entry(g.clone()).or_insert(0) += 1;
    }

    // Build candidate ngram counts
    let mut cand_counts: HashMap<Vec<u32>, usize> = HashMap::new();
    for g in &cand_ngrams {
        *cand_counts.entry(g.clone()).or_insert(0) += 1;
    }

    // Clipped count: min(cand_count, ref_count) for each ngram
    let mut clipped = 0usize;
    for (gram, &cand_c) in &cand_counts {
        let ref_c = ref_counts.get(gram).copied().unwrap_or(0);
        clipped += cand_c.min(ref_c);
    }

    (clipped, total)
}

/// Geometric mean BLEU from 4 precisions and brevity penalty.
/// Uses log-domain to avoid underflow. Zero precision in any slot → 0.
fn geometric_mean_bleu(precisions: [f32; 4], bp: f32) -> f32 {
    let mut log_sum = 0.0f32;
    for &p in &precisions {
        if p <= 0.0 {
            return 0.0;
        }
        log_sum += p.ln();
    }
    bp * (log_sum / 4.0).exp()
}

/// Self-BLEU: average BLEU of each sample against all others.
///
/// Lower = more diverse. Used for evaluating sampling diversity.
pub fn self_bleu(samples: &[Vec<u32>]) -> f32 {
    let n = samples.len();
    if n <= 1 {
        return 0.0;
    }

    let mut total_bleu = 0.0f32;
    let mut count = 0usize;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let score = BleuScore::compute(&samples[i], &samples[j]);
            total_bleu += score.bleu;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_bleu / count as f32
    }
}

/// Token entropy (per-position entropy of token distribution).
///
/// Input: `logits` — raw logit scores for vocabulary tokens.
/// Returns Shannon entropy in bits: H = -Σ p_i * log2(p_i)
pub fn token_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }

    // Numerically stable softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();

    if sum_exp == 0.0 {
        return 0.0;
    }

    let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

    // Shannon entropy in bits
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();

    entropy
}

/// Perplexity proxy from a sequence of per-token log-probabilities.
///
/// PPL = exp(-1/n * Σ log P(t_i))
pub fn perplexity_from_logprobs(log_probs: &[f32]) -> f32 {
    if log_probs.is_empty() {
        return f32::INFINITY;
    }
    let n = log_probs.len() as f32;
    let mean_neg_logprob: f32 = -log_probs.iter().sum::<f32>() / n;
    mean_neg_logprob.exp()
}

/// Repetition penalty trigger rate.
///
/// Given `tokens` (history) and `logits`, compute what fraction
/// of top-k tokens in logits were penalized (i.e., appear in `tokens`).
pub fn repetition_penalty_rate(tokens: &[u32], logits: &[f32], top_k: usize) -> f32 {
    if logits.is_empty() || top_k == 0 {
        return 0.0;
    }

    // Get top-k token indices by logit value
    let k = top_k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    // Sort descending by logit
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k_tokens: HashSet<u32> = indexed.iter().take(k).map(|(idx, _)| *idx as u32).collect();

    let token_set: HashSet<u32> = tokens.iter().cloned().collect();

    let penalized = top_k_tokens.intersection(&token_set).count();
    penalized as f32 / k as f32
}

/// Comprehensive quality report for a single generation.
#[derive(Debug, Clone)]
pub struct GenerationQualityReport {
    pub repetition: RepetitionMetrics,
    pub diversity: DiversityMetrics,
    pub entropy: f32,
    pub is_degenerate: bool,
    /// Composite score: diversity * (1 - repetition_rate).
    pub quality_score: f32,
}

impl GenerationQualityReport {
    /// Compute full quality report from token sequence + optional logits.
    pub fn compute(tokens: &[u32], logits: Option<&[f32]>) -> Self {
        let repetition = RepetitionMetrics::compute(tokens);
        let diversity = DiversityMetrics::compute(tokens);

        let entropy = match logits {
            Some(l) => token_entropy(l),
            None => 0.0,
        };

        let is_degenerate = repetition.is_degenerate();

        // Composite quality: overall diversity * (1 - unigram repetition rate)
        let quality_score = diversity.overall_diversity() * (1.0 - repetition.repeated_unigrams);

        Self {
            repetition,
            diversity,
            entropy,
            is_degenerate,
            quality_score,
        }
    }

    /// Human-readable summary of the full quality report.
    pub fn summary(&self) -> String {
        format!(
            "GenerationQualityReport {{\n  {}\n  {}\n  entropy={:.3}, degenerate={}, quality_score={:.4}\n}}",
            self.repetition.summary(),
            self.diversity.summary(),
            self.entropy,
            self.is_degenerate,
            self.quality_score,
        )
    }
}

/// Batch quality analysis across multiple generations.
pub struct BatchQualityAnalyzer {
    reports: Vec<GenerationQualityReport>,
    /// Raw token sequences kept for self-BLEU computation.
    token_sequences: Vec<Vec<u32>>,
}

impl BatchQualityAnalyzer {
    /// Create an empty analyzer.
    pub fn new() -> Self {
        Self {
            reports: Vec::new(),
            token_sequences: Vec::new(),
        }
    }

    /// Add a pre-computed quality report (token sequence unknown for self-BLEU).
    pub fn add_report(&mut self, report: GenerationQualityReport) {
        self.reports.push(report);
    }

    /// Compute quality report for `tokens` and add it.
    pub fn add_generation(&mut self, tokens: &[u32]) {
        let report = GenerationQualityReport::compute(tokens, None);
        self.reports.push(report);
        self.token_sequences.push(tokens.to_vec());
    }

    /// Mean quality score across all reports; None if empty.
    pub fn mean_quality_score(&self) -> Option<f32> {
        if self.reports.is_empty() {
            return None;
        }
        let sum: f32 = self.reports.iter().map(|r| r.quality_score).sum();
        Some(sum / self.reports.len() as f32)
    }

    /// Mean overall diversity across all reports; None if empty.
    pub fn mean_diversity(&self) -> Option<f32> {
        if self.reports.is_empty() {
            return None;
        }
        let sum: f32 = self
            .reports
            .iter()
            .map(|r| r.diversity.overall_diversity())
            .sum();
        Some(sum / self.reports.len() as f32)
    }

    /// Fraction of reports classified as degenerate.
    pub fn degenerate_fraction(&self) -> f32 {
        if self.reports.is_empty() {
            return 0.0;
        }
        let count = self.reports.iter().filter(|r| r.is_degenerate).count();
        count as f32 / self.reports.len() as f32
    }

    /// Self-BLEU across all stored token sequences (from `add_generation`).
    pub fn self_bleu_score(&self) -> f32 {
        self_bleu(&self.token_sequences)
    }

    /// Number of reports stored.
    pub fn num_reports(&self) -> usize {
        self.reports.len()
    }

    /// Human-readable batch report.
    pub fn report(&self) -> String {
        let mean_q = self
            .mean_quality_score()
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "N/A".to_string());
        let mean_d = self
            .mean_diversity()
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "N/A".to_string());
        format!(
            "BatchQualityAnalyzer {{ n={}, mean_quality={}, mean_diversity={}, degenerate_frac={:.3}, self_bleu={:.4} }}",
            self.num_reports(),
            mean_q,
            mean_d,
            self.degenerate_fraction(),
            self.self_bleu_score(),
        )
    }
}

impl Default for BatchQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn extract_ngrams_unigrams_inline() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let ngrams = extract_ngrams(&tokens, 1);
        assert_eq!(ngrams.len(), 5);
    }

    #[test]
    fn extract_ngrams_bigrams_inline() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let ngrams = extract_ngrams(&tokens, 2);
        assert_eq!(ngrams.len(), 4);
        assert_eq!(ngrams[0], vec![1, 2]);
        assert_eq!(ngrams[3], vec![4, 5]);
    }
}
