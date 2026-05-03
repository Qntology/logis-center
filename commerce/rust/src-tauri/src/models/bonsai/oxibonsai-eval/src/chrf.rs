//! chrF and chrF++ — character n-gram F-score (Popović 2015).
//!
//! chrF computes the F_β (default β=2) score over character n-grams from
//! n=1 up to n=`order` (default 6). chrF++ additionally mixes in word
//! n-gram F-scores up to `word_order` (typical 2).
//!
//! All iteration is over Unicode `char`s — byte slicing is *not* used, so
//! multi-byte UTF-8 sequences are handled correctly.
//!
//! ## Final score
//!
//! For each order `n ∈ 1..=N`:
//! ```text
//! p_n = |cand ∩ ref| / |cand|
//! r_n = |cand ∩ ref| / |ref|
//! ```
//! (multiset intersection over character / word n-grams).
//!
//! Averages `P = (1/N) Σ p_n`, `R = (1/N) Σ r_n`, combined via:
//! ```text
//! F_β = (1 + β²) · P · R / (β² · P + R)
//! ```
//!
//! For chrF++, character and word contributions are averaged with equal weight
//! over their respective orders, following the reference implementation.
//!
//! Empty candidate *and* empty reference → score = 1.0.
//! Empty candidate *or* empty reference (not both) → score = 0.0.

use std::collections::HashMap;

/// Character / word n-gram F-score result.
#[derive(Debug, Clone)]
pub struct ChrfScore {
    /// Final F-score in `[0, 1]`.
    pub score: f32,
    /// Character n-gram order used.
    pub order: usize,
    /// β for the F-score (β>1 weights recall).
    pub beta: f32,
    /// Word n-gram order (0 = chrF, >=1 = chrF++).
    pub word_order: usize,
}

/// chrF (character-n-gram F-score) default: order=6, β=2.
pub fn chrf(candidate: &str, reference: &str) -> ChrfScore {
    chrf_with(candidate, reference, 6, 2.0, 0)
}

/// chrF++ convenience: char order=6, word order=2, β=2.
pub fn chrf_plus_plus(candidate: &str, reference: &str) -> ChrfScore {
    chrf_with(candidate, reference, 6, 2.0, 2)
}

/// chrF / chrF++ with explicit parameters.
///
/// - `order`: maximum character n-gram order (≥ 1).
/// - `beta`: β for F-score (typical 2.0).
/// - `word_order`: 0 → chrF; ≥1 → mix in word n-grams up to this order (chrF++).
pub fn chrf_with(
    candidate: &str,
    reference: &str,
    order: usize,
    beta: f32,
    word_order: usize,
) -> ChrfScore {
    let order = order.max(1);
    let cand_chars: Vec<char> = candidate.chars().collect();
    let ref_chars: Vec<char> = reference.chars().collect();

    // Handle edge cases: both empty → perfect, either empty → 0.
    let both_empty = cand_chars.is_empty() && ref_chars.is_empty();
    let one_empty = cand_chars.is_empty() ^ ref_chars.is_empty();
    if both_empty {
        return ChrfScore {
            score: 1.0,
            order,
            beta,
            word_order,
        };
    }
    if one_empty {
        return ChrfScore {
            score: 0.0,
            order,
            beta,
            word_order,
        };
    }

    let cand_words: Vec<&str> = candidate.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    // Collect per-order F-beta into a single averaged score.
    let mut f_values: Vec<f32> = Vec::new();

    // Character orders
    for n in 1..=order {
        if let Some(f) = order_f_beta_chars(&cand_chars, &ref_chars, n, beta) {
            f_values.push(f);
        } else {
            f_values.push(0.0);
        }
    }

    // Word orders (chrF++)
    if word_order >= 1 {
        for n in 1..=word_order {
            if let Some(f) = order_f_beta_words(&cand_words, &ref_words, n, beta) {
                f_values.push(f);
            } else {
                f_values.push(0.0);
            }
        }
    }

    let score = if f_values.is_empty() {
        0.0
    } else {
        let sum: f32 = f_values.iter().sum();
        sum / f_values.len() as f32
    };

    ChrfScore {
        score: score.clamp(0.0, 1.0),
        order,
        beta,
        word_order,
    }
}

fn order_f_beta_chars(cand: &[char], reference: &[char], n: usize, beta: f32) -> Option<f32> {
    if cand.len() < n || reference.len() < n {
        return None;
    }
    let cand_counts = ngram_counts_char(cand, n);
    let ref_counts = ngram_counts_char(reference, n);
    Some(f_beta_from_counts(&cand_counts, &ref_counts, beta))
}

fn order_f_beta_words(cand: &[&str], reference: &[&str], n: usize, beta: f32) -> Option<f32> {
    if cand.len() < n || reference.len() < n {
        return None;
    }
    let cand_counts = ngram_counts_words(cand, n);
    let ref_counts = ngram_counts_words(reference, n);
    Some(f_beta_from_counts(&cand_counts, &ref_counts, beta))
}

fn f_beta_from_counts<K: std::hash::Hash + Eq + Clone>(
    cand: &HashMap<K, usize>,
    reference: &HashMap<K, usize>,
    beta: f32,
) -> f32 {
    let cand_total: usize = cand.values().sum();
    let ref_total: usize = reference.values().sum();
    if cand_total == 0 || ref_total == 0 {
        return 0.0;
    }
    let mut overlap = 0usize;
    for (k, &v) in cand {
        if let Some(&rv) = reference.get(k) {
            overlap += v.min(rv);
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let p = overlap as f32 / cand_total as f32;
    let r = overlap as f32 / ref_total as f32;
    let b2 = beta * beta;
    let denom = b2 * p + r;
    if denom <= 0.0 {
        0.0
    } else {
        ((1.0 + b2) * p * r) / denom
    }
}

fn ngram_counts_char(chars: &[char], n: usize) -> HashMap<Vec<char>, usize> {
    let mut counts: HashMap<Vec<char>, usize> = HashMap::new();
    if n == 0 || chars.len() < n {
        return counts;
    }
    for w in chars.windows(n) {
        *counts.entry(w.to_vec()).or_insert(0) += 1;
    }
    counts
}

fn ngram_counts_words(words: &[&str], n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    if n == 0 || words.len() < n {
        return counts;
    }
    for w in words.windows(n) {
        let key: Vec<String> = w.iter().map(|s| s.to_string()).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}
