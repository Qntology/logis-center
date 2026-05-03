//! SQuAD-style QA evaluation — Exact Match (EM) and token F1.
//!
//! Normalisation follows the official SQuAD 1.1 evaluation script:
//!
//! 1. Lowercase.
//! 2. Remove punctuation (Unicode categories are approximated by
//!    `!is_alphanumeric() && !is_whitespace()`).
//! 3. Remove articles `a`, `an`, `the` (standalone tokens).
//! 4. Collapse consecutive whitespace.
//!
//! Multi-reference: the final score for a prediction is `max` over all
//! reference answers (standard protocol).
//!
//! Empty prediction produces EM=0 and F1=0 unless the reference is also
//! empty, in which case EM=1, F1=1 (consistent with official script).

/// Per-question QA scores.
#[derive(Debug, Clone, Copy)]
pub struct QaScore {
    /// 1.0 for exact match after normalisation, else 0.0.
    pub exact_match: f32,
    /// Token-level F1 after normalisation (in [0, 1]).
    pub f1: f32,
}

/// Normalise an answer string SQuAD-style.
pub fn normalize_answer(s: &str) -> String {
    // 1. lowercase
    let lower = s.to_lowercase();
    // 2. remove punctuation (keep alphanumerics + whitespace)
    let no_punct: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    // 3. remove articles
    let no_articles: Vec<&str> = no_punct
        .split_whitespace()
        .filter(|tok| *tok != "a" && *tok != "an" && *tok != "the")
        .collect();
    // 4. collapse whitespace by joining with single space
    no_articles.join(" ")
}

/// SQuAD tokenisation after normalisation: split on ASCII whitespace.
pub fn normalize_tokens(s: &str) -> Vec<String> {
    normalize_answer(s)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// Compute EM between a single prediction and a single reference.
pub fn exact_match(prediction: &str, reference: &str) -> f32 {
    if normalize_answer(prediction) == normalize_answer(reference) {
        1.0
    } else {
        0.0
    }
}

/// Compute token F1 between a single prediction and a single reference.
pub fn f1_score(prediction: &str, reference: &str) -> f32 {
    let pred_tokens = normalize_tokens(prediction);
    let ref_tokens = normalize_tokens(reference);

    // Handle edge cases per the official SQuAD script: if either is empty,
    // F1 is 1 iff both are empty, otherwise 0.
    if pred_tokens.is_empty() && ref_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let common = common_multiset(&pred_tokens, &ref_tokens);
    if common == 0 {
        return 0.0;
    }
    let precision = common as f32 / pred_tokens.len() as f32;
    let recall = common as f32 / ref_tokens.len() as f32;
    (2.0 * precision * recall) / (precision + recall)
}

/// Compute EM and F1 for a prediction against multiple reference strings; returns max.
pub fn score_multi(prediction: &str, references: &[&str]) -> QaScore {
    if references.is_empty() {
        return QaScore {
            exact_match: 0.0,
            f1: 0.0,
        };
    }
    let mut em = 0.0f32;
    let mut f1 = 0.0f32;
    for r in references {
        em = em.max(exact_match(prediction, r));
        f1 = f1.max(f1_score(prediction, r));
    }
    QaScore {
        exact_match: em,
        f1,
    }
}

/// Average EM and F1 over a list of (prediction, references) pairs.
///
/// Returns `(avg_em, avg_f1)`. Empty input returns `(0.0, 0.0)`.
pub fn corpus_em_f1(examples: &[(String, Vec<String>)]) -> (f32, f32) {
    if examples.is_empty() {
        return (0.0, 0.0);
    }
    let mut em_sum = 0.0f32;
    let mut f1_sum = 0.0f32;
    for (pred, refs) in examples {
        let refs_slice: Vec<&str> = refs.iter().map(String::as_str).collect();
        let s = score_multi(pred, &refs_slice);
        em_sum += s.exact_match;
        f1_sum += s.f1;
    }
    let n = examples.len() as f32;
    (em_sum / n, f1_sum / n)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal
// ──────────────────────────────────────────────────────────────────────────────

fn common_multiset(a: &[String], b: &[String]) -> usize {
    use std::collections::HashMap;
    let mut map: HashMap<&str, i64> = HashMap::new();
    for t in a {
        *map.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut common = 0i64;
    for t in b {
        let entry = map.entry(t.as_str()).or_insert(0);
        if *entry > 0 {
            common += 1;
            *entry -= 1;
        }
    }
    common.max(0) as usize
}
