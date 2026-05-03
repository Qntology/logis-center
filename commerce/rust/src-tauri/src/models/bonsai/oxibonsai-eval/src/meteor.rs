//! METEOR (lexical subset) — exact-match only.
//!
//! This is Denkowski & Lavie (2014)'s METEOR, restricted to exact word
//! matching: **no stemming, no synonymy (WordNet), no paraphrase tables**.
//! This limits absolute scores relative to a full METEOR implementation but
//! preserves the core algorithmic shape:
//!
//! 1. Align candidate tokens to reference tokens (exact match, greedy).
//! 2. Compute precision `P` and recall `R` over matched tokens.
//! 3. Combine via harmonic-mean with α=0.9 default:
//!    `F = P·R / (α·P + (1-α)·R)`
//!    (α→1 weights recall, α→0 weights precision).
//! 4. Apply fragmentation penalty on the number of *chunks* of consecutive
//!    matches in the candidate that align to consecutive reference positions:
//!    `pen = γ · (chunks / matches)^β`  (γ=0.5, β=3 by default).
//! 5. Final score: `score = (1 - pen) · F`.
//!
//! Multi-reference: we compute METEOR against each reference and take the
//! maximum (standard protocol).
//!
//! The limitation is documented here for transparency; if stemming/synonymy
//! is needed in the future, it can be layered on top of [`align_tokens`].

use crate::rouge::{tokenize, TokenSeq};

/// METEOR score breakdown.
#[derive(Debug, Clone)]
pub struct MeteorScore {
    /// Final METEOR in `[0, 1]`.
    pub score: f32,
    /// Matcher precision `matches / |candidate|`.
    pub precision: f32,
    /// Matcher recall `matches / |reference|`.
    pub recall: f32,
    /// Fragmentation factor in `[0, γ]` (higher = more fragmented).
    pub fragmentation: f32,
}

/// METEOR configuration.
#[derive(Debug, Clone)]
pub struct MeteorConfig {
    /// Weight for precision vs recall in the F-mean. Standard α = 0.9.
    pub alpha: f32,
    /// Penalty weight for fragmentation (γ, default 0.5).
    pub gamma: f32,
    /// Fragmentation exponent (β, default 3.0).
    pub beta: f32,
}

impl Default for MeteorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.9,
            gamma: 0.5,
            beta: 3.0,
        }
    }
}

/// Compute METEOR between a candidate and a single reference.
pub fn meteor(candidate: &str, reference: &str, cfg: &MeteorConfig) -> MeteorScore {
    let cand = tokenize(candidate);
    let refs = tokenize(reference);
    meteor_tokens(&cand, &refs, cfg)
}

/// Compute METEOR from pre-tokenised sequences.
pub fn meteor_tokens(
    candidate: &TokenSeq,
    reference: &TokenSeq,
    cfg: &MeteorConfig,
) -> MeteorScore {
    if candidate.is_empty() && reference.is_empty() {
        return MeteorScore {
            score: 1.0,
            precision: 1.0,
            recall: 1.0,
            fragmentation: 0.0,
        };
    }
    if candidate.is_empty() || reference.is_empty() {
        return MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            fragmentation: 0.0,
        };
    }

    // Align tokens: list of (cand_idx, ref_idx) pairs.
    let alignment = align_tokens(candidate, reference);
    let matches = alignment.len();

    if matches == 0 {
        return MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            fragmentation: 0.0,
        };
    }

    let p = matches as f32 / candidate.len() as f32;
    let r = matches as f32 / reference.len() as f32;

    let denom = cfg.alpha * p + (1.0 - cfg.alpha) * r;
    let f_mean = if denom > 0.0 { (p * r) / denom } else { 0.0 };

    // Count chunks in the alignment (consecutive in both candidate and reference).
    let chunks = count_chunks(&alignment);
    let frag = (chunks as f32) / (matches as f32);
    let pen = cfg.gamma * frag.powf(cfg.beta);

    let score = ((1.0 - pen) * f_mean).clamp(0.0, 1.0);
    MeteorScore {
        score,
        precision: p,
        recall: r,
        fragmentation: pen,
    }
}

/// Compute METEOR against multiple references; returns max.
pub fn meteor_multi(candidate: &str, references: &[&str], cfg: &MeteorConfig) -> MeteorScore {
    if references.is_empty() {
        return MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            fragmentation: 0.0,
        };
    }
    let mut best: Option<MeteorScore> = None;
    for r in references {
        let s = meteor(candidate, r, cfg);
        best = match best.take() {
            None => Some(s),
            Some(b) => {
                if s.score > b.score {
                    Some(s)
                } else {
                    Some(b)
                }
            }
        };
    }
    best.unwrap_or(MeteorScore {
        score: 0.0,
        precision: 0.0,
        recall: 0.0,
        fragmentation: 0.0,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal — alignment + chunks
// ──────────────────────────────────────────────────────────────────────────────

/// Greedy left-to-right alignment: for each candidate token (in order), match
/// it to the first unclaimed reference position whose token equals it.
///
/// Returns `(cand_idx, ref_idx)` pairs in candidate order.
pub fn align_tokens(candidate: &TokenSeq, reference: &TokenSeq) -> Vec<(usize, usize)> {
    let mut used = vec![false; reference.len()];
    let mut out: Vec<(usize, usize)> = Vec::new();

    for (ci, ctok) in candidate.iter().enumerate() {
        for (ri, rtok) in reference.iter().enumerate() {
            if !used[ri] && ctok == rtok {
                used[ri] = true;
                out.push((ci, ri));
                break;
            }
        }
    }
    out
}

/// Count the number of chunks — maximal runs where both candidate and
/// reference indices are consecutive.
///
/// We sort the alignment by candidate index, then walk it: each pair
/// continues the current chunk iff `ci == prev_ci + 1 && ri == prev_ri + 1`.
fn count_chunks(alignment: &[(usize, usize)]) -> usize {
    if alignment.is_empty() {
        return 0;
    }
    let mut sorted = alignment.to_vec();
    sorted.sort_by_key(|&(ci, _)| ci);

    let mut chunks = 1usize;
    for w in sorted.windows(2) {
        let (pc, pr) = w[0];
        let (nc, nr) = w[1];
        if !(nc == pc + 1 && nr == pr + 1) {
            chunks += 1;
        }
    }
    chunks
}
