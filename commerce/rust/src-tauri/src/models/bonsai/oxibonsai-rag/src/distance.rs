//! Distance / similarity metrics for the RAG pipeline.
//!
//! The [`Distance`] enum enumerates the metrics we support out-of-the-box.
//! Every metric is implemented as a pure function over `&[f32]` slices and
//! rejects `NaN` / `±∞` inputs with [`RagError::NonFinite`] rather than
//! silently poisoning downstream scores.
//!
//! Convention (intentional): lower-is-better for *distances* (Euclidean,
//! Angular, Hamming) and higher-is-better for *similarities* (Cosine,
//! DotProduct).  The [`Distance::is_similarity`] predicate exposes this
//! polarity so that callers can sort results correctly.
//!
//! # NaN / Inf guard
//!
//! `f32::is_finite()` returns `false` for both all flavours of `NaN`
//! (quiet and signalling — Rust does not distinguish between them at the
//! API level) and for `±∞`, so a single `is_finite` filter covers every
//! non-finite bit pattern.  The helper [`Distance::validate`] is exposed
//! publicly so external callers can pre-screen inputs if they prefer.

use serde::{Deserialize, Serialize};

use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// Distance enum
// ─────────────────────────────────────────────────────────────────────────────

/// Supported distance / similarity metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Distance {
    /// Cosine similarity — `Σ(a·b) / (‖a‖·‖b‖)`, clamped to `[-1, 1]`.
    ///
    /// Higher values indicate greater similarity.  When inputs are unit
    /// vectors this reduces to [`Distance::DotProduct`].
    #[default]
    Cosine,
    /// Euclidean (L2) distance — `sqrt(Σ(a_i - b_i)²)`.
    ///
    /// Lower values indicate closer points.  Satisfies the triangle
    /// inequality.
    Euclidean,
    /// Dot product — `Σ(a_i · b_i)` with no normalisation.
    ///
    /// Higher values indicate greater alignment.  Sensitive to vector
    /// magnitude — consider [`Distance::Cosine`] if inputs vary in norm.
    DotProduct,
    /// Angular distance — `acos(clamp(cosine, -1, 1)) / π`, normalised to
    /// `[0, 1]`.
    ///
    /// Lower values mean smaller angle between vectors.  A true metric
    /// (symmetric and triangle-inequality-respecting) when applied to
    /// unit vectors.
    Angular,
    /// Hamming bit distance over the `to_bits()` representation of each
    /// `f32`, averaged across all elements.
    ///
    /// For two slices of length `n` with bitwise-different 32-bit
    /// patterns the value is in `[0.0, 32.0]` — it equals the *mean*
    /// per-element bit-difference count.  Lower is closer.
    Hamming,
}

impl Distance {
    /// Returns `true` if larger values mean *more similar* (i.e. cosine
    /// / dot-product semantics) rather than *farther apart*.
    #[inline]
    pub fn is_similarity(self) -> bool {
        matches!(self, Self::Cosine | Self::DotProduct)
    }

    /// Returns `true` if this metric is a true distance (lower-is-closer
    /// and satisfies the triangle inequality).
    #[inline]
    pub fn is_distance(self) -> bool {
        !self.is_similarity()
    }

    /// Reject `NaN` / `±∞` inputs with [`RagError::NonFinite`].
    ///
    /// This is exposed publicly so that external callers can pre-validate
    /// vectors once rather than re-checking on every call.
    #[inline]
    pub fn validate(a: &[f32], b: &[f32]) -> Result<(), RagError> {
        if a.iter().any(|x| !x.is_finite()) || b.iter().any(|x| !x.is_finite()) {
            return Err(RagError::NonFinite);
        }
        Ok(())
    }

    /// Compute this metric between `a` and `b`.
    ///
    /// Returns [`RagError::NonFinite`] if either slice contains a `NaN`
    /// or `±∞` entry, and [`RagError::DimensionMismatch`] if the slices
    /// differ in length or are empty.
    pub fn compute(self, a: &[f32], b: &[f32]) -> Result<f32, RagError> {
        if a.is_empty() || a.len() != b.len() {
            return Err(RagError::DimensionMismatch {
                expected: a.len().max(1),
                got: b.len(),
            });
        }
        Self::validate(a, b)?;

        let value = match self {
            Self::Cosine => cosine(a, b),
            Self::Euclidean => euclidean(a, b),
            Self::DotProduct => dot(a, b),
            Self::Angular => angular(a, b),
            Self::Hamming => hamming(a, b),
        };

        // Defensive: guard against metrics that might produce NaN/Inf
        // even on finite inputs (e.g. overflow in dot product).  We still
        // surface NonFinite rather than returning a poisoned score.
        if !value.is_finite() {
            return Err(RagError::NonFinite);
        }
        Ok(value)
    }

    /// Convert a distance/similarity score into a *score* where larger
    /// always means "better match".  This is used by `VectorStore` when
    /// sorting results.
    #[inline]
    pub fn to_score(self, value: f32) -> f32 {
        if self.is_similarity() {
            value
        } else {
            -value
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual metric implementations
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-12 || nb < 1e-12 {
        // Zero (or near-zero) vector — define cosine(0, _) = 0 so the
        // value is finite and unambiguous.
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

#[inline]
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    sum.sqrt()
}

#[inline]
fn angular(a: &[f32], b: &[f32]) -> f32 {
    let cos = cosine(a, b).clamp(-1.0, 1.0);
    cos.acos() / std::f32::consts::PI
}

#[inline]
fn hamming(a: &[f32], b: &[f32]) -> f32 {
    // Mean per-element bit difference.  Each f32 contributes 0..=32.
    let total: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_bits() ^ y.to_bits()).count_ones() as u64)
        .sum();
    total as f32 / a.len() as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_unit_vectors_is_one() {
        let a = vec![0.6, 0.8];
        let b = vec![0.6, 0.8];
        let d = Distance::Cosine
            .compute(&a, &b)
            .expect("finite inputs must succeed");
        assert!((d - 1.0).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn euclidean_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = Distance::Euclidean.compute(&a, &a).expect("compute");
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn angular_range_zero_to_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = Distance::Angular.compute(&a, &b).expect("compute");
        assert!((d - 1.0).abs() < 1e-5, "got {d}");
    }

    #[test]
    fn hamming_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = Distance::Hamming.compute(&a, &a).expect("compute");
        assert_eq!(d, 0.0);
    }

    #[test]
    fn nan_rejected() {
        let a = vec![1.0, f32::NAN];
        let b = vec![1.0, 2.0];
        let e = Distance::Cosine.compute(&a, &b);
        assert!(matches!(e, Err(RagError::NonFinite)));
    }

    #[test]
    fn inf_rejected() {
        let a = vec![f32::INFINITY, 2.0];
        let b = vec![1.0, 2.0];
        let e = Distance::Euclidean.compute(&a, &b);
        assert!(matches!(e, Err(RagError::NonFinite)));
    }

    #[test]
    fn dim_mismatch_rejected() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        let e = Distance::DotProduct.compute(&a, &b);
        assert!(matches!(e, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn is_similarity_classifies_correctly() {
        assert!(Distance::Cosine.is_similarity());
        assert!(Distance::DotProduct.is_similarity());
        assert!(!Distance::Euclidean.is_similarity());
        assert!(Distance::Euclidean.is_distance());
        assert!(Distance::Angular.is_distance());
        assert!(Distance::Hamming.is_distance());
    }
}
