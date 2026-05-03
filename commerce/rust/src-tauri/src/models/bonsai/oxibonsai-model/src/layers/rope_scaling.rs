//! RoPE (Rotary Position Embedding) scaling variants for extended context.
//!
//! Strategies:
//! - `None`: standard RoPE — no scaling applied.
//! - `Linear`: scale all frequencies by 1/s (simple, fast, loses high-freq info).
//! - `DynamicNtk`: NTK-aware scaling applied dynamically at inference time.
//! - `Llama31`: LLaMA 3.1 / LLaMA 3.2 scaling with low/high frequency blending.
//! - `LongRope`: LongRoPE with per-dimension rescale factors.
//!
//! ## Standard RoPE Frequency Convention
//!
//! For dimension index `i` in `[0, head_dim/2)`:
//!
//! ```text
//! freq_i = 1.0 / base^(2*i / head_dim)
//! ```
//!
//! The actual rotation angle at position `p` is `theta_i * p = freq_i * p`.

use thiserror::Error;

// ─── RopeScalingStrategy ─────────────────────────────────────────────────────

/// A RoPE scaling strategy for extended context inference.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingStrategy {
    /// No scaling — standard RoPE with unmodified frequencies.
    None,

    /// Linear scaling: divide all frequencies by `scale_factor`.
    ///
    /// Equivalent to multiplying the effective sequence length by `scale_factor`.
    /// Fast and simple but degrades quality for high-frequency dimensions.
    Linear {
        /// Must be >= 1.0.
        scale_factor: f32,
    },

    /// Dynamic NTK scaling: scales the RoPE base frequency at inference time.
    ///
    /// The effective base is computed as:
    /// ```text
    /// effective_base = base * (s * max_pos / orig_max_pos - (s - 1))^(d / (d - 2))
    /// ```
    /// where `s = current_seq_len / original_max_position`.
    ///
    /// When `current_seq_len <= original_max_position`, standard frequencies are used.
    DynamicNtk {
        /// The maximum sequence length used during pretraining.
        original_max_position: usize,
        /// Base frequency (e.g. 10000.0 for most models, 500000.0 for LLaMA 3.1).
        base: f32,
    },

    /// LLaMA 3.1 / LLaMA 3.2 scaling: blends original and scaled frequencies
    /// per dimension based on wavelength thresholds.
    ///
    /// Dimensions with long wavelengths (low frequency) are interpolated;
    /// dimensions with short wavelengths (high frequency) are left unmodified;
    /// intermediate dimensions are smoothly blended.
    Llama31 {
        /// The maximum sequence length used during pretraining.
        original_max_position: usize,
        /// Context extension factor (e.g. 8.0 for 8× context extension).
        scale_factor: f32,
        /// Low-frequency threshold factor (default 1.0 in LLaMA 3.1).
        low_freq_factor: f32,
        /// High-frequency threshold factor (default 4.0 in LLaMA 3.1).
        high_freq_factor: f32,
        /// Base frequency.
        base: f32,
    },

    /// LongRoPE: per-dimension rescale based on externally computed factors.
    ///
    /// Each of the `head_dim/2` frequency dimensions is divided by the
    /// corresponding rescale factor. Factors are typically derived from an
    /// evolutionary search optimising perplexity on long documents.
    LongRope {
        /// Per-dimension rescale factors; length must equal `head_dim / 2`.
        rescale_factors: Vec<f32>,
        /// Original pretraining context length (used for reference only).
        original_max_position: usize,
    },
}

// ─── RopeScalingError ────────────────────────────────────────────────────────

/// Errors produced by RoPE scaling operations.
#[derive(Debug, Error)]
pub enum RopeScalingError {
    /// `head_dim` was zero or odd; must be a positive even integer.
    #[error("head_dim {0} must be even and > 0")]
    InvalidHeadDim(usize),

    /// The length of `rescale_factors` does not match `head_dim / 2`.
    #[error("rescale_factors length {got} != head_dim/2 = {expected}")]
    RescaleFactorLengthMismatch { got: usize, expected: usize },

    /// `scale_factor` was less than 1.0; scaling must not compress the context.
    #[error("scale_factor must be >= 1.0, got {0}")]
    InvalidScaleFactor(f32),

    /// The length of the `q` or `k` slice did not match `head_dim`.
    #[error("q/k length {got} != head_dim {expected}")]
    VecLengthMismatch { got: usize, expected: usize },
}

// ─── compute_rope_frequencies ────────────────────────────────────────────────

/// Compute RoPE frequencies given a scaling strategy.
///
/// Returns `head_dim / 2` angular frequency values (θ_i for i = 0..head_dim/2).
/// The rotation angle at absolute position `p` is `θ_i * p`.
///
/// # Errors
///
/// - [`RopeScalingError::InvalidHeadDim`] if `head_dim` is zero or odd.
/// - [`RopeScalingError::InvalidScaleFactor`] if `scale_factor < 1.0` (Linear strategy).
/// - [`RopeScalingError::RescaleFactorLengthMismatch`] if `rescale_factors.len() != head_dim/2` (LongRope strategy).
pub fn compute_rope_frequencies(
    head_dim: usize,
    base: f32,
    strategy: &RopeScalingStrategy,
    current_seq_len: usize,
) -> Result<Vec<f32>, RopeScalingError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(RopeScalingError::InvalidHeadDim(head_dim));
    }

    match strategy {
        RopeScalingStrategy::None => Ok(standard_frequencies(head_dim, base)),

        RopeScalingStrategy::Linear { scale_factor } => {
            if *scale_factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*scale_factor));
            }
            let freqs = standard_frequencies(head_dim, base);
            Ok(freqs.into_iter().map(|f| f / scale_factor).collect())
        }

        RopeScalingStrategy::DynamicNtk {
            original_max_position,
            base: ntk_base,
        } => {
            let effective_base =
                dynamic_ntk_base(*ntk_base, head_dim, *original_max_position, current_seq_len);
            Ok(standard_frequencies(head_dim, effective_base))
        }

        RopeScalingStrategy::Llama31 {
            original_max_position,
            scale_factor,
            low_freq_factor,
            high_freq_factor,
            base: llama_base,
        } => {
            if *scale_factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*scale_factor));
            }
            Ok(llama31_frequencies(
                head_dim,
                *llama_base,
                *original_max_position,
                *scale_factor,
                *low_freq_factor,
                *high_freq_factor,
            ))
        }

        RopeScalingStrategy::LongRope {
            rescale_factors,
            original_max_position: _,
        } => {
            let half_dim = head_dim / 2;
            if rescale_factors.len() != half_dim {
                return Err(RopeScalingError::RescaleFactorLengthMismatch {
                    got: rescale_factors.len(),
                    expected: half_dim,
                });
            }
            let freqs = standard_frequencies(head_dim, base);
            Ok(freqs
                .into_iter()
                .zip(rescale_factors.iter())
                .map(|(f, &r)| {
                    // Guard against zero rescale factors to avoid division by zero.
                    if r.abs() < f32::EPSILON {
                        f
                    } else {
                        f / r
                    }
                })
                .collect())
        }
    }
}

// ─── apply_rope_with_freqs ───────────────────────────────────────────────────

/// Apply standard RoPE rotation to a query/key vector at position `pos`.
///
/// Uses precomputed frequencies from [`compute_rope_frequencies`].
/// Rotates pairs `(v[i], v[i + half_dim])` in-place for both `q` and `k`.
///
/// At position 0 the rotation is the identity (cos(0)=1, sin(0)=0).
///
/// # Errors
///
/// - [`RopeScalingError::InvalidHeadDim`] if `freqs.len() * 2` is zero or odd.
/// - [`RopeScalingError::VecLengthMismatch`] if `q.len()` or `k.len()` ≠ `freqs.len() * 2`.
pub fn apply_rope_with_freqs(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    freqs: &[f32],
) -> Result<(), RopeScalingError> {
    let half = freqs.len();
    let head_dim = half * 2;

    if half == 0 {
        return Err(RopeScalingError::InvalidHeadDim(0));
    }

    if q.len() != head_dim {
        return Err(RopeScalingError::VecLengthMismatch {
            got: q.len(),
            expected: head_dim,
        });
    }
    if k.len() != head_dim {
        return Err(RopeScalingError::VecLengthMismatch {
            got: k.len(),
            expected: head_dim,
        });
    }

    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let (sin_a, cos_a) = angle.sin_cos();

        // Rotate query pair
        let q0 = q[i];
        let q1 = q[half + i];
        q[i] = q0 * cos_a - q1 * sin_a;
        q[half + i] = q0 * sin_a + q1 * cos_a;

        // Rotate key pair
        let k0 = k[i];
        let k1 = k[half + i];
        k[i] = k0 * cos_a - k1 * sin_a;
        k[half + i] = k0 * sin_a + k1 * cos_a;
    }

    Ok(())
}

// ─── dynamic_ntk_base ────────────────────────────────────────────────────────

/// Compute the effective base frequency for Dynamic NTK scaling.
///
/// When `current_seq_len <= original_max_position`, returns `base` unchanged
/// (no scaling needed). Otherwise, the effective base is inflated so that the
/// higher-order (lower-frequency) dimensions can represent longer sequences
/// without aliasing.
///
/// Formula:
/// ```text
/// s  = current_seq_len / original_max_position
/// effective_base = base * s^(d / (d - 2))
/// ```
///
/// where `d = head_dim`. This matches the formulation in Su et al. 2023
/// ("Scaling RoPE beyond Training Context") and the HuggingFace implementation.
pub fn dynamic_ntk_base(
    base: f32,
    head_dim: usize,
    original_max_position: usize,
    current_seq_len: usize,
) -> f32 {
    if current_seq_len <= original_max_position || original_max_position == 0 {
        return base;
    }

    let s = current_seq_len as f32 / original_max_position as f32;

    // NTK exponent: d / (d - 2); fallback to 1.0 for tiny head dims.
    let ntk_exp = if head_dim > 2 {
        head_dim as f32 / (head_dim as f32 - 2.0)
    } else {
        1.0
    };

    base * s.powf(ntk_exp)
}

// ─── llama31_frequencies ─────────────────────────────────────────────────────

/// Compute LLaMA 3.1 per-dimension frequencies.
///
/// Implements the frequency blending described in the LLaMA 3.1 technical
/// report (Meta AI, 2024). For each frequency dimension `i`:
///
/// 1. Compute the standard frequency `f_i = 1 / base^(2i/d)`.
/// 2. Compute the wavelength `λ = 2π / f_i`.
/// 3. Determine low/high wavelength thresholds from the original context:
///    - `low_thresh  = original_max_position / high_freq_factor`
///    - `high_thresh = original_max_position / low_freq_factor`
/// 4. Blend:
///    - `λ < low_thresh`  → use `f_i` unchanged (high frequency, no scaling).
///    - `λ > high_thresh` → divide `f_i` by `scale_factor` (pure interpolation).
///    - otherwise         → smooth ramp between the two.
///
/// When `scale_factor == 1.0` this returns exactly the standard frequencies.
pub fn llama31_frequencies(
    head_dim: usize,
    base: f32,
    original_max_position: usize,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let orig = original_max_position as f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    // Wavelength thresholds
    // high_freq dimensions have wavelength < low_thresh  → not scaled
    // low_freq  dimensions have wavelength > high_thresh → scaled by 1/scale_factor
    let low_thresh = if high_freq_factor.abs() > f32::EPSILON {
        orig / high_freq_factor
    } else {
        f32::MAX
    };
    let high_thresh = if low_freq_factor.abs() > f32::EPSILON {
        orig / low_freq_factor
    } else {
        f32::MAX
    };

    (0..half_dim)
        .map(|i| {
            let freq = standard_freq(i, head_dim, base);
            if (scale_factor - 1.0).abs() < f32::EPSILON {
                // scale_factor == 1 → no change regardless of wavelength
                return freq;
            }

            let wavelength = if freq > f32::EPSILON {
                two_pi / freq
            } else {
                f32::MAX
            };

            if wavelength < low_thresh {
                // High-frequency dimension — leave unchanged.
                freq
            } else if wavelength > high_thresh {
                // Low-frequency dimension — apply full linear scaling.
                freq / scale_factor
            } else {
                // Intermediate — smooth linear blend.
                // ramp ∈ [0, 1]: 0 at high_thresh boundary, 1 at low_thresh boundary.
                let range = high_thresh - low_thresh;
                let ramp = if range > f32::EPSILON {
                    (wavelength - low_thresh) / range
                } else {
                    0.5
                };
                // ramp=0 → not scaled; ramp=1 → fully scaled.
                // Blend between unscaled (1.0 weight at ramp=0) and scaled (ramp=1).
                let scaled_freq = freq / scale_factor;
                (1.0 - ramp) * freq + ramp * scaled_freq
            }
        })
        .collect()
}

// ─── FreqStats ───────────────────────────────────────────────────────────────

/// Statistics summarising a set of RoPE frequencies.
#[derive(Debug, Clone)]
pub struct FreqStats {
    /// Smallest frequency value in the set.
    pub min_freq: f32,
    /// Largest frequency value in the set.
    pub max_freq: f32,
    /// Arithmetic mean of all frequency values.
    pub mean_freq: f32,
    /// Approximate maximum representable context: `1 / min_freq`.
    ///
    /// The lowest frequency completes one full rotation in roughly this many
    /// tokens, giving an upper bound on useful positional distinguishability.
    pub effective_context: f32,
}

impl FreqStats {
    /// Compute statistics from a slice of frequencies.
    ///
    /// Returns zeroed stats for an empty slice.
    pub fn compute(freqs: &[f32]) -> Self {
        if freqs.is_empty() {
            return Self {
                min_freq: 0.0,
                max_freq: 0.0,
                mean_freq: 0.0,
                effective_context: 0.0,
            };
        }

        let mut min_freq = freqs[0];
        let mut max_freq = freqs[0];
        let mut sum = 0.0_f64;

        for &f in freqs {
            if f < min_freq {
                min_freq = f;
            }
            if f > max_freq {
                max_freq = f;
            }
            sum += f as f64;
        }

        let mean_freq = (sum / freqs.len() as f64) as f32;
        let effective_context = if min_freq > f32::EPSILON {
            1.0 / min_freq
        } else {
            f32::INFINITY
        };

        Self {
            min_freq,
            max_freq,
            mean_freq,
            effective_context,
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "FreqStats {{ min={:.6e}, max={:.6e}, mean={:.6e}, effective_ctx={:.1} }}",
            self.min_freq, self.max_freq, self.mean_freq, self.effective_context
        )
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Standard RoPE frequency for dimension `i`:
/// `freq = 1 / base^(2i / head_dim)`.
#[inline]
fn standard_freq(i: usize, head_dim: usize, base: f32) -> f32 {
    1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32)
}

/// Compute standard (unscaled) RoPE frequencies for all `head_dim/2` pairs.
fn standard_frequencies(head_dim: usize, base: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    (0..half_dim)
        .map(|i| standard_freq(i, head_dim, base))
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: f32 = 10_000.0;
    const HEAD_DIM: usize = 64;

    fn standard_freqs_ref(head_dim: usize, base: f32) -> Vec<f32> {
        let half = head_dim / 2;
        (0..half)
            .map(|i| 1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect()
    }

    // ── no_scaling_standard_freqs ────────────────────────────────────────────

    #[test]
    fn no_scaling_standard_freqs() {
        let freqs = compute_rope_frequencies(HEAD_DIM, BASE, &RopeScalingStrategy::None, 4096)
            .expect("None strategy should succeed");

        let expected = standard_freqs_ref(HEAD_DIM, BASE);
        assert_eq!(freqs.len(), expected.len());
        for (i, (got, exp)) in freqs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // ── linear_scaling_divides_freqs ─────────────────────────────────────────

    #[test]
    fn linear_scaling_divides_freqs() {
        let scale = 4.0_f32;
        let freqs = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::Linear {
                scale_factor: scale,
            },
            4096,
        )
        .expect("Linear strategy should succeed");

        let standard = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, (got, std_f)) in freqs.iter().zip(standard.iter()).enumerate() {
            let expected = std_f / scale;
            assert!(
                (got - expected).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {expected}"
            );
        }
    }

    // ── linear_scaling_scale_1_unchanged ─────────────────────────────────────

    #[test]
    fn linear_scaling_scale_1_unchanged() {
        let freqs_linear = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::Linear { scale_factor: 1.0 },
            4096,
        )
        .expect("Linear scale=1 should succeed");

        let freqs_none = compute_rope_frequencies(HEAD_DIM, BASE, &RopeScalingStrategy::None, 4096)
            .expect("None strategy should succeed");

        for (i, (a, b)) in freqs_linear.iter().zip(freqs_none.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "freq[{i}]: linear scale=1 got {a}, None got {b}"
            );
        }
    }

    // ── dynamic_ntk_longer_seq_higher_base ───────────────────────────────────

    #[test]
    fn dynamic_ntk_longer_seq_higher_base() {
        let orig = 4096_usize;
        let base_short = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig);
        let base_long = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig * 4);

        assert!(
            base_long > base_short,
            "longer sequence should produce higher effective base: short={base_short}, long={base_long}"
        );
    }

    // ── dynamic_ntk_at_orig_len_unchanged ────────────────────────────────────

    #[test]
    fn dynamic_ntk_at_orig_len_unchanged() {
        let orig = 4096_usize;
        let effective = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig);
        assert!(
            (effective - BASE).abs() < 1e-3,
            "at original length, effective base should equal base: {effective} vs {BASE}"
        );

        // Also verify via compute_rope_frequencies
        let freqs_ntk = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::DynamicNtk {
                original_max_position: orig,
                base: BASE,
            },
            orig,
        )
        .expect("DynamicNtk at orig len should succeed");

        let freqs_none = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, (a, b)) in freqs_ntk.iter().zip(freqs_none.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "freq[{i}]: NTK at orig len got {a}, standard got {b}"
            );
        }
    }

    // ── llama31_freqs_length ──────────────────────────────────────────────────

    #[test]
    fn llama31_freqs_length() {
        let freqs = llama31_frequencies(HEAD_DIM, BASE, 8192, 8.0, 1.0, 4.0);
        assert_eq!(
            freqs.len(),
            HEAD_DIM / 2,
            "llama31_frequencies must return head_dim/2 values"
        );
    }

    // ── llama31_freqs_positive ────────────────────────────────────────────────

    #[test]
    fn llama31_freqs_positive() {
        let freqs = llama31_frequencies(HEAD_DIM, BASE, 8192, 8.0, 1.0, 4.0);
        for (i, &f) in freqs.iter().enumerate() {
            assert!(f > 0.0, "freq[{i}] = {f} is not positive");
        }
    }

    // ── llama31_scale_1_unchanged ─────────────────────────────────────────────

    #[test]
    fn llama31_scale_1_unchanged() {
        let freqs_scaled = llama31_frequencies(HEAD_DIM, BASE, 8192, 1.0, 1.0, 4.0);
        let freqs_standard = standard_freqs_ref(HEAD_DIM, BASE);

        for (i, (got, exp)) in freqs_scaled.iter().zip(freqs_standard.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "freq[{i}]: scale=1 got {got}, standard got {exp}"
            );
        }
    }

    // ── longrope_freqs_uses_factors ───────────────────────────────────────────

    #[test]
    fn longrope_freqs_uses_factors() {
        let half = HEAD_DIM / 2;
        let factors: Vec<f32> = (0..half).map(|i| 1.0 + i as f32 * 0.1).collect();

        let freqs = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::LongRope {
                rescale_factors: factors.clone(),
                original_max_position: 4096,
            },
            8192,
        )
        .expect("LongRope should succeed");

        let standard = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, ((got, std_f), &r)) in freqs
            .iter()
            .zip(standard.iter())
            .zip(factors.iter())
            .enumerate()
        {
            let expected = std_f / r;
            assert!(
                (got - expected).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {expected} (std={std_f}, factor={r})"
            );
        }
    }

    // ── longrope_wrong_factor_count_error ─────────────────────────────────────

    #[test]
    fn longrope_wrong_factor_count_error() {
        let wrong_factors = vec![1.0_f32; 10]; // head_dim/2 = 32, not 10
        let result = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::LongRope {
                rescale_factors: wrong_factors,
                original_max_position: 4096,
            },
            8192,
        );
        assert!(
            matches!(
                result,
                Err(RopeScalingError::RescaleFactorLengthMismatch {
                    got: 10,
                    expected: 32
                })
            ),
            "expected RescaleFactorLengthMismatch, got: {result:?}"
        );
    }

    // ── apply_rope_zero_pos_identity ──────────────────────────────────────────

    #[test]
    fn apply_rope_zero_pos_identity() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| x as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| x as f32 * 0.2 + 1.0).collect();
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rope_with_freqs(&mut q, &mut k, 0, &freqs).expect("apply at pos=0 should succeed");

        for i in 0..HEAD_DIM {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "q[{i}] should be unchanged at pos=0: {} → {}",
                q_orig[i],
                q[i]
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-5,
                "k[{i}] should be unchanged at pos=0: {} → {}",
                k_orig[i],
                k[i]
            );
        }
    }

    // ── apply_rope_changes_at_pos1 ────────────────────────────────────────────

    #[test]
    fn apply_rope_changes_at_pos1() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.5).collect();
        let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.3).collect();
        let q_orig = q.clone();

        apply_rope_with_freqs(&mut q, &mut k, 1, &freqs).expect("apply at pos=1 should succeed");

        // At least some values should have changed
        let changed = q
            .iter()
            .zip(q_orig.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(
            changed,
            "apply_rope_with_freqs at pos=1 should modify values"
        );
    }

    // ── apply_rope_invalid_head_dim_error ─────────────────────────────────────

    #[test]
    fn apply_rope_invalid_head_dim_error() {
        // Build a freq slice of length 0 to trigger InvalidHeadDim(0).
        let freqs: Vec<f32> = vec![];
        let mut q = vec![1.0_f32];
        let mut k = vec![1.0_f32];
        let result = apply_rope_with_freqs(&mut q, &mut k, 0, &freqs);
        assert!(
            matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
            "empty freqs should return InvalidHeadDim(0), got: {result:?}"
        );
    }

    // ── freq_stats_min_max_ordering ───────────────────────────────────────────

    #[test]
    fn freq_stats_min_max_ordering() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let stats = FreqStats::compute(&freqs);
        assert!(
            stats.min_freq <= stats.mean_freq,
            "min ({}) should be <= mean ({})",
            stats.min_freq,
            stats.mean_freq
        );
        assert!(
            stats.mean_freq <= stats.max_freq,
            "mean ({}) should be <= max ({})",
            stats.mean_freq,
            stats.max_freq
        );
    }

    // ── freq_stats_effective_context_positive ─────────────────────────────────

    #[test]
    fn freq_stats_effective_context_positive() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let stats = FreqStats::compute(&freqs);
        assert!(
            stats.effective_context > 0.0,
            "effective_context should be positive, got {}",
            stats.effective_context
        );
    }

    // ── compute_freqs_invalid_dim_error ───────────────────────────────────────

    #[test]
    fn compute_freqs_invalid_dim_error() {
        let result = compute_rope_frequencies(0, BASE, &RopeScalingStrategy::None, 4096);
        assert!(
            matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
            "head_dim=0 should return InvalidHeadDim(0), got: {result:?}"
        );

        let result_odd = compute_rope_frequencies(3, BASE, &RopeScalingStrategy::None, 4096);
        assert!(
            matches!(result_odd, Err(RopeScalingError::InvalidHeadDim(3))),
            "head_dim=3 (odd) should return InvalidHeadDim(3), got: {result_odd:?}"
        );
    }
}
