//! SwiGLU activation function.
//!
//! `SwiGLU(x) = SiLU(gate(x)) * up(x)`
//! where `SiLU(x) = x * sigmoid(x)`.
//!
//! Used in the MLP (feed-forward) blocks of Qwen3.

/// Apply SiLU (Swish) activation: `silu(x) = x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SwiGLU: `swiglu(gate, up) = silu(gate) * up`.
///
/// - `gate`: Output of gate projection (length n).
/// - `up`: Output of up projection (length n).
/// - `output`: Result buffer (length n).
///
/// Delegates to the SIMD-accelerated implementation in `oxibonsai_kernels`.
pub fn swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert!(output.len() >= gate.len());

    oxibonsai_kernels::swiglu_simd(gate, up, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_at_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn silu_positive() {
        // silu(1.0) = 1.0 / (1.0 + exp(-1.0)) ≈ 0.7311
        let result = silu(1.0);
        assert!((result - 0.7311).abs() < 0.001);
    }

    #[test]
    fn swiglu_basic() {
        let gate = vec![1.0, 0.0, -1.0];
        let up = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];

        swiglu(&gate, &up, &mut output);

        assert!((output[0] - silu(1.0) * 2.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5); // silu(0)*3 = 0
        assert!((output[2] - silu(-1.0) * 4.0).abs() < 1e-5);
    }
}
