//! Bufferless Q8_0 / F16 / F32 SIMD dot-product and semantic prejudice kernels
//! for `RestRescueGuard` and scalar prejudice scoring (Phase 1 – Part 3).

use half::f16;

/// Compute semantic prejudice penalty from raw `(dot(w, target_norm), ||w||^2)` without
/// NaN propagation on zero vectors or non-finite inputs.
#[inline(always)]
pub fn prejudice_from_stats(dot_wt: f32, norm_w_sq: f32) -> f32 {
    if !(norm_w_sq > 1e-24) || !norm_w_sq.is_finite() || !dot_wt.is_finite() {
        return 0.0;
    }
    let cos = dot_wt / norm_w_sq.sqrt();
    let relu = (cos - 0.65).max(0.0);
    if relu == 0.0 {
        0.0
    } else {
        (15.0 * relu).exp() - 1.0
    }
}

/// Scalar Q8_0 (34-byte blocks: 2-byte LE f16 scale + 32 i8 quants) dot product against `f32` slice.
pub fn dot_q8_0_f32_scalar(raw_q8: &[u8], h: &[f32]) -> f32 {
    let n_blocks = (h.len() / 32).min(raw_q8.len() / 34);
    let mut total = 0.0f32;
    for b in 0..n_blocks {
        let blk = &raw_q8[b * 34..b * 34 + 34];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..34];
        let xs = &h[b * 32..b * 32 + 32];
        let mut bsum = 0.0f32;
        for k in 0..32 {
            bsum = (qs[k] as i8 as f32).mul_add(xs[k], bsum);
        }
        total = d.mul_add(bsum, total);
    }
    total
}

/// Scalar fused Q8_0 dot product minus semantic prejudice penalty in one pass over raw Q8_0 bytes.
pub fn score_q8_0_prejudice_scalar(raw_q8: &[u8], h: &[f32], t: &[f32]) -> f32 {
    let n_blocks = (h.len() / 32).min(t.len() / 32).min(raw_q8.len() / 34);
    let mut dot_wh = 0.0f32;
    let mut dot_wt = 0.0f32;
    let mut norm_w_sq = 0.0f32;
    for b in 0..n_blocks {
        let blk = &raw_q8[b * 34..b * 34 + 34];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..34];
        let xb = &h[b * 32..b * 32 + 32];
        let tb = &t[b * 32..b * 32 + 32];
        let mut wh_b = 0.0f32;
        let mut wt_b = 0.0f32;
        let mut ww_b = 0.0f32;
        for k in 0..32 {
            let q = qs[k] as i8 as f32;
            wh_b = q.mul_add(xb[k], wh_b);
            wt_b = q.mul_add(tb[k], wt_b);
            ww_b = q.mul_add(q, ww_b);
        }
        dot_wh = d.mul_add(wh_b, dot_wh);
        dot_wt = d.mul_add(wt_b, dot_wt);
        norm_w_sq = (d * d).mul_add(ww_b, norm_w_sq);
    }
    dot_wh - prejudice_from_stats(dot_wt, norm_w_sq)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q8_0_f32_avx2(raw_q8: &[u8], h: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = (h.len() / 32).min(raw_q8.len() / 34);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let rp = raw_q8.as_ptr();
    let hp = h.as_ptr();

    #[inline(always)]
    unsafe fn eval_block(blk: *const u8, x: *const f32, acc: __m256) -> __m256 {
        let d_bits = u16::from_le_bytes([*blk, *blk.add(1)]) as i32;
        let d_f32 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d_bits)));
        let q_lo = _mm_loadu_si128(blk.add(2) as *const __m128i);
        let q_hi = _mm_loadu_si128(blk.add(18) as *const __m128i);
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
        let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
        let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));
        let s01 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(x), _mm256_mul_ps(q1, _mm256_loadu_ps(x.add(8))));
        let s23 = _mm256_fmadd_ps(q2, _mm256_loadu_ps(x.add(16)), _mm256_mul_ps(q3, _mm256_loadu_ps(x.add(24))));
        _mm256_fmadd_ps(_mm256_set1_ps(d_f32), _mm256_add_ps(s01, s23), acc)
    }

    let mut b = 0usize;
    while b + 2 <= n_blocks {
        acc0 = eval_block(rp.add(b * 34), hp.add(b * 32), acc0);
        acc1 = eval_block(rp.add((b + 1) * 34), hp.add((b + 1) * 32), acc1);
        b += 2;
    }
    if b < n_blocks {
        acc0 = eval_block(rp.add(b * 34), hp.add(b * 32), acc0);
    }
    hsum256_ps(_mm256_add_ps(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn score_q8_0_prejudice_avx2(raw_q8: &[u8], h: &[f32], t: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = (h.len() / 32).min(t.len() / 32).min(raw_q8.len() / 34);
    let mut wh_v = _mm256_setzero_ps();
    let mut wt_v = _mm256_setzero_ps();
    let mut ww_v = _mm256_setzero_ps();
    let rp = raw_q8.as_ptr();
    let hp = h.as_ptr();
    let tp = t.as_ptr();

    for b in 0..n_blocks {
        let blk = rp.add(b * 34);
        let d_bits = u16::from_le_bytes([*blk, *blk.add(1)]) as i32;
        let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d_bits)));
        let dv = _mm256_set1_ps(d);
        let d2v = _mm256_set1_ps(d * d);
        let q_lo = _mm_loadu_si128(blk.add(2) as *const __m128i);
        let q_hi = _mm_loadu_si128(blk.add(18) as *const __m128i);
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
        let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
        let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));
        let xb = hp.add(b * 32);
        let tb = tp.add(b * 32);
        let wh_b = _mm256_add_ps(
            _mm256_fmadd_ps(q0, _mm256_loadu_ps(xb), _mm256_mul_ps(q1, _mm256_loadu_ps(xb.add(8)))),
            _mm256_fmadd_ps(q2, _mm256_loadu_ps(xb.add(16)), _mm256_mul_ps(q3, _mm256_loadu_ps(xb.add(24)))),
        );
        let wt_b = _mm256_add_ps(
            _mm256_fmadd_ps(q0, _mm256_loadu_ps(tb), _mm256_mul_ps(q1, _mm256_loadu_ps(tb.add(8)))),
            _mm256_fmadd_ps(q2, _mm256_loadu_ps(tb.add(16)), _mm256_mul_ps(q3, _mm256_loadu_ps(tb.add(24)))),
        );
        let ww_b = _mm256_add_ps(
            _mm256_fmadd_ps(q0, q0, _mm256_mul_ps(q1, q1)),
            _mm256_fmadd_ps(q2, q2, _mm256_mul_ps(q3, q3)),
        );
        wh_v = _mm256_fmadd_ps(dv, wh_b, wh_v);
        wt_v = _mm256_fmadd_ps(dv, wt_b, wt_v);
        ww_v = _mm256_fmadd_ps(d2v, ww_b, ww_v);
    }

    let dot_wh = hsum256_ps(wh_v);
    let dot_wt = hsum256_ps(wt_v);
    let norm_w_sq = hsum256_ps(ww_v);
    dot_wh - prejudice_from_stats(dot_wt, norm_w_sq)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_q8_0_f32_neon(raw_q8: &[u8], h: &[f32]) -> f32 {
    dot_q8_0_f32_scalar(raw_q8, h)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn score_q8_0_prejudice_neon(raw_q8: &[u8], h: &[f32], t: &[f32]) -> f32 {
    score_q8_0_prejudice_scalar(raw_q8, h, t)
}

/// Dispatch bufferless Q8_0 dot product to AVX2+FMA+F16C / NEON / scalar.
#[inline]
pub fn dot_q8_0_f32_simd(raw_q8: &[u8], h: &[f32]) -> f32 {
    if !crate::is_force_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
            {
                return unsafe { dot_q8_0_f32_avx2(raw_q8, h) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { dot_q8_0_f32_neon(raw_q8, h) };
            }
        }
    }
    dot_q8_0_f32_scalar(raw_q8, h)
}

/// Dispatch bufferless fused Q8_0 dot product minus semantic prejudice.
#[inline]
pub fn score_q8_0_prejudice_simd(raw_q8: &[u8], h: &[f32], t: &[f32]) -> f32 {
    if !crate::is_force_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
            {
                return unsafe { score_q8_0_prejudice_avx2(raw_q8, h, t) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { score_q8_0_prejudice_neon(raw_q8, h, t) };
            }
        }
    }
    score_q8_0_prejudice_scalar(raw_q8, h, t)
}

/// Bufferless raw little-endian F16 row dot product against `h`.
#[inline]
pub fn dot_f16_f32_simd(raw_f16: &[u8], h: &[f32]) -> f32 {
    let n = (raw_f16.len() / 2).min(h.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut i = 0usize;
    while i + 2 <= n {
        let w0 = f16::from_le_bytes([raw_f16[2 * i], raw_f16[2 * i + 1]]).to_f32();
        let w1 = f16::from_le_bytes([raw_f16[2 * i + 2], raw_f16[2 * i + 3]]).to_f32();
        s0 = w0.mul_add(h[i], s0);
        s1 = w1.mul_add(h[i + 1], s1);
        i += 2;
    }
    if i < n {
        let w = f16::from_le_bytes([raw_f16[2 * i], raw_f16[2 * i + 1]]).to_f32();
        s0 = w.mul_add(h[i], s0);
    }
    s0 + s1
}

/// Bufferless raw little-endian F16 dot product minus semantic prejudice in one pass.
#[inline]
pub fn score_f16_prejudice_simd(raw_f16: &[u8], h: &[f32], t: &[f32]) -> f32 {
    let n = (raw_f16.len() / 2).min(h.len()).min(t.len());
    let mut dot_wh = 0.0f32;
    let mut dot_wt = 0.0f32;
    let mut norm_w_sq = 0.0f32;
    for i in 0..n {
        let w = f16::from_le_bytes([raw_f16[2 * i], raw_f16[2 * i + 1]]).to_f32();
        dot_wh = w.mul_add(h[i], dot_wh);
        dot_wt = w.mul_add(t[i], dot_wt);
        norm_w_sq = w.mul_add(w, norm_w_sq);
    }
    dot_wh - prejudice_from_stats(dot_wt, norm_w_sq)
}

/// Single-pass `f32` dot product `w . x` and squared norm `||w||^2`.
#[inline]
pub fn dot_and_norm_sq_f32_simd(w: &[f32], x: &[f32]) -> (f32, f32) {
    let n = w.len().min(x.len());
    let mut dwx = 0.0f32;
    let mut dww = 0.0f32;
    for i in 0..n {
        let wi = w[i];
        dwx = wi.mul_add(x[i], dwx);
        dww = wi.mul_add(wi, dww);
    }
    (dwx, dww)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8_row(cols: usize, seed: u64, zero_scales: bool) -> (Vec<u8>, Vec<f32>) {
        let n_blocks = cols / 32;
        let mut raw = Vec::with_capacity(n_blocks * 34);
        let mut ref_w = Vec::with_capacity(cols);
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        for b in 0..n_blocks {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let scale = if zero_scales {
                f16::from_f32(0.0)
            } else {
                f16::from_f32((((s % 200) as f32) - 100.0) * 0.015)
            };
            raw.extend_from_slice(&scale.to_le_bytes());
            let d = scale.to_f32();
            for _k in 0..32 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let q = ((s >> 56) as i8).clamp(-127, 127);
                raw.push(q as u8);
                ref_w.push(d * (q as f32));
            }
            let _ = b;
        }
        (raw, ref_w)
    }

    #[test]
    fn q8_dot_and_prejudice_match_f64_reference() {
        let cols = 2560usize;
        let (raw, w_f32) = make_q8_row(cols, 42, false);
        let h: Vec<f32> = (0..cols).map(|i| ((i % 29) as f32 - 14.0) * 0.07).collect();
        let t_raw: Vec<f32> = (0..cols).map(|i| if i < 64 { w_f32[i] } else { 0.01 * ((i % 7) as f32) }).collect();
        let tn: f32 = t_raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let t: Vec<f32> = t_raw.iter().map(|v| v / tn).collect();

        let ref_wh: f64 = w_f32.iter().zip(&h).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
        let w_norm: f64 = w_f32.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let h_norm: f64 = h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let got_scalar = dot_q8_0_f32_scalar(&raw, &h);
        let got_simd = dot_q8_0_f32_simd(&raw, &h);
        let tol = 1e-6 * w_norm * h_norm + 1e-7;
        assert!(((got_scalar as f64) - ref_wh).abs() <= tol);
        assert!(((got_simd as f64) - ref_wh).abs() <= tol);

        let ref_wt: f64 = w_f32.iter().zip(&t).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
        let ref_score = ref_wh - (prejudice_from_stats(ref_wt as f32, (w_norm * w_norm) as f32) as f64);
        let got_p = score_q8_0_prejudice_simd(&raw, &h, &t);
        let ptol = 1e-4 + 2e-5 * ref_score.abs() + 1e-6 * w_norm * h_norm;
        assert!(((got_p as f64) - ref_score).abs() <= ptol, "got={got_p} ref={ref_score}");
    }

    #[test]
    fn q8_zero_vector_and_truncated_slice_no_nan_no_panic() {
        let (zero_raw, _) = make_q8_row(64, 7, true);
        let h = vec![1.5f32; 64];
        let t = vec![1.0 / 8.0; 64];
        let v = score_q8_0_prejudice_simd(&zero_raw, &h, &t);
        assert!(v.is_finite() && v == 0.0);
        // Truncated slice must safely clamp instead of panicking
        let trunc = &zero_raw[..17];
        assert_eq!(dot_q8_0_f32_simd(trunc, &h), 0.0);
    }
}
