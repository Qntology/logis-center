//! 64-Key panel streaming attention (`stream_attention_f32`) for CPU inference (Part 2 / Pack D).
//!
//! Eliminates `[B, H, seq_q, seq_kv]` score matrix allocations by streaming transposed key
//! panels `kt[head_dim][KP]` (`KP = 64`) through an online softmax accumulator.
//!
//! Crucially, the value accumulation loop uses a 2-tier SIMD split:
//!   1. Primary 64-element loop (`8 x ymm` AVX2 FMA)
//!   2. Secondary 8-element loop (`1 x ymm` AVX2 FMA)
//!   3. Scalar tail (`< 8` remaining elements)
//! so SigLIP2 (`head_dim = 72 = 64 + 8`) executes with **0 scalar tail fallbacks**!

use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Keys per streaming panel (8 x AVX2 ymm registers = 64 f32 lanes).
pub const KP: usize = 64;
/// Queries per parallel work block.
const QB: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct CpuStreamAttnCfg {
    pub num_heads: usize,    // SigLIP2-so400m: 16 (d=72), Qwen3-0.6B: 16 (d=128)
    pub num_kv_heads: usize, // ViT: == num_heads; GQA LLM: KV heads
    pub head_dim: usize,     // 32, 64, 72, 80, 128, ...
    pub scale: f32,
    pub causal: bool,
    /// Absolute sequence position of query index 0 (for causal KV-cache decoding: k <= past_offset + q).
    pub past_offset: usize,
}

static SCALAR_TAIL_HITS: AtomicUsize = AtomicUsize::new(0);

/// Returns how many times the `< 8` scalar tail of value accumulation ran (Gate 2-B checks `0` for `head_dim = 72`).
pub fn attn_scalar_tail_hits() -> usize {
    SCALAR_TAIL_HITS.load(Ordering::Relaxed)
}

pub fn reset_attn_scalar_tail_hits() {
    SCALAR_TAIL_HITS.store(0, Ordering::Relaxed);
}

#[derive(Default)]
struct TlsScratch {
    scores: Vec<f32>,
    weights: Vec<f32>,
    acc: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<TlsScratch> = RefCell::new(TlsScratch::default());
}

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

/// Computes `scores[0..valid] = (q[0..d] · kt[0..d, 0..valid]) * scale` for a 64-key panel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn score_panel_avx2(
    q: *const f32,
    kt_panel: *const f32, // [d, KP]
    d: usize,
    scale: f32,
    scores: *mut f32, // [KP]
) {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut s4 = _mm256_setzero_ps();
    let mut s5 = _mm256_setzero_ps();
    let mut s6 = _mm256_setzero_ps();
    let mut s7 = _mm256_setzero_ps();

    for e in 0..d {
        let qv = _mm256_set1_ps(*q.add(e));
        let kr = kt_panel.add(e * KP);
        s0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr), s0);
        s1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(8)), s1);
        s2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(16)), s2);
        s3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(24)), s3);
        s4 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(32)), s4);
        s5 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(40)), s5);
        s6 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(48)), s6);
        s7 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(56)), s7);
    }

    let sv = _mm256_set1_ps(scale);
    _mm256_storeu_ps(scores, _mm256_mul_ps(s0, sv));
    _mm256_storeu_ps(scores.add(8), _mm256_mul_ps(s1, sv));
    _mm256_storeu_ps(scores.add(16), _mm256_mul_ps(s2, sv));
    _mm256_storeu_ps(scores.add(24), _mm256_mul_ps(s3, sv));
    _mm256_storeu_ps(scores.add(32), _mm256_mul_ps(s4, sv));
    _mm256_storeu_ps(scores.add(40), _mm256_mul_ps(s5, sv));
    _mm256_storeu_ps(scores.add(48), _mm256_mul_ps(s6, sv));
    _mm256_storeu_ps(scores.add(56), _mm256_mul_ps(s7, sv));
}

#[inline]
fn score_panel_scalar(q: &[f32], kt_panel: &[f32], d: usize, valid: usize, scale: f32, scores: &mut [f32]) {
    scores[..valid].fill(0.0);
    for e in 0..d {
        let qv = q[e];
        let kr = &kt_panel[e * KP..e * KP + valid];
        for j in 0..valid {
            scores[j] = qv.mul_add(kr[j], scores[j]);
        }
    }
    for j in 0..valid {
        scores[j] *= scale;
    }
}

/// 2-tier AVX2 value accumulator:
/// - Tier 1: 64-element primary loop (`8 x ymm`)
/// - Tier 2: 8-element secondary loop (`1 x ymm`) -> handles SigLIP2 `head_dim = 72` (`64 + 8`) with 0 scalar fallback
/// - Tier 3: `< 8` scalar tail safety net
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_avx2(
    acc: *mut f32,
    corr: f32,
    weights: *const f32,
    v_panel: *const f32,
    v_stride: usize,
    valid: usize,
    d: usize,
) {
    use std::arch::x86_64::*;
    let cv = _mm256_set1_ps(corr);
    let mut e = 0usize;

    // [Tier 1] 64-element primary loop (8 x ymm)
    while e + 64 <= d {
        let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e)), cv);
        let mut a1 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 8)), cv);
        let mut a2 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 16)), cv);
        let mut a3 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 24)), cv);
        let mut a4 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 32)), cv);
        let mut a5 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 40)), cv);
        let mut a6 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 48)), cv);
        let mut a7 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e + 56)), cv);

        for j in 0..valid {
            let wv = _mm256_set1_ps(*weights.add(j));
            let vr = v_panel.add(j * v_stride + e);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(8)), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(16)), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(24)), a3);
            a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(32)), a4);
            a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(40)), a5);
            a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(48)), a6);
            a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr.add(56)), a7);
        }

        _mm256_storeu_ps(acc.add(e), a0);
        _mm256_storeu_ps(acc.add(e + 8), a1);
        _mm256_storeu_ps(acc.add(e + 16), a2);
        _mm256_storeu_ps(acc.add(e + 24), a3);
        _mm256_storeu_ps(acc.add(e + 32), a4);
        _mm256_storeu_ps(acc.add(e + 40), a5);
        _mm256_storeu_ps(acc.add(e + 48), a6);
        _mm256_storeu_ps(acc.add(e + 56), a7);
        e += 64;
    }

    // [Tier 2] 8-element secondary loop (1 x ymm) — covers head_dim = 72 (64..72), 80, 96, ...
    while e + 8 <= d {
        let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(e)), cv);
        for j in 0..valid {
            let wv = _mm256_set1_ps(*weights.add(j));
            let vr = v_panel.add(j * v_stride + e);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vr), a0);
        }
        _mm256_storeu_ps(acc.add(e), a0);
        e += 8;
    }

    // [Tier 3] Non-multiple-of-8 scalar safety tail
    if e < d {
        SCALAR_TAIL_HITS.fetch_add(1, Ordering::Relaxed);
        while e < d {
            let mut s = *acc.add(e) * corr;
            for j in 0..valid {
                s = (*weights.add(j)).mul_add(*v_panel.add(j * v_stride + e), s);
            }
            *acc.add(e) = s;
            e += 1;
        }
    }
}

#[inline]
fn accumulate_scalar(
    acc: &mut [f32],
    corr: f32,
    weights: &[f32],
    v_panel: &[f32],
    v_stride: usize,
    valid: usize,
    d: usize,
) {
    for v in &mut acc[..d] {
        *v *= corr;
    }
    for j in 0..valid {
        let w = weights[j];
        let vr = &v_panel[j * v_stride..j * v_stride + d];
        for e in 0..d {
            acc[e] = w.mul_add(vr[e], acc[e]);
        }
    }
}

#[inline]
fn use_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !crate::is_force_scalar()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return true;
        }
    }
    false
}

/// 64-Key panel streaming attention over contiguous `[seq, heads * head_dim]` tensors (single batch).
///
/// - `q`: `[seq_q, num_heads * head_dim]`
/// - `k`: `[seq_kv, num_kv_heads * head_dim]`
/// - `v`: `[seq_kv, num_kv_heads * head_dim]`
/// - `out`: `[seq_q, num_heads * head_dim]`
/// - `key_mask`: optional per-key additive bias `[seq_kv]` (e.g. `0.0` for valid, `-1e9` for pad)
pub fn stream_attention_f32_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
    cfg: &CpuStreamAttnCfg,
    key_mask: Option<&[f32]>,
    out: &mut [f32],
) {
    let hq = cfg.num_heads;
    let hkv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    assert!(hkv > 0 && hq % hkv == 0 && d > 0);
    assert_eq!(q.len(), seq_q * hq * d);
    assert_eq!(k.len(), seq_kv * hkv * d);
    assert_eq!(v.len(), seq_kv * hkv * d);
    assert_eq!(out.len(), seq_q * hq * d);
    if let Some(m) = key_mask {
        assert_eq!(m.len(), seq_kv);
    }
    if seq_q == 0 || seq_kv == 0 {
        out.fill(0.0);
        return;
    }

    let group = hq / hkv;
    let n_panels = seq_kv.div_ceil(KP);
    let q_stride = hq * d;
    let kv_stride = hkv * d;
    let avx2 = use_avx2_fma();

    // 1) Transpose K once per KV head into contiguous 64-key panels:
    //    kt_all[kvh][panel][e * KP + j] = k[(panel*KP + j) * kv_stride + kvh * d + e]
    //    and pack V per KV head into contiguous [seq_kv, d] for L1-friendly streaming.
    let kt_head_elems = n_panels * d * KP;
    let mut kt_all = vec![0.0f32; hkv * kt_head_elems];
    let mut v_all = vec![0.0f32; hkv * seq_kv * d];

    for kvh in 0..hkv {
        let kt_h = &mut kt_all[kvh * kt_head_elems..(kvh + 1) * kt_head_elems];
        let v_h = &mut v_all[kvh * seq_kv * d..(kvh + 1) * seq_kv * d];
        for p in 0..n_panels {
            let j0 = p * KP;
            let valid = KP.min(seq_kv - j0);
            let kt_p = &mut kt_h[p * d * KP..(p + 1) * d * KP];
            for j in 0..valid {
                let k_src = &k[(j0 + j) * kv_stride + kvh * d..(j0 + j) * kv_stride + (kvh + 1) * d];
                let v_src = &v[(j0 + j) * kv_stride + kvh * d..(j0 + j) * kv_stride + (kvh + 1) * d];
                v_h[(j0 + j) * d..(j0 + j + 1) * d].copy_from_slice(v_src);
                for e in 0..d {
                    kt_p[e * KP + j] = k_src[e];
                }
            }
        }
    }

    // 2) Parallelize over (head, q_block)
    let q_blocks = seq_q.div_ceil(QB);
    let total_tasks = hq * q_blocks;
    let out_ptr = SendPtr(out.as_mut_ptr());

    let run_task = |task_idx: usize| {
        let out_ptr = &out_ptr;
        let head = task_idx / q_blocks;
        let qb = task_idx % q_blocks;
        let kvh = head / group;
        let q0 = qb * QB;
        let q1 = (q0 + QB).min(seq_q);

        let kt_h = &kt_all[kvh * kt_head_elems..(kvh + 1) * kt_head_elems];
        let v_h = &v_all[kvh * seq_kv * d..(kvh + 1) * seq_kv * d];

        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            if sc.scores.len() < KP {
                sc.scores.resize(KP, 0.0);
            }
            if sc.weights.len() < KP {
                sc.weights.resize(KP, 0.0);
            }
            if sc.acc.len() < d {
                sc.acc.resize(d, 0.0);
            }
            let TlsScratch {
                scores,
                weights,
                acc,
            } = &mut *sc;

            for qi in q0..q1 {
                let q_row = &q[qi * q_stride + head * d..qi * q_stride + (head + 1) * d];
                let max_k = if cfg.causal {
                    (cfg.past_offset + qi + 1).min(seq_kv)
                } else {
                    seq_kv
                };

                acc[..d].fill(0.0);
                let mut m_curr = f32::NEG_INFINITY;
                let mut l_curr = 0.0f32;

                let active_panels = max_k.div_ceil(KP);
                for p in 0..active_panels {
                    let j0 = p * KP;
                    let valid = KP.min(max_k - j0);
                    let kt_p = &kt_h[p * d * KP..(p + 1) * d * KP];

                    if avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            score_panel_avx2(q_row.as_ptr(), kt_p.as_ptr(), d, cfg.scale, scores.as_mut_ptr());
                        }
                    } else {
                        score_panel_scalar(q_row, kt_p, d, valid, cfg.scale, scores);
                    }

                    let mut p_max = f32::NEG_INFINITY;
                    if let Some(mask) = key_mask {
                        for j in 0..valid {
                            let s = scores[j] + mask[j0 + j];
                            scores[j] = s;
                            if s > p_max {
                                p_max = s;
                            }
                        }
                    } else {
                        for j in 0..valid {
                            let s = scores[j];
                            if s > p_max {
                                p_max = s;
                            }
                        }
                    }

                    let m_new = m_curr.max(p_max);
                    let corr = if m_curr.is_finite() {
                        (m_curr - m_new).exp()
                    } else {
                        0.0
                    };

                    let mut p_sum = 0.0f32;
                    for j in 0..valid {
                        let w = (scores[j] - m_new).exp();
                        weights[j] = w;
                        p_sum += w;
                    }

                    let v_p = &v_h[j0 * d..];
                    if avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            accumulate_avx2(
                                acc.as_mut_ptr(),
                                corr,
                                weights.as_ptr(),
                                v_p.as_ptr(),
                                d,
                                valid,
                                d,
                            );
                        }
                    } else {
                        accumulate_scalar(acc, corr, weights, v_p, d, valid, d);
                    }

                    l_curr = l_curr * corr + p_sum;
                    m_curr = m_new;
                }

                let inv_l = if l_curr > 0.0 { 1.0 / l_curr } else { 0.0 };
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.0.add(qi * q_stride + head * d), d)
                };
                for e in 0..d {
                    dst[e] = acc[e] * inv_l;
                }
            }
        });
    };

    if total_tasks <= 1 || seq_q * seq_kv * hq * d < 65_536 {
        (0..total_tasks).for_each(run_task);
    } else {
        (0..total_tasks).into_par_iter().for_each(run_task);
    }
}

/// 64-Key panel streaming attention without additive key mask (bidirectional ViT & causal GQA LLM).
pub fn stream_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
    cfg: &CpuStreamAttnCfg,
    out: &mut [f32],
) {
    stream_attention_f32_masked(q, k, v, seq_q, seq_kv, cfg, None, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(n: usize, salt: u32) -> Vec<f32> {
        let mut x = 0x9E37_79B9u32 ^ salt.wrapping_mul(2_654_435_761);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn reference_attn(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_q: usize,
        seq_kv: usize,
        cfg: &CpuStreamAttnCfg,
        key_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let (hq, hkv, d) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let group = hq / hkv;
        let mut out = vec![0.0f32; seq_q * hq * d];
        for h in 0..hq {
            let kvh = h / group;
            for qi in 0..seq_q {
                let max_k = if cfg.causal {
                    (cfg.past_offset + qi + 1).min(seq_kv)
                } else {
                    seq_kv
                };
                let mut scores = vec![f32::NEG_INFINITY; max_k];
                let mut mx = f32::NEG_INFINITY;
                for ki in 0..max_k {
                    let mut dot = 0.0f64;
                    for e in 0..d {
                        dot += (q[(qi * hq + h) * d + e] as f64)
                            * (k[(ki * hkv + kvh) * d + e] as f64);
                    }
                    let mut s = (dot * cfg.scale as f64) as f32;
                    if let Some(m) = key_mask {
                        s += m[ki];
                    }
                    scores[ki] = s;
                    mx = mx.max(s);
                }
                let mut denom = 0.0f64;
                let mut exps = vec![0.0f64; max_k];
                for ki in 0..max_k {
                    let w = ((scores[ki] - mx) as f64).exp();
                    exps[ki] = w;
                    denom += w;
                }
                for e in 0..d {
                    let mut acc = 0.0f64;
                    for ki in 0..max_k {
                        acc += exps[ki] * (v[(ki * hkv + kvh) * d + e] as f64);
                    }
                    out[(qi * hq + h) * d + e] = (acc / denom) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn siglip2_head_dim_72_zero_scalar_fallback_and_accurate() {
        // Gate 2-B: SigLIP2-so400m (16 heads x 72 head_dim, seq = 137 > KP=64)
        let cfg = CpuStreamAttnCfg {
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 72,
            scale: 1.0 / (72.0f32).sqrt(),
            causal: false,
            past_offset: 0,
        };
        let seq = 137;
        let q = det(seq * 16 * 72, 10);
        let k = det(seq * 16 * 72, 11);
        let v = det(seq * 16 * 72, 12);
        let mut out = vec![0.0f32; q.len()];

        reset_attn_scalar_tail_hits();
        stream_attention_f32(&q, &k, &v, seq, seq, &cfg, &mut out);
        assert_eq!(
            attn_scalar_tail_hits(),
            0,
            "head_dim=72 (64+8) MUST NOT hit scalar tail fallback!"
        );

        let want = reference_attn(&q, &k, &v, seq, seq, &cfg, None);
        let max_err = out
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "head_dim=72 max_err {max_err}");
    }

    #[test]
    fn gqa_causal_decode_and_prefill_with_key_mask() {
        // GQA causal decode (Hq=8, Hkv=2, head_dim=128, past=70, seq_q=5, seq_kv=75)
        let cfg = CpuStreamAttnCfg {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 128,
            scale: 1.0 / (128.0f32).sqrt(),
            causal: true,
            past_offset: 70,
        };
        let q = det(5 * 8 * 128, 20);
        let k = det(75 * 2 * 128, 21);
        let v = det(75 * 2 * 128, 22);
        let mut out = vec![0.0f32; q.len()];
        stream_attention_f32(&q, &k, &v, 5, 75, &cfg, &mut out);
        let want = reference_attn(&q, &k, &v, 5, 75, &cfg, None);
        let max_err = out
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "gqa causal max_err {max_err}");

        // Masked bidirectional text attention (SigLIP2 text: seq=64, 16 heads x 72)
        let tcfg = CpuStreamAttnCfg {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 72,
            scale: 1.0 / (72.0f32).sqrt(),
            causal: false,
            past_offset: 0,
        };
        let mask: Vec<f32> = (0..64).map(|i| if i < 20 { 0.0 } else { -1e9 }).collect();
        let q2 = det(64 * 4 * 72, 30);
        let k2 = det(64 * 4 * 72, 31);
        let v2 = det(64 * 4 * 72, 32);
        let mut out2 = vec![0.0f32; q2.len()];
        stream_attention_f32_masked(&q2, &k2, &v2, 64, 64, &tcfg, Some(&mask), &mut out2);
        let want2 = reference_attn(&q2, &k2, &v2, 64, 64, &tcfg, Some(&mask));
        let max_err2 = out2
            .iter()
            .zip(want2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err2 < 1e-5, "masked text attn max_err {max_err2}");
    }
}
