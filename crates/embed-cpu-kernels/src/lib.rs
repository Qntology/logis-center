//! Packed-weight, cache-blocked f32 GEMM for the granite embedding Linear layers (y = x · Wᵀ),
//! plus the one-pass RMSNorm and the fused SiLU(gate)·up epilogue. Used by logis-center
//! `models/embedding/cpu_packed.rs` ([EMBED-CPU-PACKED]) and `tools/cpu-embed-bench`.
//!
//! Bit-identity contract: every output element is one continuous FMA chain over k = 0..K in
//! ascending order, `c = fma(a[k], w[k], c)`, starting from +0.0. KC blocking stores the partial
//! chain and reloads it, which does not change the value. No split-K, no tree reduction. So a row's
//! result depends only on that input row: not on M, the row's position, MC/KC/NC, the thread count,
//! or the ISA (FMA is exactly rounded on x86 FMA3 / AVX-512 and AArch64 alike; the scalar fallback
//! uses `f32::mul_add`).
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub use half;
pub mod attn;
pub mod image;
pub mod q8;
pub use q8::*;

static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);

/// Forces scalar FMA path (for Gate 2-A ISA invariance verification against AVX-512 / AVX2).
pub fn set_force_scalar(force: bool) {
    FORCE_SCALAR.store(force, Ordering::Relaxed);
}

#[inline]
pub fn is_force_scalar() -> bool {
    FORCE_SCALAR.load(Ordering::Relaxed)
}

/// Register tile: MR rows × NR columns of accumulators. NR is the packed panel width; MR is
/// picked per ISA so the accumulators fit the register file (AVX-512: 6×32 = 12 zmm;
/// AVX2 / NEON: 3×32 = 12 ymm / 24 q). The tile shape never changes a value (one FMA chain per element).
pub const NR: usize = 32;
pub const MR_MAX: usize = 6;

/// Rows per register tile on this CPU.
pub fn mr() -> usize {
    if is_force_scalar() {
        return 3;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return 6;
        }
    }
    3
}

#[derive(Clone, Copy, Debug)]
pub struct Blocking {
    /// Rows per cache block of activations (A block MC×KC should stay in L2).
    pub mc: usize,
    /// Depth per block (A micro-panel MR×KC and W panel KC×NR should stay in L1).
    pub kc: usize,
    /// Columns per parallel task (NC/NR weight panels reused for every row of the block).
    pub nc: usize,
}

impl Default for Blocking {
    /// 측정 최적(Linux x86 VM): MC 128 · KC 256 · NC 64. 값은 블로킹과 무관합니다(계약).
    fn default() -> Self {
        Self { mc: 128, kc: 256, nc: 64 }
    }
}

/// W [N_valid, K] repacked once at load into panels of NR output columns, k-major inside a panel:
/// panel p = [K][NR] with element (k, j) = W[p*NR + j][k] (or 0.0 when p*NR + j >= n_valid).
pub struct PackedW {
    /// Padded column count (always a multiple of `NR = 32`).
    pub n: usize,
    /// Logical unpadded output column count (e.g. SigLIP2 MLP 4304 -> padded `n = 4320`).
    pub n_valid: usize,
    pub k: usize,
    pub panels: Vec<f32>,
}

impl PackedW {
    pub fn pack(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        assert!(n % NR == 0, "N={n} must be a multiple of NR={NR}");
        Self::pack_padded(w, n, k)
    }

    /// General-purpose packer allowing arbitrary `n_valid` (e.g. SigLIP2 `4304` -> `4320` with
    /// 16 zero-padded columns). Zero-padded tail columns never affect valid columns.
    pub fn pack_padded(w: &[f32], n_valid: usize, k: usize) -> Self {
        assert_eq!(w.len(), n_valid * k);
        let n_panels = n_valid.div_ceil(NR);
        let n = n_panels * NR;
        let mut panels = vec![0f32; n * k];
        for p in 0..n_panels {
            let base = p * k * NR;
            let col0 = p * NR;
            let cols = NR.min(n_valid - col0);
            for kk in 0..k {
                let dst = &mut panels[base + kk * NR..base + kk * NR + cols];
                for j in 0..cols {
                    dst[j] = w[(col0 + j) * k + kk];
                }
            }
        }
        Self { n, n_valid, k, panels }
    }

    /// Interleave two [I, K] weights (gate | up of a GLU) so one tile computes the matching gate
    /// and up columns together: panel 2p = gate cols, panel 2p+1 = up cols.
    pub fn pack_glu(w: &[f32], i: usize, k: usize) -> Self {
        // w = Wi [2I, K]: rows 0..I gate, I..2I up (embedding.rs Mlp narrow order).
        assert_eq!(w.len(), 2 * i * k);
        Self::pack_glu_pair(&w[..i * k], &w[i * k..], i, k)
    }

    /// Interleaves separate `gate [i_valid, k]` and `up [i_valid, k]` weights with `NR = 32` zero-pad.
    pub fn pack_glu_pair(gate_w: &[f32], up_w: &[f32], i_valid: usize, k: usize) -> Self {
        let gate = Self::pack_padded(gate_w, i_valid, k);
        let up = Self::pack_padded(up_w, i_valid, k);
        let n_panels_each = gate.n / NR;
        let mut panels = Vec::with_capacity(2 * gate.n * k);
        for p in 0..n_panels_each {
            panels.extend_from_slice(&gate.panels[p * k * NR..(p + 1) * k * NR]);
            panels.extend_from_slice(&up.panels[p * k * NR..(p + 1) * k * NR]);
        }
        Self {
            n: 2 * gate.n,
            n_valid: 2 * i_valid,
            k,
            panels,
        }
    }

    fn panel(&self, p: usize) -> &[f32] {
        &self.panels[p * self.k * NR..(p + 1) * self.k * NR]
    }
}

/// One MR×NR tile over k in [k0, k1): rows `rows` (≤ MR) of `a` (row stride lda) against panel `w`.
#[inline(always)]
fn tile<const MR: usize>(a: &[f32], lda: usize, rows: usize, w: &[f32], k0: usize, k1: usize, acc: &mut [[f32; NR]; MR_MAX]) {
    if rows == MR {
        let mut c = [[0f32; NR]; MR];
        for r in 0..MR {
            c[r] = acc[r];
        }
        for kk in k0..k1 {
            let wk: &[f32; NR] = w[kk * NR..kk * NR + NR].try_into().unwrap();
            for r in 0..MR {
                let av = a[r * lda + kk];
                for j in 0..NR {
                    c[r][j] = av.mul_add(wk[j], c[r][j]);
                }
            }
        }
        for r in 0..MR {
            acc[r] = c[r];
        }
    } else {
        for kk in k0..k1 {
            let wk: &[f32; NR] = w[kk * NR..kk * NR + NR].try_into().unwrap();
            for r in 0..rows {
                let av = a[r * lda + kk];
                for j in 0..NR {
                    acc[r][j] = av.mul_add(wk[j], acc[r][j]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_fma(a: &[f32], lda: usize, rows: usize, w: &[f32], k0: usize, k1: usize, acc: &mut [[f32; NR]; MR_MAX]) {
    tile::<3>(a, lda, rows, w, k0, k1, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn tile_avx512(a: &[f32], lda: usize, rows: usize, w: &[f32], k0: usize, k1: usize, acc: &mut [[f32; NR]; MR_MAX]) {
    tile::<6>(a, lda, rows, w, k0, k1, acc)
}

#[inline]
fn run_tile(a: &[f32], lda: usize, rows: usize, w: &[f32], k0: usize, k1: usize, acc: &mut [[f32; NR]; MR_MAX]) {
    if !is_force_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return unsafe { tile_avx512(a, lda, rows, w, k0, k1, acc) };
            }
            if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
                return unsafe { tile_fma(a, lda, rows, w, k0, k1, acc) };
            }
        }
    }
    tile::<3>(a, lda, rows, w, k0, k1, acc)
}

/// Epilogue applied to a finished MR×NR tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// out[r, col] = acc (+ bias)
    Store,
    /// Paired panels (gate, up): out[r, col] = silu(gate) * up, out has n_valid/2 columns.
    SiluMul,
    /// out[r, col] = gelu_pytorch_tanh(acc (+ bias))
    Gelu,
}

#[inline(always)]
fn gelu_tanh(x: f32) -> f32 {
    // PyTorch / candle exact gelu() tanh approximation:
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// out[M, N_valid] (or [M, N_valid/2] for SiluMul) = a[M, K] · Wᵀ. `threads == 1` runs on the caller.
pub fn gemm(a: &[f32], m: usize, w: &PackedW, out: &mut [f32], blk: Blocking, epi: Epilogue, threads: usize) {
    gemm_bias(a, m, w, None, out, blk, epi, threads);
}

/// Generalized packed GEMM with optional `+ bias` epilogue, arbitrary `n_valid` (zero-padded to
/// `NR=32`), and `Store` / `SiluMul` / `Gelu` fused epilogues. Preserves exact ascending FMA chain.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias(
    a: &[f32],
    m: usize,
    w: &PackedW,
    bias: Option<&[f32]>,
    out: &mut [f32],
    blk: Blocking,
    epi: Epilogue,
    threads: usize,
) {
    let (k, n_padded, n_valid) = (w.k, w.n, w.n_valid);
    assert_eq!(a.len(), m * k);
    let out_cols = match epi {
        Epilogue::Store | Epilogue::Gelu => n_valid,
        Epilogue::SiluMul => n_valid / 2,
    };
    assert_eq!(out.len(), m * out_cols);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_cols);
    }
    if m == 0 || out_cols == 0 {
        return;
    }
    let n_panels = n_padded / NR;
    // Task = a group of whole panels (for SiluMul an even number: gate/up pairs stay together).
    let mut per_task = (blk.nc / NR).max(1);
    if matches!(epi, Epilogue::SiluMul) && per_task % 2 == 1 {
        per_task += 1;
    }
    let tasks: Vec<(usize, usize)> = (0..n_panels)
        .step_by(per_task)
        .map(|p| (p, (p + per_task).min(n_panels)))
        .collect();
    let out_ptr = SendPtr(out.as_mut_ptr());
    let mr = mr();
    let body = |&(p0, p1): &(usize, usize)| {
        let out_ptr = &out_ptr;
        let kc = blk.kc.max(1);
        let mut m0 = 0;
        while m0 < m {
            let m1 = (m0 + blk.mc.max(mr)).min(m);
            let mut r0 = m0;
            while r0 < m1 {
                let rows = mr.min(m1 - r0);
                let a_blk = &a[r0 * k..];
                let mut p = p0;
                while p < p1 {
                    match epi {
                        Epilogue::Store | Epilogue::Gelu => {
                            let mut acc = [[0f32; NR]; MR_MAX];
                            let mut k0 = 0;
                            while k0 < k {
                                let k1 = (k0 + kc).min(k);
                                run_tile(a_blk, k, rows, w.panel(p), k0, k1, &mut acc);
                                k0 = k1;
                            }
                            let col0 = p * NR;
                            let valid = NR.min(out_cols.saturating_sub(col0));
                            let is_gelu = matches!(epi, Epilogue::Gelu);
                            for r in 0..rows {
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        out_ptr.0.add((r0 + r) * out_cols + col0),
                                        valid,
                                    )
                                };
                                match (bias, is_gelu) {
                                    (None, false) => dst.copy_from_slice(&acc[r][..valid]),
                                    (Some(b), false) => {
                                        let bs = &b[col0..col0 + valid];
                                        for j in 0..valid {
                                            dst[j] = acc[r][j] + bs[j];
                                        }
                                    }
                                    (None, true) => {
                                        for j in 0..valid {
                                            dst[j] = gelu_tanh(acc[r][j]);
                                        }
                                    }
                                    (Some(b), true) => {
                                        let bs = &b[col0..col0 + valid];
                                        for j in 0..valid {
                                            dst[j] = gelu_tanh(acc[r][j] + bs[j]);
                                        }
                                    }
                                }
                            }
                            p += 1;
                        }
                        Epilogue::SiluMul => {
                            let mut g = [[0f32; NR]; MR_MAX];
                            let mut u = [[0f32; NR]; MR_MAX];
                            let mut k0 = 0;
                            while k0 < k {
                                let k1 = (k0 + kc).min(k);
                                run_tile(a_blk, k, rows, w.panel(p), k0, k1, &mut g);
                                run_tile(a_blk, k, rows, w.panel(p + 1), k0, k1, &mut u);
                                k0 = k1;
                            }
                            let col0 = (p / 2) * NR;
                            let valid = NR.min(out_cols.saturating_sub(col0));
                            for r in 0..rows {
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        out_ptr.0.add((r0 + r) * out_cols + col0),
                                        valid,
                                    )
                                };
                                for j in 0..valid {
                                    let x = g[r][j];
                                    dst[j] = x / (1.0 + (-x).exp()) * u[r][j];
                                }
                            }
                            p += 2;
                        }
                    }
                }
                r0 += rows;
            }
            m0 = m1;
        }
    };
    if threads <= 1 || tasks.len() == 1 {
        tasks.iter().for_each(body);
    } else {
        tasks.par_iter().for_each(body);
    }
}

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

/// RMSNorm of each row (embedding.rs RmsNorm in F32: x / sqrt(mean(x²) + eps) * w), in one pass.
pub fn rms_norm(x: &[f32], h: usize, w: &[f32], eps: f32, out: &mut [f32]) {
    let f = |(row, o): (&[f32], &mut [f32])| {
        let mut ss = 0f32;
        for v in row {
            ss += v * v;
        }
        let d = (ss / h as f32 + eps).sqrt();
        for j in 0..h {
            o[j] = row[j] / d * w[j];
        }
    };
    if x.len() < PAR_MIN_ELEMS {
        x.chunks_exact(h).zip(out.chunks_exact_mut(h)).for_each(f);
    } else {
        x.par_chunks_exact(h).zip(out.par_chunks_exact_mut(h)).for_each(f);
    }
}

/// 1-pass RMSNorm with `(1 + w)` offset weight (Qwen3.5 / Qwen4-Exp).
pub fn rms_norm_1p(x: &[f32], h: usize, w: &[f32], eps: f32, out: &mut [f32]) {
    let f = |(row, o): (&[f32], &mut [f32])| {
        let mut ss = 0f32;
        for v in row {
            ss += v * v;
        }
        let inv = 1.0 / (ss / h as f32 + eps).sqrt();
        for j in 0..h {
            o[j] = row[j] * inv * (1.0 + w[j]);
        }
    };
    if x.len() < PAR_MIN_ELEMS {
        x.chunks_exact(h).zip(out.chunks_exact_mut(h)).for_each(f);
    } else {
        x.par_chunks_exact(h).zip(out.par_chunks_exact_mut(h)).for_each(f);
    }
}

/// 1-pass per-row LayerNorm (`(x - mean) / sqrt(var + eps) * w + bias`) for SigLIP2 CPU path.
pub fn layer_norm(x: &[f32], h: usize, w: &[f32], bias: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    assert_eq!(w.len(), h);
    if let Some(b) = bias {
        assert_eq!(b.len(), h);
    }
    let inv_h = 1.0 / h as f32;
    let f = |(row, o): (&[f32], &mut [f32])| {
        let mut sum = 0f32;
        for &v in row {
            sum += v;
        }
        let mean = sum * inv_h;
        let mut var = 0f32;
        for &v in row {
            let d = v - mean;
            var = d.mul_add(d, var);
        }
        let inv_std = 1.0 / (var * inv_h + eps).sqrt();
        match bias {
            Some(b) => {
                for j in 0..h {
                    o[j] = ((row[j] - mean) * inv_std).mul_add(w[j], b[j]);
                }
            }
            None => {
                for j in 0..h {
                    o[j] = ((row[j] - mean) * inv_std) * w[j];
                }
            }
        }
    };
    if x.len() < PAR_MIN_ELEMS {
        x.chunks_exact(h).zip(out.chunks_exact_mut(h)).for_each(f);
    } else {
        x.par_chunks_exact(h).zip(out.par_chunks_exact_mut(h)).for_each(f);
    }
}

/// 1-pass PyTorch tanh-approximation GELU in place.
pub fn gelu_tanh_inplace(x: &mut [f32]) {
    if x.len() < PAR_MIN_ELEMS {
        for v in x.iter_mut() {
            *v = gelu_tanh(*v);
        }
    } else {
        x.par_chunks_mut(PAR_CHUNK).for_each(|c| {
            for v in c.iter_mut() {
                *v = gelu_tanh(*v);
            }
        });
    }
}

/// Fast SIMD dot product for host-side REST vocabulary rescue (`TrimmedHead::RestRescueGuard`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 16)), _mm256_loadu_ps(b.add(i + 16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 24)), _mm256_loadu_ps(b.add(i + 24)), s3);
        i += 32;
    }
    let mut s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    while i + 8 <= n {
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s);
        i += 8;
    }
    let lo = _mm256_castps256_ps128(s);
    let hi = _mm256_extractf128_ps(s, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < n {
        total = (*a.add(i)).mul_add(*b.add(i), total);
        i += 1;
    }
    total
}

#[inline]
pub fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if !is_force_scalar()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_f32_avx2(a.as_ptr(), b.as_ptr(), n) };
        }
    }
    let mut s = 0.0f32;
    for i in 0..n {
        s = a[i].mul_add(b[i], s);
    }
    s
}

/// Parallel SIMD dot + deterministic tie-break argmax over dense host rows `[n_rows, d]`.
/// Used by Part 5 `TrimmedHead::RestRescueGuard` when `rescue_mask == 1` triggers.
pub fn dot_argmax_rows_f32(x: &[f32], rows: &[f32], n_rows: usize, d: usize) -> Option<(usize, f32)> {
    if n_rows == 0 || d == 0 {
        return None;
    }
    assert_eq!(x.len(), d);
    assert_eq!(rows.len(), n_rows * d);
    let better = |a: (usize, f32), b: (usize, f32)| -> (usize, f32) {
        if a.1 > b.1 || (a.1 == b.1 && a.0 < b.0) {
            a
        } else {
            b
        }
    };
    if n_rows < 256 {
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let v = dot_f32_simd(x, &rows[i * d..(i + 1) * d]);
            best = better((i, v), best);
        }
        Some(best)
    } else {
        Some(
            (0..n_rows)
                .into_par_iter()
                .map(|i| (i, dot_f32_simd(x, &rows[i * d..(i + 1) * d])))
                .reduce(|| (usize::MAX, f32::NEG_INFINITY), better),
        )
    }
}


// ---------------------------------------------------------------------------------------------
// 🌟 [EMBED-CPU-PAR] candle 의 원소별 연산과 **같은 식**을 여러 코어에서 계산합니다 (candle 경로용).
// candle CPU 의 단항·이항 연산은 한 스레드에서 돌아, 순전파의 norm(powf) · GLU(silu · 곱)가
// 3~4코어에서 25~30% 를 차지했습니다. 원소마다 같은 f32 식을 같은 순서로 계산하므로 결과 비트가
// candle 연산과 같습니다 (스레드 분할은 원소 경계에서만, 리덕션 없음). 이 크레이트는 dev 프로필에서도
// opt-level 3 이라 앱 크레이트(debug 테스트 빌드는 opt-level 0)에 두지 않습니다.
// ---------------------------------------------------------------------------------------------

/// 이보다 작은 입력은 한 스레드로 계산합니다(같은 식, 스레드 전환 비용만 아낌).
pub const PAR_MIN_ELEMS: usize = 1 << 14;
const PAR_CHUNK: usize = 1 << 12;

/// `out[i] = src[i].powf(2.0)` — candle `Tensor::powf(2.0)` (CPU f32: `v.powf(e as f32)`, 지수가 실행 시간 값이라
/// libm `powf` 호출) 와 같은 비트. 지수를 상수로 쓰면 LLVM 이 `pow(x, 2.0)` 을 `x * x` 로 바꾸는데, libm `powf` 는
/// 드물게 1 ULP 다르므로(실측) `black_box` 로 같은 libm 호출을 유지합니다.
pub fn powf2(src: &[f32], out: &mut [f32]) {
    assert_eq!(src.len(), out.len());
    let e = std::hint::black_box(2.0f32);
    let f = |o: &mut [f32], s: &[f32]| {
        for (o, v) in o.iter_mut().zip(s) {
            *o = v.powf(e);
        }
    };
    if src.len() < PAR_MIN_ELEMS {
        f(out, src);
    } else {
        out.par_chunks_mut(PAR_CHUNK).zip(src.par_chunks(PAR_CHUNK)).for_each(|(o, s)| f(o, s));
    }
}

/// GLU: 행마다 `x = [gate (i) | up (i)]` 에 대해 `out = silu(gate) * up`.
/// candle `silu` (CPU f32: `v / (1.0 + (-v).exp())`) 뒤 곱 — 같은 비트, narrow 복사 없음.
pub fn silu_mul_halves(src: &[f32], i: usize, out: &mut [f32]) {
    assert!(i > 0 && src.len() % (2 * i) == 0 && out.len() * 2 == src.len());
    let f = |o: &mut [f32], r: &[f32]| {
        let (g, u) = r.split_at(i);
        for ((o, g), u) in o.iter_mut().zip(g).zip(u) {
            *o = (g / (1.0 + (-g).exp())) * u;
        }
    };
    if src.len() < PAR_MIN_ELEMS {
        out.chunks_mut(i).zip(src.chunks(2 * i)).for_each(|(o, r)| f(o, r));
    } else {
        out.par_chunks_mut(i).zip(src.par_chunks(2 * i)).for_each(|(o, r)| f(o, r));
    }
}

// ---------------------------------------------------------------------------------------------
// 🌟 [VISION-SCORE] 패치 × 구 내적 행렬 (SigLIP2 비전 채점의 배치 경로)
// 비전 파이프라인은 패치(≤256) × 앵커 구(수백) 쌍마다 1152차원 내적을 `a.iter().zip(b).map(|(x, y)| x * y)
// .sum::<f32>()` 로 하나씩 계산했습니다. 이 합은 누산기 하나의 순차 체인이라 벡터화되지 않습니다.
// 여기서는 원소마다 **같은 식**(시작값 -0.0 — `f32` 의 `Sum` 과 같음, k 오름차순, 곱을 반올림한 뒤 더하기,
// FMA 아님)을 유지하고, 서로 독립인 열(구) 방향으로만 벡터화하고 행(패치) 방향으로 나눠 병렬 계산합니다.
// 그래서 결과 비트가 기존 스칼라 내적과 같고, 장치(CPU · Metal · Vulkan · CUDA · ROCm)와 무관합니다.
// ---------------------------------------------------------------------------------------------

/// 열 타일 폭. 누산기 한 묶음이 레지스터에 남도록 작게 둡니다.
const DOT_TILE: usize = 32;

/// `out[i * m + j] = Σ_k a[i * d + k] * bt[k * m + j]` (k 오름차순, 시작값 -0.0, 곱 뒤 덧셈).
/// `a` 는 [n, d] 행 우선, `bt` 는 [d, m] (열이 구 하나). 행마다 결과가 `a` 의 i 행과
/// `bt` 의 j 열을 `iter().zip().map(|(x, y)| x * y).sum::<f32>()` 한 값과 비트 동일합니다.
pub fn dot_rows_cols(a: &[f32], n: usize, d: usize, bt: &[f32], m: usize, out: &mut [f32]) {
    assert_eq!(a.len(), n * d);
    assert_eq!(bt.len(), d * m);
    assert_eq!(out.len(), n * m);
    if n == 0 || m == 0 {
        return;
    }
    let row = |i: usize, o: &mut [f32]| {
        let ar = &a[i * d..(i + 1) * d];
        let mut j0 = 0;
        while j0 < m {
            let w = DOT_TILE.min(m - j0);
            let mut acc = [-0.0f32; DOT_TILE];
            if w == DOT_TILE {
                for (k, &x) in ar.iter().enumerate() {
                    let b = &bt[k * m + j0..k * m + j0 + DOT_TILE];
                    for t in 0..DOT_TILE {
                        acc[t] = acc[t] + x * b[t];
                    }
                }
            } else {
                for (k, &x) in ar.iter().enumerate() {
                    let b = &bt[k * m + j0..k * m + j0 + w];
                    for t in 0..w {
                        acc[t] = acc[t] + x * b[t];
                    }
                }
            }
            o[j0..j0 + w].copy_from_slice(&acc[..w]);
            j0 += w;
        }
    };
    if n * m * d < PAR_MIN_ELEMS * 16 {
        out.chunks_mut(m).enumerate().for_each(|(i, o)| row(i, o));
    } else {
        out.par_chunks_mut(m).enumerate().for_each(|(i, o)| row(i, o));
    }
}

/// `v` 의 L2 노름을 `v.iter().map(|x| x * x).sum::<f32>().sqrt()` 와 같은 식으로.
pub fn l2_norm_seq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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

    /// The contract written out: one FMA chain per element, k ascending, from +0.0.
    fn reference(a: &[f32], m: usize, w: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc = a[r * k + kk].mul_add(w[c * k + kk], acc);
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn gemm_equals_fma_chain_for_every_m_blocking_and_thread_count() {
        let (k, n, mmax) = (200usize, 96usize, 37usize);
        let a = det(mmax * k, 1);
        let w = det(n * k, 2);
        let pw = PackedW::pack(&w, n, k);
        let want = bits(&reference(&a, mmax, &w, n, k));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let blockings = [
            Blocking::default(),
            Blocking { mc: 1, kc: 1, nc: NR },
            Blocking { mc: 7, kc: 33, nc: 2 * NR },
            Blocking { mc: 1 << 20, kc: 1 << 20, nc: 1 << 20 },
        ];
        for m in 1..=mmax {
            for blk in blockings {
                for threads in [1usize, 4] {
                    let mut out = vec![0f32; m * n];
                    pool.install(|| gemm(&a[..m * k], m, &pw, &mut out, blk, Epilogue::Store, threads));
                    assert_eq!(bits(&out), want[..m * n], "M={m} {blk:?} threads={threads}");
                }
            }
        }
        // Row position: reversed rows give the same per-row bits.
        let ar: Vec<f32> = (0..mmax).rev().flat_map(|r| a[r * k..(r + 1) * k].to_vec()).collect();
        let mut out = vec![0f32; mmax * n];
        gemm(&ar, mmax, &pw, &mut out, Blocking::default(), Epilogue::Store, 1);
        for r in 0..mmax {
            assert_eq!(bits(&out[(mmax - 1 - r) * n..(mmax - r) * n]), want[r * n..(r + 1) * n]);
        }
    }

    #[test]
    fn silu_mul_epilogue_matches_unfused() {
        let (k, i, m) = (64usize, 64usize, 9usize);
        let a = det(m * k, 3);
        let wi = det(2 * i * k, 4); // rows 0..i gate, i..2i up
        let pw = PackedW::pack_glu(&wi, i, k);
        let full = reference(&a, m, &wi, 2 * i, k);
        let mut out = vec![0f32; m * i];
        gemm(&a, m, &pw, &mut out, Blocking { mc: 4, kc: 17, nc: NR }, Epilogue::SiluMul, 1);
        for r in 0..m {
            for c in 0..i {
                let g = full[r * 2 * i + c];
                let u = full[r * 2 * i + i + c];
                let want = g / (1.0 + (-g).exp()) * u;
                assert_eq!(out[r * i + c].to_bits(), want.to_bits(), "r={r} c={c}");
            }
        }
    }

    #[test]
    fn rms_norm_rows_are_independent() {
        let h = 96;
        let x = det(5 * h, 5);
        let w = det(h, 6);
        let mut all = vec![0f32; 5 * h];
        rms_norm(&x, h, &w, 1e-5, &mut all);
        for r in 0..5 {
            let mut one = vec![0f32; h];
            rms_norm(&x[r * h..(r + 1) * h], h, &w, 1e-5, &mut one);
            assert_eq!(bits(&one), bits(&all[r * h..(r + 1) * h]));
        }
    }

    #[test]
    fn dot_rows_cols_equals_scalar_sum_bits() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for &(n, d, m) in &[(1usize, 1usize, 1usize), (3, 7, 5), (17, 1152, 33), (64, 1152, 96), (256, 1152, 379), (2, 31, 64)] {
            let mut a = det(n * d, (n * 31 + d) as u32);
            let mut b = det(m * d, (m * 7 + d) as u32 + 99); // [m, d] 행 = 구 하나
            // 0 · -0 · 큰 값 · 아주 작은 값 (부호 있는 0 과 넘침 경계)
            if a.len() > 4 { a[0] = 0.0; a[1] = -0.0; a[2] = 1e30; a[3] = 1e-30; }
            if b.len() > 4 { b[0] = -0.0; b[1] = 0.0; b[2] = 1e10; b[3] = -1e-30; }
            if n > 1 { for k in 0..d { a[d + k] = -0.0; } } // 전부 -0 인 행
            let mut bt = vec![0f32; d * m];
            for j in 0..m { for k in 0..d { bt[k * m + j] = b[j * d + k]; } }
            let mut out = vec![0f32; n * m];
            pool.install(|| dot_rows_cols(&a, n, d, &bt, m, &mut out));
            for i in 0..n {
                for j in 0..m {
                    let want: f32 = a[i * d..(i + 1) * d].iter().zip(b[j * d..(j + 1) * d].iter()).map(|(x, y)| x * y).sum();
                    assert_eq!(out[i * m + j].to_bits(), want.to_bits(), "n={n} d={d} m={m} i={i} j={j}");
                }
            }
        }
    }

    #[test]
    fn non_aligned_4304_padded_gemm_and_isa_bit_exactness() {
        // Gate 2-A & 2-B: SigLIP2 MLP 1152 -> 4304 (4304 = 32*134 + 16 -> 4320 zero-pad)
        // Verify M=1 vs M=5, threads=1 vs threads=4, SIMD vs FORCE_SCALAR are 100% bit-identical!
        let (k, n_valid) = (64usize, 4304usize);
        let a = det(5 * k, 41);
        let w = det(n_valid * k, 42);
        let b = det(n_valid, 43);
        let pw = PackedW::pack_padded(&w, n_valid, k);
        assert_eq!(pw.n, 4320);
        assert_eq!(pw.n_valid, 4304);

        let mut out_simd = vec![0f32; 5 * n_valid];
        gemm_bias(&a, 5, &pw, Some(&b), &mut out_simd, Blocking::default(), Epilogue::Store, 4);

        // M=1 single row must match row 0 of M=5 bit-for-bit
        let mut out_m1 = vec![0f32; n_valid];
        gemm_bias(&a[..k], 1, &pw, Some(&b), &mut out_m1, Blocking::default(), Epilogue::Store, 1);
        assert_eq!(bits(&out_m1), bits(&out_simd[..n_valid]));

        // FORCE_SCALAR must match AVX2/AVX-512 bit-for-bit!
        set_force_scalar(true);
        let mut out_scalar = vec![0f32; 5 * n_valid];
        gemm_bias(&a, 5, &pw, Some(&b), &mut out_scalar, Blocking::default(), Epilogue::Store, 1);
        set_force_scalar(false);
        assert_eq!(bits(&out_simd), bits(&out_scalar));

        // Fused Gelu epilogue matches Store + gelu_tanh_inplace bit-for-bit
        let mut out_gelu = vec![0f32; 5 * n_valid];
        gemm_bias(&a, 5, &pw, Some(&b), &mut out_gelu, Blocking::default(), Epilogue::Gelu, 4);
        let mut ref_gelu = out_simd.clone();
        gelu_tanh_inplace(&mut ref_gelu);
        assert_eq!(bits(&out_gelu), bits(&ref_gelu));
    }

    #[test]
    fn layer_norm_and_dot_argmax_rows() {
        let (rows, h) = (4usize, 1152usize);
        let x = det(rows * h, 50);
        let w = det(h, 51);
        let b = det(h, 52);
        let mut out = vec![0f32; rows * h];
        layer_norm(&x, h, &w, Some(&b), 1e-6, &mut out);
        // Check row 0 vs standalone
        let mut one = vec![0f32; h];
        layer_norm(&x[..h], h, &w, Some(&b), 1e-6, &mut one);
        assert_eq!(bits(&one), bits(&out[..h]));

        // dot_argmax_rows_f32 deterministic tie-break
        let q = vec![1.0f32; 64];
        let mut mat = vec![0.0f32; 512 * 64];
        for &win in &[17usize, 200, 499] {
            for j in 0..64 {
                mat[win * 64 + j] = 2.0;
            }
        }
        let (idx, val) = dot_argmax_rows_f32(&q, &mat, 512, 64).unwrap();
        assert_eq!(idx, 17);
        assert!((val - 128.0).abs() < 1e-5);
    }
}
