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

/// Register tile: MR rows × NR columns of accumulators. NR is the packed panel width; MR is
/// picked per ISA so the accumulators fit the register file (AVX-512: 6×32 = 12 zmm;
/// AVX2 / NEON: 3×32 = 12 ymm / 24 q). The tile shape never changes a value (one FMA chain per element).
pub const NR: usize = 32;
pub const MR_MAX: usize = 6;

/// Rows per register tile on this CPU.
pub fn mr() -> usize {
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

/// W [N, K] repacked once at load into panels of NR output columns, k-major inside a panel:
/// panel p = [K][NR] with element (k, j) = W[p*NR + j][k].
pub struct PackedW {
    pub n: usize,
    pub k: usize,
    pub panels: Vec<f32>,
}

impl PackedW {
    pub fn pack(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        assert!(n % NR == 0, "N={n} must be a multiple of NR={NR}");
        let mut panels = vec![0f32; n * k];
        for p in 0..n / NR {
            let base = p * k * NR;
            for kk in 0..k {
                for j in 0..NR {
                    panels[base + kk * NR + j] = w[(p * NR + j) * k + kk];
                }
            }
        }
        Self { n, k, panels }
    }

    /// Interleave two [N, K] weights (gate | up of a GLU) so one tile computes the matching gate
    /// and up columns together: panel 2p = gate cols, panel 2p+1 = up cols.
    pub fn pack_glu(w: &[f32], i: usize, k: usize) -> Self {
        // w = Wi [2I, K]: rows 0..I gate, I..2I up (embedding.rs Mlp narrow order).
        let gate = Self::pack(&w[..i * k], i, k);
        let up = Self::pack(&w[i * k..], i, k);
        let mut panels = Vec::with_capacity(2 * i * k);
        for p in 0..i / NR {
            panels.extend_from_slice(&gate.panels[p * k * NR..(p + 1) * k * NR]);
            panels.extend_from_slice(&up.panels[p * k * NR..(p + 1) * k * NR]);
        }
        Self { n: 2 * i, k, panels }
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
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { tile_avx512(a, lda, rows, w, k0, k1, acc) };
        }
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
            return unsafe { tile_fma(a, lda, rows, w, k0, k1, acc) };
        }
    }
    tile::<3>(a, lda, rows, w, k0, k1, acc)
}

/// Epilogue applied to a finished MR×NR tile.
#[derive(Clone, Copy)]
pub enum Epilogue {
    /// out[r, col] = acc
    Store,
    /// Paired panels (gate, up): out[r, col] = silu(gate) * up, out has N/2 columns.
    SiluMul,
}

/// out[M, N] (or [M, N/2] for SiluMul) = a[M, K] · Wᵀ. `threads == 1` runs on the caller.
pub fn gemm(a: &[f32], m: usize, w: &PackedW, out: &mut [f32], blk: Blocking, epi: Epilogue, threads: usize) {
    let (k, n) = (w.k, w.n);
    assert_eq!(a.len(), m * k);
    let out_cols = match epi {
        Epilogue::Store => n,
        Epilogue::SiluMul => n / 2,
    };
    assert_eq!(out.len(), m * out_cols);
    let n_panels = n / NR;
    // Task = a group of whole panels (for SiluMul an even number: gate/up pairs stay together).
    let mut per_task = (blk.nc / NR).max(1);
    if matches!(epi, Epilogue::SiluMul) && per_task % 2 == 1 {
        per_task += 1;
    }
    let tasks: Vec<(usize, usize)> = (0..n_panels).step_by(per_task).map(|p| (p, (p + per_task).min(n_panels))).collect();
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
                        Epilogue::Store => {
                            let mut acc = [[0f32; NR]; MR_MAX];
                            let mut k0 = 0;
                            while k0 < k {
                                let k1 = (k0 + kc).min(k);
                                run_tile(a_blk, k, rows, w.panel(p), k0, k1, &mut acc);
                                k0 = k1;
                            }
                            for r in 0..rows {
                                let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add((r0 + r) * out_cols + p * NR), NR) };
                                dst.copy_from_slice(&acc[r]);
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
                            let col = (p / 2) * NR;
                            for r in 0..rows {
                                let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add((r0 + r) * out_cols + col), NR) };
                                for j in 0..NR {
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
    for (row, o) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
        let mut ss = 0f32;
        for v in row {
            ss += v * v;
        }
        let d = (ss / h as f32 + eps).sqrt();
        for j in 0..h {
            o[j] = row[j] / d * w[j];
        }
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
}
