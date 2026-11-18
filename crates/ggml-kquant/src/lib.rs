//! 🌟 [PYRAMID-QUANT] GGML 블록 인코더 (Q8_0 · Q6_K · Q5_K · Q4_K).
//!
//! 가중치는 원본 값(F32 · BF16 safetensors)에서 **한 번만** 양자화합니다. 양자화된 파일을 다시 읽어 깎는 경로는 없습니다.
//!
//! 블록 척도는 candle 의 `from_float`(가중치 없는 반복 탐색 `make_qkx1_quants`, Q8_0 은 최댓값/127) 대신
//! 중요도 가중 최소제곱으로 고릅니다 (gravity-ocr `quant.py`, blueprint-sandbox · noto-sandbox 에서 잰 것과 같은 식):
//! - 중요도 `w = rms(블록) + |x|`. 곱의 대부분을 만드는 큰 가중치를 정확히 남깁니다. 열 중요도(활성 제곱 평균,
//!   llama.cpp 의 imatrix)를 주면 곱합니다.
//! - 대칭(Q8_0 · Q6_K): 후보 척도 `(nmax + 0.1 s) / amax` (s = −9 … 9) 를 반올림한 뒤 가중 최소제곱
//!   `d = Σ w x q / Σ w q²` 로 다시 맞추고, 가중 오차가 가장 작은 것을 고릅니다.
//! - 비대칭(Q4_K · Q5_K): 후보 21개 `(nmax − 1 + 0.1 s) / (max − min(min, 0))` (s = 0 … 20) 마다 (척도, 영점)을 가중 최소제곱의
//!   닫힌 해로 풉니다. GGML 은 영점을 음이 아닌 최솟값으로 저장하므로 양의 영점이 나오면 척도만 씁니다.
//! - 슈퍼블록(256) 안의 하위 블록 척도 · 최솟값은 GGML 처럼 F16 슈퍼 척도에 대해 정수로 저장합니다(Q4_K · Q5_K 6비트,
//!   Q6_K 부호 8비트). F16 으로 반올림한 슈퍼 척도 × 단계 수가 가장 큰 하위 척도보다 작으면 F16 한 칸(1 ULP) 올립니다 —
//!   내리면 그 하위 블록이 단계 상한에서 잘립니다. 값은 저장된 척도로 다시 반올림합니다.
//!
//! 비트 배치는 GGML 그대로라 candle 이 장치에서 그대로 곱합니다. 반올림은 짝수 반올림(numpy `rint`)입니다.
//! 앱의 dev 프로필은 최적화하지 않으므로 이 인코더는 별도 크레이트로 두고 opt-level 3 으로 빌드합니다.

use half::f16;
use rayon::prelude::*;

const QK_K: usize = 256;
const QK8_0: usize = 32;
/// 병렬 조각 하나의 원소 수
const PAR_ELEMS: usize = 1 << 16;

/// 이 크레이트가 쓰는 형식
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    F32,
    F16,
    Q8_0,
    Q6K,
    Q5K,
    Q4K,
}

impl Format {
    /// 한 블록의 원소 수
    pub fn block_size(self) -> usize {
        match self {
            Format::F32 | Format::F16 => 1,
            Format::Q8_0 => QK8_0,
            Format::Q6K | Format::Q5K | Format::Q4K => QK_K,
        }
    }

    /// 한 블록의 바이트 수 (GGML `type_size`)
    pub fn type_size(self) -> usize {
        match self {
            Format::F32 => 4,
            Format::F16 => 2,
            Format::Q8_0 => 2 + QK8_0,
            Format::Q6K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            Format::Q5K => 2 + 2 + 12 + QK_K / 8 + QK_K / 2,
            Format::Q4K => 2 + 2 + 12 + QK_K / 2,
        }
    }

    /// 저장 비트 수 / 가중치
    pub fn bits(self) -> f32 {
        self.type_size() as f32 * 8.0 / self.block_size() as f32
    }
}

/// `cols` 열의 행 우선 행렬 `xs` 를 `format` 의 GGML 바이트로 씁니다. `importance` 는 열 중요도입니다(없으면 모두 1).
pub fn encode(xs: &[f32], cols: usize, format: Format, importance: Option<&[f32]>) -> Result<Vec<u8>, String> {
    if cols == 0 || xs.len() % cols != 0 {
        return Err(format!("값 {}개를 {cols}열 행으로 나눌 수 없습니다", xs.len()));
    }
    if cols % format.block_size() != 0 {
        return Err(format!("{cols}열은 {format:?} 블록 {} 의 배수가 아닙니다", format.block_size()));
    }
    if let Some(w) = importance {
        if w.len() != cols {
            return Err(format!("열 {cols}개에 중요도 {}개", w.len()));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("중요도에 음수나 유한하지 않은 값이 있습니다".into());
        }
    }
    Ok(match format {
        Format::F32 => xs.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Format::F16 => xs.iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        _ => {
            let per = (PAR_ELEMS / cols).max(1) * cols;
            let parts: Vec<Vec<u8>> = xs
                .par_chunks(per)
                .map(|rows| {
                    let mut out = Vec::with_capacity(rows.len() / format.block_size() * format.type_size());
                    for row in rows.chunks_exact(cols) {
                        encode_row(row, format, importance, &mut out);
                    }
                    out
                })
                .collect();
            parts.concat()
        }
    })
}

fn encode_row(row: &[f32], format: Format, importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let bs = format.block_size();
    for (b, x) in row.chunks_exact(bs).enumerate() {
        let colw = importance.map(|w| &w[b * bs..(b + 1) * bs]);
        match format {
            Format::Q8_0 => q8_0(x, colw, out),
            Format::Q6K => q6_k(x, colw, out),
            Format::Q5K => q45_k(x, colw, 31, out),
            Format::Q4K => q45_k(x, colw, 15, out),
            Format::F32 | Format::F16 => unreachable!("encode 가 먼저 처리합니다"),
        }
    }
}

/// `rms + |x|` 를 블록 원소마다. 열 중요도가 모두 0 이 아니면 곱합니다.
#[doc(hidden)]
pub fn importance(x: &[f32], colw: Option<&[f32]>, w: &mut [f32]) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    for (wi, xi) in w.iter_mut().zip(x) {
        *wi = rms + xi.abs();
    }
    if let Some(c) = colw {
        let total: f32 = w.iter().zip(c).map(|(a, b)| a * b).sum();
        if total > 0.0 && total.is_finite() {
            w.iter_mut().zip(c).for_each(|(a, b)| *a *= b);
        }
    }
}

/// q ∈ [`qmin`, `nmax`] 반올림의 가중 최소제곱 척도
#[doc(hidden)]
pub fn fit_symmetric(x: &[f32], w: &[f32], nmax: i32, qmin: i32) -> f32 {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    if amax <= 0.0 || !amax.is_finite() {
        return 0.0;
    }
    let (lo, hi) = (qmin as f32, nmax as f32);
    let (mut best_err, mut best_d) = (f32::INFINITY, 0f32);
    for s in -9..=9 {
        let inv = (nmax as f32 + 0.1 * s as f32) / amax;
        let (mut num, mut den) = (0f32, 0f32);
        for (&xi, &wi) in x.iter().zip(w) {
            let q = (xi * inv).round_ties_even().clamp(lo, hi);
            num += wi * xi * q;
            den += wi * q * q;
        }
        let d = if den > 0.0 { num / den } else { 0.0 };
        let mut err = 0f32;
        for (&xi, &wi) in x.iter().zip(w) {
            let q = (xi * inv).round_ties_even().clamp(lo, hi);
            let e = d * q - xi;
            err += wi * e * e;
        }
        if err < best_err {
            best_err = err;
            best_d = d;
        }
    }
    best_d
}

/// `x ≈ d q − m` (q ∈ [0, `nmax`], m ≥ 0) 의 (d, m)
#[doc(hidden)]
pub fn fit_asymmetric(x: &[f32], w: &[f32], nmax: i32) -> (f32, f32) {
    let lo = x.iter().fold(0f32, |a, &v| a.min(v));
    let hi = x.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let range = hi - lo;
    let safe = if range > 0.0 { range } else { 1.0 };
    let sum_w: f32 = w.iter().sum();
    let sum_x: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
    let top = nmax as f32;
    let (mut best_err, mut best_d, mut best_m) = (f32::INFINITY, 0f32, 0f32);
    for s in 0..=20 {
        let inv = (top - 1.0 + 0.1 * s as f32) / safe;
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for (&xi, &wi) in x.iter().zip(w) {
            let q = ((xi - lo) * inv).round_ties_even().clamp(0.0, top);
            sl += wi * q;
            sl2 += wi * q * q;
            sxl += wi * xi * q;
        }
        let det = sum_w * sl2 - sl * sl;
        let scale_only = || if sl2 > 0.0 { sxl / sl2 } else { 0.0 };
        // 반올림이 모두 같으면(상수 블록) 0 이 아닌 단계는 척도만, 모두 0 인 단계는 영점만 씁니다(음의 상수 블록 = −m).
        let (mut d, mut off) = if det > 0.0 {
            ((sum_w * sxl - sum_x * sl) / det, (sl2 * sum_x - sl * sxl) / det)
        } else if sl2 > 0.0 {
            (scale_only(), 0.0)
        } else {
            (0.0, if sum_w > 0.0 { (sum_x / sum_w).min(0.0) } else { 0.0 })
        };
        if off > 0.0 {
            d = scale_only();
            off = 0.0;
        }
        let mut err = 0f32;
        for (&xi, &wi) in x.iter().zip(w) {
            let q = ((xi - lo) * inv).round_ties_even().clamp(0.0, top);
            let e = d * q + off - xi;
            err += wi * e * e;
        }
        if err < best_err {
            best_err = err;
            best_d = d;
            best_m = -off;
        }
    }
    (best_d.max(0.0), best_m.max(0.0))
}

/// 하위 척도(≥ 0)를 F16 슈퍼 척도와 `levels` 단계 정수로. F16 반올림이 내려가면 한 칸 올립니다(1 ULP).
fn super_scale<const N: usize>(v: &[f32; N], levels: u8) -> (f16, [u8; N]) {
    let max = v.iter().fold(0f32, |a, &b| a.max(b));
    let mut idx = [0u8; N];
    if max <= 0.0 || !max.is_finite() {
        return (f16::ZERO, idx);
    }
    let top = levels as f32;
    let mut d = f16::from_f32(max / top);
    if d.to_f32() * top < max {
        d = f16::from_bits(d.to_bits() + 1);
    }
    let df = d.to_f32();
    for (i, &x) in v.iter().enumerate() {
        idx[i] = (x / df).round_ties_even().clamp(0.0, top) as u8;
    }
    (d, idx)
}

/// GGML `get_scale_min_k4`: 12바이트에 묶인 j 번째 6비트 척도와 최솟값
fn scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q8_0: 32개 블록, F16 척도, −127 … 127
fn q8_0(x: &[f32], colw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut w = [0f32; QK8_0];
    importance(x, colw, &mut w);
    let d = f16::from_f32(fit_symmetric(x, &w, 127, -127).max(0.0));
    let df = d.to_f32();
    out.extend_from_slice(&d.to_bits().to_le_bytes());
    for &xi in x {
        let q = if df == 0.0 { 0 } else { (xi / df).round_ties_even().clamp(-127.0, 127.0) as i8 };
        out.push(q as u8);
    }
}

/// Q4_K (`nmax` 15) · Q5_K (`nmax` 31): 슈퍼블록 256 = 32개 블록 8개, 블록마다 6비트 척도와 최솟값
fn q45_k(x: &[f32], colw: Option<&[f32]>, nmax: i32, out: &mut Vec<u8>) {
    let mut w = [0f32; 32];
    let (mut ds, mut ms) = ([0f32; 8], [0f32; 8]);
    for j in 0..8 {
        let xb = &x[32 * j..32 * j + 32];
        importance(xb, colw.map(|c| &c[32 * j..32 * j + 32]), &mut w);
        (ds[j], ms[j]) = fit_asymmetric(xb, &w, nmax);
    }
    let (d, ls) = super_scale(&ds, 63);
    let (dmin, lm) = super_scale(&ms, 63);
    let mut scales = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            scales[j] = ls[j];
            scales[j + 4] = lm[j];
        } else {
            scales[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            scales[j - 4] |= (ls[j] >> 4) << 6;
            scales[j] |= (lm[j] >> 4) << 6;
        }
    }
    // 저장된 척도로 값을 다시 반올림합니다 (candle `to_float` 의 식).
    let (df, dmf) = (d.to_f32(), dmin.to_f32());
    let top = nmax as f32;
    let mut l = [0u8; QK_K];
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &scales);
        let dd = df * sc as f32;
        let dm = dmf * m as f32;
        if dd == 0.0 {
            continue;
        }
        for i in 0..32 {
            l[32 * j + i] = ((x[32 * j + i] + dm) / dd).round_ties_even().clamp(0.0, top) as u8;
        }
    }
    out.extend_from_slice(&d.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&scales);
    let mut qs = [0u8; QK_K / 2];
    if nmax == 31 {
        let mut qh = [0u8; QK_K / 8];
        let (mut m1, mut m2) = (1u8, 2u8);
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let (mut l1, mut l2) = (l[n + j], l[n + j + 32]);
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[n / 2 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
        out.extend_from_slice(&qh);
    } else {
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                qs[n / 2 + j] = l[n + j] | (l[n + j + 32] << 4);
            }
        }
    }
    out.extend_from_slice(&qs);
}

/// Q6_K: 슈퍼블록 256 = 16개 블록 16개, 블록마다 부호 8비트 척도, 값 −32 … 31
fn q6_k(x: &[f32], colw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut w = [0f32; 16];
    let mut ds = [0f32; 16];
    for s in 0..16 {
        let xb = &x[16 * s..16 * s + 16];
        importance(xb, colw.map(|c| &c[16 * s..16 * s + 16]), &mut w);
        ds[s] = fit_symmetric(xb, &w, 31, -32).max(0.0);
    }
    // GGML 처럼 음의 슈퍼 척도로 부호 바이트의 −128 단계까지 씁니다 (블록 척도 = 슈퍼 × 음수 ≥ 0).
    let max = ds.iter().fold(0f32, |a, &b| a.max(b));
    let mut scales = [0i8; 16];
    let mut d = f16::ZERO;
    if max > 0.0 {
        d = f16::from_f32(-max / 128.0);
        if d.to_f32().abs() * 128.0 < max {
            d = f16::from_bits(d.to_bits() + 1);
        }
        let df = d.to_f32();
        for s in 0..16 {
            scales[s] = (ds[s] / df).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
    let df = d.to_f32();
    let mut l = [32u8; QK_K];
    for s in 0..16 {
        let dd = df * scales[s] as f32;
        if dd == 0.0 {
            continue;
        }
        for i in 0..16 {
            let q = (x[16 * s + i] / dd).round_ties_even().clamp(-32.0, 31.0) as i32;
            l[16 * s + i] = (q + 32) as u8;
        }
    }
    let (mut ql, mut qh) = ([0u8; QK_K / 2], [0u8; QK_K / 4]);
    for (half, j) in (0..QK_K).step_by(128).enumerate() {
        for k in 0..32 {
            let (a, b, c, e) = (l[j + k], l[j + k + 32], l[j + k + 64], l[j + k + 96]);
            ql[64 * half + k] = (a & 0xF) | ((c & 0xF) << 4);
            ql[64 * half + k + 32] = (b & 0xF) | ((e & 0xF) << 4);
            qh[32 * half + k] = (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((e >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(scales.iter().map(|s| *s as u8));
    out.extend_from_slice(&d.to_bits().to_le_bytes());
}

/// ‖a − b‖ / ‖b‖
pub fn rel(a: &[f32], b: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}
