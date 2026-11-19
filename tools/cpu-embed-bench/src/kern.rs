//! Kernels under test: the app's packed GEMM / RMSNorm (crates/embed-cpu-kernels) plus the
//! experimental fused attention below (measured, not adopted: it loses from L ≈ 64 on).
use rayon::prelude::*;

pub use embed_cpu_kernels::*;

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

/// Fused attention for one layer, reading q|k|v straight from the QKV projection [B, L, 3H] and
/// writing [B, L, H] (the transpose(1,2).reshape layout), with RoPE applied on the fly
/// (x·cos + rotate_half(x)·sin, rotate_half = [-x2, x1]). Per (batch, head) task; every
/// reduction (q·k over head_dim, softmax max/sum over j, Σ p·v over j) runs in a fixed order,
/// so a row's output depends only on its own sequence, not on B or the thread count.
pub fn fused_attention(qkv: &[f32], b: usize, l: usize, nh: usize, hd: usize, cos: &[f32], sin: &[f32], out: &mut [f32], threads: usize) {
    let h = nh * hd;
    assert_eq!(qkv.len(), b * l * 3 * h);
    assert_eq!(out.len(), b * l * h);
    let scale = 1.0 / (hd as f64).sqrt() as f32;
    let half = hd / 2;
    let out_ptr = SendPtr(out.as_mut_ptr());
    let body = |task: usize| {
        let out_ptr = &out_ptr;
        let (bi, head) = (task / nh, task % nh);
        let mut q = vec![0f32; l * hd];
        let mut k = vec![0f32; l * hd];
        for t in 0..l {
            let row = &qkv[(bi * l + t) * 3 * h..(bi * l + t + 1) * 3 * h];
            let (qs, ks) = (&row[head * hd..(head + 1) * hd], &row[h + head * hd..h + (head + 1) * hd]);
            let (c, s) = (&cos[t * half..(t + 1) * half], &sin[t * half..(t + 1) * half]);
            for d in 0..hd {
                let (cd, sd) = (c[d % half], s[d % half]);
                let (rq, rk) = if d < half { (-qs[d + half], -ks[d + half]) } else { (qs[d - half], ks[d - half]) };
                q[t * hd + d] = qs[d] * cd + rq * sd;
                k[t * hd + d] = ks[d] * cd + rk * sd;
            }
        }
        let mut p = vec![0f32; l];
        for i in 0..l {
            let qi = &q[i * hd..(i + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..l {
                let kj = &k[j * hd..(j + 1) * hd];
                let mut dot = 0f32;
                for d in 0..hd {
                    dot = qi[d].mul_add(kj[d], dot);
                }
                p[j] = dot * scale;
                mx = mx.max(p[j]);
            }
            let mut sum = 0f32;
            for j in 0..l {
                p[j] = (p[j] - mx).exp();
                sum += p[j];
            }
            let inv = 1.0 / sum;
            let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add((bi * l + i) * h + head * hd), hd) };
            dst.fill(0.0);
            for j in 0..l {
                let pj = p[j] * inv;
                let vj = &qkv[(bi * l + j) * 3 * h + 2 * h + head * hd..(bi * l + j) * 3 * h + 2 * h + (head + 1) * hd];
                for d in 0..hd {
                    dst[d] = pj.mul_add(vj[d], dst[d]);
                }
            }
        }
    };
    if threads <= 1 {
        (0..b * nh).for_each(body);
    } else {
        (0..b * nh).into_par_iter().for_each(body);
    }
}
