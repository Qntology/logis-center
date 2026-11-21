//! Single-pass SIMD HWC `u8` -> normalized `f32` patch extraction (Part 2 / Pack D).
//!
//! Eliminates intermediate full-image CHW `f32` buffers, 6D reshapes, and permute+contiguous copies
//! by folding `(px / 255.0 - mean[c]) / std[c]` into `px * scale[c] + bias[c]` and writing directly
//! into the target patch layout in one L1-hot sweep.

use rayon::prelude::*;

#[inline(always)]
fn norm_row_scalar(
    src: &[u8],
    dst: &mut [f32],
    pixels: usize,
    scale: [f32; 3],
    bias: [f32; 3],
) {
    let (sr, sg, sb) = (scale[0], scale[1], scale[2]);
    let (br, bg, bb) = (bias[0], bias[1], bias[2]);
    for i in 0..pixels {
        let base = i * 3;
        dst[base] = (src[base] as f32).mul_add(sr, br);
        dst[base + 1] = (src[base + 1] as f32).mul_add(sg, bg);
        dst[base + 2] = (src[base + 2] as f32).mul_add(sb, bb);
    }
}

/// Processes 8 interleaved RGB pixels (24 bytes -> 24 floats) at a time with AVX2 FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_row_avx2(
    src: *const u8,
    dst: *mut f32,
    pixels: usize,
    scale: [f32; 3],
    bias: [f32; 3],
) {
    use std::arch::x86_64::*;
    // 24 floats = 3 x 8-lane ymm vectors with repeating (R, G, B) pattern:
    // vec0: [R, G, B, R, G, B, R, G]
    // vec1: [B, R, G, B, R, G, B, R]
    // vec2: [G, B, R, G, B, R, G, B]
    let (sr, sg, sb) = (scale[0], scale[1], scale[2]);
    let (br, bg, bb) = (bias[0], bias[1], bias[2]);
    let s0 = _mm256_setr_ps(sr, sg, sb, sr, sg, sb, sr, sg);
    let s1 = _mm256_setr_ps(sb, sr, sg, sb, sr, sg, sb, sr);
    let s2 = _mm256_setr_ps(sg, sb, sr, sg, sb, sr, sg, sb);
    let b0 = _mm256_setr_ps(br, bg, bb, br, bg, bb, br, bg);
    let b1 = _mm256_setr_ps(bb, br, bg, bb, br, bg, bb, br);
    let b2 = _mm256_setr_ps(bg, bb, br, bg, bb, br, bg, bb);

    let mut i = 0usize;
    while i + 8 <= pixels {
        let p = src.add(i * 3);
        let d = dst.add(i * 3);
        let v0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i)));
        let v1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p.add(8) as *const __m128i)));
        let v2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p.add(16) as *const __m128i)));
        _mm256_storeu_ps(d, _mm256_fmadd_ps(v0, s0, b0));
        _mm256_storeu_ps(d.add(8), _mm256_fmadd_ps(v1, s1, b1));
        _mm256_storeu_ps(d.add(16), _mm256_fmadd_ps(v2, s2, b2));
        i += 8;
    }

    while i < pixels {
        let base = i * 3;
        *dst.add(base) = (*src.add(base) as f32).mul_add(sr, br);
        *dst.add(base + 1) = (*src.add(base + 1) as f32).mul_add(sg, bg);
        *dst.add(base + 2) = (*src.add(base + 2) as f32).mul_add(sb, bb);
        i += 1;
    }
}

#[inline]
fn norm_row(src: &[u8], dst: &mut [f32], pixels: usize, scale: [f32; 3], bias: [f32; 3]) {
    #[cfg(target_arch = "x86_64")]
    {
        if !crate::is_force_scalar()
            && pixels >= 8
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            unsafe {
                norm_row_avx2(src.as_ptr(), dst.as_mut_ptr(), pixels, scale, bias);
            }
            return;
        }
    }
    norm_row_scalar(src, dst, pixels, scale, bias);
}

/// Converts HWC `u8` RGB buffer directly into normalized `f32` patches in a single pass.
///
/// - `merge_size == 1` (SigLIP2 NaFlex, `patch_size = 16`):
///   writes `[gh * gw, patch_size * patch_size * 3]` in `(pr, pc, py, px, c)` order, matching
///   `pixel_values.reshape((1, 3, gh, p, gw, p)).permute(0, 2, 4, 3, 5, 1).reshape((1, gh*gw, p*p*3))`.
/// - `merge_size > 1` (Qwen-VL family, `patch_size = 14, merge_size = 2`):
///   writes `[gh/m * gw/m * m*m, patch_size * patch_size * 3]` in `(mr, mc, sub_r, sub_c, py, px, c)`
///   merge-group order directly.
#[allow(clippy::too_many_arguments)]
pub fn patchify_norm_hwc_u8_f32(
    rgb_hwc: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [f32],
) {
    assert!(patch_size > 0 && merge_size > 0);
    assert!(width % (patch_size * merge_size) == 0);
    assert!(height % (patch_size * merge_size) == 0);
    assert_eq!(rgb_hwc.len(), width * height * 3);

    let gh = height / patch_size;
    let gw = width / patch_size;
    let patch_dim = patch_size * patch_size * 3;
    assert_eq!(out.len(), gh * gw * patch_dim);

    let scale = [
        1.0 / (255.0 * std[0]),
        1.0 / (255.0 * std[1]),
        1.0 / (255.0 * std[2]),
    ];
    let bias = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let row_bytes = width * 3;
    let patch_row_floats = patch_size * 3;

    if merge_size == 1 {
        let process_pr = |(pr, out_pr): (usize, &mut [f32])| {
            let y0 = pr * patch_size;
            for pc in 0..gw {
                let x0 = pc * patch_size;
                let dst_patch = &mut out_pr[pc * patch_dim..(pc + 1) * patch_dim];
                for py in 0..patch_size {
                    let src_off = (y0 + py) * row_bytes + x0 * 3;
                    let dst_off = py * patch_row_floats;
                    norm_row(
                        &rgb_hwc[src_off..src_off + patch_row_floats],
                        &mut dst_patch[dst_off..dst_off + patch_row_floats],
                        patch_size,
                        scale,
                        bias,
                    );
                }
            }
        };
        if gh * gw >= 32 {
            out.par_chunks_mut(gw * patch_dim)
                .enumerate()
                .for_each(process_pr);
        } else {
            out.chunks_mut(gw * patch_dim)
                .enumerate()
                .for_each(process_pr);
        }
    } else {
        let hm = gh / merge_size;
        let wm = gw / merge_size;
        let group_patches = merge_size * merge_size;
        let group_floats = group_patches * patch_dim;
        let process_mr = |(mr, out_mr): (usize, &mut [f32])| {
            for mc in 0..wm {
                let dst_grp = &mut out_mr[mc * group_floats..(mc + 1) * group_floats];
                let mut sub = 0usize;
                for sr in 0..merge_size {
                    let pr = mr * merge_size + sr;
                    let y0 = pr * patch_size;
                    for sc in 0..merge_size {
                        let pc = mc * merge_size + sc;
                        let x0 = pc * patch_size;
                        let dst_patch = &mut dst_grp[sub * patch_dim..(sub + 1) * patch_dim];
                        for py in 0..patch_size {
                            let src_off = (y0 + py) * row_bytes + x0 * 3;
                            let dst_off = py * patch_row_floats;
                            norm_row(
                                &rgb_hwc[src_off..src_off + patch_row_floats],
                                &mut dst_patch[dst_off..dst_off + patch_row_floats],
                                patch_size,
                                scale,
                                bias,
                            );
                        }
                        sub += 1;
                    }
                }
            }
        };
        if hm * wm >= 16 {
            out.par_chunks_mut(wm * group_floats)
                .enumerate()
                .for_each(process_mr);
        } else {
            out.chunks_mut(wm * group_floats)
                .enumerate()
                .for_each(process_mr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_siglip2_matches_scalar_reference() {
        let (w, h, p) = (64usize, 48usize, 16usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let mean = [0.5f32; 3];
        let std = [0.5f32; 3];
        let mut out = vec![0.0f32; (h / p) * (w / p) * p * p * 3];
        patchify_norm_hwc_u8_f32(&rgb, w, h, p, 1, mean, std, &mut out);

        // Compare against reference: (px / 255.0) * 2.0 - 1.0 in (pr, pc, py, px, c) order
        let gh = h / p;
        let gw = w / p;
        let mut idx = 0usize;
        for pr in 0..gh {
            for pc in 0..gw {
                for py in 0..p {
                    for px in 0..p {
                        let src_i = ((pr * p + py) * w + (pc * p + px)) * 3;
                        for c in 0..3 {
                            let want = (rgb[src_i + c] as f32 / 255.0) * 2.0 - 1.0;
                            assert!(
                                (out[idx] - want).abs() < 1e-6,
                                "mismatch at {idx}: {} vs {want}",
                                out[idx]
                            );
                            idx += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn patchify_qwen_vl_merge2_order() {
        let (w, h, p, m) = (56usize, 56usize, 14usize, 2usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 19 + 7) % 256) as u8).collect();
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];
        let mut out_simd = vec![0.0f32; (h / p) * (w / p) * p * p * 3];
        patchify_norm_hwc_u8_f32(&rgb, w, h, p, m, mean, std, &mut out_simd);

        crate::set_force_scalar(true);
        let mut out_scalar = vec![0.0f32; out_simd.len()];
        patchify_norm_hwc_u8_f32(&rgb, w, h, p, m, mean, std, &mut out_scalar);
        crate::set_force_scalar(false);

        for (a, b) in out_simd.iter().zip(out_scalar.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "AVX2 vs scalar patchify bit-exact");
        }
    }
}
