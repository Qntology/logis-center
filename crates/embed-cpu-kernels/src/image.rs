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

/// Converts a single RGB HWC `u8` image directly into Qwen-VL / Qwen3-VL patched `f32` layout
/// `[grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size]` in one host pass,
/// replicating the single frame `temporal_patch_size` times along the temporal axis.
///
/// Exact layout order per patch:
/// `(mr, mc, sr, sc, channel, temporal_idx, py, px)`, matching:
/// `img_chw.repeat(temporal_patch_size).reshape((1, t_p, 3, gh/m, m, p, gw/m, m, p))
///  .permute((0, 3, 6, 4, 7, 2, 1, 5, 8)).reshape((gh*gw, 3*t_p*p*p))`.
#[allow(clippy::too_many_arguments)]
pub fn patchify_norm_qwen_vl_u8_f32(
    rgb_hwc: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [f32],
) {
    assert!(patch_size > 0 && merge_size > 0 && temporal_patch_size > 0);
    assert!(width % (patch_size * merge_size) == 0);
    assert!(height % (patch_size * merge_size) == 0);
    assert_eq!(rgb_hwc.len(), width * height * 3);

    let gh = height / patch_size;
    let gw = width / patch_size;
    let pp = patch_size * patch_size;
    let chan_stride = temporal_patch_size * pp;
    let patch_dim = 3 * chan_stride;
    assert_eq!(out.len(), gh * gw * patch_dim);

    let scale = [
        1.0 / (255.0 * std[0]),
        1.0 / (255.0 * std[1]),
        1.0 / (255.0 * std[2]),
    ];
    let bias = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let row_bytes = width * 3;
    let hm = gh / merge_size;
    let wm = gw / merge_size;
    let group_floats = merge_size * merge_size * patch_dim;

    let process_mr = |(mr, out_mr): (usize, &mut [f32])| {
        for mc in 0..wm {
            let dst_grp = &mut out_mr[mc * group_floats..(mc + 1) * group_floats];
            let mut sub = 0usize;
            for sr in 0..merge_size {
                let y0 = (mr * merge_size + sr) * patch_size;
                for sc in 0..merge_size {
                    let x0 = (mc * merge_size + sc) * patch_size;
                    let dst_patch = &mut dst_grp[sub * patch_dim..(sub + 1) * patch_dim];
                    // Fill t=0 plane for R, G, B in a single L1-hot pass over the patch pixels.
                    for py in 0..patch_size {
                        let src_row = (y0 + py) * row_bytes + x0 * 3;
                        let dst_row = py * patch_size;
                        for px in 0..patch_size {
                            let sb = src_row + px * 3;
                            let r = (rgb_hwc[sb] as f32).mul_add(scale[0], bias[0]);
                            let g = (rgb_hwc[sb + 1] as f32).mul_add(scale[1], bias[1]);
                            let b = (rgb_hwc[sb + 2] as f32).mul_add(scale[2], bias[2]);
                            dst_patch[dst_row + px] = r;
                            dst_patch[chan_stride + dst_row + px] = g;
                            dst_patch[2 * chan_stride + dst_row + px] = b;
                        }
                    }
                    // Replicate t=0 plane to t=1..temporal_patch_size for each channel.
                    for c in 0..3 {
                        let c_base = c * chan_stride;
                        for t in 1..temporal_patch_size {
                            dst_patch.copy_within(c_base..c_base + pp, c_base + t * pp);
                        }
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

/// Generic half-precision host patchifier (`half::f16` / `half::bf16`) for Qwen-VL / Qwen3-VL.
/// Halves PCIe H2D upload bandwidth and removes the intermediate GPU F32->F16/BF16 cast copy
/// (`P4-F17` / `P4-O05`).
#[allow(clippy::too_many_arguments)]
fn patchify_norm_qwen_vl_u8_half<T: Copy + Send + Sync>(
    rgb_hwc: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [T],
    conv: fn(f32) -> T,
) {
    assert!(patch_size > 0 && merge_size > 0 && temporal_patch_size > 0);
    assert!(width % (patch_size * merge_size) == 0);
    assert!(height % (patch_size * merge_size) == 0);
    assert_eq!(rgb_hwc.len(), width * height * 3);

    let gh = height / patch_size;
    let gw = width / patch_size;
    let pp = patch_size * patch_size;
    let chan_stride = temporal_patch_size * pp;
    let patch_dim = 3 * chan_stride;
    assert_eq!(out.len(), gh * gw * patch_dim);

    let scale = [
        1.0 / (255.0 * std[0]),
        1.0 / (255.0 * std[1]),
        1.0 / (255.0 * std[2]),
    ];
    let bias = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let row_bytes = width * 3;
    let hm = gh / merge_size;
    let wm = gw / merge_size;
    let group_elems = merge_size * merge_size * patch_dim;

    let process_mr = |(mr, out_mr): (usize, &mut [T])| {
        for mc in 0..wm {
            let dst_grp = &mut out_mr[mc * group_elems..(mc + 1) * group_elems];
            let mut sub = 0usize;
            for sr in 0..merge_size {
                let y0 = (mr * merge_size + sr) * patch_size;
                for sc in 0..merge_size {
                    let x0 = (mc * merge_size + sc) * patch_size;
                    let dst_patch = &mut dst_grp[sub * patch_dim..(sub + 1) * patch_dim];
                    for py in 0..patch_size {
                        let src_row = (y0 + py) * row_bytes + x0 * 3;
                        let dst_row = py * patch_size;
                        for px in 0..patch_size {
                            let sb = src_row + px * 3;
                            let r = (rgb_hwc[sb] as f32).mul_add(scale[0], bias[0]);
                            let g = (rgb_hwc[sb + 1] as f32).mul_add(scale[1], bias[1]);
                            let b = (rgb_hwc[sb + 2] as f32).mul_add(scale[2], bias[2]);
                            dst_patch[dst_row + px] = conv(r);
                            dst_patch[chan_stride + dst_row + px] = conv(g);
                            dst_patch[2 * chan_stride + dst_row + px] = conv(b);
                        }
                    }
                    for c in 0..3 {
                        let c_base = c * chan_stride;
                        for t in 1..temporal_patch_size {
                            dst_patch.copy_within(c_base..c_base + pp, c_base + t * pp);
                        }
                    }
                    sub += 1;
                }
            }
        }
    };

    if hm * wm >= 16 {
        out.par_chunks_mut(wm * group_elems)
            .enumerate()
            .for_each(process_mr);
    } else {
        out.chunks_mut(wm * group_elems)
            .enumerate()
            .for_each(process_mr);
    }
}

/// Converts a single RGB HWC `u8` image directly into Qwen-VL / Qwen3-VL patched `f16` layout.
#[allow(clippy::too_many_arguments)]
pub fn patchify_norm_qwen_vl_u8_f16(
    rgb_hwc: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [half::f16],
) {
    patchify_norm_qwen_vl_u8_half(
        rgb_hwc,
        width,
        height,
        patch_size,
        merge_size,
        temporal_patch_size,
        mean,
        std,
        out,
        half::f16::from_f32,
    );
}

/// Converts a single RGB HWC `u8` image directly into Qwen-VL / Qwen3-VL patched `bf16` layout.
#[allow(clippy::too_many_arguments)]
pub fn patchify_norm_qwen_vl_u8_bf16(
    rgb_hwc: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [half::bf16],
) {
    patchify_norm_qwen_vl_u8_half(
        rgb_hwc,
        width,
        height,
        patch_size,
        merge_size,
        temporal_patch_size,
        mean,
        std,
        out,
        half::bf16::from_f32,
    );
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

    #[test]
    fn patchify_qwen_vl_matches_chwt_reference() {
        let (w, h, p, m, tp) = (56usize, 28usize, 14usize, 2usize, 2usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 29 + 11) % 256) as u8).collect();
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std = [0.26862954f32, 0.26130258, 0.27577711];
        let gh = h / p;
        let gw = w / p;
        let mut out = vec![0.0f32; gh * gw * 3 * tp * p * p];
        patchify_norm_qwen_vl_u8_f32(&rgb, w, h, p, m, tp, mean, std, &mut out);

        // Reference in exact (mr, mc, sr, sc, c, t, py, px) order.
        let mut idx = 0usize;
        for mr in 0..(gh / m) {
            for mc in 0..(gw / m) {
                for sr in 0..m {
                    for sc in 0..m {
                        let y0 = (mr * m + sr) * p;
                        let x0 = (mc * m + sc) * p;
                        for c in 0..3 {
                            for _t in 0..tp {
                                for py in 0..p {
                                    for px in 0..p {
                                        let v = rgb[((y0 + py) * w + (x0 + px)) * 3 + c] as f32;
                                        let want = (v / 255.0 - mean[c]) / std[c];
                                        assert!(
                                            (out[idx] - want).abs() < 1e-5,
                                            "qwen_vl patchify mismatch at {idx}: {} vs {want}",
                                            out[idx]
                                        );
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn patchify_qwen_vl_f16_bf16_match_f32() {
        let (w, h, p, m, tp) = (56usize, 28usize, 14usize, 2usize, 2usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 41 + 3) % 256) as u8).collect();
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std = [0.26862954f32, 0.26130258, 0.27577711];
        let len = (h / p) * (w / p) * 3 * tp * p * p;
        let mut f32_out = vec![0.0f32; len];
        let mut f16_out = vec![half::f16::ZERO; len];
        let mut bf16_out = vec![half::bf16::ZERO; len];
        patchify_norm_qwen_vl_u8_f32(&rgb, w, h, p, m, tp, mean, std, &mut f32_out);
        patchify_norm_qwen_vl_u8_f16(&rgb, w, h, p, m, tp, mean, std, &mut f16_out);
        patchify_norm_qwen_vl_u8_bf16(&rgb, w, h, p, m, tp, mean, std, &mut bf16_out);
        for i in 0..len {
            assert_eq!(f16_out[i].to_bits(), half::f16::from_f32(f32_out[i]).to_bits());
            assert_eq!(bf16_out[i].to_bits(), half::bf16::from_f32(f32_out[i]).to_bits());
        }
    }
}
