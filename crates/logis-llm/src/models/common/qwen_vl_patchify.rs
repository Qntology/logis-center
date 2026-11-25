//! Host CPU parallel patchifiers for Qwen-VL / Qwen3-VL (`f32`, `f16`, `bf16`).

use rayon::prelude::*;

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
