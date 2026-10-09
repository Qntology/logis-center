//! models / tensor 계층 CPU 검증 (`Device::Cpu` 전용, GPU 불필요)
//!
//! - position_embed      : RoPE 역주파수·cos/sin 표, candle_nn 기준 구현과의 동치, 회전 불변식
//!                         (노름 보존·상대 위치), M-RoPE 열 배치(Qwen2.5-VL 청크 / Qwen3-VL 인터리브),
//!                         회전 헬퍼, sinusoidal 인코더
//! - utils::tensor_utils : 마스크·인덱스·분할·정규화·풀링·패딩 헬퍼
//! - utils::resources    : KV 바이트 계산, 환경과 무관하게 결정되는 KV 배치 판정
//! - KVRegistry          : 레이어별 메타 저장/복원, 비전 토큰 태깅
//! - VisionEmbedCache    : 디스크 왕복, 재시작 후 인덱스 복원, LRU (전역 VISION_CACHE 는 건드리지 않음)
//! - siglip2             : NaFlex 전처리, bbox 헬퍼, 설정 파싱, 작은 가중치 비전 인코더 순전파
//! - 설정 serde          : Qwen3 / Qwen3.5 / Qwen3-VL / 임베딩, 작은 가중치 FloatModel 순전파
//!
//! Vulkan ↔ CPU 대조는 `vulkan.rs` 에 있으며 여기서는 반복하지 않습니다.
//! 가중치는 seed 고정 의사난수(`det`)라 실행마다 같은 값입니다.
//! `#[ignore = "BUG(n): …"]` 테스트는 '의도된 동작' 을 단언하며, 수정 전까지는 실패합니다.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use serde_json::json;

use crate::common::TempDir;
use tauri_app_lib::models::embedding::{Config as EmbeddingConfig, FloatModel};
use tauri_app_lib::models::qwen::quantized_model::{KVLocation, KVRegistry};
use tauri_app_lib::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use tauri_app_lib::models::qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig};
use tauri_app_lib::models::qwen3vl::config::{
    qwen3vl_text_config2qwen3_config, PreprocessorConfig, Qwen3VLTextConfig,
};
use tauri_app_lib::models::siglip2::preprocessor::{
    patch_index_to_bbox, patches_to_bounding_box, preprocess_image,
};
use tauri_app_lib::models::siglip2::vision::{Siglip2PatchEmbedding, Siglip2VisionModel};
use tauri_app_lib::models::siglip2::Siglip2Config;
use tauri_app_lib::models::vision_cache::{validate_embed_shape, VisionEmbedCache};
use tauri_app_lib::position_embed::rope::{
    apply_rotary_pos_emb, apply_rotary_pos_emb_roformer, apply_rotary_pos_emb_vision,
    compute_default_rope_parameters, roformer_rotate, rotate_half, Qwen2_5VLTextRotaryEmbedding,
    Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, RoPE,
};
use tauri_app_lib::position_embed::sinusoidal_pe::SinusoidalPositionEncoderCat;
use tauri_app_lib::utils::resources::{
    kv_bytes_per_token, plan_kv_residency, KvPlanInput, KvResidency, KV_RAM_SAFETY_MARGIN_BYTES,
    KV_VRAM_SAFETY_MARGIN_BYTES,
};
use tauri_app_lib::utils::tensor_utils as tu;

type R = anyhow::Result<()>;

// ─────────────────────────────────────────────────────────────────────────────
// 공용 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

fn f32s(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

/// 원소별 |got-want| <= tol * max(|want|, 1)
fn assert_close(got: &Tensor, want: &Tensor, tol: f32, what: &str) -> R {
    assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
    let (g, w) = (f32s(got)?, f32s(want)?);
    for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{what}: index {i}: got {a}, want {b} (tol {tol})"
        );
    }
    Ok(())
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let (x, y) = (f32s(a)?, f32s(b)?);
    assert_eq!(x.len(), y.len(), "max_abs_diff: length mismatch");
    Ok(x.iter().zip(&y).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max))
}

/// 결정적 의사난수 텐서 (xorshift64, 값 범위 [-scale, scale))
fn det(shape: &[usize], seed: u64, scale: f32) -> anyhow::Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x2545_F491_4F6C_DD1D) | 1;
    let v: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0) * scale
        })
        .collect();
    Ok(Tensor::from_vec(v, shape, &Device::Cpu)?)
}

fn dir_bytes(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum::<u64>()
        })
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. position_embed
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rope_inverse_frequencies_and_cos_sin_tables() -> R {
    let dev = Device::Cpu;
    // inv_freq[j] = base^(-2j/dim)
    let inv = compute_default_rope_parameters(8, 10_000.0);
    assert_eq!(inv.len(), 4);
    for (got, want) in inv.iter().zip([1.0f32, 0.1, 0.01, 0.001]) {
        assert!(((got - want) / want).abs() < 1e-5, "inv_freq {got} vs {want}");
    }
    // 홀수 차원은 ceil(dim/2) 개, 첫 값은 1, 단조 감소
    let odd = compute_default_rope_parameters(5, 10_000.0);
    assert_eq!(odd.len(), 3);
    assert_eq!(odd[0], 1.0);
    assert!(odd.windows(2).all(|w| w[1] < w[0]));

    // RoPE 표 (seq, dim): 위치 offset..offset+seq, 열 j 는 inv[j % (dim/2)] (앞/뒤 절반 동일)
    let (cos, sin) = RoPE::new(8, 10_000.0, &dev)?.forward(3, 4, &dev)?;
    assert_eq!(cos.dims(), &[4, 8]);
    assert_eq!(sin.dims(), &[4, 8]);
    let (c, s) = (cos.to_vec2::<f32>()?, sin.to_vec2::<f32>()?);
    for r in 0..4 {
        for j in 0..8 {
            let ang = (3 + r) as f32 * inv[j % 4];
            assert!((c[r][j] - ang.cos()).abs() < 1e-6, "cos[{r}][{j}]");
            assert!((s[r][j] - ang.sin()).abs() < 1e-6, "sin[{r}][{j}]");
        }
    }

    // 비전 RoPE 위상표 (seqlen, dim/2), theta 기본값 10000
    let vis = Qwen2_5VisionRotaryEmbedding::new(8, None).forward(5, &dev)?;
    assert_eq!(vis.dims(), &[5, 4]);
    let v = vis.to_vec2::<f32>()?;
    for (r, row) in v.iter().enumerate() {
        for j in 0..4 {
            assert!((row[j] - r as f32 * inv[j]).abs() < 1e-6, "vision freqs[{r}][{j}]");
        }
    }
    Ok(())
}

/// q·cos + rotate_half(q)·sin 은 candle_nn 기준 구현(rope, 반폭 cos/sin)과 같아야 하고,
/// 비전 레이아웃 / RoFormer(인접 쌍) 변형도 각자의 기준과 일치해야 한다.
#[test]
fn apply_rotary_matches_candle_reference_and_layout_variants() -> R {
    let dev = Device::Cpu;
    let q = det(&[1, 2, 5, 8], 1, 1.0)?;
    let k = det(&[1, 1, 5, 8], 2, 1.0)?;
    let (cos, sin) = RoPE::new(8, 10_000.0, &dev)?.forward(3, 5, &dev)?;
    let (qr, kr) = apply_rotary_pos_emb(&q, &k, &cos, &sin, false)?;
    let cos_h = cos.narrow(D::Minus1, 0, 4)?.contiguous()?;
    let sin_h = sin.narrow(D::Minus1, 0, 4)?.contiguous()?;
    assert_close(&qr, &candle_nn::rotary_emb::rope(&q, &cos_h, &sin_h)?, 1e-5, "q vs candle rope")?;
    assert_close(&kr, &candle_nn::rotary_emb::rope(&k, &cos_h, &sin_h)?, 1e-5, "k vs candle rope")?;

    // (bs, seq, dim) 형태의 cos/sin 도 같은 결과
    let (qb, _) = apply_rotary_pos_emb(&q, &k, &cos.unsqueeze(0)?, &sin.unsqueeze(0)?, false)?;
    assert_close(&qb, &qr, 1e-6, "rank-3 cos/sin")?;

    // 비전 레이아웃 (seq, heads, dim)
    let q_v = q.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    let k_v = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    let (qv, kv) = apply_rotary_pos_emb_vision(&q_v, &k_v, &cos, &sin)?;
    assert_close(&qv.transpose(0, 1)?.unsqueeze(0)?, &qr, 1e-6, "vision layout q")?;
    assert_close(&kv.transpose(0, 1)?.unsqueeze(0)?, &kr, 1e-6, "vision layout k")?;

    // RoFormer 는 인터리브한 cos/sin([c0,c0,c1,c1,..]) 으로 candle 의 rope_i 와 같다
    let cos_i = tu::repeat_interleave(&cos_h, 2, 1)?;
    let sin_i = tu::repeat_interleave(&sin_h, 2, 1)?;
    let (qi, _) = apply_rotary_pos_emb_roformer(&q, &k, &cos_i, &sin_i, false)?;
    assert_close(&qi, &candle_nn::rotary_emb::rope_i(&q, &cos_h, &sin_h)?, 1e-5, "roformer vs rope_i")?;

    // F16 입력: tof32 와 무관하게 결과 dtype 은 입력과 같고 값은 F32 결과에 근사
    let (q16, k16) = (q.to_dtype(DType::F16)?, k.to_dtype(DType::F16)?);
    for tof32 in [true, false] {
        let (a, b) = apply_rotary_pos_emb(&q16, &k16, &cos, &sin, tof32)?;
        assert_eq!((a.dtype(), b.dtype()), (DType::F16, DType::F16), "tof32={tof32}");
        assert_close(&a, &qr, 1e-2, "f16 q")?;
        assert_close(&b, &kr, 1e-2, "f16 k")?;
    }
    Ok(())
}

/// 회전은 노름을 보존하고, 회전된 q·k 는 상대 위치(m-n)에만 의존해야 한다.
#[test]
fn rope_rotation_preserves_norm_and_depends_on_relative_position() -> R {
    let dev = Device::Cpu;
    let rope = RoPE::new(16, 10_000.0, &dev)?;
    let q = det(&[1, 1, 1, 16], 11, 1.0)?;
    let k = det(&[1, 1, 1, 16], 12, 1.0)?;
    let rot = |x: &Tensor, pos: usize| -> anyhow::Result<Tensor> {
        let (c, s) = rope.forward(pos, 1, &dev)?;
        Ok(apply_rotary_pos_emb(x, x, &c, &s, false)?.0)
    };
    let dot = |a: &Tensor, b: &Tensor| -> anyhow::Result<f32> {
        let (x, y) = (f32s(a)?, f32s(b)?);
        Ok(x.iter().zip(&y).map(|(u, v)| u * v).sum())
    };
    let n0 = dot(&q, &q)?;
    for pos in [0usize, 1, 7, 100, 4000] {
        let r = rot(&q, pos)?;
        assert!((dot(&r, &r)? - n0).abs() < 1e-4 * n0.max(1.0), "norm changed at position {pos}");
    }
    let want = dot(&rot(&q, 3)?, &rot(&k, 0)?)?;
    for (m, n) in [(5usize, 2usize), (13, 10), (103, 100), (503, 500)] {
        let got = dot(&rot(&q, m)?, &rot(&k, n)?)?;
        assert!((got - want).abs() < 1e-3, "q·k at ({m},{n}) = {got}, at (3,0) = {want}");
    }
    Ok(())
}

/// 텍스트 토큰처럼 T/H/W 위치 행이 같으면 M-RoPE 는 1-D RoPE 와 같아야 한다.
/// (소형 / Qwen3.5 partial-rotary 64 차원 [11,11,10] / Qwen3-VL 128 차원 [24,20,20])
#[test]
fn mrope_with_identical_position_rows_equals_plain_rope() -> R {
    let dev = Device::Cpu;
    let seq = 6usize;
    let pos: Vec<u32> = (2..2 + seq as u32).collect();
    let row = Tensor::new(pos.as_slice(), &dev)?;
    let pos3 = row.reshape((1, 1, seq))?.broadcast_as((3, 1, seq))?.contiguous()?; // (3, bs, seq)
    let pos2 = row.unsqueeze(0)?; // (bs, seq)
    for (dim, theta, section) in [
        (8usize, 10_000f32, vec![2usize, 1, 1]),
        (64, 10_000_000.0, vec![11, 11, 10]),
        (128, 5_000_000.0, vec![24, 20, 20]),
    ] {
        let what = format!("dim {dim} section {section:?}");
        let (cos_ref, sin_ref) = RoPE::new(dim, theta, &dev)?.forward(2, seq, &dev)?;

        // Qwen3-VL / Qwen3.5: (bs, seq, dim)
        let q3 = Qwen3VLTextRotaryEmbedding::new(dim, theta);
        let (c, s) = q3.forward(&pos3, DType::F32, section.clone())?;
        assert_eq!(c.dims(), &[1, seq, dim], "{what}");
        assert_close(&c.squeeze(0)?, &cos_ref, 1e-6, &format!("qwen3-vl cos, {what}"))?;
        assert_close(&s.squeeze(0)?, &sin_ref, 1e-6, &format!("qwen3-vl sin, {what}"))?;
        // 2-D position_ids 는 세 축 공통으로 확장된다
        let (c2, s2) = q3.forward(&pos2, DType::F32, section.clone())?;
        assert_close(&c2, &c, 1e-6, &format!("qwen3-vl 2-D ids cos, {what}"))?;
        assert_close(&s2, &s, 1e-6, &format!("qwen3-vl 2-D ids sin, {what}"))?;
        assert_eq!(q3.forward(&pos2, DType::BF16, section.clone())?.0.dtype(), DType::BF16);

        // Qwen2.5-VL: (bs, 1, seq, dim)
        let (c25, s25) =
            Qwen2_5VLTextRotaryEmbedding::new(dim, theta).forward(&pos3, DType::F32, section.clone())?;
        assert_eq!(c25.dims(), &[1, 1, seq, dim], "{what}");
        assert_close(&c25.squeeze(0)?.squeeze(0)?, &cos_ref, 1e-6, &format!("qwen2.5-vl cos, {what}"))?;
        assert_close(&s25.squeeze(0)?.squeeze(0)?, &sin_ref, 1e-6, &format!("qwen2.5-vl sin, {what}"))?;
    }
    Ok(())
}

/// 서로 다른 T/H/W 위치(이미지 토큰)에서의 열 배치.
///   Qwen3-VL (HF apply_interleaved_mrope): H = slice(1, 3*s1, 3), W = slice(2, 3*s2, 3), 나머지 T
///   Qwen2.5-VL: [T s0 | H s1 | W s2] 구간을 앞/뒤 절반에 반복
/// 앞/뒤 절반이 같은 축을 써야 회전쌍이 유지되므로 노름도 보존되어야 한다.
#[test]
fn mrope_column_layout_for_distinct_thw_positions() -> R {
    let dev = Device::Cpu;
    let (pt, ph, pw) = (7u32, 3u32, 5u32);
    let pos3 = Tensor::new(&[[[pt]], [[ph]], [[pw]]], &dev)?; // (3, bs=1, seq=1)
    let p = [pt as f32, ph as f32, pw as f32];
    for (dim, section) in [(8usize, vec![2usize, 1, 1]), (64, vec![11, 11, 10]), (128, vec![24, 20, 20])] {
        let half = dim / 2;
        let inv = compute_default_rope_parameters(dim, 10_000.0);
        let axis_interleaved = |f: usize| {
            if f % 3 == 1 && f < 3 * section[1] {
                1
            } else if f % 3 == 2 && f < 3 * section[2] {
                2
            } else {
                0
            }
        };
        let axis_chunked = |f: usize| {
            if f < section[0] {
                0
            } else if f < section[0] + section[1] {
                1
            } else {
                2
            }
        };
        let (cos3, sin3) = Qwen3VLTextRotaryEmbedding::new(dim, 10_000.0).forward(&pos3, DType::F32, section.clone())?;
        let (cos25, sin25) =
            Qwen2_5VLTextRotaryEmbedding::new(dim, 10_000.0).forward(&pos3, DType::F32, section.clone())?;
        let (c3, s3, c25, s25) = (f32s(&cos3)?, f32s(&sin3)?, f32s(&cos25)?, f32s(&sin25)?);
        assert_eq!((c3.len(), c25.len()), (dim, dim));
        for j in 0..dim {
            let f = j % half;
            let a3 = p[axis_interleaved(f)] * inv[f];
            let a25 = p[axis_chunked(f)] * inv[f];
            assert!(
                (c3[j] - a3.cos()).abs() < 1e-6 && (s3[j] - a3.sin()).abs() < 1e-6,
                "qwen3-vl dim {dim} col {j}: axis {}",
                axis_interleaved(f)
            );
            assert!(
                (c25[j] - a25.cos()).abs() < 1e-6 && (s25[j] - a25.sin()).abs() < 1e-6,
                "qwen2.5-vl dim {dim} col {j}: axis {}",
                axis_chunked(f)
            );
        }
        // 이미지 토큰 위치에서도 회전은 노름을 보존
        let q = det(&[1, 2, 1, dim], 21, 1.0)?;
        let n0: f32 = f32s(&q)?.iter().map(|x| x * x).sum();
        for (cos, sin, name) in [(&cos3, &sin3, "qwen3-vl"), (&cos25, &sin25, "qwen2.5-vl")] {
            let (r, _) = apply_rotary_pos_emb(&q, &q, cos, sin, false)?;
            let n1: f32 = f32s(&r)?.iter().map(|x| x * x).sum();
            assert!((n1 - n0).abs() < 1e-4 * n0, "{name} dim {dim}: norm {n0} → {n1}");
        }
    }
    Ok(())
}

#[test]
fn rotate_helpers_and_odd_dim_rejection() -> R {
    let dev = Device::Cpu;
    let x = Tensor::new(&[1f32, 2., 3., 4., 5., 6.], &dev)?;
    assert_eq!(rotate_half(&x)?.to_vec1::<f32>()?, vec![-4., -5., -6., 1., 2., 3.]);
    // 90° 를 두 번 돌리면 부호 반전
    assert_eq!(rotate_half(&rotate_half(&x)?)?.to_vec1::<f32>()?, vec![-1., -2., -3., -4., -5., -6.]);
    // 인접 쌍 회전: [x0,x1,x2,x3] → [-x1,x0,-x3,x2]
    assert_eq!(
        roformer_rotate(&Tensor::new(&[1f32, 2., 3., 4.], &dev)?)?.to_vec1::<f32>()?,
        vec![-2., 1., -4., 3.]
    );
    let m = Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &dev)?;
    assert_eq!(
        roformer_rotate(&m)?.to_vec2::<f32>()?,
        vec![vec![-2., 1., -4., 3.], vec![-6., 5., -8., 7.]]
    );
    // 마지막 차원이 홀수면 조용히 잘라내지 않고 에러
    assert!(roformer_rotate(&Tensor::new(&[1f32, 2., 3.], &dev)?).is_err());
    assert!(roformer_rotate(&Tensor::zeros((2, 5), DType::F32, &dev)?).is_err());
    Ok(())
}

#[test]
fn sinusoidal_encoder_tables_and_forward() -> R {
    let dev = Device::Cpu;
    let enc = SinusoidalPositionEncoderCat::new(None, false, &dev)?;
    let pe = enc.encode(0, 4, 8, &dev, DType::F32)?;
    assert_eq!(pe.dims(), &[4, 8]);
    let rows = pe.to_vec2::<f32>()?;
    // [sin | cos] 연결: 위치 0 은 [0,0,0,0,1,1,1,1]
    assert_eq!(rows[0], vec![0., 0., 0., 0., 1., 1., 1., 1.]);
    let inv = compute_default_rope_parameters(8, 10_000.0);
    for (r, row) in rows.iter().enumerate() {
        for j in 0..4 {
            let a = r as f32 * inv[j];
            assert!(
                (row[j] - a.sin()).abs() < 1e-6 && (row[4 + j] - a.cos()).abs() < 1e-6,
                "row {r} col {j}"
            );
        }
    }
    // 오프셋은 같은 표의 뒷부분
    assert_close(&enc.encode(2, 2, 8, &dev, DType::F32)?, &pe.narrow(0, 2, 2)?, 1e-6, "offset")?;
    // 역주파수를 미리 저장해도 같은 표
    let saved = SinusoidalPositionEncoderCat::new(Some(8), true, &dev)?;
    assert_close(&saved.encode(0, 4, 8, &dev, DType::F32)?, &pe, 0.0, "save_freq")?;
    // forward: (b, seq, dim) 에 표를 더하고 입력 dtype 유지
    let xs = Tensor::ones((2, 4, 8), DType::F32, &dev)?;
    let want = (pe.unsqueeze(0)?.broadcast_as((2, 4, 8))? + 1.0)?;
    assert_close(&enc.forward(&xs, 0)?, &want, 1e-6, "forward")?;
    let out16 = enc.forward(&xs.to_dtype(DType::F16)?, 0)?;
    assert_eq!(out16.dtype(), DType::F16);
    assert_eq!(out16.dims(), &[2, 4, 8]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. utils::tensor_utils
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fill_helpers_zero_or_replace_masked_positions() -> R {
    let dev = Device::Cpu;
    let hs = Tensor::new(&[[[1f32, 2.], [3., 4.], [5., 6.]]], &dev)?; // (1, 3, 2)
    let want = vec![vec![1f32, 2.], vec![0., 0.], vec![5., 6.]];
    let keep = Tensor::new(&[[1u8, 0, 1]], &dev)?;
    assert_eq!(tu::masked_fill_zeros(&hs, &keep)?.squeeze(0)?.to_vec2::<f32>()?, want);
    // get_not_equal_mask 가 만드는 U32 마스크도 같은 의미
    let keep32 = tu::get_not_equal_mask(&Tensor::new(&[[7u32, 9, 7]], &dev)?, 9)?;
    assert_eq!(tu::masked_fill_zeros(&hs, &keep32)?.squeeze(0)?.to_vec2::<f32>()?, want);

    // attn_masked_fill: 큰 마스크를 (seq, seq) 로 잘라 0 인 곳을 on_false 로 채움
    let scores = Tensor::arange(0f32, 4., &dev)?.reshape((1, 1, 2, 2))?;
    let tril = Tensor::tril2(3, DType::U8, &dev)?;
    let filled = tu::attn_masked_fill(&scores, &tril, -100.0)?;
    assert_eq!(filled.dims(), &[1, 1, 2, 2]);
    assert_eq!(filled.flatten_all()?.to_vec1::<f32>()?, vec![0., -100., 2., 3.]);
    Ok(())
}

#[test]
fn token_masks_indices_and_onehot() -> R {
    let dev = Device::Cpu;
    let ids = Tensor::new(&[5u32, 9, 9, 3, 9], &dev)?;
    assert_eq!(tu::get_equal_mask(&ids, 9)?.to_vec1::<u32>()?, vec![0, 1, 1, 0, 1]);
    assert_eq!(tu::get_not_equal_mask(&ids, 9)?.to_vec1::<u32>()?, vec![1, 0, 0, 1, 0]);
    assert_eq!(tu::get_eq_indices(&ids, 9)?.to_vec1::<u32>()?, vec![1, 2, 4]);
    assert_eq!(tu::get_vision_next_indices(&ids, 9)?.to_vec1::<u32>()?, vec![2, 3, 5]);
    assert_eq!(tu::get_eq_indices(&ids, 42)?.dims(), &[0]);
    // 인덱스 추출은 1-D 전용
    assert!(tu::get_eq_indices(&ids.unsqueeze(0)?, 9).is_err());

    let a = Tensor::new(&[1u32, 0, 0, 1], &dev)?;
    let b = Tensor::new(&[0u32, 0, 1, 1], &dev)?;
    assert_eq!(tu::bitor_tensor(&a, &b)?.to_vec1::<u8>()?, vec![1, 0, 1, 1]);

    // 범위 밖 클래스(7)는 전부 0 인 행
    let oh = tu::onehot(&Tensor::new(&[2u32, 0, 1, 7], &dev)?, 3)?;
    assert_eq!(oh.dtype(), DType::U8);
    assert_eq!(
        oh.to_vec2::<u8>()?,
        vec![vec![0, 0, 1], vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 0]]
    );

    let (rows, cols) = tu::nonzero(&Tensor::new(&[[0u32, 1, 1], [1, 0, 0], [0, 0, 0]], &dev)?)?;
    assert_eq!(rows, vec![0, 0, 1]);
    assert_eq!(cols, vec![1, 2, 0]);
    Ok(())
}

#[test]
fn sequence_mask_marks_valid_prefix() -> R {
    let dev = Device::Cpu;
    let len = Tensor::new(&[1u32, 3, 2], &dev)?;
    assert_eq!(
        tu::sequence_mask(&len, None)?.to_vec2::<u8>()?,
        vec![vec![1, 0, 0], vec![1, 1, 1], vec![1, 1, 0]]
    );
    assert_eq!(
        tu::sequence_mask(&len, Some(5))?.to_vec2::<u8>()?,
        vec![vec![1, 0, 0, 0, 0], vec![1, 1, 1, 0, 0], vec![1, 1, 0, 0, 0]]
    );
    Ok(())
}

#[test]
fn repeat_kv_equals_repeat_interleave_over_heads() -> R {
    let dev = Device::Cpu;
    let xs = Tensor::arange(0f32, 24., &dev)?.reshape((1, 2, 3, 4))?;
    let rep = tu::repeat_kv(xs.clone(), 3)?;
    assert_eq!(rep.dims(), &[1, 6, 3, 4]);
    assert_close(&rep, &tu::repeat_interleave(&xs, 3, 1)?, 0.0, "repeat_kv")?;
    // GQA: 쿼리 헤드 h 는 KV 헤드 h / n_rep 를 본다
    for h in 0..6 {
        assert_close(&rep.i((0, h))?, &xs.i((0, h / 3))?, 0.0, &format!("query head {h}"))?;
    }
    assert_close(&tu::repeat_kv(xs.clone(), 1)?, &xs, 0.0, "n_rep 1")?;

    let m = Tensor::new(&[[1f32, 2.], [3., 4.]], &dev)?;
    assert_eq!(
        tu::repeat_interleave(&m, 2, 0)?.to_vec2::<f32>()?,
        vec![vec![1., 2.], vec![1., 2.], vec![3., 4.], vec![3., 4.]]
    );
    assert_eq!(
        tu::repeat_interleave(&m, 2, 1)?.to_vec2::<f32>()?,
        vec![vec![1., 1., 2., 2.], vec![3., 3., 4., 4.]]
    );
    assert!(tu::repeat_interleave(&m, 2, 2).is_err());
    Ok(())
}

#[test]
fn split_helpers_cover_the_axis_in_order() -> R {
    let dev = Device::Cpu;
    let t = Tensor::arange(0u32, 10, &dev)?;
    let parts = tu::split_tensor(&t, &[3, 3, 4], 0)?;
    let got: Vec<Vec<u32>> = parts.iter().map(|p| p.to_vec1::<u32>()).collect::<candle_core::Result<_>>()?;
    assert_eq!(got, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]]);
    // 분할 합이 축 길이를 넘으면 에러
    assert!(tu::split_tensor(&t, &[6, 6], 0).is_err());

    // 고정 크기 분할: 마지막 조각만 짧다
    let chunks = tu::split_tensor_with_size(&t, 4, D::Minus1)?;
    let sizes = chunks.iter().map(|c| c.dim(0)).collect::<candle_core::Result<Vec<_>>>()?;
    assert_eq!(sizes, vec![4, 4, 2]);
    assert_close(&Tensor::cat(&chunks, 0)?, &t, 0.0, "concat of chunks")?;

    let m = Tensor::arange(0f32, 12., &dev)?.reshape((2, 6))?;
    let cols = tu::split_tensor(&m, &[2, 4], D::Minus1)?;
    assert_eq!(cols[0].to_vec2::<f32>()?, vec![vec![0., 1.], vec![6., 7.]]);
    assert_eq!(cols[1].dims(), &[2, 4]);
    Ok(())
}

#[test]
fn nonzero_zero_and_run_helpers() -> R {
    let dev = Device::Cpu;
    let m = Tensor::new(&[0u8, 1, 1, 0, 1, 0, 0, 1, 1, 1], &dev)?;
    assert_eq!(tu::nonzero_index_vec(&m)?, vec![1, 2, 4, 7, 8, 9]);
    assert_eq!(tu::nonzero_index(&m)?.to_vec1::<u32>()?, vec![1, 2, 4, 7, 8, 9]);
    assert_eq!(tu::zero_index_vec(&m)?, vec![0, 3, 5, 6]);
    assert_eq!(tu::zero_index(&m)?.to_vec1::<u32>()?, vec![0, 3, 5, 6]);
    // 연속 구간 [start, end)
    assert_eq!(tu::nonzero_slice(&m)?, vec![(1, 3), (4, 5), (7, 10)]);
    assert_eq!(tu::nonzero_slice(&Tensor::new(&[0u32, 0, 1], &dev)?)?, vec![(2, 3)]);
    assert!(tu::nonzero_slice(&Tensor::zeros(4, DType::U32, &dev)?)?.is_empty());
    // F32 마스크도 정수로 바꿔 판정
    assert_eq!(tu::nonzero_index_vec(&Tensor::new(&[0f32, 2., 0.], &dev)?)?, vec![1]);
    // 랭크 0 / 2 는 거부
    let scalar = Tensor::new(1u32, &dev)?;
    let mat = Tensor::ones((2, 2), DType::U32, &dev)?;
    assert!(tu::nonzero_index_vec(&scalar).is_err());
    assert!(tu::nonzero_index(&scalar).is_err());
    assert!(tu::nonzero_index(&mat).is_err());
    assert!(tu::zero_index_vec(&mat).is_err());
    assert!(tu::zero_index(&mat).is_err());
    Ok(())
}

#[test]
fn masked_scatter_dim0_fills_masked_rows_in_order() -> R {
    let dev = Device::Cpu;
    let original = Tensor::zeros((1, 4, 2), DType::F32, &dev)?;
    let replace = Tensor::new(&[[1f32, 2.], [3., 4.]], &dev)?;
    let mask = Tensor::new(&[[0u32, 1, 1, 0]], &dev)?;
    let out = tu::masked_scatter_dim0(&original, &replace, &mask)?;
    assert_eq!(out.dims(), &[1, 4, 2]);
    assert_eq!(
        out.squeeze(0)?.to_vec2::<f32>()?,
        vec![vec![0., 0.], vec![1., 2.], vec![3., 4.], vec![0., 0.]]
    );
    // 떨어진 구간도 replace 행을 순서대로 소비
    let split = Tensor::new(&[[1u32, 0, 0, 1]], &dev)?;
    assert_eq!(
        tu::masked_scatter_dim0(&original, &replace, &split)?.squeeze(0)?.to_vec2::<f32>()?,
        vec![vec![1., 2.], vec![0., 0.], vec![0., 0.], vec![3., 4.]]
    );
    // 배치 2 이상은 거부
    let batch2 = Tensor::zeros((2, 4, 2), DType::F32, &dev)?;
    assert!(tu::masked_scatter_dim0(&batch2, &replace, &mask).is_err());
    Ok(())
}

#[test]
fn mask_index_add_and_index_select_2d() -> R {
    let dev = Device::Cpu;
    let base = Tensor::ones((4, 2), DType::F32, &dev)?;
    let mask = Tensor::new(&[0u32, 1, 0, 1], &dev)?;
    let add = Tensor::new(&[[1f32, 1.], [2., 2.]], &dev)?;
    assert_eq!(
        tu::mask_index_add(&base, &mask, &add)?.to_vec2::<f32>()?,
        vec![vec![1., 1.], vec![2., 2.], vec![1., 1.], vec![3., 3.]]
    );

    let table = Tensor::arange(0f32, 15., &dev)?.reshape((5, 3))?;
    let idx = Tensor::new(&[[0u32, 4], [2, 2]], &dev)?;
    let out = tu::index_select_2d(&table, &idx)?;
    assert_eq!(out.dims(), &[2, 2, 3]);
    assert_eq!(out.i(0)?.to_vec2::<f32>()?, vec![vec![0., 1., 2.], vec![12., 13., 14.]]);
    assert_eq!(out.i(1)?.to_vec2::<f32>()?, vec![vec![6., 7., 8.], vec![6., 7., 8.]]);
    Ok(())
}

#[test]
fn topk_and_safe_arg_sort() -> R {
    let dev = Device::Cpu;
    let w = Tensor::new(&[[0.1f32, 0.9, 0.5, 0.7], [4., 3., 2., 1.]], &dev)?;
    let (vals, idx) = tu::topk(&w, 2)?;
    assert_eq!(idx.to_vec2::<u32>()?, vec![vec![1, 3], vec![0, 1]]);
    assert_eq!(vals.to_vec2::<f32>()?, vec![vec![0.9, 0.7], vec![4., 3.]]);

    // 1024 초과 축은 CPU 정렬 경로
    let n = 1500usize;
    let v: Vec<f32> = (0..n).map(|i| ((i * 7919) % n) as f32).collect(); // 0..n 의 순열
    let t = Tensor::from_vec(v.clone(), n, &dev)?;
    let asc = tu::safe_arg_sort_last_dim(&t, true)?.to_vec1::<u32>()?;
    let sorted: Vec<f32> = asc.iter().map(|&i| v[i as usize]).collect();
    assert_eq!(sorted, (0..n).map(|i| i as f32).collect::<Vec<_>>());
    let desc = tu::safe_arg_sort_last_dim(&t, false)?.to_vec1::<u32>()?;
    assert_eq!(v[desc[0] as usize], (n - 1) as f32);
    assert_eq!(v[desc[n - 1] as usize], 0.0);
    // 짧은 축은 그대로 arg_sort
    let small = t.narrow(0, 0, 16)?.contiguous()?;
    assert_eq!(
        tu::safe_arg_sort_last_dim(&small, true)?.to_vec1::<u32>()?,
        small.arg_sort_last_dim(true)?.to_vec1::<u32>()?
    );
    Ok(())
}

#[test]
fn l1_l2_and_z_score_normalization() -> R {
    let dev = Device::Cpu;
    let t = Tensor::new(&[[3f32, 4.], [0., 0.]], &dev)?;
    let l2 = tu::l2_normalize(&t, 1)?.to_vec2::<f32>()?;
    assert!((l2[0][0] - 0.6).abs() < 1e-5 && (l2[0][1] - 0.8).abs() < 1e-5, "{l2:?}");
    assert_eq!(l2[1], vec![0.0, 0.0], "eps keeps the all-zero row finite");
    assert_eq!(
        tu::l1_normalize(&Tensor::new(&[[1f32, -3.]], &dev)?, 1)?.to_vec2::<f32>()?,
        vec![vec![0.25, -0.75]]
    );
    // z-score 는 표본(불편) 분산 기준: 평균 0, Σz²/(n-1) = 1
    let z = tu::z_score_normalize(&Tensor::new(&[1f32, 2., 3., 4., 5.], &dev)?, 0)?.to_vec1::<f32>()?;
    assert!(z.iter().sum::<f32>().abs() < 1e-5, "{z:?}");
    assert!((z.iter().map(|x| x * x).sum::<f32>() / 4.0 - 1.0).abs() < 1e-5, "{z:?}");
    assert!((z[4] - 2.0 / 2.5f32.sqrt()).abs() < 1e-5, "{z:?}");
    // 축이 랭크를 넘으면 에러
    let fns: [fn(&Tensor, usize) -> anyhow::Result<Tensor>; 3] =
        [tu::l2_normalize, tu::l1_normalize, tu::z_score_normalize];
    for f in fns {
        assert!(f(&t, 2).is_err(), "dim >= rank must be rejected");
    }
    Ok(())
}

#[test]
fn cosine_similarity_range_normalize_log10_and_linspace() -> R {
    let dev = Device::Cpu;
    let q = Tensor::new(&[[1f32, 0.]], &dev)?;
    let m = Tensor::new(&[[2f32, 0.], [0., 3.], [-1., 0.], [1., 1.]], &dev)?;
    let s = tu::cosine_similarity(&q, &m)?;
    assert_eq!(s.dims(), &[1, 4]);
    for (g, w) in f32s(&s)?.iter().zip([1.0f32, 0.0, -1.0, std::f32::consts::FRAC_1_SQRT_2]) {
        assert!((g - w).abs() < 1e-4, "cosine {g} vs {w}");
    }

    // 최대 절댓값이 1 을 넘을 때만 축소, 이후 [-1, 1] 로 자름
    assert_eq!(
        tu::float_range_normalize(&Tensor::new(&[-4f32, 2., 1.], &dev)?)?.to_vec1::<f32>()?,
        vec![-1.0, 0.5, 0.25]
    );
    assert_eq!(
        tu::float_range_normalize(&Tensor::new(&[0.5f32, -0.25], &dev)?)?.to_vec1::<f32>()?,
        vec![0.5, -0.25]
    );
    assert_eq!(
        tu::float_range_normalize(&Tensor::zeros(3, DType::F32, &dev)?)?.to_vec1::<f32>()?,
        vec![0.0; 3]
    );

    let l = tu::log10(&Tensor::new(&[1f32, 10., 100., 1000.], &dev)?)?;
    for (g, w) in f32s(&l)?.iter().zip([0f32, 1., 2., 3.]) {
        assert!((g - w).abs() < 1e-5, "log10 {g} vs {w}");
    }

    // 양 끝점 포함
    assert_eq!(tu::linspace(0.0, 1.0, 5, &dev)?.to_vec1::<f32>()?, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(tu::linspace(1.0, -1.0, 3, &dev)?.to_vec1::<f32>()?, vec![1.0, 0.0, -1.0]);
    assert_eq!(tu::linspace(3.0, 9.0, 1, &dev)?.to_vec1::<f32>()?, vec![3.0]);
    Ok(())
}

#[test]
fn pool1d_floor_ceil_modes_and_statistics_pooling() -> R {
    let dev = Device::Cpu;
    let xs = Tensor::new(&[[[1f32, 2., 3., 4., 5.]]], &dev)?; // (1, 1, 5)
    let pool = |size: usize, ceil: bool, kind: &str| -> anyhow::Result<Vec<f32>> {
        f32s(&tu::pool1d(&xs, size, ceil, kind)?)
    };
    // floor: 나머지 버림
    assert_eq!(pool(2, false, "avg")?, vec![1.5, 3.5]);
    assert_eq!(pool(2, false, "max")?, vec![2., 4.]);
    assert_eq!(pool(2, false, "min")?, vec![1., 3.]);
    // ceil: 마지막 값을 복제해 채움
    assert_eq!(pool(2, true, "avg")?, vec![1.5, 3.5, 5.]);
    assert_eq!(pool(2, true, "max")?, vec![2., 4., 5.]);
    assert_eq!(pool(5, true, "avg")?, vec![3.]);
    assert_eq!(tu::pool1d(&xs, 2, true, "avg")?.dims(), &[1, 1, 3]);
    assert!(tu::pool1d(&xs, 0, true, "avg").is_err());
    assert!(tu::pool1d(&xs, 2, true, "sum").is_err());

    // (b, t, c) 의 시간축 통계: [mean_c.., std_c..] (std 는 표본 표준편차)
    let seq = Tensor::new(&[[[1f32, 10.], [2., 20.], [3., 30.], [4., 40.]]], &dev)?;
    let st = tu::statistics_pooling(&seq, D::Minus2, false)?;
    assert_eq!(st.dims(), &[1, 4]);
    let sd = (5f32 / 3.0).sqrt();
    for (g, w) in f32s(&st)?.iter().zip([2.5f32, 25.0, sd, 10.0 * sd]) {
        assert!((g - w).abs() < 1e-4, "stat {g} vs {w}");
    }
    assert_eq!(tu::statistics_pooling(&seq, D::Minus2, true)?.dims(), &[1, 1, 4]);
    Ok(())
}

#[test]
fn quick_gelu_and_prod_over_last_dim() -> R {
    let dev = Device::Cpu;
    let xs = [-2f32, -0.5, 0., 0.5, 2.];
    let y = tu::quick_gelu(&Tensor::new(&xs, &dev)?)?.to_vec1::<f32>()?;
    for (x, got) in xs.iter().zip(&y) {
        let want = x / (1.0 + (-1.702 * x).exp());
        assert!((got - want).abs() < 1e-5, "quick_gelu({x}) = {got}, want {want}");
    }
    // grid_thw (t,h,w) → 행마다 곱
    assert_eq!(
        tu::prod_tensor_last_dim(&Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?)?.to_vec1::<u32>()?,
        vec![6, 120]
    );
    assert_eq!(
        tu::prod_tensor_last_dim(&Tensor::new(&[[2i64, -3]], &dev)?)?.to_vec1::<i64>()?,
        vec![-6]
    );
    assert_eq!(
        tu::prod_tensor_last_dim(&Tensor::new(&[2f32, 3., 0.5], &dev)?)?.to_vec1::<f32>()?,
        vec![3.0]
    );
    assert!(tu::prod_tensor_last_dim(&Tensor::zeros((1, 1, 1), DType::U32, &dev)?).is_err());
    Ok(())
}

#[test]
fn pad_helpers_single_sided() -> R {
    let dev = Device::Cpu;
    let t = Tensor::new(&[1f32, 2., 3., 4., 5.], &dev)?;
    // reflect(가장자리 제외 거울상) – 왼쪽
    assert_eq!(tu::pad_reflect_last_dim(&t, (2, 0))?.to_vec1::<f32>()?, vec![3., 2., 1., 2., 3., 4., 5.]);
    assert_eq!(tu::pad_reflect_last_dim(&t, (0, 0))?.to_vec1::<f32>()?, vec![1., 2., 3., 4., 5.]);
    assert!(tu::pad_reflect_last_dim(&t, (5, 0)).is_err());
    assert!(tu::pad_reflect_last_dim(&t, (0, 5)).is_err());
    // replicate – 한쪽씩
    let r = Tensor::new(&[1f32, 2., 3.], &dev)?;
    assert_eq!(tu::pad_replicate_last_dim(&r, (2, 0))?.to_vec1::<f32>()?, vec![1., 1., 1., 2., 3.]);
    assert_eq!(tu::pad_replicate_last_dim(&r, (0, 2))?.to_vec1::<f32>()?, vec![1., 2., 3., 3., 3.]);
    let r2 = Tensor::new(&[[1f32, 2.], [3., 4.]], &dev)?;
    assert_eq!(
        tu::pad_replicate_last_dim(&r2, (0, 1))?.to_vec2::<f32>()?,
        vec![vec![1., 2., 2.], vec![3., 4., 4.]]
    );
    Ok(())
}

#[test]
#[ignore = "BUG(12): pad_reflect_last_dim mirrors the right edge including the edge sample, and from a shifted offset after a left pad"]
fn pad_reflect_right_side_matches_torch_reflect() -> R {
    let dev = Device::Cpu;
    let t = Tensor::new(&[1f32, 2., 3., 4., 5.], &dev)?;
    // torch.nn.functional.pad(t, (l, r), mode="reflect")
    assert_eq!(tu::pad_reflect_last_dim(&t, (0, 2))?.to_vec1::<f32>()?, vec![1., 2., 3., 4., 5., 4., 3.]);
    assert_eq!(
        tu::pad_reflect_last_dim(&t, (2, 2))?.to_vec1::<f32>()?,
        vec![3., 2., 1., 2., 3., 4., 5., 4., 3.]
    );
    Ok(())
}

#[test]
#[ignore = "BUG(12): pad_replicate_last_dim reads the right edge at the pre-padding length, so a left pad shifts it"]
fn pad_replicate_both_sides_uses_each_edge() -> R {
    let dev = Device::Cpu;
    let t = Tensor::new(&[1f32, 2., 3.], &dev)?;
    assert_eq!(tu::pad_replicate_last_dim(&t, (2, 1))?.to_vec1::<f32>()?, vec![1., 1., 1., 2., 3., 3.]);
    Ok(())
}

#[test]
#[ignore = "BUG(12): index_select_2d guard uses && so a rank-3 table with a rank-2 index is accepted"]
fn index_select_2d_rejects_non_2d_inputs() -> R {
    let dev = Device::Cpu;
    let table3 = Tensor::zeros((2, 3, 4), DType::F32, &dev)?;
    let idx2 = Tensor::new(&[[0u32, 1]], &dev)?;
    assert!(tu::index_select_2d(&table3, &idx2).is_err(), "rank-3 table must be rejected");
    let table2 = Tensor::zeros((3, 4), DType::F32, &dev)?;
    let idx3 = Tensor::zeros((1, 1, 2), DType::U32, &dev)?;
    assert!(tu::index_select_2d(&table2, &idx3).is_err(), "rank-3 index must be rejected");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. utils::resources — KV 배치 판정 (환경과 무관하게 결정되는 경로만)
// ─────────────────────────────────────────────────────────────────────────────

fn kv_input(gpu_id: u32, is_cpu_mode: bool, planned_tokens: usize) -> KvPlanInput<'static> {
    KvPlanInput {
        gpu_id,
        is_cpu_mode,
        num_kv_layers: 28,
        num_kv_heads: 8,
        head_dim: 128,
        bytes_per_elem: 2,
        planned_tokens,
        label: "models_cpu",
    }
}

#[test]
fn kv_residency_bytes_and_deterministic_decisions() -> R {
    // K+V × heads × head_dim × bytes × layers
    assert_eq!(kv_bytes_per_token(&kv_input(0, true, 1)), 2 * 8 * 128 * 2 * 28);
    let fp8 = KvPlanInput { bytes_per_elem: 1, num_kv_layers: 7, ..kv_input(0, true, 1) };
    assert_eq!(kv_bytes_per_token(&fp8), 2 * 8 * 128 * 7);
    assert_eq!(KV_VRAM_SAFETY_MARGIN_BYTES, 640u64 * 1024 * 1024);
    assert_eq!(KV_RAM_SAFETY_MARGIN_BYTES, 2u64 * 1024 * 1024 * 1024);
    // CPU 모드는 필요량과 무관하게 RAM
    for tokens in [0usize, 4096, usize::MAX] {
        assert_eq!(plan_kv_residency(&kv_input(0, true, tokens)), KvResidency::Ram, "tokens {tokens}");
    }
    // 없는 GPU 번호 + 포화된 필요량 → VRAM 도 RAM 도 안 되므로 SSD
    assert_eq!(plan_kv_residency(&kv_input(7, false, usize::MAX)), KvResidency::Ssd);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. KVRegistry — 레이어별 메타 저장/복원, 비전 태깅
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kv_registry_defaults_and_vision_tagging() -> R {
    let reg = KVRegistry::new();
    {
        let e = reg.entries.read().unwrap();
        assert_eq!(e.len(), 128);
        for (i, en) in e.iter().enumerate() {
            assert_eq!((en.token_start, en.token_len), (i * 1024, 0), "entry {i}");
            assert_eq!(en.location, vec![KVLocation::SSD; 28], "entry {i}");
            assert!(en.ssd_path.is_none());
            assert_eq!(en.vision_token_ratio, 0.0);
        }
    }
    // 블록0: 1/4 이미지, 블록1: 텍스트, 블록2(프롬프트 끝, 512 토큰): 3/4 비디오
    const IMG: u32 = 151_655;
    const VID: u32 = 151_656;
    let mut ids = vec![1u32; 2560];
    ids[..256].iter_mut().for_each(|t| *t = IMG);
    ids[2048..2048 + 384].iter_mut().for_each(|t| *t = VID);
    reg.tag_vision_blocks(&ids, &[IMG, VID]);
    {
        let e = reg.entries.read().unwrap();
        assert_eq!(e[0].vision_token_ratio, 0.25);
        assert_eq!(e[1].vision_token_ratio, 0.0);
        assert_eq!(e[2].vision_token_ratio, 0.75);
        assert_eq!(e[3].vision_token_ratio, 0.0, "blocks past the prompt are untouched");
    }
    assert!(reg.is_vision_dominant(0, 0.2));
    assert!(!reg.is_vision_dominant(0, 0.25), "threshold is strict");
    assert!(!reg.is_vision_dominant(1, 0.0));
    assert!(reg.is_vision_dominant(2, 0.5));
    assert!(!reg.is_vision_dominant(500, 0.0), "out-of-range block is not dominant");
    // 비전 토큰 목록이 비었거나 입력이 비면 아무것도 바꾸지 않는다
    reg.tag_vision_blocks(&ids, &[]);
    reg.tag_vision_blocks(&[], &[IMG]);
    assert_eq!(reg.entries.read().unwrap()[0].vision_token_ratio, 0.25);
    Ok(())
}

#[test]
fn kv_registry_layer_meta_roundtrip_restores_ssd_locations_and_paths() -> R {
    let tmp = TempDir::new("kvreg_roundtrip");
    let dir = tmp.path();
    let (p0, p1) = (dir.join("block0.kv"), dir.join("block1.kv"));
    let src = KVRegistry::new();
    {
        let mut e = src.entries.write().unwrap();
        e[0].token_len = 1024;
        e[0].ssd_path = Some(p0.clone());
        e[1].token_len = 300;
        e[1].ssd_path = Some(p1.clone());
        e[1].location[3] = KVLocation::VRAM; // 레이어 3 에서는 블록1 이 VRAM 상주
        for en in e.iter_mut().skip(2) {
            en.location.iter_mut().for_each(|l| *l = KVLocation::RAM);
        }
    }
    src.save_to_file(dir)?;
    for l in 0..28 {
        assert!(dir.join(format!("layer{l}_meta.json")).is_file(), "layer {l} meta missing");
    }
    // 레이어 장부에는 그 레이어에서 SSD 에 있는 블록만 기록된다
    let read_json = |name: &str| -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::from_str(&std::fs::read_to_string(dir.join(name))?)?)
    };
    let layer0 = read_json("layer0_meta.json")?;
    let layer3 = read_json("layer3_meta.json")?;
    assert_eq!(layer0.as_array().map(|a| a.len()), Some(2));
    assert_eq!(layer3.as_array().map(|a| a.len()), Some(1));
    assert_eq!(layer3[0]["token_start"], 0);
    assert_eq!(layer3[0]["token_len"], 1024);
    assert_eq!(layer3[0]["ssd_path"], json!(p0.to_str().unwrap()));
    assert_eq!(layer0[1]["token_start"], 1024);
    assert_eq!(layer0[1]["token_len"], 300);
    assert_eq!(layer0[1]["ssd_path"], json!(p1.to_str().unwrap()));

    // 장부가 없는 폴더에서 불러오면 아무것도 바뀌지 않는다
    let fresh = KVRegistry::new();
    fresh.load_from_file(&dir.join("no-such-dir"))?;
    assert!(fresh.entries.read().unwrap().iter().all(|en| en.ssd_path.is_none()));

    // 전부 VRAM 으로 채운 장부에 복원 → 기록된 (블록, 레이어) 만 SSD, 경로 복원
    let dst = KVRegistry::new();
    {
        let mut e = dst.entries.write().unwrap();
        for en in e.iter_mut() {
            en.location.iter_mut().for_each(|l| *l = KVLocation::VRAM);
        }
    }
    dst.load_from_file(dir)?;
    let e = dst.entries.read().unwrap();
    assert_eq!(e[0].location, vec![KVLocation::SSD; 28]);
    assert_eq!(e[0].ssd_path.as_deref(), Some(p0.as_path()));
    for l in 0..28 {
        let want = if l == 3 { KVLocation::VRAM } else { KVLocation::SSD };
        assert_eq!(e[1].location[l], want, "block 1 layer {l}");
    }
    assert_eq!(e[1].ssd_path.as_deref(), Some(p1.as_path()));
    assert!(e[2..]
        .iter()
        .all(|en| en.location.iter().all(|l| *l == KVLocation::VRAM) && en.ssd_path.is_none()));
    Ok(())
}

#[test]
#[ignore = "BUG(15): load_from_file restores location/ssd_path but drops token_len (stays 0)"]
fn kv_registry_load_restores_token_len() -> R {
    let tmp = TempDir::new("kvreg_token_len");
    let src = KVRegistry::new();
    {
        let mut e = src.entries.write().unwrap();
        e[0].token_len = 1024;
        e[1].token_len = 300;
    }
    src.save_to_file(tmp.path())?;
    let dst = KVRegistry::new();
    dst.load_from_file(tmp.path())?;
    let e = dst.entries.read().unwrap();
    assert_eq!(e[0].token_len, 1024);
    assert_eq!(e[1].token_len, 300);
    Ok(())
}

#[test]
#[ignore = "BUG(15): save_to_file discards every write error and still returns Ok"]
fn kv_registry_save_reports_write_errors() -> R {
    let tmp = TempDir::new("kvreg_save_err");
    let missing = tmp.path().join("does").join("not").join("exist");
    let reg = KVRegistry::new();
    let res = reg.save_to_file(&missing);
    // 실패를 알리거나(Err), 폴더를 만들어 28 개 장부를 모두 썼거나 둘 중 하나여야 한다
    let written = (0..28).all(|l| missing.join(format!("layer{l}_meta.json")).is_file());
    assert!(res.is_err() || written, "save_to_file returned Ok although no layer meta was written");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. VisionEmbedCache — 인스턴스를 직접 만들어 TempDir 에서만 사용
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vision_cache_roundtrip_through_disk() -> R {
    let dev = Device::Cpu;
    let tmp = TempDir::new("vcache_roundtrip");
    let root = tmp.path().join("vision_embeds");
    let cache = VisionEmbedCache::new(root.clone(), u64::MAX);
    assert!(root.is_dir(), "cache dir is created eagerly");

    let px = Tensor::arange(0f32, 48., &dev)?.reshape((4, 12))?;
    let thw = Tensor::new(&[[1u32, 2, 2]], &dev)?;
    let key = VisionEmbedCache::compute_key(&px, &thw)?;
    let embeds = (Tensor::arange(0f32, 24., &dev)?.reshape((6, 4))? * 0.5)?;
    let ds = [embeds.affine(2.0, 1.0)?, embeds.affine(-1.0, 0.0)?];

    assert!(cache.try_load(key, &dev, DType::F32).is_none());
    assert_eq!(cache.stats(), (0, 0), "a lookup miss is not counted");
    cache.save(key, &embeds, &ds)?;
    assert_eq!(cache.stats(), (0, 1), "save counts as a miss");

    let entry = root.join(format!("{key:016x}"));
    for f in ["embeds.st", "deepstack_0.st", "deepstack_1.st", "meta.json"] {
        assert!(entry.join(f).is_file(), "{f} missing");
    }
    let tmp_left = std::fs::read_dir(&entry)?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .count();
    assert_eq!(tmp_left, 0, "temporary files must be renamed into place");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(entry.join("meta.json"))?)?;
    assert_eq!(meta["version"], 1);
    assert_eq!(meta["embed_dims"], json!([6, 4]));
    assert_eq!(meta["deepstack_count"], 2);

    // 저장은 F32, 복원은 요청한 dtype (값은 F16 에서도 정확히 표현됨)
    let (e, d) = cache.try_load(key, &dev, DType::F16).expect("cache hit");
    assert_eq!(e.dtype(), DType::F16);
    assert_eq!(d.len(), 2);
    assert!(d.iter().all(|t| t.dtype() == DType::F16));
    assert_close(&e, &embeds, 0.0, "embeds")?;
    assert_close(&d[0], &ds[0], 0.0, "deepstack 0")?;
    assert_close(&d[1], &ds[1], 0.0, "deepstack 1")?;
    assert_eq!(cache.stats(), (1, 1));
    assert!(cache.try_load(key ^ 1, &dev, DType::F32).is_none());
    assert_eq!(cache.stats(), (1, 1));

    // 프롬프트의 이미지 토큰 수와 맞는지 검증
    validate_embed_shape(&e, 6)?;
    assert!(validate_embed_shape(&e, 7).is_err());

    cache.clear_all();
    assert!(!entry.exists());
    assert!(cache.try_load(key, &dev, DType::F32).is_none());
    Ok(())
}

#[test]
fn vision_cache_index_survives_restart_and_purges_stale_versions() -> R {
    let dev = Device::Cpu;
    let tmp = TempDir::new("vcache_restart");
    let root = tmp.path().join("vision_embeds");
    let px = det(&[3, 8, 8], 5, 1.0)?;
    let thw = Tensor::new(&[[1u32, 4, 4]], &dev)?;
    let key = VisionEmbedCache::compute_key(&px, &thw)?;
    let embeds = det(&[4, 6], 6, 1.0)?;
    let ds = det(&[4, 6], 7, 1.0)?;
    {
        let first = VisionEmbedCache::new(root.clone(), u64::MAX);
        first.save(key, &embeds, &[ds.clone()])?;
    }
    // 이전 스키마(version 0) 항목과 키가 아닌 이름의 폴더
    let stale = root.join(format!("{:016x}", 0xABu64));
    std::fs::create_dir_all(&stale)?;
    std::fs::write(stale.join("meta.json"), br#"{"version":0,"embed_dims":[1,1],"deepstack_count":0}"#)?;
    let foreign = root.join("not-a-cache-key");
    std::fs::create_dir_all(&foreign)?;
    std::fs::write(foreign.join("meta.json"), br#"{"version":1,"embed_dims":[1,1],"deepstack_count":0}"#)?;

    // 재시작: 디스크를 스캔해 인덱스 복원
    let reopened = VisionEmbedCache::new(root.clone(), u64::MAX);
    assert_eq!(reopened.stats(), (0, 0));
    let (e, d) = reopened.try_load(key, &dev, DType::F32).expect("index rebuilt from disk");
    assert_close(&e, &embeds, 0.0, "embeds after restart")?;
    assert_eq!(d.len(), 1);
    assert_close(&d[0], &ds, 0.0, "deepstack after restart")?;
    assert_eq!(reopened.stats(), (1, 0));
    assert!(!stale.exists(), "outdated cache schema is deleted");
    assert!(reopened.try_load(0xAB, &dev, DType::F32).is_none());
    assert!(foreign.exists(), "folders that are not cache keys are left alone");
    Ok(())
}

#[test]
fn vision_cache_lru_eviction_respects_recent_hits() -> R {
    let dev = Device::Cpu;
    let tmp = TempDir::new("vcache_lru");
    let embeds = det(&[16, 8], 9, 1.0)?;
    // 항목 하나의 디스크 크기 (같은 shape 이면 같은 크기)
    let one = {
        let scratch = tmp.path().join("scratch");
        let scratch_cache = VisionEmbedCache::new(scratch.clone(), u64::MAX);
        scratch_cache.save(1, &embeds, &[])?;
        dir_bytes(&scratch.join(format!("{:016x}", 1u64)))
    };
    assert!(one > 0);

    let root = tmp.path().join("lru");
    let cache = VisionEmbedCache::new(root.clone(), 2 * one + one / 2); // 2.5 개 분량
    let pause = || std::thread::sleep(std::time::Duration::from_millis(20));
    let (a, b, c) = (0xA_u64, 0xB_u64, 0xC_u64);
    cache.save(a, &embeds, &[])?;
    pause();
    cache.save(b, &embeds, &[])?;
    pause();
    // A 를 다시 읽어 최근 사용으로 갱신 → 가장 오래 안 쓴 것은 B
    assert!(cache.try_load(a, &dev, DType::F32).is_some());
    pause();
    cache.save(c, &embeds, &[])?; // 3 개 > 상한 → B 하나만 제거
    assert_eq!(cache.stats(), (1, 3));
    assert!(!root.join(format!("{b:016x}")).exists(), "least recently used entry is deleted from disk");
    assert!(cache.try_load(b, &dev, DType::F32).is_none());
    assert!(cache.try_load(a, &dev, DType::F32).is_some());
    assert!(cache.try_load(c, &dev, DType::F32).is_some());
    Ok(())
}

#[test]
fn vision_cache_key_is_deterministic_and_input_sensitive() -> R {
    let dev = Device::Cpu;
    let px = det(&[3, 16, 16], 3, 1.0)?; // 768 원소 ≤ 4096 → 전 원소가 해시에 들어감
    let thw = Tensor::new(&[[1u32, 2, 2]], &dev)?;
    let k = VisionEmbedCache::compute_key(&px, &thw)?;
    assert_eq!(k, VisionEmbedCache::compute_key(&px.clone(), &thw)?);
    // shape / dtype / grid / 픽셀 하나만 달라도 다른 키
    assert_ne!(k, VisionEmbedCache::compute_key(&px.reshape((3, 8, 32))?, &thw)?);
    assert_ne!(k, VisionEmbedCache::compute_key(&px.to_dtype(DType::F16)?, &thw)?);
    assert_ne!(k, VisionEmbedCache::compute_key(&px, &Tensor::new(&[[1u32, 4, 1]], &dev)?)?);
    let mut v = f32s(&px)?;
    v[100] += 0.25;
    assert_ne!(k, VisionEmbedCache::compute_key(&Tensor::from_vec(v, (3, 16, 16), &dev)?, &thw)?);
    Ok(())
}

#[test]
#[ignore = "BUG(13): compute_key hashes only every (len/4096)-th value, so an image edited between samples hits a stale embedding"]
fn vision_cache_key_changes_when_any_pixel_changes() -> R {
    let dev = Device::Cpu;
    let n = 3 * 64 * 64; // stride = 3
    let base: Vec<f32> = (0..n).map(|i| (i % 251) as f32 / 251.0).collect();
    let mut edited = base.clone();
    edited[1] += 0.5; // 샘플 사이의 픽셀 하나 (문서 이미지의 글자 한 획 정도)
    let thw = Tensor::new(&[[1u32, 4, 4]], &dev)?;
    let ka = VisionEmbedCache::compute_key(&Tensor::from_vec(base, (3, 64, 64), &dev)?, &thw)?;
    let kb = VisionEmbedCache::compute_key(&Tensor::from_vec(edited, (3, 64, 64), &dev)?, &thw)?;
    assert_ne!(ka, kb, "different pixels must not share a cache entry");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. siglip2 — 전처리 / bbox / 설정 / 작은 가중치 비전 인코더
// ─────────────────────────────────────────────────────────────────────────────

fn tiny_siglip_cfg() -> Siglip2Config {
    Siglip2Config {
        vision_hidden_size: 8,
        vision_intermediate_size: 16,
        vision_num_layers: 2,
        vision_num_heads: 2,
        patch_size: 2,
        max_num_patches: 16,
        vision_layer_norm_eps: 1e-6,
        text_hidden_size: 8,
        text_intermediate_size: 16,
        text_num_layers: 1,
        text_num_heads: 2,
        text_vocab_size: 32,
        text_max_positions: 8,
        text_pad_token_id: 1,
        text_layer_norm_eps: 1e-6,
    }
}

fn put_layer_norm(w: &mut HashMap<String, Tensor>, name: &str, d: usize) -> R {
    w.insert(format!("{name}.weight"), Tensor::ones(d, DType::F32, &Device::Cpu)?);
    w.insert(format!("{name}.bias"), Tensor::zeros(d, DType::F32, &Device::Cpu)?);
    Ok(())
}

/// `vision_model.*` 접두사를 뗀 SigLIP2 비전 가중치 (LayerNorm 은 가중치 1, 편향 0)
fn siglip_vision_weights(cfg: &Siglip2Config) -> anyhow::Result<HashMap<String, Tensor>> {
    let (h, i) = (cfg.vision_hidden_size, cfg.vision_intermediate_size);
    let in_dim = cfg.patch_size * cfg.patch_size * 3;
    let mut w = HashMap::new();
    let mut seed = 1000u64;
    let mut linear = |w: &mut HashMap<String, Tensor>, name: &str, out_d: usize, in_d: usize| -> R {
        seed += 2;
        w.insert(format!("{name}.weight"), det(&[out_d, in_d], seed, 1.0 / (in_d as f32).sqrt())?);
        w.insert(format!("{name}.bias"), det(&[out_d], seed + 1, 0.1)?);
        Ok(())
    };
    linear(&mut w, "embeddings.patch_embedding", h, in_dim)?;
    w.insert("embeddings.position_embedding.weight".to_string(), det(&[cfg.max_num_patches, h], 77, 1.0)?);
    for l in 0..cfg.vision_num_layers {
        let p = format!("encoder.layers.{l}");
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            linear(&mut w, &format!("{p}.self_attn.{proj}"), h, h)?;
        }
        linear(&mut w, &format!("{p}.mlp.fc1"), i, h)?;
        linear(&mut w, &format!("{p}.mlp.fc2"), h, i)?;
        put_layer_norm(&mut w, &format!("{p}.layer_norm1"), h)?;
        put_layer_norm(&mut w, &format!("{p}.layer_norm2"), h)?;
    }
    put_layer_norm(&mut w, "post_layernorm", h)?;
    w.insert("head.probe".to_string(), det(&[1, 1, h], 78, 1.0)?);
    w.insert("head.attention.in_proj_weight".to_string(), det(&[3 * h, h], 79, 1.0 / (h as f32).sqrt())?);
    w.insert("head.attention.in_proj_bias".to_string(), det(&[3 * h], 80, 0.1)?);
    linear(&mut w, "head.attention.out_proj", h, h)?;
    put_layer_norm(&mut w, "head.layernorm", h)?;
    linear(&mut w, "head.mlp.fc1", i, h)?;
    linear(&mut w, "head.mlp.fc2", h, i)?;
    Ok(w)
}

#[test]
fn siglip2_patch_bbox_helpers() -> R {
    // 4 열 격자, 16px 패치, 리사이즈 배율 1
    assert_eq!(patch_index_to_bbox(5, 4, 16, 1.0, 1.0), (16, 16, 32, 32));
    // 원본 좌표로 되돌릴 때 x 는 2.5배, y 는 0.5배
    assert_eq!(patch_index_to_bbox(5, 4, 16, 2.5, 0.5), (40, 8, 80, 16));
    // grid_cols 0 은 1 열로 취급
    assert_eq!(patch_index_to_bbox(3, 0, 16, 1.0, 1.0), (0, 48, 16, 64));
    // 배율이 0 이어도 최소 1px 상자
    assert_eq!(patch_index_to_bbox(0, 4, 16, 0.0, 0.0), (0, 0, 1, 1));
    assert_eq!(patches_to_bounding_box(&[1, 6], 4, 16, 1.0, 1.0), Some((16, 0, 48, 32)));
    assert_eq!(patches_to_bounding_box(&[6, 1, 6], 4, 16, 1.0, 1.0), Some((16, 0, 48, 32)));
    assert_eq!(patches_to_bounding_box(&[], 4, 16, 1.0, 1.0), None);
    assert_eq!(patches_to_bounding_box(&[1], 0, 16, 1.0, 1.0), None);
    Ok(())
}

#[test]
fn siglip2_preprocess_fits_patch_budget_and_normalizes() -> R {
    let dev = Device::Cpu;
    // 100x50 흰 이미지, 패치 16 개 이하 → 비율을 지키는 가장 큰 격자 5x3 (80x48)
    let cfg = Siglip2Config { patch_size: 16, max_num_patches: 16, ..tiny_siglip_cfg() };
    let white = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(100, 50, image::Rgb([255, 255, 255])));
    let pre = preprocess_image(&white, &cfg, &dev)?;
    assert_eq!((pre.grid_cols, pre.grid_rows), (5, 3));
    assert!(pre.grid_cols * pre.grid_rows <= cfg.max_num_patches);
    assert_eq!(pre.pixel_values.dims(), &[1, 3, 48, 80]);
    assert_eq!((pre.orig_width, pre.orig_height), (100, 50));
    assert!((pre.scale_x - 1.25).abs() < 1e-12, "scale_x {}", pre.scale_x);
    assert!((pre.scale_y - 50.0 / 48.0).abs() < 1e-12, "scale_y {}", pre.scale_y);
    assert!(f32s(&pre.pixel_values)?.iter().all(|v| (v - 1.0).abs() < 1e-2), "white → +1");

    // 빨강 32x32, 패치 4 개 → 2x2 격자, 리사이즈 없음. 채널 평면(CHW) 순서와 [-1, 1] 정규화
    let cfg4 = Siglip2Config { patch_size: 16, max_num_patches: 4, ..tiny_siglip_cfg() };
    let red = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb([255, 0, 0])));
    let pre = preprocess_image(&red, &cfg4, &dev)?;
    assert_eq!((pre.grid_cols, pre.grid_rows), (2, 2));
    assert_eq!(pre.pixel_values.dims(), &[1, 3, 32, 32]);
    assert_eq!((pre.scale_x, pre.scale_y), (1.0, 1.0));
    let planes = pre.pixel_values.squeeze(0)?;
    for (ch, want) in [(0usize, 1.0f32), (1, -1.0), (2, -1.0)] {
        assert!(f32s(&planes.i(ch)?)?.iter().all(|v| (v - want).abs() < 1e-2), "channel {ch}");
    }
    Ok(())
}

#[test]
fn siglip2_config_from_json_defaults_and_overrides() -> R {
    let tmp = TempDir::new("siglip_cfg");
    let p = tmp.path().join("config.json");
    std::fs::write(&p, "{}")?;
    let d = Siglip2Config::from_json(&p)?;
    assert_eq!(
        (d.vision_hidden_size, d.vision_intermediate_size, d.vision_num_layers, d.vision_num_heads),
        (1152, 4304, 27, 16)
    );
    assert_eq!((d.patch_size, d.max_num_patches, d.pos_grid_side()), (16, 256, 16));
    assert_eq!((d.text_vocab_size, d.text_max_positions, d.text_pad_token_id), (256000, 64, 1));

    std::fs::write(
        &p,
        r#"{"vision_config": {"hidden_size": 768, "num_patches": 64, "patch_size": 14, "num_hidden_layers": 12},
            "text_config": {"max_position_embeddings": 128, "pad_token_id": 0}}"#,
    )?;
    let c = Siglip2Config::from_json(&p)?;
    assert_eq!((c.vision_hidden_size, c.patch_size, c.max_num_patches, c.vision_num_layers), (768, 14, 64, 12));
    assert_eq!(c.pos_grid_side(), 8);
    assert_eq!(c.vision_num_heads, 16, "missing keys keep the defaults");
    assert_eq!((c.text_max_positions, c.text_pad_token_id), (128, 0));
    // 격자 한 변은 최소 1
    assert_eq!(Siglip2Config { max_num_patches: 0, ..c.clone() }.pos_grid_side(), 1);

    std::fs::write(&p, "not json")?;
    assert!(Siglip2Config::from_json(&p).is_err());
    assert!(Siglip2Config::from_json(&tmp.path().join("missing.json")).is_err());
    Ok(())
}

/// HF NaFlex patchify 와 같은 (py, px, c) 순서로 패치를 펼쳐야 한다 (c 가 가장 빠름).
#[test]
fn siglip2_patch_embedding_flattens_rows_then_cols_then_channels() -> R {
    let dev = Device::Cpu;
    let cfg = Siglip2Config { vision_hidden_size: 12, patch_size: 2, ..tiny_siglip_cfg() };
    let mut w = HashMap::new();
    w.insert("weight".to_string(), Tensor::eye(12, DType::F32, &dev)?);
    w.insert("bias".to_string(), Tensor::zeros(12, DType::F32, &dev)?);
    let embed = Siglip2PatchEmbedding::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &dev))?;
    let (rows, cols) = (2usize, 3usize);
    let value = |c: usize, y: usize, x: usize| (100 * c + 10 * y + x) as f32;
    let mut data = Vec::new();
    for c in 0..3 {
        for y in 0..rows * 2 {
            for x in 0..cols * 2 {
                data.push(value(c, y, x));
            }
        }
    }
    let px = Tensor::from_vec(data, (1, 3, rows * 2, cols * 2), &dev)?;
    let out = embed.forward(&px)?;
    assert_eq!(out.dims(), &[1, rows * cols, 12]);
    let v = out.squeeze(0)?.to_vec2::<f32>()?;
    for r in 0..rows {
        for col in 0..cols {
            for py in 0..2 {
                for pxx in 0..2 {
                    for c in 0..3 {
                        assert_eq!(
                            v[r * cols + col][(py * 2 + pxx) * 3 + c],
                            value(c, 2 * r + py, 2 * col + pxx),
                            "patch ({r},{col}) py={py} px={pxx} c={c}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn siglip2_tiny_vision_encoder_forward() -> R {
    let dev = Device::Cpu;
    let cfg = tiny_siglip_cfg();
    let w = siglip_vision_weights(&cfg)?;
    let model = Siglip2VisionModel::new(&cfg, VarBuilder::from_tensors(w.clone(), DType::F32, &dev))?;
    let (rows, cols) = (2usize, 3usize);
    let (n, h) = (rows * cols, cfg.vision_hidden_size);
    let px = det(&[1, 3, rows * cfg.patch_size, cols * cfg.patch_size], 31, 1.0)?;
    let out = model.forward(&px, rows, cols)?;
    assert_eq!(out.patch_hidden.dims(), &[1, n, h]);
    assert_eq!(out.patch_shared.dims(), &[1, n, h]);
    assert_eq!(out.pooled.dims(), &[1, h]);
    for t in [&out.patch_hidden, &out.patch_shared, &out.pooled] {
        assert!(f32s(t)?.iter().all(|v| v.is_finite()));
    }
    // post_layernorm(가중치 1, 편향 0) 직후이므로 패치마다 평균 0, 분산 1
    for (r, row) in out.patch_hidden.squeeze(0)?.to_vec2::<f32>()?.iter().enumerate() {
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
        assert!(mean.abs() < 1e-4 && (var - 1.0).abs() < 1e-3, "patch {r}: mean {mean} var {var}");
    }
    // 같은 입력이면 같은 출력, 풀링 전용 경로와 편의 메서드도 같은 값
    let again = model.forward(&px, rows, cols)?;
    assert_close(&again.pooled, &out.pooled, 1e-6, "repeat pooled")?;
    assert_close(&again.patch_shared, &out.patch_shared, 1e-6, "repeat patch_shared")?;
    assert_close(&model.forward_pooled(&px, rows, cols)?, &out.pooled, 1e-6, "forward_pooled")?;
    let pe = model.get_patch_embeddings(&px, rows, cols)?;
    assert_eq!(pe.dims(), &[n, h]);
    assert_close(&pe, &out.patch_shared.squeeze(0)?, 1e-6, "get_patch_embeddings")?;
    // 같은 가중치로 다시 만든 모델도 같은 값, 다른 이미지는 다른 풀링 벡터
    let rebuilt = Siglip2VisionModel::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &dev))?;
    assert_close(&rebuilt.forward_pooled(&px, rows, cols)?, &out.pooled, 1e-6, "rebuilt model")?;
    let other = det(&[1, 3, rows * cfg.patch_size, cols * cfg.patch_size], 32, 1.0)?;
    assert!(max_abs_diff(&model.forward_pooled(&other, rows, cols)?, &out.pooled)? > 1e-4);
    Ok(())
}

/// NaFlex 위치 임베딩 보간은 align_corners=False 규약이어야 한다:
/// 원래 격자 크기면 칸 그대로, 절반이면 2x2 평균, 1x1 이면 중앙 2x2 평균.
#[test]
fn siglip2_position_grid_resampling_follows_align_corners_false() -> R {
    let dev = Device::Cpu;
    // 인코더 0 층 + 패치 투영 0 → patch_hidden = LayerNorm(보간된 위치 임베딩)
    let cfg = Siglip2Config { vision_num_layers: 0, ..tiny_siglip_cfg() };
    let (h, side) = (cfg.vision_hidden_size, cfg.pos_grid_side());
    assert_eq!(side, 4);
    let in_dim = cfg.patch_size * cfg.patch_size * 3;
    let mut w = siglip_vision_weights(&cfg)?;
    w.insert("embeddings.patch_embedding.weight".to_string(), Tensor::zeros((h, in_dim), DType::F32, &dev)?);
    w.insert("embeddings.patch_embedding.bias".to_string(), Tensor::zeros(h, DType::F32, &dev)?);
    let grid = det(&[side * side, h], 41, 1.0)?;
    w.insert("embeddings.position_embedding.weight".to_string(), grid.clone());
    let model = Siglip2VisionModel::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &dev))?;

    let g = grid.to_vec2::<f32>()?;
    let cell = |r: usize, c: usize| &g[r * side + c];
    let layer_norm = |v: &[f32]| -> Vec<f32> {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        v.iter().map(|x| (x - mean) / (var + 1e-6).sqrt()).collect()
    };
    let hidden = |rows: usize, cols: usize| -> anyhow::Result<Vec<Vec<f32>>> {
        let px = det(&[1, 3, rows * cfg.patch_size, cols * cfg.patch_size], 42, 1.0)?;
        Ok(model.forward(&px, rows, cols)?.patch_hidden.squeeze(0)?.to_vec2::<f32>()?)
    };
    let check = |got: &[f32], raw: Vec<f32>, what: &str| {
        for (k, (a, b)) in got.iter().zip(layer_norm(&raw)).enumerate() {
            assert!((a - b).abs() < 1e-4, "{what} [{k}]: {a} vs {b}");
        }
    };
    let avg4 = |cells: [(usize, usize); 4]| -> Vec<f32> {
        (0..h).map(|k| cells.iter().map(|&(r, c)| cell(r, c)[k]).sum::<f32>() / 4.0).collect()
    };

    let full = hidden(side, side)?;
    for r in 0..side {
        for c in 0..side {
            check(&full[r * side + c], cell(r, c).clone(), &format!("4x4 ({r},{c})"));
        }
    }
    let half = hidden(2, 2)?;
    for r in 0..2 {
        for c in 0..2 {
            let block = [(2 * r, 2 * c), (2 * r, 2 * c + 1), (2 * r + 1, 2 * c), (2 * r + 1, 2 * c + 1)];
            check(&half[r * 2 + c], avg4(block), &format!("2x2 ({r},{c})"));
        }
    }
    let one = hidden(1, 1)?;
    check(&one[0], avg4([(1, 1), (1, 2), (2, 1), (2, 2)]), "1x1");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. 설정 serde
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn qwen3_config_serde_defaults() -> R {
    let base = json!({
        "hidden_act": "silu", "hidden_size": 64, "head_dim": 16, "intermediate_size": 128,
        "max_position_embeddings": 512, "num_attention_heads": 4, "num_hidden_layers": 2,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 256,
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "rope_scaling": null
    });
    let cfg: Qwen3Config = serde_json::from_value(base.clone())?;
    assert_eq!(cfg.hidden_act, Activation::Silu);
    assert_eq!((cfg.bos_token_id, cfg.eos_token_id, cfg.max_window_layers), (None, None, None));
    assert_eq!(cfg.rope_theta, 10_000.0);
    assert_eq!(cfg.torch_dtype, "float16");
    assert!(!cfg.tie_word_embeddings && !cfg.attention_bias && !cfg.use_cache && !cfg.use_sliding_window);
    assert_eq!((cfg.attention_dropout, cfg.initializer_range), (0.0, 0.0));
    // 필수 필드가 빠지면 에러
    let mut missing = base.clone();
    assert!(missing.as_object_mut().unwrap().remove("head_dim").is_some());
    assert!(serde_json::from_value::<Qwen3Config>(missing).is_err());
    // 명시값은 그대로
    let mut explicit = base;
    explicit["rope_theta"] = json!(1_000_000.0);
    explicit["tie_word_embeddings"] = json!(true);
    explicit["eos_token_id"] = json!(151645);
    let cfg2: Qwen3Config = serde_json::from_value(explicit)?;
    assert_eq!((cfg2.rope_theta, cfg2.tie_word_embeddings, cfg2.eos_token_id), (1_000_000.0, true, Some(151645)));

    // generation_config.json: repetition_penalty 가 없으면 1.2
    let g: Qwen3GenerationConfig = serde_json::from_value(json!({
        "bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
        "temperature": 0.6, "top_k": 20, "top_p": 0.95, "transformers_version": "4.51.0"
    }))?;
    assert_eq!(g.repetition_penalty, 1.2);
    assert_eq!(g.eos_token_id, vec![151645, 151643]);
    assert!(g.do_sample);
    assert_eq!((g.top_k, g.temperature), (20, 0.6));
    let d = Qwen3GenerationConfig::default();
    assert_eq!((d.repetition_penalty, d.top_k, d.do_sample), (1.2, 80, false));
    assert_eq!(d.eos_token_id, vec![151643, 151645]);
    Ok(())
}

fn qwen3_5_text_json() -> serde_json::Value {
    json!({
        "attention_bias": false, "attention_dropout": 0, "attn_output_gate": true, "dtype": "float32",
        "eos_token_id": 0, "full_attention_interval": 2, "head_dim": 32, "hidden_act": "silu",
        "hidden_size": 64, "initializer_range": 0.02, "intermediate_size": 128,
        "layer_types": ["linear_attention", "full_attention"], "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16, "linear_num_key_heads": 1, "linear_num_value_heads": 1,
        "linear_value_head_dim": 16, "max_position_embeddings": 4096, "mlp_only_layers": [],
        "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": false, "num_attention_heads": 4,
        "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true, "use_cache": true, "vocab_size": 256, "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": true, "mrope_section": [2, 1, 1], "rope_type": "default",
            "rope_theta": 10000.0, "partial_rotary_factor": 0.25
        }
    })
}

#[test]
fn qwen3_5_text_and_multimodal_config_serde() -> R {
    let text = qwen3_5_text_json();
    let cfg: Qwen3_5TextConfig = serde_json::from_value(text.clone())?;
    assert_eq!(cfg.layer_types, vec!["linear_attention", "full_attention"]);
    assert_eq!(cfg.rope_parameters.mrope_section, vec![2, 1, 1]);
    assert!(cfg.rope_parameters.mrope_interleaved);
    assert_eq!(cfg.rope_parameters.partial_rotary_factor, 0.25);
    assert_eq!(cfg.tie_word_embeddings, Some(true));
    assert_eq!(cfg.hidden_act, Activation::Silu);
    // 모델이 쓰는 회전 차원 = head_dim × partial_rotary_factor 이고 M-RoPE 구간 합은 그 절반
    let rope_dim = (cfg.head_dim as f32 * cfg.rope_parameters.partial_rotary_factor) as usize;
    assert_eq!(rope_dim, 8);
    assert_eq!(cfg.rope_parameters.mrope_section.iter().sum::<usize>() * 2, rope_dim);

    let mut no_tie = text.clone();
    assert!(no_tie.as_object_mut().unwrap().remove("tie_word_embeddings").is_some());
    assert_eq!(serde_json::from_value::<Qwen3_5TextConfig>(no_tie)?.tie_word_embeddings, None);
    let mut no_rope = text.clone();
    assert!(no_rope.as_object_mut().unwrap().remove("rope_parameters").is_some());
    assert!(serde_json::from_value::<Qwen3_5TextConfig>(no_rope).is_err());

    // 최상위 config.json (text_config + vision_config)
    let full: Qwen3_5Config = serde_json::from_value(json!({
        "image_token_id": 250, "video_token_id": 251, "vision_start_token_id": 252, "vision_end_token_id": 253,
        "tie_word_embeddings": false, "text_config": text,
        "vision_config": {
            "deepstack_visual_indexes": [], "depth": 2, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 32,
            "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 64, "num_heads": 2,
            "num_position_embeddings": 64, "out_hidden_size": 64, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2
        }
    }))?;
    assert_eq!(full.text_config, cfg);
    assert_eq!(full.vision_config.hidden_act, Activation::GeluPytorchTanh);
    assert_eq!((full.image_token_id, full.video_token_id), (250, 251));
    assert!(!full.tie_word_embeddings);
    Ok(())
}

#[test]
fn qwen3vl_text_config_maps_onto_qwen3_config() -> R {
    let vl: Qwen3VLTextConfig = serde_json::from_value(json!({
        "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643, "dtype": "bfloat16",
        "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
        "initializer_range": 0.25, "intermediate_size": 6144, "max_position_embeddings": 262144,
        "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
        "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": true},
        "rope_theta": 5000000, "use_cache": true, "vocab_size": 151936
    }))?;
    // 디코더 층 구성에 쓰는 Qwen3Config 로 옮길 때 필드가 섞이거나 빠지지 않아야 한다
    let want: Qwen3Config = serde_json::from_value(json!({
        "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
        "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.25,
        "intermediate_size": 6144, "max_position_embeddings": 262144, "max_window_layers": 0,
        "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
        "rope_theta": 5000000, "tie_word_embeddings": true, "torch_dtype": "bfloat16", "use_cache": true,
        "use_sliding_window": false, "vocab_size": 151936
    }))?;
    assert_eq!(qwen3vl_text_config2qwen3_config(&vl), want);

    // 전처리 기본값은 HF preprocessor_config.json 과 같은 키 구조로 역직렬화된다 (모르는 키는 무시)
    let img: PreprocessorConfig = serde_json::from_value(json!({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16,
        "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "Qwen2VLImageProcessorFast"
    }))?;
    assert_eq!(img, PreprocessorConfig::qwen3_5_img_default());
    let video = PreprocessorConfig::qwen3_5_video_default();
    assert_ne!(video, img);
    assert_eq!((video.patch_size, video.merge_size, video.temporal_patch_size), (16, 2, 2));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. 임베딩 FloatModel (Granite/ModernBERT) — 작은 가중치 순전파
// ─────────────────────────────────────────────────────────────────────────────

fn float_model_weights(cfg: &EmbeddingConfig) -> anyhow::Result<HashMap<String, Tensor>> {
    let (h, i, v) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size);
    let dev = Device::Cpu;
    let mut w = HashMap::new();
    let mut seed = 500u64;
    let mut rnd = |shape: &[usize], scale: f32| -> anyhow::Result<Tensor> {
        seed += 1;
        det(shape, seed, scale)
    };
    let s_h = 1.0 / (h as f32).sqrt();
    w.insert("embeddings.tok_embeddings.weight".to_string(), rnd(&[v, h], 1.0)?);
    w.insert("embeddings.norm.weight".to_string(), Tensor::ones(h, DType::F32, &dev)?);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("layers.{l}");
        w.insert(format!("{p}.attn.Wqkv.weight"), rnd(&[3 * h, h], s_h)?);
        w.insert(format!("{p}.attn.Wo.weight"), rnd(&[h, h], s_h)?);
        w.insert(format!("{p}.mlp.Wi.weight"), rnd(&[2 * i, h], s_h)?);
        w.insert(format!("{p}.mlp.Wo.weight"), rnd(&[h, i], 1.0 / (i as f32).sqrt())?);
        w.insert(format!("{p}.mlp_norm.weight"), Tensor::ones(h, DType::F32, &dev)?);
        // Granite/ModernBERT 는 layers.0 에 attn_norm 이 없다
        if l > 0 {
            w.insert(format!("{p}.attn_norm.weight"), Tensor::ones(h, DType::F32, &dev)?);
        }
    }
    w.insert("final_norm.weight".to_string(), Tensor::ones(h, DType::F32, &dev)?);
    Ok(w)
}

#[test]
fn float_embedding_model_tiny_forward() -> R {
    let dev = Device::Cpu;
    let cfg_json = json!({
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "norm_eps": 1e-5, "vocab_size": 32, "model_type": "modernbert", "global_rope_theta": 160000.0
    });
    let cfg: EmbeddingConfig = serde_json::from_value(cfg_json.clone())?;
    assert_eq!(cfg.pad_token_id, None);
    let mut no_eps = cfg_json;
    assert!(no_eps.as_object_mut().unwrap().remove("norm_eps").is_some());
    assert!(serde_json::from_value::<EmbeddingConfig>(no_eps).is_err());

    let w = float_model_weights(&cfg)?;
    assert!(!w.contains_key("layers.0.attn_norm.weight"));
    let model = FloatModel::new(&cfg, VarBuilder::from_tensors(w.clone(), DType::F32, &dev), &dev)?;
    let ids = Tensor::new(&[1u32, 5, 9, 30], &dev)?;
    let out = model.forward(&ids, &dev)?;
    assert_eq!(out.dims(), &[1, 4, 8]);
    // final_norm(가중치 1) 출력은 토큰마다 평균 제곱 ≈ 1
    for (t, row) in out.squeeze(0)?.to_vec2::<f32>()?.iter().enumerate() {
        assert!(row.iter().all(|x| x.is_finite()), "token {t}");
        let ms = row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32;
        assert!((ms - 1.0).abs() < 1e-3, "token {t}: mean square {ms}");
    }
    assert_close(&model.forward(&ids, &dev)?, &out, 1e-6, "repeat forward")?;

    // 인코더(양방향) 어텐션: 마지막 토큰만 바꿔도 첫 토큰 출력이 바뀐다
    let out2 = model.forward(&Tensor::new(&[1u32, 5, 9, 31], &dev)?, &dev)?;
    let d0 = max_abs_diff(&out.i((0, 0))?, &out2.i((0, 0))?)?;
    assert!(d0 > 1e-4, "first token must attend to the last token, diff {d0}");

    // layers.0.attn_norm 이 있으면 실제로 적용된다 (없을 때와 결과가 다름)
    let mut w2 = w;
    w2.insert("layers.0.attn_norm.weight".to_string(), Tensor::ones(8, DType::F32, &dev)?.affine(2.0, 0.0)?);
    let model2 = FloatModel::new(&cfg, VarBuilder::from_tensors(w2, DType::F32, &dev), &dev)?;
    assert!(max_abs_diff(&model2.forward(&ids, &dev)?, &out)? > 1e-4);
    Ok(())
}
