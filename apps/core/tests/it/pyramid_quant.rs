//! 🌟 [PYRAMID-QUANT] 원본에서 한 번 양자화하는 블록 인코더(`ggml-kquant`)와 SigLIP2 정밀도 계획 · 지문 (CPU).
//!
//! - 비트 배치: candle 의 역양자화기가 그대로 읽는가 (배치가 틀리면 수십 % 어긋납니다)
//! - 척도 탐색: 가중 최소제곱이 candle 의 `from_float` 보다 오차가 크지 않은가, 같은 입력에서 quant.py 와 같은 척도인가
//! - F16 슈퍼 척도 1 ULP 올림: 슈퍼 척도 × 단계가 가장 큰 하위 척도를 덮는가
//! - 계획 · 지문: 기본 정밀도는 예전 키, 다른 정밀도는 꼬리표가 붙는가

use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType};
use candle_core::Device;
use ggml_kquant::{encode, fit_asymmetric, fit_symmetric, importance, rel, Format};

type R = anyhow::Result<()>;

/// 결정적인 표준 정규 난수 (64비트 LCG + Box–Muller)
fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n).map(|_| ((-2.0 * next().ln()).sqrt() * (std::f64::consts::TAU * next()).cos()) as f32).collect()
}

/// 드물게 큰 값이 섞인 행 (실제 가중치처럼 꼬리가 두꺼움)
fn heavy(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut xs = normals(rows * cols, seed);
    for (i, x) in xs.iter_mut().enumerate() {
        *x *= 0.02;
        if i % 97 == 0 {
            *x *= 6.0;
        }
    }
    xs
}

fn ggml(f: Format) -> GgmlDType {
    match f {
        Format::Q8_0 => GgmlDType::Q8_0,
        Format::Q6K => GgmlDType::Q6K,
        Format::Q5K => GgmlDType::Q5K,
        Format::Q4K => GgmlDType::Q4K,
        Format::F16 => GgmlDType::F16,
        Format::F32 => GgmlDType::F32,
    }
}

/// candle 의 역양자화기로 읽습니다.
fn decode(f: Format, bytes: &[u8], rows: usize, cols: usize) -> anyhow::Result<Vec<f32>> {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
    raw[..bytes.len()].copy_from_slice(bytes);
    let q = qtensor_from_ggml(ggml(f), &raw[..bytes.len()], vec![rows, cols], &Device::Cpu)?;
    Ok(q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

/// candle 자신의 양자화기 (`from_float`)
fn candle_plain(xs: &[f32], f: Format) -> Vec<u8> {
    let mut q = ggml(f).cpu_zeros(xs.len());
    q.from_float(xs);
    unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) }.to_vec()
}

const BLOCKS: [Format; 4] = [Format::Q4K, Format::Q5K, Format::Q6K, Format::Q8_0];

#[test]
fn block_sizes_match_ggml() {
    for f in BLOCKS {
        let g = ggml(f);
        assert_eq!((f.block_size(), f.type_size()), (g.block_size(), g.type_size()), "{f:?}");
    }
}

#[test]
fn candle_reads_the_layout_and_the_search_is_not_worse_than_candle() -> R {
    let cols = 1024;
    let xs = heavy(32, cols, 7);
    let mut errs = Vec::new();
    for f in BLOCKS {
        let ours = encode(&xs, cols, f, None).map_err(anyhow::Error::msg)?;
        assert_eq!(ours.len(), xs.len() / f.block_size() * f.type_size(), "{f:?} 바이트 수");
        let mine = rel(&decode(f, &ours, 32, cols)?, &xs);
        let theirs = rel(&decode(f, &candle_plain(&xs, f), 32, cols)?, &xs);
        // 배치가 틀리면 수십 % 어긋납니다 (32 블록의 4비트 오차는 보통 9 % 안팎).
        let bound = match f {
            Format::Q4K => 0.11,
            Format::Q5K => 0.055,
            Format::Q6K => 0.03,
            _ => 0.008,
        };
        assert!(mine < bound, "{f:?} 오차 {mine}");
        assert!(mine <= theirs * 1.02, "{f:?}: 가중 최소제곱 {mine:.5}, candle {theirs:.5}");
        errs.push(mine);
    }
    assert!(errs.windows(2).all(|w| w[0] > w[1]), "비트가 많을수록 오차가 작아야 합니다: {errs:?}");
    Ok(())
}

#[test]
fn block_fits_match_quant_py() {
    // gravity_ocr/quant.py 를 같은 입력(numpy f32)에 돌린 값 (blueprint-sandbox 에서 잰 것).
    let x: Vec<f32> = (0..32).map(|i| (i as f32 * 1.3).sin() * 0.05 + if i % 7 == 0 { 0.2 } else { 0.0 }).collect();
    let mut w = [0f32; 32];
    importance(&x, None, &mut w);
    let close = |got: f32, want: f32| (got - want).abs() <= want.abs() * 2e-5;
    let (d, m) = fit_asymmetric(&x, &w, 15);
    assert!(close(d, 1.925078221e-02) && close(m, 4.800733924e-02), "q4 {d} {m}");
    let (d, m) = fit_asymmetric(&x, &w, 31);
    assert!(close(d, 9.597460739e-03) && close(m, 4.951747879e-02), "q5 {d} {m}");
    let x16: Vec<f32> = x[..16].iter().map(|v| v - 0.01).collect();
    let mut w16 = [0f32; 16];
    importance(&x16, None, &mut w16);
    assert!(close(fit_symmetric(&x16, &w16, 31, -32), 6.606251933e-03), "q6");
    // quant.py 의 q8 은 −128 까지 씁니다 (여기 Q8_0 은 GGML 처럼 −127).
    assert!(close(fit_symmetric(&x, &w, 127, -128), 1.904964098e-03), "q8");
}

#[test]
fn the_largest_sub_block_keeps_its_scale() -> R {
    // 하위 블록 하나만 20배 큰 슈퍼블록. F16 슈퍼 척도를 올림으로 저장하므로 그 블록의 정수 척도가 단계 상한(63 · 127)에서
    // 잘리지 않습니다 — 잘리면 그 블록 값이 통째로 줄어 candle 보다 크게 어긋납니다.
    let cols = 256;
    for scale in [0.0123_f32, 0.0371, 0.10007, 1.7311] {
        let mut x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).sin() * scale * 0.05).collect();
        for v in x[96..128].iter_mut() {
            *v *= 20.0;
        }
        for f in [Format::Q4K, Format::Q5K, Format::Q6K] {
            let mine = decode(f, &encode(&x, cols, f, None).map_err(anyhow::Error::msg)?, 1, cols)?;
            let theirs = decode(f, &candle_plain(&x, f), 1, cols)?;
            let (a, b) = (rel(&mine[96..128], &x[96..128]), rel(&theirs[96..128], &x[96..128]));
            assert!(a <= b * 1.05 + 1e-4, "{f:?} 척도 {scale}: 큰 하위 블록 오차 {a:.5} (candle {b:.5})");
        }
    }
    Ok(())
}

#[test]
fn zero_and_constant_rows_stay_finite() -> R {
    let cols = 256;
    let mut xs = vec![0f32; cols];
    xs.extend(std::iter::repeat_n(0.03f32, cols));
    xs.extend(std::iter::repeat_n(-0.03f32, cols));
    for f in BLOCKS {
        let back = decode(f, &encode(&xs, cols, f, None).map_err(anyhow::Error::msg)?, 3, cols)?;
        assert!(back.iter().all(|v| v.is_finite()), "{f:?}");
        assert!(back[..cols].iter().all(|v| *v == 0.0), "{f:?}: 0 행은 0");
        assert!(rel(&back[cols..], &xs[cols..]) < 0.02, "{f:?}: 상수 행");
    }
    Ok(())
}

#[test]
fn misaligned_or_unknown_requests_are_refused() {
    assert!(encode(&[0.0; 200], 100, Format::Q8_0, None).is_err(), "블록 배수가 아닌 열");
    assert!(encode(&[0.0; 10], 3, Format::F32, None).is_err(), "행으로 안 나뉨");
    assert!(encode(&[0.0; 256], 256, Format::Q4K, Some(&[1.0; 10])).is_err(), "중요도 길이");
}

#[test]
fn siglip2_widths_round_trip_through_zero_padding() -> R {
    // so400m 의 폭: 1152 (K 형식 256 으로 안 나뉨) · 4304 (Q8_0 32 로도 안 나뉨).
    use logis_app::models::siglip2::quant::{round_trip, QType};
    for cols in [1152usize, 4304] {
        let xs = heavy(4, cols, 3);
        for q in [QType::Q8_0, QType::Q6K, QType::Q4K] {
            let back = round_trip(&xs, 4, cols, q)?;
            assert_eq!(back.len(), xs.len());
            assert!(rel(&back, &xs) < 0.11, "{cols} {q:?}");
        }
    }
    Ok(())
}

#[test]
fn precision_tags_keep_the_default_keys_and_tell_plans_apart() {
    use logis_app::models::siglip2::phrase_cache::PhraseCache;
    use logis_app::models::siglip2::quant::{dtype_for, weights_tag};
    // CPU 는 계획과 무관하게 원본 F32 (앱의 기존 규칙).
    assert_eq!(weights_tag(&Device::Cpu), "f32");
    assert_eq!(dtype_for(&Device::Cpu), candle_core::DType::F32);
    // 꼬리표가 없으면(기본 정밀도) 예전 키와 같은 계산이고, 계획마다 키가 갈립니다.
    let plain = PhraseCache::key_with("Invoice No.", None);
    assert_eq!(plain, PhraseCache::key_with("Invoice No.", None));
    assert_ne!(plain, PhraseCache::key_with("Invoice No.", Some("q8-1234abcd")));
    assert_ne!(PhraseCache::key_with("Invoice No.", Some("q8-1234abcd")), PhraseCache::key_with("Invoice No.", Some("pyr-1234abcd")));
}

#[test]
fn a_quantized_tower_pads_the_mlp_and_multiplies_like_its_blocks() -> R {
    use candle_core::{Module, Tensor};
    use std::collections::HashMap;
    use logis_app::models::siglip2::gguf;
    use logis_app::models::siglip2::quant::{self, QType, WeightPlan};

    let dir = std::env::temp_dir().join(format!("logis-pyramid-quant-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let dev = Device::Cpu;
    // fc2 의 입력 40 은 Q8_0 블록(32)으로 안 나뉩니다 → fc1 에 0 행을 붙여 64 로 냅니다 (so400m 의 4304 → 4320 과 같은 꼴).
    let (h, inter) = (64usize, 40usize);
    let mat = |r: usize, c: usize, seed: u64| -> anyhow::Result<Tensor> {
        Ok(Tensor::from_vec(normals(r * c, seed).iter().map(|v| v * 0.05).collect::<Vec<f32>>(), (r, c), &dev)?)
    };
    let vec1 = |n: usize, seed: u64| -> anyhow::Result<Tensor> { Ok(Tensor::from_vec(normals(n, seed).iter().map(|v| v * 0.1).collect::<Vec<f32>>(), n, &dev)?) };
    let p = "vision_model.encoder.layers.0.mlp";
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(format!("{p}.fc1.weight"), mat(inter, h, 1)?);
    m.insert(format!("{p}.fc1.bias"), vec1(inter, 2)?);
    m.insert(format!("{p}.fc2.weight"), mat(h, inter, 3)?);
    m.insert(format!("{p}.fc2.bias"), vec1(h, 4)?);
    m.insert("vision_model.post_layernorm.weight".into(), vec1(h, 5)?);
    m.insert("vision_model.head.attention.in_proj_weight".into(), mat(3 * h, h, 6)?);
    m.insert("vision_model.head.attention.in_proj_bias".into(), vec1(3 * h, 7)?);
    m.insert("text_model.head.weight".into(), mat(h, h, 8)?);
    m.insert("text_model.head.bias".into(), vec1(h, 9)?);
    m.insert("text_model.embeddings.token_embedding.weight".into(), mat(16, h, 10)?);
    m.insert("logit_scale".into(), Tensor::new(&[2.5f32], &dev)?);
    candle_core::safetensors::save(&m, dir.join("model.safetensors"))?;

    let plan = WeightPlan::q8();
    gguf::prepare_quant(&dir, &plan)?;
    assert!(gguf::quant_ready(&dir, &plan));
    let f = gguf::open_cached(&gguf::quant_vision_path(&dir, &plan))?;
    assert_eq!(f.info(&format!("{p}.fc1.weight")), Some((GgmlDType::Q8_0, vec![64, 64])), "fc1 은 0 행을 붙여 64 행");
    assert_eq!(f.info(&format!("{p}.fc1.bias")), Some((GgmlDType::F32, vec![64])));
    assert_eq!(f.info(&format!("{p}.fc2.weight")), Some((GgmlDType::Q8_0, vec![64, 64])), "fc2 는 0 열을 붙여 64 열");
    assert_eq!(f.info("vision_model.post_layernorm.weight"), Some((GgmlDType::F32, vec![h])), "정규화는 F32");
    assert!(f.info("vision_model.head.attention.in_proj_q.weight").is_some() && f.info("vision_model.head.attention.in_proj_weight").is_none(), "in_proj 는 q · k · v 로 나뉨");
    let text = gguf::open_cached(&gguf::quant_text_path(&dir, &plan))?;
    assert!(text.info("text_model.embeddings.token_embedding.weight").is_none(), "토큰 표는 원본에서 행만 읽음");

    // 파일에서 올린 MLP vs 같은 블록을 풀어 0 채움 없이 곱한 기준
    let vb = f.var_builder(&[], &dev)?;
    let fc1 = quant::linear(&vb.pp(format!("{p}.fc1")), Some(f.as_ref()), h, inter)?;
    let fc2 = quant::linear(&vb.pp(format!("{p}.fc2")), Some(f.as_ref()), inter, h)?;
    let x = Tensor::from_vec(normals(3 * h, 11), (3, h), &dev)?;
    let y: Vec<f32> = fc2.forward(&fc1.forward(&x)?.gelu()?)?.flatten_all()?.to_vec1()?;
    let vals = |t: &Tensor| -> anyhow::Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1::<f32>()?) };
    let w1 = Tensor::from_vec(quant::round_trip(&vals(&m[&format!("{p}.fc1.weight")])?, inter, h, QType::Q8_0)?, (inter, h), &dev)?;
    let w2 = Tensor::from_vec(quant::round_trip(&vals(&m[&format!("{p}.fc2.weight")])?, h, inter, QType::Q8_0)?, (h, inter), &dev)?;
    let hid = x.matmul(&w1.t()?)?.broadcast_add(&m[&format!("{p}.fc1.bias")])?.gelu()?;
    let want: Vec<f32> = hid.matmul(&w2.t()?)?.broadcast_add(&m[&format!("{p}.fc2.bias")])?.flatten_all()?.to_vec1()?;
    assert!(rel(&y, &want) < 1e-5, "0 채움 MLP 오차 {}", rel(&y, &want));

    drop((f, text, vb, fc1, fc2));
    gguf::close_handles(Some(&dir));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
