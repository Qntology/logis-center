//! Qwen4-Exp candle 포트 ↔ transformers 5.17 기준 logits 대조 (작은 무작위 체크포인트).
//!
//! 픽스처: `tests/fixtures/qwen4_exp_tiny/` (생성기: 같은 폴더의 `gen_tiny_ref.py`, `python -P gen_tiny_ref.py <dir>`)
//!  - 변형 A (루트): qwen4-spec §13 검증 설정. save_pretrained 키 형식 (전문가별 `experts.{e}.*`, n-gram 샤드 4개,
//!    int64 PLE 버퍼, layer_types 가 `qwen_sparse_attention` 으로 재매핑된 config).
//!  - 변형 B (`variant_b/`): VLM 래퍼 config (`text_config`, `"full_attention"`), `model.language_model.*` 접두사,
//!    융합 전문가 `gate_up_proj`, 단일 n-gram 표, sigmoid 출력 게이트, hc 3, QSA 2층, PLE 2층, eos 리스트,
//!    Dk≠Dv, conv kernel 3, norm_topk_prob=false, partial_rotary_factor 없음(→1.0).
//! `ref.safetensors`: input_ids, schedule(증분 청크), full_logits(한 번에), inc_logits(프리필+증분, HF 캐시),
//! hidden_{i}(층 출력), mixer, ngram_ids_{i}(PLE), qsa_mask_{i}(QSA 선택 마스크).
//! 판정: max |Δlogit| ≤ 1e-4 (전체·증분 모두), argmax 일치 100 %, QSA 선택 마스크·n-gram id 완전 일치.
//! PLE n-gram 표는 dense(장치 상주) 경로와 mmap 행 수집 경로(전체 크기 모델이 쓰는 경로, `ple_dense_limit_bytes = 0`)
//! 둘 다 같은 기준으로 검사합니다. 앱 등록 경로(`lang_llm` 로더 = `registry::load_lang_runtime`) 도 확인합니다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use crate::common::TempDir;
use tauri_app_lib::models::qwen4_exp::config::LayerKind;
use tauri_app_lib::models::qwen4_exp::model::{LoadOptions, Mixer};
use tauri_app_lib::models::qwen4_exp::ple::{splitmix64, NgramHasher};
use tauri_app_lib::models::qwen4_exp::qsa::selection_mask;
use tauri_app_lib::models::qwen4_exp::{
    detect_text_arch, Qwen4ExpForCausalLM, Qwen4ExpTextConfig, TextArch, TextGenerator,
};

type R = anyhow::Result<()>;

const TOL: f32 = 1e-4;

fn fixture(variant: &str) -> PathBuf {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4_exp_tiny");
    if variant.is_empty() { base } else { base.join(variant) }
}

fn f32s(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    assert_eq!(a.dims(), b.dims(), "shape mismatch");
    let (x, y) = (f32s(a)?, f32s(b)?);
    Ok(x.iter().zip(&y).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max))
}

/// 마지막 축 argmax 일치 비율
fn argmax_agree(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a = a.to_device(&Device::Cpu)?.argmax(candle_core::D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let b = b.to_device(&Device::Cpu)?.argmax(candle_core::D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    Ok(same as f32 / a.len() as f32)
}

struct Reference {
    ids: Vec<u32>,
    schedule: Vec<usize>,
    full: Tensor,
    inc: Tensor,
    t: HashMap<String, Tensor>,
}

fn load_ref(dir: &Path) -> anyhow::Result<Reference> {
    let t = candle_core::safetensors::load(dir.join("ref.safetensors"), &Device::Cpu)?;
    let ids: Vec<u32> = t["input_ids"].flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect();
    let schedule: Vec<usize> = t["schedule"].to_vec1::<i64>()?.into_iter().map(|v| v as usize).collect();
    Ok(Reference { ids, schedule, full: t["full_logits"].clone(), inc: t["inc_logits"].clone(), t })
}

#[derive(Debug)]
struct Report {
    full_diff: f32,
    full_agree: f32,
    inc_diff: f32,
    inc_agree: f32,
    layer_diffs: Vec<f32>,
    mixer_diff: f32,
}

/// dense 표 경로 (픽스처 표는 수십 KB 라 항상 장치에 올라갑니다)
fn dense_opts() -> LoadOptions {
    LoadOptions { ple_dense_limit_bytes: 512 * 1024 * 1024, max_resident_bytes: 6_000_000_000 }
}

/// mmap 행 수집 경로 강제 (전체 크기 모델의 ~82 GB/층 표가 쓰는 경로)
fn mmap_opts() -> LoadOptions {
    LoadOptions { ple_dense_limit_bytes: 0, max_resident_bytes: 6_000_000_000 }
}

/// 한 변형을 주어진 장치에서 돌려 HF 기준과 비교합니다.
fn run_variant(variant: &str, device: &Device, opts: &LoadOptions) -> anyhow::Result<Report> {
    let dir = fixture(variant);
    let rf = load_ref(&dir)?;
    let mut model = Qwen4ExpForCausalLM::load_with(&dir, DType::F32, device, opts)?;
    let mmap_tables = model.layers.iter().filter_map(|l| l.ple.as_ref()).filter(|p| p.is_mmap_table()).count();
    let ple_layers = model.layers.iter().filter(|l| l.ple.is_some()).count();
    if opts.ple_dense_limit_bytes == 0 {
        assert_eq!(mmap_tables, ple_layers, "[{variant}] mmap 경로가 강제되지 않았습니다");
    } else {
        assert_eq!(mmap_tables, 0, "[{variant}] 작은 표가 mmap 경로로 갔습니다");
    }
    // QSA 선택 결과는 기본으로 버립니다 (긴 프리필 메모리). 검증용으로만 켭니다.
    for layer in model.layers.iter_mut() {
        if let Mixer::Qsa(a) = &mut layer.mixer {
            assert!(!a.record_selection, "record_selection 기본값은 false 여야 합니다");
            a.record_selection = true;
        }
    }
    let n = rf.ids.len();
    let ids = Tensor::from_slice(rf.ids.as_slice(), (1, n), device)?;

    // ── 1. 전체 forward (캐시 없이 한 번에) + 층별 출력 ──
    let mut trace = Vec::new();
    let h = model.forward_hidden_traced(&ids, 0, Some(&mut trace))?;
    let logits = model.lm_head_forward(&h)?;
    let layer_diffs = trace
        .iter()
        .enumerate()
        .map(|(i, x)| max_abs_diff(x, &rf.t[&format!("hidden_{i}")]))
        .collect::<anyhow::Result<Vec<f32>>>()?;
    let mixer_diff = max_abs_diff(&h, &rf.t["mixer"])?;
    let full_diff = max_abs_diff(&logits, &rf.full)?;
    let full_agree = argmax_agree(&logits, &rf.full)?;

    // QSA 선택 마스크 (전체 forward) 는 HF 와 완전히 같아야 합니다.
    for (li, layer) in model.layers.iter().enumerate() {
        if let Mixer::Qsa(a) = &layer.mixer {
            let want = &rf.t[&format!("qsa_mask_{li}")];
            let want = want.to_vec2::<u8>()?;
            let got = selection_mask(&a.last_selection[0], 0, n);
            assert_eq!(got, want, "[{variant}] layer {li} QSA 선택 마스크 불일치");
        }
        if let Some(p) = &layer.ple {
            let want: Vec<u32> = rf.t[&format!("ngram_ids_{li}")]
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let ctx = vec![p.hasher.eos; p.hasher.n - 1];
            assert_eq!(p.hasher.ids(&ctx, &rf.ids), want, "[{variant}] layer {li} n-gram id 불일치");
        }
    }

    // ── 2. 프리필 + 증분 디코드 (HF 와 같은 청크 스케줄) ──
    model.clear_cache();
    let mut parts = Vec::new();
    let mut pos = 0;
    for &c in &rf.schedule {
        let chunk = Tensor::from_slice(&rf.ids[pos..pos + c], (1, c), device)?;
        parts.push(model.forward(&chunk, pos)?);
        pos += c;
    }
    assert_eq!(pos, n);
    let inc = Tensor::cat(&parts, 1)?;
    let inc_diff = max_abs_diff(&inc, &rf.inc)?;
    let inc_agree = argmax_agree(&inc, &rf.inc)?;
    // 같은 포트의 전체 vs 증분 (HF 를 거치지 않은 자기 일관성)
    let self_diff = max_abs_diff(&inc, &logits)?;

    let rep = Report { full_diff, full_agree, inc_diff, inc_agree, layer_diffs, mixer_diff };
    println!(
        "[QWEN4-EXP] {:>9} on {:?} (PLE 표 {}): full |Δ| {:.3e} (argmax {:.1}%) · incremental |Δ| {:.3e} (argmax {:.1}%) · port full↔inc {:.3e} · layers {:?} · mixer {:.3e}",
        if variant.is_empty() { "variant_a" } else { variant },
        device.location(),
        if mmap_tables > 0 { "mmap 행 수집" } else { "dense" },
        rep.full_diff,
        rep.full_agree * 100.0,
        rep.inc_diff,
        rep.inc_agree * 100.0,
        self_diff,
        rep.layer_diffs.iter().map(|d| format!("{d:.2e}")).collect::<Vec<_>>(),
        rep.mixer_diff
    );
    Ok(rep)
}

fn assert_report(variant: &str, r: &Report) {
    assert!(r.full_diff <= TOL, "[{variant}] full logits max |Δ| {} > {TOL}", r.full_diff);
    assert!(r.inc_diff <= TOL, "[{variant}] incremental logits max |Δ| {} > {TOL}", r.inc_diff);
    assert_eq!(r.full_agree, 1.0, "[{variant}] full argmax 일치율 {}", r.full_agree);
    assert_eq!(r.inc_agree, 1.0, "[{variant}] incremental argmax 일치율 {}", r.inc_agree);
}

#[test]
fn qwen4_exp_ngram_hash_matches_spec_vectors() {
    // 테스트 설정 (vocab 512, base 97, Hpn 2, seed 1234, PLE#0, eos 2)
    let h = NgramHasher::new(512, 3, 2, 97, 128, 1234, 0, 2);
    assert_eq!(h.multipliers, vec![4295158889763781, 17440218147737327, 1270917961434445]);
    assert_eq!(h.head_sizes, vec![97, 101, 103, 107]);
    assert_eq!(h.head_offsets, vec![0, 97, 198, 301]);
    assert_eq!(h.padded_rows, 512);
    let ids = [180u32, 496, 430, 186, 2, 2, 121, 267, 265, 2, 338, 449];
    let want: Vec<u32> = [
        [90, 185, 202, 366], [53, 163, 273, 322], [29, 139, 251, 337], [68, 124, 233, 319],
        [48, 108, 278, 304], [68, 139, 201, 372], [54, 108, 202, 311], [53, 131, 294, 318],
        [77, 107, 241, 393], [35, 182, 244, 355], [56, 121, 291, 376], [42, 167, 202, 309],
    ]
    .iter()
    .flatten()
    .copied()
    .collect();
    assert_eq!(h.ids(&[2, 2], &ids), want);

    // 증분: 문맥을 이어 붙여 한 토큰씩 넣어도 같아야 합니다.
    let mut ctx = vec![2u32, 2];
    let mut got = Vec::new();
    for &t in &ids {
        got.extend(h.ids(&ctx, &[t]));
        ctx = vec![ctx[1], t];
    }
    assert_eq!(got, want);

    // 기본 설정 (vocab 248320, seed 1234, Hpn 8, base 2e7)
    let d0 = NgramHasher::new(248_320, 3, 8, 20_000_000, 128, 1234, 0, 248_046);
    assert_eq!(d0.multipliers, vec![23703573157769, 20109073645365, 8052911324071]);
    assert_eq!(
        d0.head_sizes,
        vec![
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
            20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
        ]
    );
    assert_eq!(d0.padded_rows, 320_001_536);
    let d1 = NgramHasher::new(248_320, 3, 8, 20_000_000, 128, 1234, 1, 248_046);
    assert_eq!(d1.multipliers, vec![3352040966061, 19752442147475, 10742103100073]);
    assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
}

#[test]
fn qwen4_exp_config_parsing() -> R {
    let a = Qwen4ExpTextConfig::from_file(&fixture("").join("config.json"))?;
    assert_eq!(a.num_hidden_layers, 4);
    assert_eq!(a.layer_kind(0), LayerKind::Linear);
    assert_eq!(a.layer_kind(3), LayerKind::Qsa);
    assert_eq!(a.rotary_dim(), 8);
    assert_eq!(a.ple_index(1), Some(0));
    assert_eq!(a.ple_index(0), None);
    assert_eq!(a.ple_eos(), 2);

    let b = Qwen4ExpTextConfig::from_file(&fixture("variant_b").join("config.json"))?;
    assert_eq!(b.hc_count, 3);
    assert_eq!(b.layer_kind(1), LayerKind::Qsa, "\"full_attention\" → QSA 재매핑");
    assert_eq!(b.rotary_dim(), 16, "partial_rotary_factor 없음 → 1.0");
    assert_eq!(b.rope_theta(), 50_000.0);
    assert_eq!(b.ple_ids(), &[1, 3]);
    assert_eq!(b.ple_index(2), Some(1));
    assert_eq!(b.ple_eos(), 5, "eos 리스트의 첫 원소");
    assert_eq!(b.eos_ids(), vec![5, 7]);

    // 기본값 + layer_types 생성 + 검증 실패 경로
    let d = Qwen4ExpTextConfig::from_value(&json!({
        "model_type": "qwen4_exp_text", "num_hidden_layers": 8,
        "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 256,
        "indexer_budget": 2048, "indexer_compress_ratio": 4, "eos_token_id": [248046, 248044]
    }))?;
    assert_eq!(d.head_dim(), 256);
    assert_eq!(d.rotary_dim(), 256, "Qwen3.5 와 달리 기본 partial_rotary_factor = 1.0");
    assert_eq!(d.rope_theta(), 10_000.0);
    assert_eq!(
        (0..8).map(|i| d.layer_kind(i)).filter(|k| *k == LayerKind::Qsa).count(),
        2,
        "full_attention_interval 4"
    );
    let e = Qwen4ExpTextConfig::from_value(&json!({
        "num_hidden_layers": 4, "rope_parameters": {"rope_theta": 1.0e7, "partial_rotary_factor": 0.25},
        "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
        "indexer_budget": 2048, "indexer_compress_ratio": 4
    }))?;
    assert_eq!((e.rotary_dim(), e.rope_theta()), (64, 1.0e7));

    // 상주 가중치 추정 (전체 크기 모델을 통째로 올리려다 OOM 나기 전에 막는 가드의 입력)
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(fixture("").join("model.safetensors"))? };
    let limit = dense_opts().ple_dense_limit_bytes;
    let bytes = tauri_app_lib::models::qwen4_exp::model::resident_weight_bytes(&st, DType::F32, limit);
    let file = std::fs::metadata(fixture("").join("model.safetensors"))?.len() as usize;
    assert!(bytes <= file && bytes + 64 * 1024 > file, "추정 {bytes} vs 파일 {file}");
    assert_eq!(tauri_app_lib::models::qwen4_exp::model::resident_weight_bytes(&st, DType::BF16, limit) * 2, bytes);
    // mmap 행 수집으로 돌릴 표는 상주 추정에서 빠집니다.
    let without_tables = tauri_app_lib::models::qwen4_exp::model::resident_weight_bytes(&st, DType::F32, 0);
    assert!(without_tables < bytes, "{without_tables} vs {bytes}");
    // rotary_dim(256) > indexer_head_dim(128) 은 HF 와 같이 거부
    assert!(Qwen4ExpTextConfig::from_value(&json!({
        "num_hidden_layers": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 64,
        "indexer_budget": 8, "indexer_compress_ratio": 4
    }))
    .is_err());

    // 포트가 구현하지 않은 변형은 조용히 틀린 logits 대신 명확한 오류로 거부합니다.
    let base = json!({
        "num_hidden_layers": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 256,
        "indexer_budget": 8, "indexer_compress_ratio": 4
    });
    let with = |k: &str, v: serde_json::Value| {
        let mut c = base.clone();
        c[k] = v;
        Qwen4ExpTextConfig::from_value(&c)
    };
    assert!(Qwen4ExpTextConfig::from_value(&base).is_ok());
    let yarn = with("rope_parameters", json!({"rope_type": "yarn", "factor": 4.0, "rope_theta": 1.0e7}));
    assert!(yarn.unwrap_err().to_string().contains("rope_type 'yarn'"));
    assert!(with("rope_scaling", json!({"type": "linear", "factor": 2.0})).is_err(), "구버전 rope_scaling.type");
    assert!(with("rope_parameters", json!({"rope_type": "default", "rope_theta": 1.0e7})).is_ok());
    let gelu = with("hidden_act", json!("gelu"));
    assert!(gelu.unwrap_err().to_string().contains("hidden_act 'gelu'"));
    // hidden_act 가 gelu 여도 output_gate_type 이 있으면 HF 검증은 통과하지만, 전문가/conv 는 gelu 라 포트는 거부해야 합니다.
    let mut gated = base.clone();
    gated["hidden_act"] = json!("gelu");
    gated["output_gate_type"] = json!("sigmoid");
    assert!(Qwen4ExpTextConfig::from_value(&gated).is_err());
    assert!(with("hidden_act", json!("swish")).is_ok());
    Ok(())
}

/// tie_word_embeddings=false 인데 lm_head.weight 가 없는 체크포인트는 embed_tokens 로 대신하지 않고 거부합니다.
#[test]
fn qwen4_exp_missing_lm_head_is_rejected() -> R {
    let td = TempDir::new("qwen4_exp_nolmhead");
    let dir = td.path();
    std::fs::copy(fixture("").join("config.json"), dir.join("config.json"))?;
    let mut ts = candle_core::safetensors::load(fixture("").join("model.safetensors"), &Device::Cpu)?;
    assert!(ts.remove("lm_head.weight").is_some());
    candle_core::safetensors::save(&ts, dir.join("model.safetensors"))?;
    let err = match Qwen4ExpForCausalLM::load_with(dir, DType::F32, &Device::Cpu, &dense_opts()) {
        Ok(_) => panic!("lm_head.weight 없는 체크포인트가 로드되었습니다"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("lm_head.weight"), "{err}");

    // 묶인(tied) 설정이면 embed_tokens 를 LM head 로 씁니다.
    let mut cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("").join("config.json"))?)?;
    cfg["tie_word_embeddings"] = json!(true);
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg)?)?;
    let mut m = Qwen4ExpForCausalLM::load_with(dir, DType::F32, &Device::Cpu, &dense_opts())?;
    let ids = Tensor::from_slice(&[5u32, 9, 11], (1, 3), &Device::Cpu)?;
    let logits = m.forward(&ids, 0)?;
    assert_eq!(logits.dims(), &[1, 3, 512]);
    Ok(())
}

#[test]
fn qwen4_exp_tiny_cpu_matches_hf() -> R {
    let dev = Device::Cpu;
    for v in ["", "variant_b"] {
        let r = run_variant(v, &dev, &dense_opts())?;
        assert_report(if v.is_empty() { "variant_a" } else { v }, &r);
    }
    Ok(())
}

/// PLE n-gram 표를 mmap 행 수집 경로로 강제해도 (전체 크기 모델이 타는 경로) HF 기준과 같아야 합니다.
#[test]
fn qwen4_exp_tiny_cpu_mmap_ple_matches_hf() -> R {
    let dev = Device::Cpu;
    for v in ["", "variant_b"] {
        let r = run_variant(v, &dev, &mmap_opts())?;
        assert_report(if v.is_empty() { "variant_a" } else { v }, &r);
    }
    Ok(())
}

/// GPU 공통(CUDA · ROCm · Vulkan · Metal): 이 빌드의 GPU 0번(`utils::new_gpu_device`)에서 CPU 와 같은 기준.
/// 장치를 열 수 없으면 건너뜁니다(`LOGIS_REQUIRE_GPU=1` 이면 실패 — CI 에서 건너뜀을 통과로 세지 않게).
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan", feature = "metal"))]
#[test]
fn qwen4_exp_tiny_gpu_matches_hf() -> R {
    let Some(dev) = tauri_app_lib::utils::new_gpu_device(0) else {
        assert!(
            std::env::var("LOGIS_REQUIRE_GPU").map(|v| v != "1").unwrap_or(true),
            "LOGIS_REQUIRE_GPU=1 인데 GPU 장치를 열 수 없습니다"
        );
        println!("[QWEN4-EXP] GPU 장치를 열 수 없어 건너뜁니다.");
        return Ok(());
    };
    for opts in [dense_opts(), mmap_opts()] {
        for v in ["", "variant_b"] {
            let r = run_variant(v, &dev, &opts)?;
            assert_report(if v.is_empty() { "variant_a" } else { v }, &r);
        }
    }
    // BF16 matmul 은 CUDA/ROCm 만 기준 수치가 있습니다(Metal/Vulkan BF16 은 측정만).
    match bf16_top1(&dev) {
        Ok(agree) if dev.is_cuda() || dev.is_rocm() => assert!(agree >= 0.75, "GPU BF16 top-1 일치율 {agree}"),
        Ok(_) => {}
        Err(e) if !(dev.is_cuda() || dev.is_rocm()) => println!("[QWEN4-EXP] BF16 on {:?} 미지원: {e}", dev.location()),
        Err(e) => return Err(e),
    }
    Ok(())
}

/// BF16 경로(HF 캐스팅 순서 재현) 는 F32 기준 대비 top-1 일치율만 봅니다 (무작위 tiny 모델이라 logits 간격이 좁음).
/// 참고: HF 자체를 bf16 으로 돌려도 F32 대비 top-1 93.8 %, max |Δ| 1.83 입니다. CPU 백엔드는 BF16 matmul 미지원이라 GPU 만.
#[cfg_attr(not(any(feature = "cuda", feature = "rocm", feature = "vulkan", feature = "metal")), allow(dead_code))]
fn bf16_top1(device: &Device) -> anyhow::Result<f32> {
    let dir = fixture("");
    let rf = load_ref(&dir)?;
    let mut model = Qwen4ExpForCausalLM::load(&dir, DType::BF16, device)?;
    let n = rf.ids.len();
    let ids = Tensor::from_slice(rf.ids.as_slice(), (1, n), device)?;
    let logits = model.forward(&ids, 0)?.to_dtype(DType::F32)?;
    let v = f32s(&logits)?;
    assert!(v.iter().all(|x| x.is_finite()), "BF16 logits 에 NaN/inf");
    let agree = argmax_agree(&logits, &rf.full)?;
    println!(
        "[QWEN4-EXP] variant_a BF16 on {:?}: top-1 일치 {:.1}% · max |Δ| {:.3e}",
        device.location(),
        agree * 100.0,
        max_abs_diff(&logits, &rf.full)?
    );
    Ok(agree)
}

/// 픽스처를 tokenizer.json + 채팅 템플릿과 함께 임시 폴더에 펼쳐 앱 등록 경로(TextGenerator::load_dir)로 읽습니다.
#[test]
fn qwen4_exp_registered_as_text_generator() -> R {
    let td = TempDir::new("qwen4_exp_gen");
    let dir = td.path();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(fixture("").join(f), dir.join(f))?;
    }
    // 512 개 단어 WordLevel 토크나이저 (id 2 = "<|im_end|>" = config eos)
    let mut vocab = serde_json::Map::new();
    for i in 0..512u32 {
        let tok = if i == 2 { "<|im_end|>".to_string() } else { format!("t{i}") };
        vocab.insert(tok, json!(i));
    }
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tokenizer)?)?;
    std::fs::write(dir.join("chat_template.jinja"), "{% for m in messages %}{{ m.content }} {% endfor %}")?;

    assert_eq!(detect_text_arch(dir), TextArch::Qwen4Exp);
    assert_eq!(detect_text_arch(&fixture("variant_b")), TextArch::Qwen4Exp, "VLM 래퍼 config");
    let q35 = TempDir::new("qwen4_exp_q35cfg");
    std::fs::write(q35.path().join("config.json"), r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}"#)?;
    assert_eq!(detect_text_arch(q35.path()), TextArch::Qwen35);
    // Qwen3.5 MoE 는 밀집 Qwen3.5 그래프/변환기로 읽을 수 없으므로 미지원으로 판별합니다.
    let moe = TempDir::new("qwen4_exp_q35moe");
    std::fs::write(
        moe.path().join("config.json"),
        r#"{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForConditionalGeneration"],"text_config":{"model_type":"qwen3_5_moe_text"}}"#,
    )?;
    assert_eq!(detect_text_arch(moe.path()), TextArch::Unknown("qwen3_5_moe".to_string()));
    let err = match TextGenerator::load_dir(moe.path(), &Device::Cpu, None) {
        Ok(_) => panic!("qwen3_5_moe 가 로드되었습니다"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("qwen3_5_moe"), "{err}");
    // qwen35 GGUF 변환 경로는 Qwen4-Exp 체크포인트를 거부해야 합니다.
    let err = tauri_app_lib::model::lang_gguf::prepare(dir, &|_, _| {}).unwrap_err().to_string();
    assert!(err.contains("Qwen4-Exp"), "{err}");

    // 앱의 실제 로드 경로: lib.rs run() 이 lang_llm::register_loader 로 등록하는 로더 (lifecycle 의
    // ensure_qwen3_5_lang 이 이 로더로 읽어 qwen4_exp_generator 슬롯에 둡니다). 타입이 LangLlmLoader 와 맞아야 합니다.
    let app_loader: tauri_app_lib::model::lang_llm::LangLlmLoader =
        tauri_app_lib::models::qwen4_exp::registry::load_lang_runtime;
    let via_app = app_loader(dir, &Device::Cpu)?;
    assert_eq!(via_app.arch(), TextArch::Qwen4Exp, "lang_llm 로더가 Qwen4-Exp 를 Qwen3.5 GGUF 경로로 보냈습니다");
    drop(via_app);

    let mut gen = TextGenerator::load_dir(dir, &Device::Cpu, Some(DType::F32))?;
    assert_eq!(gen.arch(), TextArch::Qwen4Exp);
    assert_eq!(gen.eos_token_id(), 2);
    let mes: tauri_app_lib::openai_types::ChatCompletionParameters = serde_json::from_value(json!({
        "model": "qwen4-exp-tiny",
        "messages": [{"role": "user", "content": "t180 t496 t430 t186 t121 t267 t265 t338 t449 t77 t5 t9"}],
        "max_tokens": 6, "temperature": 0.0
    }))?;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let text = rt.block_on(gen.generate(mes, None, None, None, None, None))?;
    let n = text.split_whitespace().count();
    println!("[QWEN4-EXP] TextGenerator(qwen4_exp) 생성 결과: {text:?}");
    assert!(n <= 6 && text.split_whitespace().all(|w| w.starts_with('t')), "{text:?}");

    // generate_ids(탐욕, 반복 패널티 1.1) 의 첫 토큰은 모델 argmax 와 같아야 합니다.
    if let TextGenerator::Qwen4Exp(g) = &mut gen {
        let prompt = [180u32, 496, 430, 186, 121];
        let ids = Tensor::from_slice(&prompt, (1, 5), &Device::Cpu)?;
        g.model.clear_cache();
        let want = g.model.forward_last(&ids, 0)?.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
        let mut sampler = candle_transformers::generation::LogitsProcessor::new(0, None, None);
        let out = g.generate_ids(&prompt, 1, &mut sampler, None)?;
        if want != 2 {
            assert_eq!(out, vec![want]);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
//  [MAC-VERIFY] 융합 GDN(CPU) vs 텐서 경로(CPU · GPU) — 수치와 시간 (docs/mac-verify.md 3-5)
// ---------------------------------------------------------------------
fn det_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Qwen3.5 급 선형 어텐션 형상(Hk 8 · Hv 16 · Dk=Dv 128 · conv 4)에서 프리필 T=256 + 디코드 16 스텝(상태 이어 받기)을
/// CPU 융합 커널 · CPU 텐서 경로 · GPU 텐서 경로(Metal/Vulkan 은 융합 커널이 없음)로 돌려 출력·상태와 시간을 비교합니다.
/// Metal/Vulkan 전용 GDN 커널이 필요한지의 근거입니다(GPU 텐서 경로가 CPU 융합보다 느리면 필요).
/// 전역 `set_gdn_fused_override` 를 바꾸고 시간을 재므로 병렬 suite 에서는 빠지고 단독으로 돌립니다
/// (`cargo test --test it -- gdn_fused --ignored --nocapture --test-threads 1`).
#[test]
#[ignore]
fn gdn_fused_cpu_vs_tensor_paths_numbers_and_time() -> R {
    use std::time::Instant;
    use tauri_app_lib::models::common::set_gdn_fused_override;
    use tauri_app_lib::models::qwen4_exp::gdn::{gdn_forward, GdnConfig, GdnHeadOrder, GdnOutput};
    let (hk, hv, dk, dv, k, t, steps) = (8usize, 16usize, 128usize, 128usize, 4usize, 256usize, 16usize);
    let c = 2 * hk * dk + hv * dv;
    let cfg = GdnConfig {
        num_k_heads: hk,
        num_v_heads: hv,
        head_k_dim: dk,
        head_v_dim: dv,
        conv_kernel: k,
        use_qk_l2norm: true,
        l2norm_eps: 1e-6,
        v_head_order: GdnHeadOrder::Grouped,
    };
    let host = |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &Device::Cpu);
    let n_tok = t + steps;
    let qkv = host(det_f32(n_tok * c, 1, 1.0), &[1, n_tok, c])?;
    let a = host(det_f32(n_tok * hv, 2, 1.0), &[1, n_tok, hv])?;
    let b = host(det_f32(n_tok * hv, 3, 1.0), &[1, n_tok, hv])?;
    let neg_a = host(det_f32(hv, 4, 0.5).into_iter().map(|x| -(x.abs() + 0.1)).collect(), &[hv])?;
    let dt_bias = host(det_f32(hv, 5, 0.5), &[hv])?;
    let conv_w = host(det_f32(c * k, 6, 0.5), &[c, k])?;
    let conv_b = host(det_f32(c, 7, 0.1), &[c])?;

    // (프리필 출력, 디코드 출력들, 최종 recurrent_state, 프리필 ms, 디코드 스텝당 ms)
    let run = |dev: &Device, fused: Option<bool>| -> anyhow::Result<(Tensor, Vec<Tensor>, Tensor, f64, f64)> {
        set_gdn_fused_override(fused);
        let to = |x: &Tensor| x.to_device(dev);
        let (qkv, a, b, neg_a, dt_bias, conv_w, conv_b) = (to(&qkv)?, to(&a)?, to(&b)?, to(&neg_a)?, to(&dt_bias)?, to(&conv_w)?, to(&conv_b)?);
        let mut best_pre = f64::MAX;
        let mut pre: Option<GdnOutput> = None;
        for _ in 0..2 {
            let t0 = Instant::now();
            let o = gdn_forward(&qkv.narrow(1, 0, t)?, &conv_w, Some(&conv_b), None, &a.narrow(1, 0, t)?, &b.narrow(1, 0, t)?, &neg_a, &dt_bias, None, &cfg)?;
            let _ = o.out.to_device(&Device::Cpu)?;
            best_pre = best_pre.min(t0.elapsed().as_secs_f64() * 1e3);
            pre = Some(o);
        }
        let pre = pre.unwrap();
        let (mut conv_state, mut state) = (pre.conv_state.clone(), pre.recurrent_state.clone());
        let mut outs = Vec::new();
        let t0 = Instant::now();
        for s in 0..steps {
            let p = t + s;
            let o = gdn_forward(&qkv.narrow(1, p, 1)?, &conv_w, Some(&conv_b), Some(&conv_state), &a.narrow(1, p, 1)?, &b.narrow(1, p, 1)?, &neg_a, &dt_bias, Some(&state), &cfg)?;
            outs.push(o.out.to_device(&Device::Cpu)?);
            conv_state = o.conv_state;
            state = o.recurrent_state;
        }
        let per_step = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        set_gdn_fused_override(None);
        Ok((pre.out.to_device(&Device::Cpu)?, outs, state.to_device(&Device::Cpu)?, best_pre, per_step))
    };
    let rel = |x: &Tensor, y: &Tensor| -> anyhow::Result<f32> {
        let d = (x.to_dtype(DType::F32)? - y.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let m = y.to_dtype(DType::F32)?.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        Ok(d / m)
    };
    let cmp = |x: &(Tensor, Vec<Tensor>, Tensor, f64, f64), y: &(Tensor, Vec<Tensor>, Tensor, f64, f64)| -> anyhow::Result<(f32, f32, f32)> {
        let mut dec = 0f32;
        for (p, q) in x.1.iter().zip(&y.1) {
            dec = dec.max(rel(p, q)?);
        }
        Ok((rel(&x.0, &y.0)?, dec, rel(&x.2, &y.2)?))
    };

    let cpu_fused_supported = candle_core::delta_rule::fused_gdn_supported(&Device::Cpu);
    let cpu_fused = run(&Device::Cpu, Some(true))?;
    let cpu_tensor = run(&Device::Cpu, Some(false))?;
    let (p, d, s) = cmp(&cpu_tensor, &cpu_fused)?;
    println!(
        "[GDN-PATH] CPU 융합({}) 프리필 T={t} {:.1}ms · 디코드 {:.2}ms/스텝 | CPU 텐서 경로 {:.1}ms · {:.2}ms/스텝 | 텐서↔융합 상대오차: 프리필 {:.1e} · 디코드 {:.1e} · 상태 {:.1e}",
        if cpu_fused_supported { "커널" } else { "미지원→텐서" },
        cpu_fused.3, cpu_fused.4, cpu_tensor.3, cpu_tensor.4, p, d, s
    );
    assert!(p <= 1e-3 && d <= 1e-3 && s <= 1e-3, "CPU 텐서 경로가 융합 커널과 다릅니다: {p:.2e} {d:.2e} {s:.2e}");

    if let Some(dev) = tauri_app_lib::utils::new_gpu_device(0) {
        // 이 장치의 기본 경로(융합 지원이면 융합: Metal 스캔 커널 · 공유 메모리 Vulkan 의 CPU 커널) vs 텐서 경로
        let gpu_fused = candle_core::delta_rule::fused_gdn_supported(&dev);
        let gpu_tensor = run(&dev, Some(false))?;
        let gpu = if gpu_fused { run(&dev, Some(true))? } else { gpu_tensor.clone() };
        for (label, r) in [("텐서 경로", &gpu_tensor), ("기본(융합)", &gpu)] {
            if label == "기본(융합)" && !gpu_fused {
                continue;
            }
            let (p, d, s) = cmp(r, &cpu_fused)?;
            println!(
                "[GDN-PATH] {:?} {label} 프리필 T={t} {:.1}ms · 디코드 {:.2}ms/스텝 | CPU 융합 대비 상대오차: 프리필 {:.1e} · 디코드 {:.1e} · 상태 {:.1e} | 속도: 프리필 CPU 융합의 {:.2}배 · 디코드 {:.2}배",
                dev.location(),
                r.3, r.4, p, d, s, cpu_fused.3 / r.3, cpu_fused.4 / r.4
            );
            assert!(p <= 1e-3 && d <= 1e-3 && s <= 1e-3, "{:?} {label}: CPU 융합과 다릅니다: {p:.2e} {d:.2e} {s:.2e}", dev.location());
        }
        println!(
            "[GDN-PATH] {:?} 융합 지원 {} → 기본 경로가 텐서 경로 대비 프리필 {:.2}배 · 디코드 {:.2}배",
            dev.location(),
            gpu_fused,
            gpu_tensor.3 / gpu.3,
            gpu_tensor.4 / gpu.4
        );
    }
    Ok(())
}
