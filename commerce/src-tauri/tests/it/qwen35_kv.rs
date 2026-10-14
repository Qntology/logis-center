//! Qwen3.5 KV 계획 v2 — 목업 Qwen3.5(시드 고정 무작위 가중치, 작은 GGUF)로 장치마다 (docs/mac-verify.md 3-5)
//!
//! 모델: hidden 64 · 4층 [GDN, 전체 어텐션, GDN, 전체 어텐션] (full_attention_interval 2) · 어텐션 4/2 heads × 64
//! (출력 게이트 포함) · GDN Hk 2 / Hv 4 · Dk = Dv 16 · conv 4 · vocab 512 · F32. Qwen3.5 는 VarBuilder 로 만들면
//! 층 가중치를 비운 뒤 GGUF 에서 다시 읽으므로, 메모리에서 GGUF 를 쓰고 파일로 mmap 해 앱과 같은 경로로 엽니다.
//!
//! - `qwen35_kv_tiers_and_states`: 프리필 1100 토큰(KV 블록 1024 + 76) 뒤 디코드 4 스텝을 KV 계층
//!   Vram / Ram 으로 강제해 로짓이 같은지, GDN 상태가 장치(VRAM)에 남는지, 원본 KV(kv_exact) 판정,
//!   블록이 실제로 RAM 으로 내려갔는지(`ram_kv_bytes`, `vram_kv_tokens`)를 확인합니다.
//! - FP8 강등 · VRAM 안 재압축 · F16 원본 KV(kv_half)는 코드상 CUDA/ROCm 전용이라 Mac/리눅스에서는
//!   해당 경로가 없습니다(테스트가 그 판정을 그대로 기록).
//! 장치 전체 상태(전역 accel 설정, 정지 신호 무시 플래그)를 건드리므로 `#[ignore]` + `--test-threads 1`.

use std::io::{Cursor, Write};
use std::sync::Arc;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use tauri_app_lib::models::common::gguf::Gguf;
use tauri_app_lib::models::qwen3_5::model::{Qwen3_5Model, IGNORE_STOP_SIGNAL_FOR_TESTS};
use tauri_app_lib::utils::resources::KvResidency;
use tauri_app_lib::utils::GpuDeviceExt;

use crate::common::{det, TempDir};

type R = anyhow::Result<()>;

const H: usize = 64;
const INTER: usize = 128;
const VOCAB: usize = 512;
const LAYERS: usize = 4;
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 64;
const HK: usize = 2;
const HV: usize = 4;
const DS: usize = 16; // GDN head_k_dim = head_v_dim
const CONV_K: usize = 4;
const PROMPT: usize = 1100;
const DECODE: usize = 4;

/// 시드 고정 목업 Qwen3.5 GGUF 바이트.
fn mock_qwen35_gguf() -> anyhow::Result<Vec<u8>> {
    let mut seed = 10u64;
    let mut t = |shape: &[usize], scale: f32, offset: f32| -> anyhow::Result<QTensor> {
        seed += 1;
        let n: usize = shape.iter().product();
        let v: Vec<f32> = det(n, seed).into_iter().map(|x| x * scale + offset).collect();
        Ok(QTensor::quantize(&Tensor::from_vec(v, shape, &Device::Cpu)?, GgmlDType::F32)?)
    };
    let lin = |i: usize| 1.0 / (i as f32).sqrt();
    let key_dim = HK * DS;
    let value_dim = HV * DS;
    let conv_dim = 2 * key_dim + value_dim;
    let mut tensors: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), t(&[VOCAB, H], 0.5, 0.0)?),
        ("output_norm.weight".into(), t(&[H], 0.05, 1.0)?),
    ];
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let mut add = |name: &str, q: QTensor| tensors.push((format!("{p}.{name}"), q));
        add("attn_norm.weight", t(&[H], 0.05, 1.0)?);
        add("post_attention_norm.weight", t(&[H], 0.05, 1.0)?);
        add("ffn_gate.weight", t(&[INTER, H], lin(H), 0.0)?);
        add("ffn_up.weight", t(&[INTER, H], lin(H), 0.0)?);
        add("ffn_down.weight", t(&[H, INTER], lin(INTER), 0.0)?);
        if (l + 1) % 2 == 0 {
            // 전체 어텐션: q_proj 는 헤드마다 [q (hd) | 출력 게이트 (hd)]
            add("attn_q.weight", t(&[NH * HD * 2, H], lin(H), 0.0)?);
            add("attn_k.weight", t(&[NKV * HD, H], lin(H), 0.0)?);
            add("attn_v.weight", t(&[NKV * HD, H], lin(H), 0.0)?);
            add("attn_output.weight", t(&[H, NH * HD], lin(NH * HD), 0.0)?);
            add("attn_q_norm.weight", t(&[HD], 0.05, 1.0)?);
            add("attn_k_norm.weight", t(&[HD], 0.05, 1.0)?);
        } else {
            // GDN (선형 어텐션): ssm_a = -exp(A_log) 이므로 음수
            add("attn_qkv.weight", t(&[conv_dim, H], lin(H), 0.0)?);
            add("attn_gate.weight", t(&[value_dim, H], lin(H), 0.0)?);
            add("ssm_beta.weight", t(&[HV, H], lin(H), 0.0)?);
            add("ssm_alpha.weight", t(&[HV, H], lin(H), 0.0)?);
            add("ssm_conv1d.weight", t(&[conv_dim, CONV_K], 0.3, 0.0)?);
            add("ssm_dt.bias", t(&[HV], 0.3, 0.0)?);
            add("ssm_a", t(&[HV], 0.1, -1.0)?);
            add("ssm_norm.weight", t(&[DS], 0.05, 1.0)?);
            add("ssm_out.weight", t(&[H, value_dim], lin(value_dim), 0.0)?);
        }
    }
    let u = |v: usize| Value::U32(v as u32);
    let meta: Vec<(&str, Value)> = vec![
        ("general.architecture", Value::String("qwen35".into())),
        ("general.dtype", Value::U32(0)), // F32
        ("qwen35.block_count", u(LAYERS)),
        ("qwen35.full_attention_interval", u(2)),
        ("qwen35.rope.freq_base", Value::F32(10_000_000.0)),
        ("qwen35.rope.dimension_count", u(16)),
        ("qwen35.rope.dimension_sections", Value::Array(vec![Value::I32(3), Value::I32(3), Value::I32(2), Value::I32(0)])),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("qwen35.embedding_length", u(H)),
        ("qwen35.feed_forward_length", u(INTER)),
        ("qwen35.attention.head_count", u(NH)),
        ("qwen35.attention.head_count_kv", u(NKV)),
        ("qwen35.attention.key_length", u(HD)),
        ("qwen35.attention.value_length", u(HD)),
        ("qwen35.ssm.group_count", u(HK)),
        ("qwen35.ssm.time_step_rank", u(HV)),
        ("qwen35.ssm.conv_kernel", u(CONV_K)),
        ("qwen35.ssm.state_size", u(DS)),
        ("qwen35.ssm.inner_size", u(value_dim)),
    ];
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut out = Cursor::new(Vec::new());
    gguf_file::write(&mut out, &meta_refs, &tensor_refs)?;
    Ok(out.into_inner())
}

/// 목업 GGUF 를 파일로 쓰고 mmap 해 앱과 같은 경로(`new_from_gguf`)로 엽니다.
struct MockQwen35 {
    _dir: TempDir,
    mmap: Arc<memmap2::Mmap>,
}

impl MockQwen35 {
    fn new() -> anyhow::Result<Self> {
        let dir = TempDir::new("qwen35_kv");
        let path = dir.path().join("mock-qwen35.gguf");
        std::fs::File::create(&path)?.write_all(&mock_qwen35_gguf()?)?;
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? });
        Ok(Self { _dir: dir, mmap })
    }

    fn load(&self, dev: &Device) -> anyhow::Result<Qwen3_5Model> {
        let ct = gguf_file::Content::read(&mut Cursor::new(&self.mmap[..]))?;
        let ct2 = Arc::new(gguf_file::Content::read(&mut Cursor::new(&self.mmap[..]))?);
        let mut g = Gguf::new(ct, Cursor::new(&self.mmap[..]), dev.clone());
        Ok(Qwen3_5Model::new_from_gguf(&mut g, None::<&mut Gguf<std::fs::File>>, dev, Some(self.mmap.clone()), Some(ct2))?)
    }
}

fn tokens() -> Vec<u32> {
    det(PROMPT + DECODE, 35).into_iter().map(|x| ((x.abs() * 1e4) as u32) % VOCAB as u32).collect()
}

struct Run {
    logits: Vec<Vec<f32>>,
    kv_exact: bool,
    ram_kv_bytes: u64,
    vram_kv_tokens: usize,
    ssm_on_device: bool,
}

/// 프리필 PROMPT 토큰 → (계층 강제) → 디코드 DECODE 스텝. 디코드 로짓 DECODE 개.
fn run(rt: &tokio::runtime::Runtime, mock: &MockQwen35, dev: &Device, tier: KvResidency) -> anyhow::Result<Run> {
    let ids = tokens();
    let mut m = mock.load(dev)?;
    let fwd = |m: &mut Qwen3_5Model, toks: &[u32], off: usize| -> anyhow::Result<Vec<f32>> {
        let t = Tensor::new(toks, dev)?.unsqueeze(0)?;
        let o = rt.block_on(m.forward(&t, None, None, None, None, None, off, None, None))?;
        Ok(o.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
    };
    let _ = fwd(&mut m, &ids[..PROMPT], 0)?;
    let kv_exact = m.language_model.kv_exact;
    // 계층 강제: 첫 디코드 전에 계획을 고정합니다(자연 계획은 실제 여유 메모리에 따라 달라짐).
    m.language_model.kv_plan = Some(tier);
    for l in m.language_model.layers.iter_mut() {
        l.set_kv_residency(tier);
    }
    let mut logits = Vec::new();
    for s in 0..DECODE {
        logits.push(fwd(&mut m, &ids[PROMPT + s..PROMPT + s + 1], PROMPT + s)?);
    }
    let ssm_on_device = m
        .language_model
        .layers
        .iter()
        .filter_map(|l| l.get_ssm_states().1)
        .all(|t| t.device().same_device(dev));
    Ok(Run {
        logits,
        kv_exact,
        ram_kv_bytes: m.language_model.mem.ram_kv_bytes,
        vram_kv_tokens: m.language_model.vram_kv_tokens(),
        ssm_on_device,
    })
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0f32, f32::max)
}

#[test]
#[ignore]
fn qwen35_kv_tiers_and_states() -> R {
    IGNORE_STOP_SIGNAL_FOR_TESTS.store(true, std::sync::atomic::Ordering::SeqCst);
    let rt = tokio::runtime::Runtime::new()?;
    let mock = MockQwen35::new()?;
    let mut devices = vec![Device::Cpu];
    // get_gpu_device 는 DEVICE_CACHE 를 채웁니다(Metal 의 KV-PLAN 이 장치 여유 메모리를 읽는 경로). GPU 가 없으면 CPU.
    let gpu = tauri_app_lib::utils::get_gpu_device(0);
    if !gpu.is_cpu() {
        devices.push(gpu);
    }
    let cpu_vram = run(&rt, &mock, &Device::Cpu, KvResidency::Vram)?;
    assert!(cpu_vram.logits.iter().flatten().all(|x| x.is_finite()));
    for dev in &devices {
        let label = format!("{:?}", dev.location());
        let vram = run(&rt, &mock, dev, KvResidency::Vram)?;
        let ram = run(&rt, &mock, dev, KvResidency::Ram)?;
        let d_tier = max_abs(&vram.logits, &ram.logits);
        let d_cpu = max_abs(&vram.logits, &cpu_vram.logits);
        let fp8_path = dev.is_cuda_or_rocm();
        println!(
            "[QWEN35-KV] 목업 Qwen3.5 · {label} · 프리필 {PROMPT} + 디코드 {DECODE} | kv_exact(원본 KV) {} | Vram: VRAM KV 토큰 {} | Ram: VRAM KV 토큰 {} · RAM KV {:.2} MB | Ram↔Vram max|Δlogit| {:.2e} | CPU 대비 {:.2e} | GDN 상태 장치 유지 {} | FP8 강등 · VRAM 재압축 · kv_half 경로 {}",
            vram.kv_exact,
            vram.vram_kv_tokens,
            ram.vram_kv_tokens,
            ram.ram_kv_bytes as f64 / 1e6,
            d_tier,
            d_cpu,
            if dev.is_cpu() { "CPU(해당 없음)".to_string() } else { format!("{}", vram.ssm_on_device && ram.ssm_on_device) },
            if fp8_path { "있음 (CUDA/ROCm)" } else { "없음 (CUDA/ROCm 전용)" }
        );
        // 블록이 실제로 RAM 으로 내려가야 계층 강제를 검증한 것입니다 (마지막 블록은 남음).
        assert!(ram.ram_kv_bytes > 0 && ram.vram_kv_tokens < vram.vram_kv_tokens, "{label}: Ram 계층에서 KV 블록이 내려가지 않았습니다");
        // CPU · Metal · Vulkan 은 RAM 블록을 F32 그대로 둡니다(무손실) → 로짓이 같아야 합니다.
        if !fp8_path {
            assert!(d_tier <= 1e-5, "{label}: Ram 계층 로짓이 Vram 과 다릅니다 ({d_tier:.2e})");
        }
        if !dev.is_cpu() {
            assert!(vram.ssm_on_device && ram.ssm_on_device, "{label}: GDN 상태가 장치에 남지 않았습니다");
            assert!(d_cpu <= 1e-3 * cpu_vram.logits.iter().flatten().fold(1f32, |m, x| m.max(x.abs())), "{label}: CPU 대비 {d_cpu:.2e}");
        }
    }
    Ok(())
}

/// SSD 계층 · SSD RAM 예산: 세션이 있으면 프리필이 꽉 찬 KV 블록(1024 토큰)을 SSD 에 비동기로 쓰고 RAM 으로
/// 내립니다(`force_flush_all_active_blocks`). 디코드 계획을 Ssd 로 강제하고 RAM KV 예산을 0 으로 두면 RAM 블록이
/// SSD 로 내려가고(RAM 사본 해제), 어텐션은 SSD 파일에서 다시 읽습니다. 같은 세션 조건의 Ram 계층과 로짓 비교.
/// 세션 경로의 RAM 사본은 CPU · Metal · Vulkan 에서 BF16 을 거치고(CUDA/ROCm 은 FP8), SSD 파일은 원본
/// 정밀도라 둘은 비트 동일이 아니므로 상대 오차 · top-1 로 봅니다(그 크기를 그대로 기록).
#[test]
#[ignore]
fn qwen35_kv_ssd_tier_with_session() -> R {
    use tauri_app_lib::models::qwen::generate::{drain_global_io, init_bake_worker};
    IGNORE_STOP_SIGNAL_FOR_TESTS.store(true, std::sync::atomic::Ordering::SeqCst);
    init_bake_worker();
    let rt = tokio::runtime::Runtime::new()?;
    let mock = MockQwen35::new()?;
    let gpu = tauri_app_lib::utils::get_gpu_device(0);
    let dev = if gpu.is_cpu() { Device::Cpu } else { gpu };
    let label = format!("{:?}", dev.location());
    let kv_dir = tauri_app_lib::utils::paths::get_kv_dir(None);
    let ids = tokens();

    let session_run = |tier: KvResidency, sid: &str| -> anyhow::Result<(Vec<Vec<f32>>, u64, bool)> {
        let mut m = mock.load(&dev)?;
        let sid_s = Some(sid.to_string());
        let fwd = |m: &mut Qwen3_5Model, toks: &[u32], off: usize| -> anyhow::Result<Vec<f32>> {
            let t = Tensor::new(toks, &dev)?.unsqueeze(0)?;
            let o = rt.block_on(m.forward(&t, None, None, None, None, None, off, sid_s.clone(), None))?;
            Ok(o.flatten_all()?.to_vec1::<f32>()?)
        };
        let _ = fwd(&mut m, &ids[..PROMPT], 0)?;
        // 꽉 찬 블록 0 의 SSD 파일(전체 어텐션 층 1 · 3)이 생길 때까지 기다립니다 (쓰기는 비동기, rename 으로 원자적).
        let block_dir = kv_dir.join(format!("{sid}/inference/text/b0"));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let files_ready = || (1..LAYERS).step_by(2).all(|l| block_dir.join(format!("l{l}.st")).exists());
        while !files_ready() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        rt.block_on(drain_global_io());
        let dumped = files_ready();
        m.language_model.kv_plan = Some(tier);
        for l in m.language_model.layers.iter_mut() {
            l.set_kv_residency(tier);
        }
        let start_bytes = if tier == KvResidency::Ssd {
            // SSD RAM 예산 0: RAM 에 있는 블록은 모두 SSD 로 (u64::MAX/2 에서 줄어든 만큼이 SSD 로 보낸 바이트)
            m.language_model.mem.ram_kv_budget = 0;
            m.language_model.mem.ram_kv_bytes = u64::MAX / 2;
            u64::MAX / 2
        } else {
            m.language_model.mem.ram_kv_bytes
        };
        let mut logits = Vec::new();
        for s in 0..DECODE {
            logits.push(fwd(&mut m, &ids[PROMPT + s..PROMPT + s + 1], PROMPT + s)?);
        }
        let moved = start_bytes.saturating_sub(m.language_model.mem.ram_kv_bytes);
        Ok((logits, moved, dumped))
    };
    let pid = std::process::id();
    let (ram, _, dumped_ram) = session_run(KvResidency::Ram, &format!("it-qwen35-ram-{pid}"))?;
    let (ssd, to_ssd, dumped_ssd) = session_run(KvResidency::Ssd, &format!("it-qwen35-ssd-{pid}"))?;
    for sid in [format!("it-qwen35-ram-{pid}"), format!("it-qwen35-ssd-{pid}")] {
        let _ = std::fs::remove_dir_all(kv_dir.join(sid));
    }
    let scale = ram.iter().flatten().fold(1e-6f32, |m, x| m.max(x.abs()));
    let rel = max_abs(&ram, &ssd) / scale;
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let top1 = ram.iter().zip(&ssd).filter(|(a, b)| argmax(a) == argmax(b)).count();
    println!(
        "[QWEN35-KV] SSD 계층 · {label} · 세션 프리필 {PROMPT} + 디코드 {DECODE} | 블록 0 SSD 기록 {} / {} | RAM 예산 0 → SSD 로 내린 KV {:.2} MB | Ssd↔Ram 상대 max|Δlogit| {:.2e} · top-1 {}/{}",
        dumped_ram, dumped_ssd, to_ssd as f64 / 1e6, rel, top1, DECODE
    );
    assert!(dumped_ram && dumped_ssd, "{label}: 프리필의 SSD 기록이 60초 안에 끝나지 않았습니다");
    assert!(to_ssd > 0, "{label}: RAM 예산 0 인데 블록이 SSD 로 내려가지 않았습니다");
    assert!(ssd.iter().flatten().all(|x| x.is_finite()), "{label}: SSD 계층 로짓에 NaN/inf");
    assert!(rel <= 2e-2 && top1 == DECODE, "{label}: SSD 계층 로짓이 Ram 과 다릅니다 (상대 {rel:.2e}, top-1 {top1}/{DECODE})");
    Ok(())
}

/// 압축 KV 블록 (TurboQuant, turbo-quant 빌드의 CUDA/ROCm/Vulkan): `LOGIS_KV_PRECISION=compressed`(프로세스에서
/// 한 번 읽는 값이라 CI 가 이 테스트만 따로 그 값으로 실행)이면 원본 KV 대신 압축 블록을 씁니다. CPU(항상 원본)
/// 대비 디코드 로짓 차이 · top-1, 그리고 압축 블록이 Ram 계층으로 내려가도 같은 결과인지(압축 블록은 그대로 이동).
#[cfg(feature = "turbo-quant")]
#[test]
#[ignore]
fn qwen35_kv_compressed_blocks() -> R {
    if std::env::var("LOGIS_KV_PRECISION").map(|v| v != "compressed").unwrap_or(true) {
        println!("[QWEN35-KV] LOGIS_KV_PRECISION=compressed 로 실행할 때만 측정합니다 — 건너뜀");
        return Ok(());
    }
    IGNORE_STOP_SIGNAL_FOR_TESTS.store(true, std::sync::atomic::Ordering::SeqCst);
    let rt = tokio::runtime::Runtime::new()?;
    let mock = MockQwen35::new()?;
    let gpu = tauri_app_lib::utils::get_gpu_device(0);
    if gpu.is_cpu() {
        println!("[QWEN35-KV] GPU 장치가 없어 건너뜁니다");
        return Ok(());
    }
    let label = format!("{:?}", gpu.location());
    let tq = tauri_app_lib::models::common::turbo_kv_active(&gpu);
    let cpu = run(&rt, &mock, &Device::Cpu, KvResidency::Vram)?;
    let vram = run(&rt, &mock, &gpu, KvResidency::Vram)?;
    let ram = run(&rt, &mock, &gpu, KvResidency::Ram)?;
    let scale = cpu.logits.iter().flatten().fold(1e-6f32, |m, x| m.max(x.abs()));
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let top1 = |a: &[Vec<f32>]| a.iter().zip(&cpu.logits).filter(|(x, y)| argmax(x) == argmax(y)).count();
    println!(
        "[QWEN35-KV] 압축 KV 블록 · {label} · TurboQuant {} | kv_exact {} | CPU(원본) 대비 상대 max|Δlogit| Vram {:.2e} · Ram {:.2e} | top-1 {}/{} · {}/{} | Ram↔Vram {:.2e}",
        if tq { "활성" } else { "비활성(이 장치에는 압축 커널 없음)" },
        vram.kv_exact,
        max_abs(&vram.logits, &cpu.logits) / scale,
        max_abs(&ram.logits, &cpu.logits) / scale,
        top1(&vram.logits), DECODE, top1(&ram.logits), DECODE,
        max_abs(&vram.logits, &ram.logits)
    );
    assert!(vram.logits.iter().chain(&ram.logits).flatten().all(|x| x.is_finite()), "{label}: NaN/inf");
    if tq {
        assert!(!vram.kv_exact, "{label}: compressed 인데 원본 KV 로 판정했습니다");
        assert!(max_abs(&vram.logits, &ram.logits) <= 1e-5, "{label}: 압축 블록이 RAM 으로 내려가며 값이 바뀌었습니다");
        assert!(top1(&vram.logits) == DECODE, "{label}: 압축 KV top-1 {}/{DECODE}", top1(&vram.logits));
    }
    Ok(())
}
