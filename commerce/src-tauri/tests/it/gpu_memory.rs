//! GPU 메모리 보고 · 반환 · KV 계층 (docs/mac-verify.md 3-5 · 4, 장치 공통)
//!
//! - `gpu_memory_is_reported_and_returned`: 이 빌드의 GPU 0번이 (free, total) 을 보고하는지 — Metal 은 포크
//!   패치 0003(recommendedMaxWorkingSetSize − currentAllocatedSize). 256 MB 를 할당하면 free 가 줄고, 놓은 뒤
//!   `settle_gpu_memory`(앱의 퍼지 경로) 후 돌아오는지. 그 값으로 KV-PLAN(`decide_kv_tier`) 이 무엇을 고르는지도 기록합니다
//!   (통합 메모리 Mac 에서는 VRAM 과 RAM 이 같은 물리 메모리라, 판정의 입력이 실제 장치 값인지가 중요합니다).
//! - `mock_qwen3_kv_tiers_match_cpu_on_gpu`: 목업 Qwen3 의 KV 를 VRAM 상주 / RAM 대피로 강제해도 CPU 와 같은 로짓.
//! - `model_unload_returns_device_memory`: 목업 Qwen3(약 60 MB) 를 GPU 에 올려 돌리고 내린 뒤 장치 free · 프로세스 RSS.
//! GPU 가 없는 빌드/러너에서는 건너뜁니다(CI 는 측정 줄이 없으면 NOT RUN 으로 기록).
//! 장치 전체의 free 를 재므로 다른 테스트와 병렬로 돌면 값이 흔들립니다 — 그래서 ignore 로 두고 단독 실행합니다
//! (`cargo test --test it -- gpu_memory --ignored --nocapture --test-threads 1`).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tauri_app_lib::models::qwen3::model::Qwen3Model;
use tauri_app_lib::utils::device_utils::{device_mem_info_of, settle_gpu_memory};
use tauri_app_lib::utils::resources::{decide_kv_tier, kv_ram_margin_bytes, kv_vram_margin_bytes, KvResidency};

use crate::common::mock_qwen3;

type R = anyhow::Result<()>;
const MB: f64 = 1e6;

fn gpu() -> Option<Device> {
    let d = tauri_app_lib::utils::new_gpu_device(0);
    if d.is_none() {
        println!("[GPU-MEM] GPU 장치가 없어 건너뜁니다.");
    }
    d
}

/// CPU 형 Vulkan 장치(lavapipe, `CANDLE_VULKAN_ALLOW_CPU=1` 일 때만 열림)는 '장치 메모리' 가 곧 시스템 RAM 이라
/// free 가 다른 프로세스 · 할당기 보유분에 따라 움직입니다 — 반환 단언은 실제 GPU(Metal · MoltenVK · 외장)에서만 합니다.
fn host_memory_device(dev: &Device) -> bool {
    dev.is_vulkan() && std::env::var("CANDLE_VULKAN_ALLOW_CPU").map(|v| v == "1").unwrap_or(false)
}

fn rss_mb() -> f64 {
    use sysinfo::{Pid, ProcessRefreshKind, System};
    let mut s = System::new();
    let pid = Pid::from_u32(std::process::id());
    s.refresh_process_specifics(pid, ProcessRefreshKind::new().with_memory());
    s.process(pid).map(|p| p.memory() as f64 / MB).unwrap_or(f64::NAN)
}

fn free_total(dev: &Device) -> anyhow::Result<(u64, u64)> {
    device_mem_info_of(dev).ok_or_else(|| anyhow::anyhow!("{:?}: device_mem_info_of 가 None 입니다", dev.location()))
}

#[test]
#[ignore]
fn gpu_memory_is_reported_and_returned() -> R {
    let Some(dev) = gpu() else { return Ok(()) };
    let (free0, total) = free_total(&dev)?;
    assert!(total > 256 << 20 && free0 <= total, "{:?}: free {free0} / total {total}", dev.location());
    let n = 64 << 20; // 64 Mi f32 = 256 MiB
    let t = Tensor::zeros(n, DType::F32, &dev)?;
    let _ = t.sum_all()?.to_scalar::<f32>()?;
    let (free1, _) = free_total(&dev)?;
    drop(t);
    settle_gpu_memory(&dev, 0, true, "gpu_memory test")?;
    let (free2, _) = free_total(&dev)?;
    let used = free0 as f64 - free1 as f64;
    let back = free2 as f64 - free1 as f64;
    // KV-PLAN 판정 입력: 장치 free 와 시스템 RAM free (통합 메모리에서는 같은 물리 메모리)
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram_free = sys.available_memory();
    let vm = kv_vram_margin_bytes(64 << 20);
    let rm = kv_ram_margin_bytes(sys.total_memory());
    let tiers: Vec<String> = [48u64, 512, 2048, 8192]
        .iter()
        .map(|mb| format!("{mb}MB→{:?}", decide_kv_tier(mb * 1_000_000, Some(free2), ram_free, vm, rm, None)))
        .collect();
    println!(
        "[GPU-MEM] {:?}: total {:.0} MB · free {:.0} → 256MiB 할당 후 {:.0} (−{:.0}) → 해제·settle 후 {:.0} (+{:.0}) | RAM free {:.0} MB / total {:.0} MB | KV-PLAN: {}",
        dev.location(), total as f64 / MB, free0 as f64 / MB, free1 as f64 / MB, used / MB, free2 as f64 / MB, back / MB,
        ram_free as f64 / MB, sys.total_memory() as f64 / MB, tiers.join(" · ")
    );
    // 할당이 보이지 않으면 KV-PLAN 이 실제 사용량을 모르는 것입니다.
    assert!(used >= 128.0 * MB, "256 MiB 할당이 free 에 반영되지 않았습니다 (−{:.0} MB)", used / MB);
    if host_memory_device(&dev) {
        println!("[GPU-MEM] CPU 형 장치(시스템 RAM): 반환 판정 생략");
        return Ok(());
    }
    assert!(free2 as f64 >= free0 as f64 - 64.0 * MB, "해제 후 free 가 돌아오지 않았습니다 ({:.0} → {:.0} MB)", free0 as f64 / MB, free2 as f64 / MB);
    Ok(())
}

#[test]
#[ignore]
fn mock_qwen3_kv_tiers_match_cpu_on_gpu() -> R {
    let Some(dev) = gpu() else { return Ok(()) };
    // 계층(VRAM 상주 / RAM 대피)만 비교합니다: TurboQuant KV(손실 압축, turbo-quant 빌드의 CUDA/ROCm/Vulkan)는 끕니다.
    let saved = candle_core::accel::config();
    candle_core::accel::update_config(|c| c.turbo_quant.enabled = false);
    let res = kv_tiers(&dev);
    candle_core::accel::set_config(saved);
    res
}

fn kv_tiers(dev: &Device) -> R {
    let dev = dev.clone();
    let (cfg, w) = mock_qwen3(64, 4, 16, 2)?;
    let ids = |v: &[u32], d: &Device| Tensor::new(v, d).and_then(|t| t.unsqueeze(0));
    let prompt = [3u32, 17, 42, 99, 5, 200, 7];
    let steps = [11u32, 250, 64];
    let build = |d: &Device| Qwen3Model::new(&cfg, VarBuilder::from_tensors(w.clone(), DType::F32, d));
    let v = |t: Tensor| -> anyhow::Result<Vec<f32>> { Ok(t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?) };
    let mut cpu = build(&Device::Cpu)?;
    let mut want = vec![v(cpu.forward(Some(&ids(&prompt, &Device::Cpu)?), None, 0)?)?];
    for (i, t) in steps.iter().enumerate() {
        want.push(v(cpu.forward(Some(&ids(&[*t], &Device::Cpu)?), None, prompt.len() + i)?)?);
    }
    let mut report = Vec::new();
    for tier in [KvResidency::Vram, KvResidency::Ram] {
        let mut m = build(&dev)?;
        m.set_kv_residency(tier);
        let mut got = vec![v(m.forward(Some(&ids(&prompt, &dev)?), None, 0)?)?];
        for (i, t) in steps.iter().enumerate() {
            got.push(v(m.forward(Some(&ids(&[*t], &dev)?), None, prompt.len() + i)?)?);
        }
        let worst = got
            .iter()
            .zip(&want)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0)))
            .fold(0f32, f32::max);
        report.push(format!("{tier:?} max rel {worst:.1e}"));
        assert!(worst <= 1e-3, "{:?} {tier:?}: CPU 대비 {worst:.2e}", dev.location());
    }
    println!("[GPU-MEM] KV 계층 강제 (목업 Qwen3, 프리필 7 + 디코드 3, TurboQuant KV off) {:?}: {}", dev.location(), report.join(" · "));
    Ok(())
}

#[test]
#[ignore]
fn model_unload_returns_device_memory() -> R {
    let Some(dev) = gpu() else { return Ok(()) };
    settle_gpu_memory(&dev, 0, true, "unload test start")?;
    let (free0, _) = free_total(&dev)?;
    let rss0 = rss_mb();
    {
        let (cfg, w) = mock_qwen3(1024, 16, 64, 4)?; // 약 60 MB F32
        let mut m = Qwen3Model::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &dev))?;
        let ids = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &dev)?.unsqueeze(0)?;
        let _ = m.forward(Some(&ids), None, 0)?.to_device(&Device::Cpu)?;
        let (free_loaded, _) = free_total(&dev)?;
        println!(
            "[GPU-MEM] 목업 Qwen3 적재: 장치 free {:.0} → {:.0} MB (−{:.0}) · RSS {:.0} → {:.0} MB",
            free0 as f64 / MB, free_loaded as f64 / MB, (free0 as f64 - free_loaded as f64) / MB, rss0, rss_mb()
        );
    }
    settle_gpu_memory(&dev, 0, true, "unload test end")?;
    let (free1, _) = free_total(&dev)?;
    let rss1 = rss_mb();
    println!(
        "[GPU-MEM] 내린 뒤(settle): 장치 free {:.0} MB (시작 대비 {:+.0} MB) · RSS {:.0} MB (시작 대비 {:+.0} MB)",
        free1 as f64 / MB, (free1 as f64 - free0 as f64) / MB, rss1, rss1 - rss0
    );
    if host_memory_device(&dev) {
        println!("[GPU-MEM] CPU 형 장치(시스템 RAM): 반환 판정 생략");
        return Ok(());
    }
    assert!(free1 as f64 >= free0 as f64 - 32.0 * MB, "모델을 내린 뒤 장치 메모리가 돌아오지 않았습니다 ({:.0} → {:.0} MB)", free0 as f64 / MB, free1 as f64 / MB);
    Ok(())
}
