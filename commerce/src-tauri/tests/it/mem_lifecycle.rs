//! 메모리 수명주기 · KV 계획 v2 (P2a)
//!
//! - CPU (항상)  : KV-PLAN 판정 수식(증분 가격 · 비례 여유 · 히스테리시스), 압축 KV 행 크기,
//!                 VRAM 정착 대기의 즉시 종료 판정, F16 원본 KV 보관의 오차 정량화.
//! - GPU (ignore): LOGIS_TEST_MODELS=1 이면 Qwen3-0.6B 를 공유 장치에 두 번 올렸다 내리고
//!                 실효 free VRAM 이 시작값의 64MB 안으로 돌아오는지, 지연 오류가 없는지 확인합니다.

use candle_core::{DType, Device, Tensor};
use tauri_app_lib::model::lifecycle::vram_settle_reached;
use tauri_app_lib::utils::resources::{
    decide_kv_tier, kv_bytes_per_token_compressed, kv_bytes_per_token_exact, kv_need_bytes, kv_ram_margin_bytes,
    kv_vram_margin_bytes, tq_fp8_row_bytes, KvResidency, KV_RAM_SAFETY_MARGIN_BYTES, KV_VRAM_SAFETY_MARGIN_BYTES,
};

const MB: u64 = 1_000_000;
const MIB: u64 = 1024 * 1024;

/// 사용자 로그의 4B 합성 호출: 48 MB KV, VRAM free 678 MB, RAM free 2124 MB, 총 RAM 7887 MB.
#[test]
fn kv_plan_48mb_with_678mb_free_is_vram() {
    let need = 48 * MB;
    let vram_free = Some(678 * MB);
    let ram_free = 2124 * MB;
    // 예전: 고정 여유 640MiB / 2GiB → SSD (로그와 같은 판정이 재현되는지)
    let old = decide_kv_tier(need, vram_free, ram_free, KV_VRAM_SAFETY_MARGIN_BYTES, KV_RAM_SAFETY_MARGIN_BYTES, None);
    assert_eq!(old, KvResidency::Ssd);
    // 새: VRAM 여유 = max(96MiB, 2 × 64MiB 디코딩 순간치) = 128MiB
    let vm = kv_vram_margin_bytes(64 * MIB);
    assert_eq!(vm, 128 * MIB);
    let rm = kv_ram_margin_bytes(7887 * MB);
    let new = decide_kv_tier(need, vram_free, ram_free, vm, rm, None);
    println!(
        "[KV-PLAN-TEST] 48MB KV · free 678MB | 예전 {:?} (여유 {}MB) → 새 {:?} (여유 {}MB)",
        old,
        KV_VRAM_SAFETY_MARGIN_BYTES / MB,
        new,
        vm / MB
    );
    assert_eq!(new, KvResidency::Vram);
}

#[test]
fn kv_plan_margins_are_proportional() {
    assert_eq!(kv_vram_margin_bytes(0), 96 * MIB);
    assert_eq!(kv_vram_margin_bytes(10 * MIB), 96 * MIB);
    assert_eq!(kv_vram_margin_bytes(320 * MIB), 640 * MIB);
    // 7.5GB PC → 15% ≈ 1183MB, 32GB PC → 2GiB 상한
    let small = kv_ram_margin_bytes(7887 * MB);
    assert!(small > 1180 * MB && small < 1185 * MB, "{small}");
    assert_eq!(kv_ram_margin_bytes(32 * 1024 * MIB), 2 * 1024 * MIB);
}

#[test]
fn kv_plan_prices_only_incremental_tokens() {
    // 4B: full_attention 8층 · KV 헤드 4 · d=256, 압축 행 = 264 B
    let per_token = kv_bytes_per_token_compressed(8, 4, 256);
    assert_eq!(per_token, 2 * 8 * 4 * 264);
    // 프롬프트 890 토큰은 이미 VRAM 에 있고, max_tokens 512 → 새로 생길 토큰 = 1 + 512
    let planned = 890 + 1 + 512;
    let need = kv_need_bytes(per_token, planned, 890);
    assert_eq!(need, per_token * 513);
    // 예전: 전체 토큰(890 + 1 + 2048) × FP8 1B/원소
    let old_need = kv_bytes_per_token_exact(8, 4, 256, 1) * (890 + 1 + 2048) as u64;
    println!(
        "[KV-PLAN-TEST] 증분 가격 {:.2} MB vs 예전 {:.2} MB ({:.1}배 과대)",
        need as f64 / 1e6,
        old_need as f64 / 1e6,
        old_need as f64 / need as f64
    );
    assert!(need * 5 < old_need);
    // 이미 상주한 토큰이 계획보다 많아도 음수가 되지 않습니다.
    assert_eq!(kv_need_bytes(per_token, 10, 20), 0);
}

#[test]
fn kv_plan_hysteresis_keeps_tier_between_calls() {
    let need = 100 * MB;
    let vm = 128 * MIB;
    let rm = 1183 * MB;
    let ram_free = 500 * MB; // RAM 계층은 불가
    // 딱 들어가는 VRAM free: need + margin
    let edge = Some(need + vm + 10 * MB);
    // 직전이 Vram 이면 그대로 Vram (내릴 때는 띠 없이 판정)
    assert_eq!(decide_kv_tier(need, edge, ram_free, vm, rm, Some(KvResidency::Vram)), KvResidency::Vram);
    // 직전이 Ssd 면 띠(max(64MB, 10%)) 를 더 요구하므로 올라가지 않습니다
    assert_eq!(decide_kv_tier(need, edge, ram_free, vm, rm, Some(KvResidency::Ssd)), KvResidency::Ssd);
    // 띠를 넘으면 올라갑니다
    let roomy = Some(need + vm + 64 * MIB + 1);
    assert_eq!(decide_kv_tier(need, roomy, ram_free, vm, rm, Some(KvResidency::Ssd)), KvResidency::Vram);
    // 직전 판정이 없으면 띠 없이 판정
    assert_eq!(decide_kv_tier(need, edge, ram_free, vm, rm, None), KvResidency::Vram);
    // 측정 실패(None)는 VRAM 을 고르지 않습니다
    assert_eq!(decide_kv_tier(need, None, ram_free, vm, rm, Some(KvResidency::Vram)), KvResidency::Ssd);
}

#[test]
fn turbo_quant_row_overhead_is_counted() {
    assert_eq!(tq_fp8_row_bytes(256), 264); // 264/256
    assert_eq!(tq_fp8_row_bytes(128), 136); // 34/32
    assert_eq!(kv_bytes_per_token_exact(8, 4, 256, 4), 65536); // 4B F32 원본 = 로그의 KV-PRECISION 값
    assert_eq!(kv_bytes_per_token_exact(8, 4, 256, 2), 32768); // F16 보관
}

#[test]
fn vram_settle_returns_immediately_when_target_met() {
    let target = 1200 * MIB;
    assert!(vram_settle_reached(Some(3330 * MB), target));
    assert!(vram_settle_reached(Some(target), target));
    assert!(!vram_settle_reached(Some(target - 1), target));
    // 측정 실패는 0.00 GB 로 바꾸지 않고 '미달' 로만 봅니다
    assert!(!vram_settle_reached(None, target));
}

/// F16 원본 KV 보관이 F32 대비 어텐션 출력에 주는 오차 (Qwen 류 큰 K 채널 흉내).
#[test]
fn half_kv_storage_error_is_small() -> anyhow::Result<()> {
    use tauri_app_lib::models::qwen3_5::model::half_kv_piece;
    let dev = Device::Cpu;
    let (h, t, d) = (4usize, 512usize, 256usize);
    let mut k = Tensor::randn(0f32, 1.0, (1, h, t, d), &dev)?;
    // 큰 채널 몇 개 (QK-norm 뒤에도 남는 큰 값)
    let mut boost = vec![1f32; d];
    for i in [3usize, 77, 130, 201] {
        boost[i] = 20.0;
    }
    k = k.broadcast_mul(&Tensor::from_vec(boost, (1, 1, 1, d), &dev)?)?;
    let v = Tensor::randn(0f32, 1.0, (1, h, t, d), &dev)?;
    let q = Tensor::randn(0f32, 1.0, (1, h, 1, d), &dev)?;
    let attn = |k: &Tensor, v: &Tensor| -> anyhow::Result<Tensor> {
        let s = (q.matmul(&k.transpose(2, 3)?)? / (d as f64).sqrt())?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        Ok(p.matmul(v)?)
    };
    let o32 = attn(&k, &v)?;
    let kh = half_kv_piece(&k, None)?;
    let vh = half_kv_piece(&v, None)?;
    assert_eq!(kh.dtype(), DType::F16);
    let o16 = attn(&kh.to_dtype(DType::F32)?, &vh.to_dtype(DType::F32)?)?;
    let rel = ((&o16 - &o32)?.sqr()?.sum_all()?.to_scalar::<f32>()? / o32.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
    println!("[KV-HALF-TEST] F16 보관 어텐션 출력 상대오차 {:.2e} (벡터별 FP8 은 문서상 4~6%)", rel);
    assert!(rel < 2e-3, "rel {rel}");
    // 넘치는 값은 BF16 으로 보관
    let big = (Tensor::ones((1, 1, 2, 4), DType::F32, &dev)? * 1.0e5)?;
    assert_eq!(half_kv_piece(&big, None)?.dtype(), DType::BF16);
    // F16 블록에 이어 붙이는 값은 범위로 잘려 inf 가 되지 않습니다
    let clipped = half_kv_piece(&big, Some(DType::F16))?.to_dtype(DType::F32)?;
    assert!(clipped.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  GPU: 공유 장치에서 Qwen3-0.6B 를 두 번 올렸다 내리기 — LOGIS_TEST_MODELS=1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU + 로컬 모델 필요: LOGIS_TEST_MODELS=1"]
fn gpu_qwen3_load_unload_returns_vram() {
    if std::env::var("LOGIS_TEST_MODELS").map(|v| v != "1").unwrap_or(true) {
        println!("LOGIS_TEST_MODELS=1 이 아니라 건너뜁니다.");
        return;
    }
    let dir = crate::common::models_dir().join("Qwen3-0.6B-Instruct-gguf");
    if !dir.exists() {
        return crate::common::skip_or_fail(&format!("모델 디렉터리 {} 가 없습니다", dir.display()));
    }
    use tauri_app_lib::utils::{device_mem_info_of, get_gpu_device, settle_gpu_memory};
    let dev = get_gpu_device(0);
    if dev.is_cpu() {
        println!("GPU 장치가 없어 건너뜁니다.");
        return;
    }
    // 공유 장치: 같은 번호로 다시 만들어도 같은 장치입니다.
    #[cfg(feature = "cuda")]
    {
        let again = Device::new_cuda(0).expect("cuda");
        assert!(again.same_device(&dev), "Device::new_cuda 는 번호당 하나의 공유 장치여야 합니다");
    }
    // ROCm 빌드: get_gpu_device 가 ROCm 장치를 골랐을 때만 같은 확인 (CUDA 빌드는 위 블록 그대로).
    #[cfg(feature = "rocm")]
    if dev.is_rocm() {
        let again = Device::new_rocm(0).expect("rocm");
        assert!(again.same_device(&dev), "Device::new_rocm 는 번호당 하나의 공유 장치여야 합니다");
    }
    settle_gpu_memory(&dev, 0, true, "test start").expect("settle");
    let (start, total) = device_mem_info_of(&dev).expect("mem_info");
    println!("[VRAM-TEST] 시작 실효 free {:.0} MB / 총 {:.0} MB", start as f64 / 1e6, total as f64 / 1e6);
    let path = dir.to_string_lossy().to_string();
    for round in 1..=2 {
        let t = std::time::Instant::now();
        let gen = tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf(&path, Some(&dev), Some(DType::BF16))
            .expect("load qwen3");
        let (loaded, _) = device_mem_info_of(&dev).expect("mem_info");
        drop(gen);
        settle_gpu_memory(&dev, 0, true, "test unload").expect("settle");
        let (after, _) = device_mem_info_of(&dev).expect("mem_info");
        println!(
            "[VRAM-TEST] {}회차 | 적재 후 {:.0} MB (점유 {:.0} MB, {:.1}s) → 해제 후 {:.0} MB (시작 대비 {:+.0} MB)",
            round,
            loaded as f64 / 1e6,
            (start as f64 - loaded as f64) / 1e6,
            t.elapsed().as_secs_f64(),
            after as f64 / 1e6,
            (after as f64 - start as f64) / 1e6
        );
        assert!(after + 64 * MIB >= start, "{}회차: 해제 후 {} < 시작 {} - 64MB", round, after, start);
    }
    assert_eq!(dev.take_deferred_error(), None, "지연 CUDA 오류가 남아 있습니다");
}

// =====================================================================
//  🌟 [ROCM-EXIT] Windows ROCm 종료 경로 — 자식 프로세스를 여러 번 띄워 멈춤이 없는지 확인
// ---------------------------------------------------------------------
//  자식: ROCm 장치에서 여러 모양의 GEMM(F32/BF16, rocBLAS)을 돌린 뒤 rocm_exit_now_if_used(0) 로 끝납니다.
//  부모: 자식을 20번 실행해 매번 60초 안에 종료 코드 0 으로 끝나고 [ROCM-EXIT] 줄을 찍는지 봅니다.
//  (예전 정상 종료 경로는 rocBLAS/HIP DLL 해제 중 간헐적으로 무한 대기했습니다)
// =====================================================================
#[cfg(all(windows, feature = "rocm"))]
#[test]
#[ignore = "GPU 필요: ROCm 종료 경로를 자식 프로세스로 20회 확인"]
fn gpu_rocm_fast_exit_never_hangs() {
    const CHILD_ENV: &str = "LOGIS_ROCM_EXIT_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        let dev = tauri_app_lib::utils::new_gpu_device(0).expect("ROCm 장치");
        assert!(dev.is_rocm(), "ROCm 장치가 아닙니다: {dev:?}");
        for (m, n, k) in [(1usize, 1024usize, 1024usize), (7, 3072, 1024), (64, 1024, 3072), (256, 151_936, 1024), (900, 2048, 2048)] {
            for dt in [DType::F32, DType::BF16] {
                let a = Tensor::ones((m, k), dt, &dev).unwrap();
                let b = Tensor::ones((k, n.min(8192)), dt, &dev).unwrap();
                let _ = a.matmul(&b).unwrap();
            }
        }
        dev.synchronize().unwrap();
        println!("[ROCM-EXIT-TEST] 자식: GEMM 완료, 종료 경로 진입");
        tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
        panic!("rocm_exit_now_if_used 가 돌아왔습니다 (프로세스가 끝나야 합니다)");
    }

    let exe = std::env::current_exe().unwrap();
    let runs: usize = std::env::var("LOGIS_ROCM_EXIT_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    for i in 0..runs {
        let started = std::time::Instant::now();
        let mut child = std::process::Command::new(&exe)
            .args(["--exact", "mem_lifecycle::gpu_rocm_fast_exit_never_hangs", "--ignored", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break s;
            }
            if started.elapsed() > std::time::Duration::from_secs(60) {
                let _ = child.kill();
                panic!("자식 {i}: 60초 안에 끝나지 않았습니다 (종료 멈춤)");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        };
        let out = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        println!("[ROCM-EXIT-TEST] 자식 {i}: {status} · {:.2}s", started.elapsed().as_secs_f64());
        assert!(status.success(), "자식 {i}: 종료 코드 {status}\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
        assert!(stdout.contains("[ROCM-EXIT]"), "자식 {i}: [ROCM-EXIT] 줄이 없습니다\n{stdout}");
        assert!(!stdout.contains("test result:"), "자식 {i}: 테스트 하네스가 정상 종료 경로로 끝났습니다\n{stdout}");
    }
}
