use super::{LogisModel, ModelSize};   // ← 부모의 타입을 끌어옵니다
use crate::utils;
use anyhow::anyhow;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::qwen3::generate::Qwen3GenerateModel;
use crate::models::embedding::EmbeddingModel;
use crate::openai_types::*;
use candle_core::{Device, DType};
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, atomic::AtomicBool};
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use tokio::sync::Mutex as TokioMutex;
use std::time::{Duration, Instant};

impl LogisModel {
    pub async fn unload_embedding(&self) {
        {
            let mut emb = self.embedding_model.lock().await;
            if emb.is_some() {
                *emb = None;
                println!("[MODEL] Embedding Model unloaded to free VRAM.");
            }
        }
        
        // 🌟 [추가] RAM 확보를 위해 임베딩 텍스트 캐시 완전 삭제
        {
            let mut cache = self.embedding_cache.lock().await;
            cache.clear();
            println!("[MODEL] Embedding Memory Cache cleared to free RAM.");
        }

        // 🌟 [추가] VRAM 해제를 위해 CUDA 디바이스 동기화 및 메모리 풀 초기화
        if !self.is_cpu_mode {
            // candle의 Device 자체에 내장된 안전한 동기화 메서드 사용 (컴파일 에러 해결)
            if self.device_config.device.is_cuda() {
                let _ = self.device_config.device.synchronize();
            }
            // 이전 CUDA 컨텍스트의 캐시 풀을 OS로 강제 반환시키기 위한 유도 장치
            let _ = candle_core::Device::new_cuda(self.device_config.gpu_id as usize);
            println!("[MODEL] CUDA Context synchronized and memory pool flushed.");
        }

        // 🌟 [CROSSOVER] 임베딩만 내렸으므로 IDLE 이 아니라 '남은 슬롯 기준' 으로 맞춥니다.
        //    생성 모델이 함께 상주 중이었다면 PHASE_GENERATION 이 되어야 하는데,
        //    mark_crossover_idle 로 뭉뚱그리면 다음 enter_generation_phase 가
        //    이미 올라와 있는 모델을 다시 올리려 합니다.
        self.sync_crossover_phase().await;
    }

    // =====================================================================
    // 🌟 [CROSSOVER] 생성 슬롯 전용 부분 반환
    // ---------------------------------------------------------------------
    //  ── 왜 deep_purge_resources 로 충분하지 않은가 ──
    //   deep_purge 는 '공장 초기화' 성격입니다. 임베딩·SigLIP2·KV 스토리지까지
    //   전부 파기하고 CUDA 동기화에 최대 10초를 씁니다.
    //   그런데 크로스오버가 필요로 하는 것은 '생성 모델이 쥔 VRAM' 뿐입니다.
    //   전체 퍼지를 쓰면 임베딩을 곧바로 다시 올려야 하므로
    //   스왑 1회가 실질적으로 2회분 비용이 됩니다.
    //
    //  ── 무엇을 건드리지 않는가 ──
    //   · embedding_model  : 반환 대상이 아닙니다. 그것이 이 함수의 존재 이유입니다.
    //   · siglip2_model    : 비전 파이프라인이 별도로 release_siglip2 로 관리합니다.
    //                        ensure_qwen3_5 의 SigLIP2 보호 분기와 충돌하지 않도록
    //                        여기서는 손대지 않습니다.
    //
    //  ── 동기화 상한 ──
    //   release_siglip2 와 같은 이유로 5초 상한을 둡니다. 동기화는 정확성이 아니라
    //   반환 시점에만 영향을 주므로, 드라이버 스톨 시 영구 대기하는 편이 더 위험합니다.
    // =====================================================================
    pub async fn unload_generation_slots(&self, reason: &str) {
        let _hold = self.hold_generation();

        let had = {
            self.generator.lock().await.is_some()
                || self.qwen3_generator.lock().await.is_some()
                || self.qwen3_5_generator.lock().await.is_some()
        };
        if !had {
            println!("[CROSSOVER] ⚡ [GEN UNLOAD SKIP] 반환할 생성 슬롯이 없습니다. ({})", reason);
            self.sync_crossover_phase().await;
            return;
        }

        println!("[CROSSOVER] 🔻 [GEN UNLOAD] 생성 슬롯만 반환합니다. ({})", reason);
        crate::models::qwen::generate::wait_for_global_io().await;

        {
            let mut gen = self.generator.lock().await;
            if let Some(mut g) = gen.take() {
                println!("[CROSSOVER] Dropping Qwen(0.6B) generator...");
                let _ = g.clear_kv_cache();
                let _ = g.qwen.drop_kv_storage();
                drop(g);
            }
        }
        {
            let mut q3 = self.qwen3_generator.lock().await;
            if let Some(mut g) = q3.take() {
                println!("[CROSSOVER] Dropping Qwen3 generator...");
                g.clear_kv_cache();
                drop(g);
            }
        }
        {
            let mut q35 = self.qwen3_5_generator.lock().await;
            if let Some(mut g) = q35.take() {
                println!("[CROSSOVER] Dropping Qwen3.5 generator...");
                g.clear_kv_cache();
                drop(g);
            }
        }
        {
            *self.current_size.lock().await = None;
        }

        if !self.is_cpu_mode {
            let dev = self.device_config.device.clone();
            let sync = tokio::time::timeout(
                Duration::from_secs(5),
                tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() {
                        let _ = dev.synchronize();
                    }
                }),
            ).await;
            match sync {
                Ok(Ok(())) => {},
                Ok(Err(e)) => println!("[CROSSOVER] CUDA sync join error: {:?}", e),
                Err(_) => println!("[CROSSOVER] ⚠️ CUDA sync 5s 상한 도달. 동기화 없이 진행합니다."),
            }
            // caching allocator 가 붙들고 있는 풀을 OS 로 밀어냅니다.
            let _ = candle_core::Device::new_cuda(self.device_config.gpu_id as usize);
        }

        self.sync_crossover_phase().await;
        println!("[CROSSOVER] ✅ [GEN UNLOAD] 완료. 자유 {}MB", self.get_free_vram_mb());
    }

    /// [CLEANUP] Aggressive Factory Reset Purge (Reinforced with Diagnostics)
    pub async fn deep_purge_resources(&self) {
        let _purge_hold = self.hold_generation();
        println!("[DIAG-PURGE] Step 0: Waiting for background IO to finish...");
        crate::models::qwen::generate::wait_for_global_io().await; // [cite: 254]
        println!("[DIAG-PURGE] Step 1: Clearing ALL Generation Slots...");
        
        {
            let mut gen = self.generator.lock().await;
            if let Some(mut g) = gen.take() {
                println!("[DIAG-PURGE] Dropping Active Generator (0.6B)...");
                let _ = g.clear_kv_cache();
                let _ = g.qwen.drop_kv_storage(); 
                drop(g); 
            }
        }
        
        // 🌟 [신규] Qwen3 (텍스트 전용) 슬롯 해제 추가
        {
            let mut q3_gen = self.qwen3_generator.lock().await;
            if let Some(mut g) = q3_gen.take() {
                println!("[DIAG-PURGE] Dropping Qwen3 Generator...");
                g.clear_kv_cache(); // Qwen3 구조체에 구현된 캐시 클리어 호출
                drop(g);
            }
        }

        {
            let mut q35_gen = self.qwen3_5_generator.lock().await;
            if let Some(mut g) = q35_gen.take() {
                println!("[DIAG-PURGE] Dropping Qwen 3.5 Generator..."); //
                g.clear_kv_cache();
                drop(g);
            }
        }
        
        println!("[DIAG-PURGE] Step 2: Clearing Embedding Model & Cache...");
        {
            let mut emb = self.embedding_model.lock().await;
            if let Some(e) = emb.take() {
                drop(e);
            }
            // 🌟 램 누수 방지를 위해 캐시도 깔끔하게 비워줍니다.
            let mut cache = self.embedding_cache.lock().await;
            cache.clear();
        }

        // 🌟 [SigLIP2] 비전 엔진도 함께 해제합니다.
        //    mmap 백엔드라 상주량이 LLM 보다 작지만,
        //    GPU 로 올라간 경우 VRAM 을 계속 점유하므로 반드시 내려야 합니다.
        {
            let mut vis_guard = self.siglip2_model.lock().await;
            if vis_guard.is_some() {
                *vis_guard = None;
                println!("[DIAG-PURGE] SigLIP2 vision engine dropped.");
            }
        }

        // 🌟 [CRITICAL FIX] 모델 사이즈 상태값도 완벽하게 초기화하여 Relay 시스템 꼬임 방지
        {
            let mut size = self.current_size.lock().await;
            *size = None;
        }

        println!("[DIAG-PURGE] Step 3: Synchronizing CUDA Context...");
        if !self.is_cpu_mode {
            let dev = self.device_config.device.clone();
            let sync_res = tokio::time::timeout(Duration::from_secs(10), tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { 
                    println!("[DIAG-PURGE] Executing dev.synchronize()...");
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        dev.synchronize()
                    }))
                } else { Ok(Ok(())) }
            })).await;
            
            match sync_res {
                Ok(Ok(Ok(Ok(_)))) => println!("[DIAG-PURGE] CUDA Synchronization Successful."),
                Ok(Ok(Ok(Err(e)))) => println!("[DIAG-PURGE] CUDA Sync Error: {:?}", e),
                Ok(Err(_)) => println!("[DIAG-PURGE] CUDA Sync Task Join Error."),
                Err(_) => println!("[DIAG-PURGE] CUDA Sync Timeout! Continuing purge."),
                _ => println!("[DIAG-PURGE] CUDA Sync Panicked or Failed."),
            }

            // 🌟 [추가] 기존에 할당된 CUDA 메모리 풀을 OS에 즉시 반환시키기 위한 컨텍스트 덮어쓰기
            let _ = candle_core::Device::new_cuda(self.device_config.gpu_id as usize);
        }

        println!("[DIAG-PURGE] Step 4: Flushing OS Memory...");
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
            let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        #[cfg(target_os = "linux")]
        unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
        #[cfg(target_os = "macos")]
        unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }

        // 🌟 [VISION-CACHE] 디스크 캐시는 모델 파기와 무관하게 유지됩니다.
        //    (ViT 는 결정론적이므로 모델 인스턴스가 바뀌어도 결과가 동일합니다)
        {
            let (hits, misses) = crate::models::vision_cache::VISION_CACHE.stats();
            if hits + misses > 0 {
                let rate = (hits as f64 / (hits + misses) as f64) * 100.0;
                println!("[VISION-CACHE] Session stats — hits: {} | misses: {} | hit rate: {:.1}%", hits, misses, rate);
            }
        }

        println!("[DIAG-PURGE] Aggressive Purge Complete.");
        // 🌟 [CROSSOVER] 퍼지 후 모든 슬롯이 비었으므로 원장을 사실에 맞춥니다.
        //    이 한 줄 덕분에 호출부마다 mark_crossover_idle 을 흩뿌릴 필요가 없습니다.
        //    (이미 호출부에 남아 있는 mark_crossover_idle 은 멱등이라 무해합니다)
        self.sync_crossover_phase().await;
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    // --- [NEW] VRAM Settlement Monitor (Smart Polling) ---
    pub async fn wait_for_vram_settle(&self, target_free_mb: u64, timeout_sec: u64, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        if self.is_cpu_mode { return Ok(()); }

        println!("[VRAM-WATCH] Monitoring VRAM (Target > {} MB)...", target_free_mb);
        let start = Instant::now();
        let target_bytes = target_free_mb * 1024 * 1024;
        let mut last_free = 0;
        let mut stable_ticks = 0;
        let mut increasing_ticks = 0;
        let mut has_flushed_ram = false;

        loop {
            // 1. Cancellation Check
            if let Some(token) = &cancel_token {
                if token.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(anyhow::anyhow!("Task cancelled during VRAM wait"));
                }
            }

            // 2. Measure VRAM
            let mut current_free = 0;
            use nvml_wrapper::Nvml;
            if let Ok(nvml) = Nvml::init() {
                if let Ok(dev) = nvml.device_by_index(self.device_config.gpu_id as u32) {
                    if let Ok(mem) = dev.memory_info() {
                        current_free = mem.free;
                    }
                }
            }

            // [FAST-PATH] Immediate Success
            if current_free >= target_bytes {
                if stable_ticks >= 2 { // Confirm stability for 1 sec
                    println!("[VRAM-WATCH] Success! VRAM Secured: {:.2} GB", current_free as f64 / 1e9);
                    break;
                }
                stable_ticks += 1;
            } else {
                stable_ticks = 0;
            }

            // [ADAPTIVE-LOGIC] Analyze Trend (20MB sensitivity)
            if current_free > last_free + (20 * 1024 * 1024) { 
                increasing_ticks += 1;
                println!("[VRAM-WATCH] Reclaiming... ({:.2} GB -> {:.2} GB)", last_free as f64/1e9, current_free as f64/1e9);
            } else {
                increasing_ticks = 0;
            }

            // [ACTIVE-FLUSH] If stuck for > 1.5s, trigger OS RAM cleanup
            if start.elapsed().as_secs_f32() > 1.5 && !has_flushed_ram && current_free < target_bytes {
                println!("[VRAM-WATCH] Triggering Aggressive OS Working Set Trim...");
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
                
                has_flushed_ram = true;
                tokio::time::sleep(Duration::from_millis(300)).await;
                continue;
            }

            // [TIMEOUT-HANDLER]
            if start.elapsed().as_secs() > timeout_sec {
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Timeout reached but memory is freeing up. Extending wait...");
                    increasing_ticks = 0;
                    continue; 
                }
                println!("[VRAM-WATCH] Timeout reached. Proceeding with {:.2} GB (Target: {:.2} GB)", current_free as f64/1e9, target_free_mb as f64/1024.0);
                break;
            }

            last_free = current_free;
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Ok(())
    }

    // --- [NEW] VRAM 자유 메모리 조회 헬퍼 ---
    /// 현재 디바이스의 자유 VRAM을 MB 단위로 반환합니다.
    /// CPU 모드면 항상 충분하다는 의미로 u64::MAX를 돌려줍니다.
    /// nvml 초기화 실패 시 보수적으로 0을 반환합니다.
    pub fn get_free_vram_mb(&self) -> u64 {
        if self.is_cpu_mode { return u64::MAX; }
        use nvml_wrapper::Nvml;
        if let Ok(nvml) = Nvml::init() {
            if let Ok(dev) = nvml.device_by_index(self.device_config.gpu_id as u32) {
                if let Ok(mem) = dev.memory_info() {
                    return mem.free / (1024 * 1024);
                }
            }
        }
        0
    }

    // --- [NEW] SSD Bridge Operations ---    
    pub async fn save_kv_snapshot(&self, task_id: &str, kv_name: Option<String>, offset: usize) -> anyhow::Result<String> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let path = crate::utils::paths::get_kv_dir(None).join(format!("{}.safetensors", task_id_str));
            if is_q35 {
                // Qwen3.5는 자체 flush 매커니즘을 사용하므로 패스만 반환
                Ok(path.to_string_lossy().to_string())
            } else {
                let mut gen_guard = generator_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    println!("[SSD-BRIDGE] Saving KV snapshot to {:?}", path);
                    gen.save_kv_to_disk(&path, kv_name.as_deref(), offset)?;
                    Ok(path.to_string_lossy().to_string())
                } else {
                    Err(anyhow::anyhow!("No active generator to save snapshot from"))
                }
            }
        }).await?
    }

    pub async fn truncate_kv_cache(&self, len: usize) -> anyhow::Result<()> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        let generator_arc = self.generator.clone();
        let q35_arc = self.qwen3_5_generator.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            if is_q35 {
                let mut gen_guard = q35_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.qwen3_5.language_model.truncate_kv_cache(len).map_err(|e| anyhow::anyhow!("Truncate failed: {}", e))
                } else {
                    Ok(())
                }
            } else {
                let mut gen_guard = generator_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.truncate_kv_cache(len).map_err(|e| anyhow::anyhow!("Truncate failed: {}", e))
                } else {
                    Ok(())
                }
            }
        }).await?
    }

    pub async fn load_kv_snapshot(&self, task_id: &str, kv_name: Option<String>) -> anyhow::Result<()> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        
        let generator_arc = self.generator.clone();
        let q35_arc = self.qwen3_5_generator.clone();
        let task_id_str = task_id.to_string();
        let kv_name_str = kv_name.unwrap_or_else(|| "text".to_string());

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let kv_root = crate::utils::paths::get_kv_dir(None).join(&task_id_str);
            let kv_type = kv_name_str.split('/').last().unwrap_or("text");
            let kv_type = if kv_type == "inference" || kv_type == "reference" || kv_type.is_empty() { "text" } else { kv_type };

            // 🌟 [핵심 픽스] 현재 모델이 Qwen 3.5(2B)라면 2B 방(q35_arc)에 스냅샷을 로드합니다!
            if is_q35 {
                let mut q35_guard = q35_arc.blocking_lock();
                if let Some(gen) = q35_guard.as_mut() {
                    let target_kv_name = format!("{}/inference/{}", task_id_str, kv_type);
                    let target_kv_name = if !crate::utils::paths::get_kv_dir(None).join(&target_kv_name).exists() {
                        format!("{}/reference/{}", task_id_str, kv_type)
                    } else { target_kv_name };
                    
                    println!("[SSD-BRIDGE] Restoring Qwen 3.5 Registry from {}", target_kv_name);
                    gen.qwen3_5.language_model.restore_kv_registry(&target_kv_name)?;
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("No active Qwen 3.5 generator to load snapshot into"))
                }
            } else {
                // 0.6B / 2B 로직은 그대로 유지
                let paths_to_try = vec![
                    kv_root.join("inference").join(kv_type),
                    kv_root.join("reference").join(kv_type),
                    kv_root.clone(),
                ];

                let mut target_path = None;
                for p in paths_to_try {
                    if p.exists() && std::fs::read_dir(&p).map(|mut d| d.next().is_some()).unwrap_or(false) {
                        target_path = Some(p);
                        break;
                    }
                }

                if let Some(p) = target_path {
                    let mut gen_guard = generator_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[SSD-BRIDGE] Loading Directory-based KV snapshot from {:?}", p);
                        gen.load_kv_from_disk(&p, None)?;
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("No active generator to load snapshot into"))
                    }
                } else {
                    println!("[SSD-BRIDGE] No snapshot found for {} (Checked deep paths)", task_id_str);
                    Ok(())
                }
            }
        }).await?
    }

    // --- File: src/model.rs ---

    pub async fn secure_vram_relay(&self, target_size: ModelSize, task_id: Option<&str>, cancel_token: Option<Arc<AtomicBool>>, is_baking: bool, kv_name: Option<String>) -> anyhow::Result<()> {
        let start_time = Instant::now();

        // 🌟 [추가] 현재 로드된 모델이 목표와 같다면 로딩 과정을 건너뛰고 즉시 반환하여 VRAM 낭비 및 지연 방지
        {
            let current = self.current_size.lock().await;
            if *current == Some(target_size) {
                let is_loaded = match target_size {
                    ModelSize::Qwen => {
                        // 🌟 [VISION-JIT] secure_vram_relay(Qwen) 의 호출자는 Base PUG 베이킹 /
                        //    타이틀 추출 / ingest_pug_to_ssd 세 곳뿐이며 전부 순수 텍스트 경로입니다.
                        //    기존에는 is_baking=false 인 타이틀 추출에서도 mmproj 가 통째로 상주했습니다.
                        let mut gen_guard = self.generator.lock().await;
                        if let Some(gen) = gen_guard.as_mut() {
                            if gen.is_vision_jit_capable() && gen.vision_resident() {
                                let _ = gen.set_vision_active(false);
                                println!("[RELAY] Text-only path detected. Detached Qwen(0.6B) vision weights to free VRAM.");
                            }

                            let is_baking_loaded = match &gen.qwen {
                                crate::models::qwen::generate::ModelVariant::QuantizedVL(m) => m.language_model.baking_only,
                                crate::models::qwen::generate::ModelVariant::QuantizedText(m) => m.language_model.baking_only,
                                _ => false,
                            };
                            // 🌟 [CRITICAL FIX] 베이킹 모드(LM Head 제거)로 떠있는데, 정상 추론이 필요하면 건너뛰지 않고 리로드!
                            if !is_baking && is_baking_loaded {
                                false
                            } else {
                                true
                            }
                        } else {
                            false
                        }
                    },
                    ModelSize::Qwen3 => self.qwen3_generator.lock().await.is_some(),
                    ModelSize::Qwen3_5 => {
                        // 🌟 [VISION-JIT] secure_vram_relay 로 들어오는 Qwen3_5 요청은
                        //    thead 구조 추출 / status selector 추출 등 전부 순수 텍스트 경로입니다.
                        //    기존에는 여기서 그냥 Skipping 으로 빠져나가면서
                        //    직전 이미지 추출이 올려둔 mmproj 600MB 가 프리필 내내 VRAM 을 점유했습니다.
                        let mut guard = self.qwen3_5_generator.lock().await;
                        if let Some(gen) = guard.as_mut() {
                            if gen.vision_capable() && gen.is_vision_jit_capable() && gen.vision_resident() {
                                let _ = gen.set_vision_active(false);
                                println!("[RELAY] Text-only path detected. Detached Qwen 3.5 vision weights to free VRAM.");
                            }
                            true
                        } else {
                            false
                        }
                    },
                };
                if is_loaded {
                    println!("[RELAY] {:?} is already loaded. Skipping purge/reload.", target_size);
                    // 🌟 [CROSSOVER] fast-path 는 아무것도 내리지 않습니다.
                    //    임베딩이 함께 상주 중일 수 있으므로 실제 슬롯을 읽어 원장을 맞춥니다.
                    //    (current_size 가드를 먼저 놓아야 뒤이은 조회가 막히지 않습니다)
                    drop(current);
                    self.sync_crossover_phase().await;
                    return Ok(());
                } else {
                    // 🌟 [CRITICAL FIX] Background에서 로딩 중이라 객체(is_some)는 없지만, current_size는 target_size로 등록된 상태!
                    // 여기서 무식하게 deep_purge를 호출하면 백그라운드 로딩이 파괴되므로, 각 ensure 함수의 대기(Wait) 루프로 안전하게 진입시킵니다.
                    drop(current); // 락 해제
                    println!("[RELAY] {:?} is currently loading in background. Waiting for synchronization...", target_size);
                    match target_size {
                        ModelSize::Qwen => { self.ensure_generator_ext(ModelSize::Qwen, false, is_baking).await?; },
                        ModelSize::Qwen3 => { self.ensure_qwen3().await?; },
                        ModelSize::Qwen3_5 => { self.ensure_qwen3_5(false).await?; },
                    }
                    return Ok(());
                }
            }
        }
        
        println!("[RELAY] Performing Deep Purge before loading {:?} (Baking: {})...", target_size, is_baking);
        // 🌟 [GENERATION HOLD] 퍼지 시작부터 로드 완료까지를 하나의 전환 구간으로 묶습니다.
        //
        //  ── 왜 deep_purge 안이 아니라 여기인가 ──
        //   deep_purge 안에만 걸면 퍼지가 끝난 직후 wait_for_vram_settle(최대 5초)
        //   구간이 무방비로 열립니다. 실측 로그가 정확히 그 창에서
        //   "[VRAM-WATCH] Reclaiming... (0.00 GB -> 4.04 GB)" 를 반복한 이유입니다.
        //   4GB 를 확보하고도 로드 시점엔 756MB 만 남아 KV 가 RAM 으로 밀려났습니다.
        //
        //  ── 해제 시점 ──
        //   _relay_hold 는 이 함수가 반환될 때(정상/에러/조기반환 모두) Drop 됩니다.
        //   즉 로드가 끝나는 순간 임베딩이 다시 올라올 수 있고,
        //   STAGE-3 의 [VECTORIZING] 경로는 종전과 동일하게 동작합니다.
        let _relay_hold = self.hold_generation();
        
        // 🌟 [CRITICAL FIX] size 상태값도 초기화하여 락 꼬임 방지
        {
            *self.current_size.lock().await = None;
        }
        self.deep_purge_resources().await;
        
        if !self.is_cpu_mode {
            tokio::time::sleep(Duration::from_millis(500)).await;
            self.wait_for_vram_settle(1200, 5, cancel_token.clone()).await?;
        }

        match target_size {
            ModelSize::Qwen => {
                self.ensure_generator_ext(ModelSize::Qwen, false, is_baking).await?;

                // 🌟 [VISION-JIT] 신규 로드 직후에도 비전을 떼어냅니다.
                //    secure_vram_relay(Qwen) 는 전부 텍스트 전용 경로이며,
                //    실제 이미지가 들어오면 QuantizedQwenVLModel::forward 가 mmap 에서 자동 복원합니다.
                {
                    let mut gen_guard = self.generator.lock().await;
                    if let Some(gen) = gen_guard.as_mut() {
                        if gen.is_vision_jit_capable() && gen.vision_resident() {
                            let _ = gen.set_vision_active(false);
                            println!("[RELAY] Freshly loaded Qwen(0.6B): vision weights detached for text-only workload.");
                        }
                    }
                }

                if let Some(tid) = task_id {
                    self.load_kv_snapshot(tid, kv_name).await?;
                }
            },
            ModelSize::Qwen3 => {
                self.ensure_qwen3().await?;
            },
            ModelSize::Qwen3_5 => {
                self.ensure_qwen3_5(false).await?;
            },
        }

        println!("[RELAY] Transition to {:?} complete in {:.2}s", target_size, start_time.elapsed().as_secs_f32());
        // 🌟 [CROSSOVER] 이 경로는 deep_purge 를 거쳤으므로 임베딩이 파기된 상태입니다.
        //    그 사실을 추측하지 않고 슬롯에서 직접 읽어 원장에 반영합니다.
        self.sync_crossover_phase().await;
        Ok(())
    }

    // --- [NEW] Base Context Baking (One-time Heavy Lifting) ---
    pub async fn ingest_pug_to_ssd(&self, task_id: &str, pug_content: &str, cancel_token: Option<Arc<AtomicBool>>, kv_name: Option<String>) -> anyhow::Result<()> {
        let base_session = format!("{}_base", task_id);
        
        // 1. Load Small Model Isolated (Full layers, no baking)
        self.secure_vram_relay(ModelSize::Qwen, None, cancel_token.clone(), false, None).await?; // 🌟 Small -> Qwen

        // 2. Ingest PUG content
        {
            let prompt = format!("{}\n\n[SYSTEM] Analyze the document structure.", pug_content);
            let mut gen_guard = self.generator.lock().await;
            if let Some(gen) = gen_guard.as_mut() {
                // Just prefill, no generation needed for base context
                gen.prefill_chunk(prompt, cancel_token.clone(), None).await?;
            }
        }

        // 3. Save Base Snapshot
        self.save_kv_snapshot(&base_session, kv_name, 0).await?;
        
        // [FIX] 베이킹 직후 모델을 파괴하지 않고 그대로 유지하여 컨텍스트 오류를 방지합니다.
        // self.unload_generator().await; 제거됨
        
        Ok(())
    }

    // --- File: src/model.rs (LogisModel 내부) ---

    pub async fn ensure_qwen3(&self) -> anyhow::Result<()> {
        let needs_load = { self.qwen3_generator.lock().await.is_none() };
        if needs_load {
            // 🌟 [CRITICAL FIX] 백그라운드에서 이미 로딩이 시작되었는지 확인하고, 진행 중이라면 완료될 때까지 안전하게 대기합니다.
            {
                let size_guard = self.current_size.lock().await;
                if *size_guard == Some(ModelSize::Qwen3) {
                    drop(size_guard);
                    println!("[MODEL] Qwen3 is currently loading in background. Waiting for synchronization...");
                    loop {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        if self.qwen3_generator.lock().await.is_some() {
                            return Ok(());
                        }
                    }
                }
            }

            println!("[MODEL] Loading Qwen3 Text Model (0.6B GGUF) exclusively via NATIVE /qwen3/ logic...");
            // 🌟 [GENERATION HOLD] 이 함수는 secure_vram_relay 를 거치지 않고
            //    단독 호출되는 경로도 있으므로 자체적으로 전환 구간을 선언합니다.
            //    (holds 는 카운터이므로 relay 경유 시 2중으로 잡혀도 안전합니다)
            let _load_hold = self.hold_generation();
            
            // 🌟 [DOUBLE PURGE ELIMINATED] 퍼지를 '무조건' 이 아니라 '필요할 때만' 수행합니다.
            //
            //  ── 실측 사고 (log.txt) ──
            //   [RELAY] Performing Deep Purge before loading Qwen3 ...
            //   [DIAG-PURGE] ... Aggressive Purge Complete.     ← secure_vram_relay 가 이미 수행
            //   [VRAM-WATCH] Success! VRAM Secured: 4.04 GB
            //   [MODEL] Loading Qwen3 Text Model ...
            //   [DIAG-PURGE] ... Aggressive Purge Complete.     ← 여기서 또 수행 (완전 중복)
            //   퍼지 1회는 CUDA 동기화에만 최대 10초를 쓰며, 그 창이 곧
            //   백그라운드 임베딩 재로드가 끼어드는 창이었습니다.
            //   실측 [RELAY] Transition to Qwen3 complete in 47.40s 의 대부분이 이 낭비입니다.
            //
            //  ── 판정 근거 ──
            //   '방금 퍼지했는가' 를 시간으로 재면 매직 상수가 됩니다.
            //   대신 '내릴 것이 실제로 남아 있는가' 라는 상태를 봅니다.
            //   내릴 것이 하나도 없으면 퍼지는 정의상 무의미한 연산입니다.
            // 🌟 [CROSSOVER] 임베딩을 '퍼지 트리거' 에서 분리합니다.
            //
            //  ── 무엇이 문제였나 ──
            //   기존 purge_needed 는 embedding_model.is_some() 을 포함했습니다.
            //   그래서 임베딩이 상주하기만 하면 예산과 무관하게 전체 퍼지가 돌고,
            //   deep_purge Step 2 가 그 임베딩을 함께 파기했습니다.
            //   크로스오버의 '동시 상주' 경로가 이 한 줄 때문에
            //   구조적으로 도달 불가능했습니다.
            //
            //  ── 판정 근거 ──
            //   임베딩은 이미 VRAM 을 점유한 상태이므로, 지금 측정한 자유 메모리는
            //   '임베딩을 남긴 채 쓸 수 있는 양' 그 자체입니다.
            //   그 값이 Qwen3 예산을 넘으면 임베딩을 내릴 이유가 없습니다.
            //   예산은 디스크 가중치 크기에서 출발해 실측으로 교체되므로 상수가 아닙니다.
            //
            //  ── 무엇을 보수적으로 남겼는가 ──
            //   다른 생성 슬롯이나 SigLIP2 가 살아 있으면 기존과 동일하게 전체 퍼지합니다.
            //   그 경우는 어차피 무언가를 반드시 내려야 하고, 부분 반환으로 얻는 이득보다
            //   기존 동작을 보존하는 쪽이 안전합니다.
            let embed_resident = { self.embedding_model.lock().await.is_some() };
            let other_resident = {
                self.generator.lock().await.is_some()
                    || self.qwen3_5_generator.lock().await.is_some()
            };
            let vision_resident = { self.siglip2_model.lock().await.is_some() };
            let keep_embedding = embed_resident
                && !other_resident
                && !vision_resident
                && self.embedding_coexist_ok(ModelSize::Qwen3);

            if other_resident || vision_resident || (embed_resident && !keep_embedding) {
                // 🌟 [CRITICAL FIX] unload_generator가 소유권을 훔쳐가 KV 캐시 클리어를 방해하는 버그 해결!
                // 바로 deep_purge_resources만 단독 호출하여 VRAM을 100% 안전하게 날려줍니다.
                self.deep_purge_resources().await;
            } else if keep_embedding {
                println!(
                    "[MODEL] 🤝 [CROSSOVER/COEXIST] 자유 {}MB >= Qwen3 예산 {}MB. 임베딩을 상주시킨 채 로드합니다. (퍼지 생략)",
                    self.get_free_vram_mb(),
                    self.generation_budget_mb(ModelSize::Qwen3)
                );
            } else {
                println!("[MODEL] ⚡ [PURGE SKIP] 해제 대상 슬롯이 하나도 없어 중복 퍼지를 생략합니다.");
            }
            
            {
                *self.current_size.lock().await = Some(ModelSize::Qwen3);
            }
            
            let path = self.qwen3_model_path.clone();
            let dev = self.device_config.device.clone();
            let dtype = if self.is_cpu_mode { Some(candle_core::DType::F32) } else { Some(candle_core::DType::BF16) };
            
            // 🌟 방금 만든 init_from_gguf 를 호출합니다!
            let gen_result = tokio::task::spawn_blocking(move || -> anyhow::Result<Qwen3GenerateModel> {
                Qwen3GenerateModel::init_from_gguf(&path, Some(&dev), dtype)
            }).await?;

            match gen_result {
                Ok(gen) => {
                    println!("[MODEL] 🎉 Qwen3 (0.6B GGUF) Native Model loaded successfully!");
                    *self.qwen3_generator.lock().await = Some(gen);
                },
                Err(e) => {
                    println!("\n==================================================");
                    println!("🚨 [CRITICAL ERROR] 0.6B GGUF 로딩 실패!");
                    println!("원인: {:?}", e);
                    println!("==================================================\n");
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        self.ensure_generator_ext(size, false, false).await
    }

    pub async fn ensure_generator_ext(&self, size: ModelSize, force_text_only: bool, baking_only: bool) -> anyhow::Result<()> {
        if size == ModelSize::Qwen3_5 {
            return self.ensure_qwen3_5(false).await; 
        }
        if size == ModelSize::Qwen3 {
            return self.ensure_qwen3().await; 
        }

        // 오직 ModelSize::Qwen 만 이 아래 로직을 탐
        let mut current_size_guard = self.current_size.lock().await; // 🌟 첫 번째 자물쇠 획득!
        let mut gen_guard = self.generator.lock().await;

        if *current_size_guard == Some(size) {
            if let Some(gen) = gen_guard.as_ref() {
                let is_baking_loaded = match &gen.qwen {
                    crate::models::qwen::generate::ModelVariant::QuantizedVL(m) => m.language_model.baking_only,
                    crate::models::qwen::generate::ModelVariant::QuantizedText(m) => m.language_model.baking_only,
                    _ => false,
                };
                // 🌟 [CRITICAL FIX] 현재 모델이 Baking(LM Head 부재) 상태인데 추론 요청이 오면 Fresh Loading 진행
                if !baking_only && is_baking_loaded {
                    // 통과하여 아래 리로드 로직(Fresh Loading) 실행
                } else {
                    return Ok(());
                }
            }
        }

        println!("[LOAD] Fresh loading {:?} from disk...", size);
        let path = &self.qwen_model_path; 
        
        // 🌟 [CRITICAL FIX] 이중 자물쇠(Deadlock) 유발 코드 제거! 
        // 이미 가지고 있는 current_size_guard 에 직접 값을 할당합니다.
        *current_size_guard = Some(size); 
        
        let target_device = self.device_config.device.clone();
        let is_disk_swap = self.is_disk_swap;
        let dev_id = self.device_config.gpu_id;
        let dtype = if target_device.is_cpu() { Some(candle_core::DType::F32) } else { Some(candle_core::DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let handle_clone = self.app_handle.clone();

        let gen = match tokio::time::timeout(
            std::time::Duration::from_secs(60), 
            tokio::task::spawn_blocking(move || {
                let kv_root = crate::utils::paths::get_kv_dir(Some(&handle_clone));
                QwenVLGenerateModel::init_with_config(
                    &path_clone, None, None,
                    Some(&target_device), dev_id, Some(&target_device), dev_id, dtype, Some(limit as usize),
                    force_text_only, baking_only, is_disk_swap, kv_root
                )
            })
        ).await {
            Ok(Ok(Ok(generator))) => generator,
            Ok(Ok(Err(e))) => {
                println!("🚨 [MODEL-ERROR] 모델 초기화 실패 (로직 에러): {:?}", e);
                return Err(e);
            },
            Ok(Err(e)) => {
                println!("🚨 [MODEL-ERROR] Spawn Blocking 실패 (스레드 에러): {:?}", e);
                return Err(e.into());
            },
            Err(_) => {
                println!("🚨 [CRITICAL] 60초 타임아웃 발생! 모델 로딩 내부에서 무한 대기에 빠졌습니다!");
                return Err(anyhow::anyhow!("Model Loading Timeout"));
            }
        };

        *gen_guard = Some(gen);
        // *current_size_guard = Some(size); // 위에서 미리 등록했으므로 생략 가능
        
        Ok(())
    }

    pub async fn call_qwen3_verification_model(&self, prompt: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        // Qwen3 로드 보장
        self.ensure_qwen3().await?;
        
        let gen_arc = self.qwen3_generator.clone();
        let cancel_clone = cancel_token.clone();
        let prompt_string = prompt.to_string();
        
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = gen_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![
                        crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
                            content: "You are a precise evaluation assistant. Return strictly the requested JSON format.".to_string(),
                            name: None,
                        }),
                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt_string),
                            name: None,
                        })
                    ],
                    model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.1), top_p: Some(0.95),
                    ..Default::default()
                };
                gen.generate(params, cancel_clone, None, None).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
            } else {
                Err(anyhow::anyhow!("Qwen3 Generator is missing"))
            }
        }).await??;
        
        Ok(res)
    }

    // 🌟 [SYNONYM EXPANSION] 음차(Transliteration) 전용 Qwen3 호출.
    //    call_qwen3_verification_model 은 시스템 프롬프트가 "JSON 으로 답하라"라서
    //    음차 결과가 JSON 껍데기에 갇힙니다. 여기서는 단어별 JSON만 받도록 분리합니다.
    //
    //    시스템 프롬프트에도 언어 이름이 전혀 없습니다.
    //    목표 표기 체계는 user 프롬프트의 [TARGET LANGUAGE] 로만 전달되며,
    //    그 값은 lang_code_to_full_name() 으로 런타임에 생성됩니다.
    //
    //    temperature 0.0 : 같은 값이면 항상 같은 별칭이 나와야 재인덱싱 시 벡터가 흔들리지 않습니다.
    pub async fn call_qwen3_transliteration(&self, prompt: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        self.ensure_qwen3().await?;
        let gen_arc = self.qwen3_generator.clone();
        let cancel_clone = cancel_token.clone();
        let prompt_string = prompt.to_string();
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = gen_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![
                        crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
                            content: "You respell each word of the source text into the target writing system by sound only. You never translate meaning. You process every word independently. Return strictly the requested JSON format.".to_string(),
                            name: None,
                        }),
                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage {
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt_string),
                            name: None,
                        })
                    ],
                    model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };
                gen.generate(params, cancel_clone, None, None).map_err(|e| anyhow::anyhow!("Qwen3 transliteration failed: {}", e))
            } else {
                Err(anyhow::anyhow!("Qwen3 Generator is missing"))
            }
        }).await??;

        // 호출마다 KV 캐시를 비워 이전 값의 음차가 다음 값에 새는 것을 차단합니다.
        let q3_clear_arc = self.qwen3_generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                gen.clear_kv_cache();
            }
        }).await;

        Ok(res)
    }

    // 🌟 [SYNONYM EXPANSION - QWEN3.5] 음차(Transliteration) 전용 Qwen3.5 2B 호출.
    //    0.6B 모델은 음차 능력이 부족하여 원문을 그대로 반복하는 문제가 있습니다.
    //    Qwen3.5 2B 모델은 별도의 VRAM 슬롯(qwen3_5_generator)을 사용하며,
    //    호출 전 Qwen3 0.6B를 내리고 Qwen3.5를 올리는 분리 동작을 전제로 합니다.
    //
    //    시스템 프롬프트에도 언어 이름이 전혀 없습니다.
    //    목표 표기 체계는 user 프롬프트의 [TARGET LANGUAGE] 로만 전달되며,
    //    그 값은 lang_code_to_full_name() 으로 런타임에 생성됩니다.
    //
    //    temperature 0.0 : 같은 값이면 항상 같은 별칭이 나와야 재인덱싱 시 벡터가 흔들리지 않습니다.
    //    max_tokens 256 : 단어별 JSON 구조는 기존 단일 문자열보다 토큰 수가 많으므로 상향합니다.
    pub async fn call_qwen3_5_transliteration(&self, prompt: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        self.ensure_qwen3_5(false).await?;

        let mut gen_guard = self.qwen3_5_generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow::anyhow!("Qwen3.5 Generator is missing"))?;

        let params = crate::openai_types::ChatCompletionParameters {
            messages: vec![
                crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
                    content: "You respell each word of the source text into the target writing system by sound only. You never translate meaning. You process every word independently. Return strictly the requested JSON format.".to_string(),
                    name: None,
                }),
                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage {
                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                    name: None,
                })
            ],
            model: "qwen3.5".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
            ..Default::default()
        };

        let res = gen.generate(params, cancel_token.clone(), None, None, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("Qwen3.5 transliteration failed: {}", e))?;

        // 호출마다 KV 캐시를 비워 이전 값의 음차가 다음 값에 새는 것을 차단합니다.
        let _ = gen.clear_kv_cache();
        drop(gen_guard);

        Ok(res)
    }

    pub async fn ensure_qwen3_5(&self, needs_vision: bool) -> anyhow::Result<()> {
        // 🌟 [VISION-JIT] 이미 2B 가 상주 중이고 mmproj 재로드 소스가 등록되어 있다면,
        //    2GB 텍스트 모델을 통째로 파기/재로딩하지 않고 비전 가중치(약 600MB)만 붙였다 뗍니다.
        //    기존에는 '이미지 추출 → thead 추출' 처럼 비전/텍스트가 번갈아 올 때마다
        //    GGUF 를 처음부터 다시 읽어야 했습니다.
        {
            let mut guard = self.qwen3_5_generator.lock().await;
            if let Some(gen) = guard.as_mut() {
                if gen.vision_capable() && gen.is_vision_jit_capable() {
                    if gen.vision_resident() != needs_vision {
                        gen.set_vision_active(needs_vision)?;
                        println!(
                            "[MODEL] Qwen 3.5 vision weights {} WITHOUT full reload (2B text model stays resident).",
                            if needs_vision { "ATTACHED" } else { "DETACHED" }
                        );
                    }
                    return Ok(());
                }
            }
        }
        let needs_load = {
            let guard = self.qwen3_5_generator.lock().await;
            if let Some(gen) = guard.as_ref() {
                let is_large = gen.vision_capable();
                is_large != needs_vision // 🌟 wants_large 대신 needs_vision 직접 사용
            } else {
                true
            }
        };
        if needs_load {
            println!("[MODEL] Loading Qwen 3.5 Generator (2B) (Vision: {})...", needs_vision);
            // 🌟 [GENERATION HOLD] ensure_qwen3 와 같은 이유로 자체 홀드를 선언합니다.
            //    이 함수는 secure_vram_relay 경유 외에 ensure_generator_ext 에서도
            //    직접 호출되므로, 진입점마다 홀드가 있어야 창이 남지 않습니다.
            let _load_hold = self.hold_generation();
            // 🌟 [CRITICAL FIX] SigLIP2가 로드되어 있다면(이미지 추출 파이프라인),
            // 전체 Purge가 비전 엔진을 죽이므로 Generator만 정리합니다.
            let is_vision_pipeline_active = self.siglip2_model.lock().await.is_some();
            if is_vision_pipeline_active {
                println!("[RELAY] 🛡️ SigLIP2 is resident. Skipping deep purge to protect vision engine.");
                // Generator 슬롯만 클리어 (KV 캐시 및 스토리지 해제)
                let mut gen = self.generator.lock().await;
                if let Some(mut g) = gen.take() {
                    let _ = g.clear_kv_cache();
                    let _ = g.qwen.drop_kv_storage();
                }
                // 🌟 [VRAM] 이미지 추출 파이프라인(extract_from_image)에서는
                //    임베딩 모델(384차원 97M)을 한 번도 사용하지 않습니다.
                //    해제하여 Qwen3.5 로드 전에 VRAM 을 확보합니다.
                {
                    let mut emb = self.embedding_model.lock().await;
                    if emb.is_some() {
                        *emb = None;
                        println!("[RELAY] Embedding model released for VRAM (image pipeline).");
                    }
                }
                {
                    let mut cache = self.embedding_cache.lock().await;
                    cache.clear();
                }
            } else {
                // 🌟 [DOUBLE PURGE ELIMINATED] ensure_qwen3 와 동일한 상태 판정입니다.
                //    secure_vram_relay 가 이미 퍼지한 뒤 이 함수로 내려오면
                //    내릴 것이 하나도 없는데도 CUDA 동기화 10초를 다시 씁니다.
                //
                // 🌟 [CROSSOVER] ensure_qwen3 와 같은 이유로 임베딩을 트리거에서 뺍니다.
                //    이 분기는 is_vision_pipeline_active == false 인 경우만 도달하므로
                //    SigLIP2 는 여기서 이미 None 임이 보장됩니다.
                //
                //  ── Qwen3.5 는 2B 라 대부분 SWAP 으로 떨어집니다 ──
                //   그것이 오판이 아니라 정확한 판정입니다. 2GB 를 임베딩 위에
                //   얹을 수 있는 여유가 실제로 있을 때만 COEXIST 로 갑니다.
                let embed_resident = { self.embedding_model.lock().await.is_some() };
                let other_resident = {
                    self.generator.lock().await.is_some()
                        || self.qwen3_generator.lock().await.is_some()
                        || self.qwen3_5_generator.lock().await.is_some()
                };
                let keep_embedding = embed_resident
                    && !other_resident
                    && self.embedding_coexist_ok(ModelSize::Qwen3_5);

                if other_resident || (embed_resident && !keep_embedding) {
                    // 일반 텍스트 경로에서는 기존대로 전체 Purge 수행
                    self.deep_purge_resources().await;
                } else if keep_embedding {
                    println!(
                        "[MODEL] 🤝 [CROSSOVER/COEXIST] 자유 {}MB >= Qwen3.5 예산 {}MB. 임베딩을 상주시킨 채 로드합니다. (퍼지 생략)",
                        self.get_free_vram_mb(),
                        self.generation_budget_mb(ModelSize::Qwen3_5)
                    );
                } else {
                    println!("[MODEL] ⚡ [PURGE SKIP] 해제 대상 슬롯이 하나도 없어 중복 퍼지를 생략합니다.");
                }
            }
            // 🌟 [핵심 픽스] 여기서도 로딩 전에 미리 방주인 등록!
            {
                *self.current_size.lock().await = Some(ModelSize::Qwen3_5);
            }
            let path = self.qwen3_5_model_path.clone();
            let dev = self.device_config.device.clone();
            // 🌟 [TIMEOUT ADD] 180초 타임아웃으로 무한 로딩 방지
            let load_result = tokio::time::timeout(
                std::time::Duration::from_secs(180),
                tokio::task::spawn_blocking(move || {
                    let gguf_files = crate::utils::find_type_files(&path, "gguf").unwrap_or_default();
                    let model_gguf = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned().ok_or_else(|| anyhow::anyhow!("No model GGUF found"))?;
                    let mmproj_gguf = if needs_vision {
                        gguf_files.iter().find(|f| f.contains("mmproj")).cloned()
                    } else {
                        None
                    };
                    Qwen3_5GenerateModel::init_from_gguf(&model_gguf, mmproj_gguf.as_deref(), Some(&dev))
                })
            ).await;
            let gen = match load_result {
                Ok(Ok(Ok(g))) => g,
                Ok(Ok(Err(e))) => {
                    // 로드 실패 시 현재 사이즈 등록을 되돌리고 에러 전파
                    {
                        *self.current_size.lock().await = None;
                    }
                    return Err(anyhow::anyhow!("Qwen 3.5 GGUF load failed: {}", e));
                },
                Ok(Err(join_err)) => {
                    {
                        *self.current_size.lock().await = None;
                    }
                    return Err(anyhow::anyhow!("Qwen 3.5 load task join error: {}", join_err));
                },
                Err(_) => {
                    // 타임아웃
                    {
                        *self.current_size.lock().await = None;
                    }
                    return Err(anyhow::anyhow!("Qwen 3.5 load timeout (180s). GPU may be out of memory or driver stalled."));
                }
            };
            let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
            *q35_gen_guard = Some(gen);
            // 🌟 [CRITICAL FIX] 시스템 장부에 Qwen3.5가 켜졌음을 명시하여 스냅샷 미아 발생 방지!
            let mut current_size_guard = self.current_size.lock().await;
            *current_size_guard = Some(ModelSize::Qwen3_5);
        }
        Ok(())
    }

    pub async fn check_embedding_downloaded(&self) -> anyhow::Result<()> {
        let weights_path = self.embedding_path.join("model.safetensors");
        if !weights_path.exists() {
            let err_msg = "Embedding model is missing. Please go to the Settings tab and download the required models.";
            println!("[MODEL] 🚨 {}", err_msg);
            use tauri::Emitter;
            let _ = self.app_handle.emit("app_error_alert", serde_json::json!({
                "message": err_msg,
                "action": "open_settings"
            }));
            return Err(anyhow::anyhow!(err_msg));
        }
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        // 실제 메모리에 올리기 직전에 파일 존재 여부를 다시 한 번 방어합니다.
        self.check_embedding_downloaded().await?;
        // 🌟 [GENERATION HOLD YIELD] 생성 모델 전환 구간이면 로드를 양보합니다.
        //
        //  ── 왜 락 '밖' 에서 대기하는가 ──
        //   embedding_model 뮤텍스를 쥔 채 대기하면 deep_purge_resources 의
        //   Step 2 가 같은 뮤텍스를 못 잡아 즉시 데드락입니다.
        //   반드시 락을 잡기 전에 대기를 끝내야 합니다.
        //
        //  ── 상한의 성격 ──
        //   아래 반복 상한은 '판정 기준' 이 아니라 데드락 방지 안전핀입니다.
        //   홀드를 푸는 주체(GenerationHold::drop)가 패닉 등으로 사라지는
        //   상상 가능한 최악의 경우에도 앱이 영구히 멈추지 않게 합니다.
        //   정상 경로에서는 첫 폴에서 바로 통과하거나 전환이 끝나는 즉시 풀립니다.
        if !self.is_cpu_mode && self.is_generation_held() {
            println!("[MODEL] ⏸️ [EMBED YIELD] 생성 모델 전환 구간입니다. 임베딩 로드를 양보하고 대기합니다.");
            let yield_started = Instant::now();
            let mut polls = 0u32;
            while self.is_generation_held() {
                // 대기 중 다른 주체가 이미 올려 두었다면 즉시 종료합니다.
                if self.embedding_model.lock().await.is_some() {
                    println!("[MODEL] ▶️ [EMBED YIELD] 대기 중 임베딩이 이미 상주하게 되어 로드를 생략합니다.");
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(120)).await;
                polls += 1;
                if polls > 500 {
                    println!("[MODEL] ⚠️ [EMBED YIELD] 안전핀 도달({:.1}s). 홀드가 해제되지 않아 로드를 강행합니다.", yield_started.elapsed().as_secs_f32());
                    break;
                }
            }
            println!("[MODEL] ▶️ [EMBED YIELD] 전환 완료({:.2}s 대기). 임베딩 로드를 재개합니다.", yield_started.elapsed().as_secs_f32());
        }
        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            // 🌟 [VRAM GATE] 임베딩을 올리기 전에 자유 메모리를 체크합니다.
            //    다른 모델이 상주 중인데 메모리가 부족하면,
            //    동시 상주 대신 순차 모드(다른 모델 언로드 → 임베딩 로드)로 진입합니다.
            //    이후 다른 모델이 다시 필요해지면 각 모델의
            //    ensure_qwen3() / ensure_qwen3_5() / secure_vram_relay() 가
            //    기존처럼 자동으로 재로드합니다.
            //
            //    ⚠️ 이 게이트는 '자유 메모리 부족' 만 봅니다. 퍼지 직후에는
            //       4GB 가 비어 있어 통과하므로, 퍼지-재로드 레이스는
            //       위의 GENERATION HOLD YIELD 가 담당합니다. 두 장치는 역할이 다릅니다.
            if !self.is_cpu_mode {
                let free_mb = self.get_free_vram_mb();
                // 🌟 [CROSSOVER] 하드코딩 350MB 를 실측 기반 예산으로 교체합니다.
                //
                //  ── 왜 상수가 위험했나 ──
                //   350 은 granite-97m 을 눈대중한 값이며, dtype·GPU·드라이버가 바뀌면
                //   즉시 틀립니다. 과소평가되면 게이트를 통과한 뒤 로드에서 OOM 이 나고,
                //   과대평가되면 여유가 있는데도 매번 전체 퍼지를 유발합니다.
                //   embedding_budget_mb() 는 디스크 가중치 크기에서 출발해
                //   첫 로드 전후의 free VRAM 차이로 실측값이 되며,
                //   대량 배치에서 관측한 activation 여유까지 더해 돌려줍니다.
                let needed_mb = self.embedding_budget_mb();
                if free_mb < needed_mb {
                    println!(
                        "[MODEL] ⚠️ [VRAM GATE] 자유 {}MB < 예산 {}MB. 생성 슬롯을 먼저 반환합니다.",
                        free_mb, needed_mb
                    );
                    // 락을 해제하지 않으면 아래 반환 로직 내부에서
                    // 같은 뮤텍스를 다시 잡아 데드락이 됩니다.
                    drop(emb_guard);

                    // 🌟 [LADDER] ① 생성 슬롯만 반환 → ② 그래도 모자라면 전체 퍼지
                    //   전체 퍼지는 SigLIP2 파기 + CUDA 동기화 최대 10초를 쓰므로
                    //   정말 필요할 때만 도달하도록 단을 나눕니다.
                    //   대부분의 경우 ①에서 끝나며, 그만큼 왕복 비용이 줄어듭니다.
                    self.unload_generation_slots("embedding vram gate").await;
                    if self.get_free_vram_mb() < needed_mb {
                        println!(
                            "[MODEL] ⚠️ [VRAM GATE] 생성 슬롯 반환 후에도 자유 {}MB < 예산 {}MB. 전체 퍼지로 승격합니다.",
                            self.get_free_vram_mb(), needed_mb
                        );
                        self.deep_purge_resources().await;
                    }

                    emb_guard = self.embedding_model.lock().await;
                    println!("[MODEL] ✅ [VRAM GATE] 순차 모드 준비 완료. 자유 {}MB", self.get_free_vram_mb());
                }
            }
            let self_clone = self.embedding_path.clone();
            // 🌟 CPU 강제 할당을 제거하고 시스템 설정(GPU)을 그대로 사용하여 초고속 VRAM 연산을 수행합니다.
            let target_device = self.device_config.device.clone();
            println!("[MODEL] Loading Embedding Model on {:?}...", target_device);
            let target_device_clone = target_device.clone();
            let emb = tokio::task::spawn_blocking(move || {
                EmbeddingModel::new_with_device(&self_clone, &target_device_clone)
            }).await??;
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    /// SigLIP2 모델 파일이 디스크에 존재하는지 확인합니다.
    /// 존재하지 않으면 에러를 반환하며, 호출 측(scheduler)에서
    /// app_error_alert 이벤트를 발행하여 프론트엔드에 알립니다.
    pub async fn check_siglip2_downloaded(&self) -> anyhow::Result<()> {
        let base = std::path::Path::new(&self.siglip2_model_path);
        let safetensors_path = base.join("model.safetensors");
        let config_path = base.join("config.json");

        // model.safetensors 존재 + 최소 100MB 체크
        let model_exists = safetensors_path.exists()
            && std::fs::metadata(&safetensors_path)
                .map(|m| m.len())
                .unwrap_or(0)
                > 100_000_000;

        let config_exists = config_path.exists();

        if !model_exists || !config_exists {
            return Err(anyhow::anyhow!(
                "SigLIP2 model not downloaded. Please download it in Settings."
            ));
        }
        Ok(())
    }

    /// 🌟 [SIGLIP2 ENSURE v2] 필요한 인코더만 정확히 올립니다.
    ///
    ///  ── v1 의 결함 2가지 ──
    ///   ① needs_text 무시
    ///      `if guard.is_some() { return Ok(()); }` 가 요구 사양을 보지 않아,
    ///      비전만 상주한 상태에서 ensure_siglip2(true) 가 그냥 성공했습니다.
    ///      호출부는 텍스트가 준비된 줄 알고 진행하다가
    ///        ⚪ [GROUNDING SKIP] 텍스트 인코더가 없어 검증을 건너뜁니다.
    ///      로 조용히 실패합니다. 로그 한 줄만 남아 원인 추적이 불가능했습니다.
    ///
    ///   ② 비전 강제
    ///      load_vision_only 를 무조건 먼저 호출하므로 텍스트만 필요한 경로도
    ///      820MB 를 함께 올렸습니다. STEP 6 과 검색 질의 벡터 생성이 여기 해당하며,
    ///      검색은 질의마다 반복되므로 누적 비용이 큽니다.
    ///
    ///  ── v2 ──
    ///   (needs_vision, needs_text) 를 받아 '부족한 쪽만' 부착합니다.
    ///   이미 상주한 가중치는 그대로 재사용하므로 전체 파기/재로딩이 없습니다.
    ///   (Qwen3.5 의 VISION-JIT set_vision_active 와 같은 원리)
    pub async fn ensure_siglip2_ext(&self, needs_vision: bool, needs_text: bool) -> anyhow::Result<()> {
        let dir = std::path::PathBuf::from(&self.siglip2_model_path);

        // ── ① 이미 상주 중이면 부족한 부분만 부착합니다 ──
        {
            let mut guard = self.siglip2_model.lock().await;
            if let Some(model) = guard.as_mut() {
                let want_v = needs_vision && !model.has_vision();
                let want_t = needs_text && !model.has_text();
                if !want_v && !want_t {
                    return Ok(());
                }
                if want_v {
                    model.load_vision_encoder(&dir)?;
                }
                if want_t {
                    model.load_text_encoder(&dir)?;
                }
                println!(
                    "[MODEL] SigLIP2 upgraded in place (vision: {}, text: {}). No full reload.",
                    model.has_vision(), model.has_text()
                );
                return Ok(());
            }
        }

        // ── ② 신규 로드 ──
        let path = self.siglip2_model_path.clone();
        let dev = self.device_config.device.clone();
        let dtype = if self.is_cpu_mode {
            candle_core::DType::F32
        } else {
            candle_core::DType::BF16
        };

        println!(
            "[MODEL] Loading SigLIP2 ({:?}) | vision: {} | text: {}",
            dtype, needs_vision, needs_text
        );

        // 🌟 [NO-OP GUARD] 둘 다 필요 없다고 요청하면 아무것도 올리지 않습니다.
        //
        //  ── 왜 필요한가 ──
        //   아래 신규 로드 분기는 `!needs_vision && needs_text` 만 텍스트 전용으로 보내고,
        //   나머지를 전부 load_vision_only 로 흘려보냅니다.
        //   그래서 (false, false) 조합이 들어오면 "아무것도 필요 없다" 는 요청에
        //   비전 856MB 를 올려 주는 정반대 동작을 합니다.
        //   현재 호출부에는 이 조합이 없지만, LAZY TEXT 도입 후 슬롯이 비어 있는 상태에서
        //   도달할 여지가 생기므로 진입 지점에서 차단합니다.
        if !needs_vision && !needs_text {
            println!("[MODEL] SigLIP2 ensure requested with no encoder. Nothing to load.");
            return Ok(());
        }

        let model = tokio::task::spawn_blocking(move || {
            let dir = std::path::Path::new(&path);
            let config_path = dir.join("config.json");
            let config = crate::models::siglip2::Siglip2Config::from_json(&config_path)?;

            // 🌟 텍스트만 필요하면 비전 가중치를 아예 읽지 않습니다. (~820MB 절약)
            if !needs_vision && needs_text {
                return crate::models::siglip2::Siglip2Model::load_text_only(
                    dir, &config, &dev, dtype,
                );
            }

            let safetensors_path = dir.join("model.safetensors");
            let mut model = crate::models::siglip2::Siglip2Model::load_vision_only(
                &safetensors_path,
                &config,
                &dev,
                dtype,
            )?;

            if needs_text {
                model.load_text_encoder(dir)?;
            }

            Ok::<_, anyhow::Error>(model)
        })
        .await??;

        let mut guard = self.siglip2_model.lock().await;
        if guard.is_some() {
            // 락을 놓은 사이 다른 태스크가 로드를 마쳤습니다. 방금 만든 것은 버립니다.
            return Ok(());
        }
        *guard = Some(model);
        println!("[MODEL] SigLIP2 loaded successfully.");
        Ok(())
    }

    /// 🌟 [BACK-COMPAT] 기존 호출부를 살려 둡니다.
    ///    구 시그니처는 '비전은 항상 필요' 를 전제했으므로 needs_vision=true 로 위임합니다.
    pub async fn ensure_siglip2(&self, needs_text: bool) -> anyhow::Result<()> {
        self.ensure_siglip2_ext(true, needs_text).await
    }

    /// 🌟 [SigLIP2 RELEASE] 비전+텍스트 인코더를 통째로 내리고 CUDA 캐시까지 반환합니다.
    ///
    ///  ── 왜 별도 헬퍼인가 ──
    ///   기존에는 `*guard = None` 만 수행했습니다. candle 의 CUDA 백엔드는
    ///   caching allocator 를 쓰므로 그것만으로는 VRAM 이 OS 로 돌아오지 않습니다.
    ///   `deep_purge_resources` 가 하는 것과 같은 synchronize + 컨텍스트 재생성을
    ///   여기서도 수행해야 실제 free VRAM 이 올라갑니다.
    ///
    ///  ── 언제 부르는가 ──
    ///   STEP 1~4(패치 임베딩 · 문서분류 · 히트맵 · 크롭계획)가 끝나면
    ///   SigLIP2 는 더 이상 필요하지 않습니다.
    ///   그 시점이 곧 Qwen3.5(2B) 를 올려야 하는 시점이므로 여기서 반드시 비웁니다.
    ///   (실측: 해제 없이 진입 시 첫 크롭에서 free VRAM 147MB)
    pub async fn release_siglip2(&self, reason: &str) {
        println!("[VRAM] release_siglip2 ENTER ({}) — lock acquiring...", reason);
        let released = {
            let mut guard = self.siglip2_model.lock().await;
            println!("[VRAM] lock acquired. Dropping SigLIP2 model (cudaFree implicit sync may happen here)...");
            if guard.is_some() {
                *guard = None;
                true
            } else {
                false
            }
        };
        if !released {
            println!("[VRAM] release_siglip2: already released. skip.");
            return;
        }
        println!(
            "[VRAM] SigLIP2 fully released ({}). vision ~820MB + text ~1.4GB returned.",
            reason
        );
        if !self.is_cpu_mode {
            // 🌟 [HANG FIX] 텐서 드롭이 이미 암묵 synchronize 를 수행한 상태에서
            //    여기서 다시 무한정으로 synchronize 를 기다리면 저VRAM(3.5GB) 환경에서
            //    드라이버 스톨 시 영구 대기됩니다. 5초 상한을 두고 초과면 그냥 진행합니다.
            //    (동기화는 '언제' 끝나도 정확성에 영향이 없는 베스트에포트 정리입니다)
            let dev = self.device_config.device.clone();
            let sync_res = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() {
                        let _ = dev.synchronize();
                    }
                }),
            )
            .await;
            match sync_res {
                Ok(Ok(())) => println!("[VRAM] CUDA synchronize OK."),
                Ok(Err(e)) => println!("[VRAM] CUDA synchronize join error: {:?}", e),
                Err(_) => println!("[VRAM] ⚠️ CUDA synchronize 5s timeout — 드라이버 스톨 감지, 동기화 없이 진행합니다."),
            }
            // caching allocator 가 붙들고 있는 풀을 OS 로 밀어내기 위한 컨텍스트 재생성
            let _ = candle_core::Device::new_cuda(self.device_config.gpu_id as usize);
            println!("[VRAM] CUDA context refresh done. Proceeding to STAGE-5.");

            // 🌟 [VRAM SETTLE AFTER RELEASE] SigLIP2 해제 후 실제 여유 메모리 확인
            if let Some(token) = None::<Arc<AtomicBool>> {
                let _ = token;
            }
            let _ = self.wait_for_vram_settle(1200, 10, None).await;
        }
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
            let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        #[cfg(target_os = "linux")]
        unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
        #[cfg(target_os = "macos")]
        unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
    }

    /// 🌟 [LAZY TEXT] SigLIP2 텍스트 인코더를 '캐시 미스가 실제로 발생했을 때만' 올립니다.
    ///
    ///  ── 왜 이 구조인가 ──
    ///   "필요한 앵커 구 목록을 미리 만들어 캐시 적중 여부를 검사" 하는 방식은
    ///   build_column_heatmaps 의 구 수집 로직(자기참조 드롭 / doc_type 드롭 /
    ///   LABEL+VALUE 이중축 / 표 구조 앵커 편입)을 통째로 복제해야 합니다.
    ///   그 복제본이 원본과 한 줄이라도 어긋나면 게이트가 거짓말을 하고
    ///   파이프라인이 하드 에러로 죽습니다. 유지보수 비용이 이득보다 큽니다.
    ///
    ///   대신 '해 보고, 정말 못 하면 그때 올린다' 로 뒤집습니다.
    ///   encode_phrases_shared 는 캐시 미스가 있을 때만
    ///   ERR_TEXT_ENCODER_REQUIRED 접두어를 붙여 실패하고, 그 실패는
    ///   build_anchor_bank 가 순전파를 시작하기 '전' 에 발생하므로 낭비가 0 입니다.
    ///
    ///  ── 락 안전성 ──
    ///   1차 시도의 가드는 스코프 블록으로 수명을 고정해 .await 이전에 반드시 해제됩니다.
    ///   tokio Mutex 는 재진입이 불가능하므로(이 파일의 [SCOPED LOCK] 주석과 같은 이유)
    ///   ensure_siglip2_ext 가 같은 락을 기다리다 셀프 데드록되는 경로를 만들지 않습니다.
    pub async fn with_siglip_text<T, F>(&self, what: &str, f: F) -> anyhow::Result<T>
    where
        F: Fn(&crate::models::siglip2::Siglip2Model) -> anyhow::Result<T> + Send,
        T: Send,
    {
        use crate::models::siglip2::vision_encoder::ERR_TEXT_ENCODER_REQUIRED;

        // ── 1차 : 현재 상태 그대로 시도합니다. 앵커가 전부 캐시에 있으면 여기서 끝납니다. ──
        {
            let guard = self.siglip2_model.lock().await;
            if let Some(m) = guard.as_ref() {
                match f(m) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        if !e.to_string().contains(ERR_TEXT_ENCODER_REQUIRED) {
                            return Err(e);
                        }
                        println!(
                            "[SigLIP2] '{}' 에 필요한 앵커 구 일부가 캐시에 없습니다. 텍스트 인코더를 지금 부착합니다.",
                            what
                        );
                    }
                }
            }
        }

        // ── 2차 : 텍스트 인코더를 부착한 뒤 1회 재시도합니다. ──
        self.ensure_siglip2_ext(false, true).await?;
        let guard = self.siglip2_model.lock().await;
        let m = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SigLIP2 model not loaded for '{}'", what))?;
        f(m)
    }

    // 🌟 [CRITICAL FIX] config.json의 물리적 텐서 크기와 실제 훈련된 Context Length를 완벽히 분리합니다.
    pub async fn truncate_pug_context(&self, pug: &str, is_detail: bool, margin_tokens: usize, bottom_drop_tokens: Option<usize>) -> String {
        // 🌟 current_size 를 읽고 한 번도 쓰지 않아 불필요한 뮤텍스 획득만 발생했습니다.
        //    아래에서 제너레이터 슬롯을 순서대로 확인하므로 이 값이 필요 없습니다.
        let max_context_length: usize = if is_detail { 60_000 } else { 9_000 };
        let tokenizer_path = &self.qwen_model_path;

        // 🌟 한도(최대 토큰)를 계산하고, 버릴 하단 토큰(bottom_drop_tokens)을 파서에 함께 전달합니다.
        let final_max = max_context_length.saturating_sub(margin_tokens);

        // 2. 이미 활성화된 제너레이터가 있다면 그 안에 탑재된 토크나이저를 즉시 재사용합니다.
        if let Some(gen) = self.qwen3_5_generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }
        if let Some(gen) = self.qwen3_generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }
        if let Some(gen) = self.generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }

        // 3. 모델이 VRAM에 없을 경우, 디스크에서 가볍게 토크나이저만 읽어와서 정확한 토큰 수 기반으로 절단합니다.
        if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(tokenizer_path) {
            crate::parsing::truncate_pug_by_tokens(pug, final_max, &tokenizer, bottom_drop_tokens)
        } else {
            pug.to_string()
        }
    }

    pub async fn new(app_handle: tauri::AppHandle, device_preference: Option<&str>) -> anyhow::Result<Self> {
        // Default to true for SSD-Swap unless user explicitly wants pure CPU
        let is_disk_swap = match device_preference {
            Some("cpu") => false,
            _ => true,
        };
        
        println!("[MODEL-00] Initializing LogisModel (Preference: {:?}, DiskSwap: {})", device_preference, is_disk_swap);

        let mut config = utils::get_optimal_device_config();
        
        if device_preference == Some("cpu") {
            println!("⚠️ [MODEL] EXPLICIT CPU MODE FORCED by user/system preference.");
            config = utils::DeviceConfig {
                device: Device::Cpu,
                is_cpu: true,
                classify_chunk_size: 12000,
                extract_chunk_size: 12000,
                name: "CPU-Forced".to_string(),
                gpu_id: 0,
            };
        } else {
            // 🌟 [CRITICAL FIX] VRAM 즉각 해제를 위해 전역 캐싱(Singleton) 디바이스 사용을 중단하고 매번 새 컨텍스트를 생성합니다.
            // utils::get_cuda_device는 내부에 메모리 풀(Caching Allocator)을 영구 보존하므로 작업 관리자에서 VRAM이 떨어지지 않는 주범입니다.
            let fresh_dev = candle_core::Device::new_cuda(config.gpu_id as usize).unwrap_or(candle_core::Device::Cpu);
            config.device = fresh_dev;
            println!("🚀 [MODEL] Running in default mode ({}) with Fresh CUDA Context", config.name);
        }

        let app_dir = crate::utils::get_app_dir();
        let base_path = app_dir.join("models");
        
        // [FIX] Normalize UNC paths for Windows to prevent "builder error" in model loaders
        let normalize_path = |path: std::path::PathBuf| -> String {
            let s = path.to_string_lossy().to_string();
            if s.starts_with(r"\\?\") {
                s[4..].to_string()
            } else {
                s
            }
        };

        let qwen_model_path = normalize_path(base_path.join("Qwen3-0.6B-Instruct-gguf")); 
        let qwen3_model_path = normalize_path(base_path.join("Qwen3-0.6B-Instruct-gguf")); 
        let qwen3_5_model_path = normalize_path(base_path.join("Qwen3.5-2B-Instruct-gguf"));
        let siglip2_model_path = normalize_path(base_path.join("siglip2-so400m-patch16-naflex"));
        let embedding_path = base_path.join("granite-embedding-97m-multilingual-r2");

        let max_tokens_limit = 65536; 

        Ok(Self {
            app_handle,
            generator: Arc::new(TokioMutex::new(None)),
            qwen3_generator: Arc::new(TokioMutex::new(None)), // 🌟 추가
            qwen3_5_generator: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            embedding_cache: Arc::new(TokioMutex::new(std::collections::HashMap::new())), // 🌟 캐시 초기화
            // 🌟 [GENERATION HOLD] 0 = 전환 구간 아님 = 임베딩 자유 로드 가능
            generation_hold: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            is_cpu_mode: config.is_cpu,
            is_disk_swap,
            dual_mode_enabled: true, 
            qwen_model_path,    // 🌟 교체
            qwen3_model_path,   // 🌟 교체
            qwen3_5_model_path,
            embedding_path,
            device_config: config.clone(),
            max_tokens_limit: max_tokens_limit as u32,
            _dtype: None, 
            current_size: Arc::new(TokioMutex::new(None)),

            siglip2_model: Arc::new(TokioMutex::new(None)),
            siglip2_config: None,
            siglip2_model_path,
        })
    }
}

