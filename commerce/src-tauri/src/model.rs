use crate::utils;
use anyhow::anyhow;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::embedding::EmbeddingModel;
use crate::openai_types::{
    ChatCompletionParameters,
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ImageURL,
};
use candle_core::{Device, DType};
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, atomic::AtomicBool};
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;


pub fn generate_rich_summary(doc_type: &str, data: &Value) -> String {
    let type_map = json!({
        "CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List",
        "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit",
        "tracking": "Shipping Label / Tracking Info"
    });
    
    let full_type = type_map.get(doc_type).and_then(|s| s.as_str()).unwrap_or(doc_type);
    let mut parts = vec![format!("This is a {} document.", full_type)];

    if let Some(h) = data.get("header") {
        if let Some(no) = h.get("document_number").and_then(|s| s.as_str()) {
            if no != "N/A" && !no.is_empty() {
                parts.push(format!("Document number is {}.", no));
            }
        }
        if let Some(date) = h.get("issue_date").and_then(|s| s.as_str()) {
            if date != "N/A" && !date.is_empty() {
                parts.push(format!("Issued on {}.", date));
            }
        }
    }

    if doc_type == "tracking" {
        if let Some(tn) = data.get("tracking_number").and_then(|s| s.as_str()) {
            parts.push(format!("The tracking number is {}.", tn));
        }
        if let Some(text) = data.get("text").and_then(|s| s.as_str()) {
            parts.push(text.to_string());
        }
    }

    if let Some(p) = data.get("parties") {
        let sup = p.get("supplier_name").and_then(|s| s.as_str());
        let buy = p.get("buyer_name").and_then(|s| s.as_str());
        
        let has_sup = sup.is_some() && sup.unwrap() != "N/A";
        let has_buy = buy.is_some() && buy.unwrap() != "N/A";

        if has_sup && has_buy {
            parts.push(format!("Transaction involved {} as the supplier/shipper and {} as the buyer/consignee.", sup.unwrap(), buy.unwrap()));
        } else if has_sup {
            parts.push(format!("Supplier/Shipper is {}.", sup.unwrap()));
        } else if has_buy {
            parts.push(format!("Buyer/Consignee is {}.", buy.unwrap()));
        }
    }

    if let Some(f) = data.get("financials") {
        if let Some(amt) = f.get("amount_total") {
             let amt_str = if amt.is_number() { amt.to_string() } else { amt.as_str().unwrap_or("0").to_string() };
             let curr = f.get("currency_code").and_then(|s| s.as_str()).unwrap_or("USD");
             if amt_str != "0" && amt_str != "0.0" {
                 parts.push(format!("Total amount is {} {}.", amt_str, curr));
             }
        }
    }

    if let Some(l) = data.get("logistics") {
        let pol = l.get("location_port_of_loading").and_then(|s| s.as_str());
        let pod = l.get("location_port_of_discharge").and_then(|s| s.as_str());
        
        if let (Some(o), Some(d)) = (pol, pod) {
            if o != "N/A" && d != "N/A" {
                parts.push(format!("Shipped from {} to {}.", o, d));
            }
        }
        
        if let Some(mode) = l.get("transport_mode").and_then(|s| s.as_str()) {
            parts.push(format!("Transport mode is {}.", mode));
        }
    }

    if let Some(items) = data.get("line_items").and_then(|v| v.as_array()) {
        let mut item_descs = Vec::new();
        for item in items.iter().take(5) {
            if let Some(d) = item.get("description").and_then(|s| s.as_str()) {
                if d.len() > 3 { item_descs.push(d); }
            }
        }
        if !item_descs.is_empty() {
            parts.push(format!("Contains items: {}.", item_descs.join(", ")));
        }
    }
    
    parts.join(" ")
}

use tokio::sync::Mutex as TokioMutex;
use std::time::{Duration, Instant};

use crate::models::qwen3::generate::Qwen3GenerateModel; // 🌟 Qwen3 텍스트 전용 로직 임포트

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Qwen,    // 0.6B for Ingestion (기존 Small)
    Qwen3,   // Qwen3 Text Model (기존 Large, /qwen3/ 로직 전용)
    Qwen3_5, // 2B Qwen 3.5 (Text Optimized)
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    pub embedding_cache: Arc<TokioMutex<std::collections::HashMap<String, Vec<f32>>>>,

    pub is_cpu_mode: bool, 
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    qwen_model_path: String,      // 🌟 (기존 small_model_path 대신 이름 맞춤)
    qwen3_model_path: String,     // 🌟 Qwen3 모델 경로 추가
    qwen3_5_model_path: String,
    embedding_path: std::path::PathBuf,
    pub device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    _dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,
}

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
    }

    /// [CLEANUP] Aggressive Factory Reset Purge (Reinforced with Diagnostics)
    pub async fn deep_purge_resources(&self) {
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
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    // --- [NEW] VRAM Settlement Monitor (Smart Polling) ---
    async fn wait_for_vram_settle(&self, target_free_mb: u64, timeout_sec: u64, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
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
            
            // 🌟 [CRITICAL FIX] unload_generator가 소유권을 훔쳐가 KV 캐시 클리어를 방해하는 버그 해결!
            // 바로 deep_purge_resources만 단독 호출하여 VRAM을 100% 안전하게 날려줍니다.
            self.deep_purge_resources().await;
            
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
            
            // 🌟 [CRITICAL FIX] unload_generator가 소유권을 훔쳐가 KV 캐시 클리어를 방해하는 버그 해결!
            self.deep_purge_resources().await;
            
            // 🌟 [핵심 픽스] 여기서도 로딩 전에 미리 방주인 등록!
            {
                *self.current_size.lock().await = Some(ModelSize::Qwen3_5);
            }
            
            let path = self.qwen3_5_model_path.clone();
            let dev = self.device_config.device.clone();
            
            let gen = tokio::task::spawn_blocking(move || {
                let gguf_files = crate::utils::find_type_files(&path, "gguf").unwrap_or_default();
                let model_gguf = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned().ok_or_else(|| anyhow::anyhow!("No model GGUF found"))?;
                
                // 🌟 [수정]
                let mmproj_gguf = if needs_vision {
                    gguf_files.iter().find(|f| f.contains("mmproj")).cloned()
                } else {
                    None
                };
                
                Qwen3_5GenerateModel::init_from_gguf(&model_gguf, mmproj_gguf.as_deref(), Some(&dev))
            }).await??;
            
            let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
            *q35_gen_guard = Some(gen);
            
            // 🌟 [CRITICAL FIX] 시스템 장부에 Qwen3.5가 켜졌음을 명시하여 스냅샷 미아 발생 방지!
            let mut current_size_guard = self.current_size.lock().await;
            *current_size_guard = Some(ModelSize::Qwen3_5);
        }
        Ok(())
    }

    // 🌟 [신규 추가] 모델 파일 존재 여부만 가볍게 체크하는 함수 (메모리 로딩 안 함)
    // 🌟 [신규 추가] 모델 파일 존재 여부만 가볍게 체크하는 함수 (메모리 로딩 안 함)
    pub async fn check_embedding_downloaded(&self) -> anyhow::Result<()> {
        let weights_path = self.embedding_path.join("model.safetensors");
        if !weights_path.exists() {
            let err_msg = "Embedding model is missing. Please go to the Settings tab and download the required models.";
            println!("[MODEL] 🚨 {}", err_msg);
            use tauri::Emitter;
            let _ = self.app_handle.emit("app_error_alert", serde_json::json!({ "message": err_msg }));
            return Err(anyhow::anyhow!(err_msg));
        }
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        // 실제 메모리에 올리기 직전에 파일 존재 여부를 다시 한 번 방어합니다.
        self.check_embedding_downloaded().await?; 
        
        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
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

    // 🌟 [CRITICAL FIX] config.json의 물리적 텐서 크기와 실제 훈련된 Context Length를 완벽히 분리합니다.
    pub async fn truncate_pug_context(&self, pug: &str, is_detail: bool, margin_tokens: usize, bottom_drop_tokens: Option<usize>) -> String {
        let current_size = *self.current_size.lock().await;
        
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
        let embedding_path = base_path.join("granite-embedding-97m-multilingual-r2");

        let max_tokens_limit = 65536; 

        Ok(Self {
            app_handle,
            generator: Arc::new(TokioMutex::new(None)),
            qwen3_generator: Arc::new(TokioMutex::new(None)), // 🌟 추가
            qwen3_5_generator: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            embedding_cache: Arc::new(TokioMutex::new(std::collections::HashMap::new())), // 🌟 캐시 초기화
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
        })
    }

    pub async fn extract_from_image(
        &self,
        task_id: String,
        image_path: String,
        language: String,
        search_mode: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.clone();
        
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            use tauri::Emitter;
            let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": task_id_clone, "text": format!("{}\n", msg)}));
        };

        emit_term("\n=======================================");
        emit_term(&format!("[ENGINE] 🚀 Starting Image Extraction Pipeline for Task: {}", task_id));
        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3.5 (2B) Vision Model...");

        // 🌟 [CRITICAL FIX 1] 이미지 추출 5단계를 완벽하게 맞추기 위한 로딩 스텝(2단계) UI 추가!
        let payload_load = json!({ "task_id": task_id.clone(), "category": "Loading Model", "summary": "Initializing Vision Core...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload_load);
        crate::utils::logger::log_task_progress(app_handle, &task_id, &payload_load);

        self.ensure_qwen3_5(true).await?; 

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            
            let is_trade_doc = search_mode == "shipping";
            let mut extracted_data = json!({});

            if is_trade_doc {
                emit_term("[STAGE-2] 🚢 Trade Document Mode: Initiating Classification...");
                
                // Step A: 문서 종류 1차 판별 (768px 축소 썸네일 사용)
                let class_img = dynamic_image.resize(768, 768, image::imageops::FilterType::Triangle);
                let class_prompt = crate::parsing::get_trade_doc_classification_prompt(); // (이 프롬프트 안에 TRACKING 추가됨)
                let type_res = self.chat_with_qwen3_5_image_spinner(
                    "You are a document classifier.", &class_prompt, Some(class_img), app_handle, "extraction-progress", 
                    json!({ "category": "Vision (Step 1/2)", "summary": "Identifying document type..." }), 128, cancel_token.clone(), Some(task_id.clone()), None
                ).await?;
                
                let detected_type = crate::parsing::parse_json_from_llm(&type_res)
                    .get("doc_type").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                emit_term(&format!("✅ Document identified as: **{}**", detected_type));

                // 🌟 [개선된 분기 포인트] TRACKING(운송장)으로 판별되면 무거운 Slice & Merge를 우회합니다!
                if detected_type == "TRACKING" {
                    emit_term("[STAGE-2] 📦 Fast-Tracking Parcel Label...");
                    
                    let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                    let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language); // 🌟 Bias 호출
                    let result_str = self.chat_with_qwen3_5_image_spinner(
                        "You are a highly precise logistics data extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                        json!({ "category": "Vision Analysis", "summary": "Extracting Tracking Label data..." }), 512, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
                    ).await?;
                    
                    extracted_data = crate::parsing::parse_json_from_llm(&result_str);
                    
                    // DB 저장 시 에러가 나지 않도록 doc_type 꼬리표를 강제로 달아줍니다.
                    if let Some(obj) = extracted_data.as_object_mut() {
                        obj.insert("doc_type".to_string(), json!("TRACKING"));
                    }
                    
                } else {
                    // 🌟 B/L, CI 등 밀도 높은 무역 문서일 경우 기존처럼 Slice & Merge 파이프라인을 탑니다.
                    emit_term("[STAGE-2] 🚢 Initiating Slice & Merge Pipeline...");
                    
                    // Step B: 판별된 문서에 따른 자르기(Slice) 미션 설정
                    let missions = match detected_type.as_str() {
                        "CI" | "PI" => vec![("header", 0.0, 0.20), ("parties", 0.0, 0.40), ("logistics", 0.20, 0.50), ("items", 0.30, 0.85), ("financials", 0.70, 0.95), ("conditions", 0.80, 1.0)],
                        "BL" => vec![("header", 0.0, 0.20), ("parties", 0.0, 0.60), ("logistics", 0.35, 0.65), ("cargo", 0.50, 0.90), ("conditions", 0.80, 1.0)],
                        "AWB" => vec![("header", 0.0, 0.15), ("parties", 0.0, 0.40), ("logistics", 0.10, 0.40), ("cargo", 0.30, 0.70), ("financials", 0.60, 0.90)],
                        _ => vec![("header", 0.0, 0.30), ("parties", 0.0, 0.50), ("items", 0.30, 0.80), ("conditions", 0.70, 1.0)],
                    };

                    let w = dynamic_image.width();
                    let h = dynamic_image.height();
                    let mut final_data_map = serde_json::Map::new();
                    
                    // 🌟 [CRITICAL FIX] Python 패리티: 병합을 위한 7대 기본 뼈대(Skeleton)를 무조건 미리 생성해야 합니다!
                    final_data_map.insert("header".to_string(), json!({"doc_type": detected_type}));
                    final_data_map.insert("parties".to_string(), json!({}));
                    final_data_map.insert("logistics".to_string(), json!({}));
                    final_data_map.insert("conditions".to_string(), json!({}));
                    final_data_map.insert("financials".to_string(), json!({}));
                    final_data_map.insert("cargo".to_string(), json!({}));
                    final_data_map.insert("line_items".to_string(), json!([]));
                    final_data_map.insert("containers".to_string(), json!([]));

                    // Step C: 구역별 분할 크롭 및 LLM 타격
                    for (idx, (cat, top, bot)) in missions.iter().enumerate() {
                        if cancel_token.as_ref().map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed)) {
                            emit_term("🛑 Task cancelled by user. Terminating safely.");
                            return Ok(());
                        }
                        
                        let crop_y = (h as f32 * top) as u32;
                        let crop_h = (h as f32 * (bot - top)) as u32;
                        let img_slice = dynamic_image.crop_imm(0, crop_y, w, crop_h);
                        
                        let prompt = crate::parsing::get_trade_category_schema(cat, &detected_type);
                        let summary_msg = format!("Scanning {} ({}%)...", cat.to_uppercase(), (bot * 100.0) as i32);
                        
                        let tile_res = self.chat_with_qwen3_5_image_spinner(
                            "You are a highly precise document data extraction assistant.", &prompt, Some(img_slice), app_handle, "extraction-progress", 
                            json!({ "category": format!("Vision (Slice {}/{})", idx+1, missions.len()), "summary": summary_msg }), 1024, cancel_token.clone(), Some(task_id.clone()), None
                        ).await?;

                        let tile_json = crate::parsing::parse_json_from_llm(&tile_res);
                        
                        // 🌟 기존 병합 함수 호출 (이제 뼈대가 있으므로 정상적으로 채워집니다)
                        merge_json_manual(&mut final_data_map, cat, tile_json);
                    }
                    
                    extracted_data = Value::Object(final_data_map);
                }

            } else {
                // ============================================================
                // 🛒 [Commerce 모드] 커머스 라우팅 보완
                // ============================================================
                emit_term("[STAGE-2] 🛒 Commerce Mode: Analyzing Product/Label...");
                
                // 🌟 [개선] 기존에 무조건 "goods"(상품) 프롬프트를 먹이던 것을, 
                // 택배 운송장이 올라올 확률이 높으므로 바코드/송장 번호를 우선 추출하는 "tracking" 기반의 
                // 범용 커머스 프롬프트로 처리하도록 변경했습니다.
                let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language); // 🌟 Bias 호출
                
                let result_str = self.chat_with_qwen3_5_image_spinner(
                    "You are a precise commerce and logistics extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                    json!({ "category": "Vision Analysis", "summary": "Analyzing commerce tracking/goods..." }), 1024, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
                ).await?;
                
                extracted_data = crate::parsing::parse_json_from_llm(&result_str);
            }
            
            let mode_name = if is_trade_doc { "Trade Document" } else { "Commerce" };
            emit_term(&format!("[STAGE-2] Generating vision insights for {} mode...", mode_name));

            emit_term("\n=======================================");
            emit_term(&format!("[DEBUG-VISION] 🤖 AI Raw Response Extracted."));
            emit_term("=======================================\n");

            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            
            // [PRIVACY] 무역 문서(BL, CI 등) 및 송장(Tracking)은 개인정보 밀집 구역이므로 반드시 마스킹을 적용합니다.
            // 커머스 상품(goods) 이미지인 경우에만 예외적으로 우회합니다.
            let doc_type = if is_trade_doc { 
                extracted_data.get("header")
                    .and_then(|h| h.get("doc_type"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("shipping_doc") 
            } else { 
                "goods" 
            };
            
            let masked_nl = nl.clone(); // 마스킹은 백엔드 push_data 단계에서 동적으로 수행됩니다.

            let item_digest = crate::utils::hash::digest(&nl);

            // 🌟 [VISION-JIT] 비전 추론이 모두 끝났습니다. 이어지는 임베딩/DB 동기화 단계가
            //    VRAM 을 쓸 수 있도록 mmproj 가중치를 여기서 즉시 반환합니다.
            //    (2B 텍스트 모델 본체는 그대로 상주하므로 재로딩 비용은 0 입니다)
            {
                let mut q35_guard = self.qwen3_5_generator.lock().await;
                if let Some(gen) = q35_guard.as_mut() {
                    if gen.vision_capable() && gen.is_vision_jit_capable() && gen.vision_resident() {
                        let _ = gen.set_vision_active(false);
                        emit_term("[VISION-JIT] Vision pipeline complete. mmproj weights released before embedding stage.");
                    }
                }
            }

            emit_term("[STAGE-3] Syncing extracted data to LanceDB...");

            // 🌟 [CRITICAL FIX 2] 5단계 마무리를 위한 저장 스텝(4단계) UI 추가!
            let payload_save = json!({ "task_id": task_id.clone(), "category": "Saving", "summary": "Syncing to database...", "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload_save);
            crate::utils::logger::log_task_progress(app_handle, &task_id, &payload_save);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr); 
                let hashed_cc = crate::utils::hash::hash_id(if is_trade_doc { "local.shipping" } else { "local.commerce" });

                // 식별자(ID) 추출 기준 분기
                let raw_no = if is_trade_doc {
                    extracted_data.get("document_number").and_then(|s| s.as_str()).unwrap_or(&task_id)
                } else {
                    extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id)
                };
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(raw_no).replace("-", "").replace("_", "");
                
                // 🌟 [CRITICAL FIX] 프론트엔드 리스트(#doc-list)와 완벽 동기화하기 위해 "items" 테이블로 저장 위치를 강제 통합합니다!
                let table_name = "items"; 

                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, clean_no)));
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));

                let mut final_data = if extracted_data.is_object() { extracted_data.clone() } else { json!({ "raw_output": extracted_data }) };
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));
                // 🌟 [CRITICAL FIX] 이미지 추출 결과에도 모드 필터를 위한 mode 값을 명시적으로 주입합니다.
                final_data.as_object_mut().unwrap().insert("mode".to_string(), json!(search_mode.clone()));
                final_data.as_object_mut().unwrap().insert("text".to_string(), json!(nl));
                final_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_nl));

                // 🌟 [추가 보완] 무역 문서(Trade Doc)일 경우 Python처럼 핵심 컬럼 평탄화 (Flattening)
                if is_trade_doc {
                    let obj = final_data.as_object_mut().unwrap();
                    
                    // Header에서 날짜/문서번호 추출
                    if let Some(header) = extracted_data.get("header") {
                        obj.insert("issue_date".to_string(), header.get("issue_date").cloned().unwrap_or(json!("")));
                        obj.insert("no".to_string(), header.get("document_number").cloned().unwrap_or(json!("")));
                    }
                    // Parties에서 화주/수하인 추출
                    if let Some(parties) = extracted_data.get("parties") {
                        obj.insert("sender_name".to_string(), parties.get("supplier_name").cloned().unwrap_or(json!("")));
                        obj.insert("recipient_name".to_string(), parties.get("buyer_name").cloned().unwrap_or(json!("")));
                    }
                    // Logistics에서 선박/항구 추출
                    if let Some(logistics) = extracted_data.get("logistics") {
                        obj.insert("vessel".to_string(), logistics.get("vehicle_name").cloned().unwrap_or(json!("")));
                        obj.insert("pol".to_string(), logistics.get("location_port_of_loading").cloned().unwrap_or(json!("")));
                        obj.insert("pod".to_string(), logistics.get("location_port_of_discharge").cloned().unwrap_or(json!("")));
                    }
                    // Financials/Conditions 추출
                    if let Some(fin) = extracted_data.get("financials") {
                        obj.insert("amount".to_string(), fin.get("amount_total").cloned().unwrap_or(json!(0)));
                    }
                    if let Some(cond) = extracted_data.get("conditions") {
                        obj.insert("incoterms".to_string(), cond.get("incoterms_code").cloned().unwrap_or(json!("")));
                    }
                }
                
                let _ = db.upsert_item(
                    table_name, // 분기된 테이블 적용
                    &hashed_id, 
                    doc_type, 
                    final_data, 
                    None,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, hashed_cc))),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
                
                // 🌟 [CRITICAL FIX] 이미지 데이터 저장 직후, DB의 Task와 Message 상태도 9(DONE)로 완전히 굳혀버립니다!
                // 이 두 줄이 없어서 3초마다 UI가 이전 상태(1)를 DB에서 퍼와 덮어씌우고 있었습니다.
                let _ = db.update_task_status(&task_id, 9).await;
                let _ = db.update_message_status(&task_id, 9, Some("Extraction Complete")).await;
            }
            
            emit_term("[SUCCESS] Task Completed. Data saved.");
            
            let payload = json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            });
            
            // 🌟 [CRITICAL FIX] Done 상태를 파일에도 확실히 기록하여 상세페이지 복구 시 100% 출력되게 합니다!
            crate::utils::logger::log_task_progress(app_handle, &task_id, &payload);
            
            crate::utils::sync_utils::notify_new_task();
            
            Ok(())
        } else {
            Ok(())
        }
    }
    
    pub async fn chat_with_qwen3_5_image_spinner(
        &self, 
        system: &str,       
        user_input: &str,   
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        max_tokens: usize,
        cancellation_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        semantic_prejudice: Option<&str>   // 🌟 추가
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC] 🌟 target_size 로직 삭제하고 바로 bool 전달
        self.ensure_qwen3_5(image.is_some()).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(_app_handle, task_id, &base_payload); // 기존 변수명이 app_handle이면 app_handle로 사용
        }
        
        // 🌟 [CRITICAL FIX] 화면에 실시간 진행률(퍼센트)을 쏘아 보내는 코드를 복구합니다!
        let _ = _app_handle.emit(_event_name, &base_payload); // 기존 변수명이 app_handle이면 app_handle, _event_name이면 _event_name 사용
        
        let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
        let gen = q35_gen_guard.as_mut().ok_or_else(|| anyhow!("Qwen 3.5 Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        // User Text 할당
        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        // System 메시지 명시적 생성
        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        // User 메시지 명시적 생성
        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        // 파라미터 세팅
        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen3.5".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(
            params, 
            cancellation_token.clone(),
            session_id, // 🌟 SSD 저장 및 병합 캐시 활성화!
            Some("inference".to_string()),
            None, // 🌟 5번째 인자인 ignore_list 자리에 None을 명시적으로 추가합니다.
            semantic_prejudice  // 🌟 변경
        ).await.map_err(|e| anyhow!("Qwen 3.5 Inference failed: {}", e))
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [FIX] Default to Qwen (0.6B) for all chat tasks
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // 🌟 [VISION-JIT] chat 은 ChatCompletionRequestMessageContentPart::Text 만 조립하는
        //    순수 텍스트 경로입니다. 비전 가중치가 붙어 있다면 여기서 반환합니다.
        {
            let mut gen_guard = self.generator.lock().await;
            if let Some(gen) = gen_guard.as_mut() {
                if gen.is_vision_jit_capable() && gen.vision_resident() {
                    let _ = gen.set_vision_active(false);
                }
            }
        }
        
        let _self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        {
            let mut gen_guard = self.generator.lock().await;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_text,
                name: None,
            });

            let content_parts = vec![
                ChatCompletionRequestMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text }
                )
            ];

            let user_message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
                model: "qwen".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }
    }

    pub async fn chat_with_spinner(
        &self, 
        system: &str, 
        user_input: &str,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let content_parts = vec![
            ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
            )
        ];

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id, kv_name).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [FIX] Ensure we stay on Qwen (0.6B).
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [FIX] Removed periodic UI emits from low-level model calls.
        // Higher-level scheduler will manage the initial and final UI states.
        // let _ = app_handle.emit(event_name, &base_payload);
        
        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(app_handle, task_id, &base_payload);
        }

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn chat_with_image_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        _base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Removed redundant emit. Only log the progress if needed.
        // let _ = app_handle.emit(event_name, base_payload);

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?; // 🌟 무조건 Qwen으로 로드
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        system: &str,       // 🌟 추가
        user_input: &str,   // 🌟 변경
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(_app_handle, task_id, &base_payload);
        }

        let max_tok = self.max_tokens_limit;
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tok),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = crate::prompts::get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            "You are a highly precise document data extraction assistant.", // 🌟 System 주입
            &prompt,                                                        // 🌟 User 주입
            Some(master_img),
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        // 🌟 1. 메모리 캐시부터 확인합니다 (중복된 텍스트면 GPU 연산 원천 차단)
        {
            let cache = self.embedding_cache.lock().await;
            if let Some(vector) = cache.get(&text) {
                return Ok(vector.clone());
            }
        }

        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;

        let embedding_model_arc = self.embedding_model.clone();
        let text_clone = text.clone();
        
        let vector = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text_clone).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 384])
            }
        }).await??;

        // 🌟 2. 새로 연산된 벡터를 해시맵 캐시에 저장하여 다음 루프 때 재사용합니다.
        {
            let mut cache = self.embedding_cache.lock().await;
            cache.insert(text, vector.clone());
        }

        Ok(vector)
    }

    pub async fn get_embedding_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.ensure_embedding().await?;
        let embedding_model_arc = self.embedding_model.clone();
        
        let vectors = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed_batch(&texts).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                Ok(vec![vec![0.0; 384]; texts.len()])
            }
        }).await??;

        Ok(vectors)
    }

    // [신규] Commerce 파이프라인: 2-Stage (0.6B para2graph -> 2B graph2contexts)
    // [신규] Commerce 파이프라인: 2-Stage (2B 단일 모델 연속 처리)
    pub async fn parse_commerce_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, metrics_json: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        use tauri::Emitter;

        // 🌟 [신규] 터미널 로거 헬퍼 주입
        let emit_term = |msg: &str| {
            println!("{}", msg);
            let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
        };

        emit_term("[ENGINE] 🚀 Starting Commerce Search Pipeline...");

        // 🌟 [최초 초기화] VRAM 확보 및 불필요한 제너레이터 선제적 언로드 (캔슬 개입 포함)
        emit_term("[ENGINE] 🧹 Pre-purging memory before loading embedding model...");
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }
        
        // 🌟 [VRAM 누수 픽스] KV 캐시를 정상적으로 삭제하기 위해 None 덮어쓰기 로직을 제거하고, deep_purge_resources에 전부 일임합니다.
        self.deep_purge_resources().await;
        self.wait_for_vram_settle(1200, 5, Some(cancel_token.clone())).await.ok();

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }
        
        // ----------------------------------------------------
        // Stage 1: 세그먼트 분할 (Vector Cliff Detection) - Embedding 모델 사용
        // ----------------------------------------------------
        emit_term("[STAGE-1] Loading Models (Embedding & Qwen3) for Commerce Pipeline...");
        let payload = json!({ "task_id": task_id, "category": "Stage 1", "summary": "Segmenting semantic intents...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        // 🌟 [최적화] 파이프라인 중간에 모델을 교체하며 발생하는 Ping-Pong 로드를 방지하기 위해, 최초에 Qwen3와 Embedding 모델을 한 번에 모두 로드합니다.
        self.ensure_qwen3().await?;
        self.ensure_embedding().await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        // 🌟 [CRITICAL FIX] whatlang을 이용한 글로벌 언어 감지 로직 적용
        // 감지된 언어가 신뢰할만하면 프로젝트 내부 ISO-639 코드 체계로 변환, 그 외의 경우 UI의 language로 폴백
        let query_lang = whatlang::detect(&query)
            .map(|info| match info.lang() {
                whatlang::Lang::Kor => "ko".to_string(),
                whatlang::Lang::Jpn => "ja".to_string(),
                whatlang::Lang::Cmn => "zh-hans".to_string(),
                whatlang::Lang::Rus => "ru".to_string(),
                whatlang::Lang::Ara => "ar".to_string(),
                whatlang::Lang::Tha => "th".to_string(),
                whatlang::Lang::Ell => "el".to_string(),
                whatlang::Lang::Heb => "he".to_string(),
                whatlang::Lang::Hin => "hi".to_string(),
                whatlang::Lang::Ben => "bn".to_string(),
                whatlang::Lang::Tel => "te".to_string(),
                whatlang::Lang::Khm => "km".to_string(),
                whatlang::Lang::Eng => "en".to_string(),
                whatlang::Lang::Fra => "fr".to_string(),
                whatlang::Lang::Deu => "de".to_string(),
                whatlang::Lang::Spa => "es".to_string(),
                whatlang::Lang::Ita => "it".to_string(),
                whatlang::Lang::Por => "pt".to_string(),
                whatlang::Lang::Nld => "nl".to_string(),
                whatlang::Lang::Vie => "vi".to_string(),
                _ => language.to_string(),
            })
            .unwrap_or_else(|| language.to_string());

        let intent_anchors = vec![
            ("order", "measure sales performance or direct transactions, conversion rate, sales volume, checkout, payment, cancellation, refund, purchase, buy"),
            ("goods", "product catalog data, exposure, traffic metrics, page views, clicks, physical attributes, stock limits, unit prices, items, find, search clothes"),
            ("tracking", "manage logistics and fulfillment, shipment status, dispatch, delivery duration, courier information, tracking number, parcel"),
            ("review", "analyze the voice of the customer, feedback, ratings, reviews, CS messages, complaints, bad quality, good product"),
            ("coupon", "manage specific discount vouchers, coupon codes, issuance limits, discount amounts applied via coupons, promotion code"),
            ("event", "manage marketing campaigns, analyze broad operational trends, promotions, exhibitions, seasonal sales"),
            ("ignore", "ignore, system prompt, stop, cancel, do nothing, irrelevant noise"),
        ];

        let categories = ["order", "goods", "tracking", "review", "coupon", "event", ""];
        let mut layout_embs = std::collections::HashMap::new();
        let mut anchor_embs = std::collections::HashMap::new();

        let mut texts_to_embed = Vec::new();
        let mut emb_mappings = Vec::new();

        // 🌟 intent_anchors 임베딩 수집 추가
        for (cat, text) in &intent_anchors {
            texts_to_embed.push(text.to_string());
            emb_mappings.push((cat.to_string(), "anchor".to_string()));
        }

        // 🌟 [변경] Stage 1 멀티패스 전용 함수인 get_multi_pass_contexts를 호출하여
        // layout_list, layout_form 및 core_intent를 포함한 100% 모든 속성을 수집합니다.
        for cat in &categories {
            let contexts = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
            
            for (key, bias, prejudice) in contexts {
                texts_to_embed.push(bias);
                emb_mappings.push((cat.to_string(), format!("{}_bias", key)));

                let final_prej = if prejudice.trim().is_empty() { "random unrelated noise".to_string() } else { prejudice };
                texts_to_embed.push(final_prej);
                emb_mappings.push((cat.to_string(), format!("{}_prejudice", key)));
            }
        }

        // 2. 단 한 번의 배치 호출로 모든 임베딩 벡터를 한 장바구니에 획득
        let embedded_texts = self.get_embedding_batch(texts_to_embed).await.unwrap_or_else(|_| vec![vec![0.0; 384]; emb_mappings.len()]);

        // 3. 획득한 벡터와 카테고리/키 값을 매칭하여 해시맵에 일괄 삽입
        for (i, (cat, emb_type)) in emb_mappings.into_iter().enumerate() {
            if emb_type == "anchor" {
                anchor_embs.insert(cat, embedded_texts[i].clone());
            } else {
                layout_embs.insert(format!("{}_{}", cat, emb_type), embedded_texts[i].clone());
            }
        }

        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
        }

        // 🌟 [추가] 서술어구(verb_expression) 타이브레이커 가이드 벡터 생성
        let mut prefixed_verb_b_vals = Vec::new();
        for lang in [query_lang.as_str(), "english"] {
            let verb_val = crate::parsing::BIAS_DICT.get("verb").and_then(|v| v.get("bias")).and_then(|v| v.get(lang)).and_then(|v| v.as_str()).unwrap_or("verb, predicate");
            let expr_val = crate::parsing::BIAS_DICT.get("expression").and_then(|v| v.get("bias")).and_then(|v| v.get(lang)).and_then(|v| v.as_str()).unwrap_or("idiom, phrase");
            let combined_verb_expr = format!("{}, {}", verb_val, expr_val);
            let prefixed = combined_verb_expr.split(',').map(|s| format!("{} {}", lang, s.trim())).collect::<Vec<_>>().join(", ");
            prefixed_verb_b_vals.push(prefixed);
        }
        let combined_verb_b_val = prefixed_verb_b_vals.join(", ");
        let verb_emb = self.get_embedding(combined_verb_b_val).await.unwrap_or_else(|_| vec![0.0; 384]);

        // 🌟 [OPERATOR PHRASE BANK] 비교 대상도 구 단위로 맞춰야 공정한 비교가 됩니다.
        //    (로그: '이하로' ActionSim 0.7700 > OpSim 0.5746 → 검색 명령어로 오인 →
        //     '5000원' 과 분리되어 sale_price lte 5000 이 통째로 소멸)
        let mut op_bank_texts: Vec<String> = Vec::new();
        if let Some(ops) = crate::parsing::BIAS_DICT.get("operators").and_then(|v| v.as_object()) {
            for (_, v) in ops {
                for field in ["semantic", "bias"] {
                    if let Some(b) = v.get(field).and_then(|val| val.as_str()) {
                        for p in crate::utils::ai_utils::split_bias_phrases_full(b) {
                            if !op_bank_texts.iter().any(|e| e == &p) { op_bank_texts.push(p); }
                        }
                    }
                }
            }
        }
        let operator_embs: Vec<Vec<f32>> = if op_bank_texts.is_empty() {
            Vec::new()
        } else {
            self.get_embedding_batch(op_bank_texts.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; op_bank_texts.len()])
        };

        // 🌟 [추가] Stanza 기반 형태소 분석으로 검색어(query) 정밀 분할 반영
        let mut ext_words_string: Vec<String> = Vec::new();
        let mut stanza_lemmas: Option<Vec<String>> = None;
        let mut stanza_deprels: Option<Vec<String>> = None;
        
        let stanza_lang_code = match query_lang.as_str() {
            "korean" | "ko" => "ko",
            "english" | "en" => "en",
            "japanese" | "ja" => "ja",
            "chinese" | "zh" | "zh-tw" | "zh-hk" => "zh-hans",
            "french" | "fr" => "fr",
            "german" | "de" => "de",
            "spanish" | "es" => "es",
            "italian" | "it" => "it",
            "portuguese" | "pt" => "pt",
            "dutch" | "nl" => "nl",
            "russian" | "ru" => "ru",
            "arabic" | "ar" => "ar",
            "thai" | "th" => "th",
            "hindi" | "hi" => "hi",
            "bengali" | "bn" => "bn",
            "greek" | "el" => "el",
            "hebrew" | "he" => "he",
            "vietnamese" | "vi" => "vi",
            _ => "en",
        };

        let stanza_base_dir = crate::utils::get_app_dir().join("models").join("stanza");
        let stanza_lang_dir = stanza_base_dir.join(stanza_lang_code);

        if stanza_lang_dir.exists() {
            emit_term(&format!("[STANZA] 🧠 Loading Stanza ONNX models for Search Query ('{}')...", stanza_lang_code));
            
            struct UnsafePipelineWrapper(crate::stanza::StanzaPipeline);
            unsafe impl Send for UnsafePipelineWrapper {}
            
            let base_dir_clone = stanza_base_dir.clone();
            let lang_code_clone = stanza_lang_code.to_string();
            
            // StanzaPipeline::new는 async 함수이므로 await를 호출하여 결과를 기다려야 합니다.
            // 불필요한 OS 스레드 생성(std::thread::spawn) 및 채널을 제거하고, 현재의 비동기 런타임에서 직접 처리합니다.
            let pipeline_res = crate::stanza::StanzaPipeline::new(base_dir_clone, &lang_code_clone)
                .await
                .map(UnsafePipelineWrapper);

            match pipeline_res {
                Ok(wrapper) => {
                    let mut stanza = wrapper.0;
                    let chars: Vec<char> = query.chars().collect();
                    
                    let whitespace_split: Vec<String> = query.split_whitespace().map(|s| s.to_string()).collect();
                    let mut stanza_split: Vec<String> = Vec::new();

                    // 🌟 [CRITICAL FIX] "베이지 가디건" (공백 트랙)과 "베이", "지" (형태소 트랙)을 모두 살리는 투트랙(Dual-Track) 전략!
                    // 두 가지 방식으로 자른 단어들을 하나의 배열로 합쳐서 Plinko 윈도우에 던지면, 
                    // NMS 배틀이 알아서 문맥(Context Score)이 더 높은 진짜 덩어리를 승자로 채택하게 됩니다.
                    
                    if !chars.is_empty() {
                        let seq_len = chars.len();
                        let mut char_ids = Vec::with_capacity(seq_len);
                        for c in &chars {
                            let id = if !stanza.preprocessor.tok_char_vocab.is_empty() {
                                *stanza.preprocessor.tok_char_vocab.get(c).unwrap_or(&stanza.preprocessor.tok_char_unk_id)
                            } else {
                                *stanza.preprocessor.char_vocab.get(c).unwrap_or(&stanza.preprocessor.char_unk_id)
                            };
                            char_ids.push(id);
                        }
                        
                        if let Ok(char_tensor) = ndarray::Array2::from_shape_vec((1, seq_len), char_ids) {
                            // 🌟 [CRITICAL FIX] 모델이 요구하는 feature_dim을 직접 읽어와서 동적 생성 (한국어는 0)
                            let mut feature_dim = 0;
                            for input_meta in &stanza.tokenize_session.inputs {
                                if input_meta.name == "f" || input_meta.name == "char_features" {
                                    if let Some(&Some(d)) = input_meta.dimensions.get(2) {
                                        feature_dim = d as usize;
                                    }
                                }
                            }
                            // 🌟 [디버깅 & 픽스] feature_dim이 0일 경우 빈 텐서가 생성되어 ONNX Runtime에서 Shape 에러가 발생할 수 있으므로 최소 1 이상의 더미 차원을 부여합니다.
                            if feature_dim == 0 {
                                feature_dim = 32;
                            }
                            
                            let char_features = ndarray::Array3::<i64>::zeros((1, seq_len, feature_dim));
                            let seq_lengths = ndarray::Array1::<i64>::from_vec(vec![seq_len as i64]);
                            
                            let mut tensor_pool = std::collections::HashMap::new();
                            tensor_pool.insert("x", char_tensor.clone().into_dyn());
                            tensor_pool.insert("f", char_features.clone().into_dyn());
                            tensor_pool.insert("char_tensor", char_tensor.into_dyn());
                            tensor_pool.insert("char_features", char_features.into_dyn());
                            tensor_pool.insert("seq_lengths", seq_lengths.clone().into_dyn());
                            tensor_pool.insert("l", seq_lengths.into_dyn()); // 🌟 Tokenizer 입력 'l' 추가
                            
                            use onnxruntime::mixed::{DynInput, SessionMixedExt};

                            // 🌟 [CRITICAL FIX] E0597 & E0502 해결:
                            // tokenize_session.inputs 와 outputs 에 대한 불변 참조(immutable borrow)를 
                            // run_mixed 의 가변 참조(mutable borrow)와 분리하기 위해 
                            // 문자열을 별도의 로컬 캐시(Vec<String>)로 복제하여 라이프타임을 독립시킵니다.
                            let input_names_cache: Vec<String> = stanza.tokenize_session.inputs.iter().map(|i| i.name.clone()).collect();
                            let mut mixed_inputs = Vec::new();
                            
                            for exact_name in &input_names_cache {
                                if let Some(tensor) = tensor_pool.get(exact_name.as_str()) {
                                    // 🌟 [핵심] f32를 요구하는 피처 텐서와 i64를 요구하는 문자 텐서를 구분하여 다이나믹 타입으로 묶어버립니다.
                                    if exact_name == "f" || exact_name == "char_features" {
                                        mixed_inputs.push((exact_name.as_str(), DynInput::F32(tensor.mapv(|x| x as f32))));
                                    } else {
                                        mixed_inputs.push((exact_name.as_str(), DynInput::I64(tensor.clone())));
                                    }
                                } else {
                                    emit_term(&format!("[STANZA-WARN] Tokenizer 모델에 정의되지 않은 입력 생략: {}", exact_name));
                                }
                            }
                            
                            macro_rules! process_tok_outputs {
                                ($outputs:expr) => {
                                    let output_tensor = &$outputs[0];
                                    let shape = output_tensor.shape();
                                    let num_classes = *shape.last().unwrap() as usize;
                                    let is_3d = shape.len() == 3;
                                    
                                    let mut current_word = String::new();
                                    for i in 0..seq_len {
                                        current_word.push(chars[i]);
                                        
                                        let mut max_val = std::f32::MIN;
                                        let mut max_idx = 0;
                                        for c_idx in 0..num_classes {
                                            let val = if is_3d { output_tensor[[0, i, c_idx]] } else { output_tensor[[i, c_idx]] };
                                            if val > max_val { max_val = val; max_idx = c_idx; }
                                        }
                                        
                                        if max_idx > 0 || i == seq_len - 1 {
                                            let token_str = current_word.trim().to_string();
                                            if !token_str.is_empty() {
                                                stanza_split.push(token_str);
                                            }
                                            current_word.clear();
                                        }
                                    }
                                }
                            }

                            // 🌟 [문제 해결] 핑퐁 로직을 완전히 파기하고, 자체 구현한 확장 메서드(run_mixed)를 통해 혼합 타입을 C API 직통으로 발사합니다!
                            let out_names_cache: Vec<String> = stanza.tokenize_session.outputs.iter().map(|o| o.name.clone()).collect();
                            let out_names: Vec<&str> = out_names_cache.iter().map(|s| s.as_str()).collect();
                            match stanza.tokenize_session.run_mixed(mixed_inputs, out_names) {
                                Ok(outputs) => {
                                    emit_term("[STANZA] ✅ Tokenizer ONNX 혼합 타입(Mixed) 추론 100% 성공!");
                                    process_tok_outputs!(outputs);
                                },
                                Err(e) => {
                                    emit_term(&format!("  ⚠️ [STANZA-WARN] Tokenizer ONNX 혼합 타입 실행 실패: {:?}", e));
                                }
                            }
                        }
                    }

                    // 🌟 단일 트랙 병합 로직 (Single-Track Merge)
                    // Dual-Track 배열 이어붙이기는 NMS Battle에서 인덱스 좌표계를 파괴하여(1번 트랙과 2번 트랙이 겹치지 않는 별개의 문장으로 인식됨)
                    // 문장이 두 번 반복되거나 "이", "벤트로" 같은 파편이 결과에 중복 결합되는 치명적 버그를 유발합니다.
                    // 임베딩 모델이 자체 서브워드 토크나이저를 내장하고 있으므로, 어절(공백) 단위 분할인 whitespace_split을 기준으로 
                    // Stanza의 POS/Lemma 필터링만 적용하는 것이 가장 정확합니다.
                    if stanza_split.is_empty() {
                        ext_words_string = whitespace_split;
                    } else if whitespace_split == stanza_split {
                        ext_words_string = whitespace_split;
                        emit_term("  💡 [STANZA-INFO] 공백 분할과 Tokenizer 분할 결과가 동일하여 단일 트랙으로 진행합니다.");
                    } else {
                        emit_term("  💡 [STANZA-INFO] Stanza 토크나이저의 과잉 분할(예: 이벤트로 -> 이+벤트로) 방지 및 NMS 좌표계 보호를 위해 공백 분할(Whitespace)을 메인 트랙으로 사용합니다.");
                        ext_words_string = whitespace_split;
                    }

                    // 🌟 [STANZA POS 사전 필터링 & 로그 출력]
                    // "찾아줘", "알려줘" 등 무의미한 동사(VERB), 조사(ADP) 등을 Plinko 벡터 매칭 전에 원천 차단합니다.
                    if !ext_words_string.is_empty() {
                        let ext_words_refs: Vec<&str> = ext_words_string.iter().map(|s| s.as_str()).collect();
                        let mut chunk_size = ext_words_refs.len();
                        
                        for input_meta in &stanza.pos_session.inputs {
                            let dims = &input_meta.dimensions;
                            if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
                                if let Some(&Some(fixed_seq)) = dims.get(0) {
                                    chunk_size = fixed_seq as usize;
                                }
                            }
                        }
                        if chunk_size == 0 { chunk_size = ext_words_refs.len(); }

                        let mut padded_chunk = ext_words_refs.clone();
                        let valid_len = padded_chunk.len();
                        while padded_chunk.len() < chunk_size {
                            padded_chunk.push("<pad>");
                        }

                        match stanza.preprocessor.encode_to_tensor(&padded_chunk, &stanza.pos_session, None, None) {
                            Ok(pos_inputs) => {
                                match stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(pos_inputs) {
                                    Ok(pos_outputs) => {
                                        let output_tensor = &pos_outputs[0];
                                        let shape = output_tensor.shape();
                                        let mut pos_tags = Vec::new();
                                        let mut pos_ids = Vec::new(); // 🌟 Lemma 전달용 POS ID 수집 배열 추가

                                        let num_classes = if shape.len() == 3 { shape[2] as usize } else { shape[1] as usize };
                                        for i in 0..valid_len {
                                            let mut max_val = std::f32::MIN;
                                            let mut max_idx = 0;
                                            for c in 0..num_classes {
                                                let val = if shape.len() == 3 { output_tensor[[0, i, c]] } else { output_tensor[[i, c]] };
                                                if val > max_val { max_val = val; max_idx = c; }
                                            }
                                            let tag = stanza.preprocessor.upos_vocab.get(max_idx as usize).map(|s| s.as_str()).unwrap_or("X");
                                            pos_tags.push(tag);
                                            pos_ids.push(max_idx as i64); // 🌟 산출된 POS 태그의 Index ID 보존
                                        }

                                        // 🌟 [로그 추가] 형태소 분석 결과 전체 로그 출력
                                        let mut debug_pos_log = Vec::new();
                                        let drop_tags = ["VERB", "ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON"];
                                        let mut dropped_log = Vec::new();
                                        let mut filtered_words = Vec::new();

                                        // 🌟 [수정] 한글 하드코딩 배열을 제거하고 Stanza의 lemma_session을 직접 사용하여 동적으로 커팅합니다.
                                        let mut lemma_words: Vec<String> = vec![String::new(); valid_len];
                                        if let Ok(lemma_inputs) = stanza.preprocessor.encode_to_tensor(&padded_chunk, &stanza.lemma_session, Some(&pos_ids), None) { // 🌟 수집된 pos_ids 전달
                                            if let Ok(lemma_outputs) = stanza.lemma_session.run::<'_, '_, '_, i64, f32, _>(lemma_inputs) {
                                                let output_tensor = &lemma_outputs[0];
                                                let shape = output_tensor.shape();
                                                
                                                if shape.len() == 3 || shape.len() == 4 {
                                                    let is_4d = shape.len() == 4;
                                                    let max_char_len = if is_4d { shape[2] as usize } else { shape[1] as usize };
                                                    let num_classes = if is_4d { shape[3] as usize } else { shape[2] as usize };
                                                    
                                                    for i in 0..valid_len {
                                                        let mut lemma_str = String::new();
                                                        for j in 0..max_char_len {
                                                            let mut max_val = std::f32::MIN;
                                                            let mut max_idx = 0;
                                                            for c in 0..num_classes {
                                                                let val = if is_4d { output_tensor[[0, i, j, c]] } else { output_tensor[[i, j, c]] };
                                                                if val > max_val { max_val = val; max_idx = c; }
                                                            }
                                                            if let Some(&ch) = stanza.preprocessor.id_to_char.get(&(max_idx as i64)) {
                                                                // 패딩이나 특수 토큰('<', '>')은 제외하고 실제 문자만 조합
                                                                if ch != '<' && ch != '>' && ch != '_' {
                                                                    lemma_str.push(ch);
                                                                }
                                                            }
                                                        }
                                                        lemma_words[i] = lemma_str.trim().to_string();
                                                    }
                                                }
                                            }
                                        }

                                        let depparse_opt = crate::utils::ai_utils::run_depparse_deprels(&stanza.preprocessor, &mut stanza.depparse_session, &padded_chunk, &pos_ids);
                                        let mut filtered_lemmas = Vec::new();
                                        let mut filtered_deprels = Vec::new();

                                        for (i, word) in ext_words_string.iter().enumerate() {
                                            let tag = pos_tags[i];
                                            let lemma = if let Some(l) = lemma_words.get(i) { l.clone() } else { String::new() };
                                            
                                            debug_pos_log.push(format!("{}(tag:{}, lemma:{})", word, tag, lemma));
                                            
                                            // 1차: 기본 품사(동사, 조사 등) 드롭
                                            if drop_tags.contains(&tag) {
                                                dropped_log.push(format!("{}({})", word, tag));
                                                continue;
                                            }

                                            let mut clean_word = word.clone();
                                            let mut is_stripped = false;

                                            // 2차: Stanza 모델에서 도출된 Lemma를 있는 그대로 활용하여 한 덩어리로 묶인 꼬리 자르기
                                            // 형태소 분석기가 실패해서 "가디건찾아줘"가 한 단어로 들어왔을 때, 
                                            // Lemma가 "찾아줘" 혹은 "찾다" 등으로 도출되면 해당 문자열을 찾아 정확하게 도려냅니다.
                                            if !lemma.is_empty() && word.ends_with(&lemma) && word.len() > lemma.len() {
                                                let new_len = word.len() - lemma.len();
                                                clean_word = word[..new_len].to_string();
                                                is_stripped = true;
                                            } else if !lemma.is_empty() && word.contains(&lemma) && word.len() > lemma.len() {
                                                if let Some(idx) = word.rfind(&lemma) {
                                                    if idx >= 3 { // 최소 1글자(UTF-8 3바이트 이상) 보장하여 명사 원형 보호
                                                        clean_word = word[..idx].to_string();
                                                        is_stripped = true;
                                                    }
                                                }
                                            }

                                            if is_stripped {
                                                dropped_log.push(format!("{}(Lemma-Stripped->{})", word, clean_word));
                                            }

                                            if !clean_word.trim().is_empty() {
                                                filtered_words.push(clean_word);
                                                filtered_lemmas.push(lemma.clone());
                                                if let Some(ref d) = depparse_opt {
                                                    if i < d.len() {
                                                        filtered_deprels.push(d[i].clone());
                                                    } else {
                                                        filtered_deprels.push(String::new());
                                                    }
                                                }
                                            }
                                        }
                                        
                                        emit_term(&format!("  🧠 [STANZA-POS-LOG] 검색어 형태소 분석 결과: {:?}", debug_pos_log));

                                        if !dropped_log.is_empty() {
                                            emit_term(&format!("  ✂️ [STANZA-SEARCH-POS] 검색어에서 무의미한 단어 사전 제거 완료: {:?}", dropped_log));
                                        }
                                        
                                        // 🌟 필터링 결과가 전부 다 날아가버리면 원본을 유지 (과잉 삭제로 인한 크래시 방어)
                                        if !filtered_words.is_empty() {
                                            ext_words_string = filtered_words;
                                            stanza_lemmas = Some(filtered_lemmas);
                                            if depparse_opt.is_some() {
                                                stanza_deprels = Some(filtered_deprels);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        emit_term(&format!("  ⚠️ [STANZA-POS-ERROR] POS session run failed: {:?}", e));
                                    }
                                }
                            },
                            Err(e) => {
                                emit_term(&format!("  ⚠️ [STANZA-POS-ERROR] POS encode_to_tensor failed: {:?}", e));
                            }
                        }
                    }
                },
                Err(e) => {
                    emit_term(&format!("[STANZA] ⚠️ Failed to load Stanza models for '{}' (상세 원인): {:?}", stanza_lang_code, e));
                }
            }
        } else {
            emit_term(&format!("[STANZA] ⚠️ Stanza model directory not found: {:?}. Falling back to whitespace splitting.", stanza_lang_dir));
        }

        if ext_words_string.is_empty() {
            ext_words_string = query.split_whitespace().map(|s| s.to_string()).collect();
        }
        // 🌟 [TRACKING NUMBER DETECTION] 검색어에서 송장 번호 패턴(6자리 이상 순수 숫자 또는 숫자-하이픈 조합)을 감지합니다.
        let mut detected_tracking_numbers: Vec<String> = Vec::new();
        for word in &ext_words_string {
            let digits_only: String = word.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits_only.len() >= 6 {
                detected_tracking_numbers.push(digits_only.clone());
                emit_term(&format!("  📦 [TRACKING DETECTED] Query contains potential tracking number: '{}'", digits_only));
            }
        }
        let words: Vec<&str> = ext_words_string.iter().map(|s| s.as_str()).collect();
        let mut context_arr = Vec::new();
        // 🌟 [1차 패스] 최소 2단어 이상(2-gram)의 교차 윈도우 스팬 및 카테고리별 기본 점수 수집
        struct SpanData {
            start: usize,
            end: usize,
            text: String,
            scores: std::collections::HashMap<String, f32>,
        }
        emit_term(&format!("  🔎 [INPUT WORDS] 분할된 단어 목록: {:?}", words));

        let mut raw_spans = Vec::new();

        for start in 0..words.len() {
            let max_end = words.len().min(start + 8);
            
            // 🌟 [단어 수 제한] start + 2 로 설정하여 단일 단어(1단어)는 배제합니다.
            for end in (start + 2)..=max_end {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                    emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                    return Ok(json!({ "context": [], "cancelled": true }));
                }
                
                let test_text = words[start..end].join(" ");
                let test_emb = self.get_embedding(test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                
                // 🌟 [추가] Verb Penalty 및 단어 길이 가중치 계산
                let word_count = end - start;
                let v_sim = cosine_similarity(&test_emb, &verb_emb);
                let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                let verb_penalty = v_sim * beta;
                let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };

                let mut scores = std::collections::HashMap::new();
                for cat in &categories {
                    let contexts = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
                    let mut field_scores = Vec::new();

                    for (key, _bias, _prejudice) in contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        
                        let bias_score = cosine_similarity(&test_emb, &bias_emb);
                        let prej_score = cosine_similarity(&test_emb, &prej_emb);

                        // 🌟 [수정] penalty_weight 및 verb_penalty를 적용한 강화된 점수 차감
                        let field_score = (bias_score - (prej_score * penalty_weight) - verb_penalty).max(0.0);
                        field_scores.push(field_score);
                    }

                    // 🌟 2. Intent Anchor 점수 합산 (해당 카테고리의 anchor가 있다면)
                    let anchor_score = if let Some(anchor_emb) = anchor_embs.get(*cat) {
                        cosine_similarity(&test_emb, anchor_emb).max(0.0)
                    } else {
                        0.0
                    };

                    // 🌟 [멀티 패스 스코어 평가]
                    field_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    
                    // 🌟 [진정한 멀티패스 반영: 동적 감쇠 누적 합산 (Decaying Sum)]
                    // 상위 N개 점수를 동적으로 순회하며 가중치를 반감(1.0, 0.5, 0.25, 0.125...)시켜 합산합니다.
                    // 무한히 더해도 최대값이 수렴하므로, 필드 개수가 많은(예: 40개) 도메인이 
                    // 잡음(Noise)을 끌어모아 점수를 뻥튀기하는 현상을 수학적으로 완벽히 차단합니다.
                    let mut multi_pass_score = 0.0;
                    let mut weight = 1.0;
                    
                    // 🌟 상위 5개까지만 유의미한 멀티패스 공명으로 인정하여 합산합니다.
                    let max_pass = field_scores.len().min(5);
                    for i in 0..max_pass {
                        multi_pass_score += field_scores[i] * weight;
                        weight *= 0.5; // 다음 순위의 필드 점수는 반영 비율을 절반으로 깎습니다.
                    }
                    
                    // 🌟 [Intent Anchors 반영] 멀티패스 스코어에 Anchor 스코어를 결합 (가중치 조절 가능, 여기선 0.5 적용)
                    multi_pass_score += anchor_score * 0.5;
                    
                    // 🌟 [단어 개수 가중치 상향] 단어가 많이 합쳐질수록 문맥이 명확해지므로 길이에 비례하여 가중치를 부여합니다.
                    // 파편이 긴 문장을 잡아먹는 현상(NMS Battle 하극상)을 완벽히 막기 위해 가산점을 단어당 15%로 대폭 상향합니다.
                    let word_count = end - start;
                    let length_weight = 1.0 + ((word_count as f32 - 2.0) * 0.15); 
                    let weighted_base_score = multi_pass_score * length_weight;
                    
                    scores.insert(cat.to_string(), weighted_base_score);
                }
                
                // 🌟 [DEBUG LOG] 1차 슬라이딩 윈도우(기초 점수) 평가 결과 출력
                let mut raw_score_log = String::new();
                let mut sorted_raw: Vec<(&String, &f32)> = scores.iter().filter(|(k, _)| !k.is_empty()).collect();
                sorted_raw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                for (cat, score) in sorted_raw.iter().take(3) {
                    raw_score_log.push_str(&format!("{}: {:.4} | ", cat, score));
                }
                emit_term(&format!("    🔍 [RAW-CHUNK] '{}' -> Top: {}", test_text, raw_score_log.trim_end_matches(" | ")));

                raw_spans.push(SpanData { start, end, text: test_text, scores });
            }
        }

        // 🌟 [신규: 서브 도메인 동적 승급 (Sub-Domain Dynamic Boost) 분석 및 개선]
        // 각 서브 도메인(review, coupon, event)의 대표 키워드(layout_list, layout_form)를 결합하여 Bias로 삼고,
        // '나머지 모든 카테고리'의 대표 키워드를 결합하여 Prejudice(차감 대상)로 삼아 쿼리와의 유사도를 검증합니다.
        let query_emb = self.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384]);
        let mut sub_domain_boosts = std::collections::HashMap::new();
        
        // 1. 모든 카테고리에 대해 layout_list, layout_form의 bias 값 추출
        let all_cats = ["order", "goods", "tracking", "review", "coupon", "event"];
        let mut cat_core_texts = std::collections::HashMap::new();
        for cat in &all_cats {
            let contexts: Vec<(String, String, String)> = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
            let mut core_text = String::new();
            for (key, bias, _prej) in contexts.into_iter() {
                if key == "layout_list" || key == "layout_form" {
                    core_text.push_str(&bias);
                    core_text.push_str(", ");
                }
            }
            cat_core_texts.insert(cat.to_string(), core_text);
        }
        // 2. review, coupon, event 카테고리에 대해 각각 bias와 (나머지 카테고리의 합인) prejudice를 계산
        let target_cats = ["review", "coupon", "event"];
        for target_cat in &target_cats {
            let core_bias = cat_core_texts.get(*target_cat).cloned().unwrap_or_default();
            
            let mut core_prej = String::new();
            for other_cat in &all_cats {
                if other_cat != target_cat {
                    if let Some(other_text) = cat_core_texts.get(*other_cat) {
                        core_prej.push_str(other_text);
                    }
                }
            }

            if !core_bias.is_empty() && !core_prej.is_empty() {
                let cat_emb = self.get_embedding(core_bias).await.unwrap_or(vec![0.0; 384]);
                let prej_emb = self.get_embedding(core_prej).await.unwrap_or(vec![0.0; 384]); // 🌟 타 도메인 전체를 prejudice로 사용
                
                let b_sim = cosine_similarity(&query_emb, &cat_emb);
                let p_sim = cosine_similarity(&query_emb, &prej_emb);
                let sim = b_sim - p_sim; // 🌟 bias 유사도에서 prejudice(타 도메인) 유사도 차감
                
                // 🌟 [CRITICAL FIX] 임계값을 0.55로 엄격하게 상향 설정하여 무관한 쿼리가 승급되지 않도록 완벽 차단합니다.
                if sim > 0.55 { 
                    sub_domain_boosts.insert(target_cat.to_string(), true);
                    emit_term(&format!("  🚀 [SUB-DOMAIN BOOST] '{}' 핵심 키워드가 쿼리와 높은 유사도(Bias: {:.4} - Prej: {:.4} = Final: {:.4})를 보여 우선순위가 상향됩니다.", target_cat, b_sim, p_sim, sim));
                }
            }
        }

        // 🌟 [2차 패스] 앞뒤 교차 문장 점수 합산을 통한 최종 컨텍스트 점수 도출
        // 🌟 [2차 패스] 앞뒤 교차 문장 점수 합산 및 임시 목록 저장
        struct EvaluatedSpan {
            start: usize,
            end: usize,
            text: String,
            best_cat: String,
            context_score: f32,
            intersecting: Vec<String>,
            base_score: f32,
        }
        let mut evaluated_spans = Vec::new();

        for i in 0..raw_spans.len() {
            let target = &raw_spans[i];
            let mut contextual_scores: Vec<(String, f32)> = Vec::new();

            for cat in &categories {
                let base_score = *target.scores.get(*cat).unwrap_or(&0.0);
                
                let mut prev_bonus = 0.0;
                let mut next_bonus = 0.0;
                
                for j in 0..raw_spans.len() {
                    if i == j { continue; }
                    let other = &raw_spans[j];
                    let o_score = *other.scores.get(*cat).unwrap_or(&0.0);
                    
                    // 앞쪽 교차 문장: 시작점이 앞서면서 현재 문장과 겹침
                    if other.start < target.start && other.end > target.start {
                        if o_score > prev_bonus { prev_bonus = o_score; }
                    }
                    // 뒤쪽 교차 문장: 끝점이 뒤서면서 현재 문장과 겹침
                    if other.end > target.end && other.start < target.end {
                        if o_score > next_bonus { next_bonus = o_score; }
                    }
                }
                
                // 중심 점수에 앞뒤 교차 점수를 50% 가중치로 합산하여 자연스러운 의미 뭉치 우선순위 상향
                let final_context_score = base_score + (prev_bonus * 0.5) + (next_bonus * 0.5);
                contextual_scores.push((cat.to_string(), final_context_score));
            }

            // 최종 합산 점수 기준 내림차순 정렬
            contextual_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut current_best_cat = contextual_scores[0].0.clone();
            let current_max_contextual_score = contextual_scores[0].1;

            // 🌟 [도메인 우선순위 정의 헬퍼]
            let get_priority = |cat: &str| -> i32 {
                if sub_domain_boosts.contains_key(cat) {
                    return 1; // 🌟 쿼리 벡터 매칭 기반 동적 승급
                }
                match cat {
                    "goods" | "order" | "tracking" => 1, // 상위(핵심) 도메인
                    _ => 2, // review, coupon, event 등 부가 도메인
                }
            };

            // 🌟 [계층적 도메인 승급 (Hierarchical Promotion)] 
            // 1등이 부가 도메인(review 등)이더라도, 핵심 도메인(goods 등)이 유효 오차 범위 내에 있다면 우선권을 부여하여 강제 승급!
            if get_priority(&current_best_cat) > 1 {
                for (cat_name, c_score) in &contextual_scores {
                    if get_priority(cat_name) == 1 && *c_score >= current_max_contextual_score - 0.20 && *c_score > 0.4 {
                        current_best_cat = cat_name.clone();
                        emit_term(&format!("    🚀 [HIERARCHY PROMOTION] 핵심 도메인 우선순위 발동! '{}' -> '{}' 로 승급 (Score: {:.4})", target.text, current_best_cat, c_score));
                        break;
                    }
                }
            }

            // 🌟 [커트라인 완전 해제] 최소한의 유사도(0.0 초과)만 있다면 모두 후보군으로 올리고, 길이와 문맥이 반영된 NMS 배틀을 통해 최강자만 살아남게 합니다.
            if current_max_contextual_score > 0.0 {
                let mut intersecting_categories: Vec<String> = Vec::new();
                let mut detailed_score_log = String::new();
                
                for (cat_name, c_score) in &contextual_scores {
                    // 유의미한 점수가 있는 모든 도메인의 점수를 기록
                    if *c_score > 0.0 {
                        detailed_score_log.push_str(&format!("{}: {:.4} | ", cat_name, c_score));
                    }
                    // 🌟 [다중 허용 확장] 오차 범위를 넓혀 서브 도메인들도 다중 태그로 편입되도록 허용 (-0.30 오차)
                    if *c_score >= current_max_contextual_score - 0.30 && *c_score > 0.10 {
                        intersecting_categories.push(cat_name.clone());
                    }
                }
                if intersecting_categories.is_empty() {
                    intersecting_categories.push(current_best_cat.clone());
                }

                // 🌟 유효 텍스트 후보군 출력 (카테고리별 상세 점수 포함)
                emit_term(&format!("  🟢 [CANDIDATE] '{}' -> Domain: {} (Context Score: {:.4})", target.text, current_best_cat, current_max_contextual_score));
                emit_term(&format!("      📊 [SCORES] {}", detailed_score_log.trim_end_matches(" | ")));

                evaluated_spans.push(EvaluatedSpan {
                    start: target.start,
                    end: target.end,
                    text: target.text.clone(),
                    best_cat: current_best_cat,
                    context_score: current_max_contextual_score,
                    intersecting: intersecting_categories,
                    base_score: 0.0, // 구조체 호환성을 위해 0.0으로 고정
                });
            }
        }

        // 🌟 [3차 패스] 오버랩(교차) 충돌 해결 (계층적 도메인 우선순위 및 길이 가중치 점수 정렬)
        let get_priority = |cat: &str| -> i32 {
            if sub_domain_boosts.contains_key(cat) {
                return 1; // 🌟 쿼리 벡터 매칭 기반 동적 승급
            }
            match cat {
                "goods" | "order" | "tracking" => 1, // 상위(핵심) 도메인
                _ => 2, // review, coupon, event 등 부가 도메인
            }
        };

        evaluated_spans.sort_by(|a, b| {
            let a_pri = get_priority(&a.best_cat);
            let b_pri = get_priority(&b.best_cat);
            
            // 1. 계층적 우선순위 (상위 도메인 승리) -> 2. 점수 -> 3. 길이
            a_pri.cmp(&b_pri)
                .then(b.context_score.partial_cmp(&a.context_score).unwrap_or(std::cmp::Ordering::Equal))
                .then(b.text.len().cmp(&a.text.len()))
        });

        let mut final_selected_spans: Vec<EvaluatedSpan> = Vec::new();

        emit_term("\n  ⚔️ [NMS BATTLE] Resolving Overlaps with Hierarchical Absorption...");

        for span in evaluated_spans {
            let mut is_overlapped = false;
            let mut winner_text = String::new();

            // 이미 승리하여 선택된 상위 점수의 스팬들과 현재 스팬이 교차하는지 검사합니다.
            for selected in &mut final_selected_spans {
                let overlaps = span.start < selected.end && span.end > selected.start;
                
                if overlaps {
                    is_overlapped = true;
                    winner_text = selected.text.clone();
                    
                    // 🌟 [계층적 흡수 & 다중 허용] 패배한 조각의 후보 카테고리(intersecting) 및 best_cat을 승자에게 병합
                    for cat in &span.intersecting {
                        if !selected.intersecting.contains(cat) {
                            selected.intersecting.push(cat.clone());
                        }
                    }
                    if !selected.intersecting.contains(&span.best_cat) {
                        selected.intersecting.push(span.best_cat.clone());
                        emit_term(&format!("    ♻️ [ABSORBED] 패배한 '{}'의 '{}' 도메인이 승자 '{}'에게 다중 태그로 병합되었습니다.", span.text, span.best_cat, winner_text));
                    }
                    break;
                }
            }

            if !is_overlapped {
                emit_term(&format!("    👑 [WINNER] '{}' -> {} (Score: {:.4}) survives.", span.text, span.best_cat, span.context_score));
                final_selected_spans.push(span);
            } else {
                emit_term(&format!("    💀 [DEFEAT] '{}' is absorbed by higher priority/score winner '{}'.", span.text, winner_text));
            }
        }

        // 프론트엔드로 보내기 위해 최종 생존한 문맥들을 원래 문장의 단어 순서대로 재정렬합니다.
        final_selected_spans.sort_by(|a, b| a.start.cmp(&b.start));

        // 🌟 [4차 패스] Gap Bridging & Score Battle (고아 단어 구출 및 양방향 흡수 대결)
        // NMS 배틀에서 탈락하여 붕 떠버린 단어(Gap)들을 양쪽 승자 문맥에 각각 붙여보고, 더 높은 멀티패스 점수를 내는 쪽이 흡수합니다.
        if !final_selected_spans.is_empty() {
            emit_term("\n  🌉 [GAP BRIDGING] Rescuing orphaned words via Score Battle...");
            
            let mut final_bounds: Vec<(usize, usize, String, f32, Vec<String>)> = final_selected_spans
                .into_iter()
                .map(|s| (s.start, s.end, s.best_cat, s.context_score, s.intersecting))
                .collect();

            // 1. 왼쪽 끝(Left Edge) 고아 단어 무조건 흡수 (예: 문장 맨 앞의 "여름")
            if final_bounds[0].0 > 0 {
                let gap_start = 0;
                let gap_end = final_bounds[0].0;
                let gap_text = words[gap_start..gap_end].join(" ");
                emit_term(&format!("    🛠️ [LEFT EDGE] '{}' is absorbed by '{}'", gap_text, words[final_bounds[0].0..final_bounds[0].1].join(" ")));
                final_bounds[0].0 = 0;
            }

            // 2. 중간(Gap) 고아 단어 양방향 점수 대결 흡수 (예: "20%에", "속하지만")
            for i in 0..(final_bounds.len() - 1) {
                let gap_start = final_bounds[i].1;
                let gap_end = final_bounds[i+1].0;

                if gap_start < gap_end {
                    let gap_text = words[gap_start..gap_end].join(" ");

                    // 대결 A: 왼쪽 승자가 흡수했을 때의 멀티패스 점수 계산
                    let left_cat = &final_bounds[i].2;
                    let left_test_text = words[final_bounds[i].0..gap_end].join(" ");
                    let left_emb = self.get_embedding(left_test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    let left_contexts = crate::parsing::get_multi_pass_contexts(left_cat, &query_lang);
                    
                    let mut left_scores = Vec::new();
                    for (key, _bias, _prej) in left_contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", left_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", left_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        left_scores.push((cosine_similarity(&left_emb, &bias_emb) - cosine_similarity(&left_emb, &prej_emb)).max(0.0));
                    }
                    left_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let mut left_score = 0.0;
                    let mut weight = 1.0;
                    for j in 0..left_scores.len().min(5) { left_score += left_scores[j] * weight; weight *= 0.5; }

                    // 대결 B: 오른쪽 승자가 흡수했을 때의 멀티패스 점수 계산
                    let right_cat = &final_bounds[i+1].2;
                    let right_test_text = words[gap_start..final_bounds[i+1].1].join(" ");
                    let right_emb = self.get_embedding(right_test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    let right_contexts = crate::parsing::get_multi_pass_contexts(right_cat, &query_lang);
                    
                    let mut right_scores = Vec::new();
                    for (key, _bias, _prej) in right_contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", right_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", right_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        right_scores.push((cosine_similarity(&right_emb, &bias_emb) - cosine_similarity(&right_emb, &prej_emb)).max(0.0));
                    }
                    right_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let mut right_score = 0.0;
                    let mut weight = 1.0;
                    for j in 0..right_scores.len().min(5) { right_score += right_scores[j] * weight; weight *= 0.5; }

                    // ⚔️ 대결 결과 판정 및 최종 흡수
                    if left_score >= right_score {
                        emit_term(&format!("    ⚔️ [GAP BATTLE] Gap '{}' -> LEFT WINS! (Left: {:.4} > Right: {:.4})", gap_text, left_score, right_score));
                        final_bounds[i].1 = gap_end; 
                        final_bounds[i].3 = left_score; // 점수 갱신
                        // 패자쪽(오른쪽)의 best_cat을 승자 쪽에 추가
                        let right_cat = final_bounds[i+1].2.clone();
                        if !final_bounds[i].4.contains(&right_cat) {
                            final_bounds[i].4.push(right_cat);
                        }
                    } else {
                        emit_term(&format!("    ⚔️ [GAP BATTLE] Gap '{}' -> RIGHT WINS! (Right: {:.4} > Left: {:.4})", gap_text, right_score, left_score));
                        final_bounds[i+1].0 = gap_start; 
                        final_bounds[i+1].3 = right_score; // 점수 갱신
                        // 패자쪽(왼쪽)의 best_cat을 승자 쪽에 추가
                        let left_cat = final_bounds[i].2.clone();
                        if !final_bounds[i+1].4.contains(&left_cat) {
                            final_bounds[i+1].4.push(left_cat);
                        }
                    }
                }
            }

            // 3. 오른쪽 끝(Right Edge) 고아 단어 무조건 흡수 (예: 문장 맨 끝의 "시급해")
            let last_idx = final_bounds.len() - 1;
            if final_bounds[last_idx].1 < words.len() {
                let gap_start = final_bounds[last_idx].1;
                let gap_end = words.len();
                let gap_text = words[gap_start..gap_end].join(" ");
                emit_term(&format!("    🛠️ [RIGHT EDGE] '{}' is absorbed by '{}'", gap_text, words[final_bounds[last_idx].0..final_bounds[last_idx].1].join(" ")));
                final_bounds[last_idx].1 = words.len();
            }

            // 4. 최종 조립된 결과를 배열에 삽입
            let mut needs_category_llm = false;
            for (_, _, _, _, intersecting) in &final_bounds {
                if intersecting.len() > 1 {
                    needs_category_llm = true;
                    break;
                }
            }
            
            if needs_category_llm {
                emit_term("    🧠 [QWEN3 VERIFICATION (STAGE-1)] Verifying domain categories...");
                // 이미 최초에 로드되었으므로 생략
            }

            for (start, end, mut best_cat, context_score, mut intersecting) in final_bounds {
                let final_text = words[start..end].join(" ");
                
                // 🌟 types 배열에 best_cat이 무조건 포함되도록 보장하고 중복 제거
                if !intersecting.contains(&best_cat) {
                    intersecting.push(best_cat.clone());
                }
                intersecting.sort();
                intersecting.dedup();
                
                // 🌟 [추가] Qwen3를 이용한 카테고리(Type) 확정 로직
                if intersecting.len() > 1 {
                    let prompt = crate::prompts::verify_category_with_alternatives_prompt(
                        &final_text, &best_cat, context_score, &intersecting
                    );
                    
                    if let Ok(response) = self.call_qwen3_verification_model(&prompt, Some(cancel_token.clone())).await {
                        if let Ok(result) = serde_json::from_str::<Value>(&response) {
                            if let Some(suggested) = result.get("suggested_category").and_then(|v| v.as_str()) {
                                if intersecting.contains(&suggested.to_string()) && best_cat != suggested {
                                    emit_term(&format!("      🔄 Category corrected/confirmed from [{}] to [{}] for '{}'", best_cat, suggested, final_text));
                                    best_cat = suggested.to_string();
                                } else {
                                    emit_term(&format!("      ✅ Category [{}] confirmed for '{}'", best_cat, final_text));
                                }
                            }
                        }
                    }
                }

                emit_term(&format!("  📈 [CROSS MATCH FINAL] Intersection: {:?} -> '{}' (Context Score: {:.4})", intersecting, final_text, context_score));

                context_arr.push(json!({
                    "type": best_cat,
                    "types": intersecting,
                    "text": final_text,
                    "score": context_score
                }));
            }
        }

        let mut segments = json!({
            "original_text": query.clone(),
            "context": context_arr
        });
        
        // Stage 1 완료 후 최종 분할된 맥락 트리 전체를 출력합니다.
        emit_term("\n=======================================");
        emit_term("[STAGE-1 RESULT] 🧩 Semantic Chunking Complete:");
        emit_term(&serde_json::to_string_pretty(&segments).unwrap_or_default());
        emit_term("=======================================\n");

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        // ----------------------------------------------------
        // Stage 2 & 3: Double Plinko Attribute/Operator Mapping & LLM Normalization
        // ----------------------------------------------------
        emit_term("[STAGE-2] Extracting attributes via Double Vector Plinko & LLM Normalization...");
        
        // 🌟 [SCOPE FIX] Stage-3 CROSS-VERB 에서 참조할 수 있도록 도메인 지시어 매핑을 외부 스코프에 선언합니다.
        let mut domain_word_related: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        // 🌟 [ACTION WORD SET] 벡터 역검증을 통과해 '순수 검색 명령어'로 확정된 단어입니다.
        //    STAGE-3 이 A/FULL 티어의 FTS 검색어에서 이 단어들만 제거합니다.
        //    다국어 어휘 리터럴을 코드에 두지 않기 위해 런타임 확정 집합만 사용합니다.
        let mut global_action_words: std::collections::HashSet<String> = std::collections::HashSet::new();

        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                    emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                    return Ok(json!({ "context": [], "cancelled": true }));
                }

                let payload = json!({ "task_id": task_id, "category": format!("Stage 2 ({}/{})", idx+1, total_segments), "summary": "Mapping attributes...", "spinner": "⠋" });
                let _ = app_handle.emit("extraction-progress", &payload);
                crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

                let mut current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                
                // 🌟 [2차 분기] 세부 속성 매칭: 해당 도메인 타입의 Schema Field(Property Bias/Prej) 로드
                let fields = crate::parsing::get_detail_schema_fields(&seg_type, "", language);
                
                let mut prop_keys = Vec::new();
                let mut bias_texts = Vec::new();
                let mut prej_texts = Vec::new();
                let mut prop_types = std::collections::HashMap::new(); // 🌟 스키마 타입 저장용 맵 추가
                
                for (key, desc, bias, prej) in fields {
                    // 🌟 [추가] url, link 관련 속성은 추출 대상 및 Plinko 슬롯에서 완전히 배제
                    let lower_key = key.to_lowercase();
                    if lower_key.contains("url") || lower_key.contains("link") {
                        continue;
                    }

                    prop_keys.push(key.clone());
                    bias_texts.push(bias);
                    prej_texts.push(if prej.trim().is_empty() { "random unrelated noise".to_string() } else { prej });
                    
                    // 🌟 [DB SCHEMA CHECK] 스키마 설명(desc)에서 실제 데이터 타입을 추출합니다.
                    // 언더바(_)를 기준으로 단어를 분리(split)하여 price, amount, quantity 등 특정 키워드가 독립적으로 존재하는지 검사하여 Number/Boolean 타입을 자동 인식합니다.
                    let parts: Vec<&str> = lower_key.split('_').collect();
                    let is_number = parts.iter().any(|&p| ["price", "amount", "quantity", "discount", "fee", "weight", "width", "height", "length", "limit"].contains(&p));
                    let is_boolean = parts.iter().any(|&p| ["only", "included"].contains(&p));

                    let type_str = if desc.contains("Number") || is_number { "Number" }
                                   else if desc.contains("Boolean") || is_boolean { "Boolean" }
                                   else if desc.contains("Array") { "Array" }
                                   else { "String" };
                    prop_types.insert(key, type_str);
                }

                // 🌟 [글로벌 속성 동적 확장] 뎁스(Depth)에 무관하게 bias.json 전체를 깊이 우선 탐색(Stack DFS)으로 순회하여 속성을 추출합니다.
                let mut loaded_globals = Vec::new();
                if let Some(root_obj) = crate::parsing::BIAS_DICT.as_object() {
                    let mut stack: Vec<(String, &Value)> = Vec::new();
                    
                    let excluded_keys = [
                        "ignore", "insight", "search_bridge", 
                        "sq", "ar", "az", "bn", "bg", "ca", "zh", "hr", "cs", "da", 
                        "nl", "en", "et", "fi", "fr", "ka", "de", "el", "he", "hi", 
                        "hu", "is", "id", "it", "ja", "kk", "km", "ko", "lv", "lt", 
                        "ms", "mr", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk", 
                        "sl", "es", "sw", "sv", "tl", "te", "th", "tr", "uk", "ur", 
                        "uz", "vi"
                    ];

                    // 1. 루트 레벨 객체들을 필터링하여 스택에 삽입
                    for (g_key, g_val) in root_obj {
                        if !excluded_keys.contains(&g_key.as_str()) {
                            // emit_term(&format!("  root Property '{}' loaded for 1st Plinko.", g_key));
                            stack.push((g_key.clone(), g_val));
                        }
                    }

                    // 2. 스택 기반 뎁스 프리(Depth-Free) 무한 탐색 루프
                    while let Some((node_key, node_val)) = stack.pop() {
                        if let Some(obj) = node_val.as_object() {
                            // 현재 객체가 속성 스키마의 필수 조건 3가지를 가졌다면 추출 (1단이든 2단이든 무조건 걸림)

                            if obj.contains_key("semantic") && obj.contains_key("bias") && obj.contains_key("prejudice") {
                                if !prop_keys.contains(&node_key) {
                                    // emit_term(&format!("  child Property '{}' loaded for 1st Plinko.", node_key));

                                    let desc = obj.get("semantic").and_then(|v| v.as_str()).unwrap_or("String").to_string();
                                    let bias = obj.get("bias").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let prej = obj.get("prejudice").and_then(|v| v.as_str()).unwrap_or("random unrelated noise").to_string();
                                    
                                    prop_keys.push(node_key.clone());
                                    bias_texts.push(bias);
                                    prej_texts.push(if prej.trim().is_empty() { "random unrelated noise".to_string() } else { prej });
                                    
                                    // 언더바(_)를 기준으로 단어를 분리(split)하여 price, amount, quantity 등 특정 키워드가 독립적으로 존재하는지 검사하여 Number/Boolean 타입을 자동 인식합니다.
                                    let node_key_lower = node_key.to_lowercase();
                                    let parts: Vec<&str> = node_key_lower.split('_').collect();
                                    let is_number = parts.iter().any(|&p| ["price", "amount", "quantity", "discount", "fee", "weight", "width", "height", "length", "limit"].contains(&p));
                                    let is_boolean = parts.iter().any(|&p| ["only", "included"].contains(&p));

                                    let type_str = if desc.contains("Number") || is_number { "Number" }
                                                   else if desc.contains("Boolean") || is_boolean { "Boolean" }
                                                   else if desc.contains("Array") { "Array" }
                                                   else { "String" };
                                    prop_types.insert(node_key.clone(), type_str);
                                    loaded_globals.push(node_key);
                                }
                            } else {
                                // 필수 조건이 없는 일반 컨테이너 객체(예: metrics)라면 하위 객체들을 스택에 넣어 계속 파고듦
                                for (sub_k, sub_v) in obj {
                                    stack.push((sub_k.clone(), sub_v));
                                }
                            }
                        }
                    }
                }

                // 🌟 [3차 분기] 동적 필터 카테고리 일괄 로드 (bias.json 구조 완전 동기화)
                let filter_categories = vec![
                    "operators", "metrics", "time_filters", "season_filters", 
                    "status_filters", "substantial_filters", "find_filters", "option_filters"
                ];

                #[derive(Clone)]
                struct DynamicFilterDef {
                    category: String,
                    key: String,
                }
                
                let mut dynamic_filter_defs = Vec::new();
                let mut dynamic_bias_texts = Vec::new();
                let mut dynamic_prej_texts = Vec::new();

                for cat in &filter_categories {
                    if let Some(obj) = crate::parsing::BIAS_DICT.get(*cat).and_then(|v| v.as_object()) {
                        for (k, v) in obj {
                            let mut b_text = format!("{} context: {}", cat, k); 
                            let mut p_text = format!("{} context: not {}", cat, k);
                            
                            if let Some(b) = v.get("bias").and_then(|val| val.as_str()) { 
                                b_text = format!("{} context: {}", cat, b); 
                            }
                            if let Some(p) = v.get("prejudice").and_then(|val| val.as_str()) { 
                                p_text = format!("{} context: {}", cat, p); 
                            }
                            
                            dynamic_filter_defs.push(DynamicFilterDef { 
                                category: cat.to_string(), 
                                key: k.to_string() 
                            });
                            dynamic_bias_texts.push(b_text);
                            dynamic_prej_texts.push(p_text);
                        }
                    }
                }

                // 🌟 [PHRASE-LEVEL BIAS BANK] 센트로이드 1벡터 임베딩을 완전히 폐기합니다.
                //    기존: bias_texts[i] = "goods title, goods name, goods 상품명, goods 프리미엄 무선 헤드폰, ..." 을
                //          통째로 1개 벡터로 만들어 비교 → 9개 개념의 평균이라 어떤 단어와도 0.3x 밖에 안 나왔고,
                //          그래서 contains() 문자열 포함 시 +0.5 라는 하드코딩 보너스로 억지 보정하고 있었습니다.
                //    변경: 구 단위로 쪼개 Max-Pool 로 비교하면 원문과 동일한 구는 코사인 1.0 이 되어
                //          보너스 없이도 압도적으로 승리합니다. (베이지 → color 뱅크의 "베이지" 구와 정확히 일치)
                //    추가: semantic 앵커(예: title 의 "의류명")를 뱅크에 편입하여 정답 구를 벡터 공간에 올립니다.
                let mut prop_phrase_texts: Vec<Vec<String>> = Vec::with_capacity(prop_keys.len());
                let mut prop_raw_weights: Vec<Vec<f32>> = Vec::with_capacity(prop_keys.len());
                let mut prop_prej_texts: Vec<Vec<String>> = Vec::with_capacity(prop_keys.len());
                for (i, raw) in bias_texts.iter().enumerate() {
                    let (mut ph, mut wt) = crate::utils::ai_utils::split_bias_phrases_weighted_full(raw);
                    let anchor = crate::utils::ai_utils::semantic_anchor_text(&query_lang, &seg_type, &prop_keys[i]);
                    for p in crate::utils::ai_utils::split_bias_phrases_full(&anchor) {
                        if !ph.iter().any(|e| e == &p) {
                            ph.push(p);
                            wt.push(1.0);
                        }
                    }
                    prop_phrase_texts.push(ph);
                    prop_raw_weights.push(wt);

                    let prej_raw = prej_texts.get(i).cloned().unwrap_or_default();
                    prop_prej_texts.push(crate::utils::ai_utils::split_bias_phrases_full(&prej_raw));
                }

                // 🌟 [AMBIGUITY MASK] bias.json 을 손대지 않고 무변별 구를 구조적으로 제거합니다.
                let ambiguity_mask = crate::utils::ai_utils::cross_field_ambiguous_phrase_mask(&prop_phrase_texts, &prop_prej_texts);

                let mut flat_phrases: Vec<String> = Vec::new();
                let mut prop_phrase_weights: Vec<Vec<f32>> = Vec::with_capacity(prop_keys.len());
                let mut prop_phrase_spans: Vec<(usize, usize)> = Vec::with_capacity(prop_keys.len());
                for i in 0..prop_phrase_texts.len() {
                    let start = flat_phrases.len();
                    let mut w: Vec<f32> = Vec::new();
                    let mut dropped: Vec<String> = Vec::new();
                    for (pi, keep) in ambiguity_mask[i].iter().enumerate() {
                        if *keep {
                            flat_phrases.push(prop_phrase_texts[i][pi].clone());
                            w.push(prop_raw_weights[i].get(pi).copied().unwrap_or(1.0));
                        } else {
                            dropped.push(prop_phrase_texts[i][pi].clone());
                        }
                    }
                    if !dropped.is_empty() {
                        emit_term(&format!(
                            "    🧪 [AMBIGUOUS PHRASE DROP] '{}' 뱅크에서 타 필드와 동일하거나 자기 prejudice 와 충돌하는 무변별 구 {}개 제거: {:?}",
                            prop_keys[i], dropped.len(), dropped.iter().take(6).collect::<Vec<_>>()
                        ));
                    }
                    prop_phrase_spans.push((start, flat_phrases.len()));
                    prop_phrase_weights.push(w);
                }

                emit_term(&format!("  🧱 [PROPERTY PHRASE BANK] 속성 {}개 / 변별 구 {}개 임베딩 개시...", prop_keys.len(), flat_phrases.len()));

                let mut flat_embs: Vec<Vec<f32>> = Vec::with_capacity(flat_phrases.len());
                for chunk in flat_phrases.chunks(200) {
                    let part = self.get_embedding_batch(chunk.to_vec()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk.len()]);
                    flat_embs.extend(part);
                }

                let mut prop_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(prop_keys.len());
                for (start, end) in &prop_phrase_spans {
                    let mut bank = Vec::with_capacity(end.saturating_sub(*start));
                    for idx in *start..*end {
                        bank.push(flat_embs.get(idx).cloned().unwrap_or_else(|| vec![0.0; 384]));
                    }
                    prop_phrase_embs.push(bank);
                }

                // 🌟 [BANK SIZE EQUALIZATION] 거대 뱅크(color ~700구)가 Max-Pool 통계만으로
                //    무관한 청크의 argmax 를 독식하는 '흡수 싱크' 를 구조적으로 해체합니다.
                //    기존 1회 중앙값 컷은 603구 → 302구 로 절반만 줄여
                //        √(2 ln 603)=3.58 → √(2 ln 302)=3.38  (이득 5.6% 감소)
                //    에 그쳤고, title(11구, 2.19) 대비 여전히 1.54배 유리했습니다.
                //    그래서 로그의 '팔린'(0.5780) '남긴'(0.6365) 이 계속 color 로 흡수되었습니다.
                //    목표 규모는 '이 스키마의 유효 구 개수 중앙값' 이라는 실측값이므로 새 상수가 아닙니다.
                {
                    let seg_query_emb = self.get_embedding(current_text.clone()).await.unwrap_or(vec![0.0; 384]);

                    let mut sizes: Vec<usize> = prop_phrase_embs.iter()
                        .map(|b| b.iter().filter(|e| !e.iter().all(|&v| v == 0.0)).count())
                        .filter(|&c| c > 0)
                        .collect();
                    sizes.sort_unstable();
                    let target_size = if sizes.is_empty() {
                        0
                    } else if sizes.len() % 2 == 0 {
                        (sizes[sizes.len() / 2 - 1] + sizes[sizes.len() / 2]) / 2
                    } else {
                        sizes[sizes.len() / 2]
                    };
                    if target_size > 0 {
                        emit_term(&format!("    📐 [BANK EQUALIZE TARGET] 이 스키마의 유효 구 개수 중앙값 {}구를 목표 규모로 삼습니다.", target_size));
                    }

                    for pi in 0..prop_phrase_embs.len() {
                        let before = prop_phrase_embs[pi].iter().filter(|e| !e.iter().all(|&v| v == 0.0)).count();
                        let keep = crate::utils::ai_utils::bank_size_equalized_mask(&seg_query_emb, &prop_phrase_embs[pi], target_size);
                        let mut dropped = 0usize;
                        for (i, k) in keep.iter().enumerate() {
                            if !*k {
                                prop_phrase_embs[pi][i] = vec![0.0; 384];
                                dropped += 1;
                            }
                        }
                        if dropped > 0 {
                            emit_term(&format!("    📉 [BANK EQUALIZE] '{}' 뱅크 {}구 → {}구 (Max-Pool 구조 이득 제거를 위해 {}개 비활성화)",
                                prop_keys[pi], before, before.saturating_sub(dropped), dropped));
                        }
                    }
                }

                let prej_embs = self.get_embedding_batch(prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; prop_keys.len()]);
                
                let dynamic_bias_embs = self.get_embedding_batch(dynamic_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; dynamic_filter_defs.len()]);
                let dynamic_prej_embs = self.get_embedding_batch(dynamic_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; dynamic_filter_defs.len()]);

                // ── 0) 전담 필터 카테고리가 이미 처리하는 키는 '속성' 후보가 아닙니다.
                //       season_filters.summer / operators.gt / metrics.time 등이 루트 DFS 로
                //       속성 목록에 이중 등록되어 있어, color 를 뺏긴 청크가 'summer(0.5129)' 같은
                //       엉뚱한 슬롯으로 흘러갈 수 있었습니다.
                //       단, 스키마 컬럼과 이름이 겹치는 키(goods.quantity ↔ metrics.quantity)는
                //       루트 DFS 가 등록한 것이 아니므로(loaded_globals 에 없음) 그대로 보존합니다.
                //       color 는 어떤 filter_category 에도 속하지 않으므로 반드시 살아남습니다.
                let filter_owned_keys: std::collections::HashSet<String> =
                    dynamic_filter_defs.iter().map(|d| d.key.clone()).collect();
                let is_filter_owned = |name: &str| -> bool {
                    loaded_globals.iter().any(|g| g == name) && filter_owned_keys.contains(name)
                };

                // 🌟 [FILTER-OWNED MASK] season_filters.summer / time_filters.this_year / operators.top 은
                //    semantic+bias+prejudice 3종 세트를 갖고 있어 루트 DFS 가 '스키마 속성'으로 등록해 버립니다.
                //    기존에는 행렬 구축 시점에만 걸러서, 채점 단계에서는 여전히 1순위를 독식했습니다.
                //    (로그: '무거운' -> 1st: [summer] (0.5612))
                //    채점 루프 진입 자체를 막습니다.
                let prop_is_filter_owned: Vec<bool> = prop_keys.iter().map(|k| is_filter_owned(k)).collect();
                {
                    let owned: Vec<&String> = prop_keys.iter().enumerate()
                        .filter(|(i, _)| prop_is_filter_owned[*i]).map(|(_, k)| k).collect();
                    if !owned.is_empty() {
                        emit_term(&format!("    🚧 [FILTER-OWNED EXCLUDE] 필터 카테고리 소유 키 {}개를 속성 후보에서 제외: {:?}",
                            owned.len(), owned.iter().take(10).collect::<Vec<_>>()));
                    }
                }

                // 🌟 [TEMPORAL PHRASE BANK PRE-BUILD]
                //    기존 TEMPORAL PRE-GATE는 dynamic_bias_embs(센트로이드 1벡터)로 비교하여
                //    한국어 2음절 단어("올해")가 영어 문장 센트로이드와 코사인이 낮아 실패했습니다.
                //    여기서는 bias.json 의 time_filters / season_filters 각 키의
                //    semantic + bias 를 구 단위로 쪼개 임베딩하고 Max-Pool 로 비교합니다.
                //    '올해' 와 embed("current year") 의 코사인은 multilingual 모델에서 0.6+ 입니다.
                //    이 뱅크는 세그먼트 루프 시작 시 1회만 구축합니다.
                let temporal_phrases = crate::utils::ai_utils::temporal_semantic_phrases();
                let temporal_phrase_texts: Vec<String> = temporal_phrases.iter().map(|(_, p)| p.clone()).collect();
                let temporal_phrase_embs: Vec<Vec<f32>> = if temporal_phrase_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(temporal_phrase_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; temporal_phrase_texts.len()])
                };
                // 🌟 [NUMERIC OPERATOR PHRASE BANK PRE-BUILD]
                //    operators 카테고리 각 키의 bias 를 구 단위로 쪼개 임베딩합니다.
                //    "이하로" 와 embed("less than or equal") / embed("under") / embed("no more than") 의
                //    Max-Pool 코사인이 top/bottom 뱅크보다 높으면 비교 연산자로 확정합니다.
                let mut op_phrase_texts: Vec<String> = Vec::new();
                let mut op_phrase_is_rank: Vec<bool> = Vec::new();
                if let Some(ops_node) = crate::parsing::BIAS_DICT.get("operators").and_then(|v| v.as_object()) {
                    for (op_key, op_val) in ops_node {
                        let is_rank = op_key == "top" || op_key == "bottom";
                        if let Some(bias_str) = op_val.get("bias").and_then(|v| v.as_str()) {
                            for phrase in crate::utils::ai_utils::split_bias_phrases_full(bias_str) {
                                op_phrase_texts.push(phrase);
                                op_phrase_is_rank.push(is_rank);
                            }
                        }
                        if let Some(semantic_str) = op_val.get("semantic").and_then(|v| v.as_str()) {
                            for phrase in crate::utils::ai_utils::split_bias_phrases_full(semantic_str) {
                                op_phrase_texts.push(phrase);
                                op_phrase_is_rank.push(is_rank);
                            }
                        }
                    }
                }
                let op_phrase_embs: Vec<Vec<f32>> = if op_phrase_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(op_phrase_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; op_phrase_texts.len()])
                };
                // 🌟 [METRICS FAMILY BANK] bias.json 의 metrics.* 를 (계열, 구) 로 펼쳐 임베딩합니다.
                //    metrics.price.bias 에 "won" 이 이미 존재하므로 다국어 임베딩이 '원' ↔ 'won' 을
                //    연결해 줍니다. 이 축이 있어야 "5000원 이하로" 의 수치 대상이
                //    quantity 가 아니라 price 계열이라는 사실을 어휘 하드코딩 없이 확정할 수 있습니다.
                let metric_family_defs = crate::utils::ai_utils::metrics_family_phrases();
                let metric_family_texts: Vec<String> = metric_family_defs.iter().map(|(_, p)| p.clone()).collect();
                let metric_family_raw: Vec<Vec<f32>> = if metric_family_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(metric_family_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; metric_family_texts.len()])
                };
                let metric_family_bank: Vec<(String, Vec<f32>)> = metric_family_defs.iter()
                    .zip(metric_family_raw.into_iter())
                    .map(|((k, _), e)| (k.clone(), e))
                    .collect();
                if !metric_family_bank.is_empty() {
                    emit_term(&format!("    📐 [METRICS FAMILY BANK] metrics 계열 구 {}개 준비 완료.", metric_family_bank.len()));
                }
                // 🌟 [ALL-FILTER PHRASE BANK PRE-BUILD]
                //    substantial_filters / find_filters / status_filters / time_filters / season_filters
                //    전 카테고리의 bias+semantic 구를 임베딩합니다.
                //    (로그: '무거운'→unit, '많이'→condition, '팔린'→color 오배정의 공통 원인은
                //     이 필터들이 2nd Plinko 에만 있어 1st 에서 스키마 속성이 먼저 선점하기 때문입니다)
                //    bias.json 을 수정하지 않고 기존 bias/semantic 필드만 동적으로 읽어 구축합니다.
                let all_filter_phrases = crate::utils::ai_utils::filter_category_phrases(&[
                    "substantial_filters", "find_filters", "status_filters", "time_filters", "season_filters",
                ]);
                let all_filter_texts: Vec<String> = all_filter_phrases.iter().map(|(_, _, p)| p.clone()).collect();
                let all_filter_raw_embs: Vec<Vec<f32>> = if all_filter_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(all_filter_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; all_filter_texts.len()])
                };
                // (category, key, embedding) 트리플로 재조립
                let mut all_filter_embs: Vec<(String, String, Vec<f32>)> = all_filter_phrases.iter()
                    .zip(all_filter_raw_embs.into_iter())
                    .map(|((cat, key, _), emb)| (cat.clone(), key.clone(), emb))
                    .collect();

                // 🌟 [ABSTRACT BRIDGE BANK]
                //    bias.json 의 search_bridge.abstract_bridge 에서 영어 브릿지 구를 읽어 임베딩합니다.
                //    substantial_filters / find_filters 의 원본 bias 는 영어 3~6구뿐이라
                //    한국어 '무거운' 과의 코사인이 구조적으로 낮았습니다.
                //    브릿지 구를 얹어 뱅크 밀도를 올리고, EVT 정규화로 크기 편향까지 제거합니다.
                let bridge_defs = crate::utils::ai_utils::abstract_bridge_phrases();
                let bridge_texts: Vec<String> = bridge_defs.iter().map(|(_, _, p)| p.clone()).collect();
                let bridge_raw: Vec<Vec<f32>> = if bridge_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(bridge_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; bridge_texts.len()])
                };
                let abstract_bridge_embs: Vec<(String, String, Vec<f32>)> = bridge_defs.iter()
                    .zip(bridge_raw.into_iter())
                    .map(|((c, k, _), e)| (c.clone(), k.clone(), e))
                    .collect();
                for t in &abstract_bridge_embs { all_filter_embs.push(t.clone()); }
                if !abstract_bridge_embs.is_empty() {
                    emit_term(&format!(
                        "    🌉 [ABSTRACT BRIDGE BANK] 추상 수식어 브릿지 구 {}개 준비 완료. (substantial/find 다국어 매칭용)",
                        abstract_bridge_embs.len()
                    ));
                }

                // 🌟 [FILTER PREJUDICE BANK] bias.json 이 필터 키마다 이미 갖고 있는 편견 사전을
                //    필터 라우팅 경로에서 처음으로 활용합니다.
                //    status_filters.progress.prejudice = "draft, complete, error, stop, pause" 처럼
                //    '이 필터가 절대 아닌 개념' 이 명시되어 있어 우연 공명을 직접 상쇄합니다.
                let mut filter_prej_defs = crate::utils::ai_utils::filter_category_prejudice_phrases(&[
                    "substantial_filters", "find_filters", "status_filters", "time_filters", "season_filters",
                ]);
                for t in crate::utils::ai_utils::abstract_bridge_prejudice_phrases() {
                    if !filter_prej_defs.iter().any(|(c, k, p)| c == &t.0 && k == &t.1 && p == &t.2) {
                        filter_prej_defs.push(t);
                    }
                }
                let filter_prej_texts: Vec<String> = filter_prej_defs.iter().map(|(_, _, p)| p.clone()).collect();
                let filter_prej_raw: Vec<Vec<f32>> = if filter_prej_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(filter_prej_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; filter_prej_texts.len()])
                };
                let all_filter_prej_embs: Vec<(String, String, Vec<f32>)> = filter_prej_defs.iter()
                    .zip(filter_prej_raw.into_iter())
                    .map(|((c, k, _), e)| (c.clone(), k.clone(), e))
                    .collect();
                emit_term(&format!("    🛡️ [FILTER PREJUDICE BANK] 필터 편견 구 {}개 준비 완료.", all_filter_prej_embs.len()));

                if !temporal_phrase_embs.is_empty() {
                    emit_term(&format!("    📐 [TEMPORAL PHRASE BANK] time/season 구 {}개 준비 완료.", temporal_phrase_embs.len()));
                }
                if !op_phrase_embs.is_empty() {
                    emit_term(&format!("    📐 [OPERATOR PHRASE BANK] operators 구 {}개 준비 완료.", op_phrase_embs.len()));
                }
                if !all_filter_embs.is_empty() {
                    emit_term(&format!("    📐 [ALL-FILTER PHRASE BANK] substantial/find/status/time/season 구 {}개 준비 완료.", all_filter_embs.len()));
                }

                // 🌟 Plinko Game (1st Depth): Sliding Window Cliff Detection over words
                struct PlinkoMatch {
                    chunk: String,
                    best_prop: String,
                    best_score: f32,
                    alternatives: Vec<(String, f32)>,
                    // 🌟 [FULL SCORE VECTOR] 배타 배정 행렬을 만들려면 top-1 과 상위 5개가 아니라
                    //    '이 청크가 모든 속성에 대해 받은 점수 전체'가 필요합니다.
                    //    (베이지가 color 를 선점하면 가디건은 자기 점수표에서 다음 유효 속성을 찾아야 합니다)
                    all_scores: Vec<(String, f32)>,
                }
                let mut plinko_matches: Vec<PlinkoMatch> = Vec::new();
                let mut plinko_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
                // 🌟 [N:N ALTERNATE AXIS] 확정 속성이 틀렸을 때를 대비한 '같은 값의 차순위 속성' 목록입니다.
                //    STAGE-3 의 N:N 조합과 프론트엔드 Dexie 재질의가 이 목록을 소비합니다.
                let mut plinko_alternates: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
                // 🌟 [UNASSIGNED RESCUE] 속성 확정에 실패했거나 억지 배정으로 폐기된 청크입니다.
                //    사용자가 실제로 입력한 단어이므로 조건이 되지 못하더라도 FTS 검색어로는 반드시 살아남아야 합니다.
                //    (로그: review 세그먼트의 '메세지도' 가 B/NARROWED·C/RECALL 티어에서 통째로 사라졌습니다)
                let mut unassigned_chunks: Vec<String> = Vec::new();
                let words: Vec<&str> = current_text.split_whitespace().collect();

                // 🌟 [DOMAIN TYPE WORD DETECTION]
                //    "이벤트로", "주문에서" 같은 도메인 지시어는 속성 값이 아니라 테이블 타입 지표입니다.
                //    로컬라이즈된 타입 이름(get_localized_page_type)과의 코사인 비교로 판정합니다.
                //    bias.json 수정 없이 기존 함수를 재사용하며, 새 매직 상수 없이
                //    '도메인 코사인 > 스키마 속성 Max-Pool 코사인' 상대 비교만 사용합니다.
                let domain_type_names: Vec<(String, String)> = ["order", "goods", "tracking", "review", "coupon", "event"]
                    .iter()
                    .map(|cat| (cat.to_string(), crate::parsing::get_localized_page_type(cat, &query_lang)))
                    .collect();
                let domain_type_texts: Vec<String> = domain_type_names.iter().map(|(_, name)| name.clone()).collect();
                let domain_type_embs: Vec<Vec<f32>> = if domain_type_texts.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(domain_type_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; domain_type_texts.len()])
                };
                let mut domain_indicator_words: std::collections::HashSet<String> = std::collections::HashSet::new();
                for word in &words {
                    if word.chars().any(|c| c.is_ascii_digit()) { continue; }
                    let word_emb_d = self.get_embedding(word.to_string()).await.unwrap_or(vec![0.0; 384]);
                    if word_emb_d.iter().all(|&v| v == 0.0) { continue; }
                    let mut best_schema_for_word = f32::MIN;
                    for pi in 0..prop_phrase_embs.len() {
                        let s = crate::utils::ai_utils::weighted_max_pool_sim(&word_emb_d, &prop_phrase_embs[pi], &prop_phrase_weights[pi]);
                        if s > best_schema_for_word { best_schema_for_word = s; }
                    }
                    let mut best_domain_score = f32::MIN;
                    let mut best_domain_cat = String::new();
                    for (di, (cat, _name)) in domain_type_names.iter().enumerate() {
                        if domain_type_embs[di].iter().all(|&v| v == 0.0) { continue; }
                        let s = cosine_similarity(&word_emb_d, &domain_type_embs[di]);
                        if s > best_domain_score {
                            best_domain_score = s;
                            best_domain_cat = cat.clone();
                        }
                    }
                    if best_domain_score > best_schema_for_word && best_domain_score > 0.0 {
                        domain_indicator_words.insert(word.to_string());
                        emit_term(&format!("      🏷️ [DOMAIN TYPE WORD] '{}' 는 '{}' 도메인 지시어로 판정. 속성 배정에서 제외하고 FTS 검색어로 보존합니다.", word, best_domain_cat));
                        // 🌟 [RELATED DOMAIN COLLECT] 이 단어와 코사인이 양수인 모든 도메인을 기록합니다.
                        //    '판매된' → goods(최고) 이지만 order 와도 코사인 > 0 이면
                        //    STAGE-3 에서 order CROSS-VERB 쿼리를 발행할 수 있습니다.
                        let mut related: Vec<String> = Vec::new();
                        for (di, (cat, _name)) in domain_type_names.iter().enumerate() {
                            if domain_type_embs[di].iter().all(|&v| v == 0.0) { continue; }
                            let s = cosine_similarity(&word_emb_d, &domain_type_embs[di]);
                            if s > 0.0 {
                                related.push(cat.clone());
                            }
                        }

                        // 🌟 [SALES BRIDGE] bias.json 의 search_bridge.sales_to_order 를 읽어
                        //    "판매/팔린/매출" 계열 단어가 감지되면 order 도메인을 related 에 강제 포함합니다.
                        //    기존 코사인 > 0 조건만으로는 다국어 임베딩에서 order 앵커와
                        //    "판매된" 간 코사인이 음수가 될 수 있어 브릿지가 필요합니다.
                        {
                            let sales_bridge_bias: Vec<String> = {
                                let dict = &crate::parsing::BIAS_DICT;
                                dict.get("search_bridge")
                                    .and_then(|sb| sb.get("sales_to_order"))
                                    .and_then(|n| n.get("bias"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| crate::utils::ai_utils::split_bias_phrases_full(s))
                                    .unwrap_or_default()
                            };
                            if !sales_bridge_bias.is_empty() {
                                let bridge_embs = self.get_embedding_batch(sales_bridge_bias.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; sales_bridge_bias.len()]);
                                let bridge_score = crate::utils::ai_utils::max_pool_sim(&word_emb_d, &bridge_embs);
                                // 기존 최고 도메인 점수의 90% 이상이면 브릿지 발동
                                if bridge_score > best_domain_score * 0.9 && bridge_score > 0.0 {
                                    if !related.iter().any(|r| r == "order") {
                                        related.push("order".to_string());
                                    }
                                    emit_term(&format!(
                                        "  🌉 [SALES BRIDGE] '{}' 는 sales/order 브릿지 코사인 {:.4} 로 order 도메인 추가 포함",
                                        word, bridge_score
                                    ));
                                }
                            }
                        }

                        if !related.is_empty() {
                            domain_word_related.insert(word.to_string(), related);
                        }
                    }
                }
                
                let mut current_chunk: Vec<String> = Vec::new();
                let mut prev_max_score = -1.0;
                let mut best_prop_for_chunk = String::new();
                let mut prev_alternatives: Vec<(String, f32)> = Vec::new();
                let mut prev_all_scores: Vec<(String, f32)> = Vec::new();

                // 🌟 [FORCED FILTER ROUTES] 속성이 아니라 필터로 확정된 단어들.
                //    (word, category, key, evt_score)
                //    기존에는 FILTER TERM DROP 이 단어만 버리고 '어느 필터였는지'를 기록하지 않아
                //    substantial / find 가 끝까지 빈 값으로 남았습니다.
                let mut forced_filter_routes: Vec<(String, String, String, f32)> = Vec::new();

                emit_term(&format!("  🎯 [PLINKO GAME (1st)] Starting Sliding Window Cliff Detection for '{}'", current_text));

                // 🌟 [개선] 다국어(52개국어)의 검색/요청 의미를 갖는 단어들을 모두 기준 벡터(Centroid)에 포함시켜 하드코딩 매칭 없이 벡터만으로 완벽한 시맨틱 필터링을 수행합니다.
                let action_verbs = "find, search, query, get, question, request, 찾아, 알려줘, 보여줘, 조사, 결과, 답변, 대답, 말해, 검색, 해줘, 질문, 질의, 요청, 확인, 알아봐, 찾아봐, 가져와, 설명, 요약, 추천, 정보, gjej, kërko, pyet, merr, ابحث, بحث, استعلام, الحصول, tap, axtarış, sorğu, əldə et, খুঁজুন, অনুসন্ধান, প্রশ্ন, পান, намери, търси, заявка, получи, trobar, cercar, consulta, obtenir, 找, 搜索, 查询, 获取, 给我看, 告诉我, nađi, pretraži, upit, dobij, najít, hledat, dotaz, získat, søg, forespørgsel, hent, vind, zoek, zoekopdracht, krijg, leia, otsi, päring, saada, löydä, etsi, kysely, hae, trouver, chercher, requête, obtenir, montre-moi, dis-moi, იპოვე, ძებნა, მოთხოვნა, მიიღე, finden, suchen, abfrage, bekommen, zeig mir, sag mir, βρες, αναζήτηση, ερώτημα, πάρε, מצא, חפש, שאילתה, קבל, खोजें, खोज, क्वेरी, प्राप्त करें, talál, keres, lekérdezés, kap, finna, leita, fyrirspurn, fá, temukan, cari, kueri, dapatkan, trova, cerca, ottieni, mostrami, dimmi, 見つける, 検索, クエリ, 取得, 教えて, 見せて, табу, іздеу, сұрау, алу, ស្វែងរក, ស្រាវជ្រាវ, សំណួរ, ទទួលបាន, atrast, meklēt, vaicājums, iegūt, rasti, ieškoti, užklausa, gauti, carian, pertanyaan, शोधा, शोध, मिळवा, finn, søk, spørring, پیدا کردن, جستجو, پرس و جو, گرفتن, znajdź, szukaj, zapytanie, pobierz, encontrar, pesquisar, obter, mostre-me, diga-me, găsește, caută, interogare, obține, найти, поиск, запрос, получить, покажи, расскажи, нађи, претрага, упит, добиј, nájsť, hľadať, dopyt, získať, najdi, iskanji, poizvedba, dobi, buscar, obtener, muéstrame, dime, tafuta, utafutaji, swali, pata, hitta, sök, fråga, hämta, hanapin, maghanap, kunin, కనుగొనండి, శోధన, ప్రశ్న, పొందండి, ค้นหา, ค้น, คิวรี, รับ, bul, ara, sorgu, al, знайти, пошук, запит, отримати, تلاش, تلاش کریں, استفسار, حاصل کریں, topish, qidirish, so'rov, olish, tìm, tìm kiếm, truy vấn, lấy, cho tôi xem, nói cho tôi";
                
                // 🌟 [ACTION VERB PHRASE BANK] 500단어를 벡터 1개로 합치면 센트로이드가 되어
                //    문자열에 '보여줘' 가 이미 있는데도 코사인 0.55 를 못 넘습니다.
                //    필드 bias 에서 이미 폐기한 센트로이드 방식이 여기 남아 있었습니다.
                //    구 단위로 쪼개면 자기 자신과의 코사인이 1.0 이라 확실히 잡힙니다.
                let action_verb_phrases = crate::utils::ai_utils::split_bias_phrases_full(action_verbs);
                let action_verb_embs: Vec<Vec<f32>> = if action_verb_phrases.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(action_verb_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; action_verb_phrases.len()])
                };

                let mut retained_words = Vec::new();

                for word in words {
                    // 🌟 [ORDER FIX] 1. DOMAIN TYPE WORD DROP 을 최우선으로 올립니다.
                    //    사전 패스가 이미 '이벤트로'/'판매된'/'고객의' 를 도메인 지시어로 확정했는데
                    //    본 루프의 ACTION VERB 가 그 판정을 덮어써 왔습니다.
                    //    (log1.txt: DOMAIN TYPE WORD 로그 직후 동일 단어가 ACTION VERB IGNORED)
                    //    ACTION VERB 로 소비되면 retained_words 에도 안 들어가 B/NARROWED 텍스트에서도 증발합니다.
                    if domain_indicator_words.contains(word) {
                        retained_words.push(word);
                        if !unassigned_chunks.iter().any(|e| e == word) {
                            unassigned_chunks.push(word.to_string());
                        }
                        continue;
                    }

                    // 2. FILTER TERM DROP
                    if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("season_filters", word) {
                        emit_term(&format!("      ✂️ [FILTER TERM DROP] '{}' 는 season_filters.{}.exact_match 확정어이므로 속성 배정에서 제외합니다. (FTS 검색어로는 보존)", word, k));
                        retained_words.push(word);
                        if !unassigned_chunks.iter().any(|e| e == word) {
                            unassigned_chunks.push(word.to_string());
                        }
                        continue;
                    }
                    if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("time_filters", word) {
                        emit_term(&format!("      ✂️ [FILTER TERM DROP] '{}' 는 time_filters.{}.exact_match 확정어이므로 속성 배정에서 제외합니다. (FTS 검색어로는 보존)", word, k));
                        retained_words.push(word);
                        if !unassigned_chunks.iter().any(|e| e == word) {
                            unassigned_chunks.push(word.to_string());
                        }
                        continue;
                    }

                    // 3. ACTION VERB IGNORED — 4중 역검증
                    let word_emb = self.get_embedding(word.to_string()).await.unwrap_or(vec![0.0; 384]);
                    let action_sim = crate::utils::ai_utils::max_pool_sim(&word_emb, &action_verb_embs);
                    let op_sim = crate::utils::ai_utils::max_pool_sim(&word_emb, &operator_embs);
                    let word_has_digit = word.chars().any(|c| c.is_ascii_digit());

                    // 🌟 [REVERSE VERIFICATION] action_verbs 는 ~500구 다국어 뱅크이고
                    //    Max-Pool 은 그 중 최댓값을 취하므로, 한국어 2~3음절 단어는
                    //    구조적으로 0.65~0.80 대역의 우연 공명이 반드시 발생합니다.
                    //    (log2: '니트' 0.7400 / '가디건' 0.7429 — 둘 다 상품명)
                    //    (log1: '제품' 0.6741 vs OpSim 0.6715 — 마진 0.0026)
                    //    절대 임계치로는 이 잡음을 구분할 수 없으므로,
                    //    '이 단어가 다른 어떤 뱅크보다 명령어 뱅크에 더 가까운가' 라는
                    //    상대 우위로 판정합니다. 새 매직 상수를 도입하지 않습니다.
                    //
                    //    ① 속성 뱅크  : '니트'/'가디건'/'제품' 같은 값 명사를 구제
                    //    ② 연산자 뱅크 : '이하로' 같은 비교 표현을 구제 (P0 — 가격 조건 복원)
                    //    ③ 시간 뱅크  : '올해' 같은 시간 표현을 구제
                    //    ④ 필터 뱅크  : '팔린'/'남긴' 같은 상태·수식 표현을 구제
                    let mut max_prop_sim = 0.0f32;
                    for pi in 0..prop_phrase_embs.len() {
                        if prop_is_filter_owned[pi] { continue; }
                        let s = crate::utils::ai_utils::weighted_max_pool_sim(
                            &word_emb, &prop_phrase_embs[pi], &prop_phrase_weights[pi],
                        );
                        if s > max_prop_sim { max_prop_sim = s; }
                    }
                    let op_bank_sim = if op_phrase_embs.is_empty() {
                        op_sim
                    } else {
                        crate::utils::ai_utils::max_pool_sim(&word_emb, &op_phrase_embs).max(op_sim)
                    };
                    let temporal_sim = if temporal_phrase_embs.is_empty() {
                        0.0f32
                    } else {
                        crate::utils::ai_utils::max_pool_sim(&word_emb, &temporal_phrase_embs)
                    };
                    let filter_sim = {
                        let mut m = 0.0f32;
                        for (_, _, e) in all_filter_embs.iter() {
                            if e.iter().all(|&v| v == 0.0) { continue; }
                            let s = cosine_similarity(&word_emb, e);
                            if s > m { m = s; }
                        }
                        m
                    };

                    let rival_max = max_prop_sim
                        .max(op_bank_sim)
                        .max(temporal_sim)
                        .max(filter_sim);

                    let is_action_verb = word != "|"
                        && !word_has_digit
                        && action_sim > rival_max;

                    if is_action_verb {
                        emit_term(&format!(
                            "    🚫 [ACTION VERB IGNORED] '{}' | Action: {:.4} > Rival max {:.4} (Prop {:.4} / Op {:.4} / Time {:.4} / Filter {:.4}). Skipping Plinko mapping.",
                            word, action_sim, rival_max, max_prop_sim, op_bank_sim, temporal_sim, filter_sim
                        ));

                        // 🌟 [FTS 정화용 기록] 벡터로 확정된 순수 명령어만 STAGE-3 검색 텍스트에서 제거합니다.
                        //    다국어 어휘 하드코딩 없이 이 목록만 소비합니다.
                        if !global_action_words.contains(word) {
                            global_action_words.insert(word.to_string());
                        }

                        // 이전에 쌓인 청크가 유효하다면 즉시 강제 Cliff(저장) 처리하여 슬롯에 안전하게 넣습니다.
                        if !current_chunk.is_empty() && prev_max_score > 0.20 && !best_prop_for_chunk.is_empty() {
                            emit_term(&format!("    📉 [FORCED CLIFF] Action verb intercepted. End of semantic chunk."));
                            emit_term(&format!("      📥 [DROPPED INTO SLOT] '{}' belongs to property [{}]", current_chunk.join(" "), best_prop_for_chunk));
                            plinko_matches.push(PlinkoMatch {
                                chunk: current_chunk.join(" "),
                                best_prop: best_prop_for_chunk.clone(),
                                best_score: prev_max_score,
                                alternatives: prev_alternatives.clone(),
                                all_scores: prev_all_scores.clone(),
                            });
                        }

                        // 윈도우 완전 초기화 (명령어 단어는 버림)
                        current_chunk = Vec::new();
                        prev_max_score = -1.0;
                        best_prop_for_chunk = String::new();
                        prev_alternatives = Vec::new();
                        prev_all_scores = Vec::new();
                        continue;
                    } else if action_sim > rival_max - 0.05 {
                        emit_term(&format!(
                            "    🛡️ [ACTION VERB RESCUE] '{}' | Action: {:.4} <= Rival max {:.4} (Prop {:.4} / Op {:.4} / Time {:.4} / Filter {:.4}). 명령어가 아니라 값/연산자/시간 표현으로 판정하여 Plinko 로 보냅니다.",
                            word, action_sim, rival_max, max_prop_sim, op_bank_sim, temporal_sim, filter_sim
                        ));
                    }

                    // 4. FUNCTIONAL WORD DROP
                    if crate::utils::ai_utils::is_functional_word_chunk(
                        word,
                        &ext_words_string,
                        stanza_lemmas.as_deref(),
                        stanza_deprels.as_deref(),
                    ) {
                        emit_term(&format!("    ✂️ [FUNCTIONAL WORD DROP] '{}' 는 기능어/조사 구조라 속성 값이 될 수 없습니다. Plinko 진입 제외.", word));
                        retained_words.push(word);
                        continue;
                    }

                    // 🌟 5. [STEM SUBSTITUTION]
                    // '제품중에서' / '제품으로' 처럼 같은 질의 안의 다른 토큰과
                    // 접두를 공유하는 굴절형은, 어간이 원형보다 더 강한 근거를 갖는지 코사인으로 확인해
                    // 어간으로 치환합니다. 언어별 조사/어미 사전을 쓰지 않고
                    // '접두 공유' 라는 구조적 사실 + surprisal 비교만 사용합니다.
                    let mut effective_word: String = word.to_string();
                    if !word_has_digit {
                        let stems = crate::utils::ai_utils::shared_prefix_stems(word, &ext_words_string);
                        if !stems.is_empty() {
                            let base_emb = self.get_embedding(word.to_string()).await.unwrap_or(vec![0.0; 384]);
                            let (_, base_schema) = crate::utils::ai_utils::surprisal_dual_scores(
                                &base_emb, &all_filter_embs, &all_filter_prej_embs,
                                &prop_keys, &prop_phrase_embs, &prop_is_filter_owned,
                            );
                            let base_top = base_schema.first().map(|s| s.surprisal).unwrap_or(f32::MIN);

                            for stem in stems.iter().take(2) {
                                let se = self.get_embedding(stem.clone()).await.unwrap_or(vec![0.0; 384]);
                                if se.iter().all(|&v| v == 0.0) { continue; }
                                let (_, stem_schema) = crate::utils::ai_utils::surprisal_dual_scores(
                                    &se, &all_filter_embs, &all_filter_prej_embs,
                                    &prop_keys, &prop_phrase_embs, &prop_is_filter_owned,
                                );
                                let stem_top = stem_schema.first().map(|s| s.surprisal).unwrap_or(f32::MIN);
                                if stem_top > base_top {
                                    emit_term(&format!(
                                        "      ✂️ [STEM SUBSTITUTION] '{}' → '{}' | SchemaSurprisal {:+.4} → {:+.4} (굴절 접미가 의미를 희석시켰습니다)",
                                        word, stem, base_top, stem_top
                                    ));
                                    effective_word = stem.clone();
                                    break;
                                }
                            }
                        }
                    }

                    // 🌟 [SURPRISAL ROUTE GATE]
                    //    필터 뱅크와 스키마 뱅크를 '하나의 공통 기준선' 으로 동시 채점합니다.
                    //        surprisal = (max - μ_global)/σ_global - √(2 ln N)
                    //    surprisal > 0 = "N개를 무작위로 뽑은 기대치보다 실제로 더 가깝다"
                    //    이 0 은 극값이론에서 유도된 값이므로 매직 상수가 아닙니다.
                    //    무관한 단어는 전 뱅크에서 음수가 나와 라우팅 자체가 일어나지 않습니다.
                    //
                    //    🌟 숫자를 포함한 단어는 '정도' 가 아니라 '값' 이므로 게이트를 건너뜁니다.
                    let word_has_digit = word.chars().any(|c| c.is_ascii_digit());
                    if !word_has_digit && !all_filter_embs.is_empty() {
                        let we = self.get_embedding(effective_word.clone()).await.unwrap_or(vec![0.0; 384]);
                        if !we.iter().all(|&v| v == 0.0) {
                            let (f_scores, s_scores) = crate::utils::ai_utils::surprisal_dual_scores(
                                &we,
                                &all_filter_embs,
                                &all_filter_prej_embs,
                                &prop_keys,
                                &prop_phrase_embs,
                                &prop_is_filter_owned,
                            );

                            let schema_top = s_scores.first().map(|s| s.surprisal).unwrap_or(f32::MIN);
                            let schema_name = s_scores.first().map(|s| s.key.clone()).unwrap_or_default();

                            if let Some(top) = f_scores.first() {
                                // 진단용: 상위 3개 필터와 스키마 1위를 항상 남깁니다.
                                let brief: Vec<String> = f_scores.iter().take(3)
                                    .map(|s| format!("{}.{}({:+.3}|cos {:.3}|N{})", s.category, s.key, s.surprisal, s.max_cos, s.n))
                                    .collect();
                                emit_term(&format!(
                                    "      📐 [SURPRISAL] '{}' | Filters: {} | SchemaTop: {}({:+.3})",
                                    effective_word, brief.join(" · "), schema_name, schema_top
                                ));

                                if top.surprisal > 0.0 && top.surprisal > schema_top {
                                    let claim = ["substantial_filters", "find_filters"];
                                    let is_abstract = claim.iter().any(|c| c == &top.category);

                                    if is_abstract {
                                        // 🌟 추상 수식어 확정 → substantial / find 양쪽 argmax 를 모두 귀속합니다.
                                        for cat in claim.iter() {
                                            if let Some(b) = f_scores.iter().find(|s| &s.category == cat) {
                                                emit_term(&format!(
                                                    "      🧲 [ABSTRACT QUALIFIER ROUTE] '{}' → {}.{} | Surprisal: {:+.4} (cos {:.4}, N={}) > SchemaTop: {:+.4}",
                                                    effective_word, b.category, b.key, b.surprisal, b.max_cos, b.n, schema_top
                                                ));
                                                forced_filter_routes.push((word.to_string(), b.category.clone(), b.key.clone(), b.surprisal));
                                            }
                                        }
                                    } else {
                                        emit_term(&format!(
                                            "      ✂️ [FILTER TERM DROP] '{}' → {}.{} | Surprisal: {:+.4} (cos {:.4}, N={}) > SchemaTop: {:+.4}",
                                            effective_word, top.category, top.key, top.surprisal, top.max_cos, top.n, schema_top
                                        ));
                                        forced_filter_routes.push((word.to_string(), top.category.clone(), top.key.clone(), top.surprisal));
                                    }

                                    retained_words.push(word);
                                    if !unassigned_chunks.iter().any(|e| e == word) {
                                        unassigned_chunks.push(word.to_string());
                                    }
                                    continue;
                                } else if top.surprisal <= 0.0 {
                                    emit_term(&format!(
                                        "      ⚪ [SURPRISAL GATE] '{}' | 최고 필터 {}.{} Surprisal {:+.4} <= 0. 무작위 기대치를 넘지 못해 필터 라우팅을 하지 않습니다.",
                                        effective_word, top.category, top.key, top.surprisal
                                    ));
                                }
                            }
                        }
                    }

                    retained_words.push(word);

                    // 7. Plinko Window Logic
                    let mut test_chunk = current_chunk.clone();
                    test_chunk.push(effective_word.clone());
                    let test_text = test_chunk.join(" ");
                    let test_emb = self.get_embedding(test_text.clone()).await.unwrap_or(vec![0.0; 384]);

                    // 🌟 [추가] 1차 핀볼(속성 매칭)에도 동사 페널티(verb_penalty) 및 단어 길이 가중치 적용
                    let word_count = test_chunk.len();
                    let v_sim = cosine_similarity(&test_emb, &verb_emb);
                    let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                    let verb_penalty = v_sim * beta;
                    let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };

                    let mut candidates: Vec<(String, f32)> = Vec::new();

                    for i in 0..prop_keys.len() {
                        if prop_is_filter_owned[i] { continue; }
                        // 🌟 [PHRASE MAX-POOL] 센트로이드 대신 변별 구 단위 최대 유사도.
                        //    이로써 'contains() 문자열 포함 시 +0.5' 라는 의미 판정 하드코딩과
                        //    매직 상수 0.5 를 동시에 제거합니다.
                        //    구가 원문과 동일하면 코사인이 1.0 이므로 보너스 없이 1순위가 확정됩니다.
                        let b_score = crate::utils::ai_utils::weighted_max_pool_sim(&test_emb, &prop_phrase_embs[i], &prop_phrase_weights[i]);
                        let p_score = cosine_similarity(&test_emb, &prej_embs[i]);

                        let score = b_score - (p_score * penalty_weight) - verb_penalty;

                        candidates.push((prop_keys[i].clone(), score));
                    }
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let current_best = candidates.first().map(|c| c.0.clone()).unwrap_or_default();
                    let current_max = candidates.first().map(|c| c.1).unwrap_or(-1.0);
                    let current_alternatives: Vec<(String, f32)> = candidates.iter().skip(1).take(5).cloned().collect();

                    let mut global_scores_log = String::new();
                    if !loaded_globals.is_empty() {
                        let mut g_scores = Vec::new();
                        for g_key in &loaded_globals {
                            if let Some(c) = candidates.iter().find(|x| &x.0 == g_key) {
                                g_scores.push(format!("{}: {:.4}", g_key, c.1));
                            }
                        }
                        global_scores_log = format!(" | Globals: {}", g_scores.join(", "));
                    }

                    let sec_prop = current_alternatives.first().map(|c| c.0.clone()).unwrap_or_default();
                    let sec_score = current_alternatives.first().map(|c| c.1).unwrap_or(-1.0);
                    emit_term(&format!("    🔍 [PLINKO SLIDE] '{}' -> 1st: [{}] ({:.4}) | 2nd: [{}] ({:.4}){}", test_text, current_best, current_max, sec_prop, sec_score, global_scores_log));

                    // Score Drop (Cliff) = Cut & Drop into Slot
                    if current_max < prev_max_score && !current_chunk.is_empty() {
                        emit_term(&format!("    📉 [CLIFF DETECTED] Score dropped ({:.4} -> {:.4}). End of semantic chunk.", prev_max_score, current_max));
                        
                        // 🌟 임계값을 0.20으로 상향 조정하여 무의미한 단어가 특정 속성으로 맵핑되는 현상 방지
                        if prev_max_score > 0.20 && !best_prop_for_chunk.is_empty() {
                            emit_term(&format!("      📥 [DROPPED INTO SLOT] '{}' belongs to property [{}]", current_chunk.join(" "), best_prop_for_chunk));
                            plinko_matches.push(PlinkoMatch {
                                chunk: current_chunk.join(" "),
                                best_prop: best_prop_for_chunk.clone(),
                                best_score: prev_max_score,
                                alternatives: prev_alternatives.clone(),
                                all_scores: prev_all_scores.clone(),
                            });
                        } else {
                            emit_term(&format!("      🗑️ [SKIPPED] Score {:.4} is too low (Threshold: 0.20). Ignored.", prev_max_score));
                        }
                        
                        // Reset Window
                        current_chunk = vec![effective_word.clone()];
                        let reset_emb = self.get_embedding(effective_word.clone()).await.unwrap_or(vec![0.0; 384]);
                        
                        // 🌟 [추가] 리셋 윈도우의 단일 단어에도 동사 페널티 일관되게 적용
                        let r_v_sim = cosine_similarity(&reset_emb, &verb_emb);
                        let r_verb_penalty = r_v_sim * 0.05;
                        let r_penalty_weight = 0.3;

                        let mut r_candidates: Vec<(String, f32)> = Vec::new();
                        for i in 0..prop_keys.len() {
                            if prop_is_filter_owned[i] { continue; }
                            // 🌟 [PHRASE MAX-POOL] 리셋 윈도우도 동일하게 구 단위 최대 유사도로 통일합니다.
                            //    '가디건' 은 title 뱅크에 편입된 semantic 앵커 구('의류명')와 직접 경쟁하게 되어
                            //    더 이상 tags/brand_name 으로 흘러가지 않습니다.
                            let b_score = crate::utils::ai_utils::weighted_max_pool_sim(&reset_emb, &prop_phrase_embs[i], &prop_phrase_weights[i]);
                            let p_score = cosine_similarity(&reset_emb, &prej_embs[i]);

                            let score = b_score - (p_score * r_penalty_weight) - r_verb_penalty;

                            r_candidates.push((prop_keys[i].clone(), score));
                        }
                        r_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        
                        let r_max = r_candidates.first().map(|c| c.1).unwrap_or(-1.0);
                        let r_best = r_candidates.first().map(|c| c.0.clone()).unwrap_or_default();
                        let r_alts: Vec<(String, f32)> = r_candidates.iter().skip(1).take(5).cloned().collect();
                        let r_sec_prop = r_alts.first().map(|c| c.0.clone()).unwrap_or_default();
                        let r_sec_score = r_alts.first().map(|c| c.1).unwrap_or(-1.0);

                        let mut r_global_scores_log = String::new();
                        if !loaded_globals.is_empty() {
                            let mut r_g_scores = Vec::new();
                            for g_key in &loaded_globals {
                                if let Some(c) = r_candidates.iter().find(|x| &x.0 == g_key) {
                                    r_g_scores.push(format!("{}: {:.4}", g_key, c.1));
                                }
                            }
                            r_global_scores_log = format!(" | Globals: {}", r_g_scores.join(", "));
                        }

                        prev_max_score = r_max;
                        best_prop_for_chunk = r_best.clone();
                        prev_alternatives = r_alts;
                        prev_all_scores = r_candidates;
                        emit_term(&format!("    🔄 [WINDOW RESET] Started new chunk '{}' -> 1st: {} ({:.4}) | 2nd: {} ({:.4}){}", effective_word, r_best, r_max, r_sec_prop, r_sec_score, r_global_scores_log));
                    } else {
                        current_chunk.push(effective_word.clone());
                        prev_max_score = current_max;
                        best_prop_for_chunk = current_best;
                        prev_alternatives = current_alternatives;
                        prev_all_scores = candidates;
                    }
                }

                // 🌟 [EXCLUSIVE PROPERTY ASSIGNMENT + QWEN3 VERIFICATION (1st)]
                //    기존 구조는 HashMap<속성, Vec<청크>> 라서 한 속성에 청크가 무한히 쌓였고,
                //    Double Plinko 가 v.join(" | ") 로 병합하는 순간 서로 다른 의미가 한 값이 되었습니다.
                //    (로그: '베이지' 와 '가디건' 이 둘 다 color 로 들어가 value = "베이지 가디건")
                //    이제 한 속성은 정확히 한 청크만, 한 청크는 정확히 한 속성만 가져갑니다.
                if !plinko_matches.is_empty() {
                    emit_term("    🧠 [EXCLUSIVE ASSIGN + QWEN3 VERIFICATION (1st)] Verifying property mappings...");

                    // 🌟 [TEMPORAL / NUMERIC STRUCTURE PRE-GATE]
                    //    형식 검증을 '배정 전' 에 제대로 수행하려면, 그 청크가
                    //      ① 시간·계절 의도인가            → Date 필드 후보 자격 부여
                    //      ② (숫자 + 비교 표현) 구조인가   → 문자열/열거형 필드 후보 자격 박탈
                    //    를 먼저 결정론/코사인으로 확정해야 합니다.
                    //    ① 은 이미 만들어 둔 dynamic_filter_defs(time_filters / season_filters / metrics / operators)
                    //       벡터 뱅크를 그대로 재사용하므로 새 임베딩 자원도, 새 상수도 들지 않습니다.
                    //    ② 는 split_numeric_and_comparator 라는 순수 문자열 구조 파싱 + operators 뱅크 코사인입니다.
                    //    LLM 호출은 단 한 번도 추가되지 않으며, 오히려 후보 목록이 정화되어
                    //    기존 Qwen3 검증 호출의 정확도가 올라갑니다.
                    let mut temporal_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();
                    let mut numeric_cmp_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();

                    for pm in plinko_matches.iter() {
                        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                            return Ok(json!({ "context": [], "cancelled": true }));
                        }

                        let c_emb = self.get_embedding(pm.chunk.clone()).await.unwrap_or(vec![0.0; 384]);

                        // ① 시간성 판정 : 구 단위 Max-Pool 로 time/season 뱅크와 비교
                        //    temporal_phrase_embs 중 time/season 소속 구와의 Max-Pool 이
                        //    다른 필터 카테고리 센트로이드 최대보다 높으면 temporal 확정.
                        //    절대 임계치 없이 'temporal Max-Pool > rival centroid max' 상대 비교만 사용.
                        let mut temporal_pool = 0.0f32;
                        if !temporal_phrase_embs.is_empty() {
                            for te in &temporal_phrase_embs {
                                if te.iter().all(|&v| v == 0.0) { continue; }
                                let s = cosine_similarity(&c_emb, te);
                                if s > temporal_pool { temporal_pool = s; }
                            }
                        }
                        // 🌟 [RIVAL FIX] rival 비교도 구 뱅크 Max-Pool 로 통일합니다.
                        //    기존 센트로이드(dynamic_bias_embs) 비교는 한국어 2음절 단어와
                        //    영어 문장 센트로이드 간 코사인이 구조적으로 낮아
                        //    temporal 이 우세해도 rival 가 과대평가되는 문제가 있었습니다.
                        let mut rival_best = f32::MIN;
                        if !all_filter_embs.is_empty() {
                            for (cat, _key, emb) in &all_filter_embs {
                                if cat == "time_filters" || cat == "season_filters" { continue; }
                                if emb.iter().all(|&v| v == 0.0) { continue; }
                                let s = cosine_similarity(&c_emb, emb);
                                if s > rival_best { rival_best = s; }
                            }
                        }
                        if rival_best == f32::MIN { rival_best = 0.0; }
                        // 🌟 temporal Max-Pool 이 rival Max-Pool 보다 높으면 확정.
                        if temporal_pool > rival_best && temporal_pool > 0.0 {
                            temporal_chunks.insert(pm.chunk.trim().to_string());
                            emit_term(&format!("      🕒 [TEMPORAL PRE-GATE] '{}' 는 시간/계절 의도가 우세합니다. (TemporalMaxPool {:.4} > RivalMaxPool {:+.4}) → Date 필드 후보 자격 부여", pm.chunk, temporal_pool, rival_best));
                        }

                        // ② 수치 비교 구조 판정 : 구 단위 Max-Pool 로 operators 뱅크와 비교
                        //    "이하로" vs embed("less than or equal") / embed("under") / embed("no more than")
                        //    의 Max-Pool 이 top/bottom 구 Max-Pool 보다 높으면 비교 연산자 확정.
                        if let Some((_num, cmp_part)) = crate::utils::ai_utils::split_numeric_and_comparator(&pm.chunk) {
                            if !cmp_part.trim().is_empty() {
                                let cmp_emb = self.get_embedding(cmp_part.clone()).await.unwrap_or(vec![0.0; 384]);
                                let mut cmp_pool = 0.0f32;
                                let mut rank_pool = 0.0f32;
                                if !op_phrase_embs.is_empty() {
                                    for (oi, oe) in op_phrase_embs.iter().enumerate() {
                                        if oe.iter().all(|&v| v == 0.0) { continue; }
                                        let s = cosine_similarity(&cmp_emb, oe);
                                        if op_phrase_is_rank[oi] {
                                            if s > rank_pool { rank_pool = s; }
                                        } else {
                                            if s > cmp_pool { cmp_pool = s; }
                                        }
                                    }
                                }
                                // 폴백: 구 뱅크가 비어 있으면 기존 센트로이드 경로 사용
                                if op_phrase_embs.is_empty() {
                                    for i in 0..dynamic_filter_defs.len() {
                                        if dynamic_filter_defs[i].category != "operators" { continue; }
                                        let b = cosine_similarity(&cmp_emb, &dynamic_bias_embs[i]);
                                        let p = cosine_similarity(&cmp_emb, &dynamic_prej_embs[i]);
                                        let s = b - p;
                                        match dynamic_filter_defs[i].key.as_str() {
                                            "top" | "bottom" => { if s > rank_pool { rank_pool = s; } },
                                            _ => { if s > cmp_pool { cmp_pool = s; } },
                                        }
                                    }
                                }
                                if cmp_pool > rank_pool && cmp_pool > 0.0 {
                                    numeric_cmp_chunks.insert(pm.chunk.clone());
                                    emit_term(&format!("      🔢 [NUMERIC PRE-GATE] '{}' 는 (숫자 + 비교 표현) 구조입니다. (CmpMaxPool {:.4} > RankMaxPool {:.4}) → 문자열/열거형 필드 후보 자격 박탈", pm.chunk, cmp_pool, rank_pool));
                                }
                            }
                        }
                    }

                    // ── 1) 형식 게이트 : 배정 전(행렬 구축 시점)에 값의 생김새부터 검증합니다.
                    let chunk_count = plinko_matches.len();
                    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; chunk_count]; prop_keys.len()];
                    let mut gate_dropped: Vec<String> = Vec::new();
                    for (ci, pm) in plinko_matches.iter().enumerate() {
                        // 🌟 [TRIM FIX] temporal_chunks 에 등록 시 trim 을 적용했으므로
                        //    조회 시에도 동일하게 trim 하여 공백 차이로 인한 lookup 실패를 방지합니다.
                        let chunk_trimmed = pm.chunk.trim().to_string();
                        let t_hint = temporal_chunks.contains(&chunk_trimmed);
                        let n_hint = numeric_cmp_chunks.contains(&chunk_trimmed);
                        if t_hint {
                            emit_term(&format!("      🕒 [TEMPORAL HINT ACTIVE] '{}' → Date 필드 후보 자격 부여 확인", chunk_trimmed));
                        }
                        if n_hint {
                            emit_term(&format!("      🔢 [NUMERIC HINT ACTIVE] '{}' → 문자열/열거형 후보 자격 박탈 확인", chunk_trimmed));
                        }
                        for (name, sc) in &pm.all_scores {
                            let pi = match prop_keys.iter().position(|p| p == name) { Some(v) => v, None => continue };
                            if is_filter_owned(name) { continue; }
                            if !crate::utils::ai_utils::query_chunk_matches_property_ext(name, &chunk_trimmed, t_hint, n_hint) {
                                if gate_dropped.len() < 12 && *sc > 0.20 {
                                    gate_dropped.push(format!("{}→{}({:.4})", chunk_trimmed, name, sc));
                                }
                                continue;
                            }
                            matrix[pi][ci] = *sc;
                        }
                    }
                    if !gate_dropped.is_empty() {
                        emit_term(&format!("      🚧 [FORMAT GATE] 값 생김새가 맞지 않아 배정 후보에서 제외: {:?}", gate_dropped));
                    }

                    // ── 2) 배타 배정 : 유효한 모든 (속성 × 청크) 주장을 절대 점수 순으로 그리디 배정합니다.
                    //       기존 exclusive_assign_by_score(matrix, 0.0, 0.0) 는 rival 이
                    //       '같은 청크에 대한 다른 속성의 최고 점수' 였기 때문에
                    //       margin >= 0 이 곧 'own 이 그 청크의 argmax' 를 의미했고,
                    //       그 결과 각 청크는 argmax 속성 하나에만 주장을 낼 수 있었습니다.
                    //       그 속성을 더 높은 점수의 청크가 가져가면 차선책 없이 굶어 죽습니다.
                    //       (로그: '가디건'/'무거운'/'제품중에서'/'제품으로'/'중에서'/'메세지도'/'보여줘' 전멸)
                    //       greedy_exclusive_assign 은 margin 을 정렬이 아니라 보고 지표로만 쓰므로
                    //       선점당한 청크가 즉시 자기 점수표의 다음 '형식 통과 + 미선점' 속성으로 이동합니다.
                    let assign = crate::utils::ai_utils::greedy_exclusive_assign(&matrix);

                    let mut chunk_owner: Vec<Option<(String, f32, f32)>> = vec![None; chunk_count];
                    let mut claimed_props: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for (pi, a) in assign.iter().enumerate() {
                        if let Some((ci, own, margin)) = a {
                            chunk_owner[*ci] = Some((prop_keys[pi].clone(), *own, *margin));
                            claimed_props.insert(prop_keys[pi].clone());
                        }
                    }

                    let covered = chunk_owner.iter().filter(|c| c.is_some()).count();
                    emit_term(&format!("      📶 [ASSIGN COVERAGE] 청크 {}개 중 {}개 배정 확보 (그리디 최대 커버리지)", chunk_count, covered));

                    // ── 3) Qwen3 재판정 : 후보 목록을 '형식 통과 + 미선점' 속성으로만 한정합니다.
                    //       LLM 호출 수는 늘지 않고(청크당 최대 1회, 기존과 동일),
                    //       대신 모델이 볼 수 있는 선택지 자체를 결정론으로 좁혀 오답 경로를 물리적으로 없앱니다.
                    let mut validated_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
                    let mut alt_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

                    for (ci, pm) in plinko_matches.iter().enumerate() {
                        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                            return Ok(json!({ "context": [], "cancelled": true }));
                        }

                        let (mut owner_prop, owner_score, owner_margin) = match &chunk_owner[ci] {
                            Some(v) => (v.0.clone(), v.1, v.2),
                            None => {
                                let sec = pm.alternatives.first()
                                    .map(|c| format!("{} ({:.4})", c.0, c.1))
                                    .unwrap_or_else(|| "-".to_string());
                                emit_term(&format!("      ⚪ [UNASSIGNED CHUNK] '{}' 는 형식 통과 후보가 문서 스키마에 하나도 없어 조건에서 제외합니다. FTS 검색어로는 보존됩니다. (Plinko 1st: {} {:.4} / 2nd: {})", pm.chunk, pm.best_prop, pm.best_score, sec));
                                unassigned_chunks.push(pm.chunk.clone());
                                continue;
                            }
                        };

                        // 🌟 [NEGATIVE MARGIN GUARD] margin 이 음수라는 것은
                        //    '이 청크의 argmax 조차 아닌 필드에 억지로 배정되었다' 는 뜻입니다.
                        //    그리디가 커버리지를 최대화하면서 의미 없는 청크까지 채워 넣은 부작용이며,
                        //    로그의 '제품중에서'(-0.0091) '제품으로'(-0.0020) '보여줘'(-0.0344) 가 여기 해당합니다.
                        //    조건으로 확정하는 대신 FTS 검색어로만 보존하는 것이 리콜에 유리합니다.
                        //    (margin 은 같은 청크 내 1위-2위 차이이므로 새 임계치가 아니라 부호 판정입니다)
                        if owner_margin < 0.0 {
                            emit_term(&format!("      ⚖️ [NEGATIVE MARGIN DROP] '{}' → [{}] | Margin: {:+.4} < 0. 자기 argmax 가 아닌 억지 배정이므로 조건에서 제외하고 FTS 검색어로 보존합니다.", pm.chunk, owner_prop, owner_margin));
                            claimed_props.remove(&owner_prop);
                            unassigned_chunks.push(pm.chunk.clone());
                            continue;
                        }

                        emit_term(&format!("      🔗 [EXCLUSIVE ASSIGN] '{}' → [{}] | Score: {:.4} | Margin: {:+.4}", pm.chunk, owner_prop, owner_score, owner_margin));

                        // 🌟 [ALTERNATE AXIS] 이 청크가 갈 수 있었던 '형식 통과 + 미선점' 차순위 속성들.
                        //    확정이 틀렸을 때 STAGE-3 이 이 목록으로 대안 컨텍스트를 발행합니다.
                        let mut allowed: Vec<(String, f32)> = Vec::new();
                        for (name, sc) in &pm.all_scores {
                            if name == &owner_prop { continue; }
                            if claimed_props.contains(name) { continue; }
                            if is_filter_owned(name) { continue; }
                            if !crate::utils::ai_utils::query_chunk_matches_property(name, &pm.chunk) { continue; }
                            allowed.push((name.clone(), *sc));
                            if allowed.len() >= 5 { break; }
                        }

                        // 🌟 [DETERMINISTIC BYPASS] 기존의 절대 임계치(0.70)는 매직 상수였고,
                        //    다국어 임베딩의 짧은 한국어 청크는 0.5~0.7 대역에 촘촘히 뭉쳐 있어
                        //    (로그 실측: 0.5265 ~ 0.6798) 사실상 거의 모든 청크가 LLM 을 거치며
                        //    오히려 오판 기회를 늘렸습니다.
                        //    '형식 게이트를 통과했고 아직 선점되지 않은 대안이 하나도 없다' 는 것은
                        //    선택지가 물리적으로 하나뿐이라는 결정론적 사실이므로 LLM 에게 물을 이유가 없습니다.
                        //    (LLM 호출 수는 늘지 않고 오히려 줄어들며, 임계치가 사라집니다)
                        //
                        //    🌟 [COLOR SINK GUARD] 단, color 로 배정된 청크가 '실제로 색상을 나타내는지'는
                        //    대안이 없는 경우에도(is_empty) LLM 에게 한 번 물어서 확인합니다.
                        //    '남긴', '팔린', '많이' 같은 비색상 단어가 color 5구 뱅크의 구조적 편향으로
                        //    argmax 가 되는 것을 LLM 이 걸러냅니다. (호출 추가 없음, 기존 슬롯 활용)
                        if allowed.is_empty() {
                            if owner_prop != "color" {
                                emit_term(&format!("      ⚡ [BYPASS] 형식 통과 대안이 존재하지 않아 [{}] 로 결정론 확정합니다. ('{}' | Score {:.4})", owner_prop, pm.chunk, owner_score));
                                validated_map.insert(owner_prop.clone(), vec![pm.chunk.clone()]);
                                alt_map.insert(owner_prop.clone(), Vec::new());
                                continue;
                            } else {
                                emit_term(&format!("      🛡️ [COLOR SINK GUARD] '{}' 청크가 대안 없이 color 에 배정되었으나, 비색상 단어인지 확인하기 위해 Qwen3 검증을 거칩니다.", pm.chunk));
                            }
                        }

                        // 🌟 [FILTER CONTEXT INJECTION] Qwen3 검증 프롬프트에 필터 카테고리 정보를 주입합니다.
                        //    '팔린 제품' 같은 복합 청크에서 '팔린' 이 status 필터 의도임을
                        //    Qwen3 가 인식할 수 있도록 컨텍스트를 제공합니다.
                        //    LLM 호출 수는 늘지 않고(청크당 최대 1회, 기존과 동일),
                        //    프롬프트 내용만 풍부해져 판정 정확도가 올라갑니다.
                        //    bias.json 수정 없이 all_filter_embs 의 코사인 결과만 동적으로 전달합니다.
                        let filter_context_hint = if !all_filter_embs.is_empty() {
                            let chunk_emb_hint = self.get_embedding(pm.chunk.trim().to_string()).await.unwrap_or(vec![0.0; 384]);
                            let mut hints: Vec<String> = Vec::new();
                            for (cat, key, emb) in &all_filter_embs {
                                if emb.iter().all(|&v| v == 0.0) { continue; }
                                let s = cosine_similarity(&chunk_emb_hint, emb);
                                if s > 0.45 {
                                    hints.push(format!("{}.{}", cat, key));
                                }
                            }
                            if hints.is_empty() { String::new() } else { format!("\n[POSSIBLE FILTER INTENTS] This chunk may also express filter intents: {}. If the chunk combines a filter intent with a property value, prioritize the property value and note the filter intent separately.", hints.join(", ")) }
                        } else {
                            String::new()
                        };
                        let prompt = format!("{}{}", crate::prompts::verify_property_with_alternatives_prompt(
                            &pm.chunk, &owner_prop, owner_score, &allowed
                        ), filter_context_hint);

                        let mut picked: Option<String> = None;
                        if let Ok(response) = self.call_qwen3_verification_model(&prompt, Some(cancel_token.clone())).await {
                            if let Ok(result) = serde_json::from_str::<Value>(&response) {
                                let mut cand_list: Vec<String> = Vec::new();
                                if let Some(arr) = result.get("suggested_properties").and_then(|v| v.as_array()) {
                                    for s in arr {
                                        if let Some(t) = s.as_str() { cand_list.push(t.to_string()); }
                                    }
                                } else if let Some(s) = result.get("suggested_property").and_then(|v| v.as_str()) {
                                    cand_list.push(s.to_string());
                                }
                                for c in cand_list {
                                    if c == owner_prop { picked = Some(c); break; }
                                    if allowed.iter().any(|(n, _)| n == &c) { picked = Some(c); break; }
                                    emit_term(&format!("      🚫 [LLM REJECT] '{}' 에 대한 제안 [{}] 은 형식 불일치이거나 다른 청크가 이미 선점한 속성이라 폐기합니다.", pm.chunk, c));
                                }
                            }
                        }

                        if let Some(new_prop) = picked {
                            if new_prop != owner_prop {
                                // 🌟 [CORRECTION COSINE VERIFY] Qwen3 가 교정한 속성이
                                //    원본 속성보다 청크와 실제로 더 관련 있는지 코사인으로 검증합니다.
                                //    (로그: '남긴' → color → Qwen3 교정 → name. 그러나 name 도 '남긴' 과 무관)
                                //    교정 후 코사인이 교정 전보다 낮으면 교정을 폐기하고 UNASSIGN 합니다.
                                //    이 검사가 있어야 '남긴'→color→name 같은 연쇄 오배정이 차단됩니다.
                                let chunk_emb_verify = self.get_embedding(pm.chunk.trim().to_string()).await.unwrap_or(vec![0.0; 384]);
                                let old_pi = prop_keys.iter().position(|p| p == &owner_prop);
                                let new_pi = prop_keys.iter().position(|p| p == &new_prop);
                                let is_degraded = match (old_pi, new_pi) {
                                    (Some(opi), Some(npi)) => {
                                        crate::utils::ai_utils::correction_cosine_degraded(
                                            &chunk_emb_verify,
                                            &prop_phrase_embs[opi], &prop_phrase_weights[opi],
                                            &prop_phrase_embs[npi], &prop_phrase_weights[npi],
                                        )
                                    },
                                    _ => false,
                                };
                                if is_degraded {
                                    emit_term(&format!("      🚫 [CORRECTION DEGRADED] '{}' 에 대한 Qwen3 교정 [{}] → [{}] 은 코사인 열화로 폐기합니다. UNASSIGN 처리.", pm.chunk, owner_prop, new_prop));
                                    claimed_props.remove(&owner_prop);
                                    unassigned_chunks.push(pm.chunk.clone());
                                    continue;
                                }
                                emit_term(&format!("      🔄 Property [{}] corrected as [{}] for '{}'", owner_prop, new_prop, pm.chunk));
                                claimed_props.remove(&owner_prop);
                                claimed_props.insert(new_prop.clone());
                                // 확정에서 밀려난 기존 1순위는 그대로 대안 축의 선두가 됩니다.
                                if !allowed.iter().any(|(n, _)| n == &owner_prop) {
                                    allowed.insert(0, (owner_prop.clone(), owner_score));
                                }
                                allowed.retain(|(n, _)| n != &new_prop);
                                owner_prop = new_prop;
                            } else {
                                emit_term(&format!("      ✅ Property [{}] confirmed for '{}'", owner_prop, pm.chunk));
                            }
                        }

                        validated_map.insert(owner_prop.clone(), vec![pm.chunk.clone()]);
                        alt_map.insert(owner_prop.clone(), allowed.iter().map(|(n, _)| n.clone()).collect());
                    }

                    // 🌟 이제 한 속성 슬롯에는 정확히 한 청크만 담깁니다.
                    //    Double Plinko 의 v.join(" | ") 와 deterministic_condition_value 가
                    //    구조적으로 두 의미를 합칠 수 없게 되었습니다.
                    plinko_map = validated_map;
                    plinko_alternates = alt_map;
                }

                // 🌟 [SEASON / TIME EXACT MATCH — 확정 결과 재확인]
                //    실제 감지는 Plinko 진입 전([FILTER TERM DROP])에서 이미 수행되었습니다.
                //    여기서는 그 결과를 세그먼트 텍스트 기준으로 다시 확정하여
                //    결정론 시간 가이드와 STAGE-3 메타데이터에 전달합니다.
                //    Plinko 가 이 단어들을 아예 보지 못하므로
                //    color / region_restrictions 로 흘러가는 경로가 물리적으로 존재하지 않습니다.
                let mut exact_season_key = String::new();
                let mut exact_time_key = String::new();
                for w in current_text.split_whitespace() {
                    if exact_season_key.is_empty() {
                        if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("season_filters", w) {
                            emit_term(&format!("  🌤️ [SEASON EXACT MATCH] '{}' ∈ season_filters.{}.exact_match → 코사인 경쟁 없이 확정합니다.", w, k));
                            exact_season_key = k;
                        }
                    }
                    if exact_time_key.is_empty() {
                        if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("time_filters", w) {
                            emit_term(&format!("  🕒 [TIME EXACT MATCH] '{}' ∈ time_filters.{}.exact_match → 코사인 경쟁 없이 확정합니다.", w, k));
                            exact_time_key = k;
                        }
                    }
                }

                // 🌟 [IGNORE VECTOR CHECK] 현재 청크 전체가 명령어/분석 요청(ignore)에 해당하는지 검증
                //    🌟 [DOMAIN GUARD] STAGE-1 이 이미 유효 도메인으로 확정한 세그먼트를
                //    STAGE-2 가 뒤집는 것은 구조적 모순입니다.
                //    (로그: '이벤트로 판매된' 은 STAGE-1 에서 event 확정 + Qwen3 가 coupon→event 교정까지 했는데
                //     IGNORE 0.4526 으로 통째로 소멸했습니다. bias.json 의 ignore.bias 에 있는
                //     'show me, display, list out, find out' 등이 '판매된' 과 우연히 공명한 결과입니다)
                //    STAGE-1 이 ignore 가 아닌 도메인을 확정했다면 IGNORE 체크 자체를 건너뜁니다.
                let stage1_confirmed = seg_type != "ignore" && !seg_type.is_empty();
                let mut is_ignore_chunk = false;

                if stage1_confirmed {
                    emit_term(&format!("  🛡️ [IGNORE GUARD] STAGE-1 이 '{}' 도메인으로 확정한 세그먼트이므로 IGNORE 체크를 건너뜁니다.", seg_type));
                } else {
                    let chunk_full_emb = self.get_embedding(current_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    let chunk_word_count = current_text.split_whitespace().count();
                    let v_sim = cosine_similarity(&chunk_full_emb, &verb_emb);
                    let beta = if chunk_word_count <= 2 { 0.05 } else { 0.10 };
                    let penalty_weight = if chunk_word_count <= 2 { 0.3 } else { 0.7 };
                    let _ = (v_sim, beta);

                    if let Some(ignore_obj) = crate::parsing::BIAS_DICT.get("ignore").and_then(|p| p.as_object()) {
                        let s_bias = ignore_obj.get("bias").and_then(|v| v.as_str()).unwrap_or("");
                        let s_prej = ignore_obj.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                        if !s_bias.is_empty() {
                            let ignore_bias_emb = self.get_embedding(s_bias.to_string()).await.unwrap_or(vec![0.0; 384]);
                            let ignore_prej_emb = self.get_embedding(s_prej.to_string()).await.unwrap_or(vec![0.0; 384]);

                            let b_score = cosine_similarity(&chunk_full_emb, &ignore_bias_emb);
                            let p_score = cosine_similarity(&chunk_full_emb, &ignore_prej_emb);
                            let ignore_score = b_score - (p_score * penalty_weight);

                            if ignore_score > 0.4 {
                                emit_term(&format!("  🚫 [IGNORE VECTOR CHECK] Chunk '{}' identified as IGNORE (Score: {:.4}). Skipping LLM processing.", current_text, ignore_score));
                                is_ignore_chunk = true;
                            }
                        }
                    }
                }

                if is_ignore_chunk {
                    if let Some(obj) = seg.as_object_mut() {
                        obj.insert("type".to_string(), json!("ignore")); // 마스터 병합에서 빠지도록 타입 강제 변환
                        obj.insert("condition".to_string(), json!({}));
                    }
                    continue;
                }

                // 🌟 Formatting Plinko Fragments & [2차 선택] Double Plinko for All Dynamic Filters
                let mut fragments_text = String::new();
                let mut prop_to_op: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                let mut prop_to_exact_val: std::collections::HashMap<String, String> = std::collections::HashMap::new(); // 🌟 숫자 할루시네이션 방지용 원본 값 저장소

                // 🌟 [FORCED ROUTE SEED] Plinko 진입 전에 필터로 확정된 단어들의 결과를
                //    전역 필터 값의 초기값으로 삼습니다.
                //    이 단어들은 plinko_map 에 들어가지 않으므로 2차 Plinko 가 볼 수 없고,
                //    시딩이 없으면 substantial / find 가 영원히 빈 값으로 남습니다.
                let pick_forced = |cat: &str| -> String {
                    let mut best = f32::MIN;
                    let mut key = String::new();
                    for (_, c, k, s) in &forced_filter_routes {
                        if c != cat { continue; }
                        if *s > best { best = *s; key = k.clone(); }
                    }
                    key
                };
                let mut best_status_global = pick_forced("status_filters");
                let mut best_sub_global    = pick_forced("substantial_filters");
                let mut best_find_global   = pick_forced("find_filters");
                let mut best_time_global   = pick_forced("time_filters");
                let mut best_season_global = pick_forced("season_filters");
                if !best_sub_global.is_empty() || !best_find_global.is_empty() {
                    emit_term(&format!(
                        "    🧲 [FORCED ROUTE SEED] substantial='{}' | find='{}' (추상 수식어 라우팅 결과 확정)",
                        best_sub_global, best_find_global
                    ));
                }
                let mut filter_candidates: std::collections::HashMap<String, Vec<(String, f32)>> = std::collections::HashMap::new();
                // 🌟 [NUMERIC REROUTE PLAN] (원래속성, 대상Numeric속성, 연산자, 숫자값)
                //    문자열로 잘못 굳은 수치 조건을 최종 조립 직전에 교체합니다.
                let mut numeric_reroutes: Vec<(String, String, String, String)> = Vec::new();

                emit_term("\n  🎯 [DOUBLE PLINKO (2nd)] Matching attributes and operators...");

                for (k, v) in &plinko_map {
                    let combined_chunk = v.join(" | ");
                    
                    let chunk_emb = self.get_embedding(combined_chunk.clone()).await.unwrap_or(vec![0.0; 384]);
                    let cw_count = v.len();
                    let v_sim_local = cosine_similarity(&chunk_emb, &verb_emb);
                    let local_beta = if cw_count <= 2 { 0.05 } else { 0.10 };
                    let local_vp = v_sim_local * local_beta;
                    let local_pw = if cw_count <= 2 { 0.3 } else { 0.7 };

                    // 🌟 [EXCLUSIVE CLAIM SCORE] 이 청크를 이미 선점한 '스키마 속성'의 코사인 점수입니다.
                    //    status / substantial / find / time / season 은 청크 '자체'를 자기 값으로 가져가는
                    //    경쟁 해석이므로, 속성 선점 점수를 넘지 못하면 후보 자격이 없습니다.
                    //    (로그: '베이지' 는 color 0.8332 로 확정되었는데도 status_filters 가 'remove' 를 들고 올라와
                    //     최종 SQL 에 status = 11 이 박히면서 검색 리콜이 통째로 무너졌습니다)
                    //    이 비교가 절대 임계치 0.15 를 대체하므로 매직 상수를 제거합니다.
                    let prop_claim_score = match prop_keys.iter().position(|p| p == k) {
                        Some(pi) => {
                            let own = crate::utils::ai_utils::weighted_max_pool_sim(&chunk_emb, &prop_phrase_embs[pi], &prop_phrase_weights[pi]);
                            let pj = cosine_similarity(&chunk_emb, &prej_embs[pi]);
                            own - (pj * local_pw) - local_vp
                        },
                        None => f32::MIN,
                    };

                    let mut local_filter_candidates: std::collections::HashMap<String, Vec<(String, f32)>> = std::collections::HashMap::new();

                    // 🌟 Part 1에서 준비된 통합 벡터(dynamic_filter_defs) 순회
                    for i in 0..dynamic_filter_defs.len() {
                        let def = &dynamic_filter_defs[i];
                        let b_score = cosine_similarity(&chunk_emb, &dynamic_bias_embs[i]);
                        let p_score = cosine_similarity(&chunk_emb, &dynamic_prej_embs[i]);
                        let score = b_score - (p_score * local_pw) - local_vp;

                        // 🌟 [CLAIM CATEGORY] 청크 자체를 값으로 가져가는 카테고리는 속성과 배타 경쟁합니다.
                        //    operators / metrics / option_filters 는 '이미 확정된 속성을 어떻게 비교할지'를
                        //    서술하는 수식어이므로 경쟁 대상이 아니며 게이트를 적용하지 않습니다.
                        let is_claim_category = matches!(
                            def.category.as_str(),
                            "status_filters" | "substantial_filters" | "find_filters" | "time_filters" | "season_filters"
                        );

                        if is_claim_category && score <= prop_claim_score {
                            continue;
                        }

                        local_filter_candidates.entry(def.category.clone()).or_insert_with(Vec::new).push((def.key.clone(), score));
                    }

                    for (_, cands) in local_filter_candidates.iter_mut() {
                        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    }

                    let best_op = local_filter_candidates.get("operators").and_then(|c| c.first()).map(|c| c.0.clone()).unwrap_or_else(|| "eq".to_string());
                    let best_metric = local_filter_candidates.get("metrics").and_then(|c| c.first()).map(|c| c.0.clone()).unwrap_or_else(|| "string".to_string());
                    
                    if let Some(cands) = local_filter_candidates.get("status_filters") {
                        if let Some(c) = cands.first() { if best_status_global.is_empty() { best_status_global = c.0.clone(); } }
                        filter_candidates.insert("status_filters".to_string(), cands.clone());
                    }
                    if let Some(cands) = local_filter_candidates.get("substantial_filters") {
                        if let Some(c) = cands.first() {
                            if best_sub_global.is_empty() {
                                best_sub_global = c.0.clone();
                                emit_term(&format!("    📏 [SUBSTANTIAL MATCH] '{}' → substantial_filters.{} (Score: {:+.4})", combined_chunk, c.0, c.1));
                            }
                        }
                        filter_candidates.insert("substantial_filters".to_string(), cands.clone());
                    }
                    if let Some(cands) = local_filter_candidates.get("find_filters") {
                        if let Some(c) = cands.first() {
                            if best_find_global.is_empty() {
                                best_find_global = c.0.clone();
                                emit_term(&format!("    🔍 [FIND MATCH] '{}' → find_filters.{} (Score: {:+.4})", combined_chunk, c.0, c.1));
                            }
                        }
                        filter_candidates.insert("find_filters".to_string(), cands.clone());
                    }
                    if let Some(cands) = local_filter_candidates.get("time_filters") {
                        if let Some(c) = cands.first() { if best_time_global.is_empty() { best_time_global = c.0.clone(); } }
                        filter_candidates.insert("time_filters".to_string(), cands.clone());
                    }
                    if let Some(cands) = local_filter_candidates.get("season_filters") {
                        if let Some(c) = cands.first() { if best_season_global.is_empty() { best_season_global = c.0.clone(); } }
                        filter_candidates.insert("season_filters".to_string(), cands.clone());
                    }

                    // 🌟 [SCHEMA OVERRIDE] Vector 모델의 예측을 실제 DB 스키마 검증으로 덮어씁니다.
                    let actual_db_type = prop_types.get(k).copied().unwrap_or("String");
                    
                    let mut final_op = best_op.clone();
                    let mut final_metric = best_metric.clone();

                    // DB 스키마가 숫자나 날짜가 아닌 일반 문자열(String)인 경우
                    // 아예 통째로 스킵(continue)하지 않고, 연산자를 'contains'(부분 일치/FTS) 및 'string'으로 강제 고정하여 문맥에 포함시킵니다.
                    if actual_db_type != "Number" {
                        let is_date_field = k.contains("date") || k.contains("time") || k.ends_with("_at");
                        if !is_date_field {
                            // 🌟 [FTS FIX] 상품명, 카테고리 등 텍스트 검색은 'eq'가 아닌 'contains'로 처리해야 정상적인 Full Text Search 결과가 나옵니다.
                            emit_term(&format!("    ⏭️ [SCHEMA ADJUST] Property [{}] is strictly defined as '{}'. Adjusting operator to 'contains' (FTS).", k, actual_db_type));
                            final_op = "contains".to_string();
                            final_metric = "string".to_string();
                        }
                        // 🌟 [DETERMINISTIC VALUE BIND] 문자열/날짜 계열도 값은 '벡터가 짚어준 원문 청크' 그 자체입니다.
                        //    기존 코드는 Number 일 때만 prop_to_exact_val 에 등록했기 때문에
                        //    문자열 속성은 복구 경로가 통째로 죽어 있었습니다.
                        //    (로그: color 조건에 value 키가 없고, brand_name 조건은 아예 소멸)
                        let literal_val = crate::utils::ai_utils::deterministic_condition_value(v, false);
                        if !literal_val.is_empty() {
                            prop_to_exact_val.insert(k.clone(), literal_val);
                        }
                    } else {
                        // 🌟 [CRITICAL FIX] 숫자인 경우, 텍스트에서 실제 숫자를 미리 추출하여 LLM 환각을 방지합니다.
                        // 연산자(Operator)는 하드코딩 문자열 매칭 대신, 위에서 Double Plinko 연산을 통해 도출된 best_op 벡터 결과를 순수하게 신뢰합니다.
                        final_op = best_op.clone();

                        // 숫자 값 100% 원본 추출 (소수점 포함)
                        let final_numeric = crate::utils::ai_utils::deterministic_condition_value(v, true);

                        if !final_numeric.is_empty() {
                            prop_to_exact_val.insert(k.clone(), final_numeric);
                        }
                    }

                    // 🌟 [NUMERIC COMPARISON REROUTE — 최후 안전망]
                    //    수치 비교 구조는 이제 배정 '전' 의 NUMERIC PRE-GATE 에서
                    //    문자열/열거형 필드의 후보 자격 자체를 박탈하므로 정상 경로에서는 여기까지 오지 않습니다.
                    //    다만 PRE-GATE 를 통과한 Numeric 필드가 전부 다른 청크에 선점되어
                    //    그리디 배정이 이 청크를 문자열 필드로 흘려보내는 경우가 남습니다.
                    //
                    //    🌟 [FIX] 기존 항등식 오타 `best_num_score > best_cmp_score - best_cmp_score` (≡ > 0.0) 를
                    //    '현재 확정된 문자열 속성 [k] 의 동일 기준 코사인 점수' 와의 비교로 교정합니다.
                    //    추가로, 비교 연산자 확정도 구 단위 Max-Pool(op_phrase_embs)을 우선 사용합니다.
                    //
                    //    🌟 [NUMBER→NUMBER 허용] 기존 `actual_db_type != "Number"` 가드는
                    //    '5000원 이하로' 가 quantity(Number)로 굳었을 때 METRICS FAMILY GATE 를
                    //    아예 실행하지 않았습니다. quantity 는 convert_conditions_to_sql 의
                    //    valid_cols 에 없어 SQL 에서 통째로 폐기되므로 가격 조건이 소멸합니다.
                    //    (log1.txt: '5000원' → quantity(0.5464) → Qwen3 교정 열화 → UNASSIGN)
                    //    metrics.price.bias 에 "won" 이 있어 '원' ↔ 'won' 다국어 공명이 성립하므로,
                    //    Numeric 필드끼리도 계열 판정으로 재라우팅합니다.
                    //    자기 자신으로의 재라우팅은 best_num_score == cur_prop_score 가 되어 자연 차단됩니다.
                    {
                        if let Some((num_part, cmp_part)) = crate::utils::ai_utils::split_numeric_and_comparator(&combined_chunk) {
                            if !cmp_part.is_empty() {
                                // ① 비교 표현이 어떤 연산자인지 확정합니다.
                                //    구 단위 Max-Pool 이 있으면 그것을, 없으면 센트로이드를 사용합니다.
                                let cmp_emb = self.get_embedding(cmp_part.clone()).await.unwrap_or(vec![0.0; 384]);
                                let mut best_cmp_op = String::new();
                                let mut best_cmp_score = f32::MIN;
                                // 구 단위 Max-Pool 경로
                                let mut cmp_pool_score = 0.0f32;
                                let mut cmp_pool_key = String::new();
                                if !op_phrase_embs.is_empty() {
                                    let mut rank_pool = 0.0f32;
                                    for (oi, oe) in op_phrase_embs.iter().enumerate() {
                                        if oe.iter().all(|&v| v == 0.0) { continue; }
                                        let s = cosine_similarity(&cmp_emb, oe);
                                        if op_phrase_is_rank[oi] {
                                            if s > rank_pool { rank_pool = s; }
                                        } else {
                                            if s > cmp_pool_score {
                                                cmp_pool_score = s;
                                                // 이 구가 속한 연산자 키를 역추적
                                                // op_phrase_texts[oi]가 속한 키를 찾기 위해 bias.json 재탐색
                                                // 단순화: 가장 높은 비교 연산자 구의 인덱스로 키 확정
                                            }
                                        }
                                    }
                                    if cmp_pool_score > rank_pool && cmp_pool_score > 0.0 {
                                        // 구 뱅크에서 best 비교 연산자 확정
                                        for i in 0..dynamic_filter_defs.len() {
                                            if dynamic_filter_defs[i].category != "operators" { continue; }
                                            if dynamic_filter_defs[i].key == "top" || dynamic_filter_defs[i].key == "bottom" { continue; }
                                            let b = cosine_similarity(&cmp_emb, &dynamic_bias_embs[i]);
                                            let p = cosine_similarity(&cmp_emb, &dynamic_prej_embs[i]);
                                            let s = b - p;
                                            if s > best_cmp_score { best_cmp_score = s; best_cmp_op = dynamic_filter_defs[i].key.clone(); }
                                        }
                                    }
                                } else {
                                    // 센트로이드 폴백
                                    for i in 0..dynamic_filter_defs.len() {
                                        if dynamic_filter_defs[i].category != "operators" { continue; }
                                        let b = cosine_similarity(&cmp_emb, &dynamic_bias_embs[i]);
                                        let p = cosine_similarity(&cmp_emb, &dynamic_prej_embs[i]);
                                        let s = b - p;
                                        if s > best_cmp_score { best_cmp_score = s; best_cmp_op = dynamic_filter_defs[i].key.clone(); }
                                    }
                                }
                                // ② 이 청크가 어떤 Numeric 스키마 필드의 값인지 확정합니다.
                                let chunk_emb_local = self.get_embedding(combined_chunk.clone()).await.unwrap_or(vec![0.0; 384]);

                                // 🌟 [METRICS FAMILY GATE] 먼저 "이 청크가 어떤 계량 계열인가" 를 판정합니다.
                                //    "5000원 이하로" → metrics.price ("won" 구와 공명)
                                //    그런 다음 후보 Numeric 필드도 자기 구 뱅크로 계열을 판정하여
                                //    계열이 일치하는 필드만 경쟁시킵니다.
                                //    이 게이트가 없으면 quantity 가 미세한 점수 차로 sale_price 를 이깁니다.
                                let (chunk_metric_family, chunk_metric_score) = if metric_family_bank.is_empty() {
                                    (String::new(), 0.0f32)
                                } else {
                                    crate::utils::ai_utils::metrics_family_argmax(&chunk_emb_local, &metric_family_bank)
                                };
                                if !chunk_metric_family.is_empty() {
                                    emit_term(&format!("      📐 [METRICS FAMILY] \"{}\" → metrics.{} (MaxPool {:.4})", combined_chunk, chunk_metric_family, chunk_metric_score));
                                }

                                let mut best_num_prop = String::new();
                                let mut best_num_score = f32::MIN;
                                let mut family_filtered = 0usize;
                                for (pi, pname) in prop_keys.iter().enumerate() {
                                    if prop_types.get(pname).copied().unwrap_or("String") != "Number" { continue; }
                                    if is_filter_owned(pname) { continue; }

                                    if !chunk_metric_family.is_empty() && !metric_family_bank.is_empty() {
                                        let field_family = crate::utils::ai_utils::metrics_family_of_bank(
                                            &prop_phrase_embs[pi], &metric_family_bank,
                                        );
                                        if !field_family.is_empty() && field_family != chunk_metric_family {
                                            family_filtered += 1;
                                            continue;
                                        }
                                    }

                                    let own = crate::utils::ai_utils::weighted_max_pool_sim(&chunk_emb_local, &prop_phrase_embs[pi], &prop_phrase_weights[pi]);
                                    let pj = cosine_similarity(&chunk_emb_local, &prej_embs[pi]);
                                    let s = own - pj;
                                    if s > best_num_score { best_num_score = s; best_num_prop = pname.clone(); }
                                }
                                if family_filtered > 0 {
                                    emit_term(&format!("      🚧 [METRICS FAMILY GATE] 계열 불일치 Numeric 필드 {}개를 재라우팅 후보에서 제외했습니다.", family_filtered));
                                }
                                // ③ 현재 확정된 문자열 속성 [k] 의 점수를 '같은 기준' 으로 산출해 비교합니다.
                                let mut cur_prop_score = f32::MIN;
                                if let Some(pi) = prop_keys.iter().position(|p| p == k) {
                                    let own = crate::utils::ai_utils::weighted_max_pool_sim(&chunk_emb_local, &prop_phrase_embs[pi], &prop_phrase_weights[pi]);
                                    let pj = cosine_similarity(&chunk_emb_local, &prej_embs[pi]);
                                    cur_prop_score = own - pj;
                                }
                                // ④ 두 축이 모두 확정되고, 그 연산자가 실제 비교 연산자일 때만 재라우팅합니다.
                                //    🌟 [FIX] 기존 `best_num_score > best_cmp_score - best_cmp_score` (항상 > 0.0) 을
                                //    `best_num_score > cur_prop_score` 로 교정.
                                //    Numeric 필드의 own-prej 가 현재 문자열 필드의 own-prej 보다 높아야 교체합니다.
                                let is_comparison = matches!(best_cmp_op.as_str(), "lte" | "lt" | "gte" | "gt" | "eq");
                                if is_comparison && !best_num_prop.is_empty() && best_num_score > cur_prop_score {
                                    emit_term(&format!("    🔁 [NUMERIC REROUTE] \"{}\" → Property [{}] Operator [{}] Value [{}] | 문자열 속성 [{}] 대신 수치 비교로 재라우팅합니다. (CmpOp {:+.4} | NumProp {:+.4} > CurProp {:+.4})",
                                        combined_chunk, best_num_prop, best_cmp_op, num_part, k, best_cmp_score, best_num_score, cur_prop_score));
                                    numeric_reroutes.push((k.clone(), best_num_prop.clone(), best_cmp_op.clone(), num_part.clone()));
                                }
                            }
                        }
                    }

                    prop_to_op.insert(k.clone(), final_op.clone());

                    let mut op_alts = String::new();
                    if final_op != "contains" { 
                        if let Some(cands) = local_filter_candidates.get("operators") {
                            let alts: Vec<String> = cands.iter().skip(1).take(2).map(|c| format!("{} ({:.2})", c.0, c.1)).collect();
                            // 🌟 [CRITICAL FIX] LLM 프롬프트 가이드 문자열에서 Operator 대괄호([]) 안에 불필요한 Alts 정보가 중첩되어 들어가면 LLM이 5000을 500으로 헷갈리는 환각 증세가 발생합니다. 대괄호 밖으로 완전히 분리합니다.
                            if !alts.is_empty() { op_alts = format!(" (Alts: {})", alts.join(", ")); }
                        }
                    }

                    // 🌟 [CRITICAL FIX] 숫자가 포함된 청크("5000원")를 LLM이 "500"으로 환각 파싱하는 것을 방지하기 위해, Rust에서 원본 숫자를 추출하여 명시적으로 가이드에 꽂아 넣습니다.
                    //    문자열 속성도 동일하게 확정 값을 명시합니다. 0.6B 모델이 value 키를 통째로 빠뜨리는 것을
                    //    막고, 어차피 뒤에서 Rust 가 결정론적으로 덮어쓰므로 프롬프트와 최종 값이 100% 일치합니다.
                    let exact_value_guide = match prop_to_exact_val.get(k) {
                        Some(exact) if !exact.is_empty() => format!(", Exact Value [{}]", exact),
                        _ => String::new(),
                    };

                    let guide_log = format!("Target Text: \"{}\" -> Vector Suggests: Property [{}], Operator [{}]{}, Metric Type [{}]{}", combined_chunk, k, final_op, op_alts, final_metric, exact_value_guide);
                    emit_term(&format!("    🧲 {}", guide_log));
                    fragments_text.push_str(&format!("{}\n", guide_log));
                }

                // Qwen3로 2차 매핑 검증
                if !prop_to_op.is_empty() {
                     emit_term("    🧠 [QWEN3 VERIFICATION (2nd)] Verifying operators...");
                     self.ensure_qwen3().await?;
                    
                     // 속성별 operator 검증
                     let mut validated_prop_to_op = prop_to_op.clone();
                     for (prop, op) in &prop_to_op {
                         if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                             emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                             return Ok(json!({ "context": [], "cancelled": true }));
                         }

                         // 🌟 [CRITICAL FIX] 문자열 검색(FTS)용 연산자인 'contains'는 LLM이 문맥을 오해하여 'eq'로 바꾸지 못하도록 검증을 우회합니다.
                         if op == "contains" {
                             emit_term(&format!("      ⚡ [BYPASS] Operator [{}] is FTS. Bypassing verification for [{}]", op, prop));
                             continue;
                         }
                        
                         let prompt = crate::prompts::verify_operator_mapping_prompt(&current_text, prop, op);
                        
                         if let Ok(response) = self.call_qwen3_verification_model(&prompt, Some(cancel_token.clone())).await {
                             if let Ok(result) = serde_json::from_str::<Value>(&response) {
                                 if let Some(suggested) = result.get("suggested_operator").and_then(|v| v.as_str()) {
                                     if suggested != op {
                                         emit_term(&format!("      🔄 Operator for [{}] corrected from [{}] to [{}]", prop, op, suggested));
                                         validated_prop_to_op.insert(prop.clone(), suggested.to_string());
                                     } else {
                                         emit_term(&format!("      ✅ Operator [{}] confirmed for [{}]", op, prop));
                                     }
                                 } else {
                                     emit_term(&format!("      ✅ Operator [{}] confirmed for [{}]", op, prop));
                                 }
                             }
                         }
                     }
                    
                     prop_to_op = validated_prop_to_op;
                }
                
                // 🌟 [DEDUP] 동일한 Global Suggests 3줄이 두 번 append 되고 [FINAL VECTOR GUIDE] 가 두 번 출력되던
                //    죽은 중복 블록을 통째로 제거합니다.
                //    같은 힌트를 두 번 주입하면 0.6B 모델이 '반드시 채워야 하는 값' 으로 오인해
                //    근거 없는 status / find 를 창작합니다. (로그: status "show", find "many")
                //    또한 여기서 계산되던 get_deterministic_time_guide 결과는 바로 아래에서 재선언(shadow)되어
                //    한 번도 사용되지 않는 죽은 호출이었습니다.
                if !best_status_global.is_empty() { fragments_text.push_str(&format!("Global Status Suggests [{}]\n", best_status_global)); }
                if !best_sub_global.is_empty() { fragments_text.push_str(&format!("Global Substantial Suggests [{}]\n", best_sub_global)); }
                if !best_find_global.is_empty() { fragments_text.push_str(&format!("Global Find Suggests [{}]\n", best_find_global)); }

                emit_term(&format!("\n  🎯 [FINAL VECTOR GUIDE FOR LLM] \n{}", fragments_text.trim()));

                // 🌟 [VRAM 최적화 수정] 루프 안에서 임베딩 모델을 언로드하면 다음 세그먼트에서 다시 로드하는 Ping-Pong이 발생하므로 삭제합니다.
                // 파이프라인이 모두 종료된 후 마지막에 일괄적으로 deep_purge_resources를 통해 해제합니다.

                let mut deterministic_json = None;
                let mut llm_temporal_guide = String::new();

                // 5. LLM Normalization (Advanced Parser Mode with Vector Guide)
                if !fragments_text.is_empty() {
                    let now = chrono::Local::now();
                    let time_context = format!("Current Time: {}\nTimezone: {}\nLanguage: {}", now.format("%Y-%m-%dT%H:%M:%S"), now.format("%z"), language);

                    // 이미 최초에 Qwen3를 로드했으므로 ensure_qwen3() 호출 생략

                    // 🌟 [QWEN3 VERIFICATION: TIME & SEASON] Plinko에서 대충 잡힌 시간/시즌을 LLM으로 2차 검증하여 환각을 원천 차단합니다.
                    let mut verified_time = String::new();
                    let mut verified_season = String::new();

                    if let Some(time_cands) = filter_candidates.get("time_filters") {
                        if !time_cands.is_empty() {
                            let first_choice = &time_cands[0].0;
                            let first_score = time_cands[0].1;
                            let alternatives: Vec<(String, f32)> = time_cands.iter().skip(1).take(3).cloned().collect();
                            
                            let prompt_time = crate::parsing::extract_time_intent_prompt(&current_text, &time_context, first_choice, first_score, &alternatives);
                            if let Ok(res_time) = self.call_qwen3_verification_model(&prompt_time, Some(cancel_token.clone())).await {
                                let final_time_json = crate::parsing::parse_json_from_llm(&res_time);
                                if let Some(t) = final_time_json.get("time_intent").and_then(|v| v.as_str()) {
                                    if !t.is_empty() { 
                                        verified_time = t.to_string(); 
                                        emit_term(&format!("  🕒 [LLM-VERIFIED TIME] Time Intent explicitly confirmed as: '{}'", verified_time));
                                    } else {
                                        emit_term("  🕒 [LLM-VERIFIED TIME] Time Intent rejected (Empty).");
                                    }
                                }
                            }
                        }
                    }

                    // 🌟 [SEASON EXACT MATCH PRIORITY] bias.json 의 exact_match 로 이미 확정된 계절은
                    //    LLM 에게 되묻지 않습니다. 되물으면 로그처럼 '여름' → 'autumn' 환각이 발생하고
                    //    started_at/expired_at 에 엉뚱한 범위가 주입되어 검색이 통째로 0건이 됩니다.
                    //    LLM 호출도 1회 줄어듭니다.
                    if !exact_season_key.is_empty() {
                        verified_season = exact_season_key.clone();
                        emit_term(&format!("  🌤️ [SEASON EXACT MATCH] Season Intent 를 bias.json exact_match 로 '{}' 확정 (LLM 호출 생략).", verified_season));
                    } else if let Some(season_cands) = filter_candidates.get("season_filters") {
                        if !season_cands.is_empty() {
                            let first_choice = &season_cands[0].0;
                            let first_score = season_cands[0].1;
                            let alternatives: Vec<(String, f32)> = season_cands.iter().skip(1).take(3).cloned().collect();
                            
                            let prompt_season = crate::parsing::extract_season_intent_prompt(&current_text, first_choice, first_score, &alternatives);
                            if let Ok(res_season) = self.call_qwen3_verification_model(&prompt_season, Some(cancel_token.clone())).await {
                                let final_season_json = crate::parsing::parse_json_from_llm(&res_season);
                                if let Some(s) = final_season_json.get("season_intent").and_then(|v| v.as_str()) {
                                    if !s.is_empty() { 
                                        verified_season = s.to_string(); 
                                        emit_term(&format!("  🌤️ [LLM-VERIFIED SEASON] Season Intent explicitly confirmed as: '{}'", verified_season));
                                    } else {
                                        emit_term("  🌤️ [LLM-VERIFIED SEASON] Season Intent rejected (Empty).");
                                    }
                                }
                            }
                        }
                    }

                    if !verified_time.is_empty() { llm_temporal_guide.push_str(&format!("Time Intent [{}] ", verified_time)); }
                    if !verified_season.is_empty() { llm_temporal_guide.push_str(&format!("Season Intent [{}]", verified_season)); }

                    // Vector 기반으로 도출되고 LLM으로 검증된 의도를 바탕으로 Deterministic Time Guide(달력 SQL 필터) 획득
                    let (deterministic_guide_log, det_json_res) = crate::parsing::get_deterministic_time_guide(&llm_temporal_guide, language);
                    deterministic_json = det_json_res;

                    if !deterministic_guide_log.is_empty() {
                        emit_term(&format!("  ⏳ [DETERMINISTIC TIME GUIDE]\n  {}", deterministic_guide_log.replace("\n", "\n  ")));
                    }

                    // 🌟 [명시적 타입 선언] 추출될 속성(Property)의 스키마 타입에 따라 Number 인지 String 인지 정확하게 결정합니다.
                    let mut matched_types = Vec::new();
                    for k in prop_to_op.keys() {
                        let t = prop_types.get(k).copied().unwrap_or("String");
                        matched_types.push(t);
                    }
                    matched_types.sort();
                    matched_types.dedup();
                    
                    // 🌟 [CRITICAL FIX] String 타입일 경우 JSON 값이 큰따옴표에 제대로 감싸지도록 프롬프트 가이드를 "\"String\"" 형태로 교정합니다.
                    let value_type_str = if matched_types.is_empty() {
                        "\"String\"".to_string()
                    } else if matched_types.len() == 1 {
                        if matched_types[0] == "String" { "\"String\"".to_string() } else { "Number".to_string() }
                    } else {
                        let mut type_conditions = Vec::new();
                        for k in prop_to_op.keys() {
                            let t = prop_types.get(k).copied().unwrap_or("String");
                            let t_quoted = if t == "String" { "\"String\"" } else { "Number" };
                            type_conditions.push(format!("{} (if property is '{}')", t_quoted, k));
                        }
                        type_conditions.join(", ")
                    };

                    // 🌟 [CRITICAL FIX] 벡터 매칭 가이드와 LLM 시간 가이드를 병합하여 최종 조건 추출 프롬프트 호출
                    let combined_guide = format!("{}\n{}", fragments_text.trim(), llm_temporal_guide);
                    
                    let prompt_numeric = crate::parsing::extract_numeric_conditions(&current_text, &seg_type, metrics_json, &combined_guide, &time_context, language, &value_type_str);
                    
                    // 🌟 [CRITICAL FIX] Qwen3 모델을 사용하여 메모리 사용량을 줄이고 통일화합니다.
                    self.ensure_qwen3().await?;

                    // call_qwen3_verification_model을 통해 순차적으로 LLM Normalization 수행
                    let res_numeric = self.call_qwen3_verification_model(&prompt_numeric, Some(cancel_token.clone())).await?;

                    // 🌟 [EVIDENCE GATE] 배타 게이트를 통과한 벡터 후보가 하나도 없다는 것은
                    //    '질의에 그 의도의 근거가 존재하지 않는다' 는 결정론적 사실입니다.
                    //    근거가 없는 상태에서 0.6B 에게 물으면 반드시 아무 값이나 채워 넣습니다.
                    //    Qwen3 는 '근거가 있을 때 어떤 값인지 고르는' 판정에만 사용합니다.
                    let res_status = if !best_status_global.is_empty() && filter_candidates.get("status_filters").map_or(true, |c| c.is_empty()) {
                        emit_term(&format!("  ⚡ [STATUS DETERMINISTIC] 라우팅으로 '{}' 확정. LLM 호출 생략.", best_status_global));
                        format!("{{ \"status\": \"{}\" }}", best_status_global)
                    } else {
                        match filter_candidates.get("status_filters").filter(|c| !c.is_empty()) {
                            Some(cands) => {
                                let alternatives: Vec<(String, f32)> = cands.iter().skip(1).take(3).cloned().collect();
                                let p = crate::parsing::extract_status_intent_prompt(&current_text, &seg_type, &cands[0].0, cands[0].1, &alternatives);
                                self.call_qwen3_verification_model(&p, Some(cancel_token.clone())).await?
                            },
                            None => {
                                emit_term("  ⛔ [STATUS EVIDENCE GATE] 후보 없음. LLM 호출 없이 빈 값 확정.");
                                "{ \"status\": \"\" }".to_string()
                            }
                        }
                    };

                    let res_substantial = if !best_sub_global.is_empty() {
                        emit_term(&format!("  ⚡ [SUBSTANTIAL DETERMINISTIC] 추상 수식어 라우팅으로 '{}' 확정. LLM 호출 생략.", best_sub_global));
                        format!("{{ \"substantial\": \"{}\" }}", best_sub_global)
                    } else {
                        match filter_candidates.get("substantial_filters").filter(|c| !c.is_empty()) {
                            Some(cands) => {
                                let alternatives: Vec<(String, f32)> = cands.iter().skip(1).take(3).cloned().collect();
                                let p = crate::parsing::extract_substantial_intent_prompt(&current_text, &cands[0].0, cands[0].1, &alternatives);
                                self.call_qwen3_verification_model(&p, Some(cancel_token.clone())).await?
                            },
                            None => {
                                emit_term("  ⛔ [SUBSTANTIAL EVIDENCE GATE] 후보 없음. LLM 호출 없이 빈 값 확정.");
                                "{ \"substantial\": \"\" }".to_string()
                            }
                        }
                    };

                    let res_find = if !best_find_global.is_empty() {
                        emit_term(&format!("  ⚡ [FIND DETERMINISTIC] 추상 수식어 라우팅으로 '{}' 확정. LLM 호출 생략.", best_find_global));
                        format!("{{ \"find\": \"{}\" }}", best_find_global)
                    } else {
                        match filter_candidates.get("find_filters").filter(|c| !c.is_empty()) {
                            Some(cands) => {
                                let alternatives: Vec<(String, f32)> = cands.iter().skip(1).take(3).cloned().collect();
                                let p = crate::parsing::extract_find_intent_prompt(&current_text, &cands[0].0, cands[0].1, &alternatives);
                                self.call_qwen3_verification_model(&p, Some(cancel_token.clone())).await?
                            },
                            None => {
                                emit_term("  ⛔ [FIND EVIDENCE GATE] 후보 없음. LLM 호출 없이 빈 값 확정.");
                                "{ \"find\": \"\" }".to_string()
                            }
                        }
                    };

                    emit_term(&format!("  🤖 [LLM RAW RESPONSE - NUMERIC]\n{}", res_numeric.trim()));
                    emit_term(&format!("  🤖 [LLM RAW RESPONSE - STATUS]\n{}", res_status.trim()));
                    emit_term(&format!("  🤖 [LLM RAW RESPONSE - SUBSTANTIAL]\n{}", res_substantial.trim()));
                    emit_term(&format!("  🤖 [LLM RAW RESPONSE - FIND]\n{}", res_find.trim()));
                    
                    let final_numeric_json = crate::parsing::parse_json_from_llm(&res_numeric);
                    let final_status_json = crate::parsing::parse_json_from_llm(&res_status);
                    let final_substantial_json = crate::parsing::parse_json_from_llm(&res_substantial);
                    let final_find_json = crate::parsing::parse_json_from_llm(&res_find);
                    
                    emit_term(&format!("  ✅ [EXTRACTED DATA (NUMERIC RAW)]\n{}", serde_json::to_string_pretty(&final_numeric_json).unwrap_or_default()));
                    
                    if let Some(obj) = seg.as_object_mut() {
                        if let Some(status_val) = final_status_json.get("status") {
                            obj.insert("status".to_string(), status_val.clone());
                        }
                        if let Some(sub_val) = final_substantial_json.get("substantial") {
                            obj.insert("substantial".to_string(), sub_val.clone());
                        }
                        if let Some(find_val) = final_find_json.get("find") {
                            obj.insert("find".to_string(), find_val.clone());
                        }
                        
                        // 🌟 LLM이 뽑아준 "값"과 Rust 메모리에 저장해둔 "연산자(operator)"를 여기서 최종 조립합니다.
                        let mut structured_cond = serde_json::Map::new();
                        
                        // 🌟 [CRITICAL FIX] LLM이 배열이 아닌 단일 객체로 반환했을 때 필터가 망가지는 현상을 막기 위해 파싱을 배열 폼으로 통일합니다.
                        let condition_json = final_numeric_json.get("condition");
                        let mut cond_items = Vec::new();

                        if let Some(arr) = condition_json.and_then(|v| v.as_array()) {
                            cond_items = arr.clone();
                        } else if let Some(obj) = condition_json.and_then(|v| v.as_object()) {
                            // LLM이 { "property": "...", "operator": "...", "value": "..." } 포맷을 단일 객체로 뱉었을 경우 배열로 감싸서 넘깁니다.
                            if obj.contains_key("property") || obj.contains_key("property_name") {
                                cond_items.push(json!(obj));
                            } else {
                                // { "price": { "operator": "lt", "value": 5000 } } 맵 포맷일 경우
                                for (k, v) in obj {
                                    if deterministic_json.is_some() && (k == "started_at" || k == "expired_at" || k == "registration_date" || k == "date") {
                                        continue;
                                    }
                                    if v.is_object() {
                                        let mut final_val_obj = v.clone();
                                        if let Some(v_obj) = final_val_obj.as_object_mut() {
                                            if !v_obj.contains_key("operator") {
                                                let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                                v_obj.insert("operator".to_string(), json!(op));
                                            }
                                            // 🌟 [CRITICAL FIX] Rust 원본 숫자값을 덮어씌움
                                            if let Some(exact_val) = prop_to_exact_val.get(k) {
                                                v_obj.insert("value".to_string(), json!(exact_val));
                                            }
                                        }
                                        structured_cond.insert(k.clone(), final_val_obj);
                                    } else {
                                        let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                        let final_value = prop_to_exact_val.get(k).map(|v| json!(v)).unwrap_or_else(|| v.clone());
                                        structured_cond.insert(k.clone(), json!({
                                            "operator": op,
                                            "value": final_value
                                        }));
                                    }
                                }
                            }
                        }

                        // 단일화된 배열(cond_items) 처리
                        for item in cond_items {
                            if let Some(item_obj) = item.as_object() {
                                let mut prop_val_opt = None;
                                for (ik, iv) in item_obj {
                                    if ik.trim() == "property" || ik.trim() == "property_name" {
                                        prop_val_opt = iv.as_str();
                                        break;
                                    }
                                }

                                if let Some(prop_val) = prop_val_opt {
                                    let k = prop_val.trim().to_string();
                                    
                                    if deterministic_json.is_some() && (k == "started_at" || k == "expired_at" || k == "registration_date" || k == "date") {
                                        continue;
                                    }

                                    // 🌟 [CRITICAL FIX] 유효하지 않은 프로퍼티 이름(LLM 환각) 무시
                                    if !prop_to_op.contains_key(&k) {
                                        emit_term(&format!("      ⚠️ [DISCARD] LLM hallucinated invalid property name: [{}]. Discarding.", k));
                                        continue;
                                    }

                                    let mut op = item_obj.get("operator").and_then(|v| v.as_str())
                                        .unwrap_or_else(|| prop_to_op.get(&k).map(|s| s.as_str()).unwrap_or("eq")).to_string();
                                    
                                    let mut final_val_obj = serde_json::Map::new();
                                    for (ik, iv) in item_obj {
                                        let ik_trimmed = ik.trim();
                                        if ik_trimmed != "property" && ik_trimmed != "property_name" && ik_trimmed != "operator" {
                                            // 🌟 [CRITICAL FIX] Rust 원본 숫자값을 덮어씌움
                                            if ik_trimmed == "value" {
                                                if let Some(exact_val) = prop_to_exact_val.get(&k) {
                                                    final_val_obj.insert(ik_trimmed.to_string(), json!(exact_val));
                                                    continue;
                                                }
                                            }
                                            final_val_obj.insert(ik_trimmed.to_string(), iv.clone());
                                        }
                                    }

                                    // 🌟 [CRITICAL FIX] 숫자가 없어서 value가 빈 값인데 연산자가 부등호일 경우 퍼지(Fuzzy) 표현으로 간주하여 강제 교정
                                    let actual_db_type = prop_types.get(&k).copied().unwrap_or("String");
                                    if actual_db_type == "Number" {
                                        let val_is_empty = final_val_obj.get("value").and_then(|v| v.as_str()).map_or(false, |s| s.trim().is_empty());
                                        if val_is_empty && !prop_to_exact_val.contains_key(&k) {
                                            if op == "gt" || op == "gte" {
                                                op = "top".to_string();
                                                final_val_obj.insert("percent_total".to_string(), json!("20.0"));
                                                final_val_obj.insert("is_percent".to_string(), json!(true));
                                            } else if op == "lt" || op == "lte" {
                                                op = "bottom".to_string();
                                                final_val_obj.insert("percent_total".to_string(), json!("20.0"));
                                                final_val_obj.insert("is_percent".to_string(), json!(true));
                                            }
                                        }
                                    }
                                    
                                    final_val_obj.insert("operator".to_string(), json!(op));
                                    
                                    if !final_val_obj.contains_key("value") {
                                        if let Some(exact_val) = prop_to_exact_val.get(&k) {
                                            final_val_obj.insert("value".to_string(), json!(exact_val));
                                        }
                                    }
                                    
                                    structured_cond.insert(k, json!(final_val_obj));
                                } else {
                                    for (k, val) in item_obj {
                                        let k_trimmed = k.trim();
                                        if deterministic_json.is_some() && (k_trimmed == "started_at" || k_trimmed == "expired_at" || k_trimmed == "registration_date" || k_trimmed == "date") {
                                            continue;
                                        }

                                        let mut op = prop_to_op.get(k_trimmed).map(|s| s.as_str()).unwrap_or("eq").to_string();
                                        let mut final_val_obj = val.clone();

                                        if let Some(v_obj) = final_val_obj.as_object_mut() {
                                            if !v_obj.contains_key("operator") {
                                                v_obj.insert("operator".to_string(), json!(op));
                                            } else {
                                                op = v_obj.get("operator").and_then(|v| v.as_str()).unwrap_or(&op).to_string();
                                            }

                                            // 🌟 [CRITICAL FIX] Rust 원본 숫자값을 덮어씌움
                                            if let Some(exact_val) = prop_to_exact_val.get(k_trimmed) {
                                                v_obj.insert("value".to_string(), json!(exact_val));
                                            }
                                        } else {
                                            let final_value = prop_to_exact_val.get(k_trimmed).map(|v| json!(v)).unwrap_or_else(|| val.clone());
                                            final_val_obj = json!({
                                                "operator": op,
                                                "value": final_value
                                            });
                                        }

                                        // 퍼지 변환 동일 적용
                                        if let Some(v_obj) = final_val_obj.as_object_mut() {
                                            let actual_db_type = prop_types.get(k_trimmed).copied().unwrap_or("String");
                                            if actual_db_type == "Number" {
                                                let val_is_empty = v_obj.get("value").and_then(|v| v.as_str()).map_or(false, |s| s.trim().is_empty());
                                                if val_is_empty && !prop_to_exact_val.contains_key(k_trimmed) {
                                                    if op == "gt" || op == "gte" {
                                                        v_obj.insert("operator".to_string(), json!("top"));
                                                        v_obj.insert("percent_total".to_string(), json!("20.0"));
                                                        v_obj.insert("is_percent".to_string(), json!(true));
                                                    } else if op == "lt" || op == "lte" {
                                                        v_obj.insert("operator".to_string(), json!("bottom"));
                                                        v_obj.insert("percent_total".to_string(), json!("20.0"));
                                                        v_obj.insert("is_percent".to_string(), json!(true));
                                                    }
                                                }
                                            }
                                        }

                                        structured_cond.insert(k_trimmed.to_string(), final_val_obj);
                                    }
                                }
                            }
                        }

                        // 🌟 [CRITICAL RECOVERY] LLM이 배열에서 특정 키를 통째로 누락(환각)시켰을 경우를 대비해, 
                        // Rust에서 명시적으로 찾아둔 값(prop_to_exact_val)을 강제로 쑤셔 넣습니다.
                        // 🌟 [VALUE BIND] 키는 돌려줬지만 value 키 자체를 누락한 경우(로그의 color)도 여기서 봉합합니다.
                        for (k, exact_val) in &prop_to_exact_val {
                            if !structured_cond.contains_key(k) {
                                let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                structured_cond.insert(k.clone(), json!({
                                    "operator": op,
                                    "value": exact_val
                                }));
                                emit_term(&format!("      ⚠️ [RECOVERY] LLM missed property [{}]. Forcefully recovered with exact value [{}].", k, exact_val));
                                continue;
                            }

                            if let Some(existing) = structured_cond.get_mut(k) {
                                if let Some(obj) = existing.as_object_mut() {
                                    let needs_fill = match obj.get("value") {
                                        None => true,
                                        Some(serde_json::Value::Null) => true,
                                        Some(serde_json::Value::String(s)) => s.trim().is_empty() || s == "null",
                                        _ => false,
                                    };
                                    if needs_fill {
                                        obj.insert("value".to_string(), json!(exact_val));
                                        emit_term(&format!("      🩹 [VALUE BIND] LLM returned [{}] without a usable value. Deterministically bound to [{}].", k, exact_val));
                                    }
                                }
                            }
                        }

                        // 🌟 [PERCENT RESIDUE SWEEP] 0.6B 모델은 값이 없을 때 percent_total 을 창작합니다.
                        //    (로그: { "operator":"lt", "percent_total":"0.38", "value":"" })
                        //    percent_total 은 top/bottom 연산자에서만 의미를 갖는 필드이므로,
                        //    그 외 연산자이거나 is_percent 가 거짓이면 잔재를 완전히 제거합니다.
                        //    프론트엔드 Dexie 재질의가 이 키를 읽고 오동작하는 경로를 원천 차단합니다.
                        {
                            let mut swept: Vec<String> = Vec::new();
                            for (k, v) in structured_cond.iter_mut() {
                                let obj = match v.as_object_mut() { Some(o) => o, None => continue };
                                let op = obj.get("operator").and_then(|o| o.as_str()).unwrap_or("").to_string();
                                let is_rank = op == "top" || op == "bottom";
                                let is_percent = obj.get("is_percent").and_then(|b| b.as_bool()).unwrap_or(false);
                                if is_rank && is_percent { continue; }
                                let had_percent = obj.remove("percent_total").is_some();
                                let had_flag = obj.remove("is_percent").is_some();
                                if had_percent || had_flag { swept.push(k.clone()); }
                            }
                            if !swept.is_empty() {
                                emit_term(&format!("      🧽 [PERCENT RESIDUE SWEEP] top/bottom 이 아닌 조건 {:?} 에서 percent_total/is_percent 환각 잔재를 제거했습니다.", swept));
                            }
                        }

                        // 🌟 [ABSTRACT QUALIFIER MATERIALIZE]
                        //    substantial_filters 키(weight / sale_price / shipping_fee ...)는
                        //    실제 스키마 필드명과 동일하므로, find_filters 방향을 연산자로 환산해
                        //    '조건' 으로 물질화합니다.
                        //      heavy / many / much  → top    (상위 구간)
                        //      light / few / little → bottom (하위 구간)
                        //    방향은 문자열 판정이 아니라 위에서 코사인으로 확정한 캐노니컬 키를 그대로 씁니다.
                        //    percent_total 은 기존 퍼지(Fuzzy) 변환이 쓰는 값과 동일하게 유지합니다.
                        // 🌟 [DEAD BRANCH FIX] 기존 구조는 CROSS-DOMAIN 분기를
                        //    `prop_keys.iter().any(|p| p == &best_sub_global)` 안에 중첩시켰습니다.
                        //    그런데 CROSS-DOMAIN 은 정확히 '이 필드가 현재 스키마에 없을 때'를 위한 것이라
                        //    논리적으로 절대 도달할 수 없는 죽은 코드였습니다.
                        //    (log1.txt: '무거운' → substantial_filters.weight 확정에도 MATERIALIZE 로그 0건.
                        //     goods 스키마에 weight 가 없고 tracking 에만 있기 때문)
                        //    현재 스키마 보유 여부로 분기를 완전히 분리합니다.
                        if !best_sub_global.is_empty()
                            && !structured_cond.contains_key(&best_sub_global)
                        {
                            let dir_op = match best_find_global.as_str() {
                                "heavy" | "many" | "much"   => "top",
                                "light" | "few"  | "little" => "bottom",
                                _ => "",
                            };
                            let owned_here = prop_keys.iter().any(|p| p == &best_sub_global);

                            if owned_here && !dir_op.is_empty() {
                                structured_cond.insert(best_sub_global.clone(), json!({
                                    "operator": dir_op,
                                    "percent_total": "20.0",
                                    "is_percent": true
                                }));
                                emit_term(&format!(
                                    "      🧲 [ABSTRACT MATERIALIZE] substantial='{}' + find='{}' → 조건 '{} {} 20%' 물질화",
                                    best_sub_global, best_find_global, best_sub_global, dir_op
                                ));
                            } else {
                                // 🌟 [CROSS-DOMAIN MATERIALIZE] 현재 도메인 스키마에 그 필드가 없으면
                                //    다른 도메인 스키마를 뒤져 보유 도메인을 찾아 메타데이터로 남깁니다.
                                //    STAGE-3 이 이 값을 읽어 해당 도메인 컨텍스트를 추가 발행합니다.
                                //    (예: goods 질의의 '무거운' → weight 는 tracking 스키마에 있음)
                                let mut host_domain = String::new();
                                for cand in ["tracking", "goods", "order", "event", "coupon", "review"] {
                                    if cand == seg_type { continue; }
                                    let cand_fields = crate::parsing::get_detail_schema_fields(cand, "", &query_lang);
                                    if cand_fields.iter().any(|(n, _, _, _)| n == &best_sub_global) {
                                        host_domain = cand.to_string();
                                        break;
                                    }
                                }
                                if !host_domain.is_empty() {
                                    emit_term(&format!(
                                        "      🔀 [CROSS-DOMAIN MATERIALIZE] '{}' 는 '{}' 스키마에 없고 '{}' 스키마에 존재합니다. 교차 도메인 컨텍스트를 발행합니다. (find='{}')",
                                        best_sub_global, seg_type, host_domain, best_find_global
                                    ));
                                    obj.insert("substantial_host".to_string(), json!(host_domain));
                                } else {
                                    emit_term(&format!(
                                        "      ⚪ [ABSTRACT MATERIALIZE SKIP] substantial='{}' 을 보유한 도메인 스키마가 없어 메타데이터로만 전달합니다. (find='{}')",
                                        best_sub_global, best_find_global
                                    ));
                                }
                            }
                        }

                        // 🌟 [NUMERIC REROUTE APPLY] 문자열로 굳었던 수치 조건을 실제 Numeric 필드로 교체합니다.
                        //    이 교체가 있어야 convert_conditions_to_sql 이 `amount <= 5000` 을 생성합니다.
                        for (from_prop, to_prop, op, num_val) in &numeric_reroutes {
                            if !structured_cond.contains_key(from_prop) { continue; }
                            if structured_cond.contains_key(to_prop) { continue; }
                            structured_cond.remove(from_prop);
                            structured_cond.insert(to_prop.clone(), json!({
                                "operator": op,
                                "value": num_val
                            }));
                            emit_term(&format!("      🔁 [NUMERIC REROUTE APPLY] '{}' 조건을 '{} {} {}' 로 교체했습니다.", from_prop, to_prop, op, num_val));
                        }

                        // 🌟 [EMPTY CONDITION SWEEP] 끝내 값을 확보하지 못한 조건은 필터가 아니라 노이즈입니다.
                        //    (top / bottom 은 percent_total 로 동작하므로 값이 없어도 유효)
                        let empty_keys: Vec<String> = structured_cond.iter().filter(|(_, v)| {
                            let op = v.get("operator").and_then(|o| o.as_str()).unwrap_or("");
                            if op == "top" || op == "bottom" { return false; }
                            match v.get("value") {
                                None => true,
                                Some(serde_json::Value::Null) => true,
                                Some(serde_json::Value::String(s)) => s.trim().is_empty() || s == "null",
                                _ => false,
                            }
                        }).map(|(k, _)| k.clone()).collect();
                        for k in empty_keys {
                            structured_cond.remove(&k);
                            emit_term(&format!("      🗑️ [EMPTY CONDITION DROP] '{}' 는 끝내 값을 확보하지 못해 조건에서 제외합니다.", k));
                        }

                        // 🌟 [CRITICAL FIX] Deterministic JSON(확정된 기간)이 있다면 여기서 강력하게 덮어씌워서 LLM 환각을 원천 차단합니다!
                        if let Some(det_json) = &deterministic_json {
                            if let Some(det_obj) = det_json.as_object() {
                                for (k, v) in det_obj {
                                    structured_cond.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        
                        obj.insert("condition".to_string(), json!(structured_cond.clone()));

                        // 🌟 [N:N ALTERNATE AXIS] 실제로 조건에 실린 속성에 대해서만 대안 목록을 남깁니다.
                        //    STAGE-3 이 이 목록으로 '1순위가 틀렸을 때의 대안 쿼리'를 발행하고,
                        //    프론트엔드 Dexie 도 동일 목록으로 재질의할 수 있습니다.
                        let mut alt_payload = serde_json::Map::new();
                        for (prop, alts) in &plinko_alternates {
                            if !structured_cond.contains_key(prop) { continue; }
                            if alts.is_empty() { continue; }
                            alt_payload.insert(prop.clone(), json!(alts.clone()));
                        }
                        if !alt_payload.is_empty() {
                            emit_term(&format!("  🔀 [ALTERNATE AXIS]\n{}", serde_json::to_string_pretty(&alt_payload).unwrap_or_default()));
                        }
                        obj.insert("alternates".to_string(), Value::Object(alt_payload));

                        // 🌟 [UNASSIGNED RESCUE] 조건이 되지 못한 청크를 STAGE-3 이 FTS 검색어에 병합할 수 있도록 전달합니다.
                        if !unassigned_chunks.is_empty() {
                            emit_term(&format!("  🧷 [UNASSIGNED RESCUE] 조건 미확정 청크 {:?} 를 FTS 검색어로 보존합니다.", unassigned_chunks));
                        }
                        obj.insert("unassigned".to_string(), json!(unassigned_chunks.clone()));

                        // 🌟 완전일치로 확정된 계절/시간 키를 STAGE-3 및 결정론 시간 가이드에 넘깁니다.
                        if !exact_season_key.is_empty() {
                            obj.insert("exact_season".to_string(), json!(exact_season_key.clone()));
                        }
                        if !exact_time_key.is_empty() {
                            obj.insert("exact_time".to_string(), json!(exact_time_key.clone()));
                        }

                        // 🌟 [추가] 벡터 매칭(연산자)과 LLM(추출 값) + 확정 날짜가 최종 병합된 결과 로그 출력
                        emit_term(&format!("  🚀 [FINAL MERGED CONDITION]\n{}", serde_json::to_string_pretty(&structured_cond).unwrap_or_default()));
                    }
                } else {
                    if let Some(obj) = seg.as_object_mut() {
                        obj.insert("condition".to_string(), json!({}));
                    }
                }

                crate::models::qwen::generate::wait_for_global_io().await;
                
                if !self.is_cpu_mode {
                    let dev = self.device_config.device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if dev.is_cuda() { let _ = dev.synchronize(); }
                    }).await;
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
        }

        // 🌟 [DOMAIN-SPLIT N:N CONTEXT GENERATION]
        //  기존 STAGE-3 은 서로 다른 도메인의 세그먼트를 '마스터 컨텍스트 1개'로 강제 병합했습니다.
        //  그 결과(로그의 2번 질의)는 다음과 같이 무너졌습니다.
        //   ① event 세그먼트('올해 여름')가 만든 started_at / expired_at 이 goods 컨텍스트에 실려
        //      SQL 이 `created_at >= 1788188400000` 이 되면서 goods 상품 전체가 잘려 0건이 되었습니다.
        //   ② review 세그먼트의 title="고객의" 가 goods 세그먼트의 title="제품" 을 덮어썼습니다.
        //      (master_condition.insert 가 무조건 덮어쓰기)
        //   ③ 도메인이 goods 하나로 접히면서 target_table 도 sales 하나만 조회되어
        //      event / review 테이블은 아예 검색조차 되지 않았습니다.
        //  '지연이나 일부 오답보다 정답이 결과에 포함되는 것이 최우선' 이라는 요구에 맞춰
        //  (도메인 축) × (조건 완화 축) × (질의 텍스트 축) × (속성 대안 축) 의 N:N 조합으로 분할 발행합니다.
        //  lib.rs 의 검색 루프는 컨텍스트마다 독립 쿼리를 돌리고 id 기준으로 dedup 하므로
        //  컨텍스트가 늘어날수록 리콜만 올라가고 중복 결과는 생기지 않습니다.
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            if !ctx_arr.is_empty() {
                emit_term("[STAGE-3] Generating domain-split N:N combinatorial contexts...");

                struct DomainGroup {
                    // 🌟 [PURGED] ACTION WORD 를 제거한 정화 텍스트 — A/FULL, B/NARROWED 전용
                    text_words: Vec<String>,
                    // 🌟 [RAW] 정화 이전 세그먼트 원문 — C/RECALL, C/CANDIDATE, E/FALLBACK 전용
                    //    ACTION VERB 게이트가 오탐하면 text_words 가 통째로 비어
                    //    push_ctx 의 `body.is_empty() && condition.is_empty()` 에 걸려
                    //    모든 티어가 소멸합니다.
                    //    (new_log2.txt: '니트'/'가디건'/'찾아줘' 3단어 전부 오탐 → 발행 쿼리 7건 → 1건)
                    //    원문 축을 별도로 보존하여 리콜 티어는 게이트 오탐과 무관하게 항상 발행되도록 합니다.
                    raw_words: Vec<String>,
                    value_words: Vec<String>,
                    condition: serde_json::Map<String, Value>,
                    alternates: serde_json::Map<String, Value>,
                    status: Value,
                    substantial: Value,
                    find: Value,
                }

                // 🌟 [WORD ORDER PRESERVE] value_words 는 condition 맵(키 알파벳순) 순회로 수집되어
                //    원문 어순이 완전히 파괴됩니다.
                //    (로그: "팔린 많이 5000원 이하로 제품으로 제품중에서 제품 무거운" — 원문과 순서가 전혀 다름)
                //    FTS 는 어순에 민감하므로 원문 순서로 재정렬해야 매칭률이 유지됩니다.
                fn reorder_by_source(words: &Vec<String>, source: &Vec<String>) -> Vec<String> {
                    let mut out: Vec<String> = Vec::with_capacity(words.len());
                    for s in source {
                        if words.iter().any(|w| w == s) && !out.iter().any(|o| o == s) {
                            out.push(s.clone());
                        }
                    }
                    for w in words {
                        if !out.iter().any(|o| o == w) { out.push(w.clone()); }
                    }
                    out
                }

                fn push_ctx(
                    out: &mut Vec<Value>,
                    seen: &mut std::collections::HashSet<String>,
                    domain: &str,
                    text: &str,
                    condition: serde_json::Map<String, Value>,
                    alternates: &serde_json::Map<String, Value>,
                    status: &Value,
                    substantial: &Value,
                    find: &Value,
                    tier: &str,
                ) -> bool {
                    if domain.is_empty() { return false; }
                    let body = text.trim();
                    if body.is_empty() && condition.is_empty() { return false; }
                    let sig = format!("{}\u{1}{}\u{1}{}", domain, body, serde_json::to_string(&condition).unwrap_or_default());
                    if !seen.insert(sig) { return false; }

                    let mut ctx = serde_json::Map::new();
                    ctx.insert("type".to_string(), json!(domain));
                    ctx.insert("text".to_string(), json!(body));
                    ctx.insert("status".to_string(), status.clone());
                    ctx.insert("substantial".to_string(), substantial.clone());
                    ctx.insert("find".to_string(), find.clone());
                    ctx.insert("condition".to_string(), Value::Object(condition));
                    ctx.insert("alternates".to_string(), Value::Object(alternates.clone()));
                    ctx.insert("tier".to_string(), json!(tier));
                    out.push(Value::Object(ctx));
                    true
                }

                let mut final_contexts: Vec<Value> = Vec::new();
                let mut seen_ctx: std::collections::HashSet<String> = std::collections::HashSet::new();

                let mut groups: std::collections::HashMap<String, DomainGroup> = std::collections::HashMap::new();
                let mut group_order: Vec<String> = Vec::new();
                let mut candidate_domains: Vec<String> = Vec::new();
                // 🌟 [CROSS-DOMAIN REQUEST] (host 도메인, substantial 필드, find 방향, 세그먼트 텍스트)
                let mut sub_host_requests: Vec<(String, String, String, String)> = Vec::new();

                // ── 1) 도메인 축 : 세그먼트를 '확정 타입'별로 그룹핑합니다. 절대 서로 섞지 않습니다.
                for seg in ctx_arr.iter() {
                    let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();

                    // "ignore"는 상거래 검색 조건이 아니므로 병합하지 않고 원형 그대로 보존합니다.
                    if seg_type == "ignore" {
                        final_contexts.push(seg.clone());
                        continue;
                    }
                    if seg_type.is_empty() { continue; }

                    // STAGE-1 이 남긴 교차 후보 도메인(types)은 리콜 보증 티어에서 사용합니다.
                    if let Some(types) = seg.get("types").and_then(|v| v.as_array()) {
                        for t in types {
                            if let Some(ts) = t.as_str() {
                                if ts.is_empty() || ts == "ignore" { continue; }
                                if !candidate_domains.iter().any(|d| d == ts) { candidate_domains.push(ts.to_string()); }
                            }
                        }
                    }

                    if let Some(host) = seg.get("substantial_host").and_then(|v| v.as_str()) {
                        if !host.is_empty()
                            && !group_order.iter().any(|d| d == host)
                            && !candidate_domains.iter().any(|d| d == host)
                        {
                            candidate_domains.push(host.to_string());
                        }
                        // 🌟 [CROSS-DOMAIN REQUEST] 지금까지는 host 를 candidate_domains 에만 넣어
                        //    '조건 없는' C/CANDIDATE 쿼리만 나갔습니다.
                        //    weight/many 같은 실제 조건을 담은 host 도메인 쿼리를 발행하기 위해
                        //    (host, substantial, find, 세그먼트 텍스트) 를 요청 목록으로 보관합니다.
                        if !host.is_empty() {
                            let sub_key = seg.get("substantial").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let find_key = seg.get("find").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let seg_text_for_host = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            if !sub_key.is_empty() {
                                sub_host_requests.push((host.to_string(), sub_key, find_key, seg_text_for_host));
                            }
                        }
                    }

                    if !group_order.iter().any(|g| g == &seg_type) { group_order.push(seg_type.clone()); }
                    let g = groups.entry(seg_type.clone()).or_insert_with(|| DomainGroup {
                        text_words: Vec::new(),
                        raw_words: Vec::new(),
                        value_words: Vec::new(),
                        condition: serde_json::Map::new(),
                        alternates: serde_json::Map::new(),
                        status: json!(""),
                        substantial: json!(""),
                        find: json!(""),
                    });

                    if let Some(text) = seg.get("text").and_then(|v| v.as_str()) {
                        for w in text.split_whitespace() {
                            if w == "|" { continue; }
                            // 🌟 [RAW AXIS] 정화 여부와 무관하게 원문은 항상 보존합니다.
                            //    ACTION VERB 게이트가 전 단어를 오탐해도 리콜 티어가 살아남습니다.
                            if !g.raw_words.iter().any(|e| e == w) { g.raw_words.push(w.to_string()); }

                            // 🌟 [ACTION WORD PURGE] 벡터로 확정된 순수 명령어는 FTS 노이즈입니다.
                            //    '찾아줘'/'보여줘' 가 ngram 검색어에 남으면 무관한 문서를 끌어옵니다.
                            //    역검증을 통과한 값/연산자/시간 표현은 이 집합에 없으므로 그대로 보존됩니다.
                            if global_action_words.contains(w) { continue; }
                            if !g.text_words.iter().any(|e| e == w) { g.text_words.push(w.to_string()); }
                        }
                    }

                    if let Some(status) = seg.get("status") {
                        let s_str = status.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && g.status.as_str().unwrap_or("").is_empty() {
                            g.status = status.clone();
                        }
                    }
                    if let Some(sub) = seg.get("substantial") {
                        let s_str = sub.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && g.substantial.as_str().unwrap_or("").is_empty() {
                            g.substantial = sub.clone();
                        }
                    }
                    if let Some(find) = seg.get("find") {
                        let s_str = find.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && g.find.as_str().unwrap_or("").is_empty() {
                            g.find = find.clone();
                        }
                    }

                    if let Some(alts) = seg.get("alternates").and_then(|v| v.as_object()) {
                        for (k, v) in alts {
                            if !g.alternates.contains_key(k) { g.alternates.insert(k.clone(), v.clone()); }
                        }
                    }

                    // 🌟 [UNASSIGNED RESCUE] 조건이 되지 못한 청크도 사용자가 실제로 입력한 단어이므로
                    //    완화 티어의 FTS 검색어에 반드시 포함시킵니다.
                    //    (로그: review 세그먼트의 '메세지도' 가 B/NARROWED·C/RECALL 에서 사라졌습니다)
                    //    🌟 [FILTER TERM 포함] FILTER TERM DROP 으로 속성 배정에서 제외된 단어
                    //    ('올해', '여름', '무거운', '많이' 등)도 이 경로로 FTS 검색어에 보존됩니다.
                    if let Some(un) = seg.get("unassigned").and_then(|v| v.as_array()) {
                        for u in un {
                            if let Some(us) = u.as_str() {
                                for w in us.split_whitespace() {
                                    if !g.value_words.iter().any(|e| e == w) { g.value_words.push(w.to_string()); }
                                }
                            }
                        }
                    }
                    // 🌟 [EXACT SEASON/TIME FTS 보존] exact_season / exact_time 으로 확정된 단어도
                    //    FTS 검색어에 포함되어야 합니다. (기존 FILTER TERM RESCUE 와 동일한 맥락)
                    if let Some(es) = seg.get("exact_season").and_then(|v| v.as_str()) {
                        if !es.is_empty() {
                            for w in es.split_whitespace() {
                                if !g.value_words.iter().any(|e| e == w) { g.value_words.push(w.to_string()); }
                            }
                        }
                    }
                    if let Some(et) = seg.get("exact_time").and_then(|v| v.as_str()) {
                        if !et.is_empty() {
                            for w in et.split_whitespace() {
                                if !g.value_words.iter().any(|e| e == w) { g.value_words.push(w.to_string()); }
                            }
                        }
                    }

                    if let Some(cond) = seg.get("condition").and_then(|v| v.as_object()) {
                        for (k, v) in cond {
                            // 값(value)이 비어있는 쓰레기 데이터는 무시하고, 유효한 값만 병합
                            // 🌟 value 키 자체가 없는 경우(None)도 반드시 '비어있음' 으로 판정해야
                            //    { "color": { "operator": "contains", "percent_total": "0.5" } } 같은
                            //    값 없는 조건이 최종 컨텍스트로 새어 나가지 않습니다.
                            let mut is_empty = match v.get("value") {
                                None => true,
                                Some(serde_json::Value::String(s)) => s.trim().is_empty() || s == "null",
                                Some(serde_json::Value::Null) => true,
                                Some(serde_json::Value::Object(o)) => {
                                    o.get("value").and_then(|val| val.as_str()).map_or(false, |s| s.trim().is_empty() || s == "null")
                                },
                                _ => false,
                            };

                            // 🌟 top, bottom 연산자는 percent_total 로 동작하므로 value 가 없어도 유효합니다.
                            if let Some(op) = v.get("operator").and_then(|o| o.as_str()) {
                                if op == "top" || op == "bottom" {
                                    is_empty = false;
                                }
                            }

                            if is_empty { continue; }

                            // 🌟 [COLLISION GUARD] 같은 도메인 안에서 동일 키가 충돌하면 먼저 확정된 값을 지킵니다.
                            //    기존 무조건 덮어쓰기가 review 의 title="고객의" 로 goods 의 title="제품" 을 지운 원인입니다.
                            if g.condition.contains_key(k) {
                                emit_term(&format!("    ⚠️ [CONDITION COLLISION] 도메인 '{}' 의 '{}' 조건이 중복되어 먼저 확정된 값을 유지합니다. (폐기: {})", seg_type, k, v));
                            } else {
                                g.condition.insert(k.clone(), v.clone());
                            }

                            if let Some(val_str) = v.get("value").and_then(|x| x.as_str()) {
                                for w in val_str.split_whitespace() {
                                    if w == "|" { continue; }
                                    if !g.value_words.iter().any(|e| e == w) { g.value_words.push(w.to_string()); }
                                }
                            }
                        }
                    }
                }

                // ── 2) [TRACKING NUMBER INJECTION] 감지된 송장 번호는 전용 tracking 도메인 그룹으로 독립시킵니다.
                //       기존처럼 마스터 타입을 tracking 으로 '승급' 시키면 goods 조회가 통째로 사라집니다.
                if !detected_tracking_numbers.is_empty() {
                    if !group_order.iter().any(|g| g == "tracking") { group_order.push("tracking".to_string()); }
                    let g = groups.entry("tracking".to_string()).or_insert_with(|| DomainGroup {
                        text_words: Vec::new(),
                        raw_words: Vec::new(),
                        value_words: Vec::new(),
                        condition: serde_json::Map::new(),
                        alternates: serde_json::Map::new(),
                        status: json!(""),
                        substantial: json!(""),
                        find: json!(""),
                    });
                    for tn in &detected_tracking_numbers {
                        g.condition.insert("tracking_number".to_string(), json!({
                            "operator": "contains",
                            "value": tn
                        }));
                        if !g.value_words.iter().any(|e| e == tn) { g.value_words.push(tn.clone()); }
                        if !g.text_words.iter().any(|e| e == tn) { g.text_words.push(tn.clone()); }
                        if !g.raw_words.iter().any(|e| e == tn) { g.raw_words.push(tn.clone()); }
                        emit_term(&format!("  📦 [TRACKING INJECT] tracking_number = '{}' 를 독립 tracking 도메인 컨텍스트로 발행합니다.", tn));
                    }
                }

                // 🌟 [DOMAIN AFFINITY] bias.json 의 search_bridge.domain_affinity 를 읽어
                //    확정 도메인의 친화 도메인을 candidate_domains 에 자동 주입합니다.
                //    (예: coupon 확정 → event 자동 추가, event 확정 → coupon 자동 추가)
                //    코사인 없이 bias.json 의 명시적 매핑만 따르므로 매직 상수가 없습니다.
                {
                    let domain_affinity: std::collections::HashMap<String, Vec<String>> = {
                        let dict = &crate::parsing::BIAS_DICT;
                        let mut map = std::collections::HashMap::new();
                        if let Some(aff_obj) = dict.get("search_bridge")
                            .and_then(|sb| sb.get("domain_affinity"))
                            .and_then(|v| v.as_object())
                        {
                            for (dom, targets) in aff_obj {
                                if let Some(arr) = targets.as_array() {
                                    let t: Vec<String> = arr.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect();
                                    if !t.is_empty() {
                                        map.insert(dom.clone(), t);
                                    }
                                }
                            }
                        }
                        map
                    };

                    for dom in &group_order {
                        if let Some(affiliated) = domain_affinity.get(dom) {
                            for aff in affiliated {
                                // 이미 확정 도메인이거나 후보에 있으면 skip
                                if group_order.iter().any(|d| d == aff) { continue; }
                                if candidate_domains.iter().any(|d| d == aff) { continue; }

                                candidate_domains.push(aff.clone());
                                emit_term(&format!(
                                    "  🔗 [DOMAIN AFFINITY] '{}' 확정 → 친화 도메인 '{}' 를 C/CANDIDATE 에 자동 포함",
                                    dom, aff
                                ));
                            }
                        }
                    }
                }

                // ── 3) 확정 도메인마다 3단 티어 발행
                //       A/FULL     : 전체 조건 + 세그먼트 원문 텍스트          (정밀)
                //       B/NARROWED : SQL 을 실제로 바꾸는 조건만 + 값 텍스트   (완화)
                //       C/RECALL   : 조건 없음 + 값 텍스트                     (리콜 보증)
                let ordered_domains = group_order.clone();
                for dom in &ordered_domains {
                    let (full_text, value_text, raw_text, cond, alts, st, sb, fd) = match groups.get(dom) {
                        Some(g) => {
                            let raw = g.raw_words.join(" ");
                            // 🌟 [PURGE COLLAPSE GUARD] ACTION VERB 게이트가 세그먼트의 모든 단어를
                            //    오탐하면 text_words 가 비고, push_ctx 가 A/FULL·B/NARROWED·C/RECALL·
                            //    E/FALLBACK 을 전부 거부하여 티어 구조가 통째로 무너집니다.
                            //    (new_log2.txt 실측: 발행 쿼리 7건 → 1건, 결과 20건 → 10건)
                            //    정화본이 비면 즉시 원문으로 복구합니다.
                            let purged = if g.text_words.is_empty() { raw.clone() } else { g.text_words.join(" ") };
                            let vt = if g.value_words.is_empty() {
                                purged.clone()
                            } else {
                                // 🌟 [WORD ORDER PRESERVE] condition 맵 순회로 뒤섞인 value_words 를
                                //    세그먼트 원문(raw_words) 순서로 복원한 뒤 조립합니다.
                                reorder_by_source(&g.value_words, &g.raw_words).join(" ")
                            };
                            if g.text_words.is_empty() && !raw.trim().is_empty() {
                                emit_term(&format!(
                                    "    🛟 [PURGE COLLAPSE GUARD] type={} | ACTION WORD 정화로 텍스트가 비어 원문으로 복구합니다. text=\"{}\"",
                                    dom, raw
                                ));
                            }
                            (
                                purged,
                                vt,
                                raw,
                                g.condition.clone(),
                                g.alternates.clone(),
                                g.status.clone(),
                                g.substantial.clone(),
                                g.find.clone(),
                            )
                        },
                        None => continue,
                    };

                    if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &full_text, cond.clone(), &alts, &st, &sb, &fd, "A/FULL") {
                        emit_term(&format!("    🅰️ [TIER A/FULL] type={} | conditions={} | text=\"{}\"", dom, cond.len(), full_text));
                    }

                    let mut narrowed = serde_json::Map::new();
                    for (k, v) in cond.iter() {
                        if crate::utils::ai_utils::is_sql_effective_field(k) {
                            narrowed.insert(k.clone(), v.clone());
                        }
                    }
                    if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &value_text, narrowed.clone(), &alts, &st, &sb, &fd, "B/NARROWED") {
                        emit_term(&format!("    🅱️ [TIER B/NARROWED] type={} | conditions={} | text=\"{}\"", dom, narrowed.len(), value_text));
                    }

                    // 🌟 [RAW RECALL] 최후 리콜 티어는 정화 이전 원문으로 조회합니다.
                    //    게이트 오탐으로 제거된 실질 명사('메세지도' 등)가 여기서 반드시 되살아납니다.
                    if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &raw_text, serde_json::Map::new(), &alts, &st, &sb, &fd, "C/RECALL") {
                        emit_term(&format!("    🅲 [TIER C/RECALL] type={} | 조건 없이 원문 FTS 로 재조회 | text=\"{}\"", dom, raw_text));
                    }

                    // 🌟 [TIER E/TABLE-FALLBACK] lib.rs 의 target_table 매핑과 scheduler.rs 의 저장 테이블이 어긋나면
                    //    데이터가 존재해도 영원히 0건이 됩니다.
                    //      lib.rs   : "event" | "coupon" | "review" => "event"
                    //      scheduler: "event" | "coupon" => "event",  나머지(review 등) => "items"
                    //    즉 review 는 items 에 저장되는데 event 에서 조회되어 구조적으로 절대 못 찾습니다.
                    //    (로그: review 두 티어 모두 Table: event / total_found: 0)
                    //    items 는 scheduler 가 모든 타입을 이중 upsert 하는 미러 테이블이므로,
                    //    매핑 오류와 무관하게 리콜을 보장하는 최후 경로가 됩니다.
                    //    lib.rs 의 target_table match 는 알 수 없는 타입을 items 로 보내므로
                    //    도메인 이름을 그대로 두고 tier 만 분기하면 별도 코드 변경 없이 동작합니다.
                    let fallback_domain = format!("{}_items", dom);
                    if push_ctx(&mut final_contexts, &mut seen_ctx, &fallback_domain, &raw_text, serde_json::Map::new(), &alts, &st, &sb, &fd, "E/TABLE-FALLBACK") {
                        emit_term(&format!("    🅴 [TIER E/TABLE-FALLBACK] type={} | items 미러 테이블을 원문으로 추가 조회합니다. (target_table 매핑 오류 보험)", fallback_domain));
                    }

                    // ── 4) 속성 대안 축 : 1순위 확정이 틀렸을 때를 대비한 대안 조합.
                    //       단, 문자열 속성은 lib.rs 의 convert_conditions_to_sql 에서 물리 컬럼으로 매핑되지 않아
                    //       쿼리가 완전히 동일해집니다. SQL 을 실제로 바꾸는 조합만 별도 발행하고,
                    //       나머지 대안은 alternates 메타데이터로 프론트엔드에 전달합니다. (쿼리 낭비 방지)
                    for (prop, alt_val) in alts.iter() {
                        let alt_list = match alt_val.as_array() { Some(a) => a, None => continue };
                        let alt_name = match alt_list.first().and_then(|v| v.as_str()) { Some(s) => s.to_string(), None => continue };
                        if !cond.contains_key(prop) { continue; }
                        if !crate::utils::ai_utils::is_sql_effective_field(&alt_name)
                            && !crate::utils::ai_utils::is_sql_effective_field(prop) { continue; }

                        let mut swapped = cond.clone();
                        if let Some(moved) = swapped.remove(prop) {
                            swapped.insert(alt_name.clone(), moved);
                        }
                        if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &full_text, swapped, &alts, &st, &sb, &fd, "D/ALTERNATE") {
                            emit_term(&format!("    🔀 [TIER D/ALTERNATE] type={} | '{}' → '{}' 로 교체한 대안 조합 발행", dom, prop, alt_name));
                        }
                    }
                }

                // ── 5) 교차 후보 도메인(types) 리콜 보증 : 조건 없이 순수 FTS 로만 조회합니다.
                //       STAGE-1 의 도메인 확정이 틀렸을 때 정답 테이블이 아예 조회조차 되지 않는 사고를 막습니다.
                let global_value_text = {
                    let mut w: Vec<String> = Vec::new();
                    for dom in &ordered_domains {
                        if let Some(g) = groups.get(dom) {
                            // 🌟 [RAW FALLBACK] value_words → text_words → raw_words 순으로 내려갑니다.
                            //    C/CANDIDATE 는 '교차 후보 도메인 리콜 보증' 티어이므로
                            //    ACTION WORD 정화로 인해 비는 일이 절대 없어야 합니다.
                            let src = if !g.value_words.is_empty() {
                                &g.value_words
                            } else if !g.text_words.is_empty() {
                                &g.text_words
                            } else {
                                &g.raw_words
                            };
                            for x in src { if !w.iter().any(|e| e == x) { w.push(x.clone()); } }
                        }
                    }
                    w.join(" ")
                };
                let empty_alts = serde_json::Map::new();
                let empty_val = json!("");

                // 🌟 [TIER D/CROSS-DOMAIN] substantial 필드를 실제로 보유한 도메인에
                //    '조건이 실린' 쿼리를 발행합니다.
                //    (로그: weight 는 goods 스키마에 없고 tracking 에 존재 → 그러나 조건 있는 tracking 쿼리는 0건이었음)
                //    find 방향이 확정되어 있으면 top/bottom 조건으로 물질화하고,
                //    방향이 없으면 조건 없이 도메인 리콜만 보증합니다.
                for (host, sub_field, find_key, seg_text_for_host) in &sub_host_requests {
                    let dir_op = match find_key.as_str() {
                        "heavy" | "many" | "much"   => "top",
                        "light" | "few"  | "little" => "bottom",
                        _ => "",
                    };
                    // 🌟 [SEGMENT SCOPED] 교차 도메인 쿼리의 FTS 텍스트는
                    //    그 추상 수식어가 나온 '그 세그먼트' 로 한정합니다.
                    //    기존에는 global_value_text 를 통째로 이어붙여
                    //    tracking 쿼리에 '이벤트로', '리뷰를', '고객의' 까지 들어갔습니다.
                    //    (new_log1.txt 921행: text 가 4개 세그먼트 전 단어 융합)
                    //    세그먼트 텍스트가 비어 있을 때만 전역 텍스트로 폴백합니다.
                    let host_text = {
                        let mut w: Vec<String> = Vec::new();
                        for x in seg_text_for_host.split_whitespace() {
                            if !w.iter().any(|e| e == x) { w.push(x.to_string()); }
                        }
                        if w.is_empty() {
                            for x in global_value_text.split_whitespace() {
                                if !w.iter().any(|e| e == x) { w.push(x.to_string()); }
                            }
                        }
                        w.join(" ")
                    };

                    let mut host_cond = serde_json::Map::new();
                    if !dir_op.is_empty() {
                        host_cond.insert(sub_field.clone(), json!({
                            "operator": dir_op,
                            "percent_total": "20.0",
                            "is_percent": true
                        }));
                    }

                    if push_ctx(&mut final_contexts, &mut seen_ctx, host, &host_text, host_cond.clone(), &empty_alts, &empty_val, &json!(sub_field), &json!(find_key), "D/CROSS-DOMAIN") {
                        emit_term(&format!(
                            "    🔀 [TIER D/CROSS-DOMAIN] type={} | '{}' 를 보유한 도메인에 조건({}) 쿼리를 발행합니다. | text=\"{}\"",
                            host, sub_field,
                            if dir_op.is_empty() { "없음".to_string() } else { format!("{} {} 20%", sub_field, dir_op) },
                            host_text
                        ));
                    }

                    // items 미러 폴백도 함께 발행하여 target_table 매핑 오류에 대비합니다.
                    let host_fallback = format!("{}_items", host);
                    if push_ctx(&mut final_contexts, &mut seen_ctx, &host_fallback, &host_text, serde_json::Map::new(), &empty_alts, &empty_val, &json!(sub_field), &json!(find_key), "E/CROSS-DOMAIN-FALLBACK") {
                        emit_term(&format!("    🅴 [TIER E/CROSS-DOMAIN-FALLBACK] type={} | items 미러 테이블로 교차 도메인 리콜을 보증합니다.", host_fallback));
                    }
                }
                for dom in &candidate_domains {
                    if ordered_domains.iter().any(|d| d == dom) { continue; }
                    if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &global_value_text, serde_json::Map::new(), &empty_alts, &empty_val, &empty_val, &empty_val, "C/CANDIDATE") {
                        emit_term(&format!("    🧭 [TIER C/CANDIDATE] type={} | STAGE-1 교차 후보 도메인을 조건 없이 추가 조회합니다.", dom));
                    }
                }
                // 🌟 [CROSS-DOMAIN VERB EXPANSION]
                //    STAGE-1 교차 범위(0.30 마진)에 들지 못해 types에서 누락된 도메인을
                //    세그먼트 텍스트와 도메인별 코사인 재검사로 구출합니다.
                //    "이벤트로 판매된"에서 "판매된"은 order 앵커("sales, purchase")와 코사인이 높으나
                //    STAGE-1 멀티패스 점수에서는 coupon/event 에 밀려 types에 포함되지 못했습니다.
                //    매직 상수 없이 '코사인 > 0 이고 아직 발행되지 않은 도메인' 조건만 사용합니다.
                //
                //    🌟 [DOMAIN WORD INDIVIDUAL CHECK] 세그먼트 전체 임베딩은
                //    '이벤트' 의미에 지배되어 order 코사인이 음수가 될 수 있습니다.
                //    (로그: '이벤트로 판매된' 세그먼트에서 order CROSS-VERB 미발동)
                //    세그먼트 내 도메인 지시어(domain_word_related)의 개별 코사인을
                //    추가로 검사하여, '판매된' → order 같은 관련 도메인을 구출합니다.
                {
                    let all_doms = ["order", "goods", "tracking", "review", "coupon", "event"];
                    for seg in ctx_arr.iter() {
                        let seg_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        if seg_text.trim().is_empty() { continue; }
                        let seg_type_val = seg.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let seg_types_arr: Vec<String> = seg.get("types").and_then(|v| v.as_array())
                            .map(|a| a.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        let seg_emb = self.get_embedding(seg_text.to_string()).await.unwrap_or(vec![0.0; 384]);
                        if seg_emb.iter().all(|&v| v == 0.0) { continue; }
                        for dom in &all_doms {
                            if seg_type_val == *dom { continue; }
                            if seg_types_arr.iter().any(|t| t == dom) { continue; }
                            if ordered_domains.iter().any(|d| d == dom) { continue; }
                            if candidate_domains.iter().any(|d| d == dom) { continue; }
                            let anchor_text = crate::parsing::get_page_type_classification_bias(dom, &query_lang);
                            let anchor_emb = self.get_embedding(anchor_text).await.unwrap_or(vec![0.0; 384]);
                            if anchor_emb.iter().all(|&v| v == 0.0) { continue; }
                            // 1차: 세그먼트 전체 코사인 (기존 경로)
                            let seg_cross_sim = cosine_similarity(&seg_emb, &anchor_emb);
                            // 2차: 세그먼트 내 도메인 지시어 개별 코사인
                            //    domain_word_related 에 기록된 관련 도메인 목록을 확인합니다.
                            //    🌟 SALES BRIDGE 에 의해 order 가 추가된 경우,
                            //    코사인이 음수여도 related 에 포함되어 있으면 발동합니다.
                            let mut word_cross_sim = 0.0f32;
                            let mut matched_word = String::new();
                            let mut bridge_forced = false;
                            for w in seg_text.split_whitespace() {
                                if let Some(related) = domain_word_related.get(w) {
                                    if related.iter().any(|r| r == dom) {
                                        // 이 단어가 대상 도메인과 관련됨이 STAGE-2에서 이미 확인됨
                                        let w_emb = self.get_embedding(w.to_string()).await.unwrap_or(vec![0.0; 384]);
                                        if !w_emb.iter().all(|&v| v == 0.0) {
                                            let ws = cosine_similarity(&w_emb, &anchor_emb);
                                            if ws > word_cross_sim {
                                                word_cross_sim = ws;
                                                matched_word = w.to_string();
                                            }
                                            // 🌟 [SALES BRIDGE FORCE] related 에 포함된 도메인은
                                            //    코사인 부호와 무관하게 CROSS-VERB 를 발동시킵니다.
                                            //    (브릿지가 이미 STAGE-2 에서 코사인 검증을 통과했으므로)
                                            if ws <= 0.0 {
                                                bridge_forced = true;
                                                if matched_word.is_empty() {
                                                    matched_word = w.to_string();
                                                    word_cross_sim = 0.01; // 최소 양수 부여
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let final_cross = seg_cross_sim.max(word_cross_sim);
                            // 🌟 [SALES BRIDGE FORCE] bridge_forced 가 true 면 코사인 > 0 조건을 우회합니다.
                            if final_cross > 0.0 || bridge_forced {
                                candidate_domains.push(dom.to_string());

                                // 🌟 [DEDUP FIX] 기존에는 global_value_text + 빈 조건으로 push 하여
                                //    이미 발행된 C/CANDIDATE 와 시그니처(도메인+텍스트+조건)가 완전히 같아
                                //    seen_ctx 에 의해 조용히 삭제되었습니다.
                                //    (로그에 '이벤트로 판매된' 의 order CROSS-VERB 가 한 줄도 안 찍힌 원인)
                                //    발동 근거가 된 세그먼트 텍스트를 앞에 붙여 시그니처를 분리하고,
                                //    동시에 FTS 정밀도도 함께 올립니다.
                                let cross_text = {
                                    let mut w: Vec<String> = Vec::new();
                                    for x in seg_text.split_whitespace() {
                                        if !w.iter().any(|e| e == x) { w.push(x.to_string()); }
                                    }
                                    for x in global_value_text.split_whitespace() {
                                        if !w.iter().any(|e| e == x) { w.push(x.to_string()); }
                                    }
                                    w.join(" ")
                                };

                                if push_ctx(&mut final_contexts, &mut seen_ctx, dom, &cross_text, serde_json::Map::new(), &empty_alts, &empty_val, &empty_val, &empty_val, "C/CROSS-VERB") {
                                    if bridge_forced && word_cross_sim <= seg_cross_sim {
                                        emit_term(&format!("    🔀 [TIER C/CROSS-VERB] type={} | 세그먼트 '{}' 내 도메인 지시어 '{}' 가 SALES BRIDGE 로 '{}' 도메인 관련 확정. 코사인 무관 추가 조회.", dom, seg_text, matched_word, dom));
                                    } else if word_cross_sim > seg_cross_sim && !matched_word.is_empty() {
                                        emit_term(&format!("    🔀 [TIER C/CROSS-VERB] type={} | 세그먼트 '{}' 내 도메인 지시어 '{}' 와 '{}' 도메인 코사인 {:+.4} 로 추가 조회합니다.", dom, seg_text, matched_word, dom, word_cross_sim));
                                    } else {
                                        emit_term(&format!("    🔀 [TIER C/CROSS-VERB] type={} | 세그먼트 '{}' 와 '{}' 도메인 코사인 {:+.4} 로 추가 조회합니다.", dom, seg_text, dom, seg_cross_sim));
                                    }
                                } else {
                                    emit_term(&format!("    ⚪ [CROSS-VERB DEDUP] type={} | 동일 시그니처 컨텍스트가 이미 발행되어 중복 발행을 건너뜁니다.", dom));
                                }
                            }
                        }
                    }
                }

                // ── 6) 최후 보루 : 확정 도메인이 하나도 없으면 goods 순수 FTS 라도 반드시 발행합니다.
                if final_contexts.iter().all(|c| c.get("type").and_then(|v| v.as_str()).unwrap_or("") == "ignore") {
                    let fallback_text = if global_value_text.trim().is_empty() { query.clone() } else { global_value_text.clone() };
                    if push_ctx(&mut final_contexts, &mut seen_ctx, "goods", &fallback_text, serde_json::Map::new(), &empty_alts, &empty_val, &empty_val, &empty_val, "C/FALLBACK") {
                        emit_term("    🛟 [TIER C/FALLBACK] 확정 도메인이 없어 goods 순수 FTS 컨텍스트를 최후 보루로 발행합니다.");
                    }
                }

                *ctx_arr = final_contexts;

                let query_count = ctx_arr.iter().filter(|c| c.get("type").and_then(|v| v.as_str()).unwrap_or("") != "ignore").count();
                emit_term(&format!("  ✅ [N:N COMBINATORIAL CONTEXTS] 발행 쿼리 {}건\n{}", query_count, serde_json::to_string_pretty(&ctx_arr).unwrap_or_default()));
            }
        }

        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Analysis complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        // 🌟 [VRAM 초기화 반영] 파이프라인 종료 직후 Embedding 및 Qwen3 모델을 메모리에서 완벽히 해제하여 VRAM을 0으로 떨어뜨립니다.
        emit_term("[ENGINE] 🧹 Purging models from memory to free VRAM...");
        
        // 🌟 [VRAM 누수 픽스] KV 캐시를 정상적으로 삭제하기 위해 None 덮어쓰기 로직을 제거하고, deep_purge_resources에 전부 일임합니다.
        self.deep_purge_resources().await;

        // 🌟 [강화된 VRAM 초기화] CUDA 메모리 캐시 강제 비우기 (컴파일 에러 해결 적용)
        if !self.is_cpu_mode {
            if self.device_config.device.is_cuda() {
                let _ = self.device_config.device.synchronize();
            }
            // 새 컨텍스트를 할당하여 기존 메모리 풀을 OS로 반환시킵니다.
            let _ = candle_core::Device::new_cuda(self.device_config.gpu_id as usize);
        }

        // 🌟 [CRITICAL FIX] scheduler.rs의 함수 대신 model.rs에 내장된 강력한 VRAM 스마트 폴링 모니터(self.wait_for_vram_settle)를 호출합니다.
        // 내부에서 OS 메모리 강제 반환을 폭격하여 VRAM을 0으로 만듭니다.
        self.wait_for_vram_settle(1200, 10, Some(cancel_token.clone())).await.ok();

        Ok(segments)
    }

    // [신규] Shipping 파이프라인 (빠른 단일 처리)
    pub async fn parse_shipping_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        // 🌟 [CRITICAL FIX] 매크로 제거 후 비동기 우회 함수 장착!
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let m = msg.to_string();
            let handle = app_handle_clone.clone();
            let tid = task_id_clone.clone();
            tokio::spawn(async move {
                use tauri::Emitter;
                let _ = handle.emit("task-console-log", serde_json::json!({"task_id": tid, "text": format!("{}\n", m)}));
            });
        };

        emit_term("\n=======================================");
        emit_term("[ENGINE] 🚀 Starting Shipping Search Pipeline...");

        let payload = json!({ "task_id": task_id, "category": "Shipping", "summary": "Extracting logistics filters...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3 (0.6B) Model...");
        self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancel_token.clone()), false, None).await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        emit_term(&format!("[STAGE-1] Extracting shipping filters from query: '{}'", query));
        let prompt = crate::parsing::extract_shipping_conditions(&query, language);
        let gen_arc = self.qwen3_generator.clone();
        let cancel_clone = cancel_token.clone();
        
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = gen_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![
                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                            name: None,
                        })
                    ],
                    model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };
                gen.generate(params, Some(cancel_clone), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
            } else {
                Err(anyhow::anyhow!("Qwen3 Generator is missing"))
            }
        }).await??;

        // 🌟 추출된 결과를 터미널 화면에 꽂아줍니다!
        emit_term(&format!("[STAGE-1 RESULT]\n{}", res));

        let extracted_conditions = crate::parsing::parse_json_from_llm(&res);
        
        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Filter extraction complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        let ctx = json!([{
            "type": "tracking",
            "text": query.clone(),
            "condition": extracted_conditions
        }]);

        emit_term("[SUCCESS] Shipping Search Pipeline Completed.");
        Ok(json!({ "context": ctx }))
    }

    // [신규] Analytic 파이프라인 (임시 Dummy 함수)
    pub async fn parse_analytic_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let m = msg.to_string();
            let handle = app_handle_clone.clone();
            let tid = task_id_clone.clone();
            tokio::spawn(async move {
                use tauri::Emitter;
                let _ = handle.emit("task-console-log", serde_json::json!({"task_id": tid, "text": format!("{}\n", m)}));
            });
        };

        emit_term("\n=======================================");
        emit_term("[ENGINE] 🚀 Starting Analytic Search Pipeline (Draft Mode)...");

        // UI에 스피너 표기
        let payload = json!({ "task_id": task_id, "category": "Analytic", "summary": "Running mock analytics...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        // 🌟 취소 버튼 즉시 반응 대응
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        // [TODO] 향후 여기에 통계 분석 전용 프롬프트 및 LLM 추론 로직 (Graph2Metrics 등) 추가 예정
        tokio::time::sleep(std::time::Duration::from_millis(500)).await; // 임시 대기

        emit_term(&format!("[STAGE-1] Dummy parsing analytic intent from query: '{}'", query));
        
        // 검색 쿼리에 걸리도록 임시 컨텍스트(sales 등)를 뱉어냅니다.
        let ctx = json!([{
            "type": "sales", // 검색을 위한 기본 타겟 테이블 (임시)
            "text": query.clone(),
            "condition": {}
        }]);

        let payload_done = json!({ "task_id": task_id, "category": "Done", "summary": "Analytic processing complete (Dummy).", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload_done);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload_done);

        emit_term("[SUCCESS] Analytic Search Pipeline Completed.");
        Ok(json!({ "context": ctx }))
    }

    // --- Ported from Python (search_engine.py) ---
    // --- Ported from Python (logic.py) ---
    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);

        // 1. Context Gathering
        status_history.push_str("✅ Context gathered.\n\n");
        // [LOG-ONLY]
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        // 2. Multi-step reasoning loop
        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for step in steps.iter() {
            status_history.push_str(&format!("**⏳ {}**\n", step));
            crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None, None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            crate::models::qwen::generate::wait_for_global_io().await;
            
            // 🌟 [신규 추가] GPU 비동기 연산 찌꺼기 강제 동기화
            if !self.is_cpu_mode {
                let dev = self.device_config.device.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() { let _ = dev.synchronize(); }
                }).await;
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
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None, None).await?;
        status_history.push_str(&report);
        
        // [LOG-ONLY]
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        Ok(report)
    }

//     fn get_search_schema_definitions(&self, _doc_type: &str) -> String {
//         r###"{ 
//   "header.document_type": { "desc": "Type (Invoice, BL, AWB, PO, BC, AN, DO...)", "type": "String" },
//   "header.document_number": { "desc": "ID, Doc No, Reference No", "type": "String" },
//   "header.po_number": { "desc": "Purchase Order No (PO)", "type": "String" },
//   "header.booking_number": { "desc": "Booking Reference No (BC)", "type": "String" },
//   "header.an_number": { "desc": "Arrival Notice No (AN)", "type": "String" },
//   "header.do_number": { "desc": "Delivery Order No (DO)", "type": "String" },
//   "header.issue_date": { "desc": "Date (YYYY-MM-DD)", "type": "String" },
  
//   "parties.supplier_name": { "desc": "Seller, Shipper, Exporter, Vendor", "type": "String" },
//   "parties.buyer_name": { "desc": "Buyer, Consignee, Importer", "type": "String" },
//   "parties.notify_party_name": { "desc": "Notify Party", "type": "String" },
  
//   "financials.amount_total": { "desc": "Total Value/Amount", "type": "Number" },
//   "financials.local_charges_total": { "desc": "Total Local Charges (AN)", "type": "Number" },
  
//   "logistics.vehicle_name": { "desc": "Vessel Name, Flight No", "type": "String" },
//   "logistics.location_port_of_loading": { "desc": "POL, Origin", "type": "String" },
//   "logistics.location_port_of_discharge": { "desc": "POD, Destination", "type": "String" },
//   "logistics.pickup_location": { "desc": "Pickup Location (DO)", "type": "String" },
//   "logistics.etd": { "desc": "Estimated Departure", "type": "String" },
//   "logistics.eta": { "desc": "Estimated Arrival", "type": "String" },
  
//   "conditions.incoterms_code": { "desc": "Incoterms (FOB, CIF)", "type": "String" }
// }"###.to_string()
//     }
}

fn merge_json_manual(root: &mut Map<String, Value>, cat: &str, data: Value) {
    let target_key = if cat == "items" { "line_items" } else if cat == "containers" { "containers" } else { cat };
    
    // Some models might wrap the result in the category name or target_key
    let actual_data = if let Some(inner) = data.get(target_key) { inner.clone() } 
                      else if let Some(inner) = data.get(cat) { inner.clone() } 
                      else { data };

    if let Some(target) = root.get_mut(target_key) {
        if target.is_array() {
            let target_arr = target.as_array_mut().unwrap();
            if let Some(source_arr) = actual_data.as_array() {
                for new_item in source_arr {
                    // Check for duplicates in line_items/containers by description/number
                    let is_dup = if target_key == "line_items" {
                        let new_desc = new_item.get("description").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("description").and_then(|v| v.as_str()).unwrap_or("") == new_desc)
                    } else if target_key == "containers" {
                        let new_no = new_item.get("container_number").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("container_number").and_then(|v| v.as_str()).unwrap_or("") == new_no)
                    } else { false };

                    if !is_dup { target_arr.push(new_item.clone()); }
                }
            }
        } else if let Some(target_obj) = target.as_object_mut() {
            if let Some(source_obj) = actual_data.as_object() {
                for (k, v) in source_obj {
                    if !v.is_null() && v != "" && v != 0 { target_obj.insert(k.clone(), v.clone()); }
                }
            }
        }
    }
}