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
    Granite, // Granite H 350M
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    pub granite_generator: Arc<TokioMutex<Option<crate::models::granite::generate::GraniteGenerateModel>>>,
    
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
    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut q3_gen = self.qwen3_generator.lock().await; 
        *q3_gen = None;
        let mut q35_gen = self.qwen3_5_generator.lock().await;
        *q35_gen = None;
        let mut granite_gen = self.granite_generator.lock().await;
        *granite_gen = None;
        
        let mut size = self.current_size.lock().await;
        *size = None;
        println!("[MODEL] All generators (Active) destroyed."); 
    }

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

        {
            let mut granite_gen = self.granite_generator.lock().await;
            if let Some(mut g) = granite_gen.take() {
                println!("[DIAG-PURGE] Dropping Granite Generator...");
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
                        if let Some(gen) = self.generator.lock().await.as_ref() {
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
                    ModelSize::Qwen3_5 => self.qwen3_5_generator.lock().await.is_some(),
                    ModelSize::Granite => self.granite_generator.lock().await.is_some(),
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
                        ModelSize::Granite => { self.ensure_granite_model().await?; }
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
            ModelSize::Granite => {
                self.ensure_granite_model().await?;
            }
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

    pub async fn ensure_granite_model(&self) -> anyhow::Result<()> {
        let granite_dir = crate::utils::get_app_dir().join("models").join("granite-4.0-h-350m");
        let weights_path = granite_dir.join("model.safetensors");
        
        if !weights_path.exists() {
            let err_msg = "Granite model is missing. Please go to the Settings tab and download the required models.";
            println!("[MODEL] 🚨 {}", err_msg);
            use tauri::Emitter;
            let _ = self.app_handle.emit("app_error_alert", serde_json::json!({ "message": err_msg }));
            return Err(anyhow::anyhow!(err_msg));
        }

        // 현재 로드된 모델이 Granite이면 건너뜀
        {
            let current = self.current_size.lock().await;
            if *current == Some(ModelSize::Granite) {
                return Ok(());
            }
        }
        
        // 기존 모델 언로드
        self.deep_purge_resources().await;
        
        // Granite H 350M 로드
        let model = crate::models::granite::generate::GraniteGenerateModel::init(
            &granite_dir.to_string_lossy(),
            Some(&self.device_config.device),
            None
        )?;
        
        *self.granite_generator.lock().await = Some(model);
        *self.current_size.lock().await = Some(ModelSize::Granite);
        
        Ok(())
    }

    pub async fn call_granite_model(&self, prompt: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let gen_arc_e = self.granite_generator.clone();
        
        // 🌟 Granite 템플릿(시스템 프롬프트 포함)을 적용하여 e_prompt_text 생성
        let e_sys = "You are a precise evaluation assistant. Return strictly the requested JSON format.";
        let e_user = prompt;
        let e_prompt_text = format!("<|start_of_role|>system<|end_of_role|>{}\n<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{}\n<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", e_sys, e_user);
        
        let dev_e = self.device_config.device.clone();
        let is_cpu = self.is_cpu_mode;
        
        let response = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = gen_arc_e.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                // KV 캐시 초기화 (메모리 누수 방지)
                gen.clear_kv_cache();
                
                // 컴파일 에러 해결: map_err를 catch_unwind 내부로 이동시켜 anyhow::Error 타입으로 통일
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gen.generate(&e_prompt_text, 256, &dev_e, cancel_token)
                        .map_err(|e| anyhow::anyhow!("Granite inference failed: {}", e))
                })).unwrap_or_else(|_| Err(anyhow::anyhow!("Granite generator panicked")));
                
                // 생성 종료 후 KV 캐시 즉시 비우기
                gen.clear_kv_cache();
                
                res
            } else {
                Err(anyhow::anyhow!("Granite model not loaded"))
            }
        }).await??;
        
        // CUDA VRAM 동기화 및 강제 해제 (GPU 모드일 때만)
        if !is_cpu {
            let dev = self.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || { 
                if dev.is_cuda() { 
                    let _ = dev.synchronize(); 
                } 
            }).await;
        }

        Ok(response)
    }

    pub async fn ensure_qwen3_5(&self, needs_vision: bool) -> anyhow::Result<()> {
        let needs_load = {
            let guard = self.qwen3_5_generator.lock().await;
            if let Some(gen) = guard.as_ref() {
                let is_large = gen.pre_processor.is_some();
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
            granite_generator: Arc::new(TokioMutex::new(None)),
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
        crate::scheduler::log_task_progress(app_handle, &task_id, &payload_load);

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
                        if cancel_token.as_ref().map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed)) { return Err(anyhow!("Cancelled")); }
                        
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
                "tracking" 
            };
            
            let masked_nl = nl.clone(); // 마스킹은 백엔드 push_data 단계에서 동적으로 수행됩니다.

            let item_digest = crate::utils::hash::digest(&nl);

            emit_term("[STAGE-3] Syncing extracted data to LanceDB...");

            // 🌟 [CRITICAL FIX 2] 5단계 마무리를 위한 저장 스텝(4단계) UI 추가!
            let payload_save = json!({ "task_id": task_id.clone(), "category": "Saving", "summary": "Syncing to database...", "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload_save);
            crate::scheduler::log_task_progress(app_handle, &task_id, &payload_save);

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
            crate::scheduler::log_task_progress(app_handle, &task_id, &payload);
            
            crate::scheduler::notify_new_task();
            
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
            crate::scheduler::log_task_progress(_app_handle, task_id, &base_payload); // 기존 변수명이 app_handle이면 app_handle로 사용
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
            crate::scheduler::log_task_progress(app_handle, task_id, &base_payload);
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
            crate::scheduler::log_task_progress(_app_handle, task_id, &base_payload);
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
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        
        // 🌟 [CRITICAL FIX] scheduler.rs의 *model_lock = None; 구조 완벽 이식
        {
            *self.generator.lock().await = None;
            *self.qwen3_generator.lock().await = None;
            *self.qwen3_5_generator.lock().await = None;
            *self.embedding_model.lock().await = None;
            *self.current_size.lock().await = None;
        }
        self.deep_purge_resources().await;
        self.wait_for_vram_settle(1200, 5, Some(cancel_token.clone())).await.ok();

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        
        // ----------------------------------------------------
        // Stage 1: 세그먼트 분할 (Vector Cliff Detection) - Embedding 모델 사용
        // ----------------------------------------------------
        emit_term("[STAGE-1] Loading Embedding Model for Semantic Chunking...");
        let payload = json!({ "task_id": task_id, "category": "Stage 1", "summary": "Segmenting semantic intents...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        self.ensure_embedding().await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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

        // 🌟 [추가] Stanza 기반 형태소 분석으로 검색어(query) 정밀 분할 반영
        let mut ext_words_string: Vec<String> = Vec::new();
        
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
                    
                    if !chars.is_empty() {
                        let seq_len = chars.len();
                        let mut char_ids = Vec::with_capacity(seq_len);
                        for c in &chars {
                            let id = *stanza.preprocessor.char_vocab.get(c).unwrap_or(&stanza.preprocessor.char_unk_id);
                            char_ids.push(id);
                        }
                        
                        if let Ok(char_tensor) = ndarray::Array2::from_shape_vec((1, seq_len), char_ids) {
                            let char_features = ndarray::Array3::<i64>::zeros((1, seq_len, 5));
                            let seq_lengths = ndarray::Array1::<i64>::from_vec(vec![seq_len as i64]);
                            
                            let mut tensor_pool = std::collections::HashMap::new();
                            tensor_pool.insert("char_tensor", char_tensor.into_dyn());
                            tensor_pool.insert("char_features", char_features.into_dyn());
                            tensor_pool.insert("seq_lengths", seq_lengths.into_dyn());
                            
                            let mut tok_inputs = Vec::new();
                            for input_meta in &stanza.tokenize_session.inputs {
                                let exact_name = input_meta.name.clone();
                                if let Some(tensor) = tensor_pool.get(exact_name.as_str()) {
                                    tok_inputs.push(tensor.clone());
                                } else {
                                    emit_term(&format!("[STANZA-WARN] Tokenizer 모델에 정의되지 않은 입력 생략: {}", exact_name));
                                }
                            }
                            
                            match stanza.tokenize_session.run::<'_, '_, '_, i64, f32, _>(tok_inputs) {
                                Ok(outputs) => {
                                    let output_tensor = &outputs[0];
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
                                                ext_words_string.push(token_str);
                                            }
                                            current_word.clear();
                                        }
                                    }
                                },
                                Err(e) => {
                                    emit_term(&format!("[STANZA-ERROR] Tokenizer run failed: {:?}", e));
                                }
                            }
                        }
                    }

                    // 🌟 [STANZA POS 사전 필터링 (scheduler.rs 로직 이식)]
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

                        if let Ok(pos_inputs) = stanza.preprocessor.encode_to_tensor(&padded_chunk, &stanza.pos_session) {
                            if let Ok(pos_outputs) = stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(pos_inputs) {
                                let output_tensor = &pos_outputs[0];
                                let shape = output_tensor.shape();
                                let mut pos_tags = Vec::new();

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
                                }

                                // 🌟 무의미한 형태소 기각 태그 설정
                                // ADJ(형용사 - 예: 베이지색)나 NOUN(명사 - 가디건)은 커머스 검색 핵심이므로 보존합니다.
                                // VERB(찾아줘, 알려줘, 보여줘) 및 각종 조사/기호를 제거합니다.
                                let drop_tags = ["VERB", "ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON"];
                                let mut dropped_log = Vec::new();
                                let mut filtered_words = Vec::new();

                                for (i, word) in ext_words_string.iter().enumerate() {
                                    let tag = pos_tags[i];
                                    
                                    // 해당 단어가 드롭 태그(예: VERB)에 속하면 제거
                                    if drop_tags.contains(&tag) {
                                        dropped_log.push(format!("{}({})", word, tag));
                                    } else {
                                        filtered_words.push(word.clone());
                                    }
                                }

                                if !dropped_log.is_empty() {
                                    emit_term(&format!("  ✂️ [STANZA-SEARCH-POS] 검색어에서 무의미한 단어 사전 제거 완료: {:?}", dropped_log));
                                }
                                
                                // 🌟 필터링 결과가 전부 다 날아가버리면 원본을 유지 (과잉 삭제로 인한 크래시 방어)
                                if !filtered_words.is_empty() {
                                    ext_words_string = filtered_words;
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    emit_term(&format!("[STANZA] ⚠️ Failed to load Stanza models for '{}' (상세 원인): {:?}", stanza_lang_code, e));
                }
            }
        }

        if ext_words_string.is_empty() {
            ext_words_string = query.split_whitespace().map(|s| s.to_string()).collect();
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
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
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
            let contexts = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
            let mut core_text = String::new();
            for (key, bias, _) in contexts {
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
            for (start, end, best_cat, context_score, mut intersecting) in final_bounds {
                let final_text = words[start..end].join(" ");
                
                // 🌟 types 배열에 best_cat이 무조건 포함되도록 보장하고 중복 제거
                if !intersecting.contains(&best_cat) {
                    intersecting.push(best_cat.clone());
                }
                intersecting.sort();
                intersecting.dedup();
                
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

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // ----------------------------------------------------
        // Stage 2 & 3: Double Plinko Attribute/Operator Mapping & LLM Normalization
        // ----------------------------------------------------
        emit_term("[STAGE-2] Extracting attributes via Double Vector Plinko & LLM Normalization...");
        
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let payload = json!({ "task_id": task_id, "category": format!("Stage 2 ({}/{})", idx+1, total_segments), "summary": "Mapping attributes...", "spinner": "⠋" });
                let _ = app_handle.emit("extraction-progress", &payload);
                crate::scheduler::log_task_progress(app_handle, task_id, &payload);

                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
                    let type_str = if desc.contains("Number") { "Number" }
                                   else if desc.contains("Boolean") { "Boolean" }
                                   else if desc.contains("Array") { "Array" }
                                   else { "String" };
                    prop_types.insert(key, type_str);
                }

                // 🌟 [3차 분기] 동적 필터 카테고리 일괄 로드 (bias.json 구조 완전 동기화)
                let filter_categories = vec![
                    "operators", "metrics", "time_filters", "season_filters", 
                    "status_filters", "substantial_filters", "find_filters"
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

                // Batch Embedding 동시 장전 (스키마 프로퍼티 + 동적 필터들)
                let bias_embs = self.get_embedding_batch(bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; prop_keys.len()]);
                let prej_embs = self.get_embedding_batch(prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; prop_keys.len()]);
                
                let dynamic_bias_embs = self.get_embedding_batch(dynamic_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; dynamic_filter_defs.len()]);
                let dynamic_prej_embs = self.get_embedding_batch(dynamic_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; dynamic_filter_defs.len()]);

                // 🌟 Plinko Game (1st Depth): Sliding Window Cliff Detection over words
                let mut plinko_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
                let words: Vec<&str> = current_text.split_whitespace().collect();
                
                let mut current_chunk = Vec::new();
                let mut prev_max_score = -1.0;
                let mut best_prop_for_chunk = String::new();

                emit_term(&format!("  🎯 [PLINKO GAME (1st)] Starting Sliding Window Cliff Detection for '{}'", current_text));

                for word in words {
                    let mut test_chunk = current_chunk.clone();
                    test_chunk.push(word);
                    let test_text = test_chunk.join(" ");
                    let test_emb = self.get_embedding(test_text.clone()).await.unwrap_or(vec![0.0; 384]);

                    // 🌟 [추가] 1차 핀볼(속성 매칭)에도 동사 페널티(verb_penalty) 및 단어 길이 가중치 적용
                    let word_count = test_chunk.len();
                    let v_sim = cosine_similarity(&test_emb, &verb_emb);
                    let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                    let verb_penalty = v_sim * beta;
                    let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };

                    let mut current_max = -1.0;
                    let mut current_best = String::new();

                    for i in 0..prop_keys.len() {
                        let b_score = cosine_similarity(&test_emb, &bias_embs[i]);
                        let p_score = cosine_similarity(&test_emb, &prej_embs[i]);
                        
                        // 🌟 [수정] 단순 차감이 아닌, 페널티 가중치와 동사 페널티를 결합하여 노이즈 차단
                        let score = b_score - (p_score * penalty_weight) - verb_penalty;
                        
                        if score > current_max {
                            current_max = score;
                            current_best = prop_keys[i].clone();
                        }
                    }

                    emit_term(&format!("    🔍 [PLINKO SLIDE] '{}' -> Match: [{}] (Score: {:.4})", test_text, current_best, current_max));

                    // Score Drop (Cliff) = Cut & Drop into Slot
                    if current_max < prev_max_score && !current_chunk.is_empty() {
                        emit_term(&format!("    📉 [CLIFF DETECTED] Score dropped ({:.4} -> {:.4}). End of semantic chunk.", prev_max_score, current_max));
                        
                        // 🌟 임계값을 0.10에서 0.05로 대폭 완화하여 단문(1단어 등)이 무시되는 현상을 막습니다.
                        if prev_max_score > 0.05 && !best_prop_for_chunk.is_empty() {
                            emit_term(&format!("      📥 [DROPPED INTO SLOT] '{}' belongs to property [{}]", current_chunk.join(" "), best_prop_for_chunk));
                            plinko_map.entry(best_prop_for_chunk.clone()).or_default().push(current_chunk.join(" "));
                        } else {
                            emit_term(&format!("      🗑️ [SKIPPED] Score {:.4} is too low. Ignored.", prev_max_score));
                        }
                        
                        // Reset Window
                        current_chunk = vec![word];
                        let reset_emb = self.get_embedding(word.to_string()).await.unwrap_or(vec![0.0; 384]);
                        
                        // 🌟 [추가] 리셋 윈도우의 단일 단어에도 동사 페널티 일관되게 적용
                        let r_v_sim = cosine_similarity(&reset_emb, &verb_emb);
                        let r_verb_penalty = r_v_sim * 0.05;
                        let r_penalty_weight = 0.3;

                        let mut r_max = -1.0;
                        let mut r_best = String::new();
                        for i in 0..prop_keys.len() {
                            let b_score = cosine_similarity(&reset_emb, &bias_embs[i]);
                            let p_score = cosine_similarity(&reset_emb, &prej_embs[i]);
                            
                            // 🌟 [수정] 페널티 가중치 적용
                            let score = b_score - (p_score * r_penalty_weight) - r_verb_penalty;
                            
                            if score > r_max {
                                r_max = score;
                                r_best = prop_keys[i].clone();
                            }
                        }
                        prev_max_score = r_max;
                        best_prop_for_chunk = r_best.clone(); // 🌟 [COMPILATION FIX] 소유권 이동(move) 방지를 위해 clone() 추가
                        emit_term(&format!("    🔄 [WINDOW RESET] Started new chunk '{}' -> Top Property: {} (Score: {:.4})", word, r_best, r_max));
                    } else {
                        current_chunk.push(word);
                        prev_max_score = current_max;
                        best_prop_for_chunk = current_best;
                    }
                }
                
                // Sweep remaining chunk
                if !current_chunk.is_empty() && prev_max_score > 0.05 && !best_prop_for_chunk.is_empty() {
                    emit_term(&format!("    🧹 [SWEEP REMAINING] Final chunk '{}' belongs to property [{}] (Score: {:.4})", current_chunk.join(" "), best_prop_for_chunk, prev_max_score));
                    plinko_map.entry(best_prop_for_chunk).or_default().push(current_chunk.join(" "));
                }

                // Granite H 350M으로 1차 매핑 검증
                if !plinko_map.is_empty() {
                    emit_term("    🧠 [GRANITE VERIFICATION (1st)] Verifying property mappings...");
                    // Granite H 350M 로드
                    self.ensure_granite_model().await?;
                    
                    // plinko_map의 각 항목을 Granite H 350M으로 검증
                    let mut validated_map = std::collections::HashMap::new();
                    for (prop, chunks) in &plinko_map {
                        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                        
                        let combined = chunks.join(" ");
                        let prompt = crate::prompts::verify_property_mapping_prompt(&combined, prop);
                        
                        // Granite H 350M 호출
                        if let Ok(response) = self.call_granite_model(&prompt, Some(cancel_token.clone())).await {
                            if let Ok(result) = serde_json::from_str::<Value>(&response) {
                                if result.get("correct").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    emit_term(&format!("      ✅ Property [{}] confirmed for '{}'", prop, combined));
                                    validated_map.entry(prop.clone()).or_insert_with(Vec::new).push(combined);
                                } else if let Some(suggested) = result.get("suggested_property").and_then(|v| v.as_str()) {
                                    emit_term(&format!("      🔄 Property [{}] corrected to [{}] for '{}'", prop, suggested, combined));
                                    validated_map.entry(suggested.to_string()).or_insert_with(Vec::new).push(combined);
                                } else {
                                    validated_map.entry(prop.clone()).or_insert_with(Vec::new).push(combined);
                                }
                            } else {
                                validated_map.entry(prop.clone()).or_insert_with(Vec::new).push(combined);
                            }
                        } else {
                            validated_map.entry(prop.clone()).or_insert_with(Vec::new).push(combined);
                        }
                    }
                    plinko_map = validated_map;
                }

                // 🌟 [IGNORE VECTOR CHECK] 현재 청크 전체가 명령어/분석 요청(ignore)에 해당하는지 검증
                let chunk_full_emb = self.get_embedding(current_text.clone()).await.unwrap_or(vec![0.0; 384]);
                let chunk_word_count = current_text.split_whitespace().count();
                let v_sim = cosine_similarity(&chunk_full_emb, &verb_emb);
                let beta = if chunk_word_count <= 2 { 0.05 } else { 0.10 };
                let penalty_weight = if chunk_word_count <= 2 { 0.3 } else { 0.7 };

                let mut is_ignore_chunk = false;
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

                let mut best_status_global = String::new();
                let mut best_sub_global = String::new();
                let mut best_find_global = String::new();
                let mut best_time_global = String::new();
                let mut best_season_global = String::new();

                emit_term("\n  🎯 [DOUBLE PLINKO (2nd)] Matching attributes and operators...");

                for (k, v) in &plinko_map {
                    let combined_chunk = v.join(" | ");
                    
                    let chunk_emb = self.get_embedding(combined_chunk.clone()).await.unwrap_or(vec![0.0; 384]);
                    let cw_count = v.len();
                    let v_sim_local = cosine_similarity(&chunk_emb, &verb_emb);
                    let local_beta = if cw_count <= 2 { 0.05 } else { 0.10 };
                    let local_vp = v_sim_local * local_beta;
                    let local_pw = if cw_count <= 2 { 0.3 } else { 0.7 };

                    let mut best_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                    let mut best_matches: std::collections::HashMap<String, String> = std::collections::HashMap::new();

                    // 커트라인 초기화 (카테고리별로 점수 한계 설정)
                    for cat in &filter_categories {
                        best_scores.insert(cat.to_string(), 0.15);
                    }

                    // 🌟 Part 1에서 준비된 통합 벡터(dynamic_filter_defs) 순회
                    for i in 0..dynamic_filter_defs.len() {
                        let def = &dynamic_filter_defs[i];
                        let b_score = cosine_similarity(&chunk_emb, &dynamic_bias_embs[i]);
                        let p_score = cosine_similarity(&chunk_emb, &dynamic_prej_embs[i]);
                        let score = b_score - (p_score * local_pw) - local_vp;

                        if score > *best_scores.get(&def.category).unwrap_or(&0.15) {
                            best_scores.insert(def.category.clone(), score);
                            best_matches.insert(def.category.clone(), def.key.clone());
                        }
                    }

                    let best_op = best_matches.get("operators").cloned().unwrap_or_else(|| "eq".to_string());
                    let best_metric = best_matches.get("metrics").cloned().unwrap_or_else(|| "string".to_string());
                    
                    if let Some(s) = best_matches.get("status_filters") { best_status_global = s.clone(); }
                    if let Some(s) = best_matches.get("substantial_filters") { best_sub_global = s.clone(); }
                    if let Some(s) = best_matches.get("find_filters") { best_find_global = s.clone(); }
                    if let Some(s) = best_matches.get("time_filters") { best_time_global = s.clone(); }
                    if let Some(s) = best_matches.get("season_filters") { best_season_global = s.clone(); }

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
                    }

                    prop_to_op.insert(k.clone(), final_op.clone());

                    let guide_log = format!("Target Text: \"{}\" -> Vector Suggests: Property [{}], Operator [{}], Metric Type [{}]", combined_chunk, k, final_op, final_metric);
                    emit_term(&format!("    🧲 {}", guide_log));
                    fragments_text.push_str(&format!("{}\n", guide_log));
                }

                // Granite H 350M으로 2차 매핑 검증
                if !prop_to_op.is_empty() {
                    emit_term("    🧠 [GRANITE VERIFICATION (2nd)] Verifying operators...");
                    self.ensure_granite_model().await?;
                    
                    // 속성별 operator 검증
                    let mut validated_prop_to_op = prop_to_op.clone();
                    for (prop, op) in &prop_to_op {
                        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                        
                        let prompt = crate::prompts::verify_operator_mapping_prompt(prop, op);
                        
                        if let Ok(response) = self.call_granite_model(&prompt, Some(cancel_token.clone())).await {
                            if let Ok(result) = serde_json::from_str::<Value>(&response) {
                                if !result.get("correct").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    if let Some(suggested) = result.get("suggested_operator").and_then(|v| v.as_str()) {
                                        emit_term(&format!("      🔄 Operator for [{}] corrected from [{}] to [{}]", prop, op, suggested));
                                        validated_prop_to_op.insert(prop.clone(), suggested.to_string());
                                    }
                                } else {
                                    emit_term(&format!("      ✅ Operator [{}] confirmed for [{}]", op, prop));
                                }
                            }
                        }
                    }
                    
                    prop_to_op = validated_prop_to_op;
                }
                
                // Vector-based Time/Season Guide 조립
                let mut llm_temporal_guide = String::new();
                if !best_time_global.is_empty() { llm_temporal_guide.push_str(&format!("Time Intent [{}] ", best_time_global)); }
                if !best_season_global.is_empty() { llm_temporal_guide.push_str(&format!("Season Intent [{}]", best_season_global)); }
                
                if !best_status_global.is_empty() { fragments_text.push_str(&format!("Global Status Suggests [{}]\n", best_status_global)); }
                if !best_sub_global.is_empty() { fragments_text.push_str(&format!("Global Substantial Suggests [{}]\n", best_sub_global)); }
                if !best_find_global.is_empty() { fragments_text.push_str(&format!("Global Find Suggests [{}]\n", best_find_global)); }

                emit_term(&format!("\n  🎯 [FINAL VECTOR GUIDE FOR LLM] \n{}", fragments_text.trim()));

                // Vector 기반으로 도출된 의도를 바탕으로 Deterministic Time Guide(달력 SQL 필터) 획득
                let (deterministic_guide_log, deterministic_json) = crate::parsing::get_deterministic_time_guide(&llm_temporal_guide, language);
                if !deterministic_guide_log.is_empty() {
                    emit_term(&format!("  ⏳ [DETERMINISTIC TIME GUIDE]\n  {}", deterministic_guide_log.replace("\n", "\n  ")));
                }
                
                // Vector-based Time/Season Guide 조립
                let mut llm_temporal_guide = String::new();
                if !best_time_global.is_empty() { llm_temporal_guide.push_str(&format!("Time Intent [{}] ", best_time_global)); }
                if !best_season_global.is_empty() { llm_temporal_guide.push_str(&format!("Season Intent [{}]", best_season_global)); }
                
                if !best_status_global.is_empty() { fragments_text.push_str(&format!("Global Status Suggests [{}]\n", best_status_global)); }
                if !best_sub_global.is_empty() { fragments_text.push_str(&format!("Global Substantial Suggests [{}]\n", best_sub_global)); }
                if !best_find_global.is_empty() { fragments_text.push_str(&format!("Global Find Suggests [{}]\n", best_find_global)); }

                emit_term(&format!("  🎯 [PLINKO MAP (VECTOR GUIDE)] \n{}", fragments_text.trim()));

                // Vector 기반으로 도출된 의도를 바탕으로 Deterministic Time Guide(달력 SQL 필터) 획득
                let (deterministic_guide_log, deterministic_json) = crate::parsing::get_deterministic_time_guide(&llm_temporal_guide, language);
                if !deterministic_guide_log.is_empty() {
                    emit_term(&format!("  ⏳ [DETERMINISTIC TIME GUIDE]\n  {}", deterministic_guide_log.replace("\n", "\n  ")));
                }

                // 🌟 [VRAM/RAM 최적화] 임베딩 연산(벡터 매칭)이 모두 끝났으므로 LLM 추론을 시작하기 전에 명시적으로 해제합니다.
                self.unload_embedding().await;

                // 5. LLM Normalization (Advanced Parser Mode with Vector Guide)
                if !fragments_text.is_empty() {
                    let now = chrono::Local::now();
                    let time_context = format!("Current Time: {}\nTimezone: {}\nLanguage: {}", now.format("%Y-%m-%dT%H:%M:%S"), now.format("%z"), language);

                    // 🌟 [CRITICAL FIX] 벡터 매칭 가이드와 LLM 시간 가이드를 병합하여 최종 조건 추출 프롬프트 호출
                    let combined_guide = format!("{}\n{}", fragments_text.trim(), llm_temporal_guide);
                    
                    let prompt_numeric = crate::parsing::extract_numeric_conditions(&current_text, &seg_type, metrics_json, &combined_guide, &time_context, language);
                    let prompt_status = crate::parsing::extract_status_intent_prompt(&current_text, &seg_type, &combined_guide);
                    let prompt_substantial = crate::parsing::extract_substantial_intent_prompt(&current_text, &combined_guide);
                    let prompt_find = crate::parsing::extract_find_intent_prompt(&current_text, &combined_guide);

                    // 🌟 [CRITICAL FIX] Qwen3 대신 Granite H 350M 모델을 사용하여 메모리 사용량을 줄이고 통일화합니다.
                    self.ensure_granite_model().await?;

                    // call_granite_model을 통해 순차적으로 LLM Normalization 수행
                    let res_numeric = self.call_granite_model(&prompt_numeric, Some(cancel_token.clone())).await?;
                    let res_status = self.call_granite_model(&prompt_status, Some(cancel_token.clone())).await?;
                    let res_substantial = self.call_granite_model(&prompt_substantial, Some(cancel_token.clone())).await?;
                    let res_find = self.call_granite_model(&prompt_find, Some(cancel_token.clone())).await?;

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
                        
                        // 객체 형태("condition": { "price": {"operator": "gt", "value": 1000} }) 메인 대응
                        if let Some(cond_val) = final_numeric_json.get("condition").and_then(|v| v.as_object()) {
                            for (k, val) in cond_val {
                                if deterministic_json.is_some() && (k == "started_at" || k == "expired_at" || k == "registration_date" || k == "date") {
                                    continue;
                                }

                                if val.is_object() {
                                    let mut final_val_obj = val.clone();
                                    if let Some(v_obj) = final_val_obj.as_object_mut() {
                                        if !v_obj.contains_key("operator") {
                                            let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                            v_obj.insert("operator".to_string(), json!(op));
                                        }
                                    }
                                    structured_cond.insert(k.clone(), final_val_obj);
                                } else {
                                    let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                    structured_cond.insert(k.clone(), json!({
                                        "operator": op,
                                        "value": val.clone()
                                    }));
                                }
                            }
                        } else if let Some(cond_arr) = final_numeric_json.get("condition").and_then(|v| v.as_array()) {
                            // 구형 프롬프트(배열 반환)로 응답했을 경우의 방어 로직
                            for item in cond_arr {
                                if let Some(item_obj) = item.as_object() {
                                    if let Some(prop_val) = item_obj.get("property").or_else(|| item_obj.get("property_name")).and_then(|v| v.as_str()) {
                                        let k = prop_val.to_string();
                                        
                                        if deterministic_json.is_some() && (k == "started_at" || k == "expired_at" || k == "registration_date" || k == "date") {
                                            continue;
                                        }

                                        let op = item_obj.get("operator").and_then(|v| v.as_str())
                                            .unwrap_or_else(|| prop_to_op.get(&k).map(|s| s.as_str()).unwrap_or("eq"));
                                        
                                        let mut final_val_obj = serde_json::Map::new();
                                        for (ik, iv) in item_obj {
                                            if ik != "property" && ik != "property_name" {
                                                final_val_obj.insert(ik.clone(), iv.clone());
                                            }
                                        }
                                        
                                        if !final_val_obj.contains_key("operator") {
                                            final_val_obj.insert("operator".to_string(), json!(op));
                                        }
                                        
                                        structured_cond.insert(k, json!(final_val_obj));
                                    } else {
                                        for (k, val) in item_obj {
                                            if deterministic_json.is_some() && (k == "started_at" || k == "expired_at" || k == "registration_date" || k == "date") {
                                                continue;
                                            }

                                            let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                            let mut final_val_obj = val.clone();

                                            if let Some(v_obj) = final_val_obj.as_object_mut() {
                                                if !v_obj.contains_key("operator") {
                                                    v_obj.insert("operator".to_string(), json!(op));
                                                }
                                            } else {
                                                final_val_obj = json!({
                                                    "operator": op,
                                                    "value": val.clone()
                                                });
                                            }
                                            structured_cond.insert(k.clone(), final_val_obj);
                                        }
                                    }
                                }
                            }
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

        // 🌟 [CRITICAL FIX] 분할된 컨텍스트(Segment)들의 추출 결과를 Rust 메모리에서 하나의 마스터 객체로 완벽히 취합(Merge)합니다.
        // 이를 통해 앞단에서 조각난 맥락(예: '구매 전환율이' + '1% 미만인')이 하나의 강력한 AND 조건으로 결합되어 DB 검색 시 정보 유실을 원천 차단합니다.
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            if ctx_arr.len() > 1 {
                emit_term("[STAGE-3] Merging all fragmented conditions into a single master context...");
                let mut master_condition = serde_json::Map::new();
                let mut master_text = String::new();
                let mut master_status = json!("");
                let mut master_substantial = json!("");
                let mut master_find = json!("");
                let mut master_type = String::new();
                
                let mut final_contexts = Vec::new(); // 🌟 분리 보존을 위한 새 배열

                for seg in ctx_arr.iter() {
                    let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    
                    // 🌟 [CRITICAL FIX] "ignore"는 상거래 검색 조건이 아니므로 병합(Merge)하지 않고 독립된 개체로 따로 빼둡니다.
                    if seg_type == "ignore" {
                        final_contexts.push(seg.clone());
                        continue;
                    }

                    // 첫 번째 유효한 도메인 타입을 마스터로 고정
                    if master_type.is_empty() {
                        master_type = seg_type.to_string();
                    }
                    
                    // 텍스트는 띄어쓰기로 이어 붙임
                    let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        if !master_text.is_empty() { master_text.push_str(" "); }
                        master_text.push_str(text);
                    }
                    
                    // 상태값 병합 (첫 번째 유효값 우선)
                    if let Some(status) = seg.get("status") {
                        let s_str = status.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && master_status.as_str().unwrap_or("").is_empty() {
                            master_status = status.clone();
                        }
                    }
                    if let Some(sub) = seg.get("substantial") {
                        let s_str = sub.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && master_substantial.as_str().unwrap_or("").is_empty() {
                            master_substantial = sub.clone();
                        }
                    }
                    if let Some(find) = seg.get("find") {
                        let s_str = find.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && master_find.as_str().unwrap_or("").is_empty() {
                            master_find = find.clone();
                        }
                    }
                    
                    // 추출된 조건(Condition) 객체들 병합
                    if let Some(cond) = seg.get("condition").and_then(|v| v.as_object()) {
                        for (k, v) in cond {
                            // 값(value)이 비어있는 쓰레기 데이터는 무시하고, 유효한 값만 병합
                            let is_empty = match v.get("value") {
                                Some(serde_json::Value::String(s)) => s.trim().is_empty() || s == "null",
                                Some(serde_json::Value::Null) => true,
                                Some(serde_json::Value::Object(o)) => {
                                    o.get("value").and_then(|val| val.as_str()).map_or(false, |s| s.trim().is_empty() || s == "null")
                                },
                                _ => false,
                            };
                            
                            if !is_empty {
                                master_condition.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
                
                // 만약 모든 청크가 ignore가 아니었다면 마스터 컨텍스트를 추가합니다.
                if !master_text.is_empty() || !master_type.is_empty() {
                    if master_type.is_empty() { master_type = "goods".to_string(); }

                    let master_ctx = json!({
                        "type": master_type,
                        "text": master_text,
                        "status": master_status,
                        "substantial": master_substantial,
                        "find": master_find,
                        "condition": master_condition
                    });
                    
                    // 마스터 컨텍스트를 배열의 맨 앞에 삽입
                    final_contexts.insert(0, master_ctx);
                }
                
                *ctx_arr = final_contexts;
                
                emit_term(&format!("  ✅ [MASTER MERGED CONTEXT]\n{}", serde_json::to_string_pretty(&ctx_arr).unwrap_or_default()));
            }
        }

        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Analysis complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        // 🌟 [VRAM 초기화 반영] 파이프라인 종료 직후 Embedding 및 Qwen3 모델을 메모리에서 완벽히 해제하여 VRAM을 0으로 떨어뜨립니다.
        emit_term("[ENGINE] 🧹 Purging models from memory to free VRAM...");
        
        // 🌟 [CRITICAL FIX] scheduler.rs와 완벽히 동일한 수준의 VRAM 초기화 적용
        // 모든 모델 객체 참조를 명시적으로 해제하여 메모리 누수를 원천 차단합니다.
        {
            *self.generator.lock().await = None;
            *self.qwen3_generator.lock().await = None;
            *self.qwen3_5_generator.lock().await = None;
            *self.embedding_model.lock().await = None;
            *self.current_size.lock().await = None;
        }
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
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3 (0.6B) Model...");
        self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancel_token.clone()), false, None).await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

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
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        // 🌟 취소 버튼 즉시 반응 대응
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
        crate::scheduler::log_task_progress(app_handle, task_id, &payload_done);

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
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        // 2. Multi-step reasoning loop
        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for step in steps.iter() {
            status_history.push_str(&format!("**⏳ {}**\n", step));
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None, None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

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
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

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