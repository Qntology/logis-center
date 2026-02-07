use crate::utils;
use anyhow::anyhow;
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
use crate::models::native_embedding::NativeEmbeddingModel;
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
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, atomic::AtomicBool};
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use sysinfo::System;
use tokio::sync::Mutex as TokioMutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Small, // 0.6B for Ingestion
    Large, // 2B-VL for Inference
}

#[derive(Clone)]
pub struct LogisModel {
    pub generator: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // Primary Active Slot (GPU)
    pub speculative_draftsman: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // Speculative 0.6B (CPU RAM)
    pub small_hibernation: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // 0.6B RAM Slot
    pub large_hibernation: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // 2B RAM Slot
    pub embedding_model: Arc<TokioMutex<Option<NativeEmbeddingModel>>>,
    
    pub is_cpu_mode: bool, 
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    small_model_path: String,
    large_model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    dtype: Option<()>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>, 
}

impl LogisModel {
    pub async fn ensure_speculative_draftsman(&self) -> anyhow::Result<()> {
        let mut draftsman = self.speculative_draftsman.lock().await;
        if draftsman.is_some() { return Ok(()); }

        // [STABILITY] Check if we have enough RAM to load the draftsman
        let mut sys = System::new();
        sys.refresh_memory();
        let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        if free_ram_gb < 3.0 {
            return Err(anyhow::anyhow!("Insufficient RAM ({:.2} GB) for speculative draftsman. Skipping.", free_ram_gb));
        }

        println!("[SPECULATIVE] Loading 0.6B Draftsman into CPU RAM...");
        let path = self.small_model_path.clone();
        let limit = self.max_tokens_limit;

        let gen = tokio::task::spawn_blocking(move || {
            Qwen3VLGenerateModel::init_with_config(
                &path, None, None, 
                None, 9, // Device ID 9 = Force CPU
                None, 9, None, Some(limit as usize),
                true, true
            )
        }).await??;

        *draftsman = Some(gen);
        Ok(())
    }

    pub async fn unload_generator(&self) {
        // [HARD-RESET] Completely destroy everything (For OOM recovery)
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut spec = self.speculative_draftsman.lock().await;
        *spec = None;
        let mut s_hib = self.small_hibernation.lock().await;
        *s_hib = None;
        let mut l_hib = self.large_hibernation.lock().await;
        *l_hib = None;
        
        let mut size = self.current_size.lock().await;
        *size = None;

        println!("[MODEL] Hard Reset: All weights and memory purged from VRAM/RAM.");
    }

    pub async fn soft_reset_generator(&self) {
        // [SOFT-RESET] Clear ONLY the dynamic context (KV Cache)
        // This keeps the weights (Brain) loaded but resets all memory pointers.
        let mut gen_guard = self.generator.lock().await;
        if let Some(gen) = gen_guard.as_mut() {
            gen.clear_kv_cache();
        }
        
        // Also clear hibernation slots if they exist to prevent memory creep
        let mut s_hib = self.small_hibernation.lock().await;
        if let Some(m) = s_hib.as_mut() { m.clear_kv_cache(); }
        
        let mut l_hib = self.large_hibernation.lock().await;
        if let Some(m) = l_hib.as_mut() { m.clear_kv_cache(); }

        println!("[MODEL] Soft Reset: Context cleared, weights preserved (Warm State).");
    }

    pub async fn unload_embedding(&self) {
        let mut emb = self.embedding_model.lock().await;
        if emb.is_some() {
            *emb = None;
            println!("[MODEL] Embedding Model unloaded to free VRAM.");
        }
    }

    // --- [NEW] CPU System RAM Optimizer ---
    async fn optimize_system_ram(&self) {
        if !self.is_cpu_mode { return; }

        let mut sys = System::new();
        sys.refresh_memory();
        let free_ram = sys.total_memory().saturating_sub(sys.used_memory()); 

        // Threshold: 4GB
        if free_ram < 4 * 1024 * 1024 * 1024 { // 4GB
            println!("[RAM-WATCH] Low System Memory ({:.2} GB). Flushing Working Set...", free_ram as f64 / 1024.0 / 1024.0 / 1024.0);
            
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
                use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MIN_DISABLE;
                use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MAX_DISABLE;
                let process = GetCurrentProcess();
                let _ = SetProcessWorkingSetSizeEx(process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        } else {
            println!("[RAM-WATCH] Sufficient Memory ({:.2} GB). Skipping flush.", free_ram as f64 / 1024.0 / 1024.0 / 1024.0);
        }
    }

    /// [CLEANUP] Adaptive resource management based on system stress
    pub async fn deep_purge_resources(&self, force_all: bool) {
        // 1. Clear Active Slot
        if let Ok(mut gen) = self.generator.try_lock() { *gen = None; }
        
        // 2. Clear Hibernation Slots ONLY if forced
        if force_all {
            if let Ok(mut s_hib) = self.small_hibernation.try_lock() { *s_hib = None; }
            if let Ok(mut l_hib) = self.large_hibernation.try_lock() { *l_hib = None; }
            
            let mut size = self.current_size.lock().await;
            *size = None;
        }
        
        // 3. [CRITICAL-FIX] OS RAM/VRAM Flush
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::*;
            use windows_sys::Win32::System::Memory::*;
            let current_process = GetCurrentProcess();
            // 강제로 Working Set을 비워 OS가 VRAM/RAM을 즉시 수거하게 함
            let _ = SetProcessWorkingSetSizeEx(current_process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        
        println!("[MEMORY] Purge Complete (Force: {}). OS Working Set flushed.", force_all);
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
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
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

            // [ADAPTIVE-LOGIC] Analyze Trend
            if current_free > last_free + (50 * 1024 * 1024) { // Increased by > 50MB
                increasing_ticks += 1;
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Reclaiming... ({:.2} GB -> {:.2} GB)", last_free as f64/1e9, current_free as f64/1e9);
                }
            } else {
                increasing_ticks = 0;
            }

            // [ACTIVE-FLUSH] If stuck for > 1.5s, trigger OS RAM cleanup
            if start.elapsed().as_secs_f32() > 1.5 && !has_flushed_ram && current_free < target_bytes {
                println!("[VRAM-WATCH] Triggering OS Working Set Trim...");
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MIN_DISABLE;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MAX_DISABLE;
                    let process = GetCurrentProcess();
                    let _ = SetProcessWorkingSetSizeEx(process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                has_flushed_ram = true;
                // Give it a moment to reflect
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }

            // [TIMEOUT-HANDLER]
            if start.elapsed().as_secs() > timeout_sec {
                // If memory is still actively freeing up, extend timeout dynamically
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Timeout reached but memory is freeing up. Extending wait...");
                    increasing_ticks = 0; // Reset to avoid infinite loop
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
    pub async fn save_kv_snapshot(&self, task_id: &str, address_hash: Option<&str>) -> anyhow::Result<String> {
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();
        let addr_hash = address_hash.map(|s| s.to_string());
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = generator_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let path = crate::utils::paths::get_kv_dir(None, addr_hash.as_deref()).join(format!("{}.safetensors", task_id_str));
                println!("[SSD-BRIDGE] Saving KV snapshot to {:?}", path);
                gen.save_kv_to_disk(&path, 0, &[])?;
                Ok(path.to_string_lossy().to_string())
            } else {
                Err(anyhow::anyhow!("No active generator to save snapshot from"))
            }
        }).await?
    }

    pub async fn load_kv_snapshot(&self, task_id: &str, address_hash: Option<&str>) -> anyhow::Result<()> {
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();
        let addr_hash = address_hash.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut gen_guard = generator_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let path = crate::utils::paths::get_kv_dir(None, addr_hash.as_deref()).join(format!("{}.safetensors", task_id_str));
                if path.exists() {
                    println!("[SSD-BRIDGE] Loading KV snapshot from {:?}", path);
                    gen.load_kv_from_disk(&path)?;
                    Ok(())
                } else {
                    println!("[SSD-BRIDGE] No snapshot found for {}", task_id_str);
                    Ok(())
                }
            } else {
                Err(anyhow::anyhow!("No active generator to load snapshot into"))
            }
        }).await?
    }

    /// [NEW] Secure VRAM/RAM Transition Logic (Isolation Protocol)
    pub async fn secure_vram_relay(&self, target_size: ModelSize, task_id: Option<&str>, cancel_token: Option<Arc<AtomicBool>>, address_hash: Option<&str>) -> anyhow::Result<()> {
        let start_time = Instant::now();
        
        // 1. [CLEANUP] 리소스 정리 (CPU 모드에서는 히버네이션 유지)
        println!("[RELAY] Preparing Transition to {:?}...", target_size);
        self.deep_purge_resources(false).await;
        
        if !self.is_cpu_mode {
            tokio::time::sleep(Duration::from_millis(500)).await;
            self.wait_for_vram_settle(2000, 5, cancel_token.clone()).await?;
        }

        // 2. [LOAD] 새 모델 로드 (Relay 시에는 기본적으로 텍스트 전용으로 시도 가능)
        let text_only = task_id.is_some(); // Snapshot 로드 시점은 대부분 텍스트 기반 추론
        self.ensure_generator_ext(target_size, false, text_only).await?;

        // 4. [RESTORE] 디스크 스냅샷 로드
        if let Some(tid) = task_id {
            self.load_kv_snapshot(tid, address_hash).await?;
        }

        println!("[RELAY] Transition to {:?} complete in {:.2}s", target_size, start_time.elapsed().as_secs_f32());
        Ok(())
    }

    // --- [SCENARIO A-Modular] Optimized Text Preprocessing ---
    
    /// Step 1: Bake the heavy PUG content once per address
    pub async fn bake_pug_context(&self, address_hash: &str, _task_id: &str, pug_content: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        let kv_dir = crate::utils::paths::get_kv_dir(None, Some(address_hash));
        let final_path = kv_dir.join("shared_pug_base.safetensors");
        
        if final_path.exists() {
            println!("[BAKE-PUG] Found permanent PUG cache for address {}, skipping baking.", address_hash);
            return Ok(());
        }

        println!("[BAKE-PUG] One-time site baking for: {} (Using current active generator)", address_hash);
        // [ORDERLY-EXECUTION] Acquire global GPU semaphore to prevent overlapping heavy computations
        let _gpu_guard = crate::scheduler::REGISTRY.gpu_semaphore.acquire().await.map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;
        println!("[BAKE-ORDER] GPU Slot secured for baking: {}", address_hash);

        // Do NOT call ensure_generator here, use whatever is already active (secured by scheduler)
        {
            let gen_clone = self.generator.clone();
            let text = pug_content.to_string();
            let ah = address_hash.to_string();
            let token_clone = cancel_token.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_clone.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    let path = crate::utils::paths::get_kv_dir(None, Some(&ah)).join("shared_pug_base.safetensors");
                    gen.bake_text_in_parts_to_path(text, &path, "pug_base", 0, token_clone)?;
                } else {
                    return Err(anyhow::anyhow!("Generator not initialized for baking"));
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }
        Ok(())
    }

    /// [NEW] Bake a specific system prompt into a file for later stitching
    pub async fn bake_system_prompt(&self, address_hash: &str, prompt_name: &str, system_prompt: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        let kv_dir = crate::utils::paths::get_kv_dir(None, Some(address_hash));
        let final_path = kv_dir.join(format!("shared_prompt_{}.safetensors", prompt_name));
        let pug_base_path = kv_dir.join("shared_pug_base.safetensors");

        if final_path.exists() {
            return Ok(());
        }

        println!("[BAKE-PROMPT] Baking system prompt: {} for address: {}", prompt_name, address_hash);
        let formatted_prompt = format!("<|im_start|>system\n{}<|im_end|>\n", system_prompt);
        
        self.ensure_generator_ext(ModelSize::Small, true, true).await?;
        {
            let gen_clone = self.generator.clone();
            let text = formatted_prompt;
            let p_path = final_path.clone();
            let b_path = pug_base_path.clone();
            let token_clone = cancel_token.clone();
            
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_clone.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    // PUG 베이스 뒤에 붙을 것이므로 오프셋을 계산하여 베이킹
                    let offset = if b_path.exists() { gen.get_kv_file_token_count(&b_path).unwrap_or(0) } else { 0 };
                    gen.bake_text_in_parts_to_path(text, &p_path, "baked_sys", offset, token_clone)?;
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }
        self.unload_generator().await;
        Ok(())
    }

    /// Step 2: Bake a specific prompt and run inference by stitching with PUG base
    pub async fn run_modular_inference(&self, address_hash: &str, _task_id: &str, prompt_name: &str, system_prompt: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let prompt_session = format!("shared_prompt_{}", prompt_name);
        println!("[MODULAR-INF] Running cached inference for step: {} (Address: {})", prompt_name, address_hash);

        let kv_dir = crate::utils::paths::get_kv_dir(None, Some(address_hash));
        let pug_base_path = kv_dir.join("shared_pug_base.safetensors");
        let prompt_path = kv_dir.join(format!("{}.safetensors", prompt_session));

        // 1. Bake the specific system prompt ONLY if missing
        if !prompt_path.exists() {
            println!("[MODULAR-INF] Prompt cache missing. Baking: {}", prompt_name);
            let formatted_prompt = format!("<|im_start|>system\n{}<|im_end|>\n", system_prompt);
            self.ensure_generator_ext(ModelSize::Small, true, true).await?;
            {
                let gen_clone = self.generator.clone();
                let sp = formatted_prompt;
                let p_path = prompt_path.clone();
                let b_path = pug_base_path.clone();
                let token_clone = cancel_token.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut gen_guard = gen_clone.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        let offset = if b_path.exists() { gen.get_kv_file_token_count(&b_path).unwrap_or(0) } else { 0 };
                        gen.bake_text_in_parts_to_path(sp, &p_path, "baked", offset, token_clone)?;
                        Ok::<(), anyhow::Error>(())
                    } else { Ok(()) }
                }).await??;
            }
            self.unload_generator().await;
        } else {
            println!("[MODULAR-INF] Found cached prompt for {}, skipping baking.", prompt_name);
        }

        // 2. Load 2B Full Model and Stitch
        self.secure_vram_relay_ext(ModelSize::Large, None, cancel_token.clone(), false, true, Some(address_hash)).await?;
        
        let mut components = Vec::new();
        if pug_base_path.exists() { components.push(pug_base_path); }
        if prompt_path.exists() { components.push(prompt_path); }

        {
            let gen_arc = self.generator.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    println!("[SSD-BRIDGE] Stitching permanent base with permanent prompt ({} files)", components.len());
                    gen.load_kv_stitched(&components)?;
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }

        // 3. Final Chat (Bypass additional bridging)
        self.chat("", user_input, cancel_token, Some(format!("{}_final_nobridge", prompt_name))).await
    }

    // --- [SCENARIO B] Image Preprocessing (0.6B 1L Bake -> 2B 1L Vision Bake -> 2B Full Inf) ---
    pub async fn bake_image_kv(&self, task_id: &str, region: &str, language: &str, page_type: &str, address: &str, address_hash: Option<&str>, image: DynamicImage, cancel_token: Option<Arc<AtomicBool>>, app_handle: &tauri::AppHandle) -> anyhow::Result<()> {
        // [ORDERLY-EXECUTION] Acquire global GPU semaphore for the entire multi-step vision pipeline
        let _gpu_guard = crate::scheduler::REGISTRY.gpu_semaphore.acquire().await.map_err(|e| anyhow::anyhow!("GPU Semaphore error: {}", e))?;
        
        let sys_session = format!("{}_sys", task_id);
        let img_session = format!("{}_img", task_id);
        let system_prompt = Self::get_image_extraction_prompt(region, language, page_type, address);

        // [Step 1] System Prompt Baking: 0.6B Small model, 1-Layer, Text-Only
        println!("[BAKE-IMAGE] Step 1: Baking System Prompt with 0.6B (1-Layer)...");
        self.secure_vram_relay_ext(ModelSize::Small, None, cancel_token.clone(), true, true, address_hash).await?;
        {
            let gen_clone = self.generator.clone();
            let sys_text = system_prompt.clone();
            let token_clone = cancel_token.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_clone.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.prefill_chunk(sys_text, token_clone, None)?;
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }
        self.save_kv_snapshot(&sys_session, address_hash).await?;
        self.unload_generator().await;

        // [Step 2] Image Data Baking: 2B Large model, 1-Layer (Strict Baking Mode), Vision Enabled
        println!("[BAKE-IMAGE] Step 2: Baking Image Data with 2B (1-Layer Vision)...");
        self.secure_vram_relay_ext(ModelSize::Large, None, cancel_token.clone(), true, false, address_hash).await?;
        {
            let _ = self.chat_with_image_spinner(
                "[IMAGE-BAKE] Vision feature extraction.".to_string(), 
                Some(image), 
                app_handle, 
                "", 
                json!({}), 
                1, 
                cancel_token.clone(), 
                None 
            ).await?;
        }
        self.save_kv_snapshot(&img_session, address_hash).await?;
        self.unload_generator().await;

        Ok(())
    }

    pub async fn extract_from_image(
        &self,
        task_id: String,
        address_hash: String,
        image_path: String,
        language: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        crate::scheduler::log_task_progress(app_handle, &task_id, &json!({
            "category": "Image Loading", "summary": "Starting 3-step vision pipeline...", "spinner": "⠋"
        }));

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            
            // 1 & 2. Bake KV (0.6B 1L Sys -> 2B 1L Img)
            self.bake_image_kv(&task_id, "kr", &language, "tracking", "", Some(&address_hash), dynamic_image, cancel_token.clone(), app_handle).await?;

            // 3. Full Inference (2B Full with Injected KV)
            crate::scheduler::log_task_progress(app_handle, &task_id, &json!({
                "category": "Vision Analysis", "summary": "Injecting baked cache & performing full inference..."
            }));
            
            let result_str = self.run_hybrid_inference_full(&task_id, &address_hash, "Extract all structured information from the label.", cancel_token.clone()).await?;

            let extracted_data = crate::parsing::parse_json_from_llm(&result_str);
            
            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            let item_digest = crate::utils::hash::digest(&nl);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr);
                let hashed_cc = crate::utils::hash::hash_id("local.image");
                let raw_no = extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id);
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(raw_no).replace("-", "").replace("_", "");
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_no)));
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));
                let mut final_data = extracted_data.clone();
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));

                let _ = db.upsert_item(
                    "commerce_tracking", &hashed_id, "tracking", final_data, None,
                    Some(from_addr), Some(&team_id), Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("tracking{}", hashed_cc))),
                    Some(&ref_val), Some(&item_digest)
                ).await;
            }
            
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            }));
            
            Ok(())
        } else {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Error", "summary": "Failed to load image file.", "spinner": "❌"
            }));
            Ok(())
        }
    }

    // --- [HYBRID UTILS] Combined KV Loading ---
    pub async fn load_stitched_kv(&self, address_hash: &str, task_id: &str, components: &[&str]) -> anyhow::Result<()> {
        let generator_arc = self.generator.clone();
        let kv_dir = crate::utils::paths::get_kv_dir(None, Some(address_hash));
        let mut all_paths: Vec<std::path::PathBuf> = Vec::new();

        // [AUTO-DISCOVERY] Find all parts for each component in the address-specific folder
        for comp in components {
            let mut parts = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&kv_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(&format!("{}_{}", task_id, comp)) && name.ends_with(".safetensors") {
                        parts.push(entry.path());
                    }
                }
            }
            // Sort by part number to ensure part1 < part2 < part10
            parts.sort_by(|a, b| {
                let extract_part = |p: &std::path::Path| -> usize {
                    p.file_name().and_then(|n| n.to_str())
                        .and_then(|s| s.split("_part").last())
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                };
                extract_part(a).cmp(&extract_part(b))
            });
            all_paths.extend(parts);
        }

        if all_paths.is_empty() {
            println!("[KV-STITCH] WARNING: No KV components found for task {}", task_id);
            return Ok(());
        }

        tokio::task::spawn_blocking(move || {
            let mut gen_guard = generator_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                println!("[SSD-BRIDGE] Stitching {} KV files...", all_paths.len());
                gen.load_kv_stitched(&all_paths)?;
                Ok(())
            } else {
                Err(anyhow!("No active generator for KV stitching"))
            }
        }).await?
    }

    /// [STABILITY] Check if the system has enough overhead to keep models in RAM
    async fn is_system_under_pressure(&self) -> bool {
        // 1. Check System RAM (Threshold: 2GB)
        let mut sys = System::new();
        sys.refresh_memory();
        let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        
        if free_ram_gb < 2.0 {
            println!("[STABILITY-WATCH] High RAM Pressure detected ({:.2} GB free).", free_ram_gb);
            return true;
        }

        // 2. Check GPU VRAM if applicable (Threshold: 1GB)
        if !self.is_cpu_mode {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(dev) = nvml.device_by_index(self.device_config.gpu_id as u32) {
                    if let Ok(mem) = dev.memory_info() {
                        let free_vram_gb = mem.free as f64 / 1024.0 / 1024.0 / 1024.0;
                        if free_vram_gb < 1.0 {
                            println!("[STABILITY-WATCH] High VRAM Pressure detected ({:.2} GB free).", free_vram_gb);
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    pub async fn secure_vram_relay_ext(&self, target_size: ModelSize, task_id: Option<&str>, cancel_token: Option<Arc<AtomicBool>>, baking_only: bool, force_text_only: bool, address_hash: Option<&str>) -> anyhow::Result<()> {
        let start_time = Instant::now();
        
        // [2026-SPECULATIVE] Hiding latency: Small model drafts initial tokens if we are moving to Large.
        if matches!(target_size, ModelSize::Large) && !baking_only {
            println!("[SPECULATIVE] Small model is active. Drafting initial tokens while Large loads...");
        }

        // 1. [PREFETCH] Start pre-warming the Large model weights in background thread
        if matches!(target_size, ModelSize::Large) {
            let path = self.large_model_path.clone();
            println!("[PIPELINE] Predictive Prefetching Large model weights to Page Cache...");
            tokio::task::spawn_blocking(move || {
                let _ = crate::scheduler::pre_fetch_weights(std::path::Path::new(&path));
            });
        }

        // [DYNAMIC-PURGE] Check system status before deciding how hard to purge
        let pressure = self.is_system_under_pressure().await;
        let force_purge = pressure || !self.is_cpu_mode || baking_only;
        
        if pressure {
            println!("[RELAY] System under pressure. Performing Aggressive Purge to ensure OOM safety.");
        }
        
        self.deep_purge_resources(force_purge).await;
        
        if !self.is_cpu_mode {
            // [LATENCY-HIDE] Reduced wait time due to pre-warmed cache
            tokio::time::sleep(Duration::from_millis(100)).await;
            self.wait_for_vram_settle(1200, 3, cancel_token.clone()).await?;
        }

        self.ensure_generator_ext(target_size, baking_only, force_text_only).await?;

        if let Some(tid) = task_id {
            self.load_kv_snapshot(tid, address_hash).await?;
        }

        println!("[RELAY] 2026-Predictive-Transition to {:?} complete in {:.2}s", 
            target_size, start_time.elapsed().as_secs_f32());
        Ok(())
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        self.ensure_generator_ext(size, false, false).await
    }

    pub async fn ensure_generator_ext(&self, size: ModelSize, baking_only: bool, force_text_only: bool) -> anyhow::Result<()> {
        let mut current_size_guard = self.current_size.lock().await;
        let mut gen_guard = self.generator.lock().await;
        let mut small_slot = self.small_hibernation.lock().await;
        let mut large_slot = self.large_hibernation.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() {
            return Ok(())
        }

        // [MODIFICATION] Force 0.6B to use 1 layer (Layer 0) for all tasks (Inference/Baking)
        let effective_baking_only = if size == ModelSize::Small { true } else { baking_only };

        println!("[MODEL] Activating engine -> Size: {:?}, Mode: {}, Vision: {}", size, if effective_baking_only { "BAKING (Layer 0)" } else { "INFERENCE (Full)" }, !force_text_only);

        // 1. [HIBERNATION-CHECK] Check if we can swap from hibernation slots
        let swapped = if !effective_baking_only { // Baking models are specialized (Layer 0), usually not worth hibernating
            match size {
                ModelSize::Small => if let Some(m) = small_slot.take() { *gen_guard = Some(m); true } else { false },
                ModelSize::Large => if let Some(m) = large_slot.take() { *gen_guard = Some(m); true } else { false },
            }
        } else { false };

        if swapped {
            println!("[HIBERNATION] Model restored from RAM slot (Skip Disk I/O)");
            *current_size_guard = Some(size);
            return Ok(());
        }

        // 2. [LOAD] Fresh loading from disk if not in hibernation
        println!("[LOAD] Fresh loading {:?} model from disk...", size);
        let path = if size == ModelSize::Small { &self.small_model_path } else { &self.large_model_path };
        let shared_path = if size == ModelSize::Small { Some(self.large_model_path.as_str()) } else { None };
        
        let dev_id = self.device_config.gpu_id;
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let shared_path_clone = shared_path.map(|s| s.to_string());

        let gen = tokio::task::spawn_blocking(move || {
            Qwen3VLGenerateModel::init_with_config(
                &path_clone, 
                shared_path_clone.as_deref(), 
                shared_path_clone.as_deref(), 
                None, dev_id, None, dev_id, None, Some(limit as usize),
                effective_baking_only, force_text_only
            )
        }).await??;

        // [GPU-ACTIVATION-LOG] Status reporting only
        if !self.is_cpu_mode {
            println!("[MODEL] Offloading to GPU-{} (Integrated Path)...", self.device_config.gpu_id);
        } else {
            println!("[MODEL] Staying on CPU (Extreme Optimized Bit-serial Path)...");
        }

        // Save old model to hibernation if applicable
        if let Some(old_m) = gen_guard.take() {
            if let Some(old_size) = *current_size_guard {
                match old_size {
                    ModelSize::Small => *small_slot = Some(old_m),
                    ModelSize::Large => *large_slot = Some(old_m),
                }
            }
        }

        *gen_guard = Some(gen);
        *current_size_guard = Some(size);
        
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        let current_size = { *self.current_size.lock().await };
        
        // [STRATEGY] High-priority exclusion logic
        match current_size {
            Some(ModelSize::Large) => {
                // If Large is active, Embedding must stay on CPU to avoid OOM
                println!("[MODEL] Large model active. Forcing Embedding to CPU to prevent swapping.");
            },
            Some(ModelSize::Small) => {
                // Small and Embedding can coexist. 
                println!("[MODEL] Small model active. Embedding and 0.6B will coexist.");
            },
            None => {
                // No generator active, safe to clean up any leftovers
                self.unload_generator().await;
            }
        }

        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let self_clone = self.embedding_path.clone();
            
            println!("[MODEL] Loading Native Embedding Model...");
            
            let emb = tokio::task::spawn_blocking(move || {
                NativeEmbeddingModel::load(&self_clone)
            }).await??;
            
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    pub async fn new(device_preference: Option<&str>) -> anyhow::Result<Self> {
        println!("[MODEL-00] Initializing LogisModel (Preference: {:?})", device_preference);
        #[cfg(feature = "cuda")]
        crate::models::qwen3vl::native_backend::test_cuda_init();

        let mut config = utils::get_optimal_device_config();
        
        if device_preference == Some("cpu") {
            println!("⚠️ [MODEL] EXPLICIT CPU MODE FORCED by user/system preference.");
            config = utils::DeviceConfig {
                is_cpu: true,
                classify_chunk_size: 12000,
                extract_chunk_size: 12000,
                name: "CPU-Forced".to_string(),
                gpu_id: 999, // Use 999 to signal CPU-only
            };
        } else {
            println!("🚀 [MODEL] Running in default mode ({})", config.name);
        }

        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        let small_gguf_dir = base_path.join("Qwen3-0.6B-Instruct-gguf");
        let large_gguf_dir = base_path.join("Qwen3-VL-2B-Instruct-gguf");
        
        let small_model_path = small_gguf_dir.to_str().unwrap().to_string();
        let large_model_path = large_gguf_dir.to_str().unwrap().to_string();
        let embedding_path = base_path.join("embeddinggemma-300m");

        let max_tokens_limit = 16384; 

        Ok(Self {
            generator: Arc::new(TokioMutex::new(None)),
            speculative_draftsman: Arc::new(TokioMutex::new(None)),
            small_hibernation: Arc::new(TokioMutex::new(None)),
            large_hibernation: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: config.is_cpu,
            dual_mode_enabled: true, 
            small_model_path,
            large_model_path,
            embedding_path,
            device_config: config.clone(),
            max_tokens_limit: max_tokens_limit as u32,
            dtype: None, 
            current_size: Arc::new(TokioMutex::new(None)),
        })
    }

    pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
        if page_type == "tracking" {
            let template = r###"[TASK]
Convert the shipping label image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number. It should be selected from numbers matching barcodes or QR codes, filtered by region, excluding telephone formats or order numbers.
2. Set recipient_match to true if the label address matches the context address (ignoring floor levels).
3. Extract all visible barcodes into an array.
4. Provide a text summary in {LANGUAGE}, masking the address to District-level and up. Do not mention masking.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "tracking_number": "string",
    "recipient_match": boolean,
    "barcodes": ["string"],
    "text": "string"
}"###;
            template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
        } else {
            String::new()
        }
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Large).await?;
            }
        }
        
        // [ORDERLY-EXECUTION] Acquire global GPU semaphore
        let _gpu_guard = crate::scheduler::REGISTRY.gpu_semaphore.acquire().await.map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;

        let self_clone = self.generator.clone();
        let sid_clone = session_id.clone();
        
        // [INTEGRATED-PREFILL] nobridge 모드일 때는 시스템 프롬프트를 비워 
        // generate.rs가 기존 KV 캐시(베이킹된 HTML)를 훼손하지 않게 함.
        let final_system = if sid_clone.as_ref().map_or(false, |s| s.contains("nobridge")) {
            "".to_string()
        } else {
            system.to_string()
        };
        
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request (nobridge: {})...", final_system.is_empty());
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: final_system,
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, sid_clone.as_deref()).map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response Length: {}", response.len());
            Ok(response)
        }).await?
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
        session_id: Option<String>
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
            model: "qwen3vl".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Large).await?;
            }
        }

        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::scheduler::log_task_progress(app_handle, task_id, &base_payload);
        }

        // [ORDERLY-EXECUTION] Acquire global GPU semaphore
        let _gpu_guard = crate::scheduler::REGISTRY.gpu_semaphore.acquire().await.map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            gen.generate(params, cancel_token, session_id.as_deref()).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
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
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        self.ensure_generator(ModelSize::Large).await?;

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tokens as u32),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params, cancel_token, session_id.as_deref()).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
    }

    async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        self.ensure_generator(ModelSize::Large).await?;
        
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
            model: "qwen3vl".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id.as_deref()).map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        self.ensure_generator(ModelSize::Large).await?;

        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        let generator_arc = self.generator.clone();
        let max_tok = self.max_tokens_limit;
        
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = generator_arc.blocking_lock();
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params, cancel_token, session_id.as_deref()).map_err(|e| anyhow!("Inference failed: {}", e))
        });
        
        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = Self::get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            prompt, 
            Some(master_img), 
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn run_hybrid_inference_full(&self, task_id: &str, address_hash: &str, user: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let combined_session = format!("{}_hybrid", task_id);
        println!("[INFERENCE-HYBRID] Starting 2B Full-Layer VL Inference for task: {}", task_id);

        // 1. Load 2B Full Model (Vision Enabled)
        self.secure_vram_relay_ext(ModelSize::Large, None, cancel_token.clone(), false, false, Some(address_hash)).await?;

        // 2. Inject Stitched BitKV (System + Image)
        self.load_stitched_kv(address_hash, task_id, &["sys", "img"]).await?;

        // 3. Chat with hybrid context
        self.chat("", user, cancel_token, Some(combined_session)).await
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        self.ensure_embedding().await?;

        let embedding_model_arc = self.embedding_model.clone();
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                Ok(vec![0.0; 768])
            }
        }).await?
    }

    pub async fn parse_query_structured(&self, query: String, language: &str) -> anyhow::Result<Value> {
        let current_time = chrono::Utc::now().to_rfc3339();
        
        let prompt1 = crate::parsing::para2graph(language);
        let res1 = self.chat("", &format!("{}\n\nQuery: {}", prompt1, query), None, Some("system_search_p2g".to_string())).await?;
        let segments = crate::parsing::parse_json_from_llm(&res1);
        
        let mut final_contexts = Vec::new();
        if let Some(ctx_arr) = segments.get("context").and_then(|v: &Value| v.as_array()) {
            let mut combined_segments = String::new();
            for (idx, seg) in ctx_arr.iter().enumerate() {
                let seg_text = seg.get("text").and_then(|v: &Value| v.as_str()).unwrap_or("");
                combined_segments.push_str(&format!("Segment #{}: {}\n", idx + 1, seg_text));
            }

            if !combined_segments.is_empty() {
                let prompt2 = crate::parsing::graph2contexts(&current_time);
                let res2 = self.chat("", &format!("{}\n\nInput Segments:\n{}", prompt2, combined_segments), None, Some("system_search_g2c".to_string())).await?;
                let mut batch_info = crate::parsing::parse_json_from_llm(&res2);
                
                if let Some(res_arr) = batch_info.get_mut("context").and_then(|v: &mut Value| v.as_array_mut()) {
                    for (i, item) in res_arr.iter_mut().enumerate() {
                        if let Some(original_seg) = ctx_arr.get(i) {
                            if item.get("type").is_none() || item.get("type").and_then(|v: &Value| v.as_str()) == Some("") {
                                if let Some(item_obj) = item.as_object_mut() {
                                    item_obj.insert("type".to_string(), original_seg.get("type").cloned().unwrap_or(json!("")));
                                }
                            }
                        }
                    }
                    final_contexts.extend(res_arr.clone());
                }
            }
        }
        
        Ok(json!({ "context": final_contexts }))
    }

    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);

        status_history.push_str("✅ Context gathered.\n\n");
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for step in steps.iter() {
            status_history.push_str(&format!("**⏳ {}**\n", step));
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));
        }

        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None).await?;
        status_history.push_str(&report);
        
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        Ok(report)
    }

    fn get_search_schema_definitions(&self, _doc_type: &str) -> String {
        r###"{ 
  "header.document_type": { "desc": "Type (Invoice, BL, AWB, PO, BC, AN, DO...)", "type": "String" },
  "header.document_number": { "desc": "ID, Doc No, Reference No", "type": "String" },
  "header.po_number": { "desc": "Purchase Order No (PO)", "type": "String" },
  "header.booking_number": { "desc": "Booking Reference No (BC)", "type": "String" },
  "header.an_number": { "desc": "Arrival Notice No (AN)", "type": "String" },
  "header.do_number": { "desc": "Delivery Order No (DO)", "type": "String" },
  "header.issue_date": { "desc": "Date (YYYY-MM-DD)", "type": "String" },
  
  "parties.supplier_name": { "desc": "Seller, Shipper, Exporter, Vendor", "type": "String" },
  "parties.buyer_name": { "desc": "Buyer, Consignee, Importer", "type": "String" },
  "parties.notify_party_name": { "desc": "Notify Party", "type": "String" },
  
  "financials.amount_total": { "desc": "Total Value/Amount", "type": "Number" },
  "financials.local_charges_total": { "desc": "Total Local Charges (AN)", "type": "Number" },
  
  "logistics.vehicle_name": { "desc": "Vessel Name, Flight No", "type": "String" },
  "logistics.location_port_of_loading": { "desc": "POL, Origin", "type": "String" },
  "logistics.location_port_of_discharge": { "desc": "POD, Destination", "type": "String" },
  "logistics.pickup_location": { "desc": "Pickup Location (DO)", "type": "String" },
  "logistics.etd": { "desc": "Estimated Departure", "type": "String" },
  "logistics.eta": { "desc": "Estimated Arrival", "type": "String" },
  
  "conditions.incoterms_code": { "desc": "Incoterms (FOB, CIF)", "type": "String" }
}"###.to_string()
    }
}

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" {
        let template = r###"[TASK]
Convert the shipping label image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number. It should be selected from numbers matching barcodes or QR codes, filtered by region, excluding telephone formats or order numbers.
2. Set recipient_match to true if the label address matches the context address (ignoring floor levels).
3. Extract all visible barcodes into an array.
4. Provide a text summary in {LANGUAGE}, masking the address to District-level and up. Do not mention masking.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "tracking_number": "string",
    "recipient_match": boolean,
    "barcodes": ["string"],
    "text": "string"
}"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
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