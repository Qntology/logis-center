use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::{LogisModel, ModelSize};
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::{AppHandle, Manager, Emitter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::PathBuf;
use tokio::sync::Notify;
use once_cell::sync::Lazy;

// --- [MEMORY OPTIMIZATION] Task Data Manager (RAII) ---
pub struct TaskDataManager {
    pub task_id: String,
    created_files: Vec<PathBuf>,
    app_handle: Option<tauri::AppHandle>,
}

impl TaskDataManager {
    pub fn new(task_id: &str, app_handle: Option<tauri::AppHandle>) -> Self {
        Self {
            task_id: task_id.to_string(),
            created_files: Vec::new(),
            app_handle,
        }
    }

    pub fn offload(&mut self, content: &str, suffix: &str) -> Result<PathBuf> {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        let _ = fs::create_dir_all(&dir);
        let filename = format!("{}.txt", suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    pub fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }

    pub fn get_path(&self, suffix: &str) -> PathBuf {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        dir.join(format!("{}.txt", suffix))
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        println!("[Cleanup] TaskDataManager dropping for task: {}", self.task_id);
    }
}

// [UI-SYNC] Instant notification system to wake up the worker
pub static UI_READY_SIGNAL: Notify = Notify::const_new();
pub static TASK_QUEUED_SIGNAL: Notify = Notify::const_new();
pub static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters();
    println!("[Scheduler] UI signaled ready.");
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}

pub fn log_task_progress(app_handle: &tauri::AppHandle, task_id: &str, payload: &Value) {
    let log_file = utils::paths::get_task_log_file(Some(app_handle), task_id);
    let _ = fs::create_dir_all(log_file.parent().unwrap());
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(&log_file) {
        use std::io::Write;
        let _ = writeln!(file, "{}", payload.to_string());
    }
    let _ = app_handle.emit("extraction-progress", payload);
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>, 
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    clear_all_temp_data(Some(&app_handle));

    // Cleanup zombie tasks on startup
    {
        let store_clone = store.clone();
        tokio::spawn(async move {
            for _ in 0..10 {
                if let Ok(guard) = store_clone.try_lock() {
                    if let Some(db) = guard.as_ref() {
                        let _ = db.cleanup_zombie_tasks().await;
                        break;
                    }
                }
                sleep(Duration::from_millis(500)).await;
            }
        });
    }

    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }

        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;

        loop {
            if crate::utils::is_extraction_stopped() {
                sleep(Duration::from_millis(500)).await;
                continue;
            }

            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    if let Ok(tasks) = db.get_pending_tasks(5).await {
                        pending_tasks = tasks;
                    }
                }
            }

            if pending_tasks.is_empty() {
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => { delay_secs = (delay_secs + 1).min(10); }
                    _ = TASK_QUEUED_SIGNAL.notified() => { delay_secs = 1; }
                }
                continue;
            } else {
                delay_secs = 1;
            }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) { break; } 
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, 1).await; // 1 = Progress
                    }
                }

                // Process Task
                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, 9).await; // 9 = Complete
                        }
                        cleanup_task_resources(&task.id, Some(&app_handle));
                        current_device_pref = None;
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        // Emergency Cleanup
                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.unload_generator().await;
                            }
                            *model_lock = None;
                        }
                        
                        if err_msg.contains("Task cancelled") {
                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, 3).await; // 3 = Cancel
                            }
                            cleanup_task_resources(&task.id, Some(&app_handle));
                            current_device_pref = None;
                            break;
                        } else {
                            println!("[Scheduler] Task failed: {}. Error: {}", task.id, err_msg);
                            
                            if err_msg.contains("out of memory") || err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") {
                                println!("[Scheduler] OOM Detected! Forcing CPU mode for retry.");
                                current_device_pref = Some("cpu".to_string());
                                log_task_progress(&app_handle, &task.id, &json!({
                                    "category": "Warning", "summary": "Memory pressure detected. Retrying on CPU...", "spinner": "⚠️"
                                }));
                                sleep(Duration::from_secs(2)).await;
                                continue;
                            }

                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, 6).await; // 6 = Error
                                let _ = db.update_message_status(&task.id, 6, Some(&format!("Error: {}", err_msg))).await;
                            }
                            // [UI-FIX] Explicitly notify frontend of failure
                            log_task_progress(&app_handle, &task.id, &json!({
                                "category": "Error", "summary": format!("Error: {}", err_msg), "spinner": "❌"
                            }));
                        }
                    }
                }
            }
            cancellation_token.store(false, Ordering::SeqCst);
        }
    });
}

async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    // 1. Setup & Pre-check
    let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);
    let _ = fs::create_dir_all(&pug_logs_dir);
    
    let kv_dir = utils::paths::get_kv_dir(Some(app_handle), Some(&task.r#ref));
    if kv_dir.exists() { println!("[PROCESS] Found existing KV directory for address: {}", task.r#ref); }

    let large_model_path_hint = std::fs::canonicalize("src-tauri/models/Qwen3-VL-2B-Instruct-gguf")
        .or_else(|_| std::fs::canonicalize("models/Qwen3-VL-2B-Instruct-gguf")).ok();
    if let Some(p) = large_model_path_hint {
        let _ = std::thread::spawn(move || { let _ = pre_fetch_weights(&p); });
    }

    log_task_progress(app_handle, &task.id, &json!({ 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    }));

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));

    let effective_device_pref = device_preference.or_else(|| {
        task_data.get("device_preference").and_then(|v| {
            if v.as_str() == Some("cpu") || v.as_bool() == Some(true) { Some("cpu".to_string()) } else { None }
        })
    });

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();
    let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();

    if url.is_empty() && image_path.is_empty() { return Ok(()); }

    // 2. Fetch & Convert to PUG
    let light_pug = {
        let light_pug_path = data_manager.get_path("light_pug");
        if light_pug_path.exists() {
            data_manager.load(&light_pug_path)?
        } else {
            let raw_html_path = data_manager.get_path("raw_html");
            let raw_content = if raw_html_path.exists() {
                data_manager.load(&raw_html_path)?
            } else if let Some(h) = task_data.get("html").and_then(|s| s.as_str()) {
                 data_manager.offload(h, "raw_html")?;
                 h.to_string()
            } else {
                 let res = reqwest::get(&url).await?;
                 let bytes = res.bytes().await?;
                 let content = String::from_utf8_lossy(&bytes).to_string();
                 data_manager.offload(&content, "raw_html")?;
                 content
            };
            
            let clean = parsing::pre_clean_html(&raw_content);
            let p = parsing::convert_to_clean_pug(&clean, PugMode::FullContent);
            data_manager.offload(&p, "light_pug")?;
            p
        }
    };

    // [LOCK] Acquire Model Access
    let model = {
        let mut model_lock = model_mutex.lock().await;
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = effective_device_pref.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.unload_generator().await;
                *model_lock = None;
            }
        }

        if model_lock.is_none() {
            match LogisModel::new(effective_device_pref.as_deref()).await {
                Ok(m) => *model_lock = Some(m),
                Err(e) => return Err(anyhow::anyhow!("Model Load Failed: {}", e)),
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    // [PIPELINE-FORK] Separate Image and Text preprocessing flows
    if !image_path.is_empty() {
        // SCENARIO B: Image Preprocessing (0.6B Bake -> 2B Vision Bake -> 2B Full Inference)
        log_task_progress(app_handle, &task.id, &json!({ "category": "Image Pipeline", "summary": "Starting 3-step vision extraction...", "spinner": "⠋" }));
        let language = task_data.get("language").and_then(|s| s.as_str()).unwrap_or("korean").to_string();
        model.extract_from_image(task.id.clone(), task.r#ref.clone(), image_path, language, app_handle, Some(cancellation_token.clone()), store_mutex).await?;
        return Ok(());
    } else if !url.is_empty() {
        // SCENARIO A: Text Preprocessing (0.6B Bake -> 2B Full Inference)
        process_text_pipeline(task, &model, cancellation_token, app_handle, &light_pug, &mut data_manager, store_mutex).await?;
        return Ok(());
    }

    Ok(())
}

async fn process_text_pipeline(
    task: Task,
    model: &LogisModel,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    light_pug: &str,
    data_manager: &mut TaskDataManager,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
) -> Result<()> {
    let address_hash = &task.r#ref; // Use the URL hash from LanceDB
    
    // [MODULAR-PIPELINE] Step 0: One-time PUG Baking (Branching by address_hash)
    log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Baking PUG context (address-based)...", "spinner": "⠋" }));
    model.bake_pug_context(address_hash, &task.id, light_pug, Some(cancellation_token.clone())).await?;

    // --- PIPELINE STEP A: CLASSIFICATION ---
    let mut page_type = String::new();
    let mut selector_info = json!({});
    
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Determining page type...", "spinner": "⠋" }));

        let type_prompt = parsing::page_type_prompt();
        let task_question = format!("[TASK] {}\n\n[ACTION] RETURN JSON ONLY", type_prompt);
        // Reuse pre-baked PUG from address folder
        let res = model.run_modular_inference(address_hash, &task.id, "class", &task_question, "Classify this page.", Some(cancellation_token.clone())).await?;
        data_manager.offload(&res, "step_a_res")?;
        
        let type_info = parsing::parse_json_from_llm(&res);
        page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("unknown").to_string();
        println!("[Scheduler] Classified as: {}", page_type);
    }
    
    if page_type == "unknown" { return Ok(()); } 
    
    // --- PIPELINE STEP B: SELECTORS ---
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        log_task_progress(app_handle, &task.id, &json!({ "category": "Selector Search", "summary": "Identifying data elements...", "spinner": "⠋" }));

        let selector_prompt = parsing::page_selectors_prompt(&page_type);
        let task_question = format!("[TASK] {}\n\n[ACTION] RETURN JSON ONLY", selector_prompt);
        // Reuse pre-baked PUG
        let res = model.run_modular_inference(address_hash, &task.id, "sel", &task_question, "Identify selectors.", Some(cancellation_token.clone())).await?;
        data_manager.offload(&res, "step_b_res")?;
        selector_info = parsing::parse_json_from_llm(&res);
    }

    let is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut extracted_data = json!({});

    if !is_detail {
        let item_selector = selector_info.get("item").and_then(|s| s.as_str()).unwrap_or("");
        let mut all_extracted_items = Vec::new();
        {
            let clean_html_path = data_manager.get_path("clean_html");
            if !clean_html_path.exists() {
                 let raw = data_manager.load(&data_manager.get_path("raw_html"))?;
                 let c = parsing::pre_clean_html(&raw);
                 data_manager.offload(&c, "clean_html")?;
            }
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
             if let Ok(sel) = scraper::Selector::parse(item_selector) {
                for item in document.select(&sel) {
                    let text = item.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !text.is_empty() { all_extracted_items.push(json!({ "text": text })); }
                }
            }
        }
        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });
    } else {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Extracting details...", "spinner": "⠋" }));
        
        let node_selector = selector_info.get("node").and_then(|s| s.as_str()).unwrap_or("body");
        let content_pug = {
            let clean_html_path = data_manager.get_path("clean_html");
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            let mut pug_output = String::new();
            if let Ok(selector) = scraper::Selector::parse(node_selector) {
                if let Some(node) = document.select(&selector).next() {
                    parsing::generate_pug_lines(*node, 0, &mut pug_output, &PugMode::FullContent);
                }
            }
            parsing::sanitize_llm_input(&pug_output)
        };

        if !content_pug.trim().is_empty() {
             let extraction_instruction = parsing::item2json(&page_type, &task.data_json, "english");
             let task_question = format!("[TASK] {}\n\n[ACTION] RETURN JSON ONLY", extraction_instruction);
             // Reuse pre-baked PUG
             let res = model.run_modular_inference(address_hash, &task.id, "ext", &task_question, "Extract JSON data.", Some(cancellation_token.clone())).await?;
             data_manager.offload(&res, "step_c_res")?;
             extracted_data = parsing::parse_json_from_llm(&res);
        }
    }

    // --- PHASE 3: HANDOVER & EMBEDDING ---
    model.unload_generator().await;
    
    if let Some(obj) = extracted_data.as_object_mut() {
        if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type)); }
    }

    log_task_progress(app_handle, &task.id, &json!({ "category": "Saving", "summary": "Syncing to database..." }));

    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    let text_to_embed = parsing::json_to_natural_language(&extracted_data);
    let item_digest = crate::utils::hash::digest(&text_to_embed); 
    let vector = model.get_embedding(text_to_embed).await?;

    let generated_id = crate::utils::hash::hash_id(&format!("{}{}", task.to, task.id)); 
    let _ = store.upsert_item(
        &page_type, &generated_id, &page_type, extracted_data.clone(), Some(vector),
        Some(&task.from), Some(&task.to), Some(&task.cc), None, None, Some(&item_digest)
    ).await;

    log_task_progress(app_handle, &task.id, &json!({
        "category": "Done", "summary": "Extraction complete.", "spinner": "✅",
        "data": extracted_data
    }));

    Ok(())
}

fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
    let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
}

fn pre_fetch_weights(path: &std::path::Path) -> Result<()> {
    use std::io::Read;
    println!("[PRE-FETCH] Warming up OS Page Cache for weights in: {:?}", path);
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let p = entry.path();
                    if p.extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors") {
                        if let Ok(mut file) = std::fs::File::open(p) {
                            let mut buffer = [0u8; 1024 * 1024]; 
                            while let Ok(n) = file.read(&mut buffer) {
                                if n == 0 { break; }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("[PRE-FETCH] Warm-up complete.");
    Ok(())
}

fn clear_all_temp_data(app_handle: Option<&tauri::AppHandle>) {
    println!("[Cleanup] Clearing all temporary data directories...");
    utils::paths::cleanup_temp_dirs(app_handle);
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let mut last_report = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB)...", target_vram_mb);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;

        if let Some(ref nvml_inst) = nvml {
            if let Ok(count) = nvml_inst.device_count() {
                for i in 0..count {
                    if let Ok(dev) = nvml_inst.device_by_index(i) {
                        if let Ok(mem) = dev.memory_info() {
                            if mem.free > current_vram { current_vram = mem.free; }
                            has_gpu = true;
                        }
                    }
                }
            }
        }

        let meets_vram = !has_gpu || current_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            break; 
        }

        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 5_000_000 { 
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
        }

        if stable_ticks >= 6 && current_vram > 1_200_000_000 {
            println!("[RESOURCE-WATCH] Memory stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 5 {
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        if start_time.elapsed().as_secs() > 20 {
            println!("[RESOURCE-WATCH] Max wait reached. Proceeding anyway.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}

fn chunk_text(text: &str, target_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    while start < text.len() {
        let mut end = (start + target_size).min(text.len());
        if end < text.len() {
            let mut temp_end = end;
            while temp_end > start && bytes[temp_end] != b'\n' {
                temp_end -= 1;
            }
            if temp_end > start {
                end = temp_end + 1; 
            } else {
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
            }
        } else {
            end = text.len();
        }
        let slice = &text[start..end];
        if !slice.trim().is_empty() {
            chunks.push(slice.to_string());
        }
        start = end;
    }
    chunks
}

fn merge_json_results(target: &mut Value, source: &Value) {
    if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_obj {
            if v.is_null() { continue; }
            if let Some(s) = v.as_str() { if s.is_empty() { continue; } }
            if let Some(a) = v.as_array() { if a.is_empty() { continue; } }
            
            let should_update = match target_obj.get(k) {
                None => true,
                Some(tv) => {
                    tv.is_null() || 
                    (tv.is_string() && tv.as_str() == Some("")) ||
                    (tv.is_array() && tv.as_array().unwrap().is_empty())
                }
            };

            if should_update {
                target_obj.insert(k.clone(), v.clone());
            } else if let Some(target_inner) = target_obj.get_mut(k) {
                if target_inner.is_object() && v.is_object() {
                    merge_json_results(target_inner, v);
                }
                else if let (Some(ta), Some(sa)) = (target_inner.as_array_mut(), v.as_array()) {
                    ta.extend(sa.clone());
                }
            }
        }
    }
}
