use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::PathBuf;

// --- [MEMORY OPTIMIZATION] Task Data Manager (RAII) ---
// Handles offloading of large text data to disk to prevent RAM/VRAM bloating.
// Automatically cleans up files when the task scope ends.
struct TaskDataManager {
    task_id: String,
    created_files: Vec<PathBuf>,
    app_handle: Option<tauri::AppHandle>,
}

impl TaskDataManager {
    fn new(task_id: &str, app_handle: Option<tauri::AppHandle>) -> Self {
        Self {
            task_id: task_id.to_string(),
            created_files: Vec::new(),
            app_handle,
        }
    }

    fn offload(&mut self, content: &str, suffix: &str) -> Result<PathBuf> {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        
        // Simple unique filename
        let filename = format!("{}_{}_{}.txt", self.task_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_micros(), suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        println!("[Cleanup] TaskDataManager dropping. Keeping files for debugging: {}", self.task_id);
        for path in &self.created_files {
            println!("[DEBUG] Persisted file: {:?}", path);
            // if path.exists() {
            //     let _ = fs::remove_file(path);
            // }
        }
        // KV 캐시는 재사용을 위해 디스크에 유지합니다.
    }
}

// Helper to chunk text, strictly respecting Pug line boundaries (\n)
fn chunk_text(text: &str, target_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    while start < text.len() {
        let mut end = (start + target_size).min(text.len());

        // [BACKTRACKING] If mid-line, move back until we find a newline
        if end < text.len() {
            let mut temp_end = end;
            while temp_end > start && bytes[temp_end] != b'\n' {
                temp_end -= 1;
            }
            
            // If found a newline, use it as the end
            if temp_end > start {
                end = temp_end + 1; // Include the \n
            } else {
                // No newline found in the whole chunk? Force char boundary to avoid hang
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
            }
        } else {
            // Reached the end of string
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
// Deep merge for JSON objects
fn merge_json_results(target: &mut Value, source: &Value) {
    if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_obj {
            // If value is null or empty string/array, ignore
            if v.is_null() { continue; }
            if let Some(s) = v.as_str() { if s.is_empty() { continue; } }
            if let Some(a) = v.as_array() { if a.is_empty() { continue; } }
            
            // If target doesn't have it or target is empty, overwrite
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
                // If both are objects, recurse
                if target_inner.is_object() && v.is_object() {
                    merge_json_results(target_inner, v);
                }
                // Lists? Simply append for now (might duplicate, but safe for search)
                else if let (Some(ta), Some(sa)) = (target_inner.as_array_mut(), v.as_array()) {
                    ta.extend(sa.clone());
                }
            }
        }
    }
}

use tokio::sync::Notify;
use once_cell::sync::Lazy;

// [UI-SYNC] Instant notification system to wake up the worker
static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters(); // Wake up any sleeping tasks instantly
    println!("[Scheduler] UI signaled ready. Background worker woke up.");
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    clear_all_temp_data(Some(&app_handle));
    
    tokio::spawn(async move {
        // [EVENT-DRIVEN-WAIT] Zero CPU usage, zero delay. 
        // Wakes up exactly when mark_ui_ready is called.
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        
        loop {
            sleep(Duration::from_secs(delay_secs)).await;
            
            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    match db.get_pending_tasks(5).await {
                        Ok(tasks) => pending_tasks = tasks,
                        Err(e) => println!("[Scheduler] Failed to fetch tasks: {:?}", e),
                    }
                }
            }

            if pending_tasks.is_empty() {
                delay_secs = (delay_secs + 1).min(10); 
                continue;
            } else {
                delay_secs = 1;
            }

            for task in pending_tasks {
                // [FIX] Do NOT reset the cancellation token here. 
                // It should only be reset at the start of the entire loop iteration if needed,
                // or managed by the command that starts the extraction.
                
                if cancellation_token.load(Ordering::Relaxed) {
                    println!("[Scheduler] Cancellation detected before starting task {}, skipping batch.", task.id);
                    break;
                }

                println!("[Scheduler] Processing task: {}", task.id);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("progress")).await;
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                        }
                    },
                    Err(e) => {
                        let err_msg = e.to_string();

                        // [CRITICAL-CLEANUP] 작업 실패 시 즉시 모델을 메모리에서 해제하여 다음 작업 대비
                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.unload_generator().await;
                            }
                            *model_lock = None;
                            println!("[Scheduler] Error detected. Emergency memory release performed: {}", err_msg);
                        }

                        if err_msg.contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             let store_guard = store.lock().await;
                             if let Some(db) = store_guard.as_ref() {
                                 let _ = db.update_task_status(&task.id, crate::logic::parse_status("cancel")).await;
                                 let _ = db.update_message_status(&task.id, crate::logic::parse_status("cancel"), Some("Cancelled by user")).await;
                             }
                             let _ = app_handle.emit("extraction-progress", json!({ 
                                "task_id": task.id,
                                "category": "Done", "summary": "Cancelled by user", "spinner": "🛑", "data": null 
                             }));
                             break; 
                        } else {
                            println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                            
                            // [NEW] Automatic OOM Recovery Logic
                            if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                                println!("[Scheduler] OOM Detected! Force unloading model to protect system VRAM.");
                                let mut model_lock = model.lock().await;
                                if let Some(m) = model_lock.as_ref() {
                                    m.unload_generator().await;
                                }
                                *model_lock = None; // Explicitly drop model from GPU
                                
                                let _ = app_handle.emit("extraction-progress", json!({ 
                                    "task_id": task.id,
                                    "category": "Error", "summary": "GPU Memory Full (OOM). Task stopped.", "spinner": "❌"
                                }));
                            }

                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                        }
                    }
                }
            }
            
            // Reset token after batch is done or broken, for the next poll
            cancellation_token.store(false, Ordering::SeqCst);
        }
    });
}

async fn process_task(

    task: Task,

    store_mutex: &Arc<Mutex<Option<VectorStore>>>,

    model_mutex: &Arc<Mutex<Option<LogisModel>>>,

    cancellation_token: &Arc<AtomicBool>,

    app_handle: &tauri::AppHandle

)

-> Result<()>

{

            // [NEW] Ensure log directory exists at runtime using dynamic path

            let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);

            println!("[DEBUG] Pug logs directory: {:?}", pug_logs_dir);

    

        println!("[PROCESS] Task {} started processing.", task.id);

    // [KV-CHECK] Check if task specific KV exists
    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        println!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id);
    }

    // [NEW] Clear any leftover resources from previous failed attempts at the START
    cleanup_task_resources(&task.id, Some(app_handle));

    // [SPINNER-ACTIVATE] Ensure UI spinner is ON immediately upon task recovery/start
    let payload = json!({ 
        "task_id": task.id,
        "category": "Processing", "summary": "Resuming task...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }



        let payload = json!({ 



            "task_id": task.id,



            "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"



        });



        let _ = app_handle.emit("extraction-progress", &payload);



        log_task_progress(app_handle, &task.id, &payload);



    // [MEMORY] Initialize Data Manager for this task scope with AppHandle for dynamic paths

    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    // --- Image Extraction Logic ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        
                let mut model_lock = model_mutex.lock().await;
        
                if model_lock.is_none() {
                    let payload = json!({ 
                       "task_id": task.id,
                       "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
                    });
                    let _ = app_handle.emit("extraction-progress", &payload);
                    log_task_progress(app_handle, &task.id, &payload);
        
                    match LogisModel::new(None).await {
        
                        Ok(m) => *model_lock = Some(m),
        
                        Err(e) => {
                            let payload = json!({ 
                               "task_id": task.id,
                               "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                            });
                            let _ = app_handle.emit("extraction-progress", &payload);
                            log_task_progress(app_handle, &task.id, &payload);
        
                            if let Some(m) = model_lock.as_ref() {
        
                                m.unload_generator().await;
        
                            }
        
                            return Ok(());
        
                        }
        
                    }
        
                }
        
                
        
                if let Some(model) = model_lock.as_ref() {
        
                    let res = model.extract_from_image(
        
                        task.id.clone(),
        
                        image_path,
        
                        language.to_string(),
        
                        app_handle,
        
                        Some(cancellation_token.clone()),
        
                        store_mutex
        
                    ).await;
        
                    
        
                    drop(model_lock);
        
                    return res;
        
                }
        
        
        return Ok(());
    }

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();

    if url.is_empty() {
        return Ok (());
    }

    // [MEMORY] Fetch and immediately offload Raw HTML
    let raw_html_path = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let p = data_manager.offload(raw_html, "raw_html")?;
        if let Some(obj) = task_data.as_object_mut() {
            obj.remove("html"); // Clear from JSON object in memory
        }
        p
    } else if !url.is_empty() {
        let content = reqwest::get(&url).await?.text().await?;
        data_manager.offload(&content, "raw_html")?
    } else {
        return Ok (());
    };
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // [MEMORY] Load Raw -> Clean -> Offload Clean -> Drop Raw
    let clean_html_path = {
        let raw_content = data_manager.load(&raw_html_path)?;
        let clean = parsing::pre_clean_html(&raw_content);
        drop(raw_content); // Free Raw RAM
        data_manager.offload(&clean, "clean_html")?
    };
    
    // [MEMORY] Load Clean -> Parse Pug -> Drop Clean
    let light_pug = {
        let clean_content = data_manager.load(&clean_html_path)?;
        // [CLASSIFICATION ONLY] Use high-level entry function as requested
        parsing::convert_to_clean_pug(&clean_content, PugMode::FullContent)
    }; 
    
    // [DEBUG-LOG] Save generated Pug with nanosecond precision to prevent overwriting
    let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let log_path = pug_logs_dir.join(format!("light_{}_{}.pug", task.id, ts_nano));
    if let Err(e) = std::fs::write(&log_path, &light_pug) {
        println!("[ERROR] Failed to write light pug: {}", e);
    } else {
        println!("[DEBUG] Saved light pug to: {:?}", log_path);
    }
    
                    // [REVISED] Split Pug into line-based chunks (approx 1024 tokens ~ 3000 chars)
                    // and save each to log/pug directory for indexing.
                    let pug_chunks = chunk_text(&light_pug, 3072); 
                    let pug_chunks_len = pug_chunks.len(); 
                    println!("[Scheduler] Pug split into {} line-based chunks.", pug_chunks_len);
                
                    // Save chunks and create index
                    let mut chunk_index = Vec::new();
                    for (i, chunk) in pug_chunks.iter().enumerate() {
                        let chunk_filename = format!("chunk_{}.pug", i);
                        let chunk_path = pug_logs_dir.join(&chunk_filename);
                        if let Err(e) = std::fs::write(&chunk_path, chunk) {
                            println!("[ERROR] Failed to save pug chunk {}: {}", i, e);
                        }
                        chunk_index.push(json!({
                            "index": i,
                            "file": chunk_filename,
                            "size": chunk.len()
                        }));
                    }
                    let _ = std::fs::write(pug_logs_dir.join("index.json"), json!(chunk_index).to_string());
                
                    // [LOCK-MINIMIZATION] Acquire clones and release locks immediately.
                    // This allows stop_current_extraction to clear global references without deadlocking.
                    
                    let store = {
                        let store_guard = store_mutex.lock().await;
                        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
                    };
                
                    let model = {
                        let mut model_lock = model_mutex.lock().await;
                        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                        if model_lock.is_none() {
                            let payload = json!({
                               "task_id": task.id,
                               "category": "Loading Model", "summary": "Loading Model for Analysis...", "spinner": "⠋"
                            });
                            let _ = app_handle.emit("extraction-progress", &payload);
                            match LogisModel::new(None).await {
                                Ok(m) => *model_lock = Some(m),
                                Err(e) => return Err(anyhow::anyhow!("Model Load Failed: {}", e)),
                            }
                        }
                        model_lock.as_ref().unwrap().clone()
                    };
                
                    // Release the initial locks so tasks can re-acquire them as needed (e.g. stop command)
                    // Both 'store' and 'model' are now local clones (Arc-based).
                
                    // ==================================================================================
                    // [STRICT SWITCHING PATTERN]
                    // 1. Ingest content + Question with 0.6B (Save Cache)
                    // 2. Load 2B (Load Cache) -> Answer
                    // Repeat for each distinct task to ensure perfect cache alignment.
                    // ==================================================================================
                
                    use crate::openai_types::{
                        ChatCompletionParameters, ChatCompletionRequestMessage, 
                        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
                        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart,
                        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestAssistantMessage
                    };
                
                    let mut page_type = String::new();
                    let mut selector_info = json!({});
                
                    // --- TASK 1: CLASSIFICATION (DUAL-ENGINE REAL-TIME RELAY) ---
                    {
                        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                        println!("[Scheduler] Starting DUAL-ENGINE REAL-TIME RELAY (0.6B -> 2B Stream)");
                        let type_prompt = parsing::page_type_prompt();
                        
                        // [FIX] Correct Order: Load Small FIRST, then Large to enable DUAL-ENGINE handover
                        model.ensure_generator(crate::model::ModelSize::Small).await?;
                        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                        
                        // [INTERRUPTIBLE-LOAD] Race loading against cancellation
                        tokio::select! {
                            res = model.ensure_generator(crate::model::ModelSize::Large) => res?,
                            _ = async {
                                loop {
                                    if cancellation_token.load(Ordering::Relaxed) { break; }
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                }
                            } => return Err(anyhow::anyhow!("Task cancelled during model loading")),
                        }
                        
                        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                                // [LOOP-RELAY] Process each chunk individually
                                for (i, chunk) in pug_chunks.iter().enumerate() {
                                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                                    
                                    let is_last = i == pug_chunks_len - 1;
                                    let chunk_text = chunk.clone();
                        
                                    {
                                        let model_clone = model.clone();
                                        let token_clone = cancellation_token.clone();
                                        let chunk_to_send = chunk_text.clone();
                        
                                        println!("[Scheduler] Relay Part {}/{} ({} chars)...", i + 1, pug_chunks_len, chunk_to_send.len());
                                        
                                        tokio::task::spawn_blocking(move || -> Result<()> {
                                            let mut large_gen_lock = model_clone.generator.blocking_lock(); 
                                            let mut small_gen_lock = model_clone.small_generator.blocking_lock();
                                            
                                            if let (Some(worker), Some(target)) = (small_gen_lock.as_mut(), large_gen_lock.as_mut()) {
                                                // [LIGHTWEIGHT-RELAY] No chat templates, just raw snippet prefill
                                                worker.prefill_chunk(chunk_to_send, Some(token_clone), Some(target))?;
                                            }
                                            Ok(())
                                        }).await??;
                                    }
                        
                                    if is_last {
                                        // 3. 2B Inference (Final Question)
                                        if let Some(gen) = model.generator.lock().await.as_mut() {
                                            println!("[Scheduler] 2B Generating final classification answer...");
                                            let params = ChatCompletionParameters {
                                                messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                                    content: ChatCompletionRequestUserMessageContent::Text(format!("TASK: {}\n\nACTION: JSON ONLY", type_prompt)),
                                                    name: None,
                                                })],
                                                model: "qwen3vl".to_string(), max_tokens: Some(128), temperature: Some(0.1),
                                                ..Default::default()
                                            };
                                            let res = gen.generate(params, Some(cancellation_token.clone()), Some(task.id.clone()))?;
                                            let type_info = parsing::parse_json_from_llm(&res);
                                            page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                        }
                                    }
                                }                        
                        model.unload_generator().await; 
                        if page_type.is_empty() || page_type == "unknown" { return Ok(()); }
                    }
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
    wait_for_resources_settled(2500, 1500, Some(cancellation_token)).await?;

    // --- TASK 2: SELECTOR IDENTIFICATION (DUAL-ENGINE REAL-TIME RELAY) ---
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        println!("[Scheduler] Starting DUAL-ENGINE REAL-TIME RELAY (0.6B -> 2B Stream)");
        let selector_prompt = parsing::page_selectors_prompt(&page_type); 
        
        // [FIX] Using cloned model
        model.ensure_generator(crate::model::ModelSize::Small).await?;
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        
        // [INTERRUPTIBLE-LOAD] Race loading against cancellation
        tokio::select! {
            res = model.ensure_generator(crate::model::ModelSize::Large) => res?,
            _ = async {
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => return Err(anyhow::anyhow!("Task cancelled during model loading")),
        }
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // [LOOP-RELAY] Process each chunk individually
        for (i, chunk) in pug_chunks.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let is_last = i == pug_chunks_len - 1;
            let chunk_text = chunk.clone();

            {
                let model_clone = model.clone();
                let token_clone = cancellation_token.clone();
                let chunk_to_send = chunk_text.clone();

                println!("[Scheduler] Relay Part {}/{} ({} chars)...", i + 1, pug_chunks_len, chunk_to_send.len());

                // [THREAD-ISOLATION] Move heavy CPU work to a dedicated OS thread
                tokio::task::spawn_blocking(move || -> Result<()> {
                    let mut large_gen_lock = model_clone.generator.blocking_lock();
                    let mut small_gen_lock = model_clone.small_generator.blocking_lock();
                    
                    if let (Some(worker), Some(target)) = (small_gen_lock.as_mut(), large_gen_lock.as_mut()) {
                        worker.prefill_chunk(chunk_to_send, Some(token_clone), Some(target))?;
                    }
                    Ok(())
                }).await??;
            }

            if is_last {
                if let Some(gen) = model.generator.lock().await.as_mut() {
                    println!("[Scheduler] 2B Identifying selectors immediately...");
                    let params = ChatCompletionParameters {
                        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(format!("TASK: {}\n\nACTION: JSON ONLY", selector_prompt)),
                            name: None,
                        })],
                        model: "qwen3vl".to_string(), max_tokens: Some(512), temperature: Some(0.1),
                        ..Default::default()
                    };
                    let res = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_task2", task.id)))?;
                    selector_info = parsing::parse_json_from_llm(&res);
                    println!("[Scheduler] Selectors Identified: {}", selector_info);
                }
            }
        }
        model.unload_generator().await;
    }

    let mut final_page_info = json!({ "type": page_type });
    let is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    println!("[Scheduler] is_detail determined: {}", is_detail);

    // [STRICT PARITY] Ported from proxy/src/index.ts mechanism
    {
        let db = &store;
        {
            let team_id = if task.to.is_empty() { 
                crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
            } else { task.to.clone() };

            // [STRICT PARITY] Ported from proxy/src/index.ts mechanism
            // If URL is relative, it MUST have a valid origin to combine with.
            let origin_str = task_data.get("origin").and_then(|s| s.as_str()).ok_or_else(|| anyhow::anyhow!("Missing origin in task data"))?;
            let base_url = url::Url::parse(origin_str).map_err(|e| anyhow::anyhow!("Invalid origin URL: {}", e))?;

            let url_obj = match url::Url::parse(&url) {
                Ok(parsed) => parsed,
                Err(url::ParseError::RelativeUrlWithoutBase) => {
                    base_url.join(&url).map_err(|e| anyhow::anyhow!("Failed to join relative URL: {}", e))?
                },
                Err(e) => return Err(anyhow::anyhow!("Invalid target URL: {}", e)),
            };
            
            let raw_path = url_obj.path();
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, raw_path)); 
            
            // 2. bcc = hashId(type + (isDetail ? cc.toUpperCase() : cc)) - Crucial for Tree grouping
            let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));

            // 3. Prepare data exactly as proxy does
            let mut page_data = selector_info.clone();
            if let Some(obj) = page_data.as_object_mut() {
                obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                obj.insert("type".to_string(), json!(page_type));
            }

            let _ = db.upsert_item(
                "pages", 
                &page_id, 
                "pages", 
                page_data, 
                None,
                Some(&task.from),
                Some(&team_id),
                Some(&task.cc),
                Some(&bcc),
                Some(raw_path), // ref stored as path for lookup parity
                None
            ).await;
            println!("[Scheduler] Page learned with Proxy parity: {}", page_id);
        }
    } 

    // Merge selectors into final_page_info

    if let Some(obj) = selector_info.as_object() {
        for (k, v) in obj {
            final_page_info.as_object_mut().unwrap().insert(k.clone(), v.clone());
        }
    }
    
    let page_info = final_page_info; // Re-alias for original logic
    
    let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("");
    let node_selector = page_info.get("node").and_then(|s| s.as_str()).unwrap_or("");
    let item_selector = page_info.get("item").and_then(|s| s.as_str()).unwrap_or("");

    if page_type == "" || page_type == "unknown" {
        return Ok (());
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // [SELECTOR-COMBINATION] Robustly combine node and item selectors, handling comma-separated groups in both.
    let target_selector = if !node_selector.is_empty() && !item_selector.is_empty() {
        let nodes: Vec<&str> = node_selector.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        let items: Vec<&str> = item_selector.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        
        let mut combined = Vec::new();
        for node in &nodes {
            for item in &items {
                combined.push(format!("{} {}", node, item));
            }
        }
        combined.join(", ")
    } else if !item_selector.is_empty() {
        item_selector.to_string()
    } else if !node_selector.is_empty() {
        node_selector.to_string()
    } else {
        "body".to_string()
    };

    // [COUNT] Calculate actual matches for transparency
    let match_count = {
        if let Ok(content) = data_manager.load(&clean_html_path) {
            let doc = scraper::Html::parse_document(&content);
            scraper::Selector::parse(&target_selector).map(|s| doc.select(&s).count()).unwrap_or(0)
        } else { 0 }
    };

    let payload = json!({ 
        "task_id": task.id,
        "category": "Classification", 
        "summary": format!("Type: {}, Detail: {} ({} matches found)", 
            page_type, 
            is_detail,
            match_count
        ), 
        "spinner": "✅", 
        "data": page_info
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    let mut extracted_data = json!({});

    if !is_detail {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        let payload = json!({ 
            "task_id": task.id,
            "category": "List Processing", "summary": "Direct DOM extraction starting...", "spinner": "⠋"
        });
        let _ = app_handle.emit("extraction-progress", &payload);
        log_task_progress(app_handle, &task.id, &payload);

        let mut all_extracted_items = Vec::new();

        // [3번 가속: PRE-FETCH] 리스트 처리를 시작하기 전에 백그라운드에서 Large 모델의 무게추를 미리 로드함
        {
            tokio::spawn(async move {
                let large_dir = std::fs::canonicalize("src-tauri/models/Qwen3-VL-2B-Instruct-gguf")
                    .or_else(|_| std::fs::canonicalize("models/Qwen3-VL-2B-Instruct-gguf")).ok();
                if let Some(path) = large_dir {
                    let _ = pre_fetch_weights(&path);
                }
            });
        }

        // [FIX] Scoped block to ensure 'document' (!Send) is dropped before .await points
        {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            let field_selectors = selector_info.get("selectors").and_then(|v| v.as_object());

            // [SELECTOR-MATCHING] Use the combined target_selector directly
            let mut potential_items = Vec::new();
            
            if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                for item in document.select(&sel) {
                    potential_items.push(item);
                }
            } else {
                println!("[Scheduler] Invalid selector: {}", target_selector);
            }

            for item_node in potential_items {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                let mut item_json = json!({});
                let mut has_data = false;

                if let Some(subs) = field_selectors {
                    for (field_name, sel_str) in subs {
                        if let Ok(sub_sel) = scraper::Selector::parse(sel_str.as_str().unwrap_or("")) {
                            // 1. 현재 노드(item) 내부에서 검색
                            if let Some(found_el) = item_node.select(&sub_sel).next() {
                                let text = found_el.text().collect::<Vec<_>>().join("").trim().to_string();
                                if !text.is_empty() {
                                    item_json.as_object_mut().unwrap().insert(field_name.clone(), json!(text));
                                    has_data = true;
                                }
                            }
                            
                            // 2. [MULTI-ROW FIX] 만약 데이터를 못 찾았고, 다음 형제 노드가 '.rows'라면 거기서도 검색
                            if item_json.get(field_name).is_none() {
                                    if let Some(next_sibling) = item_node.next_sibling() {
                                        if let Some(sibling_el) = scraper::ElementRef::wrap(next_sibling) {
                                            if sibling_el.value().name() == "tr" && sibling_el.select(&sub_sel).next().is_some() {
                                                if let Some(found_el) = sibling_el.select(&sub_sel).next() {
                                                    let text = found_el.text().collect::<Vec<_>>().join("").trim().to_string();
                                                    item_json.as_object_mut().unwrap().insert(field_name.clone(), json!(text));
                                                    has_data = true;
                                                }
                                            }
                                        }
                                    }
                            }
                        }
                    }
                } else {
                    // [FALLBACK] No specific field selectors provided; extract full text
                    let text = item_node.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !text.is_empty() {
                        item_json.as_object_mut().unwrap().insert("text".to_string(), json!(text));
                        has_data = true;
                    }
                }

                if has_data || field_selectors.is_none() {
                    if item_json.get("id").is_none() {
                        if let Some(link_el) = item_node.select(&scraper::Selector::parse("a[href]").unwrap()).next() {
                            let href = link_el.value().attr("href").unwrap_or("");
                            item_json.as_object_mut().unwrap().insert("link".to_string(), json!(href));
                        }
                    }
                    all_extracted_items.push(item_json);
                }
            }
        } // 'document' is dropped here

        let total_items = all_extracted_items.len();
        println!("[Scheduler] Direct Extraction: Found {} items.", total_items);

        let payload = json!({ 
            "task_id": task.id,
            "category": "List Processing", 
            "summary": format!("Direct Extraction: Found {} items.", total_items), 
            "spinner": "✅"
        });
        let _ = app_handle.emit("extraction-progress", &payload);
        log_task_progress(app_handle, &task.id, &payload);

        // [REFINEMENT] If we have items but no specific field selectors (just raw text), use LLM to structure them.
        // Re-get field_selectors from selector_info which is still in scope, or rely on the logic that populated all_extracted_items with "text" only.
        let needs_refinement = selector_info.get("selectors").is_none() && !all_extracted_items.is_empty();

        if needs_refinement {
             let payload = json!({ 
                "task_id": task.id,
                "category": "Data Refinement", 
                "summary": "Refining extracted data with AI...", 
                "spinner": "⠋"
            });
            let _ = app_handle.emit("extraction-progress", &payload);
            log_task_progress(app_handle, &task.id, &payload);

            let batch_size = 20;
            let mut refined_items = Vec::new();
            let total_refine_count = all_extracted_items.len();
            
            // Clone items to process
            let items_to_process = all_extracted_items.clone();
            
            // Helper to merge JSON
            fn merge_json(a: &mut serde_json::Value, b: serde_json::Value) {
                if let (Some(a_obj), Some(b_obj)) = (a.as_object_mut(), b.as_object()) {
                    for (k, v) in b_obj {
                        if !a_obj.contains_key(k) || a_obj[k].is_null() {
                            a_obj.insert(k.clone(), v.clone());
                        }
                    }
                }
            }

            for (batch_idx, batch) in items_to_process.chunks(batch_size).enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                let start_item = batch_idx * batch_size + 1;
                let end_item = std::cmp::min(start_item + batch.len() - 1, total_refine_count);
                let is_last_batch = end_item == total_refine_count;

                // Construct prompt with indexed items
                let batch_text: String = batch.iter().enumerate().map(|(i, item)| {
                    format!("Item {}:\n{}", i + 1, item.get("text").and_then(|s| s.as_str()).unwrap_or(""))
                }).collect::<Vec<_>>().join("\n\n");

                let instruction = parsing::list2json(&page_type, &language);
                let action_flag = if is_last_batch { "ACTION: SAVE" } else { "ACTION: INGEST" };
                let pure_data_input = format!("[RAW ITEMS]\n{}\n\n{}", batch_text, action_flag);
                
                // [STRICT RELAY] 1. Ingest with 0.6B
                {
                    // Use the cloned model reference (lock is free)
                    model.ensure_generator(crate::model::ModelSize::Small).await?;
                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                    let app_handle_clone = app_handle.clone();
                    
                    let messages = vec![
                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            content: "You are a data recording assistant.".to_string(),
                            name: None,
                        }),
                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Array(vec![
                                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: pure_data_input.clone() })
                            ]),
                            name: None,
                        })
                    ];

                    let _ = model.chat_params_with_spinner(
                        ChatCompletionParameters { messages, model: "qwen3vl".to_string(), max_tokens: Some(16), temperature: Some(0.1), ..Default::default() },
                        &app_handle_clone, "Refinement (Small)",
                        json!({ "task_id": task.id, "category": "Refinement (Ingest)", "summary": format!("Ingesting batch {}/{}...", batch_idx + 1, (total_refine_count + batch_size - 1) / batch_size) }),
                        Some(cancellation_token.clone()), Some(task.id.clone())
                    ).await?;
                    
                    model.unload_generator().await;
                }
                
                wait_for_resources_settled(2500, 1500, Some(cancellation_token)).await?;

                // [STRICT RELAY] 2. Infer with 2B
                let refine_res_str = {
                    tokio::select! {
                        res = model.ensure_generator(crate::model::ModelSize::Large) => res?,
                        _ = async {
                            loop {
                                if cancellation_token.load(Ordering::Relaxed) { break; }
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                        } => return Err(anyhow::anyhow!("Task cancelled during model loading")),
                    }
                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                    let app_handle_clone = app_handle.clone();
                    
                    let messages = vec![
                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            content: "You are a data recording assistant.".to_string(),
                            name: None,
                        }),
                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Array(vec![
                                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: pure_data_input })
                            ]),
                            name: None,
                        }),
                        ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage { content: Some("".to_string()), ..Default::default() }),
                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Array(vec![
                                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { 
                                    text: format!("TASK: {}\n\nACTION: JSON ONLY", instruction) 
                                })
                            ]),
                            name: None,
                        })
                    ];

                    let res = model.chat_params_with_spinner(
                        ChatCompletionParameters { messages, model: "qwen3vl".to_string(), max_tokens: Some(2048), temperature: Some(0.95), ..Default::default() },
                        &app_handle_clone, "Refinement (Large)",
                        json!({ "task_id": task.id, "category": "Refinement (Infer)", "summary": format!("Refining items {}-{}...", start_item, end_item) }),
                        Some(cancellation_token.clone()), Some(task.id.clone())
                    ).await?;
                    
                    model.unload_generator().await;
                    res
                };

                if !refine_res_str.is_empty() {
                     let parsed = parsing::parse_json_from_llm(&refine_res_str);
                     // Expecting { "items": [...] } or just an array
                     let items_array = parsed.get("items").and_then(|v| v.as_array())
                                        .or_else(|| parsed.as_array());
                                        
                     if let Some(arr) = items_array {
                         let mut batch_refined = Vec::new();
                         for (i, refined) in arr.iter().enumerate() {
                             if i < batch.len() {
                                 let mut original = batch[i].clone();
                                 merge_json(&mut original, refined.clone());
                                 batch_refined.push(original.clone());
                                 refined_items.push(original);
                             }
                         }
                         
                         // Emit the batch results to be appended in the UI
                         let payload = json!({
                             "task_id": task.id,
                             "category": "Data Refinement",
                             "data": batch_refined,
                             "summary": format!("Refining items {}-{} of {}...", start_item, end_item, total_refine_count)
                         });
                         let _ = app_handle.emit("extraction-progress", &payload);
                         log_task_progress(app_handle, &task.id, &payload);

                         // If LLM returned fewer items than batch, fill with remaining originals
                         if arr.len() < batch.len() {
                             for i in arr.len()..batch.len() {
                                 refined_items.push(batch[i].clone());
                             }
                         }
                     } else {
                         // Fallback: keep original
                         refined_items.extend_from_slice(batch);
                     }
                } else {
                    refined_items.extend_from_slice(batch);
                }
            }
            
            if !refined_items.is_empty() {
                all_extracted_items = refined_items;
            }

            let payload = json!({ 
                "task_id": task.id,
                "category": "Data Refinement", 
                "summary": format!("Refined {} items with AI.", total_refine_count),
                "spinner": "✅"
            });
            let _ = app_handle.emit("extraction-progress", &payload);
            log_task_progress(app_handle, &task.id, &payload);
        }

        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // 이후 DB 저장 로직으로 연결 (기존 로직 활용)
        for mut item_json in all_extracted_items.clone() {
            if cancellation_token.load(Ordering::Relaxed) { break; }
            // [STRICT PARITY] 1개씩 처리하며 DB에 넣기
            let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let from_addr = if task.from.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from };
            let team_id = if task.to.is_empty() { 
                crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
            } else { task.to.clone() };

            let link = item_json.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let item_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));
            item_json.as_object_mut().unwrap().insert("id".to_string(), json!(item_id));
            item_json.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));
            
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
            let ref_val = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));

            let nl = parsing::json_to_natural_language(&item_json);
            let item_digest = crate::utils::hash::digest(&nl);

            {
                let db = &store;
                let _ = db.upsert_item(
                    "items", &item_id, &page_type, item_json.clone(), None,
                    Some(from_addr), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
                ).await;
            }
        }
        
        extracted_data = json!({
            "items": all_extracted_items,
            "type": page_type,
            "item": item_selector,
            "node": node_selector,
            "detail": false
        });
    } else {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        // [RESTORED] Detail Extraction Logic (Parity with a60f234b)
        // Instead of reusing StructureOnly cache, we extract specific FullContent for the target selector.
        let content_pug = {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            let mut pug_output = String::new();
            
            if let Ok(selector) = scraper::Selector::parse(&target_selector) {
                for node in document.tree.root().descendants() {
                    if let Some(element_ref) = scraper::ElementRef::wrap(node) {
                        if selector.matches(&element_ref) {
                             parsing::generate_pug_lines(node, 0, &mut pug_output, &PugMode::FullContent);
                             break;
                        }
                    }
                }
            }
            parsing::sanitize_llm_input(&pug_output)
        };

        if content_pug.trim().is_empty() {
            println!("[Scheduler] Error: No content found with selector '{}'", target_selector);
            return Ok(());
        }

                        // [DEBUG-LOG] Save content pug
                        let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                        let _ = std::fs::write(pug_logs_dir.join(format!("content_{}_{}.pug", task.id, ts_nano)), &content_pug);
                
                                // [HYBRID] Detail Extraction (REAL-TIME RELAY MODE)
                
                                let extraction_instruction = parsing::item2json(page_type, &url, language);
                
                                let detail_session_id = format!("{}_detail", task.id); 
                
                        
                
                                // 1. Prepare Unified Parameters for Detail
                
                                let params = ChatCompletionParameters {
                
                                    messages: vec![
                
                                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                
                                            content: extraction_instruction.clone(),
                
                                            name: None,
                
                                        }),
                
                                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                
                                            content: ChatCompletionRequestUserMessageContent::Array(vec![
                
                                                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { 
                
                                                    text: format!("{}\n\nTASK: JSON ONLY\n\nACTION: SAVE", content_pug) 
                
                                                })
                
                                            ]),
                
                                            name: None,
                
                                        })
                
                                    ],
                
                                    model: "qwen3vl".to_string(), max_tokens: Some(2048), temperature: Some(0.1),
                
                                    ..Default::default()
                
                                };
                
                        
                
                                        // 2. Real-time Stream from 0.6B to 2B
                
                        
                
                                        {
                                            // Using the cloned model reference (lock is free)
                                            model.ensure_generator(crate::model::ModelSize::Small).await?;
                                            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                                            // [INTERRUPTIBLE-LOAD] Race loading against cancellation
                                            tokio::select! {
                                                res = model.ensure_generator(crate::model::ModelSize::Large) => res?,
                                                _ = async {
                                                    loop {
                                                        if cancellation_token.load(Ordering::Relaxed) { break; }
                                                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                                    }
                                                } => return Err(anyhow::anyhow!("Task cancelled during model loading")),
                                            }
                                            
                                            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                                            {
                                                let model_clone = model.clone();
                                                let params_clone = params.clone();
                                                let token_clone = cancellation_token.clone();
                                                let task_id_clone = detail_session_id.clone();

                                                println!("[Scheduler] Detail Ingestion (Blocking Thread)...");

                                                let _ = tokio::task::spawn_blocking(move || -> Result<()> {
                                                    let mut large_gen_lock = model_clone.generator.blocking_lock();
                                                    let mut small_gen_lock = model_clone.small_generator.blocking_lock();
                                                    if let (Some(worker), Some(target)) = (small_gen_lock.as_mut(), large_gen_lock.as_mut()) {
                                                        worker.prefill_only(params_clone, Some(token_clone), Some(task_id_clone), Some(target))?;
                                                    }
                                                    Ok(())
                                                }).await??;
                                            }
                                            model.unload_generator().await; // Unload Small
                                        }
                
                                
                
                                wait_for_resources_settled(2500, 1500, Some(cancellation_token)).await?;
                
                        
                
        // 3. Precise Infer with 2B (Immediate)
        let response = {
            tokio::select! {
                res = model.ensure_generator(crate::model::ModelSize::Large) => res?,
                _ = async {
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                } => return Err(anyhow::anyhow!("Task cancelled during model loading")),
            }
            
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let res = if let Some(gen) = model.generator.lock().await.as_mut() {
                println!("[Scheduler] 2B Extracting details immediately...");
                gen.generate(params, Some(cancellation_token.clone()), Some(detail_session_id.clone()))?
            } else { "{}".to_string() };
            
            model.unload_generator().await;
            res
        };

        if !response.is_empty() {
            println!("[EXTRACT-FINAL] Result: {}", response);
            extracted_data = parsing::parse_json_from_llm(&response);
        }
    }    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    if let Some(obj) = extracted_data.as_object_mut() {
        if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type)); }
    }

    let mut normalized_data = json!({});
    if let Some(obj) = extracted_data.as_object() {
        for (k, v) in obj {
            let val = if let Some(inner_obj) = v.as_object() {
                if let Some(value_field) = inner_obj.get("value") { value_field.clone() } else { v.clone() }
            } else { v.clone() };
            normalized_data.as_object_mut().unwrap().insert(k.clone(), val);
        }
    }
    
    let id_val_raw = normalized_data.get("id").or_else(|| normalized_data.get("index")).cloned();
    // [STRICT PARITY] Use the task's existing destination (respecting login status)
    let team_id = if !task.to.is_empty() { 
        task.to.clone() 
    } else { 
        crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
    };

    // [STRICT PARITY] Generate numeric index using CRC32: crc32(hashId(type + team + no))
    let item_no = id_val_raw.as_ref().and_then(|v| v.as_str()).unwrap_or("").to_string();
    let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&item_no).replace("-", "").replace("_", "");
    let index_hash_input = format!("{}{}{}", page_type, team_id, clean_no);
    let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&index_hash_input));
    
    normalized_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
    normalized_data.as_object_mut().unwrap().insert("id".to_string(), json!(crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val))));
    
    extracted_data = normalized_data;

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let payload = json!({ 
        "task_id": task.id,
        "category": "Saving", "summary": "Syncing related entities...", "spinner": "⠋"
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    // [NEW] Side-Effect Logic: If order has goods, iterate and upsert tracking/goods links
    if page_type == "order" {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            println!("[Scheduler] Order contains {} goods. Processing side effects...", goods_arr.len());
            {
                let db = &store;
                let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                for (_g_idx, good) in goods_arr.iter().enumerate() {
                    let g_no = good.get("id").or_else(|| good.get("no")).and_then(|v| v.as_str()).unwrap_or("");
                    if !g_no.is_empty() {
                        let clean_g_no = crate::utils::hash::normalize_numeric_homoglyphs(g_no).replace("-", "").replace("_", "");
                        let g_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("goods{}{}", team_id, clean_g_no)));
                        
                        // Create associated tracking entry for each good in the order
                        let tracking_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, index_val, clean_g_no));
                        let mut tracking_data = extracted_data.clone();
                        tracking_data.as_object_mut().unwrap().insert("type".to_string(), json!("tracking"));
                        tracking_data.as_object_mut().unwrap().insert("goods".to_string(), json!(g_index));
                        tracking_data.as_object_mut().unwrap().insert("order".to_string(), json!(index_val));
                        
                        let _ = db.upsert_item(
                            "tracking", 
                            &tracking_id, 
                            "tracking", 
                            tracking_data, 
                            None,
                            Some(&task.from), Some(&team_id), Some(&task.cc),
                            Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                            Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                            None
                        ).await;
                    }
                }
            }
        }
    }

    if let Some((queries, merge_info)) = logic::relay(page_type, &extracted_data) {
        println!("[Scheduler] Relay logic triggered. Queries: {}", queries.len());
        
        {
            let db = &store;
            let mut target_data = extracted_data.clone();
            let team_id = if task.to.is_empty() { 
                crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
            } else { task.to.clone() };

            // [FIX] Use the already hashed r#ref from the task to maintain parity with frontend search criteria
            let mut target_id = task.r#ref.clone();

            let mut found_existing = false;
            let to_table = merge_info.to.clone(); 

            for query in queries {
                let query_table = query.table.clone(); 
                if let Ok(Some((id, existing_data))) = db.find_item_by_property(&query_table, &query.column, &query.value).await {
                    println!("[Scheduler] Found existing item: {} in {}", id, query_table);
                    logic::merge_node(&mut target_data, &existing_data);
                    target_id = id;
                    found_existing = true;
                    // Note: In proxy, multiple items could be updated, but for local LanceDB 
                    // we merge into the most relevant existing record.
                    break;
                }
            }
            
            if !found_existing {
                if let Some(id_val) = target_data.get("id").and_then(|v| v.as_str()) {
                    target_id = id_val.to_string();
                }
            }

            let mut text_to_embed = parsing::json_to_natural_language(&target_data);
            let item_digest = crate::utils::hash::digest(&text_to_embed); 
            
            // [NEW] Skip high-cost embedding if content hasn't changed
            let mut existing_vector = None;
            if let Ok(Some(existing_item)) = db.get_item_by_id(&to_table, &target_id).await {
                if existing_item.digest == item_digest {
                    println!("[Scheduler] Digest match for {}. Skipping embedding.", target_id);
                    existing_vector = Some(existing_item.vector);
                }
            }

            if text_to_embed.chars().count() > 3000 {
                text_to_embed = text_to_embed.chars().take(3000).collect();
            }
            
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let vector = if let Some(v) = existing_vector {
                Some(v)
            } else {
                let text_clone = text_to_embed.clone();
                tokio::select!{
                    res = model.get_embedding(text_clone) => Some(res?),
                    _ = async {
                        loop {
                            if cancellation_token.load(Ordering::Relaxed) { break; }
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    } => { return Err(anyhow::anyhow!("Task cancelled")); }
                }
            };
            
            {
                let db = &store;
                println!("[Scheduler] Saving item: {} to {}", target_id, to_table);
                
                let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                let from_addr = if task.from.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from };

                // [STRICT PARITY] Re-generate BCC and REF for the target
                let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
                let link = target_data.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, link));

                let _ = db.upsert_item(
                    &to_table, 
                    &target_id, 
                    page_type, 
                    target_data.clone(), 
                    vector.clone(),
                    Some(from_addr),
                    Some(&team_id),
                    Some(&task.cc),
                    Some(&bcc),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
                
                let _ = db.upsert_item(
                    "items", 
                    &target_id, 
                    page_type, 
                    target_data, 
                    vector,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&task.cc),
                    Some(&bcc),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
            }
        }
    } else {
        let target_table = page_type.to_string();
        {
            let db = &store;
            let mut text_to_embed = parsing::json_to_natural_language(&extracted_data);
                let item_digest = crate::utils::hash::digest(&text_to_embed); 

                let _target_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, task.r#ref)); 
                let mut existing_vector = None;
                // [FIX] Use task.r#ref directly as the target_id to match parity with frontend
                let target_id = task.r#ref.clone(); 
                
                if let Ok(Some(existing_item)) = db.get_item_by_id(&target_table, &target_id).await {
                    if existing_item.digest == item_digest {
                        println!("[Scheduler] Digest match for direct item {}. Skipping embedding.", target_id);
                        existing_vector = Some(existing_item.vector);
                    }
                }

                if text_to_embed.chars().count() > 3000 {
                    text_to_embed = text_to_embed.chars().take(3000).collect();
                }
                
                let vector = if let Some(v) = existing_vector {
                                Some(v)
                            } else {
                                let text_clone = text_to_embed.clone();
                                tokio::select!{
                                    res = model.get_embedding(text_clone) => Some(res?),
                                    _ = async {
                                        loop {
                                            if cancellation_token.load(Ordering::Relaxed) { break; }
                                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                        }
                                    } => { return Err(anyhow::anyhow!("Task cancelled")); }
                                }
                            };
                {
                    let db = &store;
                    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                    let from_addr = if task.from.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from };
                    let team_id = if task.to.is_empty() { 
                        crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
                    } else { task.to.clone() };

                    // [STRICT PARITY] Re-generate BCC and REF exactly as server does
                    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
                    // [FIX] Use the pre-calculated hashed r#ref from the task to match frontend criteria
                    let ref_val = task.r#ref.clone(); 
                    
                    // target_id is already set to task.r#ref above 

                    let _ = db.upsert_item(
                        &target_table, 
                        &target_id, 
                        page_type, 
                        extracted_data.clone(), 
                        vector.clone(),
                        Some(from_addr),
                        Some(&team_id),
                        Some(&task.cc),
                        Some(&bcc),
                        Some(&ref_val),
                        Some(&item_digest)
                    ).await;
                    
                    let _ = db.upsert_item(
                        "items", 
                        &target_id, 
                        page_type, 
                        extracted_data.clone(), 
                        vector,
                        Some(from_addr),
                        Some(&team_id),
                        Some(&task.cc),
                        Some(&bcc),
                        Some(&ref_val),
                        Some(&item_digest)
                    ).await;
                }
        }
    }
        // Final Done Emission
    let final_summary = if !is_detail {
        let count = extracted_data.get("items").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
        format!("Extraction Complete ({} items saved)", count)
    } else {
        "Extraction Complete".to_string()
    };

    // [STRICT PARITY] Update Chat Message in DB using integer status
    {
        let db = &store;
        {
            let status_code = logic::parse_status("complete");
            let _ = db.update_message_status(&task.id, status_code, Some(&final_summary)).await;
        }
    }

        let payload = json!({
            "task_id": task.id,
            "category": "Done",
            "summary": "Extraction complete.",
            "spinner": "✅",
            "data": if !is_detail { json!(null) } else { extracted_data }
        });
        let _ = app_handle.emit("extraction-progress", &payload);
        log_task_progress(app_handle, &task.id, &payload);
    
        // [LOCK-RELEASE] 정상 종료 시 모델 및 리소스 완전 해제
        {
            model.unload_generator().await;
            println!("[PROCESS] Task {} completed. Model unloaded and resources released.", task.id);
        }
        
        Ok(())
    }
    
    fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
        // [FIX] Only remove the specific directory for this task
        let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
    }

// [3번 가속: PRE-FETCH] OS 페이지 캐시에 무게추 파일을 미리 로드함
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
                            let mut buffer = [0u8; 1024 * 1024]; // 1MB buffer
                            // 파일 전체를 읽어서 OS가 램에 캐싱하도록 유도함
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

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        let line = format!("{}\n", payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }
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
    // Lower default RAM threshold slightly for better compatibility (1000MB)
    let adjusted_ram_target = if target_ram_mb > 1000 { 1000 } else { target_ram_mb };
    let target_ram_bytes = adjusted_ram_target * 1024 * 1024;

    let mut stable_count = 0;
    let mut last_report = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB, RAM > {}MB)...", target_vram_mb, adjusted_ram_target);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); // Only refresh memory for speed
        let current_ram = sys.available_memory();
        let mut max_free_vram = 0;
        let mut _best_gpu_id = 0;
        let mut _gpu_count = 0;

        let has_gpu = if let Some(ref nvml_inst) = nvml {
            if let Ok(count) = nvml_inst.device_count() {
                _gpu_count = count;
                for i in 0..count {
                    if let Ok(dev) = nvml_inst.device_by_index(i) {
                        if let Ok(mem) = dev.memory_info() {
                            if mem.free > max_free_vram {
                                max_free_vram = mem.free;
                                _best_gpu_id = i;
                            }
                        }
                    }
                }
                _gpu_count > 0
            } else { false }
        } else { false };

        let meets_vram = !has_gpu || max_free_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            stable_count += 1;
            if stable_count >= 2 { // Reduced from 3 to 2 for faster transition
                break;
            }
        } else {
            stable_count = 0;
            if last_report.elapsed().as_secs() >= 5 {
                let vram_status = if has_gpu {
                    format!("VRAM: {:.2} / {:.2} GB", max_free_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0)
                } else {
                    "VRAM: N/A".to_string()
                };
                let ram_status = format!("RAM: {:.2} / {:.2} GB", current_ram as f64 / 1e9, adjusted_ram_target as f64 / 1024.0);
                
                println!("[RESOURCE-DIAG] Still waiting... {} | {}", vram_status, ram_status);
                last_report = std::time::Instant::now();
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}
