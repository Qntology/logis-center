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

// Helper to chunk text with overlap, strictly respecting newlines and char boundaries for Pug     
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut target_end = (start + chunk_size).min(text.len());

        // Ensure target_end is a valid char boundary
        while !text.is_char_boundary(target_end) {
            target_end -= 1;
        }

        let mut end = target_end;

        // Find the NEXT newline after target_end to include the current line fully
        if target_end < text.len() {
            if let Some(next_newline_offset) = text[target_end..].find('\n') {
                end = target_end + next_newline_offset + 1;
            } else {
                end = text.len();
            }
        }

        // Safety check for end boundary
        while !text.is_char_boundary(end) {
            end -= 1;
        }

        chunks.push(text[start..end].to_string());

        if end >= text.len() {
            break;
        }

        // Calculate next start with overlap
        let mut next_start = end.saturating_sub(overlap);

        // Ensure next_start is a valid char boundary
        while !text.is_char_boundary(next_start) {
            next_start -= 1;
        }

        // Align next_start to the START of a line (find the first newline in the overlap zone)    
        if next_start > start && next_start < end {
             if let Some(line_start) = text[next_start..end].find('\n') {
                 next_start = next_start + line_start + 1;
             }
        }

        // Final safety for next_start
        while !text.is_char_boundary(next_start) {
            next_start += 1; // Move forward to find valid char
        }

        // Prevent infinite loops or zero progress
        if next_start >= end {
            next_start = end;
        }

        // Absolute safety check to ensure progress
        if next_start <= start {
             next_start = start + 1; // Force progress by at least 1 byte (might panic on utf8, but better than loop)
             while !text.is_char_boundary(next_start) && next_start < text.len() {
                 next_start += 1;
             }
        }

        start = next_start;
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

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker started.");
    
    // [NEW] Clear ALL temporary data directories at startup for a clean slate
    clear_all_temp_data(Some(&app_handle));
    
    tokio::spawn(async move {
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



    // [LOCK-RELEASE] 시작 시 이전 잔여 리소스 강제 정리
    {
        // 모델 락을 잠깐 잡았다가 놓아서 혹시 모를 좀비 상태 정리
        let mut model_lock = model_mutex.lock().await;
        if let Some(m) = model_lock.as_ref() {
            println!("[PROCESS] Clearing residual model state...");
            m.unload_generator().await;
        }
        *model_lock = None; 

        // KV 폴더의 잠금 파일 확인 (Windows에서는 핸들을 닫는 것만으로 충분하지만, 명시적 로그)
        let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);

        if kv_path.exists() {

            println!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id);

        }

    }



    // [NEW] Clear any leftover resources from previous failed attempts at the START

    cleanup_task_resources(&task.id, Some(app_handle));



    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }



    let _ = app_handle.emit("extraction-progress", json!({ 

        "task_id": task.id,

        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"

    }));



    // [MEMORY] Initialize Data Manager for this task scope with AppHandle for dynamic paths

    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    // --- Image Extraction Logic ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        
                let mut model_lock = model_mutex.lock().await;
        
                if model_lock.is_none() {
        
                    let _ = app_handle.emit("extraction-progress", json!({ 
        
                       "task_id": task.id,
        
                       "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
        
                    }));
        
                    match LogisModel::new(None).await {
        
                        Ok(m) => *model_lock = Some(m),
        
                        Err(e) => {
        
                            let _ = app_handle.emit("extraction-progress", json!({ 
        
                               "task_id": task.id,
        
                               "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
        
                            }));
        
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
        let document = scraper::Html::parse_document(&clean_content);
        // [FIX] Use FullContent for ingestion to maximize context for 2B later
        parsing::convert_doc_to_clean_pug(&document, PugMode::FullContent)
        // clean_content dropped here
    }; 
    
    // [DEBUG-LOG] Save generated Pug with nanosecond precision to prevent overwriting
    let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let log_path = pug_logs_dir.join(format!("light_{}_{}.pug", task.id, ts_nano));
    if let Err(e) = std::fs::write(&log_path, &light_pug) {
        println!("[ERROR] Failed to write light pug: {}", e);
    } else {
        println!("[DEBUG] Saved light pug to: {:?}", log_path);
    }
    
        // Determine optimal chunk sizes based on hardware
        // [ADAPTIVE] Increased to 12000 chars to speed up ingestion and improve context coherence.
        let device_config = utils::get_optimal_device_config();
    
                // [REVISED] Restore turn-based chunked ingestion for classification.
                // This is the original stable logic.
                let classify_chunks = chunk_text(&light_pug, device_config.classify_chunk_size, 3000); 
                let classify_chunks_len = classify_chunks.len(); 
                println!("[Scheduler] Ingestion: {} chunks created.", classify_chunks_len);    let mut model_lock = model_mutex.lock().await;
    println!("[Scheduler] Model lock acquired.");
    
    if model_lock.is_none() {
        let _ = app_handle.emit("extraction-progress", json!({ 
           "category": "Loading Model", "summary": "Loading Model for Analysis...", "spinner": "⠋"
        }));
        match LogisModel::new(None).await {
            Ok(m) => {
                *model_lock = Some(m);
                let _ = app_handle.emit("extraction-progress", json!({ 
                   "category": "Loading Model", "summary": "Model Ready.", "spinner": "✅"
                }));
            },
            Err(e) => {
                let _ = app_handle.emit("extraction-progress", json!({ 
                   "task_id": task.id,
                   "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                }));
                if let Some(m) = model_lock.as_ref() {
                    m.unload_generator().await;
                }
                return Ok(());
            }
        }
    }
    
    // Step 1: Record (Ingest) with Small Model (0.6B)
    // Goal: Just swallow the huge PUG content to build a record/cache.
    
    use crate::openai_types::{
        ChatCompletionParameters, ChatCompletionRequestMessage, 
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestAssistantMessage
    };

    let mut messages = vec![
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: "You are a data recording assistant.".to_string(),
            name: None,
        })
    ];

    if let Some(model) = model_lock.as_ref() {
        let app_handle_clone = app_handle.clone();
        model.ensure_generator(crate::model::ModelSize::Small).await?;

        for (i, chunk) in classify_chunks.iter().enumerate() {
            let is_last = i == classify_chunks_len - 1;
            println!("[Scheduler] Recording (Small): Processing chunk {}/{} (Last={})", i + 1, classify_chunks_len, is_last);
            
            let prompt = chunk.clone();
            
            // [DEBUG-LOG] Save chunk for verification
            let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let log_path = pug_logs_dir.join(format!("record_06b_{}_{}.pug", i, ts_nano));
            let _ = std::fs::write(&log_path, &prompt);

            // NO QUESTION here. Just record the data.
            let action_flag = if is_last { "ACTION: SAVE" } else { "ACTION: INGEST" };
            let effective_prompt = format!("{}\n\n{}", prompt, action_flag);

            let _ = app_handle.emit("extraction-progress", json!({ 
                "task_id": task.id,
                "category": "Data Recording (Small)",
                "summary": format!("Recording content... ({}/{})", i + 1, classify_chunks_len),
                "spinner": "⠋" 
            }));

            messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: effective_prompt })
                ]),
                name: None,
            }));

            let params = ChatCompletionParameters {
                messages: messages.clone(),
                model: "qwen3vl".to_string(),
                max_tokens: Some(16), // No output expected
                temperature: Some(0.1),
                ..Default::default()
            };

            let res = tokio::select!{
                res = model.chat_params_with_spinner(
                    params,
                    &app_handle_clone, 
                    "Data Recording (Small)",
                    json!({ "task_id": task.id, "category": "Data Recording (Small)", "summary": "Recording..." }),
                    Some(cancellation_token.clone()), 
                    Some(task.id.clone())
                ) => res?,
                _ = async {
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                } => { return Err(anyhow::anyhow!("Task cancelled")); }
            };

            messages.push(ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: Some(res.clone()),
                ..Default::default()
            }));
        }
    }

    println!("[Scheduler] Small Model Recording Complete. Handing over to LARGE model for Preprocessing.");
    
    // [CRITICAL] Unload Small Model
    if let Some(m) = model_lock.as_ref() { m.unload_generator().await; }
    wait_for_resources_settled(1000, 1500, 5000).await;

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // Step 2: Preprocessing (Classification -> Selectors) with LARGE model (2B)
    if let Some(model) = model_lock.as_ref() {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "task_id": task.id,
            "category": "Intelligence Loading", "summary": "Loading 2B Model for Preprocessing...", "spinner": "⠋"
        }));
        model.ensure_generator(crate::model::ModelSize::Large).await?;
    }

    // [IMPORTANT] 2B needs the content again to do classification. 
    // We pass the full content or a significant part of it to 2B for the first time.
    let mut large_messages = vec![
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: "You are an expert web analysis assistant. Output must be strictly JSON format.".to_string(),
            name: None,
        })
    ];

    let page_type_res = if let Some(model) = model_lock.as_ref() {
        let type_prompt = parsing::page_type_prompt();
        let app_handle_clone = app_handle.clone();
        
        // 2B must ingest the data to classify. We use ACTION: SAVE so 2B's work is also cached.
        let header_chunk = classify_chunks.get(0).cloned().unwrap_or_default();
        let classification_input = format!("PAGE CONTENT:\n{}\n\nTASK: {}\n\nACTION: JSON ONLY\n\nACTION: SAVE", header_chunk, type_prompt);

        large_messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(vec![
                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: classification_input })
            ]),
            name: None,
        }));

        let params = ChatCompletionParameters {
            messages: large_messages.clone(),
            model: "qwen3vl".to_string(),
            max_tokens: Some(512),
            temperature: Some(0.1),
            ..Default::default()
        };

        let res = tokio::select!{
            res = model.chat_params_with_spinner(
                params,
                &app_handle_clone, 
                "Preprocessing (Large)",
                json!({ "task_id": task.id, "category": "Preprocessing (Large)", "summary": "Identifying page type..." }), 
                Some(cancellation_token.clone()), 
                Some(format!("{}_large", task.id)) 
            ) => res?,
            _ = async {
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => { return Err(anyhow::anyhow!("Task cancelled")); }
        };
        
        large_messages.push(ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: Some(res.clone()),
            ..Default::default()
        }));
        
        res
    } else { "{}".to_string() };

    let type_info = parsing::parse_json_from_llm(&page_type_res);
    let page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
    println!("[Scheduler] 2B Classification Result: '{}'", page_type);
    
    if page_type.is_empty() || page_type == "unknown" {
        if let Some(m) = model_lock.as_ref() { m.unload_generator().await; }
        return Ok(());
    }

    let mut final_page_info = json!({ "type": page_type });

    // Step 2b: Identify Selectors (Using 2B Model)
    let _ = app_handle.emit("extraction-progress", json!({
        "task_id": task.id,
        "category": "Preprocessing (Large)", "summary": "Determining selectors...", "spinner": "⠋"
    }));

    let page_selectors_res = if let Some(model) = model_lock.as_ref() {
        let next_question = parsing::page_selectors_prompt(&page_type); 
        let app_handle_clone = app_handle.clone();
        
        large_messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(vec![
                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: format!("{}\n\nACTION: SAVE", next_question) })
            ]),
            name: None,
        }));

        let params = ChatCompletionParameters {
            messages: large_messages.clone(),
            model: "qwen3vl".to_string(),
            max_tokens: Some(1024),
            temperature: Some(0.1),
            ..Default::default()
        };

        let res = tokio::select!{
            res = model.chat_params_with_spinner(
                params,
                &app_handle_clone, 
                "Preprocessing (Large)",
                json!({ "task_id": task.id, "category": "Preprocessing (Large)", "summary": "Finding selectors..." }), 
                Some(cancellation_token.clone()), 
                Some(format!("{}_large", task.id)) 
            ) => res?,
            _ = async {
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => { return Err(anyhow::anyhow!("Task cancelled")); }
        };
        res
    } else { "{}".to_string() };

    let selector_info = parsing::parse_json_from_llm(&page_selectors_res);

            println!("[Scheduler] Selectors Found: {}", selector_info);        
        let is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
        println!("[Scheduler] is_detail determined: {}", is_detail);

        // [STRICT PARITY] Ported from proxy/src/index.ts mechanism
        {
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let team_id = if task.to_dest.is_empty() { 
                    crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
                } else { task.to_dest.clone() };

                // 1. page_id = hashId(cc + pathname) - Stripping search params to prevent duplicate schemas for same route
                let clean_path = if let Some(pos) = task.ref_id.find('?') { &task.ref_id[..pos] } else { &task.ref_id };
                let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, clean_path)); 
                
                // 2. bcc = hashId(type + (isDetail ? cc.toUpperCase() : cc)) - Crucial for Tree grouping
                let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));
    
                // 3. Prepare data exactly as proxy does
                let mut page_data = selector_info.clone();
                if let Some(obj) = page_data.as_object_mut() {
                    let url_obj = url::Url::parse(&url).unwrap();
                    obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                    obj.insert("link".to_string(), json!(clean_path));
                    obj.insert("type".to_string(), json!(page_type));
                }

                let _ = db.upsert_item(
                    "pages", 
                    &page_id, 
                    "pages", 
                    page_data, 
                    None,
                    Some(&task.from_source),
                    Some(&team_id),
                    Some(&task.cc),
                    Some(&bcc),
                    Some(clean_path), // ref_id stored as clean path
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

    let _ = app_handle.emit("extraction-progress", json!({ 
        "task_id": task.id,
        "category": "Classification", 
        "summary": format!("Type: {}, Detail: {} ({} matches found)", 
            page_type, 
            is_detail,
            match_count
        ), 
        "spinner": "✅", 
        "data": page_info
    }));

    let mut extracted_data = json!({});

    if !is_detail {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "task_id": task.id,
            "category": "List Processing", "summary": "Direct DOM extraction starting...", "spinner": "⠋"
        }));

        let mut all_extracted_items = Vec::new();

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

        let _ = app_handle.emit("extraction-progress", json!({ 
            "task_id": task.id,
            "category": "List Processing", 
            "summary": format!("Direct Extraction: Found {} items.", total_items), 
            "spinner": "✅"
        }));

        // [REFINEMENT] If we have items but no specific field selectors (just raw text), use LLM to structure them.
        // Re-get field_selectors from selector_info which is still in scope, or rely on the logic that populated all_extracted_items with "text" only.
        let needs_refinement = selector_info.get("selectors").is_none() && !all_extracted_items.is_empty();

        if needs_refinement {
             let _ = app_handle.emit("extraction-progress", json!({ 
                "task_id": task.id,
                "category": "Data Refinement", 
                "summary": "Refining extracted data with AI...", 
                "spinner": "⠋"
            }));

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
                let start_item = batch_idx * batch_size + 1;
                let end_item = std::cmp::min(start_item + batch.len() - 1, total_refine_count);
                let is_last_batch = end_item == total_refine_count;

                // Construct prompt with indexed items
                let batch_text: String = batch.iter().enumerate().map(|(i, item)| {
                    format!("Item {}:\n{}", i + 1, item.get("text").and_then(|s| s.as_str()).unwrap_or(""))
                }).collect::<Vec<_>>().join("\n\n");

                let instruction = parsing::list2json(&page_type, &language);
                let action_flag = if is_last_batch { "ACTION: SAVE" } else { "ACTION: INGEST" };
                let prompt = format!("{}\n\n[RAW ITEMS]\n{}\n\n{}", instruction, batch_text, action_flag);
                
                // [FIX] Reuse existing history (chunks) to ensure safetensors cache hit
                if let Some(model) = model_lock.as_ref() {
                     let mut refine_messages = messages.clone();
                     refine_messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                         content: ChatCompletionRequestUserMessageContent::Array(vec![
                             ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: prompt })
                         ]),
                         name: None,
                     }));

                     let params = ChatCompletionParameters {
                        messages: refine_messages,
                        model: "qwen3vl".to_string(),
                        max_tokens: Some(30000),
                        temperature: Some(0.95),
                        ..Default::default()
                    };
                    
                    let app_handle_clone = app_handle.clone();
                    let refine_res = model.chat_params_with_spinner(
                        params,
                        &app_handle_clone, 
                        "extraction-progress", 
                        json!({ 
                            "task_id": task.id,
                            "category": "Data Refinement", 
                            "summary": format!("Refining items {}-{} of {}...", start_item, end_item, total_refine_count)
                        }),
                        Some(cancellation_token.clone()),
                        Some(task.id.clone()) // [INTEGRATED] Use original session ID to reuse main safetensors
                    ).await;

                    if let Ok(res) = refine_res {
                         let parsed = parsing::parse_json_from_llm(&res);
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
                             let _ = app_handle.emit("extraction-progress", json!({
                                 "task_id": task.id,
                                 "category": "Data Refinement",
                                 "data": batch_refined,
                                 "summary": format!("Refining items {}-{} of {}...", start_item, end_item, total_refine_count)
                             }));

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
                } else {
                    refined_items.extend_from_slice(batch);
                }
            }
            
            if !refined_items.is_empty() {
                all_extracted_items = refined_items;
            }

            let _ = app_handle.emit("extraction-progress", json!({ 
                "task_id": task.id,
                "category": "Data Refinement", 
                "summary": format!("Refined {} items with AI.", total_refine_count),
                "spinner": "✅"
            }));
        }

        // 이후 DB 저장 로직으로 연결 (기존 로직 활용)
        for mut item_json in all_extracted_items.clone() {
            // [STRICT PARITY] 1개씩 처리하며 DB에 넣기
            let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let from_addr = if task.from_source.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from_source };
            let team_id = if task.to_dest.is_empty() { 
                crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
            } else { task.to_dest.clone() };

            let link = item_json.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let item_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));
            item_json.as_object_mut().unwrap().insert("id".to_string(), json!(item_id));
            item_json.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));
            
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
            let ref_val = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));

            let nl = parsing::json_to_natural_language(&item_json);
            let item_digest = crate::utils::hash::digest(&nl);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
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
        // [RESTORED] Detail Extraction Logic (Parity with a60f234b)
        // Instead of reusing StructureOnly cache, we extract specific FullContent for the target selector.
        let content_pug = {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            parsing::convert_doc_to_clean_pug_selector(&document, &target_selector, PugMode::FullContent)
        };

        if content_pug.trim().is_empty() {
            println!("[Scheduler] Error: No content found with selector '{}'", target_selector);
            return Ok(());
        }

        // [DEBUG-LOG] Save content pug
        let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let _ = std::fs::write(pug_logs_dir.join(format!("content_{}_{}.pug", task.id, ts_nano)), &content_pug);

        // CHUNKING: Split huge content into chunks (24000 chars to be safe for details)
        let device_config = utils::get_optimal_device_config();
        let chunks = chunk_text(&content_pug, device_config.extract_chunk_size, 3000);
        let chunks_len = chunks.len();
        
        let extraction_instruction = parsing::item2json(page_type, &url, language);
        let detail_session_id = format!("{}_detail", task.id); // Separate session

        // Phase 1: Ingest all chunks (Accumulate -> Save)
        let mut detail_messages = vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: extraction_instruction.clone(),
                name: None,
            })
        ];

        let mut extracted_data_temp = json!({});

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let is_last = chunk_idx == chunks_len - 1;
            
            // Prepare prompt with INGEST/SAVE flags
            let mut prompt = chunk.clone();
            
            // On the very last chunk, we append the JSON instruction AND the SAVE flag
            // For intermediate chunks, just INGEST
            let action_flag = if is_last { 
                format!("\n\nACTION: JSON ONLY\n\nACTION: SAVE") 
            } else { 
                format!("\n\nACTION: INGEST") 
            };
            
            let effective_prompt = format!("{}{}", prompt, action_flag);

            // Add to history WITHOUT flag to avoid polluting prefix cache logic
            detail_messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: prompt.clone() })
                ]),
                name: None,
            }));

            let _ = app_handle.emit("extraction-progress", json!({ 
                "task_id": task.id,
                "category": "Detail Ingestion", 
                "summary": format!("Reading detail part {}/{}...", chunk_idx + 1, chunks_len),
                "spinner": "⠋" 
            }));

            let model_lock = model_mutex.lock().await;
            
            let response = if let Some(model) = model_lock.as_ref() {
                let app_handle_clone = app_handle.clone();
                
                // Create temporary messages with the flag for this turn only
                let mut current_messages = detail_messages.clone();
                current_messages.pop();
                current_messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: effective_prompt })
                    ]),
                    name: None,
                }));

                let params = ChatCompletionParameters {
                    messages: current_messages,
                    model: "qwen3vl".to_string(),
                    max_tokens: Some(if is_last { 30000 } else { 1024 }),
                    temperature: Some(0.95),
                    ..Default::default()
                };

                let res = tokio::select!{
                    res = model.chat_params_with_spinner(
                        params,
                        &app_handle_clone, 
                        "extraction-progress", 
                        json!({ "task_id": task.id, "category": "Detail Ingestion" }), 
                        Some(cancellation_token.clone()), 
                        Some(detail_session_id.clone())
                    ) => res?,
                    _ = async {
                        loop {
                            if cancellation_token.load(Ordering::Relaxed) { break; }
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    } => { 
                        if let Some(m) = model_lock.as_ref() {
                            m.unload_generator().await;
                        }
                        return Err(anyhow::anyhow!("Task cancelled")); 
                    }
                };
                
                // Accumulate back assistant message if not last (to keep context flow in VRAM)
                if !is_last {
                    detail_messages.push(ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                        content: Some(res.clone()),
                        ..Default::default()
                    }));
                }
                res
            } else {
                "{}".to_string()
            };
            drop(model_lock);

            if is_last {
                println!("[EXTRACT-FINAL] Result: {}", response);
                extracted_data_temp = parsing::parse_json_from_llm(&response);
            }
        }
        
        extracted_data = extracted_data_temp;
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
    let team_id = if !task.to_dest.is_empty() { 
        task.to_dest.clone() 
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

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Saving", "summary": "Syncing related entities...", "spinner": "⠋"
    }));

    // [NEW] Side-Effect Logic: If order has goods, iterate and upsert tracking/goods links
    if page_type == "order" {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            println!("[Scheduler] Order contains {} goods. Processing side effects...", goods_arr.len());
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
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
                            Some(&task.from_source), Some(&team_id), Some(&task.cc),
                            Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                            Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.ref_id))),
                            None
                        ).await;
                    }
                }
            }
        }
    }

    if let Some((queries, merge_info)) = logic::relay(page_type, &extracted_data) {
        println!("[Scheduler] Relay logic triggered. Queries: {}", queries.len());
        
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let mut target_data = extracted_data.clone();
            let team_id = if task.to_dest.is_empty() { 
                crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
            } else { task.to_dest.clone() };

            // [FIX] Use the already hashed ref_id from the task to maintain parity with frontend search criteria
            let mut target_id = task.ref_id.clone();

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
            } else if let Some(model) = model_lock.as_ref() {
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
            } else { None };
            
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                println!("[Scheduler] Saving item: {} to {}", target_id, to_table);
                
                let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                let from_addr = if task.from_source.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from_source };

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
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
                let mut text_to_embed = parsing::json_to_natural_language(&extracted_data);
                let item_digest = crate::utils::hash::digest(&text_to_embed); 

                let _target_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, task.ref_id)); 
                let mut existing_vector = None;
                // [FIX] Use task.ref_id directly as the target_id to match parity with frontend
                let target_id = task.ref_id.clone(); 
                
                if let Ok(Some(existing_item)) = db.get_item_by_id(&target_table, &target_id).await {
                    if existing_item.digest == item_digest {
                        println!("[Scheduler] Digest match for direct item {}. Skipping embedding.", target_id);
                        existing_vector = Some(existing_item.vector);
                    }
                }

                if text_to_embed.chars().count() > 3000 {
                    text_to_embed = text_to_embed.chars().take(3000).collect();
                }
                
                drop(store_guard);

                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else if let Some(model) = model_lock.as_ref() {
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
                } else { None };

                let store_guard = store_mutex.lock().await;
                if let Some(db) = store_guard.as_ref() {
                    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                    let from_addr = if task.from_source.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from_source };
                    let team_id = if task.to_dest.is_empty() { 
                        crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
                    } else { task.to_dest.clone() };

                    // [STRICT PARITY] Re-generate BCC and REF exactly as server does
                    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
                    // [FIX] Use the pre-calculated hashed ref_id from the task to match frontend criteria
                    let ref_val = task.ref_id.clone(); 
                    
                    // target_id is already set to task.ref_id above 

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
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let status_code = logic::parse_status("complete");
            let _ = db.update_message_status(&task.id, status_code, Some(&final_summary)).await;
        }
    }

        let _ = app_handle.emit("extraction-progress", json!({
            "task_id": task.id,
            "category": "Done",
            "summary": "Extraction complete.",
            "spinner": "✅",
            "data": if !is_detail { json!(null) } else { extracted_data }
        }));
    
        // [LOCK-RELEASE] 정상 종료 시 모델 및 리소스 완전 해제
        {
            if let Some(m) = model_mutex.lock().await.as_ref() {
                m.unload_generator().await;
            }
            println!("[PROCESS] Task {} completed. Model unloaded and locks released.", task.id);
        }
        
        Ok(())
    }
    
    fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
        // [FIX] Only remove the specific directory for this task
        let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
    }

fn clear_all_temp_data(app_handle: Option<&tauri::AppHandle>) {
    println!("[Cleanup] Clearing all temporary data directories...");
    utils::paths::cleanup_temp_dirs(app_handle);
}

/// Robustly waits for both VRAM and System RAM to be reclaimed.
/// It monitors if the resource availability has "settled" (stopped increasing).
async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, max_wait_ms: u64) {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    let start_time = std::time::Instant::now();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut last_ram = 0;
    let mut stable_count = 0;

    println!("[RESOURCE-WATCH] Monitoring recovery (VRAM > {}MB, RAM > {}MB)...", target_vram_mb, target_ram_mb);

    while start_time.elapsed().as_millis() < max_wait_ms as u128 {
        sys.refresh_memory();
        let current_ram = sys.available_memory();
        let mut current_vram = 0;

        if let Some(ref nvml_inst) = nvml {
            if let Ok(dev) = nvml_inst.device_by_index(0) {
                if let Ok(mem) = dev.memory_info() {
                    current_vram = mem.free;
                }
            }
        }

        // Check if we met the minimum thresholds AND if the values have stopped changing (settled)
        let meets_threshold = (current_vram >= target_vram_bytes || nvml.is_none()) && (current_ram >= target_ram_bytes);
        let is_stable = current_vram == last_vram && current_ram == last_ram;

        if meets_threshold && is_stable {
            stable_count += 1;
            if stable_count >= 3 { // Must be stable for 3 consecutive checks (600ms)
                println!("[RESOURCE-WATCH] Stabilized. VRAM: {:.2}GB, RAM: {:.2}GB free.", 
                    current_vram as f64 / 1e9, current_ram as f64 / 1e9);
                return;
            }
        } else {
            stable_count = 0;
        }

        last_vram = current_vram;
        last_ram = current_ram;

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    println!("[RESOURCE-WATCH] Timeout reached or thresholds not met. Proceeding anyway.");
}


