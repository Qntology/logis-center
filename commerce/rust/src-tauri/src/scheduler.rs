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
}

impl TaskDataManager {
    fn new(task_id: &str) -> Self {
        Self {
            task_id: task_id.to_string(),
            created_files: Vec::new(),
        }
    }

    fn offload(&mut self, content: &str, suffix: &str) -> Result<PathBuf> {
        let dir = std::path::Path::new("tmp_task_data");
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
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
        println!("[Cleanup] TaskDataManager dropping. Cleaning up files for task: {}", self.task_id);
        for path in &self.created_files {
            if path.exists() {
                let _ = fs::remove_file(path);
            }
        }
        // Try to remove task data dir if empty
        let _ = fs::remove_dir("tmp_task_data");

        // [STRICT PARITY] Also clean up KV cache subdirectories for this task
        let base_path = std::path::Path::new("tmp_kv");
        if let Ok(entries) = fs::read_dir(base_path) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&self.task_id) {
                        let _ = fs::remove_dir_all(entry.path());
                    }
                }
            }
        }
    }
}

// Helper to chunk text strictly by lines with a "push-back" (sliding) logic.
// If a line would exceed the chunk_size, it's moved to the next chunk entirely.
fn chunk_text(text: &str, chunk_size: usize, _overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    for line in text.lines() {
        let line_with_nl = format!("{}\n", line);
        
        // If this line alone is somehow bigger than the entire chunk_size,
        // we have to break it, but this is extremely rare for Pug.
        if line_with_nl.len() > chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
                current_chunk = String::new();
            }
            chunks.push(line_with_nl);
            continue;
        }

        // Check if adding this line exceeds the 1024 char (approx tokens) limit
        if current_chunk.len() + line_with_nl.len() > chunk_size {
            // [SLIDE] Push current chunk to results and start NEW chunk with this line
            chunks.push(current_chunk);
            current_chunk = line_with_nl;
        } else {
            current_chunk.push_str(&line_with_nl);
        }
    }
    
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
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
    clear_all_temp_data();
    
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
                                let mut model_guard = model.lock().await;
                                *model_guard = None; // Explicitly drop model from GPU
                                
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

    // [NEW] Clear any leftover resources from previous failed attempts at the START

    cleanup_task_resources(&task.id);



    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let _ = app_handle.emit("extraction-progress", json!({ 
        "task_id": task.id,
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"
    }));

    // [MEMORY] Initialize Data Manager for this task scope
    let mut data_manager = TaskDataManager::new(&task.id);

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    // --- Image Extraction Logic ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        
        let mut model_guard = model_mutex.lock().await;
        if model_guard.is_none() {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task.id,
               "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
            }));
            match LogisModel::new(None).await {
                Ok(m) => *model_guard = Some(m),
                Err(e) => {
                    let _ = app_handle.emit("extraction-progress", json!({ 
                       "task_id": task.id,
                       "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                    }));
                    return Ok(());
                }
            }
        }
        
        if let Some(model) = model_guard.as_ref() {
            let res = model.extract_from_image(
                task.id.clone(),
                image_path,
                language.to_string(),
                app_handle,
                Some(cancellation_token.clone()),
                store_mutex
            ).await;
            
            drop(model_guard);
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
        parsing::convert_doc_to_clean_pug(&document, PugMode::StructureOnly)
        // clean_content dropped here
    }; 
    
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    
    // Determine optimal chunk sizes based on hardware
    // [ADAPTIVE] Increased to 4000 chars to speed up ingestion.
    let mut device_config = utils::get_optimal_device_config();
    device_config.classify_chunk_size = 4000; 
    device_config.extract_chunk_size = 4000;

    // [REVISED] Restore turn-based chunked ingestion for classification.
    // This is the original stable logic.
    let classify_chunks = chunk_text(&light_pug, 4000, 0); // [SPEED-UP] Larger chunks for fewer parts
    let classify_chunks_len = classify_chunks.len(); 
    println!("[Scheduler] Classification: {} chunks created.", classify_chunks_len);

    let mut model_guard = model_mutex.lock().await;
    println!("[Scheduler] Model lock acquired.");
    
    if model_guard.is_none() {
        let _ = app_handle.emit("extraction-progress", json!({ 
           "category": "Loading Model", "summary": "Loading Model for Analysis...", "spinner": "⠋"
        }));
        match LogisModel::new(None).await {
            Ok(m) => *model_guard = Some(m),
            Err(e) => {
                let _ = app_handle.emit("extraction-progress", json!({ 
                   "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                }));
                return Ok(());
            }
        }
    }
    
    let mut page_type_res = String::new();

    // Step 1: Ingest chunks for Classification with full history
    use crate::openai_types::{
        ChatCompletionParameters, ChatCompletionRequestMessage, 
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestAssistantMessage
    };

    let mut messages = vec![
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: parsing::page_type_prompt(),
            name: None,
        })
    ];

    if let Some(model) = model_guard.as_ref() {
        let app_handle_clone = app_handle.clone();
        
        for (i, chunk) in classify_chunks.iter().enumerate() {
            let is_last = i == classify_chunks_len - 1;
            println!("[Scheduler] Classification: Processing chunk {}/{} (Last={})", i + 1, classify_chunks_len, is_last);
            
            let mut prompt = format!("[Reading structure part {}/{}]", i + 1, classify_chunks_len)

            let mut line_counter = 1;
            for line in chunk.lines() {
                prompt.push_str(&format!("\n{} | {}", line_counter, line));
                line_counter += 1;
            }

            if is_last {
                prompt.push_str("\n\nACTION: JSON ONLY");
            }

            // [OPTIMIZATION] Don't add Assistant's "READY" to history, only User data
            messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: prompt })
                ]),
                name: None,
            }));
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "task_id": task.id,
                "category": "Classification Ingestion", 
                "summary": format!("Reading structure part {}/{}...", i + 1, classify_chunks_len),
                "spinner": "⠋"
            }));

            let params = ChatCompletionParameters {
                messages: messages.clone(), 
                model: "qwen3vl".to_string(),
                max_tokens: Some(if is_last { 1024 } else { 32 }), // Doubled for safety
                temperature: Some(0.1), // Very deterministic for classification
                ..Default::default()
            };

            tokio::select!{
                res = model.chat_params_with_spinner(
                    params,
                    &app_handle_clone, 
                    "extraction-progress", 
                    json!({ 
                        "task_id": task.id,
                        "category": "Classification Ingestion",
                        "summary": if is_last { "Identifying page type...".to_string() } else { format!("Reading structure part {}/{}...", i + 1, classify_chunks_len) }
                    }), 
                    Some(cancellation_token.clone()), 
                    Some(task.id.clone())
                ) => { 
                    let out = res?;
                    if is_last {
                        page_type_res = out.clone();
                        let _ = app_handle.emit("extraction-progress", json!({ 
                            "task_id": task.id,
                            "category": "Classification Ingestion",
                            "summary": "Page type identified.",
                            "data": out,
                            "spinner": "✅"
                        }));
                    } 
                    // [MODIFIED] Do not push Assistant message back to history here
                },
                _ = async {
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                } => { return Err(anyhow::anyhow!("Task cancelled")); }
            }
        }
    }

    println!("[Scheduler] Checkpoint: Classification Start");
    println!("[Scheduler] DEBUG: page_type_res length: {}, content: '{:?}'", page_type_res.len(), page_type_res);
    if page_type_res.is_empty() {
        println!("[Scheduler] Error: page_type_res is empty. Model returned nothing.");
        return Err(anyhow::anyhow!("Model returned empty classification response"));
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut final_page_info = json!({});
    
    println!("[Scheduler] Checkpoint: Classification End");
    
    let type_info = parsing::parse_json_from_llm(&page_type_res);
    let page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
    println!("[Scheduler] Classification Result: '{}'", page_type);
    
    if page_type.is_empty() || page_type == "unknown" {
        println!("[Scheduler] Stopping: Unknown page type. Raw response: '{}'", page_type_res);
        // [FIX] Explicitly drop model before early return
        {
            let mut model_guard = model_mutex.lock().await;
            *model_guard = None;
        }
        return Ok(());
    }

    final_page_info.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    // Step 2: Identify Selectors (Building on History)
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type), "spinner": "⠋"
    }));

    let page_selectors_res = if let Some(model) = model_guard.as_ref() {
        let next_question = parsing::page_selectors_prompt(&page_type); 
        let app_handle_clone = app_handle.clone();
        
        // Add classification result and the new question to history
        messages.push(ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: Some(page_type_res.clone()),
            ..Default::default()
        }));
        
        messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(vec![
                ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: next_question })
            ]),
            name: None,
        }));

        let params = ChatCompletionParameters {
            messages: messages.clone(),
            model: "qwen3vl".to_string(),
            max_tokens: Some(256),
            temperature: Some(0.95),
            ..Default::default()
        };

        tokio::select!{
            res = model.chat_params_with_spinner(
                params,
                &app_handle_clone, 
                "extraction-progress", 
                json!({ 
                    "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type)
                }), 
                Some(cancellation_token.clone()), 
                Some(task.id.clone())
            ) => res?,
            _ = async {
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => { return Err(anyhow::anyhow!("Task cancelled")); }
        }
    } else { "{}".to_string() };
        drop(model_guard);
        
            let selector_info = parsing::parse_json_from_llm(&page_selectors_res);
            println!("[Scheduler] Selectors Found: {}", selector_info);        
        let is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    
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
                                }        // Merge selectors into final_page_info
        if let Some(obj) = selector_info.as_object() {
            for (k, v) in obj {
                final_page_info.as_object_mut().unwrap().insert(k.clone(), v.clone());
            }
        }
        
        let page_info = final_page_info; // Re-alias for original logic
        
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "Classification", 
            "summary": format!("Map: Type={}, Detail={}", 
                page_info.get("type").and_then(|s| s.as_str()).unwrap_or("?"),
                is_detail
            ), 
            "spinner": "✅", 
            "data": page_info
        }));
        
        let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("");
        let node_selector = page_info.get("node").and_then(|s| s.as_str()).unwrap_or("");
        let item_selector = page_info.get("item").and_then(|s| s.as_str()).unwrap_or("");

    if page_type == "" || page_type == "unknown" {
        return Ok (());
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let target_selector = if !item_selector.is_empty() { item_selector } else if !node_selector.is_empty() { node_selector } else { "body" };
    let mut extracted_data = json!({});

    if !is_detail {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "List Processing", "summary": "Splitting list items...", "spinner": "⠋"
        }));

        let mut current_selector = if !item_selector.is_empty() { item_selector.to_string() } else if !node_selector.is_empty() { node_selector.to_string() } else { "body".to_string() };
        
        // [MEMORY] Load clean HTML from disk only when needed
        let mut item_pugs = {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            parsing::split_doc_to_pug_list(&document, &current_selector, PugMode::FullContent)
        }; 

        // FALLBACK LOGIC: If primary selector fails, try the alternative
        if item_pugs.is_empty() && current_selector != "body" {
            let fallback = if current_selector == item_selector && !node_selector.is_empty() { node_selector.to_string() } else { "body".to_string() };
            if fallback != current_selector {
                println!("[Scheduler] Selector '{}' failed, trying fallback '{}'", current_selector, fallback);
                current_selector = fallback;
                // [MEMORY] Reload clean HTML again for fallback
                item_pugs = {
                    let clean_content = data_manager.load(&clean_html_path)?;
                    let document = scraper::Html::parse_document(&clean_content);
                    parsing::split_doc_to_pug_list(&document, &current_selector, PugMode::FullContent)
                };
            }
        }
        
        let total_items = item_pugs.len();
        println!("[Scheduler] List Processing: Found {} items using selector '{}'", total_items, current_selector);
        
        if total_items == 0 {
            println!("[Scheduler] Stopping: No items found even with fallback.");
            return Ok(());
        }


        let extraction_instruction = parsing::list2json(page_type, language);
        let mut all_extracted_items = Vec::new();

        // [BATCH OPTIMIZATION] Process 4 items at a time to improve stability
        for chunk in item_pugs.chunks(4) {
            if cancellation_token.load(Ordering::Relaxed) { break; }
            
            let current_start = all_extracted_items.len() + 1;
            let current_end = current_start + chunk.len() - 1;

            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Item Extraction", 
                "summary": format!("Extracting Items {}~{}/{}...", current_start, current_end, total_items),
                "spinner": "⠋"
            }));

            // [OPTIMIZATION] Separate Schema (System) from Data (User)
            // Initialize prompt with header and append snippets with line numbers
            let mut prompt = format!("[Reading structure part {}~{}/{}]", current_start, current_end, total_items);
            let mut line_counter = 1;
            for pug in chunk {
                for line in pug.lines() {
                    prompt.push_str(&format!("\n{} | {}", line_counter, line));
                    line_counter += 1;
                }
            }

            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
            
            let response = if let Some(model) = model_guard.as_ref() {
                let app_handle_clone = app_handle.clone();
                let summary_text = format!("Processing {}~{}/{}...", current_start, current_end, total_items);
                tokio::select!{
                    res = model.chat_with_spinner(
                        &extraction_instruction, // Schema goes here (System Prompt)
                        &prompt,                 // Pure Data goes here (User Prompt)
                        &app_handle_clone, 
                        "extraction-progress", 
                        json!({ 
                            "category": "Item Extraction",
                            "summary": summary_text
                        }), 
                        2048, 
                        Some(cancellation_token.clone()), 
                        None 
                    ) => res?,
                    _ = async {
                        loop {
                            if cancellation_token.load(Ordering::Relaxed) { break; }
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    } => { 
                        drop(model_guard);
                        cleanup_task_resources(&task.id);
                        return Err(anyhow::anyhow!("Task cancelled")); 
                    }
                }
            } else { "[]".to_string() };
            drop(model_guard);

            let batch_results = parsing::parse_json_from_llm(&response);
            if let Some(arr) = batch_results.as_array() {
                for mut item_json in arr.iter().cloned() {
                    if !item_json.is_null() && item_json.is_object() {
                        // [STRICT PARITY] Mirror server hashing and metadata
                        // task.cc is already a hash string from content.js (hashId(domain))
                        let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                        let from_addr = if task.from_source.is_empty() { "0x0000000000000000000000000000000000000000" } else { &task.from_source };
                        let team_id = if task.to_dest.is_empty() { 
                            crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") 
                        } else { task.to_dest.clone() };

                        let link = item_json.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        // item_id = hashId(cc + link) - Note: cookies.cc is used in JS
                        let item_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));
                        item_json.as_object_mut().unwrap().insert("id".to_string(), json!(item_id));
                        
                        // [STRICT PARITY] Generate BCC and REF exactly as server does
                        // bcc = hashId(page.type + (isDetail ? cc.toUpperCase() : cc))
                        let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
                        // [FIX] ref = hashId(cc + link) - matches frontend criteria
                        let ref_val = crate::utils::hash::hash_id(&format!("{}{}", task.cc, link));

                        if !item_id.is_empty() {
                            let mut model_guard = model_mutex.lock().await;
                            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
                            
                            let nl = parsing::json_to_natural_language(&item_json);
                            let item_digest = crate::utils::hash::digest(&nl);

                            let vector = if let Some(model) = model_guard.as_ref() {
                                Some(model.get_embedding(nl).await.unwrap_or_default())
                            } else { None };
                            drop(model_guard);

                            let store_guard = store_mutex.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.upsert_item(
                                    "items", 
                                    &item_id, 
                                    page_type, 
                                    item_json.clone(), 
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
                        all_extracted_items.push(item_json);
                    }
                }
            }

                                                

                                    

                        

             else if batch_results.is_object() {
                // Fallback for single object response
                all_extracted_items.push(batch_results.clone());
            }

            // [NEW] Emit intermediate results for this batch immediately
            let display_text = parsing::json_to_natural_language(&batch_results);
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Item Extraction", 
                "summary": format!("Extracted {}/{} items...", all_extracted_items.len(), total_items),
                "data": batch_results, // Original JSON data
                "display_text": display_text, // Human-readable narrative for UI
                "spinner": "⠋"
            }));
        }
        
        extracted_data = json!({
            "items": all_extracted_items,
            "type": page_type,
            "item": item_selector,
            "node": node_selector,
            "detail": false
        });
    } else {
        let content_pug = {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            parsing::convert_doc_to_clean_pug_selector(&document, target_selector, PugMode::FullContent)
        }; 
        
        if content_pug.trim().is_empty() {
            println!("[Scheduler] Error: No content found with selector '{}'", target_selector);
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Error", "summary": format!("Selector '{}' not found.", target_selector), "spinner": "❌"
            }));
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Done", "summary": "Extraction Failed", "spinner": "🛑", "data": null
            }));
            return Ok(());
        }

        let _ = std::fs::write("debug_content_pug.txt", &content_pug); 
        
        // CHUNKING: Split huge content into dynamic chunks with 0-char overlap (Device Optimized)
        let chunks = chunk_text(&content_pug, 1000, 0); 
        let extraction_instruction = parsing::item2json(page_type, &url, language);

        // [NEW] Use a single session ID for the entire detail page to maintain structural context (table headers, etc.)
        let detail_session_id = format!("{}_detail", task.id);
        
                // Phase 1: Ingest all chunks (Prefill)
                let chunks_len = chunks.len();
                
                for (chunk_idx, chunk) in chunks.iter().enumerate() {
                    let is_last = chunk_idx == chunks_len - 1;
        
                    // Prepare prompt: Only trigger extraction on the final chunk
                    let mut prompt = format!("[Reading structure part {}/{}]", chunk_idx + 1, chunks_len)
                    
                    let mut line_counter = 1;
                    for line in chunk.lines() {
                        prompt.push_str(&format!("\n{} | {}", line_counter, line));
                        line_counter += 1;
                    }

                    if is_last {
                        prompt.push_str("\n\nACTION: JSON ONLY");
                    }

                    let max_tokens = if is_last { 2048 } else { 32 }; 
                    
                    let _ = app_handle.emit("extraction-progress", json!({
                        "task_id": task.id,
                        "category": "Ingestion", 
                        "summary": format!("Reading part {}/{}...", chunk_idx + 1, chunks_len),
                        "spinner": "⠋"
                    }));
        
                    let model_guard = model_mutex.lock().await;
                    
                    let response = if let Some(model) = model_guard.as_ref() {
                        let app_handle_clone = app_handle.clone();
                        tokio::select!{
                            res = model.chat_with_spinner(
                                &extraction_instruction, // Schema consistently sent as System Prompt
                                &prompt,                 // Pure Data Chunk as User Prompt
                                &app_handle_clone, 
                                "extraction-progress", 
                                json!({ "category": "Ingestion" }), 
                                max_tokens, 
                                Some(cancellation_token.clone()), 
                                Some(detail_session_id.clone())
                            ) => res?,
                            _ = async {
                                loop {
                                    if cancellation_token.load(Ordering::Relaxed) { break; }
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                }
                            } => { 
                                drop(model_guard);
                                cleanup_task_resources(&task.id);
                                return Err(anyhow::anyhow!("Task cancelled")); 
                            }
                        }
                    } else {
                        "{}".to_string()
                    };
                    drop(model_guard);
        
            if is_last {
                println!("[EXTRACT-FINAL] Result: {}", response);
                let full_data = parsing::parse_json_from_llm(&response);
                extracted_data = full_data;
            }
        }
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
            
            drop(store_guard); 
            
            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
            
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let vector = if let Some(v) = existing_vector {
                Some(v)
            } else if let Some(model) = model_guard.as_ref() {
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
            drop(model_guard);
            
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

                let mut model_guard = model_mutex.lock().await;
                if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
                
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else if let Some(model) = model_guard.as_ref() {
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
                drop(model_guard);

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
        "summary": final_summary, 
        "spinner": "✅", 
        "data": if !is_detail { json!(null) } else { extracted_data }
    }));

    // [REMOVED] Manual cleanup call (Now handled by DataManager drop)

    // [RESOURCE-RELEASE] Explicitly drop the model to free VRAM/RAM after task completion
    {
        let mut model_guard = model_mutex.lock().await;
        *model_guard = None;
        println!("[Scheduler] Model resources released.");
    }

    Ok(())
}

fn cleanup_task_resources(task_id: &str) {
    println!("[Cleanup] Removing KV cache for task: {}", task_id);
    let base_path = std::path::Path::new("tmp_kv");
    if let Ok(entries) = fs::read_dir(base_path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(task_id) {
                    let _ = fs::remove_dir_all(entry.path());
                }
            }
        }
    }
}

fn clear_all_temp_data() {
    println!("[Cleanup] Clearing all temporary data directories...");
    let _ = fs::remove_dir_all("tmp_task_data");
    let _ = fs::remove_dir_all("tmp_kv");
    // Re-create them as empty
    let _ = fs::create_dir_all("tmp_task_data");
    let _ = fs::create_dir_all("tmp_kv");
}



