use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;

// Helper to chunk text with overlap, strictly respecting newlines and char boundaries for Pug
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let mut target_end = (start + chunk_size).min(text.len());
        
        while !text.is_char_boundary(target_end) {
            target_end -= 1;
        }

        let mut end = target_end;
        
        if target_end < text.len() {
            if let Some(next_newline_offset) = text[target_end..].find('\n') {
                end = target_end + next_newline_offset + 1;
            } else {
                end = text.len();
            }
        }
        
        while !text.is_char_boundary(end) {
            end -= 1;
        }

        chunks.push(text[start..end].to_string());
        
        if end >= text.len() {
            break;
        }
        
        let mut next_start = end.saturating_sub(overlap);
        
        while !text.is_char_boundary(next_start) {
            next_start -= 1;
        }
        
        if next_start > start && next_start < end {
             if let Some(line_start) = text[next_start..end].find('\n') {
                 next_start = next_start + line_start + 1;
             }
        }
        
        while !text.is_char_boundary(next_start) {
            next_start += 1;
        }

        if next_start >= end {
            next_start = end; 
        }
        
        if next_start <= start {
             next_start = start + 1; 
             while !text.is_char_boundary(next_start) && next_start < text.len() {
                 next_start += 1;
             }
        }

        start = next_start;
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

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker started.");
    
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
                println!("[Scheduler] Processing task: {}", task.id);
                cancellation_token.store(false, Ordering::SeqCst);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, "processing").await;
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, "done").await;
                        }
                    },
                    Err(e) => {
                        if e.to_string().contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             let store_guard = store.lock().await;
                             if let Some(db) = store_guard.as_ref() {
                                 let _ = db.update_task_status(&task.id, "cancelled").await;
                             }
                             let _ = app_handle.emit("extraction-progress", json!({ 
                                "category": "Done", "summary": "Cancelled by user", "spinner": "🛑", "data": null 
                             }));
                             break; 
                        } else {
                            println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, e);
                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, "error").await;
                            }
                        }
                    }
                }
            }
        }
    });
}

async fn process_task(
    task: Task, 
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle
) -> Result<()> {
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"
    }));

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("");
        
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "Image Loading", "summary": "Loading image...", "spinner": "⠋"
        }));

        if let Ok(img) = image::open(image_path) {
             let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
             let prompt = parsing::image2json("kr", language, "tracking", "");
             
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let mut model_guard = model_mutex.lock().await;
             if model_guard.is_none() {
                 let _ = app_handle.emit("extraction-progress", json!({ 
                    "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
                 }));
                 match LogisModel::new(None).await {
                     Ok(m) => *model_guard = Some(m),
                     Err(e) => {
                         let err_msg = format!("Model Load Failed: {}", e);
                         let _ = app_handle.emit("extraction-progress", json!({ 
                            "category": "Error", "summary": err_msg, "spinner": "❌"
                         }));
                         return Ok(());
                     }
                 }
             }
             
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let result_str = if let Some(model) = model_guard.as_ref() {
                 let prompt_clone = prompt.clone();
                 let app_handle_clone = app_handle.clone();
                 
                 let res: Result<String> = tokio::select!{
                     res = model.chat_with_image_spinner(prompt_clone, Some(dynamic_image), &app_handle_clone, "extraction-progress", json!({ 
                        "category": "Vision Analysis", "summary": "Analyzing image content..."
                    }), 1024, Some(cancellation_token.clone()), Some(task.id.clone())) => res,
                     _ = async {
                         loop {
                             if cancellation_token.load(Ordering::Relaxed) { break; }
                             tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                         }
                     } => { return Err(anyhow::anyhow!("Task cancelled")); }
                 };
                 res?
             } else {
                 "{}".to_string()
             };
             drop(model_guard); 
             
             let extracted_data = parse_json_from_llm(&result_str);
             
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let store_guard = store_mutex.lock().await;
             if let Some(db) = store_guard.as_ref() {
                 let id = extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task.id).to_string();
                 let _ = db.upsert_item("commerce_tracking", &id, "tracking", extracted_data.clone(), None).await;
             }
             
             let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
             }));
             
             return Ok(());
        } else {
             let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Error", "summary": "Failed to load image file.", "spinner": "❌"
             }));
             return Ok(());
        }
    }

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();

    if url.is_empty() {
        return Ok (());
    }

    let raw_html_content = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() {
            obj.remove("html");
        }
        content
    } else if !url.is_empty() {
        reqwest::get(&url).await?.text().await?
    } else {
        return Ok (());
    };
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html = parsing::pre_clean_html(&raw_html_content);
    let _ = std::fs::write("debug_scheduler_input.txt", &clean_html); // DEBUG
    drop(raw_html_content); 
    
    let light_pug = {
        let document = scraper::Html::parse_document(&clean_html);
        parsing::convert_doc_to_clean_pug(&document, PugMode::StructureOnly)
    }; 
    
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    
    let mut model_guard = model_mutex.lock().await;
    if model_guard.is_none() {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "Loading Model", "summary": "Loading AI Model...", "spinner": "⠋"
        }));
        match LogisModel::new(None).await {
            Ok(m) => *model_guard = Some(m),
            Err(e) => {
                let err_msg = format!("Model Init Error: {}", e);
                println!("[Scheduler] {}", err_msg);
                let _ = app_handle.emit("extraction-progress", json!({ 
                    "category": "Error", "summary": err_msg, "spinner": "❌"
                }));
                return Err(anyhow::anyhow!("Model initialization failed"));
            }
        }
    }
    
    if let Some(model) = model_guard.as_ref() {
        if model.is_cpu() {
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Warning", 
                "summary": "Low RAM. Running on CPU (Slow)...", 
                "spinner": "⚠️"
            }));
        }
    }

    let classify_chunks = chunk_text(&light_pug, 4000, 500);
    let classify_chunks_len = classify_chunks.len();
    
    if let Some(model) = model_guard.as_ref() {
        let app_handle_clone = app_handle.clone();

        for (i, chunk) in classify_chunks.iter().enumerate() {
            let _ = std::fs::write(format!("debug_scheduler_class_chunk_{}.txt", i), chunk); // DEBUG

            println!("[Classification-Loop] Ingesting structural part {}/{} to LLM (Session: {})", i + 1, classify_chunks_len, task.id);
            
            let prompt = if i == 0 {
                format!(
r#"[CONTEXT]
You are reading the structural skeleton (Pug/Jade) of a webpage to understand its type and layout.
I will provide the content in parts. Read each part and memorize it.

[PART 1]
{}

[INSTRUCTION]
Read and memorize. Do NOT generate JSON yet. Just say "ACKNOWLEDGED"."#, 
                    chunk
                )
            } else {
                format!(
r#"[PART {}]
{}

[INSTRUCTION]
Continue reading and memorizing. Just say "ACKNOWLEDGED"."#, 
                    i + 1, chunk
                )
            };
            
            let _ = std::fs::write(format!("debug_scheduler_class_prompt_{}.txt", i), &prompt); // DEBUG
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Classification Ingestion", 
                "summary": format!("Reading structure part {}/{}...", i + 1, classify_chunks_len),
                "spinner": "⠋"
            }));

            let res: Result<String> = tokio::select!{
                res = model.chat_with_spinner(
                    "You are a helpful assistant.", 
                    &prompt,
                    &app_handle_clone, 
                    "extraction-progress", 
                    json!({ "category": "Classification Ingestion" }), 
                    20, 
                    Some(cancellation_token.clone()), 
                    Some(task.id.clone())
                ) => res,
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
            };
            res?;
        }
    }

    println!("[Classification-FINAL] All parts ingested. Requesting FINAL category decision.");
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": "Identifying page type...", "spinner": "⠋"
    }));
    
    if cancellation_token.load(Ordering::Relaxed) { 
        drop(model_guard);
        cleanup_task_resources(&task.id);
        return Err(anyhow::anyhow!("Task cancelled")); 
    }

    let mut final_page_info = json!({});
    
    let page_type_res = if let Some(model) = model_guard.as_ref() {
        let system_text = parsing::page_type_prompt(language);
        let app_handle_clone = app_handle.clone();
        let prompt = "Now that you have read the full structure, analyze it and identify the primary category.";

        let res: Result<String> = tokio::select!{
            res = model.chat_with_spinner(
                &system_text, 
                prompt,
                &app_handle_clone, 
                "extraction-progress", 
                json!({ "category": "Classification", "summary": "Identifying page type..." }), 
                256, 
                Some(cancellation_token.clone()), 
                Some(task.id.clone())
            ) => res,
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
        };
        res?
    } else { "{}".to_string() };
    
    println!("[Scheduler] Checkpoint: Classification End");
    
    let type_info = parse_json_from_llm(&page_type_res);
    let page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
    
    if page_type.is_empty() || page_type == "unknown" {
        drop(model_guard);
        return Ok(());
    }

    final_page_info.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type), "spinner": "⠋"
    }));

    let page_selectors_res = if let Some(model) = model_guard.as_ref() {
        let next_question = parsing::page_selectors_prompt(&page_type, language);
        let app_handle_clone = app_handle.clone();
        
        let res: Result<String> = tokio::select!{
            res = model.chat_with_spinner(
                "You are a helpful assistant.", 
                &next_question,
                &app_handle_clone, 
                "extraction-progress", 
                json!({ 
                    "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type)
                }), 
                256, 
                Some(cancellation_token.clone()), 
                Some(task.id.clone())
            ) => res,
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
        };
        res?
    } else { "{}".to_string() };
    
    drop(model_guard);
    
    let selector_info = parse_json_from_llm(&page_selectors_res);
    if let Some(obj) = selector_info.as_object() {
        for (k, v) in obj {
            final_page_info.as_object_mut().unwrap().insert(k.clone(), v.clone());
        }
    }
    
    let page_info = final_page_info; 
    
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", 
        "summary": format!("Map: Type={}, Detail={}", 
            page_info.get("type").and_then(|s| s.as_str()).unwrap_or("?"),
            page_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false)
        ), 
        "spinner": "✅", 
        "data": page_info
    }));
    
    let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("");
    let is_detail = page_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
            let node_selector = page_info.get("node").and_then(|s| s.as_str()).unwrap_or("").trim();
            let item_selector = page_info.get("item").and_then(|s| s.as_str()).unwrap_or("").trim();
            
            let _ = std::fs::write("debug_scheduler_selector.txt", format!("Node: '{}'\nItem: '{}'", node_selector, item_selector)); // DEBUG
        
            if page_type == "" || page_type == "unknown" {            return Ok (());
        }
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
    
        let target_selector = if !node_selector.is_empty() { node_selector } else if !item_selector.is_empty() { item_selector } else { "body" };
        let mut extracted_data = json!({});
    
        if !is_detail {
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "List Processing", "summary": "Splitting list items...", "spinner": "⠋"
            }));
    
            let mut item_pugs = {
                let document = scraper::Html::parse_document(&clean_html);
                parsing::split_doc_to_pug_list(&document, target_selector, PugMode::FullContent)
            }; 
            
                    // Fallback: If selector failed, try 'body' to capture everything
            
                    if item_pugs.is_empty() {
            
                        println!("[Scheduler] List selector '{}' returned 0 items. Falling back to 'body'.", target_selector);
            
                        let _ = app_handle.emit("extraction-progress", json!({ 
            
                            "category": "Warning", "summary": format!("Selector '{}' failed. Using full body.", target_selector), "spinner": "⚠️"
            
                        }));
            
                        let document = scraper::Html::parse_document(&clean_html);
            
                        item_pugs = parsing::split_doc_to_pug_list(&document, "body", PugMode::FullContent);
            
                    }
            
                    
            
                    let _ = std::fs::write("debug_scheduler_pug_list.txt", item_pugs.join("\n---\n")); // DEBUG
            
                    
            
                    let total_items = item_pugs.len();
            
                    let _ = app_handle.emit("extraction-progress", json!({ 
            
                        "category": "List Processing", "summary": format!("Found {} items to extract.", total_items), "spinner": "✅"
            
                    }));
            
            
            
                            let fields_prompts = parsing::list2json(page_type, language);
                            let list_session_id = format!("{}_list", task.id);
                    
                            for (i, item_pug) in item_pugs.iter().enumerate() {
                                let is_last = i == total_items - 1;
                                println!("[List-Loop] Ingesting item part {}/{} to LLM (Session: {})", i + 1, total_items, list_session_id);
                                
                                let prompt = if is_last {
                                    println!("[List-FINAL] All items ingested. Requesting FINAL JSON list extraction.");
                                    let mut schema_definitions = String::new();
                                    for (name, schema) in &fields_prompts {
                                        schema_definitions.push_str(&format!("### Field: {}\nSchema/Definition: {}\n\n", name, schema));
                                    }
                    
                                    let final_prompt = format!(
                    r#"[FINAL PART]
                    {}
                    
                    [EXTRACTION SCHEMA]
                    Please extract all items and list-level metadata precisely according to these definitions:
                    {}
                    
                    [FINAL INSTRUCTION]
                    1. Based on ALL the parts I have sent you (previous and current), extract all items into an "items" array.
                    2. Include list-level metadata like "type", "item", "node", etc.
                    3. Return ONLY a single valid JSON object.
                    4. No preamble, no explanation. Just raw JSON."#,
                                        item_pug,
                                        schema_definitions
                                    );
                                    let _ = std::fs::write("debug_scheduler_prompt_final_list.txt", &final_prompt); // DEBUG
                                    final_prompt
                                } else {
                                    format!(
                    r#"[ITEM PART]
                    {}
                    
                    [INSTRUCTION]
                    Read and memorize this item. Do NOT generate JSON yet. 
                    Just say "ACKNOWLEDGED"."#,
                                        item_pug
                                    )
                                };
                    
                                let max_tokens = if is_last { 4096 } else { 20 };            
                        
            
                        let _ = app_handle.emit("extraction-progress", json!({ 
            
                            "category": "List Ingestion", 
            
                            "summary": format!("Reading Item {}/{}...", i + 1, total_items),
            
                            "spinner": "⠋"
            
                        }));
            
            
            
                        let model_guard = model_mutex.lock().await;
            
                        let res: Result<String> = if let Some(model) = model_guard.as_ref() {
            
                            let app_handle_clone = app_handle.clone();
            
                            tokio::select!{
            
                                res = model.chat_with_spinner(
            
                                    "You are a helpful assistant. Read the context carefully.", 
            
                                    &prompt,
            
                                    &app_handle_clone, 
            
                                    "extraction-progress", 
            
                                    json!({ "category": "List Ingestion" }), 
            
                                    max_tokens, 
            
                                    Some(cancellation_token.clone()), 
            
                                    Some(list_session_id.clone())
            
                                ) => res,
            
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
            
                        } else { Ok("{}".to_string()) };
            
                        drop(model_guard);
            
            
            
                        let response = res?;
            
            
            
                        if is_last {
            
                            let _ = std::fs::write("debug_scheduler_response_list.txt", &response); // DEBUG
            
                            println!("[EXTRACT-FINAL] Result: {}", response);
            
                            extracted_data = parse_json_from_llm(&response);
            
                        }
            
                    }
    } else {
        let mut content_pug = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::convert_doc_to_clean_pug_selector(&document, target_selector, PugMode::FullContent)
        }; 
        
        // Fallback: If selector failed, try 'body'
        if content_pug.trim().is_empty() {
            println!("[Scheduler] Detail selector '{}' returned empty. Falling back to 'body'.", target_selector);
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Warning", "summary": format!("Selector '{}' failed. Using full body.", target_selector), "spinner": "⚠️"
            }));
            let document = scraper::Html::parse_document(&clean_html);
            content_pug = parsing::convert_doc_to_clean_pug_selector(&document, "body", PugMode::FullContent);
        }
        
        if content_pug.trim().is_empty() {
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Error", "summary": format!("Selector '{}' not found.", target_selector), "spinner": "❌"
            }));
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Done", "summary": "Extraction Failed", "spinner": "🛑", "data": null
            }));
            return Ok(());
        }

        let _ = std::fs::write("debug_content_pug.txt", &content_pug); 
        let _ = std::fs::write("debug_scheduler_pug_detail.txt", &content_pug); // DEBUG
        
        let chunks = chunk_text(&content_pug, 3000, 500); 
        let fields_prompts = parsing::item2json(page_type, &url, language);
        let detail_session_id = format!("{}_detail", task.id);
        let chunks_len = chunks.len();
        
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let is_last = chunk_idx == chunks_len - 1;
            println!("[Detail-Loop] Ingesting content part {}/{} to LLM (Session: {})", chunk_idx + 1, chunks_len, detail_session_id);
            
            let prompt = if is_last {
                println!("[Detail-FINAL] All content ingested. Requesting FINAL JSON extraction.");
                let mut schema_definitions = String::new();
                for (name, schema) in &fields_prompts {
                    schema_definitions.push_str(&format!("### Field: {}\nSchema/Definition: {}\n\n", name, schema));
                }

                let final_prompt = format!(
r#"[FINAL PART]
{}

[EXTRACTION SCHEMA]
Please extract the following data points precisely according to these definitions:
{}

[FINAL INSTRUCTION]
1. Based on ALL the parts I have sent you (previous and current), extract all fields mentioned in the EXTRACTION SCHEMA.
2. Return ONLY a single valid JSON object.
3. No preamble, no explanation, no markdown code blocks. Just the raw JSON.
4. If a value is missing, use null."#,
                    chunk,
                    schema_definitions
                );
                let _ = std::fs::write("debug_scheduler_prompt_final_detail.txt", &final_prompt); // DEBUG
                final_prompt
            } else {
                format!(
r#"[CONTENT PART]
{}

[INSTRUCTION]
Read and memorize this content. Do NOT generate JSON yet. 
When all parts are sent, I will ask you to extract data into JSON.
Just say "ACKNOWLEDGED"."#,
                    chunk
                )
            };

            let max_tokens = if is_last { 2048 } else { 20 };
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Ingestion", 
                "summary": format!("Reading part {}/{}...", chunk_idx + 1, chunks_len),
                "spinner": "⠋"
            }));

            let model_guard = model_mutex.lock().await;
            let res: Result<String> = if let Some(model) = model_guard.as_ref() {
                let app_handle_clone = app_handle.clone();
                tokio::select!{
                    res = model.chat_with_spinner(
                        "You are a helpful assistant. Read the context carefully.", 
                        &prompt,
                        &app_handle_clone, 
                        "extraction-progress", 
                        json!({ "category": "Ingestion" }), 
                        max_tokens, 
                        Some(cancellation_token.clone()), 
                        Some(detail_session_id.clone())
                    ) => res,
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
            } else { Ok("{}".to_string()) };
            drop(model_guard);

            let response = res?;

            if is_last {
                let _ = std::fs::write("debug_scheduler_response_detail.txt", &response); // DEBUG
                println!("[EXTRACT-FINAL] Result: {}", response);
                let full_data = parse_json_from_llm(&response);
                
                if let Some(obj) = extracted_data.as_object_mut() {
                    if let Some(source_obj) = full_data.as_object() {
                        for (k, v) in source_obj {
                            obj.insert(k.clone(), v.clone());
                        }
                    }
                } else {
                    extracted_data = full_data;
                }
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
    
    let id_opt = normalized_data.get("id").cloned();
    let index_opt = normalized_data.get("index").cloned();

    if let Some(id_val) = id_opt {
        if normalized_data.get("index").is_none() { normalized_data.as_object_mut().unwrap().insert("index".to_string(), id_val); }
    } else if let Some(idx_val) = index_opt {
        if normalized_data.get("id").is_none() { normalized_data.as_object_mut().unwrap().insert("id".to_string(), idx_val); }
    }
    
    extracted_data = normalized_data;

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Saving", "summary": "Saving to database...", "spinner": "⠋"
    }));

    if let Some((queries, merge_info)) = logic::relay(page_type, &extracted_data) {
        println!("[Scheduler] Relay logic triggered. Queries: {}", queries.len());
        
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let mut target_data = extracted_data.clone();
            let mut target_id = format!("{}_{}", page_type, chrono::Utc::now().timestamp_millis());
            let mut found_existing = false;
            let to_table = format!("commerce_{}", merge_info.to);

            for query in queries {
                let query_table = format!("commerce_{}", query.table);
                if let Ok(Some((id, existing_data))) = db.find_item_by_property(&query_table, &query.column, &query.value).await {
                    println!("[Scheduler] Found existing item: {} in {}", id, query_table);
                    logic::merge_node(&mut target_data, &existing_data);
                    target_id = id;
                    found_existing = true;
                    break;
                }
            }
            
            if !found_existing {
                if let Some(id_val) = target_data.get("id").and_then(|v| v.as_str()) {
                    target_id = id_val.to_string();
                }
            }

            let text_to_embed = target_data.to_string();
            drop(store_guard); 
            
            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
            
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            println!("[Scheduler] Checkpoint: Embedding Start (Relay)");
            let vector = if let Some(model) = model_guard.as_ref() {
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
                let _ = db.upsert_item(&to_table, &target_id, page_type, target_data, vector).await;
            }
        }
    } else {
        let target_table = format!("commerce_{}", page_type);
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
                let id = extracted_data.get("id").and_then(|s| s.as_str()).unwrap_or_else(|| task.id.as_str()).to_string();
                let text_to_embed = extracted_data.to_string();
                drop(store_guard);

                let mut model_guard = model_mutex.lock().await;
                if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
                
                println!("[Scheduler] Checkpoint: Embedding Start (Direct)");
                let vector = if let Some(model) = model_guard.as_ref() {
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
                    let _ = db.upsert_item(&target_table, &id, page_type, extracted_data.clone(), vector).await;
                }
        }
    }    
    
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Done", "summary": "Extraction Complete", "spinner": "✅", "data": extracted_data
    }));

    // Cleanup all KV Cache subdirectories for this task
    cleanup_task_resources(&task.id);

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

fn parse_json_from_llm(text: &str) -> Value {
    if let Ok(v) = serde_json::from_str(text) { return v; }
    if let Some(start) = text.find("{") {
        if let Some(end) = text.rfind("}") {
            if start < end {
                if let Ok(v) = serde_json::from_str(&text[start..=end]) { return v; }
            }
        }
    }
    if let Some(start) = text.find("[") {
        if let Some(end) = text.rfind("]") {
            if start < end {
                if let Ok(v) = serde_json::from_str(&text[start..=end]) { return v; }
            }
        }
    }
    json!({})
}
