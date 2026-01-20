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
                             break; // Stop processing pending tasks batch to prevent immediate model reload
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
) -> Result<()>
{
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"
    }));

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    // --- Image Extraction Logic ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        
        let mut model_guard = model_mutex.lock().await;
        if model_guard.is_none() {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
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
    drop(raw_html_content); 
    
    let light_pug = {
        let document = scraper::Html::parse_document(&clean_html);
        parsing::convert_doc_to_clean_pug(&document, PugMode::StructureOnly)
    }; 
    
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    
    // Determine optimal chunk sizes based on hardware
    // [ADAPTIVE] Reduced size to account for multilingual token overhead (Korean/Chinese/etc.)
    let mut device_config = utils::get_optimal_device_config();
    device_config.classify_chunk_size = 4000; // Lowered from 7000 to avoid token explosion
    device_config.extract_chunk_size = 4000;

    // [REVISED] Restore turn-based chunked ingestion for classification.
    // This is the original stable logic.
    let classify_chunks = chunk_text(&light_pug, 4000, 500);
    let classify_chunks_len = classify_chunks.len();

    let mut model_guard = model_mutex.lock().await;
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
            content: parsing::page_type_prompt(language),
            name: None,
        })
    ];

    if let Some(model) = model_guard.as_ref() {
        let app_handle_clone = app_handle.clone();
        
        for (i, chunk) in classify_chunks.iter().enumerate() {
            let is_last = i == classify_chunks_len - 1;
            
            let prompt = if is_last {
                format!("[DATA_PART]\n{}\n\n[INSTRUCTION]\nEnd of document. Identify the primary category and return JSON.", chunk)
            } else {
                format!("[DATA_PART]\n{}\n\n[INSTRUCTION]\nRead and say READY.", chunk)
            };

            // Add current chunk to history
            messages.push(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: prompt })
                ]),
                name: None,
            }));
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Classification Ingestion", 
                "summary": format!("Reading structure part {}/{}...", i + 1, classify_chunks_len),
                "spinner": "⠋"
            }));

            let params = ChatCompletionParameters {
                messages: messages.clone(), // Send full history
                model: "qwen3vl".to_string(),
                max_tokens: Some(if is_last { 256 } else { 32 }),
                temperature: Some(0.1),
                ..Default::default()
            };

            tokio::select!{
                res = model.chat_params_with_spinner(
                    params,
                    &app_handle_clone, 
                    "extraction-progress", 
                    json!({ 
                        "category": "Classification Ingestion",
                        "summary": if is_last { "Identifying page type...".to_string() } else { format!("Reading structure part {}/{}...", i + 1, classify_chunks_len) }
                    }), 
                    Some(cancellation_token.clone()), 
                    Some(task.id.clone())
                ) => { 
                    let out = res?;
                    if is_last {
                        page_type_res = out;
                    } else {
                        // Add model's ACK to history to keep it in sync with cache
                        messages.push(ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                            content: Some(out),
                            ..Default::default()
                        }));
                    }
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
    
    let type_info = parse_json_from_llm(&page_type_res);
    let page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
    println!("[Scheduler] Classification Result: '{}'", page_type);
    
    if page_type.is_empty() || page_type == "unknown" {
        println!("[Scheduler] Stopping: Unknown page type. Raw response: '{}'", page_type_res);
        drop(model_guard);
        return Ok(());
    }

    final_page_info.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    // Step 2: Identify Selectors (Building on History)
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type), "spinner": "⠋"
    }));

    let page_selectors_res = if let Some(model) = model_guard.as_ref() {
        let next_question = parsing::page_selectors_prompt(&page_type, language); 
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
    
    let selector_info = parse_json_from_llm(&page_selectors_res);
    println!("[Scheduler] Selectors Found: {}", selector_info);
    // Merge selectors into final_page_info
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
            page_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false)
        ), 
        "spinner": "✅", 
        "data": page_info
    }));
    
    let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("");
    let is_detail = page_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
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
        
        let mut item_pugs = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::split_doc_to_pug_list(&document, &current_selector, PugMode::FullContent)
        }; 

        // FALLBACK LOGIC: If primary selector fails, try the alternative
        if item_pugs.is_empty() && current_selector != "body" {
            let fallback = if current_selector == item_selector && !node_selector.is_empty() { node_selector.to_string() } else { "body".to_string() };
            if fallback != current_selector {
                println!("[Scheduler] Selector '{}' failed, trying fallback '{}'", current_selector, fallback);
                current_selector = fallback;
                item_pugs = {
                    let document = scraper::Html::parse_document(&clean_html);
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

            // Combine multiple pugs into one prompt
            let mut combined_pugs = String::new();
            for (idx, pug) in chunk.iter().enumerate() {
                combined_pugs.push_str(&format!("\n### ITEM #{} ###\n{}\n", idx + 1, pug));
            }

            let prompt = format!(
r#"{instruction}

[INPUT SNIPPETS]
{data}"#,
                instruction = extraction_instruction,
                data = combined_pugs
            );

            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
            
            let response = if let Some(model) = model_guard.as_ref() {
                let app_handle_clone = app_handle.clone();
                let summary_text = format!("Processing {}~{}/{}...", current_start, current_end, total_items);
                tokio::select!{
                    res = model.chat_with_spinner(
                        "", 
                        &prompt,
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

            let batch_results = parse_json_from_llm(&response);
            if let Some(arr) = batch_results.as_array() {
                for item_json in arr {
                    if !item_json.is_null() && item_json.is_object() {
                        all_extracted_items.push(item_json.clone());
                    }
                }
            } else if batch_results.is_object() {
                // Fallback for single object response
                all_extracted_items.push(batch_results.clone());
            }

            // [NEW] Emit intermediate results for this batch immediately
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Item Extraction", 
                "summary": format!("Extracted {}/{} items...", all_extracted_items.len(), total_items),
                "data": batch_results, // Current batch data
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
            let document = scraper::Html::parse_document(&clean_html);
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
        
        // CHUNKING: Split huge content into dynamic chunks with 500-char overlap (Device Optimized)
        let chunks = chunk_text(&content_pug, device_config.extract_chunk_size, 500); 
        let extraction_instruction = parsing::item2json(page_type, &url, language);

        // [NEW] Use a single session ID for the entire detail page to maintain structural context (table headers, etc.)
        let detail_session_id = format!("{}_detail", task.id);
        
        // Phase 1: Ingest all chunks (Prefill)
        let chunks_len = chunks.len();
        let mut full_context_accumulated = String::new(); // Accumulate context here
        
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let is_last = chunk_idx == chunks_len - 1;
            full_context_accumulated.push_str(chunk);
            full_context_accumulated.push('\n');

            // Prepare prompt
            let prompt = if is_last {
                format!(
r#"{instruction}

[INPUT CONTENT]
{data}"#,
                    instruction = extraction_instruction,
                    data = full_context_accumulated
                )
            } else {
                // Intermediate Chunk: Just read and remember
                format!(
r#"[CONTEXT]
You are reading a Pug (Jade) template. 
Here is the accumulated content so far:

[INPUT SNIPPET]
{}

[INSTRUCTION]
Read and memorize this content. Do NOT generate JSON yet. 
When all parts are sent, I will ask you to extract data into JSON.
Just say "READY"."#,
                    full_context_accumulated
                )
            };

            let max_tokens = if is_last { 2048 } else { 20 }; // Minimal tokens for acknowledgment
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Ingestion", 
                "summary": format!("Reading part {}/{} ({} chars)...", chunk_idx + 1, chunks_len, full_context_accumulated.len()),
                "spinner": "⠋"
            }));

            let model_guard = model_mutex.lock().await;
            
            let response = if let Some(model) = model_guard.as_ref() {
                let app_handle_clone = app_handle.clone();
                tokio::select!{
                    res = model.chat_with_spinner(
                        "", 
                        &prompt,
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
                let full_data = parse_json_from_llm(&response);
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

            let mut text_to_embed = json_to_natural_language(&target_data);
            
            // [FIX] Ensure text doesn't exceed embedding model limit (~2048 tokens)
            // Conservative estimate: 1 char ~= 0.5 to 1 token in multilingual context
            if text_to_embed.chars().count() > 3000 {
                text_to_embed = text_to_embed.chars().take(3000).collect();
            }
            
            drop(store_guard); 
            
            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() { if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); } }
            
            if cancellation_token.load(Ordering::Relaxed) { 
                return Err(anyhow::anyhow!("Task cancelled")); 
            }

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
                    } => { 
                        return Err(anyhow::anyhow!("Task cancelled")); 
                    }
                }
            } else { None };
            drop(model_guard);
            
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                println!("[Scheduler] Saving item: {} to {}", target_id, to_table);
                let _ = db.upsert_item(&to_table, &target_id, page_type, target_data.clone(), vector.clone()).await;
                // [GLOBAL-SEARCH-FIX] Also save to 'commerce_items' for global search visibility
                let _ = db.upsert_item("commerce_items", &target_id, page_type, target_data, vector).await;
            }
        }
    } else {
        let target_table = format!("commerce_{}", page_type);
        let store_guard = store_mutex.lock().await;
        if let Some(_db) = store_guard.as_ref() {
                let id = extracted_data.get("id").and_then(|s| s.as_str()).unwrap_or_else(|| task.id.as_str()).to_string();
                let mut text_to_embed = json_to_natural_language(&extracted_data);
                
                if text_to_embed.chars().count() > 3000 {
                    text_to_embed = text_to_embed.chars().take(3000).collect();
                }
                
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
                        } => { 
                            return Err(anyhow::anyhow!("Task cancelled")); 
                        }
                    }
                } else { None };
                drop(model_guard);

                let store_guard = store_mutex.lock().await;
                if let Some(db) = store_guard.as_ref() {
                    let _ = db.upsert_item(&target_table, &id, page_type, extracted_data.clone(), vector.clone()).await;
                    // [GLOBAL-SEARCH-FIX] Also save to 'commerce_items' for global search visibility
                    let _ = db.upsert_item("commerce_items", &id, page_type, extracted_data.clone(), vector).await;
                }
        }
    }    
    
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Done", "summary": "Extraction Complete", "spinner": "✅", "data": extracted_data
    }));

    // Cleanup all KV Cache subdirectories for this task
    cleanup_task_resources(&task.id);

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

pub fn parse_json_from_llm(text: &str) -> Value {
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

/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(value: &Value) -> String {
    let mut output = String::new();
    
    // Recursive handling for nested structures like { "value": "..." }
    if let Some(obj) = value.as_object() {
        if obj.len() == 1 && obj.contains_key("value") {
            return obj.get("value").unwrap().as_str().unwrap_or(&obj.get("value").unwrap().to_string()).to_string();
        }
    }

    if let Value::Object(map) = value {
        let page_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let is_detail = map.get("detail").and_then(|v| v.as_bool()).unwrap_or(true);

        // Define EXACT columns from parsing.rs
        let keys: Vec<&str> = match page_type {
            "tracking" => {
                if is_detail {
                    vec!["status", "id", "title", "sender_name", "sender_address", "sender_phone", "recipient_name", "recipient_address", "recipient_phone", "package_width", "package_height", "package_length", "package_weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "shipping_date", "registration_date"]
                } else {
                    vec!["status", "id", "title", "link", "registration_date"]
                }
            },
            "goods" => {
                if is_detail {
                    vec!["code", "link", "id", "status", "payment_method", "bank", "card", "model_name", "brand_name", "condition", "description", "short_description", "tags", "origin_country", "manufacturer", "release_date", "manufacture_date", "expiration_date", "gtin", "mpn", "barcode", "sale_price", "supply_price", "currency", "compare_at_price", "quantity", "stock_keeping_unit", "low_stock_threshold", "unit", "tax_included", "tax_code", "main_image_url", "additional_image_url", "video_url", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "product_width", "product_height", "product_length", "product_weight", "options", "additional_goods", "title", "registration_date"]
                } else {
                    vec!["status", "link", "id", "title", "sale_price", "supply_price", "currency", "quantity", "tracking_number", "registration_date"]
                }
            },
            "order" => {
                if is_detail {
                    vec!["link", "id", "tracking_number", "status", "goods", "sender_name", "sender_address", "sender_phone", "recipient_name", "recipient_address", "recipient_phone", "bank", "card", "order_date", "payment_date", "payment_method", "payment_origin", "registration_date"]
                } else {
                    vec!["status", "link", "id", "title", "sale_price", "supply_price", "currency", "quantity", "tracking_number", "registration_date"]
                }
            },
            "coupon" | "event" => {
                if is_detail {
                    vec!["link", "id", "type", "status", "title", "started_at", "expired_at", "code", "discount", "quantity", "usage_limit", "usage_per", "new_customer_only", "min_order_amount", "max_discount_amount", "region_restrictions", "registration_date"]
                } else {
                    vec!["status", "id", "title", "started_at", "expired_at", "registration_date"]
                }
            },
            "review" => {
                if is_detail {
                    vec!["link", "id", "status", "name", "title", "completed", "registration_date"]
                } else {
                    vec!["status", "id", "title", "link", "registration_date"]
                }
            },
            _ => map.keys().map(|s| s.as_str()).collect()
        };

        for key in keys {
            if let Some(v) = map.get(key) {
                if v.is_null() { continue; }
                let key_name = key.replace("_", " ");
                if v.is_array() {
                    let arr = v.as_array().unwrap();
                    let mut items = Vec::new();
                    for item in arr.iter().take(5) {
                        let sub = json_to_natural_language(item);
                        if !sub.is_empty() { items.push(sub); }
                    }
                    if !items.is_empty() {
                        output.push_str(&format!("{}: [{}]. ", key_name, items.join(", ")));
                    }
                } else if v.is_object() {
                    let sub = json_to_natural_language(v);
                    if !sub.is_empty() {
                        output.push_str(&format!("{}: {}. ", key_name, sub));
                    }
                } else {
                    let s = match v {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        _ => String::new(),
                    };
                    if !s.is_empty() && s != "null" {
                        let s_clean = if s.len() > 400 { format!("{}...", &s[..400]) } else { s };
                        output.push_str(&format!("{}: {}. ", key_name, s_clean));
                    }
                }
            }
        }
    } else if let Value::Array(arr) = value {
        for item in arr.iter().take(10) {
            let sub = json_to_natural_language(item);
            if !sub.is_empty() {
                output.push_str(&sub);
                output.push_str(" ");
            }
        }
    } else {
        output.push_str(&value.as_str().unwrap_or(&value.to_string()));
    }
    
    output.trim().to_string()
}
