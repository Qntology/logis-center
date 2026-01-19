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
    
    // Chunking Logic for Classification
    // Instead of truncating, we ingest the whole light_pug structure
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

    // Ingest chunks for Classification
    if let Some(model) = model_guard.as_ref() {
        let app_handle_clone = app_handle.clone();
        for (i, chunk) in classify_chunks.iter().enumerate() {
            let is_last = i == classify_chunks_len - 1;
            let system_text = parsing::page_type_prompt(language); // Consistent system text
            
            // Branch prompt based on whether this is the last chunk
            let (prompt, max_tokens) = if is_last {
                (
                    format!(
r#"[FINAL CONTEXT]
This is the last part of the page structure.
Part {}/{}.

[INPUT SNIPPET]
{}

[INSTRUCTION]
Analyze the full structure you have read so far and identify the primary category.
Return the result in the specified JSON format."#, 
                        i + 1, classify_chunks_len, chunk
                    ),
                    256 // More tokens for the JSON response
                )
            } else {
                (
                    format!(
r#"[CONTEXT]
Reading Part {}/{} of the layout.

[INPUT SNIPPET]
{}

[INSTRUCTION]
Memorize this structure. Summarize in 3 keywords and say "READY"."#,
                        i + 1, classify_chunks_len, chunk
                    ),
                    32 // Minimal tokens for ACKs
                )
            };
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Classification Ingestion", 
                "summary": format!("Reading structure part {}/{}...", i + 1, classify_chunks_len),
                "spinner": "⠋"
            }));

            tokio::select!{
                res = model.chat_with_spinner(
                    &system_text, 
                    &prompt,
                    &app_handle_clone, 
                    "extraction-progress", 
                    json!({ 
                        "category": "Classification Ingestion",
                        "summary": if is_last { "Identifying page type...".to_string() } else { format!("Reading structure part {}/{}...", i + 1, classify_chunks_len) }
                    }), 
                    max_tokens,
                    Some(cancellation_token.clone()), 
                    Some(task.id.clone())
                ) => { 
                    let out = res?;
                    if is_last {
                        page_type_res = out;
                    } else {
                        println!("[Ingestion Log] Part {}: {}", i + 1, out);
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
    if page_type_res.is_empty() {
        return Ok(());
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut final_page_info = json!({});
    
    println!("[Scheduler] Checkpoint: Classification End");
    
    let type_info = parse_json_from_llm(&page_type_res);
    let page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
    println!("[Scheduler] Classification Result: '{}'", page_type);
    
    if page_type.is_empty() || page_type == "unknown" {
        println!("[Scheduler] Stopping: Unknown page type.");
        drop(model_guard);
        return Ok(());
    }

    final_page_info.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    // Step 2: Identify Selectors (Building on History)
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type), "spinner": "⠋"
    }));

    let page_selectors_res = if let Some(model) = model_guard.as_ref() {
        let system_prompt = parsing::page_type_prompt(language);
        let next_question = parsing::page_selectors_prompt(&page_type, language); 
        let app_handle_clone = app_handle.clone();
        
        tokio::select!{
            res = model.chat_with_spinner(
                &system_prompt, 
                &next_question,
                &app_handle_clone, 
                "extraction-progress", 
                json!({ 
                    "category": "Classification", "summary": format!("Type: {}. Finding selectors...", page_type)
                }), 
                256, 
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

    let target_selector = if !node_selector.is_empty() { node_selector } else if !item_selector.is_empty() { item_selector } else { "body" };
    let mut extracted_data = json!({});

    if !is_detail {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "List Processing", "summary": "Splitting list items...", "spinner": "⠋"
        }));

        let item_pugs = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::split_doc_to_pug_list(&document, target_selector, PugMode::FullContent)
        }; 
        
        let total_items = item_pugs.len();
        println!("[Scheduler] List Processing: Found {} items using selector '{}'", total_items, target_selector);
        
        if total_items == 0 {
            println!("[Scheduler] Stopping: No items found with the specified selector.");
            return Ok(());
        }

        let fields_prompts = parsing::list2json(page_type, language);
        let list_session_id = format!("{}_list", task.id);
        let mut full_context_accumulated = String::new();

        for (i, item_pug) in item_pugs.iter().enumerate() {
            let is_last = i == total_items - 1;
            full_context_accumulated.push_str(&format!("--- ITEM {} ---\n", i + 1));
            full_context_accumulated.push_str(item_pug);
            full_context_accumulated.push('\n');

            let prompt = if is_last {
                let mut schema_definitions = String::new();
                for (name, schema) in &fields_prompts {
                    schema_definitions.push_str(&format!("### Field: {}\nSchema/Definition: {}\n\n", name, schema));
                }

                format!(
r#"[CONTEXT]
You have been reading a list of items from a webpage part by part. 
Below is the COMPLETELY ACCUMULATED list of Pug (Jade) snippets.

[FULL LIST CONTENT]
{}

[EXTRACTION SCHEMA]
Please extract all items and list-level metadata precisely according to these definitions:
{}

[FINAL INSTRUCTION]
1. Based on the FULL LIST CONTENT, extract all items into an "items" array.
2. Include list-level metadata like "type", "item", "node", etc.
3. Return ONLY a single valid JSON object.
4. No preamble, no explanation. Just raw JSON."#,
                    full_context_accumulated,
                    schema_definitions
                )
            } else {
                format!(
r#"[CONTEXT]
You are reading a list of items from a webpage.
Here is the accumulated item snippets so far:

[INPUT SNIPPET]
{}

[INSTRUCTION]
Read and memorize these items. Do NOT generate JSON yet.
Just say "ACKNOWLEDGED"."#,
                    full_context_accumulated
                )
            };

            let max_tokens = if is_last { 4096 } else { 20 };
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "List Ingestion", 
                "summary": format!("Reading Item {}/{}...", i + 1, total_items),
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
                        json!({ "category": "List Ingestion" }), 
                        max_tokens, 
                        Some(cancellation_token.clone()), 
                        Some(list_session_id.clone())
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
            } else { "{}".to_string() };
            drop(model_guard);

            if is_last {
                extracted_data = parse_json_from_llm(&response);
            }
        }
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
        
        // CHUNKING: Split huge content into 3000-char chunks with 500-char overlap (CPU Optimized)
        let chunks = chunk_text(&content_pug, 3000, 500); 
        let fields_prompts = parsing::item2json(page_type, &url, language);

        // [NEW] Use a single session ID for the entire detail page to maintain structural context (table headers, etc.)
        let detail_session_id = format!("{}_detail", task.id);
        
        // Phase 1: Ingest all chunks (Prefill)
        // We feed chunks one by one to build up the KV cache context.
        // We only ask for the result at the very end.
        let chunks_len = chunks.len();
        let mut full_context_accumulated = String::new(); // Accumulate context here
        
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let is_last = chunk_idx == chunks_len - 1;
            
            // Append current chunk to accumulator
            // Note: We don't just append the raw chunk, we append the *formatted* prompt content if possible,
            // but since the prompt template changes (Instruction wrapper), simply appending raw text might confuse the model if we re-send full prompt.
            // BETTER STRATEGY:
            // The KV cache stores the *tokens* of the previous request. 
            // If we send [Prompt_A + Chunk_1], cache has Tokens(Prompt_A + Chunk_1).
            // Next request: [Prompt_A + Chunk_1 + Chunk_2].
            // Cache Hit: Tokens(Prompt_A + Chunk_1). New input: Chunk_2.
            
            // So we MUST accumulate the content in the prompt.
            full_context_accumulated.push_str(chunk);
            full_context_accumulated.push('\n');

            // Prepare prompt
            let prompt = if is_last {
                // Final Chunk: Include the FULL SCHEMA for all fields
                let mut schema_definitions = String::new();
                for (name, schema) in &fields_prompts {
                    schema_definitions.push_str(&format!("### Field: {}\nSchema/Definition: {}\n\n", name, schema));
                }

                format!(
r#"[CONTEXT]
You have been reading a Pug (Jade) template part by part. 
Below is the COMPLETELY ACCUMULATED content of the template.

[FULL CONTENT]
{}

[EXTRACTION SCHEMA]
Please extract the following data points precisely according to these definitions:
{}

[FINAL INSTRUCTION]
1. Based on the FULL CONTENT, extract all fields mentioned in the EXTRACTION SCHEMA.
2. Return ONLY a single valid JSON object.
3. No preamble, no explanation, no markdown code blocks. Just the raw JSON.
4. If a value is missing, use null."#,
                    full_context_accumulated,
                    schema_definitions
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
Just say "ACKNOWLEDGED"."#,
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
        if let Some(_db) = store_guard.as_ref() {
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
