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

// Helper to chunk text with overlap, strictly respecting newlines for Pug
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let target_end = (start + chunk_size).min(text.len());
        let mut end = target_end;
        
        // Find the NEXT newline after target_end to include the current line fully
        if target_end < text.len() {
            if let Some(next_newline) = text[target_end..].find('\n') {
                // Include the full line by moving end to the newline position
                end = target_end + next_newline + 1;
            } else {
                // If no more newlines are found, take the rest of the text
                end = text.len();
            }
        }
        
        chunks.push(text[start..end].to_string());
        
        if end >= text.len() {
            break;
        }
        
        // Calculate next start with overlap
        let mut next_start = end.saturating_sub(overlap);
        
        // Align next_start to the START of a line (find the first newline in the overlap zone)
        if next_start > start && next_start < end {
             if let Some(line_start) = text[next_start..end].find('\n') {
                 next_start = next_start + line_start + 1;
             }
        }
        
        // Prevent infinite loops or zero progress
        if next_start >= end {
            next_start = end; 
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
                         let _ = app_handle.emit("extraction-progress", json!({ 
                            "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                         }));
                         return Ok(());
                     }
                 }
             }
             
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let result_str = if let Some(model) = model_guard.as_ref() {
                 let prompt_clone = prompt.clone();
                 let app_handle_clone = app_handle.clone();
                 
                 tokio::select!{
                     res = model.chat_with_image_spinner(prompt_clone, Some(dynamic_image), &app_handle_clone, "extraction-progress", json!({ 
                        "category": "Vision Analysis", "summary": "Analyzing image content..."
                    }), 1024) => res?,
                     _ = async {
                         loop {
                             if cancellation_token.load(Ordering::Relaxed) { break; }
                             tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                         }
                     } => { return Err(anyhow::anyhow!("Task cancelled")); }
                 }
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
    drop(raw_html_content); 
    
    let light_pug = {
        let document = scraper::Html::parse_document(&clean_html);
        parsing::convert_doc_to_clean_pug(&document, PugMode::StructureOnly)
    }; 
    
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    
    // Chunking Logic for Classification
    let classify_input = if light_pug.len() > 8000 {
        light_pug[..8000].to_string()
    } else {
        light_pug.clone()
    };

    let mut model_guard = model_mutex.lock().await;
    
    if model_guard.is_none() {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "Loading Model", "summary": "Loading AI Model...", "spinner": "⠋"
        }));
        match LogisModel::new(None).await {
            Ok(m) => *model_guard = Some(m),
            Err(e) => {
                println!("[Scheduler] Model Init Error: {:?}", e);
                return Ok(());
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

    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Classification", "summary": "Analyzing page structure...", "spinner": "⠋"
    }));
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let page_info_str = if let Some(model) = model_guard.as_ref() {
        let system = parsing::map_outline(language);
        let app_handle_clone = app_handle.clone();

        tokio::select!{
            res = model.chat_with_spinner(&system, &classify_input, &app_handle_clone, "extraction-progress", json!({ 
                "category": "Classification", "summary": "Analyzing page structure..."
            }), 512) => res?,
            _ = async {
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => { return Err(anyhow::anyhow!("Task cancelled")); }
        }
    } else {
        "{}".to_string()
    };
    drop(model_guard);
    
    let page_info: Value = parse_json_from_llm(&page_info_str);
    
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
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "List Processing", "summary": format!("Found {} items to extract.", total_items), "spinner": "✅"
        }));

        let mut items_array = Vec::new();
        let fields_prompts = parsing::item2json(page_type, &url, language);

        for (i, item_pug) in item_pugs.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            // Chunking for List Items
            let chunks = chunk_text(item_pug, 4000, 200); 
            let mut item_data = json!({});
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Extraction", "summary": format!("Extracting Item {}/{}: ...", i + 1, total_items), "spinner": "⠋"
            }));

            let model_guard = model_mutex.lock().await; 

            for (field_name, field_schema) in &fields_prompts {
                if field_name == "description" || field_name == "detail_images" { continue; }
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                for (chunk_idx, chunk) in chunks.iter().enumerate() {
                    let field_json_str = if let Some(model) = model_guard.as_ref() {
                        let prompt = format!(
r#"[CONTEXT]
This is PART {}/{} of a Pug (Jade) template snippet representing a list item on a webpage.
The snippet may start or end in the middle of a tag.

[SCHEMA]
Field: "{}"
Definition: {}

[INPUT SNIPPET]
{}

[INSTRUCTION]
1. Analyze the SNIPPET to find the value for '{}'.
2. Only extract data clearly visible in this snippet.
3. Ignore broken tags/syntax at boundaries.
4. Return valid JSON only: {{ "{}": value }}
5. If not found, return empty JSON {{}}."#,
                            chunk_idx + 1, chunks.len(),
                            field_name, field_schema,
                            chunk,
                            field_name,
                            field_name
                        );

                        let app_handle_clone = app_handle.clone();

                        tokio::select!{
                            res = model.chat_with_spinner("You are a strict data extraction parser. Return valid JSON only.", 
                                &prompt,
                                &app_handle_clone, "extraction-progress", json!({ 
                                    "category": "Extraction", 
                                    "summary": format!("Item {}/{}: {}", i + 1, total_items, field_name)
                                }), 
                                512 
                            ) => res?,
                            _ = async {
                                loop {
                                    if cancellation_token.load(Ordering::Relaxed) { break; }
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                }
                            } => { return Err(anyhow::anyhow!("Task cancelled")); }
                        }
                    } else {
                        "{}".to_string()
                    };
                    
                    let field_result = parse_json_from_llm(&field_json_str);
                    if let Some(val) = field_result.get(field_name) {
                        if !val.is_null() {
                             item_data.as_object_mut().unwrap().insert(field_name.clone(), val.clone());
                        }
                    } else if !field_result.as_object().unwrap().is_empty() {
                         item_data.as_object_mut().unwrap().insert(field_name.clone(), field_result.clone());
                    }
                }
            }
            drop(model_guard); 

            if let Some(id_val) = item_data.get("id").cloned() {
                 if item_data.get("index").is_none() { item_data.as_object_mut().unwrap().insert("index".to_string(), id_val); }
            }
            items_array.push(item_data.clone());
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Extraction", "summary": format!("Item {} extracted.", i + 1), "data": item_data
            }));
        }
        
        extracted_data.as_object_mut().unwrap().insert("items".to_string(), json!(items_array));
        extracted_data.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    } else {
        let content_pug = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::convert_doc_to_clean_pug_selector(&document, target_selector, PugMode::FullContent)
        }; 
        
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
        
        // CHUNKING: Split huge content into 3000-char chunks with 500-char overlap (CPU Optimized)
        let chunks = chunk_text(&content_pug, 3000, 500); 
        let fields_prompts = parsing::item2json(page_type, &url, language);
        
        for (field_name, field_schema) in fields_prompts {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Extraction", "summary": format!("Extracting field: {}", field_name), "spinner": "⠋"
            }));

            let model_guard = model_mutex.lock().await;
            
            // Iterate over chunks and merge results
            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                let field_json_str = if let Some(model) = model_guard.as_ref() {
                    let prompt = format!(
r#"[CONTEXT]
This is PART {}/{} of a Pug (Jade) template snippet representing a webpage.
The snippet may start or end in the middle of a tag.

[SCHEMA]
Field: "{}"
Definition: {}

[INPUT SNIPPET]
{}

[INSTRUCTION]
1. Analyze the SNIPPET to find the value for '{}'.
2. Only extract data clearly visible in this snippet.
3. Ignore broken tags/syntax at boundaries.
4. Return valid JSON only: {{ "{}": value }}
5. If not found, return empty JSON {{}}."#,
                        chunk_idx + 1, chunks.len(),
                        field_name, field_schema,
                        chunk,
                        field_name,
                        field_name
                    );

                    let app_handle_clone = app_handle.clone();

                    tokio::select!{
                        res = model.chat_with_spinner(
                            "You are a strict data extraction parser. Return valid JSON only.", 
                            &prompt,
                            &app_handle_clone, "extraction-progress", json!({ 
                                "category": "Extraction", 
                                "summary": format!("Extracting field: {} ({}/{})", field_name, chunk_idx+1, chunks.len())
                            }), 
                            1024 
                        ) => res?,
                        _ = async {
                            loop {
                                if cancellation_token.load(Ordering::Relaxed) { break; }
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                        } => { return Err(anyhow::anyhow!("Task cancelled")); }
                    }
                } else {
                    "{}".to_string()
                };
                
                let field_data = parse_json_from_llm(&field_json_str);
                
                if let Some(obj) = extracted_data.as_object_mut() {
                    let mut temp_wrapper = serde_json::Map::new();
                    if let Some(val) = field_data.get(&field_name) {
                        temp_wrapper.insert(field_name.clone(), val.clone());
                    } else if !field_data.as_object().unwrap().is_empty() {
                        temp_wrapper.insert(field_name.clone(), field_data.clone());
                    }
                    
                    merge_json_results(&mut Value::Object(obj.clone()), &Value::Object(temp_wrapper));
                    
                    if let Some(new_val) = field_data.get(&field_name) {
                         if !new_val.is_null() {
                             let existing = obj.get(&field_name);
                             let is_empty = existing.map_or(true, |v| v.is_null() || (v.is_string() && v.as_str().unwrap().is_empty()));
                             if is_empty {
                                 obj.insert(field_name.clone(), new_val.clone());
                             }
                         }
                    }
                }
            }
            drop(model_guard);
            
            let mut field_map = serde_json::Map::new();
            field_map.insert(field_name.clone(), extracted_data.get(&field_name).cloned().unwrap_or(Value::Null));
            
            let _ = app_handle.emit("extraction-progress", json!({ 
                "category": "Extraction", "summary": format!("Field '{}' extracted.", field_name), "spinner": "✅",
                "data": Value::Object(field_map)
            }));
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

            let vector = if let Some(model) = model_guard.as_ref() {
                let text_clone = text_to_embed.clone();
                tokio::select!{
                    res = model.get_embedding(text_clone) => res.ok(),
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
                let _ = db.upsert_item(&target_table, &id, page_type, extracted_data.clone(), None).await;
        }
    }    
    
    let _ = app_handle.emit("extraction-progress", json!({ 
        "category": "Done", "summary": "Extraction Complete", "spinner": "✅", "data": extracted_data
    }));

    Ok(())
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
