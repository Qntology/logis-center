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

// Ported from proxy/src/index.ts `isDiff`
#[allow(dead_code)]
fn is_diff(v1: &Value, v2: &Value) -> bool {
    if v1.is_null() && v2.is_null() { return false; }
    if v1.is_null() || v2.is_null() { return true; }
    v1 != v2
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker started.");
    
    tokio::spawn(async move {
        // Imitating the timeout/delay logic from proxy/src/index.ts
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
            } // Release lock

            if pending_tasks.is_empty() {
                // Increase delay if no tasks (up to a limit), similar to adaptive timeout
                delay_secs = (delay_secs + 1).min(10); 
                continue;
            } else {
                // Reset delay if tasks found
                delay_secs = 1;
            }

            for task in pending_tasks {
                println!("[Scheduler] Processing task: {}", task.id);
                
                // Reset cancellation token for the new task
                cancellation_token.store(false, Ordering::SeqCst);
                
                // Mark as processing
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
                             // Emit cancelled event to UI to reset state
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
) -> Result<()> {
    
    // Check Cancellation
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // Initial Progress
    let _ = app_handle.emit("extraction-progress", json!({
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋"
    }));

    // Parse Task Data
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; // Default, ideally detect from content or task

    // --- Image Extraction Logic ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("");
        
        let _ = app_handle.emit("extraction-progress", json!({
            "category": "Image Loading", "summary": "Loading image...", "spinner": "⠋"
        }));

        if let Ok(img) = image::open(image_path) {
             let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
             
             let prompt = parsing::image2json("kr", language, "tracking", ""); // Defaults
             
             // Check Cancellation
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let mut model_guard = model_mutex.lock().await;
             if model_guard.is_none() {
                 let _ = app_handle.emit("extraction-progress", json!({
                    "category": "Loading Model", "summary": "Loading Vision Model...", "spinner": "⠋"
                 }));
                 match LogisModel::new(None).await {
                     Ok(m) => *model_guard = Some(m),
                     Err(e) => {
                         println!("[Scheduler] Failed to load LogisModel: {:?}", e);
                         let _ = app_handle.emit("extraction-progress", json!({
                            "category": "Error", "summary": format!("Model Load Failed: {}", e), "spinner": "❌"
                         }));
                         return Ok(()); // Stop task processing but don't crash worker
                     }
                 }
             }
             
             // Check Cancellation before Inference
             if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

             let result_str = if let Some(model) = model_guard.as_ref() {
                 model.chat_with_image_spinner(prompt, Some(dynamic_image), app_handle, "extraction-progress", json!({
                     "category": "Vision Analysis", "summary": "Analyzing image content..."
                 }), 1024).await?
             } else {
                 "{}".to_string()
             };
             drop(model_guard);
             
             let extracted_data = parse_json_from_llm(&result_str);
             
             // Save Result
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

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("");

    if url.is_empty() {
        return Ok(()) // Or Err if URL is mandatory
    }

    // 1. Fetch Page Content
    let raw_html_content = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        raw_html.to_string()
    } else if !url.is_empty() {
        // Fallback to fetching URL
        reqwest::get(url).await?.text().await?
    } else {
        return Ok(())
    };
    
    // Check Cancellation
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // OPTIMIZATION: Pre-clean ONCE (String is Send, so we can keep it)
    let clean_html = parsing::pre_clean_html(&raw_html_content);
    // Drop raw content to free memory
    drop(raw_html_content); 
    
    // SCOPE 1: Parse for Light Pug (Structure Only)
    // We must drop 'document' before awaiting LLM because scraper::Html is !Send
    let light_pug = {
        let document = scraper::Html::parse_document(&clean_html);
        let lp = parsing::convert_doc_to_clean_pug(&document, PugMode::StructureOnly);
        lp
    }; // document dropped here
    
    // Save Light Pug to file for debugging
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    println!("[Scheduler] Saved Light Pug to debug_light_pug.txt (Length: {})", light_pug.len());
    
    // 2. LLM Call: Map Outline
    let mut model_guard = model_mutex.lock().await;
    
    if model_guard.is_none() {
        println!("[Scheduler] Model not initialized. Loading Qwen3-VL...");
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

    // Check CPU Warning
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
    
    // Check Cancellation before Inference
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let page_info_str = if let Some(model) = model_guard.as_ref() {
        let system = parsing::map_outline(language);
        model.chat_with_spinner(&system, &light_pug, app_handle, "extraction-progress", json!({
            "category": "Classification", "summary": "Analyzing page structure..."
        }), 512).await? // [Optimization] Limit to 512 tokens for classification
    } else {
        "{}".to_string()
    };
    drop(model_guard); // Release lock
    
    println!("[Scheduler] Raw Map Output: {}", page_info_str);

    // 3. Parse Map Result
    let page_info: Value = parse_json_from_llm(&page_info_str);
    
    // Emit Classification Result
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

    println!("[Scheduler] Map Result: Type={}, Detail={}, Node='{}', Item='{}'", page_type, is_detail, node_selector, item_selector);

    if page_type == "" || page_type == "unknown" {
        println!("[Scheduler] Task finished early: Could not classify page type.");
        return Ok(());
    }
    
    // Check Cancellation
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // =================================================================================
    // STEP 2: Extraction (Full Content) - "Zoom In"
    // =================================================================================

    let target_selector = if !node_selector.is_empty() { 
        node_selector 
    } else if !item_selector.is_empty() {
        item_selector
    } else {
        "body"
    };
    
    let mut extracted_data = json!({});

    if !is_detail {
        // --- List Page Logic (Item-wise Chunking) ---
        let _ = app_handle.emit("extraction-progress", json!({
            "category": "List Processing", "summary": "Splitting list items...", "spinner": "⠋"
        }));

        // SCOPE 2: Parse for Extraction (Re-parsing clean_html, but necessary for !Send)
        let item_pugs = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::split_doc_to_pug_list(&document, target_selector, PugMode::FullContent)
        }; // document dropped
        
        let total_items = item_pugs.len();
        
        let _ = app_handle.emit("extraction-progress", json!({
            "category": "List Processing", 
            "summary": format!("Found {} items to extract.", total_items),
            "spinner": "✅"
        }));

        let mut items_array = Vec::new();
        // Use item2json schema to extract details for each list item
        let fields_prompts = parsing::item2json(page_type, url, language);

        for (i, item_pug) in item_pugs.iter().enumerate() {
            // Check Cancellation inside Loop
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let mut item_data = json!({});
            
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Extraction", 
                "summary": format!("Extracting Item {}/{}: ...", i + 1, total_items), 
                "spinner": "⠋"
            }));

            // Re-acquire lock for each item extraction loop
            let model_guard = model_mutex.lock().await; 

            for (field_name, field_prompt) in &fields_prompts {
                // Optimization: Skip heavy fields for list view
                if field_name == "description" || field_name == "detail_images" { continue; }

                let field_json_str = if let Some(model) = model_guard.as_ref() {
                    model.chat_with_spinner(
                        "Extract ONLY the requested field from the Pug snippet. Return valid JSON.", 
                        &format!("{}\n\nItem Pug Snippet:\n{}", field_prompt, item_pug),
                        app_handle, "extraction-progress", json!({
                            "category": "Extraction", 
                            "summary": format!("Item {}/{}: {}", i + 1, total_items, field_name)
                        }), 
                        512 // Fast token limit for list items
                    ).await?
                } else {
                    "{}".to_string()
                };
                
                let field_result = parse_json_from_llm(&field_json_str);
                
                // Merge field result
                // field_result is likely { "field_name": { ... } } or { "value": ... }
                if let Some(val) = field_result.get(field_name) {
                    item_data.as_object_mut().unwrap().insert(field_name.clone(), val.clone());
                } else {
                    // If the model returned just the object without the field key wrapper
                    item_data.as_object_mut().unwrap().insert(field_name.clone(), field_result.clone());
                }
            }
            drop(model_guard); 

            // Normalize ID/Index for the item
            if let Some(id_val) = item_data.get("id").cloned() {
                 if item_data.get("index").is_none() { 
                     item_data.as_object_mut().unwrap().insert("index".to_string(), id_val);
                 }
            }

            items_array.push(item_data.clone());
            
            // Show intermediate item result
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Extraction", 
                "summary": format!("Item {} extracted.", i + 1),
                "data": item_data
            }));
        }
        
        extracted_data.as_object_mut().unwrap().insert("items".to_string(), json!(items_array));
        extracted_data.as_object_mut().unwrap().insert("type".to_string(), json!(page_type));

    } else {
        // --- Detail Page Logic (Field-by-Field) ---
        // Zoom in first
        // SCOPE 2: Parse for Extraction
        let content_pug = {
            let document = scraper::Html::parse_document(&clean_html);
            parsing::convert_doc_to_clean_pug_selector(&document, target_selector, PugMode::FullContent)
        }; // document dropped
        
        // --- Selector Check: If node/item selector yielded no content, stop. ---
        if content_pug.trim().is_empty() {
            println!("[Scheduler] Selector '{}' not found in HTML. Skipping extraction.", target_selector);
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Error", 
                "summary": format!("Selector '{}' not found.", target_selector), 
                "spinner": "❌"
            }));
            // Emit Done to reset UI state
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Done", "summary": "Extraction Failed", "spinner": "🛑", "data": null
            }));
            return Ok(());
        }

        // Save Debug
        let _ = std::fs::write("debug_content_pug.txt", &content_pug);

        let fields_prompts = parsing::item2json(page_type, url, language);
        
        for (field_name, field_prompt) in fields_prompts {
            // Check Cancellation inside Loop
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Extraction", 
                "summary": format!("Extracting field: {}", field_name), 
                "spinner": "⠋"
            }));

            // Acquire lock
            let model_guard = model_mutex.lock().await;

            let field_json_str = if let Some(model) = model_guard.as_ref() {
                model.chat_with_spinner(
                    "Extract ONLY the requested field from the Pug template. Return valid JSON.", 
                    &format!("{}\n\nData (Zoomed In):\n{}", field_prompt, content_pug),
                    app_handle, "extraction-progress", json!({
                        "category": "Extraction", 
                        "summary": format!("Extracting field: {}", field_name)
                    }), 
                    1024 
                ).await?
            } else {
                "{}".to_string()
            };
            drop(model_guard);
            
            // Parse and Merge
            let field_data = parse_json_from_llm(&field_json_str);
            if let Some(obj) = extracted_data.as_object_mut() {
                if let Some(val) = field_data.get(&field_name) {
                    obj.insert(field_name.clone(), val.clone());
                } else {
                    obj.insert(field_name.clone(), field_data.clone());
                }
            }
            
            // Show intermediate result in UI
            let _ = app_handle.emit("extraction-progress", json!({
                "category": "Extraction", 
                "summary": format!("Field '{}' extracted.", field_name),
                "spinner": "✅",
                "data": json!({ field_name.clone(): extracted_data.get(&field_name) })
            }));
        }
    }
    
    // Check Cancellation
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // Inject type if missing
    if let Some(obj) = extracted_data.as_object_mut() {
        if obj.get("type").is_none() {
            obj.insert("type".to_string(), json!(page_type));
        }
    }

    // --- Normalization Step ---
    let mut normalized_data = json!({});
    if let Some(obj) = extracted_data.as_object() {
        for (k, v) in obj {
            let val = if let Some(inner_obj) = v.as_object() {
                if let Some(value_field) = inner_obj.get("value") {
                    value_field.clone()
                } else {
                    v.clone()
                }
            } else {
                v.clone()
            };
            normalized_data.as_object_mut().unwrap().insert(k.clone(), val);
        }
    }
    
    // Sync 'id' and 'index'
    let id_opt = normalized_data.get("id").cloned();
    let index_opt = normalized_data.get("index").cloned();

    if let Some(id_val) = id_opt {
        if normalized_data.get("index").is_none() {
            normalized_data.as_object_mut().unwrap().insert("index".to_string(), id_val);
        }
    } else if let Some(idx_val) = index_opt {
        if normalized_data.get("id").is_none() {
            normalized_data.as_object_mut().unwrap().insert("id".to_string(), idx_val);
        }
    }
    
    extracted_data = normalized_data;

    // =================================================================================
    // STEP 3: Logic Relay, Merge & Save
    // =================================================================================
    
    // Check Cancellation
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
            if model_guard.is_none() {
                if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); }
            }
            
            let vector = if let Some(model) = model_guard.as_ref() {
                model.get_embedding(text_to_embed).await.ok()
            } else {
                None
            };
            drop(model_guard);
            
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                println!("[Scheduler] Saving item: {} to {}", target_id, to_table);
                let _ = db.upsert_item(&to_table, &target_id, page_type, target_data, vector).await;
            }
        }
    } else {
        // Simple Save (No Relay)
        let target_table = format!("commerce_{}", page_type);
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
                let id = extracted_data.get("id")
                .and_then(|s| s.as_str())
                .unwrap_or_else(|| task.id.as_str())
                .to_string();
                
                let _ = db.upsert_item(&target_table, &id, page_type, extracted_data.clone(), None).await;
        }
    }    
    
    // Final Done Event
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
