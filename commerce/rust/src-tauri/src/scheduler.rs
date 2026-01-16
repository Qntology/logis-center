use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;

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
                
                // Mark as processing
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, "processing").await;
                    }
                }

                match process_task(task.clone(), &store, &model).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, "done").await;
                        }
                    },
                    Err(e) => {
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, e);
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, "error").await;
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
    model_mutex: &Arc<Mutex<Option<LogisModel>>>)
-> Result<()> {
    
    // Parse Task Data
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("");
    let language = "english"; // Default, ideally detect from content or task

    if url.is_empty() {
        return Ok(()) // Or Err if URL is mandatory
    }

    // 1. Fetch Page Content
    // Check if HTML is already provided in the task data (e.g. from browser automation)
    let html = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        raw_html.to_string()
    } else if !url.is_empty() {
        // Fallback to fetching URL
        reqwest::get(url).await?.text().await?
    } else {
        return Ok(()); // No HTML and no URL
    };
    
    // =================================================================================
    // STEP 1: Classification (Structure Only) - "Map Outline"
    // =================================================================================
    
    // 1. Convert to lightweight Pug (Structure Only: ID, Class, Href)
    let light_pug = parsing::convert_to_clean_pug(&html, PugMode::StructureOnly);
    
    // Save Light Pug to file for debugging
    let _ = std::fs::write("debug_light_pug.txt", &light_pug);
    println!("[Scheduler] Saved Light Pug to debug_light_pug.txt (Length: {})", light_pug.len());
    
    // 2. LLM Call: Map Outline
    let mut model_guard = model_mutex.lock().await;
    
    if model_guard.is_none() {
        println!("[Scheduler] Model not initialized. Loading Qwen3-VL...");
        if let Ok(m) = LogisModel::new(None).await {
            *model_guard = Some(m);
        } else {
            return Ok(());
        }
    }

    let page_info_str = if let Some(model) = model_guard.as_ref() {
        let system = parsing::map_outline(language);
        model.chat(&system, &light_pug).await?
    } else {
        "{}".to_string()
    };
    drop(model_guard);
    
    println!("[Scheduler] Raw Map Output: {}", page_info_str);

    // 3. Parse Map Result (Using helper logic to handle markdown blocks)
    let page_info: Value = parse_json_from_llm(&page_info_str);
    
    let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("");
    let is_detail = page_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    let node_selector = page_info.get("node").and_then(|s| s.as_str()).unwrap_or("");
    let item_selector = page_info.get("item").and_then(|s| s.as_str()).unwrap_or("");

    println!("[Scheduler] Map Result: Type={}, Detail={}, Node='{}', Item='{}'", page_type, is_detail, node_selector, item_selector);

    if page_type == "" || page_type == "unknown" {
        println!("[Scheduler] Task finished early: Could not classify page type.");
        return Ok(());
    }

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
    
    let content_pug = parsing::convert_to_clean_pug_selector(&html, target_selector, PugMode::FullContent);
    
    // Save Full Content Pug to file for debugging
    let _ = std::fs::write("debug_content_pug.txt", &content_pug);
    println!("[Scheduler] Saved Content Pug to debug_content_pug.txt (Length: {})", content_pug.len());

    if content_pug.trim().is_empty() {
        println!("[Scheduler] Warning: Target selector '{}' returned empty content.", target_selector);
    }

    // 2. LLM Call: Extraction
    let prompt = if !is_detail {
        parsing::list2json(language)
    } else {
        parsing::item2json(page_type, url, language)
    };

    println!("[Scheduler] Sending Extraction Request (PUG Length: {})...", content_pug.len());
    let mut model_guard = model_mutex.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); }
    }

    let extracted_json_str = if let Some(model) = model_guard.as_ref() {
            model.chat("Extract information matching the JSON structure.", &format!("{}\n\nData (Zoomed In):\n{}", prompt, content_pug)).await?
    } else {
        "{}".to_string()
    };
    drop(model_guard);

    println!("[Scheduler] Raw Extraction Output: {}", extracted_json_str);
    let mut extracted_data: Value = parse_json_from_llm(&extracted_json_str);
    
    // Inject type/detail if missing
    if let Some(obj) = extracted_data.as_object_mut() {
        if obj.get("type").is_none() {
            obj.insert("type".to_string(), json!(page_type));
        }
        // Ensure detail flag is consistent
        // obj.insert("detail".to_string(), json!(is_detail)); 
    }

    // =================================================================================
    // STEP 3: Logic Relay, Merge & Save
    // =================================================================================

    if let Some((queries, merge_info)) = logic::relay(page_type, &extracted_data) {
        println!("[Scheduler] Relay logic triggered. Queries: {}", queries.len());
        
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let mut target_data = extracted_data.clone();
            let mut target_id = format!("{}_{}", page_type, chrono::Utc::now().timestamp_millis());
            
            let mut found_existing = false;
            
            // Map logic table names to DB table names
            let to_table = format!("commerce_{}", merge_info.to);

            for query in queries {
                let query_table = format!("commerce_{}", query.table);
                if let Ok(Some((id, existing_data))) = db.find_item_by_property(&query_table, &query.column, &query.value).await {
                    println!("[Scheduler] Found existing item: {} in {}", id, query_table);
                    
                    // Check diff before merging/saving to avoid redundant writes
                    // if !is_diff(&target_data, &existing_data) { ... } // Logic optimization possible here
                    
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

            // Generate Vector
            let text_to_embed = target_data.to_string();
            
            drop(store_guard); 
            let mut model_guard = model_mutex.lock().await;
            if model_guard.is_none() {
                if let Ok(m) = LogisModel::new(None).await {
                     *model_guard = Some(m);
                }
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
                
                let _ = db.upsert_item(&target_table, &id, page_type, extracted_data, None).await;
        }
    }    
        Ok(())
    }
    
    fn parse_json_from_llm(text: &str) -> Value {
        if let Ok(v) = serde_json::from_str(text) {
            return v;
        }
        
        // Try to find JSON block in markdown
        if let Some(start) = text.find("{") {
            if let Some(end) = text.rfind("}") {
                if start < end {
                    let json_part = &text[start..=end];
                    if let Ok(v) = serde_json::from_str(json_part) {
                        return v;
                    }
                }
            }
        }
        
        // Try array block
        if let Some(start) = text.find("[") {
            if let Some(end) = text.rfind("]") {
                if start < end {
                    let json_part = &text[start..=end];
                    if let Ok(v) = serde_json::from_str(json_part) {
                        return v;
                    }
                }
            }
        }
    
        json!({})
    }
    