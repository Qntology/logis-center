mod model;
mod store;
mod automation;
mod parsing;
mod logic;
mod scheduler;

pub mod models;
pub mod utils;
pub mod openai_types;
pub mod position_embed;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener}; // Added Manager
use tokio::sync::Mutex as TokioMutex;
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};

pub struct AppState {
    pub model: Arc<TokioMutex<Option<LogisModel>>>,
    pub store: Arc<TokioMutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn stop_current_extraction(
    state: State<'_, AppState>,
    task_id: Option<String>
) -> Result<String, String> {
    // 1. Set the flag immediately.
    state.cancellation_token.store(true, Ordering::SeqCst);
    
    // 2. Clear from DB
    if let Ok(store_guard) = state.store.try_lock() {
        if let Some(db) = store_guard.as_ref() {
            if let Some(ref id) = task_id {
                let _ = db.update_task_status(id, crate::logic::parse_status("cancel")).await;
                let _ = db.delete_message_by_task_id(id).await;
                println!("[STOP] Task and message {} cleared from DB.", id);
            } else {
                let _ = db.cleanup_zombie_tasks().await;
                println!("[STOP] All pending tasks cleared from DB.");
            }
        }
    }

    // 3. Try to clear model
    if let Ok(mut model_guard) = state.model.try_lock() {
        *model_guard = None;
    }

    Ok("Stop signal sent.".to_string())
}

#[tauri::command]
async fn delete_message(
    state: State<'_, AppState>,
    task_id: String
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.delete_message_by_task_id(&task_id).await.map_err(|e| e.to_string())?;
        Ok(format!("Message for task {} deleted.", task_id))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn unload_model(state: State<'_, AppState>) -> Result<String, String> {
    {
        let mut model_guard = state.model.lock().await;
        if let Some(m) = model_guard.as_ref() {
            m.unload_generator().await;
        }
        *model_guard = None;
    }
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    state.cancellation_token.store(false, Ordering::SeqCst);

    println!("[UNLOAD] Model, Store and Cancellation flag cleared.");
    Ok("Memory cleared.".to_string())
}

#[tauri::command]
async fn resize_window(app_handle: tauri::AppHandle, width: f64, height: f64) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
    }
}

#[tauri::command]
async fn start_drag(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.start_dragging();
    }
}

#[tauri::command]
async fn move_to_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.center();
    }
}

#[tauri::command]
async fn move_to_top_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let screen_size = monitor.size(); // PhysicalSize
            let scale_factor = monitor.scale_factor();
            let screen_width = screen_size.width as f64 / scale_factor;
            
            // Get current window size
            if let Ok(factor) = window.scale_factor() {
                if let Ok(size) = window.outer_size() {
                    let win_width = size.width as f64 / factor;
                    let new_x = (screen_width - win_width) / 2.0;
                    
                    let _ = window.set_position(tauri::Position::Logical(tauri::LogicalPosition {
                        x: new_x,
                        y: 0.0,
                    }));
                }
            }
        }
    }
}

#[tauri::command]
async fn set_ignore_cursor_events(app_handle: tauri::AppHandle, ignore: bool) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_ignore_cursor_events(ignore);
    }
}

#[tauri::command]
async fn launch_browser(
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    automation::run_browser_automation(browser, url, script, app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    let available = automation::get_available_browsers();
    // Priority: Chrome -> Edge -> Firefox
    let target = if available.iter().any(|b| b.name == "chrome") {
        "chrome"
    } else if available.iter().any(|b| b.name == "edge") {
        "edge"
    } else if available.iter().any(|b| b.name == "firefox") {
        "firefox"
    } else {
        return Err("No supported browser found.".to_string());
    };
    
    // Launch with default empty script
    automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_available_browsers() -> Vec<automation::BrowserStatus> {
    automation::get_available_browsers()
}

// --- Helper to Generate Rich Summary (Moved to model.rs) ---

#[tauri::command]
async fn summarize_image(
    state: State<'_, AppState>,
    _app_handle: tauri::AppHandle,
    image_path: String,
) -> Result<String, String> {
    println!("[INVOKE-01] summarize_image (Queue Integration) for path: {}", image_path);

    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let task_id = format!("img_{}", uuid::Uuid::new_v4().to_string());
        let now = chrono::Utc::now().timestamp_millis();
        
        let task_data = json!({
            "image_path": image_path,
            "id": task_id
        });

        let task = crate::store::Task {
            id: task_id.clone(),
            r#type: "image_extraction".to_string(),
            from: "manual_upload".to_string(),
            to: "local".to_string(),
            cc: "".to_string(),
            bcc: "".to_string(),
            r#ref: "manual".to_string(),
            data_json: task_data.to_string(),
            created_at: now,
            updated_at: now,
            status: crate::logic::parse_status("pending"),
        };

        match db.add_task(task).await {
            Ok(_) => Ok(format!("Task {} queued successfully.", task_id)),
            Err(e) => Err(format!("Failed to queue image task: {}", e)),
        }
    } else {
        Err("Database not initialized.".to_string())
    }
}

#[tauri::command]
async fn search_documents(
    state: State<'_, AppState>,
    query: String,
    limit: usize,
    _offset: usize,
    filter: Option<String>,
) -> Result<Vec<(String, String, f32)>, String> {
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        match VectorStore::new(db_path).await {
            Ok(s) => { *store_guard = Some(s); },
            Err(e) => return Err(format!("Failed to load DB: {}", e)),
        }
    }
    
    let model_guard = state.model.lock().await;
    let query_vec = if let Some(model) = model_guard.as_ref() {
        model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768])
    } else {
        vec![0.0; 768]
    };

    if let Some(store) = store_guard.as_ref() {
        store.search_items("items", &query, query_vec, limit, filter).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

// Helper to convert structured LLM conditions to SQL filter strings
fn convert_conditions_to_sql(ctx: &Value) -> Option<String> {
    let mut filters = Vec::new();
    
    // 1. Type Filter
    if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
        if !t.is_empty() { filters.push(format!("type = '{}'", t)); }
    }

    // 2. Condition Filters (Price, Date, etc.)
    if let Some(cond) = ctx.get("condition") {
        if let Some(price) = cond.get("price") {
            if let Some(gte) = price.get("gte").and_then(|v| v.as_f64()) { filters.push(format!("amount >= {}", gte)); }
            if let Some(lte) = price.get("lte").and_then(|v| v.as_f64()) { filters.push(format!("amount <= {}", lte)); }
        }
    }

    if filters.is_empty() { None } else { Some(filters.join(" AND ")) }
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let mut results = store.get_all_items("items", limit, offset, filter).await.map_err(|e| e.to_string())?;
        
        // [DYNAMIC] Convert JSON to Natural Language for UI display only
        for doc in results.iter_mut() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                doc.text = parsing::json_to_natural_language(&json_val);
            }
        }
        
        Ok(results)
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn get_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<Option<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        
        // 1. Primary search: Exact ID match
        for table_name in tables.iter() {
            if let Ok(Some(mut doc)) = store.get_item_by_id(table_name, &uuid).await {
                if doc.text.is_empty() {
                    if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                        doc.text = parsing::json_to_natural_language(&json_val);
                    }
                }
                return Ok(Some(doc));
            }
        }

        // 2. Fallback search: If uuid is numeric or a hash, look inside the data JSON
        // This fixes cases where the top-level 'id' column was saved as an empty string.
        for table_name in tables {
            // Try matching against "id" field inside JSON
            if let Ok(Some((_found_id, json_val))) = store.find_item_by_property(table_name, "id", &json!(uuid)).await {
                if let Ok(doc) = store.get_item_by_id(table_name, "").await { // Get the row with empty ID
                    // Double check it's the right one by comparing json_data
                    if let Some(mut d) = doc {
                        if d.json_data == json_val.to_string() {
                            if d.text.is_empty() { d.text = parsing::json_to_natural_language(&json_val); }
                            return Ok(Some(d));
                        }
                    }
                }
            }
            
            // Try matching against "index" field inside JSON
            let index_query = uuid.parse::<i64>().map(|n| json!(n)).unwrap_or(json!(uuid));
            if let Ok(Some((_found_id, _json_val))) = store.find_item_by_property(table_name, "index", &index_query).await {
                // To be safe, we perform a broader search for any row where data contains the index
                // Since find_item_by_property already found it, we just need to reconstruct the TradeDocument
                if let Ok(all_docs) = store.get_all_items(table_name, 1000, 0, None).await {
                    for mut d in all_docs {
                        if d.json_data.contains(&uuid) {
                            if d.text.is_empty() {
                                if let Ok(jv) = serde_json::from_str::<Value>(&d.json_data) {
                                    d.text = parsing::json_to_natural_language(&jv);
                                }
                            }
                            return Ok(Some(d));
                        }
                    }
                }
            }
        }
        
        Ok(None)
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn update_document(
    _state: State<'_, AppState>,
    _uuid: String,
    _json_data: String,
) -> Result<String, String> {
    Ok("Not implemented".to_string())
}

#[tauri::command]
async fn delete_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        // [DETAIL] 'items' 테이블뿐만 아니라 다른 가능한 테이블에서도 삭제 시도
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables {
            let _ = store.delete_item(table, &uuid).await;
        }
        Ok(format!("Document {} deleted.", uuid))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn delete_documents(
    state: State<'_, AppState>,
    uuids: Vec<String>,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        if uuids.is_empty() { return Ok("No documents to delete.".to_string()); }
        
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables {
            let _ = store.delete_items(table, uuids.clone()).await;
        }
        Ok(format!("Deleted {} documents.", uuids.len()))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn ai_search_complex(
    state: State<'_, AppState>,
    query: String,
    language: String,
    device_preference: Option<String>,
) -> Result<Value, String> {
    let mut model_guard = state.model.lock().await;
    
    // [FIX] Check if existing model matches preference
    if let Some(m) = model_guard.as_ref() {
        let wants_cpu = device_preference.as_deref() == Some("cpu");
        if m.is_cpu_mode != wants_cpu {
            println!("[AI-SEARCH] Device preference mismatch. Reloading model...");
            m.unload_generator().await;
            *model_guard = None;
        }
    }

    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(device_preference.as_deref()).await { *model_guard = Some(m); }
        else { return Err("Failed to load model".to_string()); }
    }
    let model = model_guard.as_ref().unwrap();

    // 1. Structured Parse (Para2Graph -> Graph2Contexts)
    let structured_query = model.parse_query_structured(query.clone(), &language).await.map_err(|e| e.to_string())?;
    
    // 2. Perform searches for each segment
    let mut all_results = Vec::new();
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        if let Ok(s) = VectorStore::new(db_path).await {
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }

    if let (Some(store), Some(ctx_arr)) = (store_guard.as_ref(), structured_query.get("context").and_then(|v| v.as_array())) {
        for ctx in ctx_arr {
            let text = ctx.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.is_empty() { continue; }
            
            // [FIX] Convert extracted conditions to SQL filter
            let sql_filter = convert_conditions_to_sql(ctx);
            
            let emb = model.get_embedding(text.to_string()).await.unwrap_or(vec![0.0; 768]);
            // Now passing the filter to search_items
            if let Ok(results) = store.search_items("items", text, emb, 5, sql_filter).await {
                for (id, content, score) in results {
                    all_results.push(json!({
                        "id": id,
                        "text": content,
                        "score": score,
                        "context_type": ctx.get("type")
                    }));
                }
            }
        }
    }

    Ok(json!({
        "structured": structured_query,
        "results": all_results
    }))
}

#[tauri::command]
async fn check_query_intent(
    _state: State<'_, AppState>,
    _query: String,
) -> Result<String, String> {
    Ok("SEARCH".to_string())
}

#[tauri::command]
async fn deep_research_command(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    query: String,
    _doc_id: Option<String>,
    device_preference: Option<String>,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;

    // [FIX] Check if existing model matches preference
    if let Some(m) = model_guard.as_ref() {
        let wants_cpu = device_preference.as_deref() == Some("cpu");
        if m.is_cpu_mode != wants_cpu {
            println!("[DEEP-RESEARCH] Device preference mismatch. Reloading model...");
            m.unload_generator().await;
            *model_guard = None;
        }
    }

    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(device_preference.as_deref()).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    let model = model_guard.as_ref().unwrap();

    // 1. Context Gathering
    let mut context_data = String::new();
    let mut store_guard = state.store.lock().await;
    
    if store_guard.is_none() {
        // Try init
        let db_path = "data/lancedb";
        let _ = std::fs::create_dir_all(db_path);
        if let Ok(s) = VectorStore::new(db_path).await {
            let _ = s.init_task_table().await;
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        // General search for context
        let emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768]);
        if let Ok(results) = store.search_items("items", &query, emb, 3, None).await {
            let docs: Vec<String> = results.iter()
                .map(|(_, text, _)| format!("- {}", text))
                .collect();
            context_data = docs.join("\n");
        }
    }
    
    // 2. Run Deep Research
    model.run_deep_research(query, context_data, &app_handle, Some(state.cancellation_token.clone())).await.map_err(|e| e.to_string())
}

#[tauri::command]

async fn proxy_fetch(

    url: String,

    method: String,

    headers: std::collections::HashMap<String, String>,

    body: Option<Value>,

    session_params: Option<Value>, // { hash, token, cc }

) -> Result<Value, String> {

    let client = reqwest::Client::builder()
        .use_native_tls()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| e.to_string())?;



    let mut target_url = url::Url::parse(&url).map_err(|e| e.to_string())?;



    // [DETAIL 1] Inject Session into Query Params (Content.js logic)

    if let Some(sp) = session_params {

        let mut query = target_url.query_pairs_mut();

        if let Some(hash) = sp.get("hash").and_then(|v| v.as_str()) { query.append_pair("hash", hash); }

        if let Some(token) = sp.get("token").and_then(|v| v.as_str()) { query.append_pair("token", token); }

        if let Some(cc) = sp.get("cc").and_then(|v| v.as_str()) { query.append_pair("cc", cc); }

    }



    let mut req_builder = match method.to_uppercase().as_str() {

        "POST" => client.post(target_url),

        "PUT" => client.put(target_url),

        "DELETE" => client.delete(target_url),

        _ => client.get(target_url),

    };



    for (k, v) in headers.iter() { req_builder = req_builder.header(k, v); }
    
    if let Some(b) = body { 
        if headers.get("Content-Encoding").map(|v| v.as_str()) == Some("gzip") {
            // [STRICT PARITY] Compress body if Gzip is requested
            if let Ok(compressed) = crate::utils::compression::compress_value(&b) {
                req_builder = req_builder.body(compressed);
            } else {
                req_builder = req_builder.json(&b);
            }
        } else {
            req_builder = req_builder.json(&b); 
        }
    }

    let response = req_builder.send().await.map_err(|e| e.to_string())?;
    let status = response.status();
    
    // Read response as text first to handle non-JSON cases (HTML, error pages, etc.)
    let text_res = response.text().await.map_err(|e| e.to_string())?;

    let json_res: Value = match serde_json::from_str(&text_res) {
        Ok(v) => v,
        Err(_) => {
            // If it's not JSON but request was successful, wrap it or return as text
            if status.is_success() {
                json!({ "text": text_res })
            } else {
                return Err(format!("Server error {} (Not JSON): {}", status, text_res));
            }
        }
    };

    if !status.is_success() {
        return Err(format!("Server returned {}: {}", status, json_res));
    }

    Ok(json_res)
}





#[derive(serde::Deserialize)]
struct ActiveTaskQuery {
    cc: String,
    r#ref: String,
}

#[tauri::command]
async fn check_active_task(state: State<'_, AppState>, payload: ActiveTaskQuery) -> Result<bool, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.has_active_task(&payload.cc, &payload.r#ref).await.map_err(|e| e.to_string())
    } else { Ok(false) }
}



#[tauri::command]
async fn initialize_hub(
    state: State<'_, AppState>,
    address: String,
    email: String,
    flag: String,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        match store.initialize_user_profiles(&address, &email, &flag).await {
            Ok(_) => Ok(format!("Hub initialized for address: {}", address)),
            Err(e) => Err(format!("Initialization failed: {}", e)),
        }
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn get_chat_messages(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<Value>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.get_all_messages(limit, offset, filter).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}



#[tauri::command]
async fn get_known_pages(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("pages", 100, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("users", 20, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn set_login_state(
    state: State<'_, AppState>,
    is_logged_in: bool,
    token: Option<String>,
) -> Result<String, String> {

    let store_guard = state.store.lock().await;

    if let Some(store) = store_guard.as_ref() {

        let mut config = store.load_config();

        config.is_logged_in = is_logged_in;

        config.auth_token = token;

        

        match store.save_config(&config) {

            Ok(_) => Ok(format!("Login state set to: {}", is_logged_in)),

            Err(e) => Err(e.to_string()),

        }

    } else {

        Err("Store not initialized".to_string())

    }

}



#[tauri::command]
async fn extract_html_from_current_tab() -> Result<String, String> {
    automation::extract_html_from_current_tab().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_browser_status() -> Result<String, String> {
    let guard = automation::GLOBAL_BROWSER.lock().await;
    if guard.is_some() || automation::is_browser_reachable().await { 
        return Ok("running".to_string()); 
    }
    Ok("stopped".to_string())
}

#[tauri::command]
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        // Fetch tasks with status 10 (pending) or 1 (progress)
        db.get_pending_tasks(10).await.map_err(|e| e.to_string())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_task_logs(app_handle: tauri::AppHandle, task_id: String) -> Result<Vec<Value>, String> {
    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task_id);
    if !log_path.exists() { return Ok(vec![]); }

    let content = std::fs::read_to_string(log_path).map_err(|e| e.to_string())?;
    let logs = content.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    Ok(logs)
}

#[tauri::command]
async fn upsert_items(state: State<'_, AppState>, items: Vec<Value>) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut count = 0;
        for item in items {
            // Basic parsing to determine ID and Table
            // In content.js structure: id, type are top level or in data
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let type_str = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            
            // Determine table based on type
            let _table = match type_str.as_str() {
                "sales" | "goods" | "order" => "sales",
                "tracking" | "receiving" | "shipping" => "tracking",
                "event" | "coupon" => "event",
                "member" | "team" | "user" => "users",
                "talk" => "talks",
                // Pages are stored in 'items' table with type='pages' in some contexts, 
                // or 'pages' table if strictly separated. The store supports "pages".
                // Based on previous context, page navigation items are usually type='order'/'goods' but acting as navigation nodes.
                // However, we want to store them where 'get_known_pages' looks. 
                // 'get_known_pages' looks at "pages" table.
                // Let's assume the sync sends items intended for the "pages" table if they are nav items.
                _ => {
                    // Fallback: If it looks like a page (has origin/link), put in pages
                    if item.get("origin").is_some() || item.get("link").is_some() {
                        "pages"
                    } else {
                        "items" 
                    }
                }
            };

            // Handling the "pages" and "users" specifically for the sync request
            // If the frontend sends explicit table hint, we could use that, but for now infer from type.
            let final_table = if type_str == "team" || type_str == "user" || type_str == "member" {
                "users"
            } else if item.get("data").and_then(|d| d.get("origin")).is_some() {
                "pages"
            } else {
                "items" // Default bucket
            };

            // Prepare fields
            let from = item.get("from").and_then(|v| v.as_str());
            let to = item.get("to").and_then(|v| v.as_str());
            let cc = item.get("cc").and_then(|v| v.as_str());
            let bcc = item.get("bcc").and_then(|v| v.as_str());
            let r#ref = item.get("ref").and_then(|v| v.as_str());
            let digest = item.get("digest").and_then(|v| v.as_str());

            if !id.is_empty() {
                let _ = db.upsert_item(final_table, &id, &type_str, item.clone(), None, from, to, cc, bcc, r#ref, digest).await;
                count += 1;
            }
        }
        Ok(format!("Synced {} items", count))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[derive(serde::Serialize)]
struct InitialSyncData {
    tasks: Vec<store::Task>,
    pages: Vec<store::TradeDocument>,
    users: Vec<store::TradeDocument>,
    items: Vec<store::TradeDocument>,
    browser_status: String,
    current_url: String,
    is_client: bool,
    is_admin: bool,
}

#[tauri::command]
async fn mark_ui_ready(state: State<'_, AppState>) -> Result<InitialSyncData, String> {
    scheduler::mark_ui_ready();
    
    let store_guard = state.store.lock().await;
    let mut tasks = Vec::new();
    let mut pages = Vec::new();
    let mut users = Vec::new();
    let mut items = Vec::new();
    
    if let Some(db) = store_guard.as_ref() {
        tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        pages = db.get_all_items("pages", 50, 0, None).await.unwrap_or_default();
        users = db.get_all_items("users", 20, 0, None).await.unwrap_or_default();
        items = db.get_all_items("items", 10, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let guard = automation::GLOBAL_BROWSER.lock().await;
        if guard.is_some() || automation::is_browser_reachable().await { "running".to_string() } else { "stopped".to_string() }
    };

    let (current_url, is_client, is_admin) = {
        let state = automation::LAST_DETECTED_STATE.lock().await;
        (state.url.clone(), state.is_client, state.is_admin)
    };

    Ok(InitialSyncData {
        tasks,
        pages,
        users,
        items,
        browser_status,
        current_url,
        is_client,
        is_admin,
    })
}

#[tauri::command]
async fn check_gpu_availability() -> bool {
    let config = crate::utils::get_optimal_device_config();
    !config.is_cpu
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let model = Arc::new(TokioMutex::new(None));
    let store = Arc::new(TokioMutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            model: model.clone(),
            store: store.clone(),
            cancellation_token: cancellation_token.clone(),
        })
        .setup(|app| {
            let setup_store = app.state::<AppState>().store.clone();
            tauri::async_runtime::spawn(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                let _ = std::fs::create_dir_all(db_path);
                if let Ok(s) = VectorStore::new(db_path).await {
                    println!("[Setup] VectorStore initialized.");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    *store_guard = Some(s);
                }
            });

            let scheduler_store = app.state::<AppState>().store.clone();
            let scheduler_model = app.state::<AppState>().model.clone();
            let scheduler_cancel = app.state::<AppState>().cancellation_token.clone();
            let scheduler_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                scheduler::start_background_worker(scheduler_store, scheduler_model, scheduler_cancel, scheduler_handle).await;
            });

            let auto_reconnect_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let _ = automation::try_reconnect_existing_browser(auto_reconnect_handle).await;
            });

            let event_store = app.state::<AppState>().store.clone();
            app.listen("new-task-from-browser", move |event| {
                if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {
                    let store_clone = event_store.clone();
                    tauri::async_runtime::spawn(async move {
                        let store_guard = store_clone.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let now = chrono::Utc::now().timestamp_millis();
                            let from_addr = payload_val.get("from").and_then(|v| v.as_str()).unwrap_or("0x0000000000000000000000000000000000000000").to_string();
                            let team_id = payload_val.get("to").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| crate::utils::hash::hash_id(&from_addr));
                            let task = crate::store::Task {
                                id: payload_val.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                r#type: payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                                from: from_addr, to: team_id,
                                cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                bcc: payload_val.get("bcc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                r#ref: payload_val.get("ref").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                data_json: payload_val.to_string(), created_at: now, updated_at: now, status: 10,
                            };
                            let msg_text = format!("Task Started: {}", payload_val.get("link").and_then(|v| v.as_str()).unwrap_or("Unknown URL"));
                            let _ = db.add_message(
                                &uuid::Uuid::new_v4().to_string(), 
                                "system_task", 
                                &msg_text, 
                                Some(&task.id), 
                                Some(1),
                                Some(&task.cc),
                                Some(&task.bcc),
                                Some(&task.r#ref),
                                Some(&task.from),
                                Some(&task.to),
                                Some("talk"),
                                None
                            ).await;
                            let _ = db.add_task(task).await;
                            crate::scheduler::notify_new_task();
                        }
                    });
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            summarize_image, search_documents, get_all_documents, get_document, check_query_intent, deep_research_command, ai_search_complex,
            launch_browser, launch_best_browser, extract_html_from_current_tab, stop_current_extraction, check_available_browsers,
            resize_window, start_drag, move_to_top_center, set_login_state, check_active_task, get_chat_messages, proxy_fetch,
            get_known_pages, get_known_users, initialize_hub, get_browser_status, get_active_tasks, unload_model, get_task_logs,
            upsert_items, set_ignore_cursor_events, mark_ui_ready, delete_document, delete_documents, delete_message, check_gpu_availability
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
