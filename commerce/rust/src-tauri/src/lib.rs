pub mod _parsing;
pub mod automation;
pub mod logic;
pub mod model;
pub mod models;
pub mod openai_types;
pub mod parsing;
pub mod scheduler;
pub mod store;
pub mod tokenizer;
pub mod utils;
pub mod chat_template;

use tauri::{Manager, State, Listener};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{json, Value};
use crate::model::LogisModel;
use crate::store::{VectorStore, TradeDocument};

pub struct AppState {
    pub store: Arc<Mutex<Option<VectorStore>>>,
    pub model: Arc<Mutex<Option<LogisModel>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn stop_current_extraction(
    state: State<'_, AppState>,
    task_id: Option<String>
) -> Result<String, String> {
    state.cancellation_token.store(true, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(true);
    
    if let Ok(store_guard) = state.store.try_lock() {
        if let Some(db) = store_guard.as_ref() {
            if let Some(ref id) = task_id {
                let _ = db.update_task_status(id, 3).await; // 3 = Cancel
                let _ = db.delete_message_by_task_id(id).await;
            } else {
                let _ = db.cleanup_zombie_tasks().await;
            }
        }
    }

    if let Ok(mut model_guard) = state.model.try_lock() {
        if let Some(m) = model_guard.as_ref() {
            m.unload_generator().await;
        }
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
async fn move_to_top_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let screen_size = monitor.size();
            let scale_factor = monitor.scale_factor();
            let screen_width = screen_size.width as f64 / scale_factor;
            
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
async fn launch_best_browser(
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    let available = automation::get_available_browsers();
    let target = if available.iter().any(|b| b.name == "chrome") {
        "chrome"
    } else if available.iter().any(|b| b.name == "edge") {
        "edge"
    } else if available.iter().any(|b| b.name == "firefox") {
        "firefox"
    } else {
        return Err("No supported browser found.".to_string());
    };
    
    automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_available_browsers() -> Vec<automation::BrowserStatus> {
    automation::get_available_browsers()
}

#[tauri::command]
async fn proxy_fetch(
    url: String,
    method: String,
    headers: std::collections::HashMap<String, String>,
    body: Option<Value>,
    session_params: Option<Value>,
) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .use_native_tls()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| e.to_string())?;

    let mut target_url = url::Url::parse(&url).map_err(|e| e.to_string())?;

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
    let text_res = response.text().await.map_err(|e| e.to_string())?;

    let json_res: Value = match serde_json::from_str(&text_res) {
        Ok(v) => v,
        Err(_) => {
            if status.is_success() { json!({ "text": text_res }) } 
            else { return Err(format!("Server error {} (Not JSON): {}", status, text_res)); }
        }
    };

    Ok(json_res)
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
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<crate::store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
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
    let logs = content.lines().filter_map(|line| serde_json::from_str(line).ok()).collect();
    Ok(logs)
}

#[tauri::command]
async fn mark_ui_ready(state: State<'_, AppState>) -> Result<Value, String> {
    scheduler::mark_ui_ready();
    let store_guard = state.store.lock().await;
    let mut tasks = Vec::new();
    let mut pages = Vec::new();
    if let Some(db) = store_guard.as_ref() {
        tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        pages = db.get_all_items("pages", 50, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let guard = automation::GLOBAL_BROWSER.lock().await;
        if guard.is_some() || automation::is_browser_reachable().await { "running".to_string() } else { "stopped".to_string() }
    };

    Ok(json!({
        "tasks": tasks,
        "pages": pages,
        "browser_status": browser_status,
    }))
}

#[tauri::command]
async fn check_gpu_availability() -> bool {
    let config = crate::utils::get_optimal_device_config();
    !config.is_cpu
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
        store.get_all_items("items", limit, offset, filter).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

pub fn run() {
    let model = Arc::new(Mutex::new(None));
    let store = Arc::new(Mutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            store: store.clone(),
            model: model.clone(),
            cancellation_token: cancellation_token.clone(),
        })
        .setup(|app| {
            let setup_store = app.state::<AppState>().store.clone();
            tauri::async_runtime::spawn(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                if let Ok(s) = VectorStore::new(db_path).await {
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    *store_guard = Some(s);
                }
            });

            let sch_store = app.state::<AppState>().store.clone();
            let sch_model = app.state::<AppState>().model.clone();
            let sch_cancel = app.state::<AppState>().cancellation_token.clone();
            let sch_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                scheduler::start_background_worker(sch_store, sch_model, sch_cancel, sch_handle).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            stop_current_extraction, delete_message, unload_model, resize_window,
            start_drag, move_to_top_center, set_ignore_cursor_events, launch_best_browser,
            check_available_browsers, proxy_fetch, set_login_state, get_active_tasks,
            get_task_logs, mark_ui_ready, check_gpu_availability, get_all_documents
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
