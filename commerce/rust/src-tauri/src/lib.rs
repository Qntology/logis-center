mod model;
mod store;
mod automation;
pub mod parsing;
mod logic;
mod scheduler;

pub mod models;
pub mod utils;
pub mod position_embed;
pub mod openai_types;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener}; 
use tokio::sync::Mutex as TokioMutex;
use std::sync::RwLock; // 🌟 추가
use once_cell::sync::Lazy; // 🌟 추가
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};

// 🌟 [CRITICAL FIX] 전역 검색 상태 락: 검색 중 스레드 데드락 방지
static IS_SEARCHING: AtomicBool = AtomicBool::new(false);

pub static ACTIVE_TASK_MEM: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));
pub static CURRENT_UI_CATEGORY: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::from("Processing")));
// 🌟 [CRITICAL FIX] SSD에 적히지 않는 실시간 퍼센트 데이터를 붙잡아둘 0.01초 단위 메모리 캐시!
pub static LATEST_PROGRESS_PAYLOAD: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));

pub struct AppState {
    pub model: Arc<TokioMutex<Option<LogisModel>>>,
    pub store: Arc<TokioMutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn get_active_task_context() -> Result<Value, String> {
    let mut result = json!(null);
    
    if let Some(mem_task) = crate::ACTIVE_TASK_MEM.read().unwrap().clone() {
        if mem_task.get("id").is_some() { result = mem_task; }
    }

    if !result.is_null() {
        if let Ok(mem) = crate::LATEST_PROGRESS_PAYLOAD.read() {
            if let Some(latest) = mem.as_ref() {
                if result.get("id") == latest.get("task_id") {
                    if let Some(summary) = latest.get("summary") {
                        result.as_object_mut().unwrap().insert("summary".to_string(), summary.clone());
                    }
                    // 🌟 [CRITICAL FIX] 프론트엔드가 퍼센트(%)를 즉시 복구할 수 있도록 최신 전체 페이로드도 통째로 꽂아줍니다!
                    result.as_object_mut().unwrap().insert("latest_payload".to_string(), latest.clone());
                }
            }
        }
        return Ok(result);
    }
    
    Ok(json!(null))
}

#[tauri::command]
async fn stop_current_extraction(
    state: State<'_, AppState>,
    task_id: Option<String>
) -> Result<String, String> {
    // 1. Set global stop signals (Atomic + File-based for persistence across threads)
    state.cancellation_token.store(true, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(true);
    
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
    // 🌟 [CRITICAL FIX] 검색 중일 때는 unload를 즉시 튕겨냅니다. (Mutex 락 꼬임 및 UI 프리징 완벽 차단!)
    if IS_SEARCHING.load(Ordering::SeqCst) {
        println!("[UNLOAD] AI Search is active. Skipping unload to prevent deadlock.");
        return Ok("Search active. Memory kept.".to_string());
    }

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
    state: State<'_, AppState>, // 🌟 Tauri State 주입
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    // 🌟 [CRITICAL FIX] 브라우저 실행 전, 잔류한 Stop 시그널을 강제 해제합니다!
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

    automation::run_browser_automation(browser, url, script, app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    state: State<'_, AppState>, // 🌟 Tauri State 주입
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    // 🌟 [CRITICAL FIX] 브라우저 실행 전, 잔류한 Stop 시그널을 강제 해제합니다!
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

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
    // 🌟 [추가] 프론트엔드가 텍스트 검색을 요청할 때마다 터미널에 로그를 찍습니다.
    println!("[DB-SEARCH] 텍스트 검색 요청 수신 (Query: '{}', Filter: {:?})", query, filter);

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = "data/lancedb";
            if let Ok(s) = VectorStore::new(db_path).await {
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    }; // 🌟 즉시 자물쇠 해제!
    
    // 🌟 [CRITICAL FIX] 빈 쿼리일 때는 임베딩 모델을 절대 로드하지 않음 (검색 직후 VRAM이 다시 차버리는 현상 해결!)
    let query_vec = if !query.trim().is_empty() {
        let model_opt = { state.model.lock().await.as_ref().cloned() }; // 🌟 즉시 자물쇠 해제!
        if let Some(model) = model_opt {
            model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768])
        } else {
            vec![0.0; 768]
        }
    } else {
        vec![0.0; 768]
    };

    if let Some(store) = store_opt {
        store.search_items("items", &query, query_vec, limit, filter).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

// Helper to convert structured LLM conditions to SQL filter strings
fn convert_conditions_to_sql(ctx: &Value) -> Option<String> {
    let mut filters = Vec::new();
    
    if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
        if !t.is_empty() { filters.push(format!("type = '{}'", t)); }
    }

    if let Some(status) = ctx.get("status").and_then(|v| v.as_str()) {
        if !status.is_empty() && status != "null" {
            // 🌟 [CRITICAL FIX 2] 상태값은 문자열이 아니라 Int32로 치환해서 검색해야 합니다.
            let status_int = crate::logic::parse_status(status);
            filters.push(format!("status = {}", status_int));
        }
    }

    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
        for (key, val_obj) in cond {
            // 🌟 [CRITICAL FIX] Python의 PATH_MAP 역할을 수행하도록 무역 문서용 필드 완벽 추가!
            let valid_cols = [
                "amount", "status", "type", "created_at", "updated_at",
                "no", "carrier", "shipping_method", "sender_address", "recipient_address", 
                "shipping_date", "delivery_date", "weight",
                // 🌟 [무역 문서 전용 컬럼 추가]
                "vessel", "pol", "pod", "incoterms", "sender_name", "recipient_name", "issue_date"
            ];
            
            let mapped_key = match key.as_str() {
                "price" | "sale_price" | "discount" | "supply_price" | "order" | "goods" => "amount",
                // 🌟 무역 문서의 중첩 키 평탄화 매핑 (Python PATH_MAP과 동일한 역할)
                "document_number" | "tracking_number" => "no",
                "supplier_name" | "shipper_name" => "sender_name",
                "buyer_name" | "consignee_name" => "recipient_name",
                "amount_total" | "total_amount" => "amount",
                "vehicle_name" | "flight_no" => "vessel",
                "location_port_of_loading" => "pol",
                "location_port_of_discharge" => "pod",
                "incoterms_code" => "incoterms",
                k if valid_cols.contains(&k) => k,
                _ => "" 
            };

            if mapped_key.is_empty() { continue; } // 유효하지 않은 컬럼은 무시하여 DB 크래시 방어

            if let Some(op_str) = val_obj.get("operator").and_then(|v| v.as_str()) {
                if let Some(val_val) = val_obj.get("value") {
                    let operator = match op_str {
                        "gt" => ">", "gte" => ">=", "lt" => "<", "lte" => "<=", "eq" => "=", _ => "="
                    };
                    
                    let val_str = if val_val.is_number() {
                        val_val.to_string()
                    } else if let Some(s) = val_val.as_str() {
                        let numeric: String = s.chars().filter(|c| c.is_digit(10) || *c == '.').collect();
                        if numeric.is_empty() { continue; } else { numeric }
                    } else { continue; };

                    filters.push(format!("{} {} {}", mapped_key, operator, val_str));
                }
            }
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
    // 🌟 [추가] 프론트엔드가 리스트를 요청할 때마다 터미널에 로그를 찍습니다.
    println!("[DB-FETCH] 리스트 불러오기 요청 수신 (Limit: {}, Filter: {:?})", limit, filter);

    let mut store_guard = state.store.lock().await; // 🌟 mut로 변경하여 쓰기 가능하게 만듭니다.
    
    // 🌟 [CRITICAL FIX] DB 초기화 레이스 컨디션 해결
    // 프론트엔드가 데이터를 요청했는데 DB가 아직 없으면 즉시 여기서 로드합니다.
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        let _ = std::fs::create_dir_all(db_path);
        if let Ok(s) = VectorStore::new(db_path).await {
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        } else {
            return Err("Failed to initialize LanceDB".to_string());
        }
    }

    if let Some(store) = store_guard.as_ref() {
        let mut results = store.get_all_items("items", limit, offset, filter).await.map_err(|e| e.to_string())?;
        
        // [DYNAMIC] Convert JSON to Natural Language for UI display only
        for doc in results.iter_mut() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                doc.text = crate::parsing::json_to_natural_language(&json_val);
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
    app_handle: tauri::AppHandle,
    task_id: String,
    query: String,
    language: String,
    device_preference: Option<String>,
    search_mode: String,
    cc: String,      // 🌟 [추가] 프론트에서 보낸 현재 위치
    bcc: String,     // 🌟 [추가] 프론트에서 보낸 현재 위치
    ref_id: String,  // 🌟 [추가] 프론트에서 보낸 현재 위치
) -> Result<Value, String> {
    
    let emit_term = |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
    };

    emit_term("\n==================================================");
    emit_term("🚀 [AI-SEARCH] 프론트엔드 요청 수신 완료!");
    emit_term(&format!("   - Task ID: {}", task_id));
    emit_term(&format!("   - 검색어: {}", query));
    emit_term(&format!("   - 검색 모드: {}", search_mode));
    emit_term("==================================================\n");

    IS_SEARCHING.store(true, Ordering::SeqCst);
    
    let cancel_token = state.cancellation_token.clone();
    
    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = "data/lancedb";
            if let Ok(s) = VectorStore::new(db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned() 
    };

    // 🌟 DB에 Task 및 Message 등록
    if let Some(store) = store_opt.as_ref() {
        let now = chrono::Utc::now().timestamp_millis();
        let task = crate::store::Task {
            id: task_id.clone(), 
            r#type: "ai_search".to_string(),
            from: "user".to_string(),
            to: "system".to_string(),
            cc: cc.clone(), bcc: bcc.clone(), r#ref: ref_id.clone(), 
            data_json: json!({"query": query.clone(), "mode": search_mode.clone()}).to_string(),
            created_at: now, updated_at: now,
            status: 10, // 🌟 [CRITICAL FIX] 검색도 대기열에 들어가므로 일단 Pending(10)으로 시작!
        };
        let _ = store.add_task(task).await;
        
        let _ = store.add_message(
            &task_id, "user", &query, 
            Some(&task_id), Some(10), // 🌟 [CRITICAL FIX] 말풍선도 Pending(10) 상태로 표시
            Some(&cc), Some(&bcc), Some(&ref_id), 
            None, None, Some("talk"), None
        ).await;
    }

    let model = {
        let mut model_guard = state.model.lock().await;
        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.unload_generator().await;
                *model_guard = None;
            }
        }
        if model_guard.is_none() {
            if let Ok(m) = LogisModel::new(app_handle.clone(), device_preference.as_deref()).await { 
                *model_guard = Some(m);
            } else { 
                IS_SEARCHING.store(false, Ordering::SeqCst); 
                return Err("Failed to load model".to_string());
            }
        }
        model_guard.as_ref().unwrap().clone() 
    }; 

    let search_process = async {
        let mut all_results = Vec::new();
        
        let team_id = crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000"); 
        let mut metrics_json_str = "{}".to_string();
        
        if let Some(store) = store_opt.as_ref() {
            if let Ok(Some(doc)) = store.get_item_by_id("users", &team_id).await {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                    if let Some(base) = val.get("base") {
                        metrics_json_str = base.to_string();
                    }
                }
            }
        }

        // 🌟 취소 토큰(cancel_token)을 파싱 함수에 전달!
        let structured_query = match search_mode.as_str() {
            "shipping" => {
                model.parse_shipping_query(&task_id, &app_handle, query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            "analytic" => {
                model.parse_analytic_query(&task_id, &app_handle, query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            _ => { // default: commerce
                model.parse_commerce_query(&task_id, &app_handle, query.clone(), &language, &metrics_json_str, cancel_token.clone()).await.map_err(|e| e.to_string())?
            }
        };

        if let (Some(store), Some(ctx_arr)) = (store_opt.clone(), structured_query.get("context").and_then(|v| v.as_array())) {
            for ctx in ctx_arr {
                // 🌟 매 루프마다 사용자가 Cancel 버튼을 눌렀는지 체크!
                if cancel_token.load(Ordering::Relaxed) { 
                    return Err("Search cancelled by user".to_string()); 
                }

                tokio::task::yield_now().await;

                let text = ctx.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if text.is_empty() { continue; }
                
                let ctx_type = ctx.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                
                // 🌟 [CRITICAL FIX] 데이터 파편화를 막고 프론트엔드 검색과 완벽 동기화하기 위해 타겟을 "items"로 통일!
                let target_table = match ctx_type {
                    "member" | "team" | "user" => "users",
                    "page" | "pages" => "pages",
                    "talk" => "talks",
                    _ => "items", // Shipping, Commerce, Sales 등 모든 메인 문서는 items 테이블에 있습니다.
                };

                let sql_filter = convert_conditions_to_sql(ctx);
                let emb = model.get_embedding(text.to_string()).await.unwrap_or(vec![0.0; 768]);
                
                let search_result = store.search_items(target_table, text, emb.clone(), 5, sql_filter.clone()).await;
                
                let final_results = match search_result {
                    Ok(res) => res,
                    Err(_) => {
                        store.search_items(target_table, text, emb, 5, None).await.unwrap_or_default()
                    }
                };

                for (id, content, score) in final_results {
                    all_results.push(json!({ "id": id, "text": content, "score": score, "context_type": ctx_type }));
                }
            }
        }
        Ok(json!({ "structured": structured_query, "results": all_results }))
    }.await; 

    IS_SEARCHING.store(false, Ordering::SeqCst);
    
    // 🌟 검색 종료 후 결과를 Message(히스토리)로 남기고 Task 상태를 변경
    if let Some(store) = store_opt.as_ref() {
        match &search_process {
            Ok(_) => { // 🌟 result 안 쓰므로 _ 로 변경
                let _ = store.update_task_status(&task_id, 9).await; 
                // 🌟 [CRITICAL FIX 6] 사용자의 원래 질문이 지워지지 않도록 텍스트 업데이트(Some(reply))를 None으로 바꿉니다!
                let _ = store.update_message_status(&task_id, 9, None).await;
            },
            Err(e) => {
                let status_code = if e.contains("cancelled") { 3 } else { 6 };
                let _ = store.update_task_status(&task_id, status_code).await;
                // 🌟 여기서도 텍스트 보호를 위해 None으로 처리
                let _ = store.update_message_status(&task_id, status_code, None).await;
            }
        }
    }

    model.deep_purge_resources().await; 
    
    search_process
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
        if let Ok(m) = LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
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
async fn check_active_task(
    state: State<'_, AppState>,
    payload: ActiveTaskQuery,
) -> Result<bool, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.has_active_task(&payload.cc, &payload.r#ref).await.map_err(|e| e.to_string())
    } else {
        Ok(false)
    }
}

#[tauri::command]
async fn connect_with_seed(_target_ip: String, _seed: u64) -> Result<(), String> {
    // [DEPRECATED] UDP 방식은 더 이상 사용하지 않으며, 프론트엔드에서 
    // 직접 send_signal_offer(TCP)를 호출하도록 변경되었습니다.
    Ok(())
}

#[tauri::command]
async fn start_listener_command(app_handle: tauri::AppHandle, seed: u64) -> Result<(), String> {
    crate::utils::network::start_signal_listener(app_handle, seed);
    println!("Signal Listener started on port 9999 with seed: {}", seed);
    Ok(())
}

#[tauri::command]
async fn send_signal_offer(target_ip: String, seed: u64, sdp: String) -> Result<String, String> {
    crate::utils::network::send_signal_offer(target_ip, seed, sdp).await
}

#[tauri::command]
async fn submit_signal_answer(target_ip: String, sdp: String) -> Result<(), String> {
    crate::utils::network::submit_signal_answer(target_ip, sdp).await
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
        // 1. 일반 메시지 쿼리 (프론트엔드에서 요청한 limit, offset 적용)
        let mut messages = db.get_all_messages(limit, offset, filter.clone()).await.unwrap_or_default();
        
        // 🌟 [근본 해결책] 2. limit에 밀려 잘려나가는 것을 방지하기 위해, 
        // 진행 중(1)이거나 대기 중(10)인 활성 Task는 DB에서 한 번 더 쿼리하여 무조건 포함시킵니다!
        let active_filter = if let Some(ref f) = filter {
            format!("({}) AND status IN (1, 10)", f)
        } else {
            "status IN (1, 10)".to_string()
        };

        if let Ok(active_msgs) = db.get_all_messages(50, 0, Some(active_filter)).await {
            for active_msg in active_msgs {
                let active_id = active_msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                
                // 중복 방지: 이미 일반 쿼리(1번) 결과에 포함되어 있지 않은 녀석만 배열에 쏙 끼워 넣습니다.
                if !messages.iter().any(|m| m.get("id").and_then(|v| v.as_str()).unwrap_or("") == active_id) {
                    messages.push(active_msg);
                }
            }
        }
        
        Ok(messages)
    } else { 
        Ok(vec![]) 
    }
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
    
    // 🌟 [CRITICAL FIX] 파일에 적힌 100% 확실한 과거 순서만 믿습니다! 
    // 메모리에 떠도는 최신 퍼센트를 강제로 끝에 끼워 넣으면 스텝 순서(stepMap)가 꼬입니다.
    if log_path.exists() {
        let content = std::fs::read_to_string(log_path).map_err(|e| e.to_string())?;
        Ok(content.lines().filter_map(|line| serde_json::from_str(line).ok()).collect())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn upsert_items(state: State<'_, AppState>, items: Vec<Value>) -> Result<String, String> {
    println!("[DEBUG] upsert_items called with {} items.", items.len());
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut count = 0;
        for item in items {
            println!("[DEBUG] Syncing item: {}", item);
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

#[tauri::command]
async fn save_mobile_temp_file(
    app_handle: tauri::AppHandle,
    filename: String,
    data: Vec<u8>,
) -> Result<String, String> {
    let temp_dir = app_handle.path().app_cache_dir().map_err(|e| e.to_string())?.join("mobile_uploads");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    
    let file_path = temp_dir.join(filename);
    std::fs::write(&file_path, data).map_err(|e| e.to_string())?;
    
    Ok(file_path.to_string_lossy().to_string())
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
            // [INIT] KV Bake Worker (Immediate)
            crate::models::qwen::generate::init_bake_worker();

            // [FIX] Reset stop signals immediately on app startup
            let setup_cancel = app.state::<AppState>().cancellation_token.clone();
            setup_cancel.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false);

            let setup_store = app.state::<AppState>().store.clone();
            tauri::async_runtime::spawn(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                let _ = std::fs::create_dir_all(db_path);
                if let Ok(s) = VectorStore::new(db_path).await {
                    println!("[Setup] VectorStore initialized.");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    
                    // [CRITICAL] Clear zombie tasks synchronously before the store is made available to other commands
                    let _ = s.cleanup_zombie_tasks().await;
                    
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
            let event_cancel = app.state::<AppState>().cancellation_token.clone();
            app.listen("new-task-from-browser", move |event| {
                // [NEW] Reset stop signals when a new task arrives
                event_cancel.store(false, Ordering::SeqCst);
                crate::utils::set_extraction_stop_signal(false);

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
                                &task.id, 
                                "system_task", 
                                &msg_text, 
                                Some(&task.id), 
                                Some(10), // 🌟 [수정] 오타(,,) 수정 및 Pending 상태(10) 명확히 적용
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
            upsert_items, set_ignore_cursor_events, mark_ui_ready, delete_document, delete_documents, delete_message, check_gpu_availability,
            save_mobile_temp_file, crate::utils::network::get_local_network_prefix, crate::utils::network::get_my_full_ip, connect_with_seed, start_listener_command, send_signal_offer, submit_signal_answer,
            get_active_task_context // 🌟 새로 추가됨
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
