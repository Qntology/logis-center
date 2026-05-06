mod model;
mod store;
mod automation;
pub mod parsing;
mod logic;
mod scheduler;
pub mod analytic; // 🌟 Analytic 전용 모듈 등록

pub mod models;
pub mod utils;
pub mod position_embed;
pub mod openai_types;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener, Emitter}; // 🌟 Emitter 추가!
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
// 브라우저가 실행되는 도중 상태를 방어하기 위한 락 추가
pub static IS_BROWSER_LAUNCHING: AtomicBool = AtomicBool::new(false);

// 🌟 [CRITICAL FIX] 상태 토글(깜빡임) 방어용 전역 상태 및 시간값 저장소
pub static CURRENT_BROWSER_STATE: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("stopped".to_string()));
pub static LAST_BROWSER_STATE_CHANGE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

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
                // 🌟 [수정] 존재하지 않는 cleanup_zombie_tasks 대신 통합된 cleanup_unfinished_tasks_on_startup 호출
                let _ = db.cleanup_unfinished_tasks_on_startup().await;
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

    // 함수 진입 시점에 즉시 락을 걸어 get_browser_status가 stopped를 반환하지 못하게 함
    crate::IS_BROWSER_LAUNCHING.store(true, Ordering::SeqCst);
    {
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = "running".to_string();
        crate::LAST_BROWSER_STATE_CHANGE.store(chrono::Utc::now().timestamp_millis(), Ordering::SeqCst);
    }
    
    let result = automation::run_browser_automation(browser, url, script, app_handle).await;

    // 실행 결과와 상관없이 포트가 열릴 때까지 기다리거나, 실패 시에도 IS_BROWSER_LAUNCHING은 유지됨
    for _ in 0..20 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // 충분한 안정화 시간을 거친 후 런칭 플래그 해제
    crate::IS_BROWSER_LAUNCHING.store(false, Ordering::SeqCst);

    result.map_err(|e| e.to_string())
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
    
    crate::IS_BROWSER_LAUNCHING.store(true, Ordering::SeqCst);
    
    let result = automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle).await;

    // 🌟 [CRITICAL FIX] 메모리 등재(is_some)는 즉시 이루어지므로 대기열을 바로 통과해버리는 버그를 차단합니다.
    // Error 반환과 관계없이, 크롬 프로세스 포트가 100% 물리적으로 응답(reachable)할 때까지 최대 10초간 대기합니다.
    for _ in 0..20 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    crate::IS_BROWSER_LAUNCHING.store(false, Ordering::SeqCst);

    result.map_err(|e| e.to_string())
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
        // 🌟 [CRITICAL FIX] 백그라운드 연산이 돌아가고 있을 때는 임베딩 모델 로드(VRAM 차지)를 원천 차단하고 텍스트 검색만 수행합니다.
        let is_task_active = crate::ACTIVE_TASK_MEM.read().unwrap().is_some();
        if is_task_active {
            println!("[DB-SEARCH] Background task is active. Skipping embedding model load to prevent VRAM overflow.");
            vec![0.0; 768]
        } else {
            let model_opt = { state.model.lock().await.as_ref().cloned() }; // 🌟 즉시 자물쇠 해제!
            if let Some(model) = model_opt {
                model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768])
            } else {
                vec![0.0; 768]
            }
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
    cc: String,
    bcc: String,
    ref_id: String,
) -> Result<Value, String> {
    // 🌟 [대격변] 프론트엔드(LocalStorage)에서 100% 입구를 통제하므로, 
    // 백엔드의 무거운 LanceDB 조회 및 락 대기열 로직을 전면 철거했습니다! (속도 대폭 향상)
    
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

    // 🌟 1. DB에 Task 및 Message 등록 (시간차를 두어 정렬 순서 물리적 고정)
    if let Some(store) = store_opt.as_ref() {
        let now = chrono::Utc::now().timestamp_millis();
        
        // Task 객체는 1ms 뒤로 설정
        let task = crate::store::Task {
            id: task_id.clone(), 
            r#type: "ai_search".to_string(),
            from: "user".to_string(),
            to: "system".to_string(),
            cc: cc.clone(), bcc: bcc.clone(), r#ref: ref_id.clone(), 
            data_json: json!({"query": query.clone(), "mode": search_mode.clone()}).to_string(),
            created_at: now + 1, updated_at: now + 1,
            status: 10, 
        };
        let _ = store.add_task(task).await;
        
        let from_user = "user".to_string();
        let to_system = "system".to_string();

        // 사용자 질문 메시지: 기준 시간(now)에 저장
        let user_msg_id = format!("{}_query", task_id);
        let now_str = now.to_string();
        let _ = store.add_message(
            &user_msg_id, "user", &query, 
            Some(&task_id), Some(9), 
            Some(&cc), Some(&bcc), Some(&ref_id), 
            Some(&from_user), Some(&to_system), Some("talk"), 
            Some(&now_str)
        ).await;

        // 시스템 작업 메시지: 질문보다 확실히 나중에 보이도록 50ms 뒤에 저장 (프론트엔드 복구 로직과 정렬 대칭)
        let next_now_str = (now + 50).to_string();
        let _ = store.add_message(
            &task_id, "system_task", "Task Started: AI Search", 
            Some(&task_id), Some(10), 
            Some(&cc), Some(&bcc), Some(&ref_id), 
            Some(&to_system), Some(&from_user), Some("talk"), 
            Some(&next_now_str)
        ).await;
    }

    // 🌟 [결단] 대기열 진입 시 프론트엔드에 'PENDING'임을 명시적으로 알립니다.
    let payload_pending = json!({ 
        "task_id": task_id, 
        "category": "Pending", // 🌟 Processing이 아닌 Pending 카테고리 사용
        "summary": "Waiting for AI Engine access...", 
        "spinner": "📥" 
    });
    let _ = app_handle.emit("extraction-progress", &payload_pending);
    
    emit_term("[QUEUE] Task queued. Waiting for Model Access...");
    
    let mut model_guard = state.model.lock().await;
    
    if cancel_token.load(Ordering::Relaxed) { 
        return Err("Task cancelled while waiting in queue".to_string()); 
    }

    emit_term("[QUEUE] AI Engine acquired. Starting process...");

    // [REMOVE] 백엔드 자체 검색 락 변수 조작 제거
    // 프론트엔드의 GlobalTaskManager가 이미 입구를 막고 있으므로 
    // 백엔드는 별도의 AtomicBool 락 없이 즉시 실행 로직에 집중합니다.

    {
        // 최소한의 동기화 정보만 업데이트 (UI 복구용)
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        let now = chrono::Utc::now().timestamp_millis();
        *mem_guard = Some(json!({
            "id": task_id.clone(),
            "status": 1
        }));
    }

    // 획득한 model_guard를 사용하여 모델 로드 또는 재사용
    let model = {
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

    // 🌟 2. 대기열(Pending) 통과 후 상태 업데이트
    if let Some(store) = store_opt.as_ref() {
        let _ = store.update_task_status(&task_id, 1).await; // 1: Processing
        
        // 시스템 말풍선 텍스트만 깔끔하게 변경합니다.
        let _ = store.update_message_status(&task_id, 1, Some("Analyzing semantic intent...")).await;
    }

    // 화면의 찌꺼기를 날려버리는 트리거(Processing) 발송!
    let payload_start = json!({ 
        "task_id": task_id, 
        "category": "Processing", 
        "summary": "AI Engine ready. Starting search...", 
        "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload_start);
    crate::scheduler::log_task_progress(&app_handle, &task_id, &payload_start);

    // 🌟 model_guard가 여기서 소멸되지 않고 이 아래 비동기 블록이 끝날 때까지 유지됩니다.
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
    
    if let Some(store) = store_opt.as_ref() {
        match &search_process {
            Ok(result_data) => { 
                let _ = store.update_task_status(&task_id, 9).await; 
                let _ = store.update_message_status(&task_id, 9, None).await;

                // 🌟 [수정] 프론트엔드 UI 렌더링을 위해 추출된 결과(result_data)를 페이로드에 담아 보냅니다.
                let payload_done = json!({ 
                    "task_id": task_id, 
                    "category": "Done", 
                    "summary": "AI Search Analysis Complete.", 
                    "spinner": "✅",
                    "data": result_data 
                });
                let _ = app_handle.emit("extraction-progress", &payload_done);
            },
            Err(e) => {
                let status_code = if e.contains("cancelled") { 3 } else { 6 };
                let _ = store.update_task_status(&task_id, status_code).await;
                
                // 🚨 [CRITICAL FIX] 에러 시에도 사용자의 쿼리를 덧붙이지 않고 깔끔하게 에러 사유만 표시합니다.
                let error_msg = format!("Task failed or cancelled: {}", e);
                let _ = store.update_message_status(&task_id, status_code, Some(&error_msg)).await;
            }
        }
    }

    // 🌟 [CRITICAL FIX] 검색 파이프라인 완료 후 메모리 캐시를 깔끔하게 비워줍니다.
    {
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        if let Some(mem) = mem_guard.as_ref() {
            if mem.get("id").and_then(|v| v.as_str()) == Some(task_id.as_str()) {
                *mem_guard = None;
            }
        }
    }

    model.deep_purge_resources().await; 
    
    // 🌟 함수 종료 직전 가드와 락을 명시적으로 정리
    drop(model_guard);
    IS_SEARCHING.store(false, Ordering::SeqCst);
    
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
    r#ref: String,
}

#[tauri::command]
async fn check_active_task(
    _state: State<'_, AppState>,
    payload: ActiveTaskQuery,
) -> Result<bool, String> {
    // 🌟 [CRITICAL FIX] LanceDB 대신 초고속 RAM 캐시인 ACTIVE_TASK_MEM만 확인합니다.
    if let Ok(mem_guard) = crate::ACTIVE_TASK_MEM.read() {
        if let Some(active) = mem_guard.as_ref() {
            let active_ref = active.get("ref").and_then(|v| v.as_str()).unwrap_or("");
            if active_ref == payload.r#ref {
                return Ok(true); // 현재 메모리에서 해당 페이지가 돌아가고 있음
            }
        }
    }
    Ok(false)
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
async fn get_browser_status() -> Result<Value, String> {
    let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
    
    // 1. 물리적 포트 응답 확인 및 메모리 가드 획득
    let reachable = automation::is_browser_reachable().await;
    let guard = automation::GLOBAL_BROWSER.lock().await; // 🌟 mut 제거
    
    // 🌟 [CRITICAL FIX] TcpStream 타임아웃 등 네트워크 지연으로 인한 오판독(reachable=false) 시 
    // 브라우저 객체를 강제로 None 처리해버리는 치명적 버그 삭제. 실제 종료 정리는 백그라운드 핸들러가 수행함.
    
    // 2. 현재 브라우저의 물리적 실행 여부 판별
    let is_running = is_launching || guard.is_some() || reachable;
    let target_status = if is_running { "running" } else { "stopped" };

    // 3. 브라우저 상세 상태(URL, 권한 등) 추출
    // 에러 해결: LAST_DETECTED_STATE에서 안전하게 변수를 복사해옵니다.
    let (detected_url, is_client, is_admin) = {
        let state = automation::LAST_DETECTED_STATE.lock().await;
        (state.url.clone(), state.is_client, state.is_admin)
    };

    // 4. 즉각적인 상태 반영 (플리커링의 원인이었던 3초 지연 로직 철거)
    let status = target_status.to_string();
    {
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = status.clone();
    }

    // 5. UI 버튼 숨김 여부 결정
    let hide_button = status == "running";

    Ok(json!({
        "status": status,
        "hide_button": hide_button,
        "url": detected_url,
        "is_client": is_client,
        "is_admin": is_admin
    }))
}

#[tauri::command]
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        // 🌟 [CRITICAL FIX] get_pending_tasks(1) 오타 수정: 진행 중인 작업(1)은 get_processing_tasks 로 명확히 가져옵니다.
        if let Ok(mut active) = db.get_processing_tasks(10).await {
            tasks.append(&mut active);
        }
        Ok(tasks)
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
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut count = 0;
        for item in items {
            // 🌟 [보강] id가 데이터 최상위에 없을 경우 data 객체 내부를 한 번 더 탐색하여 정합성을 확보합니다.
            let id = item.get("id").and_then(|v| v.as_str())
                        .or_else(|| item.get("data").and_then(|d| d.get("id")).and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();
            
            // 🌟 [CRITICAL FIX] 클라우드(index.ts) 로직 반영: type 문자열 무조건 공백제거 및 소문자 통일
            let type_str = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();

            // 🌟 [수정] 터미널을 도배하던 거대한 배열(Data Buffer) 로그 출력을 지우고 ID와 Type만 심플하게 남깁니다.
            println!("[DEBUG] Syncing item - ID: {}, Type: {}", id, type_str);
            
            // 🌟 [CRITICAL FIX] 세탁된 type_str을 원본 JSON(clean_item)에도 강제로 덮어씌웁니다.
            let mut clean_item = item.clone();
            if let Some(obj) = clean_item.as_object_mut() {
                obj.insert("type".to_string(), serde_json::json!(type_str));
            }
            
            // 🌟 [CRITICAL FIX] "talk" 타입의 데이터 구조를 프론트엔드 및 Cloud 백엔드의 표준 구조와 동일하게 강제 정규화합니다.
            if type_str == "talk" || type_str == "prompt" || type_str == "ai_search" {
                let text_val = clean_item.get("text")
                    .or_else(|| clean_item.get("query"))
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("text")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let link_val = clean_item.get("link")
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("link")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let origin_val = clean_item.get("origin")
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("origin")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("https://commerce.logis.center")
                    .to_string();

                // 기존 최상위 잔재 필드들을 깔끔하게 지웁니다.
                if let Some(obj) = clean_item.as_object_mut() {
                    obj.remove("text");
                    obj.remove("query");
                    obj.remove("link");
                    obj.remove("origin");
                    
                    // 프록시 서버(index.ts)와 동일하게 data 객체 안에 세 가지 필수 값을 몰아넣습니다.
                    obj.insert("data".to_string(), json!({
                        "text": text_val,
                        "link": link_val,
                        "origin": origin_val
                    }));
                }
            }

            // Determine table based on cleaned type
            let _table = match type_str.as_str() {
                "sales" | "goods" | "order" => "sales",
                "tracking" | "receiving" | "shipping" => "tracking",
                "event" | "coupon" => "event",
                "member" | "team" | "user" => "users",
                "talk" | "prompt" | "ai_search" => "talks", // 🌟 talk 관련 타입들을 명확히 talks 테이블로 라우팅
                _ => {
                    if clean_item.get("data").and_then(|d| d.get("origin")).is_some() {
                        "pages"
                    } else {
                        "items" 
                    }
                }
            };

            let final_table = if type_str == "team" || type_str == "user" || type_str == "member" {
                "users"
            } else if clean_item.get("data").and_then(|d| d.get("origin")).is_some() {
                "pages"
            } else {
                "items" 
            };

            // 🌟 [CRITICAL FIX] Move 에러 방지: clean_item 대신 원본 item을 사용하여 참조를 분리합니다.
            let from = item.get("from").and_then(|v| v.as_str());
            let to = item.get("to").and_then(|v| v.as_str());
            let cc = item.get("cc").and_then(|v| v.as_str());
            let bcc = item.get("bcc").and_then(|v| v.as_str());
            let r#ref = item.get("ref").and_then(|v| v.as_str());
            let digest = item.get("digest").and_then(|v| v.as_str());

            if !id.is_empty() {
                // 원본 item 대신 세탁된 clean_item을 DB에 밀어 넣습니다.
                let _ = db.upsert_item(final_table, &id, &type_str, clean_item, None, from, to, cc, bcc, r#ref, digest).await;
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
        let mut raw_tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        // 🌟 [CRITICAL FIX] limit=1 로 인해 다른 진행 중인 작업들이 증발하던 버그를 해결하고 전용 함수 사용
        if let Ok(mut active) = db.get_processing_tasks(10).await {
            raw_tasks.append(&mut active);
        }
        
        // 🌟 [핵심 변경] 현재 Rust 메모리에서 실제로 돌고 있는 유일한 태스크 ID 추출
        let mem_task_id = if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            mem.as_ref().and_then(|v| v.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string()
        } else { 
            "".to_string() 
        };

        for t in raw_tasks {
            // 🌟 1. DB엔 1(Processing)인데 Rust 메모리에 없다면? -> 진짜 좀비! 즉시 에러(error) 처리
            if t.status == 1 && t.id != mem_task_id {
                let error_status = crate::logic::parse_status("error");
                println!("[DB-SYNC] Zombie task detected in DB: {}. Marking as ERROR.", t.id);
                let _ = db.update_task_status(&t.id, error_status).await;
                let _ = db.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
            } 
            // 🌟 2. 진짜 돌고 있는 작업이거나 정상 대기열(10)인 경우만 프론트엔드로 전달
            else if t.status == 1 || t.status == 10 {
                tasks.push(t);
            }
        }

        pages = db.get_all_items("pages", 50, 0, None).await.unwrap_or_default();
        users = db.get_all_items("users", 20, 0, None).await.unwrap_or_default();
        items = db.get_all_items("items", 10, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
        let reachable = automation::is_browser_reachable().await;
        let guard = automation::GLOBAL_BROWSER.lock().await; // 🌟 mut 제거
        
        // 강제 메모리 해제 로직 제거
        
        let is_running = is_launching || guard.is_some() || reachable;
        let target_status = if is_running { "running" } else { "stopped" };

        // 지연 없이 물리적 상태를 즉시 동기화
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = target_status.to_string();
        current_state.clone()
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
            // 🌟 [CRITICAL FIX] 좀비 정리는 앱의 다른 기능이 시작되기 전에 '동기적'으로 완료되어야 합니다.
            // spawn 대신 block_on 계열의 처리를 통해 순서를 보장합니다.
            tauri::async_runtime::block_on(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                let _ = std::fs::create_dir_all(db_path);
                if let Ok(s) = VectorStore::new(db_path).await {
                    println!("[Setup] VectorStore initialized. Recovering zombie records...");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    
                    // 🌟 [핵심] 스케줄러 스레드가 생성되기 전에 DB를 먼저 정리합니다.
                    let _ = s.cleanup_unfinished_tasks_on_startup().await;
                    
                    let error_status = crate::logic::parse_status("error");
                    
                    if let Ok(processing_tasks) = s.get_processing_tasks(100).await {
                        for t in processing_tasks {
                            let _ = s.update_task_status(&t.id, error_status).await;
                            let _ = s.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
                        }
                    }
                    
                    if let Ok(pending_tasks) = s.get_pending_tasks(100).await {
                        for t in pending_tasks {
                            let _ = s.update_task_status(&t.id, error_status).await;
                            let _ = s.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
                        }
                    }
                    
                    *store_guard = Some(s);
                    println!("[Setup] Zombie cleanup complete. VectorStore is ready.");
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

            // 🌟 [CRITICAL FIX] Rust 백엔드에서 브라우저 상태를 주기적으로 감시하여 프론트엔드에 시그널을 보내는 전용 데몬 추가
            let status_monitor_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut last_status = String::new();
                loop {
                    let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
                    let reachable = automation::is_browser_reachable().await;
                    let guard = automation::GLOBAL_BROWSER.lock().await; // 🌟 mut 제거
                    
                    // 강제 메모리 해제 로직 제거
                    
                    let is_running = is_launching || guard.is_some() || reachable;
                    
                    // [수정] 런칭 중이거나 물리적으로 감지되었을 때는 무조건 running으로 고정합니다.
                    // 특히 is_launching이 true인 동안은 target_status가 절대로 stopped가 될 수 없습니다.
                    let target_status = if is_running { "running" } else { "stopped" };
                    
                    let current_status = {
                        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
                        *current_state = target_status.to_string();
                        current_state.clone()
                    };
                    
                    if current_status != last_status {                        
                        // 🌟 [CRITICAL FIX] 백그라운드 워커인지 사용자 화면인지 구분하기 위해 현재 상태를 조회합니다
                        let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
                        let (is_client, is_admin, url) = {
                            let state = automation::LAST_DETECTED_STATE.lock().await;
                            (state.is_client, state.is_admin, state.url.clone())
                        };
                        // [수정] 상태 모니터에서도 브라우저가 실행 중(running)이라면 새 탭(빈 URL)이든 아니든 무조건 버튼을 숨겨 
                        // 프론트엔드로 hide_button: false가 날아가 깜빡이는 현상을 원천 차단합니다.
                        let hide_button = current_status == "running";
                        
                        use tauri::Emitter;
                        // [수정] 이벤트 페이로드를 생성할 때, 런칭 중(is_launching)이라면 
                        // URL 감지 결과와 상관없이 hide_button을 무조건 true로 고정하여 발송합니다.
                        let payload = json!({
                            "status": current_status.clone(),
                            "hide_button": if is_launching { true } else { hide_button }
                        });
                        let _ = status_monitor_handle.emit("browser-status", payload);
                        last_status = current_status;
                    }
                    // 빠른 UI 복구를 위해 1초 간격으로 감시 속도 단축
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            });

            let event_store = app.state::<AppState>().store.clone();
            let event_cancel = app.state::<AppState>().cancellation_token.clone();
            // 🌟 [수정] 핸들러 내부에서 이벤트를 쏘기 위해 app_handle을 획득합니다.
            let handle_for_event = app.handle().clone(); 

            app.listen("new-task-from-browser", move |event| {
                event_cancel.store(false, Ordering::SeqCst);
                crate::utils::set_extraction_stop_signal(false);
                let app_handle = handle_for_event.clone(); // 🌟 [수정] 클로저 내부에서 사용할 이름 정의

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
                                &task.id, "system_task", &msg_text, Some(&task.id), Some(10),
                                Some(&task.cc), Some(&task.bcc), Some(&task.r#ref),
                                Some(&task.from), Some(&task.to), Some("talk"), None
                            ).await;
                            
                            let _ = db.add_task(task.clone()).await;
                            
                            // 🌟 [수정] 유효한 app_handle을 사용하여 이벤트를 발송합니다.
                            let _ = app_handle.emit("task-db-registered", json!({
                                "task_id": task.id,
                                "status": task.status,
                                "created_at": task.created_at,
                                "text": msg_text
                            }));
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
