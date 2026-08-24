mod model;
mod store;
mod automation;
pub use utils::parsing;
pub use utils::bias_schema;
pub use utils::json_parse;
pub use utils::nl_convert;
pub use utils::time_guide;
mod logic;
mod scheduler;
pub mod analytic;
pub mod stanza;
pub mod js_templates;
pub mod prompts; // 🌟 새로 추가된 프롬프트 모듈 선언
pub mod parsers; // 🌟 문서 파서 모듈 선언
pub mod models;
pub mod utils;
pub mod position_embed;
pub mod openai_types;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener, Emitter}; 
use tokio::sync::Mutex as TokioMutex;
use std::sync::RwLock; 
use once_cell::sync::Lazy; 
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};


static IS_SEARCHING: AtomicBool = AtomicBool::new(false);
// 브라우저가 실행되는 도중 상태를 방어하기 위한 락 추가
pub static IS_BROWSER_LAUNCHING: AtomicBool = AtomicBool::new(false);


pub static CURRENT_BROWSER_STATE: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("stopped".to_string()));
pub static LAST_BROWSER_STATE_CHANGE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

pub static ACTIVE_TASK_MEM: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));
pub static CURRENT_UI_CATEGORY: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::from("Processing")));

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
    
    // lock().await를 사용하여 스케줄러의 DB 작업이 끝날 때까지 찰나를 기다린 후 100% 확실하게 지워버립니다.
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        if let Some(ref id) = task_id {
            let _ = db.update_task_status(id, crate::logic::parse_status("cancel")).await;
            let _ = db.delete_message_by_task_id(id).await;
            
            // [CLEANUP] 작업 취소 시 해당 세션의 무거운 KV 캐시와 임시 데이터 폴더를 즉각 삭제하여 디스크 용량을 확보합니다.
            let kv_dir = crate::utils::paths::get_kv_dir(None).join(id);
            let base_kv_dir = crate::utils::paths::get_kv_dir(None).join(format!("{}_base", id));
            let task_data_dir = crate::utils::paths::get_task_specific_dir(None, id);
            let pug_log_dir = crate::utils::paths::get_pug_logs_dir(None, id);

            let _ = std::fs::remove_dir_all(&kv_dir);
            let _ = std::fs::remove_dir_all(&base_kv_dir);
            let _ = std::fs::remove_dir_all(&task_data_dir);
            let _ = std::fs::remove_dir_all(&pug_log_dir);

            println!("[STOP] Task {} cleared from DB and temporary files deleted.", id);
        } else {
            let _ = db.cleanup_unfinished_tasks_on_startup().await;
            println!("[STOP] All pending tasks cleared from DB.");
            // 🌟 [LOGOUT SCOPE RESET] 로그아웃 시 ACTIVE_TASK_MEM 만 정리합니다.
            //    LanceDB 문서 자체는 삭제하지 않습니다.
            //    (재로그인 시 migrate_team_identity 가 to 필드만 갱신)
            //    만약 완전 초기화가 필요하면 프론트엔드에서 reset_lancedb 를 호출합니다.
        }
    }
    drop(store_guard); // 다음 단계(Model Clear) 진행을 위해 즉시 락 해제

    // 3. Try to clear model
    if let Ok(mut model_guard) = state.model.try_lock() {
        if let Some(m) = model_guard.as_ref() {
            m.deep_purge_resources().await; // 모델 메모리 및 VRAM 캐시도 확실하게 강제 파기
        }
        *model_guard = None;
    }

    
    // 백엔드의 현재 작업 캐시를 즉시 비워야 프론트엔드의 #btn-extract 버튼이 정상적으로 부활합니다.
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = None;
    }

    Ok("Stop signal sent and resources cleaned.".to_string())
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
    
    if IS_SEARCHING.load(Ordering::SeqCst) {
        println!("[UNLOAD] AI Search is active. Skipping unload to prevent deadlock.");
        return Ok("Search active. Memory kept.".to_string());
    }

    {
        let mut model_guard = state.model.lock().await;
        if let Some(m) = model_guard.as_ref() {
            m.deep_purge_resources().await;
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

// =====================================================================
// 🌟 [TRANSLIT CACHE / RESPOND] 프론트엔드 → Rust 음차 캐시 응답 수신
// ---------------------------------------------------------------------
//  프론트엔드가 Dexie 를 조회한 뒤 invoke("translit_cache_respond") 로
//  결과를 보내면, 여기서 oneshot sender 를 통해 scheduler 에 전달합니다.
// =====================================================================
#[tauri::command]
async fn translit_cache_respond(
    request_id: String,
    results: Vec<(String, String)>,
) -> Result<(), String> {
    if let Some(sender) = crate::scheduler::TRANSLIT_PENDING.lock().unwrap().remove(&request_id) {
        let _ = sender.send(results);
    }
    Ok(())
}

// =====================================================================
// 🌟 [TRANSLIT CACHE / EMBEDDING BATCH] 복수 음차 후보 코사인 계산용
// ---------------------------------------------------------------------
//  프론트엔드가 복수 후보의 임베딩을 한 번에 요청할 때 사용합니다.
//  get_embedding_batch 를 Tauri command 로 노출합니다.
// =====================================================================
#[tauri::command]
async fn get_embedding_batch_for_translit(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, String> {
    let model = {
        let mut model_guard = state.model.lock().await;
        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), None).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }
        model_guard.as_ref().unwrap().clone()
    };
    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    model.ensure_embedding().await.map_err(|e| e.to_string())?;
    model.get_embedding_batch(texts).await.map_err(|e| e.to_string())
}

// =====================================================================
// 🌟 [CLIENT-SIDE EMBEDDING] Cloud AI 모드에서도 임베딩은 무조건 로컬에서 수행합니다.
//    ① get_query_embedding      : 클라우드 검색용 질의 벡터를 로컬 모델로 생성
//    ② reindex_pending_embeddings : 클라우드가 내려준 아이템을 로컬에서 벡터화 + 청크 인덱싱
// =====================================================================
#[tauri::command]
async fn get_query_embedding(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    text: String,
    device_preference: Option<String>,
) -> Result<Vec<f32>, String> {
    let model = {
        let mut model_guard = state.model.lock().await;

        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }

        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }

        model_guard.as_ref().unwrap().clone()
    };

    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    model.ensure_embedding().await.map_err(|e| e.to_string())?;

    let vec = model.get_embedding(text).await.map_err(|e| e.to_string())?;

    println!("[EMBED-LOCAL] Query embedding generated locally. dim = {}", vec.len());

    Ok(vec)
}

#[tauri::command]
async fn reindex_pending_embeddings(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    limit: Option<usize>,
    device_preference: Option<String>,
    // 🌟 [MODE ROUTING] commerce / shipping / analytic 트랙을 구분합니다.
    //    analytic 은 console.logis.center, 나머지는 commerce.logis.center 로 벡터가 전송됩니다.
    mode: Option<String>,
) -> Result<Value, String> {
    // 진행 중인 로컬 작업이 있으면 VRAM 충돌 방지를 위해 즉시 반환합니다.
    if IS_SEARCHING.load(Ordering::SeqCst) {
        return Ok(json!({ "processed": 0, "vectors": [], "skipped": "searching" }));
    }
    if crate::ACTIVE_TASK_MEM.read().unwrap().is_some() {
        return Ok(json!({ "processed": 0, "vectors": [], "skipped": "busy" }));
    }

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            let _ = std::fs::create_dir_all(&db_path);
            if let Ok(s) = VectorStore::new(&db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    };

    let store = match store_opt {
        Some(s) => s,
        None => return Err("DB not initialized".to_string()),
    };

    let scan_limit = limit.unwrap_or(20);
    // 🌟 [MODE ROUTING] 트랙별로 격리된 문서만 스캔합니다.
    //    (analytic 트랙 문서는 syncAnalyticsData 가 mode='analytic' 으로 태깅해 저장합니다)
    let target_mode = mode.unwrap_or_else(|| "commerce".to_string());
    let mode_filter = format!("mode = '{}'", target_mode);
    // 🌟 [SCAN STARVATION FIX]
    //  ── 무엇이 문제였나 ──
    //   기존에는 offset 0 에서 500건만 읽었습니다. 정렬 기준이 고정되어 있으므로
    //   상위 500건이 전부 임베딩 완료 상태가 되면 pending 이 비고 no_pending 으로 끝나,
    //   501번째 이후의 미처리 문서는 영원히 처리되지 않았습니다.
    //   문서가 500건을 넘는 순간 로컬 임베딩이 조용히 멈춥니다.
    //   scan_limit 만큼 채울 때까지 오프셋을 밀며 순회합니다.
    const SCAN_PAGE: usize = 500;
    const SCAN_MAX_PAGES: usize = 40; // 최대 20,000건까지 탐색
    let mut docs: Vec<TradeDocument> = Vec::new();
    for page in 0..SCAN_MAX_PAGES {
        if state.cancellation_token.load(Ordering::Relaxed) { break; }
        let batch = store
            .get_all_items("items", SCAN_PAGE, page * SCAN_PAGE, Some(mode_filter.clone()))
            .await
            .map_err(|e| e.to_string())?;
        let fetched = batch.len();
        docs.extend(batch);
        if fetched < SCAN_PAGE { break; }
        // 이번 페이지까지의 후보로 scan_limit 을 채울 수 있으면 더 읽지 않습니다.
        let rough_pending = docs.iter().filter(|d| !d.id.is_empty()).count();
        if rough_pending >= scan_limit * 20 { break; }
    }

    // 🌟 [LAZY MODEL LOAD] 대상 선별을 '모델 로드 이전' 으로 끌어올립니다.
    //    기존 구조는 LogisModel::new → ensure_embedding 을 먼저 수행한 뒤 스캔했기 때문에,
    //    처리할 문서가 0건이어도 CUDA 컨텍스트와 97M 임베딩 가중치를 매번 올렸다 내렸습니다.
    //    (로그 실측: [EMBED-LOCAL] 처리 로그가 없는데도 Loading Embedding Model 발생)
    //    이 스캔 구간은 LanceDB 조회만 사용하므로 모델이 전혀 필요 없습니다.
    let mut pending: Vec<TradeDocument> = Vec::new();
    for doc in docs {
        if pending.len() >= scan_limit { break; }
        if state.cancellation_token.load(Ordering::Relaxed) { break; }
        if doc.id.is_empty() { continue; }
        // 🌟 question / answer 는 관리자 채팅 말풍선 전용 타입입니다.
        //    upsert_items 가 items 로 라우팅하는데 제외 목록에는 없어 벡터화되고 있었고,
        //    parse_analytic_query 는 이 두 타입을 검색 스코프에서 제외하므로
        //    만들기만 하고 한 번도 쓰이지 않는 벡터였습니다.
        // 🌟 [EMBED TYPE GUARD] 검색 벡터화가 무의미한 타입만 배제합니다.
        //    pages/talk/prompt/ai_search 는 검색 대상이 아니고,
        //    question/answer 는 parse_analytic_query 가 검색 스코프에서 제외하며,
        //    team/user/member 는 통계 문서라 임베딩 대상이 아닙니다.
        //
        //    🌟 [ANALYTICS ALLOW]
        //     ── 무엇이 문제였나 ──
        //      click / hover 를 이 목록에 넣고, 바로 아래에서 report 까지 별도로 건너뛰어
        //      analytics 4종(click/hover/change/report) 중 change 하나만 벡터화되었습니다.
        //      나머지 3종은 vector 컬럼이 0벡터로 남아 search_items 의 ANN 트랙이
        //      의미 없는 비교를 하게 되고, item_chunks 도 없어 STAGE-4 가 통째로 비었습니다.
        //      "D1 에서 이벤트를 받아와도 검색이 아무것도 못 찾는" 상태의 직접 원인입니다.
        //     ── 비용 ──
        //      analytics 도메인 타입은 get_detail_schema_fields 에 스키마가 없어
        //      index_item_chunks 가 조기 종료됩니다. 즉 추가 비용은 '문서 벡터 1개' 뿐입니다.
        const EMBED_EXCLUDE_TYPES: [&str; 10] = [
            "pages", "page", "talk", "prompt", "ai_search",
            "question", "answer", "team", "user", "member",
        ];
        if EMBED_EXCLUDE_TYPES.iter().any(|t| doc.r#type == *t) {
            continue;
        }
        // 🌟 [CHUNK COUNT FIRST] 청크가 이미 존재하면 로컬 임베딩이 완료된 것이므로
        //    가장 먼저 탈락시킵니다. chunk_count는 물리적 사실이라
        //    embed 플래그(소프트 마커)보다 신뢰도가 높습니다.
        let chunk_count = store.count_chunks_by_item(&doc.id).await.unwrap_or(0);
        if chunk_count > 0 { continue; }
        // 🌟 [EMBED FLAG CHECK] data 내부의 embed 플래그를 보조 확인합니다.
        //    chunk_count가 0이지만 embed가 1인 경우(청크 삭제 후 재인덱싱 대기 등)
        //    불필요한 재처리를 방지합니다.
        if let Ok(data_val) = serde_json::from_str::<Value>(&doc.json_data) {
            // 🌟 [PAGE CACHE GUARD] 페이지 셀렉터 캐시는 type 이 도메인 타입(tracking/goods/...)
            //    이라서 EMBED_EXCLUDE_TYPES 문자열 목록으로는 절대 잡히지 않습니다.
            //    (서버 index.ts 의 home 문서 = { table:'pages', type:'tracking', data:{node,item} })
            //    그래서 '구조 마커' 로 판정합니다. 셀렉터 캐시는 검색 대상이 아니므로
            //    임베딩도, 청크 인덱싱도, 음차도 전부 불필요합니다.
            let is_page_cache = data_val.get("table")
                    .and_then(|v| v.as_str())
                    .map_or(false, |t| t == "pages" || t == "page")
                || data_val.get("node").is_some()
                || data_val.get("item").is_some();
            if is_page_cache {
                println!(
                    "[EMBED-LOCAL] ⏭️ 페이지 셀렉터 캐시 문서 '{}' (type='{}') 는 검색 대상이 아니므로 임베딩을 건너뜁니다.",
                    doc.id, doc.r#type
                );
                continue;
            }

            let already = data_val.get("embed")
                .map(|v| v.as_i64().unwrap_or(0) == 1 || v.as_bool().unwrap_or(false))
                .unwrap_or(false);
            if already {
                continue; // 이미 임베딩 완료된 아이템
            }
        }
        pending.push(doc);
    }

    // 🌟 [VRAM GUARD] 처리 대상이 없으면 모델을 만들지 않고 즉시 반환합니다.
    if pending.is_empty() {
        return Ok(json!({ "processed": 0, "vectors": [], "mode": target_mode, "skipped": "no_pending" }));
    }

    println!("[EMBED-LOCAL] {} pending item(s) detected. Loading embedding model...", pending.len());

    let model = {
        let mut model_guard = state.model.lock().await;

        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }

        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }

        model_guard.as_ref().unwrap().clone()
    };

    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    model.ensure_embedding().await.map_err(|e| e.to_string())?;

    let mut processed = 0usize;
    for doc in pending {
        if state.cancellation_token.load(Ordering::Relaxed) { break; }
        let mut data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
        // 🌟 [BLOB DECODE GUARD] json_data 내부의 "data" 키가 아직
        //    base64(gzip) 문자열로 남아 있으면 해제합니다.
        //    upsert_items 가 이미 처리하지만, 구버전 저장 데이터나
        //    직접 LanceDB 에 삽입된 행에는 미처리 상태가 있을 수 있습니다.
        if let Some(blob_b64) = data.get("data").and_then(|v| v.as_str()) {
            if blob_b64.len() > 50 {
                use base64::prelude::BASE64_STANDARD;
                use base64::Engine;
                if let Ok(decoded) = BASE64_STANDARD.decode(blob_b64) {
                    if let Ok(decompressed) = crate::utils::compression::decompress_to_value(&decoded) {
                        // 해제된 내용을 현재 data 에 병합합니다.
                        if let (Some(base_obj), Some(inner_obj)) = (data.as_object_mut(), decompressed.as_object()) {
                            for (k, v) in inner_obj {
                                if !base_obj.contains_key(k) {
                                    base_obj.insert(k.clone(), v.clone());
                                }
                            }
                            base_obj.remove("data");
                        }
                    }
                }
            }
        }
        // 🌟 [ANALYTICS] Cron Worker 가 구조화한 'action'(사용자 의도 문장)이 벡터의 본체입니다.
        //    action 이 없으면 summary → cross_action_flow → intent_evolution 순으로 폴백합니다.
        //
        //    🌟 [RAW ARRAY GUARD] content.js 가 처음 올린 원시 이벤트의 action 은
        //      ["<div class=...>...</div>"] 형태의 '배열' 입니다.
        //      as_str() 이 None 을 돌려주므로 값 자체는 무해했지만,
        //      아래 폴백 체인 끝의 json_to_natural_language 가 relate(HTML 배열)까지
        //      통째로 펼쳐 벡터에 HTML 덩어리를 밀어 넣습니다.
        //      Worker 의 GET 은 updated_at > 0 인 구조화 완료 행만 내려주므로
        //      정상 경로에서는 배열이 오지 않지만, 구버전 잔재를 위해 명시적으로 배제합니다.
        let analytic_text = data.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        // 🌟 [EMPTY-AWARE FALLBACK] Option 의 or_else 는 None 일 때만 발화합니다.
        //    summary 키가 존재하되 빈 문자열이면 Some("") 이 되어
        //    cross_action_flow 폴백이 영원히 도달하지 않았습니다.
        let pick = |k: &str| -> Option<String> {
            data.get(k)
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let analytic_fallback = pick("summary")
            .or_else(|| pick("cross_action_flow"))
            .or_else(|| pick("intent_evolution"))
            .unwrap_or_default();
        // 🌟 [RAW EVENT DETECT] 구조화 전 원시 이벤트인지 판정합니다.
        //    action 이 배열이거나 relate 가 배열이면 Cron Worker 가 아직 손대지 않은 행입니다.
        let is_raw_event = data.get("action").map_or(false, |v| v.is_array())
            || data.get("relate").map_or(false, |v| v.is_array());

        let text = if !analytic_text.is_empty() {
            analytic_text
        } else if !doc.text.trim().is_empty() {
            doc.text.clone()
        } else if !analytic_fallback.is_empty() {
            analytic_fallback
        } else if doc.mode == "analytic" || is_raw_event {
            // 🌟 [RAW GUARD] analytics 문서에 구조화 문장이 하나도 없으면
            //    json_to_natural_language 가 relate(원시 outerHTML 배열)를 전부 펼쳐
            //    수천 토큰짜리 HTML 을 임베딩 입력으로 만듭니다.
            //    그 벡터는 어떤 질의와도 유사하지 않고, 저장 용량만 잡아먹으며,
            //    embed=1 로 마킹되어 Cron Worker 가 나중에 구조화해도 재인덱싱되지 않습니다.
            //    Cron 구조화 이후 updated_at 이 갱신되면 다시 후보로 잡히도록 여기서 건너뜁니다.
            println!(
                "[EMBED-LOCAL] ⏭️ analytics 문서 '{}' (type='{}') 는 아직 구조화 문장(action/summary/cross_action_flow)이 없어 임베딩을 보류합니다.",
                doc.id, doc.r#type
            );
            continue;
        } else {
            crate::parsing::json_to_natural_language(&data)
        };

        if text.trim().is_empty() { continue; }

        let emb = model.get_embedding(text.clone()).await.map_err(|e| e.to_string())?;

        if let Some(o) = data.as_object_mut() {
            o.insert("text".to_string(), json!(text.clone()));
            if !o.contains_key("masked_text") {
                o.insert("masked_text".to_string(), json!(text.clone()));
            }
            o.insert("embed".to_string(), json!(1));
            // 🌟 [UPDATED_AT PRESERVE] reindex는 데이터 변경이 아니라
            //    임베딩 마킹이므로 updated_at을 현재 시각으로 갱신하지 않습니다.
            //    기존 updated_at이 없으면 0으로 두어 upsert_item의
            //    digest 기반 스킵 로직이 정상 동작하도록 합니다.
            if !o.contains_key("updated_at") {
                o.insert("updated_at".to_string(), json!(doc.updated_at_ts));
            }
        }

        // 🌟 [v4] sales / tracking / event 물리 테이블이 사라졌으므로
        //    '도메인 테이블 + items 미러' 이중 upsert 를 단일 upsert 로 접습니다.
        //    (쓰기 횟수가 절반으로 줄고, 두 테이블의 내용이 어긋날 여지가 사라집니다)
        let target_table = match doc.r#type.as_str() {
            "member" | "team" | "user" => "users",
            "pages" | "page" => "pages",
            _ => "items",
        };

        let digest = crate::utils::hash::digest(&text);

        let _ = store.upsert_item(
            target_table, &doc.id, &doc.r#type, data.clone(), Some(emb.clone()),
            Some(&doc.from), Some(&doc.to), Some(&doc.cc), Some(&doc.bcc), Some(&doc.r#ref), Some(&digest)
        ).await;

        let link = data.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let doc_lang = crate::utils::lang_utils::detect_document_language(&text);
        // 🌟 v4 : mode 는 봉투 물리 컬럼이므로 doc.mode 가 항상 채워져 있습니다.
        //    다만 구버전 데이터를 대비해 data.mode 까지 폴백을 둡니다.
        let mode = if !doc.mode.trim().is_empty() {
            doc.mode.clone()
        } else {
            data.get("mode").and_then(|v| v.as_str()).unwrap_or("commerce").to_string()
        };

        let cancel = state.cancellation_token.clone();
        let _ = crate::scheduler::index_item_chunks(
            &store, &model, &doc.id, &doc.r#type, &doc_lang, &data, true,
            &doc.cc, &doc.bcc, &doc.r#ref, &mode, &link, &cancel, &app_handle, "cloud_sync"
        ).await;

        processed += 1;
    }

    if processed > 0 {
        println!("[EMBED-LOCAL] Cloud-synced items embedded locally: {} item(s). (mode: {})", processed, target_mode);
    }

    model.unload_embedding().await;

    Ok(json!({ "processed": processed, "mode": target_mode }))
}

// =====================================================================
// 🌟 [ANALYTIC STRUCTURING ENTRY] D1 → LanceDB 로 내려온 원시 행동 로그를
//    HTML → PUG → 속성 제거 → Qwen3.5 2B 요약 으로 확정합니다.
//  ── 호출 시점 ──
//   main.ts 의 syncAnalyticsData 직후. 임베딩(reindex_pending_embeddings)보다
//   반드시 먼저 실행되어야 합니다. 텍스트가 없으면 임베딩 경로가
//   RAW GUARD 로 그 문서를 건너뛰기 때문입니다.
// =====================================================================
#[tauri::command]
async fn structure_pending_analytics(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    limit: Option<usize>,
    device_preference: Option<String>,
) -> Result<Value, String> {
    if IS_SEARCHING.load(Ordering::SeqCst) {
        return Ok(json!({ "processed": 0, "skipped": "searching" }));
    }
    if crate::ACTIVE_TASK_MEM.read().unwrap().is_some() {
        return Ok(json!({ "processed": 0, "skipped": "busy" }));
    }
    if state.cancellation_token.load(Ordering::Relaxed) {
        return Ok(json!({ "processed": 0, "skipped": "cancelled" }));
    }

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            let _ = std::fs::create_dir_all(&db_path);
            if let Ok(s) = VectorStore::new(&db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    };

    let store = match store_opt {
        Some(s) => s,
        None => return Err("DB not initialized".to_string()),
    };

    // 🌟 [LAZY MODEL LOAD] 대상이 0건이면 Qwen3.5 2B 를 아예 올리지 않습니다.
    //    (LanceDB 조회 1회로 끝나므로 유휴 시 부담이 사실상 없습니다)
    let type_list = crate::analytic::ANALYTIC_EVENT_TYPES
        .iter()
        .map(|t| format!("'{}'", t))
        .collect::<Vec<_>>()
        .join(", ");
    let probe_filter = format!(
        "mode = 'analytic' AND updated_at = 0 AND type IN ({})",
        type_list
    );

    let probe = store
        .get_all_items("items", 1, 0, Some(probe_filter))
        .await
        .map_err(|e| e.to_string())?;

    if probe.is_empty() {
        return Ok(json!({ "processed": 0, "skipped": "no_pending" }));
    }

    println!("[ANALYTIC] Pending raw behaviour event detected. Loading Qwen3.5(2B) for structuring...");

    let model = {
        let mut model_guard = state.model.lock().await;

        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }

        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }

        model_guard.as_ref().unwrap().clone()
    };

    let processed = crate::analytic::run_analytic_structuring(
        &store,
        &model,
        &state.cancellation_token,
        &app_handle,
        "analytic_sync",
        limit.unwrap_or(20),
    ).await.map_err(|e| e.to_string())?;

    model.deep_purge_resources().await;

    Ok(json!({ "processed": processed }))
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
    state: State<'_, AppState>, 
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    
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

    // 🌟 [CRITICAL FIX] 500ms 단위의 느린 대기를 50ms 단위의 초고속 폴링으로 변경하여 브라우저가 준비되는 즉시 락을 해제합니다.
    for _ in 0..100 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // 충분한 안정화 시간을 거친 후 런칭 플래그 해제
    crate::IS_BROWSER_LAUNCHING.store(false, Ordering::SeqCst);

    result.map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    state: State<'_, AppState>, 
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    
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

    
    // 🌟 [CRITICAL FIX] 500ms 단위의 느린 대기를 50ms 단위의 초고속 폴링으로 변경하여 브라우저가 준비되는 즉시 락을 해제합니다.
    for _ in 0..100 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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

// 🌟 [SCOPE SANITIZE] 프론트엔드가 넘긴 필터에서 봉투 컬럼 술어만 남깁니다.
//  v4 이전 프론트엔드는 status / amount 같은 도메인 컬럼을 SQL 로 보냈는데,
//  그 컬럼들이 물리적으로 사라졌으므로 그대로 내려보내면 DataFusion 이 통째로 실패합니다.
//  (실패 시 필터가 초기화되어 타 팀 데이터가 새어 나오는 보안 문제로 이어집니다)
//  따라서 여기서 화이트리스트로 걸러 '스코프만' 남기고, 도메인 술어는 조용히 버립니다.
//  버려진 조건은 프론트엔드 Dexie 가 data.* 로 다시 적용합니다.
fn sanitize_scope_filter(filter: Option<String>) -> Option<String> {
    const ENVELOPE_COLS: [&str; 9] = [
        "id", "type", "flag", "from", "to", "cc", "bcc", "ref", "mode",
    ];
    const TIME_COLS: [&str; 2] = ["created_at", "updated_at"];

    let raw = match filter {
        Some(f) if !f.trim().is_empty() => f,
        _ => return None,
    };

    let mut kept: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    for clause in raw.split(" AND ") {
        let c = clause.trim().trim_start_matches('(').trim_end_matches(')').trim();
        if c.is_empty() { continue; }

        // 술어의 좌변 컬럼명을 추출합니다. (백틱/공백/연산자 제거)
        let lhs: String = c
            .split(|ch: char| ch == '=' || ch == '<' || ch == '>' || ch == ' ')
            .next()
            .unwrap_or("")
            .trim_matches('`')
            .trim()
            .to_lowercase();

        let is_envelope = ENVELOPE_COLS.iter().any(|e| *e == lhs)
            || TIME_COLS.iter().any(|e| *e == lhs);

        if is_envelope {
            kept.push(c.to_string());
        } else {
            dropped.push(c.to_string());
        }
    }

    if !dropped.is_empty() {
        println!(
            "[SCOPE SANITIZE] 봉투 컬럼이 아닌 술어 {}개를 LanceDB 필터에서 제외했습니다(Dexie 위임): {:?}",
            dropped.len(),
            dropped.iter().take(6).collect::<Vec<_>>()
        );
    }

    if kept.is_empty() { None } else { Some(kept.join(" AND ")) }
}

#[tauri::command]
async fn search_documents(
    state: State<'_, AppState>,
    query: String,
    limit: usize,
    offset: usize, 
    filter: Option<String>,
) -> Result<Vec<(String, String, f32)>, String> {
    
    println!("[DB-SEARCH] 텍스트 검색 요청 수신 (Query: '{}', Filter: {:?})", query, filter);

    // 🌟 v4 : 도메인 술어를 걷어내고 스코프만 남깁니다.
    let scope = sanitize_scope_filter(filter);

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            if let Ok(s) = VectorStore::new(&db_path).await {
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    }; 
    
    
    let query_vec = if !query.trim().is_empty() {
        
        let is_task_active = crate::ACTIVE_TASK_MEM.read().unwrap().is_some();
        if is_task_active {
            println!("[DB-SEARCH] Background task is active. Skipping embedding model load to prevent VRAM overflow.");
            vec![0.0; 384]
        } else {
            let model_opt = { state.model.lock().await.as_ref().cloned() }; 
            if let Some(model) = model_opt {
                model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384])
            } else {
                vec![0.0; 384]
            }
        }
    } else {
        vec![0.0; 384]
    };

    if let Some(store) = store_opt {
        
        let search_result = store.search_items("items", &query, query_vec, limit, offset, scope, false).await.map_err(|e| e.to_string());
        
        
        match &search_result {
            Ok(items) => {
                println!("[DB-SEARCH] 검색 완료: {}건 반환", items.len());
                for (i, (id, text, score)) in items.iter().enumerate() {
                    // 터미널 가독성을 위해 줄바꿈 문자를 공백으로 치환하여 한 줄로 출력합니다.
                    println!("  ↳ {}. ID: [{}] | Score: {:.4} | Text: {}", i + 1, id, score, text.replace("\n", " "));
                }
            },
            Err(e) => println!("[DB-SEARCH] 검색 실패: {}", e),
        }
        
        search_result
    } else {
        Err("DB not initialized".to_string())
    }
}

// =====================================================================
// 🌟 [QUERY ROUTER v4] 조건을 '스코프'와 '정밀 필터'로 물리적으로 분리합니다.
// ---------------------------------------------------------------------
//  기존 convert_conditions_to_sql 은 하나의 함수가 두 역할을 겸하면서
//  valid_cols 화이트리스트(7개) 밖의 조건을 전부 '버리고' 있었습니다.
//  그 손실을 메우려고 STAGE-3 이 A/FULL ~ E/TABLE-FALLBACK 5개 티어를 발행했고,
//  그래도 못 메운 부분은 tracking_sql 의 LIKE '%..%' 로 땜질했습니다.
//
//  v4 구조:
//    build_scope_filter → LanceDB : 봉투 컬럼만. 조건을 '절대' 버리지 않음(애초에 안 받음)
//    build_dexie_plan   → Dexie   : 도메인 조건 전량. 버려지는 조건이 0개
//
//  → 조건이 버려지지 않으므로 보험 티어(A~E)가 불필요해지고,
//    새 도메인 필드가 생겨도 이 두 함수는 수정할 필요가 없습니다.
// =====================================================================

/// LanceDB 로 내려보낼 '스코프' SQL 을 만듭니다.
/// 봉투(Envelope) 물리 컬럼만 사용하므로 DataFusion 문법 에러가 구조적으로 발생할 수 없습니다.
fn build_scope_filter(ctx: &Value, search_mode: &str) -> Option<String> {
    let mut filters: Vec<String> = Vec::new();

    // ── type : 도메인 파티션 ──
    //  🌟 v4 : STAGE-3 이 types 배열(확정 도메인 + 교차 후보)을 실어 보냅니다.
    //     v3 는 후보 도메인마다 별도 컨텍스트를 만들어 쿼리를 늘렸는데,
    //     IN 절 하나로 같은 리콜을 1회 왕복에 얻습니다.
    let mut type_list: Vec<String> = Vec::new();

    if let Some(arr) = ctx.get("types").and_then(|v| v.as_array()) {
        for t in arr {
            if let Some(s) = t.as_str() {
                let clean = s.trim();
                if clean.is_empty() || clean == "ignore" { continue; }
                // 구버전 STAGE-3 의 "{domain}_items" 접미사 호환
                let base = clean.strip_suffix("_items").unwrap_or(clean);
                if !type_list.iter().any(|x| x == base) { type_list.push(base.to_string()); }
            }
        }
    }

    if type_list.is_empty() {
        if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
            let clean = t.trim();
            let base = clean.strip_suffix("_items").unwrap_or(clean);
            if !base.is_empty() && base != "ignore" {
                type_list.push(base.to_string());
            }
        }
    }

    if type_list.len() == 1 {
        filters.push(format!("type = '{}'", type_list[0].replace('\'', "''")));
    } else if type_list.len() > 1 {
        let quoted: Vec<String> = type_list.iter()
            .map(|t| format!("'{}'", t.replace('\'', "''")))
            .collect();
        filters.push(format!("type IN ({})", quoted.join(", ")));
    }

    // ── mode : commerce / shipping / analytic 트랙 격리 ──
    if !search_mode.trim().is_empty() {
        filters.push(format!("mode = '{}'", search_mode.replace('\'', "''")));
    }

    // ── cc / bcc / ref : 팀·도메인·페이지 스코프 ──
    for key in ["cc", "bcc", "ref"] {
        if let Some(v) = ctx.get(key).and_then(|x| x.as_str()) {
            let clean = v.trim();
            if !clean.is_empty() {
                filters.push(format!("`{}` = '{}'", key, clean.replace('\'', "''")));
            }
        }
    }

    // ── 시간 범위 : created_at / updated_at 은 봉투 물리 컬럼이므로 SQL 로 좁히는 게 이득 ──
    //    (전체 문서에서 기간 밖 데이터를 미리 걷어내면 벡터 검색 후보가 줄어 리콜 품질이 올라갑니다)
    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
        for (key, val_obj) in cond {
            let envelope_col = match key.as_str() {
                "created_at" => "created_at",
                "updated_at" => "updated_at",
                _ => continue,
            };
            let op_str = match val_obj.get("operator").and_then(|v| v.as_str()) {
                Some(s) => s.trim().to_lowercase(),
                None => continue,
            };
            // 비교 연산자만 SQL 로 내립니다. top/bottom/contains 는 Dexie 담당입니다.
            let sql_op = if op_str.starts_with("gte") { ">=" }
                else if op_str.starts_with("gt") { ">" }
                else if op_str.starts_with("lte") { "<=" }
                else if op_str.starts_with("lt") { "<" }
                else if op_str.starts_with("eq") { "=" }
                else { continue };

            let num = match val_obj.get("value") {
                Some(Value::Number(n)) => n.to_string(),
                Some(Value::String(s)) => {
                    let cleaned: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
                    if cleaned.is_empty() { continue; } else { cleaned }
                },
                _ => continue,
            };
            filters.push(format!("{} {} {}", envelope_col, sql_op, num));
        }
    }

    if filters.is_empty() { None } else { Some(filters.join(" AND ")) }
}

/// Dexie 가 실행할 '정밀 필터 플랜' 을 만듭니다.
/// 조건을 하나도 버리지 않고, 프론트엔드가 data.* 인덱스 또는 .filter() 로 처리할 수 있도록
/// 정규화된 JSON 으로 직렬화합니다.
///
/// 산출 형태:
/// {
///   "type": "goods",
///   "mode": "commerce",
///   "conditions": [
///     { "path": "data.sale_price", "op": "lte", "value": 5000, "kind": "number" },
///     { "path": "data.tracking_number", "op": "eq", "value": "123456", "kind": "string" },
///     { "path": "data.weight", "op": "top", "percent": 20.0, "kind": "rank" }
///   ],
///   "keywords": ["니트", "가디건"],
///   "alternates": { "color": ["title", "tags"] }
/// }
fn build_dexie_plan(ctx: &Value, search_mode: &str) -> Value {
    // 🌟 [PATH ALIAS] LLM 이 뽑아낸 속성명을 실제 data.* 경로로 정규화합니다.
    //    기존 convert_conditions_to_sql 의 mapped_key 는 '물리 컬럼이 없으면 버림' 이었지만,
    //    여기서는 '별칭만 통일하고 없는 건 그대로 통과' 시킵니다. Dexie 는 .filter() 로 처리 가능합니다.
    // 🌟 [PATH ALIAS v2 / EXTERNAL CONTRACT]
    //  ── 무엇이 바뀌었나 ──
    //   별칭 표를 Rust 코드에서 bias.json 의 `search_bridge.path_alias` 노드로 옮겼습니다.
    //   새 필드의 별칭이 필요해도 JSON 만 고치면 되고 재빌드가 필요 없습니다.
    //
    //   bias.json 예시:
    //   "search_bridge": {
    //     "path_alias": {
    //       "amount": ["amount_total", "total_amount", "price"],
    //       "doc_number": ["document_number", "bl_number", "awb_number", "po_number"],
    //       "sender_name": ["supplier_name", "shipper_name", "exporter_name"],
    //       "recipient_name": ["buyer_name", "consignee_name", "importer_name"],
    //       "vessel": ["vehicle_name", "flight_no", "vessel_name"],
    //       "pol": ["location_port_of_loading", "port_of_loading"],
    //       "pod": ["location_port_of_discharge", "port_of_discharge"],
    //       "incoterms": ["incoterms_code"],
    //       "container_number": ["container_no"],
    //       "seal_number": ["seal_no"],
    //       "weight_gross": ["gross_weight"],
    //       "weight_net": ["net_weight"],
    //       "volume": ["volume_measurement", "cbm"]
    //     }
    //   }
    //
    //   노드가 없으면 별칭 없이 그대로 통과하므로, 기존 동작을 깨지 않습니다.
    fn normalize_path(key: &str) -> String {
        let k = key.trim();

        if let Some(alias_obj) = crate::parsing::BIAS_DICT
            .get("search_bridge")
            .and_then(|sb| sb.get("path_alias"))
            .and_then(|v| v.as_object())
        {
            for (canonical, list) in alias_obj {
                if canonical == k {
                    return format!("data.{}", canonical);
                }
                if let Some(arr) = list.as_array() {
                    if arr.iter().any(|a| a.as_str().map_or(false, |s| s == k)) {
                        return format!("data.{}", canonical);
                    }
                }
            }
        }

        format!("data.{}", k)
    }

    // 🌟 [KIND] Dexie 실행 엔진이 인덱스 쿼리를 쓸지 .filter() 를 쓸지 판정하는 힌트입니다.
    //  ── 무엇이 바뀌었나 ──
    //   기존에는 수치 경로 15개를 문자열 배열로 나열해, 새 수치 필드를 추가할 때마다
    //   여기에도 이름을 넣어야 했습니다. (안 넣으면 kind="string" 이 되어
    //   '5000 이하' 조건이 문자열 비교로 떨어져 무력화됩니다)
    //   이제 canonicalize 와 동일한 kind_of() 규칙을 재사용하므로
    //   Dexie 에 필드를 추가해도 이 함수는 수정할 필요가 없습니다.
    fn value_kind(path: &str, v: &Value) -> &'static str {
        use crate::utils::canonical::{kind_of, CanonKind};

        if v.is_number() { return "number"; }

        // 'data.sale_price' → 'sale_price' 로 잘라 규칙 판정에 넘깁니다.
        let leaf = path.rsplit('.').next().unwrap_or(path);

        if let Some(s) = v.as_str() {
            match kind_of(leaf) {
                CanonKind::Numeric | CanonKind::Boolean => {
                    if s.chars().any(|c| c.is_ascii_digit()) { return "number"; }
                },
                _ => {}
            }
        }
        "string"
    }

    let mut conditions: Vec<Value> = Vec::new();

    // ── status : 문자열 상태값을 코드로 환산해서 넣습니다 ──
    if let Some(status) = ctx.get("status").and_then(|v| v.as_str()) {
        let clean = status.trim();
        // 🌟 [ZERO GUARD] logic::parse_status 는 매핑에 없는 문자열에 0 을 돌려줍니다.
        //    그런데 canonicalize_data 가 status 미보유 문서에 0 을 시딩하므로,
        //    코드 0 을 조건으로 내보내면 '상태가 없는 문서 전부' 를 매칭하게 됩니다.
        //    (bias.json 의 status_filters.remove 가 정확히 이 경로였습니다)
        //    코드가 확정되지 않으면 조건 자체를 만들지 않는 편이 리콜에 안전합니다.
        let code = crate::logic::parse_status(clean);
        if !clean.is_empty() && clean != "null" && code != 0 {
            conditions.push(json!({
                "path": "data.status",
                "op": "eq",
                "value": code,
                "kind": "number"
            }));
        } else if !clean.is_empty() && clean != "null" {
            println!(
                "[DEXIE-PLAN] ⚪ status='{}' 는 parse_status 매핑이 없어(코드 0) 조건에서 제외합니다.",
                clean
            );
        }
    }

    // ── condition : 전량 통과 ──
    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
        for (key, val_obj) in cond {
            // created_at / updated_at 은 build_scope_filter 가 이미 SQL 로 처리했으므로 중복 제외
            if key == "created_at" || key == "updated_at" { continue; }

            let path = normalize_path(key);
            let op_raw = val_obj.get("operator").and_then(|v| v.as_str()).unwrap_or("eq");
            // 🌟 LLM 이 "lt [Alts: lte, gte]" 같은 쓰레기를 붙여 보내는 사례가 있어 앞부분만 취합니다.
            let op_clean = op_raw.trim().to_lowercase();
            let op = if op_clean.starts_with("gte") { "gte" }
                else if op_clean.starts_with("gt") { "gt" }
                else if op_clean.starts_with("lte") { "lte" }
                else if op_clean.starts_with("lt") { "lt" }
                else if op_clean.starts_with("not_contains") { "not_contains" }
                else if op_clean.starts_with("contains") { "contains" }
                else if op_clean.starts_with("top") { "top" }
                else if op_clean.starts_with("bottom") { "bottom" }
                else if op_clean.starts_with("neq") { "neq" }
                else { "eq" };

            // ── top / bottom : 백분위 랭킹. Dexie 가 정렬 후 슬라이스합니다 ──
            if op == "top" || op == "bottom" {
                let pct = val_obj.get("percent_total")
                    .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()))
                    .unwrap_or(20.0);
                conditions.push(json!({
                    "path": path,
                    "op": op,
                    "percent": pct,
                    "kind": "rank"
                }));
                continue;
            }

            let raw_val = match val_obj.get("value") {
                Some(v) => v.clone(),
                None => continue,
            };
            // 빈 값은 조건이 아니라 노이즈입니다.
            let is_empty = match &raw_val {
                Value::Null => true,
                Value::String(s) => s.trim().is_empty() || s == "null",
                _ => false,
            };
            if is_empty { continue; }

            let kind = value_kind(&path, &raw_val);
            let final_val = if kind == "number" {
                let n = match &raw_val {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0),
                    Value::String(s) => {
                        let cleaned: String = s.chars().filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-').collect();
                        cleaned.parse::<f64>().unwrap_or(0.0)
                    },
                    _ => 0.0,
                };
                // canonicalize_data 와 동일하게 정수는 정수로 내려야 인덱스 타입이 일치합니다.
                if n.fract() == 0.0 && n.abs() < 9e15 { json!(n as i64) } else { json!(n) }
            } else {
                json!(raw_val.as_str().unwrap_or("").trim())
            };

            conditions.push(json!({
                "path": path,
                "op": op,
                "value": final_val,
                "kind": kind
            }));
        }
    }

    // ── keywords : 조건이 되지 못한 청크. Dexie 가 text/masked_text 부분 일치로 보조 필터링 ──
    let mut keywords: Vec<String> = Vec::new();
    if let Some(un) = ctx.get("unassigned").and_then(|v| v.as_array()) {
        for u in un {
            if let Some(s) = u.as_str() {
                for w in s.split_whitespace() {
                    if !keywords.iter().any(|k| k == w) { keywords.push(w.to_string()); }
                }
            }
        }
    }

    // 🌟 [TYPES] Dexie 도 IN 절과 동일하게 여러 타입을 통과시켜야 합니다.
    //    plan.type 하나만 보면 교차 후보 도메인 결과가 전부 잘려 나갑니다.
    let mut types: Vec<String> = Vec::new();
    if let Some(arr) = ctx.get("types").and_then(|v| v.as_array()) {
        for t in arr {
            if let Some(s) = t.as_str() {
                let clean = s.trim();
                if clean.is_empty() || clean == "ignore" { continue; }
                let base = clean.strip_suffix("_items").unwrap_or(clean);
                if !types.iter().any(|x| x == base) { types.push(base.to_string()); }
            }
        }
    }
    if types.is_empty() {
        if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
            let base = t.strip_suffix("_items").unwrap_or(t);
            if !base.is_empty() && base != "ignore" { types.push(base.to_string()); }
        }
    }

    json!({
        "type": ctx.get("type").and_then(|v| v.as_str()).unwrap_or(""),
        "types": types,
        "mode": search_mode,
        "conditions": conditions,
        "keywords": keywords,
        "alternates": ctx.get("alternates").cloned().unwrap_or(json!({})),
        "substantial": ctx.get("substantial").cloned().unwrap_or(json!("")),
        "find": ctx.get("find").cloned().unwrap_or(json!(""))
    })
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<TradeDocument>, String> {
    
    println!("[DB-FETCH] 리스트 불러오기 요청 수신 (Limit: {}, Filter: {:?})", limit, filter);

    // 🌟 v4 : 봉투 컬럼 술어만 남깁니다. 도메인 조건은 Dexie 가 처리합니다.
    let scope = sanitize_scope_filter(filter);

    let mut store_guard = state.store.lock().await; 
    
    
    // 프론트엔드가 데이터를 요청했는데 DB가 아직 없으면 즉시 여기서 로드합니다.
    if store_guard.is_none() {
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&db_path);
        if let Ok(s) = VectorStore::new(&db_path).await {
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        } else {
            return Err("Failed to initialize LanceDB".to_string());
        }
    }

    if let Some(store) = store_guard.as_ref() {
        let mut results = store.get_all_items("items", limit, offset, scope).await.map_err(|e| e.to_string())?;
        
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
        // 🌟 v4 : 물리 테이블이 items / users / pages 3개로 줄었습니다.
        //    (sales / tracking / event 를 순회하던 6회 왕복이 3회로 감소)
        let tables = vec!["items", "users", "pages"];

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

        // 2. Fallback search: data JSON 내부의 id / index 로 재탐색합니다.
        //    🌟 v4 : find_item_by_property 가 data ILIKE 프리필터를 쓰므로
        //    기존처럼 get_all_items(1000) 로 전량을 다시 긁을 필요가 없습니다.
        //    또한 canonicalize_data 가 index 를 String 으로 확정했으므로
        //    숫자/문자 두 갈래로 나눠 쏘던 index_query 분기도 불필요합니다.
        for table_name in tables.iter() {
            for prop in ["id", "index", "no", "code"] {
                if let Ok(Some((found_id, json_val))) = store.find_item_by_property(table_name, prop, &json!(uuid.clone())).await {
                    if let Ok(Some(mut d)) = store.get_item_by_id(table_name, &found_id).await {
                        if d.text.is_empty() {
                            d.text = parsing::json_to_natural_language(&json_val);
                        }
                        return Ok(Some(d));
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
        // 🌟 v4 : 물리 테이블 3개. resolve_table 이 중복 라우팅을 흡수하므로
        //    같은 테이블에 delete 를 여러 번 쏘던 낭비도 사라집니다.
        let tables = vec!["items", "users", "pages"];
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

        let tables = vec!["items", "users", "pages"];
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

    // 🌟 [CRITICAL FIX] 이전 작업 취소로 인해 굳어있던 취소 토큰(Cancellation Token)을 무조건 초기화하여
    // 스케줄러 큐를 타지 않고 직접 호출되는 검색 커맨드가 튕겨나가지 않도록 완벽히 방어합니다!
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

    let cancel_token = state.cancellation_token.clone();
    
    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            if let Ok(s) = VectorStore::new(&db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned() 
    };

    
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

    
    let payload_pending = json!({ 
        "task_id": task_id, 
        "category": "Pending", 
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

    // 🌟 [SEARCH FLAG RESTORE]
    //  ── 무엇이 문제였나 ──
    //   프론트엔드가 입구를 막는다는 이유로 IS_SEARCHING.store(true) 를 제거했는데,
    //   이 플래그를 '읽는' 가드는 unload_model 과 reindex_pending_embeddings 에
    //   그대로 남아 있습니다. 플래그가 영원히 false 이므로 두 가드가 사문화되었고,
    //   특히 unload_model 은 ACTIVE_TASK_MEM 가드가 없어
    //   검색 도중 들어온 언로드 요청이 그대로 통과합니다.
    //   플래그를 다시 세우는 편이 가드를 지우는 것보다 안전합니다.
    //   (해제는 아래 search_process 종료부에서 이미 두 번 수행됩니다)
    IS_SEARCHING.store(true, Ordering::SeqCst);
    {
        // 최소한의 동기화 정보만 업데이트 (UI 복구용)
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        // let now = chrono::Utc::now().timestamp_millis();
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
                m.deep_purge_resources().await;
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

    // 🌟 [CRITICAL FIX] 백그라운드에서 Qwen3를 미리 로드하면 내부의 deep_purge_resources가
    // 메인 스레드가 사용 중인 임베딩 모델을 파괴하여 VRAM/메모리 충돌(panic 또는 0점 반환)을 유발합니다.
    // 따라서 병렬 꼼수를 제거하고, 필요한 시점에 순차적으로 모델을 로드하여 안정성을 100% 보장합니다.
    if !model.is_cpu_mode {
        emit_term("[VRAM-SETUP] GPU 모드 감지됨. 임베딩 모델 단독 로딩 개시...");
        let _ = model.ensure_embedding().await;
    }

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
    crate::utils::logger::log_task_progress(&app_handle, &task_id, &payload_start);

    
    let search_process = async {
        let mut all_results = Vec::new();
        // 🌟 [DEXIE PLAN] 컨텍스트별 정밀 필터 플랜을 모아 프론트엔드로 전달합니다.
        //    LanceDB 가 버렸던 조건이 여기에 전부 살아 있습니다.
        let mut dexie_plans: Vec<Value> = Vec::new();

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

        
        let structured_query = match search_mode.as_str() {
            "shipping" => {
                model.parse_shipping_query(&task_id, &app_handle, query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            "analytic" => {
                // 🌟 [ANALYTIC QUERY v2 → model.rs]
                //    parse_analytic_search_query 를 model.rs 의 LogisModel 메서드로 이동합니다.
                //    parse_commerce_query / parse_shipping_query 와 동일한 호출 패턴으로 통일되어,
                //    모델 로드·임베딩·Qwen3.5 호출을 LogisModel 내부에서 일괄 관리합니다.
                model
                    .parse_analytic_search_query(
                        &task_id,
                        &app_handle,
                        query.clone(),
                        &language,
                        cancel_token.clone(),
                    )
                    .await
                    .map_err(|e| e.to_string())?
            },
            _ => { // default: commerce
                model.parse_commerce_query(&task_id, &app_handle, query.clone(), &language, &metrics_json_str, cancel_token.clone()).await.map_err(|e| e.to_string())?
            }
        };

        if let (Some(store), Some(ctx_arr)) = (store_opt.clone(), structured_query.get("context").and_then(|v| v.as_array())) {
            for ctx in ctx_arr {
                
                if cancel_token.load(Ordering::Relaxed) { 
                    return Err("Search cancelled by user".to_string()); 
                }

                tokio::task::yield_now().await;

                let text = ctx.get("text").and_then(|v| v.as_str()).unwrap_or("");
                // 🌟 [CONDITION-ONLY CONTEXT] STAGE-3 은 텍스트가 비어도 조건이 있으면
                //    컨텍스트를 발행합니다(union_text 가 비고 condition 이 있는 경우).
                //    기존처럼 텍스트만 보고 continue 하면 dexie_plans.push 까지 건너뛰어
                //    그 컨텍스트의 조건이 프론트엔드에 전달조차 되지 않았습니다.
                let has_condition = ctx.get("condition")
                    .and_then(|v| v.as_object())
                    .map_or(false, |o| !o.is_empty());
                if text.trim().is_empty() && !has_condition { continue; }
                let raw_ctx_type = ctx.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                // 🌟 [CRITICAL FIX] "ignore"로 분류된 명령어/분석 요청 청크는 DB 검색 단계에서 완전히 무시합니다!
                if raw_ctx_type == "ignore" {
                    continue;
                }

                // 🌟 [FALLBACK SUFFIX 제거] v4 에서 sales/tracking/event 물리 테이블이 사라지고
                //    items 단일 테이블 + type 컬럼 파티셔닝이 되었으므로,
                //    STAGE-3 이 발행하던 "{domain}_items" 폴백 접미사는 존재 이유를 잃었습니다.
                //    (저장 테이블과 조회 테이블이 어긋날 수 있는 경로 자체가 사라졌습니다)
                //    구버전 STAGE-3 가 만든 컨텍스트와의 호환을 위해 접미사만 벗겨 냅니다.
                let ctx_type: &str = raw_ctx_type.strip_suffix("_items").unwrap_or(raw_ctx_type);
                // 🌟 [SEARCH TEXT SYNTHESIS] 텍스트가 비고 조건만 있는 컨텍스트는
                //    FTS 검색어가 없어 임베딩이 빈 문자열 벡터가 됩니다.
                //    조건 값과 unassigned 키워드로 검색어를 합성해 리콜을 확보합니다.
                //    (조건 자체는 dexie_plan 이 정확히 실행하므로 여기서는 후보만 넓히면 됩니다)
                let search_text = if !text.trim().is_empty() {
                    text.to_string()
                } else {
                    let mut parts: Vec<String> = Vec::new();
                    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
                        for (_, v) in cond {
                            if let Some(s) = v.get("value").and_then(|x| x.as_str()) {
                                let t = s.trim();
                                if !t.is_empty() && !parts.iter().any(|p| p == t) {
                                    parts.push(t.to_string());
                                }
                            }
                        }
                    }
                    if let Some(un) = ctx.get("unassigned").and_then(|v| v.as_array()) {
                        for u in un {
                            if let Some(s) = u.as_str() {
                                let t = s.trim();
                                if !t.is_empty() && !parts.iter().any(|p| p == t) {
                                    parts.push(t.to_string());
                                }
                            }
                        }
                    }
                    let synthesized = parts.join(" ");
                    println!(
                        "[AI-SEARCH] 🧪 [SEARCH TEXT SYNTHESIS] type='{}' | 텍스트 부재 → 조건 값으로 합성: \"{}\"",
                        ctx_type, synthesized
                    );
                    synthesized
                };

                // 🌟 [TABLE ROUTING v4] users / pages 만 물리 분리, 나머지는 전부 items.
                //    scheduler 저장 매핑과 lib 조회 매핑이 어긋나 'review 는 items 에 저장되는데
                //    event 에서 조회' 같은 구조적 0건 버그를 만들던 match 문을 제거했습니다.
                let target_table = match ctx_type {
                    "member" | "team" | "user" => "users",
                    "page" | "pages" => "pages",
                    _ => "items",
                };

                // 🌟 [SCOPE / PLAN 분리]
                //    LanceDB 에는 봉투 컬럼만 내려보내고,
                //    도메인 조건은 하나도 버리지 않고 dexie_plan 으로 프론트에 전달합니다.
                let mut scope_ctx = ctx.clone();
                if let Some(o) = scope_ctx.as_object_mut() {
                    // 폴백 접미사가 붙은 type 이 SQL 에 그대로 나가면 `type = 'review_items'` 가 되어
                    // 항상 0건이 됩니다. 정규화된 이름으로 교체합니다.
                    o.insert("type".to_string(), json!(ctx_type));
                    // 🌟 [TENANT SCOPE] build_scope_filter 는 cc / bcc / ref 를 읽도록 되어 있는데
                    //    STAGE-3 / parse_shipping_query / parse_analytic_query 가 만드는 컨텍스트에는
                    //    그 키가 하나도 없어 해당 블록이 통째로 죽어 있었습니다.
                    //    그 결과 AI 검색이 전 팀·전 사이트 문서를 무차별로 반환했습니다.
                    //    ai_search_complex 가 인자로 이미 받고 있으므로 여기서 주입합니다.
                    //
                    //    ⚠️ bcc / ref 는 넣지 않습니다.
                    //       bcc = hash_id(doc_type + cc) 로 '문서 종류' 단위,
                    //       ref = 문서 그룹 단위이므로 검색 스코프로 쓰면 리콜이 붕괴합니다.
                    if !cc.trim().is_empty() {
                        o.insert("cc".to_string(), json!(cc.clone()));
                    }
                }
                let scope_filter = build_scope_filter(&scope_ctx, &search_mode);
                let dexie_plan = build_dexie_plan(&scope_ctx, &search_mode);

                println!(
                    "[AI-SEARCH] 🧭 [ROUTE] type='{}' | table='{}' | scope_sql={:?} | dexie_conditions={}",
                    ctx_type,
                    target_table,
                    scope_filter,
                    dexie_plan.get("conditions").and_then(|c| c.as_array()).map(|a| a.len()).unwrap_or(0)
                );

                // 🌟 [PLAN 수집] 프론트엔드가 실행할 정밀 필터 플랜을 응답에 동봉합니다.
                dexie_plans.push(dexie_plan.clone());

                // 🌟 [TRACKING] tracking_number 는 이제 LIKE 땜질이 아니라
                //    dexie_plan 의 data.tracking_number eq 조건으로 정확히 처리됩니다.
                //    여기서는 FTS 를 끄지 않습니다. 리콜을 넓게 가져가는 것이 v4 의 역할이기 때문입니다.
                let emb = model.get_embedding(search_text.clone()).await.unwrap_or(vec![0.0; 384]);

                // 🌟 [RECALL LIMIT] Dexie 가 뒤에서 조건으로 잘라내므로 여기서는 넓게 긁습니다.
                //    기존 limit 5 는 '조건이 SQL 에 이미 적용되었다' 는 전제였는데,
                //    v4 에서는 조건이 SQL 에 없으므로 5건만 가져오면 정답이 잘려 나갑니다.
                const RECALL_LIMIT: usize = 50;

                let final_results = store
                    .search_items(target_table, &search_text, emb.clone(), RECALL_LIMIT, 0, scope_filter.clone(), true)
                    .await
                    .unwrap_or_else(|e| {
                        println!("[AI-SEARCH] ⚠️ scope query failed ({}). Falling back to mode-only scope.", e);
                        Vec::new()
                    });

                // 🌟 [SCOPE FALLBACK] 스코프가 지나치게 좁아 0건이면 도메인 조건만 완화해 한 번 더 긁습니다.
                //    기존 A/FULL → B/NARROWED → C/RECALL 3티어가 하던 일을 여기서 1회로 흡수합니다.
                //    🌟 [TENANT PRESERVE] 완화 대상은 'type / 시간' 같은 도메인 스코프뿐입니다.
                //       cc(팀·사이트)까지 버리면 폴백 한 번으로 타 팀 문서가 그대로 새어 나가므로,
                //       테넌트 스코프는 폴백에서도 반드시 유지합니다.
                let final_results = if final_results.is_empty() {
                    let mode_only = if cc.trim().is_empty() {
                        format!("mode = '{}'", search_mode)
                    } else {
                        format!("mode = '{}' AND `cc` = '{}'", search_mode, cc.replace('\'', "''"))
                    };
                    println!("[AI-SEARCH] 🛟 [SCOPE FALLBACK] 0 hit with full scope. Retrying with '{}'.", mode_only);
                    store
                        .search_items(target_table, &search_text, emb.clone(), RECALL_LIMIT, 0, Some(mode_only), true)
                        .await
                        .unwrap_or_default()
                } else {
                    final_results
                };

                for (id, content, score) in final_results {
                    // 결과 배열 내 중복 방어
                    let is_dup = all_results.iter().any(|item: &serde_json::Value| item.get("id").and_then(|v| v.as_str()) == Some(&id));
                    if !is_dup {
                        all_results.push(json!({ "id": id.clone(), "text": content.clone(), "score": score, "context_type": ctx_type, "relation": "primary" }));
                    }

                    // 백엔드(LanceDB)에서의 N:N 양방향 교차 검색(FTS) 로직을 제거하고,
                    // 프론트엔드(Dexie DB)로 역할을 위임합니다.
                }

                // =====================================================================
                // 🌟 [STAGE-4] item_chunks 청크 레벨 코사인 유사도 매칭
                // ---------------------------------------------------------------------
                // 기존 search_items() 는 전체 문서 벡터 1개와 질의 벡터를 비교하므로
                // "무거운" ↔ "Its weight is 1.5" 같은 필드 레벨 매칭이 희석되어 0건이 됩니다.
                // STAGE-4 는 scheduler 가 사전 저장한 item_chunks 테이블의
                // 속성별 청크 임베딩과 코사인 유사도를 계산하여
                // 필드 레벨 매칭을 복원합니다.
                //
                // 매칭 전략:
                //   4-A: 질의 텍스트 임베딩으로 item_chunks 전체 코사인 검색
                //   4-B: PLINKO 조건(condition)에 확정된 속성이 있으면
                //        해당 속성 청크에 보너스 가중치 부여
                //   4-C: item_id 기준 그룹핑 후 기존 all_results 와 dedup 병합
                // =====================================================================
                {
                    // 4-A: item_chunks 테이블 코사인 검색
                    //      🌟 v4 : STAGE-3 의 types 배열을 IN 절로 펼칩니다.
                    //      교차 후보 도메인의 청크도 함께 훑어야 리콜이 유지됩니다.
                    let chunk_type_filter = {
                        let mut list: Vec<String> = Vec::new();
                        if let Some(arr) = ctx.get("types").and_then(|v| v.as_array()) {
                            for t in arr {
                                if let Some(s) = t.as_str() {
                                    let base = s.strip_suffix("_items").unwrap_or(s).trim();
                                    if base.is_empty() || base == "ignore" { continue; }
                                    if !list.iter().any(|x| x == base) { list.push(base.to_string()); }
                                }
                            }
                        }
                        if list.is_empty() { list.push(ctx_type.to_string()); }

                        if list.len() == 1 {
                            format!("item_type = '{}' AND mode = '{}'", list[0], search_mode)
                        } else {
                            let quoted: Vec<String> = list.iter().map(|t| format!("'{}'", t)).collect();
                            format!("item_type IN ({}) AND mode = '{}'", quoted.join(", "), search_mode)
                        }
                    };

                    // 4-B: PLINKO 조건 기반 타겟 속성 목록 구축
                    //      ① condition 키 (PLINKO 가 확정한 속성)
                    //      ② substantial (추상 수식어가 지목한 물리 속성 — weight / sale_price ...)
                    //      substantial 은 LanceDB 물리 컬럼이 아니라 SQL 로는 절대 반영되지 않지만,
                    //      item_chunks 의 property 컬럼에는 그대로 존재하므로
                    //      '무거운 → weight' 의도를 여기서 실제 검색으로 회수합니다.
                    let mut condition_props: Vec<String> = ctx.get("condition")
                        .and_then(|v| v.as_object())
                        .map(|obj| obj.keys().cloned().collect())
                        .unwrap_or_default();

                    let substantial_prop = ctx.get("substantial")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !substantial_prop.is_empty() && !condition_props.iter().any(|p| p == &substantial_prop) {
                        condition_props.push(substantial_prop.clone());
                    }

                    let mut chunk_results = store.search_chunks(
                        &emb,
                        10, // 상위 10개 청크 (item_id 그룹핑 전 오버페치)
                        Some(&chunk_type_filter),
                    ).await.unwrap_or_default();

                    // 4-C: 속성 타겟 청크 검색 (property 필터 고정)
                    //      전체 코사인 검색은 긴 질의 문장과 짧은 저장 청크 사이의 구조적 격차 때문에
                    //      정답 속성 청크가 상위 10개 밖으로 밀려날 수 있습니다.
                    //      확정된 속성마다 property 를 고정한 별도 쿼리를 던져 리콜을 보증합니다.
                    //
                    //      🌟 [ALIAS RECALL] 이 경로는 음차 별칭(_tn/_tr)이 살아 돌아오는 유일한 통로입니다.
                    //      별칭은 원본과 동일한 property 를 갖도록 저장되므로(STAGE-4B/4C 호환 목적),
                    //      limit 이 작으면 원본 청크만으로 창이 가득 차 별칭이 전멸합니다.
                    //      10개 상품 × (원본 + native + roman) = 30 행이 존재할 수 있으므로 창을 넓힙니다.
                    for target_prop in condition_props.iter() {
                        if target_prop.trim().is_empty() { continue; }
                        let escaped_prop = target_prop.replace('\'', "''");
                        let prop_filter = format!("{} AND property = '{}'", chunk_type_filter, escaped_prop);
                        let targeted = store.search_chunks(&emb, 12, Some(&prop_filter)).await.unwrap_or_default();
                        if targeted.is_empty() { continue; }
                        println!("[AI-SEARCH]   🎯 [STAGE-4C] property='{}' 타겟 청크 검색: {}건 추가 확보", target_prop, targeted.len());
                        for t in targeted {
                            if chunk_results.iter().any(|(cid, _, _, _, _, _)| cid == &t.0) { continue; }
                            chunk_results.push(t);
                        }
                    }

                    // 4-D: 조건이 하나도 없을 때의 값 청크 진입 보증
                    //      PLINKO 가 ACTION VERB 게이트 등으로 조건을 못 만들면 STAGE-4C 가 아예 돌지 않고,
                    //      전역 코사인 검색만 남습니다. 그런데 그 창을 저변별 청크(status/sale_price)가
                    //      독점하면 값이 담긴 title 청크가 후보에 진입조차 못 합니다.
                    //      (new_log2.txt: 조건 0개 → 상위 10건 전부 status/traffic_insight, title 0건)
                    //      스키마에서 '자유 서술 값을 담는 형식(Text)' 필드만 골라 타겟 검색을 겁니다.
                    //      필드 선정은 detect_field_format 의 결정론 판정만 사용하므로
                    //      다국어 어휘도, 필드명 하드코딩도 없습니다.
                    //
                    //      🌟 [ALIAS RECALL] 여기도 limit 3 은 치명적이었습니다.
                    //      store.rs 의 per_property_cap = max(2, 3/2) = 2 가 걸려
                    //      "title(16행)" 이 통째로 억제되었고(log 실측), 음차 별칭이 100% 소멸했습니다.
                    //      store.rs 가 property 고정 검색에서 캡을 해제했으므로,
                    //      여기서는 창 크기만 별칭 포함 규모로 넓혀 주면 됩니다.
                    if condition_props.is_empty() {
                        use crate::utils::ai_utils::FieldFormat;
                        let schema_fields = crate::parsing::get_detail_schema_fields(ctx_type, "", "en");
                        let mut probed = 0usize;
                        for (fname, _, _, _) in schema_fields.iter() {
                            if crate::utils::ai_utils::detect_field_format(fname) != FieldFormat::Text {
                                continue;
                            }
                            let escaped_prop = fname.replace('\'', "''");
                            let prop_filter = format!("{} AND property = '{}'", chunk_type_filter, escaped_prop);
                            let targeted = store.search_chunks(&emb, 12, Some(&prop_filter)).await.unwrap_or_default();
                            if targeted.is_empty() { continue; }
                            probed += targeted.len();
                            for t in targeted {
                                if chunk_results.iter().any(|(cid, _, _, _, _, _)| cid == &t.0) { continue; }
                                chunk_results.push(t);
                            }
                        }
                        if probed > 0 {
                            println!(
                                "[AI-SEARCH]   🧭 [STAGE-4D] 조건 부재 → Text 형식 필드 타겟 청크 검색: {}건 추가 확보 (값 청크 진입 보증)",
                                probed
                            );
                        }
                    }

                    if !chunk_results.is_empty() {
                        println!("[AI-SEARCH] 🧩 [STAGE-4] item_chunks 코사인 매칭: {}개 item 후보 발견 (ctx_type='{}')", chunk_results.len(), ctx_type);
                    }

                    for (chunk_id, item_id, chunk_text, property, group_score, best_cos) in chunk_results {
                        // 🌟 [SCALE FIX] 트랙 가중치는 반드시 '코사인(0~1)' 에만 곱합니다.
                        //    기존에는 search_chunks 가 돌려준 '그룹 합산 점수' 를 코사인이라 가정하고 곱해
                        //    (log 실측: score 3.0146 → 최종 10.5510) 상한 1.0 전제가 무너졌고,
                        //    질의와 무관해도 청크가 많이 살아남은 item 이 상위를 독점했습니다.
                        //    합산(group_score)은 '증거의 양' 이므로 별도의 완만한 보너스로만 반영합니다.
                        let raw_cosine = best_cos.clamp(0.0f32, 1.0f32);

                        // 🌟 [ALIAS FLAG] 이 매칭이 음차 별칭 청크에서 나왔는지 판정합니다.
                        //    scheduler 의 upsert_alias_chunks 가 chunk_id 에 _tn / _tr 접미어를 붙이므로
                        //    의미 판정이 아니라 '우리가 만든 접미어' 라는 구조적 사실입니다.
                        let is_alias = chunk_id.ends_with("_tn") || chunk_id.ends_with("_tr");

                        // 4-D: 3-Track 스케일 정합
                        //      store.rs 의 하이브리드 검색은 Column=3.0 / FTS=2.0 / Vector=1.0 가산 스케일입니다.
                        //      청크 매칭은 '본문 매칭' 이므로 FTS 트랙(2.0)과 동일 스케일로 환산하고,
                        //      PLINKO 가 확정한 속성과 일치하면 Column 트랙(3.0)을 추가로 얹습니다.
                        let mut score = raw_cosine * 2.0;
                        if !condition_props.is_empty() && condition_props.contains(&property) {
                            let column_track = raw_cosine * 3.0;
                            score += column_track;
                            println!("[AI-SEARCH]   🎯 [STAGE-4B] property='{}' PLINKO 조건 매칭 → Column 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}", property, column_track, raw_cosine, score);
                        }

                        // 🌟 [STAGE-4X CROSS-LINGUAL VALUE BONUS]
                        //    "니트 가디건" ↔ "Cable Knit Cardigan" 처럼 값이 서로 다른 문자 체계일 때
                        //    FTS(ngram 문자열 포함)는 물리적으로 0건입니다.
                        //    이 경우 크로스링구얼 매칭이 가능한 트랙은 값 청크 코사인뿐이므로,
                        //    '값이 의미를 나르는 속성'에 FTS 결손분을 보전합니다.
                        {
                            use crate::utils::ai_utils::FieldFormat;
                            // 🌟 [ENUM EXCLUDE] Enum 값은 'complete' / 'show' / 'hide' 같은
                            //    저카디널리티 캐노니컬 키이며, 전 아이템에서 동일 문자열입니다.
                            //    자유 서술 값을 담는 Text / Address 만 크로스링구얼 보전 대상입니다.
                            let value_bearing = matches!(
                                crate::utils::ai_utils::detect_field_format(&property),
                                FieldFormat::Text | FieldFormat::Address
                            );
                            if value_bearing {
                                let cross_lingual_track = raw_cosine * 1.5;
                                score += cross_lingual_track;
                                println!(
                                    "[AI-SEARCH]   🌐 [STAGE-4X] property='{}' 자유서술 값 속성 → 크로스링구얼 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}",
                                    property, cross_lingual_track, raw_cosine, score
                                );
                            }
                        }

                        // 🌟 [STAGE-4Y ALIAS TRACK] 음차 별칭이 매칭됐다는 것은
                        //    '문자 체계가 달라 FTS 가 물리적으로 0건인 상황에서, 발음 축으로 값이 일치했다'
                        //    는 뜻입니다. 이것이 바로 별칭을 저장한 목적이므로 전용 트랙을 부여합니다.
                        //    (원본 청크가 이미 확보한 점수를 대체하지 않고 가산만 하므로
                        //     별칭이 없는 item 이 손해를 보지 않습니다)
                        if is_alias {
                            let alias_track = raw_cosine * 1.5;
                            score += alias_track;
                            println!(
                                "[AI-SEARCH]   🔤 [STAGE-4Y] 음차 별칭 매칭 ({}) property='{}' | chunk='{}' → 별칭 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}",
                                if chunk_id.ends_with("_tn") { "native" } else { "roman" },
                                property, chunk_text, alias_track, raw_cosine, score
                            );
                        }

                        // 🌟 [EVIDENCE VOLUME] 합산 점수는 '몇 개의 청크가 함께 반응했는가' 라는
                        //    보조 신호입니다. 로그 스케일로 완만하게만 반영하여
                        //    청크 수가 많은 item 이 코사인 우위를 뒤집지 못하게 합니다.
                        if group_score > raw_cosine {
                            let volume_bonus = ((group_score - raw_cosine).max(0.0) + 1.0).ln() * 0.25;
                            score += volume_bonus;
                        }

                        // 4-C: item_id 기준 기존 결과와 dedup
                        //      이미 all_results 에 동일 item_id 가 있으면
                        //      점수만 갱신하고 중복 삽입하지 않습니다.
                        let existing = all_results.iter_mut().find(|item| {
                            item.get("id").and_then(|v| v.as_str()) == Some(&item_id)
                        });

                        if let Some(existing_item) = existing {
                            // 기존 점수보다 높으면 갱신
                            let old_score = existing_item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            if score > old_score {
                                existing_item.as_object_mut().unwrap().insert("score".to_string(), json!(score));
                                existing_item.as_object_mut().unwrap().insert("chunk_match".to_string(), json!(true));
                                existing_item.as_object_mut().unwrap().insert("matched_property".to_string(), json!(property));
                                existing_item.as_object_mut().unwrap().insert("matched_chunk".to_string(), json!(chunk_text));
                                existing_item.as_object_mut().unwrap().insert("alias_match".to_string(), json!(is_alias));
                            }
                        } else {
                            // 신규 결과 삽입
                            // chunk_id 가 아닌 item_id 를 기준으로 삽입하여
                            // 동일 item 의 여러 청크가 중복 결과로 나타나지 않도록 합니다.
                            let is_item_dup = all_results.iter().any(|item| {
                                item.get("id").and_then(|v| v.as_str()) == Some(&item_id)
                            });
                            if !is_item_dup {
                                all_results.push(json!({
                                    "id": item_id,
                                    "text": chunk_text,
                                    "score": score,
                                    "context_type": ctx_type,
                                    "relation": "chunk_match",
                                    "chunk_id": chunk_id,
                                    "matched_property": property,
                                    "chunk_match": true,
                                    "alias_match": is_alias
                                }));
                            }
                        }
                    }
                }
                // =====================================================================
                // 🌟 [STAGE-4 종료]
                // =====================================================================
            }
        }

        // =====================================================================
        // 🌟 [STAGE-5] 결과 병합 & 스코어 랭킹 & 프론트엔드 응답 포맷 조립
        // ---------------------------------------------------------------------
        // STAGE-4 까지 수집된 all_results 에는 다음 두 종류의 결과가 혼재합니다.
        //   (A) search_items()  → 전체 문서 벡터 + FTS 매칭  (relation: "primary")
        //   (B) search_chunks() → 청크 레벨 코사인 매칭       (relation: "chunk_match")
        //
        // STAGE-5 의 역할:
        //   5-A: item_id 기준 dedup (동일 item 의 여러 청크가 중복 결과로 나타나지 않도록)
        //   5-B: 점수 정규화 (문서 점수와 청크 점수를 동일한 스케일로 병합)
        //   5-C: 매칭 속성(property) 메타데이터 주입 (프론트엔드 하이라이트용)
        //   5-D: 최종 랭킹 (내림차순) 후 limit 적용
        //   5-E: 프론트엔드 응답 JSON 포맷 확정
        // =====================================================================
        {
            // ── 5-A: item_id 기준 dedup ──
            // 동일 item_id 가 여러 번 등장하면 점수를 합산하고,
            // 가장 높은 청크 매칭 정보(property, chunk_text)를 대표로 남깁니다.
            let mut merged_map: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();
            let mut merge_order: Vec<String> = Vec::new();

            for item in all_results.iter() {
                let item_id = item.get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if item_id.is_empty() { continue; }

                let score = item.get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let is_chunk_match = item.get("chunk_match")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let matched_property = item.get("matched_property")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let matched_chunk = item.get("matched_chunk")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(existing) = merged_map.get_mut(&item_id) {
                    // 기존 항목의 점수에 새 점수를 가산
                    let old_score = existing.get("score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let combined = old_score + score;
                    existing.as_object_mut().unwrap()
                        .insert("score".to_string(), json!(combined));

                    // 청크 매칭 정보가 더 구체적이면 대표 메타데이터로 교체
                    if is_chunk_match && !matched_property.is_empty() {
                        let old_prop = existing.get("matched_property")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if old_prop.is_empty() || score > old_score {
                            existing.as_object_mut().unwrap()
                                .insert("matched_property".to_string(), json!(matched_property));
                            existing.as_object_mut().unwrap()
                                .insert("matched_chunk".to_string(), json!(matched_chunk));
                            existing.as_object_mut().unwrap()
                                .insert("chunk_match".to_string(), json!(true));
                        }
                    }

                    // 매칭된 속성 목록 누적 (프론트엔드에서 다중 하이라이트 가능)
                    if is_chunk_match && !matched_property.is_empty() {
                        let props = existing.get("matched_properties")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let mut props_vec: Vec<Value> = props;
                        if !props_vec.iter().any(|p| p.as_str() == Some(&matched_property)) {
                            props_vec.push(json!(matched_property));
                        }
                        existing.as_object_mut().unwrap()
                            .insert("matched_properties".to_string(), json!(props_vec));
                    }
                } else {
                    // 신규 항목 삽입
                    let mut entry = item.clone();
                    if is_chunk_match && !matched_property.is_empty() {
                        entry.as_object_mut().unwrap()
                            .insert("matched_properties".to_string(), json!(vec![matched_property.clone()]));
                    }
                    merge_order.push(item_id.clone());
                    merged_map.insert(item_id, entry);
                }
            }

            // ── 5-B: 점수 정규화 ──
            // 문서 매칭(primary)과 청크 매칭(chunk_match)의 점수 스케일이 다를 수 있으므로
            // 최대 점수 기준으로 0.0~1.0 에 정규화합니다.
            // 단, 점수가 전부 0 이면 정규화를 건너뜁니다.
            let mut max_score: f32 = 0.0;
            for item in merged_map.values() {
                let s = item.get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                if s > max_score { max_score = s; }
            }

            let mut ranked_results: Vec<serde_json::Value> = Vec::new();
            for item_id in &merge_order {
                if let Some(mut item) = merged_map.remove(item_id) {
                    if max_score > 0.0 {
                        let raw = item.get("score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        let normalized = raw / max_score;
                        item.as_object_mut().unwrap()
                            .insert("score".to_string(), json!(normalized));
                        item.as_object_mut().unwrap()
                            .insert("raw_score".to_string(), json!(raw));
                    }
                    ranked_results.push(item);
                }
            }

            // ── 5-D: 최종 랭킹 (점수 내림차순) ──
            ranked_results.sort_by(|a, b| {
                let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            // ── 5-E: 프론트엔드 응답 포맷 확정 ──
            // 프론트엔드(main.ts)가 소비하는 필드:
            //   id, text, score, context_type, relation,
            //   chunk_match(bool), matched_property, matched_chunk, matched_properties
            //
            // chunk_match = true 인 항목은 프론트엔드에서
            // "어떤 속성이 매칭되었는지" 를 배지로 표시할 수 있습니다.
            //
            // 🌟 [RECALL PRESERVE] 기존 20건 절단은 v4 설계와 정면으로 충돌했습니다.
            //    이 시점의 ranked_results 는 '도메인 조건이 하나도 적용되지 않은 리콜 후보' 입니다.
            //    정렬 기준도 벡터·FTS 점수뿐이므로, 조건을 만족하는 정답이 21위였다면
            //    Dexie(executeDexiePlan)가 그 조건을 실행할 기회 자체를 잃습니다.
            //    컨텍스트가 3개면 RECALL_LIMIT 50 × 3 = 150 후보가 20건으로 잘립니다.
            //    최종 표시 개수는 조건을 적용한 프론트엔드가 결정해야 합니다.
            //    (여기서는 전송량 상한만 둡니다)
            const FINAL_LIMIT: usize = 200;
            if ranked_results.len() > FINAL_LIMIT {
                println!(
                    "[AI-SEARCH] ✂️ [TRANSPORT CAP] 리콜 후보 {}건 → {}건으로 상한 적용 (조건 적용은 Dexie 담당)",
                    ranked_results.len(), FINAL_LIMIT
                );
                ranked_results.truncate(FINAL_LIMIT);
            }

            // ── 검색 통계 로그 출력 ──
            let chunk_match_count = ranked_results.iter()
                .filter(|r| r.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false))
                .count();
            let primary_count = ranked_results.len() - chunk_match_count;

            println!("[AI-SEARCH] 📊 [STAGE-5] 결과 병합 완료: 총 {}건 (문서 매칭 {}건 + 청크 매칭 {}건)",
                ranked_results.len(), primary_count, chunk_match_count);

            for (rank, item) in ranked_results.iter().enumerate() {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let score = item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let prop = item.get("matched_property").and_then(|v| v.as_str()).unwrap_or("-");
                let is_chunk = item.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let is_alias = item.get("alias_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let ctx = item.get("context_type").and_then(|v| v.as_str()).unwrap_or("?");
                let matched_text = item.get("matched_chunk")
                    .or_else(|| item.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let brief: String = matched_text.chars().take(48).collect();
                println!("  [RANK {}] id={} | score={:.4} | type={} | match={} | property={} | matched='{}'",
                    rank + 1, id, score, ctx,
                    if is_chunk { if is_alias { "alias" } else { "chunk" } } else { "doc" },
                    prop, brief);
            }

            all_results = ranked_results;
        }
        // =====================================================================
        // 🌟 [STAGE-5 종료]
        // =====================================================================

        // =====================================================================
        // 🌟 [STAGE-6 / ANALYTIC REPORT] 회수한 시맨틱 기록으로 리포트를 작성합니다.
        // ---------------------------------------------------------------------
        //  ── 왜 여기서 만드는가 ──
        //   프론트엔드는 LanceDB 원문(action / summary / relate)을 갖고 있지 않습니다.
        //   결과 목록만 던지면 사용자는 로그 조각을 직접 읽어야 합니다.
        //   Rust 가 원문을 다시 꺼내 Qwen3.5 2B 로 답변형 리포트를 합성해
        //   채팅 말풍선 한 개로 전달합니다.
        //  ── 근거 고정 ──
        //   프롬프트는 "회수된 기록만 근거로 삼고, 없으면 없다고 말하라" 를 강제합니다.
        // =====================================================================
        let mut analytic_report = String::new();

        if search_mode == "analytic" && !all_results.is_empty() {
            if let Some(store) = store_opt.as_ref() {
                let _ = app_handle.emit("extraction-progress", json!({
                    "task_id": task_id,
                    "category": "Report",
                    "summary": "Writing behaviour report...",
                    "spinner": "⠋"
                }));

                let mut records: Vec<Value> = Vec::new();

                for item in all_results.iter().take(30) {
                    if cancel_token.load(Ordering::Relaxed) { break; }
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if id.is_empty() { continue; }

                    if let Ok(Some(doc)) = store.get_item_by_id("items", id).await {
                        if let Ok(d) = serde_json::from_str::<Value>(&doc.json_data) {
                            let action = d.get("action").and_then(|v| v.as_str()).unwrap_or("");
                            let summary = d.get("summary").and_then(|v| v.as_str()).unwrap_or("");
                            let flow = d.get("cross_action_flow").and_then(|v| v.as_str()).unwrap_or("");
                            let evo = d.get("intent_evolution").and_then(|v| v.as_str()).unwrap_or("");
                            let pref = d.get("consistent_preferences").and_then(|v| v.as_str()).unwrap_or("");

                            if action.is_empty() && summary.is_empty() && flow.is_empty() {
                                continue;
                            }

                            let at = d.get("created_at").and_then(|v| v.as_i64()).unwrap_or(doc.created_at_ts);
                            let at_iso = chrono::DateTime::from_timestamp_millis(at)
                                .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
                                .unwrap_or_default();

                            records.push(json!({
                                "user": doc.from,
                                "type": doc.r#type,
                                "at": at_iso,
                                "link": d.get("link").and_then(|v| v.as_str()).unwrap_or(""),
                                "action": action,
                                "summary": summary,
                                "relate": d.get("relate").cloned().unwrap_or(json!([])),
                                "cross_action_flow": flow,
                                "intent_evolution": evo,
                                "consistent_preferences": pref
                            }));
                        }
                    }
                }

                if !records.is_empty() {
                    let now_ms = chrono::Utc::now().timestamp_millis();
                    let current_iso = chrono::DateTime::from_timestamp_millis(now_ms)
                        .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
                        .unwrap_or_default();

                    let started_at = structured_query.get("started_at").and_then(|v| v.as_i64()).unwrap_or(0);
                    let expired_at = structured_query.get("expired_at").and_then(|v| v.as_i64()).unwrap_or(0);
                    let time_intent = structured_query.get("time_intent").and_then(|v| v.as_str()).unwrap_or("");
                    let season_intent = structured_query.get("season_intent").and_then(|v| v.as_str()).unwrap_or("");

                    let period_str = if started_at > 0 {
                        let s = chrono::DateTime::from_timestamp_millis(started_at)
                            .map(|dt| dt.naive_utc().format("%Y-%m-%d").to_string())
                            .unwrap_or_default();
                        let e = chrono::DateTime::from_timestamp_millis(expired_at)
                            .map(|dt| dt.naive_utc().format("%Y-%m-%d").to_string())
                            .unwrap_or_default();
                        format!("{} ~ {}", s, e)
                    } else {
                        "all time".to_string()
                    };

                    let scope = format!(
                        "Period: {} (time_intent='{}', season_intent='{}')\nRecords retrieved: {}",
                        period_str,
                        if time_intent.is_empty() { "-" } else { time_intent },
                        if season_intent.is_empty() { "-" } else { season_intent },
                        records.len()
                    );

                    let time_context = format!(
                        "- Current UTC time is \"{}\" (epoch ms {}).\n- The user locale language is \"{}\".",
                        current_iso, now_ms, language
                    );

                    let records_json = serde_json::to_string_pretty(&records).unwrap_or_else(|_| "[]".to_string());

                    let prompt = crate::prompts::analytic_report_answer_prompt(
                        &query,
                        &time_context,
                        &scope,
                        &records_json,
                        &language
                    );

                    model.secure_vram_relay(
                        crate::model::ModelSize::Qwen3_5,
                        None,
                        Some(cancel_token.clone()),
                        false,
                        None
                    ).await.map_err(|e| e.to_string())?;

                    let params = crate::openai_types::ChatCompletionParameters {
                        messages: vec![
                            crate::openai_types::ChatCompletionRequestMessage::User(
                                crate::openai_types::ChatCompletionRequestUserMessage {
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                                    name: None,
                                }
                            )
                        ],
                        model: "qwen3.5".to_string(),
                        max_tokens: Some(2048),
                        temperature: Some(0.2),
                        top_p: Some(0.95),
                        ..Default::default()
                    };

                    if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                        if let Ok(res) = gen.generate(
                            params,
                            Some(cancel_token.clone()),
                            Some(format!("{}_report", task_id)),
                            None,
                            None,
                            None
                        ).await {
                            analytic_report = res.trim().to_string();
                        }
                    }

                    emit_term(&format!(
                        "[ANALYTIC-REPORT] 📝 근거 기록 {}건으로 리포트를 작성했습니다. ({}자)",
                        records.len(),
                        analytic_report.chars().count()
                    ));
                } else {
                    emit_term("[ANALYTIC-REPORT] ⚪ 회수된 문서에 구조화 문장이 없어 리포트를 생략합니다.");
                }
            }
        }

        // 🌟 [RESPONSE v4] 프론트엔드는 아래 4개를 받습니다.
        //    structured  : PLINKO 가 확정한 의미 구조 (기존과 동일)
        //    results     : LanceDB 리콜 후보 (조건 미적용, 넓게)
        //    dexie_plans : Dexie 가 실행할 정밀 필터 플랜 (조건 100% 보존)
        //    report      : analytic 모드 전용. Qwen3.5 2B 가 쓴 답변형 리포트
        //  → main.ts 가 results 를 dexie_plans 로 걸러 최종 결과를 확정합니다.
        Ok(json!({
            "structured": structured_query,
            "results": all_results,
            "dexie_plans": dexie_plans,
            "report": analytic_report
        }))
    }.await; 

    IS_SEARCHING.store(false, Ordering::SeqCst);
    
    if let Some(store) = store_opt.as_ref() {
        match &search_process {
            Ok(result_data) => { 
                let _ = store.update_task_status(&task_id, 9).await; 
                let _ = store.update_message_status(&task_id, 9, None).await;

                
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

    
    {
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        if let Some(mem) = mem_guard.as_ref() {
            if mem.get("id").and_then(|v| v.as_str()) == Some(task_id.as_str()) {
                *mem_guard = None;
            }
        }
    }

    model.deep_purge_resources().await; 
    
    
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
            m.deep_purge_resources().await;
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
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&db_path);
        if let Ok(s) = VectorStore::new(&db_path).await {
            let _ = s.init_task_table().await;
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        // General search for context
        let emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384]);
        
        if let Ok(results) = store.search_items("items", &query, emb, 3, 0, None, false).await {
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
        // 🌟 [BORROW FIX] query_pairs_mut() 가변 빌드 중에는
        //    query_pairs() 불변 빌드를 호출할 수 없습니다.
        //    따라서 href 존재 여부를 가변 빌드 '시작 전'에 확인합니다.
        let has_href = target_url.query_pairs().any(|(k, _)| k == "href");

        let mut query = target_url.query_pairs_mut();
        if let Some(hash) = sp.get("hash").and_then(|v| v.as_str()) { query.append_pair("hash", hash); }
        if let Some(token) = sp.get("token").and_then(|v| v.as_str()) { query.append_pair("token", token); }
        if let Some(cc) = sp.get("cc").and_then(|v| v.as_str()) { query.append_pair("cc", cc); }
        // 🌟 [ANALYTICS SYNC] console.logis.center Worker 는
        //    cookies.href = decodeURIComponent(req.query.href) 를
        //    무조건 실행하므로, 이 값이 없으면 500 에러가 발생합니다.
        //    href 가 없으면 현재 앱의 기본 도메인을 시딩합니다.
        if sp.get("href").is_none() {
            if !has_href {
                query.append_pair("href", "https://console.logis.center/");
            }
        } else if let Some(href) = sp.get("href").and_then(|v| v.as_str()) {
            if !has_href {
                query.append_pair("href", href);
            }
        }
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
    
    if let Ok(mem_guard) = crate::ACTIVE_TASK_MEM.read() {
        if let Some(active) = mem_guard.as_ref() {
            let active_ref = active.get("ref").and_then(|v| v.as_str()).unwrap_or("");
            let status = active.get("status").and_then(|v| v.as_i64()).unwrap_or(0);
            
            
            // 완료된 작업(9)에 의해 추출 버튼이 영구적으로 숨겨지는 버그를 완벽히 막을 수 있습니다.
            if active_ref == payload.r#ref && (status == 1 || status == 10) {
                return Ok(true); // 현재 메모리에서 해당 페이지가 아직 처리 또는 대기 중임
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
        // 🌟 [ZERO-TO-REAL MIGRATION] 로그인 전 ZERO_ADDRESS 로 생성된 문서를
        //    실제 address 기반 team_id 로 일괄 이전합니다.
        let zero_addr = "0x0000000000000000000000000000000000000000";
        let old_team_id = crate::utils::hash::hash_id(zero_addr);
        let new_team_id = crate::utils::hash::hash_id(&address);
        if old_team_id != new_team_id {
            match store.migrate_team_identity(&old_team_id, &new_team_id, &address).await {
                Ok(n) => {
                    if n > 0 {
                        println!("[HUB] Migrated {} docs from ZERO team to real team.", n);
                    }
                },
                Err(e) => println!("[HUB] Migration warning: {}", e),
            }
        }
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
        store.get_all_items("pages", 1000, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let mut all_users = store.get_all_items("users", 50, 0, None).await.unwrap_or_default();
        
        
        if let Ok(team_docs) = store.get_all_items("users", 1, 0, Some("`type` = 'team'".to_string())).await {
            for t in team_docs {
                if !all_users.iter().any(|u| u.id == t.id) {
                    all_users.push(t);
                }
            }
        }
        Ok(all_users)
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
    let guard = automation::GLOBAL_BROWSER.lock().await; 
    
    
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
        let items_total = items.len();
        let mut count = 0;
        for item in items {
            
            let id = item.get("id").and_then(|v| v.as_str())
                        .or_else(|| item.get("data").and_then(|d| d.get("id")).and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();
            
            
            let type_str = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();

            
            println!("[DEBUG] Syncing item - ID: {}, Type: {}", id, type_str);
            
            
            let mut clean_item = item.clone();
            if let Some(obj) = clean_item.as_object_mut() {
                obj.insert("type".to_string(), serde_json::json!(type_str.clone()));
                // 🌟 [v4] 루트 호이스팅 복원 블록을 제거했습니다.
                //    프론트엔드(normalizeEnvelope)가 이미 봉투/확장을 분리해서 보내고,
                //    Rust 쪽 canonicalize_data 가 타입까지 확정하므로
                //    여기서 status / amount / is_masked 를 다시 끌어올릴 이유가 없습니다.
                //
                //    다만 봉투 값 중 data 안에 반드시 있어야 하는 것들(mode / 시간)은
                //    상위 페이로드에만 있을 수 있으므로 결손일 때만 보충합니다.
                if obj.get("mode").is_none() {
                    if let Some(mode) = item.get("mode") {
                        obj.insert("mode".to_string(), mode.clone());
                    } else {
                        // 🌟 [ANALYTICS MODE AUTO-TAG] 서버(D1)에서 내려온
                        //    analytics 트랙 항목은 mode 컬럼이 없습니다.
                        //    reindex_pending_embeddings 가 mode = 'analytic' 으로
                        //    필터링하므로, 여기서 자동 주입하지 않으면
                        //    동기화되어도 임베딩 대상에서 전량 탈락합니다.
                        let is_analytic_type = matches!(
                            type_str.as_str(),
                            "click" | "hover" | "change" | "report" | "question" | "answer"
                        );
                        if is_analytic_type {
                            obj.insert("mode".to_string(), serde_json::json!("analytic"));
                        }
                    }
                }
                if obj.get("created_at").is_none() {
                    if let Some(v) = item.get("created_at") { obj.insert("created_at".to_string(), v.clone()); }
                }
                if obj.get("updated_at").is_none() {
                    if let Some(v) = item.get("updated_at") {
                        obj.insert("updated_at".to_string(), v.clone());
                    } else {
                        // 🌟 [DRAFT PRESERVE] 서버 행에 updated_at 이 루트에도 data 에도 없으면
                        //    0(draft) 을 시딩합니다. 이 처리가 없으면 store.rs upsert_item 이
                        //    '키 없음 → 현재 시각' 경로를 타 draft → count 가 됩니다.
                        obj.insert("updated_at".to_string(), serde_json::json!(0));
                    }
                }
                if obj.get("flag").is_none() {
                    if let Some(v) = item.get("flag") { obj.insert("flag".to_string(), v.clone()); }
                }
            }

            
            // item 안에 "data"가 객체로 존재한다면, 그 안의 알맹이를 최상위로 끌어올립니다.
            if type_str != "talk" && type_str != "prompt" && type_str != "ai_search" {
                if let Some(data_obj) = clean_item.get("data").and_then(|v| v.as_object()).cloned() {
                    if let Some(main_obj) = clean_item.as_object_mut() {
                        for (k, v) in data_obj {
                            main_obj.insert(k, v);
                        }
                        main_obj.remove("data"); // 기존 껍데기 data 제거
                    }
                }
            }
            
            
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

                // 🌟 [CRITICAL FIX] 서버에서 받아온 talk/prompt 메시지는 items 테이블(upsert_item)이 아니라 
                // 반드시 messages 테이블(add_message)로 직접 인서트해야 화면의 get_chat_messages에서 정상 조회됩니다!
                let from_val = clean_item.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let to_val = clean_item.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let cc_val = clean_item.get("cc").and_then(|v| v.as_str()).unwrap_or("");
                let bcc_val = clean_item.get("bcc").and_then(|v| v.as_str()).unwrap_or("");
                let ref_val = clean_item.get("ref").and_then(|v| v.as_str()).unwrap_or("");
                let status_val = clean_item.get("status").and_then(|v| v.as_i64()).unwrap_or(9) as i32;

                // 🌟 [ARG SHIFT FIX] 기존 코드는 add_message 의 마지막 인자(data)에
                //    created_at 문자열을 넘기고 있었습니다.
                //      add_message(..., type_: Option<&str>, data: Option<&str>)
                //    그래서 talks.data 컬럼에 "1735689600000" 같은 숫자 문자열이 저장되고,
                //    실제 created_at 은 add_message_at 내부에서 현재 시각으로 대체되어
                //    서버에서 내려온 과거 메시지 시각이 전부 '지금' 으로 찍혔습니다.
                //    (프론트엔드 upsertChatMessages 의 시간순 정렬이 통째로 무너집니다)
                //    created_at 은 add_message_at 의 전용 인자로 분리해 넘깁니다.
                let created_at_num = clean_item.get("created_at")
                    .and_then(|v| v.as_i64())
                    .or_else(|| clean_item.get("created_at")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.trim().parse::<i64>().ok()))
                    .filter(|v| *v > 0)
                    .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

                // 🌟 data 컬럼에는 바로 위에서 조립한 { text, link, origin } 을 그대로 넣습니다.
                let data_json_str = clean_item.get("data")
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| json!({
                        "text": text_val.clone(),
                        "link": link_val.clone(),
                        "origin": origin_val.clone()
                    }).to_string());

                // Chrome.js처럼 role 구분 (프론트에서 address 비교로 재교정되므로 기본값 user 지정)
                let role_val = if type_str == "talk" { "user" } else { "system_task" };

                if !id.is_empty() {
                    // 🌟 [DELETE HANDLE] 기존에는 task_id 자리에 None 을 넘겨 전 talk 행이
                    //    빈 문자열 task_id 로 저장되었습니다. 그 결과 유일한 삭제 커맨드인
                    //      delete_message(task_id) → delete_message_by_task_id
                    //    로 '특정 한 행' 을 지목할 방법이 없었습니다.
                    //    (task_id = '' 로 지우면 서버에서 내려온 talk 전체가 날아갑니다)
                    //
                    //    낙관적 로컬 행을 서버 행으로 승계할 때 그 행만 정확히 삭제해야 하므로
                    //    자기 자신의 id 를 task_id 로 각인합니다.
                    //    · talk / prompt 는 스케줄러 태스크가 아니므로 태스크 id 와 충돌하지 않습니다.
                    //      (태스크 id 는 task_ / search_ / img_ 접두사)
                    //    · upsertChatMessages 의 isTask 판정은 task_id 가 "search_" 로 시작할 때만
                    //      참이므로 0x 주소가 들어가도 말풍선 종류가 바뀌지 않습니다.
                    let _ = db.add_message_at(
                        &id, role_val, &text_val,
                        Some(id.as_str()), Some(status_val),
                        Some(cc_val), Some(bcc_val), Some(ref_val),
                        Some(from_val), Some(to_val), Some(type_str.as_str()),
                        Some(data_json_str.as_str()),
                        Some(created_at_num)
                    ).await;
                    count += 1;
                }
                continue; // messages 테이블에 저장했으므로 하단의 items/talks 테이블 저장 로직은 건너뜀
            }

            // 🌟 [v4 ROUTING] 물리 테이블은 items / users / pages 3개뿐입니다.
            //    store.rs 의 resolve_table 과 동일한 규칙을 사용해야 저장/조회가 어긋나지 않습니다.
            //
            //    🌟 [HINT-FIRST FIX] 기존 구조는 주석에 "table 명시값을 1순위로 신뢰한다" 고
            //      적어 놓고 실제로는 type_str 매치를 먼저 돌린 뒤 fallback 으로만 썼습니다.
            //      그 결과 서버 index.ts 의 `home` 페이지 캐시
            //          { table:'pages', type:'tracking', data:{ node:true, item:true } }
            //      가 "tracking" arm 에 걸려 items 로 새어 들어갔고,
            //      reindex_pending_embeddings 의 items 스캔에 잡혀
            //      임베딩 → 청크 인덱싱 → 음차까지 전부 돌았습니다.
            //      (로그 실측: 'Its table is pages' / 'Its node is 1' / 'Its item is true' 청크 11건)
            //
            //      Client Worker 는 페이지 캐시 행에만 table:'pages' 를 실어 보내므로
            //      그 명시값을 '진짜로' 1순위에 둡니다. (추정이 아니라 계약입니다)
            let table_hint = item.get("table").and_then(|v| v.as_str()).unwrap_or("");
            let final_table = match table_hint {
                // ── 1순위 : 서버가 명시한 물리 테이블 ──
                "pages" | "page" => "pages",
                "users" => "users",
                // 🌟 [TABLE HINT GUARD] table_hint 가 "talks" 이면 messages 경로이므로 skip
                "talks" => continue,
                // ── 2순위 : 힌트가 없거나 레거시(sales/tracking/event)일 때만 type 으로 판정 ──
                _ => match type_str.as_str() {
                    "member" | "team" | "user" | "users" => "users",
                    "pages" | "page" => "pages",
                    // analytics 트랙 행동 로그 / 리포트 / 관리자 Q&A 는 무조건 items 입니다.
                    "click" | "hover" | "change" | "report" | "question" | "answer" => "items",
                    "sales" | "goods" | "order" | "tracking" | "event" | "coupon" | "review"
                    | "receiving" | "shipping" => "items",
                    // 🌟 [NON-SEARCH GUARD] 검색/음차/청크 인덱싱 대상이 아닌 타입은
                    //    items 테이블로 유입되면 reindex 스캔에서 불필요한 모델 호출을 유발합니다.
                    //    talk/prompt/ai_search 는 이미 위에서 messages 로 continue 되지만,
                    //    방어적으로 여기서도 skip 합니다.
                    "talk" | "prompt" | "ai_search" => continue,
                    // sales / tracking / event 같은 레거시 table 힌트는 전부 items 로 접습니다.
                    _ => "items",
                },
            };

            
            let from = item.get("from").and_then(|v| v.as_str());
            let to = item.get("to").and_then(|v| v.as_str());
            let cc = item.get("cc").and_then(|v| v.as_str());
            let bcc = item.get("bcc").and_then(|v| v.as_str());
            let r#ref = item.get("ref").and_then(|v| v.as_str());
            let digest = item.get("digest").and_then(|v| v.as_str());
            if !id.is_empty() {
                // 🌟 [EMBED FLAG PRESERVE] 서버에서 받은 데이터에는 embed 플래그가 없으므로,
                //    기존 DB에 이미 로컬 임베딩이 완료된(embed=1) 아이템이라면
                //    덮어쓰기 전에 해당 플래그를 clean_item에 복원합니다.
                //    이 처리가 없으면 syncData() 폴링(3초)마다 embed가 사라져
                //    reindex_pending_embeddings()가 매번 재인덱싱을 수행하는 무한 루프가 발생합니다.
                //
                //    🌟 v4 : canonicalize_data 가 embed 를 0|1 정수로 확정하므로
                //    as_i64() 비교가 안정적으로 동작합니다.
                //    (기존에는 true / "1" / 1 이 섞여 들어와 간헐적으로 실패했습니다)
                //
                //    🌟 [CC-INDEPENDENT DEDUP] cc 값이 달라도 동일 id 이면
                //    이미 임베딩된 문서로 간주합니다. digest 가 cc 포함 해시라면
                //    cc 변경 시 digest 가 달라져 스킵 가드가 무력화되므로,
                //    embed 플래그 + chunk_count 를 이중 확인합니다.
                if let Ok(Some(existing)) = db.get_item_by_id(final_table, &id).await {
                    let mut already_embedded = false;
                    if let Ok(existing_data) = serde_json::from_str::<Value>(&existing.json_data) {
                        already_embedded = existing_data.get("embed")
                            .map(|v| v.as_i64().unwrap_or(0) == 1 || v.as_bool().unwrap_or(false))
                            .unwrap_or(false);
                    }
                    // 🌟 chunk_count 가 0보다 크면 물리적으로 청크가 존재하므로
                    //    embed 플래그와 무관하게 재인덱싱 불필요
                    if !already_embedded {
                        let cc = db.count_chunks_by_item(&id).await.unwrap_or(0);
                        if cc > 0 { already_embedded = true; }
                    }
                    if already_embedded {
                        if let Some(obj) = clean_item.as_object_mut() {
                            obj.insert("embed".to_string(), json!(1));
                        }
                    }
                }
                // 원본 item 대신 세탁된 clean_item을 DB에 밀어 넣습니다.
                let _ = db.upsert_item(final_table, &id, &type_str, clean_item, None, from, to, cc, bcc, r#ref, digest).await;
                count += 1;
            }
        }
        println!(
            "[SYNC-RESULT] upsert_items 완료: 수신 {}건 → LanceDB 쓰기 {}건 (나머지는 digest 동일 스킵 또는 messages 테이블행)",
            items_total, count
        );
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
    crate::utils::sync_utils::mark_ui_ready();
    
    let store_guard = state.store.lock().await;
    let mut tasks = Vec::new();
    let mut pages = Vec::new();
    let mut users = Vec::new();
    let mut items = Vec::new();
    
    if let Some(db) = store_guard.as_ref() {
        let mut raw_tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        
        if let Ok(mut active) = db.get_processing_tasks(10).await {
            raw_tasks.append(&mut active);
        }
        
        
        let mem_task_id = if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            mem.as_ref().and_then(|v| v.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string()
        } else { 
            "".to_string() 
        };

        for t in raw_tasks {
            
            if t.status == 1 && t.id != mem_task_id {
                let error_status = crate::logic::parse_status("error");
                println!("[DB-SYNC] Zombie task detected in DB: {}. Marking as ERROR.", t.id);
                let _ = db.update_task_status(&t.id, error_status).await;
                let _ = db.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
            } 
            
            else if t.status == 1 || t.status == 10 {
                tasks.push(t);
            }
        }

        
        pages = db.get_all_items("pages", 1000, 0, None).await.unwrap_or_default();
        
        
        users = db.get_all_items("users", 50, 0, None).await.unwrap_or_default();
        
        if let Ok(team_docs) = db.get_all_items("users", 1, 0, Some("`type` = 'team'".to_string())).await {
            for t in team_docs {
                if !users.iter().any(|u| u.id == t.id) {
                    users.push(t);
                }
            }
        }
        
        items = db.get_all_items("items", 50, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
        let reachable = automation::is_browser_reachable().await;
        let guard = automation::GLOBAL_BROWSER.lock().await; 
        
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
async fn check_gpu_availability() -> serde_json::Value {
    let config = crate::utils::get_optimal_device_config();
    let mut vendor = "none";

    if !config.is_cpu {
        vendor = "amd"; // 기본적으로 CPU가 아니면 AMD(ROCm)로 가정

        // NVIDIA GPU 여부 판별 (nvml-wrapper 사용)
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            if nvml_wrapper::Nvml::init().is_ok() {
                vendor = "nvidia";
            }
        }

        #[cfg(target_os = "macos")]
        {
            vendor = "apple";
        }
    }

    serde_json::json!({
        "has_gpu": !config.is_cpu,
        "vendor": vendor
    })
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

#[tauri::command]
async fn check_model_status() -> Result<serde_json::Value, String> {
    let app_dir = crate::utils::get_app_dir();
    let base_path = app_dir.join("models");

    // 🌟 [CRITICAL FIX] 특정 폴더 내에 10MB 이상의 GGUF 또는 SAFETENSORS 파일이 존재하는지 검사하도록 조건 확장
    let has_valid_model = |dir: &std::path::PathBuf| -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            entries.flatten().any(|e| {
                let is_model_ext = e.path().extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors");
                is_model_ext && e.metadata().map(|m| m.len()).unwrap_or(0) > 10_000_000
            })
        } else {
            false
        }
    };
    
    // 🌟 [추가] Stanza 모델 체크용 함수 추가
    let has_valid_stanza = |dir: &std::path::PathBuf| -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            entries.flatten().any(|e| {
                e.path().extension().map_or(false, |ext| ext == "onnx") && e.metadata().map(|m| m.len()).unwrap_or(0) > 1_000_000
            })
        } else {
            false
        }
    };

    let qwen3_dir = base_path.join("Qwen3-0.6B-Instruct-gguf");
    let qwen3_5_dir = base_path.join("Qwen3.5-2B-Instruct-gguf");
    let granite_dir = base_path.join("granite-4.0-h-350m");
    let embed_dir = base_path.join("granite-embedding-97m-multilingual-r2");

    let mut status_map = serde_json::Map::new();
    status_map.insert("Qwen3".to_string(), serde_json::json!(has_valid_model(&qwen3_dir)));
    status_map.insert("Qwen3.5".to_string(), serde_json::json!(has_valid_model(&qwen3_5_dir)));
    status_map.insert("Granite".to_string(), serde_json::json!(has_valid_model(&granite_dir)));
    status_map.insert("Embedding".to_string(), serde_json::json!(has_valid_model(&embed_dir)));

    let supported_stanza_langs = [
        "korean", "english", "japanese", "chinese", "french", "german", "spanish", 
        "italian", "portuguese", "dutch", "russian", "arabic", "thai", "hindi", 
        "bengali", "greek", "hebrew", "vietnamese"
    ];

    for lang in supported_stanza_langs.iter() {
        let lang_code = match *lang {
            "korean" => "ko",
            "english" => "en",
            "japanese" => "ja",
            "chinese" => "zh-hans",
            "french" => "fr",
            "german" => "de",
            "spanish" => "es",
            "italian" => "it",
            "portuguese" => "pt",
            "dutch" => "nl",
            "russian" => "ru",
            "arabic" => "ar",
            "thai" => "th",
            "hindi" => "hi",
            "bengali" => "bn",
            "greek" => "el",
            "hebrew" => "he",
            "vietnamese" => "vi",
            _ => "en",
        };
        let stanza_dir = base_path.join(format!("stanza/{}", lang_code));
        status_map.insert(format!("stanza_{}", lang), serde_json::json!(has_valid_stanza(&stanza_dir)));
    }

    Ok(serde_json::Value::Object(status_map))
}

#[tauri::command]
async fn delete_all_models() -> Result<String, String> {
    let app_dir = crate::utils::get_app_dir();
    let models_dir = app_dir.join("models");
    if models_dir.exists() {
        std::fs::remove_dir_all(&models_dir).map_err(|e| e.to_string())?;
    }

    // 🌟 [VISION-CACHE] 모델이 사라지면 ViT 출력의 재현성도 보장할 수 없으므로
    //    캐시된 비전 임베딩을 함께 폐기합니다.
    crate::models::vision_cache::VISION_CACHE.clear_all();

    Ok("Deleted".to_string())
}

#[tauri::command]
async fn reset_lancedb(
    state: State<'_, AppState>,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.reset_database().await.map_err(|e| e.to_string())?;
        println!("[RESET] LanceDB fully reset completed.");
        Ok("LanceDB reset complete.".to_string())
    } else {
        // Store가 아직 초기화되지 않은 경우 직접 생성해서 리셋
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        match VectorStore::new(&db_path).await {
            Ok(s) => {
                s.reset_database().await.map_err(|e| e.to_string())?;
                println!("[RESET] LanceDB fully reset completed (fresh connection).");
                Ok("LanceDB reset complete.".to_string())
            },
            Err(e) => Err(format!("Failed to connect to LanceDB for reset: {}", e)),
        }
    }
}

#[tauri::command]
async fn download_model(app_handle: tauri::AppHandle, model_name: String) -> Result<String, String> {
    let app_dir = crate::utils::get_app_dir();
    let app_dir_clone = app_dir.clone();
    
    tokio::task::spawn(async move {
        let base_path = app_dir_clone.join("models");

        // 🌟 [수정] stanza 다운로드 경로 동적 맵핑 (models/stanza/{lang} 구조 지원)
        let folder_name = if model_name.starts_with("stanza_") {
            let lang = model_name.replace("stanza_", "");
            let lang_code = match lang.as_str() {
                "korean" => "ko",
                "english" => "en",
                "japanese" => "ja",
                "chinese" => "zh-hans",
                "french" => "fr",
                "german" => "de",
                "spanish" => "es",
                "italian" => "it",
                "portuguese" => "pt",
                "dutch" => "nl",
                "russian" => "ru",
                "arabic" => "ar",
                "thai" => "th",
                "hindi" => "hi",
                "bengali" => "bn",
                "greek" => "el",
                "hebrew" => "he",
                "vietnamese" => "vi",
                _ => "en",
            };
            format!("stanza/{}", lang_code)
        } else {
            match model_name.as_str() {
                "Qwen3" => "Qwen3-0.6B-Instruct-gguf".to_string(),
                "Qwen3.5" => "Qwen3.5-2B-Instruct-gguf".to_string(),
                "Embedding" => "granite-embedding-97m-multilingual-r2".to_string(),
                "Granite" => "granite-4.0-h-350m".to_string(),
                _ => "unknown".to_string()
            }
        };

        let dir_path = base_path.join(folder_name);
        if let Err(e) = std::fs::create_dir_all(&dir_path) {
            let _ = app_handle.emit("download_error", serde_json::json!({"model": model_name, "error": e.to_string()}));
            return;
        }
        
        // 🌟 [수정] 문자열 타입(String)으로 변환하여 다국어 url을 생성할 수 있도록 확장
        let files_to_download: Vec<(String, String)> = if model_name.starts_with("stanza_") {
            let lang = model_name.replace("stanza_", "");
            let lang_code = match lang.as_str() {
                "korean" => "ko",
                "english" => "en",
                "japanese" => "ja",
                "chinese" => "zh-hans",
                "french" => "fr",
                "german" => "de",
                "spanish" => "es",
                "italian" => "it",
                "portuguese" => "pt",
                "dutch" => "nl",
                "russian" => "ru",
                "arabic" => "ar",
                "thai" => "th",
                "hindi" => "hi",
                "bengali" => "bn",
                "greek" => "el",
                "hebrew" => "he",
                "vietnamese" => "vi",
                _ => "en",
            };
            vec![
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/vocab.json", lang_code), "vocab.json".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/pos.onnx", lang_code), "pos.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/tokenizer.onnx", lang_code), "tokenizer.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/depparse.onnx", lang_code), "depparse.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/lemma.onnx", lang_code), "lemma.onnx".to_string()),
            ]
        } else {
            match model_name.as_str() {
                "Qwen3" => vec![
                    ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf".to_string(), "Qwen3-0.6B-Q8_0.gguf".to_string())
                ],
                "Qwen3.5" => vec![
                    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-BF16.gguf".to_string(), "mmproj-BF16.gguf".to_string()),
                    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf".to_string(), "Qwen3.5-2B-Q8_0.gguf".to_string())
                ],
                "Embedding" => vec![
                    ("https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/resolve/main/model.safetensors".to_string(), "model.safetensors".to_string())
                ],
                "Granite" => vec![
                    ("https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/model.safetensors".to_string(), "model.safetensors".to_string())
                ],
                _ => vec![]
            }
        };

        let total_files = files_to_download.len();
        let client = reqwest::Client::new();
        let mut has_error = false;

        for (file_idx, (url, filename)) in files_to_download.iter().enumerate() {
            let file_path = dir_path.join(filename);
            let tmp_path = dir_path.join(format!("{}.tmp", filename));
            
            let min_size = if filename.ends_with(".gguf") || filename.ends_with(".safetensors") { 10_000_000 } else { 0 };
            if file_path.exists() && std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0) > min_size {
                let percent = (((file_idx as f64 + 1.0) / total_files as f64) * 100.0) as u32;
                let _ = app_handle.emit("download_progress", serde_json::json!({"model": model_name, "percent": percent}));
                continue;
            }

            // 🌟 [추가] 다운로드 링크 터미널 로깅 및 UI 이벤트 발송
            println!("[DOWNLOAD] 다운로드 시작: {} (URL: {})", filename, url);
            let _ = app_handle.emit("download_progress", serde_json::json!({
                "model": model_name,
                "percent": 0,
                "message": format!("다운로드 중: {} (URL: {})", filename, url)
            }));

            // 🌟 [수정] url이 &String 이므로 역참조(*) 없이 그대로 전달합니다.
            match client.get(url).send().await {
                Ok(res) => {
                    if !res.status().is_success() {
                        let _ = app_handle.emit("download_error", serde_json::json!({"model": model_name, "error": format!("HTTP {}", res.status())}));
                        has_error = true;
                        break;
                    }
                    
                    let total_size = res.content_length().unwrap_or(0) as f64;
                    let mut downloaded = 0.0;
                    
                    if let Ok(mut file) = tokio::fs::File::create(&tmp_path).await {
                        use tokio::io::AsyncWriteExt;
                        use futures::StreamExt;
                        let mut stream = res.bytes_stream();
                        let mut write_error = false;
                        
                        while let Some(chunk_result) = stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    if let Err(_) = file.write_all(&chunk).await {
                                        write_error = true; break;
                                    }
                                    downloaded += chunk.len() as f64;
                                    let file_progress = if total_size > 0.0 { downloaded / total_size } else { 0.0 };
                                    let percent = (((file_idx as f64 + file_progress) / total_files as f64) * 100.0) as u32;
                                    let _ = app_handle.emit("download_progress", serde_json::json!({"model": model_name, "percent": percent}));
                                },
                                Err(_) => { write_error = true; break; }
                            }
                        }
                        
                        if write_error {
                            let _ = std::fs::remove_file(&tmp_path);
                            has_error = true;
                            break;
                        } else {
                            let _ = std::fs::rename(&tmp_path, &file_path);
                        }
                    }
                },
                Err(_) => {
                    has_error = true;
                    break;
                }
            }
        }
        
        if !has_error {
            let _ = app_handle.emit("download_complete", serde_json::json!({"model": model_name}));
        }
    });

    Ok("Started".to_string())
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
            // [INIT] Copy model configs from local project source to AppData if they don't exist
            let app_dir = crate::utils::get_app_dir();
            let dest_models_dir = app_dir.join("models");
            
            let src_models_dir1 = std::env::current_dir().unwrap_or_default().join("models");
            let src_models_dir2 = std::env::current_dir().unwrap_or_default().join("src-tauri").join("models");
            
            let src_dir = if src_models_dir1.exists() {
                Some(src_models_dir1)
            } else if src_models_dir2.exists() {
                Some(src_models_dir2)
            } else {
                None
            };

            if let Some(src) = src_dir {
                println!("[Setup] Syncing model configs from {:?} to {:?}", src, dest_models_dir);
                let _ = crate::utils::paths::copy_model_configs(&src, &dest_models_dir);
            }

            // [INIT] AppData/tmp 내부의 kv, logs, task_data 디렉토리를 초기화하여 이전 실행의 찌꺼기 완벽 삭제
            crate::utils::paths::cleanup_temp_dirs(Some(app.handle()));

            // [INIT] KV Bake Worker (Immediate)
            crate::models::qwen::generate::init_bake_worker();

            // [FIX] Reset stop signals immediately on app startup
            let setup_cancel = app.state::<AppState>().cancellation_token.clone();
            setup_cancel.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false);

            let setup_store = app.state::<AppState>().store.clone();
            
            // spawn 대신 block_on 계열의 처리를 통해 순서를 보장합니다.
            tauri::async_runtime::block_on(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
                let _ = std::fs::create_dir_all(&db_path);
                if let Ok(s) = VectorStore::new(&db_path).await {
                    println!("[Setup] VectorStore initialized. Recovering zombie records...");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    
                    
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

            
            let status_monitor_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut last_status = String::new();
                loop {
                    let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
                    let reachable = automation::is_browser_reachable().await;
                    let guard = automation::GLOBAL_BROWSER.lock().await; 
                    
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
            
            let handle_for_event = app.handle().clone(); 

            app.listen("new-task-from-browser", move |event| {
                event_cancel.store(false, Ordering::SeqCst);
                crate::utils::set_extraction_stop_signal(false);
                let app_handle = handle_for_event.clone(); 

                if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {
                    let store_clone = event_store.clone();
                    tauri::async_runtime::spawn(async move {
                        let store_guard = store_clone.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let now = chrono::Utc::now().timestamp_millis();
                            
                            
                            let zero_addr = "0x0000000000000000000000000000000000000000";
                            let from_addr = payload_val.get("from").and_then(|v| v.as_str()).unwrap_or(zero_addr).to_string();
                            
                            let raw_to = payload_val.get("to").and_then(|v| v.as_str()).unwrap_or("");
                            let team_id = if raw_to.is_empty() || raw_to == zero_addr {
                                crate::utils::hash::hash_id(&from_addr)
                            } else {
                                raw_to.to_string()
                            };

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
                            
                            
                            let _ = app_handle.emit("task-db-registered", json!({
                                "task_id": task.id,
                                "status": task.status,
                                "created_at": task.created_at,
                                "text": msg_text
                            }));
                            crate::utils::sync_utils::notify_new_task();
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
            get_active_task_context, check_model_status, download_model, delete_all_models, reset_lancedb,
            get_query_embedding, reindex_pending_embeddings, structure_pending_analytics,
            translit_cache_respond, get_embedding_batch_for_translit
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("[APP] Application exiting. Shutting down browser...");

                // 1. 전역 브라우저 상태를 즉시 stopped으로 고정
                if let Ok(mut state) = crate::CURRENT_BROWSER_STATE.write() {
                    *state = "stopped".to_string();
                }
                crate::IS_BROWSER_LAUNCHING.store(false, std::sync::atomic::Ordering::SeqCst);

                // 2. automation::shutdown_browser()로 명시적 close() 호출 후 인스턴스 제거
                let rt = tokio::runtime::Runtime::new();
                if let Ok(rt) = rt {
                    rt.block_on(async {
                        automation::shutdown_browser().await;
                    });
                }

                println!("[APP] Browser shutdown complete.");
            }
        });
}