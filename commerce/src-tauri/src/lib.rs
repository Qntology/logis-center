// 🌟 [APP CORE] 모델 상주 · 저장소 · 업무 규칙 · 스케줄러 · 검색 도우미는 헤드리스 코어 crate(commerce/app, logis-app)에 있습니다.
//    이 셸(Tauri 명령 · 창 · 브라우저 자동화)은 그 모듈들을 예전 이름으로 다시 내보내므로, 명령과 통합 테스트(tests/it)의
//    경로(crate::model · tauri_app_lib::store …)는 그대로입니다.
pub use logis_app::{
    js_templates, logic, model, models, parsers, prompts, scheduler, search_fusion, search_relation, stanza, store, utils,
};
pub use logis_app::{bias_schema, json_parse, nl_convert, parsing, score_dynamics, time_guide};
pub use logis_app::{chat_template, openai_types, position_embed, tokenizer};
// 검색 중 표시 · 앱 상태와 자연어 검색 본문(범위 필터 · dexie 계획 · 조건 그라운딩 · 질의 융합)은 코어에 있습니다.
pub use logis_app::{AppState, IS_SEARCHING};
pub use logis_app::query::ground_trade_conditions;
#[allow(unused_imports)]
use logis_app::query::*;
mod automation;
// 🌟 [EVENT SINK] model · scheduler 는 tauri::AppHandle 대신 Sink 로 이벤트를 내보냅니다 (명령 진입점에서 sink::tauri 로 감쌈).
pub mod sink;

use tauri::{State, Manager, Listener, Emitter}; 
use tokio::sync::Mutex as TokioMutex;
use std::sync::RwLock; 
use once_cell::sync::Lazy; 
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};


// 브라우저가 실행되는 도중 상태를 방어하기 위한 락 추가
pub static IS_BROWSER_LAUNCHING: AtomicBool = AtomicBool::new(false);


pub static CURRENT_BROWSER_STATE: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("stopped".to_string()));
pub static LAST_BROWSER_STATE_CHANGE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

// 작업 문맥 · 화면 분류 · 마지막 진행 이벤트는 코어(스케줄러 · 로거 · 모델)가 쓰므로 logis-app 루트에 있습니다.
pub use logis_app::{ACTIVE_TASK_MEM, CURRENT_UI_CATEGORY, LATEST_PROGRESS_PAYLOAD};

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
            let kv_dir = crate::utils::paths::get_kv_dir().join(id);
            let base_kv_dir = crate::utils::paths::get_kv_dir().join(format!("{}_base", id));
            let task_data_dir = crate::utils::paths::get_task_specific_dir(id);
            let pug_log_dir = crate::utils::paths::get_pug_logs_dir(id);

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

    // 🌟 [STOP-THEN-RESUME]
    // stop_current_extraction 은 "현재 작업 중단"이지 "이후 작업 영구 차단"이 아닙니다.
    // 여기서 정지 신호를 해제하지 않으면, 특히 btn-reset-db(전체 초기화) 이후
    // 전처리/추출 작업 시 스케줄러 루프가
    // "⏸️ Extraction stop signal active. Waiting for resume..." 에서
    // 영원히 벗어나지 못합니다.
    // (window.location.reload() 는 프론트엔드만 리로드하고 백엔드 프로세스는
    //  리로드되지 않으므로, 정지 신호가 백엔드에 영구 잔존합니다)
    //
    // 현재 태스크는 이미 위쪽에서:
    //   · cancellation_token = true 로 체크되어 중단되거나
    //   · deep_purge_resources / 모델 언로드 로 사실상 중단되었으므로,
    // 여기서 즉시 해제해도 안전합니다.
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

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
            println!("[UNLOAD] {}", m.crossover_report());
            m.deep_purge_resources().await;
            // 🌟 [CROSSOVER] 상주 상태는 인스턴스의 슬롯에서 읽으므로(model/residency.rs)
            //    다음 LogisModel::new 에 이전 페이즈가 남지 않습니다. 따로 IDLE 을 선언하지 않습니다.
        }
        *model_guard = None;
    }
    // 🌟 [SDS FLUSH] 언로드는 세션 경계이므로 관측을 디스크에 확정합니다.
    //    dirty 플래그가 false 면 파일을 쓰지 않으므로 불필요한 I/O 가 없습니다.
    println!("[UNLOAD] {}", crate::utils::score_dynamics::report());
    crate::utils::score_dynamics::flush();
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    state.cancellation_token.store(false, Ordering::SeqCst);
    // 🌟 [UNLOAD RESET]
    // stop_current_extraction 이 남긴 파일 기반 정지 신호까지 반드시 해제합니다.
    // 이 줄이 없으면 리로드 후 스케줄러가
    //   "⏸️ Extraction stop signal active. Waiting for resume..."
    // 에서 영원히 빠져나오지 못합니다.
    crate::utils::set_extraction_stop_signal(false);
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
            match LogisModel::new(crate::sink::tauri(&app_handle), None).await {
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
            match LogisModel::new(crate::sink::tauri(&app_handle), device_preference.as_deref()).await {
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
    // 🌟 [MODE ROUTING] commerce / shipping 트랙을 구분합니다.
    //    (웹 분석 analytic 트랙은 premium-discount 로 옮겨 이 앱에는 없습니다)
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
    let target_mode = mode.unwrap_or_else(|| "commerce".to_string());
    let mode_filter = format!("mode = '{}'", target_mode);
    // 🌟 [SCAN STARVATION FIX]
    //  ── 무엇이 문제였나 ──
    //   기존에는 offset 0 에서 500건만 읽었습니다. 정렬 기준이 고정되어 있으므로
    //   상위 500건이 전부 임베딩 완료 상태가 되면 pending 이 비고 no_pending 으로 끝나,
    //   501번째 이후의 미처리 문서는 영원히 처리되지 않았습니다.
    //   문서가 500건을 넘는 순간 로컬 임베딩이 조용히 멈춥니다.
    //   scan_limit 만큼 채울 때까지 오프셋을 밀며 순회합니다.
    // 🌟 [SINGLE SCAN]
    //
    //  ── 무엇이 문제였나 ──
    //   get_all_items 는 limit/offset 을 SQL 로 내리지 않습니다.
    //   전량을 읽어 batch_to_docs 로 행당 String 14개를 복사하고,
    //   전량 정렬한 뒤에야 슬라이스합니다.
    //   그런데 기존 루프는 이 함수를 '페이지마다' 호출했으므로
    //   페이지 수만큼 전체 테이블을 다시 읽고 다시 정렬했습니다.
    //   (문서 10,000건 × 40페이지 = 400,000행 파싱)
    //   실측 로그에서 인덱싱 20건이 도는 사이 RAM 이 3414MB → 1172MB 로
    //   2.2GB 증발한 것이 이 반복 스캔입니다.
    //
    //  ── 조기 종료 조건도 틀려 있었습니다 ──
    //   rough_pending 은 사실상 docs.len() 이고 임계치는 scan_limit*20 = 400 입니다.
    //   첫 페이지 500건에서 즉시 break 하므로 SCAN_MAX_PAGES 40 은 도달 불가능한
    //   죽은 상수였고, 'SCAN STARVATION FIX' 가 해결했다고 적은 문제
    //   (최신 500건 밖의 문서가 영원히 처리되지 않음)가 그대로 남아 있었습니다.
    //
    //  ── 해결 ──
    //   호출을 1회로 줄이고 limit 을 크게 잡습니다. 어차피 함수 내부가
    //   전량을 읽으므로, 여러 번 부르는 것은 순수한 반복 낭비입니다.
    //   상한은 '한 번에 메모리에 올릴 문서 수' 이며 페이징 의미가 아닙니다.
    const SCAN_CEILING: usize = 20_000;
    let docs: Vec<TradeDocument> = store
        .get_all_items("items", SCAN_CEILING, 0, Some(mode_filter.clone()))
        .await
        .map_err(|e| e.to_string())?;

    // 🌟 [LAZY MODEL LOAD] 대상 선별을 '모델 로드 이전' 으로 끌어올립니다.
    //    기존 구조는 LogisModel::new → ensure_embedding 을 먼저 수행한 뒤 스캔했기 때문에,
    //    처리할 문서가 0건이어도 CUDA 컨텍스트와 97M 임베딩 가중치를 매번 올렸다 내렸습니다.
    //    (로그 실측: [EMBED-LOCAL] 처리 로그가 없는데도 Loading Embedding Model 발생)
    //    이 스캔 구간은 LanceDB 조회만 사용하므로 모델이 전혀 필요 없습니다.
    let mut pending: Vec<(TradeDocument, bool)> = Vec::new();
    for doc in docs {
        if pending.len() >= scan_limit { break; }
        if state.cancellation_token.load(Ordering::Relaxed) { break; }
        if doc.id.is_empty() { continue; }
        // 🌟 [EMBED TYPE GUARD] 검색 벡터화가 무의미한 타입만 배제합니다.
        //    pages/talk/prompt/ai_search 는 검색 대상이 아니고, question/answer 는 관리자 채팅 말풍선 전용,
        //    team/user/member 는 통계 문서라 임베딩 대상이 아닙니다.
        // 🌟 [제외 타입 위임] 기존 EMBED_EXCLUDE_TYPES 배열은 talk / user 타입 목록의
        //    네 번째 복제본이었습니다. store.rs::is_embed_excluded_type 하나로 통합합니다.
        if crate::store::is_embed_excluded_type(&doc.r#type) {
            continue;
        }
        // 🌟 [MODE INTEGRITY RECHECK] SQL 필터를 신뢰하지 않고 한 번 더 봅니다.
        //
        //  ── 왜 SQL 만으로 부족한가 ──
        //   mode_filter = "mode = 'commerce'" 로 정확히 걸었는데도
        //   무역 서식 20건이 잡혔습니다. 필터가 틀린 게 아니라
        //   그 문서들의 mode 물리 컬럼 자체가 'commerce' 로 잘못 저장되어 있었습니다.
        //   (뿌리는 upsert_items 의 mode 태깅 결손 — 위에서 수정)
        //   이미 잘못 저장된 기존 문서는 그 수정으로 고쳐지지 않으므로,
        //   소비 지점에서 타입 기준으로 재검사해 격리합니다.
        //
        //  ── 판정 근거 ──
        //   canonical_bias_type() 이 'shipping_doc' 으로 접는지만 봅니다.
        //   서식 목록을 여기에 복제하지 않습니다.
        {
            let is_shipping_doc =
                crate::utils::bias_schema::canonical_bias_type(&doc.r#type) == "shipping_doc";
            if is_shipping_doc && target_mode != "shipping" {
                println!(
                    "[EMBED-LOCAL] 🚢 [MODE MISMATCH] '{}' (type='{}') 은 무역 서식인데 mode='{}' 스캔에 잡혔습니다. 잘못 태깅된 문서이므로 이 트랙에서는 건너뜁니다.",
                    doc.id, doc.r#type, target_mode
                );
                continue;
            }
        }
        if let Ok(data_val) = serde_json::from_str::<Value>(&doc.json_data) {
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
                continue;
            }
            if (target_mode == "commerce" || target_mode == "shipping")
                && crate::utils::canonical::is_relay_placeholder(&data_val)
            {
                continue;
            }
        }
        let chunk_count = store.count_chunks_by_item(&doc.id).await.unwrap_or(0);
        if chunk_count > 0 {
            println!(
                "[EMBED-LOCAL] 🧩 [CHUNK PRESENT] '{}' 는 청크 {}개가 이미 있습니다. 청크 인덱싱만 건너뛰고 문서 벡터는 이번 회차에서 채웁니다.",
                doc.id, chunk_count
            );
        }
        pending.push((doc, chunk_count == 0));
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
            match LogisModel::new(crate::sink::tauri(&app_handle), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }
        model_guard.as_ref().unwrap().clone()
    };
    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    // 🌟 [CROSSOVER] 백그라운드 인덱싱은 순수 임베딩 페이즈입니다.
    //    전경 태스크가 남긴 생성 모델이 아직 상주 중일 수 있으므로,
    //    ensure_embedding 대신 페이즈 진입으로 예산 판정을 거칩니다.
    //    여유가 있으면 생성 모델을 그대로 두고(다음 전경 작업이 빨라집니다),
    //    없으면 먼저 반환시킵니다.
    model.enter_embedding_phase("background reindex").await.map_err(|e| e.to_string())?;

    let mut processed = 0usize;
    let mut yielded_to_task = false;
    for (doc, needs_chunks) in pending {
        if state.cancellation_token.load(Ordering::Relaxed) { break; }
        // 🌟 [BUSY RECHECK / 매 반복]
        //
        //  ── 무엇이 문제였나 ──
        //   함수 상단의 ACTIVE_TASK_MEM / IS_SEARCHING 가드는 '진입 시 1회' 입니다.
        //   실측 로그 순서가 정확히 그 사각지대를 보여줍니다.
        //     [EMBED-LOCAL] 20 pending item(s) detected.   ← 가드 통과 (태스크 없음)
        //     [Scheduler] New task signal received.        ← 그 다음에 태스크 시작
        //     [Scheduler] ✅ Model Lock acquired.
        //   가드는 이미 통과한 뒤이므로 20건 루프가 태스크와 끝까지 병렬로 돕니다.
        //
        //  ── Model Lock 이 왜 못 막는가 ──
        //   위쪽에서 model_guard 를 블록 스코프로 잡았다 clone 후 즉시 놓습니다.
        //   이 for 루프는 락 없이 clone 만 들고 돕니다.
        //   LogisModel 내부는 전부 Arc 슬롯이므로 clone 은 같은 VRAM 자원을 만집니다.
        //
        //  ── 피해 ──
        //   스케줄러의 secure_vram_relay 가 퍼지 → VRAM 확보 → 로드 하는 사이
        //   이 루프의 get_embedding 이 ensure_embedding 을 호출해 임베딩을 되올립니다.
        //   그 결과 4GB 를 확보하고도 Qwen3 로드 시점엔 756MB 만 남아
        //   KV 가 RAM 으로 밀려나고([KV-PLAN] → Ram), 전환에 47초가 걸렸습니다.
        //
        //  ── 왜 break 인가 ──
        //   이 작업은 폴링으로 재호출되는 백그라운드 인덱싱입니다.
        //   남은 문서는 다음 폴링에서 처리되므로 유실이 없습니다.
        //   반면 태스크는 사용자가 기다리는 전경 작업이라 양보가 옳습니다.
        if IS_SEARCHING.load(Ordering::SeqCst)
            || crate::ACTIVE_TASK_MEM.read().map(|m| m.is_some()).unwrap_or(false)
        {
            yielded_to_task = true;
            println!(
                "[EMBED-LOCAL] ⏸️ 전경 작업(태스크/검색)이 시작되어 로컬 인덱싱을 중단하고 VRAM 을 양보합니다. (이번 회차 {}건 처리 후 중단, 잔여분은 다음 폴링에서 재개)",
                processed
            );
            break;
        }
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
        // 문서 본문(text)이 벡터의 본체입니다. 비어 있으면 JSON 을 자연어로 풀어 씁니다.
        let text = if !doc.text.trim().is_empty() {
            doc.text.clone()
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
            None, // 🌟 임베딩 경로는 텍스트 전용
            Some(&doc.from), Some(&doc.to), Some(&doc.cc), Some(&doc.bcc), Some(&doc.r#ref), Some(&digest)
        ).await;

        let link = data.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let doc_lang = crate::utils::lang_utils::detect_document_language(&text);
        let _vocab_scope = crate::models::common::vocab_trim::DocLangScope::text(&doc_lang, &text);
        // 🌟 v4 : mode 는 봉투 물리 컬럼이므로 doc.mode 가 항상 채워져 있습니다.
        //    다만 구버전 데이터를 대비해 data.mode 까지 폴백을 둡니다.
        let mode = if !doc.mode.trim().is_empty() {
            doc.mode.clone()
        } else {
            data.get("mode").and_then(|v| v.as_str()).unwrap_or("commerce").to_string()
        };

        let cancel = state.cancellation_token.clone();
        if needs_chunks {
            let _ = crate::scheduler::indexing::index_item_chunks(
                &store, &model, &doc.id, &doc.r#type, &doc_lang, &data, true,
                &doc.cc, &doc.bcc, &doc.r#ref, &mode, &link, &cancel, &crate::sink::tauri(&app_handle), "cloud_sync",
                false,
            ).await;
        }

        processed += 1;
    }

    if processed > 0 {
        println!("[EMBED-LOCAL] Cloud-synced items embedded locally: {} item(s). (mode: {})", processed, target_mode);
    }
    // 🌟 [IMMEDIATE RELEASE] 양보로 중단했다면 VRAM 반환이 이 함수의 유일한 목적입니다.
    //    unload_embedding 은 임베딩 슬롯 + 캐시 + CUDA 풀을 정리하므로
    //    스케줄러의 wait_for_vram_settle 이 곧바로 목표치를 확보할 수 있습니다.
    model.unload_embedding().await;
    // 🌟 [CROSSOVER] unload_embedding 이 내부에서 sync_crossover_phase 를 호출하므로
    //    여기서 IDLE 로 뭉뚱그릴 필요가 없어졌습니다.
    //    생성 모델이 함께 상주 중이었다면 GENERATION 이 정확한 상태이고,
    //    IDLE 로 덮어쓰면 다음 enter_generation_phase 가 이미 올라와 있는 모델을
    //    '없다' 고 판단해 불필요한 relay 를 태웁니다.
    if yielded_to_task {
        println!("[EMBED-LOCAL] ✅ 임베딩 모델을 즉시 반환했습니다. 전경 작업이 VRAM 을 온전히 사용할 수 있습니다.");
        println!("[EMBED-LOCAL] {}", model.crossover_report());
        return Ok(json!({
            "processed": processed,
            "mode": target_mode,
            "skipped": "yielded_to_foreground"
        }));
    }
    Ok(json!({ "processed": processed, "mode": target_mode }))
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
        
        let search_result = store.search_items("items", &query, query_vec, None, limit, offset, scope, false).await.map_err(|e| e.to_string());
        
        
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

// 🌟 [QUERY] 자연어 검색 본문은 코어(logis_app::query::ai_search_complex)에 있습니다. 이 명령은 상태와 Tauri 싱크를 넘기는 껍데기입니다.
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
    logis_app::query::ai_search_complex(
        &state,
        crate::sink::tauri(&app_handle),
        task_id,
        query,
        language,
        device_preference,
        search_mode,
        cc,
        bcc,
        ref_id,
    )
    .await
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
        if let Ok(m) = LogisModel::new(crate::sink::tauri(&app_handle), device_preference.as_deref()).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    // 🌟 [LOCK ORDER] model 락을 쥔 채 store 락을 잡으면
    //    ai_search_complex(store → model)와 정반대 순서가 되어 데드락이 성립합니다.
    //    현재는 ai_search_complex 가 store_guard 를 블록 스코프로 즉시 해제해서
    //    우연히 피해 가고 있을 뿐이므로, 순서 자체를 맞춰 둡니다.
    let model = model_guard.as_ref().unwrap().clone();
    drop(model_guard);

    // 1. Context Gathering: 검색 상위 문서 + 저장된 문서 전체 집계 (model/research.rs 의 Map 단계)
    let mut top_documents = String::new();
    let mut digest = crate::model::research::ResearchDigest::default();
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
        
        if let Ok(results) = store.search_items("items", &query, emb, None, 3, 0, None, false).await {
            let docs: Vec<String> = results.iter()
                .map(|(_, text, _)| format!("- {}", text))
                .collect();
            top_documents = docs.join("\n");
        }
        // 🌟 [RESEARCH MAP] 숫자는 Rust 가 전체 문서에서 셉니다. 생성 모델은 이 집계표를 해석만 합니다.
        match store.get_all_items("items", 20_000, 0, None).await {
            Ok(docs) => {
                let rows: Vec<crate::model::research::ResearchRow> = docs.iter().map(Into::into).collect();
                digest = crate::model::research::ResearchDigest::from_rows(&rows);
            }
            Err(e) => println!("[DEEP-RESEARCH] ⚠️ 전체 집계용 조회 실패. 검색 상위 문서만으로 씁니다: {}", e),
        }
    }
    // 생성하는 동안 store 를 쥐고 있지 않습니다 (예전에는 보고서 생성 내내 다른 명령의 store 접근이 막혔습니다).
    drop(store_guard);

    // 2. Run Deep Research (Reduce: 생성 한 번)
    model.run_deep_research(query, &digest, top_documents, &crate::sink::tauri(&app_handle), Some(state.cancellation_token.clone())).await.map_err(|e| e.to_string())
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
        } else if headers.get("Content-Type").map(|v| v.as_str()) == Some("application/x-www-form-urlencoded") {
            // 🌟 [FORM-ENCODED FIX] Content-Type이 form-urlencoded이면
            //    .json()이 아니라 .form()으로 전송해야 서버가 파싱할 수 있습니다.
            //    .json()은 Content-Type을 application/json으로 덮어쓰므로
            //    api.oauth.network 같은 form 기반 서버에서 500이 발생합니다.
            if let Some(obj) = b.as_object() {
                let form_data: Vec<(String, String)> = obj.iter()
                    .filter_map(|(k, v)| {
                        v.as_str().map(|s| (k.clone(), s.to_string()))
                    })
                    .collect();
                req_builder = req_builder.form(&form_data);
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
    crate::utils::network::start_signal_listener(crate::sink::tauri(&app_handle), seed);
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
        // 🌟 [SDS REBIND] 통계 스코프를 실제 팀으로 재바인딩합니다.
        //
        //  ── 왜 필요한가 ──
        //   부팅 시점에는 로그인 전이라 ZERO_ADDRESS 기반 팀으로 로드됩니다.
        //   그 상태로 쌓인 통계를 다른 팀에 그대로 적용하면
        //   기획 6-4 의 스코프 격리 원칙을 위반합니다.
        //   migrate_team_identity 가 LanceDB 문서를 이전하는 것과 같은 시점에
        //   통계도 정리합니다. 단 문서와 달리 통계는 '이전' 이 아니라 '폐기' 입니다.
        //   ZERO 팀에서 쌓인 분포가 실제 팀의 문서 분포와 같다는 보장이 없기 때문입니다.
        crate::utils::score_dynamics::rebind_team(&new_team_id);
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
    let log_path = crate::utils::paths::get_task_log_file(&task_id);
    
    
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
            
            let raw_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_string();
            // 🌟 [예약 타입 판정 위임] 기존 COMMERCE_RESERVED / ANALYTIC_RESERVED 두 배열은
            //    store.rs 의 COMMERCE_TYPES / USER_TYPES / TALK_TYPES / ANALYTIC_TYPES 와
            //    같은 집합을 세 번째로 복제한 것이었습니다.
            //    'ID'(Import Declaration) / 'CO' / 'CA' / 'PC' 처럼 커머스 소문자 타입과
            //    대소문자만 다른 무역 코드가 있으므로 예약 판정이 반드시 먼저 와야 합니다.
            let lower = raw_type.to_lowercase();
            let is_reserved = crate::store::is_reserved_type(&lower);
            let type_str = if is_reserved {
                lower
            } else {
                match crate::utils::bias_schema::canonical_trade_doc_code(&raw_type) {
                    Some(canon) => {
                        if raw_type != canon {
                            println!(
                                "[SYNC] 🚢 [TYPE NORMALIZE] id='{}' type='{}' → '{}' (무역 서식 표준형)",
                                id, raw_type, canon
                            );
                        }
                        canon.to_string()
                    }
                    None => lower,
                }
            };

            
            println!("[DEBUG] Syncing item - ID: {}, Type: {}", id, type_str);
            
            
            let mut clean_item = item.clone();
            if let Some(obj) = clean_item.as_object_mut() {
                obj.insert("type".to_string(), serde_json::json!(type_str.clone()));

                if obj.get("mode").is_none() {
                    if let Some(mode) = item.get("mode") {
                        obj.insert("mode".to_string(), mode.clone());
                    } else {
                        // 🌟 [MODE 자동 태깅 위임] 기존 인라인 판정은 'touch' 가 빠져 있어
                        //    touch 이벤트가 commerce 로 태깅되었습니다.
                        //    (store.rs 의 추론이 뒤에서 덮어써 우연히 동작하고 있었을 뿐입니다)
                        //    이제 store.rs::infer_mode 하나가 유일한 판정자입니다.
                        let inferred = crate::store::infer_mode(type_str.as_str());
                        if inferred != "commerce" {
                            println!(
                                "[SYNC] 🧭 [MODE AUTO-TAG] id='{}' type='{}' → mode='{}'",
                                id, type_str, inferred
                            );
                            obj.insert("mode".to_string(), serde_json::json!(inferred));
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
            // 🌟 [TABLE HINT GUARD] table_hint 가 "talks" 이면 messages 경로이므로 skip
            if table_hint == "talks" { continue; }
            // 🌟 [NON-SEARCH GUARD] 검색/음차/청크 인덱싱 대상이 아닌 타입은
            //    items 테이블로 유입되면 reindex 스캔에서 불필요한 모델 호출을 유발합니다.
            //    talk/prompt/ai_search 는 이미 위에서 messages 로 continue 되지만,
            //    방어적으로 여기서도 skip 합니다.
            if crate::store::is_talk_type(type_str.as_str()) { continue; }
            // ── 1순위 : 서버가 명시한 물리 테이블. 2순위 : type 으로 판정 ──
            //    두 경로 모두 store.rs::resolve_table_for 를 통과하므로,
            //    upsert 가 쓴 테이블과 get_item_by_id / delete_item 이 여는 테이블이
            //    구조적으로 어긋날 수 없습니다.
            let final_table = if table_hint.is_empty() {
                crate::store::resolve_table_for(type_str.as_str())
            } else {
                crate::store::resolve_table_for(table_hint)
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
                let _ = db.upsert_item(final_table, &id, &type_str, clean_item, None, None, from, to, cc, bcc, r#ref, digest).await;
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
        vendor = if config.device.is_cuda() {
            "nvidia"
        } else if config.device.is_rocm() {
            "amd"
        } else if config.device.is_vulkan() {
            "vulkan"
        } else if config.device.is_metal() {
            "apple"
        } else {
            "none"
        };
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
    
    let qwen3_dir = base_path.join("Qwen3-0.6B-Instruct-gguf");
    let qwen3_5_dir = base_path.join("Qwen3.5-2B-Instruct-gguf");
    let granite_dir = base_path.join("granite-4.0-h-350m");
    let embed_dir = base_path.join("granite-embedding-97m-multilingual-r2");

    let siglip2_dir = base_path.join("siglip2-so400m-patch16-naflex");

    let mut status_map = serde_json::Map::new();
    status_map.insert("Qwen3".to_string(), serde_json::json!(has_valid_model(&qwen3_dir)));
    status_map.insert("Qwen3.5".to_string(), serde_json::json!(has_valid_model(&qwen3_5_dir)));
    status_map.insert("Granite".to_string(), serde_json::json!(has_valid_model(&granite_dir)));
    status_map.insert("Embedding".to_string(), serde_json::json!(has_valid_model(&embed_dir)));
    status_map.insert("SigLIP2".to_string(), serde_json::json!(has_valid_model(&siglip2_dir)));

    // 🌟 [STANZA] 변환본이 있는 12개 언어 (stanza-onnx-export format 2). vocab.json 과 그 파일 목록이 모두 있어야 설치로 봅니다.
    for (lang, code) in crate::stanza::UI_LANGS.iter() {
        let stanza_dir = base_path.join("stanza").join(code);
        status_map.insert(format!("stanza_{}", lang), serde_json::json!(crate::stanza::is_installed(&stanza_dir)));
    }

    Ok(serde_json::Value::Object(status_map))
}

#[tauri::command]
async fn get_lang_llm_status(language: Option<String>) -> Result<serde_json::Value, String> {
    Ok(crate::model::lang_llm::status_report(language.as_deref()))
}

#[tauri::command]
async fn delete_all_models() -> Result<String, String> {
    let app_dir = crate::utils::get_app_dir();
    let models_dir = app_dir.join("models");
    // 매핑 중인 파일은 Windows 에서 지울 수 없습니다.
    crate::models::siglip2::gguf::close_handles(None);
    if models_dir.exists() {
        std::fs::remove_dir_all(&models_dir).map_err(|e| e.to_string())?;
    }

    crate::models::vision_cache::VISION_CACHE.clear_all();
    crate::models::siglip2::layout_cache::LAYOUT_CACHE.clear_all();

    Ok("Deleted".to_string())
}

#[tauri::command]
async fn reset_lancedb(
    state: State<'_, AppState>,
) -> Result<String, String> {
    crate::utils::score_dynamics::purge();
    crate::models::siglip2::layout_cache::LAYOUT_CACHE.clear_all();
    let mut store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.reset_database().await.map_err(|e| e.to_string())?;
        println!("[RESET] LanceDB fully reset completed.");
        Ok("LanceDB reset complete.".to_string())
    } else {
        // 🌟 Store 가 비어 있는 상태 (예: unload_model 직후) 입니다.
        //    새 커넥션을 만들고 반드시 주입해야 스케줄러가 태스크를 볼 수 있습니다.
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&db_path);
        match VectorStore::new(&db_path).await {
            Ok(s) => {
                s.reset_database().await.map_err(|e| e.to_string())?;
                // 🌟 [TABLE RE-CREATE] reset_database 는 전 테이블을 드롭합니다.
                //    직후 초기화하지 않으면 add_task / add_message 가 실패합니다.
                let _ = s.init_task_table().await;
                let _ = s.init_all_tables().await;
                let _ = s.cleanup_unfinished_tasks_on_startup().await;
                // 🌟 [RE-INJECT] 스케줄러가 들고 있는 Arc 에 주입합니다.
                //    이 줄이 없으면 스케줄러는 영원히 store = None 입니다.
                *store_guard = Some(s);
                println!("[RESET] LanceDB fully reset completed (fresh connection). Store re-injected into scheduler.");
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
            format!("stanza/{}", crate::stanza::lang_code(&lang).unwrap_or("en"))
        } else {
            match model_name.as_str() {
                "Qwen3" => "Qwen3-0.6B-Instruct-gguf".to_string(),
                "Qwen3.5" => "Qwen3.5-2B-Instruct-gguf".to_string(),
                "Embedding" => "granite-embedding-97m-multilingual-r2".to_string(),
                "Granite" => "granite-4.0-h-350m".to_string(),
                "SigLIP2"  => "siglip2-so400m-patch16-naflex".to_string(),
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
            let code = crate::stanza::lang_code(&lang).unwrap_or("en");
            // 이전 판(format 1) 폴더는 이름을 바꿔 둡니다. 같은 이름(pos.onnx …)의 옛 파일이 있으면 아래 루프가 받지 않고 건너뜁니다.
            let local_vocab = dir_path.join("vocab.json");
            if local_vocab.exists() && !crate::stanza::is_installed(&dir_path) {
                let backup = dir_path.with_file_name(format!("{}.format1", code));
                let _ = std::fs::remove_dir_all(&backup);
                if std::fs::rename(&dir_path, &backup).is_ok() {
                    println!("[DOWNLOAD] 이전 Stanza 변환본을 {:?} 로 옮겼습니다", backup);
                }
                let _ = std::fs::create_dir_all(&dir_path);
            }
            let base = crate::stanza::remote_base(code);
            match crate::stanza::remote_manifest(&base).await {
                Ok(files) => files.into_iter().map(|f| (format!("{}/{}", base, f), f)).collect(),
                Err(e) => {
                    let _ = app_handle.emit("download_error", serde_json::json!({"model": model_name, "error": format!("{:#}", e)}));
                    return;
                }
            }
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
                "SigLIP2" => vec![
                    ("https://huggingface.co/google/siglip2-so400m-patch16-naflex/resolve/main/model.safetensors".to_string(), "model.safetensors".to_string()),
                    ("https://huggingface.co/google/siglip2-so400m-patch16-naflex/resolve/main/config.json".to_string(), "config.json".to_string()),
                    ("https://huggingface.co/google/siglip2-so400m-patch16-naflex/resolve/main/preprocessor_config.json".to_string(), "preprocessor_config.json".to_string()),
                    ("https://huggingface.co/google/siglip2-so400m-patch16-naflex/resolve/main/tokenizer.json".to_string(), "tokenizer.json".to_string())
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
fn app_context() -> tauri::Context {
    tauri::generate_context!()
}

/// 🌟 [SYNTH EVAL] 창과 이벤트 루프 없이 AppHandle 만 만듭니다. run() 을 부르지 않으므로 설정의 창이 뜨지 않고,
///   process_task · ai_search_complex 를 앱과 같은 경로로 부를 수 있습니다 (tests/it/synthetic_eval.rs).
#[cfg(any(windows, target_os = "linux"))]
pub fn build_headless_app(state: AppState) -> tauri::Result<tauri::App> {
    crate::model::lang_llm::register_loader(crate::models::qwen4_exp::registry::load_lang_runtime);
    crate::models::qwen::generate::init_bake_worker();
    tauri::Builder::default().any_thread().manage(state).build(app_context())
}

/// 🌟 [SYNTH EVAL] 프론트엔드 없이 ai_search_complex 를 부릅니다. build_headless_app 으로 만든 앱에서만 씁니다.
#[cfg(any(windows, target_os = "linux"))]
pub async fn eval_ai_search(
    app: &tauri::AppHandle,
    task_id: String,
    query: String,
    language: String,
    search_mode: String,
    cc: String,
) -> Result<Value, String> {
    ai_search_complex(
        app.state::<AppState>(),
        app.clone(),
        task_id,
        query,
        language,
        None,
        search_mode,
        cc,
        String::new(),
        String::new(),
    )
    .await
}

// 로컬 네트워크 도우미는 코어(utils::network)에 있고, Tauri 명령 껍데기만 여기 둡니다 (명령 이름 · 인자는 예전 그대로).
#[tauri::command]
fn get_local_network_prefix() -> String {
    crate::utils::network::get_local_network_prefix()
}

#[tauri::command]
fn get_my_full_ip() -> String {
    crate::utils::network::get_my_full_ip()
}

pub fn run() {
    // 모델이 띄우는 비동기 작업(KV 슬롯 반환 · 굽기 · 지연 해제)도 예전처럼 Tauri 전역 런타임에서 돕니다.
    logis_llm::rt::set_spawner(|task| {
        tauri::async_runtime::spawn(task);
    });
    crate::model::lang_llm::register_loader(crate::models::qwen4_exp::registry::load_lang_runtime);
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
            crate::utils::paths::cleanup_temp_dirs();

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
                    // 🌟 [MODE MIGRATION] type 과 mode 가 어긋난 기존 문서를 1회 교정합니다.
                    //    대상이 0건이면 조회 1회로 끝나므로 매 시작마다 돌아도 부담이 없고,
                    //    교정된 문서는 embed 마커가 제거되어 reindex 가 자동으로 재인덱싱합니다.
                    let _ = s.migrate_mode_by_type().await;
                    // 🌟 [SDS LOAD] 점수 동역학 통계를 세션 시작 시 1회 불러옵니다.
                    //
                    //  ── 왜 여기인가 ──
                    //   VectorStore 초기화와 같은 블록이어야 '데이터 계층이 준비되는 시점'
                    //   이라는 의미가 코드 배치로 드러납니다.
                    //   이 시점에는 아직 로그인 전이라 팀이 ZERO_ADDRESS 기반일 수 있고,
                    //   initialize_hub 가 실제 팀을 확정하면 rebind_team 이 재바인딩합니다.
                    //
                    //  ── 파일이 없으면 ──
                    //   냉간 시작 로그만 남기고 빈 통계로 출발합니다.
                    //   ASE 는 관측 부족으로 None 을 돌려주므로 판정은 현행 그대로입니다.
                    let zero_team = crate::utils::hash::hash_id(
                        "0x0000000000000000000000000000000000000000"
                    );
                    crate::utils::score_dynamics::load(&zero_team);
                    
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
            let scheduler_handle = crate::sink::tauri(app.handle());
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
                        } else {
                            // 🌟 [TASK DROP DIAGNOSTIC]
                            //  ── 무엇이 문제였나 ──
                            //   store 가 None 일 때 이 리스너는 로그 없이 태스크를 버렸습니다.
                            //   사용자는 이미지가 큐에 들어간 것처럼 보이지만,
                            //   실제로는 DB 에 아무것도 저장되지 않은 상태였습니다.
                            //   이 로그가 찍히면 reset_lancedb 의 재주입이 실패했다는 증거입니다.
                            let task_id = payload_val.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let task_type = payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                            println!(
                                "[TASK DROP] 🚨 new-task-from-browser 가 수신했으나 Store 가 None 입니다. \
                                 태스크 '{}' (type='{}') 가 DB 에 저장되지 않았습니다. \
                                 팩토리 리셋 후 reset_lancedb 가 Store 를 재주입했는지 확인하세요.",
                                task_id, task_type
                            );
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
            save_mobile_temp_file, get_local_network_prefix, get_my_full_ip, connect_with_seed, start_listener_command, send_signal_offer, submit_signal_answer,
            get_active_task_context, check_model_status, download_model, delete_all_models, reset_lancedb,
            get_query_embedding, reindex_pending_embeddings,
            translit_cache_respond, get_embedding_batch_for_translit,
            get_lang_llm_status
        ])
        .build(app_context())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("[APP] Application exiting. Shutting down browser...");
                // 🌟 [SDS FLUSH] 종료 경로에서도 관측을 확정합니다.
                //    언로드를 거치지 않고 창을 닫는 경로가 존재하므로
                //    두 지점 모두에 flush 가 있어야 관측이 유실되지 않습니다.
                //    flush 는 dirty 일 때만 파일을 쓰므로 중복 호출이 무해합니다.
                println!("[APP] {}", crate::utils::score_dynamics::report());
                crate::utils::score_dynamics::flush();
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

                // 3. 🌟 [ROCM-EXIT] Windows ROCm: rocBLAS/HIP DLL 해제 중 멈춤을 피해 여기서 바로 끝냅니다.
                crate::utils::device_utils::rocm_exit_now_if_used(0);
            }
        });
}
