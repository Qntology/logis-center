use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::Mutex;
use std::time::Duration;
use tokio::time::sleep;
use serde_json::json;
use crate::store::VectorStore;
use crate::model::LogisModel;
use crate::scheduler::{PROGRESS_TX, process_task};
use crate::utils::logger::log_task_progress;
use tauri::Emitter;

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let _ = PROGRESS_TX.set(ptx);
    let app_handle_prog = app_handle.clone();
    tokio::spawn(async move {
        use tauri::Emitter;
        while let Some(payload) = prx.recv().await {
            if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
                *w = Some(payload.clone());
            }
            let _ = app_handle_prog.emit("extraction-progress", &payload);
        }
    });
   
    tokio::spawn(async move {
        if !crate::utils::sync_utils::UI_READY_FLAG.load(std::sync::atomic::Ordering::SeqCst) {
            crate::utils::sync_utils::UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        let mut oom_retry_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        // 🌟 [RESUME-ON-START] 워커 (재)진입 = 앱 시작 또는 팩토리 리셋 직후.
        //    리셋/정지가 남겨둔 정지 신호를 여기서 한 번 해제합니다.
        //    이 플래그를 끄는 기존 코드는 태스크 처리 후에만 도달하는데,
        //    플래그가 켜져 있으면 도달할 수 없어 영구 정지되었습니다.
        crate::utils::set_extraction_stop_signal(false);
        cancellation_token.store(false, Ordering::SeqCst);
        let mut stopped_logged = false;
        // 🌟 [SILENT PATH INSTRUMENT] 워커 루프에서 로그를 남기지 않는 구간을 전부 계측합니다.
        //
        //  ── 실측 사고 ──
        //   팩토리 리셋 직후 이미지 태스크가 PENDING 에서 영원히 멈췄는데,
        //   스케줄러 로그가 단 한 줄도 남지 않아 원인 추적이 불가능했습니다.
        //   루프 전체에서 무음인 곳은 아래 세 곳뿐입니다.
        //     A. store 가 None      → `if let Some(db)` 이 else 없이 통과
        //     B. store.lock() 영구 대기 → 다른 태스크가 가드를 쥐고 놓지 않음
        //     C. 낡은 커넥션이 Ok(빈 배열) 반환 → 정상처럼 보이며 계속 슬립
        //   셋 다 증상이 동일하므로 로그로 구분하지 못하면 고칠 수도 없습니다.
        //
        //  ── 왜 플래그로 1회만 찍는가 ──
        //   루프는 1~10초마다 돌므로 매 회 출력하면 터미널이 로그로 뒤덮입니다.
        //   상태가 바뀔 때만 찍고, 복구되면 복구 사실도 한 번 찍습니다.
        let mut store_missing_logged = false;
        let mut lock_starved_logged = false;
        let mut idle_cycles: u32 = 0;
        loop {
            if crate::utils::is_extraction_stopped() {
                // 🌟 [HEARTBEAT] 의도적 정지 상태임을 로그에 1회 남겨
                //    "크래시/행" 과 "대기" 를 구분 가능하게 합니다.
                if !stopped_logged {
                    println!("[Scheduler] ⏸️ Extraction stop signal active. Waiting for resume...");
                    stopped_logged = true;
                }
                // 🌟 [STOP-WAIT WITH SIGNAL] 정지 대기 중에도 새 태스크 신호를 감지합니다.
                //    기존 500ms sleep 은 TASK_QUEUED_SIGNAL 을 확인할 기회를 주지 않아
                //    팩토리 리셋 후 새 태스크가 영원히 처리되지 않았습니다.
                //    ①번(정지 체크)에서 ③번(AUTO-RESUME)까지 도달할 수 없는 구조적 사각지대를
                //    정지 대기 블록 안에서 직접 해소합니다.
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                    _ = crate::utils::sync_utils::TASK_QUEUED_SIGNAL.notified() => {
                        println!("[Scheduler] ▶️ New task signal while stop signal active. Auto-resuming extraction.");
                        crate::utils::set_extraction_stop_signal(false);
                        cancellation_token.store(false, Ordering::SeqCst);
                    }
                }
                continue;
            }
                        stopped_logged = false;
            let mut pending_tasks = Vec::new();
            {
                // 🌟 [LOCK TIMEOUT] store.lock().await 는 대기 중 아무 것도 출력하지 않습니다.
                //    리셋 명령이 가드를 쥔 채 끝나지 않으면 워커는 영구 정지하면서
                //    로그상으로는 '아무 일도 없는' 것처럼 보입니다.
                //    3초 상한을 두고 초과하면 그 사실을 알립니다.
                //    타임아웃 시 lock 퓨처는 드롭되며, tokio Mutex 는 대기 취소가 안전합니다.
                match tokio::time::timeout(Duration::from_secs(3), store.lock()).await {
                    Ok(store_opt) => {
                        if lock_starved_logged {
                            println!("[Scheduler] ✅ [STORE LOCK RECOVERED] Store 뮤텍스를 다시 획득했습니다. 폴링을 재개합니다.");
                            lock_starved_logged = false;
                        }
                        match store_opt.as_ref() {
                            Some(db) => {
                                if store_missing_logged {
                                    println!("[Scheduler] ✅ [STORE RESTORED] Store 가 다시 주입되었습니다. 태스크 폴링을 재개합니다.");
                                    store_missing_logged = false;
                                }
                                match db.get_pending_tasks(5).await {
                                    Ok(tasks) => {
                                        let raw = tasks.len();
                                        pending_tasks = tasks.into_iter().filter(|t| t.r#type != "ai_search").collect::<Vec<_>>();
                                        // 🌟 [POLL TRACE] '조회는 성공했는데 0건' 인 상태를 명시적으로 남깁니다.
                                        //    낡은 커넥션이 새로 들어온 행을 보지 못하는 경우가 여기 해당합니다.
                                        //    idle_cycles 로 게이팅하여 평상시에는 조용합니다.
                                        if raw == 0 && idle_cycles > 0 && idle_cycles % 6 == 0 {
                                            println!(
                                                "[Scheduler] 💤 [POLL HEARTBEAT] Store 정상 · 조회 성공 · PENDING 0건 (연속 {}회). 태스크가 다른 커넥션에 저장되었을 수 있습니다.",
                                                idle_cycles
                                            );
                                        }
                                        if raw > 0 && pending_tasks.is_empty() {
                                            println!(
                                                "[Scheduler] ⚪ [POLL FILTERED] PENDING {}건을 조회했으나 전부 ai_search 라 처리 대상이 없습니다.",
                                                raw
                                            );
                                        }
                                    },
                                    Err(e) => println!("[Scheduler] Failed to fetch tasks: {:?}", e),
                                }
                            }
                            None => {
                                // 🌟 [STORE MISSING] 팩토리 리셋이 store 를 비운 뒤 재주입하지 않은 상태입니다.
                                //    이 경로가 기존 코드에서 완전히 무음이었고, 리셋 후 태스크가
                                //    PENDING 에서 멈추는 증상의 유력한 원인입니다.
                                if !store_missing_logged {
                                    println!("[Scheduler] 🚨 [STORE MISSING] 스케줄러가 들고 있는 VectorStore 가 None 입니다. 팩토리 리셋이 새 커넥션을 워커의 store 핸들에 다시 주입하지 않았습니다. 이 상태에서는 어떤 태스크도 처리되지 않습니다.");
                                    store_missing_logged = true;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // 🌟 [LOCK STARVED] 다른 태스크가 store 가드를 쥐고 놓지 않는 상태입니다.
                        //    리셋 명령이 가드를 잡은 채 .await 를 하거나 조기 반환하면 발생합니다.
                        if !lock_starved_logged {
                            println!("[Scheduler] 🔒 [STORE LOCK STARVED] Store 뮤텍스를 3초 이상 획득하지 못했습니다. 다른 작업(팩토리 리셋 등)이 가드를 점유한 채 놓지 않고 있습니다.");
                            lock_starved_logged = true;
                        }
                    }
                }
            }

            if pending_tasks.is_empty() {
                // 🌟 [IDLE COUNTER] 위 POLL HEARTBEAT 의 게이팅 축입니다.
                //    처리할 태스크가 생기면 아래 else 에서 0 으로 되돌립니다.
                idle_cycles = idle_cycles.saturating_add(1);
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => {
                        delay_secs = (delay_secs + 1).min(10);
                    }
                    _ = crate::utils::sync_utils::TASK_QUEUED_SIGNAL.notified() => {
                        delay_secs = 1;
                        println!("[Scheduler] New task signal received. Waking up immediately.");
                        // 🌟 [AUTO-RESUME] 새 태스크 신호 = 처리 요구.
                        //    리셋 잔여 정지 신호가 있으면 여기서 해제합니다.
                        if crate::utils::is_extraction_stopped() {
                            println!("[Scheduler] ▶️ New task signal while stop signal active. Auto-resuming extraction.");
                            crate::utils::set_extraction_stop_signal(false);
                            cancellation_token.store(false, Ordering::SeqCst);
                        }
                    }
                }
                continue;
            } else {
                delay_secs = 1;
                idle_cycles = 0;
            }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) {
                    println!("[Scheduler] Cancellation detected before starting task {}, skipping batch.", task.id);
                    break;
                }

                println!("[Scheduler] Processing task: {}", task.id);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {

                        let _ = db.update_task_status(&task.id, 1).await;
                        let _ = db.update_message_status(&task.id, 1, Some("Processing...")).await;
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                            *w = Some(json!({ "id": task.id, "ref": task.r#ref, "status": 1 }));
                        }
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { 
                            if let Some(task_val) = w.as_mut() {
                                if let Some(obj) = task_val.as_object_mut() {
                                    obj.insert("status".to_string(), json!(9));
                                    obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                            }
                        }

                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }
                        
                        let store_guard = store.lock().await;
                        
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                            let _ = db.update_message_status(&task.id, crate::logic::parse_status("complete"), Some("Task Completed")).await;
                        }

                        current_device_pref = None; 
                        oom_retry_map.remove(&task.id);
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }

                        {
                            let mut model_lock: tokio::sync::MutexGuard<Option<LogisModel>> = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                println!("[Scheduler] Error detected. Performing emergency memory release...");
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }

                        if err_msg.contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             

                             current_device_pref = None;
                             continue;
                        } else if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                            let retries = oom_retry_map.entry(task.id.clone()).or_insert(0);
                            
                            if *retries == 0 {
                                *retries += 1;
                                println!("[Scheduler] OOM Detected! VRAM is purged. Retrying on GPU...");
                                current_device_pref = None;

                                
                                let payload = json!({
                                    "task_id": task.id,
                                    "category": "Warning", "summary": "Memory pressure detected. VRAM cleared. Retrying on GPU...", "spinner": "♻️"
                                });
                                let _ = app_handle.emit("extraction-progress", &payload);

                                
                                let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                let _ = std::fs::remove_file(&log_path);
                                
                                {
                                    let store_guard = store.lock().await;
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, 10).await;
                                        let _ = db.update_message_status(&task.id, 10, Some("Retrying on GPU...")).await;
                                    }
                                }
                                
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                continue; 
                            } else {
                                if task.r#type == "image_extraction" {
                                    let final_err = "High-resolution image exceeds VRAM capacity. Please try a smaller image.";
                                    println!("[Scheduler] GPU retry failed for Vision. Throwing error instead of freezing on CPU.");
                                    let store_guard = store.lock().await;                            
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                        let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", final_err))).await;
                                    }
                                    let _ = app_handle.emit("extraction-progress", json!({
                                        "task_id": task.id,
                                        "category": "Error", "summary": final_err, "spinner": "❌"
                                    }));
                                    current_device_pref = None;
                                } else {
                                    println!("[Scheduler] OOM Detected twice! Activating CPU Mode for text task.");
                                    current_device_pref = Some("cpu".to_string());


                                    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                    let _ = std::fs::remove_file(&log_path);

                                    log_task_progress(&app_handle, &task.id, &json!({
                                        "category": "Warning", "summary": "Memory pressure detected. Retrying with CPU Mode...", "spinner": "💾"
                                    }));
                                    
                                    {
                                        let store_guard = store.lock().await;
                                        if let Some(db) = store_guard.as_ref() {
                                            let _ = db.update_task_status(&task.id, 10).await;
                                            let _ = db.update_message_status(&task.id, 10, Some("Retrying in CPU Mode...")).await;
                                        }
                                    }
                                    
                                    tokio::time::sleep(Duration::from_secs(2)).await;
                                    continue;
                                }
                            }
                        } else {
                            let store_guard = store.lock().await;                            
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                            
                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error", "summary": format!("Failed: {}", err_msg), "spinner": "❌"
                            }));

                            current_device_pref = None;
                        }
                    }
                }
            }
            
            cancellation_token.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false); 
        }
    });
}
