use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::Mutex;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use crate::store::{Task, VectorStore};
use crate::model::LogisModel;
use crate::scheduler::log_task_progress;

pub async fn process_analytic_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit("task-console-log", json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    emit_term("[Scheduler] Starting Analytic Extraction Pipeline...");
    let list_log = json!({ "category": "Analytic Processing", "summary": "Analyzing user behavior logs...", "spinner": "⠋" });
    log_task_progress(app_handle, &task.id, &list_log);
    
    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };
    
    // 1. DB에서 처리 대기 중인(updated_at = 0) 로그들을 가져옵니다. (Draft 상태)
    let filter = format!("ref = '{}' AND updated_at = 0 AND type IN ('click', 'hover', 'change')", task.r#ref);
    let raw_logs = store.get_all_items("items", 300, 0, Some(filter)).await.unwrap_or_default();
    
    if raw_logs.is_empty() {
        let payload = json!({ "task_id": task.id, "category": "Done", "summary": "No pending analytic logs found.", "spinner": "✅", "data": null });
        let _ = app_handle.emit("extraction-progress", &payload);
        log_task_progress(app_handle, &task.id, &payload);
        
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let _ = db.update_task_status(&task.id, 9).await;
            let _ = db.update_message_status(&task.id, 9, Some("Completed (Empty)")).await;
        }
        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }
        return Ok(());
    }

    // 2. LLM이 분석하기 좋게 Input 데이터를 URL(href) 단위로 그룹핑
    let mut inputs_map = serde_json::Map::new();
    let mut items_map = std::collections::HashMap::new();
    
    for log in &raw_logs {
        if let Ok(mut data_obj) = serde_json::from_str::<Value>(&log.json_data) {
            data_obj.as_object_mut().unwrap().insert("id".to_string(), json!(log.id.clone()));
            
            let href = data_obj.get("href").or_else(|| data_obj.get("link")).and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let arr = inputs_map.entry(href).or_insert(json!([]));
            arr.as_array_mut().unwrap().push(data_obj.clone());
            items_map.insert(log.id.clone(), log.clone());
        }
    }
    
    let inputs_json = json!(inputs_map).to_string();
    
    // 3. parsing.rs에서 분리된 시스템 프롬프트 가져오기
    let system_prompt = crate::parsing::analytic_report_prompt();

    let model = {
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            *model_lock = Some(LogisModel::new(app_handle.clone(), device_preference.as_deref()).await.map_err(|e| anyhow::anyhow!(e))?);
        }
        model_lock.as_ref().unwrap().clone()
    };

    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;
    
    let params = crate::openai_types::ChatCompletionParameters {
        messages: vec![
            crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage { content: system_prompt, name: None }),
            crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(inputs_json), name: None })
        ],
        model: "qwen3.5".to_string(), max_tokens: Some(4096), temperature: Some(0.1),
        ..Default::default()
    };
    
    // 4. LLM 모델 추론 실행
    let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
        gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_analytic", task.id)), None).await?
    } else {
        return Err(anyhow::anyhow!("Qwen 3.5 Generator not available"));
    };
    
    // 5. 응답 파싱 및 원본 데이터 병합
    let tracking_res = crate::parsing::parse_json_from_llm(&res_text);
    let now_ts = chrono::Utc::now().timestamp_millis();
    let team_id = if !task.to.is_empty() { task.to.clone() } else { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") };
    
    if let Some(actions) = tracking_res.get("actions").and_then(|v| v.as_object()) {
        for (_href, track) in actions {
            if let Some(records) = track.get("records").and_then(|v| v.as_array()) {
                for record in records {
                    if let Some(rec_id) = record.get("id").and_then(|v| v.as_str()) {
                        if let Some(orig_item) = items_map.get(rec_id) {
                            let summary = record.get("summary").and_then(|v| v.as_str()).unwrap_or("");
                            let action = record.get("action").and_then(|v| v.as_str()).unwrap_or("");
                            let relate = record.get("relate").and_then(|v| v.as_array()).cloned().unwrap_or(vec![]);
                            
                            let mut new_data = serde_json::from_str::<Value>(&orig_item.json_data).unwrap_or(json!({}));
                            new_data.as_object_mut().unwrap().insert("summary".to_string(), json!(summary));
                            new_data.as_object_mut().unwrap().insert("action".to_string(), json!(action));
                            new_data.as_object_mut().unwrap().insert("relate".to_string(), json!(relate));
                            
                            // [PRIVACY] 마스킹 로직은 Push 단계에서 수행되므로 원본을 유지합니다.
                            let combined_text = format!("Action: {}. Summary: {}.", action, summary);
                            let masked_text = combined_text.clone();
                            new_data.as_object_mut().unwrap().insert("text".to_string(), json!(combined_text));
                            new_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_text));

                            let emb = model.get_embedding(action.to_string()).await.unwrap_or(vec![0.0; 768]);
                            
                            // updated_at 을 now_ts 로 갱신하여 정식 데이터로 편입시킵니다.
                            let _ = store.upsert_item(
                                "items", rec_id, &orig_item.r#type, new_data, Some(emb),
                                Some(&orig_item.from), Some(&orig_item.to), Some(&orig_item.cc), Some(&orig_item.bcc), Some(&orig_item.r#ref), None
                            ).await;
                        }
                    }
                }
            }
        }
    }
    
    // 6. Report 생성 및 통합 저장
    let cross_action = tracking_res.get("cross_action_flow").and_then(|v| v.as_str()).unwrap_or("");
    let intent_evo = tracking_res.get("intent_evolution").and_then(|v| v.as_str()).unwrap_or("");
    let preferences = tracking_res.get("consistent_preferences").and_then(|v| v.as_str()).unwrap_or("");
    
    // [PRIVACY] 최종 리포트 결과물도 검색(FTS) 및 벡터화를 위해 텍스트로 추출
    let report_text = format!("Flow: {}. Intent: {}. Preferences: {}.", cross_action, intent_evo, preferences);
    let report_masked = report_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨

    let report_id = crate::utils::hash::hash_id(&format!("report_{}", now_ts));
    let report_data = json!({
        "cross_action_flow": cross_action,
        "intent_evolution": intent_evo,
        "consistent_preferences": preferences,
        "time": now_ts,
        "mode": "analytic",
        "text": report_text,
        "masked_text": report_masked
    });
    
    let _ = store.upsert_item(
        "items", &report_id, "report", report_data, None,
        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&task.bcc), Some(&task.r#ref), None
    ).await;
    
    // 7. 상태 업데이트 및 작업 종료
    let store_guard = store_mutex.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let _ = db.update_task_status(&task.id, 9).await;
        let _ = db.update_message_status(&task.id, 9, Some("Analytic Extraction Complete")).await;
    }

    let payload = json!({ "task_id": task.id, "category": "Done", "summary": "Analytic report generated.", "spinner": "✅", "data": null });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }
    Ok(())
}