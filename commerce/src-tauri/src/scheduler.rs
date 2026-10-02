use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::utils::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};
use once_cell::sync::{Lazy, OnceCell};
use std::sync::Mutex as StdMutex;

pub mod translit;
pub mod indexing;
mod worker;
mod entity;
pub mod relay_ledger;
pub mod trading;
pub mod list_census;

use crate::scheduler::translit::{generate_transliteration_aliases, transliterate_cross_language};
use crate::scheduler::indexing::{upsert_alias_chunks, index_item_chunks, save_item};
use crate::scheduler::trading::process_trading_task;
use crate::utils::logger::log_task_progress;
use crate::utils::ai_utils::*;
use crate::utils::pug_utils::*;
use crate::utils::json_utils::merge_node;
use crate::js_templates::*;

pub use worker::start_background_worker;
pub use entity::{normalize_entity_key, entity_index, entity_id, entity_bcc, entity_seed, entity_key_index, relay_type_key, relay_seed};

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();

pub static TRANSLIT_PENDING: Lazy<StdMutex<std::collections::HashMap<String, tokio::sync::oneshot::Sender<Vec<(String, String)>>>>> =
    Lazy::new(|| StdMutex::new(std::collections::HashMap::new()));

pub static TRANSLIT_MEM_CACHE: Lazy<StdMutex<std::collections::HashMap<String, (String, String)>>> =
    Lazy::new(|| StdMutex::new(std::collections::HashMap::new()));

fn best_thead_th_dbg(sel: &str, html: &str) -> usize {
    if sel.is_empty() { return 0; }
    let doc = scraper::Html::parse_document(html);
    let q = match scraper::Selector::parse(sel) { Ok(q) => q, Err(_) => return 0 };
    let th = match scraper::Selector::parse("th") { Ok(q) => q, Err(_) => return 0 };
    doc.select(&q).next().map(|e| e.select(&th).count()).unwrap_or(0)
}

fn derive_thead_selector(html: &str, target_selector: &str) -> Option<(String, String)> {
    let doc = scraper::Html::parse_document(html);
    let row_q = scraper::Selector::parse(target_selector).ok()?;
    let row_el = doc.select(&row_q).next()?;
    let mut table_el: Option<scraper::ElementRef> = None;
    let mut cur = row_el.parent();
    while let Some(node) = cur {
        if let Some(e) = node.value().as_element() {
            let n = e.name().to_lowercase();
            if n == "table" {
                table_el = scraper::ElementRef::wrap(node);
                break;
            }
            if n == "body" || n == "html" {
                break;
            }
        }
        cur = node.parent();
    }
    let tbl = table_el?;
    let tv = tbl.value();
    let tsel = match tv.id() {
        Some(id) => format!("table#{}", id),
        None => {
            let mut s = String::from("table");
            for c in tv.classes().take(3) {
                s.push('.');
                s.push_str(c);
            }
            s
        }
    };
    let th_q = scraper::Selector::parse("th").ok()?;
    let td_q = scraper::Selector::parse("td").ok()?;
    let own_head = scraper::Selector::parse("thead").ok().and_then(|q| tbl.select(&q).next());
    let (cand, own) = match own_head {
        Some(h) => (format!("{} thead", tsel), h),
        None => {
            let tr_q = scraper::Selector::parse("tr").ok()?;
            (format!("{} tr", tsel), tbl.select(&tr_q).next()?)
        }
    };
    let cand_q = scraper::Selector::parse(&cand).ok()?;
    let first = doc.select(&cand_q).next()?;
    if first.id() != own.id() {
        return None;
    }
    let th = first.select(&th_q).count();
    let td = first.select(&td_q).count();
    if th >= 2 && th > td {
        Some((cand, tsel))
    } else {
        None
    }
}

/// 🌟 [PREJUDICE PAIR] 영어 편견 문장 벡터에 문서 언어 편견 문장 벡터를 더합니다.
///  두 벡터 중 최댓값(max_pool_sim)으로 재므로 단일 문장 코사인에 맞춰 둔 임계 눈금이 그대로 유지됩니다.
///  수십 구의 뱅크로 바꾸면 최댓값이 구조적으로 올라가 0.35 / 0.38 / 0.50 / 0.55 임계가 한꺼번에 어긋납니다.
const SYNTHESIS_SKIP_DROP: f32 = 0.8;

async fn run_synthesis_pass(
    model: &LogisModel,
    items: &mut Vec<Value>,
    jobs: &[(usize, String, String)],
    item_pos: &std::collections::HashMap<usize, usize>,
    page_type: &str,
    doc_lang: &str,
    probe_text: &str,
    skip_drop: f32,
    cancel: &Arc<AtomicBool>,
    emit_term: &(dyn Fn(&str) + Send + Sync),
    step: &str,
) -> Result<()> {
    use crate::openai_types::{
        ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    };
    const STREAK_LIMIT: usize = 4;
    if jobs.is_empty() {
        return Ok(());
    }
    let pass_reason = format!("{} pass", step.replace('_', " "));
    let verdict = model.lang_route("commerce", step, doc_lang, probe_text).await;
    emit_term(&format!("  {}", verdict.line));
    let mut lang: Option<crate::model::lang_llm::LangSession> = match verdict.code.as_deref() {
        Some(code) => model.enter_lang_generation(code, &pass_reason).await,
        None => None,
    };
    if lang.is_none() {
        model
            .enter_generation_phase(
                crate::model::ModelSize::Qwen3,
                None,
                Some(cancel.clone()),
                false,
                Some("inference".to_string()),
                &pass_reason,
            )
            .await?;
    }
    let sds_gate = |is_lang: bool| -> Option<(f32, u64)> {
        crate::utils::score_dynamics::adaptive_recent(crate::model::lang_llm::synthesis_drop_axis(is_lang))
            .filter(|(recent, _, _)| *recent >= skip_drop)
            .map(|(recent, _, n)| (recent, n))
    };
    let total_items = items.len();
    emit_term(&format!(
        "  🧠 [SYNTHESIS PASS] 요약 {}건을 {} 로 만듭니다 | 값이 모두 확정된 아이템의 값표만 넘기고, 추출 단계의 디코딩 제약(확정된 다른 필드 값의 금지 목록 · 다른 필드 의미 편견)을 걸지 않습니다. 결과는 같은 원문 대조 게이트를 거치고, 연속 {}건이 게이트를 통과하지 못하면 다음 엔진으로 넘기거나 멈춥니다.",
        jobs.len(),
        lang.as_ref().map(|s| s.label()).unwrap_or_else(|| "Qwen3".to_string()),
        STREAK_LIMIT
    ));
    let announce_gate = |g: Option<(f32, u64)>| {
        if let Some((recent, n)) = g {
            emit_term(&format!(
                "    🔬 [SYNTHESIS PASS / SDS] 이 엔진의 최근 요약 문장 {:.0}% 가 원문 대조에서 폐기되었습니다 (관측 {}건, 기준 {:.0}%). 이번 목록은 첫 요약 하나만 관측용으로 만들고 나머지는 비워 둡니다.",
                recent * 100.0,
                n,
                skip_drop * 100.0
            ));
        }
    };
    let mut gate = sds_gate(lang.is_some());
    announce_gate(gate);
    let mut observed = false;
    let mut streak = 0usize;
    let lang_label = lang.as_ref().map(|s| s.label());
    let mut stats: [(usize, usize, u128); 2] = [(0, 0, 0); 2];
    for (done, (item_idx, field_name, field_desc)) in jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Task cancelled"));
        }
        let pos = match item_pos.get(item_idx) {
            Some(p) => *p,
            None => continue,
        };
        if gate.is_some() && observed {
            crate::utils::score_dynamics::record_baseline("commerce.synthesis_skip", 1.0);
            continue;
        }
        let item = match items.get(pos) {
            Some(v) => v.clone(),
            None => continue,
        };
        let sheet = match crate::utils::ai_utils::synthesis_value_sheet(&item, page_type, doc_lang) {
            Some(s) => s,
            None => {
                emit_term(&format!(
                    "    ⏭️ [SYNTHESIS PASS] Item {}/{} · '{}' | 확정된 값이 2개 미만이라 요약 재료가 없습니다.",
                    pos + 1,
                    total_items,
                    field_name
                ));
                continue;
            }
        };
        crate::utils::score_dynamics::record_baseline("commerce.synthesis_sheet_fields", sheet.lines().count() as f32);
        emit_term(&format!(
            "    🧠 [SYNTHESIS PASS / VALUE SHEET] Item {}/{} · '{}' | 확정된 값 {}개를 요약 재료로 넘깁니다.",
            pos + 1,
            total_items,
            field_name,
            sheet.lines().count()
        ));
        let used_lang = lang.is_some();
        let question = if used_lang {
            parsing::extract_synthesis_field_prompt_native(page_type, field_name, field_desc, doc_lang, &sheet)
        } else {
            parsing::extract_synthesis_field_prompt(page_type, field_name, field_desc, doc_lang, &sheet)
        };
        let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: format!(
                "[JSON CONTEXT]\n{}\n\n[SYNTHESIS FIELD NOTICE]\nThis field is NOT a value to copy. Read the WHOLE [JSON CONTEXT] above and write ONE short sentence that summarizes it. Never return a single cell value such as a bare number, a status word, a person name, or a branch name. If [JSON CONTEXT] is empty, return null.",
                sheet
            ),
            name: None,
        });
        let user_msg = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(question),
            name: None,
        });
        let keys: Vec<String> = field_name.split(',').map(|s| s.trim().to_string()).collect();
        let started = std::time::Instant::now();
        let res = if let Some(sess) = lang.as_ref() {
            model
                .generate_lang_json(sess.code(), vec![system_message, user_msg], 512, 0.0, None, &keys, Some(cancel.clone()))
                .await
                .map(|(text, _)| text)
        } else {
            let q3_gen = model.qwen3_generator.clone();
            let cancel_clone = cancel.clone();
            let params = ChatCompletionParameters {
                messages: vec![system_message, user_msg],
                model: "qwen3".to_string(),
                max_tokens: Some(512),
                temperature: Some(0.0),
                top_p: Some(0.95),
                ..Default::default()
            };
            let r = tokio::task::spawn_blocking(move || {
                let mut guard = q3_gen.blocking_lock();
                match guard.as_mut() {
                    Some(gen) => gen
                        .generate(params, Some(cancel_clone), None, None)
                        .map_err(|e| anyhow::anyhow!("Qwen 3 synthesis failed: {}", e)),
                    None => Err(anyhow::anyhow!("Qwen 3 Generator not available")),
                }
            })
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));
            let q3_clear = model.qwen3_generator.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if let Some(gen) = q3_clear.blocking_lock().as_mut() {
                    gen.clear_kv_cache();
                }
            })
            .await;
            if !model.is_cpu_mode {
                let dev = model.device_config.device.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() {
                        let _ = dev.synchronize();
                    }
                })
                .await;
            }
            r
        };
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Task cancelled"));
        }
        let call_ms = started.elapsed().as_millis();
        crate::utils::score_dynamics::record_baseline(
            if used_lang { "commerce.synthesis_call_ms.lang4b" } else { "commerce.synthesis_call_ms.qwen3" },
            call_ms as f32,
        );
        crate::utils::score_dynamics::record_baseline(
            if used_lang { "commerce.synthesis_ram_free_mb.lang4b" } else { "commerce.synthesis_ram_free_mb.qwen3" },
            (crate::utils::resources::free_ram_bytes() / 1_000_000) as f32,
        );
        observed = true;
        let mut lang_failed = false;
        let found: Option<(String, String)> = match res {
            Ok(raw) => {
                let mut parsed = parsing::parse_json_from_llm(&raw);
                let parsed_val = if let Some(inner) = parsed.get_mut(page_type) { inner.take() } else { parsed };
                let sheet_values: Vec<String> = sheet
                    .lines()
                    .filter_map(|l| l.split_once(": ").map(|(_, v)| v.trim().to_lowercase()))
                    .collect();
                let not_summary = |s: &str| -> bool {
                    let t = s.trim().trim_matches(|c: char| c == '.' || c == '"' || c == '\'').trim().to_lowercase();
                    !t.chars().any(|c| c.is_alphanumeric())
                        || ["null", "string", "n/a", "none", "undefined"].contains(&t.as_str())
                        || sheet_values.iter().any(|v| *v == t)
                };
                let hit = keys.iter().find_map(|k| {
                    parsed_val
                        .get(k.as_str())
                        .and_then(|v| v.as_str())
                        .map(|s| strip_markup_prefix(s.trim()))
                        .filter(|s| !not_summary(s))
                        .map(|s| (k.clone(), s))
                });
                if hit.is_none() {
                    emit_term(&format!(
                        "    ⏭️ [SYNTHESIS PASS] Item {}/{} · '{}' | 응답에 요약 문장이 없습니다 (빈 값 · 자리표시 · 값표의 값 하나를 그대로 돌려준 응답). 요약 필드의 답은 값 둘 이상을 잇는 문장이어야 합니다.",
                        pos + 1,
                        total_items,
                        field_name
                    ));
                }
                hit
            }
            Err(e) => {
                emit_term(&format!(
                    "    ⚠️ [SYNTHESIS PASS] Item {}/{} · '{}' | 생성에 실패했습니다: {}",
                    pos + 1,
                    total_items,
                    field_name,
                    e
                ));
                lang_failed = used_lang;
                None
            }
        };
        let pass = match found {
            Some((key, text)) => {
                let g = crate::utils::ai_utils::synthesis_fact_gate(&text, &sheet, doc_lang);
                if g.total > 0 {
                    crate::utils::score_dynamics::record_baseline(
                        crate::model::lang_llm::synthesis_drop_axis(used_lang),
                        g.dropped.len() as f32 / g.total as f32,
                    );
                }
                if !g.dropped.is_empty() {
                    emit_term(&format!(
                        "    🧪 [SYNTHESIS GROUNDING] Item {}/{} · '{}' | 문장 {}개 중 {}개를 뺍니다: {:?} | 남은 문장: \"{}\" — 요약 문장의 연도·날짜·금액·통화는 아이템 원문에 그대로 있어야 합니다. 원문에 없는 사실이 text·masked_text 로 들어가면 FTS 와 청크 코사인이 그 사실로 이 문서를 회수합니다.",
                        pos + 1,
                        total_items,
                        key,
                        g.total,
                        g.dropped.len(),
                        g.dropped.iter().map(|(s, why)| format!("\"{}\" ← {}", s, why.join("·"))).collect::<Vec<_>>(),
                        g.kept
                    ));
                }
                let twin = if g.kept.is_empty() {
                    None
                } else {
                    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
                    let mine = norm(&g.kept);
                    jobs.iter()
                        .filter(|(ji, jf, _)| *ji == *item_idx && jf != field_name)
                        .flat_map(|(_, jf, _)| jf.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
                        .find(|k| {
                            items
                                .get(pos)
                                .and_then(|v| v.get(k.as_str()))
                                .and_then(|v| v.as_str())
                                .map_or(false, |s| norm(s) == mine)
                        })
                };
                if let Some(other) = twin.as_ref() {
                    crate::utils::score_dynamics::record_baseline("commerce.synthesis_duplicate", 1.0);
                    emit_term(&format!(
                        "    ♻️ [SYNTHESIS DUPLICATE] Item {}/{} · '{}' | 같은 아이템의 요약 필드 '{}' 와 문장이 같아 넣지 않습니다. 같은 문장이 두 필드로 색인되면 한 질의에 청크 두 개로 걸려 이 문서가 근거 없이 앞섭니다.",
                        pos + 1,
                        total_items,
                        key,
                        other
                    ));
                } else if !g.kept.is_empty() {
                    if let Some(obj) = items.get_mut(pos).and_then(|v| v.as_object_mut()) {
                        obj.insert(key.clone(), json!(g.kept.clone()));
                    }
                    emit_term(&format!("    ✅ [SYNTHESIS PASS] Item {}/{} · \"{}\": \"{}\"", pos + 1, total_items, key, g.kept));
                }
                g.dropped.is_empty() && !g.kept.is_empty()
            }
            None => false,
        };
        crate::model::lang_llm::record_engine_outcome("commerce", step, used_lang, pass);
        let slot = &mut stats[usize::from(used_lang)];
        slot.0 += 1;
        slot.1 += usize::from(pass);
        slot.2 += call_ms;
        streak = if pass { 0 } else { streak + 1 };
        if lang_failed || streak >= STREAK_LIMIT {
            if used_lang {
                emit_term(&format!(
                    "    🔁 [SYNTHESIS ENGINE FALLBACK] {} 의 요약이 연속 {}건 원문 대조를 통과하지 못했거나 생성에 실패해, 남은 {}건은 Qwen3 로 만듭니다.",
                    lang.as_ref().map(|s| s.label()).unwrap_or_default(),
                    streak,
                    jobs.len() - done - 1
                ));
                crate::utils::score_dynamics::record_baseline("commerce.synthesis_engine_fallback", 1.0);
                lang = None;
                model
                    .enter_generation_phase(
                        crate::model::ModelSize::Qwen3,
                        None,
                        Some(cancel.clone()),
                        false,
                        Some("inference".to_string()),
                        &format!("{} (fallback)", pass_reason),
                    )
                    .await?;
                streak = 0;
                gate = sds_gate(false);
                announce_gate(gate);
                observed = false;
            } else {
                emit_term(&format!(
                    "    ⛔ [SYNTHESIS PASS STOP] Qwen3 요약이 연속 {}건 원문 대조를 통과하지 못해 남은 {}건은 비워 둡니다. 확인되지 않은 요약을 색인하지 않는 쪽이 전처리 결과를 지킵니다.",
                    streak,
                    jobs.len() - done - 1
                ));
                crate::utils::score_dynamics::record_baseline("commerce.synthesis_pass_stop", 1.0);
                for _ in (done + 1)..jobs.len() {
                    crate::utils::score_dynamics::record_baseline("commerce.synthesis_skip", 1.0);
                }
                break;
            }
        }
    }
    let summary: Vec<String> = [(1usize, lang_label.unwrap_or_default()), (0usize, "Qwen3".to_string())]
        .iter()
        .filter(|(i, _)| stats[*i].0 > 0)
        .map(|(i, label)| {
            let (calls, passes, ms) = stats[*i];
            format!("{} 호출 {}건 · 통과 {}건 · 호출당 평균 {:.1}초", label, calls, passes, ms as f64 / calls as f64 / 1000.0)
        })
        .collect();
    if !summary.is_empty() {
        emit_term(&format!(
            "  📊 [SYNTHESIS PASS / SUMMARY] {} | 원문 대조를 통과한 문장만 아이템에 들어갑니다. 같은 목록에서 엔진별 통과율과 시간을 나란히 봅니다.",
            summary.join(" | ")
        ));
    }
    crate::utils::score_dynamics::flush();
    Ok(())
}

async fn prejudice_pair(model: &LogisModel, en_text: &str, doc_lang: &str) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::new();
    if let Ok(e) = model.get_embedding(en_text.to_string()).await {
        if !e.is_empty() { out.push(e); }
    }
    if let Some(local) = crate::logic::site_chrome_sentence(doc_lang) {
        if let Ok(e) = model.get_embedding(local).await {
            if !e.is_empty() { out.push(e); }
        }
    }
    if out.is_empty() { out.push(vec![0.0f32; 384]); }
    out
}

pub async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    if cancellation_token.load(std::sync::atomic::Ordering::Relaxed) {
        println!("[PROCESS] 🛑 Task {} aborted at entry — cancellation token already set (reset in progress).", task.id);
        return Err(anyhow::anyhow!("Task cancelled at entry"));
    }
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}
", msg)}));
    };
    let zero_addr = "0x0000000000000000000000000000000000000000";
    let from_addr = if task.from.is_empty() { zero_addr.to_string() } else { task.from.clone() };
    let team_id = if task.to.is_empty() || task.to == zero_addr {
        crate::utils::hash::hash_id(&from_addr)
    } else {
        task.to.clone()
    };
    emit_term("
=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));

    if task.r#type == "analytic_extraction" {
        return crate::analytic::process_analytic_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        emit_term(&format!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id));
    }

    let payload = json!({ 
        "task_id": task.id,
        "task_type": task.r#type, 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    
    
    let search_mode = task_data.get("search_mode").and_then(|s| s.as_str()).unwrap_or("commerce").to_string();

    
    
    
    
    if search_mode == "shipping" && (task.r#type == "html_extraction" || task.r#type == "document_extraction") {
        return process_trading_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

    let kv_name = if task.r#type == "image_extraction" {
        Some("image".to_string())
    } else {
        Some("text".to_string())
    };
 
    let task_device_pref = if let Some(v) = task_data.get("device_preference") {
        if v.as_str() == Some("cpu") || v.as_bool() == Some(true) {
            Some("cpu".to_string())
        } else {
            None
        }
    } else {
        None
    };
    let effective_device_pref = task_device_pref.as_deref().or(device_preference.as_deref());
    
    let language = "english"; 
    let mut doc_lang = "en".to_string();

    let model = {
        println!("[Scheduler] 🛡️ Attempting to acquire Model Lock...");
        let mut model_lock = model_mutex.lock().await;
        println!("[Scheduler] ✅ Model Lock acquired.");
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }


        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = effective_device_pref == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                println!("[Scheduler] Device preference mismatch (Current CPU: {}, Wants CPU: {}). Reloading model...", m.is_cpu_mode, wants_cpu);
                m.deep_purge_resources().await;
                *model_lock = None;
            }
        }

        if model_lock.is_none() {
            println!("[Scheduler] Model not initialized. Starting LogisModel::new...");

            log_task_progress(app_handle, &task.id, &json!({ "category": "Loading Model", "summary": "Initializing AI Core..." }));
            
            match LogisModel::new(app_handle.clone(), effective_device_pref).await {
                Ok(m) => {
                    println!("[Scheduler] LogisModel::new successful.");
                    *model_lock = Some(m);
                },
                Err(e) => {
                    println!("[Scheduler] ❌ LogisModel::new failed: {}", e);
                    return Err(anyhow::anyhow!("Model Load Failed: {}", e));
                }
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    if task.r#type != "image_extraction" && task.r#type != "analytic_extraction" {
        model.check_embedding_downloaded().await?;
    }

    if task.r#type == "image_extraction" {
        
        
        if let Err(e) = model.check_siglip2_downloaded().await {
            println!("[Scheduler] ⚠️ SigLIP2 model missing: {}", e);
            let _ = app_handle.emit("app_error_alert", json!({
                "message": "SigLIP2 비전 모델이 필요합니다. Settings 탭이 열리고 자동으로 다운로드합니다.",
                "model": "SigLIP2",
                "task_id": task.id.clone(),
                "action": "open_settings"
            }));
            
            let error_status = crate::logic::parse_status("error");
            let _ = store_mutex.lock().await
                .as_ref()
                .map(|db| db.update_task_status(&task.id, error_status));
            return Err(anyhow::anyhow!("{}", e));
        }

        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        if !image_path.is_empty() {
            println!("[Scheduler] Starting Image Extraction for {}", task.id);
            if let Some(line) = crate::model::lang_llm::prefetch("korean", app_handle, &task.id) {
                emit_term(&format!("[Scheduler] {}", line));
            }
            model.extract_from_image(
                task.id.clone(),
                image_path,
                "korean".to_string(),
                search_mode,
                app_handle,
                Some(cancellation_token.clone()),
                store_mutex,
            ).await?;
            return Ok(());
        }
    }

    let (url, origin_candidate) = crate::utils::url_utils::resolve_absolute_url(&task_data).await;
    // 🌟 [SDS SCOPE] 이 태스크의 관측 스코프를 세웁니다.
    //
    //  ── 왜 여기인가 ──
    //   url 과 origin 이 확정된 직후이자, 어떤 판정도 시작되기 전입니다.
    //   이보다 앞이면 스코프의 2차 키(cc)가 비고, 이보다 뒤면
    //   STEP A 의 초기 관측이 스코프 없이 버려집니다.
    //
    //  ── 1차 키는 나중에 정밀화됩니다 ──
    //   지금은 page_type 을 모르므로 mode 만 넣고,
    //   STEP A 가 분류를 마치거나 MODE PROBE 가 서식을 확정하면
    //   refine_primary 가 1차 키를 교체합니다.
    //
    //  ── 이미지 경로는 여기 도달하지 않습니다 ──
    //   image_extraction 은 위쪽에서 return 하므로
    //   비전 스코프는 model/vision.rs 진입부에서 따로 세워야 합니다.
    {
        let track = if search_mode == "shipping" {
            crate::utils::score_dynamics::Track::Trading
        } else if search_mode == "analytic" {
            crate::utils::score_dynamics::Track::Analytic
        } else {
            crate::utils::score_dynamics::Track::Commerce
        };
        crate::utils::score_dynamics::enter_scope(&team_id, track, &search_mode, &task.cc);
    }
    let active_task_json = json!({
        "id": task.id.clone(),
        "type": task.r#type.clone(),
        "link": url.clone(),
        "origin": origin_candidate.clone(),
        "ref": task.r#ref.clone(),
        "status": 1, 
        "created_at": task.created_at,
        "updated_at": chrono::Utc::now().timestamp_millis()
    });
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = Some(active_task_json.clone());
    }

    if url.is_empty() { 
        return Err(anyhow::anyhow!("Task missing target URL or unsupported type for background extraction.")); 
    }

    let raw_html_content = if task.r#type == "document_extraction" {
        let file_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("");
        let ext = task_data.get("document_ext").and_then(|s| s.as_str()).unwrap_or("");
        
        let payload = json!({ 
            "task_id": task.id, 
            "category": "Document Parsing", 
            "summary": format!("Parsing {} file format...", ext.to_uppercase()), 
            "spinner": "📄" 
        });

        let _ = app_handle.emit("extraction-progress", &payload);

        let extracted_text = crate::parsers::extract_document_text(file_path).unwrap_or_else(|e| format!("Document Parsing Error: {}", e));

        let fake_html = extracted_text.lines()
            .map(|line| {
                let safe_line = line.replace("<", "&lt;").replace(">", "&gt;");
                format!("<div>{}</div>", safe_line)
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<html><body>{}</body></html>", fake_html)
    } else if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        content
    } else if !url.is_empty() {
        let response = reqwest::get(&url).await?;
        let bytes = response.bytes().await?;

        let (decoded_utf8, _, malformed_utf8) = encoding_rs::UTF_8.decode(&bytes);
        let utf8_str = decoded_utf8.as_ref();

        let needs_euc = utf8_str.to_lowercase().contains("charset=euc-kr") || 
                        utf8_str.to_lowercase().contains("charset=\"euc-kr\"") ||
                        utf8_str.to_lowercase().contains("charset=cp949") ||
                        utf8_str.to_lowercase().contains("charset=ks_c_5601");

        if needs_euc && malformed_utf8 {
            let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);

            decoded_euc.into_owned()
        } else {
            utf8_str.to_string()
        }
    } else {
        return Ok(());
    };

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);
    
    let mut raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let mut light_pug = model.truncate_pug_context(&raw_pug, false, 2000, None).await;

    let tokenizer_path = model.qwen3_tokenizer_dir().to_string();
 
    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", light_pug);

    let mut token_count = raw_system_prefix.len() / 4;

    if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(&tokenizer_path) {

        token_count = tokenizer.text_encode_vec(raw_system_prefix.clone(), false)
            .map(|v| v.len())
            .unwrap_or(token_count);
    }

    if token_count <= 6000 {
        println!("[Scheduler] Document is short enough ({} tokens). Upgrading to FullContent Mode...", token_count);
        raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::FullContent, Some(&url));
        light_pug = model.truncate_pug_context(&raw_pug, true, 2000, None).await;
    }    
    
    // 🌟 [DOC LANG SOURCE FIX] 언어 판정에 PUG 구조 문자가 섞이면 언어가 흔들립니다.
    //
    //  ── 실측 사고 ──
    //   영문 YAML 문서인데 커머스 트랙은 'fr', 트레이딩 트랙은 'en' 으로 판정했습니다.
    //   트레이딩은 '|' 뒤 본문 텍스트만 넣고, 커머스는 light_pug 전체(태그·들여쓰기 포함)를
    //   넣기 때문입니다. doc_lang 은 label_phrase_bank / prejudice_phrase_bank 의
    //   locale 키라, 틀리면 커머스 필드 뱅크가 통째로 열화됩니다.
    //
    //  ── 수정 ──
    //   판정 입력을 '본문 텍스트만' 으로 통일합니다. 어휘 사전이 아니라 입력 정제입니다.
    let lang_probe_text: String = {
        let mut out = String::new();
        for line in light_pug.lines() {
            let t = match line.find('|') { Some(p) => line[p + 1..].trim(), None => continue };
            if t.chars().count() < 2 { continue; }
            out.push_str(t);
            out.push('\n');
            if out.len() > 8000 { break; }
        }
        if out.trim().is_empty() { light_pug.clone() } else { out }
    };
    doc_lang = crate::utils::lang_utils::detect_document_language(&lang_probe_text);
    println!(
        "[Scheduler] 🌐 [DOC LANG] Early detection (cache-independent): '{}'",
        doc_lang
    );
    if let Some(line) = crate::model::lang_llm::prefetch(&doc_lang, app_handle, &task.id) {
        emit_term(&format!("[Scheduler] {}", line));
    }
    // =====================================================================
    // 🌟 [MODE REROUTE] mode 가 commerce 여도 문서가 무역 서식이면 shipping 으로 넘깁니다.
    // ---------------------------------------------------------------------
    //  ── 왜 여기인가 ──
    //   위 search_mode 분기는 UI 플래그 하나만 봅니다. 문서 내용은 여기서 처음 확보됩니다.
    //   커머스 분류(STEP A)가 시작되면 order/goods/tracking/review/coupon/event 6개 중
    //   하나로 강제 분류되므로, 그 전에 갈라야 합니다.
    //
    //  ── mode 값의 실제 집합 ──
    //   lib.rs 의 ai_search_complex 기준으로 mode 는 commerce / shipping / analytic 3종입니다.
    //   "trading" 은 파이프라인 이름일 뿐 mode 값이 아니므로, 리라우트 대상 mode 는 shipping 입니다.
    //   process_trading_task 가 저장 시 data.mode 를 'shipping' 으로 확정하므로
    //   여기서는 파이프라인만 바꾸면 됩니다.
    //
    //  ── 비용 ──
    //   자기선언 라벨이 없는 페이지(커머스 목록/상세)는 라벨 임베딩 몇 개만 쓰고 즉시 빠집니다.
    // =====================================================================
    // 🌟 [ANALYTIC GUARD] analytic 은 코사인 mode 판정 대상이 아닙니다.
    //
    //  ── 근거 ──
    //   analytic 은 이 함수 최상단에서 task.r#type == "analytic_extraction" 로
    //   결정론 분기되며, 이후 type(click / hover / change / touch / report)은
    //   Worker 가 내려준 D1 row 의 type 컬럼에서 옵니다. 코사인이 개입하지 않습니다.
    //
    //  ── 그래도 가드가 필요한 이유 ──
    //   main.ts 의 WebRTC mobile_upload 경로는 document_extraction 을 만들면서
    //   search_mode: currentSearchMode 를 그대로 실어 보냅니다.
    //   즉 search_mode='analytic' + type='document_extraction' 조합이 성립하고,
    //   그 경우 무역 프로브를 타서 mode='shipping' 으로 저장될 수 있습니다.
    //   index_item_chunks 의 DOMAIN CONSISTENCY GATE 가 청크는 막아 주지만
    //   아이템 자체는 이미 잘못된 mode 로 기록된 뒤입니다.
    if search_mode != "shipping"
        && search_mode != "analytic"
        && (task.r#type == "html_extraction" || task.r#type == "document_extraction")
    {
        model.check_embedding_downloaded().await?;
        model.ensure_embedding().await?;
        let probe = crate::scheduler::trading::probe_trade_document(
            &model, &light_pug, &doc_lang, &emit_term,
        ).await;
        if let Some(v) = probe {
            // 🌟 [AUDIT] 파이프라인 자체가 바뀌는 되돌릴 수 없는 분기이므로
            //    '무엇을 근거로' 넘어갔는지 한 줄에 전부 남깁니다.
            //    Score 는 Gumbel 보정 후 값이며, 전문/껍데기는 센터링 이전 원시 코사인입니다.
            emit_term(&format!(
                "🚢 [MODE REROUTE] mode='{}' 로 들어왔지만 표제 축이 무역 서식 '{}'({})를 지목했습니다. \
                 보정 Score {:+.4} > 커머스 최상위 '{}'({:+.4}) | 근거 표제 \"{}\" (서식 전문 {:.4} > 사이트 껍데기 {:.4}) | 구조 증거 {:?}. \
                 트레이딩 파이프라인(mode='shipping')으로 전환합니다.",
                search_mode, v.code, v.title, v.score, v.rival, v.rival_score,
                v.evidence_value, v.title_cos, v.chrome_cos, v.markers
            ));
            crate::utils::score_dynamics::record_baseline("mode_probe.reroute", 1.0);
            crate::utils::score_dynamics::flush();
            crate::utils::score_dynamics::leave_scope();
            let reroute_pref = task_device_pref.clone().or_else(|| device_preference.clone());
            return process_trading_task(
                task, store_mutex, model_mutex, cancellation_token, app_handle, reroute_pref
            ).await;
        }
        crate::utils::score_dynamics::record_baseline("mode_probe.reroute", 0.0);
    }
    let base_model_size = if token_count > 60000 {
        crate::model::ModelSize::Qwen
    } else {
        crate::model::ModelSize::Qwen3
    };

    println!("[DEBUG-PUG] Generated PUG. Length: {}. Token Count: {}. Selected Model: {:?}. Snippet: {}...", 
        light_pug.len(), 
        token_count,
        base_model_size,
        light_pug.chars().take(100).collect::<String>().replace("\n", " ")
    );

    use crate::openai_types::{
        ChatCompletionRequestSystemMessage,
        ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent
    };

    let mut page_type = String::new();
    let mut selector_info: serde_json::Value = json!({});
    
    let mut is_detail = task_data.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut skip_ai_analysis = false; 

    let (raw_path, url_obj) = {
        let mut shared_origin = None;
        if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            if let Some(json_val) = mem.as_ref() {
                if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                    if !o.is_empty() && !o.contains("localhost") {
                        let formatted = if o.starts_with("http") { o.to_string() } else { format!("http://{}", o) };
                        if let Ok(u) = url::Url::parse(&formatted) { 
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost"))); 
                        }
                    }
                }
            }
        }
        
        let origin_str = task_data.get("origin")
            .or_else(|| task_data.get("domain"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.contains("localhost"))
            .or(shared_origin)
            .unwrap_or_else(|| if let Ok(task_url) = url::Url::parse(&url) { format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost")) } else { "http://localhost".to_string() });

        let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
        let url_obj = base_url.join(&url).unwrap_or(base_url);
        (url_obj.path().to_string(), url_obj)
    };

    let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path));

    {
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let link_val = (url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()).to_lowercase();
            let path_only = url_obj.path().to_lowercase(); 
            
            let mut potential_caches = Vec::new();

            if let Ok(Some(page_doc)) = db.get_item_by_id("pages", &page_id).await {
                potential_caches.push(page_doc);
            } else if let Ok(Some(page_doc)) = db.get_item_by_id("items", &page_id).await {
                potential_caches.push(page_doc);
            }

            let tables_to_check = ["pages", "items"];
            for tbl in tables_to_check {
                if let Ok(docs) = db.get_all_items(tbl, 1000, 0, None).await {
                    for doc in docs {
                        let json_lower = doc.json_data.to_lowercase();
                        if json_lower.contains(&link_val) || json_lower.contains(&path_only) {
                            if !potential_caches.iter().any(|c| c.id == doc.id) {
                                potential_caches.push(doc);
                            }
                        }
                    }
                }
            }

            let mut final_cache = None;

            for page_doc in potential_caches {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&page_doc.json_data) {
                    let cached_detail = val.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                    let node_sel = val.get("node").or_else(|| val.get("parent")).and_then(|v| v.as_str()).unwrap_or("");
                    let item_sel = val.get("item").or_else(|| val.get("itemSelector")).and_then(|v| v.as_str()).unwrap_or("");

                    let target_sel_str = if !node_sel.is_empty() && !item_sel.is_empty() && !item_sel.contains(",") {
                        if item_sel.starts_with(node_sel) { item_sel.to_string() } else { format!("{} {}", node_sel, item_sel) }
                    } else if !item_sel.is_empty() { item_sel.to_string() } else { node_sel.to_string() };

                    let target_sel_clean = target_sel_str.replace(">", " ");

                    if !cached_detail {
                        let mut is_dom_matched = false;
                        if !target_sel_clean.is_empty() {
                            let document = scraper::Html::parse_document(&clean_html_content);
                            is_dom_matched = scraper::Selector::parse(&target_sel_clean)
                                .map(|sel| document.select(&sel).next().is_some())
                                .unwrap_or(false);
                        }

                        if is_dom_matched {

                            final_cache = Some((page_doc, val, false, target_sel_clean));
                            break;
                        }

                    } else {
                        if final_cache.is_none() {
                            final_cache = Some((page_doc, val, true, target_sel_clean));
                        }
                    }
                }
            }


            if let Some((_page_doc, val, cached_detail, target_sel_str)) = final_cache {
                emit_term(&format!("[Scheduler] ⚡ CACHE HIT! Skipping AI Pre-processing for: {}", raw_path));
                page_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();

                is_detail = cached_detail; 

                selector_info = val.clone();
                selector_info.as_object_mut().unwrap().insert("final_target_selector".to_string(), json!(target_sel_str));
                skip_ai_analysis = true; 
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Loaded valid config from cache.", "spinner": "⚡" }));
            } else {
                emit_term("[Scheduler] Cache miss or elements not found in DOM. Falling back to AI Analysis.");
            }
        }
    }

    let base_session_id = format!("{}_base", task.id);
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    if !skip_ai_analysis {

        if base_model_size == crate::model::ModelSize::Qwen {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
            if !base_kv_path.exists() {
                println!("[Scheduler] Baking Base PUG Context to SSD...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
                
                // 🌟 [CROSSOVER] 프리필은 KV 를 크게 잡으므로 여유 판정이 특히 중요합니다.
                //    임베딩이 남아 있으면 이 시점에 반환시켜 프리필 KV 공간을 확보합니다.
                model
                    .enter_generation_phase(
                        crate::model::ModelSize::Qwen,
                        None,
                        Some(cancellation_token.clone()),
                        true,
                        kv_name.clone(),
                        "base pug prefill",
                    )
                    .await?;
                
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if let Some(gen) = model.generator.lock().await.as_mut() {
                    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", system_content);

                    gen.prefill_only(raw_system_prefix, Some(cancellation_token.clone()), Some(base_session_id.clone()), None, kv_name.clone()).await?;
                }
            }

            model.deep_purge_resources().await;
        }

        let pug_lines: Vec<String> = light_pug.lines().map(|s| s.to_string()).collect();
        let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
        let mut wiped_indices = vec![false; pug_lines.len()];

        let early_doc_title = {
            let doc = scraper::Html::parse_document(&clean_html_content);
            let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
            } else {
                String::new()
            };
            let mut heading_texts = Vec::new();
            if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                for el in doc.select(&sel_h1) {
                    let txt = el.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !txt.is_empty() { heading_texts.push(txt); }
                }
            }
            if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                for el in doc.select(&sel_h2) {
                    let txt = el.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !txt.is_empty() { heading_texts.push(txt); }
                }
            }

            if !heading_texts.is_empty() {
                if t_val.is_empty() || t_val.len() < 5 {
                    t_val = heading_texts.join(" | ");
                } else {
                    t_val = format!("{} | {}", t_val, heading_texts.join(" | "));
                }
            }
            t_val
        };
        let early_title_emb = if !early_doc_title.is_empty() {
            model.get_embedding(early_doc_title.clone()).await.unwrap_or(vec![0.0; 384])
        } else {
            vec![0.0; 384]
        };
        

        let mut filtered_light_pug = light_pug.clone();
        let mut line_col_positions: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
        let mut is_table_structure = false;
        {
            let mut current_row: usize = 0;
            let mut current_col: usize = 0;
            let mut in_row = false;
            for (line_idx, line) in pug_lines.iter().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                let tag_part = trimmed.split('|').next().unwrap_or("").trim();
                let tag_name = tag_part.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
                if tag_name == "tr" {
                    is_table_structure = true;
                    if in_row { current_row += 1; }
                    current_col = 0;
                    in_row = true;
                } else if (tag_name == "td" || tag_name == "th") && in_row {

                    let mut colspan_val = 1;
                    if let Ok(re_cs) = regex::Regex::new(r#"colspan[=\\"]*(\d+)"#) {
                        if let Some(cap) = re_cs.captures(tag_part) {
                            colspan_val = cap[1].parse::<usize>().unwrap_or(1);
                        }
                    }

                    if let Some(pipe_idx) = trimmed.find('|') {
                        let txt = trimmed[pipe_idx + 1..].trim();
                        if !txt.is_empty() {
                            line_col_positions.insert(line_idx, (current_row, current_col));
                        }
                    }
                    current_col += colspan_val;
                } else if tag_name != "td" && tag_name != "th" && tag_name != "tr" && tag_name != "thead" && tag_name != "tbody" && tag_name != "table" {

                    if in_row && !["colgroup", "col", "caption"].contains(&tag_name.as_str()) {

                    }
                }
            }
        }

        let mut global_text_stats: std::collections::HashMap<String, (usize, Vec<(usize, usize, Option<(usize, usize)>)>)> = std::collections::HashMap::new();
        for (line_idx, line) in pug_lines.iter().enumerate() {
            if let Some(idx) = line.find('|') {
                let indent = line.chars().take_while(|c| c.is_whitespace()).count();
                let text_part = line[idx + 1..].trim();
                if !text_part.is_empty() && text_part.len() > 2 {
                    let col_pos = line_col_positions.get(&line_idx).cloned();
                    let entry = global_text_stats.entry(text_part.to_string()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push((line_idx, indent, col_pos));
                }
            }
        }

        let total_table_rows = if is_table_structure {
            let mut max_row = 0usize;
            for (_, &(r, _)) in &line_col_positions {
                if r > max_row { max_row = r; }
            }
            max_row + 1
        } else { 0 };

        let universal_prejudice = "global navigation, menus, footer, aside, search form, search filter.";
        let universal_prej_embs = prejudice_pair(&model, universal_prejudice, &doc_lang).await;
        
        let mut global_boilerplate_texts = std::collections::HashSet::new();
        let re_numeric = regex::Regex::new(r"^\D*\d+[\d,\.]*\D*$").unwrap();
        let re_has_digit = regex::Regex::new(r"\d").unwrap();
        
        for (text, (count, occurrences)) in global_text_stats {
            if count >= 4 {

                let is_numeric_data = re_numeric.is_match(&text) || re_has_digit.is_match(&text);
                if is_numeric_data { continue; }

                if is_table_structure && total_table_rows >= 3 {
                    let mut col_hits: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                    let mut rows_with_this_text: std::collections::HashSet<usize> = std::collections::HashSet::new();
                    for (_, _, col_pos) in &occurrences {
                        if let Some((row_idx, col_idx)) = *col_pos {
                            *col_hits.entry(col_idx).or_insert(0) += 1;
                            rows_with_this_text.insert(row_idx);
                        }
                    }

                    let row_coverage = rows_with_this_text.len() as f64 / total_table_rows as f64;
                    let max_col_hit = col_hits.values().max().copied().unwrap_or(0);
                    let is_same_col_repeated = max_col_hit >= (total_table_rows as f64 * 0.7).ceil() as usize;
                    
                    if is_same_col_repeated && row_coverage >= 0.7 {

                        let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                            let line = &pug_lines[*line_idx];
                            line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                        });
                        if has_link_or_event {
                            emit_term(&format!("  🛡️ [TABLE-COL LINK PROTECT] 동일 컬럼 반복이지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견)", text, count));
                            continue;
                        }

                        if text.len() < 10 {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [TABLE-COL DROP] 동일 컬럼({}회/{}rows) 반복 UI 탈락: '{}' ({}회 발견)", max_col_hit, total_table_rows, text, count));
                            continue;
                        } else {
                            emit_term(&format!("  🛡️ [TABLE-COL PROTECT] 동일 컬럼 반복이지만 긴 텍스트(데이터 추정): '{}' ({}회 발견)", text, count));
                            continue;
                        }
                    }
                }

                let mut is_contiguous = false;
                let mut is_dispersed = false;
                let mut is_same_depth = true;

                if occurrences.len() >= 2 {
                    let mut gaps = Vec::new();
                    let first_indent = occurrences[0].1;
                    for i in 1..occurrences.len() {
                        gaps.push(occurrences[i].0 - occurrences[i-1].0);
                        if occurrences[i].1 != first_indent {
                            is_same_depth = false;
                        }
                    }
                    let min_gap = *gaps.iter().min().unwrap_or(&0);
                    let max_gap = *gaps.iter().max().unwrap_or(&0);
                    if min_gap <= 3 && max_gap <= 5 {
                        is_contiguous = true;
                    } else if max_gap > 10 {
                        is_dispersed = true;
                    }

                    if is_same_depth && max_gap > 5 {

                        if is_table_structure {
                            let mut unique_cols: std::collections::HashSet<usize> = std::collections::HashSet::new();
                            for (_, _, col_pos) in &occurrences {
                                if let Some((_, col_idx)) = *col_pos {
                                    unique_cols.insert(col_idx);
                                }
                            }
                            if unique_cols.len() <= 1 && count >= 6 {

                                let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                                    let line = &pug_lines[*line_idx];
                                    line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                                });
                                if has_link_or_event {
                                    emit_term(&format!("  🛡️ [TABLE-SAME-COL LINK PROTECT] 단일 컬럼 반복이지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회, 컬럼 {:?})", text, count, unique_cols));
                                    continue;
                                }

                                global_boilerplate_texts.insert(text.clone());
                                emit_term(&format!("  🚫 [TABLE-SAME-COL DROP] 단일 컬럼 반복 탈락: '{}' ({}회, 컬럼 {:?})", text, count, unique_cols));
                                continue;
                            }
                        }                        emit_term(&format!("  🛡️ [GLOBAL PROTECT] 동일 구조(Depth) 내 분산 패턴 데이터 보호: '{}' ({}회 발견)", text, count));
                        continue;
                    } else if !is_same_depth && is_dispersed {
                        if text.len() < 20 {

                            let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                                let line = &pug_lines[*line_idx];
                                line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                            });
                            if has_link_or_event {
                                emit_term(&format!("  🛡️ [GLOBAL LINK PROTECT] 다중 구조 교차지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견)", text, count));
                                continue;
                            }

                            let drop_text_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                            let title_protect_sim = cosine_similarity(&early_title_emb, &drop_text_emb);
                            if title_protect_sim > 0.40 {
                                emit_term(&format!("  🛡️ [TITLE VECTOR PROTECT] 다중 구조 교차지만 타이틀 코사인 유사도({:.4})가 높아 도메인 시그널 보호: '{}' ({}회 발견)", title_protect_sim, text, count));
                                continue;
                            }
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 다중 구조(Depth) 교차 발견 노이즈 탈락: '{}' ({}회 발견)", text, count));
                            continue;
                        }
                    }
                }

                if is_contiguous {
                    if text.len() < 20 {
                        let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                            let line = &pug_lines[*line_idx];
                            line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                        });
                        if has_link_or_event {
                            emit_term(&format!("  🛡️ [GLOBAL LINK PROTECT] 뭉쳐있지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견, 연속됨)", text, count));
                        } else {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 뭉쳐있는 UI 노이즈 탈락: '{}' ({}회 발견, 연속됨)", text, count));
                        }
                    }
                } else if is_dispersed {
                    if text.len() >= 5 {
                        emit_term(&format!("  🛡️ [GLOBAL PROTECT] 분산된 데이터(상품명 추정) 보호: '{}' ({}회 발견, 간격 넓음)", text, count));
                    } else {
                        global_boilerplate_texts.insert(text.clone());
                        emit_term(&format!("  🚫 [GLOBAL DROP] 분산된 짧은 UI(버튼) 탈락: '{}' ({}회 발견)", text, count));
                    }
                } else {
                    if text.len() > 3 {
                        let text_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                        let ui_noise_score = max_pool_sim(&text_emb, &universal_prej_embs);
                        if ui_noise_score > 0.35 {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 판정 전 전역 중복 UI 탈락: '{}' ({}회 발견, NoiseScore: {:.4})", text, count, ui_noise_score));
                        }
                    }
                }
            }
        }

        for (i, line) in pug_lines.iter().enumerate() {
            if let Some(idx) = line.find('|') {
                let text_part = line[idx + 1..].trim();
                if global_boilerplate_texts.contains(text_part) {
                    wiped_indices[i] = true;
                }
            }
        }

        let mut texts_to_embed = Vec::new();
        let mut text_indices = Vec::new();
        
        for (line_idx, line) in pug_lines.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            if wiped_indices[line_idx] { continue; }
            
            let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { "" };
            if !text_part.is_empty() {
                texts_to_embed.push(text_part.to_string());
                text_indices.push(line_idx);
            }
        }

        if !texts_to_embed.is_empty() {
            // 🌟 [CROSSOVER] STEP A 는 순수 임베딩 구간입니다.
            //    페이즈를 선언해 두면 이후 Qwen 프리필까지 생성 모델이 개입하지 않습니다.
            model.enter_embedding_phase("step A line vectorization").await?;
            let vectors = model.get_embedding_batch(texts_to_embed.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; texts_to_embed.len()]);
            for (i, vector) in vectors.into_iter().enumerate() {
                if i >= text_indices.len() { break; }
                let original_idx = text_indices[i];
                line_embeddings[original_idx] = vector;
            }
        }

        let nodes_str = {
            let document_for_boa = scraper::Html::parse_document(&clean_html_content);
            let mut nodes_json = Vec::new();
            let mut node_to_idx = std::collections::HashMap::new();
            for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                node_to_idx.insert(node.id(), idx);
            }
            for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                if let Some(el) = node.value().as_element() {
                    let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                    let text: String = node.children()
                        .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                        .collect::<Vec<_>>().join(" ").trim().to_string();
                    nodes_json.push(serde_json::json!({
                        "index": idx,
                        "parentIndex": parent_idx,
                        "tagName": el.name().to_string(),
                        "id": el.id().unwrap_or("").to_string(),
                        "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                        "text": text,
                        "colspan": el.attr("colspan").unwrap_or("1"),
                        "rowspan": el.attr("rowspan").unwrap_or("1")
                    }));
                } else {
                    nodes_json.push(serde_json::json!(serde_json::Value::Null));
                }
            }
            serde_json::to_string(&nodes_json).unwrap_or_default()
        };

        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting PURE VECTOR DETERMINISTIC RELAY (Step A)");
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Cleaning global noise layouts...", "spinner": "⠋" }));

            let js_template = get_boa_block_extractor_template();

            let mut pre_processed_blocks = std::collections::HashSet::new();
            let mut track_a_candidates = Vec::new();
            let mut seen_candidates = std::collections::HashSet::new();

            for line_idx in 0..pug_lines.len() {
                let text_part = if let Some(idx) = pug_lines[line_idx].find('|') { pug_lines[line_idx][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                
                let line_prej_score = max_pool_sim(&line_embeddings[line_idx], &universal_prej_embs);
                if line_prej_score > 0.55 {
                    if !seen_candidates.contains(text_part) {
                        seen_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                    }
                }
            }

            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len();
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            let track_a_pugs: Vec<(String, String)> = {
                let mut seen_selectors = std::collections::HashSet::new();
                let mut unique_sels = Vec::new();
                for sel in track_a_selectors {
                    if !sel.is_empty() && !seen_selectors.contains(&sel) {
                        seen_selectors.insert(sel.clone());
                        unique_sels.push(sel);
                    }
                }
                
                let html_clone = clean_html_content.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                

                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };

            let mut unique_pugs_to_embed = Vec::new();
            let mut track_a_pugs_clean = Vec::new();
            for (sel, block_pug) in track_a_pugs {
                if block_pug.is_empty() || pre_processed_blocks.contains(&block_pug) { continue; }
                pre_processed_blocks.insert(block_pug.clone());
                unique_pugs_to_embed.push(block_pug.clone());
                track_a_pugs_clean.push((sel, block_pug));
            }

            let mut block_embeddings_map = std::collections::HashMap::new();
            if !unique_pugs_to_embed.is_empty() {
                for chunk in unique_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            block_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (sel, block_pug) in track_a_pugs_clean {
                let block_emb = block_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                let block_prej_score = max_pool_sim(&block_emb, &universal_prej_embs);
                
                if block_prej_score > 0.50 {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [FRONT-CLEAN] Expunged Global Layout Block: '{}' (Lines {}~{})", sel, start_idx + 1, end_idx + 1));
                        for j in start_idx..=end_idx {
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            let mut pre_filtered_pug = String::new();
            for (idx, line) in pug_lines.iter().enumerate() {
                if !wiped_indices[idx] { pre_filtered_pug.push_str(line); }
                pre_filtered_pug.push_str("\n");
            }
            filtered_light_pug = pre_filtered_pug.trim_end().to_string();

            {
                let nav_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, navigation bar, submenu, category menu, management menu, settings menu, configuration menu";
                let nav_prej_embs = prejudice_pair(&model, nav_prejudice_text, &doc_lang).await;

                let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
                let mut category_embs = Vec::new();
                for cat in &categories {
                    let anchor_text = crate::parsing::get_page_type_classification_bias(cat, &doc_lang);
                    let anchor_emb = model.get_embedding(anchor_text).await.unwrap_or(vec![0.0; 384]);
                    category_embs.push(anchor_emb);
                }

                let mut nav_wiped_count = 0usize;
                let mut nav_domain_protected = 0usize;
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if !line_embeddings[i].iter().all(|&v| v == 0.0) {
                        let nav_score = max_pool_sim(&line_embeddings[i], &nav_prej_embs);
                        if nav_score > 0.38 {
                            let title_line_sim = cosine_similarity(&early_title_emb, &line_embeddings[i]);

                            let mut max_domain_sim = 0.0;
                            for emb in &category_embs {
                                let sim = cosine_similarity(emb, &line_embeddings[i]);
                                if sim > max_domain_sim { max_domain_sim = sim; }
                            }

                            if (max_domain_sim > 0.30 && max_domain_sim >= nav_score * 0.85) || (title_line_sim > nav_score && title_line_sim > 0.40) {
                                nav_domain_protected += 1;
                                continue;
                            }
                            wiped_indices[i] = true;
                            nav_wiped_count += 1;
                        }
                    }
                }
                if nav_wiped_count > 0 || nav_domain_protected > 0 {
                    emit_term(&format!("  🚫 [STEP-A NAV PRE-FILTER] 페이지 분류 전 네비게이션/레이아웃 {}개 라인 사전 탈락 완료. (도메인/타이틀 벡터 보호: {}개)", nav_wiped_count, nav_domain_protected));
                    let mut re_filtered = String::new();
                    for (idx, line) in pug_lines.iter().enumerate() {
                        if !wiped_indices[idx] { re_filtered.push_str(line); }
                        re_filtered.push_str("\n");
                    }
                    filtered_light_pug = re_filtered.trim_end().to_string();
                }
            }

            let refined_lang = crate::utils::lang_utils::detect_document_language(&filtered_light_pug);
            if refined_lang != doc_lang {
                emit_term(&format!(
                    "  🌐 [DOC LANG REFINE] 노이즈 제거 후 언어 재확정: '{}' → '{}' (음차 캐시 키가 함께 이동합니다)",
                    doc_lang, refined_lang
                ));
            }
            doc_lang = refined_lang;
            println!("[Scheduler] Deterministic Detected Language: {}", doc_lang);
            
            let dedup_idxs: Vec<usize> = {
                let mut v: Vec<usize> = Vec::new();
                for i in 0..line_embeddings.len() {
                    if wiped_indices[i] { continue; }
                    let t = if let Some(p) = pug_lines[i].find('|') { pug_lines[i][p + 1..].trim() } else { "" };
                    if t.is_empty() { continue; }
                    if line_embeddings[i].iter().all(|&x| x == 0.0) { continue; }
                    v.push(i);
                }
                v
            };
            let dedup_floor: f32 = {
                if dedup_idxs.len() < 8 {
                    0.995
                } else {
                    let total_pairs = dedup_idxs.len() * (dedup_idxs.len().saturating_sub(1)) / 2;
                    let step = (total_pairs / 4000).max(1);
                    let mut sum = 0.0f64;
                    let mut sq = 0.0f64;
                    let mut cnt = 0usize;
                    let mut k = 0usize;
                    'outer_dd: for a in 0..dedup_idxs.len() {
                        for b in (a + 1)..dedup_idxs.len() {
                            k += 1;
                            if k % step != 0 { continue; }
                            let s = cosine_similarity(&line_embeddings[dedup_idxs[a]], &line_embeddings[dedup_idxs[b]]) as f64;
                            sum += s;
                            sq += s * s;
                            cnt += 1;
                            if cnt >= 4000 { break 'outer_dd; }
                        }
                    }
                    if cnt < 8 {
                        0.995
                    } else {
                        let mu = sum / (cnt as f64);
                        let var = (sq / (cnt as f64) - mu * mu).max(0.0);
                        let sd = var.sqrt();
                        ((mu + 3.0 * sd) as f32).clamp(0.90, 0.995)
                    }
                }
            };
            let mut evidence_weight: Vec<f32> = vec![1.0; pug_lines.len()];
            let mut is_cluster_rep: Vec<bool> = vec![false; pug_lines.len()];
            {
                let mut reps: Vec<usize> = Vec::new();
                let mut members: Vec<Vec<usize>> = Vec::new();
                for &i in dedup_idxs.iter() {
                    let mut hit: Option<usize> = None;
                    for (ri, rep) in reps.iter().enumerate() {
                        if cosine_similarity(&line_embeddings[i], &line_embeddings[*rep]) >= dedup_floor {
                            hit = Some(ri);
                            break;
                        }
                    }
                    match hit {
                        Some(ri) => members[ri].push(i),
                        None => {
                            if reps.len() >= 600 {
                                
                                
                                is_cluster_rep[i] = true;
                                continue;
                            }
                            reps.push(i);
                            members.push(vec![i]);
                        }
                    }
                }
                let mut collapsed = 0usize;
                for (ri, rep) in reps.iter().enumerate() {
                    is_cluster_rep[*rep] = true;
                    let n = members[ri].len();
                    if n <= 1 { continue; }
                    let w = 1.0f32 / (n as f32);
                    for m in members[ri].iter() { evidence_weight[*m] = w; }
                    collapsed += n - 1;
                    let sample = if let Some(p) = pug_lines[*rep].find('|') {
                        pug_lines[*rep][p + 1..].trim().chars().take(40).collect::<String>()
                    } else {
                        String::new()
                    };
                    emit_term(&format!(
                        "  ♻️ [EVIDENCE DEDUP] '{}' 유형 라인 {}개를 증거 1개로 접습니다. (라인당 가중 {:.3})",
                        sample, n, w
                    ));
                }
                emit_term(&format!(
                    "  ♻️ [EVIDENCE DEDUP] 임계치 μ+3σ = {:.4} | 판정 대상 {}라인 | 고유 증거 {}개 | 접힌 중복 {}라인",
                    dedup_floor, dedup_idxs.len(), reps.len(), collapsed
                ));
            }
            let title_candidates: Vec<String> = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let norm = |s: String| -> String {
                    s.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_string()
                };
                let mut cands: Vec<String> = Vec::new();
                if let Ok(sel) = scraper::Selector::parse("title") {
                    if let Some(el) = doc.select(&sel).next() {
                        let t = norm(el.text().collect::<Vec<_>>().join(" "));
                        if !t.is_empty() { cands.push(t); }
                    }
                }

                let mut dropped_landmark: Vec<String> = Vec::new();
                for tag in ["h1", "h2", "h3", "legend", "caption"] {
                    if let Ok(sel_h) = scraper::Selector::parse(tag) {
                        for el in doc.select(&sel_h) {
                            let t = norm(el.text().collect::<Vec<_>>().join(" "));
                            if t.is_empty() || t.chars().count() > 60 { continue; }
                            let mut in_landmark = false;
                            let mut cur = el.parent();
                            while let Some(node) = cur {
                                if let Some(e) = node.value().as_element() {
                                    let name = e.name().to_lowercase();
                                    if name == "nav" || name == "aside" || name == "header" || name == "footer" {
                                        in_landmark = true;
                                        break;
                                    }
                                    if name == "body" || name == "html" { break; }
                                }
                                cur = node.parent();
                            }
                            if in_landmark {
                                if !dropped_landmark.contains(&t) { dropped_landmark.push(t); }
                                continue;
                            }
                            if !cands.contains(&t) { cands.push(t); }
                        }
                    }
                }
                if !dropped_landmark.is_empty() {
                    emit_term(&format!(
                        "  🚫 [LANDMARK HEADING GATE] nav/aside/header/footer 내부 헤딩 {}개를 타이틀 후보에서 제외했습니다: {:?}",
                        dropped_landmark.len(),
                        dropped_landmark.iter().take(8).collect::<Vec<_>>()
                    ));
                }
                if cands.len() > 24 { cands.truncate(24); }
                cands
            };

            let mut doc_title = title_candidates.first().cloned().unwrap_or_default();
            let mut title_emb = vec![0.0f32; 384];

            let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
            let mut best_type = "".to_string();
            let mut max_total_score = -1.0;

            let mut category_scores: Vec<(String, f32, f32, f32)> = Vec::new();
            let mut category_phrase_embs: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
            let mut category_title_only_embs: Vec<(String, Vec<f32>)> = Vec::new();
            
            
            let mut category_phrase_texts: Vec<Vec<String>> = Vec::new();
            let mut category_protected: Vec<Vec<bool>> = Vec::new();
            for cat in &categories {
                let anchor_text = crate::parsing::get_page_type_classification_bias(cat, &doc_lang);
                let localized_type = crate::parsing::get_localized_page_type(cat, &doc_lang);
                let mut phrases: Vec<String> = anchor_text
                    .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
                    .flat_map(|seg| seg.split_whitespace())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                phrases.push(cat.to_string());
                phrases.push(localized_type.clone());
                phrases.push(format!("{} {}", cat, localized_type));
                let mut seen_phrase = std::collections::HashSet::new();
                phrases.retain(|p| seen_phrase.insert(p.clone()));
                if phrases.len() > 64 { phrases.truncate(64); }
                
                
                let identity: [String; 3] = [
                    cat.to_string(),
                    localized_type.clone(),
                    format!("{} {}", cat, localized_type),
                ];
                let protected: Vec<bool> = phrases
                    .iter()
                    .map(|p| identity.iter().any(|i| i == p))
                    .collect();
                let phrase_embs = model
                    .get_embedding_batch(phrases.clone())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()]);
                category_phrase_embs.push((cat.to_string(), phrase_embs));
                category_phrase_texts.push(phrases);
                category_protected.push(protected);
                let title_only_bias = format!("{} {}", cat, localized_type);
                let title_only_emb = model.get_embedding(title_only_bias).await.unwrap_or(vec![0.0; 384]);
                category_title_only_embs.push((cat.to_string(), title_only_emb));
            }
            
            {
                let mut rebuilt: Vec<(String, Vec<Vec<f32>>)> = Vec::with_capacity(categories.len());
                let mut total_dropped = 0usize;
                for ci in 0..categories.len() {
                    let bank = &category_phrase_embs[ci].1;
                    let mut kept: Vec<Vec<f32>> = Vec::new();
                    let mut dropped: Vec<String> = Vec::new();
                    for pi in 0..bank.len() {
                        let pe = &bank[pi];
                        if pe.iter().all(|&v| v == 0.0) { continue; }
                        if category_protected[ci][pi] {
                            kept.push(pe.clone());
                            continue;
                        }
                        let mut own_best = 0.0f32;
                        for pj in 0..bank.len() {
                            if pj == pi { continue; }
                            if bank[pj].iter().all(|&v| v == 0.0) { continue; }
                            let s = cosine_similarity(pe, &bank[pj]);
                            if s > own_best { own_best = s; }
                        }
                        let mut rival_best = 0.0f32;
                        for cj in 0..categories.len() {
                            if cj == ci { continue; }
                            let s = max_pool_sim(pe, &category_phrase_embs[cj].1);
                            if s > rival_best { rival_best = s; }
                        }
                        if rival_best >= own_best {
                            dropped.push(format!(
                                "{}(own {:.3} <= rival {:.3})",
                                category_phrase_texts[ci][pi], own_best, rival_best
                            ));
                            continue;
                        }
                        kept.push(pe.clone());
                    }
                    if kept.is_empty() {
                        
                        emit_term(&format!(
                            "  ⚠️ [AMBIGUITY MASK] '{}' 뱅크의 모든 구가 실격되어 마스크를 적용하지 않습니다.",
                            categories[ci]
                        ));
                        rebuilt.push(category_phrase_embs[ci].clone());
                        continue;
                    }
                    if !dropped.is_empty() {
                        total_dropped += dropped.len();
                        emit_term(&format!(
                            "  🧹 [AMBIGUITY MASK] '{}' 앵커에서 경쟁 카테고리를 더 잘 설명하는 구 {}개 제거 (잔존 {}개): {:?}",
                            categories[ci],
                            dropped.len(),
                            kept.len(),
                            dropped.iter().take(10).collect::<Vec<_>>()
                        ));
                    }
                    rebuilt.push((categories[ci].to_string(), kept));
                }
                emit_term(&format!(
                    "  🧹 [AMBIGUITY MASK] 총 {}개 모호구를 앵커에서 제거했습니다. (max-pool 오염 차단)",
                    total_dropped
                ));
                category_phrase_embs = rebuilt;
            }

            let chrome_prejudice_text = "admin page, administrator page, management page, admin home, admin main menu, main menu, dashboard, control panel, back office, console, site name, shopping mall, welcome, home, index, search, basic search, search form, filter, login, logout, settings, configuration, my page, notice, banner, footer, copyright";
            let chrome_prej_embs = prejudice_pair(&model, chrome_prejudice_text, &doc_lang).await;

            {
                if !title_candidates.is_empty() {
                    let cand_embs = model
                        .get_embedding_batch(title_candidates.clone())
                        .await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; title_candidates.len()]);                   
                    
                    let mut raw_contrast: Vec<f32> = vec![0.0; title_candidates.len()];
                    let mut raw_chrome: Vec<f32> = vec![0.0; title_candidates.len()];
                    let mut raw_echo: Vec<f32> = vec![0.0; title_candidates.len()];
                    let mut raw_domain_max: Vec<f32> = vec![0.0; title_candidates.len()];
                    let mut echo_count: Vec<usize> = vec![0; title_candidates.len()];
                    for idx in 0..title_candidates.len() {
                        let emb = &cand_embs[idx];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let mut sims: Vec<f32> = Vec::with_capacity(categories.len());
                        for ci in 0..categories.len() {
                            sims.push(max_pool_sim(emb, &category_phrase_embs[ci].1));
                        }
                        let mean_s: f32 = sims.iter().sum::<f32>() / (sims.len() as f32);
                        let max_s: f32 = sims.iter().cloned().fold(0.0f32, f32::max);
                        raw_domain_max[idx] = max_s;
                        raw_contrast[idx] = max_s - mean_s;
                        raw_chrome[idx] = max_pool_sim(emb, &chrome_prej_embs);
                        let mut c = 0usize;
                        for (li, le) in line_embeddings.iter().enumerate() {
                            if wiped_indices[li] { continue; }
                            if le.iter().all(|&v| v == 0.0) { continue; }
                            if cosine_similarity(emb, le) >= dedup_floor { c += 1; }
                        }
                        echo_count[idx] = c;
                        raw_echo[idx] = ((c as f32) + 1.0).ln();
                    }
                    let zscore = |v: &Vec<f32>| -> Vec<f32> {
                        let n = v.len() as f32;
                        if n < 2.0 { return vec![0.0f32; v.len()]; }
                        let mu = v.iter().sum::<f32>() / n;
                        let var = v.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>() / n;
                        let sd = var.sqrt();
                        if sd < 1e-6 { return vec![0.0f32; v.len()]; }
                        v.iter().map(|x| (x - mu) / sd).collect()
                    };
                    let z_contrast = zscore(&raw_contrast);
                    let z_chrome = zscore(&raw_chrome);
                    let z_echo = zscore(&raw_echo);
                    let mut best_idx = 0usize;
                    let mut best_score = f32::MIN;
                    for (idx, cand) in title_candidates.iter().enumerate() {
                        if cand_embs[idx].iter().all(|&v| v == 0.0) { continue; }
                        let cand_score = z_contrast[idx] - z_chrome[idx] + z_echo[idx];
                        emit_term(&format!(
                            "  🏷️ [TITLE CANDIDATE] '{}' | DomainMax: {:.4} | Contrast: {:.4}(z{:+.3}) | Chrome: {:.4}(z{:+.3}) | BodyEcho: {}회(z{:+.3}) | Score: {:+.4}",
                            cand,
                            raw_domain_max[idx],
                            raw_contrast[idx], z_contrast[idx],
                            raw_chrome[idx], z_chrome[idx],
                            echo_count[idx], z_echo[idx],
                            cand_score
                        ));
                        if cand_score > best_score {
                            best_score = cand_score;
                            best_idx = idx;
                        }
                    }
                    doc_title = title_candidates[best_idx].clone();
                    title_emb = cand_embs[best_idx].clone();
                    emit_term(&format!(
                        "  👑 [TITLE ANCHOR SELECTED] '{}' (Score: {:+.4} | BodyEcho: {}회)",
                        doc_title, best_score, echo_count[best_idx]
                    ));
                    // 🌟 [SDS 계측] 타이틀 후보 점수 배열의 감쇠 형상을 남깁니다.
                    //
                    //  ── 실측 ──
                    //   후보 점수가 [-3.4836, +1.0986, +3.1986, -0.8135] 이었습니다.
                    //   1위와 2위의 마진은 +2.10 으로 승자독식형이지만
                    //   '관리자 페이지'(-3.4836) 같은 극단 음수가 섞여 꼬리가 매우 두껍습니다.
                    //   이 형상이 사이트마다 어떻게 다른지는 지금 아무 데도 기록되지 않습니다.
                    //
                    //  ── 왜 재계산인가 ──
                    //   위 루프의 cand_score 는 지역 변수라 배열로 남지 않습니다.
                    //   z 배열 세 개가 그대로 살아 있으므로 같은 식으로 복원합니다.
                    //   (z_contrast - z_chrome + z_echo 는 판정에 쓴 식과 동일합니다)
                    //   임베딩이 0 인 후보는 판정에서도 제외되었으므로 여기서도 뺍니다.
                    {
                        let arr: Vec<f32> = (0..title_candidates.len())
                            .filter(|&i| !cand_embs[i].iter().all(|&v| v == 0.0))
                            .map(|i| z_contrast[i] - z_chrome[i] + z_echo[i])
                            .collect();
                        if arr.len() >= 2 {
                            crate::utils::score_dynamics::record_decay("commerce.title_anchor", &arr);
                        }
                        crate::utils::score_dynamics::record_baseline("commerce.title_score", best_score);
                        crate::utils::score_dynamics::record_baseline(
                            "commerce.title_chrome",
                            raw_chrome[best_idx],
                        );
                    }
                }
            }

            let mut category_title_scores: std::collections::HashMap<String, (f32, f32)> = std::collections::HashMap::new();
            let mut category_title_raw: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
            {
                let mut raw_anchor: Vec<f32> = Vec::new();
                let mut raw_only: Vec<f32> = Vec::new();
                for (ci, cat) in categories.iter().enumerate() {
                    let a = if doc_title.is_empty() { 0.0 } else { max_pool_sim(&title_emb, &category_phrase_embs[ci].1) };
                    let o = if doc_title.is_empty() { 0.0 } else { cosine_similarity(&title_emb, &category_title_only_embs[ci].1).max(0.0) };
                    raw_anchor.push(a);
                    raw_only.push(o);
                    category_title_raw.insert(cat.to_string(), a);
                }
                let n = categories.len() as f32;
                let mean_a: f32 = raw_anchor.iter().sum::<f32>() / n;
                let mean_o: f32 = raw_only.iter().sum::<f32>() / n;
                for (ci, cat) in categories.iter().enumerate() {
                    category_title_scores.insert(cat.to_string(), (raw_anchor[ci] - mean_a, raw_only[ci] - mean_o));
                }
            }

            
            
            
            
            let mut category_line_scores: std::collections::HashMap<String, (f32, f32)> = std::collections::HashMap::new();
            for cat in &categories {
                category_line_scores.insert(cat.to_string(), (0.0, 0.0));
            }
            let mut ambiguous_lines = 0usize;
            let mut body_sim_pool: Vec<Vec<f32>> = vec![Vec::new(); categories.len()];
            for (i, emb) in line_embeddings.iter().enumerate() {
                if wiped_indices[i] { continue; }
                let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                if emb.iter().all(|&v| v == 0.0) { continue; }
                let trimmed_line = pug_lines[i].trim();
                let tag_part = trimmed_line.split('|').next().unwrap_or("").trim().to_lowercase();
                let is_table_cell = tag_part.starts_with("td") || tag_part.starts_with("th");
                let ev_w = evidence_weight[i];
                let weight = (if is_table_cell { 1.5f32 } else { 1.0f32 }) * ev_w;
                let sim_threshold = if is_table_cell { 0.30 } else { 0.38 };
                let margin_threshold = if is_table_cell { 0.015 } else { 0.030 };
                let mut sims: Vec<(usize, f32)> = Vec::with_capacity(categories.len());
                for (ci, (_, phrase_embs)) in category_phrase_embs.iter().enumerate() {
                    sims.push((ci, max_pool_sim(emb, phrase_embs)));
                }
                
                
                if is_cluster_rep[i] {
                    for (ci, s) in &sims {
                        body_sim_pool[*ci].push(*s);
                    }
                }
                let mean_sim: f32 = sims.iter().map(|(_, s)| *s).sum::<f32>() / (sims.len() as f32);
                let mut ordered = sims.clone();
                ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let (best_ci, best_sim) = ordered[0];
                let second_sim = ordered.get(1).map(|(_, s)| *s).unwrap_or(0.0);
                let margin = best_sim - second_sim;
                if best_sim < sim_threshold { continue; }
                if margin < margin_threshold {
                    ambiguous_lines += 1;
                    continue;
                }
                let contrast = best_sim - mean_sim;
                if contrast <= 0.0 { continue; }
                let entry = category_line_scores.get_mut(categories[best_ci]).unwrap();
                entry.0 += contrast * weight;
                entry.1 += ev_w;
            }
            if ambiguous_lines > 0 {
                emit_term(&format!("  ⚖️ [AMBIGUITY GATE] 카테고리 간 마진 부족으로 배제된 범용 라인: {}개", ambiguous_lines));
            }
            crate::utils::score_dynamics::record_baseline("commerce.ambiguous_lines", ambiguous_lines as f32);
            let body_consensus: Vec<f32> = {
                let mut raw: Vec<f32> = Vec::with_capacity(categories.len());
                for ci in 0..categories.len() {
                    let mut v = body_sim_pool[ci].clone();
                    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let k = v.len().min(10);
                    let avg = if k == 0 { 0.0 } else { v[..k].iter().sum::<f32>() / (k as f32) };
                    raw.push(avg);
                }
                let mean_b: f32 = if raw.is_empty() { 0.0 } else { raw.iter().sum::<f32>() / (raw.len() as f32) };
                for (ci, cat) in categories.iter().enumerate() {
                    emit_term(&format!(
                        "  🗳️ [BODY CONSENSUS] {} | UniqueEvidence: {} | Top10Mean: {:.4} | Contrast: {:+.4}",
                        cat, body_sim_pool[ci].len(), raw[ci], raw[ci] - mean_b
                    ));
                }
                raw.iter().map(|v| v - mean_b).collect()
            };

            let title_probs: Vec<f32> = {
                let combined: Vec<f32> = categories.iter().map(|c| {
                    let (a, o) = category_title_scores.get(*c).copied().unwrap_or((0.0, 0.0));
                    a + o
                }).collect();
                let mx = combined.iter().cloned().fold(f32::MIN, f32::max);
                let temp = 0.05f32;
                let exps: Vec<f32> = combined.iter().map(|v| ((v - mx) / temp).exp()).collect();
                let sum_e: f32 = exps.iter().sum::<f32>().max(1e-6);
                exps.iter().map(|e| e / sum_e).collect()
            };

            let title_window_embs: Vec<Vec<f32>> = {
                let mut windows: Vec<String> = doc_title
                    .split(|c: char| c.is_whitespace() || c == '|' || c == '/' || c == '(' || c == ')' || c == '[' || c == ']' || c == '-' || c == ',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let title_chars: Vec<char> = doc_title.chars().filter(|c| !c.is_whitespace()).collect();
                for w in 2..=4usize {
                    if title_chars.len() < w { break; }
                    for st in 0..=(title_chars.len() - w) {
                        windows.push(title_chars[st..st + w].iter().collect::<String>());
                    }
                }
                let mut seen_win = std::collections::HashSet::new();
                windows.retain(|w| seen_win.insert(w.clone()));
                if windows.len() > 48 { windows.truncate(48); }
                if windows.is_empty() {
                    Vec::new()
                } else {
                    let raw_embs = model.get_embedding_batch(windows.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; windows.len()]);

                    let mut gated: Vec<Vec<f32>> = Vec::with_capacity(raw_embs.len());
                    let mut dropped_win = 0usize;
                    for (wi, we) in raw_embs.into_iter().enumerate() {
                        if we.iter().all(|&v| v == 0.0) { continue; }
                        let chrome_s = max_pool_sim(&we, &chrome_prej_embs);
                        let mut dom_s = 0.0f32;
                        for ci in 0..categories.len() {
                            let s = max_pool_sim(&we, &category_phrase_embs[ci].1);
                            if s > dom_s { dom_s = s; }
                        }
                        if chrome_s >= dom_s * 0.85 {
                            dropped_win += 1;
                            if dropped_win <= 8 {
                                emit_term(&format!("  🚫 [CHROME WINDOW DROP] '{}' | ChromeSim: {:.4} >= DomainMax: {:.4} x 0.85", windows[wi], chrome_s, dom_s));
                            }
                            continue;
                        }
                        gated.push(we);
                    }
                    if dropped_win > 0 {
                        emit_term(&format!("  🚫 [CHROME WINDOW GATE] 껍데기 n-gram 윈도우 {}개 제외 (잔존: {}개)", dropped_win, gated.len()));
                    }
                    gated
                }
            };

            let title_window_contrast: Vec<f32> = {
                let mut raw: Vec<f32> = Vec::new();
                for ci in 0..categories.len() {
                    let mut mx = 0.0f32;
                    for we in &title_window_embs {
                        let s = max_pool_sim(we, &category_phrase_embs[ci].1);
                        if s > mx { mx = s; }
                    }
                    raw.push(mx);
                }
                let mean_w: f32 = if raw.is_empty() { 0.0 } else { raw.iter().sum::<f32>() / (raw.len() as f32) };
                raw.iter().map(|v| v - mean_w).collect()
            };

            let title_trust: f32 = {                
                let mut sims_t: Vec<f32> = Vec::with_capacity(categories.len());
                for ci in 0..categories.len() {
                    sims_t.push(max_pool_sim(&title_emb, &category_phrase_embs[ci].1));
                }
                let dom_max = sims_t.iter().cloned().fold(0.0f32, f32::max);
                let mut ord_t = sims_t.clone();
                ord_t.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let direct_margin = (ord_t.get(0).copied().unwrap_or(0.0)
                    - ord_t.get(1).copied().unwrap_or(0.0)).max(0.0);
                let direct_trust = ((direct_margin - 0.02) / 0.08).clamp(0.0, 1.0);
                let chrome_s = max_pool_sim(&title_emb, &chrome_prej_embs);
                let chrome_trust = ((dom_max - chrome_s) / 0.15).clamp(0.0, 1.0);
                let mut wc = title_window_contrast.clone();
                wc.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let peak_margin = (wc.get(0).copied().unwrap_or(0.0) - wc.get(1).copied().unwrap_or(0.0)).max(0.0);
                let window_trust = ((peak_margin - 0.02) / 0.08).clamp(0.0, 1.0);
                let margin_trust = window_trust.max(direct_trust);
                let t = chrome_trust.min(margin_trust);
                emit_term(&format!(
                    "  🔒 [TITLE TRUST] DomainMax: {:.4} | ChromeSim: {:.4} | ChromeTrust: {:.3} | PeakMargin: {:.4} | WindowTrust: {:.3} | DirectMargin: {:.4} | DirectTrust: {:.3} → MarginTrust: {:.3} → Trust: {:.3}",
                    dom_max, chrome_s, chrome_trust, peak_margin, window_trust, direct_margin, direct_trust, margin_trust, t
                ));
                t
            };

            let (url_contrast, url_trust): (Vec<f32>, f32) = {
                let path_query = match url::Url::parse(&url) {
                    Ok(u) => format!(
                        "{} {}",
                        u.path(),
                        u.query().unwrap_or("")
                    ),
                    Err(_) => url.clone(),
                };
                let mut tokens: Vec<String> = Vec::new();
                for raw in path_query.split(|c: char| !c.is_alphanumeric()) {
                    let t = raw.trim().to_lowercase();
                    if t.is_empty() { continue; }
                    if t.chars().count() < 2 { continue; }
                    if t.chars().all(|c| c.is_ascii_digit()) { continue; }
                    if tokens.iter().any(|e| e == &t) { continue; }
                    tokens.push(t);
                }
                if tokens.len() > 16 { tokens.truncate(16); }
                if tokens.is_empty() {
                    emit_term("  🔗 [URL AXIS] 경로에서 판정 가능한 토큰을 얻지 못해 URL 축을 사용하지 않습니다.");
                    crate::utils::score_dynamics::record_baseline("commerce.url_trust", 0.0);
                    crate::utils::score_dynamics::record_baseline("commerce.url_margin", 0.0);
                    (vec![0.0f32; categories.len()], 0.0f32)
                } else {
                    let tok_embs = model
                        .get_embedding_batch(tokens.clone())
                        .await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; tokens.len()]);
                    let mut raw: Vec<f32> = Vec::with_capacity(categories.len());
                    let mut winner_tok: Vec<String> = Vec::with_capacity(categories.len());
                    for ci in 0..categories.len() {
                        let mut m = 0.0f32;
                        let mut w = String::new();
                        for (ti, te) in tok_embs.iter().enumerate() {
                            if te.iter().all(|&v| v == 0.0) { continue; }
                            let s = max_pool_sim(te, &category_phrase_embs[ci].1);
                            if s > m { m = s; w = tokens[ti].clone(); }
                        }
                        raw.push(m);
                        winner_tok.push(w);
                    }
                    let mean_u: f32 = raw.iter().sum::<f32>() / (raw.len() as f32);
                    let mut ord = raw.clone();
                    ord.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let margin = (ord.get(0).copied().unwrap_or(0.0) - ord.get(1).copied().unwrap_or(0.0)).max(0.0);
                    let trust = ((margin - 0.02) / 0.08).clamp(0.0, 1.0);
                    emit_term(&format!(
                        "  🔗 [URL AXIS] tokens: {:?} | Margin: {:.4} → Trust: {:.3}",
                        tokens, margin, trust
                    ));
                    crate::utils::score_dynamics::record_baseline("commerce.url_trust", trust);
                    crate::utils::score_dynamics::record_baseline("commerce.url_margin", margin);
                    for ci in 0..categories.len() {
                        emit_term(&format!(
                            "    🔗 [URL] {} | MaxPool: {:.4} (token '{}') | Contrast: {:+.4}",
                            categories[ci], raw[ci], winner_tok[ci], raw[ci] - mean_u
                        ));
                    }
                    (raw.iter().map(|v| v - mean_u).collect(), trust)
                }
            };

            let mut ev_line: Vec<f32> = vec![0.0; categories.len()];
            let mut ev_body: Vec<f32> = vec![0.0; categories.len()];
            let mut ev_url: Vec<f32> = vec![0.0; categories.len()];
            let mut ev_mean_line: Vec<f32> = vec![0.0; categories.len()];
            let mut ev_coverage: Vec<f32> = vec![0.0; categories.len()];
            let mut ev_count: Vec<f32> = vec![0.0; categories.len()];
            for (ci, cat) in categories.iter().enumerate() {
                
                
                
                let (line_total, contributing_lines) = category_line_scores.get(*cat).copied().unwrap_or((0.0, 0.0));
                let mean_line_contrast = if contributing_lines > 0.0 {
                    line_total / contributing_lines
                } else {
                    0.0
                };
                let coverage = if contributing_lines > 0.0 {
                    ((contributing_lines + 1.0).ln() / 4.0).min(1.2)
                } else {
                    0.0
                };
                let evidence_factor = if contributing_lines < 3.0 {
                    contributing_lines / 3.0
                } else {
                    1.0
                };
                ev_line[ci] = mean_line_contrast * 10.0 * coverage * evidence_factor;
                ev_body[ci] = body_consensus.get(ci).copied().unwrap_or(0.0).max(0.0) * 12.0;
                ev_url[ci] = url_contrast.get(ci).copied().unwrap_or(0.0).max(0.0) * 8.0 * url_trust;
                ev_mean_line[ci] = mean_line_contrast;
                ev_coverage[ci] = coverage;
                ev_count[ci] = contributing_lines;
            }
            let evidence_total: Vec<f32> = (0..categories.len())
                .map(|ci| ev_line[ci] + ev_body[ci] + ev_url[ci])
                .collect();
            
            let mut title_trust_eff = title_trust;
            {
                let mut t_pick = 0usize;
                let mut t_best = f32::MIN;
                for ci in 0..categories.len() {
                    if title_probs[ci] > t_best { t_best = title_probs[ci]; t_pick = ci; }
                }
                let mut e_pick = 0usize;
                let mut e_best = f32::MIN;
                let mut e_min = f32::MAX;
                for ci in 0..categories.len() {
                    if evidence_total[ci] > e_best { e_best = evidence_total[ci]; e_pick = ci; }
                    if evidence_total[ci] < e_min { e_min = evidence_total[ci]; }
                }
                if t_pick != e_pick {
                    let span = (e_best - e_min).max(1e-6);
                    let gap = ((e_best - evidence_total[t_pick]) / span).clamp(0.0, 1.0);
                    title_trust_eff = title_trust * (1.0 - gap);
                    emit_term(&format!(
                        "  🧪 [TITLE CORROBORATION] 타이틀은 '{}' 를 지목했지만 본문·URL 증거 1위는 '{}' 입니다. (증거 {:.3} vs {:.3} | 상대 격차 {:.3}) → TitleTrust {:.3} → {:.3}",
                        categories[t_pick], categories[e_pick],
                        evidence_total[e_pick], evidence_total[t_pick], gap,
                        title_trust, title_trust_eff
                    ));
                } else {
                    emit_term(&format!(
                        "  🧪 [TITLE CORROBORATION] 타이틀과 본문·URL 증거가 모두 '{}' 를 지목했습니다. TitleTrust {:.3} 유지.",
                        categories[e_pick], title_trust
                    ));
                }
            }
            
            for (ci, cat) in categories.iter().enumerate() {
                let (title_contrast, title_only_contrast) = category_title_scores.get(*cat).copied().unwrap_or((0.0, 0.0));
                let title_raw = category_title_raw.get(*cat).copied().unwrap_or(0.0);
                let title_signal = ((title_contrast.max(0.0) * 15.0) + (title_only_contrast.max(0.0) * 12.0)) * title_trust_eff;
                let line_signal = ev_line[ci];
                let body_signal = ev_body[ci];
                let url_signal = ev_url[ci];
                let mean_line_contrast = ev_mean_line[ci];
                let coverage = ev_coverage[ci];
                let contributing_lines = ev_count[ci];
                let body_contrast = body_consensus.get(ci).copied().unwrap_or(0.0);
                
                let title_prior_raw = 1.0 + 2.5 * title_probs[ci];
                let title_prior = 1.0 + (title_prior_raw - 1.0) * title_trust_eff;
                let win_contrast = title_window_contrast.get(ci).copied().unwrap_or(0.0);
                let boost_raw = (1.0 + 6.0 * win_contrast.max(0.0)).min(2.5);
                let title_keyword_boost = 1.0 + (boost_raw - 1.0) * title_trust_eff;
                emit_term(&format!("  🔤 [TITLE SOFT-CONTAINS] {} | WindowContrast: {:+.4} → raw {:.2}x × trust {:.3} → boost {:.2}x", cat, win_contrast, boost_raw, title_trust_eff, title_keyword_boost));
                let normalized_score = (title_signal + line_signal + body_signal + url_signal) * title_prior * title_keyword_boost;
                category_scores.push((cat.to_string(), normalized_score, title_raw, contributing_lines));
                if normalized_score > max_total_score {
                    max_total_score = normalized_score;
                    best_type = cat.to_string();
                }
                emit_term(&format!(
                    "  📐 [{}] TitleMaxPool: {:.4} | Contrast: {:+.4} | TitleP: {:.3} | Prior: {:.2}x | MeanLineContrast: {:.4} | Evidence: {:.2} | Coverage: {:.3} | BodyContrast: {:+.4} | TitleSig: {:.3} | LineSig: {:.3} | BodySig: {:.3} | UrlSig: {:.3}",
                    cat, title_raw, title_contrast, title_probs[ci], title_prior, mean_line_contrast, contributing_lines, coverage, body_contrast, title_signal, line_signal, body_signal, url_signal
                ));
            }

            emit_term("\n[PAGE-TYPE CLASSIFICATION] === Per-Category Score Breakdown ===");
            emit_term(&format!("  Document Title: '{}'", doc_title));
            // 🌟 [SDS 계측] 6개 카테고리 점수 배열의 감쇠 형상과 제목 신뢰도를 남깁니다.
            //
            //  ── 무엇을 보려는 것인가 ──
            //   현재 판정은 최고점 하나(max_total_score)로 page_type 을 확정합니다.
            //   그런데 title_trust / title_trust_eff 는 램프 상수(0.02/0.08/0.15)로
            //   유도된 값이라, 그 상수가 이 사이트에서 적절한지 알 방법이 없습니다.
            //   카테고리 점수 배열의 형상과 신뢰도를 함께 쌓아야
            //   Phase 1 의 역분산 융합이 램프 상수를 대체할 수 있습니다.
            {
                let arr: Vec<f32> = category_scores.iter().map(|(_, s, _, _)| *s).collect();
                if arr.len() >= 2 {
                    crate::utils::score_dynamics::record_decay("commerce.page_type", &arr);
                }
                crate::utils::score_dynamics::record_baseline("commerce.title_trust", title_trust);
                crate::utils::score_dynamics::record_baseline("commerce.title_trust_eff", title_trust_eff);
                crate::utils::score_dynamics::record_baseline("commerce.dedup_floor", dedup_floor);
                for (ci, cat) in categories.iter().enumerate() {
                    crate::utils::score_dynamics::record_category_max(
                        cat,
                        category_phrase_embs[ci].1.len(),
                        ev_line[ci] + ev_body[ci] + ev_url[ci],
                    );
                }
            }
            let mut sorted_scores = category_scores.clone();
            sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (cat, score, t_sim, line_cnt) in &sorted_scores {
                let marker = if *cat == best_type { "👑" } else { "  " };
                emit_term(&format!("  {} [{}] Normalized: {:.4} | TitleMaxPool: {:.4} | UniqueEvidence: {:.2}", marker, cat, score, t_sim, line_cnt));
            }
            emit_term(&format!("  Anchor Bias Sample (winner '{}'): '{}'...", best_type, crate::parsing::get_page_type_classification_bias(&best_type, &doc_lang).chars().take(120).collect::<String>()));
            emit_term("[PAGE-TYPE CLASSIFICATION] ====================================\n");
            page_type = best_type;
            println!("[Scheduler] Deterministic Classified Page Type: {} (Max Score: {:.4})", page_type, max_total_score);
            // 🌟 [SDS SCOPE] 1차 키를 확정된 page_type 으로 교체합니다.
            //    이 시점 이후의 관측은 'commerce|goods|' 처럼 도메인별로 격리됩니다.
            crate::utils::score_dynamics::refine_primary(&page_type);
            if page_type.is_empty() { 
                return Ok(()); 
            }
        }

        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Is Detail)");
            let (list_bias, form_bias, layout_prejudice) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            let prej_emb: Vec<f32> = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);

            let list_phrases = split_bias_phrases(&list_bias);
            let form_phrases = split_bias_phrases(&form_bias);
            let list_phrase_embs: Vec<Vec<f32>> = if list_phrases.is_empty() {
                vec![list_bias_emb.clone()]
            } else {
                model.get_embedding_batch(list_phrases.clone()).await.unwrap_or_else(|_| vec![list_bias_emb.clone(); list_phrases.len()])
            };
            let form_phrase_embs: Vec<Vec<f32>> = if form_phrases.is_empty() {
                vec![form_bias_emb.clone()]
            } else {
                model.get_embedding_batch(form_phrases.clone()).await.unwrap_or_else(|_| vec![form_bias_emb.clone(); form_phrases.len()])
            };
            emit_term(&format!("  🧩 [LAYOUT ANCHOR SPLIT] ListPhrases: {} | FormPhrases: {}", list_phrase_embs.len(), form_phrase_embs.len()));

            let layout_chrome_text = "global navigation, menus, header, footer, sidebar, breadcrumb, admin main menu, main menu, admin page, administrator page, dashboard, control panel, site name, shopping mall, welcome, home, index, basic search, search form, search filter, login, logout, notice, banner, copyright";
            let nav_chrome_embs = prejudice_pair(&model, layout_chrome_text, &doc_lang).await;

            {
                let nav_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, top bar, bottom bar, navigation bar, submenu, category menu, management menu, settings menu, configuration menu";
                let nav_prej_embs = prejudice_pair(&model, nav_prejudice_text, &doc_lang).await;

                let domain_phrase_embs: Vec<Vec<f32>> = {
                    let anchor_text = crate::parsing::get_page_type_classification_bias(&page_type, &doc_lang);
                    let localized_type = crate::parsing::get_localized_page_type(&page_type, &doc_lang);
                    let mut phrases: Vec<String> = anchor_text
                        .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
                        .flat_map(|seg| seg.split_whitespace())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    phrases.push(page_type.clone());
                    phrases.push(localized_type.clone());
                    let mut seen_p = std::collections::HashSet::new();
                    phrases.retain(|p| seen_p.insert(p.clone()));
                    if phrases.len() > 64 { phrases.truncate(64); }
                    if phrases.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                    }
                };

                let mut nav_wiped_count = 0usize;
                let mut nav_domain_protected = 0usize;
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }

                    if !line_embeddings[i].iter().all(|&v| v == 0.0) {
                        let nav_score = max_pool_sim(&line_embeddings[i], &nav_prej_embs);
                        if nav_score > 0.38 {

                            let mut domain_sim = 0.0f32;
                            for pe in &domain_phrase_embs {
                                let s = cosine_similarity(pe, &line_embeddings[i]);
                                if s > domain_sim { domain_sim = s; }
                            }

                            let layout_sim = cosine_similarity(&list_bias_emb, &line_embeddings[i])
                                .max(cosine_similarity(&form_bias_emb, &line_embeddings[i]));

                            let title_line_sim = cosine_similarity(&early_title_emb, &line_embeddings[i]);

                            if (domain_sim > 0.30 && domain_sim >= nav_score * 0.85)
                                || (layout_sim >= nav_score * 0.85)
                                || (title_line_sim > nav_score && title_line_sim > 0.40)
                            {
                                nav_domain_protected += 1;
                                continue;
                            }
                            wiped_indices[i] = true;
                            nav_wiped_count += 1;
                        }
                    }
                }
                if nav_wiped_count > 0 || nav_domain_protected > 0 {
                    emit_term(&format!("  🚫 [NAV PRE-FILTER] Step A-2 진입 전 네비게이션/레이아웃 {}개 라인 사전 탈락 완료. (도메인/레이아웃/타이틀 벡터 보호: {}개)", nav_wiped_count, nav_domain_protected));
                }
            }

            let system_content_a2 = format!("[PUG CONTENT]\n{}", filtered_light_pug);
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Scoring DOM blocks to determine page type...", "spinner": "⠋" }));
            emit_term("\n[CLASSIFICATION] Track B & C Vector Matching (Batch DOM Blocks)...");
            let mut list_scores = Vec::new();
            let mut form_scores = Vec::new();
            for (i, emb) in line_embeddings.iter().enumerate() {

                if wiped_indices[i] { continue; }
                let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }

                let prej_score = cosine_similarity(&prej_emb, emb);
                let list_s = cosine_similarity(&list_bias_emb, emb);
                let form_s = cosine_similarity(&form_bias_emb, emb);
                if prej_score > list_s && prej_score > form_s && prej_score > 0.35 {
                    continue;
                }
                list_scores.push((i, list_s));
                form_scores.push((i, form_s));
            }

            list_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            form_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut track_bc_candidates = Vec::new();
            let mut track_bc_indices = Vec::new();
            
            for (idx, _) in list_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }
            for (idx, _) in form_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }

            let js_template = get_boa_block_extractor_template();

            let track_bc_selectors: Vec<String> = {
                let target_len = track_bc_candidates.len(); 
                let target_titles_str = serde_json::to_string(&track_bc_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            let valid_bc_count = track_bc_selectors.iter().filter(|s| !s.is_empty()).count();
            emit_term(&format!("  📦 [Track B & C] Boa Engine successfully mapped {}/{} structural processing blocks.", valid_bc_count, track_bc_candidates.len()));

            let track_bc_pugs: Vec<(usize, String, String)> = {
                let html_clone = clean_html_content.clone();
                let selectors_with_idx: Vec<(usize, String)> = track_bc_selectors.into_iter().enumerate().collect();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_selectors = std::collections::HashSet::new();
                    let mut unique_tasks = Vec::new();
                    let mut fallback_results = Vec::new();
                    
                    for (i, sel) in selectors_with_idx {
                        if sel.is_empty() {
                            fallback_results.push((i, sel, String::new()));
                        } else if !seen_selectors.contains(&sel) {
                            seen_selectors.insert(sel.clone());
                            unique_tasks.push((i, sel));
                        } else {
                            fallback_results.push((i, sel, String::new()));
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_tasks.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_tasks.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for (i, sel) in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((i, sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results.extend(fallback_results);
                    results.sort_by_key(|k| k.0);
                    results
                }).await.unwrap_or_default()
            };

            let mut total_list_score = 0.0;
            let mut processed_list_blocks = std::collections::HashSet::new();
            let mut total_form_score = 0.0;
            let mut processed_form_blocks = std::collections::HashSet::new();

            let nav_block_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, navigation bar, submenu, category menu, management menu, settings menu, configuration menu, snb, gnb, nav, sidebar, side bar, left panel, right panel, quick links";
            let nav_block_prej_embs = prejudice_pair(&model, nav_block_prejudice_text, &doc_lang).await;

            let mut unique_bc_pugs_to_embed = Vec::new();
            let mut track_bc_pugs_clean: Vec<(usize, String, String, f32)> = Vec::new();
            for (i, sel, block_pug) in track_bc_pugs {
                let is_list_track = i < 5;
                if sel.is_empty() { 
                    let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                    emit_term(&format!("  ⚠️ [{}] Anchor Line {} failed to resolve a valid structural parent block via DOM.", track_name, track_bc_indices[i] + 1));
                    continue; 
                }

                let sel_naturalized: String = {
                    let lowered = sel.to_lowercase();
                    let mut out = String::new();
                    let mut prev_is_digit = false;
                    for ch in lowered.chars() {
                        if ch.is_alphanumeric() {
                            if prev_is_digit != ch.is_ascii_digit() && !out.is_empty() {
                                out.push(' ');
                            }
                            prev_is_digit = ch.is_ascii_digit();
                            out.push(ch);
                        } else {
                            if !out.ends_with(' ') { out.push(' '); }
                            prev_is_digit = false;
                        }
                    }
                    out.split_whitespace().collect::<Vec<_>>().join(" ")
                };

                let sel_emb = model.get_embedding(sel_naturalized.clone()).await.unwrap_or(vec![0.0f32; 384]);
                let sel_nav_score = max_pool_sim(&sel_emb, &nav_block_prej_embs);

                let sel_id_class_tokens: String = sel.to_lowercase()
                    .split(|c: char| c == ' ' || c == '>')
                    .flat_map(|part| {
                        let mut tokens = Vec::new();
                        if let Some(hash_pos) = part.find('#') {
                            let id_token: String = part[hash_pos+1..].chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
                            if !id_token.is_empty() { tokens.push(id_token.replace('_', " ").replace('-', " ")); }
                        }
                        for class_part in part.split('.') {
                            let class_token: String = class_part.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
                            if !class_token.is_empty() && !class_token.contains('#') { tokens.push(class_token.replace('_', " ").replace('-', " ")); }
                        }
                        tokens
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut sel_id_nav_score = 0.0f32;
                let mut sel_id_emb_opt: Option<Vec<f32>> = None;
                if !sel_id_class_tokens.is_empty() {
                    let sel_id_emb = model.get_embedding(sel_id_class_tokens.clone()).await.unwrap_or(vec![0.0f32; 384]);
                    sel_id_nav_score = max_pool_sim(&sel_id_emb, &nav_block_prej_embs);
                    sel_id_emb_opt = Some(sel_id_emb);
                }

                let effective_sel_nav_score = sel_nav_score.max(sel_id_nav_score);

                let sel_content_max = {
                    let mut m = cosine_similarity(&form_bias_emb, &sel_emb)
                        .max(cosine_similarity(&list_bias_emb, &sel_emb));
                    if let Some(idc) = &sel_id_emb_opt {
                        m = m.max(cosine_similarity(&form_bias_emb, idc))
                             .max(cosine_similarity(&list_bias_emb, idc));
                    }
                    m
                };

                let nav_dominance = if sel_content_max > 0.001 {
                    effective_sel_nav_score / sel_content_max
                } else {
                    f32::MAX
                };
                let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                if effective_sel_nav_score > 0.35 && nav_dominance > 1.35 {
                    emit_term(&format!("  🚫 [NAV VECTOR SELECTOR DROP] {} Anchor Line {} selector '{}' NavScore: {:.4} (ID/Class: {:.4}) | ContentSim: {:.4} | Dominance: {:.2}x > 1.35. Excluded.", track_name, track_bc_indices[i] + 1, sel, sel_nav_score, sel_id_nav_score, sel_content_max, nav_dominance));
                    continue;
                } else if effective_sel_nav_score > 0.35 {
                    emit_term(&format!("  🛡️ [NAV SELECTOR SOFT-CARRY] {} Anchor Line {} selector '{}' NavScore: {:.4} (ID/Class: {:.4}) | ContentSim: {:.4} | Dominance: {:.2}x <= 1.35. 드롭 대신 블록 단계로 이월.", track_name, track_bc_indices[i] + 1, sel, sel_nav_score, sel_id_nav_score, sel_content_max, nav_dominance));
                }
                if is_list_track {
                    if block_pug.is_empty() || processed_list_blocks.contains(&block_pug) { continue; }
                    processed_list_blocks.insert(block_pug.clone());
                } else {
                    if block_pug.is_empty() || processed_form_blocks.contains(&block_pug) { continue; }
                    processed_form_blocks.insert(block_pug.clone());
                }
                unique_bc_pugs_to_embed.push(block_pug.clone());
                track_bc_pugs_clean.push((i, sel, block_pug, effective_sel_nav_score));
            }

            let mut bc_embeddings_map = std::collections::HashMap::new();
            if !unique_bc_pugs_to_embed.is_empty() {
                for chunk in unique_bc_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            bc_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (i, sel, block_pug, sel_nav_carry) in track_bc_pugs_clean {
                let is_list_track = i < 5;
                let block_emb = bc_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);

                let nav_block_score = max_pool_sim(&block_emb, &nav_block_prej_embs);

                if nav_block_score > 0.25 {

                    let block_form_sim = cosine_similarity(&form_bias_emb, &block_emb);
                    let block_list_sim = cosine_similarity(&list_bias_emb, &block_emb);
                    let block_content_max = block_form_sim.max(block_list_sim);
                    if block_content_max > nav_block_score * 0.85 {
                        let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                        emit_term(&format!("  🛡️ [NAV BLOCK CONTENT PROTECT] {} Anchor: {} | Selector: '{}' | NavScore: {:.4} but ContentSim: {:.4} >= 85% of Nav. Protected.", track_name, track_bc_indices[i] + 1, sel, nav_block_score, block_content_max));
                    } else {
                        let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                        emit_term(&format!("  🚫 [NAV VECTOR DROP] {} Anchor: {} | Selector: '{}' | NavScore: {:.4} > 0.25. Navigation block excluded.", track_name, track_bc_indices[i] + 1, sel, nav_block_score));
                        continue;
                    }
                }
                let mut b_prej_score = cosine_similarity(&prej_emb, &block_emb);


                if nav_block_score > 0.15 {
                    b_prej_score += nav_block_score * 0.5;
                }

                if sel_nav_carry > 0.35 {
                    b_prej_score += (sel_nav_carry - 0.35) * 0.5;
                }

                if is_list_track {
                    // 🌟 [CROSSOVER] sel 은 위쪽 NAV 판정에서 이미 임베딩된 문자열입니다.
                    //    get_embedding 의 메모리 캐시가 재계산을 차단하므로
                    //    이 호출은 GPU 연산 없이 즉시 반환됩니다. (구조 변경 없음)
                    let sel_emb = model.get_embedding(sel.to_lowercase()).await.unwrap_or(vec![0.0f32; 384]);
                    let sel_list_sim = cosine_similarity(&list_bias_emb, &sel_emb);
                    if sel_list_sim > 0.30 {
                        b_prej_score *= 0.70; 
                    }
                    let b_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                    let final_score = (b_list_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_list_score += final_score;
                        emit_term(&format!("  📊 [TRACK B (LIST)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | NavScore: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_list_score, b_prej_score, nav_block_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK B (LIST)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_list_score));
                    }
                } else {
                    let sel_emb = model.get_embedding(sel.to_lowercase()).await.unwrap_or(vec![0.0f32; 384]);
                    let sel_form_sim = cosine_similarity(&form_bias_emb, &sel_emb);
                    if sel_form_sim > 0.30 {
                        b_prej_score *= 0.70;
                    }
                    let b_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                    let final_score = (b_form_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_form_score += final_score;
                        emit_term(&format!("  📊 [TRACK C (FORM)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | NavScore: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_form_score, b_prej_score, nav_block_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK C (FORM)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_form_score));
                    }
                }
            }

            let (heading_list_sim, heading_form_sim, heading_text) = {
                let heads: Vec<(usize, String)> = {
                    let doc = scraper::Html::parse_document(&clean_html_content);
                    let mut temp: Vec<(usize, String)> = Vec::new();
                    for (tier, tag) in ["h1", "h2"].iter().enumerate() {
                        if let Ok(sel_h) = scraper::Selector::parse(tag) {
                            for el in doc.select(&sel_h) {
                                let txt = el.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
                                if !txt.is_empty() && txt.chars().count() <= 60 { temp.push((tier, txt)); }
                            }
                        }
                    }
                    if temp.len() > 16 { temp.truncate(16); }
                    temp
                };

                if heads.is_empty() {
                    (0.0f32, 0.0f32, String::new())
                } else {
                    let head_texts: Vec<String> = heads.iter().map(|(_, t)| t.clone()).collect();
                    let head_embs = model.get_embedding_batch(head_texts.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; head_texts.len()]);

                    let mut best_tier = usize::MAX;
                    let mut best_gap = -1.0f32;
                    let mut sel_l = 0.0f32;
                    let mut sel_f = 0.0f32;
                    let mut sel_txt = String::new();
                    for (hi, he) in head_embs.iter().enumerate() {
                        if he.iter().all(|&v| v == 0.0) { continue; }
                        let tier = heads[hi].0;
                        let txt = &heads[hi].1;
                        let l = max_pool_sim(he, &list_phrase_embs);
                        let f = max_pool_sim(he, &form_phrase_embs);
                        let gap = (l - f).abs();
                        let chrome_s = max_pool_sim(he, &nav_chrome_embs);
                        let layout_max = l.max(f);
                        if chrome_s >= layout_max * 0.90 {
                            emit_term(&format!("  🚫 [HEADING CHROME DROP] '{}' (h{}) | ChromeSim: {:.4} >= LayoutMax: {:.4} x 0.90", txt, tier + 1, chrome_s, layout_max));
                            continue;
                        }
                        emit_term(&format!("  🧷 [HEADING CANDIDATE] '{}' (h{}) | ListMaxPool: {:.4} | FormMaxPool: {:.4} | Gap: {:+.4} | ChromeSim: {:.4}", txt, tier + 1, l, f, l - f, chrome_s));
                        if tier < best_tier || (tier == best_tier && gap > best_gap) {
                            best_tier = tier;
                            best_gap = gap;
                            sel_l = l;
                            sel_f = f;
                            sel_txt = txt.clone();
                        }
                    }
                    (sel_l, sel_f, sel_txt)
                }
            };

            let (periodicity_contrast, best_stride, periodicity_baseline) = {
                let mut content_idxs: Vec<usize> = Vec::new();
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let text_part = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" };
                    if text_part.is_empty() { continue; }
                    if line_embeddings[i].iter().all(|&v| v == 0.0) { continue; }
                    content_idxs.push(i);
                }
                let n = content_idxs.len();
                if n < 20 {
                    (0.0f32, 0usize, 0.0f32)
                } else {
                    let max_stride = (n / 3).min(40);
                    let mut stride_means: Vec<(usize, f32)> = Vec::new();
                    for stride in 2..=max_stride {
                        let mut sum = 0.0f32;
                        let mut cnt = 0usize;
                        for k in 0..(n - stride) {
                            let a = content_idxs[k];
                            let b = content_idxs[k + stride];
                            sum += cosine_similarity(&line_embeddings[a], &line_embeddings[b]);
                            cnt += 1;
                        }
                        if cnt >= 6 { stride_means.push((stride, sum / (cnt as f32))); }
                    }
                    if stride_means.is_empty() {
                        (0.0f32, 0usize, 0.0f32)
                    } else {

                        let base: f32 = stride_means.iter().map(|(_, m)| *m).sum::<f32>() / (stride_means.len() as f32);
                        let mut bs = 0usize;
                        let mut bm = -1.0f32;
                        for (s, m) in &stride_means {
                            if *s >= 5 && *m > bm { bm = *m; bs = *s; }
                        }
                        if bs == 0 { (0.0f32, 0usize, base) } else { ((bm - base).max(0.0), bs, base) }
                    }
                }
            };

            let (row_repeat_score, row_uniformity, row_baseline, row_dbg) = {
                let harvested: Vec<(Vec<String>, usize)> = {
                    let doc = scraper::Html::parse_document(&clean_html_content);
                    let mut out: Vec<(Vec<String>, usize)> = Vec::new();
                    if let (Ok(tbl_sel), Ok(tr_sel), Ok(cell_sel)) = (
                        scraper::Selector::parse("table"),
                        scraper::Selector::parse("tr"),
                        scraper::Selector::parse("td, th"),
                    ) {
                        for tbl in doc.select(&tbl_sel) {
                            let mut rows: Vec<String> = Vec::new();
                            let mut cell_counts: Vec<usize> = Vec::new();
                            for tr in tbl.select(&tr_sel) {
                                let txt = tr.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
                                if txt.chars().count() < 4 { continue; }
                                rows.push(txt.chars().take(400).collect::<String>());
                                cell_counts.push(tr.select(&cell_sel).count());
                            }
                            if rows.len() < 2 { continue; }
                            if rows.len() > 30 { rows.truncate(30); cell_counts.truncate(30); }
                            let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                            for c in &cell_counts { *freq.entry(*c).or_insert(0) += 1; }
                            let modal_cells = freq.iter().max_by_key(|(_, v)| **v).map(|(k, _)| *k).unwrap_or(0);
                            out.push((rows, modal_cells));
                        }
                    }
                    if out.len() > 10 { out.truncate(10); }
                    out
                };

                if harvested.is_empty() {
                    (0.0f32, 0.0f32, 0.0f32, String::from("no table"))
                } else {
                    let mut all_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                    for (rows, _) in &harvested {
                        let e = model.get_embedding_batch(rows.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; rows.len()]);
                        all_embs.push(e);
                    }

                    let mut best_uni = 0.0f32;
                    let mut best_ti: i32 = -1;
                    let mut best_row_stride = 0usize;
                    for (ti, (rows, modal_cells)) in harvested.iter().enumerate() {
                        if rows.len() < 3 || *modal_cells < 3 {
                            emit_term(&format!("  ⏭️ [ROW REPETITION SKIP] table[{}] rows:{} modalCells:{} (리스트 자격 미달)", ti, rows.len(), modal_cells));
                            continue;
                        }
                        let embs = &all_embs[ti];
                        for stride in 1..=3usize {
                            if embs.len() <= stride { break; }
                            let mut s = 0.0f32;
                            let mut c = 0usize;
                            for k in stride..embs.len() {
                                s += cosine_similarity(&embs[k - stride], &embs[k]);
                                c += 1;
                            }
                            if c < 2 { continue; }
                            let m = s / (c as f32);
                            if m > best_uni {
                                best_uni = m;
                                best_ti = ti as i32;
                                best_row_stride = stride;
                            }
                        }
                    }

                    if best_ti < 0 {
                        (0.0f32, 0.0f32, 0.0f32, String::from("no qualifying list table"))
                    } else {

                        let mut base_sum = 0.0f32;
                        let mut base_cnt = 0usize;
                        for a in 0..all_embs.len() {
                            for b in (a + 1)..all_embs.len() {
                                for ea in all_embs[a].iter().take(6) {
                                    for eb in all_embs[b].iter().take(6) {
                                        base_sum += cosine_similarity(ea, eb);
                                        base_cnt += 1;
                                    }
                                }
                            }
                        }

                        if base_cnt == 0 {
                            let embs = &all_embs[best_ti as usize];
                            let far = (embs.len() / 2).max(1);
                            for k in far..embs.len() {
                                base_sum += cosine_similarity(&embs[k - far], &embs[k]);
                                base_cnt += 1;
                            }
                        }
                        let baseline = if base_cnt > 0 { base_sum / (base_cnt as f32) } else { 0.0 };


                        let n_rows = all_embs[best_ti as usize].len() as f32;
                        let volume = (((n_rows - 2.0).max(0.0)).ln_1p() / 2.0).min(1.2);

                        let contrast = (best_uni - baseline).max(0.0);
                        let score = contrast * volume;
                        let dbg = format!("table[{}] rows:{} modalCells:{} rowStride:{} volume:{:.3}",
                            best_ti, n_rows as usize, harvested[best_ti as usize].1, best_row_stride, volume);
                        (score, best_uni, baseline, dbg)
                    }
                }
            };

            emit_term(&format!("  🧱 [ROW REPETITION] {} | Uniformity: {:.4} | Baseline: {:.4} | Contrast: {:+.4} | Score: {:.4}", row_dbg, row_uniformity, row_baseline, row_uniformity - row_baseline, row_repeat_score));

            let heading_gap = heading_list_sim - heading_form_sim;
            let heading_list_bonus = heading_gap.max(0.0) * 2.0;
            let heading_form_bonus = (-heading_gap).max(0.0) * 2.0;
            let periodicity_bonus = periodicity_contrast * 2.0;
            let row_repeat_bonus = row_repeat_score * 3.0;

            let list_measured = total_list_score > 0.0001;
            let form_measured = total_form_score > 0.0001;
            let track_damp = if list_measured != form_measured { 0.5f32 } else { 1.0f32 };
            let eff_list_track = total_list_score * track_damp;
            let eff_form_track = total_form_score * track_damp;
            if track_damp < 1.0 {
                emit_term(&format!("  ⚖️ [TRACK ASYMMETRY GUARD] 한쪽 트랙 측정 실패(List: {:.4} / Form: {:.4}). 양쪽 트랙 50% 감쇠 적용.", total_list_score, total_form_score));
            }
            crate::utils::score_dynamics::record_baseline("commerce.heading_gap", heading_gap);
            crate::utils::score_dynamics::record_baseline("commerce.periodicity_contrast", periodicity_contrast);
            crate::utils::score_dynamics::record_baseline("commerce.row_repeat_score", row_repeat_score);
            crate::utils::score_dynamics::record_baseline("commerce.track_damp", track_damp);
            crate::utils::score_dynamics::record_baseline("commerce.track_list_raw", total_list_score);
            crate::utils::score_dynamics::record_baseline("commerce.track_form_raw", total_form_score);

            let list_final = eff_list_track + heading_list_bonus + periodicity_bonus + row_repeat_bonus;
            let form_final = eff_form_track + heading_form_bonus;

            emit_term(&format!("  🧭 [HEADING VECTOR] '{}' | ListMaxPool: {:.4} | FormMaxPool: {:.4} | Gap: {:+.4}", heading_text, heading_list_sim, heading_form_sim, heading_gap));
            emit_term(&format!("  🔁 [PERIODICITY COSINE] BestStride: {} | PeakContrast: {:+.4} | Baseline: {:.4}", best_stride, periodicity_contrast, periodicity_baseline));
            emit_term(&format!("  🧮 [EVIDENCE SUM] ListFinal: {:.4} (track {:.4} + heading {:.4} + period {:.4} + rows {:.4}) | FormFinal: {:.4} (track {:.4} + heading {:.4})", list_final, eff_list_track, heading_list_bonus, periodicity_bonus, row_repeat_bonus, form_final, eff_form_track, heading_form_bonus));

            let decision_margin = (form_final - list_final).abs();
            crate::utils::score_dynamics::record_baseline(
                "commerce.detail_fallback",
                if decision_margin < 0.02 { 1.0 } else { 0.0 },
            );
            if decision_margin < 0.02 {
                emit_term(&format!("  ⚠️ [LOW-CONFIDENCE FALLBACK] 판정 마진 {:.4} < 0.02. 전체 PUG 직접 임베딩 폴백 가동.", decision_margin));
                let fallback_pug_emb = model.get_embedding(filtered_light_pug.clone()).await.unwrap_or(vec![0.0f32; 384]);

                let fallback_form_sim = max_pool_sim(&fallback_pug_emb, &form_phrase_embs);
                let fallback_list_sim = max_pool_sim(&fallback_pug_emb, &list_phrase_embs);
                let fallback_prej_sim = cosine_similarity(&prej_emb, &fallback_pug_emb);
                let fallback_form_final = (fallback_form_sim - fallback_prej_sim).max(0.0) + heading_form_bonus;
                let fallback_list_final = (fallback_list_sim - fallback_prej_sim).max(0.0) + heading_list_bonus + periodicity_bonus + row_repeat_bonus;
                is_detail = fallback_form_final > fallback_list_final;
                emit_term(&format!("  📊 [FALLBACK SCORE] FormMaxPool: {:.4} | ListMaxPool: {:.4} | PrejSim: {:.4} | FormFinal: {:.4} | ListFinal: {:.4} | is_detail: {}", fallback_form_sim, fallback_list_sim, fallback_prej_sim, fallback_form_final, fallback_list_final, is_detail));
            } else {
                is_detail = form_final > list_final;
            }
            println!("[Scheduler] Classified is_detail as: {} (Form: {:.4}, List: {:.4})", is_detail, form_final, list_final);
            emit_term(&format!("  ✅ Determined Detail Page: {}", is_detail));
            crate::utils::score_dynamics::record_baseline("commerce.list_final", list_final);
            crate::utils::score_dynamics::record_baseline("commerce.form_final", form_final);
            crate::utils::score_dynamics::record_baseline("commerce.detail_margin", (form_final - list_final).abs());
            crate::utils::score_dynamics::record_decay("commerce.detail_axis", &[list_final.max(form_final), list_final.min(form_final)]);
            if !page_type.is_empty() {
                crate::utils::score_dynamics::refine_primary(&format!(
                    "{}:{}",
                    page_type,
                    if is_detail { "detail" } else { "list" }
                ));
            }
        }
    }

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
    let mut census_selector_used = false;
    let list_census: Option<crate::scheduler::list_census::ListCensus> = if !is_detail && !skip_ai_analysis {
        let mut census = crate::scheduler::list_census::run(&clean_html_content);
        crate::scheduler::list_census::score_content(&mut census, &model, &page_type, &doc_lang).await;
        for line in census.report_lines() {
            emit_term(&line);
        }
        crate::utils::score_dynamics::record_baseline("commerce.census_data_groups", census.data_grade_count() as f32);
        Some(census)
    } else {
        None
    };
    model.deep_purge_resources().await;
    model.mark_crossover_idle();
 
    {
        let q3_clear_arc = model.qwen3_generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                gen.clear_kv_cache();
            }
        }).await;
        
        let gen_clear_arc = model.generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = gen_clear_arc.blocking_lock().as_mut() {
                let _ = gen.clear_kv_cache();
            }
        }).await;

        if !model.is_cpu_mode {
            let dev = model.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { let _ = dev.synchronize(); }
            }).await;
        }
    }
 
    crate::utils::resources::wait_for_resources_settled(1200, 800, Some(&cancellation_token), model.device_config.gpu_id as u32).await?;

    let mut extracted_data = json!({});

    if !is_detail {
        
        if !skip_ai_analysis {

            {
                use boa_engine::{Context, Source};
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                println!("[Scheduler] Starting JS-BASED SELECTOR ANALYSIS (LLM Titles -> Boa Engine)");
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Selector Search", "summary": "Analyzing DOM with JS engine...", "spinner": "⠋" }));


                let title_prompt = parsing::extract_titles_prompt(&page_type);
                let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY.", title_prompt);
                let snapshot_id = format!("{}_step_b_titles", task.id);



                let title_excerpt = if base_model_size == crate::model::ModelSize::Qwen {
                    None
                } else {
                    list_census.as_ref().and_then(|c| c.title_excerpt(&clean_html_content))
                };
                crate::utils::score_dynamics::record_baseline(
                    "commerce.title_excerpt_available",
                    if title_excerpt.is_some() { 1.0 } else { 0.0 },
                );
                let (titles, titles_well_formed): (Vec<String>, bool) = {
                    let mut contexts: Vec<(String, bool)> = Vec::new();
                    if let Some(ex) = title_excerpt.as_ref() {
                        contexts.push((format!("[PUG CONTENT]\n{}", ex.text), true));
                    }
                    contexts.push((system_content.clone(), false));
                    let mut q35_ready = false;
                    let mut outcome: (Vec<String>, bool) = (Vec::new(), false);
                    for (context, from_excerpt) in contexts.into_iter() {
                        let call_session = if from_excerpt { format!("{}_excerpt", snapshot_id) } else { snapshot_id.clone() };
                        let params = ChatCompletionParameters {
                            messages: vec![
                                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                                    content: context.clone(),
                                    name: None,
                                }),
                                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                    content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                                    name: None,
                                })
                            ],
                            model: if base_model_size == crate::model::ModelSize::Qwen { "qwen".to_string() } else { "qwen3.5".to_string() },
                            max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        let res = if base_model_size == crate::model::ModelSize::Qwen {
                            model
                                .enter_generation_phase(
                                    crate::model::ModelSize::Qwen,
                                    Some(&base_session_id),
                                    Some(cancellation_token.clone()),
                                    false,
                                    kv_name.clone(),
                                    "title extraction (Qwen 0.6B)",
                                )
                                .await?;
                            if let Some(gen) = model.generator.lock().await.as_mut() {
                                println!("[JS-BRIDGE] 1. Requesting titles from LLM (0.6B)...");
                                let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                                gen.generate(
                                    params,
                                    Some(cancellation_token.clone()),
                                    Some(call_session.clone()),
                                    kv_name.clone(),
                                    Some(&title_prej)
                                ).await?
                            } else {
                                return Err(anyhow::anyhow!("Qwen generator missing"));
                            }
                        } else {
                            if !q35_ready {
                                model
                                    .switch_to_generation(
                                        crate::model::ModelSize::Qwen3_5,
                                        Some(cancellation_token.clone()),
                                        kv_name.clone(),
                                        "title extraction (Qwen3.5-2B)",
                                    )
                                    .await?;
                                q35_ready = true;
                            }
                            let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                            if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                                println!("[JS-BRIDGE] 1. Requesting titles from LLM (Qwen3.5-2B)...");
                                gen.generate(
                                    params,
                                    Some(cancellation_token.clone()),
                                    Some(call_session.clone()),
                                    kv_name.clone(),
                                    None,
                                    Some(&title_prej),
                                ).await?
                            } else {
                                return Err(anyhow::anyhow!("Qwen3.5 generator missing"));
                            }
                        };
                        println!("[JS-BRIDGE] LLM Raw Response: '{}'", res);
                        let title_info = parsing::parse_json_from_llm(&res);
                        let well_formed = title_info.as_object().map_or(false, |obj| !obj.is_empty());
                        let harvested = crate::scheduler::list_census::harvest_titles(&title_info, &res);
                        println!("[JS-BRIDGE] Titles extracted (Robust): {:?}", harvested);
                        if from_excerpt {
                            crate::utils::score_dynamics::record_baseline(
                                "commerce.title_excerpt_hit",
                                if harvested.is_empty() { 0.0 } else { 1.0 },
                            );
                            if harvested.is_empty() {
                                emit_term("  ✂️ [TITLE EXCERPT / EMPTY] 반복 구조 발췌에서 제목을 찾지 못했습니다. 기존과 같은 전체 PUG 로 한 번 더 묻습니다.");
                                continue;
                            }
                            if let Some(ex) = title_excerpt.as_ref() {
                                let ratio = context.chars().count() as f32 / system_content.chars().count().max(1) as f32;
                                crate::utils::score_dynamics::record_baseline("commerce.title_excerpt_ratio", ratio);
                                emit_term(&format!(
                                    "  ✂️ [TITLE EXCERPT] 전수 조사 1위 '{}' 가 나머지 후보보다 구조점수 {:.1}배 우세해, 2B 에 전체 PUG 대신 이 반복 구조의 첫 {}행(전체 {}행)과 머리행 {}칸만 보냈습니다 | 문맥 {}자 / 전체 PUG {}자 ({:.1}%) | 제목 {}개. 제목으로 반복 구조를 찾는 Boa 판정은 그대로이며, 이 경우 아래 CENSUS SHADOW 는 같은 후보에서 나온 제목이라 독립 관측이 아닙니다.",
                                    ex.selector,
                                    ex.dominance,
                                    ex.rows,
                                    ex.members,
                                    ex.header_cells,
                                    context.chars().count(),
                                    system_content.chars().count(),
                                    ratio * 100.0,
                                    harvested.len()
                                ));
                            }
                        }
                        crate::utils::score_dynamics::record_baseline("commerce.title_llm_count", harvested.len() as f32);
                        outcome = (harvested, well_formed);
                        break;
                    }
                    outcome
                };
                if base_model_size == crate::model::ModelSize::Qwen {
                    model.deep_purge_resources().await;
                }
                let mut census_selector: Option<serde_json::Value> = None;
                if titles.is_empty() {
                    let census = list_census.as_ref();
                    if let Some(empty) = census.filter(|_| titles_well_formed).and_then(|c| c.empty_verdict()) {
                        emit_term(&format!(
                            "  📭 [EMPTY LIST VERDICT] LLM 이 정상 JSON 으로 빈 제목 목록을 돌려주었고 DOM 전수 조사도 같은 결론입니다: 표 '{}' 는 머리행 {}칸 · 본문 {}행이며 본문에 데이터 행이 없습니다 (\"{}\"). 폼·랜드마크 밖에서 여러 칸짜리 반복 구조도 0개입니다. 서로 독립인 두 근거가 일치하므로 오류로 중단하지 않고 '0건 목록' 으로 완료합니다. 셀렉터는 캐시하지 않으므로 행이 생긴 뒤 같은 페이지를 넣으면 처음부터 분석합니다.",
                            empty.selector, empty.header_cells, empty.body_rows, empty.notice
                        ));
                        crate::utils::score_dynamics::record_baseline("commerce.census_empty_verdict", 1.0);
                        let payload = json!({
                            "task_id": task.id,
                            "category": "Done",
                            "summary": "Extraction complete. The list has no rows (0 items).",
                            "spinner": "✅",
                            "data": null
                        });
                        let _ = app_handle.emit("extraction-progress", &payload);
                        log_task_progress(app_handle, &task.id, &payload);
                        emit_term(&format!("[PROCESS] {}", crate::utils::score_dynamics::report()));
                        crate::utils::score_dynamics::flush();
                        crate::utils::score_dynamics::leave_scope();
                        return Ok(());
                    }
                    match census.and_then(|c| c.decisive_fallback()) {
                        Some(g) => {
                            emit_term(&format!(
                                "  🧮 [CENSUS FALLBACK / NO TITLES] LLM 제목이 비어 Boa 가 기준으로 삼을 텍스트가 없지만, DOM 전수 조사가 결정적인 목록 1개를 찾았습니다: '{}' | 행 {} · 칸 평균 {:.1} · 내용마진 {:+.4} · 2위 대비 구조점수 2배 이상. 이 셀렉터로 이어 가되 페이지 캐시에는 남기지 않습니다.",
                                g.item_selector, g.members, g.avg_cells, g.content_margin.unwrap_or(0.0)
                            ));
                            crate::utils::score_dynamics::record_baseline("commerce.census_fallback", 1.0);
                            census_selector = Some(g.selector_json());
                        }
                        None if !titles_well_formed => {
                            return Err(anyhow::anyhow!("LLM returned invalid or unparseable JSON response during title extraction."));
                        }
                        None => {
                            return Err(anyhow::anyhow!("[JS-BRIDGE] No titles extracted from LLM. Aborting task to prevent invalid DOM fallback."));
                        }
                    }
                }

                if let Some(sel) = census_selector.take() {
                    selector_info = sel;
                    census_selector_used = true;
                } else {
                    println!("[JS-BRIDGE] 2. Starting boa-engine for DOM analysis...");
                    let mut context = Context::default();
                    
                    let document = scraper::Html::parse_document(&clean_html_content);
                    
                    let mut nodes_json = Vec::new();
                    let mut node_to_idx = std::collections::HashMap::new();

                    for (idx, node) in document.tree.root().descendants().enumerate() {
                        node_to_idx.insert(node.id(), idx);
                    }

                    for (idx, node) in document.tree.root().descendants().enumerate() {
                        if let Some(el) = node.value().as_element() {
                            let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                            
                            let text: String = node.children()
                                .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                                .collect::<Vec<_>>()
                                .join(" ")
                                .trim()
                                .to_string();
                                
                            
                            nodes_json.push(json!({
                                "index": idx,
                                "parentIndex": parent_idx,
                                "tagName": el.name().to_string(),
                                "id": el.id().unwrap_or("").to_string(),
                                "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                                "text": text,
                                "colspan": el.attr("colspan").unwrap_or("1"),
                                "rowspan": el.attr("rowspan").unwrap_or("1")
                            }));
                        } else {
                            nodes_json.push(json!(null));
                        }
                    }
                    
                    let nodes_str = serde_json::to_string(&nodes_json)?;
                    let titles_str = serde_json::to_string(&titles)?;

                    let js_template = get_boa_js_template();


                    let js_code = js_template
                        .replace("NODES_PLACEHOLDER", &nodes_str)
                        .replace("TITLES_PLACEHOLDER", &titles_str);

                    match context.eval(Source::from_bytes(js_code.as_bytes())) {
                        Ok(val) => {
                            let res_str = val.as_string().map(|s| s.to_std_string_escaped()).unwrap_or_default();
                            println!("[JS-BRIDGE] Boa Final Result: {}", res_str);

                            selector_info = serde_json::from_str(&res_str).unwrap_or(json!({}));
                        },
                        Err(e) => {
                            println!("[JS-BRIDGE] Error executing JS: {:?}", e);
                        }
                    }
                    let boa_count = selector_info.get("matchCount").and_then(|v| v.as_u64()).unwrap_or(0);
                    let boa_items = selector_info.get("itemSelector").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    if boa_count == 0 {
                        match list_census.as_ref().and_then(|c| c.anchor_titles(&titles).or_else(|| c.decisive_fallback())) {
                            Some(g) => {
                                emit_term(&format!(
                                    "  🧮 [CENSUS FALLBACK / BOA MISS] Boa 가 제목 {:?} 로 반복 구조를 찾지 못했습니다 (matchCount 0 이면 기본값 'body div' 가 페이지의 모든 div 를 아이템으로 만듭니다). DOM 전수 조사에서 제목을 품은 반복 구조 '{}' (행 {} · 칸 평균 {:.1}) 로 대신하고, 페이지 캐시에는 남기지 않습니다.",
                                    titles, g.item_selector, g.members, g.avg_cells
                                ));
                                crate::utils::score_dynamics::record_baseline("commerce.census_fallback", 1.0);
                                selector_info = g.selector_json();
                                census_selector_used = true;
                            }
                            None => {
                                return Err(anyhow::anyhow!("[JS-BRIDGE] Boa found no repeated structure for titles {:?} and the DOM census has no candidate that contains them. Aborting task to prevent invalid DOM fallback ('body div').", titles));
                            }
                        }
                    } else if let Some(census) = list_census.as_ref() {
                        match census.shadow_agreement(&clean_html_content, &boa_items) {
                            Some((jac, g, rows, records)) => {
                                crate::utils::score_dynamics::record_baseline("commerce.census_shadow_record_jaccard", jac);
                                emit_term(&format!(
                                    "  🧮 [CENSUS SHADOW] Boa '{}' 행 {}개 → 레코드 {}개 ↔ 전수 조사 1위 '{}' {}개 | 레코드 겹침(Jaccard) {:.2} — 관측만 합니다. rowspan 으로 이어진 행은 목록 분해(split_doc_to_pug_list_advanced)와 같은 규칙으로 한 레코드로 셉니다. 이 분포가 쌓이면 둘이 어긋난 문서에서 어느 쪽이 맞았는지를 숫자로 보고 승격 여부를 정합니다.",
                                    boa_items, rows, records, g.item_selector, g.members, jac
                                ));
                            }
                            None => {
                                emit_term("  🧮 [CENSUS SHADOW] 전수 조사에 여러 칸짜리 반복 구조가 없어 Boa 결과와 비교하지 않습니다.");
                            }
                        }
                    }
                }
            }
        }

        
        let target_selector = selector_info.get("final_target_selector")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                let item_selector = selector_info.get("itemSelector")
                    .or_else(|| selector_info.get("item"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let node_selector = selector_info.get("node").or_else(|| selector_info.get("parent")).and_then(|s| s.as_str()).unwrap_or("");
                
                if !node_selector.is_empty() && !item_selector.is_empty() && !item_selector.contains(",") {
                    if item_selector.starts_with(node_selector) {
                        item_selector.to_string()
                    } else {
                        format!("{} {}", node_selector, item_selector) 
                    }
                } else if !item_selector.is_empty() { 
                    item_selector.to_string() 
                } else { 
                    node_selector.to_string() 
                }
            }).replace(">", " "); 
            
        emit_term(&format!("[Scheduler] Target Selector configured as: '{}'", target_selector));

        let mut final_thead_selector = String::new();
        let mut cache_updated = false;
        let mut thead_pug = String::new();


        if let Some(sel) = selector_info.get("head").and_then(|v| v.as_str()) {
            if !sel.is_empty() && sel != "..." {
                final_thead_selector = sel.to_string();
                println!("[Scheduler] Using cached head selector: {}", final_thead_selector);
            }
        } 
        

        if final_thead_selector.is_empty() {
            match derive_thead_selector(&clean_html_content, &target_selector) {
                Some((head, table)) => {
                    emit_term(&format!(
                        "  🧭 [DETERMINISTIC THEAD / DOM FIRST] 목록 첫 행 '{}' 의 조상 표 '{}' 에서 머리행 '{}' (th {}개) 을 LLM 없이 확정했습니다. LLM 셀렉터 수확도 같은 DOM 검증(th 2개 이상 · th 가 td 보다 많음)을 통과해야 채택되므로, DOM 이 먼저 답을 가지면 2B 프리필을 쓰지 않습니다.",
                        target_selector,
                        table,
                        head,
                        best_thead_th_dbg(&head, &clean_html_content)
                    ));
                    crate::utils::score_dynamics::record_baseline("commerce.thead_dom_first", 1.0);
                    selector_info.as_object_mut().unwrap().insert("head".to_string(), json!(head.clone()));
                    selector_info.as_object_mut().unwrap().insert("wrapper".to_string(), json!(table));
                    final_thead_selector = head;
                    cache_updated = true;
                }
                None => {
                    crate::utils::score_dynamics::record_baseline("commerce.thead_dom_first", 0.0);
                }
            }
        }

        if final_thead_selector.is_empty() {

            let reference_row_for_thead = {
                let clean_content = &clean_html_content;
                let document = scraper::Html::parse_document(clean_content);
                if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                    document.select(&sel).next().map(|first_match| {
                        let mut temp_pug = String::new();
                        crate::parsing::generate_pug_lines((*first_match).into(), 0, &mut temp_pug, &PugMode::FullContent, &mut None);
                        temp_pug.trim().to_string()
                    })
                } else { None }
            };

            if let Some(ref_row) = reference_row_for_thead {
                if !ref_row.is_empty() {
                    log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Analyzing table header structure...", "spinner": "⠋" }));
                    
                    
                    let ref_row_context_size = ref_row.len() + 2000;
                    let full_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
                    let thead_light_pug = model.truncate_pug_context(&full_pug, false, 0, Some(ref_row_context_size)).await;

                    println!("ref_row: {}", ref_row);
                    
                    let thead_prompt = crate::parsing::extract_table_structure_prompt(&page_type, &target_selector, &thead_light_pug, &ref_row);
                    let params = ChatCompletionParameters {
                        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(thead_prompt),
                            name: None,
                        })],
                        model: "qwen3.5".to_string(),
                        max_tokens: Some(256), 
                        temperature: Some(0.0), 
                        top_p: Some(0.95),
                        ..Default::default()
                    };

                    // 🌟 [CROSSOVER] Qwen3.5(2B)는 임베딩과 동시 상주가 어려운 크기입니다.
                    //    예산 판정이 SWAP 으로 떨어지면 임베딩을 먼저 반환합니다.
                    model
                        .switch_to_generation(
                            crate::model::ModelSize::Qwen3_5,
                            Some(cancellation_token.clone()),
                            kv_name.clone(),
                            "thead structure analysis",
                        )
                        .await?;
                    if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                        if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_step_thead", task.id)), kv_name.clone(), None, None).await {
                            let thead_json = crate::parsing::parse_json_from_llm(&res);
                            
                            fn harvest_selector_strings(v: &serde_json::Value, out: &mut Vec<String>) {
                                match v {
                                    serde_json::Value::String(s) => {
                                        let t = s.trim();
                                        if t.is_empty() || t == "..." { return; }
                                        if t.chars().count() > 200 { return; }
                                        if !t.chars().any(|c| c.is_ascii_alphanumeric()) { return; }
                                        let cleaned = t.replace('>', " ");
                                        if !out.iter().any(|e| e == &cleaned) { out.push(cleaned); }
                                    }
                                    serde_json::Value::Array(a) => {
                                        for x in a { harvest_selector_strings(x, out); }
                                    }
                                    serde_json::Value::Object(o) => {
                                        for (_, x) in o { harvest_selector_strings(x, out); }
                                    }
                                    _ => {}
                                }
                            }
                            let mut cand_sels: Vec<String> = Vec::new();
                            harvest_selector_strings(&thead_json, &mut cand_sels);

                            let (verified_thead, verified_table) = {
                                
                                
                                let doc = scraper::Html::parse_document(&clean_html_content);
                                let th_q = scraper::Selector::parse("th").ok();
                                let td_q = scraper::Selector::parse("td").ok();
                                let mut best_thead = String::new();
                                let mut best_thead_th = 0usize;
                                let mut best_table = String::new();
                                for s in &cand_sels {
                                    let parsed = match scraper::Selector::parse(s) {
                                        Ok(p) => p,
                                        Err(_) => continue,
                                    };
                                    let el = match doc.select(&parsed).next() {
                                        Some(e) => e,
                                        None => continue,
                                    };
                                    let tag = el.value().name().to_lowercase();
                                    if tag == "table" {
                                        if best_table.is_empty() { best_table = s.clone(); }
                                        continue;
                                    }
                                    let th_cnt = th_q.as_ref().map(|q| el.select(q).count()).unwrap_or(0);
                                    let td_cnt = td_q.as_ref().map(|q| el.select(q).count()).unwrap_or(0);
                                    if th_cnt >= 2 && th_cnt > td_cnt && th_cnt > best_thead_th {
                                        best_thead_th = th_cnt;
                                        best_thead = s.clone();
                                    }
                                }
                                (best_thead, best_table)
                            };
                            emit_term(&format!(
                                "  🧾 [SELECTOR HARVEST] 후보 {}개 수집 → DOM 검증 결과 thead='{}' (th {}개) | table='{}'",
                                cand_sels.len(), verified_thead, best_thead_th_dbg(&verified_thead, &clean_html_content), verified_table
                            ));
                            final_thead_selector = verified_thead;
                            let final_table_selector = verified_table;

                            
                            if !final_thead_selector.is_empty() && final_thead_selector != "..." && !final_table_selector.is_empty() && final_table_selector != "..." {
                                if !final_thead_selector.contains(&final_table_selector) {
                                    let combined_sel = format!("{} {}", final_table_selector, final_thead_selector);
                                    let doc = scraper::Html::parse_document(&clean_html_content);
                                    

                                    let is_valid = scraper::Selector::parse(&combined_sel)
                                        .map(|parsed_sel| doc.select(&parsed_sel).next().is_some())
                                        .unwrap_or(false);

                                    if is_valid {
                                        final_thead_selector = combined_sel;
                                    }
                                }
                            }

                            if !final_thead_selector.is_empty() && final_thead_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("head".to_string(), json!(final_thead_selector.clone()));
                                println!("[Scheduler] AI determined head selector and cached: {}", final_thead_selector);
                                cache_updated = true;
                            }

                            
                            if !final_table_selector.is_empty() && !final_table_selector.contains("CSS selector") && final_table_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("wrapper".to_string(), json!(final_table_selector.clone()));
                                println!("[Scheduler] AI determined table wrapper selector and cached: {}", final_table_selector);
                                cache_updated = true;
                            }
                        }
                    }
                }
            }
        }
        
        if final_thead_selector.is_empty() || final_thead_selector == "..." {
            final_thead_selector.clear();
            emit_term("  ⚠️ [DETERMINISTIC THEAD] 목록 첫 행의 조상 표에서도, LLM 셀렉터 수확에서도 헤더 행(th 2개 이상 · th 가 td 보다 많음)을 찾지 못했습니다. 헤더 없이 진행합니다.");
        }
        model.deep_purge_resources().await;
        
        let mut trade_headers: Vec<Vec<String>> = Vec::new();
        let mut row_contract = String::new();

        if !final_thead_selector.is_empty() && final_thead_selector != "..." {
            
            
            let (sync_grid, sync_pending) = {
                let clean_content = &clean_html_content;
                let doc = scraper::Html::parse_document(clean_content);
                if let Ok(tsel) = scraper::Selector::parse(&final_thead_selector) {
                    if let Some(first_match) = doc.select(&tsel).next() {
                        let mut target_node = first_match;
                        let mut current = target_node.parent();
                        while let Some(parent) = current {
                            if let Some(el) = parent.value().as_element() {
                                let tag = el.name().to_lowercase();
                                if tag == "thead" || tag == "tr" {
                                    if let Some(wrapped) = scraper::ElementRef::wrap(parent) {
                                        target_node = wrapped;
                                        if tag == "thead" { break; }
                                    }
                                }
                            }
                            current = parent.parent();
                        }
                        let mut tpug = String::new();
                        crate::parsing::generate_pug_lines((*target_node).into(), 0, &mut tpug, &PugMode::TheadMode, &mut None);
                        thead_pug = tpug.trim().to_string();
                        if !thead_pug.is_empty() {
                            println!("[Scheduler] 🎉 thead_pug extraction successful ({} bytes)", thead_pug.len());
                        }
                    }
                }
                
                crate::parsing::extract_doc_table_headers_sync(
                    &doc, &final_thead_selector, &doc_lang
                )
            }; 

            
            trade_headers = crate::parsing::extract_doc_table_headers_async(
                sync_grid, sync_pending, &doc_lang, &model
            ).await;
            row_contract = crate::parsing::build_table_row_contract(&trade_headers, &doc_lang);
            if !trade_headers.is_empty() {
                emit_term(&format!(
                    "  📐 [HEADER GRID] {}행 x {}열 헤더 격자 확보 | row_contract {}바이트",
                    trade_headers.len(),
                    trade_headers.first().map(|r| r.len()).unwrap_or(0),
                    row_contract.len()
                ));
            }
        }

        if !skip_ai_analysis || cache_updated {
            let store = {
                let store_guard = store_mutex.lock().await;
                store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
            };
            
            let mut shared_origin = None;
            let mut shared_type = None;
            if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
                if let Some(json_val) = mem.as_ref() {
                    if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                        if let Ok(u) = url::Url::parse(o) {
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost")));
                        }
                    }
                    if let Some(t) = json_val.get("type").and_then(|v| v.as_str()) {
                        if !t.is_empty() { shared_type = Some(t.to_string()); }
                    }
                }
            }

            let origin_str = task_data.get("origin")
                .or_else(|| task_data.get("domain"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string())
                .filter(|s| !s.contains("localhost")) 
                .or(shared_origin) 
                .unwrap_or_else(|| {
                    if let Ok(task_url) = url::Url::parse(&url) {
                        format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost"))
                    } else {
                        "http://localhost".to_string()
                    }
                });

            if page_type.is_empty() || page_type == "unknown" {
                if let Some(st) = shared_type { page_type = st; }
            }
                
            let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
            let url_obj = base_url.join(&url).unwrap_or(base_url);
            let raw_path = url_obj.path();
            let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path)); 
            
            let bcc = entity_bcc(&page_type, &task.cc);
            let ref_for_page = if !task.r#ref.is_empty() { &task.r#ref } else { raw_path };

            
            if census_selector_used {
                emit_term("  🧮 [CENSUS SELECTOR / NO CACHE] 이번 목록 셀렉터는 'LLM 제목 → Boa 검증' 이 아니라 DOM 전수 조사에서 왔습니다. 페이지 캐시에 저장하지 않으므로, 다음 실행은 처음부터 다시 분석해 Boa 가 확인한 셀렉터만 캐시에 남습니다.");
            } else if !is_detail {
                let mut page_data: serde_json::Value = selector_info.clone();
                if let Some(obj) = page_data.as_object_mut() {
                    obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                    obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                    obj.insert("type".to_string(), json!(page_type.clone()));
                    
                    if let Some(item_sel) = selector_info.get("itemSelector") { obj.insert("item".to_string(), item_sel.clone()); }
                    if let Some(parent_sel) = selector_info.get("parent") { obj.insert("node".to_string(), parent_sel.clone()); }
                    obj.insert("detail".to_string(), json!(false));
                }

                save_item(&store, "pages", &page_id, &page_type, page_data, None,
                    &task.from, &team_id, &task.cc, &bcc, ref_for_page, None).await;

                println!("[Scheduler] Page cache updated in DB (including head selector).");

                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = entity_bcc(&page_type, &task.cc);
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    
                    "detail": 1,
                    "node": 1,
                    "item": ""
                });
                save_item(&store, "pages", &detail_page_id, &page_type, detail_page_data, None,
                    &task.from, &team_id, &task.cc, &detail_bcc, ref_for_page, None).await;

            } else {
                
                
                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = entity_bcc(&page_type, &task.cc);
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    "detail": 1,
                    "node": 1,
                    "item": ""
                });
                save_item(&store, "pages", &detail_page_id, &page_type, detail_page_data, None,
                    &task.from, &team_id, &task.cc, &detail_bcc, ref_for_page, None).await;
            }
        }
        

        let list_log = json!({ "category": "List Processing", "summary": "Extracting list data with LLM...", "spinner": "⠋" });
        log_task_progress(app_handle, &task.id, &list_log);

        let mut all_extracted_items = Vec::new();
        
        let mut pug_list = {
            let clean_content = &clean_html_content;
            let document = scraper::Html::parse_document(clean_content);

            let list_headers = if trade_headers.is_empty() {
                None
            } else {
                emit_term(&format!(
                    "  🏷️ [HEADER GRID → PUG] {}행 x {}열 격자를 목록 분해에 주입합니다. (alt 라벨 + field 속성, 값 구분자 '|' 는 속성에 싣지 않음)",
                    trade_headers.len(),
                    trade_headers.first().map(|r| r.len()).unwrap_or(0)
                ));
                Some(trade_headers.clone())
            };
            parsing::split_doc_to_pug_list_advanced(
                &document, 
                &target_selector, 
                PugMode::ListMode, 
                list_headers,
                Some(&url) 
            )
        };


        let mut group_size = if !thead_pug.is_empty() {
            let mut max_span = 1;

            if let Ok(re) = regex::Regex::new(r#"rowspan="(\d+)""#) {
                for cap in re.captures_iter(&thead_pug) {
                    if let Ok(val) = cap[1].parse::<usize>() {
                        if val > max_span {
                            max_span = val;
                        }
                    }
                }
            }
            
            if max_span > 1 {
                max_span
            } else {
                thead_pug.lines().filter(|line| {
                    let s = line.trim_start();
                    s == "tr" || s.starts_with("tr[")
                }).count().max(1)
            }
        } else {
            1
        };

        if group_size > 1 && !pug_list.is_empty() {

            let first_item_tr_count = pug_list.first()
                .map(|p| p.lines().filter(|l| {
                    let indent = l.chars().take_while(|c| c.is_whitespace()).count();
                    indent == 0 && (l.starts_with("tr") || l.starts_with("tr["))
                }).count())
                .unwrap_or(1);


            if first_item_tr_count >= group_size || first_item_tr_count > 1 {
                println!("[Scheduler] 🌟 Items are already grouped ({} trs per item). Skipping manual chunking.", first_item_tr_count);
                group_size = 1;
            } else {
                let mut grouped = Vec::new();
                for chunk in pug_list.chunks(group_size) {
                    grouped.push(chunk.join("\n"));
                }
                pug_list = grouped;
                println!("[Scheduler] 🌟 Grouped multi-row items: {} rows per item. Total items reduced to {}.", group_size, pug_list.len());
            }
        }

        if !pug_list.is_empty() {
            let total_items = pug_list.len();
            let mut text_frequency: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            
            
            
            
            let mut subordinate_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            
            
            let mut dead_action_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut control_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut plain_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut cell_value_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            for item_pug in &pug_list {
                let mut seen_in_this_item = std::collections::HashSet::new();
                let mut seen_control = std::collections::HashSet::new();
                let mut seen_plain = std::collections::HashSet::new();
                for line in item_pug.lines() {
                    if let Some(idx) = line.find('|') {
                        let text_part = line[idx + 1..].trim();
                        if !text_part.is_empty() && text_part.len() > 2 {
                            seen_in_this_item.insert(text_part.to_string());
                            if pug_line_is_ui_control(line) {
                                seen_control.insert(text_part.to_string());
                            } else {
                                seen_plain.insert(text_part.to_string());
                            }
                        }
                    }
                }
                for text in seen_in_this_item {
                    *text_frequency.entry(text).or_insert(0) += 1;
                }
                for text in seen_control {
                    *control_texts.entry(text).or_insert(0) += 1;
                }
                for text in seen_plain {
                    *plain_texts.entry(text).or_insert(0) += 1;
                }

                let cell_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                let mut seen_sub = std::collections::HashSet::new();
                let mut seen_dead = std::collections::HashSet::new();
                let mut seen_cell_value = std::collections::HashSet::new();

                for cell in parse_pug_grid(&cell_lines) {
                    if let Some(t) = pug_cell_sole_value(&cell_lines, &cell.line_indices) {
                        if t.len() > 2 { seen_cell_value.insert(t); }
                    }
                    let has_real_link = cell.line_indices.iter()
                        .any(|&li| line_real_href(&cell_lines[li]).is_some());
                    if !has_real_link { continue; }
                    for &li in &cell.line_indices {
                        if line_real_href(&cell_lines[li]).is_some() { continue; }
                        if let Some(p) = cell_lines[li].find('|') {
                            let t = cell_lines[li][p + 1..].trim();
                            if t.len() > 2 { seen_sub.insert(t.to_string()); }
                        }
                    }
                }

                for line in &cell_lines {
                    if !line.contains("href=") { continue; }
                    if line_real_href(line).is_some() { continue; }
                    if let Some(p) = line.find('|') {
                        let t = line[p + 1..].trim();
                        if t.len() > 2 { seen_dead.insert(t.to_string()); }
                    }
                }

                for t in seen_sub { *subordinate_texts.entry(t).or_insert(0) += 1; }
                for t in seen_dead { *dead_action_texts.entry(t).or_insert(0) += 1; }
                for t in seen_cell_value { *cell_value_texts.entry(t).or_insert(0) += 1; }
            }

            let mut boilerplate_texts = std::collections::HashSet::new();
            let mut control_boilerplate: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut hard_drop_texts: std::collections::HashSet<String> = std::collections::HashSet::new();

            let fields = parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            let enum_guard_embs: Vec<Vec<f32>> = {
                let mut phrases: Vec<String> = Vec::new();
                for (fname, _, _, _) in fields.iter() {
                    if detect_field_format(fname) != FieldFormat::Enum { continue; }
                    let (mut label_phrases, _) = label_phrase_bank(&doc_lang, &page_type, fname);
                    if label_phrases.is_empty() {
                        label_phrases = label_phrase_bank_multilingual(&doc_lang, &page_type, fname).0;
                    }
                    for p in label_phrases {
                        if !phrases.iter().any(|e| e == &p) { phrases.push(p); }
                    }
                }
                if phrases.is_empty() {
                    Vec::new()
                } else {
                    model.get_embedding_batch(phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                }
            };
            
            
            
            
            let ui_action_embs: Vec<Vec<f32>> = {
                let phrases: Vec<String> = crate::logic::anchor_phrases(
                    crate::logic::UI_ACTION_ANCHOR,
                    crate::logic::UI_ACTION_ANCHOR_ML,
                );
                if phrases.is_empty() {
                    Vec::new()
                } else {
                    model.get_embedding_batch(phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                }
            };
            emit_term(&format!(
                "  🧱 [UI ACTION BANK] enum 보호 구 {}개 | UI 액션 편견 구 {}개 준비 완료.",
                enum_guard_embs.len(), ui_action_embs.len()
            ));

            if total_items >= 2 {
                let threshold = (total_items as f32 * 0.7).ceil() as usize; 
                
                let re_numeric = regex::Regex::new(r"^\D*\d+[\d,\.]*\D*$").unwrap();

                for (text, count) in text_frequency {
                    if count >= threshold {

                        let is_numeric_data = re_numeric.is_match(&text);
                        if !is_numeric_data && text.len() > 3 && has_date_shape(&text) {
                            crate::utils::score_dynamics::record_baseline("commerce.repeat_date_protect", 1.0);
                            emit_term(&format!("[Scheduler] 🛡️ [DATE SHAPE PROTECT] 반복되지만 날짜 모양이라 UI 문구가 될 수 없어 실데이터로 보호: '{}' ({} / {} 아이템)", text, count, total_items));
                            continue;
                        }
                        if !is_numeric_data && text.len() > 3 {
                            let sub_hits = subordinate_texts.get(&text).copied().unwrap_or(0);
                            let dead_hits = dead_action_texts.get(&text).copied().unwrap_or(0);

                            let mut enum_sim = 0.0f32;
                            let mut chrome_sim = 0.0f32;
                            let mut sim_pool: Vec<f32> = Vec::new();
                            if !enum_guard_embs.is_empty() || !ui_action_embs.is_empty() {
                                let t_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                                for ge in &enum_guard_embs {
                                    let s = cosine_similarity(ge, &t_emb);
                                    sim_pool.push(s);
                                    if s > enum_sim { enum_sim = s; }
                                }
                                for ce in &ui_action_embs {
                                    let s = cosine_similarity(ce, &t_emb);
                                    sim_pool.push(s);
                                    if s > chrome_sim { chrome_sim = s; }
                                }
                            }
                            let exact_chrome = chrome_sim >= 0.999;
                            let exact_enum = enum_sim >= 0.999;
                            let exact_action = exact_chrome && !exact_enum;
                            let bank_shift = if !exact_chrome && sim_pool.len() >= 2 && !enum_guard_embs.is_empty() && !ui_action_embs.is_empty() {
                                let n = sim_pool.len() as f32;
                                let mean = sim_pool.iter().sum::<f32>() / n;
                                let sd = (sim_pool.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n).sqrt().max(1e-6);
                                sd * (gumbel_expected_z(ui_action_embs.len()) - gumbel_expected_z(enum_guard_embs.len()))
                            } else {
                                0.0
                            };
                            let control_hits = control_texts.get(&text).copied().unwrap_or(0);
                            let plain_hits = plain_texts.get(&text).copied().unwrap_or(0);
                            let semantic_data = if exact_chrome {
                                exact_enum && control_hits < threshold
                            } else {
                                enum_sim + bank_shift >= chrome_sim
                            };
                            if sub_hits >= threshold || dead_hits >= threshold {
                                boilerplate_texts.insert(text.clone());
                                let hard = dead_hits >= threshold && !semantic_data;
                                if hard {
                                    hard_drop_texts.insert(text.clone());
                                }
                                emit_term(&format!("[Scheduler] 🚫 [ACTION LINE DROP] 구조적으로 UI 액션/종속 라인 확정 탈락: '{}' ({} / {} 아이템 | Subordinate: {} | DeadHref: {}{})", text, count, total_items, sub_hits, dead_hits, if hard { " | UI 액션 쪽 죽은 링크라 헤더 소유 칸도 보호하지 않음" } else { "" }));
                                continue;
                            }
                            let control_drop = if exact_chrome {
                                control_hits > 0
                            } else {
                                control_hits >= threshold && !semantic_data
                            };
                            if control_drop {
                                control_boilerplate.insert(text.clone());
                                if exact_action {
                                    hard_drop_texts.insert(text.clone());
                                }
                                crate::utils::score_dynamics::record_baseline("commerce.ui_control_drop", 1.0);
                                emit_term(&format!("[Scheduler] 🚫 [UI CONTROL DROP] '{}' 는 버튼/입력 컨트롤로 {} / {} 아이템에 나오고 UI 액션 쪽입니다 (EnumSim {:.4} + 보정 {:+.4} vs ChromeSim {:.4}). 컨트롤 라인을 탈락 대상으로 둡니다. 컨트롤 밖 같은 글자는 {} 아이템 → {}", text, control_hits, total_items, enum_sim, bank_shift, chrome_sim, plain_hits, if exact_action { "UI 액션 글자라 아래 판정을 계속합니다." } else if plain_hits < threshold { "반복 기준 미달이라 그대로 둡니다." } else { "아래 칸·의미 판정을 계속합니다." }));
                                if plain_hits < threshold && !exact_action {
                                    continue;
                                }
                            }
                            let cell_hits = cell_value_texts.get(&text).copied().unwrap_or(0);
                            if cell_hits >= threshold && !exact_action {
                                crate::utils::score_dynamics::record_baseline("commerce.cell_value_margin", enum_sim - chrome_sim);
                                emit_term(&format!("[Scheduler] 🛡️ [CELL VALUE PROTECT] 반복되지만 {} / {} 아이템에서 링크·입력 없는 표 칸의 유일한 글자(버튼·숨김 입력은 제외하고 셈)라 컬럼 상수값으로 보호: '{}' (참고: EnumSim {:.4} + 보정 {:+.4} vs ChromeSim {:.4})", cell_hits, total_items, text, enum_sim, bank_shift, chrome_sim));
                                continue;
                            }
                            crate::utils::score_dynamics::record_baseline("commerce.ui_action_margin", enum_sim - chrome_sim);
                            crate::utils::score_dynamics::record_baseline("commerce.ui_action_bank_shift", bank_shift);
                            if semantic_data {
                                emit_term(&format!("[Scheduler] 🛡️ [ENUM VECTOR PROTECT] 반복되지만 열거형 값 구 유사도({:.4}) + 은행 크기 보정({:+.4}) >= UI 액션 유사도({:.4}) 이므로 실데이터로 보호: '{}' ({} / {} 아이템)", enum_sim, bank_shift, chrome_sim, text, count, total_items));
                                continue;
                            }
                            boilerplate_texts.insert(text.clone());
                            if exact_action {
                                hard_drop_texts.insert(text.clone());
                            }
                            emit_term(&format!("[Scheduler] 🚫 [UI ACTION DROP] 전역 중복 텍스트 탈락: '{}' ({} / {} 아이템 | EnumSim: {:.4} + 보정 {:+.4} vs ChromeSim: {:.4}{})", text, count, total_items, enum_sim, bank_shift, chrome_sim, if exact_action { " | UI 액션 구와 글자 그대로 일치 → 열거형 컬럼의 단독 값일 때만 헤더 소유 보호" } else if exact_chrome { " | 열거형·UI 액션 양쪽과 글자 그대로 일치 → 헤더 소유 칸만 보호" } else { " | UI 액션 쪽" }));
                        }
                    }
                }
            }


            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                };
                
                if t_val.is_empty() || t_val.len() < 5 {
                    let mut heading_texts = Vec::new();
                    if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                        for el in doc.select(&sel_h1) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                        for el in doc.select(&sel_h2) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if !heading_texts.is_empty() {
                        t_val = heading_texts.join(" | ");
                    }
                }
                t_val
            };

            // 🌟 [CROSSOVER / ORDER FIX] Qwen3 로드를 여기서 제거합니다.
            //
            //  ── 무엇이 문제였나 ──
            //   이 자리에서 Qwen3(0.6B)를 올려 두고, 그 뒤로
            //     field_embeddings 루프 → header_embs → label bank → idlink bank
            //     → 아이템 루프의 라인 임베딩
            //   까지 임베딩만 수백 회 돌렸습니다.
            //   Qwen3 의 첫 실사용은 아이템 루프 안의 generate 이므로,
            //   그 전 구간 내내 두 가중치가 아무 이유 없이 겹쳐 있었습니다.
            //
            //  ── 해결 ──
            //   여기서는 임베딩 페이즈만 선언하고,
            //   Qwen3 는 아이템 루프 진입 직전에 올립니다(아래 참조).
            //   코드 이동만으로 이 구간의 피크가 통째로 사라집니다.
            model.enter_embedding_phase("list schema field bank").await?;
            let mut field_embeddings = Vec::new();
            
            
            let mut field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_phrase_weights: Vec<Vec<f32>> = Vec::new();
            
            
            let mut field_is_analytic: Vec<bool> = Vec::new();
            
            
            
            let mut field_formats: Vec<FieldFormat> = Vec::new();

            for (f_idx, (fname, _, bias_target, predefined_prej)) in fields.iter().enumerate() {
                let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);

                let (phrases, phrase_weights) = split_bias_phrases_weighted(bias_target);
                let p_embs = if phrases.is_empty() {
                    vec![bias_emb.clone()]
                } else {
                    model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![bias_emb.clone(); phrases.len()])
                };
                let p_weights = if phrases.is_empty() { vec![1.0f32] } else { phrase_weights };
                field_phrase_embs.push(p_embs);
                field_phrase_weights.push(p_weights);

                let detected_fmt = detect_field_format(fname);
                field_formats.push(detected_fmt);
                emit_term(&format!("  📐 [FORMAT REGISTERED] '{}' → {:?}", fname, detected_fmt));

                let lower_fname = fname.to_lowercase();
                let is_analytic = lower_fname.contains("insight")
                    || lower_fname.contains("summary")
                    || lower_fname.contains("analysis");
                field_is_analytic.push(is_analytic);
                if is_analytic {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD REGISTERED] '{}' 는 벡터 라인 매칭에서 제외되고 전체 컨텍스트 요약 필드로 처리됩니다.", fname));
                }

                let mut dynamic_prej_texts = Vec::new();
                if !predefined_prej.trim().is_empty() {
                    dynamic_prej_texts.push(predefined_prej.clone());
                }
                for (other_idx, (_, _, other_bias, _)) in fields.iter().enumerate() {
                    if f_idx != other_idx {
                        dynamic_prej_texts.push(other_bias.clone());
                    }
                }
                let combined_prej = dynamic_prej_texts.join(" , ");
                let prej_emb = model.get_embedding(combined_prej.clone()).await.unwrap_or(vec![0.0; 384]);

                field_embeddings.push((bias_emb, prej_emb, combined_prej));
            }


            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            let mut thead_lines: Vec<String> = thead_pug.lines().map(|s| s.to_string()).collect();
            let mut thead_embeddings = vec![vec![0.0; 384]; thead_lines.len()];
            

            let header_grid = HeaderGrid::new(parse_pug_grid(&thead_lines));

            if !thead_lines.is_empty() {
                emit_term(&format!("\n[PRE-PROCESSING] Vectorizing Table Header ({} lines)...", thead_lines.len()));
                
                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in thead_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        texts_to_embed.push(line.to_string());
                        text_indices.push(line_idx);
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                        let start_idx = chunk_idx * 100;
                        if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                            for (i, vector) in vectors.into_iter().enumerate() {
                                let original_idx = text_indices[start_idx + i];
                                let emb = vector.clone();
                                let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                                
                                let original_text = text_chunk[i].trim();
                                let has_digit = original_text.chars().any(|c| c.is_ascii_digit());
                                let is_short = original_text.len() <= 3;
                                

                                let is_structure_tag = original_text.starts_with("th") 
                                    || original_text.starts_with("td") 
                                    || original_text.starts_with("tr") 
                                    || original_text.starts_with("input")
                                    || original_text.starts_with("div");
                                

                                if noise_score > 0.6 && !has_digit && !is_short && !is_structure_tag {
                                    emit_term(&format!("    🚫 [NOISE FILTERED] Header Line {} : {} (Score: {:.4})", original_idx + 1, original_text, noise_score));
                                    thead_lines[original_idx] = String::new(); 
                                } else {
                                    thead_embeddings[original_idx] = emb;
                                }
                            }
                        }
                    }
                }
            }


            let mut unique_headers: Vec<String> = Vec::new();
            let mut item_grids: Vec<Vec<GridCell>> = Vec::with_capacity(pug_list.len());
            let mut interleaved_items = 0usize;
            for item_pug in pug_list.iter() {
                let item_cell_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                let cells = parse_pug_grid(&item_cell_lines);
                if header_grid.interleaved_with(&cells) {
                    interleaved_items += 1;
                }
                item_grids.push(cells);
            }
            let page_interleaved = interleaved_items > 0 && interleaved_items * 2 > pug_list.len();
            for cells in item_grids.iter() {
                for cell in cells.iter() {
                    let label = header_grid.cell_label(cell.row, cell.col, cell.colspan, cell.rowspan, page_interleaved);
                    let clean_h = label.trim();
                    if !clean_h.is_empty() && !unique_headers.iter().any(|h| h == clean_h) {
                        unique_headers.push(clean_h.to_string());
                    }
                }
            }
            if unique_headers.is_empty() {
                for col in 0..header_grid.column_count() {
                    let label = header_grid.column_label(col);
                    let clean_h = label.trim();
                    if !clean_h.is_empty() && !unique_headers.iter().any(|h| h == clean_h) {
                        unique_headers.push(clean_h.to_string());
                    }
                }
            }
            if page_interleaved {
                emit_term(&format!(
                    "  🧩 [HEADER GRID LABEL] 헤더 {}행과 본문 칸 배치가 행마다 겹치는 교차 배치 아이템 {}/{}개 (과반) → 좌표가 정확히 겹치는 헤더 칸은 그 칸 라벨을, 나머지 칸은 조인 라벨을 씁니다. | 라벨 {}개: {:?}",
                    header_grid.rows, interleaved_items, pug_list.len(), unique_headers.len(), unique_headers
                ));
            }
            crate::utils::score_dynamics::record_baseline(
                "commerce.header_interleaved",
                if page_interleaved { 1.0 } else { 0.0 },
            );

            let mut header_to_field_map = std::collections::HashMap::new();
            let mut header_evaluated: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut header_plausible: std::collections::HashMap<String, std::collections::HashSet<String>> =
                std::collections::HashMap::new();
            let list_field_vocab: Vec<Vec<String>> = fields
                .iter()
                .map(|(n, _, _, _)| crate::utils::ai_utils::field_value_vocabulary(&doc_lang, &page_type, n))
                .collect();
            let list_status_pivot = if fields.iter().any(|(n, _, _, _)| n == "status") {
                crate::utils::ai_utils::status_pivot_bank(&model, &page_type, &doc_lang).await
            } else {
                None
            };
            if let Some(p) = list_status_pivot.as_ref() {
                crate::utils::score_dynamics::record_baseline("commerce.status_pivot_align_z", p.align_z);
                emit_term(&format!(
                    "  🌉 [STATUS PIVOT BANK] '{}' 상태 목록 {}개 ↔ 'en' 같은 자리 | 정렬 z {:+.2} (기준 {:.1}) → {}",
                    p.lang,
                    p.src.len(),
                    p.align_z,
                    crate::utils::ai_utils::STATUS_PIVOT_ALIGN_Z,
                    if p.usable() {
                        "같은 자리끼리 번역 쌍으로 확인되어, 상태 원문을 영어 캐노니컬 어휘로 옮기는 데 씁니다"
                    } else {
                        "자리 대응이 확인되지 않아 쓰지 않습니다"
                    }
                ));
            }

            if !unique_headers.is_empty() {
                let header_embs: Vec<Vec<f32>> = model
                    .get_embedding_batch(unique_headers.clone())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; unique_headers.len()]);
                let header_terminal: Vec<String> = unique_headers
                    .iter()
                    .map(|h| h.rsplit(" > ").next().unwrap_or("").trim().to_string())
                    .collect();
                let header_leaf_texts: Vec<Option<String>> = unique_headers
                    .iter()
                    .zip(header_terminal.iter())
                    .map(|(h, leaf)| {
                        if leaf.is_empty() || leaf == h.trim() { return None; }
                        let shared = header_terminal.iter().filter(|t| *t == leaf).count();
                        if shared == 1 { Some(leaf.clone()) } else { None }
                    })
                    .collect();
                let leaf_batch: Vec<String> = header_leaf_texts.iter().filter_map(|x| x.clone()).collect();
                let leaf_batch_embs: Vec<Vec<f32>> = if leaf_batch.is_empty() {
                    Vec::new()
                } else {
                    model.get_embedding_batch(leaf_batch.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; leaf_batch.len()])
                };
                let mut header_leaf_embs: Vec<Option<Vec<f32>>> = Vec::with_capacity(unique_headers.len());
                {
                    let mut k = 0usize;
                    for lt in header_leaf_texts.iter() {
                        if lt.is_some() {
                            header_leaf_embs.push(leaf_batch_embs.get(k).cloned());
                            k += 1;
                        } else {
                            header_leaf_embs.push(None);
                        }
                    }
                }

                let mut hdr_field_names: Vec<String> = Vec::new();
                let mut hdr_label_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut hdr_label_weights: Vec<Vec<f32>> = Vec::new();
                let mut hdr_prej_embs: Vec<Vec<Vec<f32>>> = Vec::new();

                for (fname, _, _, _) in &fields {
                    let (label_phrases, label_weights) = label_phrase_bank(&doc_lang, &page_type, fname);
                    if label_phrases.is_empty() { continue; }
                    let prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, fname);

                    let l_embs = model.get_embedding_batch(label_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; label_phrases.len()]);
                    let p_embs = if prej_phrases.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(prej_phrases.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
                    };

                    emit_term(&format!("  🏷️ [LABEL BANK] '{}' | 라벨 구 {}개 | 편견 구 {}개", fname, label_phrases.len(), p_embs.len()));
                    hdr_field_names.push(fname.clone());
                    hdr_label_embs.push(l_embs);
                    hdr_label_weights.push(label_weights);
                    hdr_prej_embs.push(p_embs);
                }

                let hdr_abs_floor = 0.62f32;
                let hdr_score_floor = 0.10f32;
                let hdr_margin = 0.03f32;

                let mut hdr_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_headers.len()]; hdr_field_names.len()];
                let mut hdr_leaf_won: Vec<Vec<bool>> = vec![vec![false; unique_headers.len()]; hdr_field_names.len()];
                for f in 0..hdr_field_names.len() {
                    for h in 0..unique_headers.len() {
                        if header_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        crate::utils::score_dynamics::record_field_seen(&hdr_field_names[f]);
                        let score_of = |emb: &Vec<f32>| -> (f32, f32, f32) {
                            let own = weighted_max_pool_sim(emb, &hdr_label_embs[f], &hdr_label_weights[f]);
                            let prej = if hdr_prej_embs[f].is_empty() { 0.0 } else { max_pool_sim(emb, &hdr_prej_embs[f]) };
                            (own, prej, own - prej)
                        };
                        let joined = score_of(&header_embs[h]);
                        let joined_ok = joined.0 >= hdr_abs_floor && joined.2 >= hdr_score_floor;
                        let mut chosen = joined;
                        let mut is_leaf = false;
                        if let Some(Some(le)) = header_leaf_embs.get(h) {
                            if !le.iter().all(|&v| v == 0.0) {
                                let leaf = score_of(le);
                                let leaf_ok = leaf.0 >= hdr_abs_floor && leaf.2 >= hdr_score_floor;
                                if leaf_ok && (!joined_ok || leaf.2 > joined.2) {
                                    chosen = leaf;
                                    is_leaf = true;
                                }
                            }
                        }
                        let (own, prej, score) = chosen;
                        if own < hdr_abs_floor {
                            crate::utils::score_dynamics::record_near_miss(&hdr_field_names[f]);
                            continue;
                        }
                        if score < hdr_score_floor {
                            emit_term(&format!("    🚫 [HEADER PREJUDICE DROP] '{}' → '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4} < {:.2}", unique_headers[h], hdr_field_names[f], own, prej, score, hdr_score_floor));
                            crate::utils::score_dynamics::record_field_reject(
                                &hdr_field_names[f],
                                crate::utils::score_dynamics::GateKind::Prejudice,
                            );
                            continue;
                        }
                        if is_leaf {
                            if joined_ok {
                                crate::utils::score_dynamics::record_baseline("commerce.header_leaf_evidence", 1.0);
                                emit_term(&format!("    🌿 [HEADER LEAF EVIDENCE] '{}' → '{}' | 조인 라벨 Score {:+.4} 보다 헤더 전체에서 한 번만 나오는 하위 라벨 '{}' 의 Score {:+.4} 가 높아 하위 라벨을 근거로 씁니다. 상위 그룹명은 형제 칼럼 전부에 같이 붙어 조인 라벨끼리의 마진을 깎으므로, 조인 라벨이 하한을 겨우 넘긴 칼럼은 배타 배정의 마진에서 떨어집니다.", unique_headers[h], hdr_field_names[f], joined.2, header_terminal[h], score));
                            } else {
                                emit_term(&format!("    🌿 [HEADER LEAF RESCUE] '{}' → '{}' | 조인 라벨 Score {:+.4} 가 하한 미달이라, 헤더 전체에서 한 번만 나오는 하위 라벨 '{}' 로 다시 쟀습니다 → Score {:+.4}", unique_headers[h], hdr_field_names[f], joined.2, header_terminal[h], score));
                            }
                        }
                        hdr_matrix[f][h] = score;
                        hdr_leaf_won[f][h] = is_leaf;
                    }
                }

                for h in 0..unique_headers.len() {
                    if !header_embs[h].iter().all(|&v| v == 0.0) {
                        header_evaluated.insert(unique_headers[h].clone());
                    }
                }
                for f in 0..hdr_field_names.len() {
                    if hdr_label_embs[f].iter().all(|e| e.iter().all(|&v| v == 0.0)) {
                        continue;
                    }
                    let set: std::collections::HashSet<String> = (0..unique_headers.len())
                        .filter(|&h| hdr_matrix[f][h] >= hdr_score_floor)
                        .map(|h| unique_headers[h].clone())
                        .collect();
                    header_plausible.insert(hdr_field_names[f].clone(), set);
                }
                let hdr_assign = exclusive_assign(&hdr_matrix, hdr_score_floor, hdr_margin);
                for (f, a) in hdr_assign.iter().enumerate() {
                    match a {
                        Some((h, score, margin)) => {
                            header_to_field_map.insert(unique_headers[*h].clone(), hdr_field_names[f].clone());
                            emit_term(&format!("    ✨ [HEADER COSINE MAP] Header '{}' → Field '{}' | Score: {:+.4} | Margin: {:+.4}{}", unique_headers[*h], hdr_field_names[f], score, margin, if hdr_leaf_won[f][*h] { " | 하위 라벨 기준" } else { "" }));
                            if hdr_leaf_won[f][*h] {
                                crate::utils::score_dynamics::record_baseline("commerce.header_leaf_win", 1.0);
                            }
                            crate::utils::score_dynamics::record_field_assigned(
                                &hdr_field_names[f],
                                *margin,
                            );
                        },
                        None => {
                            emit_term(&format!("    ⚪ [HEADER UNMAPPED] Field '{}' | 확정 가능한 헤더 없음. 값 라인 벡터 매칭으로 폴백합니다.", hdr_field_names[f]));
                        }
                    }
                }
            }

            
            
            
            let (idlink_label_phrases, idlink_label_weights) = label_phrase_bank(&doc_lang, &page_type, "id,link");
            let idlink_label_embs: Vec<Vec<f32>> = if idlink_label_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_label_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_label_phrases.len()])
            };

            let mut idlink_prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, "id,link");
            for extra in [
                "host name", "domain name", "website address", "server address",
                "cdn", "static asset", "image server", "protocol", "www",
                "file extension", "stylesheet", "script", "anchor", "javascript",
                "navigation menu", "layer popup",
            ] {
                let e = extra.to_string();
                if !idlink_prej_phrases.contains(&e) { idlink_prej_phrases.push(e); }
            }
            let idlink_prej_embs: Vec<Vec<f32>> = if idlink_prej_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_prej_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_prej_phrases.len()])
            };
            emit_term(&format!("  🔑 [ID/LINK COSINE BANK] 라벨 구 {}개 | 편견 구 {}개 준비 완료.", idlink_label_embs.len(), idlink_prej_embs.len()));

            
            
            
            let mut discovered_url_pattern: Option<(String, String)> = None; 
            let mut pattern_reference_link: Option<String> = None;
            let mut confirmed_id_shapes: Vec<(usize, bool)> = Vec::new();
            let mut all_item_raw_lines: Vec<Vec<String>> = Vec::new();
            let mut all_item_labeled_lines: Vec<Vec<String>> = Vec::new();

            // 🌟 [CROSSOVER] 여기서부터 임베딩과 Qwen3 가 진짜로 교차합니다.
            //
            //  아이템마다 라인 임베딩 → 필드별 generate 를 반복하므로
            //  '매번 스왑' 은 아이템 수만큼 왕복이 되어 재앙입니다.
            //  enter_generation_phase 는 예산을 보고
            //    · 여유가 있으면 임베딩을 유지한 채 Qwen3 를 얹고 (스왑 0회)
            //    · 여유가 없으면 임베딩만 반환시킨 뒤 Qwen3 를 올립니다.
            //  Qwen3 는 0.6B 라 임베딩과 함께 있어도 대부분 여유가 남습니다.
            //  이 판정이 하드코딩이 아니라 실측이므로 GPU 가 바뀌어도 유효합니다.
            let mut qwen3_ready = false;
            let list_route: Option<String> = {
                let verdict = model.lang_route("commerce", "list_field", &doc_lang, &lang_probe_text).await;
                emit_term(&format!("  {}", verdict.line));
                verdict.code
            };
            let mut field_lang: Option<crate::model::lang_llm::LangSession> = None;
            let synthesis_gate_for = |lang: bool| -> Option<(f32, u64)> {
                crate::utils::score_dynamics::adaptive_recent(crate::model::lang_llm::synthesis_drop_axis(lang))
                    .filter(|(recent, _, _)| *recent >= SYNTHESIS_SKIP_DROP)
                    .map(|(recent, _, n)| (recent, n))
            };
            let synthesis_gate_lang = synthesis_gate_for(true);
            let synthesis_gate_base = synthesis_gate_for(false);
            let mut synthesis_probe_item: Option<usize> = None;
            let mut synthesis_jobs: Vec<(usize, String, String)> = Vec::new();
            let mut synthesis_item_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            emit_term("  💤 [LAZY GENERATOR] 목록 루프의 Qwen3 는 LLM 이 실제로 필요한 첫 필드에서 올립니다. 모든 필드가 헤더 코사인 · Enum · 상태 어휘로 확정되면 이 태스크는 Qwen3 를 올리지 않습니다.");

            let vram_probe = model.spawn_vram_sampler("list item extraction loop");

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                let percent = (((idx as f32) / (total_items as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Item {}/{}", idx + 1, total_items), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("\n[STAGE-3] Processing List Item {}/{} ...", idx + 1, total_items));


                let full_item_pug = format!("{}\n{}", thead_pug, item_pug);
                

                let mut item_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                

                let item_cells = parse_pug_grid(&item_lines);
                let mut line_owner_field: Vec<Option<String>> = vec![None; item_lines.len()];
                let mut line_header_label: Vec<Option<String>> = vec![None; item_lines.len()];
                for cell in &item_cells {
                    let h_text = header_grid.cell_label(cell.row, cell.col, cell.colspan, cell.rowspan, page_interleaved);
                    let h_clean = h_text.trim();
                    if h_clean.is_empty() { continue; }
                    let owner = header_to_field_map.get(h_clean).cloned();
                    for &line_idx in &cell.line_indices {
                        if line_idx >= item_lines.len() { continue; }
                        line_header_label[line_idx] = Some(h_clean.to_string());
                        if let Some(o) = &owner {
                            line_owner_field[line_idx] = Some(o.clone());
                        }
                    }
                }
                let mut line_aux_control: Vec<bool> = vec![false; item_lines.len()];
                let mut line_sole_value: Vec<bool> = vec![false; item_lines.len()];
                for cell in &item_cells {
                    let visible: Vec<usize> = cell.line_indices.iter().copied()
                        .filter(|&li| li < item_lines.len() && pug_line_visible_data(&item_lines[li]))
                        .collect();
                    if visible.len() == 1 {
                        line_sole_value[visible[0]] = true;
                    }
                    if visible.is_empty() { continue; }
                    for &li in &cell.line_indices {
                        if li < item_lines.len() && pug_line_is_ui_control(&item_lines[li]) {
                            line_aux_control[li] = true;
                        }
                    }
                }

                for i in 0..item_lines.len() {
                    {
                        let l = item_lines[i].trim_start();
                        let is_select_control = l.starts_with("input")
                            && (l.contains("type=\"checkbox\"") || l.contains("type=\"radio\"")
                                || l.contains("type='checkbox'") || l.contains("type='radio'"));
                        if is_select_control {
                            if let Some(p) = item_lines[i].find('|') {
                                let t = item_lines[i][p + 1..].trim().to_string();
                                if !t.is_empty() {
                                    emit_term(&format!("    🚫 [FORM CONTROL DROP] Item Line {}/{} : {} (checkbox/radio 의 value 는 행 선택 인덱스입니다)", i + 1, item_lines.len(), t));
                                }
                                let head = item_lines[i][..=p].to_string();
                                item_lines[i] = format!("{} ", head);
                            }
                            continue;
                        }
                    }
                    let line = &item_lines[i];
                    if let Some(idx) = line.find('|') {
                        let text_part = line[idx + 1..].trim();
                        let is_control = pug_line_is_ui_control(line);
                        let control_dup = is_control && control_boilerplate.contains(text_part);
                        if boilerplate_texts.contains(text_part) || control_dup {
                            if !is_control {
                                let has_link_or_event = line_real_href(line).is_some() || line.contains("onclick") || line.contains("data-url");
                                if has_link_or_event {
                                    emit_term(&format!("    🛡️ [DUPLICATE LINK PROTECT] Item Line {}/{} : {} (실제 이동 href/event 포함 데이터 보호)", i + 1, item_lines.len(), text_part));
                                    continue;
                                }
                            }
                            let owner_enum = line_owner_field[i].as_ref().map_or(false, |o| detect_field_format(o) == FieldFormat::Enum);
                            let header_protectable = !(is_control && line_aux_control[i])
                                && (!hard_drop_texts.contains(text_part) || (owner_enum && line_sole_value[i] && !line.contains("href=")));
                            if header_protectable {
                                if let Some(owner) = line_owner_field[i].as_ref() {
                                    crate::utils::score_dynamics::record_baseline("commerce.header_repeat_protect", 1.0);
                                    emit_term(&format!("    🛡️ [HEADER OWNED PROTECT / REPEAT] Item Line {}/{} : {} (전 아이템 반복 텍스트지만 '{}' 컬럼으로 헤더 코사인 확정된 칸의 값이라 보호합니다)", i + 1, item_lines.len(), text_part, owner));
                                    continue;
                                }
                            }

                            emit_term(&format!("    🚫 [DUPLICATE FILTERED] Item Line {}/{} : {} ({})", i + 1, item_lines.len(), text_part, if is_control { "반복 버튼/입력 컨트롤 탈락" } else { "반복 UI 탈락" }));

                            item_lines[i] = format!("{} ", &line[..=idx]);
                        }
                    }
                }

                let mut line_enriched_texts = vec![String::new(); item_lines.len()];
                for cell in &item_cells {
                    for &line_idx in &cell.line_indices {
                        if line_idx >= item_lines.len() { continue; }
                        let original_text = if let Some(p) = item_lines[line_idx].find('|') {
                            item_lines[line_idx][p + 1..].trim()
                        } else {
                            ""
                        };
                        if original_text.is_empty() { continue; }
                        line_enriched_texts[line_idx] = match &line_header_label[line_idx] {
                            Some(h) => format!("{} | {}", h, original_text),
                            None => original_text.to_string(),
                        };
                    }
                }

                let mut item_embeddings = vec![vec![0.0; 384]; item_lines.len()];
                

                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {
                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" }
                        } else {
                            enriched.as_str()
                        };

                        if !target_text.is_empty() {
                            texts_to_embed.push(target_text.to_string());
                            text_indices.push(line_idx);
                        }
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    // 🌟 [CROSSOVER] 고정 100개 청킹을 제거하고 한 번에 넘깁니다.
                    //
                    //  ── 왜 100 을 없애는가 ──
                    //   이 값은 activation 여유와 무관하게 고정되어 있어,
                    //   Qwen3 와 동시 상주 중일 때도 100건씩 밀어 넣습니다.
                    //   그 순간 점유가 곧 피크의 두 번째 축입니다.
                    //   get_embedding_batch 가 adaptive_embed_batch 로 여유에 맞춰
                    //   쪼개므로, 여유가 넉넉하면 오히려 100보다 크게 묶어 더 빠릅니다.
                    //   캐시 적중분은 아예 연산되지 않아 실연산량 자체가 줄어듭니다.
                    let vectors = model.get_embedding_batch(texts_to_embed.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; texts_to_embed.len()]);
                    for (i, vector) in vectors.into_iter().enumerate() {
                        if i >= text_indices.len() { break; }
                        let original_idx = text_indices[i];
                        let emb = vector.clone();
                        let noise_score = cosine_similarity(&layout_prej_emb, &emb);

                        let original_text = texts_to_embed[i].trim();
                        let has_digit = original_text.chars().any(|c| c.is_ascii_digit());
                        let is_short = original_text.len() <= 3;

                        let is_structure_tag = original_text.starts_with("th")
                            || original_text.starts_with("td")
                            || original_text.starts_with("tr")
                            || original_text.starts_with("input")
                            || original_text.starts_with("div");

                        let is_header_owned = line_owner_field[original_idx].is_some();
                        if noise_score > 0.6 && !has_digit && !is_short && !is_structure_tag && !is_header_owned {
                            emit_term(&format!("    🚫 [NOISE FILTERED] Item Line {}/{} : {} (Score: {:.4})", original_idx + 1, item_lines.len(), original_text, noise_score));
                            item_lines[original_idx] = String::new(); 
                        } else {
                            if noise_score > 0.6 && is_header_owned {
                                emit_term(&format!("    🛡️ [HEADER OWNED PROTECT] Item Line {}/{} : {} (NoiseScore {:.4} 이지만 '{}' 컬럼으로 코사인 확정됨)", original_idx + 1, item_lines.len(), original_text, noise_score, line_owner_field[original_idx].clone().unwrap_or_default()));
                            } else {
                                emit_term(&format!("    [VECTORIZING] Item Line {}/{} : {}", original_idx + 1, item_lines.len(), original_text));
                            }
                            item_embeddings[original_idx] = emb;
                        }
                    }
                }


                let mut json_contexts = Vec::new();
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {

                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" }
                        } else {
                            enriched.as_str()
                        };

                        if !target_text.is_empty() {
                            if let Some(idx) = target_text.find('|') {
                                json_contexts.push(json!({
                                    "metadata": target_text[..idx].trim(),
                                    "value": target_text[idx + 1..].trim()
                                }));
                            } else {
                                json_contexts.push(json!({
                                    "value": target_text.trim()
                                }));
                            }
                        }
                    }
                }
                let filtered_full_item_pug = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();

                let mut item_val = json!({});
                let mut global_ignore_list: Vec<String> = Vec::new();
                

                let thead_lines_ref: Vec<&str> = thead_lines.iter().map(|s| s.as_str()).collect();
                let item_lines_ref: Vec<&str> = item_lines.iter().map(|s| s.as_str()).collect();


                
                
                
                

                
                
                
                let line_values: Vec<String> = item_lines_ref.iter().map(|line| {
                    match line.find('|') {
                        Some(p) => line[p + 1..].trim().to_string(),
                        None => String::new(),
                    }
                }).collect();


                let mut pre_mapped_hints = Vec::new();
                

                let mut url_pool = String::new();
                if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                    for line in &item_lines_ref {
                        for cap in href_re.captures_iter(line) {
                            if let Some(m) = cap.get(1) {
                                url_pool.push_str(&m.as_str().to_lowercase());
                                url_pool.push_str(" ");
                            }
                        }
                    }
                }

                
                
                
                
                
                
                let mut header_forced_assign: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                let mut header_owned_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let mut header_id_tokens: Vec<String> = Vec::new();

                let mut owner_line_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                let mut owner_real_href: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                let mut owner_dead_href: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for line_idx in 0..item_lines_ref.len() {
                    if let Some(o) = &line_owner_field[line_idx] {
                        if line_values[line_idx].is_empty() || line_aux_control[line_idx] { continue; }
                        *owner_line_counts.entry(o.clone()).or_insert(0) += 1;
                        let raw = item_lines_ref[line_idx];
                        if line_real_href(raw).is_some() {
                            *owner_real_href.entry(o.clone()).or_insert(0) += 1;
                        } else if raw.contains("href=") {
                            *owner_dead_href.entry(o.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let mut rep_meta: std::collections::HashMap<String, (usize, (u8, u8, i32, f32, usize))> = std::collections::HashMap::new();

                for line_idx in 0..item_lines_ref.len() {
                    let owner_field = match &line_owner_field[line_idx] {
                        Some(o) => o.clone(),
                        None => continue,
                    };
                    if item_lines_ref[line_idx].trim().is_empty() { continue; }

                    let target_text = if !line_enriched_texts[line_idx].is_empty() { &line_enriched_texts[line_idx] } else { item_lines_ref[line_idx] };
                    let clean_text = if let Some(idx) = target_text.find('|') { target_text[idx + 1..].trim() } else { "" };
                    let owner_fmt = detect_field_format(&owner_field);
                    if owner_fmt == FieldFormat::Link && !is_id_link_field(&owner_field) {
                        if let Some(src) = crate::utils::ai_utils::line_media_src(item_lines_ref[line_idx]) {
                            header_owned_lines.insert(line_idx);
                            let already = pre_mapped_hints
                                .iter()
                                .any(|h: &serde_json::Value| h.get("target_column").and_then(|v| v.as_str()) == Some(owner_field.as_str()));
                            if !already {
                                pre_mapped_hints.push(json!({
                                    "target_column": owner_field.clone(),
                                    "extracted_value": src.clone(),
                                    "line_rank": 2
                                }));
                                rep_meta.insert(owner_field.clone(), (line_idx, (0u8, 0u8, 2i32, 0.0f32, src.chars().count())));
                                crate::utils::score_dynamics::record_field_assigned(&owner_field, 0.0);
                                crate::utils::score_dynamics::record_baseline("commerce.media_src_harvest", 1.0);
                                emit_term(&format!("    🖼️ [HEADER OWNED / MEDIA SRC] '{}' ← Item Line {} | 이 칸에는 글자 값이 없고 img 의 src 만 있습니다. Link 형식 필드는 글자가 아니라 주소가 값이므로 src 를 그대로 확정합니다: {}", owner_field, line_idx + 1, src));
                            }
                            continue;
                        }
                    }
                    let min_chars = if owner_fmt == FieldFormat::Numeric { 1 } else { 2 };
                    if clean_text.is_empty() || clean_text.chars().count() < min_chars { continue; }

                    header_owned_lines.insert(line_idx);

                    if line_aux_control[line_idx] {
                        emit_term(&format!("    ⏭️ [AUX CONTROL SKIP] '{}' ← Item Line {} (\"{}\") | 같은 칸에 데이터 라인이 있는 버튼/입력 컨트롤이라 값 후보에서 뺍니다. 라인은 이 컬럼 소유로 남습니다.", owner_field, line_idx + 1, clean_text));
                        continue;
                    }

                    if is_id_link_field(&owner_field) {
                        for tok in clean_text.split(|c: char| !c.is_alphanumeric()) {
                            if tok.chars().count() < 6 { continue; }
                            if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
                            if !header_id_tokens.iter().any(|t| t == tok) { header_id_tokens.push(tok.to_string()); }
                        }
                        emit_term(&format!("    🔑 [HEADER OWNED / ID COLUMN] Item Line {} 는 '{}' 컬럼입니다. 결정론적 ID/LINK 해석기에 위임하고 타 컬럼 선점을 차단합니다.", line_idx + 1, owner_field));
                        continue;
                    }

                    let owner_fmt_ok = match owner_fmt {
                        FieldFormat::Link | FieldFormat::Synthesis => true,
                        FieldFormat::Identifier => {
                            clean_text.chars().any(|c| c.is_ascii_digit()) || value_token_in_url_pool(clean_text, &url_pool)
                        }
                        FieldFormat::Enum => {
                            value_matches_format(owner_fmt, clean_text) && enum_value_reject(&owner_field, clean_text).is_none()
                        }
                        _ => value_matches_format(owner_fmt, clean_text),
                    };
                    if !owner_fmt_ok {
                        crate::utils::score_dynamics::record_field_seen(&owner_field);
                        crate::utils::score_dynamics::record_field_reject(
                            &owner_field,
                            if owner_fmt == FieldFormat::Enum {
                                crate::utils::score_dynamics::GateKind::Enum
                            } else {
                                crate::utils::score_dynamics::GateKind::Format
                            },
                        );
                        emit_term(&format!("    🚫 [HEADER OWNED FORMAT REJECT] '{}' ({:?}) ← Item Line {} (\"{}\") | 헤더 칸의 값이 필드 형식과 맞지 않아 주입하지 않습니다. 라인은 이 컬럼 소유로 남아 다른 필드도 가져가지 못합니다.", owner_field, owner_fmt, line_idx + 1, clean_text));
                        continue;
                    }

                    let lower_owner = owner_field.to_lowercase();
                    let needs_normalization = lower_owner.contains("status")
                        || lower_owner.contains("payment_method")
                        || lower_owner.contains("payment_origin")
                        || lower_owner.contains("condition")
                        || lower_owner.contains("currency");

                    if needs_normalization {
                        header_forced_assign.entry(owner_field.clone()).or_insert(line_idx);
                        emit_term(&format!("    🎯 [HEADER FORCED ASSIGN] '{}' ← Item Line {} (\"{}\") | enum 정규화가 필요해 값 우회 대신 벡터 배정을 확정합니다.", owner_field, line_idx + 1, clean_text));
                        continue;
                    }

                    let raw_line = item_lines_ref[line_idx];
                    let multi_line = owner_line_counts.get(&owner_field).copied().unwrap_or(0) >= 2;
                    let action_scores: Option<(f32, f32)> = if multi_line && !ui_action_embs.is_empty() {
                        match fields.iter().position(|(n, _, _, _)| n == &owner_field) {
                            Some(oi) => {
                                let v_emb = model.get_embedding(clean_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);
                                if v_emb.iter().all(|&v| v == 0.0) {
                                    None
                                } else {
                                    let own = weighted_max_pool_sim(&v_emb, &field_phrase_embs[oi], &field_phrase_weights[oi]);
                                    let chrome = max_pool_sim(&v_emb, &ui_action_embs);
                                    crate::utils::score_dynamics::record_baseline("commerce.premap_action_margin", own - chrome);
                                    Some((own - chrome, chrome))
                                }
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    let action_margin = action_scores.map(|(m, _)| m);
                    let is_real_href = line_real_href(raw_line).is_some();
                    let real_links = owner_real_href.get(&owner_field).copied().unwrap_or(0);
                    let dead_links = owner_dead_href.get(&owner_field).copied().unwrap_or(0);
                    let menu_penalty: u8 = if is_real_href && dead_links >= 1 && real_links >= 2 { 1 } else { 0 };
                    let dropdown_trigger = !is_real_href && raw_line.contains("href=") && dead_links == 1 && real_links >= 2;
                    let noise_band = if dropdown_trigger && action_scores.is_some() {
                        crate::utils::score_dynamics::adaptive_baseline("commerce.premap_action_margin")
                            .map(|(_, sd)| sd)
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let trigger_hold = dropdown_trigger
                        && action_scores.map_or(false, |(m, chrome)| m < 0.0 && m >= -noise_band && chrome < 0.999);
                    if trigger_hold {
                        crate::utils::score_dynamics::record_baseline("commerce.premap_trigger_hold", 1.0);
                        emit_term(&format!(
                            "    🛡️ [PREMAP TRIGGER HOLD] '{}' ← Item Line {} (\"{}\") | 같은 칸의 실링크 {}개가 드롭다운 메뉴이고 이 줄은 그 메뉴를 여는 죽은 링크 트리거입니다. 트리거 글자는 칸의 값(주문자 이름 등)이고 UI 액션 구와 글자가 같지 않습니다. 필드 구 - UI 액션 구 = {:+.4} 가 SDS commerce.premap_action_margin 표준편차 {:.4} 안쪽이라 코사인만으로 UI 액션으로 강등하지 않습니다.",
                            owner_field, line_idx + 1, clean_text, real_links,
                            action_margin.unwrap_or(0.0), noise_band
                        ));
                    }
                    let action_like = !trigger_hold && action_margin.map_or(false, |m| m < 0.0);
                    let line_margin = action_margin.unwrap_or(0.0);
                    let line_rank: i32 = if is_real_href {
                        2
                    } else if is_bracket_annotation(clean_text) {
                        0
                    } else {
                        1
                    };
                    let new_key: (u8, u8, i32, f32, usize) = (
                        menu_penalty,
                        if action_like { 1 } else { 0 },
                        line_rank,
                        line_margin,
                        clean_text.chars().count(),
                    );
                    if action_like || menu_penalty == 1 {
                        crate::utils::score_dynamics::record_baseline("commerce.premap_action_demote", 1.0);
                        emit_term(&format!("    🧹 [PREMAP DEMOTE] '{}' ← Item Line {} (\"{}\") |{}{} | 필드 구 - UI 액션 구 = {:+.4}", owner_field, line_idx + 1, clean_text, if menu_penalty == 1 { format!(" 드롭다운 메뉴 링크 (같은 칸 죽은 링크 트리거 {}개 · 실링크 {}개)", dead_links, real_links) } else { String::new() }, if action_like { " · UI 액션 쪽" } else { "" }, line_margin));
                    }

                    if let Some(existing) = pre_mapped_hints.iter_mut().find(|h: &&mut serde_json::Value| h.get("target_column").and_then(|v| v.as_str()) == Some(owner_field.as_str())) {
                        let prev = existing.get("extracted_value").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let prev_key = rep_meta.get(&owner_field).map(|m| m.1).unwrap_or((0, 0, 1, 0.0, prev.chars().count()));

                        if is_multi_value_field(&owner_field) {
                            if action_like {
                                emit_term(&format!("    ⏭️ [SUBORDINATE SKIP] '{}' 다중값 병합에서 UI 액션 라인 \"{}\" 는 제외합니다.", owner_field, clean_text));
                            } else if prev_key.1 == 1 {
                                existing.as_object_mut().unwrap().insert("extracted_value".to_string(), json!(clean_text));
                                rep_meta.insert(owner_field.clone(), (line_idx, new_key));
                                emit_term(&format!("    🥇 [REPRESENTATIVE SWAP] '{}' 대표값 교체: \"{}\" (Rank {}) → \"{}\" (Rank {})", owner_field, prev, prev_key.2, clean_text, line_rank));
                            } else if !prev.is_empty() && prev != clean_text {
                                existing.as_object_mut().unwrap().insert("extracted_value".to_string(), json!(format!("{} {}", prev, clean_text)));
                            }
                        } else if prev.is_empty() || premap_rep_better(new_key, prev_key) {
                            existing.as_object_mut().unwrap().insert("extracted_value".to_string(), json!(clean_text));
                            rep_meta.insert(owner_field.clone(), (line_idx, new_key));
                            emit_term(&format!("    🥇 [REPRESENTATIVE SWAP] '{}' 대표값 교체: \"{}\" (Rank {}) → \"{}\" (Rank {})", owner_field, prev, prev_key.2, clean_text, line_rank));
                        } else {
                            emit_term(&format!("    ⏭️ [SUBORDINATE SKIP] '{}' 는 이미 상위 대표값(\"{}\")을 확보하여 \"{}\" (Rank {}) 는 병합하지 않습니다.", owner_field, prev, clean_text, line_rank));
                        }
                    } else {
                        pre_mapped_hints.push(json!({
                            "target_column": owner_field.clone(),
                            "extracted_value": clean_text,
                            "line_rank": line_rank
                        }));
                        rep_meta.insert(owner_field.clone(), (line_idx, new_key));
                    }
                    emit_term(&format!("    🔍 [FAST-PRE-MAP] Item Line {} mapped to '{}' (Rank {}) via Header cosine", line_idx + 1, owner_field, line_rank));
                }

                let premap_handoff: Vec<(String, usize)> = rep_meta
                    .iter()
                    .filter(|(_, (_, k))| k.1 == 1)
                    .map(|(o, (li, _))| (o.clone(), *li))
                    .collect();
                let mut premap_handoff_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
                for (owner, li) in premap_handoff {
                    pre_mapped_hints.retain(|h| h.get("target_column").and_then(|v| v.as_str()) != Some(owner.as_str()));
                    premap_handoff_fields.insert(owner.clone());
                    crate::utils::score_dynamics::record_baseline("commerce.premap_action_handoff", 1.0);
                    emit_term(&format!("    🎯 [PREMAP ACTION HANDOFF] '{}' 칸의 후보가 모두 UI 액션 쪽이라 값 우회(PRE-MAP BYPASS)를 멈추고, 한 줄 강제 없이 아이템 전체 문맥으로 LLM 이 고르게 합니다. (최선 후보 Item Line {})", owner, li + 1));
                }
                

                let pre_mapped_context = if !pre_mapped_hints.is_empty() {
                    
                    
                    let clean_hints: Vec<serde_json::Value> = pre_mapped_hints.iter().map(|h| {
                        let mut c = h.clone();
                        if let Some(o) = c.as_object_mut() { o.remove("line_rank"); }
                        c
                    }).collect();
                    serde_json::to_string_pretty(&clean_hints).unwrap_or_default()
                } else {
                    String::new()
                };

                
                
                
                
                
                let idlink_cands = collect_id_link_candidates(&item_lines_ref);
                let mut det_id_link: Option<(String, String)> = None;

                if !idlink_cands.is_empty() && !idlink_label_embs.is_empty() {
                    let role_texts: Vec<String> = idlink_cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best_score = f32::MIN;
                    let mut best_idx: Option<usize> = None;
                    let mut idlink_scores: Vec<f32> = Vec::new();
                    let mut idlink_top = f32::MIN;
                    for (ci, cand) in idlink_cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (cand.prior - 1.0);
                        idlink_scores.push(score);
                        if score > idlink_top { idlink_top = score; }
                        emit_term(&format!("      🧭 [ID/LINK CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Prior: {:.2}{} | Score: {:+.4}",
                            cand.token, cand.role_phrase, own, prej, cand.prior,
                            if cand.is_host_part { " (host)" } else { "" }, score));
                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best_score { best_score = score; best_idx = Some(ci); }
                    }
                    if idlink_scores.len() >= 2 {
                        crate::utils::score_dynamics::record_decay("commerce.idlink_row", &idlink_scores);
                    }
                    if idlink_top > f32::MIN {
                        crate::utils::score_dynamics::record_baseline("commerce.idlink_top", idlink_top);
                    }
                    if let Some(bi) = best_idx {
                        let c = &idlink_cands[bi];
                        det_id_link = Some((c.token.clone(), c.href.clone()));
                        emit_term(&format!("    🔑 [ID/LINK COSINE] 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'", c.token, c.role_phrase, best_score, c.href));
                    }
                }

                if det_id_link.is_none() {
                    det_id_link = resolve_id_link_from_lines(&item_lines_ref);
                    if let Some((fid, flink)) = &det_id_link {
                        emit_term(&format!("    🔑 [ID/LINK FALLBACK] 코사인 게이트를 통과한 후보가 없어 레거시 해석기로 확정: '{}' → '{}'", fid, flink));
                    }
                }
                match &det_id_link {
                    Some(_) => crate::utils::score_dynamics::record_field_assigned("id,link", 0.0),
                    None => crate::utils::score_dynamics::record_field_reject(
                        "id,link",
                        crate::utils::score_dynamics::GateKind::Prejudice,
                    ),
                }

                let mut det_consumed_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
                if let Some((det_id, det_link)) = &det_id_link {
                    
                    
                    for (l, v) in line_values.iter().enumerate() {
                        if v.is_empty() { continue; }
                        let matched = v.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| !tok.is_empty() && tok.eq_ignore_ascii_case(det_id.as_str()));
                        if matched { det_consumed_lines.insert(l); }
                    }
                    emit_term(&format!("    🔑 [ID/LINK DETERMINISTIC] 식별자 '{}' 가 href '{}' 안에 실제 존재함을 확인. 해당 라인은 다른 컬럼이 선점할 수 없습니다.", det_id, det_link));
                }

                
                for l in &header_owned_lines {
                    det_consumed_lines.insert(*l);
                }

                
                
                
                if !header_id_tokens.is_empty() {
                    let mut shadow_hits = 0usize;
                    for (l, v) in line_values.iter().enumerate() {
                        if v.is_empty() || det_consumed_lines.contains(&l) { continue; }
                        let hit = v.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| header_id_tokens.iter().any(|t| t.eq_ignore_ascii_case(tok)));
                        if hit {
                            det_consumed_lines.insert(l);
                            shadow_hits += 1;
                        }
                    }
                    if shadow_hits > 0 {
                        emit_term(&format!("    🧹 [ID SHADOW DROP] 식별자 컬럼 값을 복제한 라인 {}개를 다른 컬럼 후보에서 제외했습니다.", shadow_hits));
                    }
                }

                let field_header_mapped: Vec<bool> = fields
                    .iter()
                    .map(|(n, _, _, _)| header_to_field_map.values().any(|v| v == n))
                    .collect();
                let mut enum_value_blocked = 0usize;
                let mut column_contract_blocked = 0usize;
                let mut column_absent_blocked = vec![0usize; fields.len()];
                let aux_control_lines = (0..item_lines_ref.len())
                    .filter(|&l| line_aux_control[l] && !det_consumed_lines.contains(&l) && !line_values[l].is_empty())
                    .count();
                let (mut vector_assignment, vector_raw_matrix): (Vec<Option<(usize, f32, f32)>>, Vec<Vec<f32>>) = {
                    let line_count = item_lines_ref.len();
                    let field_count = field_phrase_embs.len();
                    let mut raw = vec![vec![-1.0f32; line_count]; field_count];

                    for f in 0..field_count {
                        if field_is_analytic[f] { continue; }
                        let fmt = field_formats[f];
                        for l in 0..line_count {
                            if item_lines_ref[l].trim().is_empty() { continue; }
                            if item_embeddings[l].iter().all(|&v| v == 0.0) { continue; }
                            if det_consumed_lines.contains(&l) { continue; }
                            if line_aux_control[l] { continue; }

                            let value = &line_values[l];
                            let format_ok = match fmt {
                                FieldFormat::Identifier | FieldFormat::Link => value_token_in_url_pool(value, &url_pool),
                                _ => value_matches_format(fmt, value),
                            };
                            if !format_ok { continue; }
                            if fmt == FieldFormat::Enum && enum_value_reject(&fields[f].0, value).is_some() {
                                enum_value_blocked += 1;
                                continue;
                            }
                            if field_header_mapped.get(f).copied().unwrap_or(false) {
                                if let Some(h) = &line_header_label[l] {
                                    let owner_of_line = header_to_field_map.get(h.as_str()).map(|o| o.as_str());
                                    if owner_of_line != Some(fields[f].0.as_str()) {
                                        column_contract_blocked += 1;
                                        continue;
                                    }
                                }
                            } else if crate::utils::ai_utils::header_column_foreign(
                                &fields[f].0,
                                line_header_label[l].as_deref(),
                                value,
                                &list_field_vocab[f],
                                &header_evaluated,
                                &header_plausible,
                            ) {
                                column_absent_blocked[f] += 1;
                                continue;
                            }

                            raw[f][l] = weighted_max_pool_sim(
                                &item_embeddings[l],
                                &field_phrase_embs[f],
                                &field_phrase_weights[f],
                            );
                        }
                    }

                    let centered = double_center_matrix(&raw);
                    let mut assign = exclusive_assign(&centered, 0.0, 0.005);

                    let mut claimed = vec![false; line_count];
                    for a in assign.iter() {
                        if let Some((l, _, _)) = a { claimed[*l] = true; }
                    }
                    for f in 0..field_count {
                        if assign[f].is_some() { continue; }
                        if field_is_analytic[f] { continue; }
                        let cands: Vec<usize> = (0..line_count)
                            .filter(|&l| raw[f][l] >= 0.0 && !claimed[l])
                            .collect();
                        if cands.len() == 1 {
                            let l = cands[0];
                            let lone_foreign = column_absent_blocked[f] > 0
                                || (!list_field_vocab[f].is_empty()
                                    && crate::utils::ai_utils::header_judged_foreign(
                                        &fields[f].0,
                                        line_header_label[l].as_deref(),
                                        &header_evaluated,
                                        &header_plausible,
                                    ));
                            if lone_foreign
                                && !crate::utils::ai_utils::value_equals_vocabulary(&line_values[l], &list_field_vocab[f])
                            {
                                continue;
                            }
                            assign[f] = Some((l, centered[f][l], 0.0));
                            claimed[l] = true;
                        }
                    }

                    (assign, raw)
                };
                if enum_value_blocked > 0 || column_contract_blocked > 0 || aux_control_lines > 0 {
                    crate::utils::score_dynamics::record_baseline("commerce.enum_value_block", enum_value_blocked as f32);
                    crate::utils::score_dynamics::record_baseline("commerce.column_contract_block", column_contract_blocked as f32);
                    crate::utils::score_dynamics::record_baseline("commerce.aux_control_block", aux_control_lines as f32);
                    emit_term(&format!("    🧱 [COLUMN CONTRACT] 열거형 값이 될 수 없는 후보(순수 수치·통화 표지 없음) {}건, 헤더로 컬럼이 확정된 필드의 다른 컬럼 후보 {}건을 배정 행렬에서 제외했고, 데이터 칸에 붙은 보조 버튼/입력 라인 {}개는 후보에서 뺐습니다.", enum_value_blocked, column_contract_blocked, aux_control_lines));
                }
                let column_absent_total: usize = column_absent_blocked.iter().sum();
                if column_absent_total > 0 {
                    crate::utils::score_dynamics::record_baseline("commerce.column_absent_block", column_absent_total as f32);
                    let per_field: Vec<String> = fields
                        .iter()
                        .enumerate()
                        .filter(|(f, _)| column_absent_blocked[*f] > 0)
                        .map(|(f, (n, _, _, _))| format!("{}:{}", n, column_absent_blocked[f]))
                        .collect();
                    emit_term(&format!("    🧱 [COLUMN ABSENT] 헤더 매핑이 없는 닫힌 어휘 필드(열거형·색상 등)의 후보 중, 헤더 판정에서 그 필드의 칸이 될 수 없다고 나왔고 값도 그 필드의 어휘에 걸리지 않는 라인 {}건을 배정 행렬에서 뺐습니다. {:?} (헤더가 없는 라인, 값이 어휘에 걸리는 라인, 자유 글자 필드와 통화·id,link·링크 필드는 그대로 둡니다)", column_absent_total, per_field));
                }
                let mut evt_blocked: Vec<usize> = Vec::new();
                for (f, a) in vector_assignment.iter().enumerate() {
                    let l = match a { Some((l, _, _)) => *l, None => continue };
                    if header_forced_assign.contains_key(&fields[f].0) { continue; }
                    let cands: Vec<f32> = vector_raw_matrix[f].iter().copied().filter(|v| *v > -1.0).collect();
                    if cands.len() < 3 { continue; }
                    let n = cands.len() as f32;
                    let mean = cands.iter().sum::<f32>() / n;
                    let sd = (cands.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt().max(1e-6);
                    let z = (vector_raw_matrix[f][l] - mean) / sd - gumbel_expected_z(cands.len());
                    crate::utils::score_dynamics::record_baseline("commerce.value_assign_evt", z);
                    if z < 0.0 {
                        let header_foreign = crate::utils::ai_utils::header_judged_foreign(
                            &fields[f].0,
                            line_header_label[l].as_deref(),
                            &header_evaluated,
                            &header_plausible,
                        ) && !crate::utils::ai_utils::value_equals_vocabulary(&line_values[l], &list_field_vocab[f]);
                        crate::utils::score_dynamics::record_baseline("commerce.value_assign_evt_block", if header_foreign { 1.0 } else { 0.0 });
                        if header_foreign {
                            emit_term(&format!("    🚫 [VALUE ASSIGN EVT BLOCK] '{}' ← Line {} (헤더 '{}') | 배정 유사도가 후보 {}개의 무작위 최댓값 기대치에 못 미치고 (z {:+.3}), 이 라인의 헤더도 헤더 판정에서 '{}' 의 칸이 될 수 없다고 나왔습니다. 서로 독립인 두 음성 근거가 겹쳐 배정을 풉니다. 풀지 않으면 LLM 이 이 라인을 근거로 옆 칸의 값을 빌려 옵니다.", fields[f].0, l + 1, line_header_label[l].as_deref().unwrap_or(""), cands.len(), z, fields[f].0));
                            evt_blocked.push(f);
                            continue;
                        }
                        emit_term(&format!("    📐 [VALUE ASSIGN EVT] '{}' ← Line {} | 배정된 라인의 유사도가 후보 {}개 분포에서 무작위 최댓값 기대치에 못 미칩니다 (z {:+.3}). 이 라인의 헤더가 이 필드의 칸이 될 수 없다는 판정은 없어 이번 라운드는 관측만 하고 배정은 그대로 둡니다.", fields[f].0, l + 1, cands.len(), z));
                    }
                }
                for f in evt_blocked {
                    vector_assignment[f] = None;
                }

                
                
                
                for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                    if let Some(l) = header_forced_assign.get(fname) {
                        let raw = vector_raw_matrix.get(f_i).and_then(|r| r.get(*l)).copied().unwrap_or(0.0).max(0.0);
                        vector_assignment[f_i] = Some((*l, raw, 0.0));
                        emit_term(&format!("    🧷 [HEADER OVERRIDE] '{}' 의 벡터 배정을 헤더 코사인 확정 컬럼(Line {})으로 교체했습니다.", fname, *l + 1));
                    }
                }

                {
                    for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                        if field_is_analytic[f_i] { continue; }
                        crate::utils::score_dynamics::record_field_seen(fname);
                    }
                    for l in 0..item_lines_ref.len() {
                        let mut col: Vec<f32> = Vec::new();
                        for f_i in 0..fields.len() {
                            if field_is_analytic[f_i] { continue; }
                            let s = vector_raw_matrix[f_i][l];
                            if s < 0.0 { continue; }
                            col.push(s);
                        }
                        if col.len() >= 2 {
                            crate::utils::score_dynamics::record_decay("commerce.item_row", &col);
                        }
                    }
                }
                for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                    match vector_assignment[f_i] {
                        Some((l, contrast, margin)) => {
                            let shown = if line_enriched_texts[l].is_empty() {
                                item_lines_ref[l].trim()
                            } else {
                                line_enriched_texts[l].as_str()
                            };
                            emit_term(&format!("    🔗 [EXCLUSIVE ASSIGN] '{}' ({:?}) ← Line {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4} | \"{}\"", fname, field_formats[f_i], l + 1, vector_raw_matrix[f_i][l], contrast, margin, shown));
                            crate::utils::score_dynamics::record_field_assigned(fname, margin);
                            let mut rival_name = String::new();
                            let mut rival_score = f32::MIN;
                            for (g_i, (gname, _, _, _)) in fields.iter().enumerate() {
                                if g_i == f_i { continue; }
                                if field_is_analytic[g_i] { continue; }
                                let s = vector_raw_matrix[g_i][l];
                                if s < 0.0 { continue; }
                                if s > rival_score {
                                    rival_score = s;
                                    rival_name = gname.clone();
                                }
                            }
                            if !rival_name.is_empty() {
                                let own = vector_raw_matrix[f_i][l];
                                crate::utils::score_dynamics::record_confusion(
                                    fname,
                                    &rival_name,
                                    (own - rival_score).abs(),
                                );
                            }
                        },
                        None => {
                            if !field_is_analytic[f_i] {
                                let cand_cnt = vector_raw_matrix[f_i].iter().filter(|&&v| v >= 0.0).count();
                                emit_term(&format!("    ⚪ [UNASSIGNED] '{}' ({:?}) | 형식 통과 후보 {}개 | 벡터 힌트 미주입", fname, field_formats[f_i], cand_cnt));
                                if cand_cnt == 0 {
                                    crate::utils::score_dynamics::record_field_reject(
                                        fname,
                                        crate::utils::score_dynamics::GateKind::Format,
                                    );
                                } else {
                                    crate::utils::score_dynamics::record_near_miss(fname);
                                }
                            }
                        }
                    }
                }


                let field_order: Vec<usize> = (0..fields.len())
                    .filter(|i| !field_is_analytic[*i])
                    .chain((0..fields.len()).filter(|i| field_is_analytic[*i]))
                    .collect();
                for f_idx in field_order {
                    let (field_name, field_desc, bias_target, prejudice_target) = fields[f_idx].clone();

                    let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                    let mut bypassed_values: Vec<(String, String)> = Vec::new();
                    for k in &keys {
                        for hint in &pre_mapped_hints {
                            if let Some(t_col) = hint.get("target_column").and_then(|v| v.as_str()) {
                                if t_col == *k {
                                    if let Some(e_val) = hint.get("extracted_value").and_then(|v| v.as_str()) {
                                        let clean_e_val = e_val.trim();
                                        if !clean_e_val.is_empty() {
                                            if let Some(existing) = bypassed_values.iter_mut().find(|(key, _)| key == *k) {
                                                if !existing.1.contains(clean_e_val) {
                                                    existing.1.push_str(" ");
                                                    existing.1.push_str(clean_e_val);
                                                }
                                            } else {
                                                bypassed_values.push((k.to_string(), clean_e_val.to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !bypassed_values.is_empty() {
                        let f_percent = (((f_idx as f32) / (total_fields as f32)) * 100.0) as i32;
                        let f_summary_msg = format!("Extracting {} ({}%)...", field_name, f_percent);
                        let payload = json!({ 
                            "task_id": task.id, 
                            "category": format!("List Item {}/{}", idx + 1, total_items), 
                            "summary": f_summary_msg, 
                            "spinner": "⠋" 
                        });
                        log_task_progress(app_handle, &task.id, &payload);
                        emit_term(&format!("  ▶ {}", f_summary_msg));

                        let shareable_field = ["name", "insight", "status", "payment_method", "date", "_at", "currency", "goods", "title"]
                            .iter().any(|f| field_name.contains(f));

                        let mut extracted_results = Vec::new();
                        for (k, val_str) in bypassed_values {
                            crate::utils::score_dynamics::record_field_assigned(&k, 0.0);
                            crate::utils::score_dynamics::record_baseline("commerce.premap_bypass", 1.0);
                            item_val.as_object_mut().unwrap().insert(k.clone(), json!(val_str));
                            extracted_results.push(format!("\"{}\": \"{}\"", k, val_str));
                            
                            if !shareable_field && val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                if !global_ignore_list.contains(&val_str) {
                                    global_ignore_list.push(val_str.clone());
                                    global_ignore_list.push(format!(" {}", val_str));
                                    global_ignore_list.push(val_str.to_lowercase());
                                }
                            }
                        }
                        emit_term(&format!("    ⚡ [PRE-MAP BYPASS] Successfully mapped without LLM: {}", extracted_results.join(", ")));
                        continue;
                    }

                    let field_format = field_formats[f_idx];

                    
                    
                    if is_id_link_field(&field_name) {
                        if let Some((det_id, det_link)) = det_id_link.clone() {
                            item_val.as_object_mut().unwrap().insert("id".to_string(), json!(det_id.clone()));
                            item_val.as_object_mut().unwrap().insert("link".to_string(), json!(det_link.clone()));
                            if !global_ignore_list.contains(&det_id) {
                                global_ignore_list.push(det_id.clone());
                                global_ignore_list.push(format!(" {}", det_id));
                                global_ignore_list.push(det_id.to_lowercase());
                            }
                            
                            
                            
                            let id_shape = id_shape_signature(&det_id);
                            if !confirmed_id_shapes.contains(&id_shape) { confirmed_id_shapes.push(id_shape); }

                            if discovered_url_pattern.is_none() {
                                if let Some((prefix, suffix)) = extract_url_pattern(&det_id, &det_link) {
                                    discovered_url_pattern = Some((prefix.clone(), suffix.clone()));
                                    pattern_reference_link = Some(det_link.clone());
                                    emit_term(&format!("    📐 [URL PATTERN DISCOVERED] prefix: '{}' | suffix: '{}' | IdShape: (길이 {}, 숫자전용 {}) → 이후 실패 아이템에 소급 적용 가능", prefix, suffix, id_shape.0, id_shape.1));
                                } else {
                                    emit_term(&format!("    🚫 [URL PATTERN REJECTED] 식별자 '{}' 가 path/query 구간에서 발견되지 않아 패턴화를 거부했습니다. (link: {})", det_id, det_link));
                                }
                            }
                            emit_term(&format!("    ⚡ [ID/LINK BYPASS] LLM 없이 확정: \"id\": \"{}\", \"link\": \"{}\"", det_id, det_link));
                            continue;
                        }
                    }

                    let currency_ev = if field_name.to_lowercase().contains("currency") {
                        let texts: Vec<String> = (0..item_lines_ref.len())
                            .map(|l| {
                                if line_enriched_texts[l].is_empty() {
                                    line_values[l].clone()
                                } else {
                                    line_enriched_texts[l].clone()
                                }
                            })
                            .collect();
                        Some(crate::utils::ai_utils::currency_evidence(&texts))
                    } else {
                        None
                    };
                    if let Some(ev) = currency_ev.as_ref() {
                        let unmarked_price_line = (0..item_lines_ref.len()).any(|l| {
                            let price_owned = line_owner_field[l].as_deref().map_or(false, |o| {
                                let lo = o.to_lowercase();
                                lo.contains("price") || lo.contains("amount")
                            });
                            let text = if line_enriched_texts[l].is_empty() { &line_values[l] } else { &line_enriched_texts[l] };
                            price_owned && line_sole_value[l] && crate::utils::ai_utils::currency_line_unmarked(text)
                        });
                        if let Some(code) = ev.single().filter(|_| !unmarked_price_line) {
                            item_val.as_object_mut().unwrap().insert(field_name.clone(), json!(code));
                            crate::utils::score_dynamics::record_field_assigned(&field_name, 0.0);
                            crate::utils::score_dynamics::record_baseline("commerce.currency_grounded", 1.0);
                            emit_term(&format!("    💱 [CURRENCY GROUNDED] '{}' ← '{}' | 이 아이템 라인의 통화 표지가 금액에 붙은 한 가지 코드로만 풀립니다 ({}). LLM 없이 확정합니다.", field_name, code, ev.describe()));
                            continue;
                        }
                        if ev.is_silent() {
                            crate::utils::score_dynamics::record_baseline("commerce.currency_silent_default", 1.0);
                            emit_term(&format!("    💱 [CURRENCY GROUNDED / DEFAULT] '{}' | 이 아이템 라인 어디에도 통화 표지가 없습니다. LLM 에게 묻지 않고 비워 두며, 저장 직전 문서 언어 기본 통화가 들어갑니다.", field_name));
                            continue;
                        }
                    }

                    
                    let (best_item_idx, best_item_contrast, best_item_margin, has_vector_match) = match vector_assignment[f_idx] {
                        Some((l, contrast, margin)) => (l, contrast, margin, true),
                        None => (0usize, 0.0f32, 0.0f32, false),
                    };
                    let mut best_item_raw = if has_vector_match { vector_raw_matrix[f_idx][best_item_idx] } else { 0.0f32 };
                    
                    
                    let mut best_item_idx = best_item_idx;
                    let mut best_item_contrast = best_item_contrast;
                    let mut best_item_margin = best_item_margin;
                    let mut has_vector_match = has_vector_match;

                    
                    
                    
                    let strict_format_field = matches!(
                        field_format,
                        FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric | FieldFormat::Identifier | FieldFormat::Link
                    );
                    if !field_is_analytic[f_idx] && strict_format_field && !has_vector_match {
                        let leftover: Vec<usize> = (0..item_lines_ref.len())
                            .filter(|&l| vector_raw_matrix[f_idx][l] >= 0.0)
                            .filter(|&l| !vector_assignment.iter().any(|a| matches!(a, Some((al, _, _)) if *al == l)))
                            .collect();
                        if leftover.is_empty() {
                            emit_term(&format!("    ⛔ [FORMAT SKIP] Field: '{}' ({:?}) | 형식 게이트를 통과한 후보 셀이 이 아이템에 하나도 없습니다. LLM 호출 없이 빈 값으로 확정.", field_name, field_format));
                            crate::utils::score_dynamics::record_field_reject(
                                &field_name,
                                crate::utils::score_dynamics::GateKind::Format,
                            );
                            continue;
                        }
                        let mut pick: Option<(usize, f32, f32)> = None;
                        for &l in &leftover {
                            let own = vector_raw_matrix[f_idx][l];
                            let mut rival = 0.0f32;
                            for other in 0..field_phrase_embs.len() {
                                if other == f_idx { continue; }
                                if field_is_analytic[other] { continue; }
                                let s = vector_raw_matrix[other][l];
                                if s < 0.0 { continue; }
                                if s > rival { rival = s; }
                            }
                            if own <= rival { continue; }
                            let gap = own - rival;
                            if pick.map_or(true, |(_, _, g)| gap > g) {
                                pick = Some((l, own, gap));
                            }
                        }
                        match pick {
                            Some((l, own, gap)) => {
                                best_item_idx = l;
                                best_item_contrast = own;
                                best_item_margin = gap;
                                best_item_raw = own;
                                has_vector_match = true;
                                vector_assignment[f_idx] = Some((l, own, gap));
                                emit_term(&format!("    ♻️ [SECOND CHANCE ASSIGN] '{}' ({:?}) ← Line {} | RawSim: {:.4} | 경쟁 필드 대비 우위: {:+.4} | 남은 후보 {}개 중 선택", field_name, field_format, l + 1, own, gap, leftover.len()));
                                crate::utils::score_dynamics::record_field_assigned(&field_name, gap);
                                crate::utils::score_dynamics::record_baseline("commerce.second_chance_gap", gap);
                            }
                            None => {
                                emit_term(&format!("    ⛔ [FORMAT SKIP] Field: '{}' ({:?}) | 형식 통과 후보 {}개가 모두 경쟁 필드에 더 가까워 배정을 포기합니다. (잘못 채우는 것보다 공란이 안전합니다)", field_name, field_format, leftover.len()));
                                crate::utils::score_dynamics::record_field_reject(
                                    &field_name,
                                    crate::utils::score_dynamics::GateKind::Prejudice,
                                );
                                continue;
                            }
                        }
                    }

                    if !field_is_analytic[f_idx]
                        && !has_vector_match
                        && !is_id_link_field(&field_name)
                        && !premap_handoff_fields.contains(&field_name)
                        && field_header_mapped.get(f_idx).copied().unwrap_or(false)
                    {
                        emit_term(&format!("    ⛔ [HEADER COLUMN EMPTY] Field: '{}' ({:?}) | 헤더로 컬럼이 확정된 필드인데 이 아이템의 그 칸에 형식을 통과한 값이 없습니다. 다른 칸에서 빌려 오지 않도록 LLM 호출 없이 빈 값으로 확정합니다.", field_name, field_format));
                        crate::utils::score_dynamics::record_field_reject(
                            &field_name,
                            crate::utils::score_dynamics::GateKind::Format,
                        );
                        crate::utils::score_dynamics::record_baseline("commerce.header_column_empty", 1.0);
                        continue;
                    }

                    if !field_is_analytic[f_idx]
                        && !has_vector_match
                        && !is_id_link_field(&field_name)
                        && !premap_handoff_fields.contains(&field_name)
                        && !field_header_mapped.get(f_idx).copied().unwrap_or(false)
                        && column_absent_blocked.get(f_idx).copied().unwrap_or(0) > 0
                        && vector_raw_matrix[f_idx].iter().all(|v| *v < 0.0)
                    {
                        emit_term(&format!("    ⛔ [HEADER ABSENT SKIP] Field: '{}' ({:?}) | 이 필드에 매핑된 헤더가 없고 후보 라인이 모두 막혔습니다. 그중 {}건은 헤더 판정에서 이 필드의 칸이 될 수 없다고 나왔고 값도 이 필드의 어휘에 걸리지 않은 라인입니다. 다른 칸에서 빌려 오지 않도록 LLM 호출 없이 빈 값으로 확정합니다.", field_name, field_format, column_absent_blocked[f_idx]));
                        crate::utils::score_dynamics::record_field_reject(
                            &field_name,
                            crate::utils::score_dynamics::GateKind::Prejudice,
                        );
                        crate::utils::score_dynamics::record_baseline("commerce.header_absent_skip", 1.0);
                        continue;
                    }

                    if field_name == "status" && has_vector_match {
                        let raw_status = line_values.get(best_item_idx).map(|s| s.trim().to_string()).unwrap_or_default();
                        if let Some((key, route)) = crate::utils::ai_utils::status_canonical_exact(&raw_status, &page_type, list_status_pivot.as_ref()) {
                            item_val.as_object_mut().unwrap().insert(field_name.clone(), json!(key.clone()));
                            crate::utils::score_dynamics::record_field_assigned(&field_name, 0.0);
                            crate::utils::score_dynamics::record_baseline("commerce.status_pivot_hit", 1.0);
                            emit_term(&format!("    🌉 [STATUS EXACT] '{}' ← Item Line {} (\"{}\") → '{}' | {} | 배정된 상태 칸의 값 전체가 닫힌 상태 어휘와 정확히 맞아 LLM 없이 확정합니다.", field_name, best_item_idx + 1, raw_status, key, route));
                            continue;
                        }
                    }

                    if field_format == FieldFormat::Enum
                        && field_name != "status"
                        && !field_name.to_lowercase().contains("currency")
                        && has_vector_match
                        && header_forced_assign.contains_key(&field_name)
                    {
                        let raw_cell = line_values.get(best_item_idx).map(|s| s.trim().to_string()).unwrap_or_default();
                        if let Some(plain) = crate::utils::ai_utils::enum_cell_plain(&field_name, &raw_cell) {
                            item_val.as_object_mut().unwrap().insert(field_name.clone(), json!(plain.clone()));
                            crate::utils::score_dynamics::record_field_assigned(&field_name, best_item_margin);
                            crate::utils::score_dynamics::record_baseline("commerce.enum_direct_assign", 1.0);
                            emit_term(&format!("    🎯 [ENUM DIRECT] '{}' ← Item Line {} (\"{}\") → '{}' | 헤더 코사인으로 확정된 열의 값이 짧은 닫힌 어휘 모양이라 LLM 에게 다시 묻지 않고 그대로 확정합니다. 이 자리의 LLM 은 같은 원문 표기를 그대로 돌려주고 있었고, 정규화 표가 없는 축이라 원문 표기를 보존합니다.", field_name, best_item_idx + 1, raw_cell, plain));
                            continue;
                        }
                    }

                    let (_bias_emb, _prej_emb, dynamic_prej_str) = &field_embeddings[f_idx];

                    
                    let mut best_thead_idx = 0usize;
                    let mut best_thead_score = -1.0f32;
                    let mut best_thead_own = 0.0f32;
                    for (i, emb) in thead_embeddings.iter().enumerate() {
                        if thead_lines_ref[i].trim().is_empty() { continue; }
                        if emb.iter().all(|&v| v == 0.0) { continue; }

                        let own = weighted_max_pool_sim(emb, &field_phrase_embs[f_idx], &field_phrase_weights[f_idx]);
                        let mut rival = 0.0f32;
                        for other_idx in 0..field_phrase_embs.len() {
                            if other_idx == f_idx { continue; }
                            let s = weighted_max_pool_sim(emb, &field_phrase_embs[other_idx], &field_phrase_weights[other_idx]);
                            if s > rival { rival = s; }
                        }
                        let final_score = own - rival;

                        if final_score > best_thead_score {
                            best_thead_score = final_score;
                            best_thead_idx = i;
                            best_thead_own = own;
                        }
                    }
                    let _ = best_thead_idx;

                    if field_is_analytic[f_idx] {
                        synthesis_jobs.push((idx, field_name.clone(), field_desc.clone()));
                        emit_term(&format!(
                            "    ⏳ [SYNTHESIS DEFERRED] Field: '{}' | 요약 필드는 이 목록의 추출이 끝난 뒤 요약 패스에서 만듭니다. 값이 모두 확정된 아이템의 값표를 쓰고, 추출용 디코딩 제약 없이 요약에 맞는 엔진으로 생성합니다.",
                            field_name
                        ));
                        continue;
                    }

                    let active_synthesis_gate = if qwen3_ready {
                        if field_lang.is_some() { synthesis_gate_lang } else { synthesis_gate_base }
                    } else if list_route.is_some() {
                        synthesis_gate_lang
                    } else {
                        synthesis_gate_base
                    };
                    if field_is_analytic[f_idx] && active_synthesis_gate.is_some() {
                        let last_chance = idx + 1 == total_items && synthesis_probe_item.is_none();
                        if synthesis_probe_item.map_or(qwen3_ready || last_chance, |p| p == idx) {
                            synthesis_probe_item = Some(idx);
                            emit_term(&format!(
                                "    🔬 [SYNTHESIS PROBE] Field: '{}' | 게이트가 닫힌 스코프에서도 목록마다 아이템 하나(Qwen3 가 이미 올라온 첫 아이템, 없으면 마지막 아이템)에서는 관측용으로 요약을 만듭니다. 결과는 같은 원문 대조 게이트를 거쳐 SDS 에 쌓이므로, 요약 품질이 좋아지면 이 게이트는 스스로 다시 열립니다.",
                                field_name
                            ));
                        } else {
                            emit_term(&format!(
                                "    ⏭️ [SYNTHESIS SKIP / SDS] Field: '{}' | 이 스코프에서 원문에 없는 사실로 폐기되는 비율이 기준을 넘은 요약 축이라 LLM 을 부르지 않고 비워 둡니다.",
                                field_name
                            ));
                            crate::utils::score_dynamics::record_baseline("commerce.synthesis_skip", 1.0);
                            continue;
                        }
                    }
                    let synthesis_sheet: Option<String> = if field_is_analytic[f_idx] {
                        crate::utils::ai_utils::synthesis_value_sheet(&item_val, &page_type, &doc_lang)
                    } else {
                        None
                    };
                    let targeted_pug = match synthesis_sheet.as_ref() {
                        Some(sheet) => sheet.clone(),
                        None => filtered_full_item_pug.clone(),
                    };

                    if field_is_analytic[f_idx] {
                        match synthesis_sheet.as_ref() {
                            Some(sheet) => {
                                crate::utils::score_dynamics::record_baseline("commerce.synthesis_sheet_fields", sheet.lines().count() as f32);
                                emit_term(&format!("    🧠 [SYNTHESIS FIELD / VALUE SHEET] Field: '{}' | 이 아이템에서 이미 확정된 값 {}개만 요약 재료로 넘깁니다. 원문 PUG 를 넘기면 회원정보수정·메일보내기 같은 UI 액션 구와 다른 칸의 날짜·금액이 요약에 섞여 들어옵니다. (HeaderOwn: {:.4})", field_name, sheet.lines().count(), best_thead_own));
                            }
                            None => {
                                emit_term(&format!("    🧠 [SYNTHESIS FIELD] Field: '{}' | 확정된 값이 2개 미만이라 전체 아이템 컨텍스트 요약 모드 (HeaderOwn: {:.4})", field_name, best_thead_own));
                            }
                        }
                    } else if has_vector_match {
                        emit_term(&format!("    🎯 [MATCHED CONTEXT] Field: '{}' ({:?}) | Line: {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}", field_name, field_format, best_item_idx + 1, best_item_raw, best_item_contrast, best_item_margin));
                    } else {
                        emit_term(&format!("    ⚠️ [NO CONFIDENT MATCH] Field: '{}' ({:?}) | 벡터 힌트 없이 전체 구조만 전달 (HeaderContrast: {:+.4})", field_name, field_format, best_thead_score));
                    }
                    
                    let mut final_context_str = format!("[JSON CONTEXT]\n{}", targeted_pug);

                    if field_name.contains("link") || field_name.contains("id") {
                        let mut link_cands: Vec<String> = Vec::new();
                        if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                            for line in &item_lines_ref {
                                for cap in href_re.captures_iter(line) {
                                    if let Some(m) = cap.get(1) {
                                        let v = m.as_str().trim().to_string();
                                        if !v.is_empty() && !link_cands.contains(&v) { link_cands.push(v); }
                                    }
                                }
                            }
                        }
                        if link_cands.is_empty() {
                            final_context_str.push_str("\n\n[LINK CANDIDATES]\n(none)\nThere is NO link in this item. You MUST return null for the link key.");
                        } else {
                            final_context_str.push_str(&format!("\n\n[LINK CANDIDATES]\n{}\nThe link value MUST be copied EXACTLY from this list. Never invent a URL.", link_cands.join("\n")));
                        }
                    }

                    if field_name.contains("date") || field_name.contains("_at") {
                        let mut date_cands: Vec<String> = Vec::new();
                        if let Ok(date_re) = regex::Regex::new(r"\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?") {
                            for line in &item_lines_ref {
                                for m in date_re.find_iter(line) {
                                    let v = m.as_str().trim().to_string();
                                    if !date_cands.contains(&v) { date_cands.push(v); }
                                }
                            }
                        }
                        if date_cands.is_empty() {
                            final_context_str.push_str("\n\n[DATE CANDIDATES]\n(none)\nThere is NO date literal in this item. You MUST return null.");
                        } else {
                            final_context_str.push_str(&format!("\n\n[DATE CANDIDATES]\n{}\nThe answer MUST be one of these literals, copied character by character, or null.", date_cands.join("\n")));
                        }
                    }

                    if field_is_analytic[f_idx] {
                        
                        final_context_str.push_str("\n\n[SYNTHESIS FIELD NOTICE]\nThis field is NOT a value to copy. Read the WHOLE [JSON CONTEXT] above and write ONE short sentence that summarizes it. Never return a single cell value such as a bare number, a status word, a person name, or a branch name. If [JSON CONTEXT] is empty, return null.");
                    } else if has_vector_match {
                        let matched_line = if line_enriched_texts[best_item_idx].is_empty() {
                            item_lines_ref[best_item_idx].trim()
                        } else {
                            line_enriched_texts[best_item_idx].as_str()
                        };
                        final_context_str.push_str(&format!("\n\n[VECTOR MATCH RESULT]\nThe format gate and the embedding model EXCLUSIVELY assigned this field to the single line below (RawSim {:.4}, Contrast {:+.4}, Margin {:+.4}). No other column may use this line.\nThe part BEFORE '|' is the column LABEL, the part AFTER '|' is the VALUE. Copy ONLY the value part, character for character. Do NOT copy the label. If that value does not fit the schema, return null.\n\"{}\"", best_item_raw, best_item_contrast, best_item_margin, matched_line));
                        if !pre_mapped_context.is_empty() {
                            final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field:\n{}", pre_mapped_context));
                        }
                    } else if !pre_mapped_context.is_empty() {
                        
                        
                        final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field. If nothing else in [JSON CONTEXT] fits this field, return null:\n{}", pre_mapped_context));
                    }

                    let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: final_context_str,
                        name: None,
                    });
                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                    
                    let f_percent = (((f_idx as f32) / (total_fields as f32)) * 100.0) as i32;
                    let f_summary_msg = format!("Extracting {} ({}%)...", field_name, f_percent);
                    
                    let payload = json!({ 
                        "task_id": task.id, 
                        "category": format!("List Item {}/{}", idx + 1, total_items), 
                        "summary": f_summary_msg, 
                        "spinner": "⠋" 
                    });
                    log_task_progress(app_handle, &task.id, &payload);
                    emit_term(&format!("  ▶ {}", f_summary_msg));

                    let mut metadata_str = String::new();
                    let mut target_data_str = String::new();

                    for line in targeted_pug.lines() {
                        if let Some(idx) = line.find('|') {
                            metadata_str.push_str(line[..idx].trim());
                            metadata_str.push_str("\n");
                            target_data_str.push_str(line[idx + 1..].trim());
                            target_data_str.push_str("\n");
                        } else {
                            target_data_str.push_str(line.trim());
                            target_data_str.push_str("\n");
                        }
                    }

                    let metadata_str = metadata_str.trim();
                    let target_data_str = target_data_str.trim();

                    if !qwen3_ready {
                        vram_probe.phase(format!("item {}/{} · generator load", idx + 1, total_items));
                        if let Some(code) = list_route.as_deref() {
                            field_lang = model.enter_lang_generation(code, "list item extraction loop (first LLM field)").await;
                        }
                        if field_lang.is_none() {
                            model
                                .enter_generation_phase(
                                    crate::model::ModelSize::Qwen3,
                                    None,
                                    Some(cancellation_token.clone()),
                                    false,
                                    Some("inference".to_string()),
                                    "list item extraction loop (first LLM field)",
                                )
                                .await?;
                        }
                        let engine_label = field_lang.as_ref().map(|s| s.label()).unwrap_or_else(|| "Qwen3".to_string());
                        emit_term(&format!(
                            "  💡 [LAZY GENERATOR] Item {}/{} · Field '{}' 에서 처음으로 LLM 이 필요해 {} 를 올렸습니다. {}",
                            idx + 1, total_items, field_name, engine_label, model.crossover_report()
                        ));
                        qwen3_ready = true;
                    }
                    let mut task_question = if field_name.contains("status") {
                        parsing::extract_status_intent_legacy_prompt(&targeted_pug, &page_type, &bias_target)
                    } else if field_is_analytic[f_idx] {
                        if field_lang.is_some() {
                            parsing::extract_synthesis_field_prompt_native(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                        } else {
                            parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                        }
                    } else {
                        parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                    };
                    let field_keys: Vec<String> = field_name.split(',').map(|s| s.trim().to_string()).collect();

                    let mut ignore_list: Vec<String> = global_ignore_list.clone();
                    let mut miss_counter = 0;
                    vram_probe.phase(format!("item {}/{} · field '{}'", idx + 1, total_items, field_name));
                    
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }

                        let field_name_clone = field_name.clone();
                        let used_lang = field_lang.is_some();
                        let res = if let Some(sess) = field_lang.as_ref() {
                            let user_msg = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                                name: None,
                            });
                            model
                                .generate_lang_json(
                                    sess.code(),
                                    vec![system_message.clone(), user_msg],
                                    512,
                                    0.0,
                                    Some(ignore_list.as_slice()),
                                    &field_keys,
                                    Some(cancellation_token.clone()),
                                )
                                .await
                                .map(|(text, fixed)| {
                                    if !fixed.is_empty() {
                                        crate::utils::score_dynamics::record_baseline("commerce.lang4b_key_repair", fixed.len() as f32);
                                        emit_term(&format!("    🩹 [LANG4B KEY REPAIR] {:?} — 어휘 16,384 개인 4B 가 영문 스키마 키를 한두 글자 틀리게 옮긴 것을 철자 거리로 되돌립니다. 값은 아래 검증을 그대로 거칩니다.", fixed));
                                    }
                                    text
                                })
                        } else {
                            let q3_gen = model.qwen3_generator.clone();
                            let cancel_clone = cancellation_token.clone();
                            let sys_msg = system_message.clone();
                            let prejudice_target_for_closure = dynamic_prej_str.clone();
                            let task_q = task_question.clone();
                            let ignore_list_clone = ignore_list.clone();
                            let res = tokio::task::spawn_blocking(move || {
                                let mut gen_guard = q3_gen.blocking_lock();
                                if let Some(gen) = gen_guard.as_mut() {
                                    let params = ChatCompletionParameters {
                                        messages: vec![
                                            sys_msg,
                                            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                                content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                                name: None,
                                            })
                                        ],
                                        model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                        ..Default::default()
                                    };

                                    let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };

                                    gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                                } else {
                                    Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                                }
                            }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));
                            let q3_clear_arc = model.qwen3_generator.clone();
                            let dev_for_sync = if model.is_cpu_mode {
                                None
                            } else {
                                Some(model.device_config.device.clone())
                            };
                            let _ = tokio::task::spawn_blocking(move || {
                                if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                                    gen.clear_kv_cache();
                                }
                                if let Some(dev) = dev_for_sync {
                                    if dev.is_cuda() { let _ = dev.synchronize(); }
                                }
                            }).await;
                            res
                        };
                        if miss_counter > 0 {
                            vram_probe.phase(format!(
                                "item {}/{} · field '{}' · retry {}",
                                idx + 1, total_items, field_name, miss_counter
                            ));
                        }
                        if used_lang {
                            if let Err(e) = &res {
                                emit_term(&format!("    ⚠️ [LANG4B FALLBACK] Field '{}' | 4B 생성이 실패해 이 태스크의 남은 필드는 Qwen3 로 진행합니다: {}", field_name, e));
                                crate::model::lang_llm::record_route_outcome("commerce", "list_field", false);
                                field_lang = None;
                                if field_is_analytic[f_idx] && !field_name.contains("status") {
                                    task_question = parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str);
                                }
                                model
                                    .enter_generation_phase(
                                        crate::model::ModelSize::Qwen3,
                                        None,
                                        Some(cancellation_token.clone()),
                                        false,
                                        Some("inference".to_string()),
                                        "list item extraction loop (4B fallback)",
                                    )
                                    .await?;
                                continue;
                            }
                        }

                        match res {
                            Ok(res_text) => {
                                let mut parsed = parsing::parse_json_from_llm(&res_text);
                                let mut parsed_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed };

                                
                                if let Some(obj) = parsed_val.as_object_mut() {
                                    let ks: Vec<String> = obj.keys().cloned().collect();
                                    for k in ks {
                                        let cleaned = match obj.get(&k) {
                                            Some(serde_json::Value::String(s)) => Some(strip_markup_prefix(s)),
                                            _ => None,
                                        };
                                        if let Some(c) = cleaned {
                                            obj.insert(k, json!(c));
                                        }
                                    }
                                }

                                let mut requires_retry = false;
                                let mut extracted_values_for_retry = Vec::new();
                                
                                let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                                let mut found_valid_value = false;

                                let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                                
                                
                                let synthesis_fields = ["insight", "summary", "analysis"];
                                let field_name_lower = field_name_clone.to_lowercase();
                                let is_synthesis_field = synthesis_fields.iter().any(|&f| field_name_lower.contains(f));
                                let is_enum_field = is_synthesis_field || skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                                let is_placeholder_str = |s: &str| -> bool {
                                    let t = s.trim();
                                    if t.is_empty() { return true; }
                                    let lower = t.to_lowercase();
                                    if ["...", "null", "string", "number", "boolean", "n/a", "none", "undefined"].contains(&lower.as_str()) { return true; }
                                    let compact: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
                                    if ["yyyymmddthhmmss", "yyyymmddhhmmss", "yyyymmdd", "hhmmss"].contains(&compact.as_str()) { return true; }
                                    let ymd_only = !lower.is_empty() && lower.chars().all(|c| "ymdhms-t:./ ".contains(c));
                                    if ymd_only && lower.chars().any(|c| c == 'y' || c == 'm' || c == 'd') { return true; }
                                    false
                                };

                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        let is_empty_val = match val {
                                            serde_json::Value::Null => true,
                                            serde_json::Value::String(s) => is_placeholder_str(s),
                                            serde_json::Value::Array(a) => a.is_empty(),
                                            serde_json::Value::Object(o) => o.is_empty(),
                                            _ => false,
                                        };
                                        if !is_empty_val {
                                            let extracted_str = if val.is_string() {
                                                val.as_str().unwrap_or("").trim().to_string()
                                            } else if val.is_number() {
                                                val.to_string()
                                            } else {
                                                String::new()
                                            };

                                            if is_synthesis_field && !extracted_str.is_empty() {
                                                let gate = crate::utils::ai_utils::synthesis_fact_gate(&extracted_str, &targeted_pug, &doc_lang);
                                                crate::utils::score_dynamics::record_baseline(
                                                    crate::model::lang_llm::synthesis_drop_axis(used_lang),
                                                    if gate.total == 0 { 0.0 } else { gate.dropped.len() as f32 / gate.total as f32 },
                                                );
                                                crate::model::lang_llm::record_engine_outcome("commerce", "list_field", used_lang, gate.dropped.is_empty());
                                                if !gate.dropped.is_empty() {
                                                    emit_term(&format!(
                                                        "    🧪 [SYNTHESIS GROUNDING] '{}' | 문장 {}개 중 {}개를 뺍니다: {:?} | 남은 문장: \"{}\" — 요약 문장의 연도·날짜·금액·통화는 아이템 원문에 그대로 있어야 합니다. 원문에 없는 사실이 text·masked_text 로 들어가면 FTS 와 청크 코사인이 그 사실로 이 문서를 회수합니다.",
                                                        k,
                                                        gate.total,
                                                        gate.dropped.len(),
                                                        gate.dropped.iter().map(|(_, why)| why.join("·")).collect::<Vec<_>>(),
                                                        gate.kept
                                                    ));
                                                    if let Some(o) = parsed_val.as_object_mut() {
                                                        if gate.kept.is_empty() {
                                                            o.remove(*k);
                                                        } else {
                                                            o.insert(k.to_string(), json!(gate.kept));
                                                        }
                                                    }
                                                    found_valid_value = true;
                                                    continue;
                                                }
                                            }

                                            let key_fmt = detect_field_format(k);
                                            let strict_post = matches!(
                                                key_fmt,
                                                FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Text
                                                    | FieldFormat::Numeric | FieldFormat::Enum | FieldFormat::Identifier
                                                    | FieldFormat::Phone | FieldFormat::Address
                                            );
                                            if strict_post && !extracted_str.is_empty() && !value_matches_format(key_fmt, &extracted_str) {
                                                emit_term(&format!("    🚫 [FORMAT REJECT] '{}' ({:?}) 에 형식 불일치 값 '{}' 반환. 폐기 후 재시도합니다.", k, key_fmt, extracted_str));
                                                crate::utils::score_dynamics::record_field_seen(k);
                                                crate::utils::score_dynamics::record_field_reject(
                                                    k,
                                                    crate::utils::score_dynamics::GateKind::Format,
                                                );
                                                requires_retry = true;
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                continue;
                                            }
                                            let enum_reject = if key_fmt == FieldFormat::Enum { enum_value_reject(k, &extracted_str) } else { None };
                                            if let Some(why) = enum_reject {
                                                crate::utils::score_dynamics::record_field_seen(k);
                                                crate::utils::score_dynamics::record_field_reject(
                                                    k,
                                                    crate::utils::score_dynamics::GateKind::Enum,
                                                );
                                                let amount_answer = k.to_lowercase().contains("currency") && currency_amount_like(&extracted_str);
                                                let prior_amount = ignore_list.iter().any(|s| currency_amount_like(s) && !global_ignore_list.contains(s));
                                                if amount_answer && prior_amount {
                                                    emit_term(&format!("    💱 [CURRENCY AMOUNT STOP] '{}' 에 금액 '{}' 반환 ({}). 이 필드에서 금액 답을 이미 한 번 버렸는데 다시 금액이 와서, 이 아이템에는 통화 표지가 없다고 보고 재시도를 멈춥니다. 값은 비워 두고, 저장 직전 문서 언어 기본 통화가 들어갑니다.", k, extracted_str, why));
                                                    crate::utils::score_dynamics::record_baseline("commerce.currency_amount_stop", 1.0);
                                                    if let Some(o) = parsed_val.as_object_mut() {
                                                        o.remove(*k);
                                                    }
                                                    found_valid_value = true;
                                                    continue;
                                                }
                                                emit_term(&format!("    🚫 [ENUM VALUE GATE] '{}' 에 '{}' 반환 ({}). 열거형 값이 될 수 없어 폐기 후 재시도합니다.", k, extracted_str, why));
                                                requires_retry = true;
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                continue;
                                            }

                                            if k.to_lowercase().contains("currency") {
                                                if let Some(ev) = currency_ev.as_ref() {
                                                    if !crate::utils::ai_utils::currency_answer_admitted(&extracted_str, ev) {
                                                        emit_term(&format!("    🚫 [CURRENCY UNGROUNDED] '{}' 에 '{}' 반환. 이 아이템 라인의 통화 표지({})로 뒷받침되지 않아 채우지 않습니다. 재시도하지 않고 저장 직전 문서 언어 기본 통화에 맡깁니다.", k, extracted_str, ev.describe()));
                                                        crate::utils::score_dynamics::record_baseline("commerce.currency_ungrounded", 1.0);
                                                        if let Some(o) = parsed_val.as_object_mut() {
                                                            o.remove(*k);
                                                        }
                                                        found_valid_value = true;
                                                        continue;
                                                    }
                                                }
                                            }

                                            found_valid_value = true;

                                            if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                
                                                if !is_enum_field {
                                                    let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                    let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                    let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                    
                                                    if !is_iso_date && !is_url && !is_boolean_str {
                                                        let mut is_matched = doc_title.contains(&extracted_str);
                                                        
                                                        if !is_matched {
                                                            let extracted_lower = extracted_str.to_lowercase();
                                                            let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            
                                                            for ctx_val in &json_contexts {
                                                                if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                    let target_lower = target_val_str.to_lowercase();
                                                                    
                                                                    if target_lower.contains(&extracted_lower) {
                                                                        if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                            let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                            if tokens.contains(&extracted_lower.as_str()) {
                                                                                is_matched = true;
                                                                                break;
                                                                            }
                                                                        } else {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    
                                                                    if !is_matched && digits_only.len() >= 3 {
                                                                        let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                        if target_digits.contains(&digits_only) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        if !is_matched {
                                                            requires_retry = true;
                                                        } else if !doc_title.contains(&extracted_str)
                                                            && header_plausible.get(*k).map_or(false, |s| !s.is_empty())
                                                        {
                                                            let lower_p = extracted_str.to_lowercase();
                                                            let digits_p: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            let short_digits = !digits_p.is_empty() && digits_p.len() < 3 && extracted_str.len() == digits_p.len();
                                                            let hosts: Vec<Option<&str>> = json_contexts
                                                                .iter()
                                                                .filter(|ctx| {
                                                                    let raw = ctx.get("value").and_then(|x| x.as_str()).unwrap_or("");
                                                                    let t = raw.to_lowercase();
                                                                    let direct = t.contains(&lower_p)
                                                                        && (!short_digits || t.split(|c: char| !c.is_alphanumeric()).any(|w| w == lower_p));
                                                                    let by_digits = digits_p.len() >= 3
                                                                        && raw.chars().filter(|c| c.is_ascii_digit()).collect::<String>().contains(&digits_p);
                                                                    direct || by_digits
                                                                })
                                                                .map(|ctx| ctx.get("metadata").and_then(|x| x.as_str()))
                                                                .collect();
                                                            let foreign_only = !hosts.is_empty()
                                                                && hosts.iter().all(|h| {
                                                                    crate::utils::ai_utils::header_judged_foreign(k, *h, &header_evaluated, &header_plausible)
                                                                });
                                                            crate::utils::score_dynamics::record_baseline(
                                                                "commerce.value_provenance_reject",
                                                                if foreign_only { 1.0 } else { 0.0 },
                                                            );
                                                            if foreign_only {
                                                                emit_term(&format!("    🚫 [VALUE PROVENANCE REJECT] '{}' = '{}' 는 이 아이템에서 헤더 {:?} 칸에만 있습니다. 그 헤더들은 전부 헤더 판정에서 '{}' 의 칸이 될 수 없다고 나왔고 이 필드가 가질 수 있는 칸은 따로 있습니다. 옆 칸의 값을 빌려 온 것이므로 폐기하고 재시도합니다.", k, extracted_str, hosts.iter().map(|h| h.unwrap_or("")).collect::<Vec<_>>(), k));
                                                                crate::utils::score_dynamics::record_field_seen(k);
                                                                crate::utils::score_dynamics::record_field_reject(
                                                                    k,
                                                                    crate::utils::score_dynamics::GateKind::Prejudice,
                                                                );
                                                                requires_retry = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if !found_valid_value {
                                    requires_retry = true;
                                }
                                if !is_synthesis_field || requires_retry {
                                    crate::model::lang_llm::record_engine_outcome("commerce", "list_field", used_lang, !requires_retry);
                                }

                                if requires_retry {
                                    miss_counter += 1;
                                    crate::utils::score_dynamics::record_baseline("commerce.llm_retry", miss_counter as f32);
                                    if miss_counter > 3 {
                                        emit_term(&format!("    ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                        crate::utils::score_dynamics::record_field_seen(&field_name_clone);
                                        crate::utils::score_dynamics::record_field_reject(
                                            &field_name_clone,
                                            crate::utils::score_dynamics::GateKind::Enum,
                                        );
                                        break; 
                                    }
                                    emit_term(&format!("    ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                    crate::utils::score_dynamics::record_near_miss(&field_name_clone);
                                    for ex_str in extracted_values_for_retry {
                                        ignore_list.push(ex_str.clone());
                                        ignore_list.push(format!(" {}", ex_str));
                                        ignore_list.push(ex_str.to_lowercase());
                                    }
                                    if !found_valid_value {
                                        for k in &keys {
                                            ignore_list.push(format!("\"{}\": \"\"", k));
                                            ignore_list.push(format!("\"{}\":\"\"", k));
                                        }
                                    }
                                    continue;
                                }

                                let shareable_field = ["name", "insight", "status", "payment_method", "date", "_at", "currency", "goods", "title"]
                                    .iter().any(|f| field_name_clone.contains(f));

                                let mut extracted_results = Vec::new();
                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        item_val.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                        extracted_results.push(format!("\"{}\": {}", k, val));
                                        
                                        let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                      else if val.is_number() { val.to_string() }
                                                      else { String::new() };
                                        
                                        if !shareable_field && val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                            if !global_ignore_list.contains(&val_str) {
                                                global_ignore_list.push(val_str.clone());
                                                global_ignore_list.push(format!(" {}", val_str));
                                                global_ignore_list.push(val_str.to_lowercase());
                                            }
                                        }
                                    }
                                }
                                


                                for ck in ["has_header", "has_footer", "language"] {
                                    if let Some(val) = parsed_val.get(ck) {
                                        item_val.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                    }
                                }

                                if !extracted_results.is_empty() {
                                    emit_term(&format!("    ✅ Extracted: {}", extracted_results.join(", ")));
                                } else {
                                    emit_term(&format!("    ✅ Extracted: (null or empty for {})", field_name_clone));
                                }
                                break;
                            },
                            Err(e) => {
                                println!("[Scheduler] Error extracting list item field {}: {:?}", field_name_clone, e);
                                break;
                            }
                        }
                    }
                }


                let mut temp_id = item_val.get("id").and_then(|v| if v.is_string() { v.as_str().map(|s| s.to_string()) } else { Some(v.to_string()) }).unwrap_or_default();
                let mut temp_code = item_val.get("code").and_then(|v| if v.is_string() { v.as_str().map(|s| s.to_string()) } else { Some(v.to_string()) }).unwrap_or_default();
                

                if !temp_id.is_empty() || !temp_code.is_empty() {
                    let mut url_pool = String::new();
                    if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                        for line in &item_lines_ref {
                            for cap in href_re.captures_iter(line) {
                                if let Some(m) = cap.get(1) {
                                    url_pool.push_str(&m.as_str().to_lowercase());
                                    url_pool.push_str(" ");
                                }
                            }
                        }
                    }
                    
                    let id_in_url = !temp_id.is_empty() && url_pool.contains(&temp_id.to_lowercase());
                    let code_in_url = !temp_code.is_empty() && url_pool.contains(&temp_code.to_lowercase());

                    if !id_in_url && code_in_url {
                        let swap = temp_id.clone();
                        temp_id = temp_code.clone();
                        temp_code = swap;
                        emit_term("  🔄 [DEV-LOGIC] Swapped 'id' and 'code' based on URL presence in PUG.");
                    } else if !temp_id.is_empty() && !id_in_url {
                        if temp_code.is_empty() {
                            temp_code = temp_id.clone();
                        }
                        temp_id = String::new();
                        emit_term("  🔄 [DEV-LOGIC] Moved 'id' to 'code' because it was NOT found in any URL link.");
                    }
                }

                if !temp_id.is_empty() {
                    let extracted = if let Some(idx) = temp_id.rfind('=') {
                        &temp_id[idx + 1..]
                    } else {
                        &temp_id
                    };
                    let clean_str = extracted.replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                    if !clean_str.is_empty() {
                        item_val.as_object_mut().unwrap().insert("id".to_string(), json!(clean_str.trim()));
                    } else {
                        item_val.as_object_mut().unwrap().remove("id");
                    }
                } else {
                    item_val.as_object_mut().unwrap().remove("id");
                }

                if !temp_code.is_empty() {
                    item_val.as_object_mut().unwrap().insert("code".to_string(), json!(temp_code.trim()));
                } else {
                    item_val.as_object_mut().unwrap().remove("code");
                }

                if !item_val.is_null() && (item_val.is_object() || item_val.is_array()) {
                    if let Some(link_val) = item_val.get_mut("link") {
                        if let Some(relative_path) = link_val.as_str() {
                            if let Ok(base_url) = url::Url::parse(&url) {
                                if let Ok(absolute_url) = base_url.join(relative_path) {
                                    let path_query = format!("{}{}", absolute_url.path(), absolute_url.query().map(|q| format!("?{}", q)).unwrap_or_default());
                                    *link_val = json!(path_query.to_lowercase());
                                }
                            }
                        }
                    }
                    
                    emit_term(&format!("  ✅ Successfully Merged Extracted Item {}/{}: {}", idx + 1, total_items, serde_json::to_string(&item_val).unwrap_or_default()));
                    synthesis_item_pos.insert(idx, all_extracted_items.len());
                    all_extracted_items.push(item_val);
                    
                    all_item_raw_lines.push(item_lines.clone());
                    
                    
                    let labeled_snapshot: Vec<String> = (0..item_lines.len()).map(|li| {
                        if !line_enriched_texts[li].is_empty() {
                            line_enriched_texts[li].clone()
                        } else {
                            item_lines[li].clone()
                        }
                    }).collect();
                    all_item_labeled_lines.push(labeled_snapshot);
                }
                
                crate::models::qwen::generate::wait_for_global_io().await;

                // 🌟 [PEAK SAMPLER] 아이템 경계마다 누적 최저점을 보고합니다.
                //    값이 아이템마다 계단식으로 내려가면 '누적',
                //    특정 아이템에서만 한 번 크게 내려가면 '전이' 입니다.
                //    이 한 줄로 두 가설이 갈립니다.
                {
                    let (base, low, worst) = vram_probe.snapshot();
                    emit_term(&format!(
                        "  📉 [VRAM TRACE] item {}/{} 종료 | 진입 {}MB → 누적 최저 {}MB (-{}MB) | 최저 단계 '{}' | 현재 {}MB",
                        idx + 1, total_items, base, low, base.saturating_sub(low), worst,
                        model.get_free_vram_mb()
                    ));
                }
                // 🌟 [SDS] 아이템 경계에서 관측을 확정합니다.
                //
                //  ── 왜 필요한가 ──
                //   아이템 13개 × 필드 14개는 수 분이 걸립니다.
                //   태스크 말미의 flush 는 전부 끝나야 나오므로,
                //   중간에 취소되거나 앱이 멈추면 그동안의 관측이 전부 사라집니다.
                //   아이템 경계는 자연스러운 커밋 지점이고,
                //   dirty 가 아니거나 최소 간격(5초) 미만이면 실제 쓰기는 생략됩니다.
                crate::utils::score_dynamics::flush();
            }

            drop(field_lang);
            run_synthesis_pass(
                &model,
                &mut all_extracted_items,
                &synthesis_jobs,
                &synthesis_item_pos,
                &page_type,
                &doc_lang,
                &lang_probe_text,
                SYNTHESIS_SKIP_DROP,
                cancellation_token,
                &emit_term,
                "list_synthesis",
            )
            .await?;

            {
                let total_extracted_items = all_extracted_items.len();
                let mut retry_count = 0usize;
                let mut reject_count = 0usize;
                let id_census = IdRoleCensus::build(&all_item_raw_lines);
                let self_aliases = crate::logic::relay_type_aliases(&page_type);
                let mut unique_rescued = 0usize;
                let normalize_list_link = |raw: &str| -> String {
                    match url::Url::parse(&url).ok().and_then(|base| base.join(raw).ok()) {
                        Some(abs) => format!(
                            "{}{}",
                            abs.path(),
                            abs.query().map(|q| format!("?{}", q)).unwrap_or_default()
                        )
                        .to_lowercase(),
                        None => raw.to_string(),
                    }
                };

                for item_idx in 0..total_extracted_items {
                    let (has_id, has_link) = {
                        let iv = &all_extracted_items[item_idx];
                        (
                            iv.get("id").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty()),
                            iv.get("link").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty()),
                        )
                    };
                    if has_id && has_link { continue; }

                    let mut recovered: Option<(String, String, String)> = None; 

                    
                    if let Some(raw_lines) = all_item_raw_lines.get(item_idx) {
                        let raw_refs: Vec<&str> = raw_lines.iter().map(|s| s.as_str()).collect();
                        let cands = collect_id_link_candidates(&raw_refs);

                        if !cands.is_empty() && !idlink_label_embs.is_empty() {
                            let role_texts: Vec<String> = cands.iter().map(|c| c.role_phrase.clone()).collect();
                            let role_embs = model.get_embedding_batch(role_texts.clone()).await
                                .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                            let mut best = f32::MIN;
                            let mut best_i: Option<usize> = None;
                            for (ci, c) in cands.iter().enumerate() {
                                let emb = &role_embs[ci];
                                if emb.iter().all(|&v| v == 0.0) { continue; }
                                let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                                let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                                let score = (own - prej) + 0.15 * (c.prior - 1.0);
                                emit_term(&format!("      🧭 [RETRY HREF CANDIDATE] Item {}/{}: '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                                    item_idx + 1, total_extracted_items, c.token, c.role_phrase, own, prej, score));
                                if own < 0.30 { continue; }
                                if score <= 0.0 { continue; }
                                if score > best { best = score; best_i = Some(ci); }
                            }

                            if let Some(bi) = best_i {
                                let c = &cands[bi];
                                recovered = Some((
                                    c.token.clone(),
                                    c.href.clone(),
                                    format!("href 코사인 재채점 (역할 '{}', Score {:+.4})", c.role_phrase, best),
                                ));
                            }
                        }
                    }

                    
                    if recovered.is_none() {
                        if let Some(raw_lines) = all_item_raw_lines.get(item_idx) {
                            let raw_refs: Vec<&str> = raw_lines.iter().map(|s| s.as_str()).collect();
                            let cands = collect_id_link_candidates(&raw_refs);
                            let small_list = id_census.rows < 3;
                            let mut ranked: Vec<(usize, f32)> = cands
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| {
                                    !c.is_host_part
                                        && (id_census.row_unique(&c.role_phrase)
                                            || (small_list && candidate_type_evidence(c, self_aliases) >= 2))
                                })
                                .map(|(ci, c)| (ci, id_resource_affinity(&url, c, self_aliases) + 0.01 * c.prior))
                                .collect();
                            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                            if let Some(&(bi, top)) = ranked.first() {
                                let contested = ranked.iter().skip(1).any(|(ci, s)| {
                                    (top - *s).abs() < 1e-6 && !same_id_token(&cands[*ci].token, &cands[bi].token)
                                });
                                if contested {
                                    crate::utils::score_dynamics::record_baseline("commerce.idlink_unique_contested", 1.0);
                                    emit_term(&format!(
                                        "      ⚖️ [ID/LINK ROW-UNIQUE CONTESTED] Item {}/{}: 행마다 값이 다른 식별 후보가 둘 이상이고 자기 자원 근거가 같아 고르지 않습니다. 틀린 자기 식별자는 서로 다른 문서를 한 행으로 합칩니다.",
                                        item_idx + 1, total_extracted_items
                                    ));
                                } else {
                                    let c = &cands[bi];
                                    let (present, distinct) = id_census.coverage(&c.role_phrase);
                                    unique_rescued += 1;
                                    let reason = if id_census.row_unique(&c.role_phrase) {
                                        crate::utils::score_dynamics::record_baseline("commerce.idlink_unique_rescue", 1.0);
                                        format!(
                                            "행 고유 식별자 (역할 '{}' · {}행 중 {}행 보유 · 값 {}종 전부 상이 · 자기 자원 근거 {:.2})",
                                            c.role_phrase, id_census.rows, present, distinct, top
                                        )
                                    } else {
                                        crate::utils::score_dynamics::record_baseline("commerce.idlink_small_list_rescue", 1.0);
                                        format!(
                                            "소량 목록 타입 식별자 (역할 '{}' · {}행이라 행 고유성은 판정할 수 없지만 파라미터 키가 자기 타입 별칭을 지목 · 자기 자원 근거 {:.2})",
                                            c.role_phrase, id_census.rows, top
                                        )
                                    };
                                    recovered = Some((c.token.clone(), normalize_list_link(&c.href), reason));
                                }
                            }
                        }
                    }

                    if recovered.is_none() {
                        if let Some((ref pat_prefix, ref pat_suffix)) = discovered_url_pattern {
                            let labeled = all_item_labeled_lines.get(item_idx).cloned().unwrap_or_default();
                            let cands = collect_labeled_token_candidates(&labeled);

                            let mut chosen: Option<(String, String, f32)> = None; 

                            if !cands.is_empty() && !idlink_label_embs.is_empty() {
                                let label_texts: Vec<String> = cands.iter().map(|c| c.label_phrase.clone()).collect();
                                let label_embs = model.get_embedding_batch(label_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; label_texts.len()]);

                                for (ci, c) in cands.iter().enumerate() {
                                    if !id_shape_allowed(&c.token, &confirmed_id_shapes) {
                                        reject_count += 1;
                                        emit_term(&format!("      🚫 [SHAPE REJECT] Item {}/{}: 후보 '{}' 는 학습된 식별자 생김새와 달라 URL 대입을 거부했습니다.", item_idx + 1, total_extracted_items, c.token));
                                        continue;
                                    }

                                    let emb = &label_embs[ci];
                                    if emb.iter().all(|&v| v == 0.0) { continue; }
                                    let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                                    let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                                    let score = own - prej;

                                    emit_term(&format!("      🧭 [RECOVERY CANDIDATE] Item {}/{}: '{}' ← 라벨 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                                        item_idx + 1, total_extracted_items, c.token, c.label_phrase, own, prej, score));

                                    if own < 0.40 { continue; }
                                    if score <= 0.05 { continue; }
                                    let better = chosen.as_ref().map(|(_, _, s)| score > *s).unwrap_or(true);
                                    if better { chosen = Some((c.token.clone(), c.label_phrase.clone(), score)); }
                                }
                            }

                            
                            
                            if chosen.is_none() {
                                if let Some(raw_lines) = all_item_raw_lines.get(item_idx) {
                                    if let Some(tok) = find_identifier_token_in_lines(raw_lines) {
                                        if id_shape_allowed(&tok, &confirmed_id_shapes) {
                                            chosen = Some((tok, "legacy token scan".to_string(), 0.0));
                                        } else {
                                            reject_count += 1;
                                            emit_term(&format!("      🚫 [SHAPE REJECT] Item {}/{}: 레거시 토큰 '{}' 도 학습된 식별자 생김새와 달라 폐기했습니다.", item_idx + 1, total_extracted_items, tok));
                                        }
                                    }
                                }
                            }

                            if let Some((tok, label, score)) = chosen {
                                let link = apply_url_pattern(pat_prefix, pat_suffix, &tok);
                                let host_ok = pattern_reference_link.as_ref()
                                    .map(|r| same_host(r, &link))
                                    .unwrap_or(true);
                                if host_ok {
                                    recovered = Some((tok, link, format!("라벨 코사인 (라벨 '{}', Score {:+.4})", label, score)));
                                } else {
                                    reject_count += 1;
                                    emit_term(&format!("      🚫 [HOST REJECT] Item {}/{}: 재구성 링크 '{}' 의 호스트가 기준 링크와 달라 폐기했습니다.", item_idx + 1, total_extracted_items, link));
                                }
                            }
                        }
                    }

                    
                    if let Some((found_id, constructed_link, reason)) = recovered {
                        if let Some(obj) = all_extracted_items[item_idx].as_object_mut() {
                            obj.insert("id".to_string(), json!(found_id.clone()));
                            obj.insert("link".to_string(), json!(constructed_link.clone()));
                        }
                        retry_count += 1;
                        emit_term(&format!("  🔄 [ID/LINK RETRY] Item {}/{}: {} → \"id\": \"{}\", \"link\": \"{}\"", item_idx + 1, total_extracted_items, reason, found_id, constructed_link));
                    } else {
                        emit_term(&format!("  ⚪ [ID/LINK RETRY SKIP] Item {}/{}: 코사인 게이트를 통과한 식별자 후보가 없어 id/link 를 비워 둡니다. (잘못된 링크보다 빈 값이 안전합니다)", item_idx + 1, total_extracted_items));
                    }
                }

                {
                    let own_family = crate::utils::canonical::relay_type_family(&page_type);
                    let related_types = crate::logic::related(&page_type);
                    let relay_foreign: Vec<&str> = ["goods", "order", "tracking"]
                        .iter()
                        .copied()
                        .filter(|t| *t != own_family.as_str() && related_types.iter().any(|r| r == t))
                        .collect();
                    let mut verify_swapped = 0usize;
                    let mut relay_bound = 0usize;
                    for item_idx in 0..total_extracted_items {
                        let raw_lines = match all_item_raw_lines.get(item_idx) {
                            Some(l) => l,
                            None => continue,
                        };
                        let raw_refs: Vec<&str> = raw_lines.iter().map(|s| s.as_str()).collect();
                        let cands = collect_id_link_candidates(&raw_refs);
                        if cands.is_empty() {
                            continue;
                        }
                        let mut self_token = all_extracted_items[item_idx]
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if !self_token.is_empty() && id_census.rows >= 3 {
                            let chosen: Vec<&IdLinkCandidate> = cands
                                .iter()
                                .filter(|c| same_id_token(&c.token, &self_token))
                                .collect();
                            let chosen_unique = chosen.iter().any(|c| id_census.row_unique(&c.role_phrase));
                            if !chosen.is_empty() && !chosen_unique {
                                let chosen_aff = chosen
                                    .iter()
                                    .map(|c| id_resource_affinity(&url, c, self_aliases))
                                    .fold(0.0f32, f32::max);
                                let mut alt: Vec<(usize, f32)> = cands
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, c)| !c.is_host_part && id_census.row_unique(&c.role_phrase))
                                    .map(|(ci, c)| (ci, id_resource_affinity(&url, c, self_aliases)))
                                    .collect();
                                alt.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                if let Some(&(ai, a_aff)) = alt.first() {
                                    let contested = alt.iter().skip(1).any(|(ci, s)| {
                                        (a_aff - *s).abs() < 1e-6 && !same_id_token(&cands[*ci].token, &cands[ai].token)
                                    });
                                    if a_aff > chosen_aff && !contested {
                                        let c = &cands[ai];
                                        emit_term(&format!(
                                            "  🔁 [ID/LINK UNIQUE VERIFY] Item {}/{}: 코사인이 고른 '{}' 는 여러 행이 같은 값을 공유하는 역할이라 이 행 자신의 식별자가 아니라 다른 문서를 가리키는 참조입니다. 행 고유 역할 '{}' 의 '{}' 로 교체합니다. (자기 자원 근거 {:.2} > {:.2})",
                                            item_idx + 1, total_extracted_items, self_token, c.role_phrase, c.token, a_aff, chosen_aff
                                        ));
                                        if let Some(obj) = all_extracted_items[item_idx].as_object_mut() {
                                            obj.insert("id".to_string(), json!(c.token.clone()));
                                            obj.insert("link".to_string(), json!(normalize_list_link(&c.href)));
                                        }
                                        self_token = c.token.clone();
                                        verify_swapped += 1;
                                        crate::utils::score_dynamics::record_baseline("commerce.idlink_unique_swap", 1.0);
                                    }
                                }
                            }
                        }

                        if relay_foreign.is_empty() {
                            continue;
                        }
                        for (ftype, token, role) in collect_typed_relay_refs(&cands, &self_token, &relay_foreign) {
                            let key = match crate::utils::canonical::relay_key_for_type(&ftype) {
                                Some(k) => k,
                                None => continue,
                            };
                            let f_index = entity_key_index(&ftype, &team_id, &task.cc, &token);
                            if let Some(obj) = all_extracted_items[item_idx].as_object_mut() {
                                let companion = format!("{}_title", key);
                                let text_val = obj
                                    .get(key)
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| crate::utils::canonical::relay_text_is_content(s));
                                let companion_empty = obj
                                    .get(&companion)
                                    .and_then(|v| v.as_str())
                                    .map_or(true, |s| s.trim().is_empty());
                                if let Some(t) = text_val {
                                    if companion_empty {
                                        obj.insert(companion.clone(), json!(t));
                                    }
                                }
                                obj.insert(key.to_string(), json!(f_index));
                            }
                            relay_ledger::mark_relay_bound(&mut all_extracted_items[item_idx], key);
                            relay_bound += 1;
                            emit_term(&format!(
                                "  🔗 [RELAY REF BIND] Item {}/{}: {} ← '{}' (역할 '{}') → {}.{} = {} | 행에 인쇄된 상대 문서의 식별자로 상대 index 를 결정론으로 재현했습니다. 글자 값은 {}_title 로 옮겨 검색 문장은 그대로 유지합니다.",
                                item_idx + 1, total_extracted_items, ftype, token, role, page_type, key, f_index, key
                            ));
                        }
                    }
                    crate::utils::score_dynamics::record_baseline("commerce.relay_ref_bound", relay_bound as f32);
                    if unique_rescued > 0 || verify_swapped > 0 || relay_bound > 0 {
                        emit_term(&format!(
                            "  🧬 [ID/RELAY CENSUS] {}행 | 행 고유 역할 {:?} | 자기 식별자 복구 {}건 · 교체 {}건 | 상대 참조 결속 {}건",
                            id_census.rows,
                            id_census.roles.keys().filter(|r| id_census.row_unique(r)).collect::<Vec<_>>(),
                            unique_rescued, verify_swapped, relay_bound
                        ));
                    }
                }

                crate::utils::score_dynamics::record_baseline("commerce.idlink_retry_recovered", retry_count as f32);
                crate::utils::score_dynamics::record_baseline("commerce.idlink_retry_rejected", reject_count as f32);
                if retry_count > 0 || reject_count > 0 {
                    emit_term(&format!("  🔄 [ID/LINK RETRY SUMMARY] 복구 {}개 | 생김새·호스트 게이트 거부 {}개.", retry_count, reject_count));
                }
            }
        }

        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });

    } else {

        println!("[Scheduler] Starting DISK BRIDGE RELAY for Details");
        
        let content_pug = {
            let clean_content = &clean_html_content;
            let full_pug = parsing::convert_to_clean_pug(clean_content, PugMode::DetailMode, Some(&url));
            model.truncate_pug_context(&full_pug, true, 2000, None).await
        };

        if !content_pug.trim().is_empty() {
            // 🌟 [CROSSOVER / ORDER FIX] 리스트 경로와 같은 이유로 Qwen3 로드를 미룹니다.
            //
            //  이 아래로 이어지는 구간은 전부 임베딩입니다.
            //    layout_prej_emb → line_embeddings → boa 블록 임베딩
            //    → field_phrase_embs / prejudice bank
            //    → detail_pairs 라벨·섹션 임베딩
            //    → ENUM SELECT 옵션 임베딩
            //  Qwen3 의 첫 실사용은 한참 뒤 '필드 추출 루프' 입니다.
            //  그 전까지 두 가중치를 겹쳐 둘 이유가 없습니다.
            model.enter_embedding_phase("detail vectorization").await?;
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            let fields = parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();
            let detail_route: Option<String> = {
                let verdict = model.lang_route("commerce", "detail_field", &doc_lang, &lang_probe_text).await;
                emit_term(&format!("  {}", verdict.line));
                verdict.code
            };
            let mut field_lang: Option<crate::model::lang_llm::LangSession> = None;

            let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": format!("Extracting {} detail fields sequentially...", total_fields), "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload);
            emit_term(&format!("[STAGE-3] Extracting {} detailed fields individually...", total_fields));


            let mut pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();

            
            
            
            
            let detail_pairs: Vec<DetailPair> = {
                let refs: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();
                collect_detail_label_value_pairs(&refs)
            };
            let (form_labeled, form_controls) = {
                let refs: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();
                detail_form_profile(&refs)
            };
            let form_page = form_controls >= 8 && form_controls * 10 >= form_labeled * 7;
            if form_labeled > 0 {
                crate::utils::score_dynamics::record_baseline(
                    "commerce.detail_form_control_ratio",
                    form_controls as f32 / form_labeled as f32,
                );
            }
            if form_page {
                emit_term(&format!(
                    "  📋 [DETAIL FORM PAGE] 라벨 칸 {}개 중 {}개가 입력 컨트롤 칸입니다 ({:.0}%). 이 상세는 레코드를 입력 폼으로 보여 주므로, 라벨 칸에서 값을 얻지 못한 글자(Text) 필드는 폼 밖의 메뉴 · 도움말 · 다른 칸 글자에서 LLM 으로 찾지 않습니다.",
                    form_labeled,
                    form_controls,
                    form_controls as f32 * 100.0 / form_labeled.max(1) as f32
                ));
            }

            
            
            
            let mut line_enriched_texts: Vec<String> = vec![String::new(); pug_lines.len()];
            for p in &detail_pairs {
                if p.primary_line < line_enriched_texts.len() && line_enriched_texts[p.primary_line].is_empty() {
                    line_enriched_texts[p.primary_line] = format!("{} | {}", p.label, p.value);
                }
            }
            for p in &detail_pairs {
                emit_term(&format!(
                    "  🧷 [DETAIL PAIR] Line {} | Section: '{}' | Label: '{}' | Value: '{}'",
                    p.primary_line + 1, p.section, p.label, p.value
                ));
            }

            let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
            

            let mut texts_to_embed = Vec::new();
            let mut text_indices = Vec::new();
            
            for (line_idx, line) in pug_lines.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                if line.trim().is_empty() { continue; }
                let target = if line_enriched_texts[line_idx].is_empty() {
                    line.to_string()
                } else {
                    line_enriched_texts[line_idx].clone()
                };
                texts_to_embed.push(target);
                text_indices.push(line_idx);
            }
            
            if !texts_to_embed.is_empty() {
                // 🌟 [CROSSOVER] 리스트 경로와 같은 이유로 고정 청킹을 제거합니다.
                //    여유 판정과 캐시는 get_embedding_batch 가 담당합니다.
                let vectors = model.get_embedding_batch(texts_to_embed.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; texts_to_embed.len()]);
                for (i, vector) in vectors.into_iter().enumerate() {
                    if i >= text_indices.len() { break; }
                    let original_idx = text_indices[i];
                    emit_term(&format!("  [VECTORIZING] Stage-3 Line {}/{} : {}", original_idx + 1, pug_lines.len(), texts_to_embed[i].trim()));
                    line_embeddings[original_idx] = vector;
                }
            }


            let (list_bias, form_bias, _) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            
            let mut wiped_indices = vec![false; pug_lines.len()];
            let mut processed_blocks = std::collections::HashSet::new();


            let nodes_str_detail = {
                let document_for_boa = scraper::Html::parse_document(&clean_html_content);
                let mut nodes_json = Vec::new();
                let mut node_to_idx = std::collections::HashMap::new();
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    node_to_idx.insert(node.id(), idx);
                }
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    if let Some(el) = node.value().as_element() {
                        let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                        let text: String = node.children()
                            .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                            .collect::<Vec<_>>().join(" ").trim().to_string();
                        nodes_json.push(serde_json::json!({
                            "index": idx,
                            "parentIndex": parent_idx,
                            "tagName": el.name().to_string(),
                            "id": el.id().unwrap_or("").to_string(),
                            "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                            "text": text,
                            "colspan": el.attr("colspan").unwrap_or("1"),
                            "rowspan": el.attr("rowspan").unwrap_or("1")
                        }));
                    } else {
                        nodes_json.push(serde_json::json!(serde_json::Value::Null));
                    }
                }
                serde_json::to_string(&nodes_json).unwrap_or_default()
            };
            
            let js_template_detail = get_boa_block_extractor_template();

            let mut track_a_candidates = Vec::new();
            let mut track_a_indices = Vec::new();
            let mut seen_detail_candidates = std::collections::HashSet::new();

            for line_idx in 0..pug_lines.len() {
                if wiped_indices[line_idx] { continue; }
                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&layout_prej_emb, &line_embeddings[line_idx]);
                
                if line_prej_score > 0.55 {
                    let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { line.trim() };
                    if !text_part.is_empty() && !seen_detail_candidates.contains(text_part) {
                        seen_detail_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                        track_a_indices.push(line_idx);
                    }
                }
            }


            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len();
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template_detail
                    .replace("NODES_PLACEHOLDER", &nodes_str_detail)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };


            let stage3_pugs: Vec<String> = {
                let html_clone = clean_html_content.clone();
                let selectors = track_a_selectors.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_stage3_sels = std::collections::HashSet::new();
                    let mut unique_sels = Vec::new();
                    for sel in selectors {
                        if sel.is_empty() { continue; }
                        if !seen_stage3_sels.contains(&sel) {
                            seen_stage3_sels.insert(sel.clone());
                            unique_sels.push(sel);
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        local_res.push(crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::DetailMode, None));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };


            let mut unique_stage3_pugs_to_embed = Vec::new();
            for block_pug in &stage3_pugs {
                if block_pug.is_empty() || processed_blocks.contains(block_pug) { continue; }
                processed_blocks.insert(block_pug.clone());
                unique_stage3_pugs_to_embed.push(block_pug.clone());
            }

            let mut stage3_embeddings_map = std::collections::HashMap::new();
            if !unique_stage3_pugs_to_embed.is_empty() {
                for chunk in unique_stage3_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            stage3_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for block_pug in stage3_pugs {
                if block_pug.is_empty() { continue; }
                let block_emb = stage3_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                
                let block_prej_score = cosine_similarity(&layout_prej_emb, &block_emb);
                let block_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                let block_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                
                if block_prej_score > block_list_score && block_prej_score > block_form_score {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [NOISE BLOCK DELETED] Boa Matched. Lines {}~{} (Prej: {:.4} > List: {:.4} & Form: {:.4})", start_idx + 1, end_idx + 1, block_prej_score, block_list_score, block_form_score));
                        for j in start_idx..=end_idx {
                            pug_lines[j] = String::new();
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            for line_idx in 0..pug_lines.len() {
                if !wiped_indices[line_idx] && !pug_lines[line_idx].trim().is_empty() {
                    emit_term(&format!("  [FILTERED PUG] Line {} : {}", line_idx + 1, pug_lines[line_idx].trim()));
                }
            }
            
            let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();


            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                };
                

                if t_val.is_empty() || t_val.len() < 5 {
                    let mut heading_texts = Vec::new();
                    if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                        for el in doc.select(&sel_h1) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                        for el in doc.select(&sel_h2) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if !heading_texts.is_empty() {
                        t_val = heading_texts.join(" | ");
                    }
                }
                t_val
            };


            let mut field_embeddings = Vec::new();
            
            
            let mut field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_phrase_weights: Vec<Vec<f32>> = Vec::new();
            
            let mut field_prej_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_is_analytic: Vec<bool> = Vec::new();
            let mut field_formats: Vec<FieldFormat> = Vec::new();

            for (f_idx, (fname, _, bias_target, predefined_prej)) in fields.iter().enumerate() {
                let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);

                let (phrases, phrase_weights) = split_bias_phrases_weighted(bias_target);
                let p_embs = if phrases.is_empty() {
                    vec![bias_emb.clone()]
                } else {
                    model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![bias_emb.clone(); phrases.len()])
                };
                let p_weights = if phrases.is_empty() { vec![1.0f32] } else { phrase_weights };
                field_phrase_embs.push(p_embs);
                field_phrase_weights.push(p_weights);

                let prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, fname);
                let prej_p_embs = if prej_phrases.is_empty() {
                    Vec::new()
                } else {
                    model.get_embedding_batch(prej_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
                };
                field_prej_phrase_embs.push(prej_p_embs);

                let detected_fmt = detect_field_format(fname);
                field_formats.push(detected_fmt);
                emit_term(&format!("  📐 [FORMAT REGISTERED] '{}' → {:?}", fname, detected_fmt));

                let lower_fname = fname.to_lowercase();
                let is_analytic = lower_fname.contains("insight")
                    || lower_fname.contains("summary")
                    || lower_fname.contains("analysis");
                field_is_analytic.push(is_analytic);
                if is_analytic {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD REGISTERED] '{}' 는 벡터 라인 매칭에서 제외되고 전체 컨텍스트 요약 필드로 처리됩니다.", fname));
                }

                let mut dynamic_prej_texts = Vec::new();
                if !predefined_prej.trim().is_empty() {
                    dynamic_prej_texts.push(predefined_prej.clone());
                }
                for (other_idx, (_, _, other_bias, _)) in fields.iter().enumerate() {
                    if f_idx != other_idx {
                        dynamic_prej_texts.push(other_bias.clone());
                    }
                }
                let combined_prej = dynamic_prej_texts.join(" , ");
                let prej_emb = model.get_embedding(combined_prej.clone()).await.unwrap_or(vec![0.0; 384]);

                field_embeddings.push((bias_emb, prej_emb, combined_prej));
            }


            let mut pre_mapped_hints = Vec::new();
            

            let mut url_pool = String::new();
            if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                for line in &pug_lines_ref {
                    for cap in href_re.captures_iter(line) {
                        if let Some(m) = cap.get(1) {
                            url_pool.push_str(&m.as_str().to_lowercase());
                            url_pool.push_str(" ");
                        }
                    }
                }
            }
            
            url_pool.push_str(&url.to_lowercase());
            url_pool.push_str(" ");

            
            let line_parts: Vec<(usize, String, String, String)> =
                pug_lines_ref.iter().map(|l| pug_line_parts(l)).collect();

            
            let line_values: Vec<String> = line_parts.iter().map(|p| p.3.clone()).collect();

            
            
            let control_noise = detail_control_noise_lines(&pug_lines_ref);
            if !control_noise.is_empty() {
                crate::utils::score_dynamics::record_baseline("commerce.detail_control_noise", control_noise.len() as f32);
                emit_term(&format!(
                    "  🎛️ [DETAIL CONTROL CELL] 입력 컨트롤이 값을 쥔 셀 안의 다른 라인 {}개(선택 안 된 옵션 · 체크 안 된 항목 · 버튼 · 도움말)를 값 후보에서 뺍니다. 이런 셀의 값은 컨트롤이 쥔 값(입력값 · 선택된 옵션 · 체크된 항목의 이름)뿐입니다.",
                    control_noise.len()
                ));
            }
            let line_is_non_value: Vec<bool> = line_parts
                .iter()
                .enumerate()
                .map(|(i, p)| is_non_value_role_tag(&p.1) || control_noise.contains(&i))
                .collect();
            
            let line_is_selected_option: Vec<bool> = line_parts.iter()
                .map(|p| p.1 == "option" && pug_attr_flag(&p.2, "selected"))
                .collect();

            
            let (idlink_label_phrases, idlink_label_weights) = label_phrase_bank(&doc_lang, &page_type, "id,link");
            let idlink_label_embs: Vec<Vec<f32>> = if idlink_label_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_label_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_label_phrases.len()])
            };
            let mut idlink_prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, "id,link");
            for extra in [
                "host name", "domain name", "website address", "server address",
                "cdn", "static asset", "image server", "protocol", "www",
                "file extension", "stylesheet", "script", "anchor", "javascript",
                "navigation menu", "layer popup", "delivery tracking service", "postal service",
            ] {
                let e = extra.to_string();
                if !idlink_prej_phrases.contains(&e) { idlink_prej_phrases.push(e); }
            }
            let idlink_prej_embs: Vec<Vec<f32>> = if idlink_prej_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_prej_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_prej_phrases.len()])
            };
            emit_term(&format!("  🔑 [ID/LINK COSINE BANK] 라벨 구 {}개 | 편견 구 {}개 준비 완료.", idlink_label_embs.len(), idlink_prej_embs.len()));

            let mut det_id_link: Option<(String, String)> = None;
            if let Some((tok, role, ev)) = url_typed_self_token(&url, &page_type) {
                let page_link = match url::Url::parse(&url) {
                    Ok(u) => format!("{}{}", u.path(), u.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase(),
                    Err(_) => url.clone(),
                };
                crate::utils::score_dynamics::record_baseline("commerce.detail_url_type_identity", 1.0);
                emit_term(&format!(
                    "  🔑 [PAGE-URL ID / TYPE EVIDENCE] 추출 주소의 '{}' (역할 '{}' · 타입 근거 {}) 를 이 상세 문서의 식별자로 먼저 확정합니다 → link '{}' | 주소 매개변수 이름이 이 문서 타입의 별칭이라 저장 단계의 URL 자기 식별자와 같은 근거입니다. 라벨 코사인이나 본문 링크(이미지 · 미리보기 버튼)로 정하면 목록 행의 link 를 덮고 식별자 문장도 어긋납니다.",
                    tok, role, ev, page_link
                ));
                det_id_link = Some((tok, page_link));
            }
            if det_id_link.is_none() {
                let url_cands = collect_id_link_candidates_from_url(&url);
                if !url_cands.is_empty() && !idlink_label_embs.is_empty() {
                    let role_texts: Vec<String> = url_cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best = f32::MIN;
                    let mut best_i: Option<usize> = None;
                    for (ci, c) in url_cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (c.prior - 1.0);
                        emit_term(&format!("      🧭 [PAGE-URL ID CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Prior: {:.2} | Score: {:+.4}",
                            c.token, c.role_phrase, own, prej, c.prior, score));
                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best { best = score; best_i = Some(ci); }
                    }

                    if let Some(bi) = best_i {
                        let c = &url_cands[bi];
                        let page_link = match url::Url::parse(&url) {
                            Ok(u) => format!("{}{}", u.path(), u.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase(),
                            Err(_) => url.clone(),
                        };
                        emit_term(&format!("  🔑 [PAGE-URL ID PRIORITY] 추출 주소에서 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'",
                            c.token, c.role_phrase, best, page_link));
                        det_id_link = Some((c.token.clone(), page_link));
                    }
                }
            }

            
            
            if det_id_link.is_none() && !idlink_label_embs.is_empty() {
                let cands: Vec<_> = collect_id_link_candidates(&pug_lines_ref)
                    .into_iter()
                    .filter(|c| {
                        if c.is_host_part { return false; }
                        if !same_host(&url, &c.href) {
                            return false;
                        }
                        true
                    })
                    .collect();

                if !cands.is_empty() {
                    let role_texts: Vec<String> = cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best = f32::MIN;
                    let mut best_i: Option<usize> = None;
                    for (ci, c) in cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (c.prior - 1.0);
                        emit_term(&format!("      🧭 [ID/LINK CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                            c.token, c.role_phrase, own, prej, score));
                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best { best = score; best_i = Some(ci); }
                    }

                    if let Some(bi) = best_i {
                        let c = &cands[bi];
                        emit_term(&format!("  🔑 [ID/LINK COSINE] 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'", c.token, c.role_phrase, best, c.href));
                        det_id_link = Some((c.token.clone(), c.href.clone()));
                    }
                }
            }

            
            if det_id_link.is_none() {
                det_id_link = resolve_id_link_from_lines(&pug_lines_ref);
                if let Some((fid, flink)) = &det_id_link {
                    emit_term(&format!("  🔑 [ID/LINK FALLBACK] 코사인 게이트 통과 후보가 없어 레거시 해석기로 확정: '{}' → '{}'", fid, flink));
                }
            }
            match &det_id_link {
                Some(_) => crate::utils::score_dynamics::record_field_assigned("id,link", 0.0),
                None => crate::utils::score_dynamics::record_field_reject(
                    "id,link",
                    crate::utils::score_dynamics::GateKind::Prejudice,
                ),
            }

            let mut det_consumed_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
            if let Some((det_id, _)) = &det_id_link {
                let whole_only = det_id.chars().count() < 4;
                for (l, v) in line_values.iter().enumerate() {
                    if v.is_empty() { continue; }
                    let matched = if whole_only {
                        v.trim().eq_ignore_ascii_case(det_id.as_str())
                    } else {
                        v.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| !tok.is_empty() && tok.eq_ignore_ascii_case(det_id.as_str()))
                    };
                    if matched { det_consumed_lines.insert(l); }
                }
            }

            let mut header_forced_assign: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut pair_owned_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
            
            let mut pair_line_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            if !detail_pairs.is_empty() {
                
                let mut label_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for p in &detail_pairs { *label_count.entry(p.label.clone()).or_insert(0) += 1; }

                let mut pair_phrases: Vec<String> = Vec::with_capacity(detail_pairs.len());
                for p in &detail_pairs {
                    let dup = label_count.get(&p.label).copied().unwrap_or(0) > 1;
                    if dup && !p.section.trim().is_empty() {
                        pair_phrases.push(format!("{} {}", p.section.trim(), p.label));
                    } else {
                        pair_phrases.push(p.label.clone());
                    }
                }

                
                
                
                
                let mut unique_phrases: Vec<String> = Vec::new();
                let mut unique_leaf: Vec<String> = Vec::new();
                let mut unique_section: Vec<String> = Vec::new();
                for (pi, ph) in pair_phrases.iter().enumerate() {
                    if unique_phrases.iter().any(|e| e == ph) { continue; }
                    unique_phrases.push(ph.clone());
                    unique_leaf.push(detail_pairs[pi].label.clone());
                    unique_section.push(detail_pairs[pi].section.trim().to_string());
                }

                let leaf_embs: Vec<Vec<f32>> = model.get_embedding_batch(unique_leaf.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; unique_leaf.len()]);
                let section_texts: Vec<String> = unique_section.iter()
                    .map(|s| if s.is_empty() { " ".to_string() } else { s.clone() })
                    .collect();
                let section_embs: Vec<Vec<f32>> = model.get_embedding_batch(section_texts.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; section_texts.len()]);

                let mut d_field_names: Vec<String> = Vec::new();
                let mut d_label_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut d_label_weights: Vec<Vec<f32>> = Vec::new();
                let mut d_prej_raw: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut d_prej_texts: Vec<Vec<String>> = Vec::new();

                for (fname, _, _, _) in &fields {
                    let (lp, lw) = label_phrase_bank(&doc_lang, &page_type, fname);
                    if lp.is_empty() { continue; }
                    let pp = prejudice_phrase_bank(&doc_lang, &page_type, fname);
                    let le = model.get_embedding_batch(lp.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; lp.len()]);
                    let pe = if pp.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(pp.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; pp.len()])
                    };
                    d_field_names.push(fname.clone());
                    d_label_embs.push(le);
                    d_label_weights.push(lw);
                    d_prej_raw.push(pe);
                    d_prej_texts.push(pp);
                }

                
                
                
                let mut d_prej_embs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(d_field_names.len());
                for f in 0..d_field_names.len() {
                    let mask = self_poisoned_prejudice_mask(&d_label_embs[f], &d_prej_raw[f], &d_label_embs, f);
                    let mut kept: Vec<Vec<f32>> = Vec::new();
                    let mut dropped = 0usize;
                    for (pi, poisoned) in mask.iter().enumerate() {
                        if *poisoned {
                            dropped += 1;
                            if dropped <= 6 {
                                emit_term(&format!("    🧪 [SELF-POISON DROP] '{}' 의 편견 구 '{}' 는 경쟁 필드보다 자기 자신을 더 잘 설명하므로 편견 자격을 박탈합니다.",
                                    d_field_names[f], d_prej_texts[f].get(pi).cloned().unwrap_or_default()));
                            }
                        } else {
                            kept.push(d_prej_raw[f][pi].clone());
                        }
                    }
                    emit_term(&format!("  🏷️ [LABEL BANK] '{}' | 라벨 구 {}개 | 편견 구 {}개 (자기오염 {}개 제거)",
                        d_field_names[f], d_label_embs[f].len(), kept.len(), dropped));
                    d_prej_embs.push(kept);
                }

                
                
                
                
                
                let mut phrase_single_value: Vec<String> = vec![String::new(); unique_phrases.len()];
                let mut phrase_multi_value: Vec<String> = vec![String::new(); unique_phrases.len()];
                for (pi, ph) in pair_phrases.iter().enumerate() {
                    let h = match unique_phrases.iter().position(|u| u == ph) { Some(v) => v, None => continue };
                    let p = &detail_pairs[pi];
                    if p.primary_line >= pug_lines_ref.len() { continue; }
                    if pug_lines_ref[p.primary_line].trim().is_empty() { continue; }
                    if phrase_single_value[h].is_empty() && !p.value.trim().is_empty() {
                        phrase_single_value[h] = p.value.clone();
                    }
                    let av = p.value_all.trim();
                    if !av.is_empty() && !phrase_multi_value[h].contains(av) {
                        if phrase_multi_value[h].is_empty() {
                            phrase_multi_value[h] = av.to_string();
                        } else {
                            phrase_multi_value[h].push(' ');
                            phrase_multi_value[h].push_str(av);
                        }
                    }
                }

                
                
                
                let pair_abs_floor = 0.50f32;
                let mut leaf_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];
                let mut sec_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];

                for f in 0..d_field_names.len() {
                    let f_fmt = detect_field_format(&d_field_names[f]);
                    let f_multi = is_multi_value_field(&d_field_names[f]);
                    
                    let f_strict = matches!(
                        f_fmt,
                        FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric
                            | FieldFormat::Phone | FieldFormat::Address | FieldFormat::Text
                    );

                    for h in 0..unique_phrases.len() {
                        if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(&leaf_embs[h], &d_label_embs[f], &d_label_weights[f]);
                        crate::utils::score_dynamics::record_field_seen(&d_field_names[f]);
                        if own < pair_abs_floor {
                            crate::utils::score_dynamics::record_near_miss(&d_field_names[f]);
                            continue;
                        }
                        let prej = if d_prej_embs[f].is_empty() { 0.0 } else { max_pool_sim(&leaf_embs[h], &d_prej_embs[f]) };
                        if prej >= own {
                            crate::utils::score_dynamics::record_field_reject(
                                &d_field_names[f],
                                crate::utils::score_dynamics::GateKind::Prejudice,
                            );
                            emit_term(&format!("    🚫 [PAIR PREJUDICE GATE] '{}' → '{}' | LabelMaxPool: {:.4} <= PrejMaxPool: {:.4}. 경쟁 개념이 우세하여 후보 제외.",
                                unique_phrases[h], d_field_names[f], own, prej));
                            continue;
                        }

                        
                        
                        let pair_val = if f_multi { &phrase_multi_value[h] } else { &phrase_single_value[h] };
                        if f_strict {
                            if pair_val.trim().is_empty() || !value_matches_format(f_fmt, pair_val) {
                                crate::utils::score_dynamics::record_field_reject(
                                    &d_field_names[f],
                                    crate::utils::score_dynamics::GateKind::Format,
                                );
                                emit_term(&format!("    🚫 [PAIR VALUE FORMAT GATE] '{}' → '{}' ({:?}) | 값 \"{}\" 이 형식과 불일치하여 후보 제외.",
                                    unique_phrases[h], d_field_names[f], f_fmt, pair_val));
                                continue;
                            }
                        }
                        
                        
                        
                        let enum_reject = if f_fmt == FieldFormat::Enum { enum_value_reject(&d_field_names[f], pair_val) } else { None };
                        if let Some(why) = enum_reject {
                            crate::utils::score_dynamics::record_field_reject(
                                &d_field_names[f],
                                crate::utils::score_dynamics::GateKind::Enum,
                            );
                            emit_term(&format!("    🚫 [ENUM VALUE GATE] '{}' → '{}' | 값 \"{}\" 은 열거형 후보가 될 수 없습니다 ({}).",
                                unique_phrases[h], d_field_names[f], pair_val, why));
                            continue;
                        }

                        leaf_raw[f][h] = own;

                        if unique_section[h].is_empty() { continue; }
                        if section_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        sec_raw[f][h] = weighted_max_pool_sim(&section_embs[h], &d_label_embs[f], &d_label_weights[f]);
                    }
                }
                let label_key = |s: &str| -> String {
                    s.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_lowercase()
                };
                let bank_keys: Vec<Vec<String>> = d_field_names
                    .iter()
                    .map(|n| {
                        label_phrase_bank(&doc_lang, &page_type, n)
                            .0
                            .iter()
                            .map(|p| label_key(p.as_str()))
                            .collect()
                    })
                    .collect();
                for h in 0..unique_phrases.len() {
                    let leaf = label_key(unique_leaf[h].as_str());
                    if leaf.chars().count() < 2 { continue; }
                    let owners: Vec<usize> = (0..d_field_names.len())
                        .filter(|&f| bank_keys[f].iter().any(|k| *k == leaf))
                        .collect();
                    if owners.len() != 1 { continue; }
                    let o = owners[0];
                    if leaf_raw[o][h] >= 0.0 { continue; }
                    let mut blocked: Vec<String> = Vec::new();
                    for f in 0..d_field_names.len() {
                        if f == o || leaf_raw[f][h] < 0.0 { continue; }
                        leaf_raw[f][h] = -1.0;
                        sec_raw[f][h] = -1.0;
                        blocked.push(d_field_names[f].clone());
                    }
                    if !blocked.is_empty() {
                        crate::utils::score_dynamics::record_baseline("commerce.detail_pair_owner_hold", blocked.len() as f32);
                        emit_term(&format!(
                            "    🔒 [DETAIL PAIR OWNER HOLD] Label '{}' | 인쇄 라벨이 '{}' 라벨 뱅크의 구와 글자가 같은데 그 필드가 이 칸의 값을 형식 게이트에서 받지 못했습니다. 이 칸은 '{}' 의 칸이므로 다른 필드 {:?} 에 배정하지 않습니다.",
                            unique_phrases[h], d_field_names[o], d_field_names[o], blocked
                        ));
                    }
                }
                const SECTION_WEIGHT: f32 = 0.5f32;
                let mut d_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];
                for h in 0..unique_phrases.len() {
                    let mut sec_sum = 0.0f32;
                    let mut sec_cnt = 0usize;
                    for f in 0..d_field_names.len() {
                        if leaf_raw[f][h] < 0.0 { continue; }
                        if sec_raw[f][h] < 0.0 { continue; }
                        sec_sum += sec_raw[f][h];
                        sec_cnt += 1;
                    }
                    let sec_mean = if sec_cnt > 0 { sec_sum / (sec_cnt as f32) } else { 0.0 };
                    for f in 0..d_field_names.len() {
                        if leaf_raw[f][h] < 0.0 { continue; }
                        
                        let sec_term = if sec_cnt > 1 && sec_raw[f][h] >= 0.0 {
                            sec_raw[f][h] - sec_mean
                        } else {
                            0.0
                        };
                        d_matrix[f][h] = leaf_raw[f][h] + SECTION_WEIGHT * sec_term;
                    }
                }

                
                
                
                // 🌟 [T-2] 상세 페어 배정에도 잡음 마진 게이트를 적용합니다.
                //
                //  ── 왜 이 경로도 필요한가 ──
                //   여기는 라벨-값 구조 페어를 필드에 배정합니다.
                //   섹션 축(SECTION_WEIGHT)까지 더한 뒤 마진 0.0 으로 확정하므로,
                //   섹션이 비어 있는 문서에서는 라벨 코사인만으로 동점 확정이 납니다.
                //
                //  ── 혼동 사전을 쓰는 이유 ──
                //   커머스 상세는 필드 이름 공간이 안정적이고
                //   (sender_name / recipient_name / bank / card 등)
                //   같은 쌍이 반복 충돌하면 실제로 역사적 승자가 존재합니다.
                //   트레이딩 배열 행과 달리 사전 조회가 유효합니다.
                let (d_assign, d_diags) = {
                    let names = d_field_names.clone();
                    crate::utils::ai_utils::exclusive_assign_by_score_adaptive(
                        &d_matrix,
                        0.0,
                        0.0,
                        |f: usize, rival: usize, kind: crate::utils::ai_utils::TieKind| -> bool {
                            if kind == crate::utils::ai_utils::TieKind::Flat { return false; }
                            let (a, b) = match (names.get(f), names.get(rival)) {
                                (Some(x), Some(y)) => (x.as_str(), y.as_str()),
                                _ => return false,
                            };
                            match crate::utils::score_dynamics::confusion_winner(a, b) {
                                Some((winner, rate)) => winner == a && rate > 0.5,
                                None => false,
                            }
                        },
                    )
                };
                for d in d_diags.iter() {
                    if d.tie == crate::utils::ai_utils::TieKind::Decisive { continue; }
                    let fname = d_field_names.get(d.field).cloned().unwrap_or_default();
                    let rname = d_field_names.get(d.rival).cloned().unwrap_or_default();
                    emit_term(&format!(
                        "    ⚖️ [DETAIL TIE] '{}' vs '{}' | 마진 {:+.4} < 잡음 마진 {:.4} ({:?}) → {}",
                        fname, rname, d.margin, d.noise_margin, d.tie,
                        if d.accepted { "혼동 사전 승자로 확정" } else { "보류 (LLM 패스로 이관)" }
                    ));
                    crate::utils::score_dynamics::record_confusion(&fname, &rname, d.margin);
                }
                let mut d_assign = d_assign;
                for f in 0..d_assign.len() {
                    let (h, own, margin) = match d_assign[f] {
                        Some(v) => v,
                        None => continue,
                    };
                    if margin > 1e-6 { continue; }
                    let twin = (0..d_field_names.len()).find(|&g| {
                        g != f && d_assign[g].is_none() && (d_matrix[g][h] - own).abs() <= 1e-6
                    });
                    if let Some(g) = twin {
                        d_assign[f] = None;
                        crate::utils::score_dynamics::record_baseline("commerce.detail_pair_dead_heat", 1.0);
                        emit_term(&format!(
                            "    ⚖️ [DETAIL PAIR DEAD HEAT] Label '{}' | '{}' 와 '{}' 의 점수가 {:.4} 로 같아 필드 순서로 한쪽을 고르지 않고 둘 다 보류합니다. 두 필드의 라벨 뱅크가 같은 구를 갖고 있어 이 라벨만으로는 어느 필드의 칸인지 정할 수 없습니다.",
                            unique_phrases[h], d_field_names[f], d_field_names[g], own
                        ));
                    }
                }
                let residual_eligible: Vec<bool> = (0..d_field_names.len())
                    .map(|f| {
                        let name = d_field_names[f].as_str();
                        !is_id_link_field(name)
                            && !d_diags.iter().any(|d| d.field == f && !d.accepted)
                            && matches!(
                                detect_field_format(name),
                                FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric
                                    | FieldFormat::Phone | FieldFormat::Address
                            )
                    })
                    .collect();
                let residual = crate::utils::ai_utils::residual_assign_blocked(&d_matrix, &mut d_assign, &residual_eligible, 0.0);
                for (f, h, blocker, margin) in residual.iter() {
                    crate::utils::score_dynamics::record_baseline("commerce.detail_pair_residual", *margin);
                    emit_term(&format!(
                        "    🧩 [DETAIL PAIR RESIDUAL] Label '{}' → Field '{}' | 이 라벨의 1위 필드 '{}' 는 다른 라벨로 이미 확정되어 이 칸이 비어 있었습니다. 남은 필드 가운데 1위(2위와의 차 {:+.4} · 2위가 없으면 자기 점수)이고 이 필드에게도 남은 칸 중 최선이라 배정합니다. 값 형식으로 검증되는 필드(날짜 · 수치 · 전화 · 주소 · 운송장)에만 적용합니다.",
                        unique_phrases[*h], d_field_names[*f], d_field_names[*blocker], margin
                    ));
                }
                let mut contested_floor: Option<f32> = None;
                for (f, a) in d_assign.iter().enumerate() {
                    let (h, s, m) = match a { Some(v) => *v, None => continue };
                    if m + 1e-4 >= s || is_id_link_field(&d_field_names[f]) {
                        continue;
                    }
                    if form_page && detect_field_format(&d_field_names[f]) == FieldFormat::Text && leaf_raw[f][h] < sec_raw[f][h] {
                        continue;
                    }
                    contested_floor = Some(contested_floor.map_or(s, |c: f32| c.min(s)));
                }
                for (f, a) in d_assign.iter().enumerate() {
                    let (h, score, margin) = match a { Some(v) => *v, None => continue };
                    let owner = d_field_names[f].clone();
                    if is_id_link_field(&owner) { continue; }
                    let mut targets: Vec<usize> = Vec::new();
                    for (pi, ph) in pair_phrases.iter().enumerate() {
                        if ph == &unique_phrases[h] { targets.push(pi); }
                    }
                    if targets.is_empty() { continue; }

                    let owner_fmt = detect_field_format(&owner);
                    let multi = is_multi_value_field(&owner);
                    if form_page && owner_fmt == FieldFormat::Text && leaf_raw[f][h] < sec_raw[f][h] {
                        crate::utils::score_dynamics::record_baseline("commerce.detail_pair_section_led", 1.0);
                        emit_term(&format!(
                            "    🔒 [DETAIL PAIR SECTION-LED] Label '{}' → Field '{}' | 라벨 점수 {:.4} 가 섹션 점수 {:.4} 보다 낮습니다. 입력 폼에서 라벨이 아니라 섹션 제목이 끌어온 글자 필드 배정은 값으로 받지 않습니다.",
                            unique_phrases[h], owner, leaf_raw[f][h], sec_raw[f][h]
                        ));
                        continue;
                    }
                    if form_page && owner_fmt == FieldFormat::Text && margin + 1e-4 >= score {
                        if let Some(floor) = contested_floor {
                            if score < floor {
                                crate::utils::score_dynamics::record_baseline("commerce.detail_pair_uncontested_weak", score);
                                emit_term(&format!(
                                    "    🔒 [DETAIL PAIR UNCONTESTED WEAK] Label '{}' → Field '{}' | 점수 {:.4} 에 겨룬 상대 필드가 없습니다(마진 = 점수). 이 문서에서 상대와 겨뤄 이긴 배정 가운데 가장 낮은 점수 {:.4} 에도 못 미치므로, 값 형식으로 검증할 수 없는 글자 필드에는 이 칸의 값을 넣지 않습니다.",
                                    unique_phrases[h], owner, score, floor
                                ));
                                continue;
                            }
                        }
                    }
                    let mut merged = String::new();
                    let mut primary = detail_pairs[targets[0]].primary_line;
                    for pi in &targets {
                        let p = &detail_pairs[*pi];
                        if p.primary_line >= pug_lines_ref.len() { continue; }
                        if pug_lines_ref[p.primary_line].trim().is_empty() { continue; }
                        let v = if multi { p.value_all.clone() } else { p.value.clone() };
                        if v.trim().is_empty() { continue; }
                        pair_owned_lines.insert(p.primary_line);
                        if merged.is_empty() {
                            merged = v;
                            primary = p.primary_line;
                        } else if multi && !merged.contains(&v) {
                            merged.push(' ');
                            merged.push_str(&v);
                        }
                    }
                    if merged.trim().is_empty() { continue; }

                    let lower_owner = owner.to_lowercase();
                    let needs_normalization = lower_owner.contains("status")
                        || lower_owner.contains("payment_method")
                        || lower_owner.contains("payment_origin")
                        || lower_owner.contains("condition")
                        || lower_owner.contains("currency");

                    if needs_normalization {
                        header_forced_assign.insert(owner.clone(), primary);
                        emit_term(&format!("    🎯 [DETAIL PAIR FORCED ASSIGN] '{}' ← Line {} (\"{}\") | Label '{}' | Score: {:+.4} | Margin: {:+.4} | enum 정규화가 필요해 값 우회 대신 벡터 배정을 확정합니다.",
                            owner, primary + 1, merged, unique_phrases[h], score, margin));
                        continue;
                    }

                    let fmt_ok = match owner_fmt {
                        FieldFormat::Identifier | FieldFormat::Link => true,
                        _ => value_matches_format(owner_fmt, &merged),
                    };
                    if !fmt_ok {
                        crate::utils::score_dynamics::record_field_reject(
                            &owner,
                            crate::utils::score_dynamics::GateKind::Format,
                        );
                        emit_term(&format!("    🚫 [DETAIL PAIR FORMAT REJECT] '{}' ({:?}) | 라벨 '{}' 의 값 '{}' 이 형식과 불일치하여 주입하지 않습니다.",
                            owner, owner_fmt, unique_phrases[h], merged));
                        continue;
                    }
                    pair_line_map.insert(owner.clone(), primary);
                    crate::utils::score_dynamics::record_field_assigned(&owner, margin);
                    pre_mapped_hints.push(json!({
                        "target_column": owner.clone(),
                        "extracted_value": merged.clone()
                    }));
                    emit_term(&format!("    ✨ [DETAIL PAIR COSINE MAP] Label '{}' → Field '{}' | LeafRaw: {:.4} | SecRaw: {:.4} | Centered: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
                        unique_phrases[h], owner,
                        leaf_raw[f][h].max(0.0),
                        sec_raw[f][h].max(0.0),
                        score, margin, primary + 1, merged));
                }
            }

            
            for l in &pair_owned_lines { det_consumed_lines.insert(*l); }
            
            let mut enum_resolved: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            {
                let select_groups = collect_select_groups(&clean_html_content);
                if select_groups.is_empty() {
                    emit_term("  ⚪ [ENUM SELECT] 문서에 <select> 컨트롤이 없어 상태 선택자 해석을 건너뜁니다.");
                } else {
                    let key_banks = status_key_banks(&model, &page_type).await;

                    let rival_phrases: Vec<String> = {
                        let mut v: Vec<String> = vec![
                            "delivery company".to_string(), "courier company".to_string(),
                            "shipping carrier".to_string(), "postal service".to_string(),
                            "bank name".to_string(), "bank account number".to_string(),
                            "credit card company".to_string(), "payment gateway".to_string(),
                            "please select".to_string(), "choose an option".to_string(),
                            "category".to_string(), "brand".to_string(), "country".to_string(),
                        ];
                        for fname in ["carrier", "bank", "card", "payment_origin", "payment_method"] {
                            let (lp, _) = label_phrase_bank(&doc_lang, &page_type, fname);
                            for p in lp { if !v.iter().any(|e| e == &p) { v.push(p); } }
                        }
                        if v.len() > 64 { v.truncate(64); }
                        v
                    };
                    let rival_bank = model.get_embedding_batch(rival_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; rival_phrases.len()]);

                    let mut scored: Vec<(usize, f32)> = Vec::new();
                    for (gi, g) in select_groups.iter().enumerate() {
                        let opt_embs = model.get_embedding_batch(g.options.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; g.options.len()]);
                        let mut s_sum = 0.0f32;
                        let mut r_sum = 0.0f32;
                        let mut cnt = 0usize;
                        for oe in &opt_embs {
                            if oe.iter().all(|&v| v == 0.0) { continue; }
                            let mut best_k = 0.0f32;
                            for (_, kb) in &key_banks {
                                let s = max_pool_sim(oe, kb);
                                if s > best_k { best_k = s; }
                            }
                            s_sum += best_k;
                            r_sum += max_pool_sim(oe, &rival_bank);
                            cnt += 1;
                        }
                        if cnt == 0 { continue; }
                        let s_mean = s_sum / (cnt as f32);
                        let r_mean = r_sum / (cnt as f32);

                        let role_emb = model.get_embedding(g.role_phrase.clone()).await.unwrap_or(vec![0.0; 384]);
                        let mut role_status = 0.0f32;
                        for (_, kb) in &key_banks {
                            let s = max_pool_sim(&role_emb, kb);
                            if s > role_status { role_status = s; }
                        }
                        let role_rival = max_pool_sim(&role_emb, &rival_bank);
                        let contrast = (s_mean - r_mean) + 0.5 * (role_status - role_rival);

                        emit_term(&format!("      🎛️ [SELECT CANDIDATE] '{}' | Role: '{}' | Options: {} | StatusMean: {:.4} | RivalMean: {:.4} | RoleΔ: {:+.4} | Contrast: {:+.4}",
                            g.selector, g.role_phrase, g.options.len(), s_mean, r_mean, role_status - role_rival, contrast));
                        scored.push((gi, contrast));
                    }

                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut chosen: Option<usize> = None;
                    if let Some((gi, c1)) = scored.first().copied() {
                        let c2 = scored.get(1).map(|x| x.1).unwrap_or(f32::MIN);
                        let margin = if c2 == f32::MIN { c1 } else { c1 - c2 };
                        if c1 > 0.02 && margin > 0.02 {
                            chosen = Some(gi);
                            emit_term(&format!("  🎛️ [ENUM SELECT COSINE] 상태 컨트롤 확정: '{}' | Contrast: {:+.4} | Margin: {:+.4}",
                                select_groups[gi].selector, c1, margin));
                        } else {
                            emit_term(&format!("  ⚠️ [ENUM SELECT AMBIGUOUS] 최고 Contrast {:+.4} / Margin {:+.4} 로 코사인 확정 실패. LLM CSS selector 탐색으로 넘어갑니다.", c1, margin));
                        }
                    }

                    
                    let pair_status_line = header_forced_assign.get("status").copied();
                    if chosen.is_none() {
                        if let Some(pl) = pair_status_line {
                            crate::utils::score_dynamics::record_baseline("commerce.enum_select_llm_skip", 1.0);
                            emit_term(&format!(
                                "  ⏭️ [ENUM SELECT LLM SKIP] 'status' 는 이미 라벨↔값 쌍 Line {} (\"{}\") 로 확정되어 있습니다. 코사인이 상태 컨트롤을 고르지 못했다는 이유로 2B 를 올려 <select> {}개 전체를 묻지 않습니다 (2B 적재 · 프리필 · Qwen3 재적재가 생기지 않습니다).",
                                pl + 1,
                                line_values.get(pl).map(|s| s.as_str()).unwrap_or(""),
                                select_groups.len()
                            ));
                        }
                    }
                    if chosen.is_none() && pair_status_line.is_none() {
                        let catalogue: Vec<serde_json::Value> = select_groups.iter().map(|g| json!({
                            "selector": g.selector,
                            "role": g.role_phrase,
                            "options": g.options
                        })).collect();
                        let cat_str = serde_json::to_string_pretty(&catalogue).unwrap_or_default();
                        let sel_prompt = crate::parsing::extract_status_selector_prompt(&page_type, &doc_lang, &cat_str);

                        let params = ChatCompletionParameters {
                            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(sel_prompt),
                                name: None,
                            })],
                            model: "qwen3.5".to_string(),
                            max_tokens: Some(128),
                            temperature: Some(0.0),
                            top_p: Some(0.95),
                            ..Default::default()
                        };

                        // 🌟 [CROSSOVER] 임베딩 페이즈 한복판의 유일한 생성 호출입니다.
                        //    Qwen3.5(2B)는 임베딩과 동시 상주가 어려운 크기이므로
                        //    예산 판정이 대부분 SWAP 으로 떨어집니다. 정상 동작입니다.
                        model
                            .switch_to_generation(
                                crate::model::ModelSize::Qwen3_5,
                                Some(cancellation_token.clone()),
                                kv_name.clone(),
                                "enum select fallback",
                            )
                            .await?;
                        let mut picked = String::new();
                        if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                            if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_status_selector", task.id)), kv_name.clone(), None, None).await {
                                let parsed = crate::parsing::parse_json_from_llm(&res);
                                picked = parsed.get("status_selector").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                            }
                        }
                        model.deep_purge_resources().await;
                        model.mark_crossover_idle();
                        if detail_route.is_none() {
                            model
                                .switch_to_generation(
                                    crate::model::ModelSize::Qwen3,
                                    Some(cancellation_token.clone()),
                                    Some("inference".to_string()),
                                    "detail field extraction (after enum fallback)",
                                )
                                .await?;
                        }

                        if !picked.is_empty() && picked != "null" {
                            if let Some(pos) = select_groups.iter().position(|g| g.selector == picked) {
                                chosen = Some(pos);
                                emit_term(&format!("  🤖 [ENUM SELECT LLM] LLM 이 상태 컨트롤로 '{}' 를 지목했습니다.", picked));
                            } else {
                                emit_term(&format!("  🚫 [ENUM SELECT LLM REJECT] LLM 이 반환한 '{}' 는 실제 후보 목록에 없어 폐기합니다.", picked));
                            }
                        }
                    }

                    
                    let pair_conflict = match (chosen, pair_status_line) {
                        (Some(gi), Some(pl)) => {
                            let key = |s: &str| -> String {
                                s.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_lowercase()
                            };
                            let sel = key(&select_groups[gi].selected);
                            let line = key(line_values.get(pl).map(|s| s.as_str()).unwrap_or(""));
                            !sel.is_empty() && !line.contains(&sel) && !sel.contains(&line)
                        }
                        _ => false,
                    };
                    if pair_conflict {
                        if let (Some(gi), Some(pl)) = (chosen, pair_status_line) {
                            crate::utils::score_dynamics::record_baseline("commerce.enum_select_pair_conflict", 1.0);
                            emit_term(&format!(
                                "  🔒 [ENUM SELECT PAIR CONFLICT] '{}' 의 선택값 \"{}\" 이 라벨↔값 쌍으로 확정된 Line {} (\"{}\") 와 다릅니다. 같은 상태를 두 근거가 다르게 말하므로 컨트롤 쪽으로 덮지 않고 라벨 쌍 배정을 유지합니다.",
                                select_groups[gi].selector,
                                select_groups[gi].selected,
                                pl + 1,
                                line_values.get(pl).map(|s| s.as_str()).unwrap_or("")
                            ));
                        }
                    }
                    if let Some(gi) = chosen.filter(|_| !pair_conflict) {
                        let g = &select_groups[gi];
                        let sel_emb = model.get_embedding(g.selected.clone()).await.unwrap_or(vec![0.0; 384]);
                        let key_scores = status_key_scores(&sel_emb, &key_banks);
                        for (k, s) in key_scores.iter() {
                            emit_term(&format!("      🧭 [STATUS KEY] '{}' ← selected \"{}\" | MaxPool: {:.4}", k, g.selected, s));
                        }
                        let pick = pick_status_key(&key_scores);
                        if pick.accepted() {
                            enum_resolved.insert("status".to_string(), pick.key.clone());
                            emit_term(&format!("  ✅ [ENUM SELECT RESOLVED] '{}' (selected: \"{}\") → status = '{}' | Top: {:.4} | Margin: {:+.4}",
                                g.selector, g.selected, pick.key, pick.top, pick.margin()));
                        } else {
                            emit_term(&format!("  ⚠️ [ENUM SELECT UNRESOLVED] selected \"{}\" 의 캐노니컬 마진 부족 (Top {:.4} / 2nd {:.4}). 기존 경로로 위임합니다.",
                                g.selected, pick.top, pick.second));
                        }
                    }
                }
            }

            
            if enum_resolved.contains_key("status") {
                header_forced_assign.remove("status");
            }

            let pre_mapped_context = if !pre_mapped_hints.is_empty() {
                serde_json::to_string_pretty(&pre_mapped_hints).unwrap_or_default()
            } else {
                String::new()
            };

            let mut global_ignore_list: Vec<String> = Vec::new();

            
            
            let (mut vector_assignment, vector_raw_matrix): (Vec<Option<(usize, f32, f32)>>, Vec<Vec<f32>>) = {
                let line_count = pug_lines_ref.len();
                let field_count = field_phrase_embs.len();
                let mut raw = vec![vec![-1.0f32; line_count]; field_count];

                for f in 0..field_count {
                    if field_is_analytic[f] { continue; }
                    
                    
                    
                    if is_id_link_field(&fields[f].0) && det_id_link.is_some() { continue; }
                    let fmt = field_formats[f];
                    for l in 0..line_count {
                        if pug_lines_ref[l].trim().is_empty() { continue; }
                        if line_embeddings[l].iter().all(|&v| v == 0.0) { continue; }
                        if det_consumed_lines.contains(&l) { continue; }
                        
                        if line_is_non_value[l] { continue; }
                        
                        
                        if fmt == FieldFormat::Enum && !line_is_selected_option[l] { continue; }

                        let value = &line_values[l];
                        let format_ok = match fmt {
                            FieldFormat::Identifier | FieldFormat::Link => value_token_in_url_pool(value, &url_pool),
                            _ => value_matches_format(fmt, value),
                        };
                        if !format_ok { continue; }

                        let own = weighted_max_pool_sim(
                            &line_embeddings[l],
                            &field_phrase_embs[f],
                            &field_phrase_weights[f],
                        );

                        
                        if fmt == FieldFormat::Enum {
                            let prej = if field_prej_phrase_embs[f].is_empty() {
                                0.0
                            } else {
                                max_pool_sim(&line_embeddings[l], &field_prej_phrase_embs[f])
                            };
                            if own - prej < 0.15 { continue; }
                        }

                        raw[f][l] = own;
                    }
                }

                let centered = double_center_matrix(&raw);
                let mut assign = exclusive_assign(&centered, 0.0, 0.005);

                let mut claimed = vec![false; line_count];
                for a in assign.iter() {
                    if let Some((l, _, _)) = a { claimed[*l] = true; }
                }
                for f in 0..field_count {
                    if assign[f].is_some() { continue; }
                    if field_is_analytic[f] { continue; }
                    let cands: Vec<usize> = (0..line_count)
                        .filter(|&l| raw[f][l] >= 0.0 && !claimed[l])
                        .collect();
                    if cands.len() == 1 {
                        let l = cands[0];
                        assign[f] = Some((l, centered[f][l], 0.0));
                        claimed[l] = true;
                    }
                }

                (assign, raw)
            };

            
            for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                if let Some(l) = header_forced_assign.get(fname) {
                    let raw = vector_raw_matrix.get(f_i).and_then(|r| r.get(*l)).copied().unwrap_or(0.0).max(0.0);
                    vector_assignment[f_i] = Some((*l, raw, 0.0));
                    emit_term(&format!("  🧷 [PAIR OVERRIDE] '{}' 의 벡터 배정을 구조적 페어 확정 라인(Line {})으로 교체했습니다.", fname, *l + 1));
                }
            }

            
            
            
            
            let mut family_shared: std::collections::HashSet<usize> = std::collections::HashSet::new();
            if form_page {
                let held: Vec<String> = (0..vector_assignment.len())
                    .filter(|&f| {
                        vector_assignment[f].is_none()
                            && !field_is_analytic[f]
                            && !is_id_link_field(&fields[f].0)
                            && !pair_line_map.contains_key(&fields[f].0)
                            && matches!(field_formats[f], FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric)
                    })
                    .map(|f| fields[f].0.clone())
                    .collect();
                if !held.is_empty() {
                    crate::utils::score_dynamics::record_baseline("commerce.form_family_share_skip", held.len() as f32);
                    emit_term(&format!(
                        "  ⛔ [FORMAT FAMILY SHARE / FORM] 입력 폼 상세라 같은 형식 필드의 줄을 빌려 쓰지 않습니다. 자기 라벨 칸도 벡터 배정도 없는 필드: {:?} | 폼에서는 칸마다 자기 라벨이 있으므로, 자기 라벨 칸이 없는 필드가 다른 필드(등록일시 · 포인트 등)의 줄을 가져가면 그 값은 이 필드의 사실이 아닙니다. 라벨 쌍이 있는 필드는 지금처럼 자기 칸의 값을 씁니다.",
                        held
                    ));
                }
            }
            if !form_page {
                let mut shared = 0usize;
                for f in 0..vector_assignment.len() {
                    if vector_assignment[f].is_some() { continue; }
                    if field_is_analytic[f] { continue; }
                    if is_id_link_field(&fields[f].0) { continue; }
                    let fmt = field_formats[f];
                    if !matches!(fmt, FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric) { continue; }
                    let mut best_line: Option<usize> = None;
                    let mut best_raw = f32::MIN;
                    for other in 0..vector_assignment.len() {
                        if other == f { continue; }
                        if field_formats[other] != fmt { continue; }
                        
                        
                        
                        
                        let source_line: Option<usize> = if let Some(&pl) = pair_line_map.get(&fields[other].0) {
                            Some(pl)
                        } else if let Some((l, _, _)) = vector_assignment[other] {
                            Some(l)
                        } else {
                            None
                        };
                        if let Some(l) = source_line {
                            
                            
                            if l < line_values.len() && !value_matches_format(fmt, &line_values[l]) {
                                continue;
                            }
                            let raw = vector_raw_matrix[f].get(l).copied().unwrap_or(0.0);
                            if raw > best_raw { best_raw = raw; best_line = Some(l); }
                        }
                    }
                    if let Some(l) = best_line {
                        vector_assignment[f] = Some((l, best_raw, 0.0));
                        shared += 1;
                        family_shared.insert(f);
                        emit_term(&format!("  ♻️ [FORMAT FAMILY SHARE] '{}' ({:?}) ← Line {} | RawSim: {:.4} | 같은 형식 필드가 확정한 라인을 공유합니다.",
                            fields[f].0, fmt, l + 1, best_raw));
                    }
                }
                if shared > 0 {
                    emit_term(&format!("  ♻️ [FORMAT FAMILY SHARE] 총 {}개 필드가 동일 형식 라인을 공유했습니다.", shared));
                }
            }

            {
                for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                    if field_is_analytic[f_i] { continue; }
                    crate::utils::score_dynamics::record_field_seen(fname);
                }
                for l in 0..pug_lines_ref.len() {
                    let mut col: Vec<f32> = Vec::new();
                    for f_i in 0..fields.len() {
                        if field_is_analytic[f_i] { continue; }
                        let s = vector_raw_matrix[f_i][l];
                        if s < 0.0 { continue; }
                        col.push(s);
                    }
                    if col.len() >= 2 {
                        crate::utils::score_dynamics::record_decay("commerce.detail_row", &col);
                    }
                }
            }
            for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                match vector_assignment[f_i] {
                    Some((l, contrast, margin)) => {
                        emit_term(&format!("  🔗 [EXCLUSIVE ASSIGN] '{}' ({:?}) ← Line {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4} | \"{}\"", fname, field_formats[f_i], l + 1, vector_raw_matrix[f_i][l], contrast, margin, pug_lines_ref[l].trim()));
                        crate::utils::score_dynamics::record_field_assigned(fname, margin);
                        let mut rival_name = String::new();
                        let mut rival_score = f32::MIN;
                        for (g_i, (gname, _, _, _)) in fields.iter().enumerate() {
                            if g_i == f_i { continue; }
                            if field_is_analytic[g_i] { continue; }
                            let s = vector_raw_matrix[g_i][l];
                            if s < 0.0 { continue; }
                            if s > rival_score {
                                rival_score = s;
                                rival_name = gname.clone();
                            }
                        }
                        if !rival_name.is_empty() {
                            let own = vector_raw_matrix[f_i][l];
                            crate::utils::score_dynamics::record_confusion(
                                fname,
                                &rival_name,
                                (own - rival_score).abs(),
                            );
                        }
                    },
                    None => {
                        if !field_is_analytic[f_i] {
                            let cand_cnt = vector_raw_matrix[f_i].iter().filter(|&&v| v >= 0.0).count();
                            emit_term(&format!("  ⚪ [UNASSIGNED] '{}' ({:?}) | 형식 통과 후보 {}개 | 벡터 힌트 미주입", fname, field_formats[f_i], cand_cnt));
                            if cand_cnt == 0 {
                                crate::utils::score_dynamics::record_field_reject(
                                    fname,
                                    crate::utils::score_dynamics::GateKind::Format,
                                );
                            } else {
                                crate::utils::score_dynamics::record_near_miss(fname);
                            }
                        }
                    }
                }
            }


            // 🌟 [CROSSOVER] 벡터 배정이 끝났습니다. 이 아래는 전부 Qwen3 generate 입니다.
            //    ENUM SELECT 폴백이 발동한 경우 이미 생성 페이즈이므로
            //    enter_generation_phase 는 no-op 에 가깝고,
            //    발동하지 않은 경우에만 실제 전환이 일어납니다.
            if let Some(code) = detail_route.as_deref() {
                field_lang = model.enter_lang_generation(code, "detail field extraction loop").await;
            }
            if field_lang.is_none() {
                model
                    .enter_generation_phase(
                        crate::model::ModelSize::Qwen3,
                        None,
                        Some(cancellation_token.clone()),
                        false,
                        Some("inference".to_string()),
                        "detail field extraction loop",
                    )
                    .await?;
            }
            emit_term(&format!(
                "  {} | 필드 생성 엔진: {}",
                model.crossover_report(),
                field_lang.as_ref().map(|s| s.label()).unwrap_or_else(|| "Qwen3".to_string())
            ));
            let mut detail_synthesis_jobs: Vec<(usize, String, String)> = Vec::new();
            for (idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.into_iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if let Some(canon) = enum_resolved.get(&field_name).cloned() {
                    crate::utils::score_dynamics::record_field_assigned(&field_name, 0.0);
                    crate::utils::score_dynamics::record_baseline("commerce.enum_bypass", 1.0);
                    extracted_data.as_object_mut().unwrap().insert(field_name.clone(), json!(canon.clone()));
                    if !global_ignore_list.contains(&canon) {
                        global_ignore_list.push(canon.clone());
                        global_ignore_list.push(format!(" {}", canon));
                        global_ignore_list.push(canon.to_lowercase());
                    }
                    emit_term(&format!("  ⚡ [ENUM BYPASS] LLM 없이 확정: \"{}\": \"{}\"", field_name, canon));
                    continue;
                }

                let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                let mut bypassed_values: Vec<(String, String)> = Vec::new();
                for k in &keys {
                    for hint in &pre_mapped_hints {
                        if let Some(t_col) = hint.get("target_column").and_then(|v| v.as_str()) {
                            if t_col == *k {
                                if let Some(e_val) = hint.get("extracted_value").and_then(|v| v.as_str()) {
                                    let clean_e_val = e_val.trim();
                                    if !clean_e_val.is_empty() {
                                        if let Some(existing) = bypassed_values.iter_mut().find(|(key, _)| key == *k) {
                                            if !existing.1.contains(clean_e_val) {
                                                existing.1.push_str(" ");
                                                existing.1.push_str(clean_e_val);
                                            }
                                        } else {
                                            bypassed_values.push((k.to_string(), clean_e_val.to_string()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !bypassed_values.is_empty() {
                    let percent = (((idx as f32) / (total_fields as f32)) * 100.0) as i32;
                    let summary_msg = format!("Extracting {} ({}%)...", field_name, percent);
                    let payload = json!({ 
                        "task_id": task.id, 
                        "category": format!("Detail Extraction ({}/{})", idx + 1, total_fields), 
                        "summary": summary_msg, 
                        "spinner": "⠋" 
                    });
                    log_task_progress(app_handle, &task.id, &payload);
                    emit_term(&format!("[STAGE-3] {}", summary_msg));

                    let mut extracted_results = Vec::new();
                    for (k, val_str) in bypassed_values {
                        crate::utils::score_dynamics::record_field_assigned(&k, 0.0);
                        crate::utils::score_dynamics::record_baseline("commerce.premap_bypass", 1.0);
                        extracted_data.as_object_mut().unwrap().insert(k.clone(), json!(val_str));
                        extracted_results.push(format!("\"{}\": \"{}\"", k, val_str));
                        
                        if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                            if !global_ignore_list.contains(&val_str) {
                                global_ignore_list.push(val_str.clone());
                                global_ignore_list.push(format!(" {}", val_str));
                                global_ignore_list.push(val_str.to_lowercase());
                            }
                        }
                    }
                    emit_term(&format!("    ⚡ [PRE-MAP BYPASS] Successfully mapped without LLM: {}", extracted_results.join(", ")));
                    continue;
                }

                let field_format = field_formats[idx];

                
                if is_id_link_field(&field_name) {
                    if let Some((det_id, det_link)) = det_id_link.clone() {
                        extracted_data.as_object_mut().unwrap().insert("id".to_string(), json!(det_id.clone()));
                        extracted_data.as_object_mut().unwrap().insert("link".to_string(), json!(det_link.clone()));
                        if det_id.chars().count() >= 5 && !global_ignore_list.contains(&det_id) {
                            global_ignore_list.push(det_id.clone());
                            global_ignore_list.push(format!(" {}", det_id));
                            global_ignore_list.push(det_id.to_lowercase());
                        }
                        emit_term(&format!("  ⚡ [ID/LINK BYPASS] LLM 없이 확정: \"id\": \"{}\", \"link\": \"{}\"", det_id, det_link));
                        continue;
                    }
                }
                if form_page && field_format == FieldFormat::Text && !field_is_analytic[idx] {
                    crate::utils::score_dynamics::record_baseline("commerce.form_absent_skip", 1.0);
                    emit_term(&format!(
                        "  ⛔ [FORM ABSENT] Field: '{}' (Text) | 입력 폼 상세(라벨 칸 {}개 중 입력 컨트롤 {}개)에서 이 필드에 대응하는 라벨 칸 값이 없습니다. 메뉴 · 도움말 · 다른 칸의 글자를 LLM 으로 고르지 않고 비워 둡니다. 빈 값은 목록 문서의 기존 값을 덮지 않습니다.",
                        field_name, form_labeled, form_controls
                    ));
                    continue;
                }
                let (_bias_emb, _prej_emb, dynamic_prej_str) = &field_embeddings[idx];

                
                let (best_idx, best_contrast, best_margin, has_vector_match) = match vector_assignment[idx] {
                    Some((l, contrast, margin)) => (l, contrast, margin, true),
                    None => (0usize, 0.0f32, 0.0f32, false),
                };
                let best_raw = if has_vector_match { vector_raw_matrix[idx][best_idx] } else { 0.0f32 };

                
                
                
                
                
                
                
                let strict_format_field = matches!(
                    field_format,
                    FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric
                        | FieldFormat::Identifier | FieldFormat::Link | FieldFormat::Enum
                        | FieldFormat::Phone
                );
                if !field_is_analytic[idx] && strict_format_field && !has_vector_match {
                    emit_term(&format!("  ⛔ [FORMAT SKIP] Field: '{}' ({:?}) | 형식에 맞는 후보 셀이 문서에 존재하지 않습니다. LLM 호출 없이 빈 값으로 확정.", field_name, field_format));
                    crate::utils::score_dynamics::record_field_reject(
                        &field_name,
                        crate::utils::score_dynamics::GateKind::Format,
                    );
                    continue;
                }

                
                
                if !field_is_analytic[idx] && field_format == FieldFormat::Date && has_vector_match {
                    if let Some(date_literal) = extract_date_literal(&line_values[best_idx]) {
                        let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                        let mut done = Vec::new();
                        for k in &keys {
                            crate::utils::score_dynamics::record_field_assigned(k, best_margin);
                            crate::utils::score_dynamics::record_baseline("commerce.date_regex_bypass", 1.0);
                            extracted_data.as_object_mut().unwrap().insert(k.to_string(), json!(date_literal.clone()));
                            done.push(format!("\"{}\": \"{}\"", k, date_literal));
                        }
                        if !global_ignore_list.contains(&date_literal) {
                            global_ignore_list.push(date_literal.clone());
                            global_ignore_list.push(format!(" {}", date_literal));
                            global_ignore_list.push(date_literal.to_lowercase());
                        }
                        emit_term(&format!("  ⚡ [DATE REGEX BYPASS] LLM 없이 확정: {}", done.join(", ")));
                        continue;
                    }
                }

                
                
                
                
                
                if !field_is_analytic[idx] && has_vector_match {
                    let copyable = matches!(
                        field_format,
                        FieldFormat::Phone | FieldFormat::Address | FieldFormat::TrackingCode | FieldFormat::Numeric
                    );
                    let shared_line = family_shared.contains(&idx);
                    let multi_number = field_format == FieldFormat::Numeric && numeric_run_count(&line_values[best_idx]) > 1;
                    let foreign_label = detail_pairs
                        .iter()
                        .find(|p| p.primary_line == best_idx)
                        .map(|p| p.label.clone());
                    if copyable && (shared_line || multi_number || foreign_label.is_some()) {
                        crate::utils::score_dynamics::record_baseline("commerce.value_copy_refused", 1.0);
                        let why = if shared_line {
                            "같은 형식의 다른 필드가 확정한 라인을 공유받은 배정이라".to_string()
                        } else if multi_number {
                            "라인에 수치가 여럿 있어 이 필드의 값을 하나로 정할 수 없어".to_string()
                        } else {
                            format!(
                                "이 라인은 라벨 '{}' 칸의 값이고 라벨 대조가 그 라벨을 이 필드에 배정하지 않았으므로",
                                foreign_label.clone().unwrap_or_default()
                            )
                        };
                        emit_term(&format!(
                            "  ⛔ [VALUE COPY REFUSED] Field: '{}' ({:?}) | Line {} \"{}\" | {} 값을 비워 둡니다. 다른 필드의 라인을 그대로 옮기거나 여러 수치를 한 값으로 붙이면 틀린 값이 확정되고 저장 단계가 그 값을 숫자로 굳힙니다. 빈 값은 목록 문서의 기존 값을 덮지 않습니다.",
                            field_name,
                            field_format,
                            best_idx + 1,
                            line_values[best_idx].trim(),
                            why
                        ));
                        continue;
                    }
                    if copyable {
                        let raw_val = line_values[best_idx].trim().to_string();
                        if !raw_val.is_empty() && value_matches_format(field_format, &raw_val) {
                            let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                            let mut done = Vec::new();
                            for k in &keys {
                                crate::utils::score_dynamics::record_field_assigned(k, best_margin);
                                crate::utils::score_dynamics::record_baseline("commerce.value_copy_bypass", 1.0);
                                extracted_data.as_object_mut().unwrap().insert(k.to_string(), json!(raw_val.clone()));
                                done.push(format!("\"{}\": \"{}\"", k, raw_val));
                            }
                            if !global_ignore_list.contains(&raw_val) {
                                global_ignore_list.push(raw_val.clone());
                                global_ignore_list.push(format!(" {}", raw_val));
                                global_ignore_list.push(raw_val.to_lowercase());
                            }
                            emit_term(&format!("  ⚡ [VALUE COPY BYPASS] ({:?}) LLM 없이 Line {} 값 그대로 확정: {}",
                                field_format, best_idx + 1, done.join(", ")));
                            continue;
                        }
                    }
                }

                if field_is_analytic[idx] {
                    detail_synthesis_jobs.push((0usize, field_name.clone(), field_desc.clone()));
                    emit_term(&format!(
                        "    ⏳ [SYNTHESIS DEFERRED] Field: '{}' | 상세 요약 필드는 이 문서의 추출이 끝난 뒤 요약 패스(commerce.detail_synthesis)에서 만듭니다. 문서 전체 대신 확정된 값의 값표를 쓰고, 엔진은 그 단계에서 따로 판정합니다.",
                        field_name
                    ));
                    continue;
                }
                let targeted_pug = if field_is_analytic[idx] {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD] Field: '{}' | 단일 라인 환원 불가 → 전체 컨텍스트 요약 모드", field_name));
                    content_pug.clone()
                } else if !has_vector_match {
                    emit_term(&format!("  ⚠️ [NO CONFIDENT MATCH] Field: '{}' ({:?}) | 형식 통과 후보 부족 → 전체 컨텍스트만 사용하고 벡터 힌트는 주입하지 않습니다.", field_name, field_format));
                    content_pug.clone()
                } else {
                    emit_term(&format!("  🎯 [EXCLUSIVE MATCH] Field: '{}' ({:?}) | Line: {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}", field_name, field_format, best_idx + 1, best_raw, best_contrast, best_margin));
                    extract_pug_context(&pug_lines_ref, best_idx)
                };

                let mut json_contexts = Vec::new();
                for line in targeted_pug.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if let Some(idx) = trimmed.find('|') {
                        let meta = trimmed[..idx].trim();

                        let clean_meta = meta.split('[').next().unwrap_or(meta).trim();
                        json_contexts.push(json!({
                            "metadata": clean_meta,
                            "value": trimmed[idx + 1..].trim()
                        }));
                    } else {
                        json_contexts.push(json!({
                            "value": trimmed
                        }));
                    }
                }
                let targeted_json_context = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();
                
                emit_term(&format!("  🎯 [MATCHED CONTEXT] Field: '{}' ({:?}) | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}\n{}", field_name, field_format, best_raw, best_contrast, best_margin, targeted_json_context));

                let mut final_context_str = format!("[JSON CONTEXT]\n{}", targeted_json_context);
                if field_is_analytic[idx] {
                    final_context_str.push_str("\n\n[SYNTHESIS FIELD NOTICE]\nThis field is NOT a value to copy. Read the WHOLE [JSON CONTEXT] above and write ONE short sentence that summarizes it. Never return a single cell value such as a bare number, a status word, a person name, or a branch name. If [JSON CONTEXT] is empty, return null.");
                } else if has_vector_match {
                    let matched_line = pug_lines_ref[best_idx].trim();
                    final_context_str.push_str(&format!("\n\n[VECTOR MATCH RESULT]\nThe format gate and the embedding model EXCLUSIVELY assigned this field to the single line below (RawSim {:.4}, Contrast {:+.4}, Margin {:+.4}). No other column may use this line.\nThe part BEFORE '|' is the column LABEL, the part AFTER '|' is the VALUE. Copy ONLY the value part, character for character. Do NOT copy the label. If that value does not fit the schema, return null.\n\"{}\"", best_raw, best_contrast, best_margin, matched_line));
                    if !pre_mapped_context.is_empty() {
                        final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field:\n{}", pre_mapped_context));
                    }
                } else if !pre_mapped_context.is_empty() {
                    final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field. If nothing else in [JSON CONTEXT] fits this field, return null:\n{}", pre_mapped_context));
                }

                let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: final_context_str,
                    name: None,
                });

                let percent = (((idx as f32) / (total_fields as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting {} ({}%)...", field_name, percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("Detail Extraction ({}/{})", idx + 1, total_fields), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));


                let mut metadata_str = String::new();
                let mut target_data_str = String::new();
                for line in targeted_pug.lines() {
                    if let Some(idx) = line.find('|') {
                        metadata_str.push_str(line[..idx].trim());
                        metadata_str.push_str("\n");
                        target_data_str.push_str(line[idx + 1..].trim());
                        target_data_str.push_str("\n");
                    } else {
                        target_data_str.push_str(line.trim());
                        target_data_str.push_str("\n");
                    }
                }
                let metadata_str = metadata_str.trim();
                let target_data_str = target_data_str.trim();

                let mut task_question = if field_name.contains("status") {
                    parsing::extract_status_intent_legacy_prompt(&targeted_pug, &page_type, &bias_target)
                } else if field_is_analytic[idx] {
                    if field_lang.is_some() {
                        parsing::extract_synthesis_field_prompt_native(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                    } else {
                        parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                    }
                } else {
                    parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                };
                let field_keys: Vec<String> = field_name.split(',').map(|s| s.trim().to_string()).collect();


                let mut ignore_list: Vec<String> = if field_is_analytic[idx] { Vec::new() } else { global_ignore_list.clone() };
                let mut miss_counter = 0;
                
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }

                    let field_name_clone = field_name.clone();
                    let used_lang = field_lang.is_some();
                    let res = if let Some(sess) = field_lang.as_ref() {
                        let user_msg = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                            name: None,
                        });
                        model
                            .generate_lang_json(
                                sess.code(),
                                vec![system_message.clone(), user_msg],
                                512,
                                0.0,
                                Some(ignore_list.as_slice()),
                                &field_keys,
                                Some(cancellation_token.clone()),
                            )
                            .await
                            .map(|(text, fixed)| {
                                if !fixed.is_empty() {
                                    crate::utils::score_dynamics::record_baseline("commerce.lang4b_key_repair", fixed.len() as f32);
                                    emit_term(&format!("  🩹 [LANG4B KEY REPAIR] {:?} — 어휘 16,384 개인 4B 가 영문 스키마 키를 한두 글자 틀리게 옮긴 것을 철자 거리로 되돌립니다. 값은 아래 검증을 그대로 거칩니다.", fixed));
                                }
                                text
                            })
                    } else {
                        let q3_gen = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let sys_msg = system_message.clone();
                        let prejudice_target_for_closure = if field_is_analytic[idx] { String::new() } else { dynamic_prej_str.clone() };
                        let task_q = task_question.clone();
                        let ignore_list_clone = ignore_list.clone();
                        let res = tokio::task::spawn_blocking(move || {
                            let mut gen_guard = q3_gen.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                let params = ChatCompletionParameters {
                                    messages: vec![
                                        sys_msg,
                                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                            content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                            name: None,
                                        })
                                    ],
                                    model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                    ..Default::default()
                                };


                                let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };


                                gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                            } else {
                                Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                            }
                        }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));
                        let q3_clear_arc = model.qwen3_generator.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                                gen.clear_kv_cache();
                            }
                        }).await;

                        if !model.is_cpu_mode {
                            let dev = model.device_config.device.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                if dev.is_cuda() { let _ = dev.synchronize(); }
                            }).await;
                        }
                        res
                    };
                    if used_lang {
                        if let Err(e) = &res {
                            emit_term(&format!("  ⚠️ [LANG4B FALLBACK] Field '{}' | 4B 생성이 실패해 이 문서의 남은 필드는 Qwen3 로 진행합니다: {}", field_name, e));
                            crate::model::lang_llm::record_route_outcome("commerce", "detail_field", false);
                            field_lang = None;
                            if field_is_analytic[idx] && !field_name.contains("status") {
                                task_question = parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str);
                            }
                            model
                                .enter_generation_phase(
                                    crate::model::ModelSize::Qwen3,
                                    None,
                                    Some(cancellation_token.clone()),
                                    false,
                                    Some("inference".to_string()),
                                    "detail field extraction loop (4B fallback)",
                                )
                                .await?;
                            continue;
                        }
                    }

                    match res {
                        Ok(res_text) => {
                            let mut parsed = parsing::parse_json_from_llm(&res_text);
                            

                            let mut item_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed };

                            
                            
                            if let Some(obj) = item_val.as_object_mut() {
                                let ks: Vec<String> = obj.keys().cloned().collect();
                                for k in ks {
                                    let cleaned = match obj.get(&k) {
                                        Some(serde_json::Value::String(s)) => Some(strip_markup_prefix(s)),
                                        _ => None,
                                    };
                                    if let Some(c) = cleaned {
                                        obj.insert(k, json!(c));
                                    }
                                }
                            }

                            let mut requires_retry = false;
                            let mut extracted_values_for_retry = Vec::new();
                            
                            let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                            let mut found_valid_value = false;


                            let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                            
                            let synthesis_fields = ["insight", "summary", "analysis"];
                            let field_name_lower = field_name_clone.to_lowercase();
                            let is_synthesis_field = synthesis_fields.iter().any(|&f| field_name_lower.contains(f));
                            let is_enum_field = is_synthesis_field || skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    let is_empty_val = match val {
                                        serde_json::Value::Null => true,
                                        serde_json::Value::String(s) => s.trim().is_empty() || s == "..." || s == "null",
                                        serde_json::Value::Array(a) => a.is_empty(),
                                        serde_json::Value::Object(o) => o.is_empty(),
                                        _ => false,
                                    };
                                    if !is_empty_val {
                                        let extracted_str = if val.is_string() {
                                            val.as_str().unwrap_or("").trim().to_string()
                                        } else if val.is_number() {
                                            val.to_string()
                                        } else {
                                            String::new()
                                        };

                                        if is_synthesis_field && !extracted_str.is_empty() {
                                            let answer_key = extracted_str.trim().trim_matches(|c: char| c == '.' || c == '"' || c == '\'').trim().to_lowercase();
                                            let bare_value = !answer_key.chars().any(|c| c.is_alphanumeric())
                                                || extracted_data.as_object().map_or(false, |o| {
                                                    o.iter().any(|(ok, ov)| ok.as_str() != *k && ov.as_str().map_or(false, |s| s.trim().to_lowercase() == answer_key))
                                                })
                                                || targeted_pug.lines().any(|l| l.split_once('|').map_or(false, |(_, v)| v.trim().to_lowercase() == answer_key));
                                            if bare_value {
                                                emit_term(&format!(
                                                    "  ⏭️ [SYNTHESIS NOT A SUMMARY] '{}' | 응답 \"{}\" 은 문서의 값 하나를 그대로 옮겼거나 글자가 없는 응답이라 요약으로 받지 않고 다시 묻습니다. 다시 물을 때는 다른 필드 값 전체가 아니라 이 응답만 금지 목록에 넣습니다.",
                                                    k,
                                                    extracted_str
                                                ));
                                                requires_retry = true;
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                continue;
                                            }
                                            let gate = crate::utils::ai_utils::synthesis_fact_gate(&extracted_str, &targeted_pug, &doc_lang);
                                            crate::utils::score_dynamics::record_baseline(
                                                crate::model::lang_llm::synthesis_drop_axis(used_lang),
                                                if gate.total == 0 { 0.0 } else { gate.dropped.len() as f32 / gate.total as f32 },
                                            );
                                            crate::model::lang_llm::record_engine_outcome("commerce", "detail_synthesis", used_lang, gate.dropped.is_empty());
                                            if !gate.dropped.is_empty() {
                                                emit_term(&format!(
                                                    "  🧪 [SYNTHESIS GROUNDING] '{}' | 문장 {}개 중 {}개를 뺍니다: {:?} | 남은 문장: \"{}\" — 요약 문장의 연도·날짜·금액·통화는 아이템 원문에 그대로 있어야 합니다. 원문에 없는 사실이 text·masked_text 로 들어가면 FTS 와 청크 코사인이 그 사실로 이 문서를 회수합니다.",
                                                    k,
                                                    gate.total,
                                                    gate.dropped.len(),
                                                    gate.dropped.iter().map(|(_, why)| why.join("·")).collect::<Vec<_>>(),
                                                    gate.kept
                                                ));
                                                if let Some(o) = item_val.as_object_mut() {
                                                    if gate.kept.is_empty() {
                                                        o.remove(*k);
                                                    } else {
                                                        o.insert(k.to_string(), json!(gate.kept));
                                                    }
                                                }
                                                found_valid_value = true;
                                                continue;
                                            }
                                        }

                                        let key_fmt = detect_field_format(k);
                                        let strict_post = matches!(
                                            key_fmt,
                                            FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Text
                                                | FieldFormat::Numeric | FieldFormat::Enum | FieldFormat::Identifier
                                                | FieldFormat::Phone | FieldFormat::Address
                                        );
                                        if strict_post && !extracted_str.is_empty() && !value_matches_format(key_fmt, &extracted_str) {
                                            emit_term(&format!("  🚫 [FORMAT REJECT] '{}' ({:?}) 에 형식 불일치 값 '{}' 반환. 폐기 후 재시도합니다.", k, key_fmt, extracted_str));
                                            crate::utils::score_dynamics::record_field_seen(k);
                                            crate::utils::score_dynamics::record_field_reject(
                                                k,
                                                crate::utils::score_dynamics::GateKind::Format,
                                            );
                                            requires_retry = true;
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            continue;
                                        }
                                        let enum_reject = if key_fmt == FieldFormat::Enum { enum_value_reject(k, &extracted_str) } else { None };
                                        if let Some(why) = enum_reject {
                                            crate::utils::score_dynamics::record_field_seen(k);
                                            crate::utils::score_dynamics::record_field_reject(
                                                k,
                                                crate::utils::score_dynamics::GateKind::Enum,
                                            );
                                            let amount_answer = k.to_lowercase().contains("currency") && currency_amount_like(&extracted_str);
                                            let prior_amount = ignore_list.iter().any(|s| currency_amount_like(s) && !global_ignore_list.contains(s));
                                            if amount_answer && prior_amount {
                                                emit_term(&format!("  💱 [CURRENCY AMOUNT STOP] '{}' 에 금액 '{}' 반환 ({}). 이 필드에서 금액 답을 이미 한 번 버렸는데 다시 금액이 와서, 이 문서에는 통화 표지가 없다고 보고 재시도를 멈춥니다. 값은 비워 두고, 저장 직전 문서 언어 기본 통화가 들어갑니다.", k, extracted_str, why));
                                                crate::utils::score_dynamics::record_baseline("commerce.currency_amount_stop", 1.0);
                                                if let Some(o) = item_val.as_object_mut() {
                                                    o.remove(*k);
                                                }
                                                found_valid_value = true;
                                                continue;
                                            }
                                            emit_term(&format!("  🚫 [ENUM VALUE GATE] '{}' 에 '{}' 반환 ({}). 열거형 값이 될 수 없어 폐기 후 재시도합니다.", k, extracted_str, why));
                                            requires_retry = true;
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            continue;
                                        }

                                        found_valid_value = true;

                                        if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            
                                            if !is_enum_field {
                                                let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                
                                                if !is_iso_date && !is_url && !is_boolean_str {
                                                    let mut is_matched = doc_title.contains(&extracted_str);
                                                    
                                                    if !is_matched {
                                                        let extracted_lower = extracted_str.to_lowercase();
                                                        let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                        
                                                        for ctx_val in &json_contexts {
                                                            if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                let target_lower = target_val_str.to_lowercase();
                                                                
                                                                if target_lower.contains(&extracted_lower) {
                                                                    if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                        let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                        if tokens.contains(&extracted_lower.as_str()) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    } else {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
                                                                
                                                                if !is_matched && digits_only.len() >= 3 {
                                                                    let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                    if target_digits.contains(&digits_only) {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }

                                                    if !is_matched {
                                                        requires_retry = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }


                            if !found_valid_value {
                                requires_retry = true;
                            }
                            if !is_synthesis_field || requires_retry {
                                crate::model::lang_llm::record_engine_outcome(
                                    "commerce",
                                    if is_synthesis_field { "detail_synthesis" } else { "detail_field" },
                                    used_lang,
                                    !requires_retry,
                                );
                            }

                            if requires_retry {
                                miss_counter += 1;
                                crate::utils::score_dynamics::record_baseline("commerce.llm_retry", miss_counter as f32);
                                if miss_counter > 3 {
                                    emit_term(&format!("  ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                    crate::utils::score_dynamics::record_field_seen(&field_name_clone);
                                    crate::utils::score_dynamics::record_field_reject(
                                        &field_name_clone,
                                        crate::utils::score_dynamics::GateKind::Enum,
                                    );
                                    break; 
                                }
                                emit_term(&format!("  ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                crate::utils::score_dynamics::record_near_miss(&field_name_clone);
                                for ex_str in extracted_values_for_retry {
                                    ignore_list.push(ex_str.clone());
                                    ignore_list.push(format!(" {}", ex_str));
                                    ignore_list.push(ex_str.to_lowercase());
                                }

                                if !found_valid_value {
                                    for k in &keys {
                                        ignore_list.push(format!("\"{}\": \"\"", k));
                                        ignore_list.push(format!("\"{}\":\"\"", k));
                                    }
                                }
                                continue;
                            }


                            let mut extracted_results = Vec::new();
                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    extracted_data.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                    extracted_results.push(format!("\"{}\": {}", k, val));
                                    

                                    let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                  else if val.is_number() { val.to_string() }
                                                  else { String::new() };
                                    

                                    if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                        if !global_ignore_list.contains(&val_str) {
                                            global_ignore_list.push(val_str.clone());
                                            global_ignore_list.push(format!(" {}", val_str));
                                            global_ignore_list.push(val_str.to_lowercase());
                                        }
                                    }
                                }
                            }
                            


                            for ck in ["has_header", "has_footer", "language"] {
                                if let Some(val) = item_val.get(ck) {
                                    extracted_data.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                }
                            }

                            if !extracted_results.is_empty() {
                                emit_term(&format!("  ✅ Extracted: {}", extracted_results.join(", ")));
                            } else {
                                emit_term(&format!("  ✅ Extracted: (null or empty for {})", field_name_clone));
                            }
                            break;
                        },
                        Err(e) => {
                            println!("[Scheduler] Error extracting detail field {}: {:?}", field_name_clone, e);
                            break;
                        }
                    }
                }
            }
            drop(field_lang);
            if !detail_synthesis_jobs.is_empty() {
                let mut detail_items: Vec<Value> = vec![extracted_data.clone()];
                let detail_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::from([(0usize, 0usize)]);
                run_synthesis_pass(
                    &model,
                    &mut detail_items,
                    &detail_synthesis_jobs,
                    &detail_pos,
                    &page_type,
                    &doc_lang,
                    &lang_probe_text,
                    SYNTHESIS_SKIP_DROP,
                    cancellation_token,
                    &emit_term,
                    "detail_synthesis",
                )
                .await?;
                if let Some(done) = detail_items.pop() {
                    extracted_data = done;
                }
            }
            let own_family = crate::utils::canonical::relay_type_family(&page_type);
            let related_types = crate::logic::related(&page_type);
            let relay_foreign: Vec<&str> = ["goods", "order", "tracking"]
                .iter()
                .copied()
                .filter(|t| *t != own_family.as_str() && related_types.iter().any(|r| r == t))
                .collect();
            if !relay_foreign.is_empty() {
                let self_token = extracted_data.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let cands = collect_id_link_candidates_capped(&pug_lines_ref, usize::MAX);
                let mut relay_bound = 0usize;
                for (ftype, token, role) in collect_typed_relay_refs(&cands, &self_token, &relay_foreign) {
                    let aliases = crate::logic::relay_type_aliases(&ftype);
                    let keyed = cands
                        .iter()
                        .any(|c| same_id_token(&c.token, &token) && candidate_type_evidence(c, aliases) >= 2);
                    if !keyed {
                        continue;
                    }
                    let key = match crate::utils::canonical::relay_key_for_type(&ftype) {
                        Some(k) => k,
                        None => continue,
                    };
                    let f_index = entity_key_index(&ftype, &team_id, &task.cc, &token);
                    let obj = match extracted_data.as_object_mut() {
                        Some(o) => o,
                        None => break,
                    };
                    match obj.get(key) {
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                        Some(v) if crate::utils::canonical::relay_ref_index(Some(v)).is_some() => continue,
                        _ => {}
                    }
                    let companion = format!("{}_title", key);
                    let text_val = obj
                        .get(key)
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| crate::utils::canonical::relay_text_is_content(s));
                    let companion_empty = obj
                        .get(&companion)
                        .and_then(|v| v.as_str())
                        .map_or(true, |s| s.trim().is_empty());
                    if let Some(t) = text_val {
                        if companion_empty {
                            obj.insert(companion.clone(), json!(t));
                        }
                    }
                    obj.insert(key.to_string(), json!(f_index));
                    relay_ledger::mark_relay_bound(&mut extracted_data, key);
                    relay_bound += 1;
                    emit_term(&format!(
                        "  🔗 [DETAIL RELAY REF BIND] {} ← {} '{}' (역할 '{}' · 주소 매개변수 이름이 {} 별칭) → {} = {} | 상세 페이지에 인쇄된 상대 문서 링크의 식별자로 목록과 같은 식의 상대 index 를 만들고, 글자 값은 {}_title 로 옮깁니다. 목록 문서 없이 상세만 들어와도 저장 직전 정리 단계가 이 연결을 0 으로 지우지 않습니다.",
                        page_type, ftype, token, role, ftype, key, f_index, key
                    ));
                }
                crate::utils::score_dynamics::record_baseline("commerce.detail_relay_ref_bound", relay_bound as f32);
            }
        }
    }

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    
    let search_mode_str = search_mode.clone();
    let doc_lang_str = doc_lang.clone();
    
    
    
    let task_flag = task_data.get("flag")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let normalize_data = |item: &mut serde_json::Value| {
        if let Some(obj) = item.as_object_mut() {
            if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type.clone())); }
            
            if obj.get("mode").is_none() { obj.insert("mode".to_string(), json!(search_mode_str.clone())); }
            if obj.get("flag").is_none() && !task_flag.is_empty() {
                obj.insert("flag".to_string(), json!(task_flag.clone()));
            }
            

            let currency_val = obj.get("currency").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            let currency_norm = normalize_currency_value(&currency_val, &doc_lang_str);
            if currency_code_of(&currency_val).is_none() && currency_amount_like(&currency_val) {
                println!("[Scheduler] 💱 [CURRENCY AMOUNT FALLBACK] currency='{}' 는 글자·통화기호 없는 금액 모양이라 통화가 될 수 없어 문서 언어 기본 통화 '{}' 로 대체합니다.", currency_val, currency_norm);
            }
            obj.insert("currency".to_string(), json!(currency_norm));
            

            if let Some(q) = obj.get("quantity").cloned() {
                let q_val: Option<i64> = if q.is_number() {
                    q.as_i64().or_else(|| q.as_f64().filter(|f| f.is_finite()).map(|f| f.round() as i64))
                } else if let Some(s) = q.as_str() {
                    let lead: String = s
                        .trim()
                        .chars()
                        .filter(|c| *c != ',' && !c.is_whitespace())
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                        .collect();
                    lead.parse::<i64>()
                        .ok()
                        .or_else(|| lead.parse::<f64>().ok().filter(|f| f.is_finite()).map(|f| f.round() as i64))
                } else {
                    None
                };
                match q_val {
                    Some(v) => {
                        obj.insert("quantity".to_string(), json!(v));
                    }
                    None => {
                        println!("[Scheduler] 🧮 [QUANTITY KEEP] quantity={} 는 수치로 읽을 수 없어 0 으로 바꾸지 않고 비웁니다. 0 은 '재고 없음·수량 0' 이라는 다른 사실이라, 읽지 못한 값을 0 으로 저장하면 검색 조건(재고 0 · 수량 합)이 거짓 문서를 만납니다.", q);
                        obj.remove("quantity");
                    }
                }
            }
            
            
            let date_keys = [
                "registration_date", "order_date", "payment_date", "shipping_date", 
                "manufacture_date", "expiration_date", "release_date", "started_at", "expired_at"
            ];
            if let Ok(re_date) = regex::Regex::new(r"\d+") {
                for key in date_keys.iter() {
                    if let Some(date_val) = obj.get(*key).and_then(|v| v.as_str()) {
                        let s = date_val.trim();
                        if !s.is_empty() && s != "null" {

                            if s.chars().all(char::is_numeric) && (s.len() == 10 || s.len() == 13) {
                                if let Ok(ts) = s.parse::<i64>() {
                                    let ts_ms = if s.len() == 10 { ts * 1000 } else { ts };
                                    if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
                                        let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                                        obj.insert(key.to_string(), json!(iso_date));
                                        continue;
                                    }
                                }
                            }


                            if s.contains('T') && s.len() >= 19 {
                                continue;
                            }


                            let nums: Vec<u32> = re_date.find_iter(s).filter_map(|m| m.as_str().parse().ok()).collect();
                            if nums.len() >= 3 {
                                let mut year = nums[0];
                                let mut month = nums[1];
                                let mut day = nums[2];


                                if day > 31 && year <= 31 {
                                    year = nums[2];
                                    day = nums[1];
                                    month = nums[0];
                                }


                                if year < 100 {
                                    year += if year > 50 { 1900 } else { 2000 };
                                }
                                
                                month = month.clamp(1, 12);
                                day = day.clamp(1, 31);
                                
                                let hour = if nums.len() > 3 { nums[3].clamp(0, 23) } else { 0 };
                                let minute = if nums.len() > 4 { nums[4].clamp(0, 59) } else { 0 };
                                let second = if nums.len() > 5 { nums[5].clamp(0, 59) } else { 0 };
                                
                                let iso_date = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hour, minute, second);
                                obj.insert(key.to_string(), json!(iso_date));
                            }
                        }
                    } else if let Some(date_num) = obj.get(*key).and_then(|v| v.as_i64()) {

                        let ts_ms = if date_num < 10_000_000_000 { date_num * 1000 } else { date_num };
                        if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
                            let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                            obj.insert(key.to_string(), json!(iso_date));
                        }
                    }
                }
            }


            if obj.get("started_at").is_none() || obj.get("started_at").unwrap().is_null() || obj.get("started_at").unwrap().as_str() == Some("") {
                if let Some(m) = obj.get("manufacture_date").cloned() { obj.insert("started_at".to_string(), m); }
            }
            if obj.get("expired_at").is_none() || obj.get("expired_at").unwrap().is_null() || obj.get("expired_at").unwrap().as_str() == Some("") {
                if let Some(e) = obj.get("expiration_date").cloned() { obj.insert("expired_at".to_string(), e); }
            }
            

            if let Some(cond) = obj.get("condition").and_then(|v| v.as_str()) {
                let cond_lower = cond.to_lowercase();
                if cond_lower.contains("used") { obj.insert("used".to_string(), json!(1)); }
                if cond_lower.contains("lease") { obj.insert("lease".to_string(), json!(2)); }
                if cond_lower.contains("rental") { obj.insert("rental".to_string(), json!(3)); }
                if cond_lower.contains("refurbish") { obj.insert("refurbish".to_string(), json!(4)); }
            }
        }
    };

    {
        let status_values: Vec<String> = if is_detail {
            extracted_data.get("status").and_then(|x| x.as_str()).map(|s| vec![s.to_string()]).unwrap_or_default()
        } else {
            extracted_data
                .get("items")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|it| it.get("status").and_then(|x| x.as_str()).map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut status_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let mut status_pending: Vec<String> = Vec::new();
        for s in status_values.iter() {
            let t = s.trim();
            if t.is_empty() || t == "null" || crate::logic::parse_status(t) != 0 { continue; }
            if status_map.contains_key(t) || status_pending.iter().any(|p| p == t) { continue; }
            let lower = t.to_lowercase();
            if crate::logic::parse_status(&lower) != 0 {
                status_map.insert(t.to_string(), lower);
            } else {
                status_pending.push(t.to_string());
            }
        }
        if !status_pending.is_empty() {
            let banks = status_key_banks(&model, &page_type).await;
            let pivot = crate::utils::ai_utils::status_pivot_bank(&model, &page_type, &doc_lang).await;
            if is_detail {
                if let Some(p) = pivot.as_ref() {
                    crate::utils::score_dynamics::record_baseline("commerce.status_pivot_align_z", p.align_z);
                }
            }
            for raw in status_pending.iter() {
                if let Some((key, route)) = crate::utils::ai_utils::status_canonical_exact(raw, &page_type, pivot.as_ref()) {
                    emit_term(&format!("  🌉 [STATUS PIVOT] '{}' → '{}' | {} | 닫힌 상태 어휘와 정확히 맞아 코사인 비교 없이 확정합니다.", raw, key, route));
                    crate::utils::score_dynamics::record_baseline("commerce.status_pivot_hit", 1.0);
                    status_map.insert(raw.clone(), key);
                    continue;
                }
                let emb = model.get_embedding(raw.clone()).await.unwrap_or(vec![0.0f32; 384]);
                let pick = pick_status_key(&status_key_scores(&emb, &banks));
                if pick.second > f32::MIN {
                    crate::utils::score_dynamics::record_baseline("commerce.status_canonical_margin", pick.top - pick.second);
                }
                if pick.accepted_with(0.35, 0.03) && crate::logic::parse_status(&pick.key) != 0 {
                    emit_term(&format!("  🧭 [STATUS CANONICAL] '{}' → '{}' | Top: {:.4} | Margin: {:+.4} (parse_status 가 모르는 원문은 저장 시 0 이 되므로 상태 캐노니컬 뱅크로 환산합니다)", raw, pick.key, pick.top, pick.margin()));
                    status_map.insert(raw.clone(), pick.key.clone());
                } else {
                    emit_term(&format!("  ⚪ [STATUS CANONICAL UNRESOLVED] '{}' | '{}' Top: {:.4} / 2nd: {:.4} → 확정 불가, 원문을 유지합니다.", raw, pick.key, pick.top, pick.second));
                }
            }
        }
        if !status_map.is_empty() {
            let apply_status = |v: &mut serde_json::Value| {
                let cur = v.get("status").and_then(|x| x.as_str()).map(|s| s.trim().to_string());
                if let Some(c) = cur {
                    if let Some(k) = status_map.get(&c) {
                        if let Some(o) = v.as_object_mut() {
                            o.insert("status".to_string(), json!(k));
                        }
                    }
                }
            };
            if is_detail {
                apply_status(&mut extracted_data);
            } else if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
                for it in items.iter_mut() {
                    apply_status(it);
                }
            }
        }
    }

    if is_detail {
        normalize_data(&mut extracted_data);
    } else {
        if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
            for item in items.iter_mut() {
                normalize_data(item);
            }
        }
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    
    {
        println!("[Scheduler] Generating natural language sentences for FTS/Vector matching and Privacy Masking...");
        

        let should_mask = page_type != "goods";

        if is_detail {
            let original_lang_text = parsing::json_to_natural_language(&extracted_data);
            

            let masked_lang_text = original_lang_text.clone();

            if let Some(obj) = extracted_data.as_object_mut() {
                obj.insert("text".to_string(), json!(original_lang_text));
                obj.insert("masked_text".to_string(), json!(masked_lang_text));
            }
        } else {
            if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
                for item in items.iter_mut() {
                    let original_lang_text = parsing::json_to_natural_language(item);
                    

                    let masked_lang_text = original_lang_text.clone();

                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("text".to_string(), json!(original_lang_text));
                        obj.insert("masked_text".to_string(), json!(masked_lang_text));
                    }
                }
            }
        }
    }

    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");

        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));

        let kept_lang = if crate::model::lang_llm::translit_demand(&extracted_data) > 0 {
            model.deep_purge_keep_lang(&doc_lang).await
        } else {
            model.deep_purge_resources().await;
            None
        };
        match kept_lang.as_deref() {
            Some(code) => emit_term(&format!(
                "[Scheduler] 🔁 [HANDOVER KEEP] 필드 추출에 쓴 Qwen3.5-4B-{} 를 음차 단계까지 둡니다. 음차가 같은 언어 모델을 쓰므로 내렸다가 다시 올리는 적재 한 번을 줄입니다. 임베딩 적재가 동시 상주를 허용하지 않으면 그 자리에서 기존 교차 규칙대로 내립니다.",
                code
            )),
            None => {
                model.mark_crossover_idle();

                tokio::time::sleep(std::time::Duration::from_millis(400)).await;

                crate::utils::resources::wait_for_resources_settled(1200, 800, Some(cancellation_token), model.device_config.gpu_id as u32).await?;
            }
        }
        emit_term(&format!("[Scheduler] {}", model.crossover_report()));
    }

    {
        emit_term("[Scheduler] 🔤 Preparing crossover for synonym expansion...");
        let translit_demand = crate::model::lang_llm::translit_demand(&extracted_data);
        let (translit_engine, translit_status) = crate::model::lang_llm::await_engine(
            &doc_lang,
            app_handle,
            &task.id,
            cancellation_token,
            translit_demand,
        )
        .await;
        emit_term(&format!("[Scheduler]    {}", translit_status));
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Planning VRAM crossover...", "spinner": "🔤" }));

        let free_mb = model.get_free_vram_mb();
        let embed_need = model.embedding_budget_mb();
        let gen_resident = translit_engine.code().is_some()
            && crate::model::lang_llm::resident_variant().as_deref() == translit_engine.code()
            && model.qwen3_5_generator.lock().await.is_some();
        let gen_need = match translit_engine.code() {
            Some(_) if gen_resident => 0,
            Some(code) => crate::model::lang_llm::resident_estimate_mb(code),
            None => model.generation_budget_mb(crate::model::ModelSize::Qwen3_5),
        };
        let gen_label = if gen_resident {
            format!("{} (이미 상주)", translit_engine.label())
        } else {
            translit_engine.label()
        };
        if free_mb >= embed_need + gen_need {
            emit_term(&format!(
                "[Scheduler] 🤝 [CROSSOVER/COEXIST 예상] 자유 {}MB >= 임베딩 {}MB + {} {}MB. 스왑 없이 진행할 수 있습니다.",
                free_mb, embed_need, gen_label, gen_need
            ));
        } else {
            emit_term(&format!(
                "[Scheduler] 🔁 [CROSSOVER/SWAP 예상] 자유 {}MB < 임베딩 {}MB + {} {}MB. 페이즈별 교차 상주로 진행합니다.",
                free_mb, embed_need, gen_label, gen_need
            ));
        }
        model.check_embedding_downloaded().await?;
        emit_term("[Scheduler] ✅ [CROSSOVER] 지연 로드 계획 확정. 각 페이즈 진입 시점에 필요한 모델만 올립니다.");
    }
    let extracted_id_raw = extracted_data.get("id")
        .or_else(|| extracted_data.get("no"))
        .or_else(|| extracted_data.get("code"))
        .or_else(|| extracted_data.get("tracking_number"))
        .or_else(|| extracted_data.get("index"))
        .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
        .unwrap_or_default();
    let url_self_token: Option<(String, String, u8)> = if is_detail {
        url_typed_self_token(&url, &page_type)
    } else {
        None
    };
    let id_val_raw = match url_self_token.as_ref() {
        Some((tok, role, ev)) => {
            crate::utils::score_dynamics::record_baseline("commerce.detail_url_identity", 1.0);
            emit_term(&format!(
                "  🪪 [DETAIL URL IDENTITY] {} 상세 식별자 '{}' (URL 역할 '{}' · 타입 근거 {}) | 추출기 id '{}' 대신 목록이 쓰는 것과 같은 URL 자기 식별자로 index 를 만듭니다. 그래야 목록 행과 상세 페이지가 같은 문서로 합쳐지고 다른 문서의 참조도 같은 index 로 착지합니다.",
                page_type, tok, role, ev, extracted_id_raw
            ));
            tok.clone()
        }
        None if !extracted_id_raw.trim().is_empty() => extracted_id_raw.clone(),
        None if is_detail => {
            let seed = match url::Url::parse(&url) {
                Ok(u) => format!("{}{}", u.path(), u.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase(),
                Err(_) => url.to_string(),
            };
            let auto = format!("AUTO-{}-{}", page_type, crate::utils::hash::digest(&seed));
            emit_term(&format!(
                "  ⚠️ [DETAIL ID FALLBACK] 상세 식별자가 URL 에도 추출 결과에도 없어 링크 기반 결정론 키 '{}' 를 씁니다. 빈 값으로 index 를 만들면 식별자 없는 상세 페이지가 전부 한 행으로 합쳐집니다.",
                auto
            ));
            auto
        }
        None => extracted_id_raw.clone(),
    };
    let index_val = entity_key_index(&page_type, &team_id, &task.cc, &id_val_raw);
    let generated_id = entity_id(&team_id, index_val);

    if let Some(obj) = extracted_data.as_object_mut() {
        obj.insert("index".to_string(), json!(index_val));
        obj.insert("id".to_string(), json!(generated_id.clone()));
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    log_task_progress(app_handle, &task.id, &json!({ "category": "Saving", "summary": "Syncing to database..." }));

    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    let mut stats_diff: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();

    if page_type == "order" {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            let tracking_number = extracted_data.get("tracking_number").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let clean_tracking_no = normalize_entity_key(&tracking_number);
            let first_goods = goods_arr.iter().find_map(|good| {
                good.get("id")
                    .or_else(|| good.get("no"))
                    .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.trim().to_string()) })
                    .filter(|s| !s.is_empty())
            });
            match (clean_tracking_no.is_empty(), first_goods) {
                (true, Some(_)) => {
                    emit_term(&format!(
                        "  ⚪ [ORDER-ITEM BRIDGE SKIP] 주문 '{}' 은 송장번호가 비어 tracking 문서를 만들지 않습니다. 빈 값으로 index 를 만들면 송장 없는 모든 주문이 tracking 한 행으로 합쳐집니다. 상품 연결은 goods 배열 릴레이가 처리합니다.",
                        id_val_raw
                    ));
                }
                (false, Some(g_no)) => {
                    let tracking_index = entity_key_index("tracking", &team_id, &task.cc, &tracking_number);
                    let goods_index = entity_key_index("goods", &team_id, &task.cc, &g_no);
                    let tracking_id = entity_id(&team_id, tracking_index);
                    let mut tracking_data = extracted_data.clone();
                    if let Some(obj) = tracking_data.as_object_mut() {
                        obj.insert("type".to_string(), json!("tracking"));
                        obj.insert("no".to_string(), json!(clean_tracking_no));
                        obj.insert("index".to_string(), json!(tracking_index));
                        obj.insert("goods".to_string(), json!(goods_index));
                        obj.insert("order".to_string(), json!(index_val));
                        obj.insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!("count"));
                    }
                    let tracking_text = parsing::json_to_natural_language(&tracking_data);
                    let masked_tracking_text = tracking_text.clone();
                    let tracking_vector = model.get_embedding(tracking_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(tracking_text));
                    tracking_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_tracking_text));
                    let tracking_bcc = entity_bcc("tracking", &task.cc);
                    let tracking_ref = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref));
                    let tracking_prior = crate::utils::canonical::ledger_prior(
                        store
                            .get_item_by_id("tracking", &tracking_id)
                            .await
                            .ok()
                            .flatten()
                            .and_then(|d| serde_json::from_str::<serde_json::Value>(&d.json_data).ok())
                            .as_ref(),
                    );
                    relay_ledger::add_delta(&mut stats_diff, "tracking", crate::utils::canonical::ledger_delta(tracking_prior, true));
                    save_item(&store, "tracking", &tracking_id, "tracking", tracking_data, Some(tracking_vector),
                        &task.from, &team_id, &task.cc, &tracking_bcc, &tracking_ref, None).await;
                }
                _ => {}
            }
        }
    }

    
    let target_table = match page_type.as_str() {
        "sales" | "goods" | "order" => "sales",
        "tracking" | "receiving" | "shipping" => "tracking",
        "event" | "coupon" => "event",
        "member" | "team" | "user" => "users",
        "talk" | "prompt" | "ai_search" => "talks",
        _ => "items",
    }.to_string();

    
    
    let cc_val = task.cc.clone();
    let bcc = entity_bcc(&page_type, &cc_val);
    let ref_val = task.r#ref.clone();
    let mut items_to_process = Vec::new();

    if is_detail {
        

        let text_to_embed = extracted_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&extracted_data));
        let item_digest = crate::utils::hash::digest(&text_to_embed);
        let mut target_id = generated_id.clone();

        let mut existing_vector = None;
        let mut existing_json_found: Option<serde_json::Value> = None;

        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            if let Ok(existing_json) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                let old_digest = existing_json.get("digest").and_then(|d| d.as_str()).unwrap_or("");
                if old_digest == item_digest {
                    existing_vector = Some(existing_item.vector);
                }
                extracted_data = merge_node(&existing_json, &extracted_data);
                existing_json_found = Some(existing_json);
            }
        }
        else if !url.is_empty() {
            let normalized_link = if let Ok(parsed_url) = url::Url::parse(&url) {
                format!("{}{}", parsed_url.path(), parsed_url.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase()
            } else {
                url.clone()
            };
            if let Ok(Some((found_id, json_val))) = store.find_item_by_property(&target_table, "link", &json!(normalized_link)).await {
                target_id = found_id.clone();

                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
                    if let Ok(ej) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                        let old_digest = ej.get("digest").and_then(|d| d.as_str()).unwrap_or("");
                        if old_digest == item_digest {
                            existing_vector = Some(existing_item.vector);
                        }
                    }
                }

                extracted_data = merge_node(&json_val, &extracted_data);
                if let Some(obj) = extracted_data.as_object_mut() {
                    obj.insert("id".to_string(), json!(target_id.clone()));
                    if let Some(found_index) = crate::utils::canonical::relay_ref_index(json_val.get("index")) {
                        obj.insert("index".to_string(), json!(found_index));
                    }
                }
                existing_json_found = Some(json_val);
            }
        }
        if let Some(prior) = existing_json_found.as_ref() {
            let relay_kept = crate::utils::canonical::keep_relay_index(prior, &mut extracted_data, &page_type);
            if !relay_kept.is_empty() {
                crate::utils::score_dynamics::record_baseline("commerce.detail_relay_kept", relay_kept.len() as f32);
                emit_term(&format!(
                    "  ⚓ [DETAIL MERGE / RELAY KEEP] {} '{}' | 목록이 확정한 연결 index {:?} 를 상세 병합 뒤에도 유지합니다. 상세 페이지가 같은 키에 글자 값(상품명 등)을 가져와 index 를 덮으면, 저장 직전 정리 단계가 그 글자를 {{키}}_title 로 옮기면서 연결이 0 으로 사라져 기간 연결 검색이 이 문서를 찾지 못합니다. 상세의 글자 값은 {{키}}_title 이 비어 있을 때만 그 자리에 둡니다.",
                    page_type, target_id, relay_kept
                ));
            }
            let shape_kept = crate::utils::canonical::keep_value_shapes(prior, &mut extracted_data);
            if !shape_kept.is_empty() {
                crate::utils::score_dynamics::record_baseline("commerce.detail_shape_kept", shape_kept.len() as f32);
                emit_term(&format!(
                    "  ⚓ [DETAIL MERGE / SHAPE KEEP] {} '{}' | 필드 {:?} 는 기존 값의 모양(주소 · 코드 · 수치·날짜)과 상세 값의 모양이 달라 기존 값을 유지합니다. 상세 폼의 버튼 글자('미리보기')나 라디오 코드('0')가 상품코드 · 대표 이미지 주소를 덮지 않게 합니다. 글자 값끼리의 교체는 막지 않습니다.",
                    page_type, target_id, shape_kept
                ));
            }
        }
        let self_prior = crate::utils::canonical::ledger_prior(existing_json_found.as_ref());
        let origin_ok = relay_ledger::placeholder_origin_establishes(&page_type, existing_json_found.as_ref());

        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        if let Some(t_idx) = relay_ledger::bind_tracking_ref(&mut extracted_data, &page_type, &team_id, &task.cc) {
            emit_term(&format!(
                "  🔑 [TRACKING INDEX PRE-COMPUTE] 주문의 tracking 연결 index {} 확정. 유효한 송장번호는 사이트와 무관한 시드로 만들어, 다른 사이트나 행동 로그가 같은 송장을 가리켜도 같은 index 로 만납니다.",
                t_idx
            ));
        }
        let detail_extra = relay_ledger::goods_array_refs(&mut extracted_data, &team_id, &task.cc);
        let self_index = crate::utils::canonical::relay_ref_index(extracted_data.get("index")).unwrap_or(index_val);
        let bridge = {
            let relay_env = relay_ledger::RelayEnv {
                store: &store,
                app_handle,
                task_id: &task.id,
                team_id: &team_id,
                from: &task.from,
                cc: &task.cc,
                ref_val: &ref_val,
                search_mode: &search_mode,
            };
            relay_ledger::bridge_relays(&relay_env, &page_type, &target_id, self_index, &mut extracted_data, &detail_extra, &mut stats_diff).await
        };
        let confirm = relay_ledger::self_established(self_prior, origin_ok, &bridge);
        let self_delta = crate::utils::canonical::ledger_delta(self_prior, confirm);
        relay_ledger::add_delta(&mut stats_diff, &page_type, self_delta);
        let self_state = crate::utils::canonical::ledger_state(self_prior, confirm);
        let self_ts = if self_state == "count" { chrono::Utc::now().timestamp_millis() } else { 0 };
        if let Some(obj) = extracted_data.as_object_mut() {
            obj.insert("updated_at".to_string(), json!(self_ts));
            obj.insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!(self_state));
        }
        emit_term(&format!(
            "  📒 [RELAY LEDGER / SELF] {} '{}' 상세 문서 | 이전 상태 {:?} | 성립 관계 정방향 {}건 · 역방향 {}건 · 자리 초안 출처 성립 {} | {} | 원장 변화 {:?}",
            page_type, target_id, self_prior, bridge.establishing_out, bridge.referrers, origin_ok,
            if self_state == "count" { "count 확정" } else { "draft 유지 (성립 관계 없음)" },
            self_delta
        ));

        save_item(&store, &target_table, &target_id, &page_type, extracted_data.clone(), vector,
            &task.from, &team_id, &task.cc, &bcc, &ref_val, Some(&item_digest)).await;
        items_to_process.push(extracted_data.clone());

        
        
        
        
        
        //
        
        
        {
            let natural_text = crate::nl_convert::json_to_natural_language(&extracted_data);

            
            let raw_chunks = crate::nl_convert::split_natural_language_to_chunks(&natural_text);
            emit_term(&format!("  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크", raw_chunks.len()));
            for (ci, (ct, cp, confirmed)) in raw_chunks.iter().enumerate() {
                let flag = if *confirmed { "✓" } else { "?" };
                emit_term(&format!("    [{}] {} property='{}' | text='{}'", ci, flag, cp, ct));
            }

            if !raw_chunks.is_empty() {
                
                
                let fields = crate::parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
                let mut idx_field_names: Vec<String> = Vec::new();
                let mut idx_field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut idx_field_phrase_weights: Vec<Vec<f32>> = Vec::new();
                let mut idx_field_formats: Vec<String> = Vec::new();

                for (fname, _, bias_target, _) in &fields {
                    
                    
                    
                    
                    
                    
                    let lower_fname = fname.to_lowercase();
                    let _is_synthesis = lower_fname.contains("insight")
                        || lower_fname.contains("summary")
                        || lower_fname.contains("analysis");

                    let (mut phrases, mut weights) = crate::utils::ai_utils::split_bias_phrases_weighted_full(bias_target);

                    
                    
                    
                    
                    let bridge_ph = crate::utils::ai_utils::abstract_bridge_field_phrases(fname);
                    if !bridge_ph.is_empty() {
                        emit_term(&format!("  🌉 [ABSTRACT BRIDGE MERGE] '{}' 뱅크에 추상 수식어 브릿지 구 {}개 편입", fname, bridge_ph.len()));
                    }
                    for p in bridge_ph {
                        if phrases.iter().any(|e| e == &p) { continue; }
                        phrases.push(p);
                        weights.push(1.0);
                    }

                    let phrase_embs = if phrases.is_empty() {
                        vec![vec![0.0f32; 384]]
                    } else {
                        model.get_embedding_batch(phrases.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                    };

                    let fmt_str = crate::nl_convert::field_format_to_string(fname);

                    idx_field_names.push(fname.clone());
                    idx_field_phrase_embs.push(phrase_embs);
                    idx_field_phrase_weights.push(weights);
                    idx_field_formats.push(fmt_str);
                }

                emit_term(&format!(
                    "  📐 [PHASE B+C] 필드 뱅크 구축 완료: {}개 필드 (PLINKO GAME 입력)",
                    idx_field_names.len()
                ));

                
                
                
                let model_for_embed = model.clone();
                let enriched_chunks = crate::nl_convert::run_phase_b_pipeline(
                    &raw_chunks,
                    &doc_lang,
                    &page_type,
                    &idx_field_names,
                    &idx_field_phrase_embs,
                    &idx_field_phrase_weights,
                    &idx_field_formats,
                    move |text: String| {
                        let m = model_for_embed.clone();
                        async move {
                            m.get_embedding(text).await.unwrap_or(vec![0.0; 384])
                        }
                    },
                ).await;
                crate::nl_convert::log_enriched_chunks(&enriched_chunks);

                if !enriched_chunks.is_empty() {
                    
                    let indexable_chunks: Vec<(usize, &crate::nl_convert::ChunkMetadata)> = enriched_chunks.iter()
                        .enumerate()
                        .filter(|(_, c)| c.property != "unclassified")
                        .collect();

                    let skipped_count = enriched_chunks.len() - indexable_chunks.len();
                    if skipped_count > 0 {
                        emit_term(&format!(
                            "  🚫 [PHASE D FILTER] unclassified 청크 {}개 인덱싱 제외",
                            skipped_count
                        ));
                    }

                    if indexable_chunks.is_empty() {
                        emit_term("  ⚠️ [PHASE D] 인덱싱 대상 청크가 없습니다. 건너뜁니다.");
                    } else {
                        let chunk_texts: Vec<String> = indexable_chunks.iter()
                            .map(|(_, c)| c.chunk_text.clone())
                            .collect();

                        let chunk_embs = model.get_embedding_batch(chunk_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk_texts.len()]);

                        let metas: Vec<&crate::nl_convert::ChunkMetadata> =
                            indexable_chunks.iter().map(|(_, c)| *c).collect();
                        let alias_pairs = generate_transliteration_aliases(
                            &model,
                            &metas,
                            &doc_lang,
                            &page_type,
                            cancellation_token,
                            app_handle,
                            &task.id,
                        ).await;
                        
                        let _ = store.delete_chunks_by_item(&target_id).await;

                        let mut anchor_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                        let mut localized_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                        for (_, cm) in indexable_chunks.iter() {
                            let a = crate::utils::ai_utils::indexing_anchor_text(
                                &doc_lang, &page_type, &cm.property,
                            );
                            let leaf = crate::utils::ai_utils::indexing_leaf_label(
                                &doc_lang, &page_type, &cm.property,
                            );
                            let v = cm.value_part.trim();
                            let l = if v.is_empty() { leaf.clone() } else { format!("{} {}", leaf, v) };
                            anchor_texts.push(a);
                            localized_texts.push(l);
                        }
                        let anchor_embs = model.get_embedding_batch(anchor_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_texts.len()]);
                        let localized_embs = model.get_embedding_batch(localized_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; localized_texts.len()]);

                        let mut alias_saved = 0usize;

                        for (ei, (ci, chunk_meta)) in indexable_chunks.iter().enumerate() {
                            let chunk_id = format!("{}_{}", target_id, ci);

                            let chunk_vec = &chunk_embs[ei];
                            let anchor_emb = &anchor_embs[ei];
                            let localized_emb = &localized_embs[ei];

                            
                            let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
                                "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
                                _ => (0.40f32, 0.30f32, 0.30f32),
                            };

                            let mut final_vec = vec![0.0f32; 384];
                            for d in 0..384 {
                                final_vec[d] = chunk_vec[d] * w_chunk
                                    + anchor_emb[d] * w_anchor
                                    + localized_emb[d] * w_local;
                            }
                            
                            let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for d in 0..384 { final_vec[d] /= norm; }
                            }

                            let _ = store.upsert_chunk(
                                &chunk_id,
                                &target_id,
                                &page_type,
                                &chunk_meta.chunk_text,
                                &chunk_meta.property,
                                &chunk_meta.property_format,
                                &chunk_meta.value_part,
                                Some(final_vec),
                                Some(&task.cc),
                                Some(&bcc),
                                Some(&ref_val),
                                Some(&search_mode),
                            ).await;

                            
                            alias_saved += upsert_alias_chunks(
                                &store,
                                &model,
                                &target_id,
                                &chunk_id,
                                &page_type,
                                &doc_lang,
                                chunk_meta,
                                &alias_pairs[ei],
                                &task.cc,
                                &bcc,
                                &ref_val,
                                &search_mode,
                            ).await;
                        }

                        emit_term(&format!(
                            "  🧩 [PHASE A~E] 청크 인덱싱 완료: item_id='{}' | 청크 {}개 (전체 {}개 중) | 음차 별칭 {}개 | table='item_chunks'",
                            target_id, indexable_chunks.len(), enriched_chunks.len(), alias_saved
                        ));
                    }
                }
            }
        }
        
        
        
        
    } else {
        
        if let Some(items) = extracted_data.get("items").and_then(|v| v.as_array()) {
            let relay_env = relay_ledger::RelayEnv {
                store: &store,
                app_handle,
                task_id: &task.id,
                team_id: &team_id,
                from: &task.from,
                cc: &task.cc,
                ref_val: &ref_val,
                search_mode: &search_mode,
            };
            for item_val in items.iter() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let mut single_item = item_val.clone();
                
                let original_id = single_item.get("id")
                    .or_else(|| single_item.get("no"))
                    .or_else(|| single_item.get("code"))
                    .or_else(|| single_item.get("tracking_number"))
                    .or_else(|| single_item.get("index"))
                    .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
                    .unwrap_or_else(|| single_item.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string());

                let identity_seed = if original_id.trim().is_empty() {
                    let seed = single_item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| serde_json::to_string(&single_item).unwrap_or_default());
                    let auto = format!("AUTO-{}-{}", page_type, crate::utils::hash::digest(&seed));
                    emit_term(&format!(
                        "  ⚠️ [ITEM ID FALLBACK] 식별자가 비어 내용 기반 결정론 키 '{}' 를 사용합니다. (UUID 를 쓰면 재스캔마다 중복 행이 생깁니다)",
                        auto
                    ));
                    auto
                } else {
                    original_id.clone()
                };
                let index_val = entity_key_index(&page_type, &team_id, &task.cc, &identity_seed);
                let hashed_item_id = entity_id(&team_id, index_val);

                if let Some(obj) = single_item.as_object_mut() {
                    obj.insert("type".to_string(), json!(page_type));
                    obj.insert("detail".to_string(), json!(false));
                    obj.insert("id".to_string(), json!(hashed_item_id.clone()));
                    obj.insert("index".to_string(), json!(index_val));
                    obj.insert("updated_at".to_string(), json!(0));
                }
                relay_ledger::bind_tracking_ref(&mut single_item, &page_type, &team_id, &task.cc);

                let existing_item = store.get_item_by_id(&target_table, &hashed_item_id).await.ok().flatten();
                let existing_json: Option<serde_json::Value> = existing_item
                    .as_ref()
                    .and_then(|d| serde_json::from_str::<serde_json::Value>(&d.json_data).ok());
                let self_prior = crate::utils::canonical::ledger_prior(existing_json.as_ref());
                let origin_ok = relay_ledger::placeholder_origin_establishes(&page_type, existing_json.as_ref());
                if let Some(ej) = existing_json.as_ref() {
                    single_item = merge_node(ej, &single_item);
                    if let Some(obj) = single_item.as_object_mut() {
                        obj.insert("id".to_string(), json!(hashed_item_id.clone()));
                        obj.insert("index".to_string(), json!(index_val));
                        if ej.get("detail").map_or(false, |v| v.as_i64() == Some(1) || v.as_bool() == Some(true)) {
                            obj.insert("detail".to_string(), json!(true));
                        }
                    }
                }

                let bridge = relay_ledger::bridge_relays(&relay_env, &page_type, &hashed_item_id, index_val, &mut single_item, &[], &mut stats_diff).await;
                let confirm = relay_ledger::self_established(self_prior, origin_ok, &bridge);
                let self_delta = crate::utils::canonical::ledger_delta(self_prior, confirm);
                relay_ledger::add_delta(&mut stats_diff, &page_type, self_delta);
                let self_state = crate::utils::canonical::ledger_state(self_prior, confirm);
                let kept_ts = existing_json
                    .as_ref()
                    .and_then(|e| e.get("updated_at").and_then(|v| v.as_i64()))
                    .filter(|t| *t > 0 && self_prior == crate::utils::canonical::LedgerPrior::Confirmed);
                let self_ts = if self_state == "count" {
                    kept_ts.unwrap_or_else(|| chrono::Utc::now().timestamp_millis())
                } else {
                    0
                };
                if let Some(obj) = single_item.as_object_mut() {
                    obj.insert("updated_at".to_string(), json!(self_ts));
                    obj.insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!(self_state));
                }
                emit_term(&format!(
                    "  📒 [RELAY LEDGER / SELF] {} '{}' | 이전 상태 {:?} | 성립 관계 정방향 {}건 · 역방향 {}건 · 자리 초안 출처 성립 {} | {} | 원장 변화 {:?}",
                    page_type,
                    hashed_item_id,
                    self_prior,
                    bridge.establishing_out,
                    bridge.referrers,
                    origin_ok,
                    if self_state == "count" { "count 확정" } else { "draft 유지 (성립 관계 없음)" },
                    self_delta
                ));

                let text_to_embed = if existing_json.is_some() {
                    let mut text_view = single_item.clone();
                    if let Some(o) = text_view.as_object_mut() {
                        match item_val.get("id").filter(|v| !v.is_null()) {
                            Some(raw_id) => {
                                o.insert("id".to_string(), raw_id.clone());
                            }
                            None => {
                                o.remove("id");
                            }
                        }
                    }
                    let merged_text = parsing::json_to_natural_language(&text_view);
                    if let Some(obj) = single_item.as_object_mut() {
                        obj.insert("text".to_string(), json!(merged_text.clone()));
                        obj.insert("masked_text".to_string(), json!(merged_text.clone()));
                    }
                    merged_text
                } else {
                    single_item.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&single_item))
                };
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                let existing_vector = existing_item.as_ref().and_then(|doc| {
                    let old_digest = existing_json
                        .as_ref()
                        .and_then(|e| e.get("digest").and_then(|d| d.as_str()))
                        .unwrap_or("");
                    if old_digest == item_digest { Some(doc.vector.clone()) } else { None }
                });
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else {
                    Some(model.get_embedding(text_to_embed).await?)
                };

                
                save_item(&store, &target_table, &hashed_item_id, &page_type, single_item.clone(), vector,
                    &task.from, &team_id, &task.cc, &bcc, &ref_val, Some(&item_digest)).await;
                items_to_process.push(single_item.clone());

                
                
                
                {
                    let natural_text = crate::nl_convert::json_to_natural_language(&single_item);

                    
                    let raw_chunks = crate::nl_convert::split_natural_language_to_chunks(&natural_text);
                    emit_term(&format!("  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크", raw_chunks.len()));
                    for (ci, (ct, cp, confirmed)) in raw_chunks.iter().enumerate() {
                        let flag = if *confirmed { "✓" } else { "?" };
                        emit_term(&format!("    [{}] {} property='{}' | text='{}'", ci, flag, cp, ct));
                    }

                    if !raw_chunks.is_empty() {
                        
                        let fields = crate::parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
                        let mut idx_field_names: Vec<String> = Vec::new();
                        let mut idx_field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                        let mut idx_field_phrase_weights: Vec<Vec<f32>> = Vec::new();
                        let mut idx_field_formats: Vec<String> = Vec::new();

                        for (fname, _, bias_target, _) in &fields {
                            
                            let lower_fname = fname.to_lowercase();
                            let _is_synthesis = lower_fname.contains("insight")
                                || lower_fname.contains("summary")
                                || lower_fname.contains("analysis");

                            let (mut phrases, mut weights) = crate::utils::ai_utils::split_bias_phrases_weighted_full(bias_target);

                            
                            let bridge_ph = crate::utils::ai_utils::abstract_bridge_field_phrases(fname);
                            for p in bridge_ph {
                                if phrases.iter().any(|e| e == &p) { continue; }
                                phrases.push(p);
                                weights.push(1.0);
                            }

                            let phrase_embs = if phrases.is_empty() {
                                vec![vec![0.0f32; 384]]
                            } else {
                                model.get_embedding_batch(phrases.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                            };

                            let fmt_str = crate::nl_convert::field_format_to_string(fname);

                            idx_field_names.push(fname.clone());
                            idx_field_phrase_embs.push(phrase_embs);
                            idx_field_phrase_weights.push(weights);
                            idx_field_formats.push(fmt_str);
                        }

                        let model_for_embed = model.clone();
                        let enriched_chunks = crate::nl_convert::run_phase_b_pipeline(
                            &raw_chunks,
                            &doc_lang,
                            &page_type,
                            &idx_field_names,
                            &idx_field_phrase_embs,
                            &idx_field_phrase_weights,
                            &idx_field_formats,
                            move |text: String| {
                                let m = model_for_embed.clone();
                                async move {
                                    m.get_embedding(text).await.unwrap_or(vec![0.0; 384])
                                }
                            },
                        ).await;

                        if !enriched_chunks.is_empty() {
                            
                            let indexable_chunks: Vec<(usize, &crate::nl_convert::ChunkMetadata)> = enriched_chunks.iter()
                                .enumerate()
                                .filter(|(_, c)| c.property != "unclassified")
                                .collect();

                            let skipped_count = enriched_chunks.len() - indexable_chunks.len();
                            if skipped_count > 0 {
                                emit_term(&format!(
                                    "  🚫 [PHASE D FILTER] unclassified 청크 {}개 인덱싱 제외",
                                    skipped_count
                                ));
                            }

                            if indexable_chunks.is_empty() {
                                emit_term("  ⚠️ [PHASE D] 인덱싱 대상 청크가 없습니다. 건너뜁니다.");
                            } else {
                                let chunk_texts: Vec<String> = indexable_chunks.iter()
                                    .map(|(_, c)| c.chunk_text.clone())
                                    .collect();

                                let chunk_embs = model.get_embedding_batch(chunk_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk_texts.len()]);

                                
                                
                                
                                let metas: Vec<&crate::nl_convert::ChunkMetadata> =
                                    indexable_chunks.iter().map(|(_, c)| *c).collect();
                                let alias_pairs = generate_transliteration_aliases(
                                    &model,
                                    &metas,
                                    &doc_lang,
                                    &page_type,
                                    cancellation_token,
                                    app_handle,
                                    &task.id,
                                ).await;

                                
                                let _ = store.delete_chunks_by_item(&hashed_item_id).await;

                                
                                
                                
                                let mut anchor_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                                let mut localized_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                                for (_, cm) in indexable_chunks.iter() {
                                    let a = crate::utils::ai_utils::indexing_anchor_text(
                                        &doc_lang, &page_type, &cm.property,
                                    );
                                    let leaf = crate::utils::ai_utils::indexing_leaf_label(
                                        &doc_lang, &page_type, &cm.property,
                                    );
                                    let v = cm.value_part.trim();
                                    let l = if v.is_empty() { leaf.clone() } else { format!("{} {}", leaf, v) };
                                    anchor_texts.push(a);
                                    localized_texts.push(l);
                                }
                                let anchor_embs = model.get_embedding_batch(anchor_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_texts.len()]);
                                let localized_embs = model.get_embedding_batch(localized_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; localized_texts.len()]);

                                let mut alias_saved = 0usize;

                                for (ei, (ci, chunk_meta)) in indexable_chunks.iter().enumerate() {
                                    let chunk_id = format!("{}_{}", hashed_item_id, ci);

                                    let chunk_vec = &chunk_embs[ei];
                                    let anchor_emb = &anchor_embs[ei];
                                    let localized_emb = &localized_embs[ei];

                                    let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
                                        "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
                                        _ => (0.40f32, 0.30f32, 0.30f32),
                                    };

                                    let mut final_vec = vec![0.0f32; 384];
                                    for d in 0..384 {
                                        final_vec[d] = chunk_vec[d] * w_chunk
                                            + anchor_emb[d] * w_anchor
                                            + localized_emb[d] * w_local;
                                    }
                                    let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                                    if norm > 0.0 {
                                        for d in 0..384 { final_vec[d] /= norm; }
                                    }

                                    let _ = store.upsert_chunk(
                                        &chunk_id,
                                        &hashed_item_id,
                                        &page_type,
                                        &chunk_meta.chunk_text,
                                        &chunk_meta.property,
                                        &chunk_meta.property_format,
                                        &chunk_meta.value_part,
                                        Some(final_vec),
                                        Some(&task.cc),
                                        Some(&bcc),
                                        Some(&ref_val),
                                        Some(&search_mode),
                                    ).await;

                                    
                                    alias_saved += upsert_alias_chunks(
                                        &store,
                                        &model,
                                        &hashed_item_id,
                                        &chunk_id,
                                        &page_type,
                                        &doc_lang,
                                        chunk_meta,
                                        &alias_pairs[ei],
                                        &task.cc,
                                        &bcc,
                                        &ref_val,
                                        &search_mode,
                                    ).await;
                                }

                                emit_term(&format!(
                                    "  🧩 [PHASE A~E] 청크 인덱싱 완료: item_id='{}' | 청크 {}개 (전체 {}개 중) | 음차 별칭 {}개",
                                    hashed_item_id, indexable_chunks.len(), enriched_chunks.len(), alias_saved
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    if !items_to_process.is_empty() {
        let metrics_input: Vec<Value> = items_to_process.iter().map(|it| {
            let mut v = it.clone();
            if let Some(o) = v.as_object_mut() {
                if o.get("type").is_none() { o.insert("type".to_string(), json!(page_type.clone())); }
                if o.get("mode").is_none() { o.insert("mode".to_string(), json!(search_mode.clone())); }
                if o.get("updated_at").is_none() { o.insert("updated_at".to_string(), json!(0)); }
                if o.get("created_at").is_none() {
                    o.insert("created_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                }
            }
            v
        }).collect();

        let _ = crate::utils::metrics::update_team_base_metrics(&store, &team_id, &task.cc, &metrics_input, stats_diff.clone()).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items. (Stats Diff: {:?})", metrics_input.len(), stats_diff);
    }

    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;
 
    let payload = json!({
        "task_id": task.id, 
        "category": "Done", 
        "summary": "Extraction complete. Updating list...", 
        "spinner": "✅",

        "data": null 
    });
    
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    // 🌟 [SDS] 태스크 경계에서 관측을 확정하고 스코프를 내립니다.
    //
    //  ── 왜 태스크마다 flush 인가 ──
    //   추출은 폴링 워커가 연속으로 도는 장기 세션입니다.
    //   앱이 비정상 종료되면 그동안의 관측이 통째로 날아가므로,
    //   태스크 경계라는 자연스러운 커밋 지점에서 확정합니다.
    //   dirty 가 아니면 파일을 쓰지 않으므로 I/O 낭비가 없습니다.
    emit_term(&format!("[PROCESS] {}", crate::utils::score_dynamics::report()));
    crate::utils::score_dynamics::flush();
    crate::utils::score_dynamics::leave_scope();

    println!("[PROCESS] Task {} completed. Handover to Embedding finished.", task.id);
    Ok(())
}

