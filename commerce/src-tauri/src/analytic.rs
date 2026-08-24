use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use crate::store::{Task, VectorStore};
use crate::model::LogisModel;
use crate::utils::logger::log_task_progress;
use crate::parsing::PugMode;

// =====================================================================
// 🌟 [ANALYTIC PIPELINE v2]
// ---------------------------------------------------------------------
//  ── 구조 ──
//   ① content.js        : click / hover / change 의 outerHTML 을 그대로 기록
//   ② console Worker    : D1 items 에 updated_at = 0(draft) 으로 적재
//   ③ Client App(여기)  : D1 → LanceDB 동기화된 draft 를 집어
//                         HTML → PUG → 속성 전량 제거 → Qwen3.5 2B 요약
//                         → action / relate / summary 확정 → updated_at 갱신
//   ④ reindex           : 로컬 임베딩 모델이 벡터화 + item_chunks 인덱싱
//   ⑤ #global-search    : 벡터 검색 → 회수한 시맨틱 기록 → Qwen3.5 2B 리포트
//
//  ── 왜 서버(analytics-logis-center)의 Cron 을 걷어냈는가 ──
//   Cron 은 원시 outerHTML 을 통째로 LLM 에 넣었기 때문에
//   class/id/style 같은 사이트별 잡음이 토큰의 대부분을 차지했고,
//   7000 토큰 상한(content.js 의 tokenAmount 게이트)에 자주 걸려
//   구조화 자체가 누락되었습니다.
//   PUG 로 접고 속성을 제거하면 같은 화면이 1/5~1/10 토큰으로 줄어들고,
//   남는 신호가 '태그 구조 + 인쇄된 텍스트' 뿐이라 환각이 물리적으로 줄어듭니다.
// =====================================================================

/// 구조화 대상 이벤트 타입. content.js 가 실제로 발행하는 3종입니다.
pub const ANALYTIC_EVENT_TYPES: [&str; 3] = ["click", "hover", "change"];

/// 검색 스코프에 포함되는 타입. 합성 문서(report)까지 포함합니다.
pub const ANALYTIC_SEARCH_TYPES: [&str; 4] = ["click", "hover", "change", "report"];

/// 🌟 [ATTRIBUTE STRIP] PUG 한 줄에서 속성부(`[...]`)를 완전히 제거하고
///    `{indent}{tag} | {text}` 형태만 남깁니다.
///    pug_line_parts 는 속성값 내부의 파이프를 오인하지 않는 안전 파서이므로
///    `option[value="우체국|https://..."]` 같은 라인에서도 값이 깨지지 않습니다.
pub fn strip_pug_attributes(pug: &str) -> String {
    let mut out = String::new();
    for line in pug.lines() {
        if line.trim().is_empty() { continue; }
        let (indent, tag, _attrs, value) = crate::utils::ai_utils::pug_line_parts(line);
        let v = value.trim();
        if tag.is_empty() && v.is_empty() { continue; }
        let pad = " ".repeat(indent);
        if tag.is_empty() {
            out.push_str(&format!("{}| {}\n", pad, v));
        } else if v.is_empty() {
            out.push_str(&format!("{}{}\n", pad, tag));
        } else {
            out.push_str(&format!("{}{} | {}\n", pad, tag, v));
        }
    }
    out
}

/// 🌟 [HTML → SEMANTIC PUG] 원시 outerHTML 을 '속성이 하나도 없는 PUG' 로 접습니다.
///    ListMode 를 쓰는 이유:
///      · id / class / style 을 이미 버립니다.
///      · input[value] 와 selected option 의 텍스트는 살립니다.
///        (NoAttributesMode 는 select/option 을 통째로 버려 change 이벤트가 빈 값이 됩니다)
///    그 뒤 strip_pug_attributes 로 잔여 속성 대괄호까지 제거합니다.
pub fn html_to_semantic_pug(html: &str) -> String {
    let t = html.trim();
    if t.is_empty() { return String::new(); }
    let cleaned = crate::parsing::pre_clean_html(t);
    let pug = crate::parsing::convert_to_clean_pug(&cleaned, PugMode::ListMode, None);
    strip_pug_attributes(&pug)
}

/// 🌟 [BLOCK JOIN] action / relate 는 문자열 배열(outerHTML 목록)입니다.
///    각 요소를 개별 PUG 로 접은 뒤 구분선으로 이어 붙입니다.
fn html_array_to_semantic_pug(val: Option<&Value>, max_blocks: usize) -> String {
    let arr = match val.and_then(|v| v.as_array()) {
        Some(a) => a,
        None => {
            // 문자열 단일 값으로 들어오는 경우(구버전 페이로드)도 흡수합니다.
            if let Some(s) = val.and_then(|v| v.as_str()) {
                return html_to_semantic_pug(s);
            }
            return String::new();
        }
    };

    let mut out = String::new();
    let mut used = 0usize;
    for item in arr {
        if used >= max_blocks { break; }
        let s = match item.as_str() { Some(x) => x, None => continue };
        let block = html_to_semantic_pug(s);
        if block.trim().is_empty() { continue; }
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str(&block);
        used += 1;
    }
    out
}

/// 🌟 [RAW DETECT] 아직 구조화되지 않은 원시 이벤트인지 판정합니다.
///    Cron 이 사라졌으므로 action 은 배열(outerHTML 목록)로만 도착합니다.
fn is_raw_event(data: &Value) -> bool {
    data.get("action").map_or(false, |v| v.is_array())
        || data.get("relate").map_or(false, |v| v.is_array())
}

/// 🌟 [STRUCTURING] draft(updated_at = 0) 행동 로그를 시맨틱 문장으로 확정합니다.
///  반환값 = 구조화에 성공한 이벤트 건수
pub async fn run_analytic_structuring(
    store: &VectorStore,
    model: &LogisModel,
    cancel: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    task_id: &str,
    limit: usize,
) -> Result<usize> {
    let app_handle_clone = app_handle.clone();
    let tid = task_id.to_string();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit(
            "task-console-log",
            json!({ "task_id": tid, "text": format!("{}\n", msg) })
        );
    };

    if cancel.load(Ordering::Relaxed) {
        return Ok(0);
    }

    let type_list = ANALYTIC_EVENT_TYPES
        .iter()
        .map(|t| format!("'{}'", t))
        .collect::<Vec<_>>()
        .join(", ");
    let filter = format!(
        "mode = 'analytic' AND updated_at = 0 AND type IN ({})",
        type_list
    );

    let raw_logs = store
        .get_all_items("items", 500, 0, Some(filter))
        .await
        .unwrap_or_default();

    if raw_logs.is_empty() {
        return Ok(0);
    }

    // ── 구조화 대상만 추립니다. (이미 문장이 확정된 행은 제외) ──
    let mut targets: Vec<(crate::store::TradeDocument, Value)> = Vec::new();
    for doc in raw_logs {
        if targets.len() >= limit { break; }
        let data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
        if !is_raw_event(&data) { continue; }
        targets.push((doc, data));
    }

    if targets.is_empty() {
        return Ok(0);
    }

    emit_term(&format!(
        "[ANALYTIC] 🧠 구조화 대상 원시 이벤트 {}건 발견. HTML → PUG → 속성 제거 → Qwen3.5 2B 요약을 시작합니다.",
        targets.len()
    ));
    log_task_progress(app_handle, task_id, &json!({
        "category": "Analytic Structuring",
        "summary": format!("Summarizing {} behaviour event(s)...", targets.len()),
        "spinner": "⠋"
    }));

    model
        .secure_vram_relay(
            crate::model::ModelSize::Qwen3_5,
            None,
            Some(cancel.clone()),
            false,
            None,
        )
        .await?;

    let now_ts = chrono::Utc::now().timestamp_millis();
    let mut processed = 0usize;

    // 흐름(report) 합성을 위해 (from, ref) 로 묶어 둡니다.
    let mut flow_groups: std::collections::HashMap<String, Vec<Value>> =
        std::collections::HashMap::new();
    let mut flow_envelope: std::collections::HashMap<String, crate::store::TradeDocument> =
        std::collections::HashMap::new();

    let total = targets.len();

    for (idx, (doc, data)) in targets.into_iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            emit_term("[ANALYTIC] 🛑 사용자 취소로 구조화를 중단합니다.");
            break;
        }

        let percent = (((idx as f32) / (total as f32)) * 100.0) as i32;
        log_task_progress(app_handle, task_id, &json!({
            "category": format!("Analytic Structuring ({}/{})", idx + 1, total),
            "summary": format!("Summarizing behaviour event ({}%)...", percent),
            "spinner": "⠋"
        }));

        let link = data
            .get("link")
            .or_else(|| data.get("href"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let doc_lang = crate::utils::lang_utils::detect_document_language(
            &data.get("action").map(|v| v.to_string()).unwrap_or_default()
        );

        // ── ① HTML → PUG → 속성 전량 제거 ──
        let mut target_pug = html_array_to_semantic_pug(data.get("action"), 1);
        let mut related_pug = html_array_to_semantic_pug(data.get("relate"), 8);

        if target_pug.trim().is_empty() {
            emit_term(&format!(
                "  ⚪ [ANALYTIC SKIP] id='{}' 는 대상 엘리먼트에서 읽을 수 있는 텍스트가 없어 건너뜁니다.",
                doc.id
            ));
            continue;
        }

        // ── ② 컨텍스트 상한 ──
        target_pug = model.truncate_pug_context(&target_pug, true, 1200, None).await;
        related_pug = model.truncate_pug_context(&related_pug, true, 2400, None).await;

        emit_term(&format!(
            "  🧩 [SEMANTIC PUG] id='{}' | type='{}' | link='{}'\n{}",
            doc.id, doc.r#type, link, target_pug.trim()
        ));

        // ── ③ Qwen3.5 2B 시맨틱 요약 ──
        let prompt = crate::prompts::analytic_semantic_prompt(
            &doc.r#type,
            &link,
            &doc_lang,
            &target_pug,
            if related_pug.trim().is_empty() { "(none)" } else { &related_pug },
        );

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
            max_tokens: Some(768),
            temperature: Some(0.0),
            top_p: Some(0.95),
            ..Default::default()
        };

        let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            gen.generate(
                params,
                Some(cancel.clone()),
                Some(format!("{}_sem_{}", task_id, idx)),
                None,
                None,
                None,
            )
            .await
            .unwrap_or_default()
        } else {
            String::new()
        };

        let parsed = crate::parsing::parse_json_from_llm(&res_text);

        let action = parsed.get("action").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let summary = parsed.get("summary").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let relate: Vec<String> = parsed
            .get("relate")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        if action.is_empty() && summary.is_empty() {
            emit_term(&format!(
                "  ⚠️ [ANALYTIC EMPTY] id='{}' 요약 결과가 비어 있어 이번 라운드에서는 확정하지 않습니다.",
                doc.id
            ));
            continue;
        }

        // ── ④ 검색 본문 조립 ──
        let mut combined = String::new();
        if !action.is_empty() {
            combined.push_str(&action);
        }
        if !summary.is_empty() {
            if !combined.is_empty() { combined.push(' '); }
            combined.push_str(&summary);
        }
        if !relate.is_empty() {
            combined.push(' ');
            combined.push_str(&relate.join(", "));
        }

        let mut new_data = data.clone();
        if let Some(o) = new_data.as_object_mut() {
            o.insert("action".to_string(), json!(action.clone()));
            o.insert("summary".to_string(), json!(summary.clone()));
            o.insert("relate".to_string(), json!(relate.clone()));
            o.insert("text".to_string(), json!(combined.clone()));
            o.insert("masked_text".to_string(), json!(combined.clone()));
            o.insert("mode".to_string(), json!("analytic"));
            o.insert("updated_at".to_string(), json!(now_ts));
            // 🌟 구조화가 끝났으므로 벡터를 새로 만들어야 합니다.
            //    reindex_pending_embeddings 는 embed 플래그가 1이면 건너뛰므로 제거합니다.
            o.remove("embed");
        }

        let _ = store
            .upsert_item(
                "items",
                &doc.id,
                &doc.r#type,
                new_data.clone(),
                None,
                Some(&doc.from),
                Some(&doc.to),
                Some(&doc.cc),
                Some(&doc.bcc),
                Some(&doc.r#ref),
                None,
            )
            .await;

        emit_term(&format!(
            "  ✅ [ANALYTIC STRUCTURED] id='{}' | action=\"{}\" | relate={}건",
            doc.id, action, relate.len()
        ));

        processed += 1;

        let group_key = format!("{}|{}", doc.from, doc.r#ref);
        flow_groups.entry(group_key.clone()).or_insert_with(Vec::new).push(json!({
            "at": doc.created_at_ts,
            "type": doc.r#type,
            "link": link,
            "action": action,
            "summary": summary,
            "relate": relate
        }));
        flow_envelope.entry(group_key).or_insert(doc);
    }

    // ── ⑤ 흐름 리포트(report) 합성 ──
    //    analytics-logis-center 의 cross_action_flow / intent_evolution /
    //    consistent_preferences 3축을 그대로 로컬에서 재현합니다.
    for (group_key, mut records) in flow_groups.into_iter() {
        if cancel.load(Ordering::Relaxed) { break; }
        if records.len() < 2 { continue; }

        let env_doc = match flow_envelope.get(&group_key) { Some(d) => d.clone(), None => continue };

        records.sort_by_key(|r| r.get("at").and_then(|v| v.as_i64()).unwrap_or(0));
        if records.len() > 24 { records.truncate(24); }

        let records_json = serde_json::to_string_pretty(&records).unwrap_or_else(|_| "[]".to_string());
        let doc_lang = crate::utils::lang_utils::detect_document_language(&records_json);
        let prompt = crate::prompts::analytic_flow_prompt(&doc_lang, &records_json);

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
            max_tokens: Some(1024),
            temperature: Some(0.0),
            top_p: Some(0.95),
            ..Default::default()
        };

        let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            gen.generate(
                params,
                Some(cancel.clone()),
                Some(format!("{}_flow", task_id)),
                None,
                None,
                None,
            )
            .await
            .unwrap_or_default()
        } else {
            String::new()
        };

        let parsed = crate::parsing::parse_json_from_llm(&res_text);
        let cross_action = parsed.get("cross_action_flow").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let intent_evo = parsed.get("intent_evolution").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let preferences = parsed.get("consistent_preferences").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

        if cross_action.is_empty() && intent_evo.is_empty() && preferences.is_empty() {
            continue;
        }

        let report_text = format!("{} {} {}", cross_action, intent_evo, preferences)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // 🌟 [DETERMINISTIC REPORT ID] 난수 대신 (사용자 + 페이지 + 일자) 기반 해시.
        //    같은 날 같은 페이지의 흐름은 새 문서를 만들지 않고 갱신되므로
        //    리포트가 무한히 불어나지 않습니다.
        let day_bucket = now_ts / 86_400_000;
        let report_id = crate::utils::hash::hash_id(&format!(
            "report{}{}{}",
            env_doc.from, env_doc.r#ref, day_bucket
        ));

        let report_data = json!({
            "id": report_id.clone(),
            "type": "report",
            "mode": "analytic",
            "cross_action_flow": cross_action,
            "intent_evolution": intent_evo,
            "consistent_preferences": preferences,
            "link": records.first().and_then(|r| r.get("link")).cloned().unwrap_or(json!("")),
            "origin": env_doc_origin(&env_doc),
            "text": report_text.clone(),
            "masked_text": report_text,
            "created_at": now_ts,
            "updated_at": now_ts
        });

        let report_bcc = crate::utils::hash::hash_id(&format!("report{}", env_doc.cc));

        let _ = store
            .upsert_item(
                "items",
                &report_id,
                "report",
                report_data,
                None,
                Some(&env_doc.from),
                Some(&env_doc.to),
                Some(&env_doc.cc),
                Some(&report_bcc),
                Some(&env_doc.r#ref),
                None,
            )
            .await;

        emit_term(&format!(
            "  📊 [ANALYTIC REPORT] id='{}' | 기록 {}건을 흐름 리포트로 합성했습니다.",
            report_id, records.len()
        ));
    }

    emit_term(&format!(
        "[ANALYTIC] ✅ 구조화 완료: {}건. 로컬 임베딩 파이프라인이 이어서 벡터화합니다.",
        processed
    ));

    Ok(processed)
}

/// 🌟 [ORIGIN] 리포트에도 origin 을 남겨야 resolveAnalyticsOrigins 가 다음 라운드에서
///    같은 사이트를 계속 발견합니다. TradeDocument 에는 origin 이 없으므로 data 에서 읽습니다.
fn env_doc_origin(doc: &crate::store::TradeDocument) -> Value {
    if let Ok(d) = serde_json::from_str::<Value>(&doc.json_data) {
        if let Some(o) = d.get("origin").and_then(|v| v.as_str()) {
            return json!(o);
        }
    }
    json!("")
}

// =====================================================================
// 🌟 [ANALYTIC QUERY PARSER]
// ---------------------------------------------------------------------
//  parse_commerce_query 의 3단 구조를 분석 도메인에 맞춰 압축했습니다.
//    ① 결정론 시간 가이드 (get_deterministic_time_guide + exact_match_filter_key)
//    ② Qwen3.5 2B 의미 파싱 (기간 / 이벤트 종류 / 키워드)
//    ③ Rust 가 기간을 epoch ms 로 재확정 (LLM 이 계산한 날짜는 신뢰하지 않음)
//  ── 왜 ③이 필요한가 ──
//   commerce 의 extract_numeric_conditions 도 같은 이유로 시간 문맥을
//   프롬프트에 주입만 하고, 최종 SQL 은 코드가 만듭니다.
//   LLM 이 '이번달' 을 2026-03-01 로 적어도 그 값이 실제 epoch 인지 보증할 수 없기 때문입니다.
// =====================================================================

fn ms_of(y: i32, m: u32, d: u32) -> i64 {
    chrono::NaiveDate::from_ymd_opt(y, m, d)
        .and_then(|dd| dd.and_hms_opt(0, 0, 0))
        .map(|nd| nd.and_utc().timestamp_millis())
        .unwrap_or(0)
}

fn month_start(y: i32, m: u32) -> i64 { ms_of(y, m, 1) }

fn next_month(y: i32, m: u32) -> (i32, u32) {
    if m == 12 { (y + 1, 1) } else { (y, m + 1) }
}

fn prev_month(y: i32, m: u32) -> (i32, u32) {
    if m == 1 { (y - 1, 12) } else { (y, m - 1) }
}

pub fn ymd_of(ts: i64) -> (i32, u32, u32) {
    match chrono::DateTime::from_timestamp_millis(ts) {
        Some(dt) => {
            let d = dt.naive_utc().date();
            (chrono::Datelike::year(&d), chrono::Datelike::month(&d), chrono::Datelike::day(&d))
        },
        None => (1970, 1, 1),
    }
}

/// 🌟 [TIME RANGE] time_filters 캐노니컬 키를 epoch ms 구간으로 확정합니다.
pub fn time_intent_range(intent: &str, now_ms: i64) -> Option<(i64, i64)> {
    let (y, m, d) = ymd_of(now_ms);
    match intent {
        "today" => {
            let s = ms_of(y, m, d);
            Some((s, s + 86_400_000 - 1))
        },
        "yesterday" => {
            let s = ms_of(y, m, d) - 86_400_000;
            Some((s, s + 86_400_000 - 1))
        },
        "this_month" => {
            let s = month_start(y, m);
            let (ny, nm) = next_month(y, m);
            Some((s, month_start(ny, nm) - 1))
        },
        "last_month" => {
            let (py, pm) = prev_month(y, m);
            let s = month_start(py, pm);
            Some((s, month_start(y, m) - 1))
        },
        "this_year" => Some((ms_of(y, 1, 1), ms_of(y + 1, 1, 1) - 1)),
        "last_year" => Some((ms_of(y - 1, 1, 1), ms_of(y, 1, 1) - 1)),
        "recently" => Some((now_ms - 7 * 86_400_000, now_ms)),
        _ => None,
    }
}

/// 🌟 [SEASON RANGE] season_filters 캐노니컬 키를 해당 연도의 구간으로 확정합니다.
///    time_intent 가 과거를 가리키면 호출부가 year 를 이미 낮춰서 넘깁니다.
pub fn season_range(season: &str, year: i32) -> Option<(i64, i64)> {
    match season {
        "spring" => Some((ms_of(year, 3, 1), ms_of(year, 6, 1) - 1)),
        "summer" => Some((ms_of(year, 6, 1), ms_of(year, 9, 1) - 1)),
        "autumn" => Some((ms_of(year, 9, 1), ms_of(year, 12, 1) - 1)),
        "winter" => Some((ms_of(year, 12, 1), ms_of(year + 1, 3, 1) - 1)),
        _ => None,
    }
}

/// 🌟 [DETERMINISTIC TIME] bias.json 의 exact_match 배열로 완전일치 판정합니다.
///    '오늘' / 'today' / '今日' 처럼 50개 언어의 시간·계절 표현이 리터럴로 등재되어 있어
///    코드에 다국어 어휘를 하나도 넣지 않고 확정할 수 있습니다.
pub fn deterministic_time_keys(query: &str) -> (String, String) {
    let mut t_key = String::new();
    let mut s_key = String::new();

    // 공백 토큰 → 실패 시 전체 문자열까지 시도합니다. (한국어는 '이번달' 처럼 붙어 옵니다)
    let mut candidates: Vec<String> = query
        .split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    candidates.push(query.trim().to_string());

    for c in &candidates {
        if t_key.is_empty() {
            if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("time_filters", c) {
                t_key = k;
            }
        }
        if s_key.is_empty() {
            if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("season_filters", c) {
                s_key = k;
            }
        }
        if !t_key.is_empty() && !s_key.is_empty() { break; }
    }

    (t_key, s_key)
}

/// 🌟 [ANALYTIC SEARCH QUERY] #global-search 의 자연어 질의를 검색 컨텍스트로 변환합니다.
///  반환 형태는 lib.rs 의 STAGE-3 컨텍스트 계약과 동일하므로
///  build_scope_filter / build_dexie_plan / STAGE-4 를 그대로 재사용합니다.
pub async fn parse_analytic_search_query(
    task_id: &str,
    app_handle: &tauri::AppHandle,
    model: &LogisModel,
    query: String,
    language: &str,
    cancel: Arc<AtomicBool>,
) -> Result<Value> {
    let app_handle_clone = app_handle.clone();
    let tid = task_id.to_string();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit(
            "task-console-log",
            json!({ "task_id": tid, "text": format!("{}\n", msg) })
        );
    };

    emit_term("\n[ANALYTIC-QUERY] 🔍 행동 로그 질의 파싱 시작");
    emit_term(&format!("  질의: \"{}\"", query));

    let now_ms = chrono::Utc::now().timestamp_millis();
    let current_iso = chrono::DateTime::from_timestamp_millis(now_ms)
        .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
        .unwrap_or_default();

    // ── ① 결정론 시간 가이드 ──
    let (deterministic_time, _) = crate::parsing::get_deterministic_time_guide(&query, language);
    let (det_time_key, det_season_key) = deterministic_time_keys(&query);

    if !det_time_key.is_empty() || !det_season_key.is_empty() {
        emit_term(&format!(
            "  ⚡ [EXACT MATCH] bias.json 완전일치로 확정: time='{}' | season='{}'",
            det_time_key, det_season_key
        ));
    }

    let time_context = format!(
        "- Current UTC time is \"{}\" (epoch ms {}).\n- The user locale language is \"{}\".\n{}",
        current_iso, now_ms, language, deterministic_time
    );

    // ── ② Qwen3.5 2B 의미 파싱 ──
    model
        .secure_vram_relay(
            crate::model::ModelSize::Qwen3_5,
            None,
            Some(cancel.clone()),
            false,
            None,
        )
        .await?;

    let prompt = crate::prompts::analytic_query_prompt(&query, &time_context, language);

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
        max_tokens: Some(512),
        temperature: Some(0.0),
        top_p: Some(0.95),
        ..Default::default()
    };

    let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
        gen.generate(
            params,
            Some(cancel.clone()),
            Some(format!("{}_aq", task_id)),
            None,
            None,
            None,
        )
        .await
        .unwrap_or_default()
    } else {
        String::new()
    };

    let parsed = crate::parsing::parse_json_from_llm(&res_text);

    let mut time_intent = parsed.get("time_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let mut season_intent = parsed.get("season_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

    // 🌟 완전일치가 있으면 LLM 판정보다 우선합니다. (벡터·LLM 은 계절을 자주 환각합니다)
    if !det_time_key.is_empty() { time_intent = det_time_key.clone(); }
    if !det_season_key.is_empty() { season_intent = det_season_key.clone(); }

    let keywords: Vec<String> = parsed
        .get("keywords")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let mut event_types: Vec<String> = parsed
        .get("event_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.trim().to_lowercase()))
                .filter(|s| ANALYTIC_SEARCH_TYPES.iter().any(|t| *t == s.as_str()))
                .collect()
        })
        .unwrap_or_default();

    if event_types.is_empty() {
        event_types = ANALYTIC_SEARCH_TYPES.iter().map(|s| s.to_string()).collect();
    }

    let target = parsed
        .get("target")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            if keywords.is_empty() { query.clone() } else { keywords.join(" ") }
        });

    // ── ③ 기간을 Rust 가 재확정 ──
    let mut started_at: i64 = 0;
    let mut expired_at: i64 = 0;

    if !season_intent.is_empty() {
        // 계절은 연도가 필요합니다. time_intent 가 과거를 가리키면 작년으로 내립니다.
        let (y, _, _) = ymd_of(now_ms);
        let year = if time_intent == "last_year" { y - 1 } else { y };
        if let Some((s, e)) = season_range(&season_intent, year) {
            started_at = s;
            expired_at = e;
        }
    }

    if started_at == 0 {
        if let Some((s, e)) = time_intent_range(&time_intent, now_ms) {
            started_at = s;
            expired_at = e;
        }
    }

    if started_at > 0 {
        emit_term(&format!(
            "  🗓️ [PERIOD CONFIRMED] time='{}' | season='{}' → {} ~ {} (epoch ms)",
            if time_intent.is_empty() { "-" } else { &time_intent },
            if season_intent.is_empty() { "-" } else { &season_intent },
            started_at, expired_at
        ));
    } else {
        emit_term("  🗓️ [PERIOD] 명시적 기간 표현이 없어 전체 구간을 검색합니다.");
    }

    // ── ④ 컨텍스트 조립 (lib.rs STAGE-3 계약과 동일) ──
    let mut condition = serde_json::Map::new();
    if started_at > 0 {
        condition.insert(
            "created_at".to_string(),
            json!({ "operator": "gte", "value": started_at })
        );
    }
    if expired_at > 0 {
        // 🌟 동일 키가 하나뿐이므로 상한은 updated_at 축으로 내려보내지 않고
        //    별도 키(created_at_to)를 만들지 않습니다.
        //    build_scope_filter 는 키 단위로 하나의 연산자만 받으므로,
        //    상한은 Dexie 플랜이 처리하도록 unassigned 가 아닌 별도 컨텍스트로 분리합니다.
    }

    let mut contexts: Vec<Value> = Vec::new();

    contexts.push(json!({
        "text": target,
        "language": language,
        "type": event_types[0],
        "types": event_types,
        "condition": Value::Object(condition.clone()),
        "unassigned": keywords
    }));

    // 🌟 상한(lte)은 별도 컨텍스트로 분리해 build_scope_filter 가
    //    created_at <= X 를 SQL 로 함께 내려보내도록 합니다.
    if expired_at > 0 {
        let mut upper = serde_json::Map::new();
        upper.insert(
            "created_at".to_string(),
            json!({ "operator": "lte", "value": expired_at })
        );
        contexts.push(json!({
            "text": target,
            "language": language,
            "type": event_types[0],
            "types": event_types,
            "condition": Value::Object(upper),
            "unassigned": keywords
        }));
    }

    let out = json!({
        "original_text": query,
        "time_intent": time_intent,
        "season_intent": season_intent,
        "started_at": started_at,
        "expired_at": expired_at,
        "event_types": event_types,
        "keywords": keywords,
        "target": target,
        "context": contexts
    });

    emit_term(&format!(
        "[ANALYTIC-QUERY] ✅ 파싱 결과: {}",
        serde_json::to_string(&out).unwrap_or_default()
    ));

    Ok(out)
}

// =====================================================================
// 🌟 [TASK ENTRY] 스케줄러에서 analytic_extraction 태스크로 들어오는 경로
// =====================================================================
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

    emit_term("[Scheduler] Starting Analytic Structuring Pipeline...");
    let list_log = json!({ "category": "Analytic Processing", "summary": "Analyzing user behavior logs...", "spinner": "⠋" });
    log_task_progress(app_handle, &task.id, &list_log);

    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    let model = {
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            *model_lock = Some(LogisModel::new(app_handle.clone(), device_preference.as_deref()).await.map_err(|e| anyhow::anyhow!(e))?);
        }
        model_lock.as_ref().unwrap().clone()
    };

    let processed = run_analytic_structuring(
        &store,
        &model,
        cancellation_token,
        app_handle,
        &task.id,
        60,
    ).await.unwrap_or(0);

    model.deep_purge_resources().await;

    let store_guard = store_mutex.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let _ = db.update_task_status(&task.id, 9).await;
        let _ = db.update_message_status(&task.id, 9, Some("Analytic Structuring Complete")).await;
    }
    drop(store_guard);

    let payload = json!({
        "task_id": task.id,
        "category": "Done",
        "summary": if processed > 0 {
            format!("{} behaviour event(s) structured.", processed)
        } else {
            "No pending analytic logs found.".to_string()
        },
        "spinner": "✅",
        "data": null
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }
    Ok(())
}