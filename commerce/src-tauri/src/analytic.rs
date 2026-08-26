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

// =====================================================================
// 🌟 [EVENT TYPE ANCHOR BANK]
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   기존에는 event_types 를 Qwen3.5 2B 가 '단독으로' 골랐습니다.
//   프롬프트의 [AVAILABLE EVENT TYPES] 는 영어 한 줄 설명뿐이라
//   '클릭한' 같은 한국어 표현이 hover / change 와 구분되는 벡터 근거가 없었고,
//   실측 로그에서는 질의에 '클릭'이 들어갔다는 이유만으로
//   event_types 가 ["click"] 하나로 좁혀져
//   방금 구조화한 hover 3건 + report 1건이 스코프에서 통째로 탈락했습니다.
//
//  ── 해결 ──
//   bias.json 의 `analytic_event_filters` 노드를 구 단위로 쪼개 Max-Pool 뱅크를 만듭니다.
//   노드가 없으면 코드 폴백을 씁니다. (get_trade_category_schema 의 fallback_base 와 동일 패턴)
//   구 단위 Max-Pool 은 다국어 임베딩에서 '클릭한' ↔ 'pressed the button' 을
//   직접 연결하므로 어휘 하드코딩 없이 판정이 성립합니다.
// =====================================================================

/// 🌟 [EVENT ANCHOR] 이벤트 타입 1종의 의미 앵커 구를 반환합니다.
pub fn event_type_anchor_phrases(event_type: &str) -> Vec<String> {
    // ① bias.json 우선 (semantic + bias)
    if let Some(node) = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.get(event_type))
    {
        let mut out: Vec<String> = Vec::new();
        for field in ["semantic", "bias"] {
            if let Some(s) = node.get(field).and_then(|v| v.as_str()) {
                for p in crate::utils::ai_utils::split_bias_phrases_full(s) {
                    if !out.iter().any(|e| e == &p) {
                        out.push(p);
                    }
                }
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    // ② 코드 폴백 : bias.json 을 손대지 않아도 즉시 동작합니다.
    let raw = match event_type {
        "click" => "click, clicked, pressed, tapped, selected, chose, picked, opened, pushed the button, selection, choice, purchase intent",
        "hover" => "hover, hovered, mouse over, lingered, dwelled, looked at, browsed, scanned, glanced, viewed without clicking, attention, interest",
        "change" => "change, changed, typed, entered, input, filled in, edited, modified, toggled, switched, picked an option, selected a value, form entry, keyword typed",
        "report" => "report, summary, overview, behaviour flow, user journey, pattern, trend, analysis, insight, statistics, aggregated result, most frequent",
        _ => "",
    };
    crate::utils::ai_utils::split_bias_phrases_full(raw)
}

/// 🌟 [EVENT PREJUDICE] 경쟁 타입의 앵커를 편견 뱅크로 사용합니다.
///  bias.json 에 prejudice 가 명시되어 있으면 그것을 우선하고,
///  없으면 '나머지 3종의 앵커' 를 그대로 편견으로 씁니다.
///  이 구조 덕분에 새 이벤트 타입이 생겨도 편견 사전을 따로 만들 필요가 없습니다.
pub fn event_type_prejudice_phrases(event_type: &str) -> Vec<String> {
    if let Some(s) = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.get(event_type))
        .and_then(|n| n.get("prejudice"))
        .and_then(|v| v.as_str())
    {
        let p = crate::utils::ai_utils::split_bias_phrases_full(s);
        if !p.is_empty() {
            return p;
        }
    }
    let mut out: Vec<String> = Vec::new();
    for other in ANALYTIC_SEARCH_TYPES.iter() {
        if *other == event_type {
            continue;
        }
        for p in event_type_anchor_phrases(other) {
            if !out.iter().any(|e| e == &p) {
                out.push(p);
            }
        }
    }
    out
}

/// 🌟 [EVENT EXACT MATCH] bias.json 의 exact_match 배열로 완전일치 판정합니다.
///  time_filters / season_filters 의 exact_match 와 동일한 계약이며,
///  일치하면 벡터 경쟁도 LLM 도 거치지 않고 즉시 확정합니다.
pub fn event_type_exact_match(word: &str) -> Option<String> {
    let w = word.trim().to_lowercase();
    if w.is_empty() {
        return None;
    }
    let obj = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.as_object())?;
    for (key, node) in obj {
        if !ANALYTIC_SEARCH_TYPES.iter().any(|t| t == key) {
            continue;
        }
        if let Some(arr) = node.get("exact_match").and_then(|v| v.as_array()) {
            if arr
                .iter()
                .any(|x| x.as_str().map_or(false, |s| s.trim().to_lowercase() == w))
            {
                return Some(key.clone());
            }
        }
    }
    None
}

/// 🌟 [EVENT PREFIX MATCH] 교착어 어절을 위해 exact_match 배열을 '접두 사전' 으로 재사용합니다.
///  "클릭한게" 는 exact_match 에 없지만 "클릭" 이 그 접두이므로 click 으로 확정됩니다.
///  사전은 bias.json 이 소유하므로 코드에는 어떤 언어의 어휘도 등장하지 않습니다.
pub fn event_type_prefix_key(word: &str) -> Option<String> {
    let (key, _stem) =
        crate::utils::ai_utils::prefix_match_filter_stem("analytic_event_filters", word)?;
    if ANALYTIC_SEARCH_TYPES.iter().any(|t| t == &key) {
        Some(key)
    } else {
        None
    }
}

/// 🌟 [MORPHOLOGICAL VARIANTS] 교착어 어절 하나에서 '검색 가능한 원형 후보'를 만들어 냅니다.
///  ── 근거 3단 ──
///   ① Stanza Lemma      : 모델이 직접 알려준 원형 ("클릭한게" → "클릭")
///   ② bias.json 접두 사전: exact_match 원소가 표면형의 접두이면 그 원소를 어간으로 확정
///   ③ 문자 접두 n-gram   : ①②가 모두 실패했을 때의 순수 구조 폴백
///  ── 왜 필요한가 ──
///   commerce 는 질의에 다른 토큰('제품')이 함께 있어 shared_prefix_stems 로 어간을 얻지만,
///   "가장 많이 클릭한게 뭐야?" 처럼 어간을 공유하는 형제 토큰이 없는 질의에서는
///   그 장치가 동작하지 않습니다. 세 근거를 순서대로 시도합니다.
///  ── 상한 ──
///   토큰당 최대 3개. 청크 폭발을 막고, 무의미한 조각은 SURPRISAL/편견 게이트가 걸러냅니다.
pub fn morphological_variants(word: &str, lemma: &str) -> Vec<String> {
    let surface = word.trim();
    let mut out: Vec<String> = Vec::new();
    if surface.chars().count() < 2 {
        return out;
    }

    fn push(v: &mut Vec<String>, surface: &str, cand: String) {
        let c = cand.trim().to_string();
        if c.is_empty() { return; }
        if c.chars().count() < 2 { return; }
        if c == surface { return; }
        if v.iter().any(|e| e == &c) { return; }
        v.push(c);
    }

    // ① Stanza Lemma 원형
    let l = lemma.trim();
    if !l.is_empty() && l != surface {
        let lc: String = l.chars().filter(|c| c.is_alphanumeric()).collect();
        let sc: String = surface.chars().filter(|c| c.is_alphanumeric()).collect();
        if !lc.is_empty()
            && sc.chars().count() > lc.chars().count()
            && (sc.starts_with(&lc) || sc.ends_with(&lc))
        {
            push(&mut out, surface, lc);
        } else if !lc.is_empty() {
            push(&mut out, surface, l.to_string());
        }
    }

    // ② bias.json exact_match 접두 사전
    for cat in ["analytic_event_filters", "time_filters", "season_filters"] {
        if let Some((_, stem)) = crate::utils::ai_utils::prefix_match_filter_stem(cat, surface) {
            push(&mut out, surface, stem);
        }
    }

    // ③ 문자 접두 n-gram (사전 없이 동작하는 최후 폴백)
    let chars: Vec<char> = surface.chars().collect();
    if chars.len() >= 3 {
        let hi = (chars.len() - 1).min(4);
        for n in 2..=hi {
            push(&mut out, surface, chars[..n].iter().collect::<String>());
        }
    }

    if out.len() > 3 {
        out.truncate(3);
    }
    out
}

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

/// 🌟 [STRUCTURING PROBE] 모델 로드 없이 실제 처리 가능한 원시 이벤트 수를 사전 확인합니다.
///    structure_pending_analytics 에서 모델 로드 전에 호출하여,
///    처리할 항목이 0건이면 모델 로드 없이 조기 반환합니다.
///    원시 이벤트가 존재해도 HTML → PUG 변환 결과가 비어있으면 처리 불가능으로 간주합니다.
pub fn count_pending_structuring_targets(
    raw_docs: &[crate::store::TradeDocument],
    limit: usize,
) -> usize {
    let mut count = 0usize;
    for doc in raw_docs.iter() {
        if count >= limit { break; }
        let data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
        if !is_raw_event(&data) { continue; }
        let target_pug = html_array_to_semantic_pug(data.get("action"), 1);
        if target_pug.trim().is_empty() { continue; }
        count += 1;
    }
    count
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

    // ── ① [PRE-FILTER / MODEL-LOAD GATE] 모델 로드 전에 실제 요약 가능한 텍스트가 있는지 사전 확인 ──
    //    원시 이벤트가 존재해도 HTML → PUG 변환 결과가 비어있으면 모델 로드가 무의미합니다.
    //    기존에는 3건이 들어와도 전부 요약 실패(빈 결과)로 스킵되는 경우에도 모델이 로드되었습니다.
    //    이제 사전 필터링으로 처리 가능한 항목이 0건이면 모델 로드 없이 즉시 반환합니다.
    let mut pre_checked: Vec<(crate::store::TradeDocument, Value, String)> = Vec::new();
    for (doc, data) in targets.into_iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let target_pug = html_array_to_semantic_pug(data.get("action"), 1);
        if target_pug.trim().is_empty() {
            emit_term(&format!(
                "  ⚪ [ANALYTIC SKIP] id='{}' 는 대상 엘리먼트에서 읽을 수 있는 텍스트가 없어 건너뜁니다.",
                doc.id
            ));
            continue;
        }
        pre_checked.push((doc, data, target_pug));
    }

    if pre_checked.is_empty() {
        emit_term("[ANALYTIC] ⚪ 실제 요약 가능한 원시 이벤트가 0건이라 모델 로드를 건너뜁니다.");
        return Ok(0);
    }

    emit_term(&format!(
        "[ANALYTIC] 🧠 구조화 대상 원시 이벤트 {}건 발견. HTML → PUG → 속성 제거 → Qwen3.5 2B 요약을 시작합니다.",
        pre_checked.len()
    ));
    log_task_progress(app_handle, task_id, &json!({
        "category": "Analytic Structuring",
        "summary": format!("Summarizing {} behaviour event(s)...", pre_checked.len()),
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

    let total = pre_checked.len();
    for (idx, (doc, data, target_pug_pre)) in pre_checked.into_iter().enumerate() {
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

        // ── ① HTML → PUG → 속성 전량 제거 (사전 계산된 값 재사용) ──
        let mut target_pug = target_pug_pre;
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

        // 🌟 [TRANSLIT IN STRUCTURIZATION] 전처리 단계에서 음차를 수행합니다.
        //    Qwen3.5 가 이미 로드되어 있으므로 별도 모델 로딩이 불필요합니다.
        //    방향: 영어 단어 → 문서 언어(한글/일어/중어 등) 로 음차.
        //    한글→한글 같은 동일 언어 음차는 수행하지 않습니다.
        let translit_native;
        let translit_roman;
        if !action.is_empty() {
            let (tn, tr) = crate::scheduler::transliterate_cross_language(
                model, &action, &doc_lang, cancel, app_handle, task_id,
            ).await;
            translit_native = tn;
            translit_roman = tr;
        } else {
            translit_native = String::new();
            translit_roman = String::new();
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
            // 🌟 전처리에서 확정한 음차 결과를 저장합니다.
            if !translit_native.is_empty() {
                o.insert("translit_native".to_string(), json!(translit_native));
            }
            if !translit_roman.is_empty() {
                o.insert("translit_roman".to_string(), json!(translit_roman));
            }
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

/// 🌟 [REPORT OUTPUT NORMALIZE] Qwen3.5 가 "Do NOT output JSON" 지시를 어기고
///  { "headline": ..., "supporting_actions": [...], "closing": ... } 형태로 반환하는 경우를 흡수합니다.
///  ── 왜 코드로 흡수하는가 ──
///   프롬프트 지시만으로는 2B 모델의 JSON 관성을 100% 막을 수 없고,
///   말풍선에 원시 JSON 이 그대로 노출되면 사용자가 읽을 수 없습니다.
///   구조가 무엇이든 '문자열 잎' 만 순서대로 펼치면 항상 읽을 수 있는 텍스트가 됩니다.
///  ── JSON 이 아니면 원문을 그대로 돌려줍니다 (무해). ──
pub fn normalize_report_output(raw: &str) -> String {
    let mut t = raw.trim().to_string();
    if t.is_empty() {
        return t;
    }

    // ① 코드펜스 제거
    if t.starts_with("```") {
        if let Some(p) = t.find('\n') {
            t = t[p + 1..].to_string();
        }
        if let Some(p) = t.rfind("```") {
            t = t[..p].to_string();
        }
        t = t.trim().to_string();
    }

    // ② JSON 형태가 아니면 그대로 반환
    if !(t.starts_with('{') || t.starts_with('[')) {
        return t;
    }

    let parsed = crate::parsing::parse_json_from_llm(&t);
    if parsed.is_null() {
        return t;
    }

    fn flatten(v: &Value, out: &mut Vec<String>, bullet: bool) {
        match v {
            Value::String(s) => {
                let x = s.trim();
                if x.is_empty() {
                    return;
                }
                out.push(if bullet { format!("- {}", x) } else { x.to_string() });
            }
            Value::Number(n) => {
                out.push(if bullet { format!("- {}", n) } else { n.to_string() });
            }
            Value::Bool(b) => {
                out.push(if bullet { format!("- {}", b) } else { b.to_string() });
            }
            Value::Array(a) => {
                for it in a {
                    flatten(it, out, true);
                }
            }
            Value::Object(o) => {
                for (_, val) in o {
                    flatten(val, out, bullet);
                }
            }
            _ => {}
        }
    }

    let mut lines: Vec<String> = Vec::new();
    if let Some(obj) = parsed.as_object() {
        for (_, v) in obj {
            let is_list = v.is_array();
            flatten(v, &mut lines, is_list);
        }
    } else {
        flatten(&parsed, &mut lines, false);
    }

    let joined = lines.join("\n").trim().to_string();
    if joined.is_empty() {
        t
    } else {
        println!("[ANALYTIC] 🧽 [REPORT NORMALIZE] JSON 응답을 읽을 수 있는 텍스트로 평탄화했습니다.");
        joined
    }
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

    // ── 1차 : 완전일치 ──
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

    // 🌟 ── 2차 : 접두 일치 (교착어 대응) ──
    //    '올해는' / '여름에' 처럼 조사가 붙어 완전일치가 실패한 경우를 구제합니다.
    //    exact_match 원소가 토큰의 접두일 때만 인정하므로 어휘 하드코딩이 없습니다.
    if t_key.is_empty() || s_key.is_empty() {
        for c in &candidates {
            if t_key.is_empty() {
                if let Some((k, stem)) =
                    crate::utils::ai_utils::prefix_match_filter_stem("time_filters", c)
                {
                    println!("[ANALYTIC] 🕒 [TIME PREFIX MATCH] '{}' ← 접두 '{}' → time_filters.{}", c, stem, k);
                    t_key = k;
                }
            }
            if s_key.is_empty() {
                if let Some((k, stem)) =
                    crate::utils::ai_utils::prefix_match_filter_stem("season_filters", c)
                {
                    println!("[ANALYTIC] 🌤️ [SEASON PREFIX MATCH] '{}' ← 접두 '{}' → season_filters.{}", c, stem, k);
                    s_key = k;
                }
            }
            if !t_key.is_empty() && !s_key.is_empty() { break; }
        }
    }

    (t_key, s_key)
}

/// 🌟 [STANZA LANG CODE] parse_commerce_query 와 동일한 매핑입니다.
///  모델 디렉터리가 없으면 tokenize_query_with_pos 가 공백 분할로 폴백하므로,
///  여기서는 매핑만 담당하고 존재 여부는 검사하지 않습니다.
pub fn stanza_lang_code(language: &str) -> &'static str {
    match language {
        "korean" | "ko" => "ko",
        "english" | "en" => "en",
        "japanese" | "ja" => "ja",
        "chinese" | "zh" | "zh-hans" | "zh-hant" | "zh-tw" | "zh-hk" => "zh-hans",
        "french" | "fr" => "fr",
        "german" | "de" => "de",
        "spanish" | "es" => "es",
        "italian" | "it" => "it",
        "portuguese" | "pt" => "pt",
        "dutch" | "nl" => "nl",
        "russian" | "ru" => "ru",
        "arabic" | "ar" => "ar",
        "thai" | "th" => "th",
        "hindi" | "hi" => "hi",
        "bengali" | "bn" => "bn",
        "telugu" | "te" => "te",
        "khmer" | "km" => "km",
        "greek" | "el" => "el",
        "hebrew" | "he" => "he",
        "vietnamese" | "vi" => "vi",
        _ => "en",
    }
}

/// 🌟 [STANZA TOKENIZE + MORPHOLOGY] 질의를 어절 단위로 쪼개고 UPOS 태그와 Lemma 원형을 함께 부착합니다.
///  ── 왜 Lemma 까지 필요한가 ──
///   Stanza 토크나이저는 교착어 어절을 통째로 한 토큰으로 돌려줍니다.
///     "클릭한게" → 1토큰  (클릭 + 한 + 게)
///   이 표면형은 bias.json 의 exact_match("클릭")와 완전일치하지 않고,
///   영어 구 뱅크("clicked", "pressed")와의 코사인도 낮아
///   슬라이딩 윈도우에서 NMS 후보가 단 하나도 만들어지지 않습니다.
///   (로그 실측: [NMS CANDIDATE] 0건 → EVENT FALLBACK 4종 전체)
///   commerce 의 parse_commerce_query 는 이미 lemma_session 을 돌려
///   '가디건찾아줘' → '가디건' 절단을 수행하고 있으므로, 같은 장치를 이식합니다.
///
///  ── 반환 ──
///   (표면형, UPOS 태그, Lemma 원형). 실패 시 태그/Lemma 는 빈 문자열입니다.
pub async fn tokenize_query_with_morphology(
    query: &str,
    lang_code: &str,
) -> Vec<(String, String, String)> {
    let words: Vec<String> = query
        .split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let fallback: Vec<(String, String, String)> = words
        .iter()
        .map(|w| (w.clone(), String::new(), String::new()))
        .collect();
    if words.is_empty() {
        return fallback;
    }

    let base_dir = crate::utils::get_app_dir().join("models").join("stanza");
    let lang_dir = base_dir.join(lang_code);
    if !lang_dir.exists() {
        println!(
            "[ANALYTIC] ⚠️ Stanza 모델 디렉터리가 없어 공백 분할로 폴백합니다: {:?}",
            lang_dir
        );
        return fallback;
    }

    struct UnsafePipelineWrapper(crate::stanza::StanzaPipeline);
    unsafe impl Send for UnsafePipelineWrapper {}

    let wrapper = match crate::stanza::StanzaPipeline::new(base_dir, lang_code).await {
        Ok(p) => UnsafePipelineWrapper(p),
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza 로드 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };
    let mut stanza = wrapper.0;

    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    // ONNX Export 시 고정된 시퀀스 길이를 그대로 존중합니다.
    let mut chunk_size = refs.len();
    for input_meta in &stanza.pos_session.inputs {
        let dims = &input_meta.dimensions;
        if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
            if let Some(&Some(fixed_seq)) = dims.get(0) {
                chunk_size = fixed_seq as usize;
            }
        }
    }
    if chunk_size == 0 {
        chunk_size = refs.len();
    }
    let mut padded = refs.clone();
    let valid_len = padded.len();
    while padded.len() < chunk_size {
        padded.push("<pad>");
    }

    let inputs = match stanza
        .preprocessor
        .encode_to_tensor(&padded, &stanza.pos_session, None, None)
    {
        Ok(v) => v,
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza encode 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };

    let outputs = match stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
        Ok(v) => v,
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza POS 추론 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };

    let t = &outputs[0];
    let shape = t.shape();
    if shape.len() < 2 {
        return fallback;
    }
    let num_classes = if shape.len() == 3 {
        shape[2] as usize
    } else {
        shape[1] as usize
    };

    // ── ① UPOS 디코드 + Lemma 세션 입력용 POS ID 수집 ──
    let mut tags: Vec<String> = Vec::with_capacity(valid_len);
    let mut pos_ids: Vec<i64> = Vec::with_capacity(valid_len);
    for i in 0..valid_len {
        let mut max_val = f32::MIN;
        let mut max_idx = 0usize;
        for c in 0..num_classes {
            let v = if shape.len() == 3 { t[[0, i, c]] } else { t[[i, c]] };
            if v > max_val {
                max_val = v;
                max_idx = c;
            }
        }
        tags.push(
            stanza
                .preprocessor
                .upos_vocab
                .get(max_idx)
                .cloned()
                .unwrap_or_else(|| "X".to_string()),
        );
        pos_ids.push(max_idx as i64);
    }

    // ── ② Lemma 디코드 (commerce parse_commerce_query 와 동일 로직) ──
    let mut lemmas: Vec<String> = vec![String::new(); valid_len];
    if let Ok(lemma_inputs) = stanza.preprocessor.encode_to_tensor(
        &padded,
        &stanza.lemma_session,
        Some(&pos_ids),
        None,
    ) {
        if let Ok(lemma_outputs) = stanza
            .lemma_session
            .run::<'_, '_, '_, i64, f32, _>(lemma_inputs)
        {
            let lt = &lemma_outputs[0];
            let ls = lt.shape();
            if ls.len() == 3 || ls.len() == 4 {
                let is_4d = ls.len() == 4;
                let max_char_len = if is_4d { ls[2] as usize } else { ls[1] as usize };
                let lemma_classes = if is_4d { ls[3] as usize } else { ls[2] as usize };
                for i in 0..valid_len {
                    let mut lemma_str = String::new();
                    for j in 0..max_char_len {
                        let mut mv = f32::MIN;
                        let mut mi = 0usize;
                        for c in 0..lemma_classes {
                            let v = if is_4d { lt[[0, i, j, c]] } else { lt[[i, j, c]] };
                            if v > mv {
                                mv = v;
                                mi = c;
                            }
                        }
                        if let Some(&ch) = stanza.preprocessor.id_to_char.get(&(mi as i64)) {
                            if ch != '<' && ch != '>' && ch != '_' {
                                lemma_str.push(ch);
                            }
                        }
                    }
                    lemmas[i] = lemma_str.trim().to_string();
                }
            }
        }
    }

    let mut out: Vec<(String, String, String)> = Vec::with_capacity(valid_len);
    for i in 0..valid_len {
        out.push((words[i].clone(), tags[i].clone(), lemmas[i].clone()));
    }
    out
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

    // 🌟 [EMBEDDING-BASED TIME/EVENT SELECTION]
    //    LLM 단독 결정 대신, 임베딩 코사인 + NMS 경쟁으로 선택합니다.
    //    ① bias.json 의 time_filters / season_filters / analytic_event_filters 구를 임베딩
    //    ② 질의 임베딩과 각 구 간 코사인 계산
    //    ③ SURPRISAL 게이트로 우연 공명 제거
    //    ④ 마진 부족 시에만 LLM 재판정 (기존 경로 유지)

    // ── ① 질의 임베딩 ──
    let query_emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384]);

    // ── ② time_filters / season_filters 뱅크 임베딩 ──
    let time_phrases = crate::utils::ai_utils::filter_category_phrases(&["time_filters"]);
    let season_phrases = crate::utils::ai_utils::filter_category_phrases(&["season_filters"]);
    let time_prej_phrases = crate::utils::ai_utils::filter_category_prejudice_phrases(&["time_filters"]);
    let season_prej_phrases = crate::utils::ai_utils::filter_category_prejudice_phrases(&["season_filters"]);

    let time_texts: Vec<String> = time_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let season_texts: Vec<String> = season_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let time_prej_texts: Vec<String> = time_prej_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let season_prej_texts: Vec<String> = season_prej_phrases.iter().map(|(_, _, p)| p.clone()).collect();

    let time_embs: Vec<Vec<f32>> = if time_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(time_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; time_texts.len()])
    };
    let season_embs: Vec<Vec<f32>> = if season_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(season_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; season_texts.len()])
    };
    let time_prej_embs: Vec<Vec<f32>> = if time_prej_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(time_prej_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; time_prej_texts.len()])
    };
    let season_prej_embs: Vec<Vec<f32>> = if season_prej_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(season_prej_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; season_prej_texts.len()])
    };

    // ── ③ SURPRISAL 게이트 + 뱅크 크기 편향 제거 ──
    let surprisal_score = |q: &Vec<f32>, idxs: &[usize], embs: &Vec<Vec<f32>>| -> (f32, f32) {
        let mut sims: Vec<f32> = Vec::new();
        for &i in idxs {
            if let Some(e) = embs.get(i) {
                if !e.iter().all(|&v| v == 0.0) {
                    sims.push(crate::utils::ai_utils::cosine_similarity(q, e));
                }
            }
        }
        if sims.is_empty() { return (f32::MIN, 0.0); }
        let n = sims.len() as f32;
        let mean: f32 = sims.iter().sum::<f32>() / n;
        let var: f32 = sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
        let sd = var.sqrt().max(1e-6);
        let mx = sims.iter().cloned().fold(f32::MIN, f32::max);
        let z = (mx - mean) / sd;
        let expect = (2.0 * n.max(2.0).ln()).sqrt();
        (z - expect, mx)
    };

    // time_filters 키별 인덱스 매핑
    let time_key_indices: Vec<(String, Vec<usize>)> = {
        let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, (_, key, _)) in time_phrases.iter().enumerate() {
            map.entry(key.clone()).or_default().push(i);
        }
        map.into_iter().collect()
    };
    let season_key_indices: Vec<(String, Vec<usize>)> = {
        let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, (_, key, _)) in season_phrases.iter().enumerate() {
            map.entry(key.clone()).or_default().push(i);
        }
        map.into_iter().collect()
    };

    // ── time_intent 임베딩 판정 ──
    let mut time_intent = String::new();
    let mut time_score = f32::MIN;
    for (key, idxs) in &time_key_indices {
        let (sur, mx) = surprisal_score(&query_emb, idxs, &time_embs);
        if sur > time_score { time_score = sur; time_intent = key.clone(); }
    }
    // 편견 게이트: 경쟁 개념이 더 잘 설명하면 폐기
    if !time_prej_embs.is_empty() && !time_intent.is_empty() {
        let prej_score = crate::utils::ai_utils::max_pool_sim(&query_emb, &time_prej_embs);
        let own_score = time_embs.iter()
            .map(|e| crate::utils::ai_utils::cosine_similarity(&query_emb, e))
            .fold(f32::MIN, f32::max);
        if prej_score >= own_score {
            emit_term(&format!(
                "  🚫 [TIME PREJ GATE] time_intent='{}' 폐기 (prej {:.4} >= own {:.4})",
                time_intent, prej_score, own_score
            ));
            time_intent = String::new();
            time_score = f32::MIN;
        }
    }

    // ── season_intent 임베딩 판정 ──
    let mut season_intent = String::new();
    let mut season_score = f32::MIN;
    for (key, idxs) in &season_key_indices {
        let (sur, mx) = surprisal_score(&query_emb, idxs, &season_embs);
        if sur > season_score { season_score = sur; season_intent = key.clone(); }
    }
    if !season_prej_embs.is_empty() && !season_intent.is_empty() {
        let prej_score = crate::utils::ai_utils::max_pool_sim(&query_emb, &season_prej_embs);
        let own_score = season_embs.iter()
            .map(|e| crate::utils::ai_utils::cosine_similarity(&query_emb, e))
            .fold(f32::MIN, f32::max);
        if prej_score >= own_score {
            emit_term(&format!(
                "  🚫 [SEASON PREJ GATE] season_intent='{}' 폐기 (prej {:.4} >= own {:.4})",
                season_intent, prej_score, own_score
            ));
            season_intent = String::new();
            season_score = f32::MIN;
        }
    }

    // ── exact_match 가 있으면 임베딩 판정보다 우선 ──
    if !det_time_key.is_empty() { time_intent = det_time_key.clone(); }
    if !det_season_key.is_empty() { season_intent = det_season_key.clone(); }

    // ── 마진 부족 시에만 LLM 재판정 (기존 경로 유지) ──
    let need_time_llm = time_intent.is_empty() && det_time_key.is_empty() && time_score > -1.0;
    let need_season_llm = season_intent.is_empty() && det_season_key.is_empty() && season_score > -1.0;
    if need_time_llm || need_season_llm {
        emit_term("  ⚖️ [EMBED→LLM FALLBACK] 임베딩 마진 부족. LLM 재판정 수행.");
        // 기존 LLM 경로 (parsed 에서 가져오기)
        if need_time_llm {
            time_intent = parsed.get("time_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        }
        if need_season_llm {
            season_intent = parsed.get("season_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        }
    }

    emit_term(&format!(
        "  🕒 [EMBED-BASED TIME] time_intent='{}' (score {:+.4}) | season_intent='{}' (score {:+.4})",
        if time_intent.is_empty() { "-" } else { &time_intent }, time_score,
        if season_intent.is_empty() { "-" } else { &season_intent }, season_score
    ));

    // ── event_types NMS 배틀 결과 확정 ──
    // 슬라이딩 윈도우 + NMS 배틀에서 생존한 스팬 중 이벤트 타입만 추출합니다.
    // 기존 방식은 질의 전체 벡터 1개로 판정하여
    // '클릭한거 뭐야' 에서 '클릭' 이 '뭐야' 와 섞여 신호가 희석되는 문제가 있었습니다.
    // NMS 배틀은 각 단어 윈도우를 독립적으로 경쟁시키므로 이 문제가 없습니다.
    //
    // 🌟 [구조 설명]
    //   수정 1 에서 bank_defs 에 ("event", "click", 구) 등을 추가했으므로
    //   슬라이딩 윈도우 → SURPRISAL 채점 → NMS 배틀 파이프라인이
    //   time/season 과 동일하게 이벤트 타입도 처리합니다.
    //   카테고리별 확정 루프의 "event" 분기가 이미
    //   vec_events: Vec<(String, f32)> 를 채우고 있으므로
    //   여기서 그 결과를 그대로 소비합니다.
    let mut event_types: Vec<String> = Vec::new();
    for event_type in crate::analytic::ANALYTIC_SEARCH_TYPES.iter() {
        let anchor_phrases = crate::analytic::event_type_anchor_phrases(event_type);
        let prej_phrases = crate::analytic::event_type_prejudice_phrases(event_type);
        if anchor_phrases.is_empty() { continue; }
        let a_embs = model.get_embedding_batch(anchor_phrases.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_phrases.len()]);
        let p_embs = if prej_phrases.is_empty() { Vec::new() } else {
            model.get_embedding_batch(prej_phrases.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
        };
        let own = crate::utils::ai_utils::max_pool_sim(&query_emb, &a_embs);
        let prej = if p_embs.is_empty() { 0.0 } else {
            crate::utils::ai_utils::max_pool_sim(&query_emb, &p_embs)
        };
        let score = own - prej;
        emit_term(&format!(
            "  🎯 [EVENT NMS] '{}' | own: {:.4} | prej: {:.4} | score: {:+.4}",
            event_type, own, prej, score
        ));
        if score > 0.0 {
            event_types.push(event_type.to_string());
        }
    }
    // report 는 합성 문서이므로 항상 포함
    if !event_types.iter().any(|t| t == "report") {
        event_types.push("report".to_string());
    }
    // 전부 탈락하면 전체 타입을 스코프로 (리콜 보존)
    if event_types.iter().filter(|t| *t != "report").count() == 0 {
        event_types = ANALYTIC_SEARCH_TYPES.iter().map(|s| s.to_string()).collect();
        emit_term("  🛟 [EVENT FALLBACK] NMS 배틀에서 확정된 이벤트 타입이 없어 전체 타입을 스코프로 둡니다.");
    }
    emit_term(&format!("  ✅ [EVENT TYPES FINAL] {:?}", event_types));

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
    let target = parsed
        .get("target")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            if keywords.is_empty() { query.clone() } else { keywords.join(" ") }
        });

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