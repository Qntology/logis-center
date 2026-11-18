//! 🌟 [QUERY] 자연어 검색: 예전 셸 lib.rs 의 `ai_search_complex` 명령 본문과 그 도우미(범위 필터 · dexie 계획/평가 · 저장 축 맞추기 ·
//! 조건 그라운딩 · 무역 질의 탐침 · 전체 질의 융합). 셸의 Tauri 명령은 `AppState` 와 `TauriSink` 를 넘기는 껍데기이고,
//! 헤드리스(합성 평가 · CLI)는 같은 함수를 `sink::noop()` 등으로 부를 수 있습니다. 함수 본문은 옮기며 바꾸지 않았습니다.

#![allow(clippy::too_many_arguments)]

use crate::model::LogisModel;
use crate::store::VectorStore;
use crate::{AppState, IS_SEARCHING};
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::RwLock;

pub fn sanitize_scope_filter(filter: Option<String>) -> Option<String> {
    // 🌟 [SINGLE SOURCE] 봉투 컬럼 목록을 store.rs 의 물리 스키마 선언과 공유합니다.
    //    기존에는 여기에 ENVELOPE_COLS[9] + TIME_COLS[2] 로 복제되어 있어,
    //    init_all_tables 의 Field 선언이 바뀌면 조용히 어긋났습니다.
    use crate::store::ENVELOPE_COLUMNS;
    let raw = match filter {
        Some(f) if !f.trim().is_empty() => f,
        _ => return None,
    };
    let mut kept: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for clause in raw.split(" AND ") {
        let mut c = clause.trim();
        loop {
            if !(c.starts_with('(') && c.ends_with(')')) { break; }
            let mut depth = 0i32;
            let mut wraps = true;
            let chars: Vec<char> = c.chars().collect();
            for (i, ch) in chars.iter().enumerate() {
                if *ch == '(' { depth += 1; }
                else if *ch == ')' {
                    depth -= 1;
                    if depth == 0 && i + 1 != chars.len() { wraps = false; break; }
                }
            }
            if !wraps || depth != 0 { break; }
            c = c[1..c.len() - 1].trim();
        }
        if c.is_empty() { continue; }

        // 술어의 좌변 컬럼명을 추출합니다. (백틱/공백/연산자 제거)
        let lhs: String = c
            .split(|ch: char| ch == '=' || ch == '<' || ch == '>' || ch == ' ')
            .next()
            .unwrap_or("")
            .trim_matches('`')
            .trim()
            .to_lowercase();

        let is_envelope = ENVELOPE_COLUMNS.iter().any(|e| *e == lhs);
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

/// LanceDB 로 내려보낼 '스코프' SQL 을 만듭니다.
/// 봉투(Envelope) 물리 컬럼만 사용하므로 DataFusion 문법 에러가 구조적으로 발생할 수 없습니다.
pub fn build_scope_filter(ctx: &Value, search_mode: &str) -> Option<String> {
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

    // ── mode : commerce / shipping 트랙 격리 ──
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
        for (key, val_obj) in logis_schema::json_order::sorted_iter(cond) {
            let envelope_col = match key.as_str() {
                "created_at" => "created_at",
                "updated_at" => "updated_at",
                _ => continue,
            };
            let op_str = match val_obj.get("operator").and_then(|v| v.as_str()) {
                Some(s) => s.trim().to_lowercase(),
                None => continue,
            };
            let epoch = |v: Option<&Value>| -> Option<String> {
                match v {
                    Some(Value::Number(n)) => Some(n.to_string()),
                    Some(Value::String(s)) => Some(s.chars().filter(|c| c.is_ascii_digit()).collect::<String>()).filter(|s| !s.is_empty()),
                    _ => None,
                }
            };
            // 🌟 [PERIOD RANGE] 범위는 한 컨텍스트 안에서 두 경계를 함께 겁니다 (행동 로그 기간 질의).
            if op_str.starts_with("between") {
                if let Some(lo) = epoch(val_obj.get("value")) {
                    filters.push(format!("{} >= {}", envelope_col, lo));
                }
                if let Some(hi) = epoch(val_obj.get("value_to")) {
                    filters.push(format!("{} <= {}", envelope_col, hi));
                }
                continue;
            }
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
pub fn build_dexie_plan(ctx: &Value, search_mode: &str) -> Value {
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
        format!("data.{}", crate::utils::bias_schema::canonical_field_name(key))
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
                    if crate::utils::canonical::parse_number_run(s, false).is_some() { return "number"; }
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

    let decimal_comma = ctx
        .get("text")
        .and_then(|v| v.as_str())
        .map_or(false, |t| crate::utils::canonical::decimal_comma_lang(&crate::utils::lang_utils::detect_document_language(t)));
    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
        for (key, val_obj) in logis_schema::json_order::sorted_iter(cond) {
            // created_at / updated_at 은 build_scope_filter 가 이미 SQL 로 처리했으므로 중복 제외
            if key == "created_at" || key == "updated_at" { continue; }

            let path = normalize_path(key);
            let op_raw = val_obj.get("operator").and_then(|v| v.as_str()).unwrap_or("eq");
            // 🌟 LLM 이 "lt [Alts: lte, gte]" 같은 쓰레기를 붙여 보내는 사례가 있어 앞부분만 취합니다.
            let op_clean = op_raw.trim().to_lowercase();
            if op_clean.starts_with("between") {
                for (bound, key) in [("gte", "value"), ("lte", "value_to")] {
                    let raw = match val_obj.get(key) {
                        Some(v) => v.clone(),
                        None => continue,
                    };
                    let empty = match &raw {
                        Value::Null => true,
                        Value::String(s) => s.trim().is_empty() || s == "null",
                        _ => false,
                    };
                    if empty { continue; }
                    let kind = value_kind(&path, &raw);
                    let final_val = if kind == "number" {
                        let n = match &raw {
                            Value::Number(n) => n.as_f64().unwrap_or(0.0),
                            Value::String(s) => match crate::utils::canonical::parse_number_run(s, decimal_comma) {
                                Some(v) => v,
                                None => continue,
                            },
                            _ => 0.0,
                        };
                        if n.fract() == 0.0 && n.abs() < 9e15 { json!(n as i64) } else { json!(n) }
                    } else {
                        json!(raw.as_str().unwrap_or("").trim())
                    };
                    conditions.push(json!({
                        "path": path,
                        "op": bound,
                        "value": final_val,
                        "kind": kind
                    }));
                }
                continue;
            }
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

            if op == "top" || op == "bottom" {
                let pct = val_obj.get("percent_total")
                    .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()))
                    .unwrap_or(20.0);
                let extreme = val_obj.get("extreme").and_then(|v| v.as_bool()).unwrap_or(false);
                conditions.push(json!({
                    "path": path,
                    "op": op,
                    "percent": pct,
                    "kind": "rank",
                    "extreme": extreme
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
                    Value::String(s) => match crate::utils::canonical::parse_number_run(s, decimal_comma) {
                        Some(v) => v,
                        None => continue,
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
        "cc": ctx.get("cc").and_then(|v| v.as_str()).unwrap_or(""),
        "conditions": conditions,
        "keywords": keywords,
        "alternates": ctx.get("alternates").cloned().unwrap_or(json!({})),
        "substantial": ctx.get("substantial").cloned().unwrap_or(json!("")),
        "find": ctx.get("find").cloned().unwrap_or(json!("")),
        "hints": ctx.get("hint").cloned().unwrap_or(json!({})),
        "projection": ctx.get("projection").cloned().unwrap_or(json!([])),
        "relay_period": ctx.get("relay_period").cloned().unwrap_or(Value::Null)
    })
}

pub fn dexie_value_passes(op: &str, kind: &str, have: &Value, want: &Value) -> bool {
    let as_num = |v: &Value| -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let t: String = s.chars().filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-').collect();
                t.parse::<f64>().ok()
            }
            _ => None,
        }
    };
    let as_text = |v: &Value| -> String {
        match v {
            Value::String(s) => s.trim().to_lowercase(),
            other => logis_schema::json_order::to_string_sorted(&other).trim_matches('"').to_lowercase(),
        }
    };
    if kind == "number" {
        let (h, w) = match (as_num(have), as_num(want)) {
            (Some(h), Some(w)) => (h, w),
            _ => return false,
        };
        return match op {
            "gte" => h >= w,
            "gt" => h > w,
            "lte" => h <= w,
            "lt" => h < w,
            "neq" => (h - w).abs() > 1e-9,
            _ => (h - w).abs() <= 1e-9,
        };
    }
    let (h, w) = (as_text(have), as_text(want));
    let cross = || -> bool {
        match (have.as_str(), want.as_str()) {
            (Some(hv), Some(wv)) => cross_script_value_match(wv, hv).is_some(),
            _ => false,
        }
    };
    match op {
        "contains" => h.contains(&w) || cross(),
        "not_contains" => !(h.contains(&w) || cross()),
        "neq" => h != w && !cross(),
        "gte" => h >= w,
        "gt" => h > w,
        "lte" => h <= w,
        "lt" => h < w,
        _ => h == w || cross(),
    }
}

pub fn dexie_condition_passes(cond: &Value, doc: &Value, alternates: &Value) -> bool {
    let path = cond.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let field = path.strip_prefix("data.").unwrap_or(path);
    let op = cond.get("op").and_then(|v| v.as_str()).unwrap_or("eq");
    if op == "top" || op == "bottom" { return true; }
    let want = cond.get("value").cloned().unwrap_or(Value::Null);
    let kind = cond.get("kind").and_then(|v| v.as_str()).unwrap_or("string");
    let mut axes: Vec<String> = vec![field.to_string()];
    if let Some(arr) = alternates.get(field).and_then(|v| v.as_array()) {
        for a in arr.iter() {
            if let Some(s) = a.as_str() { axes.push(s.to_string()); }
        }
    }
    axes.iter().any(|f| match doc.get(f.as_str()) {
        Some(have) if !have.is_null() => dexie_value_passes(op, kind, have, &want),
        _ => op == "neq" || op == "not_contains",
    })
}

pub fn evaluate_dexie_plan(
    plan: &Value,
    docs: &[Value],
) -> (usize, usize, Vec<(String, usize, usize, usize)>) {
    let conds: Vec<Value> = plan
        .get("conditions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let alternates = plan.get("alternates").cloned().unwrap_or(json!({}));
    let types: Vec<String> = plan
        .get("types")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.trim().to_lowercase())).collect())
        .unwrap_or_default();
    let primary = plan
        .get("type")
        .and_then(|v| v.as_str())
        .map(crate::utils::canonical::relay_type_family)
        .unwrap_or_default();
    let primary_scoped = !primary.is_empty()
        && types.iter().any(|t| crate::utils::canonical::relay_type_family(t) == primary);
    let mut fields: Vec<String> = Vec::new();
    for c in conds.iter() {
        let path = c.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let f = path.strip_prefix("data.").unwrap_or(path).to_string();
        if !f.is_empty() && !fields.contains(&f) { fields.push(f); }
    }
    let mut per: Vec<(String, usize, usize, usize)> = fields
        .iter()
        .map(|f| (f.clone(), 0usize, 0usize, 0usize))
        .collect();
    let mut eligible = 0usize;
    let mut all_pass = 0usize;
    for d in docs.iter() {
        let t = d
            .get("type")
            .or_else(|| d.get("doc_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if !types.is_empty() && !types.iter().any(|x| *x == t) { continue; }
        if primary_scoped && crate::utils::canonical::relay_type_family(&t) != primary { continue; }
        eligible += 1;
        let ok: Vec<bool> = fields
            .iter()
            .map(|f| {
                conds
                    .iter()
                    .filter(|c| {
                        let p = c.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        p.strip_prefix("data.").unwrap_or(p) == f.as_str()
                    })
                    .all(|c| dexie_condition_passes(c, d, &alternates))
            })
            .collect();
        let fails = ok.iter().filter(|x| !**x).count();
        if fails == 0 { all_pass += 1; }
        for (i, pass) in ok.iter().enumerate() {
            if *pass {
                per[i].1 += 1;
            } else {
                per[i].3 += 1;
                if fails == 1 { per[i].2 += 1; }
            }
        }
    }
    (eligible, all_pass, per)
}

/// 힌트 축이 회수 문서에서 실제로 만족되었는지 관측합니다.
///
///  ── 왜 하드와 분리하는가 ──
///   힌트는 결과 집합을 좁히지 않으므로 all_pass 계산에 넣으면 안 됩니다.
///   그러나 '그 축을 힌트로 내린 판단이 옳았는가' 는 다음 회차의 강등·치환 판정에
///   직접 쓰이는 정보인데, 지금은 evaluated 가 0 이라 관측 자체가 없습니다.
pub fn evaluate_dexie_hints(plan: &Value, docs: &[Value]) -> Vec<(String, usize, usize, usize)> {
    use crate::utils::canonical::{kind_of, CanonKind};
    let alternates = plan.get("alternates").cloned().unwrap_or(json!({}));
    let hints = match plan.get("hints").and_then(|v| v.as_object()) {
        Some(h) => h.clone(),
        None => return Vec::new(),
    };
    let mut out: Vec<(String, usize, usize, usize)> = Vec::new();
    for (field, spec) in logis_schema::json_order::sorted_iter(&hints) {
        let op = spec.get("operator").and_then(|v| v.as_str()).unwrap_or("contains");
        let want = spec.get("value").cloned().unwrap_or(Value::Null);
        if want.is_null() { continue; }
        let numeric_axis = matches!(kind_of(field), CanonKind::Numeric);
        let kind = if want.is_number()
            || (numeric_axis && want.as_str().map_or(false, |s| s.trim().parse::<f64>().is_ok()))
        {
            "number"
        } else {
            "string"
        };
        let mut axes: Vec<String> = vec![field.clone()];
        if let Some(arr) = alternates.get(field.as_str()).and_then(|v| v.as_array()) {
            for a in arr.iter() {
                if let Some(s) = a.as_str() {
                    if !axes.iter().any(|x| x == s) { axes.push(s.to_string()); }
                }
            }
        }
        let negative = op == "neq" || op == "not_contains";
        let ranking = op == "top" || op == "bottom";
        let want_script = want.as_str().map(|s| dominant_script(s)).unwrap_or("none");
        let mut present = 0usize;
        let mut satisfied = 0usize;
        let mut cross_script = 0usize;
        for d in docs.iter() {
            let values: Vec<&Value> = axes
                .iter()
                .flat_map(|k| doc_axis_values(d, k))
                .filter(|v| !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
                .collect();
            if !values.is_empty() { present += 1; }
            let pass = if values.is_empty() {
                negative
            } else if ranking {
                true
            } else if negative {
                values.iter().all(|v| dexie_value_passes(op, kind, v, &want))
            } else {
                values.iter().any(|v| dexie_value_passes(op, kind, v, &want))
            };
            if pass { satisfied += 1; }
            let foreign = kind == "string"
                && !pass
                && !values.is_empty()
                && want_script != "none"
                && values.iter().all(|v| {
                    v.as_str().map_or(false, |s| {
                        let sc = dominant_script(s);
                        sc != "none" && sc != want_script
                    })
                });
            if foreign { cross_script += 1; }
        }
        out.push((field.clone(), satisfied, present, cross_script));
    }
    out
}

pub fn doc_axis_values<'a>(doc: &'a Value, field: &str) -> Vec<&'a Value> {
    let mut out: Vec<&'a Value> = Vec::new();
    let obj = match doc.as_object() {
        Some(o) => o,
        None => return out,
    };
    let scalar = |v: &Value| !(v.is_object() || v.is_array());
    if let Some(v) = obj.get(field) {
        match v {
            Value::Array(xs) => out.extend(xs.iter().filter(|x| scalar(x))),
            other if scalar(other) => out.push(other),
            _ => {}
        }
    }
    for node in logis_schema::json_order::sorted_values(obj) {
        match node {
            Value::Object(o) => {
                if let Some(v) = o.get(field).filter(|v| scalar(v)) {
                    out.push(v);
                }
            }
            Value::Array(rows) => {
                for r in rows.iter() {
                    if let Some(v) = r.as_object().and_then(|o| o.get(field)).filter(|v| scalar(v)) {
                        out.push(v);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

pub fn plan_alternate_axis_map(plan: &Value) -> Value {
    plan.get("alternates").cloned().unwrap_or(json!({}))
}

pub fn collect_storage_axes(docs: &[Value]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for d in docs.iter() {
        if let Some(obj) = d.as_object() {
            for (k, v) in logis_schema::json_order::sorted_iter(obj) {
                if v.is_null() { continue; }
                if let Some(s) = v.as_str() {
                    if s.trim().is_empty() { continue; }
                }
                if v.is_object() || v.is_array() { continue; }
                if !out.iter().any(|e| e == k) { out.push(k.clone()); }
            }
        }
    }
    out
}

/// 🌟 [MONETARY AXIS] 이 축이 돈을 담는가. 치환 호환성 판정의 1차 관문입니다.
///
///  ── 실측 사고 ──
///   amount 가 저장되지 않아 AXIS SUBSTITUTE 가 열었습니다: amount → weight_net (0.6406).
///   두 축은 detect_field_format 상 둘 다 Numeric 이라 형식 게이트를 그대로 통과합니다.
///   그러나 amount 에는 CURRENCY COMPANION 이 붙인 `currency contains 'USD'` 가 따라다니고
///   weight_net 에는 통화가 없습니다. 조건을 유지한 채 축만 바꾼다는 치환의 전제가 깨집니다.
///   '1500 달러 이상' 이 '중량 1500 이상' 으로 해석되면 결과가 조용히 틀립니다.
pub fn is_monetary_axis(field: &str) -> bool {
    ["amount", "price", "charge", "value", "debit", "credit", "balance", "premium", "fee"]
        .iter()
        .any(|k| field.contains(k))
}

pub fn axis_format_compatible(a: &str, b: &str) -> bool {
    use crate::utils::ai_utils::{detect_field_format, FieldFormat};
    // 🌟 금액 축과 비금액 축은 형식이 같아도 치환할 수 없습니다.
    if is_monetary_axis(a) != is_monetary_axis(b) { return false; }
    let fa = detect_field_format(a);
    let fb = detect_field_format(b);
    if fa == fb { return true; }
    matches!(
        (fa, fb),
        (FieldFormat::Numeric, FieldFormat::Identifier)
            | (FieldFormat::Identifier, FieldFormat::Numeric)
            | (FieldFormat::Text, FieldFormat::Address)
            | (FieldFormat::Address, FieldFormat::Text)
    )
}

pub fn storage_page_type(page_type: &str) -> String {
    let t = page_type.trim();
    if crate::utils::bias_schema::is_trade_doc_type(t) {
        return t.to_string();
    }
    let upper = t.to_uppercase();
    if crate::utils::bias_schema::is_trade_doc_type(&upper) {
        return upper;
    }
    t.to_string()
}

pub fn axis_label_bank(page_type: &str, field: &str) -> Vec<String> {
    use crate::utils::ai_utils::{is_value_example_phrase, label_phrase_bank_multilingual, semantic_anchor_text};
    let canon = crate::utils::bias_schema::canonical_bias_type(page_type);
    let (raw, _) = if crate::utils::bias_schema::is_trade_doc_type(page_type) {
        crate::model::merge::owner_label_bank("en", canon, field)
    } else {
        label_phrase_bank_multilingual("en", canon, field)
    };
    let mut bank: Vec<String> = Vec::new();
    for p in raw.into_iter() {
        let t = p.trim();
        if t.is_empty() || is_value_example_phrase(t) { continue; }
        if bank.iter().any(|e| e.eq_ignore_ascii_case(t)) { continue; }
        bank.push(t.to_string());
        if bank.len() >= 128 { break; }
    }
    if bank.is_empty() {
        let s = semantic_anchor_text("en", page_type, field);
        if !s.trim().is_empty() { bank.push(s); }
    }
    bank
}

pub fn bank_affinity(a: &[&[f32]], b: &[&[f32]]) -> f32 {
    use crate::utils::ai_utils::cosine_similarity;
    if a.is_empty() || b.is_empty() { return 0.0; }
    let directed = |x: &[&[f32]], y: &[&[f32]]| -> f32 {
        x.iter()
            .map(|u| y.iter().map(|v| cosine_similarity(u, v)).fold(f32::MIN, f32::max))
            .sum::<f32>()
            / x.len() as f32
    };
    0.5 * (directed(a, b) + directed(b, a))
}

pub async fn nearest_storage_axis(
    model: &LogisModel,
    page_type: &str,
    blocked_field: &str,
    storage_axes: &[String],
) -> Option<(String, f32)> {
    let pt = storage_page_type(page_type);
    let page_type = pt.as_str();
    let schema: Vec<String> = crate::parsing::get_detail_schema_fields(page_type, "", "en")
        .into_iter()
        .map(|(f, _, _, _)| f)
        .filter(|f| !crate::utils::bias_schema::is_system_axis(f))
        .collect();
    if schema.is_empty() {
        println!(
            "[AI-SEARCH] ⚪ [AXIS SUBSTITUTE SKIP / NO SCHEMA] '{}' 서식의 스키마 축 목록을 얻지 못했습니다. 저장본의 created_at·updated_at·type 같은 메타 축이 후보로 섞이므로 치환하지 않고 강등 경로에 맡깁니다.",
            page_type
        );
        return None;
    }
    let compatible: Vec<String> = storage_axes
        .iter()
        .filter(|a| a.as_str() != blocked_field)
        .filter(|a| schema.iter().any(|s| s == *a))
        .filter(|a| axis_format_compatible(blocked_field, a))
        .cloned()
        .collect();
    if compatible.is_empty() { return None; }

    let mut banks: Vec<Vec<String>> = Vec::with_capacity(compatible.len() + 1);
    banks.push(axis_label_bank(page_type, blocked_field));
    for a in compatible.iter() {
        banks.push(axis_label_bank(page_type, a));
    }
    let mut uniq: Vec<String> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for b in banks.iter() {
        for p in b.iter() {
            if index.contains_key(p) { continue; }
            index.insert(p.clone(), uniq.len());
            uniq.push(p.clone());
        }
    }
    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(uniq.len());
    for part in uniq.chunks(200) {
        let e = model.get_embedding_batch(part.to_vec()).await.ok()?;
        if e.len() != part.len() { return None; }
        embs.extend(e);
    }
    let vecs: Vec<Vec<&[f32]>> = banks
        .iter()
        .map(|b| {
            b.iter()
                .filter_map(|p| index.get(p).map(|&i| embs[i].as_slice()))
                .filter(|e| !e.is_empty() && !e.iter().all(|&v| v == 0.0))
                .collect()
        })
        .collect();
    if vecs[0].is_empty() { return None; }

    let mut scored: Vec<(String, f32)> = Vec::with_capacity(compatible.len());
    for (i, a) in compatible.iter().enumerate() {
        if vecs[i + 1].is_empty() { continue; }
        scored.push((a.clone(), bank_affinity(&vecs[0], &vecs[i + 1])));
    }
    scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "[AI-SEARCH] 🧮 [AXIS SUBSTITUTE / BANK AFFINITY] '{}' ({}구) ↔ 저장 축 {}개 다국어 라벨 뱅크 양방향 최근접 평균: {:?}",
        blocked_field,
        vecs[0].len(),
        scored.len(),
        scored.iter().take(6).map(|(a, s)| format!("{}({:.4})", a, s)).collect::<Vec<_>>()
    );

    // 🌟 [SMALL POOL GUARD] 후보가 2개면 꼬리가 1개라 표준편차가 0 이거나 무의미합니다.
    //    그 상태에서 '평균 + 표준편차' 를 재면 1위가 언제나 통과합니다.
    //    remap_off_schema_axes 가 같은 이유로 이미 3개 미만을 기각하므로 기준을 맞춥니다.
    //    치환하지 못하면 아래 STORAGE GAP → RUNTIME DEMOTE 가 받아 조건을 힌트로 내립니다.
    if scored.len() < 3 {
        println!(
            "[AI-SEARCH] ⚪ [AXIS SUBSTITUTE SKIP] '{}' 를 받아 줄 호환 축이 {}개뿐이라 자기 분포로 이상치를 판정할 수 없습니다. 근거 없는 치환보다 조건 강등이 안전합니다.",
            blocked_field, scored.len()
        );
        return None;
    }
    let tail: Vec<f32> = scored[1..].iter().map(|(_, s)| *s).collect();
    let n = tail.len() as f32;
    let mean = tail.iter().sum::<f32>() / n;
    let var = tail.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let sd = var.max(0.0).sqrt();
    if sd <= 1e-6 { return None; }
    let z = (scored[0].1 - mean) / sd;
    let need = crate::utils::ai_utils::gumbel_expected_z(scored.len()).max(1.0);
    if z < need {
        println!(
            "[AI-SEARCH] ⚪ [AXIS SUBSTITUTE SKIP / NOT AN OUTLIER] '{}' → 1위 '{}' ({:.4}) 가 나머지 {}개 평균 {:.4} 보다 {:.2}σ 앞서지만, 후보 {}개 중 최댓값이 우연히 앞서는 기대치 {:.2}σ 에 못 미칩니다. 같은 계열(날짜·금액) 축끼리는 뱅크가 비슷해 누가 1위여도 이상치가 아니며, 이때 치환하면 조건이 조용히 다른 사실로 바뀝니다.",
            blocked_field, scored[0].0, scored[0].1, scored.len() - 1, mean, z, scored.len(), need
        );
        return None;
    }
    Some(scored[0].clone())
}

/// 문자 체계 판정. FTS(ngram 문자열 포함)가 물리적으로 성립할 수 있는지를 가릅니다.
///
///  ── 왜 언어가 아니라 문자 체계인가 ──
///   FTS 는 의미가 아니라 바이트/문자 일치입니다. 한국어 질의와 영어 청크는
///   같은 뜻이어도 공통 부분 문자열이 없어 FTS 가 구조적으로 0건입니다.
///   반대로 중국어 질의와 일본어 청크는 한자를 공유해 부분적으로 발화합니다.
///   언어 판정기보다 문자 체계가 이 물음에 더 정확히 답합니다.
pub fn cross_script_value_match(want: &str, have: &str) -> Option<(&'static str, f32)> {
    let w = want.trim();
    let h = have.trim();
    if w.is_empty() || h.is_empty() {
        return None;
    }
    let (ws, hs) = (dominant_script(w), dominant_script(h));
    if ws == "none" || hs == "none" || ws == hs {
        return None;
    }
    if let (Some(a), Some(b)) = (
        crate::utils::ai_utils::country_code_of(w),
        crate::utils::ai_utils::country_code_of(h),
    ) {
        if a == b {
            return Some(("country", 1.0));
        }
    }
    if let (Some(a), Some(b)) = (
        crate::utils::ai_utils::currency_name_exact(w),
        crate::utils::ai_utils::currency_name_exact(h),
    ) {
        if a == b {
            return Some(("currency", 1.0));
        }
    }
    match crate::nl_convert::phonetic_similarity(w, h) {
        Some(s) if s >= crate::nl_convert::PHONETIC_PASS => Some(("phonetic", s)),
        _ => None,
    }
}

pub fn dominant_script(s: &str) -> &'static str {
    let (mut latin, mut hangul, mut kana, mut han, mut arabic, mut cyrillic) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for c in s.chars() {
        let u = c as u32;
        if c.is_ascii_alphabetic() || (0x00C0..=0x024F).contains(&u) {
            latin += 1;
        } else if (0xAC00..=0xD7A3).contains(&u) || (0x1100..=0x11FF).contains(&u) {
            hangul += 1;
        } else if (0x3040..=0x30FF).contains(&u) {
            kana += 1;
        } else if (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u) {
            han += 1;
        } else if (0x0600..=0x06FF).contains(&u) || (0x0750..=0x077F).contains(&u) {
            arabic += 1;
        } else if (0x0400..=0x04FF).contains(&u) {
            cyrillic += 1;
        }
    }
    let mut best: (&'static str, usize) = ("none", 0);
    for (name, n) in [
        ("latin", latin),
        ("hangul", hangul),
        ("kana", kana),
        ("han", han),
        ("arabic", arabic),
        ("cyrillic", cyrillic),
    ] {
        if n > best.1 { best = (name, n); }
    }
    best.0
}

pub fn is_relay_draft(doc: &Value) -> bool {
    crate::store::is_relay_draft(doc)
}

// =====================================================================
// 🌟 [QUERY-SIDE MODE REROUTE] 커머스 질의가 무역 서식을 지목하는지 판정합니다.
// ---------------------------------------------------------------------
//  비전 경로의 [MODE REROUTE] 와 대칭입니다. 커머스 태스크로 올라온 서식 이미지는
//  mode='shipping', type='CI' 처럼 저장되므로, 같은 질의가 커머스 스코프
//  (mode='commerce', type IN (goods, ...)) 로 나가면 구조적으로 0건입니다.
// =====================================================================
pub static TRADE_QUERY_BANKS: Lazy<RwLock<Option<(Vec<(String, Vec<f32>)>, Vec<Vec<f32>>)>>> =
    Lazy::new(|| RwLock::new(None));

pub async fn trade_query_banks(
    model: &LogisModel,
) -> Option<(Vec<(String, Vec<f32>)>, Vec<Vec<f32>>)> {
    if let Ok(g) = TRADE_QUERY_BANKS.read() {
        if let Some(b) = g.as_ref() {
            return Some(b.clone());
        }
    }
    let titles = crate::utils::ai_utils::all_trade_doc_titles();
    let comm = crate::logic::anchor_phrases(
        crate::logic::COMMERCE_QUERY_ANCHOR,
        crate::logic::COMMERCE_QUERY_ANCHOR_ML,
    );
    let mut texts: Vec<String> = titles.iter().map(|(_, t)| t.clone()).collect();
    let n_title = texts.len();
    texts.extend(comm.iter().cloned());
    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for part in texts.chunks(200) {
        let e = model
            .get_embedding_batch(part.to_vec())
            .await
            .unwrap_or_else(|_| vec![Vec::new(); part.len()]);
        embs.extend(e);
    }
    if embs.len() != texts.len() {
        return None;
    }
    let title_bank: Vec<(String, Vec<f32>)> = titles
        .iter()
        .zip(embs.iter().take(n_title))
        .filter(|(_, e)| !e.is_empty())
        .map(|((c, _), e)| (c.clone(), e.clone()))
        .collect();
    let comm_bank: Vec<Vec<f32>> = embs
        .iter()
        .skip(n_title)
        .filter(|e| !e.is_empty())
        .cloned()
        .collect();
    if title_bank.len() < 3 || comm_bank.len() < 3 {
        return None;
    }
    if let Ok(mut g) = TRADE_QUERY_BANKS.write() {
        *g = Some((title_bank.clone(), comm_bank.clone()));
    }
    Some((title_bank, comm_bank))
}

/// 한 뱅크에 대한 질의의 자기 분포 초과분: (z_top − √(2lnN), top cos, top idx)
pub fn bank_excess(sims: &[f32]) -> Option<(f32, f32, usize)> {
    if sims.len() < 3 { return None; }
    let n = sims.len() as f32;
    let mean = sims.iter().sum::<f32>() / n;
    let sd = (sims.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
    if sd <= 1e-6 { return None; }
    let mut ti = 0usize;
    for (i, s) in sims.iter().enumerate() {
        if *s > sims[ti] { ti = i; }
    }
    let z = (sims[ti] - mean) / sd;
    let expected = crate::utils::ai_utils::gumbel_expected_z(sims.len());
    Some((z - expected, sims[ti], ti))
}

pub async fn probe_trade_query<E: Fn(&str)>(
    model: &LogisModel,
    query: &str,
    emit: &E,
) -> Option<String> {
    use crate::utils::ai_utils::cosine_similarity;

    // ① 서식 코드 접두 번호 (코드 직후에 숫자 또는 구분자, 숫자 4자리 이상)
    let codes: Vec<&str> = crate::logic::TRADE_GROUP_CODES
        .iter()
        .flat_map(|(_, cs)| cs.iter().copied())
        .collect();
    for tok in query.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '[' | ']' | '"' | '\'')
    }) {
        let core = tok
            .trim_start_matches(|c: char| !c.is_ascii_alphabetic())
            .trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        let t = core.to_uppercase();
        if t.len() < 6 { continue; }
        for code in codes.iter() {
            let code: &str = *code;
            if code.len() < 2 || !t.starts_with(code) { continue; }
            let rest = &t[code.len()..];
            let stripped = rest.trim_start_matches(|c: char| matches!(c, '-' | '/' | ':' | '#' | '.' | '_'));
            if stripped.len() == rest.len() && !stripped.starts_with(|c: char| c.is_ascii_digit()) {
                continue;
            }
            let digits = stripped.chars().filter(|c| c.is_ascii_digit()).count();
            let clean = stripped
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '/' | '_'));
            if digits >= 4 && clean {
                emit(&format!(
                    "[TRADE QUERY PROBE] 🔑 질의 토큰 '{}' 이 서식 코드 '{}' 의 접두 번호 형태입니다. 저장 시 resolve_trade_doc_identity 가 같은 접두 규칙으로 문서번호를 확정하므로, 이 토큰은 그 서식의 문서번호를 그대로 옮겨 적은 것입니다.",
                    tok.trim(), code
                ));
                return Some(code.to_string());
            }
        }
    }

    // ② 서식 전문 뱅크 vs 커머스 개념 뱅크
    let mentions = crate::utils::ai_utils::trade_doc_mentions_exact(query);
    let listed: Vec<String> = mentions
        .iter()
        .map(|m| {
            format!(
                "{}→{:?}{}",
                m.text,
                m.codes,
                if !m.by_title { "·코드" } else if m.partial { "·부분" } else { "·전문" }
            )
        })
        .collect();
    crate::utils::score_dynamics::record_baseline("search.query_trade_mentions", mentions.len() as f32);
    if let Some((code, why)) = crate::utils::ai_utils::trade_mentions_decisive(&mentions) {
        crate::utils::score_dynamics::record_baseline("search.query_reroute_exact", 1.0);
        emit(&format!(
            "[TRADE QUERY PROBE] 🔒 질의에 서로 다른 허브 서식(PO·CI·BL·LC) 두 가지가 닫힌 어휘로 적혀 있습니다: {} | 전체 {:?}. 허브 서식은 무역 참조 그래프의 중심이라 서로 다른 둘이 함께 적히면 무역 서식 질의로 확정합니다. 같은 허브를 코드와 이름으로 두 번 적은 경우와 공백 없는 중국어·일본어 서식명이 다른 글자에 이어 붙은 부분 일치는 여기서 세지 않고 코사인 확인을 거칩니다. 서식 코드·서식 이름 표는 shipping 질의 해석과 같은 표를 쓰되, 질의 분류용이라 처격 조사가 붙은 허브 코드(CI에)도 셉니다.",
            why, listed
        ));
        return Some(code);
    }
    let (title_bank, comm_bank) = trade_query_banks(model).await?;
    let q = model.get_embedding(query.to_string()).await.ok()?;
    if q.is_empty() { return None; }
    let t_sims: Vec<f32> = title_bank.iter().map(|(_, e)| cosine_similarity(&q, e)).collect();
    let c_sims: Vec<f32> = comm_bank.iter().map(|e| cosine_similarity(&q, e)).collect();
    let (t_ex, t_top, t_idx) = bank_excess(&t_sims)?;
    let (c_ex, c_top, _) = bank_excess(&c_sims)?;
    let code = title_bank[t_idx].0.clone();
    if t_ex > 0.0 && t_ex > c_ex {
        emit(&format!(
            "[TRADE QUERY PROBE] 📄 서식 전문 뱅크 초과분 {:+.3} (최고 '{}' cos {:.4}, N={}) 이 커머스 개념 뱅크 초과분 {:+.3} (cos {:.4}, N={}) 을 앞섭니다. 두 뱅크의 최댓값 기대치 √(2lnN) 을 각각 뺀 값이므로 뱅크 크기 차이는 이미 상쇄되어 있습니다.",
            t_ex, code, t_top, t_sims.len(), c_ex, c_top, c_sims.len()
        ));
        return Some(code);
    }
    let top_named = mentions
        .iter()
        .find(|m| crate::utils::ai_utils::trade_mention_is_hub(m) && m.codes.iter().any(|c| *c == code));
    if let Some(m) = top_named {
        if t_ex > c_ex {
            crate::utils::score_dynamics::record_baseline("search.query_reroute_exact", 1.0);
            emit(&format!(
                "[TRADE QUERY PROBE] 🔒 허브 서식 '{}' {:?}{} 이 질의에 적혀 있고, 서식 전문 뱅크 최고 '{}' (cos {:.4}) 가 바로 그 서식입니다. 서식 뱅크 초과분 {:+.3} 이 커머스 뱅크 초과분 {:+.3} 을 앞서므로(뱅크 크기 보정 뒤 비교) 음수여도 전환합니다. 서식 이름에 조건을 여럿 붙인 긴 질의는 서식 뱅크 전체와 두루 닮아 평균이 올라가므로 최댓값의 z 초과분이 작게 나옵니다. 글자 근거가 초과분의 절대 기준을 대신합니다.",
                m.text, m.codes, if m.partial { " (공백 없이 다른 글자에 이어 붙은 부분 일치라 코사인 확인을 함께 봅니다)" } else { " (닫힌 어휘 완전 일치)" }, code, t_top, t_ex, c_ex
            ));
            return Some(code);
        }
    }
    emit(&format!(
        "[TRADE QUERY PROBE] 🛒 커머스 유지 — 서식 전문 초과분 {:+.3} (최고 '{}' cos {:.4}) vs 커머스 개념 초과분 {:+.3} (cos {:.4}). 서식 접두 번호도 없고, 닫힌 어휘 서식 이름 {:?} 도 전환 근거가 되지 못했습니다.",
        t_ex, code, t_top, c_ex, c_top, listed
    ));
    None
}

/// 🌟 [CONDITION GROUNDING] 저장 데이터로 통과시킬 수 없는 조건을 힌트로 내립니다.
///
///   Plinko 는 낱말을 가장 가까운 속성에 붙이므로 '책상' 이 color, '결제금액' 이 payment_method, 285200 이 quantity 조건이 됩니다.
///   프론트엔드는 조건으로 후보를 거르기 때문에, 맞는 행이 하나도 없는 조건은 정답을 화면에서 지우기만 합니다.
///   이런 조건은 힌트로 내려 청크 검색의 근거로만 남깁니다. 조건이 전부 내려간 맥락은 플랜에서 뺍니다. 조건 없는 플랜은 후보를
///   전부 통과시키므로 남겨 두면 다른 맥락의 올바른 조건까지 무력화합니다.
///
///   질의가 가리키는 유형과 다른 맥락도 플랜에서 뺍니다. '비타민 C 세럼 들어간 주문' 이 상품 맥락의 제목 조건을 만들면
///   제목 칸이 없는 주문이 전부 걸러집니다. 이런 맥락은 회수에만 쓰입니다.
pub async fn ground_conditions<E: Fn(&str)>(store: &VectorStore, structured: &mut Value, cc: &str, domains: &[String], emit: &E) {
    use crate::utils::ai_utils::FieldFormat;
    const SAMPLE_ROWS: usize = 2000;
    const LITERAL_ROWS: usize = 200;

    fn number_of(v: &Value) -> Option<f64> {
        v.as_f64().or_else(|| v.as_str().and_then(|s| s.replace(',', "").trim().parse::<f64>().ok()))
    }
    fn in_types(row: &Value, types: &[String]) -> bool {
        types.is_empty() || row.get("type").and_then(|t| t.as_str()).map_or(false, |t| types.iter().any(|x| x == t))
    }

    let scope = commerce_scope(cc);
    let mut sample: Option<Vec<Value>> = None;
    let Some(contexts) = structured.get_mut("context").and_then(|v| v.as_array_mut()) else { return };
    for ctx in contexts.iter_mut() {
        let ctx_type = ctx.get("type").and_then(|v| v.as_str()).map(|t| t.strip_suffix("_items").unwrap_or(t).to_string()).unwrap_or_default();
        if !ctx_type.is_empty() && !domains.iter().any(|d| *d == ctx_type) {
            if let Some(obj) = ctx.as_object_mut() {
                obj.insert("plan_off".to_string(), json!(true));
            }
            emit(&format!("[AI-SEARCH] ⚓ [CONDITION GROUNDING] '{}' 맥락은 질의가 가리키는 유형 {:?} 이 아니라 플랜에서 뺍니다.", ctx_type, domains));
            continue;
        }
        let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()).cloned() else { continue };
        let types: Vec<String> = ctx
            .get("types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|t| t.as_str().map(|s| s.strip_suffix("_items").unwrap_or(s).to_string())).collect())
            .unwrap_or_default();
        let mut demoted: Vec<(String, Value)> = Vec::new();
        for (key, spec) in logis_schema::json_order::sorted_iter(&cond) {
            let field = crate::utils::bias_schema::canonical_field_name(key).to_string();
            let format = crate::utils::ai_utils::detect_field_format(&field);
            // 상태는 코드로, 날짜는 시각으로 저장되어 글자로 대조할 수 없습니다.
            if field.contains("status") || matches!(format, FieldFormat::Date | FieldFormat::Synthesis) {
                continue;
            }
            let op = spec.get("operator").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase();
            let compare = matches!(op.as_str(), "gt" | "gte" | "lt" | "lte");
            let grounded = if op == "top" || op == "bottom" {
                // 순위 조건은 맥락의 유형에 그 칸이 있어야 뜻이 있습니다. 상품에는 무게가 없는데 'Lightweight' 가 무게 하위 20% 를 만들었습니다.
                if sample.is_none() {
                    let rows = store.rows_mentioning(&scope, "", SAMPLE_ROWS).await.unwrap_or_default();
                    sample = Some(rows.iter().filter_map(|r| serde_json::from_str::<Value>(r).map(logis_schema::json_order::sorted).ok()).collect());
                }
                sample.as_deref().unwrap_or_default().iter().any(|r| {
                    r.get("type").and_then(|t| t.as_str()) == Some(ctx_type.as_str()) && r.get(field.as_str()).and_then(number_of).is_some()
                })
            } else if format == FieldFormat::Numeric && compare {
                let Some(target) = spec.get("value").and_then(number_of) else { continue };
                if sample.is_none() {
                    let rows = store.rows_mentioning(&scope, "", SAMPLE_ROWS).await.unwrap_or_default();
                    sample = Some(rows.iter().filter_map(|r| serde_json::from_str::<Value>(r).map(logis_schema::json_order::sorted).ok()).collect());
                }
                let rows = sample.as_deref().unwrap_or_default();
                let known = rows.iter().filter(|r| in_types(r, &types)).filter_map(|r| r.get(field.as_str()).and_then(number_of));
                let mut seen = false;
                let mut passes = false;
                for x in known {
                    seen = true;
                    passes |= match op.as_str() {
                        "gt" => x > target,
                        "gte" => x >= target,
                        "lt" => x < target,
                        _ => x <= target,
                    };
                }
                passes || !seen
            } else if op == "contains" || op == "eq" {
                let Some(value) = spec.get("value").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) else { continue };
                if value.contains(|c: char| matches!(c, '%' | '_')) { continue; }
                if format == FieldFormat::Numeric && value.chars().all(|c| c.is_ascii_digit() || c == '.') { continue; }
                match store.rows_mentioning(&scope, value, LITERAL_ROWS).await {
                    Ok(rows) => rows.iter().any(|r| {
                        serde_json::from_str::<Value>(r).map(logis_schema::json_order::sorted)
                            .map_or(false, |d| in_types(&d, &types) && crate::search_fusion::field_contains(&d, &field, value))
                    }),
                    Err(_) => true,
                }
            } else {
                continue;
            };
            crate::utils::score_dynamics::record_baseline("search.condition_grounded", if grounded { 1.0 } else { 0.0 });
            if !grounded {
                demoted.push((key.clone(), spec.clone()));
            }
        }
        if demoted.is_empty() { continue; }
        let has_status = ctx.get("status").and_then(|v| v.as_str()).map_or(false, |s| !s.trim().is_empty() && s.trim() != "null");
        if let Some(obj) = ctx.as_object_mut() {
            let mut left = 0usize;
            if let Some(c) = obj.get_mut("condition").and_then(|v| v.as_object_mut()) {
                for (key, _) in demoted.iter() {
                    c.remove(key);
                }
                left = c.len();
            }
            let hint = obj.entry("hint".to_string()).or_insert_with(|| json!({}));
            if !hint.is_object() { *hint = json!({}); }
            if let Some(h) = hint.as_object_mut() {
                for (key, spec) in demoted.iter() {
                    h.entry(key.clone()).or_insert_with(|| spec.clone());
                }
            }
            if left == 0 && !has_status {
                obj.insert("plan_off".to_string(), json!(true));
            }
        }
        emit(&format!(
            "[AI-SEARCH] ⚓ [CONDITION GROUNDING] 저장 데이터로 통과시킬 수 없어 힌트로 내립니다: {:?}",
            demoted.iter().map(|(k, s)| format!("{} {} {}", k, logis_schema::json_order::Sorted(s.get("operator").unwrap_or(&Value::Null)), logis_schema::json_order::Sorted(s.get("value").unwrap_or(&Value::Null)))).collect::<Vec<_>>()
        ));
    }
}

/// 🌟 [TRADE GROUNDING] 서식 질의의 하드 조건은 질의에 근거가 있을 때만 남깁니다.
///  · 식별자(번호 · 참조 · 컨테이너 · HS 코드): 정규화한 값이 질의(정규화)에 들어 있어야 합니다.
///  · 수: 질의에 그 수가 따로 선 숫자로 나와야 합니다 ('6月' 의 6 · 'LC-DE7472990' 속 2990 은 아님).
///  · 날짜: 값의 연도가 질의에 나오거나 올해여야 합니다 (지어낸 'etd > 2019-06-30' 은 버림).
///  · 통화: 코드 · 기호 · 이름이 질의에 나와야 합니다 (일본어 질의의 'currency = JPY' 는 버림).
///  · 글자 값(선박 · 항구 · 당사자 · 설명): 질의에 그대로 나오면 힌트로, 아니면 버립니다. 'none' · 'null' 같은 빈 값은 버립니다.
/// v2 질의 R7('6월에 펠릭스토우에서 선적된 서류')은 en `vessel = none` · `etd > 2019-06-30`, it `recipient_name = giugno`,
/// ja · zh `currency = JPY` · `freight_amount ≤ 6`, cs `pod = Felixstowe` 같은 조건이 걸려 정답이 통째로 걸러졌습니다.
pub fn ground_trade_conditions<E: Fn(&str)>(structured: &mut Value, query: &str, emit: &E) {
    use crate::utils::ai_utils::{detect_field_format, FieldFormat};
    let q_lower = query.to_lowercase();
    let q_key = crate::scheduler::normalize_entity_key(query);
    // 식별자 토큰 밖에서, 날짜 단위(年 · 月 · 日 · 년 · 월 · 일)가 붙지 않은 숫자
    let mut free = q_lower.clone();
    for (raw, _) in crate::search_relation::query_id_tokens(query) {
        // 글자가 섞인 번호(LC-DE7472990)와 긴 숫자 식별자만 지웁니다. '5000' 같은 수는 남깁니다.
        if raw.chars().any(|c| c.is_ascii_alphabetic()) || crate::utils::ai_utils::record_id_token(&raw).is_some() {
            free = free.replace(&raw.to_lowercase(), " ");
        }
    }
    let chars: Vec<char> = free.chars().collect();
    let mut numbers: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let s = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',') {
                i += 1;
            }
            let unit = chars.get(i).copied().unwrap_or(' ');
            let dated = matches!(unit, '年' | '月' | '日' | '년' | '월' | '일' | '号' | '號');
            let run: String = chars[s..i].iter().filter(|c| **c != ',').collect();
            if !dated {
                if let Ok(v) = run.trim_end_matches('.').parse::<f64>() {
                    numbers.push(v);
                }
            }
        } else {
            i += 1;
        }
    }
    let this_year = chrono::Datelike::year(&crate::utils::time_guide::now_local()).to_string();
    let null_like = |v: &str| matches!(v.trim().to_lowercase().as_str(), "" | "none" | "null" | "n/a" | "na" | "nil" | "unknown" | "-" | "없음" | "なし" | "无");
    let Some(ctxs) = structured.get_mut("context").and_then(|c| c.as_array_mut()) else { return };
    for ctx in ctxs.iter_mut() {
        let Some(o) = ctx.as_object_mut() else { continue };
        let conds: Vec<(String, Value)> = match o.get("condition").and_then(|v| v.as_object()) {
            Some(c) => c.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => continue,
        };
        let mut dropped: Vec<String> = Vec::new();
        let mut hinted: Vec<(String, Value)> = Vec::new();
        for (field, spec) in conds.iter() {
            let raw = match spec.get("value") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Number(n)) => n.to_string(),
                _ => continue,
            };
            if null_like(&raw) {
                dropped.push(format!("{} = {}", field, raw));
                continue;
            }
            let ident = matches!(field.as_str(), "doc_number" | "no" | "hub_reference") || field.starts_with("reference_");
            let fmt = detect_field_format(field);
            let keep = if ident || matches!(fmt, FieldFormat::Identifier | FieldFormat::TrackingCode) {
                let k = crate::scheduler::normalize_entity_key(&raw);
                k.chars().count() >= 3 && q_key.contains(&k)
            } else if field == "status" || field.ends_with("_status") {
                true
            } else if field == "currency" || field.ends_with("_currency") {
                let code = raw.trim().to_uppercase();
                q_lower.contains(&code.to_lowercase())
                    || q_lower.split_whitespace().any(|w| {
                        crate::utils::ai_utils::currency_code_of(w) == Some(code.as_str())
                            || crate::utils::ai_utils::currency_symbol_in(w) == Some(code.as_str())
                            || crate::utils::ai_utils::currency_name_exact(w) == Some(code.as_str())
                    })
            } else {
                match fmt {
                    FieldFormat::Date => {
                        let year: String = raw.chars().take(4).collect();
                        !(year.len() == 4 && year.chars().all(|c| c.is_ascii_digit())) || year == this_year || q_lower.contains(&year)
                    }
                    FieldFormat::Numeric => raw.trim().parse::<f64>().map_or(false, |v| numbers.iter().any(|n| (n - v).abs() < 1e-9)),
                    FieldFormat::Text | FieldFormat::Address => {
                        if q_lower.contains(&raw.trim().to_lowercase()) {
                            hinted.push((field.clone(), spec.clone()));
                        } else {
                            dropped.push(format!("{} = {}", field, raw));
                        }
                        continue;
                    }
                    _ => true,
                }
            };
            if !keep {
                dropped.push(format!("{} {} {}", field, spec.get("operator").and_then(|v| v.as_str()).unwrap_or(""), raw));
            }
        }
        if dropped.is_empty() && hinted.is_empty() {
            continue;
        }
        if let Some(c) = o.get_mut("condition").and_then(|v| v.as_object_mut()) {
            for d in dropped.iter() {
                let key = d.split_whitespace().next().unwrap_or("").to_string();
                c.remove(&key);
            }
            for (k, _) in hinted.iter() {
                c.remove(k);
            }
        }
        if !hinted.is_empty() {
            let hint = o.entry("hint".to_string()).or_insert_with(|| json!({}));
            if !hint.is_object() {
                *hint = json!({});
            }
            if let Some(h) = hint.as_object_mut() {
                for (k, spec) in hinted.iter() {
                    h.entry(k.clone()).or_insert_with(|| spec.clone());
                }
            }
        }
        emit(&format!(
            "[AI-SEARCH] ⚓ [TRADE GROUNDING] 질의에 근거가 없는 서식 조건을 버리고 {:?}, 글자 값 조건은 힌트로 내립니다 {:?}",
            dropped,
            hinted.iter().map(|(k, s)| format!("{} = {}", k, logis_schema::json_order::Sorted(s.get("value").unwrap_or(&Value::Null)))).collect::<Vec<_>>()
        ));
    }
}

/// 유형별 지시어: 언어별 유형 이름에 bias.json 의 search_bridge.domain_words 를 더합니다. 상품은 기본값이라 넣지 않습니다.
pub fn domain_vocabulary(language: &str) -> Vec<(String, Vec<String>)> {
    ["order", "tracking", "review", "coupon", "event"]
        .iter()
        .map(|cat| {
            let mut terms = vec![cat.to_string(), crate::parsing::get_localized_page_type(cat, language)];
            if let Some(list) = crate::parsing::BIAS_DICT
                .get("search_bridge")
                .and_then(|b| b.get("domain_words"))
                .and_then(|d| d.get(*cat))
                .and_then(|v| v.as_array())
            {
                terms.extend(list.iter().filter_map(|t| t.as_str().map(str::to_string)));
            }
            (cat.to_string(), terms)
        })
        .collect()
}

pub fn commerce_scope(cc: &str) -> String {
    if cc.trim().is_empty() {
        "mode = 'commerce'".to_string()
    } else {
        format!("mode = 'commerce' AND `cc` = '{}'", cc.replace('\'', "''"))
    }
}

/// 🌟 [WHOLE QUERY FUSION] 질의 전체의 벡터 · 청크 · 글자 일치 순위를 해석 결과의 순위와 합칩니다.
pub async fn fuse_whole_query<E: Fn(&str)>(
    store: &VectorStore,
    model: &LogisModel,
    query: &str,
    language: &str,
    cc: &str,
    ranked: &mut Vec<Value>,
    emit: &E,
) -> Value {
    use crate::search_fusion as sf;
    const TRACK_LIMIT: usize = 200;
    const CHUNK_LIMIT: usize = 50;

    let scope = commerce_scope(cc);
    let emb = model.get_embedding(query.to_string()).await.unwrap_or_default();
    let vector_rows = store.rank_items_vector(query, &emb, TRACK_LIMIT, &scope).await.unwrap_or_default();
    let fts_rows = store.rank_items_fts(query, TRACK_LIMIT, &scope).await.unwrap_or_default();
    let chunk_filter = if cc.trim().is_empty() {
        "mode = 'commerce'".to_string()
    } else {
        format!("mode = 'commerce' AND cc = '{}'", cc.replace('\'', "''"))
    };
    let chunk_rows = store.search_chunks(&emb, CHUNK_LIMIT, Some(&chunk_filter)).await.unwrap_or_default();
    let domains = sf::query_domains(query, &domain_vocabulary(language));
    let type_list = domains.iter().map(|d| format!("'{}'", d.replace('\'', "''"))).collect::<Vec<_>>().join(", ");
    let typed_rows = store
        .rank_items_vector(query, &emb, TRACK_LIMIT, &format!("{} AND type IN ({})", scope, type_list))
        .await
        .unwrap_or_default();
    let typed_chunks = store
        .search_chunks(&emb, CHUNK_LIMIT, Some(&format!("{} AND item_type IN ({})", chunk_filter, type_list)))
        .await
        .unwrap_or_default();

    let mut data_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut pool: Vec<(String, String)> = Vec::new();
    for (id, data) in vector_rows.iter().chain(typed_rows.iter()).chain(fts_rows.iter()) {
        if data_of.contains_key(id) { continue; }
        data_of.insert(id.clone(), data.clone());
        pool.push((id.clone(), sf::value_text(data)));
    }

    let pipeline: Vec<String> = ranked.iter().filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(str::to_string)).collect();
    let vector: Vec<String> = vector_rows.iter().map(|(id, _)| id.clone()).collect();
    let chunks: Vec<String> = chunk_rows.iter().map(|c| c.1.clone()).collect();
    let terms = sf::query_terms(query);
    let literal = sf::literal_ranking(&terms, &pool);
    let typed_vector: Vec<String> = typed_rows.iter().map(|(id, _)| id.clone()).collect();
    let typed_chunk_ids: Vec<String> = typed_chunks.iter().map(|c| c.1.clone()).collect();
    let typed_literal: Vec<String> = literal
        .iter()
        .filter(|id| {
            data_of
                .get(*id)
                .and_then(|d| serde_json::from_str::<Value>(d).map(logis_schema::json_order::sorted).ok())
                .and_then(|d| d.get("type").and_then(|t| t.as_str()).map(|t| domains.iter().any(|x| x == t)))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    let fused = sf::fuse(
        &[
            (pipeline.as_slice(), 1.0),
            (vector.as_slice(), 1.0),
            (chunks.as_slice(), 1.0),
            (literal.as_slice(), sf::LITERAL_WEIGHT),
            (typed_vector.as_slice(), 1.0),
            (typed_chunk_ids.as_slice(), 1.0),
            (typed_literal.as_slice(), 1.0),
        ],
        sf::FUSION_K,
    );

    let top = fused.first().map_or(1.0, |f| f.1.max(f32::MIN_POSITIVE));
    let mut known: std::collections::HashMap<String, Value> = ranked
        .drain(..)
        .filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(str::to_string).map(|id| (id, r)))
        .collect();
    let added = fused.iter().filter(|(id, _)| !known.contains_key(id)).count();
    for (id, s) in fused.iter() {
        let mut row = match known.remove(id) {
            Some(r) => r,
            None => {
                let data = data_of.get(id).cloned().unwrap_or_default();
                let doc_type = serde_json::from_str::<Value>(&data).map(logis_schema::json_order::sorted)
                    .ok()
                    .and_then(|d| d.get("type").and_then(|t| t.as_str()).map(str::to_string))
                    .unwrap_or_default();
                json!({ "id": id, "text": data, "context_type": doc_type, "relation": "whole_query" })
            }
        };
        if let Some(obj) = row.as_object_mut() {
            obj.insert("score".to_string(), json!(*s / top));
        }
        ranked.push(row);
    }
    crate::utils::score_dynamics::record_baseline("search.fusion_added", added as f32);
    emit(&format!(
        "[AI-SEARCH] 🧮 [WHOLE QUERY FUSION] 질의 유형 {:?} | 해석 {}건 · 질의 벡터 {}건 · 청크 {}건 · 글자 일치 {}건을 합쳐 {}건 (새로 들어온 문서 {}건)",
        domains, pipeline.len(), vector.len(), chunks.len(), literal.len(), ranked.len(), added
    ));
    json!({
        "pipeline": pipeline, "vector": vector, "chunks": chunks, "literal": literal,
        "domains": domains, "typed_vector": typed_vector, "typed_chunks": typed_chunk_ids
    })
}

pub static VISION_QUERY_MEMO: Lazy<std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>> =
    Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// 자연어 검색 (예전 셸의 `ai_search_complex` 명령 본문). 셸 명령은 `&AppState` 와 `TauriSink` 를 넘기고, 헤드리스는 `sink::noop()` 등을 넘깁니다.
/// `app_handle` 은 진행 이벤트를 받는 싱크입니다 (이름은 예전 그대로).
pub async fn ai_search_complex(
    state: &AppState,
    app_handle: crate::sink::Sink,
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
            data_json: logis_schema::json_order::to_string_sorted(&json!({"query": query.clone(), "mode": search_mode.clone()})),
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
    crate::utils::logger::log_task_progress(&app_handle.clone(), &task_id, &payload_start);

    
    let search_process = async {
        let mut all_results = Vec::new();
        // 🌟 [DEXIE PLAN] 컨텍스트별 정밀 필터 플랜을 모아 프론트엔드로 전달합니다.
        //    LanceDB 가 버렸던 조건이 여기에 전부 살아 있습니다.
        let mut dexie_plans: Vec<Value> = Vec::new();
        // 🌟 [VISION QUERY] shipping(무역) 트랙 전용 비전 질의 벡터. 컨텍스트 전체가 공유합니다.
        let mut vision_qvec: Option<Vec<f32>> = None;
        let mut fusion_trace = Value::Null;

        let team_id = crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000"); 
        let mut metrics_json_str = "{}".to_string();
        
        if let Some(store) = store_opt.as_ref() {
            if let Ok(Some(doc)) = store.get_item_by_id("users", &team_id).await {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&doc.json_data).map(logis_schema::json_order::sorted) {
                    if let Some(base) = val.get("base") {
                        metrics_json_str = logis_schema::json_order::to_string_sorted(&base);
                    }
                }
            }
        }

        
        let search_mode: String = if search_mode == "commerce" {
            crate::utils::score_dynamics::enter_scope(
                "",
                crate::utils::score_dynamics::Track::Search,
                "probe",
                "",
            );
            let verdict = probe_trade_query(&model, &query, &emit_term).await;
            crate::utils::score_dynamics::record_baseline(
                "search.query_reroute",
                if verdict.is_some() { 1.0 } else { 0.0 },
            );
            crate::utils::score_dynamics::leave_scope();
            // 🌟 [REROUTE DATA GUARD] 무역 문서가 하나도 없으면 전환해도 0건입니다. 코사인 초과분 +0.1 로 전환된
            //    'gadgets for listening to music' 이 빈 결과가 되었습니다.
            let verdict = match (verdict, store_opt.as_ref()) {
                (Some(code), Some(store)) if !store.has_rows("mode = 'shipping'").await => {
                    emit_term(&format!(
                        "[MODE REROUTE / QUERY] ⚪ 서식 '{}' 로 전환할 수 있지만 저장된 무역 문서가 없어 커머스 범위를 유지합니다.",
                        code
                    ));
                    None
                }
                (v, _) => v,
            };
            match verdict {
                Some(code) => {
                    emit_term(&format!(
                        "[MODE REROUTE / QUERY] 🔀 mode='commerce' 요청이지만 질의가 무역 서식 '{}' 을 지목합니다. 서식 이미지는 비전 경로의 MODE REROUTE 로 mode='shipping' 에 저장되므로 커머스 스코프로는 구조적으로 0건입니다. 질의 해석·비전 트랙·스코프·Dexie 플랜을 모두 shipping 으로 전환합니다.",
                        code
                    ));
                    "shipping".to_string()
                }
                None => search_mode.clone(),
            }
        } else {
            search_mode.clone()
        };

        // 🌟 [VOCAB TRIM] 텍스트 맥락: 질의 글의 언어(현지 언어 + 영어)와 질의에 나온 문자 체계(데이터 언어로 적힌 고객 · 수출자 이름 등)의
        //    행만 출력 투영에 남깁니다. 커머스 · 무역 · 행동 로그 모두, 검색이 끝나면 되돌립니다.
        let _vocab_scope = crate::models::common::vocab_trim::DocLangScope::text(&crate::utils::lang_utils::detect_document_language(&query), &query);
        // 답변 · 질의 해석의 "Write in {LANG}" 은 앱 언어입니다 (질의 언어와 다를 수 있음)
        _vocab_scope.also(&language);
        let mut structured_query = match search_mode.as_str() {
            "shipping" => {
                model.parse_shipping_query(&task_id, &app_handle.clone(), query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            _ => { // default: commerce
                model.parse_commerce_query(&task_id, &app_handle.clone(), query.clone(), &language, &metrics_json_str, cancel_token.clone()).await.map_err(|e| e.to_string())?
            }
        };

        if search_mode == "commerce" {
            if let Some(store) = store_opt.as_ref() {
                let domains = crate::search_fusion::query_domains(&query, &domain_vocabulary(&language));
                ground_conditions(store, &mut structured_query, &cc, &domains, &emit_term).await;
            }
        }
        if search_mode == "shipping" {
            ground_trade_conditions(&mut structured_query, &query, &emit_term);
        }

        // =====================================================================
        // 🌟 [VISION QUERY TRACK] shipping 모드에서 SigLIP2 텍스트 인코더로
        //    질의를 1152차원 공유공간에 올려 비전 벡터 검색을 활성화합니다.
        // ---------------------------------------------------------------------
        //  ── 무엇이 문제였나 ──
        //   구버전은 컨텍스트 루프 안에서 `model.siglip2_model` 을 들여다봤지만,
        //   검색 경로 어디에도 ensure_siglip2() 호출이 없었습니다.
        //   parse_shipping_query 는 granite(384) 만 올립니다.
        //   그래서 siglip2_model 은 항상 None → vision_qvec 은 항상 None →
        //   Track V 가 단 한 번도 발화하지 않았습니다.
        //   (이미지로 추출한 문서의 vision_vec 이 저장만 되고 검색에는 쓰이지 않는 상태)
        //
        //  ── 왜 루프 밖에서 1회만 만드는가 ──
        //   ① 비전 벡터는 '이미지 전체' 를 대표하는 pooled 벡터입니다.
        //      컨텍스트별로 쪼갠 텍스트보다 원 질의 문장이 더 적합합니다.
        //   ② SigLIP2(2.2GB)를 컨텍스트마다 올렸다 내리면 4GB GPU 에서 즉시 OOM 입니다.
        //      한 번 올려 인코딩하고 즉시 반환합니다.
        //
        //  ── VRAM 순서 ──
        //   parse_shipping_query 가 남긴 Qwen3.5(2GB) 를 먼저 내리고,
        //   SigLIP2 를 올려 인코딩한 뒤 즉시 반환합니다.
        //   그 다음 컨텍스트 루프가 granite(384) 를 올립니다. 동시 상주가 없습니다.
        // =====================================================================
        if search_mode == "shipping" && !query.trim().is_empty() {
            if cancel_token.load(Ordering::Relaxed) {
                return Err("Search cancelled by user".to_string());
            }
            let qkey = query.trim().to_string();
            let remembered = VISION_QUERY_MEMO
                .lock()
                .ok()
                .and_then(|m| m.get(&qkey).cloned())
                .or_else(|| {
                    crate::models::siglip2::phrase_cache::SIGLIP2_PHRASE_CACHE
                        .get(&qkey)
                        .map(|v| (*v).clone())
                });
            if let Some(v) = remembered {
                emit_term("[VISION TRACK] ⚡ 같은 질의의 비전 질의 벡터가 이미 있어 SigLIP2 를 올리지 않습니다. (퍼지·로드·해제·VRAM 대기 0회)");
                vision_qvec = Some(v);
            } else {
                emit_term("[VISION TRACK] 🖼️ SigLIP2 텍스트 인코더만 올려 비전 질의 벡터를 생성합니다. 질의 인코딩에는 비전 인코더가 쓰이지 않아 올리지 않습니다.");
                match model.check_siglip2_downloaded().await {
                    Err(e) => {
                        emit_term(&format!(
                            "[VISION TRACK] ⚪ SigLIP2 미설치로 비전 검색을 건너뜁니다: {}", e
                        ));
                    }
                    Ok(_) => {
                        model.unload_generation_slots("vision query text encoding").await;
                        match model.ensure_siglip2_ext(false, true).await {
                            Err(e) => {
                                emit_term(&format!(
                                    "[VISION TRACK] ⚪ SigLIP2 로드 실패로 비전 검색을 건너뜁니다: {}", e
                                ));
                            }
                            Ok(_) => {
                                {
                                    let guard = model.siglip2_model.lock().await;
                                    if let Some(sig) = guard.as_ref() {
                                        if sig.has_text() {
                                            vision_qvec = crate::models::siglip2::vision_encoder::encode_phrases_ephemeral(
                                                sig,
                                                &[qkey.clone()],
                                            )
                                            .ok()
                                            .and_then(|v| v.into_iter().next());
                                        }
                                    }
                                }
                                model.release_siglip2("search vision query encoded").await;
                            }
                        }
                    }
                }
                if let (Some(v), Ok(mut m)) = (vision_qvec.as_ref(), VISION_QUERY_MEMO.lock()) {
                    if m.len() >= 256 {
                        m.clear();
                    }
                    m.insert(qkey.clone(), v.clone());
                }
            }

            match vision_qvec.as_ref() {
                Some(v) => emit_term(&format!(
                    "[VISION TRACK] ✅ 비전 질의 벡터 생성 완료 (dim {}). LanceDB Track V 를 활성화합니다.",
                    v.len()
                )),
                None => emit_term("[VISION TRACK] ⚪ 비전 질의 벡터를 만들지 못해 텍스트 트랙만 사용합니다."),
            }
        }

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
                        for (_, v) in logis_schema::json_order::sorted_iter(cond) {
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

                let target_table = match ctx_type {
                    "member" | "team" | "user" => "users",
                    "page" | "pages" => "pages",
                    _ => "items",
                };

                let mut scope_ctx = ctx.clone();
                if let Some(o) = scope_ctx.as_object_mut() {
                    // 폴백 접미사가 붙은 type 이 SQL 에 그대로 나가면 `type = 'review_items'` 가 되어
                    // 항상 0건이 됩니다. 정규화된 이름으로 교체합니다.
                    o.insert("type".to_string(), json!(ctx_type));

                    if !cc.trim().is_empty() && search_mode != "shipping" {
                        o.insert("cc".to_string(), json!(cc.clone()));
                    }
                }
                let scope_filter = sanitize_scope_filter(build_scope_filter(&scope_ctx, &search_mode));
                let dexie_plan = build_dexie_plan(&scope_ctx, &search_mode);

                println!(
                    "[AI-SEARCH] 🧭 [ROUTE] type='{}' | table='{}' | scope_sql={:?} | dexie_conditions={}",
                    ctx_type,
                    target_table,
                    scope_filter,
                    dexie_plan.get("conditions").and_then(|c| c.as_array()).map(|a| a.len()).unwrap_or(0)
                );

                // 🌟 [PLAN 수집] 프론트엔드가 실행할 정밀 필터 플랜을 응답에 동봉합니다.
                if ctx.get("plan_off").and_then(|v| v.as_bool()) != Some(true) {
                    dexie_plans.push(dexie_plan.clone());
                }

                // 🌟 [TRACKING] tracking_number 는 이제 LIKE 땜질이 아니라
                //    dexie_plan 의 data.tracking_number eq 조건으로 정확히 처리됩니다.
                //    여기서는 FTS 를 끄지 않습니다. 리콜을 넓게 가져가는 것이 v4 의 역할이기 때문입니다.
                let emb = model.get_embedding(search_text.clone()).await.unwrap_or(vec![0.0; 384]);

                // 🌟 [RECALL LIMIT] Dexie 가 뒤에서 조건으로 잘라내므로 여기서는 넓게 긁습니다.
                //    기존 limit 5 는 '조건이 SQL 에 이미 적용되었다' 는 전제였는데,
                //    v4 에서는 조건이 SQL 에 없으므로 5건만 가져오면 정답이 잘려 나갑니다.
                let recall_limit: usize = 50;

                // 🌟 [비전 트랙] 루프 밖에서 1회 생성한 질의 벡터를 전 컨텍스트가 공유합니다.
                //    (구버전은 여기서 매번 siglip2_model 을 들여다봤지만 로드 코드가 없어
                //     항상 None 이었습니다. 자세한 경위는 위 [VISION QUERY TRACK] 주석 참조)
                let final_results = store
                    .search_items(target_table, &search_text, emb.clone(), vision_qvec.clone(), recall_limit, 0, scope_filter.clone(), true)
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
                    // 🌟 shipping 은 위와 동일한 이유로 cc 를 걸지 않습니다.
                    let mode_only = if cc.trim().is_empty() || search_mode == "shipping" {
                        format!("mode = '{}'", search_mode)
                    } else {
                        format!("mode = '{}' AND `cc` = '{}'", search_mode, cc.replace('\'', "''"))
                    };
                    println!("[AI-SEARCH] 🛟 [SCOPE FALLBACK] 0 hit with full scope. Retrying with '{}'.", mode_only);
                    store
                        .search_items(target_table, &search_text, emb.clone(), vision_qvec.clone(), recall_limit, 0, Some(mode_only), true)
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
                    // 🌟 [CHUNK TENANT SCOPE] 문서 트랙과 같은 사이트 범위를 청크 트랙에도 겁니다. 없으면 다른 쇼핑몰의 청크가
                    //    그대로 결과에 들어오고, 조건 없는 질의는 프론트엔드 필터도 지나갑니다. shipping 은 문서 트랙과 같이 걸지 않습니다.
                    let chunk_type_filter = if cc.trim().is_empty() || search_mode == "shipping" {
                        chunk_type_filter
                    } else {
                        format!("{} AND cc = '{}'", chunk_type_filter, cc.replace('\'', "''"))
                    };

                    // 4-B: PLINKO 조건 기반 타겟 속성 목록 구축
                    //      ① condition 키 (PLINKO 가 확정한 속성)
                    //      ② substantial (추상 수식어가 지목한 물리 속성 — weight / sale_price ...)
                    //      substantial 은 LanceDB 물리 컬럼이 아니라 SQL 로는 절대 반영되지 않지만,
                    //      item_chunks 의 property 컬럼에는 그대로 존재하므로
                    //      '무거운 → weight' 의도를 여기서 실제 검색으로 회수합니다.
                    let hard_props: Vec<String> = ctx.get("condition")
                        .and_then(|v| v.as_object())
                        .map(|obj| obj.keys().cloned().collect())
                        .unwrap_or_default();
                    let mut hint_props: Vec<String> = Vec::new();
                    let mut condition_props: Vec<String> = hard_props.clone();
                    if let Some(hint_obj) = ctx.get("hint").and_then(|v| v.as_object()) {
                        for k in logis_schema::json_order::sorted_keys(hint_obj) {
                            if hard_props.iter().any(|p| p == k) { continue; }
                            if !hint_props.iter().any(|p| p == k) { hint_props.push(k.clone()); }
                            if !condition_props.iter().any(|p| p == k) {
                                condition_props.push(k.clone());
                            }
                        }
                    }
                    if !hint_props.is_empty() {
                        println!(
                            "[AI-SEARCH]   ⚖️ [TRACK WEIGHT] 하드 조건 축 {:?} 는 Column 트랙 전액(×3.0), 힌트 축 {:?} 는 절반(×1.5)을 받습니다. 하드는 질의가 '이 축을 걸겠다' 고 확정한 것이고 힌트는 '그 축일 수도 있다' 는 추정이라, 같은 눈금으로 더하면 추정 축이 랭킹을 주도합니다. 타겟 청크 검색(STAGE-4C)은 두 축 모두에 그대로 수행해 리콜은 잃지 않습니다.",
                            hard_props, hint_props
                        );
                    }

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
                        if hard_props.contains(&property) {
                            let column_track = raw_cosine * 3.0;
                            score += column_track;
                            println!("[AI-SEARCH]   🎯 [STAGE-4B / HARD] property='{}' 확정 조건 매칭 → Column 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}", property, column_track, raw_cosine, score);
                        } else if hint_props.contains(&property) {
                            let column_track = raw_cosine * 1.5;
                            score += column_track;
                            println!("[AI-SEARCH]   🎯 [STAGE-4B / HINT] property='{}' 힌트 축 매칭 → Column 트랙 절반 +{:.4} (cos {:.4}) → 최종 {:.4}", property, column_track, raw_cosine, score);
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
                                let qs = dominant_script(&query);
                                let chunk_value = crate::nl_convert::extract_value_from_chunk(&chunk_text);
                                let cs = dominant_script(if chunk_value.trim().is_empty() { &chunk_text } else { &chunk_value });
                                let script_gap = qs != "none" && cs != "none" && qs != cs;
                                if script_gap {
                                    let cross_lingual_track = raw_cosine * 2.0;
                                    score += cross_lingual_track;
                                    crate::utils::score_dynamics::record_baseline(
                                        "search.crosslingual_cos",
                                        raw_cosine,
                                    );
                                    println!(
                                        "[AI-SEARCH]   🌐 [STAGE-4X] property='{}' | 질의 문자 체계 '{}' ≠ 청크 '{}' → FTS 가 물리적으로 0건인 구간이므로 그 트랙(×2.0)을 코사인이 대신합니다. +{:.4} (cos {:.4}) → 최종 {:.4}",
                                        property, qs, cs, cross_lingual_track, raw_cosine, score
                                    );
                                } else {
                                    crate::utils::score_dynamics::record_baseline(
                                        "search.samelingual_cos",
                                        raw_cosine,
                                    );
                                    println!(
                                        "[AI-SEARCH]   ⚪ [STAGE-4X SKIP] property='{}' | 질의와 청크가 같은 문자 체계 '{}' 라 FTS 트랙이 이미 이 값을 평가했습니다. 같은 증거를 두 번 더하지 않습니다. (cos {:.4})",
                                        property, qs, raw_cosine
                                    );
                                }
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

                        let wanted_value: Option<String> = ctx
                            .get("condition")
                            .and_then(|v| v.get(property.as_str()))
                            .or_else(|| ctx.get("hint").and_then(|v| v.get(property.as_str())))
                            .and_then(|spec| spec.get("value"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty());
                        if let Some(want) = wanted_value {
                            let chunk_value = crate::nl_convert::extract_value_from_chunk(&chunk_text);
                            let have = if chunk_value.trim().is_empty() { chunk_text.clone() } else { chunk_value };
                            if let Some((how, sim)) = cross_script_value_match(&want, &have) {
                                let value_track = if how == "phonetic" { sim } else { 1.0 };
                                score += value_track;
                                crate::utils::score_dynamics::record_baseline("search.value_match_cross_script", sim);
                                println!(
                                    "[AI-SEARCH]   🧭 [STAGE-4Z / VALUE MATCH] property='{}' | 질의 값 '{}' ↔ 청크 값 '{}' | 문자 체계가 달라 FTS 가 0건인 자리에서 {} 근거로 값이 일치합니다 ({:.2}) → 값 일치 트랙 +{:.4} → 최종 {:.4}. 코사인 {:.4} 만으로는 '같은 값' 과 '비슷한 값' 을 가르지 못하므로, 정규 코드 일치는 +1.0, 발음 골격 일치는 유사도만큼 더합니다.",
                                    property,
                                    want,
                                    have,
                                    match how { "country" => "국가 정규 코드", "currency" => "통화 정규 코드", _ => "발음 골격" },
                                    sim,
                                    value_track,
                                    score,
                                    raw_cosine
                                );
                            }
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

            // ── 5-D2: 질의 전체 트랙과 합칩니다 (search_fusion 모듈 설명 참조) ──
            if search_mode == "commerce" {
                if let Some(store) = store_opt.as_ref() {
                    fusion_trace = fuse_whole_query(store, &model, &query, &language, &cc, &mut ranked_results, &emit_term).await;
                }
            }

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

            if let Some(store) = store_opt.as_ref() {
                let relay_rep = crate::scheduler::relay_ledger::relay_join(
                    store,
                    &mut ranked_results,
                    &mut dexie_plans,
                    &search_mode,
                    &cc,
                    &team_id,
                    &emit_term,
                ).await;
                if relay_rep.changed_set() {
                    ranked_results.sort_by(|a, b| {
                        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                // 🌟 [RELATION] 고정점(식별자 · 사람 이름 · 제목 · 상태 · 서식 번호)에서 질의가 묻는 유형으로 건너간 문서를 맨 앞으로
                //    (search_relation 설명 참조). 기간 조인이 적용된 질의는 건드리지 않습니다.
                if search_mode == "commerce" {
                    crate::search_relation::relation_join(store, &model, &mut ranked_results, &mut dexie_plans, &query, &language, &cc, &emit_term).await;
                } else if search_mode == "shipping" {
                    crate::search_relation::trade_relation_join(store, &mut ranked_results, &mut dexie_plans, &query, &language, &emit_term).await;
                }
            }

            let projection_axes: Vec<String> = {
                let mut v: Vec<String> = Vec::new();
                for plan in dexie_plans.iter() {
                    if let Some(arr) = plan.get("projection").and_then(|p| p.as_array()) {
                        for a in arr.iter() {
                            if let Some(s) = a.as_str() {
                                let t = s.trim();
                                if !t.is_empty() && !v.iter().any(|e| e == t) {
                                    v.push(t.to_string());
                                }
                            }
                        }
                    }
                }
                v
            };

            if !projection_axes.is_empty() {
                match store_opt.as_ref() {
                    None => println!(
                        "[AI-SEARCH] ⚠️ [PROJECTION SKIP] 표시 축 {:?} 를 요청받았지만 Store 가 없어 값을 실을 수 없습니다.",
                        projection_axes
                    ),
                    Some(store) => {
                        let mut attached = 0usize;
                        let mut unresolved = 0usize;
                        for item in ranked_results.iter_mut() {
                            let id = item
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if id.is_empty() {
                                unresolved += 1;
                                continue;
                            }
                            let data: Value = match store.get_item_by_id("items", &id).await {
                                Ok(Some(d)) => match serde_json::from_str(&d.json_data) {
                                    Ok(v) => v,
                                    Err(_) => {
                                        unresolved += 1;
                                        continue;
                                    }
                                },
                                _ => {
                                    unresolved += 1;
                                    continue;
                                }
                            };
                            let mut picked = serde_json::Map::new();
                            for axis in projection_axes.iter() {
                                let hit = data.get(axis.as_str()).cloned().or_else(|| {
                                    data.as_object().and_then(|o| {
                                        logis_schema::json_order::sorted_values(o)
                                            .filter_map(|v| v.as_object())
                                            .find_map(|inner| inner.get(axis.as_str()).cloned())
                                    })
                                });
                                if let Some(val) = hit {
                                    let empty = val.is_null()
                                        || val
                                            .as_str()
                                            .map(|s| s.trim().is_empty())
                                            .unwrap_or(false);
                                    if !empty {
                                        picked.insert(axis.clone(), val);
                                    }
                                }
                            }
                            if picked.is_empty() {
                                continue;
                            }
                            attached += 1;
                            item.as_object_mut()
                                .unwrap()
                                .insert("projection".to_string(), Value::Object(picked));
                        }
                        println!(
                            "[AI-SEARCH] 🧷 [PROJECTION ATTACH] 질의가 '함께 보여달라' 고 지목한 축 {:?} 를 결과 {}건에 실었습니다 (값을 찾지 못한 행 {}건). 이 축은 회수 조건이 아니므로 결과 집합을 좁히지 않고 값만 덧붙입니다. 지금까지는 plan 안에 담겨 운반만 되고 아무도 읽지 않았습니다.",
                            projection_axes, attached, unresolved
                        );
                    }
                }
            }

            // ── 검색 통계 로그 출력 ──
            let chunk_match_count = ranked_results.iter()
                .filter(|r| r.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false))
                .count();
            let primary_count = ranked_results.len() - chunk_match_count;

            println!("[AI-SEARCH] 📊 [STAGE-5] 결과 병합 완료: 총 {}건 (문서 매칭 {}건 + 청크 매칭 {}건) | 비전 트랙: {}",
                ranked_results.len(), primary_count, chunk_match_count,
                if vision_qvec.is_some() { "활성" } else { "비활성" });

            for (rank, item) in ranked_results.iter().enumerate() {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let score = item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let prop = item.get("matched_property").and_then(|v| v.as_str()).unwrap_or("-");
                let is_chunk = item.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let is_alias = item.get("alias_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let ctx = item
                    .get("doc_type")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .or_else(|| item.get("context_type").and_then(|v| v.as_str()))
                    .unwrap_or("?");
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

            if !dexie_plans.is_empty() {
                let has_hard = dexie_plans.iter().any(|p| {
                    p.get("conditions")
                        .and_then(|c| c.as_array())
                        .map_or(false, |a| !a.is_empty())
                });
                let mut docs: Vec<Value> = Vec::new();
                let mut hydrated = 0usize;
                let mut unresolved = 0usize;
                for r in ranked_results.iter() {
                    let inline = match r.get("text") {
                        Some(Value::String(s)) => serde_json::from_str::<Value>(s).map(logis_schema::json_order::sorted).ok(),
                        Some(v) if v.is_object() => Some(v.clone()),
                        _ => None,
                    };
                    if let Some(d) = inline {
                        docs.push(d);
                        continue;
                    }
                    if !has_hard { continue; }
                    let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if id.is_empty() {
                        unresolved += 1;
                        continue;
                    }
                    let fetched = match store_opt.as_ref() {
                        Some(store) => store.get_item_by_id("items", id).await.ok().flatten(),
                        None => None,
                    };
                    match fetched.and_then(|d| serde_json::from_str::<Value>(&d.json_data).map(logis_schema::json_order::sorted).ok()) {
                        Some(d) => {
                            docs.push(d);
                            hydrated += 1;
                        }
                        None => unresolved += 1,
                    }
                }
                if hydrated > 0 || unresolved > 0 {
                    println!(
                        "[AI-SEARCH] 🧬 [DOC HYDRATE] 청크로만 회수된 행은 text 자리에 문장 한 줄이 들어 있어 JSON 파싱이 실패합니다. 저장본에서 {}건을 복원했고 {}건은 복원하지 못했습니다. 이 복원이 없으면 청크가 유일한 회수 경로인 질의에서 리콜 집계가 0건이 되어, 조건이 전부 걸러낸 것인지 애초에 회수가 안 된 것인지를 구분할 수 없습니다.",
                        hydrated, unresolved
                    );
                }
                for plan in dexie_plans.iter_mut() {
                    let types: Vec<String> = plan
                        .get("types")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    let scope_key = crate::utils::score_dynamics::search_scope_key(&types);
                    crate::utils::score_dynamics::enter_scope(
                        "",
                        crate::utils::score_dynamics::Track::Search,
                        &scope_key,
                        "",
                    );
                    crate::utils::score_dynamics::set_run_label(&query);
                    let (eligible, all_pass, per_field) = evaluate_dexie_plan(plan, &docs);
                    let drafts = docs.iter().filter(|d| is_relay_draft(d)).count();
                    crate::utils::score_dynamics::record_baseline("search.recall_n", ranked_results.len() as f32);
                    crate::utils::score_dynamics::record_baseline("search.hydrated_n", docs.len() as f32);
                    crate::utils::score_dynamics::record_baseline("search.eligible_n", eligible as f32);
                    crate::utils::score_dynamics::record_baseline("search.hard_n", per_field.len() as f32);
                    if !docs.is_empty() {
                        crate::utils::score_dynamics::record_baseline("search.draft_share", drafts as f32 / docs.len() as f32);
                    }
                    if eligible > 0 {
                        crate::utils::score_dynamics::record_baseline("search.pass_ratio", all_pass as f32 / eligible as f32);
                    }
                    let mut detail: Vec<String> = Vec::new();
                    for (field, satisfied, sole, blocked) in per_field.iter() {
                        if eligible > 0 {
                            crate::utils::score_dynamics::record_search_outcome(field, *satisfied > 0, *sole > 0);
                        }
                        detail.push(format!(
                            "{}(만족 {} / 단독 차단 {} / 차단 기여 {})",
                            field, satisfied, sole, blocked
                        ));
                    }
                    let hint_eval = evaluate_dexie_hints(plan, &docs);
                    if !hint_eval.is_empty() {
                        let mut hint_detail: Vec<String> = Vec::new();
                        for (field, satisfied, present, cross_script) in hint_eval.iter() {
                            let unobservable = *satisfied == 0 && *present > 0 && *cross_script == *present;
                            if !docs.is_empty() {
                                crate::utils::score_dynamics::record_baseline(
                                    "search.hint_cross_script",
                                    if unobservable { 1.0 } else { 0.0 },
                                );
                                if !unobservable {
                                    crate::utils::score_dynamics::record_search_outcome(field, *satisfied > 0, false);
                                }
                                crate::utils::score_dynamics::record_baseline(
                                    "search.hint_present_ratio",
                                    *present as f32 / docs.len() as f32,
                                );
                            }
                            hint_detail.push(if unobservable {
                                format!(
                                    "{}(평가 보류: 축 보유 {}건 전부가 힌트 값과 문자 체계가 달라 문자열 비교가 성립하지 않음 / 회수 {})",
                                    field, present, docs.len()
                                )
                            } else {
                                format!(
                                    "{}(만족 {} / 축 보유 {} / 회수 {})",
                                    field, satisfied, present, docs.len()
                                )
                            });
                        }
                        println!(
                            "[AI-SEARCH] 📈 [SDS / HINT OUTCOME] 힌트 축 {}개를 결과 집합과 무관하게 관측만 합니다: {} — 힌트는 필터가 아니므로 통과 여부가 결과를 바꾸지 않지만, '그 축을 힌트로 내린 판단이 옳았는가' 는 다음 회차의 강등·치환 판정에 쓰입니다. 지금까지는 evaluated 가 0 이라 이 정보가 원장에 전혀 쌓이지 않았습니다.",
                            hint_eval.len(), hint_detail.join(" | ")
                        );
                    }
                    println!(
                        "[AI-SEARCH] 📈 [SDS / SEARCH OUTCOME] 'search|{}|' | 리콜 {}건 · 서식 범위 안 {}건 중 하드 조건 전부 만족 {}건 | 초안 {}건 | {}",
                        scope_key,
                        docs.len(),
                        eligible,
                        all_pass,
                        drafts,
                        if detail.is_empty() { "하드 조건 없음".to_string() } else { detail.join(" | ") }
                    );
                    if all_pass == 0 && eligible > 0 && !per_field.is_empty() {
                        let corpus = crate::utils::score_dynamics::storage_doc_count(&types);
                        let page_type = types.first().cloned().unwrap_or_default();
                        let storage_axes = collect_storage_axes(&docs);

                        // 🌟 [PRE-SUBSTITUTE SNAPSHOT] 아래 진단은 '치환 이전' 상태로 해야 합니다.
                        //    실측: AXIS SUBSTITUTE 가 amount→weight_net 을 열자 TOO NARROW 이
                        //    "회수 문서에 amount 값이 실제로 들어 있다" 고 보고했습니다.
                        //    저장본에 amount 는 없습니다. 자기가 만든 대안 축을 사실로 착각한 것입니다.
                        let pre_alts = plan_alternate_axis_map(plan);
                        let mut substituted: Vec<String> = Vec::new();
                        {
                            let plan_alts_now = plan_alternate_axis_map(plan);
                            let mut add: Vec<(String, String, f32)> = Vec::new();
                            for (field, _sat, _sole, blocked) in per_field.iter() {
                                if *blocked == 0 { continue; }
                                let mut axes: Vec<String> = vec![field.clone()];
                                if let Some(arr) = plan_alts_now.get(field.as_str()).and_then(|v| v.as_array()) {
                                    for a in arr.iter() {
                                        if let Some(s) = a.as_str() { axes.push(s.to_string()); }
                                    }
                                }
                                let here = docs
                                    .iter()
                                    .filter(|d| {
                                        axes.iter().any(|k| {
                                            d.get(k.as_str()).map_or(false, |v| {
                                                !(v.is_null()
                                                    || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                                            })
                                        })
                                    })
                                    .count();
                                if here > 0 { continue; }
                                let nested = docs
                                    .iter()
                                    .filter(|d| {
                                        axes.iter().any(|k| {
                                            doc_axis_values(d, k).iter().any(|v| {
                                                !(v.is_null()
                                                    || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                                            })
                                        })
                                    })
                                    .count();
                                if nested > 0 {
                                    println!(
                                        "[AI-SEARCH] ⚪ [AXIS SUBSTITUTE SKIP / NESTED] '{}' 는 회수 문서 {}건에 표 행(배열) 안의 값으로 저장되어 있습니다. 축이 없는 것이 아니라 경로가 루트가 아닐 뿐이므로, 루트에 있는 다른 축으로 치환하면 '품목 단가' 조건이 '문서 총액' 같은 다른 사실로 바뀝니다. 치환하지 않고 아래 강등 경로에 맡깁니다.",
                                        field, nested
                                    );
                                    continue;
                                }
                                if let Some((cand, score)) =
                                    nearest_storage_axis(&model, &page_type, field, &storage_axes).await
                                {
                                    add.push((field.clone(), cand, score));
                                }
                            }
                            if !add.is_empty() {
                                if let Some(obj) = plan.as_object_mut() {
                                    let slot = obj.entry("alternates".to_string()).or_insert_with(|| json!({}));
                                    if !slot.is_object() { *slot = json!({}); }
                                    if let Some(m) = slot.as_object_mut() {
                                        for (field, cand, score) in add.iter() {
                                            let entry = m.entry(field.clone()).or_insert_with(|| json!([]));
                                            if !entry.is_array() { *entry = json!([]); }
                                            if let Some(arr) = entry.as_array_mut() {
                                                if !arr.iter().any(|x| x.as_str() == Some(cand.as_str())) {
                                                    arr.push(json!(cand));
                                                }
                                            }
                                            substituted.push(format!("{} → {}({:.4})", field, cand, score));
                                        }
                                    }
                                }
                            }
                        }

                        let (eligible, all_pass, per_field) = if substituted.is_empty() {
                            (eligible, all_pass, per_field)
                        } else {
                            let (e2, p2, f2) = evaluate_dexie_plan(plan, &docs);
                            println!(
                                "[AI-SEARCH] 🔁 [AXIS SUBSTITUTE] 회수 문서에 존재하지 않는 조건 축을, 그 문서가 실제로 가진 축 중 다국어 라벨 코사인이 자기 분포에서 유의하게 앞서고 값 형식이 같은 축으로 열었습니다: {:?}. 강등은 조건을 잃지만 치환은 조건을 유지하므로 치환을 먼저 시도합니다. 하드 조건 전부 만족 {}건 → {}건.",
                                substituted, all_pass, p2
                            );
                            crate::utils::score_dynamics::record_baseline(
                                "search.pass_ratio_after_substitute",
                                if e2 > 0 { p2 as f32 / e2 as f32 } else { 0.0 },
                            );
                            (e2, p2, f2)
                        };

                        if all_pass > 0 {
                            crate::utils::score_dynamics::leave_scope();
                            continue;
                        }

                        let plan_alts = pre_alts;
                        let present_in_recall = |f: &str| -> usize {
                            let mut axes: Vec<String> = vec![f.to_string()];
                            if let Some(arr) = plan_alts.get(f).and_then(|v| v.as_array()) {
                                for a in arr.iter() {
                                    if let Some(s) = a.as_str() { axes.push(s.to_string()); }
                                }
                            }
                            docs.iter()
                                .filter(|d| {
                                    axes.iter().any(|k| {
                                        d.get(k.as_str()).map_or(false, |v| {
                                            !(v.is_null()
                                                || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                                        })
                                    })
                                })
                                .count()
                        };
                        let mut gap: Vec<String> = Vec::new();
                        let mut gap_fields: Vec<String> = Vec::new();
                        let mut narrow: Vec<String> = Vec::new();
                        for (field, _satisfied, _sole, blocked) in per_field.iter() {
                            if *blocked == 0 { continue; }
                            let here = present_in_recall(field);
                            let prior = match crate::utils::score_dynamics::storage_fill_prior(&types, field) {
                                Some((p, filled, prior_docs)) => {
                                    crate::utils::score_dynamics::record_baseline("search.blocker_fill", p);
                                    format!("이력 {}/{}건 · 사전 {:.3}", filled, prior_docs, p)
                                }
                                None => "이력 없음".to_string(),
                            };
                            let line = format!("{}(이번 회수 {}/{}건 · {})", field, here, docs.len(), prior);
                            if here == 0 {
                                gap.push(line);
                                if !gap_fields.iter().any(|g| g == field) {
                                    gap_fields.push(field.clone());
                                }
                            } else {
                                narrow.push(line);
                            }
                        }
                        println!(
                            "[AI-SEARCH] ⚠️ [OVER-FILTER] 서식 범위 안 문서 {}건 중 하드 조건을 전부 만족하는 문서가 없습니다. (같은 서식 저장본 {}건 기준)",
                            eligible, corpus
                        );
                        if !gap.is_empty() {
                            println!(
                                "[AI-SEARCH] 🕳️ [STORAGE GAP] 이번에 회수한 문서 어디에도 이 축이 저장되어 있지 않고, 형식이 맞는 대체 축도 없습니다. 값이 조건을 벗어난 것이 아니라 비교할 값 자체가 없으므로 하드에서 내려야 합니다: {}",
                                gap.join(" | ")
                            );
                            let mut demoted: Vec<String> = Vec::new();
                            let mut carried: Vec<(String, Value)> = Vec::new();
                            if let Some(arr) = plan.get_mut("conditions").and_then(|v| v.as_array_mut()) {
                                let all: Vec<Value> = arr.drain(..).collect();
                                for c in all.into_iter() {
                                    let p = c.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let f = p.strip_prefix("data.").unwrap_or(p.as_str()).to_string();
                                    if gap_fields.iter().any(|g| *g == f) {
                                        let op = c.get("op").and_then(|v| v.as_str()).unwrap_or("eq").to_string();
                                        let val = c.get("value").cloned().unwrap_or(Value::Null);
                                        carried.push((f.clone(), json!({ "operator": op, "value": val })));
                                        if !demoted.iter().any(|d| *d == f) { demoted.push(f); }
                                    } else {
                                        arr.push(c);
                                    }
                                }
                            }
                            if !carried.is_empty() {
                                if let Some(obj) = plan.as_object_mut() {
                                    let slot = obj.entry("hints".to_string()).or_insert_with(|| json!({}));
                                    if !slot.is_object() { *slot = json!({}); }
                                    if let Some(h) = slot.as_object_mut() {
                                        for (f, v) in carried.into_iter() { h.insert(f, v); }
                                    }
                                }
                            }
                            if !demoted.is_empty() {
                                for f in demoted.iter() {
                                    crate::utils::score_dynamics::record_search_demotion(f);
                                }
                                println!(
                                    "[AI-SEARCH] 🧮 [RUNTIME DEMOTE] 하드 조건 {:?} 를 이번 회차에서 즉시 힌트로 내립니다. 이 축은 회수한 문서 {}건 전부에 저장되어 있지 않아 어떤 값을 넣어도 통과할 수 없는 조건입니다. 그대로 두면 프런트의 executeDexiePlan 이 같은 조건을 다시 적용해 화면에는 0건만 남습니다. 판정 기준은 단독 차단이 아니라 차단 기여이므로, 조건이 둘 이상 동시에 깨져도 이 경로가 침묵하지 않습니다.",
                                    demoted, docs.len()
                                );
                                let (re_eligible, re_pass, _) = evaluate_dexie_plan(plan, &docs);
                                println!(
                                    "[AI-SEARCH] 🔁 [RE-EVALUATE] 강등 후 하드 조건을 전부 만족하는 문서 {}건 / 서식 범위 안 {}건 (강등 전 {}건). 조건을 버린 것이 아니라 비교 불가능한 축만 힌트로 옮겼으므로 나머지 조건은 그대로 적용됩니다.",
                                    re_pass, re_eligible, all_pass
                                );
                                crate::utils::score_dynamics::record_baseline(
                                    "search.pass_ratio_after_demote",
                                    if re_eligible > 0 { re_pass as f32 / re_eligible as f32 } else { 0.0 },
                                );
                            }
                        }
                        if !narrow.is_empty() {
                            println!(
                                "[AI-SEARCH] 🎯 [TOO NARROW] 회수한 문서에 이 축의 값이 실제로 들어 있는데 조건을 통과하지 못했습니다. 질의 해석(연산자·단위·자릿수)을 의심할 축: {}",
                                narrow.join(" | ")
                            );
                        }
                    }
                    crate::utils::score_dynamics::leave_scope();
                }
            }
            println!("[AI-SEARCH] {}", crate::utils::score_dynamics::report());
            crate::utils::score_dynamics::flush();
            all_results = ranked_results;
        }
        // =====================================================================
        // 🌟 [STAGE-5 종료]
        // =====================================================================

        // 🌟 [RESPONSE v4] 프론트엔드는 아래 3개를 받습니다.
        //    structured  : PLINKO 가 확정한 의미 구조 (기존과 동일)
        //    results     : LanceDB 리콜 후보 (조건 미적용, 넓게)
        //    dexie_plans : Dexie 가 실행할 정밀 필터 플랜 (조건 100% 보존)
        //  → main.ts 가 results 를 dexie_plans 로 걸러 최종 결과를 확정합니다.
        Ok(json!({
            "structured": structured_query,
            "results": all_results,
            "dexie_plans": dexie_plans,
            "mode": search_mode,
            "tracks": fusion_trace
        }))
    }.await; 

    // 🌟 [FLAG HOLD] 여기서 해제하면 안 됩니다.
    //    아래 deep_purge_resources() 는 CUDA 동기화 + OS 메모리 반환으로
    //    수백 ms~수 초가 걸리는데, 그 사이 unload_model / reindex_pending_embeddings 가
    //    IS_SEARCHING 가드를 통과해 파기 중인 CUDA 컨텍스트를 건드립니다.
    //    (ACTIVE_TASK_MEM 도 아래에서 비워지므로 두 가드가 동시에 뚫립니다)
    //    플래그 해제는 정리 작업이 전부 끝난 뒤 함수 말미에서 한 번만 수행합니다.
    
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

    
    // 🌟 [PURGE FIRST] 자원 파기를 먼저 끝낸 뒤에 가드를 내립니다.
    //    순서가 뒤집히면 파기 도중에 다른 커맨드가 모델을 올려 컨텍스트가 충돌합니다.
    println!("[AI-SEARCH] {}", model.crossover_report());
    model.deep_purge_resources().await;
    drop(model_guard);

    {
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        if let Some(mem) = mem_guard.as_ref() {
            if mem.get("id").and_then(|v| v.as_str()) == Some(task_id.as_str()) {
                *mem_guard = None;
            }
        }
    }

    // 🌟 정리가 전부 끝난 뒤 단 한 번만 해제합니다.
    IS_SEARCHING.store(false, Ordering::SeqCst);
    
    search_process
}
