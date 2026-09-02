#![allow(unused_imports)]
// 🌟 [SPLIT] 원본 model.rs 27~113 + 9181~9798 행에서 이동.
//    parse_commerce_query / parse_shipping_query 가 이 함수들을 호출하는데
//    그쪽은 이제 형제 모듈이므로, 비공개 fn 이면 보이지 않습니다.
//    전부 pub(crate) 로 올립니다.
use serde_json::{Value, json, Map};

// pub fn generate_rich_summary(doc_type: &str, data: &Value) -> String { /* 27~113 */ }

// pub(crate) fn trade_resolve_condition_value(field: &str, chunk: &str) -> String { /* ... */ }
// pub(crate) fn trade_resolve_condition_operator(field: &str, chunk: &str) -> String { /* ... */ }
// pub(crate) fn is_schema_echo(s: &str) -> bool { /* ... */ }
// pub(crate) fn collect_claimed(merged: &Map<String, Value>) -> Vec<(String, String)> { /* ... */ }
// pub(crate) fn record_grounding_claims(/* ... */) { /* ... */ }
// pub(crate) fn apply_grounding_verdicts(/* ... */) { /* ... */ }
// pub(crate) fn merge_extracted(/* ... */) { /* ... */ }
// pub fn merge_json_manual(root: &mut Map<String, Value>, cat: &str, data: Value) { /* ... */ }

pub fn generate_rich_summary(doc_type: &str, data: &Value) -> String {
    let type_map = json!({
        "CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List",
        "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit",
        "tracking": "Shipping Label / Tracking Info"
    });
    
    let full_type = type_map.get(doc_type).and_then(|s| s.as_str()).unwrap_or(doc_type);
    let mut parts = vec![format!("This is a {} document.", full_type)];

    if let Some(h) = data.get("header") {
        if let Some(no) = h.get("document_number").and_then(|s| s.as_str()) {
            if no != "N/A" && !no.is_empty() {
                parts.push(format!("Document number is {}.", no));
            }
        }
        if let Some(date) = h.get("issue_date").and_then(|s| s.as_str()) {
            if date != "N/A" && !date.is_empty() {
                parts.push(format!("Issued on {}.", date));
            }
        }
    }

    if doc_type == "tracking" {
        if let Some(tn) = data.get("tracking_number").and_then(|s| s.as_str()) {
            parts.push(format!("The tracking number is {}.", tn));
        }
        if let Some(text) = data.get("text").and_then(|s| s.as_str()) {
            parts.push(text.to_string());
        }
    }

    if let Some(p) = data.get("parties") {
        let sup = p.get("supplier_name").and_then(|s| s.as_str());
        let buy = p.get("buyer_name").and_then(|s| s.as_str());
        
        let has_sup = sup.is_some() && sup.unwrap() != "N/A";
        let has_buy = buy.is_some() && buy.unwrap() != "N/A";

        if has_sup && has_buy {
            parts.push(format!("Transaction involved {} as the supplier/shipper and {} as the buyer/consignee.", sup.unwrap(), buy.unwrap()));
        } else if has_sup {
            parts.push(format!("Supplier/Shipper is {}.", sup.unwrap()));
        } else if has_buy {
            parts.push(format!("Buyer/Consignee is {}.", buy.unwrap()));
        }
    }

    if let Some(f) = data.get("financials") {
        if let Some(amt) = f.get("amount_total") {
             let amt_str = if amt.is_number() { amt.to_string() } else { amt.as_str().unwrap_or("0").to_string() };
             let curr = f.get("currency_code").and_then(|s| s.as_str()).unwrap_or("USD");
             if amt_str != "0" && amt_str != "0.0" {
                 parts.push(format!("Total amount is {} {}.", amt_str, curr));
             }
        }
    }

    if let Some(l) = data.get("logistics") {
        let pol = l.get("location_port_of_loading").and_then(|s| s.as_str());
        let pod = l.get("location_port_of_discharge").and_then(|s| s.as_str());
        
        if let (Some(o), Some(d)) = (pol, pod) {
            if o != "N/A" && d != "N/A" {
                parts.push(format!("Shipped from {} to {}.", o, d));
            }
        }
        
        if let Some(mode) = l.get("transport_mode").and_then(|s| s.as_str()) {
            parts.push(format!("Transport mode is {}.", mode));
        }
    }

    if let Some(items) = data.get("line_items").and_then(|v| v.as_array()) {
        let mut item_descs = Vec::new();
        for item in items.iter().take(5) {
            if let Some(d) = item.get("description").and_then(|s| s.as_str()) {
                if d.len() > 3 { item_descs.push(d); }
            }
        }
        if !item_descs.is_empty() {
            parts.push(format!("Contains items: {}.", item_descs.join(", ")));
        }
    }
    
    parts.join(" ")
}

pub fn trade_resolve_condition_value(field: &str, chunk: &str) -> String {
    let c = chunk.trim();
    if c.is_empty() { return String::new(); }

    // ── ① 문서번호 토큰 우선 ──
    //    'BL-55432219' / 'HBL-55432219-01' / 'AWB-180-99281014'
    if field == "doc_number" || field == "no" || field.starts_with("reference_") || field == "hub_reference" {
        for w in c.split_whitespace() {
            let core: String = w
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '/')
                .collect();
            if core.chars().count() < 4 { continue; }
            if !core.chars().any(|ch| ch.is_ascii_digit()) { continue; }
            if !core.contains('-') && !core.contains('_') && !core.contains('/') {
                // 구분자가 없어도 6자 이상 영숫자면 코드로 인정합니다. (컨테이너/씰 번호)
                if core.chars().count() < 6 { continue; }
            }
            return core;
        }
    }

    // ── ② 수치 / 날짜 ──
    match crate::utils::ai_utils::detect_field_format(field) {
        crate::utils::ai_utils::FieldFormat::Numeric => {
            let v = crate::utils::ai_utils::deterministic_condition_value(&vec![c.to_string()], true);
            if !v.is_empty() { return v; }
        },
        crate::utils::ai_utils::FieldFormat::Date => {
            if let Some(d) = crate::utils::ai_utils::extract_date_literal(c) {
                return d;
            }
            let v = crate::utils::ai_utils::deterministic_condition_value(&vec![c.to_string()], true);
            if !v.is_empty() { return v; }
        },
        _ => {},
    }

    // ── ③ 자유 텍스트 ──
    crate::utils::ai_utils::deterministic_condition_value(&vec![c.to_string()], false)
}

/// 🌟 [TRADE CONDITION OPERATOR] 비교 표현이 명시된 경우에만 기본 연산자를 바꿉니다.
///  ── 근거 ──
///   split_numeric_and_comparator 가 '5000원 이하로' 를 (숫자, 비교 표현) 으로 분해하고,
///   bias.json 의 operators 노드가 다국어 비교 표현을 이미 갖고 있습니다.
///   여기서는 그 구조를 그대로 재사용하되, 임베딩 호출 없이
///   bias.json 의 exact_match 계열 완전일치만으로 판정합니다.
///   (임베딩 판정이 필요한 애매한 경우는 Depth 3 프롬프트가 담당합니다)
pub fn trade_resolve_condition_operator(field: &str, chunk: &str) -> String {
    let default_op = crate::logic::trade_default_operator(field).to_string();

    let fmt = crate::utils::ai_utils::detect_field_format(field);
    let comparable = matches!(
        fmt,
        crate::utils::ai_utils::FieldFormat::Numeric | crate::utils::ai_utils::FieldFormat::Date
    );
    if !comparable { return default_op; }

    // 비교 표현 부분만 잘라냅니다.
    let cmp_part = match crate::utils::ai_utils::split_numeric_and_comparator(chunk) {
        Some((_, cmp)) => cmp,
        None => chunk.to_string(),
    };
    if cmp_part.trim().is_empty() { return default_op; }

    // bias.json operators.*.bias 구와 토큰 완전일치만 봅니다.
    let ops = match crate::parsing::BIAS_DICT.get("operators").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return default_op,
    };

    let tokens: Vec<String> = cmp_part
        .split_whitespace()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let mut best_key = String::new();
    let mut best_len = 0usize;

    for (key, node) in ops {
        if key == "top" || key == "bottom" { continue; }
        for field_name in ["bias", "semantic"] {
            let raw = match node.get(field_name).and_then(|v| v.as_str()) { Some(s) => s, None => continue };
            for phrase in crate::utils::ai_utils::split_bias_phrases_full(raw) {
                let p = phrase.trim().to_lowercase();
                if p.chars().count() < 2 { continue; }
                // 완전일치 또는 접두일치 (교착어 대응)
                let hit = tokens.iter().any(|t| t == &p || (t.chars().count() > p.chars().count() && t.starts_with(&p)));
                if hit && p.chars().count() > best_len {
                    best_len = p.chars().count();
                    best_key = key.clone();
                }
            }
        }
    }

    if best_key.is_empty() {
        default_op
    } else {
        best_key
    }
}

/// 🌟 [MERGE POLICY] 카테고리별 추출 결과를 하나의 객체로 합칩니다.
///
///  ── 왜 별도 함수인가 ──
///   기존 merge_json_manual 은 무조건 덮어썼습니다.
///   header 크롭이 확정한 doc_number 를 parties 크롭이 null 로 덮는 사고가 납니다.
///   크롭은 서로 다른 영역을 보므로, 값이 없다는 사실은
///   '그 영역에 없었다' 는 뜻이지 '문서에 없다' 는 뜻이 아닙니다.
///
///  ── 규칙 ──
///   ① null / 빈 문자열 / 빈 배열은 기존 값을 덮지 않습니다.
///   ② 배열 필드(items / containers 등)는 이어붙입니다.
///   ③ 이미 값이 있는 스칼라 필드는 유지합니다. 먼저 확정된 쪽이 이깁니다.
///      (크롭 계획은 점수 순이므로 근거가 강한 쪽이 먼저 들어옵니다)
/// 🌟 [SCHEMA ECHO GUARD] LLM 이 프롬프트의 스키마 플레이스홀더를 그대로 베낀 경우를 걸러냅니다.
///
///  ── 실측 사고 ──
///   logistics 크롭(서명 영역)에 voyage number 가 없자 2B 모델이
///   프롬프트의 타입 표기 `{String}` 을 값으로 그대로 반환했습니다.
///   그대로 저장하면 data.voyage_number = "{String}" 이 되어
///   Dexie 인덱스와 FTS 를 영구히 오염시킵니다.
///
///  ── 판정 근거 ──
///   어휘 사전이 아니라 '문자 구조' 입니다.
///   중괄호/꺾쇠로 감싼 토큰, 타입 이름 그 자체, 날짜 포맷 문자열은
///   어느 언어의 문서에도 값으로 등장하지 않습니다.
pub fn is_schema_echo(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    // {String} / {Number} / <value> 같은 플레이스홀더 표기
    if (t.starts_with('{') && t.ends_with('}')) || (t.starts_with('<') && t.ends_with('>')) {
        return true;
    }
    let lower = t.to_lowercase();
    matches!(
        lower.as_str(),
        "..." | "null" | "n/a" | "na" | "none" | "undefined" | "unknown"
            | "string" | "number" | "boolean" | "array" | "object" | "integer" | "float"
            | "yyyy-mm-dd" | "yyyy-mm-ddthh:mm:ss" | "iso8601" | "iso 8601"
            | "not specified" | "not available" | "not found"
    )
}

/// 🌟 [CLAIMED HARVEST] 지금까지 확정된 (필드, 값) 쌍을 뽑아 다음 크롭에 전달합니다.
///
///  ── 왜 필요한가 ──
///   크롭은 카테고리별로 순차 호출되므로 뒤 크롭은 앞 크롭의 결과를 모릅니다.
///   그래서 financials 가 이미 2000.00 을 확정했는데
///   cargo 가 근처의 같은 숫자를 자기 필드로 다시 가져가는 사고가 납니다.
///   scheduler.rs 의 커머스 추출이 [ALREADY CLAIMED VALUES] 로 같은 문제를 막는 것과
///   동일한 장치를 비전 크롭 경로에도 부여합니다.
///
///  ── 무엇을 넘기는가 ──
///   스칼라 값만 넘깁니다. 배열(line_items / containers)은 여러 행이 정상이므로
///   금지 목록에 넣으면 오히려 정답을 막습니다.
pub fn collect_claimed(merged: &serde_json::Map<String, Value>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (k, v) in merged.iter() {
        // 카테고리 그룹 객체는 루트에 이미 미러링되어 있으므로 건너뜁니다.
        if v.is_object() || v.is_array() {
            continue;
        }
        let s = match v {
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            _ => continue,
        };
        if s.is_empty() || is_schema_echo(&s) {
            continue;
        }
        // doc_type 은 우리가 시딩한 값이라 금지 대상이 아닙니다.
        if k == "doc_type" {
            continue;
        }
        if out.iter().any(|(ek, ev)| ek == k && ev == &s) {
            continue;
        }
        out.push((k.clone(), s));
    }
    out
}

/// 🌟 [GROUNDING CLAIM 수집] 한 타일이 주장한 (필드, 값) 을 출처 bbox 와 함께 기록합니다.
///
///  ── 왜 병합 전에 기록하는가 ──
///   병합 후에는 '이 값이 어느 크롭에서 왔는지' 가 사라집니다.
///   접지 검증은 반드시 '값 ↔ 그 값을 주장한 픽셀 영역' 쌍이 있어야 성립하므로
///   주장 시점에 붙잡아 둡니다.
pub fn record_grounding_claims(
    out: &mut Vec<crate::models::siglip2::value_grounding::GroundingClaim>,
    category: &str,
    incoming: &Value,
    bbox: (u32, u32, u32, u32),
) {
    use crate::models::siglip2::value_grounding::GroundingClaim;

    fn push_obj(
        out: &mut Vec<GroundingClaim>,
        category: &str,
        o: &serde_json::Map<String, Value>,
        bbox: (u32, u32, u32, u32),
    ) {
        for (k, v) in o.iter() {
            let s = match v {
                Value::String(s) => s.trim().to_string(),
                Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if s.is_empty() || is_schema_echo(&s) {
                continue;
            }
            // 같은 (필드, 값) 이 여러 타일에서 나오면 한 번만 검증합니다.
            if out.iter().any(|c| c.field == *k && c.value == s) {
                continue;
            }
            out.push(GroundingClaim {
                category: category.to_string(),
                field: k.clone(),
                value: s,
                bbox,
            });
        }
    }

    if let Some(o) = incoming.as_object() {
        push_obj(out, category, o, bbox);
    } else if let Some(arr) = incoming.as_array() {
        for e in arr {
            if let Some(o) = e.as_object() {
                push_obj(out, category, o, bbox);
            }
        }
    }
}

/// 🌟 [GROUNDING 반영] 접지 검증에서 폐기 판정된 값을 데이터에서 제거합니다.
///
///  ── 세 곳을 모두 지워야 합니다 ──
///   ① 루트 평면 배치      : Dexie 인덱스(data.*)가 소비
///   ② 카테고리 그룹 슬롯  : doc_number 탐색과 TRADING FLATTEN 이 소비
///   ③ 배열 원소 필드      : line_items / containers 안의 같은 값
///   한 곳이라도 남으면 폐기된 값이 그 경로로 되살아나 DB 를 오염시킵니다.
///
///  ── 왜 null 이 아니라 제거인가 ──
///   merge_extracted 는 '빈 값은 덮지 않는다' 는 규칙을 갖습니다.
///   null 로 남겨 두면 이후 재스캔에서 정상 값이 들어와도
///   "이미 채워진 스칼라" 로 오인될 여지가 없어야 하므로 키 자체를 제거합니다.
pub fn apply_grounding_verdicts(
    merged: &mut serde_json::Map<String, Value>,
    verdicts: &[crate::models::siglip2::value_grounding::GroundingVerdict],
    emit: &dyn Fn(&str),
) {
    let rejected: Vec<&crate::models::siglip2::value_grounding::GroundingVerdict> =
        verdicts.iter().filter(|v| !v.accepted).collect();
    if rejected.is_empty() {
        emit("  ✅ [GROUNDING APPLY] 폐기 대상이 없습니다. 전 값이 이미지에 접지되어 있습니다.");
        return;
    }

    let same = |v: &Value, target: &str| -> bool {
        match v {
            Value::String(s) => s.trim() == target,
            Value::Number(n) => n.to_string() == target,
            _ => false,
        }
    };

    let mut removed = 0usize;
    for r in rejected.iter() {
        // ① 루트
        let hit_root = merged.get(&r.field).map(|v| same(v, &r.value)).unwrap_or(false);
        if hit_root {
            merged.remove(&r.field);
            removed += 1;
        }

        // ② 카테고리 그룹 슬롯
        if let Some(slot) = merged.get_mut(&r.category) {
            if let Some(o) = slot.as_object_mut() {
                let hit = o.get(&r.field).map(|v| same(v, &r.value)).unwrap_or(false);
                if hit {
                    o.remove(&r.field);
                    removed += 1;
                }
            }
        }

        // ③ 배열 원소
        for key in ["line_items", "items", "containers", "parties", "other_parties", "charges"] {
            if let Some(arr) = merged.get_mut(key).and_then(|v| v.as_array_mut()) {
                for e in arr.iter_mut() {
                    if let Some(o) = e.as_object_mut() {
                        let hit = o.get(&r.field).map(|v| same(v, &r.value)).unwrap_or(false);
                        if hit {
                            o.remove(&r.field);
                            removed += 1;
                        }
                    }
                }
                // 전 필드가 사라진 빈 원소는 행이 아니라 잔해입니다.
                arr.retain(|e| {
                    e.as_object().map(|o| !o.is_empty()).unwrap_or(true)
                });
            }
        }

        emit(&format!(
            "  🗑️ [GROUNDING APPLY] [{}] '{}' = \"{}\" 제거 | {}",
            r.category, r.field, r.value, r.reason
        ));
    }

    emit(&format!(
        "  ✅ [GROUNDING APPLY] 폐기 {}건 | 데이터 지점 {}곳에서 제거",
        rejected.len(),
        removed
    ));
}

pub fn merge_extracted(
    merged: &mut serde_json::Map<String, Value>,
    category: &str,
    incoming: &Value,
    emit: &dyn Fn(&str),
) {
    // 🌟 [ARRAY CATEGORY COERCION]
    //
    //  ── 실측 사고 ──
    //   items 크롭의 프롬프트 스키마는 `[ { ... } ]` 배열인데,
    //   2B 모델은 행이 하나만 보이면 `{ ... }` 객체를 반환합니다.
    //   구버전은 `incoming.as_object()` 가 Some 이면 배열 분기를 타지 않아,
    //   description / quantity / unit / unit_price / total_price / hs_code 6개가
    //   전부 '최상위 스칼라' 로 흘러들어갔고 line_items 는 빈 배열로 남았습니다.
    //   (실측 결과 JSON: line_items: [] 이면서 루트에 description: "T-Shirt")
    //   그 상태로 저장되면 Dexie 의 line_items 인덱스가 영원히 비고,
    //   두 번째 행(Shorts)은 애초에 담을 그릇조차 없습니다.
    //
    //  ── 처방 ──
    //   배열 카테고리에서 객체 하나가 오면 원소 1개짜리 배열로 승격합니다.
    //   '스키마가 배열이면 결과도 배열' 이라는 계약을 코드가 강제합니다.
    let is_array_category = category == "items" || category == "containers";

    let coerced: Value;
    let incoming = if is_array_category && incoming.is_object() {
        let has_content = incoming.as_object().map(|o| {
            o.values().any(|v| {
                !(v.is_null()
                    || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
                    || v.as_array().map(|a| a.is_empty()).unwrap_or(false))
            })
        }).unwrap_or(false);
        if has_content {
            emit(&format!(
                "    🔧 [ARRAY COERCE] [{}] 단일 객체 응답을 원소 1개 배열로 승격합니다.",
                category
            ));
            coerced = Value::Array(vec![incoming.clone()]);
            &coerced
        } else {
            return;
        }
    } else {
        incoming
    };

    let obj = match incoming.as_object() {
        Some(o) => o,
        None => {
            // 카테고리 자체가 배열로 반환되는 경우 (items / containers)
            if let Some(arr) = incoming.as_array() {
                if arr.is_empty() { return; }
                let slot = merged
                    .entry(category.to_string())
                    .or_insert_with(|| Value::Array(Vec::new()));

                // 🌟 [ROW DEDUPE] 겹침 타일 분할은 같은 표 행을 두 타일이 함께 보게 만듭니다.
                //    겹침 영역에서 나온 중복 행을 여기서 제거합니다.
                //    판정은 '의미' 가 아니라 '정규화된 스칼라 값 집합의 완전 일치' 입니다.
                let row_key = |v: &Value| -> String {
                    let o = match v.as_object() { Some(o) => o, None => return v.to_string() };
                    let mut parts: Vec<String> = o
                        .iter()
                        .filter_map(|(k, x)| match x {
                            Value::String(s) if !s.trim().is_empty() => {
                                Some(format!("{}={}", k, s.trim().to_lowercase()))
                            }
                            Value::Number(nn) => Some(format!("{}={}", k, nn)),
                            _ => None,
                        })
                        .collect();
                    parts.sort();
                    parts.join("|")
                };

                // 🌟 [ROW IDENTITY GATE] 표의 '데이터 행' 은 반드시 자기 정체를 갖습니다.
                //
                //  ── 실측 사고 ──
                //   items 배열에 3행이 저장되었는데 실제 품목은 1행뿐이었습니다.
                //     { description: null, item_package_count: 4, total_price: 2000 }  ← 합계 행
                //     { description: null, item_code: "360 Footwear" }                 ← 회사명
                //   합계 행은 품명이 없고, 회사명은 품목 코드 자리에 들어간 서명란 텍스트입니다.
                //   프롬프트의 [TABLE RULES] 가 이미 합계 행 제외를 지시하지만
                //   2B 모델은 타일 경계에서 그 지시를 지키지 못합니다.
                //
                //  ── 판정 근거 (어휘 사전 아님) ──
                //   무역 품목표에서 '품명 없는 데이터 행' 은 정의상 존재하지 않습니다.
                //   합계 행 · 소계 행 · 서명란 텍스트는 전부 품명 칸이 비어 있습니다.
                //   컨테이너 표도 같은 원리로 '번호 없는 컨테이너 행' 은 성립하지 않습니다.
                //   값이 있는지만 보므로 언어와 무관합니다.
                let row_has_identity = |v: &Value| -> bool {
                    let o = match v.as_object() { Some(o) => o, None => return false };
                    let filled = |k: &str| -> bool {
                        match o.get(k) {
                            Some(Value::String(s)) => !s.trim().is_empty() && !is_schema_echo(s),
                            Some(Value::Number(_)) => true,
                            _ => false,
                        }
                    };
                    match category {
                        "items" => filled("description"),
                        "containers" => {
                            filled("container_number") || filled("seal_number") || filled("type_size")
                        }
                        _ => true,
                    }
                };
                let mut added = 0usize;
                let mut dup = 0usize;
                let mut ghost = 0usize;
                if let Some(existing) = slot.as_array_mut() {
                    let mut keys: Vec<String> = existing.iter().map(row_key).collect();
                    for e in arr {
                        if !row_has_identity(e) {
                            ghost += 1;
                            continue;
                        }
                        let k = row_key(e);
                        if k.is_empty() { continue; }
                        if keys.iter().any(|x| x == &k) { dup += 1; continue; }
                        keys.push(k);
                        existing.push(e.clone());
                        added += 1;
                    }
                }
                emit(&format!(
                    "    ➕ [{}] 배열 신규 {}건 | 겹침 중복 {}건 제거 | 정체 없는 행 {}건 폐기 (누적 {}건)",
                    category,
                    added,
                    dup,
                    ghost,
                    merged.get(category).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
                ));
            }
            return;
        }
    };

    let mut added = 0usize;
    let mut kept = 0usize;
    let mut echoed = 0usize;

    for (k, v) in obj.iter() {
        // ① 빈 값 / 스키마 에코는 덮지 않습니다.
        if let Some(s) = v.as_str() {
            if is_schema_echo(s) {
                echoed += 1;
                emit(&format!(
                    "    🚫 [SCHEMA ECHO] [{}] '{}' = \"{}\" 는 프롬프트 플레이스홀더 복사이므로 폐기합니다.",
                    category, k, s
                ));
                continue;
            }
        }
        let is_empty = v.is_null()
            || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
            || v.as_array().map(|a| a.is_empty()).unwrap_or(false);
        if is_empty { continue; }

        // ② 배열은 이어붙입니다.
        if let Some(arr) = v.as_array() {
            let slot = merged
                .entry(k.clone())
                .or_insert_with(|| Value::Array(Vec::new()));
            if slot.is_array() {
                if let Some(existing) = slot.as_array_mut() {
                    for e in arr {
                        existing.push(e.clone());
                    }
                    added += 1;
                    continue;
                }
            }
            *slot = v.clone();
            added += 1;
            continue;
        }

        // ③ 이미 채워진 스칼라는 유지합니다.
        let newly_added = match merged.get(k) {
            Some(existing) if !existing.is_null() => {
                let existing_empty = existing
                    .as_str()
                    .map(|s| s.trim().is_empty())
                    .unwrap_or(false);
                if existing_empty {
                    merged.insert(k.clone(), v.clone());
                    added += 1;
                    true
                } else {
                    kept += 1;
                    false
                }
            }
            _ => {
                merged.insert(k.clone(), v.clone());
                added += 1;
                true
            }
        };

        // 🌟 [CATEGORY SLOT MIRROR] 카테고리 그룹 객체에도 같은 값을 넣습니다.
        //
        //  ── 무엇이 문제였나 ──
        //   구버전은 category 인자를 배열 분기에서만 쓰고, 객체 분기에서는
        //   최상위에 평평하게 삽입했습니다. 그 결과
        //     final_data_map["header"] = {"doc_type":"CI"}   ← 초기값 그대로
        //     final_data_map["doc_number"] = "CI-43726"      ← 루트에 평평하게
        //   가 되어, STEP C 의 TRADING FLATTEN v3 가 header 그룹을 순회할 때
        //   승격할 잎이 doc_type 하나뿐이었습니다.
        //   (실측 로그: "data 루트로 승격한 축 1개: [\"doc_type\"]")
        //   또 doc_number 탐색이 header.document_number → 루트 순서인데
        //   header 가 비어 있어 task_id 폴백이 확정되었습니다.
        //
        //  ── 왜 미러인가 ──
        //   루트 평면 배치는 Dexie 인덱스(data.*)가 소비하므로 그대로 둡니다.
        //   그룹 슬롯은 doc_number 탐색과 FLATTEN 이 소비합니다. 둘 다 필요합니다.
        if newly_added && category != "items" && category != "containers" {
            let slot = merged
                .entry(category.to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if let Some(o) = slot.as_object_mut() {
                o.insert(k.clone(), v.clone());
            }
        }
    }

    emit(&format!(
        "    ✅ [{}] 신규 {}건 | 기존 유지 {}건 | 스키마 에코 폐기 {}건",
        category, added, kept, echoed
    ));
}

pub fn merge_json_manual(root: &mut Map<String, Value>, cat: &str, data: Value) {
    let target_key = if cat == "items" { "line_items" } else if cat == "containers" { "containers" } else { cat };
    
    // Some models might wrap the result in the category name or target_key
    let actual_data = if let Some(inner) = data.get(target_key) { inner.clone() } 
                      else if let Some(inner) = data.get(cat) { inner.clone() } 
                      else { data };

    let actual_data = if root.get(target_key).map(|t| t.is_array()).unwrap_or(false)
        && actual_data.is_object()
    {
        let has_value = actual_data.as_object()
            .map(|o| o.values().any(|v| !v.is_null()))
            .unwrap_or(false);
        if has_value { Value::Array(vec![actual_data]) } else { Value::Array(Vec::new()) }
    } else {
        actual_data
    };
    if let Some(target) = root.get_mut(target_key) {
        if target.is_array() {
            let target_arr = target.as_array_mut().unwrap();
            if let Some(source_arr) = actual_data.as_array() {
                for new_item in source_arr {
                    // Check for duplicates in line_items/containers by description/number
                    let is_dup = if target_key == "line_items" {
                        let new_desc = new_item.get("description").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("description").and_then(|v| v.as_str()).unwrap_or("") == new_desc)
                    } else if target_key == "containers" {
                        let new_no = new_item.get("container_number").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("container_number").and_then(|v| v.as_str()).unwrap_or("") == new_no)
                    } else { false };

                    if !is_dup { target_arr.push(new_item.clone()); }
                }
            }
        } else if let Some(target_obj) = target.as_object_mut() {
            if let Some(source_obj) = actual_data.as_object() {
                for (k, v) in source_obj {
                    // 🌟 [ZERO IS DATA] 기존 `v != 0` 은 값이 실제로 0 인 축을 통째로 버렸습니다.
                    //    freight_amount 0(Freight Prepaid), package_count 0, weight_net 0 은
                    //    모두 '못 찾음' 이 아니라 확정된 값입니다.
                    if v.is_null() { continue; }
                    if let Some(s) = v.as_str() {
                        // 🌟 [SCHEMA ECHO GUARD] 텍스트 경로도 get_trade_category_schema 를
                        //    그대로 쓰므로 비전 경로와 동일한 에코가 발생합니다.
                        //    "{String}" / "String" / "..." 같은 플레이스홀더를 값으로 저장하면
                        //    Dexie 인덱스와 FTS 가 영구히 오염됩니다.
                        if is_schema_echo(s) {
                            println!(
                                "[TRADING] 🚫 [SCHEMA ECHO] '{}' = \"{}\" 는 프롬프트 플레이스홀더 복사이므로 폐기합니다.",
                                k, s
                            );
                            continue;
                        }
                    }
                    target_obj.insert(k.clone(), v.clone());
                }
            }
        }
    }
}