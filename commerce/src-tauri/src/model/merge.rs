use serde_json::{Value, json, Map};
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
            if !is_schema_echo(no) {
                parts.push(format!("Document number is {}.", no));
            }
        }
        if let Some(date) = h.get("issue_date").and_then(|s| s.as_str()) {
            if !is_schema_echo(date) {
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
        
        let has_sup = sup.map_or(false, |s| !is_schema_echo(s));
        let has_buy = buy.map_or(false, |s| !is_schema_echo(s));

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
            if !is_schema_echo(o) && !is_schema_echo(d) {
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

    let identifier_axis = field == "doc_number"
        || field == "no"
        || field.starts_with("reference_")
        || field == "hub_reference"
        || matches!(
            crate::utils::ai_utils::query_value_format(field),
            crate::utils::ai_utils::FieldFormat::Identifier | crate::utils::ai_utils::FieldFormat::TrackingCode
        );
    if identifier_axis {
        for w in c.split_whitespace() {
            let core: String = w
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '/' || *ch == '.')
                .collect::<String>()
                .trim_end_matches(|ch: char| ch == '-' || ch == '_' || ch == '/' || ch == '.')
                .to_string();
            if core.chars().count() < 4 { continue; }
            if !core.chars().any(|ch| ch.is_ascii_digit()) { continue; }
            if !core.contains('-') && !core.contains('_') && !core.contains('/') && !core.contains('.') {
                if core.chars().count() < 6 { continue; }
            }
            return core;
        }
        return String::new();
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

pub fn cross_field_duplicate_groups(
    claims: &[crate::models::siglip2::value_grounding::GroundingClaim],
    verdicts: &[crate::models::siglip2::value_grounding::GroundingVerdict],
) -> Vec<(String, Vec<(String, String, String)>)> {
    use crate::utils::ai_utils::FieldFormat;
    let mut groups: Vec<(String, Vec<(String, String, String)>)> = Vec::new();
    for c in claims.iter() {
        if crate::logic::TRADE_ARRAY_CATEGORIES.iter().any(|a| *a == c.category.as_str()) { continue; }
        if matches!(c.field.trim(), "party_role" | "doc_type") { continue; }
        let fmt = crate::utils::ai_utils::detect_field_format(&c.field);
        if !matches!(fmt, FieldFormat::Text | FieldFormat::Address) { continue; }
        let v = c.value.trim();
        if v.chars().filter(|ch| ch.is_alphanumeric()).count() < 2 { continue; }
        if !v.chars().any(|ch| ch.is_alphabetic()) { continue; }
        let rejected = verdicts.iter().any(|r| {
            !r.accepted && r.category == c.category && r.field == c.field && r.value.trim() == v
        });
        if rejected { continue; }
        let key = v.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, owners)) => {
                if !owners.iter().any(|(cat, f, _)| *cat == c.category && *f == c.field) {
                    owners.push((c.category.clone(), c.field.clone(), v.to_string()));
                }
            }
            None => groups.push((key, vec![(c.category.clone(), c.field.clone(), v.to_string())])),
        }
    }
    {
        let mut kept: Vec<(String, Vec<(String, String, String)>)> = Vec::new();
        let mut same_field: Vec<String> = Vec::new();
        for (key, owners) in groups.into_iter() {
            let mut fields: Vec<&str> = owners.iter().map(|(_, f, _)| f.as_str()).collect();
            fields.sort();
            fields.dedup();
            if fields.len() >= 2 {
                kept.push((key, owners));
                continue;
            }
            if owners.len() >= 2 {
                same_field.push(format!(
                    "\"{}\" ← {:?}",
                    owners[0].2,
                    owners.iter().map(|(c, f, _)| format!("{}.{}", c, f)).collect::<Vec<_>>()
                ));
            }
        }
        if !same_field.is_empty() {
            println!(
                "    ⚪ [SAME FIELD DUPLICATE] 같은 필드명이 서로 다른 카테고리에서 같은 값을 주장한 {}건은 소유권 경쟁이 아니라 카테고리 배정의 산물입니다. 두 주장이 가리키는 축이 하나뿐이라 어느 쪽을 지워도 그 필드의 값이 통째로 사라지므로 경쟁에서 제외합니다: {:?}",
                same_field.len(),
                same_field.iter().take(6).collect::<Vec<_>>()
            );
        }
        groups = kept;
    }
    groups
}

pub fn resolve_cross_field_duplicates(
    groups: &[(String, Vec<(String, String, String)>)],
    lookup: &std::collections::HashMap<String, Vec<f32>>,
    doc_lang: &str,
    bank_type: &str,
    emit: &dyn Fn(&str),
) -> Vec<crate::models::siglip2::value_grounding::GroundingVerdict> {
    let mut out: Vec<crate::models::siglip2::value_grounding::GroundingVerdict> = Vec::new();
    for (value, owners) in groups.iter() {
        let q = match lookup.get(value) {
            Some(v) => v,
            None => continue,
        };
        if q.iter().all(|&x| x == 0.0) { continue; }
        let mut pool: Vec<f32> = Vec::new();
        let mut per: Vec<(usize, f32, usize)> = Vec::new();
        for (oi, (_, field, _)) in owners.iter().enumerate() {
            let (phrases, weights) = owner_label_bank(doc_lang, bank_type, field);
            let mut mx = f32::MIN;
            let mut live = 0usize;
            for (p, w) in phrases.iter().zip(weights.iter()) {
                let e = match lookup.get(p) { Some(e) => e, None => continue };
                if e.iter().all(|&x| x == 0.0) { continue; }
                let s = crate::utils::ai_utils::cosine_similarity(q, e) * *w;
                pool.push(s);
                live += 1;
                if s > mx { mx = s; }
            }
            if live == 0 { continue; }
            per.push((oi, mx, live));
        }
        if per.len() < 2 || pool.len() < 2 { continue; }
        let n = pool.len() as f32;
        let mean = pool.iter().sum::<f32>() / n;
        let sd = (pool.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n)
            .sqrt()
            .max(1e-6);
        let mut scored: Vec<(usize, f32, usize)> = per
            .iter()
            .map(|(oi, mx, cnt)| {
                (
                    *oi,
                    (mx - mean) / sd - crate::utils::ai_utils::gumbel_expected_z(*cnt),
                    *cnt,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (wi, w_z, w_cnt) = scored[0];
        for &(li, l_z, l_cnt) in scored.iter().skip(1) {
            let (l_cat, l_field, l_raw) = &owners[li];
            let (w_cat, w_field, _) = &owners[wi];
            if w_z - l_z > 1.0 {
                crate::utils::score_dynamics::record_confusion(w_field, l_field, w_z - l_z);
                emit(&format!(
                    "    🧭 [VALUE OWNER] \"{}\" | {}.{} (중립점수 {:+.4}, 라벨 {}구) 가 {}.{} (중립점수 {:+.4}, 라벨 {}구) 를 pooled σ 한 칸 이상 앞섭니다. 인쇄된 한 자리는 라벨을 하나만 가지므로 후자의 값을 폐기합니다. 원시 Max-Pool 은 라벨 구가 많은 축이 구조적으로 이기므로 표본 수 기대 최댓값을 차감한 뒤 비교합니다.",
                    l_raw, w_cat, w_field, w_z, w_cnt, l_cat, l_field, l_z, l_cnt
                ));
                out.push(crate::models::siglip2::value_grounding::GroundingVerdict {
                    category: l_cat.clone(),
                    field: l_field.clone(),
                    value: l_raw.clone(),
                    surprisal_in: 0.0,
                    surprisal_out: 0.0,
                    top_patch: 0,
                    top_legible: true,
                    accepted: false,
                    gate: crate::models::siglip2::value_grounding::VerdictGate::Prejudice,
                    reason: "필드 소유권 경쟁 패배 (다른 축이 같은 값을 더 잘 설명함)".to_string(),
                });
            } else {
                emit(&format!(
                    "    🤝 [VALUE OWNER KEEP] \"{}\" | {}.{} (중립점수 {:+.4}) 와 {}.{} (중립점수 {:+.4}) 의 차이가 {:+.4} 로 pooled σ 한 칸에 못 미칩니다. 같은 값이 두 축에 정당하게 인쇄되는 서식(송하인과 통지처가 같은 회사인 경우 등)이 있으므로 둘 다 유지합니다.",
                    l_raw, w_cat, w_field, w_z, l_cat, l_field, l_z, w_z - l_z
                ));
            }
        }
    }
    out
}

pub fn drop_row_echo_columns(merged: &mut serde_json::Map<String, Value>, emit: &dyn Fn(&str)) {
    const ECHO_RULES: [(&str, &str, &str, &str); 1] = [
        ("item_package_count", "quantity", "cargo", "package_count"),
    ];
    fn num_of(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let t: String = s
                    .chars()
                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                t.parse::<f64>().ok()
            }
            _ => None,
        }
    }
    for (row_field, sibling, total_cat, total_field) in ECHO_RULES.iter() {
        let total = merged
            .get(*total_cat)
            .and_then(|c| c.get(*total_field))
            .and_then(num_of)
            .or_else(|| merged.get(*total_field).and_then(num_of));
        let total = match total {
            Some(t) => t,
            None => continue,
        };
        for arr_key in ["items", "line_items"] {
            let rows = match merged.get_mut(arr_key).and_then(|v| v.as_array_mut()) {
                Some(r) => r,
                None => continue,
            };
            if rows.is_empty() { continue; }
            let mut sum = 0.0f64;
            let mut seen = 0usize;
            let mut all_echo = true;
            for r in rows.iter() {
                let a = r.get(*row_field).and_then(num_of);
                let b = r.get(*sibling).and_then(num_of);
                match (a, b) {
                    (Some(x), Some(y)) => {
                        sum += x;
                        seen += 1;
                        if (x - y).abs() > 1e-9 { all_echo = false; }
                    }
                    (Some(x), None) => {
                        sum += x;
                        seen += 1;
                        all_echo = false;
                    }
                    _ => {}
                }
            }
            if seen == 0 || !all_echo { continue; }
            if (sum - total).abs() <= 1e-9 { continue; }
            for r in rows.iter_mut() {
                if let Some(o) = r.as_object_mut() {
                    if o.contains_key(*row_field) { o.insert(row_field.to_string(), Value::Null); }
                }
            }
            emit(&format!(
                "  🧮 [ROW ECHO DROP] [{}] '{}' 가 모든 행에서 '{}' 와 같은 값이고 합계 {} 가 문서 총계 {}.{} = {} 와 어긋납니다. 인쇄되지 않은 열을 옆 열 값으로 채운 복사로 보고 비웁니다.",
                arr_key, row_field, sibling, sum, total_cat, total_field, total
            ));
        }
    }
}

/// 🌟 [MONETARY RECONCILE] 금액 축에 들어온 값이 이 문서의 돈인지 산술로 확인합니다.
///
///  ── 실측 사고 ──
///   amount      = 1270221736   ← 수하인 부가세번호 (정답 2000)
///   amount_tax  = 4832882      ← 수출자 부가세번호 (정답 없음)
///   두 값 다 라벨 코사인이 정당하게 높았습니다. amount_tax 의 앵커에
///   'VAT' 가 들어 있고 인쇄 라벨이 "EXPORTER VAT/EORI" 이기 때문입니다.
///   값 형식 게이트는 Numeric 만 보므로 등록번호를 막지 못합니다.
///
///  ── 왜 산술인가 ──
///   라벨로 못 가르는 것을 값으로 가르려면 어휘가 필요하고, 어휘는 이 코드가
///   가질 수 없습니다. 그러나 인보이스의 돈은 서로 산술로 묶여 있습니다.
///     · 세액은 과세표준을 넘을 수 없습니다.
///     · 총계는 행 합계보다 작을 수 없습니다.
///     · 총계는 소계 + 세액 + 부대비용으로 설명되어야 합니다.
///   drop_row_echo_columns 가 행 합계 대 문서 총계를 대조하는 것과 같은 판정입니다.
///
///  ── 왜 '열 배' 인가 ──
///   임계값이 아니라 자릿수 판정입니다. 부대비용 한 줄을 못 읽어 상한이
///   낮게 잡히는 일은 흔하지만, 그 경우 오차는 배수가 아니라 비율입니다.
///   상한의 열 배를 넘는다는 것은 같은 문서의 돈이 아니라는 뜻입니다.
pub fn reconcile_monetary_axes(
    merged: &mut serde_json::Map<String, Value>,
    emit: &dyn Fn(&str),
) -> usize {
    fn num_of(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let t: String = s
                    .chars()
                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                if !t.chars().any(|c| c.is_ascii_digit()) {
                    return None;
                }
                t.parse::<f64>().ok()
            }
            _ => None,
        }
    }
    fn read(m: &serde_json::Map<String, Value>, f: &str) -> Option<f64> {
        if let Some(v) = m.get(f).and_then(num_of) {
            return Some(v);
        }
        m.values()
            .filter_map(|v| v.as_object())
            .find_map(|o| o.get(f).and_then(num_of))
    }
    fn purge(m: &mut serde_json::Map<String, Value>, field: &str) {
        m.remove(field);
        let cats: Vec<String> = m.keys().cloned().collect();
        for c in cats {
            if let Some(o) = m.get_mut(&c).and_then(|v| v.as_object_mut()) {
                o.remove(field);
            }
        }
        for key in ["items", "line_items", "containers", "charges", "account_ledger"] {
            if let Some(arr) = m.get_mut(key).and_then(|v| v.as_array_mut()) {
                for e in arr.iter_mut() {
                    if let Some(o) = e.as_object_mut() {
                        o.remove(field);
                    }
                }
            }
        }
    }
    fn reject(m: &mut serde_json::Map<String, Value>, field: &str) {
        purge(m, field);
        crate::utils::score_dynamics::record_field_seen(field);
        crate::utils::score_dynamics::record_field_reject(
            field,
            crate::utils::score_dynamics::GateKind::Format,
        );
    }

    let mut items_sum = 0.0f64;
    let mut items_rows = 0usize;
    for key in ["items", "line_items"] {
        if let Some(arr) = merged.get(key).and_then(|v| v.as_array()) {
            let mut s = 0.0f64;
            let mut c = 0usize;
            for r in arr.iter() {
                if let Some(x) = r.get("total_price").and_then(num_of) {
                    s += x;
                    c += 1;
                }
            }
            if c > items_rows {
                items_sum = s;
                items_rows = c;
            }
        }
    }
    let row_anchor = if items_rows > 0 && items_sum > 0.0 { Some(items_sum) } else { None };
    let subtotal = read(merged, "amount_subtotal");
    let base = row_anchor.or(subtotal).or_else(|| read(merged, "amount"));

    let mut dropped = 0usize;

    if let (Some(t), Some(b)) = (read(merged, "amount_tax"), base) {
        if b > 0.0 && t > b {
            emit(&format!(
                "  🧮 [MONETARY RECONCILE] 'amount_tax' = {} 가 과세표준 {} 를 넘습니다. 세액이 과세표준보다 클 수는 없으므로 이 자리에 인쇄된 것은 금액이 아니라 등록번호(사업자·부가세·EORI)입니다. 라벨 뱅크에 'VAT' 가 들어 있으면 부가세번호가 세액 축으로 흘러드는데, Numeric 형식 게이트는 숫자라는 사실만 보므로 그 오배정을 걸러내지 못합니다.",
                t, b
            ));
            reject(merged, "amount_tax");
            dropped += 1;
        }
    }

    let mut ceiling = 0.0f64;
    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = row_anchor.or(subtotal) {
        if s > 0.0 {
            ceiling += s;
            parts.push(format!(
                "{} {}",
                if row_anchor.is_some() { "행 합계" } else { "소계" },
                s
            ));
        }
    }
    for extra in [
        "amount_tax",
        "freight_amount",
        "insurance_amount",
        "local_charges",
        "freight_charge",
        "insurance",
    ] {
        if let Some(v) = read(merged, extra) {
            if v > 0.0 {
                ceiling += v;
                parts.push(format!("{} {}", extra, v));
            }
        }
    }
    let floor_ref = row_anchor.or(subtotal).unwrap_or(0.0);

    for axis in ["amount", "grand_total_amount"] {
        let g = match read(merged, axis) {
            Some(v) => v,
            None => continue,
        };
        if floor_ref > 0.0 && g < floor_ref {
            emit(&format!(
                "  🧮 [MONETARY RECONCILE] '{}' = {} 가 행 합계/소계 {} 보다 작습니다. 총계는 자기 구성 항목의 합보다 작을 수 없으므로 이 값은 이 문서의 총계가 아닙니다. 비웁니다.",
                axis, g, floor_ref
            ));
            reject(merged, axis);
            dropped += 1;
            continue;
        }
        if ceiling > 0.0 && g > ceiling * 10.0 {
            emit(&format!(
                "  🧮 [MONETARY RECONCILE] '{}' = {} 는 이 문서가 설명할 수 있는 상한 {} ({}) 의 열 배를 넘습니다. 부대비용 한 줄을 놓쳐 상한이 낮게 잡히는 일은 흔하지만 그 오차는 비율이지 자릿수가 아닙니다. 자릿수가 다르면 같은 문서의 돈이 아니므로 비웁니다.",
                axis, g, ceiling, parts.join(" + ")
            ));
            reject(merged, axis);
            dropped += 1;
        }
    }

    crate::utils::score_dynamics::record_baseline("vision.monetary_reconcile", dropped as f32);
    if dropped == 0 {
        emit("  ✅ [MONETARY RECONCILE] 금액 축이 서로 산술로 정합합니다.");
    }
    dropped
}

pub fn row_identity_fields(category: &str) -> &'static [&'static str] {
    match category {
        "items" => &["description"],
        "containers" => &["container_number", "seal_number", "type_size"],
        "other_parties" => &["party_name", "signatory_name"],
        "charges" => &["charge_code", "charge_description"],
        "account_ledger" => &["transaction_date", "debit", "credit"],
        _ => &[],
    }
}

pub fn closed_vocab_echo(value: &str, vocab: &[String]) -> bool {
    let mut seen = false;
    for tok in value.split(|c: char| !c.is_alphanumeric()) {
        if tok.is_empty() { continue; }
        seen = true;
        if !vocab.iter().any(|v| same_printed_token(v, tok)) {
            return false;
        }
    }
    seen
}

/// 🌟 [WEIGHT BASIS] '총중량' 과 '순중량' 을 라벨이 아니라 산술로 가릅니다.
///
///  ── 왜 라벨로는 못 가르는가 ──
///   영어 "TOTAL WEIGHT", 독일어 "Gesamtgewicht", 프랑스어 "poids total",
///   중국어 "总重量" 에는 총/순 표지가 없습니다. 이 라벨 하나로 weight_gross 와
///   weight_net 중 어느 쪽인지 판정할 근거가 인쇄물에 존재하지 않습니다.
///   실측에서도 두 축이 +1.8818 대 +1.7124 의 잡음 차로 뒤집혔습니다.
///   어느 쪽 뱅크에 구를 넣어도 그 잡음이 결정론적 오답으로 바뀔 뿐입니다.
///
///  ── 산술은 근거가 됩니다 ──
///   순중량 총계는 품목 순중량의 합을 넘을 수 없고,
///   총중량은 그 합보다 작을 수 없습니다. 포장재는 더해질 뿐 빠지지 않습니다.
///   drop_row_echo_columns 가 행 합계 대 문서 총계를 대조하는 것과 같은 판정입니다.
///
///  ── 모든 행에 단위중량이 있을 때만 발화합니다 ──
///   한 행이라도 비어 있으면 합계가 과소 추정되어 정상 값을 오답으로 만듭니다.
///   근거가 불완전하면 판정하지 않고 관측만 남깁니다.
pub fn reconcile_weight_basis(
    merged: &mut serde_json::Map<String, Value>,
    emit: &dyn Fn(&str),
) -> usize {
    fn num_of(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let t: String = s
                    .chars()
                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                if !t.chars().any(|c| c.is_ascii_digit()) { return None; }
                t.parse::<f64>().ok()
            }
            _ => None,
        }
    }
    fn read(m: &serde_json::Map<String, Value>, f: &str) -> Option<f64> {
        if let Some(v) = m.get(f).and_then(num_of) { return Some(v); }
        m.values()
            .filter_map(|v| v.as_object())
            .find_map(|o| o.get(f).and_then(num_of))
    }
    fn move_axis(m: &mut serde_json::Map<String, Value>, from: &str, to: &str, v: Value) {
        m.remove(from);
        let cats: Vec<String> = m.keys().cloned().collect();
        for c in cats {
            if let Some(o) = m.get_mut(&c).and_then(|x| x.as_object_mut()) {
                o.remove(from);
            }
        }
        m.insert(to.to_string(), v.clone());
        let slot = m
            .entry("cargo".to_string())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        if let Some(o) = slot.as_object_mut() {
            o.insert(to.to_string(), v);
        }
    }

    let mut rows: Vec<&Value> = Vec::new();
    for key in ["items", "line_items"] {
        if let Some(arr) = merged.get(key).and_then(|v| v.as_array()) {
            if arr.len() > rows.len() {
                rows = arr.iter().collect();
            }
        }
    }
    if rows.is_empty() { return 0; }

    let mut net_sum = 0.0f64;
    let mut missing = 0usize;
    for r in rows.iter() {
        let unit = r.get("item_net_weight").and_then(num_of);
        let qty = r.get("quantity").and_then(num_of).unwrap_or(1.0);
        match unit {
            Some(u) => net_sum += u * qty,
            None => missing += 1,
        }
    }
    if missing > 0 || net_sum <= 0.0 {
        emit(&format!(
            "  👁️ [WEIGHT BASIS OBSERVE] 품목 {}행 중 {}행에 단위 순중량이 없어 총계를 산술로 검증할 수 없습니다. 'TOTAL WEIGHT' 계열 라벨은 총/순 표지가 없어 라벨만으로는 어느 축인지 알 수 없으므로, 이번 회차는 현재 배정을 그대로 둡니다.",
            rows.len(), missing
        ));
        crate::utils::score_dynamics::record_baseline("vision.weight_basis_observe", 1.0);
        return 0;
    }

    let net = read(merged, "weight_net");
    let gross = read(merged, "weight_gross");
    let eps = net_sum.abs() * 1e-6;
    let mut fixed = 0usize;

    if let Some(n) = net {
        if n > net_sum + eps {
            let v = json!(n);
            if gross.is_none() {
                emit(&format!(
                    "  🧮 [WEIGHT BASIS] 'weight_net' = {} 이 품목 순중량 합 {} 을 넘습니다. 순중량 총계는 자기 구성 항목의 합을 넘을 수 없으므로 이 자리에 인쇄된 것은 총중량입니다. 비어 있는 'weight_gross' 로 옮깁니다. 이 판정은 라벨이 아니라 산술이므로 'TOTAL WEIGHT' 처럼 총/순 표지가 없는 라벨에서도 성립합니다.",
                    n, net_sum
                ));
                move_axis(merged, "weight_net", "weight_gross", v);
            } else {
                emit(&format!(
                    "  🧮 [WEIGHT BASIS] 'weight_net' = {} 이 품목 순중량 합 {} 을 넘는데 'weight_gross' 도 이미 {} 로 차 있습니다. 옮길 자리가 없으므로 비웁니다.",
                    n, net_sum, gross.unwrap_or(0.0)
                ));
                merged.remove("weight_net");
                let cats: Vec<String> = merged.keys().cloned().collect();
                for c in cats {
                    if let Some(o) = merged.get_mut(&c).and_then(|x| x.as_object_mut()) {
                        o.remove("weight_net");
                    }
                }
            }
            crate::utils::score_dynamics::record_field_seen("weight_net");
            crate::utils::score_dynamics::record_field_reject(
                "weight_net",
                crate::utils::score_dynamics::GateKind::Format,
            );
            fixed += 1;
        }
    }

    if fixed == 0 {
        if let Some(g) = gross {
            if g + eps < net_sum {
                let v = json!(g);
                if net.is_none() {
                    emit(&format!(
                        "  🧮 [WEIGHT BASIS] 'weight_gross' = {} 이 품목 순중량 합 {} 보다 작습니다. 포장재는 더해질 뿐 빠지지 않으므로 총중량이 순중량 합보다 작을 수 없습니다. 비어 있는 'weight_net' 으로 옮깁니다.",
                        g, net_sum
                    ));
                    move_axis(merged, "weight_gross", "weight_net", v);
                } else {
                    emit(&format!(
                        "  🧮 [WEIGHT BASIS] 'weight_gross' = {} 이 품목 순중량 합 {} 보다 작은데 'weight_net' 도 이미 차 있습니다. 비웁니다.",
                        g, net_sum
                    ));
                    merged.remove("weight_gross");
                    let cats: Vec<String> = merged.keys().cloned().collect();
                    for c in cats {
                        if let Some(o) = merged.get_mut(&c).and_then(|x| x.as_object_mut()) {
                            o.remove("weight_gross");
                        }
                    }
                }
                crate::utils::score_dynamics::record_field_seen("weight_gross");
                crate::utils::score_dynamics::record_field_reject(
                    "weight_gross",
                    crate::utils::score_dynamics::GateKind::Format,
                );
                fixed += 1;
            }
        }
    }

    if fixed == 0 {
        emit(&format!(
            "  ✅ [WEIGHT BASIS] 중량 축이 품목 순중량 합 {} 과 산술로 정합합니다.",
            net_sum
        ));
    }
    crate::utils::score_dynamics::record_baseline("vision.weight_basis", fixed as f32);
    fixed
}

pub fn reconcile_package_axes(
    merged: &mut serde_json::Map<String, Value>,
    emit: &dyn Fn(&str),
) -> usize {
    fn num_of(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let t: String = s
                    .chars()
                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                if !t.chars().any(|c| c.is_ascii_digit()) {
                    return None;
                }
                t.parse::<f64>().ok()
            }
            _ => None,
        }
    }
    let total = merged
        .get("cargo")
        .and_then(|c| c.get("package_count"))
        .and_then(num_of)
        .or_else(|| merged.get("package_count").and_then(num_of));
    let total = match total {
        Some(t) if t > 0.0 => t,
        _ => return 0,
    };
    let mut dropped = 0usize;
    let mut removed = 0usize;
    if let Some(rows) = merged.get_mut("containers").and_then(|v| v.as_array_mut()) {
        for r in rows.iter_mut() {
            let o = match r.as_object_mut() {
                Some(o) => o,
                None => continue,
            };
            let n = match o.get("container_package_count").and_then(num_of) {
                Some(n) => n,
                None => continue,
            };
            if n > total {
                emit(&format!(
                    "  🧮 [PACKAGE RECONCILE] containers.container_package_count = {} 가 문서 총 포장수 cargo.package_count = {} 를 넘습니다. 컨테이너 한 대의 포장수는 문서 총계의 부분이므로 총계를 넘을 수 없습니다. 이 자리에 인쇄된 것은 포장수가 아니라 다른 총계이므로 비웁니다.",
                    n, total
                ));
                o.insert("container_package_count".to_string(), Value::Null);
                crate::utils::score_dynamics::record_field_seen("container_package_count");
                crate::utils::score_dynamics::record_field_reject(
                    "container_package_count",
                    crate::utils::score_dynamics::GateKind::Format,
                );
                dropped += 1;
            }
        }
        let before = rows.len();
        rows.retain(|r| {
            let o = match r.as_object() {
                Some(o) => o,
                None => return false,
            };
            let filled = |v: &Value| -> bool {
                match v {
                    Value::String(s) => !s.trim().is_empty() && !is_schema_echo(s),
                    Value::Number(_) => true,
                    _ => false,
                }
            };
            row_identity_fields("containers")
                .iter()
                .any(|k| o.get(*k).map(|v| filled(v)).unwrap_or(false))
        });
        removed = before - rows.len();
    }
    if dropped > 0 || removed > 0 {
        emit(&format!(
            "  🧮 [PACKAGE RECONCILE] 포장수 축 {}건 비움 | 정체성을 잃은 컨테이너 행 {}건 제거",
            dropped, removed
        ));
    }
    crate::utils::score_dynamics::record_baseline(
        "vision.package_reconcile",
        (dropped + removed) as f32,
    );
    dropped + removed
}

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
            | "nicht angegeben" | "nicht verfügbar" | "keine angabe" | "unbekannt" | "entfällt"
            | "non spécifié" | "non renseigné" | "non disponible" | "inconnu" | "sans objet"
            | "no especificado" | "no disponible" | "desconocido" | "no aplica" | "sin datos"
            | "non specificato" | "non disponibile" | "sconosciuto" | "non applicabile"
            | "não especificado" | "não disponível" | "desconhecido" | "não aplicável"
            | "niet opgegeven" | "niet beschikbaar" | "onbekend" | "niet van toepassing"
            | "neuvedeno" | "nedostupné" | "neznámé" | "nevztahuje se"
            | "غير محدد" | "غير متوفر" | "غير معروف" | "لا ينطبق"
            | "指定なし" | "該当なし" | "不明" | "未記入" | "なし"
            | "未指定" | "不适用" | "未知" | "无" | "未提供"
            | "해당 없음" | "해당없음" | "정보 없음" | "정보없음" | "미기재" | "없음" | "미상"
    )
}

const PLURAL_SUFFIXES_ML: [&str; 6] = ["S", "ES", "N", "EN", "X", "Y"];
const PLURAL_VOWEL_SWAPS_ML: [(char, char); 4] = [('O', 'I'), ('A', 'E'), ('E', 'I'), ('A', 'Y')];

fn plural_equivalent(short: &str, long: &str) -> bool {
    let sc = short.chars().count();
    let lc = long.chars().count();
    if sc == 0 || lc < sc { return false; }
    if lc > sc {
        return match long.strip_prefix(short) {
            Some("S") => true,
            Some(rest) => sc >= 3 && PLURAL_SUFFIXES_ML.iter().any(|s| *s == rest),
            None => false,
        };
    }
    if sc < 4 { return false; }
    let mut a: Vec<char> = short.chars().collect();
    let mut b: Vec<char> = long.chars().collect();
    let la = match a.pop() { Some(c) => c, None => return false };
    let lb = match b.pop() { Some(c) => c, None => return false };
    if a != b { return false; }
    PLURAL_VOWEL_SWAPS_ML
        .iter()
        .any(|(s, p)| (*s == la && *p == lb) || (*s == lb && *p == la))
}

pub fn same_printed_token(a: &str, b: &str) -> bool {
    let norm = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_uppercase())
            .collect()
    };
    let (x, y) = (norm(a), norm(b));
    if x.is_empty() || y.is_empty() { return false; }
    if x == y { return true; }
    let (short, long) = if x.chars().count() <= y.chars().count() { (&x, &y) } else { (&y, &x) };
    plural_equivalent(short, long)
}

pub fn same_printed_value(a: &str, b: &str) -> bool {
    if same_printed_token(a, b) { return true; }
    let num = |s: &str| -> Option<f64> {
        let t: String = s
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if !t.chars().any(|c| c.is_ascii_digit()) { return None; }
        t.parse::<f64>().ok()
    };
    match (num(a), num(b)) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-6,
        _ => false,
    }
}

/// 뱅크 전체의 중심 벡터. 자기 정화의 기준점입니다.
///
///  ── 왜 첫 구(head)가 아니라 중심인가 ──
///   기존 구현은 뱅크의 첫 유효 구를 대표로 삼았습니다. 라벨 뱅크가 영어 한 벌일 때는
///   첫 구가 곧 그 필드의 이름이라 무해했지만, 12개 언어 구를 합치면 첫 구는 '영어 표기' 일 뿐입니다.
///   비영어 구는 교차언어 거리 때문에 자기 대표와의 코사인이 구조적으로 낮아져,
///   의미가 정확한 번역어까지 '자기 필드를 설명하지 못한다' 는 이유로 잘려 나갑니다.
///   query_shipping.rs 의 [TIME UNIT SELF-POISON] 이 같은 이유로 이미 중심 방식을 씁니다.
pub fn bank_centroid(bank: &[Vec<f32>]) -> Vec<f32> {
    let dim = match bank.iter().map(|e| e.len()).max() {
        Some(d) if d > 0 => d,
        _ => return Vec::new(),
    };
    let mut c = vec![0.0f32; dim];
    let mut cnt = 0usize;
    for e in bank.iter() {
        if e.len() != dim || e.iter().all(|&v| v == 0.0) { continue; }
        for (k, v) in e.iter().enumerate() { c[k] += v; }
        cnt += 1;
    }
    if cnt == 0 { return Vec::new(); }
    let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for v in c.iter_mut() { *v /= norm; }
    }
    c
}

/// 소유권 경쟁·복구 게이트가 쓰는 라벨 뱅크. 다국어 + 보강표를 합칩니다.
///
///  ── 왜 함수로 빼는가 ──
///   이 뱅크를 만드는 곳과 그 구의 임베딩을 미리 계산해 두는 곳(vision.rs 의 lookup)이
///   서로 다른 파일에 있습니다. 두 곳이 각자 뱅크를 조립하면 한쪽만 다국어가 되는 순간
///   lookup 에 없는 구가 조용히 건너뛰어져, 구를 늘렸는데 점수가 그대로인 상태가 됩니다.
pub fn owner_label_bank(doc_lang: &str, bank_type: &str, field: &str) -> (Vec<String>, Vec<f32>) {
    let (mut ph, mut wt) =
        crate::utils::ai_utils::label_phrase_bank_multilingual(doc_lang, bank_type, field);
    let sup = crate::logic::trade_label_supplement(field);
    if !sup.is_empty() {
        crate::logic::merge_phrase_bank(&mut ph, &mut wt, &sup, 1.0);
    }
    (ph, wt)
}

/// 🌟 [ANCHOR SELF-POISON] 다른 필드의 이름을 품은 앵커 구를 그 필드 뱅크에서 뺍니다.
///
///  ── 실측 사고 ──
///   cargo.weight_net 의 앵커에 "CI Total net weight" 가 들어 있습니다.
///   인쇄 라벨 "TOTAL WEIGHT" 는 총중량이므로 정답이 weight_gross 인데,
///   그 구의 'Total' 이 공명해 weight_net 이 +1.8818 vs +1.7124 로 이겼습니다.
///   Max-Pool 은 뱅크 안의 어느 한 구만 반응해도 그 필드가 이기므로,
///   다른 필드의 이름을 품은 구는 그 필드를 대신 설명합니다.
///
///  ── 같은 판정이 이미 있습니다 ──
///   query_shipping.rs 의 [TIME UNIT SELF-POISON] 이 시간 단위 뱅크에서
///   "day of month" 를 'day' 뱅크에서 끄는 것과 동일한 구조입니다.
///   그쪽은 '자기 단위 이름' 을 기준 벡터로 쓰므로, 여기서도 각 필드의
///   '대표 구'(뱅크 첫 구)를 기준으로 삼습니다.
fn purge_self_poisoned_anchors(
    banks: &[(String, Vec<Vec<f32>>, Vec<f32>)],
) -> Vec<(String, Vec<Vec<f32>>, Vec<f32>)> {
    use crate::utils::ai_utils::cosine_similarity;

    // 각 필드의 대표 벡터 = 그 필드 뱅크 전체의 중심
    let heads: Vec<Option<Vec<f32>>> = banks
        .iter()
        .map(|(_, b, _)| {
            let c = bank_centroid(b);
            if c.is_empty() { None } else { Some(c) }
        })
        .collect();
    // 중심에 가장 가까운 구가 그 뱅크를 가장 잘 대표합니다. 이 구만 정화에서 면제합니다.
    // 인덱스 0(영어)을 무조건 면제하면 다국어 뱅크에서 영어만 특권을 갖습니다.
    let anchors: Vec<usize> = banks
        .iter()
        .enumerate()
        .map(|(fi, (_, b, _))| {
            let h = match heads[fi].as_ref() { Some(h) => h, None => return 0usize };
            let mut best = (0usize, f32::MIN);
            for (pi, e) in b.iter().enumerate() {
                if e.iter().all(|&v| v == 0.0) { continue; }
                let s = cosine_similarity(e, h);
                if s > best.1 { best = (pi, s); }
            }
            best.0
        })
        .collect();

    let mut out: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = Vec::with_capacity(banks.len());
    let mut dropped: Vec<String> = Vec::new();
    // 🌟 [OVER-PURGE DIAGNOSTIC] 필드별 제거 비율을 남깁니다.
    //    SDS 의 vision.anchor_self_poison 은 총합만 기록하므로
    //    '어느 필드가 뱅크를 잃었는가' 를 로그로 복원할 수 없었습니다.
    //    D-4(총중량이 순중량에 밀림)의 원인이 이 정화인지 아닌지를
    //    다음 회차에서 판정하려면 필드 단위 수치가 반드시 있어야 합니다.
    let mut per_field: Vec<(String, usize, usize)> = Vec::new();

    for (fi, (fname, bank, weights)) in banks.iter().enumerate() {
        let own_head = match heads[fi].as_ref() { Some(h) => h, None => {
            out.push((fname.clone(), bank.clone(), weights.clone()));
            continue;
        }};
        // 🌟 [COHESION RELIEF] 절대 대소(rival > own)는 여유가 0이라,
        //    동의어가 촘촘한 뱅크일수록 구조적으로 더 많이 잘립니다.
        //    편견 게이트가 이미 같은 문제를 bank_internal_cohesion 여유로 풀었으므로
        //    그 판정기를 그대로 씁니다. 새 상수가 생기지 않고 정화가 보수적으로 바뀝니다.
        let own_cohesion = crate::utils::ai_utils::bank_internal_cohesion(bank);
        let mut kept_bank: Vec<Vec<f32>> = Vec::with_capacity(bank.len());
        let mut kept_w: Vec<f32> = Vec::with_capacity(weights.len());
        let mut cut = 0usize;

        for (pi, e) in bank.iter().enumerate() {
            if e.iter().all(|&v| v == 0.0) { continue; }
            // 중심에 가장 가까운 구는 항상 유지합니다. 빼면 뱅크가 소멸합니다.
            if pi == anchors[fi] {
                kept_bank.push(e.clone());
                kept_w.push(weights.get(pi).copied().unwrap_or(1.0));
                continue;
            }
            let own = cosine_similarity(e, own_head);
            let mut rival_name = String::new();
            let mut rival = f32::MIN;
            for (gi, h) in heads.iter().enumerate() {
                if gi == fi { continue; }
                let h = match h.as_ref() { Some(h) => h, None => continue };
                let s = cosine_similarity(e, h);
                if s > rival { rival = s; rival_name = banks[gi].0.clone(); }
            }
            if rival > f32::MIN
                && crate::utils::ai_utils::prejudice_dominates(own, rival, own_cohesion)
            {
                cut += 1;
                dropped.push(format!(
                    "{}←구{} (자기 '{}' {:.4} × (1+{:.3}) < '{}' {:.4})",
                    fname, pi, fname, own, own_cohesion.clamp(0.0, 0.5), rival_name, rival
                ));
                continue;
            }
            kept_bank.push(e.clone());
            kept_w.push(weights.get(pi).copied().unwrap_or(1.0));
        }
        if cut > 0 { per_field.push((fname.clone(), cut, bank.len())); }

        if kept_bank.is_empty() {
            out.push((fname.clone(), bank.clone(), weights.clone()));
        } else {
            out.push((fname.clone(), kept_bank, kept_w));
        }
    }

    if !dropped.is_empty() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            per_field.sort_by(|a, b| {
                (b.1 as f32 / b.2.max(1) as f32)
                    .partial_cmp(&(a.1 as f32 / a.2.max(1) as f32))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            println!(
                "      🧹 [ANCHOR SELF-POISON] 자기 필드 대표 구보다 다른 필드 대표 구가 응집도 여유까지 넘어서 설명하는 앵커 구 {}개를 그 필드 뱅크에서 끕니다: {:?} — Max-Pool 은 뱅크 안의 어느 한 구만 반응해도 그 필드가 이기므로, 다른 필드의 이름을 품은 구는 그 필드를 대신 설명해 1·2위를 뒤집습니다. (뱅크는 회차 내 불변이므로 첫 정화만 출력합니다)",
                dropped.len(),
                dropped.iter().take(8).collect::<Vec<_>>()
            );
            println!(
                "      📉 [SELF-POISON BY FIELD] 제거 비율이 높은 필드: {:?} — 한 필드가 자기 뱅크의 절반 이상을 잃으면 그 축은 이후 모든 라벨 경쟁에서 구조적으로 불리해집니다. 총합만 보면 이 편향이 보이지 않습니다.",
                per_field.iter().take(10)
                    .map(|(f, c, t)| format!("{}({}/{})", f, c, t))
                    .collect::<Vec<_>>()
            );
        }
        crate::utils::score_dynamics::record_baseline(
            "vision.anchor_self_poison",
            dropped.len() as f32,
        );
    }
    out
}

/// 🌟 [NAME CONTAINMENT] 한 필드 이름이 다른 필드 이름을 토큰 단위로 품는지 봅니다.
///
///  ── 왜 필요한가 ──
///   discriminative_anchor_verdict 의 전제는 "한 필드의 이름이 다른 필드의
///   이름을 의미적으로 포함한다"(총중량⊃중량, 소계⊃총계) 입니다.
///   그 전제가 성립할 때만 두 뱅크가 공유 성분을 갖고, 공유분을 걷어낸
///   재측정이 의미를 갖습니다.
///
///  ── 전제를 확인하지 않았을 때의 실측 ──
///   incoterms ↔ container_measurement, currency ↔ exchange_rate 처럼
///   무관한 두 축에서도 발화해, 뱅크가 작다는 이유만으로 정답을 뒤집었습니다.
///   (한 회차 5건 발화 / 5건 전부 오답 방향)
fn field_name_contains(a: &str, b: &str) -> bool {
    let toks = |s: &str| -> Vec<String> {
        s.split(|c: char| c == '_' || c == ' ' || c == '-')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    };
    let (ta, tb) = (toks(a), toks(b));
    if ta.is_empty() || tb.is_empty() || ta == tb {
        return false;
    }
    let (small, big) = if ta.len() <= tb.len() { (&ta, &tb) } else { (&tb, &ta) };
    small.iter().all(|t| big.iter().any(|x| x == t))
}

pub fn recovery_label_gate(
    label_emb: &[f32],
    field: &str,
    banks: &[(String, Vec<Vec<f32>>, Vec<f32>)],
) -> (bool, f32, f32, String) {
    if label_emb.is_empty() || banks.is_empty() {
        return (true, 0.0, 0.0, String::new());
    }
    let purged = purge_self_poisoned_anchors(banks);
    let banks: &[(String, Vec<Vec<f32>>, Vec<f32>)] = &purged;
    let mut pool: Vec<f32> = Vec::new();
    let mut per: Vec<(String, f32, usize)> = Vec::new();
    for (f, bank, w) in banks.iter() {
        let mut mx = f32::MIN;
        let mut live = 0usize;
        for (i, e) in bank.iter().enumerate() {
            if e.is_empty() || e.iter().all(|&x| x == 0.0) { continue; }
            let s = crate::utils::ai_utils::cosine_similarity(label_emb, e)
                * w.get(i).copied().unwrap_or(1.0);
            pool.push(s);
            live += 1;
            if s > mx { mx = s; }
        }
        if live == 0 { continue; }
        per.push((f.clone(), mx, live));
    }
    if per.is_empty() {
        return (false, 0.0, 0.0, String::new());
    }
    if per.len() < 2 || pool.len() < 2 {
        let own = per
            .iter()
            .find(|(f, _, _)| f.as_str() == field)
            .map(|(_, s, _)| *s)
            .unwrap_or(f32::MIN);
        let out = if own == f32::MIN { 0.0 } else { own };
        return (out > 0.0, out, 0.0, String::new());
    }
    let n = pool.len() as f32;
    let mean = pool.iter().sum::<f32>() / n;
    let sd = (pool.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n)
        .sqrt()
        .max(1e-6);
    let mut scored: Vec<(String, f32)> = per
        .iter()
        .map(|(f, mx, cnt)| {
            (
                f.clone(),
                (mx - mean) / sd - crate::utils::ai_utils::gumbel_expected_z(*cnt),
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 🌟 [DISCRIMINATIVE TIE-BREAK / 일반화]
    //
    //  ── 왜 '2위가 자기 필드일 때' 라는 조건을 뗐는가 ──
    //   구버전은 자기 필드가 2위일 때만 재검사했습니다. 그래서 argmax 자체가 틀린 경우,
    //   즉 자기 필드가 3위 이하로 밀렸거나 애초에 argmax 를 조회하는 호출(field="")에서는
    //   한 번도 발화하지 못했습니다.
    //   실측: 라벨 "TOTAL WEIGHT" 의 argmax 가 item_net_weight, 2위가 weight_net 이고
    //   정답인 weight_gross 는 둘 다에 밀렸습니다. 세 뱅크 모두 'weight' 성분을 공유하므로
    //   승패를 가른 것은 변별력이 아니라 공유분의 미세차입니다.
    //
    //  ── 왜 임계값이 필요 없는가 ──
    //   discriminative_anchor_verdict 는 두 뱅크가 공유 성분을 하나도 갖지 않으면
    //   None 을 돌려줍니다. 즉 '재검사할 이유가 있는가' 자체가 함수의 반환으로 판정되며,
    //   '얼마나 붙었을 때 재검사할지' 라는 상수를 도입할 필요가 없습니다.
    if scored.len() >= 2 {
        let a = scored[0].0.clone();
        let b = scored[1].0.clone();
        let nested = field_name_contains(&a, &b);
        let ab = banks.iter().find(|(f, _, _)| *f == a).map(|(_, x, _)| x.clone());
        let bb = banks.iter().find(|(f, _, _)| *f == b).map(|(_, x, _)| x.clone());
        if !nested {
            crate::utils::score_dynamics::record_baseline("vision.discriminative_skip", 1.0);
            println!(
                "      ⏭ [DISCRIMINATIVE SKIP] 라벨 argmax '{}' 와 2위 '{}' 는 이름이 서로를 품지 않습니다. 두 뱅크가 공유 성분을 갖는다는 근거가 없으므로 잔차 재측정을 하지 않습니다. 무관한 두 축에서 재측정하면 남은 변별 구의 개수 차이가 그대로 승패가 되어, 뱅크가 작은 쪽이 구조적으로 이깁니다.",
                a, b
            );
        } else if let (Some(ab), Some(bb)) = (ab, bb) {
            if let Some((as_, bs_, an, bn)) =
                crate::utils::ai_utils::discriminative_anchor_verdict(label_emb, &ab, &bb)
            {
                let size_fair = bn <= an;
                crate::utils::score_dynamics::record_baseline(
                    "vision.discriminative_gate",
                    if bs_ > as_ && size_fair { 1.0 } else { 0.0 },
                );
                if bs_ > as_ && !size_fair {
                    println!(
                        "      ⏭ [DISCRIMINATIVE SIZE BIAS] '{}' {:.4}({}구) 가 '{}' {:.4}({}구) 를 앞섰지만, 이긴 쪽의 변별 구가 더 많습니다. Max-Pool 최댓값은 표본 수만으로도 커지므로 이 역전은 의미 차이가 아니라 뱅크 크기 차이입니다. argmax 를 유지합니다.",
                        b, bs_, bn, a, as_, an
                    );
                } else if bs_ > as_ {
                    println!(
                        "      🔬 [DISCRIMINATIVE ANCHOR] 라벨 argmax 는 '{}'({:+.4}) 였지만, 두 뱅크의 공유 성분을 걷어내고 각자의 변별 구만으로 다시 재면 '{}' {:.4}({}구) > '{}' {:.4}({}구) 로 뒤집힙니다. 이름이 서로를 품는 두 축(총중량⊃중량, 소계⊃총계)은 뱅크 성분을 공유하므로 그 공유분의 미세차가 승패를 가릅니다. 변별 구만 남긴 쪽을 1위로 확정합니다.",
                        a, scored[0].1, b, bs_, bn, a, as_, an
                    );
                    crate::utils::score_dynamics::record_confusion(&b, &a, bs_ - as_);
                    scored.swap(0, 1);
                }
            }
        }
    }

    let own = scored
        .iter()
        .find(|(f, _)| f.as_str() == field)
        .map(|(_, z)| *z)
        .unwrap_or(f32::MIN);
    let own_out = if own == f32::MIN { 0.0 } else { own };
    if scored[0].0.as_str() == field {
        let (rf, rz) = scored
            .get(1)
            .map(|(f, z)| (f.clone(), *z))
            .unwrap_or((String::new(), 0.0));
        return (true, own_out, rz, rf);
    }
    (false, own_out, scored[0].1, scored[0].0.clone())
}

/// 🌟 [PAIR ROUTE] 읽어낸 라벨↔값 쌍 하나를 스키마 축 하나로 확정합니다.
#[derive(Debug, Clone)]
pub struct PairRoute {
    pub field: String,
    pub label: String,
    pub value: String,
    /// 라벨 뱅크 전체 경쟁에서 이 축이 얻은 중립점수
    pub own: f32,
    /// 이 창이 원래 물었던 축인가
    pub in_window: bool,
}

/// 🌟 [FULL-SCHEMA PAIR ROUTING] 창이 물은 축이 아니라 '스키마 전체' 를 상대로 라우팅합니다.
///
///  ── 구버전이 무엇을 잃었나 (실측 4건) ──
///   창 6  "INCOTERM"→"DAP"            버림 → incoterms 가 우회 경로로만 회수
///   창 7  "INVOICE TOTAL"→"2000.00"   버림 → amount 영구 소실 (정답)
///   창 8  "DATE"→"Apr-19-2022"        버림 → issue_date 영구 소실 (정답)
///   창 9  "COUNTRY OF EXPORT"→"USA"   버림
///   네 건 모두 Qwen 호출 비용을 이미 지불하고 라벨까지 정확히 읽은 뒤에 버렸습니다.
///
///  ── 왜 창 안 배타 배정을 폐기하는가 ──
///   창은 '어디를 볼지' 를 정한 좌표 근거일 뿐이고,
///   읽어낸 라벨은 '그것이 무엇인지' 를 말하는 직접 근거입니다.
///   좌표 근거로 직접 근거를 가두면, 창 안에 정답 축이 없을 때 반드시 오배정이 생깁니다.
///   실측: 창 6 은 중량 축 3개만 물었는데 "TOTAL NUMBER OF PACKAGES" 가 들어오자
///   배타 배정이 남는 축(weight_gross)에 그 값을 억지로 꽂았고,
///   그 오배정을 WINDOW ARGMAX 가 '창 안 1위' 라는 이유로 승인했습니다.
///   전체 스키마에서 argmax 를 뽑으면 그 쌍은 package_count 로 가고,
///   이미 채워져 있으므로 조용히 버려집니다 — 억지 배정이 구조적으로 사라집니다.
///
///  ── 네 가지 게이트 ──
///   G1 중립점수 양수      : 인쇄된 라벨이 어떤 축도 설명하지 못하면 버립니다.
///   G2 표 행 축 제외      : items / containers 열은 '어느 행인가' 를 말해 주는 근거가
///                           쌍 하나에는 없으므로 배정하지 않습니다.
///   G3 값 형식 일치       : query_value_format 을 씁니다. detect_field_format 은
///                           reference_number 를 Text 로 보아 "Goods Sold" 같은
///                           서술문이 참조 축에 들어오는 것을 막지 못합니다.
///   G4 축당 쌍 하나       : 인쇄된 한 자리는 라벨을 하나만 가집니다.
pub fn route_pairs_to_fields(
    pairs: &[(String, String)],
    pair_embs: &[Vec<f32>],
    window_fields: &[String],
    banks: &[(String, Vec<Vec<f32>>, Vec<f32>)],
) -> (Vec<PairRoute>, Vec<String>) {
    let mut out: Vec<PairRoute> = Vec::new();
    let mut logs: Vec<String> = Vec::new();
    if pairs.is_empty() || banks.is_empty() { return (out, logs); }
    if pair_embs.len() != pairs.len() { return (out, logs); }

    for (pi, (label, value)) in pairs.iter().enumerate() {
        let e = &pair_embs[pi];
        if e.is_empty() || e.iter().all(|&v| v == 0.0) { continue; }
        if value.trim().is_empty() { continue; }

        // field 를 빈 문자열로 넘기면 recovery_label_gate 는 argmax 축과 그 점수만 돌려줍니다.
        // 변별 앵커 재검사(패치 B-2)도 이 경로에서 함께 수행됩니다.
        let (_, _, rz, rf) = recovery_label_gate(e, "", banks);

        if rf.is_empty() {
            logs.push(format!(
                "      ⚪ [PAIR ARGMAX NONE] \"{}\" → \"{}\" | 라벨 뱅크 어디에서도 점수를 얻지 못했습니다.",
                label, value
            ));
            continue;
        }
        if rz <= 0.0 {
            logs.push(format!(
                "      🚫 [PAIR ARGMAX WEAK] \"{}\" → \"{}\" | 최강 축 '{}' 의 중립점수가 {:+.4} 로 양수가 아닙니다. 읽힌 라벨이 스키마의 어떤 축도 설명하지 못한다는 뜻이므로 버립니다. 음수 근거를 통과시키면 이 서식에 존재하지 않는 축(부가세번호 등)이 형태만 맞는 아무 축에나 들어갑니다.",
                label, value, rf, rz
            ));
            continue;
        }
        let cat = crate::logic::trade_field_category(&rf);
        if cat.is_empty() {
            logs.push(format!(
                "      🚫 [PAIR NO CATEGORY] \"{}\" → \"{}\" | 최강 축 '{}' 의 소속 카테고리를 판정할 수 없어 병합 대상에서 제외합니다.",
                label, value, rf
            ));
            continue;
        }
        if crate::logic::TRADE_ARRAY_CATEGORIES.iter().any(|c| *c == cat) {
            logs.push(format!(
                "      🚫 [PAIR ROW AXIS] \"{}\" → \"{}\" | 최강 축 '{}' 는 표 행 카테고리 '{}' 의 열입니다. 라벨↔값 쌍 하나에는 어느 행인지를 말해 주는 근거가 없어 배정할 수 없습니다. 루트에 꽂으면 표에서 읽은 행 값과 서로 다른 사실이 한 이름에 공존합니다.",
                label, value, rf, cat
            ));
            continue;
        }
        let fmt = crate::utils::ai_utils::query_value_format(&rf);
        if !crate::utils::ai_utils::value_matches_format(fmt, value) {
            logs.push(format!(
                "      🚫 [PAIR FORMAT] \"{}\" → \"{}\" | 최강 축 '{}' 가 요구하는 값 형식({:?})과 맞지 않습니다.",
                label, value, rf, fmt
            ));
            continue;
        }

        match out.iter_mut().find(|r| r.field == rf) {
            Some(prev) => {
                if rz > prev.own {
                    logs.push(format!(
                        "      ♊ [PAIR FIELD CONTEST] '{}' 를 두 쌍이 주장했습니다: \"{}\"({:+.4}) 와 \"{}\"({:+.4}). 인쇄된 한 자리는 라벨을 하나만 가지므로 점수가 높은 쪽만 남깁니다.",
                        rf, label, rz, prev.label, prev.own
                    ));
                    prev.label = label.clone();
                    prev.value = value.clone();
                    prev.own = rz;
                }
            }
            None => out.push(PairRoute {
                field: rf.clone(),
                label: label.clone(),
                value: value.clone(),
                own: rz,
                in_window: window_fields.iter().any(|f| *f == rf),
            }),
        }
    }

    out.sort_by(|a, b| b.own.partial_cmp(&a.own).unwrap_or(std::cmp::Ordering::Equal));
    (out, logs)
}

/// 🌟 [RECOVERY WINDOW MERGE] 겹치는 복구 창을 하나로 합칩니다.
///
///  ── 실측 사고 ──
///   창 6 px(514,688)-(743,860) 과 창 7 px(457,630)-(686,803) 은 서로의 중심을 품습니다.
///   같은 지면을 두 번 읽어 "TOTAL NUMBER OF PACK" 과 "TOTAL WEIGHT" 가 중복으로 왔고,
///   호출이 한 번 더 들었습니다.
///   기존 COLLIDE 검사는 픽셀 완전 일치(`*bx == w.0`)만 보므로 이 쌍을 놓칩니다.
///
///  ── 왜 버리지 않고 합치는가 ──
///   겹치는 두 창의 읽기 결과가 같지 않습니다. 창 7 만 "INVOICE TOTAL"→"2000.00" 을 읽었습니다.
///   나중 창을 버리면 정답이 통째로 사라집니다. 합치면 한 번의 읽기로 양쪽 지면을 다 봅니다.
///
///  ── 왜 임계값이 없는가 ──
///   겹침 판정은 '한쪽의 중심이 다른 쪽 안에 있는가' 라는 포함 관계이고,
///   병합 허용은 '합친 사각형이 따로 읽을 때보다 픽셀을 더 먹지 않는가' 입니다.
///   둘 다 비율 상수가 아니라 구조와 비용에서 유도됩니다.
pub fn recovery_window_merge(
    a: (u32, u32, u32, u32),
    b: (u32, u32, u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let center = |x: (u32, u32, u32, u32)| {
        ((x.0 as f32 + x.2 as f32) / 2.0, (x.1 as f32 + x.3 as f32) / 2.0)
    };
    let inside = |p: (f32, f32), x: (u32, u32, u32, u32)| {
        p.0 >= x.0 as f32 && p.0 <= x.2 as f32 && p.1 >= x.1 as f32 && p.1 <= x.3 as f32
    };
    if !(inside(center(a), b) || inside(center(b), a)) {
        return None;
    }
    let u = (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3));
    let area = |x: (u32, u32, u32, u32)| {
        (x.2.saturating_sub(x.0) as u64) * (x.3.saturating_sub(x.1) as u64)
    };
    if area(u) > area(a) + area(b) {
        return None;
    }
    Some(u)
}

/// 🌟 [SINGLE ROW WRITE] 배열 카테고리에 스칼라 값을 안전하게 기입합니다.
///
///  ── 실측 사고 ──
///   signatory_name = "John Smith" 가 other_parties(배열 카테고리)로 REROUTE 되었는데,
///   기존 코드는 `if !is_trade_array_category(rcat)` 로 분기해 루트에만 쓰고 행에는 넣지 않았습니다.
///   그 결과 저장본이
///     other_parties[0] = { party_role: "SIGNATORY COMPANY", signatory_name: null }
///     signatory_name   = "John Smith"      ← 루트에만 존재
///   로 갈렸고, 자연어 변환이 서명자와 서명 회사를 서로 다른 절로 만들었습니다.
///
///  ── 왜 행이 하나일 때만 쓰는가 ──
///   행이 여럿이면 '어느 행의 값인가' 를 말해 주는 근거가 쌍에도 라벨에도 없습니다.
///   근거 없이 첫 행에 꽂으면 두 당사자의 사실이 한 행에서 뒤섞입니다.
///   '틀린 행' 보다 '루트 고립' 이 항상 안전합니다.
pub fn write_into_single_row(
    merged: &mut serde_json::Map<String, Value>,
    category: &str,
    field: &str,
    value: &str,
) -> bool {
    let arr = match merged.get_mut(category).and_then(|v| v.as_array_mut()) {
        Some(a) => a,
        None => return false,
    };
    if arr.len() != 1 { return false; }
    let o = match arr[0].as_object_mut() { Some(o) => o, None => return false };
    let empty = o.get(field).map_or(true, |x| {
        x.is_null() || x.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
    });
    if !empty { return false; }
    o.insert(field.to_string(), json!(value));
    true
}

pub fn plan_recovery_windows(
    cands: &[(String, String, usize, f32)],
    rows: usize,
    cols: usize,
    orig_w: u32,
    orig_h: u32,
    budget: usize,
) -> Vec<((u32, u32, u32, u32), Vec<(String, String, f32)>)> {
    let rows = rows.max(1);
    let cols = cols.max(1);
    let cw = orig_w as f32 / cols as f32;
    let ch = orig_h as f32 / rows as f32;
    let to_px = |r0: usize, r1: usize, c0: usize, c1: usize| -> (u32, u32, u32, u32) {
        (
            (c0 as f32 * cw).floor() as u32,
            (r0 as f32 * ch).floor() as u32,
            (((c1 + 1) as f32 * cw).ceil() as u32).min(orig_w),
            (((r1 + 1) as f32 * ch).ceil() as u32).min(orig_h),
        )
    };
    let mut sorted: Vec<&(String, String, usize, f32)> = cands.iter().collect();
    sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    struct Slot {
        row: usize,
        c_lo: usize,
        c_hi: usize,
        fields: Vec<(String, String, f32)>,
    }
    let mut slots: Vec<Slot> = Vec::new();
    let mut merged_log: Vec<String> = Vec::new();
    let mut row_log: Vec<String> = Vec::new();
    let mut overflow: Vec<String> = Vec::new();
    for c in sorted.into_iter() {
        let (cat, field, patch, z) = (&c.0, &c.1, c.2, c.3);
        let r = patch / cols;
        let col = patch % cols;
        if r >= rows { continue; }
        let lo = col.saturating_sub(1);
        let hi = (col + 2).min(cols.saturating_sub(1));

        if let Some(s) = slots
            .iter_mut()
            .find(|s| s.row == r && lo <= s.c_hi && s.c_lo <= hi)
        {
            if s.fields.iter().any(|(_, f, _)| f == field) { continue; }
            let grew = lo < s.c_lo || hi > s.c_hi;
            if grew {
                row_log.push(format!(
                    "{}(r{} c{}~{} ⊕ c{}~{})",
                    field, r, s.c_lo, s.c_hi, lo, hi
                ));
            } else {
                merged_log.push(format!("{}(z {:+.2})", field, z));
            }
            if lo < s.c_lo { s.c_lo = lo; }
            if hi > s.c_hi { s.c_hi = hi; }
            s.fields.push((cat.clone(), field.clone(), z));
            continue;
        }
        if slots.len() >= budget {
            overflow.push(format!("{}(z {:+.2})", field, z));
            continue;
        }
        slots.push(Slot {
            row: r,
            c_lo: lo,
            c_hi: hi,
            fields: vec![(cat.clone(), field.clone(), z)],
        });
    }

    let out: Vec<((u32, u32, u32, u32), Vec<(String, String, f32)>)> = slots
        .into_iter()
        .map(|s| {
            let bbox = to_px(
                s.row.saturating_sub(1),
                (s.row + 1).min(rows.saturating_sub(1)),
                s.c_lo,
                s.c_hi,
            );
            (bbox, s.fields)
        })
        .collect();

    if !merged_log.is_empty() {
        println!(
            "    🧷 [RECOVERY WINDOW CLUSTER] 같은 픽셀 창을 가리키는 필드 {}개를 한 창에 묶었습니다: {:?} — 같은 자리를 가리키는 축들은 서로 경쟁자가 아니라 그 자리의 후보 집합입니다.",
            merged_log.len(), merged_log.iter().take(12).collect::<Vec<_>>()
        );
    }
    if !row_log.is_empty() {
        println!(
            "    🧷 [RECOVERY WINDOW ROW MERGE] 앵커 패치가 같은 격자 행이고 열 창이 겹치는 후보 {}건을 하나의 창으로 확장했습니다: {:?} — 라벨과 값은 같은 인쇄 행에 놓이므로 같은 행의 겹치는 창은 서로 다른 지면이 아니라 한 줄의 조각입니다. 행을 넘지 않으므로 창 높이가 그대로이고 업스케일 배율도 보존됩니다. 이 병합이 없으면 한 줄을 두세 번 나눠 읽어 호출만 늘고, 라벨이 잘린 조각에서는 어느 축인지 판정할 근거가 사라집니다.",
            row_log.len(), row_log.iter().take(12).collect::<Vec<_>>()
        );
        crate::utils::score_dynamics::record_baseline(
            "vision.recovery_row_merge",
            row_log.len() as f32,
        );
    }
    if !overflow.is_empty() {
        println!(
            "    ⛔ [RECOVERY BUDGET OVERFLOW] 창 예산 {}개를 넘어 이번 회차에서 제외된 필드 {}개: {:?}",
            budget, overflow.len(), overflow.iter().take(12).collect::<Vec<_>>()
        );
    }
    out
}

pub fn reroute_closed_vocab_values(
    merged: &mut serde_json::Map<String, Value>,
    doc_type: &str,
    emit: &dyn Fn(&str),
) -> usize {
    let mut owners: Vec<(String, String, Vec<String>)> = Vec::new();
    if let Some(ts) = crate::parsing::BIAS_DICT.get("trade_schema") {
        for node in [ts.get("base"), ts.get("overlay").and_then(|o| o.get(doc_type))] {
            let cats = match node.and_then(|n| n.as_object()) {
                Some(c) => c,
                None => continue,
            };
            for (cat, fields) in cats.iter() {
                if crate::logic::is_trade_array_category(cat) { continue; }
                let fm = match fields.as_object() {
                    Some(f) => f,
                    None => continue,
                };
                for (f, _) in fm.iter() {
                    if owners.iter().any(|(_, x, _)| x == f) { continue; }
                    let vocab = crate::parsing::trade_expected_vocab(cat, doc_type, f);
                    if !vocab.is_empty() {
                        owners.push((cat.clone(), f.clone(), vocab));
                    }
                }
            }
        }
    }
    if owners.is_empty() { return 0; }

    let keys: Vec<String> = merged.keys().cloned().collect();
    let mut moved = 0usize;
    for k in keys.into_iter() {
        let from_cat = crate::logic::trade_field_category(&k);
        if from_cat.is_empty() || crate::logic::is_trade_array_category(from_cat) { continue; }
        let v = match merged.get(&k).and_then(|x| x.as_str()) {
            Some(s) => s.trim().to_string(),
            None => continue,
        };
        if v.is_empty() { continue; }
        let own_vocab = owners
            .iter()
            .any(|(_, f, voc)| *f == k && voc.iter().any(|t| same_printed_token(t, &v)));
        if own_vocab { continue; }
        let hits: Vec<(String, String)> = owners
            .iter()
            .filter(|(_, f, voc)| *f != k && voc.iter().any(|t| same_printed_token(t, &v)))
            .map(|(c, f, _)| (c.clone(), f.clone()))
            .collect();
        if hits.len() != 1 { continue; }
        let (to_cat, to_field) = hits[0].clone();
        let target_filled = merged
            .get(&to_field)
            .map(|x| !(x.is_null() || x.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
            .unwrap_or(false);
        if target_filled {
            let same = merged
                .get(&to_field)
                .and_then(|x| x.as_str())
                .map_or(false, |x| same_printed_token(x, &v));
            if same {
                merged.remove(&k);
                if let Some(o) = merged.get_mut(from_cat).and_then(|x| x.as_object_mut()) {
                    o.remove(&k);
                }
                emit(&format!(
                    "  🧹 [VOCAB LEAK DROP] {}.{} = \"{}\" | 이 토큰의 소유 축 '{}' 이 이미 같은 값으로 차 있습니다. 한 자리에 인쇄된 값이 두 축에 공존하면 자연어 변환이 같은 사실을 서로 다른 절로 두 번 만듭니다. 남의 축에서 지웁니다.",
                    from_cat, k, v, to_field
                ));
                moved += 1;
            }
            continue;
        }
        merged.remove(&k);
        if let Some(o) = merged.get_mut(from_cat).and_then(|x| x.as_object_mut()) {
            o.remove(&k);
        }
        merged.insert(to_field.clone(), json!(v));
        let slot = merged
            .entry(to_cat.clone())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        if let Some(o) = slot.as_object_mut() {
            o.insert(to_field.clone(), json!(v));
        }
        emit(&format!(
            "  🔀 [VOCAB OWNER REROUTE] {}.{} = \"{}\" → {}.{} | 이 값은 '{}' 의 닫힌 어휘에만 속하는 토큰입니다. 비어 있던 소유 필드로 옮깁니다.",
            from_cat, k, v, to_cat, to_field, to_field
        ));
        moved += 1;
    }
    moved
}

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
pub fn record_claim_violations(
    claimed: &[(String, String)],
    incoming: &Value,
    category: &str,
    emit: &dyn Fn(&str),
) -> usize {
    if claimed.is_empty() {
        return 0;
    }
    fn norm(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
    }
    fn substantial(s: &str) -> bool {
        s.chars().filter(|c| c.is_alphanumeric()).count() >= 2
    }
    fn harvest(o: &serde_json::Map<String, Value>, out: &mut Vec<(String, String)>) {
        for (k, v) in o.iter() {
            let s = match v {
                Value::String(s) => s.trim().to_string(),
                Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if s.is_empty() || is_schema_echo(&s) {
                continue;
            }
            out.push((k.clone(), s));
        }
    }
    let mut scan: Vec<(String, String)> = Vec::new();
    if let Some(o) = incoming.as_object() {
        harvest(o, &mut scan);
    } else if let Some(arr) = incoming.as_array() {
        for e in arr {
            if let Some(o) = e.as_object() {
                harvest(o, &mut scan);
            }
        }
    }
    let mut hits = 0usize;
    for (field, value) in scan.iter() {
        if !substantial(value) {
            continue;
        }
        let nv = norm(value);
        for (owner, owned) in claimed.iter() {
            if owner == field {
                continue;
            }
            if norm(owned) != nv {
                continue;
            }
            hits += 1;
            crate::utils::score_dynamics::record_field_seen(field);
            crate::utils::score_dynamics::record_near_miss(field);
            crate::utils::score_dynamics::record_confusion(owner, field, 0.0);
            emit(&format!(
                "    ⚠️ [CLAIM VIOLATION] [{}] '{}' = \"{}\" 는 이미 '{}' 가 확정한 값입니다. 금지 목록으로 지시했으나 모델이 되돌려주었습니다.",
                category, field, value, owner
            ));
            break;
        }
    }
    crate::utils::score_dynamics::record_baseline("vision.claim_violation", hits as f32);
    hits
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
            let schema_shaped = !k.is_empty()
                && k.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
            if !schema_shaped {
                println!(
                    "    🚫 [CLAIM KEY SHAPE] [{}] 키 '{}' 는 스키마 필드 형식이 아닙니다. 모델이 인쇄 라벨을 키로 만든 것이므로 접지 주장에서 제외합니다.",
                    category, k
                );
                continue;
            }
            if s.chars().filter(|c| c.is_alphanumeric()).count() >= 2
                && out.iter().any(|c| c.field != *k && c.value == s)
            {
                crate::utils::score_dynamics::record_baseline("vision.value_reuse", 1.0);
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
    // 🌟 [SDS / V-1] 분모를 먼저 세웁니다.
    //
    //  ── 무엇이 문제였나 (실측) ──
    //   계측이 아래 `for r in rejected` 루프 안에만 있어서,
    //   접지에 성공한 값은 record_field_seen 조차 되지 않았습니다.
    //   그 결과 vision 스코프의 필드 통계가 seen == reject_format, assigned == 0 이 되어
    //   learned_specificity(= rejected / seen) 가 항상 1.0 이라는 거짓값을 냈습니다.
    //   (score_dynamics.json: recipient_name / party_name 이 정확히 이 모양)
    //   게다가 폐기가 0건이면 아래 조기 반환에 걸려 관측이 통째로 사라졌습니다.
    //   크롭 10개를 돌려 doc_number·amount·hs_code·package_count 를 채운 실행에서도
    //   '그 필드를 시도했다' 는 사실이 한 건도 남지 않았습니다.
    //
    //  ── 왜 함수 맨 앞인가 ──
    //   조기 반환보다 앞에 두어야 '전 값이 접지 성공' 인 정상 문서에서도
    //   분모가 쌓입니다. 정상 문서만 계속 들어오면 분모만 커지는 것이 맞습니다.
    //
    //  ── 보류를 분자·분모 어디에도 넣지 않는 이유 ──
    //   verify_claims 는 임베딩 생성 실패 / 대응 패치 없음일 때
    //   accepted=true 로 두되 사유에 '검증 보류' 를 남깁니다.
    //   이건 '통과' 가 아니라 '판정 못 함' 이므로 assigned 로 세면
    //   특이도가 반대 방향으로 왜곡됩니다. seen 에만 남깁니다.
    //
    //  ── 폐기 사유를 두 종류로 가르는 이유 ──
    //   '인쇄 라벨을 값으로 읽음' 은 라벨 어휘가 값을 이긴 사고이므로
    //   의미상 편견 게이트입니다. 나머지(접지 실패 / 여백·블러 / 출처 공백)는
    //   '값의 형태가 그 필드일 수 없다' 는 형식 게이트입니다.
    //   두 축을 나눠야 기획 T-3 이 '이 서식의 이 축은 라벨과 값이 구조적으로 겹친다' 와
    //   '이 크롭은 애초에 읽을 게 없었다' 를 구분할 수 있습니다.
    {
        let mut seen = 0usize;
        let mut kept = 0usize;
        let mut held = 0usize;
        for v in verdicts.iter() {
            crate::utils::score_dynamics::record_field_seen(&v.field);
            seen += 1;
            if v.gate == crate::models::siglip2::value_grounding::VerdictGate::Held {
                held += 1;
                continue;
            }
            if !v.accepted {
                let kind = match v.gate {
                    crate::models::siglip2::value_grounding::VerdictGate::Prejudice =>
                        crate::utils::score_dynamics::GateKind::Prejudice,
                    _ => crate::utils::score_dynamics::GateKind::Format,
                };
                crate::utils::score_dynamics::record_field_reject(&v.field, kind);
                continue;
            }
            // surprisal_out 은 크롭 밖 패치가 하나도 없을 때 f32::MIN 입니다.
            // 그대로 빼면 +inf 가 되어 Welford 의 mean/m2 를 영구히 오염시킵니다.
            let margin = if v.surprisal_out <= f32::MIN / 2.0 {
                v.surprisal_in.max(0.0)
            } else {
                (v.surprisal_in - v.surprisal_out).max(0.0)
            };
            crate::utils::score_dynamics::record_field_assigned(&v.field, margin);
            kept += 1;
        }
        if seen > 0 {
            crate::utils::score_dynamics::record_baseline(
                "vision.grounding_claims",
                seen as f32,
            );
            crate::utils::score_dynamics::record_baseline(
                "vision.grounding_accept_ratio",
                kept as f32 / seen as f32,
            );
            emit(&format!(
                "  📊 [SDS / GROUNDING] 주장 {}건 | 접지 확인 {} | 검증 보류 {} | 폐기 {}",
                seen, kept, held, seen.saturating_sub(kept + held)
            ));
        }
    }
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

pub fn trade_schema_owner_of(doc_type: &str, field: &str) -> (bool, String) {
    let ts = match crate::parsing::BIAS_DICT.get("trade_schema") {
        Some(t) => t,
        None => return (false, String::new()),
    };
    let mut known = false;
    let mut found = String::new();
    for node in [
        ts.get("overlay").and_then(|o| o.get(doc_type)),
        ts.get("base"),
    ] {
        let cats = match node.and_then(|n| n.as_object()) {
            Some(c) => c,
            None => continue,
        };
        if !cats.is_empty() {
            known = true;
        }
        if !found.is_empty() {
            continue;
        }
        for (cat, fields) in cats.iter() {
            if fields.as_object().map_or(false, |fm| fm.contains_key(field)) {
                found = cat.clone();
                break;
            }
        }
    }
    (known, found)
}

pub fn merge_extracted(
    merged: &mut serde_json::Map<String, Value>,
    category: &str,
    incoming: &Value,
    emit: &dyn Fn(&str),
) {
    let is_array_category = crate::logic::is_trade_array_category(category);
    let doc_code = merged
        .get("header")
        .and_then(|h| h.get("doc_type"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let unwrapped: Value;
    let incoming = {
        let single = incoming.as_object().filter(|o| o.len() == 1).and_then(|o| {
            let (k, v) = o.iter().next()?;
            if !(v.is_object() || v.is_array()) { return None; }
            let norm = |s: &str| -> String {
                s.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(|c| c.to_lowercase())
                    .collect()
            };
            let key = norm(k);
            if key == norm(category) || (!doc_code.is_empty() && key == norm(&doc_code)) {
                Some((k.clone(), v.clone()))
            } else {
                None
            }
        });
        match single {
            Some((k, v)) => {
                emit(&format!(
                    "    📦 [CATEGORY UNWRAP] [{}] 응답 최상위가 '{}' 한 겹으로 감싸여 있습니다. 프롬프트가 카테고리명을 대문자로 지시하므로 모델이 그 이름을 래퍼 키로 되풀이한 것입니다. 한 겹 벗겨 내용을 그대로 병합합니다. 벗기지 않으면 객체 전체가 스키마 밖 키 하나로 취급되어 그 안의 값이 전부 사라집니다.",
                    category, k
                ));
                crate::utils::score_dynamics::record_baseline("vision.category_unwrap", 1.0);
                unwrapped = v;
                &unwrapped
            }
            None => incoming,
        }
    };

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

                let row_has_identity = |v: &Value| -> bool {
                    let o = match v.as_object() { Some(o) => o, None => return false };
                    let filled = |k: &str| -> bool {
                        match o.get(k) {
                            Some(Value::String(s)) => !s.trim().is_empty() && !is_schema_echo(s),
                            Some(Value::Number(_)) => true,
                            _ => false,
                        }
                    };
                    let ids = row_identity_fields(category);
                    if ids.is_empty() { return true; }
                    ids.iter().any(|k| filled(k))
                };
                let scalars = |v: &Value| -> Vec<(String, String)> {
                    let o = match v.as_object() { Some(o) => o, None => return Vec::new() };
                    o.iter()
                        .filter_map(|(k, x)| match x {
                            Value::String(s) if !s.trim().is_empty() => {
                                Some((k.clone(), s.trim().to_lowercase()))
                            }
                            Value::Number(nn) => Some((k.clone(), nn.to_string())),
                            _ => None,
                        })
                        .collect()
                };
                let subsumes = |outer: &Value, inner: &Value| -> bool {
                    let a = scalars(outer);
                    let b = scalars(inner);
                    if b.is_empty() || b.len() > a.len() { return false; }
                    b.iter()
                        .all(|(k, v)| a.iter().any(|(ak, av)| ak == k && av == v))
                };

                let mut added = 0usize;
                let mut dup = 0usize;
                let mut subset = 0usize;
                let mut absorbed = 0usize;
                let mut ghost = 0usize;
                if let Some(existing) = slot.as_array_mut() {
                    let mut keys: Vec<String> = existing.iter().map(row_key).collect();
                    for e in arr {
                        if !row_has_identity(e) {
                            ghost += 1;
                            let id_field = if category == "items" { "description" } else { "container_number" };
                            crate::utils::score_dynamics::record_field_seen(id_field);
                            crate::utils::score_dynamics::record_field_reject(
                                id_field,
                                crate::utils::score_dynamics::GateKind::SelfId,
                            );
                            continue;
                        }
                        let k = row_key(e);
                        if k.is_empty() { continue; }
                        if keys.iter().any(|x| x == &k) { dup += 1; continue; }
                        if existing.iter().any(|x| subsumes(x, e)) {
                            subset += 1;
                            continue;
                        }
                        if let Some(pos) = existing.iter().position(|x| subsumes(e, x)) {
                            existing[pos] = e.clone();
                            keys[pos] = k;
                            absorbed += 1;
                            continue;
                        }
                        keys.push(k);
                        existing.push(e.clone());
                        added += 1;
                    }
                }
                emit(&format!(
                    "    ➕ [{}] 배열 신규 {}건 | 겹침 중복 {}건 제거 | 부분집합 {}건 흡수 | 기존 행 승격 {}건 | 정체 없는 행 {}건 폐기 (누적 {}건)",
                    category,
                    added,
                    dup,
                    subset,
                    absorbed,
                    ghost,
                    merged.get(category).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
                ));
                if subset + absorbed > 0 {
                    emit(&format!(
                        "    ♊ [ROW SUBSUMED] [{}] 스칼라 값 집합이 기존 행의 부분집합인 행 {}건과, 기존 행을 부분집합으로 품은 행 {}건을 병합했습니다. 완전 일치만 보면 같은 값을 절반만 담은 행이 새 레코드로 쌓입니다. 한 크롭이 라벨↔값을 온전히 읽고 다른 크롭이 값만 읽으면 정확히 이 모양이 되며, 그대로 두면 자연어 변환과 청크 인덱싱이 같은 사실을 두 번 문장으로 만듭니다.",
                        category, subset, absorbed
                    ));
                    crate::utils::score_dynamics::record_baseline(
                        "vision.row_subsumed",
                        (subset + absorbed) as f32,
                    );
                }
            }
            return;
        }
    };

    let mut added = 0usize;
    let mut kept = 0usize;
    let mut echoed = 0usize;
    let mut off_schema = 0usize;
    let mut off_schema_null = 0usize;

    for (k, v) in obj.iter() {
        let is_empty = v.is_null()
            || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
            || v.as_array().map(|a| a.is_empty()).unwrap_or(false);

        {
            let owner = crate::logic::trade_field_category(k);
            if owner != category {
                if is_empty {
                    off_schema_null += 1;
                    continue;
                }
                let shown: String = v
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| v.to_string())
                    .chars()
                    .take(40)
                    .collect();
                let target_empty = merged
                    .get(k)
                    .map(|x| x.is_null() || x.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                    .unwrap_or(true);
                let shape_ok = v
                    .as_str()
                    .map(|s| {
                        crate::utils::ai_utils::value_matches_format(
                            crate::utils::ai_utils::detect_field_format(k),
                            s,
                        )
                    })
                    .unwrap_or(true);
                let (schema_known, schema_cat) = if doc_code.is_empty() {
                    (false, String::new())
                } else {
                    trade_schema_owner_of(&doc_code, k)
                };
                let structural = !schema_known || (!schema_cat.is_empty() && schema_cat == owner);
                if !owner.is_empty()
                    && !crate::logic::is_trade_array_category(owner)
                    && !crate::logic::is_trade_array_category(category)
                    && target_empty
                    && shape_ok
                {
                    if structural {
                        emit(&format!(
                            "    🔀 [SCHEMA REROUTE] [{}] '{}' = \"{}\" 는 이 카테고리의 축이 아니지만, 소속 '{}' 의 그 축이 비어 있고 값 형태도 맞습니다. 크롭 사각형이 옆 칸을 물고 있으면 모델은 인쇄된 값을 정직하게 읽은 것이므로 폐기하지 않고 소유 카테고리로 옮깁니다.",
                            category, k, shown, owner
                        ));
                    } else {
                        emit(&format!(
                            "    🧾 [ROOT ONLY REROUTE] [{}] '{}' = \"{}\" 의 규칙 기반 소속은 '{}' 이지만, 이 서식('{}') 의 로드된 스키마에서 그 축은 {}. 값은 인쇄되어 있을 수 있으므로 루트에 그대로 두되 카테고리 객체에는 넣지 않습니다. 스키마에 없는 축을 그룹 안에 넣으면 자연어 변환이 존재하지 않는 절을 만들고 그 문장이 그대로 임베딩됩니다.",
                            category, k, shown, owner, doc_code,
                            if schema_cat.is_empty() {
                                "어느 카테고리에도 없습니다".to_string()
                            } else {
                                format!("'{}' 소속입니다", schema_cat)
                            }
                        ));
                    }
                    merged.insert(k.clone(), v.clone());
                    if structural {
                        let slot = merged
                            .entry(owner.to_string())
                            .or_insert_with(|| Value::Object(serde_json::Map::new()));
                        if let Some(o) = slot.as_object_mut() {
                            o.insert(k.clone(), v.clone());
                        }
                    }
                    crate::utils::score_dynamics::record_field_seen(k);
                    crate::utils::score_dynamics::record_field_assigned(k, 0.0);
                    crate::utils::score_dynamics::record_baseline(
                        "vision.schema_reroute",
                        if structural { 1.0 } else { 0.0 },
                    );
                    continue;
                }
                // 🌟 [ROOT ONLY PRESERVE] 소속이 없다고 해서 값이 틀린 것은 아닙니다.
                //
                //  ── 실측 사고 ──
                //   country_of_ultimate_destination = "Germany" 는 이 문서의 정답입니다.
                //   (인쇄된 ULTIMATE_DESTINATION 란을 정확히 읽었습니다)
                //   그런데 trade_field_category 의 규칙에도, 어느 카테고리 스키마에도
                //   그 이름이 없어 '소속: 스키마 밖' 으로 폐기되었습니다.
                //   signature_name 은 규칙 기반 소속이 있어 바로 위 ROOT ONLY REROUTE 로
                //   살아남았는데, 이 축은 그 경로에 진입조차 하지 못했습니다.
                //
                //  ── 왜 루트에만 두는가 ──
                //   카테고리 객체에 넣으면 자연어 변환이 존재하지 않는 절을 만들고
                //   그 문장이 그대로 임베딩됩니다. 루트는 그 위험이 없습니다.
                //
                //  ── 세 가지 조건을 모두 요구하는 이유 ──
                //   ① 스칼라일 것      : 배열/객체를 루트에 올리면 구조가 무너집니다.
                //   ② 형식이 맞을 것    : 모델이 만든 임의의 키가 전부 쌓이는 것을 막습니다.
                //   ③ 중복이 아닐 것    : 이미 다른 축이 같은 인쇄값을 확정했다면 복제입니다.
                let full_value = v
                    .as_str()
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| v.to_string());
                let scalar_ok = v.is_string() || v.is_number();
                let dup_owner = merged.iter().any(|(ek, ev)| {
                    ek != k
                        && ev.as_str().map(|s| same_printed_value(s, &full_value)).unwrap_or(false)
                });
                if owner.is_empty() && target_empty && shape_ok && scalar_ok && !dup_owner {
                    merged.insert(k.clone(), v.clone());
                    emit(&format!(
                        "    🧾 [ROOT ONLY PRESERVE] [{}] '{}' = \"{}\" 는 이 서식의 로드된 스키마 어느 카테고리에도 없지만, 값이 스칼라이고 형태가 그 축이 요구하는 형식과 맞으며 같은 인쇄값을 가진 다른 축도 없습니다. 인쇄되어 있을 수 있으므로 루트에만 두고 카테고리 객체에는 넣지 않습니다. 스키마에 없는 축을 그룹 안에 넣으면 자연어 변환이 존재하지 않는 절을 만들고 그 문장이 그대로 임베딩됩니다.",
                        category, k, shown
                    ));
                    crate::utils::score_dynamics::record_field_seen(k);
                    crate::utils::score_dynamics::record_field_assigned(k, 0.0);
                    crate::utils::score_dynamics::record_baseline("vision.root_only_preserve", 1.0);
                    continue;
                }

                off_schema += 1;
                emit(&format!(
                    "    🚫 [SCHEMA WHITELIST] [{}] '{}' = \"{}\" 는 이 카테고리의 축이 아닙니다 (소속: {}). 소유 축이 이미 차 있거나 값 형태가 그 축과 맞지 않아 폐기합니다.",
                    category, k, shown,
                    if owner.is_empty() { "스키마 밖" } else { owner }
                ));
                crate::utils::score_dynamics::record_field_seen(k);
                crate::utils::score_dynamics::record_field_reject(
                    k,
                    crate::utils::score_dynamics::GateKind::Format,
                );
                continue;
            }
        }

        if is_empty { continue; }

        // ① 빈 값 / 스키마 에코는 덮지 않습니다.
        if let Some(s) = v.as_str() {
            if is_schema_echo(s) {
                echoed += 1;
                emit(&format!(
                    "    🚫 [SCHEMA ECHO] [{}] '{}' = \"{}\" 는 프롬프트 플레이스홀더 복사이므로 폐기합니다.",
                    category, k, s
                ));
                crate::utils::score_dynamics::record_field_seen(k);
                crate::utils::score_dynamics::record_field_reject(
                    k,
                    crate::utils::score_dynamics::GateKind::Format,
                );
                continue;
            }
        }
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
        crate::utils::score_dynamics::record_field_seen(k);
        if newly_added {
            crate::utils::score_dynamics::record_field_assigned(k, 0.0);
        }

        if newly_added && !crate::logic::is_trade_array_category(category) {
            let slot = merged
                .entry(category.to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if let Some(o) = slot.as_object_mut() {
                o.insert(k.clone(), v.clone());
            }
        }
    }

    emit(&format!(
        "    ✅ [{}] 신규 {}건 | 기존 유지 {}건 | 스키마 에코 폐기 {}건 | 스키마 밖 키 폐기 {}건 (빈 값 {}건은 조용히 무시)",
        category, added, kept, echoed, off_schema, off_schema_null
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
                    if v.is_null() { continue; }
                    if let Some(s) = v.as_str() {
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