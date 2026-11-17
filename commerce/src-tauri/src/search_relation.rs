//! 질의 속 고정점 → 목표 유형 관계 단계의 규칙(유형 · 고정점 · 연결 · 서식 질의 근거)은 crates/logis-search 로 옮겼습니다.
//! 저장소(LanceDB) · 임베딩 모델이 필요한 단계 함수만 여기 남습니다. 예전 경로(crate::search_relation)를 그대로 씁니다.
pub use logis_search::relation::*;

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::scheduler::normalize_entity_key;
use crate::store::VectorStore;
use crate::utils::canonical::{is_relay_placeholder, relay_edges, relay_key_for_type, relay_type_family};

pub async fn trade_relation_join(
    store: &VectorStore,
    results: &mut Vec<Value>,
    plans: &mut Vec<Value>,
    query: &str,
    language: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> RelationReport {
    let mut rep = RelationReport::default();
    let lang = crate::stanza::detect_query_lang(query, language).unwrap_or("en");
    let mut target = match trade_target(query, lang) {
        Some(t) => t,
        None => return rep,
    };
    let all_types = all_trade_types();
    let ids = query_id_tokens(query);
    let id_set: HashSet<&str> = ids.iter().map(|(_, n)| n.as_str()).collect();

    let rows = match store.get_all_items("items", 4000, 0, Some("mode = 'shipping'".to_string())).await {
        Ok(r) => r,
        Err(_) => return rep,
    };
    let mut docs: Vec<(String, Value)> = Vec::new();
    for r in rows.into_iter() {
        if let Ok(v) = serde_json::from_str::<Value>(&r.json_data) {
            if !is_relay_placeholder(&v) {
                docs.push((r.id, v));
            }
        }
    }
    let code_of = |d: &Value| -> String {
        let t = text_of(d.get("doc_type"));
        if t.is_empty() { text_of(d.get("type")).to_uppercase() } else { t.to_uppercase() }
    };
    // 목표 서식이 저장소에 한 건도 없으면(발주서 · 신용장은 번호로만 존재) 서식 전체로 봅니다.
    if !target.iter().any(|c| c == "*") && !docs.iter().any(|(_, d)| target.iter().any(|c| c.eq_ignore_ascii_case(&code_of(d)))) {
        target = vec!["*".to_string()];
    }
    let any = target.iter().any(|c| c == "*");
    rep.target_type = target.join("/");
    let wants = |d: &Value| any || target.iter().any(|c| c.eq_ignore_ascii_case(&code_of(d)));

    let mut tiered: Vec<(usize, usize)> = Vec::new();
    let mut anchors = 0usize;
    if !id_set.is_empty() {
        let mut by_index: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut referrers: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        for (i, (_, d)) in docs.iter().enumerate() {
            if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
                by_index.entry(ix as u32).or_default().push(i);
            }
            for (key, ix) in relay_edges(d) {
                referrers.entry((key, ix)).or_default().push(i);
            }
        }
        for (i, (_, d)) in docs.iter().enumerate() {
            let own = ["doc_number", "no"].iter().map(|k| normalize_entity_key(&text_of(d.get(*k)))).find(|k| id_set.contains(k.as_str()));
            if own.is_some() {
                anchors += 1;
                if wants(d) {
                    tiered.push((0, i));
                }
                // 정방향 rel_* · 역방향(이 서식을 rel_<코드> 로 가리키는 서식)
                for (_, ix) in relay_edges(d) {
                    for &j in by_index.get(&ix).into_iter().flatten() {
                        if wants(&docs[j].1) {
                            tiered.push((1, j));
                        }
                    }
                }
                if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
                    let key = format!("rel_{}", code_of(d).to_lowercase());
                    for &j in referrers.get(&(key, ix as u32)).into_iter().flatten() {
                        if wants(&docs[j].1) {
                            tiered.push((1, j));
                        }
                    }
                }
                continue;
            }
            // 번호를 reference_* 칸에 적은 서식 (발주서 · 신용장처럼 자기 문서가 없는 번호도 여기서 잇습니다)
            let referenced = d
                .as_object()
                .map(|o| o.iter().any(|(k, v)| k.starts_with("reference_") && id_set.contains(normalize_entity_key(&text_of(Some(v))).as_str())))
                .unwrap_or(false);
            if referenced && wants(d) {
                tiered.push((1, i));
            }
        }
    }
    // 값 고정점: 항구 · 당사자 · 선박 이름이 질의에 그대로 나오는 서식 (쉼표 앞 첫 마디: 'Felixstowe, United Kingdom' → felixstowe)
    // v2 질의 R7('6월에 펠릭스토우에서 선적된 서류') · R5('株式会社ひなた工房 에서 보낸 B/L')는 번호가 없어 고정점이 없었습니다.
    // 🌟 [PHONETIC ANCHOR] 라틴 문자 값이 질의 언어로 음역되어 나오면(フェリクストウ · 费利克斯托 · فيليكستو ↔ Felixstowe) 자음 골격으로 잇습니다
    //    ([`phonetic_value_hit`]). v2 질의 R7 일본어 · 중국어 · 아랍어는 이 고정점이 없어 MRR 0.33 · 0.20 · 0.13 이었습니다.
    let query_norm = norm_text(query);
    let pspans = phonetic_spans(query);
    let mut phonetic: HashMap<String, Option<(String, f32)>> = HashMap::new();
    let mut item_anchors = 0usize;
    for (i, (_, d)) in docs.iter().enumerate() {
        let hit = ["pol", "pod", "sender_name", "recipient_name", "vessel"].iter().any(|k| {
            let v = text_of(d.get(*k));
            let head = norm_text(v.split(',').next().unwrap_or(""));
            if name_like(&head) && contains_value(&query_norm, &head) {
                return true;
            }
            if pspans.is_empty() || head.is_empty() {
                return false;
            }
            phonetic.entry(head.clone()).or_insert_with(|| phonetic_value_hit(&pspans, &head)).is_some()
        });
        // 🌟 [ITEM ANCHOR] 품목 이름이 질의에 그대로 나오는 서식 ('세라믹 식칼 품목이 실린 선하증권'). 값 고정점이 항구 · 당사자 · 선박만 보아
        //    v2 질의 R6 한국어는 저장 품목 이름이 질의에 그대로 있는데도 고정점이 없었습니다 (MRR 0.17). 글자 일치만 씁니다 — 번역된
        //    품목 이름은 임베딩으로 판별되지 않았습니다 (탐침: 열 센터링을 해도 맞는 품목이 1위가 아님).
        let item_hit = !hit && trade_item_names(d).iter().any(|n| name_like(n) && contains_value(&query_norm, n));
        if hit || item_hit {
            anchors += 1;
            if item_hit {
                item_anchors += 1;
            }
            if wants(d) {
                tiered.push((2, i));
            }
        }
    }
    if item_anchors > 0 {
        emit(&format!("[AI-SEARCH] 🧭 [ITEM ANCHOR] 질의에 그대로 나온 품목 이름을 가진 서식 {}건을 고정점으로 씁니다.", item_anchors));
    }
    let heard: Vec<String> = phonetic
        .iter()
        .filter_map(|(head, h)| h.as_ref().map(|(span, s)| format!("'{}' ← '{}' ({:.2})", head, span, s)))
        .collect();
    if !heard.is_empty() {
        emit(&format!(
            "[AI-SEARCH] 🧭 [PHONETIC ANCHOR] 질의에 음역되어 나온 서류 값 {:?} (자음 골격 일치 ≥ {:.2}, 골격 {}개 이상)을 고정점으로 씁니다.",
            heard, PHONETIC_ANCHOR_PASS, PHONETIC_ANCHOR_MIN_SKELETON
        ));
    }
    let targets: Vec<usize> = order_targets(tiered, &docs, results, query_period(query));
    rep.anchors = anchors;
    rep.targets = targets.len();

    // ── 계획: 번호가 풀린 조건(doc_number · hub_reference · reference_* = 번호)을 빼고, 유형은 목표 서식으로 ──
    let mut consumed: Vec<String> = Vec::new();
    let mut widened = 0usize;
    for p in plans.iter_mut() {
        if !targets.is_empty() {
            if let Some(conds) = p.get_mut("conditions").and_then(|c| c.as_array_mut()) {
                conds.retain(|c| {
                    let v = cond_value_key(&c.get("value").cloned().unwrap_or(Value::Null));
                    let explained = !v.is_empty() && v.chars().count() >= 3 && id_set.iter().any(|t| t.contains(v.as_str()));
                    if explained {
                        consumed.push(format!("{} = {}", c.get("path").and_then(|x| x.as_str()).unwrap_or(""), v));
                    }
                    !explained
                });
            }
        }
        if let Some(o) = p.as_object_mut() {
            let types: Vec<String> = if any { all_types.clone() } else { upper_codes(&target) };
            let before = o.get("types").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            if !targets.is_empty() || any {
                if before != types.len() {
                    widened += 1;
                }
                o.insert("types".to_string(), json!(types));
            }
            if !targets.is_empty() {
                o.insert(
                    "relation".to_string(),
                    json!({ "target_type": target, "target_ids": targets.iter().map(|&j| docs[j].0.clone()).collect::<Vec<_>>() }),
                );
            }
        }
    }
    rep.consumed = consumed.len();
    if targets.is_empty() {
        if widened > 0 {
            emit(&format!(
                "[AI-SEARCH] 🧭 [TRADE RELATION] 질의의 머리가 서식 전체를 뜻하는 이름이라 해석기가 좁힌 유형을 서식 전체({}종)로 되돌립니다.",
                all_types.len()
            ));
        }
        return rep;
    }

    // ── 결과: 목표 서식을 맨 앞으로 (결과에 없으면 다시 읽어 넣음) ──
    let top = results.iter().filter_map(|r| r.get("score").and_then(|v| v.as_f64())).fold(0.0f64, f64::max).max(1.0);
    let pos: HashMap<String, usize> =
        results.iter().enumerate().filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s.to_string(), i))).collect();
    let mut rescued = 0usize;
    for (k, &j) in targets.iter().enumerate() {
        let (id, d) = &docs[j];
        let score = top + 2.0 - (k as f64) * 1e-4;
        match pos.get(id) {
            Some(&r) => {
                if let Some(o) = results[r].as_object_mut() {
                    o.insert("score".to_string(), json!(score));
                    o.insert("relation".to_string(), json!("relation_target"));
                }
            }
            None => {
                results.push(json!({ "id": id, "text": d.to_string(), "score": score, "context_type": code_of(d), "doc_type": code_of(d), "relation": "relation_target" }));
                rescued += 1;
            }
        }
    }
    rep.rescued = rescued;
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    emit(&format!(
        "[AI-SEARCH] 🧭 [TRADE RELATION] 목표 서식 {:?} · 번호 {:?} → 고정 서식 {}건, 목표 서식 {}건을 맨 앞으로 (재회수 {}건). 번호가 풀린 계획 조건 {}건을 뺐습니다: {:?}",
        target,
        ids.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>(),
        anchors,
        targets.len(),
        rescued,
        consumed.len(),
        consumed.iter().take(6).collect::<Vec<_>>()
    ));
    rep
}

/// 🌟 [CROSS-LINGUAL ANCHOR] 후보 (범위, 제목) 중 질의가 번역해 부른 것들: (같은 범위 · 같은 제목인 후보 위치들, 판정, 근거 조각).
/// 같은 제목이 여러 문서(목록 행 · 상세 페이지)에 있어도 한 후보로 겨룹니다 (안 그러면 1위와 2위가 같은 제목이라 차가 0).
/// 임베딩을 못 하면 빈 목록.
pub async fn cross_lingual_title_anchors(
    model: &crate::model::LogisModel,
    query: &str,
    candidates: &[(String, String)],
) -> Vec<(Vec<usize>, CrossAnchor, String)> {
    let spans = query_value_spans(query);
    let mut unique: Vec<(String, String)> = Vec::new();
    for (g, t) in candidates.iter() {
        if !unique.iter().any(|(ug, ut)| ug == g && ut == t) {
            unique.push((g.clone(), t.clone()));
        }
    }
    if unique.len() < 2 || spans.is_empty() {
        return Vec::new();
    }
    let Ok(bank) = model.get_embedding_batch_pinned(structure_bank().to_vec()).await else { return Vec::new() };
    let Ok(span_vecs) = model.get_embedding_batch(spans.clone()).await else { return Vec::new() };
    let Ok(title_vecs) = model.get_embedding_batch(unique.iter().map(|(_, t)| t.clone()).collect()).await else { return Vec::new() };
    let bank: Vec<Vec<f32>> = bank.iter().map(|v| unit(v)).collect();
    let titles: Vec<Vec<f32>> = title_vecs.iter().map(|v| unit(v)).collect();
    let cal = crate::models::embedding::calibrate_cos;
    let mut sim: Vec<Vec<f32>> = Vec::with_capacity(spans.len());
    let mut span_bank: Vec<f32> = Vec::with_capacity(spans.len());
    for v in span_vecs.iter() {
        let s = unit(v);
        sim.push(titles.iter().map(|t| cal(dot(&s, t))).collect());
        span_bank.push(cal(bank.iter().map(|b| dot(&s, b)).fold(f32::MIN, f32::max)));
    }
    let groups: Vec<String> = unique.iter().map(|(g, _)| g.clone()).collect();
    cross_anchor_pick(&sim, &span_bank, &groups)
        .into_iter()
        .map(|a| {
            let (g, t) = &unique[a.title];
            let members: Vec<usize> = (0..candidates.len()).filter(|&i| candidates[i].0 == *g && candidates[i].1 == *t).collect();
            let s = spans[a.span].clone();
            (members, a, s)
        })
        .collect()
}

pub async fn relation_join(
    store: &VectorStore,
    model: &crate::model::LogisModel,
    results: &mut Vec<Value>,
    plans: &mut Vec<Value>,
    query: &str,
    language: &str,
    cc: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> RelationReport {
    let mut rep = RelationReport::default();
    if plans.iter().any(|p| p.get("relay").and_then(|r| r.get("applied")).and_then(|v| v.as_bool()) == Some(true)) {
        return rep;
    }
    let lang = crate::stanza::detect_query_lang(query, language).unwrap_or("en");
    let mentions = type_mentions(query);
    let target = match target_type(&mentions, lang) {
        Some(t) => t,
        None => return rep,
    };
    let target_fam = relay_type_family(target);
    rep.target_type = target.to_string();
    let mentioned: HashSet<String> = mentions.iter().map(|(_, t)| relay_type_family(t)).collect();

    // ── 범위 안 커머스 문서 ──
    let mut filter = "mode = 'commerce'".to_string();
    if !cc.trim().is_empty() {
        filter.push_str(&format!(" AND `cc` = '{}'", cc.replace('\'', "''")));
    }
    let rows = match store.get_all_items("items", 4000, 0, Some(filter)).await {
        Ok(r) => r,
        Err(_) => return rep,
    };
    let mut docs: Vec<(String, Value)> = Vec::new();
    for r in rows.into_iter() {
        if let Ok(v) = serde_json::from_str::<Value>(&r.json_data) {
            if !is_relay_placeholder(&v) {
                docs.push((r.id, v));
            }
        }
    }
    let fam_of: Vec<String> = docs.iter().map(|(_, d)| relay_type_family(&text_of(d.get("type")))).collect();
    let mut by_index: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut referrers: HashMap<(String, u32), Vec<usize>> = HashMap::new();
    for (i, (_, d)) in docs.iter().enumerate() {
        if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
            by_index.entry(ix as u32).or_default().push(i);
        }
        for (key, ix) in relay_edges(d) {
            referrers.entry((key, ix)).or_default().push(i);
        }
    }

    // ── 고정점 ──
    let query_norm = norm_text(query);
    let ids = query_id_tokens(query);
    let id_set: HashSet<&str> = ids.iter().map(|(_, n)| n.as_str()).collect();
    let mut anchors: Vec<Anchor> = Vec::new();
    for (i, (_, d)) in docs.iter().enumerate() {
        if !id_set.is_empty() {
            if let Some(hit) = own_ids(d).into_iter().find(|k| id_set.contains(k.as_str())) {
                anchors.push(Anchor { doc: i, kind: "id", matched: hit, primary: true });
                continue;
            }
        }
        let person = PERSON_KEYS
            .iter()
            .map(|k| norm_text(&text_of(d.get(*k))))
            .find(|v| name_like(v) && contains_value(&query_norm, v));
        if let Some(name) = person {
            anchors.push(Anchor { doc: i, kind: "person", matched: name, primary: false });
            continue;
        }
        if fam_of[i] != "order" {
            let title = norm_text(&text_of(d.get("title")));
            if name_like(&title) && contains_value(&query_norm, &title) {
                anchors.push(Anchor { doc: i, kind: "title", matched: title, primary: false });
            }
        }
    }
    // 상태: 계획의 상태 조건, 없으면 질의의 상태 낱말. 목표가 아닌 유형 낱말이 가리키는 문서에만 겁니다.
    let mut status_codes: Vec<i64> = Vec::new();
    for p in plans.iter() {
        for c in p.get("conditions").and_then(|v| v.as_array()).into_iter().flatten() {
            if c.get("path").and_then(|v| v.as_str()) == Some("data.status") && c.get("op").and_then(|v| v.as_str()) == Some("eq") {
                if let Some(code) = c.get("value").and_then(|v| v.as_i64()).filter(|c| *c != 0) {
                    if !status_codes.contains(&code) {
                        status_codes.push(code);
                    }
                }
            }
        }
    }
    if status_codes.is_empty() {
        status_codes = query_status_codes(query);
    }
    let status_fams: Vec<String> = mentioned.iter().filter(|f| **f != target_fam).cloned().collect();
    if !status_codes.is_empty() && !status_fams.is_empty() {
        for (i, (_, d)) in docs.iter().enumerate() {
            if status_fams.contains(&fam_of[i]) && status_codes.contains(&status_code(d)) && !anchors.iter().any(|a| a.doc == i) {
                anchors.push(Anchor { doc: i, kind: "status", matched: status_code(d).to_string(), primary: true });
            }
        }
    }
    // 🌟 [CROSS-LINGUAL ANCHOR] 번호 · 사람 · 제목 · 상태 고정점이 하나도 없으면, 질의 언어로 번역해 부른 상품 · 행사 · 쿠폰 제목을 찾습니다.
    //    v2 질의 C4('orders in April that included Fleece-lined zip hoodie' ↔ 저장 제목 '기모 후드집업').
    //    그 조각은 고정점이 풀어낸 값입니다: 제목 · 이름 칸 조건의 값이 조각과 겹치면 아래에서 뺍니다.
    let mut cross_spans: Vec<String> = Vec::new();
    if anchors.is_empty() {
        let cand_docs: Vec<usize> = (0..docs.len())
            .filter(|&i| matches!(fam_of[i].as_str(), "goods" | "event" | "coupon") && !text_of(docs[i].1.get("title")).is_empty())
            .collect();
        let cands: Vec<(String, String)> =
            cand_docs.iter().map(|&i| (text_of(docs[i].1.get("cc")), text_of(docs[i].1.get("title")))).collect();
        let picked = cross_lingual_title_anchors(model, query, &cands).await;
        if !picked.is_empty() {
            emit(&format!(
                "[AI-SEARCH] 🧭 [CROSS-LINGUAL ANCHOR] 질의 조각이 번역해 부른 제목 {:?} 을 고정점으로 씁니다.",
                picked
                    .iter()
                    .map(|(m, a, span)| format!("'{}' ← '{}' (보정 코사인 {:.3} · 2위와 {:+.3} · 구조 낱말과 {:+.3})", cands[m[0]].1, span, a.cos, a.margin, a.bank_gap))
                    .collect::<Vec<_>>()
            ));
        }
        for (members, _, span) in picked.into_iter() {
            for m in members {
                anchors.push(Anchor { doc: cand_docs[m], kind: "title", matched: norm_text(&cands[m].1), primary: false });
            }
            cross_spans.push(norm_text(&span));
        }
    }
    if anchors.is_empty() {
        return rep;
    }
    for a in anchors.iter_mut() {
        let f = &fam_of[a.doc];
        a.primary = a.primary || *f == target_fam || mentioned.contains(f);
    }
    rep.anchors = anchors.len();

    // ── 연결: 고정점 → 목표 (단계: 0 목표 유형 고정점, 1 한 단계, 2 고정점을 가리키는 문서를 거친 두 단계; 보조 고정점은 +3) ──
    // 취소 · 환불 · 반품 문서를 거친 경로는 한 단계 뒤로 둡니다 (질의가 그 상태를 묻지 않을 때). v2 질의 C3 프랑스어
    // 'Felix Wagner 가 주문한 상품' 은 취소된 주문의 상품이 1위였습니다 — relay_ledger 기간 조인과 같은 규칙입니다.
    let voided = |d: &Value| {
        let c = status_code(d);
        void_status(c) && !status_codes.contains(&c)
    };
    let mut tiered: Vec<(usize, usize)> = Vec::new();
    let add = |tiered: &mut Vec<(usize, usize)>, tier: usize, j: usize| {
        if fam_of[j] == target_fam {
            tiered.push((tier, j));
        }
    };
    for a in anchors.iter() {
        let base = if a.primary { 0 } else { 3 };
        let d = &docs[a.doc].1;
        if fam_of[a.doc] == target_fam {
            add(&mut tiered, base, a.doc);
            continue;
        }
        let pen = voided(d) as usize;
        // 정방향: 고정점의 연결 칸 → 목표
        for (key, ix) in relay_edges(d) {
            if relay_type_family(&key) == target_fam {
                for &j in by_index.get(&ix).into_iter().flatten() {
                    add(&mut tiered, base + 1 + pen, j);
                }
            }
        }
        // 역방향: 목표가 고정점을 가리킴 · 두 단계: 고정점을 가리키는 문서의 연결 칸 → 목표
        let own_ix = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)).map(|n| n as u32);
        if let (Some(key), Some(ix)) = (relay_key_for_type(&text_of(d.get("type"))), own_ix) {
            for &r in referrers.get(&(key.to_string(), ix)).into_iter().flatten() {
                if fam_of[r] == target_fam {
                    add(&mut tiered, base + 1 + pen, r);
                    continue;
                }
                if fam_of[r] == fam_of[a.doc] {
                    continue;
                }
                let pen2 = pen + voided(&docs[r].1) as usize;
                for (k2, ix2) in relay_edges(&docs[r].1) {
                    if relay_type_family(&k2) == target_fam {
                        for &j in by_index.get(&ix2).into_iter().flatten() {
                            add(&mut tiered, base + 2 + pen2, j);
                        }
                    }
                }
            }
        }
    }
    // ── 기간 다리: 질의가 고정점의 기간을 말하면('X 기간에 팔린 상품' · 'sold during the X promotion') 고정점(행사 · 쿠폰)의
    //    유효 기간 안에 날짜가 든 거래 문서(주문, search_bridge.sales_to_order)를 거쳐 목표로 건넙니다. 같은 단계 안에서는
    //    두 경로가 모두 닿은 목표 → 기간 경로만 닿은 목표 → 고정점이 직접 가리키기만 한 목표(행사 진열 상품) 순입니다.
    //    v2 질의 C7 은 진열 상품 다섯 중 기간에 안 팔린 하나가 절반의 언어에서 1위였습니다. ('期間限定クーポン X を使った注文' 처럼
    //    기간 낱말이 이름에 붙은 질의는 쿠폰을 쓴 주문이 두 경로 모두에 닿아 그대로 앞입니다.)
    let mut bridged: Vec<(usize, usize)> = Vec::new();
    let mut bridge_orders = 0usize;
    if mentions_period(query) {
        let via_fam = crate::parsing::BIAS_DICT
            .get("search_bridge")
            .and_then(|sb| sb.get("sales_to_order"))
            .and_then(|n| n.get("target_domain"))
            .and_then(|v| v.as_str())
            .map(relay_type_family)
            .unwrap_or_else(|| "order".to_string());
        for a in anchors.iter() {
            let Some((start, end)) = validity_period(&docs[a.doc].1) else { continue };
            if fam_of[a.doc] == via_fam {
                continue;
            }
            let base = if a.primary { 0 } else { 3 };
            for (j, (_, d)) in docs.iter().enumerate() {
                if fam_of[j] != via_fam || !doc_dates(d).iter().any(|x| *x >= start && *x <= end) {
                    continue;
                }
                bridge_orders += 1;
                let tier = base + 1 + voided(d) as usize;
                if fam_of[j] == target_fam {
                    bridged.push((tier, j));
                    continue;
                }
                for (key, ix) in relay_edges(d) {
                    if relay_type_family(&key) == target_fam {
                        for &g in by_index.get(&ix).into_iter().flatten() {
                            if fam_of[g] == target_fam {
                                bridged.push((tier, g));
                            }
                        }
                    }
                }
            }
        }
        if !bridged.is_empty() {
            let best = |list: &[(usize, usize)]| {
                let mut m: HashMap<usize, usize> = HashMap::new();
                for &(t, j) in list {
                    let e = m.entry(j).or_insert(t);
                    *e = (*e).min(t);
                }
                m
            };
            let direct = best(&tiered);
            let period = best(&bridged);
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for (&j, &bt) in period.iter() {
                match direct.get(&j) {
                    Some(&dt) => merged.push((dt.min(bt) * 3, j)),
                    None => merged.push((bt * 3 + 1, j)),
                }
            }
            for (&j, &dt) in direct.iter() {
                if !period.contains_key(&j) {
                    merged.push((dt * 3 + 2, j));
                }
            }
            emit(&format!(
                "[AI-SEARCH] 🧭 [RELATION / PERIOD] 질의가 고정점의 기간을 말해 유효 기간 안의 거래 문서 {}건을 거친 목표 {}건을 잇습니다 (두 경로 모두 {}건 → 기간 경로만 {}건 → 직접 경로만 {}건).",
                bridge_orders,
                period.len(),
                period.keys().filter(|j| direct.contains_key(j)).count(),
                period.keys().filter(|j| !direct.contains_key(j)).count(),
                direct.keys().filter(|j| !period.contains_key(j)).count()
            ));
            tiered = merged;
        }
    }
    let targets: Vec<usize> = order_targets(tiered, &docs, results, query_period(query));
    if targets.is_empty() {
        emit(&format!(
            "[AI-SEARCH] ⚪ [RELATION] 목표 '{}' · 고정점 {}건(첫 {:?})에서 목표 유형으로 이어진 문서가 없어 결과를 바꾸지 않습니다.",
            target,
            anchors.len(),
            anchors.iter().take(3).map(|a| format!("{}:{}", a.kind, a.matched)).collect::<Vec<_>>()
        ));
        return rep;
    }
    rep.targets = targets.len();

    // ── 결과: 목표 → 주 고정점 → 나머지 ──
    let top = results.iter().filter_map(|r| r.get("score").and_then(|v| v.as_f64())).fold(0.0f64, f64::max).max(1.0);
    let pos: HashMap<String, usize> =
        results.iter().enumerate().filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s.to_string(), i))).collect();
    let primary_anchors: Vec<usize> = anchors.iter().filter(|a| a.primary && fam_of[a.doc] != target_fam).map(|a| a.doc).collect();
    let lift = |results: &mut Vec<Value>, j: usize, score: f64, relation: &str, rescued: &mut usize| {
        let (id, d) = &docs[j];
        match pos.get(id) {
            Some(&k) => {
                if let Some(o) = results[k].as_object_mut() {
                    let cur = o.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if score > cur {
                        o.insert("score".to_string(), json!(score));
                        o.insert("relation".to_string(), json!(relation));
                    }
                }
            }
            None => {
                results.push(json!({
                    "id": id,
                    "text": d.to_string(),
                    "score": score,
                    "context_type": text_of(d.get("type")),
                    "doc_type": text_of(d.get("type")),
                    "relation": relation
                }));
                *rescued += 1;
            }
        }
    };
    let mut rescued = 0usize;
    for (k, &j) in targets.iter().enumerate() {
        lift(results, j, top + 2.0 - (k as f64) * 1e-4, "relation_target", &mut rescued);
    }
    for (k, &j) in primary_anchors.iter().take(20).enumerate() {
        if !targets.contains(&j) {
            lift(results, j, top + 1.0 - (k as f64) * 1e-4, "relation_anchor", &mut rescued);
        }
    }
    rep.rescued = rescued;
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── 계획: 고정점이 풀어낸 조건을 뺍니다 ──
    // 제목 · 이름 칸 조건의 값이 고정점 이름의 조각이면('준비전' ⊂ '여름 휴가 준비전') 그 값은 고정점에서 온 것입니다. v2 보류 질의
    // C6 프랑스어는 행사 이름 조각이 상품 제목 조건으로 남아 목표 상품이 모두 걸러지고 행사 문서만 보였습니다. 색 같은 다른 칸은 그대로 둡니다.
    let anchor_texts: HashSet<String> = anchors.iter().filter(|a| a.kind != "id").map(|a| a.matched.clone()).collect();
    let name_fragment = |path: &str, s: &str| {
        let v = norm_text(s);
        let long_enough = if v.chars().any(glued_script) { v.chars().count() >= 2 } else { v.chars().count() >= 3 };
        (path.ends_with("title") || path.ends_with("name"))
            && long_enough
            && (anchor_texts.iter().any(|a| contains_value(a, &v))
                || cross_spans.iter().any(|s| contains_value(s, &v) || contains_value(&v, s)))
    };
    let anchor_status: HashSet<i64> = anchors.iter().filter(|a| a.kind == "status").filter_map(|a| a.matched.parse::<i64>().ok()).collect();
    let mut consumed: Vec<String> = Vec::new();
    for p in plans.iter_mut() {
        let plan_fam = relay_type_family(p.get("type").and_then(|v| v.as_str()).unwrap_or(""));
        if let Some(conds) = p.get_mut("conditions").and_then(|c| c.as_array_mut()) {
            conds.retain(|c| {
                let value = c.get("value").cloned().unwrap_or(Value::Null);
                let path = c.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let explained = plan_fam != target_fam
                    || id_set.contains(cond_value_key(&value).as_str())
                    || value.as_str().map_or(false, |s| anchor_texts.contains(&norm_text(s)) || name_fragment(&path, s))
                    || (path == "data.status" && value.as_i64().map_or(false, |v| anchor_status.contains(&v)));
                if explained {
                    consumed.push(format!("{} {} {}", path, c.get("op").and_then(|v| v.as_str()).unwrap_or(""), value));
                }
                !explained
            });
        }
        if let Some(o) = p.as_object_mut() {
            o.insert(
                "relation".to_string(),
                json!({
                    "target_type": target,
                    "anchors": anchors.iter().take(12).map(|a| json!({ "id": docs[a.doc].0, "kind": a.kind, "matched": a.matched, "primary": a.primary })).collect::<Vec<_>>(),
                    "target_ids": targets.iter().map(|&j| docs[j].0.clone()).collect::<Vec<_>>()
                }),
            );
        }
    }
    rep.consumed = consumed.len();
    emit(&format!(
        "[AI-SEARCH] 🧭 [RELATION] 목표 '{}' ({} 머리 규칙, 유형 낱말 {:?}) · 고정점 {}건 {:?} → 목표 문서 {}건을 맨 앞으로 (재회수 {}건). 고정점이 풀어낸 계획 조건 {}건을 뺐습니다: {:?}",
        target,
        if matches!(lang, "ko" | "ja" | "zh-hans") { "끝" } else { "처음" },
        mentions.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
        anchors.len(),
        anchors.iter().take(4).map(|a| format!("{}:{}{}", a.kind, a.matched, if a.primary { "" } else { "(보조)" })).collect::<Vec<_>>(),
        targets.len(),
        rescued,
        consumed.len(),
        consumed.iter().take(6).collect::<Vec<_>>()
    ));
    rep
}

pub struct Anchor {
    doc: usize,
    kind: &'static str,
    matched: String,
    primary: bool,
}
