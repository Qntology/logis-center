use serde_json::{json, Value};
use tauri::Emitter;
use crate::store::{TradeDocument, VectorStore};
use crate::scheduler::{entity_bcc, entity_id, entity_key_index, normalize_entity_key};
use crate::scheduler::indexing::save_item;
use crate::utils::canonical::{
    is_relay_placeholder, ledger_delta, ledger_prior, relay_edge_target, relay_edges, relay_establishes,
    relay_key_for_type, relay_ref_index, relay_type_family, LedgerPrior, LEDGER_KEY,
    LEDGER_PLACEHOLDER_DELTA, RELAY_BOUND_KEY, RELAY_LINK_KEYS, RELAY_ORIGIN_KEY, RELAY_TRANSIENT_KEYS,
};

pub type StatsDiff = std::collections::HashMap<String, (i64, i64, i64)>;

pub struct RelayEnv<'a> {
    pub store: &'a VectorStore,
    pub app_handle: &'a tauri::AppHandle,
    pub task_id: &'a str,
    pub team_id: &'a str,
    pub from: &'a str,
    pub cc: &'a str,
    pub ref_val: &'a str,
    pub search_mode: &'a str,
}

impl<'a> RelayEnv<'a> {
    fn emit(&self, msg: &str) {
        println!("{}", msg);
        let _ = self.app_handle.emit(
            "task-console-log",
            json!({ "task_id": self.task_id, "text": format!("{}\n", msg) }),
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct BridgeOutcome {
    pub referenced: bool,
    pub referrers: usize,
    pub linked: usize,
    pub drafted: usize,
    pub confirmed_foreign: usize,
    pub establishing_out: usize,
    pub uncertified: usize,
}

struct ForwardRef {
    key: String,
    ftype: Option<String>,
    index: u32,
    title: Option<String>,
    certified: bool,
}

pub fn add_delta(stats: &mut StatsDiff, t: &str, d: (i64, i64, i64)) {
    if d == (0, 0, 0) || t.trim().is_empty() {
        return;
    }
    let e = stats.entry(t.to_string()).or_insert((0, 0, 0));
    e.0 += d.0;
    e.1 += d.1;
    e.2 += d.2;
}

fn scalar_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

pub fn mark_relay_bound(item: &mut Value, key: &str) {
    let obj = match item.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    let slot = obj.entry(RELAY_BOUND_KEY.to_string()).or_insert_with(|| json!([]));
    if !slot.is_array() {
        *slot = json!([]);
    }
    if let Some(arr) = slot.as_array_mut() {
        if !arr.iter().any(|x| x.as_str() == Some(key)) {
            arr.push(json!(key));
        }
    }
}

fn take_relay_bound(item: &mut Value) -> Vec<String> {
    let mut bound: Vec<String> = Vec::new();
    if let Some(obj) = item.as_object_mut() {
        if let Some(Value::Array(arr)) = obj.remove(RELAY_BOUND_KEY) {
            for x in arr.iter() {
                if let Some(s) = x.as_str() {
                    bound.push(s.to_string());
                }
            }
        }
        for k in RELAY_TRANSIENT_KEYS.iter() {
            obj.remove(*k);
        }
    }
    bound
}

pub fn placeholder_origin_establishes(page_type: &str, existing: Option<&Value>) -> bool {
    let e = match existing {
        Some(e) => e,
        None => return false,
    };
    if ledger_prior(Some(e)) != LedgerPrior::Placeholder {
        return false;
    }
    e.get(RELAY_ORIGIN_KEY)
        .and_then(|v| v.as_array())
        .map_or(false, |arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .any(|o| relay_establishes(page_type, o))
        })
}

pub fn self_established(prior: LedgerPrior, origin_establishes: bool, bridge: &BridgeOutcome) -> bool {
    prior == LedgerPrior::Confirmed || bridge.referenced || bridge.establishing_out > 0 || origin_establishes
}

pub fn bind_tracking_ref(item: &mut Value, page_type: &str, team_id: &str, cc: &str) -> Option<u32> {
    if relay_type_family(page_type) != "order" {
        return None;
    }
    let tn = scalar_text(item.get("tracking_number"));
    if normalize_entity_key(&tn).is_empty() {
        return relay_ref_index(item.get("tracking"));
    }
    let idx = entity_key_index("tracking", team_id, cc, &tn);
    if let Some(obj) = item.as_object_mut() {
        obj.insert("tracking".to_string(), json!(idx));
    }
    mark_relay_bound(item, "tracking");
    Some(idx)
}

pub fn goods_array_refs(item: &mut Value, team_id: &str, cc: &str) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    if let Some(arr) = item.get_mut("goods").and_then(|v| v.as_array_mut()) {
        for g in arr.iter_mut() {
            let raw = {
                let id = scalar_text(g.get("id"));
                if id.is_empty() { scalar_text(g.get("no")) } else { id }
            };
            if normalize_entity_key(&raw).is_empty() {
                continue;
            }
            let idx = entity_key_index("goods", team_id, cc, &raw);
            if let Some(o) = g.as_object_mut() {
                o.insert("index".to_string(), json!(idx));
            }
            if !out.iter().any(|(_, i)| *i == idx) {
                out.push(("goods".to_string(), idx));
            }
        }
    }
    out
}

pub fn bind_url_relays(item: &mut Value, link: &str, team_id: &str, cc: &str) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    let url = link.trim();
    if url.is_empty() {
        return out;
    }
    let cands = crate::utils::ai_utils::collect_id_link_candidates_from_url(url);
    if cands.is_empty() {
        return out;
    }
    for family in ["goods", "order", "tracking"].iter() {
        let already = relay_ref_index(item.get(*family)).is_some();
        if already {
            continue;
        }
        let aliases = crate::logic::relay_type_aliases(family);
        let strong: Vec<&crate::utils::ai_utils::IdLinkCandidate> = cands
            .iter()
            .filter(|c| crate::utils::ai_utils::candidate_type_evidence(c, aliases) >= 2)
            .collect();
        let token = match strong.first() {
            Some(c) => c.token.clone(),
            None => continue,
        };
        if strong
            .iter()
            .any(|c| !crate::utils::ai_utils::same_id_token(&c.token, &token))
        {
            continue;
        }
        let idx = entity_key_index(family, team_id, cc, &token);
        if let Some(obj) = item.as_object_mut() {
            let prior_text = obj
                .get(*family)
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| crate::utils::canonical::relay_text_is_content(s));
            if let Some(t) = prior_text {
                let companion = format!("{}_title", family);
                let companion_empty = obj
                    .get(&companion)
                    .and_then(|v| v.as_str())
                    .map_or(true, |s| s.trim().is_empty());
                if companion_empty {
                    obj.insert(companion, json!(t));
                }
            }
            obj.insert(family.to_string(), json!(idx));
        }
        out.push((family.to_string(), idx));
    }
    out
}

pub fn settle_relay_keys(item: &mut Value, page_type: &str) -> Vec<String> {
    let own = relay_type_family(page_type);
    let mut moved: Vec<String> = Vec::new();
    let obj = match item.as_object_mut() {
        Some(o) => o,
        None => return moved,
    };
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own.as_str() {
            continue;
        }
        let raw = match obj.get(*key) {
            Some(Value::String(s)) => s.trim().to_string(),
            _ => continue,
        };
        obj.remove(*key);
        if raw.is_empty() || raw.eq_ignore_ascii_case("null") || raw.eq_ignore_ascii_case("n/a") {
            continue;
        }
        let companion = format!("{}_title", key);
        let companion_empty = obj
            .get(&companion)
            .and_then(|v| v.as_str())
            .map_or(true, |s| s.trim().is_empty());
        if companion_empty {
            obj.insert(companion, json!(raw));
        }
        moved.push(key.to_string());
    }
    moved
}

fn collect_forward_refs(item: &Value, page_type: &str, bound: &[String]) -> Vec<ForwardRef> {
    let own = relay_type_family(page_type);
    let mut out: Vec<ForwardRef> = Vec::new();
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own.as_str() {
            continue;
        }
        let idx = match relay_ref_index(item.get(*key)) {
            Some(i) => i,
            None => continue,
        };
        let ftype = match *key {
            "goods" | "order" | "tracking" => Some(key.to_string()),
            _ => None,
        };
        let title = item
            .get(&format!("{}_title", key))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        out.push(ForwardRef {
            key: key.to_string(),
            ftype,
            index: idx,
            title,
            certified: bound.iter().any(|b| b == key),
        });
    }
    out
}

pub async fn find_referrers(
    store: &VectorStore,
    key: &str,
    index: u32,
    types: &[&str],
    cap: usize,
) -> Vec<(String, Value)> {
    if index == 0 || types.is_empty() {
        return Vec::new();
    }
    let quoted: Vec<String> = types.iter().map(|t| format!("'{}'", t.replace('\'', "''"))).collect();
    let scalar_needle = crate::store::json_property_needle(key, &json!(index));
    let array_needle = format!("\"index\":{}", index);
    let filter = format!(
        "type IN ({}) AND (data LIKE '%{}%' OR data LIKE '%{}%')",
        quoted.join(", "),
        scalar_needle,
        array_needle
    );
    let docs = match store.get_all_items("items", cap.max(1), 0, Some(filter)).await {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    docs.into_iter()
        .filter_map(|d| {
            let v: Value = serde_json::from_str(&d.json_data).ok()?;
            if is_relay_placeholder(&v) {
                return None;
            }
            if !relay_edges(&v).iter().any(|(k, i)| k == key && *i == index) {
                return None;
            }
            Some((d.id, v))
        })
        .collect()
}

async fn find_tracking_by_number(store: &VectorStore, tn: &str, self_id: &str) -> Option<(String, u32)> {
    let needle = crate::store::json_property_needle("tracking_number", &json!(tn));
    let filter = format!("type IN ('tracking', 'receiving', 'shipping') AND data LIKE '%{}%'", needle);
    let docs = store.get_all_items("items", 4, 0, Some(filter)).await.ok()?;
    docs.into_iter().find_map(|d| {
        if d.id == self_id {
            return None;
        }
        let v: Value = serde_json::from_str(&d.json_data).ok()?;
        let idx = relay_ref_index(v.get("index"))?;
        Some((d.id, idx))
    })
}

async fn load_doc(store: &VectorStore, id: &str) -> (Option<TradeDocument>, Option<Value>) {
    let doc = store.get_item_by_id("items", id).await.ok().flatten();
    let json_val = doc.as_ref().and_then(|d| serde_json::from_str::<Value>(&d.json_data).ok());
    (doc, json_val)
}

pub async fn bridge_relays(
    env: &RelayEnv<'_>,
    page_type: &str,
    self_id: &str,
    self_index: u32,
    item: &mut Value,
    extra: &[(String, u32)],
    stats: &mut StatsDiff,
) -> BridgeOutcome {
    let mut out = BridgeOutcome::default();
    let own_family = relay_type_family(page_type);
    let bound = take_relay_bound(item);

    let settled = settle_relay_keys(item, page_type);
    if !settled.is_empty() {
        crate::utils::score_dynamics::record_baseline("commerce.relay_key_settled", settled.len() as f32);
        env.emit(&format!(
            "  🧷 [RELAY LEDGER / SETTLE] {} '{}' | 연결 키 {:?} 에 index 가 아닌 글자 값이 들어 있어 {{키}}_title 로 옮겼습니다. 연결 키에는 파이프라인이 만든 index 만 남깁니다. 저장 시 숫자 모양 글자는 수치로 굳어 다음 회차에 index 로 오인되기 때문입니다.",
            page_type, self_id, settled
        ));
    }

    let mut refs = collect_forward_refs(item, page_type, &bound);
    for (t, idx) in extra.iter() {
        if *idx == 0 || refs.iter().any(|r| r.index == *idx) {
            continue;
        }
        let key = relay_key_for_type(t).unwrap_or("").to_string();
        if key.is_empty() || key == own_family {
            continue;
        }
        refs.push(ForwardRef { key, ftype: Some(t.clone()), index: *idx, title: None, certified: true });
    }

    for r in refs.iter() {
        let mut fid = entity_id(env.team_id, r.index);
        if fid == self_id {
            continue;
        }
        let (mut existing, mut existing_json) = load_doc(env.store, &fid).await;

        if existing.is_none() && r.ftype.as_deref() == Some("tracking") {
            let tn = crate::utils::hash::normalize_identifier(&scalar_text(item.get("tracking_number")));
            if !tn.is_empty() {
                if let Some((tid, t_index)) = find_tracking_by_number(env.store, &tn, self_id).await {
                    env.emit(&format!(
                        "  🔄 [RELAY LEDGER / SECONDARY KEY] {} → tracking | index 자리는 비어 있지만 tracking_number '{}' 로 기존 문서 '{}' (index={}) 를 찾았습니다. 그 문서의 index 로 다시 묶습니다.",
                        page_type, tn, tid, t_index
                    ));
                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("tracking".to_string(), json!(t_index));
                    }
                    fid = tid;
                    let loaded = load_doc(env.store, &fid).await;
                    existing = loaded.0;
                    existing_json = loaded.1;
                }
            }
        }

        let expect_type = r.ftype.clone().unwrap_or_else(|| r.key.clone());
        let found_type = existing
            .as_ref()
            .map(|d| d.r#type.clone())
            .filter(|t| !t.trim().is_empty())
            .or_else(|| {
                existing_json
                    .as_ref()
                    .and_then(|v| v.get("type").and_then(|x| x.as_str()).map(|s| s.to_string()))
            })
            .unwrap_or_default();
        if !found_type.is_empty() && relay_type_family(&found_type) != relay_type_family(&expect_type) {
            crate::utils::score_dynamics::record_baseline("commerce.relay_type_guard", 1.0);
            env.emit(&format!(
                "  🔀 [RELAY LEDGER / TYPE GUARD] {} → {} (index={}) 자리의 문서가 '{}' 타입입니다. index 는 타입을 해시에 포함하므로 이 충돌은 기존 데이터 오염입니다. 연결도 초안 생성도 하지 않습니다.",
                page_type, expect_type, r.index, found_type
            ));
            continue;
        }

        let target_family = if found_type.is_empty() { expect_type.clone() } else { found_type.clone() };
        let establishes_self = relay_establishes(page_type, &target_family);

        match ledger_prior(existing_json.as_ref()) {
            LedgerPrior::Absent => {
                if !r.certified {
                    out.uncertified += 1;
                    crate::utils::score_dynamics::record_baseline("commerce.relay_uncertified", 1.0);
                    env.emit(&format!(
                        "  ⚪ [RELAY LEDGER / UNCERTIFIED] {} → '{}' 키(index={}) 는 이번 수집에서 파이프라인이 결속한 값이 아니고 가리키는 문서도 없습니다. 추출기가 남긴 수치일 수 있어 자리 초안을 만들지 않습니다. 상대가 같은 index 로 들어오면 그때 연결됩니다.",
                        page_type, r.key, r.index
                    ));
                    continue;
                }
                let ftype = match r.ftype.as_deref() {
                    Some(t) => t.to_string(),
                    None => {
                        env.emit(&format!(
                            "  ⚪ [RELAY LEDGER / NO TYPE] {} → '{}' 키(index={}) 는 coupon·event 공용 축이라 가리키는 타입이 하나로 정해지지 않습니다. 초안을 만들지 않고 상대 원본이 들어올 때까지 연결 대기로 둡니다.",
                            page_type, r.key, r.index
                        ));
                        continue;
                    }
                };
                let label = r.title.clone().unwrap_or_else(|| r.index.to_string());
                let mut draft = json!({
                    "id": fid.clone(),
                    "type": ftype.clone(),
                    "index": r.index,
                    "updated_at": 0,
                    "mode": env.search_mode,
                    "text": format!("{} {}", ftype, label),
                });
                if let Some(obj) = draft.as_object_mut() {
                    obj.insert(LEDGER_KEY.to_string(), json!("placeholder"));
                    obj.insert(RELAY_ORIGIN_KEY.to_string(), json!([page_type]));
                    if let Some(title) = r.title.as_ref() {
                        obj.insert("title".to_string(), json!(title));
                    }
                    if ftype == "tracking" {
                        let tn = crate::utils::hash::normalize_identifier(&scalar_text(item.get("tracking_number")));
                        if !tn.is_empty() {
                            obj.insert("tracking_number".to_string(), json!(tn.clone()));
                            obj.insert("text".to_string(), json!(format!("tracking {}", tn)));
                        }
                        if let Some(back) = relay_key_for_type(page_type) {
                            obj.insert(back.to_string(), json!(self_index));
                        }
                    }
                }
                let foreign_bcc = entity_bcc(&ftype, env.cc);
                save_item(
                    env.store, "items", &fid, &ftype, draft, None,
                    env.from, env.team_id, env.cc, &foreign_bcc, env.ref_val, None,
                ).await;
                add_delta(stats, &ftype, LEDGER_PLACEHOLDER_DELTA);
                out.drafted += 1;
                if establishes_self {
                    out.establishing_out += 1;
                }
                crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_draft", 1.0);
                env.emit(&format!(
                    "  📝 [RELAY LEDGER / DRAFT] {} → {} '{}' (index={}) 가 아직 없어 자리 초안을 만듭니다. 이 문서가 들고 있는 상대 식별자로 index 를 재현했으므로, {} 원본이 들어오면 같은 id 로 착지해 초안이 해소됩니다.",
                    page_type, ftype, fid, r.index, ftype
                ));
            }
            LedgerPrior::Placeholder => {
                out.linked += 1;
                if establishes_self {
                    out.establishing_out += 1;
                }
                if let (Some(mut ej), Some(doc)) = (existing_json.clone(), existing.as_ref()) {
                    let mut dirty = false;
                    if let Some(title) = r.title.as_ref() {
                        let has_title = ej
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map_or(false, |s| !s.trim().is_empty());
                        if !has_title {
                            if let Some(obj) = ej.as_object_mut() {
                                obj.insert("title".to_string(), json!(title));
                                obj.insert("text".to_string(), json!(format!("{} {}", found_type, title)));
                            }
                            dirty = true;
                        }
                    }
                    if let Some(obj) = ej.as_object_mut() {
                        let slot = obj.entry(RELAY_ORIGIN_KEY.to_string()).or_insert_with(|| json!([]));
                        if !slot.is_array() {
                            *slot = json!([]);
                        }
                        if let Some(arr) = slot.as_array_mut() {
                            if !arr.iter().any(|x| x.as_str() == Some(page_type)) {
                                arr.push(json!(page_type));
                                dirty = true;
                            }
                        }
                    }
                    if dirty {
                        save_item(
                            env.store, "items", &fid, &found_type, ej, None,
                            &doc.from, &doc.to, &doc.cc, &doc.bcc, &doc.r#ref, None,
                        ).await;
                    }
                }
            }
            LedgerPrior::Draft => {
                out.linked += 1;
                if establishes_self {
                    out.establishing_out += 1;
                }
                if !relay_establishes(&found_type, page_type) {
                    env.emit(&format!(
                        "  ⚪ [RELAY LEDGER / LINK ONLY] {} → {} '{}' (index={}) 는 draft 이지만 '{}' 참조는 '{}' 을 성립시키는 관계가 아니라 연결만 합니다.",
                        page_type, found_type, fid, r.index, page_type, found_type
                    ));
                    continue;
                }
                if let (Some(mut ej), Some(doc)) = (existing_json.clone(), existing.as_ref()) {
                    if let Some(obj) = ej.as_object_mut() {
                        obj.insert(LEDGER_KEY.to_string(), json!("count"));
                        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                    }
                    save_item(
                        env.store, "items", &fid, &found_type, ej, None,
                        &doc.from, &doc.to, &doc.cc, &doc.bcc, &doc.r#ref, None,
                    ).await;
                    add_delta(stats, &found_type, ledger_delta(LedgerPrior::Draft, true));
                    out.confirmed_foreign += 1;
                    crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_confirm", 1.0);
                    env.emit(&format!(
                        "  ✅ [RELAY LEDGER / CONFIRM] {} → {} '{}' (index={}) 는 draft 였고 이 문서가 성립 관계로 참조합니다. draft 에서 count 로 옮깁니다.",
                        page_type, found_type, fid, r.index
                    ));
                }
            }
            LedgerPrior::Confirmed => {
                out.linked += 1;
                if establishes_self {
                    out.establishing_out += 1;
                }
            }
        }
    }

    if let Some(key) = relay_key_for_type(page_type) {
        let related = crate::logic::related(page_type);
        let referrer_types: Vec<&str> = related
            .iter()
            .copied()
            .filter(|t| relay_type_family(t) != own_family && relay_establishes(page_type, t))
            .collect();
        let referrers = find_referrers(env.store, key, self_index, &referrer_types, 32).await;
        out.referrers = referrers.len();
        out.referenced = !referrers.is_empty();
    }

    crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_linked", out.linked as f32);
    crate::utils::score_dynamics::record_baseline("commerce.relay_establishing_out", out.establishing_out as f32);
    env.emit(&format!(
        "  🔗 [RELAY LEDGER] {} '{}' (index={}) | 정방향 연결 {}건 (성립 관계 {}건) · 자리 초안 생성 {}건 · 상대 draft→count {}건 · 미인증 {}건 | 역방향 성립 참조 {}건",
        page_type, self_id, self_index, out.linked, out.establishing_out, out.drafted, out.confirmed_foreign, out.uncertified, out.referrers
    ));
    out
}

#[derive(Debug, Default, Clone)]
pub struct RelayJoinReport {
    pub hydrated: usize,
    pub edges: usize,
    pub attached: usize,
    pub period_plans: usize,
    pub partners_in_period: usize,
    pub period_targets: usize,
    pub rescued: usize,
    pub dropped: usize,
}

impl RelayJoinReport {
    pub fn changed_set(&self) -> bool {
        self.rescued > 0 || self.dropped > 0
    }
}

fn result_doc(r: &Value) -> Option<Value> {
    match r.get("text") {
        Some(Value::Object(_)) => r.get("text").cloned(),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok().filter(|v| v.is_object()),
        _ => None,
    }
}

fn doc_team(doc: &Value, fallback: &str) -> String {
    let zero = "0x0000000000000000000000000000000000000000";
    doc.get("to")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != zero)
        .unwrap_or_else(|| fallback.to_string())
}

fn doc_label(doc: &Value) -> String {
    for k in ["title", "goods_title", "doc_number", "no", "tracking_number", "code"] {
        let v = scalar_text(doc.get(k));
        if !v.is_empty() {
            return v;
        }
    }
    String::new()
}

fn period_bound(p: &Value, key: &str) -> Option<String> {
    p.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() >= 10)
}

fn iso_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => {
            let t = s.trim();
            let b = t.as_bytes();
            if t.len() >= 10 && b[4] == b'-' && b[7] == b'-' {
                Some(t.chars().take(19).collect())
            } else {
                None
            }
        }
        Value::Number(n) => {
            let ms = n.as_i64()?;
            if ms < 100_000_000_000 {
                return None;
            }
            chrono::DateTime::from_timestamp_millis(ms)
                .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
        }
        _ => None,
    }
}

fn date_axis(doc: &Value) -> Option<(String, String)> {
    const PRIORITY: [&str; 7] = [
        "registration_date", "order_date", "ordered_at", "payment_date", "shipped_at", "delivered_at", "issue_date",
    ];
    for k in PRIORITY.iter() {
        if let Some(iso) = doc.get(*k).and_then(iso_of) {
            return Some((k.to_string(), iso));
        }
    }
    let obj = doc.as_object()?;
    for (k, v) in obj.iter() {
        if matches!(k.as_str(), "created_at" | "updated_at") {
            continue;
        }
        if crate::utils::ai_utils::detect_field_format(k) != crate::utils::ai_utils::FieldFormat::Date {
            continue;
        }
        if let Some(iso) = iso_of(v) {
            return Some((k.clone(), iso));
        }
    }
    None
}

fn status_is_void(doc: &Value) -> bool {
    let code = match doc.get("status") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) as i32,
        Some(Value::String(s)) => crate::logic::parse_status(s.trim()),
        _ => 0,
    };
    code != 0
        && [
            crate::logic::parse_status("cancel"),
            crate::logic::parse_status("refund"),
            crate::logic::parse_status("return"),
        ]
        .contains(&code)
}

fn partner_quantity(doc: &Value) -> Option<f64> {
    let q = match doc.get("quantity") {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().replace(',', "").parse::<f64>().ok(),
        _ => None,
    }?;
    if q.is_finite() && q > 0.0 {
        Some(q)
    } else {
        None
    }
}

fn unit_json(u: f64) -> Value {
    if u.fract() == 0.0 && u.abs() < 9.0e15 {
        json!(u as i64)
    } else {
        json!(u)
    }
}

fn in_period(iso: &str, start: Option<&str>, end: Option<&str>) -> bool {
    let head: String = iso.chars().take(19).collect();
    if let Some(s) = start {
        let s19: String = s.chars().take(19).collect();
        if head.as_str() < s19.as_str() {
            return false;
        }
    }
    if let Some(e) = end {
        let e19: String = e.chars().take(19).collect();
        if head.as_str() > e19.as_str() {
            return false;
        }
    }
    true
}

pub async fn relay_join(
    store: &VectorStore,
    results: &mut Vec<Value>,
    plans: &mut Vec<Value>,
    search_mode: &str,
    cc: &str,
    team_fallback: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> RelayJoinReport {
    let mut rep = RelayJoinReport::default();
    if results.is_empty() {
        return rep;
    }

    let mut docs: Vec<Option<Value>> = Vec::with_capacity(results.len());
    for r in results.iter() {
        let inline = result_doc(r);
        let d = match inline {
            Some(d) => Some(d),
            None => {
                let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    None
                } else {
                    load_doc(store, id).await.1
                }
            }
        };
        if d.is_some() {
            rep.hydrated += 1;
        }
        docs.push(d);
    }

    let pos_of: std::collections::HashMap<String, usize> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s.to_string(), i)))
        .collect();

    let mut outs: Vec<Vec<Value>> = vec![Vec::new(); results.len()];
    let mut ins: Vec<Vec<Value>> = vec![Vec::new(); results.len()];
    let mut foreign_cache: std::collections::HashMap<String, Option<Value>> = std::collections::HashMap::new();
    for (i, d) in docs.iter().enumerate() {
        let doc = match d {
            Some(x) => x,
            None => continue,
        };
        let team = doc_team(doc, team_fallback);
        let self_id = results[i].get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let self_type = scalar_text(doc.get("type"));
        for (key, index) in relay_edges(doc).into_iter() {
            rep.edges += 1;
            let tid = entity_id(&team, index);
            if tid == self_id {
                continue;
            }
            let target_type = relay_edge_target(&key).unwrap_or_default();
            let mut entry = json!({ "key": key, "index": index, "id": tid.clone(), "type": target_type });
            match pos_of.get(&tid) {
                Some(&j) => {
                    entry["in_results"] = json!(true);
                    ins[j].push(json!({ "key": key, "id": self_id.clone(), "type": self_type.clone() }));
                }
                None => {
                    entry["in_results"] = json!(false);
                    if foreign_cache.len() < 32 && !foreign_cache.contains_key(&tid) {
                        let fetched = load_doc(store, &tid).await.1;
                        foreign_cache.insert(tid.clone(), fetched);
                    }
                    if let Some(Some(fd)) = foreign_cache.get(&tid) {
                        entry["type"] = json!(scalar_text(fd.get("type")));
                        entry["label"] = json!(doc_label(fd));
                        entry["placeholder"] = json!(is_relay_placeholder(fd));
                        rep.attached += 1;
                    }
                }
            }
            if outs[i].len() < 16 {
                outs[i].push(entry);
            }
        }
    }
    for (i, r) in results.iter_mut().enumerate() {
        if let Some(o) = r.as_object_mut() {
            if let Some(Some(d)) = docs.get(i) {
                o.insert("doc_type".to_string(), json!(scalar_text(d.get("type"))));
            }
            if !outs[i].is_empty() {
                o.insert("relay_out".to_string(), json!(outs[i]));
            }
            if !ins[i].is_empty() {
                let mut v = ins[i].clone();
                v.truncate(32);
                o.insert("relay_in".to_string(), json!(v));
            }
        }
    }

    for plan in plans.iter_mut() {
        let period = match plan.get("relay_period") {
            Some(p) if p.is_object() => p.clone(),
            _ => continue,
        };
        let start = period_bound(&period, "start");
        let end = period_bound(&period, "end");
        if start.is_none() && end.is_none() {
            continue;
        }
        let primary = plan.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let primary_family = relay_type_family(&primary);
        let key = match relay_key_for_type(&primary) {
            Some(k) => k.to_string(),
            None => continue,
        };
        rep.period_plans += 1;
        let types: Vec<String> = plan
            .get("types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let via_family = period
            .get("via")
            .and_then(|v| v.as_str())
            .map(relay_type_family)
            .filter(|f| !f.is_empty() && *f != primary_family);
        let via_family = match via_family {
            Some(f) => f,
            None => {
                emit(&format!(
                    "[AI-SEARCH] ⚪ [RELAY PERIOD / HOLD] '{}' 의 기간 {:?} ~ {:?} 는 질의에 판매·거래 관계 근거가 없어 연결 타입의 날짜 축으로 옮기지 않습니다. '등록된 상품' 처럼 자기 날짜를 뜻하는 질의를 주문 날짜로 잘못 좁히지 않기 위해서입니다. 결과는 줄이지 않습니다.",
                    primary, start, end
                ));
                if let Some(o) = plan.as_object_mut() {
                    o.insert("relay".to_string(), json!({ "period": period, "key": key, "applied": false, "reason": "no_relation_intent" }));
                }
                continue;
            }
        };
        let related = crate::logic::related(&primary);
        let mut partner_types: Vec<String> = types
            .iter()
            .filter(|t| {
                relay_type_family(t) == via_family
                    && relay_type_family(t) != primary_family
                    && related.iter().any(|r| relay_type_family(r) == relay_type_family(t))
            })
            .cloned()
            .collect();
        if partner_types.is_empty() && related.iter().any(|r| relay_type_family(r) == via_family) {
            partner_types.push(via_family.clone());
        }
        let scope_key = crate::utils::score_dynamics::search_scope_key(&types);
        crate::utils::score_dynamics::enter_scope("", crate::utils::score_dynamics::Track::Search, &scope_key, "");

        let mut partners: Vec<(String, Value)> = Vec::new();
        for (i, d) in docs.iter().enumerate() {
            if let Some(doc) = d {
                let t = scalar_text(doc.get("type"));
                if partner_types.iter().any(|p| relay_type_family(p) == relay_type_family(&t)) {
                    let id = results[i].get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    partners.push((id, doc.clone()));
                }
            }
        }
        if !partner_types.is_empty() {
            let quoted: Vec<String> = partner_types.iter().map(|t| format!("'{}'", t.replace('\'', "''"))).collect();
            let mut filter = format!("type IN ({}) AND mode = '{}'", quoted.join(", "), search_mode.replace('\'', "''"));
            if !cc.trim().is_empty() && search_mode != "shipping" {
                filter.push_str(&format!(" AND `cc` = '{}'", cc.replace('\'', "''")));
            }
            if let Ok(extra_docs) = store.get_all_items("items", 1000, 0, Some(filter)).await {
                for d in extra_docs.into_iter() {
                    if partners.iter().any(|(id, _)| *id == d.id) {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(&d.json_data) {
                        if !is_relay_placeholder(&v) {
                            partners.push((d.id, v));
                        }
                    }
                }
            }
        }

        let mut hits: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut void_hits: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut void_partners = 0usize;
        let mut units: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut void_units: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut inexact: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut date_fields: Vec<String> = Vec::new();
        let mut in_period_ids: Vec<String> = Vec::new();
        let mut out_period_ids: Vec<String> = Vec::new();
        for (pid, pdoc) in partners.iter() {
            let (field, iso) = match date_axis(pdoc) {
                Some(x) => x,
                None => continue,
            };
            if !in_period(&iso, start.as_deref(), end.as_deref()) {
                out_period_ids.push(pid.clone());
                continue;
            }
            if !date_fields.iter().any(|f| *f == field) {
                date_fields.push(field);
            }
            let team = doc_team(pdoc, team_fallback);
            let voided = status_is_void(pdoc);
            let edges: Vec<u32> = relay_edges(pdoc)
                .into_iter()
                .filter(|(k, _)| *k == key)
                .map(|(_, idx)| idx)
                .collect();
            let qty = if edges.len() == 1 { partner_quantity(pdoc) } else { None };
            let per_edge = qty.unwrap_or(1.0);
            let linked = !edges.is_empty();
            for idx in edges.into_iter() {
                let tid = entity_id(&team, idx);
                if qty.is_none() {
                    inexact.insert(tid.clone());
                }
                *hits.entry(tid.clone()).or_insert(0) += 1;
                *units.entry(tid.clone()).or_insert(0.0) += per_edge;
                if voided {
                    *void_hits.entry(tid.clone()).or_insert(0) += 1;
                    *void_units.entry(tid).or_insert(0.0) += per_edge;
                }
            }
            if linked {
                in_period_ids.push(pid.clone());
                if voided {
                    void_partners += 1;
                }
            }
        }
        rep.partners_in_period += in_period_ids.len();
        rep.period_targets += hits.len();
        crate::utils::score_dynamics::record_baseline("search.relay_period_partners", in_period_ids.len() as f32);
        crate::utils::score_dynamics::record_baseline("search.relay_period_targets", hits.len() as f32);

        if hits.is_empty() {
            crate::utils::score_dynamics::record_baseline("search.relay_period_empty", 1.0);
            emit(&format!(
                "[AI-SEARCH] ⚪ [RELAY PERIOD / EMPTY] '{}' 의 기간 {:?} ~ {:?} 을 연결 타입 {:?} 의 날짜 축으로 옮겨 보았지만 기간 안에서 '{}' 를 가리키는 문서가 없습니다. 결과를 줄이지 않고 표시만 합니다.",
                primary, start, end, partner_types, key
            ));
            if let Some(o) = plan.as_object_mut() {
                o.insert("relay".to_string(), json!({ "period": period, "key": key, "partner_types": partner_types, "applied": false }));
            }
            crate::utils::score_dynamics::leave_scope();
            continue;
        }

        let min_score = results
            .iter()
            .filter_map(|r| r.get("score").and_then(|v| v.as_f64()))
            .fold(f64::MAX, f64::min);
        let rescue_score = if min_score.is_finite() && min_score < f64::MAX { min_score.max(0.05) } else { 0.5 };
        let mut rescued_ids: Vec<String> = Vec::new();
        for tid in hits.keys() {
            if results.iter().any(|r| r.get("id").and_then(|v| v.as_str()) == Some(tid.as_str())) {
                continue;
            }
            let (doc, j) = load_doc(store, tid).await;
            let (doc, j) = match (doc, j) {
                (Some(d), Some(j)) => (d, j),
                _ => continue,
            };
            if is_relay_placeholder(&j) || relay_type_family(&doc.r#type) != primary_family {
                continue;
            }
            results.push(json!({
                "id": tid.clone(),
                "text": doc.json_data.clone(),
                "score": rescue_score,
                "context_type": primary.clone(),
                "doc_type": doc.r#type.clone(),
                "relation": "relay_rescue"
            }));
            rescued_ids.push(tid.clone());
        }
        rep.rescued += rescued_ids.len();

        let pre_drop = results.len();
        results.retain(|r| {
            let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let t = r
                .get("doc_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| result_doc(r).map(|d| scalar_text(d.get("type"))))
                .unwrap_or_default();
            let fam = relay_type_family(&t);
            if fam == primary_family {
                return hits.contains_key(id);
            }
            !out_period_ids.iter().any(|x| x == id)
        });
        let dropped = pre_drop - results.len();
        rep.dropped += dropped;
        for r in results.iter_mut() {
            let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if let Some(n) = hits.get(&id) {
                if let Some(o) = r.as_object_mut() {
                    o.insert("relay_period_hits".to_string(), json!(n));
                    if let Some(v) = void_hits.get(&id) {
                        o.insert("relay_period_void".to_string(), json!(v));
                    }
                    if let Some(u) = units.get(&id) {
                        o.insert("relay_period_units".to_string(), unit_json(*u));
                    }
                    if let Some(u) = void_units.get(&id) {
                        o.insert("relay_period_void_units".to_string(), unit_json(*u));
                    }
                    o.insert("relay_period_units_exact".to_string(), json!(!inexact.contains(&id)));
                }
            } else if in_period_ids.iter().any(|x| *x == id) {
                if let Some(o) = r.as_object_mut() {
                    o.insert("relay_period_match".to_string(), json!(true));
                }
            }
        }
        let target_ids: Vec<String> = hits.keys().cloned().collect();
        let units_exact = inexact.is_empty();
        let units_total: f64 = units.values().sum();
        let void_units_total: f64 = void_units.values().sum();
        let units_map: serde_json::Map<String, Value> = units.iter().map(|(k, v)| (k.clone(), unit_json(*v))).collect();
        let void_units_map: serde_json::Map<String, Value> = void_units.iter().map(|(k, v)| (k.clone(), unit_json(*v))).collect();
        if let Some(o) = plan.as_object_mut() {
            o.insert("relay".to_string(), json!({
                "period": period,
                "key": key,
                "partner_types": partner_types,
                "date_fields": date_fields,
                "target_ids": target_ids,
                "hits": hits,
                "void_hits": void_hits,
                "units": units_map,
                "void_units": void_units_map,
                "units_exact": units_exact,
                "partner_ids": in_period_ids,
                "applied": true
            }));
        }
        crate::utils::score_dynamics::record_baseline("search.relay_rescued", rescued_ids.len() as f32);
        crate::utils::score_dynamics::record_baseline("search.relay_dropped", dropped as f32);
        crate::utils::score_dynamics::record_baseline("search.relay_units_exact", if units_exact { 1.0 } else { 0.0 });
        emit(&format!(
            "[AI-SEARCH] 🔗 [RELAY PERIOD JOIN] '{}' 질의의 기간은 판매·거래 관계의 시점이라 기간 {:?} ~ {:?} 을 연결 타입 {:?} 의 날짜 축 {:?} 로 옮겼습니다. 기간 안 연결 문서 {}건(취소·환불·반품 {}건 포함)이 '{}' 로 가리키는 {} {}건을 결과 집합으로 확정합니다 (재회수 {}건 · 기간 밖/미연결 제외 {}건). 취소 건은 빼지 않고 relay_period_void 로 따로 표시합니다. 수량 합 {}개 (취소·환불·반품 {}개){}",
            primary, start, end, partner_types, date_fields, in_period_ids.len(), void_partners, key, primary, hits.len(), rescued_ids.len(), dropped,
            unit_json(units_total), unit_json(void_units_total),
            if units_exact { " — 연결 문서의 quantity 를 그대로 더했습니다." } else { " — quantity 가 없거나 한 문서가 여러 상대를 가리키는 경우는 1개로 셌습니다(relay_period_units_exact=false)." }
        ));
        crate::utils::score_dynamics::leave_scope();
    }

    if rep.edges > 0 {
        emit(&format!(
            "[AI-SEARCH] 🔗 [RELAY JOIN] 회수 {}건의 연결 축 {}개를 index → id 로 풀어 relay_out / relay_in 을 붙였습니다 (결과 밖 상대 문서 {}건은 표시용으로 읽었습니다).",
            rep.hydrated, rep.edges, rep.attached
        ));
    }
    rep
}