use serde_json::json;

pub async fn update_team_base_metrics(
    store: &crate::store::VectorStore,
    team_id: &str,
    task_cc: &str,
    items: &Vec<serde_json::Value>,
    stats_diff: std::collections::HashMap<String, (i64, i64, i64)>,
) -> anyhow::Result<()> {
    // 🌟 [v4] TradeDocument 에서 digest 필드가 제거되었습니다. (data.digest 로 하강)
    //    이 함수는 digest 를 쓰지도 않았으므로(마지막 upsert 에 None 전달) 바인딩째 제거합니다.
    //
    //    vector 도 함께 제거합니다. get_item_by_id 는 v3 시절부터 vector 컬럼(10번)을
    //    읽지 않아 team_vector 가 항상 빈 배열이었고, upsert_item 의 `v.len() == 384`
    //    검사에서 탈락해 결국 매번 0벡터로 저장되고 있었습니다.
    //    users 테이블은 벡터 검색 대상이 아니므로 의도를 드러내 None 을 넘깁니다.
    //
    //    폴백 문서에는 mode / text 를 넣습니다. v4 에서 mode 는 봉투 물리 컬럼이고,
    //    text 가 비면 LanceDB text 컬럼이 빈 문자열이 되어 문서 식별이 어려워집니다.
    let (team_json_str, t_from, t_to, t_cc, t_bcc, t_ref) = match store.get_item_by_id("users", team_id).await {
        Ok(Some(doc)) => (doc.json_data, doc.from, doc.to, doc.cc, doc.bcc, doc.r#ref),
        _ => (
            json!({ "base": { "pages": {} }, "mode": "commerce", "text": "team" }).to_string(),
            "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string()
        )
    };

    
    let mut parsed_val: serde_json::Value = serde_json::from_str(&team_json_str).unwrap_or(json!({ "base": { "pages": {} } }));
    
    
    while let Some(inner_str) = parsed_val.get("json_data").and_then(|v| v.as_str()) {
        if let Ok(inner_obj) = serde_json::from_str(inner_str) {
            parsed_val = inner_obj;
        } else {
            break;
        }
    }
    let mut team_data = parsed_val;
    
    
    if let Some(obj) = team_data.as_object_mut() {
        obj.remove("json_data");
    }
    
    // --- [블록 1 & 2: 맵 순회로 모든 타입의 통계 업데이트] ---
    for (t_name, (pages_draft_diff, pages_count_diff, global_count_diff)) in stats_diff.iter() {
        // 페이지별 통계 업데이트
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
            let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
            let page_type_node = cc_node.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            let current_draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            let current_count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            page_type_node.insert("draft".to_string(), json!(0.max(current_draft + pages_draft_diff)));
            page_type_node.insert("count".to_string(), json!(0.max(current_count + pages_count_diff)));
        } 

        // 글로벌 전체 통계 업데이트 (aa.ts와 동일하게 draft는 건드리지 않고 count만 누적)
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
            
            let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // 글로벌 draft는 클라우드 로직 상 사용되지 않으므로 보존하거나 건드리지 않습니다.
            global_type_node.insert("count".to_string(), json!(0.max(global_count + global_count_diff)));
        }
    }

    // Min/Max 업데이트는 items 내의 데이터에 한해서 진행
    {
        // 🌟 [v4] amount / total_amount 를 추가합니다.
        //    v4 에서 amount 는 LanceDB 물리 컬럼에서 사라지고 data.amount 로 내려갔습니다.
        //    Dexie 의 top / bottom 백분위 조건은 이 min/max 통계를 기준으로 범위를 잡으므로,
        //    통계에 amount 가 없으면 '가장 비싼 20%' 같은 질의가 부정확해집니다.
        let properties = [
            "price", "amount", "total_amount", "quantity", "width", "height", "length", "weight",
            "shipping_fee", "shipping_duration", "sale_price", "supply_price", "low_stock_threshold",
            "discount", "min_order_amount", "max_discount_amount", "usage_limit",
            "usage_per", "started_at", "expired_at"
        ];

        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            for prop in properties.iter() {
                if let Some(val) = item.get(*prop) {
                    let num_val = if val.is_number() {
                        val.as_f64().unwrap_or(0.0)
                    } else if let Some(s) = val.as_str() {
                        // 🌟 [PARSE FIX] 기존 s.parse::<f64>() 는 "5,000" / "₩5000" / "5000원" 에서
                        //    전부 실패해 0.0 을 반환했고, 그 값이 아래 0 스킵에 걸려
                        //    통계에 아예 반영되지 않았습니다.
                        //    (scheduler 의 items_to_process 는 canonicalize 이전 원본이라
                        //     이런 표기가 그대로 들어옵니다)
                        //    store.rs 의 canonicalize_data 와 동일한 규칙으로 정규화합니다.
                        let cleaned: String = s.chars()
                            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                            .collect();
                        cleaned.parse::<f64>().unwrap_or(0.0)
                    } else if let Some(b) = val.as_bool() {
                        if b { 1.0 } else { 0.0 }
                    } else {
                        continue;
                    };

                    // 🌟 [MIN/MAX INIT FIX] 기존 코드는 두 규칙이 충돌하고 있었습니다.
                    //      ① started_at / expired_at 만 0 을 통계에 허용
                    //      ② `current_min == 0.0` 을 '아직 값 없음' 신호로 사용
                    //    0(=날짜 미설정)이 한 번이라도 들어오면 min 이 0 으로 굳고,
                    //    그 이후로는 ②가 항상 참이 되어 min 이 '마지막에 들어온 아이템 값' 으로
                    //    매번 덮어써졌습니다. 통계가 사실상 무작위가 됩니다.
                    //    0 은 '미설정' 이지 유효한 하한이 아니므로 모든 속성에서 동일하게 제외합니다.
                    if num_val == 0.0 { continue; }

                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 0.0, "max": 0.0 })).as_object_mut().unwrap();

                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    // 이제 0 은 '미초기화' 를 뜻하는 것이 확실하므로 판정이 안전합니다.
                    if current_min == 0.0 || num_val < current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if current_max == 0.0 || num_val > current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    
    if let Some(base_json) = team_data.get("base") {
        println!("\n[DEBUG-METRICS] 최종 반영된 Base JSON 값:\n{}", serde_json::to_string_pretty(base_json).unwrap_or_default());
    }

    
    if let Some(obj) = team_data.as_object_mut() {
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    // 5. Save back to DB
    //    🌟 digest 에 None 을 전달해 upsert_item 의 스킵 가드를 우회합니다.
    //       v4 가드는 `!new_digest.is_empty()` 조건이라, 빈 digest 면 항상 쓰기가 수행됩니다.
    //    🌟 vector 에 None 을 전달하면 upsert_item 이 vec![0.0; 384] 로 채웁니다.
    //       (기존에도 실질적으로 같은 동작이었고, 이제 의도가 코드에 드러납니다)
    let _ = store.upsert_item(
        "users",
        team_id,
        "team",
        team_data,
        None,
        Some(&t_from),
        Some(&t_to),
        Some(&t_cc),
        Some(&t_bcc),
        Some(&t_ref),
        None
    ).await;

    
    if let Ok(Some(saved_doc)) = store.get_item_by_id("users", team_id).await {
        println!("\n==================================================");
        println!("✅ [DB-VERIFY] DB에 통계(Team) 데이터가 100% 정상 저장되었습니다!");
        println!("- 타겟 ID: {}", saved_doc.id);
        println!("- 갱신된 Timestamp: {}", saved_doc.updated_at_ts);
        
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&saved_doc.json_data) {
            if let Some(base_stats) = parsed.get("base") {
                println!("- DB 내 실제 Base 통계:\n{}", serde_json::to_string_pretty(base_stats).unwrap_or_default());
            }
        }
        println!("==================================================\n");
    } else {
        println!("\n==================================================");
        println!("🚨 [DB-VERIFY] 치명적 오류: DB에 Team 데이터가 저장되지 않았습니다!");
        println!("==================================================\n");
    }

    Ok(())
}