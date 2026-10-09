//! 저장소 계층(`tauri_app_lib::store`) 검증 (T09–T11, T35–T37)
//!
//! - 순수 판정기: 예약 타입, mode 추론, 테이블 라우팅, 도메인 시딩, 릴레이 초안, LIKE 니들, 봉투 확정.
//! - LanceDB: 테스트마다 `TempDir` 하나에 `VectorStore` 를 새로 열어 작업 큐·메시지·upsert 계약을 확인합니다.
//!   (전역 DB 나 reset_database 는 건드리지 않습니다)
//! `#[ignore = "BUG(Bn): …"]` 테스트는 '의도된 동작' 을 단언하며, 수정 전까지는 실패합니다.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::{json, Value};
use tauri_app_lib::scheduler::indexing::save_item;
use tauri_app_lib::store::{
    draft_named_by_query, drop_unnamed_drafts, infer_mode, is_embed_excluded_type, is_relay_draft,
    is_reserved_type, json_property_needle, needs_domain_seed, resolve_envelope_field, resolve_table_for,
    AppConfig, Task, VectorStore,
};

use crate::common::TempDir;

// ── 헬퍼 ─────────────────────────────────────────────────────────

async fn open_store(tag: &str) -> anyhow::Result<(TempDir, VectorStore)> {
    let dir = TempDir::new(&format!("store_{tag}"));
    let path = dir.path().to_string_lossy().into_owned();
    let store = VectorStore::new(&path).await?;
    Ok((dir, store))
}

/// tasks + talks 테이블만 만든 저장소
async fn task_store(tag: &str) -> anyhow::Result<(TempDir, VectorStore)> {
    let (dir, store) = open_store(tag).await?;
    store.init_task_table().await?;
    Ok((dir, store))
}

/// items / users / pages / item_chunks 테이블을 만든 저장소
async fn item_store(tag: &str) -> anyhow::Result<(TempDir, VectorStore)> {
    let (dir, store) = open_store(tag).await?;
    store.init_all_tables().await?;
    Ok((dir, store))
}

/// items 테이블에 봉투 f/t/{cc}/b/r 로 upsert (벡터 없음)
async fn put(store: &VectorStore, id: &str, ty: &str, data: Value, cc: &str, digest: &str) -> anyhow::Result<()> {
    store
        .upsert_item(
            "items", id, ty, data, None, None,
            Some("f"), Some("t"), Some(cc), Some("b"), Some("r"), Some(digest),
        )
        .await
}

/// 저장된 data JSON
async fn data_of(store: &VectorStore, table: &str, id: &str) -> anyhow::Result<Value> {
    let doc = store
        .get_item_by_id(table, id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("{table}/{id} is missing"))?;
    Ok(serde_json::from_str(&doc.json_data)?)
}

fn task(id: &str, created_at: i64, cc: &str, reference: &str) -> Task {
    Task {
        id: id.to_string(),
        r#type: "html_extraction".to_string(),
        from: "f".to_string(),
        to: "t".to_string(),
        cc: cc.to_string(),
        bcc: "b".to_string(),
        r#ref: reference.to_string(),
        data_json: "{}".to_string(),
        created_at,
        updated_at: created_at,
        status: 10,
    }
}

fn task_ids(tasks: &[Task]) -> Vec<String> {
    tasks.iter().map(|t| t.id.clone()).collect()
}

fn msg_ids(rows: &[Value]) -> Vec<String> {
    rows.iter().map(|m| m["id"].as_str().unwrap_or("").to_string()).collect()
}

// ── T09 타입 판정 · mode · 라우팅 ─────────────────────────────────

#[test]
fn type_predicates_and_mode_inference() -> anyhow::Result<()> {
    for t in [" Goods ", "unknown", "click", "page", "users", "talk", "member", "receiving"] {
        assert!(is_reserved_type(t), "{t:?} should be reserved");
    }
    for t in ["BL", "CI", "invoice", ""] {
        assert!(!is_reserved_type(t), "{t:?} should not be reserved");
    }

    assert_eq!(infer_mode("bl"), "shipping");
    assert_eq!(infer_mode("ID"), "shipping");
    assert_eq!(infer_mode("shipping_doc"), "shipping");
    assert_eq!(infer_mode("touch"), "analytic");
    assert_eq!(infer_mode("Click"), "analytic");
    assert_eq!(infer_mode("goods"), "commerce");

    // 임베딩 제외: 페이지 캐시 · 채팅 · 사용자 · 관리자 말풍선. 행동 로그(click/report)는 포함.
    for t in ["Question", "answer", "member", "users", "page", "talk", "ai_search"] {
        assert!(is_embed_excluded_type(t), "{t:?} should be excluded");
    }
    for t in ["click", "report", "goods", "BL"] {
        assert!(!is_embed_excluded_type(t), "{t:?} should be embedded");
    }
    Ok(())
}

#[test]
fn table_routing_and_domain_seed() -> anyhow::Result<()> {
    for (input, table) in [
        ("commerce_users", "users"),
        ("commerce_goods", "items"),
        ("commerce_pages", "pages"),
        ("users", "users"),
        ("member", "users"),
        ("team", "users"),
        ("user", "users"),
        ("page", "pages"),
        ("pages", "pages"),
        ("goods", "items"),
        ("talk", "items"),
        ("BL", "items"),
        ("", "items"),
        ("commerce_", "items"),
    ] {
        assert_eq!(resolve_table_for(input), table, "{input:?}");
    }

    assert!(!needs_domain_seed("users", "goods"));
    assert!(!needs_domain_seed("pages", "goods"));
    assert!(!needs_domain_seed("items", "team"));
    assert!(!needs_domain_seed("items", " Member "));
    assert!(!needs_domain_seed("items", "click"));
    assert!(needs_domain_seed("items", "BL"));
    assert!(needs_domain_seed("items", "goods"));
    Ok(())
}

// ── T10 릴레이 초안 ──────────────────────────────────────────────

#[test]
fn relay_draft_detection() -> anyhow::Result<()> {
    assert!(is_relay_draft(&json!({"updated_at": 0, "digest": ""})));
    assert!(is_relay_draft(&json!({"updated_at": 0, "digest": "   "})));
    assert!(!is_relay_draft(&json!({"updated_at": 0, "digest": "", "embed": 1})));
    assert!(!is_relay_draft(&json!({"updated_at": 0, "digest": "", "embed": true})));
    assert!(!is_relay_draft(&json!({"updated_at": 5, "digest": ""})));
    assert!(!is_relay_draft(&json!({"updated_at": 0, "digest": "d1"})));
    assert!(!is_relay_draft(&json!("plain text")));
    Ok(())
}

#[test]
fn draft_named_by_query_tokens() -> anyhow::Result<()> {
    assert!(draft_named_by_query(&json!({"no": "ORD-12345"}), "find ord-12345 please"));
    assert!(draft_named_by_query(&json!({"no": 12345}), "12345"));
    assert!(draft_named_by_query(&json!({"tracking_number": " 603145678912 "}), "where is 603145678912?"));
    assert!(draft_named_by_query(&json!({"text": "Order ORD-12345"}), "ord-12345"));
    // 숫자가 없거나 5자 미만인 질의 토큰은 이름으로 보지 않습니다.
    assert!(!draft_named_by_query(&json!({"no": 12345}), "1234"));
    assert!(!draft_named_by_query(&json!({"no": "shoes"}), "shoes"));
    // text 의 첫 단어(타입 라벨 자리)는 이름 후보에서 빠집니다.
    assert!(!draft_named_by_query(&json!({"text": "ORD-12345 shipped"}), "ORD-12345"));
    assert!(!draft_named_by_query(&json!({"no": "ORD-99999"}), "ord-12345"));
    Ok(())
}

#[test]
fn drop_unnamed_drafts_keeps_named_and_foreign_rows() -> anyhow::Result<()> {
    let row = |s: &str| (s.to_string(), 0.5f32);
    let mut combined: HashMap<String, (String, f32)> = HashMap::new();
    combined.insert("a".to_string(), row(r#"{"updated_at":0,"digest":""}"#));
    combined.insert("b".to_string(), row(r#"{"updated_at":5}"#));
    combined.insert("c".to_string(), row("not json"));
    combined.insert("d".to_string(), row(r#"{"updated_at":0,"digest":"","no":"ORD-55555"}"#));

    // 빈 질의는 아무것도 지우지 않습니다.
    let mut blank = combined.clone();
    assert_eq!(drop_unnamed_drafts(&mut blank, "   "), 0);
    assert_eq!(blank.len(), 4);

    // 이름 없는 초안(a)만 빠지고, 질의가 이름을 댄 초안(d)과 비초안(b, c)은 남습니다.
    assert_eq!(drop_unnamed_drafts(&mut combined, "status of ord-55555"), 1);
    let mut keys: Vec<&str> = combined.keys().map(|k| k.as_str()).collect();
    keys.sort();
    assert_eq!(keys, ["b", "c", "d"]);
    Ok(())
}

// ── T11 LIKE 니들 · 봉투 확정 ─────────────────────────────────────

#[test]
fn property_needles_and_envelope_fields() -> anyhow::Result<()> {
    assert_eq!(json_property_needle("tracking_number", &json!("ABC'1")), r#""tracking_number":"ABC''1""#);
    assert_eq!(json_property_needle("no", &json!(123)), r#""no":"123""#);
    assert_eq!(json_property_needle("goods", &json!(42)), r#""goods":42"#);
    assert_eq!(json_property_needle("price", &json!("1000")), r#""price":1000"#);
    assert_eq!(json_property_needle("is_active", &json!(true)), r#""is_active":1"#);
    assert_eq!(json_property_needle("title", &json!("x")), "x");

    // 인자 → data → 빈 문자열 순
    assert_eq!(resolve_envelope_field(Some(" x "), &json!({}), "cc"), "x");
    assert_eq!(resolve_envelope_field(Some("  "), &json!({"cc": " y "}), "cc"), "y");
    assert_eq!(resolve_envelope_field(None, &json!({"cc": 5}), "cc"), "");
    assert_eq!(resolve_envelope_field(None, &json!({}), "cc"), "");
    Ok(())
}

// ── T35 작업 큐 ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_queue_lifecycle() -> anyhow::Result<()> {
    let (_dir, s) = task_store("queue").await?;
    s.add_task(task("t-1", 200, "c", "r")).await?;
    s.add_task(task("t-0", 100, "c", "r0")).await?;

    let pending = s.get_pending_tasks(10).await?;
    assert_eq!(task_ids(&pending), ["t-0", "t-1"], "FIFO by created_at");
    let t0 = &pending[0];
    assert_eq!((t0.r#type.as_str(), t0.cc.as_str(), t0.r#ref.as_str()), ("html_extraction", "c", "r0"));
    assert_eq!((t0.data_json.as_str(), t0.status), ("{}", 10));

    assert!(s.has_active_task("c", "r").await?);
    assert!(!s.has_active_task("c", "zzz").await?);
    assert!(!s.has_active_task("other", "r").await?);

    s.update_task_status("t-1", 1).await?;
    assert_eq!(task_ids(&s.get_processing_tasks(10).await?), ["t-1"]);
    assert_eq!(task_ids(&s.get_pending_tasks(10).await?), ["t-0"]);
    assert!(s.has_active_task("c", "r").await?, "processing (1) still counts as active");

    // 재시작 복구: 처리 중(1) → 대기(10), 진행 말풍선도 대기 문구로 교체
    s.add_message_at(
        "w-1", "system_task", "Processing...", Some("t-1"), Some(1),
        Some("c"), Some("b"), Some("r"), Some("f"), Some("to"), Some("talk"), None, Some(500),
    )
    .await?;
    s.cleanup_unfinished_tasks_on_startup().await?;
    assert!(s.get_processing_tasks(10).await?.is_empty());
    assert_eq!(task_ids(&s.get_pending_tasks(10).await?), ["t-0", "t-1"]);
    let talks = s.get_all_messages(10, 0, None).await?;
    assert_eq!(talks.len(), 1);
    assert_eq!(talks[0]["status"], 10);
    assert_eq!(talks[0]["text"], "App restarted. Task is queued for auto-resumption...");

    // 9(완료) / 6(오류) 는 행을 지웁니다.
    s.update_task_status("t-0", 9).await?;
    s.update_task_status("t-1", 6).await?;
    assert!(s.get_pending_tasks(10).await?.is_empty());
    assert!(!s.has_active_task("c", "r").await?);
    Ok(())
}

// ── T36 메시지 ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn message_status_update_preserves_envelope() -> anyhow::Result<()> {
    let (_dir, s) = task_store("msg").await?;
    s.add_message_at(
        "m-1", "system_task", "Task Started", Some("T-1"), Some(10),
        Some("c"), Some("b"), Some("r"), Some("f"), Some("to"), Some("talk"), None, Some(1000),
    )
    .await?;
    s.update_message_status("T-1", 1, Some("Processing...")).await?;

    let rows = s.get_all_messages(10, 0, None).await?;
    assert_eq!(rows.len(), 1, "{rows:?}");
    let m = &rows[0];
    assert_ne!(m["id"], "m-1", "the row is re-inserted under a fresh id");
    assert_eq!(m["text"], "Processing...");
    assert_eq!(m["status"], 1);
    assert_eq!(m["role"], "system_task");
    assert_eq!(m["type"], "talk");
    assert_eq!(m["task_id"], "T-1");
    assert_eq!(m["created_at"], 1000, "created_at survives delete + re-insert");
    for (k, v) in [("cc", "c"), ("bcc", "b"), ("ref", "r"), ("from", "f"), ("to", "to")] {
        assert_eq!(m[k], v, "envelope field {k}");
    }

    // text 없이 종료하면 말풍선이 사라집니다.
    s.update_message_status("T-1", 9, None).await?;
    assert!(s.get_all_messages(10, 0, None).await?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messages_are_newest_first_and_paged() -> anyhow::Result<()> {
    let (_dir, s) = task_store("paging").await?;
    for (id, at, status) in [("p-1", 1i64, 0), ("p-2", 2, 7), ("p-3", 3, 0)] {
        s.add_message_at(
            id, "user", id, None, Some(status),
            Some("c"), Some("b"), Some("r"), None, None, None, None, Some(at),
        )
        .await?;
    }
    assert_eq!(msg_ids(&s.get_all_messages(2, 0, None).await?), ["p-3", "p-2"]);
    assert_eq!(msg_ids(&s.get_all_messages(2, 2, None).await?), ["p-1"]);
    assert!(s.get_all_messages(2, 5, None).await?.is_empty());
    assert_eq!(msg_ids(&s.get_all_messages(10, 0, Some("status = 7".to_string())).await?), ["p-2"]);
    // 공백 필터는 무시됩니다.
    assert_eq!(s.get_all_messages(10, 0, Some("   ".to_string())).await?.len(), 3);
    let newest = s.get_all_messages(1, 0, None).await?;
    assert_eq!((newest[0]["type"].as_str(), newest[0]["role"].as_str()), (Some("talk"), Some("user")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "BUG(B3): update_message_status deletes every row sharing the task_id, including the user's question"]
async fn update_message_status_keeps_user_query_row() -> anyhow::Result<()> {
    let (_dir, s) = task_store("b3").await?;
    s.add_message_at(
        "q-1", "user", "where is BL-1?", Some("T-b3"), Some(0),
        Some("c"), Some("b"), Some("r"), Some("f"), Some("to"), Some("talk"), None, Some(1000),
    )
    .await?;
    s.add_message_at(
        "m-1", "system_task", "Task Started", Some("T-b3"), Some(10),
        Some("c"), Some("b"), Some("r"), Some("f"), Some("to"), Some("talk"), None, Some(1050),
    )
    .await?;
    s.update_message_status("T-b3", 1, Some("Processing...")).await?;

    let rows = s.get_all_messages(10, 0, None).await?;
    let users: Vec<&Value> = rows.iter().filter(|m| m["role"] == "user").collect();
    assert_eq!(users.len(), 1, "{rows:?}");
    assert_eq!(users[0]["text"], "where is BL-1?");
    let tasks: Vec<&Value> = rows.iter().filter(|m| m["role"] == "system_task").collect();
    assert_eq!(tasks.len(), 1, "{rows:?}");
    assert_eq!(tasks[0]["text"], "Processing...");
    Ok(())
}

// ── T37 upsert 계약 ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_canonicalizes_and_seeds_domain_axes() -> anyhow::Result<()> {
    let (_dir, s) = item_store("canon").await?;
    put(&s, "g-1", "goods", json!({"text": "blue shirt", "price": "1,000", "no": 123}), "c", "d1").await?;

    let doc = s.get_item_by_id("items", "g-1").await?.expect("stored in items");
    assert_eq!(doc.r#type, "goods");
    assert_eq!(doc.mode, "commerce");
    assert_eq!(
        (doc.from.as_str(), doc.to.as_str(), doc.cc.as_str(), doc.bcc.as_str(), doc.r#ref.as_str()),
        ("f", "t", "c", "b", "r")
    );
    assert_eq!(doc.updated_at_ts, 0, "a domain item without updated_at keeps the relay-zero stamp");
    assert!(doc.created_at_ts > 0);
    assert_eq!(doc.text, "blue shirt");
    assert_eq!(doc.masked_text, "blue shirt", "masked_text falls back to text");

    let d: Value = serde_json::from_str(&doc.json_data)?;
    assert_eq!(d["price"], json!(1000));
    assert_eq!(d["sale_price"], json!(1000), "sale_price mirrors price");
    assert_eq!(d["no"], json!("123"), "identifiers are stored as strings");
    assert_eq!(d["digest"], json!("d1"));
    assert_eq!(d["embed"], json!(0));
    assert_eq!(d["tags"], json!([]));
    for k in ["index", "goods", "order", "tracking", "status", "updated_at"] {
        assert_eq!(d[k], json!(0), "numeric seed {k}");
    }
    for k in ["code", "tracking_number", "stock_keeping_unit", "barcode"] {
        assert_eq!(d[k], json!(""), "identifier seed {k}");
    }
    assert_eq!(d["id"], json!("g-1"));
    assert_eq!(d["type"], json!("goods"));
    assert_eq!(d["mode"], json!("commerce"));
    assert_eq!(d["cc"], json!("c"));
    assert_eq!(d["created_at"], json!(doc.created_at_ts));

    // commerce_users → users 테이블, 도메인 축 시딩 없음
    s.upsert_item(
        "commerce_users", "u-1", "user", json!({"text": "alice"}), None, None,
        Some("f"), Some("t"), Some("c"), Some("b"), Some("r"), Some("du"),
    )
    .await?;
    let user = s.get_item_by_id("users", "u-1").await?.expect("stored in users");
    assert!(s.get_item_by_id("items", "u-1").await?.is_none());
    assert!(user.updated_at_ts > 0, "non-domain tables stamp wall-clock updated_at");
    let ud: Value = serde_json::from_str(&user.json_data)?;
    assert!(ud.get("index").is_none() && ud.get("tags").is_none(), "{ud}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_skips_same_digest_but_rewrites_envelope_change() -> anyhow::Result<()> {
    let (_dir, s) = item_store("skip").await?;
    let data = json!({"text": "green hat", "no": "H-1"});
    put(&s, "s-1", "goods", data.clone(), "c", "d1").await?;
    let first = s.get_item_by_id("items", "s-1").await?.expect("stored");

    // 같은 봉투 + 같은 digest → 기록 생략 (created_at 그대로)
    tokio::time::sleep(Duration::from_millis(5)).await;
    put(&s, "s-1", "goods", data.clone(), "c", "d1").await?;
    let again = s.get_item_by_id("items", "s-1").await?.expect("stored");
    assert_eq!(again.created_at_ts, first.created_at_ts, "identical upsert must be skipped");

    // 봉투(cc)만 바뀌어도 digest 와 무관하게 재기록
    put(&s, "s-1", "goods", data.clone(), "c2", "d1").await?;
    let moved = s.get_item_by_id("items", "s-1").await?.expect("stored");
    assert_eq!(moved.cc, "c2");
    assert_eq!(data_of(&s, "items", "s-1").await?["cc"], json!("c2"));

    // digest 가 바뀌면 본문도 재기록
    put(&s, "s-1", "goods", json!({"text": "green hat v2", "no": "H-1"}), "c2", "d2").await?;
    let d = data_of(&s, "items", "s-1").await?;
    assert_eq!(d["digest"], json!("d2"));
    assert_eq!(d["text"], json!("green hat v2"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_embed_flag_follows_real_vectors() -> anyhow::Result<()> {
    let (_dir, s) = item_store("embed").await?;
    // embed:1 주장만 있고 실제 벡터가 없으면 0 으로 내립니다.
    put(&s, "e-1", "goods", json!({"text": "red cap", "embed": 1}), "c", "dA").await?;
    assert_eq!(data_of(&s, "items", "e-1").await?["embed"], json!(0));

    // 같은 digest 라도 새 실벡터가 오면 기록하고 embed=1
    s.upsert_item(
        "items", "e-1", "goods", json!({"text": "red cap"}), Some(vec![0.5f32; 384]), None,
        Some("f"), Some("t"), Some("c"), Some("b"), Some("r"), Some("dA"),
    )
    .await?;
    assert_eq!(data_of(&s, "items", "e-1").await?["embed"], json!(1));

    // 벡터 없이 본문이 바뀌어도 저장 벡터를 승계하므로 embed=1 유지
    put(&s, "e-1", "goods", json!({"text": "red cap v2"}), "c", "dB").await?;
    let d = data_of(&s, "items", "e-1").await?;
    assert_eq!(d["embed"], json!(1));
    assert_eq!(d["digest"], json!("dB"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routing_lookup_and_delete_cascade() -> anyhow::Result<()> {
    let (_dir, s) = item_store("route").await?;

    // save_item: table hint 우선 → pages 에 저장하되 type 은 도메인 타입 그대로
    save_item(&s, "pages", "pg-1", "goods", json!({"text": "selector cache"}), None, "f", "t", "c", "b", "r", Some("dp")).await;
    let page = s.get_item_by_id("pages", "pg-1").await?.expect("stored in pages");
    assert_eq!(page.r#type, "goods");
    assert!(s.get_item_by_id("items", "pg-1").await?.is_none());
    // hint 가 비면 type 으로 라우팅
    save_item(&s, "", "mb-1", "member", json!({"text": "bob"}), None, "f", "t", "c", "b", "r", Some("dm")).await;
    assert!(s.get_item_by_id("users", "mb-1").await?.is_some());
    assert!(s.get_item_by_id("items", "mb-1").await?.is_none());

    // data LIKE 니들 조회
    put(&s, "it-1", "tracking", json!({"text": "parcel", "tracking_number": "603145678912"}), "c", "d1").await?;
    let (found, data) = s
        .find_item_by_property("items", "tracking_number", &json!("603145678912"))
        .await?
        .expect("found by needle");
    assert_eq!(found, "it-1");
    assert_eq!(data["tracking_number"], "603145678912");
    assert!(s.find_item_by_property("items", "tracking_number", &json!("000000000000")).await?.is_none());
    assert!(s.find_item_by_property("items", "tracking_number", &json!("")).await?.is_none());

    // 문서 삭제 시 청크도 함께 삭제
    for i in 0..3 {
        s.upsert_chunk(
            &format!("it-1#{i}"), "it-1", "tracking", "parcel", "text", "plain", "", None,
            Some("c"), Some("b"), Some("r"), Some("commerce"),
        )
        .await?;
    }
    assert!(s.count_chunks_by_item("it-1").await? >= 1);
    s.delete_item("items", "it-1").await?;
    assert!(s.get_item_by_id("items", "it-1").await?.is_none());
    assert_eq!(s.count_chunks_by_item("it-1").await?, 0, "chunks go away with their item");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settings_json_roundtrip() -> anyhow::Result<()> {
    let dir = TempDir::new("store_settings");
    let path = dir.path().to_string_lossy().into_owned();
    let s = VectorStore::new(&path).await?;

    let fresh = s.load_config();
    assert!(!fresh.is_logged_in && fresh.auth_token.is_none());

    s.save_config(&AppConfig { is_logged_in: true, auth_token: Some("tok-1".to_string()) })?;
    let back = s.load_config();
    assert!(back.is_logged_in);
    assert_eq!(back.auth_token.as_deref(), Some("tok-1"));

    // 깨진 settings.json 은 기본값으로 대체
    std::fs::write(dir.path().join("settings.json"), "{not json")?;
    let broken = s.load_config();
    assert!(!broken.is_logged_in && broken.auth_token.is_none());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "BUG(B20): count_chunks_by_item applies limit(1), so it reports 0 or 1"]
async fn count_chunks_reports_every_chunk() -> anyhow::Result<()> {
    let (_dir, s) = item_store("b20").await?;
    for i in 0..3 {
        s.upsert_chunk(
            &format!("ck-{i}"), "it-9", "goods", "blue shirt", "text", "plain", "", None,
            Some("c"), Some("b"), Some("r"), Some("commerce"),
        )
        .await?;
    }
    assert_eq!(s.count_chunks_by_item("it-9").await?, 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "BUG(B25): ids are interpolated into '…' filters without escaping single quotes"]
async fn ids_with_single_quote_roundtrip() -> anyhow::Result<()> {
    let (_dir, s) = item_store("b25").await?;
    put(&s, "o'brien-1", "goods", json!({"text": "irish wool"}), "c", "d1").await?;
    let doc = s.get_item_by_id("items", "o'brien-1").await?.expect("stored");
    assert_eq!(doc.id, "o'brien-1");
    s.delete_item("items", "o'brien-1").await?;
    assert!(s.get_item_by_id("items", "o'brien-1").await?.is_none());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "BUG(B26): upsert_item resets created_at to now when the incoming data lacks it"]
async fn envelope_rewrite_keeps_created_at() -> anyhow::Result<()> {
    let (_dir, s) = item_store("b26").await?;
    let data = json!({"text": "grey sock"});
    put(&s, "k-1", "goods", data.clone(), "c", "d1").await?;
    let first = s.get_item_by_id("items", "k-1").await?.expect("stored");
    tokio::time::sleep(Duration::from_millis(20)).await;
    put(&s, "k-1", "goods", data, "c2", "d1").await?;
    let moved = s.get_item_by_id("items", "k-1").await?.expect("stored");
    assert_eq!(moved.cc, "c2");
    assert_eq!(moved.created_at_ts, first.created_at_ts, "an envelope-only resync must not re-date the row");
    Ok(())
}
