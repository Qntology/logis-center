//! 🌟 [SYNTH EVAL] 합성 데이터를 실제 파이프라인에 태워 결과와 SDS 실행 파일을 남깁니다.
//!
//! 브라우저 · 프론트엔드 · 창 없이 앱과 같은 `process_task` / `ai_search_complex` 를 부릅니다. 계획 파일(JSONL)의
//! 줄마다 작업 하나를 순서대로 돌리고 결과를 결과 파일(JSONL)에 씁니다. 채점과 SDS 교차 확인은
//! `SyntheticData/eval` 의 파이썬이 합니다.
//!
//! ```text
//! LOGIS_APP_DIR=<평가 폴더> LOGIS_SYNTH_PLAN=<plan.jsonl> LOGIS_SYNTH_OUT=<result.jsonl> \
//! cargo test --test it --no-default-features --features rocm -- synthetic_eval --ignored --nocapture --test-threads 1
//! ```
//!
//! 평가 폴더에는 `models` 만 앱 데이터의 모델 폴더로 연결해 둡니다. DB 와 SDS 원장은 평가 폴더 안에 새로 쌓이므로
//! 사용자의 앱 데이터는 바뀌지 않습니다. `LOGIS_APP_DIR` 이 없거나 사용자의 앱 데이터와 같으면 실행을 거부합니다.
//!
//! 프론트엔드가 맡던 일은 여기서 대신합니다.
//!  · 음차 캐시 질의(`translit-cache-query`)에 빈 응답을 돌려 값마다 5초씩 기다리지 않게 합니다.
//!
//! 모델 다운로드는 앱의 로직을 그대로 씁니다. 언어 전용 4B 모델도 앱처럼 필요할 때 Hugging Face 에서 받고 변환하며,
//! 미리 받아 두려면 계획 첫 줄에 `lang_engine` 작업을 둡니다.
//!
//! 계획 줄의 `kind`
//!  · `html`     : `{id, kind, url, path, mode}` 페이지 HTML → process_task
//!  · `document` : `{id, kind, path, mode}` 문서 파일 → process_task (텍스트 경로)
//!  · `image`    : `{id, kind, path, mode}` 이미지 파일 → process_task (SigLIP2 경로)
//!  · `search`   : `{id, kind, query, mode, language, host}` 자연어 질의 → ai_search_complex (host 는 앱의 현재 사이트 범위)
//!  · `seed`     : `{id, kind, host, type, url, lang, detail, data}` 정답 레코드를 앱의 색인 경로로 바로 넣습니다.
//!    검색을 추출 오류와 떼어 재기 위한 것이고, 언어 모델 없이 임베딩만 씁니다.
//!  · `lang_engine` : `{id, kind, lang}` 앱의 `await_engine` 으로 언어 전용 모델을 받아 실행 준비까지 마칩니다.
//!  · `dump`     : `{id, kind}` 저장소의 아이템 전체를 결과 줄의 `items` 로 남깁니다(score_commerce.py --pool).
//!  · `seed2`    : `{id, kind, mode, host, type, self, links, arrays, hub, data ...}` 정답 레코드를 추출 경로와 같은 모양
//!    (id = entity_id, data.index, 연결 칸 = 대상 index)으로 넣습니다. 관계 질의를 추출 오류와 떼어 잽니다.
//!  · `keys`     : `{id, kind, hosts, items:[[mode, host, 유형, 번호] ...]}` 정답 엔터티의 연결 번호 · 사이트 cc 를 남깁니다.
//!  · `analytic` : `{id, kind, events, batch}` 행동 로그 합성 행(JSONL, Cloudflare D1 행 모양)을 앱의 동기화와 같은
//!    순서(upsert_items → structure_pending_analytics → reindex_pending_embeddings)로 넣습니다. `eval_analytic_ingest`.
#![cfg(any(windows, target_os = "linux"))]

use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tauri::Listener;
use tauri_app_lib::model::LogisModel;
use tauri_app_lib::scheduler::indexing::{index_item_chunks, save_item};
use tauri_app_lib::scheduler::{process_task, TRANSLIT_PENDING};
use tauri_app_lib::score_dynamics as sds;
use tauri_app_lib::store::{Task, VectorStore};
use tauri_app_lib::utils::hash::{self, hash_id};
use tauri_app_lib::AppState;

const ZERO_ADDR: &str = "0x0000000000000000000000000000000000000000";

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn str_of<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("")
}

/// main.ts 의 addToQueue 가 만드는 것과 같은 모양의 태스크.
fn build_task(entry: &Value, team: &str) -> Task {
    let id = str_of(entry, "id").to_string();
    let path = str_of(entry, "path");
    let mode = str_of(entry, "mode");
    let now = now_ms();
    let (r#type, cc, r#ref, payload) = match str_of(entry, "kind") {
        "html" => {
            let url = url::Url::parse(str_of(entry, "url")).expect("계획의 url");
            let host = url.host_str().unwrap_or_default().to_string();
            let origin = format!("{}://{}", url.scheme(), host);
            let raw_path = match url.query() {
                Some(q) => format!("{}?{}", url.path(), q),
                None => url.path().to_string(),
            };
            let cc = hash_id(&host);
            let r#ref = hash_id(&format!("{}{}{}", team, cc, raw_path.to_lowercase()));
            let html = std::fs::read_to_string(path).expect("계획의 html 파일");
            let payload = json!({
                "id": id, "type": "html_extraction", "html": html, "link": raw_path, "origin": origin,
                "cc": cc, "ref": r#ref, "bcc": "", "from": ZERO_ADDR, "to": team, "flag": "", "search_mode": mode,
            });
            ("html_extraction", cc, r#ref, payload)
        }
        "document" => {
            let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            let r#ref = hash_id(path);
            let payload = json!({
                "id": id, "type": "document_extraction", "image_path": path, "document_ext": ext,
                "ref": r#ref, "cc": "", "bcc": "", "link": "Local DOCUMENT", "search_mode": mode,
            });
            ("document_extraction", String::new(), r#ref, payload)
        }
        "image" => {
            let r#ref = hash_id(path);
            let payload = json!({
                "id": id, "type": "image_extraction", "image_path": path, "document_ext": "png",
                "ref": r#ref, "cc": "", "bcc": "", "link": "Local IMAGE", "search_mode": mode,
            });
            ("image_extraction", String::new(), r#ref, payload)
        }
        other => panic!("알 수 없는 작업 종류: {other}"),
    };
    Task {
        id,
        r#type: r#type.to_string(),
        from: ZERO_ADDR.to_string(),
        to: team.to_string(),
        cc,
        bcc: String::new(),
        r#ref,
        data_json: payload.to_string(),
        created_at: now,
        updated_at: now,
        status: 10,
    }
}

/// 정답 레코드 하나를 추출 없이 저장하고 색인합니다. 문서 문장 · 문서 벡터 · 청크는 앱과 같은 코드로 만듭니다.
async fn seed_record(
    db: &VectorStore,
    model: &LogisModel,
    entry: &Value,
    team: &str,
    cancel: &Arc<AtomicBool>,
    handle: &tauri::AppHandle,
) -> anyhow::Result<usize> {
    let (host, kind) = (str_of(entry, "host"), str_of(entry, "type"));
    let id = hash_id(str_of(entry, "id"));
    let cc = hash_id(host);
    let bcc = hash_id(&format!("{}{}", kind, cc));
    let r#ref = hash_id(str_of(entry, "url"));
    let mut data = entry["data"].clone();
    let text = tauri_app_lib::nl_convert::json_to_natural_language(&data);
    let digest = hash::digest(&text);
    let vector = model.get_embedding(text.clone()).await?;
    if let Some(o) = data.as_object_mut() {
        o.insert("text".into(), json!(text));
        o.insert("masked_text".into(), json!(text));
        o.insert("embed".into(), json!(1));
        o.insert("mode".into(), json!("commerce"));
        o.insert("type".into(), json!(kind));
    }
    save_item(db, "items", &id, kind, data.clone(), Some(vector), ZERO_ADDR, team, &cc, &bcc, &r#ref, Some(&digest)).await;
    index_item_chunks(
        db,
        model,
        &id,
        kind,
        str_of(entry, "lang"),
        &data,
        entry["detail"].as_bool().unwrap_or(true),
        &cc,
        &bcc,
        &r#ref,
        "commerce",
        str_of(entry, "url"),
        cancel,
        handle,
        str_of(entry, "id"),
        true,
    )
    .await
}

/// 연결 번호: 커머스는 entity_key_index(유형, 팀, cc, 번호) (사이트마다 다름), 무역은 entity_index(서식, 팀, 번호) (사이트 없음).
fn link_index(mode: &str, ty: &str, team: &str, cc: &str, raw: &str) -> u32 {
    if mode == "shipping" {
        tauri_app_lib::scheduler::entity_index(ty, team, raw)
    } else {
        tauri_app_lib::scheduler::entity_key_index(ty, team, cc, raw)
    }
}

/// 🌟 [SEED2] 정답 레코드를 추출 경로가 남기는 모양으로 저장합니다: id = entity_id(팀, index), data.index,
/// 연결 칸(goods · order · tracking · event · rel_<서식>) = 대상의 index. 추출 오류 없이 검색 로직만 재려고 씁니다.
///  `{id, kind:"seed2", mode, host, type, self:[유형, 번호], lang, detail, url, hub, data,
///    links:{칸:[유형, 번호]}, arrays:{칸:[[유형, 번호, 제목] ...]}}`
async fn seed_linked(
    db: &VectorStore,
    model: &LogisModel,
    entry: &Value,
    team: &str,
    cancel: &Arc<AtomicBool>,
    handle: &tauri::AppHandle,
) -> anyhow::Result<(String, u32, usize)> {
    let mode = match str_of(entry, "mode") { "" => "commerce", m => m };
    let kind = str_of(entry, "type");
    let cc = hash_id(str_of(entry, "host"));
    let self_ty = entry["self"][0].as_str().unwrap_or(kind);
    let self_raw = entry["self"][1].as_str().unwrap_or("");
    let index = link_index(mode, self_ty, team, &cc, self_raw);
    let id = tauri_app_lib::scheduler::entity_id(team, index);
    let bcc = tauri_app_lib::scheduler::entity_bcc(kind, &cc);
    let r#ref = match str_of(entry, "hub") {
        "" => hash_id(&format!("{}{}{}", team, cc, str_of(entry, "url").to_lowercase())),
        hub => hash_id(&format!("{}{}#{}", team, cc, hub)),
    };
    let mut data = entry["data"].clone();
    let o = data.as_object_mut().ok_or_else(|| anyhow::anyhow!("data 가 객체가 아님"))?;
    if let Some(links) = entry.get("links").and_then(|v| v.as_object()) {
        for (field, target) in links {
            let (ty, raw) = (target[0].as_str().unwrap_or(""), target[1].as_str().unwrap_or(""));
            if !ty.is_empty() && !raw.is_empty() {
                o.insert(field.clone(), json!(link_index(mode, ty, team, &cc, raw)));
            }
        }
    }
    if let Some(arrays) = entry.get("arrays").and_then(|v| v.as_object()) {
        for (field, targets) in arrays {
            let list: Vec<Value> = targets
                .as_array()
                .into_iter()
                .flatten()
                .map(|t| json!({"index": link_index(mode, t[0].as_str().unwrap_or(""), team, &cc, t[1].as_str().unwrap_or("")),
                                "title": t[2].as_str().unwrap_or("")}))
                .collect();
            o.insert(field.clone(), json!(list));
        }
    }
    let text = tauri_app_lib::nl_convert::json_to_natural_language(&data);
    let digest = hash::digest(&text);
    let vector = model.get_embedding(text.clone()).await?;
    if let Some(o) = data.as_object_mut() {
        o.insert("index".into(), json!(index));
        o.insert("ledger".into(), json!("count"));
        o.insert("text".into(), json!(text));
        o.insert("masked_text".into(), json!(text));
        o.insert("embed".into(), json!(1));
        o.insert("mode".into(), json!(mode));
        o.insert("type".into(), json!(kind));
    }
    save_item(db, "items", &id, kind, data.clone(), Some(vector), ZERO_ADDR, team, &cc, &bcc, &r#ref, Some(&digest)).await;
    let chunks = index_item_chunks(
        db, model, &id, kind, str_of(entry, "lang"), &data, entry["detail"].as_bool().unwrap_or(true), &cc, &bcc, &r#ref,
        mode, str_of(entry, "url"), cancel, handle, str_of(entry, "id"), true,
    )
    .await?;
    Ok((id, index, chunks))
}

/// 행동 로그 합성 행(JSONL) → upsert_items 가 받는 아이템. modes/analytic.ts 의 fetchAnalyticsOrigin 이 Worker 응답을
/// 바꾸는 규칙을 그대로 따릅니다: cc = hash(host), bcc = hash(type + cc), ref = hash(팀 + cc + 경로) (html 작업과 같은 식),
/// 원시 이벤트(action · relate 가 배열)는 updated_at 0, text 는 비워 둡니다(구조화가 채움).
fn analytic_rows(path: &str, team: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(path).expect("행동 로그 파일");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let row: Value = serde_json::from_str(l).expect("행동 로그 줄");
            let (host, path) = (str_of(&row, "host"), str_of(&row, "path"));
            let r#type = str_of(&row, "type").to_string();
            let cc = hash_id(host);
            let bcc = hash_id(&format!("{}{}", r#type, cc));
            let r#ref = hash_id(&format!("{}{}{}", team, cc, path.to_lowercase()));
            let mut data = row["data"].clone();
            let id = str_of(&row, "id").to_string();
            if let Some(o) = data.as_object_mut() {
                o.insert("id".into(), json!(id));
                o.insert("type".into(), json!(r#type));
                o.insert("mode".into(), json!("analytic"));
                o.insert("updated_at".into(), json!(0));
                o.insert("text".into(), json!(""));
                o.insert("masked_text".into(), json!(""));
            }
            json!({
                "id": id, "type": r#type, "flag": "", "from": str_of(&row, "from"), "to": team, "cc": cc, "bcc": bcc,
                "ref": r#ref, "status": 9, "mode": "analytic", "created_at": row["created_at"].as_i64().unwrap_or(0),
                "updated_at": 0, "text": "", "masked_text": "", "data": data,
            })
        })
        .collect()
}

/// `since` 뒤로 만들어졌거나 바뀐 아이템. 목록 행과 상세 페이지가 같은 레코드로 합쳐지는 경우까지 잡습니다.
async fn changed_items(store: &VectorStore, since: i64) -> Vec<Value> {
    let docs = store.get_all_items("items", 20_000, 0, None).await.unwrap_or_default();
    docs.into_iter()
        .filter(|d| d.updated_at_ts >= since || d.created_at_ts >= since)
        .map(|d| {
            let data: Value = serde_json::from_str(&d.json_data).unwrap_or(json!({}));
            json!({
                "id": d.id, "type": d.r#type, "mode": d.mode, "ref": d.r#ref, "cc": d.cc, "bcc": d.bcc,
                "text": d.text, "data": data, "has_vector": d.vector.iter().any(|x| *x != 0.0),
            })
        })
        .collect()
}

#[test]
#[ignore = "실모델: LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT · LOGIS_APP_DIR"]
fn run_plan() {
    let (Ok(plan_path), Ok(out_path), Some(app_dir)) = (
        std::env::var("LOGIS_SYNTH_PLAN"),
        std::env::var("LOGIS_SYNTH_OUT"),
        std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()).map(PathBuf::from),
    ) else {
        println!("LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT · LOGIS_APP_DIR 가 없어 건너뜁니다.");
        return;
    };
    let real_dir = dirs::data_local_dir().map(|d| d.join("logis-center"));
    assert_ne!(Some(app_dir.clone()), real_dir, "평가는 사용자의 앱 데이터 폴더에서 돌지 않습니다");

    let plan: Vec<Value> = std::fs::read_to_string(&plan_path)
        .expect("계획 파일")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("계획 줄"))
        .collect();
    // 끊긴 실행을 이어서 돌립니다. 이미 결과가 있는 줄은 건너뜁니다.
    let done: HashSet<String> = std::fs::read_to_string(&out_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .map(|v| str_of(&v, "id").to_string())
        .collect();
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path).expect("결과 파일");

    let store = Arc::new(tokio::sync::Mutex::new(None));
    let model = Arc::new(tokio::sync::Mutex::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    let app = tauri_app_lib::build_headless_app(AppState {
        model: model.clone(),
        store: store.clone(),
        cancellation_token: cancel.clone(),
    })
    .expect("앱 핸들");
    let handle = app.handle().clone();
    app.listen("translit-cache-query", |e| {
        let Ok(v) = serde_json::from_str::<Value>(e.payload()) else { return };
        if let Some(tx) = TRANSLIT_PENDING.lock().unwrap().remove(str_of(&v, "request_id")) {
            let _ = tx.send(Vec::new());
        }
    });
    // 판정 단계만 잴 때: 로그에 이 문구가 나오면 태스크를 취소합니다. 언어 모델 단계 앞에서 끊을 수 있습니다.
    if let Some(stop_on) = std::env::var("LOGIS_SYNTH_STOP_ON").ok().filter(|v| !v.is_empty()) {
        // 언어 모델까지 가지 않는 실행이므로 언어 엔진 다운로드도 시작하지 않습니다. 다른 프로세스가 받는 중일 수 있습니다.
        tauri_app_lib::model::lang_llm::mark_runtime_failure("kor", "판정 단계만 재는 실행");
        let stop = cancel.clone();
        app.listen("task-console-log", move |e| {
            if e.payload().contains(&stop_on) {
                stop.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        });
    }

    tauri::async_runtime::block_on(async {
        let db_path = tauri_app_lib::utils::get_app_dir().join("db");
        std::fs::create_dir_all(&db_path).expect("DB 폴더");
        let db = VectorStore::new(&db_path.to_string_lossy()).await.expect("VectorStore");
        db.init_task_table().await.expect("태스크 표");
        db.init_all_tables().await.expect("표");
        *store.lock().await = Some(db);
        let team = hash_id(ZERO_ADDR);
        sds::load(&team);

        let total = plan.len();
        for (n, entry) in plan.iter().enumerate() {
            let id = str_of(entry, "id").to_string();
            if done.contains(&id) {
                continue;
            }
            println!("\n@@EVAL BEGIN {} ({}/{}) {}", id, n + 1, total, str_of(entry, "kind"));
            let t0 = now_ms();
            let started = std::time::Instant::now();
            sds::set_run_label(&id);
            let mut line = json!({"id": id, "kind": str_of(entry, "kind"), "t0": t0});

            if str_of(entry, "kind") == "dump" {
                // 저장소 전체를 남깁니다. 상세가 앞서 저장된 목록 문서에 합쳐지면 그 작업의 '바뀐 아이템' 에는 나오지 않아,
                // 처리 순서와 상관없이 레코드 단위로 채점하려면 끝 상태가 필요합니다.
                let db = store.lock().await.clone().expect("저장소");
                line["items"] = json!(changed_items(&db, 0).await);
                line["ok"] = json!(true);
            } else if str_of(entry, "kind") == "lang_engine" {
                let (engine, note) =
                    tauri_app_lib::model::lang_llm::await_engine(str_of(entry, "lang"), &handle, &id, &cancel, 1).await;
                line["ok"] = json!(engine.is_lang());
                line["note"] = json!(note);
            } else if str_of(entry, "kind") == "seed" {
                let db = store.lock().await.clone().expect("저장소");
                let logis = {
                    let mut g = model.lock().await;
                    if g.is_none() {
                        *g = Some(LogisModel::new(handle.clone(), None).await.expect("모델"));
                    }
                    g.as_ref().unwrap().clone()
                };
                logis.ensure_embedding().await.expect("임베딩 모델");
                let r = seed_record(&db, &logis, entry, &team, &cancel, &handle).await;
                line["ok"] = json!(r.is_ok());
                match r {
                    Ok(chunks) => line["chunks"] = json!(chunks),
                    Err(e) => line["error"] = json!(format!("{:#}", e)),
                }
                sds::flush();
            } else if str_of(entry, "kind") == "seed2" {
                let db = store.lock().await.clone().expect("저장소");
                let logis = {
                    let mut g = model.lock().await;
                    if g.is_none() {
                        *g = Some(LogisModel::new(handle.clone(), None).await.expect("모델"));
                    }
                    g.as_ref().unwrap().clone()
                };
                logis.ensure_embedding().await.expect("임베딩 모델");
                match seed_linked(&db, &logis, entry, &team, &cancel, &handle).await {
                    Ok((item_id, index, chunks)) => {
                        line["ok"] = json!(true);
                        line["item_id"] = json!(item_id);
                        line["index"] = json!(index);
                        line["chunks"] = json!(chunks);
                    }
                    Err(e) => {
                        line["ok"] = json!(false);
                        line["error"] = json!(format!("{:#}", e));
                    }
                }
                sds::flush();
            } else if str_of(entry, "kind") == "keys" {
                // 정답 엔터티의 연결 번호 (전처리 채점이 연결 칸을 대조할 때 씁니다).
                let keys: Vec<Value> = entry["items"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|k| {
                        let (mode, host, ty, raw) = (k[0].as_str().unwrap_or(""), k[1].as_str().unwrap_or(""), k[2].as_str().unwrap_or(""), k[3].as_str().unwrap_or(""));
                        json!([mode, host, ty, raw, link_index(mode, ty, &team, &hash_id(host), raw)])
                    })
                    .collect();
                line["keys"] = json!(keys);
                line["cc"] = json!(entry["hosts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|h| h.as_str())
                    .map(|h| json!([h, hash_id(h)]))
                    .collect::<Vec<_>>());
                line["ok"] = json!(true);
            } else if str_of(entry, "kind") == "analytic" {
                let rows = analytic_rows(str_of(entry, "events"), &team);
                line["rows"] = json!(rows.len());
                let batch = entry["batch"].as_u64().unwrap_or(20) as usize;
                match tauri_app_lib::eval_analytic_ingest(&handle, rows, batch).await {
                    Ok(v) => {
                        line["ok"] = json!(true);
                        line["ingest"] = v;
                    }
                    Err(e) => {
                        line["ok"] = json!(false);
                        line["error"] = json!(e);
                    }
                }
                {
                    let mut g = model.lock().await;
                    if let Some(m) = g.as_ref() {
                        m.end_task_cleanup("synthetic eval").await;
                    }
                    *g = None;
                }
                println!("{}", sds::report());
                sds::flush();
                if let Some(db) = store.lock().await.as_ref() {
                    line["items"] = json!(changed_items(db, t0).await);
                }
            } else if str_of(entry, "kind") == "search" {
                let r = tauri_app_lib::eval_ai_search(
                    &handle,
                    format!("eval-{}", id),
                    str_of(entry, "query").to_string(),
                    str_of(entry, "language").to_string(),
                    str_of(entry, "mode").to_string(),
                    match str_of(entry, "host") {
                        "" => String::new(),
                        host => hash_id(host),
                    },
                )
                .await;
                match r {
                    Ok(v) => {
                        line["ok"] = json!(true);
                        line["search"] = v;
                    }
                    Err(e) => {
                        line["ok"] = json!(false);
                        line["error"] = json!(e);
                    }
                }
            } else {
                let task = build_task(entry, &team);
                cancel.store(false, std::sync::atomic::Ordering::SeqCst);
                tauri_app_lib::utils::set_extraction_stop_signal(false);
                if let Some(db) = store.lock().await.as_ref() {
                    let _ = db.add_task(task.clone()).await;
                }
                let r = process_task(task.clone(), &store, &model, &cancel, &handle, None).await;
                line["ok"] = json!(r.is_ok());
                if let Err(e) = r {
                    line["error"] = json!(format!("{:#}", e));
                }
                {
                    let mut g = model.lock().await;
                    if let Some(m) = g.as_ref() {
                        m.end_task_cleanup("synthetic eval").await;
                    }
                    *g = None;
                }
                println!("{}", sds::report());
                sds::flush();
                sds::leave_scope();
                if let Some(db) = store.lock().await.as_ref() {
                    line["items"] = json!(changed_items(db, t0).await);
                }
            }
            line["t1"] = json!(now_ms());
            line["secs"] = json!(started.elapsed().as_secs_f64());
            println!("@@EVAL END {} ok={} {:.1}s", id, line["ok"], started.elapsed().as_secs_f64());
            writeln!(out, "{}", line).expect("결과 쓰기");
            out.flush().expect("결과 쓰기");
        }
    });
    tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
}

/// 🌟 [SYNTH EVAL / VISION] 언어 모델 없이 SigLIP2 판정 단계만 잽니다.
///
/// PNG 폴더의 페이지마다 서식 판정 → 히트맵 → 아레나 → 크롭 계획까지 `extract_from_image` 와 같은 순서로 돌리고
/// 결과를 JSONL 로 씁니다. `LOGIS_SYNTH_DUMP` 를 주면 패치 임베딩과 풀링 벡터도 파일로 남겨 오프라인 실험에 씁니다.
/// `LOGIS_SYNTH_LANG` 은 앵커 언어입니다. 앱의 이미지 경로는 "korean" 으로 고정되어 있어 기본값도 그렇게 둡니다.
///
/// ```text
/// LOGIS_APP_DIR=<평가 폴더> LOGIS_SYNTH_PAGES=<PNG 폴더> LOGIS_SYNTH_OUT=<result.jsonl> \
/// cargo test --test it --no-default-features --features rocm -- synthetic_eval::vision_probe --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore = "실모델: LOGIS_SYNTH_PAGES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR"]
fn vision_probe() {
    use tauri_app_lib::models::siglip2::legibility::build_legibility_map;
    use tauri_app_lib::models::siglip2::vision_encoder::{build_column_heatmaps, classify_doc_type, encode_image};
    use tauri_app_lib::models::siglip2::{nms_arena, vision_crop, Siglip2Config, Siglip2Model};

    let (Ok(pages), Ok(out_path), Some(_)) = (
        std::env::var("LOGIS_SYNTH_PAGES"),
        std::env::var("LOGIS_SYNTH_OUT"),
        std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()),
    ) else {
        println!("LOGIS_SYNTH_PAGES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR 가 없어 건너뜁니다.");
        return;
    };
    let lang = std::env::var("LOGIS_SYNTH_LANG").unwrap_or_else(|_| "korean".to_string());
    let only = std::env::var("LOGIS_SYNTH_ONLY").unwrap_or_default();
    let dump = std::env::var("LOGIS_SYNTH_DUMP").ok().map(PathBuf::from);
    if let Some(d) = dump.as_ref() {
        std::fs::create_dir_all(d).expect("덤프 폴더");
    }
    let dir = crate::common::models_dir().join("siglip2-so400m-patch16-naflex");
    // `LOGIS_SYNTH_CPU=1`: CPU(F32)로 돕니다 — GPU 검색 평가와 함께 돌릴 때. 판정 코드는 같고 수치만 장치 · 정밀도 차이만큼 다릅니다.
    let cpu = std::env::var("LOGIS_SYNTH_CPU").map(|v| v == "1").unwrap_or(false);
    let device = if cpu { candle_core::Device::Cpu } else { tauri_app_lib::utils::new_gpu_device(0).expect("GPU") };
    let dtype = if cpu || device.is_vulkan() { candle_core::DType::F32 } else { candle_core::DType::BF16 };
    let config = Siglip2Config::from_json(&dir.join("config.json")).expect("config.json");
    let encode_only = std::env::var("LOGIS_SYNTH_ENCODE_ONLY").map(|v| v == "1").unwrap_or(false);
    let mut model = Siglip2Model::load_vision_only(&dir.join("model.safetensors"), &config, &device, dtype).expect("비전 타워");
    if !encode_only {
        model.load_text_encoder(&dir).expect("텍스트 타워");
    }
    // 패치 수 실험: 앱은 256 을 씁니다. 가중치의 위치 임베딩 격자는 그대로 두고 전처리의 상한만 바꿉니다.
    if let Some(n) = std::env::var("LOGIS_SYNTH_PATCHES").ok().and_then(|v| v.parse::<usize>().ok()) {
        model.config.max_num_patches = n;
    }
    println!("패치 상한 {}", model.config.max_num_patches);
    let emit = |m: &str| println!("{m}");

    let mut files: Vec<PathBuf> = std::fs::read_dir(&pages)
        .expect("PNG 폴더")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("png")))
        .filter(|p| only.is_empty() || p.to_string_lossy().contains(&only))
        .collect();
    files.sort();
    // `LOGIS_SYNTH_LIST`: 페이지 이름(확장자 없이) 한 줄에 하나. 그 순서로 그 페이지만 돕니다 — SDS 가 한 실행 안에서 쌓이므로
    //  기준선을 쌓을 페이지(dev)를 평가할 페이지(test) 앞에 두려면 순서를 정해야 합니다.
    if let Ok(list) = std::env::var("LOGIS_SYNTH_LIST") {
        let order: Vec<String> = std::fs::read_to_string(&list).expect("페이지 목록").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
        let stem = |p: &PathBuf| p.file_stem().unwrap_or_default().to_string_lossy().to_string();
        files = order.iter().filter_map(|s| files.iter().find(|p| stem(p) == *s).cloned()).collect();
    }
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");

    for png in files.iter() {
        let stem = png.file_stem().unwrap_or_default().to_string_lossy().to_string();
        println!("\n@@EVAL BEGIN {}", stem);
        let img = image::DynamicImage::ImageRgb8(image::open(png).expect("PNG").to_rgb8());
        // 앱의 이미지 경로처럼 'unknown' 스코프로 들어가 판정 뒤 서식 코드로 좁힙니다(SDS 관측 · 동적 하한이 같은 스코프를 봄).
        sds::enter_scope("", sds::Track::Vision, "unknown", "");
        let t0 = std::time::Instant::now();
        let grid = encode_image(&model, &img).expect("encode_image");
        let encode_secs = t0.elapsed().as_secs_f64();
        if let Some(d) = dump.as_ref() {
            let mut flat: Vec<u8> = Vec::with_capacity(grid.len() * 1152 * 4);
            for p in grid.patches.iter() {
                for x in p {
                    flat.extend_from_slice(&x.to_le_bytes());
                }
            }
            std::fs::write(d.join(format!("{stem}.patches.f32")), flat).expect("패치 덤프");
            let pooled: Vec<u8> = grid.pooled.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(d.join(format!("{stem}.pooled.f32")), pooled).expect("풀링 덤프");
        }
        if encode_only {
            writeln!(out, "{}", json!({"page": stem, "rows": grid.grid_rows, "cols": grid.grid_cols, "encode_secs": encode_secs})).expect("결과 쓰기");
            continue;
        }
        let legibility = build_legibility_map(&img, grid.grid_rows, grid.grid_cols, &emit);
        let verdict = classify_doc_type(&model, &grid, &emit).expect("classify_doc_type");
        sds::refine_primary(&verdict.code);
        let title_prej: Vec<String> =
            if verdict.title_text.is_empty() { Vec::new() } else { vec![verdict.title_text.clone()] };
        let mut heatmaps = build_column_heatmaps(&model, &grid, &verdict.code, &lang, Some(&legibility), &title_prej, &emit)
            .expect("build_column_heatmaps");
        let raw_maps: Vec<Value> = heatmaps
            .iter()
            .map(|h| {
                json!({
                    "category": h.category, "top_field": h.top_field, "top_score": h.top_score,
                    "active": h.scores.iter().filter(|s| **s > 0.0).count(),
                    "peaks": h.field_peaks.iter().map(|(f, p, z)| json!({"field": f, "patch": p, "z": z})).collect::<Vec<_>>(),
                })
            })
            .collect();
        let title_rows = (grid.grid_rows / 18).max(1).min(grid.grid_rows.saturating_sub(1));
        for hm in heatmaps.iter_mut() {
            for r in 0..=title_rows {
                for c in 0..grid.grid_cols {
                    let i = r * grid.grid_cols + c;
                    if i < hm.scores.len() && hm.scores[i] > f32::MIN {
                        hm.scores[i] = f32::MIN;
                    }
                }
            }
        }
        let mut protect: Vec<&str> = tauri_app_lib::logic::TRADE_ARRAY_CATEGORIES.to_vec();
        protect.push(tauri_app_lib::logic::TRADE_IDENTITY_CATEGORY);
        let arena = nms_arena::run_arena(&heatmaps, &grid, &legibility, &protect, &emit);
        nms_arena::apply_arena(&mut heatmaps, &arena, &emit);
        let plans = vision_crop::plan_crops(
            &heatmaps,
            &grid,
            &legibility,
            tauri_app_lib::logic::TRADE_ARRAY_CATEGORIES,
            tauri_app_lib::logic::TRADE_IDENTITY_CATEGORY,
            tauri_app_lib::logic::TRADE_IDENTITY_FIELD,
            &emit,
        );
        let line = json!({
            "page": stem, "rows": grid.grid_rows, "cols": grid.grid_cols, "encode_secs": encode_secs,
            "group": verdict.group, "group_score": verdict.group_score, "group_margin": verdict.group_margin,
            "code": verdict.code, "code_score": verdict.code_score, "code_margin": verdict.code_margin,
            "title_confirmed": verdict.title_confirmed, "title_text": verdict.title_text,
            "title_band": verdict.title_band, "conflict": verdict.conflict_candidates,
            "candidates": verdict.code_candidates.iter().take(12).collect::<Vec<_>>(),
            "heatmaps": raw_maps,
            "plans": plans.iter().map(|p| json!({
                "category": p.category, "bbox": [p.bbox.0, p.bbox.1, p.bbox.2, p.bbox.3], "score": p.score,
                "margin": p.margin, "patches": p.patch_count, "top_field": p.top_field,
            })).collect::<Vec<_>>(),
        });
        writeln!(out, "{}", line).expect("결과 쓰기");
        out.flush().expect("결과 쓰기");
        println!("@@EVAL END {} {}", stem, verdict.code);
    }
    tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
}

/// 🌟 [SYNTH EVAL / VISION GATES] 이미지 경로의 출처 판독성 게이트(verify_claims_v2)를 합성 열화 페이지로 잽니다.
///
/// `LOGIS_SYNTH_CASES` 는 SyntheticData/eval/make_vision_cases.py 의 출력 폴더(cases.jsonl · img/)입니다. 격자는 앱과 같은
/// NaFlex 전처리(preprocess_image), 판독성 지도는 build_legibility_map 으로 만들고 모델 가중치는 읽지 않습니다(config.json 만).
/// 주장의 출처 상자는 `LOGIS_SYNTH_BBOX`(block · line, 기본 block) 한 종류만 씁니다. 두 종류를 한 실행에 섞으면 같은 값이
/// SDS 기준선에 두 번 들어갑니다. SDS 는 원장을 읽지 않고 이 실행 안에서만 쌓입니다(calib 쪽이 앞에 옵니다).
///
/// ```text
/// LOGIS_APP_DIR=<평가 폴더> LOGIS_SYNTH_CASES=<cases 폴더> LOGIS_SYNTH_OUT=<result.jsonl> [LOGIS_SYNTH_BBOX=line] \
/// cargo test --test it --no-default-features --features rocm -- synthetic_eval::vision_gates --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore = "합성 페이지: LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR"]
fn vision_gates() {
    use tauri_app_lib::models::siglip2::legibility::build_legibility_map;
    use tauri_app_lib::models::siglip2::preprocessor::preprocess_image;
    use tauri_app_lib::models::siglip2::value_grounding::{verify_claims_v2, GroundingClaim};
    use tauri_app_lib::models::siglip2::Siglip2Config;

    let (Ok(dir), Ok(out_path), Some(_)) = (
        std::env::var("LOGIS_SYNTH_CASES"),
        std::env::var("LOGIS_SYNTH_OUT"),
        std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()),
    ) else {
        println!("LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR 가 없어 건너뜁니다.");
        return;
    };
    let dir = PathBuf::from(dir);
    let bbox_kind = std::env::var("LOGIS_SYNTH_BBOX").unwrap_or_else(|_| "block".to_string());
    let config = Siglip2Config::from_json(&crate::common::models_dir().join("siglip2-so400m-patch16-naflex").join("config.json"))
        .expect("config.json");
    let quiet = |_: &str| {};
    let text = std::fs::read_to_string(dir.join("cases.jsonl")).expect("cases.jsonl");
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: Value = serde_json::from_str(line).expect("사례 JSON");
        let image = PathBuf::from(str_of(&case, "image"));
        let image = if image.is_absolute() { image } else { dir.join(image) };
        let img = image::DynamicImage::ImageRgb8(image::open(&image).expect("PNG").to_rgb8());
        let pre = preprocess_image(&img, &config, &candle_core::Device::Cpu).expect("NaFlex 전처리");
        let legibility = build_legibility_map(&img, pre.grid_rows, pre.grid_cols, &quiet);
        // calib 는 서식 코드 스코프에 쌓고, 평가 사례는 사례마다 따로 된 하위 스코프에 씁니다. 하위 스코프는 표본이 없으니 읽기는
        // 서식 코드 스코프(= calib 만)로 내려갑니다 — 같은 쪽의 다른 열화본이 서로의 기준선이 되지 않습니다.
        let secondary = if str_of(&case, "kind") == "calib" { "" } else { str_of(&case, "case") };
        sds::enter_scope("", sds::Track::Vision, str_of(&case, "code"), secondary);
        let picked: Vec<&Value> = case["claims"]
            .as_array()
            .map(|a| a.iter().filter(|c| str_of(c, "bbox_kind") == bbox_kind).collect())
            .unwrap_or_default();
        let claims: Vec<GroundingClaim> = picked
            .iter()
            .map(|c| {
                let b: Vec<u32> = c["bbox"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
                GroundingClaim {
                    category: str_of(c, "category").to_string(),
                    field: str_of(c, "field").to_string(),
                    value: str_of(c, "value").to_string(),
                    bbox: (b[0], b[1], b[2], b[3]),
                }
            })
            .collect();
        let before = sds::adaptive_baseline_n("vision.source_illegible_ratio");
        let verdicts = verify_claims_v2(&claims, pre.grid_rows, pre.grid_cols, img.width(), img.height(), &legibility, str_of(&case, "lang"), &quiet);
        let rows: Vec<Value> = claims
            .iter()
            .zip(picked.iter())
            .zip(verdicts.iter())
            .map(|((c, src), v)| {
                let (lg, il, bl) = legibility.count_in_bbox(c.bbox, img.width(), img.height());
                json!({"field": c.field, "value": c.value, "expect": str_of(src, "expect"), "kept": v.accepted, "verdict_value": v.value,
                       "gate": format!("{:?}", v.gate), "reason": v.reason, "lg": lg, "il": il, "bl": bl})
            })
            .collect();
        writeln!(
            out,
            "{}",
            json!({"case": str_of(&case, "case"), "kind": str_of(&case, "kind"), "code": str_of(&case, "code"), "lang": str_of(&case, "lang"),
                   "bbox_kind": bbox_kind, "grid": [pre.grid_rows, pre.grid_cols], "split_applied": legibility.split_applied,
                   "sds_before": before.map(|(m, s, n)| json!([m, s, n])), "claims": rows})
        )
        .expect("결과 쓰기");
    }
    sds::leave_scope();
    println!("@@EVAL VISION GATES {} → {}", bbox_kind, out_path);
}

/// 🌟 [SYNTH EVAL / NAME RESIDUE] 인명 잔해 깎기(strip_name_residue)를 names.jsonl(make_vision_cases.py)로 잽니다. 모델 없음.
/// 결과 `{id, value}` 는 score_vision_gates.py names 가 읽습니다.
#[test]
#[ignore = "합성 인명: LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT"]
fn name_residue_cases() {
    use tauri_app_lib::models::siglip2::value_grounding::{strip_name_residue, GroundingClaim};
    let (Ok(dir), Ok(out_path)) = (std::env::var("LOGIS_SYNTH_CASES"), std::env::var("LOGIS_SYNTH_OUT")) else {
        println!("LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT 이 없어 건너뜁니다.");
        return;
    };
    let text = std::fs::read_to_string(PathBuf::from(dir).join("names.jsonl")).expect("names.jsonl");
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: Value = serde_json::from_str(line).expect("인명 JSON");
        let field = str_of(&case, "field").to_string();
        let mut claims = vec![GroundingClaim {
            category: "parties".to_string(),
            field: field.clone(),
            value: str_of(&case, "value").to_string(),
            bbox: (0, 0, 1, 1),
        }];
        let mut slot = serde_json::Map::new();
        slot.insert(field.clone(), json!(str_of(&case, "value")));
        let mut data = serde_json::Map::new();
        data.insert(field.clone(), json!(str_of(&case, "value")));
        data.insert("parties".to_string(), Value::Object(slot));
        strip_name_residue(&mut claims, &mut data, &|_: &str| {});
        let stored = [data.get(&field).and_then(|v| v.as_str()), data["parties"].get(&field).and_then(|v| v.as_str())];
        assert!(stored.iter().all(|s| *s == Some(claims[0].value.as_str())), "저장본과 주장이 같이 고쳐져야 합니다");
        writeln!(out, "{}", json!({"id": str_of(&case, "id"), "value": claims[0].value})).expect("결과 쓰기");
    }
    println!("@@EVAL NAME RESIDUE → {}", out_path);
}

/// 🌟 [SYNTH EVAL / LABEL GATES] 인쇄 라벨 → 스키마 축 라우팅(route_pairs_to_fields_exact)을 언어 팩 라벨로 잽니다.
///
/// `LOGIS_SYNTH_CASES` 폴더의 labels.jsonl(make_vision_cases.py)을 읽습니다. 서식(언어 × 코드) 하나를 문서 하나로 보고,
/// 앱과 같은 라벨 뱅크(스키마 필드 · 자기 참조 축 제외 · owner_label_bank_for_doc)를 granite 로 임베딩해 라벨마다 따로
/// 라우팅합니다(한 창에 넣으면 같은 축을 두 라벨이 다투는 규칙이 끼어 라벨 하나의 판정이 가려집니다). 혼동 사전은 이 실행
/// 안에서만 쌓입니다. `LOGIS_SYNTH_CPU=1` 이면 granite 를 CPU 로 돌립니다(다른 GPU 평가와 함께 돌릴 때).
#[test]
#[ignore = "실모델(granite): LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR"]
fn label_gates() {
    use std::collections::HashMap;
    use tauri_app_lib::model::merge::{
        build_label_exact_index, label_axis_scores, label_exact_field, owner_label_bank_for_doc, route_pairs_to_fields_exact,
    };
    use tauri_app_lib::models::embedding::EmbeddingModel;

    let (Ok(dir), Ok(out_path), Some(_)) = (
        std::env::var("LOGIS_SYNTH_CASES"),
        std::env::var("LOGIS_SYNTH_OUT"),
        std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()),
    ) else {
        println!("LOGIS_SYNTH_CASES · LOGIS_SYNTH_OUT · LOGIS_APP_DIR 가 없어 건너뜁니다.");
        return;
    };
    let text = std::fs::read_to_string(PathBuf::from(dir).join("labels.jsonl")).expect("labels.jsonl");
    let rows: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("라벨 JSON")).collect();
    let cpu = std::env::var("LOGIS_SYNTH_CPU").map(|v| v == "1").unwrap_or(false);
    // `LOGIS_SYNTH_SEED_CONFUSION=1`: 라벨마다 1 · 2위 축 쌍을 혼동 사전에 미리 적어 두는 상한 측정(아래).
    let seed_confusion = std::env::var("LOGIS_SYNTH_SEED_CONFUSION").map(|v| v == "1").unwrap_or(false);
    let device = if cpu { candle_core::Device::Cpu } else { tauri_app_lib::utils::new_gpu_device(0).expect("GPU") };
    let gdir = crate::common::model_subdir("granite-embedding-97m-multilingual-r2", "model.safetensors").expect("granite");
    let model = EmbeddingModel::new_with_device(&gdir, &device).expect("granite 로드");
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    let mut embed = |texts: &[String]| -> Vec<Vec<f32>> {
        let mut miss: Vec<String> = texts.iter().filter(|t| !cache.contains_key(*t)).cloned().collect();
        miss.sort();
        miss.dedup();
        for part in miss.chunks(256) {
            for (t, v) in part.iter().zip(model.embed_batch(part).expect("임베딩")) {
                cache.insert(t.clone(), v);
            }
        }
        texts.iter().map(|t| cache[t].clone()).collect()
    };

    let mut docs: Vec<(String, Vec<&Value>)> = Vec::new();
    for r in rows.iter() {
        match docs.iter_mut().find(|(d, _)| d == str_of(r, "doc")) {
            Some((_, v)) => v.push(r),
            None => docs.push((str_of(r, "doc").to_string(), vec![r])),
        }
    }
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");
    for (doc, items) in docs.iter() {
        let (code, lang) = (str_of(items[0], "code"), str_of(items[0], "lang"));
        sds::enter_scope("", sds::Track::Vision, code, "");
        let self_ref = tauri_app_lib::logic::trade_reference_field_of(code).unwrap_or("");
        let fields: Vec<String> = tauri_app_lib::parsing::get_detail_schema_fields(code, "", lang)
            .into_iter()
            .map(|(f, _, _, _)| f)
            .filter(|f| !tauri_app_lib::bias_schema::is_system_axis(f))
            .filter(|f| self_ref.is_empty() || f != self_ref)
            .collect();
        let per_field: Vec<(String, Vec<String>, Vec<f32>)> = fields
            .iter()
            .map(|f| {
                let (p, w) = owner_label_bank_for_doc(lang, "shipping_doc", f, self_ref);
                (f.clone(), p, w)
            })
            .collect();
        let phrases: Vec<String> = per_field.iter().flat_map(|(_, p, _)| p.iter().cloned()).collect();
        let table: HashMap<String, Vec<f32>> = phrases.iter().cloned().zip(embed(&phrases)).collect();
        let banks: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = per_field
            .iter()
            .map(|(f, p, w)| {
                let (mut b, mut ww) = (Vec::new(), Vec::new());
                for (ph, x) in p.iter().zip(w.iter()) {
                    if let Some(e) = table.get(ph).filter(|e| !e.is_empty()) {
                        b.push(e.clone());
                        ww.push(*x);
                    }
                }
                (f.clone(), b, ww)
            })
            .collect();
        let exact = build_label_exact_index(&per_field.iter().map(|(f, p, _)| (f.clone(), p.clone())).collect::<Vec<_>>());
        let labels: Vec<String> = items.iter().map(|r| str_of(r, "label").to_string()).collect();
        let label_embs = embed(&labels);
        for (r, e) in items.iter().zip(label_embs.iter()) {
            if seed_confusion {
                // 상한 측정: 이 라벨의 1 · 2위 축을 혼동 사전에 비전 트랙 표본 하한(8)만큼 미리 적어 둡니다 — 혼동 기록이 있으면
                // 재측정하는 규칙이 이 쌍에서 반드시 발화합니다(기록이 없는 실행과 비교).
                let top2: Vec<String> = label_axis_scores(e, &banks).into_iter().take(2).map(|(f, _)| f).collect();
                if top2.len() == 2 {
                    for _ in 0..8 {
                        sds::record_confusion(&top2[0], &top2[1], 0.0);
                    }
                }
            }
            let pair = vec![(str_of(r, "label").to_string(), str_of(r, "value").to_string())];
            let (routes, logs, row_axis) = route_pairs_to_fields_exact(&pair, std::slice::from_ref(e), &fields, &banks, &exact);
            let top: Vec<Value> = label_axis_scores(e, &banks).into_iter().take(3).map(|(f, z)| json!([f, z])).collect();
            writeln!(
                out,
                "{}",
                json!({"id": str_of(r, "id"), "doc": doc, "lang": lang, "code": code, "label": str_of(r, "label"),
                       "expect": str_of(r, "expect"), "routed": routes.first().map(|x| x.field.clone()),
                       "own": routes.first().map(|x| x.own), "row_axis": row_axis.first().map(|x| x.1.clone()),
                       "exact": label_exact_field(str_of(r, "label"), &exact, &banks), "top": top, "logs": logs,
                       "seeded": seed_confusion})
            )
            .expect("결과 쓰기");
        }
    }
    sds::leave_scope();
    println!("@@EVAL LABEL GATES {}건 → {}", rows.len(), out_path);
    if !cpu {
        tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
    }
}

/// 🌟 [SYNTH EVAL / EMBED] 문장 목록을 앱과 같은 모델 코드로 임베딩해 f32 행렬 파일로 씁니다. 오프라인 실험용입니다.
///
/// `LOGIS_SYNTH_TEXTS` 는 JSON 문자열 배열 파일, `LOGIS_SYNTH_MODEL` 은 `granite`(384차원) 또는 `siglip2`(1152차원)입니다.
#[test]
#[ignore = "실모델: LOGIS_SYNTH_TEXTS · LOGIS_SYNTH_OUT · LOGIS_SYNTH_MODEL"]
fn embed_dump() {
    use tauri_app_lib::models::embedding::EmbeddingModel;
    use tauri_app_lib::models::siglip2::vision_encoder::encode_phrase_rows;
    use tauri_app_lib::models::siglip2::{Siglip2Config, Siglip2Model};

    let (Ok(texts_path), Ok(out_path), Ok(which)) = (
        std::env::var("LOGIS_SYNTH_TEXTS"),
        std::env::var("LOGIS_SYNTH_OUT"),
        std::env::var("LOGIS_SYNTH_MODEL"),
    ) else {
        println!("LOGIS_SYNTH_TEXTS · LOGIS_SYNTH_OUT · LOGIS_SYNTH_MODEL 이 없어 건너뜁니다.");
        return;
    };
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&texts_path).expect("문장 파일")).expect("JSON 배열");
    let device = tauri_app_lib::utils::new_gpu_device(0).expect("GPU");
    let rows: Vec<Vec<f32>> = if which == "siglip2" {
        let dir = crate::common::models_dir().join("siglip2-so400m-patch16-naflex");
        let dtype = if device.is_vulkan() { candle_core::DType::F32 } else { candle_core::DType::BF16 };
        let config = Siglip2Config::from_json(&dir.join("config.json")).expect("config.json");
        let model = Siglip2Model::load_text_only(&dir, &config, &device, dtype).expect("텍스트 타워");
        encode_phrase_rows(&model, &texts, true).expect("인코딩")
    } else {
        let dir = crate::common::model_subdir("granite-embedding-97m-multilingual-r2", "model.safetensors").expect("granite");
        let model = EmbeddingModel::new_with_device(&dir, &device).expect("granite 로드");
        let mut out = Vec::with_capacity(texts.len());
        for part in texts.chunks(256) {
            out.extend(model.embed_batch(part).expect("임베딩"));
        }
        out
    };
    let mut flat: Vec<u8> = Vec::with_capacity(rows.len() * rows.first().map_or(0, |r| r.len()) * 4);
    for r in rows.iter() {
        for x in r {
            flat.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&out_path, flat).expect("결과 쓰기");
    println!("@@EVAL EMBED {} rows × {} dims → {}", rows.len(), rows.first().map_or(0, |r| r.len()), out_path);
    tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
}

/// 🌟 [SYNTH EVAL / CENSUS] DOM 구조 집계만 돌립니다. 모델이 필요 없어 GPU 작업과 함께 돌릴 수 있습니다.
#[test]
#[ignore = "합성 데이터: LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT"]
fn census_probe() {
    let (Ok(plan_path), Ok(out_path)) = (std::env::var("LOGIS_SYNTH_PLAN"), std::env::var("LOGIS_SYNTH_OUT")) else {
        println!("LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT 가 없어 건너뜁니다.");
        return;
    };
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");
    for l in std::fs::read_to_string(&plan_path).expect("계획 파일").lines() {
        let entry: Value = serde_json::from_str(l).expect("계획 줄");
        if str_of(&entry, "kind") != "html" {
            continue;
        }
        let html = std::fs::read_to_string(str_of(&entry, "path")).expect("html");
        let clean = tauri_app_lib::parsing::pre_clean_html(&html);
        let census = tauri_app_lib::scheduler::list_census::run(&clean);
        let groups: Vec<Value> = census
            .groups
            .iter()
            .map(|g| {
                json!({
                    "selector": g.item_selector, "tag": g.tag, "members": g.members, "avg_cells": g.avg_cells,
                    "distinct": g.distinct_ratio, "form_like": g.form_like, "score": g.score, "mirror": g.mirror,
                    "structural": g.structural_grade(), "sample": g.samples.first(),
                })
            })
            .collect();
        writeln!(out, "{}", json!({"id": str_of(&entry, "id"), "groups": groups})).expect("결과 쓰기");
    }
}

/// 🌟 [SYNTH EVAL / IDS] `seed` 레코드의 저장 id 를 정답 키와 잇는 표를 씁니다. 저장 id 는 해시라 파이썬에서 다시 만들 수 없습니다.
#[test]
#[ignore = "합성 데이터: LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT"]
fn seed_ids() {
    let (Ok(plan_path), Ok(out_path)) = (std::env::var("LOGIS_SYNTH_PLAN"), std::env::var("LOGIS_SYNTH_OUT")) else {
        println!("LOGIS_SYNTH_PLAN · LOGIS_SYNTH_OUT 가 없어 건너뜁니다.");
        return;
    };
    let mut map = serde_json::Map::new();
    for l in std::fs::read_to_string(&plan_path).expect("계획 파일").lines() {
        let entry: Value = serde_json::from_str(l).expect("계획 줄");
        if str_of(&entry, "kind") == "seed" {
            map.insert(hash_id(str_of(&entry, "id")), json!(str_of(&entry, "id")));
        }
    }
    std::fs::write(&out_path, Value::Object(map).to_string()).expect("결과 쓰기");
}
