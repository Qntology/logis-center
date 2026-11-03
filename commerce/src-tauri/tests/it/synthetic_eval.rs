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
    let device = tauri_app_lib::utils::new_gpu_device(0).expect("GPU");
    let dtype = if device.is_vulkan() { candle_core::DType::F32 } else { candle_core::DType::BF16 };
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
    let mut out = std::fs::File::create(&out_path).expect("결과 파일");

    for png in files.iter() {
        let stem = png.file_stem().unwrap_or_default().to_string_lossy().to_string();
        println!("\n@@EVAL BEGIN {}", stem);
        let img = image::DynamicImage::ImageRgb8(image::open(png).expect("PNG").to_rgb8());
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
