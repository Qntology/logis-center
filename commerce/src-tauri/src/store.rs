use anyhow::Result;
use lancedb::{Connection, connect};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::arrow::arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, FixedSizeListArray, Array, Int32Array, BooleanArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use serde_json::{Value, json};
use futures::TryStreamExt;

const DB_URI: &str = "data/lancedb";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub id: String,
    pub r#type: String,
    pub from: String, 
    pub to: String,     
    pub cc: String,
    pub bcc: String,
    #[serde(rename = "ref")]
    pub r#ref: String,
    #[serde(rename = "data")]
    pub data_json: String,   
    pub created_at: i64,
    pub updated_at: i64,
    pub status: i32,      
}

#[derive(Clone)]
pub struct VectorStore {
    conn: Connection,
    base_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AppConfig {
    pub is_logged_in: bool,
    pub auth_token: Option<String>,
}

pub const ENVELOPE_COLUMNS: [&str; 11] = [
    "id", "type", "flag", "from", "to", "cc", "bcc", "ref", "mode",
    "created_at", "updated_at",
];

pub const ANALYTIC_TYPES: [&str; 7] = [
    "click", "hover", "change", "touch", "report", "question", "answer",
];
pub const TALK_TYPES: [&str; 3] = ["talk", "prompt", "ai_search"];
pub const COMMERCE_TYPES: [&str; 9] = [
    "sales", "goods", "order", "tracking", "event", "coupon", "review",
    "receiving", "shipping",
];
pub const USER_TYPES: [&str; 3] = ["member", "team", "user"];

pub fn is_analytic_type(type_: &str) -> bool {
    let t = type_.trim().to_lowercase();
    ANALYTIC_TYPES.iter().any(|x| *x == t)
}
pub fn is_talk_type(type_: &str) -> bool {
    let t = type_.trim().to_lowercase();
    TALK_TYPES.iter().any(|x| *x == t)
}
pub fn is_user_type(type_: &str) -> bool {
    let t = type_.trim().to_lowercase();
    USER_TYPES.iter().any(|x| *x == t)
}
pub fn is_reserved_type(type_: &str) -> bool {
    let t = type_.trim().to_lowercase();
    t == "unknown"
        || t == "users" || t == "pages" || t == "page"
        || is_user_type(&t)
        || is_talk_type(&t)
        || is_analytic_type(&t)
        || COMMERCE_TYPES.iter().any(|x| *x == t)
}
pub fn infer_mode(type_: &str) -> &'static str {
    if crate::utils::bias_schema::is_trade_doc_type(type_) {
        "shipping"
    } else if is_analytic_type(type_) {
        "analytic"
    } else {
        "commerce"
    }
}
pub fn resolve_table_for(table_or_type: &str) -> &'static str {
    let t = if table_or_type.starts_with("commerce_") {
        &table_or_type[9..]
    } else {
        table_or_type
    };
    match t {
        // 사용자/팀 : 라이프사이클이 달라 물리 분리 유지
        "users" | "member" | "team" | "user" => "users",
        // 페이지 셀렉터 캐시 : 검색 대상이 아니라 물리 분리 유지
        "pages" | "page" => "pages",
        // 그 외 전부 items (sales/tracking/event/goods/order/coupon/review/talk/...)
        _ => "items",
    }
}

/// 커머스 도메인 조회 축 기본값(canonicalize_data 의 SEED_KEYS)을 시딩해야 하는가.
/// ⚠️ main.ts 의 NON_SEED_TYPES 와 반드시 같은 집합이어야 두 저장소가 일치합니다.
pub fn needs_domain_seed(target_table: &str, type_: &str) -> bool {
    if matches!(target_table, "users" | "pages") { return false; }
    if is_user_type(type_) { return false; }
    if is_analytic_type(type_) { return false; }
    true
}

/// 로컬 임베딩(reindex_pending_embeddings) 대상에서 제외할 타입인가.
/// analytic 원시 이벤트(click/hover/change/touch)와 report 는 제외하지 않습니다.
/// 그것들이 빠지면 D1 에서 받아온 행동 로그가 검색에 절대 잡히지 않습니다.
pub fn is_embed_excluded_type(type_: &str) -> bool {
    let t = type_.trim().to_lowercase();
    if t == "pages" || t == "page" { return true; }
    if is_talk_type(&t) { return true; }
    if t == "users" || is_user_type(&t) { return true; }
    // question / answer 는 관리자 채팅 말풍선이며 parse_analytic_query 의 검색 스코프에서도 제외됩니다.
    if t == "question" || t == "answer" { return true; }
    false
}

/// 봉투 값 1개를 확정합니다. 인자 → data → 빈 문자열 순으로 우선합니다.
/// upsert_item 이 물리 컬럼과 data 양쪽에 '같은 값' 을 쓰기 위한 단일 판정기입니다.
pub fn resolve_envelope_field(arg: Option<&str>, data: &Value, key: &str) -> String {
    if let Some(v) = arg {
        let t = v.trim();
        if !t.is_empty() { return t.to_string(); }
    }
    data.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

impl VectorStore {
    pub async fn new(base_path: &str) -> Result<Self> {
        let conn = connect(base_path).execute().await?;
        Ok(Self { conn, base_path: base_path.to_string() })
    }

    pub fn load_config(&self) -> AppConfig {
        let path = std::path::Path::new(&self.base_path).join("settings.json");
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(config) = serde_json::from_str(&content) { return config; }
            }
        }
        AppConfig::default()
    }

    pub fn save_config(&self, config: &AppConfig) -> Result<()> {
        let path = std::path::Path::new(&self.base_path).join("settings.json");
        let json = serde_json::to_string_pretty(config)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub async fn init_task_table(&self) -> Result<()> {
        let task_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("from", DataType::Utf8, false),
            Field::new("to", DataType::Utf8, false),
            Field::new("cc", DataType::Utf8, false),
            Field::new("bcc", DataType::Utf8, false),
            Field::new("ref", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false), 
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            Field::new("status", DataType::Int32, false), 
        ]));

        let uri = self.base_path.clone();
        let existing = self.conn.table_names().execute().await?;
        
        if existing.contains(&"tasks".to_string()) {
            match self.conn.open_table("tasks").execute().await {
                Ok(table) => {
                    let current_schema = table.schema().await.unwrap_or_else(|_| Arc::new(Schema::new(Vec::<Field>::new())));
                    let has_ref = current_schema.field_with_name("ref").is_ok();
                    let status_is_int = if let Ok(field) = current_schema.field_with_name("status") {
                        field.data_type() == &DataType::Int32
                    } else { false };

                    if !has_ref || !status_is_int {
                        println!("[Store] tasks table schema mismatch. Dropping for recreation.");
                        let _ = self.conn.drop_table("tasks", &[]).await;
                    }
                },
                Err(_) => {
                    
                    println!("[Store] Corrupted tasks table detected. Force dropping.");
                    let _ = self.conn.drop_table("tasks", &[]).await;
                    let _ = std::fs::remove_dir_all(format!("{}/tasks.lance", uri));
                }
            }
        }
        
        let existing = self.conn.table_names().execute().await?;
        if !existing.contains(&"tasks".to_string()) {
            if let Err(_) = self.conn.create_empty_table("tasks", task_schema.clone()).execute().await {
                println!("[Store] tasks create failed, cleaning up dir and retrying...");
                let _ = std::fs::remove_dir_all(format!("{}/tasks.lance", uri));
                let _ = self.conn.create_empty_table("tasks", task_schema).execute().await;
            }
        }

        let msg_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("role", DataType::Utf8, false), 
            Field::new("from", DataType::Utf8, true),
            Field::new("to", DataType::Utf8, true),
            Field::new("cc", DataType::Utf8, true),
            Field::new("bcc", DataType::Utf8, true),
            Field::new("ref", DataType::Utf8, true),
            Field::new("text", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, true), 
            Field::new("task_id", DataType::Utf8, true),
            Field::new("status", DataType::Int32, true), 
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
        ]));

        let existing = self.conn.table_names().execute().await?;
        if existing.contains(&"talks".to_string()) {
            match self.conn.open_table("talks").execute().await {
                Ok(table) => {
                    let current_schema = table.schema().await.unwrap_or_else(|_| Arc::new(Schema::new(Vec::<Field>::new())));
                    let needs_recreate = current_schema.field_with_name("text").is_err();
                    if needs_recreate {
                        println!("[Store] talks table schema outdated. Dropping for migration.");
                        let _ = self.conn.drop_table("talks", &[]).await;
                    }
                },
                Err(_) => {
                    println!("[Store] Corrupted talks table detected. Force dropping.");
                    let _ = self.conn.drop_table("talks", &[]).await;
                    let _ = std::fs::remove_dir_all(format!("{}/talks.lance", uri));
                }
            }
        }
        
        let existing = self.conn.table_names().execute().await?;
        if !existing.contains(&"talks".to_string()) {
            if let Err(_) = self.conn.create_empty_table("talks", msg_schema.clone()).execute().await {
                let _ = std::fs::remove_dir_all(format!("{}/talks.lance", uri));
                let _ = self.conn.create_empty_table("talks", msg_schema).execute().await;
            }
        }
        Ok(())
    }

    pub async fn has_active_task(&self, cc: &str, r#ref: &str) -> Result<bool> {
        let table = self.conn.open_table("tasks").execute().await?;
        // [FIX] Use backticks for 'ref' to avoid reserved keyword conflicts in LanceDB/DataFusion
        let filter = format!("cc = '{}' AND `ref` = '{}' AND (status = 10 OR status = 1)", cc, r#ref);
        let results = table.query()
            .only_if(filter)
            .limit(1).execute().await?.try_collect::<Vec<_>>().await?;
        Ok(!results.is_empty())
    }

    pub async fn add_message(
        &self, id: &str, role: &str, text: &str, task_id: Option<&str>, status: Option<i32>,
        cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>,
        from: Option<&str>, to: Option<&str>, type_: Option<&str>, data: Option<&str>
    ) -> Result<()> {
        // 🌟 [DELEGATE] 삽입 로직을 add_message_at 한 곳으로 모읍니다.
        //    기존 구현은 updated_at 을 0 으로 고정했는데,
        //    main.ts 의 loadMoreChat 이 `updated_at > latestUpdateTime` 으로
        //    델타 동기화를 시도하므로 그 경로가 영구히 죽어 있었습니다.
        //    (매 폴링마다 전량을 다시 가져오고 있었습니다)
        self.add_message_at(
            id, role, text, task_id, status,
            cc, bcc, r#ref, from, to, type_, data,
            None
        ).await
    }

    /// 🌟 [MESSAGE UPDATE HELPER] created_at 을 명시적으로 지정할 수 있는 삽입 함수입니다.
    ///  update_message_status 가 '삭제 후 재삽입' 방식이라,
    ///  기존 add_message 를 그대로 쓰면 created_at 이 매번 현재 시각으로 갱신되어
    ///  main.ts 가 유지하던 '질문 → 작업' 정렬이 흔들립니다.
    ///  updated_at 도 0 대신 현재 시각을 넣어 프론트엔드의 델타 동기화
    ///  (`updated_at > latestUpdateTime`)가 실제로 동작하게 합니다.
    pub async fn add_message_at(
        &self, id: &str, role: &str, text: &str, task_id: Option<&str>, status: Option<i32>,
        cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>,
        from: Option<&str>, to: Option<&str>, type_: Option<&str>, data: Option<&str>,
        created_at: Option<i64>
    ) -> Result<()> {
        let table = self.conn.open_table("talks").execute().await?;
        let schema = table.schema().await?;
        let now = chrono::Utc::now().timestamp_millis();
        let created = created_at.filter(|v| *v > 0).unwrap_or(now);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id.to_string()])),
                Arc::new(StringArray::from(vec![type_.unwrap_or("talk").to_string()])),
                Arc::new(StringArray::from(vec![role.to_string()])),
                Arc::new(StringArray::from(vec![from.unwrap_or("").to_string()])),
                Arc::new(StringArray::from(vec![to.unwrap_or("").to_string()])),
                Arc::new(StringArray::from(vec![cc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![bcc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![r#ref.unwrap_or("")])),
                Arc::new(StringArray::from(vec![text.to_string()])),
                Arc::new(StringArray::from(vec![data.unwrap_or("").to_string()])),
                Arc::new(StringArray::from(vec![task_id.unwrap_or("").to_string()])),
                Arc::new(Int32Array::from(vec![status.unwrap_or(0)])),
                Arc::new(Int64Array::from(vec![created])),
                Arc::new(Int64Array::from(vec![now])),
            ],
        )?;
        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn get_all_messages(&self, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<Value>> {
        let table = self.conn.open_table("talks").execute().await?;
        let mut q = table.query();
        if let Some(f) = filter { 
            if !f.trim().is_empty() {
                q = q.only_if(f); 
            }
        }
        
        // [FIX] Fetch all matching rows first to sort them accurately before applying limit/offset
        // Since local chat logs are typically small (<10k rows), this is safe and reliable.
        let results: Vec<RecordBatch> = q.execute().await?.try_collect::<Vec<_>>().await?;
            
        let mut msgs = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let roles = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let froms = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let tos = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            let ccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
            let bccs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
            let refs = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
            let texts = batch.column(8).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(9).as_any().downcast_ref::<StringArray>().unwrap();
            let task_ids = batch.column(10).as_any().downcast_ref::<StringArray>().unwrap();
            let statuses = batch.column(11).as_any().downcast_ref::<Int32Array>().unwrap();
            let createds = batch.column(12).as_any().downcast_ref::<Int64Array>().unwrap();
            let updateds = batch.column(13).as_any().downcast_ref::<Int64Array>().unwrap();

            for i in 0..batch.num_rows() {
                msgs.push(json!({
                    "id": ids.value(i), "type": types.value(i), "role": roles.value(i), 
                    "from": froms.value(i), "to": tos.value(i), "cc": ccs.value(i), 
                    "bcc": bccs.value(i), "ref": refs.value(i), "text": texts.value(i), 
                    "data": datas.value(i), "task_id": task_ids.value(i), "status": statuses.value(i), 
                    "created_at": createds.value(i), "updated_at": updateds.value(i)
                }));
            }
        }
        
        // [ORDER] Sort by created_at DESC (Latest messages first)
        msgs.sort_by(|a, b| b["created_at"].as_i64().unwrap_or(0).cmp(&a["created_at"].as_i64().unwrap_or(0)));
        
        // [PAGING] Apply limit and offset in memory
        let start = offset.min(msgs.len());
        let end = (start + limit).min(msgs.len());
        let paged_msgs = msgs[start..end].to_vec();
        
        Ok(paged_msgs)
    }

    pub async fn add_task(&self, task: Task) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        let schema = table.schema().await?;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![task.id])),
                Arc::new(StringArray::from(vec![task.r#type])),
                Arc::new(StringArray::from(vec![task.from])),
                Arc::new(StringArray::from(vec![task.to])),
                Arc::new(StringArray::from(vec![task.cc])),
                Arc::new(StringArray::from(vec![task.bcc])),
                Arc::new(StringArray::from(vec![task.r#ref])),
                Arc::new(StringArray::from(vec![task.data_json])),
                Arc::new(Int64Array::from(vec![task.created_at])),
                Arc::new(Int64Array::from(vec![task.updated_at])),
                Arc::new(Int32Array::from(vec![task.status])),
            ],
        )?;
        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn get_pending_tasks(&self, limit: usize) -> Result<Vec<Task>> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let filter = "status = 10"; 
        let results = table.query().only_if(filter).limit(limit).execute().await?.try_collect::<Vec<_>>().await?;
        let mut tasks = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let froms = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let tos = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let ccs = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            let bccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
            let refs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
            let crs = batch.column(8).as_any().downcast_ref::<Int64Array>().unwrap();
            let ups = batch.column(9).as_any().downcast_ref::<Int64Array>().unwrap();
            let sts = batch.column(10).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..batch.num_rows() {
                tasks.push(Task {
                    id: ids.value(i).to_string(), r#type: types.value(i).to_string(), from: froms.value(i).to_string(), 
                    to: tos.value(i).to_string(), cc: ccs.value(i).to_string(), bcc: bccs.value(i).to_string(), 
                    r#ref: refs.value(i).to_string(), data_json: datas.value(i).to_string(), 
                    created_at: crs.value(i), updated_at: ups.value(i), status: sts.value(i),
                });
            }
        }
        tasks.sort_by_key(|t| t.created_at);
        Ok(tasks)
    }

    
    pub async fn get_processing_tasks(&self, limit: usize) -> Result<Vec<Task>> {
        let table = self.conn.open_table("tasks").execute().await?;
        let filter = "status = 1"; 
        let results = table.query().only_if(filter).limit(limit).execute().await?.try_collect::<Vec<_>>().await?;
        let mut tasks = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let froms = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let tos = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let ccs = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            let bccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
            let refs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
            let crs = batch.column(8).as_any().downcast_ref::<Int64Array>().unwrap();
            let ups = batch.column(9).as_any().downcast_ref::<Int64Array>().unwrap();
            let sts = batch.column(10).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..batch.num_rows() {
                tasks.push(Task {
                    id: ids.value(i).to_string(), r#type: types.value(i).to_string(), from: froms.value(i).to_string(), 
                    to: tos.value(i).to_string(), cc: ccs.value(i).to_string(), bcc: bccs.value(i).to_string(), 
                    r#ref: refs.value(i).to_string(), data_json: datas.value(i).to_string(), 
                    created_at: crs.value(i), updated_at: ups.value(i), status: sts.value(i),
                });
            }
        }
        tasks.sort_by_key(|t| t.created_at);
        Ok(tasks)
    }

    pub async fn update_message_status(&self, task_id: &str, status: i32, text: Option<&str>) -> Result<()> {
        let table = self.conn.open_table("talks").execute().await?;

        // 🌟 [SCOPE PRESERVE]
        //  ── 무엇이 문제였나 ──
        //   기존에는 삭제 후 cc / bcc / ref / from / to 를 전부 None 으로 재삽입했습니다.
        //   ai_search_complex 는 최초 add_message 에 스코프를 담아 넣는데,
        //   첫 상태 전환(10 → 1)에서 그 값이 통째로 사라집니다.
        //   main.ts 의 loadMoreChat 은 ref / bcc / cc 로 채팅을 조회하므로,
        //   그 순간부터 작업 말풍선이 필터에서 탈락해 화면에서 사라졌습니다.
        //   created_at 도 현재 시각으로 덮여 '질문 → 작업' 정렬이 흔들렸습니다.
        //  ── 해결 ──
        //   삭제 '전' 에 기존 행의 봉투와 created_at 을 읽어 두고 그대로 복원합니다.
        let mut prev_type = "talk".to_string();
        let mut prev_from = String::new();
        let mut prev_to = String::new();
        let mut prev_cc = String::new();
        let mut prev_bcc = String::new();
        let mut prev_ref = String::new();
        let mut prev_created: i64 = 0;

        if let Ok(res) = table.query()
            .only_if(format!("task_id = '{}'", task_id))
            .limit(1)
            .execute()
            .await
        {
            if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                for b in batches {
                    if b.num_rows() == 0 { continue; }
                    // 컬럼 순서는 init_task_table 의 msg_schema 와 1:1 대응입니다.
                    // 0 id / 1 type / 2 role / 3 from / 4 to / 5 cc / 6 bcc / 7 ref
                    // 8 text / 9 data / 10 task_id / 11 status / 12 created_at / 13 updated_at
                    let types = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
                    let froms = b.column(3).as_any().downcast_ref::<StringArray>().unwrap();
                    let tos   = b.column(4).as_any().downcast_ref::<StringArray>().unwrap();
                    let ccs   = b.column(5).as_any().downcast_ref::<StringArray>().unwrap();
                    let bccs  = b.column(6).as_any().downcast_ref::<StringArray>().unwrap();
                    let refs  = b.column(7).as_any().downcast_ref::<StringArray>().unwrap();
                    let crs   = b.column(12).as_any().downcast_ref::<Int64Array>().unwrap();
                    prev_type    = types.value(0).to_string();
                    prev_from    = froms.value(0).to_string();
                    prev_to      = tos.value(0).to_string();
                    prev_cc      = ccs.value(0).to_string();
                    prev_bcc     = bccs.value(0).to_string();
                    prev_ref     = refs.value(0).to_string();
                    prev_created = crs.value(0);
                    break;
                }
            }
        }

        table.delete(&format!("task_id = '{}'", task_id)).await?;

        if let Some(t) = text {
            self.add_message_at(
                &uuid::Uuid::new_v4().to_string(), "system_task", t,
                Some(task_id), Some(status),
                Some(&prev_cc), Some(&prev_bcc), Some(&prev_ref),
                Some(&prev_from), Some(&prev_to), Some(&prev_type), None,
                Some(prev_created)
            ).await?;
        }
        Ok(())
    }

    pub async fn delete_message_by_task_id(&self, task_id: &str) -> Result<()> {
        let table = self.conn.open_table("talks").execute().await?;
        table.delete(&format!("task_id = '{}'", task_id)).await?;
        Ok(())
    }

    pub async fn update_task_status(&self, id: &str, status: i32) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        if status == 9 || status == 6 || status == 3 {
            table.delete(&format!("id = '{}'", id)).await?;
        } else {
            // [FIX] 실제로 DB의 status 값을 업데이트하여 중복 실행 방지
            table.update()
                .only_if(format!("id = '{}'", id))
                .column("status", status.to_string())
                .execute()
                .await?;
        }
        Ok(())
    }

    
    pub async fn cleanup_unfinished_tasks_on_startup(&self) -> Result<()> {
        let tasks_table = self.conn.open_table("tasks").execute().await?;
        let talks_table = self.conn.open_table("talks").execute().await?;

        println!("[Store] Initializing zombie task recovery process...");

        
        // 안전한 대기열(10) 상태로 돌려놓아 백그라운드 스케줄러가 [RESUME-LOGIC]을 타도록 유도합니다!
        // (기존 대기 중이던 10번 작업은 건드리지 않고 자연스럽게 이어서 실행되게 둡니다.)
        let _ = tasks_table.update()
            .only_if("status = 1")
            .column("status", "10") 
            .execute()
            .await;

        
        let _ = talks_table.update()
            .only_if("status = 1")
            .column("status", "10")
            .column("text", "'App restarted. Task is queued for auto-resumption...'")
            .execute()
            .await;

        println!("[Store] CRITICAL: Zombie recovery complete. (Interrupted tasks reverted to Pending for Auto-Resume)");
        Ok(())
    }

    // 🌟 [SINGLE ROUTER] 테이블 라우팅은 모듈 전역 함수 resolve_table_for 하나뿐입니다.
    //    lib.rs 의 upsert_items 도 같은 함수를 호출하므로,
    //    '저장 테이블과 조회 테이블이 어긋나는' 경로가 구조적으로 사라집니다.
    fn resolve_table(table_or_type: &str) -> &'static str {
        resolve_table_for(table_or_type)
    }

    pub async fn delete_item(&self, table_name: &str, id: &str) -> Result<()> {
        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;
        table.delete(&format!("id = '{}'", id)).await?;
        crate::utils::score_dynamics::presence_forget(id);

        // 🌟 [PHASE D] 연관 청크 동시 삭제
        let _ = self.delete_chunks_by_item(id).await;

        Ok(())
    }

    pub async fn delete_items(&self, table_name: &str, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() { return Ok(()); }

        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;
        let id_list = ids.iter().map(|id| format!("'{}'", id)).collect::<Vec<_>>().join(",");
        table.delete(&format!("id IN ({})", id_list)).await?;
        for id in ids.iter() {
            crate::utils::score_dynamics::presence_forget(id);
        }

        // 🌟 [PHASE D] 연관 청크 동시 삭제
        for id in &ids {
            let _ = self.delete_chunks_by_item(id).await;
        }

        Ok(())
    }
    
    // 🌟 [SCHEMA v4 / UNIFIED ENVELOPE]
    //  물리 컬럼을 '봉투(Envelope) 12개 + 검색 부품 3개' 로 확정합니다.
    //    봉투 : id, type, flag, from, to, cc, bcc, ref, mode, data, created_at, updated_at
    //    검색 : vector(ANN), text(FTS), masked_text(FTS)
    //  status / amount / is_masked / digest 는 전부 data JSON 으로 하강합니다.
    //  → LanceDB 는 '벡터 + FTS + 스코프 프리필터' 만 담당하고,
    //    도메인 조건(가격/수량/송장번호/상태...)은 Dexie 가 data.* 인덱스로 처리합니다.
    //  → 도메인 필드가 늘어나도 이 스키마는 영원히 그대로입니다. (Rust 재빌드 불필요)
    pub const SCHEMA_VERSION: &'static str = "v5:vision-vector";

    pub async fn init_all_tables(&self) -> Result<()> {
        // 🌟 [TABLE COLLAPSE] sales / tracking / event 물리 테이블을 폐기합니다.
        //    scheduler 가 어차피 items 에 이중 upsert 하고 있었고,
        //    lib.rs 의 target_table match 와 어긋나 'review 는 items 에 저장되는데
        //    event 에서 조회' 같은 구조적 0건 버그를 만들던 원인입니다.
        //    items 단일 테이블 + type 컬럼 파티셔닝으로 대체합니다.
        let tables = vec!["items", "users", "pages"];

        // 🌟 [LEGACY DROP] 이전 버전이 만든 '도메인 분할' 테이블만 정리합니다.
        //    talks 는 도메인 분할이 아니라 별도 스키마(role/task_id/status)를 가진
        //    메시지 테이블이므로 절대 포함시키면 안 됩니다.
        //    (init_task_table 이 별도로 관리합니다)
        for legacy in ["sales", "tracking", "event"] {
            let existing_legacy = self.conn.table_names().execute().await.unwrap_or_default();
            if existing_legacy.contains(&legacy.to_string()) {
                println!("[Store] Dropping legacy partition table: {}", legacy);
                let _ = self.conn.drop_table(legacy, &[]).await;
                let _ = std::fs::remove_dir_all(format!("{}/{}.lance", self.base_path, legacy));
            }
        }

        let item_field = Field::new("item", DataType::Float32, true);
        let vision_field = Field::new("item", DataType::Float32, true);
        let schema = Arc::new(Schema::new(vec![
            // ── 봉투(Envelope) : 3개 저장소(D1 / LanceDB / Dexie) 공통 계약 ──
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("flag", DataType::Utf8, true),
            Field::new("from", DataType::Utf8, true),
            Field::new("to", DataType::Utf8, true),
            Field::new("cc", DataType::Utf8, true),
            Field::new("bcc", DataType::Utf8, true),
            Field::new("ref", DataType::Utf8, true),
            Field::new("mode", DataType::Utf8, true),
            Field::new("data", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            // ── 검색 부품 : LanceDB 전용. 도메인 컬럼이 아님 ──
            Field::new("vector", DataType::FixedSizeList(Arc::new(item_field), 384), true),
            // 🌟 [비전 벡터] SigLIP2 풀링 벡터 (1152차원).
            //    이미지 추출 문서에만 실제 값이 들어가고,
            //    텍스트 전용 문서는 0 벡터입니다.
            //    컬럼 순서: 12=vector, 13=vision_vec, 14=text, 15=masked_text, 16=schema_v4
            Field::new("vision_vec", DataType::FixedSizeList(Arc::new(vision_field), 1152), true),
            Field::new("text", DataType::Utf8, false),
            Field::new("masked_text", DataType::Utf8, true),
            // ── 스키마 세대 각인 : 세대가 바뀌면 전량 재생성 ──
            Field::new("schema_v4", DataType::Utf8, true),
        ]));

        let uri = self.base_path.clone();
        let existing = self.conn.table_names().execute().await?;

        for name in tables {
            if existing.contains(&name.to_string()) {
                match self.conn.open_table(name).execute().await {
                    Ok(table) => {
                        let current_schema = table.schema().await.unwrap_or_else(|_| Arc::new(Schema::new(Vec::<Field>::new())));
                        let is_v4 = current_schema.field_with_name("schema_v4").is_ok();
                        let has_vision_vec = current_schema.field_with_name("vision_vec").is_ok();

                        let mut version_ok = true;
                        if is_v4 {
                            if let Ok(res) = table.query().limit(1).execute().await {
                                if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                                    for b in batches {
                                        if b.num_rows() == 0 { continue; }
                                        if let Some(col) = b.column(16).as_any().downcast_ref::<StringArray>() {
                                            if col.value(0) != Self::SCHEMA_VERSION {
                                                println!(
                                                    "[Store] Schema version drift on {}: stored='{}' expected='{}'",
                                                    name, col.value(0), Self::SCHEMA_VERSION
                                                );
                                                version_ok = false;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        // 🌟 도메인 컬럼 잔재가 있으면 구세대입니다.
                        let has_legacy_domain = current_schema.field_with_name("status").is_ok()
                            || current_schema.field_with_name("amount").is_ok()
                            || current_schema.field_with_name("is_masked").is_ok();

                        if !is_v4 || !has_vision_vec || has_legacy_domain || !version_ok {
                            println!("[Store] Schema generation mismatch for {} (v4: {}, legacy_domain: {}). Recreating...", name, is_v4, has_legacy_domain);
                            let _ = self.conn.drop_table(name, &[]).await;
                            let _ = std::fs::remove_dir_all(format!("{}/{}.lance", uri, name));
                        } else {
                            continue;
                        }
                    },
                    Err(_) => {
                        println!("[Store] Corrupted table {} detected. Force dropping.", name);
                        let _ = self.conn.drop_table(name, &[]).await;
                        let _ = std::fs::remove_dir_all(format!("{}/{}.lance", uri, name));
                    }
                }
            }

            if let Err(_) = self.conn.create_empty_table(name, schema.clone()).execute().await {
                let _ = std::fs::remove_dir_all(format!("{}/{}.lance", uri, name));
                let _ = self.conn.create_empty_table(name, schema.clone()).execute().await;
            }

            // 🌟 [FTS] items 만 마스터 검색 대상입니다.
            //    data 컬럼 FTS 는 그대로 유지합니다. 도메인 값이 전부 data 로 내려오므로
            //    오히려 이 인덱스의 가치가 올라갑니다. (송장번호/코드 substring 매칭)
            if let Ok(table) = self.conn.open_table(name).execute().await {
                if name == "items" {
                    let _ = table.create_index(&["text"], lancedb::index::Index::FTS(
                        lancedb::index::scalar::FtsIndexBuilder::default()
                            .with_position(true)
                            .base_tokenizer("ngram".to_string())
                            .ngram_min_length(2)
                            .ngram_max_length(3)
                    )).execute().await;

                    let _ = table.create_index(&["masked_text"], lancedb::index::Index::FTS(
                        lancedb::index::scalar::FtsIndexBuilder::default()
                            .with_position(true)
                            .base_tokenizer("ngram".to_string())
                            .ngram_min_length(2)
                            .ngram_max_length(3)
                    )).execute().await;

                    let _ = table.create_index(&["data"], lancedb::index::Index::FTS(
                        lancedb::index::scalar::FtsIndexBuilder::default()
                            .with_position(true)
                            .base_tokenizer("ngram".to_string())
                            .ngram_min_length(2)
                            .ngram_max_length(3)
                    )).execute().await;

                    println!("[Store] FTS Master Index verified/created exclusively for table: {}", name);
                }
            }
        }

        // 🌟 [PHASE D] item_chunks 테이블 초기화 (변경 없음 — 순수 벡터 테이블)
        self.init_chunks_table().await?;

        Ok(())
    }

    fn canonicalize_data(mut v: Value, seed_defaults: bool) -> Value {
        use crate::utils::canonical::{kind_of, iso_to_epoch_ms, CanonKind};
        const SEED_KEYS: &[(&str, CanonKind)] = &[
            // ── 식별자 ──
            ("id", CanonKind::Identifier),
            ("no", CanonKind::Identifier),
            ("code", CanonKind::Identifier),
            ("tracking_number", CanonKind::Identifier),
            ("stock_keeping_unit", CanonKind::Identifier),
            ("barcode", CanonKind::Identifier),
            ("digest", CanonKind::Identifier),
            // ── 수치 ──
            ("index", CanonKind::Numeric),
            ("goods", CanonKind::Numeric),
            ("order", CanonKind::Numeric),
            ("tracking", CanonKind::Numeric),
            ("status", CanonKind::Numeric),
            ("created_at", CanonKind::Numeric),
            ("updated_at", CanonKind::Numeric),
            // ── 불리언 ──
            ("embed", CanonKind::Boolean),
            // ── 배열 ──
            ("tags", CanonKind::Tags),
        ];

        let obj = match v.as_object_mut() {
            Some(o) => o,
            None => return json!({}),
        };
        let existing: Vec<String> = obj.keys().cloned().collect();
        for k in existing {
            let kind = kind_of(&k);
            if kind == CanonKind::Free { continue; }

            match kind {
                CanonKind::Identifier => {
                    let s = match obj.get(&k) {
                        Some(Value::Null) | None => continue,
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Number(n)) => n.to_string(),
                        Some(Value::Bool(b)) => if *b { "1".to_string() } else { "0".to_string() },
                        // 배열/객체는 식별자가 될 수 없으므로 건드리지 않습니다.
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                    };
                    obj.insert(k, json!(s));
                },
                CanonKind::Numeric => {
                    let n: f64 = match obj.get(&k) {
                        None | Some(Value::Null) => continue,
                        Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0),
                        Some(Value::Bool(b)) => if *b { 1.0 } else { 0.0 },
                        Some(Value::String(s)) => {
                            let t = s.trim();
                            if t.is_empty() || t == "null" || t == "N/A" { continue; }
                            if k == "status" {
                                crate::logic::parse_status(t) as f64
                            } else if let Some(ms) = iso_to_epoch_ms(t) {
                                ms as f64
                            } else {
                                let cleaned: String = t.chars()
                                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                                    .collect();
                                match cleaned.parse::<f64>() {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                }
                            }
                        },
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                    };
                    if n.fract() == 0.0 && n.abs() < 9e15 {
                        obj.insert(k, json!(n as i64));
                    } else {
                        obj.insert(k, json!(n));
                    }
                },
                CanonKind::Boolean => {
                    let b = match obj.get(&k) {
                        Some(Value::Bool(x)) => *x,
                        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) != 0,
                        Some(Value::String(s)) => {
                            let t = s.trim();
                            if t.is_empty() { continue; }
                            t == "1" || t.eq_ignore_ascii_case("true")
                        },
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                        None | Some(Value::Null) => continue,
                    };
                    obj.insert(k, json!(if b { 1 } else { 0 }));
                },
                CanonKind::Tags => {
                    // 🌟 [MISSING PARITY] main.ts 는 null 을 건너뜁니다.
                    //    `_ => Vec::new()` 는 '태그 없음' 을 '빈 배열' 로 확정하는데,
                    //    Dexie 의 멀티엔트리 인덱스('*data.tags')는 빈 배열도
                    //    키를 만들지 않으므로 조회 결과는 같지만
                    //    문서 본문이 두 저장소에서 달라져 digest 비교가 어긋납니다.
                    let tags: Vec<Value> = match obj.get(&k) {
                        Some(Value::Array(arr)) => arr.iter().map(|t| {
                            if let Some(o) = t.as_object() {
                                json!(o.get("tag").and_then(|x| x.as_str()).unwrap_or(""))
                            } else if let Some(s) = t.as_str() {
                                json!(s)
                            } else {
                                json!(t.to_string().trim_matches('"'))
                            }
                        }).filter(|t| t.as_str().map_or(false, |s| !s.is_empty())).collect(),
                        Some(Value::String(s)) if !s.is_empty() => vec![json!(s.clone())],
                        None | Some(Value::Null) => continue,
                        _ => Vec::new(),
                    };
                    obj.insert(k, json!(tags));
                },
                CanonKind::Free => {},
            }
        }

        // ── ② 조회 축 기본값 시딩 ──
        if seed_defaults {
            for (k, kind) in SEED_KEYS.iter() {
                if obj.get(*k).is_some() { continue; }
                let d = match kind {
                    CanonKind::Identifier => json!(""),
                    CanonKind::Numeric => json!(0),
                    CanonKind::Boolean => json!(0),
                    CanonKind::Tags => json!([]),
                    CanonKind::Free => continue,
                };
                obj.insert(k.to_string(), d);
            }
        }

        v
    }

    /// 🌟 [VECTOR CARRY] 기존 행의 벡터 2개를 그대로 읽어옵니다.
    ///
    ///  ── 왜 batch_to_docs 를 쓰지 않는가 ──
    ///   batch_to_docs 는 vector / vision_vec 을 항상 Vec::new() 로 비웁니다.
    ///   그 함수가 get_all_items 의 상시 경로이기 때문이며,
    ///   거기서 벡터를 채우면 목록 조회 한 번에 행당 1536 float 이 왕복합니다.
    ///   벡터 승계는 단건 upsert 에서만 필요하므로 전용 리더를 둡니다.
    async fn read_existing_vectors(&self, target: &str, id: &str)
        -> (Option<Vec<f32>>, Option<Vec<f32>>)
    {
        let table = match self.conn.open_table(target).execute().await {
            Ok(t) => t,
            Err(_) => return (None, None),
        };
        let batches = match table.query().only_if(format!("id = '{}'", id)).limit(1).execute().await {
            Ok(r) => r.try_collect::<Vec<_>>().await.unwrap_or_default(),
            Err(_) => return (None, None),
        };
        let read_fsl = |b: &RecordBatch, col: usize, dim: usize| -> Option<Vec<f32>> {
            let arr = b.column(col).as_any().downcast_ref::<FixedSizeListArray>()?;
            if arr.is_null(0) { return None; }
            let vals = arr.value(0);
            let f = vals.as_any().downcast_ref::<Float32Array>()?;
            if f.len() != dim { return None; }
            let v: Vec<f32> = (0..dim).map(|i| f.value(i)).collect();
            if v.iter().all(|&x| x == 0.0) { return None; }
            Some(v)
        };
        for b in batches {
            if b.num_rows() == 0 { continue; }
            // 컬럼 순서: 12 = vector(384), 13 = vision_vec(1152)
            return (read_fsl(&b, 12, 384), read_fsl(&b, 13, 1152));
        }
        (None, None)
    }

    const STORE_STAMPED_KEYS: [&'static str; 15] = [
        "id", "type", "mode", "created_at", "updated_at",
        "from", "to", "cc", "bcc", "ref",
        "digest", "has_vision", "embed", "text", "masked_text",
    ];

    fn collect_filled_fields(v: &Value, depth: usize, out: &mut Vec<String>) {
        let obj = match v.as_object() { Some(o) => o, None => return };
        for (k, val) in obj.iter() {
            if Self::STORE_STAMPED_KEYS.iter().any(|s| *s == k.as_str()) { continue; }
            let filled = match val {
                Value::Null => false,
                Value::String(s) => !s.trim().is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
                _ => true,
            };
            if filled && !out.iter().any(|x| x == k) { out.push(k.clone()); }
            if depth == 0 { continue; }
            match val {
                Value::Object(_) => Self::collect_filled_fields(val, depth - 1, out),
                Value::Array(a) => {
                    for e in a.iter() {
                        if e.is_object() { Self::collect_filled_fields(e, depth - 1, out); }
                    }
                }
                _ => {}
            }
        }
    }

    pub async fn upsert_item(
        &self, table_name: &str, id: &str, type_: &str, mut data_val: Value, vector: Option<Vec<f32>>,
        vision_vec: Option<Vec<f32>>,
        from: Option<&str>, to: Option<&str>, cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>, digest: Option<&str>
    ) -> Result<()> {
        let target = Self::resolve_table(if table_name.is_empty() { "items" } else { table_name });
        let table = self.conn.open_table(target).execute().await?;

        // 🌟 [ID FIRST] id 해석을 가장 먼저 확정합니다.
        //    기존에는 read_existing_vectors 안에서 id 해석 로직이 한 번 더 복제되어 있었고,
        //    그 복제본이 final_id 계산과 어긋나면 벡터 승계가 조용히 실패했습니다.
        let final_id = if id.is_empty() {
            data_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string()
        } else { id.to_string() };
        if final_id.is_empty() { return Ok(()); }

        // 🌟 [VECTOR PRESERVE]
        //
        //  ── 무엇이 문제였나 ──
        //   syncData / pullTradingData 는 벡터를 들고 오지 않으므로 vector = None 입니다.
        //   그런데 아래에서 None 은 vec![0.0; 384] 로 덮이고,
        //   동시에 old_embedded 분기가 data.embed = 1 을 강제 주입합니다.
        //   결과는 '0 벡터 + embed=1' 이며,
        //   runLocalEmbeddingSync 는 embed != 1 만 후보로 삼으므로
        //   그 문서의 벡터 검색이 영구히 죽습니다.
        //   (migrate_team_identity 는 이 함정을 알고 embed 를 제거하는데,
        //    3초마다 도는 상시 경로가 반대로 하고 있었습니다)
        //
        //  ── 왜 '승계' 인가 ──
        //   서버 동기화는 봉투(cc/bcc/ref)와 도메인 값만 갱신합니다.
        //   text 가 그대로면 벡터도 그대로여야 하고,
        //   text 가 바뀌었으면 reindex 가 다시 만들어야 합니다.
        //   어느 쪽이든 '0 으로 지우기' 가 정답인 경우는 없습니다.
        let (stored_vec, stored_vision) = self.read_existing_vectors(target, &final_id).await;
        let vector_gain = stored_vec.is_none()
            && vector.as_ref().map_or(false, |v| v.len() == 384 && v.iter().any(|&x| x != 0.0));
        let vision_gain = stored_vision.is_none()
            && vision_vec.as_ref().map_or(false, |v| v.len() == 1152 && v.iter().any(|&x| x != 0.0));
        let vector = vector.or(stored_vec);
        let vision_vec = vision_vec.or(stored_vision);
        // 🌟 [SKIP GUARD v2] digest 는 이제 물리 컬럼이 아니라 data.digest 입니다.
        //    기존 문서의 digest 를 읽으려면 json_data 를 파싱해야 합니다.
        //
        //  ── v1 의 결함 ①: 봉투 변경을 무시했습니다 ──
        //   digest 는 text 만으로 만들어집니다. 그래서 서버가 cc / bcc / ref / to 만 바꿔
        //   내려보내면 old_digest == new_digest 가 되어 return Ok(()) 로 빠지고,
        //   봉투가 영구히 옛 값으로 남습니다.
        //   '봉투는 LanceDB 담당' 이라는 v4 계약과 정면으로 충돌합니다.
        //   (team 마이그레이션 / 사이트 재스코프 이후 목록·검색이 어긋나는 직접 원인)
        //
        //  ── v1 의 결함 ②: 압축 페이로드에서 updated_at 을 0 으로 읽었습니다 ──
        //   updated_at 이 gzip/base64 블롭 안에 있으면 여기서는 0 이 됩니다.
        //   그러면 doc.updated_at_ts >= 0 이 항상 참이 되어 정상 갱신이 스킵될 수 있습니다.
        //   블롭이 미해제 상태이면 아예 스킵 판정을 하지 않습니다.
        let new_digest = digest.unwrap_or("").to_string();
        let new_updated_at = data_val.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0);
        let has_pending_blob = data_val
            .get("data")
            .and_then(|v| v.as_str())
            .map_or(false, |s| s.len() > 50);
        if let Some(doc) = self.get_item_by_id(target, &final_id).await? {
            let envelope_same = doc.r#type == type_
                && doc.from == resolve_envelope_field(from, &data_val, "from")
                && doc.to == resolve_envelope_field(to, &data_val, "to")
                && doc.cc == resolve_envelope_field(cc, &data_val, "cc")
                && doc.bcc == resolve_envelope_field(bcc, &data_val, "bcc")
                && doc.r#ref == resolve_envelope_field(r#ref, &data_val, "ref");
            if !has_pending_blob
                && envelope_same
                && doc.updated_at_ts >= new_updated_at
                && !new_digest.is_empty()
            {
                let old_digest = serde_json::from_str::<Value>(&doc.json_data)
                    .ok()
                    .and_then(|v| v.get("digest").and_then(|d| d.as_str()).map(|s| s.to_string()))
                    .unwrap_or_default();
                if old_digest == new_digest {
                    if vector_gain || vision_gain {
                        println!(
                            "[STORE] 🧲 [VECTOR GAIN] id='{}' digest 는 동일하지만 저장본에 없던 벡터를 이번 호출이 들고 왔습니다. (text={} / vision={}) 스킵을 취소하고 기록합니다.",
                            final_id, vector_gain, vision_gain
                        );
                    } else {
                        return Ok(());
                    }
                }
            }
            if !envelope_same {
                println!(
                    "[STORE] ✉️ [ENVELOPE CHANGED] id='{}' 의 봉투가 변경되어 digest 가 같아도 재기록합니다. (from/to/cc/bcc/ref/type)",
                    final_id
                );
            }
            let old_data = serde_json::from_str::<Value>(&doc.json_data).ok();
            let old_embedded = old_data.as_ref()
                .and_then(|v| v.get("embed"))
                .map(|v| v.as_i64().unwrap_or(0) == 1 || v.as_bool().unwrap_or(false))
                .unwrap_or(false);
            if old_embedded {
                if let Some(obj) = data_val.as_object_mut() {
                    if obj.get("embed").map_or(true, |v| v.as_i64().unwrap_or(0) != 1) {
                        obj.insert("embed".to_string(), json!(1));
                    }
                }
            }
        }

        println!("[DEBUG] store.upsert_item (v4) - Table: {}, ID: {}, Type: {}", target, final_id, type_);

        let has_real_vision = vision_vec
            .as_ref()
            .map(|v| v.len() == 1152 && v.iter().any(|&x| x != 0.0))
            .unwrap_or(false);
        let has_real_vector = vector
            .as_ref()
            .map(|v| v.len() == 384 && v.iter().any(|&x| x != 0.0))
            .unwrap_or(false);
        let presence_id = final_id.clone();

        let _ = table.delete(&format!("id = '{}'", final_id)).await;
        let mut final_data = data_val.clone();
        if let Some(blob_base64) = final_data.get("data").and_then(|v| v.as_str()) {
            if blob_base64.len() > 50 {
                use base64::prelude::BASE64_STANDARD;
                use base64::Engine;
                if let Ok(decoded) = BASE64_STANDARD.decode(blob_base64) {
                    if let Ok(decompressed) = crate::utils::compression::decompress_to_value(&decoded) {
                        if let Some(base_obj) = final_data.as_object_mut() {
                            if let Some(inner_obj) = decompressed.as_object() {
                                for (k, v) in inner_obj {
                                    // 봉투 필드는 덮어쓰지 않습니다.
                                    if !base_obj.contains_key(k) || k == "action" || k == "summary" || k == "relate" || k == "text" || k == "masked_text" || k == "embed" || k == "href" || k == "link" || k == "origin" {
                                        base_obj.insert(k.clone(), v.clone());
                                    }
                                }
                            }
                            base_obj.remove("data");
                        }
                    }
                }
            }
        }

        let src = &final_data;
        let mode_str = match src.get("mode").and_then(|v| v.as_str()) {
            Some(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => {
                let inferred = infer_mode(type_);
                if inferred != "commerce" {
                    println!(
                        "[STORE] 🧭 [MODE INFER] id='{}' type='{}' 에 mode 가 없어 '{}' 로 확정합니다.",
                        final_id, type_, inferred
                    );
                }
                inferred.to_string()
            }
        };
        let has_updated_key = src.get("updated_at").is_some();
        let new_updated_at = src.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(new_updated_at);
        let wall_now = chrono::Utc::now().timestamp_millis();
        let is_domain_item = matches!(target, "items");
        let updated_ts = if has_updated_key {
            new_updated_at
        } else if is_domain_item {
            0
        } else {
            wall_now
        };
        let created_at = src.get("created_at")
            .and_then(|v| v.as_i64())
            .filter(|v| *v > 0)
            .unwrap_or(wall_now);
        let env_from = resolve_envelope_field(from, &final_data, "from");
        let env_to   = resolve_envelope_field(to,   &final_data, "to");
        let env_cc   = resolve_envelope_field(cc,   &final_data, "cc");
        let env_bcc  = resolve_envelope_field(bcc,  &final_data, "bcc");
        let env_ref  = resolve_envelope_field(r#ref, &final_data, "ref");

        if let Some(obj) = final_data.as_object_mut() {
            // 별칭 보정 (기존 동작 유지)
            if let Some(tn) = obj.get("tracking_number").cloned() {
                if obj.get("tracking").is_none() { obj.insert("tracking".to_string(), tn); }
            }
            if let Some(p) = obj.get("price").cloned() {
                if obj.get("sale_price").is_none() { obj.insert("sale_price".to_string(), p); }
            }
            obj.insert("id".to_string(), json!(final_id.clone()));
            obj.insert("type".to_string(), json!(type_));
            obj.insert("mode".to_string(), json!(mode_str.clone()));
            obj.insert("created_at".to_string(), json!(created_at));
            obj.insert("updated_at".to_string(), json!(updated_ts));
            obj.insert("from".to_string(), json!(env_from.clone()));
            obj.insert("to".to_string(), json!(env_to.clone()));
            obj.insert("cc".to_string(), json!(env_cc.clone()));
            obj.insert("bcc".to_string(), json!(env_bcc.clone()));
            obj.insert("ref".to_string(), json!(env_ref.clone()));
            if !new_digest.is_empty() {
                obj.insert("digest".to_string(), json!(new_digest.clone()));
            }
            if has_real_vision {
                obj.insert("has_vision".to_string(), json!(1));
            }
            if has_real_vector {
                obj.insert("embed".to_string(), json!(1));
            } else if obj.get("embed").map_or(false, |v| v.as_i64().unwrap_or(0) == 1 || v.as_bool().unwrap_or(false)) {
                println!(
                    "[STORE] 🧹 [EMBED CLAIM DROP] id='{}' 의 embed=1 표식을 내립니다. 이번에 기록되는 텍스트 벡터가 0 벡터입니다. 다음 임베딩 회차의 후보로 되돌립니다.",
                    final_id
                );
                obj.insert("embed".to_string(), json!(0));
            }
        }
        let seed_defaults = needs_domain_seed(target, type_);
        let final_data = Self::canonicalize_data(final_data, seed_defaults);

        let json_str = final_data.to_string();
        let text_content = final_data.get("text").and_then(|s| s.as_str()).unwrap_or("").to_string();
        let masked_text_content = final_data.get("masked_text").and_then(|s| s.as_str()).unwrap_or(&text_content).to_string();
        let flag_str = final_data.get("flag").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let schema = table.schema().await?;

        let safe_vector = match vector {
            Some(v) if v.len() == 384 => v,
            _ => vec![0.0; 384],
        };
        let values_builder = Float32Array::from(safe_vector);
        let list_field = Field::new("item", DataType::Float32, true);
        let list_array = FixedSizeListArray::try_new(Arc::new(list_field), 384, Arc::new(values_builder), None)?;
        let safe_vision_vec = match vision_vec {
            Some(v) if v.len() == 1152 => v,
            _ => vec![0.0; 1152],
        };
        let vision_values_builder = Float32Array::from(safe_vision_vec);
        let vision_list_field = Field::new("item", DataType::Float32, true);
        let vision_list_array = FixedSizeListArray::try_new(Arc::new(vision_list_field), 1152, Arc::new(vision_values_builder), None)?;
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec![final_id])),
            Arc::new(StringArray::from(vec![type_])),
            Arc::new(StringArray::from(vec![flag_str])),
            Arc::new(StringArray::from(vec![env_from])),
            Arc::new(StringArray::from(vec![env_to])),
            Arc::new(StringArray::from(vec![env_cc])),
            Arc::new(StringArray::from(vec![env_bcc])),
            Arc::new(StringArray::from(vec![env_ref])),
            Arc::new(StringArray::from(vec![mode_str])),
            Arc::new(StringArray::from(vec![json_str])),
            Arc::new(Int64Array::from(vec![created_at])),
            Arc::new(Int64Array::from(vec![updated_ts])),
            Arc::new(list_array),
            Arc::new(vision_list_array),
            Arc::new(StringArray::from(vec![text_content])),
            Arc::new(StringArray::from(vec![masked_text_content])),
            Arc::new(StringArray::from(vec![Self::SCHEMA_VERSION])),
        ])?;
        table.add(vec![batch]).execute().await?;
        let is_relay_draft = updated_ts == 0 && new_digest.is_empty();
        if is_relay_draft {
            println!(
                "[STORE] 🧾 [PRESENCE SKIP] id='{}' type='{}' 은 릴레이가 만든 빈 초안입니다. 저장 사전에 넣지 않습니다. 초안은 '이 서식이 그 축을 보통 갖는가' 라는 관측에 기여할 수 없는데, 문서 1건을 추출할 때마다 10건 이상 생기므로 그대로 두면 껍데기가 다수를 이뤄 실제 문서의 축 보유율을 0 에 가깝게 끌어내립니다.",
                presence_id, type_
            );
        } else {
            let mut present_fields: Vec<String> = Vec::new();
            Self::collect_filled_fields(&final_data, 1, &mut present_fields);
            crate::utils::score_dynamics::presence_record(type_, &presence_id, &present_fields);
        }
        Ok(())
    }

    pub async fn initialize_user_profiles(&self, user_address: &str, user_email: &str, flag: &str) -> Result<()> {
        let team_id = crate::utils::hash::hash_id(user_address);
        let user_name = user_email.split('@').next().unwrap_or("user");
        let mut base = json!({"pages": {}, "goods": {"draft": 0, "count": 0}, "order": {"draft": 0, "count": 0}, "event": {"draft": 0, "count": 0}, "coupon": {"draft": 0, "count": 0}, "tracking": {"draft": 0, "count": 0}, "search": {"draft": 0, "count": 0}, "review": {"draft": 0, "count": 0}, "member": {"draft": 0, "count": 0}});
        let properties = vec!["price", "quantity", "width", "height", "length", "weight", "shipping_fee", "shipping_duration", "sale_price", "supply_price", "low_stock_threshold", "discount", "min_order_amount", "max_discount_amount", "usage_limit", "usage_per", "started_at", "expired_at"];
        if let Some(base_obj) = base.as_object_mut() {
            for (table_name, table_val) in base_obj.iter_mut() {
                if table_name != "pages" {
                    if let Some(table_obj) = table_val.as_object_mut() {
                        for prop in &properties { table_obj.insert(prop.to_string(), json!({"max": 0, "min": 0})); }
                    }
                }
            }
        }

        let team_data = json!({
            "flag": flag,
            "mode": "commerce",
            "name": format!("{}'s team", user_name),
            "title": "",
            "region": null,
            "page_count": 0,
            "favicon": null,
            "text": format!("{}'s team", user_name),
            "base": base
        });
        let user_data = json!({
            "flag": flag,
            "mode": "commerce",
            "name": user_name,
            "title": "",
            "region": null,
            "page_count": 0,
            "favicon": null,
            "text": user_name
        });

        self.upsert_item("users", &team_id, "team", team_data, None, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        self.upsert_item("users", user_address, "user", user_data, None, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        Ok(())
    }

    pub async fn migrate_team_identity(
        &self,
        old_to: &str,
        new_to: &str,
        new_from: &str,
    ) -> Result<usize> {
        if old_to == new_to { return Ok(0); }
        let mut migrated = 0usize;
        let tables = ["items", "users", "pages"];
        for table in tables {
            let filter = format!("`to` = '{}'", old_to);
            let docs = self.get_all_items(table, 5000, 0, Some(filter)).await.unwrap_or_default();
            for doc in docs {
                let mut data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
                if let Some(obj) = data.as_object_mut() {
                    obj.insert("to".to_string(), json!(new_to));
                    obj.insert("from".to_string(), json!(new_from));
                }
                if let Some(obj) = data.as_object_mut() {
                    obj.remove("embed");
                }
                let _ = self.delete_chunks_by_item(&doc.id).await;
                let _ = self.upsert_item(
                    table,
                    &doc.id,
                    &doc.r#type,
                    data,
                    None, // 벡터는 reindex 가 재생성합니다 (embed 마커 제거로 후보 복귀)
                    None,
                    Some(new_from),
                    Some(new_to),
                    Some(&doc.cc),
                    Some(&doc.bcc),
                    Some(&doc.r#ref),
                    None,
                ).await;
                migrated += 1;
            }
        }
        if migrated > 0 {
            println!("[MIGRATE] team identity '{}' → '{}' : {} docs migrated", old_to, new_to, migrated);
        }
        Ok(migrated)
    }

    pub async fn migrate_mode_by_type(&self) -> Result<usize> {
        use crate::utils::bias_schema::TRADE_DOC_TYPES;

        const LOWER_COLLIDE: [&str; 4] = ["co", "id", "ca", "pc"];
        let type_in = TRADE_DOC_TYPES
            .iter()
            .flat_map(|t| {
                let up = t.to_uppercase();
                let low = t.to_lowercase();
                if LOWER_COLLIDE.iter().any(|c| *c == low) {
                    vec![up]
                } else {
                    vec![up, low]
                }
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|t| format!("'{}'", t.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        // 🌟 mode 조건을 SQL 에서 빼고 Rust 에서 판정합니다.
        //    type 대소문자 교정 대상(mode 는 맞지만 type 이 소문자인 행)까지
        //    같은 순회로 처리하기 위함입니다.
        let filter = format!("type IN ({})", type_in);
        let docs = match self.get_all_items("items", 5000, 0, Some(filter)).await {
            Ok(d) => d,
            Err(e) => {
                println!("[MIGRATE] ⚠️ mode 마이그레이션 조회 실패(무시하고 진행): {}", e);
                return Ok(0);
            }
        };
        if docs.is_empty() {
            return Ok(0);
        }
        println!(
            "[MIGRATE] 🚢 mode 오태깅 무역 문서 {}건 발견. 'shipping' 으로 교정하고 재인덱싱 대상으로 되돌립니다.",
            docs.len()
        );
                let mut migrated = 0usize;
        for doc in docs {
            // 🌟 표준형(대문자)과 다르면 type 도 함께 교정합니다.
            let canon_type = crate::utils::bias_schema::canonical_trade_doc_code(&doc.r#type)
                .map(|c| c.to_string())
                .unwrap_or_else(|| doc.r#type.clone());
            let mode_wrong = doc.mode != "shipping";
            let type_wrong = doc.r#type != canon_type;
            if !mode_wrong && !type_wrong { continue; }
            let mut data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
            if let Some(obj) = data.as_object_mut() {
                obj.insert("mode".to_string(), json!("shipping"));
                obj.insert("type".to_string(), json!(canon_type.clone()));
                // 벡터·청크를 다시 만들어야 하므로 완료 마커를 지웁니다.
                obj.remove("embed");
            }
            if type_wrong {
                println!(
                    "[MIGRATE] 🚢 type 표준형 교정: id='{}' '{}' → '{}'",
                    doc.id, doc.r#type, canon_type
                );
            }
            let _ = self.delete_chunks_by_item(&doc.id).await;
            let _ = self.upsert_item(
                "items",
                &doc.id,
                &canon_type,
                data,
                None,
                None,
                Some(&doc.from),
                Some(&doc.to),
                Some(&doc.cc),
                Some(&doc.bcc),
                Some(&doc.r#ref),
                None,
            ).await;
            migrated += 1;
        }
        if migrated == 0 {
            return Ok(0);
        }
        println!("[MIGRATE] ✅ mode 교정 완료: {}건. 다음 reindex 폴링에서 shipping 트랙으로 재인덱싱됩니다.", migrated);
        Ok(migrated)
    }

    fn batch_to_docs(batch: &RecordBatch) -> Vec<TradeDocument> {
        let ids         = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let types       = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let flags       = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let froms       = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let tos         = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let ccs         = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        let bccs        = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
        let refs        = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
        let modes       = batch.column(8).as_any().downcast_ref::<StringArray>().unwrap();
        let jsons       = batch.column(9).as_any().downcast_ref::<StringArray>().unwrap();
        let createds    = batch.column(10).as_any().downcast_ref::<Int64Array>().unwrap();
        let updateds    = batch.column(11).as_any().downcast_ref::<Int64Array>().unwrap();
        let texts       = batch.column(14).as_any().downcast_ref::<StringArray>().unwrap();
        let masked      = batch.column(15).as_any().downcast_ref::<StringArray>().unwrap();

        let mut out = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            out.push(TradeDocument {
                id: ids.value(i).to_string(),
                r#type: types.value(i).to_string(),
                flag: flags.value(i).to_string(),
                from: froms.value(i).to_string(),
                to: tos.value(i).to_string(),
                cc: ccs.value(i).to_string(),
                bcc: bccs.value(i).to_string(),
                r#ref: refs.value(i).to_string(),
                mode: modes.value(i).to_string(),
                json_data: jsons.value(i).to_string(),
                created_at_ts: createds.value(i),
                updated_at_ts: updateds.value(i),
                text: texts.value(i).to_string(),
                masked_text: masked.value(i).to_string(),
                vector: Vec::new(),
                // 🌟 비전 벡터는 조회 시점에는 비워 둡니다.
                //    검색 트랙에서 ANN 질의에만 쓰고, 응답에는 싣지 않습니다.
                vision_vec: Vec::new(),
            });
        }
        out
    }

    pub async fn get_all_items(&self, table_name: &str, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<TradeDocument>> {
        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;
        let mut q = table.query();
        if let Some(f) = filter {
            if !f.trim().is_empty() { q = q.only_if(f); }
        }
        const SCAN_CEILING: usize = 20_000;
        let scan_cap = std::cmp::max(offset + limit, SCAN_CEILING);
        let results = q.limit(scan_cap).execute().await?.try_collect::<Vec<_>>().await?;
        let mut docs = Vec::new();
        for batch in results {
            docs.extend(Self::batch_to_docs(&batch));
        }
        docs.sort_by_key(|d| std::cmp::Reverse(d.created_at_ts));

        let start = offset.min(docs.len());
        let end = (start + limit).min(docs.len());
        Ok(docs[start..end].to_vec())
    }

    pub async fn get_item_by_id(&self, table_name: &str, id: &str) -> Result<Option<TradeDocument>> {
        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;
        let results = table.query().only_if(format!("id = '{}'", id)).limit(1).execute().await?.try_collect::<Vec<_>>().await?;
        if results.is_empty() || results[0].num_rows() == 0 { return Ok(None); }

        let docs = Self::batch_to_docs(&results[0]);
        Ok(docs.into_iter().next())
    }

    pub async fn search_items(&self, table_name: &str, query_text: &str, query_vec: Vec<f32>, vision_query_vec: Option<Vec<f32>>, limit: usize, offset: usize, filter: Option<String>, use_fts: bool) -> Result<Vec<(String, String, f32)>> {
         let target = Self::resolve_table(if table_name.is_empty() { "items" } else { table_name });
         let table = self.conn.open_table(target).execute().await?;

         let mut combined: std::collections::HashMap<String, (String, f32)> = std::collections::HashMap::new();
         let fetch_limit = std::cmp::max(200, (limit + offset) * 4);
         let scope: Option<String> = filter.as_ref().and_then(|f| {
             let t = f.trim();
             if t.is_empty() { None } else { Some(t.to_string()) }
         });
         if !query_text.trim().is_empty() {
             let mut q = table.query();
             let has_fts_index = target == "items"; // FTS 인덱스는 items 에만 존재

             if use_fts && has_fts_index {
                 let fts_query_str = query_text
                     .split_whitespace()
                     .map(|w| format!("\"{}\"", w.replace("\"", "\\\"")))
                     .collect::<Vec<_>>()
                     .join(" ");
                 q = q.full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(fts_query_str));
                 if let Some(ref f) = scope { q = q.only_if(f.clone()); }
             } else {
                 let sql_clean = query_text.replace("'", "''");
                 let mut ilike_conditions = Vec::new();
                 for w in sql_clean.split_whitespace() {
                     ilike_conditions.push(format!("(masked_text ILIKE '%{}%' OR text ILIKE '%{}%' OR data ILIKE '%{}%')", w, w, w));
                 }
                 let text_filter = ilike_conditions.join(" AND ");
                 let final_filter = match (&scope, text_filter.is_empty()) {
                     (Some(f), false) => format!("({}) AND ({})", f, text_filter),
                     (Some(f), true)  => f.clone(),
                     (None, false)    => text_filter,
                     (None, true)     => String::new(),
                 };
                 if !final_filter.is_empty() { q = q.only_if(final_filter); }
             }

             if let Ok(res) = q.limit(fetch_limit).execute().await {
                if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                    for b in batches {
                        let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                        // 🌟 data 컬럼 인덱스가 13 → 9 로 이동했습니다.
                        let txs = b.column(9).as_any().downcast_ref::<StringArray>().unwrap();
                        for i in 0..b.num_rows() {
                            let id = ids.value(i).to_string();
                            if let Some((_, s)) = combined.get_mut(&id) { *s += 2.0; }
                            else { combined.insert(id, (txs.value(i).to_string(), 2.0)); }
                        }
                    }
                }
             }
         }

         let is_empty_vec = query_vec.iter().all(|&x| x == 0.0);

         if !is_empty_vec {
             let mut vq = table.query();
             if let Some(ref f) = scope { vq = vq.only_if(f.clone()); }

             if let Ok(vq_with_vector) = vq.limit(fetch_limit).nearest_to(query_vec) {
                 let vq_with_vector = vq_with_vector.column("vector");
                 if let Ok(vres) = vq_with_vector.execute().await {
                     if let Ok(batches) = vres.try_collect::<Vec<_>>().await {
                         let mut rank = 0;
                         for b in batches {
                             let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                             let txs = b.column(9).as_any().downcast_ref::<StringArray>().unwrap();
                             for i in 0..b.num_rows() {
                                 let id = ids.value(i).to_string();
                                 let vec_score = 1.0 - (rank as f32 * 0.001);
                                 if let Some((_, s)) = combined.get_mut(&id) { *s += vec_score; }
                                 else { combined.insert(id, (txs.value(i).to_string(), vec_score)); }
                                 rank += 1;
                             }
                         }
                     }
                 } else {
                     println!("[STORE] ⚠️ Text vector track failed to execute (column='vector').");
                 }
             }
         }
        if let Some(ref vvec) = vision_query_vec {
            let is_empty_vvec = vvec.iter().all(|&x| x == 0.0);
            let dim_ok = vvec.len() == 1152;
            if !dim_ok {
                println!(
                    "[STORE] ⚠️ Vision query vector dim {} != 1152. Vision track skipped.",
                    vvec.len()
                );
            }
            if !is_empty_vvec && dim_ok {
                let vision_scope = match &scope {
                    Some(f) => format!("({}) AND data LIKE '%\"has_vision\":1%'", f),
                    None => "data LIKE '%\"has_vision\":1%'".to_string(),
                };
                let mut vvq = table.query();
                vvq = vvq.only_if(vision_scope.clone());
                if let Ok(vvq_with_vector) = vvq.limit(fetch_limit).nearest_to(vvec.clone()) {
                    let vvq_with_vector = vvq_with_vector.column("vision_vec");
                    if let Ok(vvres) = vvq_with_vector.execute().await {
                        if let Ok(vbatches) = vvres.try_collect::<Vec<_>>().await {
                            let mut vrank = 0;
                            for b in vbatches {
                                let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                                let txs = b.column(9).as_any().downcast_ref::<StringArray>().unwrap();
                                for i in 0..b.num_rows() {
                                    let id = ids.value(i).to_string();
                                    let v_score = 1.0 - (vrank as f32 * 0.001);
                                    if let Some((_, s)) = combined.get_mut(&id) { *s += v_score; }
                                    else { combined.insert(id, (txs.value(i).to_string(), v_score)); }
                                    vrank += 1;
                                }
                            }
                            if vrank > 0 {
                                println!("[STORE] 👁️ Vision track hit {} row(s) on column 'vision_vec' (scope: {}).", vrank, vision_scope);
                            } else {
                                println!("[STORE] ⚪ Vision track matched 0 rows. 이미지 추출 문서가 없거나, 마커 도입 이전에 저장되어 재추출이 필요합니다.");
                            }
                        }
                    } else {
                        println!("[STORE] ⚠️ Vision vector track failed to execute (column='vision_vec').");
                    }
                }
            }
        }
        if combined.is_empty() {
             let mut q = table.query();
             if let Some(ref f) = scope { q = q.only_if(f.clone()); }
             if let Ok(res) = q.limit(fetch_limit).execute().await {
                 if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                     for b in batches {
                         let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                         let txs = b.column(9).as_any().downcast_ref::<StringArray>().unwrap();
                         let createds = b.column(10).as_any().downcast_ref::<Int64Array>().unwrap();
                         for i in 0..b.num_rows() {
                             let recency = (createds.value(i) as f64 / 1.0e13) as f32;
                             combined.insert(ids.value(i).to_string(), (txs.value(i).to_string(), recency));
                         }
                     }
                 }
             }
         }

         let mut final_list: Vec<_> = combined.into_iter().map(|(id, (txt, s))| (id, txt, s)).collect();
         final_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

         let start = offset.min(final_list.len());
         let end = (start + limit).min(final_list.len());
         let result_slice = final_list[start..end].to_vec();

         if !is_empty_vec {
             let json_log = serde_json::json!({
                 "target_table": target,
                 "query_text": query_text,
                 "scope_filter": scope,
                 "use_fts": use_fts,
                 "fetch_limit": fetch_limit,
                 "total_found": final_list.len(),
                 "returned": result_slice.len(),
                 "results": result_slice.iter().map(|(id, text, score)| {
                     let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!(text));
                     serde_json::json!({ "id": id, "text": parsed_text, "score": score })
                 }).collect::<Vec<_>>()
             });
             println!("\n=======================================");
             println!("[STORE] 🔎 2-Track Recall Search (FTS + Vector) — precision filtering delegated to Dexie:");
             println!("{}", serde_json::to_string_pretty(&json_log).unwrap_or_default());
             println!("=======================================\n");
         }

         Ok(result_slice)
    }
    pub async fn find_item_by_property(&self, table_name: &str, property: &str, value: &Value) -> Result<Option<(String, Value)>> {
        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;

        let target_str = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => if *b { "1".to_string() } else { "0".to_string() },
            _ => value.to_string().trim_matches('"').to_string(),
        };
        if target_str.is_empty() { return Ok(None); }
        use crate::utils::canonical::{kind_of, CanonKind};
        let escaped_prop = property.replace('\'', "''");
        let escaped_val = target_str.replace('\'', "''");
        let needle = match kind_of(property) {
            CanonKind::Identifier => format!("\"{}\":\"{}\"", escaped_prop, escaped_val),
            CanonKind::Numeric | CanonKind::Boolean => format!("\"{}\":{}", escaped_prop, escaped_val),
            _ => escaped_val.clone(),
        };

        let prefilter = format!("data LIKE '%{}%'", needle);

        let batches = match table.query().only_if(prefilter.clone()).limit(500).execute().await {
            Ok(res) => res.try_collect::<Vec<_>>().await.unwrap_or_default(),
            Err(_) => {
                println!("[STORE] ⚠️ key-scoped prefilter failed ({}). Falling back to value-only ILIKE.", prefilter);
                let loose = format!("data ILIKE '%{}%'", escaped_val);
                match table.query().only_if(loose).limit(500).execute().await {
                    Ok(res) => res.try_collect::<Vec<_>>().await.unwrap_or_default(),
                    Err(_) => table.query().limit(2000).execute().await?.try_collect::<Vec<_>>().await?,
                }
            }
        };

        for batch in batches {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(9).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..batch.num_rows() {
                if let Ok(data) = serde_json::from_str::<Value>(datas.value(i)) {
                    if let Some(f_val) = data.get(property) {
                        let f_val_str = match f_val {
                            Value::String(s) => s.clone(),
                            Value::Number(n) => n.to_string(),
                            Value::Bool(b) => if *b { "1".to_string() } else { "0".to_string() },
                            _ => f_val.to_string().trim_matches('"').to_string(),
                        };
                        if f_val_str == target_str {
                            return Ok(Some((ids.value(i).to_string(), data)));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    pub async fn reset_database(&self) -> Result<()> {
        let tables = vec!["tasks", "talks", "items", "sales", "tracking", "event", "users", "pages", "item_chunks"];
        for name in tables {
            let _ = self.conn.drop_table(name, &[]).await;
            let _ = std::fs::remove_dir_all(format!("{}/{}.lance", self.base_path, name));
        }
        println!("[Store] LanceDB all tables dropped for factory reset.");
        self.init_task_table().await?;
        self.init_all_tables().await?;

        Ok(())
    }

    pub async fn init_chunks_table(&self) -> Result<()> {
        let uri = self.base_path.clone();
        let existing = self.conn.table_names().execute().await?;

        if existing.contains(&"item_chunks".to_string()) {
            match self.conn.open_table("item_chunks").execute().await {
                Ok(table) => {
                    let current_schema = table.schema().await.unwrap_or_else(|_| {
                        Arc::new(Schema::new(Vec::<Field>::new()))
                    });
                    let has_chunk_id = current_schema.field_with_name("chunk_id").is_ok();
                    let has_vector = current_schema.field_with_name("vector").is_ok();
                    let has_property = current_schema.field_with_name("property").is_ok();
                    let has_recipe_v3 = current_schema.field_with_name("embed_recipe_v3").is_ok();
                    if !has_chunk_id || !has_vector || !has_property || !has_recipe_v3 {
                        println!("[Store] item_chunks schema mismatch. Dropping for recreation.");
                        let _ = self.conn.drop_table("item_chunks", &[]).await;
                        let _ = std::fs::remove_dir_all(format!("{}/item_chunks.lance", uri));
                    } else {
                        return Ok(());
                    }
                },
                Err(_) => {
                    println!("[Store] Corrupted item_chunks table detected. Force dropping.");
                    let _ = self.conn.drop_table("item_chunks", &[]).await;
                    let _ = std::fs::remove_dir_all(format!("{}/item_chunks.lance", uri));
                }
            }
        }

        let existing_after = self.conn.table_names().execute().await?;
        if !existing_after.contains(&"item_chunks".to_string()) {
            let chunk_schema = Arc::new(Schema::new(vec![
                // 청크 식별
                Field::new("chunk_id", DataType::Utf8, false),
                Field::new("item_id", DataType::Utf8, false),
                Field::new("item_type", DataType::Utf8, false),

                // 청크 내용
                Field::new("chunk_text", DataType::Utf8, false),
                Field::new("property", DataType::Utf8, false),
                Field::new("property_format", DataType::Utf8, false),
                Field::new("value_part", DataType::Utf8, true),

                // 임베딩 (granite-embedding-97m-multilingual-r2 = 384차원)
                Field::new("vector", DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)), 384
                ), true),

                // 메타데이터
                Field::new("cc", DataType::Utf8, true),
                Field::new("bcc", DataType::Utf8, true),
                Field::new("ref", DataType::Utf8, true),
                Field::new("mode", DataType::Utf8, true),

                // 타임스탬프
                Field::new("created_at", DataType::Int64, false),
                Field::new("updated_at", DataType::Int64, false),

                // 🌟 [EMBEDDING RECIPE VERSION] 저장 벡터 합성식 버전 각인 (v3)
                Field::new("embed_recipe_v3", DataType::Utf8, true),
            ]));

            if let Err(_) = self.conn.create_empty_table("item_chunks", chunk_schema.clone()).execute().await {
                let _ = std::fs::remove_dir_all(format!("{}/item_chunks.lance", uri));
                let _ = self.conn.create_empty_table("item_chunks", chunk_schema).execute().await;
            }
            println!("[Store] item_chunks table created successfully.");
        }

        Ok(())
    }
    pub async fn upsert_chunk(
        &self,
        chunk_id: &str,
        item_id: &str,
        item_type: &str,
        chunk_text: &str,
        property: &str,
        property_format: &str,
        value_part: &str,
        vector: Option<Vec<f32>>,
        cc: Option<&str>,
        bcc: Option<&str>,
        ref_val: Option<&str>,
        mode: Option<&str>,
    ) -> Result<()> {
        let table = self.conn.open_table("item_chunks").execute().await?;
        let _ = table.delete(&format!("chunk_id = '{}'", chunk_id)).await;
        let safe_vector = match vector {
            Some(v) if v.len() == 384 => {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter().map(|x| x / norm).collect::<Vec<f32>>()
                } else {
                    v
                }
            },
            _ => vec![0.0; 384],
        };
        let now = chrono::Utc::now().timestamp_millis();

        let values_builder = Float32Array::from(safe_vector);
        let list_field = Field::new("item", DataType::Float32, true);
        let list_array = FixedSizeListArray::try_new(
            Arc::new(list_field), 384, Arc::new(values_builder), None
        )?;

        let schema = table.schema().await?;
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec![chunk_id.to_string()])),
            Arc::new(StringArray::from(vec![item_id.to_string()])),
            Arc::new(StringArray::from(vec![item_type.to_string()])),
            Arc::new(StringArray::from(vec![chunk_text.to_string()])),
            Arc::new(StringArray::from(vec![property.to_string()])),
            Arc::new(StringArray::from(vec![property_format.to_string()])),
            Arc::new(StringArray::from(vec![value_part.to_string()])),
            Arc::new(list_array),
            Arc::new(StringArray::from(vec![cc.unwrap_or("").to_string()])),
            Arc::new(StringArray::from(vec![bcc.unwrap_or("").to_string()])),
            Arc::new(StringArray::from(vec![ref_val.unwrap_or("").to_string()])),
            Arc::new(StringArray::from(vec![mode.unwrap_or("commerce").to_string()])),
            Arc::new(Int64Array::from(vec![now])),
            Arc::new(Int64Array::from(vec![now])),
            Arc::new(StringArray::from(vec!["v3:format-aware(chunk+anchor+leafvalue)".to_string()])),
        ])?;

        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn search_chunks(
        &self,
        query_vec: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, String, f32, f32)>> {
        let table = self.conn.open_table("item_chunks").execute().await?;

        // 벡터가 전부 0 이면 검색 불가
        if query_vec.iter().all(|&v| v == 0.0) {
            return Ok(Vec::new());
        }

        let property_pinned = filter
            .map(|f| f.contains("property = '"))
            .unwrap_or(false);

        let mut q = table.query();
        if let Some(f) = filter {
            if !f.trim().is_empty() {
                q = q.only_if(f.to_string());
            }
        }
        let normalized_query: Vec<f32> = {
            let norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                query_vec.iter().map(|x| x / norm).collect()
            } else {
                query_vec.to_vec()
            }
        };
        let overfetch = if property_pinned { limit * 12 } else { limit * 6 };
        let results = q
            .limit(overfetch)
            .nearest_to(normalized_query)?
            .column("vector")
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut chunks: Vec<(String, String, String, String, f32)> = Vec::new();

        for batch in results {
            let num_rows = batch.num_rows();
            if num_rows == 0 { continue; }
            let chunk_ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let item_ids = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let chunk_texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let properties = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

            // LanceDB nearest_to 는 _distance 컬럼을 마지막에 추가합니다
            let dist_idx = batch.num_columns() - 1;
            let distances = batch.column(dist_idx).as_any().downcast_ref::<Float32Array>();

            for i in 0..num_rows {
                let score = distances
                    .map(|d| (1.0f32 - d.value(i) / 2.0f32).clamp(0.0f32, 1.0f32))
                    .unwrap_or(0.0);

                chunks.push((
                    chunk_ids.value(i).to_string(),
                    item_ids.value(i).to_string(),
                    chunk_texts.value(i).to_string(),
                    properties.value(i).to_string(),
                    score,
                ));
            }
        }
        chunks.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        if property_pinned {
            println!(
                "  🎯 [PROPERTY PINNED] property 고정 검색 감지. 다양성 캡을 적용하지 않습니다. (후보 {}행 전량 보존)",
                chunks.len()
            );
        } else {
            let per_property_cap = std::cmp::max(2usize, limit / 2);
            let group_key = |chunk_id: &str, property: &str| -> String {
                if chunk_id.ends_with("_tn") || chunk_id.ends_with("_tr") {
                    format!("{}#alias", property)
                } else {
                    property.to_string()
                }
            };
            let mut prop_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut kept: Vec<(String, String, String, String, f32)> = Vec::with_capacity(chunks.len());
            let mut suppressed: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for c in chunks.into_iter() {
                let k = group_key(&c.0, &c.3);
                let n = prop_count.entry(k.clone()).or_insert(0);
                if *n >= per_property_cap {
                    *suppressed.entry(k).or_insert(0) += 1;
                    continue;
                }
                *n += 1;
                kept.push(c);
            }
            if !suppressed.is_empty() {
                let mut brief: Vec<String> = suppressed
                    .iter()
                    .map(|(p, n)| format!("{}({}행)", p, n))
                    .collect();
                brief.sort();
                println!(
                    "  🎛️ [PROPERTY DIVERSIFICATION] property 당 상한 {}행 적용 (별칭은 별도 그룹). 초과 억제: {:?}",
                    per_property_cap, brief
                );
            }
            chunks = kept;
        }

        // 🌟 [ALIAS HIT LOG] 어떤 별칭 청크가 실제로 창에 들어왔는지 남깁니다.
        //    지금까지 정방향 로그에 별칭이 한 줄도 찍히지 않아
        //    "저장이 안 된 것인지 검색이 안 된 것인지" 구분이 불가능했습니다.
        {
            let mut alias_hits: Vec<String> = Vec::new();
            for (cid, _iid, ctext, prop, s) in chunks.iter() {
                if cid.ends_with("_tn") || cid.ends_with("_tr") {
                    if alias_hits.len() < 8 {
                        alias_hits.push(format!("{}[{}] '{}' ({:.4})", prop, if cid.ends_with("_tn") { "native" } else { "roman" }, ctext, s));
                    }
                }
            }
            if !alias_hits.is_empty() {
                println!("  🔤 [ALIAS CHUNK HIT] 음차 별칭 청크가 후보 창에 진입했습니다: {:?}", alias_hits);
            }
        }

        let mut item_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut item_best_chunk: std::collections::HashMap<String, (String, String, String, f32)> = std::collections::HashMap::new();
        for (chunk_id, item_id, chunk_text, property, score) in &chunks {
            let entry = item_scores.entry(item_id.clone()).or_insert(0.0);
            *entry += score;

            let best = item_best_chunk.entry(item_id.clone()).or_insert_with(|| {
                (chunk_id.clone(), chunk_text.clone(), property.clone(), *score)
            });
            if *score > best.3 {
                *best = (chunk_id.clone(), chunk_text.clone(), property.clone(), *score);
            }
        }

        let mut final_results: Vec<(String, String, String, String, f32, f32)> = Vec::new();
        let mut sorted_items: Vec<(String, f32, f32)> = item_scores
            .into_iter()
            .map(|(id, total)| {
                let best = item_best_chunk.get(&id).map(|b| b.3).unwrap_or(0.0);
                (id, total, best)
            })
            .collect();
        sorted_items.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        for (item_id, total_score, _best) in sorted_items.into_iter().take(limit) {
            if let Some((chunk_id, chunk_text, property, best_score)) = item_best_chunk.remove(&item_id) {
                final_results.push((chunk_id, item_id, chunk_text, property, total_score, best_score));
            }
        }

        Ok(final_results)
    }

    pub async fn delete_chunks_by_item(&self, item_id: &str) -> Result<()> {
        let table = self.conn.open_table("item_chunks").execute().await?;
        table.delete(&format!("item_id = '{}'", item_id)).await?;
        Ok(())
    }

    pub async fn count_chunks_by_item(&self, item_id: &str) -> Result<usize> {
        let table = self.conn.open_table("item_chunks").execute().await?;
        let results = table.query()
            .only_if(format!("item_id = '{}'", item_id))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut count = 0usize;
        for batch in results {
            count += batch.num_rows();
        }
        Ok(count)
    }
}

pub fn json_property_needle(property: &str, value: &Value) -> String {
    use crate::utils::canonical::{kind_of, CanonKind};
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => if *b { "1".to_string() } else { "0".to_string() },
        _ => value.to_string().trim_matches('"').to_string(),
    };
    let ep = property.replace('\'', "''");
    let ev = raw.replace('\'', "''");
    match kind_of(property) {
        CanonKind::Identifier => format!("\"{}\":\"{}\"", ep, ev),
        CanonKind::Numeric | CanonKind::Boolean => format!("\"{}\":{}", ep, ev),
        _ => ev,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TradeDocument {
    // ── 봉투(Envelope) ──
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub flag: String,
    pub from: String,
    pub to: String,
    pub cc: String,
    pub bcc: String,
    #[serde(rename = "ref")]
    pub r#ref: String,
    pub mode: String,
    /// 확장 영역. 모든 도메인 값이 여기에 들어 있습니다. (JSON 문자열)
    pub json_data: String,
    #[serde(rename = "created_at")]
    pub created_at_ts: i64,
    #[serde(rename = "updated_at")]
    pub updated_at_ts: i64,
    // ── 검색 부품 (LanceDB 전용) ──
    pub text: String,
    pub masked_text: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub vision_vec: Vec<f32>,
}