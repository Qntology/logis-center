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
        let table = self.conn.open_table("talks").execute().await?;
        let schema = table.schema().await?;
        let now = chrono::Utc::now().timestamp_millis();
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
                Arc::new(Int64Array::from(vec![now])),
                Arc::new(Int64Array::from(vec![0])), // updated_at
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
        table.delete(&format!("task_id = '{}'", task_id)).await?;
        if let Some(t) = text {
            self.add_message(&uuid::Uuid::new_v4().to_string(), "system_task", t, Some(task_id), Some(status), None, None, None, None, None, Some("talk"), None).await?;
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

    // 🌟 [SINGLE ROUTER] 테이블 라우팅을 단 하나의 함수로 통일합니다.
    //    기존에는 delete_item / delete_items / find_item_by_property / search_items /
    //    upsert_item 이 각자 다른 match 문을 복붙해 놓아 서로 어긋났고,
    //    그 결과가 'review 는 items 에 저장되는데 event 에서 조회' 버그였습니다.
    //    v4 부터 도메인 타입은 전부 items 로 접히고, 구분은 type 컬럼이 담당합니다.
    fn resolve_table(table_or_type: &str) -> &'static str {
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

    pub async fn delete_item(&self, table_name: &str, id: &str) -> Result<()> {
        let target = Self::resolve_table(table_name);
        let table = self.conn.open_table(target).execute().await?;
        table.delete(&format!("id = '{}'", id)).await?;

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
    pub const SCHEMA_VERSION: &'static str = "v4:unified-envelope";

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
                        // 🌟 세대 각인 컬럼 하나만 확인하면 됩니다.
                        //    (기존처럼 컬럼을 하나하나 확인하는 방식은 스키마가 바뀔 때마다
                        //     검사 코드를 같이 고쳐야 해서 누락이 반복되었습니다)
                        let is_v4 = current_schema.field_with_name("schema_v4").is_ok();
                        // 🌟 도메인 컬럼 잔재가 있으면 구세대입니다.
                        let has_legacy_domain = current_schema.field_with_name("status").is_ok()
                            || current_schema.field_with_name("amount").is_ok()
                            || current_schema.field_with_name("is_masked").is_ok();

                        if !is_v4 || has_legacy_domain {
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
    
// 🌟 [CANONICALIZE v5 / RULE-BASED]
    //  ── 무엇이 바뀌었나 ──
    //   기존은 ID_KEYS / NUM_KEYS / BOOL_KEYS 라는 '이름 화이트리스트' 였습니다.
    //   그래서 Dexie 에 data.container_number 를 추가하면 여기 배열과
    //   find_item_by_property 의 복제본까지 총 4곳(배열 길이 상수 포함)을 고쳐야 했습니다.
    //
    //   이제는 '이미 존재하는 키를 순회' 하면서 crate::utils::canonical::kind_of() 로
    //   접미사/부분일치 규칙 판정을 받습니다.
    //   → 새 필드를 Dexie 에 추가해도 이 함수는 영원히 수정할 필요가 없습니다.
    //
    //  ── seed_defaults 의 의미 ──
    //   Dexie 의 data.* 인덱스는 undefined 값을 조용히 제외합니다.
    //   그래서 items 문서에는 '조회 축으로 쓰는 최소 기본값' 을 시딩해야 합니다.
    //   시딩 목록은 SEED_KEYS 하나로만 관리하며, 조회 축이 늘 때만 손댑니다.
    //   (조회 축이 아닌 단순 저장 필드는 시딩이 전혀 필요 없습니다)
    fn canonicalize_data(mut v: Value, seed_defaults: bool) -> Value {
        use crate::utils::canonical::{kind_of, iso_to_epoch_ms, CanonKind};

        // 🌟 [SEED KEYS] Dexie stores() 에 인덱스로 선언된 data.* 경로 중
        //    '값이 없을 때 기본값이 있어야 조회가 성립하는' 키만 나열합니다.
        //    이 목록은 main.ts 의 ITEMS_SCHEMA 와 대응하며,
        //    인덱스를 추가하지 않는 단순 확장 필드는 여기 넣을 필요가 없습니다.
        const SEED_KEYS: &[(&str, CanonKind)] = &[
            // ── 식별자 ──
            ("id", CanonKind::Identifier),
            ("no", CanonKind::Identifier),
            ("code", CanonKind::Identifier),
            ("index", CanonKind::Identifier),
            ("tracking_number", CanonKind::Identifier),
            ("goods", CanonKind::Identifier),
            ("order", CanonKind::Identifier),
            ("tracking", CanonKind::Identifier),
            ("stock_keeping_unit", CanonKind::Identifier),
            ("barcode", CanonKind::Identifier),
            ("digest", CanonKind::Identifier),
            // ── 수치 ──
            ("status", CanonKind::Numeric),
            ("amount", CanonKind::Numeric),
            ("sale_price", CanonKind::Numeric),
            ("supply_price", CanonKind::Numeric),
            ("quantity", CanonKind::Numeric),
            ("weight", CanonKind::Numeric),
            ("discount", CanonKind::Numeric),
            ("started_at", CanonKind::Numeric),
            ("expired_at", CanonKind::Numeric),
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

        // ── ① 기존 키 전량 정규화 (규칙 기반) ──
        //    새 필드도 여기서 자동으로 처리되므로 Rust 수정이 불필요합니다.
        let existing: Vec<String> = obj.keys().cloned().collect();
        for k in existing {
            let kind = kind_of(&k);
            if kind == CanonKind::Free { continue; }

            match kind {
                CanonKind::Identifier => {
                    let s = match obj.get(&k) {
                        Some(Value::Null) => String::new(),
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Number(n)) => n.to_string(),
                        Some(Value::Bool(b)) => if *b { "1".to_string() } else { "0".to_string() },
                        // 배열/객체는 식별자가 될 수 없으므로 건드리지 않습니다.
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                        None => String::new(),
                    };
                    obj.insert(k, json!(s));
                },
                CanonKind::Numeric => {
                    let n: f64 = match obj.get(&k) {
                        None | Some(Value::Null) => 0.0,
                        Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0),
                        Some(Value::Bool(b)) => if *b { 1.0 } else { 0.0 },
                        Some(Value::String(s)) => {
                            let t = s.trim();
                            // 🌟 status 는 'complete' 같은 상태 문자열이 들어올 수 있습니다.
                            if k == "status" {
                                crate::logic::parse_status(t) as f64
                            } else if let Some(ms) = iso_to_epoch_ms(t) {
                                // 🌟 [ISO DATE] scheduler 의 normalize_data 가 만든
                                //    "2024-01-01T12:00:00" 을 epoch ms 로 확정합니다.
                                //    이 처리가 없으면 숫자만 추출해 파싱에 실패하고
                                //    모든 기간 조건이 0 으로 뭉개집니다.
                                ms as f64
                            } else {
                                let cleaned: String = t.chars()
                                    .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                                    .collect();
                                cleaned.parse::<f64>().unwrap_or(0.0)
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
                        Some(Value::String(s)) => s == "1" || s.eq_ignore_ascii_case("true"),
                        Some(Value::Array(_)) | Some(Value::Object(_)) => continue,
                        _ => false,
                    };
                    obj.insert(k, json!(if b { 1 } else { 0 }));
                },
                CanonKind::Tags => {
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

    pub async fn upsert_item(
        &self, table_name: &str, id: &str, type_: &str, data_val: Value, vector: Option<Vec<f32>>,
        from: Option<&str>, to: Option<&str>, cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>, digest: Option<&str>
    ) -> Result<()> {
         let target = Self::resolve_table(if table_name.is_empty() { "items" } else { table_name });
         let table = self.conn.open_table(target).execute().await?;

         let final_id = if id.is_empty() {
             data_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string()
         } else { id.to_string() };

         if final_id.is_empty() { return Ok(()); }

         // 🌟 [SKIP GUARD] digest 는 이제 물리 컬럼이 아니라 data.digest 입니다.
         //    기존 문서의 digest 를 읽으려면 json_data 를 파싱해야 합니다.
         let new_digest = digest.unwrap_or("").to_string();
         let new_updated_at = data_val.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0);

         if let Some(doc) = self.get_item_by_id(target, &final_id).await? {
             if doc.updated_at_ts >= new_updated_at && !new_digest.is_empty() {
                 let old_digest = serde_json::from_str::<Value>(&doc.json_data)
                     .ok()
                     .and_then(|v| v.get("digest").and_then(|d| d.as_str()).map(|s| s.to_string()))
                     .unwrap_or_default();
                 if old_digest == new_digest {
                     return Ok(());
                 }
             }
         }

         println!("[DEBUG] store.upsert_item (v4) - Table: {}, ID: {}, Type: {}", target, final_id, type_);

         let _ = table.delete(&format!("id = '{}'", final_id)).await;

         let mut final_data = data_val.clone();

         // gzip/base64 로 압축되어 온 서버 페이로드 해제 (기존 동작 유지)
         if let Some(blob_base64) = final_data.get("data").and_then(|v| v.as_str()) {
             if blob_base64.len() > 50 {
                 use base64::prelude::BASE64_STANDARD;
                 use base64::Engine;
                 if let Ok(decoded) = BASE64_STANDARD.decode(blob_base64) {
                     if let Ok(decompressed) = crate::utils::compression::decompress_to_value(&decoded) {
                         final_data = decompressed;
                     }
                 }
             }
         }

         // 🌟 봉투 값들을 data 안에도 동봉합니다.
         //    Dexie 는 data.* 만 인덱싱하므로, 프론트엔드가 봉투/확장을 구분 없이 읽을 수 있게 됩니다.
         let mode_str = data_val.get("mode").and_then(|v| v.as_str()).unwrap_or("commerce").to_string();

         // 🌟 [DRAFT MARKER PRESERVE] updated_at = 0 은 '값 없음' 이 아니라
         //    '리스트 스캔으로 껍데기만 만들어진 draft' 라는 3개 저장소 공통 계약입니다.
         //    (proxy/src/index.ts 의 `if(updated_at){ count++ } else { draft++ }` 와 동일 규칙)
         //
         //    기존 코드는 `if new_updated_at > 0 { .. } else { now() }` 로 0 을 현재 시각으로
         //    덮어써 버렸고, 그 결과 scheduler 가 넣은 draft(0) / relay draft(0) /
         //    서버가 내려준 draft(0) 이 전부 count 로 승격되어
         //    Pages 트리의 Draft 표기가 항상 0 이 되었습니다.
         //
         //    판정 기준은 '값이 0인가' 가 아니라 '키가 존재하는가' 입니다.
         //    키가 아예 없는 경우(pages 캐시 등)에만 현재 시각을 부여합니다.
         let has_updated_key = data_val.get("updated_at").is_some();
         let wall_now = chrono::Utc::now().timestamp_millis();
         let updated_ts = if has_updated_key { new_updated_at } else { wall_now };

         // 🌟 created_at 은 draft 여부와 무관하게 항상 실제 시각이어야 합니다.
         //    (기존에는 now_ts 를 폴백으로 써서 updated_at 과 결합돼 있었습니다)
         let created_at = data_val.get("created_at")
             .and_then(|v| v.as_i64())
             .filter(|v| *v > 0)
             .unwrap_or(wall_now);

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
             if !new_digest.is_empty() {
                 obj.insert("digest".to_string(), json!(new_digest.clone()));
             }
         }

         // 🌟 Dexie 와 동일 규칙으로 정규화한 뒤 저장합니다.
         //    users / pages 는 도메인 필드 인덱스가 없으므로 기본값 시딩을 끕니다.
         //    (팀 통계 문서에 sale_price: 0 같은 키가 48개 붙는 오염을 방지)
         //
         //    🌟 [ANALYTICS] analytics 트랙 행동 로그(click / hover / change / report)와
         //       관리자 Q&A(question / answer)도 commerce 도메인 필드를 갖지 않습니다.
         //       main.ts 의 NON_SEED_TYPES 와 반드시 동일한 집합이어야 두 저장소가 일치합니다.
         let non_seed_type = matches!(
             type_,
             "team" | "user" | "member"
                 | "click" | "hover" | "change" | "report"
                 | "question" | "answer"
         );
         let seed_defaults = !matches!(target, "users" | "pages") && !non_seed_type;
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

         // 🌟 컬럼 순서는 init_all_tables 의 schema 정의와 1:1 로 일치해야 합니다.
         //    0 id / 1 type / 2 flag / 3 from / 4 to / 5 cc / 6 bcc / 7 ref / 8 mode
         //    9 data / 10 created_at / 11 updated_at / 12 vector / 13 text / 14 masked_text / 15 schema_v4
         //    🌟 updated_ts 가 0 이면 draft 입니다. 물리 컬럼에도 0 을 그대로 남겨야
         //       프론트엔드(Dexie)와 서버(proxy)의 draft 판정이 일치합니다.
         let batch = RecordBatch::try_new(schema.clone(), vec![
                Arc::new(StringArray::from(vec![final_id])),
                Arc::new(StringArray::from(vec![type_])),
                Arc::new(StringArray::from(vec![flag_str])),
                Arc::new(StringArray::from(vec![from.unwrap_or("")])),
                Arc::new(StringArray::from(vec![to.unwrap_or("")])),
                Arc::new(StringArray::from(vec![cc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![bcc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![r#ref.unwrap_or("")])),
                Arc::new(StringArray::from(vec![mode_str])),
                Arc::new(StringArray::from(vec![json_str])),
                Arc::new(Int64Array::from(vec![created_at])),
                Arc::new(Int64Array::from(vec![updated_ts])),
                Arc::new(list_array),
                Arc::new(StringArray::from(vec![text_content])),
                Arc::new(StringArray::from(vec![masked_text_content])),
                Arc::new(StringArray::from(vec![Self::SCHEMA_VERSION])),
         ])?;
         table.add(vec![batch]).execute().await?;
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

        // 🌟 [ENVELOPE] flag / mode / text 를 data 안에 명시적으로 넣습니다.
        //    canonicalize_data 가 ID/NUM/BOOL 키만 건드리므로 base 통계 트리는 그대로 보존됩니다.
        //    text 가 비면 LanceDB text 컬럼이 빈 문자열이 되어 FTS 대상에서 제외되므로
        //    최소한의 식별 문구를 넣어 둡니다.
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

        self.upsert_item("users", &team_id, "team", team_data, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        self.upsert_item("users", user_address, "user", user_data, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        Ok(())
    }

    // 🌟 [ROW READER] RecordBatch → TradeDocument 변환을 한 곳으로 모읍니다.
    //  기존에는 get_all_items / get_item_by_id 가 컬럼 인덱스를 각자 하드코딩해서
    //  스키마가 바뀔 때마다 두 곳을 동시에 고쳐야 했고, 실제로 어긋난 적이 있습니다.
    //
    //  ⚠️ [SCHEMA CONTRACT] 아래 인덱스는 init_all_tables 의 Field 선언 순서와 1:1 대응입니다.
    //     0 id / 1 type / 2 flag / 3 from / 4 to / 5 cc / 6 bcc / 7 ref / 8 mode
    //     9 data / 10 created_at / 11 updated_at / 12 vector / 13 text / 14 masked_text / 15 schema_v4
    //
    //     봉투 컬럼을 '중간에' 추가하면 뒤 인덱스가 전부 밀려 search_items(column(9)) 등
    //     다른 지점까지 조용히 깨집니다. 봉투를 늘려야 한다면 반드시 '끝에' 추가하고
    //     SCHEMA_VERSION 을 올려 구세대 테이블이 drop 되도록 하세요.
    //     그보다 먼저 'data.* 로 내릴 수 없는가' 를 검토하는 것이 v4 설계 의도입니다.
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
        let texts       = batch.column(13).as_any().downcast_ref::<StringArray>().unwrap();
        let masked      = batch.column(14).as_any().downcast_ref::<StringArray>().unwrap();

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

        // 정렬을 위해 전량을 메모리에 올린 뒤 슬라이스합니다. (기존 동작 유지)
        let results = q.execute().await?.try_collect::<Vec<_>>().await?;
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
    
    // 🌟 [2-TRACK RECALL SEARCH / v4]
    //  Track 1(Column Matching) 을 완전히 제거합니다.
    //
    //  ── 왜 제거하는가 ──
    //   v4 부터 filter 인자는 '스코프' 전용입니다. (type / mode / cc / bcc / ref / 시간)
    //   도메인 조건(가격/수량/송장번호/상태)은 SQL 로 내려오지 않고 Dexie 가 처리합니다.
    //   따라서 '조건 매칭 +3.0' 이라는 트랙이 성립할 수 없습니다.
    //   기존 has_real_condition 가드는 이 상황을 이미 부분적으로 방어하고 있었는데,
    //   v4 에서는 그 분기가 항상 false 가 되므로 코드째로 걷어냅니다.
    //
    //  ── 역할 재정의 ──
    //   LanceDB = 리콜(넓게 긁기). 점수는 '의미 근접도' 만 표현합니다.
    //   Dexie   = 정밀도(정확히 자르기). 조건/정렬/페이징 담당.
    //   → 그래서 fetch_limit 을 넉넉히(요청의 4배, 최소 200) 잡습니다.
    //     Dexie 가 뒤에서 조건으로 잘라내므로 여기서 좁히면 정답이 사라집니다.
    pub async fn search_items(&self, table_name: &str, query_text: &str, query_vec: Vec<f32>, limit: usize, offset: usize, filter: Option<String>, use_fts: bool) -> Result<Vec<(String, String, f32)>> {
         let target = Self::resolve_table(if table_name.is_empty() { "items" } else { table_name });
         let table = self.conn.open_table(target).execute().await?;

         let mut combined: std::collections::HashMap<String, (String, f32)> = std::collections::HashMap::new();

         // 🌟 [OVERFETCH] Dexie 정밀 필터가 뒤에 붙으므로 후보를 넓게 확보합니다.
         let fetch_limit = std::cmp::max(200, (limit + offset) * 4);

         // 스코프 필터를 정리합니다. 비어 있으면 아예 걸지 않습니다.
         let scope: Option<String> = filter.as_ref().and_then(|f| {
             let t = f.trim();
             if t.is_empty() { None } else { Some(t.to_string()) }
         });

         // =======================================================
         // 🌟 [Track A] Native Full Text Search (Tantivy ngram 역인덱스)
         //     가중치 2.0. 어휘 일치 신호.
         // =======================================================
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
                 // 타이핑 중(Live Search) 미완성 단어 대응 ILIKE 폴백
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

         // =======================================================
         // 🌟 [Track B] Vector Search (ANN)
         //     가중치 1.0 시작, 랭크당 -0.001. 의미 근접 신호.
         // =======================================================
         let is_empty_vec = query_vec.iter().all(|&x| x == 0.0);

         if !is_empty_vec {
             let mut vq = table.query();
             if let Some(ref f) = scope { vq = vq.only_if(f.clone()); }

             if let Ok(vq_with_vector) = vq.limit(fetch_limit).nearest_to(query_vec) {
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
                 }
             }
         }

         // =======================================================
         // 🌟 [Track C] Scope-Only Recall
         //  질의 텍스트도 없고 벡터도 0 이면(= 순수 목록 조회) 스코프 결과를 그대로 돌려줍니다.
         //  기존에는 이 경우 Track 1 이 blanket +3.0 을 뿌려 목록처럼 동작했는데,
         //  Track 1 을 없앴으므로 명시적 경로로 분리합니다.
         // =======================================================
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
                             // 최신순 타이브레이커만 부여합니다. (의미 신호 없음)
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

    // 🌟 [PROPERTY LOOKUP v5 / KEY-SCOPED PREFILTER]
    //  v4 는 `data ILIKE '%값%'` 로만 좁혔습니다. 그런데 값이 짧으면(예: index "18")
    //  전혀 무관한 문서의 다른 키(`"quantity":118`)까지 후보로 끌려와
    //  500건 상한 안에서 정답이 밀려나는 사고가 발생했습니다.
    //
    //  v5 는 canonicalize_data 가 확정한 '직렬화 형태' 를 그대로 프리필터에 씁니다.
    //    · 식별자류(String 확정) → `"property":"값"`
    //    · 수치류(Number 확정)   → `"property":값`
    //  키까지 포함시키므로 오탐이 구조적으로 사라지고, 상한 500건이 실효를 갖습니다.
    //
    //  ⚠️ 이 함수는 scheduler 의 RELAY 경로 전용입니다.
    //     사용자 검색/목록 조회의 도메인 조건은 전부 Dexie(executeDexiePlan)가 담당하며,
    //     LanceDB 는 벡터/FTS/봉투 스코프만 책임집니다.
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

        // 🌟 [SINGLE SOURCE] canonicalize_data 와 '완전히 같은 판정 함수' 를 씁니다.
        //    기존에는 배열이 복제되어 있어 한쪽만 고치면
        //    needle 이 `"key":123` vs `"key":"123"` 으로 어긋나 프리필터가 0건이 됐습니다.
        use crate::utils::canonical::{kind_of, CanonKind};

        let escaped_prop = property.replace('\'', "''");
        let escaped_val = target_str.replace('\'', "''");

        // 🌟 [KEY-SCOPED NEEDLE] serde_json 은 공백 없이 `"key":value` 로 직렬화합니다.
        let needle = match kind_of(property) {
            CanonKind::Identifier => format!("\"{}\":\"{}\"", escaped_prop, escaped_val),
            CanonKind::Numeric | CanonKind::Boolean => format!("\"{}\":{}", escaped_prop, escaped_val),
            // 배열/미분류 키는 형태를 확신할 수 없으므로 값만으로 좁히고 아래에서 정확 비교합니다.
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
        // 🌟 [v4] sales / tracking / event 는 이미 폐기되었지만,
        //    구버전에서 넘어온 사용자를 위해 drop 대상에는 남겨 둡니다.
        let tables = vec!["tasks", "talks", "items", "sales", "tracking", "event", "users", "pages", "item_chunks"];
        for name in tables {
            let _ = self.conn.drop_table(name, &[]).await;
            let _ = std::fs::remove_dir_all(format!("{}/{}.lance", self.base_path, name));
        }
        println!("[Store] LanceDB all tables dropped for factory reset.");

        // 테이블 초기화 함수 재호출하여 빈 껍데기로 복구
        self.init_task_table().await?;
        self.init_all_tables().await?;

        Ok(())
    }

    // =====================================================================
    // 🌟 [PHASE D] item_chunks 테이블 — 청크 단위 코사인 유사도 검색용
    // =====================================================================

    /// [PHASE D-1] item_chunks 테이블 스키마를 생성합니다.
    /// 앱 시작 시 init_all_tables() 이후에 호출됩니다.
    /// 기존 테이블이 존재하면 스키마 호환성 검사 후 그대로 사용합니다.
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
                    // 🌟 [EMBEDDING RECIPE VERSION] 저장 벡터 합성식이 바뀌면 기존 청크는
                    //    새 질의 벡터와 정합하지 않습니다. 스키마가 같아도 강제 재구축이 필요하므로
                    //    레시피 버전을 컬럼으로 각인하고, 버전이 다르면 테이블을 드롭합니다.
                    // 🌟 (v2 = chunk 0.5 + anchor 0.2 + localized 0.3 — 라벨 블롭이 값을 희석)
                    // 🌟 (v3 = 형식 인지 가중치 + localized 를 "{leaf_label} {value}" 로 축약 + Enum 라벨 지배)
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

    /// [PHASE D-2] 청크 1건을 item_chunks 테이블에 삽입합니다.
    /// 동일 chunk_id 가 이미 존재하면 삭제 후 재삽입합니다 (upsert 시맨틱).
    ///
    /// # 인자
    ///   - chunk_id:       UUID 기반 청크 고유 식별자
    ///   - item_id:        원본 item 의 해시 ID (FK)
    ///   - item_type:      도메인 타입 ("goods", "order", "tracking" 등)
    ///   - chunk_text:     자연어 청크 원문
    ///   - property:       PLINKO 확정 속성명 (snake_case)
    ///   - property_format: 형식 문자열 ("Numeric", "Text", "Enum" 등)
    ///   - value_part:     청크에서 추출한 실제 값 부분
    ///   - vector:         384차원 임베딩 벡터
    ///   - cc, bcc, ref_val, mode: 메타데이터
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

        // 기존 동일 chunk_id 삭제 (upsert)
        let _ = table.delete(&format!("chunk_id = '{}'", chunk_id)).await;

        let safe_vector = match vector {
            Some(v) if v.len() == 384 => v,
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

    /// [PHASE D-3] item_chunks 테이블에서 코사인 유사도 벡터 검색을 수행합니다.
    /// STAGE-4 (검색 시) 에서 호출됩니다.
    ///
    /// # 인자
    ///   - query_vec:  검색 질의의 임베딩 벡터 (384차원)
    ///   - limit:      반환할 최대 청크 수
    ///   - filter:     SQL 필터 (예: "item_type = 'goods' AND mode = 'commerce'")
    ///
    /// # 반환
    ///   Vec<(chunk_id, item_id, chunk_text, property, group_score, best_cos)>
    ///     - group_score : 그 item 이 확보한 청크 점수의 합산 (증거의 '양')
    ///     - best_cos    : 그 item 의 최고 코사인 (0.0~1.0, 증거의 '질')
    ///
    /// 🌟 [반환값 분리 이유] 기존에는 합산 점수 하나만 돌려주었고, lib.rs 가 그것을
    ///    코사인이라 가정하여 트랙 가중치(Column 3.0 / FTS 2.0 / CrossLingual 1.5)를 곱했습니다.
    ///    (log 실측: score 3.0146 × 2.0 + × 1.5 = 10.5510)
    ///    코사인 상한 1.0 을 전제로 설계된 가중치 체계가 무너져,
    ///    '청크가 많이 살아남은 item' 이 '질의와 실제로 가까운 item' 을 압도했습니다.
    ///    (Beige Wool Coat 가 '니트 가디건' 질의에서 RANK 2, Cable Knit Sweater 는 RANK 7)
    ///    이제 두 값을 분리해 돌려주고, 가중치는 best_cos 에만 곱하도록 합니다.
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

        // 🌟 [PROPERTY-PINNED DETECTION] 호출부가 `property = '...'` 로 property 를
        //    이미 하나로 고정한 타겟 검색인지 판정합니다.
        //    이 SQL 문자열은 전부 우리 코드(lib.rs STAGE-4C / 4D)가 생성하므로
        //    '의미 판정' 이 아니라 '우리가 만든 술어의 존재 여부' 라는 구조적 사실입니다.
        //    고정 검색에서 property 다양성 캡을 적용하면 정확히 정반대로 작동합니다.
        //    (log 실측: property='title' 고정 검색인데 "title(16행)" 억제 → 별칭 전멸)
        let property_pinned = filter
            .map(|f| f.contains("property = '"))
            .unwrap_or(false);

        let mut q = table.query();
        if let Some(f) = filter {
            if !f.trim().is_empty() {
                q = q.only_if(f.to_string());
            }
        }

        // 🌟 [QUERY VECTOR NORMALIZE] 저장 벡터는 upsert_chunk 직전에 L2 정규화되어 있습니다.
        //    질의 벡터는 정규화되지 않은 채 들어와 L2 거리 스케일이 어긋났습니다.
        //    양쪽을 정규화해야 L2² = 2 - 2cos 관계가 성립합니다.
        let normalized_query: Vec<f32> = {
            let norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                query_vec.iter().map(|x| x / norm).collect()
            } else {
                query_vec.to_vec()
            }
        };

        // 🌟 [OVERFETCH 확대] property 다양성 캡을 적용하려면 후보 창이 충분히 커야 합니다.
        //    저변별 청크가 상한에 걸려 버려지는 만큼을 미리 확보합니다.
        //    🌟 property 고정 검색은 캡이 없으므로 오버페치를 더 크게 잡아
        //    원본 청크와 음차 별칭(_tn/_tr)이 함께 창에 들어오도록 보장합니다.
        let overfetch = if property_pinned { limit * 12 } else { limit * 6 };
        let results = q
            .limit(overfetch)
            .nearest_to(normalized_query)?
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut chunks: Vec<(String, String, String, String, f32)> = Vec::new();

        for batch in results {
            let num_rows = batch.num_rows();
            if num_rows == 0 { continue; }

            // 컬럼 인덱스: 0=chunk_id, 1=item_id, 2=item_type, 3=chunk_text,
            //              4=property, 5=property_format, 6=value_part, 7=vector, ...
            let chunk_ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let item_ids = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let chunk_texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let properties = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

            // LanceDB nearest_to 는 _distance 컬럼을 마지막에 추가합니다
            let dist_idx = batch.num_columns() - 1;
            let distances = batch.column(dist_idx).as_any().downcast_ref::<Float32Array>();

            for i in 0..num_rows {
                // 🌟 [DISTANCE → SIMILARITY FIX]
                //    distance_type 을 지정하지 않았으므로 LanceDB 기본값인 L2(제곱거리)가 옵니다.
                //    정규화 벡터에서 L2² = 2 - 2cos 이므로 올바른 변환은 cos = 1 - d/2 입니다.
                //    기존 `1.0 - d` 는 2·cos - 1 이 되어 코사인 0.5 미만이 음수가 되고,
                //    item 별 그룹 합산이 2·Σcos - n 으로 왜곡되어
                //    매칭 청크가 많은 아이템이 구조적으로 불리해졌습니다.
                //    설령 백엔드가 코사인 거리(1-cos)를 돌려주더라도 이 식은 (1+cos)/2 로
                //    단조 증가를 유지하므로 순위가 깨지지 않고 음수도 발생하지 않습니다.
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

        // 점수 내림차순 정렬
        chunks.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        // 🌟 [PROPERTY DIVERSIFICATION] 저변별 청크가 후보 윈도우를 독점하는 것을 차단합니다.
        //    status 청크는 "It is currently in 'complete' status" 로 전 아이템에서
        //    바이트 단위로 동일하여 변별력이 0인데도, 오버페치 창을 전부 채워
        //    정작 값이 담긴 title 청크가 후보에 진입조차 못 했습니다.
        //
        //    🌟 [PINNED BYPASS] 단, 호출부가 property 를 이미 하나로 고정했다면
        //    이 캡은 존재 이유가 사라지고 오히려 정답 청크를 학살합니다.
        //    (log 실측: property='title' 고정 검색에서 "title(16행)" 억제)
        //    별칭(_tn/_tr)은 원본과 같은 property 를 쓰므로 캡의 1순위 희생양이었습니다.
        //
        //    🌟 [ALIAS GROUP SPLIT] 캡을 적용하는 전역 검색에서도, 음차 별칭은
        //    원본 청크와 '다른 표기 체계' 를 담은 별개 증거이므로 같은 슬롯을 두고
        //    경쟁시키면 안 됩니다. chunk_id 접미어(_tn/_tr)라는 구조적 사실만으로
        //    별도 그룹키를 부여하여 원본과 별칭이 나란히 생존하도록 합니다.
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

        // item_id 기준 그룹핑: 동일 item 의 여러 청크 점수를 합산하여
        // 최종 상위 limit 개 item 을 반환합니다.
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

        // 🌟 [RANKING BASIS] 대표 정렬 기준을 '최고 코사인' 으로 바꿉니다.
        //    합산(total)은 '증거의 양' 이지 '질의와의 가까움' 이 아닙니다.
        //    합산으로 정렬하면 무관한 값이라도 청크 수가 많은 item 이 이깁니다.
        //    합산은 동률을 깨는 보조 기준으로만 사용합니다.
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

    /// [PHASE D-4] 특정 item_id 에 연관된 모든 청크를 삭제합니다.
    /// item 삭제 또는 재추출 시 호출됩니다.
    pub async fn delete_chunks_by_item(&self, item_id: &str) -> Result<()> {
        let table = self.conn.open_table("item_chunks").execute().await?;
        table.delete(&format!("item_id = '{}'", item_id)).await?;
        Ok(())
    }

    /// [PHASE D-5] 특정 item_id 의 청크 개수를 반환합니다.
    /// 재인덱싱 여부 판정에 사용됩니다.
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

// 🌟 [ENVELOPE v4] 도메인 필드 55개를 전부 제거합니다.
//  기존 구조체는 무역 문서 전용 컬럼(vessel/pol/pod/incoterms/...)을 Rust 타입에 못 박아 두어
//  새 도메인이 추가될 때마다 구조체 → LanceDB 스키마 → 프론트엔드 3곳을 동시에 고쳐야 했습니다.
//  실제로는 대부분 채워지지도 않은 채(Default::default()) 직렬화 비용만 발생하고 있었습니다.
//
//  v4 부터 도메인 값은 전부 json_data(= data 컬럼) 안에 있고,
//  프론트엔드는 Dexie 의 data.* 중첩 인덱스로 쿼리합니다.
//  → 이 구조체는 앞으로 영원히 변경되지 않습니다.
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
}