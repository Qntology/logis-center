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

    pub async fn delete_item(&self, table_name: &str, id: &str) -> Result<()> {
        
        let target = match table_name {
            "sales" | "goods" | "order" => "sales",
            "tracking" | "receiving" | "shipping" => "tracking",
            "event" | "coupon" => "event",
            "member" | "team" | "user" => "users",
            "talk" | "prompt" | "ai_search" => "talks",
            "pages" => "pages",
            "items" => "items",
            t if t.starts_with("commerce_") => &t[9..],
            _ => "items"
        };
        let table = self.conn.open_table(target).execute().await?;
        table.delete(&format!("id = '{}'", id)).await?;

        // 🌟 [PHASE D] 연관 청크 동시 삭제
        let _ = self.delete_chunks_by_item(id).await;

        Ok(())
    }

    pub async fn delete_items(&self, table_name: &str, ids: Vec<String>) -> Result<()> {
        
        let target = match table_name {
            "sales" | "goods" | "order" => "sales",
            "tracking" | "receiving" | "shipping" => "tracking",
            "event" | "coupon" => "event",
            "member" | "team" | "user" => "users",
            "talk" | "prompt" | "ai_search" => "talks",
            "pages" => "pages",
            "items" => "items",
            t if t.starts_with("commerce_") => &t[9..],
            _ => "items"
        };
        let table = self.conn.open_table(target).execute().await?;
        let id_list = ids.iter().map(|id| format!("'{}'", id)).collect::<Vec<_>>().join(",");
        table.delete(&format!("id IN ({})", id_list)).await?;

        // 🌟 [PHASE D] 연관 청크 동시 삭제
        for id in &ids {
            let _ = self.delete_chunks_by_item(id).await;
        }

        Ok(())
    }
    
    pub async fn init_all_tables(&self) -> Result<()> {
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        let item_field = Field::new("item", DataType::Float32, true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("from", DataType::Utf8, true),
            Field::new("to", DataType::Utf8, true),
            Field::new("cc", DataType::Utf8, true),
            Field::new("bcc", DataType::Utf8, true),
            Field::new("ref", DataType::Utf8, true),
            Field::new("digest", DataType::Utf8, true),
            Field::new("status", DataType::Int32, true), 
            Field::new("amount", DataType::Float32, true),
            Field::new("vector", DataType::FixedSizeList(Arc::new(item_field), 384), true),
            Field::new("text", DataType::Utf8, false),
            Field::new("masked_text", DataType::Utf8, true),
            Field::new("data", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false), 
            Field::new("updated_at", DataType::Int64, false),
            Field::new("mode", DataType::Utf8, true), 
            Field::new("is_masked", DataType::Boolean, true),
        ]));
        
        let uri = self.base_path.clone();
        let existing = self.conn.table_names().execute().await?;
        
        for name in tables {
            if existing.contains(&name.to_string()) {
                match self.conn.open_table(name).execute().await {
                    Ok(table) => {
                        let current_schema = table.schema().await.unwrap_or_else(|_| Arc::new(Schema::new(Vec::<Field>::new())));
                        let has_ref = current_schema.field_with_name("ref").is_ok();
                        let has_mode = current_schema.field_with_name("mode").is_ok(); 
                        let has_masked_text = current_schema.field_with_name("masked_text").is_ok();
                        let has_is_masked = current_schema.field_with_name("is_masked").is_ok();
                        let status_is_int = if let Ok(field) = current_schema.field_with_name("status") {
                            field.data_type() == &DataType::Int32
                        } else { false };

                        if !has_ref || !status_is_int || !has_mode || !has_masked_text || !has_is_masked { 
                            println!("[Store] Schema mismatch for {}. Dropping and recreating...", name);
                            let _ = self.conn.drop_table(name, &[]).await;
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

            
            // 오직 원본 데이터를 모두 참조하고 있는 마스터 테이블인 "items"에만 전용 FTS 인덱스를 구축합니다.
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

        // 🌟 [PHASE D] item_chunks 테이블 초기화
        self.init_chunks_table().await?;

        Ok(())
    }
    
    pub async fn upsert_item(
        &self, table_name: &str, id: &str, type_: &str, data_val: Value, vector: Option<Vec<f32>>,
        from: Option<&str>, to: Option<&str>, cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>, digest: Option<&str>
    ) -> Result<()> {
         let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
         let table = self.conn.open_table(target).execute().await?;

         let final_id = if id.is_empty() { 
             data_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string() 
         } else { id.to_string() };

         if final_id.is_empty() { return Ok(()); }

         
         let existing = self.get_item_by_id(target, &final_id).await?;
         if let Some(doc) = existing {
             let new_updated_at = data_val.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0);
             let new_digest = digest.unwrap_or("");

             // 업데이트 시간이 같고 다이제스트가 같으면 불필요한 쓰기 스킵
             if doc.updated_at_ts >= new_updated_at && !new_digest.is_empty() && doc.digest == new_digest {
                 return Ok(());
             }
         }

         println!("[DEBUG] store.upsert_item (Updated) - Table: {}, ID: {}, Type: {}", target, final_id, type_);
         
         let _ = table.delete(&format!("id = '{}'", final_id)).await;
         let mut final_data = data_val.clone();
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
         if let Some(obj) = final_data.as_object_mut() {
             if let Some(tn) = obj.get("tracking_number").cloned() {
                 if obj.get("tracking").is_none() { obj.insert("tracking".to_string(), tn); }
             }
             if let Some(p) = obj.get("price").cloned() {
                 if obj.get("sale_price").is_none() { obj.insert("sale_price".to_string(), p); }
             }
         }
         let json_str = final_data.to_string();
         let text_content = final_data.get("text").and_then(|s| s.as_str()).unwrap_or("").to_string();
         let masked_text_content = final_data.get("masked_text").and_then(|s| s.as_str()).unwrap_or("").to_string();
         let status = data_val.get("status").and_then(|v| v.as_str()).map(|s| crate::logic::parse_status(s)).unwrap_or(0);
         let amount = data_val.get("total_amount").or_else(|| data_val.get("sale_price")).or_else(|| data_val.get("supply_price")).or_else(|| data_val.get("price")).or_else(|| data_val.get("shipping_fee")).or_else(|| data_val.get("discount")).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))).unwrap_or(0.0) as f32;
         let doc_date_str = data_val.get("order_date").or_else(|| data_val.get("registration_date")).or_else(|| data_val.get("release_date")).or_else(|| data_val.get("manufacture_date")).or_else(|| data_val.get("shipping_date")).or_else(|| data_val.get("started_at")).or_else(|| data_val.get("expired_at")).or_else(|| data_val.get("payment_date")).and_then(|v| v.as_str()).unwrap_or("");
         
         
         let now_ts = data_val.get("updated_at").and_then(|v| v.as_i64()).unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
         let created_at = if !doc_date_str.is_empty() {
             chrono::DateTime::parse_from_rfc3339(doc_date_str).map(|dt| dt.timestamp_millis()).unwrap_or_else(|_| chrono::NaiveDateTime::parse_from_str(doc_date_str, "%Y-%m-%dT%H:%M:%S").map(|dt| dt.and_utc().timestamp_millis()).unwrap_or(now_ts))
         } else { 
             data_val.get("created_at").and_then(|v| v.as_i64()).unwrap_or(now_ts) 
         };
         
         
         let mode_str = data_val.get("mode").and_then(|v| v.as_str()).unwrap_or("commerce").to_string();
         let is_masked_val = data_val.get("is_masked").and_then(|v| v.as_bool()).unwrap_or(false);
         
         let schema = table.schema().await?;
         
         
         let safe_vector = match vector {
             Some(v) if v.len() == 384 => v,
             _ => vec![0.0; 384],
         };
         let values_builder = Float32Array::from(safe_vector);
         
         let list_field = Field::new("item", DataType::Float32, true);
         let list_array = FixedSizeListArray::try_new(Arc::new(list_field), 384, Arc::new(values_builder), None)?;
         let batch = RecordBatch::try_new(schema.clone(), vec![
                Arc::new(StringArray::from(vec![final_id])), Arc::new(StringArray::from(vec![type_])),
                Arc::new(StringArray::from(vec![from.unwrap_or("")])), Arc::new(StringArray::from(vec![to.unwrap_or("")])),
                Arc::new(StringArray::from(vec![cc.unwrap_or("")])), Arc::new(StringArray::from(vec![bcc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![r#ref.unwrap_or("")])), Arc::new(StringArray::from(vec![digest.unwrap_or("")])),
                Arc::new(Int32Array::from(vec![status])), Arc::new(Float32Array::from(vec![amount])),
                Arc::new(list_array), Arc::new(StringArray::from(vec![text_content])), Arc::new(StringArray::from(vec![masked_text_content])), Arc::new(StringArray::from(vec![json_str])),
                Arc::new(Int64Array::from(vec![created_at])), Arc::new(Int64Array::from(vec![now_ts])),
                Arc::new(StringArray::from(vec![mode_str])), Arc::new(BooleanArray::from(vec![Some(is_masked_val)])),
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
        let team_data = json!({"flag": flag, "name": format!("{}'s team", user_name), "title": "", "region": null, "page_count": 0, "favicon": null, "base": base});
        let user_data = json!({"flag": flag, "name": user_name, "title": "", "region": null, "page_count": 0, "favicon": null});
        self.upsert_item("users", &team_id, "team", team_data, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        self.upsert_item("users", user_address, "user", user_data, None, Some(user_address), Some(&team_id), None, None, None, None).await?;
        Ok(())
    }

    pub async fn get_all_items(&self, table_name: &str, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<TradeDocument>> {
        let table = self.conn.open_table(table_name).execute().await?;
        let mut q = table.query();
        if let Some(f) = filter { q = q.only_if(f); }
        
        // 데이터를 모두 메모리에 올려 정렬한 뒤 안전하게 Slice하여 반환하도록 limit/offset을 제거합니다.
        let results = q.execute().await?.try_collect::<Vec<_>>().await?;
        let mut docs = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let froms = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let tos = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let ccs = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            let bccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
            let refs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
            let statuses = batch.column(8).as_any().downcast_ref::<Int32Array>().unwrap();
            let amounts = batch.column(9).as_any().downcast_ref::<Float32Array>().unwrap();
            let texts = batch.column(11).as_any().downcast_ref::<StringArray>().unwrap();
            let masked_texts = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();
            let jsons = batch.column(13).as_any().downcast_ref::<StringArray>().unwrap();
            let digests = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
            let createds = batch.column(14).as_any().downcast_ref::<Int64Array>().unwrap();
            let updateds = batch.column(15).as_any().downcast_ref::<Int64Array>().unwrap();
            let modes = batch.column(16).as_any().downcast_ref::<StringArray>().unwrap(); 
            let maskeds = batch.column(17).as_any().downcast_ref::<BooleanArray>().unwrap();
            
            for i in 0..batch.num_rows() {
                docs.push(TradeDocument { 
                    id: ids.value(i).to_string(), r#type: types.value(i).to_string(), 
                    from: froms.value(i).to_string(), to: tos.value(i).to_string(),
                    cc: ccs.value(i).to_string(), bcc: bccs.value(i).to_string(),
                    r#ref: refs.value(i).to_string(),
                    text: texts.value(i).to_string(), masked_text: masked_texts.value(i).to_string(), json_data: jsons.value(i).to_string(),
                    digest: digests.value(i).to_string(), total_amount: amounts.value(i),
                    status: statuses.value(i).to_string(), 
                    created_at_ts: createds.value(i), 
                    updated_at_ts: updateds.value(i),
                    mode: modes.value(i).to_string(), 
                    is_masked: maskeds.is_valid(i) && maskeds.value(i),
                    ..Default::default() 
                });
            }
        }
        docs.sort_by_key(|d| std::cmp::Reverse(d.created_at_ts));
        
        
        let start = offset.min(docs.len());
        let end = (start + limit).min(docs.len());
        Ok(docs[start..end].to_vec())
    }

    pub async fn get_item_by_id(&self, table_name: &str, id: &str) -> Result<Option<TradeDocument>> {
        let table = self.conn.open_table(table_name).execute().await?;
        let results = table.query().only_if(format!("id = '{}'", id)).limit(1).execute().await?.try_collect::<Vec<_>>().await?;
        if results.is_empty() || results[0].num_rows() == 0 { return Ok(None); }
        let batch = &results[0];
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let froms = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let tos = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let ccs = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let bccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        let refs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
        let statuses = batch.column(8).as_any().downcast_ref::<Int32Array>().unwrap();
        let amounts = batch.column(9).as_any().downcast_ref::<Float32Array>().unwrap();
        let texts = batch.column(11).as_any().downcast_ref::<StringArray>().unwrap();
        let masked_texts = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();
        let jsons = batch.column(13).as_any().downcast_ref::<StringArray>().unwrap();
        let digests = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
        let createds = batch.column(14).as_any().downcast_ref::<Int64Array>().unwrap();
        let updateds = batch.column(15).as_any().downcast_ref::<Int64Array>().unwrap();
        let modes = batch.column(16).as_any().downcast_ref::<StringArray>().unwrap(); 
        let maskeds = batch.column(17).as_any().downcast_ref::<BooleanArray>().unwrap();

        Ok(Some(TradeDocument { 
            id: ids.value(0).to_string(), r#type: types.value(0).to_string(), 
            from: froms.value(0).to_string(), to: tos.value(0).to_string(),
            cc: ccs.value(0).to_string(), bcc: bccs.value(0).to_string(),
            r#ref: refs.value(0).to_string(),
            text: texts.value(0).to_string(), masked_text: masked_texts.value(0).to_string(), json_data: jsons.value(0).to_string(), 
            digest: digests.value(0).to_string(), total_amount: amounts.value(0),
            status: statuses.value(0).to_string(), 
            created_at_ts: createds.value(0), 
            updated_at_ts: updateds.value(0),
            mode: modes.value(0).to_string(), 
            is_masked: maskeds.is_valid(0) && maskeds.value(0),
            ..Default::default() 
        }))
    }
    
    pub async fn search_items(&self, table_name: &str, query_text: &str, query_vec: Vec<f32>, limit: usize, offset: usize, filter: Option<String>, use_fts: bool) -> Result<Vec<(String, String, f32)>> {
         // 🌟 [CRITICAL FIX] "items"로 하드코딩된 라우팅을 해제하고, 요청된 실제 테이블(sales, event 등)을 100% 반영합니다.
         let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
         let table = self.conn.open_table(target).execute().await?;
         
         // 🌟 3개의 트랙에서 찾은 문서 ID를 Key로 하여, 점수를 누적(Stacking)할 HashMap
         let mut combined = std::collections::HashMap::new();
         let fetch_limit = limit + offset;

         // =======================================================
         // 🌟 [Track 1] Column Matching (SQL Filter)
         // =======================================================
         // LLM이 뽑아낸 속성 조건(예: amount <= 5000) 만 가장 높은 가중치(+3.0) 를 받습니다.
         //
         // 🌟 [SCOPE-ONLY GUARD] convert_conditions_to_sql 은 조건이 하나도 없어도
         //    항상 `type = '...'` 를 넣고, 호출부가 `mode = '...'` 를 붙입니다.
         //    따라서 filter 가 None 이 되는 경우가 없어, 기존 코드는 그 타입의
         //    '모든 행' 에 +3.0 을 상납했습니다.
         //    (log2.txt 실측: 조건 0개인데 결과가 4.0 / 3.999 / 3.998 / 3.997 / 3.0
         //     = 3.0 blanket + 벡터 랭크. FTS 는 한 건도 매칭되지 않았음)
         //    이 상태에서는 의미 신호(FTS·벡터·청크)가 전부 무력화됩니다.
         //    type / mode 는 '스코프' 이지 '조건' 이 아니므로 가산점 대상에서 제외합니다.
         let has_real_condition = match filter.as_ref() {
             None => false,
             Some(f) => {
                 // type / mode 술어와 괄호·AND 를 구조적으로 제거한 뒤 잔여 술어가 있는지 확인합니다.
                 let mut residue = String::new();
                 for clause in f.split(" AND ") {
                     let c = clause.trim().trim_start_matches('(').trim_end_matches(')').trim();
                     if c.is_empty() { continue; }
                     let lower = c.to_lowercase();
                     if lower.starts_with("type ") || lower.starts_with("type=") { continue; }
                     if lower.starts_with("mode ") || lower.starts_with("mode=") { continue; }
                     residue.push_str(c);
                 }
                 !residue.trim().is_empty()
             }
         };

         if has_real_condition {
             if let Some(ref f) = filter {
                 let q = table.query().only_if(f);
                 if let Ok(res) = q.limit(fetch_limit).execute().await {
                     if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                         for b in batches {
                             let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                             let txs = b.column(13).as_any().downcast_ref::<StringArray>().unwrap();
                             for i in 0..b.num_rows() {
                                 combined.insert(ids.value(i).to_string(), (txs.value(i).to_string(), 3.0));
                             }
                         }
                     }
                 }
             }
         } else if let Some(ref f) = filter {
             // 실질 조건이 없으면 스코프 필터로만 사용하고 가산점은 주지 않습니다.
             // (Track 2 / Track 3 이 동일 필터를 체이닝하므로 보안 스코프는 그대로 유지됩니다)
             let _ = f;
         }

         // =======================================================
         // 🌟 [Track 2] Native Full Text Search (Tantivy 역인덱스)
         // =======================================================
         // 전체 본문(text, data)에서 단어를 찾는 진짜 FTS 엔진을 단독 실행. (가중치 +2.0)
         if !query_text.is_empty() {
             let mut q = table.query();
             let has_fts_index = target == "items"; // 🌟 [CRITICAL FIX] FTS 인덱스는 items 테이블에만 생성되어 있으므로 교차 검색 시 에러 방지
             
             if use_fts && has_fts_index {
                 // LanceDB Native FTS 구문 (Tantivy 엔진)
                 let fts_query_str = query_text
                     .split_whitespace()
                     .map(|w| format!("\"{}\"", w.replace("\"", "\\\"")))
                     .collect::<Vec<_>>()
                     .join(" ");
                 q = q.full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(fts_query_str));
                 
                 // [보안 필수] 타 부서/팀 데이터를 긁어오지 못하도록 기본 필터 체이닝
                 if let Some(ref f) = filter { q = q.only_if(f); } 
             } else {
                 // 타이핑 중(Live Search)일 때 미완성 단어를 잡기 위한 ILIKE Fallback (또는 타 도메인 검색 시)
                 let sql_clean = query_text.replace("'", "''");
                 let words: Vec<&str> = sql_clean.split_whitespace().collect();
                 let mut ilike_conditions = Vec::new();
                 for w in words {
                     // 🌟 숫자나 코드가 다른 필드(예: 날짜 2015-03-14)에 매칭되는 것을 줄이기 위한 안전망 적용
                     ilike_conditions.push(format!("(masked_text ILIKE '%{}%' OR text ILIKE '%{}%' OR data ILIKE '%{}%')", w, w, w));
                 }
                 let text_filter = ilike_conditions.join(" AND ");
                 
                 let final_filter = if let Some(ref f) = filter {
                     if text_filter.is_empty() { f.to_string() } else { format!("({}) AND ({})", f, text_filter) }
                 } else { text_filter };
                 
                 if !final_filter.is_empty() { q = q.only_if(final_filter); }
             }
             
             if let Ok(res) = q.limit(fetch_limit).execute().await {
                if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                    for b in batches {
                        let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                        let txs = b.column(13).as_any().downcast_ref::<StringArray>().unwrap();
                        for i in 0..b.num_rows() {
                            let id = ids.value(i).to_string();
                            // 기존에 [Track 1]에서 찾은 문서라면 점수를 누적(+2.0), 아니면 새로 삽입
                            if let Some((_, s)) = combined.get_mut(&id) { *s += 2.0; }
                            else { combined.insert(id, (txs.value(i).to_string(), 2.0)); }
                        }
                    }
                }
             }
         }
         
         // =======================================================
         // 🌟 [Track 3] Vector Search (시맨틱 의미 기반 검색)
         // =======================================================
         // 단어가 달라도 문맥적 의미가 통하는 문서를 찾아 타이브레이커 점수를 가산합니다. (가중치 +1.0 미만)
         let is_empty_vec = query_vec.iter().all(|&x| x == 0.0);
         
         if !is_empty_vec {
             let mut vq = table.query();
             if let Some(ref f) = filter { vq = vq.only_if(f); } // 보안 스코프 유지
             
             if let Ok(vq_with_vector) = vq.limit(fetch_limit).nearest_to(query_vec) {
                 if let Ok(vres) = vq_with_vector.execute().await {
                     if let Ok(batches) = vres.try_collect::<Vec<_>>().await {
                         let mut rank = 0;
                         for b in batches {
                             let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap(); 
                             let txs = b.column(13).as_any().downcast_ref::<StringArray>().unwrap();
                             for i in 0..b.num_rows() {
                                 let id = ids.value(i).to_string();
                                 // 벡터 거리(Rank)에 따라 미세하게 점수를 차등 지급하여 완벽한 정렬 유도
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
         // 🌟 최종 결과 도출 (합산 점수 내림차순 정렬)
         // =======================================================
         let mut final_list: Vec<_> = combined.into_iter().map(|(id, (txt, s))| (id, txt, s)).collect();
         final_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
         
         let start = offset.min(final_list.len());
         let end = (start + limit).min(final_list.len());
         let result_slice = final_list[start..end].to_vec();

         // 🌟 [CRITICAL FIX] 내부 N:N 연관 검색(Cross Reference) 시 0.0 벡터가 들어오므로, 이를 감지하여 터미널 스팸(도배) 출력을 원천 차단합니다!
         // (컴파일 에러 해결: query_vec이 이미 이동(Moved)되었으므로 상단에서 미리 계산해둔 is_empty_vec을 재사용합니다)
         if !is_empty_vec {
             let json_log = serde_json::json!({
                 "target_table": target,
                 "query_text": query_text,
                 "filter": filter,
                 "use_fts": use_fts,
                 "total_found": final_list.len(),
                 "returned": result_slice.len(),
                 "results": result_slice.iter().map(|(id, text, score)| {
                     let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!(text));
                     serde_json::json!({
                         "id": id,
                         "text": parsed_text,
                         "score": score
                     })
                 }).collect::<Vec<_>>()
             });
             println!("\n=======================================");
             println!("[STORE] 🔎 3-Track Hybrid Search Results (Table: {}):", target);
             println!("{}", serde_json::to_string_pretty(&json_log).unwrap_or_default());
             println!("=======================================\n");
         }

         Ok(result_slice)
    }

    pub async fn find_item_by_property(&self, table_name: &str, property: &str, value: &Value) -> Result<Option<(String, Value)>> {
        
        let target = match table_name {
            "sales" | "goods" | "order" => "sales",
            "tracking" | "receiving" | "shipping" => "tracking",
            "event" | "coupon" => "event",
            "member" | "team" | "user" => "users",
            "talk" | "prompt" | "ai_search" => "talks",
            "pages" => "pages",
            "items" => "items",
            t if t.starts_with("commerce_") => &t[9..],
            _ => "items"
        };
        
        let table = self.conn.open_table(target).execute().await?;
        
        
        let results = table.query().execute().await?.try_collect::<Vec<_>>().await?;
        let target_str = match value { Value::String(s) => s.clone(), Value::Number(n) => n.to_string(), _ => value.to_string() };
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(13).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..batch.num_rows() {
                let json_str = datas.value(i);
                if let Ok(data) = serde_json::from_str::<Value>(json_str) {
                    if let Some(f_val) = data.get(property) {
                        let f_val_str = match f_val { Value::String(s) => s.clone(), Value::Number(n) => n.to_string(), _ => f_val.to_string() };
                        if f_val_str == target_str { return Ok(Some((ids.value(i).to_string(), data))); }
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

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TradeDocument {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub from: String,
    pub to: String,
    pub cc: String,
    pub bcc: String,
    #[serde(rename = "ref")]
    pub r#ref: String,
    pub text: String,
    pub masked_text: String,
    pub json_data: String,
    pub digest: String,
    pub vector: Vec<f32>,
    #[serde(rename = "created_at")]
    pub created_at_ts: i64, 
    #[serde(rename = "updated_at")]
    pub updated_at_ts: i64,
    pub mode: String, 
    pub is_masked: bool,
    pub item_descriptions: Vec<String>,
    pub item_hs_codes: Vec<String>,
    pub item_sku_numbers: Vec<String>,
    pub container_numbers: Vec<String>,
    pub seal_numbers: Vec<String>,
    pub related_refs: Vec<String>,
    pub transaction_group: Option<String>,
    pub link_reason: Option<String>,
    pub doc_number: String,
    pub status: String, 
    pub issue_date: String,
    pub reference_export: String,
    pub reference_buyer: String,
    pub reference_carrier: String, 
    pub expiry_date: String, 
    pub bl_type: String,
    pub name: String,
    pub supplier_name: String,
    pub supplier_address: String,
    pub supplier_tax_id: String,
    pub buyer_name: String,
    pub buyer_address: String,
    pub buyer_tax_id: String,
    pub notify_party_name: String,
    pub issuer_name: String,
    pub vessel: String,
    pub voyage_number: String,
    pub pol: String,
    pub pod: String,
    pub place_receipt: String,
    pub place_delivery: String,
    pub transport_mode: String,
    pub departure_date: String,
    pub arrival_date: String,
    pub incoterms: String,
    pub incoterms_place: String,
    pub payment_terms: String,
    pub freight_payment_term: String,
    pub lc_tenor: String,
    pub origin_criterion: String,
    pub currency: String,
    pub total_amount: f32,
    pub subtotal_amount: f32,
    pub tax_amount: f32,
    pub freight_amount: f32,
    pub insurance_amount: f32,
    pub local_charges: f32,
    pub package_count: f32,
    pub package_unit: String,
    pub weight_gross: f32,
    pub weight_net: f32,
    pub volume: f32,
    pub marks_numbers: String,
}