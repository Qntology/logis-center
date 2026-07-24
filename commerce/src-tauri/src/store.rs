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
    
    pub async fn search_items(&self, _table_name: &str, query_text: &str, query_vec: Vec<f32>, limit: usize, offset: usize, filter: Option<String>, use_fts: bool) -> Result<Vec<(String, String, f32)>> {
         
         // 100% 통합 FTS 인덱스가 구축된 "items" 마스터 테이블로만 쿼리를 강제 라우팅하여 중앙 검색을 수행합니다.
         let target = "items";
         let table = self.conn.open_table(target).execute().await?;
         let mut combined = std::collections::HashMap::new();
         
         
         let fetch_limit = limit + offset;

         if !query_text.is_empty() {
             
             let sql_clean = query_text.replace("'", "''");
             let mut q = table.query();
             
             
             // AI Deep Search(엔터/돋보기)일 때는 SDK 내장 full_text_search API를 호출합니다.
             if use_fts {
                 
                 // [CRITICAL FIX] 전체 검색어를 하나의 큰따옴표로 묶으면("베이지 가디건") 정확한 구문(Exact Phrase) 매칭이 되어 검색이 실패합니다.
                 // 띄어쓰기 단위로 쪼개어 각각 큰따옴표로 묶어 다중 N-gram 구문 검색("베이지" "가디건")이 되도록 수정합니다.
                 let fts_query_str = query_text
                     .split_whitespace()
                     .map(|w| format!("\"{}\"", w.replace("\"", "\\\"")))
                     .collect::<Vec<_>>()
                     .join(" ");
                 
                 // 공식 문서에 따른 FTS 전용 메서드 체이닝
                 q = q.full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(fts_query_str));
                 
                 // 추가 필터(status, type 등)가 존재하면 AND 조건으로 체이닝
                 if let Some(ref f) = filter {
                     q = q.only_if(f);
                 }
             } else {
                 // 기존 ILIKE 스캔 로직 유지 + 마스킹 텍스트까지 포함
                 let text_filter = format!("(masked_text ILIKE '%{}%' OR text ILIKE '%{}%' OR data ILIKE '%{}%')", sql_clean, sql_clean, sql_clean);
                 let final_filter = if let Some(ref f) = filter {
                     format!("({}) AND {}", f, text_filter)
                 } else {
                     text_filter
                 };
                 q = q.only_if(final_filter);
             }
             
             if let Ok(res) = q.limit(fetch_limit).execute().await {
                if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                    for b in batches {
                        let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                        let txs = b.column(13).as_any().downcast_ref::<StringArray>().unwrap();
                        for i in 0..b.num_rows() { combined.insert(ids.value(i).to_string(), (txs.value(i).to_string(), 1.0)); }
                    }
                }
             }
         }
         
         // LanceDB가 아직 내용이 없는 빈 껍데기(Draft) 문서들을 완벽 일치(거리 0)로 착각하여 
         // 스크롤 시 무더기로 반환하는 현상을 원천 차단합니다.
         let is_empty_vec = query_vec.iter().all(|&x| x == 0.0);
         
         if !is_empty_vec {
             let mut vq = table.query();
             if let Some(ref f) = filter { vq = vq.only_if(f); }
             let vres = vq.limit(fetch_limit).nearest_to(query_vec)?.execute().await?.try_collect::<Vec<_>>().await?;
             
             
             // HashMap에 담을 때 순서가 뒤섞이는 것을 막기 위해, 순위(rank)에 따라 점수를 미세하게 깎아서 고유 정렬 순서를 보존합니다.
             let mut rank = 0;
             for b in vres {
                 let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap(); 
                 let txs = b.column(13).as_any().downcast_ref::<StringArray>().unwrap();
                 for i in 0..b.num_rows() {
                     let id = ids.value(i).to_string();
                     let vec_score = 0.5 - (rank as f32 * 0.001);
                     if let Some((_, s)) = combined.get_mut(&id) { *s += vec_score; } 
                     else { combined.insert(id, (txs.value(i).to_string(), vec_score)); }
                     rank += 1;
                 }
             }
         }
         let mut final_list: Vec<_> = combined.into_iter().map(|(id, (txt, s))| (id, txt, s)).collect();
         final_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
         
         
         let start = offset.min(final_list.len());
         let end = (start + limit).min(final_list.len());
         let result_slice = final_list[start..end].to_vec();

         // 🌟 [추가] 검색 결과를 JSON 포맷으로 터미널에 로그 출력
         let json_log = serde_json::json!({
             "query_text": query_text,
             "filter": filter,
             "use_fts": use_fts,
             "total_found": final_list.len(),
             "returned": result_slice.len(),
             "results": result_slice.iter().map(|(id, text, score)| {
                 // 🌟 [개선] text가 JSON 문자열이면 예쁘게 객체로 파싱해서 출력하고, 아니면 원래 문자열로 출력
                 let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!(text));
                 
                 serde_json::json!({
                     "id": id,
                     "text": parsed_text,
                     "score": score
                 })
             }).collect::<Vec<_>>()
         });
         println!("\n=======================================");
         println!("[STORE] 🔎 Search Results (JSON):");
         println!("{}", serde_json::to_string_pretty(&json_log).unwrap_or_default());
         println!("=======================================\n");

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
        let tables = vec!["tasks", "talks", "items", "sales", "tracking", "event", "users", "pages"];
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