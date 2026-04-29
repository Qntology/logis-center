use anyhow::Result;
use lancedb::{Connection, connect};
use lancedb::query::{ExecutableQuery, QueryBase};
use arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, FixedSizeListArray, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
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
        let uri = format!("{}/{}", base_path, DB_URI);
        let conn = connect(&uri).execute().await?;
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

        let existing = self.conn.table_names().execute().await?;
        if existing.contains(&"tasks".to_string()) {
            let table = self.conn.open_table("tasks").execute().await?;
            let current_schema = table.schema().await?;
            // [MIGRATION] Ensure 'ref' field exists and 'status' is Int32
            let has_ref = current_schema.field_with_name("ref").is_ok();
            let status_is_int = if let Ok(field) = current_schema.field_with_name("status") {
                field.data_type() == &DataType::Int32
            } else { false };

            if !has_ref || !status_is_int {
                println!("[Store] tasks table schema mismatch. Dropping for recreation.");
                let _ = self.conn.drop_table("tasks", &[]).await;
            }
        }
        let existing = self.conn.table_names().execute().await?;
        if !existing.contains(&"tasks".to_string()) {
            let _ = self.conn.create_table("tasks", RecordBatchIterator::new(vec![], task_schema)).execute().await;
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

        if existing.contains(&"talks".to_string()) {
            let table = self.conn.open_table("talks").execute().await?;
            let current_schema = table.schema().await?;
            // [MIGRATION] Check for 'text' field to see if it's the old schema
            let needs_recreate = current_schema.field_with_name("text").is_err();
            if needs_recreate {
                println!("[Store] talks table schema outdated (missing 'text'). Dropping for migration.");
                let _ = self.conn.drop_table("talks", &[]).await;
            }
        }
        let existing = self.conn.table_names().execute().await?;
        if !existing.contains(&"talks".to_string()) {
            let _ = self.conn.create_table("talks", RecordBatchIterator::new(vec![], msg_schema)).execute().await;
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
                Arc::new(arrow_array::Int32Array::from(vec![status.unwrap_or(0)])),
                Arc::new(Int64Array::from(vec![now])),
                Arc::new(Int64Array::from(vec![0])), // updated_at
            ],
        )?;
        table.add(RecordBatchIterator::new(vec![Ok(batch)], schema)).execute().await?;
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
            let statuses = batch.column(11).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
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
                Arc::new(arrow_array::Int32Array::from(vec![task.status])),
            ],
        )?;
        table.add(RecordBatchIterator::new(vec![Ok(batch)], schema)).execute().await?;
        Ok(())
    }

    pub async fn get_pending_tasks(&self, limit: usize) -> Result<Vec<Task>> {
        let table = self.conn.open_table("tasks").execute().await?;
        // 🌟 [CRITICAL FIX] UI 복구를 위해 ai_search를 포함한 모든 대기열을 반환합니다. (스케줄러 필터링은 scheduler.rs에서 수행)
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
            let sts = batch.column(10).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
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

    // 🌟 [CRITICAL FIX] 상태가 1인 진행 중인 작업만 정확히 가져오는 전용 함수를 추가합니다.
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
            let sts = batch.column(10).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
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

    // 🌟 [정리] 불필요한 구형 로직을 삭제하고 통합합니다.
    pub async fn cleanup_unfinished_tasks_on_startup(&self) -> Result<()> {
        let tasks_table = self.conn.open_table("tasks").execute().await?;
        let talks_table = self.conn.open_table("talks").execute().await?;

        println!("[Store] Initializing zombie task recovery process...");

        // 🌟 1. [진행 중 사살] 앱이 죽었을 때 연산 중이던(1) 작업은 무조건 중단(2) 처리합니다.
        // 🌟 2. [검색 큐 사살] ai_search(10)는 프론트엔드 메모리 큐가 증발했으므로 대기 중이라도 사살합니다.
        let _ = tasks_table.update()
            .only_if("status = 1 OR (status = 10 AND type = 'ai_search')")
            .column("status", "2") 
            .execute()
            .await;

        // 🌟 3. [채팅 메시지 동기화] UI 말풍선도 동일하게 업데이트합니다.
        let _ = talks_table.update()
            .only_if("status = 1 OR (status = 10 AND task_id LIKE 'search_%')")
            .column("status", "2")
            .column("text", "'Task was interrupted due to app restart.'")
            .execute()
            .await;

        println!("[Store] CRITICAL: Zombie recovery complete. (Status 1 and search_tasks set to STOPPED)");
        Ok(())
    }

    pub async fn delete_item(&self, table_name: &str, id: &str) -> Result<()> {
        let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
        let table = self.conn.open_table(target).execute().await?;
        table.delete(&format!("id = '{}'", id)).await?;
        Ok(())
    }

    pub async fn delete_items(&self, table_name: &str, ids: Vec<String>) -> Result<()> {
        let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
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
            Field::new("vector", DataType::FixedSizeList(Arc::new(item_field), 768), true),
            Field::new("text", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false), 
            Field::new("updated_at", DataType::Int64, false),
        ]));
        let existing = self.conn.table_names().execute().await?;
        for name in tables {
            if existing.contains(&name.to_string()) {
                let table = self.conn.open_table(name).execute().await?;
                let current_schema = table.schema().await?;
                
                // [FIX] Check for 'ref' column existence and 'status' type to trigger migration
                let has_ref = current_schema.field_with_name("ref").is_ok();
                let status_is_int = if let Ok(field) = current_schema.field_with_name("status") {
                    field.data_type() == &DataType::Int32
                } else { false };

                if !has_ref || !status_is_int {
                    println!("[Store] Schema mismatch for {}. Dropping and recreating...", name);
                    let _ = self.conn.drop_table(name, &[]).await;
                } else {
                    continue;
                }
            }
            let _table = self.conn.create_table(name, RecordBatchIterator::new(vec![], schema.clone())).execute().await?;
        }
        Ok(())
    }
    
    pub async fn upsert_item(
        &self, table_name: &str, id: &str, type_: &str, data_val: Value, vector: Option<Vec<f32>>,
        from: Option<&str>, to: Option<&str>, cc: Option<&str>, bcc: Option<&str>, r#ref: Option<&str>, digest: Option<&str>
    ) -> Result<()> {
         println!("[DEBUG] store.upsert_item - Table: {}, ID: {}, Type: {}, Data: {}", table_name, id, type_, data_val);
         let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
         let table = self.conn.open_table(target).execute().await?;
         
         // [SAFETY-FIX] Ensure ID is never empty
         let mut final_id = id.to_string();
         if final_id.is_empty() {
             final_id = data_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
         }
         // Still empty? Use a fallback UUID to prevent database corruption/unreachable rows
         if final_id.is_empty() {
             final_id = uuid::Uuid::new_v4().to_string();
             println!("[Store] Warning: upsert_item called with empty ID. Generated fallback: {}", final_id);
         }

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
         let status = data_val.get("status").and_then(|v| v.as_str()).map(|s| crate::logic::parse_status(s)).unwrap_or(0);
         let amount = data_val.get("total_amount").or_else(|| data_val.get("sale_price")).or_else(|| data_val.get("supply_price")).or_else(|| data_val.get("price")).or_else(|| data_val.get("shipping_fee")).or_else(|| data_val.get("discount")).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))).unwrap_or(0.0) as f32;
         let doc_date_str = data_val.get("order_date").or_else(|| data_val.get("registration_date")).or_else(|| data_val.get("release_date")).or_else(|| data_val.get("manufacture_date")).or_else(|| data_val.get("shipping_date")).or_else(|| data_val.get("started_at")).or_else(|| data_val.get("expired_at")).or_else(|| data_val.get("payment_date")).and_then(|v| v.as_str()).unwrap_or("");
         let now_ts = chrono::Utc::now().timestamp_millis();
         let created_at = if !doc_date_str.is_empty() {
             chrono::DateTime::parse_from_rfc3339(doc_date_str).map(|dt| dt.timestamp_millis()).unwrap_or_else(|_| chrono::NaiveDateTime::parse_from_str(doc_date_str, "%Y-%m-%dT%H:%M:%S").map(|dt| dt.and_utc().timestamp_millis()).unwrap_or(now_ts))
         } else { now_ts };
         let schema = table.schema().await?;
         let values_builder = Float32Array::from(vector.unwrap_or(vec![0.0; 768]));
         let list_field = Field::new("item", DataType::Float32, true);
         let list_array = FixedSizeListArray::try_new(Arc::new(list_field), 768, Arc::new(values_builder), None)?;
         let batch = RecordBatch::try_new(schema.clone(), vec![
                Arc::new(StringArray::from(vec![final_id])), Arc::new(StringArray::from(vec![type_])),
                Arc::new(StringArray::from(vec![from.unwrap_or("")])), Arc::new(StringArray::from(vec![to.unwrap_or("")])),
                Arc::new(StringArray::from(vec![cc.unwrap_or("")])), Arc::new(StringArray::from(vec![bcc.unwrap_or("")])),
                Arc::new(StringArray::from(vec![r#ref.unwrap_or("")])), Arc::new(StringArray::from(vec![digest.unwrap_or("")])),
                Arc::new(arrow_array::Int32Array::from(vec![status])), Arc::new(arrow_array::Float32Array::from(vec![amount])),
                Arc::new(list_array), Arc::new(StringArray::from(vec![text_content])), Arc::new(StringArray::from(vec![json_str])),
                Arc::new(Int64Array::from(vec![created_at])), Arc::new(Int64Array::from(vec![now_ts])),
         ])?;
         table.add(RecordBatchIterator::new(vec![Ok(batch)], schema)).execute().await?;
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
        let results = q.limit(limit).offset(offset).execute().await?.try_collect::<Vec<_>>().await?;
        let mut docs = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let froms = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let tos = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let ccs = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            let bccs = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
            let refs = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
            let statuses = batch.column(8).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
            let amounts = batch.column(9).as_any().downcast_ref::<Float32Array>().unwrap();
            let texts = batch.column(11).as_any().downcast_ref::<StringArray>().unwrap();
            let jsons = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();
            let digests = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
            let createds = batch.column(13).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..batch.num_rows() {
                docs.push(TradeDocument { 
                    id: ids.value(i).to_string(), r#type: types.value(i).to_string(), 
                    from: froms.value(i).to_string(), to: tos.value(i).to_string(),
                    cc: ccs.value(i).to_string(), bcc: bccs.value(i).to_string(),
                    r#ref: refs.value(i).to_string(),
                    text: texts.value(i).to_string(), json_data: jsons.value(i).to_string(),
                    digest: digests.value(i).to_string(), total_amount: amounts.value(i),
                    status: statuses.value(i).to_string(), created_at_ts: createds.value(i), ..Default::default() 
                });
            }
        }
        docs.sort_by_key(|d| std::cmp::Reverse(d.created_at_ts));
        Ok(docs)
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
        let statuses = batch.column(8).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let amounts = batch.column(9).as_any().downcast_ref::<Float32Array>().unwrap();
        let texts = batch.column(11).as_any().downcast_ref::<StringArray>().unwrap();
        let jsons = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();
        let digests = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
        let createds = batch.column(13).as_any().downcast_ref::<Int64Array>().unwrap();
        Ok(Some(TradeDocument { 
            id: ids.value(0).to_string(), r#type: types.value(0).to_string(), 
            from: froms.value(0).to_string(), to: tos.value(0).to_string(),
            cc: ccs.value(0).to_string(), bcc: bccs.value(0).to_string(),
            r#ref: refs.value(0).to_string(),
            text: texts.value(0).to_string(), json_data: jsons.value(0).to_string(), 
            digest: digests.value(0).to_string(), total_amount: amounts.value(0),
            status: statuses.value(0).to_string(), created_at_ts: createds.value(0), ..Default::default() 
        }))
    }
    
    pub async fn search_items(&self, table_name: &str, query_text: &str, query_vec: Vec<f32>, limit: usize, filter: Option<String>) -> Result<Vec<(String, String, f32)>> {
         let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
         let table = self.conn.open_table(target).execute().await?;
         let mut combined = std::collections::HashMap::new();
         if !query_text.is_empty() {
             let clean = query_text.replace("'", "''");
             let mut q = table.query();
             if let Some(ref f) = filter { q = q.only_if(f); }
             if let Ok(res) = q.only_if(format!("text MATCH '{0}' OR data MATCH '{0}'", clean)).limit(limit).execute().await {
                if let Ok(batches) = res.try_collect::<Vec<_>>().await {
                    for b in batches {
                        let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                        let txs = b.column(12).as_any().downcast_ref::<StringArray>().unwrap();
                        for i in 0..b.num_rows() { combined.insert(ids.value(i).to_string(), (txs.value(i).to_string(), 1.0)); }
                    }
                }
             }
         }
         let mut vq = table.query();
         if let Some(ref f) = filter { vq = vq.only_if(f); }
         let vres = vq.limit(limit).nearest_to(query_vec)?.execute().await?.try_collect::<Vec<_>>().await?;
         for b in vres {
             let ids = b.column(0).as_any().downcast_ref::<StringArray>().unwrap(); 
             let txs = b.column(12).as_any().downcast_ref::<StringArray>().unwrap();
             for i in 0..b.num_rows() {
                 let id = ids.value(i).to_string();
                 if let Some((_, s)) = combined.get_mut(&id) { *s += 0.5; } 
                 else { combined.insert(id, (txs.value(i).to_string(), 0.5)); }
             }
         }
         let mut final_list: Vec<_> = combined.into_iter().map(|(id, (txt, s))| (id, txt, s)).collect();
         final_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
         Ok(final_list)
    }

    pub async fn find_item_by_property(&self, table_name: &str, property: &str, value: &Value) -> Result<Option<(String, Value)>> {
        let target = if table_name.starts_with("commerce_") { &table_name[9..] } else if table_name.is_empty() { "items" } else { table_name };
        let table = self.conn.open_table(target).execute().await?;
        let results = table.query().limit(1000).execute().await?.try_collect::<Vec<_>>().await?;
        let target_str = match value { Value::String(s) => s.clone(), Value::Number(n) => n.to_string(), _ => value.to_string() };
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let datas = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();
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
    pub json_data: String,
    pub digest: String,
    pub vector: Vec<f32>,
    pub created_at_ts: i64, 
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