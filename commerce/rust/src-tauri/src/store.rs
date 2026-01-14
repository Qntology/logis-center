use anyhow::Result;
use lancedb::{Connection, Table, connect};
use lancedb::query::{ExecutableQuery, QueryBase};
use arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, FixedSizeListArray, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use futures::TryStreamExt;

const DB_URI: &str = "data/lancedb";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub id: String,
    pub r#type: String,
    pub from_source: String, // 'from' is reserved
    pub to_dest: String,     // 'to' is reserved
    pub cc: String,
    pub bcc: String,
    pub ref_id: String,
    pub data_json: String,   // Store JSON as string
    pub created_at: i64,
    pub updated_at: i64,
    pub status: String,      // 'pending', 'processing', 'done', 'error'
}

#[derive(Clone)]
pub struct VectorStore {
    conn: Connection,
    base_path: String, // Store base path for settings file
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

    pub fn get_config_path(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.base_path).join("settings.json")
    }

    pub fn load_config(&self) -> AppConfig {
        let path = self.get_config_path();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(config) = serde_json::from_str(&content) {
                    return config;
                }
            }
        }
        AppConfig::default()
    }

    pub fn save_config(&self, config: &AppConfig) -> Result<()> {
        let path = self.get_config_path();
        let json = serde_json::to_string_pretty(config)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Task 테이블이 없으면 생성합니다.
    pub async fn init_task_table(&self) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("from_source", DataType::Utf8, false),
            Field::new("to_dest", DataType::Utf8, false),
            Field::new("cc", DataType::Utf8, false),
            Field::new("bcc", DataType::Utf8, false),
            Field::new("ref_id", DataType::Utf8, false),
            Field::new("data_json", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            Field::new("status", DataType::Utf8, false),
        ]));

        self.conn.create_table("tasks", RecordBatch::new_empty(schema))
            .if_not_exists()
            .execute()
            .await?;
            
        Ok(())
    }

    pub async fn add_task(&self, task: Task) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let batch = RecordBatch::try_new(
            table.schema().await?,
            vec![
                Arc::new(StringArray::from(vec![task.id])),
                Arc::new(StringArray::from(vec![task.r#type])),
                Arc::new(StringArray::from(vec![task.from_source])),
                Arc::new(StringArray::from(vec![task.to_dest])),
                Arc::new(StringArray::from(vec![task.cc])),
                Arc::new(StringArray::from(vec![task.bcc])),
                Arc::new(StringArray::from(vec![task.ref_id])),
                Arc::new(StringArray::from(vec![task.data_json])),
                Arc::new(Int64Array::from(vec![task.created_at])),
                Arc::new(Int64Array::from(vec![task.updated_at])),
                Arc::new(StringArray::from(vec![task.status])),
            ],
        )?;

        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    /// 대기 중인 작업을 가져옵니다. (created_at 오름차순)
    pub async fn get_pending_tasks(&self, limit: usize) -> Result<Vec<Task>> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let results = table.query()
            .filter("status = 'pending'")
            .limit(limit)
            // .order_by("created_at", true) // LanceDB ordering support check needed, manual sort for now
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

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
            let createds = batch.column(8).as_any().downcast_ref::<Int64Array>().unwrap();
            let updateds = batch.column(9).as_any().downcast_ref::<Int64Array>().unwrap();
            let statuses = batch.column(10).as_any().downcast_ref::<StringArray>().unwrap();

            for i in 0..batch.num_rows() {
                tasks.push(Task {
                    id: ids.value(i).to_string(),
                    r#type: types.value(i).to_string(),
                    from_source: froms.value(i).to_string(),
                    to_dest: tos.value(i).to_string(),
                    cc: ccs.value(i).to_string(),
                    bcc: bccs.value(i).to_string(),
                    ref_id: refs.value(i).to_string(),
                    data_json: datas.value(i).to_string(),
                    created_at: createds.value(i),
                    updated_at: updateds.value(i),
                    status: statuses.value(i).to_string(),
                });
            }
        }
        
        // Manual sort by created_at since basic SQL support is limited
        tasks.sort_by_key(|t| t.created_at);
        
        Ok(tasks)
    }

    pub async fn update_task_status(&self, id: &str, status: &str) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        // LanceDB doesn't support direct update yet easily.
        // We delete the old row and insert a new one with updated status.
        // First, fetch the existing task data.
        let results = table.query()
            .filter(format!("id = '{}'", id))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        if results.is_empty() {
             return Ok(()); // Task not found
        }
        
        // Extract data
        let batch = &results[0];
        if batch.num_rows() == 0 { return Ok(()); }
        
        let i = 0;
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        // ... (extract other fields) ...
        
        // Simplified: Just delete for now to simulate "Processing Done" 
        // In a real app, we would reconstruct the Task struct and re-insert with new status.
        // But for queue processing, removing 'done' tasks is also a valid strategy to keep table small.
        
        if status == "done" {
            table.delete(&format!("id = '{}'", id)).await?;
        }
        
        Ok(())
    }
    
    // --- Commerce Items (Sales, Goods, etc.) ---
    
    pub async fn init_commerce_table(&self) -> Result<()> {
        let item_field = Field::new("item", DataType::Float32, true);
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(item_field), 768), true),
            Field::new("text", DataType::Utf8, false), // Searchable text
            Field::new("data_json", DataType::Utf8, false), // Full data
            Field::new("updated_at", DataType::Int64, false),
        ]));

        self.conn.create_table("commerce_items", RecordBatch::new_empty(schema))
            .if_not_exists()
            .execute()
            .await?;
        Ok(())
    }
    
    pub async fn upsert_item(&self, id: &str, type_: &str, data: Value, vector: Option<Vec<f32>>) -> Result<()> {
         let table = self.conn.open_table("commerce_items").execute().await?;
         
         // 1. Delete existing if any (LanceDB upsert workaround)
         let _ = table.delete(&format!("id = '{}'", id)).await;
         
         let json_str = data.to_string();
         let text_content = data.get("text").and_then(|s| s.as_str()).unwrap_or("").to_string();
         let vec_data = vector.unwrap_or(vec![0.0; 768]); // Default zero vector
         let now = chrono::Utc::now().timestamp_millis();
         
         // 2. Insert new
         // Construct Arrow Arrays manually
         let id_array = StringArray::from(vec![id]);
         let type_array = StringArray::from(vec![type_]);
         let text_array = StringArray::from(vec![text_content]);
         let json_array = StringArray::from(vec![json_str]);
         let updated_array = Int64Array::from(vec![now]);
         
         // FixedSizeListArray construction is tricky
         let values_builder = Float32Array::from(vec_data);
         let list_array = FixedSizeListArray::try_new_from_values(values_builder, 768)?;

         let batch = RecordBatch::try_new(
            table.schema().await?,
            vec![
                Arc::new(id_array),
                Arc::new(type_array),
                Arc::new(list_array),
                Arc::new(text_array),
                Arc::new(json_array),
                Arc::new(updated_array),
            ],
         )?;
         
         table.add(vec![batch]).execute().await?;
         
         Ok(())
    }
    
    pub async fn search_items(&self, query_vec: Vec<f32>, limit: usize) -> Result<Vec<(String, String, f32)>> {
         let table = self.conn.open_table("commerce_items").execute().await?;
         
         // Vector search not fully exposed in this simple wrapper yet without query builder support for vectors
         // For now, we return empty or implement full scan if needed.
         // LanceDB Rust SDK supports vector search via .search()
         
         let results = table.query()
            .limit(limit)
            .nearest_to(&query_vec)?
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
            
         let mut items = Vec::new();
         for batch in results {
             let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
             let texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
             // Distance is usually added as a column named "_distance"
             // Let's assume just returning content for now
             
             for i in 0..batch.num_rows() {
                 items.push((ids.value(i).to_string(), texts.value(i).to_string(), 0.0));
             }
         }
         
         Ok(items)
    }

    /// Finds a single item where a specific JSON property matches a value.
    /// This simulates "SELECT * FROM table WHERE column = value" for the Relay logic.
    /// Currently scans 'commerce_items'. In a real SQL DB, this would be an index lookup.
    pub async fn find_item_by_property(&self, property: &str, value: &Value) -> Result<Option<(String, Value)>> {
        let table = self.conn.open_table("commerce_items").execute().await?;
        
        // LanceDB SQL filtering on JSON strings is limited. 
        // Ideally, we should promote key columns (tracking_number, order_id) to top-level columns.
        // For now, we fetch recent items and filter in memory (NOT EFFICIENT for large datasets, but works for prototype).
        // A better approach for LanceDB: Use full-text search or separate index tables.
        
        let results = table.query()
            .limit(1000) // Scan limit
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let target_val_str = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            _ => value.to_string(),
        };

        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let jsons = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap(); // data_json is col 4 based on init_commerce_table schema

            for i in 0..batch.num_rows() {
                let json_str = jsons.value(i);
                if let Ok(data) = serde_json::from_str::<Value>(json_str) {
                    if let Some(field_val) = data.get(property) {
                        let field_val_str = match field_val {
                            Value::String(s) => s.clone(),
                            Value::Number(n) => n.to_string(),
                            _ => field_val.to_string(),
                        };
                        
                        if field_val_str == target_val_str {
                            return Ok(Some((ids.value(i).to_string(), data)));
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}

// Temporary TradeDocument Struct for compatibility (kept as is)
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TradeDocument {
    pub uuid: String,
    pub doc_type: String,
    pub text: String,
    pub json_data: String,
    pub vector: Vec<f32>,
    pub item_descriptions: Vec<String>, 
    pub item_hs_codes: Vec<String>,
    pub item_sku_numbers: Vec<String>,
    pub container_numbers: Vec<String>,
    pub seal_numbers: Vec<String>,
    pub related_refs: Vec<String>,
    pub transaction_group: Option<String>,
    pub link_reason: Option<String>,
    pub doc_number: String,
    pub doc_status: String, 
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
