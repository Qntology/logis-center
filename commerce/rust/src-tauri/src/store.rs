use anyhow::Result;
use lancedb::{Connection, connect};
use lancedb::query::{ExecutableQuery, QueryBase};
use arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, FixedSizeListArray, RecordBatchIterator};
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

        // RecordBatchIterator is required for create_table
        let batches = RecordBatchIterator::new(vec![], schema.clone());
        
        let _ = self.conn.create_table("tasks", batches)
            .execute()
            .await;
            
        Ok(())
    }

    pub async fn add_task(&self, task: Task) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let schema = table.schema().await?;
        
        let batch = RecordBatch::try_new(
            schema.clone(),
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

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        table.add(batches).execute().await?;
        Ok(())
    }

    /// 대기 중인 작업을 가져옵니다.
    pub async fn get_pending_tasks(&self, limit: usize) -> Result<Vec<Task>> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let results = table.query()
            .only_if("status = 'pending'")
            .limit(limit)
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
        
        tasks.sort_by_key(|t| t.created_at);
        Ok(tasks)
    }

    pub async fn update_task_status(&self, id: &str, _status: &str) -> Result<()> {
        let table = self.conn.open_table("tasks").execute().await?;
        
        let results = table.query()
            .only_if(format!("id = '{}'", id))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        if results.is_empty() { return Ok(()); }
        
        // Delete old (Simple update via delete-insert)
        table.delete(&format!("id = '{}'", id)).await?;
        
        // Note: For a proper update, we should re-insert. 
        // But if 'done', we might just keep it deleted or archive it.
        // For now, if status is 'done', we just leave it deleted (queue behavior).
        
        Ok(())
    }
    
    // --- Commerce Items (Sales, Goods, etc.) ---
    
    pub async fn init_all_tables(&self) -> Result<()> {
        let tables = vec![
            "commerce_items", "commerce_sales", "commerce_tracking", "commerce_event", 
            "commerce_users", "commerce_talks"
        ];
        let item_field = Field::new("item", DataType::Float32, true);
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(item_field), 768), true),
            Field::new("text", DataType::Utf8, false),
            Field::new("data_json", DataType::Utf8, false),
            Field::new("updated_at", DataType::Int64, false),
        ]));

        for table_name in tables {
            let batches = RecordBatchIterator::new(vec![], schema.clone());
            let table = self.conn.create_table(table_name, batches)
                .execute()
                .await?;
            
            // Create index on 'text' column for keyword searching
            // Using Index::Auto for best compatibility across library versions
            let _ = table.create_index(&["text"], lancedb::index::Index::Auto).execute().await;
        }
        Ok(())
    }
    
    pub async fn upsert_item(&self, table_name: &str, id: &str, type_: &str, data: Value, vector: Option<Vec<f32>>) -> Result<()> {
         // Default to commerce_items if not specified or specific logic needed
         let target_table = if table_name.is_empty() { "commerce_items" } else { table_name };
         
         let table = self.conn.open_table(target_table).execute().await?;
         let _ = table.delete(&format!("id = '{}'", id)).await;
         
         let json_str = data.to_string();
         
         // [FIXED] Derive text from JSON or fallback, don't force external narrative
         let mut text_content = data.get("text").and_then(|s| s.as_str()).unwrap_or("").to_string();

         if text_content.is_empty() {
             // Generate a simple summary if text is missing
             if let Some(obj) = data.as_object() {
                 let fields: Vec<String> = obj.iter()
                    .filter(|(k, _)| !["type", "detail", "node", "item"].contains(&k.as_str()))
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .take(10)
                    .collect();
                 text_content = fields.join(". ");
             }
         }

         let vec_data = vector.unwrap_or(vec![0.0; 768]);
         let now = chrono::Utc::now().timestamp_millis();
         
         let schema = table.schema().await?;
         
         // Helper to build FixedSizeList
         let values_builder = Float32Array::from(vec_data);
         let list_field = Field::new("item", DataType::Float32, true);
         let list_array = FixedSizeListArray::try_new(Arc::new(list_field), 768, Arc::new(values_builder), None)?;

         let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(StringArray::from(vec![type_])),
                Arc::new(list_array),
                Arc::new(StringArray::from(vec![text_content])),
                Arc::new(StringArray::from(vec![json_str])),
                Arc::new(Int64Array::from(vec![now])),
            ],
         )?;
         
         let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
         table.add(batches).execute().await?;
         
         Ok(())
    }

    pub async fn get_all_items(&self, table_name: &str, limit: usize, offset: usize) -> Result<Vec<TradeDocument>> {
        let table = self.conn.open_table(table_name).execute().await?;
        let results = table.query()
            .limit(limit)
            .offset(offset)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut docs = Vec::new();
        for batch in results {
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let jsons = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

            for i in 0..batch.num_rows() {
                docs.push(TradeDocument {
                    uuid: ids.value(i).to_string(),
                    doc_type: types.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    json_data: jsons.value(i).to_string(),
                    ..Default::default()
                });
            }
        }
        Ok(docs)
    }

    pub async fn get_item_by_id(&self, table_name: &str, id: &str) -> Result<Option<TradeDocument>> {
        let table = self.conn.open_table(table_name).execute().await?;
        let results = table.query()
            .only_if(format!("id = '{}'", id))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        if results.is_empty() { return Ok(None); }
        let batch = &results[0];
        if batch.num_rows() == 0 { return Ok(None); }

        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let types = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let jsons = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

        Ok(Some(TradeDocument {
            uuid: ids.value(0).to_string(),
            doc_type: types.value(0).to_string(),
            text: texts.value(0).to_string(),
            json_data: jsons.value(0).to_string(),
            ..Default::default()
        }))
    }
    
    pub async fn search_items(&self, table_name: &str, query_text: &str, query_vec: Vec<f32>, limit: usize, filter: Option<String>) -> Result<Vec<(String, String, f32)>> {
         let target_table = if table_name.is_empty() { "commerce_items" } else { table_name };
         let table = self.conn.open_table(target_table).execute().await?;
         
         let mut combined_results = std::collections::HashMap::new();

         // 1. FULL TEXT SEARCH (Keyword Match)
         if !query_text.is_empty() {
             let clean_query = query_text.replace("'", "''");
             let mut fts_query = table.query();
             if let Some(ref f) = filter { fts_query = fts_query.only_if(f); }
             
             if let Ok(fts_results) = fts_query
                .only_if(format!("text MATCH '{}'", clean_query))
                .limit(limit)
                .execute()
                .await {
                    if let Ok(batches) = fts_results.try_collect::<Vec<_>>().await {
                        for batch in batches {
                            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                            let texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
                            for i in 0..batch.num_rows() {
                                combined_results.insert(ids.value(i).to_string(), (texts.value(i).to_string(), 1.0));
                            }
                        }
                    }
                }
         }

         // 2. VECTOR SEARCH (Semantic Match)
         let mut vector_query = table.query();
         if let Some(ref f) = filter { vector_query = vector_query.only_if(f); }

         let vector_results = vector_query
            .limit(limit)
            .nearest_to(query_vec)?
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
            
         for batch in vector_results {
             let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
             let texts = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
             for i in 0..batch.num_rows() {
                 let id = ids.value(i).to_string();
                 if let Some((_, score)) = combined_results.get_mut(&id) {
                     *score += 0.5;
                 } else {
                     combined_results.insert(id, (texts.value(i).to_string(), 0.5));
                 }
             }
         }
         
         let mut final_list: Vec<_> = combined_results.into_iter().map(|(id, (txt, score))| (id, txt, score)).collect();
         final_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
         
         Ok(final_list)
    }

    pub async fn find_item_by_property(&self, table_name: &str, property: &str, value: &Value) -> Result<Option<(String, Value)>> {
        let target_table = if table_name.is_empty() { "commerce_items" } else { table_name };
        let table = self.conn.open_table(target_table).execute().await?;
        
        let results = table.query()
            .limit(1000) 
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
            let jsons = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

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

// Full TradeDocument Struct restored for frontend compatibility and detailed schema support.
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
