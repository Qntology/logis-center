mod model;
mod store;
mod automation;
mod parsing;
mod logic;
mod scheduler;

use tauri::{State, Manager, Listener}; // Added Manager
use tokio::sync::Mutex;
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use serde_json::{Value, json};

struct AppState {
    model: Arc<Mutex<Option<LogisModel>>>,
    store: Arc<Mutex<Option<VectorStore>>>,
}

#[tauri::command]
async fn resize_window(app_handle: tauri::AppHandle, width: f64, height: f64) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
    }
}

#[tauri::command]
async fn start_drag(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.start_dragging();
    }
}

#[tauri::command]
async fn move_to_top_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let screen_size = monitor.size(); // PhysicalSize
            let scale_factor = monitor.scale_factor();
            let screen_width = screen_size.width as f64 / scale_factor;
            
            // Get current window size
            if let Ok(factor) = window.scale_factor() {
                if let Ok(size) = window.outer_size() {
                    let win_width = size.width as f64 / factor;
                    let new_x = (screen_width - win_width) / 2.0;
                    
                    let _ = window.set_position(tauri::Position::Logical(tauri::LogicalPosition {
                        x: new_x,
                        y: 0.0,
                    }));
                }
            }
        }
    }
}

#[tauri::command]
async fn launch_browser(
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    automation::run_browser_automation(browser, url, script, app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    let available = automation::get_available_browsers();
    // Priority: Chrome -> Edge -> Firefox
    let target = if available.iter().any(|b| b.name == "chrome") {
        "chrome"
    } else if available.iter().any(|b| b.name == "edge") {
        "edge"
    } else if available.iter().any(|b| b.name == "firefox") {
        "firefox"
    } else {
        return Err("No supported browser found.".to_string());
    };
    
    // Launch with default empty script
    automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_available_browsers() -> Vec<automation::BrowserStatus> {
    automation::get_available_browsers()
}

// --- Helper to Generate Rich Summary (Fully Ported from logic.py) ---
fn generate_rich_summary(doc_type: &str, data: &Value) -> String {
    let type_map = json!({
        "CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List",
        "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit"
    });
    
    let full_type = type_map.get(doc_type).and_then(|s| s.as_str()).unwrap_or(doc_type);
    let mut parts = vec![format!("This is a {} document.", full_type)];

    if let Some(h) = data.get("header") {
        if let Some(no) = h.get("document_number").and_then(|s| s.as_str()) {
            if no != "N/A" && !no.is_empty() {
                parts.push(format!("Document number is {}.", no));
            }
        }
        if let Some(date) = h.get("issue_date").and_then(|s| s.as_str()) {
            if date != "N/A" && !date.is_empty() {
                parts.push(format!("Issued on {}.", date));
            }
        }
    }

    if let Some(p) = data.get("parties") {
        let sup = p.get("supplier_name").and_then(|s| s.as_str());
        let buy = p.get("buyer_name").and_then(|s| s.as_str());
        
        // Clean check
        let has_sup = sup.is_some() && sup.unwrap() != "N/A";
        let has_buy = buy.is_some() && buy.unwrap() != "N/A";

        if has_sup && has_buy {
            parts.push(format!("Transaction involved {} as the supplier/shipper and {} as the buyer/consignee.", sup.unwrap(), buy.unwrap()));
        } else if has_sup {
            parts.push(format!("Supplier/Shipper is {}.", sup.unwrap()));
        } else if has_buy {
            parts.push(format!("Buyer/Consignee is {}.", buy.unwrap()));
        }
    }

    if let Some(f) = data.get("financials") {
        if let Some(amt) = f.get("amount_total") {
             // Handle number or string
             let amt_str = if amt.is_number() { amt.to_string() } else { amt.as_str().unwrap_or("0").to_string() };
             let curr = f.get("currency_code").and_then(|s| s.as_str()).unwrap_or("USD");
             if amt_str != "0" && amt_str != "0.0" {
                 parts.push(format!("Total amount is {} {}.", amt_str, curr));
             }
        }
    }

    if let Some(l) = data.get("logistics") {
        let pol = l.get("location_port_of_loading").and_then(|s| s.as_str());
        let pod = l.get("location_port_of_discharge").and_then(|s| s.as_str());
        
        if let (Some(o), Some(d)) = (pol, pod) {
            if o != "N/A" && d != "N/A" {
                parts.push(format!("Shipped from {} to {}.", o, d));
            }
        }
        
        if let Some(mode) = l.get("transport_mode").and_then(|s| s.as_str()) {
            parts.push(format!("Transport mode is {}.", mode));
        }
    }

    // Items summary (First 5 items)
    if let Some(items) = data.get("line_items").and_then(|v| v.as_array()) {
        let mut item_descs = Vec::new();
        for item in items.iter().take(5) {
            if let Some(d) = item.get("description").and_then(|s| s.as_str()) {
                if d.len() > 3 { item_descs.push(d); }
            }
        }
        if !item_descs.is_empty() {
            parts.push(format!("Contains items: {}.", item_descs.join(", ")));
        }
    }
    
    parts.join(" ")
}

#[tauri::command]
async fn summarize_image(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    image_path: String,
) -> Result<String, String> {
    println!("[INVOKE-01] summarize_image called with path: {}", image_path);

    // 1. Initialize Model (Auto Device)
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        println!("[INIT-01] AppState model is None. Initializing LogisModel (Auto)...");
        match LogisModel::new(None).await {
            Ok(m) => {
                println!("[INIT-02] LogisModel initialized successfully.");
                *model_guard = Some(m);
            },
            Err(e) => {
                let err_msg = format!("❌ [ERROR-INIT] Failed to load Vision model: {:?}", e);
                println!("{}", err_msg);
                return Err(err_msg);
            }
        }
    }

    // 2. Process Image (Slice & Extract) with Retry Logic for OOM
    println!("[VISION-01] Starting process_image_full...");
    
    let mut extraction_result = {
        let model = model_guard.as_ref().unwrap();
        model.process_image_full(image_path.clone(), &app_handle).await
    };

    if let Err(e) = &extraction_result {
        let err_str = e.to_string();
        if err_str.contains("out of memory") || err_str.contains("CUDA_ERROR_OUT_OF_MEMORY") {
            println!("⚠️ [WARNING] GPU Out of Memory detected. Switching to CPU fallback...");
            
            // Drop current model to free VRAM
            *model_guard = None; 
            
            // Force Init on CPU
            println!("[INIT-RETRY] Initializing LogisModel (CPU Force)...");
            match LogisModel::new(Some("cpu")).await {
                Ok(m) => {
                    println!("[INIT-RETRY] CPU Model initialized.");
                    *model_guard = Some(m);
                    
                    // Retry extraction
                    if let Some(model) = model_guard.as_ref() {
                        println!("[VISION-RETRY] Retrying process_image_full on CPU...");
                        extraction_result = model.process_image_full(image_path.clone(), &app_handle).await;
                    }
                },
                Err(init_err) => {
                    return Err(format!("❌ [ERROR-RETRY] CPU Fallback Init Failed: {:?}", init_err));
                }
            }
        }
    }

    let extracted_data: Value = match extraction_result {
        Ok(data) => {
            println!("[VISION-99] Image processing completed successfully.");
            data
        },
        Err(e) => {
            let err_msg = format!("❌ [ERROR-VISION] Extraction failed: {:?}", e);
            println!("{}", err_msg);
            return Err(err_msg);
        }
    };

    // Re-acquire model reference as guard might have changed
    let model = model_guard.as_ref().unwrap();

    // 3. Generate Summary & Metadata
    let doc_type = extracted_data.get("header")
        .and_then(|h: &Value| h.get("doc_type"))
        .and_then(|s: &Value| s.as_str())
        .unwrap_or("Unknown");
        
    let summary = generate_rich_summary(doc_type, &extracted_data);
    
    // 4. Generate Embedding (Using Gemma-300m via Model)
    let embedding = match model.get_embedding(summary.clone()).await {
        Ok(vec) => vec,
        Err(e) => return Err(format!("Embedding failed: {}", e)),
    };

    // 5. Save to Vector Store
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb"; 
        let _ = std::fs::create_dir_all(db_path);
        match VectorStore::new(db_path).await {
            Ok(s) => *store_guard = Some(s),
            Err(e) => println!("Warning: Failed to init Vector Store: {}", e), 
        }
    }

    if let Some(store) = store_guard.as_mut() {
        let doc_uuid = uuid::Uuid::new_v4().to_string();
        
        // --- Helper for Safe Extraction ---
        let get_str = |cat: &str, key: &str| -> String {
            extracted_data.get(cat).and_then(|c| c.get(key))
                .and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        let get_f32 = |cat: &str, key: &str| -> f32 {
            extracted_data.get(cat).and_then(|c| c.get(key))
                .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32
        };

        // --- Parties ---
        let sup_name = get_str("parties", "supplier_name");
        let buy_name = get_str("parties", "buyer_name");
        let primary_name = if !sup_name.is_empty() { sup_name.clone() } else { buy_name.clone() };

        // --- Lists Extraction ---
        let mut item_descs = Vec::new();
        let mut item_hscodes = Vec::new();
        let mut item_skus = Vec::new();
        
        if let Some(items) = extracted_data.get("line_items").and_then(|v| v.as_array()) {
            for item in items {
                if let Some(d) = item.get("description").and_then(|s| s.as_str()) { 
                    if !d.is_empty() { item_descs.push(d.to_string()); } 
                }
                if let Some(h) = item.get("hs_code").and_then(|s| s.as_str()) {
                    if !h.is_empty() { item_hscodes.push(h.to_string()); }
                }
                if let Some(k) = item.get("sku").and_then(|s| s.as_str()) {
                    if !k.is_empty() { item_skus.push(k.to_string()); }
                }
            }
        }

        let mut cont_nos = Vec::new();
        let mut seal_nos = Vec::new();
        if let Some(conts) = extracted_data.get("containers").and_then(|v| v.as_array()) {
            for c in conts {
                if let Some(n) = c.get("container_number").and_then(|s| s.as_str()) {
                    if !n.is_empty() { cont_nos.push(n.to_string()); }
                }
                if let Some(s) = c.get("seal_number").and_then(|s| s.as_str()) {
                    if !s.is_empty() { seal_nos.push(s.to_string()); }
                }
            }
        }
        // Fallback for flat container
        if cont_nos.is_empty() {
             let flat_cont = get_str("cargo", "container_number"); // Sometimes in cargo
             if !flat_cont.is_empty() { cont_nos.push(flat_cont); }
        }

        // --- Refs ---
        let mut refs = Vec::new();
        if let Some(h) = extracted_data.get("header") {
            for k in ["reference_buyer", "reference_carrier", "reference_export", "po_number", "booking_number", "an_number", "do_number"] {
                if let Some(v) = h.get(k).and_then(|s| s.as_str()) {
                    if !v.is_empty() && v != "N/A" && v != "0" { refs.push(v.to_string()); }
                }
            }
        }

        let doc = TradeDocument {
            uuid: doc_uuid,
            
            // Header
            doc_type: get_str("header", "doc_type"), // Assuming model injects this
            doc_number: get_str("header", "document_number"),
            doc_status: get_str("header", "document_status"),
            issue_date: get_str("header", "issue_date"),
            reference_export: get_str("header", "reference_export"),
            reference_buyer: get_str("header", "reference_buyer"),
            reference_carrier: get_str("header", "reference_carrier"),
            expiry_date: get_str("header", "expiry_date"),
            bl_type: get_str("header", "bl_type"),
            
            // Parties
            name: primary_name,
            supplier_name: sup_name,
            supplier_address: get_str("parties", "supplier_address"),
            supplier_tax_id: get_str("parties", "supplier_tax_id"),
            buyer_name: buy_name,
            buyer_address: get_str("parties", "buyer_address"),
            buyer_tax_id: get_str("parties", "buyer_tax_id"),
            notify_party_name: get_str("parties", "notify_party_name"),
            issuer_name: get_str("parties", "issuer_name"),
            
            // Logistics
            vessel: get_str("logistics", "vehicle_name"),
            voyage_number: get_str("logistics", "voyage_number"),
            pol: get_str("logistics", "location_port_of_loading"),
            pod: get_str("logistics", "location_port_of_discharge"),
            place_receipt: get_str("logistics", "place_receipt"),
            place_delivery: get_str("logistics", "place_delivery"),
            transport_mode: get_str("logistics", "transport_mode"),
            departure_date: get_str("logistics", "departure_date"),
            arrival_date: get_str("logistics", "arrival_date"), // or "eta" logic
            
            // Conditions
            incoterms: get_str("conditions", "incoterms_code"),
            incoterms_place: get_str("conditions", "incoterms_place"),
            payment_terms: get_str("conditions", "payment_terms_type"),
            freight_payment_term: get_str("conditions", "freight_payment_term"),
            lc_tenor: get_str("conditions", "lc_tenor"),
            origin_criterion: get_str("conditions", "origin_criterion"),
            
            // Financials
            currency: get_str("financials", "currency_code"),
            total_amount: get_f32("financials", "amount_total"),
            subtotal_amount: get_f32("financials", "amount_subtotal"),
            tax_amount: get_f32("financials", "amount_tax"),
            freight_amount: get_f32("financials", "charge_freight"),
            insurance_amount: get_f32("financials", "charge_insurance"),
            local_charges: get_f32("financials", "local_charges_total"),
            
            // Cargo
            package_count: get_f32("cargo", "package_count"),
            package_unit: get_str("cargo", "package_unit"),
            weight_gross: get_f32("cargo", "weight_gross"),
            weight_net: get_f32("cargo", "weight_net"),
            volume: get_f32("cargo", "volume_measurement"),
            marks_numbers: get_str("cargo", "marks_and_numbers"),
            
            // Lists
            item_descriptions: item_descs,
            item_hs_codes: item_hscodes,
            item_sku_numbers: item_skus,
            
            container_numbers: cont_nos,
            seal_numbers: seal_nos,

            // System
            text: summary.clone(),
            json_data: extracted_data.to_string(),
            vector: embedding,
            
            // Links
            related_refs: refs,
            transaction_group: None,
            link_reason: None,
        };

        match store.add_document(doc).await {
            Ok(_) => println!("Document saved."),
            Err(e) => println!("Failed to save: {}", e),
        }
    }

    Ok(serde_json::to_string_pretty(&extracted_data).unwrap_or(summary))
}

#[tauri::command]
async fn search_documents(
    state: State<'_, AppState>,
    query: String,
) -> Result<Vec<(String, String, f32)>, String> {
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        let _ = std::fs::create_dir_all(db_path);
        match VectorStore::new(db_path).await {
            Ok(s) => *store_guard = Some(s),
            Err(e) => return Err(format!("Failed to load DB: {}", e)),
        }
    }
    
    // 1. Get Model Access
    let model_guard = state.model.lock().await;
    
    // 2. Split Query (Multi-Intent)
    let sub_queries: Vec<Value> = if let Some(model) = model_guard.as_ref() {
        model.split_query_contexts(query.clone()).await.unwrap_or_else(|_| vec![])
    } else {
        vec![]
    };

    let queries_to_run = if sub_queries.is_empty() {
        vec![json!({"query": query, "$header": {"$$document_type": "ALL"}})]
    } else {
        sub_queries
    };

    let mut combined_results: Vec<(String, String, f32)> = Vec::new();
    let mut seen_uuids = std::collections::HashSet::new();

    if let Some(store) = store_guard.as_ref() {
        for ctx in queries_to_run {
            let sub_q = ctx.get("query").and_then(|s: &Value| s.as_str()).unwrap_or(&query).to_string();
            let doc_type = ctx.get("$header")
                .and_then(|h: &Value| h.get("$$document_type"))
                .and_then(|s: &Value| s.as_str())
                .map(|s: &str| s.to_string());

            // Get Embedding for Sub-Query
            let query_vec = if let Some(model) = model_guard.as_ref() {
                model.get_embedding(sub_q.clone()).await.unwrap_or(vec![0.0; 768])
            } else {
                vec![0.0; 768]
            };

            // Parse Filters & Search Text
            let (filters, search_text) = if let Some(model) = model_guard.as_ref() {
                if let Ok(parsed) = model.parse_query_to_filters(sub_q.clone(), doc_type).await {
                    let f = parsed.get("filters").cloned().or_else(|| parsed.get("$filters").cloned());
                    let t = parsed.get("search_text").and_then(|s| s.as_str()).map(|s| s.to_string());
                    (f, t)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            // Search with Hybrid Params
            if let Ok(results) = store.search(query_vec, filters, search_text).await {
                for (uuid, text, score) in results {
                    if !seen_uuids.contains(&uuid) {
                        seen_uuids.insert(uuid.clone());
                        combined_results.push((uuid, text, score));
                    }
                }
            }
        }
    } else {
        return Err("DB not initialized".to_string());
    }

    // Sort combined results by score
    combined_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(combined_results)
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
) -> Result<Vec<TradeDocument>, String> {
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb"; 
        let _ = std::fs::create_dir_all(db_path);
        match VectorStore::new(db_path).await {
            Ok(s) => *store_guard = Some(s),
            Err(e) => return Err(format!("Failed to init store: {}", e)), 
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        store.list_all(limit, offset).await.map_err(|e| e.to_string())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<Option<TradeDocument>, String> {
    let mut store_guard = state.store.lock().await;
    // Ensure store is init
    if store_guard.is_none() {
         let db_path = "data/lancedb"; 
         let _ = std::fs::create_dir_all(db_path);
         if let Ok(s) = VectorStore::new(db_path).await {
             *store_guard = Some(s);
         }
    }

    if let Some(store) = store_guard.as_ref() {
        Ok(store.get_document(&uuid).await)
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn update_document(
    state: State<'_, AppState>,
    uuid: String,
    json_data: String,
) -> Result<String, String> {
    let mut store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_mut() {
        match store.update_document(&uuid, &json_data).await {
            Ok(_) => Ok("Updated".to_string()),
            Err(e) => Err(e.to_string()),
        }
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn delete_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<String, String> {
    let mut store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_mut() {
        match store.delete_document(&uuid).await {
            Ok(_) => Ok("Deleted".to_string()),
            Err(e) => Err(e.to_string()),
        }
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn delete_documents(
    state: State<'_, AppState>,
    uuids: Vec<String>,
) -> Result<String, String> {
    let mut store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_mut() {
        match store.delete_documents(uuids).await {
            Ok(_) => Ok("Deleted batch".to_string()),
            Err(e) => Err(e.to_string()),
        }
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn check_query_intent(
    state: State<'_, AppState>,
    query: String,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    
    if let Some(model) = model_guard.as_ref() {
        model.parse_query_intent(query).await.map_err(|e| e.to_string())
    } else {
        Err("Model not initialized".to_string())
    }
}

#[tauri::command]
async fn deep_research_command(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    query: String,
    doc_id: Option<String>,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    let model = model_guard.as_ref().unwrap();

    // 1. Context Gathering
    let mut context_data = String::new();
    let mut store_guard = state.store.lock().await;
    
    if store_guard.is_none() {
        // Try init
        let db_path = "data/lancedb";
        let _ = std::fs::create_dir_all(db_path);
        if let Ok(s) = VectorStore::new(db_path).await {
            *store_guard = Some(s);
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        if let Some(uuid) = doc_id {
            // Focus on specific document
            if let Some(doc) = store.get_document(&uuid).await {
                context_data = format!("Target Document Summary: {}\nData: {}", doc.text, doc.json_data);
            }
        } else {
            // General search for context
            let emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768]);
            if let Ok(results) = store.search(emb, None, None).await {
                let docs: Vec<String> = results.iter().take(3)
                    .map(|(_, text, _)| format!("- {}", text))
                    .collect();
                context_data = docs.join("\n");
            }
        }
    }
    
    // 2. Run Deep Research
    model.run_deep_research(query, context_data, &app_handle).await.map_err(|e| e.to_string())
}

#[tauri::command]

async fn set_login_state(

    state: State<'_, AppState>,

    is_logged_in: bool,

    token: Option<String>,

) -> Result<String, String> {

    let mut store_guard = state.store.lock().await;

    if let Some(store) = store_guard.as_ref() {

        let mut config = store.load_config();

        config.is_logged_in = is_logged_in;

        config.auth_token = token;

        

        match store.save_config(&config) {

            Ok(_) => Ok(format!("Login state set to: {}", is_logged_in)),

            Err(e) => Err(e.to_string()),

        }

    } else {

        Err("Store not initialized".to_string())

    }

}



#[cfg_attr(mobile, tauri::mobile_entry_point)]

pub fn run() {

    let model = Arc::new(Mutex::new(None));

    let store = Arc::new(Mutex::new(None));



    let model_clone = model.clone();

    let store_clone = store.clone();

    let store_server = store.clone();



        // Spawn Scheduler



        tauri::async_runtime::spawn(async move {



            scheduler::start_background_worker(store_clone, model_clone).await;



        });



    



        tauri::Builder::default()



    

        .plugin(tauri_plugin_fs::init())

        .plugin(tauri_plugin_shell::init())

                .manage(AppState {

                    model: model,

                    store: store,

                })

                .setup(|app| {

                    let store = app.state::<AppState>().store.clone();

                    

                    // Listen for events from the injected browser script

                    app.listen("new-task-from-browser", move |event| {

                        println!("[Event] Received 'new-task-from-browser'");

                        

                        if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {

                            let store_clone = store.clone();

                            

                            tauri::async_runtime::spawn(async move {

                                let store_guard = store_clone.lock().await;

                                if let Some(db) = store_guard.as_ref() {

                                    let now = chrono::Utc::now().timestamp_millis();

                                    

                                    // Map JSON payload to Task struct

                                    let task = crate::store::Task {

                                        id: uuid::Uuid::new_v4().to_string(),

                                        r#type: payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),

                                        from_source: "injected_script".to_string(),

                                        to_dest: "local".to_string(),

                                        cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),

                                        bcc: "".to_string(),

                                        ref_id: payload_val.get("ref_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),

                                        data_json: payload_val.to_string(),

                                        created_at: now,

                                        updated_at: now,

                                        status: "pending".to_string(),

                                    };

        

                                    match db.add_task(task).await {

                                        Ok(_) => println!("[Event] Task saved to DB successfully."),

                                        Err(e) => eprintln!("[Event] Failed to save task: {}", e),

                                    }

                                } else {

                                    eprintln!("[Event] Database not initialized.");

                                }

                            });

                        } else {

                            eprintln!("[Event] Failed to parse payload: {}", event.payload());

                        }

                    });

                    Ok(())

                })

                .plugin(tauri_plugin_opener::init())

        

        .plugin(tauri_plugin_dialog::init())

        .invoke_handler(tauri::generate_handler![

            summarize_image, 

            search_documents, 

            get_all_documents,

            get_document,

            update_document,

            delete_document,

            delete_documents,

            check_query_intent,

            deep_research_command,

            launch_browser,

            launch_best_browser,

            check_available_browsers,

            resize_window,

            start_drag,

            move_to_top_center,

            set_login_state

        ])

        .run(tauri::generate_context!())

        .expect("error while running tauri application");

}
