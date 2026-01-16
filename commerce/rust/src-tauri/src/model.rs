use anyhow::anyhow;
use aha::models::qwen3vl::generate::Qwen3VLGenerateModel;
use aha::openai_types::{
    ChatCompletionParameters,
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ImageURL,
};
use aha::utils::get_device;
use candle_core::{Device, DType};
use image::{DynamicImage, GenericImageView};
use serde_json::{Value, json, Map};
use std::sync::{Arc, Mutex};
use regex::Regex;
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::*;
use sysinfo::System;

pub struct Spinner {
    pub frames: Vec<&'static str>,
    pub interval: u64,
}

impl Spinner {
    pub fn dots() -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            interval: 80,
        }
    }
}

pub struct LogisModel {
    generator: Arc<Mutex<Qwen3VLGenerateModel>>,
    is_cpu_mode: bool,
}

impl LogisModel {
    pub async fn new(device_preference: Option<&str>) -> anyhow::Result<Self> {
        // ... (Keep existing new implementation)
        println!("[MODEL-00] Starting LogisModel::new() - Aha (Qwen3-VL Local) Mode");

        // 0. Check System Memory (Real-time)
        let mut sys = System::new_all();
        sys.refresh_memory(); 
        
        let total_mem = sys.total_memory();
        let free_mem = sys.available_memory(); // Available includes cache/buffers that can be reclaimed
        
        // 1. Calculate Safe Limit (80% of Available RAM to allow OS breathing room)
        let safe_limit = (free_mem as f64 * 0.8) as u64;
        
        // 2. Precise/Dynamic Model Requirement Calculation
        // Qwen2.5-VL-2B-Instruct-Q4_K_M GGUF file size is approx 1.6 GB.
        let base_model_size: u64 = 1_717_986_918; // ~1.6 GB
        
        // Dynamically adjust requirement based on Total System RAM
        let estimated_req = if total_mem <= 8 * 1024 * 1024 * 1024 {
            // Case A: Low System RAM (<= 8GB)
            // Strategy: Very Conservative. 
            // Requirement: Model (1.6GB) + Context/Overhead (1.5GB) = ~3.1 GB
            println!("[SYS-INIT] Low System RAM (<= 8GB) detected. Setting Strict Minimum Requirements.");
            base_model_size + (1536 * 1024 * 1024) 
        } else {
            // Case B: Standard System RAM (> 8GB)
            // Strategy: Stable fit. Reserve space for larger context and OS overhead.
            // Requirement: Model (1.6GB) + Context (1.0GB) + Safety Buffer (0.9GB) = ~3.5 GB
            println!("[SYS-INIT] Standard System RAM (> 8GB) detected. Setting Recommended Requirements.");
            base_model_size + (1024 * 1024 * 1024) + (900 * 1024 * 1024)
        };
        
        println!("[SYS-INIT] Real-time Memory Check:");
        println!("  - Total RAM: {:.2} GB", total_mem as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("  - Available: {:.2} GB", free_mem as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("  - Safe Limit (80%): {:.2} GB", safe_limit as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("  - Required (Dynamic): {:.2} GB", estimated_req as f64 / 1024.0 / 1024.0 / 1024.0);
        
        let mut final_device_preference = device_preference;

        // [Memory Defense Logic] Restore strictly for system stability.
        // If memory is critically low, we must restrict device usage to prevent system-wide OOM.
        if total_mem <= 8 * 1024 * 1024 * 1024 && free_mem < 3 * 1024 * 1024 * 1024 {
             println!("⚠️ [DEFENSE] Critical Low RAM (< 3GB) detected on 8GB System. Ensuring safe execution mode.");
             // If GPU is not explicitly requested, or if we need to fall back for stability:
             if device_preference.is_none() {
                 final_device_preference = Some("cpu");
             }
        } else if safe_limit < estimated_req { 
            println!("⚠️ [DEFENSE] Available RAM ({:.2} GB) is below safe threshold ({:.2} GB).", 
                safe_limit as f64 / 1024.0 / 1024.0 / 1024.0,
                estimated_req as f64 / 1024.0 / 1024.0 / 1024.0
            );
            
            if device_preference.is_none() {
                final_device_preference = Some("cpu");
            }
        }

        let base_path = std::fs::canonicalize("src-tauri/models")
            .or_else(|_| std::fs::canonicalize("models"))?;

        // Prioritize GGUF directory
        let gguf_dir = base_path.join("Qwen3-VL-2B-Instruct-gguf");
        
        let model_dir = if gguf_dir.exists() {
            println!("[MODEL-01] Found GGUF directory: {}", gguf_dir.display());
            gguf_dir
        } else {
            println!("[MODEL-01] GGUF directory not found at {}, using base path...", gguf_dir.display());
            base_path
        };

        println!("[MODEL-01] Loading Qwen3-VL from: {}", model_dir.display());

        let device = if let Some("cpu") = final_device_preference {
            println!("[MODEL-01] Using CPU Device (Forced or Requested).");
            Device::Cpu
        } else {
            get_device(None)
        };

        // CPU usually requires F32 for reliable matmul support in candle
        let dtype = if device.is_cpu() {
            println!("[MODEL-CONFIG] CPU detected -> Using F32 precision.");
            Some(DType::F32)
        } else {
            println!("[MODEL-CONFIG] GPU detected -> Using BF16 precision.");
            Some(DType::BF16)
        };

        let generator = Qwen3VLGenerateModel::init(
            model_dir.to_str().unwrap(),
            Some(&device), 
            dtype 
        ).map_err(|e| anyhow!("Failed to init Qwen3VL: {}", e))?;

        println!("[MODEL-02] Qwen3-VL Generator initialized.");

        Ok(Self {
            generator: Arc::new(Mutex::new(generator)),
            is_cpu_mode: device.is_cpu(),
        })
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str) -> anyhow::Result<String> {
        // Offload the heavy inference to a blocking task to avoid blocking the async runtime
        let self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        println!("[MODEL-CHAT] System: {:.50}...", system_text.replace("\n", " "));
        
        tokio::task::spawn_blocking(move || {
            let mut gen = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_text,
                name: None,
            });

            let content_parts = vec![
                ChatCompletionRequestMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text }
                )
            ];

            let user_message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
                model: "qwen3vl".to_string(),
                max_tokens: Some(2048),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params).map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }).await?
    }

    pub async fn chat_with_spinner(
        &self, 
        system: &str, 
        user_input: &str,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        max_tokens: usize
    ) -> anyhow::Result<String> {
        let self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        
        let task = tokio::task::spawn_blocking(move || {
            let mut gen = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_text,
                name: None,
            });

            let content_parts = vec![
                ChatCompletionRequestMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text }
                )
            ];

            let user_message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tokens as u32),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params).map_err(|e| anyhow!("Inference failed: {}", e))?;
            Ok(response)
        });

        let spinner = Spinner::dots();
        let mut idx = 0;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(spinner.interval));
        
        tokio::pin!(task);

        loop {
            tokio::select! {
                res = &mut task => {
                    return res.map_err(|e| anyhow!("Task join error: {}", e))?; 
                }
                _ = interval.tick() => {
                    let frame = spinner.frames[idx % spinner.frames.len()];
                    idx += 1;
                    
                    let mut payload = base_payload.clone();
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("spinner".to_string(), json!(frame));
                    }
                    let _ = app_handle.emit(event_name, payload);
                }
            }
        }
    }

    pub async fn chat_with_image_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        max_tokens: usize
    ) -> anyhow::Result<String> {
        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || {
            let mut gen = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            
            let mut content_parts = Vec::new();
            
            if let Some(img) = image {
                let mut buf = Cursor::new(Vec::new());
                img.write_to(&mut buf, image::ImageFormat::Png)?;
                let b64 = BASE64_STANDARD.encode(buf.into_inner());
                let url = format!("data:image/png;base64,{}", b64);
                
                content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageURL { url, detail: None }
                    }
                ));
            }

            content_parts.push(ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: prompt }
            ));

            let message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![ChatCompletionRequestMessage::User(message)],
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tokens as u32),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        let spinner = Spinner::dots();
        let mut idx = 0;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(spinner.interval));
        
        tokio::pin!(task);

        loop {
            tokio::select! {
                res = &mut task => {
                    return res.map_err(|e| anyhow!("Task join error: {}", e))?; 
                }
                _ = interval.tick() => {
                    let frame = spinner.frames[idx % spinner.frames.len()];
                    idx += 1;
                    
                    let mut payload = base_payload.clone();
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("spinner".to_string(), json!(frame));
                    }
                    let _ = app_handle.emit(event_name, payload);
                }
            }
        }
    }

    fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>) -> anyhow::Result<String> {
        let mut gen = self.generator.lock().map_err(|_| anyhow!("Poisoned lock"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen3vl".to_string(),
            max_tokens: Some(1024), // Reduced from 2048 to save memory
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params).map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>, 
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value
    ) -> anyhow::Result<String> {
        let generator_arc = self.generator.clone();
        
        // Spawn the heavy task using Tokio directly for standard behavior
        let task = tokio::task::spawn_blocking(move || {
            let mut gen = generator_arc.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            
            let mut content_parts = Vec::new();
            if let Some(img) = image {
                let mut buf = Cursor::new(Vec::new());
                img.write_to(&mut buf, image::ImageFormat::Png)?;
                let b64 = BASE64_STANDARD.encode(buf.into_inner());
                let url = format!("data:image/png;base64,{}", b64);
                
                content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageURL { url, detail: None }
                    }
                ));
            }
    
            content_parts.push(ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: prompt }
            ));
    
            let message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };
    
            let params = ChatCompletionParameters {
                messages: vec![ChatCompletionRequestMessage::User(message)],
                model: "qwen3vl".to_string(),
                max_tokens: Some(1024),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params).map_err(|e| anyhow!("Inference failed: {}", e))
        });
        
        let spinner = Spinner::dots();
        let mut idx = 0;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(spinner.interval));
        
        // Pin the task to poll it in select!
        tokio::pin!(task);

        loop {
            tokio::select! {
                res = &mut task => {
                    // Task finished
                    // res is Result<Result<String, Error>, JoinError>
                    // Outer ? handles JoinError
                    // Inner logic returns Result<String, Error>
                    return res.map_err(|e| anyhow!("Task join error: {}", e))?; 
                }
                _ = interval.tick() => {
                    // Update spinner
                    let frame = spinner.frames[idx % spinner.frames.len()];
                    idx += 1;
                    
                    let mut payload = base_payload.clone();
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("spinner".to_string(), json!(frame));
                    }
                    // Emit spinner event (best effort)
                    let _ = app_handle.emit(event_name, payload);
                }
            }
        }
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle) -> anyhow::Result<Value> {
        let log_file = "debug_log.txt";
        let log = |msg: &str| {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_file) {
                use std::io::Write;
                let _ = writeln!(f, "{}", msg);
            }
            println!("{}", msg);
        };

        log(&format!("[PROCESS-01] Opening image: {}", image_path));
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Dynamic Resizing based on Device/Memory
        let mut sys = System::new_all();
        sys.refresh_memory();
        let free_mem_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        
        let (main_resize, thumb_resize) = if free_mem_gb < 4.0 {
            log(&format!("[PROCESS-CONFIG] Low RAM ({:.2} GB) detected. Using Conservative Mode (1024px/512px).", free_mem_gb));
            (1024, 512)
        } else {
            log(&format!("[PROCESS-CONFIG] RAM sufficient ({:.2} GB). Using Standard Mode (1536px/768px).", free_mem_gb));
            (1536, 768)
        };
        
        log(&format!("[PROCESS-02] Global Resize to {}px width...", main_resize));
        let master_img = full_img_raw.resize(main_resize, u32::MAX, image::imageops::FilterType::Triangle);
        let (w, h) = master_img.dimensions();

        log(&format!("[PROCESS-03] Creating Classification Thumbnail ({:?}px)...", thumb_resize));
        let class_img = master_img.resize(thumb_resize, u32::MAX, image::imageops::FilterType::Triangle);

        log("[PROCESS-04] Running Classification...");
        let class_prompt = r###"Classify document: 
        1. CONTRACT: PO (Purchase Order), PI (Proforma Invoice), SC (Sales Contract), LC (Letter of Credit)
        2. SHIPPING: CI (Commercial Invoice), PL (Packing List), BL (Bill of Lading), AWB (Airway Bill), SA (Shipping Advice), DO (Delivery Order), AN (Arrival Notice), BC (Booking Confirm)
        3. CUSTOMS: ED (Export Declaration), ID (Import Declaration), CINV (Customs Invoice), CO (Certificate of Origin)
        4. CERTIFICATES: IC (Inspection Cert), WC (Weight Cert), CA (Analysis Cert), PHYTO (Phytosanitary), HC (Health Cert), BEN_CERT (Beneficiary Cert)
        5. SPECIAL: DGD (Dangerous Goods), MSDS, POA (Power of Attorney), BIZ_LIC (Business License), INS (Insurance Policy)
        
        Return JSON: {"doc_type": "CODE"}"###;
        
        let detected_type = {
            // Emitting spinner for classification
            let res = self.run_inference_with_spinner(
                class_prompt.to_string(), 
                Some(class_img), 
                app_handle, 
                "extraction-progress", 
                json!({ "summary": "Identifying document type...", "raw": "Analyzing...", "category": "Processing" })
            ).await?;

            log(&format!("[DEBUG] Raw Classification Response: {}", res));
            let dtype = extract_json_field(&res, "doc_type").unwrap_or("Unknown".to_string());
            
            // Emit success for classification
            let _ = app_handle.emit("extraction-progress", json!({ "category": "Processing", "data": { "doc_type": dtype }, "spinner": "✅" }));
            
            dtype
        };
        
        log(&format!("[DEBUG] Detected Type: {}", detected_type));

        let mut root = Map::new();
        root.insert("header".to_string(), json!({ "doc_type": &detected_type }));
        root.insert("parties".to_string(), Value::Object(Map::new()));
        root.insert("logistics".to_string(), Value::Object(Map::new()));
        root.insert("conditions".to_string(), Value::Object(Map::new()));
        root.insert("financials".to_string(), Value::Object(Map::new()));
        root.insert("cargo".to_string(), Value::Object(Map::new()));
        root.insert("line_items".to_string(), Value::Array(vec![]));
        root.insert("containers".to_string(), Value::Array(vec![]));

        let missions = get_slice_config(&detected_type);
        
        for (i, mission) in missions.iter().enumerate() {
            log(&format!("\n[SEQ-MISSION {}] Category: {}", i + 1, mission.cat));
            
            let res_text = {
                let (top_pct, bot_pct) = mission.box_range;
                let start_y = (h as f32 * top_pct).round() as u32;
                let end_y = (h as f32 * bot_pct).round() as u32;
                let crop_h = end_y.saturating_sub(start_y).max(10); 
                
                let processed_img = master_img.crop_imm(0, start_y, w, crop_h);
                
                let schema = get_category_schema(mission.cat, &detected_type);
                let prompt = format!("MISSION: Extract fields for category '{}'.\nRULES: Valid JSON ONLY.\nSCHEMA:\n{{\n{}\n}}", mission.cat.to_uppercase(), schema);
                
                let task_desc = format!("[{}] {} ({}%~{}%)", detected_type, mission.cat.to_uppercase(), (top_pct*100.0) as i32, (bot_pct*100.0) as i32);
                
                self.run_inference_with_spinner(
                    prompt, 
                    Some(processed_img), 
                    app_handle, 
                    "extraction-progress", 
                    json!({ "summary": format!("Analyzing: {}", task_desc), "raw": format!("Analyzing {}...", task_desc), "category": mission.cat })
                ).await?
            };

            log(&format!("[DEBUG] Raw Extraction ({}): {}", mission.cat, res_text));

            if let Some(json_val) = extract_json_from_text(&res_text) {
                merge_json_manual(&mut root, mission.cat, json_val.clone());
                // Emit final success for this step (without spinner or with success symbol)
                let _ = app_handle.emit("extraction-progress", json!({ "category": mission.cat, "data": json_val, "spinner": "✅" }));
            }
        }

        Ok(Value::Object(root))
    }

    pub async fn get_embedding(&self, _text: String) -> anyhow::Result<Vec<f32>> {
        // Placeholder for future embedding model integration (e.g. Gemma-300m)
        // Currently returns a zero vector to prevent crashes while maintaining search structure.
        Ok(vec![0.0; 768])
    }

    pub async fn split_query_contexts(&self, query: String) -> anyhow::Result<Vec<Value>> {
        let system_prompt = "Split user query into JSON sub-queries with document type. Structure: [{\"query\": \"...\", \"header\": {\"document_type\": \"TYPE\"}}]";
        let prompt = format!("{}\n\nQuery: {}\nJSON Output:", system_prompt, query);
        
        let res = self.run_inference_text(prompt, None)?;
        if let Some(json_val) = extract_json_from_text(&res) {
            if json_val.is_array() {
                // Ensure the keys match what lib.rs search_documents expects ($header, $$document_type)
                let mut mapped_queries = Vec::new();
                if let Some(arr) = json_val.as_array() {
                    for item in arr {
                        let mut mapped = item.clone();
                        if let Some(h) = item.get("header") {
                            let doc_type = h.get("document_type").cloned().unwrap_or(json!("ALL"));
                            mapped.as_object_mut().unwrap().insert("$header".to_string(), json!({ "$$document_type": doc_type }));
                        }
                        mapped_queries.push(mapped);
                    }
                }
                return Ok(mapped_queries);
            }
            return Ok(vec![json_val]);
        }
        
        Ok(vec![json!({ "query": query, "$header": { "$$document_type": "ALL" } })])
    }

    pub async fn parse_query_to_filters(&self, query: String, doc_type: Option<String>) -> anyhow::Result<Value> {
        let doc_type = doc_type.unwrap_or_else(|| "ALL".to_string());
        let schema_def = self.get_search_schema_definitions(&doc_type);
        
        let prompt = format!(r###"Extract filters into JSON. 
RULES: 
1. NO HALLUCINATION. 
2. PRECISE MAPPING to Schema. 
3. OPERATORS: "$eq", "$contains", "$gt" (greater), "$lt" (less), "$gte", "$lte".

DYNAMIC SCHEMA:
{}

REQUIRED JSON FORMAT: 
{{
  "header": {{ "document_type": "{}" }},
  "filters": {{
      "category.field_path": {{ "$operator": value }}
  }}
}} 

Query: {}
JSON Output:"###, schema_def, doc_type, query);

        let res = self.run_inference_text(prompt, None)?;
        if let Some(json_val) = extract_json_from_text(&res) {
            return Ok(json_val);
        }
        
        Ok(json!({}))
    }

    // --- Ported from Python (search_engine.py) ---
    pub async fn parse_query_intent(&self, query: String) -> anyhow::Result<String> {
        let system_prompt = r###"Classify the query into one of these types:
1. SEARCH: Finding specific documents by filters (e.g., 'Find Samsung invoices').
2. RESEARCH: Complex questions requiring cross-referencing or deep analysis (e.g., 'What is the trend of our shipping delays?').

Output JSON: {"intent": "SEARCH|RESEARCH"}"###;

        let prompt = format!("{}\n\nQuery: {}\nJSON Output:", system_prompt, query);
        let res = self.run_inference_text(prompt, None)?;
        
        let intent = extract_json_from_text(&res)
            .and_then(|v| v.get("intent").and_then(|s| s.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "SEARCH".to_string());
            
        Ok(intent)
    }

    // --- Ported from Python (logic.py) ---
    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle) -> anyhow::Result<String> {
        let spinner = Spinner::dots();
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);

        // 1. Context Gathering
        status_history.push_str("✅ Context gathered.\n\n");
        let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": spinner.frames[0] }));

        // 2. Multi-step reasoning loop
        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for (i, step) in steps.iter().enumerate() {
            let frame = spinner.frames[i % spinner.frames.len()];
            status_history.push_str(&format!("**{} {}**\n", frame, step));
            let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": frame }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            // In a real implementation, we might want to stream this too, but for now we wait for the step result
            let step_result = self.run_inference_text(prompt, None)?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": "✅" }));
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None)?;
        status_history.push_str(&report);
        
        let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": "✅" }));

        Ok(report)
    }

    fn get_search_schema_definitions(&self, _doc_type: &str) -> String {
        r###"{
  "header.document_type": { "desc": "Type (Invoice, BL, AWB, PO, BC, AN, DO...)", "type": "String" },
  "header.document_number": { "desc": "ID, Doc No, Reference No", "type": "String" },
  "header.po_number": { "desc": "Purchase Order No (PO)", "type": "String" },
  "header.booking_number": { "desc": "Booking Reference No (BC)", "type": "String" },
  "header.an_number": { "desc": "Arrival Notice No (AN)", "type": "String" },
  "header.do_number": { "desc": "Delivery Order No (DO)", "type": "String" },
  "header.issue_date": { "desc": "Date (YYYY-MM-DD)", "type": "String" },
  
  "parties.supplier_name": { "desc": "Seller, Shipper, Exporter, Vendor", "type": "String" },
  "parties.buyer_name": { "desc": "Buyer, Consignee, Importer", "type": "String" },
  "parties.notify_party_name": { "desc": "Notify Party", "type": "String" },
  
  "financials.amount_total": { "desc": "Total Value/Amount", "type": "Number" },
  "financials.local_charges_total": { "desc": "Total Local Charges (AN)", "type": "Number" },
  
  "logistics.vehicle_name": { "desc": "Vessel Name, Flight No", "type": "String" },
  "logistics.location_port_of_loading": { "desc": "POL, Origin", "type": "String" },
  "logistics.location_port_of_discharge": { "desc": "POD, Destination", "type": "String" },
  "logistics.pickup_location": { "desc": "Pickup Location (DO)", "type": "String" },
  "logistics.etd": { "desc": "Estimated Departure", "type": "String" },
  "logistics.eta": { "desc": "Estimated Arrival", "type": "String" },
  
  "conditions.incoterms_code": { "desc": "Incoterms (FOB, CIF)", "type": "String" }
}"###.to_string()
    }
}

#[derive(Clone)]
struct Mission {
    cat: &'static str,
    box_range: (f32, f32),
}

fn get_slice_config(doc_type: &str) -> Vec<Mission> {
    match doc_type {
        // --- 1. Contract & Payment ---
        "CI" | "PI" | "SC" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "logistics", box_range: (0.20, 0.50) },
            Mission { cat: "items", box_range: (0.30, 0.70) },
            Mission { cat: "items", box_range: (0.50, 0.85) },
            Mission { cat: "financials", box_range: (0.70, 0.95) },
            Mission { cat: "conditions", box_range: (0.80, 1.00) },
        ],
        "PO" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "logistics", box_range: (0.20, 0.50) },
            Mission { cat: "items", box_range: (0.30, 0.80) },
            Mission { cat: "financials", box_range: (0.70, 0.95) },
            Mission { cat: "conditions", box_range: (0.80, 1.00) },
        ],
        "LC" => vec![
            Mission { cat: "header", box_range: (0.00, 0.30) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "financials", box_range: (0.20, 0.60) }, // LC has dense financial terms
            Mission { cat: "logistics", box_range: (0.40, 0.70) },
            Mission { cat: "conditions", box_range: (0.50, 1.00) }, // LC is mostly conditions
        ],

        // --- 2. Shipping & Logistics ---
        "PL" | "SA" => vec![
            Mission { cat: "header", box_range: (0.00, 0.20) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "logistics", box_range: (0.20, 0.50) },
            Mission { cat: "items", box_range: (0.30, 0.80) },
            Mission { cat: "cargo", box_range: (0.60, 0.95) }, // PL/SA focus on physical cargo
            Mission { cat: "conditions", box_range: (0.85, 1.00) },
        ],
        "BL" => vec![
            Mission { cat: "header", box_range: (0.00, 0.20) },
            Mission { cat: "parties", box_range: (0.00, 0.60) },
            Mission { cat: "logistics", box_range: (0.35, 0.65) },
            Mission { cat: "cargo", box_range: (0.50, 0.90) },
            Mission { cat: "conditions", box_range: (0.80, 1.00) },
        ],
        "AWB" => vec![
            Mission { cat: "header", box_range: (0.00, 0.15) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "logistics", box_range: (0.10, 0.40) },
            Mission { cat: "cargo", box_range: (0.30, 0.70) },
            Mission { cat: "financials", box_range: (0.60, 0.90) },
            Mission { cat: "conditions", box_range: (0.85, 1.00) },
        ],
        "BC" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "parties", box_range: (0.00, 0.50) },
            Mission { cat: "logistics", box_range: (0.30, 0.70) },
            Mission { cat: "cargo", box_range: (0.50, 0.90) },
        ],
        "AN" | "DO" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "parties", box_range: (0.00, 0.50) },
            Mission { cat: "logistics", box_range: (0.30, 0.70) },
            Mission { cat: "financials", box_range: (0.50, 0.90) }, // AN has local charges
            Mission { cat: "cargo", box_range: (0.60, 1.00) },
        ],

        // --- 3. Customs & Declarations ---
        "ED" | "ID" | "CINV" => vec![
            Mission { cat: "header", box_range: (0.00, 0.20) }, // Doc No, Date
            Mission { cat: "parties", box_range: (0.00, 0.30) }, // Exporter/Importer
            Mission { cat: "logistics", box_range: (0.20, 0.50) }, // Vessel, Voyage
            Mission { cat: "financials", box_range: (0.40, 0.70) }, // Exchange Rate, Tax
            Mission { cat: "items", box_range: (0.50, 0.90) }, // List of goods (dense)
            Mission { cat: "conditions", box_range: (0.80, 1.00) }, // Notes
        ],
        "CO" => vec![
            Mission { cat: "header", box_range: (0.00, 0.20) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "logistics", box_range: (0.30, 0.50) },
            Mission { cat: "items", box_range: (0.40, 0.80) },
            Mission { cat: "conditions", box_range: (0.75, 1.00) },
        ],

        // --- 4. Certificates & Special ---
        "IC" | "WC" | "CA" | "PHYTO" | "HC" | "BEN_CERT" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "parties", box_range: (0.00, 0.40) },
            Mission { cat: "items", box_range: (0.30, 0.80) }, // Test Results / Species List
            Mission { cat: "conditions", box_range: (0.70, 1.00) }, // "We certify..." statement
        ],
        "DGD" | "MSDS" => vec![
            Mission { cat: "header", box_range: (0.00, 0.25) },
            Mission { cat: "logistics", box_range: (0.20, 0.50) },
            Mission { cat: "cargo", box_range: (0.40, 0.90) }, // DG Details are cargo-heavy
        ],
        "POA" | "BIZ_LIC" | "INS" => vec![
            Mission { cat: "header", box_range: (0.00, 0.30) },
            Mission { cat: "parties", box_range: (0.10, 0.50) },
            Mission { cat: "conditions", box_range: (0.40, 1.00) }, // Legal text
        ],

        // --- Fallback ---
        _ => vec![
            Mission { cat: "header", box_range: (0.0, 0.3) },
            Mission { cat: "items", box_range: (0.2, 0.8) },
            Mission { cat: "conditions", box_range: (0.7, 1.0) },
        ],
    }
}

fn get_category_schema(category: &str, doc_type: &str) -> String {
    let mut schema_fields: Vec<(String, String, String, String)> = Vec::new();
    let mut set = |key: &str, desc: &str, dtype: &str, mode: &str| {
        if let Some(pos) = schema_fields.iter().position(|(k, _, _, _)| k == key) {
            schema_fields[pos] = (key.to_string(), desc.to_string(), dtype.to_string(), mode.to_string());
        } else {
            schema_fields.push((key.to_string(), desc.to_string(), dtype.to_string(), mode.to_string()));
        }
    };

    match category {
        "header" => {
            // --- Common Fields ---
            set("document_type", "CLASSIFIED TYPE", "String", "Info");
            set("document_status", "ORIGINAL, COPY, DRAFT, DUPLICATE", "String", "INFER");
            set("issue_date", "Date of Issue/Creation (YYYY-MM-DD)", "String", "FIND");
            set("page_current", "Page X", "Number", "FIND");
            set("page_total", "Total Pages", "Number", "FIND");
            set("reference_internal", "Our Ref", "String", "Optional");
            
            // --- 1. Contract & Payment (PO, PI, SC, LC) ---
            if ["CI", "PI", "PL", "PO", "SC", "CINV"].contains(&doc_type) {
                set("document_number", "Invoice No, PO No, Contract No", "String", "FIND");
                set("reference_buyer", "Buyer Ref, Order No", "String", "FIND");
                set("reference_seller", "Seller Ref, Export Ref", "String", "FIND");
                set("reference_lc", "L/C Number Ref", "String", "Optional");
                
                if doc_type == "SC" {
                    set("document_number", "Sales Contract No", "String", "FIND");
                    set("contract_date", "Date of Contract", "String", "FIND");
                    set("validity_date", "Offer Validity / Contract Expiry", "String", "FIND");
                } else if doc_type == "PO" {
                    set("document_number", "Purchase Order No", "String", "FIND");
                    set("revision_number", "Rev No", "String", "Optional");
                }
            } else if doc_type == "LC" {
                set("document_number", "Credit Number (L/C No)", "String", "FIND");
                set("expiry_date", "Date of Expiry", "String", "FIND");
                set("expiry_place", "Place of Expiry", "String", "FIND");
                set("issue_date", "Date of Issue", "String", "FIND");
                set("applicable_rules", "UCP 600 / ISP98", "String", "FIND");
                set("available_with_bank", "Available With (Bank)", "String", "FIND");
            }
            
            // --- 2. Shipping & Logistics (BL, AWB, SA, DO, AN, BC) ---
            else if ["BL", "AWB", "AN", "BC", "DO", "SA"].contains(&doc_type) {
                set("document_number", "B/L No, AWB No, Booking No", "String", "FIND");
                set("reference_carrier", "Carrier Ref / Master No", "String", "FIND");
                
                if doc_type == "SA" {
                     set("document_number", "Shipping Advice No", "String", "FIND");
                     set("related_invoice_number", "Commercial Invoice No", "String", "FIND");
                     set("related_lc_number", "L/C No", "String", "FIND");
                } else if doc_type == "DO" {
                    set("document_number", "Delivery Order No", "String", "FIND");
                    set("do_expiration_date", "D/O Expiry", "String", "FIND");
                } else if doc_type == "AN" {
                    set("document_number", "Arrival Notice No", "String", "FIND");
                } else if doc_type == "BL" {
                    set("bl_type", "ORIGINAL, WAYBILL, SURRENDER", "String", "INFER");
                    set("number_of_originals", "No. of Originals (e.g. 3/3)", "String", "FIND");
                } else if doc_type == "AWB" {
                    set("declared_value_carriage", "Declared Value for Carriage", "String", "Optional");
                    set("declared_value_customs", "Declared Value for Customs", "String", "Optional");
                }
            }

            // --- 3. Customs (ED, ID, CO) ---
            else if ["ED", "ID"].contains(&doc_type) {
                set("document_number", "Export/Import Declaration No (신고번호)", "String", "FIND");
                set("declaration_date", "Declaration Date (신고일)", "String", "FIND");
                set("transaction_code", "Transaction Type Code", "String", "FIND");
                set("refund_code", "Drawback/Refund Code", "String", "Optional");
            } else if doc_type == "CO" {
                set("document_number", "Certificate No", "String", "FIND");
                set("reference_invoice", "Invoice No Ref", "String", "FIND");
                set("issuing_authority", "Issuing Authority", "String", "FIND");
            }
            
            // --- 4. Certificates & Special (IC, WC, CA, PHYTO, HC, DGD, MSDS, BIZ_LIC) ---
            else if ["IC", "WC", "CA", "PHYTO", "HC", "BEN_CERT"].contains(&doc_type) {
                set("document_number", "Certificate/Report No", "String", "FIND");
                set("related_ref", "Ref Invoice/LC No", "String", "FIND");
            } else if doc_type == "DGD" {
                set("shipper_reference_number", "Shipper's Ref", "String", "FIND");
                set("page_of_pages", "Page X of Y", "String", "FIND");
            } else if doc_type == "MSDS" {
                set("product_name", "Material/Product Name", "String", "FIND");
                set("cas_number", "CAS Registry Number", "String", "FIND");
                set("revision_date", "Revision Date", "String", "FIND");
            } else if doc_type == "BIZ_LIC" {
                set("registration_number", "Business Reg No", "String", "FIND");
                set("license_number", "License No", "String", "FIND");
                set("establishment_date", "Date of Est.", "String", "FIND");
            } else if doc_type == "INS" {
                set("policy_number", "Policy No", "String", "FIND");
                set("certificate_number", "Cert No", "String", "FIND");
            }
        },
        "parties" => {
            // --- Common Roles ---
            set("supplier_name", "Shipper, Exporter, Seller, Vendor", "String", "FIND");
            set("supplier_address", "Address of Supplier", "String", "FIND");
            set("buyer_name", "Consignee, Importer, Buyer", "String", "FIND");
            set("buyer_address", "Address of Buyer", "String", "FIND");

            if ["SC", "PO", "PI", "CI", "CINV"].contains(&doc_type) {
                if doc_type == "SC" {
                    set("seller_name", "Seller Name (if diff from header)", "String", "FIND");
                    set("buyer_name", "Buyer Name", "String", "FIND");
                }
                set("manufacturer_name", "Manufacturer Name", "String", "Optional");
                set("supplier_tax_id", "Seller Tax ID / VAT", "String", "Optional");
                set("buyer_tax_id", "Buyer Tax ID / VAT", "String", "Optional");
                set("notify_party_name", "Notify Party (if any)", "String", "Optional");
                
                if doc_type == "CI" || doc_type == "PI" {
                    set("bank_name", "Beneficiary Bank", "String", "Optional");
                    set("bank_account", "Account No/IBAN", "String", "Optional");
                    set("bank_swift", "SWIFT Code", "String", "Optional");
                    set("authorized_signatory", "Authorized Signatory Name", "String", "FIND");
                }
            } else if ["ED", "ID"].contains(&doc_type) {
                set("exporter_name", "Exporter (수출자)", "String", "FIND");
                set("importer_name", "Importer (수입자)", "String", "FIND");
                set("declarant_name", "Declarant (신고인/관세사)", "String", "FIND");
                set("taxpayer_name", "Tax Payer (납세의무자)", "String", "FIND");
            } else if ["PHYTO", "HC"].contains(&doc_type) {
                set("consignor_name", "Name of Consignor", "String", "FIND");
                set("consignee_name", "Name of Consignee", "String", "FIND");
                set("inspection_org", "Inspection Organization", "String", "FIND");
            } else if doc_type == "POA" {
                set("grantor_name", "Grantor (Company/Person)", "String", "FIND");
                set("attorney_name", "Attorney-in-Fact", "String", "FIND");
            } else if doc_type == "BIZ_LIC" {
                set("company_name", "Company Name", "String", "FIND");
                set("representative_name", "Representative/Owner", "String", "FIND");
            } else if ["BL", "AWB", "SA"].contains(&doc_type) {
                set("notify_party_name", "Notify Party", "String", "FIND");
                set("forwarder_name", "Forwarding Agent", "String", "FIND");
                set("delivery_agent", "Delivery Agent at Dest", "String", "Optional");
            }
        },
        "logistics" => {
            // --- Common Logistics ---
            set("location_pol", "Port of Loading / Departure", "String", "FIND");
            set("location_pod", "Port of Discharge / Destination", "String", "FIND");
            set("etd", "Est. Departure Date", "String", "FIND");
            set("eta", "Est. Arrival Date", "String", "FIND");
            
            if ["BL", "AWB", "SA", "ED", "ID", "DGD", "CI", "PL"].contains(&doc_type) {
                set("vehicle_name", "Vessel Name / Flight No", "String", "FIND");
                set("voyage_number", "Voyage No", "String", "FIND");
                set("transport_mode", "Sea, Air, Road, Rail", "String", "FIND");
                
                if ["BL", "AWB"].contains(&doc_type) {
                    set("shipped_on_board_date", "Shipped on Board Date", "String", "FIND");
                    set("place_of_receipt", "Place of Receipt", "String", "Optional");
                    set("place_of_delivery", "Place of Delivery", "String", "Optional");
                    set("movement_type", "CY/CY, CFS/CFS", "String", "Optional");
                }
                
                if doc_type == "SA" {
                     set("bl_number", "B/L Number Ref", "String", "FIND");
                     set("container_count", "Total Containers", "Number", "FIND");
                } else if doc_type == "DGD" {
                    set("port_of_loading", "POL", "String", "FIND");
                    set("port_of_discharge", "POD", "String", "FIND");
                }
            }
            if ["PHYTO", "HC"].contains(&doc_type) {
                set("means_of_conveyance", "Means of Conveyance", "String", "FIND");
                set("point_of_entry", "Declared Point of Entry", "String", "FIND");
            }
            if doc_type == "PO" {
                set("ship_via", "Shipping Method", "String", "FIND");
                set("delivery_date", "Requested Delivery Date", "String", "FIND");
                set("ship_to_location", "Ship To Address", "String", "FIND");
            }
            if ["DO", "AN"].contains(&doc_type) {
                set("pickup_location", "Pickup / CY Location", "String", "FIND");
                set("empty_return_location", "Empty Return Depo", "String", "FIND");
                set("validity_date", "Valid Until", "String", "FIND");
                
                if doc_type == "AN" {
                     set("free_time_detention", "Detention Free Time", "String", "Optional");
                     set("free_time_demurrage", "Demurrage Free Time", "String", "Optional");
                }
            }
            if doc_type == "CO" {
                set("transport_details", "Transport Details (Vessel/Route)", "String", "Optional");
            }
        },
        "financials" => {
            set("currency_code", "Currency (USD, EUR, KRW)", "String", "FIND");
            
            if ["CI", "PI", "SC", "PO", "CINV", "LC"].contains(&doc_type) {
                set("amount_total", "Total Amount", "Number", "FIND");
                set("amount_subtotal", "Subtotal", "Number", "FIND");
                set("amount_tax", "Tax/VAT Amount", "Number", "FIND");
                set("amount_freight", "Freight Charges", "Number", "FIND");
                set("amount_insurance", "Insurance Charges", "Number", "FIND");
                set("amount_discount", "Discount Amount", "Number", "Optional");
                set("amount_packing", "Packing Charges", "Number", "Optional");
                set("incoterms", "Incoterms (FOB, CIF)", "String", "FIND");
                
                if doc_type == "CINV" {
                    set("market_value", "Fair Market Value", "Number", "Optional");
                }
            } else if ["ED", "ID"].contains(&doc_type) {
                set("exchange_rate", "Exchange Rate", "Number", "FIND");
                set("total_customs_value", "Total Customs Value (과세가격)", "Number", "FIND");
                set("total_tax_amount", "Total Tax (관세+부가세)", "Number", "FIND");
                set("freight_krw", "Freight (KRW)", "Number", "Optional");
                set("insurance_krw", "Insurance (KRW)", "Number", "Optional");
            } else if doc_type == "BIZ_LIC" {
                set("capital_amount", "Capital Amount", "Number", "FIND");
            } else if doc_type == "AN" {
                set("local_charges_total", "Total Local Charges", "Number", "FIND");
                set("exchange_rate", "Ex Rate", "Number", "Optional");
            }
        },
        "conditions" => {
            // --- Contracts ---
            if ["SC", "PO", "PI", "CI"].contains(&doc_type) {
                set("payment_terms", "T/T, L/C, Net30", "String", "FIND");
                set("shipping_terms", "Partial Shipment, Transshipment", "String", "FIND");
                set("packing_terms", "Packing Conditions", "String", "Optional");
                set("reason_for_export", "Reason (Sale, Sample, Return)", "String", "Optional");
                
                if doc_type == "SC" {
                    set("claim_period", "Claim Period", "String", "FIND");
                    set("arbitration", "Arbitration Clause", "String", "Optional");
                    set("governing_law", "Governing Law", "String", "Optional");
                }
            } 
            // --- Transport ---
            else if ["BL", "AWB"].contains(&doc_type) {
                set("freight_payment_term", "Freight Prepaid / Collect", "String", "FIND");
                set("place_of_issue", "Place of Issue", "String", "FIND");
                if doc_type == "AWB" {
                     set("handling_information", "Handling Info", "String", "Optional");
                }
            }
            // --- Certs ---
            else if ["PHYTO", "HC"].contains(&doc_type) {
                set("treatment_type", "Chemical/Heat Treatment", "String", "FIND");
                set("chemical_used", "Chemical Name", "String", "FIND");
                set("duration_temperature", "Duration & Temp", "String", "FIND");
                set("additional_declaration", "Add. Declaration", "String", "FIND");
            }
            else if doc_type == "CO" {
                set("origin_criterion", "Origin Criterion (P, CTC, PSR)", "String", "FIND");
                set("remarks", "Remarks", "String", "Optional");
            }
            else if doc_type == "LC" {
                 set("tenor", "Drafts at (Tenor)", "String", "FIND");
                 set("confirmation_instructions", "Confirm/Without", "String", "Optional");
                 set("partial_shipment", "Partial Shipment (Allowed/Prohibited)", "String", "FIND");
                 set("transshipment", "Transshipment (Allowed/Prohibited)", "String", "FIND");
                 set("documents_required", "Documents Required List", "String", "CAPTURE");
                 set("additional_conditions", "Additional Conditions", "String", "CAPTURE");
                 set("charges_narrative", "Charges (Who pays?)", "String", "FIND");
            }
            else if doc_type == "POA" {
                set("scope_of_authority", "Authorized Powers", "String", "FIND");
                set("validity_period", "Valid From/To", "String", "FIND");
            }
        },
        "cargo" => {
            set("package_count", "Total Packages", "Number", "FIND");
            set("package_unit", "Unit (CTN, PLT, PKG)", "String", "FIND");
            set("weight_gross", "Gross Weight (kg)", "Number", "FIND");
            set("weight_net", "Net Weight (kg)", "Number", "FIND");
            set("volume", "Volume (CBM)", "Number", "FIND");
            set("marks_and_numbers", "Marks & Nos", "String", "CAPTURE");
            
            if ["PL", "CI"].contains(&doc_type) {
                 set("net_net_weight", "Net Net Weight", "Number", "Optional");
            }

            if doc_type == "WC" {
                set("measured_weight", "Measured Weight", "Number", "FIND");
                set("weight_difference", "Difference", "Number", "Optional");
            } else if doc_type == "DGD" {
                set("flash_point", "Flash Point", "String", "FIND");
                set("marine_pollutant", "Marine Pollutant (Yes/No)", "String", "FIND");
                set("net_explosive_mass", "Net Explosive Mass", "String", "Optional");
            }
        },
        "items" => { // or line_items
            set("sequence_no", "Item No", "Number", "FIND");
            set("description", "Description of Goods", "String", "FIND");
            set("quantity", "Quantity", "Number", "FIND");
            set("unit", "Unit", "String", "FIND");
            
            if ["CI", "PI", "SC", "PO", "CINV", "ED", "ID"].contains(&doc_type) {
                set("hs_code", "HS Code", "String", "FIND");
                set("unit_price", "Unit Price", "Number", "FIND");
                set("total_price", "Total Price", "Number", "FIND");
                set("origin_country", "Country of Origin", "String", "Optional");
                
                if doc_type == "PO" {
                     set("sku", "SKU / Part No", "String", "FIND");
                     set("delivery_date", "Item Delivery Date", "String", "Optional");
                }
                if ["ED", "ID"].contains(&doc_type) {
                     set("tax_rate_customs", "Duty Rate (%)", "Number", "Optional");
                }
            }
            else if ["PL", "SA"].contains(&doc_type) {
                 set("package_type_detail", "Pkg Type (Inner/Outer)", "String", "Optional");
                 set("dimensions", "Dimensions (LxWxH)", "String", "Optional");
            }
            else if ["CA", "IC", "WC"].contains(&doc_type) {
                // For Analysis/Inspection, items are test parameters
                set("parameter_name", "Test Item / Parameter", "String", "FIND");
                set("specification", "Specification / Standard", "String", "FIND");
                set("result_value", "Result", "String", "FIND");
                set("test_method", "Test Method", "String", "Optional");
            }
            else if doc_type == "PHYTO" || doc_type == "HC" {
                set("scientific_name", "Scientific Name", "String", "FIND");
                set("number_of_packages", "No of Pkgs", "Number", "FIND");
            }
            else if doc_type == "DGD" {
                set("un_number", "UN No", "String", "FIND");
                set("proper_shipping_name", "Proper Shipping Name", "String", "FIND");
                set("class_division", "Class/Div", "String", "FIND");
                set("packing_group", "Packing Group", "String", "FIND");
                set("ems_no", "EMS No", "String", "Optional");
            }
        },
        "containers" => {
            set("container_number", "Container No", "String", "FIND");
            set("seal_number", "Seal No", "String", "FIND");
            set("type_size", "Size/Type (20GP, 40HC)", "String", "FIND");
            set("weight", "Payload Weight", "Number", "Optional");
        },
        _ => {}
    }

    if schema_fields.is_empty() { return "{}".to_string(); }
    
    let is_list = category == "items" || category == "containers";
    let cat_key = if category == "items" { "line_items" } else { category };
    let mut lines = Vec::new();
    if is_list { lines.push(format!("  \"{}\": [{{", cat_key)); } 
    else { lines.push(format!("  \"{}\": {{ ", cat_key)); }
    
    for (i, (key, desc, dtype, mode)) in schema_fields.iter().enumerate() {
        let default_val = match dtype.as_str() {
            "Number" => "0",
            "Boolean" => "false",
            _ => "\"\""
        };
        let comma = if i < schema_fields.len() - 1 { "," } else { "" };
        lines.push(format!("    \"{}\": {} // {}: {} {} ({}).", key, default_val, comma, mode, desc, dtype));
    }
    
    if is_list { lines.push("  }]".to_string()); } 
    else { lines.push("  }".to_string()); }
    lines.join("\n")
}

fn extract_json_from_text(text: &str) -> Option<Value> {
    let re = Regex::new(r"(?s)(\[.*\]|\{.*\})").ok()?;
    let caps = re.captures(text)?;
    let raw = caps.get(1)?.as_str();
    let clean = raw.replace("```json", "").replace("```", "").trim().to_string();
    serde_json::from_str(&clean).ok()
}

fn extract_json_field(text: &str, field_name: &str) -> Option<String> {
    let val = extract_json_from_text(text)?;
    val.get(field_name).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn merge_json_manual(root: &mut Map<String, Value>, cat: &str, data: Value) {
    let target_key = if cat == "items" { "line_items" } else if cat == "containers" { "containers" } else { cat };
    
    // Some models might wrap the result in the category name or target_key
    let actual_data = if let Some(inner) = data.get(target_key) { inner.clone() } 
                      else if let Some(inner) = data.get(cat) { inner.clone() } 
                      else { data };

    if let Some(target) = root.get_mut(target_key) {
        if target.is_array() {
            let target_arr = target.as_array_mut().unwrap();
            if let Some(source_arr) = actual_data.as_array() {
                for new_item in source_arr {
                    // Check for duplicates in line_items/containers by description/number
                    let is_dup = if target_key == "line_items" {
                        let new_desc = new_item.get("description").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("description").and_then(|v| v.as_str()).unwrap_or("") == new_desc)
                    } else if target_key == "containers" {
                        let new_no = new_item.get("container_number").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("container_number").and_then(|v| v.as_str()).unwrap_or("") == new_no)
                    } else { false };

                    if !is_dup { target_arr.push(new_item.clone()); }
                }
            }
        } else if let Some(target_obj) = target.as_object_mut() {
            if let Some(source_obj) = actual_data.as_object() {
                for (k, v) in source_obj {
                    if !v.is_null() && v != "" && v != 0 { target_obj.insert(k.clone(), v.clone()); }
                }
            }
        }
    }
}