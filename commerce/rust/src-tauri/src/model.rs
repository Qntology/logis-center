use anyhow::anyhow;
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
use crate::models::embedding::EmbeddingModel;
use crate::openai_types::{
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
use candle_core::{Device, DType};
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use regex::Regex;
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use sysinfo::System;
use nvml_wrapper::Nvml;

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

pub fn generate_rich_summary(doc_type: &str, data: &Value) -> String {
    let type_map = json!({
        "CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List",
        "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit",
        "tracking": "Shipping Label / Tracking Info"
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

    if doc_type == "tracking" {
        if let Some(tn) = data.get("tracking_number").and_then(|s| s.as_str()) {
            parts.push(format!("The tracking number is {}.", tn));
        }
        if let Some(text) = data.get("text").and_then(|s| s.as_str()) {
            parts.push(text.to_string());
        }
    }

    if let Some(p) = data.get("parties") {
        let sup = p.get("supplier_name").and_then(|s| s.as_str());
        let buy = p.get("buyer_name").and_then(|s| s.as_str());
        
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

pub struct LogisModel {
    generator: Arc<Mutex<Qwen3VLGenerateModel>>,
    embedding_model: Arc<Mutex<Option<EmbeddingModel>>>,
    is_cpu_mode: bool,
    max_tokens_limit: u32,
}

impl LogisModel {
    // Returns (Is Safe to Try GPU, Message, Usable VRAM in Bytes, Device ID)
    fn check_memory_status(is_cuda: bool, _is_metal: bool) -> (bool, String, u64, Vec<usize>) {
        if is_cuda {
            let nvml = match Nvml::init() {
                Ok(n) => n,
                Err(e) => return (false, format!("NVML Init Failed: {}", e), 0, vec![0]),
            };
            
            let count = nvml.device_count().unwrap_or(1);
            let mut gpu_ids = Vec::new();
            let mut max_free = 0;

            for i in 0..count {
                if let Ok(device) = nvml.device_by_index(i) {
                    if let Ok(mem) = device.memory_info() {
                        gpu_ids.push(i as usize);
                        if mem.free > max_free { max_free = mem.free; }
                    }
                }
            }
            return (true, format!("Found {} GPUs", count), max_free, gpu_ids);
        }

        let mut sys = System::new_all();
        sys.refresh_memory();
        let free_mem = sys.available_memory();
        (true, "CPU/Metal".to_string(), free_mem, vec![0])
    }

    pub async fn new(device_preference: Option<&str>) -> anyhow::Result<Self> {
        println!("[MODEL-00] Initializing LogisModel with Dual-GPU Support");

        let has_cuda = candle_core::utils::cuda_is_available();
        let has_metal = candle_core::utils::metal_is_available();
        
        let mut text_device = Device::Cpu;
        let mut vision_device = Device::Cpu;
        let mut text_device_id = 0;
        let mut vision_device_id = 0;
        let mut _detected_vram = 0_u64;

        if device_preference != Some("cpu") {
            if has_cuda {
                let (ok, _, vram, ids) = Self::check_memory_status(true, false);
                _detected_vram = vram;
                if ok {
                    text_device_id = ids[0];
                    text_device = Device::new_cuda(text_device_id).unwrap_or(Device::Cpu);
                    if ids.len() >= 2 {
                        vision_device_id = ids[1];
                        vision_device = Device::new_cuda(vision_device_id).unwrap_or(Device::Cpu);
                        println!("✅ [MULTI-GPU] Text: cuda:{}, Vision: cuda:{}", text_device_id, vision_device_id);
                    } else {
                        vision_device_id = text_device_id;
                        vision_device = text_device.clone();
                        println!("ℹ️ [SINGLE-GPU] Text & Vision sharing cuda:{}", text_device_id);
                    }
                }
            } else if has_metal {
                text_device = Device::new_metal(0).unwrap_or(Device::Cpu);
                vision_device = text_device.clone();
            }
        }

        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        let gguf_dir = base_path.join("Qwen3-VL-2B-Instruct-gguf");
        let model_path = gguf_dir.to_str().unwrap().to_string();

        let max_tokens_limit = 2048; // Optimized for 2.4GB VRAM stability
        let dtype = if text_device.is_cpu() { Some(DType::F32) } else { Some(DType::BF16) };

        let t_dev = text_device.clone();
        let v_dev = vision_device.clone();
        let m_path = model_path.clone();

        let generator = tokio::task::spawn_blocking(move || {
            Qwen3VLGenerateModel::init(&m_path, Some(&t_dev), text_device_id, Some(&v_dev), vision_device_id, dtype, Some(max_tokens_limit as usize))
        }).await??;

        let embedding_model_path = base_path.join("embeddinggemma-300m");
        let embedding_model = if embedding_model_path.exists() {
            EmbeddingModel::new(&embedding_model_path).ok()
        } else { None };

        Ok(Self {
            generator: Arc::new(Mutex::new(generator)),
            embedding_model: Arc::new(Mutex::new(embedding_model)),
            is_cpu_mode: text_device.is_cpu(),
            max_tokens_limit: max_tokens_limit as u32,
        })
    }

    pub async fn extract_from_image(
        &self,
        task_id: String,
        image_path: String,
        language: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        let _ = app_handle.emit("extraction-progress", json!({ 
            "category": "Image Loading", "summary": "Loading image...", "spinner": "⠋"
        }));

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            let prompt = get_image_extraction_prompt("kr", &language, "tracking", "");
            
            let result_str = self.chat_with_image_spinner(
                prompt, 
                Some(dynamic_image), 
                app_handle, 
                "extraction-progress", 
                json!({ "category": "Vision Analysis", "summary": "Analyzing image content..." }), 
                1024, 
                cancel_token.clone(), 
                Some(task_id.clone())
            ).await?;

            let extracted_data = crate::scheduler::parse_json_from_llm(&result_str);
            
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let id = extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id).to_string();
                let _ = db.upsert_item("commerce_tracking", &id, "tracking", extracted_data.clone(), None).await;
            }
            
            let _ = app_handle.emit("extraction-progress", json!({ 
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            }));
            
            Ok(())
        } else {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "category": "Error", "summary": "Failed to load image file.", "spinner": "❌"
            }));
            Ok(())
        }
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        let self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
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
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))?;
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
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let content_parts = vec![
            ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
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

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || {
            let mut gen = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
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
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
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
            
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
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

    fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
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
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>, 
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        let generator_arc = self.generator.clone();
        let max_tok = self.max_tokens_limit;
        
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
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
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
                    let _ = app_handle.emit(event_name, payload);
                }
            }
        }
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            prompt, 
            Some(master_img), 
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::scheduler::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        let embedding_model_arc = self.embedding_model.clone();
        
        tokio::task::spawn_blocking(move || {
            let guard = embedding_model_arc.lock().unwrap();
            if let Some(model) = guard.as_ref() {
                model.embed(&text).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 768])
            }
        }).await?
    }

    pub async fn parse_query_structured(&self, query: String, language: &str) -> anyhow::Result<Value> {
        let current_time = chrono::Utc::now().to_rfc3339();
        
        // Stage 1: Segment query (para2graph) - Using persistent session for schema caching
        let prompt1 = crate::parsing::para2graph(language);
        let res1 = self.chat("", &format!("{}\n\nQuery: {}", prompt1, query), None, Some("system_search_p2g".to_string())).await?;
        let segments = crate::scheduler::parse_json_from_llm(&res1);
        
        // Stage 2: Extract conditions for each segment (graph2contexts) in ONE BATCH
        let mut final_contexts = Vec::new();
        if let Some(ctx_arr) = segments.get("context").and_then(|v| v.as_array()) {
            // Combine all segments into one batch request
            let mut combined_segments = String::new();
            for (idx, seg) in ctx_arr.iter().enumerate() {
                let seg_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
                combined_segments.push_str(&format!("Segment #{}: {}\n", idx + 1, seg_text));
            }

            if !combined_segments.is_empty() {
                let prompt2 = crate::parsing::graph2contexts(&current_time);
                // Using persistent session for schema caching
                let res2 = self.chat("", &format!("{}\n\nInput Segments:\n{}", prompt2, combined_segments), None, Some("system_search_g2c".to_string())).await?;
                let mut batch_info = crate::scheduler::parse_json_from_llm(&res2);
                
                // Process results and ensure type parity
                if let Some(res_arr) = batch_info.get_mut("context").and_then(|v| v.as_array_mut()) {
                    for (i, item) in res_arr.iter_mut().enumerate() {
                        // Match with original segment types if LLM lost them in batch
                        if let Some(original_seg) = ctx_arr.get(i) {
                            if item.get("type").is_none() || item.get("type").and_then(|v| v.as_str()) == Some("") {
                                item.as_object_mut().unwrap().insert("type".to_string(), original_seg.get("type").cloned().unwrap_or(json!("")));
                            }
                        }
                    }
                    final_contexts.extend(res_arr.clone());
                }
            }
        }
        
        Ok(json!({ "context": final_contexts }))
    }

    // --- Ported from Python (search_engine.py) ---
    // --- Ported from Python (logic.py) ---
    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
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
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None)?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": "✅" }));
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None)?;
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

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" {
        let template = r###"[TASK]
Convert the shipping label image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number. It should be selected from numbers matching barcodes or QR codes, filtered by region, excluding telephone formats or order numbers.
2. Set recipient_match to true if the label address matches the context address (ignoring floor levels).
3. Extract all visible barcodes into an array.
4. Provide a text summary in {LANGUAGE}, masking the address to District-level and up. Do not mention masking.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "tracking_number": "string",
    "recipient_match": boolean,
    "barcodes": ["string"],
    "text": "string"
}"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
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