use crate::utils;
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
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;

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
    generator: Arc<Mutex<Option<Qwen3VLGenerateModel>>>,
    embedding_model: Arc<Mutex<Option<EmbeddingModel>>>,
    
    pub is_cpu_mode: bool, // Moved up
    
    // Config for Lazy Reloading
    model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    dtype: Option<DType>, 
}
impl LogisModel {
    pub fn unload_generator(&self) {
        let mut gen = self.generator.lock().unwrap();
        if gen.is_some() {
            *gen = None;
            println!("[MODEL] Generator (Qwen) unloaded to free VRAM.");
        }
    }

    pub fn unload_embedding(&self) {
        let mut emb = self.embedding_model.lock().unwrap();
        if emb.is_some() {
            *emb = None;
            println!("[MODEL] Embedding Model unloaded to free VRAM.");
        }
    }

    async fn load_generator_internal(&self) -> anyhow::Result<Qwen3VLGenerateModel> {
        println!("[MODEL] Loading Generator (Qwen)...");
        let m_path = self.model_path.clone();
        let dev = self.device_config.device.clone();
        
        // Extract device ID if possible (simplified)
        let mut dev_id = 0;
        if let Some(id_str) = self.device_config.name.split('-').nth(1) {
             if let Ok(id) = id_str.parse::<usize>() { dev_id = id; }
        }

        let dtype = if self.device_config.is_cpu { Some(DType::F32) } else { Some(DType::BF16) };
        let limit = self.max_tokens_limit;

        let generator = tokio::task::spawn_blocking(move || {
            Qwen3VLGenerateModel::init(&m_path, Some(&dev), dev_id, Some(&dev), dev_id, dtype, Some(limit as usize))
        }).await??;
        
        Ok(generator)
    }

    fn load_embedding_internal(&self) -> anyhow::Result<EmbeddingModel> {
        println!("[MODEL] Loading Embedding Model (Gemma)...");
        // Embedding model is forced to CPU in original logic, or we can follow config.
        // Original logic forced CPU to save VRAM. With exclusive loading, we *could* use GPU if we wanted,
        // but keeping it on CPU is safer for now or we can use the device config.
        // Let's stick to the device config to be consistent, or CPU if preferred.
        // For now, let's follow the device config but maybe force CPU if VRAM is tight?
        // Actually, exclusive loading allows us to use GPU for embedding too!
        
        // However, EmbeddingModel implementation in this codebase might be CPU optimized or
        // the `new` method in embedding.rs might have hardcoded CPU. 
        // Let's check embedding.rs... It says "Forcing CPU". 
        // So we just call new.
        
        EmbeddingModel::new(&self.embedding_path)
    }

    pub async fn ensure_generator(&self) -> anyhow::Result<()> {
        // 1. Unload Embedding if loaded
        self.unload_embedding();

        // 2. Load Generator if not loaded (Check-Load-Store pattern)
        let need_load = {
            let gen_guard = self.generator.lock().unwrap();
            gen_guard.is_none()
        };

        if need_load {
            let gen = self.load_generator_internal().await?;
            let mut gen_guard = self.generator.lock().unwrap();
            *gen_guard = Some(gen);
        }
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        // 1. Unload Generator if loaded
        self.unload_generator();

        // 2. Load Embedding if not loaded (Check-Load-Store pattern)
        let need_load = {
            let emb_guard = self.embedding_model.lock().unwrap();
            emb_guard.is_none()
        };

        if need_load {
            let self_clone = self.embedding_path.clone();
            let emb = tokio::task::spawn_blocking(move || {
                EmbeddingModel::new(&self_clone)
            }).await??;
            
            let mut emb_guard = self.embedding_model.lock().unwrap();
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    pub async fn new(device_preference: Option<&str>) -> anyhow::Result<Self> {
        println!("[MODEL-00] Initializing LogisModel with Lazy Loading Configuration");

        let mut config = utils::get_optimal_device_config();
        
        // Override if CPU preference is explicit
        if device_preference == Some("cpu") {
            println!("[MODEL] Explicit CPU preference detected.");
            config = utils::DeviceConfig {
                device: Device::Cpu,
                is_cpu: true,
                classify_chunk_size: 4000,
                extract_chunk_size: 4000,
                name: "CPU-Forced".to_string(),
            };
        }

        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        let gguf_dir = base_path.join("Qwen3-VL-2B-Instruct-gguf");
        let model_path = gguf_dir.to_str().unwrap().to_string();
        let embedding_path = base_path.join("embeddinggemma-300m");

        let max_tokens_limit = 2048; // [OPTIMIZED] Perfectly matched with embedding limit and 1000-char chunks

        // DO NOT LOAD MODELS HERE. Just return the config.
        Ok(Self {
            generator: Arc::new(Mutex::new(None)),
            embedding_model: Arc::new(Mutex::new(None)),
            is_cpu_mode: config.is_cpu, // Moved up
            model_path,
            embedding_path,
            device_config: config.clone(),
            max_tokens_limit: max_tokens_limit as u32,
            dtype: None, 
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
            "task_id": task_id.clone(),
            "category": "Image Loading", "summary": "Loading image...", "spinner": "⠋"
        }));

        // Ensure generator is loaded (and embedding is unloaded)
        self.ensure_generator().await?;

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

            let extracted_data = crate::parsing::parse_json_from_llm(&result_str);
            
            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            let item_digest = crate::utils::hash::digest(&nl);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                // Fixed identities for parity
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr); // Default team
                let hashed_cc = crate::utils::hash::hash_id("local.image");

                // [STRICT PARITY] Use stable hashing for image results
                let raw_no = extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id);
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(raw_no).replace("-", "").replace("_", "");
                
                // index = crc32(hashId(type + team + no))
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_no)));
                // id = hashId(team + index)
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                
                // ref = hashId(team + cc + no)
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));

                let mut final_data = extracted_data.clone();
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));

                let _ = db.upsert_item(
                    "commerce_tracking", 
                    &hashed_id, 
                    "tracking", 
                    final_data, 
                    None,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("tracking{}", hashed_cc))), // bcc
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
            }
            
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            }));
            
            Ok(())
        } else {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Error", "summary": "Failed to load image file.", "spinner": "❌"
            }));
            Ok(())
        }
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator().await?;
        
        let self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        tokio::task::spawn_blocking(move || {
            let mut gen_guard = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
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
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator().await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // Emit initial status once (Frontend will handle continuous spinner animation via .active-spinner class)
        let _ = app_handle.emit(event_name, base_payload);

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || {
            let mut gen_guard = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
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
        // Ensure generator is loaded
        self.ensure_generator().await?;

        // Emit initial status once
        let _ = app_handle.emit(event_name, base_payload);

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || {
            let mut gen_guard = self_clone.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
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

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
    }

    async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        self.ensure_generator().await?;
        
        let mut gen_guard = self.generator.lock().map_err(|_| anyhow!("Poisoned lock"))?;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
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
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator().await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // Emit initial status once
        let _ = app_handle.emit(event_name, base_payload);

        let generator_arc = self.generator.clone();
        let max_tok = self.max_tokens_limit;
        
        // Spawn the heavy task using Tokio directly for standard behavior
        let task = tokio::task::spawn_blocking(move || {
            let mut gen_guard = generator_arc.lock().map_err(|_| anyhow!("Poisoned lock"))?;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
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
        
        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
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
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;

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
        let segments = crate::parsing::parse_json_from_llm(&res1);
        
        // Stage 2: Extract conditions for each segment (graph2contexts) in ONE BATCH
        let mut final_contexts = Vec::new();
        if let Some(ctx_arr) = segments.get("context").and_then(|v: &Value| v.as_array()) {
            // Combine all segments into one batch request
            let mut combined_segments = String::new();
            for (idx, seg) in ctx_arr.iter().enumerate() {
                let seg_text = seg.get("text").and_then(|v: &Value| v.as_str()).unwrap_or("");
                combined_segments.push_str(&format!("Segment #{}: {}\n", idx + 1, seg_text));
            }

            if !combined_segments.is_empty() {
                let prompt2 = crate::parsing::graph2contexts(&current_time);
                // Using persistent session for schema caching
                let res2 = self.chat("", &format!("{}\n\nInput Segments:\n{}", prompt2, combined_segments), None, Some("system_search_g2c".to_string())).await?;
                let mut batch_info = crate::parsing::parse_json_from_llm(&res2);
                
                // Process results and ensure type parity
                if let Some(res_arr) = batch_info.get_mut("context").and_then(|v: &mut Value| v.as_array_mut()) {
                    for (i, item) in res_arr.iter_mut().enumerate() {
                        // Match with original segment types if LLM lost them in batch
                        if let Some(original_seg) = ctx_arr.get(i) {
                            if item.get("type").is_none() || item.get("type").and_then(|v: &Value| v.as_str()) == Some("") {
                                if let Some(item_obj) = item.as_object_mut() {
                                    item_obj.insert("type".to_string(), original_seg.get("type").cloned().unwrap_or(json!("")));
                                }
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
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            let _ = app_handle.emit("research-update", json!({ "text": status_history, "spinner": "✅" }));
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None).await?;
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