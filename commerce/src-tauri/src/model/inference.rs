use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use anyhow::anyhow;
use image::DynamicImage;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use crate::model::ModelSize;
use crate::openai_types::*;

impl crate::model::LogisModel {

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [FIX] Default to Qwen (0.6B) for all chat tasks
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // 🌟 [VISION-JIT] chat 은 ChatCompletionRequestMessageContentPart::Text 만 조립하는
        //    순수 텍스트 경로입니다. 비전 가중치가 붙어 있다면 여기서 반환합니다.
        {
            let mut gen_guard = self.generator.lock().await;
            if let Some(gen) = gen_guard.as_mut() {
                if gen.is_vision_jit_capable() && gen.vision_resident() {
                    let _ = gen.set_vision_active(false);
                }
            }
        }
        
        let _self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        {
            let mut gen_guard = self.generator.lock().await;
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
                model: "qwen".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }
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
        session_id: Option<String>,
        kv_name: Option<String>
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
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id, kv_name).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [FIX] Ensure we stay on Qwen (0.6B).
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(app_handle, task_id, &base_payload);
        }

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn chat_with_image_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        _base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Removed redundant emit. Only log the progress if needed.
        // let _ = app_handle.emit(event_name, base_payload);

        let mut gen_guard = self.generator.lock().await;
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
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?; // 🌟 무조건 Qwen으로 로드
        
        let mut gen_guard = self.generator.lock().await;
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
            model: "qwen".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        system: &str,       // 🌟 추가
        user_input: &str,   // 🌟 변경
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(_app_handle, task_id, &base_payload);
        }

        let max_tok = self.max_tokens_limit;
        
        let mut gen_guard = self.generator.lock().await;
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
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tok),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = crate::prompts::get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            "You are a highly precise document data extraction assistant.", // 🌟 System 주입
            &prompt,                                                        // 🌟 User 주입
            Some(master_img),
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        // 🌟 1. 메모리 캐시부터 확인합니다 (중복된 텍스트면 GPU 연산 원천 차단)
        {
            let cache = self.embedding_cache.lock().await;
            if let Some(vector) = cache.get(&text) {
                return Ok(vector.clone());
            }
        }

        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;

        let embedding_model_arc = self.embedding_model.clone();
        let text_clone = text.clone();
        
        let vector = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text_clone).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 384])
            }
        }).await??;

        // 🌟 2. 새로 연산된 벡터를 해시맵 캐시에 저장하여 다음 루프 때 재사용합니다.
        {
            let mut cache = self.embedding_cache.lock().await;
            cache.insert(text, vector.clone());
        }

        Ok(vector)
    }

    pub async fn get_embedding_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.ensure_embedding().await?;
        let embedding_model_arc = self.embedding_model.clone();
        
        let vectors = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed_batch(&texts).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                Ok(vec![vec![0.0; 384]; texts.len()])
            }
        }).await??;

        Ok(vectors)
    }
}