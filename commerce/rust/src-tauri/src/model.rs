use crate::utils;
use anyhow::anyhow;
use crate::models::embedding::EmbeddingModel;
use crate::openai_types::{
    ChatCompletionParameters,
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageContent,
};
use candle_core::{Device, DType};
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, atomic::AtomicBool};
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
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

use tokio::sync::Mutex as TokioMutex;
use std::time::{Duration, Instant};

use crate::models::qwen35::generate::Qwen3_5GenerateModel;

#[derive(Clone)]
pub enum ModelVariant {
    Qwen3_5(Arc<TokioMutex<Qwen3_5GenerateModel>>),
}

impl ModelVariant {
    pub async fn generate(&self, params: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, _session_id: Option<String>, _kv_name: Option<String>) -> anyhow::Result<String> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let mut gen = m.lock().await;
                gen.generate(params, cancel_flag).await
            }
        }
    }

    pub async fn clear_kv_cache(&self) -> anyhow::Result<()> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let mut gen = m.lock().await;
                gen.qwen3_5.clear_cache();
                Ok(())
            }
        }
    }

    pub async fn drop_kv_storage(&self) -> anyhow::Result<()> { Ok(()) }
    pub fn save_kv_to_disk(&self, _path: &std::path::Path, _kv_name: Option<&str>, _offset: usize) -> anyhow::Result<()> { Ok(()) }
    pub fn load_kv_from_disk(&self, _path: &std::path::Path, _kv_name: Option<&str>) -> anyhow::Result<()> { Ok(()) }
    pub fn truncate_kv_cache(&self, _len: usize) -> anyhow::Result<()> { Ok(()) }
    pub async fn prefill_chunk(&self, _text: String, _cancel_token: Option<Arc<AtomicBool>>, _kv_name: Option<String>) -> anyhow::Result<usize> { Ok(0) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Small, 
    Large, 
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<ModelVariant>>>, 
    pub small_hibernation: Arc<TokioMutex<Option<ModelVariant>>>,
    pub large_hibernation: Arc<TokioMutex<Option<ModelVariant>>>,
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    
    pub is_cpu_mode: bool, 
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    small_model_path: String,
    large_model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    _dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,
}

impl LogisModel {
    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut size = self.current_size.lock().await;
        *size = None;
        println!("[MODEL] All generators destroyed.");
    }

    pub async fn deep_purge_resources(&self) {
        {
            let mut gen = self.generator.lock().await;
            if let Some(g) = gen.take() {
                let _ = g.clear_kv_cache().await;
                drop(g); 
            }
        }
        if !self.is_cpu_mode {
            let dev = self.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { let _ = dev.synchronize(); }
            }).await;
        }
        println!("[DIAG-PURGE] Purge Complete.");
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        let mut current_size_guard = self.current_size.lock().await;
        let mut gen_guard = self.generator.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() {
            return Ok(());
        }

        println!("[MODEL] Activating engine for size: {:?}", size);
        
        let path = if size == ModelSize::Small { &self.small_model_path } else { &self.large_model_path };
        let target_device = self.device_config.device.clone();
        let dtype = if target_device.is_cpu() { Some(DType::F32) } else { Some(DType::BF16) };
        let path_clone = path.to_string();

        let gen_35 = Qwen3_5GenerateModel::init(
            &path_clone,
            Some(&target_device),
            dtype,
            true
        )?;
        
        *gen_guard = Some(ModelVariant::Qwen3_5(Arc::new(TokioMutex::new(gen_35))));
        *current_size_guard = Some(size);
        
        Ok(())
    }

    pub async fn secure_vram_relay(&self, target_size: ModelSize, _task_id: Option<&str>, _cancel_token: Option<Arc<AtomicBool>>, _is_baking: bool, _kv_name: Option<String>) -> anyhow::Result<()> {
        self.ensure_generator(target_size).await
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        let current_size = { *self.current_size.lock().await };
        let target_device = if current_size.is_some() { Device::Cpu } else { self.device_config.device.clone() };

        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let path = self.embedding_path.clone();
            let emb = tokio::task::spawn_blocking(move || {
                EmbeddingModel::new_with_device(&path, &target_device)
            }).await??;
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        self.ensure_embedding().await?;
        let emb_arc = self.embedding_model.clone();
        tokio::task::spawn_blocking(move || {
            let guard = emb_arc.blocking_lock();
            if let Some(m) = guard.as_ref() { m.embed(&text) } else { Ok(vec![0.0; 768]) }
        }).await?
    }

    pub async fn new(app_handle: tauri::AppHandle, device_preference: Option<&str>) -> anyhow::Result<Self> {
        let mut config = utils::get_optimal_device_config();
        if device_preference == Some("cpu") {
            config.device = Device::Cpu;
            config.is_cpu = true;
        }

        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        let normalize_path = |path: std::path::PathBuf| -> String {
            let s = path.to_string_lossy().to_string();
            if s.starts_with(r"\\?\") { s[4..].to_string() } else { s }
        };

        let small_model_path = normalize_path(base_path.join("Qwen3.5-0.8B-Split"));
        let large_model_path = normalize_path(base_path.join("Qwen3.5-0.8B-Split"));
        let embedding_path = base_path.join("embeddinggemma-300m");

        Ok(Self {
            app_handle,
            generator: Arc::new(TokioMutex::new(None)),
            small_hibernation: Arc::new(TokioMutex::new(None)),
            large_hibernation: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: config.is_cpu,
            is_disk_swap: true,
            dual_mode_enabled: true, 
            small_model_path,
            large_model_path,
            embedding_path,
            device_config: config,
            max_tokens_limit: 4096,
            _dtype: None, 
            current_size: Arc::new(TokioMutex::new(None)),
        })
    }

    pub async fn extract_from_image(
        &self,
        _task_id: String,
        _image_path: String,
        _language: String,
        _app_handle: &tauri::AppHandle,
        _cancel_token: Option<Arc<AtomicBool>>,
        _store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        println!("[WARN] Image extraction is not supported in text-only mode.");
        Ok(())
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, _session_id: Option<String>, _kv_name: Option<String>) -> anyhow::Result<String> {
        self.ensure_generator(ModelSize::Small).await?;
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        let params = ChatCompletionParameters {
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage { content: system.to_string(), name: None }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                    content: ChatCompletionRequestUserMessageContent::Text(user_input.to_string()),
                    name: None 
                }),
            ],
            model: "qwen3.5".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            ..Default::default()
        };
        gen.generate(params, cancel_token, None, None).await
    }

    pub async fn parse_query_structured(&self, query: String, language: &str) -> anyhow::Result<Value> {
        let current_time = chrono::Utc::now().to_rfc3339();
        let prompt1 = crate::parsing::para2graph(language);
        let res1 = self.chat("", &format!("{}\n\nQuery: {}", prompt1, query), None, None, None).await?;
        let segments = crate::parsing::parse_json_from_llm(&res1);
        
        let mut final_contexts = Vec::new();
        if let Some(ctx_arr) = segments.get("context").and_then(|v| v.as_array()) {
            let mut combined = String::new();
            for (idx, seg) in ctx_arr.iter().enumerate() {
                let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
                combined.push_str(&format!("Segment #{}: {}\n", idx + 1, text));
            }
            if !combined.is_empty() {
                let prompt2 = crate::parsing::graph2contexts(&current_time);
                let res2 = self.chat("", &format!("{}\n\nInput Segments:\n{}", prompt2, combined), None, None, None).await?;
                let batch_info = crate::parsing::parse_json_from_llm(&res2);
                if let Some(res_arr) = batch_info.get("context").and_then(|v| v.as_array()) {
                    final_contexts.extend(res_arr.clone());
                }
            }
        }
        Ok(json!({ "context": final_contexts }))
    }

    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let prompt = format!("Given context: {}\n\nDeeply research: {}", context_data, query);
        let res = self.chat("", &prompt, cancel_token, None, None).await?;
        let _ = app_handle.emit("research-update", &res);
        Ok(res)
    }
}

pub fn generate_rich_summary(_doc_type: &str, _data: &Value) -> String {
    "Summary generated.".to_string()
}
