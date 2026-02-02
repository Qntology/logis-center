use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;
use serde_json::{json, Value};
use anyhow::{Result, anyhow};
use std::sync::atomic::AtomicBool;
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
use crate::models::native_embedding::NativeEmbeddingModel;
use crate::utils;
use sysinfo::System;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Small, 
    Large,
}

#[derive(Clone)]
pub struct LogisModel {
    pub generator: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>,
    pub embedding_model: Arc<TokioMutex<Option<NativeEmbeddingModel>>>,
    
    pub is_cpu_mode: bool, 
    pub use_native: bool,
    
    small_model_path: String,
    large_model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    current_size: Arc<TokioMutex<Option<ModelSize>>>,
}

impl LogisModel {
    pub async fn new(device_preference: Option<&str>) -> Result<Self> {
        let config = utils::get_optimal_device_config();
        let use_native = device_preference == Some("native") || device_preference.is_none();
        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        
        Ok(Self {
            generator: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: config.is_cpu,
            use_native,
            small_model_path: base_path.join("Qwen3-0.6B-Instruct-gguf").to_str().unwrap().to_string(),
            large_model_path: base_path.join("Qwen3-VL-2B-Instruct-gguf").to_str().unwrap().to_string(),
            embedding_path: base_path.join("embeddinggemma-300m"),
            device_config: config,
            max_tokens_limit: 65536,
            current_size: Arc::new(TokioMutex::new(None)),
        })
    }

    pub async fn secure_vram_relay(&self, target_size: ModelSize, task_id: Option<&str>, _cancel_token: Option<Arc<AtomicBool>>) -> Result<()> {
        println!("[RELAY] Securing context for {:?} (Task: {:?})", target_size, task_id);
        
        // 1. [PURGE] Clear current generator to free RAM/VRAM
        self.unload_generator().await;
        
        // 2. [LOAD] Load new model
        self.ensure_generator(target_size).await?;
        
        // 3. [RESTORE] Load KV from disk if task_id provided
        if let Some(tid) = task_id {
            let kv_path = utils::paths::get_kv_dir(None).join(tid);
            if kv_path.exists() {
                let mut gen_guard = self.generator.lock().await;
                if let Some(gen) = gen_guard.as_mut() {
                    gen.load_kv_from_disk(&kv_path)?;
                }
            }
        }
        
        Ok(())
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> Result<()> {
        let mut current_size_guard = self.current_size.lock().await;
        let mut gen_guard = self.generator.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() { return Ok(()); }

        let path = if size == ModelSize::Small { &self.small_model_path } else { &self.large_model_path };
        println!("[MODEL] Activating Native Engine for {:?}...", size);
        
        let gen = Qwen3VLGenerateModel::init_with_config(
            path, None, None, None, 0, None, 0, None, Some(self.max_tokens_limit as usize), false
        )?;

        *gen_guard = Some(gen);
        *current_size_guard = Some(size);
        Ok(())
    }

    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut size = self.current_size.lock().await;
        *size = None;
        
        // [CRITICAL] Memory Flush
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
            let process = GetCurrentProcess();
            let _ = SetProcessWorkingSetSizeEx(process, usize::MAX, usize::MAX, 0x00000001 | 0x00000002);
        }
    }

    pub async fn ensure_embedding(&self) -> Result<()> {
        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let emb = NativeEmbeddingModel::load(&self.embedding_path)?;
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    pub async fn get_embedding(&self, text: String) -> Result<Vec<f32>> {
        self.ensure_embedding().await?;
        let guard = self.embedding_model.lock().await;
        let model = guard.as_ref().ok_or_else(|| anyhow!("Embedding model failed"))?;
        model.embed(&text)
    }

    pub async fn chat(&self, _system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, _session_id: Option<String>) -> Result<String> {
        self.ensure_generator(ModelSize::Large).await?;
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator unloaded"))?;
        
        let params = crate::openai_types::ChatCompletionParameters {
            messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage {
                content: crate::openai_types::ChatCompletionRequestUserMessageContent::String(user_input.to_string()),
                name: None,
            })],
            ..Default::default()
        };
        gen.generate(params, cancel_token, None)
    }

    pub async fn save_kv_snapshot(&self, session_id: &str) -> Result<()> {
        let path = utils::paths::get_kv_dir(None).join(session_id);
        let mut gen_guard = self.generator.lock().await;
        if let Some(gen) = gen_guard.as_mut() { gen.save_kv_to_disk(&path)?; }
        Ok(())
    }
}