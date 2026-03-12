use crate::utils;
use anyhow::anyhow;
use crate::openai_types::{
    ChatCompletionParameters,
};
use crate::utils::config::{Config, ModelType};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tauri::Manager;

use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

#[derive(Clone)]
pub enum ModelVariant {
    Qwen3_5(Arc<TokioMutex<Qwen3_5GenerateModel>>),
    Qwen3Vl,
}

impl ModelVariant {
    pub async fn generate(
        &self,
        params: ChatCompletionParameters,
        cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> anyhow::Result<String> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let mut gen = m.lock().await;
                gen.generate(params, cancel_flag, session_id, kv_name).await
            }
            ModelVariant::Qwen3Vl => {
                Err(anyhow!("Qwen3 VL generation not yet implemented"))
            }
        }
    }

    pub async fn reset_cache(&self) -> anyhow::Result<()> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let mut gen = m.lock().await;
                gen.qwen3_5.reset_mamba_cache()?;
                Ok(())
            }
            _ => Ok(())
        }
    }
}

pub struct LogisModel {
    pub device: Device,
    pub model_path: String,
    pub tokenizer_path: String,
    pub embedding_path: String,
    
    pub current_size: Arc<TokioMutex<Option<ModelSize>>>,
    pub generator: Arc<TokioMutex<Option<ModelVariant>>>,
    pub small_hibernation: Arc<TokioMutex<Option<ModelVariant>>>,
    pub large_hibernation: Arc<TokioMutex<Option<ModelVariant>>>,
    
    pub is_cpu_mode: bool,
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    pub variant: ModelVariant, // For compatibility with some code paths
    pub size: ModelSize,      // For compatibility with some code paths
}

impl LogisModel {
    pub async fn new(app_handle: tauri::AppHandle, _device_preference: Option<&str>) -> anyhow::Result<Self> {
        let device = utils::get_best_device();
        // Tauri 2.0 path API
        let resource_path = app_handle.path().resource_dir().unwrap_or_default();
        let model_path = resource_path.join("models").join("Qwen3.5-0.8B-Split").to_string_lossy().into_owned();

        // Initial dummy variant for compilation compatibility
        let dummy_model = Qwen3_5GenerateModel::init(&model_path, Some(&device), None, false)?;
        let variant = ModelVariant::Qwen3_5(Arc::new(TokioMutex::new(dummy_model)));

        Ok(Self {
            device,
            model_path: model_path.clone(),
            tokenizer_path: model_path.clone(),
            embedding_path: model_path.clone(),
            current_size: Arc::new(TokioMutex::new(Some(ModelSize::Small))),
            generator: Arc::new(TokioMutex::new(Some(variant.clone()))),
            small_hibernation: Arc::new(TokioMutex::new(None)),
            large_hibernation: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: false,
            is_disk_swap: true,
            dual_mode_enabled: true,
            variant,
            size: ModelSize::Small,
        })
    }

    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
    }

    pub async fn deep_purge_resources(&self) {
        self.unload_generator().await;
        // Additional cleanup logic if needed
    }

    pub async fn get_embedding(&self, _text: String) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 768])
    }

    pub async fn secure_vram_relay(
        &self, 
        size: ModelSize, 
        _snapshot_id: Option<&String>, 
        _cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> anyhow::Result<()> {
        self.ensure_model(size).await
    }

    pub async fn extract_from_image(
        &self,
        _image_data: Vec<u8>,
        _params: ChatCompletionParameters,
        _cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> anyhow::Result<String> {
        Err(anyhow!("extract_from_image not implemented for Qwen3.5 (text-only)"))
    }

    pub async fn ensure_model(&self, size: ModelSize) -> anyhow::Result<()> {
        let mut current_size_guard = self.current_size.lock().await;
        if current_size_guard.as_ref() == Some(&size) {
            return Ok(());
        }

        let mut gen_guard = self.generator.lock().await;
        let model = Qwen3_5GenerateModel::init(&self.model_path, Some(&self.device), None, false)?;
        *gen_guard = Some(ModelVariant::Qwen3_5(Arc::new(TokioMutex::new(model))));
        *current_size_guard = Some(size);
        Ok(())
    }

    pub async fn parse_query_structured(&self, _query: String, _lang: &str) -> anyhow::Result<Value> {
        Ok(serde_json::json!({}))
    }

    pub async fn run_deep_research(
        &self,
        _query: String,
        _context: String,
        _app_handle: &tauri::AppHandle,
        _cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> anyhow::Result<String> {
        Ok("Research result placeholder".to_string())
    }
}

pub struct AppState {
    pub model_manager: Arc<LogisModel>,
}

pub async fn run_completion(
    app_handle: tauri::AppHandle,
    params: ChatCompletionParameters,
    cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<String> {
    let state = app_handle.state::<AppState>();
    let gen_opt = state.model_manager.generator.lock().await;
    if let Some(model) = gen_opt.as_ref() {
        model.generate(params, cancel_token, None, None).await
    } else {
        Err(anyhow!("Model not loaded"))
    }
}

pub fn generate_rich_summary(_doc_type: &str, _data: &Value) -> String {
    "Summary generated.".to_string()
}
