// src-tauri/src/model.rs
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use anyhow::{Result, anyhow};
use candle_core::{Device, DType};
use crate::models::qwen3_5::Qwen3_5GenerateModel;
use crate::openai_types::ChatCompletionParameters;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
    Full,
    Split,
}

#[derive(Clone)]
pub enum ModelVariant {
    Qwen3_5(Arc<TokioMutex<Qwen3_5GenerateModel>>),
}

impl ModelVariant {
    pub async fn generate(
        &self,
        params: ChatCompletionParameters,
        cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<String> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let mut gen = m.lock().await;
                gen.generate(params, cancel_flag, session_id, kv_name)
                    .await
                    .map_err(|e| anyhow!(e.to_string()))
            }
        }
    }

    pub async fn reset_cache(&self) -> Result<()> {
        match self {
            ModelVariant::Qwen3_5(m) => {
                let gen = m.lock().await;
                gen.qwen3_5.reset_mamba_cache().map_err(|e| anyhow!(e.to_string()))?;
                Ok(())
            }
        }
    }
}

#[derive(Clone)]
pub struct LogisModel {
    pub generator: Arc<TokioMutex<Option<ModelVariant>>>,
    pub device: Device,
    pub is_cpu_mode: bool,
}

impl LogisModel {
    pub async fn new(_app_handle: tauri::AppHandle, _device_pref: Option<&str>) -> Result<Self> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).map_err(|e| anyhow!(e.to_string()))?
        } else {
            Device::Cpu
        };

        Ok(Self {
            generator: Arc::new(TokioMutex::new(None)),
            device,
            is_cpu_mode: false,
        })
    }

    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
    }

    pub async fn deep_purge_resources(&self) {
        self.unload_generator().await;
    }

    pub async fn secure_vram_relay(
        &self,
        _size: ModelSize,
        _snapshot_id: Option<&String>,
        _cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> Result<()> {
        // Implementation for loading the model based on size
        Ok(())
    }

    pub async fn extract_from_image(
        &self,
        _image_data: Vec<u8>,
        _params: ChatCompletionParameters,
        _cancel_token: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> Result<String> {
        Err(anyhow!("Image extraction not implemented"))
    }

    pub async fn get_embedding(&self, _text: String) -> Result<Vec<f32>> {
        Ok(vec![0.0; 768])
    }

    pub async fn parse_query_structured(&self, _query: String, _lang: &str) -> Result<serde_json::Value> {
        Ok(serde_json::json!({}))
    }

    pub async fn run_deep_research(
        &self,
        _query: String,
        _context: String,
        _handle: &tauri::AppHandle,
        _cancel: Option<Arc<std::sync::atomic::AtomicBool>>
    ) -> Result<String> {
        Ok("Research result".to_string())
    }
}
