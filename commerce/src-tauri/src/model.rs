use crate::utils;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::embedding::EmbeddingModel;
use candle_core::DType;
use tokio::sync::Mutex as TokioMutex;
use crate::models::qwen3::generate::Qwen3GenerateModel;
use std::sync::Arc;

pub mod lifecycle;
pub mod vision;
pub mod inference;
pub mod research;
pub mod query_shipping;
pub mod query_commerce;
pub mod merge;
pub mod query_analytic;

pub use merge::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Qwen,    // 0.6B for Ingestion (기존 Small)
    Qwen3,   // Qwen3 Text Model (기존 Large, /qwen3/ 로직 전용)
    Qwen3_5, // 2B Qwen 3.5 (Text Optimized)
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    pub embedding_cache: Arc<TokioMutex<std::collections::HashMap<String, Vec<f32>>>>,

    pub is_cpu_mode: bool, 
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    qwen_model_path: String,      // 🌟 (기존 small_model_path 대신 이름 맞춤)
    qwen3_model_path: String,     // 🌟 Qwen3 모델 경로 추가
    qwen3_5_model_path: String,
    embedding_path: std::path::PathBuf,
    pub device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    _dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,

    pub siglip2_model: Arc<TokioMutex<Option<crate::models::siglip2::Siglip2Model>>>,
    pub siglip2_config: Option<crate::models::siglip2::Siglip2Config>,
    pub siglip2_model_path: String,
}

impl LogisModel {

}