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
    // 🌟 [GENERATION HOLD] 생성 모델 '전환 구간' 동안 임베딩 재로드를 막는 카운터입니다.
    //
    //  ── 무엇을 막는가 ──
    //   deep_purge_resources 는 Step 2 에서 임베딩 락을 블록 스코프로 잡았다 놓고,
    //   Step 3 의 spawn_blocking CUDA 동기화에서 최대 10초 블로킹됩니다.
    //   그 창으로 백그라운드 인덱싱의 ensure_embedding() 이 들어와
    //   방금 내린 임베딩을 다시 올립니다.
    //   (log.txt 실측: 퍼지 블록 '내부'에 [MODEL] Loading Embedding Model 이 끼어 있음)
    //   ensure_embedding 의 기존 VRAM GATE 는 free_mb < 350 일 때만 작동하는데
    //   퍼지 직후엔 4GB 가 비어 있어 이 레이스를 구조적으로 못 막습니다.
    //
    //  ── 왜 '금지' 가 아니라 '홀드' 인가 ──
    //   STAGE-3 은 Qwen3 상주 + 임베딩 동시 사용이 정상 경로입니다
    //   ([VECTORIZING] 직후 [KV-PLAN] ... VRAM free 2392MB → Vram).
    //   따라서 상주 자체를 금지하면 안 되고, '전환 구간에만' 양보시켜야 합니다.
    //   로드가 끝나면 홀드가 풀려 임베딩이 정상적으로 다시 올라옵니다.
    //
    //  ── 왜 Arc 인가 ──
    //   LogisModel 은 Clone 이고 내부가 전부 Arc 슬롯입니다.
    //   백그라운드 인덱싱이 clone 을 들고 있어도 같은 카운터를 봐야 합니다.
    pub generation_hold: Arc<std::sync::atomic::AtomicU32>,
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

/// 🌟 [RAII] 생성 모델 전환 구간을 나타내는 가드입니다.
///
///  Drop 시 카운터를 되돌리므로, `?` 로 조기 반환하거나 패닉이 나도
///  홀드가 영구히 걸린 채 남지 않습니다. 명시적 해제 호출을 두면
///  에러 경로에서 반드시 새는데, 그 순간 임베딩이 영영 못 올라옵니다.
pub struct GenerationHold(Arc<std::sync::atomic::AtomicU32>);

impl Drop for GenerationHold {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl LogisModel {
    /// 생성 모델 전환 구간 진입을 선언합니다.
    /// 반환된 가드가 살아 있는 동안 ensure_embedding 은 로드를 양보합니다.
    pub fn hold_generation(&self) -> GenerationHold {
        self.generation_hold.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        GenerationHold(self.generation_hold.clone())
    }

    /// 지금 생성 모델 전환 구간인지 확인합니다.
    pub fn is_generation_held(&self) -> bool {
        self.generation_hold.load(std::sync::atomic::Ordering::SeqCst) > 0
    }
}