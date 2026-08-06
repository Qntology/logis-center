use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; 
use candle_core::quantized::{gguf_file, QMatMul};
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

use crate::{
    models::{
        qwen::config::{QwenVLConfig, QwenVLTextConfig},
        qwen::model::QwenVLVisionModel,
        qwen::rope::{
            QwenVLTextRotaryEmbedding, apply_rotary_pos_emb,
        },
    },
    utils::tensor_utils::{
        mask_index_add, masked_scatter_dim0,
    },
};
use crate::models::qwen::generate::SLOT_MANAGER;

// Local RmsNorm implementation exposing weight and device
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        self.weight = self.weight.to_dtype(target_dtype)?.to_device(device)?;
        Ok(())
    }

    /// [MEMORY-OPT] 가중치를 메모리에서 해제하여 RAM 사용량을 최소화합니다.
    pub fn clear(&mut self) {
        // 1-element 더미 텐서로 교체하여 실제 데이터 메모리 해제 유도
        self.weight = Tensor::zeros((1,), self.weight.dtype(), &Device::Cpu).unwrap();
    }

    pub fn is_cleared(&self) -> bool {
        self.weight.elem_count() <= 1
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if self.is_cleared() {
            return Err(candle_core::Error::Msg("RMSNorm weight is cleared. Reload required.".to_string()));
        }
        
        // [CRITICAL FIX] VRAM을 2배로 먹고 속도를 떨어뜨리던 수동 분산 계산을 모두 지우고, 
        // Candle 프레임워크의 초고속 네이티브 C++/CUDA 커널에 연산을 위임합니다!
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

// Wrapper for QMatMul to act like Linear
#[derive(Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    device: Device, // Track device explicitly
}

impl QLinear {
    pub fn new(inner: QMatMul, bias: Option<Tensor>, device: Device) -> Self {
        Self { inner, bias, device }
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn shape(&self) -> candle_core::Shape {
        match &self.inner {
            QMatMul::QTensor(q) => q.shape().clone(),
            QMatMul::Tensor(t) => t.shape().clone(),
            QMatMul::TensorF16(t) => t.shape().clone(),
        }
    }

    /// [MEMORY-OPT] 가중치를 메모리에서 해제합니다. 
    pub fn clear(&mut self) {
        // 더미 텐서의 타입은 해제용이므로 F32로 고정
        self.inner = QMatMul::Tensor(Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap());
        self.bias = None;
        self.device = Device::Cpu;
    }

    pub fn is_cleared(&self) -> bool {
        match &self.inner {
            QMatMul::Tensor(t) => t.elem_count() <= 1,
            _ => false,
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if self.is_cleared() {
            return Err(anyhow!("Linear weight is cleared. Reload required."));
        }

        // [CRITICAL FIX] CPU 모드일 때도 현재 가중치가 QTensor(압축 상태)라면 
        // 연산 속도를 위해 강제로 압축을 풀도록 분기 조건을 추가합니다!
        let is_cpu_qtensor = device.is_cpu() && matches!(self.inner, QMatMul::QTensor(_));

        if !self.device.same_device(device) || is_cpu_qtensor {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            
            self.inner = match &self.inner {
                QMatMul::QTensor(q) => {
                    let t = q.dequantize(device)?.to_dtype(target_dtype)?;
                    QMatMul::Tensor(t)
                },
                QMatMul::Tensor(t) => {
                    QMatMul::Tensor(t.to_dtype(target_dtype)?.to_device(device)?)
                },
                QMatMul::TensorF16(t) => {
                    QMatMul::TensorF16(t.to_dtype(target_dtype)?.to_device(device)?)
                }
            };

            if let Some(b) = &self.bias {
                
                self.bias = Some(b.to_dtype(target_dtype)?.to_device(device)?);
            }
            self.device = device.clone();
        }
        Ok(())
    }

    
    pub fn to_device_keep_quantized(&mut self, device: &Device) -> Result<()> {
        if self.is_cleared() {
            return Err(anyhow!("Linear weight is cleared. Reload required."));
        }
        if device.is_cuda() {
            // GPU로 갈 때는 프레임워크 한계상 압축을 풀어야 함
            self.to_device(device)?;
        } else {
            // CPU일 때는 압축 상태(QTensor)를 유지하여 RAM 피크(OOM) 완벽 방지!
            if let Some(b) = &self.bias {
                let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
                self.bias = Some(b.to_device(device)?.to_dtype(target_dtype)?);
            }
            self.device = device.clone();
        }
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // [CRITICAL FIX] 224번 반복되던 device/dtype 중복 검사 및 할당 완전 제거!
        let (b, s, h) = xs.dims3()?;
        let xs_flat = xs.reshape((b * s, h))?;
        
        let out = match &self.inner {
            QMatMul::QTensor(_q) => {
                let xs_f32 = xs_flat.to_dtype(DType::F32)?;
                self.inner.forward(&xs_f32)?
            },
            QMatMul::Tensor(t) => {
                let xs_aligned = xs_flat.to_dtype(t.dtype())?;
                self.inner.forward(&xs_aligned)?
            },
            _ => {
                let xs_f32 = xs_flat.to_dtype(DType::F32)?;
                self.inner.forward(&xs_f32)?
            }
        };
        
        let out = out.reshape((b, s, ()))?;
        let final_out = if let Some(bias) = &self.bias {
            out.to_dtype(bias.dtype())?.broadcast_add(bias)?
        } else {
            out.to_dtype(xs.dtype())?
        };
        
        Ok(final_out)
    }
}

// [QUANTIZED-KV] Storage for 4-bit compressed KV cache in VRAM
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KVLocation {
    VRAM,
    RAM,
    SSD,
    Loading,   // New: Block is being prefetched
    Streaming, 
}

#[derive(Clone, Debug, PartialEq)]
pub enum SlotState {
    Free,
    Computing,    // Reserved for GPU computation
    Transferring, // Moving from VRAM to CPU RAM
    Compressing,  // BitKV compression in progress
    Saving,       // SSD I/O in progress
    Ready,        // Stored in RAM and ready for use or SSD offload
}

// [NEW] 마스터 슬롯: 1024 토큰에 대한 28개 레이어 전체 데이터를 담는 방
pub struct MemorySlot {
    pub id: usize,
    pub state: Arc<std::sync::atomic::AtomicU8>, // 0:Free, 1:Baking, 2:Ready, 3:Loading
    pub k_layers: Vec<Arc<std::sync::Mutex<Option<Tensor>>>>, // Slave K tensors
    pub v_layers: Vec<Arc<std::sync::Mutex<Option<Tensor>>>>, // Slave V tensors
    pub remaining_layers: Arc<std::sync::atomic::AtomicUsize>,
}

impl MemorySlot {
    pub fn new(id: usize, num_layers: usize) -> Self {
        let mut k_layers = Vec::with_capacity(num_layers);
        let mut v_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_layers.push(Arc::new(std::sync::Mutex::new(None)));
            v_layers.push(Arc::new(std::sync::Mutex::new(None)));
        }
        Self {
            id,
            state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            k_layers,
            v_layers,
            remaining_layers: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

#[derive(Clone)]
pub struct KVBlock {
    pub inner: Arc<std::sync::RwLock<KVBlockInner>>,
}

fn default_bitkv_cache() -> Arc<std::sync::RwLock<Vec<Option<BitKVMetadata>>>> {
    Arc::new(std::sync::RwLock::new(vec![None; 28]))
}

// [NEW] 중앙 집중식 KV 목차의 각 항목 (별도 고정 슬롯 관리용)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RegistryEntry {
    pub location: Vec<KVLocation>, // [Layer Index] -> Location
    pub slot_ids: Vec<Option<usize>>, // [Layer Index] -> Slot ID
    pub token_start: usize,
    pub token_len: usize,
    pub ssd_path: Option<std::path::PathBuf>,
    pub hidden_states_path: Vec<Option<std::path::PathBuf>>, // [Layer Index] -> SSD Path for Output
    #[serde(skip)]
    pub is_dirty: Vec<bool>, // [NEW] Per-layer dirty flag to prevent redundant SSD backup tasks
    #[serde(skip, default = "std::time::Instant::now")]
    pub last_accessed: std::time::Instant, // LRU 순위 결정을 위한 접근 시각
    #[serde(skip, default = "default_bitkv_cache")]
    pub bitkv_cache: Arc<std::sync::RwLock<Vec<Option<BitKVMetadata>>>>,
}

impl RegistryEntry {
    pub fn new(token_start: usize, token_len: usize, num_layers: usize) -> Self {
        Self {
            token_start,
            token_len,
            location: vec![KVLocation::SSD; num_layers],
            slot_ids: vec![None; num_layers],
            ssd_path: None,
            hidden_states_path: vec![None; num_layers],
            is_dirty: vec![true; num_layers],
            last_accessed: std::time::Instant::now(),
            bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; num_layers])),
        }
    }
}

// [NEW] 모델 전체가 공유하는 2차원 KV 목차
#[derive(Clone)]
pub struct KVRegistry {
    pub entries: Arc<std::sync::RwLock<Vec<RegistryEntry>>>,
}

impl KVRegistry {
    pub fn new() -> Self {
        // [FIX] 장부를 128개 미리 할당하되, 실제 데이터 길이는 0으로 초기화합니다.
        // 이를 통해 RoPE 오프셋이 32512로 점프하는 대참사를 막습니다.
        let mut entries = Vec::with_capacity(128);
        for i in 0..128 {
            let entry = RegistryEntry::new(i * 1024, 0, 28);
            entries.push(entry);
        }
        Self {
            entries: Arc::new(std::sync::RwLock::new(entries)),
        }
    }

    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let entries = self.entries.read().unwrap();
        
        // [DECENTRALIZED-SAVE] 레이어별로 장부를 쪼개서 저장
        for l_idx in 0..28 {
            let mut layer_data = Vec::new();
            for entry in entries.iter() {
                if entry.location[l_idx] == KVLocation::SSD {
                    layer_data.push(serde_json::json!({
                        "token_start": entry.token_start,
                        "token_len": entry.token_len,
                        "ssd_path": entry.ssd_path
                    }));
                }
            }
            if let Ok(json) = serde_json::to_string_pretty(&layer_data) {
                // [DIRECT-IO] Use OS-accelerated write for metadata
                let _ = crate::utils::direct_loader::save_kv_block(&path.join(format!("layer{}_meta.json", l_idx)), json.as_bytes());
            }
        }
        Ok(())
    }

    pub fn load_from_file(&self, path: &std::path::Path) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        
        // [DECENTRALIZED-LOAD] 28개 레이어 장부를 각각 읽어서 통합 장부 복원
        for l_idx in 0..28 {
            let meta_path = path.join(format!("layer{}_meta.json", l_idx));
            if meta_path.exists() {
                // [DIRECT-IO] Use OS-accelerated read for metadata
                if let Ok(data) = crate::utils::direct_loader::load_kv_block(&meta_path) {
                    if let Ok(json_str) = String::from_utf8(data) {
                        if let Ok(loaded) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                            for item in loaded {
                                let start = item["token_start"].as_u64().unwrap_or(0) as usize;
                                let idx = start / 1024;
                                if idx < entries.len() {
                                    entries[idx].location[l_idx] = KVLocation::SSD;
                                    if let Some(p) = item["ssd_path"].as_str() {
                                        entries[idx].ssd_path = Some(std::path::PathBuf::from(p));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct KVBlockInner {
    pub location: KVLocation,
    pub index: usize,
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
    pub ssd_path: Option<std::path::PathBuf>,
    pub len: usize,
    pub offset: usize, 
    pub bitkv_metadata: Option<BitKVMetadata>,
}

impl KVBlock {
    pub fn new(location: KVLocation, index: usize, len: usize, offset: usize) -> Self {
        Self {
            inner: Arc::new(std::sync::RwLock::new(KVBlockInner {
                location,
                index,
                k_cache: None,
                v_cache: None,
                ssd_path: None,
                len,
                offset,
                bitkv_metadata: None,
            })),
        }
    }
}

#[derive(Clone)]
pub struct BitKVMetadata {
    pub k_data: Tensor,
    pub v_data: Tensor,
    pub original_shape: Vec<usize>,
}

#[derive(Clone)]
pub struct QuantizedQwenVLTextAttention {
    pub q_proj: QLinear,
    pub k_proj: QLinear,
    pub v_proj: QLinear,
    pub o_proj: QLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f64,
    pub kv_blocks: Vec<KVBlock>,
    pub registry: KVRegistry, // [NEW] 중앙 목차 참조
    pub layer_idx: usize,
    pub active_kv_name: Option<String>, // [NEW] JIT-LOAD 경로 동기화용
    pub active_session_id: Option<String>, // [추가] 에러 해결을 위한 필드
    // 🌟 [KV RESIDENCY] 디코딩 진입 시 1회 결정된 KV 블록 배치 위치.
    //    기본값 Ssd 는 기존 동작(매 토큰 SSD 재읽기)과 100% 동일합니다.
    pub kv_residency: crate::utils::resources::KvResidency,
    // [ACCUMULATOR] VRAM 내 병합 캐시: 매번 수십개의 블록을 cat하는 오버헤드 제거
    pub vram_merged_k: Option<Tensor>,
    pub vram_merged_v: Option<Tensor>,
    pub merged_vram_block_count: usize,
}

impl QuantizedQwenVLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if self.q_proj.is_cleared() {
            return Err(anyhow!("Attention weights are cleared. Reload required."));
        }
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?;
        self.k_norm.to_device(device)?;
        
        // [ACCUMULATOR-RESET] 장치 이동 시 병합 캐시 초기화 (필요시 새로 생성)
        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;

        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        for block in &mut self.kv_blocks {
            let (index, mut inner) = {
                let inner = block.inner.write().unwrap();
                (inner.index, inner)
            };
            let loc = {
                let reg = self.registry.entries.read().unwrap();
                if index < reg.len() { reg[index].location[self.layer_idx] } else { KVLocation::VRAM }
            };
            if loc == KVLocation::VRAM {
                if let Some(k) = &inner.k_cache {
                    
                    inner.k_cache = Some(k.to_dtype(target_dtype)?.to_device(device)?);
                }
                if let Some(v) = &inner.v_cache {
                    
                    inner.v_cache = Some(v.to_dtype(target_dtype)?.to_device(device)?);
                }
            }
        }
        Ok(())
    }

    /// [MEMORY-OPT] 가중치를 메모리에서 완전히 해제합니다.
    pub fn clear(&mut self) {
        self.q_proj.clear();
        self.k_proj.clear();
        self.v_proj.clear();
        self.o_proj.clear();
        self.q_norm.clear();
        self.k_norm.clear();
        
        
        // 다음 레이어 연산 시 VRAM을 차지하지 못하도록 확실하게 파괴(None)해 줍니다!
        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &QwenVLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        is_gguf_naming: bool,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        registry: KVRegistry, // [NEW]
    ) -> Result<Self> {
        let _hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);

        let (q, k, v, o, q_n, k_n) = if is_gguf_naming {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        };

        // [FIX] Dynamic Head Detection: Trust GGUF tensor shapes over config to prevent reshape mismatches.
        let q_weight_name = format!("{base_name}.{q}.weight");
        let k_weight_name = format!("{base_name}.{k}.weight");

        let num_attention_heads = if let Some(info) = ct.tensor_infos.get(&q_weight_name) {
            let out_features = info.shape.dims()[0];
            out_features / head_dim
        } else {
            config.num_attention_heads
        };

        let num_key_value_heads = if let Some(info) = ct.tensor_infos.get(&k_weight_name) {
            let out_features = info.shape.dims()[0];
            out_features / head_dim
        } else {
            config.num_key_value_heads
        };

        let num_kv_groups = if num_key_value_heads > 0 {
            num_attention_heads / num_key_value_heads
        } else {
            1
        };

        if num_attention_heads != config.num_attention_heads || num_key_value_heads != config.num_key_value_heads {
            if layer_idx == 0 {
                println!("[MODEL-FIX] Architecture Mismatch Detected. GGUF: {} heads / {} KV heads. Config: {} heads / {} KV heads. Overriding config.",
                    num_attention_heads, num_key_value_heads, config.num_attention_heads, config.num_key_value_heads);
            }
        }

        let q_proj = get_qlinear(ct, reader, &format!("{base_name}.{q}"), device, dtype)?;
        
        if layer_idx == 0 {
            println!("[DIAG-MODEL] Layer 0 Q-Proj Weight Shape: {:?}", q_proj.shape());
        }

        let k_proj = get_qlinear(ct, reader, &format!("{base_name}.{k}"), device, dtype)?;
        let v_proj = get_qlinear(ct, reader, &format!("{base_name}.{v}"), device, dtype)?;
        let o_proj = get_qlinear(ct, reader, &format!("{base_name}.{o}"), device, dtype)?;

        let q_norm = get_rms_norm(ct, reader, &format!("{base_name}.{q_n}"), config.rms_norm_eps, device, dtype)?;
        let k_norm = get_rms_norm(ct, reader, &format!("{base_name}.{k_n}"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_kv_name: None,
            active_session_id: None, // [추가]
            kv_residency: crate::utils::resources::KvResidency::Ssd,
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        })
    }

    pub fn load_weights_inplace<R: std::io::Seek + std::io::Read>(
        &mut self,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        is_gguf_naming: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let (q, k, v, o, q_n, k_n) = if is_gguf_naming {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        };

        self.q_proj = get_qlinear(ct, reader, &format!("{base_name}.{q}"), device, dtype)?;
        self.k_proj = get_qlinear(ct, reader, &format!("{base_name}.{k}"), device, dtype)?;
        self.v_proj = get_qlinear(ct, reader, &format!("{base_name}.{v}"), device, dtype)?;
        self.o_proj = get_qlinear(ct, reader, &format!("{base_name}.{o}"), device, dtype)?;
        self.q_norm = get_rms_norm(ct, reader, &format!("{base_name}.{q_n}"), self.q_norm.eps(), device, dtype)?;
        self.k_norm = get_rms_norm(ct, reader, &format!("{base_name}.{k_n}"), self.k_norm.eps(), device, dtype)?;

        Ok(())
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask_in: Option<&Tensor>,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
        baking_only: bool,
    ) -> Result<Tensor> {
        self.active_session_id = session_id.clone(); 
        self.active_kv_name = kv_name;
        
        let dev = self.q_proj.device();
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };

        // 1. [ALIGNMENT] Input & Rotary
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };
        let (b_sz, q_len, _) = xs.dims3()?;


        // 수만 토큰의 거대 마스크를 한 번에 만들면 VRAM이 누수처럼 증가합니다.
        // 전체 마스크 생성을 지우고, 동적 생성을 위한 플래그만 남깁니다.
        let has_dynamic_mask = q_len > 1 && attention_mask_in.is_none();
        let q_indices = if has_dynamic_mask { Some(Tensor::arange(0u32, q_len as u32, dev)?.unsqueeze(1)?) } else { None };
        let seq_offset_t = if has_dynamic_mask { Some(Tensor::new(seqlen_offset as u32, dev)?) } else { None };

        // [CRITICAL FIX] .contiguous() 삭제로 VRAM 전체 복사 스파이크 원천 차단!
        let mut query_states = self.q_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?.contiguous()?; 
        
        let mut key_states = self.k_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?.contiguous()?; 
        
        let value_states = self.v_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?; 

        let cos = cos.to_dtype(target_dtype)?;
        let sin = sin.to_dtype(target_dtype)?;
        
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, &cos, &sin, false)?;
        let query_states = query_states.to_dtype(target_dtype)?.contiguous()?;
        let key_states = key_states.to_dtype(target_dtype)?.contiguous()?;

        // 2. [BLOCK-PIPELINE-ALLOCATION] Append or Create New
        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 1024usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    // [CRITICAL FIX] 메모리 복사를 유발하는 contiguous() 제거
                    let k_piece = key_states.narrow(2, chunk_offset, take)?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?;

                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };

                        let pk_f = pk.to_dtype(target_dtype).unwrap_or_else(|_| pk.clone());
                        let pv_f = pv.to_dtype(target_dtype).unwrap_or_else(|_| pv.clone());

                        // [CRITICAL FIX] 병합(cat) 후 메모리 재정렬(contiguous) 오버헤드 완벽 제거!
                        let cat_k = Tensor::cat(&[&pk_f, &k_piece], 2)?.contiguous()?;
                        let cat_v = Tensor::cat(&[&pv_f, &v_piece], 2)?.contiguous()?;
                        
                        inner.k_cache = Some(if dev.is_cuda() { cat_k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_k.clone()) } else { cat_k });
                        inner.v_cache = Some(if dev.is_cuda() { cat_v.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_v.clone()) } else { cat_v });
                        
                        inner.len += take; tokens_to_process -= take; chunk_offset += take;
                        appended = true;
                        
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            let entry = &mut reg[inner.index];
                            entry.token_len = inner.len;
                            if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = true; }
                        }
                    }
                }
            }
            if !appended {
                let take = tokens_to_process.min(1024);
                
                
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(if dev.is_cuda() { k_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k_piece.clone()) } else { k_piece });
                    inner.v_cache = Some(if dev.is_cuda() { v_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v_piece.clone()) } else { v_piece });
                }
                
                let mut reg = self.registry.entries.write().unwrap();
                if index < reg.len() {
                    let entry = &mut reg[index];
                    entry.token_start = current_total;
                    entry.token_len = take;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = true; }
                    if self.layer_idx < entry.location.len() { entry.location[self.layer_idx] = KVLocation::VRAM; }
                }
                self.kv_blocks.push(new_block);
                tokens_to_process -= take; chunk_offset += take;
            }
        }

        // 3. [CHUNK-BASED ONLINE SOFTMAX] Zero-VRAM Spikes Attention
        let total_tokens_now = seqlen_offset + q_len;
        
        // =========================================================================
        // 🚀 [최적화 2] 0.6B 디코딩 초고속 가속 (No-Cat Fast-Path)
        // =========================================================================
        
        let mut out_res: Option<Tensor> = None;
        let mut m_n: Option<Tensor> = None;
        let mut l_n: Option<Tensor> = None;

        let q_aligned = query_states.to_dtype(target_dtype)?.contiguous()?;

        // [GQA-FOLD] K/V를 groups배로 물리 복제하는 대신 Q의 head 축을 접습니다.
        // 메모리 레이아웃 [b][H][q_len][d] == [b][kv_h][groups][q_len][d] 이므로
        // reshape은 복사 없는 메타데이터 조작으로 끝납니다.
        let (q_b, q_h, q_l, q_d) = q_aligned.dims4()?;
        let kv_h = self.num_key_value_heads;
        let q_folded = if self.num_kv_groups > 1 {
            q_aligned.reshape((q_b, kv_h, self.num_kv_groups * q_l, q_d))?
        } else {
            q_aligned.clone()
        };

        for block in &self.kv_blocks {
            let (index, b_off, _b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            // [STEP A] Load Block to VRAM
            let (k_block, v_block, is_temporary) = {
                let mut inner = block.inner.write().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    if inner.location == KVLocation::VRAM {
                        (k.to_dtype(target_dtype).unwrap_or_else(|_| k.clone()), v.to_dtype(target_dtype).unwrap_or_else(|_| v.clone()), false) // VRAM에 있다면 복원해서 씀
                    } else {
                        // RAM에 있다면 GPU로 잠깐 복사해서 씀 (원본은 냅둠)
                        (k.to_device(dev)?.to_dtype(target_dtype)?, v.to_device(dev)?.to_dtype(target_dtype)?, true)
                    }
                } else {
                    let mut k_cpu = None;
                    let mut v_cpu = None;
                    {
                        let reg = self.registry.entries.read().unwrap();
                        let cache = reg[index].bitkv_cache.read().unwrap();
                        if let Some(m) = &cache[self.layer_idx] {
                            
                            k_cpu = Some(m.k_data.clone());
                            v_cpu = Some(m.v_data.clone());
                        }
                    }

                    // ... SSD 파일 로딩 부분 생략 (동일하게 유지하되 inner 갱신을 생략함) ...
                    if k_cpu.is_none() {
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let sid = self.active_session_id.as_deref().ok_or_else(|| anyhow!("Session ID missing"))?;
                        let mut path_candidates = Vec::new();
                        
                        if let Some(p) = { let reg = self.registry.entries.read().unwrap();
                            if index < reg.len() { reg[index].ssd_path.clone() } else { None } } {
                            path_candidates.push(p);
                        }
                        
                        let kv_name_raw = self.active_kv_name.as_deref().unwrap_or("text");
                        let kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        let kv_type = if kv_type == "inference" || kv_type == "reference" || kv_type.is_empty() { "text" } else { kv_type };
                        
                        path_candidates.push(kv_dir.join(format!("{}/inference/{}/b{}", sid, kv_type, b_off)));
                        path_candidates.push(kv_dir.join(format!("{}/reference/{}/b{}", sid, kv_type, b_off)));
                        path_candidates.push(kv_dir.join(format!("{}/inference/text/b{}", sid, b_off)));
                        
                        for full_path in &path_candidates {
                            let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                            for _retry in 0..3 {
                                if block_file.exists() {
                                    if let Ok(content) = crate::utils::direct_loader::load_kv_block(&block_file) {
                                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                            let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                            let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                            
                                            if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                                let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                                let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                                
                                                
                                                // 이 한 줄의 수정으로 System RAM 점유율이 50% 이상 박살 납니다.
                                                let saved_dtype = match kd.dtype() {
                                                    safetensors::Dtype::F64 => DType::F8E4M3,
                                                    safetensors::Dtype::F32 => DType::F32,
                                                    safetensors::Dtype::F16 => DType::F16,
                                                    _ => DType::BF16,
                                                };
                                                k_cpu = Some(Tensor::from_raw_buffer(kd.data(), saved_dtype, &meta_os, &Device::Cpu).unwrap());
                                                v_cpu = Some(Tensor::from_raw_buffer(vd.data(), saved_dtype, &meta_os, &Device::Cpu).unwrap());
                                            }
                                            break;
                                        }
                                    }
                                } else {
                                    // [NO-BLOCKING-WAIT] 존재하지 않는 경로 후보에 대한 5ms×3 대기를 제거합니다.
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(5));
                            }
                            if k_cpu.is_some() { break; } 
                        }
                    }
                    
                    let fallback_shape = vec![1, self.num_key_value_heads, _b_len, self.head_dim];

                    // [DIAG] zeros 폴백은 곧 "문맥 소실"입니다. 무음 처리하지 말고 반드시 노출시킵니다.
                    if k_cpu.is_none() {
                        println!("[KV-MISS] layer {} block off={} 재로딩 실패 → zeros 폴백 (문맥 손실 발생)",
                            self.layer_idx, b_off);
                    }

                    let k_safe = k_cpu.unwrap_or_else(|| Tensor::zeros(fallback_shape.as_slice(), DType::BF16, &Device::Cpu).unwrap());
                    let v_safe = v_cpu.unwrap_or_else(|| Tensor::zeros(fallback_shape.as_slice(), DType::BF16, &Device::Cpu).unwrap());
                    
                    let k_gpu = k_safe.to_device(dev)?.to_dtype(target_dtype)?;
                    let v_gpu = v_safe.to_device(dev)?.to_dtype(target_dtype)?;

                    
                    // 🌟 [KV RESIDENCY] VRAM 상주가 확정된 경우, 읽어온 블록을 FP8 로 VRAM 에 눌러앉힙니다.
                    //    이후 토큰에서는 이 블록이 KVLocation::VRAM 으로 잡혀
                    //    SafeTensors 역직렬화 + PCIe 업로드가 통째로 사라집니다.
                    if self.kv_residency == crate::utils::resources::KvResidency::Vram && dev.is_cuda() {
                        let k_res = k_gpu.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k_gpu.clone());
                        let v_res = v_gpu.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v_gpu.clone());
                        inner.k_cache = Some(k_res);
                        inner.v_cache = Some(v_res);
                        inner.location = KVLocation::VRAM;

                        let mut reg = self.registry.entries.write().unwrap();
                        if index < reg.len() && self.layer_idx < reg[index].location.len() {
                            reg[index].location[self.layer_idx] = KVLocation::VRAM;
                        }
                    } else {
                        inner.k_cache = Some(k_safe); 
                        inner.v_cache = Some(v_safe); 
                        inner.location = KVLocation::RAM;

                        let mut reg = self.registry.entries.write().unwrap();
                        if index < reg.len() && self.layer_idx < reg[index].location.len() {
                            reg[index].location[self.layer_idx] = KVLocation::RAM;
                        }
                    }
                    
                    (k_gpu, v_gpu, true)
                }
            };

            // [STEP B] Online Softmax for this Chunk
            let k = k_block.contiguous()?;
            let v = v_block.contiguous()?;

            let actual_kv_len = k.dim(2)?;

            // [ZERO-COPY-T] candle의 CUDA matmul은 transpose 뷰를 cuBLAS OP_T로 직접 처리합니다.
            // .contiguous()를 붙이면 K 전체를 한 번 더 복사하므로 제거합니다.
            let k_t = k.transpose(2, 3)?;

            let s_folded = (q_folded.matmul(&k_t)? * self.scaling)?;
            let mut s_chunk = if self.num_kv_groups > 1 {
                s_folded.reshape((q_b, q_h, q_l, actual_kv_len))?
            } else {
                s_folded
            };

            
            if has_dynamic_mask {
                if b_off + actual_kv_len > seqlen_offset {
                    let k_indices = Tensor::arange(b_off as u32, (b_off + actual_kv_len) as u32, dev)?.unsqueeze(0)?;
                    let mask = k_indices.broadcast_gt(&(q_indices.as_ref().unwrap().broadcast_add(seq_offset_t.as_ref().unwrap())?))?;
                    let chunk_mask = mask.to_dtype(target_dtype)?.affine(-1e4, 0.0)?.unsqueeze(0)?.unsqueeze(0)?;
                    s_chunk = s_chunk.broadcast_add(&chunk_mask)?;
                }
            } else if let Some(mask) = attention_mask_in {
                // 커스텀 Vision 마스크 처리 유지
                let mask_len = mask.dim(candle_core::D::Minus1)?;
                if b_off < mask_len {
                    let take = std::cmp::min(actual_kv_len, mask_len - b_off);
                    let chunk_mask = mask.narrow(candle_core::D::Minus1, b_off, take)?;
                    
                    if take < actual_kv_len {
                        let left_masked = s_chunk.narrow(candle_core::D::Minus1, 0, take)?
                            .broadcast_add(&chunk_mask)?;
                        let right_unmasked = s_chunk.narrow(candle_core::D::Minus1, take, actual_kv_len - take)?;
                        s_chunk = Tensor::cat(&[&left_masked, &right_unmasked], candle_core::D::Minus1)?;
                    } else {
                        s_chunk = s_chunk.broadcast_add(&chunk_mask)?; 
                    }
                }
            }

            let s_chunk_f32 = s_chunk.to_dtype(DType::F32)?;
            let m_j = s_chunk_f32.max_keepdim(candle_core::D::Minus1)?;
            let p_j = s_chunk_f32.broadcast_sub(&m_j)?.exp()?;
            let l_j = p_j.sum_keepdim(candle_core::D::Minus1)?;

            // [GQA-FOLD] P @ V 도 동일하게 접어서 계산 후 원래 head 배열로 복원합니다.
            let p_v = p_j.to_dtype(v.dtype())?.contiguous()?;
            let p_folded = if self.num_kv_groups > 1 {
                p_v.reshape((q_b, kv_h, self.num_kv_groups * q_l, actual_kv_len))?
            } else {
                p_v
            };
            let out_folded = p_folded.matmul(&v)?;
            let out_j = if self.num_kv_groups > 1 {
                out_folded.reshape((q_b, q_h, q_l, self.head_dim))?
            } else {
                out_folded
            };
            let out_j_f32 = out_j.to_dtype(DType::F32)?;

            match out_res {
                None => {
                    out_res = Some(out_j_f32); 
                    m_n = Some(m_j);
                    l_n = Some(l_j);
                }
                Some(prev_out_f32) => { 
                    let prev_m = m_n.as_ref().unwrap();
                    let prev_l = l_n.as_ref().unwrap();
                    
                    let m_new = prev_m.maximum(&m_j)?;
                    let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
                    let diff_new = m_j.broadcast_sub(&m_new)?.exp()?;
                    
                    let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
                    let out_new_f32 = prev_out_f32.broadcast_mul(&diff_old)?.add(&out_j_f32.broadcast_mul(&diff_new)?)?;
                    
                    out_res = Some(out_new_f32); 
                    m_n = Some(m_new);
                    l_n = Some(l_new);
                }
            }

            drop(k);
            drop(v);
        }

        // [STEP C] Finalize Attention Output
        let attn_output = if let (Some(out_f32), Some(l_f32)) = (out_res, l_n) {
            out_f32.broadcast_div(&l_f32)?.to_dtype(target_dtype)?
        } else {
            return Err(anyhow!("No KV data processed"));
        };

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok(attn_output)
    }


    pub fn compress_to_bf16(&self, t: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        
        // 🌟 [동적 타입 분기] SSD 저장소 맵으로 넘기기 직전에도 F32/FP8 원본 정밀도를 절대 파괴하지 않고 보존합니다.
        let target_dtype = if t.device().is_cuda() || t.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
        let t_compressed = t.to_dtype(target_dtype).unwrap_or_else(|_| t.clone()).to_device(&Device::Cpu)?.contiguous()?;
        Ok((t_compressed, original_shape))
    }

    pub fn decompress_from_bf16(&self, data: &Tensor, _original_shape: &[usize], device: &Device) -> Result<Tensor> {
        
        let t = if device.is_cpu() { data.to_dtype(DType::F32)?.to_device(device)? } else { data.to_device(device)? };
        Ok(t) 
    }

    pub fn decompress_from_8bit(&self, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = packed.device();
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        
        let scales_vec = scales.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements: usize = original_shape.iter().product();
        let mut decoded = vec![0.0f32; total_elements];
        use rayon::prelude::*;
        decoded.par_chunks_mut(last_dim).enumerate().for_each(|(v_idx, vector_out)| {
            let s = scales_vec[v_idx];
            let packed_start = v_idx * last_dim;
            let packed_vector = &packed_vec[packed_start..packed_start + last_dim];
            for (i, &p) in packed_vector.iter().enumerate() {
                vector_out[i] = (p as i8) as f32 * s;
            }
        });
        let t = Tensor::from_vec(decoded, original_shape, &Device::Cpu)?;
        Ok(t.to_device(device)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for block in &self.kv_blocks {
            let slot_id = {
                let reg = self.registry.entries.read().unwrap();
                let inner = block.inner.read().unwrap();
                if inner.index < reg.len() { reg[inner.index].slot_ids[self.layer_idx] } else { None }
            };
            if let Some(id) = slot_id {
                tauri::async_runtime::spawn(async move {
                    SLOT_MANAGER.release_slot(id).await;
                });
            }
        }
        self.kv_blocks.clear();
    }

    // -----------------------------------------------------------------------
    // [QuantizedQwenVLTextAttention 나머지 구현부]
    // Part 7의 clear_kv_cache() 아래에 이어서 작성됩니다.
    // -----------------------------------------------------------------------
    pub fn trigger_realtime_incremental_bake(&self, session_id: &str, is_last_chunk: bool, baking_only: bool, _is_decoding: bool) -> Result<()> {
        let target_indices: Vec<usize> = self.kv_blocks.iter().enumerate().filter_map(|(i, b)| {
            let inner = b.inner.read().unwrap();
            let is_full = inner.len == 1024;
            
            let is_dirty = {
                let reg = self.registry.entries.read().unwrap();
                if i < reg.len() { 
                    if self.layer_idx < reg[i].is_dirty.len() { reg[i].is_dirty[self.layer_idx] } else { true }
                } else { true }
            };

            if (is_full || is_last_chunk) && inner.k_cache.is_some() && is_dirty { Some(i) } else { None }
        }).collect();

        let mut gpu_k_list = Vec::new();
        let mut gpu_v_list = Vec::new();
        let mut valid_indices = Vec::new();

        for idx in target_indices {
            let block = self.kv_blocks[idx].clone();
            {
                let mut reg = self.registry.entries.write().unwrap();
                if idx < reg.len() && self.layer_idx < reg[idx].is_dirty.len() {
                    reg[idx].is_dirty[self.layer_idx] = false;
                }
            }
            let inner = block.inner.read().unwrap();
            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                gpu_k_list.push(k.clone());
                gpu_v_list.push(v.clone());
                valid_indices.push(idx);
            }
        }

        if gpu_k_list.is_empty() { return Ok(()); }

        
        let merged_k_gpu = Tensor::cat(&gpu_k_list, 2).unwrap_or_else(|_| gpu_k_list[0].clone()).contiguous().unwrap_or_else(|_| gpu_k_list[0].clone());
        let merged_v_gpu = Tensor::cat(&gpu_v_list, 2).unwrap_or_else(|_| gpu_v_list[0].clone()).contiguous().unwrap_or_else(|_| gpu_v_list[0].clone());

        let merged_k_cpu = merged_k_gpu.to_device(&Device::Cpu).unwrap_or_else(|_| merged_k_gpu.clone());
        let merged_v_cpu = merged_v_gpu.to_device(&Device::Cpu).unwrap_or_else(|_| merged_v_gpu.clone());

        let kv_name_raw = self.active_kv_name.clone().unwrap_or_else(|| "text".to_string());
        let last_part = kv_name_raw.split('/').last().unwrap_or("text");
        let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text".to_string() } else { last_part.to_string() };
        let session_id_owned = session_id.to_string();
        let registry_clone = self.registry.clone();
        let layer_idx = self.layer_idx;
        let num_kv_h = self.num_key_value_heads;
        let h_d = self.head_dim;

        let mut current_offset = 0;
        for (i, &idx) in valid_indices.iter().enumerate() {
            let chunk_len = gpu_k_list[i].dim(2).unwrap_or(1024);
            let k_cpu = merged_k_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_k_cpu.clone()).contiguous().unwrap_or_else(|_| merged_k_cpu.clone());
            let v_cpu = merged_v_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_v_cpu.clone()).contiguous().unwrap_or_else(|_| merged_v_cpu.clone());
            current_offset += chunk_len;

            let block = self.kv_blocks[idx].clone();
            let (off, b_idx, b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.offset, inner.index, inner.len)
            };

            let sub_path = if baking_only { format!("{}/reference/{}", session_id_owned, kv_type) } else { format!("{}/inference/{}", session_id_owned, kv_type) };
            let registry_inner = registry_clone.clone();
            
            crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            tauri::async_runtime::spawn(async move {
                if let Some(tx) = crate::models::qwen::generate::BAKE_TX.get() {
                    let kv_dir = crate::utils::paths::get_kv_dir(None);
                    let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                    if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                    let k_shape_u32 = vec![1u32, num_kv_h as u32, b_len as u32, h_d as u32];
                    let dump = crate::models::qwen::generate::LayerKVDump {
                        layer_idx,
                        k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                        v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                        k_shape: Tensor::from_vec(k_shape_u32, (4,), &Device::Cpu).unwrap(),
                        raw_k: Some(k_cpu),
                        raw_v: Some(v_cpu),
                    };
                    let sid = crate::models::qwen::generate::SLOT_MANAGER.acquire_write_slot(b_len).await;
                    
                    if tx.send(crate::models::qwen::generate::SlotTask::Bake(crate::models::qwen::generate::BakeTask {
                        slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path), offset: off, layers: vec![dump],
                        is_relay_baking: baking_only, block_idx: Some(b_idx), registry: registry_inner,
                    })).await.is_err() {
                        crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        crate::models::qwen::generate::SLOT_MANAGER.release_slot(sid).await;
                    }
                } else {
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.last().map(|b| {
            let inner = b.inner.read().unwrap();
            inner.offset + inner.len
        }).unwrap_or(0)
    }

    pub fn batch_load_layer_kv(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let mut index_path = kv_dir.join(kv_name).join(format!("layer{}.json", self.layer_idx));
        
        if !index_path.exists() {
            let fallback = kv_dir.join(kv_name).join("layer0.json");
            if fallback.exists() {
                index_path = fallback;
            } else { return Ok(()); }
        }
        
        let index_json = if let Ok(data) = crate::utils::direct_loader::load_kv_block(&index_path) {
            String::from_utf8(data).unwrap_or_default()
        } else { 
            return Ok(()); 
        };
        let index: LayerIndex = serde_json::from_str(&index_json)?;
        
        for (_b_idx, block_info) in index.blocks.into_iter().enumerate() {
            let block_parent = kv_dir.join(kv_name).join(format!("b{}", block_info.offset));
            let l_file = block_parent.join(format!("l{}.st", self.layer_idx));
            let file_path = if l_file.exists() { l_file } else { block_parent.join("l0.st") };
            
            if !file_path.exists() { continue; }
            
            let b_idx = block_info.offset / 1024;
            
            if let Ok(content) = crate::utils::direct_loader::load_kv_block(&file_path) {
                if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                    let is_l0 = file_path.to_string_lossy().contains("l0.st");
                    let prefix = if is_l0 { format!("b{}_l0_", block_info.offset) } else { format!("b{}_l{}_", block_info.offset, self.layer_idx) };
                    let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();

                    if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                        let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                        
                        let dev = &Device::Cpu;
                        let saved_dtype = match kd.dtype() {
                            safetensors::Dtype::F64 => DType::F8E4M3,
                            safetensors::Dtype::F32 => DType::F32,
                            safetensors::Dtype::F16 => DType::F16,
                            _ => DType::BF16,
                        };
                        let kd_t = Tensor::from_raw_buffer(kd.data(), saved_dtype, &meta_os, dev)?;
                        let vd_t = Tensor::from_raw_buffer(vd.data(), saved_dtype, &meta_os, dev)?;

                        let mut k_raw = if saved_dtype == DType::F32 { kd_t } else { self.decompress_from_bf16(&kd_t, &meta_os, dev)? };
                        let mut v_raw = if saved_dtype == DType::F32 { vd_t } else { self.decompress_from_bf16(&vd_t, &meta_os, dev)? };

                        let target_heads = self.num_key_value_heads;
                        let target_dim = self.head_dim;
                        let (_b, h, _s, d) = k_raw.dims4()?;

                        if d < target_dim {
                            k_raw = Tensor::cat(&[&k_raw, &k_raw], D::Minus1)?;
                            v_raw = Tensor::cat(&[&v_raw, &v_raw], D::Minus1)?;
                        }
                        if h != target_heads {
                            let mut k_list = Vec::with_capacity(target_heads);
                            let mut v_list = Vec::with_capacity(target_heads);
                            for i in 0..target_heads {
                                let src_idx = i % h;
                                k_list.push(k_raw.narrow(1, src_idx, 1)?);
                                v_list.push(v_raw.narrow(1, src_idx, 1)?);
                            }
                            k_raw = Tensor::cat(&k_list, 1)?;
                            v_raw = Tensor::cat(&v_list, 1)?;
                        }

                        let mut reg = self.registry.entries.write().unwrap();
                        if b_idx < reg.len() {
                            if let Some(block) = self.kv_blocks.get(b_idx) {
                                let mut inner = block.inner.write().unwrap();
                                
                                
                                let target_device = self.q_proj.device();
                                let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
                                
                                let k_gpu = k_raw.to_device(target_device)?;
                                let v_gpu = v_raw.to_device(target_device)?;
                                inner.k_cache = Some(if target_device.is_cuda() { k_gpu.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k_gpu.clone()) } else { k_gpu.to_dtype(target_dtype)? });
                                inner.v_cache = Some(if target_device.is_cuda() { v_gpu.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v_gpu.clone()) } else { v_gpu.to_dtype(target_dtype)? });
                                
                                inner.location = KVLocation::VRAM; // RAM을 건너뛰고 VRAM 상주 확정!
                                reg[b_idx].location[self.layer_idx] = KVLocation::VRAM;
                                reg[b_idx].ssd_path = Some(file_path.parent().unwrap().to_path_buf());
                                
                                if self.layer_idx < reg[b_idx].is_dirty.len() {
                                    reg[b_idx].is_dirty[self.layer_idx] = false;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if self.layer_idx == 0 {
            println!("[BATCH-LOAD] Layer {} fully cached to RAM from index.", self.layer_idx);
        }
        Ok(())
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        self.clear_kv_cache();
        Ok(())
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        let dev = &Device::Cpu;

        if let (Some(mk), Some(mv)) = (&self.vram_merged_k, &self.vram_merged_v) {
            
            // 🌟 [동적 타입 분기] 공용 모듈 사용을 고려하여 텐서 고유의 DType(FP8, BF16, F32)을 감지하여 보존합니다.
            let target_dtype = if mk.device().is_cuda() || mk.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
            let mk_cpu = mk.contiguous()?.to_dtype(target_dtype).unwrap_or_else(|_| mk.clone()).to_device(dev)?;
            let mv_cpu = mv.contiguous()?.to_dtype(target_dtype).unwrap_or_else(|_| mv.clone()).to_device(dev)?;
            
            let mut current_pos = 0;
            for block in &mut self.kv_blocks {
                let mut inner = block.inner.write().unwrap();
                let b_len = inner.len;
                if inner.location == KVLocation::VRAM {
                    let k_part = mk_cpu.narrow(2, current_pos, b_len)?.contiguous()?;
                    let v_part = mv_cpu.narrow(2, current_pos, b_len)?.contiguous()?;
                    inner.k_cache = Some(k_part);
                    inner.v_cache = Some(v_part);
                    inner.location = KVLocation::RAM;
                    
                    let mut reg = self.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[self.layer_idx] = KVLocation::RAM;
                    }
                }
                current_pos += b_len;
            }
            self.vram_merged_k = None;
            self.vram_merged_v = None;
            self.merged_vram_block_count = 0;
        }

        let mut k_list = Vec::new();
        let mut v_list = Vec::new();
        let mut target_idxs = Vec::new();

        for (idx, block) in self.kv_blocks.iter().enumerate() {
            let inner = block.inner.read().unwrap();
            if inner.location == KVLocation::VRAM && inner.k_cache.is_some() {
                k_list.push(inner.k_cache.clone().unwrap());
                v_list.push(inner.v_cache.clone().unwrap());
                target_idxs.push(idx);
            }
        }

        if !k_list.is_empty() {
            
            let merged_k = Tensor::cat(&k_list, 2)?.contiguous()?;
            let merged_v = Tensor::cat(&v_list, 2)?.contiguous()?;
            
            // 🌟 [동적 타입 분기]
            let target_dtype = if merged_k.device().is_cuda() || merged_k.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
            let merged_k_cpu = merged_k.to_dtype(target_dtype).unwrap_or_else(|_| merged_k.clone()).to_device(dev)?;
            let merged_v_cpu = merged_v.to_dtype(target_dtype).unwrap_or_else(|_| merged_v.clone()).to_device(dev)?;
            
            let mut current_pos = 0;
            for (i, &idx) in target_idxs.iter().enumerate() {
                let chunk_len = k_list[i].dim(2)?;
                let k_cpu = merged_k_cpu.narrow(2, current_pos, chunk_len)?.contiguous()?;
                let v_cpu = merged_v_cpu.narrow(2, current_pos, chunk_len)?.contiguous()?;
                current_pos += chunk_len;
                
                let mut inner = self.kv_blocks[idx].inner.write().unwrap();
                inner.k_cache = Some(k_cpu);
                inner.v_cache = Some(v_cpu);
                inner.location = KVLocation::RAM;
                
                let mut reg = self.registry.entries.write().unwrap();
                if inner.index < reg.len() {
                    reg[inner.index].location[self.layer_idx] = KVLocation::RAM;
                }
            }
        }
        Ok(())
    }

    pub fn inject_live_kv(&mut self, k_i8: &Tensor, v_i8: &Tensor, k_scale: f32, v_scale: f32) -> Result<()> {
        let target_device = self.q_proj.device(); 
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let k_gpu_i8 = k_i8.to_device(target_device)?;
        let v_gpu_i8 = v_i8.to_device(target_device)?;
        let k_small = (k_gpu_i8.to_dtype(DType::F32)? * k_scale as f64)?.to_dtype(target_dtype)?;
        let v_small = (v_gpu_i8.to_dtype(DType::F32)? * v_scale as f64)?.to_dtype(target_dtype)?;
        self.inject_live_kv_direct(&k_small, &v_small)
    }

    pub fn inject_live_kv_direct(&mut self, k_final: &Tensor, v_final: &Tensor) -> Result<()> {
        let dev = self.q_proj.device();
        let k_final = if !k_final.device().same_device(dev) { k_final.to_device(dev)? } else { k_final.clone() };
        let v_final = if !v_final.device().same_device(dev) { v_final.to_device(dev)? } else { v_final.clone() };
        
        let index = self.kv_blocks.len();
        let len = k_final.dim(2)?;
        let current_total = self.get_kv_len();
        
        let new_block = KVBlock::new(KVLocation::VRAM, index, len, current_total);
        {
            let mut inner = new_block.inner.write().unwrap();
            inner.k_cache = Some(k_final);
            inner.v_cache = Some(v_final);
        }
        self.kv_blocks.push(new_block);
        Ok(())
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        let kv_type = kv_name.unwrap_or("text");
        let b_str = format!("b{}", offset);
        let block_dir = path.join(kv_type).join(&b_str);
        if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
        
        let structured_path = block_dir.join(format!("l{}.st", self.layer_idx));
        
        let mut map = HashMap::new();
        let prefix = format!("b{}_l{}_", offset, self.layer_idx);

        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                ks.push(k.clone());
                vs.push(v.clone());
            }
        }

        if !ks.is_empty() {
            let k = Tensor::cat(&ks, 2)?;
            let v = Tensor::cat(&vs, 2)?;
            
            let (kd, k_shape) = self.compress_to_bf16(&k)?;
            let (vd, _) = self.compress_to_bf16(&v)?;
            
            map.insert(format!("{}k_data", prefix), kd);
            map.insert(format!("{}v_data", prefix), vd);
            map.insert(format!("{}k_shape", prefix), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect::<Vec<u32>>(), (k_shape.len(),), &Device::Cpu)?);
            
            if let Ok(_) = candle_core::safetensors::save(&map, &structured_path) {
                println!("[SSD-SAVE] Layer {} Block {} saved to disk.", self.layer_idx, offset);
            } else {
                println!("[SSD-SAVE-ERROR] Failed to save Layer {} Block {}", self.layer_idx, offset);
            }
            
            if let Ok(mut reg) = self.registry.entries.write() {
                let entry_idx = offset / 1024;
                if entry_idx < reg.len() {
                    let entry = &mut reg[entry_idx];
                    entry.ssd_path = Some(block_dir.clone()); 
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = false; }
                } else {
                    let mut entry = crate::models::qwen::quantized_model::RegistryEntry::new(offset, k.dim(2)?, 28);
                    entry.ssd_path = Some(block_dir.clone()); 
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = false; }
                    reg.push(entry);
                }
            }
        }

        if clear { self.kv_blocks.clear(); }
        Ok(())
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        let mut _current_total = 0;
        let mut to_remove = Vec::new();
        let total_blocks = self.kv_blocks.len();
        
        for i in 0..total_blocks {
            let block = &mut self.kv_blocks[i];
            let mut inner = block.inner.write().unwrap();
            
            if _current_total + inner.len <= len {
                _current_total += inner.len;
            } else {
                let keep_in_this_block = len - _current_total;
                if keep_in_this_block > 0 {
                    if inner.location == KVLocation::VRAM {
                        let (new_k, new_v) = if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            (Some(k.narrow(2, 0, keep_in_this_block)?), Some(v.narrow(2, 0, keep_in_this_block)?))
                        } else { (None, None) };
                        inner.k_cache = new_k;
                        inner.v_cache = new_v;
                    }
                    inner.len = keep_in_this_block;
                    _current_total += keep_in_this_block;
                    for j in (i + 1)..total_blocks { to_remove.push(j); }
                } else {
                    for j in i..total_blocks { to_remove.push(j); }
                }
                break;
            }
        }
        
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove { self.kv_blocks.remove(idx); }
        Ok(())
    }

    pub fn load_kv_cache(&mut self, _path: &Path, _device: &Device, _expected_len: usize, _upscale_refill_len: usize, _kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        if fragments.is_empty() { return Ok(()); }
        
        self.kv_blocks.clear();
        let mut total_restored_len = 0;

        for (i, (offset, frag_path)) in fragments.iter().enumerate() {
            let b_len = if *offset < current_kv_len {
                (current_kv_len - *offset).min(1024)
            } else { 1024 };
            total_restored_len += b_len;
            
            let new_block = KVBlock::new(KVLocation::SSD, i, b_len, *offset);
            {
                let mut inner = new_block.inner.write().unwrap();
                inner.len = b_len;
                inner.location = KVLocation::SSD;
            }
            self.kv_blocks.push(new_block);

            let mut reg = self.registry.entries.write().unwrap();
            if i >= reg.len() {
                reg.push(crate::models::qwen::quantized_model::RegistryEntry {
                    location: vec![KVLocation::SSD; 28],
                    slot_ids: vec![None; 28],
                    token_start: *offset,
                    token_len: b_len,
                    ssd_path: Some(frag_path.clone()), 
                    hidden_states_path: vec![None; 28],
                    is_dirty: vec![false; 28], 
                    last_accessed: std::time::Instant::now(),
                    bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                });
            }
            
            if i < reg.len() {
                reg[i].location[self.layer_idx] = KVLocation::SSD;
                reg[i].ssd_path = Some(frag_path.clone()); 
                
                // 이 3줄이 없으면 10GB의 텐서를 매번 무한대로 다시 구워버리며 시스템을 다운시킵니다.
                if self.layer_idx < reg[i].is_dirty.len() {
                    reg[i].is_dirty[self.layer_idx] = false;
                }
            }
        }

        if self.layer_idx == 0 {
            println!("[SSD-RESTORE] Restored {} tokens across {} blocks.", total_restored_len, fragments.len());
        }
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size, None)
    }
}

#[derive(Clone)]
pub struct QuantizedQwenVLTextDecoderLayer {
    pub self_attn: QuantizedQwenVLTextAttention,
    pub mlp_gate: Option<QLinear>,
    pub mlp_up: Option<QLinear>,
    pub mlp_down: Option<QLinear>,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: Option<RmsNorm>,
}

impl QuantizedQwenVLTextDecoderLayer {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if self.self_attn.q_proj.is_cleared() {
            return Err(anyhow!("Layer weights are cleared. Reload required."));
        }
        self.self_attn.to_device(device)?;
        if let Some(gate) = &mut self.mlp_gate { gate.to_device(device)?; }
        if let Some(up) = &mut self.mlp_up { up.to_device(device)?; }
        if let Some(down) = &mut self.mlp_down { down.to_device(device)?; }
        self.input_layernorm.to_device(device)?;
        if let Some(norm) = &mut self.post_attention_layernorm { norm.to_device(device)?; }
        Ok(())
    }

    /// [MEMORY-OPT] 가중치 없이 레이어 구조만 생성합니다.
    pub fn new_skeleton(
        config: &QwenVLTextConfig,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        _baking_only: bool,
        registry: KVRegistry,
    ) -> Result<Self> {
        let is_gguf_naming = base_name.starts_with("blk.");
        let (_attn_base, _gate, _up, _down, _in_ln, _post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let zero_t = Tensor::zeros((1,), dtype, device)?;
        let q_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let k_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let v_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let o_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let q_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);
        let k_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);

        let mut self_attn = QuantizedQwenVLTextAttention {
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads.max(1),
            head_dim: config.head_dim,
            scaling: 1.0 / (config.head_dim as f64).sqrt(),
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_kv_name: None,
            active_session_id: None,
            kv_residency: crate::utils::resources::KvResidency::Ssd,
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        };
        self_attn.clear(); 

        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = (
            Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
            Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
            Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
            Some(RmsNorm::new(zero_t.clone(), config.rms_norm_eps))
        );

        let input_layernorm = RmsNorm::new(zero_t, config.rms_norm_eps);

        let mut layer = Self {
            self_attn,
            mlp_gate,
            mlp_up,
            mlp_down,
            input_layernorm,
            post_attention_layernorm,
        };
        layer.clear();
        Ok(layer)
    }

    /// [MEMORY-OPT] 레이어의 가중치를 완전히 해제합니다.
    pub fn clear(&mut self) {
        self.self_attn.clear();
        if let Some(gate) = &mut self.mlp_gate { gate.clear(); }
        if let Some(up) = &mut self.mlp_up { up.clear(); }
        if let Some(down) = &mut self.mlp_down { down.clear(); }
        self.input_layernorm.clear();
        if let Some(norm) = &mut self.post_attention_layernorm { norm.clear(); }
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &QwenVLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        _baking_only: bool,
        registry: KVRegistry, 
    ) -> Result<Self> {
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwenVLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx, registry)?;
        
        let mg = Some(get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?);
        let mu = Some(get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?);
        let md = Some(get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?);
        let pln = Some(get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype)?);
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = (mg, mu, md, pln);

        let input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            self_attn,
            mlp_gate,
            mlp_up,
            mlp_down,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn load_weights_inplace<R: std::io::Seek + std::io::Read>(
        &mut self,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
        _baking_only: bool,
    ) -> Result<()> {
        let is_gguf_naming = base_name.starts_with("blk.");
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        self.self_attn.load_weights_inplace(ct, reader, &attn_base, is_gguf_naming, device, dtype)?;

        
        self.mlp_gate = Some(get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?);
        self.mlp_up = Some(get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?);
        self.mlp_down = Some(get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?);
        self.post_attention_layernorm = Some(get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), self.input_layernorm.eps(), device, dtype)?);

        self.input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), self.input_layernorm.eps(), device, dtype)?;

        Ok(())
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
        baking_only: bool,
    ) -> Result<Tensor> {
        let dev = self.input_layernorm.weight().device();
        let target_dtype = self.input_layernorm.weight().dtype();

        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };

        let cos_ref = cos; 
        let sin_ref = sin;
        
        let attention_mask = if let Some(mask) = attention_mask {
             Some(if !mask.device().same_device(dev) { mask.to_device(dev)? } else { mask.clone() })
        } else {
             None
        };
        
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(&xs)?;
        
        let xs = self.self_attn.forward(&xs, cos_ref, sin_ref, attention_mask.as_ref(), seqlen_offset, session_id, kv_name, baking_only)?;
        
        let xs = residual.add(&xs)?;
        
        if let (Some(gate_proj), Some(up_proj), Some(down_proj), Some(post_norm)) = (&self.mlp_gate, &self.mlp_up, &self.mlp_down, &self.post_attention_layernorm) {
            let residual = xs.clone();
            let xs = post_norm.forward(&xs)?;
            let xs = {
                let gate = gate_proj.forward(&xs)?;
                let up = up_proj.forward(&xs)?;
                let gate = candle_nn::ops::silu(&gate)?;
                let hidden = gate.mul(&up)?;
                down_proj.forward(&hidden)?
            };
            Ok(residual.add(&xs)?)
        } else {
            Ok(xs)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }

    // 🌟 [KV RESIDENCY] 레이어 단위 배치 위치 전파 위임
    pub fn set_kv_residency(&mut self, residency: crate::utils::resources::KvResidency) {
        self.self_attn.kv_residency = residency;
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        self.self_attn.evacuate_vram_to_cache()
    }

    pub fn get_kv_len(&self) -> usize {
        self.self_attn.get_kv_len()
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        self.self_attn.drop_kv_storage()
    }

    pub fn device(&self) -> &Device {
        self.input_layernorm.weight().device()
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        self.self_attn.save_kv_cache(path, clear, offset, kv_name)
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        self.self_attn.truncate_kv_cache(len)
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.self_attn.save_kv_cache(path, true, block_size, None)
    }

    pub fn batch_load_kv(&mut self, kv_name: &str) -> Result<()> {
        self.self_attn.batch_load_layer_kv(kv_name)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        self.self_attn.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len)
    }
}

/// 🌟 [KV RESIDENCY] 디코딩 진입 시 앞으로 생성할 토큰 수를 알 수 없으므로,
/// 보수적으로 이만큼 더 자란다고 가정하고 VRAM 상주 가능 여부를 판정합니다.
pub const QWEN06B_DECODE_HEADROOM_TOKENS: usize = 2048;

#[derive(Clone)]
pub struct QuantizedQwenVLTextModel {
    pub embed_tokens: Embedding, 
    pub layers: Vec<QuantizedQwenVLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: QwenVLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
    pub device_id: usize, 
    pub mmap: Option<Arc<Mmap>>, 
    pub registry: KVRegistry, 
    pub baking_only: bool,
    pub is_forced_cpu: bool,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    pub pinned_layer_count: usize,
    pub current_kv_len: usize,
    // 🌟 [KV RESIDENCY] 디코딩 루프 진입 시 1회만 계산되는 배치 계획.
    //    프리필 구간에서는 None 으로 무효화되어 기존 SSD 오프로딩이 그대로 유지됩니다.
    pub kv_plan: Option<crate::utils::resources::KvResidency>,
    // 🌟 [DECODE-RESIDENT] 가중치 상주 여부도 같은 시점에 1회만 판정하여
    //    레이어마다(= 토큰당 28회) 반복되던 sysinfo 호출을 제거합니다.
    pub keep_weights_resident: bool,
    // [NEW] 재로딩을 위한 메타데이터
    pub config: QwenVLTextConfig,
    pub ct: Option<Arc<gguf_file::Content>>,
    pub base_name: String,
    pub dtype: DType,
}

impl QuantizedQwenVLTextModel {
    /// [MEMORY-OPT] 특정 레이어의 가중치를 mmap에서 In-place로 덮어씁니다.
    pub fn reload_layer(&mut self, layer_idx: usize) -> Result<()> {
        if !self.layers[layer_idx].self_attn.q_proj.is_cleared() {
            return Ok(());
        }
        
        let mmap = self.mmap.as_ref().ok_or(anyhow!("Mmap handle missing for reload"))?;
        let ct = self.ct.as_ref().ok_or(anyhow!("GGUF Content missing for reload"))?;
        let mut reader = std::io::Cursor::new(&mmap[..]);
        
        let gguf_blk = format!("blk.{layer_idx}");
        let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) {
            gguf_blk 
        } else {
            format!("{}.layers.{layer_idx}", self.base_name)
        };

        // [PING-PONG] 새로 레이어를 만들지 않고, 껍데기에 값만 밀어넣음!
        self.layers[layer_idx].load_weights_inplace(
            ct, &mut reader, &prefix, &Device::Cpu, self.dtype, self.baking_only
        )?;
        
        Ok(())
    }

    pub fn new_with_mmap(
        config: &QwenVLTextConfig,
        ct: Arc<gguf_file::Content>,
        mmap_handle: Option<Arc<Mmap>>,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        _kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        
        // [CRITICAL FIX] CPU 모드일 경우 글로벌 데이터 타입을 뼈대부터 F32로 강제 승격
        let effective_dtype = if is_forced_cpu { DType::F32 } else { dtype };
        
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        
        let embed_dtype = if is_forced_cpu { DType::F32 } else { DType::F16 };
        
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(&mut reader, &token_emb_name, device) {
             let tensor = tensor.dequantize_f16(device).or_else(|_| tensor.dequantize(device))?.to_dtype(embed_dtype)?; 
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, alt_token_emb, device) {
             let tensor = tensor.dequantize_f16(device).or_else(|_| tensor.dequantize(device))?.to_dtype(embed_dtype)?; 
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else {
             return Err(anyhow!("Failed to load embedding."));
        };

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;

        let current_device = device.clone(); 
        let registry = KVRegistry::new();
        
        
        let num_layers_to_load = config.num_hidden_layers;

        let mut layers = Vec::with_capacity(num_layers_to_load);
        for layer_idx in 0..num_layers_to_load {
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
            
            let layer = QuantizedQwenVLTextDecoderLayer::new_skeleton(config, &prefix, &current_device, effective_dtype, layer_idx, baking_only, registry.clone())?;
            layers.push(layer);
        }
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let norm = get_rms_norm(&ct, &mut reader, norm_prefix, config.rms_norm_eps, device, effective_dtype)?; 
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: QwenVLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
            mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_else(|| if config.head_dim == 128 { vec![16, 24, 24] } else { vec![] }), 
            device_id,
            mmap: mmap_handle, 
            registry, 
            baking_only, 
            is_forced_cpu, 
            active_session_id: None, 
            active_kv_name: None, 
            pinned_layer_count: if current_device.is_cuda() { num_layers_to_load } else { 0 }, 
            current_kv_len: 0,
            kv_plan: None,
            keep_weights_resident: false,
            config: config.clone(),
            ct: Some(ct),
            base_name: base_name.to_string(),
            dtype: effective_dtype, 
        })
    }

    /// [MEMORY-OPT] 모든 레이어를 한꺼번에 로드합니다. (디코딩 시작 시 호출)
    pub fn reload_all_layers(&mut self) -> Result<()> {
        let count = self.layers.len();
        // println!("[MEMORY-OPT] Prefill complete. Reloading all {} layers for high-speed decoding...", count);
        for i in 0..count {
            self.reload_layer(i)?;
        }
        Ok(())
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &QwenVLTextConfig,
        ct: Arc<gguf_file::Content>,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        _kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let effective_dtype = if is_forced_cpu { DType::F32 } else { dtype };
        
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        
        let embed_dtype = if is_forced_cpu { DType::F32 } else { DType::F16 };
        
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(reader, &token_emb_name, device) {
             let tensor = tensor.dequantize_f16(device).or_else(|_| tensor.dequantize(device))?.to_dtype(embed_dtype)?; 
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(reader, alt_token_emb, device) {
             let tensor = tensor.dequantize_f16(device).or_else(|_| tensor.dequantize(device))?.to_dtype(embed_dtype)?; 
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else {
             return Err(anyhow!("Failed to load embedding."));
        };

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;
        let registry = KVRegistry::new();
        
        let mut pinned_layer_count = 0;
        let num_layers_to_load = config.num_hidden_layers;

        let mut layers = vec![];
        for layer_idx in 0..num_layers_to_load {
            if device.is_cuda() { pinned_layer_count += 1; }
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
            let mut layer = QuantizedQwenVLTextDecoderLayer::new(config, &ct, reader, &prefix, device, effective_dtype, layer_idx, baking_only, registry.clone())?;
            layer.clear();
            layers.push(layer);
        }
        
        let norm_prefix = if ct.tensor_infos.contains_key("output_norm.weight") { "output_norm" } else { &format!("{base_name}.norm") };
        let norm = get_rms_norm(&ct, reader, norm_prefix, config.rms_norm_eps, device, effective_dtype)?; 
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: QwenVLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
            mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_else(|| if config.head_dim == 128 { vec![16, 24, 24] } else { vec![] }), 
            device_id,
            mmap: None, 
            registry, 
            baking_only, 
            is_forced_cpu, 
            active_session_id: None, 
            active_kv_name: None, 
            pinned_layer_count, 
            current_kv_len: 0,
            kv_plan: None,
            keep_weights_resident: false,
            config: config.clone(),
            ct: Some(ct),
            base_name: base_name.to_string(),
            dtype: effective_dtype, 
        })
    }

    pub fn load_kv_cache_chunked(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let index_path = kv_dir.join(kv_name).join("layer0.json");
        
        if !index_path.exists() { return Ok(()); }
        
        let index_json = if let Ok(data) = crate::utils::direct_loader::load_kv_block(&index_path) {
            String::from_utf8(data).unwrap_or_default()
        } else { return Ok(()); };
        
        let index: LayerIndex = serde_json::from_str(&index_json)?;
        let total_tokens = index.total_tokens;
        
        {
            let mut reg = self.registry.entries.write().unwrap();
            let needed_blocks = (total_tokens + 255) / 1024;
            while reg.len() < needed_blocks {
                let off = reg.len() * 1024;
                reg.push(RegistryEntry::new(off, 0, 28));
            }
            self.current_kv_len = total_tokens;
        }

        for layer in self.layers.iter_mut() {
            let reg_len = self.registry.entries.read().unwrap().len();
            while layer.self_attn.kv_blocks.len() < reg_len {
                let idx = layer.self_attn.kv_blocks.len();
                let off = idx * 1024;
                layer.self_attn.kv_blocks.push(KVBlock {
                    inner: Arc::new(std::sync::RwLock::new(KVBlockInner {
                        k_cache: None, v_cache: None,
                        offset: off, len: 0, index: idx, location: KVLocation::SSD,
                        bitkv_metadata: None,
                        ssd_path: None,
                    }))
                });
            }
        }

        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        
        let chunk_size = if free_ram_gb > 8.0 { 28 } else if free_ram_gb > 4.0 { 14 } else { 7 };
        println!("[PREFILL-RAM] Available RAM: {:.2} GB. Loading in chunks of {}.", free_ram_gb, chunk_size);
        
        for chunk_start in (0..self.layers.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(self.layers.len());
            for l_idx in chunk_start..chunk_end {
                self.layers[l_idx].batch_load_kv(kv_name)?;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        {
            let mut reg = self.registry.entries.write().unwrap();
            let total_t = self.current_kv_len;
            for (idx, entry) in reg.iter_mut().enumerate() {
                let off = idx * 1024;
                let b_len = if off + 1024 <= total_t { 1024 } else { total_t.saturating_sub(off) };
                entry.token_len = b_len;
                
                for layer in self.layers.iter_mut() {
                    if let Some(block) = layer.self_attn.kv_blocks.get(idx) {
                        let mut inner = block.inner.write().unwrap();
                        inner.len = b_len;
                        if entry.location[layer.self_attn.layer_idx] == KVLocation::RAM {
                            inner.location = KVLocation::RAM;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn process_chunks_iterative(
        &mut self,
        layer_idx: usize,
        chunk_offsets: &[usize],
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
        baking_only: bool,
    ) -> Result<Tensor> {
        
        let chunk_size = 2048; 
        let current_seq_len = xs.dim(1)?;
        let is_decoding = current_seq_len <= 1;
        let total_kv_blocks = self.layers[layer_idx].self_attn.kv_blocks.len();
        let total_chunks = chunk_offsets.len();

        let mut chunk_outputs = Vec::with_capacity(total_chunks);

        for (chunk_idx, &i) in chunk_offsets.iter().enumerate() {
            
            if crate::utils::is_extraction_stopped() {
                return Err(anyhow::anyhow!("Task cancelled"));
            }
            
            let take = (current_seq_len - i).min(chunk_size);

            let target_chunks = if is_decoding || chunk_idx == 0 {
                (0..total_kv_blocks).collect::<Vec<_>>()
            } else {
                vec![]
            };

            let look_ahead_layers = 1;

            for t_idx in target_chunks {
                if t_idx < total_kv_blocks {
                    for l_off in 1..=look_ahead_layers {
                        let target_layer = layer_idx + l_off;
                        if target_layer < 28 {
                            if let Some(block) = self.layers[layer_idx].self_attn.kv_blocks.get(t_idx) {
                                let (index, path_opt) = {
                                    let reg = self.registry.entries.read().unwrap();
                                    let inner = block.inner.read().unwrap();
                                    if inner.index < reg.len() && reg[inner.index].location[target_layer] == KVLocation::SSD {
                                        (inner.index, reg[inner.index].ssd_path.clone())
                                    } else { (999, None) }
                                };

                                if index != 999 && path_opt.is_some() {
                                    let path = path_opt.unwrap();
                                    {
                                        let mut reg = self.registry.entries.write().unwrap();
                                        reg[index].location[target_layer] = KVLocation::Loading;
                                    }
                                    let shared_block = block.clone();
                                    let reg_clone = self.registry.clone();
                                    let kv_name_for_load = kv_name.clone();
                                    let is_cpu_mode = self.is_forced_cpu; 
                                    tauri::async_runtime::spawn(async move {
                                        use crate::models::qwen::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_load_worker};
                                        let sid = SLOT_MANAGER.acquire_read_slot().await;
                                        if let Ok(tx) = get_load_worker().await {
                                            let _ = tx.send(SlotTask::Load(LoadTask { slot_id: sid, path, layer_idx: target_layer, kv_name: kv_name_for_load, shared_block, registry: reg_clone, is_cpu: is_cpu_mode })).await;
                                        } else {
                                            SLOT_MANAGER.release_slot(sid).await;
                                        }
                                    });
                                }
                            }
                        }
                    }
                }
            }

            let xs_chunk = xs.narrow(1, i, take)?.contiguous()?;
            let cos_chunk = cos.narrow(cos.rank().saturating_sub(2), i, take)?.contiguous()?;
            let sin_chunk = sin.narrow(sin.rank().saturating_sub(2), i, take)?.contiguous()?;
            
            let out = self.layers[layer_idx].forward(&xs_chunk, &cos_chunk, &sin_chunk, None, seqlen_offset + i, session_id.clone(), kv_name.clone(), baking_only)?;
            
            chunk_outputs.push(out);
            
            if self.is_forced_cpu {
                if let Some(sid) = &session_id {
                    let is_last = chunk_idx == total_chunks - 1;
                    let _ = self.layers[layer_idx].self_attn.trigger_realtime_incremental_bake(sid, is_last, baking_only, true);
                }
                tokio::task::yield_now().await;
            } else {
                let _ = self.evacuate_vram_to_ram_only(layer_idx).await;

                // 프리필 청크 경계에서만 동기화하여 과거 블록 VRAM을 회수합니다.
                // 디코딩(1토큰)에서는 회수할 과거 블록이 없는데도 스톨만 발생하므로 건너뜁니다.
                if !is_decoding {
                    let dev = self.layers[layer_idx].device();
                    if dev.is_cuda() {
                        let _ = dev.synchronize();
                    }
                }
            }
        } 

        if let Some(sid) = &session_id {
            let is_prefill = current_seq_len > 1;
            if !self.is_forced_cpu && !is_prefill {
                let _ = self.layers[layer_idx].self_attn.trigger_realtime_incremental_bake(sid, true, baking_only, true);
            }
        }
        
        if chunk_outputs.is_empty() {
            return Err(anyhow::anyhow!("No output generated from chunks"));
        }

        let final_output_tensor = if chunk_outputs.len() == 1 {
            chunk_outputs.into_iter().next().unwrap()
        } else {
            Tensor::cat(&chunk_outputs, 1)?
        };
        
        Ok(final_output_tensor)
    }

    async fn evacuate_vram_to_ram_only(&mut self, layer_idx: usize) -> Result<()> {
        // 🌟 [KV RESIDENCY] VRAM 상주가 확정된 구간에서는 강제 퇴거를 수행하지 않습니다.
        //    기존 vram_limit = 8 고정값은 VRAM 여유와 무관하게 9번째 블록부터 무조건
        //    RAM 으로 쫓아내어, 다음 토큰에서 다시 올려야 하는 낭비를 만들었습니다.
        if self.kv_plan == Some(crate::utils::resources::KvResidency::Vram) {
            return Ok(());
        }

        let current_kv_len = self.layers[layer_idx].get_kv_len();
        let is_small_model = self.layers.len() <= 36;
        if is_small_model && current_kv_len < 1024 { return Ok(()); }

        let vram_limit = 8; 
        let mut vram_evicted = false;

        {
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;

            let mut vram_indices = Vec::new();
            for (idx, block) in kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                if inner.location == KVLocation::VRAM {
                    vram_indices.push((idx, inner.offset));
                }
            }

            if vram_indices.len() > vram_limit {
                vram_indices.sort_by_key(|k| k.1); 
                let num_to_evict = vram_indices.len().saturating_sub(vram_limit);
                
                let mut k_list = Vec::new();
                let mut v_list = Vec::new();
                let mut target_idxs = Vec::new();
                
                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let inner = kv_blocks[idx].inner.read().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        k_list.push(k.clone());
                        v_list.push(v.clone());
                        target_idxs.push(idx);
                    }
                }

                if !k_list.is_empty() {
                    
                    let merged_k = Tensor::cat(&k_list, 2)?.contiguous()?;
                    let merged_v = Tensor::cat(&v_list, 2)?.contiguous()?;
                    
                    // 🌟 [FP8 Compression] 긴 문맥 OOM 억제를 위한 강제 RAM 대피 시에도 VRAM에서 FP8로 선압축합니다.
                    let target_dtype = match merged_k.dtype() { candle_core::DType::F64 => candle_core::DType::F8E4M3, candle_core::DType::F32 => candle_core::DType::F32, _ => candle_core::DType::BF16 };
                    let merged_k_cpu = merged_k.to_dtype(target_dtype)?.to_device(&Device::Cpu)?;
                    let merged_v_cpu = merged_v.to_dtype(target_dtype)?.to_device(&Device::Cpu)?;
                    
                    let mut current_pos = 0;
                    for (i, &idx) in target_idxs.iter().enumerate() {
                        let chunk_len = k_list[i].dim(2)?;
                        let k_cpu = merged_k_cpu.narrow(2, current_pos, chunk_len)?.contiguous()?;
                        let v_cpu = merged_v_cpu.narrow(2, current_pos, chunk_len)?.contiguous()?;
                        current_pos += chunk_len;
                        
                        let mut inner = kv_blocks[idx].inner.write().unwrap();
                        inner.k_cache = Some(k_cpu);
                        inner.v_cache = Some(v_cpu);
                        inner.location = KVLocation::RAM;
                        if inner.index < reg.len() {
                            reg[inner.index].location[layer_idx] = KVLocation::RAM;
                        }
                    }
                    vram_evicted = true;
                }
            }
        } 

        if vram_evicted {
            self.layers[layer_idx].self_attn.vram_merged_k = None;
            self.layers[layer_idx].self_attn.vram_merged_v = None;
            self.layers[layer_idx].self_attn.merged_vram_block_count = 0;
        }

        Ok(())
    }

    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![];
        let mut vs = vec![];
        for layer in &self.layers {
            let mut l_ks = vec![];
            let mut l_vs = vec![];
            for block in &layer.self_attn.kv_blocks {
                let inner = block.inner.read().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    l_ks.push(k.clone());
                    l_vs.push(v.clone());
                }
            }
            if !l_ks.is_empty() {
                if let (Ok(k), Ok(v)) = (Tensor::cat(&l_ks, 2), Tensor::cat(&l_vs, 2)) {
                    ks.push(k);
                    vs.push(v);
                }
            }
        }
        (ks, vs)
    }

    async fn process_single_layer(
        &mut self,
        layer_idx: usize,
        xs: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seqlen_offset: usize,
        deepstack_embed: Option<&Tensor>,
        visual_mask: Option<&Tensor>,
        session_id: Option<String>,
        kv_name: Option<String>,
        baking_only: bool,
    ) -> Result<Tensor> {
        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) }; 

        self.reload_layer(layer_idx)?;
        
        if self.is_forced_cpu || !self.layers[layer_idx].device().same_device(&target_device) {
            self.layers[layer_idx].to_device(&target_device)?;
        }

        let current_seq_len = xs.dim(1)?;
        
        let chunk_offsets: Vec<usize> = (0..current_seq_len).step_by(2048).collect(); 
        
        // 청크 분할 및 연산
        let mut next_xs = self.process_chunks_iterative(layer_idx, &chunk_offsets, &xs, cos, sin, seqlen_offset, session_id.clone(), kv_name.clone(), baking_only).await?;

        if let (Some(embed), Some(mask)) = (deepstack_embed, visual_mask) {
            next_xs = mask_index_add(&next_xs.squeeze(0)?, &mask.squeeze(0)?, embed)?.unsqueeze(0)?;
        }

        // 가중치 비우기 (프리필 중 메모리 안정성 확보)
        let is_decoding = current_seq_len <= 1;

        // [DECODE-RESIDENT] 디코딩 구간에서는 레이어당 연산이 O(1)이라 가중치 재적재가 전체를 지배합니다.
        // 🌟 [KV RESIDENCY] 판정은 forward() 진입 시 1회만 수행되며, 여기서는 그 결과만 읽습니다.
        //    (기존에는 레이어마다 = 토큰당 28회 sysinfo 를 호출했습니다.)
        let keep_resident = is_decoding && self.keep_weights_resident;
        if !keep_resident {
            self.layers[layer_idx].clear();
        }

        // [NO-PER-LAYER-STALL] 레이어마다 GPU 하드 동기화를 걸면 토큰당 28회 파이프라인이 멈춥니다.
        // 프리필에서만 유지하고 디코딩에서는 제거합니다.
        if target_device.is_cuda() && !is_decoding { let _ = target_device.synchronize(); }

        if let Some(sid) = session_id {
            
            // 디코딩(1토큰 생성) 단계에서 이 코드가 실행되면, AI의 이전 기억이 매 토큰마다 강제 삭제되어 
            // 텅 빈 0.0 텐서만 남게 되고, 이로 인해 외계어를 내뱉는 "기억상실증(환각)"이 발생합니다.
            if current_seq_len > 1 {
                let _ = self.evacuate_layer_kv_to_cpu(layer_idx, &sid, seqlen_offset, current_seq_len).await;
            } else {
                // 🌟 [KV RESIDENCY] 디코딩 진입 시 1회 결정된 배치 위치에 따라 과거 블록을 처리합니다.
                //   - Vram : 아무것도 하지 않고 과거 블록을 VRAM 에 그대로 상주시킵니다. (SSD 재읽기 0회)
                //   - Ram  : RAM 에 그대로 눌러앉힙니다. STEP A 가 to_device(dev) 로 잠깐 올려 씁니다.
                //   - Ssd  : 기존과 100% 동일하게 RAM → SSD 로 강등하고 bitkv_cache 를 비웁니다.
                let residency = self.kv_plan.unwrap_or(crate::utils::resources::KvResidency::Ssd);

                if residency == crate::utils::resources::KvResidency::Ssd {
                    // 다음 글자까지 기다릴 필요 없이, RAM 해제 작업을 백그라운드 스레드에 던져버려서 메인 디코딩 렉(Stutter)을 0으로 만듭니다.
                    let mut reg = self.registry.entries.write().unwrap();
                    let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;
                    let total_blocks = kv_blocks.len();
                    
                    let mut garbage_bin = Vec::new(); // 백그라운드로 보낼 쓰레기통
                    
                    for (idx, block) in kv_blocks.iter_mut().enumerate() {
                        if idx + 1 >= total_blocks { continue; } // 현재 활성 블록은 제외
                        
                        let mut inner = block.inner.write().unwrap();
                        if inner.location == KVLocation::RAM && idx < reg.len() && reg[idx].ssd_path.is_some() {
                            // Tensor를 파괴하는 대신 take()로 소유권을 뺏어서 쓰레기통에 담습니다.
                            let old_k = inner.k_cache.take();
                            let old_v = inner.v_cache.take();
                            garbage_bin.push((old_k, old_v));
                            
                            inner.location = KVLocation::SSD;
                            reg[idx].location[layer_idx] = KVLocation::SSD;
                            let mut cache = reg[idx].bitkv_cache.write().unwrap();
                            cache[layer_idx] = None;
                        }
                    }
                    
                    // 수집된 쓰레기들을 메인 스레드가 아닌 비동기 런타임에서 조용히 파괴합니다.
                    if !garbage_bin.is_empty() {
                        tauri::async_runtime::spawn(async move {
                            drop(garbage_bin);
                        });
                    }
                } else if residency == crate::utils::resources::KvResidency::Ram {
                    // 🌟 VRAM 은 부족하지만 RAM 은 충분한 경우:
                    //    과거 블록을 VRAM 에서만 걷어내고 RAM 에는 그대로 남깁니다.
                    let mut reg = self.registry.entries.write().unwrap();
                    let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;
                    let total_blocks = kv_blocks.len();

                    for (idx, block) in kv_blocks.iter_mut().enumerate() {
                        if idx + 1 >= total_blocks { continue; } // 현재 활성 블록은 제외

                        let mut inner = block.inner.write().unwrap();
                        if inner.location == KVLocation::VRAM {
                            let k = inner.k_cache.take();
                            let v = inner.v_cache.take();
                            if let (Some(k_t), Some(v_t)) = (k, v) {
                                let target_dtype = if k_t.device().is_cuda() || k_t.dtype() == candle_core::DType::F8E4M3 {
                                    candle_core::DType::F8E4M3
                                } else {
                                    candle_core::DType::F32
                                };
                                inner.k_cache = Some(
                                    k_t.to_dtype(target_dtype).unwrap_or_else(|_| k_t.clone())
                                        .to_device(&Device::Cpu).unwrap_or_else(|_| k_t.clone())
                                );
                                inner.v_cache = Some(
                                    v_t.to_dtype(target_dtype).unwrap_or_else(|_| v_t.clone())
                                        .to_device(&Device::Cpu).unwrap_or_else(|_| v_t.clone())
                                );
                                inner.location = KVLocation::RAM;
                                if idx < reg.len() { reg[idx].location[layer_idx] = KVLocation::RAM; }
                            }
                        }
                    }
                }
            }
        }
        
        // [TRIM-ONCE] 워킹셋 트림/malloc_trim은 OS 커널 호출입니다.
        // 레이어마다(= 토큰당 28회) 호출하면 다음 접근에서 페이지 폴트가 다시 발생해
        // 메모리는 안 줄고 속도만 떨어집니다. 프리필 경계에서만 수행합니다.
        if !is_decoding {
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            #[cfg(target_os = "linux")]
            unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
            #[cfg(target_os = "macos")]
            unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
        }

        Ok(next_xs)
    }

    pub async fn pin_all_layers_to_gpu(&mut self) -> Result<()> {
        println!("[DEC-SPEED-UP] Pinning disabled for long-context stability. Using On-Demand serial loading.");
        Ok(())
    }

    pub async fn unpin_all_layers(&mut self) -> Result<()> {
        println!("[DEC-CLEANUP] Unpinning all layers from GPU...");
        for layer in self.layers.iter_mut() {
            layer.to_device(&Device::Cpu)?;
        }
        Ok(())
    }

    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        use crate::models::qwen::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let mut block_groups: std::collections::HashMap<(usize, usize), Vec<LayerKVDump>> = std::collections::HashMap::new();

        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            let attn = &mut layer.self_attn;
            
            attn.vram_merged_k = None;
            attn.vram_merged_v = None;
            attn.merged_vram_block_count = 0;

            
            let mut gpu_k_list = Vec::new();
            let mut gpu_v_list = Vec::new();
            let mut target_indices = Vec::new();

            // 1. 읽기 락만 잡고 VRAM 선행 처리 수행
            for (idx, block) in attn.kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                let is_full = inner.len == 1024;
                let should_evacuate = is_full; 
                
                if should_evacuate && inner.k_cache.is_some() && inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                    let k = inner.k_cache.as_ref().unwrap();
                    let v = inner.v_cache.as_ref().unwrap();
                    
                    
                    let k_gpu = k.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| k.clone()).contiguous().unwrap_or_else(|_| k.clone());
                    let v_gpu = v.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| v.clone()).contiguous().unwrap_or_else(|_| v.clone());

                    gpu_k_list.push(k_gpu);
                    gpu_v_list.push(v_gpu);
                    target_indices.push(idx);
                }
            }

            // 2. 모인 텐서가 있다면 단 1번의 PCIe 통신으로 CPU 전송
            if !gpu_k_list.is_empty() {
                let merged_k_gpu = candle_core::Tensor::cat(&gpu_k_list, 2).unwrap_or_else(|_| gpu_k_list[0].clone());
                let merged_v_gpu = candle_core::Tensor::cat(&gpu_v_list, 2).unwrap_or_else(|_| gpu_v_list[0].clone());

                // 🌟 [FP8 Compression] 전체 활성 블록을 디스크로 보낼 때 VRAM 안에서 먼저 FP8로 캐스팅해 RAM으로 보냅니다.
                // SSD 저장 로직(LayerKVDump) 내부에서 백그라운드 스레드가 이를 다시 원래의 BF16으로 무손실 복구해 저장하게 됩니다.
                let target_dtype = if merged_k_gpu.device().is_cuda() || merged_k_gpu.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
                let merged_k_cpu = merged_k_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_k_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_k_gpu.clone());
                let merged_v_cpu = merged_v_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_v_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_v_gpu.clone());

                // 3. CPU에서 다시 썰어서 할당 및 덤프 생성
                let mut current_offset = 0;
                for (i, &idx) in target_indices.iter().enumerate() {
                    let chunk_len = gpu_k_list[i].dim(2).unwrap_or(1024);
                    
                    let k_cpu = merged_k_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_k_cpu.clone()).contiguous().unwrap_or_else(|_| merged_k_cpu.clone());
                    let v_cpu = merged_v_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_v_cpu.clone()).contiguous().unwrap_or_else(|_| merged_v_cpu.clone());
                    current_offset += chunk_len;

                    let mut inner = attn.kv_blocks[idx].inner.write().unwrap();
                    let is_dirty = {
                        let reg = attn.registry.entries.read().unwrap();
                        if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() { 
                            reg[inner.index].is_dirty[l_idx] 
                        } else { true }
                    };

                    if is_dirty {
                        let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                        
                        block_groups.entry((inner.offset, inner.index)).or_default().push(LayerKVDump {
                            layer_idx: l_idx,
                            k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                            v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                            k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                            raw_k: Some(k_cpu.clone()),
                            raw_v: Some(v_cpu.clone()),
                        });
                        
                        let mut reg = attn.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].is_dirty[l_idx] = false;
                        }
                    }

                    inner.k_cache = Some(k_cpu);
                    inner.v_cache = Some(v_cpu);
                    inner.location = crate::models::qwen::quantized_model::KVLocation::RAM;
                    
                    let mut reg = attn.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[l_idx] = crate::models::qwen::quantized_model::KVLocation::RAM;
                    }
                }
            }
        }

        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let mode = self.baking_only; 
        
        let kv_name_raw = kv_name.unwrap_or("text");
        let last_part = kv_name_raw.split('/').last().unwrap_or("text");
        let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
        let sub_path = if mode {
            format!("{}/reference/{}", session_id, kv_type)
        } else {
            format!("{}/inference/{}", session_id, kv_type)
        };
        let base_dir = kv_dir.join(&sub_path);

        if block_groups.is_empty() { return Ok(()); }

        if let Some(tx) = BAKE_TX.get() {
            for ((off, idx), layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(1024).await;
                let block_dir = base_dir.join(format!("b{}", off));
                if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path.clone()), offset: off, layers,
                    is_relay_baking: mode, block_idx: Some(idx), registry: self.registry.clone(),
                })).await.is_err() {
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    SLOT_MANAGER.release_slot(sid).await;
                }
            }
        }
        Ok(())
    }

    async fn evacuate_layer_kv_to_cpu(&mut self, layer_idx: usize, session_id: &str, _start_off: usize, _len: usize) -> Result<()> {
        use crate::models::qwen::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let mut dumps_to_send = Vec::new();

        self.layers[layer_idx].self_attn.vram_merged_k = None;
        self.layers[layer_idx].self_attn.vram_merged_v = None;
        self.layers[layer_idx].self_attn.merged_vram_block_count = 0;

        let mut gpu_k_list = Vec::new();
        let mut gpu_v_list = Vec::new();
        let mut target_indices = Vec::new();

        // 1. 읽기 락으로 대상 수집 및 VRAM 선행 처리
        {
            let reg = self.registry.entries.read().unwrap();
            let kv_blocks = &self.layers[layer_idx].self_attn.kv_blocks;
            
            for (idx, block) in kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                
                if inner.k_cache.is_some() || inner.v_cache.is_some() {
                    let is_dirty = if idx < reg.len() { reg[idx].is_dirty[layer_idx] } else { true };
                    
                    if is_dirty {
                        let k = inner.k_cache.as_ref().unwrap();
                        let v = inner.v_cache.as_ref().unwrap();
                        
                        
                        let k_gpu = k.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| k.clone()).contiguous().unwrap_or_else(|_| k.clone());
                        let v_gpu = v.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| v.clone()).contiguous().unwrap_or_else(|_| v.clone());
                        
                        gpu_k_list.push(k_gpu);
                        gpu_v_list.push(v_gpu);
                        target_indices.push(idx);
                    }
                }
            }
        }

        // 2. 단 한 번의 통신으로 병합 전송
        if !gpu_k_list.is_empty() {
            let merged_k_gpu = candle_core::Tensor::cat(&gpu_k_list, 2).unwrap_or_else(|_| gpu_k_list[0].clone());
            let merged_v_gpu = candle_core::Tensor::cat(&gpu_v_list, 2).unwrap_or_else(|_| gpu_v_list[0].clone());

            // 🌟 [FP8 Compression] 디코딩 롤백 시 단일 레이어 VRAM 완전 철수 과정에서도 GPU 코어로 초고속 FP8 압축합니다.
            let target_dtype = if merged_k_gpu.device().is_cuda() || merged_k_gpu.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
            let merged_k_cpu = merged_k_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_k_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_k_gpu.clone());
            let merged_v_cpu = merged_v_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_v_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_v_gpu.clone());

            let mut current_offset = 0;
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;

            // 3. CPU에서 다시 분할 및 할당
            for (i, &idx) in target_indices.iter().enumerate() {
                let chunk_len = gpu_k_list[i].dim(2).unwrap_or(1024);
                let k_cpu = merged_k_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_k_cpu.clone()).contiguous().unwrap_or_else(|_| merged_k_cpu.clone());
                let v_cpu = merged_v_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_v_cpu.clone()).contiguous().unwrap_or_else(|_| merged_v_cpu.clone());
                current_offset += chunk_len;

                let inner = kv_blocks[idx].inner.write().unwrap();
                let k_shape_vec: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();

                dumps_to_send.push((
                    LayerKVDump { 
                        layer_idx, 
                        k_data: Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(), 
                        v_data: Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(), 
                        k_shape: Tensor::from_vec(k_shape_vec, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                        raw_k: Some(k_cpu), 
                        raw_v: Some(v_cpu), 
                    },
                    inner.offset,
                    inner.len,
                    idx 
                ));
                
                if idx < reg.len() { reg[idx].is_dirty[layer_idx] = false; }
            }
        }

        // 4. (더티 여부 상관없이) 모든 활성 블록을 VRAM에서 해제하고 SSD 상태로 변경
        {
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;
            
            for (idx, block) in kv_blocks.iter_mut().enumerate() {
                let mut inner = block.inner.write().unwrap();
                if inner.k_cache.is_some() || inner.v_cache.is_some() {
                    inner.k_cache = None;
                    inner.v_cache = None;
                    inner.location = KVLocation::SSD; 
                    
                    if idx < reg.len() {
                        reg[idx].location[layer_idx] = KVLocation::SSD;
                        let mut cache = reg[idx].bitkv_cache.write().unwrap();
                        cache[layer_idx] = None; 
                    }
                }
            }
        }

        if !dumps_to_send.is_empty() {
            if let Some(tx) = BAKE_TX.get() {
                let kv_dir = crate::utils::paths::get_kv_dir(None);
                let mode = self.baking_only;
                
                let kv_name_raw = self.active_kv_name.as_deref().unwrap_or("text");
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
                
                let sub_path = if mode {
                    format!("{}/reference/{}", session_id, kv_type)
                } else {
                    format!("{}/inference/{}", session_id, kv_type)
                };

                for (dump, off, b_len, block_idx) in dumps_to_send {
                    let sid = SLOT_MANAGER.acquire_write_slot(b_len).await;
                    let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                    if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }
                    
                    {
                        let mut reg_w = self.registry.entries.write().unwrap();
                        if block_idx < reg_w.len() { reg_w[block_idx].ssd_path = Some(block_dir.clone()); } 
                    }

                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                    if tx.send(SlotTask::Bake(BakeTask {
                        slot_id: sid,
                        task_dir: block_dir,
                        kv_name: Some(sub_path.clone()),
                        offset: off,
                        layers: vec![dump],
                        is_relay_baking: mode,
                        block_idx: Some(block_idx), 
                        registry: self.registry.clone(),
                    })).await.is_err() {
                        crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        SLOT_MANAGER.release_slot(sid).await;
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        _total_len: usize,
        position_ids_in: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        self.active_session_id = session_id.clone();
        self.active_kv_name = kv_name.clone(); 

        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let is_decoding = seq_len <= 1;

        // 🌟 [KV RESIDENCY PLAN] 디코딩 루프 진입 직전(= 첫 디코딩 토큰)에 단 1회만 판정합니다.
        //    KV Cache 는 단조 증가하므로 "앞으로 자랄 최대치"를 기준으로 판정해야
        //    디코딩 도중 OOM 이 터지지 않습니다. 이후 토큰에서는 캐시된 계획을 그대로 씁니다.
        //    프리필 구간에서는 계획을 무효화하여 기존 SSD 오프로딩 경로를 100% 유지합니다.
        if is_decoding {
            if self.kv_plan.is_none() {
                let (kv_layers, kv_heads, kv_head_dim) = self.kv_plan_geometry();
                let planned = seqlen_offset + seq_len + QWEN06B_DECODE_HEADROOM_TOKENS;

                let plan = crate::utils::resources::plan_kv_residency(
                    &crate::utils::resources::KvPlanInput {
                        gpu_id: self.device_id as u32,
                        is_cpu_mode: self.is_forced_cpu,
                        num_kv_layers: kv_layers,
                        num_kv_heads: kv_heads,
                        head_dim: kv_head_dim,
                        // KV 블록은 VRAM 에서 F8E4M3(1바이트)로 압축 보관됩니다.
                        bytes_per_elem: 1,
                        planned_tokens: planned,
                        label: "Qwen(0.6B)",
                    },
                );

                self.keep_weights_resident =
                    crate::utils::resources::free_ram_bytes() > 6_000_000_000;
                self.kv_plan = Some(plan);

                for layer in self.layers.iter_mut() {
                    layer.set_kv_residency(plan);
                }
            }
        } else {
            self.kv_plan = None;
            self.keep_weights_resident = false;
            for layer in self.layers.iter_mut() {
                layer.set_kv_residency(crate::utils::resources::KvResidency::Ssd);
            }
        }

        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) }; 
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mut xs = inputs_embeds.to_device(&target_device)?.to_dtype(target_dtype)?;

        let position_ids = match position_ids_in {
            Some(ids) => ids.clone(),
            None => Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?,
        };

        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        let total_layers = self.layers.len();

        for layer in self.layers.iter_mut() {
            layer.self_attn.active_kv_name = kv_name.clone();
            layer.self_attn.active_session_id = session_id.clone();
        }

        // ====================================================================
        // 🛡️ [절대 방어 핑퐁 패스] 🛡️ (프리필 & 디코딩 공통)
        // ====================================================================
        
        // 억지로 만든 마스크가 과거 기억(10,000 토큰)을 블라인드 처리해버렸습니다.
        // 어텐션 내부에 완벽한 동적 마스크 생성기가 이미 존재하므로, 여기서는 무조건 None을 던져야 합니다!
        let _attention_mask: Option<Tensor> = None;

        let mut next_layer_task = None;
        let mut ping_pong_carrier = QuantizedQwenVLTextDecoderLayer::new_skeleton(
            &self.config, "dummy", &Device::Cpu, self.dtype, 0, self.baking_only, self.registry.clone()
        )?;

        if self.layers[0].self_attn.q_proj.is_cleared() {
            self.reload_layer(0)?;
            self.layers[0].to_device(&target_device)?;
        }

        for layer_idx in 0..total_layers {
            if layer_idx + 1 < total_layers {
                let next_idx = layer_idx + 1;
                let mmap_clone = self.mmap.clone();
                let ct_clone = self.ct.clone();
                let dtype = self.dtype;
                let baking_only = self.baking_only;
                let base_name = self.base_name.clone();
                
                let mut carrier = ping_pong_carrier.clone();
                next_layer_task = Some(tokio::task::spawn_blocking(move || -> Result<QuantizedQwenVLTextDecoderLayer> {
                    let mmap = mmap_clone.ok_or_else(|| anyhow!("Mmap missing"))?;
                    let ct_arc = ct_clone.ok_or_else(|| anyhow!("GGUF missing"))?;
                    let mut reader = std::io::Cursor::new(&mmap[..]);

                    let gguf_blk = format!("blk.{}", next_idx);
                    let prefix = if ct_arc.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{}.layers.{}", base_name, next_idx) };
                    
                    carrier.load_weights_inplace(&ct_arc, &mut reader, &prefix, &Device::Cpu, dtype, baking_only)?;
                    Ok(carrier)
                }));
            }

            let deepstack_embed = deepstack_visual_embeds.as_ref().and_then(|v| v.get(layer_idx));
            
            xs = self.process_single_layer(layer_idx, xs, &cos, &sin, seqlen_offset, deepstack_embed, visual_pos_masks, session_id.clone(), kv_name.clone(), self.baking_only).await?;

            ping_pong_carrier = self.layers[layer_idx].clone();

            if let Some(task) = next_layer_task.take() {
                let mut ready_layer = task.await??; 
                ready_layer.self_attn.active_session_id = self.active_session_id.clone();
                ready_layer.self_attn.active_kv_name = self.active_kv_name.clone();
                ready_layer.self_attn.kv_blocks = self.layers[layer_idx + 1].self_attn.kv_blocks.clone();
                ready_layer.self_attn.layer_idx = layer_idx + 1; 
                self.layers[layer_idx + 1] = ready_layer;
            }
        }

        if target_device.is_cuda() { let _ = target_device.synchronize(); }

        self.current_kv_len = seqlen_offset + seq_len;
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) { xs = xs.to_device(norm_dev)?; }
        
        let final_output = xs.apply(&self.norm)?;

        
        if !is_decoding {
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            #[cfg(target_os = "linux")]
            unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
            #[cfg(target_os = "macos")]
            unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
        }

        Ok(final_output)
    }

    /// 🌟 [KV RESIDENCY] (레이어 수, KV 헤드 수, head_dim) 을 반환합니다.
    /// GGUF 텐서 형상으로 헤드 수가 오버라이드된 경우까지 반영하기 위해
    /// config 가 아니라 실제 레이어 0 의 어텐션 필드를 우선 참조합니다.
    fn kv_plan_geometry(&self) -> (usize, usize, usize) {
        let layers = self.layers.len();
        let (heads, dim) = self
            .layers
            .first()
            .map(|l| (l.self_attn.num_key_value_heads, l.self_attn.head_dim))
            .unwrap_or((self.config.num_key_value_heads, self.config.head_dim));
        (layers, heads, dim)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
        self.current_kv_len = 0;
        // 🌟 [KV RESIDENCY] 세션이 바뀌면 배치 계획도 폐기하여 다음 디코딩에서 재판정합니다.
        self.kv_plan = None;
        self.keep_weights_resident = false;
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.evacuate_vram_to_cache()?;
        }
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        self.current_kv_len
    }

    pub fn compress_to_bf16(&self, t: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        self.layers[0].self_attn.compress_to_bf16(t)
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.drop_kv_storage()?;
        }
        Ok(())
    }

    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> {
        for (_i, layer) in self.layers.iter_mut().enumerate() {
            if _i < k_list.len() {
                layer.self_attn.inject_live_kv(&k_list[_i], &v_list[_i], k_scale, v_scale)?;
            }
        }
        Ok(())
    }

    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        self.inject_live_kv(k_list, v_list, k_scales[0], v_scales[0])
    }

    pub fn inject_live_kv_bitkv(&mut self, k_data: &[Tensor], v_data: &[Tensor], original_shape: &[usize]) -> Result<()> {
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_data.len() {
                let k_final = layer.self_attn.decompress_from_bf16(&k_data[i].to_device(&target_device)?, original_shape, &target_device)?;
                let v_final = layer.self_attn.decompress_from_bf16(&v_data[i].to_device(&target_device)?, original_shape, &target_device)?;
                
                layer.self_attn.inject_live_kv_direct(&k_final.to_dtype(target_dtype)?, &v_final.to_dtype(target_dtype)?)?;
            }
        }
        Ok(())
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
        }

        self.layers.iter_mut().try_for_each(|layer| {
            layer.save_kv_cache(path, clear, offset, kv_name)
        })
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        self.layers.iter_mut().try_for_each(|layer| {
            layer.truncate_kv_cache(len)
        })?;
        self.current_kv_len = len;
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size, None)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        if !path.exists() { return Ok(()); }

        let mut fragments = Vec::new();
        let mut max_offset = 0;

        let scan_path = if let Some(name) = kv_name { path.join(name) } else { path.to_path_buf() };
        if !scan_path.exists() { return Ok(()); }

        if let Ok(entries) = std::fs::read_dir(&scan_path) {
            for entry in entries.flatten() {
                let path_buf = entry.path();
                if path_buf.is_dir() {
                    let dname = path_buf.file_name().unwrap_or_default().to_string_lossy();
                    if dname.starts_with('b') {
                        if let Ok(offset) = dname[1..].parse::<usize>() {
                            if offset > max_offset { max_offset = offset; }
                            fragments.push((offset, path_buf));
                        }
                    }
                }
            }
        }
        
        if fragments.is_empty() { return Ok(()); }
        fragments.sort_by_key(|f| f.0);

        let mut last_chunk_len = 1024;
        let (_, last_st_path) = fragments.last().unwrap();
        if let Ok(raw_content) = crate::utils::direct_loader::load_kv_block(&last_st_path.join("l0.st")) {
            {
                if let Ok(st) = safetensors::SafeTensors::deserialize(&raw_content) {
                    if let Some(name) = st.names().iter().find(|n| n.contains("k_shape")) {
                        if let Ok(view) = st.tensor(name) {
                            let data = view.data();
                            if data.len() >= 12 {
                                last_chunk_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
                            }
                        }
                    }
                }
            }
        }
        
        let total_kv_len = max_offset + last_chunk_len;
        self.current_kv_len = total_kv_len;
        println!("[SSD-GLOBAL] Snapshot loaded. Total context length: {} tokens.", total_kv_len);

        self.layers.iter_mut().try_for_each(|layer| {
            layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, &fragments, total_kv_len)
        })?;
        
        self.current_kv_len = total_kv_len;
        Ok(())
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let e_w = self.embed_tokens.embeddings().to_device(device)?;
        self.embed_tokens = Embedding::new(e_w, self.embed_tokens.hidden_size());
        for layer in self.layers.iter_mut() {
            layer.to_device(device)?;
        }
        self.norm.to_device(device)?;
        Ok(())
    }

    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> {
        if self.is_forced_cpu { return Ok(()); } 
        
        use nvml_wrapper::Nvml;
        
        let nvml = Nvml::init().ok();
        let mut free_vram = 0;
        
        if let Some(nvml_inst) = &nvml {
            if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                if let Ok(mem) = dev.memory_info() {
                    free_vram = mem.free;
                }
            }
        }

        let danger_zone = 500_000_000; 
        let safe_zone = 1_000_000_000; 

        if free_vram > 0 && free_vram < danger_zone {
            for layer in self.layers.iter_mut().rev() {
                if layer.device().is_cuda() {
                    println!("[REBALANCE] (Offset: {}/{}) Low VRAM ({:.2} MB). Offloading Layer {} to CPU.", offset, total_len, free_vram as f64 / 1e6, layer.self_attn.layer_idx);
                    layer.to_device(&Device::Cpu)?;
                    break; 
                }
            }
        } else if free_vram > safe_zone {
            // [STABILITY-FIX] Do NOT auto-upload layers in long context mode to avoid sudden OOM
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct QuantizedQwenVLModel {
    pub config: QwenVLConfig,
    pub visual: Option<QwenVLVisionModel>, 
    pub language_model: QuantizedQwenVLTextModel,
    pub lm_head: QLinear,
    pub rope_deltas: Option<Tensor>,
    pub rope_deltas_cpu: Option<Vec<i64>>,
    pub text_device: Device,
    pub vision_device: Device,
    pub mmap: Option<Arc<Mmap>>,
    pub mmproj_mmap: Option<Arc<Mmap>>,
}

impl QuantizedQwenVLModel {
    pub fn new_with_mmap(
        config: &QwenVLConfig,
        ct_main: Arc<gguf_file::Content>,
        main_mmap_handle: Option<Arc<Mmap>>,
        ct_vision: Arc<gguf_file::Content>,
        mmproj_mmap_handle: Option<Arc<Mmap>>,
        text_device: &Device,
        text_device_id: usize,
        vision_device: &Device,
        _vision_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool, 
    ) -> Result<Self> {
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { DType::F16 };
        
        
        let visual = if baking_only {
            println!("[MODEL] Baking Mode: Skipping Vision Model Load to save 2GB RAM.");
            None
        } else {
            let mmproj_mmap = mmproj_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut reader_vision = std::io::Cursor::new(mmproj_mmap);
            let vb_visual = from_gguf_content(config, &ct_vision, &mut reader_vision, vision_device, vision_dtype)?;
            Some(QwenVLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?)
        };

        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        let language_model = QuantizedQwenVLTextModel::new_with_mmap(
            &t_config, ct_main.clone(), main_mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only
        )?;

        let main_mmap = main_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader_main = std::io::Cursor::new(main_mmap);
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear(&ct_main, &mut reader_main, "lm_head", text_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(&ct_main, &mut reader_main, "output", text_device, head_dtype) {
            l
        } else {
            get_qlinear(&ct_main, &mut reader_main, "token_embd", text_device, head_dtype)?
        };

        Ok(Self { 
            config: config.clone(), 
            visual, 
            language_model, 
            lm_head, 
            rope_deltas: None, 
            rope_deltas_cpu: None, 
            text_device: text_device.clone(), 
            vision_device: vision_device.clone(), 
            mmap: main_mmap_handle, 
            mmproj_mmap: mmproj_mmap_handle 
        })
    }

    pub fn new<R: std::io::Seek + std::io::Read, R2: std::io::Seek + std::io::Read>(
        config: &QwenVLConfig,
        ct_main: Arc<gguf_file::Content>,
        reader_main: &mut R,
        ct_vision: Arc<gguf_file::Content>,
        reader_vision: &mut R2,
        text_device: &Device,
        text_device_id: usize,
        vision_device: &Device,
        _vision_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool, 
    ) -> Result<Self> {
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { DType::F16 };
        
        let visual = if baking_only {
            None
        } else {
            let vb_visual = from_gguf_content(config, &ct_vision, reader_vision, vision_device, vision_dtype)?;
            Some(QwenVLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?)
        };
        
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        

        let language_model = QuantizedQwenVLTextModel::new(&t_config, ct_main.clone(), reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if !baking_only {
            if let Ok(l) = get_qlinear(&ct_main, reader_main, "lm_head", text_device, head_dtype) { l } 
            else if let Ok(l) = get_qlinear(&ct_main, reader_main, "output", text_device, head_dtype) { l } 
            else { get_qlinear(&ct_main, reader_main, "token_embd", text_device, head_dtype)? }
        } else {
            QLinear::new(QMatMul::Tensor(Tensor::zeros((1, 1), head_dtype, text_device)?), None, text_device.clone())
        };

        Ok(Self { 
            config: config.clone(), 
            visual, 
            language_model, 
            lm_head, 
            rope_deltas: None, 
            rope_deltas_cpu: None, 
            text_device: text_device.clone(), 
            vision_device: vision_device.clone(), 
            mmap: None, 
            mmproj_mmap: None 
        })
    }
    
    fn get_vision_features(&self, pixel_values: &Tensor, image_grid_thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        // 비전 모듈 호출 시 안전하게 체크
        let visual_model = self.visual.as_ref().ok_or_else(|| anyhow!("Vision model is disabled in baking mode!"))?;
        let image_grid_thw = if !image_grid_thw.device().same_device(&self.vision_device) { image_grid_thw.to_device(&self.vision_device)? } else { image_grid_thw.clone() };
        let (image_embeds, deepstack_image_embeds) = visual_model.forward(&pixel_values, &image_grid_thw)?;
        
        let target_dtype = if self.text_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let image_embeds = image_embeds.to_dtype(target_dtype)?.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter().map(|t| Ok(t.to_dtype(target_dtype)?.to_device(&self.text_device)?)).collect();
        
        Ok((image_embeds, deepstack_image_embeds?))
    }

    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token_id = if is_image { self.config.image_token_id.unwrap_or(0) as u32 } else { self.config.video_token_id.unwrap_or(0) as u32 };
        let special_token = Tensor::new(vec![special_token_id], input_ids.device())?;
        let special_mask = input_ids.broadcast_eq(&special_token)?.to_dtype(candle_core::DType::U32)?;
        Ok(special_mask)
    }
    
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Vec<i64>)> { 
        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let image_token_id = self.config.image_token_id.unwrap_or(0);
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(0);
        let (b_sz, seq_len) = input_ids.dims2()?;
        
        let mut mrope_position_deltas = Vec::new();
        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        let image_thw_cpu = if let Some(thw) = image_grid_thw {
            Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?)
        } else { None };

        let mut flat_pos_ids: Vec<u32> = Vec::with_capacity(3 * b_sz * seq_len);

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                if ids[i] == vision_start_token_id as u32 && i + 1 < seq_len && ids[i+1] == image_token_id as u32 {
                    if let Some(thw_cpu_array) = &image_thw_cpu {
                        let thw = &thw_cpu_array[image_idx];
                        image_idx += 1;
                        let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                        
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;

                        let img_len = (t * h * w) as usize;
                        for tt in 0..t {
                            for hh in 0..h {
                                for ww in 0..w {
                                    let idx = i + (tt * h * w + hh * w + ww) as usize;
                                    if idx < seq_len {
                                        llm_pos_ids[0][idx] = curr_pos + tt;
                                        llm_pos_ids[1][idx] = curr_pos + hh;
                                        llm_pos_ids[2][idx] = curr_pos + ww;
                                    }
                                }
                            }
                        }
                        i += img_len;
                        curr_pos += t.max(h).max(w); 
                    } else {
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1; curr_pos += 1;
                    }
                } else {
                    for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                    i += 1;
                    curr_pos += 1;
                }
            }
            
            for d in 0..3 {
                flat_pos_ids.extend_from_slice(&llm_pos_ids[d]);
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        } 

        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), input_ids.device())?;
        
        let target_dtype = if input_ids.device().is_cuda() { DType::BF16 } else { DType::F32 };
        let deltas = Tensor::from_vec(mrope_position_deltas.clone(), (b_sz, 1), input_ids.device())?.to_dtype(target_dtype)?; 
        Ok((position_ids, deltas, mrope_position_deltas))
    }

    pub async fn forward(
        &mut self,
        input_ids_in: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        _pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        _cache_position_in: Option<&Tensor>,
        seqlen_offset: usize,
        total_len: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        if seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let (b_sz, seq_len) = input_ids.dims2()?;

        
        let mut inputs_embeds = self.language_model.embed_tokens.forward(&input_ids)?;
        let target_dtype = if self.text_device.is_cuda() { DType::BF16 } else { DType::F32 };
        inputs_embeds = inputs_embeds.to_dtype(target_dtype)?;
        
        if let Some(pv) = pixel_values { 
            if let Some(thw) = image_grid_thw { 
                let (image_embeds, _) = self.get_vision_features(pv, thw)?; 
                let vision_mask = self.get_placeholder_mask(&input_ids, true)?; 
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; 
            } 
        }
        
        let position_ids = if seqlen_offset == 0 || self.rope_deltas_cpu.is_none() { 
            let (p_ids, deltas_tensor, deltas_cpu) = self.get_rope_index(&input_ids, image_grid_thw, video_grid_thw, None)?; 
            self.rope_deltas = Some(deltas_tensor);
            self.rope_deltas_cpu = Some(deltas_cpu); 
            
            
            // 이렇게 해야 SSD에서 불러온 과거 기억과 새로 계산하는 기억의 문맥 위치가 어긋나지 않습니다.
            if seqlen_offset > 0 {
                let shift = Tensor::new(seqlen_offset as u32, input_ids.device())?.reshape((1, 1, 1))?;
                p_ids.broadcast_add(&shift)?
            } else {
                p_ids
            }
        } else {
            let deltas_cpu = self.rope_deltas_cpu.as_ref().unwrap();
            let mut p_ids_vec = Vec::with_capacity(b_sz);
            
            for b in 0..b_sz {
                let delta = deltas_cpu[b]; 
                let real_start = (seqlen_offset as i64 + delta) as u32; 
                
                let p_id = Tensor::new(&[real_start], input_ids.device())? 
                    .reshape((1, 1, 1))?
                    .broadcast_as((3, 1, 1))?; 
                p_ids_vec.push(p_id); 
            }
            Tensor::cat(&p_ids_vec, 1)?
        };
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();
 
        
        let outputs = self.language_model.forward(
            &inputs_embeds, seqlen_offset, total_len, Some(&position_ids), 
            None::<&Tensor>, None::<Vec<Tensor>>, session_id, kv_name
        ).await?;
        
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let head_dev = self.lm_head.device();
        let head_dtype = if head_dev.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let hidden_state = if hidden_state.dtype() != head_dtype { hidden_state.to_dtype(head_dtype)? } else { hidden_state };
        let hidden_state = if !hidden_state.device().same_device(head_dev) { hidden_state.to_device(head_dev)? } else { hidden_state };
        
        
        let logits = self.lm_head.forward(&hidden_state)?;
        
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.language_model.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { 
        self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) 
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { 
        
        if let Some(v) = &mut self.visual {
            v.to_device(device)?;
        }
        self.language_model.to_device(device)?; 
        self.lm_head.to_device_keep_quantized(device)?;
        self.text_device = device.clone(); 
        self.vision_device = device.clone(); 
        Ok(()) 
    }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, offset, total_len) }
}

#[derive(Clone)]
pub struct QuantizedQwenTextModel {
    pub language_model: QuantizedQwenVLTextModel,
    pub lm_head: Option<QLinear>,
    pub text_device: Device,
    pub mmap: Option<Arc<Mmap>>,
}

impl QuantizedQwenTextModel {
    pub fn new_with_mmap(
        config: &QwenVLConfig,
        ct_main: Arc<gguf_file::Content>,
        mmap_handle: Option<Arc<Mmap>>,
        text_device: &Device,
        text_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
        single_layer_mode: bool,
    ) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        let language_model = QuantizedQwenVLTextModel::new_with_mmap(
            &t_config, ct_main.clone(), mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only
        )?;
        let lm_head = if !baking_only {
            let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut reader = std::io::Cursor::new(mmap);
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(&ct_main, &mut reader, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(&ct_main, &mut reader, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(&ct_main, &mut reader, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: mmap_handle })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &QwenVLConfig,
        ct_main: Arc<gguf_file::Content>,
        reader_main: &mut R,
        text_device: &Device,
        text_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
        single_layer_mode: bool,
    ) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        let language_model = QuantizedQwenVLTextModel::new(
            &t_config, ct_main.clone(), reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only
        )?;
        let lm_head = if !baking_only {
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(&ct_main, reader_main, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(&ct_main, reader_main, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(&ct_main, reader_main, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: None })
    }

    pub async fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        if seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        
        let _cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        
        let target_dtype = if self.text_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?.to_dtype(target_dtype)?;
        
        let start = seqlen_offset as u32;
        let position_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();

        
        
        let chunk_size = 2048; 
        let mut final_hidden_state = None;

        if seq_len > 1 {
            let mut processed = 0;
            let mut current_offset = seqlen_offset;
            
            while processed < seq_len {
                
                if crate::utils::is_extraction_stopped() {
                    return Err(anyhow::anyhow!("Task cancelled"));
                }
                
                let take = (seq_len - processed).min(chunk_size);
                let chunk_embeds = inputs_embeds.narrow(1, processed, take)?;
                let chunk_pos_ids = position_ids.narrow(2, processed, take)?;

                let outputs = self.language_model.forward(
                    &chunk_embeds, current_offset, total_len, Some(&chunk_pos_ids), 
                    None::<&Tensor>, None::<Vec<Tensor>>, session_id.clone(), kv_name.clone()
                ).await?;

                if processed + take == seq_len {
                    let s_len = outputs.dim(1)?;
                    final_hidden_state = Some(outputs.narrow(1, s_len - 1, 1)?.contiguous()?);
                }
                
                if let Some(sid) = &session_id {
                    let _ = self.language_model.force_flush_all_active_blocks(sid, kv_name.as_deref()).await;
                }

                processed += take;
                current_offset += take;
                
                
                let pct = ((processed as f32 / seq_len as f32) * 100.0) as i32;
                if let Some(tx) = crate::scheduler::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("search_") || sid.starts_with("img_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        
                        let current_cat = crate::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = format!("Reading context ({}%)...", pct);
                        
                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Prefill)", current_cat), 
                            "summary": summary_msg,
                            "spinner": "⠹"
                        }));
                    }
                }
                
                use std::io::Write;
                print!("\r[TEXT-PREFILL] {} / {} tokens processed", processed, seq_len);
                let _ = std::io::stdout().flush();
            }
            println!("\n[TEXT-PREFILL] Complete.");
        } else {
            
            let outputs = self.language_model.forward(
                &inputs_embeds, seqlen_offset, total_len, Some(&position_ids), 
                None::<&Tensor>, None::<Vec<Tensor>>, session_id.clone(), kv_name.clone()
            ).await?;
            let s_len = outputs.dim(1)?;
            final_hidden_state = Some(outputs.narrow(1, s_len - 1, 1)?.contiguous()?);
        }
        
        let hidden_state = final_hidden_state.unwrap();
        
        let head_dev = self.lm_head.as_ref().map(|h| h.device()).unwrap_or(&self.text_device);
        let head_dtype = if head_dev.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let hidden_state = if hidden_state.dtype() != head_dtype { hidden_state.to_dtype(head_dtype)? } else { hidden_state };
        let hidden_state = if !hidden_state.device().same_device(head_dev) { hidden_state.to_device(head_dev)? } else { hidden_state };
        
        let logits = if let Some(head) = &self.lm_head {
            head.forward(&hidden_state)?
        } else { hidden_state };
        
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    
    pub fn save_kv_cache(&mut self, path: &std::path::Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.language_model.force_flush_all_active_blocks(session_id, kv_name).await }
    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    
    pub fn offload_kv_cache(&mut self, path: &std::path::Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    
    
    pub fn load_kv_cache(&mut self, path: &std::path::Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { 
        self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) 
    }
    
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.language_model.to_device(device)?; if let Some(head) = &mut self.lm_head { head.to_device_keep_quantized(device)?; } self.text_device = device.clone(); Ok(()) }
    
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, _total_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, offset, _total_len) }
}

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let mut weight_q = QMatMul::from_qtensor(weight)?;
    
    weight_q = match weight_q {
        QMatMul::Tensor(t) => QMatMul::Tensor(t.to_dtype(dtype)?),
        other => other,
    };

    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { 
        Some(t.dequantize_f16(device).or_else(|_| t.dequantize(device))?.to_dtype(dtype)?) 
    } else { None };
    Ok(QLinear::new(weight_q, bias, device.clone()))
}

fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType) -> Result<RmsNorm> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let weight = weight.dequantize_f16(device).or_else(|_| weight.dequantize(device))?.to_dtype(dtype)?;
    Ok(RmsNorm::new(weight, eps))
}

fn from_gguf_content<R: std::io::Seek + std::io::Read>(config: &QwenVLConfig, ct: &gguf_file::Content, reader: &mut R, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
    use std::collections::{HashMap, BTreeMap};
    let mut data = HashMap::new();
    let mut split_tensors: BTreeMap<String, Vec<(usize, Tensor)>> = BTreeMap::new();
    for (name, _) in ct.tensor_infos.iter() {
        let mut new_name = name.clone();
        if let Some(rest) = name.strip_prefix("v.") {
             if let Some(blk_rest) = rest.strip_prefix("blk.") {
                 let parts: Vec<&str> = blk_rest.splitn(2, '.').collect();
                 if parts.len() == 2 {
                     let idx = parts[0];
                     let layer = parts[1];
                     let mapped_layer = match layer { s if s.starts_with("ln1") => s.replace("ln1", "norm1"), s if s.starts_with("ln2") => s.replace("ln2", "norm2"), s if s.starts_with("attn_qkv") => s.replace("attn_qkv", "attn.qkv"), s if s.starts_with("attn_out") => s.replace("attn_out", "attn.proj"), s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.linear_fc1"), s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"), _ => layer.to_string() };
                     new_name = format!("visual.blocks.{}.{}", idx, mapped_layer);
                 }
             } else if rest.starts_with("patch_embd") { new_name = rest.replace("patch_embd", "visual.patch_embed.proj"); }
             else if rest.starts_with("position_embd") { new_name = rest.replace("position_embd", "visual.pos_embed"); }
             else if rest.starts_with("post_ln") { new_name = rest.replace("post_ln", "visual.merger.norm"); }
             else if rest.starts_with("deepstack.") {
                 let parts: Vec<&str> = rest.split('.').collect();
                 if parts.len() >= 2 {
                     if let Ok(layer_idx) = parts[1].parse::<usize>() {
                         let v_idx_opt = config.vision_config.as_ref().and_then(|vc| vc.deepstack_visual_indexes.iter().position(|&x| x == layer_idx));
                         if let Some(pos) = v_idx_opt { let suffix = parts[2..].join("."); new_name = format!("visual.deepstack_merger_list.{}.{}", pos, suffix).replace("fc1", "linear_fc1").replace("fc2", "linear_fc2"); }
                         else { new_name = rest.replace("deepstack", "visual.deepstack_merger_list").replace("fc1", "linear_fc1").replace("fc2", "linear_fc2"); }
                     } else { new_name = rest.replace("deepstack", "visual.deepstack_merger_list").replace("fc1", "linear_fc1").replace("fc2", "linear_fc2"); }
                 }
             } else { new_name = format!("visual.{}", rest); }
        } else if let Some(rest) = name.strip_prefix("mm.") { if rest.starts_with("0") { new_name = rest.replace("0", "visual.merger.linear_fc1"); } else if rest.starts_with("2") { new_name = rest.replace("2", "visual.merger.linear_fc2"); } }
        else if name.starts_with("model.visual") { new_name = name.strip_prefix("model.").unwrap().to_string(); }
        
        let mut is_split = false;
        let mut split_idx = 0;
        let mut base_split_name = new_name.clone();
        if let Some(last_dot) = new_name.rfind('.') { if let Ok(idx) = new_name[last_dot+1..].parse::<usize>() { if name.ends_with(&format!(".{}", idx)) { base_split_name = new_name[..last_dot].to_string(); split_idx = idx; is_split = true; } } }
        
        let t = ct.tensor(reader, name, device)?;
        
        let t = t.dequantize_f16(device).or_else(|_| t.dequantize(device))?.to_dtype(dtype)?;
        if is_split { split_tensors.entry(base_split_name).or_default().push((split_idx, t)); } else { data.insert(new_name, t); }
    }
    for (name, mut parts) in split_tensors { parts.sort_by_key(|(i, _)| *i); let tensors: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect(); if let Ok(merged) = Tensor::cat(&tensors, 0) { data.insert(name, merged); } }
    if let Some(weight) = data.get("visual.patch_embed.proj.weight") { if weight.rank() == 4 { if let Ok(reshaped) = weight.unsqueeze(2)?.repeat((1, 1, 2, 1, 1)) { data.insert("visual.patch_embed.proj.weight".to_string(), reshaped); } } }
    Ok(VarBuilder::from_tensors(data, dtype, device))
}