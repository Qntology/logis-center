
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
// use rayon::prelude::*;
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

use crate::{
    models::{
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
        // [FIX] 올바른 최신 rope.rs 경로로 변경!
        qwen3vl::rope::{
            Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
        },
    },
    utils::tensor_utils::{
        mask_index_add, masked_scatter_dim0,
        prod_tensor_last_dim, split_tensor,
    },
};
use crate::models::qwen3vl::generate::SLOT_MANAGER;

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
        self.weight = self.weight.to_device(device)?.to_dtype(target_dtype)?;
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
                    QMatMul::Tensor(t.to_device(device)?.to_dtype(target_dtype)?)
                },
                QMatMul::TensorF16(t) => {
                    QMatMul::TensorF16(t.to_device(device)?.to_dtype(target_dtype)?)
                }
            };

            if let Some(b) = &self.bias {
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
            QMatMul::Tensor(_t) => {
                // 이미 완벽한 타입(BF16)이므로 다이렉트 연산
                self.inner.forward(&xs_flat)?
            },
            _ => {
                let xs_f32 = xs_flat.to_dtype(DType::F32)?;
                self.inner.forward(&xs_f32)?
            }
        };
        
        let out = out.reshape((b, s, ()))?;
        if let Some(bias) = &self.bias {
            // bias도 초기화 시 맞춰두었으므로 캐스팅 없이 바로 덧셈
            Ok(out.broadcast_add(bias)?)
        } else {
            Ok(out)
        }
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
            let entry = RegistryEntry::new(i * 256, 0, 28);
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
                                let idx = start / 256;
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
pub struct QuantizedQwen3VLTextAttention {
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
    // [ACCUMULATOR] VRAM 내 병합 캐시: 매번 수십개의 블록을 cat하는 오버헤드 제거
    pub vram_merged_k: Option<Tensor>,
    pub vram_merged_v: Option<Tensor>,
    pub merged_vram_block_count: usize,
}

impl QuantizedQwen3VLTextAttention {
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
                    inner.k_cache = Some(k.to_device(device)?.to_dtype(target_dtype)?);
                }
                if let Some(v) = &inner.v_cache {
                    inner.v_cache = Some(v.to_device(device)?.to_dtype(target_dtype)?);
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
        
        // VRAM 병합 캐시도 삭제
        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
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
        let mut vram_count = 0;
        let mut ram_count = 0;
        let mut ssd_count = 0;

        // 1. [ALIGNMENT] Input & Rotary
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };
        let (b_sz, q_len, _) = xs.dims3()?;

        // [CRITICAL] Internal Causal Mask Generation for Prefill Integrity
        let total_len = seqlen_offset + q_len;
        let attention_mask = if q_len > 1 && attention_mask_in.is_none() {
            let q_indices = Tensor::arange(0u32, q_len as u32, dev)?.unsqueeze(1)?;
            let k_indices = Tensor::arange(0u32, total_len as u32, dev)?.unsqueeze(0)?;
            let mask = k_indices.broadcast_gt(&(q_indices.broadcast_add(&Tensor::new(seqlen_offset as u32, dev)?)?))?;
            let mask = mask.to_dtype(target_dtype)?.affine(-1e4, 0.0)?; // BF16 안전 범위로 조정
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            attention_mask_in.map(|m| m.clone())
        };

        // [CRITICAL FIX] .contiguous() 삭제로 VRAM 전체 복사 스파이크 원천 차단!
        let mut query_states = self.q_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?; // contiguous() 제거
        
        let mut key_states = self.k_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;     // contiguous() 제거
        
        let value_states = self.v_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?; // contiguous() 제거

        let cos = cos.to_dtype(target_dtype)?;
        let sin = sin.to_dtype(target_dtype)?;
        // [FIX] cos와 sin 앞에 & 기호 추가
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, &cos, &sin, false)?;
        let query_states = query_states.to_dtype(target_dtype)?;
        let key_states = key_states.to_dtype(target_dtype)?;

        // 2. [BLOCK-PIPELINE-ALLOCATION] Append or Create New
        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 256usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    // [CRITICAL FIX] 메모리 복사를 유발하는 contiguous() 제거
                    let k_piece = key_states.narrow(2, chunk_offset, take)?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?;

                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };

                        // [CRITICAL FIX] 병합(cat) 후 메모리 재정렬(contiguous) 오버헤드 완벽 제거!
                        inner.k_cache = Some(Tensor::cat(&[&pk, &k_piece], 2)?);
                        inner.v_cache = Some(Tensor::cat(&[&pv, &v_piece], 2)?);
                        
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
                let take = tokens_to_process.min(256);
                
                // [CRITICAL FIX] 신규 블록을 만들 때도 VRAM 전체 복사(.contiguous())를 제거하여 속도 향상!
                let k_piece = key_states.narrow(2, chunk_offset, take)?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?;
                
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece); inner.v_cache = Some(v_piece);
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
        
        let mut out_res: Option<Tensor> = None;
        let mut m_n: Option<Tensor> = None;
        let mut l_n: Option<Tensor> = None;
        
        let q_aligned = query_states.to_dtype(target_dtype)?;

        for block in &self.kv_blocks {
            let (index, b_off, _b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            // [STEP A] Load Block to VRAM
            let (k_block, v_block, is_temporary) = {
                let inner = block.inner.read().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    (k.to_device(dev)?.to_dtype(target_dtype)?, v.to_device(dev)?.to_dtype(target_dtype)?, false)
                } else {
                    let mut k_cpu = None;
                    let mut v_cpu = None;
                    {
                        let reg = self.registry.entries.read().unwrap();
                        let cache = reg[index].bitkv_cache.read().unwrap();
                        if let Some(m) = &cache[self.layer_idx] {
                            k_cpu = Some(self.decompress_from_bf16(&m.k_data, &m.original_shape, &Device::Cpu)?);
                            v_cpu = Some(self.decompress_from_bf16(&m.v_data, &m.original_shape, &Device::Cpu)?);
                            ram_count += 1;
                        }
                    }

                    if k_cpu.is_none() {
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let sid = self.active_session_id.as_deref().unwrap_or("general");
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
                        
                        let mut load_success = false;
                        for full_path in &path_candidates {
                            let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                            if block_file.exists() {
                                match crate::utils::direct_loader::load_kv_block(&block_file) {
                                    Ok(data) => {
                                        match safetensors::SafeTensors::deserialize(&data) {
                                            Ok(st) => {
                                                let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                                let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                                
                                                if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                                    let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                                    let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                                    
                                                    let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap_or_else(|_| Tensor::zeros(meta_os.clone(), DType::BF16, &Device::Cpu).unwrap());
                                                    let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap_or_else(|_| Tensor::zeros(meta_os.clone(), DType::BF16, &Device::Cpu).unwrap());
                                                    
                                                    let cpu_dev = &Device::Cpu;
                                                    let k_unpacked = self.decompress_from_bf16(&kd_t, &meta_os, cpu_dev).unwrap_or_else(|_| kd_t.clone());
                                                    let v_unpacked = self.decompress_from_bf16(&vd_t, &meta_os, cpu_dev).unwrap_or_else(|_| vd_t.clone());

                                                    k_cpu = Some(k_unpacked.clone());
                                                    v_cpu = Some(v_unpacked.clone());
                                                    ssd_count += 1;

                                                    let mut reg = self.registry.entries.write().unwrap();
                                                    if index < reg.len() { 
                                                        reg[index].ssd_path = Some(full_path.clone());
                                                        reg[index].location[self.layer_idx] = KVLocation::RAM; // 위치를 RAM으로 격상
                                                        
                                                        let mut cache = reg[index].bitkv_cache.write().unwrap();
                                                        // [CRITICAL FIX] CPU 모드에서는 매 토큰 재변환 병목을 막기 위해 원본(kd_t) 대신 이미 풀려있는 F32 텐서(k_unpacked)를 보관합니다!
                                                        let cache_k = if self.q_proj.device().is_cpu() { k_unpacked.clone() } else { kd_t };
                                                        let cache_v = if self.q_proj.device().is_cpu() { v_unpacked.clone() } else { vd_t };
                                                        
                                                        cache[self.layer_idx] = Some(BitKVMetadata {
                                                            k_data: cache_k, 
                                                            v_data: cache_v,
                                                            original_shape: meta_os,
                                                        });
                                                    }
                                                    load_success = true;
                                                    break;
                                                }
                                            },
                                            Err(_) => {},
                                        }
                                    },
                                    Err(_) => {},
                                }
                            }
                        }
                    }
                    let k_final = k_cpu.ok_or_else(|| anyhow!("Block missing"))?.to_device(dev)?.to_dtype(target_dtype)?;
                    let v_final = v_cpu.ok_or_else(|| anyhow!("Block missing"))?.to_device(dev)?.to_dtype(target_dtype)?;
                    (k_final, v_final, true)
                }
            };
            if !is_temporary { vram_count += 1; }

            // [STEP B] Online Softmax for this Chunk
            let mut k = k_block;
            let mut v = v_block;

            // 1) GQA Expansion
            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            let actual_kv_len = k.dim(2)?;

            // 2) Q * K^T * scaling
            let mut s_chunk = (q_aligned.matmul(&k.transpose(2, 3)?)? * self.scaling)?;

            // 3) Apply corresponding Mask slice
            if let Some(mask) = &attention_mask {
                let mask_len = mask.dim(candle_core::D::Minus1)?;
                if b_off < mask_len {
                    let take = std::cmp::min(actual_kv_len, mask_len - b_off);
                    let chunk_mask = mask.narrow(candle_core::D::Minus1, b_off, take)?;
                    
                    // [CRITICAL FIX] 크기가 다를 경우 발생하는 강제 종료(Panic)를 막고 안전하게 병합!
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

            // 4) Update Online Softmax Variables (Using F32 for stability)
            let s_chunk_f32 = s_chunk.to_dtype(DType::F32)?;
            let m_j = s_chunk_f32.max_keepdim(candle_core::D::Minus1)?;
            let p_j = s_chunk_f32.broadcast_sub(&m_j)?.exp()?;
            let l_j = p_j.sum_keepdim(candle_core::D::Minus1)?;
            
            // 행렬곱(MatMul)은 GPU 텐서코어를 극대화하기 위해 BF16으로 수행
            let out_j = p_j.to_dtype(v.dtype())?.matmul(&v)?;
            
            // 누산(Accumulation) 정밀도를 위해 이 작은 결과값만 F32로 승격
            let out_j_f32 = out_j.to_dtype(DType::F32)?;

            match out_res {
                None => {
                    out_res = Some(out_j_f32); // [FIX] 처음부터 F32 상태로 저장!
                    m_n = Some(m_j);
                    l_n = Some(l_j);
                }
                Some(prev_out_f32) => { // [FIX] 이미 F32이므로 무거운 prev_out.to_dtype 삭제!
                    let prev_m = m_n.as_ref().unwrap();
                    let prev_l = l_n.as_ref().unwrap();
                    
                    let m_new = prev_m.maximum(&m_j)?;
                    let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
                    let diff_new = m_j.broadcast_sub(&m_new)?.exp()?;
                    
                    let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
                    let out_new_f32 = prev_out_f32.broadcast_mul(&diff_old)?.add(&out_j_f32.broadcast_mul(&diff_new)?)?;
                    
                    out_res = Some(out_new_f32); // [FIX] 다시 BF16으로 깎아내는 오버헤드 삭제!
                    m_n = Some(m_new);
                    l_n = Some(l_new);
                }
            }

            // [MEMORY-FIX] Drop chunk tensors explicitly to free VRAM immediately
            drop(k);
            drop(v);
        }

        // [STEP C] Finalize Attention Output
        let attn_output = if let (Some(out_f32), Some(l_f32)) = (out_res, l_n) {
            // [CRITICAL FIX] 400번 하던 캐스팅을 루프가 다 끝난 뒤 여기서 딱 1번만 합니다!
            out_f32.broadcast_div(&l_f32)?.to_dtype(target_dtype)?
        } else {
            return Err(anyhow!("No KV data processed"));
        };

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok(attn_output)
    }


    // [REPLACED] Direct BF16 Storage (16-bit precision, no compression)
    pub fn compress_to_bf16(&self, t: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        // Convert to BF16 on CPU
        let t_bf16 = t.to_device(&Device::Cpu)?.to_dtype(DType::BF16)?;
        Ok((t_bf16, original_shape))
    }

    pub fn decompress_from_bf16(&self, data: &Tensor, _original_shape: &[usize], device: &Device) -> Result<Tensor> {
        let t = data.to_device(device)?;
        // [CRITICAL FIX] CPU 환경에서는 어텐션 연산(F32)을 위해 읽어오는 즉시 F32로 승격시킵니다.
        // 이 한 줄로 매 토큰마다 발생하는 디코딩시 BF16 -> F32 변환 오버헤드가 완전히 사라집니다!
        let t = if device.is_cpu() { t.to_dtype(DType::F32)? } else { t };
        Ok(t) 
    }

    pub fn decompress_from_8bit(&self, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = packed.device();
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
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
        // [CLEANUP] 슬롯 자원 명시적 반납
        for block in &self.kv_blocks {
            let slot_id = {
                let reg = self.registry.entries.read().unwrap();
                let inner = block.inner.read().unwrap();
                if inner.index < reg.len() { reg[inner.index].slot_ids[self.layer_idx] } else { None }
            };
            if let Some(id) = slot_id {
                // 비동기 호출을 위해 spawn 사용 (clear_kv_cache는 동기 함수)
                tauri::async_runtime::spawn(async move {
                    SLOT_MANAGER.release_slot(id).await;
                });
            }
        }
        self.kv_blocks.clear();
    }

    // 883라인 근처: trigger_realtime_incremental_bake 함수를 아래 내용으로 완전히 교체
    pub fn trigger_realtime_incremental_bake(&self, session_id: &str, is_last_chunk: bool, baking_only: bool, is_decoding: bool) -> Result<()> {
        use crate::models::qwen3vl::generate::{BakeTask, SlotTask, BAKE_TX, SLOT_MANAGER, LayerKVDump};
        
        // [FIX] SSD 비동기 저장을 위한 타겟만 추출합니다.
        let target_indices: Vec<usize> = self.kv_blocks.iter().enumerate().filter_map(|(i, b)| {
            let inner = b.inner.read().unwrap();
            let is_full = inner.len == 256;
            
            let is_dirty = {
                let reg = self.registry.entries.read().unwrap();
                if i < reg.len() { 
                    if self.layer_idx < reg[i].is_dirty.len() { reg[i].is_dirty[self.layer_idx] } else { true }
                } else { true }
            };

            if (is_full || is_last_chunk) && inner.k_cache.is_some() && is_dirty { Some(i) } else { None }
        }).collect();

        for idx in target_indices {
            let block = self.kv_blocks[idx].clone();
            
            // [CRITICAL FIX] 중복 무한 저장을 막기 위해, 큐에 넣기 직전 즉시 dirty 상태 해제!
            {
                let mut reg = self.registry.entries.write().unwrap();
                if idx < reg.len() && self.layer_idx < reg[idx].is_dirty.len() {
                    reg[idx].is_dirty[self.layer_idx] = false;
                }
            }

            let (k_opt, v_opt, off, b_idx, b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.k_cache.clone(), inner.v_cache.clone(), inner.offset, inner.index, inner.len)
            };

            if let (Some(k), Some(v)) = (k_opt, v_opt) {
                let kv_name_raw = self.active_kv_name.clone().unwrap_or_else(|| "text".to_string());
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text".to_string() } else { last_part.to_string() };
                let session_id_owned = session_id.to_string();
                let registry_clone = self.registry.clone();
                let layer_idx = self.layer_idx;
                let num_kv_h = self.num_key_value_heads;
                let h_d = self.head_dim;
                
                crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                tauri::async_runtime::spawn(async move {
                    // [CRITICAL FIX] 무거운 GPU->CPU 다운로드(블로킹 연산)가 비동기 런타임을 마비시키지 않도록
                    // spawn_blocking으로 완벽하게 격리시켜 디코딩 버벅임(Stuttering)을 차단합니다!
                    let (k_vram, v_vram) = tokio::task::spawn_blocking(move || {
                        let k_res = k.to_device(&Device::Cpu).unwrap_or_else(|_| k.clone()).contiguous().unwrap_or_else(|_| k.clone()).to_dtype(DType::BF16).unwrap_or_else(|_| k.clone());
                        let v_res = v.to_device(&Device::Cpu).unwrap_or_else(|_| v.clone()).contiguous().unwrap_or_else(|_| v.clone()).to_dtype(DType::BF16).unwrap_or_else(|_| v.clone());
                        (k_res, v_res)
                    }).await.unwrap_or_else(|_| (Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(), Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap()));
                    
                    if let Some(tx) = crate::models::qwen3vl::generate::BAKE_TX.get() {
                        let sub_path = if baking_only { format!("{}/reference/{}", session_id_owned, kv_type) } else { format!("{}/inference/{}", session_id_owned, kv_type) };
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                        let k_shape_u32 = vec![1u32, num_kv_h as u32, b_len as u32, h_d as u32];
                        let dump = crate::models::qwen3vl::generate::LayerKVDump {
                            layer_idx,
                            k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            k_shape: Tensor::from_vec(k_shape_u32, (4,), &Device::Cpu).unwrap(),
                            raw_k: Some(k_vram),
                            raw_v: Some(v_vram),
                        };
                        let sid = crate::models::qwen3vl::generate::SLOT_MANAGER.acquire_write_slot(b_len).await;
                        
                        if tx.send(crate::models::qwen3vl::generate::SlotTask::Bake(crate::models::qwen3vl::generate::BakeTask {
                            slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path), offset: off, layers: vec![dump],
                            is_relay_baking: baking_only, block_idx: Some(b_idx), registry: registry_clone,
                        })).await.is_err() {
                            crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            crate::models::qwen3vl::generate::SLOT_MANAGER.release_slot(sid).await;
                        }
                    } else {
                        crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        }
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        // [CRITICAL FIX] 수천 번의 RwLock 획득 오버헤드를 단 1번(O(1))으로 줄여 디코딩 속도 극대화!
        self.kv_blocks.last().map(|b| {
            let inner = b.inner.read().unwrap();
            inner.offset + inner.len
        }).unwrap_or(0)
    }

    pub fn batch_load_layer_kv(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen3vl::generate::LayerIndex;
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
        
        for block_info in index.blocks {
            let block_parent = kv_dir.join(kv_name).join(format!("b{}", block_info.offset));
            let l_file = block_parent.join(format!("l{}.st", self.layer_idx));
            let file_path = if l_file.exists() { l_file } else { block_parent.join("l0.st") };
            
            if !file_path.exists() { continue; }
            
            let b_idx = block_info.offset / 256;
            
            if let Ok(content) = crate::utils::direct_loader::load_kv_block(&file_path) {
                if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                    // [RESTORATION] 프리픽스 매칭 로직 복구
                    let is_l0 = file_path.to_string_lossy().contains("l0.st");
                    let prefix = if is_l0 { format!("b{}_l0_", block_info.offset) } else { format!("b{}_l{}_", block_info.offset, self.layer_idx) };
                    let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();

                    if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                        let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                        
                        let dev = &Device::Cpu;
                        let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, dev)?;
                        let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, dev)?;

                        let mut k_raw = self.decompress_from_bf16(&kd_t, &meta_os, dev)?;
                        let mut v_raw = self.decompress_from_bf16(&vd_t, &meta_os, dev)?;

                        // [KV-BRIDGE] 0.6B -> 0.6B 상황에서도 규격이 다르면 정렬 (커밋 261fe0ef 매커니즘)
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
                                // [CRITICAL FIX] RAM 캐싱 단계이므로 절대 VRAM에 올리지 않고 Device::Cpu로 고정!
                                inner.k_cache = Some(k_raw.to_device(&Device::Cpu)?);
                                inner.v_cache = Some(v_raw.to_device(&Device::Cpu)?);
                                inner.location = KVLocation::RAM;
                                reg[b_idx].location[self.layer_idx] = KVLocation::RAM;
                                reg[b_idx].ssd_path = Some(file_path.parent().unwrap().to_path_buf());
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

    /// [VRAM-EVACUATION] VRAM에 있는 모든 활성 블록을 RAM 캐시로 강제 이동합니다. (SSD 저장 준비 단계)
    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        let dev = &Device::Cpu;
        // [CRITICAL FIX] CPU 모드일 때는 RAM 캐시도 F32로 유지하여 강제 변환 병목을 없앱니다.
        let target_dtype = if self.q_proj.device().is_cpu() { DType::F32 } else { DType::BF16 };

        // 1. 병합된 VRAM 누산기(vram_merged_k/v) 해체
        if let (Some(mk), Some(mv)) = (&self.vram_merged_k, &self.vram_merged_v) {
            let mk_cpu = mk.to_device(dev)?.to_dtype(target_dtype)?;
            let mv_cpu = mv.to_device(dev)?.to_dtype(target_dtype)?;
            
            let mut current_pos = 0;
            for block in &mut self.kv_blocks {
                let mut inner = block.inner.write().unwrap();
                let b_len = inner.len;
                if inner.location == KVLocation::VRAM {
                    let k_part = mk_cpu.narrow(2, current_pos, b_len)?;
                    let v_part = mv_cpu.narrow(2, current_pos, b_len)?;
                    inner.k_cache = Some(k_part);
                    inner.v_cache = Some(v_part);
                    inner.location = KVLocation::RAM;
                    
                    // 장부 업데이트
                    let mut reg = self.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[self.layer_idx] = KVLocation::RAM;
                    }
                }
                current_pos += b_len; // [FIX] 모든 블록의 길이를 더해 오프셋을 동기화함
            }
            // 누산기 초기화
            self.vram_merged_k = None;
            self.vram_merged_v = None;
            self.merged_vram_block_count = 0;
        }

        // 2. 이미 각 블록의 VRAM 캐시에 있는 데이터들도 RAM으로 이동
        for block in &mut self.kv_blocks {
            let (k_to_move, v_to_move) = {
                let inner = block.inner.read().unwrap();
                if inner.location == KVLocation::VRAM && inner.k_cache.is_some() {
                    (inner.k_cache.clone(), inner.v_cache.clone())
                } else { (None, None) }
            };

            if let (Some(k), Some(v)) = (k_to_move, v_to_move) {
                let mut inner = block.inner.write().unwrap();
                inner.k_cache = Some(k.to_device(dev)?.to_dtype(target_dtype)?);
                inner.v_cache = Some(v.to_device(dev)?.to_dtype(target_dtype)?);
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
        // [SSD-ALIGNMENT] 표준화된 경로 구조 생성 (Session/Mode/Type/Block)
        let kv_type = kv_name.unwrap_or("text");
        let b_str = format!("b{}", offset);
        let block_dir = path.join(kv_type).join(&b_str);
        if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
        
        let structured_path = block_dir.join(format!("l{}.st", self.layer_idx));
        
        let mut map = HashMap::new();
        // [PREFIX-FIX] 로드 로직(b{}_l{}_)과 일치하도록 접두어 수정
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
            
            // [COMPILATION FIX] 메모리 추출 대신 프레임워크 자체 직렬화 엔진 사용
            if let Ok(_) = candle_core::safetensors::save(&map, &structured_path) {
                println!("[SSD-SAVE] Layer {} Block {} saved to disk.", self.layer_idx, offset);
            } else {
                println!("[SSD-SAVE-ERROR] Failed to save Layer {} Block {}", self.layer_idx, offset);
            }
            
            if let Ok(mut reg) = self.registry.entries.write() {
                // 이 레이어의 세부 정보를 장부에 기록
                let entry_idx = offset / 256;
                if entry_idx < reg.len() {
                    let entry = &mut reg[entry_idx];
                    entry.ssd_path = Some(path.to_path_buf());
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = false; }
                } else {
                    // 장부가 비어있거나 부족하면 확장
                    let mut entry = crate::models::qwen3vl::quantized_model::RegistryEntry::new(offset, k.dim(2)?, 28);
                    entry.ssd_path = Some(path.to_path_buf());
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
                        // [FIX] Avoid simultaneous immutable and mutable borrow
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
        
        // [CONTEXT-RESET] Clear current session's live blocks before loading snapshot
        self.kv_blocks.clear();
        let mut total_restored_len = 0;

        for (i, (offset, frag_path)) in fragments.iter().enumerate() {
            // [FIX] Calculate block length based on current total context
            let b_len = if *offset < current_kv_len {
                (current_kv_len - *offset).min(256)
            } else { 256 };
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
                reg.push(crate::models::qwen3vl::quantized_model::RegistryEntry {
                    location: vec![KVLocation::SSD; 28],
                    slot_ids: vec![None; 28],
                    token_start: *offset,
                    token_len: b_len,
                    ssd_path: Some(frag_path.parent().unwrap().to_path_buf()),
                    hidden_states_path: vec![None; 28],
                    is_dirty: vec![false; 28], 
                    last_accessed: std::time::Instant::now(),
                    bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                });
            }
            
            if i < reg.len() {
                reg[i].location[self.layer_idx] = KVLocation::SSD;
                reg[i].ssd_path = Some(frag_path.parent().unwrap().to_path_buf());
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
pub struct QuantizedQwen3VLTextDecoderLayer {
    pub self_attn: QuantizedQwen3VLTextAttention,
    pub mlp_gate: Option<QLinear>,
    pub mlp_up: Option<QLinear>,
    pub mlp_down: Option<QLinear>,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: Option<RmsNorm>,
}

impl QuantizedQwen3VLTextDecoderLayer {
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
        config: &Qwen3VLTextConfig,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        baking_only: bool,
        registry: KVRegistry,
    ) -> Result<Self> {
        let is_gguf_naming = base_name.starts_with("blk.");
        let (_attn_base, _gate, _up, _down, _in_ln, _post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        // 빈 텐서들로 초기화
        let zero_t = Tensor::zeros((1,), dtype, device)?;
        let q_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let k_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let v_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let o_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let q_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);
        let k_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);

        let mut self_attn = QuantizedQwen3VLTextAttention {
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
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        };
        self_attn.clear(); // 내부적으로 1-element 상태로 확정

        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !baking_only {
            (
                Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
                Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
                Some(QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone())),
                Some(RmsNorm::new(zero_t.clone(), config.rms_norm_eps))
            )
        } else { (None, None, None, None) };

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
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        baking_only: bool,
        registry: KVRegistry, // [NEW]
    ) -> Result<Self> {
        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx, registry)?;
        
        // [OPTIMIZATION] Skip MLP loading if we only need to bake KV cache (MLP 0% Mode)
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !baking_only {
            let mg = Some(get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?);
            let mu = Some(get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?);
            let md = Some(get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?);
            let pln = Some(get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype)?);
            (mg, mu, md, pln)
        } else {
            (None, None, None, None)
        };

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
        baking_only: bool,
    ) -> Result<()> {
        let is_gguf_naming = base_name.starts_with("blk.");
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        self.self_attn.load_weights_inplace(ct, reader, &attn_base, is_gguf_naming, device, dtype)?;

        if !baking_only {
            self.mlp_gate = Some(get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?);
            self.mlp_up = Some(get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?);
            self.mlp_down = Some(get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?);
            self.post_attention_layernorm = Some(get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), self.input_layernorm.eps(), device, dtype)?);
        }

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

        // 2. Ensure inputs are on this device and dtype
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
        
        // 주의: 아래 forward 호출 시 cos_ref, sin_ref를 넘겨주도록 변경
        let xs = self.self_attn.forward(&xs, cos_ref, sin_ref, attention_mask.as_ref(), seqlen_offset, session_id, kv_name, baking_only)?;
        
        // [HARDENING] Residual Addition DType Guard
        
        let xs = residual.add(&xs)?;
        
        // [OPTIMIZATION] Skip MLP block if not available (MLP 0% Mode)
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
            // [HARDENING] Second Residual Addition DType Guard
            // let xs = if xs.dtype() != residual.dtype() { xs.to_dtype(residual.dtype())? } else { xs };
            Ok(residual.add(&xs)?)
        } else {
            // MLP was skipped (Attention-Only), just return result after attention
            Ok(xs)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
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

#[derive(Clone)]
pub struct QuantizedQwen3VLTextModel {
    pub embed_tokens: Embedding, 
    pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
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
    // [NEW] 재로딩을 위한 메타데이터
    pub config: Qwen3VLTextConfig,
    pub ct: Option<Arc<gguf_file::Content>>,
    pub base_name: String,
    pub dtype: DType,
}

impl QuantizedQwen3VLTextModel {
    /// [MEMORY-OPT] 특정 레이어의 가중치를 mmap에서 다시 로드합니다.
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
        config: &Qwen3VLTextConfig,
        ct: Arc<gguf_file::Content>,
        mmap_handle: Option<Arc<Mmap>>,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        
        // [CRITICAL FIX] CPU 모드일 경우 글로벌 데이터 타입을 뼈대부터 F32로 강제 승격시켜, 
        // 임베딩 테이블과 정규화(Norm), 편향(Bias) 가중치들이 BF16으로 오염되는 것을 원천 차단합니다!
        let effective_dtype = if is_forced_cpu { DType::F32 } else { dtype };
        
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(&mut reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(effective_dtype)?; // dtype -> effective_dtype 적용
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(effective_dtype)?; // dtype -> effective_dtype 적용
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
        let num_layers_to_load = if baking_only { 1 } else { config.num_hidden_layers };

        let mut layers = Vec::with_capacity(num_layers_to_load);
        for layer_idx in 0..num_layers_to_load {
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
            
            // Skeleton 레이어 생성 시에도 effective_dtype 전달
            let layer = QuantizedQwen3VLTextDecoderLayer::new_skeleton(config, &prefix, &current_device, effective_dtype, layer_idx, baking_only, registry.clone())?;
            layers.push(layer);
        }
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let norm = get_rms_norm(&ct, &mut reader, norm_prefix, config.rms_norm_eps, device, effective_dtype)?; // dtype -> effective_dtype 적용
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
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
            config: config.clone(),
            ct: Some(ct),
            base_name: base_name.to_string(),
            dtype: effective_dtype, // [CRITICAL FIX] 모델 전체의 기본 속성을 F32로 확정!
        })
    }

    /// [MEMORY-OPT] 모든 레이어를 한꺼번에 로드합니다. (디코딩 시작 시 호출)
    pub fn reload_all_layers(&mut self) -> Result<()> {
        let count = self.layers.len();
        println!("[MEMORY-OPT] Prefill complete. Reloading all {} layers for high-speed decoding...", count);
        for i in 0..count {
            self.reload_layer(i)?;
        }
        Ok(())
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: Arc<gguf_file::Content>,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let effective_dtype = if is_forced_cpu { DType::F32 } else { dtype }; // [CRITICAL FIX]
        
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(effective_dtype)?; // dtype -> effective_dtype
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(effective_dtype)?; // dtype -> effective_dtype
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
            let mut layer = QuantizedQwen3VLTextDecoderLayer::new(config, &ct, reader, &prefix, device, effective_dtype, layer_idx, baking_only, registry.clone())?;
            layer.clear();
            layers.push(layer);
        }
        
        let norm_prefix = if ct.tensor_infos.contains_key("output_norm.weight") { "output_norm" } else { &format!("{base_name}.norm") };
        let norm = get_rms_norm(&ct, reader, norm_prefix, config.rms_norm_eps, device, effective_dtype)?; // dtype -> effective_dtype
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
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
            config: config.clone(),
            ct: Some(ct),
            base_name: base_name.to_string(),
            dtype: effective_dtype, // [CRITICAL FIX]
        })
    }

    pub fn load_kv_cache_chunked(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen3vl::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let index_path = kv_dir.join(kv_name).join("layer0.json");
        
        if !index_path.exists() { return Ok(()); }
        
        // [FIX-REGISTRY-SIZE] 로딩 전 인덱스를 확인하여 레지스트리 공간 선제 확보
        // [DIRECT-IO] Use OS-accelerated read for index
        let index_json = if let Ok(data) = crate::utils::direct_loader::load_kv_block(&index_path) {
            String::from_utf8(data).unwrap_or_default()
        } else { return Ok(()); };
        
        let index: LayerIndex = serde_json::from_str(&index_json)?;
        let total_tokens = index.total_tokens;
        
        // 레지스트리 및 개별 레이어의 kv_blocks 동기화
        {
            let mut reg = self.registry.entries.write().unwrap();
            let needed_blocks = (total_tokens + 255) / 256;
            while reg.len() < needed_blocks {
                let off = reg.len() * 256;
                // RegistryEntry::new 생성자를 사용하여 모든 필드를 정확하게 초기화
                reg.push(RegistryEntry::new(off, 0, 28));
            }
            // 전체 길이 업데이트
            self.current_kv_len = total_tokens;
        }

        // 개별 레이어의 kv_blocks 도 확보된 레지스트리 길이에 맞춤
        for layer in self.layers.iter_mut() {
            let reg_len = self.registry.entries.read().unwrap().len();
            while layer.self_attn.kv_blocks.len() < reg_len {
                let idx = layer.self_attn.kv_blocks.len();
                let off = idx * 256;
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
        
        // [FIX-BLOCK-LENGTHS] 인덱스에서 읽어온 정보를 바탕으로 각 블록의 실제 길이를 확정
        {
            let mut reg = self.registry.entries.write().unwrap();
            let total_t = self.current_kv_len;
            for (idx, entry) in reg.iter_mut().enumerate() {
                let off = idx * 256;
                let b_len = if off + 256 <= total_t { 256 } else { total_t.saturating_sub(off) };
                entry.token_len = b_len;
                
                // 개별 레이어의 KVBlockInner 길이도 동기화
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

    /// [CHUNK-ASYNC-PROCESSOR]
    /// 청크 단위로 연산 -> CPU 대피 -> VRAM 소각을 실시간으로 반복합니다.
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
        // let mut final_output: Option<Tensor> = None; <-- [삭제]
        let chunk_size = 256;
        let current_seq_len = xs.dim(1)?;
        let is_decoding = current_seq_len <= 1;
        let target_device = self.layers[layer_idx].device().clone();
        let total_kv_blocks = self.layers[layer_idx].self_attn.kv_blocks.len();
        let total_chunks = chunk_offsets.len();

        // [CRITICAL FIX] 텐서를 매번 합치지 않고 벡터에 모아둘 그릇 준비
        let mut chunk_outputs = Vec::with_capacity(total_chunks);

        for (chunk_idx, &i) in chunk_offsets.iter().enumerate() {
            let take = (current_seq_len - i).min(chunk_size);

            // [CRITICAL FIX] 과거 문맥이 길고 새로운 프롬프트가 짧은 'Incremental Prefill' 상황에서 
            // 프리패처가 루프를 돌지 않아 발생하는 치명적인 SSD 동기화 렉(Sync Lag)을 원천 차단합니다!
            let target_chunks = if is_decoding || chunk_idx == 0 {
                // 디코딩 중이거나 프리필의 '첫 번째 청크'일 때, 다음 레이어 연산에 필요한 
                // '모든 과거 블록'을 한 번에 비동기 로딩 큐에 밀어 넣습니다.
                (0..total_kv_blocks).collect::<Vec<_>>()
            } else {
                // 이미 첫 청크에서 모든 로딩 지시를 내렸으므로 추가 지시는 생략합니다.
                vec![]
            };

            let look_ahead_layers = 1;

            for t_idx in target_chunks {
                // [CRITICAL FIX] chunk_offsets.len() 대신 실제 total_kv_blocks 수로 검사!
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
                                    let is_cpu_mode = self.is_forced_cpu; // [CRITICAL FIX] CPU 모드 플래그
                                    tauri::async_runtime::spawn(async move {
                                        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_load_worker};
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

            let xs_chunk = xs.narrow(1, i, take)?;
            let cos_chunk = cos.narrow(cos.rank().saturating_sub(2), i, take)?;
            let sin_chunk = sin.narrow(sin.rank().saturating_sub(2), i, take)?;
            
            let out = self.layers[layer_idx].forward(&xs_chunk, &cos_chunk, &sin_chunk, None, seqlen_offset + i, session_id.clone(), kv_name.clone(), baking_only)?;
            
            chunk_outputs.push(out);
            
            // =========================================================================
            // [CRITICAL FIX] GPU와 CPU의 아키텍처 분리 (기존 GPU 고속 동작 유지)
            // =========================================================================
            if self.is_forced_cpu {
                // [CPU 모드 전용 아키텍처] 
                // CPU는 10K 토큰 연산이 끝날 때까지 기다리면 RAM 스왑 병목으로 기절합니다.
                // 따라서 1개 청크(256토큰) 연산이 끝날 때마다 가득 찬 블록을 즉시 SSD에 구워버립니다.
                // 이 분기문 덕분에 연산 도중에도 tmp/kv/ 폴더에 파일이 실시간으로 쏟아져 나오게 됩니다!
                if let Some(sid) = &session_id {
                    let is_last = chunk_idx == total_chunks - 1;
                    let _ = self.layers[layer_idx].self_attn.trigger_realtime_incremental_bake(sid, is_last, baking_only, true);
                }
                // 배경에서 SSD 저장이 이루어질 수 있도록 메인 런타임 강제 호흡(양보)
                tokio::task::yield_now().await;
            } else {
                // [GPU 모드 전용 아키텍처] 기존 고속 파이프라인 유지
                let _ = self.evacuate_vram_to_ram_only(layer_idx).await;
            }
        } // <-- 청크 루프 종료

        // [IMPORTANT] 잔여 블록 SSD 백업 트리거 (GPU 전용 마무리)
        if let Some(sid) = &session_id {
            let is_prefill = current_seq_len > 1;
            
            // CPU 모드는 이미 루프 내부에서 실시간으로 모두 저장했으므로 건너뜁니다.
            // GPU 모드는 디코딩(1토큰)일 때만 여기서 백업하고, 프리필 중에는 속도를 위해 건너뜁니다. (기존 로직 100% 동일)
            if !self.is_forced_cpu && !is_prefill {
                let _ = self.layers[layer_idx].self_attn.trigger_realtime_incremental_bake(sid, true, baking_only, true);
            }
        }

        // [MEMORY-FIX] 프리필 후 메모리 정리 로직 (기존 2014라인 근처를 이 코드로 대체)
        if current_seq_len > 1 {
            let is_small_model = self.layers.len() <= 36;
            let current_kv_len = self.layers[layer_idx].get_kv_len();
            
            if !is_small_model || current_kv_len >= 4096 {
                let target_dtype = if self.is_forced_cpu { DType::F32 } else { DType::BF16 }; // [CRITICAL FIX]
                
                for (i, block) in self.layers[layer_idx].self_attn.kv_blocks.iter().enumerate() {
                    let (k_part, v_part) = {
                        let inner = block.inner.read().unwrap();
                        (inner.k_cache.clone(), inner.v_cache.clone())
                    };

                    if let (Some(k), Some(v)) = (k_part, v_part) {
                        let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        
                        let mut inner = block.inner.write().unwrap();
                        inner.k_cache = None;
                        inner.v_cache = None;
                        inner.location = KVLocation::RAM;

                        let mut reg = self.registry.entries.write().unwrap();
                        if i < reg.len() {
                            let entry = &mut reg[i];
                            entry.location[layer_idx] = KVLocation::RAM;
                            let mut cache = entry.bitkv_cache.write().unwrap();
                            if layer_idx < cache.len() {
                                cache[layer_idx] = Some(BitKVMetadata {
                                    k_data: k_cpu,
                                    v_data: v_cpu,
                                    original_shape: k.shape().dims().to_vec(),
                                });
                            }
                        }
                    }
                }
                println!("[ENGINE-TRACE] Layer {} Prefill Cache Moved to RAM ({} tokens).", layer_idx, current_kv_len);
            }
        }
        
        // [CRITICAL FIX] 이전의 final_output 관련 잔재를 완전히 삭제하고,
        // chunk_outputs 벡터가 비어있는지만 체크한 뒤 한 방에 병합(cat)하여 반환합니다!
        if chunk_outputs.is_empty() {
            return Err(anyhow::anyhow!("No output generated from chunks"));
        }

        // [CRITICAL FIX] 디코딩 시 1개의 청크만 있을 때 발생하는 100% VRAM 딥카피 대참사 원천 차단!
        let final_output_tensor = if chunk_outputs.len() == 1 {
            chunk_outputs.into_iter().next().unwrap()
        } else {
            Tensor::cat(&chunk_outputs, 1)?
        };
        
        Ok(final_output_tensor)
    }

    /// [VRAM-EVACUATION] 레이어 연산 직후 VRAM 압박 시 RAM으로 이동
    async fn evacuate_vram_to_ram_only(&mut self, layer_idx: usize) -> Result<()> {
        // [OPTIMIZATION] 0.6B (Small) 모델은 기본적으로 모든 문맥을 VRAM에 상주시키나,
        // 문맥이 너무 길어질 경우(예: 4096+) OOM 방지를 위해 RAM으로 대피를 허용합니다.
        let current_kv_len = self.layers[layer_idx].get_kv_len();
        let is_small_model = self.layers.len() <= 36;
        if is_small_model && current_kv_len < 4096 { return Ok(()); }

        let vram_limit = 1024; 
        let mut vram_evicted = false;

        if is_small_model {
            // println!("[ENGINE-TRACE] Small Model context large ({}). Enabling VRAM evacuation for safety.", current_kv_len);
        }

        // [DYNAMIC-LIMITS] 시스템 자원 상황에 따라 임계값 유동적 조절 (OOM 방지)
        let vram_limit = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

            // [FIX] VRAM 한도를 대폭 상향하여 불필요한 RAM 대피 방지 (기존 64 -> 1024)
            if free_ram_gb > 4.0 { 1024 } else if free_ram_gb > 2.0 { 512 } else { 128 }
        };
        let mut vram_evicted = false;

        // 1. VRAM -> RAM 계층 관리
        {
            // 전역 장부 잠금 (Registry -> Inner 순서 준수)
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;

            let mut vram_indices = Vec::new();
            for (idx, block) in kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                // [FIX] 모든 블록을 대피 대상으로 고려 (VRAM 절약 및 Tail 유실 방지)
                if inner.location == KVLocation::VRAM {
                    vram_indices.push((idx, inner.offset));
                }
            }

            // [FIX] 베이킹 모드에서의 강제 대피를 제거하여 VRAM Pinning을 유지합니다.
            if vram_indices.len() > vram_limit {
                vram_indices.sort_by_key(|k| k.1); // 오래된 순 정렬
                let num_to_evict = vram_indices.len().saturating_sub(vram_limit);
                let target_dtype = if self.is_forced_cpu { DType::F32 } else { DType::BF16 }; // [CRITICAL FIX]

                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let mut inner = kv_blocks[idx].inner.write().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        inner.k_cache = Some(k_cpu);
                        inner.v_cache = Some(v_cpu);
                        inner.location = KVLocation::RAM;
                        // 장부 업데이트
                        if inner.index < reg.len() {
                            reg[inner.index].location[layer_idx] = KVLocation::RAM;
                        }
                        vram_evicted = true;
                    }
                }
            }
        } // Registry lock dropped

        // [ACCUMULATOR-INVALIDATE] VRAM에서 방출이 일어났다면 병합 캐시 리셋
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

    /// [LAYER-UNIT-OF-WORK]
    /// 한 레이어의 모든 작업(파일 읽기, 디코딩, GPU 연산, 메모리 해제)을 완결하고 결과 xs를 반환합니다.
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
        let start_layer_time = std::time::Instant::now();
        let input_token_count = xs.dim(1).unwrap_or(0);
        let is_decoding = input_token_count <= 1;

        if is_decoding && layer_idx == 0 {
            let already_loaded = !self.layers[0].self_attn.q_proj.is_cleared();
            if !already_loaded { self.reload_all_layers()?; }
        }

        if !is_decoding {
            self.reload_layer(layer_idx)?;
        }

        // [STEP 1] 현재 레이어 가중치를 연산 장치로 세팅 (GPU는 VRAM으로, CPU는 RAM에서 압축 해제)
        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) }; 
        
        // [CRITICAL FIX] GPU는 디바이스가 달라 자동 호출되지만, CPU는 디바이스가 같아도 
        // 4bit 압축을 고속 F32로 풀기 위해 to_device를 무조건 호출하도록 분기 처리합니다!
        if self.is_forced_cpu || !self.layers[layer_idx].device().same_device(&target_device) {
            self.layers[layer_idx].to_device(&target_device)?;
        }

        // [FIRE & SHOT] 다음 레이어 로드 (VRAM에 2개만 올림)
        // if !is_decoding && layer_idx + 1 < self.layers.len() {
        //     self.reload_layer(layer_idx + 1)?;
        //     let _ = self.layers[layer_idx + 1].to_device(&target_device);
        // }

        // [STEP 3] GPU 연산 실행
        let current_seq_len = xs.dim(1)?;
        let chunk_offsets: Vec<usize> = (0..current_seq_len).step_by(256).collect();
        let mut next_xs = self.process_chunks_iterative(layer_idx, &chunk_offsets, &xs, cos, sin, seqlen_offset, session_id.clone(), kv_name.clone(), baking_only).await?;

        if let (Some(embed), Some(mask)) = (deepstack_embed, visual_mask) {
            next_xs = mask_index_add(&next_xs.squeeze(0)?, &mask.squeeze(0)?, embed)?.unsqueeze(0)?;
        }

        // [MEMORY-RELEASE] 현재 레이어 연산 종료 즉시 가중치 소각
        // CPU RAM과 GPU VRAM에서 현재 레이어의 가중치를 완전히 제거합니다.
        if !is_decoding {
            self.layers[layer_idx].clear(); 
            if target_device.is_cuda() { let _ = target_device.synchronize(); }
            println!("[MEMORY-OPT] Layer {} Weights and Gradients cleared from RAM/VRAM.", layer_idx);
        }

        // [STEP 4] 자원 반납 및 KV 대피
        let sid_opt = self.active_session_id.clone();
        if let Some(sid) = sid_opt {
            if !self.baking_only {
                let _ = self.evacuate_layer_kv_to_cpu(layer_idx, &sid, seqlen_offset, input_token_count).await;
            }
        }

        if is_decoding {
            self.layers[layer_idx].clear();
        }

        Ok(next_xs)
    }

    /// [DEC-SPEED-UP] 디코딩 속도를 위해 모든 레이어를 GPU에 상주 시킴
    pub async fn pin_all_layers_to_gpu(&mut self) -> Result<()> {
        println!("[DEC-SPEED-UP] Pinning disabled for long-context stability. Using On-Demand serial loading.");
        Ok(())
    }

    /// [DEC-CLEANUP] 디코딩이 끝나면 모든 레이어를 다시 CPU로 보냄
    pub async fn unpin_all_layers(&mut self) -> Result<()> {
        println!("[DEC-CLEANUP] Unpinning all layers from GPU...");
        for layer in self.layers.iter_mut() {
            layer.to_device(&Device::Cpu)?;
        }
        Ok(())
    }

    /// [MANUAL-FLUSH] 세션 종료 시 RAM에 남아있는 활성 블록(Active Block)을 강제로 SSD에 저장합니다.
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let mut block_groups: std::collections::HashMap<usize, Vec<LayerKVDump>> = std::collections::HashMap::new();
        
        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            for block in &mut layer.self_attn.kv_blocks {
                let inner = block.inner.write().unwrap();
                let is_dirty = {
                    let reg = self.registry.entries.read().unwrap();
                    if inner.index < reg.len() { 
                        if l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] } else { true }
                    } else { true }
                };

                if inner.k_cache.is_some() && is_dirty {
                    let k = inner.k_cache.as_ref().unwrap();
                    let k_shape_u32: Vec<u32> = k.shape().dims().iter().map(|&x| x as u32).collect();
                    
                    block_groups.entry(inner.offset).or_default().push(LayerKVDump {
                        layer_idx: l_idx,
                        k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                        v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                        k_shape: Tensor::from_vec(k_shape_u32, (k.shape().dims().len(),), &Device::Cpu)?,
                        // [SAFE-FLUSH] contiguous 실패 시 복제본으로 대체하여 에러 탈출 방지 
                        raw_k: Some(k.to_device(&Device::Cpu).unwrap_or(k.clone()).contiguous().unwrap_or_else(|_| k.clone())), 
                        raw_v: Some(inner.v_cache.as_ref().unwrap().to_device(&Device::Cpu).unwrap_or(inner.v_cache.as_ref().unwrap().clone()).contiguous().unwrap_or_else(|_| inner.v_cache.as_ref().unwrap().clone())),
                    });
                    
                    let mut reg = self.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        if l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] = false; }
                    }
                }
            }
        }

        if block_groups.is_empty() { return Ok(()); }

        if let Some(tx) = BAKE_TX.get() {
            let kv_dir = crate::utils::paths::get_kv_dir(None);
            let mode = self.baking_only;
            let kv_name_raw = kv_name.unwrap_or("text");
            let last_part = kv_name_raw.split('/').last().unwrap_or("text");
            let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text".to_string() } else { last_part.to_string() };
            
            let sub_path = if mode { format!("{}/reference/{}", session_id, kv_type) } else { format!("{}/inference/{}", session_id, kv_type) };

            for (off, layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(256).await;
                let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }

                crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path.clone()), offset: off, layers,
                    is_relay_baking: mode, block_idx: Some(off / 256), registry: self.registry.clone(),
                })).await.is_err() {
                    crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    SLOT_MANAGER.release_slot(sid).await;
                }
            }
        }
        Ok(())
    }

    /// [NEW-STRICT] 계층형 메모리 관리 (VRAM -> RAM -> SSD)
    async fn evacuate_layer_kv_to_cpu(&mut self, layer_idx: usize, session_id: &str, start_off: usize, _len: usize) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let block_start = (start_off / 256) * 256;

        // [DYNAMIC-LIMITS] 0.6B 모델은 가벼우므로 10k 문맥 전체를 VRAM에 상주시켜 PCIe 병목을 제거합니다.
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        // [CRITICAL FIX] 기존의 잘못된 섀도잉(덮어쓰기) 변수들을 삭제하여 VRAM 상주 용량을 2048로 정상 적용합니다!
        let (vram_limit, ram_limit) = {            
            // VRAM: 28개 레이어 x 40개 블록 = 1,120개. 2,048개까지 VRAM 상주 허용 (속도 극대화)
            let v_limit = 2048;
            // RAM: 만약 VRAM에서 쫓겨나더라도 RAM에 5,000개까지는 안전하게 보관
            let r_limit = if free_ram_gb > 4.0 { 5000 } else { 2048 };
            (v_limit, r_limit)
        };
        // [삭제] let v_limit = if free_ram_gb > 4.0 { 32 } else { 16 };
        // [삭제] let r_limit = if free_ram_gb > 4.0 { 1024 } else { 512 };

        let mut vram_evicted = false;

        // 1. VRAM -> RAM 계층 관리
        {
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;

            let mut vram_indices = Vec::new();
            for (idx, block) in kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                if inner.offset >= block_start && inner.location == KVLocation::VRAM && inner.len == 256 {
                    vram_indices.push((idx, inner.offset));
                }
            }

            if vram_indices.len() > vram_limit {
                vram_indices.sort_by_key(|k| if k.1 == 0 { usize::MAX } else { k.1 });
                let num_to_evict = vram_indices.len() - vram_limit;
                let target_dtype = if self.is_forced_cpu { DType::F32 } else { DType::BF16 }; // [CRITICAL FIX]

                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let mut inner = kv_blocks[idx].inner.write().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(target_dtype)?;
                        inner.k_cache = Some(k_cpu);
                        inner.v_cache = Some(v_cpu);
                        inner.location = KVLocation::RAM;
                        if inner.index < reg.len() { reg[inner.index].location[layer_idx] = KVLocation::RAM; }
                        vram_evicted = true;
                    }
                }
            }
        }

        // [ACCUMULATOR-INVALIDATE]
        if vram_evicted {
            self.layers[layer_idx].self_attn.vram_merged_k = None;
            self.layers[layer_idx].self_attn.vram_merged_v = None;
            self.layers[layer_idx].self_attn.merged_vram_block_count = 0;
        }

        // 2. RAM -> SSD 계층 관리
        let mut dumps_to_send = Vec::new();
        {
            let mut reg = self.registry.entries.write().unwrap();
            let kv_blocks = &mut self.layers[layer_idx].self_attn.kv_blocks;
            
            let mut ram_indices = Vec::new();
            for (idx, block) in kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                if inner.location == KVLocation::RAM && inner.len == 256 {
                    ram_indices.push((idx, inner.offset));
                }
            }

            if ram_indices.len() > ram_limit {
                ram_indices.sort_by_key(|k| if k.1 == 0 { usize::MAX } else { k.1 });
                let num_to_flush = ram_indices.len() - ram_limit;
                for i in 0..num_to_flush {
                    let (idx, _) = ram_indices[i];
                    let mut inner = kv_blocks[idx].inner.write().unwrap();
                    
                    let is_safe = inner.index < reg.len() && reg[inner.index].location[layer_idx] == KVLocation::SSD;
                    if is_safe {
                        inner.k_cache = None;
                        inner.v_cache = None;
                        inner.location = KVLocation::SSD;
                    } else if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        // [MODIFIED] BF16 Transfer (No bitpacking)
                        let k_shape_vec: Vec<u32> = k.shape().dims().iter().map(|&x| x as u32).collect();

                        dumps_to_send.push((
                            LayerKVDump { 
                                layer_idx, 
                                k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?, // [FIX] 더미 통과
                                v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?, // [FIX] 더미 통과
                                k_shape: Tensor::from_vec(k_shape_vec, (k.shape().dims().len(),), &Device::Cpu)?,
                                raw_k: Some(k.to_device(&Device::Cpu)?), // [CRITICAL FIX] F32 원본 전달!
                                raw_v: Some(v.to_device(&Device::Cpu)?), // [CRITICAL FIX] F32 원본 전달!
                            },
                            inner.offset,
                            inner.len
                        ));
                        inner.k_cache = None;
                        inner.v_cache = None;
                        inner.location = KVLocation::SSD;
                        if inner.index < reg.len() { 
                            reg[inner.index].location[layer_idx] = KVLocation::SSD; 
                            // [CRITICAL FIX] 좀비 RAM 캐시 완벽 제거
                            let mut cache = reg[inner.index].bitkv_cache.write().unwrap();
                            cache[layer_idx] = None; 
                        }
                    }
                }
            }
        }

        // 3. SSD 저장 위임 (Safe await)
        if !dumps_to_send.is_empty() {
            if let Some(tx) = BAKE_TX.get() {
                let kv_dir = crate::utils::paths::get_kv_dir(None);
                let mode = self.baking_only;
                let kv_name_base = self.active_kv_name.as_deref().unwrap_or("general");
                
                let sub_path = if mode {
                    format!("{}/reference/{}", session_id, kv_name_base)
                } else {
                    format!("{}/inference/{}", session_id, kv_name_base)
                };

                for (dump, off, b_len) in dumps_to_send {
                    let sid = SLOT_MANAGER.acquire_write_slot(b_len).await;
                    let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                    if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
                    
                    {
                        let mut reg_w = self.registry.entries.write().unwrap();
                        let b_idx = off / 256;
                        if b_idx < reg_w.len() { reg_w[b_idx].ssd_path = Some(block_dir.clone()); }
                    }

                    let _ = tx.send(SlotTask::Bake(BakeTask {
                        slot_id: sid,
                        task_dir: block_dir,
                        kv_name: Some(sub_path.clone()),
                        offset: off,
                        layers: vec![dump],
                        is_relay_baking: mode,
                        block_idx: Some(off / 256),
                        registry: self.registry.clone(),
                    })).await;
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
        }

        let mut next_layer_task: Option<tokio::task::JoinHandle<Result<QuantizedQwen3VLTextDecoderLayer>>> = None;
        
        // [PING-PONG CARRIER] 서로 번갈아가며(Ping-Pong) 사용할 단 하나의 여분 껍데기 버퍼입니다.
        let mut ping_pong_carrier = QuantizedQwen3VLTextDecoderLayer::new_skeleton(
            &self.config, "dummy", &Device::Cpu, self.dtype, 0, self.baking_only, self.registry.clone()
        )?;

        // 시작점: 0번 레이어는 첫 루프 진입 전 미리 띄워둡니다.
        if !is_decoding && self.layers[0].self_attn.q_proj.is_cleared() {
            self.reload_layer(0)?;
            self.layers[0].to_device(&target_device)?;
        } else if is_decoding {
            let already_loaded = !self.layers[0].self_attn.q_proj.is_cleared();
            if !already_loaded { self.reload_all_layers()?; }
        }

        for layer_idx in 0..total_layers {
            if layer_idx % 7 == 0 || layer_idx == total_layers - 1 {
                println!("[ENGINE] Running Layer {}/{} (Ping-Pong Pipeline Active)", layer_idx + 1, total_layers);
            }

            // =========================================================================
            // [TRACK 2] 백그라운드 지연 적재 (N+1번째 레이어 미리 로드)
            // =========================================================================
            if layer_idx + 1 < total_layers {
                let next_idx = layer_idx + 1;
                let mmap_clone = self.mmap.clone();
                let ct_clone = self.ct.clone();
                let dtype = self.dtype;
                let baking_only = self.baking_only;
                let base_name = self.base_name.clone();
                
                // [FIX] dev_clone 제거 (더 이상 백그라운드에서 VRAM을 건드리지 않습니다)

                // 이번 턴에 데이터를 채워넣을 빈 껍데기(Carrier)를 백그라운드 스레드로 넘깁니다.
                let mut carrier = ping_pong_carrier.clone();
                next_layer_task = Some(tokio::task::spawn_blocking(move || -> Result<QuantizedQwen3VLTextDecoderLayer> {
                    let mmap = mmap_clone.ok_or_else(|| anyhow!("Mmap missing"))?;
                    let ct_arc = ct_clone.ok_or_else(|| anyhow!("GGUF missing"))?;
                    let mut reader = std::io::Cursor::new(&mmap[..]);

                    let gguf_blk = format!("blk.{}", next_idx);
                    let prefix = if ct_arc.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{}.layers.{}", base_name, next_idx) };
                    
                    // [PING-PONG TRACK 2 최적화]
                    // SSD -> CPU RAM 로드만 수행합니다. (Quantized 텐서 상태 유지)
                    // VRAM 할당이나 Dequantization은 메인 스레드로 넘겨 VRAM 스파이크를 원천 차단합니다.
                    carrier.load_weights_inplace(&ct_arc, &mut reader, &prefix, &Device::Cpu, dtype, baking_only)?;
                    
                    // [CRITICAL FIX] carrier.to_device(&dev_clone)?; 삭제!!!
                    // 이전에는 여기서 VRAM으로 넘겨버려서 VRAM 2배 점유와 CUDA 병목이 발생했습니다.
                    
                    Ok(carrier)
                }));
            }

            // =========================================================================
            // [TRACK 1] 메인 연산 트랙 (N번째 레이어) - 청크 처리
            // =========================================================================
            let deepstack_embed = deepstack_visual_embeds.as_ref().and_then(|v| v.get(layer_idx));
            xs = self.process_single_layer(layer_idx, xs, &cos, &sin, seqlen_offset, deepstack_embed, visual_pos_masks, session_id.clone(), kv_name.clone(), self.baking_only).await?;

            // Track 1 연산이 끝났으므로 N번째 레이어의 가중치는 즉시 소각됩니다. 
            // (process_single_layer 내에서 clear()가 호출됨)
            if !is_decoding {
                // [PING-PONG SWAP] 방금 다 쓰고 비워진 껍데기(현재 레이어)를 복사하여 다음 턴의 빈 그릇으로 재활용합니다!
                ping_pong_carrier = self.layers[layer_idx].clone();
            }

            // =========================================================================
            // [TRACK SYNC] 다음 루프로 넘어가기 전, Track 2에서 준비된 N+1 레이어 장착
            // =========================================================================
            if let Some(task) = next_layer_task.take() {
                let mut ready_layer = task.await??; 
                
                // KV 캐시의 메타데이터(장부)는 기존 N+1 레이어의 것을 보존해야 하므로 덮어씌웁니다.
                ready_layer.self_attn.kv_blocks = self.layers[layer_idx + 1].self_attn.kv_blocks.clone();
                ready_layer.self_attn.active_kv_name = self.layers[layer_idx + 1].self_attn.active_kv_name.clone();
                ready_layer.self_attn.layer_idx = layer_idx + 1; 
                
                self.layers[layer_idx + 1] = ready_layer;
            }
        }

        if target_device.is_cuda() { let _ = target_device.synchronize(); }

        self.current_kv_len = seqlen_offset + seq_len;
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) { xs = xs.to_device(norm_dev)?; }
        Ok(xs.apply(&self.norm)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
        self.current_kv_len = 0;
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
        // Just use the first layer's logic as it's purely mathematical
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
        // Backward compatibility wrapper for old relay logic
        self.inject_live_kv(k_list, v_list, k_scales[0], v_scales[0])
    }

    pub fn inject_live_kv_bitkv(&mut self, k_data: &[Tensor], v_data: &[Tensor], original_shape: &[usize]) -> Result<()> {
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_data.len() {
                // Decompress directly into 0.6B shape
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

        // [STABILITY] Use sequential saving to avoid CUDA_ERROR_INVALID_CONTEXT in rayon threads
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

        // [FIX] Scan for b{offset} folders to build fragments list
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

        // [ROBUST-LENGTH-DETECTION] Get actual length of the last block
        let mut last_chunk_len = 256;
        let (_, last_st_path) = fragments.last().unwrap();
        if let Ok(content) = crate::utils::direct_loader::load_kv_block(&last_st_path.join("l0.st")) {
            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
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
        
        let total_kv_len = max_offset + last_chunk_len;
        self.current_kv_len = total_kv_len;
        println!("[SSD-GLOBAL] Snapshot loaded. Total context length: {} tokens.", total_kv_len);

        self.layers.iter_mut().try_for_each(|layer| {
            layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, &fragments, total_kv_len)
        })?;
        
        // [STABILITY-FIX] Double check if all layers have the same current_kv_len
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
        if self.is_forced_cpu { return Ok(()); } // [FIX] Never move to GPU if user wants CPU
        
        use nvml_wrapper::Nvml;
        
        // VRAM 체크 (NVML 사용)
        let nvml = Nvml::init().ok();
        let mut free_vram = 0;
        
        if let Some(nvml_inst) = &nvml {
            if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                if let Ok(mem) = dev.memory_info() {
                    free_vram = mem.free;
                }
            }
        }

        // 임계값 설정 (안전성 위주로 재조정)
        let danger_zone = 500_000_000; // 500MB 이하: 위험 (내리기 시작)
        let safe_zone = 1_000_000_000; // 1GB 이상: 여유 (올리기 시작)

        // 2. 무게추(Weights) 리밸런싱 - OOM 방지를 위해 공격적인 업로드를 차단합니다.
        if free_vram > 0 && free_vram < danger_zone {
            // [OFFLOAD] GPU -> CPU (뒤쪽 레이어부터)
            for layer in self.layers.iter_mut().rev() {
                if layer.device().is_cuda() {
                    println!("[REBALANCE] (Offset: {}/{}) Low VRAM ({:.2} MB). Offloading Layer {} to CPU.", offset, total_len, free_vram as f64 / 1e6, layer.self_attn.layer_idx);
                    layer.to_device(&Device::Cpu)?;
                    break; 
                }
            }
        } else if free_vram > safe_zone {
            // [STABILITY-FIX] Do NOT auto-upload layers in long context mode to avoid sudden OOM.
            // Layers will be handled individually in process_single_layer.
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLModel {
    pub config: Qwen3VLConfig,
    pub visual: Qwen3VLVisionModel, 
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: QLinear,
    pub rope_deltas: Option<Tensor>,
    pub rope_deltas_cpu: Option<Vec<i64>>, // [CRITICAL FIX] CPU 캐시 전용 필드 추가!
    pub text_device: Device,
    pub vision_device: Device,
    pub mmap: Option<Arc<Mmap>>,
    pub mmproj_mmap: Option<Arc<Mmap>>,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(
        config: &Qwen3VLConfig,
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
        baking_only: bool, // [NEW] Support for 1-layer vision baker
    ) -> Result<Self> {
        let mmproj_mmap = mmproj_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { dtype };
        let mut reader_vision = std::io::Cursor::new(mmproj_mmap);
        let vb_visual = from_gguf_content(config, &ct_vision, &mut reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;

        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        // [OPTIMIZATION] If baking only, limit to 1 layer to save massive VRAM/RAM
        if baking_only {
            println!("[MODEL] Vision Baker Mode: Reducing LLM to 1 layer.");
            t_config.num_hidden_layers = 1;
        }

        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
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
            rope_deltas_cpu: None, // <-- 바로 이 부분입니다
            text_device: text_device.clone(), 
            vision_device: vision_device.clone(), 
            mmap: main_mmap_handle, 
            mmproj_mmap: mmproj_mmap_handle 
        })
    }

    pub fn new<R: std::io::Seek + std::io::Read, R2: std::io::Seek + std::io::Read>(
        config: &Qwen3VLConfig,
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
        baking_only: bool, // [NEW]
    ) -> Result<Self> {
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { dtype };
        let vb_visual = from_gguf_content(config, &ct_vision, reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;
        
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        // [OPTIMIZATION] If baking only, limit to 1 layer
        if baking_only {
            println!("[MODEL] Vision Baker Mode (Reader): Reducing LLM to 1 layer.");
            t_config.num_hidden_layers = 1;
        }

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main.clone(), reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if !baking_only {
            if let Ok(l) = get_qlinear(&ct_main, reader_main, "lm_head", text_device, head_dtype) {
                l
            } else if let Ok(l) = get_qlinear(&ct_main, reader_main, "output", text_device, head_dtype) {
                l
            } else {
                get_qlinear(&ct_main, reader_main, "token_embd", text_device, head_dtype)?
            }
        } else {
            // Minimal header for baking only
            QLinear::new(QMatMul::Tensor(Tensor::zeros((1, 1), head_dtype, text_device)?), None, text_device.clone())
        };

        Ok(Self { 
            config: config.clone(), 
            visual, 
            language_model, 
            lm_head, 
            rope_deltas: None, 
            rope_deltas_cpu: None, // <-- 바로 이 부분입니다
            text_device: text_device.clone(), 
            vision_device: vision_device.clone(), 
            mmap: None, 
            mmproj_mmap: None 
        })
    }
    
    fn get_vision_features(&self, pixel_values: &Tensor, image_grid_thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let image_grid_thw = if !image_grid_thw.device().same_device(&self.vision_device) { image_grid_thw.to_device(&self.vision_device)? } else { image_grid_thw.clone() };
        let (image_embeds, deepstack_image_embeds) = self.visual.forward(&pixel_values, &image_grid_thw)?;
        
        // [CRITICAL FIX] spatial_merge_size와 split_sizes를 계산하던 무거운 유령 로직 전체 삭제!
        let image_embeds = image_embeds.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter().map(|t| Ok(t.to_device(&self.text_device)?)).collect();
        
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
        
        // [CRITICAL FIX] 안 쓰이는 Tensor::zeros 초기화 코드를 삭제하여 VRAM 낭비 제거
        
        let mut mrope_position_deltas = Vec::new();
        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        let image_thw_cpu = if let Some(thw) = image_grid_thw {
            Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?)
        } else { None };

        // [CRITICAL FIX] 바깥쪽 선언 시 타입을 <u32>로 명확히 지정!
        let mut flat_pos_ids: Vec<u32> = Vec::with_capacity(3 * b_sz * seq_len);

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                if ids[i] == vision_start_token_id as u32 && i + 1 < seq_len && ids[i+1] == image_token_id as u32 {
                    // [FIX] 루프 내부 GPU Sync(to_vec1) 삭제 및 CPU 캐시 배열 사용
                    if let Some(thw_cpu_array) = &image_thw_cpu {
                        let thw = &thw_cpu_array[image_idx];
                        image_idx += 1;
                        let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                        
                        // vision_start 토큰 위치
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;

                        // image_pad 토큰들 위치 (3D Grid)
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
                        curr_pos += t.max(h).max(w); // 이미지 점유 폭만큼 위치 점프
                    } else {
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1; curr_pos += 1;
                    }
                } else {
                    // 일반 텍스트 토큰
                    for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                    i += 1;
                    curr_pos += 1;
                }
            }
            
            // [CRITICAL FIX] 루프 내부의 중복 선언(let mut flat_pos_ids = ...)은 완전히 삭제했습니다!
            // 이제 바깥에 만들어둔 거대한 그릇에 1차원으로 계속 이어붙이기만 합니다.
            for d in 0..3 {
                flat_pos_ids.extend_from_slice(&llm_pos_ids[d]);
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        } // 처음 배치 루프 종료

        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), input_ids.device())?;
        
        let target_dtype = if input_ids.device().is_cuda() { DType::BF16 } else { DType::F32 };
        let deltas = Tensor::from_vec(mrope_position_deltas.clone(), (b_sz, 1), input_ids.device())?.to_dtype(target_dtype)?; 
        Ok((position_ids, deltas, mrope_position_deltas))
    }

    pub async fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        // [OPTIMIZATION] 매 토큰이 아닌 16토큰마다 리밸런싱을 수행하여 NVML 오버헤드 제거
        if seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let (b_sz, seq_len) = input_ids.dims2()?;

        // 1. Embedding & Vision Integration
        let mut inputs_embeds = self.language_model.embed_tokens.forward(&input_ids)?;
        
        if let Some(pv) = pixel_values { 
            if let Some(thw) = image_grid_thw { 
                let (image_embeds, _) = self.get_vision_features(pv, thw)?; 
                // let image_embeds = Tensor::cat(&image_embeds, 0)?; 
                let vision_mask = self.get_placeholder_mask(&input_ids, true)?; 
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; 
            } 
        }
        
        // 2. Position IDs calculation (Corrected for Long Context & Vision)
        let position_ids = if seqlen_offset == 0 || self.rope_deltas_cpu.is_none() { 
            let (p_ids, deltas_tensor, deltas_vec) = self.get_rope_index(&input_ids, image_grid_thw, video_grid_thw, None)?; 
            self.rope_deltas = Some(deltas_tensor);
            self.rope_deltas_cpu = Some(deltas_vec); // CPU 캐시 저장
            p_ids
        } else {
            // Decoding/Incremental Step (Zero GPU-CPU Sync!)
            let deltas_cpu = self.rope_deltas_cpu.as_ref().unwrap();
            let mut p_ids_vec = Vec::with_capacity(b_sz);
            
            for b in 0..b_sz {
                let delta = deltas_cpu[b]; // [FIX] 순수 O(1) RAM 읽기! PCIe 통신 완벽 제거!
                let real_start = (seqlen_offset as i64 + delta) as u32; 
                
                let p_id = Tensor::arange(real_start, real_start + seq_len as u32, input_ids.device())? 
                    .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, 1, seq_len))?; 
                p_ids_vec.push(p_id); 
            }
            Tensor::cat(&p_ids_vec, 1)?
        };
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();
 

        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None, session_id, kv_name).await?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let head_dev = self.lm_head.device();
        let head_dtype = if head_dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let hidden_state = if !hidden_state.device().same_device(head_dev) { hidden_state.to_device(head_dev)?
        } else { hidden_state };
        let hidden_state = if hidden_state.dtype() != head_dtype { hidden_state.to_dtype(head_dtype)? } else { hidden_state };
        let logits = self.lm_head.forward(&hidden_state)?;
        
        // [CRITICAL FIX] 5토큰마다 발생하던 GPU Sync 및 파일 I/O 디버그 코드 완전 삭제 완료!
        
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.language_model.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; self.text_device = device.clone(); self.vision_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, offset, total_len) }
}

#[derive(Clone)]
pub struct QuantizedQwen3TextModel {
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: Option<QLinear>,
    pub text_device: Device,
    pub mmap: Option<Arc<Mmap>>,
}

impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(
        config: &Qwen3VLConfig,
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
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }
        
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
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
        config: &Qwen3VLConfig,
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
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }

        let language_model = QuantizedQwen3VLTextModel::new(
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
        // [OPTIMIZATION] 16토큰 주기로 리밸런싱 수행
        if seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        let start = seqlen_offset as u32;
        let position_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();

        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None, session_id, kv_name).await?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let logits = if let Some(head) = &self.lm_head {
            let hidden_state = if !hidden_state.device().same_device(head.device()) { hidden_state.to_device(head.device())? } else { hidden_state };
            head.forward(&hidden_state)?
        } else { hidden_state };
        
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.language_model.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.language_model.to_device(device)?; if let Some(head) = &mut self.lm_head { head.to_device(device)?; } self.text_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, _total_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, offset, _total_len) }
}

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let weight = QMatMul::from_qtensor(weight)?;
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { Some(t.dequantize(device)?.to_dtype(dtype)?) } else { None };
    Ok(QLinear::new(weight, bias, device.clone()))
}

fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType) -> Result<RmsNorm> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let weight = weight.dequantize(device)?.to_dtype(dtype)?;
    Ok(RmsNorm::new(weight, eps))
}

fn from_gguf_content<R: std::io::Seek + std::io::Read>(config: &Qwen3VLConfig, ct: &gguf_file::Content, reader: &mut R, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
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
        let t = t.dequantize(device)?.to_dtype(dtype)?;
        if is_split { split_tensors.entry(base_split_name).or_default().push((split_idx, t)); } else { data.insert(new_name, t); }
    }
    for (name, mut parts) in split_tensors { parts.sort_by_key(|(i, _)| *i); let tensors: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect(); if let Ok(merged) = Tensor::cat(&tensors, 0) { data.insert(name, merged); } }
    if let Some(weight) = data.get("visual.patch_embed.proj.weight") { if weight.rank() == 4 { if let Ok(reshaped) = weight.unsqueeze(2)?.repeat((1, 1, 2, 1, 1)) { data.insert("visual.patch_embed.proj.weight".to_string(), reshaped); println!("[FIX] Reshaped visual.patch_embed.proj.weight to 5D"); } } }
    Ok(VarBuilder::from_tensors(data, dtype, device))
}
