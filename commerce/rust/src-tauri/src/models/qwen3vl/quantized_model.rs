use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use rayon::prelude::*;
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

use crate::{
    models::{
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{
        Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
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
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let target_dtype = self.weight.dtype();
        
        // if x.device().is_cpu() && (x.dtype() == DType::BF16 || target_dtype == DType::BF16) {
        //     println!("[TRACE-NORM-VIOLATION] CPU Norm with BF16! x: {:?}, weight: {:?}", x.dtype(), target_dtype);
        // }

        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let hidden_states = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let hidden_states = hidden_states.to_dtype(target_dtype)?;
        hidden_states.broadcast_mul(&self.weight)
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

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if !self.device.same_device(device) {
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
        let dev = &self.device;
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let (b, s, h) = xs.dims3()?;
        let xs_flat = xs.reshape((b * s, h))?;

        // [FIX] Handle different QMatMul variants correctly to avoid dtype mismatch
        let out = match &self.inner {
            QMatMul::QTensor(_q) => {
                // Quantized path: Candle's QMatMul::forward(QTensor) expects F32 input
                let xs_f32 = xs_flat.to_dtype(DType::F32)?;
                self.inner.forward(&xs_f32)?
            },
            QMatMul::Tensor(t) => {
                // Sliced/Unquantized path: types must match exactly (e.g. BF16 * BF16)
                let xs_typed = xs_flat.to_dtype(t.dtype())?;
                self.inner.forward(&xs_typed)?
            },
            // Handle other variants if they exist in this candle version
            _ => {
                let xs_f32 = xs_flat.to_dtype(DType::F32)?;
                self.inner.forward(&xs_f32)?
            }
        };
        
        let out = out.reshape((b, s, ()))?.to_dtype(target_dtype)?;

        if let Some(bias) = &self.bias {
            let b = if bias.dtype() != target_dtype { bias.to_dtype(target_dtype)? } else { bias.clone() };
            Ok(out.broadcast_add(&b)?)
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
            let json = serde_json::to_string_pretty(&layer_data)?;
            let _ = std::fs::write(path.join(format!("layer{}_meta.json", l_idx)), json);
        }
        Ok(())
    }

    pub fn load_from_file(&self, path: &std::path::Path) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        
        // [DECENTRALIZED-LOAD] 28개 레이어 장부를 각각 읽어서 통합 장부 복원
        for l_idx in 0..28 {
            let meta_path = path.join(format!("layer{}_meta.json", l_idx));
            if meta_path.exists() {
                if let Ok(json) = std::fs::read_to_string(meta_path) {
                    if let Ok(loaded) = serde_json::from_str::<Vec<serde_json::Value>>(&json) {
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
    // [ACCUMULATOR] VRAM 내 병합 캐시: 매번 수십개의 블록을 cat하는 오버헤드 제거
    pub vram_merged_k: Option<Tensor>,
    pub vram_merged_v: Option<Tensor>,
    pub merged_vram_block_count: usize,
}

impl QuantizedQwen3VLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
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
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask_in: Option<&Tensor>,
        seqlen_offset: usize,
        _session_id: Option<String>,
        _kv_name: Option<String>,
        _baking_only: bool,
    ) -> Result<Tensor> {
        let dev = self.q_proj.device();
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };

        let mut vram_count = 0;
        let mut ram_count = 0;
        let mut ssd_count = 0;
        let start_attn = std::time::Instant::now();

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
            let mask = mask.to_dtype(DType::F32)?.affine(-1e9, 0.0)?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            attention_mask_in.map(|m| m.clone())
        };

        let mut query_states = self.q_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        query_states = self.q_norm.forward(&query_states)?.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
        
        let mut key_states = self.k_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        key_states = self.k_norm.forward(&key_states)?.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
        
        let value_states = self.v_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?.to_dtype(target_dtype)?;

        let cos = cos.to_dtype(target_dtype)?;
        let sin = sin.to_dtype(target_dtype)?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states.to_dtype(target_dtype)?, &key_states.to_dtype(target_dtype)?, &cos, &sin, false)?;
        let query_states = query_states.to_dtype(target_dtype)?;
        let key_states = key_states.to_dtype(target_dtype)?;
        
        // 2. [BLOCK-PIPELINE-ALLOCATION] Append or Create New (Mark Dirty for Real-time SSD Write)
        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 256usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };
                        inner.k_cache = Some(Tensor::cat(&[pk.to_dtype(target_dtype)?, k_piece.to_dtype(target_dtype)?], 2)?.contiguous()?);
                        inner.v_cache = Some(Tensor::cat(&[pv.to_dtype(target_dtype)?, v_piece.to_dtype(target_dtype)?], 2)?.contiguous()?);
                        inner.len += take; tokens_to_process -= take; chunk_offset += take;
                        appended = true;
                        
                        // [SSD-TRIGGER] Mark as dirty in registry to force backup for THIS layer
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
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece); inner.v_cache = Some(v_piece);
                }
                
                // [SSD-TRIGGER] Initialize and mark dirty for new blocks
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

        // 3. [SEQUENTIAL-KV-LOADER] Strictly Ordered Hybrid Pipeline
        let mut bulk_ks: Vec<Tensor> = Vec::new();
        let mut bulk_vs: Vec<Tensor> = Vec::new();

        let total_tokens_now = seqlen_offset + q_len;
        for block in &self.kv_blocks {
            let (index, b_off, _b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            // Check physical VRAM presence
            let mut k_vram = None;
            let mut v_vram = None;
            {
                let inner = block.inner.read().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    k_vram = Some(k.to_device(dev)?.to_dtype(target_dtype)?);
                    v_vram = Some(v.to_device(dev)?.to_dtype(target_dtype)?);
                    vram_count += 1;
                }
            }

            if let (Some(k), Some(v)) = (k_vram, v_vram) {
                bulk_ks.push(k); bulk_vs.push(v);
            } else {
                // SSD/RAM Pipeline: Direct load to VRAM for high-speed inference
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
                    let ssd_path = { let reg = self.registry.entries.read().unwrap(); if index < reg.len() { reg[index].ssd_path.clone() } else { None } };
                    if let Some(p) = ssd_path {
                        let mut full_path = p.clone();
                        if full_path.is_relative() && !full_path.starts_with("tmp") { full_path = crate::utils::paths::get_kv_dir(None).join(full_path); }
                        let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                        if let Ok(data) = std::fs::read(&block_file) {
                            if let Ok(st) = safetensors::SafeTensors::deserialize(&data) {
                                let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                    let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                    let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();

                                    // [VRAM-DIRECT] Move BF16 bytes directly to GPU and decompress (cast) there
                                    let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, &Device::Cpu)?;
                                    let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, &Device::Cpu)?;

                                    bulk_ks.push(self.decompress_from_bf16(&kd_t, &meta_os, dev)?);
                                    bulk_vs.push(self.decompress_from_bf16(&vd_t, &meta_os, dev)?);
                                    ssd_count += 1;
                                }
                            }
                        }
                    }
                }
                if let Some(k) = k_cpu {
                    bulk_ks.push(k.to_device(dev)?.to_dtype(target_dtype)?);
                    bulk_vs.push(v_cpu.unwrap().to_device(dev)?.to_dtype(target_dtype)?);
                }
            }
        }

        if bulk_ks.is_empty() { return Err(anyhow!("No KV data found")); }

        // 4. [BULK-ATTENTION] GPU Parallel Concatenation
        let k = Tensor::cat(&bulk_ks, 2)?;
        let v = Tensor::cat(&bulk_vs, 2)?;
        let final_kv_len = k.dim(2)?;

        let (mut k, mut v) = (k, v);
        if self.num_kv_groups > 1 {
            let (b, h, s, d) = k.dims4()?;
            k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
        }

        // [FORCE-TYPE-STRICT] Ensure matmul operands are perfectly matched
        let lhs = query_states.to_dtype(target_dtype)?;
        let rhs = k.transpose(2, 3)?.to_dtype(target_dtype)?;

        // Scores in F32 precision for stability
        let mut attn_weights = lhs.matmul(&rhs)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&Tensor::new(&[self.scaling as f32], dev)?)?;

        // Apply Internally Generated Causal Mask
        if let Some(mask) = &attention_mask {
            let m_len = mask.dim(D::Minus1)?;
            if final_kv_len <= m_len {
                attn_weights = attn_weights.broadcast_add(&mask.narrow(D::Minus1, 0, final_kv_len)?)?;
            }
        }

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        
        // [FINAL-DTYPE-STRICT] Force match for second matmul
        let weights_final = attn_weights.to_dtype(target_dtype)?;
        let v_final = v.to_dtype(target_dtype)?;
        let attn_output = weights_final.matmul(&v_final)?;

        let (b_sz, n_h, q_len, d_h) = attn_output.dims4()?;
        let attn_output = self.o_proj.forward(&attn_output.transpose(1, 2)?.reshape((b_sz, q_len, n_h * d_h))?)?;
        
        if self.layer_idx == 0 {
            println!("[SPEED] Token {} | R:{} S:{} V:{} | Attn: {:?}", total_tokens_now, ram_count, ssd_count, vram_count, start_attn.elapsed());
        }
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
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        // Data is already BF16 tensor (loaded from safetensors), just move to device and cast
        let t = data.to_device(device)?;
        Ok(t.to_dtype(target_dtype)?)
    }

    pub fn decompress_from_1bit(&self, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = packed.device();
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements: usize = original_shape.iter().product();
        let mut decoded = vec![0.0f32; total_elements];
        use rayon::prelude::*;
        decoded.par_chunks_mut(last_dim).enumerate().for_each(|(v_idx, vector_out)| {
            let s = scales_vec[v_idx];
            let t_start = v_idx * last_dim;
            for i in 0..last_dim {
                let global_idx = t_start + i;
                let is_set = (packed_vec[global_idx / 8] & (1 << (global_idx % 8))) != 0;
                vector_out[i] = if is_set { s } else { -s };
            }
        });
        let t = Tensor::from_vec(decoded, original_shape, &Device::Cpu)?;
        Ok(t.to_device(device)?)
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

    pub fn trigger_realtime_incremental_bake(&self, session_id: &str, is_last_chunk: bool, baking_only: bool, is_decoding: bool) -> Result<()> {
        use crate::models::qwen3vl::generate::{BakeTask, SlotTask, BAKE_TX, SLOT_MANAGER, LayerKVDump};

        let total_blocks = self.kv_blocks.len();
        // [VRAM-GUARD] 디코딩 시에는 최신 3개만 VRAM에 유지 (SSD에는 모두 백업)
        let vram_limit_idx = if is_decoding { total_blocks.saturating_sub(3) } else { 0 };

        let target_indices: Vec<usize> = self.kv_blocks.iter().enumerate().filter_map(|(i, b)| {
            let inner = b.inner.read().unwrap();
            let is_full = inner.len == 256;
            
            // [DIRTY-CHECK] Only bake if THIS layer is dirty for this block
            let is_dirty = {
                let reg = self.registry.entries.read().unwrap();
                if i < reg.len() { 
                    if self.layer_idx < reg[i].is_dirty.len() { reg[i].is_dirty[self.layer_idx] } else { true }
                } else { true }
            };

            // [BACKUP-STRATEGY] 
            // 1. 완성된 블록(256개)은 즉시 SSD로 백업 대상
            // 2. 마지막 조각(is_last_chunk)도 백업 대상
            // 3. VRAM에 데이터가 있고, 아직 저장이 안 된(dirty) 경우만 수집
            if (is_full || is_last_chunk) && inner.k_cache.is_some() && is_dirty { Some(i) } else { None }
        }).collect();

        for idx in target_indices {
            // [DIRTY-RESET] Clear dirty flag immediately for THIS layer
            {
                let mut reg = self.registry.entries.write().unwrap();
                if idx < reg.len() {
                    if self.layer_idx < reg[idx].is_dirty.len() { reg[idx].is_dirty[self.layer_idx] = false; }
                }
            }

            let block = self.kv_blocks[idx].clone();
            let (k_opt, v_opt, off, b_idx, b_len) = {
                let mut inner = block.inner.write().unwrap();
                let snapshot = (inner.k_cache.clone(), inner.v_cache.clone(), inner.offset, inner.index, inner.len);
                
                // [OFFLOAD] VRAM 상주 한계를 넘은 블록만 실제 데이터를 비움 (SSD 백업은 별개)
                if is_decoding && idx < vram_limit_idx {
                    inner.k_cache = None;
                    inner.v_cache = None;
                    inner.location = KVLocation::SSD;
                }
                snapshot
            };

            if let (Some(k), Some(v)) = (k_opt, v_opt) {
                let k_cpu = k.to_device(&Device::Cpu)?;
                let v_cpu = v.to_device(&Device::Cpu)?;
                
                let kv_name_raw = self.active_kv_name.clone().unwrap_or_else(|| "text".to_string());
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { 
                    "text".to_string() 
                } else { 
                    last_part.to_string() 
                };
                
                let session_id_owned = session_id.to_string();
                let registry_clone = self.registry.clone();
                let layer_idx = self.layer_idx;
                let num_kv_h = self.num_key_value_heads;
                let h_d = self.head_dim;

                tauri::async_runtime::spawn(async move {
                    if let Some(tx) = BAKE_TX.get() {
                        let sub_path = if baking_only {
                            format!("{}/reference/{}", session_id_owned, kv_type)
                        } else {
                            format!("{}/inference/{}", session_id_owned, kv_type)
                        };

                        let block_dir = crate::utils::paths::get_kv_dir(None).join(&sub_path).join(format!("b{}", off));
                        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                        let k_shape_u32 = vec![1u32, num_kv_h as u32, b_len as u32, h_d as u32];
                        let dump = LayerKVDump {
                            layer_idx,
                            k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            k_shape: Tensor::from_vec(k_shape_u32, (4,), &Device::Cpu).unwrap(),
                            raw_k: Some(k_cpu),
                            raw_v: Some(v_cpu),
                        };

                        let sid = SLOT_MANAGER.acquire_write_slot(b_len).await;
                        let _ = tx.send(SlotTask::Bake(BakeTask {
                            slot_id: sid,
                            task_dir: block_dir,
                            kv_name: Some(sub_path),
                            offset: off,
                            layers: vec![dump],
                            is_relay_baking: baking_only,
                            block_idx: Some(b_idx),
                            registry: registry_clone,
                        })).await;
                    }
                });
            }
        }
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.iter().map(|b| b.inner.read().unwrap().len).sum()
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
        
        let index_json = fs::read_to_string(&index_path)?;
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
                                inner.k_cache = Some(k_raw.to_device(self.q_proj.device())?);
                                inner.v_cache = Some(v_raw.to_device(self.q_proj.device())?);
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
        let target_dtype = DType::F32;

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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, _kv_name: Option<&str>) -> Result<()> {
        // [SSD-ALIGNMENT] 실제 오프셋을 폴더명으로 사용
        let b_str = format!("b{}", offset);
        let block_dir = path.join(&b_str);
        if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
        
        let structured_path = block_dir.join(format!("l{}.st", self.layer_idx));
        
        let mut map = HashMap::new();
        // [PREFIX-FIX] 저장 시에도 실제 오프셋 이름을 접두어로 사용
        let prefix = format!("{}l{}_", b_str, self.layer_idx);

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
            
            candle_core::safetensors::save(&map, &structured_path)?;
            println!("[SSD-SAVE] Layer {} Block {} saved to {:?}", self.layer_idx, offset, structured_path);
            
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
        
        self.kv_blocks.clear();
        
        for (i, (offset, path)) in fragments.iter().enumerate() {
            // [FIX] Calculate actual block length based on current total context
            let b_len = if *offset < current_kv_len {
                (current_kv_len - *offset).min(256)
            } else { 256 };
            
            let new_block = KVBlock::new(KVLocation::SSD, i, b_len, *offset);
            {
                if let Ok(mut inner) = new_block.inner.write() {
                    inner.ssd_path = Some(path.clone());
                } else {
                    println!("[ERROR] Block {} inner lock poisoned during load_kv_cache.", i);
                }
            }
            self.kv_blocks.push(new_block);

            let mut reg = self.registry.entries.write().unwrap();
            if i >= reg.len() {
                reg.push(crate::models::qwen3vl::quantized_model::RegistryEntry {
                    location: vec![KVLocation::SSD; 28],
                    slot_ids: vec![None; 28],
                    token_start: *offset,
                    token_len: b_len,
                    ssd_path: Some(path.clone()),
                    hidden_states_path: vec![None; 28],
                    is_dirty: vec![false; 28], 
                    last_accessed: std::time::Instant::now(),
                    bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                });
            }
            
            // [FIX] 이미 RAM에 있거나 로딩 중이면 SSD로 덮어쓰지 않음
            let current_loc = reg[i].location[self.layer_idx];
            if current_loc != KVLocation::RAM && current_loc != crate::models::qwen3vl::quantized_model::KVLocation::Loading {
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let is_relay_source = fname == "l0.st";
                if is_relay_source || fname.contains(&format!("l{}.st", self.layer_idx)) {
                    reg[i].location[self.layer_idx] = KVLocation::SSD;
                    reg[i].ssd_path = Some(path.clone());
                }
            }
            if reg[i].ssd_path.is_none() {
                reg[i].ssd_path = Some(path.clone());
            }
        }

        if self.layer_idx == 0 {
            println!("[SSD-LOAD] Layer 0 registered {} blocks.", fragments.len());
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
        self.self_attn.to_device(device)?;
        if let Some(gate) = &mut self.mlp_gate { gate.to_device(device)?; }
        if let Some(up) = &mut self.mlp_up { up.to_device(device)?; }
        if let Some(down) = &mut self.mlp_down { down.to_device(device)?; }
        self.input_layernorm.to_device(device)?;
        if let Some(norm) = &mut self.post_attention_layernorm { norm.to_device(device)?; }
        Ok(())
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

        let mut cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
        if cos.dtype() != target_dtype { cos = cos.to_dtype(target_dtype)?; }

        let mut sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };
        if sin.dtype() != target_dtype { sin = sin.to_dtype(target_dtype)?; }

        let attention_mask = if let Some(mask) = attention_mask {
             Some(if !mask.device().same_device(dev) { mask.to_device(dev)? } else { mask.clone() })
        } else {
             None
        };

        let residual = xs.clone();
        let xs = self.input_layernorm.forward(&xs)?;
        let xs = self.self_attn.forward(&xs, &cos, &sin, attention_mask.as_ref(), seqlen_offset, session_id, kv_name, baking_only)?;
        
        // [HARDENING] Residual Addition DType Guard
        let xs = if xs.dtype() != residual.dtype() { xs.to_dtype(residual.dtype())? } else { xs };
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
            let xs = if xs.dtype() != residual.dtype() { xs.to_dtype(residual.dtype())? } else { xs };
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
    pub device_id: usize, // [NEW] 실제 할당된 GPU ID 저장
    pub mmap: Option<Arc<Mmap>>, 
    pub registry: KVRegistry, // [NEW] 모델 전체 공유 목차
    pub baking_only: bool,
    pub is_forced_cpu: bool,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    pub pinned_layer_count: usize,
    pub current_kv_len: usize,
}

impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        mmap_handle: Option<Arc<Mmap>>,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(&mut reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else {
             return Err(anyhow!("Failed to load embedding."));
        };

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;

        let nvml = nvml_wrapper::Nvml::init().ok();
        let current_device = device.clone(); 
        
        let mut layer_weight_size = 0_u64;
        let probe_prefix_gguf = "blk.0.";
        let probe_prefix_std = "model.layers.0.";
        let layer_prefix = if ct.tensor_infos.keys().any(|k| k.starts_with(probe_prefix_gguf)) { probe_prefix_gguf } else { probe_prefix_std };

        for (name, info) in &ct.tensor_infos {
            if name.starts_with(layer_prefix) {
                let elements: usize = info.shape.elem_count();
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                layer_weight_size += size as u64;
            }
        }
        
        let _cost_per_layer = if baking_only { layer_weight_size / 3 } else { layer_weight_size };
        let _estimated_activation_buffer = 200_000_000;
        let mut _simulated_free_vram: u64 = 0;
        let mut _is_vram_checked = false;
        let mut _safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         _simulated_free_vram = mem.free;
                         _is_vram_checked = true;
                         let os_reserve = 100_000_000; 
                         _safety_floor = os_reserve + kv_reserve + _estimated_activation_buffer;
                     }
                 }
             }
        }

        let mut layer_devices = vec![];
        let mut pinned_layer_count = 0;
        
        for _ in 0..config.num_hidden_layers {
            layer_devices.push(current_device.clone());
            if current_device.is_cuda() {
                pinned_layer_count += 1;
            }
        }

        let pool = rayon::ThreadPoolBuilder::new().num_threads(crate::utils::resources::get_optimal_thread_config(current_device.is_cpu()).thread_count).build()?;
        let final_config = config; 
        let num_layers_to_load = if baking_only { 1 } else { final_config.num_hidden_layers };
        let registry = KVRegistry::new();

        let layers: Result<Vec<_>> = pool.install(|| {
            (0..num_layers_to_load).into_par_iter().zip(layer_devices).map(|(layer_idx, layer_device)| {
                let mut local_cursor = std::io::Cursor::new(mmap);
                let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
                let gguf_blk = format!("blk.{layer_idx}");
                let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
                QuantizedQwen3VLTextDecoderLayer::new(final_config, ct, &mut local_cursor, &prefix, &layer_device, layer_dtype, layer_idx, baking_only, registry.clone())
            }).collect()
        });
        
        let layers = layers?;
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let last_device = layers.last().map(|l| l.device()).unwrap_or(device);
        let norm_dtype = if last_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, &mut reader, norm_prefix, config.rms_norm_eps, last_device, norm_dtype)?;
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
            mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_else(|| if config.head_dim == 128 { vec![16, 24, 24] } else { vec![] }), 
            device_id, // [NEW]
            mmap: mmap_handle, 
            registry, 
            baking_only, 
            is_forced_cpu, 
            active_session_id: None, 
            active_kv_name: None, 
            pinned_layer_count, 
            current_kv_len: 0 
        })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else {
             return Err(anyhow!("Failed to load embedding."));
        };

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;
        let registry = KVRegistry::new();
        let nvml = nvml_wrapper::Nvml::init().ok();
        let current_device = device.clone();
        
        let mut layer_weight_size = 0_u64;
        let probe_prefix_gguf = "blk.0.";
        let probe_prefix_std = "model.layers.0.";
        let layer_prefix = if ct.tensor_infos.keys().any(|k| k.starts_with(probe_prefix_gguf)) { probe_prefix_gguf } else { probe_prefix_std };

        for (name, info) in &ct.tensor_infos {
            if name.starts_with(layer_prefix) {
                let elements: usize = info.shape.elem_count();
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                layer_weight_size += size as u64;
            }
        }
        
        let _cost_per_layer = if baking_only { layer_weight_size / 3 } else { layer_weight_size };
        let mut _simulated_free_vram: u64 = 0;
        let mut _is_vram_checked = false;
        let mut _safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         _simulated_free_vram = mem.free;
                         _is_vram_checked = true;
                         _safety_floor = 50_000_000 + kv_reserve + 50_000_000;
                     }
                 }
             }
        }

        let mut layers = vec![];
        let mut pinned_layer_count = 0;
        let num_layers_to_load = config.num_hidden_layers;

        for layer_idx in 0..num_layers_to_load {
            // [FIX] 사용자 요청에 따라 모든 레이어를 GPU(Cuda)에 강제 할당
            let layer_device = current_device.clone();
            if current_device.is_cuda() { pinned_layer_count += 1; }
            
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
            layers.push(QuantizedQwen3VLTextDecoderLayer::new(config, ct, reader, &prefix, &layer_device, if layer_device.is_cpu() { DType::F32 } else { dtype }, layer_idx, baking_only, registry.clone())?);
        }
        
        let norm_prefix = if ct.tensor_infos.contains_key("output_norm.weight") { "output_norm" } else { &format!("{base_name}.norm") };
        let norm_device = layers.last().map(|l| l.device()).unwrap_or(&current_device);
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, norm_device, if norm_device.is_cpu() { DType::F32 } else { dtype })?;
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), 
            mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_else(|| if config.head_dim == 128 { vec![16, 24, 24] } else { vec![] }), 
            device_id, // [NEW]
            mmap: None, 
            registry, 
            baking_only, 
            is_forced_cpu, 
            active_session_id: None, 
            active_kv_name: None, 
            pinned_layer_count, 
            current_kv_len: 0 
        })
    }

    pub fn load_kv_cache_chunked(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen3vl::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let index_path = kv_dir.join(kv_name).join("layer0.json");
        
        if !index_path.exists() { return Ok(()); }
        
        // [FIX-REGISTRY-SIZE] 로딩 전 인덱스를 확인하여 레지스트리 공간 선제 확보
        let index_json = fs::read_to_string(&index_path)?;
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
    ) -> Result<Vec<Tensor>> {
        let mut results = Vec::with_capacity(chunk_offsets.len());
        let chunk_size = 256;
        let current_seq_len = xs.dim(1)?;
        // [STABILITY] 텐서 장치가 아닌, 레이어에 고정된 싱글톤 장치 참조 사용
        let target_device = self.layers[layer_idx].device().clone();

        for (chunk_idx, &i) in chunk_offsets.iter().enumerate() {
            let take = (current_seq_len - i).min(chunk_size);
            
            // [SLIDING-WINDOW-PREFETCH] 현재 청크 연산 중에 다음 레이어들의 대응하는 청크를 미리 로드
            let prefetch_window = 4;
            let look_ahead_layers = 2;
            let target_chunks = if chunk_idx == 0 { (0..=prefetch_window).collect::<Vec<_>>() } else { vec![chunk_idx + prefetch_window] };

            for t_idx in target_chunks {
                if t_idx < chunk_offsets.len() {
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
                                    tauri::async_runtime::spawn(async move {
                                        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_load_worker};
                                        let sid = SLOT_MANAGER.acquire_read_slot().await;
                                        if let Ok(tx) = get_load_worker().await {
                                            let _ = tx.send(SlotTask::Load(LoadTask { slot_id: sid, path, layer_idx: target_layer, kv_name: kv_name_for_load, shared_block, registry: reg_clone })).await;
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

            println!("[DIAG-CHUNK] L{} | Chunk {}/{} (Tokens {}..{}) | Device: {:?}", 
                layer_idx, chunk_idx + 1, chunk_offsets.len(), i, i + take, target_device);

            let xs_chunk = xs.narrow(1, i, take)?;
            let cos_chunk = cos.narrow(cos.rank().saturating_sub(2), i, take)?;
            let sin_chunk = sin.narrow(sin.rank().saturating_sub(2), i, take)?;

            let out = self.layers[layer_idx].forward(&xs_chunk, &cos_chunk, &sin_chunk, None, seqlen_offset + i, session_id.clone(), kv_name.clone(), baking_only)?;
            results.push(out);

            // [STRATEGY] Real-time backup during Prefill, Baking or Decoding
            if let Some(sid) = &session_id {
                let is_prefill = take > 1;
                let is_last = chunk_idx == chunk_offsets.len() - 1;
                // [FIX] Always trigger baking to ensure SSD mirror is up-to-date even during decoding
                let _ = self.layers[layer_idx].self_attn.trigger_realtime_incremental_bake(sid, is_last, baking_only, !is_prefill);
            }

            // [VRAM-EVACUATION] Manage VRAM pressure after chunk processing
            let _ = self.evacuate_vram_to_ram_only(layer_idx).await;
            if target_device.is_cuda() { let _ = target_device.synchronize(); }
        }

        // [VRAM-CLEANUP-POST-PREFILL] After prefill is fully done, clear this layer's VRAM
        if current_seq_len > 1 {
            for block in &self.layers[layer_idx].self_attn.kv_blocks {
                let mut inner = block.inner.write().unwrap();
                inner.k_cache = None;
                inner.v_cache = None;
                if inner.location == KVLocation::VRAM { inner.location = KVLocation::Streaming; }
            }
            // 누산기도 초기화
            self.layers[layer_idx].self_attn.vram_merged_k = None;
            self.layers[layer_idx].self_attn.vram_merged_v = None;
            self.layers[layer_idx].self_attn.merged_vram_block_count = 0;
        }
        
        Ok(results)
    }

    /// [VRAM-EVACUATION] 레이어 연산 직후 VRAM 압박 시 RAM으로 이동
    async fn evacuate_vram_to_ram_only(&mut self, layer_idx: usize) -> Result<()> {
        // [DYNAMIC-LIMITS] 시스템 자원 상황에 따라 임계값 유동적 조절 (OOM 방지)
        let vram_limit = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            
            // VRAM 여유분 체크 (간단히 4.0GB 이상 여유 있으면 최대치 사용)
            if free_ram_gb > 4.0 { 64 } else if free_ram_gb > 2.0 { 32 } else { 8 }
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
                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let mut inner = kv_blocks[idx].inner.write().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
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

        // ... [Steps 1 & 2 omitted for brevity] ...

        // [STEP 1] Load Weights to GPU (이미 있으면 건너뜀)
        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) }; 
        let current_device = self.layers[layer_idx].device();
        
        if !current_device.same_device(&target_device) {
            self.layers[layer_idx].to_device(&target_device)?;
            if !is_decoding { let _ = target_device.synchronize(); }
        }

        // [STEP 2] Load & Decode KV Cache 파편 (현재 레이어용)
        let mut w_blocks_to_load = Vec::new();
        let mut w_chunk_path = std::path::PathBuf::new();
        {
            let reg = self.registry.entries.read().unwrap();
            let layer_kv_blocks = &self.layers[layer_idx].self_attn.kv_blocks;
            for (b_idx, block) in layer_kv_blocks.iter().enumerate() {
                if b_idx < reg.len() {
                    let entry = &reg[b_idx];
                    let loc = entry.location[layer_idx];
                    if loc == KVLocation::SSD || loc == KVLocation::Loading {
                        w_blocks_to_load.push((layer_idx, b_idx, block.clone()));
                        if w_chunk_path.as_os_str().is_empty() {
                            w_chunk_path = entry.ssd_path.clone().unwrap_or_default();
                        }
                    }
                }
            }
        }

        if !w_blocks_to_load.is_empty() && !w_chunk_path.as_os_str().is_empty() {
            for (l_idx, b_idx, block) in w_blocks_to_load {
                let block_offset = b_idx * 256;
                
                // [PATH-STRICT] w_chunk_path (ssd_path)는 이미 .../inference/b0 폴더임
                let base_dir = w_chunk_path.clone();
                let filename = format!("l{}.st", l_idx);
                let inf_path = base_dir.join(&filename);
                
                // reference는 상위-상위 폴더/reference/b0/l0.st (필요시 조립)
                let bak_path = base_dir.parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.join("reference").join(format!("b{}", block_offset)).join("l0.st"))
                    .unwrap_or_else(|| base_dir.join("l0.st"));

                let actual_path = if inf_path.is_file() { 
                    Some(inf_path) 
                } else if bak_path.is_file() {
                    Some(bak_path)
                } else {
                    None
                };

                if let Some(act_p) = actual_path {
                    if let Ok(content) = crate::utils::direct_loader::load_kv_block(&act_p) {
                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                            let is_relay_file = act_p.file_name().map(|n| n == "l0.st").unwrap_or(false);
                            let prefix = if is_relay_file { format!("b{}_l0_", block_offset) } else { format!("b{}_l{}_", block_offset, l_idx) };
                            let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                            
                            // [MODIFIED] BF16 Load
                            if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                
                                let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, &Device::Cpu)?;
                                let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, &Device::Cpu)?;

                                let metadata = BitKVMetadata {
                                    k_data: kd_t,
                                    v_data: vd_t,
                                    original_shape: meta_os,
                                };
                                let mut inner = block.inner.write().unwrap();
                                inner.bitkv_metadata = Some(metadata);
                                inner.location = KVLocation::RAM;
                                let mut reg_w = self.registry.entries.write().unwrap();
                                reg_w[b_idx].location[l_idx] = KVLocation::RAM;
                            }
                        }
                    }
                }
            }
        }

        // [STEP 3] GPU 연산 실행
        let current_seq_len = xs.dim(1)?;
        let chunk_offsets: Vec<usize> = (0..current_seq_len).step_by(256).collect();
        let next_xs_all = self.process_chunks_iterative(layer_idx, &chunk_offsets, &xs, cos, sin, seqlen_offset, session_id, kv_name, baking_only).await?;
        let mut next_xs = Tensor::cat(&next_xs_all, 1)?;

        if let (Some(embed), Some(mask)) = (deepstack_embed, visual_mask) {
            next_xs = mask_index_add(&next_xs.squeeze(0)?, &mask.squeeze(0)?, embed)?.unsqueeze(0)?;
        }

        // [STEP 4] 실시간 VRAM 해제 및 자원 반납
        if target_device.is_cuda() { let _ = target_device.synchronize(); }
        
        // 1. KV 캐시 즉시 대피 및 VRAM 삭제
        let sid_opt = self.active_session_id.clone();
        if let Some(sid) = sid_opt {
            // [FIX] 베이킹 모드에서는 청크별로 이미 저장했으므로, 레이어 끝에서의 중복/차단 대피를 건너뜁니다.
            if !self.baking_only {
                let _ = self.evacuate_layer_kv_to_cpu(layer_idx, &sid, seqlen_offset, input_token_count).await;
            }
        }

        // 2. [VRAM-PERSISTENCE] 디코딩 시에는 무게추를 GPU에 유지, 프리필 시에만 CPU로 반납
        if !is_decoding && target_device.is_cuda() {
            self.layers[layer_idx].to_device(&Device::Cpu)?;
        }
        
        if !is_decoding {
            println!("[ENGINE-TRACE] << End Layer {} | VRAM Evicted. | Time: {:.2}s", layer_idx, start_layer_time.elapsed().as_secs_f32());
        }
        Ok(next_xs)
    }

    /// [DEC-SPEED-UP] 디코딩 속도를 위해 모든 레이어를 GPU에 상주 시킴
    pub async fn pin_all_layers_to_gpu(&mut self) -> Result<()> {
        let target_device = crate::utils::get_cuda_device(self.device_id);
        println!("[DEC-SPEED-UP] Pinning all layers to GPU for vertical inference...");
        for (_i, layer) in self.layers.iter_mut().enumerate() {
            if !layer.device().same_device(&target_device) {
                layer.to_device(&target_device)?;
            }
        }
        if target_device.is_cuda() { let _ = target_device.synchronize(); }
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
        
        // 1. VRAM 데이터를 CPU로 복제만 수행 (압축은 워커가 함)
        let mut block_groups: std::collections::HashMap<usize, Vec<LayerKVDump>> = std::collections::HashMap::new();
        
        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            for block in &mut layer.self_attn.kv_blocks {
                let inner = block.inner.write().unwrap();
                
                // [STRICT-RELAY] Include partial blocks and check per-layer dirty flag
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
                        raw_k: Some(k.to_device(&Device::Cpu)?),
                        raw_v: Some(inner.v_cache.as_ref().unwrap().to_device(&Device::Cpu)?),
                    });
                    
                    // Reset per-layer dirty flag
                    {
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            if l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] = false; }
                        }
                    }
                }
            }
        }

        if block_groups.is_empty() { return Ok(()); }

        // 2. 블록별 통합 태스크 전송
        if let Some(tx) = BAKE_TX.get() {
            let kv_dir = crate::utils::paths::get_kv_dir(None);
            let mode = self.baking_only;
            
            // [FIX] 2d4cf92 커밋의 경로 정규화 및 세션 격리
            let kv_name_raw = kv_name.unwrap_or("text");
            let last_part = kv_name_raw.split('/').last().unwrap_or("text");
            let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { 
                "text".to_string() 
            } else { 
                last_part.to_string() 
            };
            
            let sub_path = if mode {
                format!("{}/reference/{}", session_id, kv_type)
            } else {
                format!("{}/inference/{}", session_id, kv_type)
            };

            for (off, layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(256).await;
                let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }

                let _ = tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid,
                    task_dir: block_dir,
                    kv_name: Some(sub_path.clone()),
                    offset: off,
                    layers,
                    is_relay_baking: mode,
                    block_idx: Some(off / 256),
                    registry: self.registry.clone(),
                })).await;
            }
        }
        Ok(())
    }

    /// [NEW-STRICT] 계층형 메모리 관리 (VRAM -> RAM -> SSD)
    async fn evacuate_layer_kv_to_cpu(&mut self, layer_idx: usize, session_id: &str, start_off: usize, _len: usize) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let block_start = (start_off / 256) * 256;

        // [DYNAMIC-LIMITS] 0.6B 모델은 가벼우므로 10k 문맥 전체를 VRAM에 상주시켜 PCIe 병목을 제거합니다.
        let (vram_limit, ram_limit) = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let free_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            
            // VRAM: 28개 레이어 x 40개 블록 = 1,120개. 2,048개까지 VRAM 상주 허용 (속도 극대화)
            let v_limit = 2048; 
            
            // RAM: 만약 VRAM에서 쫓겨나더라도 RAM에 5,000개까지는 안전하게 보관
            let r_limit = if free_ram_gb > 4.0 { 5000 } else { 2048 };
            (v_limit, r_limit)
        };

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
                vram_indices.sort_by_key(|k| k.1);
                let num_to_evict = vram_indices.len() - vram_limit;
                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let mut inner = kv_blocks[idx].inner.write().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                        let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
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
                ram_indices.sort_by_key(|k| k.1);
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
                                k_data: k.to_device(&Device::Cpu)?.to_dtype(DType::BF16)?,
                                v_data: v.to_device(&Device::Cpu)?.to_dtype(DType::BF16)?,
                                k_shape: Tensor::from_vec(k_shape_vec, (k.shape().dims().len(),), &Device::Cpu)?,
                                raw_k: None,
                                raw_v: None,
                            },
                            inner.offset,
                            inner.len
                        ));
                        inner.k_cache = None;
                        inner.v_cache = None;
                        inner.location = KVLocation::SSD;
                        if inner.index < reg.len() { reg[inner.index].location[layer_idx] = KVLocation::SSD; }
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
        kv_name: Option<String>, // [RESTORED]
    ) -> Result<Tensor> {
        self.active_session_id = session_id.clone();
        self.active_kv_name = kv_name.clone(); // [RESTORED]

        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mut xs = inputs_embeds.to_device(&target_device)?.to_dtype(target_dtype)?.contiguous()?;

        let position_ids = match position_ids_in {
            Some(ids) => ids.clone(),
            None => Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?,
        };
        
        // [DIAG-ROPE] 위치 정보 모니터링
        if seqlen_offset % 10 == 0 || seq_len > 1 {
            let p_min = position_ids.flatten_all()?.min(0)?.to_scalar::<u32>()?;
            let p_max = position_ids.flatten_all()?.max(0)?.to_scalar::<u32>()?;
            println!("[DIAG-ENGINE-INNER] Forward | Offset: {} | SeqLen: {} | PosRange: {}..{}", seqlen_offset, seq_len, p_min, p_max);
        }

        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        
        // [FAST-PATH] 0.6B 모델은 수직적(전체) 추론을 사용하여 속도를 극대화합니다.
        // 각 레이어의 무게추는 이미 메모리에 있으며, KV 캐시만 필요시 SSD에서 읽어옵니다.
                let total_layers = self.layers.len();
                
                // [FIX] 모든 레이어에 현재 세션의 KV 폴더명을 동기화합니다.
                for layer in self.layers.iter_mut() {
                    layer.self_attn.active_kv_name = kv_name.clone();
                }
        
                for layer_idx in 0..total_layers {
            if layer_idx % 7 == 0 || layer_idx == total_layers - 1 {
                println!("[ENGINE] Running Layer {}/{}", layer_idx + 1, total_layers);
            }
            let deepstack_embed = deepstack_visual_embeds.as_ref().and_then(|v| v.get(layer_idx));
            xs = self.process_single_layer(layer_idx, xs, &cos, &sin, seqlen_offset, deepstack_embed, visual_pos_masks, session_id.clone(), kv_name.clone(), self.baking_only).await?;
        }

        if target_device.is_cuda() { let _ = target_device.synchronize(); }

        // [FLUSH-COMMIT] 베이킹 모드일 경우 이미 레이어별 루프에서 즉시 SSD 저장이 실행되었습니다.
        // 따라서 여기서 중복으로 전체 flush를 수행할 필요가 없으므로 제거하여 병목을 방지합니다.

        self.current_kv_len = seqlen_offset + seq_len;
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) { xs = xs.to_device(norm_dev)?; }
        Ok(xs.apply(&self.norm)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
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
        })
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size, None)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        if !path.exists() { return Ok(()); }

        // [CENTRALIZED-INDEX-REDIRECT] 통합 인덱스가 있으면 청크 로더 사용
        if let Some(name) = kv_name {
            let kv_dir = crate::utils::paths::get_kv_dir(None);
            if kv_dir.join(name).join("layer0.json").exists() {
                println!("[PREFILL] Centralized index found for {}. Using RAM-aware chunked load.", name);
                return self.load_kv_cache_chunked(name);
            }
        }

        let mut fragments = Vec::new();
        let mut max_offset = 0;
        let mut last_chunk_len = 0;

        // [FIX] 중첩 폴더 구조 지원 스캔 (reference/pug/b0/l0.st 등)
        let scan_path = if let Some(name) = kv_name { path.join(name) } else { path.to_path_buf() };
        if !scan_path.exists() { return Ok(()); }

        if let Ok(entries) = std::fs::read_dir(&scan_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let dname = entry.file_name().to_string_lossy().to_string();
                    if dname.starts_with('b') {
                        let offset = dname[1..].parse::<usize>().unwrap_or(0);
                        // 해당 폴더 안에서 아무 .st 파일이나 하나 찾아서 대표 경로로 지정
                        if let Ok(sub_entries) = std::fs::read_dir(entry.path()) {
                            if let Some(st_file) = sub_entries.flatten().find(|e| e.file_name().to_string_lossy().ends_with(".st")) {
                                if offset >= max_offset { max_offset = offset; }
                                fragments.push((offset, st_file.path()));
                            }
                        }
                    }
                }
            }
        }
        
        if fragments.is_empty() { return Ok(()); }
        fragments.sort_by_key(|f| f.0);

        // 마지막 프래그먼트에서 실제 길이를 읽어와 전체 길이를 확정합니다.
        let (_, last_st_path) = fragments.last().unwrap();
        if let Ok(content) = crate::utils::direct_loader::load_kv_block(last_st_path) {
            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                if let Ok(view) = st.tensor("k_shape") {
                    let shape_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) };
                    last_chunk_len = shape_u32[2] as usize;
                }
            }
        }
        
        let total_kv_len = max_offset + last_chunk_len;
        self.current_kv_len = total_kv_len;

        self.layers.iter_mut().try_for_each(|layer| {
            layer.load_kv_cache(&scan_path, device, expected_len, upscale_refill_len, kv_name, &fragments, total_kv_len)
        })
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

        // 2. 무게추(Weights) 리밸런싱
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
            // [UPLOAD] CPU -> GPU (한 번에 4개씩 안전하게 업로드)
            let mut upload_count = 0;
            let target_dev = crate::utils::get_cuda_device(device_id);
            
            for layer in self.layers.iter_mut() {
                if layer.device().is_cpu() {
                    layer.to_device(&target_dev)?;
                    upload_count += 1;
                    if upload_count >= 4 { break; } 
                }
            }
            if upload_count > 0 {
                println!("[REBALANCE] (Offset: {}/{}) Free VRAM ({:.2} GB). Uploaded {} layers to GPU.", offset, total_len, free_vram as f64 / 1e9, upload_count);
                let _ = target_dev.synchronize(); 
            }
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
    pub text_device: Device,
    pub vision_device: Device,
    pub mmap: Option<Arc<Mmap>>,
    pub mmproj_mmap: Option<Arc<Mmap>>,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        main_mmap_handle: Option<Arc<Mmap>>,
        ct_vision: &gguf_file::Content,
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
        let vb_visual = from_gguf_content(config, ct_vision, &mut reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;

        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        // [OPTIMIZATION] If baking only, limit to 1 layer to save massive VRAM/RAM
        if baking_only {
            println!("[MODEL] Vision Baker Mode: Reducing LLM to 1 layer.");
            t_config.num_hidden_layers = 1;
        }

        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
            &t_config, ct_main, main_mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only
        )?;

        let main_mmap = main_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader_main = std::io::Cursor::new(main_mmap);
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear(ct_main, &mut reader_main, "lm_head", text_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, &mut reader_main, "output", text_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, &mut reader_main, "token_embd", text_device, head_dtype)?
        };

        Ok(Self { config: config.clone(), visual, language_model, lm_head, rope_deltas: None, text_device: text_device.clone(), vision_device: vision_device.clone(), mmap: main_mmap_handle, mmproj_mmap: mmproj_mmap_handle })
    }

    pub fn new<R: std::io::Seek + std::io::Read, R2: std::io::Seek + std::io::Read>(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        reader_main: &mut R,
        ct_vision: &gguf_file::Content,
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
        let vb_visual = from_gguf_content(config, ct_vision, reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;
        
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        // [OPTIMIZATION] If baking only, limit to 1 layer
        if baking_only {
            println!("[MODEL] Vision Baker Mode (Reader): Reducing LLM to 1 layer.");
            t_config.num_hidden_layers = 1;
        }

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if !baking_only {
            if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) {
                l
            } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) {
                l
            } else {
                get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype)?
            }
        } else {
            // Minimal header for baking only
            QLinear::new(QMatMul::Tensor(Tensor::zeros((1, 1), head_dtype, text_device)?), None, text_device.clone())
        };

        Ok(Self { config: config.clone(), visual, language_model, lm_head, rope_deltas: None, text_device: text_device.clone(), vision_device: vision_device.clone(), mmap: None, mmproj_mmap: None })
    }
    
    fn get_vision_features(&self, pixel_values: &Tensor, image_grid_thw: &Tensor) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let pixel_values = if !pixel_values.device().same_device(&self.vision_device) { pixel_values.to_device(&self.vision_device)? } else { pixel_values.clone() };
        let image_grid_thw = if !image_grid_thw.device().same_device(&self.vision_device) { image_grid_thw.to_device(&self.vision_device)? } else { image_grid_thw.clone() };
        let (image_embeds, deepstack_image_embeds) = self.visual.forward(&pixel_values, &image_grid_thw)?;
        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let split_sizes: Vec<usize> = prod_tensor_last_dim(&image_grid_thw)?.to_vec1::<u32>()?.iter().map(|&x| x as usize / spatial_merge_size.pow(2)).collect();
        let image_embeds = image_embeds.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter().map(|t| Ok(t.to_device(&self.text_device)?)).collect();
        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
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
    ) -> Result<(Tensor, Tensor)> {
        // [ROPE-FIX] 이미지 격자 구조를 반영한 실제 3D mRoPE 인덱스 계산 로직
        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let image_token_id = self.config.image_token_id.unwrap_or(0);
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(0);
        
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut position_ids = Tensor::zeros((3, b_sz, seq_len), DType::U32, input_ids.device())?;
        let mut mrope_position_deltas = Vec::new();

        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];

            let mut i = 0;
            while i < seq_len {
                if ids[i] == vision_start_token_id as u32 && i + 1 < seq_len && ids[i+1] == image_token_id as u32 {
                    // 이미지 영역 발견
                    if let Some(thw_tensor) = image_grid_thw {
                        let thw = thw_tensor.i(image_idx)?.to_vec1::<u32>()?;
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
            
            // Tensor로 변환 및 삽입
            for d in 0..3 {
                let d_tensor = Tensor::from_vec(llm_pos_ids[d].clone(), (1, seq_len), input_ids.device())?;
                position_ids = position_ids.slice_assign(&[(d..d+1), (b..b+1), (0..seq_len)], &d_tensor)?;
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        }

        let deltas = Tensor::from_vec(mrope_position_deltas, (b_sz, 1), input_ids.device())?.to_dtype(input_ids.dtype())?;
        Ok((position_ids, deltas))
    }

    pub async fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        // [OPTIMIZATION] 매 토큰이 아닌 16토큰마다 리밸런싱을 수행하여 NVML 오버헤드 제거
        if seqlen_offset % 16 == 0 || seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let (b_sz, seq_len) = input_ids.dims2()?;

        // 1. Embedding & Vision Integration
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        if let Some(pv) = pixel_values { 
            if let Some(thw) = image_grid_thw { 
                let (image_embeds, _) = self.get_vision_features(pv, thw)?; 
                let image_embeds = Tensor::cat(&image_embeds, 0)?; 
                let vision_mask = self.get_placeholder_mask(&input_ids, true)?; 
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; 
            } 
        }
        
        // 2. Position IDs calculation (Corrected for Long Context & Vision)
        let (position_ids, _rope_deltas) = if (cache_position_in.is_some() && cache_position_in.unwrap().i(0)?.to_scalar::<u32>()? == 0) || self.rope_deltas.is_none() {
            let (p_ids, deltas) = self.get_rope_index(&input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(deltas);
            (p_ids, self.rope_deltas.as_ref().unwrap().clone())
        } else {
            // Decoding/Incremental Step
            let deltas = self.rope_deltas.as_ref().unwrap();
            let mut p_ids_vec = Vec::new();
            
            for b in 0..b_sz {
                let delta = deltas.i(b)?.to_scalar::<i64>()?;
                let real_start = (seqlen_offset as i64 + delta) as u32;
                let p_id = Tensor::arange(real_start, real_start + seq_len as u32, input_ids.device())?
                    .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, 1, seq_len))?;
                p_ids_vec.push(p_id);
            }
            let p_ids = Tensor::cat(&p_ids_vec, 1)?;
            (p_ids, deltas.clone())
        };
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();
        
        // [DIAG-ROPE] 위치 정보 및 텐서 상태 모니터링
        if seqlen_offset % 10 == 0 || seq_len > 1 {
            let p_min = position_ids.flatten_all()?.min(0)?.to_scalar::<u32>()?;
            let p_max = position_ids.flatten_all()?.max(0)?.to_scalar::<u32>()?;
            println!("[DIAG-ENGINE] Forward | Offset: {} | SeqLen: {} | PosRange: {}..{}", seqlen_offset, seq_len, p_min, p_max);
        }

        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None, session_id, kv_name).await?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let head_dev = self.lm_head.device();
        let head_dtype = if head_dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let hidden_state = if !hidden_state.device().same_device(head_dev) { hidden_state.to_device(head_dev)? } else { hidden_state };
        let hidden_state = if hidden_state.dtype() != head_dtype { hidden_state.to_dtype(head_dtype)? } else { hidden_state };
        
        let logits = self.lm_head.forward(&hidden_state)?;
        
        // [DIAG-LOGITS] 결과값 폭주 여부 확인 및 파일 저장
        if seqlen_offset % 5 == 0 {
            let l_max = logits.flatten_all()?.abs()?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            let p_min = position_ids.flatten_all()?.min(0)?.to_scalar::<u32>()?;
            let p_max = position_ids.flatten_all()?.max(0)?.to_scalar::<u32>()?;
            
            let log_msg = format!("[DIAG-VL] Off: {} | Pos: {}..{} | MaxLogit: {:.4}\n", seqlen_offset, p_min, p_max, l_max);
            let log_path = std::path::Path::new("tmp/logs/engine_diag.log");
            if let Some(parent) = log_path.parent() { let _ = std::fs::create_dir_all(parent); }
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(log_path) {
                use std::io::Write;
                let _ = writeln!(file, "{}", log_msg.trim());
            }
            
            if l_max > 100.0 || l_max.is_nan() {
                println!("[DIAG-WARN] Potential Tensor Explosion! Max Logit: {}", l_max);
            }
        }
        
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
    pub fn new_with_mmap(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }
        
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(&t_config, ct_main, mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        let lm_head = if !baking_only {
            let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut reader = std::io::Cursor::new(mmap);
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, &mut reader, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, &mut reader, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, &mut reader, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: mmap_handle })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, reader_main: &mut R, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        let lm_head = if !baking_only {
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: None })
    }

    pub async fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        // [OPTIMIZATION] 16토큰 주기로 리밸런싱 수행
        if seqlen_offset % 16 == 0 || seqlen_offset == 0 {
            let _ = self.rebalance_layers(self.language_model.device_id, seqlen_offset, total_len);
        }

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        let position_ids = if let Some(cp) = cache_position { 
            let start = cp.flatten_all()?.i(0)?.to_scalar::<u32>()?; 
            Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))? 
        } else {
            // [ROPE-CORRECTION] Use the updated seqlen_offset for position IDs
            let start = seqlen_offset as u32;
            Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?
        };
        
        self.language_model.active_session_id = session_id.clone();
        self.language_model.active_kv_name = kv_name.clone();
        
        // [DIAG-ROPE-TEXT]
        if seqlen_offset % 10 == 0 || seq_len > 1 {
            let p_min = position_ids.flatten_all()?.min(0)?.to_scalar::<u32>()?;
            let p_max = position_ids.flatten_all()?.max(0)?.to_scalar::<u32>()?;
            println!("[DIAG-ENGINE-TEXT] Forward | Offset: {} | SeqLen: {} | PosRange: {}..{}", seqlen_offset, seq_len, p_min, p_max);
        }

        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None, session_id, kv_name).await?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let logits = if let Some(head) = &self.lm_head {
            let hidden_state = if !hidden_state.device().same_device(head.device()) { hidden_state.to_device(head.device())? } else { hidden_state };
            head.forward(&hidden_state)?
        } else { hidden_state };

        // [DIAG-LOGITS-TEXT] 결과값 폭주 여부 확인 및 파일 저장
        if seqlen_offset % 5 == 0 {
            let l_max = logits.flatten_all()?.abs()?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            let p_min = position_ids.flatten_all()?.min(0)?.to_scalar::<u32>()?;
            let p_max = position_ids.flatten_all()?.max(0)?.to_scalar::<u32>()?;

            let log_msg = format!("[DIAG-TEXT] Off: {} | Pos: {}..{} | MaxLogit: {:.4}\n", seqlen_offset, p_min, p_max, l_max);
            let log_path = std::path::Path::new("tmp/logs/engine_diag.log");
            if let Some(parent) = log_path.parent() { let _ = std::fs::create_dir_all(parent); }
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(log_path) {
                use std::io::Write;
                let _ = writeln!(file, "{}", log_msg.trim());
            }

            if l_max > 100.0 || l_max.is_nan() {
                println!("[DIAG-WARN-TEXT] Potential Tensor Explosion! Max Logit: {}", l_max);
            }
        }
        
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