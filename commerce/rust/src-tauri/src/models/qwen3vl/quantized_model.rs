use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use rayon::prelude::*;
use nvml_wrapper::Nvml;
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
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
        masked_scatter_dim0,
        prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
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

// [QUANTIZED-KV] 데이터의 생애주기를 추적하는 정밀 상태값
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KVLocation {
    VRAM,          // GPU에서 실시간 연산 중
    RAM,           // RAM에 BitKV로 압축되어 보관 중 (VRAM 비워짐)
    SSD_PENDING,   // SSD로 저장 중인 상태
    SSD,           // SSD 저장 완료 (메모리에서 완전히 삭제 가능)
    RamSticky, 
    Loading,
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
    pub k_layers: Vec<Arc<tokio::sync::Mutex<Option<Tensor>>>>, // Slave K tensors
    pub v_layers: Vec<Arc<tokio::sync::Mutex<Option<Tensor>>>>, // Slave V tensors
    pub remaining_layers: Arc<std::sync::atomic::AtomicUsize>,
}

impl MemorySlot {
    pub fn new(id: usize, num_layers: usize) -> Self {
        let mut k_layers = Vec::with_capacity(num_layers);
        let mut v_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_layers.push(Arc::new(tokio::sync::Mutex::new(None)));
            v_layers.push(Arc::new(tokio::sync::Mutex::new(None)));
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

#[derive(Clone)]
pub struct BitKVMetadata {
    pub k_anchors: Tensor,
    pub k_packed: Tensor,
    pub k_scales: Tensor,
    pub v_anchors: Tensor,
    pub v_packed: Tensor,
    pub v_scales: Tensor,
    pub original_shape: Vec<usize>,
}

impl BitKVMetadata {
    pub fn estimate_size_bytes(&self) -> usize {
        let k_size = self.k_anchors.elem_count() * 4 + self.k_packed.elem_count() + self.k_scales.elem_count() * 4;
        let v_size = self.v_anchors.elem_count() * 4 + self.v_packed.elem_count() + self.v_scales.elem_count() * 4;
        k_size + v_size
    }
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
    #[serde(skip)]
    pub is_dirty: bool, // SSD 저장이 필요한지 여부
    #[serde(skip, default = "std::time::Instant::now")]
    pub last_accessed: std::time::Instant, // LRU 순위 결정을 위한 접근 시각
    #[serde(skip, default = "default_bitkv_cache")]
    pub bitkv_cache: Arc<std::sync::RwLock<Vec<Option<BitKVMetadata>>>>,
}

// [NEW] 모델 전체가 공유하는 2차원 KV 목차
#[derive(Clone)]
pub struct KVRegistry {
    pub entries: Arc<std::sync::RwLock<Vec<RegistryEntry>>>,
}

impl KVRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    // [NEW] 메모리 압박 시 비울 수 있는 가장 오래된 블록을 찾음
    pub fn find_eviction_candidate(&self) -> Option<usize> {
        let entries = self.entries.read().unwrap();
        let mut oldest_idx = None;
        let mut oldest_time = std::time::Instant::now();

        for (i, entry) in entries.iter().enumerate() {
            // 1. 이미 RAM이나 VRAM에 있고
            // 2. SSD 저장이 완료되었거나(not dirty), 중요도가 낮은 경우
            let in_memory = entry.location.iter().any(|&l| l == KVLocation::VRAM || l == KVLocation::RAM);
            if in_memory && !entry.is_dirty {
                if entry.last_accessed < oldest_time {
                    oldest_time = entry.last_accessed;
                    oldest_idx = Some(i);
                }
            }
        }
        oldest_idx
    }

    pub fn mark_accessed(&self, index: usize) {
        let mut entries = self.entries.write().unwrap();
        if index < entries.len() {
            entries[index].last_accessed = std::time::Instant::now();
        }
    }

    // [NEW] A-B-C 릴레이 순환 시스템 (VRAM & RAM 공통)
    pub fn enforce_relay_relay(&self, current_token_idx: usize) {
        let mut entries = self.entries.write().unwrap();
        let threshold_bytes = 100 * 1024 * 1024; // 100MB
        
        // 1. 현재 각 그룹(컨테이너)의 용량 계산
        // (단순화를 위해 100MB를 토큰 수로 환산하여 그룹핑하거나, 실제 바이트를 추적)
        // 2B 모델 기준 BF16 KV 캐시는 토큰당 약 56KB (28레이어 합산)
        // 100MB / 56KB  약 1800 토큰. 여기서는 안전하게 1800을 그룹 단위로 설정.
        let group_size = 1800; 
        let current_group_id = current_token_idx / group_size;

        let mut vram_to_ram = 0;
        let mut ram_to_ssd = 0;

        for entry in entries.iter_mut() {
            let entry_group_id = entry.token_start / group_size;

            // [VRAM 릴레이]
            // 현재 그룹(B)과 직전 그룹(A)을 제외한 모든 과거 데이터는 RAM으로 밀어냄
            if entry_group_id + 1 < current_group_id {
                let has_vram = entry.location.iter().any(|&l| l == KVLocation::VRAM);
                if has_vram {
                    for loc in entry.location.iter_mut() {
                        if *loc == KVLocation::VRAM { *loc = KVLocation::RAM; }
                    }
                    vram_to_ram += 1;
                }
            }

            // [RAM 릴레이]
            // RAM에 있는 데이터 중 현재 그룹보다 2단계 이상 뒤쳐진 것은 SSD로 밀어냄
            if entry_group_id + 2 < current_group_id {
                let has_sticky = entry.location.iter().any(|&l| l == KVLocation::RamSticky || l == KVLocation::RAM);
                if has_sticky && entry.ssd_path.is_none() {
                    // SSD_PENDING 상태로 변경하여 generate.rs가 캐치하게 함
                    for loc in entry.location.iter_mut() {
                        if *loc == KVLocation::RAM || *loc == KVLocation::RamSticky { 
                            *loc = KVLocation::SSD_PENDING; 
                        }
                    }
                    entry.is_dirty = true;
                    ram_to_ssd += 1;
                }
            }
        }

        if vram_to_ram > 0 || ram_to_ssd > 0 {
            println!("[RELAY-SYSTEM] Group {}: VRAM->RAM: {} blocks, RAM->SSD-Queue: {} blocks", 
                current_group_id, vram_to_ram, ram_to_ssd);
        }
    }

    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let entries = self.entries.read().unwrap();
        let json = serde_json::to_string_pretty(&*entries)?;
        std::fs::write(path.join("metadata.json"), json)?;
        Ok(())
    }

    pub fn load_from_file(&self, path: &std::path::Path) -> Result<()> {
        let meta_path = path.join("metadata.json");
        if !meta_path.exists() { return Ok(()); }
        let json = std::fs::read_to_string(meta_path)?;
        let loaded: Vec<RegistryEntry> = serde_json::from_str(&json)?;
        let mut entries = self.entries.write().unwrap();
        *entries = loaded;
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
}

impl QuantizedQwen3VLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?;
        self.k_norm.to_device(device)?;
        
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
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        _seqlen_offset: usize, // [NEW]
    ) -> Result<Tensor> {
        let dev = self.q_proj.device();
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };

        // if self.layer_idx == 0 {
        //     println!("[TRACE-L0] xs: {:?} {:?}, cos: {:?}, target: {:?}", xs.device(), xs.dtype(), cos.dtype(), target_dtype);
        // }

        // 1. [HARDENING] Inbound Input Alignment
        let xs = if !xs.device().same_device(dev) { 
            let moved = xs.to_device(dev)?;
            // if self.layer_idx == 0 { println!("[TRACE-MOVE] xs moved to {:?}", dev); }
            moved
        } else { xs.clone() };
        
        let xs = if xs.dtype() != target_dtype { 
            let casted = xs.to_dtype(target_dtype)?;
            // if self.layer_idx == 0 { println!("[TRACE-CAST] xs casted to {:?}", target_dtype); }
            casted
        } else { xs };

        let (b_sz, q_len, _) = xs.dims3()?;
        
        let query_states = self.q_proj.forward(&xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?.contiguous()?;
        
        let key_states = self.k_proj.forward(&xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?.contiguous()?;
        
        let value_states = self.v_proj.forward(&xs)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // 2. [HARDENING] RoPE Alignment
        let cos = if cos.dtype() != target_dtype { cos.to_dtype(target_dtype)? } else { cos.clone() };
        let sin = if sin.dtype() != target_dtype { sin.to_dtype(target_dtype)? } else { sin.clone() };
        
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, &cos, &sin, false)?;
        
        // 3. [BLOCK-ALLOCATION] Append or Create New
        let current_chunk_len = key_states.dim(2)?;
        // [BLOCK-ALLOCATION] Split large KV results into multiple 256-token slots
        // [CRITICAL-FIX] Only Layer 0 should manage the Registry to prevent slot explosion in 2B model
        let mut tokens_to_process = current_chunk_len;
        let mut chunk_offset = 0;
        
        if self.layer_idx == 0 && current_chunk_len > 256 {
            println!("[DEBUG-KV] Splitting {} tokens into 256-unit slots...", current_chunk_len);
        }

        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 256usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    if let (Some(prev_k), Some(prev_v)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        inner.k_cache = Some(Tensor::cat(&[prev_k, k_piece], 2)?.contiguous()?);
                        inner.v_cache = Some(Tensor::cat(&[prev_v, v_piece], 2)?.contiguous()?);
                        inner.len += take;
                        tokens_to_process -= take;
                        chunk_offset += take;
                        appended = true;
                    }
                }
            }

            if !appended {
                let take = tokens_to_process.min(256);
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let index = self.kv_blocks.len();
                
                // [FIX] '곱하기 2' 방지: 논리적 시작 지점 + 현재 청크 내 오프셋을 직접 사용
                let current_total = _seqlen_offset + chunk_offset;
                
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece);
                    inner.v_cache = Some(v_piece);
                }
                
                // Registry entry management only on Layer 0
                if self.layer_idx == 0 {
                    let mut reg = self.registry.entries.write().unwrap();
                    if reg.len() <= index {
                        reg.push(RegistryEntry {
                            location: vec![KVLocation::VRAM; 28],
                            slot_ids: vec![None; 28],
                            token_start: current_total,
                            token_len: take,
                            ssd_path: None,
                            is_dirty: true, // [NEW] 백업 필요 표시
                            last_accessed: std::time::Instant::now(),
                            bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                        });
                    }
                }
                
                self.kv_blocks.push(new_block);
                tokens_to_process -= take;
                chunk_offset += take;
            }
        }

        // [PARALLEL-LAYER-PREP] 연산 시작 전 이번 레이어에 필요한 모든 블록을 병렬로 준비
        let registry_clone = self.registry.clone();
        let l_idx = self.layer_idx;
        let dev = self.q_proj.device().clone();
        let model_clone = self.clone();
        
        // 1. 준비가 필요한 블록 선별
        let mut blocks_to_prepare = Vec::new();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if inner.k_cache.is_none() {
                blocks_to_prepare.push(block.clone());
            }
        }

        // 2. CPU 병렬 압축 해제 수행
        if !blocks_to_prepare.is_empty() {
            use rayon::prelude::*;
            // [SLOT-RESERVATION] 연산 동안 슬롯 사용률을 보여주기 위해 슬롯 예약 (실제 데이터는 KVBlock에 상주)
            let num_to_prep = blocks_to_prepare.len();
            
            blocks_to_prepare.into_par_iter().for_each(|block| {
                let index = block.inner.read().unwrap().index;
                let meta = {
                    let reg = registry_clone.entries.read().unwrap();
                    if index < reg.len() {
                        reg[index].bitkv_cache.read().unwrap()[l_idx].clone()
                    } else { None }
                };

                if let Some(m) = meta {
                    if let Ok(k_raw) = model_clone.decompress_from_bitkv(&m.k_anchors, &m.k_packed, &m.k_scales, &m.original_shape, &dev) {
                        if let Ok(v_raw) = model_clone.decompress_from_bitkv(&m.v_anchors, &m.v_packed, &m.v_scales, &m.original_shape, &dev) {
                            let mut inner = block.inner.write().unwrap();
                            inner.k_cache = Some(k_raw);
                            inner.v_cache = Some(v_raw);
                        }
                    }
                }
            });
        }

        // 4. [LIGHTWEIGHT-HYBRID-ENGINE]
        let mut running_m = None; 
        let mut running_s = None; 
        let mut running_o = None; 
        let mut global_start_idx = 0;

        let mut vram_ks = Vec::new();
        let mut vram_vs = Vec::new();
        let mut vram_total_len = 0;

        // [NEW] A-B-C 릴레이 순환 관리 (VRAM & RAM 공통)
        if self.layer_idx == 0 {
            self.registry.enforce_relay_relay(global_start_idx);
        }

        for block in &self.kv_blocks {
            let index = block.inner.read().unwrap().index;
            self.registry.mark_accessed(index); // 접근 시각 갱신
            
            let (mut k, mut v, mut b_len, index) = {
                let inner = block.inner.read().unwrap();
                (inner.k_cache.clone(), inner.v_cache.clone(), inner.len, inner.index)
            };

            let mut loc = {
                let reg = self.registry.entries.read().unwrap();
                if index < reg.len() { reg[index].location[self.layer_idx] } else { KVLocation::VRAM }
            };

            if loc == KVLocation::SSD {
                let (spath, reserved_sid) = {
                    let reg = self.registry.entries.read().unwrap();
                    (reg[index].ssd_path.clone(), reg[index].slot_ids[self.layer_idx])
                };

                if let Some(path) = spath {
                    // [REUSE-LOGIC] 이미 예약된 슬롯이 있다면 로딩 작업이 진행 중인 것으로 간주
                    if reserved_sid.is_some() {
                        loc = KVLocation::Loading;
                    } else {
                        {
                            let mut reg = self.registry.entries.write().unwrap();
                            reg[index].location[self.layer_idx] = KVLocation::Loading;
                        }
                        let shared_block = block.clone();
                        let reg_clone = self.registry.clone();
                        let l_idx = self.layer_idx;
                        tauri::async_runtime::spawn(async move {
                            use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_load_worker};
                            let sid = SLOT_MANAGER.acquire_read_slot().await;
                            if let Ok(tx) = get_load_worker().await {
                                let _ = tx.send(SlotTask::Load(LoadTask {
                                    slot_id: sid, path, layer_idx: l_idx, kv_name: None, shared_block, registry: reg_clone
                                })).await;
                            }
                        });
                        loc = KVLocation::Loading;
                    }
                }
            }

            // [OPTIMIZED-WAIT] Improved yielding during Loading wait
            if loc == KVLocation::Loading {
                let mut attempts = 0;
                while attempts < 5000 { // Increased timeout for CPU/SSD heavy loads
                    let current = {
                        let reg = self.registry.entries.read().unwrap();
                        reg[index].location[self.layer_idx]
                    };
                    
                    if current == KVLocation::RAM || current == KVLocation::RamSticky || current == KVLocation::VRAM { break; }
                    
                    // Release CPU time to allow worker threads to finish IO and BitKV decoding
                    std::thread::yield_now();
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    attempts += 1;
                    
                    if attempts % 500 == 0 {
                        println!("[TRACE] Layer {} Block {} still loading... ({} ms elapsed)", self.layer_idx, index, attempts * 2);
                    }
                }
            }

            let final_loc = {
                let reg = self.registry.entries.read().unwrap();
                reg[index].location[self.layer_idx]
            };

            match final_loc {
                KVLocation::VRAM => {
                    vram_ks.push(k.unwrap_or_else(|| Tensor::zeros((1, self.num_key_value_heads, b_len, self.head_dim), target_dtype, &dev).unwrap()));
                    vram_vs.push(v.unwrap_or_else(|| Tensor::zeros((1, self.num_key_value_heads, b_len, self.head_dim), target_dtype, &dev).unwrap()));
                    vram_total_len += b_len;
                    global_start_idx += b_len;
                    continue;
                }
                KVLocation::RAM | KVLocation::RamSticky => {
                    let mut inner = block.inner.write().unwrap();
                    if inner.k_cache.is_none() {
                        // 1. Try local metadata
                        let meta = inner.bitkv_metadata.clone();
                        
                        // 2. Try registry fallback if local is missing
                        let fallback_meta = if meta.is_none() {
                            let reg = self.registry.entries.read().unwrap();
                            if index < reg.len() {
                                let cache = reg[index].bitkv_cache.read().unwrap();
                                cache[self.layer_idx].clone()
                            } else { None }
                        } else { None };

                        let final_meta = meta.or(fallback_meta);

                        if let Some(meta) = final_meta {
                            let k_raw = self.decompress_from_bitkv(&meta.k_anchors, &meta.k_packed, &meta.k_scales, &meta.original_shape, &dev)?;
                            let v_raw = self.decompress_from_bitkv(&meta.v_anchors, &meta.v_packed, &meta.v_scales, &meta.original_shape, &dev)?;
                            inner.k_cache = Some(k_raw.clone());
                            inner.v_cache = Some(v_raw.clone());
                            k = Some(k_raw);
                            v = Some(v_raw);
                        } else {
                            println!("[ERROR] Layer {} Block {} location is RAM but NO metadata found! (Index: {})", 
                                self.layer_idx, index, index);
                        }
                    }
                }
                _ => { global_start_idx += b_len; continue; }
            }

            let mut k = k.ok_or_else(|| candle_core::Error::Msg("KV Cache not ready after wait".to_string()))?;
            let mut v = v.ok_or_else(|| candle_core::Error::Msg("KV Cache not ready after wait".to_string()))?;

            // [FIX] Ensure KV tensors are on the correct device AND have the correct DType before calculation
            if !k.device().same_device(&dev) { k = k.to_device(&dev)?; }
            if k.dtype() != target_dtype { k = k.to_dtype(target_dtype)?; }
            if !v.device().same_device(&dev) { v = v.to_device(&dev)?; }
            if v.dtype() != target_dtype { v = v.to_dtype(target_dtype)?; }

            // [FIX] Update b_len to actual tensor dimensions to prevent shape mismatch in broadcast_add
            // This handles the case where the last block (e.g. 205 tokens) is smaller than the standard 256.
            b_len = k.dim(2)?;

            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            let mut attn_weights = query_states.matmul(&k.transpose(2, 3)?)?
                .broadcast_mul(&Tensor::new(&[self.scaling as f32], &dev)?.to_dtype(target_dtype)?)?;

            if let Some(mask) = attention_mask {
                let mask_len = mask.dim(D::Minus1)?;
                let current_block_start = global_start_idx.saturating_sub(b_len);
                if current_block_start < mask_len {
                    let take_mask = (mask_len - current_block_start).min(b_len);
                    // [SAFETY-FIX] Double check narrow bounds to prevent slice panic
                    let safe_take = take_mask.min(mask_len.saturating_sub(current_block_start));
                    if safe_take > 0 {
                        let sub_mask = mask.narrow(D::Minus1, current_block_start, safe_take)?.to_dtype(target_dtype)?;
                        if safe_take < b_len {
                            let padded_mask = sub_mask.pad_with_zeros(D::Minus1, 0, b_len - safe_take)?;
                            attn_weights = attn_weights.broadcast_add(&padded_mask)?;
                        } else {
                            attn_weights = attn_weights.broadcast_add(&sub_mask)?;
                        }
                    }
                }
            }

            let m_i = attn_weights.max_keepdim(D::Minus1)?;
            let p_i = attn_weights.broadcast_sub(&m_i)?.exp()?;
            let s_i = p_i.sum_keepdim(D::Minus1)?;
            let o_i = p_i.matmul(&v)?;

            match (running_m, running_s, running_o) {
                (None, None, None) => { running_m = Some(m_i); running_s = Some(s_i); running_o = Some(o_i); }
                (Some(m_prev), Some(s_prev), Some(o_prev)) => {
                    let m_new = m_prev.maximum(&m_i)?;
                    let alpha_prev = m_prev.sub(&m_new)?.exp()?.to_dtype(target_dtype)?;
                    let alpha_i = m_i.sub(&m_new)?.exp()?.to_dtype(target_dtype)?;
                    running_s = Some(s_prev.broadcast_mul(&alpha_prev)?.add(&s_i.broadcast_mul(&alpha_i)?)?.to_dtype(target_dtype)?);
                    running_o = Some(o_prev.broadcast_mul(&alpha_prev)?.add(&o_i.broadcast_mul(&alpha_i)?)?.to_dtype(target_dtype)?);
                    running_m = Some(m_new);
                }
                _ => unreachable!(),
            }
        }

        // 5. [BATCH-VRAM] Fast single-pass for all VRAM blocks
        if !vram_ks.is_empty() {
            // [FIX] Pre-verify device AND dtype alignment before cat to prevent panic
            let mut final_ks = Vec::with_capacity(vram_ks.len());
            let mut final_vs = Vec::with_capacity(vram_vs.len());
            for mut tk in vram_ks { 
                if !tk.device().same_device(&dev) { tk = tk.to_device(&dev)?; }
                if tk.dtype() != target_dtype { tk = tk.to_dtype(target_dtype)?; }
                final_ks.push(tk); 
            }
            for mut tv in vram_vs { 
                if !tv.device().same_device(&dev) { tv = tv.to_device(&dev)?; }
                if tv.dtype() != target_dtype { tv = tv.to_dtype(target_dtype)?; }
                final_vs.push(tv); 
            }

            let mut k = Tensor::cat(&final_ks, 2)?;
            let mut v = Tensor::cat(&final_vs, 2)?;
            
            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            let mut attn_weights = query_states.matmul(&k.transpose(2, 3)?)?
                .broadcast_mul(&Tensor::new(&[self.scaling as f32], &dev)?.to_dtype(target_dtype)?)?;

            if let Some(mask) = attention_mask {
                let mask_len = mask.dim(D::Minus1)?;
                let vram_start = global_start_idx.saturating_sub(vram_total_len);
                if vram_start < mask_len {
                    let take_mask = (mask_len - vram_start).min(vram_total_len);
                    // [SAFETY-FIX] Double check narrow bounds to prevent slice panic
                    let safe_take = take_mask.min(mask_len.saturating_sub(vram_start));
                    if safe_take > 0 {
                        let sub_mask = mask.narrow(D::Minus1, vram_start, safe_take)?.to_dtype(target_dtype)?;
                        if safe_take < vram_total_len {
                            let padded_mask = sub_mask.pad_with_zeros(D::Minus1, 0, vram_total_len - safe_take)?;
                            attn_weights = attn_weights.broadcast_add(&padded_mask)?;
                        } else {
                            attn_weights = attn_weights.broadcast_add(&sub_mask)?;
                        }
                    }
                }
            }

            let m_i = attn_weights.max_keepdim(D::Minus1)?;
            let p_i = attn_weights.broadcast_sub(&m_i)?.exp()?;
            let s_i = p_i.sum_keepdim(D::Minus1)?;
            let o_i = p_i.matmul(&v)?;

            match (running_m, running_s, running_o) {
                (None, None, None) => { running_m = Some(m_i); running_s = Some(s_i); running_o = Some(o_i); }
                (Some(m_prev), Some(s_prev), Some(o_prev)) => {
                    let m_new = m_prev.maximum(&m_i)?;
                    let alpha_prev = m_prev.sub(&m_new)?.exp()?.to_dtype(target_dtype)?;
                    let alpha_i = m_i.sub(&m_new)?.exp()?.to_dtype(target_dtype)?;
                    running_s = Some(s_prev.broadcast_mul(&alpha_prev)?.add(&s_i.broadcast_mul(&alpha_i)?)?.to_dtype(target_dtype)?);
                    running_o = Some(o_prev.broadcast_mul(&alpha_prev)?.add(&o_i.broadcast_mul(&alpha_i)?)?.to_dtype(target_dtype)?);
                    running_m = Some(m_new);
                }
                _ => unreachable!(),
            }
        }

        let attn_output = running_o.unwrap().broadcast_div(&running_s.unwrap())?;
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        /* [RECLAIMED-BY-WORKER] 
           Manual slot release is no longer needed here as the IO worker 
           now releases slots immediately after transferring data to the RAM cache.
        */

        /* 
           [SMART-PURGE-V3] 8GB RAM 환경을 위한 지능형 자원 회수
           - 원칙: 무거운 데이터(Raw BF16)는 즉시 비우고, 가벼운 데이터(Compressed BitKV)는 최대한 RAM에 유지.
           - 이유: SSD 읽기 속도가 가장 큰 병목이므로, 압축 데이터만이라도 RAM에 들고 있으면 다음 레이어에서 CPU가 즉시 복원 가능.
           - 결과: 5만 토큰 상황에서도 SSD 접근을 최소화하면서 8GB RAM 내에서 안정적인 추론 가능.
        */
        for block in &mut self.kv_blocks {
            let mut inner = block.inner.write().unwrap();
            if inner.ssd_path.is_some() || inner.bitkv_metadata.is_some() {
                if inner.location == KVLocation::VRAM || inner.location == KVLocation::RAM {
                    // 무거운 BF16 텐서만 메모리에서 즉시 해제하여 RAM 공간 확보
                    inner.k_cache = None;
                    inner.v_cache = None;
                    
                    // 압축 메타데이터가 있다면 RAM 상태 유지 (다음 레이어에서 SSD 읽기 방지)
                    if inner.bitkv_metadata.is_some() { inner.location = KVLocation::RAM; }
                    else { inner.location = KVLocation::SSD; }
                }
            }
        }

        Ok(attn_output)
    }

    pub fn compress_to_bitkv(&self, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let (b, h, s, d) = t.dims4()?;
        let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let t_data = t_f32.flatten_all()?.to_vec1::<f32>()?;
        
        let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
        let mut anchors = vec![0.0f32; b * h * anchor_count * d];
        let mut packed_residuals = vec![0u8; (b * h * s * d + 7) / 8];
        let mut scales = vec![0.0f32; b * h * s];

        // 1. Parallel pass for Bit Packing (Read-only on t_data)
        // Since packed_residuals is indexed by global token index, we can safely parallelize by Head
        let head_token_size = s * d;
        packed_residuals.par_chunks_mut(head_token_size / 8).enumerate().for_each(|(bh_idx, head_packed)| {
            let bh_offset = bh_idx * head_token_size;
            for i in 0..head_token_size {
                if t_data[bh_offset + i] >= 0.0 {
                    head_packed[i / 8] |= 1 << (i % 8);
                }
            }
        });
        
        // 2. Parallel pass for Anchors and Scales
        // Group anchors and scales by head to avoid mutable borrow conflicts
        let anchor_head_size = anchor_count * d;
        let mut anchors_heads: Vec<&mut [f32]> = anchors.chunks_mut(anchor_head_size).collect();
        let mut scales_heads: Vec<&mut [f32]> = scales.chunks_mut(s).collect();

        // Use zip to process both together in parallel
        anchors_heads.par_iter_mut().zip(scales_heads.par_iter_mut()).enumerate().for_each(|(bh_idx, (anchor_head, head_scales))| {
            let bh_offset = bh_idx * head_token_size;
            for token_idx in 0..s {
                let token_data = &t_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                
                // Copy Anchor if index matches
                if token_idx < 4 || token_idx % 8 == 0 {
                    let anchor_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                    anchor_head[anchor_pos * d .. (anchor_pos + 1) * d].copy_from_slice(token_data);
                }
                
                // Calculate Scale (Max Absolute)
                let mut max_abs = 0.0f32;
                for &v in token_data {
                    let a = v.abs();
                    if a > max_abs { max_abs = a; }
                }
                head_scales[token_idx] = max_abs;
            }
        });

        let packed_len = packed_residuals.len();
        let anchors_tensor = Tensor::from_vec(anchors, vec![b, h, anchor_count, d], &Device::Cpu)?;
        let packed_tensor = Tensor::from_vec(packed_residuals, vec![packed_len], &Device::Cpu)?;
        let scales_tensor = Tensor::from_vec(scales, vec![b, h, s, 1], &Device::Cpu)?;
        Ok((anchors_tensor, packed_tensor, scales_tensor, original_shape))
    }

    pub fn decompress_from_bitkv(&self, anchors: &Tensor, packed: &Tensor, scales: &Tensor, original_shape: &[usize], device: &Device) -> Result<Tensor> {
        // [OPTIMIZATION] If we are on GPU, we want to decompress quickly.
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let anchors_vec = anchors.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        
        // [FIX] Derive dimensions from the actual Anchors tensor to prevent out-of-range slice errors
        let a_dims = anchors.shape().dims(); // [batch, heads, anchor_count, head_dim]
        let batch_size = a_dims[0];
        let num_heads = a_dims[1];
        let anchor_count = a_dims[2];
        let head_dim = a_dims[3];
        
        // original_shape[2] is the total sequence length (e.g. 256)
        let seq_len = original_shape[original_shape.len() - 2];
        let total_elements: usize = batch_size * num_heads * seq_len * head_dim;
        let mut decoded = vec![0.0f32; total_elements];
        
        let head_tokens = seq_len * head_dim;

        use rayon::prelude::*;
        decoded.par_chunks_mut(head_tokens).enumerate().for_each(|(bh_idx, head_out)| {
            let bh_offset = bh_idx * head_tokens;
            let anchor_offset = bh_idx * anchor_count * head_dim;
            
            for s_idx in 0..seq_len {
                let scale = if bh_idx * seq_len + s_idx < scales_vec.len() { scales_vec[bh_idx * seq_len + s_idx] } else { 0.0 };
                let token_out_start = s_idx * head_dim;
                let token_out_end = token_out_start + head_dim;
                
                if token_out_end <= head_out.len() {
                    let token_out = &mut head_out[token_out_start .. token_out_end];
                    
                    // 1. Bit-to-Sign restoration
                    for d_idx in 0..head_dim {
                        let global_bit_idx = bh_offset + s_idx * head_dim + d_idx;
                        if global_bit_idx / 8 < packed_vec.len() {
                            let is_set = (packed_vec[global_bit_idx / 8] & (1 << (global_bit_idx % 8))) != 0;
                            token_out[d_idx] = if is_set { scale } else { -scale };
                        }
                    }
                    
                    // 2. Anchor Refinement
                    if s_idx < 4 || s_idx % 8 == 0 {
                        let a_pos = if s_idx < 4 { s_idx } else { 4 + (s_idx - 4) / 8 };
                        let src_start = anchor_offset + a_pos * head_dim;
                        let src_end = src_start + head_dim;
                        
                        if src_end <= anchors_vec.len() {
                            let anchor_data = &anchors_vec[src_start .. src_end];
                            token_out.copy_from_slice(anchor_data);
                        }
                    }
                }
            }
        });

        let final_shape = vec![batch_size, num_heads, seq_len, head_dim];
        let t = Tensor::from_vec(decoded, final_shape, &Device::Cpu)?;
        Ok(t.to_device(device)?.to_dtype(target_dtype)?)
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
        // [CLEANUP] Explicit Slot Release
        for block in &self.kv_blocks {
            let (slot_id, loc) = {
                let reg = self.registry.entries.read().unwrap();
                let inner = block.inner.read().unwrap();
                if inner.index < reg.len() { 
                    (reg[inner.index].slot_ids[self.layer_idx], reg[inner.index].location[self.layer_idx]) 
                } else { (None, KVLocation::VRAM) }
            };

            // [STICKY-PROTECTION] Do not release persistent cache slots
            if loc == KVLocation::RamSticky { continue; }

            if let Some(id) = slot_id {
                tauri::async_runtime::spawn(async move {
                    SLOT_MANAGER.release_slot(id).await;
                });
            }
        }
        self.kv_blocks.clear();
    }

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.iter().map(|b| b.inner.read().unwrap().len).sum()
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        for block in &mut self.kv_blocks {
            let mut inner = block.inner.write().unwrap();
            inner.k_cache = None;
            inner.v_cache = None;
            inner.bitkv_metadata = None;
            inner.location = KVLocation::SSD;
        }
        self.kv_blocks.clear();
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _offset: usize, kv_name: Option<&str>) -> Result<()> {
        let filename = match kv_name {
            Some(name) => format!("layer_{}_kv.safetensors", name),
            None => format!("layer_{}_kv.safetensors", self.layer_idx),
        };
        let file = path.join(filename);
        let mut map = HashMap::new();

        // [BLOCK-SAVE]
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if inner.location == KVLocation::VRAM {
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    ks.push(k.clone());
                    vs.push(v.clone());
                }
            }
        }

        if !ks.is_empty() {
            let k = Tensor::cat(&ks, 2)?;
            let v = Tensor::cat(&vs, 2)?;
            let (k_anchors, k_packed, k_scales, k_shape) = self.compress_to_bitkv(&k)?;
            let (v_anchors, v_packed, v_scales, _) = self.compress_to_bitkv(&v)?;
            
            map.insert("k_anchors".to_string(), k_anchors);
            map.insert("k_packed".to_string(), k_packed);
            map.insert("k_scales".to_string(), k_scales);
            map.insert("v_anchors".to_string(), v_anchors);
            map.insert("v_packed".to_string(), v_packed);
            map.insert("v_scales".to_string(), v_scales);
            map.insert("k_shape".to_string(), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect(), (k_shape.len(),), &Device::Cpu)?);
            map.insert("mode".to_string(), Tensor::from_vec(vec![3u32], (1,), &Device::Cpu)?);
            candle_core::safetensors::save(&map, &file)?;
        }

        if clear { self.kv_blocks.clear(); }
        Ok(())
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        let mut current_total = 0;
        let mut to_remove = Vec::new();
        let total_blocks = self.kv_blocks.len();
        
        for i in 0..total_blocks {
            let block = &mut self.kv_blocks[i];
            let mut inner = block.inner.write().unwrap();
            
            if current_total + inner.len <= len {
                current_total += inner.len;
            } else {
                let keep_in_this_block = len - current_total;
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
                    current_total += keep_in_this_block;
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

    pub fn load_kv_cache(&mut self, _path: &Path, _device: &Device, _expected_len: usize, _upscale_refill_len: usize, _kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)]) -> Result<()> {
        if fragments.is_empty() { return Ok(()); }
        
        self.kv_blocks.clear();
        
        for (i, (offset, path)) in fragments.iter().enumerate() {
            // [FIX] Standard block size is 256 tokens per fragment
            let b_len = 256; 
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
                    is_dirty: false, // SSD에 이미 있으므로 백업 불필요
                    last_accessed: std::time::Instant::now(),
                    bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                });
            }
            reg[i].location[self.layer_idx] = KVLocation::SSD;
            reg[i].ssd_path = Some(path.clone());
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
        seqlen_offset: usize, // [NEW]
    ) -> Result<Tensor> {
        let dev = self.input_layernorm.weight().device();
        let target_dtype = self.input_layernorm.weight().dtype();

        // if self.self_attn.layer_idx % 5 == 0 || dev.is_cpu() {
        //     println!("[TRACE-LAYER-{}] Device: {:?}, DType: {:?}, In: {:?}", 
        //         self.self_attn.layer_idx, dev, target_dtype, xs.dtype());
        // }
        
        // 2. Ensure inputs are on this device and dtype
        //    (Clone via Cow logic or explicit clone if needed, here we use explicit clones/conversions for safety)
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };

        let mut cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
        if cos.dtype() != target_dtype { cos = cos.to_dtype(target_dtype)?; }

        let mut sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };
        if sin.dtype() != target_dtype { sin = sin.to_dtype(target_dtype)?; }

        let attention_mask = if let Some(mask) = attention_mask {
             // Mask is usually F32 or specific, but we ensure device match
             Some(if !mask.device().same_device(dev) { mask.to_device(dev)? } else { mask.clone() })
        } else {
             None
        };

        let residual = xs.clone();
        let xs = self.input_layernorm.forward(&xs)?;
        let xs = self.self_attn.forward(&xs, &cos, &sin, attention_mask.as_ref(), seqlen_offset)?;
        
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

    pub async fn prepare_layer_blocks_text(&self, layer_idx: usize, registry: &KVRegistry, device: &Device, model: &QuantizedQwen3TextModel) -> Result<()> {
        use rayon::prelude::*;
        let kv_blocks = &self.self_attn.kv_blocks;
        
        kv_blocks.into_par_iter().for_each(|block| {
            let index = block.inner.read().unwrap().index;
            let meta = {
                let reg = registry.entries.read().unwrap();
                if index < reg.len() {
                    reg[index].bitkv_cache.read().unwrap()[layer_idx].clone()
                } else { None }
            };

            if let Some(m) = meta {
                if block.inner.read().unwrap().k_cache.is_none() {
                    if let Ok(k_raw) = model.decompress_from_bitkv(&m.k_anchors, &m.k_packed, &m.k_scales, &m.original_shape, device) {
                        if let Ok(v_raw) = model.decompress_from_bitkv(&m.v_anchors, &m.v_packed, &m.v_scales, &m.original_shape, device) {
                            let mut inner = block.inner.write().unwrap();
                            inner.k_cache = Some(k_raw);
                            inner.v_cache = Some(v_raw);
                        }
                    }
                }
            }
        });
        Ok(())
    }

    pub async fn prepare_layer_blocks_vl(&self, layer_idx: usize, registry: &KVRegistry, device: &Device, model: &QuantizedQwen3VLTextModel) -> Result<()> {
        use rayon::prelude::*;
        let kv_blocks = &self.self_attn.kv_blocks;
        
        kv_blocks.into_par_iter().for_each(|block| {
            let index = block.inner.read().unwrap().index;
            let meta = {
                let reg = registry.entries.read().unwrap();
                if index < reg.len() {
                    reg[index].bitkv_cache.read().unwrap()[layer_idx].clone()
                } else { None }
            };

            if let Some(m) = meta {
                if block.inner.read().unwrap().k_cache.is_none() {
                    if let Ok(k_raw) = model.decompress_from_bitkv(&m.k_anchors, &m.k_packed, &m.k_scales, &m.original_shape, device) {
                        if let Ok(v_raw) = model.decompress_from_bitkv(&m.v_anchors, &m.v_packed, &m.v_scales, &m.original_shape, device) {
                            let mut inner = block.inner.write().unwrap();
                            inner.k_cache = Some(k_raw);
                            inner.v_cache = Some(v_raw);
                        }
                    }
                }
            }
        });
        Ok(())
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

    pub fn load_kv_cache(&mut self, path: &Path, _device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)]) -> Result<()> {
        let device = self.input_layernorm.weight().device();
        self.self_attn.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments)
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLTextModel {
    pub embed_tokens: Embedding, 
    pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
    pub mmap: Option<Arc<Mmap>>, 
    pub registry: KVRegistry, // [NEW] 모델 전체 공유 목차
    pub baking_only: bool,
    pub is_forced_cpu: bool,
    pub active_session_id: Option<String>,
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
        baking_only: bool, // [NEW]
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

        if actual_hidden_size != config.hidden_size {
            println!("[MODEL-FIX] Hidden Size Mismatch. Config: {}, Actual: {}. Patching...", config.hidden_size, actual_hidden_size);
        }

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;

        let nvml = Nvml::init().ok();
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
        
        // [OPTIMIZATION] If baking only (MLP 0%), we don't need MLP weights, so cost is much lower
        let cost_per_layer = if baking_only { layer_weight_size / 3 } else { layer_weight_size };
        let estimated_activation_buffer = 100_000_000; // Reduced for more aggressive pinning
        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        let mut layer_devices = vec![];
        let mut pinned_layer_count = 0;
        for i in 0..config.num_hidden_layers {
            if current_device.is_cuda() && i < 1 {
                layer_devices.push(current_device.clone());
                pinned_layer_count += 1;
            } else {
                layer_devices.push(Device::Cpu);
            }
        }
        println!("[MODEL] Horizontal Engine Ready: {} GPU slot pinned.", pinned_layer_count);

        // [ORGANIC] Dynamic Threading for Parallel Loading
        let thread_config = crate::utils::resources::get_optimal_thread_config(current_device.is_cpu());
        println!("[MODEL] Organic Loading: Using {} threads ({})", thread_config.thread_count, thread_config.description);
        
        let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_config.thread_count).build()?;
        
        // Ensure we capture the PATCHED config, not the original one
        let final_config = config; 

        let num_layers_to_load = if baking_only { 1 } else { final_config.num_hidden_layers };

        // [NEW] 중앙 목차 생성
        let registry = KVRegistry::new();

        let layers: Result<Vec<_>> = pool.install(|| {
            (0..num_layers_to_load).into_par_iter().zip(layer_devices).map(|(layer_idx, layer_device)| {
                let mut local_cursor = std::io::Cursor::new(mmap);
                let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
                let standard = format!("{base_name}.layers.{layer_idx}");
                let gguf_blk = format!("blk.{layer_idx}");
                let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };
                QuantizedQwen3VLTextDecoderLayer::new(
                    final_config, ct, &mut local_cursor, &prefix, &layer_device, layer_dtype, layer_idx, baking_only, registry.clone()
                )
            }).collect()
        });
        
        let layers = layers?;
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let last_device = layers.last().map(|l| l.device()).unwrap_or(device);
        let norm_dtype = if last_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, &mut reader, norm_prefix, config.rms_norm_eps, last_device, norm_dtype)?;
        
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb, 
            mrope_section, 
            mmap: mmap_handle, 
            registry, // [NEW]
            baking_only, 
            is_forced_cpu, 
            active_session_id: None, 
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
        baking_only: bool, // [NEW]
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        // ... (previous logic)
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

        let nvml = Nvml::init().ok();
        let mut current_device = device.clone();
        
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
        
        let cost_per_layer = if baking_only { layer_weight_size / 3 } else { layer_weight_size };
        let estimated_activation_buffer = 100_000_000; 

        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         let os_reserve = 100_000_000; 
                         safety_floor = os_reserve + kv_reserve + estimated_activation_buffer;
                     }
                 }
             }
        }

        let mut layers = vec![];
        let mut pinned_layer_count = 0;
        let num_layers_to_load = if baking_only { 1 } else { config.num_hidden_layers };

        for layer_idx in 0..num_layers_to_load {
            let mut layer_device = current_device.clone();
            if current_device.is_cuda() && layer_idx < 1 {
                pinned_layer_count += 1;
            } else {
                layer_device = Device::Cpu;
            }

            let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
            let standard = format!("{base_name}.layers.{layer_idx}");
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };

            let layer = QuantizedQwen3VLTextDecoderLayer::new(
                config, ct, reader, &prefix, &layer_device, layer_dtype, layer_idx, baking_only, registry.clone()
            )?;
            layers.push(layer)
        }
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let norm_device = layers.last().map(|l| l.device()).unwrap_or(&current_device);
        let norm_dtype = if norm_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, norm_device, norm_dtype)?;
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section,
            mmap: None,
            registry,
            baking_only,
            is_forced_cpu,
            active_session_id: None,
            pinned_layer_count,
            current_kv_len: 0,
        })
    }

    pub async fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        total_len: usize,
        position_ids_in: Option<&Tensor>,
        _visual_pos_masks: Option<&Tensor>,
        _deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        
        // [STRICT-GPU-COMPUTATION] 연산 장치는 무조건 첫 번째 레이어의 장치(GPU 권장)를 따릅니다.
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let xs_init = if !inputs_embeds.device().same_device(&target_device) {
            inputs_embeds.to_device(&target_device)?
        } else {
            inputs_embeds.clone()
        };
        let mut xs = xs_init.to_dtype(target_dtype)?.contiguous()?;

        let mut start_layer = 0;
        // [RESUME-LOGIC]
        if let Some(sid) = &self.active_session_id {
            let task_dir = crate::utils::paths::get_task_specific_dir(None, sid);
            for l_idx in (0..self.layers.len()).rev() {
                let checkpoint_path = task_dir.join(format!("inference_layer_{}_at_{}.safetensors", l_idx, seqlen_offset));
                if checkpoint_path.exists() {
                    if let Ok(recovered_xs) = crate::utils::tensor_utils::load_tensor(&checkpoint_path, "hidden_states", &target_device) {
                        xs = recovered_xs.to_dtype(target_dtype)?;
                        start_layer = l_idx + 1;
                        break;
                    }
                }
            }
        }

        let position_ids = match position_ids_in {
            Some(ids) => ids.clone(),
            None => Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?,
        };
        
        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        
                        // [HORIZONTAL-PIPELINE-STRATEGY]
                        let num_layers = self.layers.len();
                        let chunk_size = 256;
                        let model_clone = self.clone();
                        
                        println!("[TRACE] Starting Horizontal Pass: Parallel Pipeline (Total Seq: {}, Pinned: {}/{})", 
                            seq_len, self.pinned_layer_count, num_layers);
                
                        // 최초 레이어(0)는 미리 준비되어 있어야 함
                        self.layers[0].prepare_layer_blocks_vl(0, &self.registry, &target_device, &model_clone).await?;

                        for layer_idx in 0..num_layers {
                            if layer_idx < start_layer { continue; }
                            
                            // 1. [BACKGROUND-PREP] 다음 레이어 미리 압축 해제
                            let next_layer = layer_idx + 1;
                            if next_layer < num_layers {
                                let reg_c = self.registry.clone();
                                let dev_c = target_device.clone();
                                let model_c = model_clone.clone();
                                let next_layer_ptr = self.layers[next_layer].clone();
                                
                                tauri::async_runtime::spawn(async move {
                                    // 슬롯 예약 (시각적 피드백용)
                                    let mut s_ids = Vec::new();
                                    for _ in 0..40 {
                                        s_ids.push(crate::models::qwen3vl::generate::SLOT_MANAGER.acquire_read_slot().await);
                                    }
                                    
                                    let _ = next_layer_ptr.prepare_layer_blocks_vl(next_layer, &reg_c, &dev_c, &model_c).await;
                                    
                                    let mut reg = reg_c.entries.write().unwrap();
                                    for (i, entry) in reg.iter_mut().enumerate() {
                                        if i < s_ids.len() { entry.slot_ids[next_layer] = Some(s_ids[i]); }
                                    }
                                });
                            }

                            // 2. [GPU-EXECUTION]
                            if !self.layers[layer_idx].device().same_device(&target_device) {
                                self.layers[layer_idx].to_device(&target_device)?;
                            }

                            // [RECLAIM-SLOTS]
                            let sids_to_release: Vec<usize> = {
                                let mut reg = self.registry.entries.write().unwrap();
                                reg.iter_mut().filter_map(|entry| entry.slot_ids[layer_idx].take()).collect()
                            };
                            for sid in sids_to_release { 
                                crate::models::qwen3vl::generate::SLOT_MANAGER.release_slot(sid).await; 
                            }

                            let mut next_xs = Vec::new();
                            for i in (0..seq_len).step_by(chunk_size) {
                                let take = (seq_len - i).min(chunk_size);
                                let out = self.layers[layer_idx].forward(&xs.narrow(1, i, take)?, &cos.narrow(1, i, take)?, &sin.narrow(1, i, take)?, None, seqlen_offset + i)?;
                                next_xs.push(out);
                            }
                            xs = Tensor::cat(&next_xs, 1)?;

                            if layer_idx > 0 && layer_idx < num_layers - 1 {
                                let _ = self.layers[layer_idx].to_device(&Device::Cpu);
                            }
                        }
                
                        println!("[TRACE] Horizontal Layer Pass Complete.");
                        self.current_kv_len = seqlen_offset + seq_len;
        println!("[TRACE] Final norm.forward starting...");
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) {
            xs = xs.to_device(norm_dev)?;
        }
        let xs = xs.apply(&self.norm)?;
        println!("[TRACE] All forward pass steps complete.");
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn get_kv_len(&self) -> usize {
        // [ROPE-FIX] 물리적 캐시 크기가 아닌 논리적 진행 위치를 반환하여 
        // 페이징 후에도 다음 토큰의 RoPE Offset이 어긋나지 않게 함
        self.current_kv_len
    }

    pub fn compress_to_bitkv(&self, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        // Just use the first layer's logic as it's purely mathematical
        self.layers[0].self_attn.compress_to_bitkv(t)
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        println!("[MEMORY] Hard Resetting Language Model Registry & Layers...");
        
        // 1. Registry (중앙 통제실 TOC) 완전 리셋
        {
            let mut reg = self.registry.entries.write().unwrap();
            for entry in reg.iter_mut() {
                for loc in entry.location.iter_mut() { *loc = KVLocation::SSD; }
                for sid in entry.slot_ids.iter_mut() { *sid = None; }
            }
        }

        // 2. 각 레이어의 물리적 메모리 점유 해제
        for layer in self.layers.iter_mut() {
            layer.drop_kv_storage()?;
        }

        self.current_kv_len = 0;
        println!("[MEMORY] Registry Reset Complete.");
        Ok(())
    }

    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_list.len() {
                layer.self_attn.inject_live_kv(&k_list[i], &v_list[i], k_scale, v_scale)?;
            }
        }
        Ok(())
    }

    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        // Backward compatibility wrapper for old relay logic
        self.inject_live_kv(k_list, v_list, k_scales[0], v_scales[0])
    }

        pub fn inject_live_kv_bitkv(&mut self, k_anchors: &[Tensor], k_packed: &[Tensor], k_scales: &[Tensor], v_anchors: &[Tensor], v_packed: &[Tensor], v_scales: &[Tensor], original_shape: &[usize]) -> Result<()> {
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_anchors.len() {
                // Decompress into 0.6B shape first on GPU
                let k_small = layer.self_attn.decompress_from_bitkv(&k_anchors[i].to_device(&target_device)?, &k_packed[i].to_device(&target_device)?, &k_scales[i].to_device(&target_device)?, original_shape, &target_device)?;
                let v_small = layer.self_attn.decompress_from_bitkv(&v_anchors[i].to_device(&target_device)?, &v_packed[i].to_device(&target_device)?, &v_scales[i].to_device(&target_device)?, original_shape, &target_device)?;
                
                // Align to 2B dimensions (Head/Dim upscale)
                let target_heads = layer.self_attn.num_key_value_heads;
                let target_dim = layer.self_attn.head_dim;
                let (_b, h, _s, d) = k_small.dims4()?;

                let mut k_aligned = if d < target_dim { Tensor::cat(&[&k_small, &k_small], D::Minus1)? } else { k_small };
                let mut v_aligned = if d < target_dim { Tensor::cat(&[&v_small, &v_small], D::Minus1)? } else { v_small };

                if h != target_heads {
                    let mut k_heads = Vec::with_capacity(target_heads);
                    let mut v_heads = Vec::with_capacity(target_heads);
                    for j in 0..target_heads {
                        let src_idx = j % h;
                        k_heads.push(k_aligned.narrow(1, src_idx, 1)?);
                        v_heads.push(v_aligned.narrow(1, src_idx, 1)?);
                    }
                    k_aligned = Tensor::cat(&k_heads, 1)?;
                    v_aligned = Tensor::cat(&v_heads, 1)?;
                }

                layer.self_attn.inject_live_kv_direct(&k_aligned.to_dtype(target_dtype)?, &v_aligned.to_dtype(target_dtype)?)?;
            }
        }
        Ok(())
    }

    fn compress_to_1bit(&self, t: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let t_data = t_f32.flatten_all()?.to_vec1::<f32>()?;
        
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements = t_data.len();
        let num_vectors = total_elements / last_dim;
        
        let packed_size = (total_elements + 7) / 8;
        let mut packed = vec![0u8; packed_size];
        let mut scales = vec![0.0f32; num_vectors];
        
        for v_idx in 0..num_vectors {
            let t_start = v_idx * last_dim;
            let t_vector = &t_data[t_start..t_start + last_dim];
            
            let mut abs_sum = 0.0f32;
            for &val in t_vector { abs_sum += val.abs(); }
            let s = abs_sum / (last_dim as f32);
            scales[v_idx] = s;
            
            for (i, &val) in t_vector.iter().enumerate() {
                if val >= 0.0 {
                    let bit_pos = (t_start + i) % 8;
                    let byte_pos = (t_start + i) / 8;
                    packed[byte_pos] |= 1 << bit_pos;
                }
            }
        }
            
        let packed_tensor = Tensor::from_vec(packed, vec![packed_size], &Device::Cpu)?;
        let scales_tensor = Tensor::from_vec(scales, vec![original_shape[0], original_shape[1], original_shape[2], 1], &Device::Cpu)?;
        
        Ok((packed_tensor, scales_tensor, original_shape))
    }

    fn decompress_from_1bit(&self, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = packed.device();
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements: usize = original_shape.iter().product();
        let mut decoded = vec![0.0f32; total_elements];
        
        for v_idx in 0..(total_elements / last_dim) {
            let s = scales_vec[v_idx];
            let t_start = v_idx * last_dim;
            
            for i in 0..last_dim {
                let global_idx = t_start + i;
                let byte_pos = global_idx / 8;
                let bit_pos = global_idx % 8;
                let is_set = (packed_vec[byte_pos] & (1 << bit_pos)) != 0;
                decoded[global_idx] = if is_set { s } else { -s };
            }
        }
        
        let t = Tensor::from_vec(decoded, original_shape, &Device::Cpu)?;
        Ok(t.to_device(device)?)
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

        // [OPTIMIZATION] Scan once at Model level, then distribute to layers
        let mut fragments = Vec::new();
        let mut attempts = 0;
        let mut last_count = 0;
        let mut stable_ticks = 0;

        println!("[SSD-LOAD] Starting centralized scan for KV cache at {:?}", path);

        loop {
            fragments.clear();
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    
                    // Match pattern: bundle_relay_kv_{offset}.safetensors OR bundle_inference_kv_{offset}.safetensors
                    let is_bundle = fname.starts_with("bundle_");
                    
                    if is_bundle && fname.ends_with(".safetensors") {
                        let offset = if fname == "bundle_relay_kv.safetensors" || fname == "bundle_inference_kv.safetensors" {
                            0
                        } else {
                            fname.strip_suffix(".safetensors")
                                 .and_then(|s| s.split('_').last())
                                 .and_then(|s| s.parse::<usize>().ok())
                                 .unwrap_or(0)
                        };
                        fragments.push((offset, entry.path()));
                    }
                }
            }
            
            // Deduplicate based on offset only (prefer relay if mixed? usually won't mix)
            fragments.sort_by_key(|f| f.0);
            fragments.dedup_by_key(|f| f.0);
            
            let current_count = fragments.len();
            if current_count > 0 {
                if current_count == last_count {
                    stable_ticks += 1;
                } else {
                    stable_ticks = 0;
                }
            }
            last_count = current_count;

            // Wait for stability (at least 5 ticks stable, or 40 blocks stable for 2 ticks, or max attempts)
            if (stable_ticks >= 5 && current_count > 5) || (current_count >= 40 && stable_ticks >= 2) || attempts >= 15 {
                break;
            }

            println!("[SSD-LOAD] Central scan found {} blocks (Attempt {}, Stable: {}/5)", current_count, attempts, stable_ticks);
            std::thread::sleep(std::time::Duration::from_millis(400));
            attempts += 1;
        }

        println!("[SSD-LOAD] Finalized fragment list with {} blocks. Distributing to {} layers.", fragments.len(), self.layers.len());

        // [RELAY-JUMP] 마지막 블록의 오프셋 + 길이를 현재 진행 상태로 설정
        if let Some((last_off, _)) = fragments.last() {
            // 표준 블록 크기 256을 고려하여 전체 길이 계산
            let total_loaded = *last_off + 256; 
            self.current_kv_len = total_loaded;
            println!("[SSD-LOAD] Relay Jump: Model progress synchronized to token index {}.", total_loaded);
        }

        self.layers.iter_mut().try_for_each(|layer| {
            layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, &fragments)
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

    pub fn rebalance_layers(&mut self, device_id: usize, target_idx: usize) -> Result<()> {
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

        let danger_zone = 600_000_000;
        let safe_zone = 1_000_000_000;

        // [ACTIVE-VRAM-MANAGEMENT] 
        // 현재 레이어를 GPU에 올리고 싶은데 공간이 부족하다면, 
        // 이미 다른 레이어들이 점유하고 있는 VRAM 자원(블록 및 무게추) 중 가장 오래된 것들을 밀어냅니다.
        if free_vram < safe_zone {
            let container_size = 2048;
            let current_token_pos = self.get_kv_len();
            let current_container_id = current_token_pos / container_size;

            println!("[REBALANCE] Low VRAM ({}MB). Evicting oldest resources to GPU-pin Layer {}...", 
                free_vram / 1024 / 1024, target_idx);

            // 1단계: 모든 레이어의 '비활성 컨테이너' KV 블록을 RAM으로 밀어내기
            let mut kv_freed = 0;
            for l_idx in 0..self.layers.len() {
                if free_vram > safe_zone { break; }
                for block in &self.layers[l_idx].self_attn.kv_blocks {
                    let mut inner = block.inner.write().unwrap();
                    let block_container_id = inner.offset / container_size;
                    if inner.location == KVLocation::VRAM && block_container_id < current_container_id {
                        if let (Some(k), Some(v)) = (inner.k_cache.take(), inner.v_cache.take()) {
                            let (ka, kp, ks, os) = self.layers[l_idx].self_attn.compress_to_bitkv(&k)?;
                            let (va, vp, vs, _) = self.layers[l_idx].self_attn.compress_to_bitkv(&v)?;
                            inner.bitkv_metadata = Some(BitKVMetadata {
                                k_anchors: ka, k_packed: kp, k_scales: ks,
                                v_anchors: va, v_packed: vp, v_scales: vs, original_shape: os,
                            });
                            inner.location = KVLocation::RAM;
                            kv_freed += 1;
                        }
                    }
                }
            }

            // 2단계: 여전히 부족하다면, 가장 오래된 레이어 무게추(Weights)를 CPU로 오프로드
            let mut weights_freed = 0;
            if free_vram < safe_zone {
                for l_idx in 0..self.layers.len() {
                    // 현재 연산해야 할 레이어(target_idx)는 제외하고, 0번 레이어부터 순차적으로 퇴출
                    if l_idx == target_idx { continue; }
                    if free_vram > safe_zone { break; }
                    
                    if self.layers[l_idx].device().is_cuda() {
                        self.layers[l_idx].to_device(&Device::Cpu)?;
                        weights_freed += 1;
                        
                        // VRAM 상태 즉시 갱신
                        if let Some(nvml_inst) = &nvml {
                            if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                                if let Ok(mem) = dev.memory_info() { free_vram = mem.free; }
                            }
                        }
                    }
                }
            }

            if kv_freed > 0 || weights_freed > 0 {
                println!("[REBALANCE] Eviction Complete: {} KV-blocks, {} Layers moved to CPU/RAM. New Free VRAM: {}MB", 
                    kv_freed, weights_freed, free_vram / 1024 / 1024);
            }
        }
        

        if target_idx >= self.layers.len() { return Ok(()); }
        let layer = &mut self.layers[target_idx];

        if free_vram > 0 && free_vram < danger_zone && layer.device().is_cuda() {
            println!("[REBALANCE] >> [START] Emergency Offload: Layer {} -> CPU (Free: {}MB < Danger: {}MB)", target_idx, free_vram / 1024 / 1024, danger_zone / 1024 / 1024);
            let offload_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                layer.to_device(&Device::Cpu)
            }));
            match offload_res {
                Ok(Ok(_)) => println!("[REBALANCE] << [DONE] Emergency Offload: Layer {} moved to CPU.", target_idx),
                Ok(Err(e)) => println!("[REBALANCE] !! [ERROR] Failed to offload Layer {}: {:?}", target_idx, e),
                Err(_) => println!("[REBALANCE] !! [PANIC] Panic during offload of Layer {}", target_idx),
            }
        } else if free_vram > safe_zone && layer.device().is_cpu() {
            // [LOG] 무게추 크기 계산
            let weight_size_mb = (layer.input_layernorm.weight().elem_count() * layer.input_layernorm.weight().dtype().size_in_bytes()) as f32 / 1024.0 / 1024.0;
            println!("[REBALANCE] >> [START] Targeted Upload: Layer {} -> GPU (Free: {}MB > Safe: {}MB, Weights: ~{:.1}MB)", target_idx, free_vram / 1024 / 1024, safe_zone / 1024 / 1024, weight_size_mb * 5.0); // LN + Attention + MLP(Approx)
            
            let target_device = Device::new_cuda(device_id)?;
            
            let target_device_clone = target_device.clone();
            let move_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                layer.to_device(&target_device_clone)
            }));

            match move_res {
                Ok(Ok(_)) => {
                    println!("[REBALANCE] .. [SYNC] Synchronizing Device for Layer {}", target_idx);
                    let _ = target_device.synchronize();
                    println!("[REBALANCE] << [DONE] Targeted Upload: Layer {} moved to GPU.", target_idx);
                },
                Ok(Err(e)) => {
                    println!("[REBALANCE] !! [ERROR] Failed to move Layer {} to GPU: {:?}", target_idx, e);
                    return Err(e.into());
                },
                Err(p) => {
                    let panic_msg = if let Some(s) = p.downcast_ref::<&str>() { s.to_string() }
                                    else if let Some(s) = p.downcast_ref::<String>() { s.clone() }
                                    else { "Unknown Panic".to_string() };
                    println!("[REBALANCE] !! [PANIC] Critical failure while moving Layer {} to GPU. Panic: {}", target_idx, panic_msg);
                    return Err(anyhow::anyhow!("Panic during rebalance upload: {}", panic_msg));
                }
            }
        } else if layer.device().is_cpu() {
            // [STRICT-GPU-POLICY] CPU 연산은 절대 허용하지 않습니다.
            // 여유 공간이 safe_zone 미만이더라도, 실제 무게추를 올릴 공간이 있다면 즉시 업로드합니다.
            // (이미 위에서 Eviction 로직이 실행되었으므로 최대한의 자리가 확보된 상태입니다.)
            
            let weight_size_mb = (layer.input_layernorm.weight().elem_count() * layer.input_layernorm.weight().dtype().size_in_bytes()) as f32 / 1024.0 / 1024.0;
            let estimated_layer_size = (weight_size_mb * 5.0) as u64 * 1024 * 1024; // Approx LN + Attn + MLP

            if free_vram > estimated_layer_size {
                println!("[REBALANCE] >> [FORCE] Uploading Layer {} to GPU (Free: {}MB, Required: ~{}MB)", 
                    target_idx, free_vram / 1024 / 1024, estimated_layer_size / 1024 / 1024);
                
                let target_device = Device::new_cuda(device_id)?;
                let target_device_clone = target_device.clone();
                let move_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    layer.to_device(&target_device_clone)
                }));

                match move_res {
                    Ok(Ok(_)) => {
                        let _ = target_device.synchronize();
                        println!("[REBALANCE] << [DONE] Layer {} is now on GPU for computation.", target_idx);
                    },
                    Ok(Err(e)) => {
                        println!("[REBALANCE] !! [ERROR] Failed to move Layer {} to GPU: {:?}", target_idx, e);
                        return Err(e.into());
                    },
                    Err(p) => {
                        return Err(anyhow::anyhow!("Panic during forced GPU upload"));
                    }
                }
            } else {
                // 공간이 정말로 부족하다면 잠시 기다리며 다시 한번 비우기를 시도하거나 에러를 냅니다.
                // (CPU 연산을 시키는 것보다 자리가 날 때까지 에러를 내는 것이 사용자님 원칙에 맞습니다.)
                println!("[REBALANCE] !! [CRITICAL] Insufficient VRAM for Layer {}. Free: {}MB", target_idx, free_vram / 1024 / 1024);
                return Err(anyhow::anyhow!("VRAM Out of Memory - Cannot honor Always-GPU policy"));
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
        let start_time = std::time::Instant::now();
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
        println!("[TIMER] RoPE index generation: {:.2}s for {} tokens", start_time.elapsed().as_secs_f32(), seq_len);
        Ok((position_ids, deltas))
    }

    pub fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        // [DEVICE-ALIGNMENT] 임베딩 레이어의 장치로 입력 ID를 즉시 이동 (CUDA 보장)
        let emb_dev = self.language_model.embed_tokens.embeddings().device();
        let input_ids = if !input_ids_in.device().same_device(emb_dev) { 
            input_ids_in.to_device(emb_dev)? 
        } else { 
            input_ids_in.clone() 
        };
        let (b_sz, seq_len) = input_ids.dims2()?;

        // 1. Embedding & Vision Integration
        let start_init = std::time::Instant::now();
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        println!("[TIMER] Embedding forward: {:.2}s", start_init.elapsed().as_secs_f32());
        
        let start_vision = std::time::Instant::now();
        if let Some(pv) = pixel_values { 
            if let Some(thw) = image_grid_thw { 
                let (image_embeds, _) = self.get_vision_features(pv, thw)?; 
                let image_embeds = Tensor::cat(&image_embeds, 0)?; 
                let vision_mask = self.get_placeholder_mask(&input_ids, true)?; 
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; 
            } 
        }
        if pixel_values.is_some() {
            println!("[TIMER] Vision integration: {:.2}s", start_vision.elapsed().as_secs_f32());
        }
        
        // 2. Position IDs calculation (Corrected for Long Context & Vision)
        let start_rope = std::time::Instant::now();
        let (position_ids, _rope_deltas) = if (cache_position_in.is_some() && cache_position_in.unwrap().i(0)?.to_scalar::<u32>()? == 0) || self.rope_deltas.is_none() {
            let (p_ids, deltas) = self.get_rope_index(&input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(deltas);
            (p_ids, self.rope_deltas.as_ref().unwrap().clone())
        } else {
            // Decoding/Incremental Step
            let start = seqlen_offset as u32;
            let p_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
            (p_ids, self.rope_deltas.as_ref().unwrap().clone())
        };
        println!("[TIMER] RoPE index calculation: {:.2}s", start_rope.elapsed().as_secs_f32());
        
        self.language_model.active_session_id = session_id;
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None)?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        let head_dev = self.lm_head.device();
        let head_dtype = if head_dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let hidden_state = if !hidden_state.device().same_device(head_dev) { hidden_state.to_device(head_dev)? } else { hidden_state };
        let hidden_state = if hidden_state.dtype() != head_dtype { hidden_state.to_dtype(head_dtype)? } else { hidden_state };
        Ok(self.lm_head.forward(&hidden_state)?)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; self.text_device = device.clone(); self.vision_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, target_idx: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, target_idx) }
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

    pub async fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        // [DEVICE-ALIGNMENT] 임베딩 레이어 장치와 입력 장치 맞춤
        let emb_dev = self.language_model.embed_tokens.embeddings().device();
        let input_ids = if !input_ids_in.device().same_device(emb_dev) { 
            input_ids_in.to_device(emb_dev)? 
        } else { 
            input_ids_in.clone() 
        };
        
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
        
        self.language_model.active_session_id = session_id;
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None)?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        if let Some(head) = &self.lm_head {
            let hidden_state = if !hidden_state.device().same_device(head.device()) { hidden_state.to_device(head.device())? } else { hidden_state };
            Ok(head.forward(&hidden_state)?)
        } else { Ok(hidden_state) }
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.save_kv_cache(path, clear, offset, kv_name) }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { self.language_model.truncate_kv_cache(len) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.language_model.to_device(device)?; if let Some(head) = &mut self.lm_head { head.to_device(device)?; } self.text_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, target_idx: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, target_idx) }
}

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let weight = QMatMul::from_qtensor(weight)?;
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { Some(t.dequantize(device)?.to_dtype(dtype)?) } else { None };
    Ok(QLinear::new(weight, bias, device.clone()))
}

fn get_sliced_qlinear<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content, 
    reader: &mut R, 
    name: &str, 
    device: &Device, 
    dtype: DType,
    dim: usize,
    start: usize,
    len: usize
) -> Result<QLinear> {
    let qtensor = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight for slicing: {e}"))?;
    // We must dequantize to slice accurately along specific dimensions
    let tensor = qtensor.dequantize(device)?;
    let sliced = tensor.narrow(dim, start, len)?.to_dtype(dtype)?.contiguous()?;
    let weight = QMatMul::Tensor(sliced);
    
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { 
        let b = t.dequantize(device)?;
        // If dim was 0 (output features), we must slice the bias as well
        if dim == 0 {
            Some(b.narrow(0, start, len)?.to_dtype(dtype)?)
        } else {
            Some(b.to_dtype(dtype)?)
        }
    } else { None };
    
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