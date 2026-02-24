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
    pub is_dirty: bool, // SSD 저장이 필요한지 여부
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
            is_dirty: false,
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
        let json = serde_json::to_string_pretty(&*entries)?;
        std::fs::write(path.join("metadata.json"), json)?;
        Ok(())
    }

    pub fn load_from_file(&self, path: &std::path::Path) -> Result<()> {
        let meta_path = path.join("metadata.json");
        if !meta_path.exists() { return Ok(()); }
        let json = std::fs::read_to_string(meta_path)?;
        let mut loaded: Vec<RegistryEntry> = serde_json::from_str(&json)?;
        
        for entry in loaded.iter_mut() {
            if entry.ssd_path.is_some() {
                for loc in entry.location.iter_mut() {
                    *loc = KVLocation::SSD;
                }
            }
        }

        let mut entries = self.entries.write().unwrap();
        for (i, entry) in loaded.into_iter().enumerate() {
            if i < entries.len() {
                entries[i] = entry;
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
    pub k_anchors: Tensor,
    pub k_packed: Tensor,
    pub k_scales: Tensor,
    pub v_anchors: Tensor,
    pub v_packed: Tensor,
    pub v_scales: Tensor,
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
        seqlen_offset: usize, // [NEW]
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
        let mut tokens_to_process = current_chunk_len;
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
                let current_total = seqlen_offset + chunk_offset;
                
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece);
                    inner.v_cache = Some(v_piece);
                }
                
                // Registry entry management only on Layer 0
                if self.layer_idx == 0 {
                    let mut reg = self.registry.entries.write().unwrap();
                    if index < reg.len() {
                        let entry = &mut reg[index];
                        entry.token_start = current_total;
                        entry.token_len = take;
                        entry.is_dirty = true;
                        for loc in entry.location.iter_mut() { *loc = KVLocation::VRAM; }
                        entry.last_accessed = std::time::Instant::now();
                    } else {
                        reg.push(RegistryEntry {
                            location: vec![KVLocation::VRAM; 28],
                            slot_ids: vec![None; 28],
                            token_start: current_total,
                            token_len: take,
                            ssd_path: None,
                            hidden_states_path: vec![None; 28],
                            is_dirty: true,
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

        // [PREFETCH-REMOVED] 예독 로직은 상위 레이어 함수로 이동되었습니다.

        // 4. [LIGHTWEIGHT-HYBRID-ENGINE]
        let q_len = query_states.dim(1)?;
        let total_len = seqlen_offset + q_len;
        let mut running_m = None; 
        let mut running_s = None; 
        let mut running_o = None; 
        let mut global_start_idx = 0;

        let mut vram_ks = Vec::new();
        let mut vram_vs = Vec::new();
        let mut vram_total_len = 0;

        // [PHYSICAL-SYNC-GATE] 연산 시작 전, 이 레이어에 필요한 모든 블록이 로드될 때까지 대기
        let start_gate = std::time::Instant::now();
        loop {
            let mut pending_blocks = Vec::new();
            {
                let reg = self.registry.entries.read().unwrap();
                for block in &self.kv_blocks {
                    let inner = block.inner.read().unwrap();
                    if inner.offset < total_len {
                        let loc = if inner.index < reg.len() { reg[inner.index].location[self.layer_idx] } else { KVLocation::VRAM };
                        // [GATE-FIX] Loading(읽기 진행 중)인 경우만 기다립니다. SSD는 이미 완료된 상태입니다.
                        if loc == KVLocation::Loading {
                            pending_blocks.push((inner.index, inner.offset, reg[inner.index].ssd_path.clone()));
                        }
                    }
                }
            }

            if pending_blocks.is_empty() { break; }
            
            // 10초 이상 대기 시 강제 돌파 (데드락 방지)
            if start_gate.elapsed().as_secs() > 10 {
                println!("[GATE-TIMEOUT] 레이어 {} 연산 강제 시작 (미로드 블록 {}개)", self.layer_idx, pending_blocks.len());
                break;
            }

            // 무엇을 기다리는지 상세 로깅 (1초마다)
            if start_gate.elapsed().as_millis() % 1000 < 10 {
                for (idx, off, path) in &pending_blocks {
                    println!("[GATE-WAIT] 레이어 {} 블록 {} (Offset {}) 로딩 대기 중... 경로: {:?}", 
                        self.layer_idx, idx, off, path.as_ref().map(|p| p.join(format!("l{}.st", self.layer_idx))));
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        for block in &self.kv_blocks {
            let (mut k, mut v, b_len, index, b_off) = {
                let inner = block.inner.read().unwrap();
                (inner.k_cache.clone(), inner.v_cache.clone(), inner.len, inner.index, inner.offset)
            };

            // [STRICT-OFFSET-LIMIT] 현재 유효한 토큰 범위를 넘어서는 빈 블록은 무시합니다.
            if b_off >= total_len {
                global_start_idx += b_len;
                continue;
            }

            // [REGISTRY-LOOKUP] 2D 목차에서 이 레이어의 위치를 정확히 찾음
            let (loc, ssd_path, layer_slot_id) = {
                let reg = self.registry.entries.read().unwrap();
                if index >= reg.len() { (KVLocation::VRAM, None, None) }
                else {
                    let entry = &reg[index];
                    (entry.location[self.layer_idx], entry.ssd_path.clone(), entry.slot_ids[self.layer_idx])
                }
            };

            // [WAIT-IF-LOADING] 데이터가 로드될 때까지 대기
            if loc == KVLocation::Loading {
                let start_wait = std::time::Instant::now();
                loop { 
                    let l = {
                        let reg = self.registry.entries.read().unwrap();
                        reg[index].location[self.layer_idx]
                    };
                    if l != KVLocation::Loading { break; }
                    if start_wait.elapsed().as_secs() > 10 { break; }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }

            // Re-read location after possible wait
            let loc = {
                let reg = self.registry.entries.read().unwrap();
                reg[index].location[self.layer_idx]
            };

            match loc {
                KVLocation::VRAM => {
                    vram_ks.push(k.unwrap_or_else(|| Tensor::zeros((1, self.num_key_value_heads, b_len, self.head_dim), target_dtype, dev).unwrap()));
                    vram_vs.push(v.unwrap_or_else(|| Tensor::zeros((1, self.num_key_value_heads, b_len, self.head_dim), target_dtype, dev).unwrap()));
                    vram_total_len += b_len;
                    global_start_idx += b_len;
                    continue;
                }
                KVLocation::RAM | KVLocation::SSD => {
                    let mut k_active = None;
                    let mut v_active = None;

                    // 1. [TEMPORAL-REUSE] RAM에 이미 디코딩된 캐시가 있다면 사용
                    {
                        let inner = block.inner.read().unwrap();
                        if let (Some(kc), Some(vc)) = (&inner.k_cache, &inner.v_cache) {
                            k_active = Some(kc.clone());
                            v_active = Some(vc.clone());
                        }
                    }

                    // 2. [SLOT-DIRECT-ACCESS] RAM 슬롯에서 직접 가져오기
                    if k_active.is_none() && loc == KVLocation::RAM {
                        if let Some(sid) = layer_slot_id {
                            let slot = &crate::models::qwen3vl::generate::SLOT_MANAGER.slots[sid];
                            if let (Ok(k_guard), Ok(v_guard)) = (slot.k_layers[self.layer_idx].try_lock(), slot.v_layers[self.layer_idx].try_lock()) {
                                if let (Some(tk), Some(tv)) = (k_guard.as_ref(), v_guard.as_ref()) {
                                    k_active = Some(tk.to_device(dev)?.to_dtype(target_dtype)?);
                                    v_active = Some(tv.to_device(dev)?.to_dtype(target_dtype)?);
                                }
                            }
                        }
                    }

                    // 3. [JUST-IN-TIME-LOAD] SSD에서 직접 읽어서 연산 후 버리기
                    if k_active.is_none() {
                        let path = ssd_path.clone().unwrap_or_default();
                        let filename = format!("l{}.st", self.layer_idx);
                        let full_path = path.join("inference").join(format!("b{}", b_off)).join(&filename);
                        let fallback_path = path.join("reference").join(format!("b{}", b_off)).join("l0.st");
                        
                        let act_path = if full_path.is_file() { 
                            Some(full_path) 
                        } else if fallback_path.is_file() {
                            Some(fallback_path)
                        } else {
                            None
                        };
                        
                        if let Some(act_p) = act_path {
                            if let Ok(content) = std::fs::read(&act_p) {
                                if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                    let block_offset = index * 256;
                                    let is_relay = act_p.file_name().map(|n| n == "l0.st").unwrap_or(false);
                                    let prefix = if is_relay { format!("b{}_l0_", block_offset) } else { format!("b{}_l{}_", block_offset, self.layer_idx) };
                                    
                                    let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                    if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (get_t("k_anchors"), get_t("k_packed"), get_t("k_scales"), get_t("v_anchors"), get_t("v_packed"), get_t("v_scales")) {
                                        let bytes_to_f32 = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
                                        let expected_h = self.num_key_value_heads;
                                        let expected_d = self.head_dim;
                                        let anchor_count = ka.shape()[2];
                                        
                                        let meta = BitKVMetadata {
                                            k_anchors: Tensor::from_vec(bytes_to_f32(ka.data()), (1, expected_h, anchor_count, expected_d), &Device::Cpu)?,
                                            k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu)?,
                                            k_scales: Tensor::from_vec(bytes_to_f32(ks.data()), (1, expected_h, b_len, 1), &Device::Cpu)?,
                                            v_anchors: Tensor::from_vec(bytes_to_f32(va.data()), (1, expected_h, anchor_count, expected_d), &Device::Cpu)?,
                                            v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu)?,
                                            v_scales: Tensor::from_vec(bytes_to_f32(vs.data()), (1, expected_h, b_len, 1), &Device::Cpu)?,
                                            original_shape: vec![1, expected_h, b_len, expected_d],
                                        };
                                        
                                        k_active = Some(self.decompress_from_bitkv(&meta.k_anchors, &meta.k_packed, &meta.k_scales, &meta.original_shape, dev)?);
                                        v_active = Some(self.decompress_from_bitkv(&meta.v_anchors, &meta.v_packed, &meta.v_scales, &meta.original_shape, dev)?);
                                    }
                                }
                            }
                        }
                    }

                    if let (Some(mut k), Some(mut v)) = (k_active, v_active) {
                        // [COMPUTE-PARTIAL-ATTENTION]
                        if self.num_kv_groups > 1 {
                            let (b, h, s, d) = k.dims4()?;
                            k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                            v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                        }

                        let mut attn_weights = query_states.matmul(&k.transpose(2, 3)?)?
                            .broadcast_mul(&Tensor::new(&[self.scaling as f32], dev)?.to_dtype(target_dtype)?)?;

                        if let Some(mask) = attention_mask {
                            let mask_len = mask.dim(D::Minus1)?;
                            let current_block_start = global_start_idx - b_len;
                            if current_block_start + b_len <= mask_len {
                                let sub_mask = mask.narrow(D::Minus1, current_block_start, b_len)?.to_dtype(target_dtype)?;
                                attn_weights = attn_weights.broadcast_add(&sub_mask)?;
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
                        
                        // [EVICT-IMMEDIATELY] 연산 끝난 파편은 메모리에서 즉시 삭제
                        drop(k); drop(v);
                    }
                    
                    global_start_idx += b_len;
                }
                _ => { global_start_idx += b_len; continue; }
            }
        }

        // 5. [BATCH-VRAM] Fast single-pass for all VRAM blocks
        if !vram_ks.is_empty() {
            let mut k = Tensor::cat(&vram_ks, 2)?;
            let mut v = Tensor::cat(&vram_vs, 2)?;
            
            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            let mut attn_weights = query_states.matmul(&k.transpose(2, 3)?)?
                .broadcast_mul(&Tensor::new(&[self.scaling as f32], dev)?.to_dtype(target_dtype)?)?;

            if let Some(mask) = attention_mask {
                let mask_len = mask.dim(D::Minus1)?;
                let vram_start = global_start_idx - vram_total_len;
                if vram_start + vram_total_len <= mask_len {
                    let sub_mask = mask.narrow(D::Minus1, vram_start, vram_total_len)?.to_dtype(target_dtype)?;
                    attn_weights = attn_weights.broadcast_add(&sub_mask)?;
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
        let (b_sz, n_h, q_len, d_h) = attn_output.dims4()?;
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((b_sz, q_len, n_h * d_h))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        // [OFFLOAD-WATCH] Manage VRAM residency
        let current_kv_total = self.get_kv_len();
        if current_kv_total > 8192 {
            // [FIX] Pre-calculate take amount to avoid simultaneous borrow
            let num_to_offload = self.kv_blocks.len().saturating_sub(2);
            for block in self.kv_blocks.iter_mut().take(num_to_offload) {
                let (index, current_inner_loc) = {
                    let inner = block.inner.read().unwrap();
                    (inner.index, inner.location)
                };

                // [CRITICAL] Only purge if the registry confirms it's backed up to SSD or RAM.
                // If registry still says VRAM, this block is "dirty" and must stay in VRAM until prefill_only saves it.
                let reg_loc = {
                    let reg = self.registry.entries.read().unwrap();
                    if index < reg.len() { reg[index].location[self.layer_idx] } else { KVLocation::VRAM }
                };

                if reg_loc != KVLocation::VRAM {
                    let mut inner = block.inner.write().unwrap();
                    if inner.location == KVLocation::VRAM {
                        println!("[OFFLOAD] Purging backed-up block {} from VRAM to save space.", index);
                        inner.location = reg_loc; // Sync with registry (usually SSD)
                        inner.k_cache = None;
                        inner.v_cache = None;
                    }
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
        // For now, using a CPU-based multi-core decompression via rayon as a robust baseline.
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let anchors_vec = anchors.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        
        let last_dim = original_shape[original_shape.len() - 1];
        let seq_len = original_shape[original_shape.len() - 2];
        let _num_heads = original_shape[1];
        let _batch_size = original_shape[0];
        
        let total_elements: usize = original_shape.iter().product();
        let mut decoded = vec![0.0f32; total_elements];
        
        let head_tokens = seq_len * last_dim;
        let anchor_count = (0..seq_len).filter(|&i| i < 4 || i % 8 == 0).count();

        use rayon::prelude::*;
        decoded.par_chunks_mut(head_tokens).enumerate().for_each(|(bh_idx, head_out)| {
            let bh_offset = bh_idx * head_tokens;
            let anchor_offset = bh_idx * anchor_count * last_dim;
            
            for s_idx in 0..seq_len {
                let scale = scales_vec[bh_idx * seq_len + s_idx];
                let token_out = &mut head_out[s_idx * last_dim .. (s_idx + 1) * last_dim];
                
                // 1. Bit-to-Sign restoration
                for d_idx in 0..last_dim {
                    let global_bit_idx = bh_offset + s_idx * last_dim + d_idx;
                    let is_set = (packed_vec[global_bit_idx / 8] & (1 << (global_bit_idx % 8))) != 0;
                    token_out[d_idx] = if is_set { scale } else { -scale };
                }
                
                // 2. Anchor Refinement
                if s_idx < 4 || s_idx % 8 == 0 {
                    let a_pos = if s_idx < 4 { s_idx } else { 4 + (s_idx - 4) / 8 };
                    let anchor_data = &anchors_vec[anchor_offset + a_pos * last_dim .. anchor_offset + (a_pos + 1) * last_dim];
                    token_out.copy_from_slice(anchor_data);
                }
            }
        });

        let t = Tensor::from_vec(decoded, original_shape, &Device::Cpu)?;
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

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.iter().map(|b| b.inner.read().unwrap().len).sum()
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        self.clear_kv_cache();
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _offset: usize, _kv_name: Option<&str>) -> Result<()> {
        // [SSD-ALIGNMENT] 베이킹 시스템과 동일한 b0/lX.st 구조로 저장
        let block_dir = path.join("b0");
        if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
        
        // [INFERENCE-STORAGE] 추론 시에는 각 레이어의 지능이 다르므로 본인의 인덱스(self.layer_idx)를 사용합니다.
        let structured_path = block_dir.join(format!("l{}.st", self.layer_idx));
        
        let mut map = HashMap::new();
        let prefix = format!("b0_l{}_", self.layer_idx);

        // [COLLECT-ALL] 위치와 상관없이 모든 유효한 데이터를 수집
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
            
            // [COMPRESS] 베이킹과 동일한 BitKV 압축 수행
            let (k_anchors, k_packed, k_scales, k_shape) = self.compress_to_bitkv(&k)?;
            let (v_anchors, v_packed, v_scales, _) = self.compress_to_bitkv(&v)?;
            
            map.insert(format!("{}k_anchors", prefix), k_anchors);
            map.insert(format!("{}k_packed", prefix), k_packed);
            map.insert(format!("{}k_scales", prefix), k_scales);
            map.insert(format!("{}v_anchors", prefix), v_anchors);
            map.insert(format!("{}v_packed", prefix), v_packed);
            map.insert(format!("{}v_scales", prefix), v_scales);
            map.insert(format!("{}k_shape", prefix), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect(), (k_shape.len(),), &Device::Cpu)?);
            
            candle_core::safetensors::save(&map, &structured_path)?;
            println!("[SSD-SAVE] Layer {} backed up to {:?}", self.layer_idx, structured_path);
            
            // [REGISTRY-UPDATE] 저장된 위치를 모델 전역 목차(Registry)에 업데이트
            if let Ok(mut reg) = self.registry.entries.write() {
                if reg.is_empty() { 
                    let mut entry = crate::models::qwen3vl::quantized_model::RegistryEntry::new(0, k.dim(2)?, 28);
                    entry.ssd_path = Some(path.to_path_buf());
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    reg.push(entry);
                } else {
                    let entry = &mut reg[0];
                    entry.ssd_path = Some(path.to_path_buf());
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    entry.token_len = k.dim(2)?;
                }
            }
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
                    is_dirty: false, 
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
        
        let cost_per_layer = if baking_only { layer_weight_size / 3 } else { layer_weight_size };
        let estimated_activation_buffer = 200_000_000;
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

        let mut layer_devices = vec![];
        let mut pinned_layer_count = 0;
        
        for _ in 0..config.num_hidden_layers {
            if current_device.is_cuda() && is_vram_checked {
                 let buffer_factor = 1.2; 
                 // [FIX] 추론 모드(baking_only=false)일 때는 모든 레이어를 GPU에 배치하도록 강제합니다.
                 // 0.6B 모델은 약 1.2GB로, 10k 토큰 KV Cache와 함께 있어도 4GB 미만에서 상주 가능합니다.
                 if !baking_only || simulated_free_vram > ( (cost_per_layer as f64 * buffer_factor) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(cost_per_layer);
                     layer_devices.push(current_device.clone());
                     pinned_layer_count += 1;
                 } else {
                     layer_devices.push(Device::Cpu);
                 }
            } else {
                layer_devices.push(current_device.clone());
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
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), mmap: mmap_handle, registry, baking_only, is_forced_cpu, active_session_id: None, active_kv_name: None, pinned_layer_count, current_kv_len: 0 })
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
        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         safety_floor = 50_000_000 + kv_reserve + 50_000_000;
                     }
                 }
             }
        }

        let mut layers = vec![];
        let mut pinned_layer_count = 0;
        let num_layers_to_load = if baking_only { 1 } else { config.num_hidden_layers };

        for layer_idx in 0..num_layers_to_load {
            let mut layer_device = current_device.clone();
            if current_device.is_cuda() && is_vram_checked {
                 let buffer_factor = 1.1; 
                 // [FIX] 추론 모드일 때는 모든 레이어를 GPU에 배치하도록 강제합니다.
                 if !baking_only || simulated_free_vram > ( (cost_per_layer as f64 * buffer_factor) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(cost_per_layer);
                     pinned_layer_count += 1;
                 } else { layer_device = Device::Cpu; }
            }
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { format!("{base_name}.layers.{layer_idx}") };
            layers.push(QuantizedQwen3VLTextDecoderLayer::new(config, ct, reader, &prefix, &layer_device, if layer_device.is_cpu() { DType::F32 } else { dtype }, layer_idx, baking_only, registry.clone())?);
        }
        
        let norm_prefix = if ct.tensor_infos.contains_key("output_norm.weight") { "output_norm" } else { &format!("{base_name}.norm") };
        let norm_device = layers.last().map(|l| l.device()).unwrap_or(&current_device);
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, norm_device, if norm_device.is_cpu() { DType::F32 } else { dtype })?;
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), mmap: None, registry, baking_only, is_forced_cpu, active_session_id: None, active_kv_name: None, pinned_layer_count, current_kv_len: 0 })
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
    ) -> Result<Vec<Tensor>> {
        let mut results = Vec::with_capacity(chunk_offsets.len());
        let chunk_size = 256;
        let current_seq_len = xs.dim(1)?;
        // [STABILITY] 텐서 장치가 아닌, 레이어에 고정된 싱글톤 장치 참조 사용
        let target_device = self.layers[layer_idx].device().clone();
        let sid_opt = self.active_session_id.clone();

        for (chunk_idx, &i) in chunk_offsets.iter().enumerate() {
            let take = (current_seq_len - i).min(chunk_size);
            
            // [SLIDING-WINDOW-PREFETCH] 현재 청크 연산 중에 다음 레이어들의 대응하는 청크를 미리 로드
            // 레이어 시작 시(chunk_idx == 0)에는 초기 윈도우(4개)를 한꺼번에 예독하고, 그 후에는 하나씩 전진하며 예독합니다.
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
                                    tauri::async_runtime::spawn(async move {
                                        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_load_worker};
                                        let sid = SLOT_MANAGER.acquire_read_slot().await;
                                        if let Ok(tx) = get_load_worker().await {
                                            let _ = tx.send(SlotTask::Load(LoadTask { slot_id: sid, path, layer_idx: target_layer, kv_name: None, shared_block, registry: reg_clone })).await;
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
            let cos_chunk = cos.narrow(1, i, take)?;
            let sin_chunk = sin.narrow(1, i, take)?;

            let out = self.layers[layer_idx].forward(&xs_chunk, &cos_chunk, &sin_chunk, None, seqlen_offset + i)?;
            results.push(out);

            // [IMMEDIATE-EVACUATION] 패키징: 청크 하나 끝날 때마다 즉시 대피 (덮어쓰기 방지 포함)
            // [FIX] Baking 모드이거나 세션 ID가 명시적으로 있을 때만 대피를 수행합니다.
            if (self.baking_only || sid_opt.is_some()) && sid_opt.is_some() {
                let session_clone = sid_opt.clone().unwrap();
                let _ = self.evacuate_chunk_kv_to_cpu(layer_idx, &session_clone, seqlen_offset + i, take).await;
            }
            
            if target_device.is_cuda() { let _ = target_device.synchronize(); }
        }
        Ok(results)
    }

    /// [CHUNK-LEVEL-STRICT] 방금 연산된 특정 청크의 KV 데이터만 CPU로 던지고 VRAM에서 지웁니다.
    async fn evacuate_chunk_kv_to_cpu(&mut self, layer_idx: usize, session_id: &str, chunk_off: usize, len: usize) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        let mut dumps_to_send = Vec::new();
        
        for block in &mut self.layers[layer_idx].self_attn.kv_blocks {
            let mut inner = block.inner.write().unwrap();
            // 해당 오프셋을 포함하는 블록 찾기
            if inner.offset <= chunk_off && inner.offset + inner.len > chunk_off {
                // [OVERWRITE-PROTECTION] 이미 SSD에 안전하게 구워졌다면 중복 저장하지 않습니다.
                let is_already_on_ssd = {
                    let reg = self.registry.entries.read().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[layer_idx] == KVLocation::SSD
                    } else { false }
                };

                if is_already_on_ssd {
                    // SSD에 이미 있으니 VRAM만 비웁니다.
                    inner.k_cache = None;
                    inner.v_cache = None;
                    inner.location = KVLocation::SSD;
                    continue;
                }

                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    
                    dumps_to_send.push((LayerKVDump { 
                        layer_idx, 
                        k_tensor: k_cpu, 
                        v_tensor: v_cpu 
                    }, inner.offset));
                    
                    inner.k_cache = None;
                    inner.v_cache = None;
                    inner.location = KVLocation::RAM; 
                }
            }
        }

        if !dumps_to_send.is_empty() {
            if let Some(tx) = BAKE_TX.get() {
                // [PHYSICAL-ISOLATION] session_id 아래에 kv_name 폴더를 별도로 생성합니다.
                let mut base_path = crate::utils::paths::get_kv_dir(None).join(session_id);
                
                // 엔진의 현재 모드에 따라 서브 폴더 결정
                let sub_folder = if self.baking_only { "reference" } else { "inference" };
                let target_path = base_path.join(sub_folder);
                
                if !target_path.exists() { let _ = fs::create_dir_all(&target_path); }
                let rr = Some(self.registry.clone());
                let mode = self.baking_only;

                for (dump, off) in dumps_to_send {
                    let sid = SLOT_MANAGER.acquire_write_slot(len).await;
                    let block_dir = target_path.join(format!("b{}", off));
                    if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }
                    
                    // [REGISTRY-UPDATE] Registry에 최종 블록 디렉토리 경로 저장
                    {
                        let mut reg = self.registry.entries.write().unwrap();
                        let b_idx = off / 256;
                        if b_idx < reg.len() {
                            reg[b_idx].ssd_path = Some(block_dir.clone());
                        }
                    }

                    let _ = tx.send(SlotTask::Bake(BakeTask {
                        slot_id: sid,
                        task_dir: block_dir, // 최종 블록 폴더 전달
                        kv_name: Some(sub_folder.to_string()),
                        offset: off,
                        layers: vec![dump],
                        is_relay_baking: mode,
                        block_idx: Some(off / 256),
                        registry: rr.clone(),
                    })).await;
                }
            }
        }
        Ok(())
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
    ) -> Result<Tensor> {
        let start_layer_time = std::time::Instant::now();
        let input_token_count = xs.dim(1).unwrap_or(0);
        println!("[ENGINE-TRACE] >> Start Layer {} | Input Tokens: {}", layer_idx, input_token_count);

        // [STEP 1] Load Weights to GPU (싱글톤 장치 사용)
        let target_device = crate::utils::get_cuda_device(0); 
        if !self.layers[layer_idx].device().same_device(&target_device) {
            self.layers[layer_idx].to_device(&target_device)?;
            let _ = target_device.synchronize(); 
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
                    if let Ok(content) = std::fs::read(&act_p) {
                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                            let is_relay_file = act_p.file_name().map(|n| n == "l0.st").unwrap_or(false);
                            let prefix = if is_relay_file { format!("b{}_l0_", block_offset) } else { format!("b{}_l{}_", block_offset, l_idx) };
                            let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                            
                            if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (get_t("k_anchors"), get_t("k_packed"), get_t("k_scales"), get_t("v_anchors"), get_t("v_packed"), get_t("v_scales")) {
                                let bytes_to_f32 = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
                                let expected_h = self.layers[layer_idx].self_attn.num_key_value_heads;
                                let expected_d = self.layers[layer_idx].self_attn.head_dim;
                                let anchor_count = ka.shape()[2];
                                
                                let metadata = BitKVMetadata {
                                    k_anchors: Tensor::from_vec(bytes_to_f32(ka.data()), (1, expected_h, anchor_count, expected_d), &Device::Cpu)?,
                                    k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu)?,
                                    k_scales: Tensor::from_vec(bytes_to_f32(ks.data()), (1, expected_h, 256, 1), &Device::Cpu)?,
                                    v_anchors: Tensor::from_vec(bytes_to_f32(va.data()), (1, expected_h, anchor_count, expected_d), &Device::Cpu)?,
                                    v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu)?,
                                    v_scales: Tensor::from_vec(bytes_to_f32(vs.data()), (1, expected_h, 256, 1), &Device::Cpu)?,
                                    original_shape: vec![1, expected_h, 256, expected_d],
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
        let next_xs_all = self.process_chunks_iterative(layer_idx, &chunk_offsets, &xs, cos, sin, seqlen_offset).await?;
        let mut next_xs = Tensor::cat(&next_xs_all, 1)?;

        if let (Some(embed), Some(mask)) = (deepstack_embed, visual_mask) {
            next_xs = mask_index_add(&next_xs.squeeze(0)?, &mask.squeeze(0)?, embed)?.unsqueeze(0)?;
        }

        // [STEP 4] 실시간 VRAM 해제 및 자원 반납 (추론 시에는 VRAM 고정)
        if target_device.is_cuda() { let _ = target_device.synchronize(); }
        
        // [STRICT-INFERENCE-LOCK] 추론 모드(baking_only=false)일 때는 절대 VRAM을 비우지 않습니다.
        if self.baking_only {
            // 1. KV 캐시 즉시 대피 및 VRAM 삭제
            if let Some(sid) = self.active_session_id.clone() {
                let _ = self.evacuate_layer_kv_to_cpu(layer_idx, &sid, seqlen_offset, input_token_count);
            }

            // 2. 레이어 무게추 즉시 CPU로 반납 및 동기화 (VRAM 즉시 해방 강제)
            if target_device.is_cuda() {
                self.layers[layer_idx].to_device(&Device::Cpu)?;
                let _ = target_device.synchronize(); 
            }
            println!("[ENGINE-TRACE] << End Layer {} | VRAM Evicted & Synchronized (Baking Mode).", layer_idx);
        } else {
            // 추론 모드에서는 VRAM에 그대로 둡니다.
            // println!("[ENGINE-TRACE] << End Layer {} | Weights & KV kept in VRAM (Inference Mode).", layer_idx);
        }
        
        Ok(next_xs)
    }

    /// [NEW-STRICT] 특정 레이어의 활성 KV 데이터를 CPU로 복사하고 VRAM에서 즉시 제거
    fn evacuate_layer_kv_to_cpu(&mut self, layer_idx: usize, session_id: &str, start_off: usize, len: usize) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        let mut dumps_to_send = Vec::new();
        let block_start = (start_off / 256) * 256;
        
        // 현재 레이어의 KV 블록들 중 이번 연산에 참여한 블록들만 추출
        for block in &mut self.layers[layer_idx].self_attn.kv_blocks {
            let mut inner = block.inner.write().unwrap();
            if inner.offset >= block_start && inner.location == KVLocation::VRAM {
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    
                    dumps_to_send.push((LayerKVDump { 
                        layer_idx, 
                        k_tensor: k_cpu, 
                        v_tensor: v_cpu 
                    }, inner.offset));
                    
                    // [VRAM-소각] 복사했으니 즉시 삭제
                    inner.k_cache = None;
                    inner.v_cache = None;
                    inner.location = KVLocation::RAM; 
                }
            }
        }

        // 백그라운드 작업자에게 SSD 저장 위임
        if !dumps_to_send.is_empty() {
            if let Some(tx) = BAKE_TX.get() {
                let path = crate::utils::paths::get_kv_dir(None).join(session_id);
                if !path.exists() { let _ = fs::create_dir_all(&path); }
                let rr = Some(self.registry.clone());
                let mode = self.baking_only;

                for (dump, off) in dumps_to_send {
                    let sid = tokio::runtime::Handle::current().block_on(async {
                        SLOT_MANAGER.acquire_write_slot(len).await
                    });
                    
                    let sub_folder = if mode { "reference" } else { "inference" };
                    let block_dir = path.join(sub_folder).join(format!("b{}", off));
                    if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }

                    // [REGISTRY-UPDATE]
                    {
                        let mut reg_w = self.registry.entries.write().unwrap();
                        let b_idx = off / 256;
                        if b_idx < reg_w.len() {
                            reg_w[b_idx].ssd_path = Some(block_dir.clone());
                        }
                    }

                    let _ = tx.blocking_send(SlotTask::Bake(BakeTask {
                        slot_id: sid,
                        task_dir: block_dir, // 최종 블록 폴더 직접 전달
                        kv_name: Some(sub_folder.to_string()),
                        offset: off,
                        layers: vec![dump],
                        is_relay_baking: mode,
                        block_idx: Some(off / 256),
                        registry: rr.clone(),
                    }));
                }
            }
        }
        Ok(())
    }

    pub async fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        total_len: usize,
        position_ids_in: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let target_device = self.layers[0].device().clone();
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mut xs = inputs_embeds.to_device(&target_device)?.to_dtype(target_dtype)?.contiguous()?;

        let position_ids = match position_ids_in {
            Some(ids) => ids.clone(),
            None => Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?,
        };
        
        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        
        // [FAST-PATH] 0.6B 모델은 수직적(전체) 추론을 사용하여 속도를 극대화합니다.
        // 각 레이어의 무게추는 이미 메모리에 있으며, KV 캐시만 필요시 SSD에서 읽어옵니다.
        let total_layers = self.layers.len();
        for layer_idx in 0..total_layers {
            if layer_idx % 7 == 0 || layer_idx == total_layers - 1 {
                println!("[ENGINE] Running Layer {}/{}", layer_idx + 1, total_layers);
            }
            let deepstack_embed = deepstack_visual_embeds.as_ref().and_then(|v| v.get(layer_idx));
            xs = self.process_single_layer(layer_idx, xs, &cos, &sin, seqlen_offset, deepstack_embed, visual_pos_masks).await?;
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
    }

    pub fn get_kv_len(&self) -> usize {
        self.current_kv_len
    }

    pub fn compress_to_bitkv(&self, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        // Just use the first layer's logic as it's purely mathematical
        self.layers[0].self_attn.compress_to_bitkv(t)
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.drop_kv_storage()?;
        }
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
                // Decompress directly into 0.6B shape
                let k_final = layer.self_attn.decompress_from_bitkv(&k_anchors[i].to_device(&target_device)?, &k_packed[i].to_device(&target_device)?, &k_scales[i].to_device(&target_device)?, original_shape, &target_device)?;
                let v_final = layer.self_attn.decompress_from_bitkv(&v_anchors[i].to_device(&target_device)?, &v_packed[i].to_device(&target_device)?, &v_scales[i].to_device(&target_device)?, original_shape, &target_device)?;
                
                layer.self_attn.inject_live_kv_direct(&k_final.to_dtype(target_dtype)?, &v_final.to_dtype(target_dtype)?)?;
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

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        if path.exists() || !fragments.is_empty() {
            // [ROPE-SYNC] 모델의 논리적 위치 포인터를 로드된 길이에 맞게 업데이트합니다.
            // 이를 통해 Prefill 없이 즉시 다음 토큰부터 추론을 시작할 수 있습니다.
            self.current_kv_len = current_kv_len;
            
            self.layers.iter_mut().try_for_each(|layer| {
                layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len)
            })
        } else {
            Ok(())
        }
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
        
        // [FIX] 추론 모드(baking_only=false)일 때는 리밸런싱을 건너뛰고 기존 장치 구성을 유지합니다.
        // 이는 0.6B 28레이어가 VRAM에 안정적으로 상주하도록 강제하여 속도를 높이고 OOM을 방지합니다.
        if !self.baking_only {
            return Ok(());
        }
        
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

    pub async fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        // [OPTION A] 실시간 VRAM 재배치 활성화 (현재 위치 정보 포함)
        let _ = self.rebalance_layers(0, seqlen_offset, total_len);

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
        let (position_ids, rope_deltas) = if (cache_position_in.is_some() && cache_position_in.unwrap().i(0)?.to_scalar::<u32>()? == 0) || self.rope_deltas.is_none() {
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
        
        self.language_model.active_session_id = session_id;
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None).await?;
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
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len) }
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

    pub async fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        // [OPTION A] 실시간 VRAM 재배치 활성화 (현재 위치 정보 포함)
        let _ = self.rebalance_layers(0, seqlen_offset, total_len);

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
        
        self.language_model.active_session_id = session_id;
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None).await?;
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
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.language_model.to_device(device)?; if let Some(head) = &mut self.lm_head { head.to_device(device)?; } self.text_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, offset, total_len) }
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