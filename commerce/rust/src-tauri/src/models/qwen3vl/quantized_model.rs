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
        prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
    },
};
use crate::models::qwen3vl::generate::ACTIVE_BAKE_TASKS;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

// [NEW] 중앙 집중식 KV 목차의 각 항목 (별도 고정 슬롯 관리용)
#[derive(Clone)]
pub struct RegistryEntry {
    pub location: KVLocation,
    pub token_start: usize,
    pub token_len: usize,
    pub slot_id: Option<usize>, 
    pub ssd_path: Option<std::path::PathBuf>,
}

// [NEW] 모델 전체가 공유하는 KV 목차 (Table of Contents)
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
                reg[index].location
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
        let mut appended = false;
        if let Some(last_block) = self.kv_blocks.last_mut() {
            let mut inner = last_block.inner.write().unwrap();
            if inner.location == KVLocation::VRAM && inner.len + current_chunk_len <= 1024 {
                if let (Some(prev_k), Some(prev_v)) = (inner.k_cache.take(), inner.v_cache.take()) {
                    inner.k_cache = Some(Tensor::cat(&[prev_k, key_states.clone()], 2)?.contiguous()?);
                    inner.v_cache = Some(Tensor::cat(&[prev_v, value_states.clone()], 2)?.contiguous()?);
                    inner.len += current_chunk_len;
                    appended = true;
                }
            }
        }

        if !appended {
            let index = self.kv_blocks.len();
            let current_total = self.get_kv_len();
            let new_block = KVBlock::new(KVLocation::VRAM, index, current_chunk_len, current_total);
            {
                let mut inner = new_block.inner.write().unwrap();
                inner.k_cache = Some(key_states);
                inner.v_cache = Some(value_states);
            }
            
            // [REGISTRY-REGISTER] Immediately register this block in the central TOC
            // This ensures Layer 1-27 will find the entry even if Layer 0 just created it.
            {
                let mut reg = self.registry.entries.write().unwrap();
                if reg.len() <= index {
                    reg.push(RegistryEntry {
                        location: KVLocation::VRAM,
                        token_start: current_total,
                        token_len: current_chunk_len,
                        slot_id: None,
                        ssd_path: None,
                    });
                }
            }
            self.kv_blocks.push(new_block);
        }

        // [PREFETCH-TRIGGER] Only trigger on Layer 0 to avoid 28x redundant SSD requests
        if self.layer_idx == 0 {
            let blocks_to_prefetch: Vec<_> = self.kv_blocks.iter()
                .filter(|b| {
                    let index = b.inner.read().unwrap().index;
                    let reg = self.registry.entries.read().unwrap();
                    reg[index].location == KVLocation::SSD
                })
                .cloned()
                .collect();

            for block in blocks_to_prefetch {
                let index = block.inner.read().unwrap().index;
                let path = {
                    let reg = self.registry.entries.read().unwrap();
                    reg[index].ssd_path.clone().unwrap_or_default()
                };
                
                // Mark as loading in central registry
                {
                    let mut reg = self.registry.entries.write().unwrap();
                    reg[index].location = KVLocation::Loading;
                }
                
                let shared_block = block.clone();
                let registry_clone = self.registry.clone();
                tauri::async_runtime::spawn(async move {
                    use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, LoadTask, get_worker_channel};
                    let slot_id = SLOT_MANAGER.acquire_read_slot().await;
                    if let Ok(tx) = get_worker_channel().await {
                        let _ = tx.send(SlotTask::Load(LoadTask {
                            slot_id,
                            path,
                            layer_idx: 0,
                            kv_name: None,
                            shared_block,
                            registry: registry_clone,
                        })).await;
                    }
                });
            }
        }

        // 4. [LIGHTWEIGHT-HYBRID-ENGINE]
        let mut running_m = None; 
        let mut running_s = None; 
        let mut running_o = None; 
        let mut global_start_idx = 0;

        let mut vram_ks = Vec::new();
        let mut vram_vs = Vec::new();
        let mut vram_total_len = 0;

        for block in &self.kv_blocks {
            let (mut k, mut v, b_len, index) = {
                let inner = block.inner.read().unwrap();
                (inner.k_cache.clone(), inner.v_cache.clone(), inner.len, inner.index)
            };

            // [REGISTRY-LOOKUP] Get location and path from the central table
            let (loc, ssd_path) = {
                let reg = self.registry.entries.read().unwrap();
                let entry = &reg[index];
                (entry.location, entry.ssd_path.clone())
            };

            // [WAIT-IF-LOADING] Safer wait with backoff and timeout
            if loc == KVLocation::Loading {
                let mut attempts = 0;
                while attempts < 100 { 
                    let l = {
                        let reg = self.registry.entries.read().unwrap();
                        reg[index].location
                    };
                    if l != KVLocation::Loading { break; }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    attempts += 1;
                }
            }

            // Re-read location after possible wait
            let loc = {
                let reg = self.registry.entries.read().unwrap();
                reg[index].location
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
                    let mut inner = block.inner.write().unwrap();
                    // [TEMPORAL-REUSE] If already decompressed by a previous layer (e.g. layer 0), use it immediately
                    if let (Some(kc), Some(vc)) = (&inner.k_cache, &inner.v_cache) {
                        k = Some(kc.clone());
                        v = Some(vc.clone());
                    } 
                    
                    // [SLOT-DIRECT-ACCESS] Try to get raw tensors from RAM slots if available
                    if k.is_none() && loc == KVLocation::RAM {
                        let slot_id = {
                            let reg = self.registry.entries.read().unwrap();
                            reg[index].slot_id
                        };
                        if let Some(sid) = slot_id {
                            let slot = &crate::models::qwen3vl::generate::SLOT_MANAGER.slots[sid];
                            // [SMART-LOCK] Try to acquire tensors from the master slot
                            if let Ok(k_guard) = slot.k_layers[self.layer_idx].try_lock() {
                                if let Some(tk) = k_guard.as_ref() {
                                    if let Ok(v_guard) = slot.v_layers[self.layer_idx].try_lock() {
                                        if let Some(tv) = v_guard.as_ref() {
                                            let tk_dev = tk.to_device(dev)?.to_dtype(target_dtype)?;
                                            let tv_dev = tv.to_device(dev)?.to_dtype(target_dtype)?;
                                            // Pin to VRAM for subsequent layers in this pass
                                            inner.k_cache = Some(tk_dev.clone());
                                            inner.v_cache = Some(tv_dev.clone());
                                            k = Some(tk_dev);
                                            v = Some(tv_dev);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if k.is_none() {
                        if let Some(meta) = &inner.bitkv_metadata {
                            let k_raw = self.decompress_from_bitkv(&meta.k_anchors, &meta.k_packed, &meta.k_scales, &meta.original_shape, dev)?;
                            let v_raw = self.decompress_from_bitkv(&meta.v_anchors, &meta.v_packed, &meta.v_scales, &meta.original_shape, dev)?;
                            // [PIN] Store for subsequent layers to reuse
                            inner.k_cache = Some(k_raw.clone());
                            inner.v_cache = Some(v_raw.clone());
                            k = Some(k_raw);
                            v = Some(v_raw);
                        } else { 
                            let spath = {
                                let reg = self.registry.entries.read().unwrap();
                                reg[index].ssd_path.clone()
                            };
                            if let Some(path) = &spath {
                                let filename = if inner.offset == 0 { format!("layer_{}_kv.safetensors", self.layer_idx) } 
                                               else { format!("layer_{}_kv_{}.safetensors", self.layer_idx, inner.offset) };
                                let full_path = path.join(filename);
                                if full_path.exists() {
                                    if let Ok(content) = std::fs::read(&full_path) {
                                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                            let extract_vec = |name: &str| -> Option<Vec<f32>> {
                                                st.tensor(name).ok().map(|v| unsafe {
                                                    std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec()
                                                })
                                            };
                                            if let (Some(ka), Some(kp), Some(ks)) = (extract_vec("k_anchors"), st.tensor("k_packed").ok(), extract_vec("k_scales")) {
                                                if let (Some(va), Some(vp), Some(vs)) = (extract_vec("v_anchors"), st.tensor("v_packed").ok(), extract_vec("v_scales")) {
                                                    let original_shape = if let Ok(view) = st.tensor("k_shape") {
                                                        let shape_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) };
                                                        shape_u32.iter().map(|&x| x as usize).collect()
                                                    } else { vec![1, 8, b_len, 128] };
                                                    inner.bitkv_metadata = Some(BitKVMetadata {
                                                        k_anchors: Tensor::from_vec(ka, (original_shape[0], original_shape[1], (b_len + 7) / 8 + 4, original_shape[3]), &Device::Cpu).unwrap(),
                                                        k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(),
                                                        k_scales: Tensor::from_vec(ks, (original_shape[0], original_shape[1], b_len, 1), &Device::Cpu).unwrap(),
                                                        v_anchors: Tensor::from_vec(va, (original_shape[0], original_shape[1], (b_len + 7) / 8 + 4, original_shape[3]), &Device::Cpu).unwrap(),
                                                        v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(),
                                                        v_scales: Tensor::from_vec(vs, (original_shape[0], original_shape[1], b_len, 1), &Device::Cpu).unwrap(),
                                                        original_shape,
                                                    });
                                                    
                                                    // Update registry to RAM state
                                                    {
                                                        let mut reg = self.registry.entries.write().unwrap();
                                                        reg[index].location = KVLocation::RAM;
                                                    }

                                                    let kr = self.decompress_from_bitkv(&inner.bitkv_metadata.as_ref().unwrap().k_anchors, &inner.bitkv_metadata.as_ref().unwrap().k_packed, &inner.bitkv_metadata.as_ref().unwrap().k_scales, &inner.bitkv_metadata.as_ref().unwrap().original_shape, dev)?;
                                                    let vr = self.decompress_from_bitkv(&inner.bitkv_metadata.as_ref().unwrap().v_anchors, &inner.bitkv_metadata.as_ref().unwrap().v_packed, &inner.bitkv_metadata.as_ref().unwrap().v_scales, &inner.bitkv_metadata.as_ref().unwrap().original_shape, dev)?;
                                                    inner.k_cache = Some(kr.clone()); inner.v_cache = Some(vr.clone());
                                                    k = Some(kr); v = Some(vr);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if k.is_none() {
                        println!("[WARNING] Block {} could not be loaded at Layer {}", index, self.layer_idx);
                        global_start_idx += b_len; continue;
                    }
                }
                _ => { global_start_idx += b_len; continue; }
            }

            let mut k = k.unwrap();
            let mut v = v.unwrap();

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
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
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
                    if index < reg.len() { reg[index].location } else { KVLocation::VRAM }
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
        self.kv_blocks.clear();
    }

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.iter().map(|b| b.inner.read().unwrap().len).sum()
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
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

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        let base_filename = match kv_name {
            Some(name) => format!("layer_{}_kv", name),
            None => format!("layer_{}_kv", self.layer_idx),
        };

        // 1. Try to find all fragments (e.g., base_0.safetensors, base_1024.safetensors)
        let mut fragments = Vec::new();
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().to_string();
                if fname.starts_with(&base_filename) && fname.ends_with(".safetensors") {
                    // Extract offset from filename: "layer_pug_kv_1024.safetensors" -> 1024
                    let offset = if fname == format!("{}.safetensors", base_filename) {
                        0
                    } else {
                        fname.strip_prefix(&format!("{}_", base_filename))
                             .and_then(|s| s.strip_suffix(".safetensors"))
                             .and_then(|s| s.parse::<usize>().ok())
                             .unwrap_or(0)
                    };
                    fragments.push((offset, entry.path()));
                }
            }
        }
        
        if fragments.is_empty() { return Ok(()); }
        fragments.sort_by_key(|f| f.0); // Ensure correct order

        let mut all_k = Vec::new();
        let mut all_v = Vec::new();
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        for (_, file_path) in fragments {
            let file = std::fs::File::open(&file_path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let st = safetensors::SafeTensors::deserialize(&mmap)?;
            
            let mode = if let Ok(view) = st.tensor("mode") {
                u32::from_le_bytes(view.data()[0..4].try_into().unwrap())
            } else { 1 };

            let (k, v) = if mode == 3 {
                let dequantize_bitkv = |prefix: &str| -> Result<Tensor> {
                    let anchors_view = st.tensor(&format!("{}_anchors", prefix))?;
                    let anchors = Tensor::from_slice(unsafe { std::slice::from_raw_parts(anchors_view.data().as_ptr() as *const f32, anchors_view.data().len() / 4) }, anchors_view.shape(), device)?;
                    let packed_view = st.tensor(&format!("{}_packed", prefix))?;
                    let packed = Tensor::from_slice(packed_view.data(), packed_view.shape(), device)?;
                    let scales_view = st.tensor(&format!("{}_scales", prefix))?;
                    let scales = Tensor::from_slice(unsafe { std::slice::from_raw_parts(scales_view.data().as_ptr() as *const f32, scales_view.data().len() / 4) }, scales_view.shape(), device)?;
                    let shape_view = st.tensor("k_shape")?;
                    let shape_u32: &[u32] = unsafe { std::slice::from_raw_parts(shape_view.data().as_ptr() as *const u32, shape_view.data().len() / 4) };
                    let shape: Vec<usize> = shape_u32.iter().map(|&x| x as usize).collect();
                    let t = self.decompress_from_bitkv(&anchors, &packed, &scales, &shape, device)?;
                    Ok(t.to_dtype(target_dtype)?)
                };
                (dequantize_bitkv("k")?, dequantize_bitkv("v")?)
            } else {
                let dequantize_legacy = |prefix: &str| -> Result<Tensor> {
                    let packed_view = st.tensor(&format!("{}_packed", prefix))?;
                    let packed = Tensor::from_slice(packed_view.data(), packed_view.shape(), device)?;
                    let scales_view = st.tensor(&format!("{}_scales", prefix))?;
                    let scales = Tensor::from_slice(unsafe { std::slice::from_raw_parts(scales_view.data().as_ptr() as *const f32, scales_view.data().len() / 4) }, scales_view.shape(), device)?;
                    let shape_view = st.tensor("k_shape")?;
                    let shape_u32: &[u32] = unsafe { std::slice::from_raw_parts(shape_view.data().as_ptr() as *const u32, shape_view.data().len() / 4) };
                    let shape: Vec<usize> = shape_u32.iter().map(|&x| x as usize).collect();
                    let t = if mode == 2 { self.decompress_from_1bit(&packed, &scales, &shape)? } 
                            else { self.decompress_from_8bit(&packed, &scales, &shape)? };
                    Ok(t.to_dtype(target_dtype)?)
                };
                (dequantize_legacy("k")?, dequantize_legacy("v")?)
            };
            all_k.push(k);
            all_v.push(v);
        }

        // Stitch all blocks back together
        let mut k = Tensor::cat(&all_k, 2)?;
        let mut v = Tensor::cat(&all_v, 2)?;

        // [LINEAR-BRIDGE] Convert 0.6B (512) cache to 2B (2048) cache
        let (_b, h, _s, d) = k.dims4()?;
        let target_heads = self.num_key_value_heads;
        let target_dim = self.head_dim;

        if h != target_heads || d != target_dim {
            if self.layer_idx == 0 {
                println!("[LINEAR-BRIDGE] Convert 0.6B (512) cache to 2B (2048) cache via BitKV");
            }
            if d < target_dim {
                k = Tensor::cat(&[&k, &k], D::Minus1)?;
                v = Tensor::cat(&[&v, &v], D::Minus1)?;
            }
            if h != target_heads {
                let mut k_heads = vec![];
                let mut v_heads = vec![];
                for i in 0..target_heads {
                    let source_idx = i % h;
                    k_heads.push(k.narrow(1, source_idx, 1)?);
                    v_heads.push(v.narrow(1, source_idx, 1)?);
                }
                k = Tensor::cat(&k_heads, 1)?;
                v = Tensor::cat(&v_heads, 1)?;
            }
        }

        let actual_k_len = k.dim(2)?;
        let use_len = if expected_len == 0 { actual_k_len } else { expected_len };
        let final_len = if use_len > upscale_refill_len { use_len - upscale_refill_len } else { use_len };

        if final_len > 0 {
            let safe_len = final_len.min(actual_k_len);
            let k_final = k.narrow(2, 0, safe_len)?.contiguous()?;
            let v_final = v.narrow(2, 0, safe_len)?.contiguous()?;
            
            self.kv_blocks.clear();
            let new_block = KVBlock::new(KVLocation::VRAM, 0, safe_len, 0);
            {
                let mut inner = new_block.inner.write().unwrap();
                inner.k_cache = Some(k_final);
                inner.v_cache = Some(v_final);
            }
            self.kv_blocks.push(new_block);
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

    pub fn load_kv_cache(&mut self, path: &Path, _device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        let device = self.input_layernorm.weight().device();
        self.self_attn.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name)
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
        let estimated_activation_buffer = 200_000_000; // Increased to 200MB for 28-layer prefill
        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         // [FIX] More realistic OS/Driver reserve (100MB)
                         let os_reserve = 100_000_000; 
                         safety_floor = os_reserve + kv_reserve + estimated_activation_buffer;
                     }
                 }
             }
        }

        let mut layer_devices = vec![];
        let mut pinned_layer_count = 0;
        
        for layer_idx in 0..config.num_hidden_layers {
            let actual_cost = cost_per_layer;
            if current_device.is_cuda() && is_vram_checked {
                 let buffer_factor = 1.2; 
                 if simulated_free_vram > ( (actual_cost as f64 * buffer_factor) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(actual_cost);
                     layer_devices.push(current_device.clone());
                     pinned_layer_count += 1;
                 } else {
                     layer_devices.push(Device::Cpu);
                 }
            } else {
                layer_devices.push(current_device.clone());
            }
        }

        println!("[MODEL] Configured with {} GPU-pinned layers out of {}.", pinned_layer_count, config.num_hidden_layers);

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
        let estimated_activation_buffer = 50_000_000; 

        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         let os_reserve = 50_000_000; 
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
            if current_device.is_cuda() && is_vram_checked {
                 let buffer_factor = 1.1; 
                 if simulated_free_vram > ( (cost_per_layer as f64 * buffer_factor) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(cost_per_layer);
                     pinned_layer_count += 1;
                 } else {
                     layer_device = Device::Cpu;
                 }
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

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        total_len: usize,
        position_ids_in: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        
        use nvml_wrapper::Nvml;
        if let Ok(nvml) = Nvml::init() {
                            if let Ok(dev) = nvml.device_by_index(0) {
                                if let Ok(mem) = dev.memory_info() {
                                    let current_progress = (seqlen_offset + seq_len).min(total_len);
                                    println!("[STAT] VRAM: {}MB Used / {}MB Free | Progress: {}/{}", mem.used / 1024 / 1024, mem.free / 1024 / 1024, current_progress, total_len);
                                }
                            }        }

        let target_device = if self.layers[0].device().is_cuda() { Device::new_cuda(0)? } else { Device::Cpu };
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mut xs = inputs_embeds.to_device(&target_device)?.to_dtype(target_dtype)?.contiguous()?;

        // --- [A-B-C WINDOW STRATEGY] ---
        // c: Computing (Current Layer)
        // b: Backing up (Save in progress, memory kept)
        // a: Archived (Save confirmed, memory dropped)

        let mut start_layer = 0;
        // [RESUME-LOGIC] Try to pick up from a saved layer checkpoint on disk
        if let Some(sid) = &self.active_session_id {
            let task_dir = crate::utils::paths::get_task_specific_dir(None, sid);
            for l_idx in (0..self.layers.len()).rev() {
                let checkpoint_path = task_dir.join(format!("inference_layer_{}_at_{}.safetensors", l_idx, seqlen_offset));
                if checkpoint_path.exists() {
                    if let Ok(recovered_xs) = crate::utils::tensor_utils::load_tensor(&checkpoint_path, "hidden_states", &target_device) {
                        println!("[SWAP-RESUME] Jumping to Layer {} for Offset {}.", l_idx + 1, seqlen_offset);
                        let r_xs: Tensor = recovered_xs; 
                        xs = r_xs.to_dtype(target_dtype)?;
                        start_layer = l_idx + 1;
                        break;
                    }
                }
            }
        }

        let position_ids = match position_ids_in {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                let mask = prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    seqlen_offset,
                    xs.device(),
                )?;
                Some(mask.to_dtype(DType::F32)?.contiguous()?)
            }
        };

        // [SMART-SEQUENTIAL-LOOP]
        let num_layers = self.layers.len();
        println!("[TRACE] Entering layer loop. Total layers: {}", num_layers);
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if layer_idx < start_layer { continue; }

            // [FLOW-CONTROL] 대기열이 가득 차면 레이어 연산 전 일시 정지
            while ACTIVE_BAKE_TASKS.load(Ordering::SeqCst) >= 20 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            let is_pinned = layer_idx < self.pinned_layer_count;
            
            // 1. [C: Computing] Prepare and Run Layer
            if !is_pinned {
                layer.to_device(&target_device)?;
            }

            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref(), seqlen_offset)?;
            
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    let m_orig = visual_pos_masks.unwrap();
                    let e_orig = &deepstack_embeds[layer_idx];
                    let mask = if !m_orig.device().same_device(xs.device()) { m_orig.to_device(xs.device())? } else { m_orig.clone() };
                    let embed = if !e_orig.device().same_device(xs.device()) { e_orig.to_device(xs.device())? } else { e_orig.clone() };
                    let embed = if embed.dtype() != xs.dtype() { embed.to_dtype(xs.dtype())? } else { embed };
                    xs = mask_index_add(&xs.squeeze(0)?, &mask.squeeze(0)?, &embed)?.unsqueeze(0)?;
                }
            }

            // 2. 레이어 연산 직후 스냅샷 생성 및 비동기 워커로 오프로드
            if let Some(sid) = &self.active_session_id {
                let task_dir = crate::utils::paths::get_task_specific_dir(None, sid);
                if !task_dir.exists() { let _ = std::fs::create_dir_all(&task_dir); }
                
                // [HANDOFF] GPU 연산 결과를 CPU로 고속 복사 (PCIe 전송)
                if let Ok(xs_cpu) = xs.to_device(&Device::Cpu) {
                    // CPU로 데이터가 넘어온 시점에서 GPU의 대기열 카운트는 해소된 것으로 간주
                    // ACTIVE_BAKE_TASKS.fetch_sub(1, Ordering::SeqCst); // 이 시점은 워커에서 최종 관리하되, 
                    // 아래 로직에서 GPU는 멈추지 않고 진행함.
                    
                    let final_path = task_dir.join(format!("inference_layer_{}_at_{}.safetensors", layer_idx, seqlen_offset));
                    
                    ACTIVE_BAKE_TASKS.fetch_add(1, Ordering::SeqCst);
                    
                    // [WORKER] 비동기 저장 워커 기동 (CPU RAM -> SSD)
                    tokio::task::spawn_blocking(move || {
                        let mut map = std::collections::HashMap::new();
                        map.insert("hidden_states".to_string(), xs_cpu);
                        let _ = candle_core::safetensors::save(&map, &final_path);
                        ACTIVE_BAKE_TASKS.fetch_sub(1, Ordering::SeqCst);
                    });
                }
            }

            // [CRITICAL] 3. 레이어 메모리 포지션 강제 초기화 (Async Await)
            let dev_clone = target_device.clone();
            
            if !is_pinned {
                layer.to_device(&Device::Cpu)?; 
                
                let _ = tauri::async_runtime::spawn(async move {
                    if dev_clone.is_cuda() {
                        let _ = tokio::task::spawn_blocking(move || {
                            let _ = dev_clone.synchronize();
                        }).await;
                    }
                });
            }
        }

        println!("[TRACE] All layers done.");
        
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
        if path.exists() {
            self.layers.iter_mut().try_for_each(|layer| {
                layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name)
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

    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> {
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

        // 임계값 설정
        let danger_zone = 300_000_000; // 300MB 이하: 위험 (내리기)
        let safe_zone = 800_000_000;   // 800MB 이상: 여유 (올리기)

        if free_vram > 0 && free_vram < danger_zone {
            // [OFFLOAD] GPU -> CPU (뒤쪽 레이어부터)
            for layer in self.layers.iter_mut().rev() {
                if layer.device().is_cuda() {
                    println!("[REBALANCE] Low VRAM ({:.2} MB). Offloading Layer {} to CPU.", free_vram as f64 / 1e6, layer.self_attn.layer_idx);
                    layer.to_device(&Device::Cpu)?;
                    break; // 한 번에 하나씩만 이동 (급격한 변화 방지)
                }
            }
        } else if free_vram > safe_zone {
            // [UPLOAD] CPU -> GPU (앞쪽 레이어부터)
            // 주의: self.layers[0]의 디바이스가 GPU여야 업로드 대상 장치를 알 수 있음
            // 또는 generate 시점에 저장해둔 메인 디바이스 정보를 활용해야 함.
            // 여기서는 첫 번째 레이어의 디바이스가 CPU일 경우를 대비해, 
            // 외부에서 주입받거나, 혹은 임시로 CUDA:0 (또는 device_id)를 타겟으로 함.
            let target_device = Device::new_cuda(device_id)?;
            
            for layer in self.layers.iter_mut() {
                if layer.device().is_cpu() {
                    println!("[REBALANCE] Free VRAM ({:.2} GB). Uploading Layer {} to GPU.", free_vram as f64 / 1e9, layer.self_attn.layer_idx);
                    layer.to_device(&target_device)?;
                    break; 
                }
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

    pub fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
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
    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> { self.language_model.rebalance_layers(device_id) }
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

    pub fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
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
    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> { self.language_model.rebalance_layers(device_id) }
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