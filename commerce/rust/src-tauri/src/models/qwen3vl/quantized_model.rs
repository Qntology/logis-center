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
use memmap2::Mmap;

use crate::{
    models::{
        common::eager_attention_forward,
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

    pub fn inner_shape(&self) -> Vec<usize> {
        match &self.inner {
            QMatMul::QTensor(q) => q.shape().dims().to_vec(),
            QMatMul::Tensor(t) => t.dims().to_vec(),
            QMatMul::TensorF16(t) => t.dims().to_vec(),
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
struct QuantizedKV {
    k_packed: Tensor, // [B, H, S, D/2]
    v_packed: Tensor, // [B, H, S, D/2]
    k_scales: Tensor, // [B, H, S, 1]
    v_scales: Tensor,
    original_shape: Vec<usize>,
    block_size: usize,
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
    pub kv_cache: Option<(Tensor, Tensor)>,
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
        
        if let Some((k, v)) = &self.kv_cache {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            self.kv_cache = Some((
                k.to_device(device)?.to_dtype(target_dtype)?, 
                v.to_device(device)?.to_dtype(target_dtype)?
            ));
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
    ) -> Result<Self> {
        let _hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);

        let is_06b = config.hidden_size == 1024;
        
        let (q, k, v, o, q_n, k_n) = if is_gguf_naming {
            if is_06b {
                // [0.6B-SPECIFIC] Qwen3-0.6B uses combined QKV or specific naming
                ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
            } else {
                ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
            }
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

        // [0.6B-HYBRID-FIX] Detect and handle combined attn_qkv tensors often found in 0.6B GGUF files
        let q_proj_name = format!("{base_name}.{q}");
        let k_proj_name = format!("{base_name}.{k}");
        let v_proj_name = format!("{base_name}.{v}");
        let qkv_combined_name = format!("{base_name}.attn_qkv");

        // [DETECTION] Use GGUF info to find correct hidden size for THIS model file
        let actual_h_size = if let Some(info) = ct.tensor_infos.get(&format!("{base_name}.attn_norm.weight")) {
            info.shape.dims()[0]
        } else if let Some(info) = ct.tensor_infos.get(&format!("{base_name}.input_layernorm.weight")) {
            info.shape.dims()[0]
        } else {
            1024 // Fallback for 0.6B
        };

        let has_individual = ct.tensor_infos.contains_key(&format!("{q_proj_name}.weight"));
        let has_combined = ct.tensor_infos.contains_key(&format!("{qkv_combined_name}.weight"));

        let (q_proj, k_proj, v_proj) = if has_individual {
            let mut qp = get_qlinear(ct, reader, &q_proj_name, device, dtype)?;
            let kp = get_qlinear(ct, reader, &k_proj_name, device, dtype)?;
            let vp = get_qlinear(ct, reader, &v_proj_name, device, dtype)?;
            
            // [HARDENING] 만약 로드된 Q의 입력 차원이 2048인데 현재 모델이 1024 기반이라면 슬라이싱 시도
            if qp.inner_shape().len() >= 2 && qp.inner_shape()[1] == 2048 && actual_h_size == 1024 {
                println!("[MODEL-FIX] Slicing individual Q from 2048-dim container for 0.6B layer.");
                qp = get_sliced_qlinear(ct, reader, &q_proj_name, device, dtype, 1, 0, 1024)?;
            }
            (qp, kp, vp)
        } else if has_combined {
            let qp = get_sliced_qlinear(ct, reader, &qkv_combined_name, device, dtype, 0, 0, actual_h_size)?;
            let kp = get_sliced_qlinear(ct, reader, &qkv_combined_name, device, dtype, 0, actual_h_size, actual_h_size)?;
            let vp = get_sliced_qlinear(ct, reader, &qkv_combined_name, device, dtype, 0, 2 * actual_h_size, actual_h_size)?;
            (qp, kp, vp)
        } else {
            return Err(anyhow!("Could not find QKV tensors for layer {}", layer_idx));
        };

        let o_proj = get_qlinear(ct, reader, &format!("{base_name}.{o}"), device, dtype)?;

        if layer_idx == 0 {
            println!("[DEBUG-ATTN-L0] Q shape: {:?}, K shape: {:?}, V shape: {:?}, head_dim: {}", 
                q_proj.inner_shape(), k_proj.inner_shape(), v_proj.inner_shape(), head_dim);
        }

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
            kv_cache: None,
            layer_idx,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
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

        let (b_sz, q_len, last_dim) = xs.dims3()?;
        
        if self.layer_idx == 0 {
            println!("[TRACE-ATTN-L0] xs shape: {:?}, q_proj device: {:?}, target_dtype: {:?}", xs.shape(), self.q_proj.device(), target_dtype);
        }

        let query_states = self.q_proj.forward(&xs).map_err(|e| {
            println!("[ERROR-ATTN] q_proj mismatch at Layer {}: xs={:?}, weight_dev={:?}. Error: {}", 
                self.layer_idx, xs.shape(), self.q_proj.device(), e);
            e
        })?.reshape((
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
        
        // 3. [HARDENING] KV Cache Concatenation Guard
        let (key_states, value_states): (Tensor, Tensor) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                // Key Cache Alignment
                let pk = if !prev_k.device().same_device(dev) { prev_k.to_device(dev)? } else { prev_k.clone() };
                let pk = if pk.dtype() != target_dtype { pk.to_dtype(target_dtype)? } else { pk }.contiguous()?;
                
                // Value Cache Alignment
                let pv = if !prev_v.device().same_device(dev) { prev_v.to_device(dev)? } else { prev_v.clone() };
                let pv = if pv.dtype() != target_dtype { pv.to_dtype(target_dtype)? } else { pv }.contiguous()?;
                
                let k = Tensor::cat(&[&pk, &key_states.contiguous()?], 2)?.contiguous()?;
                let v = Tensor::cat(&[&pv, &value_states.contiguous()?], 2)?.contiguous()?;
                (k, v)
            }
        };

        // Update working cache
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // 4. [HARDENING] Adjusted Mask Logic
        let actual_seq_len = key_states.dim(2)?;
        let adjusted_mask = if let Some(mask) = attention_mask {
            let mask_len = mask.dim(candle_core::D::Minus1)?;
            if mask_len < actual_seq_len {
                let padding = Tensor::zeros((b_sz, 1, q_len, actual_seq_len - mask_len), mask.dtype(), mask.device())?;
                Some(Tensor::cat(&[padding, mask.clone()], candle_core::D::Minus1)?)
            } else if mask_len > actual_seq_len {
                Some(mask.narrow(candle_core::D::Minus1, 0, actual_seq_len)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };

        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            adjusted_mask.as_ref(),
            self.scaling,
        )?;

        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
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

    pub fn decompress_from_bitkv(&self, anchors: &Tensor, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = anchors.device();
        let (b, h, s, d) = (original_shape[0], original_shape[1], original_shape[2], original_shape[3]);
        let anchors_vec = anchors.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let packed_vec = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        
        let mut decoded = vec![0.0f32; b * h * s * d];
        let anchor_count = anchors_vec.len() / (b * h * d);

        // [TURBO] Parallel Decompression
        decoded.par_chunks_mut(s * d).enumerate().for_each(|(bh_idx, head_data)| {
            let anchor_bh_offset = bh_idx * anchor_count * d;
            let scale_head_offset = bh_idx * s;
            let packed_head_offset = bh_idx * s * d;

            for token_idx in 0..s {
                let target_offset = token_idx * d;
                let scale = scales_vec[scale_head_offset + token_idx];
                
                if token_idx < 4 || token_idx % 8 == 0 {
                    let anchor_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                    let src = &anchors_vec[anchor_bh_offset + anchor_pos * d .. anchor_bh_offset + (anchor_pos + 1) * d];
                    head_data[target_offset..target_offset + d].copy_from_slice(src);
                } else {
                    for i in 0..d {
                        let bit_idx = packed_head_offset + token_idx * d + i;
                        let is_set = (packed_vec[bit_idx / 8] & (1 << (bit_idx % 8))) != 0;
                        head_data[target_offset + i] = if is_set { scale } else { -scale };
                    }
                }
            }
        });
        
        let t = Tensor::from_vec(decoded, original_shape, &Device::Cpu)?;
        Ok(t.to_device(device)?)
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
        self.kv_cache = None;
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.kv_cache {
            None => 0,
            Some((k, _)) => k.dim(2).unwrap_or(0),
        }
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        self.kv_cache = None;
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
        self.kv_cache = match &self.kv_cache {
            None => Some((k_final, v_final)),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k_final], 2)?;
                let v = Tensor::cat(&[prev_v, &v_final], 2)?;
                Some((k, v))
            }
        };
        Ok(())
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _block_size: usize) -> Result<()> {
        let file = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
        let mut map = HashMap::new();

        if let Some((k, v)) = &self.kv_cache {
            let (k_anchors, k_packed, k_scales, k_shape) = self.compress_to_bitkv(&k)?;
            let (v_anchors, v_packed, v_scales, _) = self.compress_to_bitkv(&v)?;
            
            map.insert("k_anchors".to_string(), k_anchors);
            map.insert("k_packed".to_string(), k_packed);
            map.insert("k_scales".to_string(), k_scales);
            map.insert("v_anchors".to_string(), v_anchors);
            map.insert("v_packed".to_string(), v_packed);
            map.insert("v_scales".to_string(), v_scales);
            map.insert("k_shape".to_string(), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect(), (k_shape.len(),), &Device::Cpu)?);
            map.insert("mode".to_string(), Tensor::from_vec(vec![3u32], (1,), &Device::Cpu)?); // Mode 3: BitKV
        } else {
            return Ok(());
        }

        candle_core::safetensors::save(&map, &file)?;
        if clear { self.kv_cache = None; }
        Ok(())
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        let file_path = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
        if !file_path.exists() { return Ok(()); }

        let file = std::fs::File::open(&file_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;
        
        let mode = if let Ok(view) = st.tensor("mode") {
            u32::from_le_bytes(view.data()[0..4].try_into().unwrap())
        } else { 1 };
        
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        let (mut k, mut v) = if mode == 3 {
            // BitKV Loading
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

                let t = self.decompress_from_bitkv(&anchors, &packed, &scales, &shape)?;
                Ok(t.to_dtype(target_dtype)?)
            };
            (dequantize_bitkv("k")?, dequantize_bitkv("v")?)
        } else {
            // Fallback to old 1-bit or 8-bit
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
            self.kv_cache = Some((k.narrow(2, 0, safe_len)?, v.narrow(2, 0, safe_len)?));
        }
        Ok(())
    }

                    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
                        self.save_kv_cache(path, true, block_size)
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
        baking_only: bool, // Use this flag now
    ) -> Result<Self> {
        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx)?;
        
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
        let xs = self.self_attn.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> {
        self.self_attn.save_kv_cache(path, clear, block_size)
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.self_attn.offload_kv_cache(path, block_size)
    }

    pub fn load_kv_cache(&mut self, path: &Path, _device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        let device = self.input_layernorm.weight().device();
        self.self_attn.load_kv_cache(path, device, expected_len, upscale_refill_len)
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLTextModel {
    pub embed_tokens: Embedding, 
    pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
    pub mmap: Option<Arc<Mmap>>, // Keep mmap alive for tensors
    pub baking_only: bool, // [NEW] Skip MLP for KV baking
    pub is_forced_cpu: bool, // [FIX] Prevents rebalancer from uploading back to GPU
    pub is_text: bool,
    pub is_image: bool,
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
        is_text: bool,
        is_image: bool,
    ) -> Result<Self> {
        println!("[MODEL-INIT-DEBUG] Name: {}, Hidden: {}, Layers: {}, TextMode: {}", base_name, config.hidden_size, config.num_hidden_layers, is_text);
        let is_forced_cpu = device.is_cpu();
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);

        // [DETECTION-FIRST] Determine actual hidden size from GGUF BEFORE initializing anything
        let actual_h_size = if let Some(info) = ct.tensor_infos.get(&format!("{base_name}.attn_norm.weight")) {
            info.shape.dims()[0]
        } else if let Some(info) = ct.tensor_infos.get(&format!("blk.0.attn_norm.weight")) {
            info.shape.dims()[0]
        } else if let Some(info) = ct.tensor_infos.get(&"token_embd.weight".to_string()) {
            info.shape.dims()[1]
        } else {
            // Last resort: search for any layer norm weight
            ct.tensor_infos.keys().find(|k| k.contains("attn_norm.weight") || k.contains("input_layernorm.weight"))
                .and_then(|k| ct.tensor_infos.get(k))
                .map(|info| info.shape.dims()[0])
                .unwrap_or(config.hidden_size)
        };

        println!("[MODEL-INIT] Name: {}, Config Hidden: {}, GGUF Actual: {}, Layers: {}", base_name, config.hidden_size, actual_h_size, config.num_hidden_layers);

        let mut patched_config_owned = config.clone();
        if actual_h_size == 1024 {
            println!("[MODEL-FIX] 0.6B detected early. Overriding all settings to 1024/8 heads.");
            patched_config_owned.hidden_size = 1024;
            patched_config_owned.num_attention_heads = 8;
            patched_config_owned.num_key_value_heads = 8;
        } else {
            patched_config_owned.hidden_size = actual_h_size;
        }
        let config = &patched_config_owned;

        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let embed_tokens = if let Ok(tensor) = ct.tensor(&mut reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             Embedding::new(tensor, h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             Embedding::new(tensor, h)
        } else {
             println!("[HYBRID] Embedding not found in GGUF. Using config size: {}", config.hidden_size);
             let dummy_tensor = Tensor::zeros((config.vocab_size, config.hidden_size), dtype, device)?;
             Embedding::new(dummy_tensor, config.hidden_size)
        };

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
        
        // [FORCE-OPTIMIZATION] 4GB 내외의 VRAM 환경에서는 2B 모델이 충분히 들어갑니다.
        let _force_gpu = if current_device.is_cuda() && is_vram_checked {
            simulated_free_vram > 3_000_000_000 
        } else {
            false
        };

        for _ in 0..config.num_hidden_layers {
            let actual_cost = cost_per_layer;
            if current_device.is_cuda() && is_vram_checked {
                 // [SAFE-ALLOC] 가용 VRAM에서 20% 여유를 두고 할당 (OOM 방지)
                 // buffer_factor 1.2 = 레이어 크기의 1.2배 공간이 있어야 할당함
                 let buffer_factor = 1.2; 
                 if simulated_free_vram > ( (actual_cost as f64 * buffer_factor) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(actual_cost);
                     layer_devices.push(current_device.clone());
                 } else {
                     // 공간 부족 시 CPU로 할당 (속도는 조금 느려지지만 절대 죽지 않음)
                     layer_devices.push(Device::Cpu);
                 }
            } else {
                // CPU 모드거나 VRAM 확인 불가 시 기본 장치 사용
                layer_devices.push(current_device.clone());
            }
        }

        // [ORGANIC] Dynamic Threading for Parallel Loading
        let thread_config = crate::utils::resources::get_optimal_thread_config(current_device.is_cpu());
        println!("[MODEL] Organic Loading: Using {} threads ({})", thread_config.thread_count, thread_config.description);
        
        let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_config.thread_count).build()?;
        
        // Ensure we capture the PATCHED config, not the original one
        let final_config = config; 

        let num_layers_to_load = if baking_only { 1 } else { final_config.num_hidden_layers };

        let layers: Result<Vec<_>> = pool.install(|| {
            (0..num_layers_to_load).into_par_iter().zip(layer_devices).map(|(layer_idx, layer_device)| {
                let mut local_cursor = std::io::Cursor::new(mmap);
                let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
                let standard = format!("{base_name}.layers.{layer_idx}");
                let gguf_blk = format!("blk.{layer_idx}");
                let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };
                QuantizedQwen3VLTextDecoderLayer::new(
                    final_config, ct, &mut local_cursor, &prefix, &layer_device, layer_dtype, layer_idx, baking_only
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
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb, mrope_section, mmap: mmap_handle, baking_only, is_forced_cpu, is_text, is_image })
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
        is_text: bool,
        is_image: bool,
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
        let num_layers_to_load = if baking_only { 1 } else { config.num_hidden_layers };

        for layer_idx in 0..num_layers_to_load {
            if current_device.is_cuda() && is_vram_checked {
                 let effective_cost = if cost_per_layer > 150_000_000 { 40_000_000 } else { cost_per_layer };
                 if simulated_free_vram > ( (effective_cost as f64 * 1.02) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(effective_cost);
                 } else {
                     current_device = Device::Cpu;
                 }
            }

            let layer_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };
            let standard = format!("{base_name}.layers.{layer_idx}");
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };

            let layer = QuantizedQwen3VLTextDecoderLayer::new(
                config, ct, reader, &prefix, &current_device, layer_dtype, layer_idx, baking_only
            )?;
            layers.push(layer)
        }
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let norm_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, &current_device, norm_dtype)?;
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
            baking_only,
            is_forced_cpu,
            is_text,
            is_image,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        println!("[TRACE-MODEL] Start forward. Input: {:?} {:?}, Target: BF16/F32", inputs_embeds.device(), inputs_embeds.dtype());
        
        let position_ids = match position_ids {
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
        
        let mut xs = inputs_embeds.clone();
        let target_dtype = if xs.device().is_cuda() { DType::BF16 } else { DType::F32 };
        let mut xs = xs.to_dtype(target_dtype)?.contiguous()?;

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
                // [HARDENING] Mask DType Guard
                Some(mask.to_dtype(DType::F32)?.contiguous()?)
            }
        };

        let _total_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    let m_orig = visual_pos_masks.unwrap();
                    let e_orig = &deepstack_embeds[layer_idx];
                    
                    let mask = if !m_orig.device().same_device(xs.device()) { m_orig.to_device(xs.device())? } else { m_orig.clone() };
                    let embed = if !e_orig.device().same_device(xs.device()) { e_orig.to_device(xs.device())? } else { e_orig.clone() };
                    
                    // [HARDENING] Visual Merge DType Guard
                    let embed = if embed.dtype() != xs.dtype() { embed.to_dtype(xs.dtype())? } else { embed };

                    xs = mask_index_add(&xs.squeeze(0)?, &mask.squeeze(0)?, &embed)?.unsqueeze(0)?;
                }
            }
        }
        
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) {
            xs = xs.to_device(norm_dev)?;
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn get_kv_len(&self) -> usize {
        self.layers.get(0).map(|l| l.get_kv_len()).unwrap_or(0)
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
                let k_small = layer.self_attn.decompress_from_bitkv(&k_anchors[i].to_device(&target_device)?, &k_packed[i].to_device(&target_device)?, &k_scales[i].to_device(&target_device)?, original_shape)?;
                let v_small = layer.self_attn.decompress_from_bitkv(&v_anchors[i].to_device(&target_device)?, &v_packed[i].to_device(&target_device)?, &v_scales[i].to_device(&target_device)?, original_shape)?;
                
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
        }

        // [STABILITY] Use sequential saving to avoid CUDA_ERROR_INVALID_CONTEXT in rayon threads
        self.layers.iter_mut().try_for_each(|layer| {
            layer.save_kv_cache(path, clear, block_size)
        })
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        if path.exists() {
            self.layers.iter_mut().try_for_each(|layer| {
                layer.load_kv_cache(path, device, expected_len, upscale_refill_len)
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
        baking_only: bool,
        is_text: bool,
        is_image: bool,
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
            &t_config, ct_main, main_mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image
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
        baking_only: bool,
        is_text: bool,
        is_image: bool,
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

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image)?;
        
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
    
    fn get_rope_index(&self, input_ids: &Tensor, _image_grid_thw: Option<&Tensor>, _video_grid_thw: Option<&Tensor>, _mask: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let position_ids = Tensor::arange(0u32, input_ids.dim(1)? as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?;
        let mrope = Tensor::zeros((input_ids.dim(0)?, 1), input_ids.dtype(), input_ids.device())?;
        Ok((position_ids, mrope))
    }

    pub fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;

        // [HYBRID-BRIDGE] Upscale 1024 -> 2048 ONLY if we are in a 2048-dim model (Large)
        // [FIX] Check the ACTUAL hidden size of the underlying model layers, not just the config.
        let model_actual_dim = self.language_model.embed_tokens.hidden_size();
        let current_input_dim = inputs_embeds.dim(candle_core::D::Minus1)?;
        
        if model_actual_dim == 2048 && current_input_dim == 1024 {
            println!("[HYBRID-BRIDGE-TOP] Scaling inputs_embeds 1024 -> 2048 for Large Model.");
            inputs_embeds = candle_core::Tensor::cat(&[&inputs_embeds, &inputs_embeds], candle_core::D::Minus1)?.contiguous()?;
        }

        if let Some(pixel_values) = pixel_values { 
            if let Some(image_grid_thw) = image_grid_thw { 
                let (image_embeds, _) = self.get_vision_features(pixel_values, image_grid_thw)?; 
                let image_embeds = Tensor::cat(&image_embeds, 0)?; 
                let vision_mask = self.get_placeholder_mask(&input_ids, true)?; 
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; 
            } 
        }
        let (position_ids, _) = self.get_rope_index(&input_ids, image_grid_thw, video_grid_thw, None)?;
        let position_ids = if let Some(cache_pos) = cache_position { 
            let start = cache_pos.flatten_all()?.i(0)?.to_scalar::<u32>()?; 
            Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))? 
        } else { position_ids };
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, Some(&position_ids), None, None)?;
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
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, block_size) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; self.text_device = device.clone(); self.vision_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> { self.language_model.rebalance_layers(device_id) }
}

#[derive(Clone)]
pub struct QuantizedQwen3TextModel {
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: Option<QLinear>,
    pub text_device: Device,
    pub mmap: Option<Arc<Mmap>>,
    pub is_text: bool,
    pub is_image: bool,
}

impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool, is_text: bool, is_image: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }
        
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(&t_config, ct_main, mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image)?;
        let lm_head = if !baking_only {
            let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut reader = std::io::Cursor::new(mmap);
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, &mut reader, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, &mut reader, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, &mut reader, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: mmap_handle, is_text, is_image })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, reader_main: &mut R, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool, is_text: bool, is_image: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {})", baking_only, single_layer_mode);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image)?;
        let lm_head = if !baking_only {
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: None, is_text, is_image })
    }

    pub fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;

        let start = if let Some(cp) = cache_position { cp.flatten_all()?.i(0)?.to_scalar::<u32>()? } else { 0 };
        let position_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, Some(&position_ids), None, None)?;
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
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len) }
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