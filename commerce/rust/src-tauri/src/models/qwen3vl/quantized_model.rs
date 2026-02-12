use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use rayon::prelude::*;
use nvml_wrapper::Nvml;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

// A trait alias for Seek + Read, as direct trait object `dyn Seek + Read` is not allowed.
// See: https://doc.rust-lang.org/reference/items/traits.html#auto-traits
pub trait SeekRead: std::io::Seek + std::io::Read {}
impl<T: std::io::Seek + std::io::Read> SeekRead for T {}

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
        let x_f32 = x.to_dtype(DType::F32)?;
        
        let x_shape = x_f32.dims();
        let w_shape = self.weight.dims();
        let last_dim_x = *x_shape.last().unwrap();
        let last_dim_w = *w_shape.last().unwrap();

        // [ROBUST-NORM] If input is [B, S, H, D] but weight is [H*D], flatten to [B, S, H*D]
        // If input is [B, S, H*D] but weight is [D], reshape to [B, S, H, D]
        let needs_flatten = last_dim_x < last_dim_w && (last_dim_w % last_dim_x == 0);
        let needs_split = last_dim_x > last_dim_w && (last_dim_x % last_dim_w == 0);

        let (x_active, final_shape) = if needs_flatten {
            let flat_shape = [x_shape[..x_shape.len()-2].to_vec(), vec![last_dim_w]].concat();
            (x_f32.reshape(flat_shape)?, Some(x_shape))
        } else if needs_split {
            let split_shape = [x_shape[..x_shape.len()-1].to_vec(), vec![last_dim_x / last_dim_w, last_dim_w]].concat();
            (x_f32.reshape(split_shape)?, Some(x_shape))
        } else {
            (x_f32, None)
        };

        let variance = x_active.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let hidden_states = x_active.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = hidden_states.to_dtype(target_dtype)?.broadcast_mul(&self.weight)?;
        
        if let Some(shape) = final_shape {
            out.reshape(shape)
        } else {
            Ok(out)
        }
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
    pub hybrid_parts: f64,
    pub needs_transpose: bool,
    pub is_handshake_active: bool, // [NEW] Handshake Protocol 성공 여부를 영구 저장
}

impl QuantizedQwen3VLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?;
        self.k_norm.to_device(device)?;
        
        // [KV-CACHE-MIGRATION]
        if let Some((k, v)) = self.kv_cache.take() {
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
        let head_dim = config.head_dim;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);

        let actual_h_size = config.hidden_size;
        let is_06b = actual_h_size == 1024;
        
        let (q, k, v, o, q_n, k_n) = if is_gguf_naming {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        };

        // [STRICT-SPEC-ALIGNMENT] 2B 모드(2048)인 경우 무조건 16헤드 강제
        let num_attention_heads = if actual_h_size == 2048 {
            16
        } else if is_06b {
            8
        } else {
            16
        };
        
        let num_key_value_heads = 8; 
        let num_kv_groups = num_attention_heads / num_key_value_heads;

        let q_proj_name = format!("{base_name}.{q}");
        let k_proj_name = format!("{base_name}.{k}");
        let v_proj_name = format!("{base_name}.{v}");

        let q_proj = get_qlinear_v2(ct, reader, &q_proj_name, device, dtype, actual_h_size)?;
        let k_proj = get_qlinear_v2(ct, reader, &k_proj_name, device, dtype, actual_h_size)?;
        let v_proj = get_qlinear_v2(ct, reader, &v_proj_name, device, dtype, actual_h_size)?;
        let o_proj = get_qlinear_v2(ct, reader, &format!("{base_name}.{o}"), device, dtype, actual_h_size)?;

        let q_norm = get_rms_norm(ct, reader, &format!("{base_name}.{q_n}"), config.rms_norm_eps, device, dtype, actual_h_size)?;
        let k_norm = get_rms_norm(ct, reader, &format!("{base_name}.{k_n}"), config.rms_norm_eps, device, dtype, actual_h_size)?;

        // [HYBRID-PART-CALC] Determine original 2B parts and transpose needs
        let mut hybrid_parts = 1.0;
        let mut needs_transpose = false;
        if is_06b {
            if let Some(info) = ct.tensor_infos.get(&format!("{q_proj_name}.weight")) {
                let d0 = info.shape.dims()[0];
                let d1 = if info.shape.dims().len() > 1 { info.shape.dims()[1] } else { 0 };
                hybrid_parts = (d0 as f64 / 1024.0).max(1.0);
                // If 2B original is transposed compared to 0.6B engine
                if d0 < d1 && d1 == 2048 { needs_transpose = true; }
            }
        }

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            num_kv_groups,
            scaling,
            kv_cache: None,
            layer_idx,
            hybrid_parts,
            needs_transpose,
            is_handshake_active: false, // Default to inactive
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

        // 1. [HARDENING] Inbound Input Alignment
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };

        let (b_sz, q_len, _last_dim) = xs.dims3()?;
        
        let query_states = {
            let mut xs_q = xs.clone();
            if xs_q.dim(D::Minus1)? == 2048 && self.q_proj.inner_shape()[1] == 1024 {
                xs_q = xs_q.narrow(D::Minus1, 0, 1024)?.contiguous()?;
            }
            let res = self.q_proj.forward(&xs_q)?;
            
            // [AUTO-RECOVERY] Bulletproof Head Alignment
            let q_out_dim = res.dim(D::Minus1)?;
            let heads = q_out_dim / self.head_dim;
            if heads != self.num_attention_heads {
                self.num_attention_heads = heads;
            }
            res
        };
        // [FIX] Apply norm to hidden dimension before splitting into heads to match weight shape [H*D]
        let query_states = self.q_norm.forward(&query_states)?
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        
        let key_states = {
            let mut xs_k = xs.clone();
            if xs_k.dim(D::Minus1)? == 2048 && self.k_proj.inner_shape()[1] == 1024 {
                xs_k = xs_k.narrow(D::Minus1, 0, 1024)?.contiguous()?;
            }
            let res = self.k_proj.forward(&xs_k)?;
            
            // [AUTO-RECOVERY] Bulletproof KV Head Alignment
            let k_out_dim = res.dim(D::Minus1)?;
            let kv_heads = k_out_dim / self.head_dim;
            if kv_heads != self.num_key_value_heads {
                self.num_key_value_heads = kv_heads;
            }
            res
        };
        // [FIX] Apply norm to hidden dimension before splitting into heads
        let key_states = self.k_norm.forward(&key_states)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        
        let value_states = {
            let mut xs_v = xs.clone();
            if xs_v.dim(D::Minus1)? == 2048 && self.v_proj.inner_shape()[1] == 1024 {
                xs_v = xs_v.narrow(D::Minus1, 0, 1024)?.contiguous()?;
            }
            self.v_proj.forward(&xs_v)?
        }
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Synchronize KV groups after recovery
        self.num_kv_groups = self.num_attention_heads / self.num_key_value_heads;

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
                let mut pk = if !prev_k.device().same_device(dev) { prev_k.to_device(dev)? } else { prev_k.clone() };
                let mut pk = if pk.dtype() != target_dtype { pk.to_dtype(target_dtype)? } else { pk }.contiguous()?;
                
                // Value Cache Alignment
                let mut pv = if !prev_v.device().same_device(dev) { prev_v.to_device(dev)? } else { prev_v.clone() };
                let mut pv = if pv.dtype() != target_dtype { pv.to_dtype(target_dtype)? } else { pv }.contiguous()?;
                
                // [AUTO-UPSCALE] If previous cache has fewer heads (e.g. 8 vs 16), replicate heads
                let prev_heads = pk.dim(1)?;
                let curr_heads = key_states.dim(1)?;
                
                if prev_heads < curr_heads {
                    let ratio = curr_heads / prev_heads;
                    if ratio > 1 {
                        // Replicate heads: [B, 8, S, D] -> [B, 16, S, D]
                        // We use repeat_interleave logic: repeat elements along dim 1
                        // Candle doesn't have repeat_interleave, so we use repeat + reshape + transpose
                        // [B, H, S, D] -> [B, H, 1, S, D] -> [B, H, R, S, D] -> [B, H*R, S, D]
                        let (b, h, s, d) = pk.dims4()?;
                        pk = pk.unsqueeze(2)?.repeat((1, 1, ratio, 1, 1))?.flatten(1, 2)?.contiguous()?;
                        
                        let (b_v, h_v, s_v, d_v) = pv.dims4()?;
                        pv = pv.unsqueeze(2)?.repeat((1, 1, ratio, 1, 1))?.flatten(1, 2)?.contiguous()?;
                    }
                }

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
        
        // [STRICT-IO] Ensure the target directory exists for this layer
        if let Some(parent) = file.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| anyhow!("Failed to create directory for layer {}: {}", self.layer_idx, e))?;
            }
        }

        let mut map = HashMap::new();

        if let Some((k, v)) = &self.kv_cache {
            // [HYBRID-PROTOCOL-MARKING-V9] High-Margin Identity Encoding
            let head_count = k.dim(1)?;
            let head_dim = k.dim(D::Minus1)?;
            let current_total_dim = head_count * head_dim;
            
            let (mut k_final, mut v_final) = if current_total_dim == 1024 && self.is_handshake_active {
                if self.layer_idx == 0 { println!("[HYBRID-SAVE] Physically upscaling Heads 8 -> 16 before SSD commit."); }
                let k_up = Tensor::cat(&[k, k], 1)?; 
                let v_up = Tensor::cat(&[v, v], 1)?; 
                (k_up, (v_up * 0.70710678118)?)
            } else {
                (k.clone(), v.clone())
            };

            let mut k_marked = k_final.to_dtype(DType::F32)?;
            let mut v_marked = v_final.to_dtype(DType::F32)?;
            
            let final_total_dim = k_final.dim(1)? * k_final.dim(D::Minus1)?;
            let is_stored_as_2b = final_total_dim >= 2048;

            let multiplier = if is_stored_as_2b { 1.0 } else { 2.0 };
            let trans_flag = if self.needs_transpose { 1.0 } else { 0.0 };
            
            // [V9-IDENTITY-MARGIN] 2B=4.0, 0.6B=1.0 (제안된 간격 벌리기 적용)
            let identity_id = if is_stored_as_2b { 4.0 } else { 1.0 };
            let role_id_k = 1.0; 
            let role_id_v = 0.0; 
            
            let marker_k = -(1.0 + (role_id_k * 0.1) + (identity_id * 0.01) + (multiplier * 0.001) + (trans_flag * 0.0001) + 0.00005);
            let marker_v = -(1.0 + (role_id_v * 0.1) + (identity_id * 0.01) + (multiplier * 0.001) + (trans_flag * 0.0001) + 0.00005);
            
            if self.layer_idx == 0 {
                println!("[HYBRID-ENCODER-V9] Layer 0 | ID={} (2B={}), Parts={}, Trans={}", 
                    if is_stored_as_2b { "2B" } else { "0.6B" }, is_stored_as_2b, multiplier, self.needs_transpose);
                println!("  -> K-Marker (Safe-V9): {:.6}", marker_k);
            }
            
            // [V8-ENCODING]
            let marker_k = -(1.0 + 0.1 + identity_id * 0.01 + (multiplier * 0.001) + (trans_flag * 0.0001) + 0.00005);
            let marker_v = -(1.0 + 0.0 + identity_id * 0.01 + (multiplier * 0.001) + (trans_flag * 0.0001) + 0.00005);
            
            let mut k_data = k_marked.flatten_all()?.to_vec1::<f32>()?;
            k_data[0] = marker_k as f32;
            k_marked = Tensor::from_vec(k_data, k_final.shape(), &Device::Cpu)?;

            let mut v_data = v_marked.flatten_all()?.to_vec1::<f32>()?;
            v_data[0] = marker_v as f32;
            v_marked = Tensor::from_vec(v_data, v_final.shape(), &Device::Cpu)?;

            let (k_anchors, k_packed, k_scales, k_shape) = self.compress_to_bitkv(&k_marked)?;
            let (v_anchors, v_packed, v_scales, _) = self.compress_to_bitkv(&v_marked)?;
            
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

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, active_status: bool) -> Result<()> {
        if active_status { self.is_handshake_active = true; }
        let file_path = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
        
        // [HYBRID-PROPAGATION-LOGIC]
        if !file_path.exists() {
            if self.layer_idx > 0 {
                let layer0_path = path.join("layer_0_kv.safetensors");
                if layer0_path.exists() {
                    if self.layer_idx == 1 { println!("[HYBRID-BRIDGE] Propagating Layer 0 knowledge to all 2B layers..."); }
                    return self.load_kv_from_file_internal(&layer0_path, device, expected_len, upscale_refill_len, self.is_handshake_active);
                }
            }
            return Ok(()); 
        }

        self.load_kv_from_file_internal(&file_path, device, expected_len, upscale_refill_len, self.is_handshake_active)
    }

    fn load_kv_from_file_internal(&mut self, file_path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, _active_status: bool) -> Result<()> {
        let file = std::fs::File::open(file_path)?;
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
            // Fallback to legacy dequantizers
            let deq_leg = |prefix: &str| -> Result<Tensor> {
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
            (deq_leg("k")?, deq_leg("v")?)
        };

        // [HYBRID-HANDSHAKE-V7] Role-Aware Restoration
        let engine_dtype = k.dtype();
        let k_f32 = k.to_dtype(DType::F32)?;
        let v_f32 = v.to_dtype(DType::F32)?;
        
        let k_vec = k_f32.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v_f32.flatten_all()?.to_vec1::<f32>()?;
        let sig_k = k_vec[0];
        let sig_v = v_vec[0];
        
        if sig_k < -1.0 {
            let val_k = sig_k.abs() - 1.0;
            
            // [V9-DECODING] 1번째 자리: 역할, 2번째 자리: 신분(4=2B, 1=0.6B)
            let role_k = ((val_k * 10.0).round() as usize) % 10;
            let identity_k = ((val_k * 100.0).round() as usize) % 10;
            let multiplier_raw = ((val_k * 1000.0).round() as usize) % 10;
            let trans_val = ((val_k * 10000.0).round() as usize) % 10;
            
            // [V9-THRESHOLD] 3을 기준으로 신분 판별 (4 vs 1의 안전한 중간지점)
            let is_actually_2b = identity_k >= 3;
            let needs_retranspose = trans_val == 1;
            
            // [HYBRID-V9-SPEC] 명시적 규격 강제
            let multiplier = if is_actually_2b { 1 } else { 2 };
            
            if self.layer_idx == 0 {
                println!("[HYBRID-V9] Verified: {} Mode (ID_Digit={}), Role_K={}, Mult={}, Trans={}", 
                    if is_actually_2b { "2B" } else { "0.6B" }, identity_k, role_k, multiplier, needs_retranspose);
            }

            // Marker Healing (K & V 둘 다 수행)
            let mut k_clean = k_vec.clone(); k_clean[0] = k_vec[1];
            let mut v_clean = v_vec.clone(); v_clean[0] = v_vec[1];
            
            let k_healed = Tensor::from_vec(k_clean, k_f32.shape(), &Device::Cpu)?.to_device(k_f32.device())?;
            let v_healed = Tensor::from_vec(v_clean, v_f32.shape(), &Device::Cpu)?.to_device(v_f32.device())?;
            
            self.is_handshake_active = true;

            // [STRICT-DIMENSION-ALIGNMENT-V3]
            let mut k_final = k_healed;
            let mut v_final = v_healed;
            
            let current_engine_dim = self.num_attention_heads * self.head_dim;
            let loaded_dim = k_final.dim(D::Minus1)?;

            if current_engine_dim == 2048 && loaded_dim == 1024 && !is_actually_2b {
                println!("[HYBRID-V7-FORCE] Upscaling 0.6B -> 2B (Heads 8 -> 16) for Layer {}", self.layer_idx);
                k_final = Tensor::cat(&[&k_final, &k_final], 1)?; // Dim 1 (Heads) 결합
                v_final = Tensor::cat(&[&v_final, &v_final], 1)?; // Dim 1 (Heads) 결합
                v_final = (v_final * 0.70710678118)?; // Thermal scaling
            }
            
            if needs_retranspose {
                k_final = k_final.transpose(D::Minus1, D::Minus2)?.contiguous()?;
                v_final = v_final.transpose(D::Minus1, D::Minus2)?.contiguous()?;
            }

            let actual_k_len = k_final.dim(2)?;
            let use_len = if expected_len == 0 { actual_k_len } else { expected_len.min(actual_k_len) };
            let final_len = use_len.saturating_sub(upscale_refill_len);

            if final_len > 0 {
                let k_out = k_final.narrow(2, 0, final_len)?.to_dtype(engine_dtype)?.contiguous()?;
                let v_out = v_final.narrow(2, 0, final_len)?.to_dtype(engine_dtype)?.contiguous()?;
                self.kv_cache = Some((k_out, v_out));
            }
            return Ok(());
        }

        let mut k_f32 = k_f32;
        let mut v_f32 = v_f32;
        
        let dims_raw = k_f32.dims4()?;
        if dims_raw.2 < dims_raw.3 && dims_raw.3 > 1000 {
            k_f32 = k_f32.transpose(2, 3)?.contiguous()?;
            v_f32 = v_f32.transpose(2, 3)?.contiguous()?;
        }

        let actual_k_len = k_f32.dim(2)?;
        let use_len = if expected_len == 0 { actual_k_len } else { expected_len.min(actual_k_len) };
        let final_len = use_len.saturating_sub(upscale_refill_len);

        if final_len > 0 {
            let k_out = k_f32.narrow(2, 0, final_len)?.to_dtype(engine_dtype)?.contiguous()?;
            let v_out = v_f32.narrow(2, 0, final_len)?.to_dtype(engine_dtype)?.contiguous()?;
            
            // [STRICT-IDENTITY-MAINTENANCE] 마커가 없더라도 데이터가 2B 규격이면 신분 유지
            if k_out.dim(1)? >= 16 { self.is_handshake_active = true; }
            
            self.kv_cache = Some((k_out, v_out));
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
        if layer_idx == 0 {
            eprintln!("[DEBUG-L0] DecoderLayer::new called. Hidden: {}, Baking: {}", config.hidden_size, baking_only);
            if let Ok(cwd) = std::env::current_dir() {
                eprintln!("[DEBUG-L0] CWD: {:?}", cwd);
            }
        }

        // [HYBRID-L0-INJECTION] If this is Layer 0 and we are in a hybrid setup
        // Try to load a superior Layer 0 from a 2B-sourced mini-GGUF
        let mut l0_mmap = None;
        let mut l0_cursor = None;
        let mut l0_content = None;
        
        let (mut final_ct, mut final_reader) = (ct, reader as &mut dyn SeekRead);

        // [HYBRID-L0-INJECTION-V2] Crucial for both 0.6B Baking AND 2B Inference
        // If Layer 0 is missing from the main GGUF (which is true for Body-L1-27), we MUST inject it.
        if layer_idx == 0 {
            let mut candidates = vec![
                std::path::PathBuf::from("src-tauri/models"), 
                std::path::PathBuf::from("models"), 
                std::path::PathBuf::from("../models"),
                std::path::PathBuf::from("../src-tauri/models"),
            ];

            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(exe_dir) = exe_path.parent() {
                    // target/debug/ -> src-tauri/models
                    candidates.push(exe_dir.join("../../src-tauri/models"));
                    // target/debug/ -> models
                    candidates.push(exe_dir.join("../../models"));
                }
            }
            
            let mut base_path = None;
            for p in candidates {
                if let Ok(cp) = std::fs::canonicalize(&p) {
                    base_path = Some(cp);
                    break;
                }
            }
            
            if let Some(bp) = base_path {
                let hybrid_dir = bp.join("Qwen3-VL-2B-Hybrid-gguf");
                let l0_filename = "Qwen3-2B-L0-VL-Q4_K_M.gguf";
                let l0_gguf_path = hybrid_dir.join(l0_filename);
                
                println!("[HYBRID-L0] Checking for L0 injection at: {:?}", l0_gguf_path);
                
                if l0_gguf_path.exists() {
                    if let Ok(file) = std::fs::File::open(&l0_gguf_path) {
                        if let Ok(mmap) = unsafe { memmap2::MmapOptions::new().map(&file) } {
                            l0_mmap = Some(mmap);
                            let mut cursor = std::io::Cursor::new(&l0_mmap.as_ref().unwrap()[..]);
                            if let Ok(content) = gguf_file::Content::read(&mut cursor) {
                                l0_content = Some(content);
                                l0_cursor = Some(cursor);
                                println!("[HYBRID-L0] INJECTING unified 2B Layer 0 Intelligence from {:?}", l0_filename);
                            }
                        }
                    }
                } else {
                    println!("[HYBRID-L0] L0 File not found.");
                }
            } else {
                println!("[HYBRID-L0] Could not resolve models directory.");
            }
        }

        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        // [HYBRID-L0-INJECTION] Unified intelligence entry
        let mut final_config = config.clone();
        if config.hidden_size == 1024 {
            println!("[HYBRID-L0] Folding 2B Layer 0 Intelligence (2048 -> 1024) for 0.6B engine.");
        } else if config.hidden_size == 2048 {
            println!("[HYBRID-L0] Injecting native 2B Layer 0 Intelligence for Inference.");
            // Ensure 16 heads for 2048 hidden size
            final_config.num_attention_heads = 16;
        }

        let self_attn = if let (Some(c), Some(r)) = (&l0_content, &mut l0_cursor) {
            QuantizedQwen3VLTextAttention::new(&final_config, c, r, &attn_base, is_gguf_naming, device, dtype, layer_idx)?
        } else {
            QuantizedQwen3VLTextAttention::new(&final_config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx)?
        };
        
        // [PHYSICAL-LOGIC-SEPARATION]
        // baking_only여도 0번 레이어는 MLP까지 모두 로드하여 진짜 지능을 구움
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !baking_only || layer_idx == 0 {
            let gate_name = format!("{base_name}.{gate}");
            let up_name = format!("{base_name}.{up}");
            let down_name = format!("{base_name}.{down}");
            
            let (mg, mu, md) = if let (Some(c), Some(r)) = (&l0_content, &mut l0_cursor) {
                (
                    get_qlinear_v2(c, r, &gate_name, device, dtype, config.hidden_size)?,
                    get_qlinear_v2(c, r, &up_name, device, dtype, config.hidden_size)?,
                    get_qlinear_v2(c, r, &down_name, device, dtype, config.hidden_size)?
                )
            } else {
                (
                    get_qlinear_v2(ct, reader, &gate_name, device, dtype, config.hidden_size)?,
                    get_qlinear_v2(ct, reader, &up_name, device, dtype, config.hidden_size)?,
                    get_qlinear_v2(ct, reader, &down_name, device, dtype, config.hidden_size)?
                )
            };
            
            let pln = if let (Some(c), Some(r)) = (&l0_content, &mut l0_cursor) {
                get_rms_norm(c, r, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype, config.hidden_size)?
            } else {
                get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype, config.hidden_size)?
            };
            
            (Some(mg), Some(mu), Some(md), Some(pln))
        } else {
            (None, None, None, None)
        };

        let input_layernorm = if let (Some(c), Some(r)) = (&l0_content, &mut l0_cursor) {
            get_rms_norm(c, r, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype, config.hidden_size)?
        } else {
            get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype, config.hidden_size)?
        };

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
        
        // [JIT-VERTICAL-TRANSFER] 
        // If this layer lacks a KV cache but we have a context offset, 
        // load the baked Layer 0 memory immediately onto the current device.
        if self.self_attn.kv_cache.is_none() {
             let base_path = crate::utils::paths::get_kv_dir(None);
             // We use a specific naming convention for the baked relay snapshot
             // This needs to be coordinated with the scheduler's task ID.
             // For now, we check if any suitable Layer 0 snapshot exists to "jump-start" the intelligence.
             // (Logic simplified: if seqlen_offset is handled by the caller, we ensure cache residency here)
        }

        // [ROTATION-SYNC] Ensure KV Cache follows the layer's device
        if let Some((k, v)) = &mut self.self_attn.kv_cache {
            if !k.device().same_device(dev) {
                *k = k.to_device(dev)?;
                *v = v.to_device(dev)?;
            }
        }

        if self.self_attn.layer_idx == 0 {
            println!("[DEBUG-LAYER-IN] xs shape: {:?}", xs.shape());
        }
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

        let norm_dim = self.input_layernorm.weight().dim(0)?;
        let input_dim = xs.dim(candle_core::D::Minus1)?;
        
        // [HYBRID-BRIDGE-V7] Handshake 프로토콜 기반의 완벽한 분기
        // [RUNTIME-IDENTITY-RECOVERY] 데이터 차원을 보고 자신의 신분을 실시간으로 교정
        if input_dim == 2048 && norm_dim == 1024 {
            self.self_attn.is_handshake_active = true;
        }
        
        let is_handshake_active = self.self_attn.is_handshake_active;
        let needs_bridge = !is_handshake_active && input_dim == 2048 && norm_dim == 1024;
        
        let mut xs_active = if needs_bridge {
            if self.self_attn.layer_idx == 0 { 
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| println!("[HYBRID-BRIDGE] Baking Engine Mode - Slicing 2048 -> 1024."));
            }
            xs.narrow(candle_core::D::Minus1, 0, 1024)?.contiguous()?
        } else {
            xs.clone()
        };

        let residual_active = xs_active.clone();
        let xs_active = self.input_layernorm.forward(&xs_active)?;
        let mut xs_active = self.self_attn.forward(&xs_active, &cos, &sin, attention_mask.as_ref())?;
        
        let mut xs_active = if xs_active.dtype() != residual_active.dtype() { xs_active.to_dtype(residual_active.dtype())? } else { xs_active };
        let mut xs = residual_active.add(&xs_active)?;
        
        // [HYBRID-BRIDGE-BACK-V7-DISABLED] 
        // 베이킹 중 레이어 내부 확장은 불필요하며 에러의 원인이 되므로 비활성화
        // 확장은 오직 save_kv_cache 프로토콜(V7)에 의해서만 수행됨

        // [STRICT-DIMENSION-CHECK]
        if xs.dim(D::Minus1)? > 2048 {
             println!("[CRITICAL-WARN] Unexpected dimension expansion: {}. Check Handshake V7 status.", xs.dim(D::Minus1)?);
        }
        
        // [OPTIMIZATION] Skip MLP block if not available (MLP 0% Mode)
        if let (Some(gate_proj), Some(up_proj), Some(down_proj), Some(post_norm)) = (&self.mlp_gate, &self.mlp_up, &self.mlp_down, &self.post_attention_layernorm) {
            let residual_mlp = xs.clone();
            
            // 1. Post-Attention Norm
            let mut xs_mlp = post_norm.forward(&xs)?;
            
            // [STRICT-MLP-ENTRANCE-BRIDGE] 베이킹 모드 시 2048 -> 1024 Slicing 강제
            if !is_handshake_active && norm_dim == 1024 && xs_mlp.dim(D::Minus1)? == 2048 {
                xs_mlp = xs_mlp.narrow(D::Minus1, 0, 1024)?.contiguous()?;
            }
            
            // 2. Gate & Up Projections
            let (gate, up) = {
                let g = gate_proj.forward(&xs_mlp)?;
                let u = up_proj.forward(&xs_mlp)?;
                (g, u)
            };
            
            let gate = candle_nn::ops::silu(&gate)?;
            let hidden = gate.mul(&up)?;
            
            // 3. Down Projection
            let out = down_proj.forward(&hidden)?;
            
            // [HYBRID-MLP-EXIT-V7-DISABLED] 내부 확장 비활성화
            
            let out = if out.dtype() != residual_mlp.dtype() { out.to_dtype(residual_mlp.dtype())? } else { out };
            Ok(residual_mlp.add(&out)?)
        } else {
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

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, active_status: bool) -> Result<()> {
        self.self_attn.load_kv_cache(path, device, expected_len, upscale_refill_len, active_status)
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
    pub is_disk_swap: bool, // [NEW] SSD-Assisted GPU Inference mode
    pub active_session_id: Option<String>, // [NEW] Disk workspace ID
    pub pinned_layer_count: usize, // [NEW] How many layers to keep in VRAM
    pub current_kv_len: usize, // [NEW] Logical progress tracker (SSD-persistent)
    pub is_handshake_active: bool, // [NEW] Global hybrid state
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
        is_disk_swap: bool, // [NEW]
    ) -> Result<Self> {
        println!("[MODEL-INIT-DEBUG] Name: {}, Hidden: {}, Layers: {}, TextMode: {}, DiskSwap: {}", base_name, config.hidden_size, config.num_hidden_layers, is_text, is_disk_swap);
        let is_forced_cpu = device.is_cpu();
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);

        // [DETECTION-FIRST] Determine actual hidden size from GGUF BEFORE initializing anything
        let mut actual_h_size = if let Some(info) = ct.tensor_infos.get(&format!("{base_name}.attn_norm.weight")) {
            info.shape.dims()[0]
        } else if let Some(info) = ct.tensor_infos.get(&format!("blk.0.attn_norm.weight")) {
            info.shape.dims()[0]
        } else if let Some(info) = ct.tensor_infos.get(&"token_embd.weight".to_string()) {
            info.shape.dims()[1]
        } else {
            config.hidden_size
        };

        // [STRICT-2B-OVERRIDE] If we are loading a 2B variant, force 2048 to prevent 0.6B ghosting
        if actual_h_size == 1024 && !baking_only && config.hidden_size == 2048 {
            println!("[MODEL-INIT] Detected 1024 in 2B config. Overriding to 2048 for Inference.");
            actual_h_size = 2048;
        }

        println!("[MODEL-INIT] Name: {}, Config Hidden: {}, GGUF Actual: {}, Layers: {}", base_name, config.hidden_size, actual_h_size, config.num_hidden_layers);

        let mut patched_config_owned = config.clone();
        patched_config_owned.hidden_size = actual_h_size;
        let config = &patched_config_owned;

        

                // [EXHAUSTIVE-SEARCH] Search for any key that looks like an embedding

                let mut embed_key = None;

                

                // Priority 1: token_embd.weight (Found in Instruct models)

                if ct.tensor_infos.contains_key("token_embd.weight") {

                    embed_key = Some("token_embd.weight".to_string());

                } else {

                    // Priority 2: Other variants

                    for key in ct.tensor_infos.keys() {

                        let k_low = key.to_lowercase();

                        if k_low.contains("token_embd") || (k_low.contains("embed_tokens") && k_low.contains("weight")) || k_low == "model.embed_tokens" {

                            embed_key = Some(key.clone());

                            break;

                        }

                    }

                }

        

                // [HYBRID-EMBEDDING-RESOLVER] Unified intelligence entry for 0.6B and 2B
                let mut tensor_opt = None;
                if let Some(key) = embed_key {
                     println!("[HYBRID] Found embedding weight in main GGUF: {}", key);
                     tensor_opt = Some(ct.tensor(&mut reader, &key, device)?.dequantize(device)?.to_dtype(dtype)?);
                }

                // [HYBRID-STRATEGY] If missing (Body-L1-27 mode) or if we want 2B precision for Baking
                if tensor_opt.is_none() || (actual_h_size == 1024 && baking_only) {
                    let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models")).unwrap();
                    let hybrid_dir = base_path.join("Qwen3-VL-2B-Hybrid-gguf");
                    let b2_gguf_path = hybrid_dir.join("Qwen3-2B-L0-VL-Q4_K_M.gguf");
                    
                    if b2_gguf_path.exists() {
                        if let Ok(b2_file) = std::fs::File::open(&b2_gguf_path) {
                            if let Ok(b2_mmap) = unsafe { memmap2::MmapOptions::new().map(&b2_file) } {
                                let mut b2_cursor = std::io::Cursor::new(&b2_mmap[..]);
                                if let Ok(b2_ct) = gguf_file::Content::read(&mut b2_cursor) {
                                    if let Ok(b2_emb) = b2_ct.tensor(&mut b2_cursor, "token_embd.weight", device) {
                                        let b2_emb_t = b2_emb.dequantize(device)?.to_dtype(dtype)?;
                                        
                                        if actual_h_size == 1024 {
                                            tensor_opt = Some(b2_emb_t.narrow(1, 0, 1024)?.contiguous()?);
                                            println!("[HYBRID-INJECT-SUCCESS] 2B Embedding Truncated (2048 -> 1024) injected for Baking.");
                                        } else if actual_h_size == 2048 {
                                            tensor_opt = Some(b2_emb_t);
                                            println!("[HYBRID-INJECT-SUCCESS] Unified 2B Embedding injected for Inference.");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let embed_tokens = if let Some(tensor) = tensor_opt {
                     let h = tensor.dim(1)?;
                     Embedding::new(tensor, h)
                } else {
             println!("[HYBRID-RELAY] Embedding NOT found. Attempting shared fallback...");
             let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models")).unwrap();
             let shared_emb_path = base_path.join("qwen3_shared_emb.safetensors");
             
             let borrowed_emb = if shared_emb_path.exists() {
                 let st = candle_core::safetensors::load(shared_emb_path, device)?;
                 if let Some(s_t) = st.get("token_embd.weight") {
                     let s_t = s_t.to_dtype(dtype)?;
                     // 0.6B 베이킹 시에는 자기 규격(1024) 그대로 사용
                     if actual_h_size == 1024 {
                         s_t
                     } else {
                         // 2B 추론 시에는 2048로 확장 (2b_specs.json 명시적 지원)
                         if s_t.dim(1)? == 1024 && actual_h_size == 2048 {
                            println!("[HYBRID-RELAY] Upscaling Shared Embedding 1024 -> 2048 for 2B Model.");
                            Tensor::cat(&[&(s_t.clone()), &s_t], 1)?
                         } else {
                            s_t
                         }
                     }
                 } else {
                     Tensor::zeros((config.vocab_size, actual_h_size), dtype, device)?
                 }
             } else {
                 Tensor::zeros((config.vocab_size, actual_h_size), dtype, device)?
             };
             Embedding::new(borrowed_emb, actual_h_size)
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
                
                // [HYBRID-INDEX-OFFSET-V3]
                // Body GGUF 파일 내부에 'blk.0'이 있더라도 무조건 무시하고 blk.1부터 바디로 인식함
                let max_file_idx = ct.tensor_infos.keys()
                    .filter(|k| k.contains("blk."))
                    .filter_map(|k| k.split('.').nth(1)?.parse::<usize>().ok())
                    .max().unwrap_or(26);

                let is_hybrid_body = ct.tensor_infos.keys().any(|k| k.contains("blk.1."));
                
                let actual_file_idx = if layer_idx > 0 {
                     if is_hybrid_body { 
                         // 1번 레이어부터 로드 (0번 레이어는 주입됨)
                         (layer_idx + 1).min(max_file_idx) 
                     } else { 
                         layer_idx 
                     }
                } else {
                    layer_idx
                };

                let standard = format!("{base_name}.layers.{actual_file_idx}");
                let gguf_blk = format!("blk.{actual_file_idx}");
                let mut prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };
                
                // [HYBRID-INJECTION-ALIGN] Layer 0는 항상 주입된 파일의 blk.0를 사용함
                if layer_idx == 0 { prefix = "blk.0".to_string(); }
                
                // [STRICT-INFERENCE-FORCE] 2B 추론 모드인 경우 레이어 설정을 2048로 강제 고정
                let mut layer_config = final_config.clone();
                if !baking_only && actual_h_size == 2048 {
                    layer_config.hidden_size = 2048;
                }

                // [CRITICAL] Pass the forced layer_config
                QuantizedQwen3VLTextDecoderLayer::new(
                    &layer_config, ct, &mut local_cursor, &prefix, &layer_device, layer_dtype, layer_idx, baking_only
                )
            }).collect()
        });
        
        let layers = layers?;
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let last_device = layers.last().map(|l| l.device()).unwrap_or(device);
        let norm_dtype = if last_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, &mut reader, norm_prefix, config.rms_norm_eps, last_device, norm_dtype, config.hidden_size)?;
        
        let head_dim = config.head_dim;
        // [HYBRID-ROPE-SYNC] Force 5M theta for both 0.6B and 2B to align rotation phases
        let actual_rope_theta = if config.hidden_size == 1024 || config.rope_theta < 1000001.0 { 5000000.0 } else { config.rope_theta };
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, actual_rope_theta);
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        
        Ok(Self { 
            embed_tokens, 
            layers, 
            norm, 
            rotary_emb, 
            mrope_section, 
            mmap: mmap_handle, 
            baking_only, 
            is_forced_cpu, 
            is_disk_swap, 
            active_session_id: None, 
            pinned_layer_count: 0, 
            current_kv_len: 0, 
            is_handshake_active: false,
            is_text, 
            is_image 
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
        is_text: bool,
        is_image: bool,
        is_disk_swap: bool, // [NEW]
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let (raw_embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (tensor, h)
        } else if let Ok(tensor) = ct.tensor(reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (tensor, h)
        } else {
             return Err(anyhow!("Failed to load embedding."));
        };

        let embed_tokens = Embedding::new(raw_embed_tokens, actual_hidden_size);

        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;

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
            let mut layer_device = current_device.clone();
            if layer_device.is_cuda() && is_vram_checked {
                 // For the first layer, always use full weight cost
                 let actual_cost = if layer_idx == 0 { layer_weight_size } else { cost_per_layer };
                 if simulated_free_vram > ( (actual_cost as f64 * 1.05) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(actual_cost);
                 } else {
                     layer_device = Device::Cpu;
                 }
            }

            let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
            let standard = format!("{base_name}.layers.{layer_idx}");
            let gguf_blk = format!("blk.{layer_idx}");
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };

            // [STRICT-INFERENCE-FORCE] 2B 추론 모드인 경우 레이어 설정을 2048로 강제 고정
            let mut layer_config = config.clone();
            if !baking_only && actual_hidden_size == 2048 {
                layer_config.hidden_size = 2048;
            }

            // [CRITICAL] Pass the isolated device and forced config
            let layer = QuantizedQwen3VLTextDecoderLayer::new(
                &layer_config, ct, reader, &prefix, &layer_device, layer_dtype, layer_idx, baking_only
            )?;
            layers.push(layer)
        }
        
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) { alt_norm } else { &norm_name };
        let norm_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, &current_device, norm_dtype, config.hidden_size)?;
        let head_dim = config.head_dim;
        // [HYBRID-ROPE-SYNC] Force 5M theta for both 0.6B and 2B to align rotation phases
        let actual_rope_theta = if config.hidden_size == 1024 || config.rope_theta < 1000001.0 { 5000000.0 } else { config.rope_theta };
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, actual_rope_theta);
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
            is_disk_swap,
            active_session_id: None,
            pinned_layer_count: 0,
            current_kv_len: 0,
            is_handshake_active: false,
            is_text,
            is_image,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        total_len: usize,
        position_ids: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        
        // [PROGRESS-CALC]
        let current_pos = seqlen_offset + seq_len;
        let progress = if total_len > 0 {
            (current_pos as f32 / total_len as f32 * 100.0).min(100.0)
        } else {
            0.0
        };

        // [INFERENCE-MONITOR] Real-time System Status
        use nvml_wrapper::Nvml;
        use chrono::Local;
        if let Ok(nvml) = Nvml::init() {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(mem) = dev.memory_info() {
                    let now = Local::now().format("%H:%M:%S%.3f");
                    println!("[STAT] Time: {} | VRAM: {}MB Used / {}MB Free | Context: {}/{} tokens ({:.0}%)", 
                        now, mem.used / 1024 / 1024, mem.free / 1024 / 1024, current_pos, total_len, progress);
                }
            }
        }

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
        
        // [DYNAMIC-DEVICE-DETECTION]
        // Don't just rely on is_forced_cpu. Check where the layers actually are.
        // If Layer 0 is on GPU, we should perform the computation on GPU.
        let actual_compute_on_gpu = self.layers.first().map(|l| l.device().is_cuda()).unwrap_or(false);
        
        let target_dtype = if actual_compute_on_gpu { DType::BF16 } else { DType::F32 };
        let target_device = if actual_compute_on_gpu { Device::new_cuda(0)? } else { Device::Cpu };
        let gpu_device = Device::new_cuda(0)?;

        let mut xs = inputs_embeds.to_device(&target_device)?.to_dtype(target_dtype)?.contiguous()?;

        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            target_dtype,
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

        let total_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if layer_idx % 7 == 0 || layer_idx == total_layers - 1 {
                println!("[TRACE-LAYER] {}/{} ({:.0}%) | Device: {:?}", layer_idx + 1, total_layers, progress, layer.device());
            }
            let is_on_cpu = layer.device().is_cpu();
            let is_pinned = layer_idx < self.pinned_layer_count;
            
            // [SSD-ASSISTED-SWAP-LOGIC: CRITERION 2]
            // Only load from SSD if NOT pinned and disk_swap is active
            if self.is_disk_swap && !is_pinned {
                if let Some(session_id) = &self.active_session_id {
                    let kv_path = crate::utils::paths::get_kv_dir(None).join(session_id);
                    if kv_path.exists() {
                        // For non-pinned layers, we load JIT
                        let _ = layer.load_kv_cache(&kv_path, &gpu_device, 0, 0, self.is_handshake_active);
                    }
                }
            }

            // [SMART-STREAMING]
            if is_on_cpu { layer.to_device(&gpu_device)?; }
            
            // [GLOBAL-HANDSHAKE-SYNC] 개별 레이어가 SSD에서 오며 잃어버린 신분을 모델 레벨에서 강제 복구
            layer.self_attn.is_handshake_active = self.is_handshake_active;
            
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            
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

            // [SSD-ASSISTED-SWAP-LOGIC: CRITERION 2]
            // Dump and clear VRAM only for non-pinned layers
            if self.is_disk_swap && !is_pinned {
                if let Some(session_id) = &self.active_session_id {
                    let kv_path = crate::utils::paths::get_kv_dir(None).join(session_id);
                    // Save and clear KV to keep VRAM lean
                    let _ = layer.save_kv_cache(&kv_path, true, 1024);
                }
            }

            if is_on_cpu { layer.to_device(&Device::Cpu)?; }
        }
        
        // [FIX] Successfully advanced! Update logical progress
        self.current_kv_len = seqlen_offset + seq_len;

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
        self.current_kv_len = 0; // [FIX] Also reset tracker
    }

    pub fn get_kv_len(&self) -> usize {
        self.current_kv_len // [FIX] Return logical progress, not tensor dimension
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

                if i == 0 {
                    println!("[DEBUG-RELAY-MEM] Layer 0 Raw: Shape={:?}", k_small.shape());
                    println!("[DEBUG-RELAY-MEM] H={}, D={}. Target: H={}, D={}", h, d, target_heads, target_dim);
                }

                let mut k_aligned = if d < target_dim { 
                    if i == 0 { println!("[DEBUG-RELAY-MEM] Upscaling Dim {} -> {}", d, target_dim); }
                    Tensor::cat(&[&k_small, &k_small], D::Minus1)? 
                } else { k_small };
                
                let mut v_aligned = if d < target_dim { 
                    Tensor::cat(&[&v_small, &v_small], D::Minus1)? 
                } else { v_small };

                if h != target_heads {
                    if i == 0 { println!("[DEBUG-RELAY-MEM] Upscaling Heads {} -> {}", h, target_heads); }
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
        // [STRICT-IO] Ensure the directory exists with absolute path reliability
        let mut final_path = path.to_path_buf();
        if !final_path.is_absolute() {
            if let Ok(current) = std::env::current_dir() {
                final_path = current.join(path);
            }
        }

        if !final_path.exists() {
            std::fs::create_dir_all(&final_path)
                .map_err(|e| anyhow!("Failed to create KV directory {:?}: {}", final_path, e))?;
        }
        
        println!("[SSD-BRIDGE] Saving {} layers to ABSOLUTE directory: {:?}", self.layers.len(), final_path);
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // [STRICT-ALIGN] Use consistent filename 'layer_{}_kv.safetensors' implicitly by passing the directory
            layer.save_kv_cache(&final_path, clear, block_size)
                .map_err(|e| anyhow!("Failed to save layer {} to {:?}: {}", i, final_path, e))?;
        }

        // [DISK-BRAIN] Save logical progress and Handshake state to JSON for OOM recovery
        let metadata_path = final_path.join("metadata.json");
        let metadata = serde_json::json!({
            "current_kv_len": self.current_kv_len,
            "is_handshake_active": self.is_handshake_active,
            "timestamp": chrono::Utc::now().timestamp()
        });
        if let Ok(file) = std::fs::File::create(&metadata_path) {
            let _ = serde_json::to_writer(file, &metadata);
            println!("[DISK-BRAIN] Progress and Handshake state ({}) saved to SSD.", self.is_handshake_active);
        }

        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        if !path.exists() { return Ok(()); }
        
        let session_id = path.file_name().unwrap().to_string_lossy().to_string();
        // [ISOLATION-FIX] Use a separate sub-folder for inference-time swapping to protect original seeds
        let inference_session_id = format!("{}/inference", session_id);
        self.active_session_id = Some(inference_session_id.clone());

        // [ZERO-PREFILL-V2] Prioritize inference progress over baked snapshot
        let inference_path = crate::utils::paths::get_kv_dir(None).join(&inference_session_id);
        let l0_inference_file = inference_path.join("layer_0_kv.safetensors");
        let l0_baked_file = path.join("layer_0_kv.safetensors");
        
        let (actual_load_path, is_resuming_inference) = if l0_inference_file.exists() {
            (inference_path, true)
        } else if l0_baked_file.exists() {
            (path.to_path_buf(), false)
        } else {
            return Err(anyhow!("No KV seed found at {:?} or {:?}", l0_baked_file, l0_inference_file));
        };

        if is_resuming_inference {
            println!("[ZERO-PREFILL-RESUME] Found ongoing progress in {:?}. Resuming...", actual_load_path);
        } else {
            println!("[ZERO-PREFILL-START] Starting from baked snapshot at {:?}.", actual_load_path);
        }

        // [DISK-BRAIN-RECALL] Restore logical progress from JSON
        let metadata_path = actual_load_path.join("metadata.json");
        if metadata_path.exists() {
            if let Ok(file) = std::fs::File::open(&metadata_path) {
                let meta: serde_json::Value = serde_json::from_reader(file).unwrap_or(serde_json::json!({}));
                if let Some(len) = meta.get("current_kv_len").and_then(|v| v.as_u64()) {
                    self.current_kv_len = len as usize;
                    println!("[DISK-BRAIN] Recalled progress from SSD: {} tokens.", self.current_kv_len);
                }
                if let Some(active) = meta.get("is_handshake_active").and_then(|v| v.as_bool()) {
                    self.is_handshake_active = active;
                    println!("[DISK-BRAIN] Restored Handshake state: {}", self.is_handshake_active);
                }
            }
        }

        let mut free_vram = 0;
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(mem) = dev.memory_info() { free_vram = mem.free; }
            }
        }

        let safety_margin = 800 * 1024 * 1024;
        let available_for_kv = free_vram.saturating_sub(safety_margin);
        let layer_kv_cost = 40 * 1024 * 1024; 
        
        let can_pin = if layer_kv_cost > 0 { (available_for_kv / layer_kv_cost) as usize } else { 0 };
        self.pinned_layer_count = if self.is_disk_swap { can_pin.min(self.layers.len()) } else { self.layers.len() };

        // [ZERO-PREFILL-BRIDGE] Load the primary seed layer (Layer 0)
        self.layers[0].load_kv_cache(&actual_load_path, device, expected_len, upscale_refill_len, self.is_handshake_active)?;
        
        // [FIX] Update logical progress and Handshake state immediately
        self.current_kv_len = self.layers[0].get_kv_len();
        self.is_handshake_active = self.layers[0].self_attn.is_handshake_active;
        let is_restored = self.is_handshake_active;

        // [STRICT-IDENTITY-COMMIT] 만약 레이어 0 로드 후 신분이 복원되었다면, 이를 디스크에 즉시 커밋
        // 이렇게 해야 DiskSwap으로 나중에 로드될 레이어들이 업데이트된 신분을 읽을 수 있음
        if is_restored {
            let metadata_path = actual_load_path.join("metadata.json");
            let metadata = serde_json::json!({
                "current_kv_len": self.current_kv_len,
                "is_handshake_active": true,
                "timestamp": chrono::Utc::now().timestamp()
            });
            if let Ok(file) = std::fs::File::create(&metadata_path) {
                let _ = serde_json::to_writer(file, &metadata);
                println!("[DISK-BRAIN] Handshake success COMMITTED to metadata.json.");
            }
        }
        
        let (k_seed, v_seed) = self.layers[0].self_attn.kv_cache.as_ref()
            .ok_or_else(|| anyhow!("Failed to load seed KV cache from {:?}", actual_load_path))?.clone();

        // Create inference workspace
        let inference_workspace = crate::utils::paths::get_kv_dir(None).join(self.active_session_id.as_ref().unwrap());
        if !inference_workspace.exists() {
            std::fs::create_dir_all(&inference_workspace)?;
        }

        for i in 0..self.layers.len() {
            let is_pinned = i < self.pinned_layer_count;
            
            if i > 0 {
                // [STRICT-UPScale-PROPAGATION] 레이어 0에서 복원된 2048 데이터를 모든 레이어에 전파
                self.layers[i].self_attn.kv_cache = Some((k_seed.clone(), v_seed.clone()));
                self.layers[i].self_attn.is_handshake_active = is_restored;
            }

            if self.is_disk_swap {
                // [STRICT-DISK-SYNC] 복원된 2048 데이터를 SSD에 즉시 커밋
                // 이렇게 해야 나중에 레이어가 개별적으로 로드될 때 1024가 아닌 2048을 읽음
                let layer_name = format!("layer_{}_kv.safetensors", i);
                let layer_file_path = inference_workspace.join(layer_name);
                
                // 만약 핀에 꽂혀있지 않다면 저장 후 메모리 해제하여 VRAM 확보
                self.layers[i].save_kv_cache(&inference_workspace, !is_pinned, 1024)?;
            }
        }

        if self.is_disk_swap {
            println!("[SSD-BRIDGE] Zero-Prefill + Isolation Active. Pinning {}/{} layers. Swap workspace: {:?}", 
                self.pinned_layer_count, self.layers.len(), inference_workspace);
        }
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

    pub fn rebalance_layers(&mut self, device_id: usize, context_len: usize) -> Result<()> {
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

        if free_vram == 0 { return Ok(()); }

        // [DYNAMIC-RECOVERY] 
        // If we were forced to CPU, be EXTREMELY conservative about going back to GPU.
        // We need enough space for weights AND the massive KV cache (1.1GB).
        let kv_cache_size = (context_len as u64) * 110_592;
        let min_vram_to_recovery = 3_500_000_000 + kv_cache_size; 
        
        let can_rebalance = if self.is_forced_cpu {
            free_vram > min_vram_to_recovery // Only recover if we have near-total free VRAM
        } else {
            free_vram > 2_500_000_000
        };
        
        if !can_rebalance { return Ok(()); }
        let total_safety_margin = 400_000_000 + kv_cache_size; 
        
        let layer_size = 75_000_000; 
        let available_for_weights = free_vram.saturating_sub(total_safety_margin);
        
        if free_vram < (total_safety_margin + 100_000_000) {
            // If VRAM is too tight, evict layers
            let needed = (total_safety_margin + 200_000_000).saturating_sub(free_vram);
            let layers_to_evict = (needed / layer_size as u64) as usize + 1;
            
            let mut evicted = 0;
            for i in (0..self.layers.len()).rev() {
                if evicted >= layers_to_evict { break; }
                if self.layers[i].device().is_cuda() {
                    self.layers[i].to_device(&Device::Cpu)?;
                    evicted += 1;
                }
            }
            return Ok(());
        }

        let target_device = Device::new_cuda(device_id)?;
        let max_gpu_layers = (available_for_weights / layer_size as u64) as usize;
        let max_gpu_layers = max_gpu_layers.min(self.layers.len());

        let mut pinned_count = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let target = if i < max_gpu_layers {
                pinned_count += 1;
                &target_device
            } else {
                &Device::Cpu
            };

            if !layer.device().same_device(target) {
                layer.to_device(target)?;
                
                // [STRICT-SYNC] Force move the KV cache as well
                if let Some((k, v)) = &mut layer.self_attn.kv_cache {
                    *k = k.to_device(target)?;
                    *v = v.to_device(target)?;
                }
            }
        }
        
        if pinned_count > 0 {
            println!("[PINNING-REPORT] GPU Pinned: {}/{} Layers. (Context: {} tokens, KV: {}MB, Free: {}MB)", 
                pinned_count, self.layers.len(), context_len, kv_cache_size / 1_000_000, free_vram / 1_000_000);
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
        is_disk_swap: bool, // [NEW]
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
            &t_config, ct_main, main_mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image, is_disk_swap
        )?;

        let main_mmap = main_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader_main = std::io::Cursor::new(main_mmap);
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear_v2(ct_main, &mut reader_main, "lm_head", text_device, head_dtype, language_model.embed_tokens.hidden_size()) {
            l
        } else if let Ok(l) = get_qlinear_v2(ct_main, &mut reader_main, "output", text_device, head_dtype, language_model.embed_tokens.hidden_size()) {
            l
        } else {
            get_qlinear_v2(ct_main, &mut reader_main, "token_embd", text_device, head_dtype, language_model.embed_tokens.hidden_size())?
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
        is_disk_swap: bool, // [NEW]
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

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image, is_disk_swap)?;
        
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
        let (mut image_embeds, deepstack_image_embeds) = self.visual.forward(&pixel_values, &image_grid_thw)?;
        
        // [HYBRID-VISION-FOLDING] 2B Vision (2048) -> 0.6B Engine (1024)
        if image_embeds.dim(candle_core::D::Minus1)? == 2048 && self.language_model.embed_tokens.hidden_size() == 1024 {
            println!("[HYBRID-VISION] Truncating 2B Vision features (2048 -> 1024) for 0.6B engine.");
            image_embeds = image_embeds.narrow(candle_core::D::Minus1, 0, 1024)?.contiguous()?;
        }

        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let split_sizes: Vec<usize> = prod_tensor_last_dim(&image_grid_thw)?.to_vec1::<u32>()?.iter().map(|&x| x as usize / spatial_merge_size.pow(2)).collect();
        let image_embeds = image_embeds.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter().map(|t| {
            let mut t = t.to_device(&self.text_device)?;
            // Also fold deepstack features if they are 2048
            if t.dim(candle_core::D::Minus1)? == 2048 && self.language_model.embed_tokens.hidden_size() == 1024 {
                t = t.narrow(candle_core::D::Minus1, 0, 1024)?.contiguous()?;
            }
            Ok(t)
        }).collect();
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

    pub fn forward(&mut self, input_ids_in: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _pixel_values_video: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids_in.dims2()?;
        
        // [SYNC-SESSION]
        self.language_model.active_session_id = session_id;
        
        // [DYNAMIC-RECOVERY] Periodically check for GPU resource availability
        let _ = self.language_model.rebalance_layers(0, seqlen_offset + seq_len);

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;

        // [STRICT-EMBED-ALIGN] 로딩 시점에 이미 2B 임베딩이 주입되었으므로, 여기서 추가 확장은 하지 않음
        let mut inputs_embeds = inputs_embeds;

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
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, block_size) }
    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> { self.language_model.offload_kv_cache(path, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; self.text_device = device.clone(); self.vision_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, context_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, context_len) }
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
    pub fn new_with_mmap(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool, is_text: bool, is_image: bool, is_disk_swap: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {}, DiskSwap: {})", baking_only, single_layer_mode, is_disk_swap);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        
        // [HYBRID-ALIGN] Ensure 1024 dim for baking mode at the entry point
        if baking_only && t_config.hidden_size == 2048 {
            println!("[HYBRID-ENTRY] Adjusting entry config for 0.6B Baking (2048 -> 1024)");
            t_config.hidden_size = 1024;
        }
        
        if single_layer_mode { t_config.num_hidden_layers = 1; }
        
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(&t_config, ct_main, mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image, is_disk_swap)?;
        let lm_head = if !baking_only {
            let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut reader = std::io::Cursor::new(mmap);
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear_v2(ct_main, &mut reader, "lm_head", text_device, head_dtype, language_model.embed_tokens.hidden_size()) { Some(l) }
            else if let Ok(l) = get_qlinear_v2(ct_main, &mut reader, "output", text_device, head_dtype, language_model.embed_tokens.hidden_size()) { Some(l) }
            else { get_qlinear_v2(ct_main, &mut reader, "token_embd", text_device, head_dtype, language_model.embed_tokens.hidden_size()).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: mmap_handle, is_text, is_image })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, reader_main: &mut R, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, single_layer_mode: bool, is_text: bool, is_image: bool, is_disk_swap: bool) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text (Baking-Only: {}, Single-Layer: {}, DiskSwap: {})", baking_only, single_layer_mode, is_disk_swap);
        let mut t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?.clone();
        if single_layer_mode { t_config.num_hidden_layers = 1; }

        let language_model = QuantizedQwen3VLTextModel::new(&t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, baking_only, is_text, is_image, is_disk_swap)?;
        let lm_head = if !baking_only {
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: None, is_text, is_image })
    }

    pub fn forward(&mut self, input_ids_in: &Tensor, cache_position_in: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids_in.dims2()?;
        
        // [SYNC-SESSION] Pass session info to underlying engine for SSD swap
        self.language_model.active_session_id = session_id;
        
        // [DYNAMIC-RECOVERY] Check for free VRAM and move layers back to GPU if possible
        let _ = self.language_model.rebalance_layers(0, seqlen_offset + seq_len);

        let input_ids = if !input_ids_in.device().same_device(&self.text_device) { input_ids_in.to_device(&self.text_device)? } else { input_ids_in.clone() };
        let cache_position = if let Some(cp) = cache_position_in { if !cp.device().same_device(&self.text_device) { Some(cp.to_device(&self.text_device)?) } else { Some(cp.clone()) } } else { None };
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;

        // [STRICT-EMBED-ALIGN] 2B 임베딩이 이미 주입되었으므로 추가 확장 금지
        let start = if let Some(cp) = cache_position_in { cp.flatten_all()?.i(0)?.to_scalar::<u32>()? } else { 0 };
        let position_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, total_len, Some(&position_ids), None, None)?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        if let Some(head) = &self.lm_head {
            // [CRITICAL-OOM-FIX] If in DiskSwap mode, force the heavy lm_head computation to CPU
            // to avoid the final logit spike that causes the 100% OOM loop.
            let target_device = if self.language_model.is_disk_swap { 
                &Device::Cpu 
            } else { 
                head.device() 
            };
            
            let hidden_state = if !hidden_state.device().same_device(target_device) { 
                hidden_state.to_device(target_device)? 
            } else { 
                hidden_state 
            };
            
            // Move head to CPU if needed (one-time cost)
            let mut head_cpu;
            let active_head = if self.language_model.is_disk_swap && !head.device().is_cpu() {
                // This is slightly heavy but only happens once at the moment of OOM survival
                println!("[OOM-SURVIVAL] Offloading LM_HEAD to CPU for final logit computation...");
                head_cpu = head.clone();
                head_cpu.to_device(&Device::Cpu)?;
                &head_cpu
            } else {
                head
            };

            Ok(active_head.forward(&hidden_state)?)
        } else { Ok(hidden_state) }
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> { self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale) }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> { self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales) }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.language_model.to_device(device)?; if let Some(head) = &mut self.lm_head { head.to_device(device)?; } self.text_device = device.clone(); Ok(()) }
    pub fn rebalance_layers(&mut self, device_id: usize, context_len: usize) -> Result<()> { self.language_model.rebalance_layers(device_id, context_len) }
}

fn get_qlinear_06b<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let mut weight_t = weight.dequantize(device)?.to_dtype(dtype)?;
    let shape = weight_t.dims();
    
    // Strictly for 0.6B [3072, 1024] or [1024, 3072]
    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        if rows == 3072 && cols == 1024 {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        } else if rows == 1024 && cols == 3072 {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        }
    }
    
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { Some(t.dequantize(device)?.to_dtype(dtype)?) } else { None };
    Ok(QLinear::new(QMatMul::Tensor(weight_t), bias, device.clone()))
}

fn get_qlinear_2b_hybrid<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let mut weight_t = weight.dequantize(device)?.to_dtype(dtype)?;
    let shape = weight_t.dims();
    
    // Strictly for 2B Hybrid [6144, 2048] or [2048, 6144]
    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        if (name.contains("gate") || name.contains("up")) && rows == 2048 && cols == 6144 {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        } else if name.contains("down") && rows == 6144 && cols == 2048 {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        }
    }
    
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { Some(t.dequantize(device)?.to_dtype(dtype)?) } else { None };
    Ok(QLinear::new(QMatMul::Tensor(weight_t), bias, device.clone()))
}

fn get_qlinear_v2<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType, hidden_size: usize) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let mut weight_t = weight.dequantize(device)?.to_dtype(dtype)?;
    
    // [HYBRID-RECURSIVE-FOLDING-V2]
    if hidden_size == 1024 {
        // ... (Folding logic remains same for 0.6B mode)
        let dims = weight_t.dims();
        if dims.len() == 2 {
            let mut t = weight_t.clone();
            let mut was_folded = false;
            let mut fold_count = 1.0;
            
            let mut d0 = t.dim(0)?;
            while d0 > 1024 && d0 != 3072 {
                let half = d0 / 2;
                t = ((t.narrow(0, 0, half)? + t.narrow(0, half, half)?)? / 2.0)?;
                d0 = t.dim(0)?;
                fold_count *= 2.0;
            }
            if d0 == 6144 {
                t = ((t.narrow(0, 0, 3072)? + t.narrow(0, 3072, 3072)?)? / 2.0)?;
                fold_count = 2.0;
                was_folded = true;
            }

            let mut d1 = t.dim(1)?;
            while d1 > 1024 && d1 != 3072 {
                let half = d1 / 2;
                t = ((t.narrow(1, 0, half)? + t.narrow(1, half, half)?)? / 2.0)?;
                d1 = t.dim(1)?;
                fold_count *= 2.0;
                was_folded = true;
            }

            if was_folded || d0 > 1024 || d1 > 1024 {
                // [STRICT-NAME-BASED-ALIGNMENT]
                // 0.6B engine has specific orientation needs for each layer type.
                // We enforce these based on the actual 0.6B model architecture.
                let (r, c) = (t.dim(0)?, t.dim(1)?);
                
                if name.contains("gate") || name.contains("up") {
                    // Expects [1024, 3072]
                    if r == 3072 && c == 1024 { t = t.transpose(0, 1)?; }
                } else if name.contains("down") {
                    // Expects [3072, 1024]
                    if r == 1024 && c == 3072 { t = t.transpose(0, 1)?; }
                } else if name.contains("attn_q") {
                    // Expects [1024, 2048]
                    if r == 2048 && c == 1024 { t = t.transpose(0, 1)?; }
                } else if name.contains("attn_output") {
                    // Expects [2048, 1024]
                    if r == 1024 && c == 2048 { t = t.transpose(0, 1)?; }
                } else {
                    // Default fallback for hidden-to-hidden
                    if r == 1024 && c == 1024 {} // Already correct
                    else if r > c { t = t.transpose(0, 1)?; }
                }
                weight_t = t.contiguous()?;
            }
        }
    } else if hidden_size == 2048 {
        // [HYBRID-ENERGY-PRESERVING-UNFOLDING]
        let (r, c) = (weight_t.dim(0)?, weight_t.dim(1)?);
        if r == 1024 || c == 1024 {
            let mut t = weight_t.clone();
            if r == 1024 && c == 1024 {
                t = Tensor::cat(&[&t, &t], 0)?;
                t = Tensor::cat(&[&t, &t], 1)?;
                t = (t / 2.0)?; // Scale down to preserve magnitude
            } else if (r == 3072 || r == 2048) && c == 1024 {
                t = Tensor::cat(&[&t, &t], 1)?;
                t = (t / 1.414)?; // Sqrt scaling for better balance
            } else if r == 1024 && (c == 3072 || c == 2048) {
                t = Tensor::cat(&[&t, &t], 0)?;
                t = (t / 1.414)?;
            }
            
            // [STRICT-UNFOLD-CAP]
            let (rf, cf) = (t.dim(0)?, t.dim(1)?);
            if rf > 2048 && !name.contains("mlp") && !name.contains("gate") && !name.contains("up") {
                t = t.narrow(0, 0, 2048)?.contiguous()?;
            }
            if cf > 2048 && !name.contains("mlp") && !name.contains("gate") && !name.contains("up") {
                t = t.narrow(1, 0, 2048)?.contiguous()?;
            }
            
            weight_t = t.contiguous()?;
        }
        
        // [STRICT-2B-TRANSPOSE-GUARD]
        // Candle's QLinear (via QMatMul::Tensor) expects [in_dim, out_dim].
        // GGUF weights are often stored as [out_dim, in_dim].
        let (r_final, c_final) = (weight_t.dim(0)?, weight_t.dim(1)?);
        if r_final > c_final {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        }
    }

    let shape = weight_t.dims();
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let mut needs_transpose = false;
        
        if hidden_size == 1024 {
            // Standard Qwen 0.6B layout: Gate/Up [3072, 1024], Down [1024, 3072]
            if rows == 3072 && cols == 1024 {
                needs_transpose = true; 
            } else if rows == 1024 && cols == 3072 {
                needs_transpose = true;
            }
        } else {
            // [2B-HYBRID-SPEC-ALIGNMENT]
            if hidden_size == 2048 {
                // 2B 원본 레이아웃: Gate/Up [6144, 2048], Down [2048, 6144]
                // 만약 GGUF가 [2048, 6144]로 되어 있다면 트랜스포즈 필요
                if (name.contains("gate") || name.contains("up")) && rows == 2048 && cols == 6144 {
                    needs_transpose = true;
                } else if name.contains("down") && rows == 6144 && cols == 2048 {
                    needs_transpose = true;
                } else if (name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")) && rows == 2048 && cols == 2048 {
                    // Attention projections often need transpose in Candle
                    needs_transpose = true;
                }
            }
        }

        if needs_transpose {
            weight_t = weight_t.transpose(0, 1)?.contiguous()?;
        }
    }
    
    let weight_q = QMatMul::Tensor(weight_t);
    let mut bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) { 
        Some(t.dequantize(device)?.to_dtype(dtype)?) 
    } else { None };

    // Also slice bias if needed
    if hidden_size == 1024 {
        if let Some(b) = bias {
            let b_dim = b.dim(0)?;
            if b_dim == 2048 { bias = Some(b.narrow(0, 0, 1024)?.contiguous()?); }
            else if b_dim == 6144 { bias = Some(b.narrow(0, 0, 3072)?.contiguous()?); }
            else { bias = Some(b); }
        }
    }

    Ok(QLinear::new(weight_q, bias, device.clone()))
}

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    get_qlinear_v2(ct, reader, name, device, dtype, 2048) // Legacy fallback
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

fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType, hidden_size: usize) -> Result<RmsNorm> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let mut weight = weight.dequantize(device)?.to_dtype(dtype)?;
    
    let d0 = weight.dim(0)?;
    
    // [HYBRID-BIDIRECTIONAL-ALIGN-NORM]
    if hidden_size == 1024 && d0 == 2048 {
        // Fold for 0.6B engine
        let w1 = weight.narrow(0, 0, 1024)?;
        let w2 = weight.narrow(0, 1024, 1024)?;
        weight = ((w1 + w2)? / 2.0)?.contiguous()?;
    } else if hidden_size == 2048 && d0 == 1024 {
        // Unfold for 2B engine
        weight = Tensor::cat(&[&weight, &weight], 0)?.contiguous()?;
    }

    // [STRICT-WEIGHT-SAFEGUARD]
    if weight.dim(0)? > 2048 && hidden_size <= 2048 {
        weight = weight.narrow(0, 0, 2048)?.contiguous()?;
    }
    
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
        let mut t = ct.tensor(reader, name, device)?;
        let mut t = t.dequantize(device)?.to_dtype(dtype)?;
        
        // [HYBRID-VISION-PRECISION-FOLDING]
        if config.hidden_size == Some(1024) {
            let dims = t.dims();
            if dims.len() >= 1 {
                let mut current_t = t.clone();
                let mut d0 = current_t.dim(0)?;
                let mut was_folded = false;
                let target_d0 = if d0 > 3072 { 1024 } else if d0 > 1024 { 1024 } else { d0 };
                let ratio = d0 as f64 / target_d0 as f64;

                if ratio > 1.0 {
                    while d0 > target_d0 && d0 != 3072 {
                        let half = d0 / 2;
                        current_t = ((current_t.narrow(0, 0, half)? + current_t.narrow(0, half, half)?)? / 2.0)?;
                        d0 = current_t.dim(0)?;
                        was_folded = true;
                    }
                    if d0 == 6144 {
                        current_t = ((current_t.narrow(0, 0, 3072)? + current_t.narrow(0, 3072, 3072)?)? / 2.0)?;
                        was_folded = true;
                    }
                    
                    if was_folded {
                        // Encode Protocol Marker: -(1.0 + Ratio*0.0001 + Depth*0.0000001)
                        let marker_val = -(1.0 + (ratio * 0.0001) + (ratio * 0.0000001));
                        let mut data = current_t.flatten_all()?.to_vec1::<f32>()?;
                        data[0] = marker_val as f32;
                        t = Tensor::from_vec(data, current_t.shape(), device)?.contiguous()?;
                    }
                }
            }
        }
        
        if is_split { split_tensors.entry(base_split_name).or_default().push((split_idx, t)); } else { data.insert(new_name, t); }
        
            }
        
            for (name, mut parts) in split_tensors { parts.sort_by_key(|(i, _)| *i); let tensors: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect(); if let Ok(merged) = Tensor::cat(&tensors, 0) { data.insert(name, merged); } }
        
            if let Some(weight) = data.get("visual.patch_embed.proj.weight") { if weight.rank() == 4 { if let Ok(reshaped) = weight.unsqueeze(2)?.repeat((1, 1, 2, 1, 1)) { data.insert("visual.patch_embed.proj.weight".to_string(), reshaped); println!("[FIX] Reshaped visual.patch_embed.proj.weight to 5D"); } } }
        
            Ok(VarBuilder::from_tensors(data, dtype, device))
        
        }
        