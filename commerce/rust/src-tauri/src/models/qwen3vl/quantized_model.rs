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
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };

        // if xs.device().is_cpu() && xs.dtype() == DType::BF16 {
        //     println!("[TRACE-LINEAR-VIOLATION] CPU Linear with BF16 input!");
        // }

        let xs_f32 = xs.to_dtype(DType::F32)?;
        let (b, s, h) = xs_f32.dims3()?;
        let xs_f32_flat = xs_f32.reshape((b * s, h))?;
        
        let out = self.inner.forward(&xs_f32_flat)?;
        let out = out.reshape((b, s, ()))?;
        
        let out = out.to_dtype(target_dtype)?;

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

        if self.layer_idx == 0 {
            println!("[TRACE-L0] xs: {:?} {:?}, cos: {:?}, target: {:?}", xs.device(), xs.dtype(), cos.dtype(), target_dtype);
        }

        // 1. [HARDENING] Inbound Input Alignment
        let xs = if !xs.device().same_device(dev) { 
            let moved = xs.to_device(dev)?;
            if self.layer_idx == 0 { println!("[TRACE-MOVE] xs moved to {:?}", dev); }
            moved
        } else { xs.clone() };
        
        let xs = if xs.dtype() != target_dtype { 
            let casted = xs.to_dtype(target_dtype)?;
            if self.layer_idx == 0 { println!("[TRACE-CAST] xs casted to {:?}", target_dtype); }
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
        
        // 3. [HARDENING] KV Cache Concatenation Guard
        let (key_states, value_states): (Tensor, Tensor) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                if self.layer_idx == 0 {
                    println!("[TRACE-KV] prev_k: {:?} {:?}, new_k: {:?} {:?}", prev_k.device(), prev_k.dtype(), key_states.device(), key_states.dtype());
                }
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
            if self.layer_idx == 0 { println!("[TRACE-MASK] mask: {:?} {:?}", mask.device(), mask.dtype()); }
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

    pub fn compress_to_8bit(&self, t: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let t_data = t_f32.flatten_all()?.to_vec1::<f32>()?;
        
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements = t_data.len();
        let num_vectors = total_elements / last_dim;
        
        let mut scales = vec![0.0f32; num_vectors];
        let mut packed = vec![0u8; total_elements];
        
        // Using standard loop for absolute stability on Windows
        packed.chunks_mut(last_dim).zip(scales.iter_mut()).enumerate().for_each(|(v_idx, (packed_vector, scale))| {
            let t_start = v_idx * last_dim;
            let t_vector = &t_data[t_start..t_start + last_dim];
            
            let mut max_abs = 0.0f32;
            for &val in t_vector {
                let abs_v = val.abs();
                if abs_v > max_abs { max_abs = abs_v; }
            }
            
            let s = max_abs / 127.0;
            *scale = s;
            
            if s > 0.0 {
                for i in 0..last_dim {
                    packed_vector[i] = (t_vector[i] / s).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        });
            
        let packed_tensor = Tensor::from_vec(packed, vec![original_shape[0], original_shape[1], original_shape[2], last_dim], &Device::Cpu)?;
        let scales_tensor = Tensor::from_vec(scales, vec![original_shape[0], original_shape[1], original_shape[2], 1], &Device::Cpu)?;
        
        Ok((packed_tensor, scales_tensor, original_shape))
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

    // [LIVE-BRIDGE] Inject 512-dim KV from 0.6B into 2048-dim VRAM of 2B
    // [QUANTIZED-RELAY] Receives 8-bit data to minimize PCIe traffic
    pub fn inject_live_kv(&mut self, k_i8: &Tensor, v_i8: &Tensor, k_scale: f32, v_scale: f32) -> Result<()> {
        let target_heads = self.num_key_value_heads; // 16 heads
        let target_dim = self.head_dim;             // 128 dim
        let target_device = self.q_proj.device(); 
        let target_dtype = if target_device.is_cuda() { DType::BF16 } else { DType::F32 };
        
        // 1. Transfer tiny 8-bit data to GPU (Fast PCIe Path)
        let k_gpu_i8 = k_i8.to_device(target_device)?;
        let v_gpu_i8 = v_i8.to_device(target_device)?;

        // 2. High-speed Dequantization on GPU
        let k_small = (k_gpu_i8.to_dtype(DType::F32)? * k_scale as f64)?.to_dtype(target_dtype)?;
        let v_small = (v_gpu_i8.to_dtype(DType::F32)? * v_scale as f64)?.to_dtype(target_dtype)?;

        let (_b, _h, _s, d) = k_small.dims4()?;
        
        // 1. Dimension Alignment (e.g. 64 -> 128 if needed)
        let mut k = if d < target_dim {
            Tensor::cat(&[&k_small, &k_small], D::Minus1)?
        } else { k_small.clone() };
        
        let mut v = if d < target_dim {
            Tensor::cat(&[&v_small, &v_small], D::Minus1)?
        } else { v_small.clone() };

        // 2. Head Alignment (e.g. 4 heads -> 16 heads)
        // Replicate heads to match the 2048-dim space (16 * 128)
        if k.dim(1)? != target_heads {
            let mut k_heads = vec![];
            let mut v_heads = vec![];
            let current_h = k.dim(1)?;
            for i in 0..target_heads {
                let src_idx = i % current_h;
                k_heads.push(k.narrow(1, src_idx, 1)?);
                v_heads.push(v.narrow(1, src_idx, 1)?);
            }
            k = Tensor::cat(&k_heads, 1)?;
            v = Tensor::cat(&v_heads, 1)?;
        }
        
        let k_final = k;
        let v_final = v;

        if self.layer_idx == 0 {
            println!("[KV-BRIDGE] Quantized Relay: 8-bit PCIe transfer -> GPU Dequantization");
        }
        
        // Append to existing cache
        self.kv_cache = match &self.kv_cache {
            None => Some((k_final, v_final)),
            Some((prev_k, prev_v)) => {
                let pk = if prev_k.dtype() != target_dtype { prev_k.to_dtype(target_dtype)? } else { prev_k.clone() };
                let pv = if prev_v.dtype() != target_dtype { prev_v.to_dtype(target_dtype)? } else { prev_v.clone() };
                let k = Tensor::cat(&[&pk, &k_final], 2)?;
                let v = Tensor::cat(&[&pv, &v_final], 2)?;
                Some((k, v))
            }
        };
        Ok(())
    }

    fn compress_to_1bit(&self, t: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let t_data = t_f32.flatten_all()?.to_vec1::<f32>()?;
        
        let last_dim = original_shape[original_shape.len() - 1];
        let total_elements = t_data.len();
        let num_vectors = total_elements / last_dim;
        
        // 1-bit requires packing 8 elements into 1 byte
        let packed_size = (total_elements + 7) / 8;
        let mut packed = vec![0u8; packed_size];
        let mut scales = vec![0.0f32; num_vectors];
        
        for v_idx in 0..num_vectors {
            let t_start = v_idx * last_dim;
            let t_vector = &t_data[t_start..t_start + last_dim];
            
            // Calculate scale: mean of absolute values
            let mut abs_sum = 0.0f32;
            for &val in t_vector { abs_sum += val.abs(); }
            let s = abs_sum / (last_dim as f32);
            scales[v_idx] = s;
            
            // Pack bits
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _block_size: usize) -> Result<()> {
        let file = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
        let mut map = HashMap::new();

        if let Some((k, v)) = &self.kv_cache {
            // [EXTREME-BAKING] Use 1-bit for maximum VRAM savings
            let k_cpu = k.to_device(&Device::Cpu)?;
            let v_cpu = v.to_device(&Device::Cpu)?;
            
            let (k_packed, k_scales, k_shape) = self.compress_to_1bit(&k_cpu)?;
            let (v_packed, v_scales, _) = self.compress_to_1bit(&v_cpu)?;
            
            map.insert("k_packed".to_string(), k_packed);
            map.insert("v_packed".to_string(), v_packed);
            map.insert("k_scales".to_string(), k_scales);
            map.insert("v_scales".to_string(), v_scales);
            map.insert("k_shape".to_string(), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect(), (k_shape.len(),), &Device::Cpu)?);
            map.insert("mode".to_string(), Tensor::from_vec(vec![2u32], (1,), &Device::Cpu)?); 
            map.insert("bits".to_string(), Tensor::from_vec(vec![1u32], (1,), &Device::Cpu)?);
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
        
        let bits = if let Ok(view) = st.tensor("bits") {
            let data = view.data();
            if data.len() >= 4 { u32::from_le_bytes(data[0..4].try_into().unwrap()) } else { 8 }
        } else { 8 };
        
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        let (mut k, mut v) = {
            let dequantize_tensor = |prefix: &str| -> Result<Tensor> {
                let packed_view = st.tensor(&format!("{}_packed", prefix))?;
                let packed = Tensor::from_slice(packed_view.data(), packed_view.shape(), device)?;
                
                let scales_view = st.tensor(&format!("{}_scales", prefix))?;
                let scales_data: &[f32] = unsafe {
                    std::slice::from_raw_parts(scales_view.data().as_ptr() as *const f32, scales_view.data().len() / 4)
                };
                let scales = Tensor::from_slice(scales_data, scales_view.shape(), device)?;
                
                let shape_view = st.tensor("k_shape")?;
                let shape_u32: &[u32] = unsafe {
                    std::slice::from_raw_parts(shape_view.data().as_ptr() as *const u32, shape_view.data().len() / 4)
                };
                let shape: Vec<usize> = shape_u32.iter().map(|&x| x as usize).collect();

                if bits == 1 {
                    let t = self.decompress_from_1bit(&packed, &scales, &shape)?;
                    Ok(t.to_dtype(target_dtype)?)
                } else if bits == 8 {
                    let t = self.decompress_from_8bit(&packed, &scales, &shape)?;
                    Ok(t.to_dtype(target_dtype)?)
                } else {
                    Err(anyhow!("Unsupported bit depth: {}", bits))
                }
            };
            (dequantize_tensor("k")?, dequantize_tensor("v")?)
        };

        // [LINEAR-BRIDGE] Convert 0.6B (512) cache to 2B (2048) cache
        let (_b, h, _s, d) = k.dims4()?;
        let target_heads = self.num_key_value_heads;
        let target_dim = self.head_dim;

        if h != target_heads || d != target_dim {
            if self.layer_idx == 0 {
                println!("[LINEAR-BRIDGE] Convert 0.6B (512) cache to 2B (2048) cache");
            }
            
            // 1. Dimension Replication (e.g. 64 -> 128)
            if d < target_dim {
                k = Tensor::cat(&[&k, &k], D::Minus1)?;
                v = Tensor::cat(&[&v, &v], D::Minus1)?;
            }
            
            // 2. Head Alignment (e.g. 4 heads -> 16 heads)
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

        // [UPSCALE-REFILL] Drop last N tokens from cache to let 2B re-refine the most critical context.
        // [FIX] Must use EXACTLY the same logic as seqlen_offset calculation in generate.rs to avoid shape mismatches.
        let actual_k_len = k.dim(2)?;
        let use_len = if expected_len == 0 { actual_k_len } else { expected_len };
        let final_len = if use_len > upscale_refill_len { use_len - upscale_refill_len } else { use_len };

        if final_len > 0 {
            // Ensure we don't try to narrow more than what is available in the loaded tensor
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
    pub mlp_gate: QLinear,
    pub mlp_up: QLinear,
    pub mlp_down: QLinear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3VLTextDecoderLayer {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.mlp_gate.to_device(device)?;
        self.mlp_up.to_device(device)?;
        self.mlp_down.to_device(device)?;
        self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm.to_device(device)?;
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
    ) -> Result<Self> {
        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx)?;
        
        let mlp_gate = get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?;
        let mlp_up = get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?;
        let mlp_down = get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?;

        let input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype)?;
        let post_attention_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype)?;

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

        if self.self_attn.layer_idx % 5 == 0 || dev.is_cpu() {
            println!("[TRACE-LAYER-{}] Device: {:?}, DType: {:?}, In: {:?}", 
                self.self_attn.layer_idx, dev, target_dtype, xs.dtype());
        }
        
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
        
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = {
            let gate = self.mlp_gate.forward(&xs)?;
            let up = self.mlp_up.forward(&xs)?;
            let gate = candle_nn::ops::silu(&gate)?;
            let hidden = gate.mul(&up)?;
            self.mlp_down.forward(&hidden)?
        };
        // [HARDENING] Second Residual Addition DType Guard
        let xs = if xs.dtype() != residual.dtype() { xs.to_dtype(residual.dtype())? } else { xs };
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }

    pub fn get_kv_len(&self) -> usize {
        self.self_attn.get_kv_len()
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
    pub single_layer_mode: bool, // [NEW] Extreme memory saving mode
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
        single_layer_mode: bool, // [NEW] 
    ) -> Result<Self> {
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

        // Create a patched config for layer initialization
        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config; // Re-alias to use patched version below

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
        
        let estimated_activation_buffer = 50_000_000; // Drastically reduced to 50MB to fit 2B into 4GB VRAM
        let cost_per_layer = if layer_weight_size > 0 { layer_weight_size } else { 30_000_000 };
        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         let os_reserve = 50_000_000; // Reduced to 50MB
                         safety_floor = os_reserve + kv_reserve + estimated_activation_buffer;
                     }
                 }
             }
        }

        let mut layer_devices = vec![];
        let effective_cost = if cost_per_layer > 150_000_000 { 40_000_000 } else { cost_per_layer };

        // [FORCE-PATCH] Limit layers in single_layer_mode
        let num_actual_layers = if single_layer_mode { 1 } else { config.num_hidden_layers };

        for _ in 0..num_actual_layers {
            if current_device.is_cuda() && is_vram_checked {
                 // Use 1.05x margin for safer buffer
                 if simulated_free_vram > ( (effective_cost as f64 * 1.05) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(effective_cost);
                 } else {
                     current_device = Device::Cpu;
                 }
            }
            layer_devices.push(current_device.clone());
        }

        // [ORGANIC] Dynamic Threading for Parallel Loading
        let thread_config = crate::utils::resources::get_optimal_thread_config();
        println!("[MODEL] Organic Loading: Using {} threads ({})", thread_config.thread_count, thread_config.description);
        
        let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_config.thread_count).build()?;
        
        // Ensure we capture the PATCHED config, not the original one
        let final_config = config; 

        let layers: Result<Vec<_>> = pool.install(|| {
            (0..num_actual_layers).into_par_iter().zip(layer_devices).map(|(layer_idx, layer_device)| {
                let mut local_cursor = std::io::Cursor::new(mmap);
                let layer_dtype = if layer_device.is_cpu() { DType::F32 } else { dtype };
                let standard = format!("{base_name}.layers.{layer_idx}");
                let gguf_blk = format!("blk.{layer_idx}");
                let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { gguf_blk } else { standard };
                QuantizedQwen3VLTextDecoderLayer::new(
                    final_config, ct, &mut local_cursor, &prefix, &layer_device, layer_dtype, layer_idx
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
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb, mrope_section, mmap: mmap_handle, single_layer_mode })
    }
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64, // New param
        single_layer_mode: bool, // [NEW]
    ) -> Result<Self> {
        // Embeddings: Try standard and GGUF names
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

        // Initialize NVML for dynamic VRAM check
        let nvml = Nvml::init().ok();
        let mut current_device = device.clone();
        
        // --- Organic Budget-based Optimization ---
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
        
        let estimated_activation_buffer = 50_000_000; 
        let cost_per_layer = if layer_weight_size > 0 { layer_weight_size } else { 30_000_000 };

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
        let num_layers = if single_layer_mode { 1 } else { config.num_hidden_layers };

        for layer_idx in 0..num_layers {
            // Organic Check based on actual remaining bytes
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
                config, ct, reader, &prefix, &current_device, layer_dtype, layer_idx,
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
            single_layer_mode,
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
        let target_dtype = if inputs_embeds.device().is_cuda() { DType::BF16 } else { DType::F32 };
        println!("[TRACE-MODEL] Start forward. Input: {:?} {:?}, Target: {:?}", inputs_embeds.device(), inputs_embeds.dtype(), target_dtype);
        
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
        
        let xs = inputs_embeds.clone();
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

    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_list.len() {
                layer.self_attn.inject_live_kv(&k_list[i], &v_list[i], k_scale, v_scale)?;
            }
        }
        Ok(())
    }

    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < k_list.len() {
                let ks = k_scales.get(i).cloned().unwrap_or(1.0);
                let vs = v_scales.get(i).cloned().unwrap_or(1.0);
                layer.self_attn.inject_live_kv(&k_list[i], &v_list[i], ks, vs)?;
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

        if self.single_layer_mode {
            // [EXTREME-BAKING] Replicate first layer KV to all 28 layers using 1-bit
            // 1. Extract the KV cache from the first layer
            let kv_data = if let Some(layer0) = self.layers.get(0) {
                layer0.self_attn.kv_cache.clone()
            } else {
                None
            };

            if let Some((k, v)) = kv_data {
                // 2. Compress the data using self methods (no longer overlapping with layers borrow)
                let k_cpu = k.to_device(&Device::Cpu)?;
                let v_cpu = v.to_device(&Device::Cpu)?;
                let (kp, ks, ksh) = self.compress_to_1bit(&k_cpu)?;
                let (vp, vs, _) = self.compress_to_1bit(&v_cpu)?;

                // 3. Save to disk 28 times
                for i in 0..28 {
                    let file = path.join(format!("layer_{}_kv.safetensors", i));
                    let mut map = HashMap::new();
                    map.insert("k_packed".to_string(), kp.clone());
                    map.insert("v_packed".to_string(), vp.clone());
                    map.insert("k_scales".to_string(), ks.clone());
                    map.insert("v_scales".to_string(), vs.clone());
                    map.insert("k_shape".to_string(), Tensor::from_vec(ksh.iter().map(|&x| x as u32).collect(), (ksh.len(),), &Device::Cpu)?);
                    map.insert("mode".to_string(), Tensor::from_vec(vec![2u32], (1,), &Device::Cpu)?); 
                    map.insert("bits".to_string(), Tensor::from_vec(vec![1u32], (1,), &Device::Cpu)?);
                    candle_core::safetensors::save(&map, &file)?;
                }
            }

            // 4. Clear cache if requested
            if clear {
                if let Some(layer0) = self.layers.get_mut(0) {
                    layer0.clear_kv_cache();
                }
            }
            Ok(())
        } else {
            // [STABILITY] Use sequential saving to avoid CUDA_ERROR_INVALID_CONTEXT in rayon threads
            self.layers.iter_mut().try_for_each(|layer| {
                layer.save_kv_cache(path, clear, block_size)
            })
        }
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
    ) -> Result<Self> {
        let mmproj_mmap = mmproj_mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { dtype };
        let mut reader_vision = std::io::Cursor::new(mmproj_mmap);
        let vb_visual = from_gguf_content(config, ct_vision, &mut reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;

        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
            t_config, ct_main, main_mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, false
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
    ) -> Result<Self> {
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
        let vision_dtype = if vision_device.is_cpu() { DType::F32 } else { dtype };
        let vb_visual = from_gguf_content(config, ct_vision, reader_vision, vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;
        
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let language_model = QuantizedQwen3VLTextModel::new(t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, false)?;
        
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype)?
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
        if let Some(pixel_values) = pixel_values { if let Some(image_grid_thw) = image_grid_thw { let (image_embeds, _) = self.get_vision_features(pixel_values, image_grid_thw)?; let image_embeds = Tensor::cat(&image_embeds, 0)?; let vision_mask = self.get_placeholder_mask(&input_ids, true)?; inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?; } }
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
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(t_config, ct_main, mmap_handle.clone(), "model", text_device, text_device_id, dtype, kv_reserve, single_layer_mode)?;
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
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let language_model = QuantizedQwen3VLTextModel::new(t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve, single_layer_mode)?;
        let lm_head = if !baking_only {
            let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
            if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) { Some(l) }
            else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) { Some(l) }
            else { get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype).ok() }
        } else { None };
        Ok(Self { language_model, lm_head, text_device: text_device.clone(), mmap: None })
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