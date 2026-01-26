use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use rayon::prelude::*;
use nvml_wrapper::Nvml;
use std::path::Path;
use std::fs;
use std::collections::HashMap;

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
        if !self.weight.device().same_device(device) {
            self.weight = self.weight.to_device(device)?;
        }
        Ok(())
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let hidden_states = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let hidden_states = hidden_states.to_dtype(dtype)?;
        hidden_states.broadcast_mul(&self.weight)
    }
}

// Wrapper for QMatMul to act like Linear
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
            // Move bias
            if let Some(b) = &self.bias {
                let moved_b: Tensor = b.to_device(device)?;
                self.bias = Some(moved_b);
            }
            // Move inner QMatMul (Note: Candle QMatMul handles device internally or needs recreation)
            // For now, we update the tracked device. 
            // In a strict sense, we might need to move the inner tensors of QMatMul.
            self.device = device.clone();
        }
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // [OPTIMIZATION] Seamless Device Transfer
        // If input tensor is on a different device than this layer, move it.
        // This is the core of Layer-wise Offloading.
        let xs = if !xs.device().same_device(&self.device) {
            xs.to_device(&self.device)?
        } else {
            xs.clone()
        };

        let xs_f32 = xs.to_dtype(DType::F32)?;
        let (b, s, h) = xs_f32.dims3()?;
        let xs_f32_flat = xs_f32.reshape((b * s, h))?;
        
        let out = self.inner.forward(&xs_f32_flat)?;
        let out = out.reshape((b, s, ()))?;
        
        let target_dtype = self.bias.as_ref().map(|b: &Tensor| b.dtype()).unwrap_or(xs.dtype());
        let out = out.to_dtype(target_dtype)?;

        if let Some(bias) = &self.bias {
            Ok(out.broadcast_add(bias)?)
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
            self.kv_cache = Some((k.to_device(device)?, v.to_device(device)?));
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
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        // [HIGH-SPEED-KV] Use standard BF16 cache during active inference/ingestion
        let (key_states, value_states): (Tensor, Tensor) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let prev_k = prev_k.to_device(key_states.device())?.to_dtype(key_states.dtype())?;
                let prev_v = prev_v.to_device(value_states.device())?.to_dtype(value_states.dtype())?;
                let k = Tensor::cat(&[&prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[&prev_v, &value_states], 2)?;
                (k, v)
            }
        };

        // Update working cache
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Robust Masking
        let actual_seq_len = key_states.dim(2)?;
        let adjusted_mask = if let Some(mask) = attention_mask {
            let mask_len = mask.dim(D::Minus1)?;
            if mask_len < actual_seq_len {
                let padding = Tensor::zeros((b_sz, 1, q_len, actual_seq_len - mask_len), mask.dtype(), mask.device())?;
                Some(Tensor::cat(&[padding, mask.clone()], D::Minus1)?)
            } else if mask_len > actual_seq_len {
                Some(mask.narrow(D::Minus1, 0, actual_seq_len)?)
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

    fn compress_to_8bit(&self, t: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let t_data = t_f32.to_vec1::<f32>()?;
        
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

    fn decompress_from_8bit(&self, packed: &Tensor, scales: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let device = packed.device();
        let packed_vec = packed.to_device(&Device::Cpu)?.to_vec1::<u8>()?;
        let scales_vec = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        
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
            // [BALANCED-STABILITY] Use 8-bit (i8) for KV storage. 
            // Better precision than 4-bit, faster I/O than F16, and proven stable on CPU.
            let k_cpu = k.to_device(&Device::Cpu)?;
            let v_cpu = v.to_device(&Device::Cpu)?;
            
            let (k_packed, k_scales, k_shape) = self.compress_to_8bit(&k_cpu)?;
            let (v_packed, v_scales, _) = self.compress_to_8bit(&v_cpu)?;
            
            map.insert("k_packed".to_string(), k_packed);
            map.insert("v_packed".to_string(), v_packed);
            map.insert("k_scales".to_string(), k_scales);
            map.insert("v_scales".to_string(), v_scales);
            map.insert("k_shape".to_string(), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect(), (k_shape.len(),), &Device::Cpu)?);
            map.insert("mode".to_string(), Tensor::from_vec(vec![2u32], (1,), &Device::Cpu)?); 
            map.insert("bits".to_string(), Tensor::from_vec(vec![8u32], (1,), &Device::Cpu)?);
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

                if bits == 8 {
                    let t = self.decompress_from_8bit(&packed, &scales, &shape)?;
                    Ok(t.to_dtype(target_dtype)?)
                } else {
                    Err(anyhow!("Unsupported bit depth: {}. Expected 8.", bits))
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
        let final_len = if expected_len > upscale_refill_len { expected_len - upscale_refill_len } else { expected_len };

        if final_len > 0 {
            // Ensure we don't try to narrow more than what is available in the loaded tensor
            let safe_len = final_len.min(k.dim(2)?);
            self.kv_cache = Some((k.narrow(2, 0, safe_len)?, v.narrow(2, 0, safe_len)?));
        }
        Ok(())
    }

                    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
                        self.save_kv_cache(path, true, block_size)
                    }
}

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
        // 1. Detect device AND dtype of this layer
        let dev = self.input_layernorm.weight().device();
        let target_dtype = self.input_layernorm.weight().dtype();
        
        // 2. Ensure inputs are on this device and dtype
        //    (Clone via Cow logic or explicit clone if needed, here we use explicit clones/conversions for safety)
        let mut xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        if xs.dtype() != target_dtype { xs = xs.to_dtype(target_dtype)?; }

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

pub struct QuantizedQwen3VLTextModel {
    pub embed_tokens: Embedding, 
    pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
}

impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        mmap: &[u8],
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
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
        
        let estimated_activation_buffer = 200_000_000; 
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
                         let os_reserve = 80_000_000; 
                         safety_floor = os_reserve + kv_reserve + estimated_activation_buffer;
                     }
                 }
             }
        }

        let mut layer_devices = vec![];
        let effective_cost = if cost_per_layer > 150_000_000 { 40_000_000 } else { cost_per_layer };

        for _ in 0..config.num_hidden_layers {
            if current_device.is_cuda() && is_vram_checked {
                 if simulated_free_vram > ( (effective_cost as f64 * 1.02) as u64 + safety_floor ) {
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
            (0..final_config.num_hidden_layers).into_par_iter().zip(layer_devices).map(|(layer_idx, layer_device)| {
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
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb, mrope_section })
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
        // 1. Calculate actual Layer Weight Cost from GGUF metadata
        let mut layer_weight_size = 0_u64;
        // Determine prefix for Layer 0 to scan
        let probe_prefix_gguf = "blk.0.";
        let probe_prefix_std = "model.layers.0.";
        let layer_prefix = if ct.tensor_infos.keys().any(|k| k.starts_with(probe_prefix_gguf)) { probe_prefix_gguf } else { probe_prefix_std };

        for (name, info) in &ct.tensor_infos {
            if name.starts_with(layer_prefix) {
                let elements: usize = info.shape.elem_count();
                // Estimate size: (elements / block_size) * type_size
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                layer_weight_size += size as u64;
            }
        }
        
        // [OOM-SAFETY] Increase activation buffer reserve for Vision-Language tasks.
        // 800MB is safer for the initial forward pass overhead on 4GB cards.
        let estimated_activation_buffer = 800_000_000; 
        let cost_per_layer = if layer_weight_size > 0 { layer_weight_size } else { 30_000_000 }; // Use only weight size

        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        // 2. Get System VRAM & Set Dynamic Safety Floor (OS + KV + Activations)
        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         
                         // [OPTIMIZATION] Dynamic OS Reserve
                         let os_reserve = 80_000_000; 
                         // Activation buffer is shared across layers, so add it to the safety floor (reserved once)
                         safety_floor = os_reserve + kv_reserve + estimated_activation_buffer;

                         println!("[VRAM-BUDGET] Live Free: {:.2} GB. Safety Buffer (OS+KV+Act): {:.2} MB. Weight Per Layer: {:.2} MB", 
                            mem.free as f64/1e9, safety_floor as f64/1e6, cost_per_layer as f64/1e6);
                     }
                 }
             }
        }

        let mut layers = vec![];
        for layer_idx in 0..config.num_hidden_layers {
            // Organic Check based on actual remaining bytes
            if current_device.is_cuda() && is_vram_checked {
                 // [FIX] Use 1.02x margin instead of 1.05x, and ensure cost_per_layer is sane
                 let effective_cost = if cost_per_layer > 150_000_000 { 40_000_000 } else { cost_per_layer };
                 if simulated_free_vram > ( (effective_cost as f64 * 1.02) as u64 + safety_floor ) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(effective_cost);
                 } else {
                     if current_device.is_cuda() {
                        println!("[OFFLOAD] VRAM Budget Reached. Layer {} and subsequent moved to CPU. (Available: {:.2} MB, Required: {:.2} MB)", 
                            layer_idx, simulated_free_vram as f64/1e6, (effective_cost + safety_floor) as f64/1e6);
                     }
                     current_device = Device::Cpu;
                 }
            }

            // Determine dtype: If CPU, force F32. Else use requested dtype (BF16).
            let layer_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };

            // Determine prefix: "model.layers.N" or "blk.N"
            let standard = format!("{base_name}.layers.{layer_idx}");
            let gguf_blk = format!("blk.{layer_idx}");
            
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) {
                gguf_blk
            } else {
                standard
            };

            let layer = QuantizedQwen3VLTextDecoderLayer::new(
                config,
                ct,
                reader,
                &prefix,
                &current_device,
                layer_dtype,
                layer_idx,
            )?;
            layers.push(layer)
        }
        
        // Norm - load on same device as last layer, with matching dtype
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) {
            alt_norm
        } else {
            &norm_name
        };
        
        let norm_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, &current_device, norm_dtype)?;
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        // [FIX] rope_scaling is now optional. Unwrap or default.
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section,
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
        
        // RoPE is computed here. Usually on inputs_embeds.device() (GPU).
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    seqlen_offset,
                    inputs_embeds.device(),
                )?)
            }
        };

        let total_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // [DEBUG-LOG] Track CPU inference progress
            if layer_idx % 4 == 0 {
                println!("[CPU-INFER] Processing Layer {}/{}...", layer_idx, total_layers);
            }

            // Layer handles device transfer internally
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    // Deepstack logic might need device check too
                    let mask = visual_pos_masks.unwrap();
                    let embed = &deepstack_embeds[layer_idx];
                    
                    // Move to xs device
                    let mask = if !mask.device().same_device(xs.device()) { mask.to_device(xs.device())? } else { mask.clone() };
                    let embed = if !embed.device().same_device(xs.device()) { embed.to_device(xs.device())? } else { embed.clone() };

                    xs = mask_index_add(
                        &xs.squeeze(0)?,
                        &mask.squeeze(0)?,
                        &embed,
                    )?
                    .unsqueeze(0)?;
                }
            }
        }
        
        // Final Norm - ensure xs is on norm device
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> {
        use rayon::prelude::*;
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
        // [PARALLEL] Save all 28 layers in parallel to maximize Disk/CPU throughput
        self.layers.par_iter_mut().try_for_each(|layer| {
            layer.save_kv_cache(path, clear, block_size)
        })
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        use rayon::prelude::*;
        if path.exists() {
            // [PARALLEL] Load all 28 layers in parallel
            self.layers.par_iter_mut().try_for_each(|layer| {
                layer.load_kv_cache(path, device, expected_len, upscale_refill_len)
            })
        } else {
            Ok(())
        }
    }

    /// [SLEEP-MODE] Moves all model weights and KV cache between CPU and GPU
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens.to_device(device)?;
        for layer in self.layers.iter_mut() {
            layer.to_device(device)?;
        }
        self.norm.to_device(device)?;
        Ok(())
    }
}

pub struct QuantizedQwen3VLModel {
    pub config: Qwen3VLConfig,
    pub visual: Qwen3VLVisionModel, 
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: QLinear,
    pub rope_deltas: Option<Tensor>,
    pub text_device: Device,
    pub vision_device: Device,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        main_mmap: &[u8],
        ct_vision: &gguf_file::Content,
        mmproj_mmap: &[u8],
        text_device: &Device,
        text_device_id: usize,
        vision_device: &Device,
        vision_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
        let nvml = Nvml::init().ok();
        
        let mut vision_weight_size = 0_u64;
        for (_, info) in &ct_vision.tensor_infos {
            let elements: usize = info.shape.elem_count();
            let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
            vision_weight_size += size as u64;
        }
        let vision_overhead = (vision_weight_size as f64 * 0.30) as u64;
        
        let mut actual_vision_device = vision_device.clone();
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;

        // [OOM-SAFETY] Conservative VRAM allocation for 4GB GPUs
        let estimated_activation_buffer = 1_000_000_000; // 1GB reserved for forward pass activations
        
        if actual_vision_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(vision_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let total_vram = mem.total;
                         let os_reserve = (total_vram as f64 * 0.05) as u64; // 5% OS reserve
                         let safety_floor = os_reserve.max(500_000_000) + estimated_activation_buffer; 
                         
                         if mem.free < (vision_weight_size + vision_overhead + safety_floor) {
                             println!("[MODEL-CONFIG] Insufficient VRAM for Vision Encoder. Offloading to CPU.");
                             actual_vision_device = Device::Cpu;
                         }
                     }
                 }
             }
        }
        
        let vision_dtype = if actual_vision_device.is_cpu() { DType::F32 } else { dtype };
        
        let mut reader_vision = std::io::Cursor::new(mmproj_mmap);
        let vb_visual = from_gguf_content(config, ct_vision, &mut reader_vision, &actual_vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;

        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
            t_config, ct_main, main_mmap, "model", text_device, text_device_id, dtype, kv_reserve
        )?;

        let mut reader_main = std::io::Cursor::new(main_mmap);
        let mut head_weight_size = 0_u64;
        let head_names = ["lm_head.weight", "output.weight", "token_embd.weight"];
        for name in head_names {
            if let Some(info) = ct_main.tensor_infos.get(name) {
                let elements: usize = info.shape.elem_count();
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                head_weight_size = size as u64;
                break;
            }
        }
        
        let mut head_device = text_device.clone();
        if head_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(text_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let absolute_min_margin = 50_000_000;
                         if mem.free < (head_weight_size + absolute_min_margin) {
                             head_device = Device::Cpu;
                         }
                     }
                 }
             }
        }

        let head_dtype = if head_device.is_cpu() { DType::F32 } else { dtype };

        let lm_head = if let Ok(l) = get_qlinear(ct_main, &mut reader_main, "lm_head", &head_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, &mut reader_main, "output", &head_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, &mut reader_main, "token_embd", &head_device, head_dtype)?
        };

        Ok(Self {
            config: config.clone(),
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
            text_device: text_device.clone(),
            vision_device: actual_vision_device,
        })
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
        vision_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
        // Shared NVML instance for this scope
        let nvml = Nvml::init().ok();

        // --- Organic Budget Calculation for Vision ---
        let mut vision_weight_size = 0_u64;
        for (_, info) in &ct_vision.tensor_infos {
            let elements: usize = info.shape.elem_count();
            let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
            vision_weight_size += size as u64;
        }
        let vision_overhead = (vision_weight_size as f64 * 0.30) as u64;
        let vision_total_cost = vision_weight_size + vision_overhead;

        // 1. Vision Model - Dynamic Loading
        let mut actual_vision_device = vision_device.clone();
        let v_config = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config for VL model"))?;

        if actual_vision_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(vision_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let total_vram = mem.total;
                         let os_reserve = (total_vram as f64 * 0.02) as u64; 
                         let vision_safety_floor = os_reserve.max(100_000_000); 
                         
                         if mem.free < (vision_total_cost + vision_safety_floor) {
                             println!("[OFFLOAD] Vision Budget Exhausted. Switching to CPU.");
                             actual_vision_device = Device::Cpu;
                         }
                     }
                 }
             }
        }
        
        let vision_dtype = if actual_vision_device.is_cpu() { DType::F32 } else { dtype };

        // Load Vision from mmproj file
        let vb_visual = from_gguf_content(config, ct_vision, reader_vision, &actual_vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(v_config.clone(), vb_visual.pp("visual"))?;
        
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;

        // Load Language Model from main file
        let language_model = QuantizedQwen3VLTextModel::new(t_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve)?;
        
        // --- Organic Budget Calculation for Head ---
        let mut head_weight_size = 0_u64;
        let head_names = ["lm_head.weight", "output.weight", "token_embd.weight"];
        for name in head_names {
            if let Some(info) = ct_main.tensor_infos.get(name) {
                let elements: usize = info.shape.elem_count();
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                head_weight_size = size as u64;
                break;
            }
        }

        // LM Head - Same as Text Device
        let mut head_device = text_device.clone();
        if head_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(text_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let absolute_min_margin = 50_000_000;
                         if mem.free < (head_weight_size + absolute_min_margin) {
                             head_device = Device::Cpu;
                         }
                     }
                 }
             }
        }

        let head_dtype = if head_device.is_cpu() { DType::F32 } else { dtype };

        let lm_head = if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", &head_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", &head_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, reader_main, "token_embd", &head_device, head_dtype)?
        };

        Ok(Self {
            config: config.clone(),
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
            text_device: text_device.clone(),
            vision_device: actual_vision_device,
        })
    }
    
    fn get_vision_features(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        // Ensure inputs are on the same device as the vision model
        let pixel_values = if !pixel_values.device().same_device(&self.vision_device) {
            pixel_values.to_device(&self.vision_device)?
        } else {
            pixel_values.clone()
        };

        let image_grid_thw = if !image_grid_thw.device().same_device(&self.vision_device) {
            image_grid_thw.to_device(&self.vision_device)?
        } else {
            image_grid_thw.clone()
        };

        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(&pixel_values, &image_grid_thw)?;
        
        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let split_sizes: Vec<usize> = prod_tensor_last_dim(&image_grid_thw)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize / spatial_merge_size.pow(2))
            .collect();
        
        // Transfer results to text_device for language processing
        let image_embeds = image_embeds.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter()
            .map(|t| Ok(t.to_device(&self.text_device)?))
            .collect();

        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
        Ok((image_embeds, deepstack_image_embeds?))
    }


    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token_id = if is_image {
            self.config.image_token_id.unwrap_or(0) as u32
        } else {
            self.config.video_token_id.unwrap_or(0) as u32
        };
        let special_token = Tensor::new(vec![special_token_id], input_ids.device())?;
        let special_mask = input_ids
            .broadcast_eq(&special_token)?
            .to_dtype(candle_core::DType::U32)?;
        Ok(special_mask)
    }
    
    // Placeholder implementation for get_rope_index to satisfy compilation.
    // In production this should replicate the full logic from Qwen3VLModel.
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        _image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let position_ids = Tensor::arange(0u32, input_ids.dim(1)? as u32, input_ids.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?;
        let mrope = Tensor::zeros((input_ids.dim(0)?, 1), input_ids.dtype(), input_ids.device())?;
        Ok((position_ids, mrope))
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        _pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Flatten input_ids to Rank 1 for Embedding, then reshape back to Rank 3
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                 println!("[VRAM-LOG] Starting Vision Encoder...");
                 let (image_embeds, _) = self.get_vision_features(pixel_values, image_grid_thw)?;
                 
                 let image_embeds = Tensor::cat(&image_embeds, 0)?;
                 let vision_mask = self.get_placeholder_mask(input_ids, true)?;
                 inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
            }
        }

        // Position IDs logic (simplified for quantized model)
        // Note: Full logic requires image/video grid awareness for mrope.
        let (position_ids, _) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
        let position_ids = if let Some(cache_pos) = cache_position {
             // Basic relative position for cache (needs improvement for mrope)
             let start = cache_pos.i(0)?.to_scalar::<u32>()?;
             Tensor::arange(start, start + input_ids.dim(1)? as u32, input_ids.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
        } else {
             position_ids
        };

        let outputs = self.language_model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids),
            None, // visual_pos_masks not fully implemented in quantized wrapper yet
            None, // deepstack not fully implemented
        )?;
        
        // Output from language model might be on CPU if last layer offloaded.
        // LM Head might be on GPU or CPU.
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        
        // Ensure hidden_state is on lm_head device
        let hidden_state = if !hidden_state.device().same_device(self.lm_head.device()) {
            hidden_state.to_device(self.lm_head.device())?
        } else {
            hidden_state
        };

        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> {
        self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale)
    }

    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales)
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> {
        self.language_model.save_kv_cache(path, clear, block_size)
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.language_model.offload_kv_cache(path, block_size)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> {
        self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len)
    }

    /// [SLEEP-MODE] Moves full Vision-Language model between devices
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        // 1. Move Vision Encoder (always move to ensure no VRAM leaks)
        self.visual.to_device(device)?;
        
        // 2. Move Language Model
        self.language_model.to_device(device)?;
        
        // 3. Move LM Head
        self.lm_head.to_device(device)?;
        
        // Update device trackers
        self.text_device = device.clone();
        self.vision_device = device.clone();
        
        Ok(())
    }
}

pub struct QuantizedQwen3TextModel {
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: QLinear,
    pub text_device: Device,
}

impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        mmap: &[u8],
        text_device: &Device,
        text_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text Model (0.6B Optimized) with Mmap");
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;

        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(
            t_config, 
            ct_main, 
            mmap, 
            "model", 
            text_device, 
            text_device_id, 
            dtype, 
            kv_reserve
        )?;

        let mut reader = std::io::Cursor::new(mmap);
        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear(ct_main, &mut reader, "lm_head", text_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, &mut reader, "output", text_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, &mut reader, "token_embd", text_device, head_dtype)?
        };

        Ok(Self {
            language_model,
            lm_head,
            text_device: text_device.clone(),
        })
    }

    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        reader_main: &mut R,
        text_device: &Device,
        text_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
        println!("[MODEL] Loading as Pure Text Model (0.6B Optimized)");
        
        let t_config = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;

        let language_model = QuantizedQwen3VLTextModel::new(
            t_config, 
            ct_main, 
            reader_main, 
            "model", 
            text_device, 
            text_device_id, 
            dtype, 
            kv_reserve
        )?;

        let head_dtype = if text_device.is_cpu() { DType::F32 } else { dtype };
        let lm_head = if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", text_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", text_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, reader_main, "token_embd", text_device, head_dtype)?
        };

        Ok(Self {
            language_model,
            lm_head,
            text_device: text_device.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;

        // Pure text position IDs
        let start = if let Some(cp) = cache_position { cp.i(0)?.to_scalar::<u32>()? } else { 0 };
        let position_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;

        let outputs = self.language_model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids),
            None,
            None,
        )?;
        
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        
        let hidden_state = if !hidden_state.device().same_device(self.lm_head.device()) {
            hidden_state.to_device(self.lm_head.device())?
        } else {
            hidden_state
        };

        Ok(self.lm_head.forward(&hidden_state)?)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    pub fn inject_live_kv(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scale: f32, v_scale: f32) -> Result<()> {
        self.language_model.inject_live_kv(k_list, v_list, k_scale, v_scale)
    }
    pub fn inject_live_kv_quantized(&mut self, k_list: &[Tensor], v_list: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        self.language_model.inject_live_kv_quantized(k_list, v_list, k_scales, v_scales)
    }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, block_size: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, block_size) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len) }

    /// [SLEEP-MODE] Moves text-only model between devices
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.language_model.to_device(device)?;
        self.lm_head.to_device(device)?;
        self.text_device = device.clone();
        Ok(())
    }
}

// Helper functions

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let weight = QMatMul::from_qtensor(weight)?;
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) {
        Some(t.dequantize(device)?.to_dtype(dtype)?)
    } else {
        None
    };
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
    
    // Print all tensor names for debugging
    // for (name, _) in ct.tensor_infos.iter() {
    //    println!("[DEBUG-TENSOR] {}", name);
    // }

    for (name, _) in ct.tensor_infos.iter() {
        let mut new_name = name.clone();
        
        // 1. Prefix Handling: v. -> visual.
        if let Some(rest) = name.strip_prefix("v.") {
             if let Some(blk_rest) = rest.strip_prefix("blk.") {
                 // blk.{i}.{layer}...
                 let parts: Vec<&str> = blk_rest.splitn(2, '.').collect();
                 if parts.len() == 2 {
                     let idx = parts[0];
                     let layer = parts[1];
                     let mapped_layer = match layer {
                         s if s.starts_with("ln1") => s.replace("ln1", "norm1"),
                         s if s.starts_with("ln2") => s.replace("ln2", "norm2"),
                         s if s.starts_with("attn_qkv") => s.replace("attn_qkv", "attn.qkv"),
                         s if s.starts_with("attn_out") => s.replace("attn_out", "attn.proj"),
                         s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.linear_fc1"),
                         s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"),
                         _ => layer.to_string(),
                     };
                     new_name = format!("visual.blocks.{}.{}", idx, mapped_layer);
                 }
             } else if rest.starts_with("patch_embd") {
                 new_name = rest.replace("patch_embd", "visual.patch_embed.proj");
             } else if rest.starts_with("position_embd") {
                 new_name = rest.replace("position_embd", "visual.pos_embed");
             } else if rest.starts_with("post_ln") {
                 new_name = rest.replace("post_ln", "visual.merger.norm");
             } else if rest.starts_with("deepstack.") {
                 // v.deepstack.5.fc1.weight
                 let parts: Vec<&str> = rest.split('.').collect();
                 // parts[0]="deepstack", parts[1]="5", parts[2..]="fc1.weight"
                 if parts.len() >= 2 {
                     if let Ok(layer_idx) = parts[1].parse::<usize>() {
                         let v_idx_opt = config.vision_config.as_ref().and_then(|vc| vc.deepstack_visual_indexes.iter().position(|&x| x == layer_idx));
                         if let Some(pos) = v_idx_opt {
                             let suffix = parts[2..].join(".");
                             new_name = format!("visual.deepstack_merger_list.{}.{}", pos, suffix)
                                            .replace("fc1", "linear_fc1")
                                            .replace("fc2", "linear_fc2");
                         } else {
                             new_name = rest.replace("deepstack", "visual.deepstack_merger_list")
                                            .replace("fc1", "linear_fc1")
                                            .replace("fc2", "linear_fc2");
                         }
                     } else {
                         // fallback
                         new_name = rest.replace("deepstack", "visual.deepstack_merger_list")
                                        .replace("fc1", "linear_fc1")
                                        .replace("fc2", "linear_fc2");
                     }
                 }
             } else {
                 new_name = format!("visual.{}", rest);
             }
        } else if let Some(rest) = name.strip_prefix("mm.") {
             if rest.starts_with("0") {
                 new_name = rest.replace("0", "visual.merger.linear_fc1");
             } else if rest.starts_with("2") {
                 new_name = rest.replace("2", "visual.merger.linear_fc2");
             }
        } else if name.starts_with("model.visual") {
             new_name = name.strip_prefix("model.").unwrap().to_string();
        }

        // 2. Handle Split Tensors (.0, .1 suffix)
        // Check if ends with .number
        let mut is_split = false;
        let mut split_idx = 0;
        let mut base_split_name = new_name.clone();

        if let Some(last_dot) = new_name.rfind('.') {
            if let Ok(idx) = new_name[last_dot+1..].parse::<usize>() {
                 if name.ends_with(&format!(".{}", idx)) {
                     base_split_name = new_name[..last_dot].to_string();
                     split_idx = idx;
                     is_split = true;
                 }
            }
        }

        let t = ct.tensor(reader, name, device)?;
        let t = t.dequantize(device)?.to_dtype(dtype)?;

        if is_split {
            split_tensors.entry(base_split_name).or_default().push((split_idx, t));
        } else {
            data.insert(new_name, t);
        }
    }

    // Merge split tensors
    for (name, mut parts) in split_tensors {
        parts.sort_by_key(|(i, _)| *i);
        let tensors: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect();
        if let Ok(merged) = Tensor::cat(&tensors, 0) {
            data.insert(name, merged);
        } else {
            println!("Failed to merge split tensor: {}", name);
        }
    }

    // Fix shape mismatch for patch_embed.proj.weight
    if let Some(weight) = data.get("visual.patch_embed.proj.weight") {
        if weight.rank() == 4 {
            if let Ok(reshaped) = weight.unsqueeze(2)?.repeat((1, 1, 2, 1, 1)) {
                data.insert("visual.patch_embed.proj.weight".to_string(), reshaped);
                println!("[FIX] Reshaped visual.patch_embed.proj.weight to 5D");
            }
        }
    }
    
    Ok(VarBuilder::from_tensors(data, dtype, device))
}
