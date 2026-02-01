use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Embedding, VarBuilder};
use candle_core::quantized::{gguf_file, QMatMul};
use rayon::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
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

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self { Self { weight, eps } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.weight = self.weight.to_device(device)?.to_dtype(target_dtype)?;
        Ok(())
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let target_dtype = self.weight.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x.to_dtype(target_dtype)?.broadcast_mul(&self.weight)
    }
}

#[derive(Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    device: Device,
}

impl QLinear {
    pub fn new(inner: QMatMul, bias: Option<Tensor>, device: Device) -> Self { Self { inner, bias, device } }
    pub fn device(&self) -> &Device { &self.device }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if !self.device.same_device(device) {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            self.inner = match &self.inner {
                QMatMul::QTensor(q) => QMatMul::Tensor(q.dequantize(device)?.to_dtype(target_dtype)?),
                QMatMul::Tensor(t) => QMatMul::Tensor(t.to_device(device)?.to_dtype(target_dtype)?),
                _ => QMatMul::Tensor(self.inner.forward(&Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?)?.to_device(device)?.to_dtype(target_dtype)?),
            };
            if let Some(b) = &self.bias { self.bias = Some(b.to_device(device)?.to_dtype(target_dtype)?); }
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
        let out = match &self.inner {
            QMatMul::QTensor(_) => self.inner.forward(&xs_flat.to_dtype(DType::F32)?)?,
            QMatMul::Tensor(t) => self.inner.forward(&xs_flat.to_dtype(t.dtype())?)?,
            _ => self.inner.forward(&xs_flat.to_dtype(DType::F32)?)?,
        };
        let out = out.reshape((b, s, ()))?.to_dtype(target_dtype)?;
        if let Some(bias) = &self.bias { Ok(out.broadcast_add(bias)?) } else { Ok(out) }
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLTextAttention {
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear,
    pub q_norm: RmsNorm, pub k_norm: RmsNorm,
    pub num_attention_heads: usize, pub num_key_value_heads: usize, pub head_dim: usize,
    pub num_kv_groups: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
    pub layer_idx: usize,
}

impl QuantizedQwen3VLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj.to_device(device)?; self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?; self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?; self.k_norm.to_device(device)?;
        if let Some((k, v)) = &self.kv_cache {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            self.kv_cache = Some((k.to_device(device)?.to_dtype(target_dtype)?, v.to_device(device)?.to_dtype(target_dtype)?));
        }
        Ok(())
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let dev = self.q_proj.device();
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = xs.to_dtype(target_dtype)?;
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?.contiguous()?;
        let key_states = self.k_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?.contiguous()?;
        let value_states = self.v_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((pk, pv)) => (Tensor::cat(&[pk, &key_states], 2)?, Tensor::cat(&[pv, &value_states], 2)?),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        
        // [FIX] Causal Mask Scaling for Batched Context Baking
        let actual_kv_len = key_states.dim(2)?;
        let adjusted_mask = if let Some(m) = mask {
            let m_len = m.dim(D::Minus1)?;
            if m_len < actual_kv_len {
                // If mask is smaller than KV cache, pad it with zeros (allowed regions)
                let padding = Tensor::zeros((b_sz, 1, q_len, actual_kv_len - m_len), m.dtype(), m.device())?;
                Some(Tensor::cat(&[padding, m.clone()], D::Minus1)?)
            } else if m_len > actual_kv_len {
                Some(m.narrow(D::Minus1, 0, actual_kv_len)?)
            } else {
                Some(m.clone())
            }
        } else { None };

        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), adjusted_mask.as_ref(), self.scaling)?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None; }
    pub fn get_kv_len(&self) -> usize { self.kv_cache.as_ref().map(|(k, _)| k.dim(2).unwrap_or(0)).unwrap_or(0) }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLTextDecoderLayer {
    pub self_attn: QuantizedQwen3VLTextAttention,
    pub mlp_gate: Option<QLinear>, pub mlp_up: Option<QLinear>, pub mlp_down: Option<QLinear>,
    pub input_layernorm: RmsNorm, pub post_attention_layernorm: Option<RmsNorm>,
}

impl Module for QuantizedQwen3VLTextDecoderLayer {
    fn forward(&self, _xs: &Tensor) -> candle_core::Result<Tensor> {
        Err(candle_core::Error::Msg("Use custom forward with cos/sin/mask".to_string()))
    }
}

impl QuantizedQwen3VLTextDecoderLayer {
    pub fn forward_with_params(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask)?;
        let xs = residual.add(&xs)?;
        if let (Some(g), Some(u), Some(d), Some(n)) = (&self.mlp_gate, &self.mlp_up, &self.mlp_down, &self.post_attention_layernorm) {
            let residual = xs.clone();
            let xs = n.forward(&xs)?;
            let gate = candle_nn::ops::silu(&g.forward(&xs)?)?;
            let up = u.forward(&xs)?;
            let xs = d.forward(&gate.mul(&up)?)?;
            Ok(residual.add(&xs)?)
        } else { Ok(xs) }
    }
    pub fn new<R: std::io::Seek + std::io::Read>(config: &Qwen3VLTextConfig, ct: &gguf_file::Content, reader: &mut R, base_name: &str, device: &Device, dtype: DType, layer_idx: usize, baking_only: bool) -> Result<Self> {
        let is_gguf = base_name.starts_with("blk.");
        let (q, k, v, o, q_n, k_n) = if is_gguf { ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm") } else { ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm") };
        let self_attn = QuantizedQwen3VLTextAttention {
            q_proj: get_qlinear(ct, reader, &format!("{base_name}.{q}"), device, dtype)?,
            k_proj: get_qlinear(ct, reader, &format!("{base_name}.{k}"), device, dtype)?,
            v_proj: get_qlinear(ct, reader, &format!("{base_name}.{v}"), device, dtype)?,
            o_proj: get_qlinear(ct, reader, &format!("{base_name}.{o}"), device, dtype)?,
            q_norm: get_rms_norm(ct, reader, &format!("{base_name}.{q_n}"), config.rms_norm_eps, device, dtype)?,
            k_norm: get_rms_norm(ct, reader, &format!("{base_name}.{k_n}"), config.rms_norm_eps, device, dtype)?,
            num_attention_heads: config.num_attention_heads, num_key_value_heads: config.num_key_value_heads,
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads, head_dim: config.head_dim,
            scaling: 1f64 / f64::sqrt(config.head_dim as f64), kv_cache: None, layer_idx,
        };
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !baking_only {
            let (g, u, d, n) = if is_gguf { ("ffn_gate", "ffn_up", "ffn_down", "ffn_norm") } else { ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm") };
            (Some(get_qlinear(ct, reader, &format!("{base_name}.{g}"), device, dtype)?),
             Some(get_qlinear(ct, reader, &format!("{base_name}.{u}"), device, dtype)?),
             Some(get_qlinear(ct, reader, &format!("{base_name}.{d}"), device, dtype)?),
             Some(get_rms_norm(ct, reader, &format!("{base_name}.{n}"), config.rms_norm_eps, device, dtype)?))
        } else { (None, None, None, None) };
        let in_ln = if is_gguf { "attn_norm" } else { "input_layernorm" };
        let input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype)?;
        Ok(Self { self_attn, mlp_gate, mlp_up, mlp_down, input_layernorm, post_attention_layernorm })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?; self.input_layernorm.to_device(device)?;
        if let Some(g) = &mut self.mlp_gate { g.to_device(device)?; }
        if let Some(u) = &mut self.mlp_up { u.to_device(device)?; }
        if let Some(d) = &mut self.mlp_down { d.to_device(device)?; }
        if let Some(n) = &mut self.post_attention_layernorm { n.to_device(device)?; }
        Ok(())
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
    pub is_forced_cpu: bool,
}

impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(config: &Qwen3VLTextConfig, ct: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, _base_name: &str, device: &Device, _device_id: usize, dtype: DType, _kv_reserve: u64, baking_only: bool) -> Result<Self> {
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        
        // [9d1369 Parity] Dynamically detect hidden size from loaded tensor to prevent shape mismatch
        let (embed_tokens, actual_hidden_size) = if let Ok(tensor) = ct.tensor(&mut reader, "token_embd.weight", device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, "model.embed_tokens.weight", device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             let h = tensor.dim(1)?;
             (Embedding::new(tensor, h), h)
        } else {
             return Err(anyhow!("Failed to load token_embd.weight."));
        };

        if actual_hidden_size != config.hidden_size {
            println!("[MODEL-FIX] Hidden Size Mismatch Detected! Config: {}, File: {}. Patching config to match file...", config.hidden_size, actual_hidden_size);
        }

        // [CRITICAL] Apply patched config for ALL subsequent layer/norm initialization
        let mut patched_config = config.clone();
        patched_config.hidden_size = actual_hidden_size;
        let config = &patched_config;

        let mut layers = vec![];
        let num_layers = if baking_only { 1 } else { config.num_hidden_layers };
        for i in 0..num_layers {
            layers.push(QuantizedQwen3VLTextDecoderLayer::new(config, ct, &mut reader, &format!("blk.{i}"), device, dtype, i, baking_only)?);
        }
        let norm = get_rms_norm(ct, &mut reader, "output_norm", config.rms_norm_eps, device, dtype)?;
        let mut mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        
        // [STRUCTURAL-FIX] Head Dimension과 mRoPE 섹션 불일치 보정
        let half_head = config.head_dim / 2;
        let section_sum: usize = mrope_section.iter().sum();
        if section_sum != half_head && section_sum > 0 {
            let ratio = half_head as f32 / section_sum as f32;
            mrope_section = mrope_section.iter().map(|&s| (s as f32 * ratio).round() as usize).collect();
            // 합계가 정확히 맞지 않을 경우 마지막 섹션에서 조정
            let new_sum: usize = mrope_section.iter().sum();
            if new_sum != half_head {
                if let Some(last) = mrope_section.last_mut() {
                    *last = (*last as i32 + (half_head as i32 - new_sum as i32)) as usize;
                }
            }
            println!("[MODEL-FIX] Adjusted mRoPE sections to {:?} to match head_dim {}", mrope_section, config.head_dim);
        }

        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), mrope_section, mmap: mmap_handle, is_forced_cpu: device.is_cpu() })
    }
    pub fn forward(&mut self, embeds: &Tensor, seqlen_offset: usize, pos_ids: Option<&Tensor>, visual_mask: Option<&Tensor>, ds_embeds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (b, s, _h_dim) = embeds.dims3()?;
        let pos_ids = match pos_ids { Some(ids) => ids.clone(), None => Tensor::arange(seqlen_offset as u32, (s + seqlen_offset) as u32, embeds.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b, s))? };
        
        let (cos, sin) = self.rotary_emb.forward(&pos_ids, embeds.dtype(), self.mrope_section.clone())?;
        
        // [FIX] Prepare batched causal mask with offset awareness
        let mask = if s <= 1 { None } else { Some(prepare_causal_attention_mask(b, s, seqlen_offset, embeds.device())?) };
        
        let mut xs = embeds.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward_with_params(&xs, &cos, &sin, mask.as_ref())?;
            if let (Some(m), Some(ds)) = (visual_mask, ds_embeds.as_ref()) { if i < ds.len() { xs = mask_index_add(&xs.squeeze(0)?, &m.squeeze(0)?, &ds[i])?.unsqueeze(0)?; } }
        }
        Ok(xs.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { l.self_attn.clear_kv_cache(); } }
    pub fn get_kv_len(&self) -> usize { self.layers[0].self_attn.get_kv_len() }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let w = self.embed_tokens.embeddings().to_device(device)?;
        self.embed_tokens = Embedding::new(w, self.embed_tokens.hidden_size());
        for l in &mut self.layers { l.to_device(device)?; }
        self.norm.to_device(device)?;
        Ok(())
    }
        pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _: usize) -> Result<()> {
            if !path.exists() { fs::create_dir_all(path)?; }
            
            for (i, layer) in self.layers.iter_mut().enumerate() {
                if let Some((k, v)) = &layer.self_attn.kv_cache {
                    // [FILE-BRIDGE] 비대칭 압축 수행 후 저장
                    let res_k = Self::compress_to_bitkv_static(i, k)?;
                    let res_v = Self::compress_to_bitkv_static(i, v)?;
                    
                    let mut map = HashMap::new();
                    map.insert("k_a".to_string(), res_k.0); map.insert("k_p".to_string(), res_k.1);
                    map.insert("k_s".to_string(), res_k.2);
                    map.insert("v_a".to_string(), res_v.0); map.insert("v_p".to_string(), res_v.1);
                    map.insert("v_s".to_string(), res_v.2);
                    
                    let shape_tensor = Tensor::new(res_k.3.iter().map(|&x| x as u32).collect::<Vec<_>>().as_slice(), k.device())?;
                    map.insert("shape".to_string(), shape_tensor);
    
                    candle_core::safetensors::save(&map, path.join(format!("layer_{}_bitkv.safetensors", i)))?;
                }
                if clear { layer.self_attn.kv_cache = None; }
            }
            println!("[FILE-BRIDGE] Saved compressed KV cache to {:?}", path);
            Ok(())
        }
    
        pub fn load_kv_cache(&mut self, path: &Path, device: &Device, _: usize, _: usize) -> Result<()> {
            let target_device = if self.is_forced_cpu { Device::Cpu } else { device.clone() };
            for (i, layer) in self.layers.iter_mut().enumerate() {
                let p = path.join(format!("layer_{}_bitkv.safetensors", i));
                if p.exists() {
                    let map = candle_core::safetensors::load(p, &target_device)?;
                    let shape_vec: Vec<usize> = map.get("shape").unwrap().to_vec1::<u32>()?.into_iter().map(|x| x as usize).collect();
                    
                    let k_res = (map.get("k_a").unwrap().clone(), map.get("k_p").unwrap().clone(), map.get("k_s").unwrap().clone());
                    let v_res = (map.get("v_a").unwrap().clone(), map.get("v_p").unwrap().clone(), map.get("v_s").unwrap().clone());
                    
                    let (k, v) = Self::decompress_kv_static(i, k_res, v_res, &shape_vec, &target_device)?;
                    layer.self_attn.kv_cache = Some((k, v));
                }
            }
            println!("[FILE-BRIDGE] Loaded and decompressed KV cache from {:?}", path);
            Ok(())
        }
    
        // Helper static methods to avoid borrow conflicts
        fn compress_to_bitkv_static(layer_idx: usize, tensor: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
            let shape = tensor.dims().to_vec();
            let dev = tensor.device();
            
            if layer_idx == 0 {
                return Ok((tensor.clone(), Tensor::zeros(1, DType::U8, dev)?, Tensor::zeros(1, DType::F32, dev)?, shape));
            }
    
            let flat = tensor.flatten_all()?;
            let num_el = flat.dim(0)?;
    
            if layer_idx <= 4 {
                let scale = (flat.abs()?.max_all()?.to_scalar::<f32>()? / 3.0).max(1e-6);
                let quantized = ((flat / scale as f64)? + 1.0)?.round()?.clamp(0.0, 3.0)?;
                let q_vec = quantized.to_vec1::<f32>()?;
                let packed_size = (num_el + 3) / 4;
                let mut packed = vec![0u8; packed_size];
                for (i, &v) in q_vec.iter().enumerate() { packed[i / 4] |= (v as u8) << ((i % 4) * 2); }
                Ok((Tensor::zeros(1, tensor.dtype(), dev)?, Tensor::from_vec(packed, (packed_size,), dev)?, Tensor::new(&[scale], dev)?, shape))
            } else {
                let scale = flat.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
                let sign = flat.ge(0.0)?;
                let sign_vec = sign.to_vec1::<u8>()?;
                let packed_size = (num_el + 7) / 8;
                let mut packed = vec![0u8; packed_size];
                for (i, &s) in sign_vec.iter().enumerate() { if s > 0 { packed[i / 8] |= 1 << (i % 8); } }
                Ok((Tensor::zeros(1, tensor.dtype(), dev)?, Tensor::from_vec(packed, (packed_size,), dev)?, Tensor::new(&[scale], dev)?, shape))
            }
        }
    
    fn decompress_kv_static(
        layer_idx: usize, 
        k_res: (Tensor, Tensor, Tensor), 
        v_res: (Tensor, Tensor, Tensor), 
        original_shape: &[usize], 
        target_device: &Device
    ) -> Result<(Tensor, Tensor)> {
        let dtype = if target_device.is_cpu() { DType::F32 } else { DType::BF16 };

        let decompress = |res: (Tensor, Tensor, Tensor)| -> Result<Tensor> {
            if layer_idx == 0 { 
                return Ok(res.0.to_device(target_device)?.to_dtype(dtype)?); 
            }
            
            let s = res.2.to_scalar::<f32>()?;
            let packed_vec = res.1.to_vec1::<u8>()?; 
            let total_el: usize = original_shape.iter().product();
            let mut out = vec![0.0f32; total_el];

            if layer_idx <= 4 {
                for i in 0..total_el {
                    let val = (packed_vec[i / 4] >> ((i % 4) * 2)) & 0x03;
                    out[i] = (val as f32 - 1.0) * s;
                }
            } else {
                for i in 0..total_el {
                    let sign = (packed_vec[i / 8] >> (i % 8)) & 0x01;
                    out[i] = if sign == 1 { s } else { -s };
                }
            }
            Ok(Tensor::from_vec(out, original_shape, &Device::Cpu)?
                .to_device(target_device)?
                .to_dtype(dtype)?
                .contiguous()?)
        };

        Ok((decompress(k_res)?, decompress(v_res)?))
    }

    pub fn compress_to_bitkv(&self, layer_idx: usize, tensor: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        Self::compress_to_bitkv_static(layer_idx, tensor)
    }

    pub fn inject_live_kv_bitkv(
        &mut self, 
        layer_idx: usize, 
        k_res: (Tensor, Tensor, Tensor), 
        v_res: (Tensor, Tensor, Tensor), 
        original_shape: &[usize]
    ) -> Result<()> {
        // [FIX] 현재 모델의 실제 디바이스를 참조하여 텐서 생성 위치 결정
        let target_device = if self.is_forced_cpu { 
            Device::Cpu 
        } else { 
            self.embed_tokens.embeddings().device().clone() 
        };

        let (mut k_dec, mut v_dec) = Self::decompress_kv_static(layer_idx, k_res, v_res, original_shape, &target_device)?;

        if let Some(layer) = self.layers.get_mut(layer_idx) {
            let target_heads = layer.self_attn.num_key_value_heads;
            let current_shape = k_dec.dims().to_vec();
            let current_heads = current_shape[1];

            if current_heads != target_heads {
                println!("[RELAY-FIX] Aligning heads: {} -> {}", current_heads, target_heads);
                
                let expand = |t: Tensor| -> Result<Tensor> {
                    if current_heads < target_heads && target_heads % current_heads == 0 {
                        let repeats = target_heads / current_heads;
                        let mut res = t.clone();
                        for _ in 1..repeats {
                            res = Tensor::cat(&[res, t.clone()], 1)?;
                        }
                        Ok(res.contiguous()?)
                    } else if current_heads > target_heads {
                        Ok(t.narrow(1, 0, target_heads)?.contiguous()?)
                    } else {
                        Ok(t.contiguous()?)
                    }
                };
                k_dec = expand(k_dec)?;
                v_dec = expand(v_dec)?;
            }

            layer.self_attn.kv_cache = Some((k_dec, v_dec));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3VLModel {
    pub config: Qwen3VLConfig, pub visual: Qwen3VLVisionModel, 
    pub language_model: QuantizedQwen3VLTextModel, pub lm_head: QLinear,
    pub text_device: Device, pub vision_device: Device,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(config: &Qwen3VLConfig, ct_main: &gguf_file::Content, main_mmap: Option<Arc<Mmap>>, _ct_vision: &gguf_file::Content, _vision_mmap: Option<Arc<Mmap>>, text_device: &Device, text_device_id: usize, vision_device: &Device, _vision_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool) -> Result<Self> {
        let visual = Qwen3VLVisionModel::new(config.vision_config.as_ref().unwrap().clone(), VarBuilder::from_tensors(HashMap::new(), dtype, vision_device))?;
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(config.text_config.as_ref().unwrap(), ct_main, main_mmap.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        let mut m_reader = std::io::Cursor::new(main_mmap.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lm_head = get_qlinear(ct_main, &mut m_reader, "lm_head", text_device, dtype)?;
        Ok(Self { config: config.clone(), visual, language_model, lm_head, text_device: text_device.clone(), vision_device: vision_device.clone() })
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

    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _video_pixel_values: Option<&Tensor>, _video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        if let (Some(pv), Some(thw)) = (pixel_values, image_grid_thw) {
            let (image_embeds, _) = self.get_vision_features(pv, thw)?;
            let image_embeds = Tensor::cat(&image_embeds, 0)?;
            let vision_mask = self.get_placeholder_mask(input_ids, true)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
        }

        let start = if let Some(cp) = cache_position { cp.flatten_all()?.i(0)?.to_scalar::<u32>()? } else { seqlen_offset as u32 };
        let pos_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, Some(&pos_ids), None, None)?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        Ok(self.lm_head.forward(&hidden_state)?)
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; 
        self.text_device = device.clone(); self.vision_device = device.clone();
        Ok(())
    }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, b) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

#[derive(Clone)]
pub struct QuantizedQwen3TextModel {
    pub language_model: QuantizedQwen3VLTextModel, pub lm_head: Option<QLinear>, pub text_device: Device,
}

impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(config: &Qwen3VLConfig, ct: &gguf_file::Content, mmap: Option<Arc<Mmap>>, text_device: &Device, text_device_id: usize, dtype: DType, kv_reserve: u64, baking_only: bool, _single: bool) -> Result<Self> {
        let language_model = QuantizedQwen3VLTextModel::new_with_mmap(config.text_config.as_ref().unwrap(), ct, mmap.clone(), "model", text_device, text_device_id, dtype, kv_reserve, baking_only)?;
        let mut reader = std::io::Cursor::new(mmap.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lm_head = get_qlinear(ct, &mut reader, "lm_head", text_device, dtype).ok();
        Ok(Self { language_model, lm_head, text_device: text_device.clone() })
    }

    pub fn forward(&mut self, input_ids: &Tensor, cache_position: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        let start = if let Some(cp) = cache_position { cp.flatten_all()?.i(0)?.to_scalar::<u32>()? } else { seqlen_offset as u32 };
        let pos_ids = Tensor::arange(start, start + seq_len as u32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_sz, seq_len))?;
        
        let outputs = self.language_model.forward(&inputs_embeds, seqlen_offset, Some(&pos_ids), None, None)?;
        let hidden_state = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        if let Some(head) = &self.lm_head {
            Ok(head.forward(&hidden_state)?)
        } else {
            Ok(hidden_state)
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.language_model.to_device(device)?; if let Some(h) = &mut self.lm_head { h.to_device(device)?; } 
        self.text_device = device.clone();
        Ok(())
    }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(path, clear, b) }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(path, device, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

pub fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let bias = ct.tensor(reader, &format!("{name}.bias"), device).ok().map(|t| t.dequantize(device).unwrap().to_dtype(dtype).unwrap());
    Ok(QLinear::new(QMatMul::from_qtensor(weight)?, bias, device.clone()))
}

pub fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType) -> Result<RmsNorm> {
    let w = ct.tensor(reader, &format!("{name}.weight"), device)?.dequantize(device)?.to_dtype(dtype)?;
    Ok(RmsNorm::new(w, eps))
}

pub fn load_tensors_from_true_iq0(path: &Path, device: &Device, dtype: DType, baking_only: bool) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&file)?;
    let mut data = HashMap::new();
    println!("[MODEL] Unpacking Safetensors: {:?} (Baking Only: {})", path, baking_only);
    for (name, view) in st.tensors() {
        if name.ends_with(".scales") || name.ends_with(".scale") || name.ends_with(".shape") { continue; }

        let is_packed = name.ends_with(".packed");
        let base_name = if is_packed { name.strip_suffix(".packed").unwrap() } else { &name };
        
        // [FIX] Baking 모드일 때 0번 레이어 외에는 로드 건너뜀 (문자열 파싱 방식)
        if baking_only {
            let skip = if base_name.contains(".layers.") || base_name.contains(".blocks.") || base_name.contains("blk.") {
                // 인덱스가 .0. 이 아닌 모든 경우 제외
                !base_name.contains(".0.")
            } else {
                false // 공통 가중치(임베딩 등)는 유지
            };
            if skip { continue; }
        }
        
        let mut target_name = base_name.to_string();
        
        if base_name.starts_with("v.") {
            let rest = &base_name[2..];
            if rest.starts_with("blk.") {
                let parts: Vec<&str> = rest[4..].splitn(2, '.').collect();
                let mapped = match parts[1] {
                    s if s.starts_with("ln1") => s.replace("ln1", "norm1"), s if s.starts_with("ln2") => s.replace("ln2", "norm2"),
                    s if s.starts_with("attn_qkv") => s.replace("attn_qkv", "attn.qkv"), s if s.starts_with("attn_out") => s.replace("attn_out", "attn.proj"),
                    s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.linear_fc1"), s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"), _ => parts[1].to_string()
                };
                target_name = format!("visual.blocks.{}.{}", parts[0], mapped);
            } else if rest.starts_with("patch_embd") { target_name = "visual.patch_embed.proj".to_string(); }
            else if rest.starts_with("position_embd") { target_name = "visual.pos_embed".to_string(); }
            else if rest.starts_with("post_ln") { target_name = "visual.merger.norm".to_string(); }
            else { target_name = format!("visual.{}", rest); }
        } else if base_name.starts_with("mm.") {
            let rest = &base_name[3..];
            if rest.starts_with("0") { target_name = "visual.merger.linear_fc1".to_string(); }
            else if rest.starts_with("2") { target_name = "visual.merger.linear_fc2".to_string(); }
        } else if base_name.starts_with("blk.") || base_name.starts_with("model.layers.") {
            // [FIX] 'blk.' 와 'model.layers.' 두 가지 접두사 모두 지원
            let rest = if base_name.starts_with("blk.") { &base_name[4..] } else { &base_name[13..] };
            let parts: Vec<&str> = rest.splitn(2, '.').collect();
            
            let mapped = match parts[1] {
                "attn_q_norm.weight" => "self_attn.q_norm.weight".to_string(),
                "attn_k_norm.weight" => "self_attn.k_norm.weight".to_string(),
                s if s.starts_with("attn_q") => s.replace("attn_q", "self_attn.q_proj"), s if s.starts_with("attn_k") => s.replace("attn_k", "self_attn.k_proj"),
                s if s.starts_with("attn_v") => s.replace("attn_v", "self_attn.v_proj"), s if s.starts_with("attn_output") => s.replace("attn_output", "self_attn.o_proj"),
                s if s.starts_with("attn_norm") => s.replace("attn_norm", "input_layernorm"), s if s.starts_with("ffn_norm") => s.replace("ffn_norm", "post_attention_layernorm"),
                s if s.starts_with("ffn_gate") => s.replace("ffn_gate", "mlp.gate_proj"), s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.up_proj"),
                s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.down_proj"), _ => parts[1].to_string()
            };
            target_name = format!("language_model.layers.{}.{}", parts[0], mapped);
        } else if base_name.contains("token_embd") || base_name.eq("model.embed_tokens.weight") {
            target_name = "language_model.embed_tokens.weight".to_string();
        } else if base_name.contains("output_norm") || base_name.eq("model.norm.weight") {
            target_name = "language_model.norm.weight".to_string();
        } else if base_name.contains("lm_head") || base_name == "output.weight" {
            target_name = "lm_head.weight".to_string();
        } else if base_name.contains("output.bias") {
            target_name = "lm_head.bias".to_string();
        }

        let final_tensor = if is_packed {
            // [FIX] 파일 내부에 명시된 .shape 텐서를 읽어와서 정확한 형상 복원
            let shape_name = format!("{}.shape", base_name);
            let shape: Vec<usize> = if let Ok(shape_view) = st.tensor(&shape_name) {
                let shape_data = shape_view.data();
                shape_data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                    .collect()
            } else {
                // 백업: .shape가 없으면 기본 추측 (이미 체크하고 있으므로 거의 발생 안 함)
                view.shape().to_vec()
            };

            if base_name.contains("token_embd") || base_name.contains("embed_tokens") {
                // [V2] 2-bit Unpack (4 weights per byte)
                let scale = f32::from_le_bytes(st.tensor(&format!("{base_name}.scale"))?.data()[0..4].try_into().unwrap());
                let min_val = if let Ok(min_tensor) = st.tensor(&format!("{base_name}.min")) {
                    f32::from_le_bytes(min_tensor.data()[0..4].try_into().unwrap())
                } else {
                    println!("[MODEL-WARNING] Missing .min for {}, defaulting to 0.0", base_name);
                    0.0f32
                };
                let mut unpacked = vec![0.0f32; shape.iter().product()];
                let packed_vec = view.data();
                
                for (byte_idx, &byte) in packed_vec.iter().enumerate() {
                    for i in 0..4 {
                        let idx = byte_idx * 4 + i;
                        if idx < unpacked.len() {
                            let q_val = (byte >> (i * 2)) & 0x03;
                            unpacked[idx] = (q_val as f32 * scale) + min_val;
                        }
                    }
                }
                Tensor::from_vec(unpacked, shape.as_slice(), &Device::Cpu)?.to_device(device)?.to_dtype(dtype)?
            } else {
                // [V2] 1-bit Unpack with 512-bit Block Alignment
                let scales_vec: Vec<f32> = st.tensor(&format!("{base_name}.scales"))?.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
                let mut decoded = vec![0.0f32; shape.iter().product()];
                let packed_vec = view.data();
                
                // Parallel Decode: Each 64-byte block (512 bits) processed independently
                decoded.par_chunks_mut(512).enumerate().for_each(|(b_idx, b_out)| {
                    if b_idx < scales_vec.len() {
                        let s = scales_vec[b_idx]; 
                        let p_start = b_idx * 64; // 512 / 8 = 64 bytes
                        let chunk_len = b_out.len(); // 512 or less for the last block
                        
                        for i in 0..64 {
                            let gp = p_start + i; 
                            if gp < packed_vec.len() {
                                let byte = packed_vec[gp];
                                for bit in 0..8 {
                                    let idx = i * 8 + bit;
                                    if idx < chunk_len {
                                        b_out[idx] = if (byte & (1 << bit)) != 0 { s } else { -s };
                                    }
                                }
                            }
                        }
                    }
                });
                Tensor::from_vec(decoded, shape.as_slice(), &Device::Cpu)?
                    .to_device(device)?
                    .to_dtype(dtype)?
                    .contiguous()?
            }
        } else {
            // [FIX] 추측하지 말고 safetensors의 메타데이터(view.dtype())를 직접 참조
            let raw = view.data();
            let final_tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let f32_data: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    Tensor::from_vec(f32_data, view.shape(), &Device::Cpu)?
                },
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {
                    let f16_data: Vec<half::f16> = raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect();
                    Tensor::from_vec(f16_data, view.shape(), &Device::Cpu)?
                },
                safetensors::Dtype::I8 => {
                    // candle doesn't support i8, convert to u8 (same bits)
                    let u8_data: Vec<u8> = raw.to_vec();
                    Tensor::from_vec(u8_data, view.shape(), &Device::Cpu)?
                },
                safetensors::Dtype::U8 => {
                    Tensor::from_vec(raw.to_vec(), view.shape(), &Device::Cpu)?
                },
                safetensors::Dtype::I32 => {
                    // candle doesn't support i32, convert to i64
                    let i64_data: Vec<i64> = raw.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect();
                    Tensor::from_vec(i64_data, view.shape(), &Device::Cpu)?
                },
                _ => return Err(anyhow!("Unsupported dtype in safetensors: {:?} for tensor {}", view.dtype(), name)),
            };

            final_tensor
                .to_device(device)?
                .to_dtype(dtype)?
                .contiguous()?
        };

        if !target_name.starts_with("lm_head") {
            data.insert(format!("model.{}", target_name), final_tensor);
        } else {
            data.insert(target_name, final_tensor);
        }
    }
    Ok(data)
}

pub fn from_true_iq0_safetensors(path: &Path, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
    let data = load_tensors_from_true_iq0(path, device, dtype, false)?;
    Ok(VarBuilder::from_tensors(data, dtype, device))
}
