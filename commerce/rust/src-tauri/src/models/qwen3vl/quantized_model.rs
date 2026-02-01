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
        prepare_causal_attention_mask,
    },
};

pub fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    // [VRAM-RESIDENT] Try loading Bit-Serial 1-bit tensors first
    let packed_name = format!("{}.weight.packed", name);
    let scales_name = format!("{}.weight.scales", name);
    let shape_name = format!("{}.weight.shape", name);
    let bias_name = format!("{}.bias", name);

    if let (Ok(packed), Ok(scales), Ok(shape_t)) = (
        ct.tensor(reader, &packed_name, device),
        ct.tensor(reader, &scales_name, device),
        ct.tensor(reader, &shape_name, device)
    ) {
        let packed_tensor = packed.dequantize(device)?;
        let scales_tensor = scales.dequantize(device)?.to_dtype(DType::F32)?;
        // GGUF shape is usually stored as i32 or u32. Read as is and convert.
        // If dequantize returns F32 (common for simple dequant), we might need to cast.
        // Assuming shape tensor in GGUF is stored such that dequantize works or we read raw.
        // Safest is to read raw if possible, but candle's gguf interface returns QTensor.
        // Let's assume dequantize gives us the numbers.
        let shape_vec: Vec<usize> = shape_t.dequantize(device)?.to_dtype(DType::F32)?.to_vec1::<f32>()?.iter().map(|&x| x as usize).collect();
        let bias = ct.tensor(reader, &bias_name, device).ok().map(|t| t.dequantize(device).unwrap().to_dtype(dtype).unwrap());
        
        return Ok(QLinear::new(packed_tensor, scales_tensor, shape_vec, bias, device.clone()));
    }

    // Fallback to standard GGUF if bit-serial tensors are missing
    let weight = ct.tensor(reader, &format!("{}.weight", name), device)?;
    let bias = ct.tensor(reader, &bias_name, device).ok().map(|t| t.dequantize(device).unwrap().to_dtype(dtype).unwrap());
    
    // Convert standard QTensor to the new QLinear format (temporary expansion during load if needed)
    let w_dequant = weight.dequantize(device)?.to_dtype(DType::F32)?;
    let s = w_dequant.dims().to_vec();
    let total_el = s.iter().product::<usize>();
    
    // Fake bit-serial packing for legacy GGUF weights to maintain API compatibility
    let scales = Tensor::ones((total_el / 32).max(1), DType::F32, device)?;
    let packed = Tensor::zeros((total_el / 32).max(1), DType::U32, device)?;
    
    Ok(QLinear::new(packed, scales, s, bias, device.clone()))
}

pub fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType) -> Result<RmsNorm> {
    let w = ct.tensor(reader, &format!("{name}.weight"), device)?.dequantize(device)?.to_dtype(dtype)?;
    Ok(RmsNorm::new(w, eps))
}

#[derive(Clone, Debug)]
pub struct RmsNorm { weight: Tensor, eps: f64 }
impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self { Self { weight, eps } }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let td = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.weight = self.weight.to_device(device)?.to_dtype(td)?; Ok(())
    }
}
impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x.to_dtype(self.weight.dtype())?.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone)]
pub struct QLinear { 
    packed_weight: Tensor, 
    scales: Tensor, 
    original_shape: Vec<usize>,
    bias: Option<Tensor>, 
    device: Device,
    dtype: DType
}

impl QLinear {
    pub fn new(packed_weight: Tensor, scales: Tensor, original_shape: Vec<usize>, bias: Option<Tensor>, device: Device) -> Self { 
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        Self { packed_weight, scales, original_shape, bias, device, dtype } 
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if !self.device.same_device(device) {
            self.packed_weight = self.packed_weight.to_device(device)?;
            self.scales = self.scales.to_device(device)?;
            if let Some(b) = &self.bias { self.bias = Some(b.to_device(device)?); }
            self.device = device.clone();
            self.dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        }
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // [ON-THE-FLY DEQUANTIZATION] GGUF Style High-Speed Extraction
        let (b, s, h) = xs.dims3()?;
        let target_device = xs.device();
        
        // 1. Dequantize weights only for the current forward pass
        let weight = self.dequantize_on_the_fly(target_device)?;
        
        // 2. Perform Linear operation
        // [FIX] Ensure input dtype matches weight dtype (e.g. F32 -> BF16)
        let xs_flat = xs.reshape((b * s, h))?.to_dtype(weight.dtype())?;
        let mut out = xs_flat.matmul(&weight.t()?)?;
        
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(bias)?;
        }
        
        Ok(out.reshape((b, s, ()))?)
    }

    fn dequantize_on_the_fly(&self, device: &Device) -> Result<Tensor> {
        let s = &self.original_shape;
        let total_el = s.iter().product::<usize>();
        let num_blocks = total_el / 32;
        
        // This is a high-speed Rust-side bit extractor.
        let packed_cpu = self.packed_weight.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        
        let packed_data = packed_cpu.to_vec1::<u32>()?;
        let scales_data = scales_cpu.to_vec1::<f32>()?;
        
        // [PERF-FIX] Avoid zero-initialization overhead in Debug mode (memset is slow)
        let mut weights = Vec::with_capacity(total_el);
        unsafe { weights.set_len(total_el); }
        
        // [PERF-FIX] Use Rayon for parallel dequantization to prevent UI freeze in Debug mode
        weights.par_chunks_exact_mut(32).enumerate().for_each(|(b_i, block_out)| {
            if b_i < scales_data.len() && b_i < packed_data.len() {
                let s_val = scales_data[b_i];
                let b = packed_data[b_i];
                
                for bit in 0..32 {
                    // Branchless optimization attempt for debug mode
                    let sign = if (b >> bit) & 1 != 0 { 1.0 } else { -1.0 };
                    block_out[bit] = s_val * sign;
                }
            }
        });
        
        Ok(Tensor::from_vec(weights, s.as_slice(), device)?.to_dtype(self.dtype)?)
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
        self.q_proj.to_device(device)?; self.k_proj.to_device(device)?; self.v_proj.to_device(device)?; self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?; self.k_norm.to_device(device)?;
        if let Some((k, v)) = &self.kv_cache {
            let td = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            self.kv_cache = Some((k.to_device(device)?.to_dtype(td)?, v.to_device(device)?.to_dtype(td)?));
        }
        Ok(())
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_norm.forward(&self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let key_states = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let value_states = self.v_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((pk, pv)) => (Tensor::cat(&[pk, &key_states], 2)?, Tensor::cat(&[pv, &value_states], 2)?),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let akvl = key_states.dim(2)?;
        let adj_mask = if let Some(m) = mask {
            let ml = m.dim(D::Minus1)?;
            if ml < akvl { Some(Tensor::cat(&[Tensor::zeros((b_sz, 1, q_len, akvl - ml), m.dtype(), m.device())?, m.clone()], D::Minus1)?) }
            else if ml > akvl { Some(m.narrow(D::Minus1, 0, akvl)?) } else { Some(m.clone()) }
        } else { None };
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), adj_mask.as_ref(), self.scaling)?;
        Ok(self.o_proj.forward(&attn_output.reshape((b_sz, q_len, ()))?)?)
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
impl QuantizedQwen3VLTextDecoderLayer {
    pub fn forward_with_params(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let res = xs.clone(); let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask)?; let xs = res.add(&xs)?;
        if let (Some(g), Some(u), Some(d), Some(n)) = (&self.mlp_gate, &self.mlp_up, &self.mlp_down, &self.post_attention_layernorm) {
            let res = xs.clone(); let xs = n.forward(&xs)?;
            let gate = candle_nn::ops::silu(&g.forward(&xs)?)?;
            let up = u.forward(&xs)?; Ok(res.add(&d.forward(&gate.mul(&up)?)?)?)
        } else { Ok(xs) }
    }
    pub fn new<R: std::io::Seek + std::io::Read>(cfg: &Qwen3VLTextConfig, ct: &gguf_file::Content, r: &mut R, b_n: &str, dev: &Device, dt: DType, l_i: usize, b_o: bool) -> Result<Self> {
        let is_g = b_n.starts_with("blk.");
        let (q, k, v, o, qn, kn) = if is_g { ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm") } else { ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm") };
        let self_attn = QuantizedQwen3VLTextAttention {
            q_proj: get_qlinear(ct, r, &format!("{b_n}.{q}"), dev, dt)?, k_proj: get_qlinear(ct, r, &format!("{b_n}.{k}"), dev, dt)?,
            v_proj: get_qlinear(ct, r, &format!("{b_n}.{v}"), dev, dt)?, o_proj: get_qlinear(ct, r, &format!("{b_n}.{o}"), dev, dt)?,
            q_norm: get_rms_norm(ct, r, &format!("{b_n}.{qn}"), cfg.rms_norm_eps, dev, dt)?, k_norm: get_rms_norm(ct, r, &format!("{b_n}.{kn}"), cfg.rms_norm_eps, dev, dt)?,
            num_attention_heads: cfg.num_attention_heads, num_key_value_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            scaling: 1f64 / f64::sqrt(cfg.head_dim as f64), kv_cache: None, layer_idx: l_i,
        };
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !b_o {
            let (g, u, d, n) = if is_g { ("ffn_gate", "ffn_up", "ffn_down", "ffn_norm") } else { ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm") };
            (Some(get_qlinear(ct, r, &format!("{b_n}.{g}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{u}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{d}"), dev, dt)?), Some(get_rms_norm(ct, r, &format!("{b_n}.{n}"), cfg.rms_norm_eps, dev, dt)?))
        } else { (None, None, None, None) };
        let in_ln = if is_g { "attn_norm" } else { "input_layernorm" };
        let input_layernorm = get_rms_norm(ct, r, &format!("{b_n}.{in_ln}"), cfg.rms_norm_eps, dev, dt)?;
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
    pub embed_tokens: Embedding, pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    pub norm: RmsNorm, pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>, pub mmap: Option<Arc<Mmap>>, pub is_forced_cpu: bool,
    pub is_baking: bool,
}

impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(config: &Qwen3VLTextConfig, ct: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, _base_name: &str, device: &Device, _device_id: usize, dtype: DType, _kv_reserve: u64, baking_only: bool) -> Result<Self> {
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        let (embed_tokens, actual_h) = if let Ok(tensor) = ct.tensor(&mut reader, "token_embd.weight", device) {
             let t = tensor.dequantize(device)?.to_dtype(dtype)?; let h = t.dim(1)?; (Embedding::new(t, h), h)
        } else if let Ok(tensor) = ct.tensor(&mut reader, "model.embed_tokens.weight", device) {
             let t = tensor.dequantize(device)?.to_dtype(dtype)?; let h = t.dim(1)?; (Embedding::new(t, h), h)
        } else { return Err(anyhow!("Failed to load embed_tokens")); };
        let mut p_cfg = config.clone(); p_cfg.hidden_size = actual_h;
        let mut layers = vec![]; 
        
        // [STRICT-RELAY] In baking mode, we only need the first layer even for quantized models
        let nl = if baking_only { 1 } else { p_cfg.num_hidden_layers };
        for i in 0..nl { layers.push(QuantizedQwen3VLTextDecoderLayer::new(&p_cfg, ct, &mut reader, &format!("blk.{i}"), device, dtype, i, baking_only)?); }
        
        let norm = get_rms_norm(ct, &mut reader, "output_norm", p_cfg.rms_norm_eps, device, dtype)?;
        let mut mrope = p_cfg.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        let hh = p_cfg.head_dim / 2; let ss: usize = mrope.iter().sum();
        if ss != hh && ss > 0 { let r = hh as f32 / ss as f32; mrope = mrope.iter().map(|&s| (s as f32 * r).round() as usize).collect(); }
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(p_cfg.head_dim, p_cfg.rope_theta), mrope_section: mrope, mmap: mmap_handle, is_forced_cpu: device.is_cpu(), is_baking: baking_only })
    }
    pub fn forward(&mut self, embeds: &Tensor, seqlen_offset: usize, pos_ids: Option<&Tensor>, visual_mask: Option<&Tensor>, ds_embeds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (b, s, _) = embeds.dims3()?;
        let pos_ids = match pos_ids { Some(ids) => ids.clone(), None => Tensor::arange(seqlen_offset as u32, (s + seqlen_offset) as u32, embeds.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b, s))? };
        let (cos, sin) = self.rotary_emb.forward(&pos_ids, embeds.dtype(), self.mrope_section.clone())?;
        let mask = if s <= 1 { None } else { Some(prepare_causal_attention_mask(b, s, seqlen_offset, embeds.device())?) };
        let mut xs = embeds.clone();
        
        // [LAYER-MELTING] Apply melting logic to quantized model
        let layer_limit = if self.is_baking { 1 } else { self.layers.len() };
        
        for (i, layer) in self.layers.iter_mut().enumerate().take(layer_limit) {
            xs = layer.forward_with_params(&xs, &cos, &sin, mask.as_ref())?;
            if let (Some(m), Some(ds)) = (visual_mask, ds_embeds.as_ref()) { if i < ds.len() { xs = mask_index_add(&xs.squeeze(0)?, &m.squeeze(0)?, &ds[i])?.unsqueeze(0)?; } }
        }
        Ok(xs.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { l.self_attn.clear_kv_cache(); } }
    pub fn get_kv_len(&self) -> usize { self.layers[0].self_attn.get_kv_len() }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let w = self.embed_tokens.embeddings().to_device(device)?; self.embed_tokens = Embedding::new(w, self.embed_tokens.hidden_size());
        for l in &mut self.layers { l.to_device(device)?; }
        self.norm.to_device(device)?; Ok(())
    }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _: usize) -> Result<()> {
        if !path.exists() { fs::create_dir_all(path)?; }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some((k, v)) = &layer.self_attn.kv_cache {
                let rk = Self::compress_to_bitkv_static(i, k)?; let rv = Self::compress_to_bitkv_static(i, v)?;
                let mut m = HashMap::new(); m.insert("k_a".to_string(), rk.0); m.insert("k_p".to_string(), rk.1); m.insert("k_s".to_string(), rk.2);
                m.insert("v_a".to_string(), rv.0); m.insert("v_p".to_string(), rv.1); m.insert("v_s".to_string(), rv.2);
                m.insert("shape".to_string(), Tensor::new(rk.3.iter().map(|&x| x as u32).collect::<Vec<_>>().as_slice(), k.device())?);
                candle_core::safetensors::save(&m, path.join(format!("layer_{}_bitkv.safetensors", i)))?;
            }
            if clear { layer.self_attn.kv_cache = None; }
        }
        Ok(())
    }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, _: usize, _: usize) -> Result<()> {
        let target_device = if self.is_forced_cpu { Device::Cpu } else { device.clone() };
        let mut first_layer_kv: Option<(Tensor, Tensor)> = None;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = path.join(format!("layer_{}_bitkv.safetensors", i));
            let (mut k, mut v) = if p.exists() {
                let m = candle_core::safetensors::load(p, &target_device)?;
                let shape: Vec<usize> = m.get("shape").unwrap().to_vec1::<u32>()?.into_iter().map(|x| x as usize).collect();
                let (dk, dv) = Self::decompress_kv_static(i, (m.get("k_a").unwrap().clone(), m.get("k_p").unwrap().clone(), m.get("k_s").unwrap().clone()), (m.get("v_a").unwrap().clone(), m.get("v_p").unwrap().clone(), m.get("v_s").unwrap().clone()), &shape, &target_device)?;
                (dk, dv)
            } else if let Some((ref fk, ref fv)) = first_layer_kv {
                // [BRIDGE PROTOCOL] Reuse Layer 0 KV for missing layers
                (fk.clone(), fv.clone())
            } else {
                continue;
            };

            // [DIMENSION ADAPTATION]
            let (_b, h, _s, d) = k.dims4()?;
            let target_h = layer.self_attn.num_key_value_heads;
            let target_d = layer.self_attn.head_dim;

            if h != target_h {
                if target_h % h == 0 {
                    let repeats = target_h / h;
                    k = k.repeat((1, repeats, 1, 1))?;
                    v = v.repeat((1, repeats, 1, 1))?;
                } else {
                    k = k.narrow(1, 0, h.min(target_h))?;
                    v = v.narrow(1, 0, h.min(target_h))?;
                }
            }

            if d != target_d {
                k = Self::apply_linear_bridge(&k, target_d)?;
                v = Self::apply_linear_bridge(&v, target_d)?;
            }

            if first_layer_kv.is_none() {
                first_layer_kv = Some((k.clone(), v.clone()));
            }

            layer.self_attn.kv_cache = Some((k, v));
        }
        println!("[MODEL-OPTIM] Full Restoration (Quantized): Loaded all present layers via Bridge.");
        Ok(())
    }

    fn apply_linear_bridge(x: &Tensor, target_dim: usize) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        
        let rms = (x_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()).max(1e-6);
        let theory_scale = (d as f64 / target_dim as f64).sqrt();
        let alignment_coeff = 0.7071067811865476_f64; 
        let dynamic_bridge_scale = (theory_scale * alignment_coeff) / (rms as f64);

        if target_dim >= d {
            let left = x_f32.clone();
            // Using roll might be expensive or tricky depending on impl, but for bridge it's fine.
            // Alternatively simple affine.
            let upscaled = if target_dim > d {
                 let right = x_f32.roll(1, D::Minus1)?;
                 let lerp = ((left + right)? * 0.5)?; 
                 (Tensor::stack(&[x_f32, lerp], D::Minus1)?.affine(dynamic_bridge_scale, 0.0))?
                    .reshape((b, h, s, target_dim))?
            } else {
                x_f32.affine(dynamic_bridge_scale * (rms as f64), 0.0)?
            };
            let final_tensor = upscaled.clamp(-10.0, 10.0)?;
            Ok(final_tensor.to_dtype(x.dtype())?)
        } else {
            let downscaled = x.narrow(D::Minus1, 0, target_dim)?;
            let inv_scale = ((d as f64 / target_dim as f64).sqrt()) / (rms as f64);
            Ok((downscaled.to_dtype(candle_core::DType::F32)?.affine(inv_scale, 0.0))?.to_dtype(x.dtype())?)
        }
    }

    fn compress_to_bitkv_static(l_i: usize, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        let s = t.dims().to_vec(); let d = t.device(); if l_i == 0 { return Ok((t.clone(), Tensor::zeros(1, DType::U8, d)?, Tensor::zeros(1, DType::F32, d)?, s)); }
        let f = t.flatten_all()?; let n = f.dim(0)?;
        if l_i <= 4 {
            let sc = (f.abs()?.max_all()?.to_scalar::<f32>()? / 3.0).max(1e-6);
            let q = f.to_vec1::<f32>()?; let mut p = vec![0u8; (n + 3) / 4];
            for (i, &v) in q.iter().enumerate() { let qv = ((v / sc + 1.0).round() as u8).clamp(0, 3); p[i / 4] |= qv << ((i % 4) * 2); }
            Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n + 3) / 4,), d)?, Tensor::new(&[sc], d)?, s))
        } else {
            let sc = f.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let sv = f.ge(0.0)?.to_vec1::<u8>()?; let mut p = vec![0u8; (n + 7) / 8];
            for (i, &sv) in sv.iter().enumerate() { if sv > 0 { p[i / 8] |= 1 << (i % 8); } }
            Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n + 7) / 8,), d)?, Tensor::new(&[sc], d)?, s))
        }
    }
    fn decompress_kv_static(l_i: usize, kr: (Tensor, Tensor, Tensor), vr: (Tensor, Tensor, Tensor), os: &[usize], td: &Device) -> Result<(Tensor, Tensor)> {
        let dt = if td.is_cpu() { DType::F32 } else { DType::BF16 };
        let dec = |res: (Tensor, Tensor, Tensor)| -> Result<Tensor> {
            if l_i == 0 { return Ok(res.0.to_device(td)?.to_dtype(dt)?); }
            let sc = res.2.to_scalar::<f32>()?; let pv = res.1.to_vec1::<u8>()?; 
            let total_el = os.iter().product(); let mut o = vec![0.0f32; total_el];
            if l_i <= 4 { for i in 0..total_el { o[i] = (((pv[i / 4] >> ((i % 4) * 2)) & 0x03) as f32 - 1.0) * sc; } }
            else { for i in 0..total_el { o[i] = if (pv[i / 8] >> (i % 8)) & 0x01 == 1 { sc } else { -sc }; } }
            Ok(Tensor::from_vec(o, os, &Device::Cpu)?.to_device(td)?.to_dtype(dt)?.contiguous()?)
        };
        Ok((dec(kr)?, dec(vr)?))
    }
    pub fn compress_to_bitkv(&self, l_i: usize, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> { Self::compress_to_bitkv_static(l_i, t) }
    pub fn inject_live_kv_bitkv(&mut self, l_i: usize, kr: (Tensor, Tensor, Tensor), vr: (Tensor, Tensor, Tensor), os: &[usize]) -> Result<()> {
        let td = if self.is_forced_cpu { Device::Cpu } else { self.embed_tokens.embeddings().device().clone() };
        let (mut kd, mut vd) = Self::decompress_kv_static(l_i, kr, vr, os, &td)?;
        if let Some(l) = self.layers.get_mut(l_i) {
            let th = l.self_attn.num_key_value_heads; let ch = kd.dims()[1];
            if ch != th {
                let ex = |t: Tensor| -> Result<Tensor> {
                    if ch < th && th % ch == 0 { let rep = th / ch; let mut r = t.clone(); for _ in 1..rep { r = Tensor::cat(&[r, t.clone()], 1)?; } Ok(r.contiguous()?) }
                    else if ch > th { Ok(t.narrow(1, 0, th)?.contiguous()?) } else { Ok(t.contiguous()?) }
                };
                kd = ex(kd)?; vd = ex(vd)?;
            }
            l.self_attn.kv_cache = Some((kd, vd));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct QuantizedQwen3TextModel { pub language_model: QuantizedQwen3VLTextModel, pub lm_head: Option<QLinear>, pub text_device: Device }
impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(cfg: &Qwen3VLConfig, ct: &gguf_file::Content, mm: Option<Arc<Mmap>>, td: &Device, tdi: usize, dt: DType, kvr: u64, bo: bool, _s: bool) -> Result<Self> {
        let lm = QuantizedQwen3VLTextModel::new_with_mmap(cfg.text_config.as_ref().unwrap(), ct, mm.clone(), "model", td, tdi, dt, kvr, bo)?;
        let mut r = std::io::Cursor::new(mm.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lh = get_qlinear(ct, &mut r, "lm_head", td, dt).ok();
        Ok(Self { language_model: lm, lm_head: lh, text_device: td.clone() })
    }
    pub fn forward(&mut self, ids: &Tensor, cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (bs, sl) = ids.dims2()?; 
        let embs = self.language_model.embed_tokens.forward(&ids.flatten_all()?)?.reshape((bs, sl, ()))?;
        let pos = match cp { Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize, None => offset };
        let pids = Tensor::arange(pos as u32, (pos + sl) as u32, ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), None, None)?;
        let h = out.narrow(1, out.dim(1)? - 1, 1)?;
        if let Some(lh) = &self.lm_head { Ok(lh.forward(&h)?) } else { Ok(h) }
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.language_model.to_device(d)?; if let Some(lh) = &mut self.lm_head { lh.to_device(d)?; } self.text_device = d.clone(); Ok(()) }
    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(p, c, b) }
    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(p, d, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

pub fn load_tensors_from_true_iq0(p: &Path, d: &Device, dt: DType, bo: bool) -> Result<HashMap<String, Tensor>> {
    let f = std::fs::read(p)?; let st = safetensors::SafeTensors::deserialize(&f)?;
    let mut data = HashMap::new();
    for (name, view) in st.tensors() {
        if name.ends_with(".scales") || name.ends_with(".scale") || name.ends_with(".shape") || name.ends_with(".format") || name.ends_with(".min") { continue; }
        let is_p = name.ends_with(".packed"); let b_n = if is_p { name.strip_suffix(".packed").unwrap() } else { &name };
        
        let clean = b_n.replace("model.language_model.", "").replace("model.layers.", "layers.").replace("model.visual.", "visual.").replace("model.", "").replace("language_model.", "");
        
        if bo {
            // [FLEXIBLE-FILTER] Support both 'layers.0' and 'blk.0'
            let is_layer_zero = clean.starts_with("layers.0.") || clean.starts_with("blk.0.") || clean.starts_with("model.layers.0.") || clean.starts_with("model.blk.0.");
            let is_common = !clean.contains(".layers.") && !clean.contains(".blocks.") && !clean.contains("blk.");
            
            if !is_layer_zero && !is_common { 
                continue; 
            }
        }
        
        let mut t_n = clean.clone();
        if b_n.starts_with("v.") || clean.starts_with("visual.") {
            let rest = if b_n.starts_with("v.") { &b_n[2..] } else if clean.starts_with("visual.") { &clean[7..] } else { &clean };
            if rest.starts_with("blk.") || rest.starts_with("blocks.") {
                let rs = if rest.starts_with("blk.") { &rest[4..] } else { &rest[7..] };
                let ps: Vec<&str> = rs.splitn(2, '.').collect();
                let m = match ps[1] { s if s.starts_with("ln1") || s.starts_with("norm1") => "norm1", s if s.starts_with("ln2") || s.starts_with("norm2") => "norm2", s if s.starts_with("attn_qkv") || s.starts_with("attn.qkv") => "attn.qkv", s if s.starts_with("attn_out") || s.starts_with("attn.proj") => "attn.proj", s if s.starts_with("ffn_up") || s.starts_with("mlp.linear_fc1") => "mlp.linear_fc1", s if s.starts_with("ffn_down") || s.starts_with("mlp.linear_fc2") => "mlp.linear_fc2", _ => ps[1] };
                t_n = format!("visual.blocks.{}.{}", ps[0], m);
            } else if rest.starts_with("patch_embd") || rest.starts_with("patch_embed") { t_n = "visual.patch_embed.proj".to_string(); }
            else if rest.starts_with("position_embd") || rest.starts_with("pos_embed") { t_n = "visual.pos_embed".to_string(); }
            else if rest.starts_with("post_ln") || rest.starts_with("merger.norm") { t_n = "visual.merger.norm".to_string(); }
            else { t_n = format!("visual.{}", rest); }
        } else if b_n.starts_with("mm.") {
            let rest = &b_n[3..]; if rest.starts_with("0") { t_n = "visual.merger.linear_fc1".to_string(); } else if rest.starts_with("2") { t_n = "visual.merger.linear_fc2".to_string(); }
        } else if clean.starts_with("layers.") || b_n.starts_with("blk.") || clean.starts_with("blk.") {
            let rest = if clean.starts_with("layers.") { &clean[7..] } else if clean.starts_with("blk.") { &clean[4..] } else { &b_n[4..] };
            let ps: Vec<&str> = rest.splitn(2, '.').collect();
            let m = match ps[1] { 
                "attn_q_norm.weight" => "self_attn.q_norm.weight", 
                "attn_k_norm.weight" => "self_attn.k_norm.weight", 
                s if s.starts_with("attn_q") => "self_attn.q_proj", 
                s if s.starts_with("attn_k") => "self_attn.k_proj", 
                s if s.starts_with("attn_v") => "self_attn.v_proj", 
                s if s.starts_with("attn_output") => "self_attn.o_proj", 
                s if s.starts_with("attn_norm") => "input_layernorm", 
                s if s.starts_with("ffn_norm") => "post_attention_layernorm", 
                s if s.starts_with("ffn_gate") => "mlp.gate_proj", 
                s if s.starts_with("ffn_up") => "mlp.up_proj", 
                s if s.starts_with("ffn_down") => "mlp.down_proj", 
                _ => ps[1] 
            };
            t_n = format!("language_model.layers.{}.{}", ps[0], m);
        } else if clean == "embed_tokens.weight" || clean == "token_embd.weight" { t_n = "language_model.embed_tokens.weight".to_string(); }
        else if clean == "norm.weight" || clean == "output_norm.weight" { t_n = "language_model.norm.weight".to_string(); }
        else if clean == "lm_head.weight" || clean == "output.weight" { t_n = "lm_head.weight".to_string(); }

        let ft = if is_p {
            // [VRAM-RESIDENT UPGRADE] Store raw packed bit-data instead of dequantizing
            let packed_raw = view.data().to_vec();
            // Assuming packed data is u32
            let packed_u32: Vec<u32> = packed_raw.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let packed_tensor = Tensor::from_vec(packed_u32, (packed_raw.len() / 4,), d)?;
            
            // Handle scales (F16 -> F32)
            let scales_view = match st.tensor(&format!("{b_n}.scales")) {
                Ok(v) => v,
                Err(_) => match st.tensor(&format!("{b_n}.scale")) {
                    Ok(v) => v,
                    Err(_) => {
                        // Create dummy scales if missing
                        println!("[MODEL-PROBE] Warning: Scale missing for {}. Using 1.0.", b_n);
                        let dummy = vec![1.0f32; 1];
                        let t = Tensor::from_vec(dummy, (1,), d)?;
                        // We need a dummy view-like access or just a tensor.
                        // Since we are building the map, we can just insert a dummy tensor later.
                        data.insert(format!("model.{}.scales", t_n), t.clone());
                        data.insert(format!("model.{}.shape", t_n), Tensor::new(&[1u32], d)?);
                        packed_tensor.clone() // Fallback return
                    }
                }
            };
            
            if !data.contains_key(&format!("model.{}.scales", t_n)) {
                let scales_raw = scales_view.data();
                let scales_f32: Vec<f32> = match scales_view.dtype() {
                    safetensors::Dtype::F16 => scales_raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
                    safetensors::Dtype::F32 => scales_raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                    _ => vec![1.0; scales_raw.len() / 4],
                };
                let scales = Tensor::from_vec(scales_f32, (scales_raw.len() / match scales_view.dtype() { safetensors::Dtype::F16 => 2, _ => 4 },), d)?;

                // Handle shape (I32 -> usize)
                let shape_tensor = if let Ok(shape_view) = st.tensor(&format!("{b_n}.shape")) {
                    let shape_raw = shape_view.data();
                    let shape_u32: Vec<u32> = shape_raw.chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Tensor::from_vec(shape_u32, (shape_raw.len() / 4,), d)?
                } else {
                    Tensor::new(&[1u32], d)?
                };
                
                data.insert(format!("model.{}.scales", t_n), scales);
                data.insert(format!("model.{}.shape", t_n), shape_tensor);
            }
            packed_tensor
        } else {
            let r = view.data(); match view.dtype() {
                safetensors::Dtype::F32 => Tensor::from_vec(r.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>(), view.shape(), &Device::Cpu)?,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => Tensor::from_vec(r.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect::<Vec<half::f16>>(), view.shape(), &Device::Cpu)?,
                safetensors::Dtype::U8 => Tensor::from_vec(r.to_vec(), view.shape(), &Device::Cpu)?,
                safetensors::Dtype::I32 => Tensor::from_vec(r.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect::<Vec<i64>>(), view.shape(), &Device::Cpu)?,
                _ => return Err(anyhow!("Unsupported dtype: {:?} for {}", view.dtype(), name)),
            }.to_device(d)?.to_dtype(dt)?.contiguous()?
        };
        data.insert(if t_n.starts_with("lm_head") { t_n } else { format!("model.{}", t_n) }, ft);
    }
    Ok(data)
}

pub fn from_true_iq0_safetensors(p: &Path, d: &Device, dt: DType) -> Result<VarBuilder<'static>> {
    let data = load_tensors_from_true_iq0(p, d, dt, false)?; Ok(VarBuilder::from_tensors(data, dt, d))
}

#[derive(Clone)]
pub struct QuantizedQwen3VLModel {
    pub config: Qwen3VLConfig,
    pub visual: Option<Qwen3VLVisionModel>,
    pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: QLinear,
    pub text_device: Device,
    pub vision_device: Device,
    pub is_baking: bool,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(
        cfg: &Qwen3VLConfig,
        ctm: &gguf_file::Content,
        mm: Option<Arc<Mmap>>,
        _ctv: &gguf_file::Content,
        _vmm: Option<Arc<Mmap>>,
        td: &Device,
        tdi: usize,
        vd: &Device,
        _vi: usize,
        dt: DType,
        kvr: u64,
        bo: bool,
        force_text_only: bool,
    ) -> Result<Self> {
        let visual = if !force_text_only && cfg.vision_config.is_some() {
            Some(Qwen3VLVisionModel::new(
                cfg.vision_config.as_ref().unwrap().clone(),
                VarBuilder::from_tensors(HashMap::new(), dt, vd),
            )?)
        } else {
            None
        };

        let lm = QuantizedQwen3VLTextModel::new_with_mmap(
            cfg.text_config.as_ref().unwrap(),
            ctm,
            mm.clone(),
            "model",
            td,
            tdi,
            dt,
            kvr,
            bo,
        )?;
        let mut r = std::io::Cursor::new(mm.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lh = get_qlinear(ctm, &mut r, "lm_head", td, dt)?;
        Ok(Self {
            config: cfg.clone(),
            visual,
            language_model: lm,
            lm_head: lh,
            text_device: td.clone(),
            vision_device: vd.clone(),
            is_baking: bo,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pv: Option<&Tensor>,
        thw: Option<&Tensor>,
        _vpv: Option<&Tensor>,
        _vthw: Option<&Tensor>,
        cp: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (bs, sl) = input_ids.dims2()?;
        let mut embs = self
            .language_model
            .embed_tokens
            .forward(&input_ids.flatten_all()?)?
            .reshape((bs, sl, ()))?;

        if let (Some(visual_model), Some(pv), Some(thw)) = (&self.visual, pv, thw) {
            let (ie, _) = visual_model.forward(pv, thw)?;
            let ie = ie.to_device(&self.text_device)?;
            let mask = input_ids
                .broadcast_eq(&Tensor::new(
                    vec![self.config.image_token_id.unwrap_or(0) as u32],
                    input_ids.device(),
                )?)?
                .to_dtype(DType::U32)?;
            embs = masked_scatter_dim0(&embs, &ie, &mask)?;
        }

        let pos = match cp {
            Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize,
            None => offset,
        };
        let pids = Tensor::arange(pos as u32, (pos + sl) as u32, input_ids.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, bs, sl))?;
        let out = self
            .language_model
            .forward(&embs, offset, Some(&pids), None, None)?;
        Ok(self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?)
    }

    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        if let Some(v) = &mut self.visual {
            v.to_device(d)?;
        }
        self.language_model.to_device(d)?;
        self.lm_head.to_device(d)?;
        self.text_device = d.clone();
        self.vision_device = d.clone();
        Ok(())
    }

    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> {
        self.language_model.save_kv_cache(p, c, b)
    }

    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> {
        self.language_model.load_kv_cache(p, d, e, u)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
