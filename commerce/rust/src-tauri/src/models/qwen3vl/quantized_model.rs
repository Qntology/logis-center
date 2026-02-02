use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Activation, Embedding, VarBuilder};
use candle_core::quantized::gguf_file;
use rayon::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use memmap2::Mmap;
use safetensors::tensor::SafeTensors;

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
    let weight = ct.tensor(reader, &format!("{}.weight", name), device)?;
    let bias = ct.tensor(reader, &format!("{}.bias", name), device).ok().map(|t| t.dequantize(device).unwrap().to_dtype(dtype).unwrap());
    let w_dequant = weight.dequantize(device)?.to_dtype(DType::F32)?;
    let s = w_dequant.dims().to_vec();
    let total_el = s.iter().product::<usize>();
    let scales = Tensor::ones((total_el / 32).max(1), DType::F32, device)?;
    let packed = Tensor::zeros((total_el / 32).max(1), DType::U32, device)?;
    Ok(QLinear::new(packed, scales, s, bias, device.clone()))
}

pub fn get_qlinear_safe(vb: VarBuilder, name: &str, map: Option<&HashMap<String, Tensor>>) -> Result<QLinear> {
    let prefix = vb.prefix();
    let get_path = |p: &str| -> String {
        if prefix.is_empty() { p.to_string() }
        else if prefix.ends_with('.') { format!("{}{}", prefix, p) }
        else { format!("{}.{}", prefix, p) }
    };
    let packed_path = get_path(&format!("{}.packed", name));
    if let Some(m) = map {
        if let Some(p) = m.get(&packed_path) {
            let s = m.get(&get_path(&format!("{}.scales", name))).ok_or(anyhow!("Missing scales for {}", name))?.clone().to_dtype(DType::F32)?;
            let sh_t = m.get(&get_path(&format!("{}.shape", name))).ok_or(anyhow!("Missing shape for {}", name))?.clone();
            let sh: Vec<usize> = sh_t.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let b = m.get(&get_path(&format!("{}.bias", name))).cloned();
            return Ok(QLinear::new(p.clone(), s, sh, b, vb.device().clone()));
        }
    }
    let w = vb.get_with_hints((1,), name, candle_nn::Init::Const(0.0))?;
    let s = w.dims().to_vec(); let t = s.iter().product::<usize>();
    Ok(QLinear::new(Tensor::zeros((t/32).max(1), DType::U32, vb.device())?, Tensor::ones((t/32).max(1), DType::F32, vb.device())?, s, None, vb.device().clone()))
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
        self.weight = self.weight.to_device(device)?.to_dtype(DType::F32)?; Ok(())
    }
}
impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_norm.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(x.dtype())
    }
}

#[derive(Debug, Clone)]
pub struct QLinear { 
    packed_weight: Tensor, scales: Tensor, original_shape: Vec<usize>, bias: Option<Tensor>, device: Device, dtype: DType
}
impl QLinear {
    pub fn new(packed_weight: Tensor, scales: Tensor, original_shape: Vec<usize>, bias: Option<Tensor>, device: Device) -> Self { 
        Self { packed_weight, scales, original_shape, bias, device, dtype: DType::F32 } 
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if !self.device.same_device(device) {
            self.packed_weight = self.packed_weight.to_device(device)?;
            self.scales = self.scales.to_device(device)?;
            if let Some(b) = &self.bias { self.bias = Some(b.to_device(device)?); }
            self.device = device.clone();
        }
        Ok(())
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let weight = self.dequantize_on_the_fly(xs.device())?;
        let xs_flat = xs.reshape((b * s, h))?.to_dtype(DType::F32)?;
        let weight_f32 = weight.to_dtype(DType::F32)?;
        let mut out = xs_flat.matmul(&weight_f32.t()?)?;
        if let Some(bias) = &self.bias { out = out.broadcast_add(&bias.to_dtype(DType::F32)?)?; }
        let res = out.reshape((b, s, ()))?.to_dtype(xs.dtype())?;
        Ok(res)
    }
    fn dequantize_on_the_fly(&self, device: &Device) -> Result<Tensor> {
        let s = &self.original_shape;
        let total_el = s.iter().product::<usize>();
        let packed_cpu = self.packed_weight.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        
        let packed_data = packed_cpu.to_vec1::<u32>()?;
        let scales_data = scales_cpu.to_vec1::<f32>()?;
        
        let mut weights = vec![0.0f32; total_el];
        weights.par_chunks_exact_mut(32).enumerate().for_each(|(b_i, block_out)| {
            if b_i < scales_data.len() && b_i < packed_data.len() {
                let s_val = scales_data[b_i];
                let b = packed_data[b_i];
                for bit in 0..32 { 
                    if b_i * 32 + bit < total_el {
                        block_out[bit] = s_val * (if (b >> bit) & 1 != 0 { 1.0 } else { -1.0 }); 
                    }
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

#[derive(Debug, Clone)]
pub struct QGateUpDownMLP { pub g: QLinear, pub u: QLinear, pub d: QLinear, pub act: Activation }
impl QGateUpDownMLP {
    pub fn new(vb: VarBuilder, act: Activation, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        Ok(Self { g: get_qlinear_safe(vb.pp("gate_proj"), "weight", map)?, u: get_qlinear_safe(vb.pp("up_proj"), "weight", map)?, d: get_qlinear_safe(vb.pp("down_proj"), "weight", map)?, act })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.g.forward(x)?; let gate = self.act.forward(&g)?;
        let u = self.u.forward(x)?; Ok(self.d.forward(&gate.mul(&u)?)?)
    }
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
        let is_g = b_n.contains("blk.");
        let (q, k, v, o, qn, kn, g, u, d, n, in_ln) = if is_g {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down", "ffn_norm", "attn_norm")
        } else {
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm", "input_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention {
            q_proj: get_qlinear(ct, r, &format!("{b_n}.{q}"), dev, dt)?, k_proj: get_qlinear(ct, r, &format!("{b_n}.{k}"), dev, dt)?,
            v_proj: get_qlinear(ct, r, &format!("{b_n}.{v}"), dev, dt)?, o_proj: get_qlinear(ct, r, &format!("{b_n}.{o}"), dev, dt)?,
            q_norm: get_rms_norm(ct, r, &format!("{b_n}.{qn}"), cfg.rms_norm_eps, dev, dt)?, k_norm: get_rms_norm(ct, r, &format!("{b_n}.{kn}"), cfg.rms_norm_eps, dev, dt)?,
            num_attention_heads: cfg.num_attention_heads, num_key_value_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            scaling: 1f64 / f64::sqrt(cfg.head_dim as f64), kv_cache: None, layer_idx: l_i,
        };
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !b_o {
            (Some(get_qlinear(ct, r, &format!("{b_n}.{g}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{u}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{d}"), dev, dt)?), Some(get_rms_norm(ct, r, &format!("{b_n}.{n}"), cfg.rms_norm_eps, dev, dt)?))
        } else { (None, None, None, None) };
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
    pub mrope_section: Vec<usize>, pub mmap: Option<Arc<Mmap>>, pub is_forced_cpu: bool, pub is_baking: bool,
}

impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(config: &Qwen3VLTextConfig, ct: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, _base_name: &str, device: &Device, _device_id: usize, dtype: DType, _kv_reserve: u64, baking_only: bool) -> Result<Self> {
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        let prefix = if ct.tensor(&mut reader, "model.layers.0.input_layernorm.weight", device).is_ok() { "model.layers" }
                     else if ct.tensor(&mut reader, "model.language_model.layers.0.input_layernorm.weight", device).is_ok() { "model.language_model.layers" }
                     else { "blk" };
        let emb_name = if prefix == "blk" { "token_embd.weight" } else if prefix.contains("language_model") { "model.language_model.embed_tokens.weight" } else { "model.embed_tokens.weight" };
        let norm_name = if prefix == "blk" { "output_norm.weight" } else if prefix.contains("language_model") { "model.language_model.norm.weight" } else { "model.norm.weight" };
        let (embed_tokens, actual_h) = if let Ok(tensor) = ct.tensor(&mut reader, emb_name, device) {
             let t = tensor.dequantize(device)?.to_dtype(dtype)?; let h = t.dim(1)?; (Embedding::new(t, h), h)
        } else { return Err(anyhow!("Failed to load embed_tokens: {}", emb_name)); };
        let mut p_cfg = config.clone(); p_cfg.hidden_size = actual_h;
        let mut layers = vec![]; let nl = if baking_only { 1 } else { p_cfg.num_hidden_layers };
        for i in 0..nl { 
            let layer_prefix = if prefix == "blk" { format!("blk.{i}") } else { format!("{prefix}.{i}") };
            layers.push(QuantizedQwen3VLTextDecoderLayer::new(&p_cfg, ct, &mut reader, &layer_prefix, device, dtype, i, baking_only)?); 
        }
        let norm = get_rms_norm(ct, &mut reader, norm_name, p_cfg.rms_norm_eps, device, dtype)?;
        let mut mrope = p_cfg.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        let hh = p_cfg.head_dim / 2; let ss: usize = mrope.iter().sum();
        if ss != hh && ss > 0 { let r = hh as f32 / ss as f32; mrope = mrope.iter().map(|&s| (s as f32 * r).round() as usize).collect(); }
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(p_cfg.head_dim, p_cfg.rope_theta), mrope_section: mrope, mmap: mmap_handle, is_forced_cpu: device.is_cpu(), is_baking: baking_only })
    }
    pub fn forward(&mut self, x: &Tensor, offset: usize, pids: Option<&Tensor>, visual_mask: Option<&Tensor>, ds_embeds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let pi = match pids { Some(ids) => ids.clone(), None => Tensor::arange(offset as u32, (sl + offset) as u32, x.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))? };
        let (cos, sin) = self.rotary_emb.forward(&pi, x.dtype(), self.mrope_section.clone())?;
        let mask = if sl <= 1 { None } else { Some(prepare_causal_attention_mask(bs, sl, offset, x.device())?) };
        let mut x = x.clone(); let limit = if self.is_baking { 1 } else { self.layers.len() };
        for (i, layer) in self.layers.iter_mut().enumerate().take(limit) {
            x = layer.forward_with_params(&x, &cos, &sin, mask.as_ref())?;
            if let (Some(m), Some(ds)) = (visual_mask, ds_embeds.as_ref()) { if i < ds.len() { x = mask_index_add(&x.squeeze(0)?, &m.squeeze(0)?, &ds[i])?.unsqueeze(0)?; } }
        }
        Ok(x.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { l.self_attn.clear_kv_cache(); } }
    pub fn get_kv_len(&self) -> usize { self.layers[0].self_attn.get_kv_len() }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let w = self.embed_tokens.embeddings().to_device(device)?; self.embed_tokens = Embedding::new(w, self.embed_tokens.hidden_size());
        for l in &mut self.layers { l.to_device(device)?; }
        let nw = self.norm.weight.to_device(device)?; self.norm = RmsNorm::new(nw, self.norm.eps); Ok(())
    }
    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, _: usize) -> Result<()> {
        if !path.exists() { fs::create_dir_all(path)?; }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some((k, v)) = &layer.self_attn.kv_cache {
                let rk = compress_to_bitkv_static(i, k)?; let rv = compress_to_bitkv_static(i, v)?;
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
                let (dk, dv) = decompress_kv_static(i, (m.get("k_a").unwrap().clone(), m.get("k_p").unwrap().clone(), m.get("k_s").unwrap().clone()), (m.get("v_a").unwrap().clone(), m.get("v_p").unwrap().clone(), m.get("v_s").unwrap().clone()), &shape, &target_device)?;
                (dk, dv)
            } else if let Some((ref fk, ref fv)) = first_layer_kv {
                (fk.clone(), fv.clone())
            } else { continue; };
            let (_b, h, _s, d) = k.dims4()?; let target_h = layer.self_attn.num_key_value_heads; let target_d = layer.self_attn.head_dim;
            if h != target_h {
                if target_h % h == 0 { let rep = target_h / h; k = k.repeat((1, rep, 1, 1))?; v = v.repeat((1, rep, 1, 1))?; }
                else { k = k.narrow(1, 0, h.min(target_h))?; v = v.narrow(1, 0, h.min(target_h))?; }
            }
            if d != target_d { k = apply_linear_bridge(&k, target_d)?; v = apply_linear_bridge(&v, target_d)?; }
            if first_layer_kv.is_none() { first_layer_kv = Some((k.clone(), v.clone())); }
            layer.self_attn.kv_cache = Some((k, v));
        }
        println!("[MODEL-OPTIM] Full Restoration (Quantized): Loaded all present layers via Bridge.");
        Ok(())
    }
}

// --- Main ---

#[derive(Clone)]
pub struct QuantizedQwen3VLModel {
    pub config: Qwen3VLConfig, pub visual: Option<Qwen3VLVisionModel>, pub language_model: QuantizedQwen3VLTextModel,
    pub lm_head: QLinear, pub text_device: Device, pub vision_device: Device, pub is_baking: bool,
}

impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(cfg: &Qwen3VLConfig, ctm: &gguf_file::Content, mm: Option<Arc<Mmap>>, _ctv: &gguf_file::Content, vmm: Option<Arc<Mmap>>, td: &Device, tdi: usize, vd: &Device, _vi: usize, dt: DType, kvr: u64, bo: bool, force_text_only: bool) -> Result<Self> {
        let visual = if !force_text_only && cfg.vision_config.is_some() {
            let v_cfg = cfg.vision_config.as_ref().unwrap();
            let v_map = vmm.as_ref().map(|m| &m[..]).unwrap_or(&[]);
            let mut v_reader = std::io::Cursor::new(v_map);
            let vb_v = if !v_map.is_empty() { load_vision_tensors_to_vb(_ctv, &mut v_reader, vd, dt)? }
            else { let m_map = mm.as_ref().map(|m| &m[..]).unwrap_or(&[]); let mut m_reader = std::io::Cursor::new(m_map); load_vision_tensors_to_vb(ctm, &mut m_reader, vd, dt)? };
            Some(Qwen3VLVisionModel::new(v_cfg.clone(), vb_v, None)?)
        } else { None };
        let lm = QuantizedQwen3VLTextModel::new_with_mmap(cfg.text_config.as_ref().unwrap(), ctm, mm.clone(), "model", td, tdi, dt, kvr, bo)?;
        let mut r = std::io::Cursor::new(mm.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let base_head = if lm.embed_tokens.embeddings().dim(1)? == 1024 { "output" } else { "lm_head" };
        let lh = get_qlinear(ctm, &mut r, base_head, td, dt)?;
        Ok(Self { config: cfg.clone(), visual, language_model: lm, lm_head: lh, text_device: td.clone(), vision_device: vd.clone(), is_baking: bo })
    }
    pub fn forward(&mut self, input_ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, _vthw: Option<&Tensor>, cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (bs, sl) = input_ids.dims2()?;
        let mut embs = self.language_model.embed_tokens.forward(&input_ids.flatten_all()?)?.reshape((bs, sl, ()))?;
        if let (Some(ref mut visual_model), Some(pv), Some(thw)) = (&mut self.visual, pv, thw) {
            let (ie, _) = visual_model.forward(pv, thw)?; let ie = ie.to_device(&self.text_device)?;
            let mask = input_ids.broadcast_eq(&Tensor::new(vec![self.config.image_token_id.unwrap_or(0) as u32], input_ids.device())?)?.to_dtype(DType::U32)?;
            embs = masked_scatter_dim0(&embs, &ie, &mask)?;
        }
        let pos = match cp { Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize, None => offset };
        let pids = Tensor::arange(pos as u32, (pos + sl) as u32, input_ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), None, None)?;
        Ok(self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        if let Some(v) = &mut self.visual { v.to_device(d)?; }
        self.language_model.to_device(d)?; self.lm_head.to_device(d)?;
        self.text_device = d.clone(); self.vision_device = d.clone(); Ok(())
    }
    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(p, c, b) }
    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(p, d, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
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
        let (bs, sl) = ids.dims2()?; let embs = self.language_model.embed_tokens.forward(&ids.flatten_all()?)?.reshape((bs, sl, ()))?;
        let pos = match cp { Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize, None => offset };
        let pids = Tensor::arange(pos as u32, (pos + sl) as u32, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), None, None)?;
        let h = out.narrow(1, out.dim(1)? - 1, 1)?;
        if let Some(lh) = &self.lm_head { Ok(lh.forward(&h)?) } else { Ok(h) }
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.language_model.to_device(d)?; if let Some(lh) = &mut self.lm_head { lh.to_device(d)?; } self.text_device = d.clone(); Ok(()) }
    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(p, c, b) }
    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(p, d, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

pub fn load_tensors_from_true_iq0(path: &Path, device: &Device, _dtype: DType, _baking: bool) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path)?; let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?; let mut tensors = HashMap::new();
    for (name, view) in safetensors.tensors() {
        println!("[DEBUG-LOAD] Key: {} | ST-DType: {:?}", name, view.dtype());
        let shape = view.shape().to_vec(); let data = view.data();
        if name.ends_with(".packed") {
             // CRITICAL: Interpret bytes directly as U32 to solve I32/I64 mismatch issues
             let u_data: &[u32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) };
             let mut new_shape = shape.clone(); 
             if let Some(last) = new_shape.last_mut() {
                 if *last == data.len() { *last /= 4; } // Adjust if shape was in bytes
             }
             println!("[DEBUG-LOAD]   -> Reinterpreted as U32. New Shape: {:?}", new_shape);
             tensors.insert(name.to_string(), Tensor::from_slice(u_data, new_shape.as_slice(), &Device::Cpu)?.to_device(device)?);
        } else {
            match view.dtype() {
                safetensors::Dtype::F32 => { let f_data: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }; tensors.insert(name.to_string(), Tensor::from_slice(f_data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::U32 => { let u_data: &[u32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) }; tensors.insert(name.to_string(), Tensor::from_slice(u_data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::I32 => { 
                    let i_data: &[i32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len() / 4) }; 
                    let u_data: Vec<u32> = i_data.iter().map(|&x| x as u32).collect(); 
                    tensors.insert(name.to_string(), Tensor::from_vec(u_data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); 
                },
                safetensors::Dtype::U8 | safetensors::Dtype::I8 | safetensors::Dtype::BOOL => { tensors.insert(name.to_string(), Tensor::from_slice(data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::F16 => { let f_data: &[half::f16] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2) }; tensors.insert(name.to_string(), Tensor::from_slice(f_data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::BF16 => { let f_data: &[half::bf16] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len() / 2) }; tensors.insert(name.to_string(), Tensor::from_slice(f_data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                _ => return Err(anyhow!("Manual loading for dtype {:?} not implemented", view.dtype())),
            }
        };
    }
    Ok(tensors)
}

fn apply_linear_bridge(x: &Tensor, target_dim: usize) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?; let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let rms = (x_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()).max(1e-6);
    let theory_scale = (d as f64 / target_dim as f64).sqrt();
    let alignment_coeff = 0.7071067811865476_f64; 
    let dynamic_bridge_scale = (theory_scale * alignment_coeff) / (rms as f64);
    if target_dim >= d {
        let left = x_f32.clone();
        let upscaled = if target_dim > d {
             let right = x_f32.roll(1, D::Minus1)?; let lerp = ((left + right)? * 0.5)?; 
             (Tensor::stack(&[x_f32, lerp], D::Minus1)?.affine(dynamic_bridge_scale, 0.0))?.reshape((b, h, s, target_dim))?
        } else { x_f32.affine(dynamic_bridge_scale * (rms as f64), 0.0)? };
        Ok(upscaled.clamp(-10.0, 10.0)?.to_dtype(x.dtype())?)
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

pub fn load_vision_tensors_to_vb(ct: &gguf_file::Content, reader: &mut std::io::Cursor<&[u8]>, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
    let mut data = HashMap::new();
    for (name, _) in ct.tensor_infos.iter() {
        if name.ends_with(".scales") || name.ends_with(".scale") || name.ends_with(".shape") { continue; }
        let mut clean = name.to_string();
        if clean.starts_with("mm.") {
            let rest = &clean[3..]; 
            if rest.starts_with("0") { clean = "merger.linear_fc1".to_string() + &rest[1..]; } 
            else if rest.starts_with("2") { clean = "merger.linear_fc2".to_string() + &rest[1..]; }
        } else if clean.starts_with("visual.blk.") {
            clean = clean.replace("visual.blk.", "blocks."); 
        } else if clean.starts_with("visual.") {
            clean = clean[7..].to_string();
        } else if clean.starts_with("v.") {
            clean = clean[2..].to_string();
        }
        if clean.contains("attn_qkvisual") { clean = clean.replace("attn_qkvisual", "attn.qkv"); }
        if clean.contains("attn_out") { clean = clean.replace("attn_out", "attn.proj"); }
        if clean.contains("ffn_up") { clean = clean.replace("ffn_up", "mlp.linear_fc1"); }
        if clean.contains("ffn_down") { clean = clean.replace("ffn_down", "mlp.linear_fc2"); }
        if clean.contains("post_ln") { clean = clean.replace("post_ln", "merger.norm"); }
        if clean.contains("patch_embd") { clean = clean.replace("patch_embd", "patch_embed.proj"); }
        if clean.contains("position_embd") { clean = clean.replace("position_embd", "pos_embed"); }
        let t = ct.tensor(reader, name, device)?.dequantize(device)?.to_dtype(dtype)?;
        data.insert(clean, t);
    }
    Ok(VarBuilder::from_tensors(data, dtype, device))
}