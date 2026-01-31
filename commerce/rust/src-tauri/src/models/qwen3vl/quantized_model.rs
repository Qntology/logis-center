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
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), mask, self.scaling)?;
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

impl QuantizedQwen3VLTextDecoderLayer {
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

impl Module for QuantizedQwen3VLTextDecoderLayer {
    fn forward(&self, _xs: &Tensor) -> candle_core::Result<Tensor> {
        // Module trait implementation for basic forward (not used by our LLM loop which needs more params)
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
        let embed_tokens = Embedding::new(ct.tensor(&mut reader, "token_embd.weight", device)?.dequantize(device)?.to_dtype(dtype)?, config.hidden_size);
        let mut layers = vec![];
        let num_layers = if baking_only { 1 } else { config.num_hidden_layers };
        for i in 0..num_layers {
            layers.push(QuantizedQwen3VLTextDecoderLayer::new(config, ct, &mut reader, &format!("blk.{i}"), device, dtype, i, baking_only)?);
        }
        let norm = get_rms_norm(ct, &mut reader, "output_norm", config.rms_norm_eps, device, dtype)?;
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), mmap: mmap_handle, is_forced_cpu: device.is_cpu() })
    }
    pub fn forward(&mut self, embeds: &Tensor, seqlen_offset: usize, pos_ids: Option<&Tensor>, visual_mask: Option<&Tensor>, ds_embeds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (b, s, _) = embeds.dims3()?;
        let pos_ids = match pos_ids { Some(ids) => ids.clone(), None => Tensor::arange(seqlen_offset as u32, (s + seqlen_offset) as u32, embeds.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b, s))? };
        let (cos, sin) = self.rotary_emb.forward(&pos_ids, embeds.dtype(), self.mrope_section.clone())?;
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
                let mut map = HashMap::new();
                map.insert("k".to_string(), k.clone()); map.insert("v".to_string(), v.clone());
                candle_core::safetensors::save(&map, path.join(format!("layer_{}_kv.safetensors", i)))?;
            }
            if clear { layer.self_attn.clear_kv_cache(); }
        }
        Ok(())
    }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, _: usize, _: usize) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = path.join(format!("layer_{}_kv.safetensors", i));
            if p.exists() {
                let map = candle_core::safetensors::load(p, device)?;
                if let (Some(k), Some(v)) = (map.get("k"), map.get("v")) {
                    layer.self_attn.kv_cache = Some((k.clone(), v.clone()));
                }
            }
        }
        Ok(())
    }
    pub fn compress_to_bitkv(&self, _: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> { Err(anyhow!("Not implemented")) }
    pub fn inject_live_kv_bitkv(&mut self, _: &[Tensor], _: &[Tensor], _: &[Tensor], _: &[Tensor], _: &[Tensor], _: &[Tensor], _: &[usize]) -> Result<()> { Ok(()) }
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
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.visual.to_device(device)?; self.language_model.to_device(device)?; self.lm_head.to_device(device)?; Ok(())
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
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.language_model.to_device(device)?; if let Some(h) = &mut self.lm_head { h.to_device(device)?; } Ok(())
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

pub fn from_true_iq0_safetensors(path: &Path, device: &Device, dtype: DType, hidden_size: usize) -> Result<VarBuilder<'static>> {
    let file = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&file)?;
    let mut data = HashMap::new();
    for (name, view) in st.tensors() {
        let mut target_name = name.clone();
        if name.starts_with("v.") {
            let rest = &name[2..];
            if rest.starts_with("blk.") {
                let parts: Vec<&str> = rest[4..].splitn(2, '.').collect();
                let mapped = match parts[1] {
                    s if s.starts_with("ln1") => s.replace("ln1", "norm1"), s if s.starts_with("ln2") => s.replace("ln2", "norm2"),
                    s if s.starts_with("attn_qkv") => s.replace("attn_qkv", "attn.qkv"), s if s.starts_with("attn_out") => s.replace("attn_out", "attn.proj"),
                    s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.linear_fc1"), s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"), _ => parts[1].to_string()
                };
                target_name = format!("model.visual.blocks.{}.{}", parts[0], mapped);
            } else if rest.starts_with("patch_embd") { target_name = rest.replace("patch_embd", "model.visual.patch_embed.proj"); }
            else if rest.starts_with("position_embd") { target_name = rest.replace("position_embd", "model.visual.pos_embed"); }
            else if rest.starts_with("post_ln") { target_name = rest.replace("post_ln", "model.visual.merger.norm"); }
            else { target_name = format!("model.visual.{}", rest); }
        } else if name.starts_with("mm.") {
            let rest = &name[3..];
            if rest.starts_with("0") { target_name = rest.replace("0", "model.visual.merger.linear_fc1"); }
            else if rest.starts_with("2") { target_name = rest.replace("2", "model.visual.merger.linear_fc2"); }
        } else if name.starts_with("blk.") {
            let parts: Vec<&str> = name[4..].splitn(2, '.').collect();
            let mapped = match parts[1] {
                s if s.starts_with("attn_q") => s.replace("attn_q", "self_attn.q_proj"), s if s.starts_with("attn_k") => s.replace("attn_k", "self_attn.k_proj"),
                s if s.starts_with("attn_v") => s.replace("attn_v", "self_attn.v_proj"), s if s.starts_with("attn_output") => s.replace("attn_output", "self_attn.o_proj"),
                s if s.starts_with("attn_norm") => s.replace("attn_norm", "input_layernorm"), s if s.starts_with("ffn_norm") => s.replace("ffn_norm", "post_attention_layernorm"),
                s if s.starts_with("ffn_gate") => s.replace("ffn_gate", "mlp.gate_proj"), s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.up_proj"),
                s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"), _ => parts[1].to_string()
            };
            target_name = format!("model.language_model.layers.{}.{}", parts[0], mapped);
        } else if name == "token_embd.weight" || name == "token_embd.packed" { target_name = "model.language_model.embed_tokens.weight".to_string(); }
        else if name == "output_norm.weight" { target_name = "model.language_model.norm.weight".to_string(); }
        else if name == "output.weight" || name == "lm_head.weight" { target_name = "lm_head.weight".to_string(); }

        if name.ends_with(".packed") {
            let base = name.strip_suffix(".packed").unwrap();
            let target_base = target_name.strip_suffix(".packed").unwrap();
            if base.contains("token_embd") {
                let scale = f32::from_le_bytes(st.tensor(&format!("{base}.scale"))?.data()[0..4].try_into().unwrap());
                let unpacked: Vec<f32> = view.data().iter().map(|&p| (p as f32 - 1.5) * scale).collect();
                let vocab = unpacked.len() / hidden_size;
                data.insert(target_base.to_string(), Tensor::from_vec(unpacked, (vocab, hidden_size), &Device::Cpu)?.to_device(device)?.to_dtype(dtype)?);
            } else {
                let scales_vec: Vec<f32> = st.tensor(&format!("{base}.scales"))?.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
                let shape: Vec<usize> = st.tensor(&format!("{base}.shape"))?.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
                let mut decoded = vec![0.0f32; shape.iter().product()]; let packed_vec = view.data();
                decoded.par_chunks_mut(256).enumerate().for_each(|(b_idx, b_out)| {
                    if b_idx < scales_vec.len() {
                        let s = scales_vec[b_idx]; let p_start = b_idx * 32;
                        for i in 0..32 {
                            let gp = p_start + i; if gp >= packed_vec.len() { break; }
                            let byte = packed_vec[gp];
                            for bit in 0..8 {
                                let idx = i * 8 + bit; if idx < b_out.len() { b_out[idx] = if (byte & (1 << bit)) != 0 { s } else { -s }; }
                            }
                        }
                    }
                });
                data.insert(target_base.to_string(), Tensor::from_vec(decoded, shape.as_slice(), &Device::Cpu)?.to_device(device)?.to_dtype(dtype)?);
            }
        } else if !name.contains(".scales") && !name.contains(".scale") && !name.contains(".shape") {
            let raw = view.data();
            let f32_data: Vec<f32> = if raw.len() % 4 == 0 { raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
            else { raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect() };
            data.insert(target_name.to_string(), Tensor::from_vec(f32_data, view.shape(), &Device::Cpu)?.to_device(device)?.to_dtype(dtype)?);
        }
    }
    Ok(VarBuilder::from_tensors(data, dtype, device))
}