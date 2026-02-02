use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Activation, Embedding, VarBuilder};
use candle_core::quantized::gguf_file;
use rayon::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;
use safetensors::tensor::SafeTensors;

use crate::{
    models::{
        common::eager_attention_forward,
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig},
        qwen3vl::model::{Qwen3VLVisionModel, RmsNorm, QLinear},
    },
    position_embed::rope::{
        Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        mask_index_add, masked_scatter_dim0,
    },
};

pub fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{}.weight", name), device)?;
    let bias = ct.tensor(reader, &format!("{}.bias", name), device).ok().map(|t| t.dequantize(device).unwrap().to_dtype(dtype).unwrap());
    let w_dequant = weight.dequantize(device)?.to_dtype(dtype)?;
    Ok(QLinear::new_direct(w_dequant, bias))
}

pub fn get_qlinear_safe(vb: VarBuilder, name: &str, map: Option<&HashMap<String, Tensor>>) -> Result<QLinear> {
    let prefix = vb.prefix();
    let get_path = |p: &str| -> String { if prefix.is_empty() { p.to_string() } else if prefix.ends_with('.') { format!("{}{}", prefix, p) } else { format!("{}.{}", prefix, p) } };
    let packed_path = get_path(&format!("{}.packed", name));
    
    if let Some(m) = map {
        if let Some(p) = m.get(&packed_path) {
            let s = m.get(&get_path(&format!("{}.scales", name))).ok_or(anyhow!("Missing scales"))?.to_dtype(DType::F32)?;
            let sh_t = m.get(&get_path(&format!("{}.shape", name))).ok_or(anyhow!("Missing shape"))?;
            let sh: Vec<usize> = sh_t.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let b = m.get(&get_path(&format!("{}.bias", name))).cloned();
            
            let total_el: usize = sh.iter().product();
            let p_cpu = p.to_device(&Device::Cpu)?;
            let p_u32 = p_cpu.to_dtype(DType::U32)?;
            let p_data = p_u32.to_vec1::<u32>()?;
            let s_data = s.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            
            let mut weights = vec![0.0f32; total_el];
            weights.par_chunks_exact_mut(32).enumerate().for_each(|(b_i, block)| {
                if b_i < s_data.len() && b_i < p_data.len() {
                    let sc = s_data[b_i]; let bits = p_data[b_i];
                    for i in 0..32 { if b_i * 32 + i < total_el { block[i] = sc * (if (bits >> i) & 1 != 0 { 1.0 } else { -1.0 }); } }
                }
            });
            let w_tensor = Tensor::from_vec(weights, sh.as_slice(), vb.device())?.to_dtype(vb.dtype())?;
            return Ok(QLinear::new_direct(w_tensor, b));
        }
    }
    let w = vb.get_with_hints((1,), name, candle_nn::Init::Const(0.0))?.to_dtype(vb.dtype())?;
    Ok(QLinear::new_direct(w, None))
}

#[derive(Clone)] pub struct QuantizedQwen3VLTextAttention { pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear, pub q_norm: RmsNorm, pub k_norm: RmsNorm, pub num_attention_heads: usize, pub num_key_value_heads: usize, pub head_dim: usize, pub num_kv_groups: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)> }
impl QuantizedQwen3VLTextAttention {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj.to_device(device)?; self.k_proj.to_device(device)?; self.v_proj.to_device(device)?; self.o_proj.to_device(device)?;
        self.q_norm.weight = self.q_norm.weight.to_device(device)?; self.k_norm.weight = self.k_norm.weight.to_device(device)?;
        if let Some((k, v)) = &mut self.kv_cache { *k = k.to_device(device)?; *v = v.to_device(device)?; } Ok(())
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let qs = self.q_norm.forward(&self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let ks = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let vs = self.v_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let (qs, ks) = apply_rotary_pos_emb(&qs, &ks, cos, sin, false)?;
        let (ks, vs) = match &self.kv_cache { None => (ks, vs), Some((pk, pv)) => (Tensor::cat(&[pk, &ks], 2)?, Tensor::cat(&[pv, &vs], 2)?) };
        self.kv_cache = Some((ks.clone(), vs.clone()));
        let attn_output = eager_attention_forward(&qs, &ks, &vs, Some(self.num_kv_groups), mask, self.scaling)?;
        Ok(self.o_proj.forward(&attn_output.reshape((b_sz, q_len, ()))?)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None; }
}

#[derive(Clone)] pub struct QuantizedQwen3VLTextDecoderLayer { pub self_attn: QuantizedQwen3VLTextAttention, pub mlp_gate: Option<QLinear>, pub mlp_up: Option<QLinear>, pub mlp_down: Option<QLinear>, pub input_layernorm: RmsNorm, pub post_attention_layernorm: Option<RmsNorm> }
impl QuantizedQwen3VLTextDecoderLayer {
    pub fn forward_with_params(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let res = xs.clone(); let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask)?; let xs = res.add(&xs)?;
        if let (Some(g), Some(u), Some(d), Some(n)) = (&self.mlp_gate, &self.mlp_up, &self.mlp_down, &self.post_attention_layernorm) {
            let res_mlp = xs.clone(); let xs_mlp = n.forward(&xs)?;
            let gate = candle_nn::ops::silu(&g.forward(&xs_mlp)?)?; let up = u.forward(&xs_mlp)?; Ok(res_mlp.add(&d.forward(&gate.mul(&up)?)?)?)
        } else { Ok(xs) }
    }
    pub fn new<R: std::io::Seek + std::io::Read>(cfg: &Qwen3VLTextConfig, ct: &gguf_file::Content, r: &mut R, b_n: &str, dev: &Device, dt: DType, _l_i: usize, b_o: bool) -> Result<Self> {
        let is_g = b_n.contains("blk.");
        let (q, k, v, o, qn, kn, g, u, d, n, in_ln) = if is_g { ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down", "ffn_norm", "attn_norm") }
        else { ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm", "input_layernorm") };
        let self_attn = QuantizedQwen3VLTextAttention { q_proj: get_qlinear(ct, r, &format!("{b_n}.{q}"), dev, dt)?, k_proj: get_qlinear(ct, r, &format!("{b_n}.{k}"), dev, dt)?, v_proj: get_qlinear(ct, r, &format!("{b_n}.{v}"), dev, dt)?, o_proj: get_qlinear(ct, r, &format!("{b_n}.{o}"), dev, dt)?, q_norm: RmsNorm::new(ct.tensor(r, &format!("{b_n}.{qn}"), dev)?.dequantize(dev)?.to_dtype(dt)?, cfg.rms_norm_eps), k_norm: RmsNorm::new(ct.tensor(r, &format!("{b_n}.{kn}"), dev)?.dequantize(dev)?.to_dtype(dt)?, cfg.rms_norm_eps), num_attention_heads: cfg.num_attention_heads, num_key_value_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads, scaling: 1f64 / f64::sqrt(cfg.head_dim as f64), kv_cache: None };
        let (mlp_gate, mlp_up, mlp_down, post_attention_layernorm) = if !b_o { (Some(get_qlinear(ct, r, &format!("{b_n}.{g}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{u}"), dev, dt)?), Some(get_qlinear(ct, r, &format!("{b_n}.{d}"), dev, dt)?), Some(RmsNorm::new(ct.tensor(r, &format!("{b_n}.{n}"), dev)?.dequantize(dev)?.to_dtype(dt)?, cfg.rms_norm_eps))) } else { (None, None, None, None) };
        let input_layernorm = RmsNorm::new(ct.tensor(r, &format!("{b_n}.{in_ln}"), dev)?.dequantize(dev)?.to_dtype(dt)?, cfg.rms_norm_eps);
        Ok(Self { self_attn, mlp_gate, mlp_up, mlp_down, input_layernorm, post_attention_layernorm })
    }
}

#[derive(Clone)] pub struct QuantizedQwen3VLTextModel { pub embed_tokens: Embedding, pub layers: Vec<QuantizedQwen3VLTextDecoderLayer>, pub norm: RmsNorm, pub rotary_emb: Qwen3VLTextRotaryEmbedding, pub mrope_section: Vec<usize>, pub mmap: Option<Arc<Mmap>>, pub is_forced_cpu: bool, pub is_baking: bool }
impl QuantizedQwen3VLTextModel {
    pub fn new_with_mmap(config: &Qwen3VLTextConfig, ct: &gguf_file::Content, mmap_handle: Option<Arc<Mmap>>, _base_name: &str, device: &Device, _device_id: usize, dtype: DType, _kvr: u64, baking_only: bool) -> Result<Self> {
        let mut reader = std::io::Cursor::new(mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let prefix = if ct.tensor(&mut reader, "model.layers.0.input_layernorm.weight", device).is_ok() { "model.layers" } else if ct.tensor(&mut reader, "model.language_model.layers.0.input_layernorm.weight", device).is_ok() { "model.language_model.layers" } else { "blk" };
        let (embed_tokens, _h) = if let Ok(t) = ct.tensor(&mut reader, if prefix == "blk" { "token_embd.weight" } else { "model.embed_tokens.weight" }, device) { let t = t.dequantize(device)?.to_dtype(dtype)?; let h = t.dim(1)?; (Embedding::new(t, h), h) } else { return Err(anyhow!("Embed Load Fail")); };
        let mut layers = vec![]; for i in 0..(if baking_only { 1 } else { config.num_hidden_layers }) { layers.push(QuantizedQwen3VLTextDecoderLayer::new(config, ct, &mut reader, &format!("{prefix}.{i}"), device, dtype, i, baking_only)?); }
        let norm = RmsNorm::new(ct.tensor(&mut reader, if prefix == "blk" { "output_norm.weight" } else { "model.norm.weight" }, device)?.dequantize(device)?.to_dtype(dtype)?, config.rms_norm_eps);
        Ok(Self { embed_tokens, layers, norm, rotary_emb: Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta), mrope_section: config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), mmap: mmap_handle, is_forced_cpu: device.is_cpu(), is_baking: baking_only })
    }
    pub fn forward(&mut self, x: &Tensor, offset: usize, pids: Option<&Tensor>, vm: Option<&Tensor>, ds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let pi = match pids { Some(ids) => ids.to_dtype(DType::F32)?.contiguous()?, None => Tensor::arange(offset as f32, (sl + offset) as f32, x.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))? };
        let (cos, sin) = self.rotary_emb.forward(&pi.to_dtype(DType::U32)?, x.dtype(), self.mrope_section.clone())?;
        let (cos, sin) = (cos.contiguous()?, sin.contiguous()?);
        let mask = if sl <= 1 { None } else { let q = Tensor::arange(0f32, sl as f32, x.device())?.reshape((1, 1, sl, 1))?; let k = Tensor::arange(0f32, (offset + sl) as f32, x.device())?.reshape((1, 1, 1, offset + sl))?; Some(k.broadcast_gt(&(q.broadcast_add(&Tensor::new(offset as f32, x.device())?)?))?.to_dtype(x.dtype())?.affine(-65504.0, 0.0)?) };
        let mut x = x.clone(); for (i, layer) in self.layers.iter_mut().enumerate() { x = layer.forward_with_params(&x, &cos, &sin, mask.as_ref())?; if let (Some(m), Some(d)) = (vm, ds.as_ref()) { if i < d.len() { x = mask_index_add(&x.squeeze(0)?, &m.squeeze(0)?, &d[i])?.unsqueeze(0)?; } } }
        Ok(x.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { l.self_attn.clear_kv_cache(); } }
    pub fn get_kv_len(&self) -> usize { self.layers[0].self_attn.kv_cache.as_ref().map(|(k,_)| k.dim(2).unwrap_or(0)).unwrap_or(0) }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { let w = self.embed_tokens.embeddings().to_device(d)?; self.embed_tokens = Embedding::new(w, self.embed_tokens.hidden_size()); for l in &mut self.layers { l.self_attn.to_device(d)?; } self.norm.weight = self.norm.weight.to_device(d)?; Ok(()) }
    pub fn save_kv_cache(&mut self, _path: &Path, _clear: bool, _b: usize) -> Result<()> { Ok(()) }
    pub fn load_kv_cache(&mut self, _path: &Path, _d: &Device, _e: usize, _u: usize) -> Result<()> { Ok(()) }
}

#[derive(Clone)] pub struct QuantizedQwen3TextModel { pub language_model: QuantizedQwen3VLTextModel, pub lm_head: Option<QLinear> }
impl QuantizedQwen3TextModel {
    pub fn new_with_mmap(cfg: &Qwen3VLConfig, ct: &gguf_file::Content, mm: Option<Arc<Mmap>>, td: &Device, tdi: usize, dt: DType, kvr: u64, bo: bool, _s: bool) -> Result<Self> {
        let lm = QuantizedQwen3VLTextModel::new_with_mmap(cfg.text_config.as_ref().unwrap(), ct, mm.clone(), "model", td, tdi, dt, kvr, bo)?;
        let mut r = std::io::Cursor::new(mm.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lh = get_qlinear(ct, &mut r, "lm_head", td, dt).ok();
        Ok(Self { language_model: lm, lm_head: lh })
    }
    pub fn forward(&mut self, ids: &Tensor, cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (bs, sl) = ids.dims2()?; let embs = self.language_model.embed_tokens.forward(&ids.flatten_all()?)?.reshape((bs, sl, ()))?;
        let pos = match cp { Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize, None => offset };
        let pids = Tensor::arange(pos as u32, (pos + sl) as u32, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), None, None)?;
        let h = out.narrow(1, sl - 1, 1)?; if let Some(lh) = &self.lm_head { Ok(lh.forward(&h)?) } else { Ok(h) }
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.language_model.to_device(d)?; if let Some(lh) = &mut self.lm_head { lh.to_device(d)?; } Ok(()) }
    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(p, c, b) }
    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(p, d, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
}

#[derive(Clone)] pub struct QuantizedQwen3VLModel { pub config: Qwen3VLConfig, pub visual: Option<Qwen3VLVisionModel>, pub language_model: QuantizedQwen3VLTextModel, pub lm_head: QLinear, pub is_baking: bool }
impl QuantizedQwen3VLModel {
    pub fn new_with_mmap(cfg: &Qwen3VLConfig, ctm: &gguf_file::Content, mm: Option<Arc<Mmap>>, _ctv: &gguf_file::Content, _vmm: Option<Arc<Mmap>>, td: &Device, tdi: usize, vd: &Device, _vi: usize, dt: DType, kvr: u64, bo: bool, force_text_only: bool) -> Result<Self> {
        let visual = if !force_text_only && cfg.vision_config.is_some() { Some(Qwen3VLVisionModel::new(cfg.vision_config.as_ref().unwrap().clone(), VarBuilder::from_tensors(HashMap::new(), dt, vd))?) } else { None };
        let lm = QuantizedQwen3VLTextModel::new_with_mmap(cfg.text_config.as_ref().unwrap(), ctm, mm.clone(), "model", td, tdi, dt, kvr, bo)?;
        let mut r = std::io::Cursor::new(mm.as_ref().map(|m| &m[..]).unwrap_or(&[]));
        let lh = get_qlinear(ctm, &mut r, "lm_head", td, dt)?;
        Ok(Self { config: cfg.clone(), visual, language_model: lm, lm_head: lh, is_baking: bo })
    }
    pub fn forward(&mut self, input_ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, _vthw: Option<&Tensor>, cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (bs, sl) = input_ids.dims2()?; let mut embs = self.language_model.embed_tokens.forward(&input_ids.to_dtype(DType::U32)?.contiguous()?)?;
        if let (Some(ref mut vm), Some(pv), Some(thw)) = (&mut self.visual, pv, thw) { let (ie, _) = vm.forward(pv, thw)?; embs = masked_scatter_dim0(&embs, &ie.to_device(input_ids.device())?, &input_ids.broadcast_eq(&Tensor::new(vec![self.config.image_token_id.unwrap_or(0) as u32], input_ids.device())?)?.to_dtype(DType::U32)?)?; }
        let pos = match cp { Some(p) => p.flatten_all()?.i(0)?.to_scalar::<u32>()? as usize, None => offset };
        let pi = Tensor::arange(pos as f32, (pos + sl) as f32, input_ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))?;
        let out = self.language_model.forward(&embs, offset, Some(&pi), None, None)?;
        Ok(self.lm_head.forward(&out.narrow(1, sl - 1, 1)?)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { if let Some(v) = &mut self.visual { v.to_device(d)?; } self.language_model.to_device(d)?; self.lm_head.to_device(d)?; Ok(()) }
    pub fn save_kv_cache(&mut self, p: &Path, c: bool, b: usize) -> Result<()> { self.language_model.save_kv_cache(p, c, b) }
    pub fn load_kv_cache(&mut self, p: &Path, d: &Device, e: usize, u: usize) -> Result<()> { self.language_model.load_kv_cache(p, d, e, u) }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

pub fn load_tensors_from_true_iq0(path: &Path, device: &Device, _dtype: DType, _baking: bool) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path)?; let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?; let mut tensors = HashMap::new();
    for (name, view) in safetensors.tensors() {
        let shape = view.shape().to_vec(); let data = view.data();
        if name.ends_with(".packed") {
             let u_data: Vec<u32> = data.chunks_exact(4).map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
             let mut new_shape = shape.clone(); if let Some(last) = new_shape.last_mut() { if *last == data.len() { *last /= 4; } }
             tensors.insert(name.to_string(), Tensor::from_vec(u_data, new_shape.as_slice(), device)?);
        } else {
            match view.dtype() {
                safetensors::Dtype::F32 => { tensors.insert(name.to_string(), Tensor::from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()/4) }, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::U32 => { tensors.insert(name.to_string(), Tensor::from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()/4) }, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
                safetensors::Dtype::I32 => { 
                    let u_data: Vec<u32> = data.chunks_exact(4).map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
                    tensors.insert(name.to_string(), Tensor::from_vec(u_data, shape.as_slice(), device)?);
                },
                _ => { tensors.insert(name.to_string(), Tensor::from_slice(data, shape.as_slice(), &Device::Cpu)?.to_device(device)?); },
            }
        };
    }
    Ok(tensors)
}
