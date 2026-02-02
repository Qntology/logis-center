use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Activation, Embedding, Init, Linear, VarBuilder};
pub use crate::models::common::RmsNorm;
use std::collections::HashMap;

use crate::{
    models::{
        common::{GateUpDownMLP, NaiveAttention, TwoLinearMLP, eager_attention_forward, LayerNorm},
    },
    position_embed::rope::{
        Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        mask_index_add, masked_scatter_dim0,
        nonzero_index,
        get_vision_next_indices,
    },
};

// --- QLinear ---
#[derive(Debug, Clone)]
pub struct QLinear { pub weight: Tensor, pub bias: Option<Tensor> }
impl QLinear {
    pub fn new_direct(weight: Tensor, bias: Option<Tensor>) -> Self { Self { weight, bias } }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(b) = &mut self.bias { *b = b.to_device(device)?; } Ok(())
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let xs_flat = xs.flatten(0, 1)?.contiguous()?;
        let mut out = xs_flat.matmul(&self.weight.t()?.contiguous()?)?;
        if let Some(bias) = &self.bias { out = out.broadcast_add(bias)?; }
        Ok(out.reshape((b, s, ()))?)
    }
}

// --- Vision (Minimal for Text Stability) ---
#[derive(Debug, Clone)] pub struct Qwen3VLVisionModel { patch_embed: Tensor }
impl Qwen3VLVisionModel {
    pub fn new(_cfg: crate::models::qwen3vl::config::Qwen3VLVisionConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self { patch_embed: Tensor::zeros(1, DType::F32, &_vb.device())? })
    }
    pub fn forward(&mut self, x: &Tensor, _thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> { Ok((x.clone(), vec![])) }
    pub fn to_device(&mut self, _d: &Device) -> Result<()> { Ok(()) }
}

// --- Text Components ---
#[derive(Debug, Clone)] pub struct Qwen3VLTextAttention {
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear,
    pub q_norm: RmsNorm, pub k_norm: RmsNorm, pub num_attention_heads: usize, pub num_key_value_heads: usize, pub num_kv_groups: usize,
    pub head_dim: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
}
impl Qwen3VLTextAttention {
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bs, ql, _) = x.dims3()?;
        let qs = self.q_norm.forward(&self.q_proj.forward(x)?.reshape((bs, ql, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let ks = self.k_norm.forward(&self.k_proj.forward(x)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?.contiguous()?;
        let vs = self.v_proj.forward(x)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let (qs, ks) = apply_rotary_pos_emb(&qs, &ks, cos, sin, false)?;
        let (ks, vs) = match &self.kv_cache { None => (ks, vs), Some((pk, pv)) => (Tensor::cat(&[pk, &ks], 2)?, Tensor::cat(&[pv, &vs], 2)?) };
        self.kv_cache = Some((ks.clone(), vs.clone()));
        let out = eager_attention_forward(&qs, &ks, &vs, Some(self.num_kv_groups), mask, self.scaling)?;
        Ok(self.o_proj.forward(&out.reshape((bs, ql, ()))?)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.q_proj.to_device(d)?; self.k_proj.to_device(d)?; self.v_proj.to_device(d)?; self.o_proj.to_device(d)?;
        self.q_norm.weight = self.q_norm.weight.to_device(d)?; self.k_norm.weight = self.k_norm.weight.to_device(d)?;
        if let Some((k, v)) = &mut self.kv_cache { *k = k.to_device(d)?; *v = v.to_device(d)?; } Ok(())
    }
}
#[derive(Debug, Clone)] pub struct QGateUpDownMLP { pub g: QLinear, pub u: QLinear, pub d: QLinear, pub act: Activation }
#[derive(Debug, Clone)] pub struct Qwen3VLTextDecoderLayer { pub attn: Qwen3VLTextAttention, pub mlp: QGateUpDownMLP, pub in_ln: RmsNorm, pub post_ln: RmsNorm }

#[derive(Debug, Clone)] pub struct Qwen3VLTextModel { pub embed: Embedding, pub layers: Vec<Option<Qwen3VLTextDecoderLayer>>, pub norm: RmsNorm, pub rotary: Qwen3VLTextRotaryEmbedding, pub mrope: Vec<usize>, pub is_baking: bool }
impl Qwen3VLTextModel {
    pub fn forward(&mut self, x: &Tensor, offset: usize, pids: Option<&Tensor>) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let pi = match pids { Some(ids) => ids.to_dtype(DType::F32)?.contiguous()?, None => Tensor::arange(offset as f32, (sl + offset) as f32, x.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))? };
        let (cos, sin) = self.rotary.forward(&pi.to_dtype(DType::U32)?, x.dtype(), self.mrope.clone())?;
        let (cos, sin) = (cos.contiguous()?, sin.contiguous()?);
        let mask = if sl <= 1 { None } else {
            let q = Tensor::arange(0f32, sl as f32, x.device())?.reshape((1, 1, sl, 1))?;
            let k = Tensor::arange(0f32, (offset + sl) as f32, x.device())?.reshape((1, 1, 1, offset + sl))?;
            Some(k.broadcast_gt(&(q.broadcast_add(&Tensor::new(offset as f32, x.device())?)?))?.to_dtype(x.dtype())?.affine(-65504.0, 0.0)?)
        };
        let mut x = x.clone(); let limit = if self.is_baking { 1 } else { self.layers.len() };
        for l_opt in self.layers.iter_mut().take(limit) { if let Some(l) = l_opt {
            let res = x.clone(); let x_norm = l.in_ln.forward(&x)?; let x_attn = l.attn.forward(&x_norm, &cos, &sin, mask.as_ref())?; x = res.add(&x_attn)?;
            let res = x.clone(); let x_norm = l.post_ln.forward(&x)?; let gate = l.mlp.act.forward(&l.mlp.g.forward(&x_norm)?)?; let up = l.mlp.u.forward(&x_norm)?; x = res.add(&l.mlp.d.forward(&gate.mul(&up)?)?)?;
        } }
        Ok(x.apply(&self.norm)?)
    }
    pub fn get_kv_len(&self) -> usize { for l in &self.layers { if let Some(li) = l { if let Some((k, _)) = &li.attn.kv_cache { return k.dim(2).unwrap_or(0); } } } 0 }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { if let Some(li) = l { li.attn.kv_cache = None; } } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let ew = self.embed.embeddings().to_device(d)?; self.embed = Embedding::new(ew, self.embed.hidden_size());
        for l in &mut self.layers { if let Some(li) = l { li.attn.to_device(d)?; li.in_ln.weight = li.in_ln.weight.to_device(d)?; li.post_ln.weight = li.post_ln.weight.to_device(d)?; } }
        self.norm.weight = self.norm.weight.to_device(d)?; Ok(())
    }
}

// --- Main ---
#[derive(Debug, Clone)] pub struct Qwen3VLModel { pub config: crate::models::qwen3vl::config::Qwen3VLConfig, pub visual: Option<Qwen3VLVisionModel>, pub language_model: Qwen3VLTextModel, pub lm_head: QLinear, pub is_baking: bool }
impl Qwen3VLModel {
    pub fn set_baking(&mut self, b: bool) { self.is_baking = b; self.language_model.is_baking = b; }
    pub fn new_ext(cfg: crate::models::qwen3vl::config::Qwen3VLConfig, vb: VarBuilder, _map: Option<HashMap<String, Tensor>>, _fto: bool, bo: bool) -> Result<Self> {
        let map_r = _map.as_ref().ok_or(anyhow!("Map required"))?;
        let model_vb = if vb.get_with_hints((1,), "model.embed_tokens.weight", Init::Const(0.)).is_ok() { vb.pp("model") } else { vb.clone() };
        let vb_lm = if model_vb.get_with_hints((1,), "language_model.layers.0.input_layernorm.weight", Init::Const(0.)).is_ok() { model_vb.pp("language_model") } else { model_vb.clone() };
        let (nh, hd, nkv, hs, eps, theta, mrope, nl, vs) = if let Some(tc) = &cfg.text_config { (tc.num_attention_heads, tc.head_dim, tc.num_key_value_heads, tc.hidden_size, tc.rms_norm_eps, tc.rope_theta, tc.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), tc.num_hidden_layers, tc.vocab_size) } else { (16, 128, 8, 1024, 1e-6, 1000000.0, vec![], 28, 151936) };
        let mut blks = vec![]; for i in 0..(if bo { 1 } else { nl }) {
            let vbl = vb_lm.pp("layers").pp(&i.to_string());
            // CRITICAL: Use empty string as name to avoid .weight.weight duplication
            let attn = Qwen3VLTextAttention { 
                q_proj: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn").pp("q_proj"), "", Some(map_r))?, 
                k_proj: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn").pp("k_proj"), "", Some(map_r))?, 
                v_proj: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn").pp("v_proj"), "", Some(map_r))?, 
                o_proj: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn").pp("o_proj"), "", Some(map_r))?, 
                q_norm: RmsNorm::new(vbl.get(hd, "self_attn.q_norm.weight")?, eps), k_norm: RmsNorm::new(vbl.get(hd, "self_attn.k_norm.weight")?, eps), num_attention_heads: nh, num_key_value_heads: nkv, num_kv_groups: nh/nkv, head_dim: hd, scaling: 1.0/(hd as f64).sqrt(), kv_cache: None 
            };
            blks.push(Some(Qwen3VLTextDecoderLayer { attn, mlp: QGateUpDownMLP { g: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp").pp("gate_proj"), "", Some(map_r))?, u: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp").pp("up_proj"), "", Some(map_r))?, d: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp").pp("down_proj"), "", Some(map_r))?, act: Activation::Silu }, in_ln: RmsNorm::new(vbl.get(hs, "input_layernorm.weight")?, eps), post_ln: RmsNorm::new(vbl.get(hs, "post_attention_layernorm.weight")?, eps) }));
        }
        let lm = Qwen3VLTextModel { embed: Embedding::new(vb_lm.get((vs, hs), "embed_tokens.weight")?, hs), layers: blks, norm: RmsNorm::new(vb_lm.get(hs, if vb_lm.get_with_hints((1,), "norm.weight", Init::Const(0.)).is_ok() { "norm.weight" } else { "output_norm.weight" })?, eps), rotary: Qwen3VLTextRotaryEmbedding::new(hd, theta), mrope, is_baking: bo };
        Ok(Self { config: cfg, visual: None, language_model: lm, lm_head: crate::models::qwen3vl::quantized_model::get_qlinear_safe(vb_lm.pp("lm_head"), "", Some(map_r))?, is_baking: bo })
    }
    pub fn forward(&mut self, ids: &Tensor, _pv: Option<&Tensor>, _thw: Option<&Tensor>, _vpv: Option<&Tensor>, _vthw: Option<&Tensor>, _cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (bs, sl) = ids.dims2()?; let mut embs = self.language_model.embed.forward(&ids.to_dtype(DType::U32)?.contiguous()?)?;
        if let Some(ref mut vm) = &mut self.visual { if let (Some(p), Some(t)) = (_pv, _thw) { let (ie, _) = vm.forward(p, t)?; embs = masked_scatter_dim0(&embs, &ie.to_device(ids.device())?, &ids.broadcast_eq(&Tensor::new(vec![self.config.image_token_id.unwrap_or(0) as u32], ids.device())?)?.to_dtype(DType::U32)?)?; } }
        let out = self.language_model.forward(&embs, offset, None)?;
        let out_last = out.narrow(1, sl - 1, 1)?; if self.is_baking { return Ok(out_last); } Ok(self.lm_head.forward(&out_last)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { if let Some(v) = &mut self.visual { v.to_device(d)?; } self.language_model.to_device(d)?; self.lm_head.to_device(d)?; Ok(()) }
    pub fn device(&self) -> &Device { self.language_model.embed.embeddings().device() }
    pub fn get_kv_len(&self) -> usize { self.language_model.get_kv_len() }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn save_kv_cache(&mut self, p: &std::path::Path) -> Result<()> {
        if !p.exists() { std::fs::create_dir_all(p)?; }
        for (i, l_opt) in self.language_model.layers.iter_mut().enumerate() {
            if let Some(l) = l_opt { if let Some((k, v)) = &l.attn.kv_cache {
                let mut m = HashMap::new(); let rk = comp_bit_static(i, k)?; let rv = comp_bit_static(i, v)?;
                m.insert("k_a".to_string(), rk.0); m.insert("k_p".to_string(), rk.1); m.insert("k_s".to_string(), rk.2);
                m.insert("v_a".to_string(), rv.0); m.insert("v_p".to_string(), rv.1); m.insert("v_s".to_string(), rv.2);
                m.insert("shape".to_string(), Tensor::from_vec(rk.3.iter().map(|&x| x as u32).collect::<Vec<_>>(), (rk.3.len(),), k.device())?);
                candle_core::safetensors::save(&m, p.join(format!("layer_{}_bitkv.safetensors", i)))?;
            } }
        } Ok(())
    }
    pub fn load_kv_cache(&mut self, _p: &std::path::Path, _d: &Device) -> Result<()> { Ok(()) }
}
fn comp_bit_static(i: usize, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
    let s = t.dims().to_vec(); let d = t.device(); if i == 0 { return Ok((t.clone(), Tensor::zeros(1, DType::U8, d)?, Tensor::zeros(1, DType::F32, d)?, s)); }
    let f = t.flatten_all()?; let n = f.dim(0)?; let sc = if i <= 4 { (f.abs()?.max_all()?.to_scalar::<f32>()? / 3.0).max(1e-6) } else { f.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6) };
    if i <= 4 {
        let q = f.to_vec1::<f32>()?; let mut p = vec![0u8; (n + 3) / 4];
        for (j, &v) in q.iter().enumerate() { p[j/4] |= ((v/sc + 1.0).round() as u8).clamp(0,3) << ((j%4)*2); }
        Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n+3)/4,), d)?, Tensor::new(&[sc], d)?, s))
    } else {
        let sv = f.ge(0.0)?.to_vec1::<u8>()?; let mut p = vec![0u8; (n + 7) / 8];
        for (j, &v) in sv.iter().enumerate() { if v > 0 { p[j/8] |= 1 << (j%8); } }
        Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n+7)/8,), d)?, Tensor::new(&[sc], d)?, s))
    }
}
fn apply_bridge_static(x: &Tensor, td: usize) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?; let x_f = x.to_dtype(DType::F32)?;
    let rms = (x_f.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()).max(1e-6);
    let sc = (d as f64 / td as f64).sqrt() * 0.707 / (rms as f64);
    if td >= d {
        let l = x_f.clone();
        let u = if td > d { let r = x_f.roll(1, D::Minus1)?; let lr = ((l + r)? * 0.5)?; Tensor::stack(&[x_f, lr], D::Minus1)?.reshape((b, h, s, td))? }
        else { x_f.affine(sc * (rms as f64), 0.0)? };
        Ok(u.clamp(-10.0, 10.0)?.to_dtype(x.dtype())?)
    } else { Ok(x.narrow(D::Minus1, 0, td)?.to_dtype(DType::F32)?.affine(sc, 0.0)?.to_dtype(x.dtype())?) }
}
