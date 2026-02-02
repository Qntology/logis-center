use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Activation, Embedding, Init, Linear, VarBuilder};
pub use crate::models::common::RmsNorm;
use std::collections::HashMap;

use crate::{
    models::{
        common::{GateUpDownMLP, NaiveAttention, TwoLinearMLP, eager_attention_forward, LayerNorm},
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLVisionConfig},
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

// --- Vision ---
#[derive(Debug, Clone)] pub struct Qwen3VLVisionPatchEmbed { conv3d_weight: Tensor, conv3d_bias: Tensor }
impl Qwen3VLVisionPatchEmbed {
    pub fn new(_cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let w = vb.get_with_hints((1,), "proj.weight", Init::Const(0.))?;
        let w = vb.get(w.dims(), "proj.weight")?.flatten(1, 4)?.t()?;
        let b = vb.get_with_hints((1,), "proj.bias", Init::Const(0.))?;
        let b = vb.get(b.dims(), "proj.bias")?.unsqueeze(0)?;
        Ok(Self { conv3d_weight: w, conv3d_bias: b })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.matmul(&self.conv3d_weight.to_dtype(x.dtype())?)?.broadcast_add(&self.conv3d_bias.to_dtype(x.dtype())?)?) }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.conv3d_weight = self.conv3d_weight.to_device(d)?; self.conv3d_bias = self.conv3d_bias.to_device(d)?; Ok(()) }
}
#[derive(Debug, Clone)] pub struct Qwen3VLVisionPatchMerger { norm: LayerNorm, linear_fc1: Linear, act_fn: Activation, linear_fc2: Linear }
impl Qwen3VLVisionPatchMerger {
    pub fn new(_cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let ln_w = vb.get((), "norm.weight")?; let ln_b = vb.get((), "norm.bias")?;
        let fc1_w = vb.get((), "linear_fc1.weight")?; let fc1_b = vb.get((), "linear_fc1.bias")?;
        let fc2_w = vb.get((), "linear_fc2.weight")?; let fc2_b = vb.get((), "linear_fc2.bias")?;
        Ok(Self { norm: LayerNorm::new(ln_w, ln_b, 1e-6), linear_fc1: Linear::new(fc1_w, Some(fc1_b)), act_fn: Activation::Gelu, linear_fc2: Linear::new(fc2_w, Some(fc2_b)) })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_fc1.forward(&self.norm.forward(x)?)?; let x = self.act_fn.forward(&x)?; Ok(self.linear_fc2.forward(&x)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { 
        let nw = self.norm.weight.to_device(d)?; let nb = self.norm.bias.to_device(d)?; self.norm = LayerNorm::new(nw, nb, 1e-6);
        let f1w = self.linear_fc1.weight().to_device(d)?; let f1b = self.linear_fc1.bias().as_ref().unwrap().to_device(d)?; self.linear_fc1 = Linear::new(f1w, Some(f1b));
        let f2w = self.linear_fc2.weight().to_device(d)?; let f2b = self.linear_fc2.bias().as_ref().unwrap().to_device(d)?; self.linear_fc2 = Linear::new(f2w, Some(f2b)); Ok(())
    }
}
#[derive(Debug, Clone)] pub struct Qwen3VLVisionBlock { attn: NaiveAttention, mlp: TwoLinearMLP, norm1: LayerNorm, norm2: LayerNorm }
impl Qwen3VLVisionBlock {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let n1w = vb.get((), "norm1.weight")?; let n1b = vb.get((), "norm1.bias")?;
        let n2w = vb.get((), "norm2.weight")?; let n2b = vb.get((), "norm2.bias")?;
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size); let head_dim = embed_dim / cfg.num_heads;
        Ok(Self { attn: NaiveAttention::new(vb.pp("attn"), cfg.hidden_size, cfg.num_heads, cfg.num_heads, Some(head_dim), true, Some("o_proj"))?, mlp: TwoLinearMLP::new(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size, Activation::Gelu, true, "linear1", "linear2")?, norm1: LayerNorm::new(n1w, n1b, 1e-6), norm2: LayerNorm::new(n2w, n2b, 1e-6) })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let r = x.clone(); let x = self.attn.forward(&self.norm1.forward(x)?, Some(cos), Some(sin), None, false)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.norm2.forward(&x)?)?; Ok(r.add(&x)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.attn.to_device(d)?; self.mlp.to_device(d)?;
        let n1w = self.norm1.weight.to_device(d)?; let n1b = self.norm1.bias.to_device(d)?; self.norm1 = LayerNorm::new(n1w, n1b, 1e-6);
        let n2w = self.norm2.weight.to_device(d)?; let n2b = self.norm2.bias.to_device(d)?; self.norm2 = LayerNorm::new(n2w, n2b, 1e-6); Ok(())
    }
}
#[derive(Debug, Clone)] pub struct Qwen3VLVisionModel { patch_embed: Qwen3VLVisionPatchEmbed, blocks: Vec<Qwen3VLVisionBlock>, merger: Qwen3VLVisionPatchMerger, rotary: Qwen2_5VisionRotaryEmbedding }
impl Qwen3VLVisionModel {
    pub fn new(cfg: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let pe = Qwen3VLVisionPatchEmbed::new(&cfg, vb.pp("patch_embed"))?;
        let mut blks = vec![]; for i in 0..cfg.depth { blks.push(Qwen3VLVisionBlock::new(&cfg, vb.pp("blocks").pp(&i.to_string()))?); }
        let me = Qwen3VLVisionPatchMerger::new(&cfg, vb.pp("merger"))?;
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size);
        Ok(Self { patch_embed: pe, blocks: blks, merger: me, rotary: Qwen2_5VisionRotaryEmbedding::new(embed_dim / cfg.num_heads, None) })
    }
    pub fn forward(&mut self, x: &Tensor, thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let x = self.patch_embed.forward(x)?; let cos_sin = self.rotary.forward(thw.dim(0)?, x.device())?; let (cos, sin) = (cos_sin.clone(), cos_sin); 
        let mut x = x; let mut ds = vec![]; for b in &mut self.blocks { x = b.forward(&x, &cos, &sin)?; ds.push(x.clone()); } Ok((self.merger.forward(&x)?, ds))
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.patch_embed.to_device(d)?; for b in &mut self.blocks { b.to_device(d)?; } self.merger.to_device(d)?; Ok(()) }
}

// --- Text ---
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
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.q_proj.to_device(d)?; self.k_proj.to_device(d)?; self.v_proj.to_device(d)?; self.o_proj.to_device(d)?;
        self.q_norm.weight = self.q_norm.weight.to_device(d)?; self.k_norm.weight = self.k_norm.weight.to_device(d)?;
        if let Some((k, v)) = &mut self.kv_cache { *k = k.to_device(d)?; *v = v.to_device(d)?; } Ok(())
    }
}
#[derive(Debug, Clone)] pub struct QGateUpDownMLP { pub g: QLinear, pub u: QLinear, pub d: QLinear, pub act: Activation }
impl QGateUpDownMLP {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.act.forward(&self.g.forward(x)?)?; let up = self.u.forward(x)?; Ok(self.d.forward(&gate.mul(&up)?)?)
    }
}
#[derive(Debug, Clone)] pub struct Qwen3VLTextDecoderLayer { pub attn: Qwen3VLTextAttention, pub mlp: QGateUpDownMLP, pub in_ln: RmsNorm, pub post_ln: RmsNorm }
impl Qwen3VLTextDecoderLayer {
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = x.clone(); let x = self.attn.forward(&self.in_ln.forward(x)?, cos, sin, mask)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.post_ln.forward(&x)?)?; Ok(r.add(&x)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.attn.to_device(d)?; self.in_ln.weight = self.in_ln.weight.to_device(d)?; self.post_ln.weight = self.post_ln.weight.to_device(d)?; Ok(())
    }
}

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
        for l_opt in self.layers.iter_mut().take(limit) { if let Some(l) = l_opt { x = l.forward(&x, &cos, &sin, mask.as_ref())?; } }
        Ok(x.apply(&self.norm)?)
    }
    pub fn get_kv_len(&self) -> usize { for l in &self.layers { if let Some(li) = l { if let Some((k, _)) = &li.attn.kv_cache { return k.dim(2).unwrap_or(0); } } } 0 }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { if let Some(li) = l { li.attn.clear_kv_cache() } } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let ew = self.embed.embeddings().to_device(d)?; self.embed = Embedding::new(ew, self.embed.hidden_size());
        for l in &mut self.layers { if let Some(li) = l { li.to_device(d)?; } }
        self.norm.weight = self.norm.weight.to_device(d)?; Ok(())
    }
}

// --- Main ---
#[derive(Debug, Clone)] pub struct Qwen3VLModel { pub config: Qwen3VLConfig, pub visual: Option<Qwen3VLVisionModel>, pub language_model: Qwen3VLTextModel, pub lm_head: QLinear, pub is_baking: bool }
impl Qwen3VLModel {
    pub fn set_baking(&mut self, b: bool) { self.is_baking = b; self.language_model.is_baking = b; }
    pub fn new_ext(cfg: Qwen3VLConfig, vb: VarBuilder, _map: Option<HashMap<String, Tensor>>, fto: bool, bo: bool) -> Result<Self> {
        let map_r = _map.as_ref();
        let root = if vb.get_with_hints((1,), "model.layers.0.input_layernorm.weight", Init::Const(0.)).is_ok() { "model" } else if vb.get_with_hints((1,), "language_model.layers.0.input_layernorm.weight", Init::Const(0.)).is_ok() { "language_model" } else { "" };
        let vb_base = if root.is_empty() { vb.clone() } else { vb.pp(root) };
        let vb_lm = if vb_base.get_with_hints((1,), "language_model.layers.0.input_layernorm.weight", Init::Const(0.)).is_ok() { vb_base.pp("language_model") } else { vb_base.clone() };
        let vis = if !fto && cfg.vision_config.is_some() { Some(Qwen3VLVisionModel::new(cfg.vision_config.as_ref().unwrap().clone(), vb_base.pp("visual"))?) } else { None };
        
        let (nh, hd, nkv, hs, is, eps, theta, mrope, nl, vs) = if let Some(tc) = &cfg.text_config {
            (tc.num_attention_heads, tc.head_dim, tc.num_key_value_heads, tc.hidden_size, tc.intermediate_size, tc.rms_norm_eps, tc.rope_theta, tc.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), tc.num_hidden_layers, tc.vocab_size)
        } else {
            (cfg.hidden_size.unwrap_or(1024)/128, 128, cfg.hidden_size.unwrap_or(1024)/128, cfg.hidden_size.unwrap_or(1024), cfg.hidden_size.unwrap_or(1024)*3, 1e-6, 1000000.0, vec![], 28, 151936)
        };

        let mut blks = vec![]; for i in 0..(if bo { 1 } else { nl }) {
            let vbl = vb_lm.pp("layers").pp(&i.to_string());
            let q_proj = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn.q_proj"), "weight", map_r)?;
            let k_proj = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn.k_proj"), "weight", map_r)?;
            let v_proj = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn.v_proj"), "weight", map_r)?;
            let o_proj = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("self_attn.o_proj"), "weight", map_r)?;
            let qnw = vbl.get(hd, "self_attn.q_norm.weight")?; let knw = vbl.get(hd, "self_attn.k_norm.weight")?;
            let attn = Qwen3VLTextAttention { q_proj, k_proj, v_proj, o_proj, q_norm: RmsNorm::new(qnw, eps), k_norm: RmsNorm::new(knw, eps), num_attention_heads: nh, num_key_value_heads: nkv, num_kv_groups: nh/nkv, head_dim: hd, scaling: 1.0/(hd as f64).sqrt(), kv_cache: None };
            let g = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp.gate_proj"), "weight", map_r)?;
            let u = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp.up_proj"), "weight", map_r)?;
            let d = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vbl.pp("mlp.down_proj"), "weight", map_r)?;
            blks.push(Some(Qwen3VLTextDecoderLayer { attn, mlp: QGateUpDownMLP { g, u, d, act: Activation::Silu }, in_ln: RmsNorm::new(vbl.get(hs, "input_layernorm.weight")?, eps), post_ln: RmsNorm::new(vbl.get(hs, "post_attention_layernorm.weight")?, eps) }));
        }
        let ew = vb_lm.get((vs, hs), "embed_tokens.weight")?;
        let lm = Qwen3VLTextModel { embed: Embedding::new(ew, hs), layers: blks, norm: RmsNorm::new(vb_lm.get(hs, "norm.weight")?, eps), rotary: Qwen3VLTextRotaryEmbedding::new(hd, theta), mrope, is_baking: bo };
        let head = crate::models::qwen3vl::quantized_model::get_qlinear_safe(vb_lm.pp("lm_head"), "weight", map_r)?;
        Ok(Self { config: cfg, visual: vis, language_model: lm, lm_head: head, is_baking: bo })
    }
    pub fn forward(&mut self, ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, vthw: Option<&Tensor>, _cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (_bs, sl) = ids.dims2()?; let mut embs = self.language_model.embed.forward(&ids.to_dtype(DType::U32)?.contiguous()?)?;
        if let Some(ref mut vm) = &mut self.visual { if let (Some(p), Some(t)) = (pv, thw) { let (ie, _) = vm.forward(p, t)?; let m = ids.broadcast_eq(&Tensor::new(vec![self.config.image_token_id.unwrap_or(0) as u32], ids.device())?)?.to_dtype(DType::U32)?; embs = masked_scatter_dim0(&embs, &ie.to_device(ids.device())?, &m)?; } }
        let (pids, _) = Qwen3VLModel::get_rope_index_static(&self.config, ids, thw, vthw, None)?;
        let out = self.language_model.forward(&embs, offset, Some(&pids))?;
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
    fn get_rope_index_static(cfg: &Qwen3VLConfig, ids: &Tensor, thw: Option<&Tensor>, vthw: Option<&Tensor>, m: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let ms = cfg.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let (tid, vtid, vsid) = (cfg.image_token_id.unwrap_or(0) as u32, cfg.video_token_id.unwrap_or(0) as u32, cfg.vision_start_token_id.unwrap_or(0) as u32);
        let mut deltas = vec![]; let mut pos_ids = Tensor::ones((3, ids.dim(0)?, ids.dim(1)?), DType::U32, ids.device())?;
        let (mut ii, mut vi) = (0, 0);
        for i in 0..ids.dim(0)? {
            let mut cur_ids = ids.i(i)?; if let Some(mask) = m { if mask.i(i)?.sum_all()?.to_scalar::<u32>()? != mask.dim(1)? as u32 { cur_ids = cur_ids.gather(&nonzero_index(&mask.i(i)?)?, 0)?; } }
            let (mut ts, mut te) = (0, 0); let mut list: Vec<Tensor> = vec![];
            if let Ok(vidx) = get_vision_next_indices(&cur_ids, vsid) {
                let vtoks = cur_ids.gather(&vidx, 0)?.to_vec1::<u32>()?; let vidx_v = vidx.to_vec1::<u32>()?;
                for (j, &t) in vtoks.iter().enumerate() {
                    let thw_v = if t == tid { let v = thw.unwrap().i(ii)?.to_vec1::<u32>()?; ii += 1; te = vidx_v[j]; v }
                    else if t == vtid { let v = vthw.as_ref().unwrap().i(vi)?.to_vec1::<u32>()?; vi += 1; te = vidx_v[j]; v } else { continue; };
                    let (gt, gh, gw) = (thw_v[0], thw_v[1] / ms as u32, thw_v[2] / ms as u32);
                    let tlen = te - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 };
                    list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?);
                    let base = sidx + tlen;
                    let ti = Tensor::arange(base, base + gt, ids.device())?.to_dtype(DType::U32)?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, (gh * gw) as usize))?.flatten_all()?;
                    let hi = Tensor::arange(base, base + gh, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    let wi = Tensor::arange(base, base + gw, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    list.push(Tensor::stack(&[ti, hi, wi], 0)?); ts = te + gt * gh * gw;
                }
            }
            if ts < cur_ids.dim(0)? as u32 { let tlen = cur_ids.dim(0)? as u32 - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 }; list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?); }
            let lp = Tensor::cat(&list, 1)?.reshape((3, 1, ()))?; pos_ids = pos_ids.slice_assign(&[(0..3), (i..i + 1), (0..ids.dim(1)?)], &lp)?;
            deltas.push(lp.max_all()?.to_scalar::<u32>()? as i64 + 1 - cur_ids.dim(0)? as i64);
        }
        let mut d_t = Tensor::new(deltas, ids.device())?; if d_t.rank() == 1 { d_t = d_t.unsqueeze(0)?; } Ok((pos_ids.contiguous()?, d_t))
    }
    pub fn load_kv_cache(&mut self, p: &std::path::Path, d: &Device) -> Result<()> {
        let mut fkv: Option<(Tensor, Tensor)> = None; let dt = if d.is_cpu() { DType::F32 } else { DType::BF16 };
        for (i, l_opt) in self.language_model.layers.iter_mut().enumerate() {
            if let Some(l) = l_opt {
                let p_f = p.join(format!("layer_{}_kv.safetensors", i));
                let (mut k, mut v) = if p_f.exists() { let m = candle_core::safetensors::load(p_f, d)?; (m.get("k").unwrap().to_dtype(dt)?, m.get("v").unwrap().to_dtype(dt)?) }
                else if let Some((ref fk, ref fv)) = fkv { (fk.clone(), fv.clone()) } else { continue; };
                if k.dim(1)? != l.attn.num_key_value_heads { if l.attn.num_key_value_heads % k.dim(1)? == 0 { let r = l.attn.num_key_value_heads / k.dim(1)?; k = k.repeat((1, r, 1, 1))?; v = v.repeat((1, r, 1, 1))?; } else { k = k.narrow(1, 0, l.attn.num_key_value_heads)?; v = v.narrow(1, 0, l.attn.num_key_value_heads)?; } }
                if k.dim(3)? != l.attn.head_dim { k = apply_bridge_static(&k, l.attn.head_dim)?; v = apply_bridge_static(&v, l.attn.head_dim)?; }
                if fkv.is_none() { fkv = Some((k.clone(), v.clone())); } l.attn.kv_cache = Some((k, v));
            }
        } Ok(())
    }
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
