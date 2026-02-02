use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{Activation, Embedding, Init, Linear, VarBuilder};
use std::collections::HashMap;

use crate::{
    models::{
        common::{GateUpDownMLP, NaiveAttention, TwoLinearMLP, eager_attention_forward, RmsNorm, LayerNorm},
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig},
            quantized_model::QLinear,
        },
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

// --- Safe Loading Helpers ---

fn safe_probe(v: &VarBuilder, p: &str, map: Option<&HashMap<String, Tensor>>) -> bool {
    let prefix = v.prefix();
    let path = if prefix.is_empty() { p.to_string() } 
    else if prefix.ends_with('.') || p.is_empty() { format!("{}{}", prefix, p) } 
    else { format!("{}.{}", prefix, p) };

    if let Some(m) = map { return m.contains_key(&path); }
    v.get_with_hints((1,), p, Init::Const(0.)).is_ok()
}

fn safe_get(v: &VarBuilder, p: &str, map: Option<&HashMap<String, Tensor>>) -> Result<Tensor> {
    let prefix = v.prefix();
    let path = if prefix.is_empty() { p.to_string() } 
    else if prefix.ends_with('.') || p.is_empty() { format!("{}{}", prefix, p) } 
    else { format!("{}.{}", prefix, p) };

    if let Some(m) = map {
        if let Some(t) = m.get(&path) { return Ok(t.clone()); }
        return Err(anyhow!("Missing tensor: {}", path));
    }
    match v.get_with_hints((1,), p, Init::Const(0.)) {
        Ok(t) => Ok(t),
        Err(e) => {
            let es = e.to_string();
            if es.contains("shape mismatch") {
                if let Some(s) = es.find("got: [") {
                    let r = &es[s + 6..];
                    if let Some(e) = r.find(']') {
                        let dims: Vec<usize> = r[..e].split(',').map(|s| s.trim().parse::<usize>()).filter_map(|r| r.ok()).collect();
                        if !dims.is_empty() { return Ok(v.get(dims, p)?); }
                    }
                }
            }
            Err(anyhow!("Load failed '{}': {}", p, e))
        }
    }
}

fn get_qlinear_safe(vb: VarBuilder, name: &str, map: Option<&HashMap<String, Tensor>>) -> Result<QLinear> {
    let (p_key, s_key, sh_key) = if name.is_empty() { ("packed".to_string(), "scales".to_string(), "shape".to_string()) } 
                                 else { (format!("{}.packed", name), format!("{}.scales", name), format!("{}.shape", name)) };
    
    if safe_probe(&vb, &p_key, map) {
        let p = safe_get(&vb, &p_key, map)?;
        let s = safe_get(&vb, &s_key, map)?.to_dtype(DType::F32)?;
        let sh_t = safe_get(&vb, &sh_key, map)?;
        let sh: Vec<usize> = sh_t.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
        let b = if safe_probe(&vb, &format!("{}.bias", name), map) { Some(safe_get(&vb, &format!("{}.bias", name), map)?) } else { None };
        return Ok(QLinear::new(p, s, sh, b, vb.device().clone()));
    }
    let w = safe_get(&vb, if name.is_empty() { "weight" } else { name }, map)?;
    let s = w.dims().to_vec(); let t = s.iter().product::<usize>();
    let b = if safe_probe(&vb, &format!("{}.bias", name), map) { Some(safe_get(&vb, &format!("{}.bias", name), map)?) } else { None };
    Ok(QLinear::new(Tensor::zeros((t/32).max(1), DType::U32, vb.device())?, Tensor::ones((t/32).max(1), DType::F32, vb.device())?, s, b, vb.device().clone()))
}

// --- Vision ---

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchEmbed { conv3d_weight: Tensor, conv3d_bias: Tensor }
impl Qwen3VLVisionPatchEmbed {
    pub fn new(_cfg: &Qwen3VLVisionConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let (wn, bn) = if safe_probe(&vb, "proj.weight", map) { ("proj.weight", "proj.bias") } else { ("weight", "bias") };
        let w = safe_get(&vb, wn, map)?.flatten(1, 4)?.t()?;
        let b = safe_get(&vb, bn, map)?.unsqueeze(0)?;
        Ok(Self { conv3d_weight: w, conv3d_bias: b })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.matmul(&self.conv3d_weight)?.broadcast_add(&self.conv3d_bias)?) }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.conv3d_weight = self.conv3d_weight.to_device(d)?; self.conv3d_bias = self.conv3d_bias.to_device(d)?; Ok(()) }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchMerger { hidden_size: usize, use_postshuffle_norm: bool, norm: LayerNorm, linear_fc1: Linear, act_fn: Activation, linear_fc2: Linear }
impl Qwen3VLVisionPatchMerger {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder, upn: bool, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let hs = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let ln_w = safe_get(&vb.pp("norm"), "weight", map)?; let ln_b = safe_get(&vb.pp("norm"), "bias", map)?;
        let fc1_w = safe_get(&vb.pp("linear_fc1"), "weight", map)?; let fc1_b = safe_get(&vb.pp("linear_fc1"), "bias", map)?;
        let fc2_w = safe_get(&vb.pp("linear_fc2"), "weight", map)?; let fc2_b = safe_get(&vb.pp("linear_fc2"), "bias", map)?;
        Ok(Self { hidden_size: hs, use_postshuffle_norm: upn, norm: LayerNorm::new(ln_w, ln_b, 1e-6), linear_fc1: Linear::new(fc1_w, Some(fc1_b)), act_fn: Activation::Gelu, linear_fc2: Linear::new(fc2_w, Some(fc2_b)) })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if !self.use_postshuffle_norm { self.norm.forward(x)? } else { x.clone() };
        let x = self.linear_fc1.forward(&x)?; let x = self.act_fn.forward(&x)?;
        let x = self.linear_fc2.forward(&x)?;
        if self.use_postshuffle_norm { Ok(self.norm.forward(&x)?) } else { Ok(x) }
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { 
        let nw = self.norm.weight.to_device(d)?; let nb = self.norm.bias.to_device(d)?; self.norm = LayerNorm::new(nw, nb, 1e-6);
        let f1w = self.linear_fc1.weight().to_device(d)?; let f1b = self.linear_fc1.bias().as_ref().unwrap().to_device(d)?; self.linear_fc1 = Linear::new(f1w, Some(f1b));
        let f2w = self.linear_fc2.weight().to_device(d)?; let f2b = self.linear_fc2.bias().as_ref().unwrap().to_device(d)?; self.linear_fc2 = Linear::new(f2w, Some(f2b));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionBlock { attn: NaiveAttention, mlp: TwoLinearMLP, norm1: LayerNorm, norm2: LayerNorm }
impl Qwen3VLVisionBlock {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let n1w = safe_get(&vb.pp("norm1"), "weight", map)?; let n1b = safe_get(&vb.pp("norm1"), "bias", map)?;
        let n2w = safe_get(&vb.pp("norm2"), "weight", map)?; let n2b = safe_get(&vb.pp("norm2"), "bias", map)?;
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size);
        let head_dim = embed_dim / cfg.num_heads;
        Ok(Self { 
            attn: NaiveAttention::new(vb.pp("attn"), cfg.hidden_size, cfg.num_heads, cfg.num_heads, Some(head_dim), true, Some("o_proj"))?,
            mlp: TwoLinearMLP::new(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size, Activation::Gelu, true, "linear1", "linear2")?,
            norm1: LayerNorm::new(n1w, n1b, 1e-6), norm2: LayerNorm::new(n2w, n2b, 1e-6)
        })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let r = x.clone(); let x = self.attn.forward(&self.norm1.forward(x)?, Some(cos), Some(sin), None, false)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.norm2.forward(&x)?)?; Ok(r.add(&x)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.attn.to_device(d)?; self.mlp.to_device(d)?;
        let n1w = self.norm1.weight.to_device(d)?; let n1b = self.norm1.bias.to_device(d)?; self.norm1 = LayerNorm::new(n1w, n1b, 1e-6);
        let n2w = self.norm2.weight.to_device(d)?; let n2b = self.norm2.bias.to_device(d)?; self.norm2 = LayerNorm::new(n2w, n2b, 1e-6);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionModel { patch_embed: Qwen3VLVisionPatchEmbed, blocks: Vec<Qwen3VLVisionBlock>, merger: Qwen3VLVisionPatchMerger, rotary: Qwen2_5VisionRotaryEmbedding }
impl Qwen3VLVisionModel {
    pub fn new(cfg: Qwen3VLVisionConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let pe = Qwen3VLVisionPatchEmbed::new(&cfg, vb.pp("patch_embed"), map)?;
        let mut blks = vec![]; for i in 0..cfg.depth { blks.push(Qwen3VLVisionBlock::new(&cfg, vb.pp("blocks").pp(&i.to_string()), map)?); }
        let me = Qwen3VLVisionPatchMerger::new(&cfg, vb.pp("merger"), true, map)?;
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size);
        Ok(Self { patch_embed: pe, blocks: blks, merger: me, rotary: Qwen2_5VisionRotaryEmbedding::new(embed_dim / cfg.num_heads, None) })
    }
    pub fn forward(&mut self, x: &Tensor, thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let x = self.patch_embed.forward(x)?; 
        let cos_sin = self.rotary.forward(thw.dim(0)?, x.device())?;
        let (cos, sin) = (cos_sin.clone(), cos_sin); 
        let mut x = x; let mut ds = vec![]; for b in &mut self.blocks { x = b.forward(&x, &cos, &sin)?; ds.push(x.clone()); }
        Ok((self.merger.forward(&x)?, ds))
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.patch_embed.to_device(d)?; for b in &mut self.blocks { b.to_device(d)?; } self.merger.to_device(d)?; Ok(()) }
}

// --- Text ---

#[derive(Debug, Clone)]
pub struct Qwen3VLTextAttention {
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear,
    pub q_norm: RmsNorm, pub k_norm: RmsNorm, pub num_attention_heads: usize, pub num_key_value_heads: usize, pub num_kv_groups: usize,
    pub head_dim: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
}
impl Qwen3VLTextAttention {
    pub fn new(cfg: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let (nh, hd, nkv) = (cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads);
        let qnw = safe_get(&vb.pp("q_norm"), "weight", map)?; let knw = safe_get(&vb.pp("k_norm"), "weight", map)?;
        Ok(Self { q_proj: get_qlinear_safe(vb.pp("q_proj"), "weight", map)?, k_proj: get_qlinear_safe(vb.pp("k_proj"), "weight", map)?, v_proj: get_qlinear_safe(vb.pp("v_proj"), "weight", map)?, o_proj: get_qlinear_safe(vb.pp("o_proj"), "weight", map)?, q_norm: RmsNorm::new(qnw, cfg.rms_norm_eps), k_norm: RmsNorm::new(knw, cfg.rms_norm_eps), num_attention_heads: nh, num_key_value_heads: nkv, num_kv_groups: nh/nkv, head_dim: hd, scaling: 1f64 / (hd as f64).sqrt(), kv_cache: None })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bs, ql, _) = x.dims3()?;
        let qs = self.q_norm.forward(&self.q_proj.forward(x)?.reshape((bs, ql, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?;
        let ks = self.k_norm.forward(&self.k_proj.forward(x)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?;
        let vs = self.v_proj.forward(x)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (qs, ks) = apply_rotary_pos_emb(&qs, &ks, cos, sin, false)?;
        let (ks, vs) = match &self.kv_cache { None => (ks, vs), Some((pk, pv)) => (Tensor::cat(&[pk, &ks], 2)?, Tensor::cat(&[pv, &vs], 2)?) };
        self.kv_cache = Some((ks.clone(), vs.clone()));
        let out = eager_attention_forward(&qs, &ks, &vs, Some(self.num_kv_groups), mask, self.scaling)?;
        Ok(self.o_proj.forward(&out.reshape((bs, ql, ()))?)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let qw = self.q_norm.weight.to_device(d)?; self.q_norm = RmsNorm::new(qw, self.q_norm.eps);
        let kw = self.k_norm.weight.to_device(d)?; self.k_norm = RmsNorm::new(kw, self.k_norm.eps);
        if let Some((k, v)) = &self.kv_cache { self.kv_cache = Some((k.to_device(d)?, v.to_device(d)?)); } Ok(())
    }
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

#[derive(Debug, Clone)]
pub struct Qwen3VLTextDecoderLayer { pub attn: Qwen3VLTextAttention, pub mlp: QGateUpDownMLP, pub in_ln: RmsNorm, pub post_ln: RmsNorm }
impl Qwen3VLTextDecoderLayer {
    pub fn new(cfg: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let inw = safe_get(&vb.pp("input_layernorm"), "weight", map)?;
        let pnw = safe_get(&vb.pp("post_attention_layernorm"), "weight", map)?;
        Ok(Self { attn: Qwen3VLTextAttention::new(cfg.clone(), vb.pp("self_attn"), map)?, mlp: QGateUpDownMLP::new(vb.pp("mlp"), cfg.hidden_act, map)?, in_ln: RmsNorm::new(inw, cfg.rms_norm_eps), post_ln: RmsNorm::new(pnw, cfg.rms_norm_eps) })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = x.clone(); let x = self.attn.forward(&self.in_ln.forward(x)?, cos, sin, mask)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.post_ln.forward(&x)?)?; Ok(r.add(&x)?)
    }
    pub fn clear_kv_cache(&mut self) { self.attn.clear_kv_cache(); }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.attn.to_device(d)?; let iw = self.in_ln.weight.to_device(d)?; self.in_ln = RmsNorm::new(iw, self.in_ln.eps);
        let pw = self.post_ln.weight.to_device(d)?; self.post_ln = RmsNorm::new(pw, self.post_ln.eps); Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextModel { pub embed: Embedding, pub layers: Vec<Option<Qwen3VLTextDecoderLayer>>, pub norm: RmsNorm, pub rotary: Qwen3VLTextRotaryEmbedding, pub mrope: Vec<usize>, pub is_baking: bool }
impl Qwen3VLTextModel {
    pub fn new(cfg: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let ew = if safe_probe(&vb.pp("embed_tokens"), "weight", map) { safe_get(&vb.pp("embed_tokens"), "weight", map)? }
        else { Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, vb.device())? };
        let mut blks = vec![]; let vbl = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let p_ln = format!("layers.{}.input_layernorm.weight", i);
            if safe_probe(&vb, &p_ln, map) || safe_probe(&vbl.pp(&i.to_string()).pp("input_layernorm"), "weight", map) {
                blks.push(Some(Qwen3VLTextDecoderLayer::new(cfg.clone(), vbl.pp(&i.to_string()), map)?));
            } else { blks.push(None); }
        }
        let nw = if safe_probe(&vb.pp("norm"), "weight", map) { safe_get(&vb.pp("norm"), "weight", map)? }
        else if safe_probe(&vb, &format!("output_norm.weight"), map) { safe_get(&vb, "output_norm.weight", map)? }
        else { Tensor::ones(cfg.hidden_size, DType::F32, vb.device())? };
        Ok(Self { embed: Embedding::new(ew, cfg.hidden_size), layers: blks, norm: RmsNorm::new(nw, cfg.rms_norm_eps), rotary: Qwen3VLTextRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta), mrope: cfg.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), is_baking: false })
    }
    pub fn forward(&mut self, x: &Tensor, offset: usize, pids: Option<&Tensor>, vm: Option<&Tensor>, ds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let pi = match pids { Some(ids) => ids.clone(), None => Tensor::arange(offset as u32, (sl + offset) as u32, x.device())?.to_dtype(DType::U32)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))? };
        let (cos, sin) = self.rotary.forward(&pi, x.dtype(), self.mrope.clone())?;
        
        let mask = if sl <= 1 { None } else {
            let q_idx = Tensor::arange(0u32, sl as u32, x.device())?.reshape((1, 1, sl, 1))?;
            let kv_idx = Tensor::arange(0u32, (offset + sl) as u32, x.device())?.reshape((1, 1, 1, offset + sl))?;
            let m = kv_idx.broadcast_gt(&(q_idx.broadcast_add(&Tensor::new(offset as u32, x.device())?)?))?;
            let m = m.to_dtype(x.dtype())?.affine(-65504.0, 0.0)?;
            Some(m)
        };
        
        let mut x = x.clone(); let limit = if self.is_baking { 1 } else { self.layers.len() };
        for (i, l_opt) in self.layers.iter_mut().enumerate().take(limit) {
            if let Some(l) = l_opt {
                x = l.forward(&x, &cos, &sin, mask.as_ref())?;
                if let (Some(m), Some(dse)) = (vm, ds.as_ref()) { if i < dse.len() { x = mask_index_add(&x.squeeze(0)?, &m.squeeze(0)?, &dse[i])?.unsqueeze(0)?; } }
            }
        }
        Ok(x.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for l in &mut self.layers { if let Some(li) = l { li.clear_kv_cache() } } }
    pub fn get_kv_len(&self) -> usize {
        for layer in &self.layers { if let Some(l) = layer { if let Some((k, _)) = &l.attn.kv_cache { return k.dim(2).unwrap_or(0); } } }
        0
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let ew = self.embed.embeddings().to_device(d)?; self.embed = Embedding::new(ew, self.embed.hidden_size());
        for l in &mut self.layers { if let Some(li) = l { li.to_device(d)?; } }
        let nw = self.norm.weight.to_device(d)?; self.norm = RmsNorm::new(nw, self.norm.eps); Ok(())
    }
}

// --- Main ---

#[derive(Debug, Clone)]
pub struct Qwen3VLModel { pub config: Qwen3VLConfig, pub visual: Option<Qwen3VLVisionModel>, pub language_model: Qwen3VLTextModel, pub lm_head: QLinear, pub is_baking: bool }
impl Qwen3VLModel {
    pub fn set_baking(&mut self, b: bool) { self.is_baking = b; self.language_model.is_baking = b; }
    pub fn new(c: Qwen3VLConfig, vb: VarBuilder) -> Result<Self> { Self::new_ext(c, vb, None, false, false) }
    pub fn new_ext(cfg: Qwen3VLConfig, vb: VarBuilder, map: Option<HashMap<String, Tensor>>, fto: bool, bo: bool) -> Result<Self> {
        let map_r = map.as_ref();
        let vis = if !fto && cfg.vision_config.is_some() {
            let v_cfg = cfg.vision_config.as_ref().unwrap();
            let vb_v = if safe_probe(&vb, "model.visual.patch_embed.proj.weight", map_r) { Some(vb.pp("model").pp("visual")) }
            else if safe_probe(&vb, "visual.patch_embed.proj.weight", map_r) { Some(vb.pp("visual")) } else { None };
            if let Some(v) = vb_v { Some(Qwen3VLVisionModel::new(v_cfg.clone(), v, map_r)?) } else { None }
        } else { None };
        let t_cfg = cfg.text_config.clone().ok_or(anyhow!("Missing text_config"))?;
        let vb_lm = if safe_probe(&vb, "model.language_model.layers.0.input_layernorm.weight", map_r) { vb.pp("model").pp("language_model") }
        else if safe_probe(&vb, "model.layers.0.input_layernorm.weight", map_r) { vb.pp("model") }
        else { vb.pp("model").pp("language_model") };
        let lm = Qwen3VLTextModel::new(t_cfg, vb_lm, map_r)?;
        let head = if safe_probe(&vb, "model.language_model.lm_head.weight", map_r) { get_qlinear_safe(vb.pp("model").pp("language_model").pp("lm_head"), "weight", map_r)? }
        else if safe_probe(&vb, "model.lm_head.weight", map_r) { get_qlinear_safe(vb.pp("model").pp("lm_head"), "weight", map_r)? }
        else if bo {
            let tc = cfg.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
            let shape = vec![tc.vocab_size, tc.hidden_size];
            QLinear::new(Tensor::zeros((1,), DType::U32, vb.device())?, Tensor::ones((1,), DType::F32, vb.device())?, shape, None, vb.device().clone())
        } else { get_qlinear_safe(vb.pp("model").pp("language_model").pp("lm_head"), "weight", map_r)? };
        let mut model = Self { config: cfg, visual: vis, language_model: lm, lm_head: head, is_baking: bo }; model.set_baking(bo); Ok(model)
    }
    pub fn forward(&mut self, ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, vthw: Option<&Tensor>, _cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let mut embs = self.language_model.embed.forward(ids)?;
        let (mut im, mut dse) = (None, None);
        if let Some(ref mut vm) = &mut self.visual { if let (Some(p), Some(t)) = (pv, thw) {
            let (ie, ds) = vm.forward(p, t)?; let vm_m = ids.broadcast_eq(&Tensor::new(vec![self.config.image_token_id.unwrap_or(0) as u32], ids.device())?)?.to_dtype(DType::U32)?;
            embs = masked_scatter_dim0(&embs, &Tensor::cat(&[ie], 0)?, &vm_m)?; im = Some(vm_m); dse = Some(ds);
        } }
        let (pids, _) = Qwen3VLModel::get_rope_index_static(&self.config, ids, thw, vthw, None)?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), im.as_ref(), dse)?;
        if self.is_baking { return Ok(out.narrow(1, out.dim(1)? - 1, 1)?); }
        Ok(self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?)
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { if let Some(v) = &mut self.visual { v.to_device(d)?; } self.language_model.to_device(d)?; Ok(()) }
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
        }
        Ok(())
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
        }
        Ok(())
    }
    fn get_rope_index_static(cfg: &Qwen3VLConfig, ids: &Tensor, thw: Option<&Tensor>, vthw: Option<&Tensor>, m: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let ms = cfg.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let (tid, vtid, vsid) = (cfg.image_token_id.unwrap_or(0), cfg.video_token_id.unwrap_or(0), cfg.vision_start_token_id.unwrap_or(0));
        let mut deltas = vec![]; let mut pos_ids = Tensor::ones((3, ids.dim(0)?, ids.dim(1)?), DType::U32, ids.device())?;
        let (mut ii, mut vi) = (0, 0);
        for i in 0..ids.dim(0)? {
            let mut cur_ids = ids.i(i)?; if let Some(mask) = m { if mask.i(i)?.sum_all()?.to_scalar::<u32>()? != mask.dim(1)? as u32 { cur_ids = cur_ids.gather(&nonzero_index(&mask.i(i)?)?, 0)?; } }
            let (mut ts, mut te) = (0, 0); let mut list: Vec<Tensor> = vec![];
            if let Ok(vidx) = get_vision_next_indices(&cur_ids, vsid as u32) {
                let vtoks = cur_ids.gather(&vidx, 0)?.to_vec1::<u32>()?; let vidx_v = vidx.to_vec1::<u32>()?;
                for (j, &t) in vtoks.iter().enumerate() {
                    let thw_v = if t == tid as u32 { let v = thw.unwrap().i(ii)?.to_vec1::<u32>()?; ii += 1; te = vidx_v[j]; v }
                    else if t == vtid as u32 { let v = vthw.as_ref().unwrap().i(vi)?.to_vec1::<u32>()?; vi += 1; te = vidx_v[j]; v } else { continue; };
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
    pub fn device(&self) -> &Device { self.language_model.embed.embeddings().device() }
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
