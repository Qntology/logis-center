use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor, Module};
use candle_nn::{
    Activation, Embedding, Init, Linear, VarBuilder,
};
use std::collections::HashMap;

use crate::{
    models::{
        common::{TwoLinearMLP, eager_attention_forward, get_layer_norm, RmsNorm, LayerNorm, rms_norm as get_rms_norm},
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig},
            quantized_model::QLinear,
        },
    },
    position_embed::rope::{
        Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
        apply_rotary_pos_emb_vision,
    },
    utils::tensor_utils::{
        linspace, mask_index_add, masked_scatter_dim0,
        nonzero_index, prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
        get_vision_next_indices,
    },
};

// [SAFE-PROBE] Helper to check tensor existence via HashMap without triggering CUDA kernels
fn probe(v: &VarBuilder, p: &str, map: Option<&HashMap<String, Tensor>>) -> bool {
    if let Some(m) = map {
        let full_path = if v.prefix().is_empty() { p.to_string() } else { format!("{}{}", v.prefix(), p) };
        return m.contains_key(&full_path);
    }
    match v.get_with_hints((1,), p, Init::Const(0.)) {
        Ok(_) => true,
        Err(e) => {
            let s = e.to_string().to_lowercase();
            s.contains("shape mismatch") || s.contains("dtype mismatch")
        }
    }
}

// [SAFE-GET] Helper to load tensor with dynamic shape, avoids DriverError on missing keys
fn get_any(v: &VarBuilder, p: &str, map: Option<&HashMap<String, Tensor>>) -> Result<Tensor> {
    if let Some(m) = map {
        let full_path = if v.prefix().is_empty() { p.to_string() } else { format!("{}{}", v.prefix(), p) };
        if let Some(t) = m.get(&full_path) { return Ok(t.clone()); }
    }
    match v.get_with_hints((1,), p, Init::Const(0.)) {
        Ok(t) => Ok(t),
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("shape mismatch") {
                if let Some(start) = err_str.find("got: [") {
                    let rest = &err_str[start + 6..];
                    if let Some(end) = rest.find(']') {
                        let dims: Vec<usize> = rest[..end].split(',').map(|s| s.trim().parse::<usize>()).filter_map(|r| r.ok()).collect();
                        if !dims.is_empty() { return Ok(v.get(dims, p)?); }
                    }
                }
            }
            Err(anyhow!("Failed to load tensor '{}': {}", p, e))
        }
    }
}

fn get_qlinear_from_vb(vb: VarBuilder, name: &str, map: Option<&HashMap<String, Tensor>>) -> Result<QLinear> {
    let prefix = if name.is_empty() { "".to_string() } else { format!("{}.", name) };
    let (p_n, s_n, sh_n, b_n) = (format!("{}packed", prefix), format!("{}scales", prefix), format!("{}shape", prefix), format!("{}bias", prefix));
    if probe(&vb, &p_n, map) {
        let p = get_any(&vb, &p_n, map)?; let s_r = get_any(&vb, &s_n, map)?; let sh_t = get_any(&vb, &sh_n, map)?;
        let s = s_r.to_dtype(DType::F32)?;
        let sh: Vec<usize> = sh_t.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
        let b = if probe(&vb, &b_n, map) { Some(get_any(&vb, &b_n, map)?) } else { None };
        return Ok(QLinear::new(p, s, sh, b, vb.device().clone()));
    }
    let w_n = if name.is_empty() { "weight".to_string() } else { name.to_string() };
    let w = get_any(&vb, &w_n, map)?; let s = w.dims().to_vec(); let t = s.iter().product::<usize>();
    let sc = Tensor::ones((t / 32).max(1), DType::F32, vb.device())?;
    let pk = Tensor::zeros((t / 32).max(1), DType::U32, vb.device())?;
    let b = if probe(&vb, &b_n, map) { Some(get_any(&vb, &b_n, map)?) } else { None };
    Ok(QLinear::new(pk, sc, s, b, vb.device().clone()))
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchEmbed { conv3d_weight: Tensor, conv3d_bias: Tensor }
impl Qwen3VLVisionPatchEmbed {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let (w_n, b_n) = if vb.get_with_hints((1,), "proj.weight", Init::Const(0.)).is_ok() { ("proj.weight", "proj.bias") }
        else if vb.get_with_hints((1,), "weight", Init::Const(0.)).is_ok() { ("weight", "bias") }
        else { ("proj.weight", "proj.bias") };
        let w = vb.get_with_hints((cfg.embed_dim.unwrap_or(cfg.hidden_size), cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size), w_n, Init::Const(1.))?.flatten(1, 4)?.t()?;
        let b = vb.get_with_hints((cfg.embed_dim.unwrap_or(cfg.hidden_size),), b_n, Init::Const(0.))?.unsqueeze(0)?;
        Ok(Self { conv3d_weight: w, conv3d_bias: b })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { self.conv3d_weight = self.conv3d_weight.to_device(d)?; self.conv3d_bias = self.conv3d_bias.to_device(d)?; Ok(()) }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.matmul(&self.conv3d_weight)?.broadcast_add(&self.conv3d_bias)?) }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchMerger { hidden_size: usize, use_postshuffle_norm: bool, norm: LayerNorm, l1: Linear, act: Activation, l2: Linear }
impl Qwen3VLVisionPatchMerger {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder, upn: bool) -> Result<Self> {
        let hs = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let ns = if upn { hs } else { cfg.hidden_size };
        let (f1, f2, n) = if vb.pp("linear_fc1").get_with_hints((1,), "weight", Init::Const(0.)).is_ok() { ("linear_fc1", "linear_fc2", "norm") }
        else { ("0", "2", "norm") };
        let norm = get_layer_norm(vb.pp(n), 1e-6, ns)?;
        let w1 = vb.pp(f1).get((hs, hs), "weight")?; let b1 = vb.pp(f1).get(hs, "bias").ok();
        let w2 = vb.pp(f2).get((cfg.out_hidden_size, hs), "weight")?; let b2 = vb.pp(f2).get(cfg.out_hidden_size, "bias").ok();
        Ok(Self { hidden_size: hs, use_postshuffle_norm: upn, norm, l1: Linear::new(w1, b1), act: Activation::Gelu, l2: Linear::new(w2, b2) })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let nw = self.norm.weight.to_device(d)?; let nb = self.norm.bias.to_device(d)?; self.norm = LayerNorm::new(nw, nb, 1e-6);
        let w1 = self.l1.weight().to_device(d)?; let b1 = self.l1.bias().map(|b| b.to_device(d)).transpose()?; self.l1 = Linear::new(w1, b1);
        let w2 = self.l2.weight().to_device(d)?; let b2 = self.l2.bias().map(|b| b.to_device(d)).transpose()?; self.l2 = Linear::new(w2, b2); Ok(())
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.use_postshuffle_norm { x.reshape(((), self.hidden_size))? } else { x.clone() };
        let x = self.norm.forward(&x)?.reshape(((), self.hidden_size))?;
        Ok(self.l2.forward(&self.act.forward(&self.l1.forward(&x)?)?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionAttention { num_heads: usize, qkv: Linear, proj: Linear, scaling: f64 }
impl Qwen3VLVisionAttention {
    pub fn new(cfg: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hs = cfg.hidden_size; let nh = cfg.num_heads; let hd = hs / nh;
        let qkv_w = vb.get((hs * 3, hs), "qkv.weight")?; let qkv_b = vb.get(hs * 3, "qkv.bias").ok();
        let p_w = vb.get((hs, hs), "proj.weight")?; let p_b = vb.get(hs, "proj.bias").ok();
        Ok(Self { num_heads: nh, qkv: Linear::new(qkv_w, qkv_b), proj: Linear::new(p_w, p_b), scaling: 1.0 / (hd as f64).sqrt() })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let w1 = self.qkv.weight().to_device(d)?; let b1 = self.qkv.bias().map(|b| b.to_device(d)).transpose()?; self.qkv = Linear::new(w1, b1);
        let w2 = self.proj.weight().to_device(d)?; let b2 = self.proj.bias().map(|b| b.to_device(d)).transpose()?; self.proj = Linear::new(w2, b2); Ok(())
    }
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, cu: &Tensor) -> Result<Tensor> {
        let sl = x.dim(0)?; let qkv = x.apply(&self.qkv)?.reshape((sl, 3, self.num_heads, ()))?.permute((1, 0, 2, 3))?;
        let (q, k, v) = (qkv.i(0)?.contiguous()?, qkv.i(1)?.contiguous()?, qkv.i(2)?.contiguous()?);
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;
        let (q, k, v) = (q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, v.transpose(0, 1)?.unsqueeze(0)?.contiguous()?);
        let lens = cu.i(1..)?.sub(&cu.i(..cu.dim(0)?-1)?)?.to_vec1::<u32>()?;
        let qs = split_tensor(&q, &lens.iter().map(|&l| l as usize).collect::<Vec<_>>(), 2)?;
        let ks = split_tensor(&k, &lens.iter().map(|&l| l as usize).collect::<Vec<_>>(), 2)?;
        let vs = split_tensor(&v, &lens.iter().map(|&l| l as usize).collect::<Vec<_>>(), 2)?;
        let mut outs = vec![];
        for (qi, (ki, vi)) in qs.iter().zip(ks.iter().zip(vs.iter())) { outs.push(eager_attention_forward(qi, ki, vi, None, None, self.scaling)?); }
        Ok(Tensor::cat(&outs, 1)?.reshape((sl, ()))?.contiguous()?.apply(&self.proj)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionBlock { n1: LayerNorm, n2: LayerNorm, attn: Qwen3VLVisionAttention, mlp: TwoLinearMLP }
impl Qwen3VLVisionBlock {
    pub fn new(cfg: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let n1 = get_layer_norm(vb.pp("norm1"), 1e-6, cfg.hidden_size)?;
        let n2 = get_layer_norm(vb.pp("norm2"), 1e-6, cfg.hidden_size)?;
        let attn = Qwen3VLVisionAttention::new(cfg.clone(), vb.pp("attn"))?;
        let mlp = TwoLinearMLP::new(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size, Activation::Gelu, false, "linear_fc1", "linear_fc2")?;
        Ok(Self { n1, n2, attn, mlp })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        let w1 = self.n1.weight.to_device(d)?; let b1 = self.n1.bias.to_device(d)?; self.n1 = LayerNorm::new(w1, b1, 1e-6);
        let w2 = self.n2.weight.to_device(d)?; let b2 = self.n2.bias.to_device(d)?; self.n2 = LayerNorm::new(w2, b2, 1e-6);
        self.attn.to_device(d)?; self.mlp.to_device(d)?; Ok(())
    }
    pub fn forward(&self, x: &Tensor, cu: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let r = x.clone(); let x = self.n1.forward(x)?; let x = self.attn.forward(&x, cos, sin, cu)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.n2.forward(&x)?)?; Ok(r.add(&x)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionModel { 
    pub sm_size: usize, pub patch_embed: Qwen3VLVisionPatchEmbed, pub pos_embed: Embedding, pub num_grid: u32,
    pub rotary: Qwen2_5VisionRotaryEmbedding, pub blocks: Vec<Qwen3VLVisionBlock>, pub merger: Qwen3VLVisionPatchMerger,
    pub ds_idx: Vec<usize>, pub ds_mergers: Vec<Qwen3VLVisionPatchMerger>, pub dtype: DType
}
impl Qwen3VLVisionModel {
    pub fn new(cfg: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let pe = Qwen3VLVisionPatchEmbed::new(&cfg, vb.pp("patch_embed"))?;
        let pw = vb.pp("pos_embed").get((cfg.num_position_embeddings, cfg.hidden_size), "weight")?;
        let mut blocks = vec![]; for i in 0..cfg.depth { blocks.push(Qwen3VLVisionBlock::new(cfg.clone(), vb.pp("blocks").pp(i))?); }
        let ds_mergers = cfg.deepstack_visual_indexes.iter().enumerate().map(|(i, _)| Qwen3VLVisionPatchMerger::new(&cfg, vb.pp("deepstack_merger_list").pp(i), true)).collect::<Result<Vec<_>>>()?;
        Ok(Self { sm_size: cfg.spatial_merge_size, patch_embed: pe, pos_embed: Embedding::new(pw, cfg.hidden_size), num_grid: (cfg.num_position_embeddings as f32).sqrt() as u32, rotary: Qwen2_5VisionRotaryEmbedding::new(cfg.hidden_size/cfg.num_heads/2, None), blocks, merger: Qwen3VLVisionPatchMerger::new(&cfg, vb.pp("merger"), false)?, ds_idx: cfg.deepstack_visual_indexes.clone(), ds_mergers, dtype: vb.dtype() })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        self.patch_embed.to_device(d)?; let pw = self.pos_embed.embeddings().to_device(d)?; self.pos_embed = Embedding::new(pw, self.pos_embed.hidden_size());
        for b in &mut self.blocks { b.to_device(d)?; } self.merger.to_device(d)?; for m in &mut self.ds_mergers { m.to_device(d)?; } Ok(())
    }
    pub fn forward(&self, pv: &Tensor, thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let x = self.patch_embed.forward(pv)?; let pos = self.fast_pos_embed_interpolate(thw)?; let x = x.broadcast_add(&pos)?;
        let rot = self.rot_pos_emb(thw)?; let sl = x.dim(0)?; let mut x = x.reshape((sl, ()))?; let rot = rot.reshape((sl, ()))?;
        let emb = Tensor::cat(&[&rot, &rot], D::Minus1)?; let (cos, sin) = (emb.cos()?, emb.sin()?);
        let cu_full = Tensor::cat(&thw.i((.., 1))?.mul(&thw.i((.., 2))?)?.to_vec1::<u32>()?.iter().enumerate().map(|(i, &s)| thw.i((i, 0)).map_err(anyhow::Error::from)?.repeat(s as usize).map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?, 0)?.flatten_all()?;
        let cu = cu_full.to_dtype(DType::F64)?.cumsum(0)?.to_dtype(DType::U32)?.pad_with_zeros(D::Minus1, 1, 0)?;
        let mut dsf = vec![];
        for (i, b) in self.blocks.iter().enumerate() { x = b.forward(&x, &cu, &cos, &sin)?; if let Some(idx) = self.ds_idx.iter().position(|&v| v == i) { dsf.push(self.ds_mergers[idx].forward(&x)?); } }
        Ok((self.merger.forward(&x)?, dsf))
    }
    fn fast_pos_embed_interpolate(&self, thw: &Tensor) -> Result<Tensor> {
        let mut idxs = vec![vec![]; 4]; let mut wts = vec![vec![]; 4]; let mut split = vec![];
        for i in 0..thw.dim(0)? {
            let [_, h, w] = thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("Expected 3 elements")); };
            split.push((h * w) as usize); let n_sub = (self.num_grid - 1) as f32;
            let hi = linspace(0.0, n_sub, h as usize, thw.device())?; let wi = linspace(0.0, n_sub, w as usize, thw.device())?;
            let hf = hi.to_dtype(DType::U32)?; let wf = wi.to_dtype(DType::U32)?;
            let hc = hf.affine(1.0, 1.0)?.clamp(0u32, n_sub as u32)?; let wc = wf.affine(1.0, 1.0)?.clamp(0u32, n_sub as u32)?;
            let dh = hi.sub(&hf.to_dtype(hi.dtype())?)?.unsqueeze(D::Minus1)?; let dw = wi.sub(&wf.to_dtype(hi.dtype())?)?.unsqueeze(0)?;
            let bh = hf.affine(self.num_grid as f64, 0.0)?.unsqueeze(D::Minus1)?; let bc = hc.affine(self.num_grid as f64, 0.0)?.unsqueeze(D::Minus1)?;
            idxs[0].extend_from_slice(&bh.broadcast_add(&wf.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idxs[1].extend_from_slice(&bh.broadcast_add(&wc.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idxs[2].extend_from_slice(&bc.broadcast_add(&wf.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idxs[3].extend_from_slice(&bc.broadcast_add(&wc.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            let (oh, ow) = (Tensor::ones_like(&dh)?.sub(&dh)?, Tensor::ones_like(&dw)?.sub(&dw)?);
            wts[0].extend_from_slice(&oh.broadcast_mul(&ow)?.flatten_all()?.to_vec1::<f32>()?);
            wts[1].extend_from_slice(&oh.broadcast_mul(&dw)?.flatten_all()?.to_vec1::<f32>()?);
            wts[2].extend_from_slice(&dh.broadcast_mul(&ow)?.flatten_all()?.to_vec1::<f32>()?);
            wts[3].extend_from_slice(&dh.broadcast_mul(&dw)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let pos = self.pos_embed.forward(&Tensor::new(idxs, thw.device())?)?.broadcast_mul(&Tensor::new(wts, thw.device())?.to_dtype(self.dtype)?.unsqueeze(D::Minus1)?)?;
        let res = pos.i(0)?.add(&pos.i(1)?)?.add(&pos.i(2)?)?.add(&pos.i(3)?)?;
        let mut outs = vec![]; let split_res = split_tensor(&res, &split, 0)?;
        for (i, p) in split_res.iter().enumerate() {
            let [t, h, w] = thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("Expected 3 elements")); };
            let ld: usize = p.dim(D::Minus1)?; let p = p.repeat((t as usize, 1))?;
            outs.push(p.reshape((t as usize, h as usize / self.sm_size, self.sm_size, w as usize / self.sm_size, self.sm_size, ld))?.permute((0, 1, 3, 2, 4, 5))?.flatten(0, 4)?);
        }
        Ok(Tensor::cat(&outs, 0)?)
    }
    fn rot_pos_emb(&self, thw: &Tensor) -> Result<Tensor> {
        let max_hw = thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let ft = self.rotary.forward(max_hw as usize, thw.device())?;
        let mut list = vec![];
        for i in 0..thw.dim(0)? {
            let [t, h, w] = thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("Expected 3 elements")); };
            let (mh, mw) = (h / self.sm_size as u32, w / self.sm_size as u32);
            let (br, bc) = (Tensor::arange(0, mh, thw.device())?, Tensor::arange(0, mw, thw.device())?);
            let (ir, ic) = (Tensor::arange(0, self.sm_size as u32, thw.device())?, Tensor::arange(0, self.sm_size as u32, thw.device())?);
            let ri = br.reshape(((), 1, 1, 1))?.contiguous()?.affine(self.sm_size as f64, 0.0)?.broadcast_add(&ir.reshape((1, 1, (), 1))?.contiguous()?)?.expand((mh as usize, mw as usize, self.sm_size, self.sm_size))?.flatten_all()?;
            let ci = bc.reshape((1, (), 1, 1))?.contiguous()?.affine(self.sm_size as f64, 0.0)?.broadcast_add(&ic.reshape((1, 1, 1, ()))?.contiguous()?)?.expand((mh as usize, mw as usize, self.sm_size, self.sm_size))?.flatten_all()?;
            let mut coords = Tensor::stack(&[ri, ci], D::Minus1)?.contiguous()?; if t > 1 { coords = coords.repeat((t as usize, 1))?; } list.push(coords);
        }
        let ids = Tensor::cat(&list, 0)?;
        Ok(Tensor::cat(&[ft.index_select(&ids.i((.., 0))?.contiguous()?, 0)?, ft.index_select(&ids.i((.., 1))?.contiguous()?, 0)?], 1)?.contiguous()?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextAttention {
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear,
    pub q_norm: RmsNorm, pub k_norm: RmsNorm,
    pub num_attention_heads: usize, pub num_key_value_heads: usize, pub head_dim: usize,
    pub num_kv_groups: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3VLTextAttention {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let nh = config.num_attention_heads; let hd = config.head_dim; let nkv = config.num_key_value_heads;
        let nkvg = nh / nkv; let sc = 1f64 / f64::sqrt(hd as f64);
        let q = get_qlinear_from_vb(vb.pp("q_proj"), "weight", map)?; let k = get_qlinear_from_vb(vb.pp("k_proj"), "weight", map)?;
        let v = get_qlinear_from_vb(vb.pp("v_proj"), "weight", map)?; let o = get_qlinear_from_vb(vb.pp("o_proj"), "weight", map)?;
        let qn = get_rms_norm(hd, config.rms_norm_eps, vb.pp("q_norm"))?; let kn = get_rms_norm(hd, config.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self { q_proj: q, k_proj: k, v_proj: v, o_proj: o, q_norm: qn, k_norm: kn, num_attention_heads: nh, num_key_value_heads: nkv, num_kv_groups: nkvg, head_dim: hd, scaling: sc, kv_cache: None })
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bs, ql, _) = xs.dims3()?;
        let qs = self.q_norm.forward(&self.q_proj.forward(xs)?.reshape((bs, ql, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?;
        let ks = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?;
        let vs = self.v_proj.forward(xs)?.reshape((bs, ql, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (qs, ks) = apply_rotary_pos_emb(&qs, &ks, cos, sin, false)?;
        let (ks, vs) = match &self.kv_cache { None => (ks, vs), Some((pk, pv)) => (Tensor::cat(&[pk, &ks], 2)?, Tensor::cat(&[pv, &vs], 2)?) };
        self.kv_cache = Some((ks.clone(), vs.clone()));
        let out = eager_attention_forward(&qs, &ks, &vs, Some(self.num_kv_groups), mask, self.scaling)?;
        Ok(self.o_proj.forward(&out.reshape((bs, ql, ()))?)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None }
}

#[derive(Debug, Clone)]
pub struct QGateUpDownMLP { pub g: QLinear, pub u: QLinear, pub d: QLinear, pub act: Activation }
impl QGateUpDownMLP {
    pub fn new(vb: VarBuilder, act: Activation, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let g = get_qlinear_from_vb(vb.pp("gate_proj"), "weight", map)?; let u = get_qlinear_from_vb(vb.pp("up_proj"), "weight", map)?; let d = get_qlinear_from_vb(vb.pp("down_proj"), "weight", map)?;
        Ok(Self { g, u, d, act })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.g.forward(x)?; let gate = self.act.forward(&gate)?;
        let up = self.u.forward(x)?; Ok(self.d.forward(&gate.mul(&up)?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextDecoderLayer { pub attn: Qwen3VLTextAttention, pub mlp: QGateUpDownMLP, pub in_ln: RmsNorm, pub post_ln: RmsNorm }
impl Qwen3VLTextDecoderLayer {
    pub fn new(cfg: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let attn = Qwen3VLTextAttention::new(cfg.clone(), vb.pp("self_attn"), map)?;
        let mlp = QGateUpDownMLP::new(vb.pp("mlp"), cfg.hidden_act, map)?;
        let in_ln = RmsNorm::new(get_any(&vb.pp("input_layernorm"), "weight", map)?, cfg.rms_norm_eps);
        let post_ln = RmsNorm::new(get_any(&vb.pp("post_attention_layernorm"), "weight", map)?, cfg.rms_norm_eps);
        Ok(Self { attn, mlp, in_ln, post_ln })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = x.clone(); let x = self.attn.forward(&self.in_ln.forward(x)?, cos, sin, mask)?; let x = r.add(&x)?;
        let r = x.clone(); let x = self.mlp.forward(&self.post_ln.forward(&x)?)?; Ok(r.add(&x)?)
    }
    pub fn clear_kv_cache(&mut self) { self.attn.clear_kv_cache(); }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextModel { pub embed: Embedding, pub layers: Vec<Option<Qwen3VLTextDecoderLayer>>, pub norm: RmsNorm, pub rotary: Qwen3VLTextRotaryEmbedding, pub mrope: Vec<usize>, pub is_baking: bool }
impl Qwen3VLTextModel {
    pub fn new(mut cfg: Qwen3VLTextConfig, vb: VarBuilder, map: Option<&HashMap<String, Tensor>>) -> Result<Self> {
        let ew = if probe(&vb.pp("embed_tokens"), "weight", map) { get_any(&vb.pp("embed_tokens"), "weight", map)? } else { Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, vb.device())? };
        if ew.dim(1)? != cfg.hidden_size { cfg.hidden_size = ew.dim(1)?; }
        let mut layers = vec![]; let vbl = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let mut ex = probe(&vbl.pp(i).pp("input_layernorm"), "weight", map);
            if !ex { ex = probe(&vbl.pp(i).pp("attn_norm"), "weight", map); }
            if ex { layers.push(Some(Qwen3VLTextDecoderLayer::new(cfg.clone(), vbl.pp(i), map)?)); } else { layers.push(None); }
        }
        let n = if probe(&vb.pp("norm"), "weight", map) { RmsNorm::new(get_any(&vb.pp("norm"), "weight", map)?, cfg.rms_norm_eps) }
        else if probe(&vb.pp("output_norm"), "weight", map) { RmsNorm::new(get_any(&vb.pp("output_norm"), "weight", map)?, cfg.rms_norm_eps) }
        else { RmsNorm::new(Tensor::ones(cfg.hidden_size, DType::F32, vb.device())?, cfg.rms_norm_eps) };
        Ok(Self { embed: Embedding::new(ew, cfg.hidden_size), layers, norm: n, rotary: Qwen3VLTextRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta), mrope: cfg.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), is_baking: false })
    }
    pub fn forward(&mut self, x: &Tensor, offset: usize, pids: Option<&Tensor>, vm: Option<&Tensor>, ds: Option<Vec<Tensor>>) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let pids = match pids { Some(ids) => ids.clone(), None => Tensor::arange(offset as u32, (sl + offset) as u32, x.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, bs, sl))? };
        let (cos, sin) = self.rotary.forward(&pids, x.dtype(), self.mrope.clone())?;
        let mask: Option<Tensor> = if sl <= 1 { None } else { Some(prepare_causal_attention_mask(bs, sl, 0, x.device())?) };
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
    pub fn save_kv_cache(&mut self, path: &std::path::Path) -> Result<()> {
        if !path.exists() { std::fs::create_dir_all(path)?; }
        for (i, l_opt) in self.layers.iter_mut().enumerate() {
            if let Some(l) = l_opt { if let Some((k, v)) = &l.attn.kv_cache {
                let rk = Self::compress_to_bitkv(i, k)?; let rv = Self::compress_to_bitkv(i, v)?;
                let mut m = HashMap::new(); m.insert("k_a".to_string(), rk.0); m.insert("k_p".to_string(), rk.1); m.insert("k_s".to_string(), rk.2);
                m.insert("v_a".to_string(), rv.0); m.insert("v_p".to_string(), rv.1); m.insert("v_s".to_string(), rv.2);
                m.insert("shape".to_string(), Tensor::from_vec(rk.3.iter().map(|&x| x as u32).collect::<Vec<_>>(), (rk.3.len(),), k.device())?);
                candle_core::safetensors::save(&m, path.join(format!("layer_{}_bitkv.safetensors", i)))?;
            } }
        }
        Ok(())
    }
    fn compress_to_bitkv(l_i: usize, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
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
    pub fn load_kv_cache(&mut self, path: &std::path::Path, device: &Device) -> Result<()> {
        let mut fkv: Option<(Tensor, Tensor)> = None;
        let dt = if device.is_cpu() { DType::F32 } else { DType::BF16 };
        for (i, l_opt) in self.layers.iter_mut().enumerate() {
            if let Some(l) = l_opt {
                let p = path.join(format!("layer_{}_kv.safetensors", i));
                let (mut k, mut v) = if p.exists() { let m = candle_core::safetensors::load(p, device)?; (m.get("k").unwrap().to_dtype(dt)?, m.get("v").unwrap().to_dtype(dt)?) }
                else if let Some((ref fk, ref fv)) = fkv { (fk.clone(), fv.clone()) } else { continue; };
                let (th, td) = (l.attn.num_key_value_heads, l.attn.head_dim);
                if k.dim(1)? != th { if th % k.dim(1)? == 0 { let r = th / k.dim(1)?; k = k.repeat((1, r, 1, 1))?; v = v.repeat((1, r, 1, 1))?; } else { k = k.narrow(1, 0, th)?; v = v.narrow(1, 0, th)?; } }
                if k.dim(3)? != td { k = Self::apply_linear_bridge(&k, td)?; v = Self::apply_linear_bridge(&v, td)?; }
                if fkv.is_none() { fkv = Some((k.clone(), v.clone())); } l.attn.kv_cache = Some((k, v));
            }
        }
        Ok(())
    }
    fn apply_linear_bridge(x: &Tensor, target: usize) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?; let xf = x.to_dtype(DType::F32)?;
        let rms = (xf.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()).max(1e-6);
        let sc = ((d as f64 / target as f64).sqrt() * 0.7071067811865476_f64) / (rms as f64);
        if target >= d {
            let up = if target > d { (Tensor::stack(&[xf.clone(), ((xf.clone() + xf.roll(1, D::Minus1)?)? * 0.5)?], D::Minus1)?.affine(sc, 0.0))?.reshape((b, h, s, target))? }
            else { xf.affine(sc * (rms as f64), 0.0)? };
            Ok(up.clamp(-10.0, 10.0)?.to_dtype(x.dtype())?)
        } else { Ok((x.narrow(D::Minus1, 0, target)?.to_dtype(DType::F32)?.affine(((d as f64 / target as f64).sqrt()) / (rms as f64), 0.0))?.to_dtype(x.dtype())?) }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLModel {
    config: Qwen3VLConfig, visual: Option<Qwen3VLVisionModel>, pub language_model: Qwen3VLTextModel, lm_head: QLinear, rope_deltas: Option<Tensor>, pub is_baking: bool,
}

impl Qwen3VLModel {
    pub fn set_baking(&mut self, baking: bool) { self.is_baking = baking; self.language_model.is_baking = baking; }
    pub fn new(config: Qwen3VLConfig, vb: VarBuilder) -> Result<Self> { Self::new_ext(config, vb, None, false, false) }
    pub fn new_ext(config: Qwen3VLConfig, vb: VarBuilder, map: Option<HashMap<String, Tensor>>, force_text_only: bool, is_baking: bool) -> Result<Self> {
        let config = config.clone(); let map_ref = map.as_ref();
        let visual = if !force_text_only && config.vision_config.is_some() {
            let v_cfg = config.vision_config.as_ref().unwrap();
            let vb_v = if probe(&vb, "model.visual.patch_embed.proj.weight", map_ref) { Some(vb.pp("model").pp("visual")) }
            else if probe(&vb, "visual.patch_embed.proj.weight", map_ref) { Some(vb.pp("visual")) }
            else if probe(&vb, "v.patch_embd.weight", map_ref) { Some(vb.pp("v")) } else { None };
            if let Some(vf) = vb_v { Some(Qwen3VLVisionModel::new(v_cfg.clone(), vf)?) } else { None }
        } else { None };
        let text_cfg = config.text_config.clone().ok_or(anyhow!("Missing text_config"))?;
        let vb_lm = if probe(&vb, "model.language_model.layers.0.input_layernorm.weight", map_ref) { vb.pp("model").pp("language_model") }
        else if probe(&vb, "model.layers.0.input_layernorm.weight", map_ref) { vb.pp("model") }
        else { vb.pp("model").pp("language_model") };
        let language_model = Qwen3VLTextModel::new(text_cfg, vb_lm, map_ref)?;
        if is_baking && !language_model.layers.is_empty() && language_model.layers[0].is_none() { return Err(anyhow!("Layer 0 missing")); }
        let lm_head = {
            let vh = if probe(&vb, "model.language_model.lm_head.weight", map_ref) { vb.pp("model").pp("language_model").pp("lm_head") }
            else if probe(&vb, "model.lm_head.weight", map_ref) { vb.pp("model").pp("lm_head") }
            else { vb.pp("model").pp("language_model").pp("lm_head") };
            get_qlinear_from_vb(vh, "weight", map_ref)?
        };
        let mut model = Self { config, visual, language_model, lm_head, rope_deltas: None, is_baking }; model.set_baking(is_baking); Ok(model)
    }
    pub fn save_kv_cache(&mut self, path: &std::path::Path) -> Result<()> { self.language_model.save_kv_cache(path) }
    pub fn load_kv_cache(&mut self, path: &std::path::Path, device: &Device) -> Result<()> { self.language_model.load_kv_cache(path, device) }
    fn get_vision_features(&self, vm: &Qwen3VLVisionModel, pv: &Tensor, thw: &Tensor) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (ie, dse) = vm.forward(pv, thw)?; let m = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let ss: Vec<usize> = prod_tensor_last_dim(thw)?.to_vec1::<u32>()?.iter().map(|&x| x as usize / m.pow(2)).collect(); Ok((split_tensor(&ie, &ss, 0)?, dse))
    }
    fn get_placeholder_mask(&self, ids: &Tensor, img: bool) -> Result<Tensor> {
        let tid = if img { self.config.image_token_id.unwrap_or(0) } else { self.config.video_token_id.unwrap_or(0) };
        Ok(ids.broadcast_eq(&Tensor::new(vec![tid as u32], ids.device())?)?.to_dtype(DType::U32)?)
    }
    fn get_rope_index(&self, ids: &Tensor, thw: Option<&Tensor>, vthw: Option<&Tensor>, m: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let m_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let (tid, vtid, vsid) = (self.config.image_token_id.unwrap_or(0), self.config.video_token_id.unwrap_or(0), self.config.vision_start_token_id.unwrap_or(0));
        let mut deltas = vec![]; let mut pos_ids = Tensor::ones((3, ids.dim(0)?, ids.dim(1)?), ids.dtype(), ids.device())?;
        let (mut ii, mut vi) = (0, 0);
        for i in 0..ids.dim(0)? {
            let mut cur_ids = ids.i(i)?; if let Some(mask) = m { if mask.i(i)?.sum_all()?.to_scalar::<u32>()? != mask.dim(1)? as u32 { cur_ids = cur_ids.gather(&nonzero_index(&mask.i(i)?)?, 0)?; } }
            let (mut ts, mut te) = (0, 0); let mut list: Vec<Tensor> = vec![];
            if let Ok(vidx) = get_vision_next_indices(&cur_ids, vsid as u32) {
                let vtoks = cur_ids.gather(&vidx, 0)?.to_vec1::<u32>()?; let vidx_v = vidx.to_vec1::<u32>()?;
                for (j, &t) in vtoks.iter().enumerate() {
                    let thw_v = if t == tid as u32 { let v = thw.unwrap().i(ii)?.to_vec1::<u32>()?; ii += 1; te = vidx_v[j]; v }
                    else if t == vtid as u32 { let v = vthw.as_ref().unwrap().i(vi)?.to_vec1::<u32>()?; vi += 1; te = vidx_v[j]; v } else { continue; };
                    let (gt, gh, gw) = (thw_v[0], thw_v[1] / m_size as u32, thw_v[2] / m_size as u32);
                    let tlen = te - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 };
                    list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?);
                    let base = sidx + tlen;
                    let ti = Tensor::arange(base, base + gt, ids.device())?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, (gh * gw) as usize))?.flatten_all()?;
                    let hi = Tensor::arange(base, base + gh, ids.device())?.unsqueeze(0)?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    let wi = Tensor::arange(base, base + gw, ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    list.push(Tensor::stack(&[ti, hi, wi], 0)?); ts = te + gt * gh * gw;
                }
            }
            if ts < cur_ids.dim(0)? as u32 { let tlen = cur_ids.dim(0)? as u32 - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 }; list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?); }
            let lp = Tensor::cat(&list, 1)?.reshape((3, 1, ()))?; pos_ids = pos_ids.slice_assign(&[(0..3), (i..i + 1), (0..ids.dim(1)?)], &lp)?;
            deltas.push(lp.max_all()?.to_scalar::<u32>()? as i64 + 1 - cur_ids.dim(0)? as i64);
        }
        let mut d_t = Tensor::new(deltas, ids.device())?; if d_t.rank() == 1 { d_t = d_t.unsqueeze(0)?; } Ok((pos_ids.contiguous()?, d_t))
    }
    pub fn forward(&mut self, ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, vthw: Option<&Tensor>, _cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let mut embs = self.language_model.embed.forward(ids)?;
        let (mut im, mut dse) = (None, None);
        if let Some(vm) = &self.visual { if let (Some(p), Some(t)) = (pv, thw) {
            let (ie, ds) = self.get_vision_features(vm, p, t)?; let vm_m = self.get_placeholder_mask(ids, true)?;
            embs = masked_scatter_dim0(&embs, &Tensor::cat(&ie, 0)?, &vm_m)?; im = Some(vm_m); dse = Some(ds);
        } }
        let (pids, _) = self.get_rope_index(ids, thw, vthw, None)?;
        let out = self.language_model.forward(&embs, offset, Some(&pids), im.as_ref(), dse)?;
        Ok(self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?)
    }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn device(&self) -> &Device { self.language_model.embed.embeddings().device() }
}