use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use candle_core::safetensors::MmapedSafetensors;
use half::f16;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::{get_conv1d, softplus},
    },
    position_embed::rope::{
        Qwen3_5TextRotaryEmbedding, glm_asr_apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        prepare_causal_attention_mask,
    },
};

pub struct QuantizedRegistry {
    mmaps: HashMap<String, Arc<MmapedSafetensors>>,
    tensor_to_file: HashMap<String, String>,
}

impl QuantizedRegistry {
    pub fn new(model_list: &[String]) -> Result<Self> {
        let mut mmaps = HashMap::new();
        let mut tensor_to_file = HashMap::new();
        for path in model_list {
            let mmap = unsafe { Arc::new(MmapedSafetensors::new(path)?) };
            for name in mmap.tensors().iter().map(|(n, _)| n.clone()) {
                tensor_to_file.insert(name, path.clone());
            }
            mmaps.insert(path.clone(), mmap);
        }
        Ok(Self { mmaps, tensor_to_file })
    }

    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let file_path = self.tensor_to_file.get(name).ok_or_else(|| anyhow!("Tensor {} not found", name))?;
        let mmap = self.mmaps.get(file_path).unwrap();
        let t = mmap.load(name, &Device::Cpu)?;
        match t.dtype() {
            DType::I64 => Ok(t.to_dtype(DType::U32)?),
            _ => Ok(t),
        }
    }
    
    pub fn get_q_tensor(&self, full_name: &str, suffix: &str) -> Result<Tensor> {
        let n1 = format!("{}.{}", full_name, suffix);
        let n2 = format!("{}.weight.{}", full_name, suffix);
        if let Ok(t) = self.get_tensor(&n1) { return Ok(t); }
        if let Ok(t) = self.get_tensor(&n2) { return Ok(t); }
        if suffix == "weight" { if let Ok(t) = self.get_tensor(full_name) { return Ok(t); } }
        anyhow::bail!("Strict find FAILED: {} ({})", full_name, suffix)
    }
}

pub struct QRmsNorm { full_name: String, eps: f64, persistent_weight: Option<Tensor> }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64) -> Result<Self> { Ok(Self { full_name: name.to_string(), eps, persistent_weight: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let w = reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?.to_dtype(DType::F16)?;
        self.persistent_weight = Some(w); Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } 
                else { reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?.to_dtype(DType::F16)? };
        let x_f32 = x.to_dtype(DType::F32)?;
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x_f32.broadcast_div(&(var + self.eps)?.sqrt()?)?.to_dtype(DType::F16)?;
        
        let w_size = w.elem_count();
        let last_dim = x.dim(D::Minus1)?;
        if w_size == last_dim {
            Ok(norm.broadcast_mul(&w)?)
        } else if last_dim % w_size == 0 {
            // Per-head normalization
            let num_heads = last_dim / w_size;
            let mut dims = x.dims().to_vec();
            dims.pop();
            dims.push(num_heads);
            dims.push(w_size);
            let norm_reshaped = norm.reshape(dims.as_slice())?;
            let mut w_dims = vec![1; dims.len()];
            w_dims[dims.len() - 1] = w_size;
            let out = norm_reshaped.broadcast_mul(&w.reshape(w_dims.as_slice())?)?;
            Ok(out.reshape(x.dims())?)
        } else {
            // Fallback: simple broadcast if possible
            Ok(norm.broadcast_mul(&w.reshape(vec![1; x.rank() - 1].into_iter().chain(std::iter::once(w_size)).collect::<Vec<_>>())?)?)
        }
    }
}

pub struct QLinear { full_name: String, persistent_weight: Option<Tensor>, persistent_bias: Option<Tensor> }
impl QLinear {
    pub fn new(name: &str) -> Result<Self> { Ok(Self { full_name: name.to_string(), persistent_weight: None, persistent_bias: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; self.persistent_bias = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let name = &self.full_name;
        if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "q_scales"), reg.get_q_tensor(name, "q_data"), reg.get_q_tensor(name, "q_shape")) {
            let s = s_t.to_device(dev)?.to_dtype(DType::F16)?; let d = d_t.to_device(dev)?.to_dtype(DType::F16)?;
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let high = (d.clone() / 16.0)?.floor()?; let low = d.sub(&(high.clone() * 16.0)?)?;
            let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.reshape(((), 32))?;
            self.persistent_weight = Some(combined.broadcast_mul(&s.unsqueeze(D::Minus1)?)?.reshape(shape.as_slice())?);
        } else {
            self.persistent_weight = Some(reg.get_q_tensor(name, "weight")?.to_device(dev)?.to_dtype(DType::F16)?);
        }
        if let Ok(b) = reg.get_q_tensor(name, "bias") { self.persistent_bias = Some(b.to_device(dev)?.to_dtype(DType::F16)?); }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device(); let name = &self.full_name;
        let w_raw = if let Some(pw) = &self.persistent_weight { pw.clone() } 
        else { 
            if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "q_scales"), reg.get_q_tensor(name, "q_data"), reg.get_q_tensor(name, "q_shape")) {
                let s = s_t.to_device(dev)?.to_dtype(DType::F16)?; let d = d_t.to_device(dev)?.to_dtype(DType::F16)?;
                let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
                let high = (d.clone() / 16.0)?.floor()?; let low = d.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.reshape(((), 32))?;
                combined.broadcast_mul(&s.unsqueeze(D::Minus1)?)?.reshape(shape.as_slice())?
            } else { reg.get_q_tensor(name, "weight")?.to_device(dev)?.to_dtype(DType::F16)? }
        };
        let x_in_dim = x.dim(D::Minus1)?;
        let w = if w_raw.dim(1)? == x_in_dim { w_raw.t()? } else { w_raw };
        let res = x.contiguous()?.broadcast_matmul(&w.to_dtype(DType::F16)?.contiguous()?)?;
        if let Some(pb) = &self.persistent_bias { Ok(res.broadcast_add(pb)?) }
        else if let Ok(b) = reg.get_q_tensor(name, "bias") { Ok(res.broadcast_add(&b.to_device(dev)?.to_dtype(DType::F16)?)?) } else { Ok(res) }
    }
}

fn join_name(p: &str, n: &str) -> String {
    if p.is_empty() { n.to_string() }
    else if p.ends_with('.') { format!("{}{}", p, n) }
    else { format!("{}.{}", p, n) }
}

pub struct QGatedDeltaNet { 
    in_proj_qkv: Option<QLinear>, in_proj_a: Option<QLinear>, in_proj_b: Option<QLinear>, in_proj_z: Option<QLinear>, 
    out_proj: QLinear, norm: QRmsNorm, 
    pub nk: usize, pub dk: usize, pub nv: usize, pub dv: usize,
    pub conv1d: Option<candle_nn::Conv1d>,
    pub dt_bias: Option<Tensor>, pub a_log: Option<Tensor>,
    pub delta_state: Option<Tensor>, pub conv_state: Option<Tensor> 
}
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        let nk = config.linear_num_key_heads; let dk = config.linear_key_head_dim;
        let nv = config.linear_num_value_heads; let dv = config.linear_value_head_dim;
        
        let in_proj_qkv = if vb.contains_tensor("in_proj_qkv.weight") { Some(QLinear::new(&join_name(&p, "in_proj_qkv"))?) } else { None };
        let in_proj_a = if vb.contains_tensor("in_proj_a.weight") { Some(QLinear::new(&join_name(&p, "in_proj_a"))?) } else { None };
        let in_proj_b = if vb.contains_tensor("in_proj_b.weight") { Some(QLinear::new(&join_name(&p, "in_proj_b"))?) } else { None };
        let in_proj_z = if vb.contains_tensor("in_proj_z.weight") { Some(QLinear::new(&join_name(&p, "in_proj_z"))?) } else { None };
        
        Ok(Self { 
            in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, 
            out_proj: QLinear::new(&join_name(&p, "out_proj"))?, 
            norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps)?, 
            nk, dk, nv, dv,
            conv1d: None, dt_bias: None, a_log: None,
            delta_state: None, conv_state: None 
        })
    }
    pub fn clear_vram(&mut self) { 
        if let Some(ref mut l) = self.in_proj_qkv { l.clear_vram(); } 
        if let Some(ref mut l) = self.in_proj_a { l.clear_vram(); } 
        if let Some(ref mut l) = self.in_proj_b { l.clear_vram(); } 
        if let Some(ref mut l) = self.in_proj_z { l.clear_vram(); } 
        self.out_proj.clear_vram(); self.norm.clear_vram(); 
        self.dt_bias = None; self.a_log = None; self.conv1d = None;
    }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { 
        if let Some(ref mut l) = self.in_proj_qkv { l.load_to_vram(reg, dev)?; } 
        if let Some(ref mut l) = self.in_proj_a { l.load_to_vram(reg, dev)?; } 
        if let Some(ref mut l) = self.in_proj_b { l.load_to_vram(reg, dev)?; } 
        if let Some(ref mut l) = self.in_proj_z { l.load_to_vram(reg, dev)?; } 
        self.out_proj.load_to_vram(reg, dev)?; self.norm.load_to_vram(reg, dev)?; 
        
        let p = self.norm.full_name.replace(".norm", ".linear_attn");
        if let Ok(dt) = reg.get_tensor(&format!("{}.dt_bias", p)) { self.dt_bias = Some(dt.to_device(dev)?.to_dtype(DType::F16)?); }
        if let Ok(a) = reg.get_tensor(&format!("{}.A_log", p)) { self.a_log = Some(a.to_device(dev)?.to_dtype(DType::F16)?); }
        
        if let Ok(cw) = reg.get_tensor(&format!("{}.conv1d.weight", p)) {
            let cb = reg.get_tensor(&format!("{}.conv1d.bias", p)).ok();
            let c_w = cw.to_device(dev)?.to_dtype(DType::F16)?;
            let c_b = if let Some(b) = cb { Some(b.to_device(dev)?.to_dtype(DType::F16)?) } else { None };
            let (out_c, in_c, k) = c_w.dims3()?;
            self.conv1d = Some(candle_nn::conv1d(in_c, out_c, k, candle_nn::Conv1dConfig { padding: k - 1, groups: in_c, ..Default::default() }, vb_from_weights(c_w, c_b, dev)?)?);
        }
        Ok(()) 
    }
    pub fn forward(&mut self, x: &Tensor, _mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?; let (nk, dk, nv, dv) = (self.nk, self.dk, self.nv, self.dv);
        let (mut q, mut k, mut v, z) = if let Some(ref l) = self.in_proj_qkv {
            let proj = l.forward(x, reg)?;
            let q = proj.narrow(D::Minus1, 0, nk * dk)?;
            let k = proj.narrow(D::Minus1, nk * dk, nk * dk)?;
            let v = proj.narrow(D::Minus1, nk * dk * 2, nv * dv)?;
            let z = if let Some(ref lz) = self.in_proj_z { lz.forward(x, reg)? } else { x.clone() };
            (q, k, v, z)
        } else {
            (self.in_proj_a.as_ref().unwrap().forward(x, reg)?, self.in_proj_b.as_ref().unwrap().forward(x, reg)?, self.in_proj_b.as_ref().unwrap().forward(x, reg)?, self.in_proj_z.as_ref().unwrap().forward(x, reg)?)
        };
        
        if let Some(ref conv) = self.conv1d {
            let mut c_in = Tensor::cat(&[&q, &k, &v], D::Minus1)?.transpose(1, 2)?;
            if let Some(ref pc) = self.conv_state { c_in = Tensor::cat(&[pc, &c_in], 2)?; }
            self.conv_state = Some(c_in.narrow(2, c_in.dim(2)? - conv.weight().dim(2)? + 1, conv.weight().dim(2)? - 1)?.detach());
            let c_out = conv.forward(&c_in)?.transpose(1, 2)?;
            q = c_out.narrow(D::Minus1, 0, nk * dk)?;
            k = c_out.narrow(D::Minus1, nk * dk, nk * dk)?;
            v = c_out.narrow(D::Minus1, nk * dk * 2, nv * dv)?;
        }

        let q = q.reshape((bs, sl, nk, dk))?; let k = k.reshape((bs, sl, nk, dk))?; let v = v.reshape((bs, sl, nv, dv))?;
        
        // Gating parameters
        let dt = if let Some(ref d) = self.dt_bias { d.reshape((1, 1, nv, 1))?.broadcast_as((bs, sl, nv, 1))? } else { Tensor::ones((bs, sl, nv, 1), DType::F16, x.device())? };
        let a = if let Some(ref al) = self.a_log { al.exp()?.neg()?.reshape((1, 1, nv, 1))?.broadcast_as((bs, sl, nv, 1))? } else { Tensor::zeros((bs, sl, nv, 1), DType::F16, x.device())? };
        
        let mut state = self.delta_state.take().unwrap_or_else(|| Tensor::zeros((bs, nk, dk, dv), DType::F16, x.device()).unwrap());
        let mut outputs = vec![];
        
        for t in 0..sl {
            let qt = q.narrow(1, t, 1)?.squeeze(1)?; // (bs, nk, dk)
            let kt = k.narrow(1, t, 1)?.squeeze(1)?; // (bs, nk, dk)
            let vt = v.narrow(1, t, 1)?.squeeze(1)?; // (bs, nv, dv)
            let dtt = softplus(&dt.narrow(1, t, 1)?.squeeze(1)?)?; // (bs, nv, 1)
            let at = a.narrow(1, t, 1)?.squeeze(1)?.mul(&dtt)?.exp()?; // (bs, nv, 1)
            
            // Linear recurrence: S = S * A + (dt * K^T * V)
            let update = kt.unsqueeze(D::Minus1)?.matmul(&vt.unsqueeze(D::Minus2)?)?; // (bs, nk, dk, dv)
            let scaled_update = update.broadcast_mul(&dtt.unsqueeze(D::Minus1)?)?;
            state = state.broadcast_mul(&at.unsqueeze(D::Minus1)?)?.add(&scaled_update)?;
            
            let out_t = qt.unsqueeze(D::Minus2)?.matmul(&state)?.squeeze(D::Minus2)?; // (bs, nv, dv)
            outputs.push(out_t.unsqueeze(1)?);
        }
        let out = Tensor::cat(&outputs, 1)?.reshape((bs, sl, nv * dv))?;
        self.delta_state = Some(state.detach());
        let res = self.out_proj.forward(&(out * z.silu()?)?, reg)?;
        self.norm.forward(&res, reg)
    }
}

fn vb_from_weights(w: Tensor, b: Option<Tensor>, dev: &Device) -> Result<VarBuilder> {
    let mut map = HashMap::new(); map.insert("weight".to_string(), w);
    if let Some(bv) = b { map.insert("bias".to_string(), bv); }
    Ok(VarBuilder::from_tensors(map, DType::F16, dev))
}

pub struct QAttention { q_proj: QLinear, k_proj: QLinear, v_proj: QLinear, o_proj: QLinear, q_norm: QRmsNorm, k_norm: QRmsNorm, pub nh: usize, pub nkv: usize, pub hd: usize, scaling: f64, pub h: usize, pub kv_cache: Option<(Tensor, Tensor)> }
impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self {
            q_proj: QLinear::new(&join_name(&p, "q_proj"))?, k_proj: QLinear::new(&join_name(&p, "k_proj"))?, v_proj: QLinear::new(&join_name(&p, "v_proj"))?, o_proj: QLinear::new(&join_name(&p, "o_proj"))?,
            q_norm: QRmsNorm::new(&join_name(&p, "q_norm"), config.rms_norm_eps)?, k_norm: QRmsNorm::new(&join_name(&p, "k_norm"), config.rms_norm_eps)?,
            nh: config.num_attention_heads, nkv: config.num_key_value_heads, hd: config.head_dim, scaling: 1.0 / (config.head_dim as f64).sqrt(), h: config.hidden_size, kv_cache: None,
        })
    }
    pub fn clear_vram(&mut self) { self.q_proj.clear_vram(); self.k_proj.clear_vram(); self.v_proj.clear_vram(); self.o_proj.clear_vram(); self.q_norm.clear_vram(); self.k_norm.clear_vram(); }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.q_proj.load_to_vram(reg, dev)?; self.k_proj.load_to_vram(reg, dev)?; self.v_proj.load_to_vram(reg, dev)?; self.o_proj.load_to_vram(reg, dev)?;
        self.q_norm.load_to_vram(reg, dev)?; self.k_norm.load_to_vram(reg, dev)?; Ok(())
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let q_raw = self.q_proj.forward(x, reg)?; let k_raw = self.k_proj.forward(x, reg)?; let v_raw = self.v_proj.forward(x, reg)?; 
        let q_h = self.nh * self.hd;
        let q_states = q_raw.narrow(D::Minus1, 0, q_h)?; let gate = q_raw.narrow(D::Minus1, q_h, q_h)?;
        let q = self.q_norm.forward(&q_states.reshape((bs, sl, self.nh, self.hd))?, reg)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k_raw.reshape((bs, sl, self.nkv, k_raw.dim(D::Minus1)? / self.nkv))?, reg)?.transpose(1, 2)?;
        let v = v_raw.reshape((bs, sl, self.nkv, v_raw.dim(D::Minus1)? / self.nkv))?.transpose(1, 2)?;
        let (q, k) = glm_asr_apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
        let (k, v) = match &self.kv_cache { None => (k, v), Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?) };
        self.kv_cache = Some((k.clone(), v.clone()));
        let attn = crate::models::common::eager_attention_forward(&q, &k, &v, Some(self.nh / self.nkv), mask, self.scaling)?;
        let out = attn.reshape((bs, sl, ()))?.contiguous()?;
        self.o_proj.forward(&(out * gate.silu()?)?, reg)
    }
}

pub struct QDecoderLayer { pub self_attn: Option<QAttention>, pub linear_attn: Option<QGatedDeltaNet>, mlp_gate: QLinear, mlp_up: QLinear, mlp_down: QLinear, in_norm: QRmsNorm, post_norm: QRmsNorm }
impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let lt = config.layer_types[idx].clone(); let p = vb.prefix();
        let (sa, la) = if lt == "linear_attention" { (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?)) } else { (Some(QAttention::new(vb.pp("self_attn"), config)?), None) };
        Ok(Self { self_attn: sa, linear_attn: la, mlp_gate: QLinear::new(&join_name(&p, "mlp.gate_proj"))?, mlp_up: QLinear::new(&join_name(&p, "mlp.up_proj"))?, mlp_down: QLinear::new(&join_name(&p, "mlp.down_proj"))?, in_norm: QRmsNorm::new(&join_name(&p, "input_layernorm"), config.rms_norm_eps)?, post_norm: QRmsNorm::new(&join_name(&p, "post_attention_layernorm"), config.rms_norm_eps)? })
    }
    pub fn clear_vram(&mut self) { self.in_norm.clear_vram(); if let Some(ref mut sa) = self.self_attn { sa.clear_vram(); } if let Some(ref mut la) = self.linear_attn { la.clear_vram(); } self.post_norm.clear_vram(); self.mlp_gate.clear_vram(); self.mlp_up.clear_vram(); self.mlp_down.clear_vram(); }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { self.in_norm.load_to_vram(reg, dev)?; if let Some(ref mut sa) = self.self_attn { sa.load_to_vram(reg, dev)?; } if let Some(ref mut la) = self.linear_attn { la.load_to_vram(reg, dev)?; } self.post_norm.load_to_vram(reg, dev)?; self.mlp_gate.load_to_vram(reg, dev)?; self.mlp_up.load_to_vram(reg, dev)?; self.mlp_down.load_to_vram(reg, dev)?; Ok(()) }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone(); let mut h = self.in_norm.forward(x, reg)?;
        if let Some(ref mut sa) = self.self_attn { h = sa.forward(&h, cos, sin, mask, reg)?; } else if let Some(ref mut la) = self.linear_attn { h = la.forward(&h, mask, reg)?; }
        let h2 = (h + residual)?; let residual_2 = h2.clone(); let h = self.post_norm.forward(&h2, reg)?;
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?; let up = self.mlp_up.forward(&h, reg)?;
        Ok((self.mlp_down.forward(&(gate * up)?, reg)? + residual_2)?)
    }
}

pub struct QEmbedding { full_name: String, persistent_weight: Option<Tensor> }
impl QEmbedding {
    pub fn new(name: &str) -> Self { Self { full_name: name.to_string(), persistent_weight: None } }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let name = &self.full_name;
        if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "q_scales"), reg.get_q_tensor(name, "q_data"), reg.get_q_tensor(name, "q_shape")) {
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let s = s_t.to_device(dev)?.to_dtype(DType::F16)?; let d = d_t.to_device(dev)?.to_dtype(DType::F16)?;
            let high = (d.clone() / 16.0)?.floor()?; let low = d.sub(&(high.clone() * 16.0)?)?;
            let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.reshape(((), 32))?;
            self.persistent_weight = Some(combined.broadcast_mul(&s.unsqueeze(D::Minus1)?)?.reshape(shape.as_slice())?);
        } else {
            self.persistent_weight = Some(reg.get_q_tensor(name, "weight")?.to_device(dev)?.to_dtype(DType::F16)?);
        }
        Ok(())
    }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } else { reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?.to_dtype(DType::F16)? };
        Ok(w.index_select(&ids.flatten_all()?.to_dtype(DType::U32)?, 0)?.reshape((b, s, ()))?.to_device(dev)?.to_dtype(DType::F16)?)
    }
}

pub struct QTextModel { pub embed: QEmbedding, pub layers: Vec<QDecoderLayer>, pub norm: QRmsNorm, pub rotary: Qwen3_5TextRotaryEmbedding, pub mrope: Vec<usize> }
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let p = vb.prefix();
        Ok(Self { embed: QEmbedding::new(&join_name(&p, "embed_tokens")), layers, norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps)?, rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, config.rope_parameters.rope_theta), mrope: config.rope_parameters.mrope_section.clone() })
    }
    pub fn forward(&mut self, _ids: &Tensor, _pos: &Tensor, _offset: usize, _reg: &QuantizedRegistry) -> Result<Tensor> { anyhow::bail!("Not used") }
}

pub struct QuantizedQwen3_5Model { pub model: QTextModel, pub head: QLinear }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self { model: QTextModel::new(vb.pp("model.language_model"), &config.text_config)?, head: QLinear::new(&join_name(&p, "model.language_model.embed_tokens"))? })
    }
    pub fn load_all_to_vram(&mut self, _reg: &QuantizedRegistry, _dev: &Device) -> Result<()> { Ok(()) }
    pub fn forward(&mut self, _ids: &Tensor, _offset: usize, _reg: &QuantizedRegistry, _img: Option<&Tensor>) -> Result<Tensor> { anyhow::bail!("Not used") }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } if let Some(ref mut la) = l.linear_attn { la.delta_state = None; } } }
}

use memmap2::MmapMut;
use std::fs::OpenOptions;
use candle_core::Shape;

pub enum LayerContext { Attention { k: Tensor, v: Tensor }, DeltaNet { state: Tensor } }
pub struct DiskStateManager { mmap: MmapMut, layer_stride: usize }
impl DiskStateManager {
    pub fn new(num_layers: usize) -> Result<Self> {
        let layer_stride = 128 * 1024 * 1024;
        let path = crate::utils::paths::get_kv_dir(None).join("kv_cache_pool.bin");
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        file.set_len((num_layers * layer_stride) as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, layer_stride })
    }
    pub fn save_layer_context(&mut self, layer_idx: usize, ctx: &LayerContext) -> Result<()> {
        let offset = layer_idx * self.layer_stride;
        match ctx {
            LayerContext::Attention { k, v } => { self.write_tensor(offset, k)?; self.write_tensor(offset + (k.elem_count() * 2), v)?; },
            LayerContext::DeltaNet { state } => { self.write_tensor(offset, state)?; }
        }
        Ok(())
    }
    pub fn load_layer_context(&self, layer_idx: usize, lt: &str, shape_k: &Shape, shape_v: &Shape, dev: &Device) -> Result<LayerContext> {
        let offset = layer_idx * self.layer_stride;
        if lt == "linear_attention" { Ok(LayerContext::DeltaNet { state: self.read_tensor(offset, shape_k, dev)? }) }
        else { Ok(LayerContext::Attention { k: self.read_tensor(offset, shape_k, dev)?, v: self.read_tensor(offset + (shape_k.elem_count() * 2), shape_v, dev)? }) }
    }
    fn write_tensor(&mut self, offset: usize, t: &Tensor) -> Result<()> {
        let data = t.detach().to_device(&Device::Cpu)?.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<f16>()?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        self.mmap[offset..offset + bytes.len()].copy_from_slice(bytes); Ok(())
    }
    fn read_tensor(&self, offset: usize, shape: &Shape, dev: &Device) -> Result<Tensor> {
        let raw_slice = &self.mmap[offset..offset + shape.elem_count() * 2];
        Ok(Tensor::from_raw_buffer(raw_slice, DType::F16, shape.dims(), dev)?)
    }
}
