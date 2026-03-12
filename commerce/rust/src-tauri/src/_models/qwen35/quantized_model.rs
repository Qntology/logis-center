use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor, Module};
use candle_nn::{VarBuilder};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use candle_core::safetensors::MmapedSafetensors;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::softplus,
    },
    position_embed::rope::{
        Qwen3_5TextRotaryEmbedding,
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
        if let Some(file_path) = self.tensor_to_file.get(name) {
            let mmap = self.mmaps.get(file_path).unwrap();
            let t = mmap.load(name, &Device::Cpu)?;
            match t.dtype() {
                DType::I64 => Ok(t.to_dtype(DType::U32)?),
                _ => Ok(t),
            }
        } else {
            let parts: Vec<&str> = name.split('.').collect();
            let n = parts.len();
            if n >= 3 {
                let last_three = format!("{}.{}.{}", parts[n-3], parts[n-2], parts[n-1]);
                if let Some(file_path) = self.tensor_to_file.get(&last_three) {
                    let mmap = self.mmaps.get(file_path).unwrap();
                    return Ok(mmap.load(&last_three, &Device::Cpu)?);
                }
            }
            if n >= 2 {
                let last_two = format!("{}.{}", parts[n-2], parts[n-1]);
                if let Some(file_path) = self.tensor_to_file.get(&last_two) {
                    let mmap = self.mmaps.get(file_path).unwrap();
                    return Ok(mmap.load(&last_two, &Device::Cpu)?);
                }
            }
            let last_one = parts.last().unwrap().to_string();
            if let Some(file_path) = self.tensor_to_file.get(&last_one) {
                let mmap = self.mmaps.get(file_path).unwrap();
                return Ok(mmap.load(&last_one, &Device::Cpu)?);
            }
            anyhow::bail!("Tensor {} not found even with fuzzy match", name)
        }
    }
    
    pub fn get_weight_tensor(&self, full_name: &str, suffix: &str) -> Result<Tensor> {
        let n1 = format!("{}.{}", full_name, suffix);
        let n2 = format!("{}.weight.{}", full_name, suffix);
        if let Ok(t) = self.get_tensor(&n1) { return Ok(t); }
        if let Ok(t) = self.get_tensor(&n2) { return Ok(t); }
        if suffix == "weight" { if let Ok(t) = self.get_tensor(full_name) { return Ok(t); } }
        anyhow::bail!("Weight not found: {} ({})", full_name, suffix)
    }
}

fn prepare_weight(t: &Tensor, dev: &Device) -> Result<Tensor> {
    Ok(t.to_device(dev)?.to_dtype(DType::F16)?)
}

pub struct QRmsNorm { pub full_name: String, eps: f64, offset: f64, persistent_weight: Arc<RwLock<Option<Tensor>>> }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64, offset: f64) -> Result<Self> { Ok(Self { full_name: name.to_string(), eps, offset, persistent_weight: Arc::new(RwLock::new(None)) }) }
    pub fn clear_vram(&mut self) { *self.persistent_weight.write().unwrap() = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        let w = reg.get_weight_tensor(&self.full_name, "weight")?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let w = if self.offset != 0.0 { w.affine(1.0, self.offset)? } else { w };
        let w = w.to_dtype(DType::F16)?.to_device(dev)?;
        *self.persistent_weight.write().unwrap() = Some(w); Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let w = {
            let cache = self.persistent_weight.read().unwrap();
            if let Some(pw) = &*cache { Some(pw.clone()) } else { None as Option<Tensor> }
        };
        let w = if let Some(w) = w { w } 
                else { 
                    let mut cache = self.persistent_weight.write().unwrap();
                    let w = reg.get_weight_tensor(&self.full_name, "weight")?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let w = if self.offset != 0.0 { w.affine(1.0, self.offset)? } else { w };
                    let w = w.to_dtype(DType::F16)?.to_device(dev)?;
                    *cache = Some(w.clone());
                    w
                };
        
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm = x_f32.powf(2.0)?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm)?;
        let res = x_normed.broadcast_mul(&w.to_dtype(DType::F32)?)?;
        Ok(res.to_dtype(DType::F16)?)
    }
}

pub struct QLinear { pub full_name: String, pub persistent_weight: Arc<RwLock<Option<Tensor>>>, pub persistent_bias: Arc<RwLock<Option<Tensor>>> }
impl QLinear {
    pub fn new(name: &str) -> Result<Self> { Ok(Self { full_name: name.to_string(), persistent_weight: Arc::new(RwLock::new(None)), persistent_bias: Arc::new(RwLock::new(None)) }) }
    pub fn clear_vram(&mut self) { *self.persistent_weight.write().unwrap() = None; *self.persistent_bias.write().unwrap() = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        let t = reg.get_weight_tensor(&self.full_name, "weight")?;
        *self.persistent_weight.write().unwrap() = Some(prepare_weight(&t, dev)?);
        if let Ok(b) = reg.get_weight_tensor(&self.full_name, "bias") { 
            *self.persistent_bias.write().unwrap() = Some(b.to_device(dev)?.to_dtype(DType::F16)?); 
        }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let in_features = x.dim(D::Minus1)?;
        let w = {
            let cache = self.persistent_weight.read().unwrap();
            if let Some(pw) = &*cache {
                if pw.rank() == 2 && pw.dim(0)? == in_features { Some(pw.clone()) } else { None }
            } else { None }
        };
        let w = if let Some(w) = w { w } else {
            let mut cache = self.persistent_weight.write().unwrap();
            let weight = if let Some(pw) = &*cache { pw.clone() } 
                         else { prepare_weight(&reg.get_weight_tensor(&self.full_name, "weight")?, dev)? };
            let out_features = weight.elem_count() / in_features;
            let optimized = if weight.rank() == 2 && weight.dim(0)? == in_features { weight } 
                            else { weight.reshape((out_features, in_features))?.t()?.contiguous()? };
            *cache = Some(optimized.clone());
            optimized
        };
        let res = x.contiguous()?.broadcast_matmul(&w)?;
        let b = self.persistent_bias.read().unwrap().clone();
        if let Some(pb) = b { Ok(res.broadcast_add(&pb)?) } else { Ok(res) }
    }
}

pub struct QGatedDeltaNet { 
    pub in_proj_qkv: QLinear, pub in_proj_z: QLinear, pub in_proj_b: QLinear, pub in_proj_a: QLinear, 
    pub out_proj: QLinear, pub norm: QRmsNorm, pub nk: usize, pub dk: usize, pub nv: usize, pub dv: usize,
    pub conv1d: Option<candle_nn::Conv1d>, pub dt_bias: Option<Tensor>, pub a_log: Option<Tensor>,
    pub recurrent_state_cache: Option<Tensor>, pub conv_state_cache: Option<Tensor>,
    pub scale: f64,
}
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self { 
            in_proj_qkv: QLinear::new(&join_name(&p, "in_proj_qkv"))?, in_proj_z: QLinear::new(&join_name(&p, "in_proj_z"))?,
            in_proj_b: QLinear::new(&join_name(&p, "in_proj_b"))?, in_proj_a: QLinear::new(&join_name(&p, "in_proj_a"))?,
            out_proj: QLinear::new(&join_name(&p, "out_proj"))?, norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps, 0.0)?, 
            nk: config.linear_num_key_heads, dk: config.linear_key_head_dim, nv: config.linear_num_value_heads, dv: config.linear_value_head_dim,
            conv1d: None, dt_bias: None, a_log: None, recurrent_state_cache: None, conv_state_cache: None,
            scale: 1.0 / (config.linear_key_head_dim as f64).sqrt(),
        })
    }
    pub fn clear_vram(&mut self) { 
        self.in_proj_qkv.clear_vram(); self.in_proj_z.clear_vram(); self.in_proj_b.clear_vram(); self.in_proj_a.clear_vram();
        self.out_proj.clear_vram(); self.norm.clear_vram(); self.dt_bias = None; self.a_log = None; self.conv1d = None;
    }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { 
        if self.in_proj_qkv.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        self.in_proj_qkv.load_to_vram(reg, dev)?; self.in_proj_z.load_to_vram(reg, dev)?; self.in_proj_b.load_to_vram(reg, dev)?; self.in_proj_a.load_to_vram(reg, dev)?;
        self.out_proj.load_to_vram(reg, dev)?; self.norm.load_to_vram(reg, dev)?; 
        let p = self.norm.full_name.replace(".norm", "");
        if let Ok(dt) = reg.get_tensor(&format!("{}.dt_bias", p)) { self.dt_bias = Some(dt.to_device(dev)?.to_dtype(DType::F32)?); }
        if let Ok(a) = reg.get_tensor(&format!("{}.A_log", p)) { self.a_log = Some(a.to_device(dev)?.to_dtype(DType::F32)?); }
        if let Ok(cw) = reg.get_weight_tensor(&format!("{}.conv1d", p), "weight") {
            let weight = prepare_weight(&cw, dev)?;
            let channels = weight.elem_count() / 4;
            let c_w = weight.reshape((channels, 1, 4))?;
            let mut map = HashMap::new(); map.insert("weight".to_string(), c_w);
            let vb = VarBuilder::from_tensors(map, DType::F16, dev);
            self.conv1d = Some(crate::models::common::get_conv1d(vb, channels, channels, 4, 3, 1, 1, channels, false)?);
        }
        Ok(()) 
    }

    pub fn forward(&mut self, x: &Tensor, _mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?; 
        let (nk, dk, nv, dv) = (self.nk, self.dk, self.nv, self.dv);
        let mixed_qkv = self.in_proj_qkv.forward(x, reg)?; 
        let z = self.in_proj_z.forward(x, reg)?;
        let b = self.in_proj_b.forward(x, reg)?; 
        let a = self.in_proj_a.forward(x, reg)?;
        
        let q = mixed_qkv.narrow(D::Minus1, 0, nk * dk)?; 
        let k = mixed_qkv.narrow(D::Minus1, nk * dk, nk * dk)?;
        let v = mixed_qkv.narrow(D::Minus1, nk * dk * 2, nv * dv)?;
        
        // Conv1D part
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?;
        let qkv = qkv.transpose(1, 2)?; // [bs, dim, sl]
        let qkv_conv = if let Some(ref conv) = self.conv1d {
            if sl == 1 && self.conv_state_cache.is_some() {
                let state = self.conv_state_cache.as_ref().unwrap().to_device(x.device())?;
                let state_new = Tensor::cat(&[&state, &qkv], D::Minus1)?;
                let out = conv.forward(&state_new)?;
                self.conv_state_cache = Some(state_new.narrow(D::Minus1, 1, 3)?);
                out
            } else {
                let state = qkv.pad_with_zeros(D::Minus1, 3, 0)?;
                let out = conv.forward(&state)?;
                self.conv_state_cache = Some(state.narrow(D::Minus1, sl, 3)?);
                out
            }
        } else {
            qkv
        };
        let qkv_conv = qkv_conv.transpose(1, 2)?.silu()?;
        
        let q = qkv_conv.narrow(D::Minus1, 0, nk * dk)?;
        let k = qkv_conv.narrow(D::Minus1, nk * dk, nk * dk)?;
        let v = qkv_conv.narrow(D::Minus1, nk * dk * 2, nv * dv)?;
        
        let q = crate::utils::tensor_utils::l2_normalize(&q.reshape((bs, sl, nk, dk))?, 3)?;
        let k = crate::utils::tensor_utils::l2_normalize(&k.reshape((bs, sl, nk, dk))?, 3)?;
        let v = v.reshape((bs, sl, nv, dv))?;

        // Gating
        let dt_bias = self.dt_bias.as_ref().unwrap().to_device(x.device())?;
        let a_log = self.a_log.as_ref().unwrap().to_device(x.device())?;
        let soft_a = softplus(&a.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?)?;
        let g = a_log.exp()?.affine(-1.0, 0.0)?.broadcast_mul(&soft_a)?.exp()?;
        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)?)?;

        let mut state = self.recurrent_state_cache.take().unwrap_or_else(|| Tensor::zeros((bs, nv, dk, dv), DType::F32, x.device()).unwrap()).to_device(x.device())?;
        let mut outputs = vec![];
        
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;

        for t in 0..sl {
            let q_t = q.i((.., t, ..))?.broadcast_mul(&Tensor::new(&[self.scale as f32], x.device())?)?;
            let k_t = k.i((.., t, ..))?;
            let v_t = v.i((.., t, ..))?;
            let g_t = g.i((.., t, ..))?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta_t = beta.i((.., t, ..))?.unsqueeze(D::Minus1)?;

            state = state.broadcast_mul(&g_t)?;
            let kv_mem = state.broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            let delta = v_t.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_t)?;
            state = state.broadcast_add(&k_t.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?)?;
            
            outputs.push(state.broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?.sum_keepdim(D::Minus2)?);
        }
        self.recurrent_state_cache = Some(state);
        
        let out = Tensor::cat(&outputs, 2)?.transpose(1, 2)?.reshape((bs, sl, ()))?;
        let gated_out = self.norm.forward(&out.to_dtype(x.dtype())?, reg)?.broadcast_mul(&z.to_dtype(x.dtype())?.silu()?)?;
        self.out_proj.forward(&gated_out, reg)
    }
    pub fn clear_cache(&mut self) { self.conv_state_cache = None; self.recurrent_state_cache = None; }
}

pub struct QAttention { 
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear, 
    pub q_norm: QRmsNorm, pub k_norm: QRmsNorm, 
    pub nh: usize, pub nkv: usize, pub hd: usize, 
    scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
    pub attn_output_gate: bool,
}
impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        let q_out_dim = if config.attn_output_gate { config.num_attention_heads * config.head_dim * 2 } else { config.num_attention_heads * config.head_dim };
        Ok(Self {
            q_proj: QLinear::new(&join_name(&p, "q_proj"))?, k_proj: QLinear::new(&join_name(&p, "k_proj"))?, v_proj: QLinear::new(&join_name(&p, "v_proj"))?, o_proj: QLinear::new(&join_name(&p, "o_proj"))?,
            q_norm: QRmsNorm::new(&join_name(&p, "q_norm"), config.rms_norm_eps, 0.0)?, k_norm: QRmsNorm::new(&join_name(&p, "k_norm"), config.rms_norm_eps, 0.0)?,
            nh: config.num_attention_heads, nkv: config.num_key_value_heads, hd: config.head_dim, scaling: 1.0 / (config.head_dim as f64).sqrt(), kv_cache: None,
            attn_output_gate: config.attn_output_gate,
        })
    }
    pub fn clear_vram(&mut self) { self.q_proj.clear_vram(); self.k_proj.clear_vram(); self.v_proj.clear_vram(); self.o_proj.clear_vram(); self.q_norm.clear_vram(); self.k_norm.clear_vram(); self.kv_cache = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.q_proj.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        self.q_proj.load_to_vram(reg, dev)?; self.k_proj.load_to_vram(reg, dev)?; self.v_proj.load_to_vram(reg, dev)?;
        self.o_proj.load_to_vram(reg, dev)?; self.q_norm.load_to_vram(reg, dev)?; self.k_norm.load_to_vram(reg, dev)?;
        Ok(())
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, reg: &QuantizedRegistry, interleaved: bool) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q_raw = self.q_proj.forward(xs, reg)?; 
        
        let (q_states, gate) = if self.attn_output_gate {
            let q_gate = q_raw.reshape((b_sz, q_len, self.nh, self.hd, 2))?;
            let qs = q_gate.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
            let gt = q_gate.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
            (qs, Some(gt))
        } else {
            (q_raw.reshape((b_sz, q_len, self.nh, self.hd))?, None)
        };

        let key_states = self.k_proj.forward(xs, reg)?.reshape((b_sz, q_len, self.nkv, self.hd))?;
        let value_states = self.v_proj.forward(xs, reg)?.reshape((b_sz, q_len, self.nkv, self.hd))?;
        let q_states = self.q_norm.forward(&q_states, reg)?;
        let key_states = self.k_norm.forward(&key_states, reg)?;
        let q_states = q_states.transpose(1, 2)?; 
        let key_states = key_states.transpose(1, 2)?;   
        let (q_states, key_states) = crate::position_embed::rope::apply_rotary_pos_emb_partial(&q_states, &key_states, cos, sin, interleaved, true)?;
        let (key_states, value_states) = match self.kv_cache.take() {
            None => (key_states, value_states.transpose(1, 2)?),
            Some((prev_k, prev_v)) => (Tensor::cat(&[prev_k, key_states], 2)?, Tensor::cat(&[prev_v, value_states.transpose(1, 2)?], 2)?)
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let y = crate::models::common::eager_attention_forward(&q_states.to_dtype(DType::F32)?, &key_states.to_dtype(DType::F32)?, &value_states.to_dtype(DType::F32)?, Some(self.nh / self.nkv), attention_mask, self.scaling)?; 
        
        let y = if let Some(gate) = gate {
            let gt = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?;
            y.broadcast_mul(&gt)?
        } else {
            y
        };
        
        let y = y.reshape((b_sz, q_len, ()))?.contiguous()?;
        self.o_proj.forward(&y.to_dtype(DType::F16)?, reg)
    }
}

pub struct QDecoderLayer { pub idx: usize, pub self_attn: Option<QAttention>, pub linear_attn: Option<QGatedDeltaNet>, pub mlp_gate: QLinear, pub mlp_up: QLinear, pub mlp_down: QLinear, pub in_norm: QRmsNorm, pub post_norm: QRmsNorm, pub mrope_interleaved: bool }
impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let p = vb.prefix();
        let (sa, la) = if config.layer_types[idx] == "linear_attention" { (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?)) } else { (Some(QAttention::new(vb.pp("self_attn"), config)?), None) };
        Ok(Self { 
            idx, self_attn: sa, linear_attn: la, mlp_gate: QLinear::new(&join_name(&p, "mlp.gate_proj"))?, mlp_up: QLinear::new(&join_name(&p, "mlp.up_proj"))?,
            mlp_down: QLinear::new(&join_name(&p, "mlp.down_proj"))?, in_norm: QRmsNorm::new(&join_name(&p, "input_layernorm"), config.rms_norm_eps, 0.0)?, post_norm: QRmsNorm::new(&join_name(&p, "post_attention_layernorm"), config.rms_norm_eps, 0.0)?,
            mrope_interleaved: config.rope_parameters.mrope_interleaved
        })
    }
    pub fn clear_vram(&mut self) { self.in_norm.clear_vram(); if let Some(ref mut sa) = self.self_attn { sa.clear_vram(); } if let Some(ref mut la) = self.linear_attn { la.clear_vram(); } self.post_norm.clear_vram(); self.mlp_gate.clear_vram(); self.mlp_up.clear_vram(); self.mlp_down.clear_vram(); }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.in_norm.load_to_vram(reg, dev)?; if let Some(ref mut sa) = self.self_attn { sa.load_to_vram(reg, dev)?; } if let Some(ref mut la) = self.linear_attn { la.load_to_vram(reg, dev)?; }
        self.post_norm.load_to_vram(reg, dev)?; self.mlp_gate.load_to_vram(reg, dev)?; self.mlp_up.load_to_vram(reg, dev)?; self.mlp_down.load_to_vram(reg, dev)?; Ok(())
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone(); let h = self.in_norm.forward(x, reg)?;
        let h = if let Some(ref mut sa) = self.self_attn { sa.forward(&h, cos, sin, mask, reg, self.mrope_interleaved)? } else if let Some(ref mut la) = self.linear_attn { la.forward(&h, mask, reg)? } else { h };
        let h = (h + residual)?; let residual = h.clone();
        let h = self.post_norm.forward(&h, reg)?;
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?;
        let mlp_out = self.mlp_down.forward(&(gate * self.mlp_up.forward(&h, reg)?)?, reg)?;
        let res = (mlp_out + residual)?;
        Ok(res)
    }
}

pub struct QEmbedding { pub full_name: String, pub persistent_weight: Arc<RwLock<Option<Tensor>>>, pub vocab_size: usize, pub hidden_size: usize }
impl QEmbedding {
    pub fn new(name: &str, vocab_size: usize, hidden_size: usize) -> Self { Self { full_name: name.to_string(), persistent_weight: Arc::new(RwLock::new(None)), vocab_size, hidden_size } }
    pub fn clear_vram(&mut self) { *self.persistent_weight.write().unwrap() = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        let t = reg.get_weight_tensor(&self.full_name, "weight")?;
        *self.persistent_weight.write().unwrap() = Some(prepare_weight(&t, dev)?.reshape((self.vocab_size, self.hidden_size))?); Ok(())
    }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let w = {
            let cache = self.persistent_weight.read().unwrap();
            if let Some(pw) = &*cache { Some(pw.clone()) } else { None }
        };
        let w = if let Some(w) = w { w } 
                else { 
                    let mut cache = self.persistent_weight.write().unwrap();
                    let weight = prepare_weight(&reg.get_weight_tensor(&self.full_name, "weight")?, dev)?.reshape((self.vocab_size, self.hidden_size))?;
                    *cache = Some(weight.clone());
                    weight
                };
        let emb = w.index_select(&ids.flatten_all()?.to_dtype(DType::U32)?, 0)?.reshape((b, s, ()))?.to_dtype(DType::F16)?;
        Ok(emb)
    }
}

pub struct QTextModel { pub embed: QEmbedding, pub layers: Vec<QDecoderLayer>, pub norm: QRmsNorm, pub rotary: Qwen3_5TextRotaryEmbedding, pub mrope: Vec<usize>, pub mrope_interleaved: bool }
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        Ok(Self { 
            embed: QEmbedding::new(&join_name(&vb.prefix(), "embed_tokens"), config.vocab_size, config.hidden_size), 
            layers, 
            norm: QRmsNorm::new(&join_name(&vb.prefix(), "norm"), config.rms_norm_eps, 0.0)?, 
            rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, config.rope_parameters.rope_theta), 
            mrope: config.rope_parameters.mrope_section.clone(),
            mrope_interleaved: config.rope_parameters.mrope_interleaved
        })
    }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { self.embed.load_to_vram(reg, dev)?; for layer in self.layers.iter_mut() { layer.load_to_vram(reg, dev)?; } self.norm.load_to_vram(reg, dev)?; Ok(()) }
}

pub struct QuantizedQwen3_5Model { pub model: QTextModel, pub head: QLinear, pub tie: bool, pub mrope_interleaved: bool, pub hidden_size: f64 }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let model = QTextModel::new(vb.pp("model.language_model"), &config.text_config)?;
        let head_name = if config.tie_word_embeddings { "model.language_model.embed_tokens" } else { "model.language_model.lm_head" };
        Ok(Self { model, head: QLinear::new(head_name)?, tie: config.tie_word_embeddings, mrope_interleaved: config.text_config.rope_parameters.mrope_interleaved, hidden_size: config.text_config.hidden_size as f64 })
    }    
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { 
        self.model.load_all_to_vram(reg, dev)?; self.head.load_to_vram(reg, dev)?; Ok(()) 
    }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } if let Some(ref mut la) = l.linear_attn { la.recurrent_state_cache = None; la.conv_state_cache = None; } } }
}

#[derive(Clone)]
pub enum LayerContext { Attention { k: Tensor, v: Tensor }, DeltaNet { state: Tensor, conv: Tensor } }

fn join_name(p: &str, n: &str) -> String {
    if p.is_empty() { n.to_string() } else if p.ends_with('.') { format!("{}{}", p, n) } else { format!("{}.{}", p, n) }
}
