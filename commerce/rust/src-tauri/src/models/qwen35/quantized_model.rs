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

fn dequantize_q2(t: &Tensor, dev: &Device) -> Result<Tensor> {
    if t.rank() == 2 && t.dim(1)? == 10 && t.dtype() == DType::U8 {
        let num_blocks = t.dim(0)?;
        let scale_bytes = t.narrow(1, 0, 2)?; 
        let data_bytes = t.narrow(1, 2, 8)?;  
        
        let scales = Tensor::from_raw_buffer(&scale_bytes.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?, DType::F16, &[num_blocks], dev)?;

        let mut parts = vec![];
        let d_f32 = data_bytes.to_dtype(DType::F32)?;
        for i in 0..4 {
            let div = (1 << (i * 2)) as f64;
            let vals = d_f32.affine(1.0 / div, 0.0)?;
            let floor_val = vals.affine(1.0 / 4.0, 0.0)?.floor()?.affine(4.0, 0.0)?;
            let bit_vals = vals.sub(&floor_val)?;
            parts.push(bit_vals.to_dtype(DType::F16)?);
        }
        let combined = Tensor::stack(&parts, D::Minus1)?.reshape((num_blocks, 32))?;
        
        let offset = Tensor::new(1.5f32, dev)?.to_dtype(DType::F16)?;
        Ok(combined.broadcast_sub(&offset)?.broadcast_mul(&scales.unsqueeze(D::Minus1)?)?)
    } else {
        Ok(t.to_dtype(DType::F16)?)
    }
}

pub struct QRmsNorm { full_name: String, eps: f64, persistent_weight: Option<Tensor> }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64) -> Result<Self> { Ok(Self { full_name: name.to_string(), eps, persistent_weight: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let w = reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?.to_dtype(DType::F32)?;
        let w = w.affine(1.0, 1.0)?.to_dtype(DType::F16)?;
        self.persistent_weight = Some(w); Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } 
                else { 
                    let w = reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?.to_dtype(DType::F32)?;
                    w.affine(1.0, 1.0)?.to_dtype(DType::F16)?
                };
        
        let x_f32 = x.to_dtype(DType::F32)?;
        let original_shape = x.dims().to_vec();
        let last_dim = x.dim(D::Minus1)?;
        let w_size = w.elem_count();

        let norm_ = x_f32.powf(2.0)?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm_)?;
        
        let res = if w_size == last_dim {
            x_normed.broadcast_mul(&w.to_dtype(DType::F32)?)?
        } else {
            let reshaped = x_normed.reshape(((), w_size))?;
            let r = reshaped.broadcast_mul(&w.to_dtype(DType::F32)?.reshape((1, w_size))?)?;
            r.reshape(original_shape)?
        };
        Ok(res.to_dtype(DType::F16)?)
    }
}

pub struct QLinear { full_name: String, persistent_weight: Option<Tensor>, persistent_bias: Option<Tensor> }
impl QLinear {
    pub fn new(name: &str) -> Result<Self> { Ok(Self { full_name: name.to_string(), persistent_weight: None, persistent_bias: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; self.persistent_bias = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let name = &self.full_name;
        let t = reg.get_q_tensor(name, "weight")?.to_device(dev)?;
        self.persistent_weight = Some(dequantize_q2(&t, dev)?.flatten_all()?);
        if let Ok(b) = reg.get_q_tensor(name, "bias") { self.persistent_bias = Some(b.to_device(dev)?.to_dtype(DType::F16)?); }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device(); let name = &self.full_name;
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } 
                else { dequantize_q2(&reg.get_q_tensor(name, "weight")?.to_device(dev)?, dev)?.flatten_all()? };
        
        let in_features = x.dim(D::Minus1)?;
        let out_features = w.elem_count() / in_features;
        let w = w.reshape((out_features, in_features))?.t()?;
        
        let res = x.contiguous()?.broadcast_matmul(&w.contiguous()?)?;
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
    in_proj_qkv: QLinear, in_proj_z: QLinear, in_proj_b: QLinear, in_proj_a: QLinear, 
    out_proj: QLinear, norm: QRmsNorm, 
    pub nk: usize, pub dk: usize, pub nv: usize, pub dv: usize,
    pub conv1d: Option<candle_nn::Conv1d>,
    pub dt_bias: Option<Tensor>, pub a_log: Option<Tensor>,
    pub recurrent_state_cache: Option<Tensor>, pub conv_state_cache: Option<Tensor> 
}
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        let nk = config.linear_num_key_heads; let dk = config.linear_key_head_dim;
        let nv = config.linear_num_value_heads; let dv = config.linear_value_head_dim;
        Ok(Self { 
            in_proj_qkv: QLinear::new(&join_name(&p, "in_proj_qkv"))?,
            in_proj_z: QLinear::new(&join_name(&p, "in_proj_z"))?,
            in_proj_b: QLinear::new(&join_name(&p, "in_proj_b"))?,
            in_proj_a: QLinear::new(&join_name(&p, "in_proj_a"))?,
            out_proj: QLinear::new(&join_name(&p, "out_proj"))?, 
            norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps)?, 
            nk, dk, nv, dv,
            conv1d: None, dt_bias: None, a_log: None,
            recurrent_state_cache: None, conv_state_cache: None 
        })
    }
    pub fn clear_vram(&mut self) { 
        self.in_proj_qkv.clear_vram(); self.in_proj_z.clear_vram(); self.in_proj_b.clear_vram(); self.in_proj_a.clear_vram();
        self.out_proj.clear_vram(); self.norm.clear_vram(); 
        self.dt_bias = None; self.a_log = None; self.conv1d = None;
    }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { 
        self.in_proj_qkv.load_to_vram(reg, dev)?; self.in_proj_z.load_to_vram(reg, dev)?; self.in_proj_b.load_to_vram(reg, dev)?; self.in_proj_a.load_to_vram(reg, dev)?;
        self.out_proj.load_to_vram(reg, dev)?; self.norm.load_to_vram(reg, dev)?; 
        let p = self.norm.full_name.replace(".norm", "");
        if let Ok(dt) = reg.get_tensor(&format!("{}.dt_bias", p)) { self.dt_bias = Some(dt.to_device(dev)?.to_dtype(DType::F32)?); }
        if let Ok(a) = reg.get_tensor(&format!("{}.A_log", p)) { self.a_log = Some(a.to_device(dev)?.to_dtype(DType::F32)?); }
        if let Ok(cw) = reg.get_q_tensor(&format!("{}.conv1d", p), "weight") {
            let cw = cw.to_device(dev)?;
            let dequant = dequantize_q2(&cw, dev)?;
            let channels = dequant.elem_count() / 4;
            let c_w = dequant.reshape((channels, 1, 4))?;
            let vb = vb_from_weights(c_w, None, dev)?;
            self.conv1d = Some(crate::models::common::get_conv1d(vb, channels, channels, 4, 3, 1, 1, channels, false)?);
        }
        Ok(()) 
    }

    pub fn forward(&mut self, x: &Tensor, _mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let (nk, dk, nv, dv) = (self.nk, self.dk, self.nv, self.dv);

        let mut mixed_qkv = self.in_proj_qkv.forward(x, reg)?.transpose(1, 2)?;
        let z = self.in_proj_z.forward(x, reg)?;
        let b = self.in_proj_b.forward(x, reg)?;
        let a = self.in_proj_a.forward(x, reg)?;
        
        if let Some(ref conv) = self.conv1d {
            let kernel_size = 4;
            let state_len = kernel_size - 1;
            if sl == 1 && self.conv_state_cache.is_some() {
                let conv_state = self.conv_state_cache.as_ref().unwrap().to_device(x.device())?;
                let conv_state_new = Tensor::cat(&[&conv_state, &mixed_qkv], D::Minus1)?;
                self.conv_state_cache = Some(conv_state_new.narrow(D::Minus1, sl, state_len)?.detach());
                let conv_out = crate::models::common::conv1d_depthwise(&conv_state_new, conv.weight(), conv.bias())?;
                mixed_qkv = conv_out.narrow(D::Minus1, conv_out.dim(D::Minus1)? - sl, sl)?.silu()?;
            } else {
                let conv_in = mixed_qkv.pad_with_zeros(D::Minus1, state_len, state_len)?;
                mixed_qkv = crate::models::common::conv1d_depthwise(&conv_in, conv.weight(), conv.bias())?;
                mixed_qkv = mixed_qkv.narrow(D::Minus1, state_len, sl)?.silu()?;
                self.conv_state_cache = Some(conv_in.narrow(D::Minus1, sl, state_len)?.detach());
            }
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let q = mixed_qkv.narrow(D::Minus1, 0, nk * dk)?.reshape((bs, sl, nk, dk))?;
        let k = mixed_qkv.narrow(D::Minus1, nk * dk, nk * dk)?.reshape((bs, sl, nk, dk))?;
        let v = mixed_qkv.narrow(D::Minus1, nk * dk * 2, nv * dv)?.reshape((bs, sl, nv, dv))?;

        let q = crate::utils::tensor_utils::l2_normalize(&q, 3)?;
        let k = crate::utils::tensor_utils::l2_normalize(&k, 3)?;

        let scale = 1.0 / (dk as f64).sqrt();
        let mut query = q.affine(scale, 0.0)?.to_dtype(DType::F32)?;
        let mut key = k.to_dtype(DType::F32)?;
        let value = v.to_dtype(DType::F32)?;
        let beta = candle_nn::ops::sigmoid(&b)?.to_dtype(DType::F32)?;
        
        let a_plus_bias = softplus(
            &a.to_dtype(DType::F32)?
                .broadcast_add(&self.dt_bias.as_ref().unwrap_or(&Tensor::zeros((nv,), DType::F32, x.device())?).to_dtype(DType::F32)?)?
        )?;
        let g = self.a_log.as_ref().unwrap_or(&Tensor::zeros((nv,), DType::F32, x.device())?).to_dtype(DType::F32)?
            .exp()?
            .affine(-1.0, 0.0)?
            .broadcast_mul(&a_plus_bias)?;
        let g = g.to_dtype(DType::F32)?;

        if nv / nk > 1 {
            query = crate::utils::tensor_utils::repeat_interleave(&query, nv / nk, 2)?;
            key = crate::utils::tensor_utils::repeat_interleave(&key, nv / nk, 2)?;
        }

        let mut state = self.recurrent_state_cache.take()
            .unwrap_or_else(|| Tensor::zeros((bs, nv, dk, dv), DType::F32, x.device()).unwrap())
            .to_device(x.device())?
            .to_dtype(DType::F32)?;
        
        let mut outputs = vec![];
        for t in 0..sl {
            let q_i = query.i((.., t, ..))?;
            let k_i = key.i((.., t, ..))?;
            let v_i = value.i((.., t, ..))?;
            let g_i = g.i((.., t, ..))?.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta_i = beta.i((.., t, ..))?.unsqueeze(D::Minus1)?;

            state = state.broadcast_mul(&g_i)?;
            let kv_mem = state.broadcast_mul(&k_i.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            state = state.broadcast_add(&k_i.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?)?;
            
            let out_i = state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?.sum_keepdim(D::Minus2)?;
            outputs.push(out_i.to_dtype(DType::F16)?);
        }

        let out = Tensor::cat(&outputs, 1)?.reshape((bs * sl, nv, dv))?;
        let z_flat = z.reshape((bs * sl, nv, dv))?;
        self.recurrent_state_cache = Some(state.detach());
        let gated_out = self.norm.forward(&out, reg)?;
        let gated_out = gated_out.broadcast_mul(&z_flat.silu()?.to_dtype(DType::F16)?)?;
        self.out_proj.forward(&gated_out.reshape((bs, sl, nv * dv))?, reg)
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
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_chunk = self.q_proj.forward(xs, reg)?
            .reshape((b_sz, q_len, self.nh, self.hd * 2))?
            .chunk(2, D::Minus1)?;
        
        let query_states = query_chunk[0].reshape((b_sz, q_len, self.nh, self.hd))?;
        let gate = query_chunk[1].reshape((b_sz, q_len, self.nh * self.hd))?;

        let query_states = self.q_norm.forward(&query_states, reg)?.transpose(1, 2)?;
        
        let k_raw = self.k_proj.forward(xs, reg)?;
        let key_states = k_raw.reshape((b_sz, q_len, self.nkv, self.hd))?;
        let key_states = self.k_norm.forward(&key_states, reg)?.transpose(1, 2)?;
        
        let v_raw = self.v_proj.forward(xs, reg)?;
        let value_states = v_raw.reshape((b_sz, q_len, self.nkv, self.hd))?.transpose(1, 2)?;

        let (query_states, key_states) = glm_asr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let attn_output = crate::models::common::eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.nh / self.nkv),
            attention_mask,
            self.scaling,
        )?;

        let attn_output = attn_output.reshape((b_sz, q_len, self.nh * self.hd))?.contiguous()?;
        let attn_output = attn_output.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?.to_dtype(DType::F16)?)?;
        self.o_proj.forward(&attn_output, reg)
    }
}

pub struct QDecoderLayer { 
    pub idx: usize,
    pub self_attn: Option<QAttention>, 
    pub linear_attn: Option<QGatedDeltaNet>, 
    mlp_gate: QLinear,
    mlp_up: QLinear,
    mlp_down: QLinear,
    in_norm: QRmsNorm, 
    post_norm: QRmsNorm 
}

impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let p = vb.prefix();
        let lt = config.layer_types[idx].clone();
        
        let (sa, la) = if lt == "linear_attention" {
            (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?))
        } else {
            (Some(QAttention::new(vb.pp("self_attn"), config)?), None)
        };

        Ok(Self { 
            idx,
            self_attn: sa, 
            linear_attn: la, 
            mlp_gate: QLinear::new(&join_name(&p, "mlp.gate_proj"))?,
            mlp_up: QLinear::new(&join_name(&p, "mlp.up_proj"))?,
            mlp_down: QLinear::new(&join_name(&p, "mlp.down_proj"))?,
            in_norm: QRmsNorm::new(&join_name(&p, "input_layernorm"), config.rms_norm_eps)?, 
            post_norm: QRmsNorm::new(&join_name(&p, "post_attention_layernorm"), config.rms_norm_eps)? 
        })
    }

    pub fn clear_vram(&mut self) {
        self.in_norm.clear_vram();
        if let Some(ref mut sa) = self.self_attn { sa.clear_vram(); }
        if let Some(ref mut la) = self.linear_attn { la.clear_vram(); }
        self.post_norm.clear_vram();
        self.mlp_gate.clear_vram();
        self.mlp_up.clear_vram();
        self.mlp_down.clear_vram();
    }

    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.in_norm.load_to_vram(reg, dev)?;
        if let Some(ref mut sa) = self.self_attn { sa.load_to_vram(reg, dev)?; }
        if let Some(ref mut la) = self.linear_attn { la.load_to_vram(reg, dev)?; }
        self.post_norm.load_to_vram(reg, dev)?;
        self.mlp_gate.load_to_vram(reg, dev)?;
        self.mlp_up.load_to_vram(reg, dev)?;
        self.mlp_down.load_to_vram(reg, dev)?;
        Ok(())
    }

    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.in_norm.forward(x, reg)?;
        
        let h = if let Some(ref mut sa) = self.self_attn {
            sa.forward(&h, cos, sin, mask, reg)?
        } else if let Some(ref mut la) = self.linear_attn {
            la.forward(&h, mask, reg)?
        } else {
            h
        };
        
        let h = (h + residual)?;
        let residual = h.clone();
        let h = self.post_norm.forward(&h, reg)?;
        
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?;
        let up = self.mlp_up.forward(&h, reg)?;
        let mlp_out = self.mlp_down.forward(&(gate * up)?, reg)?;
        
        Ok((mlp_out + residual)?)
    }
}

pub struct QEmbedding { full_name: String, persistent_weight: Option<Tensor>, vocab_size: usize, hidden_size: usize }
impl QEmbedding {
    pub fn new(name: &str, vocab_size: usize, hidden_size: usize) -> Self { Self { full_name: name.to_string(), persistent_weight: None, vocab_size, hidden_size } }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let name = &self.full_name;
        let t = reg.get_q_tensor(name, "weight")?.to_device(dev)?;
        let dequant = dequantize_q2(&t, dev)?;
        self.persistent_weight = Some(dequant.reshape((self.vocab_size, self.hidden_size))?);
        Ok(())
    }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } else { 
            let t = reg.get_q_tensor(&self.full_name, "weight")?.to_device(dev)?;
            dequantize_q2(&t, dev)?.reshape((self.vocab_size, self.hidden_size))?
        };
        let embeds = w.index_select(&ids.flatten_all()?.to_dtype(DType::U32)?, 0)?.reshape((b, s, ()))?.to_device(dev)?.to_dtype(DType::F16)?;
        Ok(embeds)
    }
}

pub struct QTextModel { pub embed: QEmbedding, pub layers: Vec<QDecoderLayer>, pub norm: QRmsNorm, pub rotary: Qwen3_5TextRotaryEmbedding, pub mrope: Vec<usize> }
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let p = vb.prefix();
        let rope_theta = 1000000.0;
        Ok(Self { 
            embed: QEmbedding::new(&join_name(&p, "embed_tokens"), config.vocab_size, config.hidden_size), 
            layers, 
            norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps)?, 
            rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, rope_theta), 
            mrope: config.rope_parameters.mrope_section.clone() 
        })
    }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.embed.load_to_vram(reg, dev)?;
        for layer in self.layers.iter_mut() { layer.load_to_vram(reg, dev)?; }
        self.norm.load_to_vram(reg, dev)?;
        Ok(())
    }
    pub fn forward(&mut self, _ids: &Tensor, _pos: &Tensor, _offset: usize, _reg: &QuantizedRegistry) -> Result<Tensor> { anyhow::bail!("Not used") }
}

pub struct QuantizedQwen3_5Model { pub model: QTextModel, pub head: QLinear, tie: bool }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let p = vb.prefix();
        let head_name = if config.tie_word_embeddings {
            join_name(&p, "model.language_model.embed_tokens")
        } else {
            join_name(&p, "model.language_model.lm_head")
        };
        Ok(Self { 
            model: QTextModel::new(vb.pp("model.language_model"), &config.text_config)?, 
            head: QLinear::new(&head_name)?,
            tie: config.tie_word_embeddings
        })
    }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.model.load_all_to_vram(reg, dev)?;
        if self.tie {
            self.head.persistent_weight = self.model.embed.persistent_weight.clone();
        } else {
            if let Err(_) = self.head.load_to_vram(reg, dev) {
                 let name = self.model.embed.full_name.clone();
                 self.head = QLinear::new(&name)?;
                 self.head.load_to_vram(reg, dev)?;
            }
        }
        Ok(())
    }
    pub fn forward(&mut self, _ids: &Tensor, _offset: usize, _reg: &QuantizedRegistry, _img: Option<&Tensor>) -> Result<Tensor> { anyhow::bail!("Not used") }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } if let Some(ref mut la) = l.linear_attn { la.recurrent_state_cache = None; la.conv_state_cache = None; } } }
}

use memmap2::MmapMut;
use std::fs::OpenOptions;
use candle_core::Shape;

#[derive(Clone)]
pub enum LayerContext { Attention { k: Tensor, v: Tensor }, DeltaNet { state: Tensor } }
pub struct KVStateManager { 
    mmap: Option<MmapMut>, 
    layer_stride: usize,
    memory_cache: Vec<Option<LayerContext>>,
    use_memory: bool
}

impl KVStateManager {
    pub fn new(num_layers: usize, use_memory: bool) -> Result<Self> {
        let layer_stride = 32 * 1024 * 1024; // 32MB
        let (mmap, memory_cache) = if use_memory {
            (None, vec![None; num_layers])
        } else {
            let path = crate::utils::paths::get_kv_dir(None).join("kv_cache_pool.bin");
            let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
            file.set_len((num_layers * layer_stride) as u64)?;
            let m = unsafe { MmapMut::map_mut(&file)? };
            (Some(m), vec![])
        };
        Ok(Self { mmap, layer_stride, memory_cache, use_memory })
    }
    pub fn save_layer_context(&mut self, layer_idx: usize, ctx: LayerContext) -> Result<()> {
        if self.use_memory {
            self.memory_cache[layer_idx] = Some(ctx);
            return Ok(());
        }
        
        let offset = layer_idx * self.layer_stride;
        match &ctx {
            LayerContext::Attention { k, v } => { 
                self.write_tensor(offset, k, DType::F16)?; 
                self.write_tensor(offset + (k.elem_count() * 2), v, DType::F16)?; 
            },
            LayerContext::DeltaNet { state } => { 
                self.write_tensor(offset, state, DType::F32)?; 
            }
        }
        Ok(())
    }
    pub fn load_layer_context(&self, layer_idx: usize, lt: &str, shape_k: &Shape, shape_v: &Shape, dev: &Device) -> Result<LayerContext> {
        if self.use_memory {
            if let Some(ctx) = &self.memory_cache[layer_idx] {
                return match ctx {
                    LayerContext::Attention { k, v } => Ok(LayerContext::Attention { k: k.to_device(dev)?, v: v.to_device(dev)? }),
                    LayerContext::DeltaNet { state } => Ok(LayerContext::DeltaNet { state: state.to_device(dev)? }),
                };
            }
            anyhow::bail!("Layer context not found in memory");
        }

        let offset = layer_idx * self.layer_stride;
        if lt == "linear_attention" { 
            Ok(LayerContext::DeltaNet { state: self.read_tensor(offset, shape_k, DType::F32, dev)? }) 
        } else { 
            Ok(LayerContext::Attention { 
                k: self.read_tensor(offset, shape_k, DType::F16, dev)?, 
                v: self.read_tensor(offset + (shape_k.elem_count() * 2), shape_v, DType::F16, dev)? 
            }) 
        }
    }
    fn write_tensor(&mut self, offset: usize, t: &Tensor, dtype: DType) -> Result<()> {
        let mmap = self.mmap.as_mut().ok_or_else(|| anyhow!("Mmap not initialized"))?;
        let t = t.detach().to_device(&Device::Cpu)?.to_dtype(dtype)?;
        let bytes = match dtype {
            DType::F32 => {
                let data = t.flatten_all()?.to_vec1::<f32>()?;
                let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                b.to_vec()
            },
            _ => {
                let data = t.flatten_all()?.to_vec1::<f16>()?;
                let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
                b.to_vec()
            }
        };
        mmap[offset..offset + bytes.len()].copy_from_slice(&bytes); Ok(())
    }
    fn read_tensor(&self, offset: usize, shape: &Shape, dtype: DType, dev: &Device) -> Result<Tensor> {
        let mmap = self.mmap.as_ref().ok_or_else(|| anyhow!("Mmap not initialized"))?;
        let byte_count = shape.elem_count() * (if dtype == DType::F32 { 4 } else { 2 });
        let raw_slice = &mmap[offset..offset + byte_count];
        Ok(Tensor::from_raw_buffer(raw_slice, dtype, shape.dims(), dev)?)
    }
}
