use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use candle_core::safetensors::MmapedSafetensors;
use half::f16;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::softplus,
    },
    position_embed::rope::{
        Qwen3_5TextRotaryEmbedding, glm_interleaved_apply_rotary_pos_emb,
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
    if t.dtype() == DType::U8 && t.rank() == 2 && t.dim(1)? == 10 {
        let num_blocks = t.dim(0)?;
        let t_gpu = t.to_device(dev)?;
        let scales_u16 = t.narrow(1, 0, 2)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let scales = Tensor::from_raw_buffer(&scales_u16, DType::F16, &[num_blocks], dev)?.unsqueeze(D::Minus1)?;
        
        let data_bytes = t_gpu.narrow(1, 2, 8)?.to_dtype(DType::F16)?;
        let mut final_res = vec![];
        
        for i in 0..4 {
            let div = (1 << (i * 2)) as f64;
            let shifted = data_bytes.affine(1.0 / div, 0.0)?.floor()?;
            let floored = shifted.affine(1.0 / 4.0, 0.0)?.floor()?.affine(4.0, 0.0)?;
            let val = shifted.sub(&floored)?.affine(1.0, -1.5)?;
            final_res.push(val.broadcast_mul(&scales)?);
        }
        
        // Stack results and reshape directly to f16 to save space
        Ok(Tensor::stack(&final_res, D::Minus1)?.reshape((num_blocks, 32))?)
    } else {
        Ok(t.to_device(dev)?.to_dtype(DType::F16)?)
    }
}

pub struct QRmsNorm { pub full_name: String, eps: f64, offset: f64, persistent_weight: Arc<RwLock<Option<Tensor>>> }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64, offset: f64) -> Result<Self> { Ok(Self { full_name: name.to_string(), eps, offset, persistent_weight: Arc::new(RwLock::new(None)) }) }
    pub fn clear_vram(&mut self) { *self.persistent_weight.write().unwrap() = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        let w = reg.get_q_tensor(&self.full_name, "weight")?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
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
                    let w = reg.get_q_tensor(&self.full_name, "weight")?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
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
        let t = reg.get_q_tensor(&self.full_name, "weight")?;
        *self.persistent_weight.write().unwrap() = Some(dequantize_q2(&t, dev)?);
        if let Ok(b) = reg.get_q_tensor(&self.full_name, "bias") { 
            *self.persistent_bias.write().unwrap() = Some(b.to_device(dev)?.to_dtype(DType::F16)?); 
        }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let in_features = x.dim(D::Minus1)?;
        
        // Try to get optimized weight from cache
        let w = {
            let cache = self.persistent_weight.read().unwrap();
            if let Some(pw) = &*cache {
                if pw.rank() == 2 && pw.dim(0)? == in_features {
                    Some(pw.clone())
                } else {
                    None as Option<Tensor>
                }
            } else {
                None as Option<Tensor>
            }
        };

        let w = if let Some(w) = w {
            w
        } else {
            // Need to load or optimize
            let mut cache = self.persistent_weight.write().unwrap();
            let dequant = if let Some(pw) = &*cache {
                pw.clone() as Tensor
            } else {
                dequantize_q2(&reg.get_q_tensor(&self.full_name, "weight")?, dev)?
            };
            
            let out_features = dequant.elem_count() / in_features;
            let optimized = if dequant.rank() == 2 && dequant.dim(0)? == in_features {
                dequant
            } else {
                dequant.reshape((out_features, in_features))?.t()?.contiguous()?
            };
            *cache = Some(optimized.clone());
            optimized
        };

        let res = x.contiguous()?.broadcast_matmul(&w)?;
        
        let b = {
            let cache = self.persistent_bias.read().unwrap();
            cache.clone()
        };

        if let Some(pb) = b { 
            Ok(res.broadcast_add(&pb)?) 
        } else if let Ok(b_tensor) = reg.get_q_tensor(&self.full_name, "bias") {
            let pb = b_tensor.to_device(dev)?.to_dtype(DType::F16)?;
            *self.persistent_bias.write().unwrap() = Some(pb.clone());
            Ok(res.broadcast_add(&pb)?)
        } else { 
            Ok(res) 
        }
    }
}

fn join_name(p: &str, n: &str) -> String {
    if p.is_empty() { n.to_string() } else if p.ends_with('.') { format!("{}{}", p, n) } else { format!("{}.{}", p, n) }
}

pub struct QGatedDeltaNet { 
    pub in_proj_qkv: QLinear, pub in_proj_z: QLinear, pub in_proj_b: QLinear, pub in_proj_a: QLinear, 
    pub out_proj: QLinear, pub norm: QRmsNorm, pub nk: usize, pub dk: usize, pub nv: usize, pub dv: usize,
    pub conv1d: Option<candle_nn::Conv1d>, pub dt_bias: Option<Tensor>, pub a_log: Option<Tensor>,
    pub recurrent_state_cache: Option<Tensor>, pub conv_state_cache: Option<Tensor> 
}
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self { 
            in_proj_qkv: QLinear::new(&join_name(&p, "in_proj_qkv"))?, in_proj_z: QLinear::new(&join_name(&p, "in_proj_z"))?,
            in_proj_b: QLinear::new(&join_name(&p, "in_proj_b"))?, in_proj_a: QLinear::new(&join_name(&p, "in_proj_a"))?,
            out_proj: QLinear::new(&join_name(&p, "out_proj"))?, norm: QRmsNorm::new(&join_name(&p, "norm"), config.rms_norm_eps, 0.0)?, 
            nk: config.linear_num_key_heads, dk: config.linear_key_head_dim, nv: config.linear_num_value_heads, dv: config.linear_value_head_dim,
            conv1d: None, dt_bias: None, a_log: None, recurrent_state_cache: None, conv_state_cache: None 
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
        if let Ok(cw) = reg.get_q_tensor(&format!("{}.conv1d", p), "weight") {
            let dequant = dequantize_q2(&cw, dev)?;
            let channels = dequant.elem_count() / 4;
            let c_w = dequant.reshape((channels, 1, 4))?;
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
        let z = self.in_proj_z.forward(&x, reg)?;
        let b = self.in_proj_b.forward(&x, reg)?; 
        let a = self.in_proj_a.forward(&x, reg)?;
        
        let q = mixed_qkv.narrow(D::Minus1, 0, nk * dk)?; 
        let k = mixed_qkv.narrow(D::Minus1, nk * dk, nk * dk)?;
        let v = mixed_qkv.narrow(D::Minus1, nk * dk * 2, nv * dv)?;
        
        let (q_out, k_out, v_out) = if let Some(ref conv) = self.conv1d {
            let q_c = q.reshape((bs, sl, nk, dk))?.transpose(1, 2)?; 
            let k_c = k.reshape((bs, sl, nk, dk))?.transpose(1, 2)?; 
            let v_c = v.reshape((bs, sl, nv, dv))?.transpose(1, 2)?;
            
            let mut qkv_conv = Tensor::cat(&[&q_c, &k_c, &v_c], 1)? // (bs, 48, sl, 128)
                .transpose(2, 3)? // (bs, 48, 128, sl)
                .reshape((bs, (), sl))? // (bs, 6144, sl)
                .to_dtype(DType::F32)?;
            
            if sl == 1 && self.conv_state_cache.is_some() {
                let conv_state = self.conv_state_cache.as_ref().unwrap().to_device(x.device())?.to_dtype(DType::F32)?;
                let conv_state_new = Tensor::cat(&[&conv_state, &qkv_conv], D::Minus1)?;
                self.conv_state_cache = Some(conv_state_new.narrow(D::Minus1, 1, 3)?.detach());
                let conv_out = crate::models::common::conv1d_depthwise(&conv_state_new, &conv.weight().to_dtype(DType::F32)?, conv.bias().map(|b| b.to_dtype(DType::F32)).transpose()?.as_ref())?;
                qkv_conv = conv_out.narrow(D::Minus1, conv_out.dim(D::Minus1)? - 1, 1)?.silu()?;
            } else {
                let conv_in = qkv_conv.pad_with_zeros(D::Minus1, 3, 0)?;
                qkv_conv = crate::models::common::conv1d_depthwise(&conv_in, &conv.weight().to_dtype(DType::F32)?, conv.bias().map(|b| b.to_dtype(DType::F32)).transpose()?.as_ref())?;
                qkv_conv = qkv_conv.narrow(D::Minus1, 0, sl)?.silu()?;
                self.conv_state_cache = Some(conv_in.narrow(D::Minus1, sl, 3)?.detach());
            }
            
            let qkv_conv = qkv_conv.reshape((bs, (), sl))?.to_dtype(DType::F16)?;
            let channels_per_head = nk + nk + nv;
            let total_qkv = qkv_conv.reshape((bs, channels_per_head, dk, sl))?.transpose(2, 3)?; // (bs, 48, sl, 128)
            
            let q_new = total_qkv.narrow(1, 0, nk)?.transpose(1, 2)?; // (bs, sl, nk, dk)
            let k_new = total_qkv.narrow(1, nk, nk)?.transpose(1, 2)?; // (bs, sl, nk, dk)
            let v_new = total_qkv.narrow(1, nk * 2, nv)?.transpose(1, 2)?; // (bs, sl, nv, dv)
            
            (q_new, k_new, v_new)
        } else { 
            (q.reshape((bs, sl, nk, dk))?, k.reshape((bs, sl, nk, dk))?, v.reshape((bs, sl, nv, dv))?) 
        };
        
        let q_out = crate::utils::tensor_utils::l2_normalize(&q_out, 3)?; 
        let k_out = crate::utils::tensor_utils::l2_normalize(&k_out, 3)?;
        
        let mut query = q_out.affine(1.0 / (dk as f64).sqrt(), 0.0)?.to_dtype(DType::F32)?;
        let mut key = k_out.to_dtype(DType::F32)?; 
        let value = v_out.to_dtype(DType::F32)?; 
        let beta = candle_nn::ops::sigmoid(&b)?.to_dtype(DType::F32)?;
        
        let a_plus_bias = softplus(&a.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias.as_ref().unwrap_or(&Tensor::zeros((nv,), DType::F32, x.device())?).to_dtype(DType::F32)?)?)?;
        let g = self.a_log.as_ref().unwrap_or(&Tensor::zeros((nv,), DType::F32, x.device())?).to_dtype(DType::F32)?.exp()?.affine(-1.0, 0.0)?.broadcast_mul(&a_plus_bias)?;
        
        if nv / nk > 1 { 
            query = crate::utils::tensor_utils::repeat_interleave(&query, nv / nk, 2)?; 
            key = crate::utils::tensor_utils::repeat_interleave(&key, nv / nk, 2)?; 
        }
        
        let mut state = self.recurrent_state_cache.take().unwrap_or_else(|| Tensor::zeros((bs, nv, dk, dv), DType::F32, x.device()).unwrap()).to_device(x.device())?.to_dtype(DType::F32)?;
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
            
            outputs.push(state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?.sum_keepdim(D::Minus2)?);
        }
        
        self.recurrent_state_cache = Some(state.detach());
        
        let out = Tensor::cat(&outputs, 2)?.transpose(1, 2)?.reshape((bs * sl, nv, dv))?; 
        let z_flat = z.reshape((bs * sl, nv, dv))?.to_dtype(DType::F32)?;
        
        let gated_out = self.norm.forward(&out, reg)?.to_dtype(DType::F32)?.broadcast_mul(&z_flat.silu()?)?;
        self.out_proj.forward(&gated_out.reshape((bs, sl, nv * dv))?.to_dtype(DType::F16)?, reg)
    }

}

pub struct QAttention { pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear, pub q_norm: QRmsNorm, pub k_norm: QRmsNorm, pub nh: usize, pub nkv: usize, pub hd: usize, scaling: f64, pub kv_cache: Option<(Tensor, Tensor)> }
impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self {
            q_proj: QLinear::new(&join_name(&p, "q_proj"))?, k_proj: QLinear::new(&join_name(&p, "k_proj"))?, v_proj: QLinear::new(&join_name(&p, "v_proj"))?, o_proj: QLinear::new(&join_name(&p, "o_proj"))?,
            q_norm: QRmsNorm::new(&join_name(&p, "q_norm"), config.rms_norm_eps, 1.0)?, k_norm: QRmsNorm::new(&join_name(&p, "k_norm"), config.rms_norm_eps, 1.0)?,
            nh: config.num_attention_heads, nkv: config.num_key_value_heads, hd: config.head_dim, scaling: 1.0 / (config.head_dim as f64).sqrt(), kv_cache: None,
        })
    }
    pub fn clear_vram(&mut self) {
        self.q_proj.clear_vram();
        self.k_proj.clear_vram();
        self.v_proj.clear_vram();
        self.o_proj.clear_vram();
        self.q_norm.clear_vram();
        self.k_norm.clear_vram();
        self.kv_cache = None;
    }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.q_proj.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        self.q_proj.load_to_vram(reg, dev)?;
        self.k_proj.load_to_vram(reg, dev)?;
        self.v_proj.load_to_vram(reg, dev)?;
        self.o_proj.load_to_vram(reg, dev)?;
        self.q_norm.load_to_vram(reg, dev)?;
        self.k_norm.load_to_vram(reg, dev)?;
        Ok(())
    }

    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let dev = xs.device();

        // 1. Q-Proj and Split (Q and Gate)
        // Qwen3.5: q_proj contains [Q, Gate] concatenated in the last dimension
        let q_raw = self.q_proj.forward(xs, reg)?; // (B, S, NH * HD * 2)
        let q_gate = q_raw.reshape((b_sz, q_len, self.nh, self.hd * 2))?;
        let query_states = q_gate.narrow(3, 0, self.hd)?; // (B, S, NH, HD)
        let gate_states = q_gate.narrow(3, self.hd, self.hd)?; // (B, S, NH, HD)

        // 2. K, V Projs
        let key_states = self.k_proj.forward(xs, reg)?.reshape((b_sz, q_len, self.nkv, self.hd))?;
        let value_states = self.v_proj.forward(xs, reg)?.reshape((b_sz, q_len, self.nkv, self.hd))?;

        // 3. Normalization (Per-head RMSNorm)
        let query_states = self.q_norm.forward(&query_states, reg)?;
        let key_states = self.k_norm.forward(&key_states, reg)?;

        // 4. RoPE (Rotary Position Embedding)
        // Expected shape for RoPE: (B, S, NH, HD) or (B, NH, S, HD)
        // glm_interleaved expects (B, NH, S, HD)
        let query_states = query_states.transpose(1, 2)?; // (B, NH, S, HD)
        let key_states = key_states.transpose(1, 2)?;   // (B, NKV, S, HD)

        let (query_states, key_states) = crate::position_embed::rope::apply_rotary_pos_emb(
            &query_states, 
            &key_states, 
            cos, 
            sin, 
            true
        )?;

        // 5. KV Cache
        let (key_states, value_states) = match self.kv_cache.take() {
            None => (key_states, value_states.transpose(1, 2)?),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, key_states], 2)?;
                let v = Tensor::cat(&[prev_v, value_states.transpose(1, 2)?], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // 6. Attention
        // query: (B, NH, S, HD), key: (B, NKV, S_kv, HD), value: (B, NKV, S_kv, HD)
        let attn_output = crate::models::common::eager_attention_forward(
            &query_states.to_dtype(DType::F32)?,
            &key_states.to_dtype(DType::F32)?,
            &value_states.to_dtype(DType::F32)?,
            Some(self.nh / self.nkv),
            attention_mask,
            self.scaling
        )?; // Output shape: (B, NH, S, HD)

        // 7. Gating (vllm style)
        // gate_states: (B, S, NH, HD)
        let gate = candle_nn::ops::sigmoid(&gate_states.to_dtype(DType::F32)?)?;

        // Ensure attn_output matches gate shape (B, S, NH, HD)
        let mut final_attn = attn_output.to_dtype(DType::F32)?;
        if final_attn.rank() == 4 && final_attn.dim(1)? == self.nh && gate.dim(1)? == q_len {
            // attn_output is (B, NH, S, HD), gate is (B, S, NH, HD) -> Transpose attn_output
            final_attn = final_attn.transpose(1, 2)?.contiguous()?;
        }

        let attn_output = final_attn.broadcast_mul(&gate)?; // Should be (B, S, NH, HD)

        let attn_output = attn_output.reshape((b_sz, q_len, ()))?.contiguous()?;

        // 8. O-Proj
        self.o_proj.forward(&attn_output.to_dtype(DType::F16)?, reg)
        }
        }

pub struct QDecoderLayer { pub idx: usize, pub self_attn: Option<QAttention>, pub linear_attn: Option<QGatedDeltaNet>, pub mlp_gate: QLinear, pub mlp_up: QLinear, pub mlp_down: QLinear, pub in_norm: QRmsNorm, pub post_norm: QRmsNorm }
impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let p = vb.prefix();
        let (sa, la) = if config.layer_types[idx] == "linear_attention" { (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?)) } else { (Some(QAttention::new(vb.pp("self_attn"), config)?), None) };
        Ok(Self { 
            idx, self_attn: sa, linear_attn: la, mlp_gate: QLinear::new(&join_name(&p, "mlp.gate_proj"))?, mlp_up: QLinear::new(&join_name(&p, "mlp.up_proj"))?,
            mlp_down: QLinear::new(&join_name(&p, "mlp.down_proj"))?, in_norm: QRmsNorm::new(&join_name(&p, "input_layernorm"), config.rms_norm_eps, 1.0)?, post_norm: QRmsNorm::new(&join_name(&p, "post_attention_layernorm"), config.rms_norm_eps, 1.0)? 
        })
    }
    pub fn clear_vram(&mut self) { self.in_norm.clear_vram(); if let Some(ref mut sa) = self.self_attn { sa.clear_vram(); } if let Some(ref mut la) = self.linear_attn { la.clear_vram(); } self.post_norm.clear_vram(); self.mlp_gate.clear_vram(); self.mlp_up.clear_vram(); self.mlp_down.clear_vram(); }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.in_norm.load_to_vram(reg, dev)?; if let Some(ref mut sa) = self.self_attn { sa.load_to_vram(reg, dev)?; } if let Some(ref mut la) = self.linear_attn { la.load_to_vram(reg, dev)?; }
        self.post_norm.load_to_vram(reg, dev)?; self.mlp_gate.load_to_vram(reg, dev)?; self.mlp_up.load_to_vram(reg, dev)?; self.mlp_down.load_to_vram(reg, dev)?; Ok(())
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone(); let h = self.in_norm.forward(x, reg)?;
        let h = if let Some(ref mut sa) = self.self_attn { sa.forward(&h, cos, sin, mask, reg)? } else if let Some(ref mut la) = self.linear_attn { la.forward(&h, mask, reg)? } else { h };
        let h = (h + residual)?; let residual = h.clone();
        let h = self.post_norm.forward(&h, reg)?;
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?;
        let mlp_out = self.mlp_down.forward(&(gate * self.mlp_up.forward(&h, reg)?)?, reg)?;
        Ok((mlp_out + residual)?)
    }
}

pub struct QEmbedding { pub full_name: String, pub persistent_weight: Arc<RwLock<Option<Tensor>>>, pub vocab_size: usize, pub hidden_size: usize }
impl QEmbedding {
    pub fn new(name: &str, vocab_size: usize, hidden_size: usize) -> Self { Self { full_name: name.to_string(), persistent_weight: Arc::new(RwLock::new(None)), vocab_size, hidden_size } }
    pub fn clear_vram(&mut self) { *self.persistent_weight.write().unwrap() = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        if self.persistent_weight.read().unwrap().is_some() { return Ok(()); }
        let t = reg.get_q_tensor(&self.full_name, "weight")?;
        *self.persistent_weight.write().unwrap() = Some(dequantize_q2(&t, dev)?.reshape((self.vocab_size, self.hidden_size))?); Ok(())
    }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let w = {
            let cache = self.persistent_weight.read().unwrap();
            if let Some(pw) = &*cache { Some(pw.clone()) } else { None as Option<Tensor> }
        };
        let w = if let Some(w) = w { w } 
                else { 
                    let mut cache = self.persistent_weight.write().unwrap();
                    let w = dequantize_q2(&reg.get_q_tensor(&self.full_name, "weight")?, dev)?.reshape((self.vocab_size, self.hidden_size))?;
                    *cache = Some(w.clone());
                    w
                };
        Ok(w.index_select(&ids.flatten_all()?.to_dtype(DType::U32)?, 0)?.reshape((b, s, ()))?.to_dtype(DType::F16)?)
    }
}

pub struct QTextModel { pub embed: QEmbedding, pub layers: Vec<QDecoderLayer>, pub norm: QRmsNorm, pub rotary: Qwen3_5TextRotaryEmbedding, pub mrope: Vec<usize> }
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        Ok(Self { 
            embed: QEmbedding::new(&join_name(&vb.prefix(), "embed_tokens"), config.vocab_size, config.hidden_size), 
            layers, 
            norm: QRmsNorm::new(&join_name(&vb.prefix(), "norm"), config.rms_norm_eps, 1.0)?, 
            rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, config.rope_parameters.rope_theta), 
            mrope: config.rope_parameters.mrope_section.clone() 
        })
    }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { self.embed.load_to_vram(reg, dev)?; for layer in self.layers.iter_mut() { layer.load_to_vram(reg, dev)?; } self.norm.load_to_vram(reg, dev)?; Ok(()) }
}

pub struct QuantizedQwen3_5Model { pub model: QTextModel, pub head: QLinear, pub tie: bool }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let model = QTextModel::new(vb.pp("model.language_model"), &config.text_config)?;
        let head_name = if config.tie_word_embeddings { join_name(&vb.prefix(), "model.language_model.embed_tokens") } else { join_name(&vb.prefix(), "model.language_model.lm_head") };
        let mut head = QLinear::new(&head_name)?;
        
        if config.tie_word_embeddings {
            // Share the same persistent weight handle
            head.persistent_weight = model.embed.persistent_weight.clone();
        }
        
        Ok(Self { model, head, tie: config.tie_word_embeddings })
    }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> { 
        self.model.load_all_to_vram(reg, dev)?; 
        if !self.tie { 
            self.head.load_to_vram(reg, dev)?; 
        } 
        Ok(()) 
    }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } if let Some(ref mut la) = l.linear_attn { la.recurrent_state_cache = None; la.conv_state_cache = None; } } }
}

#[derive(Clone)]
pub enum LayerContext { Attention { k: Tensor, v: Tensor }, DeltaNet { state: Tensor, conv: Tensor } }
pub struct KVStateManager { mmap: Option<memmap2::MmapMut>, layer_stride: usize, memory_cache: Vec<Option<LayerContext>>, use_memory: bool }
impl KVStateManager {
    pub fn new(num_layers: usize, use_memory: bool) -> Result<Self> {
        let layer_stride = 64 * 1024 * 1024; // Increased stride for double states
        let (mmap, memory_cache) = if use_memory { (None, vec![None; num_layers]) } else {
            let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(crate::utils::paths::get_kv_dir(None).join("kv_cache_pool.bin"))?;
            file.set_len((num_layers * layer_stride) as u64)?;
            (Some(unsafe { memmap2::MmapMut::map_mut(&file)? }), vec![])
        };
        Ok(Self { mmap, layer_stride, memory_cache, use_memory })
    }
    pub fn save_layer_context(&mut self, layer_idx: usize, ctx: LayerContext) -> Result<()> {
        if self.use_memory {
            let cpu_ctx = match ctx {
                LayerContext::Attention { k, v } => LayerContext::Attention { k: k.to_device(&Device::Cpu)?, v: v.to_device(&Device::Cpu)? },
                LayerContext::DeltaNet { state, conv } => LayerContext::DeltaNet { state: state.to_device(&Device::Cpu)?, conv: conv.to_device(&Device::Cpu)? },
            };
            self.memory_cache[layer_idx] = Some(cpu_ctx); return Ok(());
        }
        let offset = layer_idx * self.layer_stride;
        match &ctx {
            LayerContext::Attention { k, v } => { self.write_tensor(offset, k, DType::F16)?; self.write_tensor(offset + (k.elem_count() * 2), v, DType::F16)?; },
            LayerContext::DeltaNet { state, conv } => { self.write_tensor(offset, state, DType::F32)?; self.write_tensor(offset + (state.elem_count() * 4), conv, DType::F32)?; }
        }
        Ok(())
    }
    pub fn load_layer_context(&self, layer_idx: usize, lt: &str, shape_k: &candle_core::Shape, shape_v: &candle_core::Shape, dev: &Device) -> Result<LayerContext> {
        if self.use_memory {
            if let Some(ctx) = &self.memory_cache[layer_idx] {
                return match ctx {
                    LayerContext::Attention { k, v } => Ok(LayerContext::Attention { k: k.to_device(dev)?, v: v.to_device(dev)? }),
                    LayerContext::DeltaNet { state, conv } => Ok(LayerContext::DeltaNet { state: state.to_device(dev)?, conv: conv.to_device(dev)? }),
                };
            }
            anyhow::bail!("Layer context not found in memory");
        }
        let offset = layer_idx * self.layer_stride;
        if lt == "linear_attention" { 
            let state = self.read_tensor(offset, shape_k, DType::F32, dev)?;
            let (nk, dk, nv, dv) = (shape_k.dims()[1], shape_k.dims()[2], shape_k.dims()[1], shape_k.dims()[3]); // Assuming nk=nv for simplicity in shape recovery
            let conv_shape = candle_core::Shape::from((1, (nk + nk + nv) * dk, 3)); 
            let conv = self.read_tensor(offset + (shape_k.elem_count() * 4), &conv_shape, DType::F32, dev)?;
            Ok(LayerContext::DeltaNet { state, conv })
        } 
        else { Ok(LayerContext::Attention { k: self.read_tensor(offset, shape_k, DType::F16, dev)?, v: self.read_tensor(offset + (shape_k.elem_count() * 2), shape_v, DType::F16, dev)? }) }
    }
    fn write_tensor(&mut self, offset: usize, t: &Tensor, dtype: DType) -> Result<()> {
        let mmap = self.mmap.as_mut().ok_or_else(|| anyhow!("Mmap not initialized"))?;
        let t = t.detach().to_device(&Device::Cpu)?.to_dtype(dtype)?;
        let bytes = if dtype == DType::F32 { let data = t.flatten_all()?.to_vec1::<f32>()?; unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec() } }
                    else { let data = t.flatten_all()?.to_vec1::<f16>()?; unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec() } };
        mmap[offset..offset + bytes.len()].copy_from_slice(&bytes); Ok(())
    }
    fn read_tensor(&self, offset: usize, shape: &candle_core::Shape, dtype: DType, dev: &Device) -> Result<Tensor> {
        let mmap = self.mmap.as_ref().ok_or_else(|| anyhow!("Mmap not initialized"))?;
        let byte_count = shape.elem_count() * (if dtype == DType::F32 { 4 } else { 2 });
        Ok(Tensor::from_raw_buffer(&mmap[offset..offset + byte_count], dtype, shape.dims(), dev)?)
    }
}
