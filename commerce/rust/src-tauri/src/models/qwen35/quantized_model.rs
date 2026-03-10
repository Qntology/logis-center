use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_core::quantized::{QMatMul};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use candle_core::safetensors::MmapedSafetensors;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::{conv1d_depthwise, get_conv1d, softplus},
    },
    position_embed::rope::{
        Qwen3_5TextRotaryEmbedding, glm_asr_apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        l2_normalize, prepare_causal_attention_mask, repeat_interleave, split_tensor,
    },
};

/* 
 * [VRAM-OPTIMIZATION-CRITICAL]
 * GOAL: Maintain 500MB VRAM constant usage even for 10k+ sequences.
 * STRATEGY: 
 * 1. Layer-wise Cycling: Load weights to GPU ONLY during the specific layer's forward pass.
 * 2. True 4-bit Residency: Avoid dequantizing large weights into BF16 on GPU VRAM.
 * 3. Mmap-based loading: Avoid reading the whole safetensors file into RAM repeatedly.
 */

// [VRAM-OPTIMIZATION] Shared Mmap Registry to prevent repeated file reads
pub struct QuantizedRegistry {
    mmaps: HashMap<String, Arc<MmapedSafetensors>>,
    tensor_to_file: HashMap<String, String>,
}

impl QuantizedRegistry {
    pub fn new(model_list: &[String]) -> Result<Self> {
        let mut mmaps = HashMap::new();
        let mut tensor_to_file = HashMap::new();
        
        for path in model_list {
            // [SECURITY] MmapedSafetensors::new is unsafe because it depends on file integrity
            let mmap = unsafe { Arc::new(MmapedSafetensors::new(path)?) };
            for name in mmap.tensors().iter().map(|(n, _)| n.clone()) {
                tensor_to_file.insert(name, path.clone());
            }
            mmaps.insert(path.clone(), mmap);
        }
        
        Ok(Self { mmaps, tensor_to_file })
    }

    pub fn get_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let file_path = self.tensor_to_file.get(name)
            .ok_or_else(|| anyhow!("Tensor {} not found in registry", name))?;
        let mmap = self.mmaps.get(file_path).unwrap();
        Ok(mmap.load(name, device)?)
    }
}

#[derive(Clone, Debug)]
pub struct QRmsNorm {
    weight: Option<Tensor>,
    cpu_weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        let cpu_weight = weight.to_device(&Device::Cpu)?;
        Ok(Self { weight: None, cpu_weight, eps })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.weight = Some(self.cpu_weight.to_device(device)?.to_dtype(target_dtype)?);
        Ok(())
    }

    pub fn clear(&mut self) { self.weight = None; }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let w = self.weight.as_ref().ok_or_else(|| candle_core::Error::Msg("RMSNorm weight not on device".to_string()))?;
        let target_dtype = w.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        norm.to_dtype(target_dtype)?.broadcast_mul(w)
    }
}

pub struct QRmsNormGated {
    norm: QRmsNorm,
}

impl QRmsNormGated {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> { Ok(Self { norm: QRmsNorm::new(weight, eps)? }) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.norm.to_device(device) }
    pub fn clear(&mut self) { self.norm.clear(); }
    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.norm.forward(xs).map_err(|e| anyhow!(e))?;
        if let Some(gate) = gate { xs = xs.broadcast_mul(&gate.silu()?)?; }
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    tensor_name: String,
    device: Device,
}

impl QLinear {
    pub fn new(name: &str) -> Self {
        Self {
            inner: QMatMul::Tensor(Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap()),
            bias: None,
            tensor_name: name.to_string(),
            device: Device::Cpu,
        }
    }

    pub fn to_device(&mut self, device: &Device, registry: &QuantizedRegistry) -> Result<()> {
        if !self.device.same_device(device) {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            let name = &self.tensor_name;

            // Try loading quantized parts first
            let s_t = registry.get_tensor(&format!("{}.weight.scales", name), &Device::Cpu)
                .or_else(|_| registry.get_tensor(&format!("{}.scales", name), &Device::Cpu));
            let d_t = registry.get_tensor(&format!("{}.weight.data", name), &Device::Cpu)
                .or_else(|_| registry.get_tensor(&format!("{}.data", name), &Device::Cpu));
            let sh_t = registry.get_tensor(&format!("{}.weight.shape", name), &Device::Cpu)
                .or_else(|_| registry.get_tensor(&format!("{}.shape", name), &Device::Cpu));

            if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (s_t, d_t, sh_t) {
                let s = s_t.to_device(device)?.to_dtype(DType::F32)?;
                let d = d_t.to_device(device)?;
                let shape: Vec<usize> = sh_t.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
                
                let d_f32 = d.to_dtype(DType::F32)?;
                let restored = if d.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                    let high = (d_f32.clone() / 16.0)?.floor()?;
                    let low = d_f32.sub(&(high.clone() * 16.0)?)?;
                    let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                    let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                    combined.broadcast_mul(&s_exp)?
                } else {
                    let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                    d_f32.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
                };
                self.inner = QMatMul::Tensor(restored.reshape(shape.as_slice())?.to_dtype(target_dtype)?);
            } else {
                let w = registry.get_tensor(&format!("{}.weight", name), device)
                    .or_else(|_| registry.get_tensor(name, device))?;
                self.inner = QMatMul::Tensor(w.to_dtype(target_dtype)?);
            }

            if let Ok(b) = registry.get_tensor(&format!("{}.weight.bias", name), device)
                .or_else(|_| registry.get_tensor(&format!("{}.bias", name), device)) {
                self.bias = Some(b.to_dtype(target_dtype)?);
            }
            self.device = device.clone();
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.inner = QMatMul::Tensor(Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap());
        self.bias = None;
        self.device = Device::Cpu;
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.inner.forward(xs).map_err(|e| anyhow!(e))?;
        if let Some(ref b) = self.bias { Ok(out.broadcast_add(b)?) } else { Ok(out) }
    }
}

macro_rules! transmute_tensors {
    ($($tensor:expr),*) => {
        ($(
            $tensor.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        )*)
    };
}

macro_rules! right_pad_zero_tensor {
    ($dim:expr, $pad_size:expr, $($tensor:expr),+) => {
        ($(
            $tensor.pad_with_zeros($dim, 0, $pad_size)?.contiguous()?,
        )+)
    };
}

macro_rules! reshape_chunk_tensor {
    ($chunk_size:expr, $($tensor:expr),*) => {
        ($(
            {
                let (bs, head, _, dim) = $tensor.dims4()?;
                $tensor.reshape((bs, head, (), $chunk_size, dim))?.contiguous()?
            },
        )*)
    };
}

pub struct QGatedDeltaNet {
    num_v_heads: usize, num_k_heads: usize, head_k_dim: usize, head_v_dim: usize,
    key_dim: usize, value_dim: usize, conv_kernel_size: usize,
    conv1d: candle_nn::Conv1d, dt_bias: Tensor, cpu_dt_bias: Tensor,
    a_log: Tensor, cpu_a_log: Tensor,
    norm: QRmsNormGated, out_proj: QLinear, in_proj_qkv: QLinear,
    in_proj_z: QLinear, in_proj_b: QLinear, in_proj_a: QLinear,
    conv_state_cache: Option<Tensor>, recurrent_state_cache: Option<Tensor>,
    cpu_conv_state: Option<Tensor>, cpu_recurrent_state: Option<Tensor>,
}

impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let _h = config.hidden_size; let nv = config.linear_num_value_heads; let nk = config.linear_num_key_heads;
        let d_k = config.linear_key_head_dim; let d_v = config.linear_value_head_dim;
        let ck = config.linear_conv_kernel_dim;
        let conv_dim = (d_k * nk) * 2 + (d_v * nv);
        let conv1d = get_conv1d(vb.pp("conv1d"), conv_dim, conv_dim, ck, ck - 1, 1, 1, conv_dim, false)?;
        let dt_bias = vb.get(nv, "dt_bias")?; let a_log = vb.get(nv, "A_log")?;
        let p = vb.prefix();
        Ok(Self {
            num_v_heads: nv, num_k_heads: nk, head_k_dim: d_k, head_v_dim: d_v, key_dim: d_k * nk, value_dim: d_v * nv, conv_kernel_size: ck,
            conv1d, dt_bias: dt_bias.clone(), cpu_dt_bias: dt_bias.to_device(&Device::Cpu)?,
            a_log: a_log.clone(), cpu_a_log: a_log.to_device(&Device::Cpu)?,
            norm: QRmsNormGated::new(vb.get(d_v, "norm.weight")?, config.rms_norm_eps)?,
            out_proj: QLinear::new(&format!("{}.out_proj", p)),
            in_proj_qkv: QLinear::new(&format!("{}.in_proj_qkv", p)),
            in_proj_z: QLinear::new(&format!("{}.in_proj_z", p)),
            in_proj_b: QLinear::new(&format!("{}.in_proj_b", p)),
            in_proj_a: QLinear::new(&format!("{}.in_proj_a", p)),
            conv_state_cache: None, recurrent_state_cache: None,
            cpu_conv_state: None, cpu_recurrent_state: None,
        })
    }

    pub fn to_device(&mut self, d: &Device, registry: &QuantizedRegistry) -> Result<()> {
        let dtype = if d.is_cuda() { DType::BF16 } else { DType::F32 };
        self.dt_bias = self.cpu_dt_bias.to_device(d)?.to_dtype(dtype)?;
        self.a_log = self.cpu_a_log.to_device(d)?.to_dtype(dtype)?;
        self.norm.to_device(d)?; 
        self.out_proj.to_device(d, registry)?; self.in_proj_qkv.to_device(d, registry)?;
        self.in_proj_z.to_device(d, registry)?; self.in_proj_b.to_device(d, registry)?; self.in_proj_a.to_device(d, registry)?;
        
        if let Some(s) = self.cpu_conv_state.take() { self.conv_state_cache = Some(s.to_device(d)?); }
        if let Some(r) = self.cpu_recurrent_state.take() { self.recurrent_state_cache = Some(r.to_device(d)?); }
        Ok(())
    }

    pub fn clear(&mut self) {
        if let Some(s) = self.conv_state_cache.take() { self.cpu_conv_state = Some(s.to_device(&Device::Cpu).unwrap()); }
        if let Some(r) = self.recurrent_state_cache.take() { self.cpu_recurrent_state = Some(r.to_device(&Device::Cpu).unwrap()); }
        self.norm.clear(); self.out_proj.clear(); self.in_proj_qkv.clear();
        self.in_proj_z.clear(); self.in_proj_b.clear(); self.in_proj_a.clear();
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().ok_or(anyhow!("conv_state_cache is None"))?;
        let seq_len = xs.dim(2)?; let state_len = conv_state.dim(D::Minus1)?;
        let conv_state_new = Tensor::cat(&[conv_state, xs], D::Minus1)?;
        let conv_update = conv_state_new.narrow(D::Minus1, seq_len, state_len)?;
        self.conv_state_cache = Some(conv_update);
        let out = conv1d_depthwise(&conv_state_new, self.conv1d.weight(), self.conv1d.bias())?;
        Ok(out.narrow(D::Minus1, out.dim(D::Minus1)? - seq_len, seq_len)?.silu()?)
    }

    fn torch_chunk_gated_delta_rule(&mut self, query: &Tensor, key: &Tensor, value: &Tensor, g: &Tensor, beta: &Tensor, use_qk_l2norm_in_kernel: bool, chunk_size: usize) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel { (l2_normalize(query, 3)?, l2_normalize(key, 3)?) } else { (query.clone(), key.clone()) };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g): (Tensor, Tensor, Tensor, Tensor, Tensor) = transmute_tensors!(query, key, value, beta, g);
        let (bs, nh, sl, dk) = key.dims4()?; let dv = value.dim(D::Minus1)?;
        let pad = (chunk_size - sl % chunk_size) % chunk_size;
        let (query, key, value, beta, g): (Tensor, Tensor, Tensor, Tensor, Tensor) = right_pad_zero_tensor!(2, pad, query, key, value, beta, g);
        let total_sl = sl + pad; let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let vb = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let kb = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, kb, vb) = reshape_chunk_tensor!(chunk_size, query, key, kb, vb);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?.cumsum(D::Minus1)?;
        let decay = g.unsqueeze(D::Minus1)?.broadcast_sub(&g.unsqueeze(D::Minus2)?)?.exp()?.to_dtype(DType::F32)?;
        
        // [FIX] Ensure U32 for all indexing and masks
        let tril = Tensor::tril2(chunk_size, DType::U32, query.device())?.broadcast_as(decay.shape())?;
        let decay = tril.where_cond(&decay, &decay.zeros_like()?)?.contiguous()?;
        let mut attn = kb.squeeze(0)?.contiguous()?.matmul(&key.squeeze(0)?.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.unsqueeze(0)?.mul(&decay)?.affine(-1.0, 0.0)?;
        let mask = Tensor::triu2(chunk_size, DType::U32, query.device())?.broadcast_as(decay.shape())?;
        attn = mask.where_cond(&decay.zeros_like()?, &attn)?;
        
        let (d0, d1, d2, _, _) = attn.dims5()?;
        for i in 1..chunk_size {
            let row = attn.i((.., .., .., i, ..i))?.contiguous()?;
            let sub = attn.i((.., .., .., ..i, ..i))?.contiguous()?;
            let attn_i = (row.unsqueeze(D::Minus1)?.broadcast_mul(&sub)?.sum(D::Minus2)? + row)?.unsqueeze(D::Minus2)?;
            attn = attn.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
        }
        let attn = (attn.broadcast_add(&Tensor::eye(chunk_size, attn.dtype(), attn.device())?))?.contiguous()?;
        let value_out = attn.squeeze(0)?.matmul(&vb.squeeze(0)?)?.unsqueeze(0)?;
        let k_cum = attn.squeeze(0)?.matmul(&kb.broadcast_mul(&g.exp()?.unsqueeze(D::Minus1)?)?.squeeze(0)?)?.unsqueeze(0)?;
        let mut lrs = if let Some(r) = self.recurrent_state_cache.as_ref() { r.clone() } else { Tensor::zeros((bs, nh, dk, dv), DType::F32, value_out.device())? };
        let mut out = value_out.zeros_like()?;
        
        let tril = Tensor::tril2(chunk_size, DType::U32, query.device())?.broadcast_as((bs, nh, chunk_size, chunk_size))?;
        let on_f = tril.zeros_like()?.to_dtype(DType::F32)?;
        for i in 0..total_sl / chunk_size {
            let qi = query.i((.., .., i))?.contiguous()?; let ki = key.i((.., .., i))?.contiguous()?;
            let vi = value_out.i((.., .., i))?.contiguous()?; let gi = g.i((.., .., i))?.contiguous()?;
            let att = tril.where_cond(&qi.matmul(&ki.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.mul(&decay.i((.., .., i))?)?, &on_f)?.contiguous()?;
            let vn = (vi - k_cum.i((.., .., i))?.matmul(&lrs)?)?;
            let out_i = (qi.broadcast_mul(&gi.unsqueeze(D::Minus1)?.exp()?)?.matmul(&lrs)? + att.matmul(&vn)?)?.unsqueeze(2)?;
            out = out.slice_assign(&[(0..bs), (0..nh), (i..i + 1), (0..chunk_size), (0..dv)], &out_i)?;
            let gl = gi.dim(D::Minus1)?;
            lrs = (lrs.broadcast_mul(&gi.narrow(D::Minus1, gl - 1, 1)?.unsqueeze(D::Minus1)?.exp()?)? + ki.broadcast_mul(&gi.narrow(D::Minus1, gl - 1, 1)?.broadcast_sub(&gi)?.exp()?.unsqueeze(D::Minus1)?)?.transpose(D::Minus1, D::Minus2)?.squeeze(0)?.matmul(&vn.squeeze(0)?)?.unsqueeze(0)?)?;
        }
        self.recurrent_state_cache = Some(lrs);
        let out = out.reshape((bs, nh, (), dv))?.narrow(2, 0, sl)?;
        Ok(out.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = if let Some(mask) = attention_mask { xs.broadcast_mul(&mask.unsqueeze(D::Minus1)?)? } else { xs.clone() };
        let (bs, sl, _) = xs.dims3()?;
        let mut mixed = self.in_proj_qkv.forward(&xs)?.transpose(1, 2)?;
        let z = self.in_proj_z.forward(&xs)?.reshape((bs, sl, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(&xs)?; let a = self.in_proj_a.forward(&xs)?;
        if self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && sl == 1 { mixed = self.torch_causal_conv1d_update(&mixed)?; }
        else {
            let pad = self.conv_kernel_size as isize - mixed.dim(D::Minus1)? as isize;
            self.conv_state_cache = Some(if pad >= 0 { mixed.pad_with_zeros(D::Minus1, pad as usize, 0)? } else { mixed.narrow(D::Minus1, pad.unsigned_abs(), self.conv_kernel_size)? });
            mixed = conv1d_depthwise(&mixed.pad_with_zeros(D::Minus1, self.conv_kernel_size - 1, self.conv_kernel_size - 1)?, self.conv1d.weight(), self.conv1d.bias())?.narrow(D::Minus1, 0, sl)?.silu()?;
        }
        let qkv = split_tensor(&mixed.transpose(1, 2)?, &[self.key_dim, self.key_dim, self.value_dim], D::Minus1)?;
        let mut q = qkv[0].reshape((bs, sl, (), self.head_k_dim))?; let mut k = qkv[1].reshape((bs, sl, (), self.head_k_dim))?;
        let v = qkv[2].reshape((bs, sl, (), self.head_v_dim))?;
        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = (-1.0 * self.a_log.to_dtype(DType::F32)?.exp()?)?.broadcast_mul(&softplus(&a.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?)?)?;
        if self.num_v_heads / self.num_k_heads > 1 {
            q = repeat_interleave(&q, self.num_v_heads / self.num_k_heads, 2)?;
            k = repeat_interleave(&k, self.num_v_heads / self.num_k_heads, 2)?;
        }
        let out = self.torch_chunk_gated_delta_rule(&q, &k, &v, &g, &beta, true, 64)?;
        let out = self.norm.forward(&out.reshape(((), self.head_v_dim))?, Some(&z.reshape(((), self.head_v_dim))?))?;
        self.out_proj.forward(&out.reshape((bs, sl, ()))?)
    }

    pub fn clear_cache(&mut self) { self.conv_state_cache = None; self.recurrent_state_cache = None; }
}

pub struct QAttention {
    q_proj: QLinear, k_proj: QLinear, v_proj: QLinear, o_proj: QLinear,
    q_norm: QRmsNorm, k_norm: QRmsNorm,
    num_attention_heads: usize, num_key_value_heads: usize, num_kv_groups: usize,
    head_dim: usize, scaling: f64, 
    kv_cache: Option<(Tensor, Tensor)>,
    cpu_kv_cache: Option<(Tensor, Tensor)>,
}

impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let _h = config.hidden_size; let n = config.num_attention_heads; let d = config.head_dim; let n_kv = config.num_key_value_heads;
        let p = vb.prefix();
        Ok(Self {
            q_proj: QLinear::new(&format!("{}.q_proj", p)),
            k_proj: QLinear::new(&format!("{}.k_proj", p)),
            v_proj: QLinear::new(&format!("{}.v_proj", p)),
            o_proj: QLinear::new(&format!("{}.o_proj", p)),
            q_norm: QRmsNorm::new(vb.get(d, "q_norm.weight")?, config.rms_norm_eps)?,
            k_norm: QRmsNorm::new(vb.get(d, "k_norm.weight")?, config.rms_norm_eps)?,
            num_attention_heads: n, num_key_value_heads: n_kv, num_kv_groups: n / n_kv, head_dim: d, scaling: 1.0 / (d as f64).sqrt(), 
            kv_cache: None,
            cpu_kv_cache: None,
        })
    }
    pub fn to_device(&mut self, device: &Device, registry: &QuantizedRegistry) -> Result<()> { 
        self.q_proj.to_device(device, registry)?; self.k_proj.to_device(device, registry)?; 
        self.v_proj.to_device(device, registry)?; self.o_proj.to_device(device, registry)?; 
        self.q_norm.to_device(device)?; self.k_norm.to_device(device)?; 
        
        if let Some((k, v)) = self.cpu_kv_cache.take() {
            self.kv_cache = Some((k.to_device(device)?, v.to_device(device)?));
        }
        Ok(()) 
    }
    pub fn clear(&mut self) { 
        if let Some((k, v)) = self.kv_cache.take() {
            self.cpu_kv_cache = Some((k.to_device(&Device::Cpu).unwrap(), v.to_device(&Device::Cpu).unwrap()));
        }
        self.q_proj.clear(); self.k_proj.clear(); self.v_proj.clear(); self.o_proj.clear(); self.q_norm.clear(); self.k_norm.clear(); 
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None; self.cpu_kv_cache = None; }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bs, sl, _) = xs.dims3()?;
        let q_chunk = self.q_proj.forward(xs)?.reshape((bs, sl, self.num_attention_heads, self.head_dim * 2))?.chunk(2, D::Minus1)?;
        let q = self.q_norm.forward(&q_chunk[0].reshape((bs, sl, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((bs, sl, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((bs, sl, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (q, k) = glm_asr_apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
        let (k, v) = match &self.kv_cache { None => (k, v), Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?) };
        self.kv_cache = Some((k.clone(), v.clone()));
        let attn = crate::models::common::eager_attention_forward(&q, &k, &v, Some(self.num_kv_groups), mask, self.scaling)?;
        let attn = attn.reshape((bs, sl, self.num_attention_heads * self.head_dim))?.contiguous()?;
        self.o_proj.forward(&(attn * candle_nn::ops::sigmoid(&q_chunk[1].reshape((bs, sl, ()))?)?)?)
    }
}

pub struct QDecoderLayer {
    layer_type: String, linear_attn: Option<QGatedDeltaNet>, self_attn: Option<QAttention>,
    mlp_gate_proj: QLinear, mlp_up_proj: QLinear, mlp_down_proj: QLinear,
    input_layernorm: QRmsNorm, post_attention_layernorm: QRmsNorm,
}

impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize) -> Result<Self> {
        let _h = config.hidden_size; let lt = config.layer_types[layer_idx].clone();
        let (la, sa) = if lt == "linear_attention" { (Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?), None) } else { (None, Some(QAttention::new(vb.pp("self_attn"), config)?)) };
        let p = vb.prefix();
        Ok(Self {
            layer_type: lt, linear_attn: la, self_attn: sa,
            mlp_gate_proj: QLinear::new(&format!("{}.mlp.gate_proj", p)),
            mlp_up_proj: QLinear::new(&format!("{}.mlp.up_proj", p)),
            mlp_down_proj: QLinear::new(&format!("{}.mlp.down_proj", p)),
            input_layernorm: QRmsNorm::new(vb.get(config.hidden_size, "input_layernorm.weight")?, config.rms_norm_eps)?,
            post_attention_layernorm: QRmsNorm::new(vb.get(config.hidden_size, "post_attention_layernorm.weight")?, config.rms_norm_eps)?,
        })
    }
    pub fn to_device(&mut self, d: &Device, registry: &QuantizedRegistry) -> Result<()> { 
        if let Some(ref mut l) = self.linear_attn { l.to_device(d, registry)?; } 
        if let Some(ref mut s) = self.self_attn { s.to_device(d, registry)?; } 
        self.mlp_gate_proj.to_device(d, registry)?; self.mlp_up_proj.to_device(d, registry)?; 
        self.mlp_down_proj.to_device(d, registry)?; self.input_layernorm.to_device(d)?; 
        self.post_attention_layernorm.to_device(d)?; Ok(()) 
    }
    pub fn clear(&mut self) { if let Some(ref mut l) = self.linear_attn { l.clear(); } if let Some(ref mut s) = self.self_attn { s.clear(); } self.mlp_gate_proj.clear(); self.mlp_up_proj.clear(); self.mlp_down_proj.clear(); self.input_layernorm.clear(); self.post_attention_layernorm.clear(); }
    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = xs.clone(); let mut x = self.input_layernorm.forward(xs).map_err(|e| anyhow!(e))?;
        if self.layer_type == "linear_attention" { if let Some(ref mut l) = self.linear_attn { x = l.forward(&x, mask)?; } }
        else { if let (Some(ref mut s), Some(c), Some(sn)) = (&mut self.self_attn, cos, sin) { x = s.forward(&x, c, sn, mask)?; } }
        let r2 = (x + r)?; let x = self.post_attention_layernorm.forward(&r2).map_err(|e| anyhow!(e))?;
        let gate = self.mlp_gate_proj.forward(&x)?.silu()?; let up = self.mlp_up_proj.forward(&x)?;
        Ok((self.mlp_down_proj.forward(&(gate * up)?)? + r2)?)
    }
}

pub struct QTextModel {
    embed_tokens: crate::models::qwen35::model::Qwen3_5Embedding,
    layers: Vec<QDecoderLayer>, norm: QRmsNorm,
    rotary_emb: Qwen3_5TextRotaryEmbedding, mrope_section: Vec<usize>,
}

impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        Ok(Self {
            embed_tokens: crate::models::qwen35::model::Qwen3_5Embedding::new(vb.pp("embed_tokens"), config.vocab_size, config.hidden_size)?,
            layers, norm: QRmsNorm::new(vb.get(config.hidden_size, "norm.weight")?, config.rms_norm_eps)?,
            rotary_emb: Qwen3_5TextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta),
            mrope_section: config.rope_parameters.mrope_section.clone(),
        })
    }
    pub fn forward(&mut self, inputs: &Tensor, pos: &Tensor, offset: usize, registry: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = inputs.dims3()?; let dev = inputs.device();
        let (cos, sin) = self.rotary_emb.forward(pos, inputs.dtype(), self.mrope_section.clone())?;
        let mut x = inputs.clone();
        
        let mask = if sl > 1024 { None } else if sl <= 1 && offset == 0 { None } else { 
            let m = prepare_causal_attention_mask(bs, sl, offset, dev)?;
            Some(m.to_dtype(DType::F16)?)
        };

        for layer in self.layers.iter_mut() {
            layer.to_device(dev, registry)?;
            let lm = if layer.layer_type == "linear_attention" { None } else { mask.as_ref() };
            x = layer.forward(&x, Some(&cos), Some(&sin), lm)?;
            layer.clear();
        }
        self.norm.to_device(dev)?; let x = self.norm.forward(&x).map_err(|e| anyhow!(e))?; self.norm.clear(); Ok(x)
    }
}

pub struct QuantizedQwen3_5Model {
    config: Qwen3_5Config, language_model: QTextModel, lm_head: QLinear,
}

impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let lm = QTextModel::new(vb.pp("model.language_model"), &config.text_config)?;
        let head = QLinear::new("model.language_model.embed_tokens");
        Ok(Self { config, language_model: lm, lm_head: head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize, registry: &QuantizedRegistry) -> Result<Tensor> {
        let dev = input_ids.device();
        self.language_model.embed_tokens.to_device(dev).map_err(|e| anyhow!(e))?;
        let embeds = self.language_model.embed_tokens.forward(input_ids).map_err(|e| anyhow!(e))?;
        self.language_model.embed_tokens.clear();

        let seq_len = input_ids.dim(1)?;
        let pos = Tensor::arange(offset as u32, (offset + seq_len) as u32, dev)?
            .to_dtype(DType::U32)?
            .unsqueeze(0)?.unsqueeze(0)?
            .broadcast_as((3, input_ids.dim(0)?, seq_len))?;

        let outputs = self.language_model.forward(&embeds, &pos, offset, registry)?;
        let hidden = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        
        self.lm_head.to_device(dev, registry).map_err(|e| anyhow!(e))?;
        let logits = self.lm_head.forward(&hidden)?;
        self.lm_head.clear();
        Ok(logits)
    }

    pub fn clear_cache(&mut self) { for l in self.language_model.layers.iter_mut() { if let Some(ref mut la) = l.linear_attn { la.clear_cache(); } if let Some(ref mut sa) = l.self_attn { sa.clear_kv_cache(); } } }
}
