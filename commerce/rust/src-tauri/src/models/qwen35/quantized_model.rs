use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_core::quantized::{gguf_file, QMatMul};
use std::sync::Arc;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::{conv1d_depthwise, get_conv1d, softplus},
    },
    position_embed::rope::{
        Qwen3VLTextRotaryEmbedding, glm_asr_apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        get_equal_mask, l2_normalize, masked_scatter_dim0,
        prepare_causal_attention_mask, repeat_interleave, split_tensor,
    },
};

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

#[derive(Clone, Debug)]
pub struct QRmsNorm {
    weight: Tensor,
    cpu_weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        let cpu_weight = weight.to_device(&Device::Cpu)?;
        Ok(Self { weight, cpu_weight, eps })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.weight = self.cpu_weight.to_device(device)?.to_dtype(target_dtype)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), self.cpu_weight.dtype(), &Device::Cpu).unwrap();
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let target_dtype = self.weight.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let norm = norm.to_dtype(target_dtype)?;
        norm.broadcast_mul(&self.weight)
    }
}

pub struct QRmsNormGated {
    norm: QRmsNorm,
}

impl QRmsNormGated {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { norm: QRmsNorm::new(weight, eps)? })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.norm.to_device(device) }
    pub fn clear(&mut self) { self.norm.clear(); }
    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.norm.forward(xs)?;
        if let Some(gate) = gate {
            xs = xs.broadcast_mul(&gate.silu()?)?;
        }
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    cpu_inner: Arc<QMatMul>,
    cpu_bias: Option<Tensor>,
    device: Device,
}

impl QLinear {
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, device: &Device) -> Result<Self> {
        let qmatmul = QMatMul::from_arc(Arc::new(weight));
        let cpu_bias = if let Some(ref b) = bias { Some(b.to_device(&Device::Cpu)?) } else { None };
        Ok(Self {
            inner: qmatmul.clone(),
            bias,
            cpu_inner: Arc::new(qmatmul),
            cpu_bias,
            device: device.clone(),
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if !self.device.same_device(device) {
            let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            // In a real quantization scenario, we'd handle QTensor dequantization or relocation
            if let Some(ref b) = self.cpu_bias {
                self.bias = Some(b.to_device(device)?.to_dtype(target_dtype)?);
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
        let out = self.inner.forward(xs)?;
        if let Some(ref bias) = self.bias {
            Ok(out.broadcast_add(bias)?)
        } else {
            Ok(out)
        }
    }
}

pub struct QGatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv1d: candle_nn::Conv1d,
    dt_bias: Tensor,
    cpu_dt_bias: Tensor,
    a_log: Tensor,
    cpu_a_log: Tensor,
    norm: QRmsNormGated,
    out_proj: QLinear,
    in_proj_qkv: QLinear,
    in_proj_z: QLinear,
    in_proj_b: QLinear,
    in_proj_a: QLinear,
    conv_state_cache: Option<Tensor>,
    recurrent_state_cache: Option<Tensor>,
}

impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let conv1d = get_conv1d(vb.pp("conv1d"), conv_dim, conv_dim, conv_kernel_size, conv_kernel_size - 1, 1, 1, conv_dim, false)?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        
        let norm = QRmsNormGated::new(vb.get(head_v_dim, "norm.weight")?, config.rms_norm_eps)?;
        let out_proj = QLinear::from_weights(vb.get((hidden_size, value_dim), "out_proj.weight")?, None, vb.device())?;
        let in_proj_qkv = QLinear::from_weights(vb.get((conv_dim, hidden_size), "in_proj_qkv.weight")?, None, vb.device())?;
        let in_proj_z = QLinear::from_weights(vb.get((value_dim, hidden_size), "in_proj_z.weight")?, None, vb.device())?;
        let in_proj_b = QLinear::from_weights(vb.get((num_v_heads, hidden_size), "in_proj_b.weight")?, None, vb.device())?;
        let in_proj_a = QLinear::from_weights(vb.get((num_v_heads, hidden_size), "in_proj_a.weight")?, None, vb.device())?;

        Ok(Self {
            num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, conv_kernel_size,
            conv1d, dt_bias: dt_bias.clone(), cpu_dt_bias: dt_bias.to_device(&Device::Cpu)?,
            a_log: a_log.clone(), cpu_a_log: a_log.to_device(&Device::Cpu)?,
            norm, out_proj, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
            conv_state_cache: None, recurrent_state_cache: None,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.dt_bias = self.cpu_dt_bias.to_device(device)?.to_dtype(dtype)?;
        self.a_log = self.cpu_a_log.to_device(device)?.to_dtype(dtype)?;
        self.norm.to_device(device)?; self.out_proj.to_device(device)?;
        self.in_proj_qkv.to_device(device)?; self.in_proj_z.to_device(device)?;
        self.in_proj_b.to_device(device)?; self.in_proj_a.to_device(device)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.dt_bias = Tensor::zeros((1,), self.cpu_dt_bias.dtype(), &Device::Cpu).unwrap();
        self.a_log = Tensor::zeros((1,), self.cpu_a_log.dtype(), &Device::Cpu).unwrap();
        self.norm.clear(); self.out_proj.clear(); self.in_proj_qkv.clear();
        self.in_proj_z.clear(); self.in_proj_b.clear(); self.in_proj_a.clear();
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().ok_or(anyhow!("conv_state_cache is None"))?;
        let seq_len = xs.dim(2)?;
        let state_len = conv_state.dim(D::Minus1)?;
        let conv_state_new = Tensor::cat(&[conv_state, xs], D::Minus1)?;
        let conv_update = conv_state_new.narrow(D::Minus1, seq_len, state_len)?;
        self.conv_state_cache = Some(conv_update);
        let out = conv1d_depthwise(&conv_state_new, self.conv1d.weight(), self.conv1d.bias())?;
        let start = out.dim(D::Minus1)? - seq_len;
        let out = out.narrow(D::Minus1, start, seq_len)?.silu()?;
        Ok(out)
    }

    fn torch_chunk_gated_delta_rule(&mut self, query: &Tensor, key: &Tensor, value: &Tensor, g: &Tensor, beta: &Tensor, use_qk_l2norm_in_kernel: bool, chunk_size: usize) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel { (l2_normalize(query, 3)?, l2_normalize(key, 3)?) } else { (query.clone(), key.clone()) };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g): (Tensor, Tensor, Tensor, Tensor, Tensor) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let (query, key, value, beta, g): (Tensor, Tensor, Tensor, Tensor, Tensor) = right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g);
        let total_sequence_length = sequence_length + pad_size;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, k_beta, v_beta) = reshape_chunk_tensor!(chunk_size, query, key, k_beta, v_beta);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?;
        let g = g.cumsum(D::Minus1)?;
        let decay_mask = g.unsqueeze(D::Minus1)?.broadcast_sub(&g.unsqueeze(D::Minus2)?)?.exp()?.to_dtype(candle_core::DType::F32)?;
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::U32, query.device())?.broadcast_as(decay_mask.shape())?;
        let on_false = decay_mask.zeros_like()?;
        let decay_mask = tril_mask.where_cond(&decay_mask, &on_false)?.contiguous()?;
        let mut attn = k_beta.squeeze(0)?.contiguous()?.matmul(&key.squeeze(0)?.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.unsqueeze(0)?.mul(&decay_mask)?.affine(-1.0, 0.0)?;
        let mask = Tensor::triu2(chunk_size, candle_core::DType::U32, query.device())?.broadcast_as(decay_mask.shape())?;
        attn = mask.where_cond(&on_false, &attn)?;
        let (d0, d1, d2, _, _) = attn.dims5()?;
        for i in 1..chunk_size {
            let row = attn.i((.., .., .., i, ..i))?.contiguous()?;
            let sub = attn.i((.., .., .., ..i, ..i))?.contiguous()?;
            let attn_i = row.unsqueeze(D::Minus1)?.broadcast_mul(&sub)?.sum(D::Minus2)?.add(&row)?.unsqueeze(D::Minus2)?;
            attn = attn.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
        }
        let attn = attn.broadcast_add(&Tensor::eye(chunk_size, attn.dtype(), attn.device())?)?.contiguous()?;
        let value = attn.squeeze(0)?.matmul(&v_beta.squeeze(0)?)?.unsqueeze(0)?;
        let k_cumdecay = attn.squeeze(0)?.matmul(&k_beta.broadcast_mul(&g.exp()?.unsqueeze(D::Minus1)?)?.squeeze(0)?)?.unsqueeze(0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref() { recurrent.clone() } else { Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), candle_core::DType::F32, value.device())? };
        let mut core_attn_out = value.zeros_like()?;
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::U32, query.device())?.broadcast_as((batch_size, num_heads, chunk_size, chunk_size))?;
        let on_false = tril_mask.zeros_like()?.to_dtype(candle_core::DType::F32)?;
        let last_dim = core_attn_out.dim(D::Minus1)?;
        for i in 0..total_sequence_length / chunk_size {
            let q_i = query.i((.., .., i))?.contiguous()?;
            let k_i = key.i((.., .., i))?.contiguous()?;
            let v_i = value.i((.., .., i))?.contiguous()?;
            let g_i = g.i((.., .., i))?.contiguous()?;
            let attn = q_i.matmul(&k_i.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.mul(&decay_mask.i((.., .., i))?)?;
            let attn = tril_mask.where_cond(&attn, &on_false)?.contiguous()?;
            let v_prime = k_cumdecay.i((.., .., i))?.matmul(&last_recurrent_state)?;
            let v_new = v_i.sub(&v_prime)?;
            let attn_inter = q_i.broadcast_mul(&g_i.unsqueeze(D::Minus1)?.exp()?)?.matmul(&last_recurrent_state)?;
            let out_i = attn_inter.add(&attn.matmul(&v_new)?)?.unsqueeze(2)?;
            core_attn_out = core_attn_out.slice_assign(&[(0..batch_size), (0..num_heads), (i..i + 1), (0..chunk_size), (0..last_dim)], &out_i)?;
            let g_i_last_dim = g_i.dim(D::Minus1)?;
            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?.unsqueeze(D::Minus1)?.exp()?)?.add(&k_i.broadcast_mul(&g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?.broadcast_sub(&g_i)?.exp()?.unsqueeze(D::Minus1)?)?.transpose(D::Minus1, D::Minus2)?.squeeze(0)?.matmul(&v_new.squeeze(0)?)?.unsqueeze(0)?)?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out = core_attn_out.reshape((batch_size, num_heads, (), core_attn_out.dim(D::Minus1)?))?.narrow(2, 0, sequence_length)?;
        Ok(core_attn_out.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = if let Some(mask) = attention_mask { xs.broadcast_mul(&mask.unsqueeze(D::Minus1)?)? } else { xs.clone() };
        let (bs, seq_len, _) = xs.dims3()?;
        let mut mixed_qkv = self.in_proj_qkv.forward(&xs)?.transpose(1, 2)?;
        let z = self.in_proj_z.forward(&xs)?.reshape((bs, seq_len, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(&xs)?;
        let a = self.in_proj_a.forward(&xs)?;
        
        if self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && seq_len == 1 {
            mixed_qkv = self.torch_causal_conv1d_update(&mixed_qkv)?;
        } else {
            let pad = self.conv_kernel_size as isize - mixed_qkv.dim(D::Minus1)? as isize;
            let conv_state = if pad >= 0 { mixed_qkv.pad_with_zeros(D::Minus1, pad as usize, 0)? } else { mixed_qkv.narrow(D::Minus1, pad.unsigned_abs(), self.conv_kernel_size)? };
            self.conv_state_cache = Some(conv_state);
            mixed_qkv = mixed_qkv.pad_with_zeros(D::Minus1, self.conv_kernel_size - 1, self.conv_kernel_size - 1)?;
            mixed_qkv = conv1d_depthwise(&mixed_qkv, self.conv1d.weight(), self.conv1d.bias())?;
            mixed_qkv = mixed_qkv.narrow(D::Minus1, 0, seq_len)?.silu()?;
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        let qkv_split = split_tensor(&mixed_qkv, &[self.key_dim, self.key_dim, self.value_dim], D::Minus1)?;
        let mut query = qkv_split[0].reshape((bs, seq_len, (), self.head_k_dim))?;
        let mut key = qkv_split[1].reshape((bs, seq_len, (), self.head_k_dim))?;
        let value = qkv_split[2].reshape((bs, seq_len, (), self.head_v_dim))?;
        let beta = candle_nn::ops::sigmoid(&b)?;
        let a_plus_bias = softplus(&a.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?)?;
        let g = (-1.0 * self.a_log.to_dtype(DType::F32)?.exp()?)?.broadcast_mul(&a_plus_bias)?;
        if self.num_v_heads / self.num_k_heads > 1 {
            query = repeat_interleave(&query, self.num_v_heads / self.num_k_heads, 2)?;
            key = repeat_interleave(&key, self.num_v_heads / self.num_k_heads, 2)?;
        }
        let core_attn_out = self.torch_chunk_gated_delta_rule(&query, &key, &value, &g, &beta, true, 64)?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.norm.forward(&core_attn_out, Some(&z))?;
        self.out_proj.forward(&core_attn_out.reshape((bs, seq_len, ()))?)
    }

    pub fn clear_cache(&mut self) { self.conv_state_cache = None; self.recurrent_state_cache = None; }
}

pub struct QAttention {
    q_proj: QLinear, k_proj: QLinear, v_proj: QLinear, o_proj: QLinear,
    q_norm: QRmsNorm, k_norm: QRmsNorm,
    num_attention_heads: usize, num_key_value_heads: usize, num_kv_groups: usize,
    head_dim: usize, scaling: f64, kv_cache: Option<(Tensor, Tensor)>,
}

impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let h = config.hidden_size; let n = config.num_attention_heads; let d = config.head_dim; let n_kv = config.num_key_value_heads;
        Ok(Self {
            q_proj: QLinear::from_weights(vb.get((n * d * 2, h), "q_proj.weight")?, None, vb.device())?,
            k_proj: QLinear::from_weights(vb.get((n_kv * d, h), "k_proj.weight")?, None, vb.device())?,
            v_proj: QLinear::from_weights(vb.get((n_kv * d, h), "v_proj.weight")?, None, vb.device())?,
            o_proj: QLinear::from_weights(vb.get((h, n * d), "o_proj.weight")?, None, vb.device())?,
            q_norm: QRmsNorm::new(vb.get(d, "q_norm.weight")?, config.rms_norm_eps)?,
            k_norm: QRmsNorm::new(vb.get(d, "k_norm.weight")?, config.rms_norm_eps)?,
            num_attention_heads: n, num_key_value_heads: n_kv, num_kv_groups: n / n_kv, head_dim: d, scaling: 1.0 / (d as f64).sqrt(), kv_cache: None,
        })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.q_proj.to_device(device)?; self.k_proj.to_device(device)?; self.v_proj.to_device(device)?; self.o_proj.to_device(device)?; self.q_norm.to_device(device)?; self.k_norm.to_device(device)?; Ok(()) }
    pub fn clear(&mut self) { self.q_proj.clear(); self.k_proj.clear(); self.v_proj.clear(); self.o_proj.clear(); self.q_norm.clear(); self.k_norm.clear(); }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q_chunk = self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim * 2))?.chunk(2, D::Minus1)?;
        let q_states = self.q_norm.forward(&q_chunk[0].reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?)?.transpose(1, 2)?;
        let k_states = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?;
        let v_states = self.v_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (q_states, k_states) = glm_asr_apply_rotary_pos_emb(&q_states, &k_states, cos, sin, false)?;
        let (k_states, v_states) = match &self.kv_cache { None => (k_states, v_states), Some((pk, pv)) => (Tensor::cat(&[pk, &k_states], 2)?, Tensor::cat(&[pv, &v_states], 2)?) };
        self.kv_cache = Some((k_states.clone(), v_states.clone()));
        let attn = crate::models::common::eager_attention_forward(&q_states, &k_states, &v_states, Some(self.num_kv_groups), attention_mask, self.scaling)?;
        let attn = attn.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?.contiguous()?;
        self.o_proj.forward(&(attn * candle_nn::ops::sigmoid(&q_chunk[1].reshape((b_sz, q_len, ()))?)?)?)
    }
}

pub struct QDecoderLayer {
    layer_type: String,
    linear_attn: Option<QGatedDeltaNet>,
    self_attn: Option<QAttention>,
    mlp_gate_proj: QLinear, mlp_up_proj: QLinear, mlp_down_proj: QLinear,
    input_layernorm: QRmsNorm, post_attention_layernorm: QRmsNorm,
}

impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize) -> Result<Self> {
        let h = config.hidden_size; let i = config.intermediate_size; let lt = config.layer_types[layer_idx].clone();
        let (la, sa) = if lt == "linear_attention" { (Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?), None) } else { (None, Some(QAttention::new(vb.pp("self_attn"), config)?)) };
        Ok(Self {
            layer_type: lt, linear_attn: la, self_attn: sa,
            mlp_gate_proj: QLinear::from_weights(vb.get((i, h), "mlp.gate_proj.weight")?, None, vb.device())?,
            mlp_up_proj: QLinear::from_weights(vb.get((i, h), "mlp.up_proj.weight")?, None, vb.device())?,
            mlp_down_proj: QLinear::from_weights(vb.get((h, i), "mlp.down_proj.weight")?, None, vb.device())?,
            input_layernorm: QRmsNorm::new(vb.get(h, "input_layernorm.weight")?, config.rms_norm_eps)?,
            post_attention_layernorm: QRmsNorm::new(vb.get(h, "post_attention_layernorm.weight")?, config.rms_norm_eps)?,
        })
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { if let Some(ref mut l) = self.linear_attn { l.to_device(d)?; } if let Some(ref mut s) = self.self_attn { s.to_device(d)?; } self.mlp_gate_proj.to_device(d)?; self.mlp_up_proj.to_device(d)?; self.mlp_down_proj.to_device(d)?; self.input_layernorm.to_device(d)?; self.post_attention_layernorm.to_device(d)?; Ok(()) }
    pub fn clear(&mut self) { if let Some(ref mut l) = self.linear_attn { l.clear(); } if let Some(ref mut s) = self.self_attn { s.clear(); } self.mlp_gate_proj.clear(); self.mlp_up_proj.clear(); self.mlp_down_proj.clear(); self.input_layernorm.clear(); self.post_attention_layernorm.clear(); }
    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = xs.clone(); let mut x = self.input_layernorm.forward(xs)?;
        if self.layer_type == "linear_attention" { if let Some(ref mut l) = self.linear_attn { x = l.forward(&x, mask)?; } }
        else { if let (Some(ref mut s), Some(c), Some(sn)) = (&mut self.self_attn, cos, sin) { x = s.forward(&x, c, sn, mask)?; } }
        let r2 = (x + r)?; let mut x = self.post_attention_layernorm.forward(&r2)?;
        let gate = self.mlp_gate_proj.forward(&x)?.silu()?;
        let up = self.mlp_up_proj.forward(&x)?;
        x = (gate * up)?; (self.mlp_down_proj.forward(&x)? + r2)
    }
}

pub struct QTextModel {
    embed_tokens: crate::models::qwen35::model::Qwen3_5Embedding,
    layers: Vec<QDecoderLayer>, norm: QRmsNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding, mrope_section: Vec<usize>,
}

impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        Ok(Self {
            embed_tokens: crate::models::qwen35::model::Qwen3_5Embedding::new(vb.pp("embed_tokens"), config.vocab_size, config.hidden_size)?,
            layers, norm: QRmsNorm::new(vb.get(config.hidden_size, "norm.weight")?, config.rms_norm_eps)?,
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta),
            mrope_section: config.rope_parameters.mrope_section.clone(),
        })
    }
    pub fn forward(&mut self, inputs: &Tensor, pos: &Tensor, offset: usize) -> Result<Tensor> {
        let (bs, sl, _) = inputs.dims3()?; let dev = inputs.device();
        let (cos, sin) = self.rotary_emb.forward(pos, inputs.dtype(), self.mrope_section.clone())?;
        let mut x = inputs.clone();
        let mask = if sl <= 1 && offset == 0 { None } else { Some(prepare_causal_attention_mask(bs, sl, offset, dev)?) };
        for layer in self.layers.iter_mut() {
            layer.to_device(dev)?;
            let lm = if layer.layer_type != "linear_attention" || (sl != 1 && bs != 1) { mask.as_ref() } else { None };
            x = layer.forward(&x, Some(&cos), Some(&sin), lm)?;
            layer.clear();
        }
        self.norm.to_device(dev)?; let x = self.norm.forward(&x)?; self.norm.clear(); Ok(x)
    }
}

pub struct QuantizedQwen3_5Model {
    config: Qwen3_5Config,
    language_model: QTextModel,
    lm_head: QLinear,
}

impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let lm = QTextModel::new(vb.pp("model.language_model"), &config.text_config)?;
        let head = QLinear::from_weights(lm.embed_tokens.embeddings().clone(), None, vb.device())?;
        Ok(Self { config, language_model: lm, lm_head: head })
    }
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let dev = input_ids.device(); self.language_model.embed_tokens.to_device(dev)?;
        let embeds = self.language_model.embed_tokens.forward(input_ids)?; self.language_model.embed_tokens.clear();
        let pos = Tensor::arange(offset as u32, (offset + input_ids.dim(1)?) as u32, dev)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?;
        let outputs = self.language_model.forward(&embeds, &pos, offset)?;
        let hidden = outputs.narrow(1, outputs.dim(1)? - 1, 1)?;
        self.lm_head.to_device(dev)?; let logits = self.lm_head.forward(&hidden)?; self.lm_head.clear(); Ok(logits)
    }
    pub fn clear_cache(&mut self) { for l in self.language_model.layers.iter_mut() { if let Some(ref mut la) = l.linear_attn { la.clear_cache(); } if let Some(ref mut sa) = l.self_attn { sa.clear_kv_cache(); } } }
}
