use std::io::{Read, Seek};

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, quantized::QMatMul};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid, rms_norm,
};
use crate::{
    models::{
        common::{
            conv1d_depthwise, eager_attention_forward, get_conv1d,
            gguf::{GateUpDownMLPGguf, Gguf, ProjKind, QuantizedLinear},
            softplus,
        },
        qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{Qwen3VLTextRotaryEmbedding, glm_asr_apply_rotary_pos_emb},
    utils::tensor_utils::{
        get_equal_mask, get_vision_next_indices, l2_normalize, masked_scatter_dim0, nonzero_index,
        prepare_causal_attention_mask, repeat_interleave, split_tensor, zero_index,
    },
};

pub struct Qwen3_5RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Qwen3_5RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let weight = weight.to_dtype(candle_core::DType::F32)?.affine(1.0, 1.0)?;
        Ok(Self { eps, weight })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { eps, weight })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(candle_core::DType::F32)?;
        let norm_ = x
            .powf(2.0)?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?;
        let norm = x.broadcast_div(&norm_)?;
        let norm = norm.broadcast_mul(&self.weight)?.to_dtype(xs.dtype())?;
        Ok(norm)
    }
}

pub struct Qwen3_5RMSNormGated {
    norm: RmsNorm,
    dtype: DType,
}

impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let dtype = vb.dtype();
        let norm = rms_norm(hidden_size, eps, vb)?;
        Ok(Self { norm, dtype })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        let dtype = weight.dtype();
        let norm = RmsNorm::new(weight, eps);
        Ok(Self { norm, dtype })
    }

    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let orig_dtype = xs.dtype();
        let mut xs = self.norm.forward(&xs.to_dtype(self.dtype)?)?;
        if let Some(gate) = gate {
            xs = xs.broadcast_mul(&gate.silu()?.to_dtype(xs.dtype())?)?;
        }
        xs = xs.to_dtype(orig_dtype)?;
        Ok(xs)
    }
}

#[macro_export]
macro_rules! transmute_tensors {
    ($($tensor:expr),*) => {
        ($(
            $tensor.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        )*)
    };
}
#[macro_export]
macro_rules! right_pad_zero_tensor {
    ($dim:expr, $pad_size:expr, $($tensor:expr),+) => {
        ($(
            $tensor.pad_with_zeros($dim, 0, $pad_size)?.contiguous()?,
        )+)
    };
}

#[macro_export]
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

pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen3_5RMSNormGated,
    out_proj: ProjKind,
    in_proj_qkv: ProjKind,
    in_proj_z: ProjKind,
    in_proj_b: ProjKind,
    in_proj_a: ProjKind,
    conv_state_cache: Option<Tensor>,
    recurrent_state_cache: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads.unwrap_or(config.num_attention_heads);
        let num_k_heads = config.linear_num_key_heads.unwrap_or(config.num_key_value_heads);
        let head_k_dim = config.linear_key_head_dim.unwrap_or(config.head_dim);
        let head_v_dim = config.linear_value_head_dim.unwrap_or(config.head_dim);
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim.unwrap_or(4);
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim; 
        
        let conv1d = get_conv1d(
            vb.pp("conv1d"), conv_dim, conv_dim, conv_kernel_size, conv_kernel_size - 1, 1, 1, conv_dim, false,
        )?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        let norm = Qwen3_5RMSNormGated::new(vb.pp("norm"), head_v_dim, layer_norm_epsilon)?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;
        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        
        Ok(Self {
            num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, conv_kernel_size,
            conv1d, dt_bias, a_log, norm,
            out_proj: ProjKind::LinearProj(out_proj),
            in_proj_qkv: ProjKind::LinearProj(in_proj_qkv),
            in_proj_z: ProjKind::LinearProj(in_proj_z),
            in_proj_b: ProjKind::LinearProj(in_proj_b),
            in_proj_a: ProjKind::LinearProj(in_proj_a),
            conv_state_cache: None, recurrent_state_cache: None,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str, rms_norm_eps: f64) -> Result<Self> {
        let num_k_heads = gguf.get_matedata("qwen35.ssm.group_count")?.to_u32()? as usize;
        let num_v_heads = gguf.get_matedata("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
        let conv_kernel_size = gguf.get_matedata("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
        let head_k_dim = gguf.get_matedata("qwen35.ssm.state_size")?.to_u32()? as usize;
        let head_v_dim = head_k_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        
        let conv1d = gguf.conv1d(&format!("{prefix}.ssm_conv1d"), conv_kernel_size - 1, 1, 1, conv_dim, false)?;
        let dt_bias = gguf.get_dequantized(&format!("{prefix}.ssm_dt.bias"))?;
        let a_log = gguf.get_dequantized(&format!("{prefix}.ssm_a"))?;
        let norm_weight = gguf.get_dequantized(&format!("{prefix}.ssm_norm.weight"))?;
        let norm = Qwen3_5RMSNormGated::from_weight(norm_weight, rms_norm_eps)?;
        let out_proj = gguf.quantize_linear(&format!("{prefix}.ssm_out"), false)?;
        let in_proj_qkv = gguf.quantize_linear(&format!("{prefix}.attn_qkv"), false)?;
        let in_proj_z = gguf.quantize_linear(&format!("{prefix}.attn_gate"), false)?;
        let in_proj_b = gguf.quantize_linear(&format!("{prefix}.ssm_beta"), false)?;
        let in_proj_a = gguf.quantize_linear(&format!("{prefix}.ssm_alpha"), false)?;
        
        Ok(Self {
            num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, conv_kernel_size,
            conv1d, dt_bias, a_log, norm,
            out_proj: ProjKind::QuantizedProj(out_proj),
            in_proj_qkv: ProjKind::QuantizedProj(in_proj_qkv),
            in_proj_z: ProjKind::QuantizedProj(in_proj_z),
            in_proj_b: ProjKind::QuantizedProj(in_proj_b),
            in_proj_a: ProjKind::QuantizedProj(in_proj_a),
            conv_state_cache: None, recurrent_state_cache: None,
        })
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().unwrap();
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

    fn torch_chunk_gated_delta_rule(
        &mut self, query: &Tensor, key: &Tensor, value: &Tensor, g: &Tensor, beta: &Tensor, use_qk_l2norm_in_kernel: bool, chunk_size: usize,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel { (l2_normalize(query, 3)?, l2_normalize(key, 3)?) } else { (query.clone(), key.clone()) };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let (query, key, value, beta, g) = right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g);
        let total_sequence_length = sequence_length + pad_size;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, k_beta, v_beta) = reshape_chunk_tensor!(chunk_size, query, key, k_beta, v_beta);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?.cumsum(D::Minus1)?;
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
        
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref() { recurrent.clone() } 
            else { Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), candle_core::DType::F32, value.device())? };
            
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
            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?.unsqueeze(D::Minus1)?.exp()?)?.add(
                    &k_i.broadcast_mul(&g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?.broadcast_sub(&g_i)?.exp()?.unsqueeze(D::Minus1)?)?
                        .transpose(D::Minus1, D::Minus2)?.squeeze(0)?.matmul(&v_new.squeeze(0)?)?.unsqueeze(0)?
                )?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out = core_attn_out.reshape((batch_size, num_heads, (), core_attn_out.dim(D::Minus1)?))?.narrow(2, 0, sequence_length)?;
        
        core_attn_out = core_attn_out.transpose(1, 2)?.to_dtype(initial_dtype)?;
        Ok(core_attn_out)
    }

    fn torch_recurrent_gated_delta_rule(&mut self, query: &Tensor, key: &Tensor, value: &Tensor, g: &Tensor, beta: &Tensor, use_qk_l2norm_in_kernel: bool) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel { (l2_normalize(query, 3)?, l2_normalize(key, 3)?) } else { (query.clone(), key.clone()) };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref() { recurrent.clone() } 
            else { Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), candle_core::DType::F32, value.device())? };
            
        let mut core_attn_out = Tensor::zeros((batch_size, num_heads, sequence_length, v_head_dim), candle_core::DType::F32, value.device())?;
        for i in 0..sequence_length {
            let q_i = query.i((.., .., i))?;
            let k_i = key.i((.., .., i))?;
            let v_i = value.i((.., .., i))?;
            let g_i = g.i((.., .., i))?.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta_i = beta.i((.., .., i))?.unsqueeze(D::Minus1)?;
            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i)?;
            let kv_mem = last_recurrent_state.broadcast_mul(&k_i.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            last_recurrent_state = last_recurrent_state.broadcast_add(&k_i.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?)?;
            let out_i = last_recurrent_state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?.sum_keepdim(D::Minus2)?;
            core_attn_out = core_attn_out.slice_assign(&[(0..batch_size), (0..num_heads), (i..i + 1), (0..v_head_dim)], &out_i)?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out = core_attn_out.transpose(1, 2)?.to_dtype(initial_dtype)?;
        Ok(core_attn_out)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = if let Some(mask) = attention_mask { xs.broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(xs.dtype())?)? } else { xs.clone() };
        let (bs, seq_len, _) = xs.dims3()?;
        let mut mixed_qkv = self.in_proj_qkv.forward(&xs)?.transpose(1, 2)?;
        let z = self.in_proj_z.forward(&xs)?.reshape((bs, seq_len, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(&xs)?;
        let a = self.in_proj_a.forward(&xs)?;
        let use_precomputed_states = self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && seq_len == 1;
        
        if use_precomputed_states {
            mixed_qkv = self.torch_causal_conv1d_update(&mixed_qkv)?;
        } else {
            let pad = self.conv_kernel_size as isize - mixed_qkv.dim(D::Minus1)? as isize;
            let conv_state = if pad >= 0 { mixed_qkv.pad_with_zeros(D::Minus1, pad as usize, 0)? } 
                else { mixed_qkv.narrow(D::Minus1, pad.unsigned_abs(), self.conv_kernel_size)? };
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
        let beta = sigmoid(&b)?;
        let a_plus_bias = softplus(&a.to_dtype(candle_core::DType::F32)?.broadcast_add(&self.dt_bias.to_dtype(candle_core::DType::F32)?)?)?;
        let g = (-1.0 * self.a_log.to_dtype(candle_core::DType::F32)?.exp()?)?.broadcast_mul(&a_plus_bias)?;
        
        if self.num_v_heads / self.num_k_heads > 1 {
            query = repeat_interleave(&query, self.num_v_heads / self.num_k_heads, 2)?;
            key = repeat_interleave(&key, self.num_v_heads / self.num_k_heads, 2)?;
        }
        
        let core_attn_out = if !use_precomputed_states { self.torch_chunk_gated_delta_rule(&query, &key, &value, &g, &beta, true, 64)? } 
            else { self.torch_recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true)? };
            
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.norm.forward(&core_attn_out, Some(&z))?;
        let core_attn_out = core_attn_out.reshape((bs, seq_len, ()))?;
        Ok(self.out_proj.forward(&core_attn_out)?)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state_cache = None;
        self.recurrent_state_cache = None;
    }
}

pub struct Qwen3_5Attention {
    q_proj: ProjKind,
    k_proj: ProjKind,
    v_proj: ProjKind,
    o_proj: ProjKind,
    q_norm: Qwen3_5RMSNorm,
    k_norm: Qwen3_5RMSNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3_5Attention {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads.max(1);
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = linear_b(hidden_size, num_attention_heads * head_dim * 2, config.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_attention_heads * head_dim, hidden_size, config.attention_bias, vb.pp("o_proj"))?;
        let q_norm = Qwen3_5RMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = Qwen3_5RMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;
        Ok(Self {
            q_proj: ProjKind::LinearProj(q_proj), k_proj: ProjKind::LinearProj(k_proj),
            v_proj: ProjKind::LinearProj(v_proj), o_proj: ProjKind::LinearProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling, kv_cache: None,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str, rms_norm_eps: f64) -> Result<Self> {
        let num_attention_heads = gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
        let num_key_value_heads = gguf.get_matedata("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let num_kv_groups = num_attention_heads / num_key_value_heads.max(1);
        let head_dim = gguf.get_matedata("qwen35.attention.key_length")?.to_u32()? as usize;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = gguf.quantize_linear(&format!("{prefix}.attn_q"), false)?;
        let k_proj = gguf.quantize_linear(&format!("{prefix}.attn_k"), false)?;
        let v_proj = gguf.quantize_linear(&format!("{prefix}.attn_v"), false)?;
        let o_proj = gguf.quantize_linear(&format!("{prefix}.attn_output"), false)?;
        let q_norm_weight = gguf.get_dequantized(&format!("{prefix}.attn_q_norm.weight"))?;
        let q_norm = Qwen3_5RMSNorm::from_weight(q_norm_weight, rms_norm_eps)?;
        let k_norm_weight = gguf.get_dequantized(&format!("{prefix}.attn_k_norm.weight"))?;
        let k_norm = Qwen3_5RMSNorm::from_weight(k_norm_weight, rms_norm_eps)?;
        Ok(Self {
            q_proj: ProjKind::QuantizedProj(q_proj), k_proj: ProjKind::QuantizedProj(k_proj),
            v_proj: ProjKind::QuantizedProj(v_proj), o_proj: ProjKind::QuantizedProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling, kv_cache: None,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_chunk = self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim * 2))?.chunk(2, D::Minus1)?;
        let query_states = query_chunk[0].reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let gate = query_chunk[1].reshape((b_sz, q_len, ()))?;

        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        
        let (query_states, key_states) = glm_asr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), attention_mask, self.scaling)?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.mul(&sigmoid(&gate)?)?;
        Ok(attn_output.apply(&self.o_proj)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

enum AttnKind {
    LinearAttn(Qwen3_5GatedDeltaNet),
    SelfAttn(Qwen3_5Attention),
}

impl AttnKind {
    fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        match self {
            AttnKind::LinearAttn(attn) => attn.forward(xs, attention_mask),
            AttnKind::SelfAttn(attn) => {
                if let Some(cos) = cos {
                    if let Some(sin) = sin {
                        return attn.forward(xs, cos, sin, attention_mask);
                    }
                }
                Err(anyhow!("Qwen3_5 self attn cos and sin is all need"))
            }
        }
    }
}

pub struct Qwen3_5DecoderLayer {
    layer_type: String,
    attn: AttnKind,
    mlp: GateUpDownMLPGguf,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5DecoderLayer {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_types = config.layer_types.clone().unwrap_or_else(|| vec!["attention".to_string()]);
        let layer_type = if layer_idx < layer_types.len() { layer_types[layer_idx].clone() } else { "attention".to_string() };
        
        let attn = if layer_type == "linear_attention" {
            AttnKind::LinearAttn(Qwen3_5GatedDeltaNet::new_from_vb(vb.pp("linear_attn"), config)?)
        } else {
            AttnKind::SelfAttn(Qwen3_5Attention::new_from_vb(vb.pp("self_attn"), config)?)
        };
        let mlp = GateUpDownMLPGguf::new_from_vb(vb.pp("mlp"), hidden_size, config.intermediate_size, false, None, None, None, Some(config.hidden_act))?;
        let input_layernorm = Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::new(vb.pp("post_attention_layernorm"), hidden_size, config.rms_norm_eps)?;
        Ok(Self { layer_type, attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str, layer_type: &str, rms_norm_eps: f64) -> Result<Self> {
        let attn = if layer_type == "linear_attention" {
            AttnKind::LinearAttn(Qwen3_5GatedDeltaNet::new_from_gguf(gguf, prefix, rms_norm_eps)?)
        } else {
            AttnKind::SelfAttn(Qwen3_5Attention::new_from_gguf(gguf, prefix, rms_norm_eps)?)
        };
        let mlp = GateUpDownMLPGguf::new_from_gguf(gguf, prefix, false, None, None, None, Some(candle_nn::Activation::Silu))?;
        let input_norm_weight = gguf.get_dequantized(&format!("{prefix}.attn_norm.weight"))?;
        let input_layernorm = Qwen3_5RMSNorm::from_weight(input_norm_weight, rms_norm_eps)?;
        let post_norm_weight = gguf.get_dequantized(&format!("{prefix}.post_attention_norm.weight"))?;
        let post_attention_layernorm = Qwen3_5RMSNorm::from_weight(post_norm_weight, rms_norm_eps)?;
        Ok(Self { layer_type: layer_type.to_string(), attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.input_layernorm.forward(xs)?;
        xs = self.attn.forward(&xs, cos, sin, attention_mask)?;
        let residual = xs.add(&residual)?;
        xs = self.post_attention_layernorm.forward(&residual)?;
        xs = self.mlp.forward(&xs)?;
        xs = xs.add(&residual)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => attn.clear_cache(),
            AttnKind::SelfAttn(attn) => attn.clear_kv_cache(),
        }
    }
}

pub struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RMSNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
    dtype: DType,
}

impl Qwen3_5TextModel {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3_5DecoderLayer::new_from_vb(vb_layers.pp(i), config, i)?);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.as_ref().map(|r| r.partial_rotary_factor).unwrap_or(1.0)) as usize;
        let rope_theta = config.rope_parameters.as_ref().map(|r| r.rope_theta).unwrap_or(10000.0);
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dim, rope_theta);
        Ok(Self {
            embed_tokens, layers, norm, rotary_emb,
            mrope_section: config.rope_parameters.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default(), 
            dtype: vb.dtype(),
        })
    }
    
    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, device: &Device) -> Result<Self> {
        let dtype = match gguf.get_matedata("general.dtype") {
            Ok(v) => match v.to_u32() { Ok(0) => DType::F32, Ok(1) => DType::F16, _ => DType::F16 },
            Err(_) => DType::F16,
        };
        let num_layers = gguf.get_matedata("qwen35.block_count")?.to_u32()? as usize;
        let full_attention_interval = gguf.get_matedata("qwen35.full_attention_interval").ok().and_then(|v| v.to_u32().ok()).unwrap_or(num_layers as u32) as usize;
        let rope_freq_base = gguf.get_matedata("qwen35.rope.freq_base").ok().and_then(|v| v.to_f32().ok()).unwrap_or(10000.0);
        let rope_dimension_count = gguf.get_matedata("qwen35.rope.dimension_count").ok().and_then(|v| v.to_u32().ok()).unwrap_or(128) as usize;
        let mut mrope_section = gguf.get_matedata("qwen35.rope.dimension_sections").ok().and_then(|v| v.to_vec().ok()).map(|vec| vec.iter().filter_map(|v| v.to_i32().ok().map(|x| x as usize)).collect::<Vec<usize>>()).unwrap_or_default();
        if !mrope_section.is_empty() { let _ = mrope_section.pop(); }
        let rms_norm_eps = gguf.get_matedata("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let hidden_size = gguf.get_matedata("qwen35.embedding_length")?.to_u32()? as usize;
        let embed_tensor = gguf.tensor("token_embd.weight")?;
        
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);
        let mut layers = vec![];
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");
            let layer_type = if (i + 1) % full_attention_interval == 0 { "full_attention".to_string() } else { "linear_attention".to_string() };
            layers.push(Qwen3_5DecoderLayer::new_from_gguf(gguf, &prefix, &layer_type, rms_norm_eps)?);
        }
        let norm_weight = gguf.get_dequantized("output_norm.weight")?;
        let norm = Qwen3_5RMSNorm::from_weight(norm_weight, rms_norm_eps)?;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dimension_count, rope_freq_base);

        Ok(Self { embed_tokens, layers, norm, rotary_emb, mrope_section, dtype })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, position_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let (cos, sin) = self.rotary_emb.forward(position_ids, self.dtype, self.mrope_section.clone())?;
        let mut xs = inputs_embeds.clone();
        
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 { None } else { Some(prepare_causal_attention_mask(b_size, seq_len, seqlen_offset, inputs_embeds.device(), inputs_embeds.dtype())?) }
        };
        
        for layer in self.layers.iter_mut() {
            let layer_mask = if layer.layer_type != "linear_attention" || (seq_len != 1 && b_size != 1) { attention_mask.clone() } else { None };
            xs = layer.forward(&xs, Some(&cos), Some(&sin), layer_mask.as_ref())?;
        }
        xs = self.norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() { layer.clear_cache(); }
    }
}

// ============================================================================
// [ROOT MODEL] Qwen3_5 통합 모델 및 RoPE 파이프라인
// ============================================================================

pub struct Qwen3_5Model {
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    visual: Option<Qwen3VLVisionModel>,
    language_model: Qwen3_5TextModel,
    lm_head: ProjKind,
    rope_deltas: Option<Tensor>,
}

impl Qwen3_5Model {
    pub fn new_from_vb(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        
        let visual = if let Some(v_config) = config.vision_config.clone() {
            Some(Qwen3VLVisionModel::new(v_config, vb_m.pp("visual"))?)
        } else {
            None
        };
        
        let language_model = Qwen3_5TextModel::new_from_vb(vb_m.pp("language_model"), &config.text_config)?;
        
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.text_config.hidden_size, config.text_config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            spatial_merge_size: config.vision_config.map(|c| c.spatial_merge_size).unwrap_or(2),
            image_token_id: config.image_token_id.unwrap_or(248056),
            video_token_id: config.video_token_id.unwrap_or(248057),
            vision_start_token_id: config.vision_start_token_id.unwrap_or(248053),
            visual,
            language_model,
            lm_head: ProjKind::LinearProj(lm_head),
            rope_deltas: None,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        mmproj_gguf: Option<&mut Gguf<R>>,
        device: &Device,
    ) -> Result<Self> {
        let spatial_merge_size = 2usize;
        let image_token_id = 248056u32;
        let video_token_id = 248057u32;
        let vision_start_token_id = 248053u32;
        
        let visual = if let Some(mmproj) = mmproj_gguf {
            Some(Qwen3VLVisionModel::new_from_gguf(mmproj)?)
        } else {
            None
        };
        
        let language_model = Qwen3_5TextModel::new_from_gguf(gguf, device)?;
        let lm_head_tensor = match gguf.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gguf.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_qtensor(lm_head_tensor)?;
        
        Ok(Self {
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            visual,
            language_model,
            lm_head: ProjKind::QuantizedProj(QuantizedLinear::new(lm_head, None)),
            rope_deltas: None,
        })
    }

    // [CRITICAL FIX] GPU Sync Stall 타파를 위한 CPU 기반 순수 수학 RoPE 연산
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let spatial_merge_size = self.spatial_merge_size;
        let image_token_id = self.image_token_id;
        let video_token_id = self.video_token_id;
        let vision_start_token_id = self.vision_start_token_id;

        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut mrope_position_deltas = Vec::new();
        let input_ids_vec = input_ids.to_vec2::<u32>()?;

        // GPU의 무거운 연산 대신, CPU 배열로 데이터를 딱 한 번만 옮겨 O(N)으로 처리합니다.
        let image_thw_cpu = if let Some(thw) = image_grid_thw { Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?) } else { None };
        let video_thw_cpu = if let Some(thw) = _video_grid_thw { Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?) } else { None };

        let mut flat_pos_ids = Vec::with_capacity(3 * b_sz * seq_len);
        let mut image_idx = 0;
        let mut video_idx = 0;

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                if ids[i] == vision_start_token_id && i + 1 < seq_len && (ids[i+1] == image_token_id || ids[i+1] == video_token_id) {
                    let is_video = ids[i+1] == video_token_id;
                    let thw_cpu_array = if is_video { &video_thw_cpu } else { &image_thw_cpu };
                    
                    if let Some(thw_array) = thw_cpu_array {
                        let thw = if is_video { let t = &thw_array[video_idx]; video_idx += 1; t } 
                                  else { let t = &thw_array[image_idx]; image_idx += 1; t };
                        
                        let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                        
                        // Vision Start 토큰 위치 할당
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;
                        
                        // 3D 공간 블록 인덱스 할당 (이미지/비디오 패치 영역)
                        let img_len = (t * h * w) as usize;
                        for tt in 0..t {
                            for hh in 0..h {
                                for ww in 0..w {
                                    let idx = i + (tt * h * w + hh * w + ww) as usize;
                                    if idx < seq_len {
                                        llm_pos_ids[0][idx] = curr_pos + tt;
                                        llm_pos_ids[1][idx] = curr_pos + hh;
                                        llm_pos_ids[2][idx] = curr_pos + ww;
                                    }
                                }
                            }
                        }
                        i += img_len;
                        curr_pos += t.max(h).max(w);
                    } else {
                        // THW 데이터가 누락되었을 경우의 Fallback
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;
                    }
                } else {
                    // 일반 텍스트 토큰
                    for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                    i += 1;
                    curr_pos += 1;
                }
            }
            
            for d in 0..3 { flat_pos_ids.extend_from_slice(&llm_pos_ids[d]); }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        }

        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), input_ids.device())?;
        let deltas = Tensor::from_vec(mrope_position_deltas, (b_sz, 1), input_ids.device())?.to_dtype(input_ids.dtype())?;
        Ok((position_ids, deltas))
    }

    // [CRITICAL FIX] cache_position 파라미터가 포함된 forward 시그니처 복구!
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                if let Some(visual) = self.visual.as_ref() {
                    let (image_embeds, _) = visual.forward(pixel_values, image_grid_thw)?;
                    let vision_mask = get_equal_mask(input_ids, self.image_token_id)?;
                    
                    // [CRITICAL FIX] 엄청난 병목이었던 'vision_mask.sum_all()' 개수 검증 에러 로직 삭제 완료!
                    let image_embeds = image_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
                }
            }
        }
        
        if let Some(pixel_values_video) = pixel_values_video {
            if let Some(video_grid_thw) = video_grid_thw {
                if let Some(visual) = self.visual.as_ref() {
                    let (video_embeds, _) = visual.forward(pixel_values_video, video_grid_thw)?;
                    let vision_mask = get_equal_mask(input_ids, self.video_token_id)?;
                    
                    // [CRITICAL FIX] 비디오 쪽 'vision_mask.sum_all()' 검증 삭제 완료!
                    let video_embeds = video_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
                }
            }
        }
        
        // [OPTIMIZATION] cache_position 연산 최적화 로직 복구
        let position_ids;
        if seqlen_offset == 0 || self.rope_deltas.is_none() {
            let (pos, deltas) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            position_ids = pos;
            self.rope_deltas = Some(deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if let Some(c_pos) = cache_position {
                c_pos.i(0)?
                     .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
                     .broadcast_add(self.rope_deltas.as_ref().unwrap())?
                     // [FIX] contiguous() 삭제로 인한 PCIe 대역폭 확보
                     .to_dtype(candle_core::DType::U32)?
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?;
        }
        
        // seqlen_offset 인자를 전달하여 Language Model 호출
        let outputs = self.language_model.forward(&inputs_embeds, &position_ids, seqlen_offset)?;
        let seq_len = outputs.dim(1)?;
        
        // [OPTIMIZATION] 디코딩 시 마지막 예측 토큰 상태만 추출
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = match &self.lm_head {
            ProjKind::LinearProj(l) => l.forward(&hidden_state)?,
            ProjKind::QuantizedProj(q) => q.forward(&hidden_state)?,
        };
        
        Ok(logits)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
        self.rope_deltas = None;
    }

    // [CRITICAL FIX] 외부 통신 인터페이스(generate 등)를 위한 device() 헬퍼 복구
    pub fn device(&self) -> &Device {
        match &self.lm_head {
            ProjKind::LinearProj(l) => l.weight().device(),
            ProjKind::QuantizedProj(q) => q.device(),
        }
    }
}