use std::io::{Read, Seek};

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, quantized::QMatMul};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid,
};

use crate::models::qwen::quantized_model::{KVBlock, KVLocation, KVRegistry};

use crate::{
    models::{
        common::{
            conv1d_depthwise, get_conv1d,
            gguf::{GateUpDownMLPGguf, Gguf, ProjKind, QuantizedLinear},
            softplus,
        },
        qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb},
    utils::tensor_utils::{
        l2_normalize, masked_scatter_dim0,
        prepare_causal_attention_mask, repeat_interleave, split_tensor,
    },
};

#[derive(Clone)]
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
        // F32로 저장된 가중치를 xs(BF16) 타입에 맞춰준 뒤 네이티브 커널 실행!
        let w = self.weight.to_dtype(xs.dtype())?;
        Ok(candle_nn::ops::rms_norm(xs, &w, self.eps as f32)?)
    }

    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
    }
    pub fn eps(&self) -> f64 { self.eps }
    
    // 현재 가중치가 1바이트 껍데기인지(비워져 있는지) 확인
    pub fn is_cleared(&self) -> bool { self.weight.elem_count() <= 1 }
}

pub struct Qwen3_5RMSNormGated {
    weight: Tensor,
    eps: f64,
    dtype: DType,
}

impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let dtype = vb.dtype();
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps, dtype })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        let dtype = weight.dtype();
        Ok(Self { weight, eps, dtype })
    }

    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let w = self.weight.to_dtype(xs.dtype())?;
        let mut out = candle_nn::ops::rms_norm(xs, &w, self.eps as f32)?;
        if let Some(gate) = gate {
            
            let gate_val = gate.to_dtype(candle_core::DType::F32)?.silu()?.to_dtype(xs.dtype())?;
            out = out.broadcast_mul(&gate_val)?;
        }
        Ok(out)
    }

    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
    }
    pub fn eps(&self) -> f64 { self.eps }
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
    pub is_state_dirty: bool,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size; 
        let num_v_heads = config.linear_num_value_heads; 
        let num_k_heads = config.linear_num_key_heads; 
        let head_k_dim = config.linear_key_head_dim; 
        let head_v_dim = config.linear_value_head_dim; 
        let key_dim = head_k_dim * num_k_heads; 
        let value_dim = head_v_dim * num_v_heads; 
        let conv_kernel_size = config.linear_conv_kernel_dim; 
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim; 
        
        
        let conv1d = get_conv1d(vb.pp("ssm_conv1d"), conv_dim, conv_dim, conv_kernel_size, 0, 1, 1, conv_dim, false)?;
        let dt_bias = vb.get(num_v_heads, "ssm_dt.bias")?;
        let a_log = vb.get(num_v_heads, "ssm_a")?;
        let norm = Qwen3_5RMSNormGated::new(vb.pp("ssm_norm"), value_dim, layer_norm_epsilon)?;

        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;
        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj: ProjKind::LinearProj(out_proj),
            in_proj_qkv: ProjKind::LinearProj(in_proj_qkv),
            in_proj_z: ProjKind::LinearProj(in_proj_z),
            in_proj_b: ProjKind::LinearProj(in_proj_b),
            in_proj_a: ProjKind::LinearProj(in_proj_a),
            conv_state_cache: None,
            recurrent_state_cache: None,
            is_state_dirty: false,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let num_k_heads = gguf.get_matedata("qwen35.ssm.group_count")?.to_u32()? as usize;
        let num_v_heads = gguf.get_matedata("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
        let conv_kernel_size = gguf.get_matedata("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
        let head_k_dim = gguf.get_matedata("qwen35.ssm.state_size")?.to_u32()? as usize;
        let head_v_dim = head_k_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d = gguf.conv1d(
            &format!("{prefix}.ssm_conv1d"),
            0, 
            1,
            1,
            conv_dim,
            false,
        )?;
        
        
        let dt_bias = gguf.get_dequantized_f16(&format!("{prefix}.ssm_dt.bias"))?.to_dtype(DType::F32)?;
        let a_log = gguf.get_dequantized_f16(&format!("{prefix}.ssm_a"))?.to_dtype(DType::F32)?;
        let norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.ssm_norm.weight"))?.to_dtype(DType::F32)?;
        let norm = Qwen3_5RMSNormGated::from_weight(norm_weight, rms_norm_eps)?;
        
        let out_proj = gguf.quantize_linear(&format!("{prefix}.ssm_out"), false)?;
        let in_proj_qkv = gguf.quantize_linear(&format!("{prefix}.attn_qkv"), false)?;
        let in_proj_z = gguf.quantize_linear(&format!("{prefix}.attn_gate"), false)?;
        let in_proj_b = gguf.quantize_linear(&format!("{prefix}.ssm_beta"), false)?;
        let in_proj_a = gguf.quantize_linear(&format!("{prefix}.ssm_alpha"), false)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj: ProjKind::QuantizedProj(out_proj),
            in_proj_qkv: ProjKind::QuantizedProj(in_proj_qkv),
            in_proj_z: ProjKind::QuantizedProj(in_proj_z),
            in_proj_b: ProjKind::QuantizedProj(in_proj_b),
            in_proj_a: ProjKind::QuantizedProj(in_proj_a),
            conv_state_cache: None,
            recurrent_state_cache: None,
            is_state_dirty: false,
        })
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().unwrap();
        let seq_len = xs.dim(2)?; 
        let state_len = conv_state.dim(D::Minus1)?;
        let take_len = self.conv_kernel_size - 1; 
        
        let state_to_use = if state_len > take_len {
            conv_state.narrow(D::Minus1, state_len - take_len, take_len)?
        } else if state_len < take_len {
            conv_state.pad_with_zeros(D::Minus1, take_len - state_len, 0)?
        } else {
            conv_state.clone()
        };
        
        let conv_state_new = Tensor::cat(&[&state_to_use, xs], D::Minus1)?;
        
        let next_cache_len = conv_state_new.dim(D::Minus1)?;
        let conv_update = conv_state_new.narrow(D::Minus1, next_cache_len - take_len, take_len)?;
        self.conv_state_cache = Some(conv_update);
        
        let out = conv1d_depthwise(&conv_state_new, self.conv1d.weight(), self.conv1d.bias())?;
        
        let out_len = out.dim(D::Minus1)?;
        let final_out = if out_len > seq_len {
            out.narrow(D::Minus1, out_len - seq_len, seq_len)?
        } else {
            out
        };
        
        Ok(final_out.silu()?)
    }

    fn torch_chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let (query, key, value, beta, g) =
            right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g);
        let total_sequence_length = sequence_length + pad_size;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, k_beta, v_beta) =
            reshape_chunk_tensor!(chunk_size, query, key, k_beta, v_beta);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?;
        let g = g.cumsum(D::Minus1)?;
        let decay_mask = g
            .unsqueeze(D::Minus1)?
            .broadcast_sub(&g.unsqueeze(D::Minus2)?)?
            .exp()?
            .to_dtype(candle_core::DType::F32)?;
            
        
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as(decay_mask.shape())?;
        
        let on_false = decay_mask.zeros_like()?;
        let decay_mask = tril_mask.where_cond(&decay_mask, &on_false)?.contiguous()?;
        
        let mut attn = k_beta.squeeze(0)?.contiguous()?.matmul(&key.squeeze(0)?.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.unsqueeze(0)?.mul(&decay_mask)?.affine(-1.0, 0.0)?;
        
        
        let mask = Tensor::triu2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as(decay_mask.shape())?;
        
        attn = mask.where_cond(&on_false, &attn)?;
        let (d0, d1, d2, _, _) = attn.dims5()?;
        for i in 1..chunk_size {
            let row = attn.i((.., .., .., i, ..i))?.contiguous()?;
            let sub = attn.i((.., .., .., ..i, ..i))?.contiguous()?;
            let attn_i = row
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&sub)?
                .sum(D::Minus2)?
                .add(&row)?
                .unsqueeze(D::Minus2)?;
            attn = attn.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
        }
        let attn = attn
            .broadcast_add(&Tensor::eye(chunk_size, attn.dtype(), attn.device())?)?
            .contiguous()?;
        
        let value = attn.squeeze(0)?.matmul(&v_beta.squeeze(0)?)?.unsqueeze(0)?;
        let k_cumdecay = attn
            .squeeze(0)?
            .matmul(
                &k_beta
                    .broadcast_mul(&g.exp()?.unsqueeze(D::Minus1)?)?
                    .squeeze(0)?,
            )?
            .unsqueeze(0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref()
        {
            recurrent.clone()
        } else {
            Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                candle_core::DType::F32,
                value.device(),
            )?
        };

        let mut core_attn_out = value.zeros_like()?;
        
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as((batch_size, num_heads, chunk_size, chunk_size))?;
        let on_false = tril_mask.zeros_like()?.to_dtype(candle_core::DType::F32)?;
        let last_dim = core_attn_out.dim(D::Minus1)?;
        for i in 0..total_sequence_length / chunk_size {
            let q_i = query.i((.., .., i))?.contiguous()?;
            let k_i = key.i((.., .., i))?.contiguous()?;
            let v_i = value.i((.., .., i))?.contiguous()?;
            let g_i = g.i((.., .., i))?.contiguous()?;
            let attn = q_i
                .matmul(&k_i.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
                .mul(&decay_mask.i((.., .., i))?)?;
            let attn = tril_mask.where_cond(&attn, &on_false)?.contiguous()?;
            let v_prime = k_cumdecay.i((.., .., i))?.matmul(&last_recurrent_state)?;
            let v_new = v_i.sub(&v_prime)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.unsqueeze(D::Minus1)?.exp()?)?
                .matmul(&last_recurrent_state)?;
            let out_i = attn_inter.add(&attn.matmul(&v_new)?)?.unsqueeze(2)?;
            core_attn_out = core_attn_out.slice_assign(
                &[
                    (0..batch_size),
                    (0..num_heads),
                    (i..i + 1),
                    (0..chunk_size),
                    (0..last_dim),
                ],
                &out_i,
            )?;
            let g_i_last_dim = g_i.dim(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(
                    &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                        .unsqueeze(D::Minus1)?
                        .exp()?,
                )?
                .add(
                    &k_i.broadcast_mul(
                        &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )?
                    .transpose(D::Minus1, D::Minus2)?
                    .squeeze(0)?
                    .matmul(&v_new.squeeze(0)?)?
                    .unsqueeze(0)?,
                )?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out =
            core_attn_out.reshape((batch_size, num_heads, (), core_attn_out.dim(D::Minus1)?))?;
        core_attn_out = core_attn_out.narrow(2, 0, sequence_length)?;
        core_attn_out = core_attn_out
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    fn torch_recurrent_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        
        let (batch_size, sequence_length, num_heads, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let scale = 1.0 / (k_head_dim as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref() {
            recurrent.clone()
        } else {
            Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), candle_core::DType::F32, value.device())?
        };

        if sequence_length == 1 {
            
            let q_i = query.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            let k_i = key.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            let v_i = value.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            
            let g_i = g.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.contiguous()?;
            let beta_i = beta.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?.unsqueeze(D::Minus1)?.contiguous()?;
            
            // println!("[DEBUG-CONTIG] SSM Fast-Path Q: {}, K: {}, V: {}", q_i.is_contiguous(), k_i.is_contiguous(), v_i.is_contiguous());

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i)?;
            let kv_mem = last_recurrent_state.broadcast_mul(&k_i.unsqueeze(D::Minus1)?.contiguous()?)?.sum(D::Minus2)?;
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            last_recurrent_state = last_recurrent_state.broadcast_add(
                &k_i.unsqueeze(D::Minus1)?.contiguous()?.broadcast_mul(&delta.unsqueeze(D::Minus2)?.contiguous()?)?,
            )?;
            let out_i = last_recurrent_state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?.contiguous()?)?.sum_keepdim(D::Minus2)?;
            
            self.recurrent_state_cache = Some(last_recurrent_state);
            
            
            return Ok(out_i.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?); 
        }

        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let mut core_attn_out = Tensor::zeros(
            (batch_size, num_heads, sequence_length, v_head_dim),
            candle_core::DType::F32,
            value.device(),
        )?;
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
        core_attn_out = core_attn_out.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let dev = xs.device();
        let dtype = xs.dtype();
        
        if let Some(conv) = self.conv_state_cache.take() {
            self.conv_state_cache = Some(conv.to_device(dev)?.to_dtype(dtype)?);
        }
        if let Some(rec) = self.recurrent_state_cache.take() {
            self.recurrent_state_cache = Some(rec.to_device(dev)?.to_dtype(candle_core::DType::F32)?);
        }

        let (bs, seq_len, _) = xs.dims3()?;
        
        
        let mut mixed_qkv = self.in_proj_qkv.forward(xs)?
            .to_dtype(dtype)? // 👈 SSM 입력 타입 보정
            .transpose(1, 2)?
            .contiguous()?; 
        
        let z = self.in_proj_z.forward(xs)?
            .to_dtype(dtype)? // 👈 SSM 게이트 타입 보정
            .reshape((bs, seq_len, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(xs)?.to_dtype(dtype)?;
        let a = self.in_proj_a.forward(xs)?.to_dtype(dtype)?;
        let use_precomputed_states =
            self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && seq_len == 1;
        if use_precomputed_states {
            mixed_qkv = self.torch_causal_conv1d_update(&mixed_qkv)?;
        } else {
            let take_len = self.conv_kernel_size - 1;
            let (bs, dim, seq_len) = mixed_qkv.dims3()?;
            
            let next_conv_state = if seq_len >= take_len {
                mixed_qkv.narrow(D::Minus1, seq_len - take_len, take_len)?
            } else {
                mixed_qkv.pad_with_zeros(D::Minus1, take_len - seq_len, 0)?
            };
            
            let prev_causal = if let Some(prev_state) = &self.conv_state_cache {
                let prev_len = prev_state.dim(D::Minus1)?;
                if prev_len >= take_len {
                    prev_state.narrow(D::Minus1, prev_len - take_len, take_len)?
                } else {
                    prev_state.pad_with_zeros(D::Minus1, take_len - prev_len, 0)?
                }
            } else {
                Tensor::zeros((bs, dim, take_len), mixed_qkv.dtype(), mixed_qkv.device())?
            };
            
            mixed_qkv = Tensor::cat(&[&prev_causal, &mixed_qkv], D::Minus1)?;
            self.conv_state_cache = Some(next_conv_state);
            mixed_qkv = conv1d_depthwise(&mixed_qkv, self.conv1d.weight(), self.conv1d.bias())?;
            
            let out_len = mixed_qkv.dim(D::Minus1)?;
            mixed_qkv = if out_len > seq_len {
                mixed_qkv.narrow(D::Minus1, out_len - seq_len, seq_len)?
            } else {
                mixed_qkv
            };
            
            mixed_qkv = mixed_qkv.silu()?;
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        let qkv_split = split_tensor(
            &mixed_qkv,
            &[self.key_dim, self.key_dim, self.value_dim],
            D::Minus1,
        )?;
        
        
        let mut query = qkv_split[0].contiguous()?.reshape((bs, seq_len, (), self.head_k_dim))?;
        let mut key = qkv_split[1].contiguous()?.reshape((bs, seq_len, (), self.head_k_dim))?;
        let value = qkv_split[2].contiguous()?.reshape((bs, seq_len, (), self.head_v_dim))?;

        
        let beta = sigmoid(&b)?.to_dtype(dtype)?; 
        
        
        let a_plus_bias = softplus(
            &a.to_dtype(candle_core::DType::F32)?.broadcast_add(&self.dt_bias.to_dtype(candle_core::DType::F32)?)?,
        )?.to_dtype(dtype)?;
        let g = self.a_log.to_dtype(dtype)?.broadcast_mul(&a_plus_bias)?;

        if self.num_v_heads / self.num_k_heads > 1 {
            query = repeat_interleave(&query, self.num_v_heads / self.num_k_heads, 2)?;
            key = repeat_interleave(&key, self.num_v_heads / self.num_k_heads, 2)?;
        }
        let core_attn_out = if !use_precomputed_states {
            self.torch_chunk_gated_delta_rule(&query, &key, &value, &g, &beta, true, 64)?
        } else {
            self.torch_recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true)?
        };
        
        
        // println!("[DEBUG-CONTIG] SSM Output: {}, Z: {}", core_attn_out.is_contiguous(), z.is_contiguous());
        let core_attn_out = core_attn_out.contiguous()?.reshape(((), self.head_v_dim))?;
        let z = z.contiguous()?.reshape(((), self.head_v_dim))?;
        
        let core_attn_out = self.norm.forward(&core_attn_out, Some(&z))?.to_dtype(dtype)?;
        
        let core_attn_out = core_attn_out.contiguous()?.reshape((bs, seq_len, ()))?; 
        
        let output = self.out_proj.forward(&core_attn_out)?.to_dtype(dtype)?; 

        self.is_state_dirty = true; 
        Ok(output)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state_cache = None;
        self.recurrent_state_cache = None;
        self.is_state_dirty = false;
    }

    pub fn get_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        (self.conv_state_cache.clone(), self.recurrent_state_cache.clone())
    }

    pub fn set_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        self.conv_state_cache = conv;
        self.recurrent_state_cache = recurrent;
        self.is_state_dirty = false; 
    }

    pub fn clear_weights(&mut self) {
        let dummy_p = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.in_proj_qkv = dummy_p.clone();
        self.in_proj_z = dummy_p.clone();
        self.in_proj_b = dummy_p.clone();
        self.in_proj_a = dummy_p.clone();
        self.out_proj = dummy_p;
        
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.dt_bias = dummy_t.clone();
        self.a_log = dummy_t.clone();
        self.norm.clear();
        self.conv1d = candle_nn::Conv1d::new(dummy_t.clone(), None, candle_nn::Conv1dConfig::default());
    }

    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &Device) -> Result<()> {
        let conv_dim = self.key_dim * 2 + self.value_dim;
        
        
        let t_conv1d = ct.tensor(reader, &format!("{prefix}.ssm_conv1d.weight"), device)?;
        let conv1d_weight_raw = t_conv1d.dequantize_f16(device).or_else(|_| t_conv1d.dequantize(device))?.to_dtype(DType::F32)?.contiguous()?;
        
        let conv1d_weight = conv1d_weight_raw.reshape((conv_dim, 1, self.conv_kernel_size))?;
        
        self.conv1d = candle_nn::Conv1d::new(conv1d_weight, None, candle_nn::Conv1dConfig { 
            padding: 0, stride: 1, dilation: 1, groups: conv_dim, cudnn_fwd_algo: None 
        });
        
        
        let t_dt = ct.tensor(reader, &format!("{prefix}.ssm_dt.bias"), device)?;
        let dt_bias_raw = t_dt.dequantize_f16(device).or_else(|_| t_dt.dequantize(device))?;
        self.dt_bias = dt_bias_raw.to_dtype(DType::F32)?; 
        
        
        let t_a = ct.tensor(reader, &format!("{prefix}.ssm_a"), device)?;
        let a_log_raw = t_a.dequantize_f16(device).or_else(|_| t_a.dequantize(device))?;
        self.a_log = a_log_raw.to_dtype(DType::F32)?;  
        
        
        let t_norm = ct.tensor(reader, &format!("{prefix}.ssm_norm.weight"), device)?;
        let norm_weight = t_norm.dequantize_f16(device).or_else(|_| t_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.norm = Qwen3_5RMSNormGated::from_weight(norm_weight, 1e-6)?;
        
        self.out_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ssm_out.weight"), device)?)?, None));
        self.in_proj_qkv = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?)?, None));
        self.in_proj_z = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_gate.weight"), device)?)?, None));
        self.in_proj_b = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ssm_beta.weight"), device)?)?, None));
        self.in_proj_a = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ssm_alpha.weight"), device)?)?, None));
        Ok(())
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
    kv_blocks: Vec<KVBlock>,
    registry: KVRegistry,
    layer_idx: usize,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    
    
    pub vram_merged_k: Option<Tensor>,
    pub vram_merged_v: Option<Tensor>,
    pub merged_vram_block_count: usize,
}

impl Qwen3_5Attention {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize, registry: KVRegistry) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = linear_b(hidden_size, num_attention_heads * head_dim * 2, config.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_attention_heads * head_dim, hidden_size, config.attention_bias, vb.pp("o_proj"))?;
        let q_norm = Qwen3_5RMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = Qwen3_5RMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;

        Ok(Self {
            q_proj: ProjKind::LinearProj(q_proj),
            k_proj: ProjKind::LinearProj(k_proj),
            v_proj: ProjKind::LinearProj(v_proj),
            o_proj: ProjKind::LinearProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_session_id: None,
            active_kv_name: None,
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str, rms_norm_eps: f64, layer_idx: usize, registry: KVRegistry) -> Result<Self> {
        let num_attention_heads = gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
        let num_key_value_heads = gguf.get_matedata("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
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
            q_proj: ProjKind::QuantizedProj(q_proj),
            k_proj: ProjKind::QuantizedProj(k_proj),
            v_proj: ProjKind::QuantizedProj(v_proj),
            o_proj: ProjKind::QuantizedProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_session_id: None,
            active_kv_name: None,
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let target_dtype = xs.dtype();
        
        let query_chunk = self.q_proj.forward(xs)?
            .to_dtype(target_dtype)? // 👈 타입 불일치 방어 1
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim * 2))?
            .chunk(2, D::Minus1)?;
            
        
        let query_states = query_chunk[0].contiguous()?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let gate = query_chunk[1].contiguous()?.reshape((b_sz, q_len, ()))?;

        
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?.contiguous()?;
        let key_states = self.k_proj.forward(xs)?
            .to_dtype(target_dtype)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?.contiguous()?;
        let value_states = self.v_proj.forward(xs)?
            .to_dtype(target_dtype)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        
        
        // 전체 차원(256) 중 RoPE 차원(64)만 잘라내서 회전시킨 뒤, 나머지 차원과 다시 결합합니다!
        let rot_dim = cos.dim(D::Minus1)?;
        let (query_states, key_states) = if rot_dim < self.head_dim {
            let q_rot = query_states.narrow(D::Minus1, 0, rot_dim)?;
            let q_pass = query_states.narrow(D::Minus1, rot_dim, self.head_dim - rot_dim)?;
            let k_rot = key_states.narrow(D::Minus1, 0, rot_dim)?;
            let k_pass = key_states.narrow(D::Minus1, rot_dim, self.head_dim - rot_dim)?;
            
            let (q_rot, k_rot) = apply_rotary_pos_emb(&q_rot, &k_rot, cos, sin, false)?;
            
            (Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?, Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?)
        } else {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?
        };
        
        let dev = xs.device();
        let target_dtype = xs.dtype();

        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 1024usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    
                    
                    let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;

                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };

                        inner.k_cache = Some(Tensor::cat(&[&pk, &k_piece], 2)?.contiguous()?);
                        inner.v_cache = Some(Tensor::cat(&[&pv, &v_piece], 2)?.contiguous()?);
                        inner.len += take; tokens_to_process -= take; chunk_offset += take;
                        appended = true;
                        
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].token_len = inner.len;
                            if self.layer_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[self.layer_idx] = true; }
                        }
                    }
                }
            }
            if !appended {
                let take = tokens_to_process.min(1024);
                
                
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece); inner.v_cache = Some(v_piece);
                }
                
                let mut reg = self.registry.entries.write().unwrap();
                if index < reg.len() {
                    reg[index].token_start = current_total;
                    reg[index].token_len = take;
                    if self.layer_idx < reg[index].is_dirty.len() { reg[index].is_dirty[self.layer_idx] = true; }
                    reg[index].location[self.layer_idx] = KVLocation::VRAM;
                }
                self.kv_blocks.push(new_block);
                tokens_to_process -= take; chunk_offset += take;
            }
        }

        let total_tokens_now = seqlen_offset + q_len;

        
        // 이제 0.6B처럼 무조건 청크 단위 Online Softmax 로직을 타게 되어 수압이 0으로 방어됩니다.

        let mut out_res: Option<Tensor> = None;
        let mut m_n: Option<Tensor> = None;
        let mut l_n: Option<Tensor> = None;
        
        let q_aligned = (query_states * self.scaling)?;

        for block in &self.kv_blocks {
            let (index, b_off, b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            
            let (k_block, v_block, _is_temporary) = {
                let mut inner = block.inner.write().unwrap(); 
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    if inner.location == KVLocation::VRAM {
                        (k.clone(), v.clone(), false) // VRAM에 있다면 그대로 씀
                    } else {
                        // RAM에 있다면 GPU로 잠깐 복사해서 씀
                        (k.to_device(dev)?.to_dtype(target_dtype)?, v.to_device(dev)?.to_dtype(target_dtype)?, true)
                    }
                } else {
                    let mut k_cpu = None;
                    let mut v_cpu = None;
                    
                    {
                        let reg = self.registry.entries.read().unwrap();
                        let cache = reg[index].bitkv_cache.read().unwrap();
                        if let Some(m) = &cache[self.layer_idx] {
                            let kd_t = if dev.is_cpu() { m.k_data.to_dtype(DType::F32)? } else { m.k_data.clone() };
                            let vd_t = if dev.is_cpu() { m.v_data.to_dtype(DType::F32)? } else { m.v_data.clone() };
                            k_cpu = Some(kd_t); v_cpu = Some(vd_t);
                        }
                    }

                    // ... SSD 읽어오는 로직 생략 (기존과 동일하게 작동함) ...
                    if k_cpu.is_none() {
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let sid = self.active_session_id.as_deref().unwrap_or("default_session");
                        let kv_name_raw = self.active_kv_name.as_deref().unwrap_or("text");
                        let kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        
                        let mut path_candidates = Vec::new();
                        
                        
                        // 이 경로가 누락되어 Base 스냅샷을 놔두고 빈 껍데기(0.0)를 읽어오는 환각이 발생했습니다.
                        if let Some(p) = {
                            let reg = self.registry.entries.read().unwrap();
                            if index < reg.len() { reg[index].ssd_path.clone() } else { None }
                        } {
                            path_candidates.push(p);
                        }

                        path_candidates.push(kv_dir.join(format!("{}/inference/{}/b{}", sid, kv_type, b_off)));
                        path_candidates.push(kv_dir.join(format!("{}/reference/{}/b{}", sid, kv_type, b_off)));

                        for full_path in path_candidates {
                            let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                            for _retry in 0..3 {
                                if block_file.exists() {
                                    if let Ok(encrypted_content) = crate::utils::direct_loader::load_kv_block(&block_file) {
                                        if let Ok(content) = crate::utils::crypto::decrypt_data(&encrypted_content) {
                                            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                                let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                                let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                                
                                                if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                                    let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                                    let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                                    
                                                    let mut kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap();
                                                    let mut vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap();
                                                    
                                                    if dev.is_cpu() {
                                                        kd_t = kd_t.to_dtype(DType::F32)?;
                                                        vd_t = vd_t.to_dtype(DType::F32)?;
                                                    }

                                                    k_cpu = Some(kd_t.clone());
                                                    v_cpu = Some(vd_t.clone());
                                                    
                                                    let mut reg = self.registry.entries.write().unwrap();
                                                    if index < reg.len() { 
                                                        reg[index].ssd_path = Some(full_path.clone());
                                                        reg[index].location[self.layer_idx] = KVLocation::RAM; 
                                                        let mut cache = reg[index].bitkv_cache.write().unwrap();
                                                        cache[self.layer_idx] = Some(crate::models::qwen::quantized_model::BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: meta_os });
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                std::thread::sleep(std::time::Duration::from_millis(5));
                            }
                            if k_cpu.is_some() { break; }
                        }
                    }
                    
                    let fallback_shape = vec![1, self.num_key_value_heads, b_len, self.head_dim]; 
                    let k_safe = k_cpu.unwrap_or_else(|| Tensor::zeros(fallback_shape.as_slice(), DType::BF16, &Device::Cpu).unwrap());
                    let v_safe = v_cpu.unwrap_or_else(|| Tensor::zeros(fallback_shape.as_slice(), DType::BF16, &Device::Cpu).unwrap());
                    
                    let k_gpu = k_safe.to_device(dev)?.to_dtype(target_dtype)?;
                    let v_gpu = v_safe.to_device(dev)?.to_dtype(target_dtype)?;

                    
                    inner.k_cache = Some(k_safe);
                    inner.v_cache = Some(v_safe);
                    inner.location = KVLocation::RAM;

                    (k_gpu, v_gpu, true)
                }
            };

            let mut k = k_block;
            let mut v = v_block;

            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            k = k.contiguous()?;
            v = v.contiguous()?;

            let actual_kv_len = k.dim(2)?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            
            let mut s_chunk = q_aligned.matmul(&k_t)?;

            if let Some(mask) = &attention_mask {
                let mask_len = mask.dim(candle_core::D::Minus1)?;
                if b_off < mask_len {
                    let take = std::cmp::min(actual_kv_len, mask_len - b_off);
                    let chunk_mask = mask.narrow(candle_core::D::Minus1, b_off, take)?;
                    if take < actual_kv_len {
                        let left_masked = s_chunk.narrow(candle_core::D::Minus1, 0, take)?.broadcast_add(&chunk_mask)?;
                        let right_unmasked = s_chunk.narrow(candle_core::D::Minus1, take, actual_kv_len - take)?;
                        s_chunk = Tensor::cat(&[&left_masked, &right_unmasked], candle_core::D::Minus1)?;
                    } else {
                        s_chunk = s_chunk.broadcast_add(&chunk_mask)?; 
                    }
                }
            }

            
            let s_chunk_f32 = s_chunk.to_dtype(DType::F32)?;
            let m_j = s_chunk_f32.max_keepdim(candle_core::D::Minus1)?;
            
            
            // -inf - (-inf) = NaN이 발생하여 모델 뇌가 파괴되는 현상(!!!!!!!! 출력)을 원천 차단합니다.
            let safe_floor = Tensor::new(-10000.0_f32, m_j.device())?.broadcast_as(m_j.shape())?;
            let m_j_safe = m_j.maximum(&safe_floor)?;

            // m_j 대신 m_j_safe를 사용하여 빼기 연산 수행
            let p_j = s_chunk_f32.broadcast_sub(&m_j_safe)?.exp()?;
            let l_j = p_j.sum_keepdim(candle_core::D::Minus1)?;
            
            // v와 곱할 때는 타겟 타입(BF16)으로 맞추고, 다시 누적기(out_res)에 넣기 위해 F32로 올립니다.
            let out_j = p_j.to_dtype(v.dtype())?.matmul(&v)?;
            let out_j_f32 = out_j.to_dtype(DType::F32)?;

            match out_res.as_ref() {
                None => {
                    out_res = Some(out_j_f32); 
                    m_n = Some(m_j); 
                    l_n = Some(l_j);
                }
                Some(prev_out_f32) => {
                    let prev_m = m_n.as_ref().unwrap();
                    let prev_l = l_n.as_ref().unwrap();
                    
                    let m_new = prev_m.maximum(&m_j)?;
                    let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
                    let diff_new = m_j.broadcast_sub(&m_new)?.exp()?;
                    
                    let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
                    let out_new_f32 = prev_out_f32.broadcast_mul(&diff_old)?.add(&out_j_f32.broadcast_mul(&diff_new)?)?;
                    
                    out_res = Some(out_new_f32);
                    m_n = Some(m_new);
                    l_n = Some(l_new);
                }
            }
            
            drop(k);
            drop(v);
        }

        
        let attn_output = if let (Some(out_res_val), Some(l_val)) = (out_res, l_n) {
            out_res_val.broadcast_div(&l_val)?.to_dtype(target_dtype)? // 👈 F32 정밀도 연산 후 BF16 변환
        } else {
            return Err(anyhow!("No KV data processed"));
        };

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?.contiguous()?;
        
        // 💡 gate(F32/BF16)를 확실히 target_dtype으로 sigmoid 취한 뒤 다시 target_dtype으로 고정
        let gate_final = candle_nn::ops::sigmoid(&gate.to_dtype(target_dtype)?)?.to_dtype(target_dtype)?; 
        
        
        let attn_output = attn_output.mul(&gate_final)?;
        
        Ok(attn_output.apply(&self.o_proj)?.to_dtype(target_dtype)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_blocks.clear();

        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        let mut _current_total = 0;
        let mut to_remove = Vec::new();
        let total_blocks = self.kv_blocks.len();
        
        for i in 0..total_blocks {
            let block = &mut self.kv_blocks[i];
            let mut inner = block.inner.write().unwrap();
            
            if _current_total + inner.len <= len {
                _current_total += inner.len;
            } else {
                let keep_in_this_block = len - _current_total;
                if keep_in_this_block > 0 {
                    if inner.location == KVLocation::VRAM {
                        let (new_k, new_v) = if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            (Some(k.narrow(2, 0, keep_in_this_block)?), Some(v.narrow(2, 0, keep_in_this_block)?))
                        } else { (None, None) };
                        inner.k_cache = new_k;
                        inner.v_cache = new_v;
                    }
                    inner.len = keep_in_this_block;
                    _current_total += keep_in_this_block;
                    for j in (i + 1)..total_blocks { to_remove.push(j); }
                } else {
                    for j in i..total_blocks { to_remove.push(j); }
                }
                break;
            }
        }
        
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove { self.kv_blocks.remove(idx); }

        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;
        
        Ok(())
    }

    pub fn clear_weights(&mut self) {
        let dummy = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.q_proj = dummy.clone();
        self.k_proj = dummy.clone();
        self.v_proj = dummy.clone();
        self.o_proj = dummy;
        self.q_norm.clear();
        self.k_norm.clear();
        
        
        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;
    }

    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &Device) -> Result<()> {
        self.q_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?)?, None));
        self.k_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?)?, None));
        self.v_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?)?, None));
        self.o_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?)?, None));
        
        
        let t_q_norm = ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?;
        let q_norm_w = t_q_norm.dequantize_f16(device).or_else(|_| t_q_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.q_norm = Qwen3_5RMSNorm::from_weight(q_norm_w, self.q_norm.eps())?;
        
        
        let t_k_norm = ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?;
        let k_norm_w = t_k_norm.dequantize_f16(device).or_else(|_| t_k_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.k_norm = Qwen3_5RMSNorm::from_weight(k_norm_w, self.k_norm.eps())?;
        Ok(())
    }
}

enum AttnKind {
    LinearAttn(Qwen3_5GatedDeltaNet),
    SelfAttn(Qwen3_5Attention),
}

impl AttnKind {
    fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            AttnKind::LinearAttn(attn) => attn.forward(xs, attention_mask),
            AttnKind::SelfAttn(attn) => {
                if let (Some(c), Some(s)) = (cos, sin) {
                    attn.forward(xs, c, s, attention_mask, seqlen_offset)
                } else {
                    Err(anyhow!("Qwen3_5 self attn cos and sin is all need"))
                }
            }
        }
    }

    pub fn get_ssm_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.get_states()
        } else {
            (None, None)
        }
    }

    pub fn set_ssm_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.set_states(conv, recurrent);
        }
    }

    pub fn is_ssm_dirty(&self) -> bool {
        if let AttnKind::LinearAttn(attn) = self {
            attn.is_state_dirty
        } else {
            false
        }
    }

    pub fn set_ssm_dirty(&mut self, dirty: bool) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.is_state_dirty = dirty;
        }
    }
}

pub struct Qwen3_5DecoderLayer {
    pub layer_type: String,
    attn: AttnKind,
    mlp: crate::models::common::gguf::GateUpDownMLPGguf,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5DecoderLayer {
    pub fn new_from_vb(
        vb: VarBuilder,
        config: &Qwen3_5TextConfig,
        layer_idx: usize,
        registry: KVRegistry, 
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_type = config.layer_types[layer_idx].clone();
        let attn = if layer_type.eq("linear_attention") {
            let attn = Qwen3_5GatedDeltaNet::new_from_vb(vb.pp("linear_attn"), config)?;
            AttnKind::LinearAttn(attn)
        } else {
            let attn = Qwen3_5Attention::new_from_vb(vb.pp("self_attn"), config, layer_idx, registry)?;
            AttnKind::SelfAttn(attn)
        };
        let mlp = GateUpDownMLPGguf::new_from_vb(
            vb.pp("mlp"),
            hidden_size,
            config.intermediate_size,
            false,
            None,
            None,
            None,
            Some(config.hidden_act),
        )?;
        let input_layernorm =
            Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::new(
            vb.pp("post_attention_layernorm"),
            hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            layer_type,
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    
    pub fn new_skeleton<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        layer_type: &str,
        rms_norm_eps: f64,
        layer_idx: usize,
        registry: KVRegistry,
    ) -> Result<Self> {
        let dummy_p = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu)?;
        let dummy_norm = Qwen3_5RMSNorm { eps: rms_norm_eps, weight: dummy_t.clone() };

        let attn = if layer_type == "linear_attention" {
            let num_k_heads = gguf.get_matedata("qwen35.ssm.group_count")?.to_u32()? as usize;
            let num_v_heads = gguf.get_matedata("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
            let conv_kernel_size = gguf.get_matedata("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
            let head_k_dim = gguf.get_matedata("qwen35.ssm.state_size")?.to_u32()? as usize;
            let key_dim = head_k_dim * num_k_heads;
            let value_dim = head_k_dim * num_v_heads;
            
            AttnKind::LinearAttn(Qwen3_5GatedDeltaNet {
                num_v_heads, num_k_heads, head_k_dim, head_v_dim: head_k_dim, key_dim, value_dim, conv_kernel_size,
                conv1d: candle_nn::Conv1d::new(dummy_t.clone(), None, candle_nn::Conv1dConfig::default()),
                dt_bias: dummy_t.clone(), a_log: dummy_t.clone(),
                norm: Qwen3_5RMSNormGated { weight: dummy_t.clone(), eps: rms_norm_eps, dtype: DType::F32 },
                out_proj: dummy_p.clone(), in_proj_qkv: dummy_p.clone(), in_proj_z: dummy_p.clone(),
                in_proj_b: dummy_p.clone(), in_proj_a: dummy_p.clone(),
                conv_state_cache: None, recurrent_state_cache: None, is_state_dirty: false,
            })
        } else {
            let num_attention_heads = gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
            let num_key_value_heads = gguf.get_matedata("qwen35.attention.head_count_kv")?.to_u32()? as usize;
            let num_kv_groups = num_attention_heads / num_key_value_heads;
            let head_dim = gguf.get_matedata("qwen35.attention.key_length")?.to_u32()? as usize;
            let scaling = 1f64 / f64::sqrt(head_dim as f64);

            AttnKind::SelfAttn(Qwen3_5Attention {
                q_proj: dummy_p.clone(), k_proj: dummy_p.clone(), v_proj: dummy_p.clone(), o_proj: dummy_p.clone(),
                q_norm: dummy_norm.clone(), k_norm: dummy_norm.clone(),
                num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
                kv_blocks: Vec::new(), registry, layer_idx, active_session_id: None, active_kv_name: None,
                vram_merged_k: None, vram_merged_v: None, merged_vram_block_count: 0,
            })
        };

        
        let mlp = crate::models::common::gguf::GateUpDownMLPGguf::new_dummy(&candle_core::Device::Cpu);

        Ok(Self {
            layer_type: layer_type.to_string(),
            attn, mlp, input_layernorm: dummy_norm.clone(), post_attention_layernorm: dummy_norm,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        layer_type: &str,
        rms_norm_eps: f64,
        layer_idx: usize, 
        registry: KVRegistry, 
    ) -> Result<Self> {
        let attn = if layer_type.eq("linear_attention") {
            let attn = Qwen3_5GatedDeltaNet::new_from_gguf(gguf, prefix, rms_norm_eps)?;
            AttnKind::LinearAttn(attn)
        } else {
            let attn = Qwen3_5Attention::new_from_gguf(gguf, prefix, rms_norm_eps, layer_idx, registry)?;
            AttnKind::SelfAttn(attn)
        };
        let mlp = GateUpDownMLPGguf::new_from_gguf(
            gguf,
            prefix,
            false,
            None,
            None,
            None,
            Some(candle_nn::Activation::Silu),
        )?;
        
        let input_norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.attn_norm.weight"))?.to_dtype(candle_core::DType::BF16)?;
        let input_layernorm = Qwen3_5RMSNorm::from_weight(input_norm_weight, rms_norm_eps)?;
        let post_norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.post_attention_norm.weight"))?.to_dtype(candle_core::DType::BF16)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::from_weight(post_norm_weight, rms_norm_eps)?;
        Ok(Self {
            layer_type: layer_type.to_string(),
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let residual = xs.clone(); // xs는 BF16
        let mut xs = self.input_layernorm.forward(xs)?;
        xs = self.attn.forward(&xs, cos, sin, attention_mask, seqlen_offset)?;
        
        // 💡 attn 출력이 F32로 오염되었을 경우를 대비해 BF16으로 내린 뒤 더함
        let residual = xs.to_dtype(residual.dtype())?.add(&residual)?; 
        
        let mut xs = self.post_attention_layernorm.forward(&residual)?;
        xs = self.mlp.forward(&xs)?;
        
        // 💡 mlp 출력(F32 가능성 높음)을 BF16으로 내린 뒤 최종 더하기
        let xs = xs.to_dtype(residual.dtype())?.add(&residual)?;
        Ok(xs)
    }

    pub fn get_ssm_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        self.attn.get_ssm_states()
    }

    pub fn set_ssm_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        self.attn.set_ssm_states(conv, recurrent);
    }

    pub fn is_ssm_dirty(&self) -> bool {
        self.attn.is_ssm_dirty()
    }

    pub fn set_ssm_dirty(&mut self, dirty: bool) {
        self.attn.set_ssm_dirty(dirty);
    }

    pub fn clear_cache(&mut self) {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => { attn.clear_cache(); }
            AttnKind::SelfAttn(attn) => { attn.clear_kv_cache(); }
        }
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        if let AttnKind::SelfAttn(attn) = &mut self.attn {
            attn.truncate_kv_cache(len)?;
        }
        // SSM (LinearAttn)은 토큰 단위 롤백이 불가하므로 무시합니다.
        Ok(())
    }

    pub fn clear_weights(&mut self) {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => attn.clear_weights(),
            AttnKind::SelfAttn(attn) => attn.clear_weights(),
        }
        self.mlp.clear_weights();
        self.input_layernorm.clear();
        self.post_attention_layernorm.clear();
    }

    pub fn is_cleared(&self) -> bool { self.input_layernorm.is_cleared() }

    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &Device) -> Result<()> {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => attn.load_weights_inplace(ct, reader, prefix, device)?,
            AttnKind::SelfAttn(attn) => attn.load_weights_inplace(ct, reader, prefix, device)?,
        }
        self.mlp.load_weights_inplace(ct, reader, prefix, device)?;
        
        
        let t_in = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
        let in_norm_w = t_in.dequantize_f16(device).or_else(|_| t_in.dequantize(device))?.to_dtype(DType::F32)?;
        self.input_layernorm = Qwen3_5RMSNorm::from_weight(in_norm_w, self.input_layernorm.eps())?;
        
        
        let t_post = ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?;
        let post_norm_w = t_post.dequantize_f16(device).or_else(|_| t_post.dequantize(device))?.to_dtype(DType::F32)?;
        self.post_attention_layernorm = Qwen3_5RMSNorm::from_weight(post_norm_w, self.post_attention_layernorm.eps())?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------------
// [Qwen3_5TextModel] SSD 캐싱 관리자 연결
// ----------------------------------------------------------------------------------
pub struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RMSNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
    dtype: DType,
    pub registry: KVRegistry,
    pub current_kv_len: usize,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    
    // Mmap 핑퐁을 위한 메타데이터 필드
    pub mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    pub ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    pub base_name: String,
}

impl Qwen3_5TextModel {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let registry = KVRegistry::new(); // 장부 초기화
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let mut layer = Qwen3_5DecoderLayer::new_from_vb(vb_layers.pp(i), config, i, registry.clone())?;
            layer.clear_weights(); // 초기화 즉시 1바이트 껍데기로 만들어버림
            layers.push(layer);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta);
        Ok(Self {
            embed_tokens, layers, norm, rotary_emb, mrope_section: config.rope_parameters.mrope_section.clone(), dtype: vb.dtype(),
            registry, current_kv_len: 0, active_session_id: None, active_kv_name: None,
            mmap: None, ct: None, base_name: "model".to_string(), 
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>, 
        device: &Device,
        mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
        ct_handle: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>
    ) -> Result<Self> {
        let dtype = match gguf.get_matedata("general.dtype") {
            Ok(v) => match v.to_u32() as Result<u32, candle_core::Error> { Ok(0) => DType::F32, Ok(1) => DType::F16, _ => DType::F16 },
            Err(_) => DType::F16,
        };
        let num_layers = gguf.get_matedata("qwen35.block_count")?.to_u32()? as usize;
        let full_attention_interval = gguf.get_matedata("qwen35.full_attention_interval")?.to_u32()? as usize;
        let rope_freq_base = gguf.get_matedata("qwen35.rope.freq_base")?.to_f32()?;
        let rope_dimension_count = gguf.get_matedata("qwen35.rope.dimension_count")?.to_u32()? as usize;
        let mut mrope_section = gguf.get_matedata("qwen35.rope.dimension_sections")?.to_vec()?.iter().map(|v: &candle_core::quantized::gguf_file::Value| v.to_i32().map(|x| x as usize)).collect::<Result<Vec<usize>, candle_core::Error>>()?;
        let _ = mrope_section.pop();
        let rms_norm_eps = gguf.get_matedata("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let hidden_size = gguf.get_matedata("qwen35.embedding_length")?.to_u32()? as usize;
        
        let embed_tensor = gguf.tensor("token_embd.weight")?;
        
        
        let embed_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
        let embed_tokens = Embedding::new(
            embed_tensor.dequantize_f16(device).or_else(|_| embed_tensor.dequantize(device))?.to_dtype(embed_dtype)?, 
            hidden_size
        );
        
        
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
            let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" { fn malloc_trim(pad: usize) -> i32; }
            malloc_trim(0);
        }
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; }
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }

        let registry = KVRegistry::new();
        let mut layers = vec![];
        for i in 0..num_layers {
            let layer_type = if (i + 1) % full_attention_interval == 0 { "full_attention".to_string() } else { "linear_attention".to_string() };
            
            let layer = Qwen3_5DecoderLayer::new_skeleton(gguf, &layer_type, rms_norm_eps, i, registry.clone())?;
            layers.push(layer);
        }
        
        
        let norm_weight = gguf.get_dequantized_f16("output_norm.weight").or_else(|_| gguf.get_dequantized("output_norm.weight"))?.to_dtype(candle_core::DType::BF16)?;
        let norm = Qwen3_5RMSNorm::from_weight(norm_weight, rms_norm_eps)?;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dimension_count, rope_freq_base);

        Ok(Self {
            embed_tokens, layers, norm, rotary_emb, mrope_section, dtype,
            registry, current_kv_len: 0, active_session_id: None, active_kv_name: None,
            mmap: mmap_handle, ct: ct_handle, base_name: "model".to_string(),
        })
    }

    // 찰나의 순간에 디스크에서 가중치를 퍼 올리는 JIT 로더
    pub fn reload_layer(&mut self, layer_idx: usize, device: &Device) -> Result<()> {
        if let (Some(mmap), Some(ct)) = (self.mmap.as_ref(), self.ct.as_ref()) {
            let mut reader = std::io::Cursor::new(&mmap[..]);
            // Qwen 3.5 gguf 블록 접두어
            let prefix = format!("blk.{layer_idx}");
            self.layers[layer_idx].load_weights_inplace(ct, &mut reader, &prefix, device)?;
        }
        Ok(())
    }

    // 메모리는 유지하되 PCIe & SSD 병목은 완벽 차단한 궁극의 파이프라인
    pub async fn forward(&mut self, inputs_embeds: &Tensor, position_ids: &Tensor, seqlen_offset: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        self.active_session_id = session_id.clone();
        self.active_kv_name = kv_name.clone();

        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let is_decoding = seq_len <= 1; // 현재 AI가 대답 중인지 판별

        let (cos, sin) = self.rotary_emb.forward(position_ids, self.dtype, self.mrope_section.clone())?;
        
        let mut xs = if !inputs_embeds.device().same_device(&cos.device()) { 
            inputs_embeds.to_device(&cos.device())? 
        } else { 
            inputs_embeds.clone() 
        };
        xs = xs.to_dtype(self.dtype)?; 

        let attention_mask: Option<Tensor> = { 
            if seq_len <= 1 { None } 
            else { Some(prepare_causal_attention_mask(b_size, seq_len, seqlen_offset, xs.device())?) } 
        };
        let attention_mask = if let Some(m) = attention_mask { Some(m.to_dtype(self.dtype)?) } else { None };

        let total_layers = self.layers.len();

        for l_idx in 0..total_layers {
            // 가중치가 비워져 있을 때만 Mmap 로드! 
            if self.layers[l_idx].is_cleared() {
                self.reload_layer(l_idx, xs.device())?;
            }

            let layer = &mut self.layers[l_idx];

            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                attn.active_session_id = session_id.clone();
                attn.active_kv_name = kv_name.clone();
            }

            if layer.layer_type == "linear_attention" && seqlen_offset > 0 {
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if conv_opt.is_none() || rec_opt.is_none() {
                    if let Some(sid) = &session_id {
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        
                        let kv_name_raw = kv_name.as_deref().unwrap_or("text");
                        let mut safe_kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        if safe_kv_type == "inference" || safe_kv_type == "reference" || safe_kv_type.is_empty() {
                            safe_kv_type = "text";
                        }
                        
                        let path_candidates = vec![
                            kv_dir.join(format!("{}/inference/{}/ssm", sid, safe_kv_type)),
                            kv_dir.join(format!("{}/reference/{}/ssm", sid, safe_kv_type)),
                            kv_dir.join(format!("{}/inference/text/ssm", sid)), // 최후의 폴백
                        ];

                        let mut found_file = false;
                        
                        for ssm_dir in path_candidates {
                            let st_path = ssm_dir.join(format!("l{}.st", l_idx));
                            if st_path.exists() {
                                found_file = true;
                                let mut loaded = false;
                                for _retry in 0..3 {
                                    if let Ok(encrypted_data) = std::fs::read(&st_path) {
                                        if let Ok(plain_data) = crate::utils::crypto::decrypt_data(&encrypted_data) {
                                            // Device 강제 할당 제거: CPU에서 풀어서 안전하게 GPU로 넘깁니다.
                                            if let Ok(tensors) = candle_core::safetensors::load_buffer(&plain_data, &candle_core::Device::Cpu) {
                                                let loaded_conv = tensors.get("conv_state").map(|t| t.to_device(xs.device()).unwrap_or(t.clone()));
                                                let loaded_rec = tensors.get("recurrent_state").map(|t| t.to_device(xs.device()).unwrap_or(t.clone()));
                                                if loaded_conv.is_some() && loaded_rec.is_some() {
                                                    layer.set_ssm_states(loaded_conv, loaded_rec);
                                                    loaded = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(5));
                                }
                                if loaded { break; }
                            }
                        }
                        if !found_file {
                            println!("[WARNING] SSM state file completely missing for Layer {} (Session: {})! Model context destroyed.", l_idx, sid);
                        }
                    }
                }
            }

            let layer_mask = if layer.layer_type.ne("linear_attention") || (seq_len != 1 && b_size != 1) { attention_mask.clone() } else { None };
            
            xs = layer.forward(&xs, Some(&cos), Some(&sin), layer_mask.as_ref(), seqlen_offset)?;

            
            if layer.layer_type == "linear_attention" {
                let is_dirty_backup = layer.is_ssm_dirty(); 
                
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if let (Some(conv), Some(rec)) = (conv_opt, rec_opt) {
                    let safe_conv = conv.to_device(&candle_core::Device::Cpu).unwrap_or(conv);
                    let safe_rec = rec.to_device(&candle_core::Device::Cpu).unwrap_or(rec);
                    layer.set_ssm_states(Some(safe_conv), Some(safe_rec));
                }

                layer.set_ssm_dirty(is_dirty_backup); 
            }
            layer.clear_weights();

            
            if is_decoding {
                if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                    let mut reg = attn.registry.entries.write().unwrap();
                    let total_blocks = attn.kv_blocks.len();
                    
                    let mut garbage_bin = Vec::new();
                    
                    for (idx, block) in attn.kv_blocks.iter_mut().enumerate() {
                        if idx + 1 >= total_blocks { continue; } 
                        
                        let mut inner = block.inner.write().unwrap();
                        if inner.location == KVLocation::RAM && idx < reg.len() && reg[idx].ssd_path.is_some() {
                            garbage_bin.push((inner.k_cache.take(), inner.v_cache.take()));
                            inner.location = KVLocation::SSD;
                            
                            reg[idx].location[l_idx] = KVLocation::SSD;
                            let mut cache = reg[idx].bitkv_cache.write().unwrap();
                            cache[l_idx] = None;
                        }
                    }
                    
                    if !garbage_bin.is_empty() {
                        tokio::task::spawn_blocking(move || {
                            drop(garbage_bin);
                        });
                    }
                }
            }

            
            if let Some(sid) = &session_id {
                let kv_dir = crate::utils::paths::get_kv_dir(None);
                let kv_name_raw = kv_name.as_deref().unwrap_or("text");
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
                let sub_path = format!("{}/inference/{}", sid, kv_type);
                let base_dir = kv_dir.join(&sub_path);

                match &mut layer.attn {
                    AttnKind::SelfAttn(attn) => {
                        let mut dumps = Vec::new();
                        for block in attn.kv_blocks.iter_mut() {
                            let inner = block.inner.write().unwrap(); 
                            let is_full = inner.len == 1024;
                            
                            
                            // 오직 꽉 찬 블록만 '복사본'을 디스크로 보낼 뿐, VRAM에서 절대 삭제하지 않습니다!
                            let should_evacuate = is_full; 
                            
                            if should_evacuate && inner.k_cache.is_some() && inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                                let is_dirty = {
                                    let reg = attn.registry.entries.read().unwrap();
                                    if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] } else { true }
                                };

                                if is_dirty {
                                    let k = inner.k_cache.as_ref().unwrap();
                                    let v = inner.v_cache.as_ref().unwrap();
                                    let k_cpu = k.to_device(&candle_core::Device::Cpu).unwrap_or(k.clone());
                                    let v_cpu = v.to_device(&candle_core::Device::Cpu).unwrap_or(v.clone());
                                    let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                                    
                                    dumps.push((
                                        crate::models::qwen::generate::LayerKVDump {
                                            layer_idx: l_idx,
                                            k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                            v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                            k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                                            raw_k: Some(k_cpu.contiguous().unwrap_or(k_cpu.clone())),
                                            raw_v: Some(v_cpu.contiguous().unwrap_or(v_cpu.clone())),
                                        },
                                        inner.offset,
                                        inner.index
                                    ));
                                    
                                    let mut reg = attn.registry.entries.write().unwrap();
                                    if inner.index < reg.len() {
                                        reg[inner.index].is_dirty[l_idx] = false;
                                    }
                                }

                                
                                // VRAM 클리어 및 메모리 청소는 모든 문맥 파악이 끝난 뒤 `force_flush_all_active_blocks` 함수가 알아서 담당합니다.
                            }
                        }

                        if !dumps.is_empty() {
                            if let Some(tx) = crate::models::qwen::generate::BAKE_TX.get() {
                                let reg_clone = attn.registry.clone();
                                let sub_path_clone = sub_path.clone();
                                tauri::async_runtime::spawn(async move {
                                    for (dump, off, b_idx) in dumps {
                                        let sid = crate::models::qwen::generate::SLOT_MANAGER.acquire_write_slot(1024).await;
                                        let block_dir = base_dir.join(format!("b{}", off));
                                        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }
                                        
                                        crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                        if tx.send(crate::models::qwen::generate::SlotTask::Bake(crate::models::qwen::generate::BakeTask {
                                            slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path_clone.clone()), offset: off, layers: vec![dump],
                                            is_relay_baking: false, block_idx: Some(b_idx), registry: reg_clone.clone(),
                                        })).await.is_err() {
                                            crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                                            crate::models::qwen::generate::SLOT_MANAGER.release_slot(sid).await;
                                        }
                                    }
                                });
                            }
                        }
                    },
                    AttnKind::LinearAttn(_attn) => {
                        // 디코딩 중 SSM 디스크 I/O 스팸 방지
                    }
                }
            }
        }
        
        tokio::task::yield_now().await;
        xs = self.norm.forward(&xs)?;
        self.current_kv_len = seqlen_offset + seq_len;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
        self.current_kv_len = 0;
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(len)?;
        }
        self.current_kv_len = len;
        Ok(())
    }

    pub fn restore_kv_registry(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let index_path = kv_dir.join(kv_name).join("layer0.json");
        
        if !index_path.exists() { return Ok(()); }
        
        let index_json = String::from_utf8(crate::utils::direct_loader::load_kv_block(&index_path).unwrap_or_default()).unwrap_or_default();
        let index: LayerIndex = serde_json::from_str(&index_json).unwrap_or_else(|_| LayerIndex { layer_idx: 0, total_tokens: 0, blocks: vec![] });
        let total_tokens = index.total_tokens;

        {
            let mut reg = self.registry.entries.write().unwrap();
            let needed_blocks = (total_tokens + 1023) / 1024;
            while reg.len() < needed_blocks {
                let off = reg.len() * 1024;
                reg.push(crate::models::qwen::quantized_model::RegistryEntry::new(off, 0, 28));
            }
        }

        for layer in self.layers.iter_mut() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                let reg_len = self.registry.entries.read().unwrap().len();
                while attn.kv_blocks.len() < reg_len {
                    let idx = attn.kv_blocks.len();
                    let off = idx * 1024;
                    attn.kv_blocks.push(crate::models::qwen::quantized_model::KVBlock::new(
                        crate::models::qwen::quantized_model::KVLocation::SSD, idx, 0, off
                    ));
                }
            }
        }

        {
            let mut reg = self.registry.entries.write().unwrap();
            for (idx, entry) in reg.iter_mut().enumerate() {
                let off = idx * 1024;
                let b_len = if off + 1024 <= total_tokens { 1024 } else { total_tokens.saturating_sub(off) };
                entry.token_len = b_len;
                
                
                // 이 처리가 없으면 스냅샷을 로드하자마자 전체 문맥을 다시 디스크에 덮어쓰는 I/O 스팸이 발생합니다.
                entry.is_dirty.fill(false);
                entry.ssd_path = Some(kv_dir.join(kv_name).join(format!("b{}", off)));
                
                for layer in self.layers.iter_mut() {
                    if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                        if let Some(block) = attn.kv_blocks.get(idx) {
                            let mut inner = block.inner.write().unwrap();
                            inner.len = b_len;
                            inner.ssd_path = entry.ssd_path.clone();
                        }
                    }
                }
            }
        }
        self.current_kv_len = total_tokens;
        Ok(())
    }

    
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        use crate::models::qwen::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let mut block_groups: std::collections::HashMap<(usize, usize), Vec<LayerKVDump>> = std::collections::HashMap::new();
        let mut ssm_dumps = Vec::new();

        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                
                attn.vram_merged_k = None;
                attn.vram_merged_v = None;
                attn.merged_vram_block_count = 0;

                let mut gpu_k_list = Vec::new();
                let mut gpu_v_list = Vec::new();
                let mut target_indices = Vec::new();

                // 1. 읽기 락만 짧게 잡고 넘길 대상들을 수집 및 VRAM 선행 처리
                for (idx, block) in attn.kv_blocks.iter().enumerate() {
                    let inner = block.inner.read().unwrap(); 
                    let is_full = inner.len == 1024;
                    
                    if is_full && inner.k_cache.is_some() && inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                        let k = inner.k_cache.as_ref().unwrap();
                        let v = inner.v_cache.as_ref().unwrap();
                        
                        
                        let k_gpu = k.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| k.clone()).contiguous().unwrap_or_else(|_| k.clone());
                        let v_gpu = v.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| v.clone()).contiguous().unwrap_or_else(|_| v.clone());
                        
                        gpu_k_list.push(k_gpu);
                        gpu_v_list.push(v_gpu);
                        target_indices.push(idx);
                    }
                }

                // 2. 모인 VRAM 텐서가 있다면 통째로 합쳐서 한 방에 보냄
                if !gpu_k_list.is_empty() {
                    
                    let merged_k_gpu = candle_core::Tensor::cat(&gpu_k_list, 2).unwrap_or_else(|_| gpu_k_list[0].clone());
                    let merged_v_gpu = candle_core::Tensor::cat(&gpu_v_list, 2).unwrap_or_else(|_| gpu_v_list[0].clone());

                    
                    let merged_k_cpu = merged_k_gpu.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_k_gpu.clone());
                    let merged_v_cpu = merged_v_gpu.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_v_gpu.clone());

                    // 3. CPU(RAM)로 무사히 넘어온 거대 텐서를 다시 원래 블록 크기대로 썰어서 캐시에 재할당
                    let mut current_offset = 0;
                    for (i, &idx) in target_indices.iter().enumerate() {
                        let chunk_len = gpu_k_list[i].dim(2).unwrap_or(1024);
                        
                        let k_cpu = merged_k_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_k_cpu.clone()).contiguous().unwrap_or_else(|_| merged_k_cpu.clone());
                        let v_cpu = merged_v_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_v_cpu.clone()).contiguous().unwrap_or_else(|_| merged_v_cpu.clone());
                        current_offset += chunk_len;

                        let mut inner = attn.kv_blocks[idx].inner.write().unwrap();
                        let is_dirty = {
                            let reg = attn.registry.entries.read().unwrap();
                            if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() { 
                                reg[inner.index].is_dirty[l_idx] 
                            } else { true }
                        };

                        if is_dirty {
                            let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                            
                            block_groups.entry((inner.offset, inner.index)).or_default().push(LayerKVDump {
                                layer_idx: l_idx,
                                k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                                raw_k: Some(k_cpu.clone()),
                                raw_v: Some(v_cpu.clone()),
                            });
                            
                            let mut reg = attn.registry.entries.write().unwrap();
                            if inner.index < reg.len() {
                                reg[inner.index].is_dirty[l_idx] = false;
                            }
                        }

                        inner.k_cache = Some(k_cpu);
                        inner.v_cache = Some(v_cpu);
                        inner.location = crate::models::qwen::quantized_model::KVLocation::RAM;
                        
                        let mut reg = attn.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].location[l_idx] = crate::models::qwen::quantized_model::KVLocation::RAM;
                        }
                    }
                }
            }

            if layer.is_ssm_dirty() {
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if let (Some(conv), Some(rec)) = (conv_opt, rec_opt) {
                    let conv_cpu = conv.to_device(&candle_core::Device::Cpu).unwrap_or(conv);
                    let rec_cpu = rec.to_device(&candle_core::Device::Cpu).unwrap_or(rec);
                    ssm_dumps.push((l_idx, conv_cpu, rec_cpu));
                    layer.set_ssm_dirty(false);
                }
            }
        }

        let kv_dir = crate::utils::paths::get_kv_dir(None);
        let mode = false; 
        
        
        let kv_name_raw = kv_name.unwrap_or("text");
        let last_part = kv_name_raw.split('/').last().unwrap_or("text");
        let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
        let sub_path = format!("{}/inference/{}", session_id, kv_type);
        let base_dir = kv_dir.join(&sub_path);

        if !ssm_dumps.is_empty() {
            let ssm_dir = base_dir.join("ssm");
            if !ssm_dir.exists() { let _ = std::fs::create_dir_all(&ssm_dir); }

            let dump_count = ssm_dumps.len();
            crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(dump_count, std::sync::atomic::Ordering::SeqCst);

            tokio::task::spawn_blocking(move || {
                for (l_idx, conv, rec) in ssm_dumps {
                    let mut map = std::collections::HashMap::new();
                    map.insert("conv_state".to_string(), conv);
                    map.insert("recurrent_state".to_string(), rec);

                    let st_path = ssm_dir.join(format!("l{}.st", l_idx));
                    let tmp_path = st_path.with_extension("tmp");
                    
                    if candle_core::safetensors::save(&map, &tmp_path).is_ok() {
                        if let Ok(plain_data) = std::fs::read(&tmp_path) {
                            if let Ok(encrypted_data) = crate::utils::crypto::encrypt_data(&plain_data) {
                                let _ = std::fs::write(&st_path, encrypted_data);
                            }
                        }
                        let _ = std::fs::remove_file(tmp_path);
                    }
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }

        if block_groups.is_empty() { return Ok(()); }

        if let Some(tx) = BAKE_TX.get() {
            for ((off, idx), layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(1024).await;
                let block_dir = base_dir.join(format!("b{}", off));
                if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path.clone()), offset: off, layers,
                    is_relay_baking: mode, block_idx: Some(idx), registry: self.registry.clone(),
                })).await.is_err() {
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    SLOT_MANAGER.release_slot(sid).await;
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------------
// [Qwen3_5Model] 최상위 모델 구조 수정 및 인자 전달
// ----------------------------------------------------------------------------------
pub struct Qwen3_5Model {
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    visual: Option<Qwen3VLVisionModel>,
    pub language_model: Qwen3_5TextModel,
    lm_head: ProjKind,
    rope_deltas: Option<Tensor>,
    rope_deltas_cpu: Option<Vec<i64>>,
}

impl Qwen3_5Model {
    pub fn new_from_vb(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_m.pp("visual"))?;
        let language_model =
            Qwen3_5TextModel::new_from_vb(vb_m.pp("language_model"), &config.text_config)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            spatial_merge_size: config.vision_config.spatial_merge_size,
            image_token_id: config.image_token_id,
            video_token_id: config.video_token_id,
            vision_start_token_id: config.vision_start_token_id,
            visual: Some(visual),
            language_model,
            lm_head: ProjKind::LinearProj(lm_head),
            rope_deltas: None,
            rope_deltas_cpu: None,
        })
    }

    pub fn new_from_gguf<R1: Read + Seek, R2: Read + Seek>(
        gguf: &mut Gguf<R1>,
        mmproj_gguf: Option<&mut Gguf<R2>>,
        device: &Device,
        mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
        ct_handle: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>
    ) -> Result<Self> {
        let spatial_merge_size = 2usize;
        let image_token_id = 248056u32;
        let video_token_id = 248057u32;
        let vision_start_token_id = 248053u32;
        let visual = if let Some(mmproj) = mmproj_gguf {
            let visual = Qwen3VLVisionModel::new_from_gguf(mmproj)?;
            Some(visual)
        } else {
            None
        };

        let language_model = Qwen3_5TextModel::new_from_gguf(gguf, device, mmap_handle, ct_handle)?;
        
        let lm_head_tensor = match gguf.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gguf.tensor("token_embd.weight")?,
        };
        
        
        // 기존처럼 15만 개짜리 매트릭스를 F32로 압축 해제하지 않습니다!
        // Quantized(압축된) 상태 그대로 QMatMul에 물려주어 RAM 피크를 완벽히 제거합니다.
        let qmatmul = QMatMul::from_qtensor(lm_head_tensor)?;
        let lm_head = QuantizedLinear::new(qmatmul, None);
        
        Ok(Self {
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            visual,
            language_model,
            
            lm_head: ProjKind::QuantizedProj(lm_head),
            rope_deltas: None,
            rope_deltas_cpu: None,
        })
    }

    
    pub fn compute_and_set_rope_deltas(
        &mut self,
        full_input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
    ) -> Result<()> {
        let (_, deltas, deltas_cpu) = self.get_rope_index(full_input_ids, image_grid_thw, video_grid_thw, None)?;
        self.rope_deltas = Some(deltas);
        self.rope_deltas_cpu = Some(deltas_cpu);
        Ok(())
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Vec<i64>)> {
        let spatial_merge_size = self.spatial_merge_size;
        let image_token_id = self.image_token_id;
        let vision_start_token_id = self.vision_start_token_id;
        
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut mrope_position_deltas = Vec::new();

        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        let image_thw_cpu = if let Some(thw) = image_grid_thw {
            Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?)
        } else { None };

        let mut flat_pos_ids: Vec<u32> = Vec::with_capacity(3 * b_sz * seq_len);

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                // 💡 vision_start를 믿지 않고, 확실한 image_token_id만으로 3D 격자를 발동시킵니다.
                if ids[i] == image_token_id {
                    if let Some(thw_cpu_array) = &image_thw_cpu {
                        if image_idx < thw_cpu_array.len() {
                            let thw = &thw_cpu_array[image_idx];
                            image_idx += 1;
                            let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                            
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
                            continue; // 이미지 블록 처리가 끝났으므로 다음 루프로 직행
                        }
                    }
                }
                
                // 일반 텍스트 토큰 및 쪼개진 특수 토큰 처리
                for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                i += 1;
                curr_pos += 1;
            }
            
            for d in 0..3 {
                flat_pos_ids.extend_from_slice(&llm_pos_ids[d]);
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        } 

        
        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(input_ids.device())?;
            
        let target_dtype = if input_ids.device().is_cuda() { DType::BF16 } else { DType::F32 };
        let deltas = Tensor::from_vec(mrope_position_deltas.clone(), (b_sz, 1), &Device::Cpu)?
            .to_dtype(DType::F32)? 
            .to_device(input_ids.device())? 
            .to_dtype(target_dtype)?; 
        
        Ok((position_ids.contiguous()?, deltas, mrope_position_deltas))
    }

    fn compute_3d_position_ids(
        &mut self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = inputs_embeds.dims3()?;

        
        // 디코딩 중에는 절대 재계산하지 않고, 저장해둔 오프셋(delta)을 유지하여 위치 연속성을 100% 보장합니다.
        if seqlen_offset == 0 {
            let (pos_ids, rope_deltas, deltas_cpu) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
            self.rope_deltas_cpu = Some(deltas_cpu);
            return Ok(pos_ids.to_device(inputs_embeds.device())?);
        }

        // 이후 1글자씩 뱉어낼 때는 과거로 돌아가지 않고 정상적으로 +1씩 전진합니다.
        if self.rope_deltas_cpu.is_none() {
            self.rope_deltas_cpu = Some(vec![0i64; bs]);
        }

        let deltas_cpu = self.rope_deltas_cpu.as_ref().unwrap();
        let mut p_ids_vec = Vec::with_capacity(bs);
        for b in 0..bs {
            let delta = deltas_cpu[b];
            let real_start = (seqlen_offset as i64 + delta) as f32;
            
            // arange 대신 단일 스칼라 텐서를 즉시 생성
            let p_id = Tensor::new(&[real_start], &Device::Cpu)?
                .reshape((1, 1, 1))?
                .broadcast_as((3, 1, 1))?; // seq_len은 항상 1이므로 하드코딩
            p_ids_vec.push(p_id);
        }
        
        
        // GPU로 넘기기 전에 무조건 contiguous()로 물리적 메모리 연속성을 100% 보장합니다!
        let position_ids = Tensor::cat(&p_ids_vec, 1)?.contiguous()?.to_device(inputs_embeds.device())?;
        Ok(position_ids)
    }

    pub async fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>, 
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        let input_ids_cpu = input_ids.to_device(&Device::Cpu)?;
        
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        let target_dtype = if input_ids.device().is_cuda() { DType::BF16 } else { DType::F32 };
        inputs_embeds = inputs_embeds.to_dtype(target_dtype)?;
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                if let Some(visual) = self.visual.as_ref() {
                    let (image_embeds, _): (Tensor, _) = visual.forward(pixel_values, image_grid_thw)?;
                    
                    // 외부 유틸리티 대신 CPU에서 직접 마스크를 만들어 CUDA U32/U8 충돌 원천 차단
                    let img_token_t = Tensor::new(vec![self.image_token_id as f32], &Device::Cpu)?;
                    let mask_cpu = input_ids_cpu.to_dtype(DType::F32)?.broadcast_eq(&img_token_t)?.to_dtype(DType::U8)?;
                    let vision_mask = mask_cpu.to_device(input_ids.device())?;
                    
                    let image_embeds = image_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
                }
            }
        }
        if let Some(pixel_values_video) = pixel_values_video {
            if let Some(video_grid_thw) = video_grid_thw {
                if let Some(visual) = self.visual.as_ref() {
                    let (video_embeds, _): (Tensor, _) = visual.forward(pixel_values_video, video_grid_thw)?;
                    
                    let vid_token_t = Tensor::new(vec![self.video_token_id as f32], &Device::Cpu)?;
                    let mask_cpu = input_ids_cpu.to_dtype(DType::F32)?.broadcast_eq(&vid_token_t)?.to_dtype(DType::U8)?;
                    let vision_mask = mask_cpu.to_device(input_ids.device())?;
                    
                    let video_embeds = video_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
                }
            }
        }

        let position_ids;
        if cache_position.is_some() && seqlen_offset > 0 {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            
            
            let delta = cache_position.unwrap().i(0)?
                .to_dtype(candle_core::DType::F32)? 
                .to_device(&Device::Cpu)? 
                .broadcast_add(&Tensor::new(seqlen_offset as f32, &Device::Cpu)?)?;
            
            position_ids = Tensor::arange(0f32, seq_len as f32, &Device::Cpu)?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?
                .to_device(input_ids.device())?;
        } else {
            position_ids = self.compute_3d_position_ids(input_ids, &inputs_embeds, image_grid_thw, video_grid_thw, seqlen_offset)?;
        }
        
        let total_len = inputs_embeds.dim(1)?;
        
        let chunk_size = 2048; 
        let mut final_hidden_state = None;

        if total_len > 1 {
            let mut processed = 0;
            let mut current_offset = seqlen_offset;
            
            while processed < total_len {
                
                if crate::utils::is_extraction_stopped() {
                    return Err(anyhow::anyhow!("Task cancelled"));
                }

                let take = (total_len - processed).min(chunk_size);
                let chunk_embeds = inputs_embeds.narrow(1, processed, take)?;
                let chunk_pos_ids = position_ids.narrow(2, processed, take)?;

                let outputs = self.language_model.forward(&chunk_embeds, &chunk_pos_ids, current_offset, session_id.clone(), kv_name.clone()).await?;

                if processed + take == total_len {
                    let seq_len = outputs.dim(1)?;
                    
                    final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
                }

                if let Some(sid) = &session_id {
                    let _ = self.language_model.force_flush_all_active_blocks(sid, kv_name.as_deref()).await;
                }

                processed += take;
                current_offset += take;
                
                
                let pct = ((processed as f32 / total_len as f32) * 100.0) as i32;
                if let Some(tx) = crate::scheduler::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("search_") || sid.starts_with("img_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        
                        let current_cat = crate::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = format!("Reading context ({}%)...", pct);

                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Prefill)", current_cat),
                            "summary": summary_msg,
                            "spinner": "⠹"
                        }));
                    }
                }

                use std::io::Write;
                print!("\r[Qwen3.5-PREFILL] {} / {} tokens processed", processed, total_len);
                let _ = std::io::stdout().flush();
            }
            println!("\n[Qwen3.5-PREFILL] Complete. Starting Generation...");
        } else {
            let outputs = self.language_model.forward(&inputs_embeds, &position_ids, seqlen_offset, session_id.clone(), kv_name.clone()).await?;
            let seq_len = outputs.dim(1)?;
            
            final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
        }

        let hidden_state = final_hidden_state.unwrap();
        
        
        let hidden_state_aligned = hidden_state.contiguous()?.to_dtype(DType::F32)?;
        let logits_gpu = self.lm_head.forward(&hidden_state_aligned)?;
        
        
        Ok(logits_gpu.to_device(&Device::Cpu)?)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
    }
}