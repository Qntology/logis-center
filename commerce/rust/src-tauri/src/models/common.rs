use anyhow::Result;
use candle_core::{D, DType, Device, Tensor, Module};
use candle_nn::{
    Activation, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig,
    Linear, VarBuilder, batch_norm, conv2d, conv2d_no_bias,
};

use crate::{position_embed::rope::apply_rotary_pos_emb, utils::tensor_utils::repeat_kv};

pub fn find_flexible_weight(v: &VarBuilder, name: &str, out_dim: usize, in_dim: usize) -> Result<Tensor> {
    for suffix in &["", ".packed", ".min"] {
        let full_name = format!("{}{}", name, suffix);
        match v.get_with_hints((1,), &full_name, candle_nn::Init::Const(0.)) {
            Ok(t) => return Ok(v.get(t.shape(), &full_name)?),
            Err(candle_core::Error::ShapeMismatch { shape, .. }) => {
                return Ok(v.get(shape, &full_name)?);
            }
            Err(candle_core::Error::Msg(s)) if s.contains("shape mismatch") => {
                return Ok(v.get((out_dim, in_dim), &full_name).or_else(|_| v.get_with_hints((1,), &full_name, candle_nn::Init::Const(0.)))?);
            }
            _ => {}
        }
    }
    Ok(v.get((out_dim, in_dim), name)?)
}

// [SAFE-OPS] Avoid CUDA_ERROR_NOT_FOUND by using basic ops instead of optimized kernels
pub fn safe_softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let max = xs_f32.max_keepdim(D::Minus1)?;
    let diff = xs_f32.broadcast_sub(&max)?;
    let exp = diff.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    Ok(exp.broadcast_div(&sum)?.to_dtype(xs.dtype())?)
}

pub fn safe_sigmoid(xs: &Tensor) -> Result<Tensor> {
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let exp_neg_x = xs_f32.neg()?.exp()?;
    let den = exp_neg_x.affine(1.0, 1.0)?;
    Ok(den.recip()?.to_dtype(xs.dtype())?)
}

pub fn safe_silu(xs: &Tensor) -> Result<Tensor> {
    let sig = safe_sigmoid(xs)?;
    Ok(xs.broadcast_mul(&sig)?)
}

pub fn safe_gelu(xs: &Tensor) -> Result<Tensor> {
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let c1 = (2.0f32 / std::f32::consts::PI).sqrt();
    let x_cube = xs_f32.sqr()?.broadcast_mul(&xs_f32)?;
    let inner = xs_f32.broadcast_add(&x_cube.affine(0.044715, 0.0)?)?.affine(c1 as f64, 0.0)?;
    let tanh = inner.tanh()?;
    Ok(xs_f32.broadcast_mul(&tanh.affine(1.0, 1.0)?)?.affine(0.5, 0.0)?.to_dtype(xs.dtype())?)
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    pub fn weight(&self) -> &Tensor { &self.weight }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        x_norm.broadcast_mul(&weight_f32)?.to_dtype(x.dtype())
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn bias(&self) -> Option<&Tensor> { Some(&self.bias) }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(D::Minus1)?;
        let x_mu = x_f32.broadcast_sub(&mean)?;
        let variance = x_mu.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x_mu.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let bias_f32 = self.bias.to_dtype(DType::F32)?;
        x_norm.broadcast_mul(&weight_f32)?.broadcast_add(&bias_f32)?.to_dtype(x.dtype())
    }
}

pub fn get_layer_norm(vb: VarBuilder, eps: f64, size: usize) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(vb: VarBuilder, hidden_size: usize, intermediate_size: usize, act_fn: Activation, bias: bool) -> Result<Self> {
        let gate_w = find_flexible_weight(&vb.pp("gate_proj"), "weight", intermediate_size, hidden_size)?;
        let up_w = find_flexible_weight(&vb.pp("up_proj"), "weight", intermediate_size, hidden_size)?;
        let down_w = find_flexible_weight(&vb.pp("down_proj"), "weight", hidden_size, intermediate_size)?;
        let (gate_b, up_b, down_b) = if bias {
            (Some(vb.pp("gate_proj").get(intermediate_size, "bias")?), Some(vb.pp("up_proj").get(intermediate_size, "bias")?), Some(vb.pp("down_proj").get(hidden_size, "bias")?))
        } else { (None, None, None) };
        Ok(Self { gate_proj: Linear::new(gate_w, gate_b), up_proj: Linear::new(up_w, up_b), down_proj: Linear::new(down_w, down_b), act_fn })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let g_w = self.gate_proj.weight().to_device(device)?;
        let g_b = self.gate_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.gate_proj = Linear::new(g_w, g_b);
        let u_w = self.up_proj.weight().to_device(device)?;
        let u_b = self.up_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.up_proj = Linear::new(u_w, u_b);
        let d_w = self.down_proj.weight().to_device(device)?;
        let d_b = self.down_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.down_proj = Linear::new(d_w, d_b); Ok(())
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        let lhs = match self.act_fn {
            Activation::Silu => safe_silu(&gate).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
            Activation::Gelu => safe_gelu(&gate).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
            _ => gate.apply(&self.act_fn)?,
        };
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct TwoLinearMLP {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl TwoLinearMLP {
    pub fn new(vb: VarBuilder, embedding_dim: usize, mlp_dim: usize, act: Activation, bias: bool, linear1_pp_name: &str, linear2_pp_name: &str) -> Result<Self> {
        let l1_w = find_flexible_weight(&vb.pp(linear1_pp_name), "weight", mlp_dim, embedding_dim)?;
        let l2_w = find_flexible_weight(&vb.pp(linear2_pp_name), "weight", embedding_dim, mlp_dim)?;
        let (l1_b, l2_b) = if bias { (Some(vb.pp(linear1_pp_name).get(mlp_dim, "bias")?), Some(vb.pp(linear2_pp_name).get(embedding_dim, "bias")?)) } else { (None, None) };
        Ok(Self { linear1: Linear::new(l1_w, l1_b), linear2: Linear::new(l2_w, l2_b), act })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let l1_w = self.linear1.weight().to_device(device)?;
        let l1_b = self.linear1.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear1 = Linear::new(l1_w, l1_b);
        let l2_w = self.linear2.weight().to_device(device)?;
        let l2_b = self.linear2.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear2 = Linear::new(l2_w, l2_b); Ok(())
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let l1 = xs.apply(&self.linear1)?;
        let act = match self.act {
            Activation::Silu => safe_silu(&l1)?,
            Activation::Gelu => safe_gelu(&l1)?,
            _ => l1.apply(&self.act)?,
        };
        Ok(act.apply(&self.linear2)?)
    }
}

#[derive(Debug, Clone)]
pub struct NaiveAttention {
    q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear,
    num_heads: usize, num_kv_heads: usize, num_kv_groups: usize,
    head_dim: usize, middle_size: usize, kv_cache: Option<(Tensor, Tensor)>,
}

impl NaiveAttention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_attention_heads: usize, num_key_value_heads: usize, head_dim: Option<usize>, bias: bool, o_proj_pp_name: Option<&str>) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = match head_dim { None => hidden_size / num_attention_heads, Some(dim) => dim };
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let q_w = find_flexible_weight(&vb.pp("q_proj"), "weight", num_attention_heads * head_dim, hidden_size)?;
        let k_w = find_flexible_weight(&vb.pp("k_proj"), "weight", num_key_value_heads * head_dim, hidden_size)?;
        let v_w = find_flexible_weight(&vb.pp("v_proj"), "weight", num_key_value_heads * head_dim, hidden_size)?;
        let o_w = find_flexible_weight(&vb.pp(o_proj_pp_name), "weight", hidden_size, num_attention_heads * head_dim)?;
        let (q_b, k_b, v_b, o_b) = if bias { (Some(vb.pp("q_proj").get(num_attention_heads * head_dim, "bias")?), Some(vb.pp("k_proj").get(num_key_value_heads * head_dim, "bias")?), Some(vb.pp("v_proj").get(num_key_value_heads * head_dim, "bias")?), Some(vb.pp(o_proj_pp_name).get(hidden_size, "bias")?)) } else { (None, None, None, None) };
        Ok(Self { q_proj: Linear::new(q_w, q_b), k_proj: Linear::new(k_w, k_b), v_proj: Linear::new(v_w, v_b), o_proj: Linear::new(o_w, o_b), num_heads: num_attention_heads, num_kv_heads: num_key_value_heads, num_kv_groups, head_dim, middle_size: num_attention_heads * head_dim, kv_cache: None })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let q_w = self.q_proj.weight().to_device(device)?; let q_b = self.q_proj.bias().map(|b| b.to_device(device)).transpose()?; self.q_proj = Linear::new(q_w, q_b);
        let k_w = self.k_proj.weight().to_device(device)?; let k_b = self.k_proj.bias().map(|b| b.to_device(device)).transpose()?; self.k_proj = Linear::new(k_w, k_b);
        let v_w = self.v_proj.weight().to_device(device)?; let v_b = self.v_proj.bias().map(|b| b.to_device(device)).transpose()?; self.v_proj = Linear::new(v_w, v_b);
        let o_w = self.o_proj.weight().to_device(device)?; let o_b = self.o_proj.bias().map(|b| b.to_device(device)).transpose()?; self.o_proj = Linear::new(o_w, o_b);
        if let Some((k, v)) = &self.kv_cache { self.kv_cache = Some((k.to_device(device)?, v.to_device(device)?)); }
        Ok(())
    }
    pub fn forward(&self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, tof32: bool) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?; let key_states = self.k_proj.forward(xs)?; let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states.reshape((b_sz, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let value_states = value_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let (query_states, key_states) = if let (Some(c), Some(s)) = (cos, sin) { apply_rotary_pos_emb(&query_states, &key_states, c, s, tof32)? } else { (query_states, key_states) };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), attention_mask, scale)?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?; Ok(attn_output.apply(&self.o_proj)?)
    }
    pub fn forward_with_cache(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, tof32: bool) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?; let key_states = self.k_proj.forward(xs)?; let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states.reshape((b_sz, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let value_states = value_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;
        let (key_states, value_states) = match &self.kv_cache { None => (key_states, value_states), Some((pk, pv)) => (Tensor::cat(&[pk, &key_states], 2)?, Tensor::cat(&[pv, &value_states], 2)?) };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), attention_mask, scale)?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?; Ok(attn_output.apply(&self.o_proj)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None }
}

pub struct NaiveAttnTwoLinearMLPBlock {
    self_attn: NaiveAttention, mlp: TwoLinearMLP, pub input_layernorm: LayerNorm, pub post_attention_layernorm: LayerNorm,
}

impl NaiveAttnTwoLinearMLPBlock {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_attention_heads: usize, num_key_value_heads: Option<usize>, head_dim: Option<usize>, attn_bias: bool, attn_pp_name: &str, o_proj_pp_name: Option<&str>, intermediate_size: usize, hidden_act: Activation, mlp_bias: bool, mlp_pp_name: &str, linear1_pp_name: &str, linear2_pp_name: &str, norm_eps: f64, input_norm_pp_name: &str, post_norm_pp_name: &str) -> Result<Self> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let self_attn = NaiveAttention::new(vb.pp(attn_pp_name), hidden_size, num_attention_heads, num_key_value_heads, head_dim, attn_bias, o_proj_pp_name)?;
        let mlp = TwoLinearMLP::new(vb.pp(mlp_pp_name), hidden_size, intermediate_size, hidden_act, mlp_bias, linear1_pp_name, linear2_pp_name)?;
        let input_layernorm = get_layer_norm(vb.pp(input_norm_pp_name), norm_eps, hidden_size)?;
        let post_attention_layernorm = get_layer_norm(vb.pp(post_norm_pp_name), norm_eps, hidden_size)?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }
    pub fn forward(&self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, tof32: bool) -> Result<Tensor> {
        let residual = xs.clone(); let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask, tof32)?;
        let residual = residual.add(&xs)?; let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?; Ok(residual.add(&xs)?)
    }
}

pub struct NaiveAttnGateUpDownMLPBlock {
    self_attn: NaiveAttention, mlp: GateUpDownMLP, pub input_layernorm: RmsNorm, pub post_attention_layernorm: RmsNorm,
}

impl NaiveAttnGateUpDownMLPBlock {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_attention_heads: usize, num_key_value_heads: Option<usize>, head_dim: Option<usize>, attn_bias: bool, attn_pp_name: &str, o_proj_pp_name: Option<&str>, intermediate_size: usize, hidden_act: Activation, mlp_bias: bool, mlp_pp_name: &str, norm_eps: f64, input_norm_pp_name: &str, post_norm_pp_name: &str) -> Result<Self> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let self_attn = NaiveAttention::new(vb.pp(attn_pp_name), hidden_size, num_attention_heads, num_key_value_heads, head_dim, attn_bias, o_proj_pp_name)?;
        let mlp = GateUpDownMLP::new(vb.pp(mlp_pp_name), hidden_size, intermediate_size, hidden_act, mlp_bias)?;
        let input_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(input_norm_pp_name))?;
        let post_attention_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(post_norm_pp_name))?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone(); let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_with_cache(&xs, cos, sin, attention_mask, false)?;
        let residual = residual.add(&xs)?; let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?; Ok(residual.add(&xs)?)
    }
    pub fn clear_kv_cache(&mut self) { self.self_attn.clear_kv_cache() }
}

pub fn eager_attention_forward(query_states: &Tensor, key_states: &Tensor, value_states: &Tensor, num_key_value_groups: Option<usize>, attention_mask: Option<&Tensor>, scaling: f64) -> Result<Tensor> {
    // println!("[TRACE] Attention forward: Q={:?}, K={:?}, V={:?}", query_states.dims(), key_states.dims(), value_states.dims());
    let ks = match num_key_value_groups { Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?, None => key_states.clone() };
    let vs = match num_key_value_groups { Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?, None => value_states.clone() };
    let qs = query_states.contiguous()?; let ks = ks.contiguous()?; let vs = vs.contiguous()?;
    let attn_output = {
        let attn_weights = qs.matmul(&ks.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights * scaling)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => {
                let (b_sz, _, q_len, kv_len) = attn_weights.dims4()?; let m_len = mask.dim(D::Minus1)?;
                let aligned_mask = if m_len < kv_len { let padding = Tensor::zeros((b_sz, 1, q_len, kv_len - m_len), mask.dtype(), mask.device())?; Tensor::cat(&[padding, mask.clone()], D::Minus1)? }
                else if m_len > kv_len { mask.narrow(D::Minus1, 0, kv_len)? } else { mask.clone() };
                let mask_f32 = aligned_mask.to_dtype(DType::F32)?; let weights_f32 = attn_weights.to_dtype(DType::F32)?;
                weights_f32.broadcast_add(&mask_f32)?.to_dtype(attn_weights.dtype())?
            }
        };
        let attn_weights = safe_softmax_last_dim(&attn_weights).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        attn_weights.matmul(&vs)?
    };
    Ok(attn_output.transpose(1, 2)?.contiguous()?)
}

pub fn get_conv2d(vb: VarBuilder, in_c: usize, out_c: usize, kernel_size: usize, padding: usize, stride: usize, dilation: usize, groups: usize, bias: bool) -> Result<Conv2d> {
    let cfg = Conv2dConfig { padding, stride, dilation, groups, cudnn_fwd_algo: None };
    Ok(if bias { conv2d(in_c, out_c, kernel_size, cfg, vb)? } else { conv2d_no_bias(in_c, out_c, kernel_size, cfg, vb)? })
}

pub fn get_batch_norm(vb: VarBuilder, eps: f64, dim: usize) -> Result<BatchNorm> {
    let bn_config = BatchNormConfig { eps, remove_mean: true, affine: true, momentum: 0.1 };
    Ok(batch_norm(dim, bn_config, vb)?)
}

pub fn deform_conv2d_kernel(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, offset: &Tensor, mask: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
    let (_, in_c, in_h, in_w) = input.dims4()?; let (out_channel, _, ker_h, ker_w) = weight.dims4()?;
    let out_h = ((in_h + 2 * padding - ker_h) / stride) + 1; let out_w = ((in_w + 2 * padding - ker_w) / stride) + 1;
    let num_kernels = in_c * out_h * out_w;
    let mask_vec = if let Some(mask) = mask { Some(mask.squeeze(0)?.to_vec3::<f32>()?) } else { None };
    let offset_vec = offset.squeeze(0)?.to_vec3::<f32>()?; let input_vec = input.squeeze(0)?.to_vec3::<f32>()?;
    let mut columns_vec = vec![vec![0.0f32; out_h * out_w]; in_c * ker_h * ker_w];
    for index in 0..num_kernels {
        let out_x = index % out_w; let out_y = (index / out_w) % out_h; let in_c = index / (out_w * out_h); let out_c = in_c * ker_h * ker_w;
        for i in 0..ker_h { for j in 0..ker_w {
            let mask_idx = i * ker_w + j; let offset_idx = 2 * mask_idx;
            let mask_value = if mask.is_some() { mask_vec.as_ref().unwrap()[mask_idx][out_y][out_x] } else { 1.0 };
            let offset_h = offset_vec[offset_idx][out_y][out_x]; let offset_w = offset_vec[offset_idx + 1][out_y][out_x];
            let y = ((out_y * stride - padding) + i) as f32 + offset_h; let x = ((out_x * stride - padding) + j) as f32 + offset_w;
            let val = if y <= -1.0 || in_h as f32 <= y || x <= -1.0 || in_w as f32 <= x { 0.0 } else {
                let h_low = y.floor(); let w_low = x.floor(); let h_high = h_low + 1.0; let w_high = w_low + 1.0;
                let lh = y - h_low; let lw = x - w_low; let hh = 1.0 - lh; let hw = 1.0 - lw;
                let w1 = hh * hw; let w2 = hh * lw; let w3 = lh * hw; let w4 = lh * lw;
                let v1 = if h_low >= 0.0 && w_low >= 0.0 { input_vec[in_c][h_low as usize][w_low as usize] } else { 0.0 };
                let v2 = if h_low >= 0.0 && w_high <= (in_w - 1) as f32 { input_vec[in_c][h_low as usize][w_high as usize] } else { 0.0 };
                let v3 = if h_high <= (in_h - 1) as f32 && w_low >= 0.0 { input_vec[in_c][h_high as usize][w_low as usize] } else { 0.0 };
                let v4 = if h_high <= (in_h - 1) as f32 && w_high <= (in_w - 1) as f32 { input_vec[in_c][h_high as usize][w_high as usize] } else { 0.0 };
                w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
            };
            columns_vec[out_c + i * ker_w + j][out_y * out_w + out_x] = mask_value * val;
        }}
    }
    let columns = Tensor::new(columns_vec, weight.device())?;
    let mut out = weight.flatten_from(1)?.matmul(&columns)?.reshape((1, out_channel, out_h, out_w))?;
    if let Some(bias) = bias { out = out.broadcast_add(bias)?; }
    Ok(out)
}