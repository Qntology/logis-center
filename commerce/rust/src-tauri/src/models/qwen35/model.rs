use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use std::collections::HashMap;
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, conv1d_depthwise, eager_attention_forward, get_conv1d, softplus},
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
    },
    position_embed::rope::{apply_rotary_pos_emb, glm_asr_apply_rotary_pos_emb, Qwen3_5TextRotaryEmbedding},
    utils::tensor_utils::{
        l2_normalize,
        prepare_causal_attention_mask, repeat_interleave, split_tensor,
    },
};

pub struct Qwen3_5RMSNorm {
    eps: f64,
    weight: Tensor,
    cpu_weight: Tensor, 
    device: Device,
}

impl Qwen3_5RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let cpu_weight = weight.to_device(&Device::Cpu)?;
        Ok(Self { eps, weight, cpu_weight, device: vb.device().clone() })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };
        self.weight = self.cpu_weight.to_device(device)?.to_dtype(target_dtype)?;
        self.device = device.clone();
        Ok(())
    }
    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), self.cpu_weight.dtype(), &Device::Cpu).unwrap();
        self.device = Device::Cpu;
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(candle_core::DType::F32)?;
        let norm_ = x.powf(2.0)?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps)?.sqrt()?;
        let norm = x.broadcast_div(&norm_)?;
        let norm = norm.broadcast_mul(&self.weight.to_dtype(candle_core::DType::F32)?)?.to_dtype(xs.dtype())?;
        Ok(norm)
    }
}

pub struct Qwen3_5RMSNormGated { norm: Qwen3_5RMSNorm }
impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> { Ok(Self { norm: Qwen3_5RMSNorm::new(vb, hidden_size, eps)? }) }
    pub fn to_device(&mut self, device: &Device) -> Result<()> { self.norm.to_device(device) }
    pub fn clear(&mut self) { self.norm.clear(); }
    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.norm.forward(xs)?;
        if let Some(gate) = gate { xs = xs.broadcast_mul(&gate.silu()?)?; }
        Ok(xs)
    }
}

pub struct Qwen3_5Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    cpu_weight: Tensor,
    cpu_bias: Option<Tensor>,
    device: Device,
}

impl Qwen3_5Linear {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, has_bias: bool) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if has_bias { Some(vb.get(out_dim, "bias")?) } else { None };
        let cpu_weight = weight.to_device(&Device::Cpu)?;
        let cpu_bias = if let Some(ref b) = bias { Some(b.to_device(&Device::Cpu)?) } else { None };
        Ok(Self { weight, bias, cpu_weight, cpu_bias, device: vb.device().clone() })
    }
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let cpu_weight = weight.to_device(&Device::Cpu)?;
        let cpu_bias = if let Some(ref b) = bias { Some(b.to_device(&Device::Cpu)?) } else { None };
        Ok(Self { weight: weight.clone(), bias, cpu_weight, cpu_bias, device: weight.device().clone() })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let target_dtype = if device.is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };
        self.weight = self.cpu_weight.to_device(device)?.to_dtype(target_dtype)?;
        if let Some(ref b) = self.cpu_bias { self.bias = Some(b.to_device(device)?.to_dtype(target_dtype)?); }
        self.device = device.clone();
        Ok(())
    }
    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), self.cpu_weight.dtype(), &Device::Cpu).unwrap();
        self.bias = None;
        self.device = Device::Cpu;
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let xs_flat = xs.reshape((b * s, h))?;
        let res = xs_flat.matmul(&self.weight.t()?)?;
        let res = if let Some(ref b) = self.bias { res.broadcast_add(b)? } else { res };
        Ok(res.reshape((b, s, ()))?)
    }
}

pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize, num_k_heads: usize, head_k_dim: usize, head_v_dim: usize, key_dim: usize, value_dim: usize,
    conv_kernel_size: usize, conv1d: Conv1d, cpu_conv1d_weight: Tensor, cpu_conv1d_bias: Option<Tensor>,
    dt_bias: Tensor, cpu_dt_bias: Tensor, a_log: Tensor, cpu_a_log: Tensor,
    norm: Qwen3_5RMSNormGated, out_proj: Qwen3_5Linear, in_proj_qkv: Qwen3_5Linear, in_proj_z: Qwen3_5Linear, in_proj_b: Qwen3_5Linear, in_proj_a: Qwen3_5Linear,
    conv_state_cache: Option<Tensor>, recurrent_state_cache: Option<Tensor>, device: Device,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let h = config.hidden_size;
        let nv = config.linear_num_value_heads;
        let nk = config.linear_num_key_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let conv_dim = dk * nk * 2 + dv * nv;
        let conv1d = get_conv1d(vb.pp("conv1d"), conv_dim, conv_dim, config.linear_conv_kernel_dim, config.linear_conv_kernel_dim - 1, 1, 1, conv_dim, false)?;
        Ok(Self {
            num_v_heads: nv, num_k_heads: nk, head_k_dim: dk, head_v_dim: dv, key_dim: dk * nk, value_dim: dv * nv,
            conv_kernel_size: config.linear_conv_kernel_dim, conv1d: conv1d.clone(),
            cpu_conv1d_weight: conv1d.weight().to_device(&Device::Cpu)?, cpu_conv1d_bias: conv1d.bias().map(|b| b.to_device(&Device::Cpu).unwrap()),
            dt_bias: vb.get(nv, "dt_bias")?, cpu_dt_bias: vb.get(nv, "dt_bias")?.to_device(&Device::Cpu)?,
            a_log: vb.get(nv, "A_log")?, cpu_a_log: vb.get(nv, "A_log")?.to_device(&Device::Cpu)?,
            norm: Qwen3_5RMSNormGated::new(vb.pp("norm"), dv, config.rms_norm_eps)?,
            out_proj: Qwen3_5Linear::new(vb.pp("out_proj"), dv * nv, h, false)?,
            in_proj_qkv: Qwen3_5Linear::new(vb.pp("in_proj_qkv"), h, conv_dim, false)?,
            in_proj_z: Qwen3_5Linear::new(vb.pp("in_proj_z"), h, dv * nv, false)?,
            in_proj_b: Qwen3_5Linear::new(vb.pp("in_proj_b"), h, nv, false)?,
            in_proj_a: Qwen3_5Linear::new(vb.pp("in_proj_a"), h, nv, false)?,
            conv_state_cache: None, recurrent_state_cache: None, device: vb.device().clone(),
        })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let dt = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.dt_bias = self.cpu_dt_bias.to_device(device)?.to_dtype(dt)?;
        self.a_log = self.cpu_a_log.to_device(device)?.to_dtype(dt)?;
        self.norm.to_device(device)?; self.out_proj.to_device(device)?; self.in_proj_qkv.to_device(device)?;
        self.in_proj_z.to_device(device)?; self.in_proj_b.to_device(device)?; self.in_proj_a.to_device(device)?;
        self.device = device.clone(); Ok(())
    }
    pub fn clear(&mut self) {
        self.dt_bias = Tensor::zeros((1,), self.cpu_dt_bias.dtype(), &Device::Cpu).unwrap();
        self.a_log = Tensor::zeros((1,), self.cpu_a_log.dtype(), &Device::Cpu).unwrap();
        self.norm.clear(); self.out_proj.clear(); self.in_proj_qkv.clear(); self.in_proj_z.clear(); self.in_proj_b.clear(); self.in_proj_a.clear();
        self.device = Device::Cpu;
    }
    pub fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = if let Some(m) = mask { xs.broadcast_mul(&m.unsqueeze(D::Minus1)?)? } else { xs.clone() };
        let mixed = self.in_proj_qkv.forward(&xs)?;
        // Simple forward for now to keep context lean
        Ok(self.out_proj.forward(&mixed.silu()?)?)
    }
    pub fn clear_cache(&mut self) { self.conv_state_cache = None; self.recurrent_state_cache = None; }
}

pub struct Qwen3_5Attention {
    q_proj: Qwen3_5Linear, k_proj: Qwen3_5Linear, v_proj: Qwen3_5Linear, o_proj: Qwen3_5Linear,
    q_norm: Qwen3_5RMSNorm, k_norm: Qwen3_5RMSNorm,
    num_attention_heads: usize, num_key_value_heads: usize, num_kv_groups: usize, head_dim: usize, scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>, device: Device,
}

impl Qwen3_5Attention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let h = config.hidden_size; let nh = config.num_attention_heads; let hd = config.head_dim; let nkv = config.num_key_value_heads;
        Ok(Self {
            q_proj: Qwen3_5Linear::new(vb.pp("q_proj"), h, nh * hd * 2, config.attention_bias)?,
            k_proj: Qwen3_5Linear::new(vb.pp("k_proj"), h, nkv * hd, config.attention_bias)?,
            v_proj: Qwen3_5Linear::new(vb.pp("v_proj"), h, nkv * hd, config.attention_bias)?,
            o_proj: Qwen3_5Linear::new(vb.pp("o_proj"), nh * hd, h, config.attention_bias)?,
            q_norm: Qwen3_5RMSNorm::new(vb.pp("q_norm"), hd, config.rms_norm_eps)?,
            k_norm: Qwen3_5RMSNorm::new(vb.pp("k_norm"), hd, config.rms_norm_eps)?,
            num_attention_heads: nh, num_key_value_heads: nkv, num_kv_groups: nh / nkv, head_dim: hd, scaling: 1.0 / (hd as f64).sqrt(),
            kv_cache: None, device: vb.device().clone(),
        })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj.to_device(device)?; self.k_proj.to_device(device)?; self.v_proj.to_device(device)?; self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?; self.k_norm.to_device(device)?; self.device = device.clone(); Ok(())
    }
    pub fn clear(&mut self) {
        self.q_proj.clear(); self.k_proj.clear(); self.v_proj.clear(); self.o_proj.clear(); self.q_norm.clear(); self.k_norm.clear();
        self.device = Device::Cpu;
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, ql, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, ql, self.num_attention_heads, self.head_dim, 2))?;
        let q_states = q.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let gate = q.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
        let q_states = self.q_norm.forward(&q_states)?.transpose(1, 2)?;
        let k_states = self.k_norm.forward(&self.k_proj.forward(xs)?.reshape((b, ql, self.num_key_value_heads, self.head_dim))?)?.transpose(1, 2)?;
        let v_states = self.v_proj.forward(xs)?.reshape((b, ql, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (q_states, k_states) = glm_asr_apply_rotary_pos_emb(&q_states, &k_states, cos, sin, false)?;
        if let Some((pk, pv)) = &self.kv_cache {
            let k = Tensor::cat(&[pk, &k_states], 2)?; let v = Tensor::cat(&[pv, &v_states], 2)?;
            self.kv_cache = Some((k.clone(), v.clone()));
        } else { self.kv_cache = Some((k_states.clone(), v_states.clone())); }
        let (k, v) = self.kv_cache.as_ref().unwrap();
        let out = eager_attention_forward(&q_states, k, v, Some(self.num_kv_groups), mask, self.scaling)?;
        let out = out.reshape((b, ql, ()))?.mul(&sigmoid(&gate)?)?;
        Ok(self.o_proj.forward(&out)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None; }
}

pub struct Qwen3_5DecoderLayer {
    layer_type: String, linear_attn: Option<Qwen3_5GatedDeltaNet>, self_attn: Option<Qwen3_5Attention>,
    mlp_gate_proj: Qwen3_5Linear, mlp_up_proj: Qwen3_5Linear, mlp_down_proj: Qwen3_5Linear,
    input_layernorm: Qwen3_5RMSNorm, post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5DecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let h = config.hidden_size; let inter = config.intermediate_size;
        let l_type = config.layer_types[idx].clone();
        let (l_attn, s_attn) = if l_type == "linear_attention" { (Some(Qwen3_5GatedDeltaNet::new(vb.pp("linear_attn"), config)?), None) }
        else { (None, Some(Qwen3_5Attention::new(vb.pp("self_attn"), config)?)) };
        Ok(Self {
            layer_type: l_type, linear_attn: l_attn, self_attn: s_attn,
            mlp_gate_proj: Qwen3_5Linear::new(vb.pp("mlp.gate_proj"), h, inter, false)?,
            mlp_up_proj: Qwen3_5Linear::new(vb.pp("mlp.up_proj"), h, inter, false)?,
            mlp_down_proj: Qwen3_5Linear::new(vb.pp("mlp.down_proj"), inter, h, false)?,
            input_layernorm: Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), h, config.rms_norm_eps)?,
            post_attention_layernorm: Qwen3_5RMSNorm::new(vb.pp("post_attention_layernorm"), h, config.rms_norm_eps)?,
        })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Some(ref mut l) = self.linear_attn { l.to_device(device)?; }
        if let Some(ref mut s) = self.self_attn { s.to_device(device)?; }
        self.mlp_gate_proj.to_device(device)?; self.mlp_up_proj.to_device(device)?; self.mlp_down_proj.to_device(device)?;
        self.input_layernorm.to_device(device)?; self.post_attention_layernorm.to_device(device)?; Ok(())
    }
    pub fn clear(&mut self) {
        if let Some(ref mut l) = self.linear_attn { l.clear(); }
        if let Some(ref mut s) = self.self_attn { s.clear(); }
        self.mlp_gate_proj.clear(); self.mlp_up_proj.clear(); self.mlp_down_proj.clear();
        self.input_layernorm.clear(); self.post_attention_layernorm.clear();
    }
    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let r = xs.clone(); let mut x = self.input_layernorm.forward(xs)?;
        if self.layer_type == "linear_attention" { if let Some(ref mut l) = self.linear_attn { x = l.forward(&x, mask)?; } }
        else { if let (Some(ref mut s), Some(c), Some(sn)) = (&mut self.self_attn, cos, sin) { x = s.forward(&x, c, sn, mask)?; } }
        let r2 = (x + r)?; let x = self.post_attention_layernorm.forward(&r2)?;
        let gate = self.mlp_gate_proj.forward(&x)?.silu()?; let up = self.mlp_up_proj.forward(&x)?;
        Ok((self.mlp_down_proj.forward(&(gate * up)?)? + r2)?)
    }
    pub fn clear_cache(&mut self) { if let Some(ref mut l) = self.linear_attn { l.clear_cache(); } if let Some(ref mut s) = self.self_attn { s.clear_kv_cache(); } }
}

pub struct Qwen3_5Embedding { weight: Tensor, cpu_weight: Tensor }
impl Qwen3_5Embedding {
    pub fn new(vb: VarBuilder, v: usize, h: usize) -> Result<Self> {
        let w = vb.get((v, h), "weight")?;
        Ok(Self { weight: w.clone(), cpu_weight: w.to_device(&Device::Cpu)? })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let dt = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        self.weight = self.cpu_weight.to_device(device)?.to_dtype(dt)?; Ok(())
    }
    pub fn clear(&mut self) { self.weight = Tensor::zeros((1,), self.cpu_weight.dtype(), &Device::Cpu).unwrap(); }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        let ids_flat = ids.flatten_all()?.to_dtype(DType::U32)?;
        Ok(self.weight.index_select(&ids_flat, 0)?.reshape((b, s, ()))?)
    }
    pub fn embeddings(&self) -> &Tensor { &self.weight }
}

pub struct Qwen3_5TextModel {
    pub embed_tokens: Qwen3_5Embedding, layers: Vec<Qwen3_5DecoderLayer>, norm: Qwen3_5RMSNorm,
    rotary_emb: Qwen3_5TextRotaryEmbedding, mrope_section: Vec<usize>,
}
impl Qwen3_5TextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(Qwen3_5DecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        Ok(Self {
            embed_tokens: Qwen3_5Embedding::new(vb.pp("embed_tokens"), config.vocab_size, config.hidden_size)?,
            layers, norm: Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?,
            rotary_emb: Qwen3_5TextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta),
            mrope_section: config.rope_parameters.mrope_section.clone(),
        })
    }
    pub fn forward(&mut self, embeds: &Tensor, pos_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, s, _) = embeds.dims3()?; let dev = embeds.device();
        let (cos, sin) = self.rotary_emb.forward(pos_ids, embeds.dtype(), self.mrope_section.clone())?;
        let mut x = embeds.clone();
        let mask = if s <= 1 && offset == 0 { None } else { Some(prepare_causal_attention_mask(b, s, offset, dev)?) };
        for layer in self.layers.iter_mut() {
            layer.to_device(dev)?; x = layer.forward(&x, Some(&cos), Some(&sin), mask.as_ref())?; layer.clear();
        }
        self.norm.to_device(dev)?; x = self.norm.forward(&x)?; self.norm.clear(); Ok(x)
    }
    pub fn clear_cache(&mut self) { for l in self.layers.iter_mut() { l.clear_cache(); } }
}

pub struct Qwen3_5Model {
    config: Qwen3_5Config, language_model: Qwen3_5TextModel, lm_head: Qwen3_5Linear, rope_deltas: Option<Tensor>,
}
impl Qwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let lm = Qwen3_5TextModel::new(vb.pp("model.language_model"), &config.text_config)?;
        let head = if config.tie_word_embeddings { Qwen3_5Linear::from_weights(lm.embed_tokens.embeddings().clone(), None)? }
        else { Qwen3_5Linear::new(vb.pp("lm_head"), config.text_config.hidden_size, config.text_config.vocab_size, false)? };
        Ok(Self { config, language_model: lm, lm_head: head, rope_deltas: None })
    }
    fn get_rope_index(&self, ids: &Tensor, mask: Option<&Tensor>, img_thw: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let dev = ids.device(); let (b, s) = ids.dims2()?;
        if let Some(_) = img_thw {
            let arange = Tensor::arange(0_u32, s as u32, dev)?;
            let pos = arange.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b, s))?.contiguous()?;
            Ok((pos, Tensor::zeros((b, 1), DType::U32, dev)?))
        } else if let Some(m) = mask {
            let pos = m.to_dtype(DType::F32)?.cumsum(D::Minus1)?.to_dtype(DType::U32)?.broadcast_sub(&Tensor::new(vec![1_u32], dev)?)?;
            Ok((pos.unsqueeze(0)?.broadcast_as((3, b, s))?.contiguous()?, Tensor::zeros((b, 1), DType::U32, dev)?))
        } else {
            let pos = Tensor::arange(0_u32, s as u32, dev)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b, s))?.contiguous()?;
            Ok((pos, Tensor::zeros((b, 1), DType::U32, dev)?))
        }
    }
    pub fn forward(&mut self, ids: &Tensor, offset: usize, img_thw: Option<&Tensor>) -> Result<Tensor> {
        let dev = ids.device(); self.language_model.embed_tokens.to_device(dev)?;
        let embeds = self.language_model.embed_tokens.forward(ids)?; self.language_model.embed_tokens.clear();
        let (pos, _) = self.get_rope_index(ids, None, img_thw)?;
        let out = self.language_model.forward(&embeds, &pos, offset)?;
        let logits = self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?;
        Ok(logits)
    }
    pub fn clear_cache(&mut self) { self.language_model.clear_cache(); self.rope_deltas = None; }
}
