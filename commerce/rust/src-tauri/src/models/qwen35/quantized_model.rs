use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;
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
        prepare_causal_attention_mask, split_tensor,
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
    
    pub fn get_q_tensor(&self, name: &str, suffix: &str) -> Result<Tensor> {
        self.get_tensor(&format!("{}.weight.{}", name, suffix))
            .or_else(|_| self.get_tensor(&format!("{}.{}", name, suffix)))
    }
}

pub struct QRmsNorm { weight_name: String, eps: f64 }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64) -> Result<Self> { Ok(Self { weight_name: name.to_string(), eps }) }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let w = reg.get_tensor(&self.weight_name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.weight_name)))?.to_device(dev)?.to_dtype(DType::F16)?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x_f32.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        Ok(norm.to_dtype(DType::F16)?.broadcast_mul(&w)?)
    }
}

pub struct QLinear { weight_name: String }
impl QLinear {
    pub fn new(name: &str) -> Self { Self { weight_name: name.to_string() } }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let name = &self.weight_name;
        let w_raw = if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "scales"), reg.get_q_tensor(name, "data"), reg.get_q_tensor(name, "shape")) {
            let s = s_t.to_device(dev)?.to_dtype(DType::F32)?;
            let d = d_t.to_device(dev)?;
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let d_f32 = d.to_dtype(DType::F32)?;
            let restored = if d.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d_f32.clone() / 16.0)?.floor()?; let low = d_f32.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                d_f32.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else {
            reg.get_tensor(name).or_else(|_| reg.get_tensor(&format!("{}.weight", name)))?.to_device(dev)?
        };

        // Use broadcast_matmul for more flexible dimension matching
        let x_in_dim = x.dim(D::Minus1)?;
        let w = if w_raw.dim(1)? == x_in_dim { w_raw.t()? } else { w_raw };
        let w = w.to_dtype(DType::F16)?.contiguous()?;
        
        let res = x.broadcast_matmul(&w)?;

        if let Ok(b) = reg.get_q_tensor(name, "bias").or_else(|_| reg.get_tensor(&format!("{}.bias", name))) {
            return Ok(res.broadcast_add(&b.to_device(dev)?.to_dtype(DType::F16)?)?);
        }
        Ok(res)
    }
}

pub struct QGatedDeltaNet {
    in_proj: QLinear, out_proj: QLinear, norm: QRmsNorm, h: usize,
}
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self {
            in_proj: QLinear::new(&format!("{}.in_proj_qkv", p)),
            out_proj: QLinear::new(&format!("{}.out_proj", p)),
            norm: QRmsNorm::new(&format!("{}.norm", p), config.rms_norm_eps)?,
            h: config.hidden_size,
        })
    }
    pub fn forward(&mut self, x: &Tensor, _mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let projected = self.in_proj.forward(x, reg)?;
        let chunks = projected.chunk(6, D::Minus1)?;
        let (q, _k, _v, z, _b, _a) = (&chunks[0], &chunks[1], &chunks[2], &chunks[3], &chunks[4], &chunks[5]);
        let gate = z.silu()?;
        let res = self.out_proj.forward(&(q.silu()? * gate)?, reg)?;
        Ok(self.norm.forward(&res, reg)?)
    }
}

pub struct QAttention {
    qkv_proj: QLinear, o_proj: QLinear, q_norm: QRmsNorm, k_norm: QRmsNorm,
    nh: usize, nkv: usize, hd: usize, scaling: f64, h: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}
impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self {
            qkv_proj: QLinear::new(&format!("{}.q_proj", p)),
            o_proj: QLinear::new(&format!("{}.o_proj", p)),
            q_norm: QRmsNorm::new(&format!("{}.q_norm", p), config.rms_norm_eps)?,
            k_norm: QRmsNorm::new(&format!("{}.k_norm", p), config.rms_norm_eps)?,
            nh: config.num_attention_heads, nkv: config.num_key_value_heads, hd: config.head_dim,
            scaling: 1.0 / (config.head_dim as f64).sqrt(), h: config.hidden_size, kv_cache: None,
        })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let qkv_raw = self.qkv_proj.forward(x, reg)?;
        let chunks = qkv_raw.chunk(4, D::Minus1)?;
        let (q_raw, k_raw, v_raw, g_raw) = (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

        let q = self.q_norm.forward(&q_raw.reshape((bs, sl, self.nh, self.hd))?, reg)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k_raw.reshape((bs, sl, self.nkv, self.hd))?, reg)?.transpose(1, 2)?;
        let v = v_raw.reshape((bs, sl, self.nkv, self.hd))?.transpose(1, 2)?;

        let (q, k) = glm_asr_apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
        let (k, v) = match &self.kv_cache { None => (k, v), Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?) };
        self.kv_cache = Some((k.clone(), v.clone()));
        
        let attn = crate::models::common::eager_attention_forward(&q, &k, &v, Some(self.nh / self.nkv), mask, self.scaling)?;
        let out = attn.reshape((bs, sl, ()))?.contiguous()?;
        self.o_proj.forward(&(out * g_raw.silu()?)?, reg)
    }
}

pub struct QDecoderLayer {
    layer_type: String, self_attn: Option<QAttention>, linear_attn: Option<QGatedDeltaNet>,
    mlp_gate: QLinear, mlp_up: QLinear, mlp_down: QLinear, in_norm: QRmsNorm, post_norm: QRmsNorm,
}
impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let lt = config.layer_types[idx].clone(); let p = vb.prefix();
        let (sa, la) = if lt == "linear_attention" { (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?)) }
        else { (Some(QAttention::new(vb.pp("self_attn"), config)?), None) };
        Ok(Self {
            layer_type: lt, self_attn: sa, linear_attn: la,
            mlp_gate: QLinear::new(&format!("{}.mlp.gate_proj", p)), mlp_up: QLinear::new(&format!("{}.mlp.up_proj", p)), mlp_down: QLinear::new(&format!("{}.mlp.down_proj", p)),
            in_norm: QRmsNorm::new(&format!("{}.input_layernorm", p), config.rms_norm_eps)?,
            post_norm: QRmsNorm::new(&format!("{}.post_attention_layernorm", p), config.rms_norm_eps)?,
        })
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone(); let mut h = self.in_norm.forward(x, reg)?;
        if let Some(ref mut sa) = self.self_attn { h = sa.forward(&h, cos, sin, mask, reg)?; }
        else if let Some(ref mut la) = self.linear_attn { h = la.forward(&h, mask, reg)?; }
        let h2 = (h + residual)?; let h = self.post_norm.forward(&h2, reg)?;
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?; let up = self.mlp_up.forward(&h, reg)?;
        Ok((self.mlp_down.forward(&(gate * up)?, reg)? + h2)?)
    }
}

pub struct QEmbedding { name: String }
impl QEmbedding {
    pub fn new(name: &str) -> Self { Self { name: name.to_string() } }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let ids_cpu = ids.flatten_all()?.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let w = if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(&self.name, "scales"), reg.get_q_tensor(&self.name, "data"), reg.get_q_tensor(&self.name, "shape")) {
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let d_f32 = d_t.to_dtype(DType::F32)?; let s_f32 = s_t.to_dtype(DType::F32)?;
            let restored = if d_t.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d_f32.clone() / 16.0)?.floor()?; let low = d_f32.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s_f32.unsqueeze(D::Minus1)?.broadcast_as((s_f32.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s_f32.unsqueeze(D::Minus1)?.broadcast_as((s_f32.dim(0)?, 32))?.flatten_all()?;
                d_f32.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else { reg.get_tensor(&self.name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.name)))? };
        Ok(w.index_select(&ids_cpu, 0)?.reshape((b, s, ()))?.to_device(dev)?.to_dtype(DType::F16)?)
    }
}

pub struct QTextModel {
    embed: QEmbedding, layers: Vec<QDecoderLayer>, norm: QRmsNorm, rotary: Qwen3_5TextRotaryEmbedding, mrope: Vec<usize>,
}
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let p = vb.prefix();
        Ok(Self {
            embed: QEmbedding::new(&format!("{}.embed_tokens", p)), layers, norm: QRmsNorm::new(&format!("{}.norm", p), config.rms_norm_eps)?,
            rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, config.rope_parameters.rope_theta),
            mrope: config.rope_parameters.mrope_section.clone(),
        })
    }
    pub fn forward(&mut self, ids: &Tensor, pos: &Tensor, offset: usize, reg: &QuantizedRegistry) -> Result<Tensor> {
        let bs = ids.dim(0)?; let sl = ids.dim(1)?;
        let mut x = self.embed.forward(ids, reg)?;
        let (cos, sin) = self.rotary.forward(pos, DType::F16, self.mrope.clone())?;
        let mask = if sl <= 1 && offset == 0 { None } else { Some(prepare_causal_attention_mask(bs, sl, offset, ids.device())?.to_dtype(DType::F16)?) };
        for l in self.layers.iter_mut() { x = l.forward(&x, &cos, &sin, mask.as_ref(), reg)?; }
        self.norm.forward(&x, reg)
    }
}

pub struct QuantizedQwen3_5Model { model: QTextModel, head: QLinear }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        Ok(Self { model: QTextModel::new(vb.pp("model.language_model"), &config.text_config)?, head: QLinear::new("model.language_model.embed_tokens") })
    }
    pub fn forward(&mut self, ids: &Tensor, offset: usize, reg: &QuantizedRegistry, _img: Option<&Tensor>) -> Result<Tensor> {
        let dev = ids.device(); let sl = ids.dim(1)?;
        let pos_vec: Vec<u32> = (offset as u32..(offset + sl) as u32).collect();
        let pos = Tensor::from_vec(pos_vec, (1, sl), dev)?.to_dtype(DType::U32)?.unsqueeze(0)?.broadcast_as((3, ids.dim(0)?, sl))?.contiguous()?;
        let h = self.model.forward(ids, &pos, offset, reg)?;
        let logits = self.head.forward(&h.narrow(1, sl - 1, 1)?, reg)?;
        Ok(logits)
    }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } } }
}
