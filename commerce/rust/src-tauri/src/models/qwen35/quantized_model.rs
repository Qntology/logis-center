use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use candle_core::safetensors::MmapedSafetensors;
use half::f16;

use crate::{
    models::{
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        common::{get_conv1d},
    },
    position_embed::rope::{
        Qwen3_5TextRotaryEmbedding, glm_asr_apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        prepare_causal_attention_mask,
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

pub struct QRmsNorm { weight_name: String, eps: f64, persistent_weight: Option<Tensor> }
impl QRmsNorm {
    pub fn new(name: &str, eps: f64) -> Result<Self> { Ok(Self { weight_name: name.to_string(), eps, persistent_weight: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let w = reg.get_tensor(&self.weight_name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.weight_name)))?.to_device(dev)?.to_dtype(DType::F16)?;
        self.persistent_weight = Some(w);
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device();
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() } 
                else { reg.get_tensor(&self.weight_name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.weight_name)))?.to_device(dev)?.to_dtype(DType::F16)? };
        let x_f32 = x.to_dtype(DType::F32)?;
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x_f32.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let norm_f16 = norm.to_dtype(DType::F16)?;
        let w_size = w.dim(0)?;
        if w_size == norm_f16.dim(D::Minus1)? { Ok(norm_f16.broadcast_mul(&w)?) }
        else {
            let mul = norm_f16.reshape(((), w_size))?.broadcast_mul(&w)?;
            Ok(mul.reshape(norm_f16.dims())?)
        }
    }
}

pub struct QLinear { weight_name: String, persistent_weight: Option<Tensor>, persistent_bias: Option<Tensor> }
impl QLinear {
    pub fn new(name: &str) -> Result<Self> { Ok(Self { weight_name: name.to_string(), persistent_weight: None, persistent_bias: None }) }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; self.persistent_bias = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let name = &self.weight_name;
        let w_raw = if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "scales"), reg.get_q_tensor(name, "data"), reg.get_q_tensor(name, "shape")) {
            let s = s_t.to_device(dev)?.to_dtype(DType::F16)?; let d = d_t.to_device(dev)?.to_dtype(DType::F16)?;
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let restored = if d.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d.clone() / 16.0)?.floor()?; let low = d.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                d.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else { reg.get_tensor(name).or_else(|_| reg.get_tensor(&format!("{}.weight", name)))?.to_device(dev)?.to_dtype(DType::F16)? };
        self.persistent_weight = Some(w_raw.to_dtype(DType::F16)?);
        if let Ok(b) = reg.get_q_tensor(name, "bias").or_else(|_| reg.get_tensor(&format!("{}.bias", name))) {
            self.persistent_bias = Some(b.to_device(dev)?.to_dtype(DType::F16)?);
        }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let dev = x.device(); let name = &self.weight_name;
        let w_raw = if let Some(pw) = &self.persistent_weight { pw.clone() } 
        else if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(name, "scales"), reg.get_q_tensor(name, "data"), reg.get_q_tensor(name, "shape")) {
            let s = s_t.to_device(dev)?.to_dtype(DType::F16)?; let d = d_t.to_device(dev)?.to_dtype(DType::F16)?;
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let restored = if d.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d.clone() / 16.0)?.floor()?; let low = d.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s.unsqueeze(D::Minus1)?.broadcast_as((s.dim(0)?, 32))?.flatten_all()?;
                d.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else { reg.get_tensor(name).or_else(|_| reg.get_tensor(&format!("{}.weight", name)))?.to_device(dev)?.to_dtype(DType::F16)? };
        let x_in_dim = x.dim(D::Minus1)?;
        let w = if w_raw.dim(1)? == x_in_dim { w_raw.t()? } else { w_raw };
        let res = x.contiguous()?.broadcast_matmul(&w.to_dtype(DType::F16)?.contiguous()?)?;
        if let Some(pb) = &self.persistent_bias {
            Ok(res.broadcast_add(pb)?)
        } else if let Ok(b) = reg.get_q_tensor(name, "bias").or_else(|_| reg.get_tensor(&format!("{}.bias", name))) {
            Ok(res.broadcast_add(&b.to_device(dev)?.to_dtype(DType::F16)?)?)
        } else { Ok(res) }
    }
}

pub struct QGatedDeltaNet { in_proj: QLinear, out_proj: QLinear, norm: QRmsNorm, pub h: usize, pub delta_state: Option<Tensor> }
impl QGatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self { in_proj: QLinear::new(&format!("{}.in_proj_qkv", p))?, out_proj: QLinear::new(&format!("{}.out_proj", p))?, norm: QRmsNorm::new(&format!("{}.norm", p), config.rms_norm_eps)?, h: config.hidden_size, delta_state: None })
    }
    pub fn clear_vram(&mut self) { self.in_proj.clear_vram(); self.out_proj.clear_vram(); self.norm.clear_vram(); }
    pub fn forward(&mut self, x: &Tensor, _mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let h = self.h;
        let projected = self.in_proj.forward(x, reg)?;
        let q = projected.narrow(D::Minus1, 0, h)?;
        let k = projected.narrow(D::Minus1, h, h)?;
        let v = projected.narrow(D::Minus1, h*2, h)?;
        let z = projected.narrow(D::Minus1, h*3, h)?;
        
        let mut current_state = match self.delta_state.take() {
            Some(st) => st.to_device(x.device())?,
            None => Tensor::zeros((bs, h, h), DType::F16, x.device())?,
        };

        let mut outputs = vec![];
        for t in 0..sl {
            let qt = q.narrow(1, t, 1)?.squeeze(1)?; 
            let kt = k.narrow(1, t, 1)?.squeeze(1)?; 
            let vt = v.narrow(1, t, 1)?.squeeze(1)?; 
            let update = vt.unsqueeze(D::Minus1)?.matmul(&kt.unsqueeze(D::Minus2)?)?;
            current_state = (current_state + update)?;
            let out_t = current_state.matmul(&qt.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)?;
            outputs.push(out_t.unsqueeze(1)?);
        }
        
        let out = Tensor::cat(&outputs, 1)?;
        self.delta_state = Some(current_state.detach().to_device(&Device::Cpu)?);
        let gate = z.silu()?;
        let res = self.out_proj.forward(&(out * gate)?, reg)?;
        self.norm.forward(&res, reg)
    }
}

pub struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    scaling: f64,
    pub h: usize,
    pub kv_cache: Option<(Tensor, Tensor)>,
}

impl QAttention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let p = vb.prefix();
        Ok(Self {
            q_proj: QLinear::new(&format!("{}.q_proj", p))?,
            k_proj: QLinear::new(&format!("{}.k_proj", p))?,
            v_proj: QLinear::new(&format!("{}.v_proj", p))?,
            o_proj: QLinear::new(&format!("{}.o_proj", p))?,
            q_norm: QRmsNorm::new(&format!("{}.q_norm", p), config.rms_norm_eps)?,
            k_norm: QRmsNorm::new(&format!("{}.k_norm", p), config.rms_norm_eps)?,
            nh: config.num_attention_heads,
            nkv: config.num_key_value_heads,
            hd: config.head_dim,
            scaling: 1.0 / (config.head_dim as f64).sqrt(),
            h: config.hidden_size,
            kv_cache: None,
        })
    }
    pub fn clear_vram(&mut self) { self.q_proj.clear_vram(); self.k_proj.clear_vram(); self.v_proj.clear_vram(); self.o_proj.clear_vram(); self.q_norm.clear_vram(); self.k_norm.clear_vram(); }

    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        reg: &QuantizedRegistry,
    ) -> Result<Tensor> {
        let (bs, sl, _) = x.dims3()?;
        let q_raw = self.q_proj.forward(x, reg)?; 
        let k_raw = self.k_proj.forward(x, reg)?; 
        let v_raw = self.v_proj.forward(x, reg)?; 
        let head_total = self.nh * self.hd;
        let q_states = q_raw.narrow(D::Minus1, 0, head_total)?;
        let gate = q_raw.narrow(D::Minus1, head_total, head_total)?;
        let q = self.q_norm.forward(&q_states.reshape((bs, sl, self.nh, self.hd))?, reg)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k_raw.reshape((bs, sl, self.nkv, k_raw.dim(D::Minus1)? / self.nkv))?, reg)?.transpose(1, 2)?;
        let v = v_raw.reshape((bs, sl, self.nkv, v_raw.dim(D::Minus1)? / self.nkv))?.transpose(1, 2)?;
        let (q, k) = glm_asr_apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let attn = crate::models::common::eager_attention_forward(&q, &k, &v, Some(self.nh / self.nkv), mask, self.scaling)?;
        let out = attn.reshape((bs, sl, ()))?.contiguous()?;
        let out_gated = (out * gate.silu()?)?;
        self.o_proj.forward(&out_gated, reg)
    }
}

pub struct QDecoderLayer { pub self_attn: Option<QAttention>, pub linear_attn: Option<QGatedDeltaNet>, mlp_gate: QLinear, mlp_up: QLinear, mlp_down: QLinear, in_norm: QRmsNorm, post_norm: QRmsNorm }
impl QDecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, idx: usize) -> Result<Self> {
        let lt = config.layer_types[idx].clone(); let p = vb.prefix();
        let (sa, la) = if lt == "linear_attention" { (None, Some(QGatedDeltaNet::new(vb.pp("linear_attn"), config)?)) } else { (Some(QAttention::new(vb.pp("self_attn"), config)?), None) };
        Ok(Self { self_attn: sa, linear_attn: la, mlp_gate: QLinear::new(&format!("{}.mlp.gate_proj", p))?, mlp_up: QLinear::new(&format!("{}.mlp.up_proj", p))?, mlp_down: QLinear::new(&format!("{}.mlp.down_proj", p))?, in_norm: QRmsNorm::new(&format!("{}.input_layernorm", p), config.rms_norm_eps)?, post_norm: QRmsNorm::new(&format!("{}.post_attention_layernorm", p), config.rms_norm_eps)? })
    }
    pub fn clear_vram(&mut self) {
        self.in_norm.clear_vram();
        if let Some(ref mut sa) = self.self_attn { sa.clear_vram(); }
        if let Some(ref mut la) = self.linear_attn { la.clear_vram(); }
        self.post_norm.clear_vram();
        self.mlp_gate.clear_vram(); self.mlp_up.clear_vram(); self.mlp_down.clear_vram();
    }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.in_norm.load_to_vram(reg, dev)?;
        if let Some(ref mut sa) = self.self_attn {
            sa.q_proj.load_to_vram(reg, dev)?; sa.k_proj.load_to_vram(reg, dev)?; sa.v_proj.load_to_vram(reg, dev)?; sa.o_proj.load_to_vram(reg, dev)?;
            sa.q_norm.load_to_vram(reg, dev)?; sa.k_norm.load_to_vram(reg, dev)?;
        }
        if let Some(ref mut la) = self.linear_attn {
            la.in_proj.load_to_vram(reg, dev)?; la.out_proj.load_to_vram(reg, dev)?; la.norm.load_to_vram(reg, dev)?;
        }
        self.post_norm.load_to_vram(reg, dev)?;
        self.mlp_gate.load_to_vram(reg, dev)?; self.mlp_up.load_to_vram(reg, dev)?; self.mlp_down.load_to_vram(reg, dev)?;
        Ok(())
    }
    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, reg: &QuantizedRegistry) -> Result<Tensor> {
        let residual = x.clone();
        let mut h = self.in_norm.forward(x, reg)?;
        if let Some(ref mut sa) = self.self_attn { h = sa.forward(&h, cos, sin, mask, reg)?; }
        else if let Some(ref mut la) = self.linear_attn { h = la.forward(&h, mask, reg)?; }
        let h2 = if h.dims() != residual.dims() { (h.narrow(D::Minus1, 0, residual.dim(D::Minus1)?)? + residual)? }
        else { (h + residual)? };
        let residual_2 = h2.clone();
        let h = self.post_norm.forward(&h2, reg)?;
        let gate = self.mlp_gate.forward(&h, reg)?.silu()?;
        let up = self.mlp_up.forward(&h, reg)?;
        let mlp_out = self.mlp_down.forward(&(gate * up)?, reg)?;
        if mlp_out.dims() != residual_2.dims() { Ok((mlp_out.narrow(D::Minus1, 0, residual_2.dim(D::Minus1)?)? + residual_2)?) }
        else { Ok((mlp_out + residual_2)?) }
    }
}

pub struct QEmbedding { name: String, persistent_weight: Option<Tensor> }
impl QEmbedding {
    pub fn new(name: &str) -> Self { Self { name: name.to_string(), persistent_weight: None } }
    pub fn clear_vram(&mut self) { self.persistent_weight = None; }
    pub fn load_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        let w = if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(&self.name, "scales"), reg.get_q_tensor(&self.name, "data"), reg.get_q_tensor(&self.name, "shape")) {
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let d_f16 = d_t.to_device(dev)?.to_dtype(DType::F16)?; let s_f16 = s_t.to_device(dev)?.to_dtype(DType::F16)?;
            let restored = if d_t.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d_f16.clone() / 16.0)?.floor()?; let low = d_f16.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s_f16.unsqueeze(D::Minus1)?.broadcast_as((s_f16.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s_f16.unsqueeze(D::Minus1)?.broadcast_as((s_f16.dim(0)?, 32))?.flatten_all()?;
                d_f16.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else { reg.get_tensor(&self.name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.name)))?.to_device(dev)?.to_dtype(DType::F16)? };
        self.persistent_weight = Some(w);
        Ok(())
    }
    pub fn forward(&self, ids: &Tensor, reg: &QuantizedRegistry) -> Result<Tensor> {
        let (b, s) = ids.dims2()?; let dev = ids.device();
        let ids_u32 = ids.flatten_all()?.to_dtype(DType::U32)?;
        let w = if let Some(pw) = &self.persistent_weight { pw.clone() }
        else if let (Ok(s_t), Ok(d_t), Ok(sh_t)) = (reg.get_q_tensor(&self.name, "scales"), reg.get_q_tensor(&self.name, "data"), reg.get_q_tensor(&self.name, "shape")) {
            let shape: Vec<usize> = sh_t.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
            let d_f16 = d_t.to_device(dev)?.to_dtype(DType::F16)?; let s_f16 = s_t.to_device(dev)?.to_dtype(DType::F16)?;
            let restored = if d_t.dim(D::Minus1)? * 2 == shape.iter().product::<usize>() {
                let high = (d_f16.clone() / 16.0)?.floor()?; let low = d_f16.sub(&(high.clone() * 16.0)?)?;
                let combined = Tensor::stack(&[low.affine(1.0, -8.0)?, high.affine(1.0, -8.0)?], D::Minus1)?.flatten_all()?;
                let s_exp = s_f16.unsqueeze(D::Minus1)?.broadcast_as((s_f16.dim(0)?, 32))?.flatten_all()?;
                combined.broadcast_mul(&s_exp)?
            } else {
                let s_exp = s_f16.unsqueeze(D::Minus1)?.broadcast_as((s_f16.dim(0)?, 32))?.flatten_all()?;
                d_f16.affine(1.0, -128.0)?.broadcast_mul(&s_exp)?
            };
            restored.reshape(shape.as_slice())?
        } else { reg.get_tensor(&self.name).or_else(|_| reg.get_tensor(&format!("{}.weight", self.name)))?.to_device(dev)?.to_dtype(DType::F16)? };
        Ok(w.index_select(&ids_u32, 0)?.reshape((b, s, ()))?.to_device(dev)?.to_dtype(DType::F16)?)
    }
}

pub struct QTextModel { pub embed: QEmbedding, pub layers: Vec<QDecoderLayer>, pub norm: QRmsNorm, pub rotary: Qwen3_5TextRotaryEmbedding, pub mrope: Vec<usize> }
impl QTextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let mut layers = vec![]; for i in 0..config.num_hidden_layers { layers.push(QDecoderLayer::new(vb.pp("layers").pp(i), config, i)?); }
        let p = vb.prefix();
        Ok(Self { embed: QEmbedding::new(&format!("{}.embed_tokens", p)), layers, norm: QRmsNorm::new(&format!("{}.norm", p), config.rms_norm_eps)?, rotary: Qwen3_5TextRotaryEmbedding::new((config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize, config.rope_parameters.rope_theta), mrope: config.rope_parameters.mrope_section.clone() })
    }
    pub fn forward(&mut self, ids: &Tensor, pos: &Tensor, offset: usize, reg: &QuantizedRegistry) -> Result<Tensor> {
        let bs = ids.dim(0)?; let sl = ids.dim(1)?; let mut x = self.embed.forward(ids, reg)?;
        let (cos, sin) = self.rotary.forward(pos, DType::F16, self.mrope.clone())?;
        let mask = if sl <= 1 && offset == 0 { None } else { Some(prepare_causal_attention_mask(bs, sl, offset, ids.device())?.to_dtype(DType::F16)?) };
        for l in self.layers.iter_mut() { 
            x = l.forward(&x, &cos, &sin, mask.as_ref(), reg)?; 
            #[cfg(feature = "cuda")]
            x.device().synchronize()?;
        }
        self.norm.forward(&x, reg)
    }
}

pub struct QuantizedQwen3_5Model { pub model: QTextModel, pub head: QLinear }
impl QuantizedQwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> { Ok(Self { model: QTextModel::new(vb.pp("model.language_model"), &config.text_config)?, head: QLinear::new("model.language_model.embed_tokens")? }) }
    pub fn load_all_to_vram(&mut self, reg: &QuantizedRegistry, dev: &Device) -> Result<()> {
        self.model.embed.load_to_vram(reg, dev)?;
        for l in self.model.layers.iter_mut() { l.load_to_vram(reg, dev)?; }
        self.model.norm.load_to_vram(reg, dev)?;
        self.head.load_to_vram(reg, dev)?;
        Ok(())
    }
    pub fn forward(&mut self, ids: &Tensor, offset: usize, reg: &QuantizedRegistry, _img: Option<&Tensor>) -> Result<Tensor> {
        let dev = ids.device(); let sl = ids.dim(1)?;
        let pos_vec: Vec<u32> = (offset as u32..(offset + sl) as u32).collect();
        let pos = Tensor::from_vec(pos_vec, (1, sl), dev)?.to_dtype(DType::U32)?.unsqueeze(0)?.broadcast_as((3, ids.dim(0)?, sl))?.contiguous()?;
        let h = self.model.forward(ids, &pos, offset, reg)?;
        self.head.forward(&h.narrow(1, sl - 1, 1)?, reg)
    }
    pub fn clear_cache(&mut self) { for l in self.model.layers.iter_mut() { if let Some(ref mut sa) = l.self_attn { sa.kv_cache = None; } if let Some(ref mut la) = l.linear_attn { la.delta_state = None; } } }
}

use memmap2::MmapMut;
use std::fs::OpenOptions;
use candle_core::Shape;

pub enum LayerContext {
    Attention { k: Tensor, v: Tensor },
    DeltaNet { state: Tensor },
}

pub struct DiskStateManager {
    mmap: MmapMut,
    layer_stride: usize,
}

impl DiskStateManager {
    pub fn new(num_layers: usize) -> Result<Self> {
        let layer_stride = 128 * 1024 * 1024; // 128MB per layer
        let total_size = num_layers * layer_stride;
        let path = crate::utils::paths::get_kv_dir(None).join("kv_cache_pool.bin");
        let file = OpenOptions::new()
            .read(true).write(true).create(true).open(path)?;
        file.set_len(total_size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, layer_stride })
    }

    pub fn save_layer_context(&mut self, layer_idx: usize, ctx: &LayerContext) -> Result<()> {
        let offset = layer_idx * self.layer_stride;
        match ctx {
            LayerContext::Attention { k, v } => {
                self.write_tensor(offset, k)?;
                let v_offset = offset + (k.elem_count() * 2);
                self.write_tensor(v_offset, v)?;
            },
            LayerContext::DeltaNet { state } => {
                self.write_tensor(offset, state)?;
            }
        }
        Ok(())
    }

    pub fn load_layer_context(&self, layer_idx: usize, lt: &str, shape_k: &Shape, shape_v: &Shape, dev: &Device) -> Result<LayerContext> {
        let offset = layer_idx * self.layer_stride;
        if lt == "linear_attention" {
            let state = self.read_tensor(offset, shape_k, dev)?;
            Ok(LayerContext::DeltaNet { state })
        } else {
            let k = self.read_tensor(offset, shape_k, dev)?;
            let v_offset = offset + (shape_k.elem_count() * 2);
            let v = self.read_tensor(v_offset, shape_v, dev)?;
            Ok(LayerContext::Attention { k, v })
        }
    }

    fn write_tensor(&mut self, offset: usize, t: &Tensor) -> Result<()> {
        let cpu_t = t.detach().to_device(&Device::Cpu)?.to_dtype(DType::F16)?;
        let data = cpu_t.flatten_all()?.to_vec1::<f16>()?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        };
        self.mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn read_tensor(&self, offset: usize, shape: &Shape, dev: &Device) -> Result<Tensor> {
        let byte_len = shape.elem_count() * 2;
        let raw_slice = &self.mmap[offset..offset + byte_len];
        Ok(Tensor::from_raw_buffer(raw_slice, DType::F16, shape.dims(), dev)?)
    }
}
