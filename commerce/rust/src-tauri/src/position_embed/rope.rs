use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_transformers::models::deepseek2::SplitOp;

use crate::utils::tensor_utils::{index_select_2d, split_tensor};

pub fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0_f32 / base.powf(i as f32 / dim as f32))
        .collect();
    inv_freq
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x2 = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?.contiguous()?;
    Ok(rotate_x)
}

pub fn apply_multimodel_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device();
    let cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
    let sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };
    
    let mrope_section = mrope_section.repeat(2);
    let cos_select: Vec<Tensor> = cos
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
        .collect();
    let cos = Tensor::cat(&cos_select, D::Minus1)?
        .unsqueeze(1)?
        .contiguous()?;
    let sin_select: Vec<Tensor> = sin
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
        .collect();
    let sin = Tensor::cat(&sin_select, D::Minus1)?
        .unsqueeze(1)?
        .contiguous()?;
    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

pub fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device();
    let cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
    let sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };

    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;
    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device();
    let mut cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
    let mut sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };

    if cos.rank() == 2 {
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;

    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen2_5VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }
    pub fn forward(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        let mrope_section = mrope_section.repeat(2);
        let cos_select: Vec<Tensor> = cos
            .split(&mrope_section, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
            .collect();
        let cos = Tensor::cat(&cos_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;
        let sin_select: Vec<Tensor> = sin
            .split(&mrope_section, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
            .collect();
        let sin = Tensor::cat(&sin_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VisionRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen2_5VisionRotaryEmbedding {
    pub fn new(dim: usize, theta_base: Option<f32>) -> Self {
        let theta_base = theta_base.unwrap_or(10000.0_f32);
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    pub fn forward(&self, seqlen: usize, device: &Device) -> Result<Tensor> {
        let seq = Tensor::arange(0.0_f32, seqlen as f32, device)?.reshape((seqlen, 1))?;
        let inv_freq = Tensor::from_vec(self.inv_freq.clone(), (1, self.inv_freq.len()), device)?;
        let freqs = seq.matmul(&inv_freq)?;
        Ok(freqs)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen3VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    pub fn forward(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let position_ids = if position_ids.rank() == 2 {
            let (bs, len) = position_ids.dims2()?;
            position_ids.unsqueeze(0)?.expand((3, bs, len))?
        } else {
            position_ids.clone()
        };

        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        // Calculate frequencies for T, H, W dimensions
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?; // (3, b_sz, seq_len, dim/2)

        // Standard Qwen2-VL/Qwen3-VL block-based mRoPE assembly
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos_all = emb.cos()?;
        let sin_all = emb.sin()?;

        // If no sections defined, fallback to first dimension (Time)
        if mrope_section.is_empty() {
            let cos = cos_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?;
            let sin = sin_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?;
            return Ok((cos, sin));
        }

        // Split by sections and select based on dimension index
        let mrope_section_doubled = mrope_section.iter().map(|&s| s * 2).collect::<Vec<_>>();
        let dev = position_ids.device();
        
        let cos_select: Vec<Tensor> = cos_all
            .split(&mrope_section_doubled, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap().to_device(dev).unwrap())
            .collect();
        let cos = Tensor::cat(&cos_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;

        let sin_select: Vec<Tensor> = sin_all
            .split(&mrope_section_doubled, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap().to_device(dev).unwrap())
            .collect();
        let sin = Tensor::cat(&sin_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;

        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}

pub struct RoPE {
    inv_freq: Tensor, 
}

impl RoPE {
    pub fn new(dim: usize, theta_base: f32, device: &Device) -> Result<Self> {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?;

        Ok(Self { inv_freq })
    }
    pub fn forward(
        &self,
        seqlen_offset: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(
            seqlen_offset as f32,
            (seqlen_offset + seq_len) as f32,
            device,
        )?
        .reshape((seq_len, 1))?; 
        let freqs = positions.matmul(&self.inv_freq)?; 
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?; 
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

pub fn get_xd_cos_sin(
    cos: &Tensor,
    sin: &Tensor,
    position_ids: &Tensor,
    xdrope_section: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    let x_dim = xdrope_section.len();
    let mut cos_vec = vec![];
    let mut sin_vec = vec![];
    let bs = position_ids.dim(0)?;
    for i in 0..bs {
        let pos_i = position_ids.i(i)?;
        let cos_i = index_select_2d(cos, &pos_i)?;
        let sin_i = index_select_2d(sin, &pos_i)?;
        cos_vec.push(cos_i);
        sin_vec.push(sin_i);
    }
    let cos = Tensor::stack(&cos_vec, 0)?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    let sin = Tensor::stack(&sin_vec, 0)?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    let xdrope_section: Vec<usize> = xdrope_section.iter().map(|&i| i * 2).collect();
    let cos_select: Vec<Tensor> = split_tensor(&cos, &xdrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i((.., .., i % x_dim)).unwrap())
        .collect();
    let sin_select: Vec<Tensor> = split_tensor(&sin, &xdrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i((.., .., i % x_dim)).unwrap())
        .collect();

    let cos = Tensor::cat(&cos_select, D::Minus1)?;
    let sin = Tensor::cat(&sin_select, D::Minus1)?;
    Ok((cos, sin))
}
