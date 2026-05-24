use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_transformers::models::deepseek2::SplitOp;

use crate::utils::tensor_utils::{split_tensor};

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
    
    // [CRITICAL FIX] 매 Attention 마다 발생하는 무거운 메모리 재정렬/복사 오버헤드(.contiguous()) 제거!
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
    Ok(rotate_x)
}

pub fn apply_multimodel_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    let mrope_section = mrope_section.repeat(2);
    let cos_select: Vec<Tensor> = cos
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
        .collect();
    let cos = Tensor::cat(&cos_select, D::Minus1)?.unsqueeze(1)?;
    let sin_select: Vec<Tensor> = sin
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m): (usize, &Tensor)| m.i(i % 3).unwrap())
        .collect();
    let sin = Tensor::cat(&sin_select, D::Minus1)?.unsqueeze(1)?;
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
    // 1. 차원 확장만 수행 (메모리 복사 없음)
    let cos = cos.unsqueeze(D::Minus2)?; 
    let sin = sin.unsqueeze(D::Minus2)?; 
    
    // [CRITICAL FIX] 이미 외부에서 q.dtype()과 일치하게 넘어오므로 
    // .to_dtype() 캐스팅 과정을 완전히 삭제하여 GPU Sync Stall을 제거합니다. 
    
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
    // 1. [FIX] 무조건적인 clone() 제거. 필요한 랭크일 때만 가상 뷰(unsqueeze) 생성
    let cos = if cos.rank() == 2 { cos.unsqueeze(0)?.unsqueeze(0)? } 
              else if cos.rank() == 3 { cos.unsqueeze(1)? } 
              else { cos.clone() }; 
    let sin = if sin.rank() == 2 { sin.unsqueeze(0)?.unsqueeze(0)? } 
              else if sin.rank() == 3 { sin.unsqueeze(1)? } 
              else { sin.clone() }; 

    let orig_dtype = q.dtype();
    
    // 2. [FIX] tof32 플래그가 참일 때만 물리적 타입 변환 수행 (No-Op 방지)
    let (q_work, k_work) = if tof32 { 
        (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?) 
    } else { 
        (q.clone(), k.clone()) 
    };

    // cos, sin의 타입이 연산 대상(q_work)과 다를 때만 1번 캐스팅
    let cos = if cos.dtype() != q_work.dtype() { cos.to_dtype(q_work.dtype())? } else { cos }; 
    let sin = if sin.dtype() != q_work.dtype() { sin.to_dtype(q_work.dtype())? } else { sin }; 

    let q_embed = q_work.broadcast_mul(&cos)?.add(&rotate_half(&q_work)?.broadcast_mul(&sin)?)?; 
    let k_embed = k_work.broadcast_mul(&cos)?.add(&rotate_half(&k_work)?.broadcast_mul(&sin)?)?; 

    // 3. [FIX] 결과 반환 시에도 불필요한 재캐스팅 방지
    let (q_final, k_final) = if tof32 {
        (q_embed.to_dtype(orig_dtype)?, k_embed.to_dtype(orig_dtype)?) 
    } else {
        (q_embed, k_embed)
    };

    Ok((q_final, k_final)) 
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
pub struct QwenVLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl QwenVLTextRotaryEmbedding {
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
        // [CRITICAL FIX] 확장을 가상으로만 유지하기 위해 F32 캐스팅을 최우선으로 수행
        let pos_f32 = position_ids.to_dtype(DType::F32)?;
        
        let position_ids = if pos_f32.rank() == 2 {
            let (bs, len) = pos_f32.dims2()?;
            pos_f32.unsqueeze(0)?.expand((3, bs, len))? 
        } else {
            pos_f32
        };
        
        let position_ids_expanded = position_ids.unsqueeze(D::Minus2)?;

        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?; // <-- contiguous() 삭제!

        // Calculate frequencies for T, H, W dimensions
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?; // (3, b_sz, seq_len, dim/2)

        // [CRITICAL FIX] cat이나 unsqueeze 이후의 contiguous()도 모두 삭제!
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
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
        
        let cos_select: Vec<Tensor> = cos_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        
        let cos = Tensor::cat(&cos_select, D::Minus1)?.unsqueeze(1)?.contiguous()?; 

        let sin_select: Vec<Tensor> = sin_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        let sin = Tensor::cat(&sin_select, D::Minus1)?.unsqueeze(1)?.contiguous()?; 

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
    let bs = position_ids.dim(0)?;
    let seq_len = position_ids.dim(1)?;

    // [CRITICAL FIX] O(N) 루프와 느린 stack, permute를 단 1번의 커널 호출로 압축!
    // 1. 인덱스를 1차원으로 쭉 폅니다 (이전에 겪으신 에러를 막기 위해 contiguous 보장)
    let flat_pos = position_ids.flatten_all()?.contiguous()?;

    // 2. 단 한 번의 index_select로 전체 배치의 코사인/사인 값을 가져옵니다.
    let cos_flat = cos.index_select(&flat_pos, 0)?;
    let sin_flat = sin.index_select(&flat_pos, 0)?;

    // 3. 목표했던 최종 Shape (bs, 1, seq_len, head_dim)으로 즉시 변환
    let head_dim = cos_flat.dim(D::Minus1)?;
    let cos = cos_flat.reshape((bs, seq_len, head_dim))?.unsqueeze(1)?;
    let sin = sin_flat.reshape((bs, seq_len, head_dim))?.unsqueeze(1)?;

    // 이후 로직은 동일 (메모리 이동 없이 메타데이터만 조작하므로 초고속)
    let xdrope_section: Vec<usize> = xdrope_section.iter().map(|&i| i * 2).collect();
    let cos_select: Vec<Tensor> = split_tensor(&cos, &xdrope_section, D::Minus1)?
        .iter().enumerate().map(|(i, m)| m.i((.., .., i % x_dim)).unwrap()).collect();
    let sin_select: Vec<Tensor> = split_tensor(&sin, &xdrope_section, D::Minus1)?
        .iter().enumerate().map(|(i, m)| m.i((.., .., i % x_dim)).unwrap()).collect();

    let cos = Tensor::cat(&cos_select, D::Minus1)?;
    let sin = Tensor::cat(&sin_select, D::Minus1)?;
    Ok((cos, sin))
}