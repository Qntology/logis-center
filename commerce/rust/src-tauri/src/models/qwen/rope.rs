use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

#[cfg(feature = "cuda")]
extern "C" {
    fn fused_apply_rotary_pos_emb(
        q_ptr: *mut std::ffi::c_void,
        k_ptr: *mut std::ffi::c_void,
        cos_ptr: *const std::ffi::c_void,
        sin_ptr: *const std::ffi::c_void,
        batch_size: std::ffi::c_int,
        seq_len: std::ffi::c_int,
        q_heads: std::ffi::c_int,
        k_heads: std::ffi::c_int,
        head_dim: std::ffi::c_int,
    );
}

pub fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if q.device().is_cuda() && q.dtype() == DType::F16 {
            let mut q_work = q.clone();
            let mut k_work = k.clone();
            let (seq_len, q_heads, head_dim) = q_work.dims3()?;
            let (_, k_heads, _) = k_work.dims3()?;

            unsafe {
                use candle_core::Storage;
                use candle_core::backend::BackendStorage;
                let get_mut_ptr = |t: &mut Tensor| -> *mut std::ffi::c_void {
                    let (storage, _) = t.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *mut std::ffi::c_void,
                        _ => std::ptr::null_mut(),
                    }
                };
                let get_const_ptr = |t: &Tensor| -> *const std::ffi::c_void {
                    let (storage, _) = t.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *const std::ffi::c_void,
                        _ => std::ptr::null(),
                    }
                };

                let q_ptr = get_mut_ptr(&mut q_work);
                let k_ptr = get_mut_ptr(&mut k_work);
                let cos_ptr = get_const_ptr(cos);
                let sin_ptr = get_const_ptr(sin);

                if !q_ptr.is_null() && !k_ptr.is_null() && !cos_ptr.is_null() && !sin_ptr.is_null() {
                    fused_apply_rotary_pos_emb(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        1, 
                        seq_len as i32,
                        q_heads as i32,
                        k_heads as i32,
                        head_dim as i32,
                    );
                    return Ok((q_work, k_work));
                }
            }
        }
    }

    // 1. 차원 확장만 수행 (메모리 복사 없음)
    let cos_ex = cos.unsqueeze(D::Minus2)?; 
    let sin_ex = sin.unsqueeze(D::Minus2)?; 
    
    let q_embed = q
        .broadcast_mul(&cos_ex)?
        .add(&rotate_half(q)?.broadcast_mul(&sin_ex)?)?; 
    let k_embed = k
        .broadcast_mul(&cos_ex)?
        .add(&rotate_half(k)?.broadcast_mul(&sin_ex)?)?; 
    Ok((q_embed, k_embed)) 
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    let cos_orig = if cos.rank() == 2 { cos.unsqueeze(0)?.unsqueeze(0)? } 
              else if cos.rank() == 3 { cos.unsqueeze(1)? } 
              else { cos.clone() }; 
    let sin_orig = if sin.rank() == 2 { sin.unsqueeze(0)?.unsqueeze(0)? } 
              else if sin.rank() == 3 { sin.unsqueeze(1)? } 
              else { sin.clone() }; 

    let orig_dtype = q.dtype();
    
    let (mut q_work, mut k_work) = if tof32 { 
        (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?) 
    } else { 
        (q.clone(), k.clone()) 
    };

    let cos_f = if cos_orig.dtype() != q_work.dtype() { cos_orig.to_dtype(q_work.dtype())? } else { cos_orig }; 
    let sin_f = if sin_orig.dtype() != q_work.dtype() { sin_orig.to_dtype(q_work.dtype())? } else { sin_orig }; 

    #[cfg(feature = "cuda")]
    {
        if q_work.device().is_cuda() && q_work.dtype() == DType::F16 {
            let (b_sz, q_heads, seq_len, head_dim) = q_work.dims4()?;
            let (_, k_heads, _, _) = k_work.dims4()?;

            unsafe {
                use candle_core::Storage;
                use candle_core::backend::BackendStorage;
                let get_mut_ptr = |t: &mut Tensor| -> *mut std::ffi::c_void {
                    let (storage, _) = t.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *mut std::ffi::c_void,
                        _ => std::ptr::null_mut(),
                    }
                };
                let get_const_ptr = |t: &Tensor| -> *const std::ffi::c_void {
                    let (storage, _) = t.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *const std::ffi::c_void,
                        _ => std::ptr::null(),
                    }
                };

                let q_ptr = get_mut_ptr(&mut q_work);
                let k_ptr = get_mut_ptr(&mut k_work);
                let cos_ptr = get_const_ptr(&cos_f);
                let sin_ptr = get_const_ptr(&sin_f);

                if !q_ptr.is_null() && !k_ptr.is_null() && !cos_ptr.is_null() && !sin_ptr.is_null() {
                    fused_apply_rotary_pos_emb(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        b_sz as i32,
                        seq_len as i32,
                        q_heads as i32,
                        k_heads as i32,
                        head_dim as i32,
                    );
                    
                    let (q_final, k_final) = if tof32 {
                        (q_work.to_dtype(orig_dtype)?, k_work.to_dtype(orig_dtype)?) 
                    } else {
                        (q_work, k_work)
                    };
                    
                    return Ok((q_final, k_final));
                }
            }
        }
    }

    let q_embed = q_work.broadcast_mul(&cos_f)?.add(&rotate_half(&q_work)?.broadcast_mul(&sin_f)?)?; 
    let k_embed = k_work.broadcast_mul(&cos_f)?.add(&rotate_half(&k_work)?.broadcast_mul(&sin_f)?)?; 

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
        let cos_all = emb.cos()?;
        let sin_all = emb.sin()?;
        let mrope_section_doubled = mrope_section.iter().map(|&s| s * 2).collect::<Vec<_>>();
        
        if mrope_section_doubled.is_empty() {
            return Ok((
                cos_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?, 
                sin_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?
            ));
        }

        let sec0 = *mrope_section_doubled.get(0).unwrap_or(&0) as i32;
        let sec1 = *mrope_section_doubled.get(1).unwrap_or(&0) as i32;
        let sec2 = *mrope_section_doubled.get(2).unwrap_or(&0) as i32;

        #[cfg(feature = "cuda")]
        {
            if cos_all.device().is_cuda() && cos_all.dtype() == candle_core::DType::F16 && mrope_section_doubled.len() == 3 {
                let (dim3, bs, seq_len, head_dim) = cos_all.dims4()?;
                if dim3 == 3 {
                    let mut cos_out = Tensor::zeros((bs, seq_len, head_dim), candle_core::DType::F16, cos_all.device())?;
                    let mut sin_out = Tensor::zeros((bs, seq_len, head_dim), candle_core::DType::F16, sin_all.device())?;
                    
                    unsafe {
                        use candle_core::Storage;
                        use candle_core::backend::BackendStorage;
                        let get_const_ptr = |t: &Tensor| -> *const std::ffi::c_void {
                            let (storage, _) = t.storage_and_layout();
                            match &*storage { Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *const std::ffi::c_void, _ => std::ptr::null() }
                        };
                        let get_mut_ptr = |t: &mut Tensor| -> *mut std::ffi::c_void {
                            let (storage, _) = t.storage_and_layout();
                            match &*storage { Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *mut std::ffi::c_void, _ => std::ptr::null_mut() }
                        };

                        let c_in_ptr = get_const_ptr(&cos_all);
                        let s_in_ptr = get_const_ptr(&sin_all);
                        let c_out_ptr = get_mut_ptr(&mut cos_out);
                        let s_out_ptr = get_mut_ptr(&mut sin_out);

                        if !c_in_ptr.is_null() && !s_in_ptr.is_null() && !c_out_ptr.is_null() && !s_out_ptr.is_null() {
                            fused_mrope_select(c_in_ptr, c_out_ptr, bs as i32, seq_len as i32, head_dim as i32, sec0, sec1, sec2);
                            fused_mrope_select(s_in_ptr, s_out_ptr, bs as i32, seq_len as i32, head_dim as i32, sec0, sec1, sec2);
                            return Ok((cos_out.unsqueeze(1)?.to_dtype(dtype)?, sin_out.unsqueeze(1)?.to_dtype(dtype)?));
                        }
                    }
                }
            }
        }

        let cos_select: Vec<Tensor> = cos_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        let cos = Tensor::cat(&cos_select, D::Minus1)?.unsqueeze(1)?.contiguous()?; 

        let sin_select: Vec<Tensor> = sin_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        let sin = Tensor::cat(&sin_select, D::Minus1)?.unsqueeze(1)?.contiguous()?; 

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

        let mrope_section_doubled = mrope_section.iter().map(|&s| s * 2).collect::<Vec<_>>();
        if mrope_section_doubled.is_empty() {
            let cos = cos_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?;
            let sin = sin_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?;
            return Ok((cos, sin));
        }

        let sec0 = *mrope_section_doubled.get(0).unwrap_or(&0) as i32;
        let sec1 = *mrope_section_doubled.get(1).unwrap_or(&0) as i32;
        let sec2 = *mrope_section_doubled.get(2).unwrap_or(&0) as i32;

        #[cfg(feature = "cuda")]
        {
            if cos_all.device().is_cuda() && cos_all.dtype() == candle_core::DType::F16 && mrope_section_doubled.len() == 3 {
                let (dim3, bs, seq_len, head_dim) = cos_all.dims4()?;
                if dim3 == 3 {
                    let mut cos_out = Tensor::zeros((bs, seq_len, head_dim), candle_core::DType::F16, cos_all.device())?;
                    let mut sin_out = Tensor::zeros((bs, seq_len, head_dim), candle_core::DType::F16, sin_all.device())?;
                    
                    unsafe {
                        use candle_core::Storage;
                        use candle_core::backend::BackendStorage;
                        let get_const_ptr = |t: &Tensor| -> *const std::ffi::c_void {
                            let (storage, _) = t.storage_and_layout();
                            match &*storage { Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *const std::ffi::c_void, _ => std::ptr::null() }
                        };
                        let get_mut_ptr = |t: &mut Tensor| -> *mut std::ffi::c_void {
                            let (storage, _) = t.storage_and_layout();
                            match &*storage { Storage::Cuda(c) => c.as_cuda_slice::<half::f16>().unwrap().device_ptr(&c.device().cuda_stream()).0 as *mut std::ffi::c_void, _ => std::ptr::null_mut() }
                        };

                        let c_in_ptr = get_const_ptr(&cos_all);
                        let s_in_ptr = get_const_ptr(&sin_all);
                        let c_out_ptr = get_mut_ptr(&mut cos_out);
                        let s_out_ptr = get_mut_ptr(&mut sin_out);

                        if !c_in_ptr.is_null() && !s_in_ptr.is_null() && !c_out_ptr.is_null() && !s_out_ptr.is_null() {
                            fused_mrope_select(c_in_ptr, c_out_ptr, bs as i32, seq_len as i32, head_dim as i32, sec0, sec1, sec2);
                            fused_mrope_select(s_in_ptr, s_out_ptr, bs as i32, seq_len as i32, head_dim as i32, sec0, sec1, sec2);
                            return Ok((cos_out.unsqueeze(1)?.to_dtype(dtype)?, sin_out.unsqueeze(1)?.to_dtype(dtype)?));
                        }
                    }
                }
            }
        }

        let cos_select: Vec<Tensor> = cos_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        let cos = Tensor::cat(&cos_select, D::Minus1)?.unsqueeze(1)?; 

        let sin_select: Vec<Tensor> = sin_all.split(&mrope_section_doubled, D::Minus1)?
            .iter().enumerate().map(|(i, m)| m.i(i % 3).unwrap()).collect();
        let sin = Tensor::cat(&sin_select, D::Minus1)?.unsqueeze(1)?; 

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


#[cfg(feature = "cuda")]
extern "C" {
    fn fused_mrope_select(
        in_all_ptr: *const std::ffi::c_void,
        out_ptr: *mut std::ffi::c_void,
        bs: std::ffi::c_int, seq_len: std::ffi::c_int, head_dim: std::ffi::c_int,
        sec0: std::ffi::c_int, sec1: std::ffi::c_int, sec2: std::ffi::c_int,
    );
}