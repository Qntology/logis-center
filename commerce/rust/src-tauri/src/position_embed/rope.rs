use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let mrope_section = mrope_section.repeat(2);
    let cos_select: Vec<Tensor> = cos
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i(i % 3).unwrap())
        .collect();
    let cos = Tensor::cat(&cos_select, D::Minus1)?
        .unsqueeze(1)?
        .contiguous()?;
    let sin_select: Vec<Tensor> = sin
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i(i % 3).unwrap())
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
    let cos_ex = cos.unsqueeze(D::Minus2)?;
    let sin_ex = sin.unsqueeze(D::Minus2)?;
    let cos_f = cos_ex.to_dtype(q.dtype())?;
    let sin_f = sin_ex.to_dtype(q.dtype())?;

    if q.device().is_cpu() && q.dtype() == candle_core::DType::F32 {
        use rayon::prelude::*;
        let (seq_len, q_heads, head_dim) = q.dims3()?;
        let (_, k_heads, _) = k.dims3()?;
        let half_dim = head_dim / 2;

        let q_vec = q.to_vec1::<f32>().unwrap_or_else(|_| q.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let k_vec = k.to_vec1::<f32>().unwrap_or_else(|_| k.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let cos_vec = cos_f.to_vec1::<f32>().unwrap_or_else(|_| cos_f.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let sin_vec = sin_f.to_vec1::<f32>().unwrap_or_else(|_| sin_f.flatten_all().unwrap().to_vec1::<f32>().unwrap());

        let mut q_out = vec![0.0f32; q_vec.len()];
        let mut k_out = vec![0.0f32; k_vec.len()];

        q_out.par_chunks_mut(head_dim).enumerate().for_each(|(idx, q_chunk)| {
            let seq_idx = idx / q_heads;
            let q_base = idx * head_dim;
            let cos_base = seq_idx * head_dim;

            for d in 0..half_dim {
                let q1 = q_vec[q_base + d];
                let q2 = q_vec[q_base + d + half_dim];
                let c = cos_vec[cos_base + d];
                let s = sin_vec[cos_base + d];

                q_chunk[d] = q1 * c - q2 * s;
                q_chunk[d + half_dim] = q2 * c + q1 * s;
            }
        });

        k_out.par_chunks_mut(head_dim).enumerate().for_each(|(idx, k_chunk)| {
            let seq_idx = idx / k_heads;
            let k_base = idx * head_dim;
            let cos_base = seq_idx * head_dim;

            for d in 0..half_dim {
                let k1 = k_vec[k_base + d];
                let k2 = k_vec[k_base + d + half_dim];
                let c = cos_vec[cos_base + d];
                let s = sin_vec[cos_base + d];

                k_chunk[d] = k1 * c - k2 * s;
                k_chunk[d + half_dim] = k2 * c + k1 * s;
            }
        });

        let q_final = Tensor::from_vec(q_out, q.shape().clone(), &Device::Cpu)?;
        let k_final = Tensor::from_vec(k_out, k.shape().clone(), &Device::Cpu)?;
        return Ok((q_final, k_final));
    }

    let q_embed = q
        .broadcast_mul(&cos_f)?
        .add(&rotate_half(q)?.broadcast_mul(&sin_f)?)?;
    let k_embed = k
        .broadcast_mul(&cos_f)?
        .add(&rotate_half(k)?.broadcast_mul(&sin_f)?)?;
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

    if q_work.device().is_cpu() && q_work.dtype() == candle_core::DType::F32 {
        use rayon::prelude::*;
        let (b_sz, q_heads, seq_len, head_dim) = q_work.dims4()?;
        let (_, k_heads, _, _) = k_work.dims4()?;
        let half_dim = head_dim / 2;

        let q_vec = q_work.to_vec1::<f32>().unwrap_or_else(|_| q_work.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let k_vec = k_work.to_vec1::<f32>().unwrap_or_else(|_| k_work.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let cos_vec = cos_f.to_vec1::<f32>().unwrap_or_else(|_| cos_f.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let sin_vec = sin_f.to_vec1::<f32>().unwrap_or_else(|_| sin_f.flatten_all().unwrap().to_vec1::<f32>().unwrap());

        let mut q_out = vec![0.0f32; q_vec.len()];
        let mut k_out = vec![0.0f32; k_vec.len()];

        q_out.par_chunks_mut(head_dim).enumerate().for_each(|(idx, q_chunk)| {
            let rem = idx % (seq_len * q_heads);
            let seq_idx = rem / q_heads;
            let q_base = idx * head_dim;
            let cos_base = seq_idx * head_dim;

            for d in 0..half_dim {
                let q1 = q_vec[q_base + d];
                let q2 = q_vec[q_base + d + half_dim];
                let c = cos_vec[cos_base + d];
                let s = sin_vec[cos_base + d];

                q_chunk[d] = q1 * c - q2 * s;
                q_chunk[d + half_dim] = q2 * c + q1 * s;
            }
        });

        k_out.par_chunks_mut(head_dim).enumerate().for_each(|(idx, k_chunk)| {
            let rem = idx % (seq_len * k_heads);
            let seq_idx = rem / k_heads;
            let k_base = idx * head_dim;
            let cos_base = seq_idx * head_dim;

            for d in 0..half_dim {
                let k1 = k_vec[k_base + d];
                let k2 = k_vec[k_base + d + half_dim];
                let c = cos_vec[cos_base + d];
                let s = sin_vec[cos_base + d];

                k_chunk[d] = k1 * c - k2 * s;
                k_chunk[d + half_dim] = k2 * c + k1 * s;
            }
        });

        let q_final_t = Tensor::from_vec(q_out, q_work.shape().clone(), &Device::Cpu)?;
        let k_final_t = Tensor::from_vec(k_out, k_work.shape().clone(), &Device::Cpu)?;
        
        let (q_final, k_final) = if tof32 {
            (q_final_t.to_dtype(orig_dtype)?, k_final_t.to_dtype(orig_dtype)?) 
        } else {
            (q_final_t, k_final_t)
        };
        return Ok((q_final, k_final));
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

pub fn glm_asr_apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    // sin/cos: to (bs, 1, seq_len, head_dim/2)
    // q/k: (bs, n_head, seq_len, head_dim)
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        // (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        // (bs, seq_len, head_dim/2) -> (bs, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;
    let rotary_dim = cos.dim(D::Minus1)?;
    let q_dim = q.dim(D::Minus1)?;
    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let q_pass = q.narrow(D::Minus1, rotary_dim, q_dim - rotary_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;
    let k_pass = k.narrow(D::Minus1, rotary_dim, q_dim - rotary_dim)?;

    let q_embed = q_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half(&q_rot)?.broadcast_mul(&sin)?)?;
    let k_embed = k_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half(&k_rot)?.broadcast_mul(&sin)?)?;
    let q_embed = Tensor::cat(&[q_embed, q_pass], D::Minus1)?.to_dtype(orig_dtype)?;
    let k_embed = Tensor::cat(&[k_embed, k_pass], D::Minus1)?.to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

/// Interleaved rotation used by GLM-OCR text decoder.
///
/// Python `rotate_half_llm`:
///   x1 = x[..., 0::2]   # even indices
///   x2 = x[..., 1::2]   # odd indices
///   return stack((-x2, x1), dim=-1).flatten(-2)
///   # e.g. [q0,q1,q2,q3] → [-q1, q0, -q3, q2]
///
/// Each adjacent pair (x_{2i}, x_{2i+1}) is rotated to (-x_{2i+1}, x_{2i}).
/// This is the correct counterpart to `repeat_interleave(2)` style cos/sin.
fn rotate_half_llm(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;
    // Reshape (..., D) → (..., D/2, 2) so each row is one adjacent pair
    let mut pair_shape = x.dims().to_vec();
    let rank = pair_shape.len();
    pair_shape[rank - 1] = half;
    pair_shape.push(2);
    let x_pairs = x.reshape(pair_shape)?; // (..., half, 2)
    // col 0 = even elements [x0, x2, ...], col 1 = odd elements [x1, x3, ...]
    let x_even = x_pairs.narrow(D::Minus1, 0, 1)?; // (..., half, 1)
    let x_odd = x_pairs.narrow(D::Minus1, 1, 1)?; // (..., half, 1)
    let neg_x_odd = x_odd.affine(-1.0, 0.0)?;
    // Concatenate [-x_odd, x_even] → [[-x1,x0], [-x3,x2], ...]
    let result_pairs = Tensor::cat(&[&neg_x_odd, &x_even], D::Minus1)?; // (..., half, 2)
    // Flatten last two dims back to D: [-x1, x0, -x3, x2, ...]
    Ok(result_pairs.reshape(x.dims().to_vec())?)
}

pub fn glm_ocr_apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // GLM-OCR applies rotary to only the first rotary_dim of head_dim
    // cos/sin: (bs, seq_len, head_dim) - already doubled via cat(freqs, freqs)
    // q/k: (bs, n_head, seq_len, head_dim)
    // Python: unsqueeze_dim=1
    //   - rank 2 (seq_len, head_dim) -> unsqueeze(1) -> (seq_len, 1, head_dim)
    //   - rank 3 (bs, seq_len, head_dim) -> unsqueeze(1) -> (bs, 1, seq_len, head_dim)
    let mut cos = cos.clone();
    let mut sin = sin.clone();

    cos = cos.unsqueeze(1)?; // (seq_len, head_dim) -> (seq_len, 1, head_dim)
    sin = sin.unsqueeze(1)?;

    // Python: cos = cos[..., :cos.shape[-1]//2].repeat_interleave(2, dim=-1)
    // Take first half and interleave each element
    let full_dim = cos.dim(D::Minus1)?;
    let half_dim = full_dim / 2;
    let cos_half = cos.narrow(D::Minus1, 0, half_dim)?;
    let sin_half = sin.narrow(D::Minus1, 0, half_dim)?;
    // repeat_interleave(2, dim=-1): [a,b,c] -> [a,a,b,b,c,c]
    let cos_interleaved = cos_half
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&Tensor::ones(
            &[1, 1, 1, 1, 2],
            cos_half.dtype(),
            cos_half.device(),
        )?)?
        .reshape(cos.shape())?;
    let sin_interleaved = sin_half
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&Tensor::ones(
            &[1, 1, 1, 1, 2],
            sin_half.dtype(),
            sin_half.device(),
        )?)?
        .reshape(sin.shape())?;

    let cos = cos_interleaved.to_dtype(q.dtype())?;
    let sin = sin_interleaved.to_dtype(q.dtype())?;

    let rotary_dim = cos.dim(D::Minus1)?;
    // Split q/k into rotary and pass-through portions
    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let q_pass = q.narrow(D::Minus1, rotary_dim, q.dim(D::Minus1)? - rotary_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;
    let k_pass = k.narrow(D::Minus1, rotary_dim, k.dim(D::Minus1)? - rotary_dim)?;

    // Apply rotary: q_rot * cos + rotate_half_llm(q_rot) * sin
    // Must use interleaved rotate_half_llm (not split-half rotate_half) because
    // cos/sin use repeat_interleave(2) format: [c0,c0,c1,c1,...].
    // rotate_half_llm rotates adjacent pairs (q_{2i},q_{2i+1}) → (-q_{2i+1}, q_{2i}),
    // which is the correct counterpart for this cos/sin format.
    let q_embed = q_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half_llm(&q_rot)?.broadcast_mul(&sin)?)?;
    let k_embed = k_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half_llm(&k_rot)?.broadcast_mul(&sin)?)?;

    // Concatenate rotary and pass-through portions
    let q_embed = Tensor::cat(&[&q_embed, &q_pass], D::Minus1)?;
    let k_embed = Tensor::cat(&[&k_embed, &k_pass], D::Minus1)?;
    Ok((q_embed, k_embed))
}

pub fn roformer_rotate(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let last_dim = dims
        .last()
        .ok_or(anyhow!("Input tensor must have at least one dimension"))?;
    if last_dim % 2 != 0 {
        return Err(anyhow!(
            "Last dimension size must be even, got {}",
            last_dim
        ));
    }
    let new_dims: Vec<usize> = dims[..dims.len() - 1]
        .iter()
        .copied()
        .chain([last_dim / 2, 2])
        .collect();
    let x_reshape = x.reshape(new_dims)?;
    let x_chunks = x_reshape.chunk(2, D::Minus1)?;
    let x1 = &x_chunks[0];
    let x2 = &x_chunks[1];
    // let x1 = x_reshape.narrow(D::Minus1, 0, 1)?;
    // let x2 = x_reshape.narrow(D::Minus1, 1, 1)?;
    let x2_neg = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2_neg, x1], D::Minus1)?;
    Ok(rotate_x.flatten(D::Minus2, D::Minus1)?)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn fused_apply_rotary_pos_emb_roformer(
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

pub fn apply_rotary_pos_emb_roformer(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    let mut cos_orig = cos.clone();
    let mut sin_orig = sin.clone();
    if cos_orig.rank() == 2 {
        cos_orig = cos_orig.unsqueeze(0)?.unsqueeze(0)?;
        sin_orig = sin_orig.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos_orig.rank() == 3 {
        cos_orig = cos_orig.unsqueeze(1)?;
        sin_orig = sin_orig.unsqueeze(1)?;
    }

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
                    fused_apply_rotary_pos_emb_roformer(
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

    let q_embed = q_work.broadcast_mul(&cos_f)?.add(&roformer_rotate(&q_work)?.broadcast_mul(&sin_f)?)?.to_dtype(orig_dtype)?;
    let k_embed = k_work.broadcast_mul(&cos_f)?.add(&roformer_rotate(&k_work)?.broadcast_mul(&sin_f)?)?.to_dtype(orig_dtype)?;

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

        if cos_all.device().is_cpu() && cos_all.dtype() == DType::F32 {
            use rayon::prelude::*;
            let (dim3, bs, seq_len, head_dim) = cos_all.dims4()?;
            let total_elements = bs * seq_len * head_dim;
            let out_shape = (bs, seq_len, head_dim);
            
            let cos_all_vec = cos_all.to_vec1::<f32>().unwrap_or_else(|_| cos_all.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            let sin_all_vec = sin_all.to_vec1::<f32>().unwrap_or_else(|_| sin_all.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            
            let mut cos_out = vec![0.0f32; total_elements];
            let mut sin_out = vec![0.0f32; total_elements];
            
            cos_out.par_chunks_mut(head_dim).zip(sin_out.par_chunks_mut(head_dim)).enumerate().for_each(|(row, (c_chunk, s_chunk))| {
                for d in 0..head_dim {
                    let spatial_idx = if d >= sec0 as usize && d < (sec0 + sec1) as usize { 1 } else if d >= (sec0 + sec1) as usize { 2 } else { 0 };
                    let in_idx = spatial_idx * total_elements + row * head_dim + d;
                    c_chunk[d] = cos_all_vec[in_idx];
                    s_chunk[d] = sin_all_vec[in_idx];
                }
            });
            
            let cos_t = Tensor::from_vec(cos_out, out_shape.clone(), &Device::Cpu)?.unsqueeze(1)?.to_dtype(dtype)?;
            let sin_t = Tensor::from_vec(sin_out, out_shape, &Device::Cpu)?.unsqueeze(1)?.to_dtype(dtype)?;
            return Ok((cos_t, sin_t));
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
pub struct Qwen3VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen3VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    pub fn apply_interleaved_mrope(
        &self,
        freqs: &Tensor,
        mrope_section: Vec<usize>,
    ) -> Result<Tensor> {
        if freqs.device().is_cpu() && freqs.dtype() == DType::F32 {
            use rayon::prelude::*;
            let (_dim3, bs, seq_len, half_dim) = freqs.dims4()?;
            let total_elements = bs * seq_len * half_dim;
            let freqs_vec = freqs.to_vec1::<f32>().unwrap_or_else(|_| freqs.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            
            let mut out_vec = vec![0.0f32; total_elements];
            
            out_vec.par_chunks_mut(half_dim).enumerate().for_each(|(row, chunk)| {
                for d in 0..half_dim {
                    let mut spatial_idx = 0;
                    for dim in 1..3 {
                        if let Some(&sec) = mrope_section.get(dim) {
                            if d < sec * 3 && d % 3 == dim {
                                spatial_idx = dim;
                            }
                        }
                    }
                    let in_idx = spatial_idx * total_elements + row * half_dim + d;
                    chunk[d] = freqs_vec[in_idx];
                }
            });
            return Ok(Tensor::from_vec(out_vec, (bs, seq_len, half_dim), &Device::Cpu)?);
        }

        let mut freqs_t = freqs.i(0)?.contiguous()?; 

        for (dim, section) in mrope_section.iter().enumerate().skip(1) {
            let length = section * 3;
            let idx = Tensor::arange_step(dim as u32, length as u32, 3, freqs.device())?;
            let src = freqs.i(dim)?.contiguous()?; 
            let src = src.index_select(&idx, D::Minus1)?.contiguous()?;
            let idx = idx
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(src.shape())?
                .contiguous()?;
            freqs_t = freqs_t.scatter(&idx, &src, D::Minus1)?;
        }
        Ok(freqs_t)
    }

    pub fn apply_interleaved_mrope_asr(
        &self,
        freqs: &Tensor,
        mrope_section: Vec<usize>,
    ) -> Result<Tensor> {
        if freqs.device().is_cpu() && freqs.dtype() == DType::F32 {
            use rayon::prelude::*;
            let (_dim3, bs, seq_len, half_dim) = freqs.dims4()?;
            let total_elements = bs * seq_len * half_dim;
            let freqs_vec = freqs.to_vec1::<f32>().unwrap_or_else(|_| freqs.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            
            let mut out_vec = vec![0.0f32; total_elements];
            
            out_vec.par_chunks_mut(half_dim).enumerate().for_each(|(row, chunk)| {
                for d in 0..half_dim {
                    let mut spatial_idx = 0;
                    for (offset_idx, offset) in (1..3).enumerate() {
                        let dim = offset_idx + 1;
                        if let Some(&length) = mrope_section.get(dim) {
                            if d < length && d % 3 == offset {
                                spatial_idx = dim;
                            }
                        }
                    }
                    let in_idx = spatial_idx * total_elements + row * half_dim + d;
                    chunk[d] = freqs_vec[in_idx];
                }
            });
            return Ok(Tensor::from_vec(out_vec, (bs, seq_len, half_dim), &Device::Cpu)?);
        }

        let mut freqs_t = freqs.i(0)?.contiguous()?; 

        for (dim, offset) in (1..3).enumerate() {
            let dim = dim + 1;
            let length = mrope_section[dim];
            let idx = Tensor::arange_step(offset as u32, length as u32, 3, freqs.device())?;
            let src = freqs.i(dim)?.contiguous()?; 
            let src = src.index_select(&idx, D::Minus1)?.contiguous()?;
            let idx = idx
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(src.shape())?
                .contiguous()?;
            freqs_t = freqs_t.scatter(&idx, &src, D::Minus1)?;
        }
        Ok(freqs_t)
    }

    pub fn forward_asr(
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
        let position_ids_expanded = position_ids.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?.contiguous()?;
        
        let inv_freq_expanded = Tensor::from_vec(self.inv_freq.clone(), (1, 1, self.inv_freq.len(), 1), position_ids.device())?
            .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
            .to_dtype(DType::F32)?.contiguous()?;

        let freqs = inv_freq_expanded.matmul(&position_ids_expanded)?.transpose(2, 3)?;
        let freqs = self.apply_interleaved_mrope_asr(&freqs, mrope_section)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
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
        let position_ids_expanded = position_ids.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?.contiguous()?;
        
        let inv_freq_expanded = Tensor::from_vec(self.inv_freq.clone(), (1, 1, self.inv_freq.len(), 1), position_ids.device())?
            .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
            .to_dtype(DType::F32)?.contiguous()?;

        let freqs = inv_freq_expanded.matmul(&position_ids_expanded)?.transpose(2, 3)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos_all = emb.cos()?;
        let sin_all = emb.sin()?;
        
        let mrope_section_doubled = mrope_section.iter().map(|&s| s * 2).collect::<Vec<_>>();
        if mrope_section_doubled.is_empty() {
            return Ok((cos_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?, sin_all.i(0)?.unsqueeze(1)?.to_dtype(dtype)?));
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

pub struct RoPE {
    inv_freq: Tensor, // (1, dim / 2)
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
            self.inv_freq.device(),
        )?
        .reshape((seq_len, 1))?; // (seq_len, 1)
        let freqs = positions.matmul(&self.inv_freq)?; // (seq_len, dim / 2)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?
            .contiguous()?
            .to_device(device)?; // (seq_len, dim)
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

    let flat_pos = position_ids.flatten_all()?.contiguous()?;

    let cos_flat = cos.index_select(&flat_pos, 0)?;
    let sin_flat = sin.index_select(&flat_pos, 0)?;

    let head_dim = cos_flat.dim(candle_core::D::Minus1)?;
    let cos = cos_flat.reshape((bs, seq_len, head_dim))?.unsqueeze(1)?;
    let sin = sin_flat.reshape((bs, seq_len, head_dim))?.unsqueeze(1)?;

    let xdrope_section: Vec<usize> = xdrope_section.iter().map(|&i| i * 2).collect();
    let cos_select: Vec<Tensor> = split_tensor(&cos, &xdrope_section, candle_core::D::Minus1)?
        .iter().enumerate().map(|(i, m)| m.i((.., .., i % x_dim)).unwrap()).collect();
    let sin_select: Vec<Tensor> = split_tensor(&sin, &xdrope_section, candle_core::D::Minus1)?
        .iter().enumerate().map(|(i, m)| m.i((.., .., i % x_dim)).unwrap()).collect();

    let cos = Tensor::cat(&cos_select, candle_core::D::Minus1)?;
    let sin = Tensor::cat(&sin_select, candle_core::D::Minus1)?;
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