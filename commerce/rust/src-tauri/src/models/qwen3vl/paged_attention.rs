use candle_core::{Tensor, Result, DType, Storage, Device};
use candle_core::backend::BackendDevice; // 👈 [FIX] 추가: synchronize 사용을 위해 필수
use std::ffi::c_void;

extern crate half;

extern "C" {
    fn launch_paged_flash_decoding_wrapper(
        query: *const c_void,
        k_blocks: *const *const c_void,
        v_blocks: *const *const c_void,
        block_lens: *const i32, // 👈 각 블록의 실제 길이를 담은 배열
        out: *mut c_void,
        num_blocks: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        block_size: i32,
    );
}

/// 내부 함수: 텐서에서 CUDA 포인터를 추출합니다. (연속성 검사는 밖에서 수행)
fn get_cuda_raw_ptr(tensor: &Tensor) -> Result<*const c_void> {
    if !tensor.is_contiguous() {
        candle_core::bail!("Tensor must be contiguous to extract raw pointer");
    }

    let (storage, layout) = tensor.storage_and_layout();

    match &*storage {
        Storage::Cuda(cuda_storage) => {
            let element_size = tensor.dtype().size_in_bytes();
            let offset_bytes = layout.start_offset() * element_size;
            
            let device_ptr: u64 = match tensor.dtype() {
                DType::BF16 => unsafe { *(cuda_storage.as_cuda_slice::<half::bf16>()? as *const _ as *const u64) },
                DType::F32 => unsafe { *(cuda_storage.as_cuda_slice::<f32>()? as *const _ as *const u64) },
                DType::I64 => unsafe { *(cuda_storage.as_cuda_slice::<i64>()? as *const _ as *const u64) },
                DType::U32 => unsafe { *(cuda_storage.as_cuda_slice::<u32>()? as *const _ as *const u64) },
                DType::U8 => unsafe { *(cuda_storage.as_cuda_slice::<u8>()? as *const _ as *const u64) },
                _ => unsafe { *(cuda_storage.as_cuda_slice::<u8>()? as *const _ as *const u64) },
            };
            
            let raw_ptr = device_ptr as *const u8;
            let final_ptr = unsafe { raw_ptr.add(offset_bytes) as *const c_void };
            Ok(final_ptr)
        }
        _ => candle_core::bail!("Tensor is not on a CUDA device."),
    }
}

pub fn run_paged_flash_decoding(
    query: &Tensor, 
    k_blocks: &[&Tensor], 
    v_blocks: &[&Tensor], 
    num_kv_heads: usize,
    scale: f64,
) -> Result<Tensor> {
    let (b_sz, num_heads, q_len, head_dim) = query.dims4()?;
    let device = query.device();
    let num_blocks = k_blocks.len();

    let mut k_ptrs = Vec::with_capacity(num_blocks);
    let mut v_ptrs = Vec::with_capacity(num_blocks);
    // 👈 [FIX] i32 대신 u32를 사용하여 WithDType 에러 해결
    let mut b_lens: Vec<u32> = Vec::with_capacity(num_blocks); 
    
    for (k, v) in k_blocks.iter().zip(v_blocks.iter()) {
        k_ptrs.push(get_cuda_raw_ptr(k)? as i64);
        v_ptrs.push(get_cuda_raw_ptr(v)? as i64);
        // 👈 [FIX] u32로 저장
        b_lens.push(k.dim(2)? as u32); 
    }
    
    let k_table_gpu = Tensor::from_vec(k_ptrs, (num_blocks,), device)?;
    let v_table_gpu = Tensor::from_vec(v_ptrs, (num_blocks,), device)?;
    // 👈 [FIX] 이제 정상적으로 GPU 텐서 생성이 가능합니다.
    let l_table_gpu = Tensor::from_vec(b_lens, (num_blocks,), device)?;

    let out_tensor = Tensor::zeros((b_sz, num_heads, q_len, head_dim), DType::BF16, device)?;
    let out_ptr = get_cuda_raw_ptr(&out_tensor)? as *mut c_void;

    unsafe {
        launch_paged_flash_decoding_wrapper(
            get_cuda_raw_ptr(query)?, 
            get_cuda_raw_ptr(&k_table_gpu)? as *const *const c_void, 
            get_cuda_raw_ptr(&v_table_gpu)? as *const *const c_void, 
            // 👈 [FIX] 포인터만 i32 상수로 캐스팅하여 전달 (데이터 레이아웃은 u32와 i32가 동일함)
            get_cuda_raw_ptr(&l_table_gpu)? as *const i32,
            out_ptr,
            num_blocks as i32, 
            num_heads as i32, 
            num_kv_heads as i32, 
            head_dim as i32, 
            scale as f32, 
            256
        );
    }

    match device {
        Device::Cuda(c) => c.synchronize().map_err(candle_core::Error::wrap)?,
        _ => {}
    }

    Ok(out_tensor)
}