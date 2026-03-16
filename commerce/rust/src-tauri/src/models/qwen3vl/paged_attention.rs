use candle_core::{Tensor, Result, DType, Storage, Device};
use candle_core::backend::BackendDevice; // [FIX] 트레이트 임포트
use std::ffi::c_void;

extern crate half;

extern "C" {
    fn launch_paged_flash_decoding_wrapper(
        query: *const c_void,
        k_blocks: *const *const c_void,
        v_blocks: *const *const c_void,
        block_lens: *const i32, // 👈 각 블록의 실제 길이를 담은 배열 (32비트)
        out: *mut c_void,
        num_blocks: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        block_size: i32,
    );
}

pub fn get_cuda_raw_ptr(tensor: &Tensor) -> Result<*const c_void> {
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

    let mut k_ptrs: Vec<i64> = Vec::with_capacity(num_blocks);
    let mut v_ptrs: Vec<i64> = Vec::with_capacity(num_blocks);
    
    // [FIX 1] u32로 변경 (Candle 지원 타입, 4바이트이므로 i32와 메모리 호환 완벽)
    let mut b_lens: Vec<u32> = Vec::with_capacity(num_blocks); 
    
    for (k, v) in k_blocks.iter().zip(v_blocks.iter()) {
        // 주의: 모델 로드 및 KV 캐시 저장 단계에서 이미 contiguous()를 보장하도록 구조가 잡혀 있어야 합니다.
        // (현재 generate.rs의 저장 로직에 contiguous()가 이미 적용되어 있으므로 안전합니다)
        k_ptrs.push(get_cuda_raw_ptr(k)? as i64);
        v_ptrs.push(get_cuda_raw_ptr(v)? as i64);
        b_lens.push(k.dim(2)? as u32);
    }
    
    let k_table_gpu = Tensor::from_vec(k_ptrs, (num_blocks,), device)?;
    let v_table_gpu = Tensor::from_vec(v_ptrs, (num_blocks,), device)?;
    let l_table_gpu = Tensor::from_vec(b_lens, (num_blocks,), device)?;

    // [FIX 2] 누락되었던 out_tensor 생성 로직 복구
    let out_tensor = Tensor::zeros((b_sz, num_heads, q_len, head_dim), DType::BF16, device)?;

    unsafe {
        launch_paged_flash_decoding_wrapper(
            get_cuda_raw_ptr(query)?, 
            get_cuda_raw_ptr(&k_table_gpu)? as *const *const c_void, 
            get_cuda_raw_ptr(&v_table_gpu)? as *const *const c_void, 
            // l_table_gpu는 u32 텐서이지만, C++이 원하는 i32와 크기가 같으므로 안전하게 캐스팅 가능
            get_cuda_raw_ptr(&l_table_gpu)? as *const i32, 
            get_cuda_raw_ptr(&out_tensor)? as *mut c_void,
            num_blocks as i32, num_heads as i32, num_kv_heads as i32, head_dim as i32, scale as f32, 256
        );
    }

    match device {
        Device::Cuda(c) => c.synchronize().map_err(candle_core::Error::wrap)?,
        _ => {}
    }
    Ok(out_tensor)
}