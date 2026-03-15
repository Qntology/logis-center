use candle_core::{Tensor, Result, DType, Storage, Device};
use candle_core::backend::BackendDevice;
use std::ffi::c_void;

extern crate half;

extern "C" {
    fn launch_paged_flash_decoding_wrapper(
        query: *const c_void,
        k_blocks: *const *const c_void,
        v_blocks: *const *const c_void,
        block_lens: *const i32, 
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
            
            // cudarc 트레이트 충돌을 피하기 위한 로우레벨 포인터 추출 (CudaSlice의 첫 필드 접근)
            let device_ptr: u64 = match tensor.dtype() {
                DType::BF16 => unsafe { *(cuda_storage.as_cuda_slice::<half::bf16>()? as *const _ as *const u64) },
                DType::F16  => unsafe { *(cuda_storage.as_cuda_slice::<half::f16>()? as *const _ as *const u64) },
                DType::F32  => unsafe { *(cuda_storage.as_cuda_slice::<f32>()? as *const _ as *const u64) },
                DType::I64  => unsafe { *(cuda_storage.as_cuda_slice::<i64>()? as *const _ as *const u64) },
                DType::U32  => unsafe { *(cuda_storage.as_cuda_slice::<u32>()? as *const _ as *const u64) },
                DType::U8   => unsafe { *(cuda_storage.as_cuda_slice::<u8>()? as *const _ as *const u64) },
                _ => candle_core::bail!("Unsupported dtype for CUDA: {:?}", tensor.dtype()),
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
    let mut b_lens: Vec<u32> = Vec::with_capacity(num_blocks);
    
    // [CRITICAL FIX] 비동기 커널이 끝날 때까지 임시 텐서들이 VRAM에서 소멸하지 않도록 생명 연장(Keep Alive)
    let mut _keep_alive_k = Vec::with_capacity(num_blocks);
    let mut _keep_alive_v = Vec::with_capacity(num_blocks);

    for (k, v) in k_blocks.iter().zip(v_blocks.iter()) {
        let kc = if k.is_contiguous() { (*k).clone() } else { k.contiguous()? };
        let vc = if v.is_contiguous() { (*v).clone() } else { v.contiguous()? };
        
        k_ptrs.push(get_cuda_raw_ptr(&kc)? as i64);
        v_ptrs.push(get_cuda_raw_ptr(&vc)? as i64);
        b_lens.push(kc.dim(2)? as u32);
        
        // 여기에 담아두면 함수가 끝날 때까지 Tensor(메모리)가 안전하게 살아있습니다.
        _keep_alive_k.push(kc);
        _keep_alive_v.push(vc);
    }
    
    let k_table_gpu = Tensor::from_vec(k_ptrs, (num_blocks,), device)?;
    let v_table_gpu = Tensor::from_vec(v_ptrs, (num_blocks,), device)?;
    let l_table_gpu = Tensor::from_vec(b_lens, (num_blocks,), device)?;
    
    let out_tensor = Tensor::zeros((b_sz, num_heads, q_len, head_dim), DType::BF16, device)?;
    
    unsafe {
        launch_paged_flash_decoding_wrapper(
            get_cuda_raw_ptr(query)?, 
            get_cuda_raw_ptr(&k_table_gpu)? as *const *const c_void, 
            get_cuda_raw_ptr(&v_table_gpu)? as *const *const c_void, 
            get_cuda_raw_ptr(&l_table_gpu)? as *const i32, 
            get_cuda_raw_ptr(&out_tensor)? as *mut c_void,
            num_blocks as i32, num_heads as i32, num_kv_heads as i32, head_dim as i32, scale as f32, 256
        );
    }

    match device {
        Device::Cuda(c) => c.synchronize().map_err(candle_core::Error::wrap)?,
        _ => {}
    }
    
    // 동기화(synchronize)가 끝난 후, _keep_alive 백터가 소멸하면서 VRAM이 안전하게 해제됩니다.
    Ok(out_tensor)
}