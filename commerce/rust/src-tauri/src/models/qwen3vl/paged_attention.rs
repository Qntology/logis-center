use candle_core::{Tensor, Result, DType, Storage};
use std::ffi::c_void;

extern "C" {
    fn launch_paged_flash_decoding_wrapper(
        query: *const c_void,
        k_blocks: *const *const c_void,
        v_blocks: *const *const c_void,
        out: *mut c_void,
        num_blocks: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        block_size: i32,
    );
}

/// 내부 함수: Candle 텐서에서 CUDA Raw Pointer를 매우 강력하고 안전한 방법으로 추출합니다.
fn get_cuda_raw_ptr(tensor: &Tensor) -> Result<*const c_void> {
    let t = if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()?
    };

    let (storage, layout) = t.storage_and_layout();

    match &*storage {
        Storage::Cuda(cuda_storage) => {
            let element_size = t.dtype().size_in_bytes(); 
            let offset_bytes = layout.start_offset() * element_size;
            
            let slice = cuda_storage.as_cuda_slice::<u8>()?;
            
            // [HACK] cudarc API(CudaStream 요구)를 우회하기 위해, CudaSlice 구조체의 첫 번째 필드인
            // id (CUdeviceptr, u64) 값을 포인터 캐스팅을 통해 직접 강제로 읽어냅니다.
            let device_ptr: u64 = unsafe { *(slice as *const _ as *const u64) };
            
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
    if q_len != 1 || query.dtype() != DType::BF16 {
        candle_core::bail!("Only supports q_len == 1 and DType::BF16");
    }

    let num_blocks = k_blocks.len();
    let block_size = 256; 
    let device = query.device();

    let query_ptr = get_cuda_raw_ptr(query)?;

    // 1. K, V 포인터 추출 (Rust의 Vec에 수집)
    // [FIX] Candle은 u64 Tensor를 지원하지 않으므로 호환되는 i64를 사용합니다.
    let mut k_ptrs: Vec<i64> = Vec::with_capacity(num_blocks);
    let mut v_ptrs: Vec<i64> = Vec::with_capacity(num_blocks);
    
    for (k, v) in k_blocks.iter().zip(v_blocks.iter()) {
        k_ptrs.push(get_cuda_raw_ptr(*k)? as i64);
        v_ptrs.push(get_cuda_raw_ptr(*v)? as i64);
    }
    
    // 2. 포인터 배열을 담을 VRAM 텐서 생성 (i64 타입)
    let k_table = Tensor::from_vec(k_ptrs, (num_blocks,), device)?;
    let v_table = Tensor::from_vec(v_ptrs, (num_blocks,), device)?;

    // 3. VRAM 텐서의 포인터를 추출하여 C++ 커널에 넘기기 위한 캐스팅
    // [FIX] 함수 호출 전 변수에 담아 ? 연산자 충돌을 방지합니다.
    let k_table_ptr = get_cuda_raw_ptr(&k_table)? as *const *const c_void;
    let v_table_ptr = get_cuda_raw_ptr(&v_table)? as *const *const c_void;

    // 4. 출력 텐서 할당 (BF16)
    let out_tensor = Tensor::zeros((b_sz, num_heads, q_len, head_dim), DType::BF16, device)?;
    let out_ptr = get_cuda_raw_ptr(&out_tensor)? as *mut c_void;

    // 5. CUDA 커널 실행
    unsafe {
        launch_paged_flash_decoding_wrapper(
            query_ptr, 
            k_table_ptr, 
            v_table_ptr, 
            out_ptr,
            num_blocks as i32, 
            num_heads as i32, 
            num_kv_heads as i32, 
            head_dim as i32, 
            scale as f32, 
            block_size as i32
        );
    }

    Ok(out_tensor)
}