use candle_core::{Tensor, Result, DType, Device};
use std::ffi::c_void;
use rayon::prelude::*;

extern "C" {
    // 기존 dequantize...
    fn launch_dequantize_q2_bf16(packed: *const c_void, scales: *const c_void, out: *mut c_void, num_packed_bytes: i32);
    
    // [NEW] Fused Q2 GEMV 커널
    fn launch_fused_q2_gemv_bf16(
        packed_w: *const c_void, 
        scales: *const c_void, 
        in_vec: *const c_void, 
        out_vec: *mut c_void,
        in_features: i32, 
        out_features: i32
    );
}

// [Rust Wrapper] 안전한 Fused GEMV 호출 함수
pub fn fused_q2_gemv_cuda(packed_w: &Tensor, scales: &Tensor, in_vec: &Tensor, out_vec: &mut Tensor) -> candle_core::Result<()> {
    let in_features = in_vec.dim(candle_core::D::Minus1)? as i32;
    let out_features = out_vec.dim(candle_core::D::Minus1)? as i32;
    
    unsafe {
        let p_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(packed_w)?;
        let s_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(scales)?;
        let i_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(in_vec)?;
        let o_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(out_vec)? as *mut c_void;
        
        launch_fused_q2_gemv_bf16(p_ptr, s_ptr, i_ptr, o_ptr, in_features, out_features);
    }
    
    Ok(())
}

/// [CPU SIMD] CPU 환경에서 Q2를 BF16 텐서로 초고속 해제합니다.
pub fn dequantize_q2_cpu(packed: &[u8], scales: &[half::f16], shape: &[usize]) -> Result<Tensor> {
    let total_elements: usize = shape.iter().product();
    let mut output = vec![0.0f32; total_elements];
    
    // Rayon 병렬 루프 (Rust 컴파일러가 AVX2/NEON 명령어로 자동 최적화)
    output.par_chunks_exact_mut(32).enumerate().for_each(|(b_idx, block_out)| {
        let scale = scales[b_idx].to_f32();
        let packed_start = b_idx * 8; // 32개 엘리먼트 = 8바이트
        
        for i in 0..8 {
            let p = packed[packed_start + i];
            let idx = i * 4;
            block_out[idx + 0] = (((p >> 6) & 0x03) as f32 - 1.5) * scale;
            block_out[idx + 1] = (((p >> 4) & 0x03) as f32 - 1.5) * scale;
            block_out[idx + 2] = (((p >> 2) & 0x03) as f32 - 1.5) * scale;
            block_out[idx + 3] = ((p & 0x03) as f32 - 1.5) * scale;
        }
    });

    let t_f32 = Tensor::from_vec(output, shape, &Device::Cpu)?;
    Ok(t_f32.to_dtype(DType::BF16)?)
}

// [개선 코드]
pub fn dequantize_q2_cuda(packed: &Tensor, scales: &Tensor, out_tensor: &mut Tensor) -> candle_core::Result<()> {
    let num_packed_bytes = packed.elem_count() as i32;
    
    unsafe {
        let p_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(packed)?;
        let s_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(scales)?;
        // 이미 할당된 out_tensor의 포인터를 바로 넘김 (VRAM 할당 발생 안함!)
        let o_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(out_tensor)? as *mut c_void;
        
        launch_dequantize_q2_bf16(p_ptr, s_ptr, o_ptr, num_packed_bytes);
    }
    
    Ok(())
}