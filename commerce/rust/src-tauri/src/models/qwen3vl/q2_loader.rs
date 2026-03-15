use candle_core::{Tensor, Result, DType, Device};
use std::ffi::c_void;
use rayon::prelude::*;

extern "C" {
    fn launch_dequantize_q2_bf16(packed: *const c_void, scales: *const c_void, out: *mut c_void, num_packed_bytes: i32);
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

/// [GPU CUDA] 2-bit 데이터를 GPU로 복사한 뒤 VRAM 내부에서 해제합니다.
pub fn dequantize_q2_cuda(packed: Tensor, scales: Tensor, shape: &[usize], device: &Device) -> Result<Tensor> {
    // 1. 2비트 텐서를 그대로 GPU로 전송 (PCIe 대역폭 8배 절약)
    let packed_gpu = packed.to_device(device)?;
    let scales_gpu = scales.to_device(device)?;
    
    // 2. 출력용 BF16 텐서 할당
    let out_tensor = Tensor::zeros(shape, DType::BF16, device)?;
    
    let num_packed_bytes = packed.elem_count() as i32;
    
    unsafe {
        // (이전에 구현한 get_cuda_raw_ptr 활용)
        let p_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(&packed_gpu)?;
        let s_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(&scales_gpu)?;
        let o_ptr = crate::models::qwen3vl::paged_attention::get_cuda_raw_ptr(&out_tensor)? as *mut c_void;
        
        launch_dequantize_q2_bf16(p_ptr, s_ptr, o_ptr, num_packed_bytes);
    }
    
    // 3. 임시 2비트 텐서는 함수 종료 시 자동 파괴됨 (단일 참조 원칙 준수)
    Ok(out_tensor)
}

/// [I8-DEQUANTIZE] 8비트(I8) 텐서를 스케일 값을 사용하여 BF16/F32로 복원합니다.
pub fn dequantize_i8(data: Tensor, scale: Tensor, device: &Device) -> Result<Tensor> {
    let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    
    // 1. I8 데이터를 타겟 장치로 이동 후 float로 변환
    let data = data.to_device(device)?.to_dtype(DType::F32)?;
    let scale = scale.to_device(device)?.to_dtype(DType::F32)?;
    
    // 2. 스케일 곱하기 (Broadcasting 지원)
    let out = data.broadcast_mul(&scale)?;
    
    // 3. 최종 데이터 타입으로 변환
    out.to_dtype(target_dtype)
}