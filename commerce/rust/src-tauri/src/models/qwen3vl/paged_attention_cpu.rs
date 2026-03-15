use candle_core::{Tensor, Result, DType};
use rayon::prelude::*;

pub fn run_paged_flash_decoding_cpu(
    query: &Tensor, 
    k_blocks: &[&Tensor], 
    v_blocks: &[&Tensor], 
    num_kv_heads: usize,
    scale: f64,
) -> Result<Tensor> {
    let (b_sz, num_heads, q_len, head_dim) = query.dims4()?;
    
    // CPU 모드에서는 F32 포맷이 가장 빠르고 안정적입니다.
    let q_vec = query.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let scale_f32 = scale as f32;

    // 1. K, V 블록 데이터 추출 (메모리 재할당 최소화)
    let mut blocks_data = Vec::with_capacity(k_blocks.len());
    for (k, v) in k_blocks.iter().zip(v_blocks.iter()) {
        let actual_len = k.dim(2)?; // 해당 블록의 실제 토큰 길이 (256 이하일 수 있음)
        let k_vec = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        blocks_data.push((actual_len, k_vec, v_vec));
    }

    // 2. 결과를 담을 배열
    let mut out_vec = vec![0.0f32; num_heads * head_dim];

    // 3. Rayon을 이용해 다수의 Attention Head를 여러 CPU 코어에 분산 처리
    out_vec.par_chunks_exact_mut(head_dim)
        .enumerate()
        .for_each(|(head_idx, out_head)| {
            let kv_head_idx = head_idx / (num_heads / num_kv_heads);
            let q_head = &q_vec[head_idx * head_dim .. (head_idx + 1) * head_dim];

            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];

            // Paged Blocks 순회
            for (actual_len, k_block, v_block) in &blocks_data {
                for t in 0..*actual_len {
                    let token_offset = (kv_head_idx * (*actual_len) * head_dim) + (t * head_dim);
                    let k_slice = &k_block[token_offset .. token_offset + head_dim];
                    
                    // [A] Dot Product (Q * K^T)
                    // Rust LLVM이 이 zip.map.sum 체인을 AVX/NEON SIMD 명령어로 자동 최적화합니다.
                    let qk: f32 = q_head.iter()
                        .zip(k_slice.iter())
                        .map(|(q, k)| q * k)
                        .sum::<f32>() * scale_f32;
                    
                    // [B] Online Softmax (LogSumExp)
                    let m_ij = m_i.max(qk);
                    let p = (qk - m_ij).exp();
                    let exp_diff = (m_i - m_ij).exp();

                    l_i = l_i * exp_diff + p;
                    m_i = m_ij;

                    // [C] V 누적
                    let v_slice = &v_block[token_offset .. token_offset + head_dim];
                    for d in 0..head_dim {
                        acc[d] = acc[d] * exp_diff + p * v_slice[d];
                    }
                }
            }

            // 최종 정규화 및 기록
            for d in 0..head_dim {
                out_head[d] = acc[d] / l_i;
            }
        });

    // 4. 결과를 Tensor로 변환하여 리턴
    let out_tensor = Tensor::from_vec(out_vec, (b_sz, num_heads, q_len, head_dim), query.device())?;
    
    // CPU 모드의 타겟 타입(F32)에 맞게 캐스팅
    Ok(out_tensor.to_dtype(DType::F32)?)
}