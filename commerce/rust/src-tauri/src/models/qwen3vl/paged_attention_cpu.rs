use candle_core::{Tensor, Result, DType, Storage};
use rayon::prelude::*;

pub fn run_paged_flash_decoding_cpu(
    query: &Tensor, 
    k_blocks: &[&Tensor], 
    v_blocks: &[&Tensor], 
    num_kv_heads: usize,
    scale: f64,
) -> Result<Tensor> {
    let (b_sz, num_heads, q_len, head_dim) = query.dims4()?;
    let scale_f32 = scale as f32;

    // 1. Query 데이터 준비
    let q_vec = query.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    // 2. K, V 블록들을 가공 가능한 형태로 준비
    // 수명 문제를 피하기 위해 데이터를 직접 참조하지 않고 텐서 객체 자체를 보관합니다.
    let blocks: Vec<_> = k_blocks.iter().zip(v_blocks.iter()).collect();

    // 3. 결과 버퍼 준비
    let mut out_vec = vec![0.0f32; num_heads * head_dim];

    // 4. Rayon 병렬 처리
    out_vec.par_chunks_exact_mut(head_dim)
        .enumerate()
        .for_each(|(head_idx, out_head)| {
            let kv_head_idx = head_idx / (num_heads / num_kv_heads);
            let q_head = &q_vec[head_idx * head_dim .. (head_idx + 1) * head_dim];

            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0.0f32;
            let mut acc = [0.0f32; 128]; // head_dim=128 가정

            for (k, v) in &blocks {
                // 연산 시점에 CPU 스토리지를 확보하여 수명 충돌 회피
                let (k_storage, _) = k.storage_and_layout();
                let (v_storage, _) = v.storage_and_layout();
                
                let k_data = match &*k_storage {
                    Storage::Cpu(cpu) => cpu.as_slice::<f32>().unwrap_or(&[]),
                    _ => &[],
                };
                let v_data = match &*v_storage {
                    Storage::Cpu(cpu) => cpu.as_slice::<f32>().unwrap_or(&[]),
                    _ => &[],
                };

                let actual_len = k.dim(2).unwrap_or(0);
                let head_offset = kv_head_idx * actual_len * head_dim;

                for t in 0..actual_len {
                    let token_offset = head_offset + (t * head_dim);
                    let k_slice = &k_data[token_offset .. token_offset + head_dim];
                    
                    let mut qk = 0.0f32;
                    for d in 0..head_dim {
                        qk += q_head[d] * k_slice[d];
                    }
                    qk *= scale_f32;
                    
                    let m_ij = m_i.max(qk);
                    let p = (qk - m_ij).exp();
                    let exp_diff = (m_i - m_ij).exp();

                    l_i = l_i * exp_diff + p;
                    m_i = m_ij;

                    let v_slice = &v_data[token_offset .. token_offset + head_dim];
                    for d in 0..head_dim {
                        acc[d] = acc[d] * exp_diff + p * v_slice[d];
                    }
                }
            }

            for d in 0..head_dim {
                out_head[d] = acc[d] / l_i;
            }
        });

    // 5. 결과를 Tensor로 복구
    Tensor::from_vec(out_vec, (b_sz, num_heads, q_len, head_dim), query.device())
}