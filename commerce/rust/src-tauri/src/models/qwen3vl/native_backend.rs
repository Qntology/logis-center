use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeDType {
    F32,
    F16,
    BF16,
    U32, 
}

#[derive(Clone)]
pub struct NativeTensor {
    pub data_ptr: *const u8,
    pub shape: Vec<usize>,
    pub dtype: NativeDType,
    pub _mmap: Option<Arc<Mmap>>,
}

unsafe impl Send for NativeTensor {}
unsafe impl Sync for NativeTensor {}

impl NativeTensor {
    pub fn from_mmap(mmap: Arc<Mmap>, offset: usize, shape: Vec<usize>, dtype: NativeDType) -> Self {
        unsafe {
            let data_ptr = mmap.as_ptr().add(offset);
            Self {
                data_ptr,
                shape,
                dtype,
                _mmap: Some(mmap),
            }
        }
    }

    pub fn get_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        unsafe { std::slice::from_raw_parts(self.data_ptr as *const T, size) }
    }
}

// --- Native Kernels ---

pub fn native_silu_f16(input: &mut [f16]) {
    input.par_iter_mut().for_each(|x| {
        let val = x.to_f32();
        let silu = val / (1.0 + (-val).exp());
        *x = f16::from_f32(silu);
    });
}

pub fn native_softmax_f16(input: &mut [f16], seq_len: usize) {
    input.par_chunks_exact_mut(seq_len).for_each(|row| {
        let mut max = f32::MIN;
        for &x in row.iter() {
            let val = x.to_f32();
            if val > max { max = val; }
        }
        
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            let val = (x.to_f32() - max).exp();
            sum += val;
            *x = f16::from_f32(val);
        }
        
        let inv_sum = 1.0 / sum;
        for x in row.iter_mut() {
            *x = f16::from_f32(x.to_f32() * inv_sum);
        }
    });
}

pub fn native_apply_rope_f16_with_offset(
    q: &mut [f16], k: &mut [f16], 
    q_len: usize, seqlen_offset: usize,
    n_heads: usize, head_dim: usize, 
    theta_base: f32
) {
    let half_dim = head_dim / 2;
    q.par_chunks_exact_mut(head_dim).enumerate().for_each(|(i, q_head)| {
        let pos = (seqlen_offset + i / n_heads) as f32;
        for d in 0..half_dim {
            let freq = pos / theta_base.powf((2 * d) as f32 / head_dim as f32);
            let (sin, cos) = freq.sin_cos();
            let (q0, q1) = (q_head[d].to_f32(), q_head[d + half_dim].to_f32());
            q_head[d] = f16::from_f32(q0 * cos - q1 * sin);
            q_head[d + half_dim] = f16::from_f32(q0 * sin + q1 * cos);
        }
    });
    k.par_chunks_exact_mut(head_dim).enumerate().for_each(|(i, k_head)| {
        let pos = (seqlen_offset + i / n_heads) as f32;
        for d in 0..half_dim {
            let freq = pos / theta_base.powf((2 * d) as f32 / head_dim as f32);
            let (sin, cos) = freq.sin_cos();
            let (k0, k1) = (k_head[d].to_f32(), k_head[d + half_dim].to_f32());
            k_head[d] = f16::from_f32(k0 * cos - k1 * sin);
            k_head[d + half_dim] = f16::from_f32(k0 * sin + k1 * cos);
        }
    });
}

pub fn native_rms_norm_f16(input: &[f16], weight: &[f16], eps: f32, hidden_size: usize) -> Vec<f16> {
    let mut output = vec![f16::ZERO; input.len()];
    input.par_chunks_exact(hidden_size).zip(output.par_chunks_exact_mut(hidden_size)).for_each(|(in_row, out_row)| {
        let mut variance = 0.0f32;
        for &x in in_row {
            let val = x.to_f32();
            variance += val * val;
        }
        variance /= hidden_size as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        for j in 0..hidden_size {
            if j >= weight.len() { break; } // [SAFETY] Prevent index out of bounds
            let val = in_row[j].to_f32() * inv_std * weight[j].to_f32();
            out_row[j] = f16::from_f32(val);
        }
    });
    output
}

pub fn native_embedding_lookup_f16(ids: &[u32], table: &[f16], hidden_size: usize) -> Vec<f16> {
    let mut output = vec![f16::ZERO; ids.len() * hidden_size];
    output.par_chunks_exact_mut(hidden_size).enumerate().for_each(|(i, out_row)| {
        let id = ids[i] as usize;
        let start = id * hidden_size;
        out_row.copy_from_slice(&table[start..start + hidden_size]);
    });
    output
}

pub fn native_linear_f16(input: &[f16], weight: &[f16], bias: Option<&[f16]>, m: usize, n: usize, k: usize) -> Vec<f16> {
    let mut output = vec![f16::ZERO; m * n];
    output.par_chunks_exact_mut(n).enumerate().for_each(|(i, row_out)| {
        let row_in = &input[i * k .. (i + 1) * k];
        for j in 0..n {
            let mut sum = 0.0f32;
            let weight_row = &weight[j * k .. (j + 1) * k];
            for l in 0..k {
                sum += row_in[l].to_f32() * weight_row[l].to_f32();
            }
            if let Some(b) = bias {
                sum += b[j].to_f32();
            }
            row_out[j] = f16::from_f32(sum);
        }
    });
    output
}

pub fn native_patch_embed_f16(
    input: &[f16],
    weight: &[f16],
    bias: &[f16],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f16> {
    let m = input.len() / in_dim;
    native_linear_f16(input, weight, Some(bias), m, out_dim, in_dim)
}

pub fn native_patch_merger_f16(
    input: &[f16],
    fc1_w: &[f16], fc1_b: &[f16],
    fc2_w: &[f16], fc2_b: &[f16],
    norm_w: &[f16], norm_b: &[f16],
    eps: f32,
    in_features: usize,
    out_features: usize,
) -> Vec<f16> {
    let m = input.len() / in_features;
    let x_norm = native_rms_norm_f16(input, norm_w, eps, in_features);
    let mut x = native_linear_f16(&x_norm, fc1_w, Some(fc1_b), m, in_features, in_features);
    x.par_iter_mut().for_each(|val| {
        let v = val.to_f32();
        *val = f16::from_f32(0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v.powi(3))).tanh()));
    });
    native_linear_f16(&x, fc2_w, Some(fc2_b), m, out_features, in_features)
}

pub fn native_add_pos_embed_f16(
    x: &mut [f16],
    pos_embed: &[f16],
    grid_thw: &[u32; 3],
    num_grid_per_side: u32,
    hidden_size: usize,
) {
    let num_pos = pos_embed.len() / hidden_size;
    x.par_chunks_exact_mut(hidden_size).enumerate().for_each(|(i, patch)| {
        let p_idx = i % num_pos;
        for d in 0..hidden_size {
            let val = patch[d].to_f32() + pos_embed[p_idx * hidden_size + d].to_f32();
            patch[d] = f16::from_f32(val);
        }
    });
}

pub fn native_vision_attn_f16(
    q: &[f16], k_full: &[f16], v_full: &[f16],
    hidden_size: usize, n_heads: usize,
    q_len: usize, total_seq_len: usize,
) -> Vec<f16> {
    let head_dim = hidden_size / n_heads;
    let mut output = vec![f16::ZERO; q_len * hidden_size];
    let scale = 1.0 / (head_dim as f32).sqrt();

    output.par_chunks_exact_mut(hidden_size).enumerate().for_each(|(i, out_patch)| {
        for h in 0..n_heads {
            let mut head_scores = vec![0.0f32; total_seq_len];
            let qi = &q[(i * n_heads + h) * head_dim .. (i * n_heads + h + 1) * head_dim];
            for j in 0..total_seq_len {
                let kj = &k_full[(j * n_heads + h) * head_dim .. (j * n_heads + h + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += qi[d].to_f32() * kj[d].to_f32(); }
                head_scores[j] = dot * scale;
            }
            let max_score = head_scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in head_scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for j in 0..total_seq_len {
                let vj = &v_full[(j * n_heads + h) * head_dim .. (j * n_heads + h + 1) * head_dim];
                let s = head_scores[j] * inv_sum;
                for d in 0..head_dim {
                    let val = out_patch[h * head_dim + d].to_f32() + s * vj[d].to_f32();
                    out_patch[h * head_dim + d] = f16::from_f32(val);
                }
            }
        }
    });
    output
}

// --- SIMD Accelerated Bit Operations ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_f16_to_u32_avx2(src: *const f16) -> u32 {
    // 1. Load 8 f16s (128-bit)
    let v_half = _mm_loadu_si128(src as *const __m128i);
    // 2. Convert to 8 f32s (256-bit) - Requires F16C instruction set (Standard with AVX2)
    let v_float = _mm256_cvtph_ps(v_half);
    // 3. Compare >= 0.0
    let v_cmp = _mm256_cmp_ps(v_float, _mm256_setzero_ps(), _CMP_GE_OQ);
    // 4. Movemask to get 8 bits
    _mm256_movemask_ps(v_cmp) as u32
}

pub fn bit_serial_matmul_f32(
    input: &[f16],
    weight_packed: &[u32],
    scales: &[f16], 
    m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    let k_blocks = k / 32;
    
    // [OPTIMIZATION] Parallelize over Output Rows (m), but compute blocks of columns (n)
    // to improve cache locality for Input.
    output.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let input_offset = i * k;
        let input_row = &input[input_offset .. input_offset + k];
        
        // 1. Pack Input Row (F16 -> U32 bits)
        // Using stack-allocated vector to avoid heap allocation overhead in loop
        let mut input_packed = vec![0u32; k_blocks];
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                let mut kb = 0;
                // Process 32 elements at a time (4 * 8-bit AVX packs)
                while kb < k_blocks {
                    unsafe {
                        let ptr = input_row.as_ptr().add(kb * 32);
                        // Unroll 4 AVX2 calls to fill one u32 (32 bits)
                        let b0 = pack_f16_to_u32_avx2(ptr);
                        let b1 = pack_f16_to_u32_avx2(ptr.add(8));
                        let b2 = pack_f16_to_u32_avx2(ptr.add(16));
                        let b3 = pack_f16_to_u32_avx2(ptr.add(24));
                        input_packed[kb] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    }
                    kb += 1;
                }
            } else {
                // Fallback for non-AVX2
                for kb in 0..k_blocks {
                    let mut bits = 0u32;
                    let start = kb * 32;
                    for b in 0..32 {
                        if input_row[start + b].to_f32() >= 0.0 { bits |= 1 << b; }
                    }
                    input_packed[kb] = bits;
                }
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // ARM / Others
            for kb in 0..k_blocks {
                let mut bits = 0u32;
                let start = kb * 32;
                for b in 0..32 {
                    if input_row[start + b].to_f32() >= 0.0 { bits |= 1 << b; }
                }
                input_packed[kb] = bits;
            }
        }

        // 2. Blocked Matrix Multiplication
        // Process columns in blocks of 4 to allow loop unrolling and better pipelining
        let mut j = 0;
        while j + 4 <= n {
            let mut dot0 = 0i32; let mut dot1 = 0i32;
            let mut dot2 = 0i32; let mut dot3 = 0i32;
            
            let idx0 = j * k_blocks;
            let w_ptr0 = &weight_packed[idx0 .. idx0 + k_blocks];
            let idx1 = (j+1) * k_blocks;
            let w_ptr1 = &weight_packed[idx1 .. idx1 + k_blocks];
            let idx2 = (j+2) * k_blocks;
            let w_ptr2 = &weight_packed[idx2 .. idx2 + k_blocks];
            let idx3 = (j+3) * k_blocks;
            let w_ptr3 = &weight_packed[idx3 .. idx3 + k_blocks];

            for kb in 0..k_blocks {
                let inp = input_packed[kb];
                let x0 = inp ^ w_ptr0[kb];
                let x1 = inp ^ w_ptr1[kb];
                let x2 = inp ^ w_ptr2[kb];
                let x3 = inp ^ w_ptr3[kb];
                
                dot0 += x0.count_ones() as i32;
                dot1 += x1.count_ones() as i32;
                dot2 += x2.count_ones() as i32;
                dot3 += x3.count_ones() as i32;
            }
            
            // Formula: dot = (32*k_blocks) - 2*popcnt(XOR)
            let total_bits = (k_blocks * 32) as i32;
            row_out[j]   = ((total_bits - 2 * dot0) as f32) * scales[j].to_f32();
            row_out[j+1] = ((total_bits - 2 * dot1) as f32) * scales[j+1].to_f32();
            row_out[j+2] = ((total_bits - 2 * dot2) as f32) * scales[j+2].to_f32();
            row_out[j+3] = ((total_bits - 2 * dot3) as f32) * scales[j+3].to_f32();
            
            j += 4;
        }

        // Handle remaining columns
        while j < n {
            let mut dot = 0i32;
            let idx = j * k_blocks;
            let w_ptr = &weight_packed[idx .. idx + k_blocks];
            for kb in 0..k_blocks {
                dot += (input_packed[kb] ^ w_ptr[kb]).count_ones() as i32;
            }
            let total_bits = (k_blocks * 32) as i32;
            row_out[j] = ((total_bits - 2 * dot) as f32) * scales[j].to_f32();
            j += 1;
        }
    });
    output
}
