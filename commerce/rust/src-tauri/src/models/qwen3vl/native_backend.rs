// --- CUDA Interface ---

#[cfg(feature = "cuda")]
extern "C" {
    fn bit_serial_matmul_kernel_wrapper(
        input: *const f32,
        weight: *const u32,
        scales: *const f32,
        output: *mut f32,
        m: i32, n: i32, k: i32
    );
}

pub fn bit_serial_matmul_gpu(
    input: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    m: usize, n: usize, k: usize
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    unsafe {
        let mut output = vec![0.0f32; m * n];
        bit_serial_matmul_kernel_wrapper(
            input.as_ptr(),
            weight_packed.as_ptr(),
            scales.as_ptr(),
            output.as_mut_ptr(),
            m as i32, n as i32, k as i32
        );
        output
    }
    #[cfg(not(feature = "cuda"))]
    {
        panic!("CUDA support not enabled");
    }
}
use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

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
    x.par_chunks_exact_mut(hidden_size).enumerate().for_each(|(i, patch)| {
        let p_idx = i % pos_embed.len();
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

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pack_f32_to_u32_avx2(src: *const f32) -> u32 {
    let zero = _mm256_setzero_ps();
    let m0 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(src), zero, _CMP_GE_OQ)) as u32;
    let m1 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(src.add(8)), zero, _CMP_GE_OQ)) as u32;
    let m2 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(src.add(16)), zero, _CMP_GE_OQ)) as u32;
    let m3 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(src.add(24)), zero, _CMP_GE_OQ)) as u32;
    m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
}

pub fn bit_serial_matmul_f32(
    input: &[f32],
    weight_packed: &[u32],
    scales: &[f16], 
    m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    let k_blocks = k / 32;
    output.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let input_ptr = unsafe { input.as_ptr().add(i * k) };
        let mut input_packed = vec![0u32; k_blocks];
        for kb in 0..k_blocks {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") { input_packed[kb] = unsafe { pack_f32_to_u32_avx2(input_ptr.add(kb * 32)) }; }
                else { let mut bits = 0u32; let row = unsafe { std::slice::from_raw_parts(input_ptr.add(kb * 32), 32) }; for b in 0..32 { if row[b] >= 0.0 { bits |= 1 << b; } } input_packed[kb] = bits; }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { let mut bits = 0u32; let row = unsafe { std::slice::from_raw_parts(input_ptr.add(kb * 32), 32) }; for b in 0..32 { if row[b] >= 0.0 { bits |= 1 << b; } } input_packed[kb] = bits; }
        }
        for j in 0..n {
            let mut total_dot = 0i32;
            let weight_row = &weight_packed[j * k_blocks .. (j + 1) * k_blocks];
            let scale = scales[j].to_f32();
            for kb in 0..k_blocks {
                let xor_val = input_packed[kb] ^ weight_row[kb];
                total_dot += 32 - 2 * (xor_val.count_ones() as i32);
            }
            row_out[j] = (total_dot as f32) * scale;
        }
    });
    output
}