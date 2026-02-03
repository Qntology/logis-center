use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
extern "C" {
    fn bit_serial_matmul_cuda_direct(
        d_input: *const f32,
        d_weight: *const u32,
        d_scales: *const f32,
        d_output: *mut f32,
        m: i32, n: i32, k: i32,
        device_id: i32
    );
}

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
    pub gpu_ptr: Option<*mut std::ffi::c_void>, // GPU Pointer if allocated
    pub shape: Vec<usize>,
    pub dtype: NativeDType,
    pub _mmap: Option<Arc<Mmap>>,
    pub device_id: i32,
}

unsafe impl Send for NativeTensor {}
unsafe impl Sync for NativeTensor {}

impl NativeTensor {
    pub fn from_mmap(mmap: Arc<Mmap>, offset: usize, shape: Vec<usize>, dtype: NativeDType) -> Self {
        unsafe {
            let data_ptr = mmap.as_ptr().add(offset);
            Self {
                data_ptr,
                gpu_ptr: None,
                shape,
                dtype,
                _mmap: Some(mmap),
                device_id: -1,
            }
        }
    }

    pub fn get_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        unsafe { std::slice::from_raw_parts(self.data_ptr as *const T, size) }
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        let size_bytes = self.shape.iter().product::<usize>() * match self.dtype {
            NativeDType::F32 | NativeDType::U32 => 4,
            _ => 2,
        };
        
        unsafe {
            use cudarc::driver::sys::*;
            let mut ptr: CUdeviceptr = 0;
            cuInit(0);
            cuMemAlloc_v2(&mut ptr, size_bytes);
            cuMemcpyHtoD_v2(ptr, self.data_ptr as *const std::ffi::c_void, size_bytes);
            self.gpu_ptr = Some(ptr as *mut std::ffi::c_void);
            self.device_id = device_id;
        }
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
            if j >= weight.len() { break; } 
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

// --- SIMD Accelerated Bit Operations ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_f16_to_u32_avx2(src: *const f16) -> u32 {
    let v_half = _mm_loadu_si128(src as *const __m128i);
    let v_float = _mm256_cvtph_ps(v_half);
    let v_cmp = _mm256_cmp_ps(v_float, _mm256_setzero_ps(), _CMP_GE_OQ);
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
    output.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let input_offset = i * k;
        let input_row = &input[input_offset .. input_offset + k];
        let mut input_packed = vec![0u32; k_blocks];
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                for kb in 0..k_blocks {
                    unsafe {
                        let ptr = input_row.as_ptr().add(kb * 32);
                        let b0 = pack_f16_to_u32_avx2(ptr);
                        let b1 = pack_f16_to_u32_avx2(ptr.add(8));
                        let b2 = pack_f16_to_u32_avx2(ptr.add(16));
                        let b3 = pack_f16_to_u32_avx2(ptr.add(24));
                        input_packed[kb] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    }
                }
            } else {
                for kb in 0..k_blocks {
                    let mut bits = 0u32; let start = kb * 32;
                    for b in 0..32 { if input_row[start + b].to_f32() >= 0.0 { bits |= 1 << b; } }
                    input_packed[kb] = bits;
                }
            }
        }
        
        let mut j = 0;
        while j + 4 <= n {
            let mut dot0 = 0i32; let mut dot1 = 0i32; let mut dot2 = 0i32; let mut dot3 = 0i32;
            let idx0 = j * k_blocks; let w_ptr0 = &weight_packed[idx0 .. idx0 + k_blocks];
            let idx1 = (j+1) * k_blocks; let w_ptr1 = &weight_packed[idx1 .. idx1 + k_blocks];
            let idx2 = (j+2) * k_blocks; let w_ptr2 = &weight_packed[idx2 .. idx2 + k_blocks];
            let idx3 = (j+3) * k_blocks; let w_ptr3 = &weight_packed[idx3 .. idx3 + k_blocks];
            for kb in 0..k_blocks {
                let inp = input_packed[kb];
                dot0 += (inp ^ w_ptr0[kb]).count_ones() as i32;
                dot1 += (inp ^ w_ptr1[kb]).count_ones() as i32;
                dot2 += (inp ^ w_ptr2[kb]).count_ones() as i32;
                dot3 += (inp ^ w_ptr3[kb]).count_ones() as i32;
            }
            let total_bits = (k_blocks * 32) as i32;
            row_out[j]   = ((total_bits - 2 * dot0) as f32) * scales[j].to_f32();
            row_out[j+1] = ((total_bits - 2 * dot1) as f32) * scales[j+1].to_f32();
            row_out[j+2] = ((total_bits - 2 * dot2) as f32) * scales[j+2].to_f32();
            row_out[j+3] = ((total_bits - 2 * dot3) as f32) * scales[j+3].to_f32();
            j += 4;
        }
        while j < n {
            let mut dot = 0i32; let idx = j * k_blocks; let w_ptr = &weight_packed[idx .. idx + k_blocks];
            for kb in 0..k_blocks { dot += (input_packed[kb] ^ w_ptr[kb]).count_ones() as i32; }
            let total_bits = (k_blocks * 32) as i32;
            row_out[j] = ((total_bits - 2 * dot) as f32) * scales[j].to_f32();
            j += 1;
        }
    });
    output
}

#[cfg(feature = "cuda")]
pub fn bit_serial_matmul_gpu(
    input: &[f16],
    weight_packed: &NativeTensor,
    scales: &NativeTensor,
    m: usize, n: usize, k: usize,
    device_id: usize
) -> Vec<f16> {
    unsafe {
        use cudarc::driver::sys::*;
        let m_i32 = m as i32; let n_i32 = n as i32; let k_i32 = k as i32;
        
        // 1. Prepare Input/Output on GPU
        let mut d_input: CUdeviceptr = 0;
        let mut d_output: CUdeviceptr = 0;
        let mut d_scales_f32: CUdeviceptr = 0;
        
        let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        cuMemAlloc_v2(&mut d_input, m * k * 4);
        cuMemcpyHtoD_v2(d_input, input_f32.as_ptr() as *const _, m * k * 4);
        
        cuMemAlloc_v2(&mut d_output, m * n * 4);
        
        // Convert f16 scales to f32 for CUDA kernel
        let scales_f16 = scales.get_slice::<f16>();
        let scales_f32: Vec<f32> = scales_f16.iter().map(|v| v.to_f32()).collect();
        cuMemAlloc_v2(&mut d_scales_f32, n * 4);
        cuMemcpyHtoD_v2(d_scales_f32, scales_f32.as_ptr() as *const _, n * 4);

        // 2. Launch Direct Kernel
        bit_serial_matmul_cuda_direct(
            d_input as *const f32,
            weight_packed.gpu_ptr.unwrap() as *const u32,
            d_scales_f32 as *const f32,
            d_output as *mut f32,
            m_i32, n_i32, k_i32,
            device_id as i32
        );

        // 3. Read back
        let mut output_f32 = vec![0.0f32; m * n];
        cuMemcpyDtoH_v2(output_f32.as_mut_ptr() as *mut _, d_output, m * n * 4);
        
        cuMemFree_v2(d_input);
        cuMemFree_v2(d_output);
        cuMemFree_v2(d_scales_f32);

        output_f32.into_iter().map(f16::from_f32).collect()
    }
}