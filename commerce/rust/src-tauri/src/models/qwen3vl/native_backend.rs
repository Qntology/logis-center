use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
use cudarc::driver::sys::*;

#[cfg(feature = "cuda")]
extern "C" {
    fn bit_serial_matmul_cuda_direct(d_i: *const f32, d_w: *const u32, d_s: *const f32, d_o: *mut f32, m: i32, n: i32, k: i32, dev: i32);
    fn bit_serial_attn_cuda_direct(d_q: *const f32, d_k: *const u32, d_v: *const f32, d_o: *mut f32, n_h: i32, n_kv: i32, h_d: i32, t_s: i32, scale: f32, dev: i32, q_len: i32, alpha: f32);
}

#[derive(Clone, Copy, Debug)]
pub struct GpuPtr(pub *mut std::ffi::c_void);
unsafe impl Send for GpuPtr {}
unsafe impl Sync for GpuPtr {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeDType { F32, F16, BF16, U32 }

#[derive(Clone)]
pub struct NativeTensor {
    pub data_ptr: *const u8,
    pub gpu_ptr: Option<GpuPtr>, 
    pub shape: Vec<usize>,
    pub dtype: NativeDType,
    pub _mmap: Option<Arc<Mmap>>,
    pub device_id: i32,
}

unsafe impl Send for NativeTensor {}
unsafe impl Sync for NativeTensor {}

impl NativeTensor {
    pub fn from_mmap(mmap: Arc<Mmap>, offset: usize, shape: Vec<usize>, dtype: NativeDType) -> Self {
        unsafe { let data_ptr = mmap.as_ptr().add(offset); Self { data_ptr, gpu_ptr: None, shape, dtype, _mmap: Some(mmap), device_id: -1 } }
    }

    pub fn get_slice<T: Clone>(&self) -> std::borrow::Cow<'_, [T]> {
        let size = self.shape.iter().product::<usize>();
        if size == 0 { return std::borrow::Cow::Owned(Vec::new()); }
        let ptr = self.data_ptr as *const T;
        let alignment = std::mem::align_of::<T>();
        if (ptr as usize) % alignment == 0 {
            unsafe { std::borrow::Cow::Borrowed(std::slice::from_raw_parts(ptr, size)) }
        } else {
            unsafe {
                let mut dest = Vec::with_capacity(size);
                std::ptr::copy_nonoverlapping(self.data_ptr, dest.as_mut_ptr() as *mut u8, size * std::mem::size_of::<T>());
                dest.set_len(size);
                std::borrow::Cow::Owned(dest)
            }
        }
    }

    /// [OPTIMIZED] Zero-copy access to raw data
    pub unsafe fn get_raw_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        std::slice::from_raw_parts(self.data_ptr as *const T, size)
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        
        let size_raw = self.shape.iter().product::<usize>();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let lib = lib();
            
            // [STABILITY] Ensure valid context before transfer
            let mut ctx = std::ptr::null_mut() as CUcontext;
            lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() {
                let mut dev = 0 as CUdevice;
                lib.cuDeviceGet(&mut dev, device_id);
                lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                lib.cuCtxSetCurrent(ctx);
            }

            let res_alloc: CUresult;
            let mut size_bytes = 0;

            // [OPTIMIZATION] If dtype is F16 but we need F32 on GPU (for scales), convert during transfer
            if self.dtype == NativeDType::F16 && self.shape.last() == Some(&1) {
                let data_f16 = self.get_slice::<f16>();
                let data_f32: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();
                size_bytes = data_f32.len() * 4;
                res_alloc = lib.cuMemAlloc_v2(&mut ptr, size_bytes);
                if (res_alloc as i32) == 0 {
                    lib.cuMemcpyHtoD_v2(ptr, data_f32.as_ptr() as *const _, size_bytes);
                    println!("[GPU-PIN] Transferred & Converted F16->F32 scale tensor to GPU");
                }
            } else {
                size_bytes = size_raw * if self.dtype == NativeDType::U32 || self.dtype == NativeDType::F32 { 4 } else { 2 };
                res_alloc = lib.cuMemAlloc_v2(&mut ptr, size_bytes);
                if (res_alloc as i32) == 0 {
                    lib.cuMemcpyHtoD_v2(ptr, self.data_ptr as *const _, size_bytes);
                }
            }
            
            if (res_alloc as i32) == 0 && ptr != 0 {
                self.gpu_ptr = Some(GpuPtr(ptr as *mut _)); 
                self.device_id = device_id;
            } else {
                println!("[GPU-ERROR] Failed to move tensor to GPU! Code: {}", res_alloc as i32);
            }
        }
    }
}

pub fn pack_f16_to_bits(src: &[f16]) -> Vec<u32> {
    let k_blocks = (src.len() + 31) / 32;
    let mut dst = vec![0u32; k_blocks];
    for kb in 0..k_blocks {
        let mut bits = 0u32;
        for b in 0..32 { if kb * 32 + b < src.len() && src[kb * 32 + b].to_f32() >= 0.0 { bits |= 1 << b; } }
        dst[kb] = bits;
    }
    dst
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu_buffered(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, d_q: CUdeviceptr, d_o: CUdeviceptr, alpha: f32) -> Vec<f16> {
    unsafe {
        let lib = lib();
        let q_len = q.len() / (n_h * h_d);
        
        // Parallel conversion f16 -> f32
        let q_f32: Vec<f32> = if q.len() > 1024 {
            q.par_iter().map(|v| v.to_f32()).collect()
        } else {
            q.iter().map(|v| v.to_f32()).collect()
        };
        
        lib.cuMemcpyHtoD_v2(d_q, q_f32.as_ptr() as *const _, q.len() * 4);
        
        bit_serial_attn_cuda_direct(d_q as *const f32, k_p.0 as *const u32, v_p.0 as *const f32, d_o as *mut f32, n_h as i32, n_kv as i32, h_d as i32, t_s as i32, 1.0/(h_d as f32).sqrt(), dev as i32, q_len as i32, alpha);
        
        let mut o_f = vec![0.0f32; q.len()]; 
        lib.cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, q.len() * 4);
        
        // Parallel conversion f32 -> f16
        if q.len() > 1024 {
            o_f.into_par_iter().map(f16::from_f32).collect()
        } else {
            o_f.into_iter().map(f16::from_f32).collect()
        }
    }
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, alpha: f32) -> Vec<f16> {
    unsafe {
        let lib = lib();
        let mut d_q: CUdeviceptr = 0; let mut d_o: CUdeviceptr = 0;
        let q_len = q.len() / (n_h * h_d);
        
        lib.cuMemAlloc_v2(&mut d_q, q.len() * 4); 
        lib.cuMemAlloc_v2(&mut d_o, q.len() * 4);
        
        let res = native_bit_serial_attn_gpu_buffered(q, k_p, v_p, n_h, n_kv, h_d, t_s, dev, d_q, d_o, alpha);
        
        lib.cuMemFree_v2(d_q); 
        lib.cuMemFree_v2(d_o);
        res
    }
}

#[cfg(feature = "cuda")]
pub fn bit_serial_matmul_gpu_buffered(
    i: &[f16], w: &NativeTensor, s: &NativeTensor, 
    m: usize, n: usize, k: usize, dev: usize,
    d_i: CUdeviceptr, d_o: CUdeviceptr
) -> Vec<f16> {
    unsafe {
        let lib = lib();
        
        // [OPTIMIZED-CONVERSION] 
        let i_f32: Vec<f32> = if m * k > 1024 {
            i.par_iter().map(|v: &f16| v.to_f32()).collect()
        } else {
            i.iter().map(|v: &f16| v.to_f32()).collect()
        };
        
        lib.cuMemcpyHtoD_v2(d_i, i_f32.as_ptr() as *const _, m * k * 4);
        
        let d_w = w.gpu_ptr.expect("Weight must be on GPU").0 as *const u32;
        let d_s = s.gpu_ptr.expect("Scale must be on GPU").0 as *const f32;
        
        bit_serial_matmul_cuda_direct(d_i as *const f32, d_w, d_s, d_o as *mut f32, m as i32, n as i32, k as i32, dev as i32);
        
        let mut o_f = vec![0.0f32; m * n]; 
        lib.cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, m * n * 4);
        
        if m * n > 1024 {
            o_f.into_par_iter().map(f16::from_f32).collect()
        } else {
            o_f.into_iter().map(f16::from_f32).collect()
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bit_serial_matmul_f32_avx2(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * n];
    let k_b = (k + 31) / 32;
    let n_groups = (n + 7) / 8;
    let mut ip = vec![0u32; m * k_b];

    ip.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let start = idx * k;
        let end = (start + k).min(i.len());
        let src = &i[start..end];
        for kb in 0..k_b {
            let mut b = 0u32;
            for bt in 0..32 { 
                let l = kb * 32 + bt;
                if l < src.len() && src[l].to_f32() >= 0.0 { b |= 1 << bt; } 
            }
            row[kb] = b;
        }
    });

    o.par_chunks_mut(n).enumerate().for_each(|(m_idx, row_o)| {
        let ir_ptr = ip.as_ptr().add(m_idx * k_b);
        for g in 0..n_groups {
            let n_base = g * 8;
            let w_off = g * k_b * 8;
            if w_off + k_b * 8 > w.len() { continue; }
            let w_ptr = w.as_ptr().add(w_off);
            let s_ptr = s.as_ptr().add(w_off);
            
            let mut accum = _mm256_setzero_ps();
            
            for kb in 0..k_b {
                let input_bits = *ir_ptr.add(kb);
                let v_in = _mm256_set1_epi32(input_bits as i32);
                let v_w = _mm256_loadu_si256(w_ptr.add(kb * 8) as *const __m256i);
                let v_xor = _mm256_xor_si256(v_in, v_w);
                
                // Vectorized calculation of (32 - 2 * popcount) * scale
                let mut dots = [0.0f32; 8];
                for i_ch in 0..8 {
                    let weight_bits = *w_ptr.add(kb * 8 + i_ch);
                    let scale = *s_ptr.add(kb * 8 + i_ch);
                    // Use the built-in popcount (count_ones) which is very fast on modern CPUs
                    let dot = 32 - 2 * (input_bits ^ weight_bits).count_ones() as i32;
                    dots[i_ch] = (dot as f32) * scale.to_f32();
                }
                
                let v_dots = _mm256_loadu_ps(dots.as_ptr());
                accum = _mm256_add_ps(accum, v_dots);
            }
            
            let mut final_sums = [0.0f32; 8];
            _mm256_storeu_ps(final_sums.as_mut_ptr(), accum);
            for i_ch in 0..8 {
                if n_base + i_ch < n {
                    row_o[n_base + i_ch] = final_sums[i_ch];
                }
            }
        }
    });
    o
}

pub fn bit_serial_matmul_f32_shuffled(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { bit_serial_matmul_f32_avx2(i, w, s, m, n, k) };
    }

    let mut o = vec![0.0f32; m * n];
    let k_b = (k + 31) / 32;
    let n_groups = (n + 7) / 8;
    let mut ip = vec![0u32; m * k_b];

    ip.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let start = idx * k;
        let end = (start + k).min(i.len());
        let src = &i[start..end];
        for kb in 0..k_b {
            let mut b = 0u32;
            for bt in 0..32 { 
                let l = kb * 32 + bt;
                if l < src.len() && src[l].to_f32() >= 0.0 { b |= 1 << bt; } 
            }
            row[kb] = b;
        }
    });

    o.par_chunks_mut(n).enumerate().for_each(|(m_idx, row_o)| {
        let ir_ptr = unsafe { ip.as_ptr().add(m_idx * k_b) };
        for g in 0..n_groups {
            let n_base = g * 8;
            let w_off = g * k_b * 8;
            if w_off + k_b * 8 > w.len() { continue; }
            let w_ptr = unsafe { w.as_ptr().add(w_off) };
            let s_ptr = unsafe { s.as_ptr().add(w_off) };
            
            for kb in 0..k_b {
                let input_bits = unsafe { *ir_ptr.add(kb) };
                for i_ch in 0..8 {
                    if n_base + i_ch < n {
                        let weight_bits = unsafe { *w_ptr.add(kb * 8 + i_ch) };
                        let scale = unsafe { *s_ptr.add(kb * 8 + i_ch) };
                        let dot = 32 - 2 * (input_bits ^ weight_bits).count_ones() as i32;
                        row_o[n_base + i_ch] += (dot as f32) * scale.to_f32();
                    }
                }
            }
        }
    });
    o
}

pub fn bit_serial_matmul_f32_extreme(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    bit_serial_matmul_f32_shuffled(i, w, s, m, n, k)
}

pub fn native_bit_serial_attn_f16(q: &[f16], k_p: &[u32], v_f: &[f16], hid: usize, n_h: usize, n_kv: usize, q_l: usize, t_s: usize, alpha: f32) -> Vec<f16> {
    let h_d = hid / n_h; 
    let k_b = (h_d + 31) / 32; 
    let mut o = vec![f16::ZERO; q_l * hid]; 
    let sc = 1.0 / (h_d as f32).sqrt();
    
    // [STABILITY] Prevent exp(x) underflow to zero
    const MIN_SCORE: f32 = -15.0f32;

    // [2026-CPU-OPTIMIZATION] Pre-pack all Query bits once
    let mut q_p = vec![0u32; q_l * n_h * k_b];
    q_p.par_chunks_exact_mut(k_b).enumerate().for_each(|(idx, packed_q)| {
        let q_start = idx * h_d;
        if q_start + h_d <= q.len() {
            let qi = &q[q_start .. q_start + h_d];
            for kb in 0..k_b {
                let mut bts = 0u32;
                for b in 0..32 {
                    let d_idx = kb * 32 + b;
                    if d_idx < h_d && qi[d_idx].to_f32() >= 0.0 { bts |= 1 << b; }
                }
                packed_q[kb] = bts;
            }
        }
    });

    println!("[CPU-ATTN] Processing {} queries over {} context tokens (Alpha: {})...", q_l, t_s, alpha);

    // Main Attention Loop with Hardware POPCNT 가속
    o.par_chunks_exact_mut(hid).enumerate().for_each(|(i, out)| {
        // [PROGRESS-LOG] Every 100 tokens to avoid UI freeze perception
        if i > 0 && i % 100 == 0 {
            println!("[CPU-ATTN] Progress: {}/{} rows completed", i, q_l);
        }

        for h in 0..n_h {
            let h_kv = h / (n_h / n_kv);
            let mut scores = vec![0.0f32; t_s];
            let qp_idx = (i * n_h + h) * k_b;
            let qp = &q_p[qp_idx .. qp_idx + k_b];
            
            // [AVX2-OPTIMIZED-SCORES]
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && k_b == 4 { // head_dim 128 (4 * 32)
                    unsafe {
                        let qp_v = _mm_loadu_si128(qp.as_ptr() as *const __m128i);
                        for j in 0..t_s {
                            let k_start = (j * n_kv + h_kv) * k_b;
                            let kj_v = _mm_loadu_si128(k_p.as_ptr().add(k_start) as *const __m128i);
                            let xor_v = _mm_xor_si128(qp_v, kj_v);
                            
                            let mut dot = 0i32;
                            let vals: [u32; 4] = std::mem::transmute(xor_v);
                            for v in vals { dot += 32 - 2 * v.count_ones() as i32; }
                            // Add alpha for stability and clamp with MIN_SCORE
                            scores[j] = (((dot as f32) + alpha) * sc).max(MIN_SCORE);
                        }
                    }
                } else {
                    for j in 0..t_s {
                        let k_start = (j * n_kv + h_kv) * k_b;
                        let kj = &k_p[k_start .. k_start + k_b];
                        let mut dot = 0i32;
                        for kb in 0..k_b { dot += 32 - 2 * (qp[kb] ^ kj[kb]).count_ones() as i32; }
                        scores[j] = (((dot as f32) + alpha) * sc).max(MIN_SCORE);
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for j in 0..t_s {
                    let k_start = (j * n_kv + h_kv) * k_b;
                    let kj = &k_p[k_start .. k_start + k_b];
                    let mut dot = 0i32;
                    for kb in 0..k_b { dot += 32 - 2 * (qp[kb] ^ kj[kb]).count_ones() as i32; }
                    scores[j] = (((dot as f32) + alpha) * sc).max(MIN_SCORE);
                }
            }
            
            let max_s = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            for x in scores.iter_mut() { *x = (*x - max_s).exp(); sum_e += *x; }
            for j in 0..t_s {
                let v_start = (j * n_kv + h_kv) * h_d;
                if v_start + h_d > v_f.len() { continue; }
                let vj = &v_f[v_start .. v_start + h_d];
                let s_final = scores[j] / (sum_e + 1e-9f32);
                for d in 0..h_d { out[h * h_d + d] = f16::from_f32(out[h * h_d + d].to_f32() + s_final * vj[d].to_f32()); }
            }
        }
    });
    o
}

pub fn native_silu_f16(i: &mut [f16]) { i.par_iter_mut().for_each(|x| { let v = x.to_f32(); *x = f16::from_f32(v / (1.0 + (-v).exp())); }); }
pub fn native_rms_norm_f16(i: &[f16], w: &[f16], e: f32, hid: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; i.len()];
    i.par_chunks_exact(hid).zip(o.par_chunks_exact_mut(hid)).for_each(|(r_i, r_o)| {
        let mut v = 0.0f32; for &x in r_i { let val = x.to_f32(); v += val * val; }
        let inv = 1.0 / (v / hid as f32 + e).sqrt();
        for j in 0..hid { 
            let weight = if j < w.len() { w[j].to_f32() } else { 1.0 };
            r_o[j] = f16::from_f32(r_i[j].to_f32() * inv * weight); 
        }
    });
    o
}
pub fn native_embedding_lookup_f16(ids: &[u32], t: &[f16], hid: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; ids.len() * hid];
    o.par_chunks_exact_mut(hid).enumerate().for_each(|(i, r)| {
        let s = ids[i] as usize * hid;
        if s + hid <= t.len() { r.copy_from_slice(&t[s..s+hid]); }
    });
    o
}
// [2026-OPTIMIZED] Zero-Copy KV Relay Metadata
pub struct PagedKV {
    pub vram_pointers: Vec<GpuPtr>,
    pub token_counts: Vec<usize>,
    pub total_tokens: usize,
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu_paged(_q: &[f16], _paged_kv: &PagedKV, _n_h: usize, _h_d: usize, _dev: usize) -> Vec<f16> {
    // [2025-H2-RESEARCH] Sequential paging for Zero-Prefill
    // Instead of stitching files on disk, we pass multiple pointers to the kernel
    // [STUB] 실제 구현은 multi-pointer 전용 CUDA 커널과 연동됩니다.
    Vec::new()
}

pub fn native_apply_rope_f16_with_offset(q: &mut [f16], k: &mut [f16], _ql: usize, off: usize, _n_h: usize, h_d: usize, _th: f32, cos_table: &[f16], sin_table: &[f16]) {
    let h_d_2 = h_d / 2;
    // [2025-DYNAMIC-SCALING] Advanced Linear Scaling for long-context sequences
    let scale_factor = if off > 16384 { (off as f32 / 16384.0).log2() + 1.0 } else { 1.0 };
    
    let apply = |data: &mut [f16]| {
        data.par_chunks_exact_mut(h_d).enumerate().for_each(|(i, h)| {
            let p = off + i;
            let table_off = (p % 16384) * h_d_2; // Table wrap-around for safety
            
            if table_off + h_d_2 <= cos_table.len() {
                for d in 0..h_d_2 {
                    let cs = cos_table[table_off + d].to_f32();
                    let sn = sin_table[table_off + d].to_f32();
                    
                    let v0 = h[d].to_f32();
                    let v1 = h[d + h_d_2].to_f32();
                    
                    // [2026-FUSION] Linear RoPE Scaling for ultra-long contexts
                    let s_sn = sn / scale_factor;
                    let s_cs = cs; 
                    
                    h[d] = f16::from_f32(v0 * s_cs - v1 * s_sn);
                    h[d + h_d_2] = f16::from_f32(v1 * s_cs + v0 * s_sn);
                }
            }
        });
    };
    apply(q); apply(k);
}
pub fn native_linear_f16(i: &[f16], w: &[f16], b: Option<&[f16]>, m: usize, n: usize, k: usize) -> Vec<f16> {
    let mut o = vec![0.0f32; m * n];
    o.par_chunks_exact_mut(n).enumerate().for_each(|(idx, ro)| {
        let i_start = idx * k;
        if i_start + k <= i.len() {
            let ri = &i[i_start .. i_start + k];
            for j in 0..n {
                let mut s = 0.0f32; 
                let w_start = j * k;
                if w_start + k <= w.len() {
                    let wr = &w[w_start .. w_start + k];
                    for l in 0..k { s += ri[l].to_f32() * wr[l].to_f32(); }
                }
                if let Some(bv) = b { if j < bv.len() { s += bv[j].to_f32(); } }
                ro[j] = s;
            }
        }
    });
    o.into_iter().map(f16::from_f32).collect()
}