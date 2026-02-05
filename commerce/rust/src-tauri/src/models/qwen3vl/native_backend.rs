use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::{lib, CUdeviceptr, CUcontext, CUdevice, CUresult};

pub static GPU_PANIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn is_gpu_panicked() -> bool {
    GPU_PANIC.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn trigger_gpu_panic() {
    if !is_gpu_panicked() {
        println!("[GPU-PANIC] Critical error detected. Disabling GPU for all future operations.");
        GPU_PANIC.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

// --- [CRITICAL] EXTERN C DECLARATIONS WITH LINK_NAME ---
// This ensures that the linker looks for the exact C name regardless of Rust module hierarchy.
#[cfg(feature = "cuda")]
extern "C" {
    #[link_name = "bit_serial_matmul_cuda_direct"]
    fn cuda_matmul_f32(d_i: *const f32, d_w: *const u32, d_s: *const f32, d_o: *mut f32, m: i32, n: i32, k: i32, dev: i32);
    
    #[link_name = "bit_serial_matmul_cuda_f16"]
    fn cuda_matmul_f16(d_i: *const f16, d_w: *const u32, d_s: *const f16, d_o: *mut f16, m: i32, n: i32, k: i32, dev: i32);
    
    #[link_name = "bit_serial_attn_cuda_direct"]
    fn cuda_attn_f32(d_q: *const f32, d_k: *const u32, d_v: *const f32, d_o: *mut f32, n_h: i32, n_kv: i32, h_d: i32, t_s: i32, scale: f32, dev: i32, q_len: i32, alpha: f32);
    
    #[link_name = "bit_serial_attn_cuda_f16"]
    fn cuda_attn_f16(d_q: *const f16, d_k: *const u32, d_v: *const f16, d_o: *mut f16, n_h: i32, n_kv: i32, h_d: i32, t_s: i32, scale: f32, dev: i32, q_len: i32, alpha: f32);

    #[link_name = "standard_matmul_cuda_f16"]
    pub fn standard_matmul_cuda_f16(d_i: *const f16, d_w: *const f16, d_o: *mut f16, m: i32, n: i32, k: i32);
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu_buffered(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, d_q: CUdeviceptr, d_o: CUdeviceptr, alpha: f32) -> Vec<f16> {
    if d_q == 0 || d_o == 0 || k_p.0.is_null() || v_p.0.is_null() {
        return vec![f16::ZERO; q.len()];
    }
    unsafe {
        let q_len = q.len() / (n_h * h_d);
        // Upload original f16 data directly
        lib().cuMemcpyHtoD_v2(d_q, q.as_ptr() as *const _, q.len() * 2);
        
        // Call via link_name mapped function
        cuda_attn_f16(d_q as *const f16, k_p.0 as *const u32, v_p.0 as *const f16, d_o as *mut f16, n_h as i32, n_kv as i32, h_d as i32, t_s as i32, 1.0/(h_d as f32).sqrt(), dev as i32, q_len as i32, alpha);
        
        let mut o_f = vec![f16::ZERO; q.len()]; 
        lib().cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, q.len() * 2);
        o_f
    }
}

#[cfg(feature = "cuda")]
pub fn bit_serial_matmul_gpu_buffered(
    i: &[f16], w: &NativeTensor, s: &NativeTensor, 
    m: usize, n: usize, k: usize, dev: usize,
    d_i: CUdeviceptr, d_o: CUdeviceptr
) -> Vec<f16> {
    if d_i == 0 || d_o == 0 {
        return vec![f16::ZERO; m * n];
    }
    unsafe {
        // Upload original f16 data directly
        let res_cp = lib().cuMemcpyHtoD_v2(d_i, i.as_ptr() as *const _, m * k * 2);
        if (res_cp as i32) != 0 {
            let wp_ref = w.get_raw_slice::<u32>(); 
            let s_ref = s.get_raw_slice::<f16>();
            let out = bit_serial_matmul_f32_shuffled(i, wp_ref, s_ref, m, n, k);
            return out.into_iter().map(f16::from_f32).collect();
        }
        
        let d_w = w.gpu_ptr.expect("Weight must be on GPU").0 as *const u32;
        let d_s = s.gpu_ptr.expect("Scale must be on GPU").0 as *const f16;
        
        // Call via link_name mapped function
        cuda_matmul_f16(d_i as *const f16, d_w, d_s, d_o as *mut f16, m as i32, n as i32, k as i32, dev as i32);
        
        let mut o_f = vec![f16::ZERO; m * n]; 
        lib().cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, m * n * 2);
        o_f
    }
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

    pub unsafe fn get_raw_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        std::slice::from_raw_parts(self.data_ptr as *const T, size)
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu_f16(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        unsafe {
            let cuda_lib = lib();
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }
            let data_f16 = self.get_slice::<f16>();
            let size_bytes = data_f16.len() * 2;
            let mut ptr: CUdeviceptr = 0;
            let res = cuda_lib.cuMemAlloc_v2(&mut ptr, size_bytes);
            if (res as i32) == 0 && ptr != 0 {
                let res_cp = cuda_lib.cuMemcpyHtoD_v2(ptr, data_f16.as_ptr() as *const _, size_bytes);
                if (res_cp as i32) == 0 {
                    self.gpu_ptr = Some(GpuPtr(ptr as *mut _));
                    self.device_id = device_id;
                } else {
                    cuda_lib.cuMemFree_v2(ptr);
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu_forced_f32(&mut self, device_id: i32) {
        self.move_to_gpu_f16(device_id);
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        let size_raw = self.shape.iter().product::<usize>();
        unsafe {
            let cuda_lib = lib();
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }
            let mut ptr: CUdeviceptr = 0;
            let size_bytes = size_raw * if self.dtype == NativeDType::U32 || self.dtype == NativeDType::F32 { 4 } else { 2 };
            let res_alloc = cuda_lib.cuMemAlloc_v2(&mut ptr, size_bytes);
            if (res_alloc as i32) == 0 && ptr != 0 {
                let res_cp = cuda_lib.cuMemcpyHtoD_v2(ptr, self.data_ptr as *const _, size_bytes);
                if (res_cp as i32) == 0 {
                    self.gpu_ptr = Some(GpuPtr(ptr as *mut _)); 
                    self.device_id = device_id;
                } else {
                    cuda_lib.cuMemFree_v2(ptr);
                }
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
pub fn native_bit_serial_attn_gpu(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, alpha: f32) -> Vec<f16> {
    unsafe {
        let mut d_q: CUdeviceptr = 0; let mut d_o: CUdeviceptr = 0;
        lib().cuMemAlloc_v2(&mut d_q, q.len() * 2); 
        lib().cuMemAlloc_v2(&mut d_o, q.len() * 2);
        let res = native_bit_serial_attn_gpu_buffered(q, k_p, v_p, n_h, n_kv, h_d, t_s, dev, d_q, d_o, alpha);
        lib().cuMemFree_v2(d_q); lib().cuMemFree_v2(d_o);
        res
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
                let mut dots = [0.0f32; 8];
                for i_ch in 0..8 {
                    let weight_bits = *w_ptr.add(kb * 8 + i_ch);
                    let scale = *s_ptr.add(kb * 8 + i_ch);
                    let dot = 32 - 2 * (input_bits ^ weight_bits).count_ones() as i32;
                    dots[i_ch] = (dot as f32) * scale.to_f32();
                }
                let v_dots = _mm256_loadu_ps(dots.as_ptr());
                accum = _mm256_add_ps(accum, v_dots);
            }
            let mut final_sums = [0.0f32; 8];
            _mm256_storeu_ps(final_sums.as_mut_ptr(), accum);
            for i_ch in 0..8 { if n_base + i_ch < n { row_o[n_base + i_ch] = final_sums[i_ch]; } }
        }
    });
    o
}

pub fn bit_serial_matmul_f32_shuffled(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") { return unsafe { bit_serial_matmul_f32_avx2(i, w, s, m, n, k) }; }
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
    let h_kv_ratio = n_h / n_kv;
    let mut q_bits = vec![0u32; q_l * n_h * k_b];
    q_bits.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let h_idx = idx % n_h;
        let q_idx = idx / n_h;
        let start = (q_idx * n_h * h_d) + (h_idx * h_d);
        let src = &q[start..start + h_d];
        for kb in 0..k_b {
            let mut b = 0u32;
            for bt in 0..32 { if kb * 32 + bt < h_d && src[kb * 32 + bt].to_f32() >= 0.0 { b |= 1 << bt; } }
            row[kb] = b;
        }
    });
    let mut o = vec![f16::ZERO; q_l * hid];
    o.par_chunks_mut(hid).enumerate().for_each(|(q_idx, row_o)| {
        for h in 0..n_h {
            let h_kv = h / h_kv_ratio;
            let q_b_ptr = unsafe { q_bits.as_ptr().add((q_idx * n_h + h) * k_b) };
            let mut running_max = -1e20f32;
            let mut running_sum = 0.0f32;
            let mut local_o = vec![0.0f32; h_d];
            for j in 0..t_s {
                let mut dot = 0;
                let k_p_ptr = unsafe { k_p.as_ptr().add((j * n_kv + h_kv) * k_b) };
                for kb in 0..k_b { dot += 32 - 2 * (unsafe { *q_b_ptr.add(kb) ^ *k_p_ptr.add(kb) }).count_ones() as i32; }
                let score = ((dot as f32) + alpha) * (1.0 / (h_d as f32).sqrt());
                let n_max = running_max.max(score);
                let e_scale = (running_max - n_max).exp();
                let e_score = (score - n_max).exp();
                running_sum = running_sum * e_scale + e_score;
                running_max = n_max;
                let v_ptr = unsafe { v_f.as_ptr().add((j * n_kv + h_kv) * h_d) };
                for d in 0..h_d { local_o[d] = local_o[d] * e_scale + e_score * unsafe { *v_ptr.add(d) }.to_f32(); }
            }
            for d in 0..h_d { row_o[h * h_d + d] = f16::from_f32(local_o[d] / (running_sum + 1e-9)); }
        }
    });
    o
}

pub fn native_rms_norm_f16(x: &[f16], w: &[f16], eps: f32, hid: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; x.len()];
    out.par_chunks_mut(hid).enumerate().for_each(|(i, row)| {
        let start = i * hid;
        let mut v = 0.0f32;
        for j in 0..hid { let val = x[start + j].to_f32(); v += val * val; }
        let inv = 1.0 / (v / hid as f32 + eps).sqrt();
        for j in 0..hid { row[j] = f16::from_f32(x[start + j].to_f32() * inv * w[j].to_f32()); }
    });
    out
}

pub fn native_silu_f16(x: &mut [f16]) {
    x.par_iter_mut().for_each(|val| {
        let f = val.to_f32();
        *val = f16::from_f32(f * (1.0 / (1.0 + (-f).exp())));
    });
}

pub fn native_silu_mul_f16(gate: &[f16], up: &[f16]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; gate.len()];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let g = gate[i].to_f32();
        let u = up[i].to_f32();
        *val = f16::from_f32((g * (1.0 / (1.0 + (-g).exp()))) * u);
    });
    out
}

pub fn native_embedding_lookup_f16(ids: &[u32], table: &[f16], hid: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; ids.len() * hid];
    for (i, &id) in ids.iter().enumerate() {
        let start = id as usize * hid;
        out[i * hid..(i + 1) * hid].copy_from_slice(&table[start..start + hid]);
    }
    out
}

pub fn native_linear_f16(x: &[f16], w: &[f16], b: Option<&[f16]>, m: usize, out_f: usize, in_f: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; m * out_f];
    out.par_chunks_mut(out_f).enumerate().for_each(|(i, row)| {
        let x_row = &x[i * in_f..(i + 1) * in_f];
        for j in 0..out_f {
            let mut acc = 0.0f32;
            let w_row = &w[j * in_f..(j + 1) * in_f];
            for k in 0..in_f { acc += x_row[k].to_f32() * w_row[k].to_f32(); }
            if let Some(bias) = b { acc += bias[j].to_f32(); }
            row[j] = f16::from_f32(acc);
        }
    });
    out
}

pub fn native_precompute_rope_f16(h_d: usize, max_len: usize, theta: f32) -> (Vec<f16>, Vec<f16>) {
    let mut cos = Vec::with_capacity(max_len * (h_d / 2));
    let mut sin = Vec::with_capacity(max_len * (h_d / 2));
    for p in 0..max_len {
        for d in 0..(h_d / 2) {
            let freq = 1.0 / theta.powf((2.0 * d as f32) / (h_d as f32));
            let (sn, cs) = ((p as f32) * freq).sin_cos();
            cos.push(f16::from_f32(cs)); sin.push(f16::from_f32(sn));
        }
    }
    (cos, sin)
}

pub fn native_apply_rope_f16_with_offset(q: &mut [f16], k: &mut [f16], q_l: usize, s_o: usize, n_h: usize, h_d: usize, _theta: f32, rope_cos: &[f16], rope_sin: &[f16]) {
    q.par_chunks_mut(h_d).enumerate().for_each(|(idx, q_head)| {
        let q_idx = idx / n_h;
        let p = q_idx + s_o;
        if p < 16384 {
            let cos = &rope_cos[p * (h_d / 2)..(p + 1) * (h_d / 2)];
            let sin = &rope_sin[p * (h_d / 2)..(p + 1) * (h_d / 2)];
            for i in 0..(h_d / 2) {
                let q0 = q_head[i].to_f32(); let q1 = q_head[i + h_d / 2].to_f32();
                let c = cos[i].to_f32(); let s = sin[i].to_f32();
                q_head[i] = f16::from_f32(q0 * c - q1 * s);
                q_head[i + h_d / 2] = f16::from_f32(q0 * s + q1 * c);
            }
        }
    });
    let n_kv = if q_l > 0 { k.len() / (q_l * h_d) } else { 0 };
    if n_kv > 0 {
        k.par_chunks_mut(h_d).enumerate().for_each(|(idx, k_head)| {
            let k_idx = idx / n_kv;
            let p = k_idx + s_o;
            if p < 16384 {
                let cos = &rope_cos[p * (h_d / 2)..(p + 1) * (h_d / 2)];
                let sin = &rope_sin[p * (h_d / 2)..(p + 1) * (h_d / 2)];
                for i in 0..(h_d / 2) {
                    let k0 = k_head[i].to_f32(); let k1 = k_head[i + h_d / 2].to_f32();
                    let c = cos[i].to_f32(); let s = sin[i].to_f32();
                    k_head[i] = f16::from_f32(k0 * c - k1 * s);
                    k_head[i + h_d / 2] = f16::from_f32(k0 * s + k1 * c);
                }
            }
        });
    }
}