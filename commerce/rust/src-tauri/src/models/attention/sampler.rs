use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Sampler {
    token_pos: AtomicU64,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            token_pos: AtomicU64::new(0),
        }
    }

    fn next_token_pos(&self) -> u64 {
        let current = self.token_pos.fetch_add(1, Ordering::Relaxed);
        if current >= u32::MAX as u64 {
            self.token_pos.store(0, Ordering::Relaxed);
            0
        } else {
            current
        }
    }

    #[cfg(feature = "cuda")]
    pub fn sample_cuda(
        &self,
        logits: &Tensor,
        k: usize,
        p: f32,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let token_pos = self.next_token_pos();
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, result};
        use candle_core::cuda_backend::CudaStorageSlice;
        use crate::models::attention::cuda_utils::{WrapErr, get_raw_stream};
        use candle_core::DType;

        let (b, v) = logits.dims2()?;
        let dev = logits.device().as_cuda_device()?;
        let dtype = logits.dtype();

        let logits = if !logits.is_contiguous() { logits.contiguous()? } else { logits.clone() };
        let (storage, _) = logits.storage_and_layout();
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Sampler expects CUDA tensor"),
        };

        let stream = dev.cuda_stream();
        let out_tokens = unsafe { dev.alloc::<i32>(b) }.w()?;
        let out_ptr = out_tokens.device_ptr(&stream).0 as *mut i32;
        let stream_ptr = get_raw_stream(dev);

        match dtype {
            DType::F32 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::F32(inp) => inp.device_ptr(&stream).0 as *const f32,
                    _ => candle_core::bail!("Dtype mismatch"),
                };
                unsafe {
                    ffi::sampling_f32(logits_ptr, out_ptr, b as i32, v as i32, k as i32, temperature, p, seed, token_pos, stream_ptr);
                }
            }
            DType::F16 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::F16(inp) => inp.device_ptr(&stream).0 as *const core::ffi::c_void,
                    _ => candle_core::bail!("Dtype mismatch"),
                };
                unsafe {
                    ffi::sampling_f16(logits_ptr, out_ptr, b as i32, v as i32, k as i32, temperature, p, seed, token_pos, stream_ptr);
                }
            }
            DType::BF16 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::BF16(inp) => inp.device_ptr(&stream).0 as *const core::ffi::c_void,
                    _ => candle_core::bail!("Dtype mismatch"),
                };
                unsafe {
                    ffi::sampling_bf16(logits_ptr, out_ptr, b as i32, v as i32, k as i32, temperature, p, seed, token_pos, stream_ptr);
                }
            }
            _ => candle_core::bail!("Sampler unsupported dtype {:?}", dtype),
        }

        let mut host_out = vec![0i32; b];
        unsafe {
            result::memcpy_dtoh_sync(&mut host_out, out_tokens.device_ptr(&stream).0).map_err(candle_core::Error::wrap)?;
        }

        Ok(host_out.into_iter().map(|x| x as u32).collect())
    }

    #[cfg(feature = "metal")]
    pub fn sample(&self, _: &Tensor, _: usize, _: f32, _: f32, _: u64) -> Result<Vec<u32>> {
        candle_core::bail!("Metal sampler not implemented yet")
    }
}
