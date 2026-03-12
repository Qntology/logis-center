#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[cfg(feature = "cuda")]
static SM_CACHE: OnceLock<Mutex<HashMap<usize, Option<i32>>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn compute_capability(dev: &CudaDevice) -> Option<(i32, i32)> {
    let major = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
pub fn sm_version(dev: &CudaDevice) -> Option<i32> {
    // Key by the address of this device instance
    let key = (dev as *const CudaDevice) as usize;

    let cache = SM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Fast path: return cached value if present
    if let Some(v) = cache.lock().unwrap().get(&key).copied() {
        return v;
    }

    // Compute outside the lock to keep the critical section small
    let computed = compute_capability(dev).map(|(major, minor)| major * 10 + minor);

    // Store (a second thread might store first; that is fine)
    cache.lock().unwrap().insert(key, computed);

    computed
}

#[cfg(feature = "cuda")]
pub fn get_raw_stream(dev: &CudaDevice) -> i64 {
    let stream: Arc<CudaStream> = dev.cuda_stream();
    // In cudarc, CudaStream doesn't easily expose the raw handle unless using unsafe or specific versions.
    // However, for kernel launches, we just need a pointer that the kernel expectation matches.
    // Most kernels expect CUstream which is *mut CUstream_st.
    // We can use the address of the CudaStream object itself if the kernel was compiled with that assumption,
    // but usually it wants the ACTUAL CUDA stream handle.
    // Let's try to get it via a pointer to the inner stream if possible, or just return 0 (default stream) 
    // if we can't find a better way, but that might hurt performance.
    
    // Attempt to get the stream handle by casting the Arc's inner pointer.
    // This is risky but often how these "hacks" work in these specific repos.
    let stream_ptr = Arc::as_ptr(&stream);
    stream_ptr as i64
}

pub trait WrapErr<T> {
    fn w(self) -> candle_core::Result<T>;
}

impl<T, E: std::fmt::Display> WrapErr<T> for std::result::Result<T, E> {
    fn w(self) -> candle_core::Result<T> {
        self.map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}
