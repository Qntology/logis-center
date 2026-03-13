#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
#[cfg(feature = "cuda")]
use candle_core::CudaDevice;
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
    let key = (dev as *const CudaDevice) as usize;
    let cache = SM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(&key).copied() {
        return v;
    }
    let computed = compute_capability(dev).map(|(major, minor)| major * 10 + minor);
    cache.lock().unwrap().insert(key, computed);
    computed
}

#[cfg(feature = "cuda")]
pub fn get_raw_stream(dev: &CudaDevice) -> i64 {
    *dev.cu_stream() as i64
}

#[cfg(feature = "cuda")]
pub fn get_cuda_stream_ptr(dev: &CudaDevice) -> *mut sys::CUstream_st {
    *dev.cu_stream() as *mut sys::CUstream_st
}

pub trait WrapErr<T> {
    fn w(self) -> candle_core::Result<T>;
}

impl<T, E: std::fmt::Display> WrapErr<T> for std::result::Result<T, E> {
    fn w(self) -> candle_core::Result<T> {
        self.map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}
