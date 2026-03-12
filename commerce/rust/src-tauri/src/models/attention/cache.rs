#[allow(unused_imports)]
use candle_core::{backend::BackendDevice, Device, Result, Storage, Tensor};
use std::collections::HashMap;

pub fn swap_blocks(
    src: &Tensor,
    dst: &Tensor,
    block_mapping: &HashMap<usize, usize>,
) -> Result<()> {
    use candle_core::DType;
    use half::{bf16, f16};
    #[cfg(feature = "cuda")]
    fn call_fwd<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::WithDType,
    >(
        src: &Tensor,
        dst: &Tensor,
        block_mapping: &HashMap<usize, usize>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
        use crate::models::attention::cuda_utils::get_cuda_stream_ptr;
        use std::slice;
        let block_size_elements = src.elem_count() / src.dim(0)?;
        let (src_storage, _) = src.storage_and_layout();
        let (dst_storage, _) = dst.storage_and_layout();
        let dtype_size = src.dtype().size_in_bytes();

        match (src.device(), dst.device()) {
            (Device::Cpu, Device::Cuda(dst_dev)) => {
                let Storage::Cpu(src_storage) = &*src_storage else { candle_core::bail!("Invalid source kvcache storage!") };
                let Storage::Cuda(dst_storage) = &*dst_storage else { candle_core::bail!("Invalid dst kvcache storage!") };
                let cpu_num_blocks = src.dim(0)?;
                let gpu_num_blocks = dst.dim(0)?;

                let _stream = dst_dev.cu_stream();
                let dst_ptr = *dst_storage.as_cuda_slice::<T>()?.device_ptr();
                let src_slice: &[T] = src_storage.as_slice()?;
                let stream_ptr = get_cuda_stream_ptr(dst_dev);

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset: usize = src_block_number * block_size_elements;
                    let dst_offset = (*dst_block_number * block_size_elements * dtype_size) as u64;
                    unsafe {
                        result::memcpy_htod_async(dst_ptr.wrapping_add(dst_offset), &src_slice[src_offset..src_offset + block_size_elements], stream_ptr).map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            (Device::Cuda(src_dev), Device::Cpu) => {
                let Storage::Cuda(src_storage) = &*src_storage else { candle_core::bail!("Invalid source kvcache storage!") };
                let Storage::Cpu(dst_storage) = &*dst_storage else { candle_core::bail!("Invalid dst kvcache storage!") };
                let gpu_num_blocks = src.dim(0)?;
                let cpu_num_blocks = dst.dim(0)?;

                let _stream = src_dev.cu_stream();
                let src_ptr = *src_storage.as_cuda_slice::<T>()?.device_ptr();
                let dst_slice: &[T] = dst_storage.as_slice()?;
                let ptr = dst_slice.as_ptr() as *mut u8;
                let stream_ptr = get_cuda_stream_ptr(src_dev);

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset = (*src_block_number * block_size_elements * dtype_size) as u64;
                    let dst_offset = *dst_block_number * block_size_elements * dtype_size;
                    let dst_chunk = unsafe { slice::from_raw_parts_mut(ptr.wrapping_add(dst_offset) as *mut T, block_size_elements) };
                    unsafe {
                        result::memcpy_dtoh_async(dst_chunk, src_ptr.wrapping_add(src_offset), stream_ptr).map_err(candle_core::Error::wrap)?;
                    }
                }
                src_dev.synchronize()
            }
            (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
                let Storage::Cuda(src_storage) = &*src_storage else { candle_core::bail!("Invalid source kvcache storage!") };
                let Storage::Cuda(dst_storage) = &*dst_storage else { candle_core::bail!("Invalid dst kvcache storage!") };
                let _src_stream = src_dev.cu_stream();
                let _dst_stream = dst_dev.cu_stream();
                let src_ptr = *src_storage.as_cuda_slice::<T>()?.device_ptr();
                let dst_ptr = *dst_storage.as_cuda_slice::<T>()?.device_ptr();
                let stream_ptr = get_cuda_stream_ptr(dst_dev);

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_off = (*src_block_number * block_size_elements * dtype_size) as u64;
                    let dst_off = (*dst_block_number * block_size_elements * dtype_size) as u64;
                    unsafe {
                        result::memcpy_dtod_async(dst_ptr.wrapping_add(dst_off), src_ptr.wrapping_add(src_off), block_size_elements * dtype_size, stream_ptr).map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            _ => candle_core::bail!("Swap failed: unsupported device combo")
        }
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        src: &Tensor, dst: &Tensor, block_mapping: &HashMap<usize, usize>,
    ) -> Result<()> {
        candle_core::bail!("Metal swap blocks not implemented in this pass")
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor, _: &Tensor, _: &HashMap<usize, usize>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("swap_blocks not implemented")
    }

    match src.dtype() {
        DType::F16 => call_fwd::<f16>(src, dst, block_mapping),
        DType::BF16 => call_fwd::<bf16>(src, dst, block_mapping),
        DType::U8 => call_fwd::<u8>(src, dst, block_mapping),
        _ => candle_core::bail!("Unsupported kvcache dtype")
    }
}

pub fn clear_blocks(cache: &Tensor, block_ids: &Vec<u32>) -> Result<()> {
    use candle_core::DType;
    use half::{bf16, f16};
    #[cfg(feature = "cuda")]
    fn call_fwd<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::WithDType,
    >(
        cache: &Tensor, block_ids: &Vec<u32>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
        let block_size_elements = cache.elem_count() / cache.dim(0)?;
        let (cache_storage, _) = cache.storage_and_layout();
        let dtype_size = cache.dtype().size_in_bytes();
        let dst_dev = cache.device().as_cuda_device()?;
        let Storage::Cuda(cache_storage) = &*cache_storage else { candle_core::bail!("Invalid kvcache storage!") };
        let _stream = dst_dev.cu_stream();
        let cache_ptr = *cache_storage.as_cuda_slice::<T>()?.device_ptr();

        for block_number in block_ids {
            let offset = (*block_number as u64 * block_size_elements as u64 * dtype_size as u64);
            unsafe {
                result::memset_d8_sync(cache_ptr.wrapping_add(offset), 0, block_size_elements * dtype_size).map_err(candle_core::Error::wrap)?
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        cache: &Tensor, block_ids: &Vec<u32>,
    ) -> Result<()> {
        candle_core::bail!("Metal clear blocks not implemented")
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor, _: &Vec<u32>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("clear_blocks not implemented")
    }

    match cache.dtype() {
        DType::F16 => call_fwd::<f16>(cache, block_ids),
        DType::BF16 => call_fwd::<bf16>(cache, block_ids),
        DType::U8 => call_fwd::<u8>(cache, block_ids),
        _ => candle_core::bail!("Unsupported kvcache dtype")
    }
}
