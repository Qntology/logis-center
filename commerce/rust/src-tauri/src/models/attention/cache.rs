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
        use std::slice;
        let block_size_elements = src.elem_count() / src.dim(0)?;
        let (src_storage, _) = src.storage_and_layout();
        let (dst_storage, _) = dst.storage_and_layout();
        let dtype_size = src.dtype().size_in_bytes();

        match (src.device(), dst.device()) {
            (Device::Cpu, Device::Cuda(dst_dev)) => {
                let Storage::Cpu(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cuda(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let cpu_num_blocks = src.dim(0)?;
                let gpu_num_blocks = dst.dim(0)?;

                let stream = dst_dev.cuda_stream();
                let dst_ptr = dst_storage.as_cuda_slice::<T>()?.device_ptr(&stream).0;
                let src_slice: &[T] = src_storage.as_slice()?;

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset: usize = src_block_number * block_size_elements;
                    assert!(
                        *src_block_number < cpu_num_blocks,
                        "Invalid cpu block {} / {}",
                        src_block_number,
                        cpu_num_blocks
                    );
                    assert!(
                        *dst_block_number < gpu_num_blocks,
                        "Invalid gpu block {} / {}",
                        dst_block_number,
                        gpu_num_blocks
                    );

                    let dst_offset: u64 = (*dst_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();

                    unsafe {
                        result::memcpy_htod_async(
                            dst_ptr.wrapping_add(dst_offset),
                            &src_slice[src_offset..src_offset + block_size_elements],
                            *stream as *mut _,
                        )
                        .map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            (Device::Cuda(src_dev), Device::Cpu) => {
                let Storage::Cuda(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cpu(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let gpu_num_blocks = src.dim(0)?;
                let cpu_num_blocks = dst.dim(0)?;

                let stream = src_dev.cuda_stream();
                let src_ptr = src_storage
                    .as_cuda_slice::<T>()
                    .map_err(candle_core::Error::wrap)?
                    .device_ptr(&stream).0;
                let dst_slice: &[T] = dst_storage.as_slice().map_err(candle_core::Error::wrap)?;
                let ptr = dst_slice.as_ptr() as *mut u8;

                for (src_block_number, dst_block_number) in block_mapping {
                    assert!(
                        *src_block_number < gpu_num_blocks,
                        "Invalid gpu block {} / {}",
                        src_block_number,
                        gpu_num_blocks
                    );
                    assert!(
                        *dst_block_number < cpu_num_blocks,
                        "Invalid cpu block {} / {}",
                        dst_block_number,
                        cpu_num_blocks
                    );

                    let src_offset: u64 = (*src_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_offset: usize = *dst_block_number * block_size_elements * dtype_size;
                    let dst_chunk = unsafe {
                        slice::from_raw_parts_mut(
                            ptr.wrapping_add(dst_offset) as *mut T,
                            block_size_elements,
                        )
                    };

                    unsafe {
                        result::memcpy_dtoh_async(
                            dst_chunk,
                            src_ptr.wrapping_add(src_offset),
                            *stream as *mut _,
                        )
                        .map_err(candle_core::Error::wrap)?;
                    }
                }
                src_dev.synchronize()
            }
            (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
                let Storage::Cuda(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cuda(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let src_num_blocks = src.dim(0)?;
                let dst_num_blocks = dst.dim(0)?;

                let src_stream = src_dev.cuda_stream();
                let dst_stream = dst_dev.cuda_stream();
                let src_ptr = src_storage.as_cuda_slice::<T>()?.device_ptr(&src_stream).0;
                let dst_ptr = dst_storage.as_cuda_slice::<T>()?.device_ptr(&dst_stream).0;

                for (src_block_number, dst_block_number) in block_mapping {
                    assert!(*src_block_number < src_num_blocks);
                    assert!(*dst_block_number < dst_num_blocks);

                    let src_offset: u64 = (*src_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_offset: u64 = (*dst_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();

                    unsafe {
                        result::memcpy_dtod_async(
                            dst_ptr.wrapping_add(dst_offset),
                            src_ptr.wrapping_add(src_offset),
                            block_size_elements * dtype_size,
                            *dst_stream as *mut _,
                        )
                        .map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            (src, dst) => {
                candle_core::bail!("Tensors must be on either the GPU or CPU to swap, or GPU-GPU transfer, got {src:?} (src) and {dst:?} (dst).")
            }
        }
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        src: &Tensor,
        dst: &Tensor,
        block_mapping: &HashMap<usize, usize>,
    ) -> Result<()> {
        use metal::{self, MTLStorageMode};
        let block_size_elements = src.elem_count() / src.dim(0)?;
        let (src_storage, _) = src.storage_and_layout();
        let (dst_storage, _) = dst.storage_and_layout();
        let dtype_size = src.dtype().size_in_bytes();
        let block_size_bytes = block_size_elements * dtype_size;

        match (src.device(), dst.device()) {
            (Device::Cpu, Device::Metal(_)) => {
                let Storage::Cpu(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Metal(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_slice: &[T] = src_storage.as_slice()?;
                let dst_buffer = dst_storage.buffer();
                let dst_ptr = dst_buffer.contents() as *mut T;
                if dst_ptr.is_null() {
                    candle_core::bail!("Failed to get Metal buffer contents.");
                }
                let is_managed = dst_buffer.storage_mode() == MTLStorageMode::Managed;

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset = src_block_number * block_size_elements;
                    let dst_offset = dst_block_number * block_size_elements;

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_slice.as_ptr().add(src_offset),
                            dst_ptr.add(dst_offset),
                            block_size_elements,
                        );
                    }

                    if is_managed {
                        dst_buffer.did_modify_range(metal::NSRange {
                            location: (dst_offset * dtype_size) as u64,
                            length: block_size_bytes as u64,
                        });
                    }
                }
                Ok(())
            }
            (Device::Metal(_), Device::Cpu) => {
                let Storage::Metal(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cpu(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_buffer = src_storage.buffer();
                let dst_slice: &[T] = dst_storage.as_slice()?;
                let dst_ptr = dst_slice.as_ptr() as *mut T;
                let src_ptr = src_buffer.contents() as *const T;
                if src_ptr.is_null() {
                    candle_core::bail!("Failed to get Metal buffer contents.");
                }

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset = src_block_number * block_size_elements;
                    let dst_offset = dst_block_number * block_size_elements;

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_ptr.add(src_offset),
                            dst_ptr.add(dst_offset),
                            block_size_elements,
                        );
                    }
                }
                Ok(())
            }
            (Device::Metal(_), Device::Metal(dst_dev)) => {
                let Storage::Metal(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Metal(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_buffer = src_storage.buffer();
                let dst_buffer = dst_storage.buffer();

                let command_queue = dst_dev.new_command_queue();
                let command_buffer = command_queue.new_command_buffer();
                let blit_encoder = command_buffer.new_blit_command_encoder();

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_off = (*src_block_number * block_size_elements * dtype_size) as u64;
                    let dst_off = (*dst_block_number * block_size_elements * dtype_size) as u64;

                    blit_encoder.copy_from_buffer(
                        src_buffer,
                        src_off,
                        dst_buffer,
                        dst_off,
                        block_size_bytes as u64,
                    );
                }

                blit_encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            }
            (src, dst) => {
                candle_core::bail!("Unsupported device combination for Metal swap.")
            }
        }
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor,
        _: &Tensor,
        _: &HashMap<usize, usize>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("swap_blocks is not implemented on this platform.")
    }

    match src.dtype() {
        DType::F16 => call_fwd::<f16>(src, dst, block_mapping),
        DType::BF16 => call_fwd::<bf16>(src, dst, block_mapping),
        DType::U8 => call_fwd::<u8>(src, dst, block_mapping),
        _ => {
            candle_core::bail!("swap_blocks only accept f16/bf16/u8 kvcache dtypes!")
        }
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
        cache: &Tensor,
        block_ids: &Vec<u32>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
        let block_size_elements = cache.elem_count() / cache.dim(0)?;
        let (cache_storage, _) = cache.storage_and_layout();
        let dtype_size = cache.dtype().size_in_bytes();
        let dst_dev = cache.device().as_cuda_device()?;

        let Storage::Cuda(cache_storage) = &*cache_storage else {
            candle_core::bail!("Invalid kvcache storage!")
        };

        let stream = dst_dev.cuda_stream();
        let cache_ptr = cache_storage.as_cuda_slice::<T>()?.device_ptr(&stream).0;

        for block_number in block_ids {
            let offset: u64 = (*block_number as u64 * block_size_elements as u64 * dtype_size as u64);
            unsafe {
                result::memset_d8_sync(
                    cache_ptr.wrapping_add(offset),
                    0,
                    block_size_elements * dtype_size,
                )
                .map_err(candle_core::Error::wrap)?
            }
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        cache: &Tensor,
        block_ids: &Vec<u32>,
    ) -> Result<()> {
        let block_size_elements = cache.elem_count() / cache.dim(0)?;
        let (cache_storage, _) = cache.storage_and_layout();
        let dtype_size = cache.dtype().size_in_bytes();

        let Storage::Metal(cache_storage) = &*cache_storage else {
            candle_core::bail!("Invalid kvcache storage!")
        };

        let cache_buffer = cache_storage.buffer();
        let num_blocks = cache.dim(0)?;

        let cache_ptr = cache_buffer.contents() as *mut T;
        if cache_ptr.is_null() {
            candle_core::bail!("Failed to get Metal buffer contents.");
        }

        for block_number in block_ids {
            let src_offset = (*block_number as usize) * block_size_elements;
            assert!((*block_number as usize) < num_blocks);

            unsafe {
                std::ptr::write_bytes(cache_ptr.add(src_offset) as *mut u8, 0, block_size_elements * dtype_size);
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor,
        _: &Vec<u32>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("clear_blocks is not implemented on this platform.")
    }

    match cache.dtype() {
        DType::F16 => call_fwd::<f16>(cache, block_ids),
        DType::BF16 => call_fwd::<bf16>(cache, block_ids),
        DType::U8 => call_fwd::<u8>(cache, block_ids),
        _ => {
            candle_core::bail!("clear_blocks only accept f16/bf16/u8 kvcache dtypes!")
        }
    }
}
