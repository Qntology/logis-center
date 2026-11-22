//! Asynchronous host-to-device prefetch helper for offloaded KV slabs and layer weights.
//!
//! Uses `candle_core::{rocm,cuda}_backend::offload::AsyncPrefetchSlot` when the payload reaches
//! `ASYNC_DMA_MIN_BYTES` (4 MiB, re-exported from `candle_core::accel`) so PCIe DMA overlaps
//! with compute kernels, and performs an explicit synchronous copy on smaller transfers or
//! CPU/Vulkan/Metal devices (never a silent no-op — `P4-F02`).
//!
//! Also re-exports `candle_nn::kv_offload::{OffloadedTurboKvEngine, PinnedKvSlab}` when built
//! with `turbo-quant` so offloaded KV slab streaming shares one unified engine surface (`P1-2`).

use anyhow::{bail, Result};
use candle_core::{DType, Device, Shape, Storage, Tensor};

/// Minimum payload size (4 MiB) for pinned async DMA routing (single source of truth in `candle-core`).
pub use candle_core::accel::ASYNC_DMA_MIN_BYTES;

#[cfg(feature = "turbo-quant")]
pub use candle_nn::kv_offload::{OffloadedTurboKvEngine, PinnedKvSlab};

enum BackendSlot {
    #[cfg(feature = "rocm")]
    Rocm(candle_core::rocm_backend::offload::AsyncPrefetchSlot),
    #[cfg(feature = "cuda")]
    Cuda(candle_core::cuda_backend::offload::AsyncPrefetchSlot),
    Sync,
}

/// Handle to a prefetched tensor (either async DMA in flight on the copy stream or already copied).
pub struct PrefetchedTensor {
    tensor: Tensor,
    queued_async: bool,
}

impl PrefetchedTensor {
    /// Returns true if the transfer was queued asynchronously on the device copy stream.
    pub fn is_async(&self) -> bool {
        self.queued_async
    }

    /// Orders subsequent kernels on the device compute stream after the DMA upload (if async)
    /// and yields the device tensor.
    pub fn wait_and_take(self, prefetcher: &mut OffloadPrefetcher) -> Result<Tensor> {
        if self.queued_async {
            prefetcher.wait_on_compute()?;
        }
        Ok(self.tensor)
    }
}

/// Handle to a prefetched `(K, V)` KV block pair.
pub struct PrefetchedKv {
    pub k: PrefetchedTensor,
    pub v: PrefetchedTensor,
}

impl PrefetchedKv {
    pub fn wait_and_take(self, prefetcher: &mut OffloadPrefetcher) -> Result<(Tensor, Tensor)> {
        if self.k.queued_async || self.v.queued_async {
            prefetcher.wait_on_compute()?;
        }
        Ok((self.k.tensor, self.v.tensor))
    }
}

/// Reusable page-locked staging prefetcher bound to a single target `Device`.
pub struct OffloadPrefetcher {
    device: Device,
    slot: BackendSlot,
    min_async_bytes: usize,
}

/// Borrows the raw contiguous host bytes of a CPU tensor without allocating an intermediate `Vec<u8>`.
fn with_cpu_tensor_raw_bytes<R>(
    t: &Tensor,
    f: impl FnOnce(&[u8]) -> Result<R>,
) -> Result<Option<R>> {
    if !t.device().is_cpu() {
        return Ok(None);
    }
    let t = if t.is_contiguous() && t.layout().start_offset() == 0 {
        t.clone()
    } else {
        t.contiguous()?
    };
    let (st, layout) = t.storage_and_layout();
    let n = layout.shape().elem_count();
    let raw: &[u8] = match &*st {
        Storage::Cpu(cpu) => {
            use candle_core::CpuStorage;
            match cpu {
                CpuStorage::U8(v) => &v[..n],
                CpuStorage::U32(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4)
                },
                CpuStorage::I64(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 8)
                },
                CpuStorage::F32(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4)
                },
                CpuStorage::F16(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 2)
                },
                CpuStorage::BF16(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 2)
                },
                CpuStorage::F8E4M3(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n)
                },
                CpuStorage::F64(v) => unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 8)
                },
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(f(raw)?))
}

impl OffloadPrefetcher {
    pub fn new(device: &Device, capacity_bytes: usize) -> Result<Self> {
        let cap = capacity_bytes.max(1);
        #[cfg(feature = "rocm")]
        if let Device::Rocm(r) = device {
            if let Ok(slot) = candle_core::rocm_backend::offload::AsyncPrefetchSlot::new(r, cap) {
                return Ok(Self {
                    device: device.clone(),
                    slot: BackendSlot::Rocm(slot),
                    min_async_bytes: ASYNC_DMA_MIN_BYTES,
                });
            }
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = device {
            if let Ok(slot) = candle_core::cuda_backend::offload::AsyncPrefetchSlot::new(c, cap) {
                return Ok(Self {
                    device: device.clone(),
                    slot: BackendSlot::Cuda(slot),
                    min_async_bytes: ASYNC_DMA_MIN_BYTES,
                });
            }
        }
        let _ = cap;
        Ok(Self {
            device: device.clone(),
            slot: BackendSlot::Sync,
            min_async_bytes: ASYNC_DMA_MIN_BYTES,
        })
    }

    pub fn is_async_capable(&self) -> bool {
        !matches!(self.slot, BackendSlot::Sync)
    }

    pub fn capacity_bytes(&self) -> usize {
        match &self.slot {
            #[cfg(feature = "rocm")]
            BackendSlot::Rocm(s) => s.capacity_bytes(),
            #[cfg(feature = "cuda")]
            BackendSlot::Cuda(s) => s.capacity_bytes(),
            BackendSlot::Sync => 0,
        }
    }

    /// Uploads raw host bytes into an existing preallocated destination tensor (`dst`) at `dst_off`
    /// bytes.
    ///
    /// - Returns `Ok(true)` when queued asynchronously on the device copy stream (callers must call
    ///   [`Self::wait_on_compute`] before compute kernels read `dst`).
    /// - Returns `Ok(false)` when performed synchronously (payload below `ASYNC_DMA_MIN_BYTES`,
    ///   larger than pinned slot capacity, or non-CUDA/ROCm device). Never silently drops data (`P4-F02`).
    pub fn prefetch_into(&mut self, src: &[u8], dst: &Tensor, dst_off: usize) -> Result<bool> {
        if src.is_empty() {
            return Ok(false);
        }
        match &mut self.slot {
            #[cfg(feature = "rocm")]
            BackendSlot::Rocm(slot) => {
                if src.len() <= slot.capacity_bytes() {
                    let (st, _) = dst.storage_and_layout();
                    if let Storage::Rocm(r_dst) = &*st {
                        slot.prefetch_bytes(src, r_dst, dst_off)?;
                        return Ok(src.len() >= self.min_async_bytes);
                    }
                }
            }
            #[cfg(feature = "cuda")]
            BackendSlot::Cuda(slot) => {
                if src.len() <= slot.capacity_bytes() {
                    let (st, _) = dst.storage_and_layout();
                    if let Storage::Cuda(c_dst) = &*st {
                        slot.prefetch_bytes(src, c_dst, dst_off)?;
                        return Ok(src.len() >= self.min_async_bytes);
                    }
                }
            }
            BackendSlot::Sync => {}
        }

        // Synchronous fallback: actually write `src` into `dst` at `dst_off` (never a silent no-op).
        let esz = dst.dtype().size_in_bytes();
        if esz == 0 || src.len() % esz != 0 || dst_off % esz != 0 {
            bail!(
                "prefetch_into sync fallback: unaligned byte range off={dst_off} len={} for dtype {:?}",
                src.len(),
                dst.dtype()
            );
        }
        let elem_off = dst_off / esz;
        let elem_len = src.len() / esz;
        let dst_total = dst.elem_count();
        if elem_off.saturating_add(elem_len) > dst_total {
            bail!(
                "prefetch_into sync fallback: range {elem_off}..{} exceeds dst elems {dst_total}",
                elem_off + elem_len
            );
        }
        let src_dev = Tensor::from_raw_buffer(src, dst.dtype(), &[elem_len], dst.device())?;
        // Flatten 1D view shares the underlying storage buffer on contiguous tensors.
        let dst_1d = if dst.rank() == 1 {
            dst.clone()
        } else {
            dst.reshape((dst_total,))?
        };
        dst_1d.slice_set(&src_dev, 0, elem_off)?;
        Ok(false)
    }

    /// Uploads a host tensor to `self.device`, using async copy-stream DMA when the tensor payload
    /// is at least `ASYNC_DMA_MIN_BYTES` and fits the pinned slot capacity, or synchronous
    /// `to_device` otherwise.
    pub fn prefetch_tensor(&mut self, host: &Tensor) -> Result<PrefetchedTensor> {
        if host.device().same_device(&self.device) {
            return Ok(PrefetchedTensor {
                tensor: host.clone(),
                queued_async: false,
            });
        }
        let byte_len = host.elem_count() * host.dtype().size_in_bytes();
        if self.is_async_capable()
            && byte_len >= self.min_async_bytes
            && byte_len <= self.capacity_bytes()
            && host.device().is_cpu()
        {
            if let Some(prefetched) = with_cpu_tensor_raw_bytes(host, |raw| {
                let dst = self.alloc_synced_dst(host.shape(), host.dtype())?;
                let queued = self.prefetch_into(raw, &dst, 0)?;
                Ok(PrefetchedTensor {
                    tensor: dst,
                    queued_async: queued,
                })
            })? {
                return Ok(prefetched);
            }
        }
        Ok(PrefetchedTensor {
            tensor: host.to_device(&self.device)?,
            queued_async: false,
        })
    }

    /// Prefetches a `(K, V)` host block pair onto `self.device`.
    pub fn prefetch_kv_pair(&mut self, k_host: &Tensor, v_host: &Tensor) -> Result<PrefetchedKv> {
        let k = self.prefetch_tensor(k_host)?;
        let v = self.prefetch_tensor(v_host)?;
        Ok(PrefetchedKv { k, v })
    }

    fn alloc_synced_dst(&self, shape: &Shape, dtype: DType) -> Result<Tensor> {
        let dst = Tensor::zeros(shape, dtype, &self.device)?;
        // Ensure compute-stream zero-init finishes before copy-stream DMA writes into `dst`.
        if self.device.is_cuda() || self.device.is_rocm() {
            self.device.synchronize()?;
        }
        Ok(dst)
    }

    /// Orders subsequent kernels on the device compute stream after any queued async DMA upload
    /// without blocking the host thread.
    pub fn wait_on_compute(&mut self) -> Result<()> {
        match &mut self.slot {
            #[cfg(feature = "rocm")]
            BackendSlot::Rocm(slot) => {
                if let Device::Rocm(r) = &self.device {
                    slot.wait_on_compute(r)?;
                }
            }
            #[cfg(feature = "cuda")]
            BackendSlot::Cuda(slot) => {
                if let Device::Cuda(c) = &self.device {
                    slot.wait_on_compute(c)?;
                }
            }
            BackendSlot::Sync => {}
        }
        Ok(())
    }

    /// Blocks the host until any queued async DMA upload finishes.
    pub fn synchronize(&mut self) -> Result<()> {
        match &mut self.slot {
            #[cfg(feature = "rocm")]
            BackendSlot::Rocm(slot) => slot.synchronize()?,
            #[cfg(feature = "cuda")]
            BackendSlot::Cuda(slot) => slot.synchronize()?,
            BackendSlot::Sync => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_into_performs_sync_copy_on_small_payload_and_cpu() -> Result<()> {
        let mut p = OffloadPrefetcher::new(&Device::Cpu, 1024)?;
        let dst = Tensor::zeros((8,), DType::F32, &Device::Cpu)?;
        let vals = [1.5f32, -2.25, 3.75, 4.0];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let queued = p.prefetch_into(&raw, &dst, 8)?; // offset 2 floats = 8 bytes
        assert!(!queued, "CPU path must copy synchronously");
        let got = dst.to_vec1::<f32>()?;
        assert_eq!(got, vec![0.0, 0.0, 1.5, -2.25, 3.75, 4.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn prefetch_kv_pair_roundtrips_exact_bits_on_cpu() -> Result<()> {
        let mut p = OffloadPrefetcher::new(&Device::Cpu, 1024)?;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &Device::Cpu)?;
        let v = Tensor::from_vec(vec![-5.0f32, 6.0, -7.0, 8.0], (1, 1, 2, 2), &Device::Cpu)?;
        let pf = p.prefetch_kv_pair(&k, &v)?;
        let (kd, vd) = pf.wait_and_take(&mut p)?;
        assert_eq!(kd.flatten_all()?.to_vec1::<f32>()?, k.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(vd.flatten_all()?.to_vec1::<f32>()?, v.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }
}
