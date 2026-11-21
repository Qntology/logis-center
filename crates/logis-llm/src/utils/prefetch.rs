//! Asynchronous host-to-device prefetch helper for offloaded KV slabs and layer weights.
//!
//! Uses `candle_core::{rocm,cuda}_backend::offload::AsyncPrefetchSlot` when the payload exceeds
//! `ASYNC_DMA_MIN_BYTES` (4 MiB) so PCIe DMA overlaps with compute kernels, and falls back to
//! direct synchronous copies on smaller transfers or CPU/Vulkan/Metal devices.

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Minimum payload size (4 MiB) for pinned async DMA routing.
pub const ASYNC_DMA_MIN_BYTES: usize = 4 * 1024 * 1024;

enum BackendSlot {
    #[cfg(feature = "rocm")]
    Rocm(candle_core::rocm_backend::offload::AsyncPrefetchSlot),
    #[cfg(feature = "cuda")]
    Cuda(candle_core::cuda_backend::offload::AsyncPrefetchSlot),
    Sync,
}

/// Reusable page-locked staging prefetcher bound to a single target `Device`.
pub struct OffloadPrefetcher {
    #[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
    device: Device,
    slot: BackendSlot,
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
                });
            }
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = device {
            if let Ok(slot) = candle_core::cuda_backend::offload::AsyncPrefetchSlot::new(c, cap) {
                return Ok(Self {
                    device: device.clone(),
                    slot: BackendSlot::Cuda(slot),
                });
            }
        }
        let _ = cap;
        Ok(Self {
            device: device.clone(),
            slot: BackendSlot::Sync,
        })
    }

    pub fn is_async_capable(&self) -> bool {
        !matches!(self.slot, BackendSlot::Sync)
    }

    /// Uploads raw host bytes into an existing preallocated destination tensor (`dst`) at `dst_off`
    /// bytes. On ROCm/CUDA when `src.len() >= ASYNC_DMA_MIN_BYTES`, queues an async DMA transfer
    /// on the device copy stream; call [`Self::wait_on_compute`] before kernels read `dst`.
    pub fn prefetch_into(&mut self, src: &[u8], dst: &Tensor, dst_off: usize) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        match &mut self.slot {
            #[cfg(feature = "rocm")]
            BackendSlot::Rocm(slot) => {
                if src.len() >= ASYNC_DMA_MIN_BYTES && src.len() <= slot.capacity_bytes() {
                    let (st, _) = dst.storage_and_layout();
                    if let candle_core::Storage::Rocm(r_dst) = &*st {
                        slot.prefetch_bytes(src, r_dst, dst_off)?;
                        return Ok(());
                    }
                }
            }
            #[cfg(feature = "cuda")]
            BackendSlot::Cuda(slot) => {
                if src.len() >= ASYNC_DMA_MIN_BYTES && src.len() <= slot.capacity_bytes() {
                    let (st, _) = dst.storage_and_layout();
                    if let candle_core::Storage::Cuda(c_dst) = &*st {
                        slot.prefetch_bytes(src, c_dst, dst_off)?;
                        return Ok(());
                    }
                }
            }
            BackendSlot::Sync => {}
        }
        let _ = (src, dst, dst_off);
        Ok(())
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
