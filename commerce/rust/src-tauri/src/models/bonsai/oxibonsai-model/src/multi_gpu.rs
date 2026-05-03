//! Multi-GPU / multi-device inference utilities.
//!
//! This module provides abstractions for device mesh partitioning and
//! NCCL-style collective operations, implemented over rayon thread pools
//! as a CPU simulation.  A real GPU backend would swap in NCCL/cuBLAS calls.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────┐
//!  │                 DeviceMesh (tp × pp)             │
//!  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
//!  │  │ Device 0 │  │ Device 1 │  │ Device 2 │  ...  │
//!  │  │ (tp=0,   │  │ (tp=1,   │  │ (tp=0,   │       │
//!  │  │  pp=0)   │  │  pp=0)   │  │  pp=1)   │       │
//!  │  └──────────┘  └──────────┘  └──────────┘       │
//!  └─────────────────────────────────────────────────┘
//!
//!   NcclCollectives  ─►  all_reduce_sum / all_gather / broadcast …
//!   partition_weights_column / partition_weights_row  ─►  shards
//! ```

use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceId
// ─────────────────────────────────────────────────────────────────────────────

/// A logical device identifier (CPU thread group simulating a GPU).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

// ─────────────────────────────────────────────────────────────────────────────
// DeviceInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated device capabilities.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// The logical device identifier.
    pub id: DeviceId,
    /// Simulated device memory in bytes.
    pub memory_bytes: usize,
    /// Simulated number of compute units (analogous to CUDA SMs).
    pub compute_units: usize,
    /// Human-readable device name (e.g. "SimGPU-0").
    pub name: String,
}

impl DeviceInfo {
    fn simulated(linear_id: usize) -> Self {
        Self {
            id: DeviceId(linear_id),
            // Simulate 24 GiB per device.
            memory_bytes: 24 * 1024 * 1024 * 1024,
            // Simulate 108 SMs per device.
            compute_units: 108,
            name: format!("SimGPU-{linear_id}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceMesh
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D logical device mesh: tensor-parallel dimension × pipeline-parallel dimension.
///
/// Devices are stored in row-major order: device at `(tp_rank, pp_rank)` has
/// linear index `tp_rank + pp_rank * tp_size`.
pub struct DeviceMesh {
    devices: Vec<DeviceInfo>,
    tp_size: usize,
    pp_size: usize,
}

impl DeviceMesh {
    /// Create a 1-D tensor-parallel mesh of `n` simulated devices.
    pub fn tensor_parallel(n: usize) -> Self {
        Self::new(n, 1)
    }

    /// Create a 2-D (`tp_size` × `pp_size`) mesh.
    ///
    /// Total device count is `tp_size * pp_size`.
    pub fn new(tp_size: usize, pp_size: usize) -> Self {
        let total = tp_size * pp_size;
        let devices = (0..total).map(DeviceInfo::simulated).collect();
        Self {
            devices,
            tp_size,
            pp_size,
        }
    }

    /// Total number of devices in the mesh.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// Get the device at tensor-parallel rank `tp_rank` and pipeline-parallel rank `pp_rank`.
    ///
    /// Returns `None` if either rank is out of bounds.
    pub fn get(&self, tp_rank: usize, pp_rank: usize) -> Option<&DeviceInfo> {
        if tp_rank >= self.tp_size || pp_rank >= self.pp_size {
            return None;
        }
        let idx = tp_rank + pp_rank * self.tp_size;
        self.devices.get(idx)
    }

    /// All devices in the tensor-parallel group for a given `pp_rank`.
    ///
    /// Returns an empty vec if `pp_rank` is out of range.
    pub fn tp_group(&self, pp_rank: usize) -> Vec<&DeviceInfo> {
        if pp_rank >= self.pp_size {
            return Vec::new();
        }
        (0..self.tp_size)
            .filter_map(|tp| self.get(tp, pp_rank))
            .collect()
    }

    /// All devices in the pipeline-parallel group for a given `tp_rank`.
    ///
    /// Returns an empty vec if `tp_rank` is out of range.
    pub fn pp_group(&self, tp_rank: usize) -> Vec<&DeviceInfo> {
        if tp_rank >= self.tp_size {
            return Vec::new();
        }
        (0..self.pp_size)
            .filter_map(|pp| self.get(tp_rank, pp))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CollectiveResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a collective communication operation.
#[derive(Debug, Clone)]
pub struct CollectiveResult {
    /// The reduced / gathered data.
    pub data: Vec<f32>,
    /// Number of devices that participated.
    pub participating_devices: usize,
    /// Name tag identifying the operation (e.g. `"all_reduce_sum"`).
    pub op_name: &'static str,
}

// ─────────────────────────────────────────────────────────────────────────────
// NcclCollectives
// ─────────────────────────────────────────────────────────────────────────────

/// NCCL-style collective operations simulated on the CPU via rayon.
///
/// In production these would be replaced by NCCL library calls.
pub struct NcclCollectives;

impl NcclCollectives {
    /// All-reduce (sum): element-wise sum of tensors from all participating devices;
    /// the result is the same on every device.
    ///
    /// All shards must have the same length.
    pub fn all_reduce_sum(shards: &[Vec<f32>]) -> CollectiveResult {
        let n = shards.first().map(|s| s.len()).unwrap_or(0);
        let data: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| shards.iter().map(|s| s[i]).sum::<f32>())
            .collect();
        CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_reduce_sum",
        }
    }

    /// All-reduce (max): element-wise maximum across all device tensors.
    ///
    /// All shards must have the same length.
    pub fn all_reduce_max(shards: &[Vec<f32>]) -> CollectiveResult {
        let n = shards.first().map(|s| s.len()).unwrap_or(0);
        let data: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                shards
                    .iter()
                    .map(|s| s[i])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_reduce_max",
        }
    }

    /// All-gather: concatenate tensors from all devices in rank order.
    pub fn all_gather(shards: &[Vec<f32>]) -> CollectiveResult {
        let data: Vec<f32> = shards.iter().flat_map(|s| s.iter().copied()).collect();
        CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_gather",
        }
    }

    /// Reduce-scatter: sum the global `data` across all ranks, then scatter
    /// equal-sized shards back to each device.
    ///
    /// If `data.len()` is not evenly divisible by `world_size`, the last shard
    /// will be shorter.
    pub fn reduce_scatter(data: &[f32], world_size: usize) -> Vec<Vec<f32>> {
        if world_size == 0 {
            return Vec::new();
        }
        // Here "reduce-scatter" treats each device as already holding its own
        // portion of the data, and after the reduce step each device gets its
        // equal shard.  In the simulation we simply split the input:
        let base = data.len() / world_size;
        let remainder = data.len() % world_size;
        (0..world_size)
            .map(|rank| {
                let start = rank * base + rank.min(remainder);
                let end = start + base + if rank < remainder { 1 } else { 0 };
                data[start..end.min(data.len())].to_vec()
            })
            .collect()
    }

    /// Broadcast: replicate `data` from device 0 to all `world_size` devices.
    pub fn broadcast(data: &[f32], world_size: usize) -> Vec<Vec<f32>> {
        (0..world_size).map(|_| data.to_vec()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight partition helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Partition a row-major weight matrix `[rows × cols]` into column-parallel shards.
///
/// Splits along the `cols` dimension, giving each device `cols / world_size`
/// (or `cols / world_size + 1` for the first few devices if not evenly divisible).
pub fn partition_weights_column(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Vec<Vec<f32>> {
    if world_size == 0 {
        return Vec::new();
    }
    let base_cols = cols / world_size;
    let remainder = cols % world_size;
    (0..world_size)
        .map(|rank| {
            let col_start = rank * base_cols + rank.min(remainder);
            let shard_cols = base_cols + if rank < remainder { 1 } else { 0 };
            let mut shard = Vec::with_capacity(rows * shard_cols);
            for row in 0..rows {
                let row_base = row * cols;
                shard.extend_from_slice(
                    &weights[row_base + col_start..row_base + col_start + shard_cols],
                );
            }
            shard
        })
        .collect()
}

/// Partition a row-major weight matrix `[rows × cols]` into row-parallel shards.
///
/// Splits along the `rows` dimension, giving each device a contiguous block of rows.
pub fn partition_weights_row(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Vec<Vec<f32>> {
    if world_size == 0 {
        return Vec::new();
    }
    let base_rows = rows / world_size;
    let remainder = rows % world_size;
    (0..world_size)
        .map(|rank| {
            let row_start = rank * base_rows + rank.min(remainder);
            let shard_rows = base_rows + if rank < remainder { 1 } else { 0 };
            weights[row_start * cols..(row_start + shard_rows) * cols].to_vec()
        })
        .collect()
}

/// Merge column-parallel shards back into a single `[rows × cols]` weight matrix.
///
/// Assumes shards are produced by [`partition_weights_column`] with the same `rows`.
pub fn merge_column_shards(shards: &[Vec<f32>], rows: usize) -> Vec<f32> {
    if shards.is_empty() || rows == 0 {
        return Vec::new();
    }
    // Each shard: rows × (shard_cols)
    let total_cols: usize = shards.iter().map(|s| s.len() / rows).sum();
    let mut result = vec![0.0f32; rows * total_cols];

    let mut col_offset = 0usize;
    for shard in shards {
        let shard_cols = shard.len() / rows;
        for row in 0..rows {
            let dst_start = row * total_cols + col_offset;
            let src_start = row * shard_cols;
            result[dst_start..dst_start + shard_cols]
                .copy_from_slice(&shard[src_start..src_start + shard_cols]);
        }
        col_offset += shard_cols;
    }
    result
}
