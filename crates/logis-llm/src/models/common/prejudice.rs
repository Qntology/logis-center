//! Shared 16-slot LRU `PrejudiceCache` over active vocabulary rows (`K`) + single-row cold
//! prejudice evaluation (Phase 3 – Part 1 D3).
//!
//! Eliminates OOM from full-vocabulary F32 promotion (`models/qwen/generate.rs`) and shares
//! normalized target embeddings across `qwen3`, `qwen3_5`, and `qwen` generation loops.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use once_cell::sync::Lazy;

const LRU_CAP: usize = 16;
const VOCAB_CHUNK: usize = 16_384;

#[derive(Clone)]
pub struct CachedPrejudice {
    pub key: String,
    /// Target normalized vector `[hidden]` on CPU (F32) for single-row cold rescue prejudice checks.
    pub target_norm_cpu: Vec<f32>,
    /// Prejudice values `[K]` aligned to `kept_ids` (or `[V]` when untrimmed) on device (F32).
    /// Always >= 0.0 (`exp(15 * relu(cos - 0.65)) - 1`), preserving upper-bound admissibility!
    pub values: Tensor,
}

static CACHE: Lazy<Mutex<VecDeque<CachedPrejudice>>> =
    Lazy::new(|| Mutex::new(VecDeque::with_capacity(LRU_CAP)));

/// Compute exact scalar prejudice penalty for a single decoded weight row `w_row` (CPU F32)
/// in a single pass over `w_row` and `target_norm`. Guaranteed `>= 0.0`.
#[inline]
pub fn scalar_prejudice(w_row: &[f32], target_norm: &[f32]) -> f32 {
    if w_row.len() != target_norm.len() || w_row.is_empty() {
        return 0.0;
    }
    let (dot_wt, norm_w_sq) = embed_cpu_kernels::dot_and_norm_sq_f32_simd(w_row, target_norm);
    embed_cpu_kernels::prejudice_from_stats(dot_wt, norm_w_sq)
}

/// Look up or build a chunked F32 prejudice vector using a row-block provider closure
/// `rows_f32(offset, count) -> Result<Tensor>` (shape `[count, hidden]`).
pub fn get_or_build<F>(
    cache_key: &str,
    target_vec: &Tensor,
    num_rows: usize,
    device: &Device,
    mut rows_f32: F,
) -> Result<CachedPrejudice>
where
    F: FnMut(usize, usize) -> Result<Tensor>,
{
    {
        let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = g.iter().position(|e| e.key == cache_key && e.values.device().same_device(device)) {
            let entry = g.remove(pos).unwrap();
            g.push_back(entry.clone());
            return Ok(entry);
        }
    }

    let target_f32 = target_vec.to_dtype(DType::F32)?;
    let target_norm = target_f32.sqr()?.sum_all()?.sqrt()?;
    let target_normalized = target_f32.broadcast_div(&target_norm)?;
    let target_norm_cpu = target_normalized.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let target_col = target_normalized.unsqueeze(1)?.contiguous()?;

    let mut sim_parts: Vec<Tensor> = Vec::with_capacity(num_rows / VOCAB_CHUNK + 1);
    let mut off = 0usize;
    while off < num_rows {
        let take = (num_rows - off).min(VOCAB_CHUNK);
        let blk = rows_f32(off, take)?.to_dtype(DType::F32)?;
        let blk_norm = blk.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let blk_normalized = blk.broadcast_div(&blk_norm)?;
        sim_parts.push(blk_normalized.matmul(&target_col)?.squeeze(1)?);
        off += take;
    }
    let sim = Tensor::cat(&sim_parts, 0)?;
    let threshold = Tensor::new(0.65f32, device)?;
    let one = Tensor::new(1.0f32, device)?;
    let sim_relu = sim.broadcast_sub(&threshold)?.relu()?;
    let values = sim_relu.affine(15.0, 0.0)?.exp()?.broadcast_sub(&one)?;

    let entry = CachedPrejudice {
        key: cache_key.to_string(),
        target_norm_cpu,
        values,
    };
    {
        let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() >= LRU_CAP {
            g.pop_front();
        }
        g.push_back(entry.clone());
    }
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_prejudice_is_always_nonnegative() {
        let t = vec![1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(scalar_prejudice(&[0.5, 0.866, 0.0, 0.0], &t), 0.0);
        assert!(scalar_prejudice(&[1.0, 0.0, 0.0, 0.0], &t) > 100.0);
        assert_eq!(scalar_prejudice(&[-1.0, 0.0, 0.0, 0.0], &t), 0.0);
    }
}
