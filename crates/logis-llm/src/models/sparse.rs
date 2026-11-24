//! Neural sparse MLM head + long-document sliding-window pooling (Phase 4 – Part 2 D2).
//!
//! Shares the exact `granite-embedding-97m-r2` encoder backbone so dense CLS embeddings
//! remain bit-identical (`E0`), while optionally producing `log(1 + relu(logits))`
//! max-pooled sparse token weights (Sparseup-fr / SPLADE style).

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMode {
    Off,
    Bm25,
    Neural,
    Hybrid,
}

impl SparseMode {
    pub fn from_env() -> Self {
        match std::env::var("LOGIS_SPARSE_MODE")
            .unwrap_or_else(|_| "bm25".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" | "none" => Self::Off,
            "neural" | "splade" => Self::Neural,
            "hybrid" | "both" => Self::Hybrid,
            _ => Self::Bm25,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Bm25 => "bm25",
            Self::Neural => "neural",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SparseConfig {
    pub top_k_doc: usize,
    pub top_k_query: usize,
    pub max_window: usize,
    pub overlap: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            top_k_doc: 128,
            top_k_query: 48,
            max_window: 510,
            overlap: 64,
        }
    }
}

/// Split long token sequences into overlapping `[CLS] ... [SEP]` windows for sparse max-pooling.
pub fn split_sparse_windows(
    ids: &[u32],
    special_mask: &[u32],
    max_window: usize,
    overlap: usize,
) -> Vec<Vec<u32>> {
    let n = ids.len();
    let wrapped = n >= 2 && special_mask.first() == Some(&1) && special_mask.last() == Some(&1);
    let max_body = max_window.max(8);
    if n <= max_body + 2 || !wrapped {
        return vec![ids.to_vec()];
    }
    let head = ids[0];
    let tail = ids[n - 1];
    let body = &ids[1..n - 1];
    let step = max_body.saturating_sub(overlap).max(1);
    let mut windows = Vec::new();
    let mut pos = 0usize;
    while pos < body.len() {
        let end = (pos + max_body).min(body.len());
        let mut w = Vec::with_capacity(end - pos + 2);
        w.push(head);
        w.extend_from_slice(&body[pos..end]);
        w.push(tail);
        windows.push(w);
        if end == body.len() {
            break;
        }
        pos += step;
    }
    windows
}

/// Optional neural MLM sparse projection head (`decoder` or tied embedding rows).
pub struct SparseMlmHead {
    decoder: Linear,
    fold_map: Option<Vec<u32>>,
}

impl SparseMlmHead {
    /// Load MLM decoder weights if present in `weights_path` (returns `Ok(None)` cleanly if absent).
    pub fn try_load(weights_path: &Path, hidden_size: usize, vocab_size: usize, dtype: DType, device: &Device) -> Result<Option<Self>> {
        if !weights_path.exists() {
            return Ok(None);
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, device)? };
        let w_opt = vb
            .pp("decoder")
            .get((vocab_size, hidden_size), "weight")
            .ok()
            .or_else(|| vb.pp("cls.predictions.decoder").get((vocab_size, hidden_size), "weight").ok());
        let Some(weight) = w_opt else {
            return Ok(None);
        };
        let bias = vb
            .pp("decoder")
            .get(vocab_size, "bias")
            .ok()
            .or_else(|| vb.pp("cls.predictions.decoder").get(vocab_size, "bias").ok());
        Ok(Some(Self {
            decoder: Linear::new(weight, bias),
            fold_map: None,
        }))
    }

    pub fn with_fold_map(mut self, fold_map: Vec<u32>) -> Self {
        self.fold_map = Some(fold_map);
        self
    }

    /// Compute SPLADE-style `max_t log(1 + relu(decoder(h_t)))` over `[1, seq_len, H]` -> top-K `(token_id, weight)`.
    pub fn project_top_k(&self, hidden: &Tensor, top_k: usize) -> Result<Vec<(u32, f32)>> {
        let logits = self.decoder.forward(hidden)?.to_dtype(DType::F32)?;
        // log(1 + relu(logits)) then max-pool along sequence axis (dim 1)
        let pos = logits.relu()?;
        let one = Tensor::ones_like(&pos)?;
        let sat = pos.add(&one)?.log()?;
        let pooled = sat.max(1)?.squeeze(0)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;

        let mut folded: HashMap<u32, f32> = HashMap::new();
        for (i, &v) in pooled.iter().enumerate() {
            if !v.is_finite() || v <= 1e-4 {
                continue;
            }
            let tid = self
                .fold_map
                .as_ref()
                .and_then(|fm| fm.get(i).copied())
                .unwrap_or(i as u32);
            let e = folded.entry(tid).or_insert(0.0);
            if v > *e {
                *e = v;
            }
        }
        let mut pairs: Vec<(u32, f32)> = folded.into_iter().collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(top_k);
        Ok(pairs)
    }
}

/// Merge multiple window sparse vectors via element-wise max pooling and top-K truncation.
pub fn merge_sparse_windows(windows: &[Vec<(u32, f32)>], top_k: usize) -> Vec<(u32, f32)> {
    let mut acc: HashMap<u32, f32> = HashMap::new();
    for w in windows {
        for &(tid, val) in w {
            let e = acc.entry(tid).or_insert(0.0);
            if val > *e {
                *e = val;
            }
        }
    }
    let mut out: Vec<(u32, f32)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(top_k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_windows_preserve_cls_sep_and_overlap() {
        let mut ids = vec![1u32];
        ids.extend(10..1110u32);
        ids.push(2u32);
        let mut mask = vec![0u32; ids.len()];
        mask[0] = 1;
        *mask.last_mut().unwrap() = 1;

        let wins = split_sparse_windows(&ids, &mask, 510, 64);
        assert_eq!(wins.len(), 3);
        for w in &wins {
            assert_eq!(w[0], 1);
            assert_eq!(*w.last().unwrap(), 2);
            assert!(w.len() <= 512);
        }
        // Check 64-token overlap between window 0 and window 1
        assert_eq!(&wins[0][wins[0].len() - 65..wins[0].len() - 1], &wins[1][1..65]);
    }
}
