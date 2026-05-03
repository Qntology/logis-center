//! Tensor parallelism utilities for OxiBonsai.
//!
//! Splits weight matrices across shards (threads/devices).
//! Two parallelism modes are supported:
//!
//! * **Column-parallel** — split along the output dimension; each shard
//!   produces a partial output that is all-gathered to form the full result.
//! * **Row-parallel** — split along the input dimension; each shard takes the
//!   full input and produces a partial sum that is all-reduced (summed) to form
//!   the final result.

use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// ShardDim
// ─────────────────────────────────────────────────────────────────────────────

/// Which dimension of the weight matrix is split across shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardDim {
    /// Split along the output dimension (column-parallel).
    Output,
    /// Split along the input dimension (row-parallel).
    Input,
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which portion of a tensor a particular shard holds.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardInfo {
    /// Zero-based index of this shard.
    pub shard_id: usize,
    /// Total number of shards.
    pub num_shards: usize,
    /// Which dimension is being split.
    pub dim: ShardDim,
    /// First row/column index owned by this shard.
    pub offset: usize,
    /// Number of rows/columns owned by this shard.
    pub size: usize,
}

impl ShardInfo {
    /// Create a `ShardInfo` by evenly dividing `total_size` across
    /// `num_shards`.  The last shard receives any remainder rows/columns so
    /// that all elements are covered.
    ///
    /// # Panics
    ///
    /// Panics (in debug mode) if `num_shards` is zero.
    pub fn new(shard_id: usize, num_shards: usize, total_size: usize, dim: ShardDim) -> Self {
        assert!(num_shards > 0, "num_shards must be > 0");
        let base = total_size / num_shards;
        let remainder = total_size % num_shards;
        let offset = shard_id * base;
        // Last shard absorbs the remainder.
        let size = if shard_id + 1 == num_shards {
            base + remainder
        } else {
            base
        };
        Self {
            shard_id,
            num_shards,
            dim,
            offset,
            size,
        }
    }

    /// Extract the portion of a flat row-major weight matrix that belongs to
    /// this shard.
    ///
    /// * For [`ShardDim::Output`] — returns rows `[offset, offset+size)`.
    /// * For [`ShardDim::Input`] — returns column slice within each row
    ///   `[offset, offset+size)`.  The returned slice is **not** contiguous
    ///   across rows; callers that need individual row slices should iterate
    ///   over rows themselves.  This variant returns the flat sub-block only
    ///   when the shard covers the full column range (size == cols), which is
    ///   the common case when `num_shards == 1`.  For the general column
    ///   sub-selection case the partition helpers below handle copying.
    ///
    /// In practice this method is used for the output-dimension (row) slice.
    pub fn slice_weights<'a>(&self, weights: &'a [f32], rows: usize, cols: usize) -> &'a [f32] {
        match self.dim {
            ShardDim::Output => {
                let start = self.offset * cols;
                let end = (self.offset + self.size) * cols;
                &weights[start..end.min(rows * cols)]
            }
            ShardDim::Input => {
                // For input-parallel the shard covers full rows but only a
                // column sub-range.  We return the full weight slice here;
                // callers must use `offset`/`size` to extract columns per row.
                let _ = rows;
                weights
            }
        }
    }

    /// Returns `true` if this is the final shard.
    #[inline]
    pub fn is_last_shard(&self) -> bool {
        self.shard_id + 1 == self.num_shards
    }

    /// Returns `true` if `idx` falls within `[offset, offset + size)`.
    #[inline]
    pub fn covers_index(&self, idx: usize) -> bool {
        idx >= self.offset && idx < self.offset + self.size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorParallelMode
// ─────────────────────────────────────────────────────────────────────────────

/// Parallelism strategy for tensor-parallel linear layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelMode {
    /// Column-parallel: each shard produces partial output; combine with
    /// all-gather (concatenation).
    ColumnParallel,
    /// Row-parallel: each shard takes the full input and produces a partial
    /// sum; combine with all-reduce (element-wise sum).
    RowParallel,
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardedLinear
// ─────────────────────────────────────────────────────────────────────────────

/// A single shard of a linear (fully-connected) layer.
///
/// Holds a contiguous slice of the weight matrix and, optionally, the
/// corresponding bias vector for this shard.
pub struct ShardedLinear {
    /// Shard weights only (row-major, `shard_out × shard_in` elements).
    pub weights: Vec<f32>,
    /// Optional bias slice for this shard.
    pub bias: Option<Vec<f32>>,
    /// Metadata about which portion of the full matrix this shard holds.
    pub shard: ShardInfo,
    /// Full input feature count (i.e., columns of the complete weight matrix).
    pub in_features: usize,
    /// Full output feature count (i.e., rows of the complete weight matrix).
    pub out_features: usize,
}

impl ShardedLinear {
    /// Construct a `ShardedLinear` without a bias.
    pub fn new(
        weights: Vec<f32>,
        shard: ShardInfo,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weights,
            bias: None,
            shard,
            in_features,
            out_features,
        }
    }

    /// Attach a bias vector to this shard (builder-style).
    pub fn with_bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Compute this shard's contribution to the layer output.
    ///
    /// * **Column-parallel** — performs `output[shard_out] = W_shard × input`
    ///   and returns a vector of length `shard.size` (shard output features).
    /// * **Row-parallel** — performs `output[out_features] = W_full × input_shard`
    ///   where each shard operates on a slice of the input and the results must
    ///   be all-reduced by the caller.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        match self.shard.dim {
            ShardDim::Output => {
                // Column-parallel: W_shard is (shard.size × in_features).
                let shard_out = self.shard.size;
                let in_f = self.in_features;
                let mut out = vec![0.0f32; shard_out];
                for (row, o) in out.iter_mut().enumerate() {
                    let row_start = row * in_f;
                    let mut acc = 0.0f32;
                    for (col, &inp_col) in input.iter().enumerate().take(in_f) {
                        acc += self.weights[row_start + col] * inp_col;
                    }
                    if let Some(ref b) = self.bias {
                        acc += b[row];
                    }
                    *o = acc;
                }
                out
            }
            ShardDim::Input => {
                // Row-parallel: W_shard is (out_features × shard.size).
                // Each shard operates on input[offset..offset+size].
                let shard_in = self.shard.size;
                let in_offset = self.shard.offset;
                let out_f = self.out_features;
                let mut out = vec![0.0f32; out_f];
                for row in 0..out_f {
                    let row_start = row * shard_in;
                    let mut acc = 0.0f32;
                    for col in 0..shard_in {
                        acc += self.weights[row_start + col] * input[in_offset + col];
                    }
                    // Bias applied on the last shard only (to avoid double-adding).
                    if self.shard.is_last_shard() {
                        if let Some(ref b) = self.bias {
                            acc += b[row];
                        }
                    }
                    out[row] = acc;
                }
                out
            }
        }
    }

    /// Number of output features produced by this shard's `forward` call.
    pub fn shard_output_size(&self) -> usize {
        match self.shard.dim {
            ShardDim::Output => self.shard.size,
            ShardDim::Input => self.out_features,
        }
    }

    /// Memory consumed by this shard's weight and bias data in bytes.
    pub fn memory_bytes(&self) -> usize {
        let w = self.weights.len() * std::mem::size_of::<f32>();
        let b = self
            .bias
            .as_ref()
            .map(|b| b.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        w + b
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Partition helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Partition a full weight matrix into column-parallel shards.
///
/// The output dimension (`out_features`) is divided evenly across shards.
/// Shard `i` owns rows `[offset_i, offset_i + size_i)` of the weight matrix.
///
/// * `weights`     — row-major, shape `[out_features × in_features]`
/// * `bias`        — optional, shape `[out_features]`
/// * `in_features` — number of input features (columns)
/// * `out_features`— number of output features (rows)
/// * `num_shards`  — how many shards to create
pub fn partition_column_parallel(
    weights: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    num_shards: usize,
) -> Vec<ShardedLinear> {
    (0..num_shards)
        .map(|shard_id| {
            let info = ShardInfo::new(shard_id, num_shards, out_features, ShardDim::Output);
            // Extract contiguous row slice.
            let row_start = info.offset * in_features;
            let row_end = (info.offset + info.size) * in_features;
            let shard_weights = weights[row_start..row_end].to_vec();
            // Extract bias slice if present.
            let shard_bias = bias.map(|b| b[info.offset..info.offset + info.size].to_vec());
            let mut sl = ShardedLinear::new(shard_weights, info, in_features, out_features);
            if let Some(b) = shard_bias {
                sl = sl.with_bias(b);
            }
            sl
        })
        .collect()
}

/// Partition a full weight matrix into row-parallel shards.
///
/// The input dimension (`in_features`) is divided evenly across shards.
/// Shard `i` owns columns `[offset_i, offset_i + size_i)` of each row.
///
/// * `weights`     — row-major, shape `[out_features × in_features]`
/// * `bias`        — optional, shape `[out_features]` (applied by last shard)
/// * `in_features` — number of input features (columns)
/// * `out_features`— number of output features (rows)
/// * `num_shards`  — how many shards to create
pub fn partition_row_parallel(
    weights: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    num_shards: usize,
) -> Vec<ShardedLinear> {
    (0..num_shards)
        .map(|shard_id| {
            let info = ShardInfo::new(shard_id, num_shards, in_features, ShardDim::Input);
            // Copy the sub-columns for each row.
            let mut shard_weights = Vec::with_capacity(out_features * info.size);
            for row in 0..out_features {
                let row_base = row * in_features;
                shard_weights.extend_from_slice(
                    &weights[row_base + info.offset..row_base + info.offset + info.size],
                );
            }
            // Bias only on the last shard (applied during forward).
            let shard_bias = if info.is_last_shard() {
                bias.map(|b| b.to_vec())
            } else {
                None
            };
            let mut sl = ShardedLinear::new(shard_weights, info, in_features, out_features);
            if let Some(b) = shard_bias {
                sl = sl.with_bias(b);
            }
            sl
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Collective communication primitives
// ─────────────────────────────────────────────────────────────────────────────

/// All-reduce: element-wise sum of partial results from all row-parallel shards.
///
/// All `partials` slices must have the same length.
pub fn all_reduce(partials: &[Vec<f32>]) -> Vec<f32> {
    if partials.is_empty() {
        return Vec::new();
    }
    let len = partials[0].len();
    let mut result = vec![0.0f32; len];
    for partial in partials {
        for (r, &p) in result.iter_mut().zip(partial.iter()) {
            *r += p;
        }
    }
    result
}

/// All-gather: concatenate outputs from column-parallel shards in shard order.
pub fn all_gather(partials: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = partials.iter().map(|v| v.len()).sum();
    let mut result = Vec::with_capacity(total);
    for partial in partials {
        result.extend_from_slice(partial);
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Run a tensor-parallel forward pass across all shards using Rayon.
///
/// * [`TensorParallelMode::ColumnParallel`] — shards compute partial outputs
///   that are all-gathered (concatenated) to produce the full output.
/// * [`TensorParallelMode::RowParallel`] — shards compute partial sums that
///   are all-reduced (summed) to produce the full output.
pub fn tensor_parallel_forward(
    shards: &[ShardedLinear],
    input: &[f32],
    parallel_mode: TensorParallelMode,
) -> Vec<f32> {
    // Compute each shard's output in parallel.
    let partials: Vec<Vec<f32>> = shards
        .par_iter()
        .map(|shard| shard.forward(input))
        .collect();

    match parallel_mode {
        TensorParallelMode::ColumnParallel => all_gather(&partials),
        TensorParallelMode::RowParallel => all_reduce(&partials),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sharding plan
// ─────────────────────────────────────────────────────────────────────────────

/// Sharding assignment for a single named layer.
#[derive(Debug, Clone)]
pub struct LayerSharding {
    /// Fully-qualified layer name (e.g. `"blk.0.attn_q"`).
    pub layer_name: String,
    /// Parallelism mode for this layer.
    pub mode: TensorParallelMode,
    /// Number of shards for this layer.
    pub num_shards: usize,
}

/// A complete sharding plan describing how every layer in a model is split.
pub struct ShardingPlan {
    /// Default number of shards (used when adding layers without explicit count).
    pub num_shards: usize,
    /// Per-layer assignments, in insertion order.
    pub layer_assignments: Vec<LayerSharding>,
}

impl ShardingPlan {
    /// Create an empty sharding plan with a global `num_shards` default.
    pub fn new(num_shards: usize) -> Self {
        Self {
            num_shards,
            layer_assignments: Vec::new(),
        }
    }

    /// Append a layer assignment using the plan's default `num_shards`.
    pub fn add_layer(&mut self, name: &str, mode: TensorParallelMode) {
        self.layer_assignments.push(LayerSharding {
            layer_name: name.to_owned(),
            mode,
            num_shards: self.num_shards,
        });
    }

    /// Build a standard transformer sharding plan for a model with
    /// `num_layers` Transformer blocks.
    ///
    /// Convention (Qwen3 / LLaMA naming):
    /// * `attn_q`, `attn_k`, `attn_v`, `ffn_gate`, `ffn_up` → `ColumnParallel`
    /// * `attn_output`, `ffn_down` → `RowParallel`
    pub fn standard_transformer_plan(num_shards: usize, num_layers: usize) -> Self {
        let mut plan = Self::new(num_shards);
        for layer in 0..num_layers {
            let prefix = format!("blk.{layer}");
            for suffix in &["attn_q", "attn_k", "attn_v"] {
                plan.add_layer(
                    &format!("{prefix}.{suffix}"),
                    TensorParallelMode::ColumnParallel,
                );
            }
            plan.add_layer(
                &format!("{prefix}.attn_output"),
                TensorParallelMode::RowParallel,
            );
            for suffix in &["ffn_gate", "ffn_up"] {
                plan.add_layer(
                    &format!("{prefix}.{suffix}"),
                    TensorParallelMode::ColumnParallel,
                );
            }
            plan.add_layer(
                &format!("{prefix}.ffn_down"),
                TensorParallelMode::RowParallel,
            );
        }
        plan
    }

    /// Look up the sharding assignment for a layer by name.
    pub fn get(&self, layer_name: &str) -> Option<&LayerSharding> {
        self.layer_assignments
            .iter()
            .find(|a| a.layer_name == layer_name)
    }

    /// Rough estimate of total weight memory across all sharded layers.
    ///
    /// Uses simplified transformer weight dimensions:
    /// * Attention: 4 projection matrices of shape `[hidden × hidden]`
    /// * FFN: gate+up (2 × `[intermediate × hidden]`) + down `[hidden × intermediate]`
    ///
    /// Divides by `num_shards` to reflect the per-device footprint and returns
    /// total bytes (assuming `f32`).
    pub fn total_weight_memory_estimate(
        &self,
        hidden: usize,
        intermediate: usize,
        num_layers: usize,
    ) -> usize {
        // Per-layer parameter count (full precision).
        // Attention: Q, K, V, O each hidden×hidden.
        let attn_params = 4 * hidden * hidden;
        // FFN: gate, up each intermediate×hidden; down hidden×intermediate.
        let ffn_params = 2 * intermediate * hidden + hidden * intermediate;
        let total_params = num_layers * (attn_params + ffn_params);
        // Each device holds 1/num_shards of the weights.
        let per_device = total_params / self.num_shards.max(1);
        per_device * std::mem::size_of::<f32>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShardInfo ────────────────────────────────────────────────────────────

    #[test]
    fn test_shard_info_even_split() {
        let info = ShardInfo::new(0, 4, 16, ShardDim::Output);
        assert_eq!(info.offset, 0);
        assert_eq!(info.size, 4);

        let info2 = ShardInfo::new(3, 4, 16, ShardDim::Output);
        assert_eq!(info2.offset, 12);
        assert_eq!(info2.size, 4);
    }

    #[test]
    fn test_shard_info_uneven_split_last_gets_remainder() {
        // 10 elements, 3 shards → base=3, remainder=1 → sizes [3, 3, 4]
        let s0 = ShardInfo::new(0, 3, 10, ShardDim::Output);
        let s1 = ShardInfo::new(1, 3, 10, ShardDim::Output);
        let s2 = ShardInfo::new(2, 3, 10, ShardDim::Output);
        assert_eq!(s0.size, 3);
        assert_eq!(s1.size, 3);
        assert_eq!(s2.size, 4); // last shard gets remainder
        assert_eq!(s0.offset + s0.size, s1.offset);
        assert_eq!(s1.offset + s1.size, s2.offset);
        assert_eq!(s2.offset + s2.size, 10);
    }

    #[test]
    fn test_shard_info_covers_index() {
        let info = ShardInfo::new(1, 4, 16, ShardDim::Output);
        // offset=4, size=4 → covers 4..8
        assert!(!info.covers_index(3));
        assert!(info.covers_index(4));
        assert!(info.covers_index(7));
        assert!(!info.covers_index(8));
    }

    // ── partition_column_parallel ────────────────────────────────────────────

    #[test]
    fn test_partition_column_parallel_count() {
        let weights = vec![1.0f32; 8 * 4]; // 8 out × 4 in
        let shards = partition_column_parallel(&weights, None, 4, 8, 4);
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn test_partition_column_parallel_output_sizes() {
        let weights = vec![1.0f32; 8 * 4];
        let shards = partition_column_parallel(&weights, None, 4, 8, 4);
        for shard in &shards {
            // Each shard: 2 output rows × 4 input cols
            assert_eq!(shard.weights.len(), 2 * 4);
            assert_eq!(shard.shard_output_size(), 2);
        }
    }

    // ── partition_row_parallel ───────────────────────────────────────────────

    #[test]
    fn test_partition_row_parallel_count() {
        let weights = vec![1.0f32; 4 * 8]; // 4 out × 8 in
        let shards = partition_row_parallel(&weights, None, 8, 4, 4);
        assert_eq!(shards.len(), 4);
    }

    // ── ShardedLinear::forward (column-parallel) ─────────────────────────────

    #[test]
    fn test_sharded_linear_forward_column() {
        // 1 shard covering all 2 output rows, 3 input cols.
        // W = [[1,0,0],[0,1,0]] → output = [input[0], input[1]]
        let weights = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let info = ShardInfo::new(0, 1, 2, ShardDim::Output);
        let sl = ShardedLinear::new(weights, info, 3, 2);
        let input = vec![5.0f32, 7.0, 9.0];
        let out = sl.forward(&input);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    // ── all_reduce ───────────────────────────────────────────────────────────

    #[test]
    fn test_all_reduce_sums_correctly() {
        let p1 = vec![1.0f32, 2.0, 3.0];
        let p2 = vec![4.0f32, 5.0, 6.0];
        let p3 = vec![7.0f32, 8.0, 9.0];
        let result = all_reduce(&[p1, p2, p3]);
        assert_eq!(result, vec![12.0f32, 15.0, 18.0]);
    }

    // ── all_gather ───────────────────────────────────────────────────────────

    #[test]
    fn test_all_gather_concatenates() {
        let p1 = vec![1.0f32, 2.0];
        let p2 = vec![3.0f32, 4.0];
        let p3 = vec![5.0f32, 6.0];
        let result = all_gather(&[p1, p2, p3]);
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── tensor_parallel_forward ──────────────────────────────────────────────

    #[test]
    fn test_tensor_parallel_forward_column() {
        // 4 output rows, 2 input cols, 2 shards (each 2 output rows).
        // W = identity-like: [[1,0],[0,1],[1,0],[0,1]]
        let weights = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let shards = partition_column_parallel(&weights, None, 2, 4, 2);
        let input = vec![3.0f32, 7.0];
        let out = tensor_parallel_forward(&shards, &input, TensorParallelMode::ColumnParallel);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_parallel_forward_row() {
        // 2 output rows, 4 input cols, 2 shards (each covers 2 input cols).
        // W = [[1,1,1,1],[2,2,2,2]] → output = [sum(input), 2*sum(input)]
        let weights = vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let shards = partition_row_parallel(&weights, None, 4, 2, 2);
        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // sum = 10
        let out = tensor_parallel_forward(&shards, &input, TensorParallelMode::RowParallel);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 10.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-5, "out[1]={}", out[1]);
    }

    // ── ShardingPlan ─────────────────────────────────────────────────────────

    #[test]
    fn test_sharding_plan_standard_transformer() {
        let plan = ShardingPlan::standard_transformer_plan(4, 2);
        // 2 layers × 7 assignments per layer = 14 total.
        assert_eq!(plan.layer_assignments.len(), 14);
    }

    #[test]
    fn test_sharding_plan_get_layer() {
        let plan = ShardingPlan::standard_transformer_plan(4, 3);
        let q = plan.get("blk.0.attn_q").expect("layer should exist");
        assert_eq!(q.mode, TensorParallelMode::ColumnParallel);
        assert_eq!(q.num_shards, 4);

        let down = plan.get("blk.2.ffn_down").expect("layer should exist");
        assert_eq!(down.mode, TensorParallelMode::RowParallel);

        assert!(plan.get("blk.99.ffn_up").is_none());
    }
}
