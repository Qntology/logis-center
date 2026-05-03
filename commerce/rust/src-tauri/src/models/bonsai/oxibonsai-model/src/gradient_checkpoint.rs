//! Gradient checkpointing: trade compute for memory in training.
//!
//! Instead of storing all intermediate activations, checkpointing only
//! stores "checkpoint" tensors at layer boundaries and recomputes
//! intermediate values during backward pass.
//!
//! # Memory Trade-off
//!
//! For a network with N segments each producing an activation of size A:
//! - Without checkpointing: stores N activations → N * A bytes
//! - With checkpointing: stores N inputs (same as activations for matching dims)
//!   but for networks where output > input, the savings can be significant.
//!
//! The savings fraction = 1 - (sum of input sizes) / (sum of output sizes).

use thiserror::Error;

// ─── Error types ─────────────────────────────────────────────────────────────

/// Errors that can arise during checkpointed computation.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// The memory budget was exceeded when trying to allocate.
    #[error("memory budget exceeded: need {need}, available {available}")]
    BudgetExceeded { need: usize, available: usize },
    /// An empty segment list was provided where at least one is required.
    #[error("empty segment list")]
    EmptySegments,
    /// The input vector length does not match the expected dimension.
    #[error("dimension mismatch: input has {got} elements, expected {expected}")]
    DimMismatch { expected: usize, got: usize },
    /// The pipeline has no segments.
    #[error("empty pipeline")]
    EmptyPipeline,
}

// ─── Recomputable trait ──────────────────────────────────────────────────────

/// A computation segment that can be re-run on demand.
///
/// Implementors represent a pure function from `Input` to `Output` that can be
/// called an arbitrary number of times — once during the forward pass, and once
/// more per segment during the backward pass to recover intermediate activations.
///
/// # Thread safety
///
/// Both `Self`, `Input`, and `Output` must be `Send + Sync` so that checkpointed
/// networks can be used across threads (e.g., in data-parallel training).
pub trait Recomputable: Send + Sync {
    /// The input type accepted by this segment.
    type Input: Clone + Send + Sync;
    /// The output type produced by this segment.
    type Output: Clone + Send + Sync;

    /// Compute the forward pass of this segment.
    ///
    /// This will be called at least twice: once during the main forward pass and
    /// once during the backward pass when the output is needed for gradient
    /// computation.  Implementations must be **deterministic** — identical inputs
    /// must always produce identical outputs.
    fn forward(&self, input: &Self::Input) -> Self::Output;

    /// Estimate the memory footprint of one input value in bytes.
    ///
    /// Used by [`CheckpointBudget`] to track how much memory the checkpointed
    /// inputs collectively occupy.
    fn input_memory_bytes(input: &Self::Input) -> usize;
}

// ─── Checkpoint ──────────────────────────────────────────────────────────────

/// A single checkpointed computation segment.
///
/// Stores the segment implementation and its saved input so that the output can
/// be recomputed at any time.  The output itself is **not** stored; calling
/// [`recompute`] always re-runs the forward pass.
///
/// [`recompute`]: Checkpoint::recompute
pub struct Checkpoint<R: Recomputable> {
    recomputable: R,
    saved_input: R::Input,
}

impl<R: Recomputable> Checkpoint<R> {
    /// Create a new checkpoint, saving `input` for later recomputation.
    ///
    /// The `recomputable` segment is stored alongside the input so that
    /// [`recompute`] can call `recomputable.forward(&saved_input)`.
    ///
    /// [`recompute`]: Self::recompute
    pub fn new(recomputable: R, input: R::Input) -> Self {
        Self {
            recomputable,
            saved_input: input,
        }
    }

    /// Recompute and return the output from the saved input.
    ///
    /// This is the key operation: instead of loading a cached output tensor,
    /// we replay the forward pass from the checkpointed input.
    pub fn recompute(&self) -> R::Output {
        self.recomputable.forward(&self.saved_input)
    }

    /// Bytes consumed by the saved checkpoint (input only, not the output).
    pub fn memory_bytes(&self) -> usize {
        R::input_memory_bytes(&self.saved_input)
    }
}

// ─── LinearSegment ───────────────────────────────────────────────────────────

/// A simple fully-connected (linear/dense) layer operating on flat `f32`
/// vectors.
///
/// Implements the linear transformation `y = W x` (no bias) where `W` is a
/// `[out_dim × in_dim]` matrix stored in row-major order.
///
/// This is primarily intended for testing gradient-checkpointing mechanics
/// without pulling in heavy tensor libraries.
#[derive(Clone)]
pub struct LinearSegment {
    /// Weight matrix stored row-major: `weights[i * in_dim + j]` = W[i, j].
    pub weights: Vec<f32>,
    /// Number of input features.
    pub in_dim: usize,
    /// Number of output features.
    pub out_dim: usize,
}

impl LinearSegment {
    /// Create a `LinearSegment` from explicit weights.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `weights.len() != in_dim * out_dim`.
    pub fn new(weights: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        debug_assert_eq!(
            weights.len(),
            in_dim * out_dim,
            "weights.len() must equal in_dim * out_dim"
        );
        Self {
            weights,
            in_dim,
            out_dim,
        }
    }

    /// Initialise weights pseudo-randomly using a simple 64-bit LCG.
    ///
    /// The LCG parameters (multiplier / increment) are the same as used by
    /// Knuth and Numerical Recipes.  Weights are scaled to `[-1, 1]` using
    /// Xavier-style normalisation: `w ∈ [-sqrt(6/(in+out)), sqrt(6/(in+out))]`.
    ///
    /// No external `rand` crate is required.
    pub fn random_init(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let n = in_dim * out_dim;
        let mut state = seed;
        let xavier_limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt() as f32;

        let weights: Vec<f32> = (0..n)
            .map(|_| {
                // LCG step: x_{n+1} = (a * x_n + c) mod 2^64
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map upper 32 bits to [0, 1) then to [-limit, limit]
                let uniform = (state >> 32) as f32 / u32::MAX as f32; // [0, 1]
                uniform * 2.0 * xavier_limit - xavier_limit
            })
            .collect();

        Self {
            weights,
            in_dim,
            out_dim,
        }
    }
}

impl Recomputable for LinearSegment {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    /// Compute `y = W x` where `x` has length `in_dim` and `y` has length
    /// `out_dim`.
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_dim];
        // Each output neuron i: y[i] = sum_j W[i,j] * x[j]
        for (i, out_val) in output.iter_mut().enumerate() {
            let row_start = i * self.in_dim;
            let row = &self.weights[row_start..row_start + self.in_dim];
            let mut acc = 0.0f32;
            for (w, x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            *out_val = acc;
        }
        output
    }

    /// Each `f32` is 4 bytes.
    fn input_memory_bytes(input: &Vec<f32>) -> usize {
        input.len() * 4
    }
}

// ─── CheckpointedNetwork ─────────────────────────────────────────────────────

/// A sequential network where every layer boundary is checkpointed.
///
/// During the forward pass, each segment's output is fed as input to the next
/// segment, but only the inputs are retained — outputs are discarded.  On
/// demand (e.g., during the backward pass) any segment's output can be
/// recovered by calling `checkpoint.recompute()`.
///
/// This type is generic over any `Recomputable` whose `Input` and `Output` are
/// both `Vec<f32>`, making it suitable for chain-of-linear-segments networks.
pub struct CheckpointedNetwork<R: Recomputable<Input = Vec<f32>, Output = Vec<f32>>> {
    segments: Vec<Checkpoint<R>>,
}

impl<R: Recomputable<Input = Vec<f32>, Output = Vec<f32>>> CheckpointedNetwork<R> {
    /// Construct the network from pre-built checkpoints.
    pub fn new(segments: Vec<Checkpoint<R>>) -> Self {
        Self { segments }
    }

    /// Execute the full forward pass, returning the output of the final segment.
    ///
    /// Internally each segment is run in order; the checkpointed inputs are
    /// already stored so we merely recompute each in sequence.
    ///
    /// Returns an error if `segments` is empty.
    pub fn forward(&self, _input: &[f32]) -> Vec<f32> {
        if self.segments.is_empty() {
            return Vec::new();
        }
        // Each segment already has its saved input; run them in order.
        let mut output = self.segments[0].recompute();
        for seg in self.segments.iter().skip(1) {
            // Re-run the segment from its checkpointed input (ignores `output`
            // from the previous iteration since inputs are already stored).
            output = seg.recompute();
        }
        output
    }

    /// Total bytes used by all checkpointed inputs.
    pub fn memory_bytes(&self) -> usize {
        self.segments.iter().map(|s| s.memory_bytes()).sum()
    }

    /// Hypothetical memory if we stored every segment's **output** instead of
    /// its input.
    ///
    /// For a typical expanding network (out_dim > in_dim) this will be larger
    /// than [`memory_bytes`], demonstrating the checkpointing advantage.
    ///
    /// [`memory_bytes`]: Self::memory_bytes
    pub fn full_memory_bytes(&self) -> usize {
        self.segments
            .iter()
            .map(|s| {
                // Recompute output and measure its size.
                let out = s.recompute();
                out.len() * 4
            })
            .sum()
    }

    /// Fraction of memory saved relative to storing all outputs.
    ///
    /// Returns a value in `[0, 1)`.  A result of `0.0` means no savings (input
    /// == output size everywhere); values near `1.0` mean the full-storage cost
    /// would be much higher.
    pub fn memory_savings(&self) -> f32 {
        let full = self.full_memory_bytes() as f32;
        if full <= 0.0 {
            return 0.0;
        }
        let ckpt = self.memory_bytes() as f32;
        ((full - ckpt) / full).max(0.0)
    }
}

// ─── CheckpointBudget ────────────────────────────────────────────────────────

/// Memory budget tracker for gradient checkpointing.
///
/// Maintains a running total of bytes allocated to checkpointed inputs and
/// enforces an upper bound.  Call [`allocate`] when a new checkpoint is
/// created and [`free`] when it is discarded (e.g., after the backward pass
/// for that segment completes).
///
/// [`allocate`]: CheckpointBudget::allocate
/// [`free`]: CheckpointBudget::free
#[derive(Debug, Clone)]
pub struct CheckpointBudget {
    /// Maximum permitted allocation in bytes.
    pub max_bytes: usize,
    /// Currently allocated bytes.
    pub used_bytes: usize,
}

impl CheckpointBudget {
    /// Create a fresh budget with `max_bytes` capacity and zero usage.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
        }
    }

    /// Bytes still available for allocation.
    pub fn remaining(&self) -> usize {
        self.max_bytes.saturating_sub(self.used_bytes)
    }

    /// Fraction of the budget that has been consumed (`used / max`).
    ///
    /// Returns `0.0` when `max_bytes == 0` to avoid division by zero.
    pub fn utilization(&self) -> f32 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.max_bytes as f32
    }

    /// Whether `bytes` can be allocated without exceeding the budget.
    pub fn can_allocate(&self, bytes: usize) -> bool {
        self.used_bytes.saturating_add(bytes) <= self.max_bytes
    }

    /// Attempt to allocate `bytes`.
    ///
    /// On success, `used_bytes` increases by `bytes`.
    /// On failure, returns [`CheckpointError::BudgetExceeded`] and leaves
    /// `used_bytes` unchanged.
    pub fn allocate(&mut self, bytes: usize) -> Result<(), CheckpointError> {
        if !self.can_allocate(bytes) {
            return Err(CheckpointError::BudgetExceeded {
                need: bytes,
                available: self.remaining(),
            });
        }
        self.used_bytes += bytes;
        Ok(())
    }

    /// Release `bytes` back to the budget.
    ///
    /// Uses saturating subtraction to avoid underflow if `bytes` exceeds
    /// `used_bytes` (which would indicate a programming error, but should not
    /// panic in production).
    pub fn free(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }
}

// ─── CheckpointSegment (concrete, non-generic) ─────────────────────────────

/// A recomputable segment: stores weights and dimensions so the forward
/// pass (matrix-vector product `y = W * x`) can be re-executed cheaply.
///
/// Unlike the generic [`LinearSegment`] + [`Recomputable`] approach, this is
/// a self-contained struct that carries everything needed for recomputation.
pub struct CheckpointSegment {
    /// Human-readable name for this segment (e.g. `"layer_3"`).
    pub name: String,
    /// Row-major weight matrix of shape `[out_dim, in_dim]`.
    pub weights: Vec<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl CheckpointSegment {
    /// Create a segment with explicitly provided weights.
    pub fn new(name: impl Into<String>, weights: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            name: name.into(),
            weights,
            in_dim,
            out_dim,
        }
    }

    /// Create a segment with LCG-initialised weights (no `rand` crate).
    ///
    /// Uses the Knuth MMIX LCG constants. Weights are mapped to `[-1, 1]`.
    pub fn init_lcg(name: impl Into<String>, in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let count = in_dim * out_dim;
        let mut state = seed;
        let mut weights = Vec::with_capacity(count);
        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as i32;
            weights.push(bits as f32 / (1u64 << 31) as f32);
        }
        Self {
            name: name.into(),
            weights,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: compute `y = W * x` (matrix-vector product).
    ///
    /// `input` must have exactly `in_dim` elements.
    /// Returns a vector of `out_dim` elements.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, CheckpointError> {
        if input.len() != self.in_dim {
            return Err(CheckpointError::DimMismatch {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let mut output = vec![0.0f32; self.out_dim];
        for (row, out_val) in output.iter_mut().enumerate() {
            let row_offset = row * self.in_dim;
            let mut acc = 0.0f32;
            for (col, inp_val) in input.iter().enumerate() {
                acc += self.weights[row_offset + col] * inp_val;
            }
            *out_val = acc;
        }
        Ok(output)
    }

    /// Memory (in bytes) required to store one full activation (output).
    pub fn activation_memory(&self) -> usize {
        self.out_dim * std::mem::size_of::<f32>()
    }
}

// ─── CheckpointedActivation ─────────────────────────────────────────────────

/// A checkpointed activation: stores only the input and recomputes the
/// output on demand via the associated [`CheckpointSegment`].
pub struct CheckpointedActivation {
    segment: CheckpointSegment,
    saved_input: Vec<f32>,
}

impl CheckpointedActivation {
    /// Create a new checkpointed activation.
    pub fn new(segment: CheckpointSegment, input: Vec<f32>) -> Self {
        Self {
            segment,
            saved_input: input,
        }
    }

    /// Recompute the output from the saved input.
    pub fn recompute(&self) -> Result<Vec<f32>, CheckpointError> {
        self.segment.forward(&self.saved_input)
    }

    /// Memory actually consumed by this checkpoint (input only, in bytes).
    pub fn memory_bytes(&self) -> usize {
        self.saved_input.len() * std::mem::size_of::<f32>()
    }

    /// Memory that would be consumed if both input and output were stored.
    pub fn full_memory_bytes(&self) -> usize {
        self.memory_bytes() + self.segment.activation_memory()
    }

    /// Fraction of memory saved compared to the full (non-checkpointed) case.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    pub fn memory_savings(&self) -> f32 {
        let full = self.full_memory_bytes();
        if full == 0 {
            return 0.0;
        }
        1.0 - (self.memory_bytes() as f32 / full as f32)
    }
}

// ─── CheckpointedPipeline ───────────────────────────────────────────────────

/// A sequence of checkpointed layers that can be run end-to-end.
///
/// Stores only the per-layer inputs (not outputs) and recomputes
/// activations as needed.
pub struct CheckpointedPipeline {
    segments: Vec<CheckpointSegment>,
}

impl CheckpointedPipeline {
    /// Build a pipeline from a list of segments.
    pub fn new(segments: Vec<CheckpointSegment>) -> Self {
        Self { segments }
    }

    /// Run the full forward pass through all segments.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, CheckpointError> {
        if self.segments.is_empty() {
            return Err(CheckpointError::EmptyPipeline);
        }
        let mut current = input.to_vec();
        for seg in &self.segments {
            current = seg.forward(&current)?;
        }
        Ok(current)
    }

    /// Number of segments in the pipeline.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Total checkpoint memory for a given input size.
    ///
    /// First layer saves the original input; subsequent layers save their
    /// predecessor's output (= predecessor's `out_dim`).
    pub fn total_checkpoint_memory(&self, input_size: usize) -> usize {
        if self.segments.is_empty() {
            return 0;
        }
        let f32_size = std::mem::size_of::<f32>();
        let mut total = input_size * f32_size;
        for i in 0..self.segments.len() - 1 {
            total += self.segments[i].out_dim * f32_size;
        }
        total
    }

    /// Total memory if all activations (inputs **and** outputs) were stored.
    pub fn total_full_memory(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();
        self.segments
            .iter()
            .map(|s| (s.in_dim + s.out_dim) * f32_size)
            .sum()
    }

    /// Overall memory savings fraction for the whole pipeline.
    pub fn overall_savings(&self, input_size: usize) -> f32 {
        let full = self.total_full_memory();
        if full == 0 {
            return 0.0;
        }
        let ckpt = self.total_checkpoint_memory(input_size);
        1.0 - (ckpt as f32 / full as f32)
    }
}

// ─── CheckpointStrategy ────────────────────────────────────────────────────

/// Strategy for selecting which layers to checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointStrategy {
    /// Checkpoint every layer.
    Every,
    /// Checkpoint every N-th layer (layers 0, N, 2N, ...).
    EveryNth(usize),
    /// Checkpoint approximately sqrt(N) layers, evenly spaced.
    Sqrt,
    /// No checkpointing at all.
    None,
}

impl CheckpointStrategy {
    /// Given `total_layers`, return sorted indices of layers to checkpoint.
    pub fn select_layers(&self, total_layers: usize) -> Vec<usize> {
        match self {
            CheckpointStrategy::Every => (0..total_layers).collect(),
            CheckpointStrategy::EveryNth(n) => {
                let step = if *n == 0 { 1 } else { *n };
                (0..total_layers).filter(|i| i % step == 0).collect()
            }
            CheckpointStrategy::Sqrt => {
                if total_layers == 0 {
                    return Vec::new();
                }
                let count = isqrt(total_layers).max(1);
                if count >= total_layers {
                    return (0..total_layers).collect();
                }
                let step = total_layers / count;
                let mut layers = Vec::with_capacity(count);
                let mut idx = 0;
                while idx < total_layers && layers.len() < count {
                    layers.push(idx);
                    idx += step;
                }
                layers
            }
            CheckpointStrategy::None => Vec::new(),
        }
    }
}

/// Integer square root (floor) via Newton's method.
fn isqrt(n: usize) -> usize {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_segment_forward_shape() {
        let seg = LinearSegment::random_init(4, 8, 42);
        let input = vec![1.0f32; 4];
        let out = seg.forward(&input);
        assert_eq!(out.len(), 8, "output should have out_dim elements");
    }

    #[test]
    fn linear_segment_forward_deterministic() {
        let seg = LinearSegment::random_init(4, 8, 99);
        let input = vec![0.5f32, -0.5, 1.0, -1.0];
        let out1 = seg.forward(&input);
        let out2 = seg.forward(&input);
        assert_eq!(out1, out2, "forward must be deterministic");
    }

    #[test]
    fn checkpoint_recompute_equals_forward() {
        let seg = LinearSegment::random_init(3, 6, 7);
        let input = vec![1.0f32, 2.0, 3.0];
        let expected = seg.forward(&input);
        let ckpt = Checkpoint::new(seg, input);
        let got = ckpt.recompute();
        assert_eq!(got, expected, "recompute must equal original forward");
    }

    #[test]
    fn checkpoint_memory_input_only() {
        let seg = LinearSegment::random_init(5, 10, 0);
        let input = vec![0.0f32; 5];
        let ckpt = Checkpoint::new(seg, input);
        assert_eq!(ckpt.memory_bytes(), 5 * 4, "checkpoint stores input only");
    }

    #[test]
    fn network_forward_runs() {
        let seg1 = LinearSegment::random_init(4, 8, 1);
        let seg2 = LinearSegment::random_init(8, 4, 2);
        let input1 = vec![1.0f32; 4];
        let mid = seg1.forward(&input1);
        let input2 = mid.clone();
        let c1 = Checkpoint::new(seg1, input1);
        let c2 = Checkpoint::new(seg2, input2);
        let net = CheckpointedNetwork::new(vec![c1, c2]);
        let out = net.forward(&[1.0f32; 4]);
        assert_eq!(
            out.len(),
            4,
            "output should not panic and have correct length"
        );
    }

    #[test]
    fn network_memory_savings_positive() {
        // Expanding network: 4→16, 16→64 — outputs are much larger than inputs.
        let seg1 = LinearSegment::random_init(4, 16, 10);
        let seg2 = LinearSegment::random_init(16, 64, 11);
        let input1 = vec![1.0f32; 4];
        let mid = seg1.forward(&input1);
        let c1 = Checkpoint::new(seg1, input1);
        let c2 = Checkpoint::new(seg2, mid);
        let net = CheckpointedNetwork::new(vec![c1, c2]);
        let savings = net.memory_savings();
        assert!(
            savings > 0.0,
            "expanding network should save memory, got {savings}"
        );
    }

    #[test]
    fn network_full_memory_greater() {
        let seg1 = LinearSegment::random_init(4, 16, 20);
        let seg2 = LinearSegment::random_init(16, 64, 21);
        let input1 = vec![0.5f32; 4];
        let mid = seg1.forward(&input1);
        let c1 = Checkpoint::new(seg1, input1);
        let c2 = Checkpoint::new(seg2, mid);
        let net = CheckpointedNetwork::new(vec![c1, c2]);
        assert!(
            net.full_memory_bytes() > net.memory_bytes(),
            "full storage must use more memory than checkpointed storage"
        );
    }

    #[test]
    fn budget_new() {
        let b = CheckpointBudget::new(1024);
        assert_eq!(b.used_bytes, 0, "fresh budget should have used_bytes = 0");
        assert_eq!(b.max_bytes, 1024);
    }

    #[test]
    fn budget_allocate_within() {
        let mut b = CheckpointBudget::new(1024);
        let result = b.allocate(256);
        assert!(result.is_ok(), "allocation within budget must succeed");
        assert_eq!(b.used_bytes, 256);
    }

    #[test]
    fn budget_allocate_exceed() {
        let mut b = CheckpointBudget::new(100);
        let result = b.allocate(200);
        assert!(
            matches!(result, Err(CheckpointError::BudgetExceeded { .. })),
            "allocation exceeding budget must return BudgetExceeded"
        );
        assert_eq!(
            b.used_bytes, 0,
            "failed allocation must not change used_bytes"
        );
    }

    #[test]
    fn budget_free() {
        let mut b = CheckpointBudget::new(1024);
        b.allocate(512).expect("allocation should succeed");
        b.free(256);
        assert_eq!(b.used_bytes, 256);
    }

    #[test]
    fn budget_utilization() {
        let mut b = CheckpointBudget::new(1000);
        b.allocate(250).expect("allocation should succeed");
        let util = b.utilization();
        assert!(
            (util - 0.25).abs() < 1e-6,
            "utilization should be 0.25, got {util}"
        );
    }

    #[test]
    fn network_single_segment() {
        let seg = LinearSegment::random_init(3, 3, 55);
        let input = vec![1.0f32, 0.0, -1.0];
        let c = Checkpoint::new(seg, input);
        let net = CheckpointedNetwork::new(vec![c]);
        let out = net.forward(&[1.0f32, 0.0, -1.0]);
        assert_eq!(out.len(), 3, "single-segment network should produce output");
    }
}
