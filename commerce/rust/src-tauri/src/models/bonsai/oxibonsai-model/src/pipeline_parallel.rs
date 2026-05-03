//! Pipeline parallelism utilities for OxiBonsai.
//!
//! Pipeline parallelism divides the transformer layer stack into consecutive
//! *stages*, each of which is assigned to a separate worker (thread/device).
//! During inference micro-batches flow through the pipeline from stage 0 to
//! stage N-1, enabling concurrent processing of different micro-batches.
//!
//! This module provides:
//! * [`PipelineStage`] — describes which layers belong to one stage.
//! * [`partition_layers`] — evenly splits layers across stages.
//! * [`MicroBatch`] — a token sequence (or hidden states) being processed.
//! * [`PipelineSchedule`] — generates the execution order for GPipe scheduling.

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStage
// ─────────────────────────────────────────────────────────────────────────────

/// A single stage in the pipeline, owning a contiguous range of transformer
/// layers.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Zero-based stage index.
    pub stage_id: usize,
    /// Total number of stages in the pipeline.
    pub num_stages: usize,
    /// Index of the first layer owned by this stage (inclusive).
    pub layer_start: usize,
    /// Index one past the last layer owned by this stage (exclusive).
    pub layer_end: usize,
    /// `true` for stage 0 — receives raw token embeddings.
    pub is_first: bool,
    /// `true` for the last stage — emits final logits.
    pub is_last: bool,
}

impl PipelineStage {
    /// Construct a `PipelineStage` by evenly distributing `num_layers` across
    /// `num_stages`.  The last stage absorbs any remainder layers.
    pub fn new(stage_id: usize, num_stages: usize, num_layers: usize) -> Self {
        assert!(num_stages > 0, "num_stages must be > 0");
        let base = num_layers / num_stages;
        let remainder = num_layers % num_stages;
        let layer_start = stage_id * base;
        let layer_end = if stage_id + 1 == num_stages {
            stage_id * base + base + remainder
        } else {
            (stage_id + 1) * base
        };
        Self {
            stage_id,
            num_stages,
            layer_start,
            layer_end,
            is_first: stage_id == 0,
            is_last: stage_id + 1 == num_stages,
        }
    }

    /// Number of transformer layers assigned to this stage.
    pub fn layer_count(&self) -> usize {
        self.layer_end - self.layer_start
    }

    /// Returns `true` if `layer_idx` falls within this stage's layer range.
    pub fn contains_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.layer_start && layer_idx < self.layer_end
    }

    /// The half-open range of layer indices owned by this stage.
    pub fn layer_range(&self) -> std::ops::Range<usize> {
        self.layer_start..self.layer_end
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// partition_layers
// ─────────────────────────────────────────────────────────────────────────────

/// Partition `num_layers` transformer layers evenly across `num_stages`.
///
/// The last stage receives any remainder layers so that all layers are
/// covered.  Returns one [`PipelineStage`] per stage in stage order.
pub fn partition_layers(num_layers: usize, num_stages: usize) -> Vec<PipelineStage> {
    (0..num_stages)
        .map(|id| PipelineStage::new(id, num_stages, num_layers))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MicroBatch
// ─────────────────────────────────────────────────────────────────────────────

/// A micro-batch of tokens flowing through the pipeline.
///
/// At the first stage, `hidden_states` is `None` and `tokens` contains the
/// raw token IDs.  Subsequent stages receive the hidden states produced by the
/// previous stage.
pub struct MicroBatch {
    /// Unique identifier for this micro-batch within a scheduling step.
    pub batch_id: usize,
    /// Token IDs (meaningful only at the first stage).
    pub tokens: Vec<u32>,
    /// Hidden states passed from the previous stage (`None` at stage 0).
    pub hidden_states: Option<Vec<f32>>,
    /// The stage at which this micro-batch currently resides.
    pub stage_id: usize,
}

impl MicroBatch {
    /// Create a new micro-batch with raw tokens, destined for stage 0.
    pub fn new(batch_id: usize, tokens: Vec<u32>) -> Self {
        Self {
            batch_id,
            tokens,
            hidden_states: None,
            stage_id: 0,
        }
    }

    /// Create a micro-batch carrying hidden states from a previous stage.
    pub fn with_hidden_states(batch_id: usize, hidden_states: Vec<f32>, stage_id: usize) -> Self {
        Self {
            batch_id,
            tokens: Vec::new(),
            hidden_states: Some(hidden_states),
            stage_id,
        }
    }

    /// Returns `true` if this micro-batch is at the first stage (no prior
    /// hidden states).
    #[inline]
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// Execution schedule for a pipeline with a fixed number of stages and
/// micro-batches.
pub struct PipelineSchedule {
    /// Number of pipeline stages.
    pub num_stages: usize,
    /// Number of micro-batches in flight per global step.
    pub num_micro_batches: usize,
}

impl PipelineSchedule {
    /// Create a new `PipelineSchedule`.
    pub fn new(num_stages: usize, num_micro_batches: usize) -> Self {
        Self {
            num_stages,
            num_micro_batches,
        }
    }

    /// Generate a GPipe schedule.
    ///
    /// GPipe runs all forward passes for all micro-batches through all stages
    /// in order, then all backward passes in reverse order.
    ///
    /// The returned vector contains tuples `(stage_id, micro_batch_id,
    /// is_forward)`.  Stages are iterated in order for each micro-batch, then
    /// the same sequence is repeated with `is_forward = false` (backward).
    ///
    /// Forward order: for each stage s in 0..num_stages, for each micro-batch
    /// m in 0..num_micro_batches emit `(s, m, true)`.
    /// Backward order: same but with `is_forward = false`.
    pub fn gpipe_schedule(&self) -> Vec<(usize, usize, bool)> {
        let fwd_steps = self.num_stages * self.num_micro_batches;
        let mut schedule = Vec::with_capacity(fwd_steps * 2);
        // All forward passes: iterate stages outer, micro-batches inner.
        for stage in 0..self.num_stages {
            for mb in 0..self.num_micro_batches {
                schedule.push((stage, mb, true));
            }
        }
        // All backward passes: same order (GPipe runs bwd after all fwd).
        for stage in 0..self.num_stages {
            for mb in 0..self.num_micro_batches {
                schedule.push((stage, mb, false));
            }
        }
        schedule
    }

    /// Total number of steps in the complete schedule (forward + backward).
    pub fn total_steps(&self) -> usize {
        2 * self.num_stages * self.num_micro_batches
    }

    /// Pipeline bubble fraction for GPipe.
    ///
    /// The bubble overhead is the fraction of pipeline cycles wasted due to
    /// the fill and drain phases:
    ///
    /// ```text
    /// bubble = (num_stages - 1) / (num_stages - 1 + num_micro_batches)
    /// ```
    ///
    /// A lower value means higher pipeline utilisation.
    pub fn bubble_fraction(&self) -> f32 {
        if self.num_stages == 0 || self.num_micro_batches == 0 {
            return 0.0;
        }
        let numerator = (self.num_stages as f32) - 1.0;
        let denominator = numerator + (self.num_micro_batches as f32);
        if denominator <= 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory estimate
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate memory required per pipeline stage.
///
/// Assumes weights are evenly split across stages; activations occupy
/// `activation_bytes_per_token` bytes per token per micro-batch.
///
/// Returns the estimated bytes for one stage:
/// * weight memory   = `total_params * 4 / num_stages` (f32)
/// * activation mem  = `activation_bytes_per_token * micro_batch_tokens`
pub fn pipeline_memory_per_stage(
    total_params: u64,
    num_stages: usize,
    activation_bytes_per_token: usize,
    micro_batch_tokens: usize,
) -> usize {
    let weight_bytes = ((total_params as usize) * std::mem::size_of::<f32>())
        .checked_div(num_stages)
        .unwrap_or(0);
    let activation_bytes = activation_bytes_per_token * micro_batch_tokens;
    weight_bytes + activation_bytes
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineStage / partition_layers ──────────────────────────────────────

    #[test]
    fn test_pipeline_stage_partition_even() {
        // 8 layers, 4 stages → 2 layers per stage
        let stages = partition_layers(8, 4);
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[0].layer_end, 2);
        assert_eq!(stages[1].layer_start, 2);
        assert_eq!(stages[1].layer_end, 4);
        assert_eq!(stages[2].layer_start, 4);
        assert_eq!(stages[2].layer_end, 6);
        assert_eq!(stages[3].layer_start, 6);
        assert_eq!(stages[3].layer_end, 8);
    }

    #[test]
    fn test_pipeline_stage_partition_uneven() {
        // 10 layers, 3 stages → base=3, remainder=1 → sizes [3, 3, 4]
        let stages = partition_layers(10, 3);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].layer_count(), 3);
        assert_eq!(stages[1].layer_count(), 3);
        assert_eq!(stages[2].layer_count(), 4); // last gets remainder
                                                // Verify contiguity and full coverage.
        assert_eq!(stages[0].layer_end, stages[1].layer_start);
        assert_eq!(stages[1].layer_end, stages[2].layer_start);
        assert_eq!(stages[2].layer_end, 10);
    }

    #[test]
    fn test_pipeline_stage_contains_layer() {
        let stage = PipelineStage::new(1, 4, 8);
        // Stage 1 owns layers 2..4
        assert!(!stage.contains_layer(1));
        assert!(stage.contains_layer(2));
        assert!(stage.contains_layer(3));
        assert!(!stage.contains_layer(4));
    }

    #[test]
    fn test_pipeline_stage_is_first_last() {
        let stages = partition_layers(6, 3);
        assert!(stages[0].is_first);
        assert!(!stages[0].is_last);
        assert!(!stages[1].is_first);
        assert!(!stages[1].is_last);
        assert!(!stages[2].is_first);
        assert!(stages[2].is_last);
    }

    // ── MicroBatch ────────────────────────────────────────────────────────────

    #[test]
    fn test_micro_batch_new() {
        let mb = MicroBatch::new(42, vec![1, 2, 3]);
        assert_eq!(mb.batch_id, 42);
        assert_eq!(mb.tokens, vec![1u32, 2, 3]);
        assert!(mb.hidden_states.is_none());
        assert_eq!(mb.stage_id, 0);
        assert!(mb.is_first_stage());
    }

    #[test]
    fn test_micro_batch_with_hidden_states() {
        let hs = vec![0.1f32, 0.2, 0.3];
        let mb = MicroBatch::with_hidden_states(7, hs.clone(), 2);
        assert_eq!(mb.batch_id, 7);
        assert_eq!(mb.stage_id, 2);
        assert!(!mb.is_first_stage());
        assert_eq!(mb.hidden_states.as_deref(), Some(hs.as_slice()));
    }

    // ── PipelineSchedule ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_schedule_gpipe() {
        // 2 stages, 3 micro-batches.
        let sched = PipelineSchedule::new(2, 3);
        let steps = sched.gpipe_schedule();
        // Total = 2 * 2 * 3 = 12 steps.
        assert_eq!(steps.len(), 12);
        // First 6 are all forward passes.
        for &(_, _, is_fwd) in &steps[..6] {
            assert!(is_fwd);
        }
        // Next 6 are all backward passes.
        for &(_, _, is_fwd) in &steps[6..] {
            assert!(!is_fwd);
        }
        // Verify first forward step is (stage=0, mb=0, true).
        assert_eq!(steps[0], (0, 0, true));
        // Stage 0 processes all 3 micro-batches first.
        assert_eq!(steps[0], (0, 0, true));
        assert_eq!(steps[1], (0, 1, true));
        assert_eq!(steps[2], (0, 2, true));
        assert_eq!(steps[3], (1, 0, true));
    }

    #[test]
    fn test_pipeline_schedule_bubble_fraction() {
        // 4 stages, 8 micro-batches → bubble = 3 / (3 + 8) ≈ 0.2727
        let sched = PipelineSchedule::new(4, 8);
        let bubble = sched.bubble_fraction();
        let expected = 3.0f32 / 11.0;
        assert!((bubble - expected).abs() < 1e-5, "bubble={bubble}");

        // 1 stage → bubble = 0 / (0 + n) = 0
        let sched1 = PipelineSchedule::new(1, 4);
        assert!((sched1.bubble_fraction()).abs() < 1e-6);
    }

    // ── pipeline_memory_per_stage ─────────────────────────────────────────────

    #[test]
    fn test_pipeline_memory_per_stage() {
        // 8B params (8_000_000_000), 4 stages, 512 bytes/token, 128 tokens.
        let total_params: u64 = 8_000_000_000;
        let num_stages = 4;
        let activation_bytes_per_token = 512;
        let micro_batch_tokens = 128;
        let mem = pipeline_memory_per_stage(
            total_params,
            num_stages,
            activation_bytes_per_token,
            micro_batch_tokens,
        );
        // weight bytes = 8e9 * 4 / 4 = 8_000_000_000
        let expected_weights = (total_params as usize) * 4 / num_stages;
        let expected_act = activation_bytes_per_token * micro_batch_tokens;
        assert_eq!(mem, expected_weights + expected_act);
    }
}
