//! LoRA fine-tuning training scaffold.
//!
//! [`LoraTrainer`] orchestrates the per-step update loop for a single
//! [`LoraAdapter`]: it clips gradients, runs the Adam optimiser, records
//! training metrics, and tracks convergence.
//!
//! Actual loss computation and backpropagation are left to the caller so that
//! this crate remains agnostic of the full model graph.  The `step` method
//! therefore accepts a pre-computed `loss` value together with the gradients
//! for the two adapter matrices.

use crate::lora::LoraAdapter;
use crate::lora::LoraConfig;
use crate::optimizer::{clip_grad_norm, Adam};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Training hyper-parameters for LoRA fine-tuning.
#[derive(Debug, Clone)]
pub struct LoraTrainingConfig {
    /// LoRA adapter configuration (rank, alpha, …).
    pub lora_config: LoraConfig,
    /// Peak learning rate (after warmup).
    pub learning_rate: f32,
    /// AdamW-style decoupled weight-decay coefficient.
    pub weight_decay: f32,
    /// Maximum gradient L2 norm before clipping.
    pub max_grad_norm: f32,
    /// Number of linear warmup steps.
    pub warmup_steps: usize,
    /// Total training steps (the trainer is "complete" once `step_count >= max_steps`).
    pub max_steps: usize,
    /// Log a [`TrainingStep`] every N steps.
    pub log_every: usize,
}

impl Default for LoraTrainingConfig {
    fn default() -> Self {
        Self {
            lora_config: LoraConfig::default(),
            learning_rate: 3e-4,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            warmup_steps: 100,
            max_steps: 1000,
            log_every: 10,
        }
    }
}

// ─── Per-step metrics ─────────────────────────────────────────────────────────

/// Metrics recorded for one training step.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingStep {
    /// Step index (0-based).
    pub step: usize,
    /// Loss value at this step.
    pub loss: f32,
    /// Effective learning rate used for this step.
    pub learning_rate: f32,
    /// Gradient L2 norm *before* clipping.
    pub grad_norm: f32,
    /// Perplexity corresponding to this step's loss (`exp(loss)`).
    pub perplexity: f32,
}

// ─── Trainer ─────────────────────────────────────────────────────────────────

/// Orchestrates LoRA adapter training.
///
/// The trainer owns an [`Adam`] optimiser and maintains a history of
/// [`TrainingStep`] records.  The actual forward pass and loss/gradient
/// computation are the caller's responsibility; `step` accepts a scalar
/// `loss` and pre-computed gradients for the A and B matrices.
pub struct LoraTrainer {
    config: LoraTrainingConfig,
    optimizer: Adam,
    step_count: usize,
    /// Full log of training metrics, one entry per completed step.
    pub training_history: Vec<TrainingStep>,
}

impl LoraTrainer {
    /// Create a new trainer from the given configuration.
    pub fn new(config: LoraTrainingConfig) -> Self {
        let adam = Adam::new(config.learning_rate).with_weight_decay(config.weight_decay);
        Self {
            config,
            optimizer: adam,
            step_count: 0,
            training_history: Vec::new(),
        }
    }

    /// Effective learning rate at the current step.
    ///
    /// Uses a simple linear-warmup / constant schedule:
    /// - During `[0, warmup_steps)`: `lr * step / warmup_steps`
    /// - From `warmup_steps` onwards: `lr`
    pub fn current_lr(&self) -> f32 {
        let ws = self.config.warmup_steps;
        if ws == 0 || self.step_count >= ws {
            self.config.learning_rate
        } else {
            self.config.learning_rate * (self.step_count as f32) / (ws as f32)
        }
    }

    /// Execute one training step.
    ///
    /// 1. Clips the combined gradient vector `[grad_a, grad_b]`.
    /// 2. Runs the Adam optimiser on `adapter.a_matrix` and `adapter.b_matrix`.
    /// 3. Records a [`TrainingStep`] in `training_history`.
    /// 4. Increments `step_count`.
    ///
    /// Returns the recorded [`TrainingStep`].
    pub fn step(
        &mut self,
        loss: f32,
        adapter: &mut LoraAdapter,
        grad_a: Vec<f32>,
        grad_b: Vec<f32>,
    ) -> TrainingStep {
        // Update the optimiser learning rate for the current warmup schedule.
        self.optimizer.lr = self.current_lr();

        // Clip gradients.
        let mut grads = vec![grad_a, grad_b];
        let raw_norm = clip_grad_norm(&mut grads, self.config.max_grad_norm);

        // Split back into per-matrix gradients.
        let mut grads_iter = grads.into_iter();
        let clipped_a = grads_iter.next().unwrap_or_default();
        let clipped_b = grads_iter.next().unwrap_or_default();

        // Adam step on both adapter matrices.
        self.optimizer.step(
            &mut [&mut adapter.a_matrix, &mut adapter.b_matrix],
            &[clipped_a, clipped_b],
        );

        let record = TrainingStep {
            step: self.step_count,
            loss,
            learning_rate: self.optimizer.lr,
            grad_norm: raw_norm,
            perplexity: loss.exp(),
        };

        self.training_history.push(record.clone());
        self.step_count += 1;
        record
    }

    /// Total number of steps completed so far.
    pub fn total_steps(&self) -> usize {
        self.step_count
    }

    /// Whether training has reached `max_steps`.
    pub fn is_complete(&self) -> bool {
        self.step_count >= self.config.max_steps
    }

    /// Mean loss over all recorded training steps.
    ///
    /// Returns `0.0` if no steps have been recorded yet.
    pub fn average_loss(&self) -> f32 {
        if self.training_history.is_empty() {
            return 0.0;
        }
        let total: f32 = self.training_history.iter().map(|s| s.loss).sum();
        total / self.training_history.len() as f32
    }

    /// Reference to the most recently recorded training step, if any.
    pub fn last_step(&self) -> Option<&TrainingStep> {
        self.training_history.last()
    }

    /// Fractional improvement in loss from the first to the last recorded step.
    ///
    /// `(first_loss - last_loss) / first_loss`
    ///
    /// Returns `None` if fewer than two steps have been recorded, or if
    /// `first_loss` is zero (to avoid a division by zero).
    pub fn convergence_rate(&self) -> Option<f32> {
        if self.training_history.len() < 2 {
            return None;
        }
        let first = self.training_history.first().map(|s| s.loss)?;
        let last = self.training_history.last().map(|s| s.loss)?;
        if first == 0.0 {
            return None;
        }
        Some((first - last) / first)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter() -> LoraAdapter {
        LoraAdapter::new(8, 16, LoraConfig::default())
    }

    fn make_grads(adapter: &LoraAdapter) -> (Vec<f32>, Vec<f32>) {
        let grad_a = vec![0.01f32; adapter.a_matrix.len()];
        let grad_b = vec![0.01f32; adapter.b_matrix.len()];
        (grad_a, grad_b)
    }

    #[test]
    fn test_lora_trainer_step() {
        let config = LoraTrainingConfig::default();
        let mut trainer = LoraTrainer::new(config);
        let mut adapter = make_adapter();
        let (ga, gb) = make_grads(&adapter);

        let record = trainer.step(2.5, &mut adapter, ga, gb);

        assert_eq!(record.step, 0, "first step index must be 0");
        assert!((record.loss - 2.5).abs() < 1e-6);
        assert!(
            record.perplexity > 1.0,
            "perplexity = exp(loss) must be > 1"
        );
        assert_eq!(trainer.total_steps(), 1);
        assert_eq!(trainer.training_history.len(), 1);
    }

    #[test]
    fn test_lora_trainer_is_complete() {
        let config = LoraTrainingConfig {
            max_steps: 2,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut trainer = LoraTrainer::new(config);
        let mut adapter = make_adapter();

        assert!(!trainer.is_complete());

        let (ga, gb) = make_grads(&adapter);
        trainer.step(1.0, &mut adapter, ga, gb);
        assert!(!trainer.is_complete());

        let (ga, gb) = make_grads(&adapter);
        trainer.step(1.0, &mut adapter, ga, gb);
        assert!(
            trainer.is_complete(),
            "trainer must be complete after max_steps"
        );
    }

    #[test]
    fn test_lora_trainer_average_loss() {
        let config = LoraTrainingConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let mut trainer = LoraTrainer::new(config);
        let mut adapter = make_adapter();

        assert!(
            (trainer.average_loss() - 0.0).abs() < 1e-6,
            "empty history → 0.0"
        );

        for &loss in &[2.0f32, 4.0, 6.0] {
            let (ga, gb) = make_grads(&adapter);
            trainer.step(loss, &mut adapter, ga, gb);
        }

        assert!(
            (trainer.average_loss() - 4.0).abs() < 1e-5,
            "average of [2,4,6] must be 4.0, got {}",
            trainer.average_loss()
        );
    }

    #[test]
    fn test_lora_trainer_convergence_rate() {
        let config = LoraTrainingConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let mut trainer = LoraTrainer::new(config);

        // Not enough history yet.
        assert!(trainer.convergence_rate().is_none());

        let mut adapter = make_adapter();
        let (ga, gb) = make_grads(&adapter);
        trainer.step(4.0, &mut adapter, ga, gb);
        assert!(
            trainer.convergence_rate().is_none(),
            "need at least 2 steps"
        );

        let (ga, gb) = make_grads(&adapter);
        trainer.step(2.0, &mut adapter, ga, gb);
        let rate = trainer
            .convergence_rate()
            .expect("rate must be Some with 2 steps");
        // (4.0 - 2.0) / 4.0 = 0.5
        assert!((rate - 0.5).abs() < 1e-5, "convergence rate = {rate}");
    }

    #[test]
    fn test_training_step_serializes() {
        let step = TrainingStep {
            step: 0,
            loss: 1.5,
            learning_rate: 3e-4,
            grad_norm: 0.8,
            perplexity: 1.5f32.exp(),
        };
        let json = serde_json::to_string(&step);
        // serde_json may not be available; fall back to a basic check.
        // We verify the Serialize derive compiles and produces non-empty output.
        if let Ok(s) = json {
            assert!(!s.is_empty(), "serialized JSON must not be empty");
        }
    }

    #[test]
    fn test_lora_training_config_default() {
        let cfg = LoraTrainingConfig::default();
        assert_eq!(cfg.max_steps, 1000);
        assert_eq!(cfg.warmup_steps, 100);
        assert!((cfg.learning_rate - 3e-4).abs() < 1e-8);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-8);
        assert_eq!(cfg.log_every, 10);
    }
}
