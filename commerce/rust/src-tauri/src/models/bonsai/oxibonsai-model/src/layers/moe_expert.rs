//! Mixture-of-Experts expert networks — individual FFN experts and the combined MoE FFN layer.
//!
//! Each expert is a small SwiGLU feed-forward network. The [`MoeFfnLayer`]
//! combines a [`TopKRouter`] with `N` experts, dispatching tokens to their
//! assigned experts and re-combining weighted outputs.

use crate::layers::moe_router::{
    combine_expert_outputs, ExpertBuffer, MoeConfig, RoutingDecision, TopKRouter,
};

// ──────────────────────────────────────────────────────────────────
// Single expert
// ──────────────────────────────────────────────────────────────────

/// A single MoE expert implemented as a SwiGLU feed-forward network.
pub struct Expert {
    /// Identifier for this expert (0-based index).
    pub expert_id: usize,
    /// Gate projection weight `[expert_hidden × hidden_size]`.
    pub gate_weight: Vec<f32>,
    /// Up projection weight `[expert_hidden × hidden_size]`.
    pub up_weight: Vec<f32>,
    /// Down projection weight `[hidden_size × expert_hidden]`.
    pub down_weight: Vec<f32>,
    /// Input/output hidden dimension.
    pub hidden_size: usize,
    /// Inner expert dimension (typically `hidden_size / num_experts * top_k`).
    pub expert_hidden: usize,
}

impl Expert {
    /// Create a new expert with zero-initialised weights.
    ///
    /// The caller is responsible for loading weights from a checkpoint.
    pub fn new(expert_id: usize, hidden_size: usize, expert_hidden: usize) -> Self {
        Self {
            expert_id,
            gate_weight: vec![0.0f32; expert_hidden * hidden_size],
            up_weight: vec![0.0f32; expert_hidden * hidden_size],
            down_weight: vec![0.0f32; hidden_size * expert_hidden],
            hidden_size,
            expert_hidden,
        }
    }

    /// Forward pass for a single token vector `x` of length `hidden_size`.
    ///
    /// Implements SwiGLU:
    /// ```text
    /// gate = silu(gate_weight @ x)
    /// up   = up_weight @ x
    /// out  = down_weight @ (gate ⊙ up)
    /// ```
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.hidden_size);
        let eh = self.expert_hidden;
        let h = self.hidden_size;

        // gate projection → [expert_hidden]
        let mut gate = vec![0.0f32; eh];
        for (i, slot) in gate.iter_mut().enumerate().take(eh) {
            let mut sum = 0.0f32;
            let row = &self.gate_weight[i * h..(i + 1) * h];
            for (j, &wj) in row.iter().enumerate() {
                sum += wj * x[j];
            }
            *slot = silu(sum);
        }

        // up projection → [expert_hidden]
        let mut up = vec![0.0f32; eh];
        for (i, slot) in up.iter_mut().enumerate().take(eh) {
            let mut sum = 0.0f32;
            let row = &self.up_weight[i * h..(i + 1) * h];
            for (j, &wj) in row.iter().enumerate() {
                sum += wj * x[j];
            }
            *slot = sum;
        }

        // Element-wise gate * up → [expert_hidden]
        let gated: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

        // down projection → [hidden_size]
        let mut out = vec![0.0f32; h];
        for (i, slot) in out.iter_mut().enumerate().take(h) {
            let mut sum = 0.0f32;
            let row = &self.down_weight[i * eh..(i + 1) * eh];
            for (j, &wj) in row.iter().enumerate() {
                sum += wj * gated[j];
            }
            *slot = sum;
        }

        out
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        // gate + up + down
        2 * self.expert_hidden * self.hidden_size + self.hidden_size * self.expert_hidden
    }

    /// Memory footprint in bytes (assuming f32 storage).
    pub fn memory_bytes(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }

    /// Forward pass for a batch of `num_tokens` token vectors.
    ///
    /// `tokens` must have length `num_tokens * hidden_size`.
    /// Returns `[num_tokens × hidden_size]`.
    pub fn forward_batch(&self, tokens: &[f32], num_tokens: usize) -> Vec<f32> {
        let h = self.hidden_size;
        debug_assert_eq!(tokens.len(), num_tokens * h);
        let mut out = Vec::with_capacity(num_tokens * h);
        for t in 0..num_tokens {
            let x = &tokens[t * h..(t + 1) * h];
            let result = self.forward(x);
            out.extend_from_slice(&result);
        }
        out
    }
}

// ──────────────────────────────────────────────────────────────────
// SwiGLU activation
// ──────────────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ──────────────────────────────────────────────────────────────────
// MoE FFN layer
// ──────────────────────────────────────────────────────────────────

/// A complete MoE feed-forward network layer: router + N experts.
pub struct MoeFfnLayer {
    /// MoE configuration.
    pub config: MoeConfig,
    /// Top-K router.
    pub router: TopKRouter,
    /// Expert networks.
    pub experts: Vec<Expert>,
}

impl MoeFfnLayer {
    /// Create a new MoE FFN layer with zero-weight experts.
    pub fn new(config: MoeConfig, expert_hidden: usize) -> Self {
        let router = TopKRouter::new(config.clone());
        let experts: Vec<Expert> = (0..config.num_experts)
            .map(|id| Expert::new(id, config.hidden_size, expert_hidden))
            .collect();
        Self {
            config,
            router,
            experts,
        }
    }

    /// Forward pass: route tokens, run experts, combine outputs.
    ///
    /// # Arguments
    /// - `input`: `[num_tokens × hidden_size]`
    /// - `num_tokens`: number of tokens in the batch
    ///
    /// # Returns
    /// - Combined output `[num_tokens × hidden_size]`
    /// - [`RoutingDecision`] for loss computation
    pub fn forward(&self, input: &[f32], num_tokens: usize) -> (Vec<f32>, RoutingDecision) {
        let h = self.config.hidden_size;
        let e = self.config.num_experts;
        let k = self.config.top_k;

        // Step 1: Route tokens
        let routing = self.router.route(input, num_tokens);

        // Step 2: Build per-expert token lists using a buffer
        let capacity =
            ((self.config.capacity_factor * num_tokens as f32 / e as f32).ceil() as usize).max(1);
        let mut buffer = ExpertBuffer::new(e, capacity);

        for t in 0..num_tokens {
            if routing.overflow_mask[t] {
                continue;
            }
            for &ex in &routing.expert_indices[t] {
                buffer.try_assign(t, ex);
            }
        }

        // Step 3: Run each expert on its assigned tokens
        let mut expert_outputs: Vec<Vec<f32>> = Vec::with_capacity(e);
        for ex in 0..e {
            let token_indices = buffer.tokens_for_expert(ex);
            if token_indices.is_empty() {
                expert_outputs.push(Vec::new());
                continue;
            }
            // Gather token vectors for this expert
            let mut expert_input = Vec::with_capacity(token_indices.len() * h);
            for &t in token_indices {
                expert_input.extend_from_slice(&input[t * h..(t + 1) * h]);
            }
            let ex_out = self.experts[ex].forward_batch(&expert_input, token_indices.len());
            expert_outputs.push(ex_out);
        }

        // Step 4: Combine weighted expert outputs
        let combined = combine_expert_outputs(&expert_outputs, &routing, num_tokens, h);

        let _ = k;
        (combined, routing)
    }

    /// Total parameter count across all experts and the router.
    pub fn total_param_count(&self) -> usize {
        let router_params = self.config.num_experts * self.config.hidden_size;
        let expert_params: usize = self.experts.iter().map(|ex| ex.param_count()).sum();
        router_params + expert_params
    }

    /// Number of parameters active in a single forward pass (top_k experts out of num_experts).
    pub fn active_param_count(&self, top_k: usize) -> usize {
        let router_params = self.config.num_experts * self.config.hidden_size;
        let single_expert_params = if self.experts.is_empty() {
            0
        } else {
            self.experts[0].param_count()
        };
        // top_k experts active plus full router
        router_params + top_k * single_expert_params
    }

    /// Total memory footprint of all parameters in bytes.
    pub fn memory_bytes(&self) -> usize {
        let router_bytes =
            self.config.num_experts * self.config.hidden_size * std::mem::size_of::<f32>();
        let expert_bytes: usize = self.experts.iter().map(|ex| ex.memory_bytes()).sum();
        router_bytes + expert_bytes
    }

    /// Sparsity: fraction of experts NOT activated per forward pass.
    ///
    /// = 1 - top_k / num_experts
    pub fn sparsity(&self) -> f32 {
        if self.config.num_experts == 0 {
            return 0.0;
        }
        1.0 - self.config.top_k as f32 / self.config.num_experts as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// Configuration presets
// ──────────────────────────────────────────────────────────────────

/// Mixtral-style 8x7B MoE configuration.
///
/// 8 experts total, top-2 routing, 4096-dimensional hidden state.
pub fn moe_8x7b_config() -> MoeConfig {
    MoeConfig {
        num_experts: 8,
        top_k: 2,
        hidden_size: 4096,
        capacity_factor: 1.25,
        z_loss_coeff: 1e-3,
        aux_loss_coeff: 1e-2,
    }
}

/// Tiny MoE configuration for unit tests.
///
/// 4 experts total, top-2 routing, 64-dimensional hidden state.
pub fn moe_tiny_config() -> MoeConfig {
    MoeConfig {
        num_experts: 4,
        top_k: 2,
        hidden_size: 64,
        capacity_factor: 1.25,
        z_loss_coeff: 1e-3,
        aux_loss_coeff: 1e-2,
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_expert(hidden_size: usize, expert_hidden: usize) -> Expert {
        Expert::new(0, hidden_size, expert_hidden)
    }

    #[test]
    fn test_expert_forward_shape() {
        let h = 16;
        let eh = 8;
        let expert = make_expert(h, eh);
        let x = vec![0.5f32; h];
        let out = expert.forward(&x);
        assert_eq!(out.len(), h, "output length must equal hidden_size");
    }

    #[test]
    fn test_expert_forward_batch_shape() {
        let h = 16;
        let eh = 8;
        let num_tokens = 5;
        let expert = make_expert(h, eh);
        let tokens = vec![0.1f32; num_tokens * h];
        let out = expert.forward_batch(&tokens, num_tokens);
        assert_eq!(
            out.len(),
            num_tokens * h,
            "batch output must be num_tokens * hidden_size"
        );
    }

    #[test]
    fn test_expert_param_count() {
        let h = 32;
        let eh = 16;
        let expert = make_expert(h, eh);
        // gate: eh*h, up: eh*h, down: h*eh  = 3 * eh * h
        let expected = 3 * eh * h;
        assert_eq!(expert.param_count(), expected);
        assert_eq!(expert.memory_bytes(), expected * 4);
    }

    #[test]
    fn test_moe_ffn_layer_forward_shape() {
        let cfg = moe_tiny_config();
        let h = cfg.hidden_size;
        let expert_hidden = h / 2;
        let layer = MoeFfnLayer::new(cfg.clone(), expert_hidden);
        let num_tokens = 8;
        let input = vec![0.1f32; num_tokens * h];
        let (out, routing) = layer.forward(&input, num_tokens);
        assert_eq!(out.len(), num_tokens * h, "output shape mismatch");
        assert_eq!(routing.expert_indices.len(), num_tokens);
    }

    #[test]
    fn test_moe_ffn_active_params_less_than_total() {
        let cfg = moe_tiny_config();
        let h = cfg.hidden_size;
        let expert_hidden = h / 2;
        let layer = MoeFfnLayer::new(cfg.clone(), expert_hidden);
        let active = layer.active_param_count(cfg.top_k);
        let total = layer.total_param_count();
        assert!(
            active <= total,
            "active params {} should be <= total params {}",
            active,
            total
        );
    }

    #[test]
    fn test_moe_ffn_sparsity() {
        let cfg = moe_tiny_config(); // 4 experts, top-2
        let layer = MoeFfnLayer::new(cfg.clone(), cfg.hidden_size / 2);
        let sparsity = layer.sparsity();
        // 1 - 2/4 = 0.5
        assert!(
            (sparsity - 0.5).abs() < 1e-6,
            "expected 0.5 sparsity, got {}",
            sparsity
        );
    }

    #[test]
    fn test_moe_8x7b_config() {
        let cfg = moe_8x7b_config();
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.top_k, 2);
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_moe_tiny_forward() {
        let cfg = moe_tiny_config();
        let h = cfg.hidden_size;
        let expert_hidden = 32;
        let layer = MoeFfnLayer::new(cfg.clone(), expert_hidden);
        // Use a varied input to get meaningful routing
        let num_tokens = 4;
        let input: Vec<f32> = (0..num_tokens * h)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let (out, routing) = layer.forward(&input, num_tokens);
        assert_eq!(out.len(), num_tokens * h);
        // With zero weights, all expert outputs are zero → combined output is zero
        for &v in &out {
            assert!(
                v.abs() < 1e-6,
                "expected zero output with zero-weight experts, got {}",
                v
            );
        }
        // Routing should assign top_k experts per token
        for indices in &routing.expert_indices {
            assert_eq!(indices.len(), cfg.top_k);
        }
    }
}
