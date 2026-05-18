pub mod config;
pub mod viterbi;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, VarBuilder, Embedding};
use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;
use self::config::{ModelConfig, ViterbiConfig};
use self::viterbi::PrivacySpan;
use candle_transformers::models::deepseek2::TopKLastDimOp;

const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;

pub struct PrivacyFilterModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    score_weight: Tensor,
    score_bias: Tensor,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
    viterbi_config: ViterbiConfig,
    device: Device,
    dtype: DType,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

struct RotaryEmbedding {
    inv_freq: Tensor,
    attention_scaling: f64,
}

impl RotaryEmbedding {
    fn new_yarn(cfg: &config::RopeParameters, head_dim: usize, max_seq_len: usize, dtype: DType, device: &Device) -> Result<Self> {
        let dim = head_dim;
        let half_dim = dim / 2;
        let theta = cfg.rope_theta;
        let factor = cfg.factor;

        let attention_scaling = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };

        let mut inv_freq_extrapolation = vec![0f32; half_dim];
        let mut inv_freq_interpolation = vec![0f32; half_dim];
        for i in 0..half_dim {
            let freq = theta.powf(2.0 * i as f64 / dim as f64);
            inv_freq_extrapolation[i] = 1.0 / freq as f32;
            inv_freq_interpolation[i] = 1.0 / (factor * freq) as f32;
        }

        let find_correction_dim = |num_rotations: f64| -> f64 {
            (dim as f64 * (cfg.original_max_position_embeddings as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * theta.ln())
        };

        let low_raw = find_correction_dim(cfg.beta_fast);
        let high_raw = find_correction_dim(cfg.beta_slow);

        let (low, high) = if cfg.truncate {
            (low_raw.floor(), high_raw.ceil())
        } else {
            (low_raw, high_raw)
        };
        let low = low.max(0.0).min((dim - 1) as f64);
        let high = high.max(0.0).min((dim - 1) as f64);

        let mut inv_freq = vec![0f32; half_dim];
        for i in 0..half_dim {
            let ramp = if (high - low).abs() < 1e-9 { 1.0 } else { ((i as f64 - low) / (high - low)).clamp(0.0, 1.0) as f32 };
            inv_freq[i] = inv_freq_interpolation[i] * ramp + inv_freq_extrapolation[i] * (1.0 - ramp);
        }

        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?.to_dtype(dtype)?;
        Ok(Self { inv_freq, attention_scaling })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let device = x.device();
        let t = Tensor::arange(0u32, s as u32, device)?.to_dtype(x.dtype())?.unsqueeze(1)?;
        let freqs = t.matmul(&self.inv_freq.unsqueeze(0)?)?;
        
        let cos = freqs.cos()?.affine(self.attention_scaling, 0.0)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = freqs.sin()?.affine(self.attention_scaling, 0.0)?.unsqueeze(0)?.unsqueeze(0)?;

        let x_pairs = x.reshape((b, h, s, d / 2, 2))?;
        let x0 = x_pairs.narrow(candle_core::D::Minus1, 0, 1)?.reshape((b, h, s, d / 2))?;
        let x1 = x_pairs.narrow(candle_core::D::Minus1, 1, 1)?.reshape((b, h, s, d / 2))?;

        let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let r1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;

        let interleaved = Tensor::cat(&[r0.unsqueeze(candle_core::D::Minus1)?, r1.unsqueeze(candle_core::D::Minus1)?], candle_core::D::Minus1)?;
        interleaved.reshape((b, h, s, d))
    }
}

struct SparseMoE {
    router_weight: Tensor,
    router_bias: Tensor,
    gate_up_proj: Tensor,
    gate_up_proj_bias: Tensor,
    down_proj: Tensor,
    down_proj_bias: Tensor,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
}

impl SparseMoE {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let router_weight = vb.get((cfg.num_local_experts, cfg.hidden_size), "router.weight")?.transpose(0, 1)?;
        let router_bias = vb.get(cfg.num_local_experts, "router.bias")?;
        let gate_up_proj = vb.get((cfg.num_local_experts, cfg.hidden_size, 2 * cfg.intermediate_size), "experts.gate_up_proj")?;
        let gate_up_proj_bias = vb.get((cfg.num_local_experts, 2 * cfg.intermediate_size), "experts.gate_up_proj_bias")?;
        let down_proj = vb.get((cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size), "experts.down_proj")?;
        let down_proj_bias = vb.get((cfg.num_local_experts, cfg.hidden_size), "experts.down_proj_bias")?;

        Ok(Self {
            router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let router_logits = (x_flat.matmul(&self.router_weight)?.broadcast_add(&self.router_bias))?;
        
        let top_k = router_logits.topk(self.num_experts_per_tok)?;
        let routing_weights = candle_nn::ops::softmax(&top_k.values, candle_core::D::Minus1)?;
        
        let mut final_out = Tensor::zeros((b * s, h), x.dtype(), x.device())?;
        
        let indices_flat = top_k.indices.flatten_all()?;
        let weights_flat = routing_weights.flatten_all()?;

        for eidx in 0..self.num_experts {
            let mask = indices_flat.eq(eidx as u32)?;
            let count = mask.to_dtype(DType::F32)?.sum_all()?.to_vec0::<f32>()? as usize;
            if count == 0 { continue; }

            let pos_indices = mask.where_cond(&Tensor::arange(0u32, (b * s * self.num_experts_per_tok) as u32, x.device())?, &Tensor::new(u32::MAX, x.device())?.broadcast_as(mask.shape())?)?;
            let mut pos_vec: Vec<u32> = pos_indices.to_device(&Device::Cpu)?.to_vec1()?;
            pos_vec.retain(|&i| i != u32::MAX);
            let pos_indices_tensor = Tensor::new(pos_vec.as_slice(), x.device())?;

            let token_indices = pos_indices_tensor.to_dtype(DType::F32)?.affine(1.0 / self.num_experts_per_tok as f64, 0.0)?.to_dtype(DType::U32)?;
            let expert_input = x_flat.index_select(&token_indices, 0)?;
            let expert_weights = weights_flat.index_select(&pos_indices_tensor, 0)?.to_dtype(x.dtype())?.unsqueeze(1)?;

            let gu_w = self.gate_up_proj.get(eidx)?;
            let gu_b = self.gate_up_proj_bias.get(eidx)?;
            let gate_up = (expert_input.matmul(&gu_w)?.broadcast_add(&gu_b))?;
            
            let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?.minimum(LIMIT as f64)?;
            let up = gate_up.narrow(candle_core::D::Minus1, self.intermediate_size, self.intermediate_size)?.clamp(-(LIMIT as f64), LIMIT as f64)?;
            let glu = (gate.clone() * candle_nn::ops::sigmoid(&(gate * ALPHA as f64)?)?)?;
            let expert_out_mid = (up.affine(1.0, 1.0)?.mul(&glu))?;
            
            let dp_w = self.down_proj.get(eidx)?;
            let dp_b = self.down_proj_bias.get(eidx)?;
            let expert_output = (expert_out_mid.matmul(&dp_w)?.broadcast_add(&dp_b))?;

            let weighted_output = expert_output.broadcast_mul(&expert_weights)?;
            final_out = final_out.index_add(&token_indices, &weighted_output, 0)?;
        }
        
        final_out.affine(self.num_experts_per_tok as f64, 0.0)?.reshape((b, s, h))
    }
}

struct TransformerLayer {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    sinks: Tensor,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    moe: SparseMoE,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f32,
}

impl TransformerLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, vb.pp("self_attn.q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.v_proj"))?;
        let o_proj = linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, vb.pp("self_attn.o_proj"))?;
        let sinks = vb.get(cfg.num_attention_heads, "self_attn.sinks")?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let moe = SparseMoE::new(cfg, vb.pp("mlp"))?;
        
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj, sinks, input_layernorm, post_attention_layernorm, moe,
            num_heads: cfg.num_attention_heads, num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim, scaling: (cfg.head_dim as f32).powf(-0.25),
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let x_norm = self.input_layernorm.forward(x)?;
        
        let q = self.q_proj.forward(&x_norm)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;

        let q = q.affine(self.scaling as f64, 0.0)?;
        let k = k.affine(self.scaling as f64, 0.0)?;
        
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 { repeat_kv(&k, n_rep)? } else { k };
        let v = if n_rep > 1 { repeat_kv(&v, n_rep)? } else { v };

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)?.broadcast_add(mask))?;
        let sinks = self.sinks.reshape((1, self.num_heads, 1, 1))?.expand((b, self.num_heads, s, 1))?;
        let combined = Tensor::cat(&[attn_weights, sinks], 3)?;
        
        let max_vals = combined.max_keepdim(3)?;
        let combined = combined.broadcast_sub(&max_vals)?;
        
        let probs = candle_nn::ops::softmax(&combined, 3)?;
        let scores = probs.narrow(3, 0, s)?.contiguous()?;
        
        let attn_out = scores.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, s, ()))?;
        let x_attn = (x + self.o_proj.forward(&attn_out)?)?;
        let x_norm = self.post_attention_layernorm.forward(&x_attn)?;
        x_attn + self.moe.forward(&x_norm)?
    }
}

pub fn repeat_kv(x: &Tensor, num_repeats: usize) -> candle_core::Result<Tensor> {
    if num_repeats == 1 { return Ok(x.clone()); }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, n_kv, num_repeats, s, d))?.flatten(1, 2)
}

pub fn create_sliding_window_mask(seq_len: usize, window_size: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if (i as isize - j as isize).abs() > window_size as isize {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
}

impl PrivacyFilterModel {
    pub fn get_label_list(&self) -> Vec<String> {
        let num_labels = self.config.num_labels();
        self.config.id2label.as_ref().map(|m| {
            let mut l = vec![String::new(); num_labels];
            for (id, name) in m {
                if let Ok(idx) = id.parse::<usize>() {
                    if idx < num_labels {
                        l[idx] = name.clone();
                    }
                }
            }
            l
        }).unwrap_or_else(crate::privacy_filter::config::build_label_list)
    }

    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let config = ModelConfig::from_file(&model_dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, &Device::Cpu)? };
        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb_cpu.pp("model.embed_tokens"))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, device)? };
        let layers = (0..config.num_hidden_layers).map(|i| TransformerLayer::new(&config, vb.pp(format!("model.layers.{}", i)))).collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let score_weight = vb.get((config.num_labels(), config.hidden_size), "score.weight")?.transpose(0, 1)?;
        let score_bias = vb.get(config.num_labels(), "score.bias")?;
        let viterbi_config = ViterbiConfig::from_file(&model_dir.join("viterbi_calibration.json"), "default").unwrap_or_default();

        Ok(Self { embed_tokens, layers, norm, score_weight, score_bias, tokenizer, config, viterbi_config, device: device.clone(), dtype })
    }

    pub fn forward(&self, input_ids: &[u32]) -> candle_core::Result<Tensor> {
        // Look up on CPU, then move to device
        let mut x = self.embed_tokens.forward(&Tensor::new(input_ids, &Device::Cpu)?)?.to_device(&self.device)?.unsqueeze(0)?;
        
        let rope = RotaryEmbedding::new_yarn(&self.config.rope_parameters, self.config.head_dim, input_ids.len(), self.dtype, &self.device).map_err(candle_core::Error::msg)?;
        let mask = create_sliding_window_mask(input_ids.len(), self.config.sliding_window, &self.device)?.to_dtype(self.dtype)?;

        for layer in &self.layers {
            x = layer.forward(&x, &rope, &mask)?;
        }

        let x = self.norm.forward(&x)?;
        x.matmul(&self.score_weight.unsqueeze(0)?)?.broadcast_add(&self.score_bias.unsqueeze(0)?.unsqueeze(0)?)
    }

    pub fn predict(&self, text: &str) -> Result<Vec<PrivacySpan>> {
        let tokens = self.tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let input_ids = tokens.get_ids();
        let s = input_ids.len();
        let logits = self.forward(input_ids).map_err(anyhow::Error::msg)?;
        let logits_data = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        
        let num_labels = self.config.num_labels();
        let label_list = self.config.id2label.as_ref().map(|m| {
            let mut l = vec![String::new(); num_labels];
            for (id, name) in m { l[id.parse::<usize>().unwrap()] = name.clone(); }
            l
        }).unwrap_or_else(crate::privacy_filter::config::build_label_list);

        Ok(viterbi::extract_spans(&viterbi::viterbi_decode(&logits_data, s, num_labels, &self.viterbi_config), &logits_data, num_labels, &label_list, &tokens.get_tokens().iter().map(|s| s.to_string()).collect::<Vec<_>>(), tokens.get_offsets(), text))
    }
}
