use anyhow::{Result};
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::{VarBuilder, linear_no_bias as linear};
use serde::Deserialize;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub pad_token_id: Option<usize>,
}

impl Config {
    pub fn gemma_300m() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 1152,
            num_hidden_layers: 24,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 262144,
            max_position_embeddings: 2048,
            pad_token_id: Some(0),
        }
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;
        let x = x.to_dtype(internal_dtype)?;
        let variance = x.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (theta.powf(i as f64 / dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        // FIX: [max_seq_len] -> [max_seq_len, 1] for correct broadcasting in matmul
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        
        let cos = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?;

        let apply_rotary = |x: &Tensor| -> candle_core::Result<Tensor> {
            let last_dim = x.dim(candle_core::D::Minus1)?;
            let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
            let x2 = x.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
            let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
            let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
            Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
        };

        Ok((apply_rotary(q)?, apply_rotary(k)?))
    }
}

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = (gate.gelu_erf()? * up)?; // GeGLU approx
        Ok(self.down_proj.forward(&act)?)
    }
}

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, rotary: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden_size = cfg.hidden_size;

        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self { q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim, rotary })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = self.rotary.forward(&q, &k, seq_len)?;

        // Repeat K/V for GQA
        let k = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        let att = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        
        Ok(self.o_proj.forward(&y)?)
    }
}

fn repeat_kv(x: &Tensor, num_repeats: usize) -> candle_core::Result<Tensor> {
    if num_repeats == 1 { return Ok(x.clone()); }
    let (b, n_kv, l, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.broadcast_as((b, n_kv, num_repeats, l, d))?.flatten(1, 2)?)
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder, rotary: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"), rotary)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x_norm = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&x_norm)?;
        let x = (residual + attn_out)?;
        
        let residual = &x;
        let x_norm = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        Ok((residual + mlp_out)?)
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device)?);
        
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{}", i)), rotary.clone())?);
        }
        
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        
        Ok(Self { embed_tokens, layers, norm })
    }

    pub fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        x = x.reshape((1, seq_len, ()))?;
        
        let scale = (x.dim(candle_core::D::Minus1)? as f64).sqrt();
        x = (x * scale)?;

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        
        self.norm.forward(&x)
    }
}

pub struct EmbeddingModel {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let tokenizer_path = model_path.join("tokenizer.json");
        let weights_path = model_path.join("model.safetensors");

        // Force CPU for Embedding to prevent VRAM OOM with Qwen
        let device = Device::Cpu;
        println!("[EmbeddingModel] Forcing CPU to save VRAM for Vision Model.");

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).map_err(anyhow::Error::msg)? };
        let model = Model::new(&config, vb, &device).map_err(anyhow::Error::msg)?;

        Ok(Self { model, tokenizer, device })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        
        if token_ids.is_empty() { return Ok(vec![0.0; 768]); }

        // [FIX] Use 1024 to stay within safe VRAM/RAM limits
        let chunk_size = 1024;
        let chunks: Vec<&[u32]> = token_ids.chunks(chunk_size).collect();
        
        let mut accumulated_vector = vec![0.0; 768];
        let mut total_chunks = 0.0;

        for chunk in chunks {
            // Pass Rank 1 Tensor [seq_len]
            let input_tensor = Tensor::new(chunk, &self.device).map_err(anyhow::Error::msg)?;
            let hidden_states = self.model.forward(&input_tensor).map_err(anyhow::Error::msg)?;
            
            let (_b, s, _h) = hidden_states.dims3().map_err(anyhow::Error::msg)?;
            let sum = hidden_states.sum(1).map_err(anyhow::Error::msg)?; 
            let mean = (sum / (s as f64)).map_err(anyhow::Error::msg)?;
            
            // Normalize the chunk vector
            let norm = mean.sqr().map_err(anyhow::Error::msg)?.sum_all().map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
            let normalized = mean.broadcast_div(&norm).map_err(anyhow::Error::msg)?;
            
            let vec: Vec<f32> = normalized.flatten_all().map_err(anyhow::Error::msg)?.to_vec1().map_err(anyhow::Error::msg)?;
            
            // Accumulate
            for (i, val) in vec.iter().enumerate() {
                accumulated_vector[i] += val;
            }
            total_chunks += 1.0;
        }

        // Average Pooling
        if total_chunks > 0.0 {
            for val in accumulated_vector.iter_mut() {
                *val /= total_chunks;
            }
            
            // Re-normalize the final averaged vector (Optional but recommended for cosine similarity)
            let sum_sq: f32 = accumulated_vector.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > 1e-6 {
                for val in accumulated_vector.iter_mut() {
                    *val /= norm;
                }
            }
        }

        Ok(accumulated_vector)
    }
}
