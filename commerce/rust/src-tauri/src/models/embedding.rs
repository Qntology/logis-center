use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::{VarBuilder, linear_no_bias as linear};
use candle_core::quantized::{gguf_file, QMatMul};
use serde::Deserialize;
use std::sync::Arc;
use tokenizers::Tokenizer;
use std::path::Path;

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

// --- Common Layers ---

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = if x_dtype == DType::BF16 { DType::BF16 } else { DType::F32 };
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

fn repeat_kv(x: &Tensor, num_repeats: usize) -> candle_core::Result<Tensor> {
    if num_repeats == 1 { return Ok(x.clone()); }
    let (b, n_kv, l, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.broadcast_as((b, n_kv, num_repeats, l, d))?.flatten(1, 2)?)
}

// --- Float Model ---

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
        let act = (gate.gelu_erf()? * up)?; 
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
        let q_proj = linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, num_heads: cfg.num_attention_heads, num_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rotary })
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
        let k = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        let att = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&y)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder, rotary: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"), rotary.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }
}

pub struct FloatModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl FloatModel {
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        let vb_cpu = vb.clone().set_device(Device::Cpu);
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_cpu.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device)?);
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{}", i)), rotary.clone())?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self { embed_tokens, layers, norm })
    }

    pub fn forward(&self, input_ids: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        // Lookup on CPU, then move to device
        let mut x = self.embed_tokens.forward(input_ids)?.to_device(device)?;
        x = x.reshape((1, seq_len, ()))?;
        let scale = (x.dim(candle_core::D::Minus1)? as f64).sqrt();
        x = (x * scale)?;
        for layer in &self.layers {
            let residual = &x;
            let x_norm = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&x_norm)?;
            let x_attn = (residual + attn_out)?;
            let residual = &x_attn;
            let x_norm = layer.post_attention_layernorm.forward(&x_attn)?;
            let mlp_out = layer.mlp.forward(&x_norm)?;
            x = (residual + mlp_out)?;
        }
        self.norm.forward(&x)
    }
}

// --- Quantized Model ---

struct QLinear {
    inner: QMatMul,
}

impl QLinear {
    fn from_gguf<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device) -> Result<Self> {
        let tensor = ct.tensor(reader, name, device)?;
        let qmm = QMatMul::from_qtensor(tensor)?;
        Ok(Self { inner: qmm })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let out = self.inner.forward(&x_flat)?;
        let out = out.reshape((b, s, ()))?;
        out.to_dtype(x_dtype)
    }
}

struct QuantizedMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QuantizedMlp {
    fn new<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, layer_idx: usize, device: &Device) -> Result<Self> {
        // Try different naming conventions
        let prefix = format!("blk.{}.", layer_idx);
        let gate = QLinear::from_gguf(ct, reader, &format!("{}ffn_gate.weight", prefix), device)?;
        let up = QLinear::from_gguf(ct, reader, &format!("{}ffn_up.weight", prefix), device)?;
        let down = QLinear::from_gguf(ct, reader, &format!("{}ffn_down.weight", prefix), device)?;
        Ok(Self { gate_proj: gate, up_proj: up, down_proj: down })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = (gate.gelu_erf()? * up)?; 
        Ok(self.down_proj.forward(&act)?)
    }
}

struct QuantizedAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl QuantizedAttention {
    fn new<R: std::io::Seek + std::io::Read>(cfg: &Config, ct: &gguf_file::Content, reader: &mut R, layer_idx: usize, device: &Device, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        let prefix = format!("blk.{}.", layer_idx);
        let q = QLinear::from_gguf(ct, reader, &format!("{}attn_q.weight", prefix), device)?;
        let k = QLinear::from_gguf(ct, reader, &format!("{}attn_k.weight", prefix), device)?;
        let v = QLinear::from_gguf(ct, reader, &format!("{}attn_v.weight", prefix), device)?;
        let o = QLinear::from_gguf(ct, reader, &format!("{}attn_output.weight", prefix), device)?;
        Ok(Self { q_proj: q, k_proj: k, v_proj: v, o_proj: o, num_heads: cfg.num_attention_heads, num_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rotary })
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
        let k = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        let att = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&y)?)
    }
}

struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDecoderLayer {
    fn new<R: std::io::Seek + std::io::Read>(cfg: &Config, ct: &gguf_file::Content, reader: &mut R, layer_idx: usize, device: &Device, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        let self_attn = QuantizedAttention::new(cfg, ct, reader, layer_idx, device, rotary)?;
        let mlp = QuantizedMlp::new(ct, reader, layer_idx, device)?;
        
        let prefix = format!("blk.{}.", layer_idx);
        
        
        let t_in = ct.tensor(reader, &format!("{}attn_norm.weight", prefix), &Device::Cpu)?;
        let input_ln_w = t_in.dequantize_f16(&Device::Cpu).or_else(|_| t_in.dequantize(&Device::Cpu))?.to_dtype(candle_core::DType::F32)?.to_device(device)?;
        
        let t_post = ct.tensor(reader, &format!("{}ffn_norm.weight", prefix), &Device::Cpu)?;
        let post_ln_w = t_post.dequantize_f16(&Device::Cpu).or_else(|_| t_post.dequantize(&Device::Cpu))?.to_dtype(candle_core::DType::F32)?.to_device(device)?;
        
        Ok(Self { 
            self_attn, 
            mlp, 
            input_layernorm: RmsNorm::from_tensor(input_ln_w, cfg.rms_norm_eps),
            post_attention_layernorm: RmsNorm::from_tensor(post_ln_w, cfg.rms_norm_eps)
        })
    }
}

pub struct QuantizedModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
}

impl QuantizedModel {
    pub fn new<R: std::io::Seek + std::io::Read>(cfg: &Config, ct: &gguf_file::Content, reader: &mut R, device: &Device) -> Result<Self> {
        
        let t_emb = ct.tensor(reader, "token_embd.weight", &Device::Cpu)?;
        let tok_emb = t_emb.dequantize_f16(&Device::Cpu).or_else(|_| t_emb.dequantize(&Device::Cpu))?.to_dtype(candle_core::DType::F32)?;
        let embed_tokens = candle_nn::Embedding::new(tok_emb, cfg.hidden_size);
        
        let rotary = Arc::new(RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device)?);
        
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedDecoderLayer::new(cfg, ct, reader, i, device, rotary.clone())?);
        }
        
        
        let t_norm = ct.tensor(reader, "output_norm.weight", &Device::Cpu)?;
        let norm_w = t_norm.dequantize_f16(&Device::Cpu).or_else(|_| t_norm.dequantize(&Device::Cpu))?.to_dtype(candle_core::DType::F32)?.to_device(device)?;
        let norm = RmsNorm::from_tensor(norm_w, cfg.rms_norm_eps);
        
        Ok(Self { embed_tokens, layers, norm })
    }

    pub fn forward(&self, input_ids: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        // Lookup on CPU, then move result to device
        let mut x = self.embed_tokens.forward(input_ids)?.to_device(device)?;
        x = x.reshape((1, seq_len, ()))?;
        let scale = (x.dim(candle_core::D::Minus1)? as f64).sqrt();
        x = (x * scale)?;
        for layer in &self.layers {
            let residual = &x;
            let x_norm = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&x_norm)?;
            let x_attn = (residual + attn_out)?;
            let residual = &x_attn;
            let x_norm = layer.post_attention_layernorm.forward(&x_attn)?;
            let mlp_out = layer.mlp.forward(&x_norm)?;
            x = (residual + mlp_out)?;
        }
        self.norm.forward(&x)
    }
}

// --- Wrapper ---

enum ModelEnum {
    Float(FloatModel),
    Quantized(QuantizedModel),
}

pub struct EmbeddingModel {
    model: ModelEnum,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        // Default to CPU for safety on 4GB cards
        Self::new_with_device(model_path, &Device::Cpu)
    }

    pub fn new_with_device<P: AsRef<Path>>(model_path: P, device: &Device) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let tokenizer_path = model_path.join("tokenizer.json");
        let weights_path = model_path.join("model.safetensors");
        let gguf_path = model_path.join("embeddinggemma-300m-Q4_0.gguf");

        println!("[EmbeddingModel] Loading on {:?}...", device);

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        let model_enum = if gguf_path.exists() {
            println!("[EmbeddingModel] Loading GGUF model from {:?}", gguf_path);
            let mut file = std::fs::File::open(&gguf_path)?;
            let content = gguf_file::Content::read(&mut file)?;
            let model = QuantizedModel::new(&config, &content, &mut file, device)?;
            ModelEnum::Quantized(model)
        } else if weights_path.exists() {
             println!("[EmbeddingModel] Loading Safetensors model from {:?}", weights_path);
             let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device).map_err(anyhow::Error::msg)? };
             let model = FloatModel::new(&config, vb, device).map_err(anyhow::Error::msg)?;
             ModelEnum::Float(model)
        } else {
            return Err(anyhow!("No valid model found (looked for model.safetensors or embeddinggemma-300m-Q4_0.gguf)"));
        };

        Ok(Self { model: model_enum, tokenizer, device: device.clone() })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        
        if token_ids.is_empty() { return Ok(vec![0.0; 768]); }

        // VRAM 피크 방지를 위해 청크 사이즈를 512로 축소
        let chunk_size = 512;
        let chunks: Vec<&[u32]> = token_ids.chunks(chunk_size).collect();
        
        let mut accumulated_vector = vec![0.0; 768];
        let mut total_chunks = 0.0;

        for chunk in chunks {
            // input_tensor를 CPU에 생성 (임베딩 룩업용)
            let input_tensor = Tensor::new(chunk, &Device::Cpu).map_err(anyhow::Error::msg)?;
            let hidden_states = match &self.model {
                ModelEnum::Float(m) => m.forward(&input_tensor, &self.device).map_err(anyhow::Error::msg)?,
                ModelEnum::Quantized(m) => m.forward(&input_tensor, &self.device).map_err(anyhow::Error::msg)?,
            };
            
            let (_b, s, _h) = hidden_states.dims3().map_err(anyhow::Error::msg)?;
            let sum = hidden_states.sum(1).map_err(anyhow::Error::msg)?; 
            let mean = (sum / (s as f64)).map_err(anyhow::Error::msg)?;
            
            let norm = mean.sqr().map_err(anyhow::Error::msg)?.sum_all().map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
            let normalized = mean.broadcast_div(&norm).map_err(anyhow::Error::msg)?;
            
            let vec: Vec<f32> = normalized.flatten_all().map_err(anyhow::Error::msg)?.to_vec1().map_err(anyhow::Error::msg)?;
            
            for (i, val) in vec.iter().enumerate() {
                accumulated_vector[i] += val;
            }
            total_chunks += 1.0;
        }

        if total_chunks > 0.0 {
            for val in accumulated_vector.iter_mut() {
                *val /= total_chunks;
            }
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
