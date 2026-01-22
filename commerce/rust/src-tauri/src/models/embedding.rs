use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Module};
use candle_core::quantized::{gguf_file, QMatMul};
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

struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
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

struct QAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl QAttention {
    fn new(
        cfg: &Config,
        ct: &gguf_file::Content,
        reader: &mut std::fs::File,
        prefix: &str,
        dev: &Device,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let q_proj = load_linear_fallback(ct, reader, prefix, "q_proj", "attn_q", dev)?;
        let k_proj = load_linear_fallback(ct, reader, prefix, "k_proj", "attn_k", dev)?;
        let v_proj = load_linear_fallback(ct, reader, prefix, "v_proj", "attn_v", dev)?;
        let o_proj = load_linear_fallback(ct, reader, prefix, "o_proj", "attn_output", dev)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_flat = x_f32.reshape((batch_size * seq_len, ()))?;

        let q = self.q_proj.forward(&x_flat)?.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(&x_flat)?.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(&x_flat)?.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.rotary.forward(&q, &k, seq_len)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape((batch_size * seq_len, ()))?;
        
        let out = self.o_proj.forward(&y)?;
        out.reshape((batch_size, seq_len, ()))?.to_dtype(x.dtype())
    }
}

struct QMlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl QMlp {
    fn new(
        _cfg: &Config,
        ct: &gguf_file::Content,
        reader: &mut std::fs::File,
        prefix: &str,
        dev: &Device,
    ) -> Result<Self> {
        let gate_proj = load_linear(ct, reader, prefix, "gate_proj", "ffn_gate", dev)?;
        let up_proj = load_linear(ct, reader, prefix, "up_proj", "ffn_up", dev)?;
        let down_proj = load_linear(ct, reader, prefix, "down_proj", "ffn_down", dev)?;
        
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let (b, s, h) = x_f32.dims3()?;
        let x_flat = x_f32.reshape((b * s, h))?;

        let gate = self.gate_proj.forward(&x_flat)?;
        let up = self.up_proj.forward(&x_flat)?;
        let act = (gate.gelu_erf()? * up)?; 
        let out = self.down_proj.forward(&act)?;
        
        out.reshape((b, s, ()))?.to_dtype(x_dtype)
    }
}

struct QDecoderLayer {
    self_attn: QAttention,
    mlp: QMlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

impl QDecoderLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x_norm = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&x_norm)?;
        let x = (residual + attn_out)?;
        
        let residual = &x;
        let x_norm = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        residual + mlp_out
    }
}

pub struct QuantizedModel {
    embed_tokens: Tensor, // Dequantized for speed in embedding lookup
    layers: Vec<QDecoderLayer>,
    norm: QRmsNorm,
}

impl QuantizedModel {
    pub fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        // Lookup in embed_tokens
        let mut x = Tensor::embedding(input_ids, &self.embed_tokens)?;
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
    model: QuantizedModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let gguf_path = model_path.join("embeddinggemma-300m-Q4_0.gguf");
        let tokenizer_path = model_path.join("tokenizer.json");
        let config_path = model_path.join("config.json");

        let device = Device::Cpu; 
        
        let mut file = std::fs::File::open(&gguf_path).map_err(|e| anyhow!("Failed to open GGUF: {}", e))?;
        let ct = gguf_file::Content::read(&mut file)?;
        
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let rotary = Arc::new(RotaryEmbedding::new(config.head_dim, config.max_position_embeddings, config.rope_theta, &device)?);

        // [FIX] Robust Embedding Loading
        let embed_tokens = if let Ok(t) = ct.tensor(&mut file, "token_embd.weight", &device) {
            t.dequantize(&device)?
        } else {
            ct.tensor(&mut file, "model.embed_tokens.weight", &device)?.dequantize(&device)?
        };

        // Load Layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            // Check for GGUF standard 'blk.N' or original 'model.layers.N'
            let (prefix, is_gguf) = if ct.tensor_infos.contains_key(&format!("blk.{}.attn_norm.weight", i)) {
                (format!("blk.{}", i), true)
            } else {
                (format!("model.layers.{}", i), false)
            };

            let (in_ln, post_ln, attn, mlp) = if is_gguf {
                ("attn_norm", "ffn_norm", "", "ffn")
            } else {
                ("input_layernorm", "post_attention_layernorm", ".self_attn", ".mlp")
            };

            let input_layernorm = QRmsNorm::new(
                ct.tensor(&mut file, &format!("{}.{}.weight", prefix, in_ln), &device)?.dequantize(&device)?,
                config.rms_norm_eps
            );
            let post_attention_layernorm = QRmsNorm::new(
                ct.tensor(&mut file, &format!("{}.{}.weight", prefix, post_ln), &device)?.dequantize(&device)?,
                config.rms_norm_eps
            );
            
            let self_attn = QAttention::new(&config, &ct, &mut file, &format!("{}{}", prefix, attn), &device, rotary.clone())?;
            let mlp_prefix = if is_gguf { format!("{}.{}", prefix, mlp) } else { format!("{}{}", prefix, mlp) };
            let mlp = QMlp::new(&config, &ct, &mut file, &mlp_prefix, &device)?;
            
            layers.push(QDecoderLayer { self_attn, mlp, input_layernorm, post_attention_layernorm });
        }

        // Load final Norm
        let norm_name = if ct.tensor_infos.contains_key("output_norm.weight") { "output_norm.weight" } else { "model.norm.weight" };
        let norm = QRmsNorm::new(
            ct.tensor(&mut file, norm_name, &device)?.dequantize(&device)?,
            config.rms_norm_eps
        );

        let model = QuantizedModel { embed_tokens, layers, norm };

        Ok(Self { model, tokenizer, device })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        
        if token_ids.is_empty() { return Ok(vec![0.0; 768]); }

        // [REVISED] Set to exactly 1000 as requested
        let chunk_size = 1000;
        let chunks: Vec<&[u32]> = token_ids.chunks(chunk_size).collect();
        
        let mut accumulated_vector = vec![0.0; 768];
        let mut total_chunks = 0.0;

        for chunk in chunks {
            let input_tensor = Tensor::new(chunk, &self.device).map_err(anyhow::Error::msg)?;
            let hidden_states = self.model.forward(&input_tensor).map_err(anyhow::Error::msg)?;
            
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

// --- Helper Functions to Avoid Borrow Checker Conflicts ---

fn load_linear_fallback<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    suffix1: &str,
    suffix2: &str,
    device: &Device
) -> Result<QMatMul> {
    let name1 = format!("{}.{}.weight", prefix, suffix1);
    let name2 = format!("{}.{}.weight", prefix, suffix2);
    let tensor = if let Ok(t) = ct.tensor(reader, &name1, device) { t }
                 else if let Ok(t) = ct.tensor(reader, &name2, device) { t }
                 else { ct.tensor(reader, &format!("{}.weight", prefix), device)? };
    Ok(QMatMul::from_qtensor(tensor)?)
}

fn load_linear<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    suffix1: &str,
    suffix2: &str,
    device: &Device
) -> Result<QMatMul> {
    let name1 = format!("{}.{}.weight", prefix, suffix1);
    let name2 = format!("{}.{}.weight", prefix, suffix2);
    let tensor = if let Ok(t) = ct.tensor(reader, &name1, device) { t }
                 else { ct.tensor(reader, &name2, device)? };
    Ok(QMatMul::from_qtensor(tensor)?)
}