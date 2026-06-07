use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::{VarBuilder, linear_no_bias as linear};
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
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub pad_token_id: Option<usize>,
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
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (theta.powf(i as f64 / dim as f64) as f32))
            .collect();
        // VRAM 메모리 절약을 위해 초기화 시점부터 디바이스의 데이터 타입(GPU=BF16, CPU=F32)으로 캐스팅하여 저장합니다.
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(dtype)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        
        // 초기화 시점부터 이미 q.dtype()과 동일하게 맞춰져 있으므로 매번 실행되던 캐스팅 오버헤드를 제거합니다.
        
        let cos = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?;

        let apply_rotary = |x: &Tensor| -> candle_core::Result<Tensor> {
            let last_dim = x.dim(candle_core::D::Minus1)?;
            // narrow 이후 메모리 연속성 보장
            let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?.contiguous()?;
            let x2 = x.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?.contiguous()?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
            let cos = cos.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
            let sin = sin.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
            Ok((x.contiguous()?.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
        };

        Ok((apply_rotary(q)?, apply_rotary(k)?))
    }
}

// (ModernBERT 아키텍처로 넘어가면서 repeat_kv 함수가 제거되었습니다.)

// --- Float Model ---

struct Mlp {
    wi: candle_nn::Linear,
    wo: candle_nn::Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let wi = linear(hidden_size, intermediate_size * 2, vb.pp("Wi"))?;
        let wo = linear(intermediate_size, hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let wi_out = self.wi.forward(x)?;
        let last_dim = wi_out.dim(candle_core::D::Minus1)?;
        // narrow 이후 메모리 연속성 보장
        let gate = wi_out.narrow(candle_core::D::Minus1, 0, last_dim / 2)?.contiguous()?;
        let up = wi_out.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?.contiguous()?;
        // 연산 결과에 대해 Linear 계층(matmul) 진입 전 연속성 보장
        let act = (candle_nn::ops::silu(&gate)? * up)?.contiguous()?; 
        Ok(self.wo.forward(&act)?)
    }
}

struct Attention {
    wqkv: candle_nn::Linear,
    wo: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, rotary: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let wqkv = linear(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("Wqkv"))?;
        let wo = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wqkv, wo, num_heads: cfg.num_attention_heads, head_dim, rotary })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let qkv = self.wqkv.forward(x)?;
        
        let hidden_size = self.num_heads * self.head_dim;
        // narrow 이후 메모리 연속성 보장
        let q_all = qkv.narrow(2, 0, hidden_size)?.contiguous()?;
        let k_all = qkv.narrow(2, hidden_size, hidden_size)?.contiguous()?;
        let v_all = qkv.narrow(2, 2 * hidden_size, hidden_size)?.contiguous()?;

        // reshape 및 transpose 이후 메모리 연속성 보장
        let q = q_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.rotary.forward(&q, &k, seq_len)?;

        // matmul 진입 전 q와 k.transpose의 연속성 완벽 보장
        let q_cont = q.contiguous()?;
        let k_trans_cont = k.transpose(2, 3)?.contiguous()?;

        let att = (q_cont.matmul(&k_trans_cont)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        
        // 두 번째 matmul 진입 전 att와 v의 연속성 보장, 그리고 최종 reshape 전 transpose 결과의 연속성 보장
        let y = att.contiguous()?.matmul(&v.contiguous()?)?.transpose(1, 2)?.contiguous()?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.wo.forward(&y)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    attn_norm: Option<RmsNorm>,
    mlp_norm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder, rotary: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("attn"), rotary.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        // 🌟 [CRITICAL FIX] Granite(ModernBERT)는 layers.0에서 attn_norm을 생략하므로 Option(.ok())으로 안전하게 받습니다.
        let attn_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("attn_norm")).ok();
        let mlp_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self { self_attn, mlp, attn_norm, mlp_norm })
    }
}

pub struct FloatModel {
    tok_embeddings: candle_nn::Embedding,
    embeddings_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
}

impl FloatModel {
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        // CPU 강제 할당을 제거하고 전달받은 디바이스(VRAM)와 데이터 타입(BF16)을 그대로 사용합니다.
        let tok_embeddings = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embeddings.tok_embeddings"))?;
        let embeddings_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("embeddings.norm"))?;
        
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 32768, 150000.0, device)?);
        
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{}", i)), rotary.clone())?);
        }
        let final_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("final_norm"))?;
        Ok(Self { tok_embeddings, embeddings_norm, layers, final_norm })
    }

    pub fn forward(&self, input_ids: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        // 임베딩 텐서가 이미 VRAM에 올라가 있으므로 to_device() 복사 오버헤드를 제거합니다.
        let mut x = self.tok_embeddings.forward(input_ids)?;
        x = x.reshape((1, seq_len, ()))?;
        
        x = self.embeddings_norm.forward(&x)?;
        
        for layer in &self.layers {
            let residual = &x;
            // 🌟 [CRITICAL FIX] attn_norm이 없는 레이어(layers.0)는 직전의 x를 그대로 사용합니다.
            let x_norm = match &layer.attn_norm {
                Some(norm) => norm.forward(&x)?,
                None => x.clone(),
            };
            let attn_out = layer.self_attn.forward(&x_norm)?;
            x = (residual + attn_out)?;
            
            let residual = &x;
            let x_norm = layer.mlp_norm.forward(&x)?;
            let mlp_out = layer.mlp.forward(&x_norm)?;
            x = (residual + mlp_out)?;
        }
        self.final_norm.forward(&x)
    }
}

// --- Wrapper ---

pub struct EmbeddingModel {
    model: FloatModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_device(model_path, &Device::Cpu)
    }

    pub fn new_with_device<P: AsRef<Path>>(model_path: P, device: &Device) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let tokenizer_path = model_path.join("tokenizer.json");
        let weights_path = model_path.join("model.safetensors");

        println!("[EmbeddingModel] Loading on {:?}...", device);

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        let model = if weights_path.exists() {
             println!("[EmbeddingModel] Loading Safetensors model from {:?}", weights_path);
             let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], dtype, device).map_err(anyhow::Error::msg)? };
             FloatModel::new(&config, vb, device).map_err(anyhow::Error::msg)?
        } else {
            return Err(anyhow!("No valid model found (looked for model.safetensors)"));
        };

        Ok(Self { model, tokenizer, device: device.clone() })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        
        if token_ids.is_empty() { return Ok(vec![0.0; 384]); }

        let chunk_size = 512;
        let chunks: Vec<&[u32]> = token_ids.chunks(chunk_size).collect();
        
        let mut accumulated_vector = vec![0.0; 384];
        let mut total_chunks = 0.0;

        for chunk in chunks {
            // 입력 텐서를 처음부터 타겟 디바이스(VRAM)로 올립니다.
            let input_tensor = Tensor::new(chunk, &self.device).map_err(anyhow::Error::msg)?;
            let hidden_states = self.model.forward(&input_tensor, &self.device).map_err(anyhow::Error::msg)?;
            
            let (_b, s, _h) = hidden_states.dims3().map_err(anyhow::Error::msg)?;
            let sum = hidden_states.sum(1).map_err(anyhow::Error::msg)?; 
            let mean = (sum / (s as f64)).map_err(anyhow::Error::msg)?;
            
            let norm = mean.sqr().map_err(anyhow::Error::msg)?.sum_all().map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
            let normalized = mean.broadcast_div(&norm).map_err(anyhow::Error::msg)?;
            
            // 🌟 [CRITICAL FIX] VRAM(GPU)에 존재하는 최종 연산 결과 텐서를 시스템 메모리(CPU)로 안전하게 명시적 복사한 뒤, F32 배열로 캐스팅합니다.
            let normalized_cpu = normalized.to_device(&Device::Cpu).map_err(anyhow::Error::msg)?;
            let normalized_f32 = normalized_cpu.to_dtype(candle_core::DType::F32).map_err(anyhow::Error::msg)?;
            let vec: Vec<f32> = normalized_f32.flatten_all().map_err(anyhow::Error::msg)?.to_vec1().map_err(anyhow::Error::msg)?;
            
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

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }
        
        let mut results = vec![vec![0.0; 384]; texts.len()];
        
        // 🌟 [CRITICAL FIX] 장치(Device) 타입에 따른 스레드 동적 분기 및 VRAM 최적화
        // GPU(CUDA)일 경우 워프 스케줄링 한계와 PCIe 대역폭 병목을 방지하기 위해 가장 이상적인 3개로 제한합니다.
        // CPU일 경우 스레드 경합(Thread Contention)과 L3 캐시 오염을 막기 위해 무조건 1개(직렬)로 강제 고정합니다.
        let num_threads = if self.device.is_cpu() {
            1
        } else {
            3.min(texts.len())
        }.max(1); 
        
        let chunk_size = (texts.len() + num_threads - 1) / num_threads; 
        
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            
            for (chunk_idx, chunk) in texts.chunks(chunk_size).enumerate() {
                let start_idx = chunk_idx * chunk_size;
                handles.push(s.spawn(move || {
                    let mut local_res = Vec::with_capacity(chunk.len());
                    for text in chunk {
                        local_res.push(self.embed(text).unwrap_or(vec![0.0; 384]));
                    }
                    (start_idx, local_res)
                }));
            }
            
            for handle in handles {
                if let Ok((start_idx, local_res)) = handle.join() {
                    for (i, vector) in local_res.into_iter().enumerate() {
                        if start_idx + i < results.len() {
                            results[start_idx + i] = vector;
                        }
                    }
                }
            }
        });
        
        // 🌟 [MEMORY CLEAR] 임베딩 배치 연산 종료 후 즉각적인 VRAM 동기화 및 가비지 텐서 회수
        if !self.device.is_cpu() {
            let _ = self.device.synchronize();
        }
        
        Ok(results)
    }
}
