use std::sync::Arc;
use memmap2::Mmap;
use crate::models::qwen3vl::native_backend::*;
use crate::models::qwen3vl::native_model::{NativeLayer, NativeLinear, LinearVariant};
use crate::models::qwen3vl::config::Qwen3VLTextConfig;
use anyhow::{Result, anyhow};
use half::f16;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub struct NativeEmbeddingModel {
    pub tokenizer: Tokenizer,
    pub embed_tokens: NativeTensor,
    pub layers: Vec<NativeLayer>,
    pub norm: NativeTensor,
    pub hidden_size: usize,
}

impl NativeEmbeddingModel {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let weights_path = path.join("model.safetensors");
        let tokenizer_path = path.join("tokenizer.json");
        let config_path = path.join("config.json");

        let file = std::fs::File::open(weights_path)?;
        let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? });
        let st = SafeTensors::deserialize(&mmap)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
        
        let config_str = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(768) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(24) as usize;

        let get_t = |name: &str| -> Result<NativeTensor> {
            let view = st.tensor(name)?;
            let data_ptr = view.data().as_ptr();
            let mmap_ptr = mmap.as_ptr();
            let offset = unsafe { data_ptr.offset_from(mmap_ptr) } as usize;
            Ok(NativeTensor::from_mmap(mmap.clone(), offset, view.shape().to_vec(), NativeDType::F16))
        };

        let embed_tokens = get_t("model.embed_tokens.weight")?;
        
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p))?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p))?,
                q_norm: None,
                k_norm: None,
                q_proj: NativeLinear { in_features: hidden_size, out_features: hidden_size, variant: LinearVariant::Standard { weight: get_t(&format!("{}.self_attn.q_proj.weight", p))?, bias: None } },
                k_proj: NativeLinear { in_features: hidden_size, out_features: hidden_size, variant: LinearVariant::Standard { weight: get_t(&format!("{}.self_attn.k_proj.weight", p))?, bias: None } },
                v_proj: NativeLinear { in_features: hidden_size, out_features: hidden_size, variant: LinearVariant::Standard { weight: get_t(&format!("{}.self_attn.v_proj.weight", p))?, bias: None } },
                o_proj: NativeLinear { in_features: hidden_size, out_features: hidden_size, variant: LinearVariant::Standard { weight: get_t(&format!("{}.self_attn.o_proj.weight", p))?, bias: None } },
                gate_proj: NativeLinear { in_features: hidden_size, out_features: 1152, variant: LinearVariant::Standard { weight: get_t(&format!("{}.mlp.gate_proj.weight", p))?, bias: None } },
                up_proj: NativeLinear { in_features: hidden_size, out_features: 1152, variant: LinearVariant::Standard { weight: get_t(&format!("{}.mlp.up_proj.weight", p))?, bias: None } },
                down_proj: NativeLinear { in_features: 1152, out_features: hidden_size, variant: LinearVariant::Standard { weight: get_t(&format!("{}.mlp.down_proj.weight", p))?, bias: None } },
                kv_cache: std::sync::Mutex::new(None),
            });
        }

        let norm = get_t("model.norm.weight")?;

        Ok(Self { tokenizer, embed_tokens, layers, norm, hidden_size })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(|e| anyhow!(e))?;
        let token_ids = tokens.get_ids();
        if token_ids.is_empty() { return Ok(vec![0.0; 768]); }

        let cfg = Qwen3VLTextConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            intermediate_size: 1152,
            num_hidden_layers: self.layers.len(),
            vocab_size: 262144,
            max_position_embeddings: 2048,
            dtype: None,
            rope_scaling: None,
        };

        let table = self.embed_tokens.get_slice::<f16>();
        let mut x = native_embedding_lookup_f16(token_ids, table, self.hidden_size);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cfg, 0, i);
        }

        let norm_w = self.norm.get_slice::<f16>();
        let x_norm = native_rms_norm_f16(&x, norm_w, 1e-6, self.hidden_size);
        
        let mut pooled = vec![0.0f32; self.hidden_size];
        for i in 0..token_ids.len() {
            for d in 0..self.hidden_size {
                pooled[d] += x_norm[i * self.hidden_size + d].to_f32();
            }
        }
        for d in 0..self.hidden_size { pooled[d] /= token_ids.len() as f32; }

        Ok(pooled)
    }
}
