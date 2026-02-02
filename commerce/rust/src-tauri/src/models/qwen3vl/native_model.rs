use std::sync::Arc;
use memmap2::Mmap;
use crate::models::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig};
use crate::models::qwen3vl::native_backend::*;
use half::f16;
use safetensors::SafeTensors;
use anyhow::{Result, anyhow};

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

pub struct NativeLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub variant: LinearVariant,
}

impl NativeLinear {
    pub fn forward(&self, x: &[f16]) -> Vec<f16> {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                let w = weight.get_slice::<f16>();
                let b = bias.as_ref().map(|t| t.get_slice::<f16>());
                native_linear_f16(x, w, b, m, self.out_features, self.in_features)
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                let wp = weight_packed.get_slice::<u32>();
                let s = scales.get_slice::<f16>();
                let b = bias.as_ref().map(|t| t.get_slice::<f16>());
                
                let x_f32: Vec<f32> = x.iter().map(|&v| v.to_f32()).collect();
                let mut out_f32 = bit_serial_matmul_f32(&x_f32, wp, s, m, self.out_features, self.in_features);
                
                if let Some(bias_data) = b {
                    for i in 0..m {
                        for j in 0..self.out_features {
                            out_f32[i * self.out_features + j] += bias_data[j].to_f32();
                        }
                    }
                }
                out_f32.into_iter().map(f16::from_f32).collect()
            }
        }
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor,
    pub post_attention_layernorm: NativeTensor,
    pub q_proj: NativeLinear,
    pub k_proj: NativeLinear,
    pub v_proj: NativeLinear,
    pub o_proj: NativeLinear,
    pub gate_proj: NativeLinear,
    pub up_proj: NativeLinear,
    pub down_proj: NativeLinear,
    pub kv_cache: std::sync::Mutex<Option<(Vec<f16>, Vec<f16>)>>, 
}

impl NativeLayer {
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize) -> Vec<f16> {
        let hidden_size = config.hidden_size;
        let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;

        let residual = x.to_vec();
        let norm_w = self.input_layernorm.get_slice::<f16>();
        let x_norm = native_rms_norm_f16(x, norm_w, config.rms_norm_eps as f32, hidden_size);
        
        let mut q = self.q_proj.forward(&x_norm);
        let mut k = self.k_proj.forward(&x_norm);
        let v = self.v_proj.forward(&x_norm);

        native_apply_rope_f16_with_offset(&mut q, &mut k, q_len, seqlen_offset, n_heads, head_dim, config.rope_theta);

        let mut cache_guard = self.kv_cache.lock().unwrap();
        let (k_full, v_full) = if let Some((prev_k, prev_v)) = cache_guard.take() {
            let mut new_k = prev_k;
            let mut new_v = prev_v;
            new_k.extend_from_slice(&k);
            new_v.extend_from_slice(&v);
            (new_k, new_v)
        } else {
            (k, v)
        };
        let total_seq_len = k_full.len() / (n_kv_heads * head_dim);
        *cache_guard = Some((k_full.clone(), v_full.clone()));
        drop(cache_guard);

        let attn_out_vec = native_vision_attn_f16(&q, &k_full, &v_full, hidden_size, n_heads, q_len, total_seq_len);
        
        let mut x_attn = self.o_proj.forward(&attn_out_vec);
        for i in 0..x_attn.len() { x_attn[i] += residual[i]; }

        let residual_mlp = x_attn.clone();
        let norm_w_mlp = self.post_attention_layernorm.get_slice::<f16>();
        let x_norm_mlp = native_rms_norm_f16(&x_attn, norm_w_mlp, config.rms_norm_eps as f32, hidden_size);

        let mut gate = self.gate_proj.forward(&x_norm_mlp);
        let up = self.up_proj.forward(&x_norm_mlp);
        native_silu_f16(&mut gate);
        for i in 0..gate.len() { gate[i] *= up[i]; }

        let mut x_mlp = self.down_proj.forward(&gate);
        for i in 0..x_mlp.len() { x_mlp[i] += residual_mlp[i]; }

        x_mlp
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig,
    pub embed_tokens: NativeTensor,
    pub layers: Vec<NativeLayer>,
    pub norm: NativeTensor,
}

impl NativeQwen3TextModel {
    pub fn forward(&self, input_ids: &[u32], seqlen_offset: usize) -> Vec<f16> {
        let hidden_size = self.config.hidden_size;
        let table = self.embed_tokens.get_slice::<f16>();
        let mut x = native_embedding_lookup_f16(input_ids, table, hidden_size);
        for layer in &self.layers {
            x = layer.forward(&x, &self.config, seqlen_offset);
        }
        let norm_w = self.norm.get_slice::<f16>();
        native_rms_norm_f16(&x, norm_w, self.config.rms_norm_eps as f32, hidden_size)
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            let mut cache = layer.kv_cache.lock().unwrap();
            *cache = None;
        }
    }
}

pub struct NativeVisionModel {
    pub patch_embed: NativeLinear,
    pub pos_embed: NativeTensor,
    pub blocks: Vec<NativeLayer>, 
    pub merger: NativeLayer,
}

impl NativeVisionModel {
    pub fn forward(&self, _pixel_values: &[f16], _grid_thw: &[u32; 3]) -> Vec<f16> {
        vec![] // Placeholder
    }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig,
    pub text_model: NativeQwen3TextModel,
    pub vision_model: NativeVisionModel,
    pub lm_head: NativeLinear,
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, main_mmap: Arc<Mmap>, _vision_mmap: Arc<Mmap>) -> Result<Self> {
        let st = SafeTensors::deserialize(&main_mmap)?;
        let t_cfg = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        
        let get_t = |name: &str| -> Result<NativeTensor> {
            let view = st.tensor(name)?;
            let data_ptr = view.data().as_ptr();
            let mmap_ptr = main_mmap.as_ptr();
            let offset = unsafe { data_ptr.offset_from(mmap_ptr) } as usize;
            Ok(NativeTensor::from_mmap(main_mmap.clone(), offset, view.shape().to_vec(), NativeDType::F16))
        };

        let embed_tokens = get_t("model.embed_tokens.weight")?;
        let mut layers = Vec::new();
        for i in 0..t_cfg.num_hidden_layers {
            let p = format!("model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p))?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p))?,
                q_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.hidden_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.self_attn.q_proj.packed", p))?, scales: get_t(&format!("{}.self_attn.q_proj.scales", p))?, bias: None } },
                k_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.hidden_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.self_attn.k_proj.packed", p))?, scales: get_t(&format!("{}.self_attn.k_proj.scales", p))?, bias: None } },
                v_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.hidden_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.self_attn.v_proj.packed", p))?, scales: get_t(&format!("{}.self_attn.v_proj.scales", p))?, bias: None } },
                o_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.hidden_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.self_attn.o_proj.packed", p))?, scales: get_t(&format!("{}.self_attn.o_proj.scales", p))?, bias: None } },
                gate_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.intermediate_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.mlp.gate_proj.packed", p))?, scales: get_t(&format!("{}.mlp.gate_proj.scales", p))?, bias: None } },
                up_proj: NativeLinear { in_features: t_cfg.hidden_size, out_features: t_cfg.intermediate_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.mlp.up_proj.packed", p))?, scales: get_t(&format!("{}.mlp.up_proj.scales", p))?, bias: None } },
                down_proj: NativeLinear { in_features: t_cfg.intermediate_size, out_features: t_cfg.hidden_size, variant: LinearVariant::BitSerial { weight_packed: get_t(&format!("{}.mlp.down_proj.packed", p))?, scales: get_t(&format!("{}.mlp.down_proj.scales", p))?, bias: None } },
                kv_cache: std::sync::Mutex::new(None),
            });
        }
        let norm = get_t("model.norm.weight")?;
        let lm_head = NativeLinear { in_features: t_cfg.hidden_size, out_features: 151936, variant: LinearVariant::Standard { weight: get_t("lm_head.weight")?, bias: None } };

        Ok(Self {
            config: config.clone(),
            text_model: NativeQwen3TextModel { config: t_cfg.clone(), embed_tokens, layers, norm },
            vision_model: unsafe { std::mem::zeroed() }, 
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &[u32], _pixel_values: Option<&[f16]>, _grid_thw: Option<&[u32; 3]>, seqlen_offset: usize) -> Vec<f16> {
        let x = self.text_model.forward(input_ids, seqlen_offset);
        self.lm_head.forward(&x)
    }

    pub fn clear_kv_cache(&self) {
        self.text_model.clear_kv_cache();
    }
}
