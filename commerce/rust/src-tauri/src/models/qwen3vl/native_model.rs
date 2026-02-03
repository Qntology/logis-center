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
    pub device_id: i32,
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
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 && weight_packed.gpu_ptr.is_some() {
                    let mut res = bit_serial_matmul_gpu(x, weight_packed, scales, m, self.out_features, self.in_features, self.device_id as usize);
                    if let Some(b) = bias {
                        let b_data = b.get_slice::<f16>();
                        for i in 0..m { for j in 0..self.out_features { res[i * self.out_features + j] += b_data[j]; } }
                    }
                    return res;
                }

                let wp = weight_packed.get_slice::<u32>();
                let s = scales.get_slice::<f16>();
                let mut out_f32 = bit_serial_matmul_f32(x, wp, s, m, self.out_features, self.in_features);
                if let Some(bias_data) = bias.as_ref().map(|t| t.get_slice::<f16>()) {
                    for i in 0..m { for j in 0..self.out_features { out_f32[i * self.out_features + j] += bias_data[j].to_f32(); } }
                }
                out_f32.into_iter().map(f16::from_f32).collect()
            }
        }
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        match &mut self.variant {
            LinearVariant::Standard { .. } => {}, // Standard layers stay on CPU for now to save VRAM
            LinearVariant::BitSerial { weight_packed, scales, .. } => {
                #[cfg(feature = "cuda")]
                {
                    weight_packed.move_to_gpu(device_id);
                    // Scales are small, keep on CPU and convert to f32 on the fly or move if needed
                }
            }
        }
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor,
    pub post_attention_layernorm: NativeTensor,
    pub q_norm: Option<NativeTensor>, 
    pub k_norm: Option<NativeTensor>, 
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
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize, layer_idx: usize) -> Vec<f16> {
        let hidden_size = config.hidden_size;
        let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;

        let residual = x.to_vec();
        let x_norm = native_rms_norm_f16(x, self.input_layernorm.get_slice::<f16>(), config.rms_norm_eps as f32, hidden_size);
        
        let mut q = self.q_proj.forward(&x_norm);
        let mut k = self.k_proj.forward(&x_norm);
        let v = self.v_proj.forward(&x_norm);

        if let Some(ref qw) = self.q_norm { q = native_rms_norm_f16(&q, qw.get_slice::<f16>(), config.rms_norm_eps as f32, head_dim); }
        if let Some(ref kw) = self.k_norm { k = native_rms_norm_f16(&k, kw.get_slice::<f16>(), config.rms_norm_eps as f32, head_dim); }

        native_apply_rope_f16_with_offset(&mut q, &mut k, q_len, seqlen_offset, n_heads, head_dim, config.rope_theta);

        let mut cache_guard = self.kv_cache.lock().unwrap();
        let (k_full, v_full) = if let Some((prev_k, prev_v)) = cache_guard.take() {
            let mut new_k = prev_k; let mut new_v = prev_v;
            new_k.extend_from_slice(&k); new_v.extend_from_slice(&v);
            (new_k, new_v)
        } else { (k, v) };
        let total_seq_len = k_full.len() / (n_kv_heads * head_dim);
        *cache_guard = Some((k_full.clone(), v_full.clone()));
        drop(cache_guard);

        let attn_out_vec = native_vision_attn_f16(&q, &k_full, &v_full, hidden_size, n_heads, q_len, total_seq_len);
        let mut x_attn = self.o_proj.forward(&attn_out_vec);
        for i in 0..x_attn.len() { x_attn[i] += residual[i]; }

        let residual_mlp = x_attn.clone();
        let x_norm_mlp = native_rms_norm_f16(&x_attn, self.post_attention_layernorm.get_slice::<f16>(), config.rms_norm_eps as f32, hidden_size);
        let mut gate = self.gate_proj.forward(&x_norm_mlp);
        let up = self.up_proj.forward(&x_norm_mlp);
        native_silu_f16(&mut gate);
        for i in 0..gate.len() { gate[i] *= up[i]; }
        let mut x_mlp = self.down_proj.forward(&gate);
        for i in 0..x_mlp.len() { x_mlp[i] += residual_mlp[i]; }
        x_mlp
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.q_proj.move_to_gpu(device_id);
        self.k_proj.move_to_gpu(device_id);
        self.v_proj.move_to_gpu(device_id);
        self.o_proj.move_to_gpu(device_id);
        self.gate_proj.move_to_gpu(device_id);
        self.up_proj.move_to_gpu(device_id);
        self.down_proj.move_to_gpu(device_id);
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig,
    pub embed_tokens: NativeLinear, 
    pub layers: Vec<NativeLayer>,
    pub norm: NativeTensor,
}

impl NativeQwen3TextModel {
    pub fn forward(&self, input_ids: &[u32], seqlen_offset: usize) -> Vec<f16> {
        let hidden_size = self.config.hidden_size;
        let x = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(input_ids, weight.get_slice::<f16>(), hidden_size),
            _ => vec![f16::ZERO; input_ids.len() * hidden_size], // Embeddings always Standard
        };
        let mut current_x = x;
        for (i, layer) in self.layers.iter().enumerate() { current_x = layer.forward(&current_x, &self.config, seqlen_offset, i); }
        native_rms_norm_f16(&current_x, self.norm.get_slice::<f16>(), self.config.rms_norm_eps as f32, hidden_size)
    }
}

pub struct NativeVisionModel {
    pub patch_embed: NativeLinear,
    pub pos_embed: NativeTensor,
    pub blocks: Vec<NativeLayer>, 
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig,
    pub text_model: NativeQwen3TextModel,
    pub vision_model: Option<NativeVisionModel>, 
    pub lm_head: NativeLinear,
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, main_mmap: Arc<Mmap>, vision_mmap: Arc<Mmap>, baking_only: bool) -> Result<Self> {
        let st_main = SafeTensors::deserialize(&main_mmap)?;
        let t_cfg = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        
        let get_main_t = |name: &str| -> Result<NativeTensor> {
            let view = st_main.tensor(name)?;
            let offset = unsafe { view.data().as_ptr().offset_from(main_mmap.as_ptr()) } as usize;
            Ok(NativeTensor::from_mmap(main_mmap.clone(), offset, view.shape().to_vec(), NativeDType::F16))
        };

        let get_linear = |st: &SafeTensors, mmap: &Arc<Mmap>, base_name: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {
            let packed_name = format!("{}.packed", base_name);
            let scales_name = format!("{}.scales", base_name);
            let variant = if st.tensor(&packed_name).is_ok() {
                let view_p = st.tensor(&packed_name)?; let view_s = st.tensor(&scales_name)?;
                let off_p = unsafe { view_p.data().as_ptr().offset_from(mmap.as_ptr()) } as usize;
                let off_s = unsafe { view_s.data().as_ptr().offset_from(mmap.as_ptr()) } as usize;
                LinearVariant::BitSerial {
                    weight_packed: NativeTensor::from_mmap(mmap.clone(), off_p, view_p.shape().to_vec(), NativeDType::U32),
                    scales: NativeTensor::from_mmap(mmap.clone(), off_s, view_s.shape().to_vec(), NativeDType::F16),
                    bias: None,
                }
            } else {
                let view = st.tensor(base_name)?;
                let off = unsafe { view.data().as_ptr().offset_from(mmap.as_ptr()) } as usize;
                LinearVariant::Standard { weight: NativeTensor::from_mmap(mmap.clone(), off, view.shape().to_vec(), NativeDType::F16), bias: None }
            };
            Ok(NativeLinear { in_features: in_f, out_features: out_f, variant, device_id: -1 })
        };

        // 1. Load Embeddings (CRITICAL: Handle TensorNotFound)
        let embed_tokens = get_linear(&st_main, &main_mmap, "model.embed_tokens.weight", 151936, t_cfg.hidden_size)
            .or_else(|_| get_linear(&st_main, &main_mmap, "model.layers.0.input_layernorm.weight", t_cfg.hidden_size, t_cfg.hidden_size))?; // Fallback placeholder if missing

        // 2. Load Layers
        let mut layers = Vec::new();
        let layers_to_load = if baking_only { 1 } else { t_cfg.num_hidden_layers };
        for i in 0..layers_to_load {
            let p = format!("model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_main_t(&format!("{}.input_layernorm.weight", p))?,
                post_attention_layernorm: get_main_t(&format!("{}.post_attention_layernorm.weight", p))?,
                q_norm: get_main_t(&format!("{}.self_attn.q_norm.weight", p)).ok(),
                k_norm: get_main_t(&format!("{}.self_attn.k_norm.weight", p)).ok(),
                q_proj: get_linear(&st_main, &main_mmap, &format!("{}.self_attn.q_proj.weight", p), t_cfg.hidden_size, t_cfg.hidden_size)?,
                k_proj: get_linear(&st_main, &main_mmap, &format!("{}.self_attn.k_proj.weight", p), t_cfg.hidden_size, t_cfg.hidden_size)?,
                v_proj: get_linear(&st_main, &main_mmap, &format!("{}.self_attn.v_proj.weight", p), t_cfg.hidden_size, t_cfg.hidden_size)?,
                o_proj: get_linear(&st_main, &main_mmap, &format!("{}.self_attn.o_proj.weight", p), t_cfg.hidden_size, t_cfg.hidden_size)?,
                gate_proj: get_linear(&st_main, &main_mmap, &format!("{}.mlp.gate_proj.weight", p), t_cfg.hidden_size, t_cfg.intermediate_size)?,
                up_proj: get_linear(&st_main, &main_mmap, &format!("{}.mlp.up_proj.weight", p), t_cfg.hidden_size, t_cfg.intermediate_size)?,
                down_proj: get_linear(&st_main, &main_mmap, &format!("{}.mlp.down_proj.weight", p), t_cfg.intermediate_size, t_cfg.hidden_size)?,
                kv_cache: std::sync::Mutex::new(None),
            });
        }

        let norm = get_main_t("model.norm.weight")?;
        let lm_head = get_linear(&st_main, &main_mmap, "lm_head.weight", t_cfg.hidden_size, 151936).unwrap_or_else(|_| {
            NativeLinear { in_features: t_cfg.hidden_size, out_features: 1, variant: LinearVariant::Standard { weight: norm.clone(), bias: None }, device_id: -1 }
        });

        Ok(Self {
            config: config.clone(),
            text_model: NativeQwen3TextModel { config: t_cfg.clone(), embed_tokens, layers, norm },
            vision_model: None, 
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &[u32], _pixel_values: Option<&[f16]>, _grid_thw: Option<&[u32; 3]>, seqlen_offset: usize) -> Vec<f16> {
        let x = self.text_model.forward(input_ids, seqlen_offset);
        self.lm_head.forward(&x)
    }

    pub fn clear_kv_cache(&self) { self.text_model.clear_kv_cache(); }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        for layer in &mut self.text_model.layers { layer.move_to_gpu(device_id); }
    }
}
