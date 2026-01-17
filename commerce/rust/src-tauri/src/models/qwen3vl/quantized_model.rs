use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use nvml_wrapper::Nvml;

use crate::{
    models::{
        common::eager_attention_forward,
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{
        Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        mask_index_add, masked_scatter_dim0,
        prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
    },
};

// Local RmsNorm implementation exposing weight and device
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let hidden_states = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let hidden_states = hidden_states.to_dtype(dtype)?;
        hidden_states.broadcast_mul(&self.weight)
    }
}

// Wrapper for QMatMul to act like Linear
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    device: Device, // Track device explicitly
}

impl QLinear {
    pub fn new(inner: QMatMul, bias: Option<Tensor>, device: Device) -> Self {
        Self { inner, bias, device }
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Auto-move input to this layer's device
        let xs = if !xs.device().same_device(&self.device) {
            xs.to_device(&self.device)?
        } else {
            xs.clone() // QMatMul likely needs owned or cow, but here we clone for safety
        };

        // QMatMul typically expects F32 input
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let out = self.inner.forward(&xs_f32)?;
        
        // Cast back to target dtype (from bias if available, else xs dtype)
        let target_dtype = self.bias.as_ref().map(|b| b.dtype()).unwrap_or(xs.dtype());
        let out = out.to_dtype(target_dtype)?;

        if let Some(bias) = &self.bias {
            Ok(out.broadcast_add(bias)?)
        } else {
            Ok(out)
        }
    }
}

pub struct QuantizedQwen3VLTextAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QuantizedQwen3VLTextAttention {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        is_gguf_naming: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let _hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);

        let (q, k, v, o, q_n, k_n) = if is_gguf_naming {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        };

        let q_proj = get_qlinear(ct, reader, &format!("{base_name}.{q}"), device, dtype)?;
        let k_proj = get_qlinear(ct, reader, &format!("{base_name}.{k}"), device, dtype)?;
        let v_proj = get_qlinear(ct, reader, &format!("{base_name}.{v}"), device, dtype)?;
        let o_proj = get_qlinear(ct, reader, &format!("{base_name}.{o}"), device, dtype)?;

        let q_norm = get_rms_norm(ct, reader, &format!("{base_name}.{q_n}"), config.rms_norm_eps, device, dtype)?;
        let k_norm = get_rms_norm(ct, reader, &format!("{base_name}.{k_n}"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        // Norms are also on specific device
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                (k, v)
            }
        };

        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;

        // Update KV cache without cloning if possible. 
        // Since key_states and value_states are owned here, we can just move them.
        self.kv_cache = Some((key_states, value_states));

        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct QuantizedQwen3VLTextDecoderLayer {
    self_attn: QuantizedQwen3VLTextAttention,
    mlp_gate: QLinear,
    mlp_up: QLinear,
    mlp_down: QLinear,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3VLTextDecoderLayer {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype)?;
        
        let mlp_gate = get_qlinear(ct, reader, &format!("{base_name}.{gate}"), device, dtype)?;
        let mlp_up = get_qlinear(ct, reader, &format!("{base_name}.{up}"), device, dtype)?;
        let mlp_down = get_qlinear(ct, reader, &format!("{base_name}.{down}"), device, dtype)?;

        let input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{in_ln}"), config.rms_norm_eps, device, dtype)?;
        let post_attention_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.{post_ln}"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            self_attn,
            mlp_gate,
            mlp_up,
            mlp_down,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Detect device AND dtype of this layer
        let dev = self.input_layernorm.weight().device();
        let target_dtype = self.input_layernorm.weight().dtype();
        
        // 2. Ensure inputs are on this device and dtype
        //    (Clone via Cow logic or explicit clone if needed, here we use explicit clones/conversions for safety)
        let mut xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        if xs.dtype() != target_dtype { xs = xs.to_dtype(target_dtype)?; }

        let mut cos = if !cos.device().same_device(dev) { cos.to_device(dev)? } else { cos.clone() };
        if cos.dtype() != target_dtype { cos = cos.to_dtype(target_dtype)?; }

        let mut sin = if !sin.device().same_device(dev) { sin.to_device(dev)? } else { sin.clone() };
        if sin.dtype() != target_dtype { sin = sin.to_dtype(target_dtype)?; }

        let attention_mask = if let Some(mask) = attention_mask {
             // Mask is usually F32 or specific, but we ensure device match
             Some(if !mask.device().same_device(dev) { mask.to_device(dev)? } else { mask.clone() })
        } else {
             None
        };

        let residual = xs.clone();
        let xs = self.input_layernorm.forward(&xs)?;
        let xs = self.self_attn.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        let xs = residual.add(&xs)?;
        
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = {
            let gate = self.mlp_gate.forward(&xs)?;
            let up = self.mlp_up.forward(&xs)?;
            let gate = candle_nn::ops::silu(&gate)?;
            let hidden = (gate * up)?;
            self.mlp_down.forward(&hidden)?
        };
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct QuantizedQwen3VLTextModel {
    pub embed_tokens: Embedding, 
    layers: Vec<QuantizedQwen3VLTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl QuantizedQwen3VLTextModel {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3VLTextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Embeddings: Try standard and GGUF names
        let token_emb_name = format!("{base_name}.embed_tokens.weight");
        let alt_token_emb = "token_embd.weight";
        
        let embed_tokens = if let Ok(tensor) = ct.tensor(reader, &token_emb_name, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             Embedding::new(tensor, config.hidden_size)
        } else if let Ok(tensor) = ct.tensor(reader, alt_token_emb, device) {
             let tensor = tensor.dequantize(device)?.to_dtype(dtype)?;
             Embedding::new(tensor, config.hidden_size)
        } else {
             return Err(anyhow!("Failed to load embedding. Tried {}, {}", token_emb_name, alt_token_emb));
        };

        // Initialize NVML for dynamic VRAM check
        let nvml = Nvml::init().ok();
        let mut current_device = device.clone();
        
        // --- Organic Budget-based Optimization ---
        // Instead of a hard threshold, we simulate filling the VRAM.
        // Qwen2-VL-2B Q4 Estimates:
        // - Weight per layer: ~50 MB
        // - Runtime Buffer (KV + Act) per layer: ~30 MB
        // - Total Cost per Layer: ~80 MB
        // - Safety Floor (LM Head + OS Overhead): 600 MB
        
        let cost_per_layer: u64 = 80_000_000; 
        let safety_floor: u64 = 600_000_000;
        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;

        // Get initial reading
        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(0) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         println!("[VRAM-BUDGET] Initial Free: {:.2} GB. Target Floor: {:.2} GB. Cost/Layer: {:.2} MB", 
                            mem.free as f64/1e9, safety_floor as f64/1e9, cost_per_layer as f64/1e6);
                     }
                 }
             }
        }

        let mut layers = vec![];
        for layer_idx in 0..config.num_hidden_layers {
            // Organic Check
            if current_device.is_cuda() && is_vram_checked {
                 // Check if we have budget for this layer + safety floor
                 if simulated_free_vram > (cost_per_layer + safety_floor) {
                     // Allocating to GPU
                     simulated_free_vram = simulated_free_vram.saturating_sub(cost_per_layer);
                 } else {
                     // Budget exhausted
                     if current_device.is_cuda() { // Only print once when switching
                        println!("[OFFLOAD] Budget Exhausted (Est. Free: {:.2} MB). Switching Layer {} and subsequent to CPU.", 
                            simulated_free_vram as f64/1e6, layer_idx);
                     }
                     current_device = Device::Cpu;
                 }
            }

            // Determine dtype: If CPU, force F32. Else use requested dtype (BF16).
            let layer_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };

            // Determine prefix: "model.layers.N" or "blk.N"
            let standard = format!("{base_name}.layers.{layer_idx}");
            let gguf_blk = format!("blk.{layer_idx}");
            
            let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) {
                gguf_blk
            } else {
                standard
            };

            let layer = QuantizedQwen3VLTextDecoderLayer::new(
                config,
                ct,
                reader,
                &prefix,
                &current_device,
                layer_dtype,
            )?;
            layers.push(layer)
        }
        
        // Norm - load on same device as last layer, with matching dtype
        let norm_name = format!("{base_name}.norm");
        let alt_norm = "output_norm";
        let norm_prefix = if ct.tensor_infos.contains_key(&format!("{}.weight", alt_norm)) {
            alt_norm
        } else {
            &norm_name
        };
        
        let norm_dtype = if current_device.is_cpu() { DType::F32 } else { dtype };
        let norm = get_rms_norm(ct, reader, norm_prefix, config.rms_norm_eps, &current_device, norm_dtype)?;
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.mrope_section.clone();
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        
        // RoPE is computed here. Usually on inputs_embeds.device() (GPU).
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    inputs_embeds.device(),
                )?)
            }
        };

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Layer handles device transfer internally
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    // Deepstack logic might need device check too
                    let mask = visual_pos_masks.unwrap();
                    let embed = &deepstack_embeds[layer_idx];
                    
                    // Move to xs device
                    let mask = if !mask.device().same_device(xs.device()) { mask.to_device(xs.device())? } else { mask.clone() };
                    let embed = if !embed.device().same_device(xs.device()) { embed.to_device(xs.device())? } else { embed.clone() };

                    xs = mask_index_add(
                        &xs.squeeze(0)?,
                        &mask.squeeze(0)?,
                        &embed,
                    )?
                    .unsqueeze(0)?;
                }
            }
        }
        
        // Final Norm - ensure xs is on norm device
        let norm_dev = self.norm.weight().device();
        if !xs.device().same_device(norm_dev) {
            xs = xs.to_device(norm_dev)?;
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct QuantizedQwen3VLModel {
    config: Qwen3VLConfig,
    visual: Qwen3VLVisionModel, 
    language_model: QuantizedQwen3VLTextModel,
    lm_head: QLinear,
    rope_deltas: Option<Tensor>,
}

impl QuantizedQwen3VLModel {
    pub fn new<R: std::io::Seek + std::io::Read, R2: std::io::Seek + std::io::Read>(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        reader_main: &mut R,
        ct_vision: &gguf_file::Content,
        reader_vision: &mut R2,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Load Vision from mmproj file
        let vb_visual = from_gguf_content(config, ct_vision, reader_vision, device, dtype)?;
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_visual.pp("visual"))?;
        
        // Load Language Model from main file
        let language_model = QuantizedQwen3VLTextModel::new(&config.text_config, ct_main, reader_main, "model", device, dtype)?;
        
        // LM Head - Check VRAM again to decide device
        let nvml = Nvml::init().ok();
        let mut head_device = device.clone();
        if head_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(0) {
                     if let Ok(mem) = dev.memory_info() {
                         if mem.free < 200_000_000 { // 200MB threshold (Safety Floor preserved this)
                             println!("[OFFLOAD] VRAM Free: {:.2} GB. Switching Head to CPU.", mem.free as f64/1e9);
                             head_device = Device::Cpu;
                         }
                     }
                 }
             }
        }

        // Determine head dtype: F32 if CPU, else inherit
        let head_dtype = if head_device.is_cpu() { DType::F32 } else { dtype };

        // Try 'lm_head', then 'output', then fallback to 'token_embd' (tied weights)
        let lm_head = if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", &head_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", &head_device, head_dtype) {
            l
        } else {
            // Fallback for tied embeddings or GGUF specific naming 'token_embd'
            get_qlinear(ct_main, reader_main, "token_embd", &head_device, head_dtype)?
        };

        Ok(Self {
            config: config.clone(),
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
        })
    }
    
    fn get_vision_features(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(pixel_values, image_grid_thw)?;
        let split_sizes: Vec<usize> = prod_tensor_last_dim(image_grid_thw)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize / self.config.vision_config.spatial_merge_size.pow(2))
            .collect();
        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
        Ok((image_embeds, deepstack_image_embeds))
    }

    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token = if is_image {
            Tensor::new(vec![self.config.image_token_id as u32], input_ids.device())?
        } else {
            Tensor::new(vec![self.config.video_token_id as u32], input_ids.device())?
        };
        let special_mask = input_ids
            .broadcast_eq(&special_token)?
            .to_dtype(candle_core::DType::U32)?;
        Ok(special_mask)
    }
    
    // Placeholder implementation for get_rope_index to satisfy compilation.
    // In production this should replicate the full logic from Qwen3VLModel.
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        _image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let position_ids = Tensor::arange(0u32, input_ids.dim(1)? as u32, input_ids.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?;
        let mrope = Tensor::zeros((input_ids.dim(0)?, 1), input_ids.dtype(), input_ids.device())?;
        Ok((position_ids, mrope))
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        _pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                 let (image_embeds, _) = self.get_vision_features(pixel_values, image_grid_thw)?;
                 let image_embeds = Tensor::cat(&image_embeds, 0)?;
                 let vision_mask = self.get_placeholder_mask(input_ids, true)?;
                 inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
            }
        }

        // Position IDs logic (simplified for quantized model)
        // Note: Full logic requires image/video grid awareness for mrope.
        let (position_ids, _) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
        let position_ids = if let Some(cache_pos) = cache_position {
             // Basic relative position for cache (needs improvement for mrope)
             let start = cache_pos.i(0)?.to_scalar::<u32>()?;
             Tensor::arange(start, start + input_ids.dim(1)? as u32, input_ids.device())?
                .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
        } else {
             position_ids
        };

        let outputs = self.language_model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids),
            None, // visual_pos_masks not fully implemented in quantized wrapper yet
            None, // deepstack not fully implemented
        )?;
        
        // Output from language model might be on CPU if last layer offloaded.
        // LM Head might be on GPU or CPU.
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        
        // Ensure hidden_state is on lm_head device
        let hidden_state = if !hidden_state.device().same_device(self.lm_head.device()) {
            hidden_state.to_device(self.lm_head.device())?
        } else {
            hidden_state
        };

        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}

// Helper functions

fn get_qlinear<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, device: &Device, dtype: DType) -> Result<QLinear> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device).map_err(|e| anyhow!("Failed to load {name}.weight: {e}"))?;
    let weight = QMatMul::from_qtensor(weight)?;
    let bias = if let Ok(t) = ct.tensor(reader, &format!("{name}.bias"), device) {
        Some(t.dequantize(device)?.to_dtype(dtype)?)
    } else {
        None
    };
    Ok(QLinear::new(weight, bias, device.clone()))
}

fn get_rms_norm<R: std::io::Seek + std::io::Read>(ct: &gguf_file::Content, reader: &mut R, name: &str, eps: f64, device: &Device, dtype: DType) -> Result<RmsNorm> {
    let weight = ct.tensor(reader, &format!("{name}.weight"), device)?;
    let weight = weight.dequantize(device)?.to_dtype(dtype)?;
    Ok(RmsNorm::new(weight, eps))
}

fn from_gguf_content<R: std::io::Seek + std::io::Read>(config: &Qwen3VLConfig, ct: &gguf_file::Content, reader: &mut R, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
    use std::collections::{HashMap, BTreeMap};
    let mut data = HashMap::new();
    let mut split_tensors: BTreeMap<String, Vec<(usize, Tensor)>> = BTreeMap::new();
    
    // Print all tensor names for debugging
    // for (name, _) in ct.tensor_infos.iter() {
    //    println!("[DEBUG-TENSOR] {}", name);
    // }

    for (name, _) in ct.tensor_infos.iter() {
        let mut new_name = name.clone();
        
        // 1. Prefix Handling: v. -> visual.
        if let Some(rest) = name.strip_prefix("v.") {
             if let Some(blk_rest) = rest.strip_prefix("blk.") {
                 // blk.{i}.{layer}...
                 let parts: Vec<&str> = blk_rest.splitn(2, '.').collect();
                 if parts.len() == 2 {
                     let idx = parts[0];
                     let layer = parts[1];
                     let mapped_layer = match layer {
                         s if s.starts_with("ln1") => s.replace("ln1", "norm1"),
                         s if s.starts_with("ln2") => s.replace("ln2", "norm2"),
                         s if s.starts_with("attn_qkv") => s.replace("attn_qkv", "attn.qkv"),
                         s if s.starts_with("attn_out") => s.replace("attn_out", "attn.proj"),
                         s if s.starts_with("ffn_up") => s.replace("ffn_up", "mlp.linear_fc1"),
                         s if s.starts_with("ffn_down") => s.replace("ffn_down", "mlp.linear_fc2"),
                         _ => layer.to_string(),
                     };
                     new_name = format!("visual.blocks.{}.{}", idx, mapped_layer);
                 }
             } else if rest.starts_with("patch_embd") {
                 new_name = rest.replace("patch_embd", "visual.patch_embed.proj");
             } else if rest.starts_with("position_embd") {
                 new_name = rest.replace("position_embd", "visual.pos_embed");
             } else if rest.starts_with("post_ln") {
                 new_name = rest.replace("post_ln", "visual.merger.norm");
             } else if rest.starts_with("deepstack.") {
                 // v.deepstack.5.fc1.weight
                 let parts: Vec<&str> = rest.split('.').collect();
                 // parts[0]="deepstack", parts[1]="5", parts[2..]="fc1.weight"
                 if parts.len() >= 2 {
                     if let Ok(layer_idx) = parts[1].parse::<usize>() {
                         if let Some(pos) = config.vision_config.deepstack_visual_indexes.iter().position(|&x| x == layer_idx) {
                             let suffix = parts[2..].join(".");
                             new_name = format!("visual.deepstack_merger_list.{}.{}", pos, suffix)
                                            .replace("fc1", "linear_fc1")
                                            .replace("fc2", "linear_fc2");
                         } else {
                             new_name = rest.replace("deepstack", "visual.deepstack_merger_list")
                                            .replace("fc1", "linear_fc1")
                                            .replace("fc2", "linear_fc2");
                         }
                     } else {
                         // fallback
                         new_name = rest.replace("deepstack", "visual.deepstack_merger_list")
                                        .replace("fc1", "linear_fc1")
                                        .replace("fc2", "linear_fc2");
                     }
                 }
             } else {
                 new_name = format!("visual.{}", rest);
             }
        } else if let Some(rest) = name.strip_prefix("mm.") {
             if rest.starts_with("0") {
                 new_name = rest.replace("0", "visual.merger.linear_fc1");
             } else if rest.starts_with("2") {
                 new_name = rest.replace("2", "visual.merger.linear_fc2");
             }
        } else if name.starts_with("model.visual") {
             new_name = name.strip_prefix("model.").unwrap().to_string();
        }

        // 2. Handle Split Tensors (.0, .1 suffix)
        // Check if ends with .number
        let mut is_split = false;
        let mut split_idx = 0;
        let mut base_split_name = new_name.clone();

        if let Some(last_dot) = new_name.rfind('.') {
            if let Ok(idx) = new_name[last_dot+1..].parse::<usize>() {
                 if name.ends_with(&format!(".{}", idx)) {
                     base_split_name = new_name[..last_dot].to_string();
                     split_idx = idx;
                     is_split = true;
                 }
            }
        }

        let t = ct.tensor(reader, name, device)?;
        let t = t.dequantize(device)?.to_dtype(dtype)?;

        if is_split {
            split_tensors.entry(base_split_name).or_default().push((split_idx, t));
        } else {
            data.insert(new_name, t);
        }
    }

    // Merge split tensors
    for (name, mut parts) in split_tensors {
        parts.sort_by_key(|(i, _)| *i);
        let tensors: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect();
        if let Ok(merged) = Tensor::cat(&tensors, 0) {
            data.insert(name, merged);
        } else {
            println!("Failed to merge split tensor: {}", name);
        }
    }

    // Fix shape mismatch for patch_embed.proj.weight
    if let Some(weight) = data.get("visual.patch_embed.proj.weight") {
        if weight.rank() == 4 {
            if let Ok(reshaped) = weight.unsqueeze(2)?.repeat((1, 1, 2, 1, 1)) {
                data.insert("visual.patch_embed.proj.weight".to_string(), reshaped);
                println!("[FIX] Reshaped visual.patch_embed.proj.weight to 5D");
            }
        }
    }
    
    Ok(VarBuilder::from_tensors(data, dtype, device))
}
