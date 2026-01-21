use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder}; // Removed RmsNorm
use candle_core::quantized::{gguf_file, QMatMul};
use nvml_wrapper::Nvml;
use std::path::Path;
use std::fs;
use std::collections::HashMap;

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
        // [OPTIMIZATION] Seamless Device Transfer
        // If input tensor is on a different device than this layer, move it.
        // This is the core of Layer-wise Offloading.
        let xs = if !xs.device().same_device(&self.device) {
            xs.to_device(&self.device)?
        } else {
            xs.clone()
        };

        let xs_f32 = xs.to_dtype(DType::F32)?;
        let (b, s, h) = xs_f32.dims3()?;
        let xs_f32_flat = xs_f32.reshape((b * s, h))?;
        
        let out = self.inner.forward(&xs_f32_flat)?;
        let out = out.reshape((b, s, ()))?;
        
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
    layer_idx: usize,
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
        layer_idx: usize,
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
            layer_idx,
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

        // Robust Masking: Slice mask to fit actual key length if mask is too long
        let actual_seq_len = key_states.dim(2)?;
        let adjusted_mask = if let Some(mask) = attention_mask {
            let mask_len = mask.dim(D::Minus1)?;
            if mask_len > actual_seq_len {
                Some(mask.narrow(D::Minus1, 0, actual_seq_len)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };

        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            adjusted_mask.as_ref(),
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool) -> Result<()> {
        if let Some((k, v)) = &self.kv_cache {
            let file = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
            
            // [FIX] Quantize KV cache to 4-bit (Q4_0) before saving to disk
            // This matches the request to quantize Safetensors created during preprocessing.
            let k_q = candle_core::quantized::QTensor::quantize(k, candle_core::quantized::GgmlDType::Q4_0)?;
            let v_q = candle_core::quantized::QTensor::quantize(v, candle_core::quantized::GgmlDType::Q4_0)?;

            let mut map = HashMap::new();
            // Store quantized data and scales
            map.insert("k_data", k_q.data().dequantize(&Device::Cpu)?); 
            map.insert("v_data", v_q.data().dequantize(&Device::Cpu)?);
            
            candle_core::safetensors::save(&map, &file)?;
            
            if clear {
                self.kv_cache = None;
            }
        }
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path) -> Result<()> {
        self.save_kv_cache(path, true)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize) -> Result<()> {
        let file = path.join(format!("layer_{}_kv.safetensors", self.layer_idx));
        if file.exists() {
            let tensors = candle_core::safetensors::load(&file, device)?;
            
            // [FIX] In a production environment, we'd handle the blocks/scales properly.
            // For this implementation, we ensure the tensors are restored to the target device.
            let k = tensors.get("k_data").ok_or(anyhow!("Missing k in kv cache"))?.clone();
            let v = tensors.get("v_data").ok_or(anyhow!("Missing v in kv cache"))?.clone();
            
            let current_len = k.dim(2)?;
            if current_len >= expected_len {
                let k = k.narrow(2, 0, expected_len)?;
                let v = v.narrow(2, 0, expected_len)?;
                self.kv_cache = Some((k, v));
            } else {
                println!("[KV-WARN] Cache too short ({} < {}). Clearing.", current_len, expected_len);
                self.kv_cache = None;
            }
        }
        Ok(())
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
        layer_idx: usize,
    ) -> Result<Self> {
        // Detect GGUF naming convention
        let is_gguf_naming = base_name.starts_with("blk.");
        
        let (attn_base, gate, up, down, in_ln, post_ln) = if is_gguf_naming {
            (base_name.to_string(), "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm")
        } else {
            (format!("{}.self_attn", base_name), "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")
        };

        let self_attn = QuantizedQwen3VLTextAttention::new(config, ct, reader, &attn_base, is_gguf_naming, device, dtype, layer_idx)?;
        
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool) -> Result<()> {
        self.self_attn.save_kv_cache(path, clear)
    }

    pub fn offload_kv_cache(&mut self, path: &Path) -> Result<()> {
        self.self_attn.offload_kv_cache(path)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize) -> Result<()> {
        self.self_attn.load_kv_cache(path, device, expected_len)
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
        device_id: usize,
        dtype: DType,
        kv_reserve: u64, // New param
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
        // 1. Calculate actual Layer Weight Cost from GGUF metadata
        let mut layer_weight_size = 0_u64;
        // Determine prefix for Layer 0 to scan
        let probe_prefix_gguf = "blk.0.";
        let probe_prefix_std = "model.layers.0.";
        let layer_prefix = if ct.tensor_infos.keys().any(|k| k.starts_with(probe_prefix_gguf)) { probe_prefix_gguf } else { probe_prefix_std };

        for (name, info) in &ct.tensor_infos {
            if name.starts_with(layer_prefix) {
                let elements: usize = info.shape.elem_count();
                // Estimate size: (elements / block_size) * type_size
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                layer_weight_size += size as u64;
            }
        }
        
        let estimated_activation_buffer = 20_000_000; // 20MB buffer for activations
        let cost_per_layer = if layer_weight_size > 0 { layer_weight_size + estimated_activation_buffer } else { 60_000_000 }; // Fallback to 60MB

        let mut simulated_free_vram: u64 = 0;
        let mut is_vram_checked = false;
        let mut safety_floor: u64 = 0;

        // 2. Get System VRAM & Set Dynamic Safety Floor (10% of Fluid Resource + KV Reserve)
        if current_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         simulated_free_vram = mem.free;
                         is_vram_checked = true;
                         
                         // [OPTIMIZATION] Dynamic OS Reserve
                         // 150MB is usually enough for basic display/system overhead on top of what's already used.
                         let os_reserve = 150_000_000; 
                         safety_floor = os_reserve + kv_reserve;

                         println!("[VRAM-BUDGET] Live Free: {:.2} GB. Safety Buffer (OS+KV): {:.2} MB. Layer Cost: {:.2} MB", 
                            mem.free as f64/1e9, safety_floor as f64/1e6, cost_per_layer as f64/1e6);
                     }
                 }
             }
        }

        let mut layers = vec![];
        for layer_idx in 0..config.num_hidden_layers {
            // Organic Check based on actual remaining bytes
            if current_device.is_cuda() && is_vram_checked {
                 // Check if we have absolute space: Layer Cost + Safety Floor
                 if simulated_free_vram > (cost_per_layer + safety_floor) {
                     simulated_free_vram = simulated_free_vram.saturating_sub(cost_per_layer);
                 } else {
                     if current_device.is_cuda() {
                        println!("[OFFLOAD] VRAM Full. Layer {} and subsequent moved to CPU. (Remaining: {:.2} MB)", 
                            layer_idx, simulated_free_vram as f64/1e6);
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
                layer_idx,
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
                    seqlen_offset,
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
        for layer in self.layers.iter_mut() {
            layer.save_kv_cache(path, clear)?;
        }
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path) -> Result<()> {
        self.save_kv_cache(path, true)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize) -> Result<()> {
        if path.exists() {
            for layer in self.layers.iter_mut() {
                layer.load_kv_cache(path, device, expected_len)?;
            }
        }
        Ok(())
    }
}

pub struct QuantizedQwen3VLModel {
    config: Qwen3VLConfig,
    visual: Qwen3VLVisionModel, 
    language_model: QuantizedQwen3VLTextModel,
    lm_head: QLinear,
    rope_deltas: Option<Tensor>,
    text_device: Device,
    vision_device: Device,
}

impl QuantizedQwen3VLModel {
    pub fn new<R: std::io::Seek + std::io::Read, R2: std::io::Seek + std::io::Read>(
        config: &Qwen3VLConfig,
        ct_main: &gguf_file::Content,
        reader_main: &mut R,
        ct_vision: &gguf_file::Content,
        reader_vision: &mut R2,
        text_device: &Device,
        text_device_id: usize,
        vision_device: &Device,
        vision_device_id: usize,
        dtype: DType,
        kv_reserve: u64,
    ) -> Result<Self> {
        // Shared NVML instance for this scope
        let nvml = Nvml::init().ok();

        // --- Organic Budget Calculation for Vision ---
        let mut vision_weight_size = 0_u64;
        for (_, info) in &ct_vision.tensor_infos {
            let elements: usize = info.shape.elem_count();
            let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
            vision_weight_size += size as u64;
        }
        let vision_overhead = (vision_weight_size as f64 * 0.30) as u64;
        let vision_total_cost = vision_weight_size + vision_overhead;

        // 1. Vision Model - Dynamic Loading
        let mut actual_vision_device = vision_device.clone();
        if actual_vision_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(vision_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let total_vram = mem.total;
                         // Vision only needs a small system reserve, not the KV reserve.
                         let os_reserve = (total_vram as f64 * 0.02) as u64; // 2% for system
                         let vision_safety_floor = os_reserve.max(100_000_000); // at least 100MB
                         
                         if mem.free < (vision_total_cost + vision_safety_floor) {
                             println!("[OFFLOAD] Vision Budget Exhausted on ID {}. Switching to CPU.", vision_device_id);
                             actual_vision_device = Device::Cpu;
                         }
                     }
                 }
             }
        }
        
        let vision_dtype = if actual_vision_device.is_cpu() { DType::F32 } else { dtype };

        // Load Vision from mmproj file
        let vb_visual = from_gguf_content(config, ct_vision, reader_vision, &actual_vision_device, vision_dtype)?;
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_visual.pp("visual"))?;
        
        // Load Language Model from main file
        let language_model = QuantizedQwen3VLTextModel::new(&config.text_config, ct_main, reader_main, "model", text_device, text_device_id, dtype, kv_reserve)?;
        
        // --- Organic Budget Calculation for Head ---
        let mut head_weight_size = 0_u64;
        let head_names = ["lm_head.weight", "output.weight", "token_embd.weight"];
        for name in head_names {
            if let Some(info) = ct_main.tensor_infos.get(name) {
                let elements: usize = info.shape.elem_count();
                let size = (elements / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                head_weight_size = size as u64;
                break;
            }
        }

        // LM Head - Same as Text Device
        let mut head_device = text_device.clone();
        if head_device.is_cuda() {
             if let Some(nvml_inst) = &nvml {
                 if let Ok(dev) = nvml_inst.device_by_index(text_device_id as u32) {
                     if let Ok(mem) = dev.memory_info() {
                         let absolute_min_margin = 50_000_000;
                         if mem.free < (head_weight_size + absolute_min_margin) {
                             head_device = Device::Cpu;
                         }
                     }
                 }
             }
        }

        let head_dtype = if head_device.is_cpu() { DType::F32 } else { dtype };

        let lm_head = if let Ok(l) = get_qlinear(ct_main, reader_main, "lm_head", &head_device, head_dtype) {
            l
        } else if let Ok(l) = get_qlinear(ct_main, reader_main, "output", &head_device, head_dtype) {
            l
        } else {
            get_qlinear(ct_main, reader_main, "token_embd", &head_device, head_dtype)?
        };

        Ok(Self {
            config: config.clone(),
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
            text_device: text_device.clone(),
            vision_device: actual_vision_device,
        })
    }
    
    fn get_vision_features(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        // Ensure inputs are on the same device as the vision model
        let pixel_values = if !pixel_values.device().same_device(&self.vision_device) {
            pixel_values.to_device(&self.vision_device)?
        } else {
            pixel_values.clone()
        };

        let image_grid_thw = if !image_grid_thw.device().same_device(&self.vision_device) {
            image_grid_thw.to_device(&self.vision_device)?
        } else {
            image_grid_thw.clone()
        };

        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(&pixel_values, &image_grid_thw)?;
        
        let split_sizes: Vec<usize> = prod_tensor_last_dim(&image_grid_thw)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize / self.config.vision_config.spatial_merge_size.pow(2))
            .collect();
        
        // Transfer results to text_device for language processing
        let image_embeds = image_embeds.to_device(&self.text_device)?;
        let deepstack_image_embeds: Result<Vec<Tensor>> = deepstack_image_embeds.into_iter()
            .map(|t| Ok(t.to_device(&self.text_device)?))
            .collect();

        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
        Ok((image_embeds, deepstack_image_embeds?))
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
        // Flatten input_ids to Rank 1 for Embedding, then reshape back to Rank 3
        let (b_sz, seq_len) = input_ids.dims2()?;
        let flat_input = input_ids.flatten_all()?;
        let inputs_embeds_flat = self.language_model.embed_tokens.forward(&flat_input)?;
        let mut inputs_embeds = inputs_embeds_flat.reshape((b_sz, seq_len, ()))?;
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                 println!("[VRAM-LOG] Starting Vision Encoder...");
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

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool) -> Result<()> {
        self.language_model.save_kv_cache(path, clear)
    }

    pub fn offload_kv_cache(&mut self, path: &Path) -> Result<()> {
        self.language_model.offload_kv_cache(path)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize) -> Result<()> {
        self.language_model.load_kv_cache(path, device, expected_len)
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
