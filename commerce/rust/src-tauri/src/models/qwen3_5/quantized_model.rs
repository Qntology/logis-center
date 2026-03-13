use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_core::quantized::{gguf_file, QMatMul};
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

use crate::{
    models::{
        qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
        qwen3vl::quantized_model::{RmsNorm, QLinear, KVRegistry, KVBlock, KVLocation, RegistryEntry, MemorySlot, SLOT_MANAGER},
    },
    position_embed::rope::{
        Qwen3VLTextRotaryEmbedding, glm_asr_apply_rotary_pos_emb,
    },
    utils::tensor_utils::{
        masked_scatter_dim0, split_tensor,
    },
};

// Re-use logic from qwen3vl where possible
use crate::models::qwen3vl::quantized_model::{get_qlinear, get_rms_norm};

#[derive(Clone)]
pub struct QuantizedQwen3_5Attention {
    pub q_proj: QLinear,
    pub k_proj: QLinear,
    pub v_proj: QLinear,
    pub o_proj: QLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f64,
    pub kv_blocks: Vec<KVBlock>,
    pub registry: KVRegistry,
    pub layer_idx: usize,
}

impl QuantizedQwen3_5Attention {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3_5TextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        registry: KVRegistry,
    ) -> Result<Self> {
        let head_dim = config.head_dim;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        
        let q_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.q_proj"), device, dtype)?;
        let k_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.k_proj"), device, dtype)?;
        let v_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.v_proj"), device, dtype)?;
        let o_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.o_proj"), device, dtype)?;

        let q_norm = get_rms_norm(ct, reader, &format!("{base_name}.self_attn.q_norm"), config.rms_norm_eps, device, dtype)?;
        let k_norm = get_rms_norm(ct, reader, &format!("{base_name}.self_attn.k_norm"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads,
            head_dim,
            scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
        })
    }

    // Forward logic would be similar to Qwen3VL but adapted for Qwen3.5's Gated Attention if needed
}

#[derive(Clone)]
pub struct QuantizedQwen3_5GatedDeltaNet {
    pub in_proj_qkv: QLinear,
    pub in_proj_z: QLinear,
    pub in_proj_b: QLinear,
    pub in_proj_a: QLinear,
    pub out_proj: QLinear,
    // Note: DeltaNet specific state management and conv1d would go here
    // For now, focusing on weight quantization
}

#[derive(Clone)]
pub struct QuantizedQwen3_5DecoderLayer {
    pub layer_type: String,
    pub self_attn: Option<QuantizedQwen3_5Attention>,
    pub linear_attn: Option<QuantizedQwen3_5GatedDeltaNet>,
    pub mlp: crate::models::qwen3vl::quantized_model::QuantizedMLP, // Re-use MLPGate/Up/Down logic
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3_5DecoderLayer {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3_5TextConfig,
        ct: &gguf_file::Content,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        registry: KVRegistry,
    ) -> Result<Self> {
        let layer_type = config.layer_types[layer_idx].clone();
        
        let (self_attn, linear_attn) = if layer_type == "linear_attention" {
            // Placeholder for GatedDeltaNet quantization
            (None, None) 
        } else {
            (Some(QuantizedQwen3_5Attention::new(config, ct, reader, base_name, device, dtype, layer_idx, registry.clone())?), None)
        };

        // Re-use MLPGate/Up/Down from Qwen3VL implementation style
        let mlp_gate = get_qlinear(ct, reader, &format!("{base_name}.mlp.gate_proj"), device, dtype)?;
        let mlp_up = get_qlinear(ct, reader, &format!("{base_name}.mlp.up_proj"), device, dtype)?;
        let mlp_down = get_qlinear(ct, reader, &format!("{base_name}.mlp.down_proj"), device, dtype)?;
        
        let mlp = crate::models::qwen3vl::quantized_model::QuantizedMLP {
            gate_proj: mlp_gate,
            up_proj: mlp_up,
            down_proj: mlp_down,
        };

        let input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.input_layernorm"), config.rms_norm_eps, device, dtype)?;
        let post_attention_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.post_attention_layernorm"), config.rms_norm_eps, device, dtype)?;

        Ok(Self {
            layer_type,
            self_attn,
            linear_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

pub struct QuantizedQwen3_5Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<QuantizedQwen3_5DecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub registry: KVRegistry,
    pub device: Device,
    pub dtype: DType,
}

impl QuantizedQwen3_5Model {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Qwen3_5TextConfig,
        ct: Arc<gguf_file::Content>,
        reader: &mut R,
        base_name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let token_emb = ct.tensor(reader, &format!("{base_name}.embed_tokens.weight"), device)?;
        let embed_tokens = Embedding::new(token_emb.dequantize(device)?.to_dtype(dtype)?, config.hidden_size);
        
        let registry = KVRegistry::new();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_base = format!("{base_name}.layers.{i}");
            layers.push(QuantizedQwen3_5DecoderLayer::new(config, &ct, reader, &layer_base, device, dtype, i, registry.clone())?);
        }

        let norm = get_rms_norm(&ct, reader, &format!("{base_name}.norm"), config.rms_norm_eps, device, dtype)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta),
            registry,
            device: device.clone(),
            dtype,
        })
    }
}
