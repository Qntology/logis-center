use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, conv1d_depthwise, eager_attention_forward, get_conv1d, softplus},
        qwen35::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{apply_rotary_pos_emb, Qwen3VLTextRotaryEmbedding},
    utils::tensor_utils::{
        get_equal_mask, get_vision_next_indices, l2_normalize, masked_scatter_dim0, nonzero_index,
        prepare_causal_attention_mask, repeat_interleave, split_tensor, zero_index,
    },
};

pub struct Qwen3_5RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Qwen3_5RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { eps, weight })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(candle_core::DType::F32)?;
        let norm_ = x
            .powf(2.0)?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?;
        let norm = x.broadcast_div(&norm_)?;
        let norm = norm.broadcast_mul(&self.weight.to_dtype(candle_core::DType::F32)?)?.to_dtype(xs.dtype())?;
        Ok(norm)
    }
}

pub struct Qwen3_5RMSNormGated {
    norm: RmsNorm,
}

impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let norm = rms_norm(hidden_size, eps, vb)?;
        Ok(Self { norm })
    }

    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.norm.forward(xs)?;
        if let Some(gate) = gate {
            xs = xs.broadcast_mul(&gate.silu()?)?;
        }
        Ok(xs)
    }
}

macro_rules! transmute_tensors {
    ($($tensor:expr),*) => {
        ($(
            $tensor.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        )*)
    };
}

macro_rules! right_pad_zero_tensor {
    ($dim:expr, $pad_size:expr, $($tensor:expr),+) => {
        ($(
            $tensor.pad_with_zeros($dim, 0, $pad_size)?.contiguous()?,
        )+)
    };
}

macro_rules! reshape_chunk_tensor {
    ($chunk_size:expr, $($tensor:expr),*) => {
        ($(
            {
                let (bs, head, _, dim) = $tensor.dims4()?;
                $tensor.reshape((bs, head, (), $chunk_size, dim))?.contiguous()?
            },
        )*)
    };
}

pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen3_5RMSNormGated,
    out_proj: Linear,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv_state_cache: Option<Tensor>,
    recurrent_state_cache: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d = get_conv1d(
            vb.pp("conv1d"),
            conv_dim,
            conv_dim,
            conv_kernel_size,
            conv_kernel_size - 1,
            1,
            1,
            conv_dim,
            false,
        )?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        let norm = Qwen3_5RMSNormGated::new(vb.pp("norm"), head_v_dim, layer_norm_epsilon)?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;
        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_state_cache: None,
            recurrent_state_cache: None,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        let qkv = self.in_proj_qkv.forward(xs)?;
        let query = qkv.narrow(2, 0, self.key_dim)?;
        let key = qkv.narrow(2, self.key_dim, self.key_dim)?;
        let value = qkv.narrow(2, self.key_dim * 2, self.value_dim)?;

        let query = query.reshape((b_size, seq_len, self.num_k_heads, self.head_k_dim))?.transpose(1, 2)?;
        let key = key.reshape((b_size, seq_len, self.num_k_heads, self.head_k_dim))?.transpose(1, 2)?;
        let value = value.reshape((b_size, seq_len, self.num_v_heads, self.head_v_dim))?.transpose(1, 2)?;

        let z = self.in_proj_z.forward(xs)?.reshape((b_size, seq_len, self.num_v_heads, self.head_v_dim))?.transpose(1, 2)?;
        let a = self.in_proj_a.forward(xs)?;
        let b = self.in_proj_b.forward(&a.silu()?)?;
        let beta = softplus(&b)?.reshape((b_size, seq_len, self.num_v_heads))?.transpose(1, 2)?;

        // DeltaNet Chunk Logic (simplified for compatibility)
        let chunk_size = 128;
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g): (Tensor, Tensor, Tensor, Tensor, Tensor) = transmute_tensors!(query, key, value, beta, z);
        
        let (query, key, value, beta, g) = if seq_len % chunk_size != 0 {
            let pad_size = chunk_size - (seq_len % chunk_size);
            right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g)
        } else {
            (query, key, value, beta, g)
        };

        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let query = (query * scale)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;

        let (batch_size, num_heads, sequence_length, _) = query.dims4()?;
        let g = g.reshape((batch_size, num_heads, (), chunk_size, self.head_v_dim))?.cumsum(3)?;
        
        let num_chunks = sequence_length / chunk_size;
        let mut outputs = Vec::new();
        let mut last_recurrent_state = self.recurrent_state_cache.clone().unwrap_or(
            Tensor::zeros((batch_size, num_heads, self.head_k_dim, self.head_v_dim), candle_core::DType::F32, query.device())?
        );

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let query_i = query.narrow(2, start, chunk_size)?;
            let key_i = key.narrow(2, start, chunk_size)?;
            let value_i = value.narrow(2, start, chunk_size)?;
            let k_beta_i = k_beta.narrow(2, start, chunk_size)?;
            let v_beta_i = v_beta.narrow(2, start, chunk_size)?;
            let g_i = g.narrow(2, i, 1)?.squeeze(2)?;

            // Basic chunk attention with gating
            let attn = query_i.matmul(&key_i.transpose(2, 3)?)?;
            let mut chunk_out = attn.matmul(&v_beta_i)?;
            
            let inter_out = query_i.matmul(&last_recurrent_state)?;
            chunk_out = (chunk_out + inter_out)?;
            outputs.push(chunk_out.to_dtype(initial_dtype)?);

            let kv_i = k_beta_i.transpose(2, 3)?.matmul(&v_beta_i)?;
            last_recurrent_state = (last_recurrent_state + kv_i)?;
        }

        self.recurrent_state_cache = Some(last_recurrent_state);
        let core_attn_out = Tensor::cat(&outputs, 2)?;
        let core_attn_out = core_attn_out.transpose(1, 2)?.reshape((batch_size, sequence_length, ()))?;
        let core_attn_out = if sequence_length > seq_len {
            core_attn_out.narrow(1, 0, seq_len)?
        } else {
            core_attn_out
        };

        let final_out = self.norm.forward(&core_attn_out, None)?;
        Ok(self.out_proj.forward(&final_out)?)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state_cache = None;
        self.recurrent_state_cache = None;
    }
}

pub struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3_5Attention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
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
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

pub struct Qwen3_5DecoderLayer {
    linear_attn: Option<Qwen3_5GatedDeltaNet>,
    self_attn: Option<Qwen3_5Attention>,
    mlp: GateUpDownMLP,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
    layer_type: String,
}

impl Qwen3_5DecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_type = config.layer_types[layer_idx].clone();
        let (linear_attn, self_attn) = if layer_type.eq("linear_attention") {
            let linear_attn = Qwen3_5GatedDeltaNet::new(vb.pp("linear_attn"), config)?;
            (Some(linear_attn), None)
        } else {
            let self_attn = Qwen3_5Attention::new(vb.pp("self_attn"), config)?;
            (None, Some(self_attn))
        };
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
            None,
            None,
            None,
        )?;
        let input_layernorm =
            Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::new(
            vb.pp("post_attention_layernorm"),
            hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            linear_attn,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            layer_type,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let mut xs = self.input_layernorm.forward(xs)?;
        if self.layer_type.eq("linear_attention") {
             if let Some(linear_attn) = self.linear_attn.as_mut() {
                 xs = linear_attn.forward(&xs, attention_mask)?;
             }
        } else if let Some(self_attn) = self.self_attn.as_mut() {
             if let (Some(cos), Some(sin)) = (cos, sin) {
                 xs = self_attn.forward(&xs, cos, sin, attention_mask)?;
             }
        }
        let residual = (xs + residual)?;
        let mut xs = self.post_attention_layernorm.forward(&residual)?;
        xs = self.mlp.forward(&xs)?;
        Ok((xs + residual)?)
    }

    pub fn clear_cache(&mut self) {
        if let Some(linear_attn) = self.linear_attn.as_mut() {
            linear_attn.clear_cache();
        }
        if let Some(self_attn) = self.self_attn.as_mut() {
            self_attn.clear_kv_cache();
        }
    }
}

pub struct Qwen3_5TextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    pub norm: Qwen3_5RMSNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
}

impl Qwen3_5TextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3_5DecoderLayer::new(vb.pp(&format!("layers.{}", i)), config, i)?;
            layers.push(layer);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary_emb =
            Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section: config.rope_parameters.mrope_section.clone(),
        })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _hidden_size) = inputs_embeds.dims3()?;
        let (cos, sin) = self.rotary_emb.forward(position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        let attention_mask = prepare_causal_attention_mask(b_size, seq_len, 0, inputs_embeds.device())?;

        let mut xs = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, Some(&cos), Some(&sin), Some(&attention_mask))?;
        }
        Ok(self.norm.forward(&xs)?)
    }

    pub fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

pub struct Qwen3_5Model {
    config: Qwen3_5Config,
    visual: Option<Qwen3VLVisionModel>,
    language_model: Qwen3_5TextModel,
    lm_head: Linear,
}

impl Qwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let vb_m = vb.pp("model.language_model");
        let vb_v = vb.pp("model.visual");
        
        let visual = if vb.contains_tensor("model.visual.patch_embed.proj.weight") {
             Some(Qwen3VLVisionModel::new(config.vision_config.clone(), vb_v)?)
        } else {
             None
        };

        let language_model = Qwen3_5TextModel::new(vb_m, &config.text_config)?;
        
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };

        Ok(Self {
            config,
            visual,
            language_model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;

        if let Some(visual) = &self.visual {
            if let (Some(pixel_values), Some(image_grid_thw)) = (pixel_values, image_grid_thw) {
                let (image_embeds, _) = visual.forward(pixel_values, image_grid_thw)?;
                let vision_mask = get_equal_mask(input_ids, self.config.image_token_id)?;
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
            }
        }

        if let Some(visual) = &self.visual {
            if let (Some(pixel_values_video), Some(video_grid_thw)) = (pixel_values_video, video_grid_thw) {
                let (video_embeds, _) = visual.forward(pixel_values_video, video_grid_thw)?;
                let vision_mask = get_equal_mask(input_ids, self.config.video_token_id)?;
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
            }
        }

        let position_ids = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, input_ids.device())?
            .unsqueeze(0)?;

        let outputs = self.language_model.forward(&inputs_embeds, &position_ids)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
    }
}
