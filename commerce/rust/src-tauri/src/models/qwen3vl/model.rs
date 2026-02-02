use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor, Module};
use candle_nn::{
    Activation, Embedding, Init, Linear, VarBuilder,
};

use crate::{
    models::{
        common::{TwoLinearMLP, eager_attention_forward, get_layer_norm, RmsNorm, LayerNorm, rms_norm as get_rms_norm},
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig},
            quantized_model::QLinear, // [VRAM-OPTIM] Import QLinear
        },
    },
    position_embed::rope::{
        Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
        apply_rotary_pos_emb_vision,
    },
    utils::tensor_utils::{
        bitor_tensor, get_vision_next_indices, linspace, mask_index_add, masked_scatter_dim0,
        nonzero_index, prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
        zero_index,
    },
};

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchEmbed {
    conv3d_weight: Tensor,
    conv3d_bias: Tensor,
}

impl Qwen3VLVisionPatchEmbed {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let temporal_patch_size = cfg.temporal_patch_size;
        let in_channels = cfg.in_channels;
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size);
        
        // [SMART-PROBE] Match mmproj naming variations
        let (weight_name, bias_name) = if vb.get_with_hints((1,), "proj.weight", Init::Const(0.)).is_ok() {
            ("proj.weight", "proj.bias")
        } else if vb.get_with_hints((1,), "weight", Init::Const(0.)).is_ok() {
            ("weight", "bias")
        } else if vb.get_with_hints((1,), "weight.packed", Init::Const(0.)).is_ok() {
            ("weight.packed", "bias")
        } else {
            ("proj.weight", "proj.bias")
        };

        let conv3d_weight = vb
            .get_with_hints(
                (
                    embed_dim,
                    in_channels,
                    temporal_patch_size,
                    patch_size,
                    patch_size,
                ),
                weight_name,
                Init::Const(1.),
            )?
            .flatten(1, 4)?
            .t()?;

        let conv3d_bias = vb
            .get_with_hints((embed_dim,), bias_name, Init::Const(0.))?
            .unsqueeze(0)?;
        Ok(Self {
            conv3d_weight,
            conv3d_bias,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.conv3d_weight = self.conv3d_weight.to_device(device)?;
        self.conv3d_bias = self.conv3d_bias.to_device(device)?;
        Ok(())
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = hidden_states.matmul(&self.conv3d_weight)?;
        let hidden_states = hidden_states.broadcast_add(&self.conv3d_bias)?;
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionPatchMerger {
    hidden_size: usize,
    use_postshuffle_norm: bool,
    norm: LayerNorm,
    linear_fc1: Linear,
    act_fn: Activation,
    linear_fc2: Linear,
}

impl Qwen3VLVisionPatchMerger {
    pub fn new(config: &Qwen3VLVisionConfig, vb: VarBuilder, use_postshuffle_norm: bool) -> Result<Self> {
        let hidden_size = config.hidden_size * config.spatial_merge_size.pow(2);
        let norm_size = if use_postshuffle_norm { hidden_size } else { config.hidden_size };
        let (fc1_name, fc2_name, norm_name) = if vb.pp("linear_fc1").get_with_hints((1,), "weight", Init::Const(0.)).is_ok() {
            ("linear_fc1", "linear_fc2", "norm")
        } else if vb.get_with_hints((1,), "0.weight", Init::Const(0.)).is_ok() || vb.get_with_hints((1,), "0.weight.packed", Init::Const(0.)).is_ok() {
            ("0", "2", "norm")
        } else {
            ("linear_fc1", "linear_fc2", "norm")
        };
        let norm = get_layer_norm(vb.pp(norm_name), 1e-6, norm_size)?;
        let find_v_weight = |v: &VarBuilder, out_d: usize, in_d: usize| -> Result<Tensor> {
            for suffix in &["", ".packed", ".min"] {
                let full_n = format!("weight{}", suffix);
                if let Ok(t) = v.get_with_hints((1,), &full_n, Init::Const(0.)) {
                    return Ok(v.get(t.shape(), &full_n)?);
                }
            }
            Ok(v.get((out_d, in_d), "weight")?)
        };
        let fc1_w = find_v_weight(&vb.pp(fc1_name), hidden_size, hidden_size)?;
        let fc1_b = vb.pp(fc1_name).get(hidden_size, "bias").ok();
        let fc2_w = find_v_weight(&vb.pp(fc2_name), config.out_hidden_size, hidden_size)?;
        let fc2_b = vb.pp(fc2_name).get(config.out_hidden_size, "bias").ok();
        Ok(Self { hidden_size, use_postshuffle_norm, norm, linear_fc1: Linear::new(fc1_w, fc1_b), act_fn: Activation::Gelu, linear_fc2: Linear::new(fc2_w, fc2_b) })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let n_w = self.norm.weight.to_device(device)?;
        let n_b = self.norm.bias.to_device(device)?;
        self.norm = LayerNorm::new(n_w, n_b, 1e-6);
        let l1_w = self.linear_fc1.weight().to_device(device)?;
        let l1_b = self.linear_fc1.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear_fc1 = Linear::new(l1_w, l1_b);
        let l2_w = self.linear_fc2.weight().to_device(device)?;
        let l2_b = self.linear_fc2.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear_fc2 = Linear::new(l2_w, l2_b); Ok(())
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.use_postshuffle_norm { xs.reshape(((), self.hidden_size))? } else { xs.clone() };
        let xs = self.norm.forward(&xs)?.reshape(((), self.hidden_size))?;
        let xs = self.linear_fc2.forward(&self.act_fn.forward(&self.linear_fc1.forward(&xs)?)?)?; Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionAttention {
    num_heads: usize, qkv: Linear, proj: Linear, scaling: f64,
}

impl Qwen3VLVisionAttention {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hidden_size / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let find_attn_weight = |v: &VarBuilder, name: &str, out_d: usize, in_d: usize| -> Result<Tensor> {
            for n in &[name, &format!("attn_{}", name)] {
                for suffix in &["", ".packed", ".min"] {
                    let full_n = format!("{}{}", n, suffix);
                    if let Ok(t) = v.get_with_hints((1,), &full_n, Init::Const(0.)) { return Ok(v.get(t.shape(), &full_n)?); }
                }
            }
            Ok(v.get((out_d, in_d), name)?)
        };
        let qkv_w = find_attn_weight(&vb, "qkv", hidden_size * 3, hidden_size)?;
        let qkv_b = vb.get(hidden_size * 3, "qkv.bias").or_else(|_| vb.get(hidden_size * 3, "attn_qkv.bias")).ok();
        let proj_w = find_attn_weight(&vb, "proj", hidden_size, hidden_size).or_else(|_| find_attn_weight(&vb, "out", hidden_size, hidden_size))?;
        let proj_b = vb.get(hidden_size, "proj.bias").or_else(|_| vb.get(hidden_size, "attn_out.bias")).ok();
        Ok(Self { num_heads, qkv: Linear::new(qkv_w, qkv_b), proj: Linear::new(proj_w, proj_b), scaling })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let qkv_w = self.qkv.weight().to_device(device)?; let qkv_b = self.qkv.bias().map(|b| b.to_device(device)).transpose()?; self.qkv = Linear::new(qkv_w, qkv_b);
        let proj_w = self.proj.weight().to_device(device)?; let proj_b = self.proj.bias().map(|b| b.to_device(device)).transpose()?; self.proj = Linear::new(proj_w, proj_b); Ok(())
    }
    pub fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor, cu_seqlens: &Tensor) -> Result<Tensor> {
        let seq_length = xs.dim(0)?;
        let qkv_states = xs.apply(&self.qkv)?.reshape((seq_length, 3, self.num_heads, ()))?.permute((1, 0, 2, 3))?;
        let query_states = qkv_states.i(0)?.contiguous()?; let key_states = qkv_states.i(1)?.contiguous()?; let value_states = qkv_states.i(2)?.contiguous()?;
        let (query_states, key_states) = apply_rotary_pos_emb_vision(&query_states, &key_states, cos, sin)?;
        let query_states = query_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let key_states = key_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let value_states = value_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let cu_last_id = cu_seqlens.dim(0)? - 1;
        let lengths = cu_seqlens.i(1..)?.sub(&cu_seqlens.i(..cu_last_id)?)?;
        let chunks: Vec<usize> = lengths.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
        let q_splits = split_tensor(&query_states, &chunks, 2)?; let k_splits = split_tensor(&key_states, &chunks, 2)?; let v_splits = split_tensor(&value_states, &chunks, 2)?;
        let mut attn_outputs = Vec::new();
        for (q, (k, v)) in q_splits.iter().zip(k_splits.iter().zip(v_splits.iter())) {
            let output = eager_attention_forward(q, k, v, None, None, self.scaling)?; attn_outputs.push(output);
        }
        let attn_output = Tensor::cat(&attn_outputs, 1)?;
        let attn_output = attn_output.reshape((seq_length, ()))?.contiguous()?; Ok(attn_output.apply(&self.proj)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionBlock {
    norm1: LayerNorm, norm2: LayerNorm, attn: Qwen3VLVisionAttention, mlp: TwoLinearMLP,
}

impl Qwen3VLVisionBlock {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let (n1_n, n2_n, attn_n) = if vb.get_with_hints((1,), "norm1.weight", Init::Const(0.)).is_ok() { ("norm1", "norm2", "attn") } else { ("ln1", "ln2", "") };
        let norm1 = get_layer_norm(vb.pp(n1_n), 1e-6, config.hidden_size)?;
        let norm2 = get_layer_norm(vb.pp(n2_n), 1e-6, config.hidden_size)?;
        let attn_vb = if attn_n.is_empty() { vb.clone() } else { vb.pp(attn_n) };
        let attn = Qwen3VLVisionAttention::new(config.clone(), attn_vb)?;
        let mlp_vb = if vb.pp("mlp").get_with_hints((1,), "linear_fc1.weight", Init::Const(0.)).is_ok() { vb.pp("mlp") } else { vb.clone() };
        let mlp = TwoLinearMLP::new(mlp_vb, config.hidden_size, config.intermediate_size, Activation::Gelu, false, "linear_fc1", "linear_fc2").or_else(|_| TwoLinearMLP::new(vb.clone(), config.hidden_size, config.intermediate_size, Activation::Gelu, false, "ffn_up", "ffn_down"))?;
        Ok(Self { norm1, norm2, attn, mlp })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let n1_w = self.norm1.weight.to_device(device)?; let n1_b = self.norm1.bias.to_device(device)?; self.norm1 = LayerNorm::new(n1_w, n1_b, 1e-6);
        let n2_w = self.norm2.weight.to_device(device)?; let n2_b = self.norm2.bias.to_device(device)?; self.norm2 = LayerNorm::new(n2_w, n2_b, 1e-6);
        self.attn.to_device(device)?; self.mlp.to_device(device)?; Ok(())
    }
    pub fn forward(&self, xs: &Tensor, cu_seqlens: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = xs.clone(); let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, cu_seqlens)?;
        let xs = residual.add(&xs)?; let residual = xs.clone();
        let xs = self.mlp.forward(&self.norm2.forward(&xs)?)?; Ok(residual.add(&xs)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionModel {
    pub spatial_merge_size: usize, pub patch_embed: Qwen3VLVisionPatchEmbed, pub pos_embed: Embedding, pub num_grid_per_side: u32,
    pub rotary_pos_emb: Qwen2_5VisionRotaryEmbedding, pub blocks: Vec<Qwen3VLVisionBlock>, pub merger: Qwen3VLVisionPatchMerger,
    pub deepstack_visual_indexes: Vec<usize>, pub deepstack_merger_list: Vec<Qwen3VLVisionPatchMerger>, pub dtype: DType,
}

impl Qwen3VLVisionModel {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = config.spatial_merge_size;
        let pe_vb = if vb.pp("patch_embed").get_with_hints((1,), "proj.weight", Init::Const(0.)).is_ok() { vb.pp("patch_embed") } else { vb.pp("patch_embd") };
        let patch_embed = Qwen3VLVisionPatchEmbed::new(&config, pe_vb)?;
        let pos_vb = if vb.pp("pos_embed").get_with_hints((1,), "weight", Init::Const(0.)).is_ok() { vb.pp("pos_embed") } else { vb.pp("position_embd") };
        let pos_w = if let Ok(t) = pos_vb.get_with_hints((1,), "weight", Init::Const(0.)) { pos_vb.get(t.shape(), "weight")? } else { pos_vb.get((config.num_position_embeddings, config.hidden_size), "weight")? };
        let pos_embed = Embedding::new(pos_w, config.hidden_size);
        let num_grid_per_side = (config.num_position_embeddings as f32).sqrt() as u32;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let mut blocks = Vec::new();
        let blocks_vb = if vb.pp("blocks").pp(0).get_with_hints((1,), "norm1.weight", Init::Const(0.)).is_ok() { vb.pp("blocks") } else { vb.pp("blk") };
        for i in 0..config.depth { blocks.push(Qwen3VLVisionBlock::new(config.clone(), blocks_vb.pp(i))?); }
        let merger_vb = if vb.pp("merger").get_with_hints((1,), "linear_fc1.weight", Init::Const(0.)).is_ok() { vb.pp("merger") } else { vb.pp("mm") };
        let merger = Qwen3VLVisionPatchMerger::new(&config, merger_vb, false)?;
        let deepstack_visual_indexes = config.deepstack_visual_indexes.clone();
        let mut deepstack_merger_list = Vec::new();
        let vb_deepstack = vb.pp("deepstack_merger_list");
        for i in 0..deepstack_visual_indexes.len() { deepstack_merger_list.push(Qwen3VLVisionPatchMerger::new(&config, vb_deepstack.pp(i), true)?); }
        Ok(Self { spatial_merge_size, patch_embed, pos_embed, num_grid_per_side, rotary_pos_emb, blocks, merger, deepstack_visual_indexes, deepstack_merger_list, dtype: vb.dtype() })
    }
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embed.to_device(device)?; let p_w = self.pos_embed.embeddings().to_device(device)?; self.pos_embed = Embedding::new(p_w, self.pos_embed.hidden_size());
        for block in self.blocks.iter_mut() { block.to_device(device)?; }
        self.merger.to_device(device)?; for merger in self.deepstack_merger_list.iter_mut() { merger.to_device(device)?; } Ok(())
    }
    pub fn forward(&self, hidden_states: &Tensor, grid_thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let hidden_states = self.patch_embed.forward(hidden_states)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let hidden_states = hidden_states.broadcast_add(&pos_embeds)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?; let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?; let sin = emb.sin()?;
        let cu_seqlens = grid_thw.i((.., 1))?.mul(&grid_thw.i((.., 2))?)?;
        let grid_t = grid_thw.i((.., 0))?.to_vec1::<u32>()?;
        let mut cu_seqlens_repeat = Vec::new();
        for (index, t) in grid_t.iter().enumerate() { cu_seqlens_repeat.push(cu_seqlens.i(index)?.repeat(*t as usize)?); }
        let cu_seqlens_full = Tensor::cat(&cu_seqlens_repeat, 0)?.flatten_all()?;
        let cu_seqlens = cu_seqlens_full.to_dtype(DType::F64)?.cumsum(0)?.to_dtype(DType::U32)?.pad_with_zeros(D::Minus1, 1, 0)?;
        let mut deepstack_feature_lists = vec![];
        for (layer_num, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
            if let Some(index) = self.deepstack_visual_indexes.iter().position(|&x| x == layer_num) {
                deepstack_feature_lists.push(self.deepstack_merger_list[index].forward(&hidden_states)?);
            }
        }
        hidden_states = self.merger.forward(&hidden_states)?; Ok((hidden_states, deepstack_feature_lists))
    }
    pub fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let mut idx_list = vec![vec![]; 4]; let mut weight_list = vec![vec![]; 4]; let mut split_idx = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("grid_thw Expected exactly 3 elements")); };
            split_idx.push((h * w) as usize); let num_grid_per_side_sub_one = (self.num_grid_per_side - 1) as f32;
            let h_idxs = linspace(0.0, num_grid_per_side_sub_one, h as usize, grid_thw.device())?;
            let w_idxs = linspace(0.0, num_grid_per_side_sub_one, w as usize, grid_thw.device())?;
            let h_idxs_f = h_idxs.to_dtype(DType::U32)?; let w_idxs_f = w_idxs.to_dtype(DType::U32)?;
            let h_idxs_c = h_idxs_f.affine(1.0, 1.0)?.clamp(0u32, num_grid_per_side_sub_one as u32)?;
            let w_idxs_c = w_idxs_f.affine(1.0, 1.0)?.clamp(0u32, num_grid_per_side_sub_one as u32)?;
            let dh = h_idxs.sub(&h_idxs_f.to_dtype(h_idxs.dtype())?)?.unsqueeze(D::Minus1)?;
            let dw = w_idxs.sub(&w_idxs_f.to_dtype(h_idxs.dtype())?)?.unsqueeze(0)?;
            let base_h = h_idxs_f.affine(self.num_grid_per_side as f64, 0.0)?.unsqueeze(D::Minus1)?;
            let base_h_c = h_idxs_c.affine(self.num_grid_per_side as f64, 0.0)?.unsqueeze(D::Minus1)?;
            idx_list[0].extend_from_slice(&base_h.broadcast_add(&w_idxs_f.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idx_list[1].extend_from_slice(&base_h.broadcast_add(&w_idxs_c.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idx_list[2].extend_from_slice(&base_h_c.broadcast_add(&w_idxs_f.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            idx_list[3].extend_from_slice(&base_h_c.broadcast_add(&w_idxs_c.unsqueeze(0)?)?.flatten_all()?.to_vec1::<u32>()?);
            let one_sub_dh = Tensor::ones_like(&dh)?.sub(&dh)?; let one_sub_dw = Tensor::ones_like(&dw)?.sub(&dw)?;
            weight_list[0].extend_from_slice(&one_sub_dh.broadcast_mul(&one_sub_dw)?.flatten_all()?.to_vec1::<f32>()?);
            weight_list[1].extend_from_slice(&one_sub_dh.broadcast_mul(&dw)?.flatten_all()?.to_vec1::<f32>()?);
            weight_list[2].extend_from_slice(&dh.broadcast_mul(&one_sub_dw)?.flatten_all()?.to_vec1::<f32>()?);
            weight_list[3].extend_from_slice(&dh.broadcast_mul(&dw)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let idx_tensor = Tensor::new(idx_list, grid_thw.device())?; let weight_tensor = Tensor::new(weight_list, grid_thw.device())?.to_dtype(self.dtype)?;
        let pos_embeds = self.pos_embed.forward(&idx_tensor)?.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let patch_pos_embeds = pos_embeds.i(0)?.add(&pos_embeds.i(1)?)?.add(&pos_embeds.i(2)?)?.add(&pos_embeds.i(3)?)?;
        let mut patch_pos_embeds_p = vec![]; let patch_pos_embeds = split_tensor(&patch_pos_embeds, &split_idx, 0)?;
        let merge_size = self.spatial_merge_size;
        for (i, pos_embed) in patch_pos_embeds.iter().enumerate() {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("grid_thw Expected exactly 3 elements")); };
            let last_dim: usize = pos_embed.dim(D::Minus1)?; let pos_embed = pos_embed.repeat((t as usize, 1))?;
            let shape = Shape::from(vec![t as usize, h as usize / merge_size, merge_size, w as usize / merge_size, merge_size, last_dim]);
            patch_pos_embeds_p.push(pos_embed.reshape(shape)?.permute((0, 1, 3, 2, 4, 5))?.flatten(0, 4)?);
        }
        Ok(Tensor::cat(&patch_pos_embeds_p, 0)?)
    }
    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let merge_size = self.spatial_merge_size; let max_hw = grid_thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let freq_table = self.rotary_pos_emb.forward(max_hw as usize, grid_thw.device())?;
        let mut pos_ids_vec = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else { return Err(anyhow!("grid_thw Expected exactly 3 elements")); };
            let merged_h = h / merge_size as u32; let merged_w = w / merge_size as u32;
            let blocks_rows = Tensor::arange(0, merged_h, grid_thw.device())?; let blocks_cols = Tensor::arange(0, merged_w, grid_thw.device())?;
            let intra_row = Tensor::arange(0, merge_size as u32, grid_thw.device())?; let intra_col = Tensor::arange(0, merge_size as u32, grid_thw.device())?;
            let row_idx = blocks_rows.reshape(((), 1, 1, 1))?.contiguous()?.affine(merge_size as f64, 0.0)?.broadcast_add(&intra_row.reshape((1, 1, (), 1))?.contiguous()?)?;
            let col_idx = blocks_cols.reshape((1, (), 1, 1))?.contiguous()?.affine(merge_size as f64, 0.0)?.broadcast_add(&intra_col.reshape((1, 1, 1, ()))?.contiguous()?)?;
            let mut coords = Tensor::stack(&[row_idx.expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?.flatten_all()?, col_idx.expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?.flatten_all()?], D::Minus1)?.contiguous()?;
            if t > 1 { coords = coords.repeat((t as usize, 1))?; } pos_ids_vec.push(coords);
        }
        let pos_ids = Tensor::cat(&pos_ids_vec, 0)?;
        let rotary_pos_emb_h = freq_table.index_select(&pos_ids.i((.., 0))?.contiguous()?, 0)?;
        let rotary_pos_emb_w = freq_table.index_select(&pos_ids.i((.., 1))?.contiguous()?, 0)?;
        Ok(Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?)
    }
}

fn get_qlinear_from_vb(vb: VarBuilder, name: &str) -> Result<QLinear> {
    let prefix = if name.is_empty() { "".to_string() } else { format!("{}.", name) };
    let packed_name = format!("{}packed", prefix);
    let scales_name = format!("{}scales", prefix);
    let shape_name = format!("{}shape", prefix);
    let bias_name = format!("{}bias", prefix);
    if probe(&vb, &packed_name) {
        let packed = get_any(&vb, &packed_name)?; let scales_raw = get_any(&vb, &scales_name)?; let shape_t = get_any(&vb, &shape_name)?;
        let scales = scales_raw.to_dtype(DType::F32)?;
        let shape_vec: Vec<usize> = shape_t.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect();
        let bias = if probe(&vb, &bias_name) { Some(get_any(&vb, &bias_name)?) } else { None };
        return Ok(QLinear::new(packed, scales, shape_vec, bias, vb.device().clone()));
    }
    let weight_name = if name.is_empty() { "weight".to_string() } else { name.to_string() };
    let weight = get_any(&vb, &weight_name)?; let s = weight.dims().to_vec(); let total_el = s.iter().product::<usize>();
    let scales = Tensor::ones((total_el / 32).max(1), DType::F32, vb.device())?;
    let packed = Tensor::zeros((total_el / 32).max(1), DType::U32, vb.device())?;
    let bias = if probe(&vb, &bias_name) { Some(get_any(&vb, &bias_name)?) } else { None };
    Ok(QLinear::new(packed, scales, s, bias, vb.device().clone()))
}

fn probe(v: &VarBuilder, p: &str) -> bool {
    match v.get_with_hints((1,), p, Init::Const(0.)) {
        Ok(_) => true,
        Err(e) => { let s = e.to_string().to_lowercase(); s.contains("shape mismatch") || s.contains("dtype mismatch") }
    }
}

fn get_any(v: &VarBuilder, p: &str) -> Result<Tensor> {
    match v.get_with_hints((1,), p, Init::Const(0.)) {
        Ok(t) => Ok(t),
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("shape mismatch") {
                if let Some(start) = err_str.find("got: [") {
                    let rest = &err_str[start + 6..];
                    if let Some(end) = rest.find(']') {
                        let dims: Vec<usize> = rest[..end].split(',').map(|s| s.trim().parse::<usize>()).filter_map(|r| r.ok()).collect();
                        if !dims.is_empty() { return Ok(v.get(dims, p)?); }
                    }
                }
            }
            Err(anyhow!("Failed to load tensor '{}': {}", p, e))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextAttention {
    pub q_proj: QLinear, pub k_proj: QLinear, pub v_proj: QLinear, pub o_proj: QLinear,
    pub q_norm: RmsNorm, pub k_norm: RmsNorm,
    pub num_attention_heads: usize, pub num_key_value_heads: usize, pub head_dim: usize,
    pub num_kv_groups: usize, pub scaling: f64, pub kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3VLTextAttention {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        println!("[DEBUG-ATTN] Creating Attention. Prefix: {}", vb.prefix());
        let num_attention_heads = config.num_attention_heads; let head_dim = config.head_dim; let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads; let scaling = 1f64 / f64::sqrt(head_dim as f64);
        println!("[DEBUG-ATTN] Loading projections...");
        let q_proj = get_qlinear_from_vb(vb.pp("q_proj"), "weight")?; let k_proj = get_qlinear_from_vb(vb.pp("k_proj"), "weight")?;
        let v_proj = get_qlinear_from_vb(vb.pp("v_proj"), "weight")?; let o_proj = get_qlinear_from_vb(vb.pp("o_proj"), "weight")?;
        println!("[DEBUG-ATTN] Loading norms...");
        let q_norm = get_rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = get_rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
        println!("[DEBUG-ATTN] Attention created successfully.");
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling, kv_cache: None })
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        println!("[DEBUG-ATTN] forward start. xs: {:?}", xs.dims());
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache { None => (key_states, value_states), Some((pk, pv)) => (Tensor::cat(&[pk, &key_states], 2)?, Tensor::cat(&[pv, &value_states], 2)?) };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let attn_output = eager_attention_forward(&query_states, &key_states, &value_states, Some(self.num_kv_groups), attention_mask, self.scaling)?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?; Ok(self.o_proj.forward(&attn_output)?)
    }
    pub fn clear_kv_cache(&mut self) { self.kv_cache = None }
}

#[derive(Debug, Clone)]
pub struct QGateUpDownMLP {
    pub gate_proj: QLinear, pub up_proj: QLinear, pub down_proj: QLinear, pub act_fn: Activation,
}

impl QGateUpDownMLP {
    pub fn new(vb: VarBuilder, _hidden: usize, _inter: usize, act: Activation) -> Result<Self> {
        println!("[DEBUG-MLP] Creating MLP. Prefix: {}", vb.prefix());
        let gate_proj = get_qlinear_from_vb(vb.pp("gate_proj"), "weight")?;
        let up_proj = get_qlinear_from_vb(vb.pp("up_proj"), "weight")?;
        let down_proj = get_qlinear_from_vb(vb.pp("down_proj"), "weight")?;
        println!("[DEBUG-MLP] MLP created successfully.");
        Ok(Self { gate_proj, up_proj, down_proj, act_fn: act })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?; let gate = self.act_fn.forward(&gate)?;
        let up = self.up_proj.forward(x)?; let x = gate.mul(&up)?; Ok(self.down_proj.forward(&x)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextDecoderLayer {
    pub self_attn: Qwen3VLTextAttention, pub mlp: QGateUpDownMLP, pub input_layernorm: RmsNorm, pub post_attention_layernorm: RmsNorm,
}

impl Qwen3VLTextDecoderLayer {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        println!("[DEBUG-LAYER] Creating Layer. Prefix: {}", vb.prefix());
        let self_attn = Qwen3VLTextAttention::new(config.clone(), vb.pp("self_attn"))?;
        let mlp = QGateUpDownMLP::new(vb.pp("mlp"), config.hidden_size, config.intermediate_size, config.hidden_act)?;
        println!("[DEBUG-LAYER] Loading norms...");
        let input_layernorm = { let w = get_any(&vb.pp("input_layernorm"), "weight")?; RmsNorm::new(w, config.rms_norm_eps) };
        let post_attention_layernorm = { let w = get_any(&vb.pp("post_attention_layernorm"), "weight")?; RmsNorm::new(w, config.rms_norm_eps) };
        println!("[DEBUG-LAYER] Layer created successfully.");
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }
    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        println!("[DEBUG-LAYER] forward start. xs: {:?}", xs.dims());
        let residual = xs.clone(); let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = residual.add(&xs)?; let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?; Ok(residual.add(&xs)?)
    }
    pub fn clear_kv_cache(&mut self) { self.self_attn.clear_kv_cache(); }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextModel {
    pub embed_tokens: Embedding, pub layers: Vec<Option<Qwen3VLTextDecoderLayer>>, pub norm: RmsNorm, pub rotary_emb: Qwen3VLTextRotaryEmbedding, pub mrope_section: Vec<usize>, pub is_baking: bool,
}

impl Qwen3VLTextModel {
    pub fn new(mut config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        println!("[DEBUG-TEXT] Creating TextModel. Prefix: {}", vb.prefix());
        let vocab_size = config.vocab_size;
        println!("[DEBUG-TEXT] Loading embeddings...");
        let embed_tokens_weight = if probe(&vb.pp("embed_tokens"), "weight") { get_any(&vb.pp("embed_tokens"), "weight")? } else { println!("[DEBUG-TEXT] Embeddings missing, using dummy."); Tensor::zeros((vocab_size, config.hidden_size), DType::F32, vb.device())? };
        let actual_h = embed_tokens_weight.dim(1)?; if actual_h != config.hidden_size && actual_h > 0 { println!("[MODEL-FIX] Hidden Size Mismatch. Config: {}, Actual: {}. Patching...", config.hidden_size, actual_h); config.hidden_size = actual_h; }
        let embed_tokens = Embedding::new(embed_tokens_weight, config.hidden_size);
        let mut layers = vec![]; let vb_l = vb.pp("layers");
        println!("[DEBUG-TEXT] Loading {} layers...", config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let check_path = vb_l.pp(layer_idx).pp("input_layernorm");
            let mut layer_exists = probe(&check_path, "weight");
            if !layer_exists { layer_exists = probe(&vb_l.pp(layer_idx).pp("attn_norm"), "weight"); }
            if !layer_exists { layer_exists = probe(&vb, &format!("layers.{}.input_layernorm.weight", layer_idx)); }
            println!("[MODEL-DEBUG] Probing Layer {}: Found={}", layer_idx, layer_exists);
            if layer_exists { layers.push(Some(Qwen3VLTextDecoderLayer::new(config.clone(), vb_l.pp(layer_idx))?)); } else { layers.push(None); }
        }
        println!("[DEBUG-TEXT] Loading final norm...");
        let norm = if probe(&vb.pp("norm"), "weight") { RmsNorm::new(get_any(&vb.pp("norm"), "weight")?, config.rms_norm_eps) } else if probe(&vb.pp("output_norm"), "weight") { RmsNorm::new(get_any(&vb.pp("output_norm"), "weight")?, config.rms_norm_eps) } else { println!("[DEBUG-TEXT] Final norm missing, using dummy."); RmsNorm::new(Tensor::ones(config.hidden_size, DType::F32, vb.device())?, config.rms_norm_eps) };
        println!("[DEBUG-TEXT] Initializing RoPE...");
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        println!("[DEBUG-TEXT] TextModel created successfully.");
        Ok(Self { embed_tokens, layers, norm, rotary_emb, mrope_section, is_baking: false })
    }
    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize, position_ids: Option<&Tensor>, visual_pos_masks: Option<&Tensor>, deepstack_visual_embeds: Option<Vec<Tensor>>) -> Result<Tensor> {
        println!("[TRACE-0] Qwen3VLModel::forward entry. input_ids: {:?}", inputs_embeds.dims());
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?,
        };
        println!("[TRACE-1] Computing RoPE cos/sin...");
        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), self.mrope_section.clone())?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = if seq_len <= 1 { None } else { println!("[TRACE-2] Preparing attention mask..."); Some(prepare_causal_attention_mask(b_size, seq_len, 0, inputs_embeds.device())?) };
        let layer_limit = if self.is_baking { 1 } else { self.layers.len() };
        let mut layers_executed = 0;
        for (layer_idx, layer_opt) in self.layers.iter_mut().enumerate().take(layer_limit) {
            if let Some(layer) = layer_opt {
                println!("[TRACE-L{}] Layer forward start", layer_idx);
                xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?; layers_executed += 1;
                if let Some(ds) = deepstack_visual_embeds.as_ref() { if layer_idx < ds.len() { xs = mask_index_add(&xs.squeeze(0)?, &visual_pos_masks.unwrap().squeeze(0)?, &ds[layer_idx])?.unsqueeze(0)?; } }
                println!("[TRACE-L{}] Layer forward complete", layer_idx);
            }
        }
        if layers_executed == 0 { println!("[MODEL-WARNING] 0 layers executed!"); }
        println!("[TRACE-3] Final norm..."); Ok(xs.apply(&self.norm)?)
    }
    pub fn clear_kv_cache(&mut self) { for layer in self.layers.iter_mut() { if let Some(l) = layer { l.clear_kv_cache() } } }
    pub fn save_kv_cache(&mut self, path: &std::path::Path) -> Result<()> {
        if !path.exists() { std::fs::create_dir_all(path)?; }
        for (i, layer_opt) in self.layers.iter_mut().enumerate() {
            if let Some(layer) = layer_opt {
                if let Some((k, v)) = &layer.self_attn.kv_cache {
                    let rk = Self::compress_to_bitkv(i, k)?; let rv = Self::compress_to_bitkv(i, v)?;
                    let mut m = std::collections::HashMap::new();
                    m.insert("k_a".to_string(), rk.0); m.insert("k_p".to_string(), rk.1); m.insert("k_s".to_string(), rk.2);
                    m.insert("v_a".to_string(), rv.0); m.insert("v_p".to_string(), rv.1); m.insert("v_s".to_string(), rv.2);
                    m.insert("shape".to_string(), Tensor::from_vec(rk.3.iter().map(|&x| x as u32).collect::<Vec<_>>(), (rk.3.len(),), k.device())?);
                    candle_core::safetensors::save(&m, path.join(format!("layer_{}_bitkv.safetensors", i)))?;
                }
            }
        }
        Ok(())
    }
    fn compress_to_bitkv(l_i: usize, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        let s = t.dims().to_vec(); let d = t.device(); if l_i == 0 { return Ok((t.clone(), Tensor::zeros(1, DType::U8, d)?, Tensor::zeros(1, DType::F32, d)?, s)); }
        let f = t.flatten_all()?; let n = f.dim(0)?;
        if l_i <= 4 {
            let sc = (f.abs()?.max_all()?.to_scalar::<f32>()? / 3.0).max(1e-6);
            let q = f.to_vec1::<f32>()?; let mut p = vec![0u8; (n + 3) / 4];
            for (i, &v) in q.iter().enumerate() { let qv = ((v / sc + 1.0).round() as u8).clamp(0, 3); p[i / 4] |= qv << ((i % 4) * 2); }
            Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n + 3) / 4,), d)?, Tensor::new(&[sc], d)?, s))
        } else {
            let sc = f.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let sv = f.ge(0.0)?.to_vec1::<u8>()?; let mut p = vec![0u8; (n + 7) / 8];
            for (i, &sv) in sv.iter().enumerate() { if sv > 0 { p[i / 8] |= 1 << (i % 8); } }
            Ok((Tensor::zeros(1, t.dtype(), d)?, Tensor::from_vec(p, ((n + 7) / 8,), d)?, Tensor::new(&[sc], d)?, s))
        }
    }
    pub fn load_kv_cache(&mut self, path: &std::path::Path, device: &Device) -> Result<()> {
        let mut first_layer_kv: Option<(Tensor, Tensor)> = None;
        let target_dtype = if device.is_cpu() { candle_core::DType::F32 } else { candle_core::DType::BF16 };
        for (i, layer_opt) in self.layers.iter_mut().enumerate() {
            if let Some(layer) = layer_opt {
                let p = path.join(format!("layer_{}_kv.safetensors", i));
                let (mut k, mut v) = if p.exists() {
                    let m = candle_core::safetensors::load(p, device)?; (m.get("k").unwrap().to_dtype(target_dtype)?, m.get("v").unwrap().to_dtype(target_dtype)?)
                } else if let Some((ref fk, ref fv)) = first_layer_kv { (fk.clone(), fv.clone()) } else { continue; };
                let target_h = layer.self_attn.num_key_value_heads; let target_d = layer.self_attn.head_dim;
                if k.dim(1)? != target_h { if target_h % k.dim(1)? == 0 { let r = target_h / k.dim(1)?; k = k.repeat((1, r, 1, 1))?; v = v.repeat((1, r, 1, 1))?; } else { k = k.narrow(1, 0, target_h)?; v = v.narrow(1, 0, target_h)?; } }
                if k.dim(3)? != target_d { k = Self::apply_linear_bridge(&k, target_d)?; v = Self::apply_linear_bridge(&v, target_d)?; }
                if first_layer_kv.is_none() { first_layer_kv = Some((k.clone(), v.clone())); }
                layer.self_attn.kv_cache = Some((k, v));
            }
        }
        Ok(())
    }
    fn apply_linear_bridge(x: &Tensor, target_dim: usize) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?; let x_f32 = x.to_dtype(DType::F32)?;
        let rms = (x_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()).max(1e-6);
        let dynamic_bridge_scale = ((d as f64 / target_dim as f64).sqrt() * 0.7071067811865476_f64) / (rms as f64);
        if target_dim >= d {
            let upscaled = if target_dim > d { (Tensor::stack(&[x_f32.clone(), ((x_f32.clone() + x_f32.roll(1, D::Minus1)?)? * 0.5)?], D::Minus1)?.affine(dynamic_bridge_scale, 0.0))?.reshape((b, h, s, target_dim))? }
            else { x_f32.affine(dynamic_bridge_scale * (rms as f64), 0.0)? };
            Ok(upscaled.clamp(-10.0, 10.0)?.to_dtype(x.dtype())?)
        } else {
            Ok((x.narrow(D::Minus1, 0, target_dim)?.to_dtype(DType::F32)?.affine(((d as f64 / target_dim as f64).sqrt()) / (rms as f64), 0.0))?.to_dtype(x.dtype())?)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLModel {
    config: Qwen3VLConfig, visual: Option<Qwen3VLVisionModel>, pub language_model: Qwen3VLTextModel, lm_head: QLinear, rope_deltas: Option<Tensor>, pub is_baking: bool,
}

impl Qwen3VLModel {
    pub fn set_baking(&mut self, baking: bool) { self.is_baking = baking; self.language_model.is_baking = baking; }
    pub fn new(config: Qwen3VLConfig, vb: VarBuilder) -> Result<Self> { Self::new_ext(config, vb, false, false) }
    pub fn new_ext(config: Qwen3VLConfig, vb: VarBuilder, force_text_only: bool, is_baking: bool) -> Result<Self> {
        let config = config.clone();
        let probe = |v: &VarBuilder, p: &str| -> bool { match v.get_with_hints((1,), p, Init::Const(0.)) { Ok(_) => true, Err(e) => { let s = e.to_string().to_lowercase(); s.contains("shape mismatch") || s.contains("dtype mismatch") } } };
        let visual = if !force_text_only && config.vision_config.is_some() {
            let v_config = config.vision_config.as_ref().unwrap();
            let vb_v = if probe(&vb, "model.visual.patch_embed.proj.weight") || probe(&vb, "model.visual.patch_embd.weight") { Some(vb.pp("model").pp("visual")) }
            else if probe(&vb, "visual.patch_embed.proj.weight") || probe(&vb, "visual.patch_embd.weight") { Some(vb.pp("visual")) }
            else if probe(&vb, "model.v.patch_embd.weight") || probe(&vb, "model.v.blk.0.ln1.weight") { Some(vb.pp("model").pp("v")) }
            else if probe(&vb, "v.patch_embd.weight") || probe(&vb, "v.blk.0.ln1.weight") { Some(vb.pp("v")) }
            else if probe(&vb, "patch_embd.weight") || probe(&vb, "patch_embed.proj.weight") { Some(vb.clone()) } else { None };
            if let Some(vf) = vb_v { println!("[MODEL-PROBE] Vision root: {:?}", vf.prefix()); Some(Qwen3VLVisionModel::new(v_config.clone(), vf)?) } else { None }
        } else { None };
        println!("[MODEL-DEBUG] Probing Language Model Root...");
        let text_config = config.text_config.clone().ok_or(anyhow!("Missing text_config"))?;
        let (vb_lm, _) = if probe(&vb, "model.language_model.layers.0.input_layernorm.weight") { println!("[MODEL-PROBE] Deep root"); (vb.pp("model").pp("language_model"), true) }
        else if probe(&vb, "model.layers.0.input_layernorm.weight") { println!("[MODEL-PROBE] Intermediate root"); (vb.pp("model"), true) }
        else if probe(&vb, "language_model.layers.0.input_layernorm.weight") { println!("[MODEL-PROBE] language_model root"); (vb.pp("language_model"), true) }
        else if probe(&vb, "layers.0.input_layernorm.weight") { println!("[MODEL-PROBE] Flat root"); (vb.clone(), true) }
        else if probe(&vb, "model.embed_tokens.weight") { println!("[MODEL-PROBE] model root via embed_tokens"); (vb.pp("model"), true) }
        else if probe(&vb, "blk.0.attn_norm.weight") { println!("[MODEL-PROBE] GGUF root"); (vb.clone(), true) }
        else { println!("[MODEL-DEBUG] Root Probe FAILED. Fallback."); (vb.pp("model").pp("language_model"), false) };
        let language_model = Qwen3VLTextModel::new(text_config.clone(), vb_lm)?;
        if is_baking && !language_model.layers.is_empty() && language_model.layers[0].is_none() { return Err(anyhow!("Layer 0 missing in baking mode")); }
        println!("[MODEL-DEBUG] Probing LM Head...");
        let lm_head = {
            let probe_h = |v: &VarBuilder, path: &str| -> bool { for s in &["", ".packed", ".min"] { if probe(v, &format!("{}.weight{}", path, s)) { return true; } } false };
            let vh = if probe_h(&vb, "model.language_model.lm_head") { Some(vb.pp("model").pp("language_model").pp("lm_head")) }
            else if probe_h(&vb, "model.lm_head") { Some(vb.pp("model").pp("lm_head")) }
            else if probe_h(&vb, "language_model.lm_head") { Some(vb.pp("language_model").pp("lm_head")) }
            else if probe_h(&vb, "lm_head") { Some(vb.pp("lm_head")) }
            else if probe_h(&vb, "model.output") { Some(vb.pp("model").pp("output")) }
            else if probe_h(&vb, "output") { Some(vb.pp("output")) } else { None };
            if let Some(v) = vh { println!("[MODEL-PROBE] LM Head: {:?}", v.prefix()); get_qlinear_from_vb(v, "weight")? }
            else { println!("[MODEL-PROBE] lm_head missing. Dummy tied weights."); let w = language_model.embed_tokens.embeddings().clone(); let t = w.elem_count(); QLinear::new(Tensor::zeros((t / 32).max(1), DType::U32, w.device())?, Tensor::ones((t / 32).max(1), DType::F32, w.device())?, w.dims().to_vec(), None, w.device().clone()) }
        };
        let mut model = Self { config, visual, language_model, lm_head, rope_deltas: None, is_baking }; model.set_baking(is_baking); Ok(model)
    }
    pub fn save_kv_cache(&mut self, path: &std::path::Path) -> Result<()> { self.language_model.save_kv_cache(path) }
    pub fn load_kv_cache(&mut self, path: &std::path::Path, device: &Device) -> Result<()> { self.language_model.load_kv_cache(path, device) }
    fn get_vision_features(&self, vm: &Qwen3VLVisionModel, pv: &Tensor, thw: &Tensor) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (ie, dse) = vm.forward(pv, thw)?; let m = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let ss: Vec<usize> = prod_tensor_last_dim(thw)?.to_vec1::<u32>()?.iter().map(|&x| x as usize / m.pow(2)).collect(); Ok((split_tensor(&ie, &ss, 0)?, dse))
    }
    fn get_placeholder_mask(&self, ids: &Tensor, img: bool) -> Result<Tensor> {
        let tid = if img { self.config.image_token_id.unwrap_or(0) } else { self.config.video_token_id.unwrap_or(0) };
        Ok(ids.broadcast_eq(&Tensor::new(vec![tid as u32], ids.device())?)?.to_dtype(DType::U32)?)
    }
    fn get_rope_index(&self, ids: &Tensor, thw: Option<&Tensor>, vthw: Option<&Tensor>, m: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let m_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let tid = self.config.image_token_id.unwrap_or(0); let vtid = self.config.video_token_id.unwrap_or(0); let vsid = self.config.vision_start_token_id.unwrap_or(0);
        let mut deltas = vec![]; let mut pos_ids = Tensor::ones((3, ids.dim(0)?, ids.dim(1)?), ids.dtype(), ids.device())?;
        let (mut ii, mut vi) = (0, 0);
        for i in 0..ids.dim(0)? {
            let mut cur_ids = ids.i(i)?; if let Some(mask) = m { if mask.i(i)?.sum_all()?.to_scalar::<u32>()? != mask.dim(1)? as u32 { cur_ids = cur_ids.gather(&nonzero_index(&mask.i(i)?)?, 0)?; } }
            let (mut ts, mut te) = (0, 0); let mut thw_v = vec![]; let mut list: Vec<Tensor> = vec![];
            if let Ok(vidx) = get_vision_next_indices(&cur_ids, vsid as u32) {
                let vtoks = cur_ids.gather(&vidx, 0)?.to_vec1::<u32>()?; let vidx_v = vidx.to_vec1::<u32>()?;
                for (j, &t) in vtoks.iter().enumerate() {
                    if t == tid as u32 { thw_v = thw.unwrap().i(ii)?.to_vec1::<u32>()?; ii += 1; te = vidx_v[j]; }
                    if t == vtid as u32 { thw_v = vthw.as_ref().unwrap().i(vi)?.to_vec1::<u32>()?; vi += 1; te = vidx_v[j]; }
                    let (gt, gh, gw) = (thw_v[0], thw_v[1] / m_size as u32, thw_v[2] / m_size as u32);
                    let tlen = te - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 };
                    list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?);
                    let base = sidx + tlen;
                    let ti = Tensor::arange(base, base + gt, ids.device())?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, (gh * gw) as usize))?.flatten_all()?;
                    let hi = Tensor::arange(base, base + gh, ids.device())?.unsqueeze(0)?.unsqueeze(D::Minus1)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    let wi = Tensor::arange(base, base + gw, ids.device())?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((gt as usize, gh as usize, gw as usize))?.flatten_all()?;
                    list.push(Tensor::stack(&[ti, hi, wi], 0)?); ts = te + gt * gh * gw;
                }
            }
            if ts < cur_ids.dim(0)? as u32 { let tlen = cur_ids.dim(0)? as u32 - ts; let sidx = if list.is_empty() { 0 } else { list[list.len()-1].max_all()?.to_scalar::<u32>()? + 1 }; list.push(Tensor::arange(sidx, sidx + tlen, ids.device())?.unsqueeze(0)?.broadcast_as((3, tlen as usize))?); }
            let lp = Tensor::cat(&list, 1)?.reshape((3, 1, ()))?; pos_ids = pos_ids.slice_assign(&[(0..3), (i..i + 1), (0..ids.dim(1)?)], &lp)?;
            deltas.push(lp.max_all()?.to_scalar::<u32>()? as i64 + 1 - cur_ids.dim(0)? as i64);
        }
        let mut d_t = Tensor::new(deltas, ids.device())?; if d_t.rank() == 1 { d_t = d_t.unsqueeze(0)?; } Ok((pos_ids.contiguous()?, d_t))
    }
    pub fn forward(&mut self, ids: &Tensor, pv: Option<&Tensor>, thw: Option<&Tensor>, _vpv: Option<&Tensor>, vthw: Option<&Tensor>, _cp: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        println!("[TRACE-0] entry. ids: {:?}", ids.dims());
        let mut embs = self.language_model.embed_tokens.forward(ids)?;
        let (mut im, mut dse) = (None, None);
        if let Some(vm) = &self.visual {
            if let (Some(p), Some(t)) = (pv, thw) {
                let (ie, ds) = self.get_vision_features(vm, p, t)?;
                let vm_m = self.get_placeholder_mask(ids, true)?; embs = masked_scatter_dim0(&embs, &Tensor::cat(&ie, 0)?, &vm_m)?;
                im = Some(vm_m); dse = Some(ds);
            }
        }
        let (pids, _) = self.get_rope_index(ids, thw, vthw, None)?;
        println!("[TRACE-6] Calling language_model.forward...");
        let out = self.language_model.forward(&embs, offset, Some(&pids), im.as_ref(), dse)?;
        println!("[TRACE-7] Computing logits...");
        let res = self.lm_head.forward(&out.narrow(1, out.dim(1)? - 1, 1)?)?;
        println!("[TRACE-8] success."); Ok(res)
    }
    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
    pub fn device(&self) -> &Device { self.language_model.embed_tokens.embeddings().device() }
}