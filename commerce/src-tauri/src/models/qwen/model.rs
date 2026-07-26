use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{
    Activation, Embedding, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm,
};

use crate::{
    models::qwen::{
        common::{GateUpDownMLP, TwoLinearMLP, eager_attention_forward, get_layer_norm},
        config::{QwenVLConfig, QwenVLTextConfig, QwenVLVisionConfig},
        // [FIX] 올바른 최신 rope.rs 경로로 변경!
        rope::{
            Qwen2_5VisionRotaryEmbedding, QwenVLTextRotaryEmbedding, apply_rotary_pos_emb,
            apply_rotary_pos_emb_vision,
        },
    },
    utils::tensor_utils::{
        bitor_tensor, linspace, mask_index_add, masked_scatter_dim0,
        prepare_causal_attention_mask, split_tensor,
    },
};

#[derive(Debug, Clone)]
pub struct QwenVLVisionPatchEmbed {
    conv3d_weight: Tensor,
    conv3d_bias: Tensor,
}

impl QwenVLVisionPatchEmbed {
    pub fn new(cfg: &QwenVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let temporal_patch_size = cfg.temporal_patch_size;
        let in_channels = cfg.in_channels;
        // Use embed_dim if present, otherwise fallback to hidden_size
        let embed_dim = cfg.embed_dim.unwrap_or(cfg.hidden_size);
        
        // conv3d weight key: visual.patch_embed.proj.weight, value: Tensor[dims 1024, 3, 2, 16, 16; bf16, cuda:0]
        // (1024, 3, 2, 16, 16) -> (1024, 1536) -> (1536, 1024)
        let conv3d_weight = vb
            .get_with_hints(
                (
                    embed_dim,
                    in_channels,
                    temporal_patch_size,
                    patch_size,
                    patch_size,
                ),
                "proj.weight",
                Init::Const(1.),
            )?
            .flatten(1, 4)?
            .t()?;
        // (1024) -> (1, 1024)
        let conv3d_bias = vb
            .get_with_hints((embed_dim,), "proj.bias", Init::Const(0.))?
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
        // hidden_states shape:  (grid_t*grid_h*grid_w, c*temporal_patch_size*patch_size*patch_size)
        // ((), 1536) matmul (1536, 1024) -> ((), 1024)
        let hidden_states = hidden_states.matmul(&self.conv3d_weight)?;
        let hidden_states = hidden_states.broadcast_add(&self.conv3d_bias)?;
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLVisionPatchMerger {
    hidden_size: usize,
    use_postshuffle_norm: bool,
    norm: LayerNorm,
    linear_fc1: Linear,
    act_fn: Activation,
    linear_fc2: Linear,
}

impl QwenVLVisionPatchMerger {
    pub fn new(
        config: &QwenVLVisionConfig,
        vb: VarBuilder,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size * config.spatial_merge_size.pow(2);
        let norm_size = if use_postshuffle_norm {
            hidden_size
        } else {
            config.hidden_size
        };
        let norm = get_layer_norm(vb.pp("norm"), 1e-6, norm_size)?;
        let linear_fc1 = linear(hidden_size, hidden_size, vb.pp("linear_fc1"))?;
        let act_fn = Activation::Gelu;
        let linear_fc2 = linear(hidden_size, config.out_hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            hidden_size,
            use_postshuffle_norm,
            norm,
            linear_fc1,
            act_fn,
            linear_fc2,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        // [MEMORY-FIX] LayerNorm/Linear don't have to_device, so we recreate with moved tensors
        let n_w = self.norm.weight().to_device(device)?;
        let n_b = self.norm.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm = LayerNorm::new(n_w, n_b, 1e-6);
    
        let l1_w = self.linear_fc1.weight().to_device(device)?;
        let l1_b = self.linear_fc1.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear_fc1 = Linear::new(l1_w, l1_b);
    
        let l2_w = self.linear_fc2.weight().to_device(device)?;
        let l2_b = self.linear_fc2.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear_fc2 = Linear::new(l2_w, l2_b);
        Ok(())
    }
        pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.use_postshuffle_norm {
            xs.reshape(((), self.hidden_size))?
        } else {
            xs.clone()
        };
        let xs = self.norm.forward(&xs)?.reshape(((), self.hidden_size))?;
        let xs = self
            .linear_fc2
            .forward(&self.act_fn.forward(&self.linear_fc1.forward(&xs)?)?)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLVisionAttention {
    num_heads: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
}

impl QwenVLVisionAttention {
    pub fn new(config: QwenVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hidden_size / num_heads;
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            num_heads,
            qkv,
            proj,
            scaling,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let qkv_w = self.qkv.weight().to_device(device)?;
        let qkv_b = self.qkv.bias().map(|b| b.to_device(device)).transpose()?;
        self.qkv = Linear::new(qkv_w, qkv_b);

        let proj_w = self.proj.weight().to_device(device)?;
        let proj_b = self.proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.proj = Linear::new(proj_w, proj_b);
        Ok(())
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        chunks: &[usize], 
    ) -> Result<Tensor> {
        let seq_length = xs.dim(0)?;
        let qkv_states = xs.apply(&self.qkv)?.reshape((seq_length, 3, self.num_heads, ()))?.permute((1, 0, 2, 3))?; 
        
        
        let query_states = qkv_states.i(0)?.contiguous()?; 
        let key_states = qkv_states.i(1)?.contiguous()?; 
        let value_states = qkv_states.i(2)?.contiguous()?; 
        
        let (query_states, key_states) = apply_rotary_pos_emb_vision(&query_states, &key_states, cos, sin)?;
        let query_states = query_states.transpose(0, 1)?.unsqueeze(0)?;
        let key_states = key_states.transpose(0, 1)?.unsqueeze(0)?;
        let value_states = value_states.transpose(0, 1)?.unsqueeze(0)?;
        
        let q_splits = split_tensor(&query_states, chunks, 2)?;
        let k_splits = split_tensor(&key_states, chunks, 2)?;
        let v_splits = split_tensor(&value_states, chunks, 2)?;
        
        let mut attn_outputs = Vec::new();
        for (q, (k, v)) in q_splits.iter().zip(k_splits.iter().zip(v_splits.iter())) {
            let output = eager_attention_forward(q, k, v, None, None, self.scaling)?;
            attn_outputs.push(output);
        }
        
        // [CRITICAL FIX] 취합 후에도 reshape 시 metadata만 조작하므로 contiguous() 불필요!
        let attn_output = Tensor::cat(&attn_outputs, 1)?.reshape((seq_length, ()))?;
        Ok(attn_output.apply(&self.proj)?)
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLVisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: QwenVLVisionAttention,
    mlp: TwoLinearMLP,
}

impl QwenVLVisionBlock {
    pub fn new(config: QwenVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-6, config.hidden_size)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-6, config.hidden_size)?;
        let attn = QwenVLVisionAttention::new(config.clone(), vb.pp("attn"))?;
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            Activation::Gelu,
            false,
            "linear_fc1",
            "linear_fc2",
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let n1_w = self.norm1.weight().to_device(device)?;
        let n1_b = self.norm1.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm1 = LayerNorm::new(n1_w, n1_b, 1e-6);

        let n2_w = self.norm2.weight().to_device(device)?;
        let n2_b = self.norm2.bias().map(|b| b.to_device(device)).transpose()?.expect("LayerNorm bias is required");
        self.norm2 = LayerNorm::new(n2_w, n2_b, 1e-6);

        self.attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        chunks: &[usize], // [CRITICAL FIX]
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, chunks)?; // [FIX]
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        Ok(residual.add(&xs)?)
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLVisionModel {
    pub spatial_merge_size: usize,
    pub patch_embed: QwenVLVisionPatchEmbed,
    pub pos_embed: Embedding,
    pub num_grid_per_side: u32,
    pub rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
    pub blocks: Vec<QwenVLVisionBlock>,
    pub merger: QwenVLVisionPatchMerger,
    pub deepstack_visual_indexes: Vec<usize>,
    pub deepstack_merger_list: Vec<QwenVLVisionPatchMerger>,
    pub dtype: DType,
}

impl QwenVLVisionModel {
    pub fn new(config: QwenVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = config.spatial_merge_size;
        let patch_embed = QwenVLVisionPatchEmbed::new(&config, vb.pp("patch_embed"))?;
        let pos_embed = embedding(
            config.num_position_embeddings,
            config.hidden_size,
            vb.pp("pos_embed"),
        )?;
        let num_grid_per_side = (config.num_position_embeddings as f32).sqrt() as u32;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..config.depth {
            let block = QwenVLVisionBlock::new(config.clone(), vb_blocks.pp(i))?;
            blocks.push(block);
        }
        let merger = QwenVLVisionPatchMerger::new(&config, vb.pp("merger"), false)?;
        let deepstack_visual_indexes = config.deepstack_visual_indexes.clone();
        let mut deepstack_merger_list = Vec::new();
        let vb_deepstack = vb.pp("deepstack_merger_list");
        for i in 0..deepstack_visual_indexes.len() {
            let merger_i = QwenVLVisionPatchMerger::new(&config, vb_deepstack.pp(i), true)?;
            deepstack_merger_list.push(merger_i);
        }
        Ok(Self {
            spatial_merge_size,
            patch_embed,
            pos_embed,
            num_grid_per_side,
            rotary_pos_emb,
            blocks,
            merger,
            deepstack_visual_indexes,
            deepstack_merger_list,
            dtype: vb.dtype(),
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embed.to_device(device)?;
        
        let p_w = self.pos_embed.embeddings().to_device(device)?;
        self.pos_embed = Embedding::new(p_w, self.pos_embed.hidden_size());

        for block in self.blocks.iter_mut() {
            block.to_device(device)?;
        }
        self.merger.to_device(device)?;
        for merger in self.deepstack_merger_list.iter_mut() {
            merger.to_device(device)?;
        }
        Ok(())
    }

    pub fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let dev = grid_thw.device();
        let grid_thw_cpu = grid_thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        
        let side_tensor = Tensor::new(self.num_grid_per_side as f64, dev)?;
        let one_t_u32 = Tensor::new(1u32, dev)?;
        
        // [OPTIMIZATION] CPU 통신 없이 GPU 상에 텐서 목록만 모아둠
        let mut idx_tensors: [Vec<Tensor>; 4] = Default::default();
        let mut weight_tensors: [Vec<Tensor>; 4] = Default::default();
        let mut split_idx = vec![];

        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw_cpu[i][..] else { return Err(anyhow!("...")); };
            split_idx.push((h * w) as usize);
            let num_grid_per_side_sub_one = (self.num_grid_per_side - 1) as f32;
            let h_idxs = linspace(
                0.0,
                num_grid_per_side_sub_one,
                h as usize,
                grid_thw.device(),
            )?;
            let w_idxs = linspace(
                0.0,
                num_grid_per_side_sub_one,
                w as usize,
                grid_thw.device(),
            )?;
            let h_idxs_floor = h_idxs.to_dtype(candle_core::DType::U32)?;
            let w_idxs_floor = w_idxs.to_dtype(candle_core::DType::U32)?;
            let h_idxs_ceil = h_idxs_floor.broadcast_add(&one_t_u32)? 
                .clamp(0u32, num_grid_per_side_sub_one as u32)?;
            
            let w_idxs_ceil = w_idxs_floor
                .affine(1.0, 1.0)?
                .clamp(0u32, num_grid_per_side_sub_one as u32)?;
            let dh = h_idxs
                .sub(&h_idxs_floor.to_dtype(h_idxs.dtype())?)?
                .unsqueeze(D::Minus1)?;
            let dw = w_idxs
                .sub(&w_idxs_floor.to_dtype(h_idxs.dtype())?)?
                .unsqueeze(0)?;
            let base_h = h_idxs_floor.broadcast_mul(&side_tensor)?.unsqueeze(D::Minus1)?;
            let base_h_ceil = h_idxs_ceil
                .affine(self.num_grid_per_side as f64, 0.0)?
                .unsqueeze(D::Minus1)?;

            idx_tensors[0].push(base_h.broadcast_add(&w_idxs_floor.unsqueeze(0)?)?.flatten_all()?);
            idx_tensors[1].push(base_h.broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?.flatten_all()?);
            idx_tensors[2].push(base_h_ceil.broadcast_add(&w_idxs_floor.unsqueeze(0)?)?.flatten_all()?);
            idx_tensors[3].push(base_h_ceil.broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?.flatten_all()?);

            let one_sub_dh = dh.affine(-1.0, 1.0)?;
            let one_sub_dw = dw.affine(-1.0, 1.0)?;

            weight_tensors[0].push(one_sub_dh.broadcast_mul(&one_sub_dw)?.flatten_all()?);
            weight_tensors[1].push(one_sub_dh.broadcast_mul(&dw)?.flatten_all()?);
            weight_tensors[2].push(dh.broadcast_mul(&one_sub_dw)?.flatten_all()?);
            weight_tensors[3].push(dh.broadcast_mul(&dw)?.flatten_all()?);
        }

        // [OPTIMIZATION] 모든 연산이 끝난 후 한 번에 합침 (GPU 병렬성 극대화)
        let idx_tensor = Tensor::stack(&[
            Tensor::cat(&idx_tensors[0], 0)?,
            Tensor::cat(&idx_tensors[1], 0)?,
            Tensor::cat(&idx_tensors[2], 0)?,
            Tensor::cat(&idx_tensors[3], 0)?,
        ], 0)?;

        let weight_tensor = Tensor::stack(&[
            Tensor::cat(&weight_tensors[0], 0)?,
            Tensor::cat(&weight_tensors[1], 0)?,
            Tensor::cat(&weight_tensors[2], 0)?,
            Tensor::cat(&weight_tensors[3], 0)?,
        ], 0)?.to_dtype(self.dtype)?;
        let pos_embeds = self
            .pos_embed
            .forward(&idx_tensor)?
            .broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let patch_pos_embeds = pos_embeds
            .i(0)?
            .add(&pos_embeds.i(1)?)?
            .add(&pos_embeds.i(2)?)?
            .add(&pos_embeds.i(3)?)?;
        let mut patch_pos_embeds_permute = vec![];
        let patch_pos_embeds = split_tensor(&patch_pos_embeds, &split_idx, 0)?;
        let merge_size = self.spatial_merge_size;
        for (i, pos_embed) in patch_pos_embeds.iter().enumerate() {
            let [t, h, w] = grid_thw_cpu[i][..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let pos_emebd_last_dim: usize = pos_embed.dim(D::Minus1)?;
            let pos_embed = pos_embed.repeat((t as usize, 1))?;
            let shape = Shape::from(vec![
                t as usize,
                h as usize / merge_size,
                merge_size,
                w as usize / merge_size,
                merge_size,
                pos_emebd_last_dim,
            ]);
            let pos_embed = pos_embed
                .reshape(shape)?
                .permute((0, 1, 3, 2, 4, 5))?
                .flatten(0, 4)?;
            patch_pos_embeds_permute.push(pos_embed);
        }
        let patch_pos_embeds = Tensor::cat(&patch_pos_embeds_permute, 0)?;
        Ok(patch_pos_embeds)
    }

    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let merge_size = self.spatial_merge_size;
        
        // [CRITICAL FIX] 한 번만 CPU로 복사한 뒤, GPU Sync(max_all.to_scalar)를 순수 Rust 수학 연산으로 대체합니다!
        let grid_thw_cpu = grid_thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let max_hw = grid_thw_cpu.iter().flat_map(|thw| [thw[1], thw[2]]).max().unwrap_or(0);

        let freq_table = self.rotary_pos_emb.forward(max_hw as usize, grid_thw.device())?;
        let mut pos_ids_vec = vec![];
        
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw_cpu[i][..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let merged_h = h / merge_size as u32;
            let merged_w = w / merge_size as u32;
            let blocks_rows = Tensor::arange(0, merged_h, grid_thw.device())?;
            let blocks_cols = Tensor::arange(0, merged_w, grid_thw.device())?;
            let intra_row = Tensor::arange(0, merge_size as u32, grid_thw.device())?;
            let intra_col = Tensor::arange(0, merge_size as u32, grid_thw.device())?;

            // [CRITICAL FIX] 3D Grid 인덱스 계산 공식 복구 완료
            let row_idx = blocks_rows
                .unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?
                .broadcast_mul(&Tensor::new(merge_size as u32, grid_thw.device())?)?
                .broadcast_add(&intra_row.unsqueeze(0)?.unsqueeze(1)?.unsqueeze(3)?)?;

            let col_idx = blocks_cols
                .unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?
                .broadcast_mul(&Tensor::new(merge_size as u32, grid_thw.device())?)?
                .broadcast_add(&intra_col.unsqueeze(0)?.unsqueeze(1)?.unsqueeze(2)?)?;

            let row_idx = row_idx
                .expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?
                .contiguous()? 
                .flatten_all()?;
                
            let col_idx = col_idx
                .expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?
                .contiguous()? 
                .flatten_all()?;
                
            let mut coords = Tensor::stack(&[row_idx, col_idx], D::Minus1)?.contiguous()?;
            if t > 1 {
                coords = coords.repeat((t as usize, 1))?;
            }
            pos_ids_vec.push(coords);
        }
        let pos_ids = Tensor::cat(&pos_ids_vec, 0)?;

        // [CRITICAL FIX] 텐서를 잘라낸 직후 비연속 메모리를 피하기 위해 .contiguous()로 묶어줌
        let pos_ids_h = pos_ids.i((.., 0))?.contiguous()?; 
        let pos_ids_w = pos_ids.i((.., 1))?.contiguous()?; 
        
        let rotary_pos_emb_h = freq_table.index_select(&pos_ids_h, 0)?;
        let rotary_pos_emb_w = freq_table.index_select(&pos_ids_w, 0)?;
        
        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?;
        Ok(rotary_pos_emb)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let hidden_states = self.patch_embed.forward(hidden_states)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let hidden_states = hidden_states.broadcast_add(&pos_embeds)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(self.dtype)?;
        let sin = emb.sin()?.to_dtype(self.dtype)?;
        
        // [CRITICAL FIX] 수십 개의 GPU 커널과 하드 동기화(to_vec1)를 유발하던 로직을 순수 CPU 수학으로 100% 압축!
        let grid_thw_cpu = grid_thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let mut chunks = Vec::new();
        
        for thw in grid_thw_cpu {
            let (t, h, w) = (thw[0] as usize, thw[1] as usize, thw[2] as usize);
            let hw = h * w;
            for _ in 0..t {
                chunks.push(hw);
            }
        }

        let mut deepstack_feature_lists = vec![];
        for (layer_num, block) in self.blocks.iter().enumerate() {
            // 미리 계산된 chunks 슬라이스 전달
            hidden_states = block.forward(&hidden_states, &chunks, &cos, &sin)?;
            if self.deepstack_visual_indexes.contains(&layer_num) {
                if let Some(index) = self
                    .deepstack_visual_indexes
                    .iter()
                    .position(|&x| x == layer_num)
                {
                    let deepstack_feature =
                        self.deepstack_merger_list[index].forward(&hidden_states)?;
                    deepstack_feature_lists.push(deepstack_feature);
                }
            }
        }
        hidden_states = self.merger.forward(&hidden_states)?;
        Ok((hidden_states, deepstack_feature_lists))
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLTextAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f64,
    pub kv_cache: Option<(Tensor, Tensor)>,
}

impl QwenVLTextAttention {
    pub fn new(config: QwenVLTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let (q_proj, k_proj, v_proj, o_proj) = if config.attention_bias {
            let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
            let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
            let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
            let o_proj = linear(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
            (q_proj, k_proj, v_proj, o_proj)
        } else {
            let q_proj =
                linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
            let k_proj =
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
            let v_proj =
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
            let o_proj =
                linear_no_bias(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
            (q_proj, k_proj, v_proj, o_proj)
        };
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
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
                let target_dtype = if xs.device().is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };
                let prev_k_res = prev_k.to_dtype(target_dtype).unwrap_or_else(|_| prev_k.clone());
                let prev_v_res = prev_v.to_dtype(target_dtype).unwrap_or_else(|_| prev_v.clone());
                let key_states = Tensor::cat(&[&prev_k_res, &key_states], 2)?;
                let value_states = Tensor::cat(&[&prev_v_res, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some(if xs.device().is_cuda() { 
            (key_states.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| key_states.clone()), 
             value_states.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| value_states.clone())) 
        } else { 
            (key_states.clone(), value_states.clone()) 
        });
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLTextDecoderLayer {
    pub self_attn: QwenVLTextAttention,
    pub mlp: GateUpDownMLP,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl QwenVLTextDecoderLayer {
    pub fn new(config: QwenVLTextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = QwenVLTextAttention::new(config.clone(), vb.pp("self_attn"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
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
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLTextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<QwenVLTextDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: QwenVLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
}

impl QwenVLTextModel {
    pub fn new(config: QwenVLTextConfig, vb: VarBuilder) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let embed_tokens = embedding(vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = QwenVLTextDecoderLayer::new(config.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let head_dim = config.head_dim;
        let rotary_emb = QwenVLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        // [FIX] rope_scaling optional 처리
        let mrope_section = config.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
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
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref() {
                if layer_idx < deepstack_embeds.len() {
                    xs = mask_index_add(
                        &xs.squeeze(0)?,
                        &visual_pos_masks.unwrap().squeeze(0)?,
                        &deepstack_embeds[layer_idx],
                    )?
                    .unsqueeze(0)?;
                }
            }
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

#[derive(Debug, Clone)]
pub struct QwenVLModel {
    pub config: QwenVLConfig,
    visual: QwenVLVisionModel,
    pub language_model: QwenVLTextModel, // 🌟 [CRITICAL FIX] pub을 추가하여 외부에서 임베딩 접근을 허용합니다.
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl QwenVLModel {
    pub fn new(config: QwenVLConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let config = config.clone();
        let v_config = config.vision_config.clone().ok_or(anyhow!("Missing vision_config for QwenVLModel"))?;
        let visual = QwenVLVisionModel::new(v_config, vb_m.pp("visual"))?;
        
        let text_config = config.text_config.clone().ok_or(anyhow!("Missing text_config for QwenVLModel"))?;
        
        let language_model =
            QwenVLTextModel::new(text_config.clone(), vb_m.pp("language_model"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                text_config.hidden_size,
                text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            config,
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
    ) -> Result<(Tensor, Vec<Tensor>)> { 
        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(pixel_values, image_grid_thw)?;
            
        // [CRITICAL FIX] 쪼개는 로직 삭제하고 원본 텐서를 그대로 반환!
        Ok((image_embeds, deepstack_image_embeds))
    }

    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token_id = if is_image {
            self.config.image_token_id.unwrap_or(0) as u32
        } else {
            self.config.video_token_id.unwrap_or(0) as u32
        };
        let special_token = Tensor::new(vec![special_token_id], input_ids.device())?;
        let special_mask = input_ids
            .broadcast_eq(&special_token)?
            .to_dtype(candle_core::DType::U32)?;
        Ok(special_mask)
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // [ROPE-FIX] 이미지 격자 구조를 반영한 실제 3D mRoPE 인덱스 계산 로직
        let spatial_merge_size = self.config.vision_config.as_ref().map(|c| c.spatial_merge_size).unwrap_or(2);
        let image_token_id = self.config.image_token_id.unwrap_or(0);
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(0);
        
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut mrope_position_deltas = Vec::new();

        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        let image_thw_cpu = if let Some(thw) = image_grid_thw { Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?) } else { None };
        
        // [CRITICAL FIX] 루프 밖에서 전체 데이터를 담을 그릇을 미리 할당합니다.
        let mut flat_pos_ids = Vec::with_capacity(3 * b_sz * seq_len);

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                if ids[i] == vision_start_token_id as u32 && i + 1 < seq_len && ids[i+1] == image_token_id as u32 {
                    // 이미지 영역 발견
                    if let Some(thw_cpu_array) = &image_thw_cpu {
                        // [수정] GPU Sync 없이 CPU 배열에서 즉시 읽어옴!
                        let thw = &thw_cpu_array[image_idx];
                        image_idx += 1;
                        
                        let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                        
                        // vision_start 토큰 위치
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;

                        // image_pad 토큰들 위치 (3D Grid)
                        let img_len = (t * h * w) as usize;
                        for tt in 0..t {
                            for hh in 0..h {
                                for ww in 0..w {
                                    let idx = i + (tt * h * w + hh * w + ww) as usize;
                                    if idx < seq_len {
                                        llm_pos_ids[0][idx] = curr_pos + tt;
                                        llm_pos_ids[1][idx] = curr_pos + hh;
                                        llm_pos_ids[2][idx] = curr_pos + ww;
                                    }
                                }
                            }
                        }
                
                        i += img_len;
                        curr_pos += t.max(h).max(w);
                    } else {
                        // fallback (thw 없을 때)
                        for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                        i += 1;
                        curr_pos += 1;
                    }
                } else {
                    // 일반 텍스트 토큰
                    for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                    i += 1;
                    curr_pos += 1;
                }
            }
            
            // Tensor로 변환 및 삽입
            for d in 0..3 {
                flat_pos_ids.extend_from_slice(&llm_pos_ids[d]);
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        }

        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), input_ids.device())?;
        let deltas = Tensor::from_vec(mrope_position_deltas, (b_sz, 1), input_ids.device())?.to_dtype(input_ids.dtype())?;
        Ok((position_ids, deltas))
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        let mut image_mask = None;
        let mut video_mask = None;
        let mut deepstack_image_embeds = None;
        let mut deepstack_video_embeds = None;
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                let (image_embeds, deepstack_img_embed) =
                    self.get_vision_features(pixel_values, image_grid_thw)?;
                let vision_mask = self.get_placeholder_mask(input_ids, true)?;
                
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
                image_mask = Some(vision_mask);
                deepstack_image_embeds = Some(deepstack_img_embed);
            }
        }
        if let Some(pixel_values_video) = pixel_values_video {
            if let Some(video_grid_thw) = video_grid_thw {
                let (video_embeds, deepstack_video_embed) =
                    self.get_vision_features(pixel_values_video, video_grid_thw)?;
                let vision_mask = self.get_placeholder_mask(input_ids, false)?;
                
                inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
                video_mask = Some(vision_mask);
                deepstack_video_embeds = Some(deepstack_video_embed);
            }
        }
        let mut visual_pos_mask = None;
        let mut deepstack_visual_embeds = None;
        if let Some(image_mask_) = image_mask {
            if let Some(video_mask_) = video_mask {
                let image_mask_ = image_mask_.squeeze(0)?;
                let video_mask_ = video_mask_.squeeze(0)?;
                let _visual_mask = bitor_tensor(&image_mask_, &video_mask_)?;
                
                // [CRITICAL FIX] 3연속 GPU Sync Stall을 1번의 CPU 스캔으로 통합 압축
                let img_mask_vec = image_mask_.to_vec1::<u32>()?;
                let vid_mask_vec = video_mask_.to_vec1::<u32>()?;

                let mut visual_indices = Vec::new();
                let mut image_joint_indices = Vec::new();
                let mut video_joint_indices = Vec::new();

                let mut visual_counter = 0;
                for i in 0..img_mask_vec.len() {
                    let is_img = img_mask_vec[i] > 0;
                    let is_vid = vid_mask_vec[i] > 0;

                    if is_img || is_vid {
                        visual_indices.push(i as u32);
                        if is_img { image_joint_indices.push(visual_counter as u32); }
                        if is_vid { video_joint_indices.push(visual_counter as u32); }
                        visual_counter += 1;
                    }
                }

                let dev = image_mask_.device();
                let visual_none_zero_index = Tensor::from_vec(visual_indices.clone(), visual_indices.len(), dev)?;
                let image_nonzero_joint = Tensor::from_vec(image_joint_indices.clone(), image_joint_indices.len(), dev)?;
                let video_nonzero_joint = Tensor::from_vec(video_joint_indices.clone(), video_joint_indices.len(), dev)?;
                let mut deepstack_embeds = vec![];
                let visual_len = visual_none_zero_index.dim(0)?;
                for (img_embed, vid_embed) in deepstack_image_embeds
                    .unwrap()
                    .iter()
                    .zip(deepstack_video_embeds.unwrap().iter())
                {
                    let embed_joint = Tensor::zeros(
                        (visual_len, img_embed.dim(D::Minus1)?),
                        img_embed.dtype(),
                        img_embed.device(),
                    )?;
                    let embed_joint = embed_joint.index_add(&image_nonzero_joint, img_embed, 0)?;
                    let embed_joint = embed_joint.index_add(&video_nonzero_joint, vid_embed, 0)?;
                    deepstack_embeds.push(embed_joint);
                }
                visual_pos_mask = Some(visual_none_zero_index.clone());
                deepstack_visual_embeds = Some(deepstack_embeds);
            } else {
                visual_pos_mask = Some(image_mask_);
                deepstack_visual_embeds = deepstack_image_embeds;
            }
        } else if let Some(video_mask_) = video_mask {
            visual_pos_mask = Some(video_mask_);
            deepstack_visual_embeds = deepstack_video_embeds;
        }

        let position_ids;
        let rope_deltas;
        if seqlen_offset == 0 || self.rope_deltas.is_none() { 
            (position_ids, rope_deltas) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if let Some(cache_position) = cache_position {
                cache_position
                    .i(0)?
                    .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
                    .broadcast_add(self.rope_deltas.as_ref().unwrap())?
                    // [FIX] contiguous() 삭제로 인한 PCIe 대역폭 확보
                    .to_dtype(candle_core::DType::U32)?
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?;
        }
        let outputs = self.language_model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids),
            visual_pos_mask.as_ref(),
            deepstack_visual_embeds,
        )?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    pub fn device(&self) -> &Device {
        self.lm_head.weight().device()
    }
}