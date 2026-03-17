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
        // qwen3vl 쪽의 공용 KV 캐시 구조체 및 RmsNorm, QLinear를 그대로 사용합니다.
        qwen3vl::quantized_model::{
            RmsNorm, QLinear, KVRegistry, KVBlock, KVLocation, RegistryEntry, MemorySlot, 
            KVBlockInner, BitKVMetadata, get_qlinear, get_rms_norm
        },
        // qwen3_5 내부의 최적화된 rope.rs 참조
        qwen3_5::rope::{
            Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
        },
    },
    utils::tensor_utils::{
        masked_scatter_dim0, split_tensor,
    },
};

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
    pub active_kv_name: Option<String>,
    pub active_session_id: Option<String>,
    pub vram_merged_k: Option<Tensor>,
    pub vram_merged_v: Option<Tensor>,
    pub merged_vram_block_count: usize,
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
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads.max(1),
            head_dim,
            scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_kv_name: None,
            active_session_id: None,
            vram_merged_k: None,
            vram_merged_v: None,
            merged_vram_block_count: 0,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if self.q_proj.is_cleared() { return Err(anyhow!("Attention weights cleared.")); }
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.o_proj.to_device(device)?;
        self.q_norm.to_device(device)?;
        self.k_norm.to_device(device)?;
        
        self.vram_merged_k = None;
        self.vram_merged_v = None;
        self.merged_vram_block_count = 0;

        let target_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        for block in &mut self.kv_blocks {
            let (index, mut inner) = {
                let inner = block.inner.write().unwrap();
                (inner.index, inner)
            };
            let loc = {
                let reg = self.registry.entries.read().unwrap();
                if index < reg.len() { reg[index].location[self.layer_idx] } else { KVLocation::VRAM }
            };
            if loc == KVLocation::VRAM {
                if let Some(k) = &inner.k_cache { inner.k_cache = Some(k.to_device(device)?.to_dtype(target_dtype)?); }
                if let Some(v) = &inner.v_cache { inner.v_cache = Some(v.to_device(device)?.to_dtype(target_dtype)?); }
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.q_proj.clear(); self.k_proj.clear(); self.v_proj.clear(); self.o_proj.clear();
        self.q_norm.clear(); self.k_norm.clear();
        self.vram_merged_k = None; self.vram_merged_v = None; self.merged_vram_block_count = 0;
    }

    pub fn load_weights_inplace<R: std::io::Seek + std::io::Read>(
        &mut self, ct: &gguf_file::Content, reader: &mut R, base_name: &str, device: &Device, dtype: DType,
    ) -> Result<()> {
        self.q_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.q_proj"), device, dtype)?;
        self.k_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.k_proj"), device, dtype)?;
        self.v_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.v_proj"), device, dtype)?;
        self.o_proj = get_qlinear(ct, reader, &format!("{base_name}.self_attn.o_proj"), device, dtype)?;
        self.q_norm = get_rms_norm(ct, reader, &format!("{base_name}.self_attn.q_norm"), self.q_norm.eps(), device, dtype)?;
        self.k_norm = get_rms_norm(ct, reader, &format!("{base_name}.self_attn.k_norm"), self.k_norm.eps(), device, dtype)?;
        Ok(())
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask_in: Option<&Tensor>,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
        baking_only: bool,
    ) -> Result<Tensor> {
        self.active_session_id = session_id.clone();
        self.active_kv_name = kv_name;

        let dev = self.q_proj.device();
        let target_dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };
        let (b_sz, q_len, _) = xs.dims3()?;

        let total_len = seqlen_offset + q_len;
        let attention_mask = if q_len > 1 && attention_mask_in.is_none() {
            let q_indices = Tensor::arange(0u32, q_len as u32, dev)?.unsqueeze(1)?;
            let k_indices = Tensor::arange(0u32, total_len as u32, dev)?.unsqueeze(0)?;
            let mask = k_indices.broadcast_gt(&(q_indices.broadcast_add(&Tensor::new(seqlen_offset as u32, dev)?)?))?;
            let mask = mask.to_dtype(target_dtype)?.affine(-1e4, 0.0)?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            attention_mask_in.cloned()
        };

        // [CRITICAL FIX] .contiguous() 완전히 제거
        let mut query_states = self.q_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?; 
        
        let mut key_states = self.k_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;     
        
        let value_states = self.v_proj.forward(&xs)?.reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;

        let cos = cos.to_dtype(target_dtype)?;
        let sin = sin.to_dtype(target_dtype)?;
        
        // 올바른 Qwen3.5 RoPE 호출
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, &cos, &sin, false)?;
        let query_states = query_states.to_dtype(target_dtype)?;
        let key_states = key_states.to_dtype(target_dtype)?;

        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 256usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    let k_piece = key_states.narrow(2, chunk_offset, take)?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?;

                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };
                        inner.k_cache = Some(Tensor::cat(&[&pk, &k_piece], 2)?);
                        inner.v_cache = Some(Tensor::cat(&[&pv, &v_piece], 2)?);
                        inner.len += take; tokens_to_process -= take; chunk_offset += take;
                        appended = true;
                        
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            let entry = &mut reg[inner.index];
                            entry.token_len = inner.len;
                            if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = true; }
                        }
                    }
                }
            }
            if !appended {
                let take = tokens_to_process.min(256);
                let k_piece = key_states.narrow(2, chunk_offset, take)?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?;
                
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece); inner.v_cache = Some(v_piece);
                }
                
                let mut reg = self.registry.entries.write().unwrap();
                if index < reg.len() {
                    let entry = &mut reg[index];
                    entry.token_start = current_total;
                    entry.token_len = take;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = true; }
                    if self.layer_idx < entry.location.len() { entry.location[self.layer_idx] = KVLocation::VRAM; }
                }
                self.kv_blocks.push(new_block);
                tokens_to_process -= take; chunk_offset += take;
            }
        }

        // [CHUNK-BASED ONLINE SOFTMAX] Zero-VRAM Spikes Attention
        let total_tokens_now = seqlen_offset + q_len;
        let mut out_res: Option<Tensor> = None;
        let mut m_n: Option<Tensor> = None;
        let mut l_n: Option<Tensor> = None;
        let q_aligned = query_states.to_dtype(target_dtype)?;

        for block in &self.kv_blocks {
            let (index, b_off, _b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            // [STEP A] RAM/SSD에서 VRAM으로 불러오기 로직 (qwen3vl과 동일)
            let (k_block, v_block, is_temporary) = {
                let inner = block.inner.read().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    (k.to_device(dev)?.to_dtype(target_dtype)?, v.to_device(dev)?.to_dtype(target_dtype)?, false)
                } else {
                    let mut k_cpu = None;
                    let mut v_cpu = None;
                    {
                        let reg = self.registry.entries.read().unwrap();
                        let cache = reg[index].bitkv_cache.read().unwrap();
                        if let Some(m) = &cache[self.layer_idx] {
                            k_cpu = Some(m.k_data.to_device(&Device::Cpu)?);
                            v_cpu = Some(m.v_data.to_device(&Device::Cpu)?);
                        }
                    }

                    if k_cpu.is_none() {
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let sid = self.active_session_id.as_deref().unwrap_or("general");
                        let mut path_candidates = Vec::new();
                        
                        if let Some(p) = { let reg = self.registry.entries.read().unwrap();
                        if index < reg.len() { reg[index].ssd_path.clone() } else { None } } {
                            path_candidates.push(p);
                        }
                        
                        let kv_name_raw = self.active_kv_name.as_deref().unwrap_or("text");
                        let kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        let kv_type = if kv_type == "inference" || kv_type == "reference" || kv_type.is_empty() { "text" } else { kv_type };
                        
                        path_candidates.push(kv_dir.join(format!("{}/inference/{}/b{}", sid, kv_type, b_off)));
                        path_candidates.push(kv_dir.join(format!("{}/reference/{}/b{}", sid, kv_type, b_off)));
                        
                        for full_path in &path_candidates {
                            let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                            if block_file.exists() {
                                if let Ok(data) = crate::utils::direct_loader::load_kv_block(&block_file) {
                                    if let Ok(st) = safetensors::SafeTensors::deserialize(&data) {
                                        let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                        let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                        
                                        if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                            let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                            let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                            
                                            let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap_or_else(|_| Tensor::zeros(meta_os.clone(), DType::BF16, &Device::Cpu).unwrap());
                                            let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &meta_os, &Device::Cpu).unwrap_or_else(|_| Tensor::zeros(meta_os.clone(), DType::BF16, &Device::Cpu).unwrap());
                                            
                                            k_cpu = Some(kd_t.clone());
                                            v_cpu = Some(vd_t.clone());

                                            let mut reg = self.registry.entries.write().unwrap();
                                            if index < reg.len() { 
                                                reg[index].ssd_path = Some(full_path.clone());
                                                reg[index].location[self.layer_idx] = KVLocation::RAM; 
                                                
                                                let mut cache = reg[index].bitkv_cache.write().unwrap();
                                                cache[self.layer_idx] = Some(BitKVMetadata {
                                                    k_data: kd_t,
                                                    v_data: vd_t,
                                                    original_shape: meta_os,
                                                });
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let k_final = k_cpu.ok_or_else(|| anyhow!("Block missing"))?.to_device(dev)?.to_dtype(target_dtype)?;
                    let v_final = v_cpu.ok_or_else(|| anyhow!("Block missing"))?.to_device(dev)?.to_dtype(target_dtype)?;
                    (k_final, v_final, true)
                }
            };

            let mut k = k_block;
            let mut v = v_block;

            if self.num_kv_groups > 1 {
                let (b, h, s, d) = k.dims4()?;
                k = k.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
                v = v.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))?;
            }

            let actual_kv_len = k.dim(2)?;
            let mut s_chunk = (q_aligned.matmul(&k.transpose(2, 3)?)? * self.scaling)?;
            
            if let Some(mask) = &attention_mask {
                let mask_len = mask.dim(candle_core::D::Minus1)?;
                if b_off < mask_len {
                    let take = std::cmp::min(actual_kv_len, mask_len - b_off);
                    let chunk_mask = mask.narrow(candle_core::D::Minus1, b_off, take)?;
                    
                    if take < actual_kv_len {
                        let left_masked = s_chunk.narrow(candle_core::D::Minus1, 0, take)?.broadcast_add(&chunk_mask)?;
                        let right_unmasked = s_chunk.narrow(candle_core::D::Minus1, take, actual_kv_len - take)?;
                        s_chunk = Tensor::cat(&[&left_masked, &right_unmasked], candle_core::D::Minus1)?;
                    } else {
                        s_chunk = s_chunk.broadcast_add(&chunk_mask)?;
                    }
                }
            }

            let s_chunk_f32 = s_chunk.to_dtype(DType::F32)?;
            let m_j = s_chunk_f32.max_keepdim(candle_core::D::Minus1)?;
            let p_j = s_chunk_f32.broadcast_sub(&m_j)?.exp()?;
            let l_j = p_j.sum_keepdim(candle_core::D::Minus1)?;
            
            let out_j = p_j.to_dtype(v.dtype())?.matmul(&v)?;
            let out_j_f32 = out_j.to_dtype(DType::F32)?;

            match out_res {
                None => {
                    out_res = Some(out_j_f32);
                    m_n = Some(m_j);
                    l_n = Some(l_j);
                }
                Some(prev_out_f32) => {
                    let prev_m = m_n.as_ref().unwrap();
                    let prev_l = l_n.as_ref().unwrap();
                    
                    let m_new = prev_m.maximum(&m_j)?;
                    let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
                    let diff_new = m_j.broadcast_sub(&m_new)?.exp()?;
                    
                    let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
                    let out_new_f32 = prev_out_f32.broadcast_mul(&diff_old)?.add(&out_j_f32.broadcast_mul(&diff_new)?)?;
                    
                    out_res = Some(out_new_f32);
                    m_n = Some(m_new);
                    l_n = Some(l_new);
                }
            }
            drop(k); drop(v);
        }

        let attn_output = if let (Some(out_f32), Some(l_f32)) = (out_res, l_n) {
            out_f32.broadcast_div(&l_f32)?.to_dtype(target_dtype)?
        } else {
            return Err(anyhow!("No KV data processed"));
        };

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        Ok(attn_output)
    }

    pub fn get_kv_len(&self) -> usize {
        self.kv_blocks.last().map(|b| {
            let inner = b.inner.read().unwrap();
            inner.offset + inner.len
        }).unwrap_or(0)
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        let dev = &Device::Cpu;
        let target_dtype = DType::BF16;

        if let (Some(mk), Some(mv)) = (&self.vram_merged_k, &self.vram_merged_v) {
            let mk_cpu = mk.to_device(dev)?.to_dtype(target_dtype)?;
            let mv_cpu = mv.to_device(dev)?.to_dtype(target_dtype)?;
            
            let mut current_pos = 0;
            for block in &mut self.kv_blocks {
                let mut inner = block.inner.write().unwrap();
                let b_len = inner.len;
                if inner.location == KVLocation::VRAM {
                    let k_part = mk_cpu.narrow(2, current_pos, b_len)?;
                    let v_part = mv_cpu.narrow(2, current_pos, b_len)?;
                    inner.k_cache = Some(k_part);
                    inner.v_cache = Some(v_part);
                    inner.location = KVLocation::RAM;
                    let mut reg = self.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[self.layer_idx] = KVLocation::RAM;
                    }
                }
                current_pos += b_len;
            }
            self.vram_merged_k = None;
            self.vram_merged_v = None;
            self.merged_vram_block_count = 0;
        }

        for block in &mut self.kv_blocks {
            let (k_to_move, v_to_move) = {
                let inner = block.inner.read().unwrap();
                if inner.location == KVLocation::VRAM && inner.k_cache.is_some() {
                    (inner.k_cache.clone(), inner.v_cache.clone())
                } else { (None, None) }
            };
            if let (Some(k), Some(v)) = (k_to_move, v_to_move) {
                let mut inner = block.inner.write().unwrap();
                inner.k_cache = Some(k.to_device(dev)?.to_dtype(target_dtype)?);
                inner.v_cache = Some(v.to_device(dev)?.to_dtype(target_dtype)?);
                inner.location = KVLocation::RAM;
                
                let mut reg = self.registry.entries.write().unwrap();
                if inner.index < reg.len() {
                    reg[inner.index].location[self.layer_idx] = KVLocation::RAM;
                }
            }
        }
        Ok(())
    }

    pub fn trigger_realtime_incremental_bake(&self, session_id: &str, is_last_chunk: bool, baking_only: bool, is_decoding: bool) -> Result<()> {
        use crate::models::qwen3vl::generate::{BakeTask, SlotTask, BAKE_TX, SLOT_MANAGER, LayerKVDump};
        
        let target_indices: Vec<usize> = self.kv_blocks.iter().enumerate().filter_map(|(i, b)| {
            let inner = b.inner.read().unwrap();
            let is_full = inner.len == 256;
            
            let is_dirty = {
                let reg = self.registry.entries.read().unwrap();
                if i < reg.len() { 
                    if self.layer_idx < reg[i].is_dirty.len() { reg[i].is_dirty[self.layer_idx] } else { true }
                } else { true }
            };

            if (is_full || is_last_chunk) && inner.k_cache.is_some() && is_dirty { Some(i) } else { None }
        }).collect();

        for idx in target_indices {
            let block = self.kv_blocks[idx].clone();
            
            {
                let mut reg = self.registry.entries.write().unwrap();
                if idx < reg.len() && self.layer_idx < reg[idx].is_dirty.len() {
                    reg[idx].is_dirty[self.layer_idx] = false;
                }
            }

            let (k_opt, v_opt, off, b_idx, b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.k_cache.clone(), inner.v_cache.clone(), inner.offset, inner.index, inner.len)
            };

            if let (Some(k), Some(v)) = (k_opt, v_opt) {
                let kv_name_raw = self.active_kv_name.clone().unwrap_or_else(|| "text".to_string());
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text".to_string() } else { last_part.to_string() };
                let session_id_owned = session_id.to_string();
                let registry_clone = self.registry.clone();
                let layer_idx = self.layer_idx;
                let num_kv_h = self.num_key_value_heads;
                let h_d = self.head_dim;
                
                crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tauri::async_runtime::spawn(async move {
                    let (k_vram, v_vram) = tokio::task::spawn_blocking(move || {
                        let k_res = k.to_device(&Device::Cpu).unwrap_or_else(|_| k.clone()).to_dtype(DType::BF16).unwrap_or_else(|_| k.clone());
                        let v_res = v.to_device(&Device::Cpu).unwrap_or_else(|_| v.clone()).to_dtype(DType::BF16).unwrap_or_else(|_| v.clone());
                        (k_res, v_res)
                    }).await.unwrap_or_else(|_| (Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(), Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap()));
                     
                    if let Some(tx) = crate::models::qwen3vl::generate::BAKE_TX.get() {
                        let sub_path = if baking_only { format!("{}/reference/{}", session_id_owned, kv_type) } else { format!("{}/inference/{}", session_id_owned, kv_type) };
                        let kv_dir = crate::utils::paths::get_kv_dir(None);
                        let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                        let k_shape_u32 = vec![1u32, num_kv_h as u32, b_len as u32, h_d as u32];
                        let dump = crate::models::qwen3vl::generate::LayerKVDump {
                            layer_idx,
                            k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu).unwrap(),
                            k_shape: Tensor::from_vec(k_shape_u32, (4,), &Device::Cpu).unwrap(),
                            raw_k: Some(k_vram),
                            raw_v: Some(v_vram),
                        };
                        let sid = crate::models::qwen3vl::generate::SLOT_MANAGER.acquire_write_slot(b_len).await;
                        if tx.send(crate::models::qwen3vl::generate::SlotTask::Bake(crate::models::qwen3vl::generate::BakeTask {
                            slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path), offset: off, layers: vec![dump],
                            is_relay_baking: baking_only, block_idx: Some(b_idx), registry: registry_clone,
                        })).await.is_err() {
                             crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                             crate::models::qwen3vl::generate::SLOT_MANAGER.release_slot(sid).await;
                        }
                    } else {
                        crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        }
        Ok(())
    }

    pub fn compress_to_bf16(&self, t: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let original_shape = t.shape().dims().to_vec();
        let t_bf16 = t.to_device(&Device::Cpu)?.to_dtype(DType::BF16)?;
        Ok((t_bf16, original_shape))
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        let kv_type = kv_name.unwrap_or("text");
        let b_str = format!("b{}", offset);
        let block_dir = path.join(kv_type).join(&b_str);
        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }
        
        let structured_path = block_dir.join(format!("l{}.st", self.layer_idx));
        let mut map = std::collections::HashMap::new();
        let prefix = format!("b{}_l{}_", offset, self.layer_idx);
        
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                ks.push(k.clone());
                vs.push(v.clone());
            }
        }

        if !ks.is_empty() {
            let k = Tensor::cat(&ks, 2)?;
            let v = Tensor::cat(&vs, 2)?;
            
            let (kd, k_shape) = self.compress_to_bf16(&k)?;
            let (vd, _) = self.compress_to_bf16(&v)?;
            
            map.insert(format!("{}k_data", prefix), kd);
            map.insert(format!("{}v_data", prefix), vd);
            map.insert(format!("{}k_shape", prefix), Tensor::from_vec(k_shape.iter().map(|&x| x as u32).collect::<Vec<u32>>(), (k_shape.len(),), &Device::Cpu)?);
            
            if let Ok(data) = safetensors::serialize(&map, &None) {
                let _ = crate::utils::direct_loader::save_kv_block(&structured_path, &data);
            }
            
            if let Ok(mut reg) = self.registry.entries.write() {
                let entry_idx = offset / 256;
                if entry_idx < reg.len() {
                    let entry = &mut reg[entry_idx];
                    entry.ssd_path = Some(path.to_path_buf());
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = false; }
                } else {
                    let mut entry = RegistryEntry::new(offset, k.dim(2)?, 28);
                    entry.ssd_path = Some(path.to_path_buf());
                    entry.location[self.layer_idx] = KVLocation::SSD;
                    if self.layer_idx < entry.is_dirty.len() { entry.is_dirty[self.layer_idx] = false; }
                    reg.push(entry);
                }
            }
        }

        if clear { self.kv_blocks.clear(); }
        Ok(())
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        let mut _current_total = 0;
        let mut to_remove = Vec::new();
        let total_blocks = self.kv_blocks.len();
        
        for i in 0..total_blocks {
            let block = &mut self.kv_blocks[i];
            let mut inner = block.inner.write().unwrap();
            
            if _current_total + inner.len <= len {
                _current_total += inner.len;
            } else {
                let keep_in_this_block = len - _current_total;
                if keep_in_this_block > 0 {
                    if inner.location == KVLocation::VRAM {
                        let (new_k, new_v) = if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            (Some(k.narrow(2, 0, keep_in_this_block)?), Some(v.narrow(2, 0, keep_in_this_block)?))
                        } else { (None, None) };
                        inner.k_cache = new_k;
                        inner.v_cache = new_v;
                    }
                    inner.len = keep_in_this_block;
                    _current_total += keep_in_this_block;
                    for j in (i + 1)..total_blocks { to_remove.push(j); }
                } else {
                    for j in i..total_blocks { to_remove.push(j); }
                }
                break;
            }
        }
        
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove { self.kv_blocks.remove(idx); }
        Ok(())
    }

    pub fn load_kv_cache(&mut self, _path: &Path, _device: &Device, _expected_len: usize, _upscale_refill_len: usize, _kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        if fragments.is_empty() { return Ok(()); }
        
        self.kv_blocks.clear();
        let mut total_restored_len = 0;

        for (i, (offset, frag_path)) in fragments.iter().enumerate() {
            let b_len = if *offset < current_kv_len {
                (current_kv_len - *offset).min(256)
            } else { 256 };
            total_restored_len += b_len;
            
            let new_block = KVBlock::new(KVLocation::SSD, i, b_len, *offset);
            {
                let mut inner = new_block.inner.write().unwrap();
                inner.len = b_len;
                inner.location = KVLocation::SSD;
            }
            self.kv_blocks.push(new_block);
            
            let mut reg = self.registry.entries.write().unwrap();
            if i >= reg.len() {
                reg.push(RegistryEntry {
                    location: vec![KVLocation::SSD; 28],
                    slot_ids: vec![None; 28],
                    token_start: *offset,
                    token_len: b_len,
                    ssd_path: Some(frag_path.parent().unwrap().to_path_buf()),
                    hidden_states_path: vec![None; 28],
                    is_dirty: vec![false; 28], 
                    last_accessed: std::time::Instant::now(),
                    bitkv_cache: Arc::new(std::sync::RwLock::new(vec![None; 28])),
                });
            }
            
            if i < reg.len() {
                reg[i].location[self.layer_idx] = KVLocation::SSD;
                reg[i].ssd_path = Some(frag_path.parent().unwrap().to_path_buf());
            }
        }
        Ok(())
    }
}

// Qwen 3.5 Linear Attention/DeltaNet 지원 뼈대
#[derive(Clone)]
pub struct QuantizedQwen3_5GatedDeltaNet {
    pub in_proj_qkv: QLinear,
    pub in_proj_z: QLinear,
    pub in_proj_b: QLinear,
    pub in_proj_a: QLinear,
    pub out_proj: QLinear,
}

impl QuantizedQwen3_5GatedDeltaNet {
    pub fn to_device(&mut self, _device: &Device) -> Result<()> { Ok(()) }
    pub fn clear(&mut self) {}
    pub fn load_weights_inplace<R: std::io::Seek + std::io::Read>(&mut self, _ct: &gguf_file::Content, _reader: &mut R, _base_name: &str, _device: &Device, _dtype: DType) -> Result<()> { Ok(()) }
    pub fn forward(&mut self, xs: &Tensor, _seqlen_offset: usize) -> Result<Tensor> { Ok(xs.clone()) } 
}

#[derive(Clone)]
pub struct QuantizedQwen3_5DecoderLayer {
    pub layer_type: String,
    pub self_attn: Option<QuantizedQwen3_5Attention>,
    pub linear_attn: Option<QuantizedQwen3_5GatedDeltaNet>,
    pub mlp: crate::models::qwen3vl::quantized_model::QuantizedMLP, 
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3_5DecoderLayer {
    pub fn new_skeleton(
        config: &Qwen3_5TextConfig,
        base_name: &str,
        device: &Device,
        dtype: DType,
        layer_idx: usize,
        registry: KVRegistry,
    ) -> Result<Self> {
        let layer_type = config.layer_types[layer_idx].clone();
        let zero_t = Tensor::zeros((1,), dtype, device)?;
        
        let (self_attn, linear_attn) = if layer_type == "linear_attention" {
            let dummy_q = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
            (None, Some(QuantizedQwen3_5GatedDeltaNet { in_proj_qkv: dummy_q.clone(), in_proj_z: dummy_q.clone(), in_proj_b: dummy_q.clone(), in_proj_a: dummy_q.clone(), out_proj: dummy_q.clone() }))
        } else {
            let q_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
            let k_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
            let v_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
            let o_proj = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
            let q_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);
            let k_norm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps);
            
            let mut attn = QuantizedQwen3_5Attention {
                q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                num_kv_groups: config.num_attention_heads / config.num_key_value_heads.max(1),
                head_dim: config.head_dim,
                scaling: 1.0 / (config.head_dim as f64).sqrt(),
                kv_blocks: Vec::new(),
                registry,
                layer_idx,
                active_kv_name: None,
                active_session_id: None,
                vram_merged_k: None,
                vram_merged_v: None,
                merged_vram_block_count: 0,
            };
            attn.clear();
            (Some(attn), None)
        };

        let mlp_gate = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let mlp_up = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let mlp_down = QLinear::new(QMatMul::Tensor(zero_t.clone()), None, device.clone());
        let mut mlp = crate::models::qwen3vl::quantized_model::QuantizedMLP { gate_proj: mlp_gate, up_proj: mlp_up, down_proj: mlp_down };
        mlp.clear();

        let mut input_layernorm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps); input_layernorm.clear();
        let mut post_attention_layernorm = RmsNorm::new(zero_t.clone(), config.rms_norm_eps); post_attention_layernorm.clear();

        Ok(Self { layer_type, self_attn, linear_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn load_weights_inplace<R: std::io::Seek + std::io::Read>(
        &mut self, ct: &gguf_file::Content, reader: &mut R, base_name: &str, device: &Device, dtype: DType, baking_only: bool,
    ) -> Result<()> {
        if self.layer_type == "linear_attention" {
            if let Some(attn) = &mut self.linear_attn {
                attn.load_weights_inplace(ct, reader, base_name, device, dtype)?;
            }
        } else {
            if let Some(attn) = &mut self.self_attn {
                attn.load_weights_inplace(ct, reader, base_name, device, dtype)?;
            }
        }
        
        if !baking_only {
            self.mlp.gate_proj = get_qlinear(ct, reader, &format!("{base_name}.mlp.gate_proj"), device, dtype)?;
            self.mlp.up_proj = get_qlinear(ct, reader, &format!("{base_name}.mlp.up_proj"), device, dtype)?;
            self.mlp.down_proj = get_qlinear(ct, reader, &format!("{base_name}.mlp.down_proj"), device, dtype)?;
            self.post_attention_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.post_attention_layernorm"), self.input_layernorm.eps(), device, dtype)?;
        }
        self.input_layernorm = get_rms_norm(ct, reader, &format!("{base_name}.input_layernorm"), self.input_layernorm.eps(), device, dtype)?;
        Ok(())
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.to_device(device)?; }
        if let Some(attn) = &mut self.linear_attn { attn.to_device(device)?; }
        self.mlp.gate_proj.to_device(device)?;
        self.mlp.up_proj.to_device(device)?;
        self.mlp.down_proj.to_device(device)?;
        self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        if let Some(attn) = &mut self.self_attn { attn.clear(); }
        if let Some(attn) = &mut self.linear_attn { attn.clear(); }
        self.mlp.gate_proj.clear(); self.mlp.up_proj.clear(); self.mlp.down_proj.clear();
        self.input_layernorm.clear(); self.post_attention_layernorm.clear();
    }

    pub fn forward(
        &mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, seqlen_offset: usize, session_id: Option<String>, kv_name: Option<String>, baking_only: bool,
    ) -> Result<Tensor> {
        let dev = self.input_layernorm.weight().device();
        let target_dtype = self.input_layernorm.weight().dtype();
        let xs = if !xs.device().same_device(dev) { xs.to_device(dev)? } else { xs.clone() };
        let xs = if xs.dtype() != target_dtype { xs.to_dtype(target_dtype)? } else { xs };

        let residual = xs.clone();
        let mut xs = self.input_layernorm.forward(&xs)?;
        
        if self.layer_type == "linear_attention" {
            if let Some(attn) = &mut self.linear_attn {
                xs = attn.forward(&xs, seqlen_offset)?;
            }
        } else {
            if let Some(attn) = &mut self.self_attn {
                xs = attn.forward(&xs, cos, sin, attention_mask, seqlen_offset, session_id, kv_name, baking_only)?;
            }
        }
        
        let xs = residual.add(&xs)?;
        if !baking_only {
            let residual = xs.clone();
            let xs = self.post_attention_layernorm.forward(&xs)?;
            let gate = candle_nn::ops::silu(&self.mlp.gate_proj.forward(&xs)?)?;
            let up = self.mlp.up_proj.forward(&xs)?;
            let xs = self.mlp.down_proj.forward(&(gate * up)?)?;
            Ok(residual.add(&xs)?)
        } else {
            Ok(xs)
        }
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.evacuate_vram_to_cache()?; }
        // linear_attn 구현체가 고도화되면 동일하게 적용
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        if let Some(attn) = &self.self_attn { attn.get_kv_len() } else { 0 }
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.save_kv_cache(path, clear, offset, kv_name)?; }
        Ok(())
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.truncate_kv_cache(len)?; }
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.save_kv_cache(path, true, block_size, None)?; }
        Ok(())
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        if let Some(attn) = &mut self.self_attn { attn.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len)?; }
        Ok(())
    }
}

// ============================================================================
// [ROOT MODEL] 메인 모델 파이프라인 (비동기 핑퐁 로딩 및 청크 분할)
// ============================================================================
pub struct QuantizedQwen3_5Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<QuantizedQwen3_5DecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: Qwen3VLTextRotaryEmbedding,
    pub registry: KVRegistry,
    pub device: Device,
    pub device_id: usize,
    pub dtype: DType,
    pub is_forced_cpu: bool,
    pub baking_only: bool,
    pub current_kv_len: usize,
    pub mmap: Option<Arc<Mmap>>,
    pub config: Qwen3_5TextConfig,
    pub ct: Option<Arc<gguf_file::Content>>,
    pub base_name: String,
}

impl QuantizedQwen3_5Model {
    pub fn new_with_mmap(
        config: &Qwen3_5TextConfig,
        ct: Arc<gguf_file::Content>,
        mmap_handle: Option<Arc<Mmap>>,
        base_name: &str,
        device: &Device,
        device_id: usize,
        dtype: DType,
        baking_only: bool,
    ) -> Result<Self> {
        let is_forced_cpu = device.is_cpu();
        let mmap = mmap_handle.as_ref().map(|m| &m[..]).unwrap_or(&[]);
        let mut reader = std::io::Cursor::new(mmap);
        
        // 1. 임베딩 로드
        let token_emb = ct.tensor(&mut reader, &format!("{base_name}.embed_tokens.weight"), device)?;
        let embed_tokens = Embedding::new(token_emb.dequantize(device)?.to_dtype(dtype)?, config.hidden_size);
        
        // 2. 중앙 집중식 장부(Registry) 초기화
        let registry = KVRegistry::new();
        let num_layers_to_load = if baking_only { 1 } else { config.num_hidden_layers };
        
        // 3. 레이어 뼈대(Skeleton) 생성 (Zero-RAM Startup 최적화)
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_base = format!("{base_name}.layers.{i}");
            layers.push(QuantizedQwen3_5DecoderLayer::new_skeleton(
                config, &layer_base, device, dtype, i, registry.clone()
            )?);
        }

        // 4. 최종 정규화 및 RoPE 로드
        let norm = get_rms_norm(&ct, &mut reader, &format!("{base_name}.norm"), config.rms_norm_eps, device, dtype)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        
        Ok(Self {
            embed_tokens, 
            layers, 
            norm,
            rotary_emb: Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta),
            registry, 
            device: device.clone(), 
            device_id, 
            dtype,
            is_forced_cpu, 
            baking_only, 
            current_kv_len: 0,
            mmap: mmap_handle, 
            config: config.clone(), 
            ct: Some(ct), 
            base_name: base_name.to_string(),
        })
    }

    /// [MEMORY-OPT] 특정 레이어의 가중치를 SSD/Mmap에서 즉시(In-place) 읽어옵니다.
    pub fn reload_layer(&mut self, layer_idx: usize) -> Result<()> {
        let is_loaded = match self.layers[layer_idx].layer_type.as_str() {
            "linear_attention" => !self.layers[layer_idx].linear_attn.as_ref().unwrap().in_proj_qkv.is_cleared(),
            _ => !self.layers[layer_idx].self_attn.as_ref().unwrap().q_proj.is_cleared(),
        };
        if is_loaded { return Ok(()); }
        
        let mmap = self.mmap.as_ref().ok_or(anyhow!("Mmap missing"))?;
        let ct = self.ct.as_ref().ok_or(anyhow!("GGUF Content missing"))?;
        let mut reader = std::io::Cursor::new(&mmap[..]);
        let prefix = format!("{}.layers.{}", self.base_name, layer_idx);
        
        self.layers[layer_idx].load_weights_inplace(ct, &mut reader, &prefix, &Device::Cpu, self.dtype, self.baking_only)?;
        Ok(())
    }

    pub fn get_kv_len(&self) -> usize {
        self.current_kv_len
    }

    /// [CHUNK-ASYNC-PROCESSOR]
    /// 너무 긴 입력이 들어왔을 때 VRAM OOM(Out of Memory)이 발생하지 않도록
    /// 256 토큰 단위로 쪼개어 연산하고 결과를 합칩니다.
    async fn process_single_layer(
        &mut self, 
        layer_idx: usize, 
        xs: Tensor, 
        cos: &Tensor, 
        sin: &Tensor, 
        seqlen_offset: usize, 
        session_id: Option<String>, 
        kv_name: Option<String>
    ) -> Result<Tensor> {
        let input_token_count = xs.dim(1).unwrap_or(0);
        let is_decoding = input_token_count <= 1;

        // 프리필(Prefill) 모드일 때만 레이어를 동적으로 로드합니다.
        if !is_decoding { self.reload_layer(layer_idx)?; }
        
        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) };
        if !self.layers[layer_idx].input_layernorm.weight().device().same_device(&target_device) {
            self.layers[layer_idx].to_device(&target_device)?;
        }

        // [CRITICAL FIX] 256 사이즈로 청크 분할하여 Flash-Decoding 에뮬레이트
        let chunk_size = 256;
        let mut chunk_outputs = Vec::new();
        let chunk_offsets: Vec<usize> = (0..input_token_count).step_by(chunk_size).collect();
        
        for &i in &chunk_offsets {
            let take = (input_token_count - i).min(chunk_size);
            let xs_chunk = xs.narrow(1, i, take)?;
            let cos_chunk = cos.narrow(cos.rank().saturating_sub(2), i, take)?;
            let sin_chunk = sin.narrow(sin.rank().saturating_sub(2), i, take)?;
            
            let out = self.layers[layer_idx].forward(
                &xs_chunk, &cos_chunk, &sin_chunk, None, seqlen_offset + i, 
                session_id.clone(), kv_name.clone(), self.baking_only
            )?;
            
            chunk_outputs.push(out);
        }

        // 연산이 끝났으므로 GPU/RAM 메모리에서 즉시 가중치 소각 (방빼기)
        if !is_decoding { self.layers[layer_idx].clear(); }
        
        // 조각난 청크 결과물을 합칩니다. 단일 청크라면 불필요한 cat 연산 방지
        let final_out = if chunk_outputs.len() == 1 { 
            chunk_outputs.into_iter().next().unwrap() 
        } else { 
            Tensor::cat(&chunk_outputs, 1)? 
        };
        Ok(final_out)
    }

    /// [MAIN-FORWARD] 비동기 핑퐁 파이프라인 (Zero-Loading Time)
    pub async fn forward(
        &mut self, 
        inputs_embeds: &Tensor, 
        seqlen_offset: usize, 
        session_id: Option<String>, 
        kv_name: Option<String>
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let is_decoding = seq_len <= 1;
        let target_device = if self.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(self.device_id) };
        let mut xs = inputs_embeds.to_device(&target_device)?;

        // 1. Position IDs 및 RoPE 준비
        let position_ids = Tensor::arange(seqlen_offset as u32, (seq_len + seqlen_offset) as u32, inputs_embeds.device())?
            .unsqueeze(0)?.unsqueeze(0)?.broadcast_as((3, b_size, seq_len))?;
        let (cos, sin) = self.rotary_emb.forward(&position_ids, inputs_embeds.dtype(), vec![])?;

        let total_layers = self.layers.len();
        
        // 2. [PING-PONG CARRIER] 백그라운드 스레드로 넘겨줄 빈 껍데기 버퍼
        let mut ping_pong_carrier = QuantizedQwen3_5DecoderLayer::new_skeleton(
            &self.config, "dummy", &Device::Cpu, self.dtype, 0, self.registry.clone()
        )?;
        let mut next_layer_task: Option<tokio::task::JoinHandle<Result<QuantizedQwen3_5DecoderLayer>>> = None;

        // 0번 레이어는 첫 번째 루프 진입 전에 미리 띄워둡니다.
        if !is_decoding { 
            self.reload_layer(0)?; 
            self.layers[0].to_device(&target_device)?; 
        }

        // 3. 레이어 루프
        for layer_idx in 0..total_layers {
            // =========================================================================
            // [TRACK 2] 백그라운드 지연 적재 (N+1번째 레이어 미리 로드)
            // =========================================================================
            if layer_idx + 1 < total_layers {
                let next_idx = layer_idx + 1;
                let mmap_clone = self.mmap.clone();
                let ct_clone = self.ct.clone();
                let dtype = self.dtype;
                let baking_only = self.baking_only;
                let base_name = self.base_name.clone();
                
                let mut carrier = ping_pong_carrier.clone();
                
                // 디스크 I/O를 메인 스레드(연산)에서 완벽히 분리
                next_layer_task = Some(tokio::task::spawn_blocking(move || -> Result<QuantizedQwen3_5DecoderLayer> {
                    let mmap = mmap_clone.ok_or_else(|| anyhow!("Mmap missing"))?;
                    let ct_arc = ct_clone.ok_or_else(|| anyhow!("GGUF missing"))?;
                    let mut reader = std::io::Cursor::new(&mmap[..]);
                    let prefix = format!("{}.layers.{}", base_name, next_idx);
                    
                    // VRAM이 아닌 CPU RAM으로만 미리 불러서 대기시킴 (VRAM 스파이크 차단)
                    carrier.load_weights_inplace(&ct_arc, &mut reader, &prefix, &Device::Cpu, dtype, baking_only)?;
                    Ok(carrier)
                }));
            }

            // =========================================================================
            // [TRACK 1] 메인 GPU 연산 트랙 (N번째 레이어 처리)
            // =========================================================================
            xs = self.process_single_layer(
                layer_idx, xs, &cos, &sin, seqlen_offset, session_id.clone(), kv_name.clone()
            ).await?;

            if !is_decoding { 
                // 연산이 끝나서 텅 빈 N번째 레이어를 다음 루프의 껍데기(Carrier)로 재활용합니다.
                ping_pong_carrier = self.layers[layer_idx].clone(); 
            }

            // =========================================================================
            // [TRACK SYNC] 다음 턴으로 가기 전 백그라운드 장착 완료 대기
            // =========================================================================
            if let Some(task) = next_layer_task.take() {
                let mut ready_layer = task.await??;
                
                // 가중치만 교체하고, KV 캐시 데이터와 상태는 기존 N+1 레이어의 것을 보존해야 함
                if let (Some(old_attn), Some(new_attn)) = (&self.layers[layer_idx + 1].self_attn, &mut ready_layer.self_attn) {
                    new_attn.kv_blocks = old_attn.kv_blocks.clone();
                    new_attn.active_kv_name = old_attn.active_kv_name.clone();
                    new_attn.layer_idx = layer_idx + 1;
                }
                self.layers[layer_idx + 1] = ready_layer;
            }
        }
        
        self.current_kv_len = seqlen_offset + seq_len;
        Ok(xs.apply(&self.norm)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear();
        }
        self.current_kv_len = 0;
    }

    pub fn evacuate_vram_to_cache(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.evacuate_vram_to_cache()?;
        }
        Ok(())
    }

    pub fn save_kv_cache(&mut self, path: &Path, clear: bool, offset: usize, kv_name: Option<&str>) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        // [STABILITY] 순차적 저장을 통해 CUDA_ERROR_INVALID_CONTEXT 방지
        self.layers.iter_mut().try_for_each(|layer| {
            layer.save_kv_cache(path, clear, offset, kv_name)
        })
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        self.layers.iter_mut().try_for_each(|layer| {
            layer.truncate_kv_cache(len)
        })?;
        self.current_kv_len = len;
        Ok(())
    }

    pub fn offload_kv_cache(&mut self, path: &Path, block_size: usize) -> Result<()> {
        self.save_kv_cache(path, true, block_size, None)
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        if !path.exists() { return Ok(()); }

        let mut fragments = Vec::new();
        let mut max_offset = 0;
        let scan_path = if let Some(name) = kv_name { path.join(name) } else { path.to_path_buf() };
        if !scan_path.exists() { return Ok(()); }

        if let Ok(entries) = std::fs::read_dir(&scan_path) {
            for entry in entries.flatten() {
                let path_buf = entry.path();
                if path_buf.is_dir() {
                    let dname = path_buf.file_name().unwrap_or_default().to_string_lossy();
                    if dname.starts_with('b') {
                        if let Ok(offset) = dname[1..].parse::<usize>() {
                            if offset > max_offset { max_offset = offset; }
                            fragments.push((offset, path_buf));
                        }
                    }
                }
            }
        }
        
        if fragments.is_empty() { return Ok(()); }
        fragments.sort_by_key(|f| f.0);
        
        // 마지막 청크의 실제 길이를 추출
        let mut last_chunk_len = 256;
        let (_, last_st_path) = fragments.last().unwrap();
        if let Ok(content) = crate::utils::direct_loader::load_kv_block(&last_st_path.join("l0.st")) {
            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                if let Some(name) = st.names().iter().find(|n| n.contains("k_shape")) {
                    if let Ok(view) = st.tensor(name) {
                        let data = view.data();
                        if data.len() >= 12 {
                            last_chunk_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
                        }
                    }
                }
            }
        }
        
        let total_kv_len = max_offset + last_chunk_len;
        self.current_kv_len = total_kv_len;
        println!("[SSD-GLOBAL] Snapshot loaded. Total context length: {} tokens.", total_kv_len);
        
        self.layers.iter_mut().try_for_each(|layer| {
            layer.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, &fragments, total_kv_len)
        })?;
        
        self.current_kv_len = total_kv_len;
        Ok(())
    }

    /// [MANUAL-FLUSH] 세션 종료 시 RAM에 남아있는 활성 블록(Active Block)을 강제로 SSD에 저장합니다.
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        use crate::models::qwen3vl::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        let mut block_groups: std::collections::HashMap<usize, Vec<LayerKVDump>> = std::collections::HashMap::new();
        
        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            if let Some(attn) = &mut layer.self_attn {
                for block in &mut attn.kv_blocks {
                    let inner = block.inner.write().unwrap();
                    let is_dirty = {
                        let reg = self.registry.entries.read().unwrap();
                        if inner.index < reg.len() { 
                            if l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] } else { true }
                        } else { true }
                    };
                    
                    if inner.k_cache.is_some() && is_dirty {
                        let k = inner.k_cache.as_ref().unwrap();
                        let k_shape_u32: Vec<u32> = k.shape().dims().iter().map(|&x| x as u32).collect();
                        
                        block_groups.entry(inner.offset).or_default().push(LayerKVDump {
                            layer_idx: l_idx,
                            k_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                            v_data: Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                            k_shape: Tensor::from_vec(k_shape_u32, (k.shape().dims().len(),), &Device::Cpu)?,
                            // [SAFE-FLUSH] 복제본 대체로 에러 탈출 방지
                            raw_k: Some(k.to_device(&Device::Cpu).unwrap_or(k.clone())), 
                            raw_v: Some(inner.v_cache.as_ref().unwrap().to_device(&Device::Cpu).unwrap_or(inner.v_cache.as_ref().unwrap().clone())),
                        });
                        
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() {
                            reg[inner.index].is_dirty[l_idx] = false;
                        }
                    }
                }
            }
        }

        if block_groups.is_empty() { return Ok(()); }

        if let Some(tx) = BAKE_TX.get() {
            let kv_dir = crate::utils::paths::get_kv_dir(None);
            let mode = self.baking_only;
            let kv_name_raw = kv_name.unwrap_or("text");
            let last_part = kv_name_raw.split('/').last().unwrap_or("text");
            let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text".to_string() } else { last_part.to_string() };
            let sub_path = if mode { format!("{}/reference/{}", session_id, kv_type) } else { format!("{}/inference/{}", session_id, kv_type) };
            
            for (off, layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(256).await;
                let block_dir = kv_dir.join(&sub_path).join(format!("b{}", off));
                if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path.clone()), offset: off, layers,
                    is_relay_baking: mode, block_idx: Some(off / 256), registry: self.registry.clone(),
                })).await.is_err() {
                    crate::models::qwen3vl::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    SLOT_MANAGER.release_slot(sid).await;
                }
            }
        }
        Ok(())
    }
}