    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        // [STAGE-RESET] 작업 시작 전 슬롯 및 모델 캐시 초기화
        SLOT_MANAGER.reset_all_slots().await;
        self.clear_kv_cache();

        // [CENTRAL-STAGING-READY] 중앙 통제 슬롯 초기화
        let (hidden_size, num_heads) = match self.qwen3_vl {
            ModelVariant::QuantizedText(ref m) => (m.language_model.embed_tokens.embeddings().dim(1).unwrap_or(1024), m.language_model.layers[0].self_attn.num_attention_heads),
            ModelVariant::QuantizedVL(ref m) => (m.language_model.embed_tokens.embeddings().dim(1).unwrap_or(1024), m.language_model.layers[0].self_attn.num_attention_heads),
            _ => (1024, 16),
        };
        let _ = SLOT_MANAGER.init_staging_buffer(&self.text_device, hidden_size, num_heads);

        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            if path.exists() { let _ = std::fs::remove_dir_all(&path); }
            let _ = std::fs::create_dir_all(&path);
        }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();
        
        let prefill_chunk_size = 256;
        let mut current_pos = self.get_kv_len();
        let mut chunk_idx = 0; 
        if current_pos > 0 { chunk_idx = current_pos / 256; }

        while current_pos < total_tokens {
            if let Some(flag) = &cancel_flag { 
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } 
            }

            let end = (current_pos + prefill_chunk_size).min(total_tokens);
            let chunk_len = end - current_pos;
            let chunk = &full_input_ids_vec[current_pos..end];
            
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_len), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            // [ROUND-ROBIN-RESERVATION] 베이스 슬롯 예약
            let (ks_temp, _) = self.get_current_kv();
            let layer_count = ks_temp.len();
            for l_idx in 0..layer_count.min(28) {
                let base_slot_id = if layer_count <= 1 { chunk_idx % 28 } else { l_idx };
                SLOT_MANAGER.wait_for_base_slot_capacity(base_slot_id).await;
                SLOT_MANAGER.get_base_slot(base_slot_id).state.store(1, Ordering::SeqCst);
            }

            // 1. GPU 추론 진행
            println!("[BAKING] {} to {} / Total: {}", current_pos, end, total_tokens);
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone()).await?;

            // 2. [NON-BLOCKING-CENTRAL-HANDOFF] 중앙 통제 슬롯으로 비동기 인계
            if let Some(sid) = &session_id {
                let current_chunk_idx = chunk_idx;
                let current_chunk_pos = current_pos;
                let current_chunk_len = chunk_len;
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                let (ks, vs) = self.get_current_kv();
                let registry = match self.qwen3_vl {
                    ModelVariant::QuantizedText(ref m) => Some(m.language_model.registry.clone()),
                    ModelVariant::QuantizedVL(ref m) => Some(m.language_model.registry.clone()),
                    _ => None,
                };
                let kv_name_clone = kv_name.clone();

                if let Ok(sub_slot_id) = SLOT_MANAGER.acquire_sub_slot_non_blocking().await {
                    let sub_idx = sub_slot_id - 28;
                    let staging_offset = sub_idx * 256; 

                    // VRAM -> VRAM 복사 (매우 빠름)
                    {
                        let sk_opt = SLOT_MANAGER.staging_k.read().unwrap();
                        let sv_opt = SLOT_MANAGER.staging_v.read().unwrap();
                        if let (Some(sk), Some(sv)) = (sk_opt.as_ref(), sv_opt.as_ref()) {
                            for (l_idx, (k, v)) in ks.iter().zip(vs.iter()).enumerate() {
                                if l_idx < 28 {
                                    let s_len = k.dim(2).unwrap_or(0);
                                    let k_src = k.narrow(2, s_len.saturating_sub(current_chunk_len), current_chunk_len).and_then(|t| t.contiguous()).unwrap();
                                    let v_src = v.narrow(2, s_len.saturating_sub(current_chunk_len), current_chunk_len).and_then(|t| t.contiguous()).unwrap();
                                    let mut k_dst = sk.narrow(2, staging_offset, current_chunk_len).unwrap();
                                    let mut v_dst = sv.narrow(2, staging_offset, current_chunk_len).unwrap();
                                    let _ = k_dst.copy_(&k_src);
                                    let _ = v_dst.copy_(&v_src);
                                }
                            }
                        }
                    }

                    // 배경 태스크
                    tauri::async_runtime::spawn(async move {
                        let mut layer_dumps = Vec::new();
                        let block_unit = 256;
                        let global_block_index = current_chunk_pos / block_unit;
                        let layer_count_inner = ks.len();

                        {
                            let sk_opt = SLOT_MANAGER.staging_k.read().unwrap();
                            let sv_opt = SLOT_MANAGER.staging_v.read().unwrap();
                            if let (Some(sk), Some(sv)) = (sk_opt.as_ref(), sv_opt.as_ref()) {
                                for l_idx in 0..layer_count_inner.min(28) {
                                    let base_slot_id = if layer_count_inner <= 1 { current_chunk_idx % 28 } else { l_idx };
                                    let k_cpu = sk.narrow(2, staging_offset, current_chunk_len).and_then(|t| t.to_device(&Device::Cpu)).unwrap();
                                    let v_cpu = sv.narrow(2, staging_offset, current_chunk_len).and_then(|t| t.to_device(&Device::Cpu)).unwrap();
                                    
                                    let base_slot = SLOT_MANAGER.get_base_slot(base_slot_id);
                                    {
                                        if let Some(reg_obj) = &registry { *base_slot.tenant_registry.write().unwrap() = Some(reg_obj.clone()); }
                                        base_slot.tenant_index.store(global_block_index, Ordering::SeqCst);
                                        let mut k_guard = base_slot.k_layers[0].lock().await;
                                        let mut v_guard = base_slot.v_layers[0].lock().await;
                                        *k_guard = Some(k_cpu.clone());
                                        *v_guard = Some(v_cpu.clone());
                                        base_slot.state.store(2, Ordering::SeqCst);
                                        base_slot.ready_signal.notify_waiters();
                                    }

                                    if let Some(reg_obj) = &registry {
                                        let mut reg = reg_obj.entries.write().unwrap();
                                        if global_block_index < reg.len() {
                                            reg[global_block_index].location[l_idx] = KVLocation::RAM;
                                            reg[global_block_index].slot_ids[l_idx] = Some(base_slot.id);
                                        }
                                    }

                                    layer_dumps.push(LayerKVDump { layer_idx: l_idx, base_slot_id, k: k_cpu, v: v_cpu });
                                }
                            }
                        }

                        if let Ok(tx) = get_worker_channel().await {
                            ACTIVE_BAKE_TASKS.fetch_add(1, Ordering::SeqCst);
                            let _ = tx.send(SlotTask::Bake(BakeTask {
                                slot_id: sub_slot_id, task_dir: path.clone(), kv_name: kv_name_clone.clone(),
                                offset: current_chunk_pos, layers: layer_dumps,
                                registry: registry.clone(), vram_released_tx: None,
                            })).await;
                        }
                    });
                }
            }

            self.clear_temporal_kv_caches();
            current_pos = end;
            chunk_idx += 1;
            SLOT_MANAGER.debug_stats();
        }

        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) { let _ = serde_json::to_writer(file, &full_input_ids_vec); }
        }

        SLOT_MANAGER.wait_all_sub_slots().await;
        Ok(current_pos)
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, chunk_size, None).await?;
        if let Some(ref mut target) = relay_target {
            let (ks, vs) = self.get_current_kv();
            let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                let s_len = k.dim(candle_core::D::Minus2)?;
                let k_new = k.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                let v_new = v.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                    let rk = m.language_model.compress_to_bitkv(&k_new)?;
                    let rv = m.language_model.compress_to_bitkv(&v_new)?;
                    Ok((rk, rv))
                } else { Err(anyhow!("Unsupported")) }
            }).collect();
            let results = results?;
            let mut ka = vec![]; let mut kp = vec![]; let mut ks_ = vec![];
            let mut va = vec![]; let mut vp = vec![]; let mut vs_ = vec![];
            let mut os = vec![];
            for (rk, rv) in results {
                ka.push(rk.0); kp.push(rk.1); ks_.push(rk.2);
                va.push(rv.0); vp.push(rv.1); vs_.push(rv.2);
                os = rk.3;
            }
            if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &ks_, &va, &vp, &vs_, &os)?; }
        }
        Ok(chunk_size)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let top_p = mes.top_p.unwrap_or(0.9) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(40), seed);
        let mut all_ids = vec![];
        let mut generated_text = String::new();
        let mut seqlen_offset = self.get_kv_len();

        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let progress_path = path.join("generation_progress.json");
            if progress_path.exists() {
                if let Ok(file) = fs::File::open(progress_path) {
                    let progress: serde_json::Value = serde_json::from_reader(file).unwrap_or_default();
                    if let Some(ids) = progress["all_ids"].as_array() {
                        for id in ids { if let Some(u) = id.as_u64() { all_ids.push(u as u32); } }
                    }
                    if let Some(text) = progress["generated_text"].as_str() { generated_text = text.to_string(); }
                }
            }
        }

        let mut current_ids = all_ids.clone();
        for _ in 0..mes.max_tokens.unwrap_or(1024) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            let input_ids = if current_ids.is_empty() {
                let mes_render = self.chat_template.apply_chat_template(&mes)?;
                self.tokenizer.text_encode_vec(mes_render, false)?
            } else { vec![*current_ids.last().unwrap()] };

            let input_tensor = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &self.text_device)?;
            let logits = self.qwen3_vl.forward(&input_tensor, None, None, None, None, None, seqlen_offset, 0, session_id.clone()).await?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let pr = apply_repeat_penalty(&logits, 1.1, &current_ids)?;
            let next_token = logit_processor.sample(&pr)?;
            
            current_ids.push(next_token);
            all_ids.push(next_token);
            seqlen_offset += input_ids.len();

            let decoded = self.tokenizer.decode(&[next_token])?;
            generated_text.push_str(&decoded);
            if self.tokenizer.is_special_token(next_token) { break; }
        }
        Ok(generated_text)
    }

    pub fn get_kv_len(&self) -> usize { self.qwen3_vl.get_kv_len() }
    pub fn clear_kv_cache(&mut self) { self.qwen3_vl.clear_kv_cache(); }
    pub fn clear_temporal_kv_caches(&mut self) { 
        if let ModelVariant::QuantizedText(ref mut m) = self.qwen3_vl { m.language_model.clear_temporal_kv_caches(); }
        else if let ModelVariant::QuantizedVL(ref mut m) = self.qwen3_vl { m.language_model.clear_temporal_kv_caches(); }
    }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) { self.qwen3_vl.get_current_kv() }
    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { self.qwen3_vl.save_kv_cache(path, false, offset, kv_name) }
}
