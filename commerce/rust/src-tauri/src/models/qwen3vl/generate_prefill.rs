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
        
        // [증분 저장] 256 토큰 단위로 기억 조각을 생성 (중앙 통제 슬롯 최적화)
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

            // [ROUND-ROBIN-RESERVATION] 이번 청크가 사용할 베이스 슬롯 예약
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