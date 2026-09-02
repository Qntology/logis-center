use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use anyhow::anyhow;
use image::DynamicImage;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use crate::model::ModelSize;
use crate::openai_types::*;

impl crate::model::LogisModel {

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [FIX] Default to Qwen (0.6B) for all chat tasks
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // 🌟 [VISION-JIT] chat 은 ChatCompletionRequestMessageContentPart::Text 만 조립하는
        //    순수 텍스트 경로입니다. 비전 가중치가 붙어 있다면 여기서 반환합니다.
        {
            let mut gen_guard = self.generator.lock().await;
            if let Some(gen) = gen_guard.as_mut() {
                if gen.is_vision_jit_capable() && gen.vision_resident() {
                    let _ = gen.set_vision_active(false);
                }
            }
        }
        
        let _self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        {
            let mut gen_guard = self.generator.lock().await;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_text,
                name: None,
            });

            let content_parts = vec![
                ChatCompletionRequestMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text }
                )
            ];

            let user_message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
                model: "qwen".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }
    }

    pub async fn chat_with_spinner(
        &self, 
        system: &str, 
        user_input: &str,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let content_parts = vec![
            ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
            )
        ];

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id, kv_name).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [FIX] Ensure we stay on Qwen (0.6B).
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(app_handle, task_id, &base_payload);
        }

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn chat_with_image_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        _base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Removed redundant emit. Only log the progress if needed.
        // let _ = app_handle.emit(event_name, base_payload);

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?; // 🌟 무조건 Qwen으로 로드
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        system: &str,       // 🌟 추가
        user_input: &str,   // 🌟 변경
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(_app_handle, task_id, &base_payload);
        }

        let max_tok = self.max_tokens_limit;
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tok),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = crate::prompts::get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            "You are a highly precise document data extraction assistant.", // 🌟 System 주입
            &prompt,                                                        // 🌟 User 주입
            Some(master_img),
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    // =====================================================================
    // 🌟 [CROSSOVER] 임베딩 캐시 RAM 상한
    // ---------------------------------------------------------------------
    //  ── 왜 필요해졌나 ──
    //   get_embedding_batch 가 캐시를 쓰게 되면서 적재량이 크게 늘어납니다.
    //   기존에는 단건 호출만 캐시했기에 사실상 자라지 않았고,
    //   그래서 상한이 없어도 문제가 드러나지 않았습니다.
    //
    //  ── 이 값의 성격 ──
    //   판정 임계치가 아니라 'RAM 사용 상한' 입니다. 정확도에 전혀 영향이 없고,
    //   초과 시 캐시를 비워도 결과가 달라지지 않습니다(다시 계산할 뿐입니다).
    //   기존 코드가 이미 unload_embedding / deep_purge 에서 cache.clear() 를
    //   수행하므로, 전체 비우기는 이 코드베이스의 기존 정리 방식과 동일합니다.
    const EMBED_CACHE_RAM_BUDGET_BYTES: usize = 64 * 1024 * 1024;

    /// 캐시 엔트리 하나의 대략적 RAM 점유(바이트).
    ///   벡터 384 × f32 = 1536B + 키 문자열/해시 오버헤드 ≈ 96B
    const EMBED_CACHE_ENTRY_BYTES: usize = 384 * 4 + 96;

    fn guard_embedding_cache(cache: &mut std::collections::HashMap<String, Vec<f32>>) {
        let cap = Self::EMBED_CACHE_RAM_BUDGET_BYTES / Self::EMBED_CACHE_ENTRY_BYTES;
        if cache.len() > cap {
            println!(
                "[MODEL] 🧹 [EMBED CACHE] 엔트리 {}개가 RAM 예산({}MB)을 초과하여 캐시를 비웁니다. (정확도 영향 없음)",
                cache.len(),
                Self::EMBED_CACHE_RAM_BUDGET_BYTES / (1024 * 1024)
            );
            cache.clear();
        }
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        // 🌟 1. 메모리 캐시부터 확인합니다 (중복된 텍스트면 GPU 연산 원천 차단)
        {
            let cache = self.embedding_cache.lock().await;
            if let Some(vector) = cache.get(&text) {
                return Ok(vector.clone());
            }
        }
        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;
        // 🌟 [CROSSOVER] 이 호출이 실제로 쓴 순간 점유를 학습합니다.
        //    가중치가 아니라 activation 축이며, embedding_budget_mb() 에 더해져
        //    다음 예산 판정이 '연산 중 여유' 까지 포함하게 만듭니다.
        let free_before = self.get_free_vram_mb();
        let embedding_model_arc = self.embedding_model.clone();
        let text_clone = text.clone();
        
        let vector = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text_clone).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 384])
            }
        }).await??;
        self.observe_activation_headroom(free_before);
        // 🌟 2. 새로 연산된 벡터를 해시맵 캐시에 저장하여 다음 루프 때 재사용합니다.
        {
            let mut cache = self.embedding_cache.lock().await;
            Self::guard_embedding_cache(&mut cache);
            cache.insert(text, vector.clone());
        }
        Ok(vector)
    }

    // =====================================================================
    // 🌟 [CROSSOVER] 배치 임베딩 — 캐시 · 중복 제거 · 적응 청킹
    // ---------------------------------------------------------------------
    //  ── 무엇이 문제였나 ──
    //   ① 캐시 부재
    //      get_embedding 은 캐시를 읽고 쓰는데 이 함수는 둘 다 하지 않았습니다.
    //      indexing.rs 의 anchor_texts 는 property 가 같으면 문자열이 완전히
    //      동일하고, upsert_alias_chunks 는 한 배치 안에서 같은 anchor 를
    //      변종 수만큼 중복해서 넣습니다. 전부 재계산되고 있었습니다.
    //      연산하지 않아도 되는 것을 연산하면 그만큼 activation 이 커지고,
    //      그것이 곧 '가중치 합' 과 별개인 두 번째 피크 축입니다.
    //
    //   ② 청킹 계층 오류
    //      호출부(index_item_chunks)에서 직접 쪼개면 scheduler.rs 의 수십 개
    //      다른 호출부는 보호받지 못합니다. 여기가 올바른 계층입니다.
    //
    //   ③ 락 재획득 위험
    //      청크마다 spawn_blocking 을 새로 하면 그 틈에 unload_embedding 이
    //      끼어들어 모델이 사라질 수 있고, 이후 청크는 조용히 0벡터를 반환합니다.
    //      틀린 벡터가 그대로 저장되는 무증상 손상입니다.
    //      spawn_blocking 하나 안에서 락을 쥔 채 청크를 순회하면
    //      activation 축소라는 목적은 그대로 달성하면서 이 위험이 사라집니다.
    // =====================================================================
    pub async fn get_embedding_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // ── ① 캐시 조회 + 배치 내 중복 제거 ──
        //    miss_texts[i] 를 계산하면 miss_slots[i] 의 모든 자리에 채워 넣습니다.
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_texts: Vec<String> = Vec::new();
        let mut miss_slots: Vec<Vec<usize>> = Vec::new();
        {
            let cache = self.embedding_cache.lock().await;
            let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
            for (i, t) in texts.iter().enumerate() {
                if let Some(v) = cache.get(t) {
                    out[i] = Some(v.clone());
                    continue;
                }
                match seen.get(t.as_str()) {
                    Some(&mi) => miss_slots[mi].push(i),
                    None => {
                        seen.insert(t.as_str(), miss_texts.len());
                        miss_texts.push(t.clone());
                        miss_slots.push(vec![i]);
                    }
                }
            }
        }

        let dup_folded: usize = miss_slots.iter().map(|s| s.len() - 1).sum();
        let hit_count = out.iter().filter(|v| v.is_some()).count();

        // ── ② 전부 캐시 적중이면 모델을 건드리지 않습니다 ──
        //    ensure_embedding 조차 부르지 않으므로, 생성 모델이 상주 중이어도
        //    임베딩이 되올라오는 일이 없습니다. 피크가 물리적으로 발생하지 않습니다.
        if miss_texts.is_empty() {
            if hit_count > 0 {
                println!(
                    "[MODEL] ⚡ [EMBED CACHE] {}건 전부 캐시 적중. 모델 로드 없이 반환합니다.",
                    hit_count
                );
            }
            return Ok(out.into_iter().map(|v| v.unwrap_or_else(|| vec![0.0; 384])).collect());
        }

        self.ensure_embedding().await?;

        // ── ③ 실연산 규모 로깅 ──
        //
        //  🌟 [CROSSOVER / 정정] embedding.rs 대조 결과 embed_batch 는
        //     이름과 달리 배치 연산이 아니라 self.embed(text) 1건씩 순회입니다.
        //     시퀀스 길이도 embed() 내부에서 512 로 고정되어 있습니다.
        //     따라서 '한 번에 몇 건을 넘기는가' 는 activation 에 영향이 없고,
        //     실제 축은 embed_batch 내부의 '동시 순전파 스레드 수' 입니다.
        //     그 조절은 embedding.rs 의 adaptive_thread_count 가 담당하며,
        //     여기서는 청킹으로 activation 을 줄이려 하지 않습니다.
        //
        //  ── 그래도 청킹을 남기는 이유 ──
        //     한 번에 수천 건을 넘기면 embed_batch 가 결과 Vec 전체를
        //     RAM 에 한꺼번에 들고 있게 됩니다. 그것은 VRAM 이 아니라 RAM 축이며,
        //     아래 상한은 그 목적만 갖습니다. VRAM 판정과 무관합니다.
        const RAM_CHUNK: usize = 512;
        if hit_count > 0 || dup_folded > 0 {
            println!(
                "[MODEL] 🧮 [EMBED BATCH] 요청 {}건 | 캐시 적중 {}건 | 배치 내 중복 {}건 접음 | 실연산 {}건",
                texts.len(), hit_count, dup_folded, miss_texts.len()
            );
        }

        let free_before = self.get_free_vram_mb();
        let embedding_model_arc = self.embedding_model.clone();
        let miss_for_compute = miss_texts.clone();
        let miss_len = miss_texts.len();

        let (computed, used_threads) = tokio::task::spawn_blocking(
            move || -> anyhow::Result<(Vec<Vec<f32>>, usize)> {
                // 🌟 락을 '한 번만' 잡고 순회합니다.
                //    이 스코프가 끝날 때까지 모델이 파기될 수 없으므로
                //    중간에 0벡터로 대체되는 무증상 손상이 원천 차단됩니다.
                let guard = embedding_model_arc.blocking_lock();
                let model = match guard.as_ref() {
                    Some(m) => m,
                    None => return Ok((vec![vec![0.0; 384]; miss_for_compute.len()], 0)),
                };
                // 관측에 쓸 실제 스레드 수를 미리 확보합니다.
                let threads = model.last_thread_count(miss_for_compute.len().min(RAM_CHUNK));
                let mut acc: Vec<Vec<f32>> = Vec::with_capacity(miss_for_compute.len());
                for part in miss_for_compute.chunks(RAM_CHUNK) {
                    let v = model
                        .embed_batch(part)
                        .map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;
                    acc.extend(v);
                }
                Ok((acc, threads))
            },
        ).await??;

        // 🌟 [CROSSOVER] 두 축을 함께 관측합니다.
        //    ① 총 순간 점유 → embedding_budget_mb 에 반영 (가중치 축)
        //    ② 스레드당 점유 → adaptive_thread_count 에 반영 (activation 축)
        self.observe_activation_headroom(free_before);
        if used_threads > 0 && miss_len > 0 {
            crate::models::embedding::EmbeddingModel::record_unit_activation(
                free_before,
                self.get_free_vram_mb(),
                used_threads,
            );
        }

        // ── ④ 결과 흩뿌리기 + 캐시 적재 ──
        {
            let mut cache = self.embedding_cache.lock().await;
            Self::guard_embedding_cache(&mut cache);
            for (mi, vec) in computed.into_iter().enumerate() {
                if mi >= miss_slots.len() { break; }
                for &slot in &miss_slots[mi] {
                    out[slot] = Some(vec.clone());
                }
                cache.insert(miss_texts[mi].clone(), vec);
            }
        }

        // 모델이 사라졌거나 embed_batch 가 짧게 반환한 자리는 0벡터로 채웁니다.
        // (호출부는 전부 이 길이를 전제로 인덱싱하므로 길이 보존이 계약입니다)
        Ok(out.into_iter().map(|v| v.unwrap_or_else(|| vec![0.0; 384])).collect())
    }
}