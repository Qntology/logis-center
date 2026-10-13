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
    // 🌟 [EMBED-CACHE] 임베딩 벡터 캐시 = 프로세스 전역 GRANITE_TEXT_CACHE
    // ---------------------------------------------------------------------
    //  ── 무엇이 바뀌었나 ──
    //   예전에는 self.embedding_cache(HashMap)를 썼고, 64MB 를 넘으면 전체를 비웠으며
    //   unload_embedding / deep_purge / Qwen3.5 릴레이도 매번 비웠습니다.
    //   이제 get_embedding / get_embedding_batch 는 models::embedding::text_cache 의
    //   전역 캐시를 읽고 씁니다. 퍼지와 무관하게 유지되고, 상한을 넘으면
    //   오래된 LRU 엔트리 10% 만 퇴출하며, 뱅크 구는 고정 계층에 둡니다.
    //   self.embedding_cache 필드는 API 호환을 위해 남겨 두었지만 더 이상 쓰지 않습니다.
    //
    //  ── 캐시 키 ──
    //   (모델 지문 + 장치 dtype 계열 + 순전파 태그, 텍스트).
    //   CUDA/ROCm 배치 순전파(embed_many)와 per-text 순전파는 서로 다른 공간이라
    //   두 경로의 벡터가 섞이지 않습니다.
    //
    //  ── 0벡터 ──
    //   모델이 사라졌거나 연산이 실패해 생긴 0벡터는 반환 길이를 지키기 위해
    //   돌려주기만 하고 캐시에는 넣지 않습니다(영구 캐시 오염 방지).
    // =====================================================================

    /// 이 LogisModel 의 임베딩 캐시 공간 id.
    pub fn embed_cache_space(&self) -> u64 {
        crate::models::embedding::text_cache::space_for(&self.embedding_path, &self.device_config.device)
    }

    /// 배치 순전파가 켜져 있어 '쓰일지 모르는 텍스트까지 미리 임베딩' 하는 추측성 프리패치가 싼지 여부.
    /// (per-text 경로에서는 추측분 1건마다 순전파 1회가 그대로 들기 때문에 하지 않습니다)
    pub fn embed_speculation_cheap(&self) -> bool {
        !self.is_cpu_mode && crate::models::embedding::batched_enabled(&self.device_config.device)
    }

    // =====================================================================
    // 🌟 [RAM WATCH / EMBED] 작업 집합 반환 폭주 제거 (E16)
    // ---------------------------------------------------------------------
    //  ── 무엇이 문제였나 ──
    //   여유 RAM 이 압박 기준(2GB) 미만이면 미적중 배치마다(2초 간격) 프로세스
    //   작업 집합을 통째로 반환했습니다. 이 PC 는 평소 여유가 1.8GB 라 거의 매번
    //   발동했고, 반환된 토크나이저·bias JSON·캐시 페이지가 곧바로 소프트 폴트로
    //   되돌아오며 임베딩 루프를 느리게 하고 RAM 그래프를 톱니로 만들었습니다.
    //
    //  ── 해결 ──
    //   반환은 '저수위(압박 기준의 절반) 미만' 일 때만, 10초에 한 번까지 합니다.
    //   압박 기준과 저수위 사이에서는 반환하지 않고 30초에 한 번 기록만 남깁니다.
    //   SDS 기준선 기록은 기존과 같이 매번 합니다.
    // =====================================================================
    fn embed_ram_watch(&self, texts: usize) {
        static LAST_TRIM_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static LAST_NOTE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let (avail, total) = crate::model::ram_snapshot();
        crate::utils::score_dynamics::record_baseline("model.ram_free_mb.embed", (avail / 1_000_000) as f32);
        crate::utils::score_dynamics::record_baseline("model.ram_used_pct.embed", crate::model::ram_used_pct(avail, total));
        if avail >= crate::model::RAM_PRESSURE_MARGIN_BYTES {
            return;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let low_water = crate::model::RAM_PRESSURE_MARGIN_BYTES / 2;
        if avail >= low_water {
            let last = LAST_NOTE_MS.load(std::sync::atomic::Ordering::Relaxed);
            if last == 0 || now_ms.saturating_sub(last) >= 30_000 {
                LAST_NOTE_MS.store(now_ms, std::sync::atomic::Ordering::Relaxed);
                println!(
                    "[MODEL] 🧠 [RAM WATCH / EMBED] 실연산 {}건 직전 여유 {}MB / 총 {}MB — 압박 기준 {}MB 미만이지만 저수위 {}MB 위라 작업 집합 반환을 생략합니다 (반환 폭주 방지 · 이 줄은 30초에 한 번)",
                    texts,
                    avail / 1_000_000,
                    total / 1_000_000,
                    crate::model::RAM_PRESSURE_MARGIN_BYTES / 1_000_000,
                    low_water / 1_000_000
                );
            }
            return;
        }
        let last_ms = LAST_TRIM_MS.load(std::sync::atomic::Ordering::Relaxed);
        if last_ms > 0 && now_ms.saturating_sub(last_ms) < 10_000 {
            return;
        }
        LAST_TRIM_MS.store(now_ms, std::sync::atomic::Ordering::Relaxed);
        // 🌟 [EMBED-CACHE] RAM 비상(저수위 미만): 질의 텍스트 LRU 계층을 비웁니다.
        //    뱅크 구(고정 계층)는 남기므로 다음 검색의 뱅크 재계산은 생기지 않습니다.
        let dropped = crate::models::embedding::text_cache::GRANITE_TEXT_CACHE.drop_lru_tier();
        crate::model::release_os_working_set();
        let (free_now, _) = crate::model::ram_snapshot();
        crate::utils::score_dynamics::record_baseline(
            "model.ram_trim_gain_mb.embed",
            (free_now.saturating_sub(avail) / 1_000_000) as f32,
        );
        let cache = crate::models::embedding::text_cache::GRANITE_TEXT_CACHE.stats();
        println!(
            "[MODEL] 🧠 [RAM WATCH / EMBED] 실연산 {}건 직전 {} → 저수위 {}MB 미만이라 임베딩 캐시 LRU {}건 비움 + 작업 집합 반환 후 여유 {}MB | 임베딩 장치 {} · 임베딩 캐시 {}건(고정 {}) | 반환은 10초에 한 번까지 (SDS 는 매번 기록)",
            texts,
            crate::model::ram_pressure_line(avail, total),
            low_water / 1_000_000,
            dropped,
            free_now / 1_000_000,
            if self.is_cpu_mode { "CPU(가중치가 RAM 에 있음)" } else { "GPU(가중치가 VRAM 에 있음)" },
            cache.entries,
            cache.pinned
        );
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        use crate::models::embedding::text_cache::{is_cacheable, GRANITE_TEXT_CACHE};
        // 🌟 1. 전역 캐시부터 확인합니다 (중복된 텍스트면 GPU 연산·모델 로드 원천 차단)
        let space = self.embed_cache_space();
        if let Some(vector) = GRANITE_TEXT_CACHE.get(space, &text) {
            return Ok(vector.to_vec());
        }
        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;
        // 🌟 [CROSSOVER] 이 호출이 실제로 쓴 순간 점유를 학습합니다.
        //    가중치가 아니라 activation 축이며, embedding_budget_mb() 에 더해져
        //    다음 예산 판정이 '연산 중 여유' 까지 포함하게 만듭니다.
        //    (NVML 은 이제 전역 핸들이라 질의 1회가 ~0.01ms 입니다)
        let free_before = self.get_free_vram_mb();
        let embedding_model_arc = self.embedding_model.clone();
        let text_clone = text.clone();

        let vector = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                if model.batched_enabled() {
                    // 🌟 [EMBED-BATCHED] 1건도 배치와 같은 고정 모양 마이크로배치를 거칩니다.
                    //    그래서 단건으로 구한 벡터와 어떤 배치 안에서 구한 벡터가 비트 단위로 같고,
                    //    캐시 공간도 배치 경로와 같습니다.
                    let mut v = model
                        .embed_many(std::slice::from_ref(&text_clone), &crate::models::embedding::BatchOpts::default())
                        .map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;
                    Ok(v.pop().unwrap_or_else(|| vec![0.0; crate::models::embedding::EMBED_DIM]))
                } else {
                    model.embed(&text_clone).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
                }
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; crate::models::embedding::EMBED_DIM])
            }
        }).await??;
        self.observe_activation_headroom(free_before);
        // 🌟 2. 새로 연산된 벡터를 전역 캐시에 저장합니다. (0벡터는 저장하지 않음)
        if is_cacheable(&vector) {
            GRANITE_TEXT_CACHE.insert(space, text, std::sync::Arc::from(vector.as_slice()), false);
        } else {
            println!("[MODEL] ⚠️ [EMBED FAIL] 1건이 0벡터로 반환되었습니다. 캐시에 넣지 않고 다음 호출에서 다시 계산합니다.");
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
    //
    //   ④ 🌟 [EMBED-BATCHED] CUDA/ROCm 에서는 embed_many(진짜 배치 순전파)를 씁니다.
    //      실패한 묶음만 per-text 로 다시 계산하고, 그 벡터는 다른 공간의 값이므로
    //      캐시에 넣지 않습니다.
    // =====================================================================
    pub async fn get_embedding_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.get_embedding_batch_inner(texts, false).await
    }

    /// 🌟 [EMBED-CACHE] 질의와 무관한 뱅크 구(속성·필터·동사·레이아웃·라벨 뱅크) 전용.
    /// 결과는 get_embedding_batch 와 같고, 캐시의 '고정' 계층에 들어가 LRU 퇴출 대상이 아닙니다.
    pub async fn get_embedding_batch_pinned(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.get_embedding_batch_inner(texts, true).await
    }

    /// 🌟 [EMBED-PREFETCH] 루프가 곧 1건씩 요청할 텍스트를 한 번의 배치로 미리 캐시에 올립니다.
    /// 이후 get_embedding 은 전부 캐시 적중이라 결과가 같고(같은 벡터), 모델·NVML·spawn 비용이 사라집니다.
    pub async fn prefetch_embeddings(&self, texts: Vec<String>) {
        if texts.is_empty() {
            return;
        }
        if let Err(e) = self.get_embedding_batch_inner(texts, false).await {
            println!("[MODEL] ⚠️ [EMBED-PREFETCH] 프리패치 실패({}). 루프가 1건씩 계산합니다.", e);
        }
    }

    /// 뱅크 구 프리패치(고정 계층).
    pub async fn prefetch_embeddings_pinned(&self, texts: Vec<String>) {
        if texts.is_empty() {
            return;
        }
        if let Err(e) = self.get_embedding_batch_inner(texts, true).await {
            println!("[MODEL] ⚠️ [EMBED-PREFETCH] 뱅크 프리패치 실패({}). 루프가 1건씩 계산합니다.", e);
        }
    }

    async fn get_embedding_batch_inner(&self, texts: Vec<String>, pinned: bool) -> anyhow::Result<Vec<Vec<f32>>> {
        use crate::models::embedding::text_cache::{is_cacheable, GRANITE_TEXT_CACHE};
        use crate::models::embedding::EMBED_DIM;
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // ── ① 캐시 조회 + 배치 내 중복 제거 ──
        //    miss_texts[i] 를 계산하면 miss_slots[i] 의 모든 자리에 채워 넣습니다.
        let space = self.embed_cache_space();
        let cached = GRANITE_TEXT_CACHE.get_many(space, &texts, pinned);
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_texts: Vec<String> = Vec::new();
        let mut miss_slots: Vec<Vec<usize>> = Vec::new();
        {
            let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
            for (i, (t, c)) in texts.iter().zip(cached.into_iter()).enumerate() {
                if let Some(v) = c {
                    out[i] = Some(v.to_vec());
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
            if hit_count > 1 {
                println!(
                    "[MODEL] ⚡ [EMBED CACHE] {}건 전부 캐시 적중. 모델 로드 없이 반환합니다.",
                    hit_count
                );
            }
            return Ok(out.into_iter().map(|v| v.unwrap_or_else(|| vec![0.0; EMBED_DIM])).collect());
        }

        self.ensure_embedding().await?;
        self.embed_ram_watch(miss_texts.len());
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

        struct Computed {
            vecs: Vec<Vec<f32>>,
            cacheable: Vec<bool>,
            threads: usize,
            batched: Option<crate::models::embedding::BatchRunStats>,
            ms: f64,
        }

        let computed = tokio::task::spawn_blocking(
            move || -> anyhow::Result<Computed> {
                // 🌟 락을 '한 번만' 잡고 순회합니다.
                //    이 스코프가 끝날 때까지 모델이 파기될 수 없으므로
                //    중간에 0벡터로 대체되는 무증상 손상이 원천 차단됩니다.
                let started = std::time::Instant::now();
                let guard = embedding_model_arc.blocking_lock();
                let n = miss_for_compute.len();
                let model = match guard.as_ref() {
                    Some(m) => m,
                    None => {
                        return Ok(Computed {
                            vecs: vec![vec![0.0; EMBED_DIM]; n],
                            cacheable: vec![false; n],
                            threads: 0,
                            batched: None,
                            ms: 0.0,
                        })
                    }
                };
                let mut acc: Vec<Vec<f32>> = Vec::with_capacity(n);
                let mut cacheable: Vec<bool> = Vec::with_capacity(n);
                if model.batched_enabled() {
                    // 🌟 [EMBED-BATCHED] 진짜 배치 순전파. 실패한 묶음만 per-text 로 대체합니다.
                    let opts = crate::models::embedding::BatchOpts::default();
                    let mut st = crate::models::embedding::BatchRunStats::default();
                    for part in miss_for_compute.chunks(RAM_CHUNK) {
                        match model.embed_many_with_stats(part, &opts) {
                            Ok((v, s)) => {
                                st.texts += s.texts;
                                st.seqs += s.seqs;
                                st.micro_batches += s.micro_batches;
                                st.padded_batches += s.padded_batches;
                                st.real_tokens += s.real_tokens;
                                st.computed_tokens += s.computed_tokens;
                                cacheable.extend(std::iter::repeat(true).take(v.len()));
                                acc.extend(v);
                            }
                            Err(e) => {
                                println!(
                                    "[EMBED-BATCHED] ⚠️ 배치 순전파 실패({}). 이 묶음 {}건은 per-text 경로로 계산하고 캐시에는 넣지 않습니다.",
                                    e, part.len()
                                );
                                let v = model
                                    .embed_batch(part)
                                    .map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;
                                cacheable.extend(std::iter::repeat(false).take(v.len()));
                                acc.extend(v);
                            }
                        }
                    }
                    Ok(Computed { vecs: acc, cacheable, threads: 0, batched: Some(st), ms: started.elapsed().as_secs_f64() * 1000.0 })
                } else {
                    // 관측에 쓸 실제 스레드 수를 미리 확보합니다.
                    let threads = model.last_thread_count(n.min(RAM_CHUNK));
                    for part in miss_for_compute.chunks(RAM_CHUNK) {
                        let v = model
                            .embed_batch(part)
                            .map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;
                        cacheable.extend(std::iter::repeat(true).take(v.len()));
                        acc.extend(v);
                    }
                    Ok(Computed { vecs: acc, cacheable, threads, batched: None, ms: started.elapsed().as_secs_f64() * 1000.0 })
                }
            },
        ).await??;

        // 🌟 [CROSSOVER] 두 축을 함께 관측합니다.
        //    ① 총 순간 점유 → embedding_budget_mb 에 반영 (가중치 축)
        //    ② 스레드당 점유 → adaptive_thread_count 에 반영 (activation 축, per-text 경로만)
        self.observe_activation_headroom(free_before);
        if computed.threads > 0 && miss_len > 0 {
            crate::models::embedding::EmbeddingModel::record_unit_activation(
                free_before,
                self.get_free_vram_mb(),
                computed.threads,
            );
        }
        if miss_len >= 2 {
            match computed.batched {
                Some(st) => println!(
                    "[EMBED-PROF] 배치 순전파 {}건 ({}청크) | 마이크로배치 {}개 (패딩 {}) | 실토큰 {} / 연산토큰 {} | {:.0}ms ({:.0}건/s){}",
                    miss_len,
                    st.seqs,
                    st.micro_batches,
                    st.padded_batches,
                    st.real_tokens,
                    st.computed_tokens,
                    computed.ms,
                    miss_len as f64 * 1000.0 / computed.ms.max(0.001),
                    if pinned { " | 고정 계층" } else { "" }
                ),
                None => println!(
                    "[EMBED-PROF] per-text 순전파 {}건 | 스레드 {} | {:.0}ms ({:.0}건/s){}",
                    miss_len,
                    computed.threads,
                    computed.ms,
                    miss_len as f64 * 1000.0 / computed.ms.max(0.001),
                    if pinned { " | 고정 계층" } else { "" }
                ),
            }
        }

        // ── ④ 결과 흩뿌리기 + 캐시 적재 (0벡터·대체 경로 벡터는 적재하지 않음) ──
        let mut inserts: Vec<(String, std::sync::Arc<[f32]>)> = Vec::with_capacity(miss_len);
        let mut failed = 0usize;
        for (mi, vec) in computed.vecs.into_iter().enumerate() {
            if mi >= miss_slots.len() { break; }
            let ok = is_cacheable(&vec);
            if !ok {
                failed += 1;
            } else if computed.cacheable.get(mi).copied().unwrap_or(false) {
                inserts.push((miss_texts[mi].clone(), std::sync::Arc::from(vec.as_slice())));
            }
            for &slot in &miss_slots[mi] {
                out[slot] = Some(vec.clone());
            }
        }
        GRANITE_TEXT_CACHE.insert_many(space, inserts, pinned);
        if failed > 0 {
            println!(
                "[MODEL] ⚠️ [EMBED FAIL] {}건이 0벡터로 반환되었습니다. 길이 계약을 위해 0벡터로 채우되 캐시에는 넣지 않습니다.",
                failed
            );
        }

        // 모델이 사라졌거나 embed_batch 가 짧게 반환한 자리는 0벡터로 채웁니다.
        // (호출부는 전부 이 길이를 전제로 인덱싱하므로 길이 보존이 계약입니다)
        Ok(out.into_iter().map(|v| v.unwrap_or_else(|| vec![0.0; EMBED_DIM])).collect())
    }
}
