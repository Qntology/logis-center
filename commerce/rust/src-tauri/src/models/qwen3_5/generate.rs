use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::path::{Path, PathBuf};
use std::fs;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        common::gguf::Gguf,
        qwen3_5::{config::Qwen3_5Config, model::Qwen3_5Model, quantized_model::QuantizedQwen3_5Model},
        qwen3vl::processor::Qwen3VLProcessor,
        // [FIX] qwen3vl의 글로벌 IO 매니저를 가져와서 공유합니다.
        qwen3vl::generate::{wait_for_global_io, SLOT_MANAGER},
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

// [NEW] Qwen3_5 모델도 Standard와 Quantized 변형을 모두 지원하도록 래핑
#[derive(Clone)]
pub enum Qwen3_5ModelVariant {
    Standard(Qwen3_5Model),
    QuantizedText(QuantizedQwen3_5Model),
}

impl Qwen3_5ModelVariant {
    pub async fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        match self {
            Self::Standard(m) => {
                // 기존 Standard 모델은 동기적으로 작동하므로 바로 실행
                m.forward(input_ids, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw, seqlen_offset)
            },
            Self::QuantizedText(m) => {
                // Quantized 모델은 입력 토큰을 임베딩으로 변환 후 넘김
                let target_device = if m.is_forced_cpu { Device::Cpu } else { crate::utils::get_cuda_device(m.device_id) };
                let input_ids = if !input_ids.device().same_device(&target_device) { input_ids.to_device(&target_device)? } else { input_ids.clone() };
                
                let inputs_embeds = m.embed_tokens.forward(&input_ids)?;
                m.forward(&inputs_embeds, seqlen_offset, session_id, kv_name).await
            }
        }
    }

    pub fn get_kv_len(&self) -> usize {
        match self {
            Self::QuantizedText(m) => m.get_kv_len(),
            _ => 0,
        }
    }

    pub fn clear_kv_cache(&mut self) {
        if let Self::QuantizedText(m) = self {
            for layer in m.layers.iter_mut() {
                if let Some(attn) = &mut layer.self_attn { attn.clear(); }
                if let Some(attn) = &mut layer.linear_attn { attn.clear(); }
            }
            m.current_kv_len = 0;
        }
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        if let Self::QuantizedText(m) = self { m.truncate_kv_cache(len)?; }
        Ok(())
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> {
        if let Self::QuantizedText(m) = self { m.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name)?; }
        Ok(())
    }

    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        if let Self::QuantizedText(m) = self { m.force_flush_all_active_blocks(session_id, kv_name).await?; }
        Ok(())
    }
}

pub struct Qwen3_5GenerateModel<'a> {
    pub chat_template: ChatTemplate<'a>,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Option<Qwen3VLProcessor>,
    pub qwen3_5: Qwen3_5ModelVariant,
    pub device: Device,
    pub text_device_id: usize,
    pub eos_token_id: u32,
    pub model_name: String,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub kv_root: PathBuf,
}

impl<'a> Qwen3_5GenerateModel<'a> {
    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
        text_device_id: usize,
        baking_only: bool,
    ) -> Result<Self> {
        let device = get_device(device);
        
        let mut reader = std::fs::File::open(model_file)?;
        let content = Arc::new(gguf_file::Content::read(&mut reader)?);
        
        // [FIX] Gguf Metadata Parsing
        let mut model_gguf = Gguf::new((*content).clone(), reader, device.clone());
        let chat_template_str = model_gguf.get_matedata("tokenizer.chat_template")?.to_string()?.clone();
        let chat_template = ChatTemplate::str_init(&chat_template_str)?;
        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        
        let eos_token_id = model_gguf.get_matedata("tokenizer.ggml.eos_token_id")?.to_u32()?;
        let stem = std::path::Path::new(model_file).file_stem().and_then(|s| s.to_str()).unwrap_or("qwen3.5");

        // [FIX] 임시로 설정 파일을 하드코딩하거나 GGUF에서 추출해야 함
        // 여기서는 qwen3_5 구조를 가정하고 QuantizedText 모델로 감싸줍니다.
        let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(model_file)?)? };
        
        // 실제로는 Qwen3_5TextConfig가 GGUF 메타데이터에서 동적으로 생성되어야 합니다.
        // 아래는 qwen3vl과 동일한 매커니즘을 타기 위한 래퍼(Wrapper) 처리입니다.
        let qwen3_5 = if model_file.ends_with("gguf") && !baking_only {
            // 원본 qwen3_5.forward()를 타는 Standard 모드
            Qwen3_5ModelVariant::Standard(Qwen3_5Model::new_from_gguf(&mut model_gguf, None, &device)?)
        } else {
            // [TODO] Qwen3.5 전용 TextConfig 파싱 로직 추가 필요
            return Err(anyhow!("Advanced Quantized mode requires Qwen3_5TextConfig implementation."));
        };

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: None,
            qwen3_5,
            device,
            text_device_id,
            eos_token_id,
            model_name: stem.to_string(),
            repeat_penalty: 1.05, // [FIX] 패널티 완화
            repeat_last_n: 64,
            kv_root: crate::utils::paths::get_kv_dir(None),
        })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<usize> {
        self.qwen3_5.clear_kv_cache();

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let total_toks = input_ids.dim(1)?;

        self.qwen3_5.forward(&input_ids, None, None, None, None, 0, session_id.clone(), _kv_name.clone()).await?;
        
        if let Some(s_id) = &session_id {
            let path = self.kv_root.join(s_id);
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&input_ids.flatten_all()?.to_vec1::<u32>()?)?)?;
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            println!("[PREFILL-WAIT] Waiting for SSD write to complete...");
            wait_for_global_io().await; 
        }
        Ok(total_toks)
    }

    pub async fn prefill_chunk(&mut self, text: String) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.qwen3_5.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.device)?;
        
        self.qwen3_5.forward(&chunk_ids, None, None, None, None, current_pos, None, None).await?;
        Ok(chunk_size)
    }
    
    // [CRITICAL FIX] qwen3vl의 디코딩 매커니즘 전체 적용 (SSD 복원, 괄호 오버라이드 포함)
    pub async fn generate_advanced(
        &mut self,
        mes: ChatCompletionParameters,
        cancel_flag: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<ChatCompletionResponse> {
        let mut is_reference_snapshot = false;

        // ====================================================================
        // 0. [CONTEXT-RESTORATION] SSD 스냅샷 불러오기 (이전 대화 캐시 복원)
        // ====================================================================
        if let Some(s_id) = &session_id {
            let snapshot_root = self.kv_root.join(s_id);
            let paths_to_try = vec![
                snapshot_root.join("inference").join("text"),
                snapshot_root.join("reference").join("text"),
                snapshot_root.clone(),
            ];

            for snapshot_path in paths_to_try {
                if snapshot_path.exists() && fs::read_dir(&snapshot_path).map(|mut d| d.next().is_some()).unwrap_or(false) {
                    println!("[GEN-LOAD] Loading existing snapshot from {:?}...", snapshot_path);
                    if snapshot_path.to_string_lossy().contains("reference") {
                        is_reference_snapshot = true;
                    }

                    let _ = self.load_kv_from_disk(&snapshot_path, None);
                    
                    if is_reference_snapshot {
                        println!("[GEN-LOAD] Reference snapshot detected. Resetting Registry Entry states for Full Prefill...");
                        let reset_reg = |reg: &crate::models::qwen3vl::quantized_model::KVRegistry| {
                            let mut entries = reg.entries.write().unwrap();
                            for (i, entry) in entries.iter_mut().enumerate() {
                                for loc in entry.location.iter_mut() { *loc = crate::models::qwen3vl::quantized_model::KVLocation::RAM; }
                                for slot in entry.slot_ids.iter_mut() { *slot = None; }
                                entry.token_start = i * 256;
                                entry.token_len = 0;
                                entry.is_dirty.fill(true);
                                let mut cache = entry.bitkv_cache.write().unwrap();
                                cache.fill(None);
                            }
                        };
                        
                        if let Qwen3_5ModelVariant::QuantizedText(m) = &mut self.qwen3_5 {
                            reset_reg(&m.registry);
                            let _ = m.truncate_kv_cache(0);
                        }
                        self.clear_kv_cache();
                    }
                    break;
                }
            }
        }

        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.4) as f32;
        let top_p = mes.top_p.unwrap_or(0.95) as f32;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(20), seed);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let f_ids = self.tokenizer.text_encode_vec(mes_render.clone(), false)?;
        let total_toks = f_ids.len();

        let kv_len = self.get_kv_len();
        let mut gen_text = String::new();
        
        // ====================================================================
        // 1. Incremental Prefill (부분 프리필) 로직
        // ====================================================================
        let (input_ids, offset) = if kv_len > 0 && !is_reference_snapshot {
            if kv_len >= total_toks {
                println!("[SKIP-PREFILL] Snapshot covers prompt. Capping offset.");
                let last_id = *f_ids.last().unwrap_or(&0);
                (Tensor::from_vec(vec![last_id], (1, 1), &self.device)?, total_toks - 1)
            } else {
                let missing_ids = f_ids[kv_len..].to_vec();
                let missing_len = missing_ids.len();
                println!("[PARTIAL-PREFILL] Restored {}. Prefilling remaining {}.", kv_len, missing_len);
                (Tensor::from_vec(missing_ids, (1, missing_len), &self.device)?, kv_len)
            }
        } else {
            (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.device)?, 0)
        };

        let total_tokens_after_prefill = offset + input_ids.dim(1)?;
        
        // SSD 읽기가 완료될 때까지 대기
        wait_for_global_io().await; 

        // ====================================================================
        // 2. 초기 프리필 실행
        // ====================================================================
        let mut logits = self.qwen3_5.forward(
            &input_ids, None, None, None, None, offset, session_id.clone(), kv_name.clone()
        ).await?;

        let mut gen_ids = vec![];
        let sample_len = mes.max_tokens.unwrap_or(1024);

        // [DENSE-MODE] 편향 토큰 준비
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(999999);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(999999);

        // ====================================================================
        // 3. 디코딩 루프
        // ====================================================================
        for i in 0..sample_len {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            
            // [CRITICAL FIX] CPU로 한 번만 다운로드 후 In-place 연산으로 PCIe 병목 제거
            let mut logits_vec = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let len = logits_vec.len();

            // Repetition Penalty
            if !gen_ids.is_empty() && self.repeat_penalty != 1.0 {
                let penalty = self.repeat_penalty;
                let mut set = std::collections::HashSet::new();
                let start_at = gen_ids.len().saturating_sub(self.repeat_last_n);
                for &t in &gen_ids[start_at..] {
                    if !set.contains(&t) && (t as usize) < len {
                        let logit = logits_vec[t as usize];
                        logits_vec[t as usize] = if logit < 0.0 { logit * penalty } else { logit / penalty };
                        set.insert(t);
                    }
                }
            }

            // 토큰 바이아스 설정 (JSON 강제, Think 패널티)
            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 100.0; }
            if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 10.0; }
            
            if i == 0 && (open_bracket_id as usize) < len {
                logits_vec[open_bracket_id as usize] += 20.0;
                if (self.eos_token_id as usize) < len { logits_vec[self.eos_token_id as usize] = -1000.0; }
            }

            // GPU 업로드 후 샘플링
            let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
            let mut next_token = logit_processor.sample(&logits_tensor)?;
            
            // [FORCE-START] JSON 강제: 첫 토큰이 버그로 EOS가 나오면 '{' 로 덮어쓰기
            if i == 0 && next_token == self.eos_token_id {
                println!("[DEBUG-GEN] EOS detected on first token. Overriding with '{{' to force JSON.");
                next_token = if open_bracket_id != 999999 { open_bracket_id } else { 123 }; // 123은 '{' 의 기본 ID
            }

            if next_token == self.eos_token_id { break; }
            
            gen_ids.push(next_token);
            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);

            // [EARLY-STOP] JSON 괄호 깊이 추적 로직 적용
            if gen_text.contains('{') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '{' { depth += 1; has_started = true; }
                    else if c == '}' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                    println!("[DEBUG-GEN] Balanced JSON detected. Stopping at token {}.", i + 1);
                    break;
                }
            }
            
            let current_pos = total_tokens_after_prefill + i as usize;
            wait_for_global_io().await; // 베이킹 백그라운드 태스크 대기
            
            logits = self.qwen3_5.forward(
                &Tensor::from_vec(vec![next_token], (1, 1), &self.device)?,
                None, None, None, None, current_pos, session_id.clone(), kv_name.clone()
            ).await?;
        }

        // [TODO] SSD 강제 플러시 호출 (세션 종료 시 잔여 블록 백업)
        if let Some(s_id) = &session_id { 
            let _ = self.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await; 
        }

        let completion_tokens = gen_ids.len() as u32;
        let res = self.tokenizer.token_decode(gen_ids)?;
        
        Ok(build_completion_response(res, &self.model_name, Some(completion_tokens), Some(total_toks as u32)))
    }

    // 기존 GenerateModel Trait 구현체 유지 (하위 호환성)
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<Box<dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>> + Send + Unpin + '_>> {
        // [수정] qwen3vl의 비동기 스트리밍 로직과 Tool Call 지원 통합 적용
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut tool_call_id = None;
            let mut tool_call_content = String::new();
            let mut generate = Vec::new();

            for _ in 0..sample_len {
                wait_for_global_io().await; // SSD 로드 대기
                let logits = self.qwen3_5.forward(
                    &input_ids, None, None, None, None, seqlen_offset, None, None
                ).await?;

                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = generate.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, &generate[start_at..])?
                };

                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);

                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() { decode_ids.extend_from_slice(&error_tokens); }
                decode_ids.push(next_token);

                let decoded_token = self.tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("stream decode error: {e}")))?;
                
                // 특수 문자 깨짐 보정
                if decoded_token.contains("") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 { error_tokens.clear(); }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                    continue;
                }
                error_tokens.clear();

                // Tool Call 태그 처리
                match decoded_token.as_str() {
                    "<tool_call>" => {
                        tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                        seqlen_offset += seq_len; seq_len = 1;
                        input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                        continue;
                    }
                    "</tool_call>" => {
                        let chunk = build_completion_chunk_response(decoded_token, &self.model_name, tool_call_id.clone(), Some(tool_call_content.clone()));
                        tool_call_id = None; tool_call_content = String::new();
                        yield Ok(chunk);
                    }
                    _ => {
                        if tool_call_id.is_some() {
                            tool_call_content.push_str(&decoded_token);
                            seqlen_offset += seq_len; seq_len = 1;
                            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                            continue;
                        } else {
                            let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                            yield Ok(chunk);
                        }
                    }
                }
                if next_token == self.eos_token_id { break; }
                
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            }
            self.qwen3_5.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }

    pub fn get_kv_len(&self) -> usize {
        self.qwen3_5.get_kv_len()
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_5.clear_kv_cache();
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        self.qwen3_5.truncate_kv_cache(len)
    }

    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> {
        self.qwen3_5.load_kv_cache(path, &self.device, 0, 128, kv_name)
    }

    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        self.qwen3_5.force_flush_all_active_blocks(session_id, kv_name).await
    }
}

// 기존 인터페이스 호환을 위한 래퍼
impl<'a> GenerateModel for Qwen3_5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        // async 컨텍스트에서 실행되도록 브릿지 역할 수행
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            self.generate_advanced(mes, None, None, None).await
        })
    }
}