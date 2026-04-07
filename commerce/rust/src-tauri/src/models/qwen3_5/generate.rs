use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart, ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText, ImageURL};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use std::io::Write;
// [추가] 취소 토큰 및 비동기 상태 처리를 위한 패키지
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use crate::{
    chat_template::ChatTemplate,
    models::{
        common::gguf::Gguf,
        qwen3_5::{config::Qwen3_5Config, model::Qwen3_5Model},
        qwen3vl::processor::Qwen3VLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct GenerationResult {
    pub text: String,
    pub is_finished: bool,
    pub next_offset: usize,
    pub last_token: u32,
}

pub struct Qwen3_5GenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    pub pre_processor: Option<Qwen3VLProcessor>,
    qwen3_5: Qwen3_5Model,
    device: Device,
    pub eos_token_id: u32,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Qwen3_5GenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let eos_token_id = cfg.text_config.eos_token_id;
        let qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5,
            device,
            eos_token_id,
            model_name: model_name.to_string(),
            repeat_penalty: 1.01,
            repeat_last_n: 64,
        })
    }

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        // 🌟 [핵심 변경] 단순 File 오픈이 아닌 Mmap으로 메모리에 매핑합니다.
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);
        
        let mut reader = std::io::Cursor::new(&mmap_arc[..]);
        let content = gguf_file::Content::read(&mut reader)?;
        
        // 🌟 [수정] Content는 Clone이 불가능하므로, 메타데이터만 가볍게 한 번 더 읽어서 Arc에 담아줍니다.
        let mut reader_clone = std::io::Cursor::new(&mmap_arc[..]);
        let content_clone = gguf_file::Content::read(&mut reader_clone)?;
        let ct_arc = std::sync::Arc::new(content_clone);
        
        let device = get_device(device);
        let mut model_gguf = Gguf::new(content, reader, device.clone());
        let _chat_template_str = model_gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        
        let model_dir = std::path::Path::new(model_file).parent().unwrap().to_str().unwrap();
        let chat_template = ChatTemplate::init(model_dir)?;

        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        let (pre_processor, mut mmproj_gguf) = if let Some(mmproj_f) = mmproj_file {
            let mut reader = std::fs::File::open(mmproj_f)?;
            let content = gguf_file::Content::read(&mut reader)?;
            let mmproj_gguf = Gguf::new(content, reader, device.clone());
            let processor = Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?;
            (Some(processor), Some(mmproj_gguf))
        } else {
            (None, None)
        };

        let eos_token_id = model_gguf
            .get_matedata("tokenizer.ggml.eos_token_id")?
            .to_u32()?;
            
        // 🌟 [수정] 밥줄(Mmap, Ct)을 쥐여준 채로 모델을 생성합니다.
        let qwen3_5 = Qwen3_5Model::new_from_gguf(
            &mut model_gguf, 
            mmproj_gguf.as_mut(), 
            &device,
            Some(mmap_arc.clone()), // 🌟 [핵심 수정] .clone()을 붙여서 포인터만 우아하게 넘겨줍니다!
            Some(ct_arc)
        )?;
        
        let stem = std::path::Path::new(model_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
            
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_5,
            device,
            eos_token_id,
            model_name: stem.to_string(),
            repeat_penalty: 1.0, 
            repeat_last_n: 1024,
        })
    }

    // 🌟 [핵심 변경점] 0.6B 모델과 완벽하게 동일한 파라미터 구조를 가지도록 시그니처 수정
    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>, 
        session_id: Option<String>, 
        kv_name: Option<String>
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.4);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mut logit_processor =
            get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(20), seed);
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw): (String, Option<Tensor>, Option<Tensor>, Option<Tensor>, Option<Tensor>) =
            if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(&mes, &mes_render)?;
                (
                    input.replace_text,
                    input.pixel_values,
                    input.image_grid_thw,
                    input.pixel_values_video,
                    input.video_grid_thw,
                )
            } else {
                (mes_render, None, None, None, None)
            };
        let mut input_ids = self.tokenizer.text_encode(mes_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let pixel_values: Option<&Tensor> = pixel_values.as_ref();
        let image_grid_thw: Option<&Tensor> = image_grid_thw.as_ref();
        let pixel_values_video: Option<&Tensor> = pixel_values_video.as_ref();
        let video_grid_thw: Option<&Tensor> = video_grid_thw.as_ref();
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        
        let mut cur_pixel_values = pixel_values;
        let mut cur_pixel_values_video = pixel_values_video;

        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(123);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(999999);
        let is_strict_json = mes_text.contains("/no_think") || mes_text.contains("RETURN JSON ONLY");
        let mut gen_text_buffer = String::new(); // 깊이 추적용

        for i in 0..sample_len {
            // [취소 감지] 사용자가 멈춤 버튼을 눌렀을 때 즉시 중단
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }

            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values,
                image_grid_thw,
                cur_pixel_values_video,
                video_grid_thw,
                seqlen_offset,
                session_id.clone(), 
                kv_name.clone()     
            ).await?;
            
            let mut logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            // 👇 [추가] DENSE-BIAS: 첫 토큰 강제 주입 로직
            if i == 0 {
                // Tensor를 Vec로 빼서 확률 직접 조작
                let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
                let len = logits_vec.len();
                
                if (self.eos_token_id as usize) < len { logits_vec[self.eos_token_id as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                
                if (open_bracket_id as usize) < len {
                    let boost = if is_strict_json { 10000.0 } else { 20.0 };
                    logits_vec[open_bracket_id as usize] += boost;
                }
                
                // 조작된 확률을 다시 Tensor로 복구 (Device 주의)
                logits = Tensor::from_vec(logits_vec, (len,), &self.device)?;
            }

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = generate.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &generate[start_at..],
                )?
            };
            
            let mut next_token = logit_processor.sample(&logits)?;

            // 👇 [추가] FORCE-START: AI가 헛소리를 하려 해도 멱살 잡고 '{' 로 강제 시작
            if i == 0 && (is_strict_json || next_token == self.eos_token_id) {
                next_token = open_bracket_id;
            }

            generate.push(next_token);
            
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print!("{}", piece);
                let _ = std::io::stdout().flush();
                gen_text_buffer.push_str(&piece);
                
                // 👇 [수정] 단순 contains 대신 qwen의 정밀한 중첩 깊이 추적 로직 이식
                if gen_text_buffer.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with('}') {
                        println!("\n[DEBUG-GEN] Balanced JSON detected (Depth 0). Stopping at token {}.", i + 1);
                        break;
                    }
                }
            }

            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            // 🌟 [최적화 복구] 기준점(input_ids)은 반드시 GPU에 있어야 이후의 1,000GB/s 매트릭스 연산이 GPU에서 돕니다!
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cur_pixel_values = None;
            cur_pixel_values_video = None;
        }
        println!(); // 🌟 [추가] 출력이 끝나면 줄바꿈

        let res = self.tokenizer.token_decode(generate)?;
        
        // 🌟 [핵심] 생성 종료 후, RAM에 남아있는 KV 캐시 텐서들을 백그라운드 IO 큐를 통해 SSD로 암호화 백업!
        if let Some(s_id) = &session_id {
            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
            println!("[GEN-SAVE] Qwen 3.5 active KV blocks flushed to disk.");
        }

        self.qwen3_5.clear_cache();
        Ok(res)
    }

    // `generate_part` 도 동일하게 캐싱 파이프라인 적용
    pub async fn generate_part(
        &mut self,
        mes: &ChatCompletionParameters,
        is_continuation: bool,
        last_offset: usize,
        last_token: Option<u32>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> Result<GenerationResult> {
        let mut logit_processor = get_logit_processor(
            Some(mes.temperature.unwrap_or(0.1) as f32), 
            Some(mes.top_p.unwrap_or(0.9) as f32), 
            Some(20), 
            mes.seed.unwrap_or(32768) as u64
        );
        
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        let (mut input_ids, mut seqlen_offset, mut cur_pixel_values, mut cur_image_thw, mut cur_pixel_values_video, mut cur_video_thw) = if !is_continuation {
            self.clear_kv_cache();

            let mes_render = self.chat_template.apply_chat_template(mes)?;
            let (text, px_vals, img_thw, vid_px_vals, vid_thw) = if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(mes, &mes_render)?;
                (input.replace_text, input.pixel_values, input.image_grid_thw, input.pixel_values_video, input.video_grid_thw)
            } else {
                (mes_render, None, None, None, None)
            };
            
            let ids = self.tokenizer.text_encode(text, &self.device)?;
            (ids, 0, px_vals, img_thw, vid_px_vals, vid_thw)
        } else {
            let ids = Tensor::from_vec(vec![last_token.unwrap()], (1, 1), &self.device)?;
            (ids, last_offset, None, None, None, None)
        };

        let mut seq_len = input_ids.dim(1)?;
        let mut is_finished = false;
        let mut final_token = 0;
        
        println!("\n[AI Thinking...]");

        // 🚨 [여기에 있던 중복된 Prefill Chunking while 루프를 완전히 삭제했습니다!] 🚨

        // 상태 추적 및 Logit 조작용 (DENSE-BIAS 및 JSON 추적)
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(123);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v| v.first().cloned()).unwrap_or(999999);
        let mut gen_text_buffer = String::new();

        for i in 0..sample_len {
            // 🌟 14,928개의 전체 input_ids가 한 번에 넘어가고, model.rs 내부에서 안전하게 256개씩 청킹됩니다!
            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values.as_ref(),
                cur_image_thw.as_ref(),
                cur_pixel_values_video.as_ref(),
                cur_video_thw.as_ref(),
                seqlen_offset,
                session_id.clone(),
                kv_name.clone()
            ).await?;
            
            let mut logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // 첫 턴 방어 로직 (EOS 출력 방지 및 강제 괄호 시작 확률업)
            if i == 0 {
                let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
                let len = logits_vec.len();
                if (self.eos_token_id as usize) < len { logits_vec[self.eos_token_id as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                if (open_bracket_id as usize) < len { logits_vec[open_bracket_id as usize] += 10000.0; }
                logits = Tensor::from_vec(logits_vec, (len,), &self.device)?;
            }

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = generate.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, &generate[start_at..])?
            };
            
            let mut next_token = logit_processor.sample(&logits)?;
            
            // 강제 방어
            if i == 0 && next_token == self.eos_token_id {
                next_token = open_bracket_id;
            }

            generate.push(next_token);
            final_token = next_token;

            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print!("{}", piece);
                let _ = std::io::stdout().flush();
                gen_text_buffer.push_str(&piece);

                // 중첩 깊이 추적 기반 조기 종료
                if gen_text_buffer.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with('}') {
                        println!("\n[DEBUG-GEN] Balanced JSON detected. Stopping.");
                        is_finished = true;
                        break;
                    }
                }
            }

            if next_token == self.eos_token_id {
                is_finished = true;
                break;
            }

            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            
            // 첫 번째 순회 이후에는 비전 텐서들을 초기화하여 오버헤드 방지
            cur_pixel_values = None;
            cur_image_thw = None;
            cur_pixel_values_video = None;
            cur_video_thw = None;
        }
        println!(); 

        let res_text = self.tokenizer.token_decode(generate)?;
        
        if let Some(s_id) = &session_id {
            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
        }

        Ok(GenerationResult {
            text: res_text,
            is_finished,
            next_offset: seqlen_offset,
            last_token: final_token,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_5.clear_cache();
    }
}