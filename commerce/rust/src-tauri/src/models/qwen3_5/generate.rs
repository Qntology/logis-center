use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use std::io::Write;

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

pub struct Qwen3_5GenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    pre_processor: Option<Qwen3VLProcessor>,
    qwen3_5: Qwen3_5Model,
    device: Device,
    eos_token_id: u32,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

pub struct GenerationResult {
    pub text: String,
    pub is_finished: bool,
    pub next_offset: usize,
    pub last_token: u32,
}

impl Qwen3_5GenerateModel {
    // [CRITICAL FIX] 더 이상 쓰이지 않는 safetensors 로딩용 init 함수(new_from_vb 호출부) 완전 삭제!

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);
        
        let device = get_device(device);
        let model_dir = std::path::Path::new(model_file).parent().unwrap().to_str().unwrap();
        let chat_template = ChatTemplate::init(model_dir)?;

        // 초기 파싱 전용 임시 Reader
        let mut reader = std::io::Cursor::new(&mmap_arc[..]);
        let ct_for_tokenizer = gguf_file::Content::read(&mut reader)?;
        let mut reader2 = std::io::Cursor::new(&mmap_arc[..]);
        let model_gguf = Gguf::new(ct_for_tokenizer, &mut reader2, device.clone());
        
        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        
        let (pre_processor, mut mmproj_gguf): (Option<Qwen3VLProcessor>, Option<Gguf<std::fs::File>>) = if let Some(mmproj_f) = mmproj_file {
            let reader3 = std::fs::File::open(mmproj_f)?;
            let mut reader3_cl = std::fs::File::open(mmproj_f)?; 
            let content3 = gguf_file::Content::read(&mut reader3_cl)?;
            
            // 👇 [CRITICAL FIX] 여기를 다시 device.clone()으로 되돌립니다!
            let mmproj_gguf = Gguf::new(content3, reader3, device.clone()); 
            
            let processor = Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?;
            (Some(processor), Some(mmproj_gguf))
        } else { (None, None) };

        let eos_token_id = model_gguf.get_matedata("tokenizer.ggml.eos_token_id")?.to_u32()?;
        
        // Mmap 원본 그대로 넘김 (Content 파싱은 모델 내부에서 직접 실행)
        let qwen3_5 = Qwen3_5Model::new_from_gguf(mmap_arc.clone(), mmproj_gguf.as_mut(), &device)?;
        
        let stem = std::path::Path::new(model_file).file_stem().and_then(|s| s.to_str()).unwrap_or("qwen3.5");
        Ok(Self {
            chat_template, tokenizer, pre_processor, qwen3_5, device, eos_token_id,
            model_name: stem.to_string(), 
            repeat_penalty: 1.2, // 반복 루프 탈출을 위해 페널티 상향
            repeat_last_n: 64,
        })
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String> {
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
        let mut input_ids = self.tokenizer.text_encode(mes_text, &self.device)?;
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
        let session_id = "default_session"; 

        for _ in 0..sample_len {
            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values,
                image_grid_thw,
                cur_pixel_values_video,
                video_grid_thw,
                seqlen_offset,
                session_id,
            ).await?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
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
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cur_pixel_values = None;
            cur_pixel_values_video = None;
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen3_5.clear_cache();
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_5.clear_cache();
    }

    pub async fn generate_part(
        &mut self,
        mes: &ChatCompletionParameters,
        is_continuation: bool,
        last_offset: usize,
        last_token: Option<u32>,
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
            let mes_render = self.chat_template.apply_chat_template(mes)?;
            let input = self.pre_processor.as_ref().unwrap().process_info(mes, &mes_render)?;
            let ids = self.tokenizer.text_encode(input.replace_text, &self.device)?;
            (ids, 0, input.pixel_values, input.image_grid_thw, input.pixel_values_video, input.video_grid_thw)
        } else {
            let ids = Tensor::from_vec(vec![last_token.unwrap()], (1, 1), &self.device)?;
            (ids, last_offset, None, None, None, None)
        };

        let mut seq_len = input_ids.dim(1)?;
        let mut is_finished = false;
        let mut final_token = 0;

        let session_id = "default_session"; 
        
        println!("\n[AI Thinking...]");

        for _ in 0..sample_len {
            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values.as_ref(),
                cur_image_thw.as_ref(),
                cur_pixel_values_video.as_ref(),
                cur_video_thw.as_ref(),
                seqlen_offset,
                session_id,
            ).await?;
            
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // 반복(Repetition) 방지 페널티 적용
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
            
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            final_token = next_token;

            // 콘솔에 AI가 생성한 단어를 실시간으로 출력
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print!("{}", piece);
                let _ = std::io::stdout().flush();
            }

            if next_token == self.eos_token_id {
                is_finished = true;
                break;
            }

            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cur_pixel_values = None;
            cur_image_thw = None;
            cur_pixel_values_video = None;
            cur_video_thw = None;
        }
        println!(); // 답변이 끝나면 줄바꿈 처리

        let res_text = self.tokenizer.token_decode(generate)?;

        if is_finished {
            self.qwen3_5.clear_cache();
        }

        Ok(GenerationResult {
            text: res_text,
            is_finished,
            next_offset: seqlen_offset,
            last_token: final_token,
        })
    }
}