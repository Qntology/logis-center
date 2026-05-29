use crate::openai_types::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3::config::Qwen3GenerationConfig,
        qwen3vl::{config::Qwen3VLConfig, model::Qwen3VLModel, processor::Qwen3VLProcessor},
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct Qwen3VLGenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    qwen3_vl: Qwen3VLModel,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3GenerationConfig,
}

impl Qwen3VLGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let qwen3_vl = Qwen3VLModel::new(cfg, vb)?;
        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl,
            device,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>) -> Result<String> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut cur_pixel_values: Option<&Tensor> = input.pixel_values.as_ref();
        let mut cur_image_grid_thw: Option<&Tensor> = input.image_grid_thw.as_ref();
        let mut cur_pixel_values_video: Option<&Tensor> = input.pixel_values_video.as_ref();
        let mut cur_video_grid_thw: Option<&Tensor> = input.video_grid_thw.as_ref();
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let logits = self.qwen3_vl.forward(
                &input_ids,
                cur_pixel_values,
                cur_image_grid_thw,
                cur_pixel_values_video,
                cur_video_grid_thw,
                Some(&cache_position),
                seqlen_offset,
            )?;
            
            // 프리필(초기 문맥 파악) 통과 후 다음 글자(디코딩)부터는 무거운 비전 연산을 완전히 생략합니다.
            cur_pixel_values = None;
            cur_image_grid_thw = None;
            cur_pixel_values_video = None;
            cur_video_grid_thw = None;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;

            // 🌟 [VRAM/RAM 최적화] 15토큰마다 OS 시스템 RAM 스파이크 억제 및 반환
            if generate.len() % 15 == 0 {
                if self.device.is_cuda() { let _ = self.device.synchronize(); }

                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            }
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen3_vl.clear_kv_cache();
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_vl.clear_kv_cache();
    }
}
