use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;

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
        let mut reader = std::fs::File::open(model_file)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let device = get_device(device);
        let mut model_gguf = Gguf::new(content, reader, device.clone());

        let _chat_template_str = model_gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        
        let model_dir = std::path::Path::new(model_file).parent().unwrap().to_str().unwrap();
        let chat_template = ChatTemplate::init(model_dir)?;

        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        let (pre_processor, mut mmproj_gguf): (Option<Qwen3VLProcessor>, Option<Gguf<std::fs::File>>) = if let Some(mmproj_f) = mmproj_file {
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
        let qwen3_5 = Qwen3_5Model::new_from_gguf(&mut model_gguf, mmproj_gguf.as_mut(), &device)?;
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
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String> {
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

        for _ in 0..sample_len {
            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values,
                image_grid_thw,
                cur_pixel_values_video,
                video_grid_thw,
                seqlen_offset,
            )?;
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
}
