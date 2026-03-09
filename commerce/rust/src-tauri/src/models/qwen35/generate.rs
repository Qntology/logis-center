use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen35::{config::Qwen3_5Config, model::Qwen3_5Model},
        qwen3vl::processor::Qwen3VLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device, get_dtype, get_logit_processor,
    },
    openai_types::ChatCompletionParameters,
};

pub struct Qwen3_5GenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Qwen3VLProcessor,
    pub qwen3_5: Qwen3_5Model,
    pub device: Device,
    pub eos_token_id: u32,
    pub model_name: String,
}

impl Qwen3_5GenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>, force_text_only: bool) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        
        let mut model_list = find_type_files(path, "st")?; // Use .st files as per split structure
        
        if force_text_only {
            // Remove vision.st from the loading list to save VRAM and load as "Small" model
            model_list.retain(|f| !f.contains("vision.st"));
            println!("[MODEL-QWEN35] Loading in TEXT-ONLY mode (Small).");
        } else {
            println!("[MODEL-QWEN35] Loading in FULL mode (Large/Vision).");
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let eos_token_id = cfg.text_config.eos_token_id;
        let qwen3_5 = Qwen3_5Model::new(vb, cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_5,
            device,
            eos_token_id,
            model_name: "qwen3.5".to_string(),
        })
    }

    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(
            mes.temperature.map(|t| t as f32), 
            mes.top_p.map(|p| p as f32), 
            None, 
            seed
        );
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        
        let mut input_ids = self.tokenizer.text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        
        let mut pixel_values = input.pixel_values.as_ref();
        let image_grid_thw = input.image_grid_thw.as_ref();
        let mut pixel_values_video = input.pixel_values_video.as_ref();
        let video_grid_thw = input.video_grid_thw.as_ref();
        
        let mut generate = Vec::new();
        let mut gen_text = String::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        for _i in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let logits = self.qwen3_5.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                seqlen_offset,
            )?;

            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            
            if next_token == self.eos_token_id {
                break;
            }

            generate.push(next_token);
            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);

            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            
            // Vision inputs only used in first step (prefill)
            pixel_values = None;
            pixel_values_video = None;
        }

        self.qwen3_5.clear_cache();
        Ok(gen_text)
    }
}
