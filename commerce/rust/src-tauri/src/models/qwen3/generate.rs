use crate::openai_types::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::utils::{
    find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

pub struct Qwen3GenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    qwen3: Qwen3Model,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl Qwen3GenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let qwen3 = Qwen3Model::new(&cfg, vb)?;
        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;

        Ok(Qwen3GenerateModel {
            chat_template,
            tokenizer,
            qwen3,
            device: device.clone(),
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
            model_name: "qwen3".to_string(),
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(top_k), seed);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        for _ in 0..sample_len {
            let logits = self.qwen3.forward(Some(&input_ids), None, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen3.clear_kv_cache();
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3.clear_kv_cache();
    }
}
