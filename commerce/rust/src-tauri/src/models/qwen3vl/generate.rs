use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            model::Qwen3VLModel,
            quantized_model::QuantizedQwen3VLModel,
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
    openai_types::ChatCompletionParameters,
};

enum ModelVariant {
    Standard(Qwen3VLModel),
    Quantized(QuantizedQwen3VLModel),
}

pub struct Qwen3VLGenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    qwen3_vl: ModelVariant,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3VLGenerationConfig,
    model_name: String,
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
        
        // Check for GGUF files first
        let gguf_files = find_type_files(path, "gguf")?;
        println!("[DEBUG] Found {} GGUF files in {}", gguf_files.len(), path);
        
        let qwen3_vl = if !gguf_files.is_empty() {
            // Find mmproj (vision) and main model
            let mmproj_path = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
            let model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned();

            if let (Some(mmproj), Some(main)) = (mmproj_path, model_path.clone()) {
                println!("[DEBUG] Loading GGUF Vision from: {:?}", mmproj);
                println!("[DEBUG] Loading GGUF Main from: {:?}", main);
                
                let mut mmproj_file = std::fs::File::open(&mmproj)?;
                println!("[DEBUG] Reading mmproj content...");
                let mmproj_content = gguf_file::Content::read(&mut mmproj_file)?;
                println!("[DEBUG] mmproj content read successfully.");
                
                let mut main_file = std::fs::File::open(&main)?;
                println!("[DEBUG] Reading main model content...");
                let main_content = gguf_file::Content::read(&mut main_file)?;
                println!("[DEBUG] main model content read successfully.");
                
                println!("[DEBUG] Initializing QuantizedQwen3VLModel...");
                let model = QuantizedQwen3VLModel::new(&cfg, &main_content, &mut main_file, &mmproj_content, &mut mmproj_file, &device, dtype)?;
                println!("[DEBUG] QuantizedQwen3VLModel initialized.");
                ModelVariant::Quantized(model)
            } else if let Some(main) = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }) {
                 // Fallback if only one file found (maybe combined?)
                 println!("Loading Single GGUF from: {:?}", main);
                 let mut file = std::fs::File::open(&main)?;
                 let content = gguf_file::Content::read(&mut file)?;
                 // Pass same file for both if combined? Or empty?
                 // We'll duplicate for now, assuming combined.
                 let mut file2 = std::fs::File::open(&main)?; 
                 let content2 = gguf_file::Content::read(&mut file2)?;
                 let model = QuantizedQwen3VLModel::new(&cfg, &content, &mut file, &content2, &mut file2, &device, dtype)?;
                 ModelVariant::Quantized(model)
            } else {
                 return Err(anyhow!("No valid GGUF model found"));
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig =
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
            model_name: "qwen3vl".to_string(),
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String> {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem as f32,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p as f32,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // Make input mutable to take ownership of fields
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        
        // Take ownership of large tensors
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take(); // Clone needed if reused? No, standard Qwen3VLModel usually expects Option<&Tensor> or Tensor.
        // Wait, forward methods expect Option<&Tensor>. We need to own the Tensor and pass reference.
        // But to drop it, we must own it in a mutable Option.
        
        // We'll keep image_grid_thw as Option<Tensor> to pass reference, but pixel_values is the big one.
        // Let's redefine image_grid_thw variable.
        let image_grid_thw_tensor = image_grid_thw; // Move ownership
        
        let mut pixel_values_video = input.pixel_values_video.take();
        let video_grid_thw_tensor = input.video_grid_thw.take();
        
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);

        // Wrap generation in a closure/block to ensure cleanup happens
        let generation_result: Result<Vec<u32>> = (|| {
            let mut generate = Vec::new();
            for _ in 0..sample_len {
                let logits = match &mut self.qwen3_vl {
                    ModelVariant::Standard(m) => m.forward(
                        &input_ids,
                        pixel_values.as_ref(),
                        image_grid_thw_tensor.as_ref(),
                        pixel_values_video.as_ref(),
                        video_grid_thw_tensor.as_ref(),
                        Some(&cache_position),
                        seqlen_offset,
                    )?,
                    ModelVariant::Quantized(m) => m.forward(
                        &input_ids,
                        pixel_values.as_ref(),
                        image_grid_thw_tensor.as_ref(),
                        pixel_values_video.as_ref(),
                        video_grid_thw_tensor.as_ref(),
                        Some(&cache_position),
                        seqlen_offset,
                    )?,
                };
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
                
                // CRITICAL: Drop pixel_values immediately after first iteration (prefill)
                // This frees the massive vision tensor from GPU memory.
                if pixel_values.is_some() {
                    pixel_values = None;
                    pixel_values_video = None;
                }
            }
            Ok(generate)
        })();

        // Always clear cache to prevent VRAM leaks
        match &mut self.qwen3_vl {
            ModelVariant::Standard(m) => m.clear_kv_cache(),
            ModelVariant::Quantized(m) => m.clear_kv_cache(),
        }

        let generate = generation_result?;
        let res = self.tokenizer.token_decode(generate)?;
        Ok(res)
    }
}
