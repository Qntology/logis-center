use anyhow::{Result, anyhow};
use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            native_model::NativeQwen3VLModel,
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    openai_types::ChatCompletionParameters,
};
use serde_json::Value;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::path::Path;
use half::f16;

#[derive(Clone)]
pub enum ModelVariant {
    Native(Arc<NativeQwen3VLModel>),
}

impl ModelVariant {
    pub fn forward(&self, input_ids: &[u32], pixel_values: Option<&[f16]>, grid_thw: Option<&[u32; 3]>, seqlen_offset: usize) -> Vec<f16> {
        match self {
            Self::Native(m) => m.forward(input_ids, pixel_values, grid_thw, seqlen_offset),
        }
    }
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Qwen3VLProcessor,
    pub qwen3_vl: ModelVariant,
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: Qwen3VLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: Option<usize>,
}

impl Qwen3VLGenerateModel {
    pub fn init_with_config(
        path: &str,
        tokenizer_path: Option<&str>,
        config_path: Option<&str>,
        _text_device: Option<&()>, 
        _text_device_id: usize,
        _vision_device: Option<&()>,
        _vision_device_id: usize,
        _dtype: Option<()>,
        hard_token_limit: Option<usize>,
        baking_only: bool, // Renamed from _force_text_only
    ) -> Result<Self> {
        let path_obj = Path::new(path);
        let tok_path = tokenizer_path.unwrap_or(path);
        let cfg_path = config_path.unwrap_or(path);

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        
        // [FIX] Load Text Config from Text Model Directory (path) to match weights
        let text_config_path = path_obj.join("config.json");
        let text_raw_bytes = std::fs::read(&text_config_path)?;
        let text_json: Value = serde_json::from_slice(&text_raw_bytes)?;
        
        // [FIX] Load Vision Config from Vision Model Directory (cfg_path)
        let vision_config_path = Path::new(cfg_path).join("config.json");
        let vision_raw_bytes = std::fs::read(&vision_config_path)?;
        let vision_json: Value = serde_json::from_slice(&vision_raw_bytes)?;

        // [ROBUST-CONFIG] Merge configurations: Text params from 0.6B, Vision params from 2B
        let mut vl_config: Qwen3VLConfig = if vision_json.get("text_config").is_some() {
            serde_json::from_value(vision_json)?
        } else {
            // Fallback for flat config, but we must override text params
            serde_json::from_value(vision_json.clone())?
        };

        // OVERRIDE text_config with the actual 0.6B parameters
        let correct_text_config = if text_json.get("text_config").is_some() {
             serde_json::from_value(text_json.get("text_config").unwrap().clone())?
        } else {
             // Flat config (0.6B style)
             crate::models::qwen3vl::config::Qwen3VLTextConfig {
                hidden_size: text_json.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
                intermediate_size: text_json.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072) as usize,
                num_hidden_layers: text_json.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(28) as usize,
                num_attention_heads: text_json.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                num_key_value_heads: text_json.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
                head_dim: text_json.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
                rms_norm_eps: text_json.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
                rope_theta: text_json.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1000000.0) as f32,
                vocab_size: text_json.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(151936) as usize,
                max_position_embeddings: text_json.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(40960) as usize,
                dtype: text_json.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()),
                rope_scaling: None,
            }
        };
        
        vl_config.text_config = Some(correct_text_config);
        
        // Ensure hidden_size at root matches text config for consistency
        vl_config.hidden_size = Some(vl_config.text_config.as_ref().unwrap().hidden_size);

        // [HYBRID-FILE-SELECTION] 
        // 텍스트는 path(0.6B)에서, 비전은 config_path(2B-VL)에서 가져옴
        let main_filename = if baking_only { "model-BITSERIAL_LAYER0.safetensors" } else { "model-BITSERIAL_ALL.safetensors" };
        let vision_filename = if baking_only { "mmproj-BITSERIAL_LAYER0.safetensors" } else { "mmproj-BITSERIAL_ALL.safetensors" };

        let main_path = path_obj.join(main_filename);
        // 비전 파일은 항상 Large 모델 경로(config_path)에서 탐색 시도
        let vision_root = config_path.map(Path::new).unwrap_or(path_obj);
        let vision_path = vision_root.join(vision_filename);

        println!("[MODEL] Hybrid Load -> Text: {:?}, Vision: {:?}", main_path.file_name(), vision_path.file_name());

        let main_file = std::fs::File::open(main_path)?;
        let main_mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&main_file)? });
        
        let vision_mmap = if vision_path.exists() {
            let vision_file = std::fs::File::open(vision_path)?;
            Arc::new(unsafe { memmap2::MmapOptions::new().map(&vision_file)? })
        } else {
            let placeholder_file = std::fs::File::open(&vision_config_path)?;
            Arc::new(unsafe { memmap2::MmapOptions::new().map(&placeholder_file)? })
        };

        let native_model = NativeQwen3VLModel::load(vl_config.clone(), main_mmap, vision_mmap, baking_only)?;
        let qwen3_vl = ModelVariant::Native(Arc::new(native_model));

        let pre_processor = Qwen3VLProcessor::new_native(tok_path)?;

        let generation_config_path = Path::new(cfg_path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)? 
        } else {
            Qwen3VLGenerationConfig::default()
        };

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl,
            eos_token_id1: 151643,
            eos_token_id2: 151645,
            generation_config,
            model_name: "native-qwen3vl".to_string(),
            hard_token_limit,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, _session_id: Option<String>) -> Result<String> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info_native(&mes, &mes_render)?;
        let mut all_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?;
        
        let mut seqlen_offset = self.get_kv_len();
        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            let input_ids = if i == 0 { all_ids.clone() } else { vec![*all_ids.last().unwrap()] };
            let (pixels, grid) = if i == 0 && seqlen_offset == 0 { (input.pixel_values.as_deref(), input.image_grid_thw.as_ref()) } else { (None, None) };

            let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, seqlen_offset);
            let next_id = self.sample_greedy(&logits);
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            
            all_ids.push(next_id);
            let new_word = self.tokenizer.token_decode(vec![next_id])?;
            generated_text.push_str(&new_word);
            
            seqlen_offset += input_ids.len();
        }

        Ok(generated_text)
    }

    pub fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?;
        let seqlen = self.get_kv_len();
        self.qwen3_vl.forward(&ids, None, None, seqlen);
        Ok(ids.len())
    }

    pub fn save_kv_to_disk(&self, path: &Path) -> Result<()> {
        // [NATIVE-KV-SAVE] Implement state serialization if needed
        Ok(())
    }

    pub fn load_kv_from_disk(&self, path: &Path) -> Result<()> {
        // [NATIVE-KV-LOAD] Implement state deserialization if needed
        Ok(())
    }

    fn sample_greedy(&self, logits: &[f16]) -> u32 {
        let vocab_size = 151936;
        let last_logits = &logits[logits.len() - vocab_size ..];
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for (idx, &val) in last_logits.iter().enumerate() {
            let v = val.to_f32();
            if v > max_val { max_val = v; max_idx = idx; }
        }
        max_idx as u32
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => {
                let cache = m.text_model.layers[0].kv_cache.lock().unwrap();
                if let Some((k, _)) = cache.as_ref() {
                    k.len() / (m.text_model.config.num_key_value_heads * m.text_model.config.head_dim)
                } else { 0 }
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.clear_kv_cache(),
        }
    }
}
