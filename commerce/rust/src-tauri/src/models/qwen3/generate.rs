use crate::openai_types::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::utils::{
    find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

pub struct Qwen3GenerateModel {
    chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
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

    pub fn init_from_gguf(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        
        let gguf_files = crate::utils::find_type_files(path, "gguf")?;
        let model_file = gguf_files.first().ok_or_else(|| anyhow::anyhow!("No GGUF found"))?;
        
        let mut file = std::fs::File::open(model_file)?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        
        println!("[QWEN3-NATIVE] Dequantizing GGUF weights directly into RAM for Qwen3Model...");
        
        // GGUF 텐서 이름을 Qwen3Model 구조체 이름에 맞게 자동 번역
        let mut data = std::collections::HashMap::new();
        for (name, _) in ct.tensor_infos.iter() {
            let t = ct.tensor(&mut file, name, &device)?;
            let t = t.dequantize_f16(&device).or_else(|_| t.dequantize(&device))?.to_dtype(dtype)?;
            
            let mut new_name = name.clone();
            if let Some(rest) = name.strip_prefix("blk.") {
                let parts: Vec<&str> = rest.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let idx = parts[0];
                    let layer = if parts[1].starts_with("attn_q_norm") { parts[1].replace("attn_q_norm", "self_attn.q_norm") }
                    else if parts[1].starts_with("attn_k_norm") { parts[1].replace("attn_k_norm", "self_attn.k_norm") }
                    else if parts[1].starts_with("attn_q") { parts[1].replace("attn_q", "self_attn.q_proj") }
                    else if parts[1].starts_with("attn_k") { parts[1].replace("attn_k", "self_attn.k_proj") }
                    else if parts[1].starts_with("attn_v") { parts[1].replace("attn_v", "self_attn.v_proj") }
                    else if parts[1].starts_with("attn_output") { parts[1].replace("attn_output", "self_attn.o_proj") }
                    else if parts[1].starts_with("attn_norm") { parts[1].replace("attn_norm", "input_layernorm") }
                    else if parts[1].starts_with("ffn_norm") { parts[1].replace("ffn_norm", "post_attention_layernorm") }
                    else if parts[1].starts_with("ffn_gate") { parts[1].replace("ffn_gate", "mlp.gate_proj") }
                    else if parts[1].starts_with("ffn_up") { parts[1].replace("ffn_up", "mlp.up_proj") }
                    else if parts[1].starts_with("ffn_down") { parts[1].replace("ffn_down", "mlp.down_proj") }
                    else { parts[1].to_string() };
                    new_name = format!("model.layers.{}.{}", idx, layer);
                }
            } else if name.starts_with("token_embd") {
                new_name = name.replace("token_embd", "model.embed_tokens");
            } else if name.starts_with("output_norm") {
                new_name = name.replace("output_norm", "model.norm");
            } else if name.starts_with("output") {
                new_name = name.replace("output", "lm_head");
            }
            data.insert(new_name, t);
        }
        
        let vb = VarBuilder::from_tensors(data, dtype, &device);
        let qwen3 = Qwen3Model::new(&cfg, vb)?;
        
        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path).unwrap_or_default()).unwrap_or_default();

        Ok(Qwen3GenerateModel {
            chat_template,
            tokenizer,
            qwen3,
            device: device.clone(),
            eos_token_id1: generation_config.eos_token_id.get(0).cloned().unwrap_or(151643) as u32,
            eos_token_id2: generation_config.eos_token_id.get(1).cloned().unwrap_or(151645) as u32,
            generation_config,
            model_name: "qwen3".to_string(),
        })
    }

    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.qwen3.get_kv_cache()
    }

    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        self.qwen3.set_kv_cache(cache);
    }

    pub fn save_kv_cache(&self, session_id: &str, cache_dir: Option<String>) -> Result<()> {
        let kv_cache = self.get_kv_cache();
        let mut tensors = std::collections::HashMap::new();
        for (i, layer_cache) in kv_cache.iter().enumerate() {
            if let Some((k, v)) = layer_cache {
                tensors.insert(format!("k_{}", i), k.clone());
                tensors.insert(format!("v_{}", i), v.clone());
            }
        }
        if tensors.is_empty() { return Ok(()); }
        
        let dir = cache_dir.unwrap_or_else(|| crate::utils::paths::get_kv_dir(None).to_string_lossy().into_owned());
        std::fs::create_dir_all(&dir)?;
        let path = std::path::Path::new(&dir).join(format!("{}.safetensors", session_id));
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub fn load_kv_cache(&mut self, session_id: &str, cache_dir: Option<String>) -> Result<bool> {
        let dir = cache_dir.unwrap_or_else(|| crate::utils::paths::get_kv_dir(None).to_string_lossy().into_owned());
        let path = std::path::Path::new(&dir).join(format!("{}.safetensors", session_id));
        if !path.exists() { return Ok(false); }
        
        let tensors = candle_core::safetensors::load(&path, &self.device)?;
        let mut cache = Vec::new();
        let mut i = 0;
        loop {
            let k_key = format!("k_{}", i);
            let v_key = format!("v_{}", i);
            if let (Some(k), Some(v)) = (tensors.get(&k_key), tensors.get(&v_key)) {
                cache.push(Some((k.clone(), v.clone())));
            } else {
                break;
            }
            i += 1;
        }
        if cache.is_empty() { return Ok(false); }
        self.set_kv_cache(cache);
        Ok(true)
    }

    pub async fn prefill_only(
        &mut self,
        prompt: String,
        _cancellation_token: Option<Arc<AtomicBool>>,
        save_session_id: Option<String>,
        _task_id: Option<String>,
        cache_dir: Option<String>,
    ) -> Result<()> {
        self.clear_kv_cache();
        let input_ids = self.tokenizer.text_encode(prompt, &self.device)?;
        let seq_len = input_ids.dim(1)?;
        if seq_len == 0 { return Ok(()); }

        let _ = self.qwen3.forward(Some(&input_ids), None, 0)?;

        if let Some(session_id) = save_session_id {
            self.save_kv_cache(&session_id, cache_dir)?;
        }
        Ok(())
    }

    pub async fn generate_part(
        &mut self,
        mes: &ChatCompletionParameters,
        is_prefill: bool,
        start_len: usize,
        _task_id: Option<String>,
        load_session_id: Option<String>,
        cache_dir: Option<String>,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<String> {
        if is_prefill {
            self.clear_kv_cache();
        } else if let Some(session_id) = load_session_id {
            self.load_kv_cache(&session_id, cache_dir)?;
        }

        let temperature = mes.temperature.unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(top_k), seed);

        let mes_render = self.chat_template.apply_chat_template(mes)?;
        let f_ids = self.tokenizer.text_encode_vec(mes_render, false)?;
        
        if start_len >= f_ids.len() {
            return Err(anyhow::anyhow!("start_len exceeds or equals prompt length"));
        }

        let p_ids = &f_ids[start_len..];
        let mut input_ids = Tensor::from_vec(p_ids.to_vec(), (1, p_ids.len()), &self.device)?;
        
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = start_len;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);

        for _ in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }

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

        let res_text = self.tokenizer.token_decode(generate)?;
        
        Ok(res_text)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>) -> Result<String> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        
        // 🌟 [CRITICAL FIX] 프론트엔드/API에서 넘어온 top_k 값이 있다면 최우선 적용합니다.
        // (만약 openai_types.rs의 ChatCompletionParameters에 top_k 필드를 추가하셨다면 아래 1번 주석을 풀고, 2번 라인을 지워주세요)
        
        // 1. let top_k = mes.top_k.map(|k| k as usize).unwrap_or(self.generation_config.top_k);
        let top_k = self.generation_config.top_k; // 2. 현재는 구조체 안전성을 위해 기본값으로 유지
        
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
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }

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
