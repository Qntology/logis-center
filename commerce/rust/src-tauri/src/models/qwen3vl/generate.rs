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

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, semantic_prejudice: Option<&str>) -> Result<String> {
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
        
        let is_strict_json = mes_render.contains("/no_think") || mes_render.contains("RETURN JSON ONLY") || mes_render.contains("Return ONLY");
        
        // 🌟 [CRITICAL FIX] 누락되어 있던 특수 토큰 ID 추출 로직을 추가합니다.
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let mut gen_text = String::new();

        // 🌟 [Contrastive Semantic Steering] 오답 레이블 진영 밀어내기 (Prejudice)
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        if let Some(target_text) = semantic_prejudice {
            if let Ok(target_ids) = self.tokenizer.text_encode_vec(target_text.to_string(), false) {
                if !target_ids.is_empty() {
                    let calc_prej = || -> Result<Tensor> {
                        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
                        let target_emb = self.qwen3_vl.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
                        let target_emb_sum = target_emb.sum_keepdim(1)?;
                        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
                        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
                        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;
                        
                        let all_embs = self.qwen3_vl.get_embed_tokens().to_dtype(DType::F32)?;
                        let target_norm = target_vec.sqr()?.sum_all()?.sqrt()?;
                        let target_normalized = target_vec.broadcast_div(&target_norm)?;
                        
                        let all_sqr = all_embs.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                        let all_norm = all_sqr.sqrt()?;
                        let all_normalized = all_embs.broadcast_div(&all_norm)?;
                        
                        let sim = all_normalized.matmul(&target_normalized.unsqueeze(1)?)?.squeeze(1)?;
                        // 🌟 [방향 B: Threshold 노이즈 게이트 + Exponential 증폭]
                        let threshold = Tensor::new(0.65f32, &self.device)?;
                        let one = Tensor::new(1.0f32, &self.device)?;
                        let sim_relu = sim.broadcast_sub(&threshold)?.relu()?;
                        let prejudice = sim_relu.affine(15.0, 0.0)?.exp()?.broadcast_sub(&one)?;
                        Ok(prejudice) 
                    };
                    match calc_prej() {
                        Ok(prej) => {
                            semantic_prejudice_tensor = Some(prej);
                            println!("[SEMANTIC-PREJUDICE] Generated Vector Prejudice for target: '{}'", target_text);
                        }
                        Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                    }
                }
            }
        }

        for i in 0..sample_len {
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

            // 🌟 오답 진영 억제력(Sub) 적용
            let logits = if let Some(ref prej) = semantic_prejudice_tensor {
                logits.broadcast_sub(prej)?
            } else {
                logits
            };

            let mut logits_vec = logits.to_vec1::<f32>()?;
            let len = logits_vec.len();

            // 🌟 [CRITICAL FIX] Qwen3VL에도 Repetition Penalty 방어 로직을 추가하여 무한 반복과 JSON 파괴를 모두 막습니다.
            if !generate.is_empty() {
                let penalty = self.generation_config.repetition_penalty; 
                if penalty > 1.0 {
                    let start_at = generate.len().saturating_sub(64);
                    let mut set = std::collections::HashSet::new();
                    for &t in &generate[start_at..] {
                        if !set.contains(&t) && (t as usize) < len {
                            // 🌟 JSON 필수 문법(따옴표, 괄호 등)이 페널티를 먹고 붕괴하는 현상 방어
                            let mut apply_rep_penalty = true;
                            if is_strict_json {
                                if let Ok(piece) = self.tokenizer.token_decode(vec![t]) {
                                    let p = piece.trim();
                                    if p == "\"" || p == "{" || p == "[" || p == "}" || p == "]" || p == "," || p == ":" {
                                        apply_rep_penalty = false;
                                    }
                                }
                            }

                            if apply_rep_penalty {
                                let logit = logits_vec[t as usize];
                                logits_vec[t as usize] = if logit < 0.0 { logit * penalty } else { logit / penalty };
                            }
                            set.insert(t);
                        }
                    }
                }
            }

            // <think> 지속 억제
            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }

            if is_strict_json {
                if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 50.0; }
                
                let is_url_single = gen_text.ends_with("http:/") || gen_text.ends_with("https:/");
                let is_url_double = gen_text.ends_with("http:") || gen_text.ends_with("https:");
                
                if !is_url_single && gen_text.ends_with('/') {
                    if (slash_id as usize) < len { logits_vec[slash_id as usize] -= 10000.0; }
                }
                if !is_url_double {
                    if (double_slash_id as usize) < len { logits_vec[double_slash_id as usize] -= 10000.0; }
                    if (space_double_slash_id as usize) < len { logits_vec[space_double_slash_id as usize] -= 10000.0; }
                }
            }

            // 🌟 [CRITICAL FIX] JSON 강제 출력을 위해 첫 번째 토큰을 { 로 강제 고정합니다.
            if i == 0 {
                if (self.eos_token_id1 as usize) < len { logits_vec[self.eos_token_id1 as usize] = -10000.0; }
                if (self.eos_token_id2 as usize) < len { logits_vec[self.eos_token_id2 as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                
                if (open_bracket_id as usize) < len {
                    let boost = if is_strict_json { 10000.0 } else { 20.0 };
                    logits_vec[open_bracket_id as usize] += boost;
                }
            }

            let logits_tensor = Tensor::from_vec(logits_vec, (len,), &self.device)?;
            let mut next_token = logit_processor.sample(&logits_tensor)?;
            
            // 🌟 [CRITICAL FIX] 첫 번째 토큰 오버라이드
            if i == 0 {
                if is_strict_json {
                    next_token = open_bracket_id;
                } else if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                    next_token = enter_id;
                }
            }

            generate.push(next_token);
            
            let mut is_json_finished = false;
            
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                gen_text.push_str(&piece);
                
                // 🌟 JSON 닫힘 감지 조기 종료 (추론 속도 향상)
                if is_strict_json && gen_text.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                        is_json_finished = true; 
                    }
                }
            }

            if is_json_finished || next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
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
