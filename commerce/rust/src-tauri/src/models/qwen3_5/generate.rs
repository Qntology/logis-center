use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart};
use anyhow::Result;
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use std::io::Write;
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
    pub tokenizer: TokenizerModel, 
    pub pre_processor: Option<Qwen3VLProcessor>,
    pub qwen3_5: Qwen3_5Model,
    device: Device,
    pub eos_token_id: u32,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
    pub max_position_embeddings: usize, 
}

impl Qwen3_5GenerateModel {
    /// 로드 시 설정된 max_position_embeddings 값을 반환합니다.
    pub fn get_max_tokens(&self) -> usize {
        self.max_position_embeddings
    }

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
        
        let max_pos = cfg.text_config.max_position_embeddings;
        let qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5,
            device,
            eos_token_id,
            model_name: model_name.to_string(),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_position_embeddings: max_pos, 
        })
    }

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        // Mmap으로 메모리에 매핑
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);
        
        let mut reader = std::io::Cursor::new(&mmap_arc[..]);
        let content = gguf_file::Content::read(&mut reader)?;
        
        // Content Clone 우회를 위한 메타데이터 Arc 래핑
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

        
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let max_pos = if config_path.exists() {
            let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?)?;
            cfg.get("max_position_embeddings").or_else(|| cfg.get("text_config").and_then(|t| t.get("max_position_embeddings"))).and_then(|v| v.as_u64()).unwrap_or(262144) as usize
        } else {
            
            model_gguf.get_matedata("qwen35.context_length").ok().and_then(|m| m.to_u32().ok()).unwrap_or(262144) as usize
        };

        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        
        // 🚀 mmproj_file 인자가 None이더라도, 같은 폴더에 다운로드된 mmproj-BF16.gguf 파일이 존재하면 자동으로 합체하도록 개선
        let auto_mmproj = std::path::Path::new(model_file).with_file_name("mmproj-BF16.gguf");
        let target_mmproj = if let Some(f) = mmproj_file {
            Some(f.to_string())
        } else if auto_mmproj.exists() {
            Some(auto_mmproj.to_string_lossy().to_string())
        } else {
            None
        };

        let (pre_processor, mut mmproj_gguf) = if let Some(mmproj_f) = target_mmproj {
            let mut reader = std::fs::File::open(&mmproj_f)?;
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
            
        // 밥줄(Mmap, Ct)을 쥐여준 채로 모델 생성
        let qwen3_5 = Qwen3_5Model::new_from_gguf(
            &mut model_gguf, 
            mmproj_gguf.as_mut(), 
            &device,
            Some(mmap_arc.clone()), 
            Some(ct_arc)
        )?;
        
        let stem = std::path::Path::new(model_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
            
        
        // OS 할당자가 붙잡고 있는 수 기가바이트의 찌꺼기 램을 OS 커널 레벨에서 강제로 토해내게 만듭니다.
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

        // 구조체 반환 부분은 그대로 유지
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
            max_position_embeddings: max_pos, 
        })
    }

    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>, 
        session_id: Option<String>, 
        kv_name: Option<String>
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.0);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mut logit_processor =
            get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(80), seed);
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        
        // 메세지 내 비전 URL 유무 판별
        let has_vision = mes.messages.iter().any(|msg| {
            if let ChatCompletionRequestMessage::User(user_msg) = msg {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &user_msg.content {
                    parts.iter().any(|p| matches!(p, ChatCompletionRequestMessageContentPart::ImageURL(_) | ChatCompletionRequestMessageContentPart::VideoURL(_)))
                } else { false }
            } else { false }
        });

        // 비전 유무에 따른 전처리 분기
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw): (String, Option<Tensor>, Option<Tensor>, Option<Tensor>, Option<Tensor>) =
            if has_vision && self.pre_processor.is_some() {
                let processor = self.pre_processor.as_ref().unwrap();
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

        let mut ids_vec = Vec::new();
        for (i, p_start) in mes_text.split("<|vision_start|>").enumerate() {
            if i > 0 { ids_vec.push(248053); } // vision_start_token_id
            for (j, p_end) in p_start.split("<|vision_end|>").enumerate() {
                if j > 0 { ids_vec.push(248054); } // vision_end_token_id
                for (k, p_img) in p_end.split("<|image_pad|>").enumerate() {
                    if k > 0 { ids_vec.push(248056); } // image_token_id
                    for (l, p_vid) in p_img.split("<|video_pad|>").enumerate() {
                        if l > 0 { ids_vec.push(248057); } // video_token_id
                        if !p_vid.is_empty() {
                            let mut t_ids = self.tokenizer.text_encode_vec(p_vid.to_string(), false)?;
                            ids_vec.append(&mut t_ids);
                        }
                    }
                }
            }
        }
        
        let total_toks = ids_vec.len();
        let kv_len = self.qwen3_5.language_model.current_kv_len;

        let (mut input_ids, mut seqlen_offset) = if kv_len > 0 {
            println!("[PARTIAL-PREFILL] Context partially restored ({}). Prefilling remaining tokens.", kv_len);
            let (ids, offset) = if kv_len >= total_toks {
                let last_id = *ids_vec.last().unwrap_or(&0);
                (Tensor::from_vec(vec![last_id], (1, 1), &self.device)?, total_toks.saturating_sub(1))
            } else {
                let missing_ids = ids_vec[kv_len..].to_vec();
                (Tensor::from_vec(missing_ids.clone(), (1, missing_ids.len()), &self.device)?, kv_len)
            };
            (ids, offset)
        } else {
            self.clear_kv_cache(); 
            let full_ids_tensor = Tensor::from_vec(ids_vec.clone(), (1, total_toks), &self.device)?;
            let _ = self.qwen3_5.compute_and_set_rope_deltas(&full_ids_tensor, image_grid_thw.as_ref(), video_grid_thw.as_ref());
            (Tensor::from_vec(ids_vec, (1, total_toks), &self.device)?, 0)
        };

        let mut seq_len = input_ids.dim(1)?;
        let pixel_values: Option<&Tensor> = pixel_values.as_ref();
        let image_grid_thw: Option<&Tensor> = image_grid_thw.as_ref();
        let pixel_values_video: Option<&Tensor> = pixel_values_video.as_ref();
        let video_grid_thw: Option<&Tensor> = video_grid_thw.as_ref();
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        
        let mut cur_pixel_values = pixel_values;
        let mut cur_pixel_values_video = pixel_values_video;

        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let is_strict_json = mes_text.contains("/no_think") || mes_text.contains("RETURN JSON ONLY") || mes_text.contains("Return ONLY");
        let mut gen_text_buffer = String::new(); 
        let mut print_buffer = String::new();

        for i in 0..sample_len {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }

            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values,
                image_grid_thw,
                cur_pixel_values_video,
                video_grid_thw,
                None, 
                seqlen_offset,
                session_id.clone(), 
                kv_name.clone()     
            ).await?;
            
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
            let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let len = logits_vec.len();
            
            if !generate.is_empty() {
                let penalty = 1.1; 
                let mut set = std::collections::HashSet::new();
                let start_at = generate.len().saturating_sub(self.repeat_last_n);
                for &t in &generate[start_at..] {
                    if !set.contains(&t) && (t as usize) < len {
                        let logit = logits_vec[t as usize];
                        logits_vec[t as usize] = if logit < 0.0 { logit * penalty } else { logit / penalty };
                        set.insert(t);
                    }
                }
            }

            if is_strict_json {
                if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 10000.0; }
                if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 50.0; }
            }

            if i == 0 {
                if (self.eos_token_id as usize) < len { logits_vec[self.eos_token_id as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                if (open_bracket_id as usize) < len {
                    let boost = if is_strict_json { 10000.0 } else { 20.0 };
                    logits_vec[open_bracket_id as usize] += boost;
                }
            }
            
            let logits = Tensor::from_vec(logits_vec, (len,), &self.device)?;
            let mut next_token = logit_processor.sample(&logits)?;
            
            if i == 0 {
                if is_strict_json {
                    next_token = open_bracket_id;
                } else if next_token == self.eos_token_id {
                    next_token = enter_id;
                }
            }

            generate.push(next_token);

            let mut is_json_finished = false; 

            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print_buffer.push_str(&piece);
                gen_text_buffer.push_str(&piece);

                if print_buffer.len() >= 10 || piece.contains('\n') || next_token == self.eos_token_id {
                    print!("{}", print_buffer);
                    let _ = std::io::stdout().flush();
                    print_buffer.clear(); 
                }

                if gen_text_buffer.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with('}') {
                        if !print_buffer.is_empty() {
                            print!("{}", print_buffer);
                            let _ = std::io::stdout().flush();
                        }
                        println!("\n[DEBUG-GEN] Balanced JSON detected. Stopping.");
                        is_json_finished = true; 
                    }
                }
            }

            let current_pos = seqlen_offset + seq_len;
            let is_eos = next_token == self.eos_token_id;
            
            
            if true {
                let pct = if is_json_finished || is_eos {
                    100
                } else {
                    
                    15 + (84.0 * (1.0 - (-0.05 * (i as f32)).exp())) as i32
                };
                
                if let Some(tx) = crate::scheduler::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = if current_cat.contains("Classification") {
                            format!("Analyzing structure ({}%)...", pct)
                        } else if task_id.starts_with("search_") {
                            format!("Generating insights ({}%)...", pct)
                        } else {
                            format!("Extracting data ({}%)...", pct)
                        };

                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Decoding)", current_cat), 
                            "summary": summary_msg,
                            "spinner": "⠧"
                        }));
                    }
                }
                if !is_strict_json {
                    print!("\r[Qwen3.5-DECODING] {} tokens generated (Context: {})    ", i + 1, current_pos + 1);
                    let _ = std::io::stdout().flush();
                }
            }

            if is_eos || is_json_finished {
                break;
            }
            
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cur_pixel_values = None;
            cur_pixel_values_video = None;
        }
        println!(); 

        let res = self.tokenizer.token_decode(generate)?;
        
        if let Some(s_id) = &session_id {
            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
            println!("[GEN-SAVE] Qwen 3.5 active KV blocks flushed to disk");
        }

        self.qwen3_5.clear_cache();
        Ok(res)
    }

    pub async fn generate_part(
        &mut self,
        mes: &ChatCompletionParameters,
        is_continuation: bool,
        last_offset: usize,
        cancel_flag: Option<Arc<AtomicBool>>,
        last_token: Option<u32>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> Result<GenerationResult> {
        let mut logit_processor = get_logit_processor(
            Some(mes.temperature.unwrap_or(0.0) as f32), 
            Some(mes.top_p.unwrap_or(0.95) as f32), 
            Some(80), 
            mes.seed.unwrap_or(32768) as u64
        );
        
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        let (mut input_ids, mut seqlen_offset, mut cur_pixel_values, mut cur_image_thw, mut cur_pixel_values_video, mut cur_video_thw) = if !is_continuation {

            let mes_render = self.chat_template.apply_chat_template(mes)?;
            
            let has_vision = mes.messages.iter().any(|msg| {
                if let ChatCompletionRequestMessage::User(user_msg) = msg {
                    if let ChatCompletionRequestUserMessageContent::Array(parts) = &user_msg.content {
                        parts.iter().any(|p| matches!(p, ChatCompletionRequestMessageContentPart::ImageURL(_) | ChatCompletionRequestMessageContentPart::VideoURL(_)))
                    } else { false }
                } else { false }
            });

            let (text, px_vals, img_thw, vid_px_vals, vid_thw) = if has_vision && self.pre_processor.is_some() {
                let processor = self.pre_processor.as_ref().unwrap();
                let input = processor.process_info(mes, &mes_render)?;
                (input.replace_text, input.pixel_values, input.image_grid_thw, input.pixel_values_video, input.video_grid_thw)
            } else {
                (mes_render, None, None, None, None)
            };
            
            let mut ids_vec = Vec::new();
            for (i, p_start) in text.split("<|vision_start|>").enumerate() {
                if i > 0 { ids_vec.push(248053); } 
                for (j, p_end) in p_start.split("<|vision_end|>").enumerate() {
                    if j > 0 { ids_vec.push(248054); } 
                    for (k, p_img) in p_end.split("<|image_pad|>").enumerate() {
                        if k > 0 { ids_vec.push(248056); } 
                        for (l, p_vid) in p_img.split("<|video_pad|>").enumerate() {
                            if l > 0 { ids_vec.push(248057); } 
                            if !p_vid.is_empty() {
                                let mut t_ids = self.tokenizer.text_encode_vec(p_vid.to_string(), false)?;
                                ids_vec.append(&mut t_ids);
                            }
                        }
                    }
                }
            }
            let total_toks = ids_vec.len();
            let kv_len = self.qwen3_5.language_model.current_kv_len;

            if kv_len > 0 {
                println!("[PARTIAL-PREFILL] Context partially restored ({}). Prefilling remaining tokens.", kv_len);
                let (ids, offset) = if kv_len >= total_toks {
                    let last_id = *ids_vec.last().unwrap_or(&0);
                    (Tensor::from_vec(vec![last_id], (1, 1), &self.device)?, total_toks.saturating_sub(1))
                } else {
                    let missing_ids = ids_vec[kv_len..].to_vec();
                    (Tensor::from_vec(missing_ids.clone(), (1, missing_ids.len()), &self.device)?, kv_len)
                };
                (ids, offset, px_vals, img_thw, vid_px_vals, vid_thw)
            } else {
                self.clear_kv_cache();
                let full_ids_tensor = Tensor::from_vec(ids_vec.clone(), (1, total_toks), &self.device)?;
                let _ = self.qwen3_5.compute_and_set_rope_deltas(&full_ids_tensor, img_thw.as_ref(), vid_thw.as_ref());
                
                let ids = Tensor::from_vec(ids_vec, (1, total_toks), &self.device)?;
                (ids, 0, px_vals, img_thw, vid_px_vals, vid_thw)
            }
        } else {
            let ids = Tensor::from_vec(vec![last_token.unwrap()], (1, 1), &self.device)?;
            (ids, last_offset, None, None, None, None)
        };

        let mut seq_len = input_ids.dim(1)?;
        let mut is_finished = false;
        let mut final_token = 0;
        
        println!("\n[AI Thinking...]");

        crate::models::qwen::generate::wait_for_global_io().await;

        let mes_check = self.chat_template.apply_chat_template(mes).unwrap_or_default();
        let is_strict_json = mes_check.contains("/no_think") || mes_check.contains("RETURN JSON ONLY") || mes_check.contains("Return ONLY");
        
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let mut gen_text_buffer = String::new();
        let mut print_buffer = String::new();

        for i in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            crate::models::qwen::generate::wait_for_global_io().await;

            let logits = self.qwen3_5.forward(
                &input_ids,
                cur_pixel_values.as_ref(),
                cur_image_thw.as_ref(),
                cur_pixel_values_video.as_ref(),
                cur_video_thw.as_ref(),
                None, 
                seqlen_offset,
                session_id.clone(),
                kv_name.clone()
            ).await?;
            
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
            let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let len = logits_vec.len();

            if !generate.is_empty() {
                let penalty = 1.1; 
                let mut set = std::collections::HashSet::new();
                let start_at = generate.len().saturating_sub(self.repeat_last_n);
                for &t in &generate[start_at..] {
                    if !set.contains(&t) && (t as usize) < len {
                        let logit = logits_vec[t as usize];
                        logits_vec[t as usize] = if logit < 0.0 { logit * penalty } else { logit / penalty };
                        set.insert(t);
                    }
                }
            }

            if is_strict_json {
                if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 10000.0; }
                if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 50.0; }
            }

            if i == 0 {
                if (self.eos_token_id as usize) < len { logits_vec[self.eos_token_id as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                if (open_bracket_id as usize) < len {
                    let boost = if is_strict_json { 10000.0 } else { 20.0 };
                    logits_vec[open_bracket_id as usize] += boost;
                }
            }

            let logits = Tensor::from_vec(logits_vec, (len,), &self.device)?;
            let mut next_token = logit_processor.sample(&logits)?;
            
            if i == 0 {
                if is_strict_json {
                    next_token = open_bracket_id;
                } else if next_token == self.eos_token_id {
                    next_token = enter_id;
                }
            }

            generate.push(next_token);
            final_token = next_token;

            let mut is_json_finished = false;

            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print_buffer.push_str(&piece);
                gen_text_buffer.push_str(&piece);

                if print_buffer.len() >= 10 || piece.contains('\n') || next_token == self.eos_token_id {
                    print!("{}", print_buffer);
                    let _ = std::io::stdout().flush();
                    print_buffer.clear(); 
                }

                if gen_text_buffer.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with('}') {
                        if !print_buffer.is_empty() {
                            print!("{}", print_buffer);
                            let _ = std::io::stdout().flush();
                        }
                        println!("\n[DEBUG-GEN] Balanced JSON detected. Stopping.");
                        is_json_finished = true; 
                    }
                }
            }

            let current_pos = seqlen_offset + seq_len;
            let is_eos = next_token == self.eos_token_id;
            
            
            if true {
                let pct = if is_json_finished || is_eos {
                    100
                } else {
                    
                    15 + (84.0 * (1.0 - (-0.05 * (i as f32)).exp())) as i32
                };
                
                if let Some(tx) = crate::scheduler::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = if current_cat.contains("Classification") {
                            format!("Analyzing structure ({}%)...", pct)
                        } else if task_id.starts_with("search_") {
                            format!("Generating insights ({}%)...", pct)
                        } else {
                            format!("Extracting data ({}%)...", pct)
                        };

                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Decoding)", current_cat), 
                            "summary": summary_msg,
                            "spinner": "⠧"
                        }));
                    }
                }
                if !is_strict_json {
                    print!("\r[Qwen3.5-DECODING] {} tokens generated (Context: {})    ", i + 1, current_pos + 1);
                    let _ = std::io::stdout().flush();
                }
            }

            if is_eos || is_json_finished {
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

    pub async fn prefill_only(
        &mut self, 
        mes: ChatCompletionParameters, 
        session_id: Option<String>, 
        kv_name: Option<String>
    ) -> Result<usize> {
        self.clear_kv_cache();

        let mut mes_render = String::new();
        for msg in &mes.messages {
            match msg {
                ChatCompletionRequestMessage::System(sys) => {
                    mes_render.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.content));
                }
                ChatCompletionRequestMessage::User(user) => {
                    match &user.content {
                        ChatCompletionRequestUserMessageContent::Text(text) => {
                            mes_render.push_str(&format!("<|im_start|>user\n{}", text));
                        }
                        ChatCompletionRequestUserMessageContent::Array(parts) => {
                            let mut combined = String::new();
                            for part in parts {
                                if let ChatCompletionRequestMessageContentPart::Text(t) = part {
                                    combined.push_str(&t.text);
                                }
                            }
                            mes_render.push_str(&format!("<|im_start|>user\n{}", combined));
                        }
                    }
                }
                _ => {}
            }
        }

        // ✅ 수정 후 (안전 마진 10토큰 남기기 적용)
        let input_ids = self.tokenizer.text_encode_vec(mes_render.clone(), false)?;
        let total_toks = input_ids.len();

        
        // Qwen 3.5의 Mamba(SSM) 레이어는 과거로 롤백(Truncate)이 불가능합니다.
        // 뒤에 이어붙을 질문([TASK]) 때문에 마지막 토큰 경계가 변형되어 뇌가 깨지는 것을 막기 위해,
        // 안전하게 마지막 10개 토큰은 굽지 않고 남겨둡니다. (generate_part에서 자연스럽게 이어서 연산됨)
        let safe_toks = if total_toks > 20 { total_toks - 10 } else { total_toks };
        let safe_input_ids = input_ids[..safe_toks].to_vec();

        let ids_tensor = Tensor::from_vec(safe_input_ids.clone(), (1, safe_toks), &self.device)?;

        self.qwen3_5.forward(
            &ids_tensor, None, None, None, None, 
            None, 
            0,
            session_id.clone(), kv_name.clone()
        ).await?;

        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if !path.exists() { std::fs::create_dir_all(&path)?; }
            
            std::fs::write(path.join("tokens.json"), serde_json::to_string(&safe_input_ids)?)?;

            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            println!("[PREFILL-WAIT] Waiting for SSD write to complete...");
            crate::models::qwen::generate::wait_for_global_io().await;
            println!("[PREFILL-SAVE] Confirm: Qwen 3.5 Base Context prefilled and safely flushed to disk. ({} tokens)", safe_toks);
        }

        Ok(safe_toks)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_5.clear_cache();
    }
}