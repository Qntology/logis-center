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
            // 🌟 [VRAM 최적화 1] GPU에서 직접 압축을 풀면 압축 원본과 해제본이 동시에 VRAM을 점유하여(메모리 파편화) 2.4GB까지 치솟습니다.
            // CPU(RAM)에서 압축을 해제한 후 깨끗한 텐서만 GPU로 전송하여 VRAM 적재량을 1.2GB 수준으로 반토막 냅니다!
            let t_cpu = ct.tensor(&mut file, name, &candle_core::Device::Cpu)?;
            let t = t_cpu.dequantize_f16(&candle_core::Device::Cpu).or_else(|_| t_cpu.dequantize(&candle_core::Device::Cpu))?.to_device(&device)?.to_dtype(dtype)?;
            
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

        // 🌟 [Chunked Prefill] VRAM 및 RAM 널뛰기를 원천 방어하기 위해 청크 사이즈를 256으로 축소하여 강하게 압박합니다.
        let chunk_size = 256;
        let mut seqlen_offset = 0;
        
        while seqlen_offset < seq_len {
            if let Some(flag) = &_cancellation_token {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            let take = std::cmp::min(chunk_size, seq_len - seqlen_offset);
            let chunk_ids = input_ids.narrow(1, seqlen_offset, take)?;
            let _ = self.qwen3.forward(Some(&chunk_ids), None, seqlen_offset)?;
            
            // 🌟 [VRAM/RAM 최적화] 청크 연산 직후 GPU를 동기화하고 시스템 메모리(RAM)를 강제로 반환시킵니다.
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

            seqlen_offset += take;
        }

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
        ignore_list: Option<&[String]>,
        semantic_prejudice: Option<&str>,
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
        // 🌟 [CRITICAL FIX] 소유권 박탈(E0382) 방지를 위해 clone()을 넘겨줍니다.
        let f_ids = self.tokenizer.text_encode_vec(mes_render.clone(), false)?;
        
        if start_len >= f_ids.len() {
            return Err(anyhow::anyhow!("start_len exceeds or equals prompt length"));
        }

        let p_ids = &f_ids[start_len..];
        let input_ids = Tensor::from_vec(p_ids.to_vec(), (1, p_ids.len()), &self.device)?;
        
        let prompt_seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = start_len;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);

        // 🌟 [JSON Enforcement Prep] 프롬프트를 기반으로 JSON 모드 여부를 판단하고 특수 토큰을 준비합니다.
        let is_strict_json = mes_render.contains("/no_think") || mes_render.contains("RETURN JSON ONLY");
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let mut gen_text = String::new();

        // 🌟 [Contrastive Semantic Steering] 오답 레이블 진영 밀어내기 (Prejudice)
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        if let Some(target_text) = semantic_prejudice {
            if let Ok(target_ids) = self.tokenizer.text_encode_vec(target_text.to_string(), false) {
                if !target_ids.is_empty() {
                    let calc_prej = || -> Result<Tensor> {
                        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
                        let target_emb = self.qwen3.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
                        let target_emb_sum = target_emb.sum_keepdim(1)?;
                        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
                        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
                        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;
                        
                        let all_embs = self.qwen3.get_embed_tokens().to_dtype(DType::F32)?;
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

        // 🌟 [Phase 1: Chunked Prefill] VRAM 폭발과 RAM 널뛰기를 막기 위해 256 토큰 단위로 강하게 압박합니다.
        let chunk_size = 256;
        let mut next_token = 0;
        let mut current_chunk_offset = 0;
        
        while current_chunk_offset < prompt_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(String::new());
                }
            }
            let take = std::cmp::min(chunk_size, prompt_seq_len - current_chunk_offset);
            let chunk_ids = input_ids.narrow(1, current_chunk_offset, take)?;
            
            let logits = self.qwen3.forward(Some(&chunk_ids), None, seqlen_offset)?;
            
            // 🌟 [VRAM/RAM 최적화] 청크 연산 직후 GPU를 동기화하고 시스템 메모리(RAM)를 강제로 반환시킵니다.
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

            seqlen_offset += take;
            current_chunk_offset += take;
            
            // 프리필이 완전히 끝나는 마지막 청크의 끝에서만 첫 번째 토큰을 샘플링합니다.
            if current_chunk_offset == prompt_seq_len {
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

                // 🌟 오답 진영 억제력(Sub) 적용
                let logits = if let Some(ref prej) = semantic_prejudice_tensor {
                    logits.broadcast_sub(prej)?
                } else {
                    logits
                };
                
                let mut logits_vec = logits.to_vec1::<f32>()?;
                let len = logits_vec.len();

                // 🌟 JSON 강제 룰 적용 (첫 토큰)
                if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }
                if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 10.0; }
                if (self.eos_token_id1 as usize) < len { logits_vec[self.eos_token_id1 as usize] = -10000.0; }
                if (self.eos_token_id2 as usize) < len { logits_vec[self.eos_token_id2 as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }

                if is_strict_json {
                    if (open_bracket_id as usize) < len { logits_vec[open_bracket_id as usize] += 10000.0; }
                    if (double_slash_id as usize) < len { logits_vec[double_slash_id as usize] -= 10000.0; }
                    if (space_double_slash_id as usize) < len { logits_vec[space_double_slash_id as usize] -= 10000.0; }
                }

                let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
                next_token = logit_processor.sample(&logits_tensor)?;

                if is_strict_json { next_token = open_bracket_id; }

                generate.push(next_token);
                if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) { gen_text.push_str(&piece); }
            }
        }

        // 🌟 [KV RESIDENCY PLAN] 디코딩 루프 진입 직전에 단 1회만 VRAM/RAM 여유를 계산합니다.
        //    KV Cache 는 단조 증가하므로 "마지막 토큰까지 자랐을 때의 최대 크기"로 판정해야
        //    디코딩 도중 OOM 이 발생하지 않습니다. 매 토큰 재확인은 불필요합니다.
        {
            let (kv_layers, kv_heads, head_dim) = self.qwen3.kv_geometry();
            let plan = crate::utils::resources::plan_kv_residency(
                &crate::utils::resources::KvPlanInput {
                    gpu_id: crate::utils::resources::primary_gpu_id(),
                    is_cpu_mode: self.device.is_cpu(),
                    num_kv_layers: kv_layers,
                    num_kv_heads: kv_heads,
                    head_dim,
                    // VRAM 상주 시 Fp8VramKVCache 가 FP8(1바이트)로 압축 보관합니다.
                    bytes_per_elem: 1,
                    planned_tokens: seqlen_offset + sample_len as usize,
                    label: "Qwen3(0.6B) generate_part",
                },
            );
            self.qwen3.set_kv_residency(plan);
        }

        // 🌟 [Phase 2: Decoding] 첫 토큰을 얻은 후 1글자씩 이어서 생성합니다.
        for i in 1..sample_len {
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }

            let single_input = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let logits = self.qwen3.forward(Some(&single_input), None, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            // 🌟 오답 진영의 단어들은 아예 생성되지 못하도록 억제력(Sub)을 가합니다!
            let logits = if let Some(ref prej) = semantic_prejudice_tensor {
                logits.broadcast_sub(prej)?
            } else {
                logits
            };
            
            let mut logits_vec = logits.to_vec1::<f32>()?;
            let len = logits_vec.len();

            // <think> 지속 억제
            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }

            // 🌟 [CRITICAL FIX] 주석(//) 지속 억제 (웹 주소 http://, https:// 제외)
            if is_strict_json {
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

            // 🌟 [CRITICAL FIX] ignore_list에 등재된 잘못된 추출값의 토큰 시퀀스 생성을 억제(Bias)합니다.
            if let Some(ignores) = ignore_list {
                for ign in ignores {
                    if is_strict_json && (ign.trim().starts_with('{') || ign.trim().starts_with('[')) {
                        continue;
                    }

                    let ign_toks = self.tokenizer.text_encode_vec(ign.to_string(), false).unwrap_or_default();
                    if ign_toks.is_empty() { continue; }
                    
                    let mut overlap = 0;
                    for l in (1..=ign_toks.len().min(generate.len())).rev() {
                        if generate.ends_with(&ign_toks[..l]) {
                            overlap = l;
                            break;
                        }
                    }
                    
                    if overlap < ign_toks.len() {
                        let next_tok = ign_toks[overlap] as usize;
                        if next_tok < len {
                            let mut apply_bias = false;
                            if overlap > 0 {
                                apply_bias = true;
                            } else if gen_text.ends_with('"') || gen_text.ends_with(':') || gen_text.ends_with(": ") {
                                apply_bias = true;
                            }
                            
                            // 🌟 [최강 방어 로직] JSON 필수 문법(따옴표, 괄호 등)을 환각 방지(Bias) 억제 대상에서 면제합니다.
                            if apply_bias && is_strict_json {
                                if let Ok(piece) = self.tokenizer.token_decode(vec![next_tok as u32]) {
                                    let p = piece.trim();
                                    if p == "\"" || p == "{" || p == "[" || p == "}" || p == "]" || p == "," || p == ":" {
                                        apply_bias = false;
                                    }
                                }
                            }

                            if apply_bias {
                                logits_vec[next_tok] -= 10000.0;
                            }
                        }
                    }
                }
            }

            // 🌟 [CRITICAL FIX] 모델이 같은 문장을 무한 반복하는 현상(Loop)을 끊기 위해 페널티를 연산합니다.
            let penalty = self.generation_config.repetition_penalty; // 1.0으로 무효화하던 꼼수 제거!
            if penalty > 1.0 {
                // 최근 512개 토큰에 대해서만 페널티를 적용하여 정상적인 문맥 훼손을 방지합니다.
                let start_idx = generate.len().saturating_sub(512); 
                for &t in &generate[start_idx..] {
                    let t_idx = t as usize;
                    if t_idx < len {
                        // 🌟 [추가] JSON 필수 문법이 반복 페널티를 먹고 붕괴하는 현상을 막기 위해 똑같이 보호합니다!
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
                            if logits_vec[t_idx] <= 0.0 {
                                logits_vec[t_idx] *= penalty;
                            } else {
                                logits_vec[t_idx] /= penalty;
                            }
                        }
                    }
                }
            }

            let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
            next_token = logit_processor.sample(&logits_tensor)?;

            generate.push(next_token);
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                gen_text.push_str(&piece);
            }

            // 🌟 JSON 닫힘 감지 조기 종료 (추론 속도 향상)
            if is_strict_json && gen_text.contains('{') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '{' { depth += 1; has_started = true; }
                    else if c == '}' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                    break;
                }
            }

            seqlen_offset += 1;

            // 🌟 메모리 최적화 (15토큰마다 OS 시스템 RAM 스파이크 억제 및 반환, 주기를 짧게 압박)
            if i > 0 && i % 15 == 0 {
                // 🌟 [VRAM 최적화 3] 디코딩 중 발생하는 KV Cache 병합(Cat) 찌꺼기 텐서들을 즉시 날려버립니다.
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

        // BPE 디코딩 무결성을 보장하기 위해 최종 조립은 배열 기반으로 수행합니다.
        let res_text = self.tokenizer.token_decode(generate)?;
        
        Ok(res_text)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, ignore_list: Option<&[String]>, semantic_prejudice: Option<&str>) -> Result<String> {
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
        let input_ids = self.tokenizer.text_encode(mes_render.clone(), &self.device)?;
        let prompt_seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        
        if prompt_seq_len == 0 { return Ok(String::new()); }

        // 🌟 [JSON Enforcement Prep] 프롬프트를 기반으로 JSON 모드 여부를 판단하고 특수 토큰을 준비합니다.
        let is_strict_json = mes_render.contains("/no_think") || mes_render.contains("RETURN JSON ONLY");
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let mut gen_text = String::new();

        // 🌟 [Contrastive Semantic Steering] 오답 레이블 진영 밀어내기 (Prejudice)
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        if let Some(target_text) = semantic_prejudice {
            if let Ok(target_ids) = self.tokenizer.text_encode_vec(target_text.to_string(), false) {
                if !target_ids.is_empty() {
                    let calc_prej = || -> Result<Tensor> {
                        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
                        let target_emb = self.qwen3.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
                        let target_emb_sum = target_emb.sum_keepdim(1)?;
                        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
                        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
                        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;
                        
                        let all_embs = self.qwen3.get_embed_tokens().to_dtype(DType::F32)?;
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

        // 🌟 [Phase 1: Chunked Prefill] 긴 문맥을 256 토큰 단위로 강하게 잘라 VRAM 및 RAM 널뛰기를 막습니다.
        let chunk_size = 256;
        let mut next_token = 0;
        
        while seqlen_offset < prompt_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(String::new());
                }
            }
            let take = std::cmp::min(chunk_size, prompt_seq_len - seqlen_offset);
            let chunk_ids = input_ids.narrow(1, seqlen_offset, take)?;
            
            let logits = self.qwen3.forward(Some(&chunk_ids), None, seqlen_offset)?;
            
            // 🌟 [VRAM/RAM 최적화] 청크 연산 직후 GPU 동기화 및 시스템 메모리 강제 반환을 수행합니다.
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

            seqlen_offset += take;
            
            // 프리필이 완전히 끝나는 마지막 청크의 끝에서만 첫 번째 토큰을 샘플링합니다.
            if seqlen_offset == prompt_seq_len {
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

                // 🌟 오답 진영 억제력(Sub) 적용
                let logits = if let Some(ref prej) = semantic_prejudice_tensor {
                    logits.broadcast_sub(prej)?
                } else {
                    logits
                };
                
                let mut logits_vec = logits.to_vec1::<f32>()?;
                let len = logits_vec.len();

                // 🌟 JSON 강제 룰 적용 (첫 토큰)
                if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }
                if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 10.0; }
                if (self.eos_token_id1 as usize) < len { logits_vec[self.eos_token_id1 as usize] = -10000.0; }
                if (self.eos_token_id2 as usize) < len { logits_vec[self.eos_token_id2 as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }

                if is_strict_json {
                    if (open_bracket_id as usize) < len { logits_vec[open_bracket_id as usize] += 10000.0; }
                    if (double_slash_id as usize) < len { logits_vec[double_slash_id as usize] -= 10000.0; }
                    if (space_double_slash_id as usize) < len { logits_vec[space_double_slash_id as usize] -= 10000.0; }
                }

                let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
                next_token = logit_processor.sample(&logits_tensor)?;

                if is_strict_json { next_token = open_bracket_id; }

                generate.push(next_token);
                if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) { gen_text.push_str(&piece); }
            }
        }

        // 🌟 [KV RESIDENCY PLAN] 디코딩 루프 진입 직전에 단 1회만 VRAM/RAM 여유를 계산합니다.
        //    VRAM 이 충분하면 FP8 캐시를 VRAM 에 그대로 두고(왕복 0회),
        //    부족하면 RAM 으로 대피시켜 OOM 을 원천 차단합니다.
        {
            let (kv_layers, kv_heads, head_dim) = self.qwen3.kv_geometry();
            let plan = crate::utils::resources::plan_kv_residency(
                &crate::utils::resources::KvPlanInput {
                    gpu_id: crate::utils::resources::primary_gpu_id(),
                    is_cpu_mode: self.device.is_cpu(),
                    num_kv_layers: kv_layers,
                    num_kv_heads: kv_heads,
                    head_dim,
                    bytes_per_elem: 1,
                    planned_tokens: seqlen_offset + sample_len as usize,
                    label: "Qwen3(0.6B) generate",
                },
            );
            self.qwen3.set_kv_residency(plan);
        }

        // 🌟 [Phase 2: Decoding] 첫 번째 토큰을 얻은 후 1글자씩 이어서 생성합니다.
        for i in 1..sample_len {
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }

            let single_input = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let logits = self.qwen3.forward(Some(&single_input), None, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;


            // 🌟 오답 진영의 단어들은 아예 생성되지 못하도록 억제력(Sub)을 가합니다!
            let logits = if let Some(ref prej) = semantic_prejudice_tensor {
                logits.broadcast_sub(prej)?
            } else {
                logits
            };
            
            let mut logits_vec = logits.to_vec1::<f32>()?;
            let len = logits_vec.len();

            // <think> 지속 억제
            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }

            // 🌟 [CRITICAL FIX] 주석(//) 지속 억제 (웹 주소 http://, https:// 제외)
            if is_strict_json {
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

            // 🌟 [CRITICAL FIX] ignore_list에 등재된 잘못된 추출값의 토큰 시퀀스 생성을 억제(Bias)합니다.
            if let Some(ignores) = ignore_list {
                for ign in ignores {
                    if is_strict_json && (ign.trim().starts_with('{') || ign.trim().starts_with('[')) {
                        continue;
                    }

                    let ign_toks = self.tokenizer.text_encode_vec(ign.to_string(), false).unwrap_or_default();
                    if ign_toks.is_empty() { continue; }
                    
                    let mut overlap = 0;
                    for l in (1..=ign_toks.len().min(generate.len())).rev() {
                        if generate.ends_with(&ign_toks[..l]) {
                            overlap = l;
                            break;
                        }
                    }
                    
                    if overlap < ign_toks.len() {
                        let next_tok = ign_toks[overlap] as usize;
                        if next_tok < len {
                            let mut apply_bias = false;
                            if overlap > 0 {
                                apply_bias = true;
                            } else if gen_text.ends_with('"') || gen_text.ends_with(':') || gen_text.ends_with(": ") {
                                apply_bias = true;
                            }
                            
                            // 🌟 [최강 방어 로직] JSON 필수 문법(따옴표, 괄호 등)을 환각 방지(Bias) 억제 대상에서 면제합니다.
                            if apply_bias && is_strict_json {
                                if let Ok(piece) = self.tokenizer.token_decode(vec![next_tok as u32]) {
                                    let p = piece.trim();
                                    if p == "\"" || p == "{" || p == "[" || p == "}" || p == "]" || p == "," || p == ":" {
                                        apply_bias = false;
                                    }
                                }
                            }

                            if apply_bias {
                                logits_vec[next_tok] -= 10000.0;
                            }
                        }
                    }
                }
            }

            // 🌟 [CRITICAL FIX] 모델이 같은 문장을 무한 반복하는 현상(Loop)을 끊기 위해 페널티를 연산합니다.
            let penalty = self.generation_config.repetition_penalty; // 1.0으로 무효화하던 꼼수 제거!
            if penalty > 1.0 {
                // 최근 512개 토큰에 대해서만 페널티를 적용하여 정상적인 문맥 훼손을 방지합니다.
                let start_idx = generate.len().saturating_sub(512); 
                for &t in &generate[start_idx..] {
                    let t_idx = t as usize;
                    if t_idx < len {
                        // 🌟 [추가] JSON 필수 문법이 반복 페널티를 먹고 붕괴하는 현상을 막기 위해 똑같이 보호합니다!
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
                            if logits_vec[t_idx] <= 0.0 {
                                logits_vec[t_idx] *= penalty;
                            } else {
                                logits_vec[t_idx] /= penalty;
                            }
                        }
                    }
                }
            }

            let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
            next_token = logit_processor.sample(&logits_tensor)?;
            
            generate.push(next_token);
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                gen_text.push_str(&piece);
            }

            // 🌟 JSON 닫힘 감지 조기 종료 (추론 속도 향상)
            if is_strict_json && gen_text.contains('{') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '{' { depth += 1; has_started = true; }
                    else if c == '}' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                    break;
                }
            }

            seqlen_offset += 1;

            // 🌟 메모리 최적화 (15토큰마다 OS 시스템 RAM 스파이크 억제 및 반환, 잦은 주기로 압박)
            if i > 0 && i % 15 == 0 {
                // 🌟 [VRAM 최적화 3] 디코딩 중 발생하는 KV Cache 병합(Cat) 찌꺼기 텐서들을 즉시 날려버립니다.
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
        self.qwen3.clear_kv_cache();
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3.clear_kv_cache();
    }
}
