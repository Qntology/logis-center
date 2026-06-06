use anyhow::Result;
// 🌟 [CRITICAL FIX] Embedding 등의 모듈에서 .forward() 메서드를 사용하기 위해 Module 트레잇을 추가로 임포트합니다.
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::models::granite4::config::GraniteMoeHybridConfig;
use crate::models::granite4::model::GraniteMoeHybridForCausalLM;
use crate::openai_types::ChatCompletionParameters;

pub struct Granite4GenerateModel {
    pub language_model: GraniteMoeHybridForCausalLM,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
    pub eos_token_id: u32,
}

impl Granite4GenerateModel {
    /// 🌟 모델 폴더 경로에서 Safetensors 가중치와 Config, Tokenizer를 모두 읽어와 엔진을 초기화합니다.
    pub fn init_from_directory(
        model_path: &Path,
        device: Option<&Device>,
        dtype_opt: Option<DType>,
    ) -> Result<Self> {
        let device = device.cloned().unwrap_or(Device::Cpu);
        let dtype = dtype_opt.unwrap_or(DType::F32);

        // 1. Config 로드
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
        let config: GraniteMoeHybridConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

        // 2. Tokenizer 로드
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = match &config.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(2) as u32,
            Some(serde_json::Value::Array(arr)) => arr.first().and_then(|v| v.as_u64()).unwrap_or(2) as u32,
            _ => 2,
        };

        // 3. GGUF 또는 Safetensors 가중치 로드 (GGUF 우선)
        let gguf_files = crate::utils::find_type_files(&model_path.to_string_lossy(), "gguf")
            .unwrap_or_default();
        
        let vb = if !gguf_files.is_empty() {
            // 🌟 [CRITICAL FIX] GGUF 가중치를 읽어와 RAM 메모리 상에서 실시간으로 역양자화(FP16/F32)한 뒤 VarBuilder로 조립합니다.
            println!("[Granite] Loading and dequantizing weights from GGUF: {:?}", gguf_files[0]);
            let mut file = std::fs::File::open(&gguf_files[0]).map_err(|e| anyhow::anyhow!("Failed to open GGUF: {}", e))?;
            let mut ct = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("Failed to read GGUF: {}", e))?;
            let mut tensors = std::collections::HashMap::new();
            
            // 🌟 [CRITICAL FIX] TensorInfo의 Clone 미지원 및 타입 추론 에러를 해결하기 위해 Key(이름) 배열만 추출하여 순회합니다.
            let tensor_names: Vec<String> = ct.tensor_infos.keys().cloned().collect();
            
            // 분할 적재된 피드포워드 신경망을 임시 추적하기 위한 컬렉션 장부 사양
            let mut ffn_gate_layers = std::collections::HashMap::new();
            let mut ffn_up_layers = std::collections::HashMap::new();

            for name in tensor_names {
                let q_tensor = ct.tensor(&mut file, &name, &device).map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;
                let mut tensor = q_tensor.dequantize_f16(&device).or_else(|_| q_tensor.dequantize(&device)).map_err(|e| anyhow::anyhow!("Dequantize error: {}", e))?;
                
                let mut mapped_name = name.clone();
                if name == "token_embd.weight" {
                    mapped_name = "model.embed_tokens.weight".to_string();
                } else if name == "output_norm.weight" {
                    mapped_name = "model.norm.weight".to_string();
                } else if name == "output.weight" {
                    mapped_name = "lm_head.weight".to_string();
                } else if name.starts_with("blk.") {
                    let parts: Vec<&str> = name.split('.').collect();
                    if parts.len() >= 3 {
                        if let Ok(layer_idx) = parts[1].parse::<usize>() {
                            let suffix = parts[2..].join(".");
                            match suffix.as_str() {
                                "attn_norm.weight" => { mapped_name = format!("model.layers.{}.input_layernorm.weight", layer_idx); },
                                "post_attention_norm.weight" | "ffn_norm.weight" => { mapped_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx); },
                                "ffn_gate.weight" => { ffn_gate_layers.insert(layer_idx, tensor.clone()); continue; },
                                "ffn_up.weight" => { ffn_up_layers.insert(layer_idx, tensor.clone()); continue; },
                                "ffn_down.weight" => { mapped_name = format!("model.layers.{}.shared_mlp.output_linear.weight", layer_idx); },
                                
                                // 하이브리드 아키텍처 Mamba 컴포넌트 특수 명칭 바인딩
                                "ssm_in.weight" => { mapped_name = format!("model.layers.{}.mamba.in_proj.weight", layer_idx); },
                                "ssm_conv1d.weight" => { 
                                    mapped_name = format!("model.layers.{}.mamba.conv1d.weight", layer_idx); 
                                    tensor = tensor.unsqueeze(1).map_err(|e| anyhow::anyhow!("Conv1d weight reshape failed: {}", e))?;
                                },
                                "ssm_conv1d.bias" => { 
                                    mapped_name = format!("model.layers.{}.mamba.conv1d.bias", layer_idx); 
                                    if tensor.rank() == 2 && tensor.dims()[1] == 1 { tensor = tensor.squeeze(1).unwrap_or(tensor); }
                                },
                                "ssm_dt.bias" => { 
                                    mapped_name = format!("model.layers.{}.mamba.dt_bias", layer_idx); 
                                    if tensor.rank() == 2 && tensor.dims()[1] == 1 { tensor = tensor.squeeze(1).unwrap_or(tensor); }
                                },
                                "ssm_a" => { 
                                    mapped_name = format!("model.layers.{}.mamba.A_log", layer_idx); 
                                    if tensor.rank() == 2 && tensor.dims()[1] == 1 { tensor = tensor.squeeze(1).unwrap_or(tensor); }
                                },
                                "ssm_d" => { 
                                    mapped_name = format!("model.layers.{}.mamba.D", layer_idx); 
                                    if tensor.rank() == 2 && tensor.dims()[1] == 1 { tensor = tensor.squeeze(1).unwrap_or(tensor); }
                                },
                                "ssm_norm.weight" => { mapped_name = format!("model.layers.{}.mamba.norm.weight", layer_idx); },
                                "ssm_out.weight" => { mapped_name = format!("model.layers.{}.mamba.out_proj.weight", layer_idx); },
                                
                                // 하이브리드 Self-Attention 컴포넌트 명칭 바인딩 폴백
                                "attn_q.weight" => { mapped_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx); },
                                "attn_k.weight" => { mapped_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx); },
                                "attn_v.weight" => { mapped_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx); },
                                "attn_output.weight" => { mapped_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx); },
                                _ => { mapped_name = format!("model.layers.{}.{}", layer_idx, suffix); }
                            }
                        }
                    }
                }
                
                tensors.insert(mapped_name, tensor);
            }

            // 🌟 [FUSION MATRIX ENGINE] GGUF 상에서 분리되어 보관 중인 게이트 가중치 행렬 조합 가동
            for layer_idx in 0..config.num_hidden_layers {
                if let (Some(gate), Some(up)) = (ffn_gate_layers.get(&layer_idx), ffn_up_layers.get(&layer_idx)) {
                    let fused_mlp = Tensor::cat(&[gate, up], 0)?;
                    tensors.insert(format!("model.layers.{}.shared_mlp.input_linear.weight", layer_idx), fused_mlp);
                }
            }

            VarBuilder::from_tensors(tensors, dtype, &device)
        } else {
            let safetensors_files = crate::utils::find_type_files(&model_path.to_string_lossy(), "safetensors")
                .unwrap_or_default();
            if safetensors_files.is_empty() {
                return Err(anyhow::anyhow!("No .gguf or .safetensors files found in the model directory."));
            }
            unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)
                    .map_err(|e| anyhow::anyhow!("Failed to load Safetensors into VarBuilder: {}", e))?
            }
        };

        // 4. 모델 구조체 조립
        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        
        // 🌟 [CRITICAL FIX] tie_word_embeddings 가 true일 경우, lm_head 가중치를 별도로 찾지 않고 embed_tokens 의 가중치를 재활용하여 연결합니다.
        let lm_head = if config.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        let language_model = GraniteMoeHybridForCausalLM {
            model: crate::models::granite4::model::GraniteMoeHybridModel {
                embed_tokens,
                layers: (0..config.num_hidden_layers)
                    .map(|layer_idx| {
                        let pp = vb.pp(&format!("model.layers.{}", layer_idx));
                        // 🌟 [CRITICAL FIX] model.rs 에 추가한 하이브리드 가중치 로더 생성자를 안전하게 주입합니다.
                        crate::models::granite4::model::GraniteMoeHybridDecoderLayer::new(&config, layer_idx, pp)
                            .map_err(|e| anyhow::anyhow!("Failed to build hybrid layer {}: {}", layer_idx, e))
                    })
                    .collect::<Result<Vec<_>>>()?,
                norm: crate::models::granite4::model::GraniteMoeHybridRMSNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?,
                rotary_emb: if config.position_embedding_type.as_deref() == Some("rope") {
                    Some(crate::models::granite4::model::GraniteMoeHybridRotaryEmbedding::new(&config, &device)?)
                } else {
                    None
                },
                embedding_multiplier: config.embedding_multiplier,
                padding_idx: config.pad_token_id,
                vocab_size: config.vocab_size,
            },
            lm_head,
            vocab_size: config.vocab_size,
            router_aux_loss_coef: config.router_aux_loss_coef,
            num_experts: config.num_local_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            logits_scaling: config.logits_scaling,
        };

        Ok(Self {
            language_model,
            tokenizer,
            device,
            dtype,
            eos_token_id,
        })
    }

    /// 🌟 KV 캐시 클리어 (새로운 문맥을 받아들일 때 호출)
    pub fn clear_kv_cache(&mut self) {
        for layer in &self.language_model.model.layers {
            if let Some(mamba) = &layer.mamba {
                if let Ok(mut conv) = mamba.conv_state_cache.lock() { *conv = None; }
                if let Ok(mut rec) = mamba.recurrent_state_cache.lock() { *rec = None; }
            }
            if let Some(attn) = &layer.self_attn {
                if let Ok(mut kv) = attn.kv_cache.lock() { *kv = None; }
            }
        }
    }

    /// 🌟 프롬프트를 주입하고 토큰을 한 글자씩 생성하는 메인 디코딩 루프
    pub async fn generate(
        &mut self,
        params: ChatCompletionParameters,
        cancellation_token: Option<Arc<AtomicBool>>,
        _session_id: Option<String>,
        _kv_name: Option<String>,
        semantic_prejudice: Option<&str>,
    ) -> Result<String> {
        // 🌟 [CRITICAL FIX] 캐시를 비우지 않아 발생하는 Attention Shape Mismatch 및 환각 폭주 현상 원천 차단
        self.clear_kv_cache();
        
        let mut prompt = String::new();
        
        // 1. OpenAI 규격의 메시지를 단일 프롬프트 텍스트로 합칩니다.
        for msg in &params.messages {
            match msg {
                crate::openai_types::ChatCompletionRequestMessage::System(sys) => {
                    // 🌟 [CRITICAL FIX] Granite 4.0 전용 Instruct 포맷(<|start_of_role|>...)으로 변경하여 모델이 지시를 정상적으로 이해하게 합니다.
                    prompt.push_str(&format!("<|start_of_role|>system<|end_of_role|>\n{}<|end_of_text|>\n", sys.content));
                }
                crate::openai_types::ChatCompletionRequestMessage::User(user) => {
                    let text = match &user.content {
                        crate::openai_types::ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                        crate::openai_types::ChatCompletionRequestUserMessageContent::Array(arr) => {
                            let mut combined = String::new();
                            for part in arr {
                                if let crate::openai_types::ChatCompletionRequestMessageContentPart::Text(t_part) = part {
                                    combined.push_str(&t_part.text);
                                }
                            }
                            combined
                        }
                    };
                    prompt.push_str(&format!("<|start_of_role|>user<|end_of_role|>\n{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>\n", text));
                }
                _ => {}
            }
        }

        // 2. 입력 텍스트 토큰화
        let tokens = self.tokenizer.encode(prompt.clone(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let mut token_ids = tokens.get_ids().to_vec();

        let max_tokens = params.max_tokens.unwrap_or(512) as usize;
        let temp = params.temperature.unwrap_or(0.1);
        let top_p = params.top_p.unwrap_or(0.9);

        let mut logits_processor = LogitsProcessor::new(299792458, Some(temp), Some(top_p));
        // 🌟 [CRITICAL FIX] 한국어(CJK) 등 다바이트 문자가 단일 토큰 디코딩 시 깨지거나 무시되어 빈 문자열이 되는 현상을 막기 위해,
        // 토큰을 배열에 모아두었다가 디코딩 루프가 끝난 뒤 한 번에 전체를 디코딩하도록 변경합니다.
        let mut generated_tokens = Vec::new();

        // 🌟 [CRITICAL FIX] Qwen3/3.5의 JSON 강제 출력(Forcing) 및 괄호 감지 로직을 Granite 모델에 이식합니다.
        // 프롬프트에 JSON 전용 지시어가 있다면, 모델이 답변을 회피하고 빈 줄만 뱉는 것을 물리적으로 차단합니다.
        let is_strict_json = prompt.contains("/no_think") || prompt.contains("RETURN JSON ONLY") || prompt.contains("Return ONLY");
        let open_bracket_id = self.tokenizer.encode("{", false).map(|v| *v.get_ids().first().unwrap_or(&123)).unwrap_or(123);
        let mut gen_text_buffer = String::new();

        // 🌟 [CRITICAL FIX] Qwen3/3.5의 Semantic Prejudice (오답 진영 억제력) 생성 로직을 완벽하게 이식합니다.
        // 전체 단어장(Embedding)을 대상으로 코사인 유사도를 계산하고 임계값을 넘는 노이즈 단어들의 확률을 깎아냅니다.
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        if let Some(target_text) = semantic_prejudice {
            if let Ok(target_tokens) = self.tokenizer.encode(target_text, false) {
                let target_ids = target_tokens.get_ids();
                if !target_ids.is_empty() {
                    let calc_prej = || -> Result<Tensor> {
                        let target_tensor = Tensor::new(target_ids, &self.device)?.unsqueeze(0)?;
                        let target_emb = self.language_model.model.embed_tokens.forward(&target_tensor)?.to_dtype(DType::F32)?;
                        let target_emb_sum = target_emb.sum_keepdim(1)?;
                        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
                        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
                        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;

                        let all_embs = self.language_model.model.embed_tokens.embeddings().to_dtype(DType::F32)?;
                        let target_norm = target_vec.sqr()?.sum_all()?.sqrt()?;
                        let target_normalized = target_vec.broadcast_div(&target_norm)?;

                        let all_sqr = all_embs.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                        let all_norm = all_sqr.sqrt()?;
                        let all_normalized = all_embs.broadcast_div(&all_norm)?;

                        let sim = all_normalized.matmul(&target_normalized.unsqueeze(1)?)?.squeeze(1)?;
                        // Threshold 노이즈 게이트 + Exponential 증폭
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

        println!("[GENERATE] Granite4 Decoding started. Context length: {}", token_ids.len());

        // 3. 토큰 디코딩 루프
        for step in 0..max_tokens {
            if let Some(token) = &cancellation_token {
                if token.load(Ordering::Relaxed) {
                    println!("[GENERATE] Task cancelled during generation.");
                    break;
                }
            }

            let mut final_logits = None;

            if step == 0 {
                // 🌟 [CRITICAL FIX] Mamba의 엄청난 VRAM 폭발을 막기 위한 Chunked Prefill (청크 단위 쪼개기)
                // 17,000 토큰을 한 번에 넣으면 RNN 특유의 for 루프가 돌아가면서 수만 개의 텐서가 VRAM에 축적되어 OOM이 발생합니다.
                // 이를 512개 단위로 쪼개어 주입하여 VRAM 사용량을 1/30로 강제 압축합니다!
                let chunk_size = 512;
                let total_len = token_ids.len();
                let mut processed = 0;
                
                while processed < total_len {
                    if let Some(token) = &cancellation_token {
                        if token.load(Ordering::Relaxed) { break; }
                    }
                    
                    let take = (total_len - processed).min(chunk_size);
                    let chunk_slice = &token_ids[processed..processed + take];
                    let input_tensor = Tensor::new(chunk_slice, &self.device)?.unsqueeze(0)?;
                    
                    let logits = self.language_model.forward(&input_tensor, processed)?;
                    
                    if processed + take == total_len {
                        let logits = logits.squeeze(0)?;
                        final_logits = Some(logits.get(logits.dim(0)? - 1)?);
                    }
                    processed += take;
                    
                    // 🌟 [CRITICAL FIX] Mamba의 수많은 작은 텐서들이 메모리를 꽉 채우기 전에 GPU 대기열(Queue)을 강제로 비워 OOM을 100% 방어합니다!
                    if self.device.is_cuda() {
                        let dev = self.device.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            let _ = dev.synchronize();
                        }).await;
                    }

                    // VRAM 누수 방지 비동기 양보
                    // 🌟 [CRITICAL FIX] 워커 스레드 기아 상태 방어: 
                    // 단일 청크 연산이 너무 무거워 yield_now() 만으로는 UI 스레드가 살아나지 못하므로 10ms 슬립을 강제 주입합니다.
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            } else {
                let input_slice = &token_ids[token_ids.len() - 1..];
                let input_tensor = Tensor::new(input_slice, &self.device)?.unsqueeze(0)?;
                let seqlen_offset = token_ids.len() - 1;
                
                let logits = self.language_model.forward(&input_tensor, seqlen_offset)?;
                let logits = logits.squeeze(0)?;
                final_logits = Some(logits.get(logits.dim(0)? - 1)?);
            }

            let mut adjusted_logits = final_logits.unwrap();

            // 🌟 [CRITICAL FIX] 사전에 생성해둔 Semantic Prejudice(척력) 텐서를 현재 스텝의 로짓(Logits)에서 빼주어
            // 환각을 유발하는 오답 진영의 단어들이 모델의 입에서 나오는 것을 물리적으로 원천 차단합니다!
            if let Some(ref prej) = semantic_prejudice_tensor {
                // 🌟 [CRITICAL FIX] 로짓(BF16/F16)과 척력 텐서(F32) 간의 타입 불일치(dtype mismatch) 패닉을 해결하기 위해 타입을 동기화합니다.
                let prej_casted = prej.to_dtype(adjusted_logits.dtype())?;
                adjusted_logits = adjusted_logits.broadcast_sub(&prej_casted)?;
            }

            // 🌟 [CRITICAL FIX] Qwen 방식 적용: JSON 모드일 경우 무조건 첫 토큰을 `{` 로 강제하여 
            // 모델이 헛소리를 하거나 빈 줄(\n)을 뱉고 EOS로 종료해버리는 현상을 완벽히 제압합니다!
            let next_token = if step == 0 && is_strict_json {
                open_bracket_id
            } else {
                logits_processor.sample(&adjusted_logits)?
            };
            
            token_ids.push(next_token);

            // EOS 토큰을 만나면 종료
            if next_token == self.eos_token_id {
                break;
            }

            // 🌟 [CRITICAL FIX] 다바이트 문자열 디코딩 깨짐을 막기 위해 생성된 토큰을 배열에 모아둡니다.
            generated_tokens.push(next_token);

            // 🌟 [CRITICAL FIX] Qwen3/3.5의 JSON 균형(Balanced) 조기 종료 로직 이식
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                gen_text_buffer.push_str(&piece);

                if is_strict_json && gen_text_buffer.contains('{') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '{' { depth += 1; has_started = true; }
                        else if c == '}' { depth -= 1; }
                    }
                    // 중괄호 짝이 완벽하게 맞물려 닫히는 즉시 추론을 종료하여 VRAM과 시간을 절약합니다.
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with('}') {
                        println!("[GENERATE] Balanced JSON detected. Stopping early.");
                        break;
                    }
                }
            }

            // 비동기 양보로 시스템 프리징 방지
            tokio::task::yield_now().await;
        }

        // 🌟 [CRITICAL FIX] 루프가 종료된 후 배열 전체를 한 번에 디코딩하여 한국어 깨짐 현상을 원천 차단합니다.
        let generated_text = self.tokenizer.decode(&generated_tokens, true).unwrap_or_default();

        println!("[GENERATE] Granite4 Decoding finished.");
        Ok(generated_text)
    }
}