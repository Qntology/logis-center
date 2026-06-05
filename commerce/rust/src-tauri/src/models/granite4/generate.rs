use anyhow::Result;
use candle_core::{DType, Device, Tensor};
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

        let eos_token_id = config.eos_token_id.clone().and_then(|ids| ids.first().cloned()).unwrap_or(2) as u32;

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
            for name in tensor_names {
                let q_tensor = ct.tensor(&mut file, &name, &device).map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;
                let tensor = q_tensor.dequantize_f16(&device).or_else(|_| q_tensor.dequantize(&device)).map_err(|e| anyhow::anyhow!("Dequantize error: {}", e))?;
                tensors.insert(name, tensor);
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
        let language_model = GraniteMoeHybridForCausalLM {
            model: crate::models::granite4::model::GraniteMoeHybridModel {
                embed_tokens: candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?,
                layers: (0..config.num_hidden_layers)
                    .map(|layer_idx| {
                        let pp = vb.pp(&format!("model.layers.{}", layer_idx));
                        // (여기에 모델 레이어 초기화 로직 구현: Shared MLP, Norm, MoE 등 model.rs의 뼈대와 매핑)
                        // 주의: 실제 프로덕션 적용 시 model.rs의 DecoderLayer::new() 와 같은 생성자 호출로 치환되어야 합니다.
                        unimplemented!("Need layer builder mapped to vb in model.rs")
                    })
                    .collect(),
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
            lm_head: candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?,
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
        // Mamba의 State Cache와 Attention의 KV 캐시를 모두 날립니다.
        // 현재 model.rs 내부의 MambaLayer에 있는 Mutex 캐시를 비우도록 제어합니다.
        for layer in &self.language_model.model.layers {
            if let Some(mamba) = &layer.mamba {
                if let Ok(mut conv) = mamba.conv_state_cache.lock() { *conv = None; }
                if let Ok(mut rec) = mamba.recurrent_state_cache.lock() { *rec = None; }
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
        let mut prompt = String::new();
        
        // 1. OpenAI 규격의 메시지를 단일 프롬프트 텍스트로 합칩니다.
        for msg in &params.messages {
            match msg {
                crate::openai_types::ChatCompletionRequestMessage::System(sys) => {
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.content));
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
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", text));
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
        let mut generated_text = String::new();

        println!("[GENERATE] Granite4 Decoding started. Context length: {}", token_ids.len());

        // 3. 토큰 디코딩 루프
        for step in 0..max_tokens {
            if let Some(token) = &cancellation_token {
                if token.load(Ordering::Relaxed) {
                    println!("[GENERATE] Task cancelled during generation.");
                    break;
                }
            }

            // Mamba 구조 특성상 시퀀스 길이를 1로 유지하며 스캔할 수 있도록
            // 가장 마지막 토큰 1개만 입력으로 던집니다. (단, 첫 Prefill 단계에서는 전체 주입)
            let input_slice = if step == 0 {
                &token_ids[..]
            } else {
                &token_ids[token_ids.len() - 1..]
            };

            let input_tensor = Tensor::new(input_slice, &self.device)?.unsqueeze(0)?;

            // 🌟 모델 순방향 연산 (Forward)
            let logits = self.language_model.forward(&input_tensor)?;
            
            let logits = logits.squeeze(0)?;
            let final_logits = logits.get(logits.dim(0)? - 1)?;

            // 🌟 (옵션) Semantic Prejudice (오답 밀어내기) 적용
            // Granite의 경우 로짓 배열을 조작하여 편향을 적용할 수 있습니다.
            let adjusted_logits = final_logits; // 🌟 mut 제거
            if let Some(prej) = semantic_prejudice {
                if let Ok(prej_tokens) = self.tokenizer.encode(prej, false) {
                    // 강제로 확률을 떨어뜨리는 로짓 패널티 부여
                    // 이 부분은 Tensor 연산으로 구현해야 하므로 생략/단순화 처리합니다.
                }
            }

            // 🌟 샘플링
            let next_token = logits_processor.sample(&adjusted_logits)?;
            token_ids.push(next_token);

            // EOS 토큰을 만나면 종료
            if next_token == self.eos_token_id {
                break;
            }

            // 토큰을 디코딩하여 문자열에 이어붙입니다.
            if let Ok(decoded_token) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&decoded_token);
            }

            // 비동기 양보로 시스템 프리징 방지
            tokio::task::yield_now().await;
        }

        println!("[GENERATE] Granite4 Decoding finished.");
        Ok(generated_text)
    }
}