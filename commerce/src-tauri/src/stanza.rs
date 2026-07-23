use std::collections::HashMap;
use std::path::Path;
use ndarray::Array2;
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;

#[derive(Debug, Clone)]
pub struct StanzaPreprocessor {
    pub word_vocab: HashMap<String, i64>,
    pub char_vocab: HashMap<char, i64>,
    pub id_to_char: HashMap<i64, char>,
    pub upos_vocab: Vec<String>,
    pub word_unk_id: i64,
    pub char_unk_id: i64,
}

impl StanzaPreprocessor {
    pub fn new<P: AsRef<Path>>(vocab_path: P) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(vocab_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to read vocab.json: {}", e))?;
        
        let json_val: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| anyhow::anyhow!("Failed to parse vocab.json as JSON: {}", e))?;
            
        let mut word_vocab: HashMap<String, i64> = HashMap::new();
        let mut char_vocab: HashMap<char, i64> = HashMap::new();
        let mut id_to_char: HashMap<i64, char> = HashMap::new();
        let mut upos_vocab = Vec::new();
        
        // 1. Word Vocab 파싱
        let word_target = if let Some(pos) = json_val.get("pos") {
            pos.get("word").unwrap_or(&json_val)
        } else if let Some(tokenize) = json_val.get("tokenize") {
            tokenize.get("main").unwrap_or(&json_val)
        } else {
            &json_val
        };

        Self::extract_vocab_from_node(word_target, &mut word_vocab);

        // 2. Char Vocab 파싱
        let char_target = if let Some(lemma) = json_val.get("lemma") {
            lemma.get("char").unwrap_or(&serde_json::Value::Null)
        } else if let Some(pos) = json_val.get("pos") {
            pos.get("char").unwrap_or(&serde_json::Value::Null)
        } else if let Some(ner) = json_val.get("ner") {
            ner.get("char").unwrap_or(&serde_json::Value::Null)
        } else {
            &serde_json::Value::Null
        };

        let mut temp_char_vocab: HashMap<String, i64> = HashMap::new();
        Self::extract_vocab_from_node(char_target, &mut temp_char_vocab);
        
        for (k, v) in temp_char_vocab {
            if let Some(c) = k.chars().next() {
                char_vocab.insert(c, v);
                id_to_char.insert(v, c);
            }
        }

        // 3. UPOS Vocab 파싱
        if let Some(pos_node) = json_val.get("pos") {
            if let Some(upos_arr) = pos_node.get("upos").and_then(|v| v.as_array()) {
                for v in upos_arr {
                    if let Some(s) = v.as_str() {
                        upos_vocab.push(s.to_string());
                    }
                }
            }
        }

        if word_vocab.is_empty() {
            return Err(anyhow::anyhow!("vocab.json 내부에서 단어 매핑(Vocab) 구조를 찾을 수 없습니다."));
        }
        
        let word_unk_id = *word_vocab.get("<unk>")
            .or_else(|| word_vocab.get("<UNK>"))
            .or_else(|| word_vocab.get("[UNK]"))
            .unwrap_or(&0);
            
        let char_unk_id = *char_vocab.get(&'<').unwrap_or(&0); 
        
        Ok(Self { word_vocab, char_vocab, id_to_char, upos_vocab, word_unk_id, char_unk_id })
    }

    fn extract_vocab_from_node(target_value: &serde_json::Value, vocab: &mut HashMap<String, i64>) {
        if let Some(arr) = target_value.as_array() {
            for (i, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    vocab.insert(s.to_string(), i as i64);
                } else if let Some(obj) = v.as_object() {
                    let word_opt = obj.get("word").and_then(|w| w.as_str());
                    let id_opt = obj.get("id").and_then(|id| id.as_i64()).unwrap_or(i as i64);
                    if let Some(w) = word_opt {
                        vocab.insert(w.to_string(), id_opt);
                    } else {
                        for (k, val) in obj {
                            if let Some(id_val) = val.get("id").and_then(|id| id.as_i64()) {
                                vocab.insert(k.clone(), id_val);
                            } else if let Some(id_val) = val.as_i64() {
                                vocab.insert(k.clone(), id_val);
                            }
                        }
                    }
                }
            }
        } else {
            let target_obj = if let Some(model) = target_value.get("model") {
                model.get("vocab").and_then(|v| v.as_object())
            } else if let Some(vocab_node) = target_value.get("vocab") {
                vocab_node.as_object()
            } else if let Some(id_to_string) = target_value.get("id_to_string") {
                if let Some(obj) = id_to_string.as_object() {
                    for (id_str, word_val) in obj {
                        if let (Ok(parsed_id), Some(w)) = (id_str.parse::<i64>(), word_val.as_str()) {
                            vocab.insert(w.to_string(), parsed_id);
                        }
                    }
                }
                None
            } else {
                target_value.as_object()
            };

            if let Some(obj) = target_obj {
                for (k, v) in obj {
                    if let Some(id) = v.as_i64() {
                        vocab.insert(k.clone(), id);
                    } else if let Some(s) = v.as_str() {
                        if let Ok(parsed_id) = s.parse::<i64>() {
                            vocab.insert(k.clone(), parsed_id);
                        }
                    } else if let Some(id_val) = v.get("id").and_then(|i| i.as_i64()) {
                        vocab.insert(k.clone(), id_val);
                    } else if v.is_object() || v.is_array() {
                        if let Some(id_val) = v.get("id").and_then(|i| i.as_i64()) {
                            vocab.insert(k.clone(), id_val);
                        }
                    }
                }
            }
        }
    }

    pub fn encode_to_tensor(&self, words: &[&str], session: &Session<'static>) -> Result<Vec<ndarray::ArrayD<i64>>, anyhow::Error> {
        let seq_len = words.len();
        
        if seq_len == 0 {
            return Err(anyhow::anyhow!("입력된 단어 배열이 비어있어 ONNX 텐서 변환을 수행할 수 없습니다."));
        }

        let mut word_ids = Vec::with_capacity(seq_len);
        let mut wlen_vec = Vec::with_capacity(seq_len);
        let mut oidx_vec = Vec::with_capacity(seq_len);
        
        let max_word_len = 32; 
        
        let mut chars_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));
        let mut chars_mask_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));

        for (w_idx, w) in words.iter().enumerate() {
            let token_id = *self.word_vocab.get(*w)
                .or_else(|| self.word_vocab.get(&w.to_lowercase()))
                .unwrap_or(&self.word_unk_id);
            word_ids.push(token_id);
            
            let w_chars: Vec<char> = w.chars().collect();
            let safe_wlen = w_chars.len().min(32);
            wlen_vec.push(safe_wlen as i64);
            oidx_vec.push(w_idx as i64);
            
            for (c_idx, c) in w_chars.iter().take(32).enumerate() {
                let c_id = *self.char_vocab.get(c).unwrap_or(&self.char_unk_id);
                chars_raw[[w_idx, c_idx]] = c_id;
                chars_mask_raw[[w_idx, c_idx]] = 1; 
            }
        }
        
        let word_tensor = ndarray::Array2::from_shape_vec((1, seq_len), word_ids)
            .map_err(|e| anyhow::anyhow!("Failed to build word tensor: {}", e))?.into_dyn();
        let mask_tensor = ndarray::Array2::<i64>::ones((1, seq_len)).into_dyn();
        let chars_tensor = chars_raw.into_dyn();
        let chars_mask_tensor = chars_mask_raw.into_dyn();
        let pre_tensor = ndarray::Array2::<i64>::zeros((1, seq_len)).into_dyn();
        let oidx_tensor = ndarray::Array1::from_vec(oidx_vec).into_dyn();
        let slen_tensor = ndarray::Array1::from_vec(vec![seq_len as i64]).into_dyn();
        let wlen_tensor = ndarray::Array1::from_vec(wlen_vec).into_dyn();
        
        let mut tensor_pool = std::collections::HashMap::new();
        tensor_pool.insert("word", word_tensor.clone());
        tensor_pool.insert("word_mask", mask_tensor.clone());
        tensor_pool.insert("mask", mask_tensor.clone());
        
        tensor_pool.insert("wordchar", chars_tensor.clone());
        tensor_pool.insert("chars", chars_tensor.clone());
        tensor_pool.insert("char", chars_tensor.clone());
        
        tensor_pool.insert("wordchar_mask", chars_mask_tensor.clone());
        tensor_pool.insert("chars_mask", chars_mask_tensor.clone());
        tensor_pool.insert("char_mask", chars_mask_tensor.clone());
        
        tensor_pool.insert("pretrained", pre_tensor.clone());
        tensor_pool.insert("pre", pre_tensor.clone());
        
        let pos_tensor = ndarray::Array2::<i64>::zeros((1, seq_len)).into_dyn();
        tensor_pool.insert("pos", pos_tensor.clone());
        tensor_pool.insert("upos", pos_tensor.clone());
        
        tensor_pool.insert("word_len", wlen_tensor.clone());
        tensor_pool.insert("wordchar_len", wlen_tensor.clone());
        tensor_pool.insert("wlen", wlen_tensor.clone());
        
        tensor_pool.insert("oidx", oidx_tensor.clone());
        tensor_pool.insert("orig", oidx_tensor.clone());
        
        tensor_pool.insert("seq_lengths", slen_tensor.clone());
        tensor_pool.insert("seq", slen_tensor.clone());
        tensor_pool.insert("slen", slen_tensor.clone());

        let mut final_inputs = Vec::new();

        for input_meta in &session.inputs {
            let exact_name = input_meta.name.clone();
            
            if let Some(tensor) = tensor_pool.get(exact_name.as_str()) {
                final_inputs.push(tensor.clone());
            } else {
                return Err(anyhow::anyhow!("ONNX Schema 불일치: 모델이 알 수 없는 입력({})을 요구합니다.", exact_name));
            }
        }
        
        Ok(final_inputs)
    }
}

static STANZA_ENV: once_cell::sync::Lazy<&'static onnxruntime::environment::Environment> = once_cell::sync::Lazy::new(|| {
    Box::leak(Box::new(
        onnxruntime::environment::Environment::builder()
            .with_name("stanza_global_env")
            .build()
            .expect("Failed to initialize global ONNX Runtime Environment")
    ))
});

pub struct StanzaPipeline {
    pub preprocessor: StanzaPreprocessor,
    pub tokenize_session: Session<'static>,
    pub pos_session: Session<'static>,
    pub lemma_session: Session<'static>,
}

impl StanzaPipeline {
    pub async fn ensure_models_downloaded<P: AsRef<Path>>(lang_dir: P, lang: &str) -> anyhow::Result<()> {
        let dir = lang_dir.as_ref();
        if !dir.exists() {
            std::fs::create_dir_all(dir)
                .map_err(|e| anyhow::anyhow!("Stanza 모델 디렉터리 생성 실패 {:?}: {}", dir, e))?;
        }

        let required_files = [
            "vocab.json",
            "tokenizer.onnx",
            "pos.onnx",
            "lemma.onnx",
        ];

        let remote_base_url = format!("https://huggingface.co/stanfordnlp/stanza-{}/resolve/main/onnx", lang);

        for file_name in required_files.iter() {
            let file_path = dir.join(file_name);
            if !file_path.exists() {
                println!("[STANZA] 필수 모델 파일이 존재하지 않습니다: {:?}. 다운로드를 시작합니다...", file_path);
                let download_url = format!("{}/{}", remote_base_url, file_name);

                let response = reqwest::get(&download_url).await
                    .map_err(|e| anyhow::anyhow!("{} 다운로드 요청 실패: {}", file_name, e))?;

                if !response.status().is_success() {
                    return Err(anyhow::anyhow!("{} 다운로드 실패 (HTTP 상태 코드: {})", file_name, response.status()));
                }

                let bytes = response.bytes().await
                    .map_err(|e| anyhow::anyhow!("{} 응답 데이터 읽기 실패: {}", file_name, e))?;

                std::fs::write(&file_path, &bytes)
                    .map_err(|e| anyhow::anyhow!("{} 파일 저장 실패 ({:?}): {}", file_name, file_path, e))?;

                println!("[STANZA] ✅ 다운로드 완료: {:?}", file_path);
            }
        }

        Ok(())
    }

    pub async fn new<P: AsRef<Path>>(base_dir: P, lang: &str) -> anyhow::Result<Self> {
        let lang_dir = base_dir.as_ref().join(lang);

        Self::ensure_models_downloaded(&lang_dir, lang).await?;

        let vocab_path = lang_dir.join("vocab.json");
        let tokenize_path = lang_dir.join("tokenizer.onnx");
        let pos_path = lang_dir.join("pos.onnx");
        let lemma_path = lang_dir.join("lemma.onnx"); 

        let preprocessor = StanzaPreprocessor::new(&vocab_path)?;

        let total_start_time = std::time::Instant::now();

        let env = *STANZA_ENV;

        let tokenize_path_static: &'static str = Box::leak(tokenize_path.to_string_lossy().into_owned().into_boxed_str());
        let pos_path_static: &'static str = Box::leak(pos_path.to_string_lossy().into_owned().into_boxed_str());
        let lemma_path_static: &'static str = Box::leak(lemma_path.to_string_lossy().into_owned().into_boxed_str()); 

        let tok_start_time = std::time::Instant::now();
        println!("[STANZA] TOKENIZER 모델 세션을 빌드합니다...");
        
        let tokenize_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("Tokenizer Session builder error: {}", e))?
            .with_model_from_file(tokenize_path_static)
            .map_err(|e| anyhow::anyhow!("tokenizer.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ TOKENIZER 모델 세션 빌드 완료! (소요 시간: {:.2}초)", tok_start_time.elapsed().as_secs_f32());

        let pos_start_time = std::time::Instant::now();
        println!("[STANZA] POS 모델 세션을 빌드합니다 (onnxruntime 0.0.14)...");
        
        let pos_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("POS Session builder error: {}", e))?
            .with_model_from_file(pos_path_static)
            .map_err(|e| anyhow::anyhow!("pos.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ POS 모델 세션 빌드 완료! (소요 시간: {:.2}초)", pos_start_time.elapsed().as_secs_f32());

        let lemma_start_time = std::time::Instant::now();
        println!("[STANZA] LEMMA 모델 세션을 빌드합니다...");
        
        let lemma_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("Lemma Session builder error: {}", e))?
            .with_model_from_file(lemma_path_static)
            .map_err(|e| anyhow::anyhow!("lemma.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ LEMMA 모델 세션 빌드 완료! (소요 시간: {:.2}초)", lemma_start_time.elapsed().as_secs_f32());

        println!("[STANZA] 🚀 모든 세션 로드 완료! (총 소요 시간: {:.2}초)", total_start_time.elapsed().as_secs_f32());
        
        Ok(Self {
            preprocessor,
            tokenize_session,
            pos_session,
            lemma_session,
        })
    }
}