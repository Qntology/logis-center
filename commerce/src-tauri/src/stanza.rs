use std::collections::HashMap;
use std::path::Path;
use onnxruntime::session::Session;

#[derive(Debug, Clone)]
pub struct StanzaPreprocessor {
    pub word_vocab: HashMap<String, i64>,
    pub char_vocab: HashMap<char, i64>,
    pub tok_char_vocab: HashMap<char, i64>, // 🌟 Tokenizer 전용 독립 Vocab 추가
    pub id_to_char: HashMap<i64, char>, // 🌟 Lemma 복원용 역방향 맵 추가
    pub upos_vocab: Vec<String>,
    // 🌟 [PHASE 2] Depparse 의존관계(UD DEPREL) 레이블 사전.
    // 값 자체를 vocab.json 에서 동적으로 읽으므로 언어별 하드코딩이 전혀 없습니다.
    pub deprel_vocab: Vec<String>,
    pub word_unk_id: i64,
    pub char_unk_id: i64,
    pub tok_char_unk_id: i64, // 🌟 Tokenizer 전용 UNK ID
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
        // 🌟 [PHASE 2] UD DEPREL 레이블 사전 컨테이너 (언어별 vocab.json 에서 동적 수집)
        let mut deprel_vocab: Vec<String> = Vec::new();
        
        // 🌟 1. Word Vocab 파싱 (기존 로직 보존 및 통합)
        let word_target = if let Some(pos) = json_val.get("pos") {
            pos.get("word").unwrap_or(&json_val)
        } else if let Some(tokenize) = json_val.get("tokenize") {
            tokenize.get("main").unwrap_or(&json_val)
        } else {
            &json_val
        };

        Self::extract_vocab_from_node(word_target, &mut word_vocab);

        // 🌟 1.5. Tokenizer Char Vocab 파싱 (Tokenizer 전용 독립 사전)
        let mut tok_char_vocab: HashMap<char, i64> = HashMap::new();
        let tok_target = json_val.get("tokenize").and_then(|t| t.get("main")).unwrap_or(&serde_json::Value::Null);
        
        let mut temp_tok_vocab: HashMap<String, i64> = HashMap::new();
        Self::extract_vocab_from_node(tok_target, &mut temp_tok_vocab);
        
        for (k, v) in temp_tok_vocab {
            if let Some(c) = k.chars().next() {
                tok_char_vocab.insert(c, v);
            }
        }
        let tok_char_unk_id = *tok_char_vocab.get(&'<').unwrap_or(&0); // '<unk>' 처리용

        // 🌟 2. Char Vocab 파싱 (Stanza OOV 극복의 핵심 + Lemma 지원)
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
                id_to_char.insert(v, c); // 🌟 원형(Lemma) 문자열 복원용 생성
            }
        }

        // 🌟 3. UPOS Vocab 동적 파싱 (하드코딩을 파괴하고 파일에서 인덱스 배열 정답을 그대로 수집)
        if let Some(pos_node) = json_val.get("pos") {
            if let Some(upos_arr) = pos_node.get("upos").and_then(|v| v.as_array()) {
                for v in upos_arr {
                    if let Some(s) = v.as_str() {
                        upos_vocab.push(s.to_string());
                    }
                }
            }
        }

        // 🌟 [PHASE 2] 3-1. Depparse DEPREL Vocab 동적 파싱
        // final_definitive_conversion.py 의 MultiVocab 덤프 구조상 depparse.deprel 로 저장되며,
        // 변환본에 따라 rel / dep / main 키로 떨어지는 경우까지 폴백 탐색합니다.
        if let Some(dep_node) = json_val.get("depparse") {
            if let Some(dep_arr) = dep_node.get("deprel").and_then(|v| v.as_array()) {
                for v in dep_arr {
                    if let Some(s) = v.as_str() {
                        deprel_vocab.push(s.to_string());
                    }
                }
            }
            if deprel_vocab.is_empty() {
                for alt_key in ["rel", "dep", "main"] {
                    if let Some(alt_arr) = dep_node.get(alt_key).and_then(|v| v.as_array()) {
                        for v in alt_arr {
                            if let Some(s) = v.as_str() {
                                deprel_vocab.push(s.to_string());
                            }
                        }
                        if !deprel_vocab.is_empty() { break; }
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
            
        let char_unk_id = *char_vocab.get(&'<').unwrap_or(&0); // '<unk>' 처리용
        
        Ok(Self { word_vocab, char_vocab, tok_char_vocab, id_to_char, upos_vocab, deprel_vocab, word_unk_id, char_unk_id, tok_char_unk_id })
    }

    // 🌟 중복된 JSON 파싱 로직을 공통 헬퍼 함수로 분리
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

    /// 품사 태깅(pos.onnx)을 위해 분할된 단어 배열을 Word 텐서와 Wordchar(길이) 텐서로 변환합니다.
    pub fn encode_to_tensor(&self, words: &[&str], session: &Session<'static>, pos_ids: Option<&[i64]>, lemma_ids: Option<&[i64]>) -> Result<Vec<ndarray::ArrayD<i64>>, anyhow::Error> {
        let seq_len = words.len();
        
        // 🌟 [CRITICAL FIX] 빈 배열(seq_len == 0)이 주어지면 ONNX LSTM Reshape 노드에서 치명적인 에러가 발생하므로 사전에 차단합니다.
        if seq_len == 0 {
            return Err(anyhow::anyhow!("입력된 단어 배열이 비어있어 ONNX 텐서 변환을 수행할 수 없습니다."));
        }

        let mut word_ids = Vec::with_capacity(seq_len);
        let mut wlen_vec = Vec::with_capacity(seq_len);
        let mut oidx_vec = Vec::with_capacity(seq_len);
        
        // 🌟 [CRITICAL FIX] Python Export 시 charmodel의 시퀀스 길이가 32로 고정(Hardcoded)되어 있습니다.
        // 동적 길이를 사용하면 ONNX Runtime에서 차원 불일치(Shape Mismatch) 에러가 발생하므로 32로 강제 고정합니다.
        let max_word_len = 32; 
        
        let mut chars_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));
        let mut chars_mask_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));

        for (w_idx, w) in words.iter().enumerate() {
            let token_id = *self.word_vocab.get(*w)
                .or_else(|| self.word_vocab.get(&w.to_lowercase()))
                .unwrap_or(&self.word_unk_id);
            word_ids.push(token_id);
            
            let w_chars: Vec<char> = w.chars().collect();
            // 🌟 [CRITICAL FIX] 32자를 초과하는 단어 길이는 ONNX Gather 연산 시 Out of Bounds 에러를 유발하므로 32로 제한(Clamp)합니다.
            let safe_wlen = w_chars.len().min(32);
            wlen_vec.push(safe_wlen as i64);
            oidx_vec.push(w_idx as i64);
            
            for (c_idx, c) in w_chars.iter().take(32).enumerate() {
                let c_id = *self.char_vocab.get(c).unwrap_or(&self.char_unk_id);
                chars_raw[[w_idx, c_idx]] = c_id;
                chars_mask_raw[[w_idx, c_idx]] = 1; // ONNX Runtime 0.0.14 대응을 위해 bool을 1/0 i64로 강제 래핑
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
        
        // 🌟 [개선] 휴리스틱(조건부) 탐색을 배제하고 ONNX 파이프라인에서 튀어나올 수 있는 모든 변형 스키마를 1:1 Key-Value 매핑
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
        
        let pos_1d_tensor = if let Some(ids) = pos_ids {
            ndarray::Array1::from_vec(ids.to_vec()).into_dyn()
        } else {
            ndarray::Array1::<i64>::zeros(seq_len).into_dyn()
        }; // 🌟 실제 POS 태그 ID 수신 (1D)
        
        // 🌟 [CRITICAL FIX] Depparse를 위해 upos는 2D 텐서 (1, seq_len) 형태로 변환하여 제공해야 함
        let upos_2d_tensor = if let Some(ids) = pos_ids {
            ndarray::Array2::from_shape_vec((1, seq_len), ids.to_vec())
                .unwrap_or_else(|_| ndarray::Array2::<i64>::zeros((1, seq_len)))
                .into_dyn()
        } else {
            ndarray::Array2::<i64>::zeros((1, seq_len)).into_dyn()
        };

        // 🌟 [CRITICAL FIX] Depparse를 위해 lemma_ids를 수신하여 2D 텐서로 매핑
        let lemma_2d_tensor = if let Some(ids) = lemma_ids {
            ndarray::Array2::from_shape_vec((1, seq_len), ids.to_vec())
                .unwrap_or_else(|_| ndarray::Array2::<i64>::zeros((1, seq_len)))
                .into_dyn()
        } else {
            word_tensor.clone() // 🌟 Lemma가 없을 경우 기본 word_tensor로 Fallback하여 차원 에러 방지
        };

        tensor_pool.insert("pos", pos_1d_tensor.clone()); // Lemma 모델 용 1D
        tensor_pool.insert("upos", upos_2d_tensor.clone()); // Depparse 모델 용 2D
        tensor_pool.insert("lemma", lemma_2d_tensor.clone()); // Depparse 모델 용 2D
        
        // 🌟 [CRITICAL FIX] Tokenizer 예외 방지용 더미 텐서 (tokenize_session이 직접 호출될 경우 에러 우회)
        let x_tensor = chars_tensor.clone(); 
        let f_tensor = ndarray::Array3::<i64>::zeros((1, seq_len, 32)).into_dyn();
        tensor_pool.insert("x", x_tensor);
        tensor_pool.insert("f", f_tensor);
        
        // 🌟 Lemma 모델의 필수 입력 텐서(src, src_mask, tgt_in) 매핑 추가
        tensor_pool.insert("src", chars_tensor.clone());
        tensor_pool.insert("src_mask", chars_mask_tensor.clone());
        tensor_pool.insert("tgt_in", chars_tensor.clone());
        
        tensor_pool.insert("word_len", wlen_tensor.clone());
        tensor_pool.insert("wordchar_len", wlen_tensor.clone());
        tensor_pool.insert("wlen", wlen_tensor.clone());
        
        tensor_pool.insert("oidx", oidx_tensor.clone());
        tensor_pool.insert("orig", oidx_tensor.clone());
        
        tensor_pool.insert("seq_lengths", slen_tensor.clone());
        tensor_pool.insert("seq", slen_tensor.clone());
        tensor_pool.insert("slen", slen_tensor.clone());
        tensor_pool.insert("l", slen_tensor.clone()); // Tokenizer 입력 'l' 추가

        let mut final_inputs = Vec::new();

        for input_meta in &session.inputs {
            let exact_name = input_meta.name.clone();
            
            // 모델 메타데이터의 정확한 이름(Exact Key)으로만 풀에서 텐서를 꺼내옵니다.
            if let Some(tensor) = tensor_pool.get(exact_name.as_str()) {
                final_inputs.push(tensor.clone());
            } else {
                // 모델을 있는 그대로 존중하므로, 사전에 정의되지 않은 입력을 모델이 요구할 경우 유추하지 않고 즉시 에러를 반환합니다.
                return Err(anyhow::anyhow!("ONNX Schema 불일치: 모델이 알 수 없는 입력({})을 요구합니다.", exact_name));
            }
        }
        
        Ok(final_inputs)
    }
}

pub static STANZA_ENV: once_cell::sync::Lazy<&'static onnxruntime::environment::Environment> = once_cell::sync::Lazy::new(|| {
    Box::leak(Box::new(
        onnxruntime::environment::Environment::builder()
            .with_name("stanza_global_env")
            .build()
            .expect("Failed to initialize global ONNX Runtime Environment")
    ))
});

// 🌟 [추가] ONNX Runtime 세션을 초기화하고 보유하는 파이프라인 구조체
pub struct StanzaPipeline {
    pub preprocessor: StanzaPreprocessor,
    pub tokenize_session: Session<'static>,
    pub pos_session: Session<'static>,
    pub lemma_session: Session<'static>, // 🌟 Lemma 세션 추가
    pub depparse_session: Session<'static>, // 🌟 Depparse 세션 추가
}

// (로cul 라이브러리 onnxruntime crate 자체에 Send/Sync를 구현하였으므로 더 이상 unsafe 래퍼가 필요 없습니다!)

impl StanzaPipeline {
    /// Stanza 파이프라인에 필요한 필수 모델 파일들의 존재 여부를 체크하고,
    /// 디렉터리나 파일이 없을 경우 원격 서버에서 자동으로 다운로드합니다.
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
            "depparse.onnx", // 🌟 Depparse 파일 추가
        ];

        // Stanza ONNX 모델 파일 저장 원격 Base URL
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

        // 🌟 [자동 다운로드 검사] 세션 생성 전 필요한 모델 파일 존재 여부 검사 및 다운로드 실행
        Self::ensure_models_downloaded(&lang_dir, lang).await?;

        let vocab_path = lang_dir.join("vocab.json");
        let tokenize_path = lang_dir.join("tokenizer.onnx");
        let pos_path = lang_dir.join("pos.onnx");
        let lemma_path = lang_dir.join("lemma.onnx"); // 🌟 Lemma 경로 추가
        let depparse_path = lang_dir.join("depparse.onnx"); // 🌟 Depparse 경로 추가

        let preprocessor = StanzaPreprocessor::new(&vocab_path)?;

        let total_start_time = std::time::Instant::now();

        // onnxruntime 0.0.14 요구사항: Environment 전역 싱글톤 사용 (메모리 릭 방지)
        let env = *STANZA_ENV;

        // 🌟 [onnxruntime 0.0.14 버그 우회] 
        // 구버전 라이브러리의 설계 결함으로 인해, 파일 경로 문자열의 수명(Lifetime)이 
        // Session<'static>과 동일하게 'static으로 유지되어야 컴파일이 통과됩니다.
        // 경로 문자열을 메모리에 영구 고정(Leak)하여 수명 문제를 완벽히 해결합니다.
        let tokenize_path_static: &'static str = Box::leak(tokenize_path.to_string_lossy().into_owned().into_boxed_str());
        let pos_path_static: &'static str = Box::leak(pos_path.to_string_lossy().into_owned().into_boxed_str());
        let lemma_path_static: &'static str = Box::leak(lemma_path.to_string_lossy().into_owned().into_boxed_str()); // 🌟 Leak 생성
        let depparse_path_static: &'static str = Box::leak(depparse_path.to_string_lossy().into_owned().into_boxed_str()); // 🌟 Depparse Leak 추가

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

        let depparse_start_time = std::time::Instant::now();
        println!("[STANZA] DEPPARSE 모델 세션을 빌드합니다...");
        
        let depparse_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("Depparse Session builder error: {}", e))?
            .with_model_from_file(depparse_path_static)
            .map_err(|e| anyhow::anyhow!("depparse.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ DEPPARSE 모델 세션 빌드 완료! (소요 시간: {:.2}초)", depparse_start_time.elapsed().as_secs_f32());

        println!("[STANZA] 모든 세션 로드 완료! (총 소요 시간: {:.2}초)", total_start_time.elapsed().as_secs_f32());
        
        Ok(Self {
            preprocessor,
            tokenize_session,
            pos_session,
            lemma_session,
            depparse_session, // 🌟 Depparse 세션 추가 반환
        })
    }
}