use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};

fn merge_node(obj1: &Value, obj2: &Value) -> Value {
    let mut merged = obj1.clone();
    if let (Some(m_obj), Some(o2_obj)) = (merged.as_object_mut(), obj2.as_object()) {
        for (k, v) in o2_obj {
            let is_empty = match v {
                Value::Null => true,
                Value::String(s) => s.is_empty(),
                Value::Number(n) => n.as_f64().unwrap_or(0.0) == 0.0,
                _ => false,
            };
            if !is_empty {
                m_obj.insert(k.clone(), v.clone());
            }
        }
    }
    merged
}

use tokio::sync::Notify;
use once_cell::sync::Lazy;
use once_cell::sync::OnceCell;

// 🌟 [추가] Stanza ONNX 모델(pos.onnx, depparse.onnx) 입력용 전처리 모듈 (Vocab -> ndarray Tensor)
use std::collections::HashMap;
use std::path::Path;
use ndarray::Array2;
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use onnxruntime::GraphOptimizationLevel;

#[derive(Debug, Clone)]
pub struct StanzaPreprocessor {
    pub word_vocab: HashMap<String, i64>,
    pub char_vocab: HashMap<char, i64>,
    pub id_to_char: HashMap<i64, char>, // 🌟 Lemma 복원용 역방향 맵 추가
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
        
        // 🌟 1. Word Vocab 파싱 (기존 로직 보존 및 통합)
        let word_target = if let Some(pos) = json_val.get("pos") {
            pos.get("word").unwrap_or(&json_val)
        } else if let Some(tokenize) = json_val.get("tokenize") {
            tokenize.get("main").unwrap_or(&json_val)
        } else {
            &json_val
        };

        Self::extract_vocab_from_node(word_target, &mut word_vocab);

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

        if word_vocab.is_empty() {
            return Err(anyhow::anyhow!("vocab.json 내부에서 단어 매핑(Vocab) 구조를 찾을 수 없습니다."));
        }
        
        let word_unk_id = *word_vocab.get("<unk>")
            .or_else(|| word_vocab.get("<UNK>"))
            .or_else(|| word_vocab.get("[UNK]"))
            .unwrap_or(&0);
            
        let char_unk_id = *char_vocab.get(&'<').unwrap_or(&0); // '<unk>' 처리용
        
        Ok(Self { word_vocab, char_vocab, id_to_char, upos_vocab, word_unk_id, char_unk_id })
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
    pub fn encode_to_tensor(&self, words: &[&str], session: &Session<'static>) -> Result<Vec<ndarray::ArrayD<i64>>, anyhow::Error> {
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

static STANZA_ENV: once_cell::sync::Lazy<&'static onnxruntime::environment::Environment> = once_cell::sync::Lazy::new(|| {
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
}

// (로컬 라이브러리 onnxruntime crate 자체에 Send/Sync를 구현하였으므로 더 이상 unsafe 래퍼가 필요 없습니다!)

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

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();

// [UI-SYNC] Instant notification system to wake up the worker
static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static TASK_QUEUED_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters(); // Wake up any sleeping tasks instantly
    println!("[Scheduler] UI signaled ready. Background worker woke up.");
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let _ = PROGRESS_TX.set(ptx);
    let app_handle_prog = app_handle.clone();
    tokio::spawn(async move {
        use tauri::Emitter;
        while let Some(payload) = prx.recv().await {
            if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
                *w = Some(payload.clone());
            }
            let _ = app_handle_prog.emit("extraction-progress", &payload);
        }
    });

    
    // 여기서 다시 spawn 하여 불필요한 DB 락 경쟁을 일으킬 필요가 없습니다.
    
    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        
        let mut oom_retry_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        loop {
            if crate::utils::is_extraction_stopped() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }

            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    match db.get_pending_tasks(5).await {
                        Ok(tasks) => {
                            
                            pending_tasks = tasks.into_iter().filter(|t| t.r#type != "ai_search").collect();
                        },
                        Err(e) => println!("[Scheduler] Failed to fetch tasks: {:?}", e),
                    }
                }
            }

            if pending_tasks.is_empty() {
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => {
                        delay_secs = (delay_secs + 1).min(10); 
                    }
                    _ = TASK_QUEUED_SIGNAL.notified() => {
                        delay_secs = 1;
                        println!("[Scheduler] New task signal received. Waking up immediately.");
                    }
                }
                continue;
            } else {
                delay_secs = 1;
            }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) {
                    println!("[Scheduler] Cancellation detected before starting task {}, skipping batch.", task.id);
                    break;
                }


                println!("[Scheduler] Processing task: {}", task.id);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        // DB의 상태값만 안전하게 1(Processing)로 동기화합니다.
                        let _ = db.update_task_status(&task.id, 1).await;
                        let _ = db.update_message_status(&task.id, 1, Some("Processing...")).await;
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                            *w = Some(json!({ "id": task.id, "ref": task.r#ref, "status": 1 }));
                        }
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { 
                            if let Some(task_val) = w.as_mut() {
                                if let Some(obj) = task_val.as_object_mut() {
                                    obj.insert("status".to_string(), json!(9));
                                    obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                            }
                        }

                        // 일정 시간 뒤에 메모리를 비워주거나, 다음 작업 시작 시 덮어씌워지도록 유지합니다.

                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                            
                            // 마지막 RAM, VRAM 초기화 반영
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
                        
                        let store_guard = store.lock().await;
                        
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                            let _ = db.update_message_status(&task.id, crate::logic::parse_status("complete"), Some("Task Completed")).await;
                        }

                        current_device_pref = None; 
                        oom_retry_map.remove(&task.id); // 성공 시 장부 삭제
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }

                        {
                            let mut model_lock: tokio::sync::MutexGuard<Option<LogisModel>> = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                println!("[Scheduler] Error detected. Performing emergency memory release...");
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;

                            // 에러 발생 시에도 마지막 RAM, VRAM 초기화 반영
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

                        if err_msg.contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             
                             // 여기서 백엔드가 메시지를 다시 생성하거나 이벤트를 쏘면 UI가 좀비처럼 부활하므로 조용히 종료만 합니다.
                             current_device_pref = None;
                             continue;
                        } else if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                            let retries = oom_retry_map.entry(task.id.clone()).or_insert(0);
                            
                            if *retries == 0 {
                                *retries += 1;
                                println!("[Scheduler] OOM Detected! VRAM is purged. Retrying on GPU...");
                                current_device_pref = None;

                                
                                let payload = json!({
                                    "task_id": task.id,
                                    "category": "Warning", "summary": "Memory pressure detected. VRAM cleared. Retrying on GPU...", "spinner": "♻️"
                                });
                                let _ = app_handle.emit("extraction-progress", &payload);

                                
                                let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                let _ = std::fs::remove_file(&log_path);
                                
                                {
                                    let store_guard = store.lock().await;
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, 10).await;
                                        let _ = db.update_message_status(&task.id, 10, Some("Retrying on GPU...")).await;
                                    }
                                }
                                
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                continue; 
                            } else {
                                if task.r#type == "image_extraction" {
                                    let final_err = "High-resolution image exceeds VRAM capacity. Please try a smaller image.";
                                    println!("[Scheduler] GPU retry failed for Vision. Throwing error instead of freezing on CPU.");
                                    let store_guard = store.lock().await;                            
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                        let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", final_err))).await;
                                    }
                                    let _ = app_handle.emit("extraction-progress", json!({
                                        "task_id": task.id,
                                        "category": "Error", "summary": final_err, "spinner": "❌"
                                    }));
                                    current_device_pref = None;
                                } else {
                                    println!("[Scheduler] OOM Detected twice! Activating CPU Mode for text task.");
                                    current_device_pref = Some("cpu".to_string());

                                    // 여기도 더러워진 로그 청소
                                    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                    let _ = std::fs::remove_file(&log_path);

                                    log_task_progress(&app_handle, &task.id, &json!({
                                        "category": "Warning", "summary": "Memory pressure detected. Retrying with CPU Mode...", "spinner": "💾"
                                    }));
                                    
                                    {
                                        let store_guard = store.lock().await;
                                        if let Some(db) = store_guard.as_ref() {
                                            let _ = db.update_task_status(&task.id, 10).await;
                                            let _ = db.update_message_status(&task.id, 10, Some("Retrying in CPU Mode...")).await;
                                        }
                                    }
                                    
                                    tokio::time::sleep(Duration::from_secs(2)).await;
                                    continue;
                                }
                            }
                        } else {
                            let store_guard = store.lock().await;                            
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                            
                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error", "summary": format!("Failed: {}", err_msg), "spinner": "❌"
                            }));

                            current_device_pref = None;
                        }
                    }
                }
            }
            
            cancellation_token.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false); 
        }
    });
}

// 추출된 DOM 뭉치가 실제 원본 PUG의 몇 번째 줄부터 몇 번째 줄까지인지 정확하게 매핑합니다.
pub fn find_block_indices_in_pug<S: AsRef<str>>(full_lines: &[S], block_pug: &str) -> Option<(usize, usize)> {
    let b_lines: Vec<&str> = block_pug.lines().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if b_lines.is_empty() { return None; }
    
    for i in 0..full_lines.len() {
        if full_lines[i].as_ref().trim() == b_lines[0] {
            let mut match_count = 1;
            let mut j = i + 1;
            let mut k = 1;
            while j < full_lines.len() && k < b_lines.len() {
                if full_lines[j].as_ref().trim().is_empty() { j += 1; continue; }
                if full_lines[j].as_ref().trim() == b_lines[k] { match_count += 1; k += 1; } 
                else { break; }
                j += 1;
            }
            if match_count == b_lines.len() { return Some((i, j - 1)); }
        }
    }
    None
}

#[derive(Debug, Clone)]
struct GridCell {
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
    text: String,
    line_indices: Vec<usize>,
}

fn parse_pug_grid(lines: &[String]) -> Vec<GridCell> {
    let mut cells = Vec::new();
    let mut current_row = 0;
    let mut occupied = std::collections::HashSet::new();
    
    let mut i = 0;
    let mut in_tr = false;
    let mut tr_indent = 0;
    
    let re_row = regex::Regex::new(r#"rowspan="(\d+)""#).unwrap();
    let re_col = regex::Regex::new(r#"colspan="(\d+)""#).unwrap();
    
    while i < lines.len() {
        let line = &lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() { i += 1; continue; }
        let indent = line.chars().take_while(|c| c.is_whitespace()).count();
        
        if trimmed.starts_with("tr") && (trimmed.len() == 2 || trimmed.starts_with("tr[") || trimmed.starts_with("tr ")) {
            if in_tr { current_row += 1; }
            in_tr = true;
            tr_indent = indent;
            i += 1;
            continue;
        }
        
        if in_tr && indent <= tr_indent && !trimmed.starts_with("tr") {
            in_tr = false;
            current_row += 1;
        }
        
        if trimmed.starts_with("th") || trimmed.starts_with("td") {
            let is_tag = trimmed.starts_with("th[") || trimmed.starts_with("th ") || trimmed == "th" ||
                         trimmed.starts_with("td[") || trimmed.starts_with("td ") || trimmed == "td" ||
                         trimmed.starts_with("th|") || trimmed.starts_with("td|");
                         
            if is_tag {
                let mut rowspan = 1;
                let mut colspan = 1;
                
                if let Some(cap) = re_row.captures(trimmed) { rowspan = cap[1].parse().unwrap_or(1); }
                if let Some(cap) = re_col.captures(trimmed) { colspan = cap[1].parse().unwrap_or(1); }
                
                let mut current_col = 0;
                while occupied.contains(&(current_row, current_col)) {
                    current_col += 1;
                }
                
                for r in 0..rowspan {
                    for c in 0..colspan {
                        occupied.insert((current_row + r, current_col + c));
                    }
                }
                
                let mut text_acc = String::new();
                let mut line_indices = vec![i];
                
                if let Some(idx) = trimmed.find('|') {
                    text_acc.push_str(trimmed[idx + 1..].trim());
                    text_acc.push(' ');
                }
                
                let cell_indent = indent;
                let mut j = i + 1;
                while j < lines.len() {
                    let sub_line = &lines[j];
                    let sub_trim = sub_line.trim();
                    if sub_trim.is_empty() { j += 1; continue; }
                    let sub_indent = sub_line.chars().take_while(|c| c.is_whitespace()).count();
                    
                    if sub_indent <= cell_indent { break; }
                    
                    if let Some(idx) = sub_trim.find('|') {
                        text_acc.push_str(sub_trim[idx + 1..].trim());
                        text_acc.push(' ');
                    }
                    
                    line_indices.push(j);
                    j += 1;
                }
                
                cells.push(GridCell {
                    row: current_row, col: current_col, rowspan, colspan,
                    text: text_acc.trim().to_string(), line_indices
                });
                
                i = j;
                continue;
            }
        }
        i += 1;
    }
    cells
}

async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    
    
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    
    let zero_addr = "0x0000000000000000000000000000000000000000";
    let from_addr = if task.from.is_empty() { zero_addr.to_string() } else { task.from.clone() };
    let team_id = if task.to.is_empty() || task.to == zero_addr { 
        crate::utils::hash::hash_id(&from_addr) 
    } else { 
        task.to.clone() 
    };

    emit_term("\n=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));

    
    if task.r#type == "analytic_extraction" {
        return crate::analytic::process_analytic_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        emit_term(&format!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id));
    }

    
    let payload = json!({ 
        "task_id": task.id,
        "task_type": task.r#type, 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    
    
    let search_mode = task_data.get("search_mode").and_then(|s| s.as_str()).unwrap_or("commerce").to_string();

    // [FIX] 작업 유형에 따라 파일명을 자동으로 결정합니다.
    let kv_name = if task.r#type == "image_extraction" {
        Some("image".to_string())
    } else {
        Some("text".to_string())
    };
    
    // [FIX] Robust device preference parsing (supports both "cpu" string and true/false boolean)
    let task_device_pref = if let Some(v) = task_data.get("device_preference") {
        if v.as_str() == Some("cpu") || v.as_bool() == Some(true) {
            Some("cpu".to_string())
        } else {
            None
        }
    } else {
        None
    };
    let effective_device_pref = task_device_pref.as_deref().or(device_preference.as_deref());
    
    let language = "english"; 
    let mut doc_lang = "en".to_string(); // 🌟 신규 다국어 감지 변수 추가

    // [LOCK] Acquire Model Access
    let model = {
        println!("[Scheduler] 🛡️ Attempting to acquire Model Lock...");
        let mut model_lock = model_mutex.lock().await;
        println!("[Scheduler] ✅ Model Lock acquired.");
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // [FIX] If current model doesn't match preference, unload it to force switch (CPU <-> GPU)
        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = effective_device_pref == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                println!("[Scheduler] Device preference mismatch (Current CPU: {}, Wants CPU: {}). Reloading model...", m.is_cpu_mode, wants_cpu);
                m.deep_purge_resources().await;
                *model_lock = None;
            }
        }

        if model_lock.is_none() {
            println!("[Scheduler] Model not initialized. Starting LogisModel::new...");
            // [LOG-ONLY] No emit here to keep UI clean
            log_task_progress(app_handle, &task.id, &json!({ "category": "Loading Model", "summary": "Initializing AI Core..." }));
            
            match LogisModel::new(app_handle.clone(), effective_device_pref).await {
                Ok(m) => {
                    println!("[Scheduler] LogisModel::new successful.");
                    *model_lock = Some(m);
                },
                Err(e) => {
                    println!("[Scheduler] ❌ LogisModel::new failed: {}", e);
                    return Err(anyhow::anyhow!("Model Load Failed: {}", e));
                }
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    // 🌟 [CRITICAL FIX] 텍스트 추출 시작 단계에서는 임베딩 모델의 "파일 다운로드 여부"만 가볍게 확인합니다.
    // 실제 텐서 메모리 로딩은 추후 AI가 분석을 끝내고 추출 단계(Stage 3)에 진입하여 모델이 '진짜 쓰일 때' 지연 로딩됩니다!
    if task.r#type != "image_extraction" && task.r#type != "analytic_extraction" {
        model.check_embedding_downloaded().await?;
    }

    // --- Image Extraction Logic (Qwen 3.5 Pipeline) ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        

        if !image_path.is_empty() {
            println!("[Scheduler] Starting Image Extraction for {}", task.id);
            
            
            // log_task_progress(app_handle, &task.id, &json!({ "category": "Vision", "summary": "Analyzing visual context with Qwen 3.5...", "spinner": "⠋" }));
            
            model.extract_from_image(
                task.id.clone(),
                image_path,
                "korean".to_string(),
                search_mode, 
                app_handle,
                Some(cancellation_token.clone()),
                store_mutex,
            ).await?;
            
            return Ok(()); 
        }
    }

    let mut url = task_data.get("href")
        .or_else(|| task_data.get("link"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    let mut origin_candidate = task_data.get("origin")
        .or_else(|| task_data.get("domain"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    
    // 브라우저 자동화 모듈이 감지한 '진짜 현재 활성화 탭 URL'을 강제로 끌어와서 완벽한 절대 주소로 병합(Join)합니다!
    {
        let state = crate::automation::LAST_DETECTED_STATE.lock().await;
        let active_tab_url = state.url.clone();
        
        if !active_tab_url.is_empty() {
            if let Ok(active_parsed) = url::Url::parse(&active_tab_url) {
                let active_origin = format!("{}://{}", active_parsed.scheme(), active_parsed.host_str().unwrap_or("localhost"));
                
                if origin_candidate.is_empty() || origin_candidate.contains("localhost") {
                    origin_candidate = active_origin;
                }
                
                if url.is_empty() {
                    url = active_tab_url;
                } else if !url.starts_with("http") {
                    
                    if let Ok(joined) = active_parsed.join(&url) {
                        url = joined.to_string();
                    }
                }
            }
        }
    }

    
    if !url.starts_with("http") && !origin_candidate.is_empty() && !origin_candidate.contains("localhost") {
        let scheme = if origin_candidate.starts_with("http") { "" } else { "http://" };
        let base_str = format!("{}{}", scheme, origin_candidate);
        if let Ok(base) = url::Url::parse(&base_str) {
            if let Ok(joined) = base.join(&url) {
                url = joined.to_string();
            }
        }
    }
    
    
    let active_task_json = json!({
        "id": task.id.clone(),
        "type": task.r#type.clone(),
        "link": url.clone(),
        "origin": origin_candidate.clone(),
        "ref": task.r#ref.clone(),
        "status": 1, 
        "created_at": task.created_at,
        "updated_at": chrono::Utc::now().timestamp_millis()
    });
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = Some(active_task_json.clone());
    }

    
    if url.is_empty() { 
        return Err(anyhow::anyhow!("Task missing target URL or unsupported type for background extraction.")); 
    }

    // [MEMORY] Fetch and process directly in memory
    let raw_html_content = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        content
    } else if !url.is_empty() {
        let response = reqwest::get(&url).await?;
        let bytes = response.bytes().await?;
        
        // [ENCODING-FIX] UTF-8 First Strategy
        let (decoded_utf8, _, malformed_utf8) = encoding_rs::UTF_8.decode(&bytes);
        let utf8_str = decoded_utf8.as_ref();
        
        // Check for explicit EUC-KR/CP949 markers in the UTF-8 decoded string
        let needs_euc = utf8_str.to_lowercase().contains("charset=euc-kr") || 
                        utf8_str.to_lowercase().contains("charset=\"euc-kr\"") ||
                        utf8_str.to_lowercase().contains("charset=cp949") ||
                        utf8_str.to_lowercase().contains("charset=ks_c_5601");

        if needs_euc && malformed_utf8 {
            // Only use EUC-KR if it's explicitly requested AND UTF-8 decoding had issues
            let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);
            decoded_euc.into_owned()
        } else {
            // Default to UTF-8 (Lossy fallback if needed)
            utf8_str.to_string()
        }
    } else {
        return Ok(());
    };

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);
    
    let mut raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let mut light_pug = model.truncate_pug_context(&raw_pug, false, 2000, None).await;

    // 1. 정확한 토큰 수 측정을 위해 Tokenizer 로드 (파일 경로 탐색)
    let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models")).unwrap_or_default();
    let tokenizer_path = base_path.join("Qwen3-0.6B-Instruct-gguf").to_string_lossy().to_string();
    
    // 1. 모델이 실제로 받게 될 전체 서식을 먼저 만듭니다. (scheduler.rs 539라인 참고)
    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", light_pug);

    // 2. 이 전체 문자열을 인코딩해야 [TEXT-PREFILL]과 100% 일치합니다.
    let mut token_count = raw_system_prefix.len() / 4; // 폴백용

    if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(&tokenizer_path) {
        // light_pug가 아니라 서식이 포함된 raw_system_prefix를 넣습니다.
        token_count = tokenizer.text_encode_vec(raw_system_prefix.clone(), false)
            .map(|v| v.len())
            .unwrap_or(token_count);
    }

    // // 2. 실제 계산된 토큰 수가 3000 이하일 경우 FullContent 모드로 승급
    if token_count <= 6000 {
        println!("[Scheduler] Document is short enough ({} tokens). Upgrading to FullContent Mode...", token_count);
        raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::FullContent, Some(&url));
        light_pug = model.truncate_pug_context(&raw_pug, true, 2000, None).await;
    }

    let base_model_size = if token_count > 60000 {
        crate::model::ModelSize::Qwen
    } else {
        crate::model::ModelSize::Qwen3
    };

    println!("[DEBUG-PUG] Generated PUG. Length: {}. Token Count: {}. Selected Model: {:?}. Snippet: {}...", 
        light_pug.len(), 
        token_count,
        base_model_size,
        light_pug.chars().take(100).collect::<String>().replace("\n", " ")
    );


    use crate::openai_types::{
        ChatCompletionRequestSystemMessage,
        ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent
    };

    let mut page_type = String::new();
    let mut selector_info: serde_json::Value = json!({});
    
    let mut is_detail = task_data.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut skip_ai_analysis = false; 

    let (raw_path, url_obj) = {
        let mut shared_origin = None;
        if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            if let Some(json_val) = mem.as_ref() {
                if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                    if !o.is_empty() && !o.contains("localhost") {
                        let formatted = if o.starts_with("http") { o.to_string() } else { format!("http://{}", o) };
                        if let Ok(u) = url::Url::parse(&formatted) { 
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost"))); 
                        }
                    }
                }
            }
        }
        
        let origin_str = task_data.get("origin")
            .or_else(|| task_data.get("domain"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.contains("localhost"))
            .or(shared_origin)
            .unwrap_or_else(|| if let Ok(task_url) = url::Url::parse(&url) { format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost")) } else { "http://localhost".to_string() });

        let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
        let url_obj = base_url.join(&url).unwrap_or(base_url);
        (url_obj.path().to_string(), url_obj)
    };

    
    let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path));

    {
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            
            
            // 클라우드(aa.ts)는 원본 대소문자를 유지하여 저장하고, 로컬(main.ts)은 소문자로 변환하여 요청합니다.
            // 경로 비교 시 반드시 소문자로 통일하여 검색해야 100% 매칭됩니다!
            let link_val = (url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()).to_lowercase();
            let path_only = url_obj.path().to_lowercase(); 
            
            let mut potential_caches = Vec::new();

            // 1. ID 기반 조회 (정확한 매칭 1차 수집)
            if let Ok(Some(page_doc)) = db.get_item_by_id("pages", &page_id).await {
                potential_caches.push(page_doc);
            } else if let Ok(Some(page_doc)) = db.get_item_by_id("items", &page_id).await {
                potential_caches.push(page_doc);
            }

            // 2. URL 경로 기반 역추적 조회 (대소문자 무시)
            let tables_to_check = ["pages", "items"];
            for tbl in tables_to_check {
                if let Ok(docs) = db.get_all_items(tbl, 1000, 0, None).await {
                    for doc in docs {
                        let json_lower = doc.json_data.to_lowercase();
                        if json_lower.contains(&link_val) || json_lower.contains(&path_only) {
                            if !potential_caches.iter().any(|c| c.id == doc.id) {
                                potential_caches.push(doc);
                            }
                        }
                    }
                }
            }

            // 3. 수집된 캐시 중 현재 DOM 구조와 가장 잘 맞는 캐시 선별
            let mut final_cache = None;

            for page_doc in potential_caches {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&page_doc.json_data) {
                    let cached_detail = val.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                    let node_sel = val.get("node").or_else(|| val.get("parent")).and_then(|v| v.as_str()).unwrap_or("");
                    let item_sel = val.get("item").or_else(|| val.get("itemSelector")).and_then(|v| v.as_str()).unwrap_or("");

                    let target_sel_str = if !node_sel.is_empty() && !item_sel.is_empty() && !item_sel.contains(",") {
                        if item_sel.starts_with(node_sel) { item_sel.to_string() } else { format!("{} {}", node_sel, item_sel) }
                    } else if !item_sel.is_empty() { item_sel.to_string() } else { node_sel.to_string() };

                    
                    let target_sel_clean = target_sel_str.replace(">", " ");

                    if !cached_detail {
                        let mut is_dom_matched = false;
                        if !target_sel_clean.is_empty() {
                            let document = scraper::Html::parse_document(&clean_html_content);
                            is_dom_matched = scraper::Selector::parse(&target_sel_clean)
                                .map(|sel| document.select(&sel).next().is_some())
                                .unwrap_or(false);
                        }

                        if is_dom_matched {
                            // DOM까지 완벽 일치하는 리스트 캐시 -> 최우선 채택 및 탐색 종료
                            final_cache = Some((page_doc, val, false, target_sel_clean));
                            break;
                        } 
                        
                        // (빈 리스트일 가능성보다, 동일한 주소 체계를 가진 상세 페이지일 가능성이 99%이기 때문입니다.)
                    } else {
                        // Detail 캐시인 경우
                        if final_cache.is_none() {
                            final_cache = Some((page_doc, val, true, target_sel_clean));
                        }
                    }
                }
            }

            // 4. 최종 결정된 캐시 적용 및 파이프라인 패스
            if let Some((_page_doc, val, cached_detail, target_sel_str)) = final_cache {
                emit_term(&format!("[Scheduler] ⚡ CACHE HIT! Skipping AI Pre-processing for: {}", raw_path));
                page_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();
                
                
                is_detail = cached_detail; 
                
                selector_info = val.clone();
                selector_info.as_object_mut().unwrap().insert("final_target_selector".to_string(), json!(target_sel_str));
                skip_ai_analysis = true; 
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Loaded valid config from cache.", "spinner": "⚡" }));
            } else {
                emit_term("[Scheduler] Cache miss or elements not found in DOM. Falling back to AI Analysis.");
            }
        }
    }


    // ==================================================================================
    // [ULTRA-OPTIMIZED PIPELINE]
    // Step 0: 0.6B Base Baking [System: PUG] -> Save task_id_base
    // Step 1: 0.6B Loads base -> Ask Classification [User: Task] -> Save task_id_step_a
    // Step 2: 0.6B Loads base -> Ask Selectors [User: Task] -> Save task_id_step_b
    // ==================================================================================

    let base_session_id = format!("{}_base", task.id);
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    
    if !skip_ai_analysis {
        // --- STEP 0: BASE BAKING (공통 컨텍스트 딱 1번만 굽기) ---
        if base_model_size == crate::model::ModelSize::Qwen {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
            if !base_kv_path.exists() {
                println!("[Scheduler] Baking Base PUG Context to SSD...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
                
                
                model.secure_vram_relay(crate::model::ModelSize::Qwen, None, Some(cancellation_token.clone()), true, kv_name.clone()).await?;
                
                
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if let Some(gen) = model.generator.lock().await.as_mut() {
                    
                    // 이렇게 해야 f_ids[kv_len..] 슬라이싱 시 토큰이 엇갈려 환각(Hallucination)이 발생하는 것을 원천 차단할 수 있습니다.
                    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", system_content);
                    
                    // System 메시지(PUG)만 1만 토큰을 읽어서 base_session_id 로 저장합니다.
                    gen.prefill_only(raw_system_prefix, Some(cancellation_token.clone()), Some(base_session_id.clone()), None, kv_name.clone()).await?;
                }
            }
            
            // Qwen 모델을 VRAM에 올리고 사용하지 않을 때 즉시 내림 (임베딩 사용 준비)
            model.deep_purge_resources().await;
        }

        // 🌟 [CRITICAL OPTIMIZATION] PUG 라인 벡터화 및 DOM 파싱을 Step A와 A-2가 공유하도록 단 한 번만 실행합니다!
        // 태그 껍데기를 제외한 '순수 텍스트' 영역만 벡터화하여 연산량을 70% 이상 대폭 단축시킵니다.
        let pug_lines: Vec<String> = light_pug.lines().map(|s| s.to_string()).collect();
        let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
        let mut wiped_indices = vec![false; pug_lines.len()];
        
        // 🌟 [CRITICAL FIX] VRAM(GPU) 사용률 0% 병목 현상 원천 해결!
        // 한 줄씩 CPU가 던지고 기다리던 코드를 대량 일괄(Batch) 처리로 변경하여 GPU 코어를 100% 혹사시킵니다.
        let mut texts_to_embed = Vec::new();
        let mut text_indices = Vec::new();
        
        for (line_idx, line) in pug_lines.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { "" };
            if !text_part.is_empty() {
                texts_to_embed.push(text_part.to_string());
                text_indices.push(line_idx);
            }
        }

        if !texts_to_embed.is_empty() {
            // VRAM 용량 초과 방지를 위해 100줄 단위로 끊어서 GPU에 병렬 주입합니다.
            for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                let start_idx = chunk_idx * 100;
                if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                    for (i, vector) in vectors.into_iter().enumerate() {
                        let original_idx = text_indices[start_idx + i];
                        line_embeddings[original_idx] = vector;
                    }
                }
            }
        }

        let nodes_str = {
            let document_for_boa = scraper::Html::parse_document(&clean_html_content);
            let mut nodes_json = Vec::new();
            let mut node_to_idx = std::collections::HashMap::new();
            for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                node_to_idx.insert(node.id(), idx);
            }
            for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                if let Some(el) = node.value().as_element() {
                    let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                    let text: String = node.children()
                        .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                        .collect::<Vec<_>>().join(" ").trim().to_string();
                    nodes_json.push(serde_json::json!({
                        "index": idx,
                        "parentIndex": parent_idx,
                        "tagName": el.name().to_string(),
                        "id": el.id().unwrap_or("").to_string(),
                        "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                        "text": text,
                        "colspan": el.attr("colspan").unwrap_or("1"),
                        "rowspan": el.attr("rowspan").unwrap_or("1")
                    }));
                } else {
                    nodes_json.push(serde_json::json!(serde_json::Value::Null));
                }
            }
            serde_json::to_string(&nodes_json).unwrap_or_default()
        };

        // --- STEP A: CLASSIFICATION (인메모리 초고속 벡터 및 유니코드 분별 가동) ---
        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting PURE VECTOR DETERMINISTIC RELAY (Step A)");
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Cleaning global noise layouts...", "spinner": "⠋" }));

            let universal_prejudice = "global navigation, menus, footer, aside, search form, search filter.";
            let universal_prej_emb = model.get_embedding(universal_prejudice.to_string()).await.unwrap_or(vec![0.0; 384]);
            let js_template = get_boa_block_extractor_template();

            let mut pre_processed_blocks = std::collections::HashSet::new();
            let mut track_a_candidates = Vec::new();
            let mut seen_candidates = std::collections::HashSet::new(); // 🌟 [핵심 최적화 1] 텍스트 중복 수집 차단으로 JS 엔진 폭주 방지

            for line_idx in 0..pug_lines.len() {
                let text_part = if let Some(idx) = pug_lines[line_idx].find('|') { pug_lines[line_idx][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&universal_prej_emb, &line_embeddings[line_idx]);
                if line_prej_score > 0.55 {
                    if !seen_candidates.contains(text_part) {
                        seen_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                    }
                }
            }

            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len();
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            let track_a_pugs: Vec<(String, String)> = {
                let mut seen_selectors = std::collections::HashSet::new();
                let mut unique_sels = Vec::new();
                for sel in track_a_selectors {
                    if !sel.is_empty() && !seen_selectors.contains(&sel) {
                        seen_selectors.insert(sel.clone());
                        unique_sels.push(sel);
                    }
                }
                
                let html_clone = clean_html_content.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                
                                // 🌟 8개의 스레드가 각각 독립적인 DOM 트리를 구축하고 동시에 CSS 선택자를 사냥합니다!
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };

            // 🌟 [핵심 최적화 3] 완성된 PUG 블록들을 한데 모아 Batch 임베딩 타격! (VRAM 0% 대기현상 완전 파괴)
            let mut unique_pugs_to_embed = Vec::new();
            let mut track_a_pugs_clean = Vec::new();
            for (sel, block_pug) in track_a_pugs {
                if block_pug.is_empty() || pre_processed_blocks.contains(&block_pug) { continue; }
                pre_processed_blocks.insert(block_pug.clone());
                unique_pugs_to_embed.push(block_pug.clone());
                track_a_pugs_clean.push((sel, block_pug));
            }

            let mut block_embeddings_map = std::collections::HashMap::new();
            if !unique_pugs_to_embed.is_empty() {
                for chunk in unique_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            block_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (sel, block_pug) in track_a_pugs_clean {
                let block_emb = block_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                let block_prej_score = cosine_similarity(&universal_prej_emb, &block_emb);
                
                if block_prej_score > 0.50 {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [FRONT-CLEAN] Expunged Global Layout Block: '{}' (Lines {}~{})", sel, start_idx + 1, end_idx + 1));
                        for j in start_idx..=end_idx {
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            let mut pre_filtered_pug = String::new();
            for (idx, line) in pug_lines.iter().enumerate() {
                if !wiped_indices[idx] { pre_filtered_pug.push_str(line); }
                pre_filtered_pug.push_str("\n");
            }
            light_pug = pre_filtered_pug.trim_end().to_string();

            // 🌟 [클린 인계] 이제 노이즈 메뉴가 완벽히 소멸된 상태에서 안전하게 언어 및 카테고리를 식별합니다.
            let mut ko_count = 0;
            let mut ja_count = 0;
            for c in light_pug.chars() {
                let u = c as u32;
                if u >= 0xAC00 && u <= 0xD7A3 { ko_count += 1; }
                else if (u >= 0x3040 && u <= 0x309F) || (u >= 0x30A0 && u <= 0x30FF) { ja_count += 1; }
            }
            
            doc_lang = if ko_count > 5 { "ko".to_string() }
                       else if ja_count > 5 { "ja".to_string() }
                       else { "en".to_string() };

            println!("[Scheduler] Deterministic Detected Language: {}", doc_lang);

            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                }
            };
            
            // 🌟 [CRITICAL FIX] 1000자 자르기를 전면 폐지하고, 타이틀을 독립적으로 벡터화합니다.
            let title_emb = if !doc_title.is_empty() {
                model.get_embedding(doc_title.clone()).await.unwrap_or(vec![0.0; 384])
            } else {
                vec![0.0; 384]
            };

            let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
            let mut best_type = "".to_string();
            let mut max_total_score = -1.0;

            for cat in &categories {
                // bias.json 내의 해당 카테고리에 속한 '모든 속성의 bias 데이터'를 통째로 긁어옵니다.
                let anchor_text = crate::parsing::get_page_type_full_bias(cat, &doc_lang);
                let anchor_emb = model.get_embedding(anchor_text).await.unwrap_or(vec![0.0; 384]);
                
                let mut total_sim = 0.0;
                
                // 1. 타이틀 점수 합산 (타이틀은 페이지 정체성에 매우 중요하므로 5배 가중치 부여)
                if !doc_title.is_empty() {
                    let title_sim = cosine_similarity(&title_emb, &anchor_emb);
                    if title_sim > 0.0 {
                        total_sim += title_sim * 5.0; 
                    }
                }

                // 2. 문서 전체를 끝까지 읽고 판단 (노이즈가 제거된 모든 유효 라인 순회)
                for (i, emb) in line_embeddings.iter().enumerate() {
                    // 노이즈로 판정되어 삭제될 줄(wiped_indices)은 철저히 배제합니다.
                    if wiped_indices[i] { continue; } 
                    
                    let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                    if text_part.is_empty() { continue; }

                    let sim = cosine_similarity(&anchor_emb, emb);
                    
                    // 연관성이 뚜렷한 라인(임계치 0.25 초과)의 점수만 누적하여 전체 페이지의 카테고리 밀도를 정밀 측정합니다.
                    if sim > 0.25 {
                        total_sim += sim;
                    }
                }

                if total_sim > max_total_score {
                    max_total_score = total_sim;
                    best_type = cat.to_string();
                }
            }

            page_type = best_type;
            println!("[Scheduler] Deterministic Classified Page Type: {} (Max Score: {:.4})", page_type, max_total_score);

            if page_type.is_empty() { 
                return Ok(()); 
            }
        }

        // --- STEP A-2: DETAIL CLASSIFICATION (디테일 페이지 여부 판별) ---
        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Is Detail)");

            let (list_bias, form_bias, layout_prejudice) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            
            let prej_emb: Vec<f32> = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            
            // 🌟 [CRITICAL OPTIMIZATION] 중복 생성되던 pug_lines, line_embeddings, nodes_str을 전면 삭제하고 Step A의 데이터를 그대로 계승하여 대기시간을 완전히 소멸시킵니다.
            let system_content_a2 = format!("[PUG CONTENT]\n{}", light_pug);
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Scoring DOM blocks to determine page type...", "spinner": "⠋" }));

            emit_term("\n[CLASSIFICATION] Track B & C Vector Matching (Batch DOM Blocks)...");
            
            let mut list_scores = Vec::new();
            let mut form_scores = Vec::new();

            for (i, emb) in line_embeddings.iter().enumerate() {
                // 🌟 노이즈로 청소된 줄(wiped_indices)과 텍스트가 없는 줄은 평가에서 완벽히 배제합니다.
                if wiped_indices[i] { continue; }
                let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                
                list_scores.push((i, cosine_similarity(&list_bias_emb, emb)));
                form_scores.push((i, cosine_similarity(&form_bias_emb, emb)));
            }

            list_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            form_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut track_bc_candidates = Vec::new();
            let mut track_bc_indices = Vec::new();
            
            for (idx, _) in list_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }
            for (idx, _) in form_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }

            let js_template = get_boa_block_extractor_template();

            let track_bc_selectors: Vec<String> = {
                let target_len = track_bc_candidates.len(); 
                let target_titles_str = serde_json::to_string(&track_bc_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            let valid_bc_count = track_bc_selectors.iter().filter(|s| !s.is_empty()).count();
            emit_term(&format!("  📦 [Track B & C] Boa Engine successfully mapped {}/{} structural processing blocks.", valid_bc_count, track_bc_candidates.len()));

            let track_bc_pugs: Vec<(usize, String, String)> = {
                let html_clone = clean_html_content.clone();
                let selectors_with_idx: Vec<(usize, String)> = track_bc_selectors.into_iter().enumerate().collect();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_selectors = std::collections::HashSet::new();
                    let mut unique_tasks = Vec::new();
                    let mut fallback_results = Vec::new();
                    
                    for (i, sel) in selectors_with_idx {
                        if sel.is_empty() {
                            fallback_results.push((i, sel, String::new()));
                        } else if !seen_selectors.contains(&sel) {
                            seen_selectors.insert(sel.clone());
                            unique_tasks.push((i, sel));
                        } else {
                            fallback_results.push((i, sel, String::new()));
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_tasks.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_tasks.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for (i, sel) in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((i, sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results.extend(fallback_results);
                    results.sort_by_key(|k| k.0); // 🌟 인덱스 정렬 보존 (Track B/C 리스트, 폼 구분 유지)
                    results
                }).await.unwrap_or_default()
            };

            let mut total_list_score = 0.0;
            let mut processed_list_blocks = std::collections::HashSet::new();
            let mut total_form_score = 0.0;
            let mut processed_form_blocks = std::collections::HashSet::new();

            // 🌟 [핵심 최적화 3] 생성된 BC 블록 일괄 Batch 병렬 타격
            let mut unique_bc_pugs_to_embed = Vec::new();
            let mut track_bc_pugs_clean = Vec::new();

            for (i, sel, block_pug) in track_bc_pugs {
                let is_list_track = i < 5;
                if sel.is_empty() { 
                    let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                    emit_term(&format!("  ⚠️ [{}] Anchor Line {} failed to resolve a valid structural parent block via DOM.", track_name, track_bc_indices[i] + 1));
                    continue; 
                }
                
                if is_list_track {
                    if block_pug.is_empty() || processed_list_blocks.contains(&block_pug) { continue; }
                    processed_list_blocks.insert(block_pug.clone());
                } else {
                    if block_pug.is_empty() || processed_form_blocks.contains(&block_pug) { continue; }
                    processed_form_blocks.insert(block_pug.clone());
                }
                unique_bc_pugs_to_embed.push(block_pug.clone());
                track_bc_pugs_clean.push((i, sel, block_pug));
            }

            let mut bc_embeddings_map = std::collections::HashMap::new();
            if !unique_bc_pugs_to_embed.is_empty() {
                for chunk in unique_bc_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            bc_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (i, sel, block_pug) in track_bc_pugs_clean {
                let is_list_track = i < 5;
                let block_emb = bc_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                let b_prej_score = cosine_similarity(&prej_emb, &block_emb);
                
                if is_list_track {
                    let b_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                    let final_score = (b_list_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_list_score += final_score;
                        emit_term(&format!("  📊 [TRACK B (LIST)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_list_score, b_prej_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK B (LIST)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_list_score));
                    }
                } else {
                    let b_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                    let final_score = (b_form_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_form_score += final_score;
                        emit_term(&format!("  📊 [TRACK C (FORM)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_form_score, b_prej_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK C (FORM)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_form_score));
                    }
                }
            }

            is_detail = total_form_score > total_list_score;

            println!("[Scheduler] Classified is_detail as: {} (Total Form: {:.4}, Total List: {:.4})", is_detail, total_form_score, total_list_score);
            emit_term(&format!("  ✅ Determined Detail Page: {}", is_detail));
        } // 👈 🌟 [핵심 변경 1 끝] 0.6B 분석 블록 종료
    } // 🌟 [CRITICAL FIX] 누락된 if !skip_ai_analysis 블록 닫기 괄호를 복구합니다!

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // 임베딩 모델을 VRAM에 올리고 사용하지 않을 때 즉시 내림
    model.deep_purge_resources().await;
    
    // 🌟 [KV CACHE CLEAR] 모델 본체는 살려두고, 이전 단계 연산으로 팽창한 KV 캐시만 제거하여 VRAM을 확보합니다.
    {
        let q3_clear_arc = model.qwen3_generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                gen.clear_kv_cache();
            }
        }).await;
        
        let gen_clear_arc = model.generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = gen_clear_arc.blocking_lock().as_mut() {
                let _ = gen.clear_kv_cache();
            }
        }).await;

        if !model.is_cpu_mode {
            let dev = model.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { let _ = dev.synchronize(); }
            }).await;
        }
    }
    
    // VRAM이 OS에서 완전히 반환될 때까지 잠시 대기합니다.
    wait_for_resources_settled(1200, 800, Some(&cancellation_token), model.device_config.gpu_id as u32).await?;

    let mut extracted_data = json!({});

    // --- PHASE 2 Continue: Detail Extraction (If needed) ---
    if !is_detail {
        
        if !skip_ai_analysis {
            // --- STEP B: SELECTORS (선택자 추출 - JS 기반 신규 로직) ---
            {
                use boa_engine::{Context, Source};
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                println!("[Scheduler] Starting JS-BASED SELECTOR ANALYSIS (LLM Titles -> Boa Engine)");
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Selector Search", "summary": "Analyzing DOM with JS engine...", "spinner": "⠋" }));

                // 1. LLM에게 상품명(titles) 추출 요청
                let title_prompt = parsing::extract_titles_prompt(&page_type);
                let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY.", title_prompt);
                let snapshot_id = format!("{}_step_b_titles", task.id);

                // println!("title_prompt {}", title_prompt);

                let mut titles = Vec::new();
                {
                    let params = ChatCompletionParameters {
                        messages: vec![
                            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                                content: system_content.clone(),
                                name: None,
                            }),
                            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                                name: None,
                            })
                        ],
                        model: if base_model_size == crate::model::ModelSize::Qwen { "qwen".to_string() } else { "qwen3".to_string() }, 
                        max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                        ..Default::default()
                    };

                    let res = if base_model_size == crate::model::ModelSize::Qwen {
                        model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;
                        if let Some(gen) = model.generator.lock().await.as_mut() {
                            println!("[JS-BRIDGE] 1. Requesting titles from LLM (0.6B)...");
                            
                            // 🎯 [SEMANTIC STEERING] 상품 제목에 집중하도록 방향타 고정!
                            let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                            gen.generate(
                                params, 
                                Some(cancellation_token.clone()), 
                                Some(snapshot_id.clone()), 
                                kv_name.clone(),
                                Some(&title_prej) 
                            ).await?
                        } else {
                            return Err(anyhow::anyhow!("Qwen generator missing"));
                        }
                    } else {
                        model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, None).await?;
                        let q3_gen_arc = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                            let mut gen_guard = q3_gen_arc.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                println!("[JS-BRIDGE] 1. Requesting titles from LLM (Qwen3)...");
                                gen.generate(params, Some(cancel_clone), None, Some(&title_prej)).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e)) 
                            } else {
                                Err(anyhow::anyhow!("Qwen3 generator missing"))
                            }
                        }).await??
                    };
                    
                    println!("[JS-BRIDGE] LLM Raw Response: '{}'", res);

                    // res.text 가 아닌 res 를 그대로 파싱
                    let title_info = parsing::parse_json_from_llm(&res);
                        
                    if title_info.as_object().map_or(true, |obj| obj.is_empty()) {
                        return Err(anyhow::anyhow!("LLM returned invalid or unparseable JSON response during title extraction."));
                    }

                    let items_opt = title_info.get("order")
                        .or(title_info.get("goods"))
                        .or(title_info.get("title"))
                        .or(title_info.get("titles"))
                        .or(title_info.get("product"))
                        .and_then(|v| v.as_array());

                    if let Some(items) = items_opt {
                        for item in items {
                            let t_val = if let Some(t) = item.as_str() {
                                Some(t)
                            } else if let Some(t) = item.get("title").and_then(|v| v.as_str()) {
                                Some(t)
                            } else {
                                None
                            };
                            
                            if let Some(t) = t_val {
                                
                                let clean_t = t.replace(",", "").replace(".", "").trim().to_string();
                                let is_only_numbers = !clean_t.is_empty() && clean_t.chars().all(|c| c.is_ascii_digit());
                                
                                if !is_only_numbers {
                                    titles.push(t.to_string());
                                }
                            }
                        }
                    }
                    println!("[JS-BRIDGE] Titles extracted (Robust): {:?}", titles);
                }

                // 다른 LLM 모델(Qwen/Qwen3)을 VRAM에 올리고 사용하지 않을 때 즉시 내림
                model.deep_purge_resources().await;

                if titles.is_empty() {
                    
                    return Err(anyhow::anyhow!("[JS-BRIDGE] No titles extracted from LLM. Aborting task to prevent invalid DOM fallback."));
                }

                // 2. Boa Engine으로 DOM 분석
                {
                    println!("[JS-BRIDGE] 2. Starting boa-engine for DOM analysis...");
                    let mut context = Context::default();
                    
                    let document = scraper::Html::parse_document(&clean_html_content);
                    
                    let mut nodes_json = Vec::new();
                    let mut node_to_idx = std::collections::HashMap::new();

                    // 1단계: 모든 노드 ID 매핑 (부모 참조 안정성 확보)
                    for (idx, node) in document.tree.root().descendants().enumerate() {
                        node_to_idx.insert(node.id(), idx);
                    }

                    // 2단계: 노드 정보 수집 (Element 노드 중심) 
                    for (idx, node) in document.tree.root().descendants().enumerate() {
                        if let Some(el) = node.value().as_element() {
                            let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                            
                            let text: String = node.children()
                                .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                                .collect::<Vec<_>>()
                                .join(" ")
                                .trim()
                                .to_string();
                                
                            
                            nodes_json.push(json!({
                                "index": idx,
                                "parentIndex": parent_idx,
                                "tagName": el.name().to_string(),
                                "id": el.id().unwrap_or("").to_string(),
                                "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                                "text": text,
                                "colspan": el.attr("colspan").unwrap_or("1"),
                                "rowspan": el.attr("rowspan").unwrap_or("1")
                            }));
                        } else {
                            nodes_json.push(json!(null));
                        }
                    }
                    
                    let nodes_str = serde_json::to_string(&nodes_json)?;
                    let titles_str = serde_json::to_string(&titles)?;

                    let js_template = get_boa_js_template();


                    let js_code = js_template
                        .replace("NODES_PLACEHOLDER", &nodes_str)
                        .replace("TITLES_PLACEHOLDER", &titles_str);

                    match context.eval(Source::from_bytes(js_code.as_bytes())) {
                        Ok(val) => {
                            let res_str = val.as_string().unwrap().to_std_string_escaped();
                            println!("[JS-BRIDGE] Boa Final Result: {}", res_str);

                            selector_info = serde_json::from_str(&res_str).unwrap_or(json!({}));
                        },
                        Err(e) => {
                            println!("[JS-BRIDGE] Error executing JS: {:?}", e);
                        }
                    }
                }
            }
        } // 👈 🌟 [핵심 변경 2 끝] JS 선택자 분석 스킵 괄호 닫기!

        
        let target_selector = selector_info.get("final_target_selector")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                let item_selector = selector_info.get("itemSelector")
                    .or_else(|| selector_info.get("item"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let node_selector = selector_info.get("node").or_else(|| selector_info.get("parent")).and_then(|s| s.as_str()).unwrap_or("");
                
                if !node_selector.is_empty() && !item_selector.is_empty() && !item_selector.contains(",") {
                    if item_selector.starts_with(node_selector) {
                        item_selector.to_string()
                    } else {
                        format!("{} {}", node_selector, item_selector) 
                    }
                } else if !item_selector.is_empty() { 
                    item_selector.to_string() 
                } else { 
                    node_selector.to_string() 
                }
            }).replace(">", " "); 
            
        emit_term(&format!("[Scheduler] Target Selector configured as: '{}'", target_selector));

        let mut final_thead_selector = String::new();
        let mut cache_updated = false; // DB 업데이트가 필요한지 추적하는 플래그
        let mut thead_pug = String::new();

        // 1. 사용자 요청대로 'head' 키로 캐시된 선택자가 있는지 확인합니다.
        if let Some(sel) = selector_info.get("head").and_then(|v| v.as_str()) {
            if !sel.is_empty() && sel != "..." {
                final_thead_selector = sel.to_string();
                println!("[Scheduler] Using cached head selector: {}", final_thead_selector);
            }
        } 
        
        // 2. 캐시가 없거나 비어있는 경우 AI를 통해 테이블 헤더 구조를 분석합니다.
        if final_thead_selector.is_empty() {
            // Document를 다시 생성하여 안전하게 target_selector 기반으로 샘플 첫 행(ref_row)을 뽑아냅니다.
            let reference_row_for_thead = {
                let clean_content = &clean_html_content;
                let document = scraper::Html::parse_document(clean_content);
                if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                    document.select(&sel).next().map(|first_match| {
                        let mut temp_pug = String::new();
                        crate::parsing::generate_pug_lines((*first_match).into(), 0, &mut temp_pug, &PugMode::FullContent, &mut None);
                        temp_pug.trim().to_string()
                    })
                } else { None }
            };

            if let Some(ref_row) = reference_row_for_thead {
                if !ref_row.is_empty() {
                    log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Analyzing table header structure...", "spinner": "⠋" }));
                    
                    
                    let ref_row_context_size = ref_row.len() + 2000;
                    let full_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
                    let thead_light_pug = model.truncate_pug_context(&full_pug, false, 0, Some(ref_row_context_size)).await;

                    println!("ref_row: {}", ref_row);
                    
                    let thead_prompt = crate::parsing::extract_table_structure_prompt(&page_type, &target_selector, &thead_light_pug, &ref_row);
                    let params = ChatCompletionParameters {
                        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(thead_prompt),
                            name: None,
                        })],
                        model: "qwen3.5".to_string(),
                        max_tokens: Some(256), 
                        temperature: Some(0.0), 
                        top_p: Some(0.95),
                        ..Default::default()
                    };

                    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                    if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                        // 🌟 [CRITICAL FIX] Qwen 3.5 생성기의 파라미터가 추가되었으므로 마지막 인자로 None(semantic_prejudice)을 추가로 전달합니다.
                        if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_step_thead", task.id)), kv_name.clone(), None, None).await {
                            let thead_json = crate::parsing::parse_json_from_llm(&res);
                            
                            // JSON 응답에서 page_type에 맞는 선택자 추출
                            let mut thead_val = thead_json.get(&page_type);
                            if thead_val.is_none() {
                                if let Some(obj) = thead_json.as_object() {
                                    for (k, v) in obj {
                                        if k.to_lowercase() == page_type.to_lowercase() { thead_val = Some(v); break; }
                                    }
                                }
                            }

                            // 1. thead 선택자 추출 (Flat 구조에 맞추어 get("table") 제거)
                            final_thead_selector = thead_val
                                .and_then(|v| v.get("thead"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " "); 
                            
                            // 2. tbody와 thead를 감싸는 부모 wrapper(table) 선택자 추출
                            let final_table_selector = thead_val
                                .and_then(|v| v.get("table"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " ");

                            
                            if !final_thead_selector.is_empty() && final_thead_selector != "..." && !final_table_selector.is_empty() && final_table_selector != "..." {
                                if !final_thead_selector.contains(&final_table_selector) {
                                    let combined_sel = format!("{} {}", final_table_selector, final_thead_selector);
                                    let doc = scraper::Html::parse_document(&clean_html_content);
                                    
                                    // 컴파일 에러 해결: Result의 에러 객체가 참조를 붙잡지 않도록 즉시 boolean으로 변환 후 스코프를 닫습니다.
                                    let is_valid = scraper::Selector::parse(&combined_sel)
                                        .map(|parsed_sel| doc.select(&parsed_sel).next().is_some())
                                        .unwrap_or(false);

                                    if is_valid {
                                        final_thead_selector = combined_sel;
                                    }
                                }
                            }

                            if !final_thead_selector.is_empty() && final_thead_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("head".to_string(), json!(final_thead_selector.clone()));
                                println!("[Scheduler] AI determined head selector and cached: {}", final_thead_selector);
                                cache_updated = true; // 새로운 head를 찾았으므로 DB 업데이트 예약
                            }

                            
                            if !final_table_selector.is_empty() && !final_table_selector.contains("CSS selector") && final_table_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("wrapper".to_string(), json!(final_table_selector.clone()));
                                println!("[Scheduler] AI determined table wrapper selector and cached: {}", final_table_selector);
                                cache_updated = true;
                            }
                        }
                    }
                }
            }
        }

        // 다른 LLM 모델(Qwen 3.5)을 VRAM에 올리고 사용하지 않을 때 즉시 내림
        model.deep_purge_resources().await;

        // 3. 최종 결정된 selector를 사용하여 head PUG를 추출합니다.
        if !final_thead_selector.is_empty() && final_thead_selector != "..." {
            let clean_content = &clean_html_content;
            let doc = scraper::Html::parse_document(clean_content);
            if let Ok(tsel) = scraper::Selector::parse(&final_thead_selector) {
                if let Some(first_match) = doc.select(&tsel).next() {
                    
                    let mut target_node = first_match;
                    let mut current = target_node.parent();
                    
                    while let Some(parent) = current {
                        if let Some(el) = parent.value().as_element() {
                            let tag = el.name().to_lowercase();
                            if tag == "thead" || tag == "tr" {
                                if let Some(wrapped) = scraper::ElementRef::wrap(parent) {
                                    target_node = wrapped;
                                    // thead를 찾으면 가장 완벽한 다중 행 헤더 그룹이므로 즉시 탐색 종료
                                    if tag == "thead" { break; } 
                                }
                            }
                        }
                        current = parent.parent();
                    }
                    
                    let mut tpug = String::new();
                    
                    crate::parsing::generate_pug_lines((*target_node).into(), 0, &mut tpug, &PugMode::TheadMode, &mut None);
                    thead_pug = tpug.trim().to_string();

                    if !thead_pug.is_empty() {
                        println!("[Scheduler] 🎉 thead_pug extraction successful ({} bytes)", thead_pug.len());
                    }
                }
            }
        }

        // 4. DB 저장을 head 추출 이후로 실행하여 head 정보를 포함한 selector_info를 영구 저장합니다.
        if !skip_ai_analysis || cache_updated {
            let store = {
                let store_guard = store_mutex.lock().await;
                store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
            };
            
            let mut shared_origin = None;
            let mut shared_type = None;
            if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
                if let Some(json_val) = mem.as_ref() {
                    if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                        if let Ok(u) = url::Url::parse(o) {
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost")));
                        }
                    }
                    if let Some(t) = json_val.get("type").and_then(|v| v.as_str()) {
                        if !t.is_empty() { shared_type = Some(t.to_string()); }
                    }
                }
            }

            let origin_str = task_data.get("origin")
                .or_else(|| task_data.get("domain"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string())
                .filter(|s| !s.contains("localhost")) 
                .or(shared_origin) 
                .unwrap_or_else(|| {
                    if let Ok(task_url) = url::Url::parse(&url) {
                        format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost"))
                    } else {
                        "http://localhost".to_string()
                    }
                });

            if page_type.is_empty() || page_type == "unknown" {
                if let Some(st) = shared_type { page_type = st; }
            }
                
            let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
            let url_obj = base_url.join(&url).unwrap_or(base_url);
            let raw_path = url_obj.path();
            let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path)); 
            
            let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));

            let ref_for_page = if !task.r#ref.is_empty() { &task.r#ref } else { raw_path };

            
            if !is_detail {
                let mut page_data: serde_json::Value = selector_info.clone();
                if let Some(obj) = page_data.as_object_mut() {
                    obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                    obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                    obj.insert("type".to_string(), json!(page_type.clone()));
                    
                    if let Some(item_sel) = selector_info.get("itemSelector") { obj.insert("item".to_string(), item_sel.clone()); }
                    if let Some(parent_sel) = selector_info.get("parent") { obj.insert("node".to_string(), parent_sel.clone()); }
                    obj.insert("detail".to_string(), json!(false));
                }

                
                let _ = store.upsert_item("pages", &page_id, &page_type, page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
                
                println!("[Scheduler] Page cache updated in DB (including head selector).");

                
                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, task.cc.to_uppercase()));
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    "detail": true,
                    "node": true,
                    "item": ""
                });
                let _ = store.upsert_item("pages", &detail_page_id, &page_type, detail_page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &detail_page_id, "pages", detail_page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;

            } else {
                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, task.cc.to_uppercase()));
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    "detail": true,
                    "node": true,
                    "item": ""
                });
                let _ = store.upsert_item("pages", &detail_page_id, &page_type, detail_page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &detail_page_id, "pages", detail_page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
            }
        }
        
        // [LIST MODE] 지능형 리스트 추출 (LLM 기반)
        let list_log = json!({ "category": "List Processing", "summary": "Extracting list data with LLM...", "spinner": "⠋" });
        log_task_progress(app_handle, &task.id, &list_log);

        let mut all_extracted_items = Vec::new();
        
        let mut pug_list = {
            let clean_content = &clean_html_content;
            let document = scraper::Html::parse_document(clean_content);
            
            parsing::split_doc_to_pug_list_advanced(
                &document, 
                &target_selector, 
                PugMode::ListMode, 
                None,
                Some(&url) 
            )
        };

        // 2순위: 본문 아이템이 여러 줄(tr)로 구성된 경우 완벽하게 묶어냅니다.
        let mut group_size = if !thead_pug.is_empty() {
            let mut max_span = 1;
            // [CRITICAL FIX] colspan은 가로 병합이므로 세로 행 그룹화에 포함하면 다중 병합 오류가 발생합니다. rowspan만 추출합니다.
            if let Ok(re) = regex::Regex::new(r#"rowspan="(\d+)""#) {
                for cap in re.captures_iter(&thead_pug) {
                    if let Ok(val) = cap[1].parse::<usize>() {
                        if val > max_span {
                            max_span = val;
                        }
                    }
                }
            }
            
            if max_span > 1 {
                max_span
            } else {
                thead_pug.lines().filter(|line| {
                    let s = line.trim_start();
                    s == "tr" || s.starts_with("tr[")
                }).count().max(1)
            }
        } else {
            1
        };

        if group_size > 1 && !pug_list.is_empty() {
            // [CRITICAL FIX] 이미 split_doc_to_pug_list_advanced 내부에서 병합되어 반환된 경우 이중 병합을 방지합니다.
            let first_item_tr_count = pug_list.first()
                .map(|p| p.lines().filter(|l| {
                    let indent = l.chars().take_while(|c| c.is_whitespace()).count();
                    indent == 0 && (l.starts_with("tr") || l.starts_with("tr["))
                }).count())
                .unwrap_or(1);

            // 이미 PUG 문자열 내부에 tr 태그가 group_size 만큼(혹은 그 이상) 존재한다면, 이미 완벽하게 그룹화된 상태입니다.
            if first_item_tr_count >= group_size || first_item_tr_count > 1 {
                println!("[Scheduler] 🌟 Items are already grouped ({} trs per item). Skipping manual chunking.", first_item_tr_count);
                group_size = 1;
            } else {
                let mut grouped = Vec::new();
                for chunk in pug_list.chunks(group_size) {
                    grouped.push(chunk.join("\n"));
                }
                pug_list = grouped;
                println!("[Scheduler] 🌟 Grouped multi-row items: {} rows per item. Total items reduced to {}.", group_size, pug_list.len());
            }
        }

        if !pug_list.is_empty() {
            let total_items = pug_list.len();

            // 리스트 전용 스키마 정의를 호출하여 핵심 필드만 개별 추출
            let fields = parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            // 환각 검증을 위한 문서 타이틀 추출
            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                }
            };

            // 모델 가중치 변경(스위칭) 없이 가장 가벼운 모델인 Qwen3 하나만으로 전체 파이프라인을 관통하여 속도를 극대화합니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            // 🌟 [NOISE FILTER] bias.json의 layout_list.prejudice 값을 가져와 노이즈 필터링용 벡터를 생성합니다.
            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            // [최적화] thead는 모든 아이템에 공통으로 적용되므로 루프 바깥에서 단 한 번만 벡터화합니다.
            let mut thead_lines: Vec<String> = thead_pug.lines().map(|s| s.to_string()).collect();
            let mut thead_embeddings = vec![vec![0.0; 384]; thead_lines.len()];
            
            // 🌟 [GRID ALIGNMENT] thead 파싱 및 컬럼 인덱스별 헤더 텍스트 머지
            let thead_cells = parse_pug_grid(&thead_lines);
            let mut header_cols: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
            for cell in &thead_cells {
                for c in cell.col..(cell.col + cell.colspan) {
                    let existing = header_cols.entry(c).or_insert(String::new());
                    if !existing.is_empty() && !cell.text.is_empty() {
                        existing.push_str(" > ");
                    }
                    if !cell.text.is_empty() {
                        existing.push_str(&cell.text);
                    }
                }
            }

            if !thead_lines.is_empty() {
                emit_term(&format!("\n[PRE-PROCESSING] Vectorizing Table Header ({} lines)...", thead_lines.len()));
                
                // 🌟 배치 임베딩 적용
                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in thead_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        texts_to_embed.push(line.to_string());
                        text_indices.push(line_idx);
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                        let start_idx = chunk_idx * 100;
                        if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                            for (i, vector) in vectors.into_iter().enumerate() {
                                let original_idx = text_indices[start_idx + i];
                                let emb = vector.clone();
                                let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                                
                                if noise_score > 0.55 {
                                    emit_term(&format!("    🚫 [NOISE FILTERED] Header Line {} : {} (Score: {:.4})", original_idx + 1, text_chunk[i].trim(), noise_score));
                                    thead_lines[original_idx] = String::new(); 
                                } else {
                                    thead_embeddings[original_idx] = emb;
                                }
                            }
                        }
                    }
                }
            }

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                let percent = (((idx as f32) / (total_items as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Item {}/{}", idx + 1, total_items), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("\n[STAGE-3] Processing List Item {}/{} ...", idx + 1, total_items));

                // 헤더(Thead)와 개별 아이템(Item)의 PUG를 결합하여 검증 텍스트 생성 (나중의 본문 텍스트 검증용)
                let full_item_pug = format!("{}\n{}", thead_pug, item_pug);
                
                // [수정] 아이템 영역만 분리하여 가볍게 벡터화 진행 및 노이즈 필터링
                let mut item_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                
                // 🌟 [GRID ALIGNMENT] tbody 아이템 셀 파싱 및 thead 헤더 텍스트 융합
                let item_cells = parse_pug_grid(&item_lines);
                let mut line_enriched_texts = vec![String::new(); item_lines.len()];
                
                for cell in &item_cells {
                    let h_text = header_cols.get(&cell.col).cloned().unwrap_or_default();
                    for &line_idx in &cell.line_indices {
                        let original_text = if let Some(p) = item_lines[line_idx].find('|') {
                            item_lines[line_idx][p + 1..].trim()
                        } else {
                            ""
                        };
                        if !original_text.is_empty() {
                            line_enriched_texts[line_idx] = if h_text.is_empty() {
                                original_text.to_string()
                            } else {
                                format!("{} | {}", h_text, original_text)
                            };
                        }
                    }
                }

                let mut item_embeddings = vec![vec![0.0; 384]; item_lines.len()];
                
                // 🌟 배치 임베딩 적용
                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {
                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" }
                        } else {
                            enriched.as_str()
                        };

                        if !target_text.is_empty() {
                            texts_to_embed.push(target_text.to_string());
                            text_indices.push(line_idx);
                        }
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                        let start_idx = chunk_idx * 100;
                        if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                            for (i, vector) in vectors.into_iter().enumerate() {
                                let original_idx = text_indices[start_idx + i];
                                let emb = vector.clone();
                                let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                                
                                if noise_score > 0.55 {
                                    emit_term(&format!("    🚫 [NOISE FILTERED] Item Line {}/{} : {} (Score: {:.4})", original_idx + 1, item_lines.len(), text_chunk[i].trim(), noise_score));
                                    item_lines[original_idx] = String::new(); 
                                } else {
                                    emit_term(&format!("    [VECTORIZING] Item Line {}/{} : {}", original_idx + 1, item_lines.len(), text_chunk[i].trim()));
                                    item_embeddings[original_idx] = emb;
                                }
                            }
                        }
                    }
                }

                // 🌟 [CRITICAL FIX] LLM에게 노이즈가 제거된 [VECTORIZING] 텍스트들만 모아서 전달하여 완벽한 추출 환경 구성
                let mut json_contexts = Vec::new();
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {
                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() }
                        } else {
                            enriched.as_str()
                        };
                        if !target_text.is_empty() {
                            if let Some(idx) = target_text.find('|') {
                                json_contexts.push(json!({
                                    "metadata": target_text[..idx].trim(),
                                    "value": target_text[idx + 1..].trim()
                                }));
                            } else {
                                json_contexts.push(json!({
                                    "value": target_text.trim()
                                }));
                            }
                        }
                    }
                }
                let filtered_full_item_pug = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();

                let mut item_val = json!({});
                let mut global_ignore_list: Vec<String> = Vec::new();
                
                // 🌟 String 벡터를 &str 배열로 참조 변환하여 이후 컨텍스트 추출기에 완벽 호환되게 연결합니다.
                let thead_lines_ref: Vec<&str> = thead_lines.iter().map(|s| s.as_str()).collect();
                let item_lines_ref: Vec<&str> = item_lines.iter().map(|s| s.as_str()).collect();

                // 상세 페이지와 완벽히 동일하게 필드별로 순회하며 개별 타격 추출
                for (f_idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.clone().into_iter().enumerate() {
                    
                    // 2. Bias와 Prejudice 임베딩 (인메모리 코사인 검색용)
                    let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);
                    let prej_emb = if prejudice_target.trim().is_empty() { 
                        vec![0.0; 384] 
                    } else { 
                        model.get_embedding(prejudice_target.clone()).await.unwrap_or(vec![0.0; 384]) 
                    };
                    
                    // Header 영역 독립 매칭
                    let mut best_thead_idx = 0;
                    let mut best_thead_score = -1.0;
                    for (i, emb) in thead_embeddings.iter().enumerate() {
                        if thead_lines_ref[i].trim().is_empty() { continue; }
                        let b_score = cosine_similarity(&bias_emb, emb);
                        let p_score = if prejudice_target.trim().is_empty() { 0.0 } else { cosine_similarity(&prej_emb, emb) };
                        let final_score = b_score - p_score;

                        if final_score > best_thead_score {
                            best_thead_score = final_score;
                            best_thead_idx = i;
                        }
                    }
                    // 🌟 [CRITICAL FIX] extract_pug_context Slicing 비활성화!
                    // 점수 기반 검색은 로그 출력 및 검증용으로만 계산하고, 실제 LLM에게는 rowspan/colspan 구조적 대칭성이 
                    // 완벽하게 보존된 전체 블록(full_item_pug)을 무조건 넘깁니다.
                    
                    let mut best_item_idx = 0;
                    let mut best_item_score = -1.0;
                    for (i, emb) in item_embeddings.iter().enumerate() {
                        if item_lines_ref[i].trim().is_empty() { continue; }
                        let b_score = cosine_similarity(&bias_emb, emb);
                        let p_score = if prejudice_target.trim().is_empty() { 0.0 } else { cosine_similarity(&prej_emb, emb) };
                        let final_score = b_score - p_score;

                        if final_score > best_item_score {
                            best_item_score = final_score;
                            best_item_idx = i;
                        }
                    }
                    
                    // 🌟 [개선] 리스트 아이템은 크기가 작으므로 특정 줄만 잘라서 던지지 않고, 
                    // 노이즈가 완벽히 필터링된 직관적인 Key-Value 묶음(filtered_full_item_pug)을 통째로 던집니다.
                    let targeted_pug = filtered_full_item_pug.clone();
                    
                    emit_term(&format!("    🎯 [MATCHED CONTEXT] Field: '{}' | Header Score: {:.4} | Item Score: {:.4} (Using filtered structure)", field_name, best_thead_score, best_item_score));
                    
                    let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: format!("[JSON CONTEXT]\n{}", targeted_pug),
                        name: None,
                    });
                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                    
                    let f_percent = (((f_idx as f32) / (total_fields as f32)) * 100.0) as i32;
                    let f_summary_msg = format!("Extracting {} ({}%)...", field_name, f_percent);
                    
                    let payload = json!({ 
                        "task_id": task.id, 
                        "category": format!("List Item {}/{}", idx + 1, total_items), 
                        "summary": f_summary_msg, 
                        "spinner": "⠋" 
                    });
                    log_task_progress(app_handle, &task.id, &payload);
                    emit_term(&format!("  ▶ {}", f_summary_msg));

                    let mut metadata_str = String::new();
                    let mut target_data_str = String::new();
                    for line in targeted_pug.lines() {
                        if let Some(idx) = line.find('|') {
                            metadata_str.push_str(line[..idx].trim());
                            metadata_str.push_str("\n");
                            target_data_str.push_str(line[idx + 1..].trim());
                            target_data_str.push_str("\n");
                        } else {
                            target_data_str.push_str(line.trim());
                            target_data_str.push_str("\n");
                        }
                    }
                    let metadata_str = metadata_str.trim();
                    let target_data_str = target_data_str.trim();

                    let task_question = if field_name.contains("status") {
                        parsing::extract_status_intent_prompt(&targeted_pug, &page_type, &bias_target)
                    } else {
                        parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                    };
                    
                    let mut ignore_list: Vec<String> = global_ignore_list.clone();
                    let mut miss_counter = 0;
                    
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }

                        let q3_gen = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let sys_msg = system_message.clone();
                        
                        let field_name_clone = field_name.clone();
                        let bias_target_for_closure = bias_target.clone(); 
                        let prejudice_target_for_closure = prejudice_target.clone(); 
                        
                        let task_q = task_question.clone();
                        let ignore_list_clone = ignore_list.clone();
                        
                        let res = tokio::task::spawn_blocking(move || {
                            let mut gen_guard = q3_gen.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                let params = ChatCompletionParameters {
                                    messages: vec![
                                        sys_msg,
                                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                            content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                            name: None,
                                        })
                                    ],
                                    model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                    ..Default::default()
                                };
                                
                                let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };
                                
                                gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                            } else {
                                Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                            }
                        }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                        // 환각 캐시를 즉시 제거하여 다음 재시도 시 방어
                        let q3_clear_arc = model.qwen3_generator.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                                gen.clear_kv_cache();
                            }
                        }).await;

                        if !model.is_cpu_mode {
                            let dev = model.device_config.device.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                if dev.is_cuda() { let _ = dev.synchronize(); }
                            }).await;
                        }

                        match res {
                            Ok(res_text) => {
                                let mut parsed = parsing::parse_json_from_llm(&res_text);
                                let parsed_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed }; // 🌟 mut 제거

                                let mut requires_retry = false;
                                let mut extracted_values_for_retry = Vec::new();
                                
                                let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                                let mut found_valid_value = false;

                                let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                                let is_enum_field = skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        let is_empty_val = match val {
                                            serde_json::Value::Null => true,
                                            serde_json::Value::String(s) => s.trim().is_empty() || s == "..." || s == "null",
                                            serde_json::Value::Array(a) => a.is_empty(),
                                            serde_json::Value::Object(o) => o.is_empty(),
                                            _ => false,
                                        };

                                        if !is_empty_val {
                                            found_valid_value = true;

                                            let extracted_str = if val.is_string() {
                                                val.as_str().unwrap_or("").trim().to_string()
                                            } else if val.is_number() {
                                                val.to_string()
                                            } else {
                                                String::new()
                                            };

                                            if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                
                                                if !is_enum_field {
                                                    let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                    let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                    let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                    
                                                    if !is_iso_date && !is_url && !is_boolean_str {
                                                        let mut is_matched = doc_title.contains(&extracted_str);
                                                        
                                                        if !is_matched {
                                                            let extracted_lower = extracted_str.to_lowercase();
                                                            let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            
                                                            for ctx_val in &json_contexts {
                                                                if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                    let target_lower = target_val_str.to_lowercase();
                                                                    
                                                                    if target_lower.contains(&extracted_lower) {
                                                                        if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                            let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                            if tokens.contains(&extracted_lower.as_str()) {
                                                                                is_matched = true;
                                                                                break;
                                                                            }
                                                                        } else {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    
                                                                    if !is_matched && digits_only.len() >= 3 {
                                                                        let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                        if target_digits.contains(&digits_only) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        if !is_matched {
                                                            requires_retry = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if !found_valid_value {
                                    requires_retry = true;
                                }

                                if requires_retry {
                                    miss_counter += 1;
                                    if miss_counter > 3 {
                                        emit_term(&format!("    ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                        break; 
                                    }
                                    emit_term(&format!("    ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                    for ex_str in extracted_values_for_retry {
                                        ignore_list.push(ex_str.clone());
                                        ignore_list.push(format!(" {}", ex_str));
                                        ignore_list.push(ex_str.to_lowercase());
                                    }
                                    if !found_valid_value {
                                        for k in &keys {
                                            ignore_list.push(format!("\"{}\": \"\"", k));
                                            ignore_list.push(format!("\"{}\":\"\"", k));
                                        }
                                    }
                                    continue;
                                }

                                let mut extracted_results = Vec::new();
                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        item_val.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                        extracted_results.push(format!("\"{}\": {}", k, val));
                                        
                                        let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                      else if val.is_number() { val.to_string() }
                                                      else { String::new() };
                                        
                                        if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                            if !global_ignore_list.contains(&val_str) {
                                                global_ignore_list.push(val_str.clone());
                                                global_ignore_list.push(format!(" {}", val_str));
                                                global_ignore_list.push(val_str.to_lowercase());
                                            }
                                        }
                                    }
                                }
                                
                                for ck in ["has_header", "has_footer", "title", "language"] {
                                    if let Some(val) = parsed_val.get(ck) {
                                        item_val.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                    }
                                }

                                if !extracted_results.is_empty() {
                                    emit_term(&format!("    ✅ Extracted: {}", extracted_results.join(", ")));
                                } else {
                                    emit_term(&format!("    ✅ Extracted: (null or empty for {})", field_name_clone));
                                }
                                break;
                            },
                            Err(e) => {
                                println!("[Scheduler] Error extracting list item field {}: {:?}", field_name_clone, e);
                                break;
                            }
                        }
                    } // loop end
                } // fields for loop end

                // 포스트 프로세싱: id, link 등 병합
                if let Some(id_val) = item_val.get("id").and_then(|v| v.as_str()) {
                    let extracted = if let Some(idx) = id_val.rfind('=') {
                        &id_val[idx + 1..]
                    } else {
                        id_val
                    };
                    
                    let clean_str = extracted.replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                    if !clean_str.is_empty() {
                        item_val.as_object_mut().unwrap().insert("id".to_string(), json!(clean_str.trim()));
                    }
                }

                if !item_val.is_null() && (item_val.is_object() || item_val.is_array()) {
                    if let Some(link_val) = item_val.get_mut("link") {
                        if let Some(relative_path) = link_val.as_str() {
                            if let Ok(base_url) = url::Url::parse(&url) {
                                if let Ok(absolute_url) = base_url.join(relative_path) {
                                    let path_query = format!("{}{}", absolute_url.path(), absolute_url.query().map(|q| format!("?{}", q)).unwrap_or_default());
                                    *link_val = json!(path_query.to_lowercase());
                                }
                            }
                        }
                    }
                    
                    emit_term(&format!("  ✅ Successfully Merged Extracted Item {}/{}: {}", idx + 1, total_items, serde_json::to_string(&item_val).unwrap_or_default()));
                    all_extracted_items.push(item_val);
                }
                
                // 🌟 [CRITICAL OPTIMIZATION] 동일 모델(Qwen3)을 연속 재사용하므로, 루프 내부에서 커널 워킹셋 메모리를 탈탈 비우고 대기하던 심각한 병목 가비지 코드를 원천 삭제하여 무지연 고속 순회 체계를 확립합니다.
                crate::models::qwen::generate::wait_for_global_io().await;
            } // items for loop end
        }

        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });

    } else {
        // [DETAIL MODE] Disk Bridge Relay
        println!("[Scheduler] Starting DISK BRIDGE RELAY for Details");
        
        let content_pug = {
            let clean_content = &clean_html_content;
            let full_pug = parsing::convert_to_clean_pug(clean_content, PugMode::DetailMode, Some(&url));
            model.truncate_pug_context(&full_pug, true, 2000, None).await
        };

        if !content_pug.trim().is_empty() {
            // 모델 스위칭 딜레이 없이, 가장 가벼운 모델인 Qwen3 하나만으로 필드별 순차 추출(Loop)을 진행합니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            // 🌟 [NOISE FILTER] bias.json의 prejudice 값을 가져와 노이즈 필터링용 벡터를 생성합니다.
            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            // (이전 답변에서 생성된 parsing.rs의 get_detail_schema_fields를 호출합니다)
            let fields = parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": format!("Extracting {} detail fields sequentially...", total_fields), "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload);
            emit_term(&format!("[STAGE-3] Extracting {} detailed fields individually...", total_fields));

            // 1. PUG 각 개별줄 내용을 임베딩 모델로 384차원 인메모리 벡터 저장 및 노이즈 필터링 적용
            let mut pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();
            // 🌟 [CRITICAL FIX] 인덱스 정렬(Alignment)을 위해 미리 크기를 할당합니다.
            let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
            
            // 🌟 배치 임베딩 적용
            let mut texts_to_embed = Vec::new();
            let mut text_indices = Vec::new();
            
            for (line_idx, line) in pug_lines.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                if line.trim().is_empty() { continue; }
                texts_to_embed.push(line.to_string());
                text_indices.push(line_idx);
            }
            
            if !texts_to_embed.is_empty() {
                for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                    let start_idx = chunk_idx * 100;
                    if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            let original_idx = text_indices[start_idx + i];
                            emit_term(&format!("  [VECTORIZING] Stage-3 Line {}/{} : {}", original_idx + 1, pug_lines.len(), text_chunk[i].trim()));
                            line_embeddings[original_idx] = vector;
                        }
                    }
                }
            }

            // 🎯 Track A: Stage 3 Detail (Boa Engine 기반 부모 뭉치 단위 노이즈 삭제)
            let (list_bias, form_bias, _) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            
            let mut wiped_indices = vec![false; pug_lines.len()];
            let mut processed_blocks = std::collections::HashSet::new();

            // Boa 엔진용 HTML 노드 JSON 생성 (Stage 3 컨텍스트 용)
            let nodes_str_detail = {
                let document_for_boa = scraper::Html::parse_document(&clean_html_content);
                let mut nodes_json = Vec::new();
                let mut node_to_idx = std::collections::HashMap::new();
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    node_to_idx.insert(node.id(), idx);
                }
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    if let Some(el) = node.value().as_element() {
                        let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                        let text: String = node.children()
                            .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                            .collect::<Vec<_>>().join(" ").trim().to_string();
                        nodes_json.push(serde_json::json!({
                            "index": idx,
                            "parentIndex": parent_idx,
                            "tagName": el.name().to_string(),
                            "id": el.id().unwrap_or("").to_string(),
                            "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                            "text": text,
                            "colspan": el.attr("colspan").unwrap_or("1"),
                            "rowspan": el.attr("rowspan").unwrap_or("1")
                        }));
                    } else {
                        nodes_json.push(serde_json::json!(serde_json::Value::Null));
                    }
                }
                serde_json::to_string(&nodes_json).unwrap_or_default()
            };
            
            let js_template_detail = get_boa_block_extractor_template(); // 🌟 Batch 처리용 템플릿 사용

            let mut track_a_candidates = Vec::new();
            let mut track_a_indices = Vec::new();
            let mut seen_detail_candidates = std::collections::HashSet::new(); // 🌟 JS 텍스트 중복 파싱 차단

            for line_idx in 0..pug_lines.len() {
                if wiped_indices[line_idx] { continue; }
                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&layout_prej_emb, &line_embeddings[line_idx]);
                
                if line_prej_score > 0.55 {
                    let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { line.trim() };
                    if !text_part.is_empty() && !seen_detail_candidates.contains(text_part) {
                        seen_detail_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                        track_a_indices.push(line_idx);
                    }
                }
            }

            // 🌟 Boa 엔진 1번으로 Stage 3 노이즈 후보군 일괄 처리
            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len(); // 🌟 [CRITICAL FIX] 클로저 이동 전 길이 미리 추출
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template_detail
                    .replace("NODES_PLACEHOLDER", &nodes_str_detail)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            // 🌟 [CRITICAL OPTIMIZATION] 스레드 8개를 가동하여 거대한 DOM 순회 작업을 병렬 해체, 속도를 800% 부스팅합니다!
            let stage3_pugs: Vec<String> = {
                let html_clone = clean_html_content.clone();
                let selectors = track_a_selectors.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_stage3_sels = std::collections::HashSet::new();
                    let mut unique_sels = Vec::new();
                    for sel in selectors {
                        if sel.is_empty() { continue; }
                        if !seen_stage3_sels.contains(&sel) {
                            seen_stage3_sels.insert(sel.clone());
                            unique_sels.push(sel);
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        local_res.push(crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::DetailMode, None));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };

            // 🌟 [핵심 최적화 3] Detail Noise 블록 일괄 Batch 병렬 타격
            let mut unique_stage3_pugs_to_embed = Vec::new();
            for block_pug in &stage3_pugs {
                if block_pug.is_empty() || processed_blocks.contains(block_pug) { continue; }
                processed_blocks.insert(block_pug.clone());
                unique_stage3_pugs_to_embed.push(block_pug.clone());
            }

            let mut stage3_embeddings_map = std::collections::HashMap::new();
            if !unique_stage3_pugs_to_embed.is_empty() {
                for chunk in unique_stage3_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            stage3_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for block_pug in stage3_pugs {
                if block_pug.is_empty() { continue; }
                let block_emb = stage3_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                
                let block_prej_score = cosine_similarity(&layout_prej_emb, &block_emb);
                let block_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                let block_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                
                if block_prej_score > block_list_score && block_prej_score > block_form_score {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [NOISE BLOCK DELETED] Boa Matched. Lines {}~{} (Prej: {:.4} > List: {:.4} & Form: {:.4})", start_idx + 1, end_idx + 1, block_prej_score, block_list_score, block_form_score));
                        for j in start_idx..=end_idx {
                            pug_lines[j] = String::new(); // 인덱스 보존을 위해 줄 내용만 삭제
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            for line_idx in 0..pug_lines.len() {
                if !wiped_indices[line_idx] && !pug_lines[line_idx].trim().is_empty() {
                    emit_term(&format!("  [FILTERED PUG] Line {} : {}", line_idx + 1, pug_lines[line_idx].trim()));
                }
            }
            
            let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();

            // 문서 타이틀 추출 (환각 검증용)
            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                }
            };

            let mut global_ignore_list: Vec<String> = Vec::new(); // 🌟 전역 무시 리스트 추가

            // 필드 단위로 하나씩 쪼개어 순차 추출 (병렬 처리 시의 VRAM 초과/컨텍스트 환각 방지)
            // 🌟 [CRITICAL FIX] prejudice_target 매개변수 추가
            for (idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.into_iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                // 2. Bias와 Prejudice 임베딩 (인메모리 코사인 검색용)
                let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);
                let prej_emb = if prejudice_target.trim().is_empty() { 
                    vec![0.0; 384] 
                } else { 
                    model.get_embedding(prejudice_target.clone()).await.unwrap_or(vec![0.0; 384]) 
                };
                
                let mut best_idx = 0;
                let mut best_score = -1.0;
                
                for (i, emb) in line_embeddings.iter().enumerate() {
                    if pug_lines_ref[i].trim().is_empty() { continue; }
                    let b_score = cosine_similarity(&bias_emb, emb);
                    let p_score = if prejudice_target.trim().is_empty() { 0.0 } else { cosine_similarity(&prej_emb, emb) };
                    let final_score = b_score - p_score;

                    if final_score > best_score {
                        best_score = final_score;
                        best_idx = i;
                    }
                }
                
                // 3. 찾은 컨텍스트 블록으로 추론을 위한 시스템 메시지 동적 조립 및 Fallback 처리
                let targeted_pug = if best_score < 0.25 {
                    emit_term(&format!("  ⚠️ [FALLBACK] Field: '{}' | Best Score ({:.4}) is too low. Using full context.", field_name, best_score));
                    content_pug.clone()
                } else {
                    extract_pug_context(&pug_lines_ref, best_idx)
                };

                let mut json_contexts = Vec::new();
                for line in targeted_pug.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if let Some(idx) = trimmed.find('|') {
                        let meta = trimmed[..idx].trim();
                        // 껍데기 HTML 태그명(td, th 등)이 있으면 제거하고 순수 메타데이터 텍스트만 추출
                        let clean_meta = meta.split('[').next().unwrap_or(meta).trim();
                        json_contexts.push(json!({
                            "metadata": clean_meta,
                            "value": trimmed[idx + 1..].trim()
                        }));
                    } else {
                        json_contexts.push(json!({
                            "value": trimmed
                        }));
                    }
                }
                let targeted_json_context = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();
                
                emit_term(&format!("  🎯 [MATCHED CONTEXT] Field: '{}' | Score: {:.4}\n{}", field_name, best_score, targeted_json_context));
                
                let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: format!("[JSON CONTEXT]\n{}", targeted_json_context),
                    name: None,
                });

                let percent = (((idx as f32) / (total_fields as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting {} ({}%)...", field_name, percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("Detail Extraction ({}/{})", idx + 1, total_fields), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));

                // 🌟 [CRITICAL FIX] 새롭게 추가된 title 파라미터 규격에 맞춰 &doc_title을 주입합니다.
                let mut metadata_str = String::new();
                let mut target_data_str = String::new();
                for line in targeted_pug.lines() {
                    if let Some(idx) = line.find('|') {
                        metadata_str.push_str(line[..idx].trim());
                        metadata_str.push_str("\n");
                        target_data_str.push_str(line[idx + 1..].trim());
                        target_data_str.push_str("\n");
                    } else {
                        target_data_str.push_str(line.trim());
                        target_data_str.push_str("\n");
                    }
                }
                let metadata_str = metadata_str.trim();
                let target_data_str = target_data_str.trim();

                let task_question = if field_name.contains("status") {
                    parsing::extract_status_intent_prompt(&targeted_pug, &page_type, &bias_target)
                } else {
                    parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                };
                
                // 🌟 [BIAS SKIP LOGIC] 본문에 존재하지 않는 잘못된 추출값 기록용 리스트 및 카운터
                let mut ignore_list: Vec<String> = global_ignore_list.clone(); // 🌟 매 필드마다 전역 리스트를 복사하여 누적 시작
                let mut miss_counter = 0;
                
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }

                    let q3_gen = model.qwen3_generator.clone();
                    let cancel_clone = cancellation_token.clone();
                    let sys_msg = system_message.clone();
                    
                    let field_name_clone = field_name.clone();
                    let bias_target_for_closure = bias_target.clone(); // 🌟 다국어 Bias 할당
                    
                    // 🌟 [CRITICAL FIX] 다국어 Prejudice(배제) 타겟 클로저용 변수 생성
                    let prejudice_target_for_closure = prejudice_target.clone(); 
                    
                    let task_q = task_question.clone();
                    let ignore_list_clone = ignore_list.clone();
                    
                    let res = tokio::task::spawn_blocking(move || {
                        let mut gen_guard = q3_gen.blocking_lock();
                        if let Some(gen) = gen_guard.as_mut() {
                            let params = ChatCompletionParameters {
                                messages: vec![
                                    sys_msg,
                                    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                        content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                        name: None,
                                    })
                                ],
                                model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                ..Default::default()
                            };
                            
                            // 🌟 [CRITICAL FIX] Prejudice(오답 밀어내기) 파라미터를 Qwen3 모델에 전달합니다!
                            let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };
                            // let b_target = if bias_target_for_closure.is_empty() { None } else { Some(bias_target_for_closure.as_str()) };
                            
                            gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                        } else {
                            Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                        }
                    }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                    // 🌟 [개선 2] 한 번의 생성이 끝날 때마다 (성공/실패 무관하게) 즉시 KV 캐시를 초기화하여, 
                    // 다음 재시도 시 모델이 과거의 환각 데이터를 바탕으로 헛소리를 이어가는 것을 원천 차단합니다!
                    let q3_clear_arc = model.qwen3_generator.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                            gen.clear_kv_cache();
                        }
                    }).await;

                    if !model.is_cpu_mode {
                        let dev = model.device_config.device.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            if dev.is_cuda() { let _ = dev.synchronize(); }
                        }).await;
                    }

                    match res {
                        Ok(res_text) => {
                            let mut parsed = parsing::parse_json_from_llm(&res_text);
                            
                            // type 래퍼 제거 로직
                            let item_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed }; // 🌟 mut 제거

                            // 🌟 [BIAS SKIP LOGIC] 추출된 값이 실제로 PUG 본문이나 제목에 존재하는지 검증
                            let mut requires_retry = false;
                            let mut extracted_values_for_retry = Vec::new();
                            
                            let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                            let mut found_valid_value = false;

                            // 🎯 [핵심 개선] ENUM 속성이나 영단어로 번역되어 출력될 수 있는 필드들은 본문 텍스트 완벽 매칭 검사에서 제외합니다.
                            let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                            let is_enum_field = skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    let is_empty_val = match val {
                                        serde_json::Value::Null => true,
                                        serde_json::Value::String(s) => s.trim().is_empty() || s == "..." || s == "null",
                                        serde_json::Value::Array(a) => a.is_empty(),
                                        serde_json::Value::Object(o) => o.is_empty(),
                                        _ => false,
                                    };

                                    if !is_empty_val {
                                        found_valid_value = true;

                                        let extracted_str = if val.is_string() {
                                            val.as_str().unwrap_or("").trim().to_string()
                                        } else if val.is_number() {
                                            val.to_string() // 숫자형 데이터도 검증을 위해 문자열로 변환
                                        } else {
                                            String::new()
                                        };

                                        if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            
                                            if !is_enum_field {
                                                let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                
                                                if !is_iso_date && !is_url && !is_boolean_str {
                                                    let mut is_matched = doc_title.contains(&extracted_str);
                                                    
                                                    if !is_matched {
                                                        let extracted_lower = extracted_str.to_lowercase();
                                                        let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                        
                                                        for ctx_val in &json_contexts {
                                                            if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                let target_lower = target_val_str.to_lowercase();
                                                                
                                                                if target_lower.contains(&extracted_lower) {
                                                                    if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                        let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                        if tokens.contains(&extracted_lower.as_str()) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    } else {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
                                                                
                                                                if !is_matched && digits_only.len() >= 3 {
                                                                    let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                    if target_digits.contains(&digits_only) {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // 010-0356789 같은 환각 전화번호는 2차 검증(숫자 배열)에서도 걸러져서 확실히 재시도됩니다!
                                                    if !is_matched {
                                                        requires_retry = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // 🌟 [CRITICAL FIX] LLM이 요구한 키를 뱉지 않았거나, 빈 값("")을 뱉었을 경우에도 재시도 (최대 3회)
                            if !found_valid_value {
                                requires_retry = true;
                            }

                            if requires_retry {
                                miss_counter += 1;
                                if miss_counter > 3 {
                                    emit_term(&format!("  ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                    break; 
                                }
                                emit_term(&format!("  ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                for ex_str in extracted_values_for_retry {
                                    ignore_list.push(ex_str.clone());
                                    ignore_list.push(format!(" {}", ex_str));
                                    ignore_list.push(ex_str.to_lowercase());
                                }
                                // 🎯 빈 값을 뱉는 것을 억제하기 위해 JSON 빈 문자열 패턴을 ignore_list에 추가합니다.
                                if !found_valid_value {
                                    for k in &keys {
                                        ignore_list.push(format!("\"{}\": \"\"", k));
                                        ignore_list.push(format!("\"{}\":\"\"", k));
                                    }
                                }
                                continue;
                            }

                            // 검증 통과 후 저장 및 결과 로그 출력
                            let mut extracted_results = Vec::new();
                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    extracted_data.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                    extracted_results.push(format!("\"{}\": {}", k, val));
                                    
                                    // 🌟 [전역 무시 리스트 업데이트] 성공적으로 찾은 값을 다른 필드에서 중복 추출하지 않도록 방어합니다.
                                    let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                  else if val.is_number() { val.to_string() }
                                                  else { String::new() };
                                    
                                    // 이름이나 짧은 단어가 실수로 억제되는 것을 방지하기 위해 길이가 5 이상인 고유값만 추가합니다.
                                    if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                        if !global_ignore_list.contains(&val_str) {
                                            global_ignore_list.push(val_str.clone());
                                            global_ignore_list.push(format!(" {}", val_str));
                                            global_ignore_list.push(val_str.to_lowercase());
                                        }
                                    }
                                }
                            }
                            
                            // 공통 속성(has_header, title 등)도 데이터에 병합하되, 로그에서는 생략하여 깔끔하게 유지합니다.
                            for ck in ["has_header", "has_footer", "title", "language"] {
                                if let Some(val) = item_val.get(ck) {
                                    extracted_data.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                }
                            }

                            if !extracted_results.is_empty() {
                                emit_term(&format!("  ✅ Extracted: {}", extracted_results.join(", ")));
                            } else {
                                emit_term(&format!("  ✅ Extracted: (null or empty for {})", field_name_clone));
                            }
                            break; // 정상 추출 시 무한루프 탈출
                        },
                        Err(e) => {
                            println!("[Scheduler] Error extracting detail field {}: {:?}", field_name_clone, e);
                            break;
                        }
                    }
                }
            }
        }
    }

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // --- DB OPS & SIDE EFFECTS ---
    
    let search_mode_str = search_mode.clone();
    let doc_lang_str = doc_lang.clone(); // 🌟 다국어 Currency 매핑을 위한 변수 복제
    let normalize_data = |item: &mut serde_json::Value| {
        if let Some(obj) = item.as_object_mut() {
            if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type.clone())); }
            
            if obj.get("mode").is_none() { obj.insert("mode".to_string(), json!(search_mode_str.clone())); }
            
            // 🌟 [CRITICAL FIX] 통화 대문자 변환 및 국가 언어 기반 기본 통화 자동 매핑 (Fallback)
            let currency_val = obj.get("currency").and_then(|v| v.as_str()).unwrap_or("").trim();
            if currency_val.is_empty() || currency_val == "null" {
                let default_currency = match doc_lang_str.as_str() {
                    "ko" => "KRW",
                    "ja" => "JPY",
                    "zh" | "zh-tw" | "zh-hk" => "CNY",
                    "en" => "USD",
                    "de" | "fr" | "it" | "es" | "nl" => "EUR",
                    _ => "USD",
                };
                obj.insert("currency".to_string(), json!(default_currency));
            } else {
                obj.insert("currency".to_string(), json!(currency_val.to_uppercase()));
            }
            
            // 수량 정수형 캐스팅
            if let Some(q) = obj.get("quantity").cloned() {
                let q_val = if q.is_number() { q.as_i64().unwrap_or(0) }
                            else if let Some(s) = q.as_str() { s.parse::<i64>().unwrap_or(0) }
                            else { 0 };
                obj.insert("quantity".to_string(), json!(q_val));
            }
            
            
            let date_keys = [
                "registration_date", "order_date", "payment_date", "shipping_date", 
                "manufacture_date", "expiration_date", "release_date", "started_at", "expired_at"
            ];
            if let Ok(re_date) = regex::Regex::new(r"\d+") {
                for key in date_keys.iter() {
                    if let Some(date_val) = obj.get(*key).and_then(|v| v.as_str()) {
                        let s = date_val.trim();
                        if !s.is_empty() && s != "null" {
                            // 1. Unix Timestamp 감지 (순수 숫자 10자리 혹은 13자리)
                            if s.chars().all(char::is_numeric) && (s.len() == 10 || s.len() == 13) {
                                if let Ok(ts) = s.parse::<i64>() {
                                    let ts_ms = if s.len() == 10 { ts * 1000 } else { ts };
                                    if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
                                        let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                                        obj.insert(key.to_string(), json!(iso_date));
                                        continue;
                                    }
                                }
                            }

                            // 2. 이미 완벽한 ISO 포맷인 경우 스킵 (T 포함, 글자수 충분)
                            if s.contains('T') && s.len() >= 19 {
                                continue;
                            }

                            // 3. 다양한 형태의 문자열 분해 및 논리적 역추론 (MM/DD/YYYY, YY-MM-DD 등)
                            let nums: Vec<u32> = re_date.find_iter(s).filter_map(|m| m.as_str().parse().ok()).collect();
                            if nums.len() >= 3 {
                                let mut year = nums[0];
                                let mut month = nums[1];
                                let mut day = nums[2];

                                // MM/DD/YYYY 또는 DD/MM/YYYY 형태 대응 (마지막 숫자가 31을 초과하면 연도로 간주)
                                if day > 31 && year <= 31 {
                                    year = nums[2];
                                    day = nums[1]; // 월/일 판별은 모호하므로 순서 유지
                                    month = nums[0];
                                }

                                // 2자리 연도 보정 (예: 24 -> 2024, 99 -> 1999)
                                if year < 100 {
                                    year += if year > 50 { 1900 } else { 2000 };
                                }
                                
                                month = month.clamp(1, 12);
                                day = day.clamp(1, 31);
                                
                                let hour = if nums.len() > 3 { nums[3].clamp(0, 23) } else { 0 };
                                let minute = if nums.len() > 4 { nums[4].clamp(0, 59) } else { 0 };
                                let second = if nums.len() > 5 { nums[5].clamp(0, 59) } else { 0 };
                                
                                let iso_date = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hour, minute, second);
                                obj.insert(key.to_string(), json!(iso_date));
                            }
                        }
                    } else if let Some(date_num) = obj.get(*key).and_then(|v| v.as_i64()) {
                        // LLM이 문자열이 아닌 정수형(Unix Time)으로 뱉어냈을 경우 방어
                        let ts_ms = if date_num < 10_000_000_000 { date_num * 1000 } else { date_num };
                        if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
                            let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                            obj.insert(key.to_string(), json!(iso_date));
                        }
                    }
                }
            }

            // 날짜 기본값(Fallback) 매핑 (비어있는 경우도 확실히 체크)
            if obj.get("started_at").is_none() || obj.get("started_at").unwrap().is_null() || obj.get("started_at").unwrap().as_str() == Some("") {
                if let Some(m) = obj.get("manufacture_date").cloned() { obj.insert("started_at".to_string(), m); }
            }
            if obj.get("expired_at").is_none() || obj.get("expired_at").unwrap().is_null() || obj.get("expired_at").unwrap().as_str() == Some("") {
                if let Some(e) = obj.get("expiration_date").cloned() { obj.insert("expired_at".to_string(), e); }
            }
            
            // 상태(Condition) 텍스트의 정수형 플래그 매핑
            if let Some(cond) = obj.get("condition").and_then(|v| v.as_str()) {
                let cond_lower = cond.to_lowercase();
                if cond_lower.contains("used") { obj.insert("used".to_string(), json!(1)); }
                if cond_lower.contains("lease") { obj.insert("lease".to_string(), json!(2)); }
                if cond_lower.contains("rental") { obj.insert("rental".to_string(), json!(3)); }
                if cond_lower.contains("refurbish") { obj.insert("refurbish".to_string(), json!(4)); }
            }
        }
    };

    if is_detail {
        normalize_data(&mut extracted_data);
    } else {
        if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
            for item in items.iter_mut() {
                normalize_data(item);
            }
        }
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    
    {
        println!("[Scheduler] Generating natural language sentences for FTS/Vector matching and Privacy Masking...");
        
        // [PRIVACY] goods(상품) 타입은 개인정보가 없으므로 필터를 우회하여 속도를 최적화합니다.
        let should_mask = page_type != "goods";

        if is_detail {
            let original_lang_text = parsing::json_to_natural_language(&extracted_data);
            
            // [PRIVACY] AI를 통한 개인정보 마스킹 로직 주입 (조건부)
            let masked_lang_text = original_lang_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨

            if let Some(obj) = extracted_data.as_object_mut() {
                obj.insert("text".to_string(), json!(original_lang_text));
                obj.insert("masked_text".to_string(), json!(masked_lang_text));
            }
        } else {
            if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
                for item in items.iter_mut() {
                    let original_lang_text = parsing::json_to_natural_language(item);
                    
                    // [PRIVACY] AI를 통한 개인정보 마스킹 로직 주입 (조건부)
                    let masked_lang_text = original_lang_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨

                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("text".to_string(), json!(original_lang_text));
                        obj.insert("masked_text".to_string(), json!(masked_lang_text));
                    }
                }
            }
        }
    }

    // --- PHASE 3: HANDOVER (Unload Qwen -> Load Embedding) ---
    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");
        
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));
        
        // 1. Explicitly Unload to free VRAM for Embedding Model
        model.deep_purge_resources().await;
        
        // 🌟 [OS WIRE-TRIM] 인계 시점에 가비지 페이징을 완전히 회수하여 시스템 프리징 현상을 원천 차단합니다.
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

        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        
        // 2. Wait for VRAM to settle (Driver latency)
        wait_for_resources_settled(1200, 800, Some(cancellation_token), model.device_config.gpu_id as u32).await?;
    }

    // [PARITY] ID Generation
    let id_val_raw = extracted_data.get("id")
        .or_else(|| extracted_data.get("no"))
        .or_else(|| extracted_data.get("code"))
        .or_else(|| extracted_data.get("tracking_number"))
        .or_else(|| extracted_data.get("index"))
        .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
        .unwrap_or_default();
    
    
    let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&id_val_raw)
        .replace("-", "").replace("_", "").replace(".", "").replace(",", "");
    
    let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
    let generated_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));

    if let Some(obj) = extracted_data.as_object_mut() {
        obj.insert("index".to_string(), json!(index_val));
        obj.insert("id".to_string(), json!(generated_id.clone()));
        
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    log_task_progress(app_handle, &task.id, &json!({ "category": "Saving", "summary": "Syncing to database..." }));

    // Re-acquire Store for final ops
    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    if page_type == "order" {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            for good in goods_arr {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let g_no = good.get("id").or_else(|| good.get("no")).and_then(|v| v.as_str()).unwrap_or("");
                if !g_no.is_empty() {
                    let clean_g_no = crate::utils::hash::normalize_numeric_homoglyphs(g_no).replace("-", "").replace("_", "");
                    
                    
                    let tracking_number = extracted_data.get("tracking_number").and_then(|v| v.as_str()).unwrap_or("");
                    let clean_tracking_no = crate::utils::hash::normalize_numeric_homoglyphs(tracking_number).replace("-", "").replace("_", "");
                    let tracking_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_tracking_no)));
                    let goods_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("goods{}{}", team_id, clean_g_no)));
                    
                    let tracking_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, clean_tracking_no, clean_g_no));
                    let mut tracking_data = extracted_data.clone();
                    
                    if let Some(obj) = tracking_data.as_object_mut() {
                        obj.insert("type".to_string(), json!("tracking"));
                        obj.insert("no".to_string(), json!(clean_tracking_no));
                        obj.insert("index".to_string(), json!(tracking_index));
                        obj.insert("goods".to_string(), json!(goods_index));
                        obj.insert("order".to_string(), json!(index_val)); // 부모 오더 index 매핑
                    }
                    
                    
                    let tracking_text = parsing::json_to_natural_language(&tracking_data);
                    let masked_tracking_text = tracking_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                    let tracking_vector = model.get_embedding(tracking_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    
                    tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(tracking_text));
                    tracking_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_tracking_text));
                    
                    let _ = store.upsert_item(
                        "tracking", &tracking_id, "tracking", tracking_data.clone(), Some(tracking_vector.clone()),
                        Some(&task.from), Some(&team_id), Some(&task.cc),
                        Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                        Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                        None
                    ).await;

                    
                    let _ = store.upsert_item(
                        "items", &tracking_id, "tracking", tracking_data, Some(tracking_vector),
                        Some(&task.from), Some(&team_id), Some(&task.cc),
                        Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                        Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                        None
                    ).await;
                }
            }
        }
    }

    
    let target_table = match page_type.as_str() {
        "sales" | "goods" | "order" => "sales",
        "tracking" | "receiving" | "shipping" => "tracking",
        "event" | "coupon" => "event",
        "member" | "team" | "user" => "users",
        "talk" | "prompt" | "ai_search" => "talks",
        _ => "items",
    }.to_string();

    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
    let ref_val = task.r#ref.clone();

    let mut items_to_process = Vec::new();
    let mut stats_diff: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();

    if is_detail {
        
        // Phase 2.5에서 주입된 영문 FTS 키워드가 포함된 text 속성을 최우선으로 사용하여 벡터화합니다.
        let text_to_embed = extracted_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&extracted_data));
        let item_digest = crate::utils::hash::digest(&text_to_embed); 
        let mut target_id = generated_id.clone(); 
        
        let mut existing_vector = None;
        let mut is_new = true;
        let mut was_draft = false;

        
        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            is_new = false;
            was_draft = if existing_item.updated_at_ts == 0 {
                true
            } else if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                !json_val.get("detail").and_then(|v| v.as_bool()).unwrap_or(true)
            } else {
                false
            };
            
            if existing_item.digest == item_digest {
                existing_vector = Some(existing_item.vector);
            }

            
            if let Ok(existing_json) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                extracted_data = merge_node(&existing_json, &extracted_data);
            }
        } 
        
        else if !url.is_empty() {
            let normalized_link = if let Ok(parsed_url) = url::Url::parse(&url) {
                format!("{}{}", parsed_url.path(), parsed_url.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase()
            } else {
                url.clone()
            };
            if let Ok(Some((found_id, json_val))) = store.find_item_by_property(&target_table, "link", &json!(normalized_link)).await {
                target_id = found_id.clone();
                is_new = false;
                
                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
                    was_draft = if existing_item.updated_at_ts == 0 {
                        true
                    } else if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                        !json_val.get("detail").and_then(|v| v.as_bool()).unwrap_or(true)
                    } else {
                        false
                    };
                    
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }
                
                
                extracted_data = merge_node(&json_val, &extracted_data);
                if let Some(obj) = extracted_data.as_object_mut() {
                    obj.insert("id".to_string(), json!(target_id.clone()));
                }
            }
        }

        
        // 새 항목이면 pages: count++, global: count++
        // 기존 Draft 항목이었다면 pages: draft--, count++ 승급 (global은 이미 리스트에서 올렸으므로 변동 없음)
        if is_new {
            let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
            e.1 += 1; // pages count++
            e.2 += 1; // global count++
        } else if was_draft {
            let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
            e.0 -= 1; // pages draft--
            e.1 += 1; // pages count++
        }

        
        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        
        let related_types = crate::logic::related(&page_type);
        for foreign_type in related_types {
            if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &extracted_data) {
                for q in queries {
                    match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                        Ok(Some((foreign_id, mut foreign_data))) => {
                            let was_foreign_draft = foreign_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                            let mut needs_update = false;

                            // 2. Update 속성 병합 (Import/Export)
                            if let Some(update) = &merge_rule.update {
                                for field in &update.includes {
                                    if update.from == page_type {
                                        if let Some(val) = extracted_data.get(field).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if update.to == page_type {
                                        if let Some(val) = foreign_data.get(field).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(field.clone(), val);
                                        }
                                    }
                                }
                                if let Some(foreign_info) = &update.foreign {
                                    if update.from == page_type {
                                        if let Some(val) = extracted_data.get(&foreign_info.to).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if update.to == page_type {
                                        if let Some(val) = foreign_data.get(&foreign_info.to).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                        }
                                    }
                                }
                            }

                            // 3. Upsert 속성 병합
                            if let Some(upsert) = &merge_rule.upsert {
                                for field in &upsert.includes {
                                    if upsert.from == page_type {
                                        if let Some(val) = extracted_data.get(field).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if upsert.to == page_type {
                                        if let Some(val) = foreign_data.get(field).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(field.clone(), val);
                                        }
                                    }
                                }
                            }

                            // 4. 연관 문서에 변경 사항이 있다면 벡터 재생성 후 DB 재저장
                            if needs_update {
                                if was_foreign_draft && merge_rule.update.as_ref().map_or(false, |u| u.to == foreign_type) {
                                    let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                    e.0 -= 1; // pages draft--
                                    e.1 += 1; // pages count++
                                    
                                    e.2 += 1; // global count++
                                    foreign_data.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                                let merged_text = parsing::json_to_natural_language(&foreign_data);
                                let masked_merged_text = merged_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                
                                foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                foreign_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));

                                let _ = store.upsert_item(
                                    &q.table, &foreign_id, foreign_type, foreign_data.clone(), Some(merged_vector.clone()),
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                                
                                let _ = store.upsert_item(
                                    "items", &foreign_id, foreign_type, foreign_data, Some(merged_vector),
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                            }
                        },
                        Ok(None) => {
                            
                            let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                            e.0 += 1; // pages draft++
                            e.2 += 1; // global count++

                            let mut draft_data = json!({});
                            
                            
                            let val_str = match &q.value {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Number(n) => n.to_string(),
                                _ => q.value.to_string(),
                            };
                            let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, foreign_type, val_str));
                            
                            if let Some(obj) = draft_data.as_object_mut() {
                                obj.insert("id".to_string(), json!(draft_id.clone()));
                                obj.insert("type".to_string(), json!(foreign_type));
                                obj.insert(q.column.clone(), q.value.clone());
                                obj.insert("updated_at".to_string(), json!(0)); // Draft 플래그
                            }

                            let _ = store.upsert_item(
                                &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                            ).await;

                            let _ = store.upsert_item(
                                "items", &draft_id, foreign_type, draft_data, None,
                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                            ).await;
                        },
                        _ => {}
                    }
                }
            }
        }

        
        // 여기서 다시 덮어씌우는 과정을 생략하여 보호합니다.

        let _ = store.upsert_item(
            &target_table, &target_id, &page_type, extracted_data.clone(), vector.clone(),
            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
        ).await;
        
        let _ = store.upsert_item(
            "items", &target_id, &page_type, extracted_data.clone(), vector,
            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
        ).await;

        items_to_process.push(extracted_data.clone());
        
    } else {
        
        if let Some(items) = extracted_data.get("items").and_then(|v| v.as_array()) {
            for item_val in items.iter() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let mut single_item = item_val.clone();
                
                
                let original_id = single_item.get("id")
                    .or_else(|| single_item.get("no"))
                    .or_else(|| single_item.get("code"))
                    .or_else(|| single_item.get("tracking_number"))
                    .or_else(|| single_item.get("index"))
                    .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
                    .unwrap_or_else(|| single_item.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string());
                
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&original_id)
                    .replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
                let hashed_item_id = if original_id.is_empty() {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, uuid::Uuid::new_v4()))
                } else {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val))
                };

                if let Some(obj) = single_item.as_object_mut() {
                    obj.insert("type".to_string(), json!(page_type));
                    obj.insert("detail".to_string(), json!(false));
                    obj.insert("id".to_string(), json!(hashed_item_id.clone()));
                    obj.insert("index".to_string(), json!(index_val));
                    
                    obj.insert("updated_at".to_string(), json!(0));
                }

                // Phase 2.5에서 주입된 영문 FTS 키워드가 포함된 text 속성을 최우선으로 사용하여 벡터화합니다.
                let text_to_embed = single_item.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&single_item));
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                
                let mut existing_vector = None;
                let mut is_new = true;
                // let is_fully_processed = false;

                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &hashed_item_id).await {
                    is_new = false;
                    // 이미 상세 페이지에서 처리되어 updated_at이 0보다 큰지 확인
                    // if existing_item.updated_at_ts > 0 {
                    //     is_fully_processed = true;
                    // }
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }

                
                // 새 항목이면 pages: draft++, global: count++
                if is_new {
                    let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
                    e.0 += 1; // pages draft++
                    e.2 += 1; // global count++
                }
                
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else {
                    Some(model.get_embedding(text_to_embed).await?)
                };

                
                let related_types = crate::logic::related(&page_type);
                for foreign_type in related_types {
                    if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &single_item) {
                        for q in queries {
                            match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                                Ok(Some((foreign_id, mut foreign_data))) => {
                                    let mut needs_update = false;

                                    // 2. Update 속성 병합
                                    if let Some(update) = &merge_rule.update {
                                        for field in &update.includes {
                                            if update.from == page_type {
                                                if let Some(val) = single_item.get(field).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if update.to == page_type {
                                                if let Some(val) = foreign_data.get(field).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(field.clone(), val);
                                                }
                                            }
                                        }
                                        if let Some(foreign_info) = &update.foreign {
                                            if update.from == page_type {
                                                if let Some(val) = single_item.get(&foreign_info.to).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if update.to == page_type {
                                                if let Some(val) = foreign_data.get(&foreign_info.to).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                                }
                                            }
                                        }
                                    }

                                    // 3. Upsert 속성 병합
                                    if let Some(upsert) = &merge_rule.upsert {
                                        for field in &upsert.includes {
                                            if upsert.from == page_type {
                                                if let Some(val) = single_item.get(field).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if upsert.to == page_type {
                                                if let Some(val) = foreign_data.get(field).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(field.clone(), val);
                                                }
                                            }
                                        }
                                    }

                                    // 4. 연관 문서에 변경 사항이 있다면 벡터 재생성 후 DB 재저장
                                    if needs_update {
                                        let merged_text = parsing::json_to_natural_language(&foreign_data);
                                        let masked_merged_text = merged_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                                        let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                        
                                        foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                        foreign_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));

                                        let _ = store.upsert_item(
                                            &q.table, &foreign_id, foreign_type, foreign_data.clone(), Some(merged_vector.clone()),
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                        
                                        let _ = store.upsert_item(
                                            "items", &foreign_id, foreign_type, foreign_data, Some(merged_vector),
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                    }
                                },
                                Ok(None) => {
                                    
                                    let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                    e.0 += 1; // pages draft++
                                    e.2 += 1; // global count++

                                    let mut draft_data = json!({});
                                    
                                    
                                    let val_str = match &q.value {
                                        serde_json::Value::String(s) => s.clone(),
                                        serde_json::Value::Number(n) => n.to_string(),
                                        _ => q.value.to_string(),
                                    };
                                    let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, foreign_type, val_str));
                                    
                                    if let Some(obj) = draft_data.as_object_mut() {
                                        obj.insert("id".to_string(), json!(draft_id.clone()));
                                        obj.insert("type".to_string(), json!(foreign_type));
                                        obj.insert(q.column.clone(), q.value.clone());
                                        obj.insert("updated_at".to_string(), json!(0)); // Draft 플래그
                                    }

                                    let _ = store.upsert_item(
                                        &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;

                                    let _ = store.upsert_item(
                                        "items", &draft_id, foreign_type, draft_data, None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                },
                                _ => {}
                            }
                        }
                    }
                }

                
                // 여기서 다시 덮어씌우는 과정을 생략하여 보호합니다.

                let _ = store.upsert_item(
                    &target_table, &hashed_item_id, &page_type, single_item.clone(), vector.clone(),
                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
                ).await;
                
                let _ = store.upsert_item(
                    "items", &hashed_item_id, &page_type, single_item.clone(), vector,
                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
                ).await;

                items_to_process.push(single_item);
            }
        }
    }

    if !items_to_process.is_empty() {
        let _ = update_team_base_metrics(&store, &team_id, &task.cc, &items_to_process, stats_diff.clone()).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items. (Stats Diff: {:?})", items_to_process.len(), stats_diff);
    }

    // Final Status Update
    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;

    
    // 대신 프론트엔드가 이 'Done' 신호를 받고 내부적으로 app.fetch()를 트리거하여
    // DB에서 완벽하게 세팅된(id, ref, bcc 등) 데이터를 가져가도록 유도해야 합니다.
    let payload = json!({
        "task_id": task.id, 
        "category": "Done", 
        "summary": "Extraction complete. Updating list...", 
        "spinner": "✅",
        // data를 null로 보내어 프론트엔드가 기존에 그리던 캐시를 초기화하도록 합니다.
        "data": null 
    });
    
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    println!("[PROCESS] Task {} completed. Handover to Embedding finished.", task.id);
    Ok(())
}

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    use tauri::Emitter;

    
    let mut final_payload = payload.clone();
    if let Some(obj) = final_payload.as_object_mut() {
        obj.insert("task_id".to_string(), serde_json::json!(task_id));
    }

    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        let line = format!("{}\n", final_payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }

    if let Some(cat) = final_payload.get("category").and_then(|v| v.as_str()) {
        if let Ok(mut w) = crate::CURRENT_UI_CATEGORY.write() {
            *w = cat.to_string();
        }
    }
    if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
        *w = Some(final_payload.clone());
    }

    let _ = app.emit("extraction-progress", &final_payload);
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>, target_gpu_id: u32) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let mut last_report = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB) on GPU {}...", target_vram_mb, target_gpu_id);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;

        if let Some(ref nvml_inst) = nvml {
            if let Ok(dev) = nvml_inst.device_by_index(target_gpu_id) {
                if let Ok(mem) = dev.memory_info() {
                    current_vram = mem.free;
                    has_gpu = true;
                }
            }
        }

        let meets_vram = !has_gpu || current_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            break; // Perfect state reached
        }

        // [STABILITY-LOGIC] Even if below target, if memory release has stopped changing,
        // it means we've recovered all we can. Don't wait forever.
        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 10_000_000 { // Change < 10MB (more lenient)
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
        }

        // [FAST-EXIT] If stable for 1.5 seconds OR we have at least 600MB free (enough for Embedding/0.6B)
        // This prevents being stuck at 0.7GB when target is 1.1GB.
        if (stable_ticks >= 3 && current_vram > 600_000_000) || current_vram > target_vram_bytes {
            println!("[RESOURCE-WATCH] Memory sufficient or stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 2 { // Faster reporting
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        // Absolute maximum wait 10s (reduced from 20s)
        if start_time.elapsed().as_secs() > 10 {
            println!("[RESOURCE-WATCH] Timeout or sufficient VRAM reached. Proceeding.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}

pub fn get_boa_js_template() -> &'static str {
    r##"
        const nodes = NODES_PLACEHOLDER;
        const titles = TITLES_PLACEHOLDER;
        
        function cleanClassList(classes, stripNumbers = false) {
            if (!classes) return [];
            const skip = ['active', 'selected', 'on', 'current', 'focus', 'hover', 'enabled', 'disabled'];
            return classes
                .filter(c => {
                    const lowerC = c.toLowerCase();
                    return !skip.includes(lowerC) && c.indexOf('__') === -1 && !/^[a-z0-9]{8,}$/.test(c);
                })
                .map(c => stripNumbers ? c.replace(/\d+$/, '') : c)
                .sort();
        }

        function getSignature(node, includeId = true) {
            if (!node || !node.tagName) return "";
            let s = node.tagName;
            if (includeId && node.id) s += "#" + node.id;
            const cls = cleanClassList(node.classes);
            if (cls.length > 0) {
                s += "." + [...new Set(cls)].join(".");
            }
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        function calculateSimilarity(nodeA, nodeB) {
            if (nodeA.tagName !== nodeB.tagName) return 0;
            
            
            // (예: 일반 데이터 행 vs colspan=10 인 안내/합계 행)
            if (nodeA.tagName === 'tr') {
                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                
                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                
                // 두 행의 가로 칸 수(colspan 총합)가 2칸 이상 차이난다면 구조가 아예 다른 것입니다.
                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) {
                    return 0;
                }
            }
            
            
            if (nodeA.tagName === 'td' || nodeA.tagName === 'th') {
                if (nodeA.colspan !== nodeB.colspan || nodeA.rowspan !== nodeB.rowspan) return 0;
            }

            const clsA = cleanClassList(nodeA.classes, true);
            const clsB = cleanClassList(nodeB.classes, true);
            if (clsA.length === 0 && clsB.length === 0) return 100;
            
            let matchCount = 0;
            clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
            return clsA.length ? (matchCount / clsA.length) * 100 : 0;
        }

        function detect(tIdx) {
            let cur = tIdx;
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) break;
                
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;
                
                if (node.tagName === "td" || node.tagName === "th") {
                    
                    // 이는 단일 항목이 아니라 복잡한 그리드의 부속품입니다. 묻지도 따지지도 않고 부모(tr)로 올라갑니다.
                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                        cur = pIdx;
                        continue;
                    }
                    
                    const pNode = nodes[pIdx];
                    if (pNode && pNode.tagName === "tr") {
                        const gpIdx = pNode.parentIndex; 
                        if (gpIdx !== undefined && gpIdx !== -1) {
                            const trSiblings = getChildren(gpIdx);
                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                            
                            // 부모(tr)가 유사한 구조의 다른 형제(tr)들을 여럿 거느리고 있다면 진짜 세로 리스트입니다.
                            if (similarTrs.length >= 2) {
                                cur = pIdx;
                                continue;
                            }
                        }
                    }
                }

                const parentNode = nodes[pIdx];
                const siblings = getChildren(pIdx);
                
                const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);

                if (similarSiblings.length >= 2) {
                    let finalParent = parentNode;
                    let walkIdx = pIdx;
                    for(let j=0; j<5; j++) {
                        let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                        if (gIdx !== -1 && nodes[gIdx]) {
                            const grand = nodes[gIdx];
                            if (grand.id || ["table", "ul", "ol", "nav"].includes(grand.tagName)) {
                                finalParent = grand;
                                if (grand.id || grand.tagName === "table") break;
                            }
                            walkIdx = gIdx;
                        }
                    }

                    const parentSig = getSignature(finalParent, true);
                    const uniqueSigs = [];
                    similarSiblings.forEach(s => {
                        const sig = getSignature(s, false);
                        if (!uniqueSigs.includes(sig)) uniqueSigs.push(sig);
                    });

                    const fullSelector = uniqueSigs.map(sig => parentSig + " " + sig).join(", ");

                    return { 
                        parent: parentSig, 
                        itemSelector: fullSelector,
                        matchCount: similarSiblings.length
                    };
                }
                cur = pIdx;
            }
            return null;
        }

        let bestResult = { "parent": "body", "itemSelector": "div", "matchCount": 0 };
        let firstMatchD = null;
        
        for (let i = 0; i < titles.length; i++) {
            let t = titles[i].toLowerCase().replace(/\s+/g, ' ');
            let potentialMatches = [];
            
            // 깨진 문자(\uFFFD)가 포함되어 있는지 확인합니다.
            if (t.includes('\uFFFD')) {
                // 깨진 경우: 쪼개서 조각들로 유연하게 검색
                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                if (chunks.length === 0) continue;
                
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return chunks.every(chunk => nText.includes(chunk));
                });
            } else {
                // 온전한 경우: 전체 문자열을 하나의 컬렉션으로 취급하여 정확하게 포함 여부 검색
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return nText.includes(t);
                });
            }
            
            if (potentialMatches.length > 0) {
                // 부모 노드(body, tr 등)를 배제하고, 텍스트 길이가 가장 짧은(가장 타이트한) 진짜 제목 단일 노드만 추출합니다.
                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                let d = detect(potentialMatches[0].index);
                if (d) {
                    if (!firstMatchD) firstMatchD = d;
                    // 형제 노드가 가장 많은(리스트에 가장 가까운) 결과를 최우선으로 저장
                    if (d.matchCount > bestResult.matchCount) {
                        bestResult = { "parent": d.parent, "itemSelector": d.itemSelector, "matchCount": d.matchCount };
                    }
                }
            }
        }
        
        // 반복 구조(matchCount > 0)를 하나도 못 찾았다면 첫 번째로 찾은 구조라도 사용
        if (bestResult.matchCount === 0 && firstMatchD) {
            bestResult = { "parent": firstMatchD.parent, "itemSelector": firstMatchD.itemSelector, "matchCount": firstMatchD.matchCount };
        }
        
        JSON.stringify(bestResult);
    "##
}

// 🌟 [CRITICAL OPTIMIZATION] 여러 개의 텍스트 타겟을 한 번의 JS 엔진 구동으로 일괄(Batch) 역추적하여 반환하는 템플릿입니다.
pub fn get_boa_block_extractor_template() -> &'static str {
    r##"
        const nodes = NODES_PLACEHOLDER;
        const targetTitles = TARGET_TITLES_PLACEHOLDER;
        
        function cleanClassList(classes, stripNumbers = false) {
            if (!classes) return [];
            const skip = ['active', 'selected', 'on', 'current', 'focus', 'hover', 'enabled', 'disabled'];
            return classes
                .filter(c => {
                    const lowerC = c.toLowerCase();
                    return !skip.includes(lowerC) && c.indexOf('__') === -1 && !/^[a-z0-9]{8,}$/.test(c);
                })
                .map(c => stripNumbers ? c.replace(/\d+$/, '') : c)
                .sort();
        }

        function getSignature(node, includeId = true) {
            if (!node || !node.tagName) return "";
            let s = node.tagName;
            if (includeId && node.id) s += "#" + node.id;
            const cls = cleanClassList(node.classes);
            if (cls.length > 0) {
                s += "." + [...new Set(cls)].join(".");
            }
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        function calculateSimilarity(nodeA, nodeB) {
            if (nodeA.tagName !== nodeB.tagName) return 0;
            if (nodeA.tagName === 'tr') {
                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) { return 0; }
            }
            if (nodeA.tagName === 'td' || nodeA.tagName === 'th') {
                if (nodeA.colspan !== nodeB.colspan || nodeA.rowspan !== nodeB.rowspan) return 0;
            }
            const clsA = cleanClassList(nodeA.classes, true);
            const clsB = cleanClassList(nodeB.classes, true);
            if (clsA.length === 0 && clsB.length === 0) return 100;
            let matchCount = 0;
            clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
            return clsA.length ? (matchCount / clsA.length) * 100 : 0;
        }

        function detect(tIdx) {
            let cur = tIdx;
            let fallbackParent = null;
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) break;
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;

                const parentNode = nodes[pIdx];
                if (parentNode && !fallbackParent) {
                    // 🌟 단독으로 배치된 Form 이나 고유 ID 클래스가 있는 시맨틱 컨테이너를 백업 부모로 동적 추적합니다.
                    if (parentNode.id || (parentNode.classes && parentNode.classes.length > 0) || ["form", "table", "nav", "tbody", "fieldset"].includes(parentNode.tagName)) {
                        fallbackParent = parentNode;
                    }
                }
                
                if (node.tagName === "td" || node.tagName === "th") {
                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                        cur = pIdx; continue;
                    }
                    const pNode = nodes[pIdx];
                    if (pNode && pNode.tagName === "tr") {
                        const gpIdx = pNode.parentIndex; 
                        if (gpIdx !== undefined && gpIdx !== -1) {
                            const trSiblings = getChildren(gpIdx);
                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                            if (similarTrs.length >= 2) { cur = pIdx; continue; }
                        }
                    }
                }
                const siblings = getChildren(pIdx);
                const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);
                if (similarSiblings.length >= 2) {
                    let finalParent = parentNode;
                    let walkIdx = pIdx;
                    for(let j=0; j<5; j++) {
                        let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                        if (gIdx !== -1 && nodes[gIdx]) {
                            const grand = nodes[gIdx];
                            if (grand.id || ["table", "ul", "ol", "nav", "form"].includes(grand.tagName)) {
                                finalParent = grand;
                                if (grand.id || grand.tagName === "table" || grand.tagName === "form") break;
                            }
                            walkIdx = gIdx;
                        }
                    }
                    return getSignature(finalParent, true);
                }
                cur = pIdx;
            }
            // 🌟 형제 노드 반복 패턴이 없는 독자적인 레이아웃일 경우 추적 수집된 시맨틱 백업 부모 블록을 안전하게 반환합니다.
            if (fallbackParent) { return getSignature(fallbackParent, true); }
            return null;
        }

        let finalResults = [];
        for (let k = 0; k < targetTitles.length; k++) {
            let t = targetTitles[k].toLowerCase().replace(/\s+/g, ' ');
            let potentialMatches = [];
            if (t.includes('\uFFFD')) {
                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                if (chunks.length > 0) {
                    potentialMatches = nodes.filter(n => {
                        if (!n || !n.text) return false;
                        let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                        return chunks.every(chunk => nText.includes(chunk));
                    });
                }
            } else {
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return nText.includes(t);
                });
            }
            
            let parentSel = "";
            if (potentialMatches.length > 0) {
                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                const d = detect(potentialMatches[0].index);
                if (d && d !== "body") { parentSel = d; }
            }
            finalResults.push(parentSel);
        }
        JSON.stringify(finalResults);
    "##
}

async fn update_team_base_metrics(
    store: &crate::store::VectorStore,
    team_id: &str,
    task_cc: &str,
    items: &Vec<serde_json::Value>,
    stats_diff: std::collections::HashMap<String, (i64, i64, i64)>,
) -> anyhow::Result<()> {
    let (team_json_str, team_vector, t_from, t_to, t_cc, t_bcc, t_ref, t_digest) = match store.get_item_by_id("users", team_id).await {
        Ok(Some(doc)) => (doc.json_data, doc.vector, doc.from, doc.to, doc.cc, doc.bcc, doc.r#ref, doc.digest),
        _ => (
            json!({ "base": { "pages": {} } }).to_string(),
            vec![0.0; 384],
            "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string()
        )
    };

    
    let mut parsed_val: serde_json::Value = serde_json::from_str(&team_json_str).unwrap_or(json!({ "base": { "pages": {} } }));
    
    
    while let Some(inner_str) = parsed_val.get("json_data").and_then(|v| v.as_str()) {
        if let Ok(inner_obj) = serde_json::from_str(inner_str) {
            parsed_val = inner_obj;
        } else {
            break;
        }
    }
    let mut team_data = parsed_val;
    
    
    if let Some(obj) = team_data.as_object_mut() {
        obj.remove("json_data");
    }
    
    // --- [블록 1 & 2: 맵 순회로 모든 타입의 통계 업데이트] ---
    for (t_name, (pages_draft_diff, pages_count_diff, global_count_diff)) in stats_diff.iter() {
        // 페이지별 통계 업데이트
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
            let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
            let page_type_node = cc_node.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            let current_draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            let current_count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            page_type_node.insert("draft".to_string(), json!(0.max(current_draft + pages_draft_diff)));
            page_type_node.insert("count".to_string(), json!(0.max(current_count + pages_count_diff)));
        } 

        // 글로벌 전체 통계 업데이트 (aa.ts와 동일하게 draft는 건드리지 않고 count만 누적)
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
            
            let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // 글로벌 draft는 클라우드 로직 상 사용되지 않으므로 보존하거나 건드리지 않습니다.
            global_type_node.insert("count".to_string(), json!(0.max(global_count + global_count_diff)));
        }
    }

    // Min/Max 업데이트는 items 내의 데이터에 한해서 진행
    {
        let properties = [
            "price", "quantity", "width", "height", "length", "weight", "shipping_fee", 
            "shipping_duration", "sale_price", "supply_price", "low_stock_threshold", 
            "discount", "min_order_amount", "max_discount_amount", "usage_limit", 
            "usage_per", "started_at", "expired_at"
        ];

        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            for prop in properties.iter() {
                if let Some(val) = item.get(*prop) {
                    let num_val = if val.is_number() {
                        val.as_f64().unwrap_or(0.0)
                    } else if let Some(s) = val.as_str() {
                        s.parse::<f64>().unwrap_or(0.0)
                    } else {
                        continue;
                    };

                    if num_val == 0.0 && *prop != "started_at" && *prop != "expired_at" { continue; }

                    
                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 0.0, "max": 0.0 })).as_object_mut().unwrap();
                    
                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    
                    
                    if current_min == 0.0 || num_val <= current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if current_max == 0.0 || num_val >= current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    
    if let Some(base_json) = team_data.get("base") {
        println!("\n[DEBUG-METRICS] 최종 반영된 Base JSON 값:\n{}", serde_json::to_string_pretty(base_json).unwrap_or_default());
    }

    
    if let Some(obj) = team_data.as_object_mut() {
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    // 5. Save back to DB (digest 파라미터에 None을 전달하여 강제 쓰기를 유도합니다)
    let _ = store.upsert_item(
        "users", 
        team_id, 
        "team", 
        team_data, 
        Some(team_vector),
        Some(&t_from),
        Some(&t_to),
        Some(&t_cc),
        Some(&t_bcc),
        Some(&t_ref),
        None
    ).await;

    
    if let Ok(Some(saved_doc)) = store.get_item_by_id("users", team_id).await {
        println!("\n==================================================");
        println!("✅ [DB-VERIFY] DB에 통계(Team) 데이터가 100% 정상 저장되었습니다!");
        println!("- 타겟 ID: {}", saved_doc.id);
        println!("- 갱신된 Timestamp: {}", saved_doc.updated_at_ts);
        
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&saved_doc.json_data) {
            if let Some(base_stats) = parsed.get("base") {
                println!("- DB 내 실제 Base 통계:\n{}", serde_json::to_string_pretty(base_stats).unwrap_or_default());
            }
        }
        println!("==================================================\n");
    } else {
        println!("\n==================================================");
        println!("🚨 [DB-VERIFY] 치명적 오류: DB에 Team 데이터가 저장되지 않았습니다!");
        println!("==================================================\n");
    }

    Ok(())
}


// [IN-MEMORY VECTOR SEARCH] 코사인 유사도 계산 및 PUG 부모 컨텍스트 추출
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

fn extract_pug_context(lines: &[&str], target_idx: usize) -> String {
    if lines.is_empty() { return String::new(); }
    let mut parent_idx = target_idx;
    let target_indent = lines[target_idx].chars().take_while(|c| c.is_whitespace()).count();
    
    // 1. 위로 거슬러 올라가며 들여쓰기가 더 적은(부모) 노드를 찾습니다.
    for i in (0..target_idx).rev() {
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent < target_indent && !lines[i].trim().is_empty() {
            parent_idx = i;
            break;
        }
    }

    let parent_indent = lines[parent_idx].chars().take_while(|c| c.is_whitespace()).count();
    let mut context_lines = vec![lines[parent_idx]];
    
    // 2. 부모 노드의 하위(자식) 노드들을 모두 긁어옵니다.
    for i in (parent_idx + 1)..lines.len() {
        if lines[i].trim().is_empty() { continue; }
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        // 다시 부모와 같거나 밖으로 나가는 들여쓰기를 만나면 블록 종료
        if indent <= parent_indent {
            break;
        }
        context_lines.push(lines[i]);
    }
    
    context_lines.join("\n")
}