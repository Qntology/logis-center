use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart};
use anyhow::Result;
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use crate::utils::GpuDeviceExt;
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

fn build_json_protected_ids(tokenizer: &TokenizerModel) -> std::collections::HashSet<u32> {
    let mut protected = std::collections::HashSet::new();
    for (tok_str, &id) in tokenizer.tokenizer.get_vocab(true).iter() {
        let maybe_punct = tok_str.len() <= 16
            && (tok_str.starts_with("<0x")
                || tok_str
                    .bytes()
                    .any(|b| matches!(b, b'"' | b'{' | b'[' | b'}' | b']' | b',' | b':')));
        if !maybe_punct {
            continue;
        }
        if let Ok(piece) = tokenizer.token_decode(vec![id]) {
            let p = piece.trim();
            if matches!(p, "\"" | "{" | "[" | "}" | "]" | "," | ":") {
                protected.insert(id);
            }
        }
    }
    protected
}

fn build_json_starter_ids(tokenizer: &TokenizerModel) -> std::collections::HashSet<u32> {
    let mut starters = std::collections::HashSet::new();
    for (tok_str, &id) in tokenizer.tokenizer.get_vocab(true).iter() {
        if tok_str.contains('{')
            || tok_str.contains('[')
            || tok_str.starts_with("<0x")
            || tok_str.starts_with('\u{0120}')
        {
            if let Ok(piece) = tokenizer.token_decode(vec![id]) {
                let trimmed = piece.trim_start_matches(|c: char| c == ' ' || c == '\t');
                if (trimmed.starts_with('{') || trimmed.starts_with('[')) && !trimmed.contains("//") {
                    starters.insert(id);
                }
            }
        }
    }
    if let Ok(v) = tokenizer.text_encode_vec("{".to_string(), false) {
        if let Some(&id) = v.first() {
            starters.insert(id);
        }
    }
    starters
}

fn hash_str_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn hybrid_prefix_cache_enabled() -> bool {
    !matches!(
        std::env::var("LOGIS_HYBRID_PREFIX_CACHE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "off" | "false" | "no"
    )
}

fn backend_label_for(dev: &Device) -> &'static str {
    if dev.is_rocm() {
        "rocm"
    } else if dev.is_cuda() {
        "cuda"
    } else if dev.is_vulkan() {
        "vulkan"
    } else if dev.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

fn sync_special_tokens_from_metadata_and_tokenizer(
    tokenizer: &TokenizerModel,
    eos_token_id: &mut u32,
    qwen3_5: &mut Qwen3_5Model,
) -> Result<()> {
    let tok = &tokenizer.tokenizer;
    if let Some(id) = tok.token_to_id("<|im_end|>") {
        *eos_token_id = id;
    }
    if let Some(id) = tok.token_to_id("<|vision_start|>") {
        qwen3_5.vision_start_token_id = id;
    }
    if let Some(id) = tok.token_to_id("<|vision_end|>") {
        qwen3_5.vision_end_token_id = id;
    }
    if let Some(id) = tok.token_to_id("<|image_pad|>") {
        qwen3_5.image_token_id = id;
    }
    if let Some(id) = tok.token_to_id("<|video_pad|>") {
        qwen3_5.video_token_id = id;
    }
    let vocab_size = qwen3_5.embed_vocab()?;
    if vocab_size > 0 {
        crate::models::qwen3_5::model::validate_host_token_ids(
            &[
                *eos_token_id,
                qwen3_5.vision_start_token_id,
                qwen3_5.vision_end_token_id,
                qwen3_5.image_token_id,
                qwen3_5.video_token_id,
            ],
            vocab_size,
            "qwen3_5.special_tokens",
        )?;
    }
    Ok(())
}

fn pretokenize_ignore_list(
    tokenizer: &TokenizerModel,
    ignore_list: Option<&[String]>,
    is_strict_json: bool,
) -> Vec<Vec<u32>> {
    let Some(ignores) = ignore_list else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ign in ignores {
        if is_strict_json && (ign.trim().starts_with('{') || ign.trim().starts_with('[')) {
            continue;
        }
        let toks = tokenizer
            .text_encode_vec(ign.to_string(), false)
            .unwrap_or_default();
        if !toks.is_empty() {
            out.push(toks);
        }
    }
    out
}

fn persist_oneshot_kv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("LOGIS_PERSIST_ONESHOT_KV")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
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
    /// 🌟 [SPD-02] JSON 문법 보호 토큰 ID 캐시
    json_protected_ids: std::collections::HashSet<u32>,
    /// Valid opening JSON starter tokens (`{`, `{"`, `{\n`, `[`, `[{"`, ...) preserved at i == 0
    json_starter_ids: std::collections::HashSet<u32>,
    /// 2-tier system/context prefix snapshot cache (Host L_sys + VRAM L_ctx)
    pub snapshot_cache: crate::models::qwen3_5::model::HybridSnapshotCache,
    /// Phase 3-1 D2: 직전 생성의 가드/구출 통계 (`StepGuardStats`)
    last_guard_stats: Option<crate::models::common::vocab_trim::StepGuardStats>,
    /// 직전 생성 타이밍 `(prefill_ms, decode_ms, decode_steps)`
    last_gen_stats: Option<(f64, f64, usize)>,
}

impl Qwen3_5GenerateModel {
    pub fn last_guard_stats(&self) -> Option<&crate::models::common::vocab_trim::StepGuardStats> {
        self.last_guard_stats.as_ref()
    }

    pub fn last_gen_stats(&self) -> Option<(f64, f64, usize)> {
        self.last_gen_stats
    }

    pub fn resident_vram_bytes(&self) -> u64 {
        self.qwen3_5.resident_vram_bytes()
    }

    fn prejudice_cached_active(&self, target_text: &str) -> Result<crate::models::common::prejudice::CachedPrejudice> {
        let target_ids = self.tokenizer.text_encode_vec(target_text.to_string(), false)?;
        if target_ids.is_empty() {
            return Err(anyhow::anyhow!("empty target tokens"));
        }
        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
        let target_emb = self.qwen3_5.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
        let target_emb_sum = target_emb.sum_keepdim(1)?;
        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
        let target_vec = target_emb_sum.broadcast_div(&len_tensor)?.squeeze(0)?.squeeze(0)?;

        let vocab_label = self
            .qwen3_5
            .trim_head
            .as_ref()
            .map(|h| format!("{}:{:016x}", h.label, h.kept_hash))
            .unwrap_or_else(|| "full".into());
        let num_rows = self
            .qwen3_5
            .trim_head
            .as_ref()
            .map(|h| h.kept.len())
            .unwrap_or(self.qwen3_5.embed_vocab()?);
        let cache_key = format!("qwen35:k:{:016x}:{vocab_label}", hash_str_u64(target_text));
        crate::models::common::prejudice::get_or_build(
            &cache_key,
            &target_vec,
            num_rows,
            &self.device,
            |off, take| self.qwen3_5.embed_kept_rows_f32(off, take, &self.device),
        )
    }

    fn prejudice_full_tensor(&self, target_text: &str) -> Result<Tensor> {
        let target_ids = self.tokenizer.text_encode_vec(target_text.to_string(), false)?;
        if target_ids.is_empty() {
            return Err(anyhow::anyhow!("empty target tokens"));
        }
        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
        let target_emb = self.qwen3_5.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
        let target_emb_sum = target_emb.sum_keepdim(1)?;
        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
        let target_vec = target_emb_sum.broadcast_div(&len_tensor)?.squeeze(0)?.squeeze(0)?;

        let vocab = self.qwen3_5.embed_vocab()?;
        let vocab_label = self
            .qwen3_5
            .trim_head
            .as_ref()
            .map(|h| h.label.as_str())
            .unwrap_or("full");
        let cache_key = format!("qwen35:v{vocab}:{vocab_label}:{target_text}");
        let cached = crate::models::common::prejudice::get_or_build(
            &cache_key,
            &target_vec,
            vocab,
            &self.device,
            |off, take| Ok(self.qwen3_5.embed_rows_f32(off, take)?),
        )?;
        Ok(crate::models::common::vocab_trim::exclude_dropped(
            self.qwen3_5.trim_head.as_ref(),
            cached.values,
        )?)
    }
    /// 🌟 [VOCAB TRIM] 지금 흐름의 문서 언어(`vocab_trim::DocLangScope`)로 출력 투영의 남긴 행을 맞춥니다. 생성마다 처음에 부릅니다.
    pub fn refresh_vocab(&mut self) -> Result<()> {
        self.refresh_vocab_for("", false)
    }

    /// [`Self::refresh_vocab`] + 이번 생성의 프롬프트 글에 나온 문자 체계. 샘플링(온도 > 0)이면 전체 행 (`vocab_trim::refresh_for`)
    pub fn refresh_vocab_for(&mut self, prompt: &str, sampling: bool) -> Result<()> {
        if let Some(src) = self.qwen3_5.vocab_table() {
            crate::models::common::vocab_trim::refresh_for(&mut self.qwen3_5.trim_head, &src, &self.tokenizer.tokenizer, &self.device, prompt, sampling)?;
        }
        Ok(())
    }

    fn refresh_vocab_from_payload_ids(&mut self, payload_text: &str, payload_ids: &[u32], sampling: bool) -> Result<()> {
        if let Some(src) = self.qwen3_5.vocab_table() {
            crate::models::common::vocab_trim::refresh_for_ids(
                &mut self.qwen3_5.trim_head,
                &src,
                &self.tokenizer.tokenizer,
                &self.device,
                payload_text,
                payload_ids,
                Some(self.model_name.as_str()),
                sampling,
            )?;
        }
        Ok(())
    }

    /// Extract single-pass tokenized IDs with byte spans relative to `mes_text` and project
    /// data-payload sections (`[TARGET DATA]`, `[SOURCE DATA]`, etc.) without a second tokenize pass.
    fn encode_prompt_with_spans(&self, mes_text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let v_start = self.qwen3_5.vision_start_token_id;
        let v_end = self.qwen3_5.vision_end_token_id;
        let v_img = self.qwen3_5.image_token_id;
        let v_vid = self.qwen3_5.video_token_id;
        let mut ids_vec = Vec::new();
        let mut spans = Vec::new();
        let mut base = 0usize;

        fn push_special(
            tok: u32,
            tag: &str,
            cursor: &mut usize,
            ids_vec: &mut Vec<u32>,
            spans: &mut Vec<(usize, usize)>,
        ) {
            let s = *cursor;
            *cursor += tag.len();
            ids_vec.push(tok);
            spans.push((s, *cursor));
        }

        let mut i = 0;
        for p_start in mes_text.split("<|vision_start|>") {
            if i > 0 {
                push_special(v_start, "<|vision_start|>", &mut base, &mut ids_vec, &mut spans);
            }
            i += 1;
            let mut j = 0;
            for p_end in p_start.split("<|vision_end|>") {
                if j > 0 {
                    push_special(v_end, "<|vision_end|>", &mut base, &mut ids_vec, &mut spans);
                }
                j += 1;
                let mut k = 0;
                for p_img in p_end.split("<|image_pad|>") {
                    if k > 0 {
                        push_special(v_img, "<|image_pad|>", &mut base, &mut ids_vec, &mut spans);
                    }
                    k += 1;
                    let mut l = 0;
                    for p_vid in p_img.split("<|video_pad|>") {
                        if l > 0 {
                            push_special(v_vid, "<|video_pad|>", &mut base, &mut ids_vec, &mut spans);
                        }
                        l += 1;
                        if !p_vid.is_empty() {
                            let enc = self
                                .tokenizer
                                .tokenizer
                                .encode(p_vid, false)
                                .map_err(anyhow::Error::msg)?;
                            for (&tid, &(os, oe)) in enc.get_ids().iter().zip(enc.get_offsets().iter()) {
                                ids_vec.push(tid);
                                spans.push((base + os, base + oe));
                            }
                            base += p_vid.len();
                        }
                    }
                }
            }
        }
        Ok((ids_vec, spans))
    }

    fn collect_payload_text_and_ids(
        &self,
        mes_text: &str,
        user_text: &str,
        ids_vec: &[u32],
        spans: &[(usize, usize)],
    ) -> (String, Vec<u32>) {
        let ranges = crate::models::common::vocab_trim::extract_data_payload_ranges(mes_text);
        if ranges.is_empty() {
            return (user_text.to_string(), ids_vec.to_vec());
        }
        let mut payload_text = String::new();
        let mut payload_ids = Vec::new();
        for &(rs, re) in &ranges {
            let chunk = mes_text[rs..re].trim();
            if !chunk.is_empty() {
                if !payload_text.is_empty() {
                    payload_text.push('\n');
                }
                payload_text.push_str(chunk);
            }
            for (&tid, &(ts, te)) in ids_vec.iter().zip(spans.iter()) {
                if te > rs && ts < re {
                    payload_ids.push(tid);
                }
            }
        }
        if payload_text.is_empty() {
            (user_text.to_string(), ids_vec.to_vec())
        } else {
            (payload_text, payload_ids)
        }
    }

    fn find_safe_prefix_checkpoints(
        &self,
        mes_text: &str,
        spans: &[(usize, usize)],
    ) -> (Option<usize>, Option<usize>) {
        use crate::models::qwen3_5::model::MIN_PREFIX_SNAPSHOT_TOKENS;
        let total = spans.len();
        let sys_tok = mes_text
            .find("<|im_start|>user\n")
            .map(|b| spans.iter().take_while(|&&(_, te)| te <= b).count())
            .filter(|&c| c >= MIN_PREFIX_SNAPSHOT_TOKENS && c < total);

        let ranges = crate::models::common::vocab_trim::extract_data_payload_ranges(mes_text);
        let ctx_tok = ranges
            .last()
            .map(|&(_rs, re)| spans.iter().take_while(|&&(_, te)| te <= re).count())
            .filter(|&c| c >= MIN_PREFIX_SNAPSHOT_TOKENS && c < total && Some(c) != sys_tok);

        (sys_tok, ctx_tok)
    }

    /// 로드 시 설정된 max_position_embeddings 값을 반환합니다.
    pub fn get_max_tokens(&self) -> usize {
        self.max_position_embeddings
    }

    /// 🌟 [VISION-JIT] 이 인스턴스가 비전 입력을 처리할 자격(전처리기 + mmproj)을 갖췄는지 확인합니다.
    /// 가중치의 물리적 상주 여부와는 무관합니다.
    pub fn vision_capable(&self) -> bool {
        self.pre_processor.is_some()
    }

    /// 🌟 [VISION-JIT] mmproj 를 mmap 에서 다시 읽어올 수 있는 상태인지 확인합니다.
    pub fn is_vision_jit_capable(&self) -> bool {
        self.qwen3_5.is_vision_jit_capable()
    }

    /// 🌟 [VISION-JIT] 현재 비전 가중치가 실제로 메모리에 올라와 있는지 확인합니다.
    pub fn vision_resident(&self) -> bool {
        self.qwen3_5.is_vision_resident()
    }

    /// 🌟 [VISION-JIT] 2B 텍스트 모델 전체를 파기하지 않고 비전 가중치만 붙였다 뗍니다.
    pub fn set_vision_active(&mut self, active: bool) -> Result<()> {
        self.qwen3_5.set_vision_active(active)?;

        // 비전을 뗀 직후 핫패스 워킹셋 트림은 페이지 폴트 스래싱을 유발하므로
        // `legacy_working_set_trim()` 계약(기본 무동작)으로 일원화합니다.
        if !active {
            crate::models::common::legacy_working_set_trim();
        }
        Ok(())
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let load_t0 = std::time::Instant::now();
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let vram_before = crate::utils::device_utils::device_vram_snapshot(&device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let mut eos_token_id = cfg.text_config.eos_token_id;
        let max_pos = cfg.text_config.max_position_embeddings;
        let mut qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg)?;
        sync_special_tokens_from_metadata_and_tokenizer(&tokenizer, &mut eos_token_id, &mut qwen3_5)?;
        let vram_after = crate::utils::device_utils::device_vram_snapshot(&device);

        let json_protected_ids = build_json_protected_ids(&tokenizer);
        let json_starter_ids = build_json_starter_ids(&tokenizer);
        let gen = Self {
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
            json_protected_ids,
            json_starter_ids,
            snapshot_cache: crate::models::qwen3_5::model::HybridSnapshotCache::default(),
            last_guard_stats: None,
            last_gen_stats: None,
        };
        gen.record_load_reflection("safetensors", load_t0.elapsed().as_secs_f32() * 1e3, vram_before, vram_after);
        Ok(gen)
    }

    fn record_load_reflection(
        &self,
        recipe: &str,
        load_ms: f32,
        vram_before: Option<crate::utils::device_utils::DeviceVramSnapshot>,
        vram_after: Option<crate::utils::device_utils::DeviceVramSnapshot>,
    ) {
        if logis_schema::reflection::ReflectionLogMode::from_env() == logis_schema::reflection::ReflectionLogMode::Off {
            return;
        }
        let vocab_v = self.qwen3_5.embed_vocab().unwrap_or(0) as u32;
        let active_k = self
            .qwen3_5
            .trim_head
            .as_ref()
            .map(|h| h.kept.len() as u32)
            .unwrap_or(vocab_v);
        let (table_mode, resident_table_mb, mapped_table_mb) = match self.qwen3_5.vocab_table() {
            Some(src) => {
                let row_bytes = (src.cols / src.dtype.block_size().max(1)) * src.dtype.type_size();
                let total_bytes = (src.rows * row_bytes) as f32;
                let head_bytes = self.qwen3_5.trim_head.as_ref().map_or(0.0, |h| h.head_bytes() as f32);
                ("mapped".to_string(), head_bytes / 1e6, total_bytes / 1e6)
            }
            None => (
                "resident".to_string(),
                self.qwen3_5.resident_vram_bytes() as f32 / 1e6,
                0.0,
            ),
        };
        let rec = logis_schema::reflection::ReflectionRecord {
            ts_ms: logis_schema::reflection::current_ts_ms(),
            run_id: logis_schema::reflection::current_run_id(),
            task_id: self.model_name.clone(),
            git_rev: option_env!("LOGIS_GIT_REV").unwrap_or("dev").into(),
            build_profile: logis_schema::reflection::build_profile_label(),
            features: Vec::new(),
            backend: backend_label_for(&self.device).into(),
            device_name: format!("{:?}", self.device),
            artifact: logis_schema::reflection::ArtifactReflection {
                slot: self.model_name.clone(),
                bytes: self.qwen3_5.resident_vram_bytes(),
                sha256_16: String::new(),
                recipe: recipe.into(),
                vocab_v,
                artifact_k: active_k,
                plan_sha256: None,
                space_tag: None,
            },
            load: Some(logis_schema::reflection::LoadReflection {
                load_ms,
                vram_before_mb: vram_before.map_or(0.0, |s| s.resident_used_mb_f32()),
                vram_after_mb: vram_after.map_or(0.0, |s| s.resident_used_mb_f32()),
                table_mode,
                resident_table_mb,
                mapped_table_mb,
                env_overrides: Vec::new(),
                accel_fallbacks: crate::models::common::accel_check::recorded_fallbacks(&self.device),
                route_block_reason: None,
            }),
            exec: None,
        };
        logis_schema::reflection::record(&rec);
    }

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        let load_t0 = std::time::Instant::now();
        // Mmap으로 메모리에 매핑
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);
        
        let mut reader = std::io::Cursor::new(&mmap_arc[..]);
        let content = gguf_file::Content::read(&mut reader)?;
        // 🌟 [QWEN4-EXP] arch 'qwen4exp' GGUF 를 Qwen3.5 그래프로 잘못 읽지 않도록 먼저 막습니다.
        crate::models::qwen4_exp::registry::ensure_not_qwen4exp_gguf(&content)?;
        
        // Content Clone 우회를 위한 메타데이터 Arc 래핑
        let mut reader_clone = std::io::Cursor::new(&mmap_arc[..]);
        let content_clone = gguf_file::Content::read(&mut reader_clone)?;
        let ct_arc = std::sync::Arc::new(content_clone);
        
        let device = get_device(device);
        let vram_before = crate::utils::device_utils::device_vram_snapshot(&device);
        let mut model_gguf = Gguf::new(content, reader, device.clone());
        let _chat_template_str = model_gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        
        let model_dir = std::path::Path::new(model_file).parent().unwrap().to_str().unwrap();
        let chat_template = ChatTemplate::init(model_dir)?;

        
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let max_pos = if config_path.exists() {
            let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?).map(logis_schema::json_order::sorted)?;
            cfg.get("max_position_embeddings").or_else(|| cfg.get("text_config").and_then(|t| t.get("max_position_embeddings"))).and_then(|v| v.as_u64()).unwrap_or(262144) as usize
        } else {
            
            model_gguf.get_matedata("qwen35.context_length").ok().and_then(|m| m.to_u32().ok()).unwrap_or(262144) as usize
        };

        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        
        // 외부에서 명시적으로 비전 가중치 경로를 지정했을 때만 로드하도록 제한하여 텍스트 전용 모드 최적화
        let target_mmproj = mmproj_file.map(|f| f.to_string());
        // 🌟 [VISION-SKELETON] 전처리기만 만들고 mmproj Gguf 는 만들지 않습니다.
        //
        //  ── 구버전이 무엇을 했나 ──
        //   ① 여기서 mmproj Content 를 파싱해 Gguf 를 만들고
        //   ② new_from_gguf 가 그 Gguf 로 24블록 1.44GB 를 VRAM 에 적재하고
        //   ③ set_mmproj_path 가 같은 파일을 '또' 열어 Content 를 재파싱하고
        //   ④ set_block_streaming(true) 로 플래그만 켰습니다.
        //   ②가 이미 끝난 뒤라 ④는 첫 forward 전까지 아무 효과가 없었고,
        //   그래서 첫 이미지 전처리 시점 free VRAM 이 586MB 까지 떨어졌습니다.
        //
        //  ── 새 흐름 ──
        //   ① 전처리기만 생성 (VRAM 0)
        //   ② new_from_gguf 에 비전을 넘기지 않음 → 언어 모델만 구성
        //   ③ attach_vision_skeleton 이 메타데이터만 읽어 구조를 붙임 (VRAM 0, Content 1회 파싱)
        //   ④ 실제 가중치는 forward 가 캐시 MISS 를 확인한 뒤 블록 단위로 읽음
        let pre_processor = if target_mmproj.is_some() {
            Some(Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?)
        } else {
            None
        };
        let mut eos_token_id = model_gguf
            .get_matedata("tokenizer.ggml.eos_token_id")?
            .to_u32()?;

        // 밥줄(Mmap, Ct)을 쥐여준 채로 모델 생성 (비전 제외)
        let mut qwen3_5 = Qwen3_5Model::new_from_gguf(
            &mut model_gguf,
            None::<&mut Gguf<std::fs::File>>,
            &device,
            Some(mmap_arc.clone()),
            Some(ct_arc)
        )?;
        sync_special_tokens_from_metadata_and_tokenizer(&tokenizer, &mut eos_token_id, &mut qwen3_5)?;
        // 🌟 [WEIGHT-RESIDENT / SPD-05] mmproj 스켈레톤은 스트리밍되고, `clear_cache()` 가 KV 해제 후
        //    실제 VRAM 여유(`keep_warm_ok`)를 확인하므로 2B+mmproj 포함 모든 Qwen3.5 에서 가중치 웜 유지 허용.
        qwen3_5.language_model.keep_warm = true;

        // 🌟 [VISION-SKELETON] 가중치 0바이트로 비전 구조만 붙입니다.
        //    실패하면 순수 텍스트 모드로 안전하게 강등됩니다.
        if let Some(mmproj_f) = target_mmproj.as_deref() {
            match qwen3_5.attach_vision_skeleton(mmproj_f, &device) {
                Ok(_) => {
                    println!("[VISION-SKELETON] mmproj attached without loading weights. Init peak reduced by the full vision tower.");
                }
                Err(e) => {
                    println!("[VISION-SKELETON] Failed to attach mmproj skeleton ({}). Falling back to text-only.", e);
                }
            }
        }
        // 🌟 [VISION-STREAM] 스켈레톤은 이미 스트리밍 상태로 태어나지만,
        //    명시적으로 한 번 더 확정해 의도를 코드에 남깁니다.
        qwen3_5.set_block_streaming(true);
        let vram_after = crate::utils::device_utils::device_vram_snapshot(&device);

        let stem = std::path::Path::new(model_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
            
        
        crate::models::common::legacy_working_set_trim();

        let json_protected_ids = build_json_protected_ids(&tokenizer);
        let json_starter_ids = build_json_starter_ids(&tokenizer);

        // 구조체 반환 부분은 그대로 유지
        let gen = Self {
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
            json_protected_ids,
            json_starter_ids,
            snapshot_cache: crate::models::qwen3_5::model::HybridSnapshotCache::default(),
            last_guard_stats: None,
            last_gen_stats: None,
        };
        gen.record_load_reflection("gguf", load_t0.elapsed().as_secs_f32() * 1e3, vram_before, vram_after);
        Ok(gen)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_token_from_hidden(
        &mut self,
        hidden: &Tensor,
        step_idx: usize,
        is_strict_json: bool,
        temperature: f64,
        logit_processor: &mut candle_transformers::generation::LogitsProcessor,
        generate_hist: &[u32],
        penalized_history: &[u32],
        gen_text_buffer: &str,
        enter_id: u32,
        think_token_id: u32,
        open_bracket_id: u32,
        lt_id: u32,
        slash_id: u32,
        double_slash_id: u32,
        space_double_slash_id: u32,
        pretokenized_ignores: &[Vec<u32>],
        active_prej: Option<&crate::models::common::prejudice::CachedPrejudice>,
        full_prej: Option<&mut Tensor>,
        scratch: &mut crate::models::common::vocab_trim::DecodeStepScratch,
        guard_stats: &mut crate::models::common::vocab_trim::StepGuardStats,
    ) -> Result<u32> {
        scratch.banned_ids.clear();
        if is_strict_json {
            scratch.banned_ids.push(think_token_id);
            scratch.banned_ids.push(lt_id);
            let is_url_single =
                gen_text_buffer.ends_with("http:/") || gen_text_buffer.ends_with("https:/");
            let is_url_double =
                gen_text_buffer.ends_with("http:") || gen_text_buffer.ends_with("https:");
            if !is_url_single && gen_text_buffer.ends_with('/') {
                scratch.banned_ids.push(slash_id);
            }
            if !is_url_double {
                scratch.banned_ids.push(double_slash_id);
                scratch.banned_ids.push(space_double_slash_id);
            }
        }
        for ign_toks in pretokenized_ignores {
            let mut overlap = 0;
            for l in (1..=ign_toks.len().min(generate_hist.len())).rev() {
                if generate_hist.ends_with(&ign_toks[..l]) {
                    overlap = l;
                    break;
                }
            }
            if overlap < ign_toks.len() {
                let next_tok = ign_toks[overlap];
                let mut apply_bias = overlap > 0
                    || gen_text_buffer.ends_with('"')
                    || gen_text_buffer.ends_with(':')
                    || gen_text_buffer.ends_with(": ");
                if apply_bias && is_strict_json && self.json_protected_ids.contains(&next_tok) {
                    apply_bias = false;
                }
                if apply_bias {
                    scratch.banned_ids.push(next_tok);
                }
            }
        }
        if step_idx == 0 {
            scratch.banned_ids.push(self.eos_token_id);
            scratch.banned_ids.push(enter_id);
            if is_strict_json {
                if let Some(head) = self.qwen3_5.trim_head.as_ref() {
                    for &vid in head.kept.iter() {
                        if !self.json_starter_ids.contains(&vid) {
                            scratch.banned_ids.push(vid);
                        }
                    }
                }
            }
        }

        let recent_start = penalized_history.len().saturating_sub(self.repeat_last_n);
        let seen_tokens = &penalized_history[recent_start..];
        let fallback_eos = if is_strict_json && step_idx == 0 {
            open_bracket_id
        } else {
            self.eos_token_id
        };

        if let Some(head) = self.qwen3_5.trim_head.as_ref() {
            let banned_ids = std::mem::take(&mut scratch.banned_ids);
            let pp = crate::models::common::vocab_trim::StepPostProcess {
                seen_tokens,
                repeat_penalty: self.repeat_penalty,
                banned_ids: &banned_ids,
                prejudice_kept: active_prej.map(|cp| &cp.values),
                target_norm_cpu: active_prej.map(|cp| cp.target_norm_cpu.as_slice()),
                fallback_eos_id: fallback_eos,
            };
            let mut cand = if temperature <= 0.0 {
                let (tok, st) = head.greedy_single_pass_k(hidden, &pp, scratch)?;
                guard_stats.merge(&st);
                tok
            } else {
                guard_stats.steps += 1;
                head.sample_single_pass_k(hidden, logit_processor, &pp, scratch)?
            };
            scratch.banned_ids = banned_ids;
            if step_idx == 0 {
                if is_strict_json && !self.json_starter_ids.contains(&cand) {
                    cand = open_bracket_id;
                } else if !is_strict_json && cand == self.eos_token_id {
                    cand = enter_id;
                }
            }
            return Ok(cand);
        }

        // Untrimmed fallback path (when `trim_head` is None)
        if temperature <= 0.0 && full_prej.is_none() && pretokenized_ignores.is_empty() && !(is_strict_json && step_idx == 0) {
            let (mut cand, st) = self.qwen3_5.greedy_from_hidden_with_stats(hidden)?;
            guard_stats.merge(&st);
            let banned_hit = scratch.banned_ids.contains(&cand)
                || (self.repeat_penalty > 1.0 && seen_tokens.contains(&cand));
            if !banned_hit {
                if step_idx == 0 && cand == self.eos_token_id {
                    cand = enter_id;
                }
                return Ok(cand);
            }
        }

        let logits = self.qwen3_5.logits_from_hidden(hidden)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
        let logits = if let Some(prej) = full_prej {
            if !prej.device().same_device(logits.device()) {
                *prej = prej.to_device(logits.device())?;
            }
            logits.broadcast_sub(prej)?
        } else {
            logits
        };
        let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let len = logits_vec.len();
        if self.repeat_penalty > 1.0 && !seen_tokens.is_empty() {
            let mut dedup = std::collections::HashSet::new();
            for &t in seen_tokens {
                if dedup.insert(t) && (t as usize) < len {
                    let v = logits_vec[t as usize];
                    logits_vec[t as usize] = if v < 0.0 {
                        v * self.repeat_penalty
                    } else {
                        v / self.repeat_penalty
                    };
                }
            }
        }
        for &bid in &scratch.banned_ids {
            if (bid as usize) < len {
                logits_vec[bid as usize] = f32::NEG_INFINITY;
            }
        }
        if step_idx == 0 && is_strict_json {
            for (j, slot) in logits_vec.iter_mut().enumerate() {
                if !self.json_starter_ids.contains(&(j as u32)) {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
        let logits_t = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
        let mut cand = logit_processor.sample(&logits_t)?;
        guard_stats.steps += 1;
        if step_idx == 0 {
            if is_strict_json && !self.json_starter_ids.contains(&cand) {
                cand = open_bracket_id;
            } else if !is_strict_json && cand == self.eos_token_id {
                cand = enter_id;
            }
        }
        Ok(cand)
    }

    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>, 
        session_id: Option<String>, 
        kv_name: Option<String>,
        ignore_list: Option<&[String]>,
        semantic_prejudice: Option<&str> // 🌟 [추가] Semantic Prejudice 타겟
    ) -> Result<String> {
        use crate::models::qwen3_5::model::{
            CacheAccessMode, HYBRID_SNAPSHOT_VRAM_MIN_FREE_BYTES,
        };
        let user_text = crate::models::common::vocab_trim::prompt_text(&mes);
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

        let (ids_vec, spans) = self.encode_prompt_with_spans(&mes_text)?;
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&mes_text, &user_text, &ids_vec, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, temperature > 1e-7)?;

        self.qwen3_5.validate_input_ids(&ids_vec)?;
        let total_toks = ids_vec.len();
        if total_toks == 0 {
            return Ok(String::new());
        }
        let kv_len = self.qwen3_5.language_model.current_kv_len;

        let pixel_values: Option<&Tensor> = pixel_values.as_ref();
        let image_grid_thw: Option<&Tensor> = image_grid_thw.as_ref();
        let pixel_values_video: Option<&Tensor> = pixel_values_video.as_ref();
        let video_grid_thw: Option<&Tensor> = video_grid_thw.as_ref();

        let access_mode = if pixel_values.is_some()
            || pixel_values_video.is_some()
            || session_id.is_some()
            || kv_name.is_some()
            || kv_len > 0
            || !hybrid_prefix_cache_enabled()
        {
            CacheAccessMode::ReadOnly
        } else {
            CacheAccessMode::ReadWrite
        };

        let mut generate = Vec::new();
        let mut penalized_history = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        self.qwen3_5.language_model.set_planned_decode_tokens(sample_len as usize); // 🌟 [KV-PLAN v2] KV 여유 토큰 = 실제 max_tokens
        // 🌟 [SPD-12] 일회성 `generate()` 는 종료 직후 `clear_cache()` 되므로 불필요한 프리필·종료 디스크 플러시를 생략합니다.
        let prev_persist_kv = self.qwen3_5.language_model.persist_kv;
        self.qwen3_5.language_model.persist_kv = persist_oneshot_kv_enabled();

        let gen_t0 = std::time::Instant::now();
        let (mut input_ids, mut seqlen_offset, mut seq_len, mut initial_hidden): (Tensor, usize, usize, Option<Tensor>) =
            if access_mode == CacheAccessMode::ReadWrite {
                let (l_sys, l_ctx) = self.find_safe_prefix_checkpoints(&mes_text, &spans);
                let hit = self.snapshot_cache.lookup_best(&ids_vec).map(|(s, m)| (s.clone(), m));
                let dev = self.device.clone();
                if let Some((snap, matched)) = hit {
                    let restored_hidden = self.qwen3_5.restore_hybrid_snapshot(&snap, &dev)?;
                    if matched == total_toks {
                        let dummy_in = Tensor::zeros((1, 1), DType::U32, &dev)?;
                        (dummy_in, total_toks, 0, Some(restored_hidden))
                    } else {
                        let mut checkpoints = Vec::new();
                        if let Some(cp) = l_sys {
                            if cp > matched {
                                checkpoints.push(cp);
                            }
                        }
                        if let Some(cp) = l_ctx {
                            if cp > matched {
                                checkpoints.push(cp);
                            }
                        }
                        let mut captured_sys = None;
                        let mut captured_ctx = None;
                        let h = self
                            .qwen3_5
                            .forward_tokens_segmented(&ids_vec, matched, &checkpoints, &dev, |m, cp, hidden| {
                                if Some(cp) == l_sys {
                                    if let Ok(Some(s)) = m.capture_hybrid_snapshot(&ids_vec[..cp], hidden, &Device::Cpu) {
                                        captured_sys = Some(s);
                                    }
                                }
                                if Some(cp) == l_ctx {
                                    let active_target = if !dev.is_cpu()
                                        && crate::utils::device_utils::device_vram_snapshot(&dev)
                                            .map_or(false, |st| st.effective_free_bytes() >= HYBRID_SNAPSHOT_VRAM_MIN_FREE_BYTES)
                                    {
                                        &dev
                                    } else {
                                        &Device::Cpu
                                    };
                                    if let Ok(Some(s)) = m.capture_hybrid_snapshot(&ids_vec[..cp], hidden, active_target) {
                                        captured_ctx = Some(s);
                                    }
                                }
                            })
                            .await?;
                        if let Some(s) = captured_sys {
                            self.snapshot_cache.store_system_host(s);
                        }
                        if let Some(s) = captured_ctx {
                            self.snapshot_cache.store_active(s);
                        }
                        let dummy_in = Tensor::zeros((1, 1), DType::U32, &dev)?;
                        (dummy_in, total_toks, 0, Some(h))
                    }
                } else {
                    self.clear_kv_cache();
                    let mut checkpoints = Vec::new();
                    if let Some(cp) = l_sys {
                        checkpoints.push(cp);
                    }
                    if let Some(cp) = l_ctx {
                        checkpoints.push(cp);
                    }
                    let mut captured_sys = None;
                    let mut captured_ctx = None;
                    let h = self
                        .qwen3_5
                        .forward_tokens_segmented(&ids_vec, 0, &checkpoints, &dev, |m, cp, hidden| {
                            if Some(cp) == l_sys {
                                if let Ok(Some(s)) = m.capture_hybrid_snapshot(&ids_vec[..cp], hidden, &Device::Cpu) {
                                    captured_sys = Some(s);
                                }
                            }
                            if Some(cp) == l_ctx {
                                let active_target = if !dev.is_cpu()
                                    && crate::utils::device_utils::device_vram_snapshot(&dev)
                                        .map_or(false, |st| st.effective_free_bytes() >= HYBRID_SNAPSHOT_VRAM_MIN_FREE_BYTES)
                                {
                                    &dev
                                } else {
                                    &Device::Cpu
                                };
                                if let Ok(Some(s)) = m.capture_hybrid_snapshot(&ids_vec[..cp], hidden, active_target) {
                                    captured_ctx = Some(s);
                                }
                            }
                        })
                        .await?;
                    if let Some(s) = captured_sys {
                        self.snapshot_cache.store_system_host(s);
                    }
                    if let Some(s) = captured_ctx {
                        self.snapshot_cache.store_active(s);
                    }
                    let dummy_in = Tensor::zeros((1, 1), DType::U32, &dev)?;
                    (dummy_in, total_toks, 0, Some(h))
                }
            } else if kv_len > 0 {
                println!("[PARTIAL-PREFILL] Context partially restored ({}). Prefilling remaining tokens.", kv_len);
                let (ids, offset) = if kv_len >= total_toks {
                    let last_id = *ids_vec.last().unwrap_or(&0);
                    (Tensor::from_vec(vec![last_id], (1, 1), &self.device)?, total_toks.saturating_sub(1))
                } else {
                    let missing_ids = ids_vec[kv_len..].to_vec();
                    (Tensor::from_vec(missing_ids.clone(), (1, missing_ids.len()), &self.device)?, kv_len)
                };
                let slen = ids.dim(1)?;
                (ids, offset, slen, None)
            } else {
                self.clear_kv_cache();
                let full_ids_tensor = Tensor::from_vec(ids_vec.clone(), (1, total_toks), &self.device)?;
                let _ = self.qwen3_5.compute_and_set_rope_deltas(&full_ids_tensor, image_grid_thw, video_grid_thw);
                let slen = full_ids_tensor.dim(1)?;
                (full_ids_tensor, 0, slen, None)
            };

        let mut cur_pixel_values = pixel_values;
        let mut cur_pixel_values_video = pixel_values_video;

        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        let is_strict_json = mes_text.contains("/no_think") || mes_text.contains("RETURN JSON ONLY") || mes_text.contains("Return ONLY");
        let mut gen_text_buffer = String::new(); 
        let mut print_buffer = String::new();
        // 🌟 [UTF8-STREAM] 표준출력에는 완성된 UTF-8 글자만 내보냅니다. (gen_text_buffer·최종 텍스트는 그대로)
        let mut utf8_stream = crate::tokenizer::Utf8TokenStream::new();

        // 🌟 [Contrastive Semantic Steering] 공용 16-슬롯 LRU PrejudiceCache (OOM 방지 + active K 지원)
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        let active_prej_cached = if self.qwen3_5.trim_head.is_some() {
            semantic_prejudice.and_then(|t| self.prejudice_cached_active(t).ok())
        } else {
            None
        };
        if self.qwen3_5.trim_head.is_none() {
            if let Some(target_text) = semantic_prejudice {
                match self.prejudice_full_tensor(target_text) {
                    Ok(prej) => {
                        semantic_prejudice_tensor = Some(prej);
                        println!("[SEMANTIC-PREJUDICE] Cached Vector Prejudice for target ({} chars)", target_text.chars().count());
                    }
                    Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                }
            }
        }

        let pretokenized_ignores = pretokenize_ignore_list(&self.tokenizer, ignore_list, is_strict_json);
        let mut scratch = crate::models::common::vocab_trim::DecodeStepScratch::default();
        let mut prefill_ms = 0.0f64;
        let mut decode_t0: Option<std::time::Instant> = None;
        let mut guard_stats = crate::models::common::vocab_trim::StepGuardStats::default();

        for i in 0..sample_len {
            if i == 1 {
                prefill_ms = gen_t0.elapsed().as_secs_f64() * 1e3;
                decode_t0 = Some(std::time::Instant::now());
                crate::utils::trim_idle_gpu_pool(&self.device);
            }
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }

            let hidden = if let Some(h0) = initial_hidden.take() {
                h0
            } else {
                self.qwen3_5
                    .forward_last_hidden(
                        &input_ids,
                        cur_pixel_values,
                        image_grid_thw,
                        cur_pixel_values_video,
                        video_grid_thw,
                        None,
                        seqlen_offset,
                        session_id.clone(),
                        kv_name.clone(),
                    )
                    .await?
            };

            let next_token = self.select_token_from_hidden(
                &hidden,
                i as usize,
                is_strict_json,
                temperature,
                &mut logit_processor,
                &generate,
                &penalized_history,
                &gen_text_buffer,
                enter_id,
                think_token_id,
                open_bracket_id,
                lt_id,
                slash_id,
                double_slash_id,
                space_double_slash_id,
                &pretokenized_ignores,
                active_prej_cached.as_ref(),
                semantic_prejudice_tensor.as_mut(),
                &mut scratch,
                &mut guard_stats,
            )?;

            generate.push(next_token);
            if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
                penalized_history.push(next_token);
            }

            let mut is_json_finished = false; 

            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print_buffer.push_str(&utf8_stream.push(&self.tokenizer, next_token));
                gen_text_buffer.push_str(&piece);

                if print_buffer.len() >= 10 || piece.contains('\n') || next_token == self.eos_token_id {
                    if next_token == self.eos_token_id {
                        print_buffer.push_str(&utf8_stream.finish(&self.tokenizer));
                    }
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
                        print_buffer.push_str(&utf8_stream.finish(&self.tokenizer));
                        if !print_buffer.is_empty() {
                            print!("{}", print_buffer);
                            let _ = std::io::stdout().flush();
                        }
                        println!("\n[DEBUG-GEN] Balanced JSON detected. Stopping.");
                        is_json_finished = true; 
                    }
                } else if gen_text_buffer.trim_start().starts_with('[') {
                    let mut depth = 0;
                    let mut has_started = false;
                    for c in gen_text_buffer.chars() {
                        if c == '[' { depth += 1; has_started = true; }
                        else if c == ']' { depth -= 1; }
                    }
                    if has_started && depth == 0 && gen_text_buffer.trim_end().ends_with(']') {
                        print_buffer.push_str(&utf8_stream.finish(&self.tokenizer));
                        if !print_buffer.is_empty() {
                            print!("{}", print_buffer);
                            let _ = std::io::stdout().flush();
                        }
                        println!("\n[DEBUG-GEN] Balanced JSON array detected. Stopping.");
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
                
                if let Some(tx) = crate::progress::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::progress::CURRENT_UI_CATEGORY.read().unwrap().clone();

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
        crate::utils::trim_idle_gpu_pool(&self.device);

        let decode_ms = decode_t0
            .map(|t| t.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        if prefill_ms == 0.0 {
            prefill_ms = gen_t0.elapsed().as_secs_f64() * 1e3;
        }
        let decode_steps = generate.len().saturating_sub(1);
        self.last_gen_stats = Some((prefill_ms, decode_ms, decode_steps));
        self.last_guard_stats = Some(guard_stats.clone());
        crate::utils::score_dynamics::record_baseline("qwen35.prefill_ms", prefill_ms as f32);
        crate::utils::score_dynamics::record_baseline("qwen35.decode_ms", decode_ms as f32);
        crate::utils::score_dynamics::record_baseline("qwen35.decode_tokens", generate.len() as f32);

        if logis_schema::reflection::ReflectionLogMode::from_env() != logis_schema::reflection::ReflectionLogMode::Off {
            let head = self.qwen3_5.trim_head.as_ref();
            let vocab_v = self.qwen3_5.embed_vocab().unwrap_or(0) as u32;
            let active_k = head.map(|h| h.kept.len() as u32).unwrap_or(vocab_v);
            let scope_label = head.map(|h| h.label.clone()).unwrap_or_else(|| "full".into());
            let head_path = if guard_stats.quant_mode_a {
                "quant_mode_a".into()
            } else if head.is_some() {
                "trimmed_head".into()
            } else {
                "linear_proj".into()
            };
            let ctx = crate::models::common::vocab_trim::current_lang_context();
            let effective_langs = ctx.langs.union(crate::models::common::vocab_trim::Langs::present(&payload_text));
            let rec = logis_schema::reflection::ReflectionRecord {
                ts_ms: logis_schema::reflection::current_ts_ms(),
                run_id: logis_schema::reflection::current_run_id(),
                task_id: self.model_name.clone(),
                git_rev: option_env!("LOGIS_GIT_REV").unwrap_or("dev").into(),
                build_profile: logis_schema::reflection::build_profile_label(),
                features: Vec::new(),
                backend: backend_label_for(&self.device).into(),
                device_name: format!("{:?}", self.device),
                artifact: logis_schema::reflection::ArtifactReflection {
                    slot: self.model_name.clone(),
                    bytes: self.qwen3_5.resident_vram_bytes(),
                    sha256_16: String::new(),
                    recipe: "gguf".into(),
                    vocab_v,
                    artifact_k: active_k,
                    plan_sha256: None,
                    space_tag: None,
                },
                load: None,
                exec: Some(logis_schema::reflection::ExecReflection {
                    step: "generate".into(),
                    langs: [ctx.doc_lang, ctx.ui_lang, effective_langs.label()],
                    scope_label,
                    vocab_k: [vocab_v, active_k],
                    head_path,
                    guard_kind: head.map(|h| h.guard_kind_label().into()).unwrap_or_else(|| "none".into()),
                    tokens: [total_toks as u32, decode_steps as u32],
                    timing_ms: [prefill_ms as f32, decode_ms as f32, guard_stats.rescue_ms() as f32],
                    steps: [
                        guard_stats.fast_steps as u32,
                        0,
                        guard_stats.constrained_steps as u32,
                        guard_stats.flagged as u32,
                        guard_stats.rest_wins as u32,
                    ],
                    sampling: temperature > 1e-7,
                    trim_mode: head.map(|h| h.exec_mode.as_str().into()).unwrap_or_else(|| "full_untrimmed".into()),
                    clusters_checked: guard_stats.clusters_checked as u32,
                    rows_scored: guard_stats.rows_scored as u64,
                    ..Default::default()
                }),
            };
            logis_schema::reflection::record(&rec);
        }

        let res = self.tokenizer.token_decode(generate)?;

        if self.qwen3_5.language_model.persist_kv {
            if let Some(s_id) = &session_id {
                let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
                println!("[GEN-SAVE] Qwen 3.5 active KV blocks flushed to disk");
            }
        }
        self.qwen3_5.language_model.persist_kv = prev_persist_kv;

        self.qwen3_5.clear_cache();

        // 🌟 [종료 직후 메모리 즉각 강제 반환] 
        // 80% -> 65% 지연 반환 현상 해결! OS 레벨 메모리를 즉시 강제 회수하여 65% 상태를 유지시킵니다.
        if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }
        crate::models::common::legacy_working_set_trim();

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
        kv_name: Option<String>,
        ignore_list: Option<&[String]>,
        semantic_prejudice: Option<&str> // 🌟 [추가] Semantic Prejudice 타겟
    ) -> Result<GenerationResult> {
        let user_text = crate::models::common::vocab_trim::prompt_text(mes);
        let temperature = mes.temperature.unwrap_or(0.0);
        let sampling = temperature > 1e-7;
        let mut logit_processor = get_logit_processor(
            Some(temperature as f32), 
            Some(mes.top_p.unwrap_or(0.95) as f32), 
            Some(80), 
            mes.seed.unwrap_or(32768) as u64
        );

        self.qwen3_5.language_model.persist_kv = true;
        let mut generate = Vec::new();
        let mut penalized_history = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        self.qwen3_5.language_model.set_planned_decode_tokens(sample_len as usize); // 🌟 [KV-PLAN v2] KV 여유 토큰 = 실제 max_tokens

        let (mut input_ids, mut seqlen_offset, mut cur_pixel_values, mut cur_image_thw, mut cur_pixel_values_video, mut cur_video_thw, is_strict_json) = if !is_continuation {

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
            
            let strict_json = text.contains("/no_think") || text.contains("RETURN JSON ONLY") || text.contains("Return ONLY");
            let (ids_vec, spans) = self.encode_prompt_with_spans(&text)?;
            let (payload_text, payload_ids) =
                self.collect_payload_text_and_ids(&text, &user_text, &ids_vec, &spans);
            self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, sampling)?;

            self.qwen3_5.validate_input_ids(&ids_vec)?;
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
                (ids, offset, px_vals, img_thw, vid_px_vals, vid_thw, strict_json)
            } else {
                self.clear_kv_cache();
                let full_ids_tensor = Tensor::from_vec(ids_vec.clone(), (1, total_toks), &self.device)?;
                let _ = self.qwen3_5.compute_and_set_rope_deltas(&full_ids_tensor, img_thw.as_ref(), vid_thw.as_ref());
                
                let ids = Tensor::from_vec(ids_vec, (1, total_toks), &self.device)?;
                (ids, 0, px_vals, img_thw, vid_px_vals, vid_thw, strict_json)
            }
        } else {
            let payload_text = crate::models::common::vocab_trim::extract_data_payload_only(&user_text);
            let payload_ids = self.tokenizer.text_encode_vec(payload_text.clone(), false).unwrap_or_default();
            self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, sampling)?;
            let mes_check = self.chat_template.apply_chat_template(mes).unwrap_or_default();
            let strict_json = mes_check.contains("/no_think") || mes_check.contains("RETURN JSON ONLY") || mes_check.contains("Return ONLY");
            let ids = Tensor::from_vec(vec![last_token.unwrap()], (1, 1), &self.device)?;
            (ids, last_offset, None, None, None, None, strict_json)
        };

        let mut seq_len = input_ids.dim(1)?;
        let mut is_finished = false;
        let mut final_token = 0;
        
        println!("\n[AI Thinking...]");

        crate::models::qwen::generate::wait_for_global_io().await;
        
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        
        let mut gen_text_buffer = String::new();
        let mut print_buffer = String::new();
        // 🌟 [UTF8-STREAM] 표준출력에는 완성된 UTF-8 글자만 내보냅니다. (gen_text_buffer·최종 텍스트는 그대로)
        let mut utf8_stream = crate::tokenizer::Utf8TokenStream::new();

        // 🌟 [Contrastive Semantic Steering] 공용 16-슬롯 LRU PrejudiceCache (active K 우선)
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        let active_prej_cached = if self.qwen3_5.trim_head.is_some() {
            semantic_prejudice.and_then(|t| self.prejudice_cached_active(t).ok())
        } else {
            None
        };
        if self.qwen3_5.trim_head.is_none() {
            if let Some(target_text) = semantic_prejudice {
                match self.prejudice_full_tensor(target_text) {
                    Ok(prej) => {
                        semantic_prejudice_tensor = Some(prej);
                        println!("[SEMANTIC-PREJUDICE] Cached Vector Prejudice for target ({} chars)", target_text.chars().count());
                    }
                    Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                }
            }
        }

        let pretokenized_ignores = pretokenize_ignore_list(&self.tokenizer, ignore_list, is_strict_json);
        let mut scratch = crate::models::common::vocab_trim::DecodeStepScratch::default();
        let mut guard_stats = crate::models::common::vocab_trim::StepGuardStats::default();

        for i in 0..sample_len {
            if i == 1 {
                crate::utils::trim_idle_gpu_pool(&self.device);
            }
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            crate::models::qwen::generate::wait_for_global_io().await;

            let hidden = self
                .qwen3_5
                .forward_last_hidden(
                    &input_ids,
                    cur_pixel_values.as_ref(),
                    cur_image_thw.as_ref(),
                    cur_pixel_values_video.as_ref(),
                    cur_video_thw.as_ref(),
                    None,
                    seqlen_offset,
                    session_id.clone(),
                    kv_name.clone(),
                )
                .await?;

            let next_token = self.select_token_from_hidden(
                &hidden,
                i as usize,
                is_strict_json,
                temperature,
                &mut logit_processor,
                &generate,
                &penalized_history,
                &gen_text_buffer,
                enter_id,
                think_token_id,
                open_bracket_id,
                lt_id,
                slash_id,
                double_slash_id,
                space_double_slash_id,
                &pretokenized_ignores,
                active_prej_cached.as_ref(),
                semantic_prejudice_tensor.as_mut(),
                &mut scratch,
                &mut guard_stats,
            )?;

            generate.push(next_token);
            if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
                penalized_history.push(next_token);
            }
            final_token = next_token;

            let mut is_json_finished = false;

            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                print_buffer.push_str(&utf8_stream.push(&self.tokenizer, next_token));
                gen_text_buffer.push_str(&piece);

                if print_buffer.len() >= 10 || piece.contains('\n') || next_token == self.eos_token_id {
                    if next_token == self.eos_token_id {
                        print_buffer.push_str(&utf8_stream.finish(&self.tokenizer));
                    }
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
                        print_buffer.push_str(&utf8_stream.finish(&self.tokenizer));
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
                
                if let Some(tx) = crate::progress::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::progress::CURRENT_UI_CATEGORY.read().unwrap().clone();

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
        crate::utils::trim_idle_gpu_pool(&self.device);

        let res_text = self.tokenizer.token_decode(generate)?;
        
        if let Some(s_id) = &session_id {
            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
        }

        // 🌟 [종료 직후 메모리 즉각 강제 반환] 
        if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }
        crate::models::common::legacy_working_set_trim();

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
        self.qwen3_5.language_model.persist_kv = true;

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
            let path = crate::utils::paths::get_kv_dir().join(s_id);
            if !path.exists() { std::fs::create_dir_all(&path)?; }
            
            std::fs::write(path.join("tokens.json"), serde_json::to_string(&safe_input_ids)?)?;

            let _ = self.qwen3_5.language_model.force_flush_all_active_blocks(s_id, kv_name.as_deref()).await;
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            println!("[PREFILL-WAIT] Waiting for SSD write to complete...");
            crate::models::qwen::generate::drain_global_io().await;
            println!("[PREFILL-SAVE] Confirm: Qwen 3.5 Base Context prefilled and safely flushed to disk. ({} tokens)", safe_toks);
        }

        // 🌟 [종료 직후 메모리 즉각 강제 반환]
        if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }
        crate::models::common::legacy_working_set_trim();

        Ok(safe_toks)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3_5.clear_cache();
    }

    /// Clear live KV/GDN caches AND the 2-tier system/context prefix snapshot cache (TaskEnd / model swap).
    pub fn clear_all_caches(&mut self) {
        self.qwen3_5.clear_cache();
        self.snapshot_cache.clear();
    }
}