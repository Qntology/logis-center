use crate::openai_types::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use crate::utils::GpuDeviceExt;
use candle_nn::VarBuilder;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::utils::resources::KvResidency;
use crate::utils::{
    find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

struct PromptKvSnapshot {
    prompt_hash: u64,
    vocab_label: String,
    seqlen: usize,
    last_hidden: Tensor,
    layer_kvs: Vec<Option<(Tensor, Tensor)>>,
    kv_exact: bool,
}

fn hash_str_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
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

fn retry_kv_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("LOGIS_RETRY_KV_CACHE")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

pub struct Qwen3GenerateModel {
    chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    qwen3: Qwen3Model,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
    /// 🌟 [KV-PLAN v2] 직전 디코딩 진입의 KV 배치 판정 (히스테리시스). 생성기를 버리면(퍼지 · 작업 종료) 함께 사라집니다.
    last_kv_plan: Option<KvResidency>,
    /// 🌟 [SPD-02] JSON 문법 보호 토큰 ID 캐시 (디코딩 스텝마다 BPE decode 제거)
    json_protected_ids: std::collections::HashSet<u32>,
    /// Valid opening JSON starter tokens (`{`, `{"`, `{\n`, `[`, `[{"`, ...) preserved at i == 0
    json_starter_ids: std::collections::HashSet<u32>,
    /// 🌟 [SPD-02] 편견 벡터 LRU 캐시 (최대 16개, `(hash(target), vocab_label) -> Tensor`)
    prejudice_cache: Vec<((u64, String), Tensor)>,
    /// 🌟 [PRC-05] 호출부별 repetition_penalty 일시 오버라이드 (리터럴 복사 필드 1.0 등)
    repetition_penalty_override: Option<f32>,
    /// 🌟 [SPD-03] 1-슬롯 프롬프트 프리필 KV 스냅샷 (재시도 시 프리필 생략)
    prompt_kv_snapshot: Option<PromptKvSnapshot>,
    /// 🌟 [SPD-01] 직전 generate 호출 프로파일링: `(prefill_ms, decode_ms, decode_steps, slow_path_steps, retry_kv_hit)`
    last_gen_stats: Option<(f64, f64, usize, usize, bool)>,
    /// Phase 3-1 D2: 누적/직전 생성 가드 통계 (`StepGuardStats`)
    last_guard_stats: Option<crate::models::common::vocab_trim::StepGuardStats>,
}

/// Resolve EOS token IDs dynamically from tokenizer & generation_config without hardcoded 151643/151645 literals.
fn resolve_qwen3_eos(tokenizer: &TokenizerModel, cfg: &Qwen3GenerationConfig) -> (u32, u32) {
    let vocab = tokenizer.tokenizer.get_vocab_size(true);
    let tok_lookup = |name: &str| -> Option<u32> {
        tokenizer.tokenizer.token_to_id(name).filter(|&id| (id as usize) < vocab)
    };
    let e1 = cfg
        .eos_token_id
        .first()
        .copied()
        .map(|v| v as u32)
        .filter(|&id| (id as usize) < vocab)
        .or_else(|| tok_lookup("<|endoftext|>"))
        .or_else(|| tok_lookup("<|im_end|>"))
        .unwrap_or(0);
    let e2 = cfg
        .eos_token_id
        .get(1)
        .copied()
        .map(|v| v as u32)
        .filter(|&id| (id as usize) < vocab)
        .or_else(|| tok_lookup("<|im_end|>"))
        .or_else(|| tok_lookup("<|endoftext|>"))
        .unwrap_or(e1);
    (e1, e2)
}

/// 🌟 [KV-PLAN v2] Qwen3-0.6B 디코딩 한 스텝의 활성값 순간치(대략).
///  층마다 KV 를 한 토큰씩 이어 붙인 새 텐서(원본 BF16 이면 1k 토큰에 층당 약 4MB, 압축이면 그것을 푼 사본)와
///  logits(151,936 × F32 ≈ 0.6MB) · 편견 벡터 뺄셈 정도라 실제로는 수십 MB 아래입니다.
///  Qwen3.5 의 `DECODE_TRANSIENT_BYTES` 와 같은 64MB 로 두어 VRAM 여유는 max(96MB, 2 × 64MB) = 128MB 입니다 (예전 고정 640MB).
pub const QWEN3_DECODE_TRANSIENT_BYTES: u64 = 64 * 1024 * 1024;

/// 🌟 [KV-PLAN v2] 이번 생성의 Qwen3 KV 가 VRAM 에 실제로 차지하는 토큰당 바이트 (전 층 K+V).
///  원본 KV(`exact`)는 원소 크기 그대로(GPU 는 BF16 2바이트), 압축 KV 는 TurboQuant 벡터별 FP8 행(d + 8 바이트)입니다.
///  (압축 캐시를 쓰지 않는 빌드의 FP8 보관은 d 바이트라 이 값이 6% 크게 잡습니다 — 과대평가 쪽이라 안전합니다)
pub fn qwen3_kv_bytes_per_token(layers: usize, kv_heads: usize, head_dim: usize, exact: bool, exact_elem_bytes: usize) -> u64 {
    if exact {
        crate::utils::resources::kv_bytes_per_token_exact(layers, kv_heads, head_dim, exact_elem_bytes)
    } else {
        crate::utils::resources::kv_bytes_per_token_compressed(layers, kv_heads, head_dim)
    }
}

// =====================================================================
// 🌟 [VOCAB CHUNK] 편견 벡터 계산의 순간 VRAM 을 상수로 고정합니다.
// ---------------------------------------------------------------------
//  ── 왜 캐시가 아니라 청킹인가 ──
//   정규화 행렬을 캐시하면 순간 피크는 사라지지만 622MB 가 '상주' 합니다.
//   정상 구간 VRAM 이 2.0GB → 2.6GB 로 올라가므로,
//   임베딩 모델과의 동시 상주 여유가 그만큼 줄어듭니다.
//   청킹은 상주 0MB · 순간 134MB 이므로 두 축 모두에서 우월합니다.
//   qwen3_5/generate.rs 가 이미 같은 판단으로 VOCAB_CHUNK 를 쓰고 있으며,
//   그쪽에는 캐시가 아예 없습니다. 두 모델의 처리를 일치시킵니다.
//
//  ── 실측 규모 (Qwen3-0.6B) ──
//   token_embd = 151,936 × 1,024
//     기존: all_embs(F32) 622MB + sqr 622MB + normalized 622MB
//           = 순간 1,866MB (계측 실측 1,657MB 와 일치)
//     청킹: blk(F32) 67MB + blk_normalized 67MB = 순간 134MB
//
//  ── 왜 결과가 동일한가 ──
//   코사인 유사도는 어휘 행마다 완전히 독립입니다.
//   행을 블록으로 잘라 각각 정규화·내적한 뒤 이어 붙이면
//   전량 계산과 비트 단위로 같은 벡터가 나옵니다.
//   임계치·증폭 수식은 sim 이 완성된 뒤에 적용하므로 손대지 않습니다.
//
//  ── 값의 성격 ──
//   16,384 는 판정 임계치가 아니라 '한 번에 GPU 로 올릴 행 수' 입니다.
//   크게 잡으면 순간 점유가 늘고 작게 잡으면 커널 호출이 늘 뿐,
//   어떤 값을 써도 계산 결과는 같습니다.
//   qwen3_5 와 같은 값을 써서 두 경로의 동작을 일치시킵니다.
const VOCAB_CHUNK: usize = 16_384;

impl Qwen3GenerateModel {
    /// 🌟 [VOCAB TRIM] 지금의 문서 언어로 출력 투영의 남긴 행을 맞춥니다. 생성마다 처음에 부릅니다.
    pub fn refresh_vocab(&mut self) -> Result<()> {
        self.refresh_vocab_for("", false)
    }

    /// [`Self::refresh_vocab`] + 이번 생성의 프롬프트 글에 나온 문자 체계. 샘플링(온도 > 0)이면 전체 행 (`vocab_trim::refresh_for`)
    pub fn refresh_vocab_for(&mut self, prompt: &str, sampling: bool) -> Result<()> {
        if let Some(src) = self.qwen3.vocab_table() {
            crate::models::common::vocab_trim::refresh_for(&mut self.qwen3.trim_head, &src, &self.tokenizer.tokenizer, &self.device, prompt, sampling)?;
        }
        Ok(())
    }

    fn refresh_vocab_from_payload_ids(&mut self, payload_text: &str, payload_ids: &[u32], sampling: bool) -> Result<()> {
        if let Some(src) = self.qwen3.vocab_table() {
            crate::models::common::vocab_trim::refresh_for_ids(
                &mut self.qwen3.trim_head,
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

    fn encode_with_spans(&self, text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let enc = self
            .tokenizer
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }

    fn collect_payload_text_and_ids(
        &self,
        mes_render: &str,
        user_text: &str,
        ids_vec: &[u32],
        spans: &[(usize, usize)],
    ) -> (String, Vec<u32>) {
        let ranges = crate::models::common::vocab_trim::extract_data_payload_ranges(mes_render);
        if ranges.is_empty() {
            return (user_text.to_string(), ids_vec.to_vec());
        }
        let mut payload_text = String::new();
        let mut payload_ids = Vec::new();
        for &(rs, re) in &ranges {
            let chunk = mes_render[rs..re].trim();
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
        let vocab_v = self.qwen3.embed_vocab().unwrap_or(0) as u32;
        let active_k = self
            .qwen3
            .trim_head
            .as_ref()
            .map(|h| h.kept.len() as u32)
            .unwrap_or(vocab_v);
        let (table_mode, resident_table_mb, mapped_table_mb) = match self.qwen3.vocab_table() {
            Some(src) => {
                let row_bytes = (src.cols / src.dtype.block_size().max(1)) * src.dtype.type_size();
                let total_bytes = (src.rows * row_bytes) as f32;
                let head_bytes = self.qwen3.trim_head.as_ref().map_or(0.0, |h| h.head_bytes() as f32);
                ("mapped".to_string(), head_bytes / 1e6, total_bytes / 1e6)
            }
            None => (
                if self.qwen3.is_q8_resident() { "quant".to_string() } else { "resident".to_string() },
                self.qwen3.resident_vram_bytes() as f32 / 1e6,
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
                slot: "qwen3_0_6b".into(),
                bytes: self.qwen3.resident_vram_bytes(),
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

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let load_t0 = std::time::Instant::now();
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let vram_before = crate::utils::device_utils::device_vram_snapshot(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let qwen3 = Qwen3Model::new(&cfg, vb)?;
        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let vram_after = crate::utils::device_utils::device_vram_snapshot(device);

        let json_protected_ids = build_json_protected_ids(&tokenizer);
        let json_starter_ids = build_json_starter_ids(&tokenizer);
        let (eos_token_id1, eos_token_id2) = resolve_qwen3_eos(&tokenizer, &generation_config);

        let gen = Qwen3GenerateModel {
            chat_template,
            tokenizer,
            qwen3,
            device: device.clone(),
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name: "qwen3".to_string(),
            last_kv_plan: None,
            json_protected_ids,
            json_starter_ids,
            prejudice_cache: Vec::new(),
            repetition_penalty_override: None,
            prompt_kv_snapshot: None,
            last_gen_stats: None,
            last_guard_stats: None,
        };
        gen.record_load_reflection("safetensors", load_t0.elapsed().as_secs_f32() * 1e3, vram_before, vram_after);
        Ok(gen)
    }
    /// 🌟 [Q8-RESIDENT] CUDA/ROCm/Vulkan 에서는 Q8_0 가중치를 그대로 GPU 에 상주시키고(정확 W8A16), 그 외 장치나
    /// `LOGIS_QWEN3_DENSE=1` 이면 기존처럼 BF16/F32 로 역양자화해 올립니다.
    pub fn init_from_gguf(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let dense_env = std::env::var("LOGIS_QWEN3_DENSE")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
            .unwrap_or(false);
        let q8 = !dense_env && { let d = get_device(device); d.is_cuda() || d.is_rocm() || d.is_vulkan() };
        if dense_env {
            println!("[QWEN3-Q8] LOGIS_QWEN3_DENSE=1 → 기존 BF16 역양자화 경로로 로드합니다.");
        }
        Self::init_from_gguf_with(path, device, dtype, q8)
    }

    /// 🌟 [Q8-RESIDENT] `q8_resident` 를 직접 고르는 로더 (테스트·A/B 용). Q8 로드가 실패하면 기존 경로로 돌아갑니다.
    pub fn init_from_gguf_with(path: &str, device: Option<&Device>, dtype: Option<DType>, q8_resident: bool) -> Result<Self> {
        let load_t0 = std::time::Instant::now();
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let vram_before = crate::utils::device_utils::device_vram_snapshot(&device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);

        let gguf_files = crate::utils::find_type_files(path, "gguf")?;
        let model_file = gguf_files.first().ok_or_else(|| anyhow::anyhow!("No GGUF found"))?;

        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        if q8_resident {
            // mmap 페이지에서 GPU 로 바로 복사합니다 (힙 사본·F32/F16 중간 사본 없음). 맵은 로드 직후 닫습니다.
            let loaded = (|| -> Result<Qwen3Model> {
                let file = std::fs::File::open(model_file)?;
                let mmap = std::sync::Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? });
                let ct = candle_core::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..]))?;
                println!(
                    "[QWEN3-Q8] Q8_0 가중치를 역양자화 없이 GPU 에 상주시킵니다 (정확 W8A16 · 활성 {:?} · F32 logits)",
                    dtype
                );
                // 🌟 [VOCAB TRIM / SPD-04] 매핑 테이블이 켜지면 `token_embd.weight` (~165MB) 의 초기 GPU 업로드를 생략합니다.
                let mapped = crate::models::common::vocab_trim::mapped_table(&device, Some(&mmap), Some(&ct));
                let m = Qwen3Model::new_from_gguf_q8_with_mapped(&cfg, &ct, &mmap, &device, dtype, mapped)?;
                Ok(m)
            })();
            match loaded {
                Ok(qwen3) => {
                    let vram_after = crate::utils::device_utils::device_vram_snapshot(&device);
                    let generation_config: Qwen3GenerationConfig =
                        serde_json::from_slice(&std::fs::read(&generation_config_path).unwrap_or_default()).unwrap_or_default();
                    let json_protected_ids = build_json_protected_ids(&tokenizer);
                    let json_starter_ids = build_json_starter_ids(&tokenizer);
                    let (eos_token_id1, eos_token_id2) = resolve_qwen3_eos(&tokenizer, &generation_config);
                    let gen = Qwen3GenerateModel {
                        chat_template,
                        tokenizer,
                        qwen3,
                        device: device.clone(),
                        eos_token_id1,
                        eos_token_id2,
                        generation_config,
                        model_name: "qwen3".to_string(),
                        last_kv_plan: None,
                        json_protected_ids,
                        json_starter_ids,
                        prejudice_cache: Vec::new(),
                        repetition_penalty_override: None,
                        prompt_kv_snapshot: None,
                        last_gen_stats: None,
                        last_guard_stats: None,
                    };
                    gen.record_load_reflection("Q8_0", load_t0.elapsed().as_secs_f32() * 1e3, vram_before, vram_after);
                    return Ok(gen);
                }
                Err(e) => println!("[QWEN3-Q8] ⚠️ Q8 상주 로드 실패 → 기존 BF16 역양자화 경로로 로드합니다: {e}"),
            }
        }

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
        let vram_after = crate::utils::device_utils::device_vram_snapshot(&device);

        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path).unwrap_or_default()).unwrap_or_default();

        let json_protected_ids = build_json_protected_ids(&tokenizer);
        let json_starter_ids = build_json_starter_ids(&tokenizer);
        let (eos_token_id1, eos_token_id2) = resolve_qwen3_eos(&tokenizer, &generation_config);

        let gen = Qwen3GenerateModel {
            chat_template,
            tokenizer,
            qwen3,
            device: device.clone(),
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name: "qwen3".to_string(),
            last_kv_plan: None,
            json_protected_ids,
            json_starter_ids,
            prejudice_cache: Vec::new(),
            repetition_penalty_override: None,
            prompt_kv_snapshot: None,
            last_gen_stats: None,
            last_guard_stats: None,
        };
        gen.record_load_reflection("BF16", load_t0.elapsed().as_secs_f32() * 1e3, vram_before, vram_after);
        Ok(gen)
    }

    fn effective_repetition_penalty(&self) -> f32 {
        self.repetition_penalty_override
            .unwrap_or(self.generation_config.repetition_penalty)
    }

    /// 🌟 [PRC-05] 리터럴 복사 필드 등 반복 페널티를 일시 조정합니다 (`None` = generation_config 기본값).
    pub fn set_repetition_penalty_override(&mut self, penalty: Option<f32>) {
        self.repetition_penalty_override = penalty;
    }

    /// 🌟 [SPD-03] 재시도용 프롬프트 프리필 KV 스냅샷을 비웁니다.
    pub fn clear_prompt_kv_snapshot(&mut self) {
        self.prompt_kv_snapshot = None;
    }

    /// Clear live KV caches and retry snapshot together.
    pub fn clear_all_caches(&mut self) {
        self.clear_kv_cache();
    }

    /// 🌟 [SPD-03] 활성 KV 캐시만 비우고, 동일 프롬프트 재시도 스냅샷은 유지합니다.
    pub fn clear_kv_cache_keep_retry_snapshot(&mut self) {
        self.qwen3.clear_kv_cache();
    }

    /// 🌟 [SPD-01] 직전 `generate` 호출의 타이밍 `(prefill_ms, decode_ms, decode_steps, slow_path_steps, retry_kv_hit)`.
    pub fn last_gen_stats(&self) -> Option<(f64, f64, usize, usize, bool)> {
        self.last_gen_stats
    }

    /// Phase 3-1 D2: 직전 생성의 가드/구출 통계 (`StepGuardStats`).
    pub fn last_guard_stats(&self) -> Option<&crate::models::common::vocab_trim::StepGuardStats> {
        self.last_guard_stats.as_ref()
    }

    pub fn resident_vram_bytes(&self) -> u64 {
        self.qwen3.resident_vram_bytes()
    }

    fn prejudice_cached_active(&self, target_text: &str) -> Result<crate::models::common::prejudice::CachedPrejudice> {
        let target_ids = self.tokenizer.text_encode_vec(target_text.to_string(), false)?;
        if target_ids.is_empty() {
            return Err(anyhow::anyhow!("empty target tokens"));
        }
        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
        let target_emb = self.qwen3.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
        let target_emb_sum = target_emb.sum_keepdim(1)?;
        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
        let target_vec = target_emb_sum.broadcast_div(&len_tensor)?.squeeze(0)?.squeeze(0)?;

        let vocab_label = self
            .qwen3
            .trim_head
            .as_ref()
            .map(|h| format!("{}:{:016x}", h.label, h.kept_hash))
            .unwrap_or_else(|| "full".into());
        let num_rows = self
            .qwen3
            .trim_head
            .as_ref()
            .map(|h| h.kept.len())
            .unwrap_or(self.qwen3.embed_vocab()?);
        let cache_key = format!("qwen3:k:{:016x}:{vocab_label}", hash_str_u64(target_text));
        crate::models::common::prejudice::get_or_build(
            &cache_key,
            &target_vec,
            num_rows,
            &self.device,
            |off, take| self.qwen3.embed_kept_rows_f32(off, take, &self.device),
        )
    }

    // =====================================================================
    // 🌟 [VOCAB CHUNK] 편견 벡터 공용 계산기
    // ---------------------------------------------------------------------
    //  ── 왜 메서드로 빼는가 ──
    //   기존에는 generate 와 generate_part 가 같은 클로저를 각각 복제해
    //   갖고 있었습니다. 한쪽만 고치면 다른 쪽이 조용히 옛 동작(1.6GB 순간
    //   점유)을 유지하므로 단일 메서드로 통합합니다.
    //   실측 로그의 [SEMANTIC-PREJUDICE] 130회는 전부 generate 경로였지만,
    //   generate_part 도 동일한 폭탄을 갖고 있어 언제든 재현될 수 있습니다.
    //
    //  ── 어디를 바꿨고 어디를 그대로 뒀나 ──
    //   바꾼 곳: all_embs 를 통째로 F32 로 올리던 세 줄을 블록 루프로 대체.
    //   그대로 둔 곳: target 벡터 계산, threshold 0.65, affine(15.0) 증폭.
    //   sim 이 완성된 뒤의 수식은 한 글자도 손대지 않았으므로
    //   같은 입력에 대해 비트 단위로 같은 결과가 나옵니다.
    //
    //  ── qwen3_5 와의 대조 ──
    //   qwen3_5/generate.rs 는 get_embed_tokens() 뒤에 to_dtype 을 붙이지
    //   않고 narrow 이후에만 F32 로 올립니다. 그 한 줄의 위치 차이가
    //   134MB 와 1,866MB 를 가릅니다. 여기서 같은 위치로 맞춥니다.
    // =====================================================================
    fn prejudice_vector(&mut self, target_text: &str) -> Result<Tensor> {
        // 🌟 [SPD-02] 동일 (target_text, vocab_trim_label) 편견 벡터 캐시 (최대 16개, ~600KB/항목)
        let vocab_label = self
            .qwen3
            .trim_head
            .as_ref()
            .map(|h| h.label.clone())
            .unwrap_or_default();
        let cache_key = (hash_str_u64(target_text), vocab_label);
        if let Some(pos) = self.prejudice_cache.iter().position(|(k, _)| *k == cache_key) {
            let entry = self.prejudice_cache.remove(pos);
            let cached = entry.1.clone();
            self.prejudice_cache.push(entry);
            return Ok(cached);
        }

        let target_ids = self.tokenizer.text_encode_vec(target_text.to_string(), false)?;
        if target_ids.is_empty() {
            return Err(anyhow::anyhow!("empty target tokens"));
        }

        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.device)?;
        let target_emb = self.qwen3.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
        let target_emb_sum = target_emb.sum_keepdim(1)?;
        let len_tensor = Tensor::new(target_ids.len() as f32, &self.device)?;
        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;

        let target_norm = target_vec.sqr()?.sum_all()?.sqrt()?;
        let target_normalized = target_vec.broadcast_div(&target_norm)?;
        // 🌟 루프 밖에서 한 번만 contiguous 로 확정합니다.
        //    블록마다 unsqueeze 를 반복하면 stride 재계산이 블록 수만큼 발생합니다.
        let target_col = target_normalized.unsqueeze(1)?.contiguous()?;

        // 🌟 [핵심] to_dtype 을 여기 붙이지 않습니다. 원본 참조만 들고,
        //    narrow 로 잘라낸 블록에만 F32 승격을 겁니다.
        // 🌟 [Q8-RESIDENT] Dense 표면 narrow 후 F32 승격(기존과 동일), Q8_0 표면 블록 행만 정확 복원합니다.
        let vocab = self.qwen3.embed_vocab()?;
        let mut sim_parts: Vec<Tensor> = Vec::with_capacity(vocab / VOCAB_CHUNK + 1);
        let mut off = 0usize;
        while off < vocab {
            let take = (vocab - off).min(VOCAB_CHUNK);
            let blk = self.qwen3.embed_rows_f32(off, take)?;
            let blk_norm = blk.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
            let blk_normalized = blk.broadcast_div(&blk_norm)?;
            sim_parts.push(blk_normalized.matmul(&target_col)?.squeeze(1)?);
            // 다음 블록을 올리기 전에 이번 블록을 확실히 떨어뜨립니다.
            // 이것이 없으면 candle 의 지연 해제로 두 블록이 겹칠 수 있습니다.
            drop(blk_normalized);
            drop(blk_norm);
            drop(blk);
            off += take;
        }
        let sim = Tensor::cat(&sim_parts, 0)?;
        drop(sim_parts);

        // 🌟 [방향 B: Threshold 노이즈 게이트 + Exponential 증폭]
        //    원본 수식 그대로입니다.
        let threshold = Tensor::new(0.65f32, &self.device)?;
        let one = Tensor::new(1.0f32, &self.device)?;
        let sim_relu = sim.broadcast_sub(&threshold)?.relu()?;
        let prejudice = sim_relu.affine(15.0, 0.0)?.exp()?.broadcast_sub(&one)?;

        // 🌟 [로그 축약] 기존에는 수천 자 target 전문을 130회 출력해
        //    터미널이 그것만으로 가득 찼고 정작 필요한 진단이 묻혔습니다.
        //    길이와 블록 수만 남깁니다.
        println!(
            "[SEMANTIC-PREJUDICE] 편견 벡터 계산 (target {}자 | vocab {} → {}블록)",
            target_text.chars().count(),
            vocab,
            (vocab + VOCAB_CHUNK - 1) / VOCAB_CHUNK
        );
        // 🌟 [VOCAB TRIM] 잘린 행 자리는 +∞ (빼면 −∞): 선입견 블록의 채움값이 로짓으로 새어 나가지 않게
        let out = crate::models::common::vocab_trim::exclude_dropped(self.qwen3.trim_head.as_ref(), prejudice)?;
        if self.prejudice_cache.len() >= 16 {
            self.prejudice_cache.remove(0);
        }
        self.prejudice_cache.push((cache_key, out.clone()));
        Ok(out)
    }

    /// 🌟 [KV-PRECISION] 이번 생성의 KV 를 원본 정밀도(BF16)로 둘지, TurboQuant 로 압축할지 정해 모델에 설정하고 돌려줍니다.
    /// 캐시가 빈 상태(새 생성)에서 프리필 전에 호출합니다. 판정 기준은 kv_turbo::exact_kv_affordable 을 보세요.
    fn plan_kv_precision(&mut self, planned_tokens: usize) -> bool {
        // 🌟 [KV-PLAN v2] 직전 호출이 KV 를 RAM 으로 내렸어도 이번 프리필 KV 는 VRAM 에 만듭니다 (프로세스의 첫 호출과 같은 상태).
        //    예전에는 그 판정이 다음 호출의 프리필까지 남아 청크마다 층마다 KV 를 CPU 로 내렸다 올렸습니다.
        //    VRAM 이 빠듯하면 아래 정밀도 판정이 압축 KV 를 고르고, 디코딩 진입 계획(plan_decode_kv)이 다시 정합니다.
        self.qwen3.set_kv_residency(KvResidency::Vram);
        self.qwen3.set_planned_kv_tokens(planned_tokens);
        let (layers, heads, head_dim) = self.qwen3.kv_geometry();
        let elem = if self.device.is_cuda_or_rocm() { 2 } else { 4 };
        let exact_bytes = (layers * heads * head_dim * 2 * elem * planned_tokens) as u64;
        // 🌟 [P4-F15] 고정 640 MiB 마진 대신 Qwen3.5 와 동일한 비례 마진(`kv_vram_margin_bytes`) 사용
        //    (`LOGIS_KV_PLAN=legacy` 일 때만 예전 640 MiB 고정값).
        let margin = if crate::utils::resources::kv_plan_legacy() {
            crate::utils::resources::KV_VRAM_SAFETY_MARGIN_BYTES
        } else {
            crate::utils::resources::kv_vram_margin_bytes(96 * 1024 * 1024)
        };
        let (exact, free) = crate::models::common::kv_turbo::exact_kv_affordable_with_margin(
            &self.device,
            exact_bytes,
            0,
            margin,
        );
        self.qwen3.set_kv_exact(exact);
        if self.device.is_cuda_or_rocm() || self.device.is_vulkan() {
            println!(
                "[KV-PRECISION] Qwen3(0.6B) | 토큰 {} · 원본 KV {:.0} MB + 여유 {:.0} MB vs VRAM free {} → {}",
                planned_tokens,
                exact_bytes as f64 / 1e6,
                margin as f64 / 1e6,
                free.map(|f| format!("{:.0} MB", f as f64 / 1e6)).unwrap_or_else(|| "?".to_string()),
                if exact {
                    "원본 KV (압축하지 않음 · 정확도 우선)"
                } else {
                    "TurboQuant 압축 KV (메모리 우선)"
                }
            );
        }
        exact
    }

    /// 🌟 [KV-PLAN v2] 디코딩 진입 직전에 KV 배치 위치를 정합니다 (generate · generate_part 공용).
    ///
    ///  ── 무엇이 문제였나 ──
    ///   이 경로만 예전 서명(`plan_kv_residency`)을 써서 (1) 프리필로 이미 VRAM 에 만든 프롬프트 KV 를 free 에서 한 번 빼고
    ///   필요량으로 또 요구했고, (2) VRAM 여유가 고정 640MB 였고, (3) 원본 KV(BF16)도 FP8 1바이트로 셌습니다.
    ///   합성 평가(2026-11-03, RX 6600 ROCm)의 필드 추출 호출에서 KV 47MB 가 VRAM free 541~578MB 에서 RAM 으로 내려갔습니다.
    ///   RAM 계층은 디코딩 토큰마다 층마다 KV 전체를 GPU 로 올리고 다시 CPU 로 내립니다 (28층 × 왕복 2회 + 층마다 동기화).
    ///  ── 해결 ──
    ///   Qwen3.5 와 같은 v2 판정: 새로 생길 토큰만 실제 보관 크기로 사고, VRAM 여유 = max(96MB, 2 × 디코딩 순간치),
    ///   직전보다 좋은 계층으로 올라갈 때만 띠를 더 요구합니다. `LOGIS_KV_PLAN=legacy` 이면 예전 판정 그대로입니다.
    fn plan_decode_kv(&mut self, prompt_tokens: usize, sample_len: usize, exact: bool, label: &str) {
        use crate::utils::resources::{plan_kv_residency_ex, primary_gpu_id, KvPlanExtra, KvPlanInput};
        let (kv_layers, kv_heads, head_dim) = self.qwen3.kv_geometry();
        let elem = if self.device.is_cuda_or_rocm() { 2 } else { 4 };
        let plan = plan_kv_residency_ex(
            &KvPlanInput {
                gpu_id: primary_gpu_id(),
                is_cpu_mode: self.device.is_cpu(),
                num_kv_layers: kv_layers,
                num_kv_heads: kv_heads,
                head_dim,
                // 예전 판정(`LOGIS_KV_PLAN=legacy`)만 쓰는 값: RAM 대피본의 FP8 1바이트
                bytes_per_elem: 1,
                planned_tokens: prompt_tokens + sample_len,
                label,
            },
            &KvPlanExtra {
                // 프리필 KV 는 plan_kv_precision 이 VRAM 계층으로 되돌린 뒤 만들어 free 에서 이미 빠져 있습니다.
                already_tokens: prompt_tokens,
                decode_transient_bytes: QWEN3_DECODE_TRANSIENT_BYTES,
                bytes_per_token: Some(qwen3_kv_bytes_per_token(kv_layers, kv_heads, head_dim, exact, elem)),
                prev: self.last_kv_plan,
            },
        );
        self.last_kv_plan = Some(plan);
        self.qwen3.set_kv_residency(plan);
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
        
        let dir = cache_dir.unwrap_or_else(|| crate::utils::paths::get_kv_dir().to_string_lossy().into_owned());
        std::fs::create_dir_all(&dir)?;
        let path = std::path::Path::new(&dir).join(format!("{}.safetensors", session_id));
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub fn load_kv_cache(&mut self, session_id: &str, cache_dir: Option<String>) -> Result<bool> {
        let dir = cache_dir.unwrap_or_else(|| crate::utils::paths::get_kv_dir().to_string_lossy().into_owned());
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
        // 🌟 [Q8-RESIDENT] Q8 상주 경로는 프리필마다 가중치를 타일로 풀어 GEMM 하므로 청크를 1024 로 키웁니다 (기존 BF16 경로는 256).
        let chunk_size = if self.qwen3.is_q8_resident() { 1024 } else { 256 };
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
            if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }

            crate::models::common::legacy_working_set_trim();

            seqlen_offset += take;
        }

        if let Some(session_id) = save_session_id {
            self.save_kv_cache(&session_id, cache_dir)?;
        }
        crate::utils::trim_idle_gpu_pool(&self.device);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn select_token_from_hidden(
        &mut self,
        hidden: &Tensor,
        step_idx: usize,
        is_strict_json: bool,
        temperature: f64,
        penalty: f32,
        logit_processor: &mut candle_transformers::generation::LogitsProcessor,
        generate_hist: &[u32],
        penalized_history: &[u32],
        gen_text: &str,
        enter_id: u32,
        think_token_id: u32,
        open_bracket_id: u32,
        slash_id: u32,
        double_slash_id: u32,
        space_double_slash_id: u32,
        pretokenized_ignores: &[Vec<u32>],
        active_prej: Option<&crate::models::common::prejudice::CachedPrejudice>,
        full_prej: Option<&Tensor>,
        scratch: &mut crate::models::common::vocab_trim::DecodeStepScratch,
        guard_stats: &mut crate::models::common::vocab_trim::StepGuardStats,
    ) -> Result<u32> {
        scratch.banned_ids.clear();
        scratch.banned_ids.push(think_token_id);
        if is_strict_json {
            let is_url_single = gen_text.ends_with("http:/") || gen_text.ends_with("https:/");
            let is_url_double = gen_text.ends_with("http:") || gen_text.ends_with("https:");
            if !is_url_single && gen_text.ends_with('/') {
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
                    || gen_text.ends_with('"')
                    || gen_text.ends_with(':')
                    || gen_text.ends_with(": ");
                if apply_bias && is_strict_json && self.json_protected_ids.contains(&next_tok) {
                    apply_bias = false;
                }
                if apply_bias {
                    scratch.banned_ids.push(next_tok);
                }
            }
        }
        if step_idx == 0 {
            scratch.banned_ids.push(self.eos_token_id1);
            scratch.banned_ids.push(self.eos_token_id2);
            scratch.banned_ids.push(enter_id);
            if is_strict_json {
                if let Some(head) = self.qwen3.trim_head.as_ref() {
                    for &vid in head.kept.iter() {
                        if !self.json_starter_ids.contains(&vid) {
                            scratch.banned_ids.push(vid);
                        }
                    }
                }
            }
        }

        let recent_start = penalized_history.len().saturating_sub(512);
        let seen_tokens = &penalized_history[recent_start..];
        let fallback_eos = if is_strict_json && step_idx == 0 {
            open_bracket_id
        } else {
            self.eos_token_id1
        };

        if let Some(head) = self.qwen3.trim_head.as_ref() {
            let banned_ids = std::mem::take(&mut scratch.banned_ids);
            let pp = crate::models::common::vocab_trim::StepPostProcess {
                seen_tokens,
                repeat_penalty: penalty,
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
                } else if !is_strict_json && (cand == self.eos_token_id1 || cand == self.eos_token_id2) {
                    cand = enter_id;
                }
            }
            return Ok(cand);
        }

        if temperature <= 0.0
            && full_prej.is_none()
            && pretokenized_ignores.is_empty()
            && !(is_strict_json && step_idx == 0)
        {
            let (mut cand, st) = self.qwen3.greedy_from_hidden_with_stats(hidden)?;
            guard_stats.merge(&st);
            let banned_hit = scratch.banned_ids.contains(&cand)
                || (penalty > 1.0 && seen_tokens.contains(&cand));
            if !banned_hit {
                if step_idx == 0 && (cand == self.eos_token_id1 || cand == self.eos_token_id2) {
                    cand = enter_id;
                }
                return Ok(cand);
            }
        }

        let logits = self.qwen3.logits_from_hidden(hidden)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if let Some(prej) = full_prej {
            logits.broadcast_sub(prej)?
        } else {
            logits
        };
        let mut logits_vec = logits.to_vec1::<f32>()?;
        let len = logits_vec.len();
        if penalty > 1.0 && !seen_tokens.is_empty() {
            let mut dedup = std::collections::HashSet::new();
            for &t in seen_tokens {
                if dedup.insert(t) && (t as usize) < len {
                    let v = logits_vec[t as usize];
                    logits_vec[t as usize] = if v <= 0.0 { v * penalty } else { v / penalty };
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
            } else if !is_strict_json && (cand == self.eos_token_id1 || cand == self.eos_token_id2) {
                cand = enter_id;
            }
        }
        Ok(cand)
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
        let user_text = crate::models::common::vocab_trim::prompt_text(mes);
        if is_prefill {
            self.clear_kv_cache();
        } else {
            // 🌟 [KV-PLAN v2] 이어 쓰는 KV(세션 · 직전 캐시)와 남은 프리필도 VRAM 에 둡니다 (디코딩 진입 계획이 이미 상주로 셉니다).
            self.qwen3.set_kv_residency(KvResidency::Vram);
            if let Some(session_id) = load_session_id {
                self.load_kv_cache(&session_id, cache_dir)?;
            }
        }

        let temperature = mes.temperature.unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(top_k), seed);

        let mes_render = self.chat_template.apply_chat_template(mes)?;
        let (f_ids, spans) = self.encode_with_spans(&mes_render)?;
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&mes_render, &user_text, &f_ids, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, temperature > 1e-7)?;
        
        if start_len >= f_ids.len() {
            return Err(anyhow::anyhow!("start_len exceeds or equals prompt length"));
        }

        let p_ids = &f_ids[start_len..];
        let input_ids = Tensor::from_vec(p_ids.to_vec(), (1, p_ids.len()), &self.device)?;
        
        let prompt_seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = start_len;
        let mut generate = Vec::new();
        let mut penalized_history = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);

        // 🌟 [JSON Enforcement Prep] 프롬프트를 기반으로 JSON 모드 여부를 판단하고 특수 토큰을 준비합니다.
        let is_strict_json = mes_render.contains("/no_think") || mes_render.contains("RETURN JSON ONLY");
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(123);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let mut gen_text = String::new();
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        let active_prej_cached = if self.qwen3.trim_head.is_some() {
            semantic_prejudice.and_then(|t| self.prejudice_cached_active(t).ok())
        } else {
            None
        };
        if self.qwen3.trim_head.is_none() {
            if let Some(target_text) = semantic_prejudice {
                match self.prejudice_vector(target_text) {
                    Ok(prej) => semantic_prejudice_tensor = Some(prej),
                    Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                }
            }
        }
        // 🌟 [KV-PRECISION] 새로 시작하는 생성이면 VRAM 여유에 따라 KV 정밀도를 정합니다.
        let kv_exact = if is_prefill {
            self.plan_kv_precision(start_len + prompt_seq_len + sample_len as usize)
        } else {
            self.qwen3.set_kv_exact(false);
            false
        };

        let chunk_size = if self.qwen3.is_q8_resident() { 1024 } else { 256 };
        let mut first_hidden: Option<Tensor> = None;
        let mut current_chunk_offset = 0;
        let mut guard_stats = crate::models::common::vocab_trim::StepGuardStats::default();
        self.qwen3.validate_input_ids(p_ids)?;

        while current_chunk_offset < prompt_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(String::new());
                }
            }
            let take = std::cmp::min(chunk_size, prompt_seq_len - current_chunk_offset);
            let chunk_ids = input_ids.narrow(1, current_chunk_offset, take)?;
            let is_last_chunk = current_chunk_offset + take == prompt_seq_len;

            let last_hidden = self.qwen3.forward_last_hidden(Some(&chunk_ids), None, seqlen_offset)?;

            if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }
            crate::models::common::legacy_working_set_trim();

            seqlen_offset += take;
            current_chunk_offset += take;

            if is_last_chunk {
                first_hidden = Some(last_hidden);
            }
        }

        let pretokenized_ignores = pretokenize_ignore_list(&self.tokenizer, ignore_list, is_strict_json);
        let penalty = self.effective_repetition_penalty();
        let mut scratch = crate::models::common::vocab_trim::DecodeStepScratch::default();

        let fh = first_hidden.ok_or_else(|| anyhow::anyhow!("missing prefill hidden"))?;
        let mut next_token = self.select_token_from_hidden(
            &fh,
            0,
            is_strict_json,
            temperature,
            penalty,
            &mut logit_processor,
            &generate,
            &penalized_history,
            &gen_text,
            enter_id,
            think_token_id,
            open_bracket_id,
            slash_id,
            double_slash_id,
            space_double_slash_id,
            &pretokenized_ignores,
            active_prej_cached.as_ref(),
            semantic_prejudice_tensor.as_ref(),
            &mut scratch,
            &mut guard_stats,
        )?;
        generate.push(next_token);
        if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
            penalized_history.push(next_token);
        }
        if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) { gen_text.push_str(&piece); }

        // 🌟 [KV RESIDENCY PLAN] 디코딩 루프 진입 직전에 단 1회만 VRAM/RAM 여유를 계산합니다.
        self.plan_decode_kv(seqlen_offset, sample_len as usize, kv_exact, "Qwen3(0.6B) generate_part");

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
            let last_hidden = self.qwen3.forward_last_hidden(Some(&single_input), None, seqlen_offset)?;
            next_token = self.select_token_from_hidden(
                &last_hidden,
                i as usize,
                is_strict_json,
                temperature,
                penalty,
                &mut logit_processor,
                &generate,
                &penalized_history,
                &gen_text,
                enter_id,
                think_token_id,
                open_bracket_id,
                slash_id,
                double_slash_id,
                space_double_slash_id,
                &pretokenized_ignores,
                active_prej_cached.as_ref(),
                semantic_prejudice_tensor.as_ref(),
                &mut scratch,
                &mut guard_stats,
            )?;

            generate.push(next_token);
            if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
                penalized_history.push(next_token);
            }
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
            } else if is_strict_json && gen_text.trim_start().starts_with('[') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '[' { depth += 1; has_started = true; }
                    else if c == ']' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with(']') {
                    break;
                }
            }

            seqlen_offset += 1;

            // 🌟 메모리 최적화 (15토큰마다 OS 시스템 RAM 스파이크 억제 및 반환, 주기를 짧게 압박)
            if i > 0 && i % 15 == 0 {
                // 🌟 [VRAM 최적화 3] 디코딩 중 발생하는 KV Cache 병합(Cat) 찌꺼기 텐서들을 즉시 날려버립니다.
                if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }

                crate::models::common::legacy_working_set_trim();
            }
        }

        // BPE 디코딩 무결성을 보장하기 위해 최종 조립은 배열 기반으로 수행합니다.
        self.last_guard_stats = Some(guard_stats);
        let res_text = self.tokenizer.token_decode(generate)?;
        crate::utils::trim_idle_gpu_pool(&self.device);

        Ok(res_text)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, ignore_list: Option<&[String]>, semantic_prejudice: Option<&str>) -> Result<String> {
        let user_text = crate::models::common::vocab_trim::prompt_text(&mes);
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature as f64);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p as f64);
        let top_k = self.generation_config.top_k;
        
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(top_k), seed);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (f_ids, spans) = self.encode_with_spans(&mes_render)?;
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&mes_render, &user_text, &f_ids, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, temperature > 1e-7)?;

        let prompt_seq_len = f_ids.len();
        if prompt_seq_len == 0 { return Ok(String::new()); }
        self.qwen3.validate_input_ids(&f_ids)?;
        let input_ids = Tensor::from_vec(f_ids, (1, prompt_seq_len), &self.device)?;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let mut penalized_history = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);

        // 🌟 [JSON Enforcement Prep] 프롬프트를 기반으로 JSON 모드 여부를 판단하고 특수 토큰을 준비합니다.
        let is_strict_json = mes_render.contains("/no_think") || mes_render.contains("RETURN JSON ONLY");
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(123);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        
        let mut gen_text = String::new();
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        let active_prej_cached = if self.qwen3.trim_head.is_some() {
            semantic_prejudice.and_then(|t| self.prejudice_cached_active(t).ok())
        } else {
            None
        };
        if self.qwen3.trim_head.is_none() {
            if let Some(target_text) = semantic_prejudice {
                match self.prejudice_vector(target_text) {
                    Ok(prej) => semantic_prejudice_tensor = Some(prej),
                    Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                }
            }
        }
        let vocab_label = self
            .qwen3
            .trim_head
            .as_ref()
            .map(|h| format!("{}:{:016x}", h.label, h.kept_hash))
            .unwrap_or_default();
        let prompt_hash = hash_str_u64(&mes_render);
        let prefill_t0 = std::time::Instant::now();
        let mut retry_kv_hit = false;
        let mut first_hidden: Option<Tensor> = None;
        let mut kv_exact = false;

        if retry_kv_cache_enabled() {
            if let Some(snap) = self.prompt_kv_snapshot.as_ref() {
                if snap.prompt_hash == prompt_hash
                    && snap.vocab_label == vocab_label
                    && snap.seqlen == prompt_seq_len
                {
                    self.qwen3.clear_kv_cache();
                    self.qwen3.set_kv_residency(KvResidency::Vram);
                    self.qwen3
                        .set_planned_kv_tokens(prompt_seq_len + sample_len as usize);
                    self.qwen3.set_kv_exact(snap.kv_exact);
                    self.qwen3.set_kv_cache(snap.layer_kvs.clone());
                    seqlen_offset = snap.seqlen;
                    first_hidden = Some(snap.last_hidden.clone());
                    kv_exact = snap.kv_exact;
                    retry_kv_hit = true;
                }
            }
        }

        // 🌟 [Phase 1: Chunked Prefill]
        if !retry_kv_hit {
            // 🌟 [KV-PRECISION] VRAM 여유가 충분하면 이번 생성의 KV 를 압축하지 않습니다.
            kv_exact = self.plan_kv_precision(prompt_seq_len + sample_len as usize);
            let chunk_size = if self.qwen3.is_q8_resident() { 1024 } else { 256 };

            while seqlen_offset < prompt_seq_len {
                if let Some(flag) = &cancel_flag {
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        return Ok(String::new());
                    }
                }
                let take = std::cmp::min(chunk_size, prompt_seq_len - seqlen_offset);
                let chunk_ids = input_ids.narrow(1, seqlen_offset, take)?;

                let h = self.qwen3.forward_last_hidden(Some(&chunk_ids), None, seqlen_offset)?;

                if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }

                crate::models::common::legacy_working_set_trim();

                seqlen_offset += take;
                if seqlen_offset == prompt_seq_len {
                    first_hidden = Some(h);
                }
            }

            // 🌟 [SPD-03] 정확 VRAM 프리필 KV 를 1-슬롯 스냅샷으로 보존하여 동일 프롬프트 재시도 시 프리필을 건너뜁니다.
            if retry_kv_cache_enabled() && kv_exact && !self.device.is_cpu() {
                if let Some(ref fh) = first_hidden {
                    let layer_kvs = self.qwen3.get_kv_cache();
                    if !layer_kvs.is_empty() && layer_kvs.iter().all(|c| c.is_some()) {
                        self.prompt_kv_snapshot = Some(PromptKvSnapshot {
                            prompt_hash,
                            vocab_label,
                            seqlen: prompt_seq_len,
                            last_hidden: fh.clone(),
                            layer_kvs,
                            kv_exact,
                        });
                    }
                }
            }
        }

        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

        let pretokenized_ignores = pretokenize_ignore_list(&self.tokenizer, ignore_list, is_strict_json);
        let penalty = self.effective_repetition_penalty();
        let mut scratch = crate::models::common::vocab_trim::DecodeStepScratch::default();
        let mut guard_stats = crate::models::common::vocab_trim::StepGuardStats::default();

        let fh = first_hidden.ok_or_else(|| anyhow::anyhow!("missing prefill hidden"))?;
        let mut next_token = self.select_token_from_hidden(
            &fh,
            0,
            is_strict_json,
            temperature,
            penalty,
            &mut logit_processor,
            &generate,
            &penalized_history,
            &gen_text,
            enter_id,
            think_token_id,
            open_bracket_id,
            slash_id,
            double_slash_id,
            space_double_slash_id,
            &pretokenized_ignores,
            active_prej_cached.as_ref(),
            semantic_prejudice_tensor.as_ref(),
            &mut scratch,
            &mut guard_stats,
        )?;
        generate.push(next_token);
        if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
            penalized_history.push(next_token);
        }
        if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) { gen_text.push_str(&piece); }

        // 🌟 [KV RESIDENCY PLAN] 디코딩 루프 진입 직전에 단 1회만 VRAM/RAM 여유를 계산합니다.
        self.plan_decode_kv(seqlen_offset, sample_len as usize, kv_exact, "Qwen3(0.6B) generate");

        let decode_t0 = std::time::Instant::now();
        let mut decode_steps = 0usize;

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
            decode_steps += 1;

            let single_input = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let last_hidden = self.qwen3.forward_last_hidden(Some(&single_input), None, seqlen_offset)?;
            next_token = self.select_token_from_hidden(
                &last_hidden,
                i as usize,
                is_strict_json,
                temperature,
                penalty,
                &mut logit_processor,
                &generate,
                &penalized_history,
                &gen_text,
                enter_id,
                think_token_id,
                open_bracket_id,
                slash_id,
                double_slash_id,
                space_double_slash_id,
                &pretokenized_ignores,
                active_prej_cached.as_ref(),
                semantic_prejudice_tensor.as_ref(),
                &mut scratch,
                &mut guard_stats,
            )?;
            
            generate.push(next_token);
            if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
                penalized_history.push(next_token);
            }
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
            } else if is_strict_json && gen_text.trim_start().starts_with('[') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '[' { depth += 1; has_started = true; }
                    else if c == ']' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with(']') {
                    break;
                }
            }

            seqlen_offset += 1;

            // 🌟 메모리 최적화 (15토큰마다 OS 시스템 RAM 스파이크 억제 및 반환, 잦은 주기로 압박)
            if i > 0 && i % 15 == 0 {
                // 🌟 [VRAM 최적화 3] 디코딩 중 발생하는 KV Cache 병합(Cat) 찌꺼기 텐서들을 즉시 날려버립니다.
                if self.device.is_cuda_or_rocm() { let _ = self.device.synchronize(); }

                crate::models::common::legacy_working_set_trim();
            }
        }

        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;
        self.last_gen_stats = Some((prefill_ms, decode_ms, decode_steps, 0, retry_kv_hit));
        self.last_guard_stats = Some(guard_stats.clone());
        crate::utils::score_dynamics::record_baseline("qwen3.prefill_ms", prefill_ms as f32);
        crate::utils::score_dynamics::record_baseline("qwen3.decode_ms", decode_ms as f32);
        crate::utils::score_dynamics::record_baseline("qwen3.retry_kv_hit", if retry_kv_hit { 1.0 } else { 0.0 });
        crate::utils::score_dynamics::record_baseline("qwen3.slow_path_steps", 0.0);

        if logis_schema::reflection::ReflectionLogMode::from_env() != logis_schema::reflection::ReflectionLogMode::Off {
            let head = self.qwen3.trim_head.as_ref();
            let vocab_v = self.qwen3.embed_vocab().unwrap_or(0) as u32;
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
                    slot: "qwen3_0_6b".into(),
                    bytes: self.qwen3.resident_vram_bytes(),
                    sha256_16: String::new(),
                    recipe: if self.qwen3.is_q8_resident() { "Q8_0".into() } else { "BF16".into() },
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
                    tokens: [prompt_seq_len as u32, decode_steps as u32],
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
        self.qwen3.clear_kv_cache();
        crate::utils::trim_idle_gpu_pool(&self.device);
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        self.qwen3.clear_kv_cache();
        self.prompt_kv_snapshot = None;
    }

    /// 🌟 [Q8-RESIDENT] 정확도·속도 비교 테스트용: 내부 모델에 직접 접근합니다.
    pub fn model_mut(&mut self) -> &mut Qwen3Model {
        &mut self.qwen3
    }
}
