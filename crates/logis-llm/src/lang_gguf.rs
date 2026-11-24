use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;

pub const RUNTIME_DIR: &str = "runtime-q4km";
pub const RUNTIME_FILE: &str = "model-q4km.gguf";
const READY_FILE: &str = "READY";
pub const READY_JSON_FILE: &str = "READY.json";
const RECIPE: &str = "qwen35-q4km/2";
const LEGACY_RECIPE: &str = "qwen35-q4km/1";
pub const ALIGN: u64 = 32;
const ROUND_ELEMS: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReadyManifest {
    pub recipe: String,
    pub engine_fp: String,
    pub source_revision: String,
    pub gguf_sha256: String,
    pub gguf_bytes: u64,
    pub backbone_bytes: u64,
    pub embd_row_bytes: u64,
    pub vocab_size: usize,
    pub v_head_order: String,
    pub ascii_intact: bool,
}

pub const CHAT_TEMPLATE: &str = "{%- for message in messages -%}{{- '<|im_start|>' + message.role + '\\n' -}}{%- if message.content is string -%}{{- message.content -}}{%- else -%}{%- for part in message.content -%}{%- if part.text is defined -%}{{- part.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}";

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    Q4KM,
    Exact,
    /// 측정 기준: 양자화하는 텐서를 모두 Q8_0 으로 (앱은 쓰지 않습니다)
    Q8,
}

/// 🌟 [PYRAMID-QUANT] 블록 척도를 고르는 방법.
///
/// `Candle` 은 candle 의 `QTensor::quantize` (Q4_K · Q5_K 는 가중치 없는 반복 탐색 `make_qkx1_quants`, Q6_K 는
/// `make_qx_quants`, Q8_0 은 최댓값/127). `Wls` 는 `ggml-kquant` 의 중요도(`rms + |x|`) 가중 최소제곱 탐색과 F16 슈퍼 척도
/// 1 ULP 올림입니다. 둘 다 원본 BF16 에서 한 번 양자화합니다.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quantizer {
    Candle,
    Wls,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Store {
    F32,
    Q(GgmlDType),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Xform {
    Plain,
    NormPlusOne,
    NegExp,
    ConvSqueeze,
}

#[derive(Clone, Copy, Debug)]
enum VPerm {
    Rows { start: usize, head_dim: usize },
    Cols { head_dim: usize },
}

struct Item {
    gguf: String,
    src: String,
    src_shape: Vec<usize>,
    shape: Vec<usize>,
    store: Store,
    xform: Xform,
    vperm: Option<VPerm>,
}

pub struct StEntry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    begin: u64,
    end: u64,
}

pub struct StFile {
    data_start: u64,
    pub entries: HashMap<String, StEntry>,
    /// 파일 전체의 읽기 전용 매핑. 텐서마다 파일을 다시 열어 힙에 복사하지 않습니다.
    map: memmap2::Mmap,
}

impl StFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path).with_context(|| format!("{:?} 를 열 수 없습니다", path))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        if n > 100_000_000 {
            bail!("{:?} 의 safetensors 헤더 길이 {} 가 비정상입니다", path, n);
        }
        let mut head = vec![0u8; n as usize];
        f.read_exact(&mut head)?;
        let v: Value = serde_json::from_slice(&head).map(logis_schema::json_order::sorted).context("safetensors 헤더 JSON 을 읽을 수 없습니다")?;
        let mut entries = HashMap::new();
        for (name, e) in v.as_object().ok_or_else(|| anyhow!("safetensors 헤더가 객체가 아닙니다"))? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match e.get("dtype").and_then(|d| d.as_str()) {
                Some("BF16") => DType::BF16,
                Some("F16") => DType::F16,
                Some("F32") => DType::F32,
                other => bail!("{} 의 dtype {:?} 는 지원하지 않습니다", name, other),
            };
            let shape: Vec<usize> = e
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_default();
            let offs: Vec<u64> = e
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            if offs.len() != 2 || offs[1] < offs[0] {
                bail!("{} 의 data_offsets 가 올바르지 않습니다", name);
            }
            entries.insert(name.clone(), StEntry { dtype, shape, begin: offs[0], end: offs[1] });
        }
        f.seek(SeekFrom::Start(0))?;
        let map = unsafe { memmap2::Mmap::map(&f) }.with_context(|| format!("{:?} 를 매핑할 수 없습니다", path))?;
        Ok(Self { data_start: 8 + n, entries, map })
    }

    /// 텐서의 저장 바이트. 매핑을 그대로 빌려 줍니다.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entries.get(name).ok_or_else(|| anyhow!("model.safetensors 에 {} 가 없습니다", name))?;
        let (start, end) = ((self.data_start + e.begin) as usize, (self.data_start + e.end) as usize);
        let raw = self.map.get(start..end).ok_or_else(|| {
            anyhow!(
                "model.safetensors 에서 {} 를 읽지 못했습니다. 파일이 잘렸거나 손상된 것으로 보이며, 모델 폴더를 지우면 다음 실행에서 다시 받습니다",
                name
            )
        })?;
        let expect = e.shape.iter().product::<usize>() * e.dtype.size_in_bytes();
        if raw.len() != expect {
            bail!("{} 의 바이트 수 {} 가 형상 {:?} 와 맞지 않습니다", name, raw.len(), e.shape);
        }
        Ok(raw)
    }

    /// 파일 처음부터 텐서 바이트가 시작하는 위치
    pub fn offset(&self, name: &str) -> Result<usize> {
        let e = self.entries.get(name).ok_or_else(|| anyhow!("model.safetensors 에 {} 가 없습니다", name))?;
        Ok((self.data_start + e.begin) as usize)
    }

    fn load(&self, name: &str) -> Result<Tensor> {
        let raw = self.bytes(name)?;
        let e = &self.entries[name];
        Ok(Tensor::from_raw_buffer(raw, e.dtype, &e.shape, &Device::Cpu)?)
    }
}

/// 🌟 [SHARDED] 한 파일(`model.safetensors`) 또는 색인(`model.safetensors.index.json`)으로 나뉜 조각 파일들.
/// 원본 Qwen/Qwen3.5-4B 는 두 조각입니다.
pub struct StSet {
    files: Vec<StFile>,
    which: HashMap<String, usize>,
}

impl StSet {
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files = Vec::new();
        let mut which = HashMap::new();
        for (i, p) in weight_files(dir)?.iter().enumerate() {
            let f = StFile::open(p)?;
            for k in f.entries.keys() {
                if which.insert(k.clone(), i).is_some() {
                    bail!("{} 가 두 조각 파일에 함께 있습니다", k);
                }
            }
            files.push(f);
        }
        Ok(Self { files, which })
    }

    pub fn names(&self) -> HashSet<String> {
        self.which.keys().cloned().collect()
    }

    fn load(&self, name: &str) -> Result<Tensor> {
        let i = *self.which.get(name).ok_or_else(|| anyhow!("가중치 파일에 {} 가 없습니다", name))?;
        self.files[i].load(name)
    }
}

/// 가중치 파일: `model.safetensors` 가 있으면 그것, 없으면 색인이 가리키는 조각들 (이름 순)
pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    let v = read_json(&index).with_context(|| format!("{:?} 에 model.safetensors 도 색인도 없습니다", dir))?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| anyhow!("{:?} 에 weight_map 이 없습니다", index))?;
    let mut names: Vec<String> = logis_schema::json_order::sorted_values(map).filter_map(|x| x.as_str()).map(str::to_string).collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        bail!("{:?} 의 weight_map 이 비었습니다", index);
    }
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        // 받은 색인은 믿지 않습니다: 폴더 밖을 가리키는 이름은 거부
        if n.contains('/') || n.contains('\\') || n.contains("..") || !n.ends_with(".safetensors") {
            bail!("색인의 조각 이름 {:?} 를 쓸 수 없습니다", n);
        }
        out.push(dir.join(n));
    }
    Ok(out)
}

struct TextCfg {
    vocab: usize,
    hidden: usize,
    inter: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    interval: usize,
    eps: f32,
    rope_theta: f32,
    rope_dim: usize,
    mrope: Vec<i32>,
    conv_kernel: usize,
    k_heads: usize,
    v_heads: usize,
    lin_head: usize,
    max_pos: usize,
    tied: bool,
}

pub enum Meta {
    U32(u32),
    F32(f32),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
}

pub fn runtime_dir(model_dir: &Path) -> PathBuf {
    model_dir.join(RUNTIME_DIR)
}

pub fn runtime_gguf(model_dir: &Path) -> PathBuf {
    runtime_dir(model_dir).join(RUNTIME_FILE)
}

pub fn runtime_bytes(model_dir: &Path) -> u64 {
    std::fs::metadata(runtime_gguf(model_dir)).map(|m| m.len()).unwrap_or(0)
}

pub fn v_head_order() -> Result<&'static str> {
    let t = Tensor::new(&[0f32, 1.0], &Device::Cpu)?.reshape((1, 1, 2, 1))?;
    let r = crate::utils::tensor_utils::repeat_interleave(&t, 2, 2)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if r == [0.0, 0.0, 1.0, 1.0] {
        Ok("grouped")
    } else if r == [0.0, 1.0, 0.0, 1.0] {
        Ok("tiled")
    } else {
        bail!("repeat_interleave 결과 {:?} 를 해석할 수 없습니다", r)
    }
}

fn recipe_key(model_dir: &Path, precision: Precision) -> Result<String> {
    let mut src = 0u64;
    for p in weight_files(model_dir)? {
        src += std::fs::metadata(&p).with_context(|| format!("{:?} 가 없습니다", p))?.len();
    }
    Ok(format!(
        "{}|{:?}|vhead={}|src={}",
        RECIPE,
        precision,
        v_head_order()?,
        src
    ))
}

pub fn sha256_bytes(data: &[u8]) -> String {
    premium_discount_quant::vocab_plan::sha256_hex(data)
}

pub fn read_ready_manifest(model_dir: &Path) -> Option<ReadyManifest> {
    let raw = std::fs::read_to_string(runtime_dir(model_dir).join(READY_JSON_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn write_ready_manifest(model_dir: &Path, manifest: &ReadyManifest) -> Result<()> {
    let dir = runtime_dir(model_dir);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(READY_JSON_FILE), serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}

pub fn runtime_ready(model_dir: &Path) -> bool {
    let gguf_len = runtime_bytes(model_dir);
    if gguf_len == 0 {
        return false;
    }
    let expected_vhead = match v_head_order() {
        Ok(v) => v,
        Err(_) => return false,
    };
    // 1. Prefer READY.json (survives original .safetensors shard deletion)
    if let Some(m) = read_ready_manifest(model_dir) {
        return m.gguf_bytes == gguf_len
            && m.v_head_order == expected_vhead
            && (m.recipe == RECIPE || m.recipe == LEGACY_RECIPE);
    }
    // 2. Fallback to legacy READY text file when shards are still present
    let ready = std::fs::read_to_string(runtime_dir(model_dir).join(READY_FILE)).unwrap_or_default();
    let trimmed = ready.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Ok(key) = recipe_key(model_dir, Precision::Q4KM) {
        if trimmed == key {
            return true;
        }
        let legacy_key = key.replacen(RECIPE, LEGACY_RECIPE, 1);
        if trimmed == legacy_key {
            return true;
        }
    }
    false
}

static CONVERT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn prepare(model_dir: &Path, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<PathBuf> {
    // 🌟 [QWEN4-EXP] qwen4_exp 체크포인트는 qwen35 GGUF 로 변환하면 그래프가 달라집니다 (hc/PLE/QSA/MoE).
    if crate::models::qwen4_exp::registry::is_qwen4_exp_dir(model_dir) {
        bail!(
            "[LANG-GGUF] {:?} 는 Qwen4-Exp(qwen4_exp) 체크포인트입니다. qwen35 Q4_K_M 변환은 지원하지 않으며, models::qwen4_exp::TextGenerator::load_dir 로 safetensors 를 직접 읽어야 합니다.",
            model_dir
        );
    }
    if runtime_ready(model_dir) {
        return Ok(runtime_gguf(model_dir));
    }
    let _guard = CONVERT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if runtime_ready(model_dir) {
        return Ok(runtime_gguf(model_dir));
    }
    let out = runtime_gguf(model_dir);
    if let Err(e) = convert_with(model_dir, &out, Precision::Q4KM, progress) {
        let _ = std::fs::remove_file(out.with_extension("gguf.part"));
        return Err(e);
    }
    let _ = std::fs::write(
        runtime_dir(model_dir).join(READY_FILE),
        recipe_key(model_dir, Precision::Q4KM)?,
    );
    if let Ok(cfg) = read_text_cfg(model_dir) {
        let gguf_bytes = runtime_bytes(model_dir);
        let embd_row_bytes = (cfg.hidden as u64 / 32) * 34;
        let embd_bytes = cfg.vocab as u64 * embd_row_bytes;
        let backbone_bytes = gguf_bytes.saturating_sub(embd_bytes);
        let sha = premium_discount_quant::vocab_plan::sha256_hex(
            &format!("{gguf_bytes}:{}:{}", cfg.vocab, v_head_order().unwrap_or("grouped")).into_bytes(),
        );
        let manifest = ReadyManifest {
            recipe: RECIPE.into(),
            engine_fp: format!("q35_4b-v{}k-q4km", cfg.vocab / 1000),
            source_revision: "local".into(),
            gguf_sha256: sha,
            gguf_bytes,
            backbone_bytes,
            embd_row_bytes,
            vocab_size: cfg.vocab,
            v_head_order: v_head_order()?.into(),
            ascii_intact: true,
        };
        let _ = write_ready_manifest(model_dir, &manifest);
    }
    Ok(out)
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| anyhow!("경로를 UTF-8 로 읽을 수 없습니다: {:?}", p))
}

pub fn load_runtime(model_dir: &Path, device: &Device) -> Result<Qwen3_5GenerateModel> {
    let gguf = prepare(model_dir, &|_, _| {})?;
    load_from(model_dir, &gguf, device)
}

pub fn load_from(model_dir: &Path, gguf: &Path, device: &Device) -> Result<Qwen3_5GenerateModel> {
    let rdir = gguf
        .parent()
        .ok_or_else(|| anyhow!("{:?} 의 상위 폴더를 알 수 없습니다", gguf))?;
    install_chat_template(rdir)?;
    let mut gen = Qwen3_5GenerateModel::init_from_gguf(path_str(gguf)?, None, Some(device))?;
    gen.tokenizer = crate::tokenizer::TokenizerModel::init(path_str(model_dir)?)?;
    // Sync vision special token IDs from tokenizer.json so remapped/trimmed artifacts never use stale literals.
    let single_tok_id = |tok: &crate::tokenizer::TokenizerModel, s: &str| -> Option<u32> {
        tok.text_encode_vec(s.to_string(), false)
            .ok()
            .filter(|v| v.len() == 1)
            .map(|v| v[0])
    };
    if let Some(id) = single_tok_id(&gen.tokenizer, "<|vision_start|>") {
        gen.qwen3_5.vision_start_token_id = id;
    }
    if let Some(id) = single_tok_id(&gen.tokenizer, "<|vision_end|>") {
        gen.qwen3_5.vision_end_token_id = id;
    }
    if let Some(id) = single_tok_id(&gen.tokenizer, "<|image_pad|>") {
        gen.qwen3_5.image_token_id = id;
    }
    if let Some(id) = single_tok_id(&gen.tokenizer, "<|video_pad|>") {
        gen.qwen3_5.video_token_id = id;
    }
    let probe = gen
        .tokenizer
        .text_encode_vec("<|im_end|>".to_string(), false)
        .unwrap_or_default();
    if probe != vec![gen.eos_token_id] {
        bail!(
            "[LANG-GGUF] tokenizer.json 이 '<|im_end|>' 를 {:?} 로 나누어 GGUF 종료 토큰 {} 과 일치하지 않습니다.",
            probe,
            gen.eos_token_id
        );
    }
    let vocab_rows = gen.qwen3_5.embed_vocab().unwrap_or(0);
    if vocab_rows > 0 && (gen.eos_token_id as usize) >= vocab_rows {
        bail!(
            "[LANG-GGUF] 종료 토큰 {} 이 임베딩 행 수 {} 범위 밖입니다.",
            gen.eos_token_id,
            vocab_rows
        );
    }
    Ok(gen)
}

fn install_chat_template(rdir: &Path) -> Result<()> {
    std::fs::create_dir_all(rdir)?;
    std::fs::write(
        rdir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&logis_schema::json_order::Sorted(&serde_json::json!({
            "chat_template": CHAT_TEMPLATE,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        })))?,
    )?;
    std::fs::write(rdir.join("chat_template.jinja"), CHAT_TEMPLATE)?;
    let probe_text = "ALPHAEDGE-TEMPLATE-PROBE";
    let params = crate::openai_types::ChatCompletionParameters {
        messages: vec![
            crate::openai_types::ChatCompletionRequestMessage::System(
                crate::openai_types::ChatCompletionRequestSystemMessage {
                    content: "system".to_string(),
                    name: None,
                },
            ),
            crate::openai_types::ChatCompletionRequestMessage::User(
                crate::openai_types::ChatCompletionRequestUserMessage {
                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(
                        probe_text.to_string(),
                    ),
                    name: None,
                },
            ),
        ],
        model: "qwen3.5".to_string(),
        ..Default::default()
    };
    let rendered = crate::chat_template::ChatTemplate::init(path_str(rdir)?)?
        .apply_chat_template(&params)?;
    if !rendered.contains(probe_text) {
        bail!("채팅 템플릿이 사용자 메시지를 담지 못했습니다: {:?}", rendered);
    }
    if !rendered.contains("<|im_start|>assistant") {
        println!(
            "[LANG-GGUF] ⚠️ 채팅 템플릿 렌더링 결과에 '<|im_start|>assistant' 가 없습니다: {:?}",
            rendered
        );
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("{:?} 를 읽을 수 없습니다", path))?;
    serde_json::from_slice(&bytes).with_context(|| format!("{:?} 가 올바른 JSON 이 아닙니다", path))
}

fn cfg_usize(t: &Value, key: &str) -> Result<usize> {
    t.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("config.json text_config.{} 가 없습니다", key))
}

fn cfg_f64(t: &Value, key: &str) -> Result<f64> {
    t.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow!("config.json text_config.{} 가 없습니다", key))
}

fn read_text_cfg(model_dir: &Path) -> Result<TextCfg> {
    let root = read_json(&model_dir.join("config.json"))?;
    let t = root.get("text_config").unwrap_or(&root);
    let rope = t
        .get("rope_parameters")
        .ok_or_else(|| anyhow!("config.json text_config.rope_parameters 가 없습니다"))?;
    let head_dim = cfg_usize(t, "head_dim")?;
    let partial = rope.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let mut mrope: Vec<i32> = rope
        .get("mrope_section")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).map(|x| x as i32).collect())
        .unwrap_or_else(|| vec![11, 11, 10]);
    while mrope.len() < 4 {
        mrope.push(0);
    }
    let k_head = cfg_usize(t, "linear_key_head_dim")?;
    let v_head = cfg_usize(t, "linear_value_head_dim")?;
    if k_head != v_head {
        bail!(
            "linear_key_head_dim({}) 와 linear_value_head_dim({}) 이 달라 현재 Qwen3.5 런타임(head_v_dim = ssm.state_size)으로 표현할 수 없습니다",
            k_head,
            v_head
        );
    }
    if t.get("attention_bias").and_then(|v| v.as_bool()).unwrap_or(false) {
        bail!("attention_bias=true 는 현재 Qwen3.5 런타임이 읽지 않습니다");
    }
    if t.get("attn_output_gate").and_then(|v| v.as_bool()) == Some(false) {
        bail!("attn_output_gate=false 는 현재 Qwen3.5 런타임(q_proj 게이트 절반 가정)과 맞지 않습니다");
    }
    if t
        .get("mlp_only_layers")
        .and_then(|v| v.as_array())
        .map_or(false, |a| !a.is_empty())
    {
        bail!("mlp_only_layers 가 비어 있지 않아 지원하지 않습니다");
    }
    let layers = cfg_usize(t, "num_hidden_layers")?;
    let interval = cfg_usize(t, "full_attention_interval")?;
    if interval == 0 {
        bail!("full_attention_interval 이 0 입니다");
    }
    if let Some(types) = t.get("layer_types").and_then(|v| v.as_array()) {
        if types.len() != layers {
            bail!("layer_types 길이 {} 가 num_hidden_layers {} 와 다릅니다", types.len(), layers);
        }
        for (i, ty) in types.iter().enumerate() {
            let want = if (i + 1) % interval == 0 { "full_attention" } else { "linear_attention" };
            if ty.as_str() != Some(want) {
                bail!(
                    "layer_types[{}]={:?} 가 런타임 규칙((i+1) % {} == 0 → full_attention)과 다릅니다",
                    i,
                    ty,
                    interval
                );
            }
        }
    }
    let k_heads = cfg_usize(t, "linear_num_key_heads")?;
    let v_heads = cfg_usize(t, "linear_num_value_heads")?;
    if k_heads == 0 || v_heads % k_heads != 0 {
        bail!("linear_num_value_heads({}) 가 linear_num_key_heads({}) 의 배수가 아닙니다", v_heads, k_heads);
    }
    let heads = cfg_usize(t, "num_attention_heads")?;
    let kv_heads = cfg_usize(t, "num_key_value_heads")?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        bail!("num_attention_heads({}) 가 num_key_value_heads({}) 의 배수가 아닙니다", heads, kv_heads);
    }
    let tied = t
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| root.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(false);
    Ok(TextCfg {
        vocab: cfg_usize(t, "vocab_size")?,
        hidden: cfg_usize(t, "hidden_size")?,
        inter: cfg_usize(t, "intermediate_size")?,
        layers,
        heads,
        kv_heads,
        head_dim,
        interval,
        eps: cfg_f64(t, "rms_norm_eps")? as f32,
        rope_theta: rope
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("rope_parameters.rope_theta 가 없습니다"))? as f32,
        rope_dim: (head_dim as f64 * partial) as usize,
        mrope,
        conv_kernel: cfg_usize(t, "linear_conv_kernel_dim")?,
        k_heads,
        v_heads,
        lin_head: k_head,
        max_pos: cfg_usize(t, "max_position_embeddings")?,
        tied,
    })
}

fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i >= n / 8 && (i - n / 8) % 3 == 2)
}

fn fit(store: Store, cols: usize) -> Store {
    match store {
        Store::Q(q) if cols % q.block_size() != 0 => {
            if cols % GgmlDType::Q8_0.block_size() == 0 {
                Store::Q(GgmlDType::Q8_0)
            } else {
                Store::F32
            }
        }
        s => s,
    }
}

fn find_prefix(names: &HashSet<String>) -> Result<String> {
    let mut hits: Vec<&String> = names
        .iter()
        .filter(|n| {
            n.ends_with("embed_tokens.weight")
                && !n.contains("visual")
                && !n.starts_with("mtp")
                && !n.contains(".mtp.")
        })
        .collect();
    hits.sort();
    match hits.first() {
        Some(n) => Ok(n.trim_end_matches("embed_tokens.weight").to_string()),
        None => bail!("model.safetensors 에서 embed_tokens.weight 를 찾지 못했습니다"),
    }
}

fn plan(cfg: &TextCfg, names: &HashSet<String>, precision: Precision, tiled: bool) -> Result<Vec<Item>> {
    let p = find_prefix(names)?;
    let h = cfg.hidden;
    let key_dim = cfg.k_heads * cfg.lin_head;
    let value_dim = cfg.v_heads * cfg.lin_head;
    let conv_dim = key_dim * 2 + value_dim;
    let q = |d: GgmlDType| -> Store {
        match precision {
            Precision::Q4KM => Store::Q(d),
            Precision::Exact => Store::F32,
            Precision::Q8 => Store::Q(GgmlDType::Q8_0),
        }
    };
    let mut items: Vec<Item> = Vec::new();
    let mut add = |gguf: String, src: String, src_shape: Vec<usize>, shape: Vec<usize>, store: Store, xform: Xform, vperm: Option<VPerm>| {
        let cols = *shape.last().unwrap_or(&1);
        let store = if shape.len() >= 2 { fit(store, cols) } else { Store::F32 };
        items.push(Item { gguf, src, src_shape, shape, store, xform, vperm });
    };
    add(
        "token_embd.weight".into(),
        format!("{p}embed_tokens.weight"),
        vec![cfg.vocab, h],
        vec![cfg.vocab, h],
        q(GgmlDType::Q8_0),
        Xform::Plain,
        None,
    );
    for i in 0..cfg.layers {
        let l = format!("{p}layers.{i}.");
        let b = format!("blk.{i}.");
        let more = use_more_bits(i, cfg.layers);
        add(format!("{b}attn_norm.weight"), format!("{l}input_layernorm.weight"), vec![h], vec![h], Store::F32, Xform::NormPlusOne, None);
        add(format!("{b}post_attention_norm.weight"), format!("{l}post_attention_layernorm.weight"), vec![h], vec![h], Store::F32, Xform::NormPlusOne, None);
        add(format!("{b}ffn_gate.weight"), format!("{l}mlp.gate_proj.weight"), vec![cfg.inter, h], vec![cfg.inter, h], q(GgmlDType::Q4K), Xform::Plain, None);
        add(format!("{b}ffn_up.weight"), format!("{l}mlp.up_proj.weight"), vec![cfg.inter, h], vec![cfg.inter, h], q(GgmlDType::Q4K), Xform::Plain, None);
        add(
            format!("{b}ffn_down.weight"),
            format!("{l}mlp.down_proj.weight"),
            vec![h, cfg.inter],
            vec![h, cfg.inter],
            q(if more { GgmlDType::Q6K } else { GgmlDType::Q4K }),
            Xform::Plain,
            None,
        );
        if (i + 1) % cfg.interval == 0 {
            let qd = cfg.heads * cfg.head_dim;
            let kvd = cfg.kv_heads * cfg.head_dim;
            add(format!("{b}attn_q.weight"), format!("{l}self_attn.q_proj.weight"), vec![qd * 2, h], vec![qd * 2, h], q(GgmlDType::Q4K), Xform::Plain, None);
            add(format!("{b}attn_k.weight"), format!("{l}self_attn.k_proj.weight"), vec![kvd, h], vec![kvd, h], q(GgmlDType::Q4K), Xform::Plain, None);
            add(
                format!("{b}attn_v.weight"),
                format!("{l}self_attn.v_proj.weight"),
                vec![kvd, h],
                vec![kvd, h],
                q(if more { GgmlDType::Q6K } else { GgmlDType::Q4K }),
                Xform::Plain,
                None,
            );
            add(format!("{b}attn_output.weight"), format!("{l}self_attn.o_proj.weight"), vec![h, qd], vec![h, qd], q(GgmlDType::Q4K), Xform::Plain, None);
            add(format!("{b}attn_q_norm.weight"), format!("{l}self_attn.q_norm.weight"), vec![cfg.head_dim], vec![cfg.head_dim], Store::F32, Xform::NormPlusOne, None);
            add(format!("{b}attn_k_norm.weight"), format!("{l}self_attn.k_norm.weight"), vec![cfg.head_dim], vec![cfg.head_dim], Store::F32, Xform::NormPlusOne, None);
        } else {
            let vr = |start: usize, head_dim: usize| if tiled { Some(VPerm::Rows { start, head_dim }) } else { None };
            add(
                format!("{b}attn_qkv.weight"),
                format!("{l}linear_attn.in_proj_qkv.weight"),
                vec![conv_dim, h],
                vec![conv_dim, h],
                q(GgmlDType::Q5K),
                Xform::Plain,
                vr(key_dim * 2, cfg.lin_head),
            );
            add(format!("{b}attn_gate.weight"), format!("{l}linear_attn.in_proj_z.weight"), vec![value_dim, h], vec![value_dim, h], q(GgmlDType::Q4K), Xform::Plain, vr(0, cfg.lin_head));
            add(format!("{b}ssm_beta.weight"), format!("{l}linear_attn.in_proj_b.weight"), vec![cfg.v_heads, h], vec![cfg.v_heads, h], q(GgmlDType::Q8_0), Xform::Plain, vr(0, 1));
            add(format!("{b}ssm_alpha.weight"), format!("{l}linear_attn.in_proj_a.weight"), vec![cfg.v_heads, h], vec![cfg.v_heads, h], q(GgmlDType::Q8_0), Xform::Plain, vr(0, 1));
            add(
                format!("{b}ssm_out.weight"),
                format!("{l}linear_attn.out_proj.weight"),
                vec![h, value_dim],
                vec![h, value_dim],
                q(GgmlDType::Q4K),
                Xform::Plain,
                if tiled { Some(VPerm::Cols { head_dim: cfg.lin_head }) } else { None },
            );
            add(
                format!("{b}ssm_conv1d.weight"),
                format!("{l}linear_attn.conv1d.weight"),
                vec![conv_dim, 1, cfg.conv_kernel],
                vec![conv_dim, cfg.conv_kernel],
                Store::F32,
                Xform::ConvSqueeze,
                vr(key_dim * 2, cfg.lin_head),
            );
            add(format!("{b}ssm_a"), format!("{l}linear_attn.A_log"), vec![cfg.v_heads], vec![cfg.v_heads], Store::F32, Xform::NegExp, vr(0, 1));
            add(format!("{b}ssm_dt.bias"), format!("{l}linear_attn.dt_bias"), vec![cfg.v_heads], vec![cfg.v_heads], Store::F32, Xform::Plain, vr(0, 1));
            add(format!("{b}ssm_norm.weight"), format!("{l}linear_attn.norm.weight"), vec![cfg.lin_head], vec![cfg.lin_head], Store::F32, Xform::Plain, None);
        }
    }
    add("output_norm.weight".into(), format!("{p}norm.weight"), vec![h], vec![h], Store::F32, Xform::NormPlusOne, None);
    if !cfg.tied {
        add("output.weight".into(), "lm_head.weight".into(), vec![cfg.vocab, h], vec![cfg.vocab, h], q(GgmlDType::Q6K), Xform::Plain, None);
    }
    for it in items.iter() {
        if !names.contains(&it.src) {
            bail!("model.safetensors 에 {} 가 없습니다", it.src);
        }
    }
    let used: HashSet<&str> = items.iter().map(|i| i.src.as_str()).collect();
    let layer_prefix = format!("{p}layers.");
    let stray: Vec<&String> = names
        .iter()
        .filter(|n| n.starts_with(layer_prefix.as_str()) && !used.contains(n.as_str()))
        .collect();
    if !stray.is_empty() {
        bail!("언어 모델 영역에 변환 규칙이 없는 텐서가 있습니다: {:?}", stray.iter().take(8).collect::<Vec<_>>());
    }
    Ok(items)
}

fn store_bytes(it: &Item) -> u64 {
    let n: usize = it.shape.iter().product();
    match it.store {
        Store::F32 => (n * 4) as u64,
        Store::Q(d) => (n / d.block_size() * d.type_size()) as u64,
    }
}

fn ggml_type_id(store: Store) -> Result<u32> {
    Ok(match store {
        Store::F32 => 0,
        Store::Q(GgmlDType::F16) => 1,
        Store::Q(GgmlDType::Q8_0) => 8,
        Store::Q(GgmlDType::Q4K) => 12,
        Store::Q(GgmlDType::Q5K) => 13,
        Store::Q(GgmlDType::Q6K) => 14,
        Store::Q(other) => bail!("{:?} 는 이 변환기가 쓰지 않는 형식입니다", other),
    })
}

fn v_perm_indices(len: usize, start: usize, count: usize, k_heads: usize, v_per_k: usize, head_dim: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..len as u32).collect();
    for v in 0..v_per_k {
        for g in 0..k_heads {
            for d in 0..head_dim {
                let new_pos = (v * k_heads + g) * head_dim + d;
                let old_pos = (g * v_per_k + v) * head_dim + d;
                if new_pos < count {
                    idx[start + new_pos] = (start + old_pos) as u32;
                }
            }
        }
    }
    idx
}

fn read_tokenizer_meta(model_dir: &Path, vocab: usize) -> Result<(Vec<String>, Vec<i32>, u32, u32)> {
    let tj = read_json(&model_dir.join("tokenizer.json"))?;
    let mut tokens: Vec<Option<String>> = vec![None; vocab];
    let mut types: Vec<i32> = vec![1; vocab];
    let mut by_content: HashMap<String, u32> = HashMap::new();
    if let Some(v) = tj.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
        for (tok, id) in logis_schema::json_order::sorted_iter(v) {
            if let Some(id) = id.as_u64() {
                let id = id as usize;
                if id >= vocab {
                    bail!("tokenizer.json 토큰 id {} 가 vocab_size {} 를 넘습니다", id, vocab);
                }
                tokens[id] = Some(tok.clone());
                by_content.insert(tok.clone(), id as u32);
            }
        }
    }
    if let Some(added) = tj.get("added_tokens").and_then(|v| v.as_array()) {
        for a in added.iter() {
            let id = a.get("id").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            let content = a.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if id >= vocab || content.is_empty() {
                continue;
            }
            let special = a.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
            tokens[id] = Some(content.clone());
            types[id] = if special { 3 } else { 4 };
            by_content.insert(content, id as u32);
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| match t {
            Some(t) if seen.insert(t.clone()) => t,
            _ => {
                types[i] = 5;
                format!("<|unused_{}|>", i)
            }
        })
        .collect();
    let tc = read_json(&model_dir.join("tokenizer_config.json")).unwrap_or(Value::Null);
    let eos_name = match tc.get("eos_token") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("content").and_then(|v| v.as_str()).map(|s| s.to_string()),
        _ => None,
    };
    let eos = eos_name
        .and_then(|n| by_content.get(&n).copied())
        .or_else(|| by_content.get("<|im_end|>").copied())
        .ok_or_else(|| anyhow!("종료 토큰(<|im_end|>) id 를 tokenizer 에서 찾지 못했습니다"))?;
    let pad = by_content.get("<|endoftext|>").copied().unwrap_or(eos);
    Ok((tokens, types, eos, pad))
}

pub fn put_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

pub fn put_meta<W: Write>(w: &mut W, key: &str, v: &Meta) -> Result<()> {
    put_str(w, key)?;
    match v {
        Meta::U32(x) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        Meta::F32(x) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        Meta::Str(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            put_str(w, s)?;
        }
        Meta::ArrStr(a) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&8u32.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            for s in a.iter() {
                put_str(w, s)?;
            }
        }
        Meta::ArrI32(a) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            for x in a.iter() {
                w.write_all(&x.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

pub fn pad_to<W: Write>(w: &mut W, written: u64) -> Result<u64> {
    let pad = (ALIGN - written % ALIGN) % ALIGN;
    if pad > 0 {
        w.write_all(&vec![0u8; pad as usize])?;
    }
    Ok(written + pad)
}

fn load_src(st: &StSet, it: &Item) -> Result<Tensor> {
    let t = st.load(&it.src)?;
    if t.dims() != it.src_shape.as_slice() {
        bail!("{} 의 형상 {:?} 가 config 기준 {:?} 와 다릅니다", it.src, t.dims(), it.src_shape);
    }
    let t = match it.xform {
        Xform::ConvSqueeze => t.reshape(it.shape.as_slice())?,
        _ => t,
    };
    Ok(t)
}

pub fn quantize_rows(t: &Tensor, dtype: GgmlDType, threads: usize) -> Result<Vec<u8>> {
    quantize_rows_with(t, dtype, threads, Quantizer::Candle)
}

fn kquant_format(dtype: GgmlDType) -> Option<ggml_kquant::Format> {
    Some(match dtype {
        GgmlDType::Q8_0 => ggml_kquant::Format::Q8_0,
        GgmlDType::Q6K => ggml_kquant::Format::Q6K,
        GgmlDType::Q5K => ggml_kquant::Format::Q5K,
        GgmlDType::Q4K => ggml_kquant::Format::Q4K,
        _ => return None,
    })
}

pub fn quantize_rows_with(t: &Tensor, dtype: GgmlDType, threads: usize, quantizer: Quantizer) -> Result<Vec<u8>> {
    if let (Quantizer::Wls, Some(fmt)) = (quantizer, kquant_format(dtype)) {
        let cols = t.dim(1)?;
        let data = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        return ggml_kquant::encode(&data, cols, fmt, None).map_err(|e| anyhow!("{e}"));
    }
    let rows = t.dim(0)?;
    let per = rows.div_ceil(threads.max(1)).max(1);
    let chunks: Vec<(usize, usize)> = (0..rows)
        .step_by(per)
        .map(|s| (s, per.min(rows - s)))
        .collect();
    let parts: Vec<Result<Vec<u8>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|&(s, n)| {
                scope.spawn(move || -> Result<Vec<u8>> {
                    let cols = t.dim(1)?;
                    let data = t.narrow(0, s, n)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let part = Tensor::from_vec(data, (n, cols), &Device::Cpu)?;
                    let q = QTensor::quantize(&part, dtype)?;
                    Ok(q.data()?.into_owned())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| anyhow!("양자화 스레드가 비정상 종료했습니다"))?)
            .collect()
    });
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(&p?);
    }
    Ok(out)
}

fn produce(st: &StSet, it: &Item, cfg: &TextCfg, threads: usize, quantizer: Quantizer, sink: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let src = load_src(st, it)?;
    let v_per_k = cfg.v_heads / cfg.k_heads;
    let two_d = if src.rank() == 1 { src.reshape((src.dim(0)?, 1))? } else { src };
    let rows = two_d.dim(0)?;
    let cols = two_d.dim(1)?;
    let row_idx = match it.vperm {
        Some(VPerm::Rows { start, head_dim }) => {
            let count = cfg.v_heads * head_dim;
            Some(Tensor::from_vec(v_perm_indices(rows, start, count, cfg.k_heads, v_per_k, head_dim), rows, &Device::Cpu)?)
        }
        _ => None,
    };
    let col_idx = match it.vperm {
        Some(VPerm::Cols { head_dim }) => {
            Some(Tensor::from_vec(v_perm_indices(cols, 0, cols, cfg.k_heads, v_per_k, head_dim), cols, &Device::Cpu)?)
        }
        _ => None,
    };
    let round_rows = (ROUND_ELEMS / cols.max(1)).max(1);
    let mut r0 = 0usize;
    while r0 < rows {
        let n = round_rows.min(rows - r0);
        let mut blk = match &row_idx {
            Some(idx) => two_d.index_select(&idx.narrow(0, r0, n)?, 0)?,
            None => two_d.narrow(0, r0, n)?,
        };
        if let Some(ci) = &col_idx {
            blk = blk.index_select(ci, 1)?;
        }
        let blk = match it.xform {
            Xform::NormPlusOne => blk.to_dtype(DType::F32)?.affine(1.0, 1.0)?,
            Xform::NegExp => blk.to_dtype(DType::F32)?.exp()?.neg()?,
            _ => blk,
        };
        match it.store {
            Store::F32 => {
                let v = blk.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                sink(&bytes)?;
            }
            Store::Q(d) => {
                let bytes = quantize_rows_with(&blk, d, threads, quantizer)?;
                sink(&bytes)?;
            }
        }
        r0 += n;
    }
    Ok(())
}

pub fn convert_with(model_dir: &Path, out: &Path, precision: Precision, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<()> {
    convert_variant(model_dir, out, precision, QUANTIZER, progress)
}

/// 앱이 쓰는 블록 척도 탐색.
///
/// 가중 최소제곱은 바꾸지 않았습니다. Qwen3.5-4B-kor, 같은 원본의 Q8_0 기준, 한국어 무역 · 주문 문장 640 토큰
/// (tests/it/p2b_models.rs `p2b_qwen35_4b_quantizer`, RX 6600):
///   candle   ΔNLL +0.032 [+0.005, +0.060] · KLD 0.0455 · top-1 89.6 %
///   wls      ΔNLL +0.040 [+0.013, +0.067] · KLD 0.0398 · top-1 89.3 %
/// KLD 는 13 % 줄지만 NLL 은 나아지지 않아(구간 겹침) 레시피를 그대로 둡니다 — 바꾸면 언어마다 4B 를 다시 변환해야 합니다.
/// 같은 측정에서 Q8_0 은 디코드 72 ms/토큰으로 Q4_K_M(87 ms)보다 빠르지만 3.8 GB 대 2.2 GB 입니다.
pub const QUANTIZER: Quantizer = Quantizer::Candle;

/// 정밀도 · 척도 탐색을 골라 `out` 에 변환합니다 (측정용). `out` 이 모델의 런타임 폴더 밖이면 앱의 준비 표식은 건드리지 않습니다.
pub fn convert_variant(model_dir: &Path, out: &Path, precision: Precision, quantizer: Quantizer, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<()> {
    let cfg = read_text_cfg(model_dir)?;
    let st = StSet::open_dir(model_dir)?;
    let names: HashSet<String> = st.names();
    let tiled = v_head_order()? == "tiled";
    let items = plan(&cfg, &names, precision, tiled)?;
    let (tokens, token_types, eos, pad) = read_tokenizer_meta(model_dir, cfg.vocab)?;
    let meta: Vec<(&str, Meta)> = vec![
        ("general.architecture", Meta::Str("qwen35".into())),
        ("general.name", Meta::Str(model_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default())),
        ("general.alignment", Meta::U32(ALIGN as u32)),
        ("general.dtype", Meta::U32(0)),
        ("qwen35.block_count", Meta::U32(cfg.layers as u32)),
        ("qwen35.context_length", Meta::U32(cfg.max_pos as u32)),
        ("qwen35.embedding_length", Meta::U32(cfg.hidden as u32)),
        ("qwen35.feed_forward_length", Meta::U32(cfg.inter as u32)),
        ("qwen35.attention.head_count", Meta::U32(cfg.heads as u32)),
        ("qwen35.attention.head_count_kv", Meta::U32(cfg.kv_heads as u32)),
        ("qwen35.attention.key_length", Meta::U32(cfg.head_dim as u32)),
        ("qwen35.attention.value_length", Meta::U32(cfg.head_dim as u32)),
        ("qwen35.attention.layer_norm_rms_epsilon", Meta::F32(cfg.eps)),
        ("qwen35.rope.freq_base", Meta::F32(cfg.rope_theta)),
        ("qwen35.rope.dimension_count", Meta::U32(cfg.rope_dim as u32)),
        ("qwen35.rope.dimension_sections", Meta::ArrI32(cfg.mrope.clone())),
        ("qwen35.full_attention_interval", Meta::U32(cfg.interval as u32)),
        ("qwen35.ssm.conv_kernel", Meta::U32(cfg.conv_kernel as u32)),
        ("qwen35.ssm.state_size", Meta::U32(cfg.lin_head as u32)),
        ("qwen35.ssm.group_count", Meta::U32(cfg.k_heads as u32)),
        ("qwen35.ssm.time_step_rank", Meta::U32(cfg.v_heads as u32)),
        ("qwen35.ssm.inner_size", Meta::U32((cfg.v_heads * cfg.lin_head) as u32)),
        ("tokenizer.ggml.model", Meta::Str("gpt2".into())),
        ("tokenizer.ggml.tokens", Meta::ArrStr(tokens)),
        ("tokenizer.ggml.token_type", Meta::ArrI32(token_types)),
        ("tokenizer.ggml.merges", Meta::ArrStr(Vec::new())),
        ("tokenizer.ggml.eos_token_id", Meta::U32(eos)),
        ("tokenizer.ggml.padding_token_id", Meta::U32(pad)),
        ("tokenizer.chat_template", Meta::Str(CHAT_TEMPLATE.into())),
        ("alphaedge.recipe", Meta::Str(RECIPE.into())),
        ("alphaedge.v_head_order", Meta::Str(if tiled { "tiled" } else { "grouped" }.into())),
        ("alphaedge.precision", Meta::Str(format!("{:?}", precision))),
        ("logis.recipe", Meta::Str(RECIPE.into())),
        ("logis.v_head_order", Meta::Str(if tiled { "tiled" } else { "grouped" }.into())),
        ("logis.quantizer", Meta::Str(format!("{:?}", quantizer))),
    ];
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if out.starts_with(runtime_dir(model_dir)) {
        let _ = std::fs::remove_file(runtime_dir(model_dir).join(READY_FILE));
    }
    let part = out.with_extension("gguf.part");
    let file = std::fs::File::create(&part).with_context(|| format!("{:?} 를 만들 수 없습니다", part))?;
    let mut w = BufWriter::with_capacity(8 << 20, file);
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(items.len() as u64).to_le_bytes());
    head.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in meta.iter() {
        put_meta(&mut head, k, v)?;
    }
    let mut offset = 0u64;
    let mut sizes: Vec<u64> = Vec::with_capacity(items.len());
    for it in items.iter() {
        put_str(&mut head, &it.gguf)?;
        head.extend_from_slice(&(it.shape.len() as u32).to_le_bytes());
        for d in it.shape.iter().rev() {
            head.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        head.extend_from_slice(&ggml_type_id(it.store)?.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        let size = store_bytes(it);
        sizes.push(size);
        offset = (offset + size).div_ceil(ALIGN) * ALIGN;
    }
    w.write_all(&head)?;
    pad_to(&mut w, head.len() as u64)?;
    let total: u64 = offset;
    let threads = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(1)
        .min(8);
    let mut done = 0u64;
    progress(0, total);
    for (it, size) in items.iter().zip(sizes.iter()) {
        let mut wrote = 0u64;
        produce(&st, it, &cfg, threads, quantizer, &mut |bytes: &[u8]| {
            w.write_all(bytes)?;
            wrote += bytes.len() as u64;
            Ok(())
        })?;
        if wrote != *size {
            bail!("{} 가 {} 바이트로 기록되어 예상 {} 와 다릅니다", it.gguf, wrote, size);
        }
        let padded = pad_to(&mut w, wrote)?;
        done += padded;
        progress(done, total);
    }
    let file = w.into_inner().map_err(|e| anyhow!("{}", e))?;
    file.sync_all()?;
    drop(file);
    if out.exists() {
        std::fs::remove_file(out)?;
    }
    std::fs::rename(&part, out)?;
    Ok(())
}