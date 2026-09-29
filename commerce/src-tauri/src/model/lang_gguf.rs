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
const RECIPE: &str = "qwen35-q4km/1";
const ALIGN: u64 = 32;
const ROUND_ELEMS: usize = 16 * 1024 * 1024;

pub const CHAT_TEMPLATE: &str = "{%- for message in messages -%}{{- '<|im_start|>' + message.role + '\\n' -}}{%- if message.content is string -%}{{- message.content -}}{%- else -%}{%- for part in message.content -%}{%- if part.text is defined -%}{{- part.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}";

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Precision {
    Q4KM,
    Exact,
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

struct StEntry {
    dtype: DType,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

struct StFile {
    path: PathBuf,
    data_start: u64,
    entries: HashMap<String, StEntry>,
}

impl StFile {
    fn open(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path).with_context(|| format!("{:?} 를 열 수 없습니다", path))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        if n > 100_000_000 {
            bail!("{:?} 의 safetensors 헤더 길이 {} 가 비정상입니다", path, n);
        }
        let mut head = vec![0u8; n as usize];
        f.read_exact(&mut head)?;
        let v: Value = serde_json::from_slice(&head).context("safetensors 헤더 JSON 을 읽을 수 없습니다")?;
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
        Ok(Self { path: path.to_path_buf(), data_start: 8 + n, entries })
    }

    fn load(&self, name: &str) -> Result<Tensor> {
        let e = self.entries.get(name).ok_or_else(|| anyhow!("model.safetensors 에 {} 가 없습니다", name))?;
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_start + e.begin))?;
        let mut buf = vec![0u8; (e.end - e.begin) as usize];
        f.read_exact(&mut buf).with_context(|| {
            format!(
                "model.safetensors 에서 {} 를 읽지 못했습니다. 파일이 잘렸거나 손상된 것으로 보이며, 모델 폴더를 지우면 다음 실행에서 다시 받습니다",
                name
            )
        })?;
        let expect = e.shape.iter().product::<usize>() * e.dtype.size_in_bytes();
        if buf.len() != expect {
            bail!("{} 의 바이트 수 {} 가 형상 {:?} 와 맞지 않습니다", name, buf.len(), e.shape);
        }
        Ok(Tensor::from_raw_buffer(&buf, e.dtype, &e.shape, &Device::Cpu)?)
    }
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

enum Meta {
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
    let src = std::fs::metadata(model_dir.join("model.safetensors"))
        .with_context(|| format!("{:?} 에 model.safetensors 가 없습니다", model_dir))?
        .len();
    Ok(format!(
        "{}|{:?}|vhead={}|src={}",
        RECIPE,
        precision,
        v_head_order()?,
        src
    ))
}

pub fn runtime_ready(model_dir: &Path) -> bool {
    let key = match recipe_key(model_dir, Precision::Q4KM) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let ready = std::fs::read_to_string(runtime_dir(model_dir).join(READY_FILE)).unwrap_or_default();
    ready.trim() == key && runtime_bytes(model_dir) > 0
}

static CONVERT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn prepare(model_dir: &Path, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<PathBuf> {
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
    std::fs::write(
        runtime_dir(model_dir).join(READY_FILE),
        recipe_key(model_dir, Precision::Q4KM)?,
    )?;
    Ok(out)
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| anyhow!("경로를 UTF-8 로 읽을 수 없습니다: {:?}", p))
}

pub fn load_runtime(model_dir: &Path, device: &Device) -> Result<Qwen3_5GenerateModel> {
    let gguf = prepare(model_dir, &|_, _| {})?;
    load_from(model_dir, &gguf, device)
}

pub(crate) fn load_from(model_dir: &Path, gguf: &Path, device: &Device) -> Result<Qwen3_5GenerateModel> {
    let rdir = gguf
        .parent()
        .ok_or_else(|| anyhow!("{:?} 의 상위 폴더를 알 수 없습니다", gguf))?;
    install_chat_template(rdir)?;
    let mut gen = Qwen3_5GenerateModel::init_from_gguf(path_str(gguf)?, None, Some(device))?;
    gen.tokenizer = crate::tokenizer::TokenizerModel::init(path_str(model_dir)?)?;
    let probe = gen
        .tokenizer
        .text_encode_vec("<|im_end|>".to_string(), false)
        .unwrap_or_default();
    if probe != vec![gen.eos_token_id] {
        println!(
            "[LANG-GGUF] ⚠️ tokenizer.json 이 '<|im_end|>' 를 {:?} 로 나눕니다. 종료 토큰 {} 과 달라 생성이 max_tokens 까지 이어질 수 있습니다.",
            probe, gen.eos_token_id
        );
    }
    Ok(gen)
}

fn install_chat_template(rdir: &Path) -> Result<()> {
    std::fs::create_dir_all(rdir)?;
    std::fs::write(
        rdir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "chat_template": CHAT_TEMPLATE,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        }))?,
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
        for (tok, id) in v.iter() {
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

fn put_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn put_meta<W: Write>(w: &mut W, key: &str, v: &Meta) -> Result<()> {
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

fn pad_to<W: Write>(w: &mut W, written: u64) -> Result<u64> {
    let pad = (ALIGN - written % ALIGN) % ALIGN;
    if pad > 0 {
        w.write_all(&vec![0u8; pad as usize])?;
    }
    Ok(written + pad)
}

fn load_src(st: &StFile, it: &Item) -> Result<Tensor> {
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

fn produce(st: &StFile, it: &Item, cfg: &TextCfg, threads: usize, sink: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
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
                let bytes = quantize_rows(&blk, d, threads)?;
                sink(&bytes)?;
            }
        }
        r0 += n;
    }
    Ok(())
}

pub(crate) fn convert_with(model_dir: &Path, out: &Path, precision: Precision, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<()> {
    let cfg = read_text_cfg(model_dir)?;
    let st = StFile::open(&model_dir.join("model.safetensors"))?;
    let names: HashSet<String> = st.entries.keys().cloned().collect();
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
    ];
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(runtime_dir(model_dir).join(READY_FILE));
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
        produce(&st, it, &cfg, threads, &mut |bytes: &[u8]| {
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