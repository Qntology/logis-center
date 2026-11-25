//! 🌟 [PYRAMID-QUANT] SigLIP2 양자화 계획의 타워 GGUF 를 원본 F32 safetensors 에서 한 번 만듭니다.
//!
//!   runtime-<계획 꼬리표>/mmproj-siglip2.gguf   비전 타워
//!   runtime-<계획 꼬리표>/siglip2-text.gguf     텍스트 타워 (토큰 표 제외 — 원본에서 행만 읽습니다)
//!
//!  ── 지문 ──
//!   READY 에 `siglip2-quant/1|<계획 이름>-<배정 해시>|src=<원본 바이트 수>` 를 씁니다. 계획(묶음 · 형식)이나 원본이 바뀌면
//!   폴더 이름과 지문이 달라져 다시 만들고, 같으면 매핑만 엽니다. 꼬리표는 `quant::weights_tag` 와 같아 구 캐시 · 레이아웃
//!   캐시 · SDS 비전 스코프의 꼬리표와 한 줄로 이어집니다.
//!
//!  ── 저장 ──
//!   · 계획이 양자화하는 가중치: 그 블록 형식 (ggml-kquant, 가중 최소제곱 척도). 나머지는 F32 (1계층).
//!   · MapHead 의 in_proj 는 q · k · v 세 텐서(`in_proj_{q,k,v}.weight/.bias`)로 나눠 둡니다.
//!   · 블록이 나누지 못하는 입력 폭은 0 으로 채웁니다. fc2 의 입력 4304 는 fc1 의 출력이라, fc1 에 0 행(바이어스 0)을 붙여
//!     fc1 이 채운 폭을 바로 내게 합니다 — gelu(0) = 0 이라 값이 같고 실행 중 채우기 복사가 없습니다.
//!     그 밖의 폭(K 형식의 1152)은 실행 중 활성을 0 으로 채웁니다 (`quant::SLinear`).

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::DType;

use super::gguf::TOKEN_TABLE;
use super::quant::{QType, WeightPlan};
use logis_llm::lang_gguf::{pad_to, put_meta, put_str, Meta, StFile, ALIGN};

const RECIPE: &str = "siglip2-quant/1";
const VISION_FILE: &str = "mmproj-siglip2.gguf";
const TEXT_FILE: &str = "siglip2-text.gguf";
const READY_FILE: &str = "READY";

fn plan_dir(model_dir: &Path, plan: &WeightPlan) -> PathBuf {
    model_dir.join(format!("runtime-{}", plan_tag(plan)))
}

/// 폴더 · 지문에 쓰는 꼬리표 (`q8-1a2b3c4d`)
pub fn plan_tag(plan: &WeightPlan) -> String {
    format!("{}-{:08x}", plan.tag, plan.hash() as u32)
}

pub fn quant_vision_path(model_dir: &Path, plan: &WeightPlan) -> PathBuf {
    plan_dir(model_dir, plan).join(VISION_FILE)
}

pub fn quant_text_path(model_dir: &Path, plan: &WeightPlan) -> PathBuf {
    plan_dir(model_dir, plan).join(TEXT_FILE)
}

fn recipe_key(model_dir: &Path, plan: &WeightPlan) -> Result<String> {
    let src = std::fs::metadata(model_dir.join("model.safetensors"))
        .with_context(|| format!("{:?} 에 model.safetensors 가 없습니다", model_dir))?
        .len();
    Ok(format!("{}|{}|src={}", RECIPE, plan_tag(plan), src))
}

pub fn quant_ready(model_dir: &Path, plan: &WeightPlan) -> bool {
    let Ok(key) = recipe_key(model_dir, plan) else { return false };
    let mark = std::fs::read_to_string(plan_dir(model_dir, plan).join(READY_FILE)).unwrap_or_default();
    mark.trim() == key && quant_vision_path(model_dir, plan).exists() && quant_text_path(model_dir, plan).exists()
}

static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// 계획의 타워 GGUF 를 만듭니다. 이미 지문이 맞으면 아무것도 하지 않습니다.
pub fn prepare_quant(model_dir: &Path, plan: &WeightPlan) -> Result<()> {
    if !plan.is_quantized() {
        bail!("양자화하지 않는 계획 '{}' 은 GGUF 를 만들지 않습니다", plan.tag);
    }
    if quant_ready(model_dir, plan) {
        return Ok(());
    }
    let _guard = BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if quant_ready(model_dir, plan) {
        return Ok(());
    }
    super::gguf::close_handles(Some(model_dir));
    let t0 = std::time::Instant::now();
    let dir = plan_dir(model_dir, plan);
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(dir.join(READY_FILE));
    let st = StFile::open(&model_dir.join("model.safetensors"))?;
    let (vision_mb, vision_items) = write_tower(&st, plan, "vision_model.", &quant_vision_path(model_dir, plan))?;
    let (text_mb, text_items) = write_tower(&st, plan, "text_model.", &quant_text_path(model_dir, plan))?;
    std::fs::write(dir.join(READY_FILE), recipe_key(model_dir, plan)?)?;
    println!(
        "[SigLIP2-QUANT] 계획 '{}' 타워 GGUF 를 원본에서 만들었습니다: 비전 {:.0} MB ({}개) · 텍스트 {:.0} MB ({}개, 토큰 표 제외) · {:.1} s",
        plan_tag(plan),
        vision_mb,
        vision_items,
        text_mb,
        text_items,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// 파일에 쓸 텐서 하나
struct Item {
    name: String,
    src: String,
    /// 원본에서 가져올 행 범위 (in_proj 를 나눌 때)
    rows_from: usize,
    rows: usize,
    cols: usize,
    /// 저장 형상 (0 으로 채운 뒤)
    rows_out: usize,
    cols_out: usize,
    q: QType,
}

impl Item {
    fn bytes(&self) -> u64 {
        let n = self.rows_out * self.cols_out;
        match self.q {
            QType::F32 => (n * 4) as u64,
            q => (n / q.format().block_size() * q.format().type_size()) as u64,
        }
    }

    fn type_id(&self) -> u32 {
        match self.q {
            QType::F32 => 0,
            QType::Q8_0 => 8,
            QType::Q4K => 12,
            QType::Q5K => 13,
            QType::Q6K => 14,
        }
    }
}

/// 타워 텐서 목록과 저장 형상을 정합니다.
fn items_for(st: &StFile, plan: &WeightPlan, prefix: &str) -> Result<Vec<Item>> {
    let mut names: Vec<&String> = st
        .entries
        .keys()
        .filter(|n| (n.starts_with(prefix) || n.as_str() == "logit_scale" || n.as_str() == "logit_bias") && n.as_str() != TOKEN_TABLE)
        .collect();
    names.sort();
    let shape = |n: &str| -> Result<(usize, usize)> {
        let e = st.entries.get(n).ok_or_else(|| anyhow!("model.safetensors 에 {} 가 없습니다", n))?;
        Ok(match e.shape.as_slice() {
            [r, c] => (*r, *c),
            [r] => (*r, 1),
            [] => (1, 1),
            other => (other[0], other[1..].iter().product()),
        })
    };
    // fc1 의 출력(= fc2 의 입력)을 채울 폭: fc2 가 양자화되면 그 블록의 배수
    let mlp_width = |mlp: &str| -> Result<(usize, usize)> {
        let (_, inter) = shape(&format!("{mlp}.fc2.weight"))?;
        let q2 = plan.qtype(&format!("{mlp}.fc2.weight"));
        Ok((inter, if q2 == QType::F32 { inter } else { inter.next_multiple_of(q2.format().block_size()) }))
    };
    let mut out = Vec::new();
    for name in names {
        let (rows, cols) = shape(name)?;
        let item = |name: String, src: &str, rows_from: usize, rows: usize, rows_out: usize, cols_out: usize, q: QType| Item {
            name,
            src: src.to_string(),
            rows_from,
            rows,
            cols,
            rows_out,
            cols_out,
            q,
        };
        if let Some(head) = name.strip_suffix(".in_proj_weight") {
            // MapHead q · k · v 를 나눕니다.
            let q = plan.qtype(name);
            let d = rows / 3;
            let cols_out = if q == QType::F32 { cols } else { cols.next_multiple_of(q.format().block_size()) };
            for (s, tag) in ["q", "k", "v"].iter().enumerate() {
                out.push(item(format!("{head}.in_proj_{tag}.weight"), name, s * d, d, d, cols_out, q));
            }
            continue;
        }
        if let Some(head) = name.strip_suffix(".in_proj_bias") {
            let d = rows / 3;
            for (s, tag) in ["q", "k", "v"].iter().enumerate() {
                out.push(item(format!("{head}.in_proj_{tag}.bias"), name, s * d, d, d, 1, QType::F32));
            }
            continue;
        }
        let q = if st.entries[name].shape.len() == 2 { plan.qtype(name) } else { QType::F32 };
        let (mut rows_out, mut cols_out) = (rows, cols);
        if let Some(mlp) = name.strip_suffix(".fc1.weight").or_else(|| name.strip_suffix(".fc1.bias")) {
            let (inter, padded) = mlp_width(mlp)?;
            if rows == inter {
                rows_out = padded;
            }
        }
        if let Some(mlp) = name.strip_suffix(".fc2.weight") {
            cols_out = mlp_width(mlp)?.1;
        }
        if q != QType::F32 {
            cols_out = cols_out.next_multiple_of(q.format().block_size());
        }
        out.push(item(name.clone(), name, 0, rows, rows_out, cols_out, q));
    }
    Ok(out)
}

/// 원본 값을 F32 로 읽어 행 범위를 자르고 0 으로 채웁니다.
fn source_values(st: &StFile, it: &Item) -> Result<Vec<f32>> {
    let e = &st.entries[&it.src];
    let raw = st.bytes(&it.src)?;
    let all: Vec<f32> = match e.dtype {
        DType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        DType::BF16 => raw.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        DType::F16 => raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        other => bail!("{} 의 dtype {:?} 는 지원하지 않습니다", it.src, other),
    };
    let mut out = vec![0f32; it.rows_out * it.cols_out];
    for r in 0..it.rows {
        let s = (it.rows_from + r) * it.cols;
        out[r * it.cols_out..r * it.cols_out + it.cols].copy_from_slice(&all[s..s + it.cols]);
    }
    Ok(out)
}

fn write_tower(st: &StFile, plan: &WeightPlan, prefix: &str, out: &Path) -> Result<(f64, usize)> {
    let items = items_for(st, plan, prefix)?;
    let meta: Vec<(&str, Meta)> = vec![
        ("general.architecture", Meta::Str("siglip2".into())),
        ("general.name", Meta::Str(format!("siglip2-{}{}", prefix.trim_end_matches("_model."), plan_tag(plan)))),
        ("general.alignment", Meta::U32(ALIGN as u32)),
        ("logis.recipe", Meta::Str(RECIPE.into())),
        ("logis.plan", Meta::Str(plan_tag(plan))),
    ];
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(items.len() as u64).to_le_bytes());
    head.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in meta.iter() {
        put_meta(&mut head, k, v)?;
    }
    let mut offset = 0u64;
    for it in &items {
        put_str(&mut head, &it.name)?;
        // 1차원(바이어스 · 정규화 · 스칼라)은 1차원, 3차원 이상(probe)은 원본 형상 그대로 둡니다.
        let dims: Vec<u64> = match st.entries[&it.src].shape.len() {
            0 | 1 => vec![it.rows_out as u64],
            2 => vec![it.rows_out as u64, it.cols_out as u64],
            _ => st.entries[&it.src].shape.iter().map(|d| *d as u64).collect(),
        };
        head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims.iter().rev() {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&it.type_id().to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        offset = (offset + it.bytes()).div_ceil(ALIGN) * ALIGN;
    }
    let part = out.with_extension("gguf.part");
    let result = (|| -> Result<()> {
        let file = std::fs::File::create(&part).with_context(|| format!("{:?} 를 만들 수 없습니다", part))?;
        let mut w = BufWriter::with_capacity(8 << 20, file);
        w.write_all(&head)?;
        pad_to(&mut w, head.len() as u64)?;
        for it in &items {
            let values = source_values(st, it)?;
            let bytes = match it.q {
                QType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                q => premium_discount_quant::kquant::encode(&values, it.cols_out, q.format(), None).map_err(|e| anyhow!("{}: {e}", it.name))?,
            };
            if bytes.len() as u64 != it.bytes() {
                bail!("{} 의 바이트 수 {} 가 예상 {} 와 다릅니다", it.name, bytes.len(), it.bytes());
            }
            w.write_all(&bytes)?;
            pad_to(&mut w, bytes.len() as u64)?;
        }
        let file = w.into_inner().map_err(|e| anyhow!("{}", e))?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    if out.exists() {
        std::fs::remove_file(out)?;
    }
    std::fs::rename(&part, out)?;
    Ok((offset as f64 / 1e6, items.len()))
}
