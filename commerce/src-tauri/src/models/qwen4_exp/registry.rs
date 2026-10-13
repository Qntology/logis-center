//! 텍스트 생성기 아키텍처 판별 + 로드 분기 (qwen4-spec Q4-01).
//!
//! 지금까지 앱은 아키텍처를 판별하지 않고 Qwen3.5 로만 읽었습니다. 여기서
//!  - HF 디렉터리: config.json 의 `model_type` (`qwen4_exp` / `qwen4_exp_text`, 또는 `text_config.model_type`) /
//!    `architectures` (`Qwen4ExpFor*`) 로 Qwen4-Exp 를 알아보고,
//!  - GGUF: `general.architecture == "qwen4exp"` 를 알아봅니다 (메타데이터 규격이 아직 없으므로 명확한 미지원 오류).
//! [`TextGenerator`] 는 Qwen3.5 / Qwen4-Exp 생성기를 같은 `generate(..)` 호출로 감쌉니다.
//!
//! 앱 연결: `lang_llm` 의 런타임 로더(`register_loader`) 로 [`load_lang_runtime`] 이 등록됩니다. lifecycle 의
//! `ensure_qwen3_5_lang` 이 이 로더로 읽은 결과를 아키텍처에 맞는 슬롯(`qwen3_5_generator` / `qwen4_exp_generator`) 에
//! 두고, `generate_lang_json` · 음차 생성이 상주한 쪽으로 호출합니다.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device};

use super::generate::Qwen4ExpGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::openai_types::ChatCompletionParameters;

/// 판별된 텍스트 아키텍처.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextArch {
    Qwen35,
    Qwen4Exp,
    Unknown(String),
}

fn arch_of_value(v: &serde_json::Value) -> TextArch {
    let mt = |x: &serde_json::Value| x.get("model_type").and_then(|m| m.as_str()).map(|s| s.to_string());
    let mut types: Vec<String> = Vec::new();
    if let Some(t) = mt(v) {
        types.push(t);
    }
    if let Some(t) = v.get("text_config").and_then(mt) {
        types.push(t);
    }
    if let Some(a) = v.get("architectures").and_then(|a| a.as_array()) {
        types.extend(a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())));
    }
    if types.iter().any(|t| t == "qwen4_exp" || t == "qwen4_exp_text" || t.starts_with("Qwen4ExpFor")) {
        return TextArch::Qwen4Exp;
    }
    let q35: Vec<&String> = types.iter().filter(|t| t.starts_with("qwen3_5") || t.starts_with("Qwen3_5")).collect();
    // qwen3_5_moe(_text) / Qwen3_5MoeFor* 는 Qwen3.5 밀집 그래프(와 GGUF 변환기) 로 읽을 수 없습니다.
    if let Some(moe) = q35.iter().find(|t| t.to_ascii_lowercase().contains("moe")) {
        return TextArch::Unknown((*moe).clone());
    }
    if !q35.is_empty() {
        return TextArch::Qwen35;
    }
    TextArch::Unknown(types.first().cloned().unwrap_or_default())
}

/// HF 모델 디렉터리의 텍스트 아키텍처 (config.json 이 없거나 읽을 수 없으면 `Unknown("")`).
pub fn detect_text_arch(model_dir: &Path) -> TextArch {
    match std::fs::read(model_dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    {
        Some(v) => arch_of_value(&v),
        None => TextArch::Unknown(String::new()),
    }
}

pub fn is_qwen4_exp_dir(model_dir: &Path) -> bool {
    detect_text_arch(model_dir) == TextArch::Qwen4Exp
}

/// GGUF `general.architecture` 값.
pub fn gguf_arch(content: &gguf_file::Content) -> Option<String> {
    content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
}

/// Qwen3.5 GGUF 로더 앞단 검사: Qwen4-Exp GGUF 를 Qwen3.5 그래프로 잘못 읽지 않도록 막습니다.
pub fn ensure_not_qwen4exp_gguf(content: &gguf_file::Content) -> Result<()> {
    if gguf_arch(content).as_deref() == Some("qwen4exp") {
        bail!(
            "[QWEN4-EXP] GGUF arch 'qwen4exp' 는 아직 지원하지 않습니다 (공식 GGUF 메타데이터 규격 없음). \
             HF safetensors 디렉터리를 models::qwen4_exp::TextGenerator::load_dir 로 읽으세요."
        );
    }
    Ok(())
}

/// Qwen3.5 / Qwen4-Exp 텍스트 생성기를 같은 호출 형태로 감쌉니다.
pub enum TextGenerator {
    Qwen35(Box<Qwen3_5GenerateModel>),
    Qwen4Exp(Box<Qwen4ExpGenerateModel>),
}

impl TextGenerator {
    /// HF 디렉터리를 아키텍처에 맞는 생성기로 읽습니다 (config.json 판별).
    ///  - Qwen4-Exp: safetensors 를 직접 읽습니다 (`dtype` 기본: CPU F32, GPU 는 config dtype).
    ///  - Qwen3.5: 앱이 지금 쓰는 경로 그대로 `lang_gguf::load_runtime` (Q4_K_M GGUF 변환 → `init_from_gguf`).
    ///    `Qwen3_5GenerateModel::init` 의 safetensors 로더는 GGUF 식 텐서 이름을 기대해 HF 체크포인트를 읽지 못합니다.
    ///    이 경로는 `dtype` 을 쓰지 않습니다.
    pub fn load_dir(model_dir: &Path, device: &Device, dtype: Option<DType>) -> Result<Self> {
        match detect_text_arch(model_dir) {
            TextArch::Qwen4Exp => Self::load_qwen4_exp(model_dir, device, dtype),
            TextArch::Qwen35 => Ok(Self::Qwen35(Box::new(crate::model::lang_gguf::load_runtime(model_dir, device)?))),
            TextArch::Unknown(t) => bail!("[QWEN4-EXP] {:?}: 지원하지 않는 텍스트 아키텍처 '{t}'", model_dir),
        }
    }

    fn load_qwen4_exp(model_dir: &Path, device: &Device, dtype: Option<DType>) -> Result<Self> {
        let path = model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("경로를 UTF-8 로 읽을 수 없습니다: {:?}", model_dir))?;
        println!("[QWEN4-EXP] 🔎 {:?} → Qwen4-Exp (qwen4_exp) 로 판별했습니다.", model_dir);
        Ok(Self::Qwen4Exp(Box::new(Qwen4ExpGenerateModel::init(path, Some(device), dtype)?)))
    }

    pub fn arch(&self) -> TextArch {
        match self {
            Self::Qwen35(_) => TextArch::Qwen35,
            Self::Qwen4Exp(_) => TextArch::Qwen4Exp,
        }
    }

    pub async fn generate(
        &mut self,
        mes: ChatCompletionParameters,
        cancel_flag: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>,
        ignore_list: Option<&[String]>,
        semantic_prejudice: Option<&str>,
    ) -> Result<String> {
        match self {
            Self::Qwen35(g) => g.generate(mes, cancel_flag, session_id, kv_name, ignore_list, semantic_prejudice).await,
            Self::Qwen4Exp(g) => g.generate(mes, cancel_flag, session_id, kv_name, ignore_list, semantic_prejudice).await,
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen35(g) => g.clear_kv_cache(),
            Self::Qwen4Exp(g) => g.clear_kv_cache(),
        }
    }

    pub fn get_max_tokens(&self) -> usize {
        match self {
            Self::Qwen35(g) => g.get_max_tokens(),
            Self::Qwen4Exp(g) => g.get_max_tokens(),
        }
    }

    pub fn eos_token_id(&self) -> u32 {
        match self {
            Self::Qwen35(g) => g.eos_token_id,
            Self::Qwen4Exp(g) => g.eos_token_id,
        }
    }
}

/// 앱의 언어 엔진 런타임 로더 (`lang_llm::register_loader` 에 등록, `lang_llm::LangLlmLoader` 형태).
///  - config.json 이 Qwen4-Exp 면 safetensors 를 직접 읽어 `TextGenerator::Qwen4Exp`
///  - 그 밖(Qwen3.5 alphaedge 엔진 등) 은 지금까지와 똑같이 `lang_gguf::load_runtime` → `TextGenerator::Qwen35`
///    (config 의 model_type 이 무엇이든 기존 동작을 바꾸지 않습니다; 변환기가 직접 판단합니다)
pub fn load_lang_runtime(model_dir: &Path, device: &Device) -> Result<TextGenerator> {
    match detect_text_arch(model_dir) {
        TextArch::Qwen4Exp => TextGenerator::load_qwen4_exp(model_dir, device, None),
        _ => Ok(TextGenerator::Qwen35(Box::new(crate::model::lang_gguf::load_runtime(model_dir, device)?))),
    }
}
