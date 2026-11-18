//! Qwen4-Exp 텍스트 생성기 — 다른 텍스트 생성기(`Qwen3_5GenerateModel::generate`)와 같은 호출 형태.
//!
//! - `init(path, device, dtype)`: HF 디렉터리 (config.json + safetensors + tokenizer.json + 채팅 템플릿)
//! - `generate(mes, cancel_flag, session_id, kv_name, ignore_list, semantic_prejudice) -> String`
//! - `generate_ids(prompt, max_new, ..)`: 토큰 단위 핵심 루프 (프리필 청크 → 증분 디코드)
//!
//! 세션 KV 디스크 저장/부분 복원, 비전 입력, semantic prejudice 는 아직 없습니다 (해당 인자는 받아서 무시).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use super::config::Qwen4ExpTextConfig;
use super::model::Qwen4ExpForCausalLM;
use crate::chat_template::ChatTemplate;
use crate::openai_types::ChatCompletionParameters;
use crate::tokenizer::TokenizerModel;
use crate::utils::{get_device, get_dtype, get_logit_processor};

/// 프리필 청크 크기 (마스크/MoE 순간 메모리 상한). 모든 상태가 청크 사이로 이어지므로 결과는 같습니다.
pub const PREFILL_CHUNK: usize = 2048;

pub struct Qwen4ExpGenerateModel {
    chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub model: Qwen4ExpForCausalLM,
    device: Device,
    pub eos_token_id: u32,
    /// 생성 종료 토큰 집합 (config eos 리스트 + `<|im_end|>`)
    pub stop_token_ids: Vec<u32>,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
    pub max_position_embeddings: usize,
}

impl Qwen4ExpGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let dir = std::path::Path::new(path);
        let model_name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("qwen4-exp").to_string();
        let cfg = Qwen4ExpTextConfig::from_file(&dir.join("config.json"))?;
        let device = get_device(device);
        let dtype = match dtype {
            Some(d) => d,
            None if device.is_cpu() => DType::F32,
            None => get_dtype(None, cfg.dtype.as_deref().unwrap_or("bfloat16")),
        };
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let model = Qwen4ExpForCausalLM::load(dir, dtype, &device)?;
        let mut stop = cfg.eos_ids();
        if let Ok(v) = tokenizer.text_encode_vec("<|im_end|>".to_string(), false) {
            if v.len() == 1 && !stop.contains(&v[0]) {
                stop.push(v[0]);
            }
        }
        let eos_token_id = stop.first().copied().unwrap_or(cfg.ple_eos());
        let max_pos = cfg.max_position_embeddings;
        println!(
            "[QWEN4-EXP] 🧠 생성기 준비 | {model_name} · dtype {dtype:?} · 종료 토큰 {stop:?} · max_pos {max_pos}"
        );
        Ok(Self {
            chat_template,
            tokenizer,
            model,
            device,
            eos_token_id,
            stop_token_ids: stop,
            model_name,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_position_embeddings: max_pos,
        })
    }

    pub fn get_max_tokens(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_cache();
    }

    /// 프롬프트 토큰을 처음부터 넣고 최대 `max_new` 토큰을 생성합니다 (종료 토큰은 결과에 포함하지 않음).
    pub fn generate_ids(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampler: &mut LogitsProcessor,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt.is_empty(), "[QWEN4-EXP] 빈 프롬프트");
        self.model.clear_cache();
        let mut logits = None;
        let mut pos = 0;
        while pos < prompt.len() {
            let n = (prompt.len() - pos).min(PREFILL_CHUNK);
            let ids = Tensor::from_slice(&prompt[pos..pos + n], (1, n), &self.device)?;
            logits = Some(self.model.forward_last(&ids, pos)?);
            pos += n;
        }
        let mut logits = logits.expect("non-empty prompt");
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            if cancel_flag.map(|f| f.load(Ordering::Relaxed)).unwrap_or(false) {
                break;
            }
            let mut lv = logits.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            if self.repeat_penalty != 1.0 && !out.is_empty() {
                let mut seen = std::collections::HashSet::new();
                for &t in &out[out.len().saturating_sub(self.repeat_last_n)..] {
                    if seen.insert(t) && (t as usize) < lv.len() {
                        let l = lv[t as usize];
                        lv[t as usize] = if l < 0.0 { l * self.repeat_penalty } else { l / self.repeat_penalty };
                    }
                }
            }
            let n = lv.len();
            let next = sampler.sample(&Tensor::from_vec(lv, n, &Device::Cpu)?)?;
            if self.stop_token_ids.contains(&next) {
                break;
            }
            out.push(next);
            if prompt.len() + out.len() >= self.max_position_embeddings {
                break;
            }
            let ids = Tensor::from_slice(&[next], (1, 1), &self.device)?;
            logits = self.model.forward_last(&ids, prompt.len() + out.len() - 1)?;
        }
        Ok(out)
    }

    /// `Qwen3_5GenerateModel::generate` 와 같은 시그니처. 세션 KV/semantic prejudice/ignore_list 는 아직 쓰지 않습니다.
    pub async fn generate(
        &mut self,
        mes: ChatCompletionParameters,
        cancel_flag: Option<Arc<AtomicBool>>,
        _session_id: Option<String>,
        _kv_name: Option<String>,
        _ignore_list: Option<&[String]>,
        _semantic_prejudice: Option<&str>,
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.0);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mut sampler = get_logit_processor(Some(temperature as f32), Some(top_p as f32), Some(80), seed);
        let rendered = self.chat_template.apply_chat_template(&mes)?;
        let ids = self.tokenizer.text_encode_vec(rendered, false)?;
        let max_new = mes.max_tokens.unwrap_or(1024) as usize;
        let started = std::time::Instant::now();
        let out = self.generate_ids(&ids, max_new, &mut sampler, cancel_flag.as_deref())?;
        let text = self
            .tokenizer
            .token_decode(out.clone())
            .map_err(|e| anyhow!("[QWEN4-EXP] 디코드 실패: {e}"))?;
        println!(
            "[QWEN4-EXP] ✍️ 생성 {} 토큰 (프롬프트 {}) · {:.2}s",
            out.len(),
            ids.len(),
            started.elapsed().as_secs_f32()
        );
        let _ = std::io::stdout().flush();
        self.model.clear_cache();
        Ok(text)
    }
}
