pub mod gguf;
pub mod vision;
pub mod text;
pub mod preprocessor;
pub mod vision_crop;
pub mod vision_encoder;
pub mod legibility;
pub mod nms_arena;
pub mod value_grounding;
pub mod tokenizer;
pub mod phrase_cache;
pub mod layout_cache;
pub mod quant;
pub mod gguf_quant;


use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

pub struct Siglip2Model {
    pub vision: Option<vision::Siglip2VisionModel>,
    pub text: Option<text::Siglip2TextModel>,
    pub tokenizer: Option<tokenizer::Siglip2Tokenizer>,
    pub logit_scale: f32,
    pub logit_bias: f32,
    pub device: Device,
    pub dtype: DType,
    pub config: Siglip2Config,
}

/// 모델 설정 (config.json에서 파싱)
#[derive(Debug, Clone)]
pub struct Siglip2Config {
    // vision_config
    pub vision_hidden_size: usize,       // 1152
    pub vision_intermediate_size: usize, // 4304
    pub vision_num_layers: usize,        // 27
    pub vision_num_heads: usize,         // 16
    pub patch_size: usize,               // 16
    pub max_num_patches: usize,          // 256 (NaFlex)
    pub vision_layer_norm_eps: f64,      // 1e-6
    // text_config
    pub text_hidden_size: usize,         // 1152
    pub text_intermediate_size: usize,   // 4304
    pub text_num_layers: usize,          // 27
    pub text_num_heads: usize,           // 16
    pub text_vocab_size: usize,          // 256000
    pub text_max_positions: usize,       // 64  ← 512 아님
    pub text_pad_token_id: u32,          // 1
    pub text_layer_norm_eps: f64,        // 1e-6
}

impl Siglip2Config {
    pub fn from_json(config_path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let vc = &v["vision_config"];
        let tc = &v["text_config"];
        Ok(Self {
            vision_hidden_size: vc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            vision_intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            vision_num_layers: vc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            vision_num_heads: vc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            max_num_patches: vc["num_patches"].as_u64().unwrap_or(256) as usize,
            vision_layer_norm_eps: vc["layer_norm_eps"].as_f64().unwrap_or(1e-6),
            text_hidden_size: tc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            text_intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            text_num_layers: tc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            text_num_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            text_vocab_size: tc["vocab_size"].as_u64().unwrap_or(256000) as usize,
            text_max_positions: tc["max_position_embeddings"].as_u64().unwrap_or(64) as usize,
            text_pad_token_id: tc["pad_token_id"].as_u64().unwrap_or(1) as u32,
            text_layer_norm_eps: tc["layer_norm_eps"].as_f64().unwrap_or(1e-6),
        })
    }

    /// 위치 임베딩 격자 한 변의 길이. 256 → 16.
    pub fn pos_grid_side(&self) -> usize {
        let s = (self.max_num_patches as f64).sqrt().round() as usize;
        s.max(1)
    }
}

/// logit_scale / logit_bias 는 루트 스칼라입니다. 없으면 기본값.
fn root_scalar(vb: &VarBuilder, name: &str) -> f32 {
    vb.get(1, name)
        .ok()
        .and_then(|t| t.to_dtype(DType::F32).ok())
        .and_then(|t| t.to_vec1::<f32>().ok())
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0)
}

/// 양자화 타워 GGUF 의 매핑 (블록 형식 가중치를 여기서 올립니다)
type QuantFile = Option<std::sync::Arc<gguf::MappedGguf>>;

/// `LOGIS_SIGLIP2_QUANT_SIM=1`: 양자화 계획을 블록 그대로 곱하지 않고 F32 로 풀어 올립니다 (민감도 측정 경로, quant.rs).
fn quant_sim() -> bool {
    std::env::var("LOGIS_SIGLIP2_QUANT_SIM").map(|v| v == "1").unwrap_or(false)
}

/// 🌟 [PYRAMID-QUANT] 계획의 타워 GGUF 를 (없으면 원본에서 만들어) 엽니다. 실패하면 `None` — 호출부는 원본 F32 로 읽습니다.
fn quant_tower(model_dir: &Path, plan: &quant::WeightPlan, path: std::path::PathBuf, skip: &[&str], device: &Device) -> Option<(VarBuilder<'static>, std::sync::Arc<gguf::MappedGguf>)> {
    let opened = gguf::prepare_quant(model_dir, plan)
        .and_then(|_| gguf::open_cached(&path))
        .and_then(|f| Ok((f.var_builder(skip, device)?, f)));
    match opened {
        Ok(parts) => Some(parts),
        Err(e) => {
            println!("[SigLIP2-QUANT] 계획 '{}' 의 타워 GGUF 를 쓰지 못해 원본 F32 로 읽습니다: {}", plan.tag, e);
            None
        }
    }
}

/// 🌟 [SPLIT-GGUF] 비전 타워를 읽을 VarBuilder. 타워별 GGUF 가 있으면 매핑에서 바로 올리고, 아니면 safetensors 를 읽습니다.
fn vision_builder(model_dir: &Path, device: &Device, dtype: DType) -> anyhow::Result<(VarBuilder<'static>, QuantFile)> {
    // 🌟 [PYRAMID-QUANT] 양자화 계획이면 원본에서 한 번 만든 블록을 그대로 올립니다 (gguf_quant.rs).
    let plan = quant::plan_for(device);
    if plan.is_quantized() {
        if quant_sim() {
            return Ok((quant::dequantized_tower(model_dir, "vision_model.", &[], device, &plan)?, None));
        }
        if let Some((vb, f)) = quant_tower(model_dir, &plan, gguf::quant_vision_path(model_dir, &plan), &[], device) {
            return Ok((vb, Some(f)));
        }
    }
    if gguf::usable(model_dir, dtype) {
        match gguf::open_cached(&gguf::vision_path(model_dir)).and_then(|f| f.var_builder(&[], device)) {
            Ok(vb) => return Ok((vb, None)),
            Err(e) => println!("[SigLIP2] mmproj 를 읽지 못해 safetensors 로 읽습니다: {}", e),
        }
    }
    let path = model_dir.join("model.safetensors");
    Ok((unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? }, None))
}

/// 🌟 [SPLIT-GGUF] 텍스트 타워를 읽을 VarBuilder 와 토큰 표의 위치.
fn text_builder(model_dir: &Path, device: &Device, dtype: DType) -> anyhow::Result<(VarBuilder<'static>, text::TokenSource<'static>, QuantFile)> {
    let plan = quant::plan_for(device);
    if plan.is_quantized() {
        let source = model_dir.join("model.safetensors");
        if quant_sim() {
            let vb = quant::dequantized_tower(model_dir, "text_model.", &[gguf::TOKEN_TABLE], device, &plan)?;
            return Ok((vb, text::TokenSource::Mapped(gguf::MappedRows::from_safetensors(&source, gguf::TOKEN_TABLE)?), None));
        }
        if let Some((vb, f)) = quant_tower(model_dir, &plan, gguf::quant_text_path(model_dir, &plan), &[], device) {
            // 토큰 표는 원본 F32 에서 쓰는 행만 읽습니다 (VRAM · 호스트 메모리 0).
            match gguf::MappedRows::from_safetensors(&source, gguf::TOKEN_TABLE) {
                Ok(rows) => return Ok((vb, text::TokenSource::Mapped(rows), Some(f))),
                Err(e) => println!("[SigLIP2-QUANT] 원본 토큰 표를 매핑하지 못해 원본 F32 로 읽습니다: {}", e),
            }
        }
    }
    if gguf::usable(model_dir, dtype) {
        let opened = gguf::open_cached(&gguf::text_path(model_dir)).and_then(|f| {
            let vb = f.var_builder(&[gguf::TOKEN_TABLE], device)?;
            Ok((vb, text::TokenSource::Mapped(f.rows(gguf::TOKEN_TABLE)?), None))
        });
        match opened {
            Ok(parts) => return Ok(parts),
            Err(e) => println!("[SigLIP2] 텍스트 GGUF 를 읽지 못해 safetensors 로 읽습니다: {}", e),
        }
    }
    let safetensors_path = model_dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, device)? };
    // 🌟 [CPU EMBEDDING] 590MB 를 VRAM 에서 뺍니다.
    let tokens = if device.is_cpu() {
        text::TokenSource::Inline
    } else {
        match unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &Device::Cpu) } {
            Ok(v) => text::TokenSource::Host(v.pp("text_model")),
            Err(e) => {
                println!(
                    "[SigLIP2] CPU 임베딩 VarBuilder 생성 실패({}). token_embedding 을 VRAM 에 유지합니다.",
                    e
                );
                text::TokenSource::Inline
            }
        }
    };
    Ok((vb, tokens, None))
}

impl Siglip2Model {
    pub fn load_vision_only(
        safetensors_path: &Path,
        config: &Siglip2Config,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let model_dir = safetensors_path.parent().unwrap_or_else(|| Path::new("."));
        // 양자화 계획은 F32 활성으로 돕니다. 호출부가 예전 규칙(GPU BF16)으로 dtype 을 넘겨도 계획을 따릅니다.
        let dtype = if quant::plan_for(device).is_quantized() { DType::F32 } else { dtype };
        let (vb, q) = vision_builder(model_dir, device, dtype)?;
        quant::note_loaded(device);

        let vision_model = vision::Siglip2VisionModel::new_with(config, vb.pp("vision_model"), q.as_deref())?;

        let logit_scale = root_scalar(&vb, "logit_scale");
        let logit_bias = root_scalar(&vb, "logit_bias");

        println!(
            "[SigLIP2] Vision encoder loaded (layers={}, hidden={}, patch={}, max_patches={}, logit_scale={:.4})",
            config.vision_num_layers,
            config.vision_hidden_size,
            config.patch_size,
            config.max_num_patches,
            logit_scale
        );

        Ok(Self {
            vision: Some(vision_model),
            text: None,
            tokenizer: None,
            logit_scale,
            logit_bias,
            device: device.clone(),
            dtype,
            config: config.clone(),
        })
    }

    pub fn load_text_only(
        model_dir: &Path,
        config: &Siglip2Config,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let dtype = if quant::plan_for(device).is_quantized() { DType::F32 } else { dtype };
        let (vb, tokens, q) = text_builder(model_dir, device, dtype)?;
        quant::note_loaded(device);
        let text_model = text::Siglip2TextModel::new_with(config, vb.pp("text_model"), tokens, q.as_deref())?;

        let logit_scale = root_scalar(&vb, "logit_scale");
        let logit_bias = root_scalar(&vb, "logit_bias");

        let tok = tokenizer::Siglip2Tokenizer::from_dir(
            model_dir,
            config.text_pad_token_id,
            config.text_max_positions,
        )?;

        println!(
            "[SigLIP2] Text-only mode loaded (layers={}, vocab={}, seq_len={}). Vision weights NOT loaded (~820MB saved).",
            config.text_num_layers,
            config.text_vocab_size,
            config.text_max_positions
        );

        Ok(Self {
            vision: None,
            text: Some(text_model),
            tokenizer: Some(tok),
            logit_scale,
            logit_bias,
            device: device.clone(),
            dtype,
            config: config.clone(),
        })
    }

    pub fn load_vision_encoder(&mut self, model_dir: &Path) -> anyhow::Result<()> {
        if self.vision.is_some() {
            return Ok(());
        }
        let (vb, q) = vision_builder(model_dir, &self.device, self.dtype)?;
        quant::note_loaded(&self.device);
        let vision_model = vision::Siglip2VisionModel::new_with(&self.config, vb.pp("vision_model"), q.as_deref())?;
        self.vision = Some(vision_model);
        println!(
            "[SigLIP2] Vision encoder ATTACHED to existing instance (layers={}, patch={}).",
            self.config.vision_num_layers, self.config.patch_size
        );
        Ok(())
    }

    pub fn load_text_encoder(&mut self, model_dir: &Path) -> anyhow::Result<()> {
        let (vb, tokens, q) = text_builder(model_dir, &self.device, self.dtype)?;
        quant::note_loaded(&self.device);
        let text_model = text::Siglip2TextModel::new_with(&self.config, vb.pp("text_model"), tokens, q.as_deref())?;
        self.text = Some(text_model);

        let tok = tokenizer::Siglip2Tokenizer::from_dir(
            model_dir,
            self.config.text_pad_token_id,
            self.config.text_max_positions,
        )?;
        self.tokenizer = Some(tok);

        println!(
            "[SigLIP2] Text encoder + tokenizer loaded (layers={}, vocab={}, seq_len={}, pad_id={})",
            self.config.text_num_layers,
            self.config.text_vocab_size,
            self.config.text_max_positions,
            self.config.text_pad_token_id
        );
        Ok(())
    }

    pub fn has_text(&self) -> bool {
        self.text.is_some() && self.tokenizer.is_some()
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn detach_vision(&mut self) -> bool {
        if self.vision.is_none() {
            return false;
        }
        self.vision = None;
        println!("[SigLIP2] Vision encoder DETACHED (약 856MB VRAM 반납).");
        true
    }

    pub fn detach_text(&mut self) -> bool {
        if self.text.is_none() && self.tokenizer.is_none() {
            return false;
        }
        self.text = None;
        self.tokenizer = None;
        println!("[SigLIP2] Text encoder DETACHED.");
        true
    }
}