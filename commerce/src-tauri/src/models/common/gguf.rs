use std::io::{Read, Seek};

use ahash::AHashMap;
use anyhow::{Result, anyhow};
use candle_core::{
    DType, Device, Tensor,
    quantized::{
        QMatMul, QTensor,
        gguf_file::{self, Value},
    },
};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, LayerNorm, Linear, Module, RmsNorm, VarBuilder, linear_b,
};
use tokenizers::{self, AddedToken, Tokenizer, models::bpe::BPE};

use crate::tokenizer::TokenizerModel;

/// 🌟 [MMAP-LOAD] GGUF 텐서를 파일 전체 바이트(메모리 맵 `&mmap[..]`)에서 바로 읽습니다.
///
/// `Content::tensor` 는 텐서마다 힙 버퍼를 새로 잡아 맵의 페이지를 한 번 복사한 뒤 장치로 올립니다.
/// 층 가중치를 레이어마다 다시 읽는 스트리밍 경로에서는 그 복사가 층 크기만큼 매번 반복되고
/// 힙도 그때마다 크게 출렁입니다. 여기서는 맵의 바이트를 그대로 넘겨 GPU 업로드가 맵 페이지를 직접 읽습니다.
/// (CPU 장치는 어차피 자기 저장소로 복사합니다. 결과 텐서는 `Content::tensor` 와 비트 단위로 같습니다)
pub fn qtensor_from_bytes(
    ct: &gguf_file::Content,
    file: &[u8],
    name: &str,
    device: &Device,
) -> candle_core::Result<QTensor> {
    let info = ct
        .tensor_infos
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor info for {name}")))?;
    let elems = info.shape.elem_count();
    let block = info.ggml_dtype.block_size();
    if block == 0 || elems % block != 0 {
        return Err(candle_core::Error::Msg(format!(
            "{name}: {elems} elements are not a multiple of the block size {block}"
        )));
    }
    let size = elems / block * info.ggml_dtype.type_size();
    let start = usize::try_from(ct.tensor_data_offset.saturating_add(info.offset))
        .map_err(|_| candle_core::Error::Msg(format!("{name}: offset does not fit in memory")))?;
    if start.checked_add(size).map_or(true, |end| end > file.len()) {
        return Err(candle_core::Error::Msg(format!(
            "{name}: needs {size} bytes at offset {start}, the file has {}",
            file.len()
        )));
    }
    let raw = &file[start..start + size];
    // GGML 블록(f16/f32 필드)으로 다시 해석되므로 정렬이 맞지 않으면(정상 GGUF·맵에서는 생기지 않음) 복사 경로로 읽습니다.
    if (raw.as_ptr() as usize) % 4 != 0 {
        return info.read(&mut std::io::Cursor::new(file), ct.tensor_data_offset, device);
    }
    candle_core::quantized::ggml_file::qtensor_from_ggml(
        info.ggml_dtype,
        raw,
        info.shape.dims().to_vec(),
        device,
    )
}

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    /// 🌟 [VISION-JIT] 비전 모델이 재로드용 device를 보관할 수 있도록 노출합니다.
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_matedata(&self, name: &str) -> Result<Value> {
        match self.ct.metadata.get(name) {
            None => Err(anyhow!("cannot find {name} in metadata")),
            Some(v) => Ok(v.clone()),
        }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        Ok(QMatMul::from_qtensor(ws)?)
    }

    pub fn quantize_linear(&mut self, prefix: &str, bias: bool) -> Result<QuantizedLinear> {
        let weight = self.qmatmul(&format!("{prefix}.weight"))?;
        let bias = if bias {
            self.get_dequantized_f16(&format!("{prefix}.bias")).ok() 
        } else { None };
        Ok(QuantizedLinear::new(weight, bias))
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        
        let weight = ws.dequantize_f16(&self.device).or_else(|_| ws.dequantize(&self.device))?.to_dtype(candle_core::DType::F32)?; 
        Ok(RmsNorm::new(weight, eps))
    }

    pub fn layer_norm(&mut self, prefix: &str, eps: f64) -> Result<LayerNorm> {
        let ws = self.ct.tensor(&mut self.reader, &format!("{prefix}.weight"), &self.device)?;
        
        let weight = ws.dequantize_f16(&self.device).or_else(|_| ws.dequantize(&self.device))?.to_dtype(candle_core::DType::F32)?;
        let bias = self.ct.tensor(&mut self.reader, &format!("{prefix}.bias"), &self.device);
        let bias = match bias {
            
            Ok(b) => b.dequantize_f16(&self.device).or_else(|_| b.dequantize(&self.device)).ok().map(|t| t.to_dtype(candle_core::DType::F32).unwrap_or(t)),
            Err(_) => None,
        };
        match bias {
            Some(bias) => Ok(LayerNorm::new(weight, bias, eps)),
            None => Ok(LayerNorm::new_no_bias(weight, eps)),
        }
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        Ok(self.ct.tensor(&mut self.reader, name, &self.device)?)
    }

    pub fn get_dequantized(&mut self, name: &str) -> Result<Tensor> {
        Ok(self.tensor(name)?.dequantize(&self.device)?)
    }

    pub fn get_dequantized_f16(&mut self, name: &str) -> Result<Tensor> {
        let t = self.tensor(name)?;
        
        Ok(t.dequantize_f16(&self.device).or_else(|_| t.dequantize(&self.device))?)
    }

    pub fn conv1d(
        &mut self,
        prefix: &str,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Conv1d> {
        
        let weight = self.get_dequantized_f16(&format!("{prefix}.weight"))?.to_dtype(candle_core::DType::F32)?;
        let bias = if bias {
            self.get_dequantized_f16(&format!("{prefix}.bias")).ok().map(|t| t.to_dtype(candle_core::DType::F32).unwrap_or(t))
        } else {
            None
        };
        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        Ok(Conv1d::new(weight, bias, cfg))
    }

    pub fn build_tokenizer(
        &self,
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
    ) -> Result<TokenizerModel> {
        let model_type = self
            .get_matedata("tokenizer.ggml.model")?
            .to_string()?
            .clone();
        match model_type.as_str() {
            "gpt2" | "llama" => {
                let vocab = self
                    .get_matedata("tokenizer.ggml.tokens")?
                    .to_vec()?
                    .clone();
                let vocab: Vec<String> = vocab
                    .into_iter()
                    .map(|tokens| tokens.to_string().cloned())
                    .collect::<Result<Vec<String>, candle_core::Error>>()?;
                let mut vocab_map = AHashMap::new();
                for (id, token) in vocab.iter().enumerate() {
                    vocab_map.insert(token.clone(), id as u32);
                }

                let merges = self
                    .get_matedata("tokenizer.ggml.merges")?
                    .to_vec()?
                    .clone();
                let merges: Vec<String> = merges
                    .into_iter()
                    .map(|tokens| tokens.to_string().cloned())
                    .collect::<Result<Vec<String>, candle_core::Error>>()?;
                let merges: Vec<(String, String)> = merges
                    .into_iter()
                    .map(|token_merge| {
                        let merge: Vec<&str> = token_merge.split(" ").collect();
                        if merge.len() != 2 {
                            // 处理格式不正确的merge规则
                            return ("".to_string(), "".to_string());
                        }
                        (merge[0].to_string(), merge[1].to_string())
                    })
                    .filter(|(a, b)| !a.is_empty() && !b.is_empty())
                    .collect();
                let bpe_model = BPE::new(vocab_map, merges);
                let mut tokenizer = Tokenizer::new(bpe_model);
                let add_prefix_space = add_prefix_space.unwrap_or(false);
                let trim_offsets = trim_offsets.unwrap_or(false);
                let use_regex = use_regex.unwrap_or(false);
                let pre_byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::default()
                    .add_prefix_space(add_prefix_space) // 是否在文本开头添加空格，gpt-2默认是true
                    .trim_offsets(trim_offsets) // 是否删除首尾空白字符
                    .use_regex(use_regex); // 是否使用正则表达式来分割特殊字符
                tokenizer.with_pre_tokenizer(Some(pre_byte_level));
                let dec_byte_level = tokenizers::decoders::byte_level::ByteLevel::default();
                tokenizer.with_decoder(Some(dec_byte_level));
                let token_types = self
                    .get_matedata("tokenizer.ggml.token_type")?
                    .to_vec()?
                    .clone();
                let token_types = token_types
                    .into_iter()
                    .map(|types| types.to_i32())
                    .collect::<Result<Vec<i32>, candle_core::Error>>()?;

                let mut add_tokens = vec![];
                for (id, type_) in token_types.into_iter().enumerate() {
                    if type_ == 3 {
                        if let Some(token_str) = vocab.get(id) {
                            let add_token = AddedToken::from(token_str.clone(), true);
                            add_tokens.push(add_token);
                        }
                    } else if type_ == 4 {
                        if let Some(token_str) = vocab.get(id) {
                            let add_token = AddedToken::from(token_str.clone(), false);
                            add_tokens.push(add_token);
                        }
                    }
                }
                let _ = tokenizer.add_special_tokens(&add_tokens);
                let tokenizer_model = TokenizerModel::new(tokenizer);
                Ok(tokenizer_model)
            }
            _ => Err(anyhow!("Unsupported tokenizer model type: {model_type}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    pub fn new(inner: QMatMul, bias: Option<Tensor>) -> Self {
        Self { inner, bias }
    }

    pub fn inner_dequantize(&self) -> Result<Tensor> {
        Ok(self.inner.dequantize_f16()?)
    }

    /// 🌟 [W8A16] F16/BF16 입력을 정확 경로(`forward_exact`)로 보낼지: CUDA/ROCm 에 상주한 Q8_0 가중치일 때만.
    /// (K-quant 가중치·다른 장치는 기존 경로 그대로. ROCm 은 포크의 QRocmStorage 정확 경로를 씁니다. RX 6600 실측: Qwen3-0.6B top-1 100%, KLD 5.0e-4)
    fn exact_ok(&self, xs: &Tensor) -> bool {
        matches!(xs.dtype(), DType::F16 | DType::BF16)
            && q8_exact_enabled()
            && matches!(&self.inner, QMatMul::QTensor(t)
                if t.dtype() == candle_core::quantized::GgmlDType::Q8_0 && (t.device().is_cuda() || t.device().is_rocm()))
    }
}

// =====================================================================
// 🌟 [W8A16] Q8_0 가중치 × F16/BF16 활성값의 정확 행렬곱
// ---------------------------------------------------------------------
//  기존: F16 입력은 `forward_via_f16` (호출마다 가중치 전체를 F16 으로 풀고 F16 matmul — 디코드 1토큰에도
//        25MB 급 순간 버퍼), BF16 입력은 `forward` (MMVQ/MMQ: 활성값을 Q8_1 로 양자화).
//  지금: `QMatMul::forward_exact` — 활성값을 양자화하지 않습니다. 디코드(≤8행)는 Q8_0 블록을 커널 안에서
//        f32 로 풀어 f32 누산, 프리필은 활성 dtype 타일로 풀어 f32 누산 GEMM. 정확도는 같거나 낫습니다.
//  `LOGIS_Q8_EXACT=0` 이면 기존 경로. 테스트는 [`set_q8_exact_override`] 로 같은 프로세스에서 비교합니다.
// =====================================================================
static Q8_EXACT_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// 🌟 [W8A16] 테스트용 강제값. `None` = 환경변수 기본, `Some(true/false)` = 강제로 켜기/끄기.
pub fn set_q8_exact_override(v: Option<bool>) {
    let code = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    Q8_EXACT_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
}

fn q8_exact_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    match Q8_EXACT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => *ON.get_or_init(|| {
            let v = std::env::var("LOGIS_Q8_EXACT").unwrap_or_default().trim().to_ascii_lowercase();
            let on = !matches!(v.as_str(), "0" | "off" | "false" | "no" | "legacy");
            if !on {
                println!("[W8A16] LOGIS_Q8_EXACT={v} → Q8_0 선형층을 기존 경로(F16 전체 역양자화 / Q8_1 활성 양자화)로 계산합니다.");
            }
            on
        }),
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = if self.exact_ok(xs) {
            self.inner.forward_exact(xs)?
        } else if xs.dtype() == DType::F16 {
            self.inner.forward_via_f16(xs)?
        } else {
            self.inner.forward(xs)?
        };
        if let Some(bias) = &self.bias {
            xs.broadcast_add(&bias.to_dtype(xs.dtype())?)
        } else {
            Ok(xs)
        }
    }
}

/// 🌟 [W8A16] 양자화 가중치를 그대로 상주시키고 활성값은 양자화하지 않는 정확 선형층 (Qwen3-0.6B Q8_0).
/// 출력 dtype = 입력 dtype (F32 입력이면 F32 logits).
#[derive(Debug, Clone)]
pub struct ExactQLinear {
    w: std::sync::Arc<QTensor>,
}

impl ExactQLinear {
    pub fn new(w: std::sync::Arc<QTensor>) -> Self {
        Self { w }
    }

    pub fn weight(&self) -> &std::sync::Arc<QTensor> {
        &self.w
    }
}

impl Module for ExactQLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.w.forward_exact(xs)
    }
}

#[derive(Debug, Clone)]
pub enum ProjKind {
    QuantizedProj(QuantizedLinear),
    LinearProj(Linear),
    /// 🌟 [W8A16] 정확 양자화 선형층 (활성 양자화 없음)
    Exact(ExactQLinear),
}

impl Module for ProjKind {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            ProjKind::QuantizedProj(q) => q.forward(xs),
            ProjKind::LinearProj(l) => l.forward(xs),
            ProjKind::Exact(e) => e.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateUpDownMLPGguf {
    gate_proj: ProjKind,
    up_proj: ProjKind,
    down_proj: ProjKind,
    act: Activation,
}

pub fn dummy_proj(device: &Device) -> ProjKind {
    ProjKind::LinearProj(Linear::new(Tensor::zeros((1, 1), DType::F32, device).unwrap(), None))
}


impl GateUpDownMLPGguf {
    pub fn new_dummy(device: &Device) -> Self {
        let dummy = dummy_proj(device);
        Self {
            gate_proj: dummy.clone(),
            up_proj: dummy.clone(),
            down_proj: dummy,
            act: candle_nn::Activation::Silu,
        }
    }
    
    pub fn clear_weights(&mut self) {
        let dummy = dummy_proj(&Device::Cpu);
        self.gate_proj = dummy.clone();
        self.up_proj = dummy.clone();
        self.down_proj = dummy;
    }

    /// `data`: GGUF 파일 전체 바이트(메모리 맵). [`qtensor_from_bytes`] 로 힙 복사 없이 읽습니다.
    pub fn load_weights_inplace(&mut self, ct: &gguf_file::Content, data: &[u8], prefix: &str, device: &Device) -> Result<()> {
        let gate_w = qtensor_from_bytes(ct, data, &format!("{prefix}.ffn_gate.weight"), device)?;
        self.gate_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(gate_w)?, None));
        
        let up_w = qtensor_from_bytes(ct, data, &format!("{prefix}.ffn_up.weight"), device)?;
        self.up_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(up_w)?, None));
        
        let down_w = qtensor_from_bytes(ct, data, &format!("{prefix}.ffn_down.weight"), device)?;
        self.down_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(down_w)?, None));
        
        Ok(())
    }
    
    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        bias: bool,
        gate_name: Option<&str>,
        up_name: Option<&str>,
        down_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let gate_name = gate_name.unwrap_or("ffn_gate");
        let up_name = up_name.unwrap_or("ffn_up");
        let down_name = down_name.unwrap_or("ffn_down");
        let gate_proj = gguf.qmatmul(&format!("{prefix}.{gate_name}.weight"))?;
        let gate_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{gate_name}.bias"))
                .ok()
        } else {
            None
        };
        let up_proj = gguf.qmatmul(&format!("{prefix}.{up_name}.weight"))?;
        let up_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{up_name}.bias"))
                .ok()
        } else {
            None
        };
        let down_proj = gguf.qmatmul(&format!("{prefix}.{down_name}.weight"))?;
        let down_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{down_name}.bias"))
                .ok()
        } else {
            None
        };
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            gate_proj: ProjKind::QuantizedProj(QuantizedLinear::new(gate_proj, gate_bias)),
            up_proj: ProjKind::QuantizedProj(QuantizedLinear::new(up_proj, up_bias)),
            down_proj: ProjKind::QuantizedProj(QuantizedLinear::new(down_proj, down_bias)),
            act,
        })
    }
    pub fn new_from_vb(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        gate_pp_name: Option<&str>,
        up_pp_name: Option<&str>,
        down_pp_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let gate_pp_name = gate_pp_name.unwrap_or("gate_proj");
        let up_pp_name = up_pp_name.unwrap_or("up_proj");
        let down_pp_name = down_pp_name.unwrap_or("down_proj");
        let gate_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(gate_pp_name))?;
        let up_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(up_pp_name))?;
        let down_proj = linear_b(intermediate_size, hidden_size, bias, vb.pp(down_pp_name))?;
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            gate_proj: ProjKind::LinearProj(gate_proj),
            up_proj: ProjKind::LinearProj(up_proj),
            down_proj: ProjKind::LinearProj(down_proj),
            act,
        })
    }
}

impl Module for GateUpDownMLPGguf {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let w1 = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let w3 = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(w1 * w3)?)
    }
}

pub struct TwoLinearMLPGguf {
    linear1: ProjKind,
    linear2: ProjKind,
    act: Activation,
}

impl TwoLinearMLPGguf {
    /// 🌟 [VISION-SKELETON] 가중치 없이 구조만 만듭니다. VRAM 점유 0 바이트.
    ///
    ///  ── 왜 필요한가 ──
    ///   Qwen3VLVisionBlock 을 '빈 껍데기' 로 먼저 만든 뒤 mmap 에서 하나씩
    ///   읽어 쓰려면 mlp 도 껍데기로 태어날 수 있어야 합니다.
    ///   그런데 linear1 / linear2 는 이 모듈 밖에서 접근할 수 없어
    ///   호출부가 구조체 리터럴로 만들 방법이 없습니다.
    ///   GateUpDownMLPGguf 에는 이미 new_dummy 가 있는데 이쪽에만 없었습니다.
    pub fn new_dummy(device: &Device, act: Activation) -> Self {
        let dummy = dummy_proj(device);
        Self {
            linear1: dummy.clone(),
            linear2: dummy,
            act,
        }
    }

    pub fn new(
        vb: VarBuilder,
        // embedding_dim: usize,
        // mlp_dim: usize,
        in_dim: usize,
        middle_dim: usize,
        out_dim: usize,
        act: Activation,
        bias: bool,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
    ) -> Result<Self> {
        let linear1 = linear_b(in_dim, middle_dim, bias, vb.pp(linear1_pp_name))?;
        let linear2 = linear_b(middle_dim, out_dim, bias, vb.pp(linear2_pp_name))?;

        Ok(Self {
            linear1: ProjKind::LinearProj(linear1),
            linear2: ProjKind::LinearProj(linear2),
            act,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        bias: bool,
        linear1_name: Option<&str>,
        linear2_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let linear1_name = linear1_name.unwrap_or("ffn_up");
        let linear2_name = linear2_name.unwrap_or("ffn_down");
        let linear1 = gguf.quantize_linear(&format!("{prefix}.{linear1_name}"), bias)?;
        let linear2 = gguf.quantize_linear(&format!("{prefix}.{linear2_name}"), bias)?;
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            linear1: ProjKind::QuantizedProj(linear1),
            linear2: ProjKind::QuantizedProj(linear2),
            act,
        })
    }
    /// 🌟 [VISION-JIT] mmproj MLP 가중치를 1바이트 껍데기로 교체하여 메모리를 즉시 반환합니다.
    pub fn clear_weights(&mut self) {
        let dummy = dummy_proj(&Device::Cpu);
        self.linear1 = dummy.clone();
        self.linear2 = dummy;
    }

    /// 🌟 [VISION-STREAM] 레이어 스트리밍이 Gguf 래퍼를 거치지 않고
    ///   &Content 로 직접 읽은 프로젝션을 밀어 넣을 수 있도록 노출합니다.
    pub fn set_projs(&mut self, linear1: ProjKind, linear2: ProjKind) {
        self.linear1 = linear1;
        self.linear2 = linear2;
    }

    /// 🌟 [VISION-JIT] 비전 입력이 감지되면 mmproj GGUF에서 in-place로 재로드합니다.
    pub fn load_weights_inplace<R: Read + Seek>(
        &mut self,
        gguf: &mut Gguf<R>,
        prefix: &str,
        bias: bool,
        linear1_name: Option<&str>,
        linear2_name: Option<&str>,
    ) -> Result<()> {
        let linear1_name = linear1_name.unwrap_or("ffn_up");
        let linear2_name = linear2_name.unwrap_or("ffn_down");
        self.linear1 = ProjKind::QuantizedProj(
            gguf.quantize_linear(&format!("{prefix}.{linear1_name}"), bias)?,
        );
        self.linear2 = ProjKind::QuantizedProj(
            gguf.quantize_linear(&format!("{prefix}.{linear2_name}"), bias)?,
        );
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}
