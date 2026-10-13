//! 🌟 [TURBO-KV-BLOCK] 블록 단위 KV 캐시(Qwen3.5)를 TurboQuant 로 압축 보관하는 코덱과
//! 블록을 순회하며 계산하는 온라인 소프트맥스 어텐션 누산기입니다.
//!
//! 저장 형식
//! - 블록의 K/V 는 `(batch, kv_heads, tokens, words)` 모양의 u32 텐서입니다. 토큰·헤드 하나(행)가
//!   head_dim 개의 값을 담습니다. 기본은 벡터별 스케일 FP8(정확도 우선, [`kv_quant_options`]),
//!   `CANDLE_TURBO_QUANT_BITS=4` 등을 지정하면 TurboQuant 코드북 / NVFP4 / MXFP4 같은 4bit 형식입니다.
//! - 회전을 쓰는 형식(4bit)은 고정된 직교 회전 Π 를 곱한 "회전 좌표계"(x·Πᵀ)에서 양자화됩니다.
//!   어텐션은 q 에도 같은 회전을 적용해 회전 좌표계에서 바로 계산하고(q·kᵀ 는 회전에 불변),
//!   출력만 마지막에 한 번 원래 좌표계로 되돌립니다. 그래서 블록마다 d×d 역회전을 하지 않습니다.
//!   기본 FP8 은 회전 없이 원래 좌표계에서 압축합니다.
//! - 형식·비트·회전·시드는 [`TqMeta`] 로 블록, RAM 캐시(BitKVMetadata), SSD 파일(`tq_meta` 텐서)에
//!   함께 기록됩니다. 다른 GPU 나 다른 설정에서 읽어도 파일에 적힌 그대로 코덱을 다시 만듭니다.
//! - 메타데이터가 없는 블록(기존 FP8 / F32 / BF16)은 예전 방식 그대로 읽고 계산합니다.

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, Tensor};
use std::sync::Arc;

/// SSD 파일의 `tq_meta` 텐서 식별자 ("TQKV", little endian)
pub const TQ_META_MAGIC: u32 = 0x564B_5154;
/// 형식 버전. 저장 구조가 바뀌면 올립니다.
pub const TQ_META_VERSION: u32 = 1;

#[cfg(not(feature = "turbo-quant"))]
const NO_FEATURE: &str =
    "turbo-kv: 이 빌드는 turbo-quant feature 없이 컴파일되어 TurboQuant 블록을 만들거나 읽을 수 없습니다";

/// TurboQuant 블록 하나의 압축 형식.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TqMeta {
    /// head_dim
    pub dim: u32,
    /// K 형식 (`candle_core::accel::formats::mode`) 과 비트 수 (TurboQuant 코드북일 때만 의미 있음)
    pub k_mode: u32,
    pub k_bits: u32,
    /// V 형식과 비트 수
    pub v_mode: u32,
    pub v_bits: u32,
    /// 0 = 회전 없음, 1 = Hadamard, 2 = Dense(무작위 직교)
    pub rotation: u32,
    /// 회전 행렬·코드북 시드
    pub seed: u64,
    /// 행 하나(토큰·헤드)당 u32 워드 수
    pub k_words: u32,
    pub v_words: u32,
}

impl TqMeta {
    pub const WORDS: usize = 12;

    pub fn to_words(&self) -> [u32; Self::WORDS] {
        [
            TQ_META_MAGIC,
            TQ_META_VERSION,
            self.dim,
            self.k_mode,
            self.k_bits,
            self.v_mode,
            self.v_bits,
            self.rotation,
            self.seed as u32,
            (self.seed >> 32) as u32,
            self.k_words,
            self.v_words,
        ]
    }

    pub fn from_words(w: &[u32]) -> Option<Self> {
        if w.len() < Self::WORDS || w[0] != TQ_META_MAGIC || w[1] != TQ_META_VERSION {
            return None;
        }
        let meta = Self {
            dim: w[2],
            k_mode: w[3],
            k_bits: w[4],
            v_mode: w[5],
            v_bits: w[6],
            rotation: w[7],
            seed: (w[8] as u64) | ((w[9] as u64) << 32),
            k_words: w[10],
            v_words: w[11],
        };
        if meta.dim == 0 || meta.k_words == 0 || meta.v_words == 0 {
            return None;
        }
        Some(meta)
    }

    /// safetensors 에서 읽은 little-endian 바이트(U32 텐서)로부터 복원합니다.
    pub fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        let w: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_words(&w)
    }

    /// SSD 저장용 U32 텐서 (CPU)
    pub fn to_tensor(&self) -> Result<Tensor> {
        Ok(Tensor::from_vec(self.to_words().to_vec(), (Self::WORDS,), &Device::Cpu)?)
    }

    /// 회전 좌표계 식별자. 같으면 같은 Π 입니다. 회전이 없으면 None(원래 좌표계).
    pub fn domain(&self) -> Option<(u32, u32, u64)> {
        if self.rotation == 0 {
            None
        } else {
            Some((self.dim, self.rotation, self.seed))
        }
    }

    /// 토큰 1개·KV 헤드 1개가 차지하는 K+V 바이트 수
    pub fn bytes_per_token_head(&self) -> usize {
        (self.k_words as usize + self.v_words as usize) * 4
    }
}

#[cfg(feature = "turbo-quant")]
mod ids {
    use anyhow::{Result, bail};
    use candle_core::accel::{QuantFormat, RotationKind, formats::mode};

    pub fn format_to_ids(f: QuantFormat) -> (u32, u32) {
        let bits = match f {
            QuantFormat::TurboMse { bits } | QuantFormat::TurboProd { bits } => bits as u32,
            _ => 0,
        };
        (f.mode_id(), bits)
    }

    pub fn format_from_ids(m: u32, bits: u32) -> Result<QuantFormat> {
        Ok(match m {
            mode::TURBO_MSE => QuantFormat::TurboMse { bits: bits as u8 },
            mode::TURBO_PROD => QuantFormat::TurboProd { bits: bits as u8 },
            mode::MXFP4 => QuantFormat::Mxfp4,
            mode::NVFP4 => QuantFormat::Nvfp4,
            mode::MXFP8 => QuantFormat::Mxfp8,
            mode::FP8 => QuantFormat::Fp8,
            other => bail!("turbo-kv: 알 수 없는 형식 id {other}"),
        })
    }

    pub fn rotation_to_id(r: RotationKind) -> u32 {
        match r {
            RotationKind::None => 0,
            RotationKind::Hadamard => 1,
            RotationKind::Dense | RotationKind::Auto => 2,
        }
    }

    pub fn rotation_from_id(id: u32) -> RotationKind {
        match id {
            0 => RotationKind::None,
            1 => RotationKind::Hadamard,
            _ => RotationKind::Dense,
        }
    }
}

/// `x(…, d) · m(d, d)` (f32)
fn matmul_last(x: &Tensor, m: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = *dims.last().ok_or_else(|| anyhow!("turbo-kv: 스칼라 텐서는 회전할 수 없습니다"))?;
    let rows = x.elem_count() / d.max(1);
    let y = x.reshape((rows, d))?.matmul(m)?;
    Ok(y.reshape(dims)?)
}

/// 한 장치에서 TurboQuant 블록을 압축·복원하는 코덱.
pub struct TqBlockCodec {
    pub meta: TqMeta,
    device: Device,
    /// (Π, Πᵀ) f32. 회전 없음이면 None.
    rot: Option<(Tensor, Tensor)>,
    #[cfg(feature = "turbo-quant")]
    kq: candle_nn::turbo_quant::TurboQuantizer,
    #[cfg(feature = "turbo-quant")]
    vq: candle_nn::turbo_quant::TurboQuantizer,
}

impl TqBlockCodec {
    #[cfg(feature = "turbo-quant")]
    fn new(meta: TqMeta, device: &Device) -> Result<Self> {
        use candle_core::accel::RotationKind;
        use candle_nn::turbo_quant::TurboQuantizer;
        let d = meta.dim as usize;
        let kf = ids::format_from_ids(meta.k_mode, meta.k_bits)?;
        let vf = ids::format_from_ids(meta.v_mode, meta.v_bits)?;
        // 회전은 코덱이 직접(Π) 처리하므로 양자화기 자체는 회전 없이 만듭니다.
        let kq = TurboQuantizer::new(kf, d, RotationKind::None, meta.seed, device)?;
        let vq = TurboQuantizer::new(vf, d, RotationKind::None, meta.seed.wrapping_add(2), device)?;
        if kq.row_words() != meta.k_words as usize || vq.row_words() != meta.v_words as usize {
            bail!(
                "turbo-kv: 행 워드 수 불일치 (K {} vs {}, V {} vs {})",
                kq.row_words(),
                meta.k_words,
                vq.row_words(),
                meta.v_words
            );
        }
        let rot = match ids::rotation_from_id(meta.rotation) {
            RotationKind::None => None,
            kind => {
                let m = candle_core::accel::turbo::rotation_matrix(d, kind, meta.seed)?;
                let m = Tensor::from_vec(m, (d, d), device)?;
                let mt = m.t()?.contiguous()?;
                Some((m, mt))
            }
        };
        Ok(Self { meta, device: device.clone(), rot, kq, vq })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `x(…, d)` → 회전 좌표계 `x·Πᵀ` (f32, 코덱 장치)
    pub fn rotate(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_device(&self.device)?.to_dtype(DType::F32)?;
        match &self.rot {
            Some((_, mt)) => matmul_last(&x, mt),
            None => Ok(x),
        }
    }

    /// 회전 좌표계 `y(…, d)` → 원래 좌표계 `y·Π` (f32, 코덱 장치)
    pub fn unrotate(&self, y: &Tensor) -> Result<Tensor> {
        let y = y.to_device(&self.device)?.to_dtype(DType::F32)?;
        match &self.rot {
            Some((m, _)) => matmul_last(&y, m),
            None => Ok(y),
        }
    }

    /// K `(b, h, t, d)` → 압축 `(b, h, t, k_words)` u32
    pub fn pack_k(&self, k: &Tensor) -> Result<Tensor> {
        self.pack(k, true)
    }

    /// V `(b, h, t, d)` → 압축 `(b, h, t, v_words)` u32
    pub fn pack_v(&self, v: &Tensor) -> Result<Tensor> {
        self.pack(v, false)
    }

    /// 압축 K → 회전 좌표계 `(b, h, t, d)` (`dt`)
    pub fn unpack_k_rot(&self, packed: &Tensor, dt: DType) -> Result<Tensor> {
        self.unpack(packed, true, dt)
    }

    /// 압축 V → 회전 좌표계 `(b, h, t, d)` (`dt`)
    pub fn unpack_v_rot(&self, packed: &Tensor, dt: DType) -> Result<Tensor> {
        self.unpack(packed, false, dt)
    }

    /// 압축 K → 원래 좌표계 `(b, h, t, d)` (`dt`). 스냅샷·진단용입니다.
    pub fn unpack_k(&self, packed: &Tensor, dt: DType) -> Result<Tensor> {
        Ok(self.unrotate(&self.unpack(packed, true, DType::F32)?)?.to_dtype(dt)?)
    }

    /// 압축 V → 원래 좌표계 `(b, h, t, d)` (`dt`). 스냅샷·진단용입니다.
    pub fn unpack_v(&self, packed: &Tensor, dt: DType) -> Result<Tensor> {
        Ok(self.unrotate(&self.unpack(packed, false, DType::F32)?)?.to_dtype(dt)?)
    }

    /// 🌟 [OFFLOAD-ATTN] 압축된 K/V 행 `(b, kv_heads, t, words)` 위에서 바로 어텐션을 계산합니다.
    /// `q`: `(b, q_heads, q_len, d)` (스케일 전, 원래 좌표계), 출력은 `q` 와 같은 dtype·모양입니다.
    /// GPU 에서는 포크의 fused TurboQuant 커널이 압축 행을 직접 읽고(복원본을 만들지 않음),
    /// 커널을 쓸 수 없으면(CPU, 자가 검증 실패, 지원 밖 형태) 포크가 행을 복원해 flash-attn 으로 계산합니다.
    /// 회전 형식이면 q 를 회전 좌표계로 옮겨 계산하고 출력만 원래 좌표계로 되돌립니다.
    #[cfg(feature = "turbo-quant")]
    pub fn attend_packed(
        &self,
        q: &Tensor,
        k_rows: &Tensor,
        v_rows: &Tensor,
        kv_len: usize,
        scaling: f64,
    ) -> Result<Tensor> {
        let dt = q.dtype();
        let cfg = candle_nn::flash_attn::FlashAttnConfig::new(scaling as f32).causal();
        let q_in = if self.rot.is_some() {
            self.rotate(q)?.to_dtype(dt)?.contiguous()?
        } else {
            q.contiguous()?
        };
        let o = candle_nn::turbo_quant::attention_packed(
            &q_in, k_rows, v_rows, kv_len, &self.kq, &self.vq, &cfg,
        )?;
        Ok(if self.rot.is_some() {
            self.unrotate(&o)?.to_dtype(dt)?
        } else {
            o.to_dtype(dt)?
        })
    }

    #[cfg_attr(not(feature = "turbo-quant"), allow(dead_code))]
    fn words(&self, keys: bool) -> usize {
        if keys {
            self.meta.k_words as usize
        } else {
            self.meta.v_words as usize
        }
    }

    #[cfg(feature = "turbo-quant")]
    fn pack(&self, x: &Tensor, keys: bool) -> Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        if d != self.meta.dim as usize {
            bail!("turbo-kv: head_dim {d} 은 코덱의 {} 와 다릅니다", self.meta.dim);
        }
        let rows = b * h * t;
        let xr = self.rotate(&x.reshape((rows, d))?)?.contiguous()?;
        let q = if keys { &self.kq } else { &self.vq };
        let packed = q.quantize_rows(&xr)?;
        Ok(packed.reshape((b, h, t, self.words(keys)))?)
    }

    #[cfg(not(feature = "turbo-quant"))]
    fn pack(&self, _x: &Tensor, _keys: bool) -> Result<Tensor> {
        bail!(NO_FEATURE)
    }

    #[cfg(feature = "turbo-quant")]
    fn unpack(&self, packed: &Tensor, keys: bool, dt: DType) -> Result<Tensor> {
        let (b, h, t, w) = packed.dims4()?;
        if packed.dtype() != DType::U32 || w != self.words(keys) {
            bail!(
                "turbo-kv: 압축 블록 형태 불일치 ({:?} {:?}, 행 워드 {} 기대)",
                packed.dtype(),
                packed.dims(),
                self.words(keys)
            );
        }
        let rows = b * h * t;
        let p = packed.to_device(&self.device)?.reshape((rows, w))?.contiguous()?;
        let q = if keys { &self.kq } else { &self.vq };
        // 양자화기에는 회전이 없으므로 결과는 회전 좌표계 값입니다.
        let y = q.dequantize_rows(&p)?;
        Ok(y.reshape((b, h, t, self.meta.dim as usize))?.to_dtype(dt)?)
    }

    #[cfg(not(feature = "turbo-quant"))]
    fn unpack(&self, _packed: &Tensor, _keys: bool, _dt: DType) -> Result<Tensor> {
        bail!(NO_FEATURE)
    }
}

#[cfg(feature = "turbo-quant")]
type CodecMap = std::collections::HashMap<(TqMeta, candle_core::DeviceLocation), Vec<Arc<TqBlockCodec>>>;

#[cfg(feature = "turbo-quant")]
fn codec_cache() -> &'static std::sync::Mutex<CodecMap> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<CodecMap>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// `meta` 형식의 코덱 (장치별로 한 번만 만들어 재사용합니다).
///
/// 🌟 [CODEC-CACHE] 위치(gpu 번호)가 같아도 장치 인스턴스가 다르면(`same_device` 거짓) 다른 코덱입니다.
/// 예전에는 위치만으로 찾아서 첫 작업의 장치가 영구히 붙잡히고, 이후 작업은 매 스텝 장치 간 복사를 했습니다.
pub fn codec_for(meta: &TqMeta, device: &Device) -> Result<Arc<TqBlockCodec>> {
    #[cfg(feature = "turbo-quant")]
    {
        let key = (*meta, device.location());
        {
            let mut cache = codec_cache().lock().map_err(|_| anyhow!("turbo-kv: 코덱 캐시 잠금 실패"))?;
            if let Some(list) = cache.get_mut(&key) {
                if let Some(c) = list.iter().find(|c| c.device.same_device(device)) {
                    return Ok(c.clone());
                }
                // 같은 위치의 옛 장치 코덱은 버립니다 (옛 장치를 붙잡지 않도록).
                list.clear();
            }
        }
        let codec = Arc::new(TqBlockCodec::new(*meta, device)?);
        codec_cache()
            .lock()
            .map_err(|_| anyhow!("turbo-kv: 코덱 캐시 잠금 실패"))?
            .entry(key)
            .or_default()
            .push(codec.clone());
        return Ok(codec);
    }
    #[cfg(not(feature = "turbo-quant"))]
    {
        let _ = (meta, device);
        bail!(NO_FEATURE)
    }
}

/// 🌟 [CODEC-CACHE] 코덱 캐시를 비웁니다 (퍼지·작업 종료). 다음 사용 때 같은 시드로 다시 만들어지므로 결과는 같습니다.
pub fn clear_codec_cache() {
    #[cfg(feature = "turbo-quant")]
    if let Ok(mut c) = codec_cache().lock() {
        c.clear();
    }
}

/// 현재 캐시에 든 코덱 수 (테스트·진단용).
pub fn codec_cache_len() -> usize {
    #[cfg(feature = "turbo-quant")]
    {
        return codec_cache().lock().map(|c| c.values().map(|v| v.len()).sum()).unwrap_or(0);
    }
    #[cfg(not(feature = "turbo-quant"))]
    0
}

/// 사용자가 KV 압축 형식·비트를 직접 고르는 환경 변수들
const FORMAT_ENV_VARS: [&str; 6] = [
    "CANDLE_TURBO_QUANT_FORMAT",
    "CANDLE_TURBO_QUANT_K_FORMAT",
    "CANDLE_TURBO_QUANT_V_FORMAT",
    "CANDLE_TURBO_QUANT_BITS",
    "CANDLE_TURBO_QUANT_K_BITS",
    "CANDLE_TURBO_QUANT_V_BITS",
];

fn env_is_set(var: &str) -> bool {
    std::env::var(var).map(|v| !v.trim().is_empty()).unwrap_or(false)
}

/// 🌟 [TURBO-KV] 이 앱의 KV 압축 형식 정책 (Qwen3 / QwenVL / Qwen3.5 공통).
///
/// - 아무것도 지정하지 않으면 **정확도 우선**: K/V 모두 벡터별 스케일 FP8(e4m3), 회전 없음.
///   기존 FP8 KV 캐스팅과 같은 원소별 정밀도에 벡터별 스케일이 붙어 448 초과 포화나 작은 값 소실이
///   없습니다. 크기는 원소당 약 1.03–1.06 바이트(BF16 의 절반)입니다.
///   (Qwen 류 키(QK-norm 의 큰 채널)를 흉내 낸 합성 데이터에서 어텐션 출력 상대오차:
///   기존 FP8 캐스팅 4–10%, 벡터별 FP8 4–6%, 4bit(NVFP4 / TurboQuant 4bit) 17–27%.
///   FP8 은 회전을 걸면 오히려 나빠지고(큰 채널의 일정한 오차가 softmax 에서 상쇄되지 않음),
///   MXFP8 은 32원소 블록 스케일 때문에 큰 채널 옆 값의 정밀도를 잃습니다.)
/// - `CANDLE_TURBO_QUANT_{FORMAT,K_FORMAT,V_FORMAT,BITS,K_BITS,V_BITS}` 중 하나라도 지정하면
///   포크의 설정을 그대로 따릅니다. 예) `CANDLE_TURBO_QUANT_BITS=4` → RTX 50 계열 NVFP4,
///   그 외 TurboQuant 4bit 코드북 (FP8 의 약 절반 크기, 소형 모델에서는 정확도 저하 위험).
/// - `CANDLE_TURBO_QUANT_ROTATION` 을 지정하면 FP8 에서도 그 회전을 씁니다.
pub fn kv_quant_options() -> candle_core::accel::TurboQuantOptions {
    use candle_core::accel::{QuantFormat, RotationKind};
    static USER: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    let (user_formats, user_rotation) = *USER.get_or_init(|| {
        (
            FORMAT_ENV_VARS.iter().any(|v| env_is_set(v)),
            env_is_set("CANDLE_TURBO_QUANT_ROTATION"),
        )
    });
    let mut o = candle_core::accel::config().turbo_quant;
    if !user_formats {
        o.key_format = Some(QuantFormat::Fp8);
        o.value_format = Some(QuantFormat::Fp8);
        o.key_bits = 8;
        o.value_bits = 8;
        if !user_rotation {
            o.rotation = RotationKind::None;
        }
    }
    o
}

/// 🌟 [KV-PRECISION] KV 정밀도 정책 (`LOGIS_KV_PRECISION`).
/// - `auto` (기본): VRAM 여유가 충분하면 KV 를 압축하지 않고 모델 dtype(원본) 그대로 두고,
///   모자랄 때만 TurboQuant(벡터별 스케일 FP8)로 압축합니다.
/// - `exact`: 항상 원본 KV (메모리가 모자라 RAM/SSD 로 내릴 때만 기존처럼 압축)
/// - `compressed`: 항상 압축 KV (이전 동작)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecisionPolicy {
    Auto,
    Exact,
    Compressed,
}

pub fn kv_precision_policy() -> KvPrecisionPolicy {
    static P: std::sync::OnceLock<KvPrecisionPolicy> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        match std::env::var("LOGIS_KV_PRECISION").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "exact" | "full" | "bf16" | "f16" | "exact32" | "f32" | "half" | "exact16" => KvPrecisionPolicy::Exact,
            "compressed" | "turbo" | "fp8" | "compact" => KvPrecisionPolicy::Compressed,
            _ => KvPrecisionPolicy::Auto,
        }
    })
}

/// 🌟 [KV-PRECISION] 이번 생성의 KV 를 압축하지 않고 원본 정밀도로 둘 수 있는지. `(원본, 잰 VRAM free)`
///
/// 실측(Qwen3-0.6B 실모델, 같은 토큰 열 강제): 벡터별 FP8 KV 는 JSON 추출·검증 프롬프트에서 다음 토큰이
/// 원본 KV 와 100% 같았지만, 수천 토큰 뒤의 긴 자유 서술에서는 4% 안팎의 스텝에서 1순위 토큰이 바뀌었습니다.
/// 압축의 이득(메모리·전송량)은 KV 가 VRAM 에 넉넉히 들어가는 짧은 문맥에서는 거의 없으므로,
/// `exact_bytes`(원본 KV 예상치) + KV 안전 여유(640MB) + `reserve`(이번 계산의 다른 순간치) 가
/// VRAM free 안에 들어가면 원본을 씁니다. 들어가지 않으면(긴 문맥·VRAM 부족·오프로딩) 압축합니다.
/// CPU 는 원래 KV 를 압축하지 않으므로 그대로(원본) 둡니다.
pub fn exact_kv_affordable(device: &Device, exact_bytes: u64, reserve: u64) -> (bool, Option<u64>) {
    match kv_precision_policy() {
        KvPrecisionPolicy::Exact => return (true, None),
        KvPrecisionPolicy::Compressed => return (false, None),
        KvPrecisionPolicy::Auto => {}
    }
    use crate::utils::GpuDeviceExt;
    if !(device.is_cuda_or_rocm() || device.is_vulkan()) {
        return (true, None);
    }
    // 메모리 풀이 쥐고 있던 이전 계산의 여분을 돌려준 뒤 잽니다.
    let _ = device.synchronize();
    match crate::utils::gpu_mem_info(crate::utils::resources::primary_gpu_id() as usize) {
        Some((free, _)) => (
            free >= exact_bytes + crate::utils::resources::KV_VRAM_SAFETY_MARGIN_BYTES + reserve,
            Some(free),
        ),
        None => (false, None),
    }
}

/// 🌟 [KV-HALF] F32 로 도는 모델의 원본 KV 를 F16 으로 보관할지 (`LOGIS_KV_PRECISION`).
/// - `Auto` (기본): F32 원본 KV 가 VRAM 에 들어가면 F32, 안 들어가고 F16 이면 들어가면 F16, 둘 다 안 되면 압축.
///   (예전에는 F32 가 안 들어가면 곧바로 FP8 압축이었으므로 어떤 경우에도 정밀도가 같거나 높습니다)
/// - `Prefer` (`half` / `exact16` / `f16`): 원본 KV 를 고르면 항상 F16 으로 보관합니다.
/// - `Off` (`exact32` / `f32`): 예전처럼 F32 로만 보관합니다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvHalfMode {
    Auto,
    Prefer,
    Off,
}

pub fn kv_half_mode() -> KvHalfMode {
    static M: std::sync::OnceLock<KvHalfMode> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        match std::env::var("LOGIS_KV_PRECISION").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "exact32" | "f32" => KvHalfMode::Off,
            "half" | "exact16" | "f16" => KvHalfMode::Prefer,
            _ => KvHalfMode::Auto,
        }
    })
}

/// F16 로 담을 수 있는 최댓값 (넘으면 BF16 으로 보관합니다).
pub const F16_SAFE_MAX: f32 = 60000.0;

/// [`exact_kv_affordable`] 과 같되 KV 안전 여유를 호출부가 정합니다 (비례 여유, [`crate::utils::resources::kv_vram_margin_bytes`]).
pub fn exact_kv_affordable_with_margin(device: &Device, exact_bytes: u64, reserve: u64, margin: u64) -> (bool, Option<u64>) {
    match kv_precision_policy() {
        KvPrecisionPolicy::Exact => return (true, None),
        KvPrecisionPolicy::Compressed => return (false, None),
        KvPrecisionPolicy::Auto => {}
    }
    use crate::utils::GpuDeviceExt;
    if !(device.is_cuda_or_rocm() || device.is_vulkan()) {
        return (true, None);
    }
    match crate::utils::gpu_mem_info(crate::utils::resources::primary_gpu_id() as usize) {
        Some((free, _)) => (free >= exact_bytes + margin + reserve, Some(free)),
        None => (false, None),
    }
}

/// 이 장치에서 실제로 쓰일 (K, V) 압축 형식
pub fn kv_formats(device: &Device) -> (candle_core::accel::QuantFormat, candle_core::accel::QuantFormat) {
    candle_core::accel::profile(device).turbo_formats(&kv_quant_options())
}

/// 로그용 형식 이름 (예: `K fp8 / V fp8`)
pub fn kv_formats_label(device: &Device) -> String {
    let (k, v) = kv_formats(device);
    format!("K {k} / V {v}")
}

/// K 또는 V 가 8bit 미만(4bit 등) 형식인지.
/// 이때 Qwen3 / QwenVL 은 프리필을 기존 경로로 하고 디코딩부터 압축 캐시를 씁니다.
/// (프롬프트 표현이 저비트 오차를 레이어마다 누적하지 않도록)
pub fn kv_is_compact(device: &Device) -> bool {
    use candle_core::accel::QuantFormat;
    let (k, v) = kv_formats(device);
    let eight = |f: QuantFormat| matches!(f, QuantFormat::Fp8 | QuantFormat::Mxfp8);
    !(eight(k) && eight(v))
}

/// 현재 정책([`kv_quant_options`])과 GPU 에 맞는 블록 형식.
pub fn default_meta(head_dim: usize, device: &Device) -> Result<TqMeta> {
    #[cfg(feature = "turbo-quant")]
    {
        use candle_core::accel;
        let conf = kv_quant_options();
        let (kf, vf) = accel::profile(device).turbo_formats(&conf);
        let rotation = conf.rotation.resolve(head_dim);
        let kl = accel::formats::row_layout(kf, head_dim)?;
        let vl = accel::formats::row_layout(vf, head_dim)?;
        let (k_mode, k_bits) = ids::format_to_ids(kf);
        let (v_mode, v_bits) = ids::format_to_ids(vf);
        return Ok(TqMeta {
            dim: head_dim as u32,
            k_mode,
            k_bits,
            v_mode,
            v_bits,
            rotation: ids::rotation_to_id(rotation),
            seed: conf.seed,
            k_words: kl.row_words as u32,
            v_words: vl.row_words as u32,
        });
    }
    #[cfg(not(feature = "turbo-quant"))]
    {
        let _ = (head_dim, device);
        bail!(NO_FEATURE)
    }
}

/// 새 블록을 TurboQuant 로 압축할 코덱. 쓸 수 없으면(turbo-quant 미빌드, CPU,
/// `CANDLE_TURBO_QUANT=off`, GPU 자가 검증 실패, 지원하지 않는 head_dim 등) None 이고
/// 호출부는 기존 FP8 경로를 씁니다.
pub fn block_codec(head_dim: usize, device: &Device) -> Option<Arc<TqBlockCodec>> {
    if !super::turbo_kv_ready(device, head_dim) {
        return None;
    }
    match default_meta(head_dim, device).and_then(|m| codec_for(&m, device)) {
        Ok(c) => Some(c),
        Err(e) => {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                println!("[TURBO-KV-BLOCK] TurboQuant 블록 코덱을 만들 수 없어 기존 FP8 블록을 사용합니다: {e}");
            }
            None
        }
    }
}

/// 누산기의 좌표계: 원래 좌표계 또는 특정 코덱의 회전 좌표계
#[derive(Clone)]
enum Domain {
    Plain,
    Rot(Arc<TqBlockCodec>),
}

impl Domain {
    fn key(&self) -> Option<(u32, u32, u64)> {
        match self {
            Domain::Plain => None,
            Domain::Rot(c) => c.meta.domain(),
        }
    }

    fn of(codec: &Arc<TqBlockCodec>) -> Self {
        if codec.meta.domain().is_some() {
            Domain::Rot(codec.clone())
        } else {
            Domain::Plain
        }
    }
}

/// `x` (f32) 를 `from` 좌표계에서 `to` 좌표계로 옮깁니다.
fn convert(x: &Tensor, from: &Domain, to: &Domain) -> Result<Tensor> {
    if from.key() == to.key() {
        return Ok(x.clone());
    }
    let plain = match from {
        Domain::Plain => x.clone(),
        Domain::Rot(c) => c.unrotate(x)?,
    };
    match to {
        Domain::Plain => Ok(plain),
        Domain::Rot(c) => c.rotate(&plain),
    }
}

/// 블록을 하나씩 받아 온라인 소프트맥스로 합치는 어텐션 누산기.
///
/// `Qwen3_5Attention` 의 기존 블록 루프(GQA 접기, 블록별 마스크 자르기, -10000 하한,
/// f32 통계)를 그대로 옮겼습니다. 평문 블록(FP8/BF16/F32)은 예전과 같은 계산이고,
/// TurboQuant 블록은 회전 좌표계에서 계산한 결과를 좌표계를 맞춰 합칩니다.
pub struct BlockAttention {
    /// `(b, q_heads, q_len, d)`, 스케일 적용됨
    q: Tensor,
    dims: (usize, usize, usize, usize),
    kv_heads: usize,
    groups: usize,
    dt: DType,
    q_plain: Option<Tensor>,
    q_rot: Vec<((u32, u32, u64), Tensor)>,
    out: Option<Tensor>,
    m: Option<Tensor>,
    l: Option<Tensor>,
    domain: Option<Domain>,
    blocks: usize,
}

impl BlockAttention {
    /// `q_scaled`: `(b, q_heads, q_len, d)` 에 `scaling` 을 이미 곱한 쿼리.
    pub fn new(q_scaled: &Tensor, kv_heads: usize, groups: usize) -> Result<Self> {
        let q = q_scaled.contiguous()?;
        let dims = q.dims4()?;
        Ok(Self {
            dt: q.dtype(),
            q,
            dims,
            kv_heads,
            groups: groups.max(1),
            q_plain: None,
            q_rot: Vec::new(),
            out: None,
            m: None,
            l: None,
            domain: None,
            blocks: 0,
        })
    }

    /// 지금까지 합친 블록 수
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// [GQA-FOLD] `(b, q_h, q_l, d)` → `(b, kv_h, groups*q_l, d)`
    fn fold(&self, q: &Tensor) -> Result<Tensor> {
        let (b, _h, l, d) = self.dims;
        if self.groups > 1 {
            Ok(q.reshape((b, self.kv_heads, self.groups * l, d))?)
        } else {
            Ok(q.clone())
        }
    }

    fn q_for(&mut self, domain: &Domain) -> Result<Tensor> {
        match domain {
            Domain::Plain => {
                if self.q_plain.is_none() {
                    self.q_plain = Some(self.fold(&self.q)?);
                }
                Ok(self.q_plain.clone().expect("q_plain"))
            }
            Domain::Rot(codec) => {
                let key = codec.meta.domain().expect("rotated domain");
                if let Some((_, t)) = self.q_rot.iter().find(|(k, _)| *k == key) {
                    return Ok(t.clone());
                }
                let qr = codec.rotate(&self.q)?.to_dtype(self.dt)?.contiguous()?;
                let folded = self.fold(&qr)?;
                self.q_rot.push((key, folded.clone()));
                Ok(folded)
            }
        }
    }

    /// 평문 블록 `k`, `v`: `(b, kv_heads, len, d)`. `b_off` 는 블록의 첫 토큰 위치(마스크 열 위치)입니다.
    pub fn add_plain(&mut self, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, b_off: usize) -> Result<()> {
        let k = k.to_dtype(self.dt)?.contiguous()?;
        let v = v.to_dtype(self.dt)?.contiguous()?;
        let q = self.q_for(&Domain::Plain)?;
        let (o, m, l) = self.block_stats(&q, &k, &v, mask, b_off)?;
        self.merge(o, m, l, Domain::Plain)
    }

    /// TurboQuant 블록 `k`, `v`: `(b, kv_heads, len, words)` u32 (어느 장치에 있어도 됩니다)
    pub fn add_packed(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        codec: &Arc<TqBlockCodec>,
        mask: Option<&Tensor>,
        b_off: usize,
    ) -> Result<()> {
        let k = codec.unpack_k_rot(k, self.dt)?.contiguous()?;
        let v = codec.unpack_v_rot(v, self.dt)?.contiguous()?;
        let domain = Domain::of(codec);
        let q = self.q_for(&domain)?;
        let (o, m, l) = self.block_stats(&q, &k, &v, mask, b_off)?;
        drop(k);
        drop(v);
        self.merge(o, m, l, domain)
    }

    fn block_stats(
        &self,
        q_folded: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        b_off: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (q_b, q_h, q_l, q_d) = self.dims;
        let kv_len = k.dim(2)?;

        // [ZERO-COPY-T] transpose 뷰를 그대로 matmul 에 전달합니다.
        let k_t = k.transpose(2, 3)?;
        let s_folded = q_folded.matmul(&k_t)?;
        let mut s_chunk = if self.groups > 1 {
            s_folded.reshape((q_b, q_h, q_l, kv_len))?
        } else {
            s_folded
        };

        if let Some(mask) = mask {
            let mask_len = mask.dim(D::Minus1)?;
            if b_off < mask_len {
                let take = std::cmp::min(kv_len, mask_len - b_off);
                let chunk_mask = mask.narrow(D::Minus1, b_off, take)?;
                if take < kv_len {
                    let left = s_chunk.narrow(D::Minus1, 0, take)?.broadcast_add(&chunk_mask)?;
                    let right = s_chunk.narrow(D::Minus1, take, kv_len - take)?;
                    s_chunk = Tensor::cat(&[&left, &right], D::Minus1)?;
                } else {
                    s_chunk = s_chunk.broadcast_add(&chunk_mask)?;
                }
            }
        }

        let s_f32 = s_chunk.to_dtype(DType::F32)?;
        let m_j = s_f32.max_keepdim(D::Minus1)?;
        // -inf - (-inf) = NaN 방지
        let safe_floor = Tensor::new(-10000.0_f32, m_j.device())?.broadcast_as(m_j.shape())?;
        let m_j_safe = m_j.maximum(&safe_floor)?;
        let p_j = s_f32.broadcast_sub(&m_j_safe)?.exp()?;
        let l_j = p_j.sum_keepdim(D::Minus1)?;

        // [GQA-FOLD] P @ V 도 접어서 계산 후 원래 head 배열로 복원합니다.
        let p_v = p_j.to_dtype(v.dtype())?.contiguous()?;
        let p_folded = if self.groups > 1 {
            p_v.reshape((q_b, self.kv_heads, self.groups * q_l, kv_len))?
        } else {
            p_v
        };
        let out_folded = p_folded.matmul(v)?;
        let out_j = if self.groups > 1 {
            out_folded.reshape((q_b, q_h, q_l, q_d))?
        } else {
            out_folded
        };
        Ok((out_j.to_dtype(DType::F32)?, m_j, l_j))
    }

    fn merge(&mut self, out_j: Tensor, m_j: Tensor, l_j: Tensor, domain: Domain) -> Result<()> {
        self.blocks += 1;
        let (prev_out, prev_m, prev_l, cur) = match (self.out.take(), self.m.take(), self.l.take(), self.domain.clone()) {
            (Some(o), Some(m), Some(l), Some(d)) => (o, m, l, d),
            _ => {
                self.out = Some(out_j);
                self.m = Some(m_j);
                self.l = Some(l_j);
                self.domain = Some(domain);
                return Ok(());
            }
        };
        let out_j = convert(&out_j, &domain, &cur)?;
        let m_new = prev_m.maximum(&m_j)?;
        let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
        let diff_new = m_j.broadcast_sub(&m_new)?.exp()?;
        let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
        let out_new = prev_out.broadcast_mul(&diff_old)?.add(&out_j.broadcast_mul(&diff_new)?)?;
        self.out = Some(out_new);
        self.m = Some(m_new);
        self.l = Some(l_new);
        Ok(())
    }

    /// 정규화된 어텐션 출력 `(b, q_heads, q_len, d)` (f32, 원래 좌표계). 블록이 하나도 없으면 None.
    pub fn finish(self) -> Result<Option<Tensor>> {
        let (Some(out), Some(l)) = (self.out, self.l) else {
            return Ok(None);
        };
        let o = out.broadcast_div(&l)?;
        Ok(Some(match self.domain {
            Some(Domain::Rot(c)) => c.unrotate(&o)?,
            _ => o,
        }))
    }
}

/// 🌟 [OFFLOAD-ATTN] 블록 하나의 K/V (어텐션 계산 직전, 계산 장치에 올라온 상태).
/// `tq` 가 Some 이면 `k` / `v` 는 그 형식의 압축 u32 행 `(b, kv_heads, len, words)`,
/// None 이면 평문(FP8/F16/F32) `(b, kv_heads, len, d)` 입니다.
pub struct KvPart {
    pub k: Tensor,
    pub v: Tensor,
    pub tq: Option<TqMeta>,
    /// 블록의 첫 토큰 위치
    pub off: usize,
    /// 블록에 든 토큰 수
    pub len: usize,
}

impl KvPart {
    /// 블록 K/V 가 차지하는 바이트 수
    pub fn bytes(&self) -> usize {
        self.k.elem_count() * self.k.dtype().size_in_bytes()
            + self.v.elem_count() * self.v.dtype().size_in_bytes()
    }
}

/// 🌟 [OFFLOAD-ATTN] 디코딩(q_len 1): 모든 블록이 같은 TurboQuant 형식이면 압축 행을 그대로 이어 붙여
/// fused 커널 한 번으로 계산합니다. (블록마다 복원 → matmul → 온라인 소프트맥스 병합을 반복하지 않습니다)
/// RAM 에 있는 블록은 압축된 채로 올라옵니다. 조건이 맞지 않으면 `Ok(None)`.
pub fn fused_decode_attention(q: &Tensor, parts: &[KvPart], scaling: f64) -> Result<Option<Tensor>> {
    #[cfg(feature = "turbo-quant")]
    {
        let Some(meta) = parts.first().and_then(|p| p.tq) else {
            return Ok(None);
        };
        if parts.iter().any(|p| p.tq != Some(meta) || p.len == 0) {
            return Ok(None);
        }
        let dev = q.device();
        let codec = codec_for(&meta, dev)?;
        let total: usize = parts.iter().map(|p| p.len).sum();
        let rows = |pick: &dyn Fn(&KvPart) -> &Tensor| -> Result<Tensor> {
            let mut v = Vec::with_capacity(parts.len());
            for p in parts {
                v.push(pick(p).narrow(2, 0, p.len)?.to_device(dev)?);
            }
            Ok(if v.len() == 1 { v.remove(0) } else { Tensor::cat(&v, 2)? })
        };
        let k_rows = rows(&|p| &p.k)?;
        let v_rows = rows(&|p| &p.v)?;
        return Ok(Some(codec.attend_packed(q, &k_rows, &v_rows, total, scaling)?));
    }
    #[cfg(not(feature = "turbo-quant"))]
    {
        let _ = (q, parts, scaling);
        Ok(None)
    }
}

/// 🌟 [OFFLOAD-ATTN] 과거 이력(`hist_end` 앞 위치, 압축 블록이면 복원) 뒤에 이번 스텝의 **원본** K/V 를
/// 이어 붙여 타일 flash-attn 으로 계산합니다. (causal, 쿼리는 마지막 q_len 위치)
/// - 프리필 청크가 자기 자신(같은 청크의 K/V)을 볼 때 압축 오차가 없습니다. 기존 블록 경로는 방금 압축한
///   블록을 다시 풀어 썼기 때문에 프롬프트 전체가 FP8 오차를 안고 계산되었습니다.
/// - S(q_len × kv_len) 행렬을 만들지 않으므로 프리필 순간 VRAM 이 블록 수와 무관하게 작습니다.
/// `q`: `(b, q_heads, q_len, d)` 스케일 전, `k_cur` / `v_cur`: `(b, kv_heads, q_len, d)`.
pub fn flash_history_attention(
    q: &Tensor,
    k_cur: &Tensor,
    v_cur: &Tensor,
    parts: &[KvPart],
    hist_end: usize,
    scaling: f64,
) -> Result<Tensor> {
    let dev = q.device();
    let dt = q.dtype();
    let mut ks: Vec<Tensor> = Vec::with_capacity(parts.len() + 1);
    let mut vs: Vec<Tensor> = Vec::with_capacity(parts.len() + 1);
    for p in parts {
        if p.off >= hist_end {
            continue;
        }
        let take = p.len.min(hist_end - p.off);
        if take == 0 {
            continue;
        }
        let k = p.k.narrow(2, 0, take)?.contiguous()?;
        let v = p.v.narrow(2, 0, take)?.contiguous()?;
        match p.tq {
            Some(meta) => {
                let codec = codec_for(&meta, dev)?;
                ks.push(codec.unpack_k(&k, dt)?);
                vs.push(codec.unpack_v(&v, dt)?);
            }
            None => {
                ks.push(k.to_device(dev)?.to_dtype(dt)?);
                vs.push(v.to_device(dev)?.to_dtype(dt)?);
            }
        }
    }
    ks.push(k_cur.to_dtype(dt)?);
    vs.push(v_cur.to_dtype(dt)?);
    let k = if ks.len() == 1 { ks.remove(0) } else { Tensor::cat(&ks, 2)? }.contiguous()?;
    let v = if vs.len() == 1 { vs.remove(0) } else { Tensor::cat(&vs, 2)? }.contiguous()?;
    let cfg = candle_nn::flash_attn::FlashAttnConfig::new(scaling as f32).causal();
    Ok(candle_nn::flash_attn::flash_attn_bhsd(&q.contiguous()?, &k, &v, &cfg)?)
}
