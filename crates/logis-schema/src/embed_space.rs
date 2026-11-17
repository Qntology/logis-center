//! granite 임베딩 공간의 차원 · 순전파 모드(LOGIS_EMBED_FORWARD) · 코사인 눈금 보정 표. 예전 models/embedding.rs 에서 옮김:
//! 검색 · 배정 문턱(ai_utils 코사인)과 SDS 원장 폴더가 모델 없이 이 값을 씁니다.

/// 임베딩 차원 (granite-embedding-97m-multilingual-r2 hidden_size).
pub const EMBED_DIM: usize = 384;

// =====================================================================
// 🌟 [EMBED-FORWARD] granite-embedding-97m-multilingual-r2 순전파 판본
// ---------------------------------------------------------------------
//  ── 무엇이 다른가 ──
//   모델 config.json 은 ModernBertModel 입니다: LayerNorm(norm_bias=false), 전역 층(3층마다)
//   RoPE θ 150000 · 지역 층 θ 160000 + ±64 슬라이딩 창(local_attention 128), 공식 문장 임베딩은
//   CLS 풀링(1_Pooling: pooling_mode_cls_token) + 정규화입니다.
//   이 파일의 기존 순전파(Legacy)는 RmsNorm · 모든 층 θ 150000 · 전역 어텐션 · 평균 풀링이고,
//   GPU 에서는 RoPE 표를 BF16 으로 만들어 256 을 넘는 위치가 짝수로 뭉개집니다.
//   두 판본 벡터의 코사인은 0.79~0.89 입니다(로컬 가중치 재현).
//
//  ── 기본값: ModernBERT 기준 순전파 + CLS 풀링 ──
//   HF transformers 대비 코사인 CPU 1.00000 · CUDA ≥ 0.9999 (40개 텍스트, 4~423 토큰).
//   Legacy 는 평균 0.923, 256 토큰을 넘으면 0.71~0.74 까지 벌어집니다.
//   풀링은 두 재현으로 골랐습니다(로컬 가중치 · 순전파를 파이썬으로 재현한 오프라인 측정).
//    · 라벨 뱅크 2,349구 자기 제외 라우팅(12개 언어, merge.rs z 점수 재현):
//        Legacy 60.5% · CLS 63.4% · 평균 65.5%
//    · 문서 제목 다국어 질의 471개 → 합성 무역 문서 45건(512 토큰) 검색 acc@1 / MRR:
//        Legacy 0.125 / 0.241 · CLS 0.335 / 0.471 · 평균 0.221 / 0.338
//   짧은 라벨끼리는 평균 풀링이 2점 높지만, 저장 벡터의 본업인 검색에서 CLS(모델이 학습한 공식 풀링)가
//   평균보다 MRR 0.13 높아 CLS 로 둡니다.
//
//  ── 판본을 바꾸면 ──
//   LanceDB 에 저장된 벡터 공간이 달라지므로 store.rs migrate_embedding_space 가 아이템 벡터를 대기열로
//   돌리고 청크를 비운 뒤 백그라운드 재색인이 새 공간으로 다시 만듭니다. SDS 원장도 공간마다 따로 둡니다
//   (score_dynamics.rs). 텍스트 캐시는 판본마다 공간(순전파 태그)이 달라 섞이지 않습니다.
//   LOGIS_EMBED_FORWARD=legacy 로 되돌리면 같은 경로가 거꾸로 돕니다.
// =====================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    /// 예전 순전파 (RmsNorm · 단일 θ · 전역 어텐션 · 평균 풀링). LOGIS_EMBED_FORWARD=legacy 로만 씁니다.
    Legacy,
    /// ModernBERT 기준 순전파 + CLS 풀링 (공식 문장 임베딩). 기본값.
    Exact,
    /// ModernBERT 기준 순전파 + 평균 풀링 (비교 실험용).
    ExactMean,
}

impl ForwardMode {
    pub fn from_env_value(v: Option<&str>) -> Self {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("legacy") => ForwardMode::Legacy,
            Some("exact-mean") => ForwardMode::ExactMean,
            _ => ForwardMode::Exact,
        }
    }

    /// 저장 벡터 공간의 이름(LanceDB 표지 · SDS 원장 폴더). 파일 이름에 그대로 씁니다.
    pub fn space_tag(&self) -> &'static str {
        match self {
            ForwardMode::Legacy => "legacy",
            ForwardMode::Exact => "modernbert-cls-v1",
            ForwardMode::ExactMean => "modernbert-mean-v1",
        }
    }

    /// 캐시 공간을 나누는 순전파 태그 접미사. Legacy 는 예전 태그 그대로입니다.
    pub fn tag_suffix(&self) -> Option<&'static str> {
        match self {
            ForwardMode::Legacy => None,
            ForwardMode::Exact => Some("+modernbert-cls/v1"),
            ForwardMode::ExactMean => Some("+modernbert-mean/v1"),
        }
    }

    pub fn is_exact(&self) -> bool {
        !matches!(self, ForwardMode::Legacy)
    }

    pub fn cls_pooling(&self) -> bool {
        matches!(self, ForwardMode::Exact)
    }
}

/// 값은 프로세스 안에서 한 번만 읽습니다(캐시 공간이 이 판정에 묶여 있습니다).
pub static EMBED_FORWARD_ENV: once_cell::sync::Lazy<ForwardMode> = once_cell::sync::Lazy::new(|| {
    let raw = std::env::var("LOGIS_EMBED_FORWARD").ok();
    let mode = ForwardMode::from_env_value(raw.as_deref());
    println!(
        "[EMBED-FORWARD] LOGIS_EMBED_FORWARD → {} ({})",
        raw.as_deref().unwrap_or("미설정"),
        match mode {
            ForwardMode::Legacy => "예전 순전파 · 평균 풀링 (되돌림)",
            ForwardMode::Exact => "ModernBERT 기준 순전파 · CLS 풀링 (기본)",
            ForwardMode::ExactMean => "ModernBERT 기준 순전파 · 평균 풀링 (실험)",
        }
    );
    mode
});

/// 이 프로세스의 임베딩 순전파 판본.
pub fn forward_mode() -> ForwardMode {
    *EMBED_FORWARD_ENV
}

// =====================================================================
// 🌟 [COS CALIBRATION] CLS 공간 코사인 → 예전 순전파 눈금
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   CLS 벡터는 한쪽으로 몰려 있어(이방성) 코사인이 좁은 띠에 모입니다. 같은 텍스트 쌍 635만 개
//   (라벨 뱅크 12개 언어 2,391구 × 레이아웃 긴 텍스트 포함)에서 예전 순전파는 평균 0.551 · 표준편차 0.076,
//   CLS 는 평균 0.781 · 표준편차 0.034 였고, 같은 쌍의 상관은 0.58 이라 상수 하나씩 옮겨 적을 수 없습니다.
//   코드 곳곳의 절대 문턱(pair_abs_floor 0.50 · 헤더 0.62 · 노이즈 0.35/0.55 · 상태 0.35 · 비율 ×0.85 …)과
//   검색 융합 가중치, SDS 기준선은 전부 예전 눈금에서 맞춘 값입니다.
//
//  ── 무엇을 하나 ──
//   granite(384차원) 코사인을 분위수 대응(같은 순위 → 같은 예전 값)으로 옮깁니다. 단조 함수라 순위
//   · 최근접 검색 결과는 그대로이고, 각 문턱을 넘는 쌍의 비율이 예전과 같아집니다
//   (0.50 초과 77.0% → 76.9% · 0.62 초과 16.8% → 16.7%). z 점수 라우팅은 오히려 좋아집니다:
//   라벨 뱅크 자기 제외 라우팅 CLS 그대로 63.4% → 보정 66.1% (예전 60.5%).
//   동일 텍스트(1.0)는 1.0 그대로입니다. 표는 위 635만 쌍의 두 판본 코사인 분위수(확률 0 · 1e-5 · 1e-4 · 1e-3 ·
//   0.005 · 0.01 · 0.02 · 0.05 · 0.1 … 0.9 · 0.95 · 0.98 · 0.99 · 0.995 · 0.999 · 0.9999 · 0.99999 · 1)를 짝지은
//   것에 (−1, −1) · (1, 1) 을 더한 것입니다. 모델 가중치나 라벨 뱅크(bias.json)가 바뀌면 다시 맞춰야 합니다.
//   다시 맞추는 절차는 embedding/calibration.rs, 실행은 tests/it/calibration.rs 의 refit_cls_to_legacy_cos 입니다.
//   표를 바꾸면 CLS 공간의 SDS 기준선과 섞이지 않게 ForwardMode::space_tag 의 판본도 올립니다.
// =====================================================================
pub const CLS_TO_LEGACY_COS: [(f32, f32); 27] = [
    (-1.0000, -1.0000),
    (0.6330, 0.1070),
    (0.6540, 0.1710),
    (0.6688, 0.2132),
    (0.6869, 0.2716),
    (0.7018, 0.3246),
    (0.7091, 0.3520),
    (0.7171, 0.3820),
    (0.7291, 0.4234),
    (0.7397, 0.4564),
    (0.7527, 0.4921),
    (0.7622, 0.5161),
    (0.7705, 0.5359),
    (0.7784, 0.5538),
    (0.7865, 0.5714),
    (0.7954, 0.5900),
    (0.8065, 0.6117),
    (0.8236, 0.6424),
    (0.8394, 0.6692),
    (0.8600, 0.7033),
    (0.8755, 0.7300),
    (0.8904, 0.7587),
    (0.9259, 0.8304),
    (0.9684, 0.9224),
    (0.9915, 0.9710),
    (0.9997, 0.9944),
    (1.0000, 1.0000),
];

/// 지금 쓰는 CLS → 예전 눈금 보정표 (다시 맞춘 표와 비교할 때 씁니다).
pub fn cls_to_legacy_knots() -> &'static [(f32, f32)] {
    &CLS_TO_LEGACY_COS
}

/// 보정표 사이를 직선으로 잇습니다. 표 밖은 양 끝 값, NaN · 무한대는 그대로 돌려줍니다.
pub fn interp_knots(x: f32, knots: &[(f32, f32)]) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x <= knots[0].0 {
        return knots[0].1;
    }
    for w in knots.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    knots[knots.len() - 1].1
}

/// granite 벡터 사이의 코사인을 이 프로세스의 판본에서 예전 순전파 눈금으로 옮깁니다(예전 판본은 그대로).
#[inline]
pub fn calibrate_cos(raw: f32) -> f32 {
    match forward_mode() {
        ForwardMode::Exact => interp_knots(raw, &CLS_TO_LEGACY_COS),
        ForwardMode::Legacy | ForwardMode::ExactMean => raw,
    }
}

/// LRU 계층 기본 상한(엔트리). 384×f32 = 1.5KB 이므로 약 38MB.
pub const DEFAULT_LRU_CAP: usize = 24_576;
