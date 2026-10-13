// =====================================================================
// 🌟 [EMBED-CPU-PACKED] CPU 임베딩 순전파용 패킹 GEMM (opt-in: LOGIS_EMBED_CPU_KERNEL=packed)
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   CPU 배치 순전파 시간의 대부분은 Linear(gemm 크레이트)입니다. 단건 호출(짧은 구, M = L ≈ 12)은
//   Linear 가 78% 이고, gemm 크레이트는 작은 M 에서 패킹·스레드 분배 비용이 연산보다 큽니다.
//   큰 배치에서는 RMSNorm(powf·mean·sqrt·div·mul) · SiLU·곱(narrow·contiguous) 같은 원소 연산이
//   텐서 할당과 메모리 왕복으로 30~40% 를 차지했습니다 (docs/mac-verify.md 3-2, tools/cpu-embed-bench).
//  ── 해결 ──
//   가중치를 로드 시 한 번 NR 열 패널(k 우선)로 다시 배치하고, MR×NR 레지스터 타일로 곱합니다
//   (AVX-512 6×32 · AVX2/NEON 3×32). RMSNorm 은 한 번의 패스, Wi 는 gate·up 패널을 짝지어
//   SiLU(gate)·up 을 에필로그에서 바로 써서 [M, 2I] 중간 텐서를 만들지 않습니다. attention 은 기존 candle 연산.
//  ── 비트 동일 계약 ──
//   출력 원소마다 k = 0..K 를 오름차순으로 이어지는 FMA 한 줄(c = fma(a[k], w[k], c), +0 시작)로만
//   누산합니다. KC 블로킹은 부분합을 저장했다가 이어 받으므로 값이 같고, 분할-K · 트리 리덕션이 없습니다.
//   그래서 한 행의 결과는 그 입력 행만의 함수입니다: M(행 수) · 행 위치 · MC/KC/NC · 스레드 수와 무관하고,
//   gemm 크레이트와 달리 M = 1 특수 경로(gemv)도 없습니다. per-text 와 배치가 같은 커널을 쓰므로
//   이 모드에서도 per-text ↔ embed_many 는 비트 동일입니다(캐시 공간은 device_class "cpu-f32-packed" 로 분리).
//  ── 측정 (Linux x86 VM, Xeon AVX-512 4코어, 목업 가중치) ──
//   GEMM M=12: gemm 크레이트 대비 5~8배 · M=256: 0.8~1.9배 · M=1020: 0.7~1.0배.
//   순전파 전체(패킹 GEMM + 융합 norm/GLU): 단건 L=12 2.0배 · 짧은 구 배치 1.3~1.5배 · 긴 텍스트 1.07~1.24배.
//   Apple Silicon(NEON) 수치는 macOS 워크플로(tools/cpu-embed-bench)에서 확인 후 기본값을 정합니다.
// =====================================================================
pub(crate) use embed_cpu_kernels::*;

/// 층 하나의 패킹 가중치.
pub(crate) struct PackedLayer {
    pub wqkv: PackedW,
    pub attn_wo: PackedW,
    /// Wi (gate | up) 를 짝지은 패널.
    pub wi: PackedW,
    pub mlp_wo: PackedW,
    pub attn_norm: Option<Vec<f32>>,
    pub mlp_norm: Vec<f32>,
}

/// FloatModel 의 CPU 패킹 판본 (Linear 가중치 복사본 + 정규화 가중치).
pub(crate) struct PackedModel {
    pub layers: Vec<PackedLayer>,
    pub emb_norm: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub eps: f32,
}

/// 패킹 경로를 쓸지 (프로세스에서 한 번 읽음). `LOGIS_EMBED_CPU_KERNEL=packed`.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("LOGIS_EMBED_CPU_KERNEL")
            .map(|v| v.trim().eq_ignore_ascii_case("packed"))
            .unwrap_or(false);
        if on {
            println!("[EMBED-CPU-PACKED] CPU 임베딩: 패킹 GEMM + 융합 RMSNorm/SiLU·up (LOGIS_EMBED_CPU_KERNEL=packed, 타일 {}×{})", mr(), NR);
        }
        on
    })
}
