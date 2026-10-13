# Mac 검증 체크리스트 — 결과 기록 (docs/mac-verify-prompt.md)

브랜치: logis-center `claude/mac-verify` · 포크 변경은 **모드 B**(패치 시리즈 `patches/candle-rocm-vulkan/`).

## 상태 표기

| 표기 | 뜻 |
|---|---|
| `[x]` | 실행했고 통과 (근거: 명령 · 수치) |
| `[~]` | 일부만 검증 (리눅스만 / 빌드만 / 목업만) — 나머지는 ⏳ |
| `[ ]` | 미실행 |
| `[!]` | 결정 · 수정이 필요했던 항목 (무엇을 정했는지 함께 기록) |
| ⏳ | push 후 macOS 워크플로(`.github/workflows/mac-verify.yml`)가 돌려야 하는 부분 |

**이 세션의 제약 (실행 못 한 이유)**

- GitHub 접근 없음: `gh api repos/sungu-che/logis-center` → `HTTP 403 GitHub access to this repository is not enabled for this session`,
  `git push` → `sungu-che/logis-center is not in this session's authorized repository set` (포크도 같은 오류).
  그래서 **macOS 러너 실행 결과는 0건**입니다. 아래 macOS 항목은 모두 ⏳ 이고 통과로 세지 않습니다.
- Hugging Face 차단: `huggingface.co:443 connect_rejected` → 실모델 없이 **목업 모델**(`tests/fixtures/granite_mock`)로만 검증.
  판정 일치(질의 150 × 뱅크 170 1위)는 목업에서는 의미가 없으므로 **'못 함 (목업만)'** 입니다(목업 결과는 참고용으로만 기록).
- 리눅스 VM: Intel Xeon (Cascade Lake, AVX-512) 4 vCPU @ 2.8 GHz, KVM. L1d 32 KB · L2 1 MB (코어별) · L3 33 MB (공유).
  VM 이라 시간 측정 편차가 큽니다(같은 측정 반복 시 ±15~40%). 배수는 같은 실행 안의 비교만 씁니다.

---

## 0. 현재 상태 (작업 전)

- Mac 검증 0건. Metal 배치 순전파 OFF, Mac CPU · MoltenVK 는 코드상 ON 이지만 실행 검증 0건.
- 작업 중 발견한 Mac 빌드/실행 차단 요인 (모두 이 브랜치에서 수정, 아래 1·2·3 참고):
  1. `onnxruntime-sys` 가 Apple Silicon(aarch64) 프리빌트 triplet 이 없어 **macOS arm64 빌드에서 panic**.
  2. 포크 Vulkan 백엔드가 `VK_KHR_portability_enumeration` 을 켜지 않아 **최신 로더에서 MoltenVK 장치가 안 보임**.
  3. 앱 `utils::new_gpu_device` 에 Metal 분기가 없어 **Metal 빌드의 GPU 테스트가 조용히 건너뜀**.
  4. 테스트 모델 경로가 `%LOCALAPPDATA%` 전용 → Mac/리눅스에서 **조용히 건너뜀**.
  5. Metal `Device::mem_info` 가 `None` → Mac 에서 KV-PLAN 이 free 메모리를 모름.

## 1. 환경 (GitHub Actions)

- [!] **러너 선택**: `macos-15` (주 실행) + `macos-14` (환경 기록). 첫 잡 `probe` 가 `system_profiler SPDisplaysDataType`,
  Swift `MTLCreateSystemDefaultDevice()` (+ `recommendedMaxWorkingSetSize` · `currentAllocatedSize`), `vulkaninfo --summary` 를 기록하고,
  포크 `examples/device_probe.rs` 가 candle `Device::new_metal(0)` / `Device::new_vulkan(0)` 으로 작은 matmul 을 실제로 돌려
  `DEVICE_PROBE metal OK|FAIL` 을 남깁니다. FAIL 이면 그 백엔드의 실행 단계는 `NOT RUN — no GPU on hosted runner` 로 기록되고
  빌드 검증만 합니다(요약표 `summary.md`, 통과로 세지 않음). → ⏳ 결과 대기. **실행 검증이 안 되면 self-hosted Apple Silicon 러너 필요.**
- [~] **CPU 정보 기록**: 리눅스 `lscpu` + `/sys/devices/system/cpu/cpu0/cache/index*`:
  L1d 32K / L1i 32K / L2 1024K (코어별) / L3 33792K (cpu 0-3 공유), 4코어 1스레드/코어, avx512f·fma. ⏳ macOS: `sysctl hw.perflevel*/l1/l2/cachelinesize` (probe 잡 `cpu.txt`, 앱 CPU 잡 `cpu-caches.txt`).
- [~] **Rust · Xcode CLT · MoltenVK**: 워크플로가 `dtolnay/rust-toolchain@stable`, `brew install molten-vk vulkan-loader vulkan-tools`,
  `VK_ICD_FILENAMES=$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json`. SIP 가 보호된 실행기에서 `DYLD_*` 를 지우므로
  포크 로더가 `/opt/homebrew/lib` · `/usr/local/lib` · `$VULKAN_SDK/lib` 의 `libvulkan.dylib` 를 직접 찾도록 고쳤습니다(패치 0002). ⏳
- [x] **모델 파일 경로**: `tests/it/common.rs` 에 `models_dir()` 추가 — `LOGIS_MODELS_DIR` 우선, 없으면 앱과 같은 `dirs::data_local_dir()/logis-center/models`
  (Windows `%LOCALAPPDATA%` · macOS `~/Library/Application Support` · Linux `~/.local/share`). `app_embed.rs granite_dir`, `p2b_models.rs models_dir`,
  `mem_lifecycle.rs` 가 이를 씁니다. `LOGIS_REQUIRE_MODELS=1` 이면 모델이 없을 때 **실패**(CI 는 이 값으로 돌려 건너뜀이 통과로 세어지지 않게 함).
  실모델은 워크플로가 Hugging Face 에서 받아 `actions/cache` 로 보관(실패 시 `NOT RUN (mock only)`). 목업은 `gen_granite_mock.py` 로 매번 같은 바이트 생성.
- [~] **앱 크레이트 macOS 빌드**: 리눅스에서 `cargo test --test it --no-default-features --no-run` 통과. Windows 전용 의존성 점검(코드 감사):
  `direct-storage` · `windows` 는 `[target.'cfg(windows)']`, `SetProcessWorkingSetSizeEx` 등 Win32 호출 13곳 모두 `cfg(target_os="windows")`,
  `malloc_trim` 은 `cfg(linux)`, macOS 는 `malloc_zone_pressure_relief`, DXGI 보정은 `cfg(all(windows, feature="rocm"))`, `direct_loader` 는 macOS 에
  Metal IO 구현. `windows-sys` 는 무조건 의존이지만 macOS 에서 컴파일만 되고 쓰이지 않음.
  **수정**: `onnxruntime-sys` 에 `(MacOs, Arm64)` → `onnxruntime-osx-arm64-1.28.0.tgz` (자산 존재 확인: HTTP 302) 추가. ⏳ `--features metal` / `--features vulkan` / CPU 의 macOS 빌드.

## 2. 포크(candle-rocm-vulkan) 빌드·테스트

- [ ] ⏳ `cargo test -p candle-core --features metal --no-run`, `-p candle-nn --features metal --no-run` (워크플로 `fork (metal)` 잡).
- [~] **Vulkan 테스트**: 리눅스 lavapipe(Mesa llvmpipe, `CANDLE_VULKAN_ALLOW_CPU=1`)에서
  `cargo test --release -p candle-core --features vulkan --test vulkan_tests fixed_exec_modes_are_batch_invariant` → **ok**.
  `--features vulkan` 전체 `cargo check --tests --examples` 통과(패치 0002 포함). ⏳ MoltenVK 실행(`fork (vulkan)` 잡: `vulkan_tests` 전체 ·
  `fixed_exec_modes_are_batch_invariant` · `vulkan_accel_tests` · candle-nn `accel_vulkan` · `w8a16_generic_cuda::vulkan`).
  코드 감사: Auto 모드는 unified memory(INTEGRATED 또는 HOST_VISIBLE|DEVICE_LOCAL) 이면 native 원소 커널 기본 ON, 가중치는 미러 없이 host 버퍼를 직접 바인딩,
  `b·m ≤ 8` 이면 GEMV. Gpu 모드는 native 원소 커널 OFF(→ CPU 경로) · 가중치 matmul 은 항상 dense GEMM · M=1 은 2행 복제.
  **주의(발견)**: Gpu 모드에서도 `GpuQ8::matmul` 은 `m ≤ 8` 이면 GEMV(공유메모리 리덕션)라 **Q8_0 가중치는 배치 불변이 아님** — 임베딩(F32 dense)과는 무관, 기록만.
- [x] **CPU 행 결과의 M · 위치 · 스레드 무관성 (x86 AVX-512)**: 새 테스트 `candle-core/tests/matmul_m_invariance_tests.rs` (패치 0001).
  `cargo test --release -p candle-core --test matmul_m_invariance_tests -- --nocapture`:

  | 모양 (K→N) | M=2..300 중 M=300 과 다른 M | M=1 | 역순 위치 불일치 | 2행 위치 불일치 |
  |---|---|---|---|---|
  | 384→1152 (Wqkv) | 0 | **다름** (gemv 경로) | 0/300 | 0 |
  | 384→384 (Wo) | 0 | **다름** | 0/300 | 0 |
  | 384→3072 (Wi) | 0 | **다름** | 0/300 | 0 |
  | 1536→384 (Wo) | 0 | **다름** | 0/300 | 0 |

  배치 attention matmul `[B,12,L,32]@[B,12,32,L]` · `[B,12,L,L]@[B,12,L,32]` (L=3,16,64,130, B=1..24): 배치 원소 0 이 B 와 무관(불일치 0).
  스레드 수: `RAYON_NUM_THREADS=1,2,3,4` 모두 digest `20e4a1376e76b382` (동일).
  원인(gemm 0.19 코드): M ≤ 1 이면 `mixed_gemv_rowmajor`(k 를 8개 누산기로 나눠 트리 합), M ≥ 2 는 패킹 마이크로커널(kc 안에서 순차 FMA) — split-K 없음,
  스레드는 (행, 열) 타일로만 나눔. x86 에는 `m·n ≤ 256` 수평 커널 경계도 있지만 임베딩 Linear(N ≥ 384)에는 해당 없음.
  **임베딩은 M=1 이 될 수 없음**: 모든 텍스트가 `[CLS] … [SEP]` 로 최소 3토큰 → per-text M=L ≥ 3, 배치 M=B·L ≥ 3. ⏳ aarch64 NEON(`fork (cpu)` 잡, 같은 테스트 + 1스레드/전체 digest 비교).
- [x] **Metal GEMM 이 M 에 따라 커널을 바꾸는지 (코드 확인)**: candle-metal-kernels 0.11 (MLX 포트, MPS 아님) `mlx_gemm.rs`:
  `m == 1 || n == 1` → `call_mlx_gemv` (배치 접기 전에 판정), `select_tile_config`: `m < 16` → `32x32x16_2_2`, 그 밖은 장치 계열(Phone/Pro/Max/Ultra) ·
  dtype · nn/nt · `is_large = b·m·n ≥ 2^20` 에 따라 BK 16/32 타일이 바뀜, split-K 는 사용 안 함(`should_use_split_k` dead code).
  → **같은 행도 M 에 따라 k 누산 순서가 바뀔 수 있음 ⇒ Metal 은 고정 모양 필요**(3-3). ⏳ 실측: `metal_matmul_m_invariance_report` (`--ignored`, Metal 장치 필요).

## 3. 앱 검증 — 임베딩 배치 순전파

### 3-1. CPU (리눅스 x86 ✔ · macOS arm64 ⏳)

- [x] **app_embed (LOGIS_TEST_EMBED_DEVICE=cpu, 목업 가중치)** — 명령:
  `LOGIS_TEST_MODELS=1 LOGIS_REQUIRE_MODELS=1 LOGIS_MODELS_DIR=<mock> LOGIS_TEST_EMBED_DEVICE=cpu cargo test --test it --no-default-features -- app_embed::gpu_embed_many_matches_per_text_and_is_faster --ignored --nocapture`
  (텍스트 320건: ≤64토큰 304 · 65~512 16 · 토큰 중앙값 5 · 최대 333 / 마이크로배치 46개, 패딩 0, 연산토큰 = 실토큰)

  | 항목 | Windows CPU (Ryzen 5600X, 실모델) | **리눅스 x86 VM (목업) 기본 경로** | 리눅스 x86 VM (목업) `LOGIS_EMBED_CPU_KERNEL=packed` | macOS arm64 |
  |---|---|---|---|---|
  | per-text 와 비트 동일 | 320/320 | **320/320** | **320/320** | ⏳ |
  | 구성 무관(단건 · 표본 · 역순 절반) | 비트 동일 | **12/12 · 12/12 · 160/160** | **12/12 · 12/12 · 160/160** | ⏳ |
  | 재현성 per-text 2회 · embed_many 2회 | — | 320/320 · 320/320 | 320/320 · 320/320 | ⏳ |
  | 전체 속도 (embed_many / per-text) | 1.6배 | **2.2배** (18.91 s vs 42.46 s) | 1.5배 (12.74 s vs 19.52 s) | ⏳ |
  | 짧은 구 304건 | 2.8배 | **4.1배** (7.74 s vs 31.91 s) | 2.4배 (4.02 s vs 9.56 s) | ⏳ |
  | 단건 호출 (짧은 구 20건 평균) | 같은 비용 | per-text 86.5 ms · embed_many 98.0 ms (**0.88배, 아래 주석**) | 27.9 ms · 27.8 ms (같은 비용) | ⏳ |
  | F32 기준 평균 코사인 (≤64 / >64) | 기준 통과 | 1.000000 / 1.000000 (CPU 는 기준 자체) | 1.000000 / 1.000000 | ⏳ |
  | 판정 일치 150 × 170 | — | 150/150 (**목업 · 참고용**) | 150/150 (목업) | ⏳ 실모델 |

  단건 호출 0.88배: VM 측정 편차 범위(같은 측정 반복 시 ±15% 이상)와 겹치고, packed 경로에서는 27.9 vs 27.8 ms 로 같습니다.
  두 경로는 같은 순전파 연산열(B=1 · L 동일)이라 차이는 텍스트 토큰화·버킷 오버헤드뿐입니다. ⏳ macOS 에서 같은 항목 재측정.
- [~] **동적 모드가 Apple Silicon 에서도 이득인지**: 리눅스 x86 에서는 동적 모드 유지가 맞음(위 표). 워크플로가 macOS 에서
  동적(`embed-cpu-mock`) vs 고정 모양 강제(`LOGIS_EMBED_FIXED_SHAPE=1`, `embed-cpu-mock-fixed`) 를 같은 잡에서 비교합니다. ⏳ 결과에 따라
  `BatchOpts::for_device` 를 아키텍처별로 나눌지 결정(지금은 변경 없음).
- [x] **판정 수준 일치 (리눅스 CPU, 목업)**: 새 테스트 `app_embed::embed_many_decisions_match_per_text` — 아래 3-5 참고.

### 3-2. CPU 커스텀 커널 · 캐시 계층 (L1/L2/L3/SLC)

측정 도구: `commerce/tools/cpu-embed-bench` (앱 FloatModel 을 연산 그대로 복제 + 연산 범주별 시간, 패킹 GEMM 스윕, 비트 동일 검사, 전체 순전파 비교).
`cargo run --release --manifest-path commerce/tools/cpu-embed-bench/Cargo.toml -- <granite dir>` · 커널: `commerce/crates/embed-cpu-kernels`.

- [x] **프로파일 (candle 경로 = 앱의 현재 CPU 경로, 리눅스 x86 4스레드, 목업)**

  | 작업 | ms/순전파 | Linear | RMSNorm | SiLU·곱(GLU) | RoPE·복사 | attn matmul | softmax |
  |---|---|---|---|---|---|---|---|
  | 단건 짧은 구 B=1 L=12 | 45.4 | **79%** | 6% | 5% | 3% | 6% | 1% |
  | 짧은 구 배치 B=85 L=12 | 1298 | 42% | 16% | **26%** | 11% | 3% | 1% |
  | 질의 n-gram B=64 L=6 | 380 | 51% | 18% | 16% | 9% | 4% | 1% |
  | 중간 B=16 L=32 | 490 | 48% | 18% | 17% | 9% | 3% | 3% |
  | 중간 B=8 L=64 | 614 | 45% | 17% | 16% | 9% | 5% | 7% |
  | 중간 B=1 L=96 | 151 | 56% | 13% | 10% | 5% | 6% | 8% |
  | 긴 텍스트 B=1 L=256 | 417 | 37% | 12% | 10% | 6% | 14% | 21% |
  | 긴 텍스트 B=4 L=256 | 2196 | 24% | 8% | 16% | 7% | 18% | 26% |
  | 청크 B=2 L=512 | 2615 | 21% | 7% | 7% | 5% | 16% | **44%** |

  층당 연산 묶음 207개(순전파당, candle 연산 ~730개). **짧은 구 단건은 연산 수·할당보다 Linear(gemm 크레이트의 작은 M 비용)가 지배**합니다(79%).
  배치에서는 원소 연산(RMSNorm 의 powf·mean·sqrt·div·mul, GLU 의 narrow·contiguous·silu·mul)이 할당·메모리 왕복으로 40~50%.
  긴 텍스트는 softmax(candle CPU 구현)가 커집니다.
- [x] **GEMM 블로킹 (패킹 가중치 + MR×NR 레지스터 타일, MC/KC/NC 스윕)**: NR=32 열 패널(k 우선)로 로드 시 한 번 재배치.
  AVX-512 6×32 (zmm 누산기 12개) · AVX2/NEON 3×32. 패널 크기 = K×32×4 B → K=384 에서 48 KB(L1d 32~48 KB 근처, L2 에 상주), K=1536 에서 192 KB(L2).
  A 블록 MC×KC: MC 128 × KC 256 × 4 B = 128 KB → L2(1 MB/코어; Apple P코어 L2 공유 16 MB). 스윕 최적은 대부분 **MC 128 · KC 256 · NC 64**.

  | M | Wqkv 384→1152 | Wo 384→384 | Wi 384→3072 | Wo 1536→384 |
  |---|---|---|---|---|
  | 12 (단건) | 5.4~8.1배 | 5.7~6.8배 | 2.6~5.0배 | 3.6~6.8배 |
  | 256 | 0.82~1.92배 | 0.76~1.23배 | 0.82~1.09배 | 0.82~1.17배 |
  | 1020 | 0.85~0.91배 | 0.94~0.98배 | 0.73~0.84배 | 0.94~1.93배 |

  (gemm 크레이트 대비 패킹 커널 속도, 리눅스 x86 VM 전체 실행 2회의 범위 — VM 편차가 커서 범위로 적음. 예: M=12 Wqkv 461 → 57 µs.) 큰 M 에서는 gemm 크레이트(A 도 패킹)가 아직 빠릅니다. M 에 따라 커널을 고르는 하이브리드는
  **비트 동일 계약을 깨므로 금지**(한 행의 결과가 M 에 의존하게 됨) → 패킹 커널을 모든 M 에 씁니다. ⏳ Apple Silicon(NEON, L1D 128 KB P코어)에서 같은 스윕.
- [x] **레이어 융합**: RMSNorm 한 패스(중간 텐서 6개 → 0), Wi 의 gate·up 패널을 짝지어 SiLU(gate)·up 을 에필로그에서 바로 기록([M, 2I] 중간 텐서와
  narrow·contiguous 2회 제거). 융합 attention(QKV 분리 + RoPE + QKᵀ + softmax + PV 한 번에, 스칼라 구현)도 측정했으나 **채택 안 함**:
  L=12 에서 2.1배였다가 L=64 0.88배 · L=96 0.90배 · L=256 0.50배로 역전(앞선 실행들과 같은 경향). L 은 텍스트의 함수라 L 기준 선택은 구성 무관이지만,
  이득이 작고 불안정해 보류(벡터화한 attention 커널이 필요).
- [x] **가중치 사전 패킹 + 스레드 분할**: 로드 시 1회 패킹(Linear 가중치 F32 복사본 약 113 MB, 목업/실모델 같은 층 모양). 스레드는 NC 열 패널 묶음 단위로
  나누고 각 작업이 모든 행을 처리(가중치 패널 재사용). P/E 코어: candle 의 `get_num_threads` 는 macOS 에서 P코어 수(`hw.perflevel0.logicalcpu`)를 씀.
- [!] **비트 동일 조건 (결정: per-text 와 같은 커널)**: 패킹 커널은 출력 원소마다 k 오름차순 FMA 한 줄(+0 시작)만 씁니다 — 분할-K · 트리 리덕션 없음,
  KC 블록은 부분합을 이어 받음. 그래서 결과가 **M · 행 위치 · MC/KC/NC · 스레드 수 · ISA 와 무관**하고 M=1 특수 경로도 없습니다.
  per-text 와 embed_many 가 **같은 커널**을 쓰므로 둘 사이도 비트 동일. 단언:
  - `embed-cpu-kernels` 단위 테스트 3개: 패킹 GEMM == 스칼라 FMA 체인 **비트 단위 완전 일치**(M 1..37 × 블로킹 4종 × 스레드 1/4, 역순 행), SiLU·곱 에필로그 == 비융합, RMSNorm 행 독립 → `cargo test --release` 3 passed.
  - 앱 `app_embed` (packed) 320/320 + 구성 무관(위 표), 벤치 3절: M=1..300 × (1스레드, MC64/KC128/NC32) · (4스레드, MC7/NC512) 불일치 0, 역순 0/300.
  - 캐시 공간: 패킹 모드는 `device_class = "cpu-f32-packed"` 로 지문이 달라 기존 `cpu-f32` 공간과 섞이지 않음.
- [x] **결과 (리눅스 x86 VM, 목업, 앱 경로 기준)**

  | 방안 | 단건 짧은 구 | 짧은 구 배치 | 긴 텍스트 | F32(candle) 대비 | 비트 동일(구성 무관 · per-text) | 반영 |
  |---|---|---|---|---|---|---|
  | 현재(candle/gemm) | 86.5 ms (per-text) | 7.74 s / 304건 | — | 기준 | ✔ · ✔ | 기본값 |
  | **패킹 GEMM + 융합 RMSNorm/GLU** | **27.9 ms (3.1배)** · 벤치 1.75배 | **4.02 s (1.9배)** · 벤치 1.41~1.44배 | 벤치 L=32~96 1.11~1.47배 · L=256 1.09~1.22배 · L=512 0.99배 | 최소 코사인 1.0000000 (GEMM max\|Δ\| ≤ 2.7e-6) | ✔ · ✔ | **앱에 opt-in** (`LOGIS_EMBED_CPU_KERNEL=packed`) |
  | + 융합 attention | 벤치 2.10배 | 1.54~1.61배 | L=32 1.22배 · L=64 0.88배 · L=96 0.90배 · L=256 0.50~0.62배 · L=512 0.48배 | 1.0000000 | ✔ | 채택 안 함 |

  ("벤치" = `cpu-embed-bench` 4절의 순전파 시간 비, candle 경로 대비. 앱 열은 `app_embed` 의 per-text 시간 기준.)

  기본값으로 켜지 않은 이유: Windows CPU(Ryzen) 기본 동작을 바꾸지 않기 위해(4절), 그리고 Apple Silicon NEON(3×32 타일) 수치가 아직 없음.
  ⏳ macOS CPU 잡이 `embed-cpu-mock-packed` · `cpu-kernel-contract` · `cpu-kernel-bench` 를 돌립니다. Mac · Ryzen 에서 이득이 확인되면 기본 ON 후보.

### 3-3. Metal

- [!] **정책 결정**: Metal GEMM 이 M 에 따라 gemv/타일을 바꾸므로(2절) **고정 모양**(CUDA/ROCm 과 같은 `fixed_shape = true`, 모자란 행은 더미).
  dtype 은 지금처럼 F32(`device_class "metal-f32"`). ⏳ `LOGIS_EMBED_FIXED_SHAPE=0` 강제 실행(`embed-metal-mock-dynamic`)으로 동적 모드가 Metal 에서
  깨지는지 실측(실패하면 고정 모양 결정이 맞다는 근거).
- [x] **batched_enabled · BatchOpts · forward_tag**: `batched_enabled` 에 Metal 분기 추가 — **opt-in** (`LOGIS_EMBED_BATCHED=1` 일 때만, 실행 검증 전이므로 기본은 per-text 유지).
  `BatchOpts::for_device` 는 Metal 을 고정 모양으로(코드상 기존과 같음, 주석 명시). `text_cache::forward_tag` 는 `batched_enabled` + `fixed_shape` 로 정해지므로
  Metal opt-in 시 `granite-r2/fwd-batched/v1` × `metal-f32` 공간 — per-text 공간과 분리됨. `new_gpu_device` 에 Metal 분기 추가(테스트가 Metal 장치를 열 수 있게).
- [ ] ⏳ **실행 가능 러너에서 app_embed**: 워크플로 `app (metal)` 잡이 probe OK 일 때만 실행, 아니면 `NOT RUN (no GPU on hosted runner)`.

### 3-4. Mac Vulkan (MoltenVK)

- [ ] ⏳ **app_embed (--features vulkan, MoltenVK)**: Gpu 모드(`embed-vulkan-mock`) · Host 모드(`LOGIS_EMBED_VK_MODE=host`, `embed-vulkan-mock-host`) 속도 비교,
  per-text 2회 재현성(테스트 출력 '재현성' 줄).
  참고(리눅스 lavapipe, **MoltenVK 대표성 없음** — CPU 형 장치라 host 경로): `CANDLE_VULKAN_ALLOW_CPU=1 … --features vulkan -- app_embed::gpu_embed_many… --ignored`
  → Gpu 모드, 비트 동일 320/320 · 구성 무관 12/12 · 12/12 · 160/160 · 재현성 320/320 · 320/320 · 전체 2.3배 · 짧은 구 3.9배 · 단건 99.6 ms vs 99.0 ms.
  `vulkan_matmul_m_invariance_report` → Gpu/Host/Auto 모두 M≥2 불일치 0, Auto 만 M=1 다름; Gpu/Host digest 가 CPU 와 같음(= lavapipe 에서는 host 경로).
- [ ] ⏳ **unified memory 에서 GPU 원소 커널 고정 모드가 더 빠른지**: 지금 Gpu 모드는 원소 연산을 CPU 로 보냅니다. MoltenVK 결과(Gpu vs Host 시간,
  `fork (vulkan)` 의 M 불변 보고)를 보고 새 `ExecMode` 필요 여부 결정. 코드상 근거: Auto 의 native 원소 커널은 크기 임계(`1<<15` 원소 / `1<<22` FLOP)로
  경로가 갈려 구성 무관이 아님 → 새 모드는 '원소 커널도 항상 GPU' 로 고정해야 함.

## 3-5. 10/12~10/13 작업 중 Mac 미검증 항목

- [~] **Qwen4-Exp**: GPU 테스트를 장치 공통으로 변경 — `qwen4_exp_tiny_gpu_matches_hf` (`cfg(any(cuda, rocm, vulkan, metal))`, `utils::new_gpu_device(0)`,
  `LOGIS_REQUIRE_GPU=1` 이면 장치 없음 = 실패). BF16 top-1 ≥ 0.75 단언은 CUDA/ROCm 만(기준 수치가 있는 곳), Metal/Vulkan 은 측정만.
  리눅스 CPU: `qwen4_exp_tiny_cpu_matches_hf` · `..._cpu_mmap_ple_matches_hf` 통과(전체·증분 max|Δ| ≤ 1e-4, argmax 100%, QSA 마스크 · n-gram id 일치).
  리눅스 Vulkan(lavapipe, `--features vulkan`): 새 `qwen4_exp_tiny_gpu_matches_hf` 통과 — variant_a full |Δ| 2.46e-5 · inc 1.53e-5, variant_b 6.3e-6 · 6.1e-6,
  argmax 100% (dense · mmap PLE 모두), BF16 top-1 90.6% (측정만). 장치 공통 테스트가 CUDA 외 장치에서 도는 것을 확인(MoltenVK/Metal 은 ⏳).
  MoE: 앱은 `indexed_moe_forward`(포크에서 CUDA 외 panic)를 **호출하지 않음** — `qwen4_exp/moe.rs` 는 라우터 → CPU top-k → 전문가별 index_select·matmul·index_add
  (장치 공통). 전문가 matmul 의 M = 그 전문가에 배정된 토큰 수라 CPU/Metal 의 M=1 경로 차이가 데이터 의존적으로 생길 수 있음(정확도 기준 1e-4 안에서는 무관). ⏳ Mac CPU · Metal · MoltenVK.
- [ ] ⏳ **FlashAttention/TurboQuant 정확도 (teacher-forcing KLD · top-1)**: 이 세션에서 미실행. 필요한 것: Qwen4-Exp tiny 로 `CANDLE_FLASH_ATTN` on/off 비교 테스트.
- [~] **융합 GDN**: `fused_gdn_supported` = CPU · CUDA · ROCm(커널 이미지 있을 때)만 true, Metal/Vulkan 은 텐서 경로(`delta_rule.rs:118-128`). ⏳ Mac CPU `delta_rule_tests`(`fork (cpu)` 잡), Metal(`fork (metal)` 잡).
- [ ] ⏳ **Vulkan FA/TurboQuant GLSL (MoltenVK)**: `fork (vulkan)` 잡이 `vulkan_accel_tests` · candle-nn `accel_vulkan` 실행.
- [ ] ⏳ **W8A16 forward_exact (test_device!)**: `w8a16_generic_cpu` / `_metal` / `_cuda::vulkan` — `fork` 잡 3개에 배치.
- [x] **판정 수준 일치 — Stage-1 승자 · NMS · Plinko 슬롯**: 새 테스트 `app_embed::embed_many_decisions_match_per_text` (`--ignored`, 모델 필요).
  per-text(`embed`) vs 배치(앱처럼 질의 스팬을 `embed_many` 로 먼저 묶고 캐시, 처음 보는 텍스트는 단건 `embed_many`). Stage-1 = `LayoutScoreTable` 채점 승자,
  NMS = query_commerce 와 같은 우선순위 → 점수 → 길이 정렬 후 겹침 제거 생존 목록, Plinko = `nl_convert::plinko_game_for_indexing` 확정 속성.
  리눅스 CPU(목업): **Stage-1 승자 154/154 · NMS 생존 목록 4/4 질의 · Plinko 슬롯 32/32 청크 일치** (CPU 동적 모드는 완전 일치를 단언). ⏳ 실모델 · Mac.
- [x] **스트리밍 UTF-8 · WS-TRIM**: 새 테스트 `utils_text::utf8_token_stream_emits_only_complete_chars` — ByteLevel BPE(목업 토크나이저)로 한글 · 이모지 · 일본어가
  토큰 사이에서 쪼개지는 문장 3개, 단독 디코드 시 깨지는 토큰이 실제로 있음을 확인한 뒤 `Utf8TokenStream` 출력에 U+FFFD 없음 + 합 == 전체 디코드. 리눅스 통과, ⏳ Mac(CPU 스위트에 포함).
  WS-TRIM: `legacy_working_set_trim` 의 Win32 호출은 `cfg(target_os="windows")`, macOS 는 `malloc_zone_pressure_relief` — Mac 빌드에서 빠짐(코드 확인).
- [~] **Qwen3.5 KV 계획 v2 (unified memory)**: 포크 Metal `Device::mem_info` 가 `None` 이라 Mac 에서 KV-PLAN 이 free 메모리를 모름 → **패치 0003** 으로
  `(recommendedMaxWorkingSetSize − currentAllocatedSize, recommendedMaxWorkingSetSize)` 반환. 앱 `device_mem_info_of` 가 그대로 받음.
  Metal 코드는 리눅스에서 컴파일할 수 없어 ⏳ macOS `fork (metal)` 빌드가 첫 컴파일 검증입니다. 계층 강제 시 결과 동일 여부는 미실행.

## 4. 회귀 · 메모리

- [x] **Windows 결과 영향**: 기본 동작 변경 없음 — CPU/Vulkan/ROCm/CUDA 정책 그대로. 변경은 (a) Metal opt-in 분기, (b) CPU 패킹 커널 opt-in,
  (c) Vulkan portability 는 Apple 대상에서만 기본 ON(`CANDLE_VULKAN_PORTABILITY` 로 강제 가능), (d) Metal `mem_info`, (e) 테스트 경로 함수(`LOCALAPPDATA` 는
  `dirs::data_local_dir()` 와 같은 값). onnxruntime-sys 는 triplet 1개 추가.
- [ ] ⏳ **작업 후 메모리 복귀** (프로세스 RSS · Metal `currentAllocatedSize`): 미실행. `mem_lifecycle` GPU 테스트는 Qwen3-0.6B 실모델 필요.
- [x] **기존 CPU 테스트**: `cargo test --test it --no-default-features` → **193 passed · 0 failed · 44 ignored** (리눅스, 기본 · `LOGIS_EMBED_CPU_KERNEL=packed` 둘 다).
  `cargo check --test it --no-default-features --features vulkan` 통과(리눅스).
  처음 실행에서 `app_embed::global_text_cache_survives_purge_paths` 가 **main 에서도 실패**(기대 퍼지 지점 3, 실제 4) — 6dc0245f 가 `deep_purge` 분기에
  `note_purge` 를 하나 더 넣었고(캐시는 비우지 않음, 테스트 의도와 일치) 기대값만 낡았음 → 4 로 갱신. `LOGIS_EMBED_CPU_KERNEL=packed` 로도 전체 스위트 통과.

## 5. 완료 기준 (장치마다)

| 장치 | F32 기준 평균 코사인 | 판정 150/150 | 구성 무관 비트 동일 + 재현성 | 속도 (단건 손해 없음 포함) |
|---|---|---|---|---|
| 리눅스 x86 CPU (목업) | 통과(CPU = 기준) | 못 함 (목업만, 참고 150/150) | **통과** | 통과 (단건 0.88배는 VM 편차 범위 — packed 27.9/27.8 ms, lavapipe 99.6/99.0 ms 로 같은 비용) |
| 리눅스 lavapipe Vulkan (목업, 참고용) | 통과 | 못 함 (목업만) | 통과 | 통과 |
| macOS arm64 CPU | ⏳ | ⏳ (실모델 다운로드 시) | ⏳ | ⏳ |
| Metal | ⏳ / 러너에 GPU 없으면 못 함 | ⏳ | ⏳ | ⏳ |
| MoltenVK | ⏳ / 러너에 GPU 없으면 못 함 | ⏳ | ⏳ | ⏳ |

## 실행 방법 (재현)

```bash
# 목업 모델 (매 플랫폼 같은 바이트, sha256 3c920073…ac8b65)
python -I commerce/src-tauri/tests/fixtures/granite_mock/gen_granite_mock.py /tmp/models
# 앱 (포크 경로 의존은 로컬/CI 에서만: .github/scripts/use-fork-path.sh)
cd commerce/src-tauri
LOGIS_TEST_MODELS=1 LOGIS_REQUIRE_MODELS=1 LOGIS_MODELS_DIR=/tmp/models LOGIS_TEST_EMBED_DEVICE=cpu \
  cargo test --test it --no-default-features -- app_embed:: --ignored --nocapture --test-threads 1
# 3-2 벤치
cargo run --release --manifest-path ../tools/cpu-embed-bench/Cargo.toml -- /tmp/models/granite-embedding-97m-multilingual-r2
```

## push · macOS CI 대기 목록

이 세션은 GitHub 에 쓸 수 없어(위 403) 아래가 로컬 커밋으로만 남아 있습니다.

- logis-center `claude/mac-verify` (origin/main `56e1d4c5` 위):
  - `5236b372` ci: mac-verify workflow (macOS CPU / Metal / MoltenVK) + fork patch series (mode B)
  - `679432e8` embedding: opt-in packed CPU kernel (LOGIS_EMBED_CPU_KERNEL=packed) + 3-2 bench
  - `34d2b2c3` mac: onnxruntime arm64 prebuilt, Metal in new_gpu_device, Metal batched embedding opt-in
  - `3f4f5a16` test: model dir for Mac/Linux, seeded mock granite, decision-level + UTF-8 stream tests
  - (이 문서 커밋)
- 포크 패치 시리즈 `patches/candle-rocm-vulkan/` (포크 master `f107a7e` 기준, `git am` 으로 적용 확인):
  - 0001-tests-matmul-row-invariance-M-1.300-position-threads.patch
  - 0002-vulkan-find-MoltenVK-on-macOS-portability-enumeratio.patch
  - 0003-metal-Device-mem_info-from-the-recommended-working-s.patch
- push 후 `mac-verify` 워크플로가 자동 실행(push: `claude/mac-*`). 결과는 잡별 아티팩트 `summary.md` 의 PASS / FAIL / NOT RUN.
  필요한 것: Claude GitHub App 을 sungu-che/logis-center (+ sungu-che/candle-rocm-vulkan) 에 설치한 새 세션, 또는 직접 push.
  워크플로 파일 push 에는 `workflow` 권한이 필요합니다.
