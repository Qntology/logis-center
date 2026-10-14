# Mac 검증 체크리스트 — 결과 기록 (docs/mac-verify-prompt.md)

- logis-center 작업 브랜치: `claude/mac-verify`
- 포크 작업 브랜치: `sungu-che/candle-rocm-vulkan` `claude/mac-verify`
- 포크 변경은 **모드 A + B**로 둡니다.
  - 포크 브랜치에 push했습니다.
  - CI는 `patches/candle-rocm-vulkan/` 시리즈를 포크 master에 `git am`해서 빌드합니다.
  - 포크 master `4e7ebd0`에는 PR #10으로 처음 3개 커밋(행 불변성 테스트, MoltenVK 로더, Metal mem_info)이 이미 들어 있습니다.
- 러너: GitHub `macos-15` = "Apple M1 (Virtual)" 3코어 VM입니다. 시간 측정은 같은 잡 안의 상대 비교만 씁니다.

## 상태 표기

| 표기 | 뜻 |
|---|---|
| `[x]` | 실행했고 통과 (근거: 런 · 수치) |
| `[~]` | 일부만 검증 (목업만 · 리눅스만 · 빌드만) |
| `[ ]` | 미실행 |
| `[!]` | 결정 · 수정이 필요했던 항목 (무엇을 정했는지 함께 기록) |

**목업 / 실모델**
- 목업은 `tests/fixtures/granite_mock/gen_granite_mock.py`가 만드는 시드 고정 무작위 가중치 모델입니다(같은 구조, vocab 16k, sha256 `3c920073…ac8b65`).
- 실모델 `granite-embedding-97m-multilingual-r2`은 macOS 러너가 Hugging Face에서 받습니다(`actions/cache`).
- 판정 수준 일치(질의 150 × 뱅크 170)는 **실모델 줄만** 근거로 씁니다.

**CI 런**

| 런 | 내용 |
|---|---|
| [38499518771](https://github.com/sungu-che/logis-center/actions/runs/38499518771) | 전체: probe 2, fork 3, app 3 |
| [38506246088](https://github.com/sungu-che/logis-center/actions/runs/38506246088) | 포크 개선 3건 검증: fork Metal/Vulkan, app Metal/Vulkan의 suite · embed · real · accel |
| [38508388995](https://github.com/sungu-che/logis-center/actions/runs/38508388995) | Metal GDN 정밀화(`806aba6`)만 재확인: fork `delta_rule_tests`, app `gdn-path` · `qwen4-exp-gpu` |
| [38508978460](https://github.com/sungu-che/logis-center/actions/runs/38508978460) | Metal 빈 버퍼 수정(`e4a7314`) 후 fork `delta_rule_tests`만 |
| [38513842101](https://github.com/sungu-che/logis-center/actions/runs/38513842101) | 개선 4건(CPU 병렬 원소 연산 · Qwen3.5 KV · 언로드 RSS · Metal conv 상태): fork Metal `delta-rule-metal`, app 3종 suite · embed(기본 정책만) · accel `gdn,gpu-mem,kv35`. CPU 실모델 단계는 사용자가 취소 |
| [38516449786](https://github.com/sungu-che/logis-center/actions/runs/38516449786) | 취소된 CPU 실모델 단계 + 고친 테스트 확인: app CPU suite · real, app Vulkan suite |
| [38520324489](https://github.com/sungu-che/logis-center/actions/runs/38520324489) | CPU 실모델 단건 재측정(번갈아 · 중앙값), `EMBED_COMPARE=speed` |

- 결과는 잡 annotation으로 읽었습니다. 이 세션에서는 로그와 아티팩트 blob이 403이라서입니다.

---

## 1. 환경 (GitHub Actions)

- [x] **러너 · GPU 가시성** (run 38499518771 probe 잡)
  - `macos-15`(15.7.9, Xcode 16.4, rustc 1.98.1):
    - `MTLCreateSystemDefaultDevice` → **"Apple Paravirtual device"**, unified=true, recommendedMaxWorkingSetSize 5,010,800,640
    - candle `Device::new_metal(0)` matmul OK
    - MoltenVK `vulkaninfo`: apiVersion 1.4.357, **INTEGRATED_GPU** "Apple Paravirtual device"
    - → Metal · MoltenVK 모두 **실행 검증 가능**
  - `macos-14`: Metal 장치 없음 → 환경 기록만 합니다.
- [x] **CPU 정보** (`sysctl`, macos-15)
  - Apple M1 (Virtual), `hw.ncpu` 3 (perflevel0만 있음, P/E 구분 없음)
  - L1D 128 KB · L1I 192 KB · L2 12 MB(3코어 공유) · 캐시 라인 128 B · RAM 7.5 GB · SLC 정보 없음
  - 리눅스 VM: Xeon (Cascade Lake, AVX-512) 4 vCPU · L1d 32K · L2 1M · L3 33M
- [x] **Rust · Xcode CLT · MoltenVK**
  - `brew install molten-vk vulkan-loader vulkan-tools`를 쓰고 `VK_ICD_FILENAMES`를 설정합니다.
  - SIP가 `DYLD_*`를 지워서, 포크 로더가 `/opt/homebrew/lib` 등을 직접 찾게 했습니다(포크 `91a7ab2`).
- [!] **앱 빌드 차단 요인과 수정**
  1. `onnxruntime-sys`에 arm64 triplet이 없었습니다 → 추가했습니다.
  2. `.cargo/config.toml`의 `target-cpu=native`가 가상 M1에서 generic aarch64로 떨어져 ring 0.17 const assert가 실패했습니다 → 잡에 `RUSTFLAGS=-C target-cpu=apple-m1`을 넣었습니다(Rust 버전 문제 아님).
  3. lance-encoding이 protoc를 요구합니다 → `brew install protobuf`.
  4. `direct_loader` macOS 경로가 존재하지 않는 Metal IO API를 불러 **컴파일되지 않았습니다** → 다시 작성했습니다(아래 SSD).
  - 결과: `cargo test --test it --no-run`이 CPU · `--features metal` · `--features vulkan` 모두 빌드됩니다.
- [x] **모델 경로**
  - `tests/it/common.rs`의 `models_dir()`: `LOGIS_MODELS_DIR`가 있으면 그것을, 없으면 `dirs::data_local_dir()/logis-center/models`를 씁니다.
  - `LOGIS_REQUIRE_MODELS=1`이면 모델이 없을 때 **실패**합니다. CI의 `require_measured`는 측정 줄이 없는 PASS를 NOT RUN으로 바꿉니다.

## 2. 포크(candle-rocm-vulkan) 빌드·테스트 — run 38499518771

| 잡 | 결과 |
|---|---|
| fork (cpu, NEON) | M-불변성: 1스레드 · 3스레드 digest **`4bfc0461001d5039`** 동일. M≥2는 M=300과 불일치 0, M=1만 다름(gemv). delta_rule 9 pass, w8a16_generic_cpu, matmul 7, candle-nn cpu_flash_attn 7 |
| fork (metal) | core/nn `--features metal` 빌드, device_probe OK. **Metal M-불변성: Linear 불일치 0 · 배치 attention 불일치 0, digest `4bfc0461001d5039`**(CPU와 비트 동일). w8a16_generic_metal, delta_rule 9 pass, matmul 12 pass |
| fork (vulkan, MoltenVK) | device_probe OK. `vulkan_tests` **19/19**, `fixed_exec_modes_are_batch_invariant`, `vulkan_accel_tests` 3/3, candle-nn `accel_vulkan` 4/4, `w8a16_generic_cuda::vulkan`. **Gpu · Host · Auto 모드 M-불변성 불일치 0, digest `4bfc0461001d5039`** |

- [!] **Metal GEMM과 M**
  - 코드상으로는 candle-metal-kernels MLX GEMM이 `m==1`에서 gemv, `m<16`에서 다른 타일을 씁니다.
  - 실측(M 1..300, 가상 M1 GPU)에서는 **M≥2 행이 M과 무관**했고, 결과가 CPU와 비트 동일했습니다.
  - → Metal도 **동적 모드**로 정했습니다(3-3).
- [!] **MoltenVK에서 고친 것**
  - Q8_0 테스트 기준: GPU 경로는 정확한 W8A32이고, CPU는 활성값을 Q8_0로 양자화합니다. 그래서 오차 0.18–0.62는 기준 차이였습니다.
  - f16 `forward_exact`: Apple Silicon의 gemm-f16이 f16으로 누산하던 것을 CPU · Vulkan 모두 f32 누산으로 바꿨습니다(`c1de501`, `a08ba34`).
  - GLSL `barrier()` → 워크그룹 배리어로 바꿨습니다(`c663a23`).

## 3. 앱 검증 — 임베딩 배치 순전파

### 3-1. CPU

| 항목 | Windows (Ryzen, 실모델) | 리눅스 x86 (목업) | **macOS arm64 실모델** | macOS arm64 목업 |
|---|---|---|---|---|
| per-text 비트 동일 | 320/320 | 320/320 | **320/320** | 320/320 |
| 구성 무관 (단건 · 표본 · 역순 절반) | 동일 | 12/12 · 12/12 · 160/160 | **12/12 · 12/12 · 160/160** | 동일 |
| 재현성 (per-text 2회 · embed_many 2회) | — | 320 · 320 | 320 · 320 | 320 · 320 |
| 전체 속도 | 1.6배 | 2.2배 | **0.94배** | 2.69배 |
| 짧은 구 | 2.8배 | 4.1배 | **1.57배** | 1.68배 |
| 단건 (per-text / embed_many) | 같은 비용 | 86.5 / 98.0 ms | **44.4 / 44.9 ms** | 26.0 / 32.6 ms |
| 판정 150 × 170 | — | 목업 150/150 | **150/150 · Stage-1 154/154 · NMS 4/4 · Plinko 25/25** | 150/150 |

- [x] 기존 suite: macOS **194 pass**(기본 · packed 둘 다), 리눅스 193 pass.
- [!] **Apple Silicon에서 동적 모드**
  - 비트 동일성과 판정은 모두 지킵니다.
  - 실모델 전체 속도는 0.94배로 이득이 없지만, 짧은 구에서는 1.57배입니다.
  - 고정 모양 강제(`LOGIS_EMBED_FIXED_SHAPE=1`)는 0.28배이고 단건이 2324.8 ms라 훨씬 나쁩니다.
  - → `BatchOpts::for_device`는 CPU 동적 모드 그대로 둡니다.
  - 비교 런에서 테스트가 "CPU는 per-text와 320/320"을 고정 모양에도 요구해 실패(304/320)했습니다. 고정 모양은 긴 텍스트를 패딩하므로 설계대로입니다. 이 단언은 기본 정책에만 적용하게 고쳤습니다.

### 3-2. CPU 커스텀 커널 · 캐시 계층

- 앱 경로: `crates/embed-cpu-kernels` + `src/models/embedding/cpu_packed.rs`, opt-in `LOGIS_EMBED_CPU_KERNEL=packed`
- 벤치: `tools/cpu-embed-bench`

| 방안 | 리눅스 x86 (AVX-512 6×32) | macOS M1 (NEON 3×32, release 벤치) |
|---|---|---|
| 패킹 GEMM, M=12 Wqkv | 5.4~8.1배 빠름 | **0.35배 (396 vs 137 µs)** |
| 패킹 GEMM, M=1020 Wi | 0.73~0.84배 | 0.11배 |
| 순전파, 단건 B=1 L=12 | 1.75배 | **0.70배 (43.2 vs 30.1 ms)** |
| 순전파, 짧은 구 B=85 L=12 | 1.41배 | 0.21배 |
| 비트 계약 (M 1..300 × 블로킹 × 스레드, 역순 행) | 불일치 0 | **불일치 0** (`cpu-kernel-contract` 3 pass) |
| 앱 실모델 (debug 테스트 빌드) | — | per-text 16.9 ms vs 기본 44.4 ms, 비트 동일 320/320, 판정 150/150 |

- [x] **프로파일 (M1, 목업, candle 경로)**
  - 단건 B=1 L=12: Linear 78%, norm 10%.
  - 짧은 구 배치 B=85: Linear 59%, GLU 15%, norm 15%.
  - 청크 B=2 L=512: softmax 34%, Linear 31%.
  - 층당 연산 묶음 207개입니다.
- [!] **반영 결정: packed는 opt-in으로 유지합니다.**
  - Mac 앱 테스트에서 2.6배 빨라 보였습니다. 하지만 debug 빌드라 앱 크레이트는 opt-level 0이고 커널은 3이어서 생긴 차이였습니다.
  - release 벤치에서는 M1 NEON에서 gemm 크레이트가 더 빠릅니다.
  - macOS 기본 ON을 한 번 넣었다가 되돌렸습니다(`8ff2aeaa`).
  - x86 단건에는 이득이 있지만, Windows 기본 동작을 바꾸지 않으려고 opt-in으로 둡니다.
- [!] **비트 동일 조건**
  - 출력 원소마다 k 오름차순 FMA 체인 하나만 씁니다. 분할-K와 M=1 특수 경로가 없습니다.
  - → M · 행 위치 · 블로킹 · 스레드 · ISA와 무관하고, per-text와 같은 커널입니다. 단위 테스트와 앱 테스트로 단언합니다.

### 3-3. Metal

- [!] **정책: 동적 모드, 기본 ON** (`85919c78`, `batched.rs`: `fixed_shape = !(cpu || vulkan || metal)`)
  - 근거는 Metal M-불변성 실측(불일치 0)과 아래 속도입니다.
  - dtype은 F32이고 `device_class "metal-f32"`입니다.

| macos-15 Metal | 전체 | 짧은 구 | 단건 per-text / embed_many | 비트 동일 | 구성 무관 | 판정 |
|---|---|---|---|---|---|---|
| **실모델, 동적 (기본)** | **2.44배** | **4.77배** | 12.5 / 12.2 ms | 320/320 | 12/12 · 12/12 · 160/160 | **150/150 · Stage-1 154/154 · NMS 4/4 · Plinko 25/25** |
| 목업, 동적 | 2.88배 | 5.56배 | 13.2 / 14.1 ms | 320/320 | 동일 | 150/150 |
| 목업, 고정 모양 (비교) | 0.47배 | 0.46배 | 10.3 / 325.6 ms | 304/320 | 동일 | 150/150 |

- [x] suite 195 pass (Metal 빌드).

### 3-4. Mac Vulkan (MoltenVK)

| macos-15 MoltenVK | 전체 | 짧은 구 | 단건 per-text / embed_many | 구성 무관 | 판정 |
|---|---|---|---|---|---|
| 실모델, Gpu 모드 (기본) | 2.20배 | 5.28배 | 49.0 / **58.6 ms** | 12/12 · 12/12 · 160/160 | 150/150 · Stage-1 154/154 · NMS 4/4 · Plinko 25/25 |
| 목업, Gpu 모드 | 3.50배 | 4.88배 | 53.2 / 59.9 ms | 동일 | 150/150 |
| 목업, Host 모드 (비교) | 1.09배 | 1.98배 | 42.4 / 76.9 ms | 동일 | 150/150 |

- per-text와 비트가 같은 텍스트는 278/320입니다.
  - per-text는 Auto 경로이고 embed_many는 Gpu 고정 모드라서 생기는 차이입니다.
  - 평균 코사인 차는 ≤ 3e-8이고, 재현성은 320/320 · 320/320입니다.
- suite 213 pass.
- [!] **단건 20% 손해 → 포크에서 개선** (`51e2256`, 패치 0006)
  - 원인: 12층 × GPU 제출 1회 더(층당 0.8 ms). Gpu 모드는 모든 가중치 행렬곱을 GPU로 보내 제출과 대기를 매번 기다립니다.
  - 수정: host 공유 메모리에서는 행이 2..=32인 F32 가중치 행렬곱을 CPU에서 계산합니다.
  - 안전장치: 장치 · 형태마다 실제 가중치 16행으로 **CPU 결과가 GPU 커널 결과와 비트 동일할 때만** 이렇게 합니다. 그래서 행 결과는 여전히 배치와 무관합니다. 확인 결과는 `[VK-HOST-SMALL-M]` 줄로 남습니다.
  - **결과(run 38506246088): 실모델 단건 per-text 34.9 ms / embed_many 22.5 ms** — embed_many가 1.55배 빨라졌습니다(이전 58.6 ms, 20% 느림).
    - 목업은 34.7 / 33.4 ms입니다. 전체 2.05배, 짧은 구 4.36배입니다.
    - 구성 무관 12/12 · 12/12 · 160/160, 재현 320/320, 판정 150/150, Stage-1 154/154, NMS 4/4, Plinko 25/25가 모두 유지됐습니다.
    - `[VK-HOST-SMALL-M]`에서 384×1152 · 384×384 · 384×3072 · 1536×384 모두 CPU == GPU로 확인돼 적용됐습니다.
- [!] **Auto 모드(원소 연산도 GPU) 비교**
  - Auto는 크기 임계에 따라 경로가 바뀌어 배치 구성에 따라 비트가 달라질 수 있습니다. 그래서 기본값 후보가 아니고 측정 전용입니다(`LOGIS_EMBED_VK_MODE=auto`, `LOGIS_EMBED_MEASURE_ONLY=1`).
  - 실측(run 38506246088, 목업):
    - 전체 2.15배, 짧은 구 3.20배, 단건 36.1/35.3 ms입니다.
    - 하지만 **구성 무관 1/12 · 1/12 · 121/160**, 비트 동일 34/320입니다. 결정 정책으로 쓸 수 없습니다.
  - 새 ExecMode는 만들지 않았습니다. 단건 손해의 원인은 원소 연산이 아니라 제출 횟수였고, 작은 M 경로로 해결했습니다.

## 3-5. 10/12~10/13 작업 중 Mac 미검증 항목

- [x] **Qwen4-Exp** (tiny 체크포인트, 장치 공통 `qwen4_exp_tiny_gpu_matches_hf`)
  - Metal: full |Δ| 3.37e-5, inc 2.60e-5, argmax 100% (dense · mmap PLE).
  - MoltenVK: full 3.00e-5, inc 2.03e-5, argmax 100%.
  - CPU suite 통과.
  - MoE는 `indexed_moe_forward`(CUDA 전용)를 쓰지 않습니다. 장치 공통 index_select / matmul 경로입니다.
- [x] **FlashAttention / TurboQuant 정확도** — 목업 Qwen3 teacher-forcing (`tests/it/accel_accuracy.rs`, 프리필 64 + 디코드 96)
  - 리눅스 CPU: FA on KLD 9.6e-14, top-1 100%, max|Δ| 2.3e-5. TQ는 CPU에서 비활성입니다.
  - 리눅스 lavapipe Vulkan: FA on KLD 5.7e-14, top-1 100%. TQ KV on 평균 KLD 1.66e-4, 최대 1.55e-2, top-1 100%.
  - Metal: FA가 비융합 폴백뿐이었습니다 → **포크에서 Metal sdpa 융합 커널로 연결**했습니다(`3ad5050`, 패치 0004). 앱 `accel_check` 자가 점검이 Metal도 검사합니다.
  - **Mac (run 38506246088, 목업)**

    | 장치 | FA 자가 점검 | FA on/off max|Δlogit| | KLD | top-1 | TurboQuant KV on |
    |---|---|---|---|---|---|
    | Metal (sdpa) | 통과 (7.0e-7, 5개 형태) | 1.82e-5 | 1.6e-13 | 100% | Metal에는 없음(비활성) |
    | MoltenVK | 통과 (6.8e-7) | 2.19e-5 | 3.7e-14 | 100% | 평균 KLD 1.66e-4, 최대 1.55e-2, top-1 100%, max|Δlogit| 0.80. 자가 점검 K/V fp8 오차 평균 2.36% |
    | Mac CPU | — | 2.46e-5 | 4.8e-14 | 100% | CPU 비활성 |
- [x] **융합 GDN**
  - 포크 CPU `delta_rule_tests` 9 pass (Mac).
  - Metal과 Vulkan에는 융합 경로가 없었습니다. 리눅스 lavapipe 측정에서 텐서 경로는 CPU 융합 대비 프리필 0.09배, 디코드 0.32배였습니다.
  - **포크에서 개선** (`f947435`, 패치 0005):
    - Vulkan(host 공유 메모리)은 CPU 커널을 장치 텐서 위에서 돌립니다.
    - Metal은 네이티브 스캔 커널을 쓰고, CPU 커널과 자가 점검을 통과해야 켜집니다.
    - 포크 `delta_rule_tests`의 GPU 케이스(dk 256, Tiled, conv 2–5, dv 200 등)가 Metal과 Vulkan에서 돕니다.
  - **Mac (Qwen3.5급 형태 Hk 8 · Hv 16 · Dk = Dv 128, 프리필 T=256 + 디코드 16)**

    | 경로 | 프리필 | 디코드/스텝 | CPU 융합 대비 오차 |
    |---|---|---|---|
    | Mac CPU 융합 | 21.0 ms | 0.39 ms | — |
    | MoltenVK 텐서 경로 (이전 기본) | 419.6 ms | 2.03 ms | 4.8e-7 |
    | **MoltenVK 융합 (`f947435`)** | **20.0 ms (20.9배)** | **1.22 ms (1.67배)** | **0 (CPU 커널 그대로)** |
    | Metal 텐서 경로 (이전 기본) | 167.6 ms | 7.60 ms | 6.6e-7 |
    | Metal 융합 1차 커널 (fast math) | 31.9 ms (5.26배) | 3.77 ms (2.02배) | 8.2e-7 |
    | **Metal 융합 정밀화 (`806aba6`, run 38508388995)** | **27.0 ms (같은 잡 텐서 190.5 ms 대비 7.06배)** | **2.09 ms (10.81 ms 대비 5.18배)** | ≤ 7.3e-7 |

  - 포크 테스트:
    - MoltenVK `delta_rule_tests` GPU 케이스 7 pass.
    - Metal 1차 커널은 `gpu_matches_f64_reference` · `half_inputs`가 실패했습니다.
    - `806aba6` 정밀화 후에는 오차가 CPU 대비 0~1.2e-7로 줄었지만, 같은 두 테스트가 남았습니다.
    - 원인은 conv 폭 1 케이스의 **빈 conv 상태 텐서를 Metal로 보낼 때 `newBufferWithBytes`(길이 0)가 nil을 돌려주는 백엔드 문제**였습니다.
    - `e4a7314`로 고친 뒤 **Metal `delta_rule_tests` 7 pass**(run 38508978460)입니다.
  - 앱: Qwen4-Exp tiny가 Metal GDN 커널 경로로 max |Δ| 2.9e-5 · 3.4e-5, argmax 100%.
  - 참고: 이 가상 GPU에서는 Metal 경로가 CPU 융합 커널보다 느립니다. 명령 버퍼 동기화 비용 때문입니다.
- [x] **Vulkan FA/TurboQuant GLSL**: MoltenVK에서 `vulkan_accel_tests` 3/3, `accel_vulkan` 4/4.
- [x] **W8A16 forward_exact**: CPU · Metal · MoltenVK 모두 통과(포크 f32 누산 수정 후).
- [x] **판정 수준 일치**: 새 테스트 `embed_many_decisions_match_per_text`, 실모델 기준.
  - Metal · MoltenVK · Mac CPU 모두 Stage-1 154/154, NMS 4/4, Plinko 25/25입니다.
  - 목업에서는 32/32 청크입니다.
- [x] **스트리밍 UTF-8**: `utf8_token_stream_emits_only_complete_chars`가 Mac suite에서 통과했습니다. WS-TRIM은 `cfg(windows)`이고, macOS는 `malloc_zone_pressure_relief`를 씁니다.
- [~] **KV 계획 v2 (unified memory)**
  - Metal `mem_info`를 recommendedMaxWorkingSetSize − currentAllocatedSize로 바꿨습니다(포크 `30e502f`, master 반영).
  - 새 테스트 `gpu_memory.rs`: 할당과 해제가 free에 반영되는지, KV-PLAN 판정, VRAM/RAM KV 계층 강제 결과가 CPU와 같은지.
    - lavapipe: 계층 강제 VRAM · RAM 모두 CPU 대비 3.8e-6.
  - **Mac (run 38506246088)**

    | 장치 | total | 256 MiB 할당 → 해제 | 모델 언로드 후 장치 free | KV 계층 강제 VRAM · RAM | KV-PLAN 48/512/2048/8192 MB |
    |---|---|---|---|---|---|
    | Metal | 5011 MB | −270 → +268 MB | 시작 대비 +0 MB | CPU 대비 5.7e-6 | Vram · Vram · Vram · Ssd |
    | MoltenVK | 7516 MB (free 5011) | −270 → +268 MB | 시작 대비 +0 MB | CPU 대비 0 | Vram · Vram · Vram · Ssd |

  - 프로세스 RSS는 언로드 후 시작 대비 +146~154 MB 남습니다(할당기 보유분).
- [x] **macOS SSD 오프로드** (`594f5b4e`): 8 MiB × 4스레드 병렬 pread/pwrite, F_NOCACHE, F_PREALLOCATE

| macos-15, 512 MiB 블록 | std::fs | 병렬 F_NOCACHE (기본 4×8MiB) | 최고 (8×32MiB) |
|---|---|---|---|
| 콜드 읽기 | 399 MB/s | **10,148 MB/s (25.5배)** | 12,528 MB/s (31.4배) |
| 쓰기 | 896 MB/s | 962 MB/s (1.07배) | — |
| 파일 캐시 증가 | — | **+0 MB** | 캐시 사용 시 +519 MB |

  리눅스(`LOGIS_DIRECT_LOADER=parallel`)는 읽기 1.50배, 쓰기 1.55배이고 캐시 증가가 +262 MB에서 0 MB로 줄었습니다.

## 4. 회귀 · 메모리 · 최종 런

- [x] **Windows 영향**
  - 기본 동작은 바뀌지 않습니다. 바뀐 기본값은 Metal 동적 모드 ON(Metal 전용)과 macOS direct_loader 백엔드입니다.
  - 포크 변경도 Windows 기본 동작을 바꾸지 않습니다.
    - Vulkan GDN과 작은 M host 경로는 host 공유 메모리 장치에서만 켜집니다. 이산 GPU는 그대로입니다.
    - Metal FA와 GDN은 Metal 전용입니다.
- [x] **포크 개선 검증 런 38506246088**
  - fork Vulkan: 전부 통과.
  - app Metal · app Vulkan: 전부 통과(suite 195 / 213, Qwen4-Exp Metal 2.69e-5 · Vulkan 3.10e-5, argmax 100%).
  - fork Metal: GDN 테스트 2건 실패 → 수정 `806aba6`.
- [x] **Metal GDN 재확인**
  - run 38508388995: app Metal `gdn-path` · `qwen4-exp-gpu` 통과, fork 테스트 2건 실패 → 빈 버퍼 수정.
  - run 38508978460: fork Metal `delta_rule_tests` 7 pass.
- [x] 기존 CPU 테스트: 리눅스 193 pass, macOS 194 pass.

## 5. 완료 기준 (장치마다)

| 장치 | F32 기준 평균 코사인 | 판정 150/150 (실모델) | 구성 무관 + 재현성 | 속도 (단건 손해 없음 포함) |
|---|---|---|---|---|
| macOS CPU | 1.000000 | **통과** | **통과** | 전체 0.94배 · 짧은 구 1.57배 · 단건 44.4/44.9 ms (손해 없음) |
| Metal | 1.000000 | **통과** | **통과** | 2.44배 · 단건 12.5/12.2 ms |
| MoltenVK | 1.000000 (Δ ≤ 3e-8) | **통과** | **통과** | 2.05배 · 짧은 구 4.36배 · **단건 34.9/22.5 ms** (포크 `51e2256` 후, 이전 49.0/58.6) |
| 리눅스 x86 CPU (목업) | 1.000000 | 목업만 | 통과 | 2.2배 |

## 커밋

- **logis-center** `claude/mac-verify`
  - main에는 PR #62로 첫 5커밋(작업 초기 체크리스트까지)이 병합돼 있습니다.
  - 그 뒤 커밋: Metal 동적 기본 ON, macOS SSD 오프로드, 목업 accel 테스트(FA/TQ · GDN · GPU 메모리), CI 단계 · 선택 입력, 포크 패치 시리즈, 이 문서.
- **포크** `sungu-che/candle-rocm-vulkan` `claude/mac-verify`. master `4e7ebd0`(처음 3개는 PR #10으로 반영) 위의 커밋:

  | 커밋 | 내용 |
  |---|---|
  | `c663a23` | vulkan: GLSL barrier() → workgroup barrier |
  | `c1de501` | quantized: CPU f16 forward_exact를 f32로 누산 |
  | `a08ba34` | vulkan tests: GPU-resident Q8_0는 정확한 W8A32 + Vulkan f16 forward_exact f32 누산 |
  | `3ad5050` | metal: FlashAttention을 fused sdpa 커널로 |
  | `f947435` | gated delta rule: Vulkan(host 공유 메모리) · Metal 융합 경로 |
  | `51e2256` | vulkan: 고정 GPU 모드에서 작은 M 가중치 행렬곱을 host로 (비트 일치 시) |
  | `806aba6` | gated delta rule (metal): conv · norm · 게이트를 precise math 커널로 |
  | `e4a7314` | metal: 빈 텐서를 host 데이터에서 만들 때 풀 버퍼 사용 (길이 0 `newBufferWithBytes` nil) |
  | `454e15f` · `1d746d8` | docs/mac-verify.md (개발 환경 · 진행 방식 · 결과 기록) |

- 같은 내용의 패치 시리즈: `patches/candle-rocm-vulkan/` (포크 master 기준 전체 시리즈, CI는 `git am --3way`로 적용).

## 6. 개선 4건 검증 — run [38513842101](https://github.com/sungu-che/logis-center/actions/runs/38513842101)

포크 PR #11 병합으로 패치 0001–0011이 포크 master에 들어가 첫 디스패치(38513670742)는 `git am`에서 멈췄습니다(테스트 미실행).
`apply-fork-patches.sh`가 이미 master에 있는 패치를 patch-id로 건너뛰게 고치고, 시리즈를 master 기준으로 다시 만들었습니다.

| 항목 | 장치 | 결과 |
|---|---|---|
| **Metal GDN conv 상태를 커널이 기록** (포크 `a633c43`, 스텝당 디스패치 2개) | fork Metal | `delta_rule_tests` 7 pass, 자가 점검 max rel 3.3e-7 |
| | app Metal | `gdn-path`: 융합 프리필 23.5 ms · 디코드 1.15 ms/스텝, 같은 잡 텐서 경로 대비 7.15배 · 4.97배(이전 7.06배 · 5.18배), CPU 융합 대비 오차 ≤ 7.3e-7. Qwen4-Exp Metal argmax 100% |
| **CPU 병렬 원소 연산** (`cpu_par`, 같은 비트) | app CPU | suite 195 · packed 195 pass(새 비트 단언 `embedding_cpu_par_ops_match_candle_bits` 포함, aarch64). 목업 embed: 비트 동일 320/320, 구성 무관 12/12 · 12/12 · 160/160, 재현 320/320, 판정 150/150, Stage-1 154/154, NMS 4/4, Plinko 32/32 |
| | | 속도(목업, debug 빌드): 전체 1.90배 · **짧은 구 3.31배**(이전 1.68배) · 단건 54.2 / 66.0 ms(이전 26.0 / 32.6) |
| | app Metal · Vulkan | 같은 코드 경로(GPU는 원래 candle 연산, q/k/v 분리만 복사 6→3회). Metal 목업 320/320 · 판정 150/150 · 2.27배 · 짧은 구 4.09배. MoltenVK 목업 판정 150/150 · 구성 무관 · 재현 유지, 2.93배 · 짧은 구 4.41배 |
| **Qwen3.5 KV 계획 v2** (목업 Qwen3.5, 프리필 1100 + 디코드 4) | app Metal | kv_exact true, Ram↔Vram max\|Δlogit\| 0, GDN 상태 장치 유지 true, SSD 계층 8.39 MB 기록 · 상대 5.26e-4 · top-1 4/4 |
| | app Vulkan | Ram↔Vram 0, GDN 상태 장치 유지 true, SSD 상대 1.05e-3 · top-1 4/4. **압축 KV(TurboQuant)**: kv_exact false, CPU 원본 대비 2.05e-3, top-1 4/4, Ram↔Vram 0 |
| **언로드 RSS** (목업 Qwen3 2사이클, 앱과 같은 mimalloc + `release_os_working_set`) | app Metal | 장치 free 시작 대비 +0 MB. RSS 첫 사이클 잔여 +160 MB(1회 초기화), **둘째 사이클 증가 +23 MB** (기준 ≤ 32 MB) |
| | app Vulkan | 장치 free +0 MB. RSS 첫 사이클 +154 MB, **둘째 사이클 +27 MB** |

- [!] **Vulkan suite 213/214**: 실패 1건은 `mem_lifecycle::half_kv_storage_error_is_small`(CPU 전용, 기존 테스트)입니다.
  - 시드 없는 `randn` 입력이라 무작위 입력 2000개 기준 오차 중앙값 7.7e-4 · p99 1.8e-3 · 최대 2.6e-3으로 약 0.5% 확률로 기준 2e-3을 넘습니다. 이 잡은 2.27e-3이었습니다.
  - 고정 시드(ChaCha8)로 바꿔 매번 6.26e-4입니다. 리눅스 lavapipe `--features vulkan` suite 214/214, **Mac MoltenVK suite 214/214 (38516449786)**.
- 참고 — **목업 CPU 속도**(38513842101): 짧은 구 1.68배 → 3.31배, 전체 2.69배 → 1.90배, 단건 절대 시간 약 2배.
  - 같은 런에서 코드가 바뀌지 않은 CPU 융합 GDN도 이전 21.0 ms → 25.3 ms(Vulkan 잡) · 40.0 ms(Metal 잡)로 러너 CPU 속도가 1.2~1.9배 흔들렸고, 테스트 바이너리 할당기가 시스템 malloc → mimalloc으로 바뀌었습니다. 목업 한 번의 debug 빌드 측정으로는 원인을 가를 수 없었고, 판단은 아래 실모델 런으로 했습니다.
  - **CPU 실모델, 새 코드 (38516449786)**: 비트 동일 320/320(짧은 303 · 긴 17), 구성 무관 12/12 · 12/12 · 160/160, 재현 320/320, **판정 150/150 · Stage-1 154/154 · NMS 4/4 · Plinko 25/25**. suite 195 · packed 195.
    - 속도: **전체 0.94배 → 1.39배, 짧은 구 1.57배 → 3.00배** (per-text 40.59 s → embed_many 29.21 s, 짧은 구 33.70 s → 11.24 s).
    - 단건은 57.0 / 97.6 ms로 나왔지만, 단건 embed_many(B=1, 정확한 길이, 마스크 없음)는 per-text와 같은 연산열입니다. per-text 20회 → embed_many 20회를 몰아서 잰 평균이라 VM 순간 지연이 한쪽에 실린 것으로 보고, 측정을 3바퀴 번갈아 · 중앙값으로 바꿨습니다(리눅스 목업 92.6 / 90.4 ms, 예전 방식 86.5 / 98.0). Mac 재측정: run 38520324489 (`EMBED_COMPARE=speed`, 속도 · 비트 동일 테스트만).
    - **38520324489: 단건 중앙값 per-text 87.9 ms / embed_many 76.8 ms → 단건 손해 없음.** 비트 동일 320/320, 구성 무관 12/12 · 12/12 · 160/160, 재현 320/320, F32 판정 150/150.
    - 이 런은 평균이 193.5 / 179.9 ms일 만큼 순간 지연이 컸고 per-text 순차가 40.59 s → 84.92 s로 느려져, 전체 3.36배 · 짧은 구 6.09배는 부풀려진 값입니다. embed_many 절대 시간은 29.21 s → 25.30 s로 비슷합니다. 개선 폭은 보수적으로 38516449786의 **전체 1.39배 · 짧은 구 3.00배**(이전 0.94배 · 1.57배)로 기록합니다.
- [x] **Mac CPU 배치 속도 (새 코드, 실모델)**: 전체 0.94배 → 1.39배, 짧은 구 1.57배 → 3.00배, 단건 손해 없음, 판정 · 비트 동일 유지.
- 가상 GPU의 동기화 비용은 런마다 크게 다릅니다(MoltenVK GDN 텐서 경로 디코드 2.03 → 3.47 ms, 코드 변경 없음). MoltenVK 단건 47.5 / 67.2 ms(이전 목업 34.7 / 33.4)도 이 영향으로 봅니다. 실제 Apple GPU 수치는 `runner` 입력에 셀프호스티드 Apple Silicon 라벨을 넣어 같은 테스트로 잽니다.
- 결정: `force_flush_all_active_blocks`의 RAM 사본 BF16 변환은 사용자 결정으로 그대로 둡니다(CUDA/ROCm은 FP8, 그 밖은 F32 저장).
- FP8 강등 · VRAM 재압축 · kv_half는 CUDA/ROCm 전용 경로라 Mac · 리눅스 VM에서 실행할 수 없습니다.
