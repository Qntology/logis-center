[Execution environment and CI loop — read this first]
You are running in a Claude Code cloud session: an Ubuntu 24.04 x86_64 VM (about 4 vCPU, 16 GB RAM, 30 GB disk). It is NOT a Mac. Split the work like this:
- Linux CPU verification: run it directly in this VM (cargo build/test here).
- macOS verification (Apple Silicon CPU, Metal, MoltenVK/Vulkan): run it on GitHub-hosted macOS runners through GitHub Actions. Push commits, let the workflow run, read the results, fix, push again.

If GitHub access is missing, do not stop. A checkout without a remote, a push refused with "not in this session's authorized repository set", or `gh api` returning 403 all mean GitHub access is missing. In that case:
- Say so once at the start, with the exact error. The fix on my side: this session was started with `claude --cloud` from my terminal, and that uploads a bundle instead of cloning when the Claude GitHub App is not installed on the repository. I install the app on sungu-che/logis-center (and sungu-che/candle-rocm-vulkan) and start a new session.
- Then keep going with everything that does not need GitHub access, in this order:
  1. The mock model generator (item 10) and the test model-path fix (`LOGIS_MODELS_DIR`).
  2. The Linux CPU verification and the CPU kernel / cache-hierarchy experiments (section 3-2).
  3. The code changes the checklist calls for, including the `.github/workflows/mac-verify.yml` files.
- Commit locally on the working branches (`git clone` of the public fork still works for reading). If this checkout has no remote, add `origin` = https://github.com/sungu-che/logis-center and branch from `origin/main` if it can be fetched.
- At the end, list exactly what is waiting for push and macOS CI.

How to drive GitHub Actions from this session:
1. Workflow file: create or update `.github/workflows/mac-verify.yml` in the repo you are testing (runs-on: macos-15, plus macos-14 if useful). Trigger it with `on: push` on your working branch and `on: workflow_dispatch`. Give each job a hard `timeout-minutes`, cache `~/.cargo/registry`, `~/.cargo/git` and `target/` with actions/cache, and upload test output as an artifact (actions/upload-artifact) in addition to printing it to the job log.
2. Pushing: git push goes through the GitHub proxy. Push to a working branch (for example `claude/mac-verify`), never to master/main. Pushes that touch `.github/workflows/` need the `workflow` permission. If such a push is rejected, stop and tell me; I will commit the workflow file myself. Do not try to work around it.
3. Reading results: use the REST-based gh commands only, because GraphQL-based ones (`gh pr ...`, `gh issue ...`) return 403 here:
   - `gh run list --repo OWNER/REPO --branch <branch> --limit 5`
   - `gh run view <run-id> --repo OWNER/REPO` (status) and `gh run view <run-id> --repo OWNER/REPO --log-failed` / `--log` (logs)
   - `gh run download <run-id> --repo OWNER/REPO` (artifacts)
   - `gh workflow run mac-verify.yml --repo OWNER/REPO --ref <branch>` (manual dispatch)
   - or `gh api repos/OWNER/REPO/actions/runs?branch=<branch>` and `.../actions/jobs/<job-id>/logs`
   `GH_TOKEN`/`GITHUB_TOKEN` read as a placeholder (`proxy-injected`). The proxy adds the real credential, so do not pass that value to other tools.
4. Waiting for CI: a single shell command times out after 2 minutes by default, up to 10 minutes on request. The VM pauses after a few minutes of inactivity, and background commands are lost when it resumes. Do NOT run one long blocking wait (for example `gh run watch` for 40 minutes, or a `sleep` loop). Poll instead: call `gh run view <id> --json status,conclusion` every few minutes. In between, do useful work such as the Linux CPU experiments.
5. Two repositories: logis-center is the repository attached to this session (started with `claude --cloud` from my logis-center checkout). The fork is a second repository you clone yourself:
   - Setup at the start: `git clone https://github.com/sungu-che/candle-rocm-vulkan ../candle-rocm-vulkan` (sibling of the logis-center checkout; the fork is public, so cloning works). Create a working branch there too, for example `claude/mac-verify`.
   - Local builds in this VM: point the app at the sibling fork with path dependencies. Use the commented `path = "../../../candle-rocm-vulkan/..."` lines in `commerce/src-tauri/Cargo.toml`: the `candle-rocm` dependency plus the three `[patch.crates-io]` lines. Do not commit that path switch.
   - Pushing fork changes: try `git push` of the fork working branch once. A terminal-started cloud session attaches only one repository, so the proxy may refuse it with "not in this session's authorized repository set". Pick mode A or B below by the result, and say which one you use.
   - Mode A (the fork push works): push fork changes to the fork working branch. On the app working branch, point the four Cargo.toml lines at `git = "https://github.com/sungu-che/candle-rocm-vulkan", branch = "<fork working branch>"` (or `rev = "<fork sha>"`). Push the fork branch first, so the app CI builds exactly that commit. Fork test suites can also run from a fork-side `.github/workflows/mac-verify.yml`.
   - Mode B (the fork push is refused): this is the expected case. Do not stop. Do everything from logis-center:
     - Keep the fork changes as commits on the local fork working branch.
     - Export them with `git -C ../candle-rocm-vulkan format-patch master..HEAD -o <logis-center>/patches/candle-rocm-vulkan/`. Replace the folder contents each time, so it always holds the full series from fork master.
     - Commit the patches on the logis-center working branch.
     - The logis-center workflow does the fork setup itself. Check out logis-center with `actions/checkout` `path: logis-center`, and the fork with `repository: sungu-che/candle-rocm-vulkan`, `ref: master`, `path: candle-rocm-vulkan`. That makes the path dependencies `../../../candle-rocm-vulkan` resolve. Then apply the series with `git -C candle-rocm-vulkan am ../logis-center/patches/candle-rocm-vulkan/*.patch` (set a git user name/email in the job first). Switch the four Cargo.toml lines to the path form inside the job only (sed), not in the commit. Build and test the app against that patched fork, and run the fork test suites on it in the same workflow.
     - The fork is public, so the runner can check it out without extra tokens.
   - Do not merge any working branch into master/main. When you finish, report:
     - the logis-center working branch and its last commit;
     - the fork working branch and its last commit (mode A), or the patch series path and the list of patches (mode B).
     I will put the fork changes into sungu-che/candle-rocm-vulkan and Qntology/candle-rocm-vulkan by PR. The app's main keeps `git = "https://github.com/Qntology/candle-rocm-vulkan", branch = "master"`.
6. Network: this VM's default allowlist covers GitHub, crates.io and static.rust-lang.org, but NOT huggingface.co. For Linux CPU tests that need the embedding model, try the download. If it is blocked, tell me to add `huggingface.co` (and `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) under Network access → Custom. Downloads that happen inside the macOS runner (brew, Hugging Face, LunarG) are not affected by this VM's allowlist.
7. Metal on GitHub-hosted runners: macOS runners on Apple Silicon are virtual machines, and Metal / MPS GPU access may be unavailable there. The very first workflow run must check it: `system_profiler SPDisplaysDataType`, a tiny Swift/ObjC or Rust check of `MTLCreateSystemDefaultDevice()`, and `candle_core::Device::new_metal(0)`. Do the same for MoltenVK (`vulkaninfo --summary`). If either is unavailable, report it with the evidence. Continue with build-only verification for that backend and mark its runtime items as "not run (no GPU on hosted runner)". Never count a skipped run as passed.
8. Evidence: for every checklist item, report the run URL (or local command), the exact numbers, and pass/fail. Commit the filled-in checklist (for example `docs/mac-verify.md` in logis-center on the working branch) so the history keeps it.
9. Parallel work with subagents, one branch per backend: split the macOS work across subagents. For example, one each for Mac CPU, Metal, and MoltenVK/Vulkan, plus one for the Linux CPU kernel/cache experiments in this VM.
   - Each subagent works in its own git worktree (`git worktree add ../wt-mac-metal -b claude/mac-metal`) with its own `CARGO_TARGET_DIR`. Subagents share this VM, so a shared checkout or target dir would clobber each other.
   - Each subagent pushes only its own branch, and its workflow runs only for that branch.
   - Each subagent polls only its own runs (`gh run list --branch <its branch>`) and reports results to you.
   - You merge the findings into one checklist.
   - Expect macOS runner concurrency limits: runs may queue, so do not treat "queued" as a failure.
   - Keep only one subagent editing a given file at a time. Shared files such as `batched.rs` and `embedding.rs` are edited by you after the subagents report.
10. Mock models are allowed and preferred for verification. This is verification, not real inference.
   - What it covers: bit-identity, batch-composition invariance, row-count (M) invariance, determinism and relative speed. These do not need the real weights.
   - How to build it: generate a mock embedding model with the same architecture as granite-embedding-97m-multilingual-r2. Same config.json shape: hidden_size 384, num_hidden_layers, num_attention_heads, intermediate_size, norm_eps and pad_token_id from the real config if you can fetch it; otherwise use the values the code expects. Use the same tensor names as `models/embedding.rs` loads. Use seeded random weights, a reduced vocab size (e.g. 32k instead of 250k: compute per token is unchanged because the vocab only affects the embedding lookup), and a small `tokenizer.json` (e.g. a WordLevel or BPE tokenizer built with the `tokenizers` Python package) that yields realistic token lengths. Include short phrases of 2–64 tokens and long texts of 65–512 tokens.
   - Where to build it: generate it here in the Linux VM with Python (numpy, safetensors and tokenizers are on PyPI, which the default allowlist allows) or deterministically inside a Rust test. Either commit a small generator script (like the existing `tests/fixtures/qwen4_exp_tiny/gen_tiny_ref.py`) or a seeded in-test builder, so the macOS runner can rebuild the identical model without downloading anything.
   - Make the test model path configurable (`LOGIS_MODELS_DIR` or a test-only override) so `app_embed` runs on the mock.
   - Label results clearly as mock or real. Decision-level checks (bank top-1 agreement: query 150 × bank 170) are only meaningful with the real weights. Run those on the macOS runner, which can download from Hugging Face (only this VM's allowlist blocks it). If the real model cannot be obtained, mark them "not run (mock only)".

[요청]
GitHub Actions(macOS 러너)와 이 리눅스 클라우드 세션에서 Mac 검증 및 개선을 진행해줘. 로그는 Actions 에서 확인할 수 있으니, 실행 결과를 읽으면서 아래 체크리스트를 순서대로 검증하고 개선해줘.

목표: logis-center 의 [EMBED-BATCHED] 배치 순전파를 Mac 의 CPU 모드 · Metal 모드 · Mac Vulkan(MoltenVK) 모드에서도 기본으로 쓸 수 있게 만든다. 지금 Metal 은 비활성(per-text)이고, Mac 의 CPU · Vulkan 은 한 번도 검증하지 않았다.
- Vulkan 에 별도 커널이 필요하면 candle-rocm-vulkan 에서 개발한다.
- CPU 에 별도 커널이 필요하면 개발한다. 특히 CPU 커스텀 커널로 배치 순전파를 가속하는 방안을 찾아줘. L1 · L2 · L3(· SLC/L4) 캐시 계층을 활용할 수 있는지(GEMM 블로킹, 타일 크기, 가중치 재사용) 검토하고 측정해줘.
- CPU 는 리눅스 클라우드 세션(x86)과 GitHub macOS(Apple Silicon) 양쪽에서 각각 검증·개선해야 한다.

저장소(각 repo 에 맞게 반영, 필요하면 커밋해서 Actions 로 돌리며 검증):
- https://github.com/sungu-che/candle-rocm-vulkan (포크, master)
- https://github.com/sungu-che/logis-center (앱, main). Cargo.toml 은 포크를 git = https://github.com/Qntology/candle-rocm-vulkan (master) 로 참조한다. 포크를 고치면 sungu-che 에 push 한 뒤, 앱 검증은 [patch] 를 sungu-che git(또는 rev)로 잠시 돌려서 하고, 결과를 보고해줘(Qntology 반영은 내가 PR 머지로 한다).

[이미 Windows 에서 끝난 것 — 이 구조를 그대로 따르고 같은 기준으로 비교]
- logis-center/commerce/src-tauri/src/models/embedding.rs: batched_enabled() 는 CUDA/ROCm/CPU/Vulkan 기본 ON, Metal 만 OFF.
- models/embedding/batched.rs: BatchOpts::for_device(). CUDA/ROCm = 고정 모양(fixed_shape=true, 모자란 행은 더미). CPU/Vulkan = 동적 모드(fixed_shape=false, exact_len_max=512: 같은 길이끼리만 묶고 패딩 없음).
  - 근거: CPU 는 연산량이 곧 시간이라 고정 모양이 오히려 0.4배로 느렸다. CPU gemm 은 행 결과가 행 수(M ≥ 2)·행 위치와 무관해 동적 모드에서도 구성 무관 비트 동일. 단, M = 1 은 다른 경로를 타서 비트가 달라짐(Vulkan 고정 모드는 1행을 2행으로 복제해 처리).
- 포크 candle-core/src/vulkan_backend/shaders.rs: scoped_exec_mode(ExecMode::{Auto,Host,Gpu}). Vulkan 백엔드는 연산마다 GPU/CPU 를 크기·가중치 사용 횟수로 골라(CANDLE_VULKAN_MIRROR_AFTER=2, GEMV/GEMM 전환, 실패 시 조용한 CPU 폴백) 같은 텍스트도 비트가 달라졌다. 배치 구간은 Gpu 모드(가중치 행렬곱은 항상 GPU GEMM, 나머지 CPU)로 묶는다. 테스트: candle-core/tests/vulkan_tests.rs 의 fixed_exec_modes_are_batch_invariant.
- 캐시 공간 태그: text_cache.rs 의 FORWARD_TAG_BATCHED(고정 모양) / FORWARD_TAG_BATCHED_DYN(동적). 캐시는 RAM 전용.
- 검증 테스트: commerce/src-tauri/tests/it/app_embed.rs 의 gpu_embed_many_matches_per_text_and_is_faster (LOGIS_TEST_MODELS=1, --ignored). LOGIS_TEST_EMBED_DEVICE=cpu 면 CPU 로 돈다. 출력: 속도(per-text vs embed_many, 짧은 구, 단건 호출), 최소 코사인, 비트 동일 수, 구성 무관성(단건 · 표본 · 역순 절반), 재현성, F32 기준 코사인, 판정 일치(질의 150 × 뱅크 170 1위).
- Windows 기준 수치(비교용): CPU(Ryzen 5600X) 전체 1.6배 · 짧은 구 2.8배 · 단건 같은 비용 · per-text 와 320/320 비트 동일. Vulkan(RX 6600) 1.7배 · 3.7배 · 구성 무관 비트 동일. ROCm 1.8배.
- 임베딩 판정 기준(확정): per-text 대비 최소 코사인 0.9999 는 쓰지 않는다. 'embed_many 의 F32 기준 평균 코사인 ≥ per-text 평균 − 2e-4'(테스트가 이미 단언) + 구성 무관 비트 동일 + 판정 일치.

[Mac 검증 체크리스트]
상태 표기: [ ] 미실행 · [!] 먼저 결정/수정 필요. 결과는 항목마다 수치와 함께 이 체크리스트 형식으로 기록해 보고해줘(커밋하거나 PR 설명에 남겨도 됨).

## 0. 현재 상태
- Mac 에서는 아무것도 검증하지 않았다. Metal 은 배치 순전파 OFF(per-text), Mac Vulkan(MoltenVK)·Mac CPU 는 코드상 배치 ON 이지만 실행 검증 0건.

## 1. 환경 (GitHub Actions)
- [!] 러너 선택: macos-14 / macos-15 (Apple Silicon arm64). GitHub 호스티드 Apple Silicon 러너는 VM 이라 Metal(MPS) GPU 를 쓸 수 없을 가능성이 높다. 첫 워크플로에서 MTLCreateSystemDefaultDevice / candle Device::new_metal(0) 가 되는지, system_profiler SPDisplaysDataType 출력을 먼저 확인하고, 안 되면 Metal 은 빌드·컴파일 검증만 하고 실행 검증은 self-hosted Mac 러너가 필요하다고 보고해줘(추측으로 통과 처리하지 말 것).
- [ ] CPU 정보 기록: sysctl -a | grep -E 'hw.(perflevel|l1|l2|l3|cachelinesize|ncpu)|machdep.cpu' (Apple Silicon 은 P/E 코어별 L1/L2, L3 대신 SLC). 리눅스 세션은 lscpu, /sys/devices/system/cpu/cpu0/cache/index*/size.
- [ ] Rust stable, Xcode CLT. Mac Vulkan 은 brew install molten-vk vulkan-loader (또는 LunarG Vulkan SDK), VK_ICD_FILENAMES 확인, vulkaninfo --summary. VM 에서 MoltenVK 가 Metal 장치를 못 잡으면 Metal 과 같은 이유로 실행 불가 → 그대로 보고.
- [ ] 모델 파일: 테스트는 %LOCALAPPDATA%/logis-center/models 를 본다(LOCALAPPDATA 는 Mac/리눅스에 없음 → 지금은 조용히 건너뜀!). granite-embedding-97m-multilingual-r2(model.safetensors · tokenizer.json · config.json)를 Hugging Face 에서 받아 캐시(actions/cache)하고, 테스트의 모델 경로 함수(app_embed.rs granite_dir, p2b_models.rs models_dir, mem_lifecycle.rs)를 LOGIS_MODELS_DIR 환경변수 또는 OS 별 데이터 경로로 찾게 고쳐줘. 모델이 없어 건너뛴 실행을 통과로 세지 말 것.
- [ ] logis-center 앱 크레이트가 macOS 에서 빌드되는지: cargo test --test it --no-default-features --features metal --no-run, --features vulkan, (CPU) --no-default-features. Windows 전용 의존성(direct-storage, windows, DXGI 보정 등)은 cfg(windows) 인지, onnxruntime · boa 등 로컬 크레이트가 mac 에서 빌드되는지 확인.

## 2. 포크(candle-rocm-vulkan) 빌드·테스트
- [ ] cargo test -p candle-core --features metal --no-run, -p candle-nn --features metal --no-run (러너에서 Metal 셰이더 컴파일 포함)
- [ ] cargo test -p candle-core --features vulkan --test vulkan_tests (MoltenVK). 특히 fixed_exec_modes_are_batch_invariant. MoltenVK 는 unified memory(integrated)라 Auto 모드 기본이 native 이고 경로 결정이 Windows 이산 GPU 와 다르다 → Gpu/Host 모드가 Mac 에서도 구성 무관인지.
- [ ] CPU: 행 결과의 M·행 위치 무관성(aarch64 NEON gemm 커널과 x86 AVX 커널 각각). M = 1 만 다른지, 다른 M 도 달라지는지 행 수 1..300 으로 측정.
- [ ] Metal 행렬곱(MPS / candle 자체 GEMM)이 행 수(M)에 따라 커널을 바꾸는지 코드로 확인하고, 실행 가능하면 같은 행이 M 에 따라 비트가 바뀌는지 측정(고정 모양이 필요한지 판단 근거).

## 3. 앱 검증 — 임베딩 배치 순전파
### 3-1. CPU (리눅스 세션 x86 + GitHub macOS arm64 각각)
- [ ] app_embed 테스트(LOGIS_TEST_EMBED_DEVICE=cpu): per-text 와 비트 동일 수(Windows 는 320/320), 구성 무관성, 속도(전체 · 짧은 구 · 단건)를 Windows 수치와 표로 비교.
- [ ] 동적 모드가 Apple Silicon 에서도 이득인지(메모리 대역폭·코어 구성이 다름). 아니면 장치/아키텍처별 정책을 BatchOpts::for_device 에 반영.

### 3-2. CPU 커스텀 커널 · 캐시 계층(L1/L2/L3/SLC) 가속 검토
- [ ] 프로파일: 배치 순전파에서 시간이 어디에 드는지(Linear GEMM / attention matmul / RMSNorm · softmax · silu 등 원소 연산 / 할당·복사). 짧은 구는 연산량보다 연산 수(약 730 op)와 할당이 지배하는지 확인.
- [ ] GEMM 블로킹: 임베딩 모델은 hidden 384, Wqkv 384×1152, MLP 가중치 수 MB. 가중치 패널이 L2 에, 활성 행 블록이 L1 에 들어가도록 MC/KC/NC 를 정하면 현재 gemm 크레이트 대비 빨라지는지. Apple Silicon(L1D 128KB P코어, L2 공유 12~16MB, SLC) / x86(L1 32-48KB, L2 0.5-2MB, L3) 각각 측정.
- [ ] 레이어 융합: RMSNorm + Linear, Linear + SiLU·곱(GLU), QKV 분리 + RoPE 등을 캐시에 남은 데이터로 한 번에 처리하는 커스텀 CPU 커널. 중간 텐서 할당 · 메모리 왕복 감소 측정.
- [ ] 가중치 사전 패킹(로드 시 한 번 패널 레이아웃으로) + 스레드 분할(행 블록 기준, P/E 코어 고려).
- [!] 비트 동일 조건 유지: 커스텀 커널도 원소마다 k 누산 순서가 행 수·행 위치·스레드 수와 무관해야 한다(분할-K 금지, 고정 순서 리덕션). 이걸 깨면 캐시 공간이 오염된다. per-text 경로와 같은 커널을 쓰면 per-text 와도 비트 동일을 유지할 수 있다 — 어느 쪽인지 정하고 테스트로 단언.
- [ ] 결과: 방안별 속도(짧은 구 · 긴 텍스트 · 단건) · 정확도(F32 기준 코사인) · 비트 동일 여부 표. 효과 있는 것만 포크(또는 앱)에 반영.

### 3-3. Metal
- [ ] Metal 에서 배치 순전파를 켤지: 고정 모양(GPU 처럼) vs 동적. Metal GEMM 이 M 에 따라 커널을 바꾸면 고정 모양 또는 경로 고정(Vulkan 의 scoped_exec_mode 같은) 필요. BF16/F32 dtype 정책(지금 Metal 임베딩은 F32, device_class 'metal-f32')도 확인.
- [ ] batched_enabled 에 Metal 추가 + BatchOpts::for_device 정책 + text_cache forward_tag 정합.
- [ ] 실행 가능 러너에서 app_embed 테스트: 속도, 구성 무관 비트 동일, F32 기준 코사인, 판정 일치. 실행 불가면 빌드까지만 하고 그렇다고 보고.

### 3-4. Mac Vulkan (MoltenVK)
- [ ] app_embed 테스트(--features vulkan): 구성 무관 비트 동일(Gpu 모드), Host 모드와 속도 비교, per-text 2회 재현성(Windows 에서는 per-text Auto 경로가 319/320 만 같았음).
- [ ] unified memory 라 Auto 의 native 원소 커널이 켜진다 → 배치 구간 Gpu 모드에서 원소 연산은 CPU 로 가는데, Mac 에서는 GPU 원소 커널을 고정 사용하는 모드가 더 빠른지(새 ExecMode 필요 여부) 측정.

## 3-5. 10/12~10/13 작업 중 Mac 에서 검증하지 않은 항목 (커밋 기록 대조로 찾은 누락분)
- [ ] Qwen4-Exp (logis-center ca3242fa · 5cea2adf · f4c6d4e0): tests/it/qwen4_exp.rs 는 tiny 무작위 체크포인트(tests/fixtures/qwen4_exp_tiny, variant_b)를 쓰므로 모델 다운로드 없이 검증할 수 있다. 지금 GPU 테스트는 cfg(feature = "cuda") + Device::new_cuda 전용이라 Metal/Vulkan/ROCm 은 0건. GPU 테스트를 장치 공통(Metal · Vulkan · ROCm 도 받도록)으로 고치고, Mac CPU · Metal · MoltenVK 에서 전체·증분 logits max|Δ| ≤ 1e-4, argmax 100%, QSA 선택 마스크 · n-gram id 완전 일치를 확인. MoE 가 CUDA 전용 경로(indexed_moe_forward 등)를 쓰는지 확인하고, 쓰면 Metal/Vulkan 폴백이 맞게 도는지 측정.
- [ ] FlashAttention/TurboQuant 정확도 (logis-center 0d3ef235 '정확도 저하 이슈 있음' → f54732e9 디버깅, 해결 여부를 앱 단에서 확인한 기록 없음): Mac 에서 같은 프롬프트로 CANDLE_FLASH_ATTN on/off(및 TurboQuant KV on/off) 생성 결과를 teacher-forcing 으로 비교(KLD, top-1). Metal 은 융합 어텐션이 없어 언퓨즈드 폴백이므로 그 폴백 경로의 정확도도 같은 방법으로 확인. 모델은 Qwen4-Exp tiny 체크포인트나 목업으로 해도 된다(목업이면 '목업'이라고 표기).
- [ ] 융합 Gated DeltaNet (포크 929d276, 앱 f4c6d4e0): fused_gdn_supported 는 CPU/CUDA/ROCm 만 true 라 Metal/Vulkan 의 Qwen3.5 · Qwen4-Exp 는 텐서 경로다. Mac CPU 에서 delta_rule_tests(fused CPU 경로) 통과 확인, Metal/Vulkan 은 텐서 경로 vs CPU fused 수치 비교, 그리고 Metal/Vulkan 전용 GDN 커널이 필요한지(속도 측정 근거) 판단.
- [ ] Vulkan FA/TurboQuant GLSL 포팅 (포크 b7a744a · ae72946 · eeb9564): MoltenVK 에서 candle-nn accel_vulkan, candle-core vulkan_accel_tests (GQA-packed decode 비트 동일, paged vs cat, FP8 fast path 비트 동일, split 규칙). Windows Vulkan(RX 6600)에서는 3개 통과.
- [ ] W8A16 forward_exact 의 백엔드 공통 테스트 (포크 0f551df, quantized_tests 의 w8a16_generic: test_device! 로 CPU/Metal/Vulkan 생성): Mac CPU · Metal · MoltenVK 에서 실행. Metal/Vulkan 경로는 역양자화 + matmul.
- [ ] 판정 수준 일치 — Stage-1 승자 · NMS 결과 · plinko 슬롯 (logis-center 8fc21066): 원래 AMD 체크리스트에도 있었으나 아직 어떤 GPU 테스트도 이 셋을 비교하지 않는다. CPU/Vulkan 배치 순전파를 켰으므로 per-text 와 embed_many 경로에서 이 셋이 같은지 확인하는 테스트를 추가(임베딩 벡터가 비트 동일한 CPU 는 완전 일치가 기대값).
- [ ] 스트리밍 UTF-8 · 작업 집합 반환 스위치 (logis-center f4c6d4e0 [UTF8-STREAM] · [WS-TRIM]): Mac 에서 표준출력 스트리밍이 완성된 UTF-8 만 내보내는지(한글·이모지), WS-TRIM 은 Windows 전용 API 이므로 Mac 빌드에서 cfg 로 빠지는지 확인.
- [ ] Qwen3.5 KV 계획 v2 (logis-center 387258f2): GDN 상태 VRAM 유지, VRAM 내 재압축, 벡터별 FP8 강등, SSD RAM 예산, F16 원본 KV. Windows 에서는 KV 계층 강제를 Qwen3 로만 했다. Mac(unified memory: VRAM 과 RAM 이 같은 물리 메모리)에서 KV-PLAN 판정이 맞는지(Metal 의 free 메모리 산정, recommendedMaxWorkingSetSize), 계층 강제 시 결과 동일 여부를 Qwen4-Exp tiny 또는 목업으로 확인.

## 4. 회귀 · 메모리
- [ ] Windows 결과가 바뀌지 않는지: CPU/Vulkan/ROCm 정책을 바꾸면 그 이유와 영향 명시.
- [ ] 작업 후 메모리 복귀(프로세스 RSS, Metal 은 MTLDevice.currentAllocatedSize).
- [ ] 기존 CPU 테스트(cargo test --test it, ignored 제외) 전부 통과.

## 5. 완료 기준 (장치마다 모두)
- 수치: F32 기준 평균 코사인 기준 통과, 판정 150/150(또는 per-text 와 같은 수준)
- 구성 무관 비트 동일(단건 · 표본 · 역순 절반) + embed_many 재현성 100%
- 속도: per-text 대비 이득(단건 호출 손해 없음 포함)
- 실행 못 한 항목은 '못 함 + 이유'로 남기고 통과로 세지 않는다
