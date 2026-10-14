#!/usr/bin/env bash
# logis-center app build + tests for one backend, one phase per workflow step (each step has its own
# timeout and publishes its own annotations, so a slow later phase cannot lose the earlier results).
# Run inside logis-center/commerce.
#   app-tests.sh cpu|metal|vulkan <out-dir> build|suite|embed|bench|real|accel
# Env: MOCK_MODELS (gen_granite_mock.py output root), REAL_MODELS (Hugging Face download root, optional)
set -euo pipefail
BACKEND="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; export OUT; PHASE="${3:-all}"
. "$(dirname "$0")/lib.sh"

case "$BACKEND" in
  cpu)    FEAT=(--no-default-features)
          ACCEL_FEAT=(--no-default-features --features flash-attn,turbo-quant) ;;
  metal)  FEAT=(--no-default-features --features metal)
          ACCEL_FEAT=(--no-default-features --features metal,flash-attn,turbo-quant) ;;
  vulkan) FEAT=(--no-default-features --features vulkan)
          ACCEL_FEAT=(--no-default-features --features vulkan,flash-attn,turbo-quant)
          # SIP strips DYLD_* from protected launchers; set it here so cargo and the test binary see it.
          if command -v brew >/dev/null; then
            export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix vulkan-loader)/lib:$(brew --prefix molten-vk)/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
          fi ;;
  *) echo "unknown backend $BACKEND"; exit 2 ;;
esac

# tauri::generate_context! needs ../dist; bundle.resources "dlls/*" needs one file.
[ -f dist/index.html ] || { mkdir -p dist; echo '<html></html>' > dist/index.html; }
mkdir -p src-tauri/dlls; [ -n "$(ls -A src-tauri/dlls)" ] || touch src-tauri/dlls/.gitkeep
# Keep the app data dir of the runner clean.
XDG_DATA_HOME="$(mktemp -d)"; export XDG_DATA_HOME
cd src-tauri

# embed <label> <models-root> main|compare [extra env...]
#   main    : the device's default policy — bit-identity / invariance / speed test + decision-level test
#   compare : a forced alternative policy — the bit-identity / speed test only (the decision test makes
#             hundreds of single calls; under the fixed shape on a 3-core CPU that alone takes hours)
embed() {
  local label="$1" root="$2" kind="$3"; shift 3
  local tests=(app_embed::gpu_embed_many_matches_per_text_and_is_faster)
  [ "$kind" = main ] && tests+=(app_embed::embed_many_decisions_match_per_text)
  run "$label" env LOGIS_TEST_MODELS=1 LOGIS_REQUIRE_MODELS=1 LOGIS_MODELS_DIR="$root" "$@" \
    cargo test --test it "${FEAT[@]}" -- "${tests[@]}" --ignored --nocapture --test-threads 1
  require_measured "$label" "\\[EMBED-SUMMARY\\]"
}
CPU_ENV=(); [ "$BACKEND" = cpu ] && CPU_ENV=(LOGIS_TEST_EMBED_DEVICE=cpu)

case "$PHASE" in
build)
  git -C ../../../candle-rocm-vulkan log --oneline -n 1 > "$OUT/fork-head.txt" 2>/dev/null || true
  grep -n "candle-rocm-vulkan" Cargo.toml > "$OUT/cargo-fork-lines.txt" || true
  run build cargo test --test it "${FEAT[@]}" --no-run
  ;;
suite)
  # regression: the non-ignored suite (CPU paths; GPU modules for metal/vulkan builds)
  run suite cargo test --test it "${FEAT[@]}" -- --test-threads 2
  if [ "$BACKEND" = cpu ]; then
    run suite-packed env LOGIS_EMBED_CPU_KERNEL=packed cargo test --test it "${FEAT[@]}" -- --test-threads 2
  else
    run qwen4-exp-gpu env LOGIS_REQUIRE_GPU=0 cargo test --test it "${FEAT[@]}" -- qwen4_exp_tiny_gpu_matches_hf --nocapture
    require_measured qwen4-exp-gpu "QWEN4-EXP\\] +variant_a on (Metal|Vulkan)"
  fi
  ;;
embed)
  # [EMBED-BATCHED] mock model: default policy (main) + forced alternatives (compare)
  embed "embed-$BACKEND-mock" "$MOCK_MODELS" main ${CPU_ENV[@]+"${CPU_ENV[@]}"}
  case "$BACKEND" in
    cpu)
      embed embed-cpu-mock-packed "$MOCK_MODELS" compare ${CPU_ENV[@]+"${CPU_ENV[@]}"} LOGIS_EMBED_CPU_KERNEL=packed
      embed embed-cpu-mock-fixed "$MOCK_MODELS" compare ${CPU_ENV[@]+"${CPU_ENV[@]}"} LOGIS_EMBED_FIXED_SHAPE=1 ;;
    metal)
      embed embed-metal-mock-fixed "$MOCK_MODELS" compare LOGIS_EMBED_FIXED_SHAPE=1 ;;
    vulkan)
      embed embed-vulkan-mock-host "$MOCK_MODELS" compare LOGIS_EMBED_VK_MODE=host ;;
  esac
  ;;
bench)
  [ "$BACKEND" = cpu ] || { echo "bench phase: cpu only"; exit 0; }
  sysctl -a 2>/dev/null | grep -E 'hw\.(perflevel|l1|l2|l3|cachelinesize|ncpu)' > "$OUT/cpu-caches.txt" || lscpu > "$OUT/cpu-caches.txt" || true
  # [MAC-SSD-IO] KV block SSD offload: parallel pread/pwrite + F_NOCACHE vs std::fs
  run kv-io-bench env LOGIS_IO_BENCH_MB=512 cargo test --test it "${FEAT[@]}" -- direct_loader::kv_block_io_bench --ignored --nocapture --test-threads 1
  require_measured kv-io-bench "\\[KV-IO\\] best"
  # 3-2: packed-kernel contract + CPU profile / custom kernels (mock weights)
  run cpu-kernel-contract cargo test --release --manifest-path ../crates/embed-cpu-kernels/Cargo.toml
  run cpu-kernel-bench cargo run --release --manifest-path ../tools/cpu-embed-bench/Cargo.toml -- \
    "$MOCK_MODELS/granite-embedding-97m-multilingual-r2" --quick
  ;;
real)
  # real weights: decision-level checks (bank top-1 150 x 170) under the default policy
  if [ -f "${REAL_MODELS:-/nonexistent}/granite-embedding-97m-multilingual-r2/model.safetensors" ]; then
    embed "embed-$BACKEND-real" "$REAL_MODELS" main ${CPU_ENV[@]+"${CPU_ENV[@]}"}
    if [ "$BACKEND" = cpu ]; then
      embed embed-cpu-real-packed "$REAL_MODELS" compare ${CPU_ENV[@]+"${CPU_ENV[@]}"} LOGIS_EMBED_CPU_KERNEL=packed
    fi
  else
    not_run "embed-$BACKEND-real" "real granite weights could not be downloaded (mock only)"
  fi
  ;;
accel)
  # Mock / tiny-model checks of the acceleration paths (docs/mac-verify.md 3-5). Each test changes process-wide
  # state (GDN override, accel config) or measures device-wide memory, so each runs alone (--test-threads 1).
  # Fused GDN (CPU kernel) vs the tensor path on CPU and on this build's GPU: numbers + time.
  run gdn-path cargo test --test it "${FEAT[@]}" -- qwen4_exp::gdn_fused --ignored --nocapture --test-threads 1
  require_measured gdn-path "\\[GDN-PATH\\] CPU"
  if [ "$BACKEND" != cpu ]; then
    # Device memory reported / returned (Metal: fork patch 0003), KV-PLAN inputs, KV tiers on the GPU.
    run gpu-mem cargo test --test it "${FEAT[@]}" -- gpu_memory:: --ignored --nocapture --test-threads 1
    require_measured gpu-mem "\\[GPU-MEM\\] .*total"
  fi
  if [ "$BACKEND" = vulkan ]; then
    # Auto (per-op GPU element-wise) vs the default Gpu mode: speed only — Auto is not batch-invariant.
    embed embed-vulkan-mock-auto "$MOCK_MODELS" compare LOGIS_EMBED_VK_MODE=auto LOGIS_EMBED_MEASURE_ONLY=1
  fi
  # FlashAttention / TurboQuant KV on/off teacher forcing on a seeded mock Qwen3 (KLD, top-1). The app ships with
  # both features on; this is a second build of the app with them.
  run accel-build cargo test --test it "${ACCEL_FEAT[@]}" --no-run
  run accel-acc cargo test --test it "${ACCEL_FEAT[@]}" -- accel_accuracy --ignored --nocapture --test-threads 1
  require_measured accel-acc "\\[ACCEL-ACC\\]"
  ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
finish
