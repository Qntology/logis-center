#!/usr/bin/env bash
# logis-center app build + tests for one backend. Run inside logis-center/commerce.
#   app-tests.sh cpu|metal|vulkan <out-dir>
# Env: MOCK_MODELS (gen_granite_mock.py output root), REAL_MODELS (Hugging Face download root, optional)
set -euo pipefail
BACKEND="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; export OUT
. "$(dirname "$0")/lib.sh"

case "$BACKEND" in
  cpu)    FEAT=(--no-default-features) ;;
  metal)  FEAT=(--no-default-features --features metal) ;;
  vulkan) FEAT=(--no-default-features --features vulkan)
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
git -C ../../../candle-rocm-vulkan log --oneline -n 1 > "$OUT/fork-head.txt" 2>/dev/null || true
grep -n "candle-rocm-vulkan" Cargo.toml > "$OUT/cargo-fork-lines.txt" || true

run build cargo test --test it "${FEAT[@]}" --no-run
if grep -q "FAIL" <(grep "^| build |" "$SUMMARY"); then finish; exit 1; fi

# ── regression: the non-ignored suite (CPU paths; GPU modules for metal/vulkan builds) ──
run suite cargo test --test it "${FEAT[@]}" -- --test-threads 2
if [ "$BACKEND" = cpu ]; then
  run suite-packed env LOGIS_EMBED_CPU_KERNEL=packed cargo test --test it "${FEAT[@]}" -- --test-threads 2
fi

# ── GPU visibility for this build ──
if [ "$BACKEND" != cpu ]; then
  run qwen4-exp-gpu env LOGIS_REQUIRE_GPU=0 cargo test --test it "${FEAT[@]}" -- qwen4_exp_tiny_gpu_matches_hf --nocapture
  require_measured qwen4-exp-gpu "QWEN4-EXP\\] +variant_a on (Metal|Vulkan)"
fi

# ── [EMBED-BATCHED] app_embed on the mock model (bit-identity · composition invariance · speed) ──
embed() { # embed <label> <models-root> [extra env...]
  local label="$1" root="$2"; shift 2
  run "$label" env LOGIS_TEST_MODELS=1 LOGIS_REQUIRE_MODELS=1 LOGIS_MODELS_DIR="$root" "$@" \
    cargo test --test it "${FEAT[@]}" -- app_embed::gpu_embed_many_matches_per_text_and_is_faster app_embed::embed_many_decisions_match_per_text --ignored --nocapture --test-threads 1
  require_measured "$label" "\\[GPU-EQ\\] 수치"
}
if [ "$BACKEND" = cpu ]; then
  embed embed-cpu-mock "$MOCK_MODELS" LOGIS_TEST_EMBED_DEVICE=cpu
  # Apple Silicon: is the dynamic mode still the better policy? (fixed shape forced for comparison)
  embed embed-cpu-mock-fixed "$MOCK_MODELS" LOGIS_TEST_EMBED_DEVICE=cpu LOGIS_EMBED_FIXED_SHAPE=1
  # 3-2 opt-in CPU kernel (packed GEMM + fused RMSNorm / SiLU·up) on Apple Silicon NEON
  embed embed-cpu-mock-packed "$MOCK_MODELS" LOGIS_TEST_EMBED_DEVICE=cpu LOGIS_EMBED_CPU_KERNEL=packed
else
  embed "embed-$BACKEND-mock" "$MOCK_MODELS"
  if [ "$BACKEND" = metal ]; then
    # Comparison: the fixed shape (CUDA/ROCm policy) on Metal (measured 0.4x vs dynamic 2.7x).
    embed embed-metal-mock-fixed "$MOCK_MODELS" LOGIS_EMBED_FIXED_SHAPE=1
  fi
  if [ "$BACKEND" = vulkan ]; then
    embed embed-vulkan-mock-host "$MOCK_MODELS" LOGIS_EMBED_VK_MODE=host
  fi
fi

# ── 3-2: CPU profile + cache-blocked custom kernels (tools/cpu-embed-bench, mock weights) ──
if [ "$BACKEND" = cpu ]; then
  sysctl -a 2>/dev/null | grep -E 'hw\.(perflevel|l1|l2|l3|cachelinesize|ncpu)' > "$OUT/cpu-caches.txt" || lscpu > "$OUT/cpu-caches.txt" || true
  # [MAC-SSD-IO] KV 블록 SSD 오프로딩: 병렬 pread/pwrite + F_NOCACHE vs std::fs (차가운 읽기 · 파일 캐시 증가량)
  run kv-io-bench env LOGIS_IO_BENCH_MB=512 cargo test --test it "${FEAT[@]}" -- direct_loader::kv_block_io_bench --ignored --nocapture --test-threads 1
  require_measured kv-io-bench "\\[KV-IO\\] best"
  run cpu-kernel-contract cargo test --release --manifest-path ../crates/embed-cpu-kernels/Cargo.toml
  run cpu-kernel-bench cargo run --release --manifest-path ../tools/cpu-embed-bench/Cargo.toml -- \
    "$MOCK_MODELS/granite-embedding-97m-multilingual-r2"
fi

# ── real weights: decision-level checks (bank top-1 150 × 170) ──
if [ -f "${REAL_MODELS:-/nonexistent}/granite-embedding-97m-multilingual-r2/model.safetensors" ]; then
  if [ "$BACKEND" = cpu ]; then
    embed embed-cpu-real "$REAL_MODELS" LOGIS_TEST_EMBED_DEVICE=cpu
    embed embed-cpu-real-packed "$REAL_MODELS" LOGIS_TEST_EMBED_DEVICE=cpu LOGIS_EMBED_CPU_KERNEL=packed
  else
    embed "embed-$BACKEND-real" "$REAL_MODELS"
  fi
else
  not_run "embed-$BACKEND-real" "real granite weights could not be downloaded (mock only)"
fi

ls -la target/debug/deps/it-* 2>/dev/null | head -3 > "$OUT/binaries.txt" || true
finish
