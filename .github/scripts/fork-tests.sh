#!/usr/bin/env bash
# Fork (candle-rocm-vulkan) build + tests for one backend. Run inside the fork checkout.
#   fork-tests.sh cpu|metal|vulkan <out-dir>
set -euo pipefail
BACKEND="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; export OUT
. "$(dirname "$0")/lib.sh"
echo "fork HEAD $(git rev-parse --short HEAD)" | tee "$OUT/fork-head.txt"
git log --oneline -n 15 >> "$OUT/fork-head.txt"

case "$BACKEND" in
cpu)
  sysctl -a | grep -E 'hw\.(perflevel|l1|l2|l3|cachelinesize|ncpu)|machdep\.cpu' > "$OUT/cpu-caches.txt" || true
  run probe-cpu cargo run --release -p candle-core --example device_probe
  # 2-3: row results independent of M (1..300), row position and thread count (NEON gemm on arm64).
  run m-invariance-1thread env RAYON_NUM_THREADS=1 cargo test --release -p candle-core --test matmul_m_invariance_tests -- --nocapture --test-threads 1
  digest=$(grep -oE 'digest [0-9a-f]{16}' "$OUT/m-invariance-1thread.log" | head -1 | cut -d' ' -f2 || true)
  if [ -n "$digest" ]; then
    run m-invariance-allthreads env CANDLE_MATMUL_INVARIANCE_DIGEST="$digest" cargo test --release -p candle-core --test matmul_m_invariance_tests -- --nocapture --test-threads 1
  else
    not_run m-invariance-allthreads "no digest from the 1-thread run"
  fi
  run delta-rule-cpu cargo test --release -p candle-core --test delta_rule_tests -- --nocapture
  run w8a16-generic-cpu cargo test --release -p candle-core --test quantized_tests w8a16_generic_cpu -- --nocapture
  run matmul-tests-cpu cargo test --release -p candle-core --test matmul_tests
  run candle-nn-cpu-flash-attn cargo test --release -p candle-nn --test cpu_flash_attn
  ;;
metal)
  run build-core-metal cargo test --release -p candle-core --features metal --no-run
  run build-nn-metal cargo test --release -p candle-nn --features metal --no-run
  run probe-metal cargo run --release -p candle-core --features metal --example device_probe
  if grep -q "DEVICE_PROBE metal OK" "$OUT/probe-metal.log"; then
    run m-invariance-metal cargo test --release -p candle-core --features metal --test matmul_m_invariance_tests metal_ -- --ignored --nocapture
    run w8a16-generic-metal cargo test --release -p candle-core --features metal --test quantized_tests w8a16_generic_metal -- --nocapture
    run delta-rule-metal cargo test --release -p candle-core --features metal --test delta_rule_tests -- --nocapture
    run matmul-tests-metal cargo test --release -p candle-core --features metal --test matmul_tests
  else
    for s in m-invariance-metal w8a16-generic-metal delta-rule-metal matmul-tests-metal; do
      not_run "$s" "no Metal device on hosted runner (see probe-metal.log)"
    done
  fi
  ;;
vulkan)
  run build-core-vulkan cargo test --release -p candle-core --features vulkan --test vulkan_tests --test vulkan_accel_tests --test quantized_tests --test matmul_m_invariance_tests --no-run
  run build-nn-vulkan cargo test --release -p candle-nn --features vulkan --test accel_vulkan --no-run
  run probe-vulkan cargo run --release -p candle-core --features vulkan --example device_probe
  if grep -q "DEVICE_PROBE vulkan OK" "$OUT/probe-vulkan.log"; then
    run vulkan-tests cargo test --release -p candle-core --features vulkan --test vulkan_tests -- --nocapture --test-threads 1
    run fixed-exec-modes cargo test --release -p candle-core --features vulkan --test vulkan_tests fixed_exec_modes_are_batch_invariant -- --nocapture --exact
    run vulkan-accel-tests cargo test --release -p candle-core --features vulkan --test vulkan_accel_tests -- --nocapture --test-threads 1
    run nn-accel-vulkan cargo test --release -p candle-nn --features vulkan --test accel_vulkan -- --nocapture --test-threads 1
    run w8a16-generic-vulkan cargo test --release -p candle-core --features vulkan --test quantized_tests w8a16_generic_cuda::vulkan -- --nocapture
    run m-invariance-vulkan cargo test --release -p candle-core --features vulkan --test matmul_m_invariance_tests vulkan_ -- --ignored --nocapture
  else
    for s in vulkan-tests fixed-exec-modes vulkan-accel-tests nn-accel-vulkan w8a16-generic-vulkan m-invariance-vulkan; do
      not_run "$s" "no Vulkan (MoltenVK) device on hosted runner (see probe-vulkan.log)"
    done
  fi
  ;;
*) echo "unknown backend $BACKEND"; exit 2 ;;
esac
finish
