#!/usr/bin/env bash
# 리눅스에서 Rust 통합 테스트(src-tauri/tests/it)와 프론트엔드 테스트(src/tests, vitest)를 실행합니다.
#
#   ./test_linux.sh [vulkan|cuda|rocm] [cargo test 추가 인자...]
#   예) ./test_linux.sh vulkan                 # 기본 스위트
#       ./test_linux.sh vulkan -- --ignored    # 알려진 버그(#[ignore = "BUG(..)"]) 재현
#
# 필요 패키지(Ubuntu 24.04 기준):
#   libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev libayatana-appindicator3-dev librsvg2-dev
#   protobuf-compiler clang libvulkan-dev mesa-vulkan-drivers
# ONNX Runtime 을 직접 받아 둔 경우 ORT_LIB_LOCATION=<onnxruntime-linux-x64-1.28.0 경로> 를 지정하세요.
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-vulkan}"
[ $# -gt 0 ] && shift
case "$BACKEND" in
  cuda) FEATURE_ARGS=() ;;
  vulkan|rocm) FEATURE_ARGS=(--no-default-features --features "$BACKEND") ;;
  *) echo "unknown backend: $BACKEND (vulkan|cuda|rocm)"; exit 1 ;;
esac

if [ -n "${ORT_LIB_LOCATION:-}" ]; then
  export ORT_STRATEGY=system
  export LD_LIBRARY_PATH="$ORT_LIB_LOCATION/lib:${LD_LIBRARY_PATH:-}"
fi

# GPU 가 없는 머신에서는 Mesa llvmpipe 로 Vulkan 을 검증합니다.
# (포크는 기본적으로 CPU 타입 Vulkan 디바이스를 건너뜀. 실제 GPU 가 있으면 그것이 0번으로 선택됨)
if [ "$BACKEND" = vulkan ]; then export CANDLE_VULKAN_ALLOW_CPU="${CANDLE_VULKAN_ALLOW_CPU:-1}"; fi

# 앱 데이터 디렉터리(~/.local/share/logis-center)를 오염시키지 않도록 격리
XDG_DATA_HOME="$(mktemp -d)"; export XDG_DATA_HOME
trap 'rm -rf "$XDG_DATA_HOME"' EXIT

# tauri::generate_context! 는 ../dist 가 있어야 컴파일됩니다 (프론트 빌드 전이면 빈 placeholder)
[ -f dist/index.html ] || { mkdir -p dist; echo '<html></html>' > dist/index.html; }
# tauri.conf.json 의 bundle.resources "dlls/*" 글롭이 최소 1개 파일과 매치되어야 합니다
mkdir -p src-tauri/dlls
[ -n "$(ls -A src-tauri/dlls)" ] || touch src-tauri/dlls/.gitkeep

(cd src-tauri && cargo test --test it "${FEATURE_ARGS[@]}" "$@")

[ -d node_modules ] || npm install
npm test
