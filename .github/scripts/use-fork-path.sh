#!/usr/bin/env bash
# Switch the four candle-rocm-vulkan lines of commerce/src-tauri/Cargo.toml (candle-rocm + 3 [patch.crates-io])
# from the Qntology git form to the commented ../../../candle-rocm-vulkan path form. CI / local only — never commit.
set -euo pipefail
f="${1:-commerce/src-tauri/Cargo.toml}"
for c in candle-rocm candle-core candle-nn candle-flash-attn; do
  sed -i.bak -E \
    -e "s|^(${c} = \{ git = \"https://github.com/Qntology/candle-rocm-vulkan\".*)$|# \1|" \
    -e "s|^# (${c} = \{ path = \"\.\./\.\./\.\./candle-rocm-vulkan/.*)$|\1|" "$f"
done
rm -f "$f.bak"
n=$(grep -cE '^candle-(rocm|core|nn|flash-attn) = \{ path = "\.\./\.\./\.\./candle-rocm-vulkan/' "$f")
[ "$n" -eq 4 ] || { echo "expected 4 path lines, found $n"; grep -n candle-rocm-vulkan "$f"; exit 1; }
echo "Cargo.toml: 4 fork lines -> path ../../../candle-rocm-vulkan"
