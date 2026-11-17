#!/usr/bin/env bash
# Switch the four candle-rocm-vulkan lines from the Qntology git form to the commented local path form:
#   commerce/src-tauri/Cargo.toml   candle-rocm                                      -> ../../../candle-rocm-vulkan/...
#   Cargo.toml (workspace root)     [patch.crates-io] candle-core · candle-nn · candle-flash-attn -> ../candle-rocm-vulkan/...
# Both resolve to the fork checkout next to logis-center. CI / local only — never commit.
set -euo pipefail
app="${1:-commerce/src-tauri/Cargo.toml}"
root="${2:-$(dirname "$app")/../../Cargo.toml}"
swap() { # <manifest> <crate>
  sed -i.bak -E \
    -e "s|^($2 = \{ git = \"https://github.com/Qntology/candle-rocm-vulkan\".*)$|# \1|" \
    -e "s|^# ($2 = \{ path = \"(\.\./)+candle-rocm-vulkan/.*)$|\1|" "$1"
  rm -f "$1.bak"
}
swap "$app" candle-rocm
for c in candle-core candle-nn candle-flash-attn; do
  swap "$root" "$c"
done
n=$(cat "$app" "$root" | grep -cE '^candle-(rocm|core|nn|flash-attn) = \{ path = "(\.\./)+candle-rocm-vulkan/')
[ "$n" -eq 4 ] || { echo "expected 4 path lines, found $n"; grep -n candle-rocm-vulkan "$app" "$root"; exit 1; }
echo "Cargo.toml: 4 fork lines -> path candle-rocm-vulkan (app manifest: candle-rocm, workspace root: 3 patches)"
