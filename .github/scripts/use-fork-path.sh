#!/usr/bin/env bash
# Switch the four candle-rocm-vulkan lines of the workspace root Cargo.toml from the Qntology git form to the
# commented local path form (../candle-rocm-vulkan/..., the fork checkout next to logis-center):
#   [workspace.dependencies] candle-rocm (used by crates/logis-llm)
#   [patch.crates-io]        candle-core · candle-nn · candle-flash-attn
# Argument: the workspace root Cargo.toml (an app manifest commerce/src-tauri/Cargo.toml is mapped to its root).
# CI / local only — never commit.
set -euo pipefail
root="${1:-Cargo.toml}"
grep -q '^\[workspace\]' "$root" || root="$(dirname "$root")/../../Cargo.toml"
ws="$(dirname "$root")"
swap() { # <manifest> <crate>
  sed -i.bak -E \
    -e "s|^($2 = \{ git = \"https://github.com/Qntology/candle-rocm-vulkan\".*)$|# \1|" \
    -e "s|^# ($2 = \{ path = \"(\.\./)+candle-rocm-vulkan/.*)$|\1|" "$1"
  rm -f "$1.bak"
}
for c in candle-rocm candle-core candle-nn candle-flash-attn; do
  swap "$root" "$c"
done
n=$(grep -cE '^candle-(rocm|core|nn|flash-attn) = \{ path = "(\.\./)+candle-rocm-vulkan/' "$root")
[ "$n" -eq 4 ] || { echo "expected 4 path lines, found $n"; grep -n candle-rocm-vulkan "$root"; exit 1; }
if grep -rlE '^candle-rocm = \{ git = ' --include=Cargo.toml "$ws/commerce/src-tauri" "$ws/crates" 2>/dev/null; then
  echo "a member manifest above still pins candle-rocm by git (the pin belongs in the workspace root)"; exit 1
fi
echo "Cargo.toml: 4 fork lines -> path candle-rocm-vulkan (workspace root: candle-rocm + 3 patches)"
