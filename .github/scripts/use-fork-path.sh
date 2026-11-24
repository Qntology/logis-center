#!/usr/bin/env bash
# Switch the four candle-rocm-vulkan lines of the workspace root Cargo.toml from the git form
# (Qntology or sungu-che fork) to the local path form (../candle-rocm-vulkan/..., the fork checkout
# next to logis-center):
#   [workspace.dependencies] candle-rocm (used by crates/logis-llm)
#   [patch.crates-io]        candle-core · candle-nn · candle-flash-attn
# Argument: the workspace root Cargo.toml (an app manifest apps/desktop/Cargo.toml is mapped to its root).
# CI / local only — never commit.
set -euo pipefail
root="${1:-Cargo.toml}"
grep -q '^\[workspace\]' "$root" || root="$(dirname "$root")/../../Cargo.toml"
ws="$(dirname "$root")"
swap_clean() { # <manifest> <crate> <subdir>
  local tmp="$1.tmp"
  local has_commented=0
  grep -qE "^# $2 = \{ path = \"(\.\./)+candle-rocm-vulkan/" "$1" && has_commented=1
  awk -v crate="$2" -v subdir="$3" -v has_c="$has_commented" '
    $0 ~ ("^" crate " = \\{ git = \"https://github\\.com/(Qntology|sungu-che)/candle-rocm-vulkan\"") {
      print "# " $0
      if (!has_c) print crate " = { path = \"../candle-rocm-vulkan/" subdir "\" }"
      next
    }
    $0 ~ ("^# " crate " = \\{ path = \"(\\.\\./)+candle-rocm-vulkan/") {
      print substr($0, 3)
      next
    }
    { print }
  ' "$1" > "$tmp" && mv "$tmp" "$1"
}
swap_clean "$root" candle-rocm candle-backend
swap_clean "$root" candle-core candle-core
swap_clean "$root" candle-nn candle-nn
swap_clean "$root" candle-flash-attn candle-flash-attn
n=$(grep -cE '^candle-(rocm|core|nn|flash-attn) = \{ path = "(\.\./)+candle-rocm-vulkan/' "$root")
[ "$n" -eq 4 ] || { echo "expected 4 path lines, found $n"; grep -n candle-rocm-vulkan "$root"; exit 1; }
if grep -rlE '^candle-rocm = \{ git = ' --include=Cargo.toml "$ws/apps" "$ws/crates" 2>/dev/null; then
  echo "a member manifest above still pins candle-rocm by git (the pin belongs in the workspace root)"; exit 1
fi
echo "Cargo.toml: 4 fork lines -> path candle-rocm-vulkan (workspace root: candle-rocm + 3 patches)"
