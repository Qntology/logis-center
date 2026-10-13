#!/usr/bin/env bash
# Mode B: apply logis-center/patches/candle-rocm-vulkan/*.patch (series from fork master) to the fork
# checkout next to logis-center. Run from the directory that holds both checkouts.
set -euo pipefail
FORK="${FORK_DIR:-candle-rocm-vulkan}"
PATCHES="${PATCH_DIR:-logis-center/patches/candle-rocm-vulkan}"
git -C "$FORK" config user.name  "mac-verify CI"
git -C "$FORK" config user.email "mac-verify@users.noreply.github.com"
shopt -s nullglob
series=("$PATCHES"/*.patch)
if [ ${#series[@]} -eq 0 ]; then
  echo "no fork patches; building fork master $(git -C "$FORK" rev-parse --short HEAD)"
  exit 0
fi
abs=(); for p in "${series[@]}"; do abs+=("$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"); done
git -C "$FORK" am --3way "${abs[@]}"
echo "fork: master + ${#series[@]} patches -> $(git -C "$FORK" rev-parse --short HEAD)"
git -C "$FORK" log --oneline -n "${#series[@]}"
