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
# A patch whose commit is already in fork master (the working branch was merged) has the same
# patch-id as a master commit: skip it instead of failing `git am`.
in_master="$(git -C "$FORK" log -p --no-merges HEAD | git patch-id --stable | cut -d' ' -f1 | sort -u)"
abs=(); merged=0
for p in "${series[@]}"; do
  a="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  id="$(git patch-id --stable < "$a" | cut -d' ' -f1)"
  if [ -n "$id" ] && grep -qx "$id" <<<"$in_master"; then
    echo "skip $(basename "$p") (already in fork master)"; merged=$((merged + 1))
  else
    abs+=("$a")
  fi
done
if [ ${#abs[@]} -gt 0 ]; then git -C "$FORK" am --3way "${abs[@]}"; fi
echo "fork: master + ${#abs[@]} patches ($merged already merged) -> $(git -C "$FORK" rev-parse --short HEAD)"
if [ ${#abs[@]} -gt 0 ]; then git -C "$FORK" log --oneline -n "${#abs[@]}"; fi
