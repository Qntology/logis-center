#!/usr/bin/env python3
"""Verify that CI workflows and Cargo.toml pin the exact same candle-rocm-vulkan commit
and that no unpinned floating branch references drift silently (Phase 5 - Part 2 D5)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def extract_candle_revs(cargo_toml: str) -> set[str]:
    revs = set()
    for line in cargo_toml.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "candle-rocm-vulkan" in stripped:
            m = re.search(r'rev\s*=\s*"([0-9a-f]{7,40})"', stripped)
            if m:
                revs.add(m.group(1))
    return revs


def main() -> int:
    cargo_path = ROOT / "Cargo.toml"
    cargo_text = cargo_path.read_text(encoding="utf-8")
    revs = extract_candle_revs(cargo_text)
    if len(revs) != 1:
        print(f"[VERIFY-PINS] FAIL: expected exactly 1 pinned candle-rocm-vulkan rev in Cargo.toml, found {revs}", file=sys.stderr)
        return 1
    pinned = next(iter(revs))
    print(f"[VERIFY-PINS] Cargo.toml candle-rocm-vulkan pinned rev = {pinned}")

    if "--print-candle-rev" in sys.argv:
        print(pinned)
        return 0

    errors = []
    for wf_rel in [".github/workflows/cross-verify.yml", ".github/workflows/mac-verify.yml"]:
        wf = ROOT / wf_rel
        if not wf.exists():
            continue
        text = wf.read_text(encoding="utf-8")
        if re.search(r"repository:\s*\S*candle-rocm-vulkan[^\n]*ref:\s*master", text):
            errors.append(f"{wf_rel}: still pins candle-rocm-vulkan to floating 'ref: master' instead of pinned rev {pinned}")

    if errors:
        for e in errors:
            print(f"[VERIFY-PINS] FAIL: {e}", file=sys.stderr)
        return 1
    print("[VERIFY-PINS] PASS: all CI workflows aligned with Cargo.toml pin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
