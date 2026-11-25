#!/usr/bin/env python3
"""Verify that CI workflows and Cargo.toml pin the exact same candle-rocm-vulkan,
Qntology/premium-discount (Tier 0), and Qntology/louis-vision (Tier 1) commits,
and that duplicated local kernel/quant crates remain absent."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CANDLE_REV = "f73415d4d149d7949e8a6a94911af670e4e10f7f"
EXPECTED_PD_REV = "dae2667e498b65e6eb92ca0bd3ebf7e10c75c675"
EXPECTED_LV_REV = "341e54cdc822aa7ee6201fba8f48472df11e1bd6"


def extract_revs(cargo_toml: str, needle: str) -> set[str]:
    revs = set()
    for line in cargo_toml.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or re.match(r"^repository\s*=", stripped):
            continue
        if needle in stripped:
            m = re.search(r'rev\s*=\s*"([0-9a-f]{7,40})"', stripped)
            if m:
                revs.add(m.group(1))
    return revs


def main() -> int:
    cargo_path = ROOT / "Cargo.toml"
    cargo_text = cargo_path.read_text(encoding="utf-8")

    candle_revs = extract_revs(cargo_text, "candle-rocm-vulkan")
    pd_revs = extract_revs(cargo_text, "Qntology/premium-discount")
    lv_revs = extract_revs(cargo_text, "Qntology/louis-vision")

    if "--print-candle-rev" in sys.argv:
        if len(candle_revs) != 1:
            return 1
        print(next(iter(candle_revs)))
        return 0

    errors = []
    if candle_revs != {EXPECTED_CANDLE_REV}:
        errors.append(f"Cargo.toml candle-rocm-vulkan revs {sorted(candle_revs)} != [{EXPECTED_CANDLE_REV}]")
    if pd_revs != {EXPECTED_PD_REV}:
        errors.append(f"Cargo.toml premium-discount revs {sorted(pd_revs)} != [{EXPECTED_PD_REV}]")
    if lv_revs != {EXPECTED_LV_REV}:
        errors.append(f"Cargo.toml louis-vision revs {sorted(lv_revs)} != [{EXPECTED_LV_REV}]")

    for forbidden in ("crates/embed-cpu-kernels", "crates/ggml-kquant"):
        if (ROOT / forbidden).exists():
            errors.append(f"Duplicated local crate directory still exists: {forbidden}")

    for wf_rel in [".github/workflows/cross-verify.yml", ".github/workflows/mac-verify.yml"]:
        wf = ROOT / wf_rel
        if not wf.exists():
            continue
        text = wf.read_text(encoding="utf-8")
        if re.search(r"repository:\s*\S*candle-rocm-vulkan[^\n]*ref:\s*master", text):
            errors.append(f"{wf_rel}: still pins candle-rocm-vulkan to floating 'ref: master'")

    if errors:
        for e in errors:
            print(f"[VERIFY-PINS] FAIL: {e}", file=sys.stderr)
        return 1
    print(
        f"[VERIFY-PINS] PASS candle={EXPECTED_CANDLE_REV[:8]} pd={EXPECTED_PD_REV[:8]} lv={EXPECTED_LV_REV[:8]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
