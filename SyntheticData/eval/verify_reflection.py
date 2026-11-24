#!/usr/bin/env python3
"""Verify 4-layer ReflectionRecord JSONL ledger integrity (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TOP = ("ts_ms", "run_id", "task_id", "build_profile", "backend", "artifact")
REQUIRED_ARTIFACT = ("slot", "bytes", "vocab_v", "artifact_k")


def validate_record(obj: dict) -> list[str]:
    errs: list[str] = []
    for k in REQUIRED_TOP:
        if k not in obj:
            errs.append(f"missing top key '{k}'")
    art = obj.get("artifact")
    if not isinstance(art, dict):
        errs.append("artifact must be object")
    else:
        for k in REQUIRED_ARTIFACT:
            if k not in art:
                errs.append(f"missing artifact.{k}")
        if art.get("artifact_k", 0) > art.get("vocab_v", 0) and art.get("vocab_v", 0) > 0:
            errs.append(f"artifact_k ({art.get('artifact_k')}) > vocab_v ({art.get('vocab_v')})")
    if obj.get("load") is None and obj.get("exec") is None:
        errs.append("both load and exec are null")
    if isinstance(obj.get("exec"), dict):
        ex = obj["exec"]
        if "vocab_k" in ex and len(ex["vocab_k"]) == 2:
            v, k = ex["vocab_k"]
            if v > 0 and k > v:
                errs.append(f"exec.vocab_k active_k ({k}) > vocab_v ({v})")
    return errs


def selftest() -> int:
    good = {
        "ts_ms": 1795420800000,
        "run_id": "selftest",
        "task_id": "qwen3_5_4b",
        "git_rev": "abc1234",
        "build_profile": "release-abort",
        "features": ["rocm", "flash-attn"],
        "backend": "rocm",
        "device_name": "RX 6600",
        "artifact": {"slot": "qwen3_5_4b", "bytes": 2300000000, "sha256_16": "0123456789abcdef", "recipe": "Q4_K_M", "vocab_v": 49152, "artifact_k": 12288},
        "load": None,
        "exec": {"step": "generate", "vocab_k": [49152, 12288], "timing_ms": [12.0, 220.0, 1.2], "steps": [48, 0, 0, 1, 0]},
    }
    bad = {"ts_ms": 1, "artifact": {"slot": "x", "bytes": 1, "vocab_v": 100, "artifact_k": 200}}
    assert not validate_record(good), f"expected good to pass: {validate_record(good)}"
    assert validate_record(bad), "expected bad to fail"
    print("[REFLECTION-VERIFY] selftest PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.jsonl or not args.jsonl.exists():
        print("missing jsonl path", file=sys.stderr)
        return 1
    total, bad = 0, 0
    for idx, line in enumerate(args.jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"line {idx}: invalid json ({e})")
            bad += 1
            continue
        errs = validate_record(obj)
        if errs:
            print(f"line {idx}: {', '.join(errs)}")
            bad += 1
    print(f"[REFLECTION-VERIFY] checked={total} invalid={bad}")
    return 0 if (total > 0 and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
