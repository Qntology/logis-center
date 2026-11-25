#!/usr/bin/env python3
"""Verify 4-layer ReflectionRecord JSONL ledger integrity (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TOP = ("ts_ms", "run_id", "task_id", "build_profile", "backend", "artifact")
REQUIRED_ARTIFACT = ("slot", "bytes", "vocab_v", "artifact_k")
RETIRED_OR_FORBIDDEN_SLOTS = {
    "granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-350m",
    "embeddinggemma-300m",
    "google/embeddinggemma-300m",
}


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
        slot = str(art.get("slot", "")).strip().lower()
        if slot in RETIRED_OR_FORBIDDEN_SLOTS:
            errs.append(f"retired or forbidden model slot '{art.get('slot')}'")
        if art.get("artifact_k", 0) > art.get("vocab_v", 0) and art.get("vocab_v", 0) > 0:
            errs.append(f"artifact_k ({art.get('artifact_k')}) > vocab_v ({art.get('vocab_v')})")
    if obj.get("load") is None and obj.get("exec") is None:
        errs.append("both load and exec are null")
    if obj.get("load") is not None:
        ld = obj["load"]
        if not isinstance(ld, dict):
            errs.append("load must be object")
        else:
            for num_k in ("load_ms", "vram_before_mb", "vram_after_mb"):
                val = ld.get(num_k)
                if not isinstance(val, (int, float)) or val < 0:
                    errs.append(f"load.{num_k} must be >= 0 (got {val!r})")
            tm = ld.get("table_mode")
            if not isinstance(tm, str) or not tm.strip():
                errs.append(f"load.table_mode must be non-empty string (got {tm!r})")
    if isinstance(obj.get("exec"), dict):
        ex = obj["exec"]
        if "vocab_k" in ex and len(ex["vocab_k"]) == 2:
            v, k = ex["vocab_k"]
            if v > 0 and k > v:
                errs.append(f"exec.vocab_k active_k ({k}) > vocab_v ({v})")
        if "langs" in ex:
            langs = ex["langs"]
            if not isinstance(langs, list) or len(langs) != 3 or not all(isinstance(s, str) for s in langs):
                errs.append(f"exec.langs must be a 3-element string list (got {langs!r})")
        if "steps" in ex:
            steps = ex["steps"]
            if not isinstance(steps, list) or len(steps) != 5:
                errs.append(f"exec.steps must be a 5-element list (got {steps!r})")
            elif steps[1] != 0:
                errs.append(f"exec.steps[1] penalty fallback must be 0 (got {steps[1]!r})")
    elif obj.get("exec") is not None:
        errs.append("exec must be object")
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
        "load": {"load_ms": 420.0, "vram_before_mb": 200.0, "vram_after_mb": 2100.0, "table_mode": "mapped"},
        "exec": {"step": "generate", "langs": ["ko", "ko", "hangul"], "vocab_k": [49152, 12288], "timing_ms": [12.0, 220.0, 1.2], "steps": [48, 0, 0, 1, 0]},
    }
    good_embed = {
        "ts_ms": 1795420800001,
        "run_id": "selftest",
        "task_id": "embed",
        "build_profile": "release-abort",
        "backend": "rocm",
        "artifact": {"slot": "granite-embedding-97m-multilingual-r2", "bytes": 195000000, "vocab_v": 65536, "artifact_k": 65536},
        "load": {"load_ms": 95.0, "vram_before_mb": 200.0, "vram_after_mb": 410.0, "table_mode": "mapped"},
        "exec": None,
    }
    bad = {"ts_ms": 1, "artifact": {"slot": "x", "bytes": 1, "vocab_v": 100, "artifact_k": 200}}
    bad_langs = dict(good, exec={"step": "generate", "langs": ["ko"], "steps": [10, 0, 0, 0, 0]})
    bad_steps = dict(good, exec={"step": "generate", "langs": ["ko", "ko", "hangul"], "steps": [10, 2, 0, 0, 0]})
    bad_load = dict(good, load={"load_ms": -1.0, "vram_before_mb": 1.0, "vram_after_mb": 2.0, "table_mode": ""})
    bad_retired = dict(good, artifact={"slot": "granite-4.0-h-350m", "bytes": 1, "vocab_v": 100, "artifact_k": 100})
    assert not validate_record(good), f"expected good to pass: {validate_record(good)}"
    assert not validate_record(good_embed), f"expected granite-97m-r2 to pass: {validate_record(good_embed)}"
    assert validate_record(bad), "expected bad to fail"
    assert validate_record(bad_langs), "expected bad_langs to fail"
    assert validate_record(bad_steps), "expected bad_steps to fail"
    assert validate_record(bad_load), "expected bad_load to fail"
    assert validate_record(bad_retired), "expected retired model to fail"
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
