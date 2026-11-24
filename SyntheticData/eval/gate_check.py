#!/usr/bin/env python3
"""Evaluate benchmark / eval JSON reports against SyntheticData/eval/gates.toml.
Exits 0 only if every applicable gate passes; exits 1 on any violation.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


def check_alphaedge_report(report: dict, cfg: dict) -> list[str]:
    errs: list[str] = []
    ae_cfg = cfg.get("alphaedge_superiority", {})
    vi_cfg = cfg.get("vocab_integrity", {})

    for entry in report.get("comparisons", [report] if "ours" in report and "alphaedge" in report else []):
        name = entry.get("model", "unknown")
        ours = entry["ours"]
        ae = entry["alphaedge"]

        if ae_cfg.get("require_lower_vram", True):
            if not (ours["vram_mb"] < ae["vram_mb"]):
                errs.append(f"[{name}] VRAM FAIL: ours={ours['vram_mb']:.1f}MB >= alphaedge={ae['vram_mb']:.1f}MB")
        if ae_cfg.get("require_faster_ms_per_tok", True):
            if not (ours["decode_ms_per_tok"] < ae["decode_ms_per_tok"]):
                errs.append(
                    f"[{name}] SPEED (ms/tok) FAIL: ours={ours['decode_ms_per_tok']:.2f}ms >= alphaedge={ae['decode_ms_per_tok']:.2f}ms"
                )
        if ae_cfg.get("require_faster_total_gen_ms", True):
            if not (ours["total_gen_ms"] < ae["total_gen_ms"]):
                errs.append(
                    f"[{name}] SPEED (total_gen_ms) FAIL: ours={ours['total_gen_ms']:.1f}ms >= alphaedge={ae['total_gen_ms']:.1f}ms"
                )
        if ae_cfg.get("require_higher_precision", True):
            if not (ours["precision_score"] > ae["precision_score"]):
                errs.append(
                    f"[{name}] PRECISION FAIL: ours={ours['precision_score']:.4f} <= alphaedge={ae['precision_score']:.4f}"
                )
        if ours.get("orphan_bpe_tokens", 0) > vi_cfg.get("max_orphan_bpe_tokens", 0):
            errs.append(f"[{name}] ORPHAN TOKENS FAIL: ours={ours['orphan_bpe_tokens']} > 0")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="Benchmark/eval JSON report")
    ap.add_argument("--gates", type=Path, default=Path(__file__).with_name("gates.toml"))
    args = ap.parse_args()

    cfg = tomllib.loads(args.gates.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))

    errs = check_alphaedge_report(report, cfg)
    if errs:
        print("GATE CHECK FAILED:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"GATE CHECK PASSED: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
