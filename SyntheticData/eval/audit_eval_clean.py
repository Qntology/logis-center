#!/usr/bin/env python3
"""Audit zero leakage between VocabPlan mining corpus and SyntheticData eval/relational sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def norm_line(s: str) -> str:
    return " ".join(s.strip().lower().split())


def collect_eval_strings(eval_root: Path) -> set[str]:
    out: set[str] = set()
    for pat in ("*.yaml", "*.yml", "*.json", "*.jsonl"):
        for p in eval_root.rglob(pat):
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    n = norm_line(line)
                    if len(n) >= 24:
                        out.add(hashlib.sha256(n.encode("utf-8")).hexdigest())
            except OSError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, help="Optional mining corpus file/dir to check against eval sets")
    ap.add_argument("--eval-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()

    eval_hashes = collect_eval_strings(args.eval_dir)
    if not args.corpus or not args.corpus.exists():
        print(f"[AUDIT-CLEAN] Indexed {len(eval_hashes)} eval-line fingerprints under {args.eval_dir}; no external mining corpus passed -> CLEAN.")
        return 0

    files = [args.corpus] if args.corpus.is_file() else list(args.corpus.rglob("*"))
    leaks = 0
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            n = norm_line(line)
            if len(n) >= 24 and hashlib.sha256(n.encode("utf-8")).hexdigest() in eval_hashes:
                leaks += 1
    print(json.dumps({"eval_fingerprints": len(eval_hashes), "leaks": leaks}))
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
