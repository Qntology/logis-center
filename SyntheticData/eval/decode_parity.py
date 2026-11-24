#!/usr/bin/env python3
"""Compare decoded outputs and identifier integrity across model arms (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--max-cer", type=float, default=0.01)
    args = ap.parse_args()
    ref_rows = [json.loads(l) for l in args.ref.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand_rows = [json.loads(l) for l in args.cand.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand_map = {r["id"]: r.get("text", "") for r in cand_rows}
    tot_edits, tot_chars, exact = 0, 0, 0
    for r in ref_rows:
        rt = r.get("text", "")
        ct = cand_map.get(r["id"], "")
        ed = edit_distance(rt, ct)
        tot_edits += ed
        tot_chars += max(len(rt), 1)
        exact += int(ed == 0)
    cer = tot_edits / max(tot_chars, 1)
    print(json.dumps({"cases": len(ref_rows), "exact": exact, "cer": cer}, indent=2))
    return 0 if cer <= args.max_cer else 1


if __name__ == "__main__":
    sys.exit(main())
