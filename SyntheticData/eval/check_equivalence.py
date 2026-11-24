#!/usr/bin/env python3
"""E0/E1/E2/E3 numerical and token equivalence verifier (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_f32_vectors(ref_path: Path, cand_path: Path, dim: int, tier: str) -> tuple[bool, dict]:
    ref = np.fromfile(ref_path, dtype=np.float32).reshape(-1, dim)
    cand = np.fromfile(cand_path, dtype=np.float32).reshape(-1, dim)
    if ref.shape != cand.shape:
        return False, {"error": f"shape mismatch {ref.shape} vs {cand.shape}"}
    max_abs = float(np.max(np.abs(ref - cand))) if ref.size else 0.0
    cos_list = [cosine(ref[i], cand[i]) for i in range(ref.shape[0])]
    min_cos = min(cos_list) if cos_list else 1.0
    mean_cos = sum(cos_list) / len(cos_list) if cos_list else 1.0
    if tier == "E0":
        ok = max_abs == 0.0
    elif tier == "E1":
        ok = max_abs <= 1e-4 and min_cos >= 0.9999
    else:
        ok = min_cos >= 0.995
    return ok, {"tier": tier, "rows": ref.shape[0], "max_abs_err": max_abs, "min_cos": min_cos, "mean_cos": mean_cos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--tier", choices=["E0", "E1", "E2", "E3"], default="E0")
    args = ap.parse_args()
    ok, stats = check_f32_vectors(args.ref, args.cand, args.dim, args.tier)
    print(json.dumps(stats, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
