#!/usr/bin/env python3
"""Compare nearest-neighbor rankings (Recall@K) between two f32 embedding dumps (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-recall", type=float, default=0.98)
    args = ap.parse_args()
    a = np.fromfile(args.ref, dtype=np.float32).reshape(-1, args.dim)
    b = np.fromfile(args.cand, dtype=np.float32).reshape(-1, args.dim)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    sa = a @ a.T
    sb = b @ b.T
    k = min(args.k, a.shape[0])
    overlaps = []
    for i in range(a.shape[0]):
        ta = set(np.argsort(-sa[i])[:k].tolist())
        tb = set(np.argsort(-sb[i])[:k].tolist())
        overlaps.append(len(ta & tb) / max(k, 1))
    mean_r = float(np.mean(overlaps)) if overlaps else 1.0
    print(json.dumps({"n": a.shape[0], "k": k, "mean_recall_at_k": mean_r}, indent=2))
    return 0 if mean_r >= args.min_recall else 1


if __name__ == "__main__":
    sys.exit(main())
