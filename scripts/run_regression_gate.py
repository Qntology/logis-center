#!/usr/bin/env python3
"""Single-command regression gate orchestrator (Phase 5 – Part 1)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cargo", action="store_true")
    args = ap.parse_args()

    checks = [
        [sys.executable, ".github/scripts/verify-pins.py"],
        [sys.executable, "SyntheticData/eval/verify_reflection.py", "--selftest"],
        [sys.executable, "SyntheticData/eval/audit_eval_clean.py"],
    ]
    if not args.skip_cargo:
        checks.append([
            "cargo", "test", "--offline",
            "-p", "logis-schema", "-p", "logis-bias", "-p", "logis-pug",
            "-p", "logis-search", "-p", "logis-vision-doc", "-p", "logis-llm",
            "--no-default-features", "--lib",
        ])
    for cmd in checks:
        rc = run(cmd)
        if rc != 0:
            print(f"[REGRESSION-GATE] FAILED (exit {rc}): {' '.join(cmd)}")
            return rc
    print("[REGRESSION-GATE] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
