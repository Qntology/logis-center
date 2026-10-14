#!/usr/bin/env python3
"""
cuda-arch-sweep.py — compile the CUDA kernels (candle-kernels) once per GPU-architecture *branch*, for the
toolkit that is installed.

A CI runner has no GPU, so nothing can pick the architecture the way the real build does (nvidia-smi ->
CUDA_COMPUTE_CAP). The kernels branch on `__CUDA_ARCH__` and the toolkit decides which architectures exist
at all (CUDA 13 dropped everything below Turing), so the set is derived, not written down:

  1. thresholds  every `__CUDA_ARCH__ <op> N` and every MIN_CC_* / GGML_CUDA_CC_* constant in the kernel sources
  2. supported   `nvcc --list-gpu-code` of the installed toolkit
  3. picks       the lowest supported architecture of every region between two thresholds, plus the newest one
  4. compile     `cargo check -p candle-kernels` in a private target dir. Its build script does not watch
                 CUDA_COMPUTE_CAP, so two architectures must never share a target dir (or the cached main one).

One line per architecture. A failure that matches a registered signature is KNOWN (warning + reason) and does not
fail the job; any other failure does. Run from commerce/src-tauri.
Env: SWEEP_MINUTES (budget, default 60), SWEEP_SKIP (comma list of sm numbers already verified elsewhere),
     SWEEP_CMD (compile command, default `cargo check -p candle-kernels`; tests replace it).
"""
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

# A failure is "known" when its log matches one of these (warning with the reason instead of a red job).
# Empty on purpose: the CUDA 13 + sm_<80 clash in candle-kernels' compatibility.cuh ("__hmax_nan ... has already been
# defined") was fixed in the fork (candle-kernels/FORK-NOTES.md); keeping its signature here would hide a regression.
# Entry shape:  (re.compile(r"<regex over the compiler output>"), "<why it fails and what would fix it>")
KNOWN = []


def annotate(level, title, lines):
    msg = "\n".join(lines).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::{level} title={title}::{msg[:3800]}", flush=True)


def kernel_sources():
    """Where the candle-kernels sources are, wherever cargo got them from (crates.io or a path inside the fork)."""
    import json
    for extra in (["--offline"], []):
        r = subprocess.run(["cargo", "metadata", "--format-version", "1", *extra], capture_output=True, text=True)
        if r.returncode == 0:
            break
    else:
        r = None
    if r is not None and r.returncode == 0:
        for pkg in json.loads(r.stdout)["packages"]:
            if pkg["name"] == "candle-kernels":
                return pkg["version"], pathlib.Path(pkg["manifest_path"]).parent
    # last resort: the registry copy named in Cargo.lock
    lock = pathlib.Path("Cargo.lock").read_text()
    m = re.search(r'name = "candle-kernels"\s*\nversion = "([^"]+)"', lock)
    if not m:
        sys.exit("candle-kernels is not in the dependency graph")
    ver = m.group(1)
    home = pathlib.Path(os.environ.get("CARGO_HOME") or (pathlib.Path.home() / ".cargo"))
    found = sorted(glob.glob(str(home / "registry" / "src" / "*" / f"candle-kernels-{ver}")))
    if not found:
        sys.exit(f"candle-kernels-{ver} sources not found (cargo metadata failed and no registry copy)")
    return ver, pathlib.Path(found[-1])


def thresholds(src):
    """Architecture numbers (sm_75 -> 750) at which the kernel code takes another branch."""
    out = set()
    cmp_re = re.compile(r"__CUDA_ARCH__\s*(<=|>=|==|<|>)\s*(\d+)")
    def_re = re.compile(r"#\s*define\s+(?:MIN_CC_[A-Z0-9_]+|GGML_CUDA_CC_[A-Z0-9_]+)\s+(\d+)\b")
    for f in src.rglob("*"):
        if f.suffix not in (".cu", ".cuh"):
            continue
        text = f.read_text(errors="ignore")
        for op, n in cmp_re.findall(text):
            n = int(n)
            out.update({">=": [n], "<": [n], "<=": [n + 1], ">": [n + 1], "==": [n, n + 1]}[op])
        out.update(int(n) for n in def_re.findall(text))
    return sorted(x for x in out if 300 <= x <= 1300)


def supported():
    """sm numbers the installed toolkit can generate code for, e.g. [500, ..., 900]."""
    r = subprocess.run(["nvcc", "--list-gpu-code"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("nvcc --list-gpu-code failed: " + (r.stderr or r.stdout)[:300])
    return sorted({int(n) * 10 for n in re.findall(r"sm_(\d+)", r.stdout)})


def picks(sup, th):
    """Lowest supported architecture of every region [th[i-1], th[i]) + the newest one."""
    edges = [0] + th + [10**9]
    chosen, regions = [], {}
    for lo, hi in zip(edges, edges[1:]):
        members = [a for a in sup if lo <= a < hi]
        if members:
            chosen.append(members[0])
            regions[members[0]] = (lo, hi)
    if sup and sup[-1] not in chosen:
        chosen.append(sup[-1])
        regions[sup[-1]] = ("newest", "")
    return sorted(set(chosen)), regions


def classify(log):
    for rx, why in KNOWN:
        if rx.search(log):
            return "KNOWN", why
    return "FAIL", ""


def first_errors(log, n=6):
    keep = [l.strip()[:230] for l in log.splitlines() if re.search(r"error|nvcc|fatal|NotFound", l, re.I)]
    return keep[:n]


def main():
    ver, src = kernel_sources()
    th = thresholds(src)
    sup = supported()
    chosen, regions = picks(sup, th)
    skip = {int(x) * 10 for x in os.environ.get("SWEEP_SKIP", "").split(",") if x.strip().isdigit()}
    budget = float(os.environ.get("SWEEP_MINUTES", "60")) * 60
    cmd = os.environ.get("SWEEP_CMD", "cargo check -p candle-kernels").split()
    clean = os.environ.get("SWEEP_CLEAN", "cargo clean -p candle-kernels").split()

    print(f"candle-kernels {ver}: branch thresholds {th}")
    print(f"toolkit supports sm_{sup[0] // 10}..sm_{sup[-1] // 10} ({len(sup)} architectures)")
    print(f"architectures to compile: {[f'sm_{a // 10}' for a in chosen]} (skipping verified: {sorted(skip)})")

    target = pathlib.Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "cuda-sweep-target"
    rows, failed, started = [], False, time.time()
    for a in chosen:
        tag = f"sm_{a // 10}"
        lo, hi = regions[a]
        where = f"newest supported" if lo == "newest" else f"branch region {lo}..{hi - 1 if hi < 10**9 else 'up'}"
        if a in skip:
            rows.append((tag, where, "COVERED", "compiled by the main type-check step", 0.0))
            continue
        if time.time() - started > budget:
            rows.append((tag, where, "NOT REACHED", f"sweep budget {budget / 60:.0f} min used up", 0.0))
            continue
        env = dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_COMPUTE_CAP=str(a // 10))
        subprocess.run(clean, env=env, capture_output=True, text=True)
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2700)
            log, rc = (r.stdout or "") + "\n" + (r.stderr or ""), r.returncode
        except subprocess.TimeoutExpired:
            log, rc = "timeout after 45 min", 124
        mins = (time.time() - t0) / 60
        if rc == 0:
            rows.append((tag, where, "OK", "", mins))
            continue
        status, why = classify(log)
        errs = first_errors(log)
        rows.append((tag, where, status, why or (errs[0] if errs else "see log"), mins))
        if status == "KNOWN":
            annotate("warning", f"cuda sweep {tag}: KNOWN failure", [why] + errs)
        else:
            failed = True
            annotate("error", f"cuda sweep {tag}: FAILED", errs or ["(no error line captured)"] + log.splitlines()[-8:])
        # drop the failed attempt's output; the next architecture starts from a clean target
    shutil.rmtree(target, ignore_errors=True)

    width = max(len(r[0]) for r in rows)
    lines = [f"{r[0]:<{width}}  {r[2]:<11} {r[4]:5.1f} min  {r[1]}" + (f"  — {r[3]}" if r[3] else "") for r in rows]
    print("\n".join(lines))
    annotate("notice", f"cuda arch sweep (candle-kernels {ver}, sm_{sup[0] // 10}..sm_{sup[-1] // 10})", lines)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("#### CUDA architecture sweep\n\n| arch | result | min | branch | note |\n|---|---|---|---|---|\n")
            for t, w, s, n, m in rows:
                f.write(f"| {t} | {s} | {m:.1f} | {w} | {n} |\n")
    unreached = [r[0] for r in rows if r[2] == "NOT REACHED"]
    if unreached:
        # an architecture that was not compiled is not verified: the sweep is incomplete, which is not a pass
        annotate("error", "cuda sweep: INCOMPLETE", [", ".join(unreached) + " were not compiled (budget used up)"])
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
