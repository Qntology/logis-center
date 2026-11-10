"""
SDS 실행 파일 읽기
=================

앱 데이터의 score_dynamics 폴더에서 원장과 실행 파일을 읽고, 평가 결과의 시간 창(t0~t1)으로
실행 파일을 페이지 · 문서 · 질의에 연결합니다. 하네스가 작업마다 report() 로 실행을 닫으므로
실행 파일 하나는 작업 하나의 관측입니다.
"""
import json
from pathlib import Path


def ledger_root(app_dir: Path) -> Path:
    """기본 순전파의 공간 폴더가 있으면 그쪽, 없으면 예전 자리입니다."""
    spaces = app_dir / "score_dynamics" / "spaces"
    if spaces.exists():
        tags = sorted(p for p in spaces.iterdir() if p.is_dir())
        if tags:
            return tags[-1]
    return app_dir / "score_dynamics"


def load_runs(app_dir: Path) -> list:
    runs = []
    root = ledger_root(app_dir) / "runs"
    if not root.exists():
        return runs
    for ledger_dir in sorted(root.iterdir()):
        for f in sorted(ledger_dir.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_file"] = f.name
            d["_ledger"] = ledger_dir.name
            runs.append(d)
    return runs


def run_label(raw: str) -> str:
    """앱 score_dynamics::sanitize_label 과 같은 규칙: ASCII 영숫자 · '-' · '_' 만 두고 나머지는 '_' 하나로, 48자, 앞뒤 '_' 제거."""
    out = ""
    for ch in raw.strip():
        if len(out) >= 48:
            break
        if (ch.isascii() and ch.isalnum()) or ch in "-_":
            out += ch
        elif not out.endswith("_"):
            out += "_"
    return out.strip("_") or "run"


def attach_runs(results: list, runs: list) -> dict:
    """결과 id → 그 작업의 실행 파일 목록.

    하네스는 작업마다 set_run_label(id) 를 부르므로 실행 라벨이 id 와 같으면 그 작업입니다. 라벨이 다른 실행
    (검색은 질의문으로 다시 라벨을 붙이고, 다른 원장은 기본 라벨로 엶)은 시작 시각이 작업의 시간 창(t0~t1) 안에 드는
    작업에, 없으면 ±2초 안에서 t0 가 가장 가까운 작업에 붙입니다. 예전에는 ±2초 창을 시간순으로 훑어 처음 맞는
    작업에 붙여서, 작업이 잇달아 돌면(시드는 건당 1초 안팎) 다음 작업의 실행이 앞 작업에 붙었습니다.
    """
    out = {r["id"]: [] for r in results}
    by_label = {}
    for r in results:
        by_label.setdefault(run_label(r["id"]), []).append(r)
    for run in runs:
        t = run.get("started_at", 0)
        pool = by_label.get(run.get("label", ""), [])
        if not pool:
            pool = [r for r in results if r["t0"] <= t <= r.get("t1", r["t0"])]
        if not pool:
            pool = [r for r in results if r["t0"] - 2000 <= t <= r.get("t1", r["t0"]) + 2000]
        if pool:
            out[min(pool, key=lambda r: abs(r["t0"] - t))["id"]].append(run)
    return out


def welford_sd(w: dict) -> float:
    n = w.get("n", 0)
    return (w.get("m2", 0.0) / (n - 1)) ** 0.5 if n > 1 else 0.0


def axes_of(run: dict) -> dict:
    """실행 하나의 baseline 축 → (n, 평균). 스코프는 합칩니다."""
    out = {}
    for scope in run.get("scopes", {}).values():
        for name, w in scope.get("baseline", {}).items():
            n, mean = out.get(name, (0, 0.0))
            total = n + w["n"]
            if total:
                out[name] = (total, (mean * n + w["mean"] * w["n"]) / total)
    return out


def decay_of(run: dict) -> dict:
    """실행 하나의 decay 축 → 마진 평균."""
    out = {}
    for scope in run.get("scopes", {}).values():
        for name, d in scope.get("decay", {}).items():
            m = d.get("margin", {})
            if m.get("n"):
                out[name] = m["mean"]
    return out


def fields_of(run: dict) -> dict:
    out = {}
    for scope in run.get("scopes", {}).values():
        for name, f in scope.get("field", {}).items():
            e = out.setdefault(name, {"seen": 0, "assigned": 0, "near_miss": 0, "reject_format": 0,
                                      "reject_prejudice": 0, "reject_enum": 0, "reject_self_id": 0})
            for k in e:
                e[k] += f.get(k, 0)
    return out


def confusions_of(run: dict) -> dict:
    out = {}
    for scope in run.get("scopes", {}).values():
        for name, c in scope.get("confusion", {}).items():
            e = out.setdefault(name, {"ties": 0, "a_wins": 0, "b_wins": 0})
            for k in e:
                e[k] += c.get(k, 0)
    return out
