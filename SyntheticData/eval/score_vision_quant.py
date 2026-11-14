"""
SigLIP2 정밀도 계획 비교 (PYRAMID-QUANT)
=====================================

  python score_vision_quant.py <PNG 폴더> <기준 이름>=<폴더> <비교 이름>=<폴더> ... [--boot 2000]

폴더마다 vision_probe 결과(result.jsonl)와 히트맵 덤프(dump/<쪽>.heat.json, LOGIS_SYNTH_HEAT=1)가 있습니다.
첫 번째가 기준(원본 F32)이고, 나머지를 기준과 쌍으로 비교합니다.

- 서식 판정: 첫 쪽 코드 정답률 (쌍둥이 코드는 같은 것으로 봄), 기준과 같은 판정 비율
- 크롭 커버리지: 정답 값이 크롭 안에 드는 비율 (어느 크롭 / 자기 카테고리 크롭) — score_vision.py 와 같은 정의
- 문서번호 회수: 식별 필드(TRADE_IDENTITY_FIELD = doc_number) 값을 어느 크롭이든 덮는가. 크롭 계획이 따로 보호하는
  값이라(identity 보호) 지시문의 '표제란 · 문서번호 회수 100 %' 에 해당합니다. 기준이 덮은 것을 놓친 수(회귀)를 셉니다.
- 히트맵 AP: 정답 값마다 그 값 카테고리의 히트맵으로 패치를 줄 세웠을 때 값 상자와 겹치는 패치의 평균 정밀도.
  ΔAP 는 기준과의 차이, 신뢰구간은 쪽 단위 부트스트랩입니다.
- 시간: 쪽당 인코딩 · 히트맵(첫 쪽은 구 인코딩 포함) 평균
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from score_vision import field_categories, find_value_box, inside, same_code
from truth import trading_docs

IDENTITY_FIELD = "doc_number"


def ap_of(scores: np.ndarray, positive: np.ndarray) -> float:
    """점수 내림차순 순위의 평균 정밀도. 같은 점수는 안정 정렬(패치 순서)."""
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = positive[order]
    cum = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    return float((cum[hits] / ranks[hits]).sum() / n_pos)


def cell_overlap(box, rows: int, cols: int, width: int, height: int) -> np.ndarray:
    """값 상자와 겹치는 패치 (rows*cols 불리언)."""
    ch, cw = height / rows, width / cols
    r0, r1 = int(box[1] // ch), int(min(rows - 1, box[3] // ch))
    c0, c1 = int(box[0] // cw), int(min(cols - 1, box[2] // cw))
    m = np.zeros(rows * cols, dtype=bool)
    for r in range(max(0, r0), r1 + 1):
        m[r * cols + max(0, c0): r * cols + c1 + 1] = True
    return m


def load_arm(folder: Path) -> dict:
    rows = [json.loads(l) for l in (folder / "result.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["page"]: r for r in rows}


def heat_of(folder: Path, page: str):
    p = folder / "dump" / f"{page}.heat.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    maps = {m["category"]: np.array([(-3.0e38 if s is None else s) for s in m["scores"]], dtype=np.float64) for m in h["maps"]}
    return h["rows"], h["cols"], maps


def page_items(pages: Path, page: str, doc: dict, cats: dict) -> list:
    meta = json.loads((pages / f"{page}.words.json").read_text(encoding="utf-8"))
    out = []
    for field, values in doc["truth"].items():
        for v in values:
            box = find_value_box(v, meta["words"])
            if box is None:
                continue
            out.append({
                "field": field, "cat": cats.get(field), "box": box,
                "br": field == IDENTITY_FIELD,
                "w": meta["width"], "h": meta["height"],
            })
    return out


def measure(folder: Path, arm: dict, pages: Path, docs: dict, cats: dict) -> dict:
    per_page = {}
    for page, r in arm.items():
        stem, num = page.rsplit("_p", 1)
        if stem not in docs or not (pages / f"{page}.words.json").exists():
            continue
        items = page_items(pages, page, docs[stem], cats)
        heat = heat_of(folder, page)
        res = {"code_ok": same_code(r["code"], docs[stem]["code"]) if num == "1" else None, "code": r["code"] if num == "1" else None,
               "values": [], "encode": r.get("encode_secs"), "heat": r.get("heat_secs")}
        for it in items:
            any_hit = any(inside(it["box"], p["bbox"]) for p in r["plans"])
            cat_hit = any(p["category"] == it["cat"] and inside(it["box"], p["bbox"]) for p in r["plans"])
            ap = float("nan")
            if heat and it["cat"] in heat[2]:
                rows, cols, maps = heat
                ap = ap_of(maps[it["cat"]], cell_overlap(it["box"], rows, cols, it["w"], it["h"]))
            res["values"].append({"key": (it["field"], tuple(round(x, 1) for x in it["box"])), "any": any_hit, "cat": cat_hit, "br": it["br"], "ap": ap})
        per_page[page] = res
    return per_page


def summarize(name: str, m: dict) -> dict:
    vals = [v for p in m.values() for v in p["values"]]
    first = [p for p in m.values() if p["code_ok"] is not None]
    aps = [v["ap"] for v in vals if not np.isnan(v["ap"])]
    br = [v for v in vals if v["br"]]
    enc = [p["encode"] for p in m.values() if p["encode"] is not None]
    heat = sorted(p["heat"] for p in m.values() if p["heat"] is not None)
    return {
        "arm": name, "pages": len(m), "docs": len(first),
        "code_acc": sum(p["code_ok"] for p in first) / max(1, len(first)),
        "any": sum(v["any"] for v in vals) / max(1, len(vals)), "cat": sum(v["cat"] for v in vals) / max(1, len(vals)),
        "br": sum(v["any"] for v in br) / max(1, len(br)), "br_n": len(br),
        "ap": float(np.mean(aps)) if aps else float("nan"), "ap_n": len(aps),
        "encode_ms": 1000 * float(np.mean(enc)) if enc else float("nan"),
        "heat_ms_median": 1000 * float(np.median(heat)) if heat else float("nan"),
        "heat_ms_first": 1000 * heat[-1] if heat else float("nan"),
    }


def paired(ref: dict, arm: dict, boot: int, seed: int = 7) -> dict:
    pages = sorted(set(ref) & set(arm))
    d_page = []
    code_same = sum(1 for p in pages if ref[p]["code"] is not None and ref[p]["code"] == arm[p]["code"])
    code_n = sum(1 for p in pages if ref[p]["code"] is not None)
    lost_any = lost_br = gained_any = 0
    for p in pages:
        rv = {v["key"]: v for v in ref[p]["values"]}
        av = {v["key"]: v for v in arm[p]["values"]}
        diffs = []
        for k, a in rv.items():
            b = av.get(k)
            if b is None:
                continue
            lost_any += a["any"] and not b["any"]
            gained_any += b["any"] and not a["any"]
            lost_br += a["br"] and a["any"] and not b["any"]
            if not (np.isnan(a["ap"]) or np.isnan(b["ap"])):
                diffs.append(b["ap"] - a["ap"])
        d_page.append(diffs)
    flat = np.array([x for d in d_page for x in d])
    rng = np.random.default_rng(seed)
    stats = []
    nonempty = [np.array(d) for d in d_page if d]
    for _ in range(boot):
        pick = rng.integers(0, len(nonempty), len(nonempty))
        sample = np.concatenate([nonempty[i] for i in pick])
        stats.append(sample.mean())
    lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats else (float("nan"), float("nan")))
    return {"dap": float(flat.mean()) if len(flat) else float("nan"), "dap_lo": float(lo), "dap_hi": float(hi),
            "code_same": f"{code_same}/{code_n}", "lost_any": int(lost_any), "gained_any": int(gained_any), "lost_br": int(lost_br)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", type=Path, nargs="+", help="PNG 폴더(들) 다음에 이름=폴더")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    page_dirs = [p for p in a.pages if "=" not in str(p)]
    arms = [str(p).split("=", 1) for p in a.pages if "=" in str(p)]
    docs = {d["stem"]: d for d in trading_docs()}
    cats = field_categories()

    measured = []
    for name, folder in arms:
        folder = Path(folder)
        arm = load_arm(folder)
        m = {}
        for pd in page_dirs:
            m.update(measure(folder, {k: v for k, v in arm.items() if (pd / f"{k}.words.json").exists()}, pd, docs, cats))
        measured.append((name, m))

    rows = [summarize(n, m) for n, m in measured]
    ref_name, ref = measured[0]
    print(f"{'계획':<14}{'쪽':>5}{'서식':>8}{'어느크롭':>9}{'자기카테':>9}{'문서번호':>9}{'AP':>8}{'ΔAP [95%]':>26}{'같은판정':>10}{'잃음':>6}{'번호잃음':>8}{'인코딩ms':>9}{'히트맵ms':>9}{'첫쪽ms':>9}")
    out = []
    for (name, m), s in zip(measured, rows):
        d = paired(ref, m, a.boot) if name != ref_name else {"dap": 0.0, "dap_lo": 0.0, "dap_hi": 0.0, "code_same": "-", "lost_any": 0, "gained_any": 0, "lost_br": 0}
        s.update(d)
        out.append(s)
        print(f"{name:<14}{s['pages']:>5}{s['code_acc']:>8.1%}{s['any']:>9.1%}{s['cat']:>9.1%}{s['br']:>8.1%}({s['br_n']}){s['ap']:>8.4f}"
              f"{d['dap']:>+10.4f} [{d['dap_lo']:+.4f},{d['dap_hi']:+.4f}]{d['code_same']:>10}{d['lost_any']:>6}{d['lost_br']:>8}"
              f"{s['encode_ms']:>9.0f}{s['heat_ms_median']:>9.0f}{s['heat_ms_first']:>9.0f}")
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
