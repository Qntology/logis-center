"""
SigLIP2 판정 채점
=================

  python score_vision.py <vision_probe 결과.jsonl> <PNG 폴더> [--split dev|test|all]

서식 판정: 첫 페이지의 코드가 정답과 같은가. 쌍둥이 코드(제목이 같은 코드)는 맞은 것으로 함께 셉니다.
크롭 커버리지: 정답 값이 인쇄된 위치(단어 좌표)가 크롭 안에 들어오는가. '어느 크롭이든' 과 '그 값의 카테고리 크롭' 을 나눠 봅니다.
마진 판별력: 판정이 맞은 페이지와 틀린 페이지를 마진이 얼마나 가르는가(AUC). SDS 가 쌓는 마진이 쓸모 있는지 보는 값입니다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from truth import ROOT, trading_docs

TWINS = {"BC": "BK", "CA": "COA", "PHYTO": "PC", "INS": "IP"}
BIAS = ROOT.parent / "commerce" / "src-tauri" / "src" / "utils" / "bias.json"


def same_code(a: str, b: str) -> bool:
    return TWINS.get(a, a) == TWINS.get(b, b)


def field_categories() -> dict:
    base = json.loads(BIAS.read_text(encoding="utf-8"))["trade_schema"]["base"]
    return {f: cat for cat, fields in base.items() for f in fields}


def auc(pos: list, neg: list) -> float:
    """pos 가 neg 보다 높게 나올 확률. 표본이 없으면 nan."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def find_value_box(value, words: list):
    """값의 낱말이 연속으로 인쇄된 자리의 경계 상자. 없으면 None."""
    tokens = [t.strip(",.:;()[]").lower() for t in str(value).split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    norm = [w["text"].strip(",.:;()[]").lower() for w in words]
    for i in range(len(norm) - len(tokens) + 1):
        if norm[i: i + len(tokens)] == tokens:
            run = words[i: i + len(tokens)]
            return (min(w["x0"] for w in run), min(w["y0"] for w in run),
                    max(w["x1"] for w in run), max(w["y1"] for w in run))
    return None


def inside(box, plan_box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return plan_box[0] <= cx <= plan_box[2] and plan_box[1] <= cy <= plan_box[3]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("pages", type=Path)
    ap.add_argument("--split", default="all")
    ap.add_argument("--dump", type=Path)
    a = ap.parse_args()
    docs = {d["stem"]: d for d in trading_docs() if a.split == "all" or d["split"] == a.split}
    cats = field_categories()
    rows = [json.loads(l) for l in a.results.read_text(encoding="utf-8").splitlines() if l.strip()]

    first = []
    cover = Counter()
    per_cat = defaultdict(Counter)
    plan_sizes = []
    for r in rows:
        stem, page = r["page"].rsplit("_p", 1)
        if stem not in docs:
            continue
        doc = docs[stem]
        if page == "1":
            first.append({
                "stem": stem, "truth": doc["code"], "code": r["code"], "ok": same_code(r["code"], doc["code"]),
                "margin": r["code_margin"], "title_confirmed": r["title_confirmed"],
                "top5": [c for c, _ in r["candidates"][:5]],
            })
        meta = json.loads((a.pages / f"{r['page']}.words.json").read_text(encoding="utf-8"))
        words = meta["words"]
        area = meta["width"] * meta["height"]
        plans = r["plans"]
        plan_sizes.append(sum((p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) for p in plans) / area)
        for field, values in doc["truth"].items():
            cat = cats.get(field)
            for v in values:
                box = find_value_box(v, words)
                if box is None:
                    continue
                cover["값"] += 1
                any_hit = any(inside(box, p["bbox"]) for p in plans)
                cat_hit = any(p["category"] == cat and inside(box, p["bbox"]) for p in plans)
                cover["어느 크롭"] += any_hit
                cover["자기 카테고리 크롭"] += cat_hit
                if cat:
                    per_cat[cat]["값"] += 1
                    per_cat[cat]["hit"] += cat_hit

    n = len(first)
    ok = sum(x["ok"] for x in first)
    top5 = sum(any(same_code(c, x["truth"]) for c in x["top5"]) for x in first)
    print(f"서식 판정 {ok}/{n} = {ok / max(1, n):.1%} | 후보 상위 5 안 {top5}/{n} = {top5 / max(1, n):.1%} | 제목 확인 {sum(x['title_confirmed'] for x in first)}/{n}")
    print(f"마진 AUC(맞음 vs 틀림) {auc([x['margin'] for x in first if x['ok']], [x['margin'] for x in first if not x['ok']]):.3f}")
    print("예측 분포", Counter(x["code"] for x in first).most_common(8))
    if cover["값"]:
        print(f"크롭 커버리지 값 {cover['값']}개 | 어느 크롭 {cover['어느 크롭'] / cover['값']:.1%} | 자기 카테고리 크롭 {cover['자기 카테고리 크롭'] / cover['값']:.1%}")
    print(f"페이지 대비 크롭 면적 합 평균 {sum(plan_sizes) / max(1, len(plan_sizes)):.2f}")
    for cat, c in sorted(per_cat.items()):
        print(f"  {cat:14s} {c['hit']:4d}/{c['값']:4d} = {c['hit'] / max(1, c['값']):.1%}")
    if a.dump:
        a.dump.write_text(json.dumps(first, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
