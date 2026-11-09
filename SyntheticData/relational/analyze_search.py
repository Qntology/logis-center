"""
검색 해석 비교 (v2)
==================

  python analyze_search.py <검색 결과.jsonl> <채점 json(score_relational --json)> [--template C1] [--langs ko,en,de]

사례(틀 × 쇼핑몰)마다 언어별로 질의가 어떻게 해석됐는지(계획의 유형 · 조건 · 기간 연결)와 점수를 나란히 보여 줍니다.
가장 잘한 언어와 같은 사례에서 다른 언어가 무엇을 다르게 해석했는지 찾는 데 씁니다.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def plan_brief(plan: dict) -> str:
    types = plan.get("types") or [plan.get("type")]
    conds = [f"{c.get('path', '').replace('data.', '')} {c.get('op')} {str(c.get('value'))[:24]}" for c in plan.get("conditions") or []]
    relay = plan.get("relay") or {}
    r = ""
    if relay:
        r = f" relay[{relay.get('key')} via {relay.get('partner_types')} applied={relay.get('applied')} targets={len(relay.get('target_ids') or [])}]"
    alts = ",".join(sorted((plan.get("alternates") or {}).keys()))
    return f"{'/'.join(t for t in types if t)} {{{'; '.join(conds)}}}" + r + (f" alts:{alts}" if alts else "")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("search")
    ap.add_argument("scores")
    ap.add_argument("--template", default="")
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    res = {r["id"]: r for r in read_jsonl(a.search)}
    scores = {r["id"]: r for r in json.loads(Path(a.scores).read_text(encoding="utf-8"))}
    langs = [x for x in a.langs.split(",") if x]
    cases = defaultdict(list)
    for sid, s in scores.items():
        if a.template and s["template"] != a.template:
            continue
        if langs and s["lang"] not in langs:
            continue
        cases[(s["template"], s["shop"])].append(s)
    for (tmpl, shop), rows in sorted(cases.items()):
        print(f"\n=== {tmpl} {shop} ({rows[0]['kind']}, 데이터 {rows[0]['data_lang']}) ===")
        for s in sorted(rows, key=lambda r: -r["mrr"]):
            r = res.get(s["id"]) or {}
            search = r.get("search") or {}
            plans = search.get("dexie_plans") or []
            q = (search.get("structured") or {}).get("original_text") or ""
            print(f"  {s['lang']:7s} MRR {s['mrr']:.2f} R@all {s['rall']:.2f} 화면 {s['shown']:3d} | {q[:70]}")
            for p in plans[:3]:
                print(f"           · {plan_brief(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
