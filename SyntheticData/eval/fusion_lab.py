"""
검색 융합 실험
==============

  python fusion_lab.py <결과.jsonl> --seed-map <id 표> --seeds <시드 계획>

앱이 응답에 남긴 트랙별 순위(tracks)로 가중치를 다시 합쳐 재실행 없이 비교합니다. dev 질의로만 고릅니다.
화면 순위는 score_search 와 같이 프론트엔드 조건 필터를 거친 뒤 잽니다.
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

from apply_plans import visible_ids
from score_search import rank_metrics, read_jsonl, seed_table
from truth import commerce_queries


def fuse(lists: list, k: float) -> list:
    score = defaultdict(float)
    for ids, w in lists:
        for rank, i in enumerate(ids):
            score[i] += w / (k + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))


DOMAIN_WORDS = {
    "order": ["order", "purchase", "주문", "구매"],
    "tracking": ["tracking", "shipment", "shipping", "delivery", "parcel", "배송", "택배", "송장", "운송"],
    "review": ["review", "리뷰", "후기"],
    "coupon": ["coupon", "voucher", "쿠폰"],
    "event": ["event", "campaign", "promotion", "이벤트", "기획전", "프로모션", "캠페인"],
}


def query_domains(text: str) -> list:
    """질의가 가리키는 유형. 가리키는 낱말이 없으면 상품입니다."""
    words = text.lower().split()
    out = [cat for cat, vocab in DOMAIN_WORDS.items() if any(v in w if not v.isascii() else w.startswith(v) for w in words for v in vocab)]
    return out or ["goods"]


def evaluate(rows: list, queries: list, id_map: dict, table: dict, order) -> dict:
    per = defaultdict(list)
    for res in rows:
        q = queries[int(res["id"][1:])]
        s = dict(res.get("search") or {})
        s["_prior"] = query_domains(q["query"])
        s["_table"] = table
        ids = order(s.get("tracks") or {}, s)
        vis = visible_ids([{"id": i} for i in ids], s.get("dexie_plans") or [], table, q.get("shop", ""))
        ranked, seen = [], set()
        for i in vis:
            key = id_map.get(i, f"?{i}")
            if key not in seen:
                seen.add(key)
                ranked.append(key)
        m = rank_metrics(ranked, set(q["relevant"]))
        per[q["kind"]].append(m)
        per["(전체)"].append(m)
    return {k: (len(v), sum(m["mrr"] for m in v) / len(v), sum(m["recall"] for m in v) / len(v)) for k, v in per.items()}


def show(name: str, out: dict) -> None:
    kinds = sorted(out)
    print(f"{name:42s} " + " ".join(f"{k[:9]:>9s}" for k in kinds))
    print(f"{'  MRR':42s} " + " ".join(f"{out[k][1]:9.3f}" for k in kinds))
    print(f"{'  R@10':42s} " + " ".join(f"{out[k][2]:9.3f}" for k in kinds))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--seed-map", type=Path, required=True)
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--grid", action="store_true", help="가중치 격자를 훑어 상위 설정을 출력합니다")
    a = ap.parse_args()
    rows = [r for r in read_jsonl(a.results) if (r.get("search") or {}).get("tracks")]
    queries = commerce_queries()
    id_map = json.loads(a.seed_map.read_text(encoding="utf-8"))
    table = seed_table(a.seeds, id_map)
    print(f"질의 {len(rows)}")

    def track(name):
        return lambda t, s: t.get(name) or []

    for name, key in (("해석 순위만", "pipeline"), ("질의 벡터만", "vector"), ("질의 청크만", "chunks"), ("글자 일치만", "literal"), ("유형 벡터만", "typed_vector"), ("유형 청크만", "typed_chunks")):
        show(name, evaluate(rows, queries, id_map, table, track(key)))

    def weighted(wp, wv, wc, wl, wtv, wtc, wtl, k):
        def order(t, s):
            tb = s["_table"]
            domains = t.get("domains") or []
            lit = t.get("literal") or []
            typed_lit = [i for i in lit if i in tb and tb[i]["type"] in domains]
            return fuse(
                [
                    (t.get("pipeline") or [], wp), (t.get("vector") or [], wv), (t.get("chunks") or [], wc), (lit, wl),
                    (t.get("typed_vector") or [], wtv), (t.get("typed_chunks") or [], wtc), (typed_lit, wtl),
                ],
                k,
            )
        return order

    show("앱 설정 (1,1,1,2 | 1,1,1 k=20)", evaluate(rows, queries, id_map, table, weighted(1, 1, 1, 2, 1, 1, 1, 20)))
    if a.grid:
        best = []
        grid = itertools.product((0, 0.5, 1), (0, 1), (0, 1), (1, 2), (0.5, 1, 2), (0.5, 1, 2), (0, 1), (10, 20, 60))
        for cfg in grid:
            out = evaluate(rows, queries, id_map, table, weighted(*cfg))
            kinds = [v for key, v in out.items() if key != "(전체)"]
            macro = sum(v[1] for v in kinds) / len(kinds)
            best.append((macro, out["(전체)"][1], cfg))
        best.sort(reverse=True)
        print("종류 평균 MRR · 전체 MRR · (해석, 벡터, 청크, 글자, 유형벡터, 유형청크, 유형글자, k)")
        for row in best[:12]:
            print(f"  {row[0]:.3f} {row[1]:.3f} {row[2]}")
        show(f"최고 {best[0][2]}", evaluate(rows, queries, id_map, table, weighted(*best[0][2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
