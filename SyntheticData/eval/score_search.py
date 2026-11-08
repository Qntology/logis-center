"""
검색 채점
=========

  python score_search.py <결과.jsonl> <저장 아이템 결과.jsonl ...> [--domain commerce|shipping] [--split dev]

첫 인자는 search 계획의 결과, 나머지는 같은 평가 폴더에서 돌린 추출 결과입니다. 추출 결과로 저장 아이템 id 와
정답 레코드를 잇습니다. 정답 레코드가 저장되지 않았으면 그 질의의 관련 집합에서 빼지 않고 놓친 것으로 셉니다.
그래야 추출이 놓친 레코드도 '문서 전처리부터 자연어 검색까지' 의 실패로 잡힙니다.

지표는 질의 종류별 MRR · Recall@10 · nDCG@10 입니다. 관련 레코드가 여럿인 질의는 Recall@10 을 min(10, 관련 수) 로 나눕니다.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from apply_plans import visible_ids
from truth import commerce_queries, commerce_records, multilingual_queries, trading_docs, trading_queries


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def commerce_id_map(extractions: list) -> dict:
    """저장 아이템 id → 레코드 키. 원본 식별자는 아이템의 text · link 에 남아 있습니다."""
    records = commerce_records()
    by_raw = defaultdict(list)
    for r in records:
        by_raw[str(r["id"])].append(r)
    out = {}
    for res in extractions:
        shop = res["id"].split("-")[0]
        for item in res.get("items") or []:
            blob = (item.get("text") or "") + json.dumps(item.get("data") or {}, ensure_ascii=False)
            kind = {"sales": "order", "order": "order", "goods": "goods", "tracking": "tracking"}.get(item.get("type"))
            for raw, rs in by_raw.items():
                if raw in blob:
                    for r in rs:
                        if r["shop"] == shop and (kind is None or r["type"] == kind):
                            out[item["id"]] = r["key"]
    return out


def trading_id_map(extractions: list) -> dict:
    numbers = {str(d["truth"]["doc_number"][0]): d["stem"] for d in trading_docs()}
    out = {}
    for res in extractions:
        for item in res.get("items") or []:
            data = item.get("data") or {}
            no = str(data.get("doc_number") or data.get("no") or "")
            if no in numbers:
                out[item["id"]] = numbers[no]
    return out


def rank_metrics(ranked: list, relevant: set, k: int = 10) -> dict:
    rr = 0.0
    for i, key in enumerate(ranked, start=1):
        if key in relevant:
            rr = 1.0 / i
            break
    top = ranked[:k]
    hit = len(set(top) & relevant)
    recall = hit / max(1, min(k, len(relevant)))
    dcg = sum(1.0 / math.log2(i + 2) for i, key in enumerate(top) if key in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"mrr": rr, "recall": recall, "ndcg": dcg / ideal if ideal else 0.0, "hit1": 1.0 if ranked[:1] and ranked[0] in relevant else 0.0}


def seed_table(path: Path, id_map: dict) -> dict:
    """시드 계획을 Dexie 행 표로 바꿉니다. id 는 저장 해시입니다."""
    by_key = {r["id"]: r for r in read_jsonl(path) if r.get("kind") == "seed"}
    table = {}
    for item_id, key in id_map.items():
        r = by_key.get(key)
        if r is not None:
            table[item_id] = {"id": item_id, "type": r["type"], "mode": "commerce", "data": r["data"], "shop": key.split(":")[0]}
    return table


def score(search_results: list, truth_queries: list, id_map: dict, table: dict | None = None, by: str = "kind") -> dict:
    per_kind = defaultdict(list)
    detail = []
    for res in search_results:
        q = truth_queries[int(res["id"][1:])]
        rows = ((res.get("search") or {}).get("results")) or []
        if table is not None:
            ids = visible_ids(rows, (res.get("search") or {}).get("dexie_plans") or [], table, q.get("shop", ""))
            rows = [{"id": i} for i in ids]
        ranked, seen = [], set()
        for row in rows:
            key = id_map.get(row.get("id"))
            if key is None:
                key = f"?{row.get('id')}"
            if key not in seen:
                seen.add(key)
                ranked.append(key)
        m = rank_metrics(ranked, set(q["relevant"]))
        per_kind[q[by]].append(m)
        per_kind["(전체)"].append(m)
        detail.append({"query": q["query"], "kind": q["kind"], "lang": q["lang"], **m, "top": ranked[:5], "relevant": q["relevant"][:5]})
    table = {}
    for kind, ms in per_kind.items():
        table[kind] = {"n": len(ms), **{f: sum(m[f] for m in ms) / len(ms) for f in ("mrr", "recall", "ndcg", "hit1")}}
    return {"table": table, "detail": detail}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("search", type=Path)
    ap.add_argument("extractions", type=Path, nargs="*")
    ap.add_argument("--seed-map", type=Path, help="정답 레코드를 바로 색인한 실행의 id 표 (seed_ids)")
    ap.add_argument("--domain", default="commerce", choices=["commerce", "shipping"])
    ap.add_argument("--dump", type=Path, help="질의별 상세를 JSON 으로 씁니다")
    ap.add_argument("--seeds", type=Path, help="시드 계획. 주면 프론트엔드 조건 필터를 거친 화면 순위로 채점합니다")
    ap.add_argument("--queries", default="default", choices=["default", "12lang"],
                    help="12lang: queries_12lang.yaml (make_plan.py ml12 · mlship12 계획의 결과)")
    ap.add_argument("--by", default="kind", choices=["kind", "lang", "key"], help="표를 나눌 기준")
    a = ap.parse_args()
    extractions = []
    for p in a.extractions:
        extractions += read_jsonl(p)
    if a.domain == "commerce":
        queries, id_map = commerce_queries(), commerce_id_map(extractions)
    else:
        queries, id_map = trading_queries(), trading_id_map(extractions)
    if a.queries == "12lang":
        queries = multilingual_queries(a.domain)
    if a.seed_map:
        id_map.update(json.loads(a.seed_map.read_text(encoding="utf-8")))
    table = seed_table(a.seeds, id_map) if a.seeds else None
    out = score(read_jsonl(a.search), queries, id_map, table, a.by)
    print(f"연결된 아이템 {len(id_map)}건")
    print(f"{'종류':18s} {'질의':>5s} {'MRR':>6s} {'R@10':>6s} {'nDCG':>6s} {'Hit@1':>6s}")
    for kind, row in sorted(out["table"].items()):
        print(f"{kind:18s} {row['n']:5d} {row['mrr']:6.3f} {row['recall']:6.3f} {row['ndcg']:6.3f} {row['hit1']:6.3f}")
    if a.dump:
        a.dump.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
