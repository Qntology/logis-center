"""
평가 계획 만들기
===============

tests/it/synthetic_eval.rs 가 읽는 JSONL 계획을 씁니다.

  python make_plan.py commerce --split dev --out plan.jsonl
  python make_plan.py trading  --split dev --out plan.jsonl
  python make_plan.py images   --split dev --pages-dir <PNG 폴더> --out plan.jsonl
  python make_plan.py search   --split dev --out plan.jsonl
  python make_plan.py shipsearch --split dev --out plan.jsonl
  python make_plan.py seeds    --split dev --out plan.jsonl   (정답 레코드를 바로 색인)

--only 는 id 부분 문자열 필터입니다(쉼표로 여러 개). --limit 은 앞에서 N 건만 씁니다.
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from score_commerce import STATUS_CODE, status_table
from truth import COMMERCE, commerce_pages, commerce_queries, trading_docs, trading_queries


def commerce_plan(split: str) -> list:
    out = []
    for p in commerce_pages():
        if split != "all" and p["split"] != split:
            continue
        out.append({
            "id": p["page_id"], "kind": "html", "url": p["url"], "mode": "commerce",
            "path": (COMMERCE / "output_html" / f"{p['page_id']}.html").as_posix(),
        })
    return out


def trading_plan(split: str) -> list:
    return [
        {"id": d["stem"], "kind": "document", "path": d["pdf"], "mode": "shipping"}
        for d in trading_docs() if split == "all" or d["split"] == split
    ]


def image_plan(split: str, pages_dir: Path) -> list:
    out = []
    wanted = {d["stem"] for d in trading_docs() if split == "all" or d["split"] == split}
    for png in sorted(pages_dir.glob("*.png")):
        if png.stem.rsplit("_p", 1)[0] in wanted:
            out.append({"id": f"img-{png.stem}", "kind": "image", "path": png.as_posix(), "mode": "shipping"})
    return out


def search_plan(split: str) -> list:
    hosts = {p["shop"]: urlparse(p["url"]).hostname for p in commerce_pages()}
    out = []
    for i, q in enumerate(commerce_queries()):
        if split != "all" and q["split"] != split:
            continue
        out.append({
            "id": f"q{i:04d}", "kind": "search", "query": q["query"], "mode": "commerce",
            "language": "korean", "host": hosts[q["shop"]], "truth_index": i,
        })
    return out


def ship_search_plan(split: str) -> list:
    out = []
    for i, q in enumerate(trading_queries()):
        if split != "all" and q["split"] != split:
            continue
        out.append({
            "id": f"s{i:04d}", "kind": "search", "query": q["query"], "mode": "shipping",
            "language": "korean", "host": "", "truth_index": i,
        })
    return out


def seed_plan(split: str) -> list:
    """정답 레코드를 앱이 저장하는 모양으로 바꿉니다. 목록과 상세에 나뉜 값은 합칩니다."""
    statuses = status_table()
    records = {}
    for p in commerce_pages():
        if split != "all" and p["split"] != split:
            continue
        for row in p["items"] if p["layout"] == "list" else [p["fields"]]:
            rid = str(row["id,link"])
            key = (p["shop"], p["page_type"], rid)
            rec = records.setdefault(key, {"page": p, "row": {}, "detail": False})
            rec["row"].update(row)
            rec["detail"] = rec["detail"] or p["layout"] == "detail"
    out = []
    for (shop, kind, rid), rec in records.items():
        host = urlparse(rec["page"]["url"]).hostname
        data = {k: v for k, v in rec["row"].items() if k != "id,link"}
        allowed = statuses.get(str(data.get("status", "")).lower())
        data["status"] = STATUS_CODE[allowed[0]] if allowed else 0
        data["link"] = f"https://{host}/{kind}/{rid}"
        # 번호는 앱이 저장하는 칸에도 둡니다. 없으면 식별자 조건이 정답을 지워 검색이 부당하게 나빠 보입니다.
        data.setdefault({"order": "no", "tracking": "tracking_number"}.get(kind, "code"), rid)
        data["currency"] = "KRW" if rec["page"]["lang"] == "ko" else "USD"
        if "goods" in data:
            data["goods_title"] = data.pop("goods")
        out.append({
            "id": f"{shop}:{kind}:{rid}", "kind": "seed", "host": host, "type": kind, "url": data["link"],
            "lang": rec["page"]["lang"], "detail": rec["detail"], "data": data,
        })
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["commerce", "trading", "images", "search", "shipsearch", "seeds"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--pages-dir", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.what == "commerce":
        plan = commerce_plan(a.split)
    elif a.what == "trading":
        plan = trading_plan(a.split)
    elif a.what == "images":
        plan = image_plan(a.split, a.pages_dir)
    elif a.what == "shipsearch":
        plan = ship_search_plan(a.split)
    elif a.what == "seeds":
        plan = seed_plan(a.split)
    else:
        plan = search_plan(a.split)
    if a.only:
        keys = [k for k in a.only.split(",") if k]
        plan = [x for x in plan if any(k in x["id"] for k in keys)]
    if a.limit:
        plan = plan[: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plan), encoding="utf-8")
    print(f"{a.what} {a.split}: {len(plan)}줄 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
