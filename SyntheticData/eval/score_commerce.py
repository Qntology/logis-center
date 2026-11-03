"""
커머스 추출 채점 (전 과정)
=========================

  python score_commerce.py <결과.jsonl> [--split dev|test|all]

저장된 아이템을 페이지 정답과 맞춥니다. 레코드는 원본 식별자로 잇습니다(저장 id 는 해시이고 원본은 link · text 에 남습니다).

레코드 재현율 : 정답 레코드 중 저장된 비율
필드 재현율   : 정답 값 중 맞게 저장된 비율 (스키마에 있는 필드만)
필드 정밀도   : 저장된 값 중 맞는 비율. 정답에 없는 값(스키마 밖 숫자를 끌어온 경우 등)을 잡습니다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from normalize import contains_value, same_value
from truth import ROOT, commerce_pages

BIAS = ROOT.parent / "commerce" / "src-tauri" / "src" / "utils" / "bias.json"
STATUS_CODE = {
    "progress": 1, "stop": 2, "cancel": 3, "refund": 4, "return": 5, "error": 6, "expire": 7,
    "exchange": 8, "complete": 9, "draft": 10, "show": 11, "hide": 12,
}
SYSTEM_KEYS = {
    "id", "index", "cc", "bcc", "ref", "from", "to", "digest", "embed", "ledger", "detail", "link", "mode", "type",
    "created_at", "updated_at", "text", "masked_text", "no", "tags", "currency", "general_insight", "conversion_insight",
    "traffic_insight",
}
# 값이 레코드 안 다른 문서를 가리키는 키는 relay 가 숫자로 바꾸고 원래 글자를 *_title 에 둡니다.
TITLE_SUFFIX = "_title"


# bias.json 의 exact_match 에 없는 흔한 상태어. 뜻이 둘로 읽히는 말은 둘 다 맞은 것으로 칩니다.
STATUS_EXTRA = {
    "품절": ("stop",), "sold out": ("stop",), "active": ("show", "progress"), "입금대기": ("draft",),
    "awaiting payment": ("draft",), "paid": ("progress",), "shipped": ("progress",),
    "대기중": ("draft", "hide", "stop", "progress"), "pending": ("draft", "hide", "stop", "progress"),
    "scheduled": ("draft", "hide", "stop"), "종료": ("expire", "complete"), "ended": ("expire", "complete"),
    "live": ("progress", "show"), "승인": ("complete", "show", "progress"), "approved": ("complete", "show", "progress"),
}


def status_table() -> dict:
    """상태 라벨(소문자) → 허용하는 표준 상태들."""
    data = json.loads(BIAS.read_text(encoding="utf-8"))["status_filters"]
    out = {}
    for name, node in data.items():
        for word in node.get("exact_match", []):
            out.setdefault(word.lower(), (name,))
    out.update(STATUS_EXTRA)
    return out


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def raw_id_of(row: dict) -> str:
    return str(row.get("id,link") or "")


def field_value(data: dict, field: str):
    for k in (field, field + TITLE_SUFFIX):
        v = data.get(k)
        if v not in (None, "", 0, [], {}):
            return v
    # 0 이 정답인 필드(재고 0 · 배송비 0)는 그대로 읽습니다.
    return data.get(field)


def check_field(field: str, truth, data: dict, statuses: dict) -> str:
    """맞음 / 틀림 / 없음."""
    got = field_value(data, field)
    if field == "status":
        want = statuses.get(str(truth).lower())
        if want is None:
            return "맞음" if got not in (None, 0, "") else "없음"
        if got in (None, 0, ""):
            return "없음"
        return "맞음" if got in [STATUS_CODE[w] for w in want] else "틀림"
    if got is None or got == "":
        return "없음"
    if same_value(truth, got):
        return "맞음"
    if field in ("goods", "title") and contains_value(truth, got):
        return "맞음"
    if isinstance(got, (int, float)) and got == 0 and not isinstance(truth, (int, float)):
        return "없음"
    return "틀림"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--split", default="all")
    ap.add_argument("--errors", type=int, default=0, help="틀린 값 예시를 N 건 출력합니다")
    ap.add_argument("--pool", type=Path, help="dump 계획의 결과. 주면 페이지별 아이템 대신 저장소 끝 상태로 레코드를 찾습니다")
    a = ap.parse_args()
    results = {r["id"]: r for r in read_jsonl(a.results)}
    if a.pool:
        # 상세가 앞서 저장된 목록 문서에 합쳐지면 그 페이지의 결과에는 아이템이 남지 않습니다. 처리 순서에 좌우되지 않게 끝 상태를 씁니다.
        pool = [it for r in read_jsonl(a.pool) for it in (r.get("items") or [])]
        hosts = {p["page_id"]: urlparse(p["url"]).hostname for p in commerce_pages()}
        def same_site(item: dict, pid: str) -> bool:
            # 목록 행의 링크는 '/admin/order/view?...' 처럼 상대 경로라 호스트가 있을 때만 사이트를 가립니다.
            link = str((item.get("data") or {}).get("link") or "")
            return not link.startswith("http") or hosts.get(pid, "") in link

        results = {pid: {**res, "items": [it for it in pool if same_site(it, pid)]} for pid, res in results.items()}
    statuses = status_table()
    tally = defaultdict(Counter)
    per_field = defaultdict(Counter)
    errors = []
    for p in commerce_pages():
        if (a.split != "all" and p["split"] != a.split) or p["page_id"] not in results:
            continue
        res = results[p["page_id"]]
        items = res.get("items") or []
        rows = p.get("items") if p["layout"] == "list" else [p["fields"]]
        group_keys = (p["theme"], f"유형:{p['page_type']}", f"레이아웃:{p['layout']}", "(전체)")
        for row in rows:
            rid = raw_id_of(row)
            match = None
            for it in items:
                blob = str((it.get("data") or {}).get("link") or "") + " " + (it.get("text") or "")
                if rid and rid in blob:
                    match = it
                    break
            for g in group_keys:
                tally[g]["레코드"] += 1
                tally[g]["레코드 저장"] += match is not None
            if match is None:
                for field, v in row.items():
                    if field == "id,link":
                        continue
                    for g in group_keys:
                        tally[g]["값"] += 1
                        tally[g]["없음"] += 1
                    per_field[(p["page_type"], field)]["없음"] += 1
                continue
            data = match.get("data") or {}
            for field, v in row.items():
                if field == "id,link":
                    continue
                verdict = check_field(field, v, data, statuses)
                for g in group_keys:
                    tally[g]["값"] += 1
                    tally[g][verdict] += 1
                per_field[(p["page_type"], field)][verdict] += 1
                if verdict == "틀림" and len(errors) < 4000:
                    errors.append((p["page_id"], field, v, field_value(data, field)))
    print(f"{'묶음':24s} {'레코드':>10s} {'재현':>7s} {'값':>6s} {'맞음':>7s} {'틀림':>7s} {'없음':>7s} {'정밀도':>7s}")
    for g in sorted(tally):
        t = tally[g]
        n = max(1, t["값"])
        got = t["맞음"] + t["틀림"]
        print(f"{g:24s} {t['레코드 저장']:4d}/{t['레코드']:<5d} {t['레코드 저장'] / max(1, t['레코드']):7.1%} {t['값']:6d} {t['맞음'] / n:7.1%} {t['틀림'] / n:7.1%} {t['없음'] / n:7.1%} {t['맞음'] / max(1, got):7.1%}")
    worst = sorted(per_field.items(), key=lambda kv: (kv[1]["맞음"] / max(1, sum(kv[1].values())), -sum(kv[1].values())))
    print("\n필드별 (낮은 순)")
    for (ptype, field), c in worst[:24]:
        total = sum(c.values())
        print(f"  {ptype:9s} {field:22s} {c['맞음']:4d}/{total:<4d} 맞음 {c['맞음'] / total:6.1%} 틀림 {c['틀림']:4d} 없음 {c['없음']:4d}")
    for e in errors[: a.errors]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
