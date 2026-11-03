"""
커머스 페이지 판정 채점 (임베딩 단계)
====================================

  python score_pages.py <실행 로그> [--split dev|test|all]

process_task 는 페이지 유형과 목록/상세 판정을 반환하지 않고 로그로만 남기므로 로그에서 읽습니다.
LOGIS_SYNTH_STOP_ON="Determined Detail Page" 로 돌리면 언어 모델 없이 이 단계까지만 돕니다.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from truth import commerce_pages

BEGIN = re.compile(r"@@EVAL BEGIN (\S+)")
PAGE_TYPE = re.compile(r"Deterministic Classified Page Type: (\w+) \(Max Score: ([\d.\-]+)\)")
DETAIL = re.compile(r"Determined Detail Page: (true|false)")
EVIDENCE = re.compile(r"\[EVIDENCE SUM\] ListFinal: ([\d.\-]+) \(track ([\d.\-]+) \+ heading ([\d.\-]+) \+ period ([\d.\-]+) \+ rows ([\d.\-]+)\) \| FormFinal: ([\d.\-]+) \(track ([\d.\-]+) \+ heading ([\d.\-]+)\)")
HEADING = re.compile(r"\[HEADING VECTOR\] '(.*?)' \| ListMaxPool: ([\d.\-]+) \| FormMaxPool: ([\d.\-]+)")
REROUTE = re.compile(r"\[MODE REROUTE\]")
SCORES = re.compile(r"^\s+(?:👑 )?\[(\w+)\] Normalized: ([\d.\-]+)")


def parse_log(path: Path) -> dict:
    out, cur = {}, None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BEGIN.search(line)
            if m:
                cur = {"scores": {}}
                out[m.group(1)] = cur
                continue
            if cur is None:
                continue
            m = PAGE_TYPE.search(line)
            if m:
                cur["page_type"] = m.group(1)
                continue
            m = DETAIL.search(line)
            if m:
                cur["layout"] = "detail" if m.group(1) == "true" else "list"
                continue
            m = EVIDENCE.search(line)
            if m:
                v = [float(x) for x in m.groups()]
                cur.update(list_final=v[0], track_l=v[1], head_l=v[2], period=v[3], rows=v[4], form_final=v[5], track_f=v[6], head_f=v[7])
                continue
            m = HEADING.search(line)
            if m:
                cur["heading"] = m.group(1)
                continue
            if REROUTE.search(line):
                cur["rerouted"] = True
            m = SCORES.search(line)
            if m:
                cur["scores"][m.group(1)] = float(m.group(2))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--split", default="all")
    ap.add_argument("--errors", action="store_true", help="틀린 페이지를 모두 출력합니다")
    a = ap.parse_args()
    log = parse_log(a.log)
    tally = defaultdict(Counter)
    confusion = Counter()
    errors = []
    for p in commerce_pages():
        if a.split != "all" and p["split"] != a.split:
            continue
        got = log.get(p["page_id"])
        if got is None:
            continue
        key_t = f"{p['theme']}"
        for group in (key_t, f"유형:{p['page_type']}", f"레이아웃:{p['layout']}", "(전체)"):
            tally[group]["n"] += 1
            tally[group]["type_ok"] += got.get("page_type") == p["page_type"]
            tally[group]["layout_ok"] += got.get("layout") == p["layout"]
            tally[group]["both_ok"] += got.get("page_type") == p["page_type"] and got.get("layout") == p["layout"]
        if got.get("page_type") != p["page_type"]:
            confusion[(p["page_type"], got.get("page_type"))] += 1
        if got.get("page_type") != p["page_type"] or got.get("layout") != p["layout"]:
            errors.append((p["page_id"], got.get("page_type"), got.get("layout"), got.get("heading"), got.get("list_final"), got.get("form_final"), got.get("rerouted", False)))
    print(f"{'묶음':24s} {'n':>4s} {'유형':>6s} {'레이아웃':>7s} {'둘 다':>6s}")
    for g in sorted(tally):
        t = tally[g]
        print(f"{g:24s} {t['n']:4d} {t['type_ok'] / t['n']:6.1%} {t['layout_ok'] / t['n']:7.1%} {t['both_ok'] / t['n']:6.1%}")
    print("유형 혼동(정답→판정):", confusion.most_common(12))
    if a.errors:
        for e in errors:
            print(" ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
