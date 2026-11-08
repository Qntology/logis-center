"""
무역 추출 채점 (전 과정)
=======================

  python score_trading.py <결과.jsonl> [--split dev|test|all]

문서 하나의 결과에서 릴레이 초안(ledger=placeholder)을 빼고 그 문서의 아이템을 찾아 서식 코드와 필드를 봅니다.
같은 필드가 표의 여러 행에 있으면 행마다 셉니다. 저장 구조가 중첩이라 필드 이름으로 전체를 훑습니다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from normalize import same_value
from score_vision import same_code
from truth import trading_docs


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def collect(node, field: str, out: list) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == field and not isinstance(v, (dict, list)):
                out.append(v)
            else:
                collect(v, field, out)
    elif isinstance(node, list):
        for x in node:
            collect(x, field, out)


def real_items(res: dict) -> list:
    return [i for i in res.get("items") or [] if (i.get("data") or {}).get("ledger") != "placeholder"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--split", default="all")
    ap.add_argument("--errors", type=int, default=0)
    ap.add_argument("--dump", type=Path)
    a = ap.parse_args()
    results = {r["id"]: r for r in read_jsonl(a.results)}
    per_field = defaultdict(Counter)
    code_ok, doc_ok, docs_n = 0, 0, 0
    errors, doc_rows = [], []
    for d in trading_docs():
        if (a.split != "all" and d["split"] != a.split) or d["stem"] not in results:
            continue
        res = results[d["stem"]]
        docs_n += 1
        items = real_items(res)
        number = str(d["truth"]["doc_number"][0])
        match = None
        for it in items:
            found = []
            collect(it.get("data") or {}, "doc_number", found)
            if any(same_value(number, v) for v in found):
                match = it
                break
        if match is None and items:
            match = items[0]
        got_code = (match or {}).get("type") or ""
        ok_code = same_code(str(got_code).upper(), d["code"])
        code_ok += ok_code
        doc_ok += match is not None
        hit = total = 0
        for field, values in d["truth"].items():
            if field == "doc_type":
                continue
            got = []
            collect((match or {}).get("data") or {}, field, got)
            for v in values:
                total += 1
                ok = any(same_value(v, g) for g in got)
                hit += ok
                c = per_field[field]
                if ok:
                    c["맞음"] += 1
                elif got:
                    c["틀림"] += 1
                    if len(errors) < 2000:
                        errors.append((d["stem"], field, v, got[:3]))
                else:
                    c["없음"] += 1
        doc_rows.append({"stem": d["stem"], "code": d["code"], "lang": d.get("lang", "en"), "got": got_code,
                         "ok_code": ok_code, "hit": hit, "total": total})
    print(f"문서 {docs_n} | 저장 {doc_ok} | 서식 코드 {code_ok}/{docs_n} = {code_ok / max(1, docs_n):.1%}")
    t = Counter()
    for c in per_field.values():
        t.update(c)
    n = sum(t.values())
    print(f"값 {n} | 맞음 {t['맞음'] / max(1, n):.1%} | 틀림 {t['틀림'] / max(1, n):.1%} | 없음 {t['없음'] / max(1, n):.1%} | 정밀도 {t['맞음'] / max(1, t['맞음'] + t['틀림']):.1%}")
    print("필드별 (값 많은 순)")
    for f, c in sorted(per_field.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        print(f"  {f:22s} {c['맞음']:4d}/{total:<4d} {c['맞음'] / total:6.1%} 틀림 {c['틀림']:3d} 없음 {c['없음']:3d}")
    langs = sorted({r["lang"] for r in doc_rows})
    if len(langs) > 1:  # 분할 ml: 언어별 서식 코드 · 값 맞음
        print("언어별")
        for lang in langs:
            rows = [r for r in doc_rows if r["lang"] == lang]
            hit, total = sum(r["hit"] for r in rows), sum(r["total"] for r in rows)
            print(f"  {lang:8s} 문서 {len(rows):2d} | 서식 코드 {sum(r['ok_code'] for r in rows)}/{len(rows)} | "
                  f"값 {hit}/{total} = {hit / max(1, total):.1%}")
    for e in errors[: a.errors]:
        print("  ", e)
    if a.dump:
        a.dump.write_text(json.dumps(doc_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
