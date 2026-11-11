"""
비전 게이트 합성 평가 채점
==========================

  python score_vision_gates.py legibility <vision_gates 결과.jsonl> [...]   출처 판독성 (verify_claims_v2)
  python score_vision_gates.py labels <label_gates 결과.jsonl>             라벨 → 축 라우팅
  python score_vision_gates.py names <names.jsonl> [<name_residue 결과.jsonl>]  인명 잔해

판독성: 열화로 못 읽게 된 값(smudge · blot)을 버린 비율(재현율)과, 읽히는 값(clean · light · stamp · sign, 열화 밖의 값)을
버린 비율(오폐기율). calib 쪽(영어 dev, 열화 없음)은 SDS 기준선을 쌓는 데만 쓰고 채점하지 않습니다.
라벨: 라우팅된 축이 정답인 비율 · 틀린 축으로 간 비율 · 버려진 비율. 인명: 잔해를 깎아 원래 이름이 된 비율 · 정당한 이름을 그대로 둔 비율.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def rows_of(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else "0/0"


def legibility(paths: list) -> None:
    for path in paths:
        rows = rows_of(path)
        kind_of = rows[0]["bbox_kind"] if rows else "?"
        by = defaultdict(Counter)
        ratios = defaultdict(list)
        for r in rows:
            if r["kind"] == "calib":
                continue
            for c in r["claims"]:
                tot = max(1, c["lg"] + c["il"] + c["bl"])
                key = (r["kind"], c["expect"])
                by[key]["n"] += 1
                by[key]["dropped"] += 0 if c["kept"] else 1
                by[key]["il_major"] += 1 if c["il"] > c["lg"] else 0
                ratios[c["expect"]].append(c["il"] / tot)
        print(f"\n== 판독성 · 출처 상자 {kind_of} · {path}")
        print("| 열화 | 정답 | 주장 | 버림 | il > lg |")
        print("|---|---|---:|---:|---:|")
        for (kind, exp), cnt in sorted(by.items()):
            print(f"| {kind} | {exp} | {cnt['n']} | {pct(cnt['dropped'], cnt['n'])} | {pct(cnt['il_major'], cnt['n'])} |")
        rej = [v for (k, e), v in by.items() if e == "reject"]
        keep = [v for (k, e), v in by.items() if e == "keep"]
        rn, rd = sum(v["n"] for v in rej), sum(v["dropped"] for v in rej)
        kn, kd = sum(v["n"] for v in keep), sum(v["dropped"] for v in keep)
        print(f"재현율(못 읽는 값을 버림) {pct(rd, rn)} · 오폐기율(읽히는 값을 버림) {pct(kd, kn)}")
        for exp, xs in sorted(ratios.items()):
            xs = sorted(xs)
            q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
            print(f"  판독불가 비중 [{exp}] n={len(xs)} 중앙 {q(0.5):.3f} · 90% {q(0.9):.3f} · 99% {q(0.99):.3f}")


def labels(path: str) -> None:
    rows = rows_of(path)
    tot = Counter()
    per_lang = defaultdict(Counter)
    per_field = defaultdict(Counter)
    wrong = Counter()
    for r in rows:
        ok = r["routed"] == r["expect"]
        state = "ok" if ok else ("drop" if r["routed"] is None else "wrong")
        tot[state] += 1
        per_lang[r["lang"]][state] += 1
        per_field[r["expect"]][state] += 1
        if state == "wrong":
            wrong[(r["expect"], r["routed"])] += 1
    n = len(rows)
    print(f"\n== 라벨 라우팅 · {path}")
    print(f"정답 {pct(tot['ok'], n)} · 틀린 축 {pct(tot['wrong'], n)} · 버림 {pct(tot['drop'], n)}")
    print("| 언어 | 정답 | 틀린 축 | 버림 |")
    print("|---|---:|---:|---:|")
    for lang, c in sorted(per_lang.items()):
        m = sum(c.values())
        print(f"| {lang} | {pct(c['ok'], m)} | {c['wrong']} | {c['drop']} |")
    print("| 정답 축 | 정답 | 틀린 축 | 버림 |")
    print("|---|---:|---:|---:|")
    for f, c in sorted(per_field.items()):
        m = sum(c.values())
        print(f"| {f} | {pct(c['ok'], m)} | {c['wrong']} | {c['drop']} |")
    print("틀린 축 상위:", ", ".join(f"{a}→{b} {k}" for (a, b), k in wrong.most_common(12)))


RESIDUE = set("-–—_/|\\~")


def observe_flag(v: str) -> bool:
    """현재 코드의 관측 규칙: 인명 축 값의 양끝이 영숫자가 아니면 잔해로 표시."""
    t = v.strip()
    return bool(t) and (not t[0].isalnum() or not t[-1].isalnum())


def names(cases: str, result: str = "") -> None:
    rows = rows_of(cases)
    out = {r["id"]: r for r in rows_of(result)} if result else {}
    c = defaultdict(Counter)
    for r in rows:
        got = out[r["id"]]["value"] if r["id"] in out else r["value"]
        kind = "residue" if r["kind"].startswith("residue") else "legit"
        c[kind]["n"] += 1
        c[kind]["right"] += 1 if got == r["expect"] else 0
        c[kind]["flag"] += 1 if observe_flag(r["value"]) else 0
    print(f"\n== 인명 잔해 · {result or '(손대지 않음)'}")
    for kind in ("residue", "legit"):
        print(f"  {kind}: 결과가 정답 {pct(c[kind]['right'], c[kind]['n'])} · 관측 규칙(양끝 비영숫자) 표시 {pct(c[kind]['flag'], c[kind]['n'])}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "legibility":
        legibility(args)
    elif mode == "labels":
        labels(args[0])
    elif mode == "names":
        names(args[0], args[1] if len(args) > 1 else "")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
