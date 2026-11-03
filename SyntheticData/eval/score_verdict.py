"""
이미지 경로의 서식 최종 판정 채점
================================

  python score_verdict.py <실행 로그> [--split dev|test|all]

LOGIS_SYNTH_STOP_ON="Document identified as" 로 image 계획을 돌리면 언어 모델 재판정까지 거친 최종 서식이 로그에 남습니다.
score_vision.py 는 SigLIP2 단계만 재고, 이 파일은 앱이 실제로 쓰는 값을 잽니다.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from score_vision import TWINS
from truth import trading_docs

BEGIN = re.compile(r"@@EVAL BEGIN img-(\S+?)_p\d+")
VERDICT = re.compile(r"Document identified as: \*\*(.+?)\*\*")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--split", default="all")
    a = ap.parse_args()
    truth = {d["stem"]: d for d in trading_docs()}
    got, cur = {}, None
    with a.log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BEGIN.search(line)
            if m:
                cur = m.group(1)
                continue
            m = VERDICT.search(line)
            if m and cur:
                got.setdefault(cur, m.group(1).strip())
    ok, n, wrong = 0, 0, Counter()
    for stem, verdict in sorted(got.items()):
        d = truth.get(stem)
        if d is None or (a.split != "all" and d["split"] != a.split):
            continue
        n += 1
        if TWINS.get(verdict, verdict) == TWINS.get(d["code"], d["code"]):
            ok += 1
        else:
            wrong[(d["code"], verdict)] += 1
    print(f"최종 서식 판정 {ok}/{n} = {ok / max(1, n):.1%}")
    print("틀림(정답→판정):", wrong.most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
