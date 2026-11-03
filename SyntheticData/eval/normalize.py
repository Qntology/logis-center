"""
값 비교
=======

파이프라인은 값을 정규화해서 저장합니다(가격은 숫자, 날짜는 ISO, 상태는 코드). 정답도 표기가 페이지마다 달라서
비교는 '같은 값인가' 만 봅니다. 표기 차이(쉼표 · 통화 기호 · 공백 · 대소문자)는 틀린 것으로 세지 않습니다.
"""
import re
from datetime import datetime

MONEY_CHARS = re.compile(r"[,\s₩$€£¥원]")
DATE_PATTERNS = (
    (re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})"), "ymd"),
    (re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일"), "ymd"),
    (re.compile(r"\b(\d{2})[-.](\d{1,2})[-.](\d{1,2})\b"), "ymd2"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "mdy"),
    (re.compile(r"\b([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b"), "bdy"),
)
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def to_number(v):
    """숫자로 읽히면 float, 아니면 None. '29,900원' · '$12.50' · '3.5kg' 은 숫자입니다."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    t = MONEY_CHARS.sub("", v.strip())
    t = re.sub(r"(kg|KG|cm|CM|일|days?|CBM|cbm)$", "", t)
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t)
    return None


def to_date(v):
    """날짜로 읽히면 'YYYY-MM-DD', 아니면 None. 에폭 밀리초도 받습니다."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1e11:
        return datetime.fromtimestamp(v / 1000).strftime("%Y-%m-%d")
    if not isinstance(v, str):
        return None
    for pat, kind in DATE_PATTERNS:
        m = pat.search(v)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            if kind == "ymd":
                return f"{int(a):04d}-{int(b):02d}-{int(c):02d}"
            if kind == "ymd2":
                return f"{2000 + int(a):04d}-{int(b):02d}-{int(c):02d}"
            if kind == "mdy":
                return f"{int(c):04d}-{int(a):02d}-{int(b):02d}"
            return f"{int(c):04d}-{MONTHS[a]:02d}-{int(b):02d}"
        except (KeyError, ValueError):
            return None
    return None


def fold(v) -> str:
    """대소문자 · 공백 · 문장부호를 무시한 글자열."""
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def same_value(truth, got) -> bool:
    if got is None or got == "":
        return False
    tn, gn = to_number(truth), to_number(got)
    if tn is not None and gn is not None:
        return abs(tn - gn) <= 1e-6 * max(1.0, abs(tn))
    td = to_date(truth)
    if td is not None:
        return td == to_date(got)
    ft, fg = fold(truth), fold(got)
    return bool(ft) and ft == fg


def contains_value(truth, got) -> bool:
    """정답이 값 안에 들어 있으면 참. '하늘상점 외 1건' 처럼 앞뒤가 붙은 값의 부분 점수에 씁니다."""
    if same_value(truth, got):
        return True
    ft, fg = fold(truth), fold(got)
    return bool(ft) and bool(fg) and (ft in fg or fg in ft)
