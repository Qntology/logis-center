"""
프론트엔드 조건 필터 재현
=========================

앱은 백엔드가 준 후보를 main.ts 의 executeDexiePlan 으로 거른 뒤 보여 줍니다. 후보 순위만 채점하면 잘못된 조건이 정답을
지우는 실패가 보이지 않으므로, 시드 레코드를 Dexie 행으로 보고 같은 규칙을 파이썬으로 옮겼습니다.

  · 조건이 없는 플랜이 하나라도 있으면 후보 전부가 통과합니다.
  · 조건이 있으면 후보 안에서 거르고, 표 전체에서도 다시 찾아(rescue) 뒤에 덧붙입니다.
"""
import math
import re


def read_path(row: dict, path: str):
    cur = row
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def match_condition(row: dict, cond: dict) -> bool:
    op = cond.get("op")
    if op in ("top", "bottom"):
        return True
    raw = read_path(row, cond.get("path", ""))
    if cond.get("kind") == "number":
        target = _num(cond.get("value"))
        if target is None:
            return True
        if raw is None or raw == "":
            return op == "neq"
        actual = raw if isinstance(raw, (int, float)) else _num(re.sub(r"[^\d.\-]", "", str(raw)))
        if actual is None:
            return False
        return {"gt": actual > target, "gte": actual >= target, "lt": actual < target, "lte": actual <= target, "neq": actual != target}.get(op, actual == target)
    value = cond.get("value")
    if isinstance(raw, list):
        t0 = "" if value is None else str(value).lower()
        if not t0:
            return True
        hit = any(str(el).lower().find(t0) >= 0 if op == "contains" else str(el).lower() == t0 for el in raw if el is not None)
        return not hit if op in ("not_contains", "neq") else hit
    a = "" if raw is None else str(raw).lower()
    t = "" if value is None else str(value).lower()
    if not t:
        return True
    if op == "contains":
        return t in a
    if op == "not_contains":
        return t not in a
    if op == "neq":
        return a != t
    return {"gt": a > t, "gte": a >= t, "lt": a < t, "lte": a <= t}.get(op, a == t)


def execute_plan(plan: dict, rows: list) -> list:
    conds = plan.get("conditions") or []
    allowed = plan.get("types") or ([plan["type"]] if plan.get("type") else [])
    if allowed:
        rows = [r for r in rows if r.get("type") in allowed]
    if plan.get("mode"):
        rows = [r for r in rows if (r.get("mode") or "commerce") == plan["mode"]]
    plain = [c for c in conds if c.get("op") not in ("top", "bottom")]
    rows = [r for r in rows if all(match_condition(r, c) for c in plain)]
    for rc in [c for c in conds if c.get("op") in ("top", "bottom")]:
        if not rows:
            break
        ranked = [r for r in rows if _num(read_path(r, rc["path"])) is not None]
        if not ranked:
            continue
        pct = max(1, min(100, rc.get("percent") or 20))
        take = max(1, math.ceil(len(ranked) * pct / 100))
        ranked.sort(key=lambda r: _num(read_path(r, rc["path"])), reverse=rc["op"] == "top")
        if rc.get("extreme"):
            edge = _num(read_path(ranked[0], rc["path"]))
            rows = [r for r in ranked if _num(read_path(r, rc["path"])) == edge]
        else:
            rows = ranked[:take]
    return rows


def visible_ids(results: list, plans: list, table: dict, shop: str = "") -> list:
    """화면에 나오는 순서의 id 목록. table 은 {id: Dexie 행} 입니다. 플랜에 cc 가 있으면 질의의 쇼핑몰(shop)로 좁힙니다."""
    ids = [r.get("id") for r in results if r.get("id")]
    if any(p.get("cc") for p in plans):
        table = {k: v for k, v in table.items() if v.get("shop") == shop}
    if not plans:
        return ids
    accepted = []
    seen = set()

    def add(i):
        if i not in seen:
            seen.add(i)
            accepted.append(i)

    candidates = [table[i] for i in ids if i in table]
    for plan in plans:
        if not plan.get("conditions"):
            for i in ids:
                add(i)
            continue
        for row in execute_plan(plan, candidates):
            add(row["id"])
        for row in execute_plan(plan, list(table.values()))[:200]:
            add(row["id"])
    keep = [i for i in ids if i in seen]
    rest = [i for i in accepted if i not in set(keep)]
    return keep + rest
