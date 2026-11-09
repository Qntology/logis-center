"""
N:N 채점 (v2)
============

  python score_relational.py search <검색 결과.jsonl> <추출 결과.jsonl ...> [--fixed-relay] [--json out.json]
  python score_relational.py extract <추출 결과.jsonl ...>

search
  화면에 나오는 순서를 commerce/src/main.ts 그대로 재현해 채점합니다.
    1) 계획 거르기: 조건 없는 계획은 후보 전부 통과, 조건 있으면 후보 안(최대 500) + 표 전체 구출(최대 200),
       기간 연결(relay.applied)이면 구출을 target_ids 안으로 좁히고 void_only 를 뺍니다.
    2) 백엔드 순서대로 통과한 후보 → 구출 문서(Dexie anyOf 라 id 순) → 연결 문서(Relay v4: 정방향 · 역방향 · 2단).
       2단 연결은 main.ts 가 String(index) 로 찾는데 Dexie 의 연결 칸은 숫자라 실제로는 아무것도 못 찾습니다.
       --fixed-relay 는 이것을 숫자로 고치고, 조건의 대체 필드(plan.alternates)까지 읽는 고친 화면입니다.
    행동 로그(analytic)는 화면이 결과 목록 대신 요약 · 리포트를 보이므로 백엔드 결과 순서로 채점합니다.
  지표: MRR · R@10 · R@all(화면 전체) · Hit@1 · 찾음(화면에 정답이 하나라도) · nDCG@10.
  묶음: 질의 언어 · 데이터 언어 · 질의 언어 × 데이터 언어 · 틀 · 영역.

extract
  커머스 페이지 정답(commerce/input_yamls)과 무역 서류 정답(trade/input_yamls)을 저장 아이템과 대조합니다.
  레코드 찾음 · 필드 일치율 · 연결(goods · order · tracking · event 칸이 정답 문서의 index 를 가리키는지)을 언어별로.
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

HERE = Path(__file__).resolve().parent
LINK_KEYS = ["goods", "order", "tracking", "event"]
LINK_PATHS = ["index", "no", "code", "tracking_number", "goods", "order", "tracking", "stock_keeping_unit", "barcode",
              "doc_number", "reference_invoice", "reference_lc", "reference_booking", "container_number", "seal_number"] + [
    f"rel_{c}" for c in ("bl hbl swb awb ci cinv csi pi pl po sc lc llc co bc bk sr do an sa fcr pod cm fi wr ed id ccc cnm el ic wc "
                         "ca coa pc fc hc cdr ip icf lg tr soa dn cn ti cp be ins dgd").split()]


def read_jsonl(path) -> list:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def records() -> list:
    return yaml.safe_load((HERE / "records.yaml").read_text(encoding="utf-8"))


def weblog_truth() -> list:
    return yaml.safe_load((HERE / "weblog" / "truth.yaml").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# 저장 아이템 → 세계 레코드 키
# ──────────────────────────────────────────────
def cc_to_shop(extractions: list) -> dict:
    """사이트 cc(host 해시) → 쇼핑몰. html 추출 결과의 아이템, 또는 keys 작업이 남긴 (host, cc) 쌍에서."""
    by_host = {r["domain"]: r["shop"] for r in records()}
    out = {}
    for res in extractions:
        for host, cc in res.get("cc") or []:
            if host in by_host:
                out[cc] = by_host[host]
    for res in extractions:
        if res.get("kind") != "html":
            continue
        shop = res["id"].split("-")[0]
        for it in res.get("items") or []:
            if it.get("cc"):
                out[it["cc"]] = shop
    return out


def final_table(extractions: list) -> dict:
    """마지막 dump 줄의 저장 아이템 전체 {id: 행}. dump 가 없으면 작업별 바뀐 아이템을 합칩니다."""
    dumps = [r for r in extractions if r.get("kind") == "dump" and r.get("items")]
    rows = dumps[-1]["items"] if dumps else [it for r in extractions for it in (r.get("items") or [])]
    return {r["id"]: r for r in rows}


TYPE_FAMILY = {"sales": "order", "receiving": "tracking", "shipping": "tracking"}


def raw_tokens(row: dict) -> list:
    d = row.get("data") or {}
    toks = []
    for k in ("id", "no", "code", "tracking_number", "doc_number"):
        v = d.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            toks.append(str(v).strip())
    link = d.get("link") or ""
    if isinstance(link, str) and "?" in link:
        for vals in parse_qs(urlparse(link).query).values():
            toks += vals
    return toks


def build_id_map(table: dict, recs: list, ccmap: dict, events: list) -> dict:
    """저장 아이템 id → 레코드 키 (커머스 '<shop>:<type>:<id>', 무역 '<shop>_<CODE><n>', 행동 로그 이벤트 id).

    번호는 대문자로 맞춰 비교합니다(앱은 link 를 소문자로 남기고 entity 키는 대문자로 정규화합니다). 같은 유형의 레코드를
    먼저 찾고, 주문은 번호가 저장되지 않는 경우가 있어(추출이 운송장 링크를 자기 링크로 잡음) 주문 일시 · 금액으로도 찾습니다.
    자리표시(ledger placeholder)는 실제 문서가 아니라서 대응하지 않습니다.
    """
    by_shop, order_sig = {}, {}
    for r in recs:
        k = r["shop"]
        idx = {}
        for t, rows in (("goods", r["goods"]), ("order", r["orders"]), ("tracking", r["shipments"]), ("review", r["reviews"]),
                        ("coupon", r["coupons"]), ("event", r["events"])):
            for x in rows:
                idx.setdefault(str(x["id"]).upper(), []).append((t, x["key"]))
        by_shop[k] = idx
        order_sig[k] = {(o["when"][:16], round(float(o["amount"]), 2)): o["key"] for o in r["orders"]}
    trade = {d["number"].upper(): d["stem"] for r in recs for d in r["trade"]["docs"]}
    event_ids = {e["id"] for e in events}
    out = {}
    for iid, row in table.items():
        d = row.get("data") or {}
        mode = row.get("mode") or d.get("mode")
        if iid in event_ids:
            out[iid] = iid
            continue
        if d.get("ledger") == "placeholder":
            continue
        if mode == "shipping":
            for tok in raw_tokens(row):
                if tok.upper() in trade:
                    out[iid] = trade[tok.upper()]
                    break
            continue
        shop = ccmap.get(row.get("cc"))
        if not shop:
            continue
        typ = TYPE_FAMILY.get(row.get("type"), row.get("type"))
        idx = by_shop[shop]
        hit = next((key for tok in raw_tokens(row) for (t, key) in idx.get(tok.upper(), []) if t == typ), None)
        if hit is None and typ == "order":
            when = str(d.get("registration_date") or "")[:16].replace(" ", "T")
            amount = _num(d.get("amount"))
            if when and amount is not None:
                hit = order_sig[shop].get((when, round(amount, 2)))
        if hit:
            out[iid] = hit
    return out


# ──────────────────────────────────────────────
# 프론트엔드 재현 (main.ts)
# ──────────────────────────────────────────────
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
    """main.ts matchCondition. alternates 는 읽지 않습니다(main.ts 와 같음)."""
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
        actual = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else _num(re.sub(r"[^\d.\-]", "", str(raw)))
        if actual is None:
            return False
        return {"gt": actual > target, "gte": actual >= target, "lt": actual < target, "lte": actual <= target,
                "neq": actual != target}.get(op, actual == target)
    value = cond.get("value")
    if isinstance(raw, list):
        t0 = "" if value is None else str(value).lower()
        if not t0:
            return True
        hit = any((str(el).lower().find(t0) >= 0) if op == "contains" else str(el).lower() == t0 for el in raw if el is not None)
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


def cond_pass(row: dict, cond: dict, plan: dict, use_alts: bool) -> bool:
    """고친 main.ts: 조건 필드가 안 맞으면 plan.alternates 의 대체 필드로도 봅니다(lib.rs dexie_condition_passes 규칙)."""
    if match_condition(row, cond):
        return True
    if not use_alts:
        return False
    path = cond.get("path", "")
    field = path[5:] if path.startswith("data.") else path
    return any(match_condition(row, {**cond, "path": "data." + a}) for a in (plan.get("alternates") or {}).get(field, []))


def execute_plan(plan: dict, rows: list, limit: int, use_alts: bool = False) -> list:
    conds = plan.get("conditions") or []
    allowed = plan.get("types") or ([plan["type"]] if plan.get("type") else [])
    if allowed:
        rows = [r for r in rows if r.get("type") in allowed]
    if plan.get("mode"):
        rows = [r for r in rows if (r.get("mode") or "commerce") == plan["mode"]]
    if plan.get("cc"):
        rows = [r for r in rows if r.get("cc") == plan["cc"]]
    plain = [c for c in conds if c.get("op") not in ("top", "bottom")]
    rows = [r for r in rows if all(cond_pass(r, c, plan, use_alts) for c in plain)]
    for rc in [c for c in conds if c.get("op") in ("top", "bottom")]:
        ranked = [r for r in rows if _num(read_path(r, rc["path"])) is not None]
        if not ranked:
            continue
        take = max(1, math.ceil(len(ranked) * max(1, min(100, rc.get("percent") or 20)) / 100))
        ranked.sort(key=lambda r: _num(read_path(r, rc["path"])), reverse=rc["op"] == "top")
        rows = [r for r in ranked if _num(read_path(r, rc["path"])) == _num(read_path(ranked[0], rc["path"]))] if rc.get("extreme") \
            else ranked[:take]
    return rows[:limit]


def same_key(a, b) -> bool:
    """IndexedDB 키 비교: 숫자끼리 · 문자열끼리만 같습니다."""
    na = isinstance(a, (int, float)) and not isinstance(a, bool)
    nb = isinstance(b, (int, float)) and not isinstance(b, bool)
    if na and nb:
        return float(a) == float(b)
    return isinstance(a, str) and isinstance(b, str) and a == b


def relay_family(t: str) -> str:
    k = (t or "").strip().lower()
    if k in ("receiving", "shipping", "tracking"):
        return "tracking"
    if k in ("sales", "order"):
        return "order"
    if k in ("coupon", "event"):
        return "event"
    return k


class LinkIndex:
    """findLinked: LINK_PATHS 중 하나라도 값이 같은 행 (id 순)."""

    def __init__(self, table: dict):
        self.by_val = defaultdict(list)
        for iid in sorted(table):
            d = (table[iid].get("data") or {})
            for p in LINK_PATHS:
                v = d.get(p)
                if isinstance(v, bool) or v is None or v == "" or isinstance(v, (list, dict)):
                    continue
                key = ("n", float(v)) if isinstance(v, (int, float)) else ("s", str(v))
                self.by_val[key].append(iid)

    def find(self, value) -> list:
        if value in ("", 0, None):
            return []
        key = ("n", float(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else ("s", str(value))
        seen, out = set(), []
        for iid in self.by_val.get(key, []):
            if iid not in seen:
                seen.add(iid)
                out.append(iid)
        return out


def frontend_view(search: dict, table: dict, links: LinkIndex, fixed_relay: bool = False) -> list:
    results = [r for r in (search.get("results") or []) if r.get("id")]
    plans = search.get("dexie_plans") or []
    ids = [r["id"] for r in results]
    plan_ids = None
    if plans:
        cand_rows = [table[i] for i in ids if i in table]
        accepted, order = set(), []

        def add(i):
            if i not in accepted:
                accepted.add(i)
                order.append(i)

        for plan in plans:
            relay = plan.get("relay") or {}
            locked = relay.get("applied") is True
            targets = relay.get("target_ids") or [] if locked else []
            void = set(relay.get("void_only") or [])
            if not plan.get("conditions"):
                for i in ids:
                    if i not in void:
                        add(i)
                continue
            for r in execute_plan(plan, cand_rows, 500, fixed_relay):
                if r["id"] not in void:
                    add(r["id"])
            pool = [table[i] for i in targets if i in table] if locked else list(table.values())
            for r in (execute_plan(plan, pool, 200, fixed_relay) if (not locked or targets) else []):
                if r["id"] not in void:
                    add(r["id"])
        plan_ids = accepted
    docs = [i for i in ids if plan_ids is None or i in plan_ids]
    if plan_ids:
        seen = set(docs)
        docs += sorted(i for i in plan_ids if i not in seen and i in table)
    # Relay v4
    fams, partners, void = set(), set(), set()
    for p in plans:
        r = p.get("relay") or {}
        if r.get("applied") is not True:
            continue
        fams |= {relay_family(t) for t in r.get("partner_types") or []}
        partners |= set(r.get("partner_ids") or [])
        void |= set(r.get("void_partner_ids") or []) | set(r.get("void_only") or [])

    def admits(row):
        if not fams:
            return True
        if row["id"] in void:
            return False
        if relay_family(row.get("type") or (row.get("data") or {}).get("type") or "") not in fams:
            return True
        return row["id"] in partners

    existing, relay_docs = set(docs), []
    relay_set = set()

    def absorb(iid):
        if iid in existing or iid in relay_set or iid not in table or not admits(table[iid]):
            return False
        relay_set.add(iid)
        relay_docs.append(iid)
        return True

    for iid in list(docs):
        d = (table.get(iid) or {}).get("data") or {}
        idx = _num(d.get("index"))
        if idx:
            for m in links.find(idx):
                if m != iid:
                    absorb(m)
        for key in ("tracking_number", "no", "code", "goods", "order", "tracking", "stock_keeping_unit", "barcode"):
            raw = d.get(key)
            if raw is None or raw == "" or isinstance(raw, (list, dict)):
                continue
            val = _num(raw) if key in ("goods", "order", "tracking") else str(raw)
            if val is None:
                continue
            for m in links.find(val):
                if m == iid or not absorb(m):
                    continue
                b = ((table[m].get("data") or {}).get("index"))
                if not b:
                    continue
                for d2 in links.find(_num(b) if fixed_relay else str(b)):
                    if d2 not in (m, iid):
                        absorb(d2)
    return docs + relay_docs


def metrics(ranked: list, relevant: set, k: int = 10) -> dict:
    rr = next((1.0 / i for i, x in enumerate(ranked, 1) if x in relevant), 0.0)
    top = ranked[:k]
    hit = len(set(top) & relevant)
    allhit = len(set(ranked) & relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(top) if x in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"mrr": rr, "r10": hit / max(1, min(k, len(relevant))), "rall": allhit / max(1, len(relevant)),
            "hit1": 1.0 if ranked[:1] and ranked[0] in relevant else 0.0, "found": 1.0 if allhit else 0.0,
            "ndcg": dcg / ideal if ideal else 0.0, "shown": len(ranked)}


def dedupe_keys(view: list, id_map: dict) -> list:
    out, seen = [], set()
    for i in view:
        k = id_map.get(i, "?" + i)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def score_search(search_path, extract_paths, fixed_relay=False, json_out=None):
    queries = {q["id"]: q for q in yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))}
    extractions = [r for p in extract_paths for r in read_jsonl(p)]
    table = final_table(extractions)
    recs = records()
    events = weblog_truth()
    id_map = build_id_map(table, recs, cc_to_shop(extractions), events)
    links = LinkIndex(table)
    rows = []
    for res in read_jsonl(search_path):
        q = queries.get(res["id"])
        if not q:
            continue
        s = res.get("search") or {}
        if q["domain"] == "analytic":
            view = [r["id"] for r in (s.get("results") or []) if r.get("id")]
        else:
            view = frontend_view(s, table, links, fixed_relay)
        ranked = dedupe_keys(view, id_map)
        m = metrics(ranked, set(q["relevant"]))
        rows.append({**{k: q[k] for k in ("id", "template", "kind", "domain", "shop", "data_lang", "lang")}, **m,
                     "ok": res.get("ok"), "top": ranked[:5]})
    report(rows)
    if json_out:
        Path(json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def report(rows):
    def table_of(group_key, title):
        g = defaultdict(list)
        for r in rows:
            g[r[group_key] if isinstance(group_key, str) else group_key(r)].append(r)
        print(f"\n## {title}")
        print(f"{'':16s} {'n':>4s} {'MRR':>6s} {'R@10':>6s} {'R@all':>6s} {'Hit1':>6s} {'찾음':>6s} {'nDCG':>6s}")
        for k in sorted(g, key=lambda x: -avg(g[x], "mrr")):
            v = g[k]
            print(f"{str(k):16s} {len(v):4d} {avg(v, 'mrr'):6.3f} {avg(v, 'r10'):6.3f} {avg(v, 'rall'):6.3f} {avg(v, 'hit1'):6.3f} "
                  f"{avg(v, 'found'):6.3f} {avg(v, 'ndcg'):6.3f}")

    print(f"질의 {len(rows)}건 | MRR {avg(rows, 'mrr'):.3f} | R@10 {avg(rows, 'r10'):.3f} | R@all {avg(rows, 'rall'):.3f} | "
          f"찾음 {avg(rows, 'found'):.3f}")
    table_of("lang", "질의 언어")
    table_of("data_lang", "데이터 언어")
    table_of("domain", "영역")
    table_of("template", "틀")
    table_of(lambda r: f"{r['domain'][:4]}/{r['lang']}", "영역 × 질의 언어")
    table_of(lambda r: "같은 언어" if r["lang"] == r["data_lang"] else "다른 언어", "질의 언어 = 데이터 언어?")
    for dom in ("commerce", "shipping", "analytic"):
        sub = [r for r in rows if r["domain"] == dom]
        if not sub:
            continue
        dls = sorted({r["data_lang"] for r in sub})
        qls = sorted({r["lang"] for r in sub}, key=lambda l: -avg([r for r in sub if r["lang"] == l], "rall"))
        print()
        print(f"## {dom}: 질의 언어 × 데이터 언어 (R@all / MRR)")
        print(f"{'':8s}" + "".join(f"{d:>14s}" for d in dls))
        for ql in qls:
            cells = []
            for dl in dls:
                v = [r for r in sub if r["lang"] == ql and r["data_lang"] == dl]
                cells.append(f"{avg(v, 'rall'):.2f}/{avg(v, 'mrr'):.2f}" if v else "-")
            print(f"{ql:8s}" + "".join(f"{c:>14s}" for c in cells))
    tm = sorted({r["template"] for r in rows}, key=lambda t: (t[0] != "C", t[0] != "R", int(t[1:])))
    langs = sorted({r["lang"] for r in rows})
    print()
    print("## 틀 × 질의 언어 (R@all)")
    print(f"{'':6s}" + "".join(f"{l:>8s}" for l in langs))
    for t in tm:
        print(f"{t:6s}" + "".join(f"{avg([r for r in rows if r['template'] == t and r['lang'] == l], 'rall'):8.2f}" for l in langs))


# ──────────────────────────────────────────────
# 추출 채점
# ──────────────────────────────────────────────
def norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


STATUS_CODE = {"Delivered": 9, "Shipped": 1, "Returned": 5, "Cancelled": 3, "Paid": 9, "Processing": 1, "Awaiting payment": 1}


def record_status(rec: dict, ptype: str, rid: str):
    rows = {"order": rec["orders"], "tracking": rec["shipments"]}.get(ptype) or []
    return next((x["status"] for x in rows if str(x["id"]) == rid), None)


def expected_keys(paths: list) -> dict:
    """keys 작업 결과 → {(host, 유형, 번호 대문자): index}."""
    out = {}
    for p in paths:
        for r in read_jsonl(p):
            for mode, host, ty, raw, index in r.get("keys") or []:
                out[(host, ty, str(raw).upper())] = index
    return out


def score_extract(extract_paths, key_paths, only=None):
    """표본 페이지 · 서류의 정답과 저장 아이템을 대조합니다 (데이터 언어별).

    · 찾음: 표본 페이지에 나온 레코드가 저장 아이템으로 대응되는지
    · 자기 index: 저장 아이템의 index 가 그 레코드의 정답 번호로 센 index 와 같은지 (틀리면 다른 문서의 연결이 이 문서를 못 찾음)
    · 연결: 운송장 → 주문 · 상품, 주문 → 상품 · 운송장 · 쿠폰, 무역 rel_<서식> 이 정답 대상의 index 를 가리키는지
    · 필드: 페이지 정답(truth_field)의 값이 저장 data 에 있는지 (정규화 문자열 비교)
    """
    extractions = [r for p in extract_paths for r in read_jsonl(p)]
    table = final_table(extractions)
    recs = records()
    keys = expected_keys(key_paths)
    ccmap = cc_to_shop(extractions)
    id_map = build_id_map(table, recs, ccmap, [])
    by_key = defaultdict(list)
    for iid, k in id_map.items():
        by_key[k].append(table[iid])
    done = {r["id"] for r in extractions if r.get("ok")}
    if only:
        done &= set(only)
    stats = defaultdict(lambda: defaultdict(float))
    misses = defaultdict(list)

    def check(st, label, ok):
        st[label + "_n"] += 1
        st[label + "_ok"] += 1 if ok else 0

    for r in recs:
        k, host = r["shop"], r["domain"]
        st = stats[f"commerce/{r['lang']}"]
        exp = lambda ty, raw: keys.get((host, ty, str(raw).upper()))  # noqa: E731
        orders = {o["id"]: o for o in r["orders"]}
        for page in sorted((HERE / "commerce" / "input_yamls").glob(f"{k}-*.yaml")):
            if page.stem not in done:
                continue
            spec = yaml.safe_load(page.read_text(encoding="utf-8"))
            ptype = spec["page_type"]
            items = spec.get("items") or ([spec["fields"]] if spec.get("fields") else [])
            for truth in items:
                rid = str(truth.get("id,link") or "")
                key = f"{k}:{ptype}:{rid}"
                rows = by_key.get(key) or []
                check(st, "found", bool(rows))
                if not rows:
                    misses["found"].append(key)
                    continue
                d = rows[0].get("data") or {}
                if keys:
                    ok = _num(d.get("index")) == _num(exp(ptype, rid))
                    check(st, "self", ok)
                    if not ok:
                        misses["self"].append(key)
                for f, v in truth.items():
                    if f == "id,link" or v is None:
                        continue
                    if f == "status":
                        # 앱은 상태를 코드로 저장합니다(logic::parse_status). 세계의 영어 상태 → 코드로 비교합니다.
                        want = STATUS_CODE.get(record_status(r, ptype, rid))
                        if want is not None:
                            check(st, "field", _num(d.get("status")) == want)
                            continue
                    blob = norm(json.dumps(d, ensure_ascii=False))
                    check(st, "field", norm(v) == norm(d.get(f, "")) or norm(v) in blob)
                if not keys:
                    continue
                links = []
                if ptype == "tracking":
                    sh = next(x for x in r["shipments"] if x["id"] == rid)
                    links = [("order", "order", sh["order"]), ("goods", "goods", sh["goods"][0])]
                elif ptype == "order":
                    o = orders[rid]
                    links = [("goods", "goods", o["goods"][0])]
                    if o["tracking"]:
                        links.append(("tracking", "tracking", o["tracking"][0]))
                    if o["coupon"]:
                        links.append(("event", "coupon", o["coupon"]))
                for field, ty, raw in links:
                    ok = _num(d.get(field)) == _num(exp(ty, raw))
                    check(st, "link", ok)
                    if not ok:
                        misses["link"].append(f"{key}.{field}")
        st2 = stats[f"trade/{r['trade']['lang']}"]
        numbers = {d["number"]: d["code"] for d in r["trade"]["docs"]}
        numbers[r["trade"]["numbers"]["po"]] = "PO"
        numbers[r["trade"]["numbers"]["lc"]] = "LC"
        for doc in r["trade"]["docs"]:
            if f"v2-{doc['stem']}" not in done:
                continue
            rows = by_key.get(doc["stem"]) or []
            check(st2, "found", bool(rows))
            if not rows:
                misses["found"].append(doc["stem"])
                continue
            d = rows[0].get("data") or {}
            if keys:
                check(st2, "self", _num(d.get("index")) == _num(keys.get(("trading.logis.center", doc["code"], doc["number"].upper()))))
            truth = yaml.safe_load((HERE / "trade" / "input_yamls" / f"{doc['stem']}.yaml").read_text(encoding="utf-8"))
            blob = norm(json.dumps(d, ensure_ascii=False))
            for f, v in truth["metadata"].items():
                if v:
                    check(st2, "field", norm(v) in blob or (f == "issue_date" and norm(v) in blob.replace("t00:00:00", "")))
            for ref in doc["refs"]:
                code = numbers.get(ref)
                if not code or not keys:
                    continue
                exp_i = keys.get(("trading.logis.center", code, ref.upper()))
                if exp_i is None and code in ("PO", "LC"):
                    continue
                ok = _num(d.get(f"rel_{code.lower()}")) == _num(exp_i)
                check(st2, "link", ok)
                if not ok:
                    misses["link"].append(f"{doc['stem']}.rel_{code.lower()}")
    print(f"{'':16s} {'찾음':>12s} {'자기index':>12s} {'필드':>14s} {'연결':>12s}")
    for k in sorted(stats):
        s_ = stats[k]
        cell = lambda lab: f"{s_[lab + '_ok'] / max(1, s_[lab + '_n']):.3f}/{int(s_[lab + '_n'])}"  # noqa: E731
        print(f"{k:16s} {cell('found'):>12s} {cell('self'):>12s} {cell('field'):>14s} {cell('link'):>12s}")
    for lab, xs in misses.items():
        print(f"  놓침 {lab} {len(xs)}: {xs[:8]}")
    return stats


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["search", "extract"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fixed-relay", action="store_true")
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--json")
    ap.add_argument("--only", nargs="*", default=[], help="extract: 이 작업 id 만 채점 (실행끼리 같은 페이지로 견줄 때)")
    a = ap.parse_args()
    if a.what == "search":
        score_search(a.paths[0], a.paths[1:], a.fixed_relay, a.json)
    else:
        score_extract(a.paths, a.keys, a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
