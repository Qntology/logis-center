"""
평가 계획 (v2)
=============

  python make_plans.py <출력 폴더> [--langs ko,en,...] [--templates C1,R2,...]

  plan_store.jsonl    검색 저장소: keys → 커머스 · 무역 seed2(연결 번호 포함) → 행동 로그 analytic → dump
  plan_search.jsonl   queries.yaml 의 질의 (앱 언어 korean, 커머스는 그 쇼핑몰 사이트로 범위, 무역 · 행동 로그는 범위 없음)

seed2 레코드는 추출 경로가 남기는 모양을 따릅니다: 상태는 앱의 상태 이름(complete · progress · ...), 날짜는 ISO,
연결 칸은 대상의 (유형, 번호)로 주고 하네스가 앱의 entity_key_index 로 번호를 셉니다.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TRADE_HOST = "trading.logis.center"

ORDER_STATUS = {"Delivered": "complete", "Shipped": "progress", "Returned": "return", "Cancelled": "cancel",
                "Paid": "progress", "Processing": "progress", "Awaiting payment": "progress"}
GOODS_STATUS = {"Active": "show", "Sold out": "stop"}
REVIEW_STATUS = {"Approved": "show", "Pending": "draft", "Hidden": "hide"}


def window_status(start: str, end: str, today: str = "2026-10-05") -> str:
    return "draft" if today < start else ("expire" if today > end else "progress")


def commerce_seeds(rec: dict) -> list:
    k, host, lang = rec["shop"], rec["domain"], rec["lang"]
    goods = {g["id"]: g for g in rec["goods"]}
    coupons = {c["id"]: c for c in rec["coupons"]}
    out = []

    def seed(ty, raw, url, data, links=None, arrays=None, detail=False):
        out.append({"id": f"seed-{k}-{ty}-{raw}", "kind": "seed2", "mode": "commerce", "host": host, "type": ty,
                    "self": [ty, raw], "lang": lang, "detail": detail, "url": url, "data": data,
                    "links": links or {}, "arrays": arrays or {}})

    for g in rec["goods"]:
        seed("goods", g["id"], f"https://{host}/admin/goods/edit?it_id={g['id']}",
             {"title": g["title"], "status": GOODS_STATUS[g["status"]], "sale_price": g["price"], "quantity": g["stock"],
              "registration_date": g["registered"], "link": f"/admin/goods/edit?it_id={g['id'].lower()}"})
    for o in rec["orders"]:
        n = len(o["goods"])
        main = goods[o["goods"][0]]["title"]
        label = main if n == 1 else f"{main} (+{n - 1})"
        links = {"goods": ["goods", o["goods"][0]]}
        data = {"registration_date": o["when"], "status": ORDER_STATUS[o["status"]], "goods_title": label,
                "sender_name": o["buyer"], "recipient_name": o["receiver"], "quantity": sum(o["qty"].values()),
                "amount": o["amount"], "link": f"/admin/order/view?od_id={o['id']}"}
        if o["tracking"]:
            links["tracking"] = ["tracking", o["tracking"][0]]
            data["tracking_number"] = o["tracking"][0]
        if o["coupon"]:
            links["event"] = ["coupon", o["coupon"]]
            data["event_title"] = coupons[o["coupon"]]["title"]
        seed("order", o["id"], f"https://{host}/admin/order/view?od_id={o['id']}", data, links)
    for s in rec["shipments"]:
        n = len(s["goods"])
        main = goods[s["goods"][0]]["title"]
        seed("tracking", s["id"], f"https://{host}/admin/shipping/view?invoice={s['id']}",
             {"tracking_number": s["id"], "carrier": s["carrier"], "status": ORDER_STATUS[s["status"]],
              "title": main if n == 1 else f"{main} (+{n - 1})", "shipping_date": s["shipped"],
              "recipient_name": s["receiver"], "link": f"/admin/shipping/view?invoice={s['id']}",
              **({"delivered_at": s["delivered"]} if s["delivered"] else {})},
             {"order": ["order", s["order"]], "goods": ["goods", s["goods"][0]]})
    for x in rec["reviews"]:
        seed("review", x["id"], f"https://{host}/admin/review/view?rv_id={x['id']}",
             {"title": x.get("text") or "", "name": x["author"], "registration_date": x["posted"],
              "status": REVIEW_STATUS[x["status"]], "rating": x["rating"], "goods_title": goods[x["goods"]]["title"],
              "link": f"/admin/review/view?rv_id={x['id']}"},
             {"goods": ["goods", x["goods"]]})
    for c in rec["coupons"]:
        links = {"event": ["event", c["event"]]} if c["event"] else {}
        seed("coupon", c["id"], f"https://{host}/admin/coupon/view?cp_id={c['id']}",
             {"code": c["code"], "title": c["title"], "started_at": c["start"], "expired_at": c["end"],
              "status": window_status(c["start"], c["end"]), "link": f"/admin/coupon/view?cp_id={c['id']}"}, links)
    for e in rec["events"]:
        seed("event", e["id"], f"https://{host}/admin/event/view?ev_id={e['id']}",
             {"title": e["title"], "started_at": e["start"], "expired_at": e["end"],
              "status": window_status(e["start"], e["end"]), "link": f"/admin/event/view?ev_id={e['id']}"},
             arrays={"goods": [["goods", gid, goods[gid]["title"]] for gid in e["goods"]]}, detail=True)
    return out


def trade_seeds(rec: dict) -> list:
    """무역 서류 정답 YAML → 앱의 무역 스키마 필드 (doc_type · doc_number · reference_* · rel_<서식>)."""
    out = []
    t = rec["trade"]
    numbers = {d["number"]: d["code"] for d in t["docs"]}
    numbers[t["numbers"]["po"]] = "PO"
    numbers[t["numbers"]["lc"]] = "LC"
    for d in t["docs"]:
        doc = yaml.safe_load((HERE / "trade" / "input_yamls" / f"{d['stem']}.yaml").read_text(encoding="utf-8"))
        m = doc["metadata"]
        parties = doc.get("parties") or {}
        sender = (parties.get("shipper_exporter") or parties.get("shipper") or parties.get("exporter") or parties.get("insurer") or {}).get("name", "")
        recipient = (parties.get("consignee") or parties.get("importer_consignee") or parties.get("insured") or {}).get("name", "")
        ship = doc.get("shipping_details") or doc.get("transport_details") or {}
        data = {"doc_type": d["code"], "doc_number": d["number"], "no": d["number"], "issue_date": m.get("issue_date"),
                "sender_name": sender, "recipient_name": recipient, "pol": ship.get("port_of_loading", ""),
                "pod": ship.get("port_of_discharge", ""), "vessel": ship.get("vessel_flight", ""),
                "incoterms": ship.get("incoterms", ""), "currency": m.get("currency") or (doc.get("coverage_details") or {}).get("currency", "")}
        refs = {"reference_invoice": m.get("related_invoice_number"), "reference_po": m.get("reference_po_number") or m.get("related_po_number"),
                "reference_lc": m.get("related_lc_number"), "reference_bl": m.get("related_bl_number")}
        data.update({k: v for k, v in refs.items() if v})
        cargo = doc.get("cargo_details")
        if isinstance(cargo, dict):
            data["container_number"] = cargo.get("container_number", "")
            data["description"] = cargo.get("description_of_goods", "")
        lines = doc.get("line_items") or doc.get("packing_details") or (cargo if isinstance(cargo, list) else [])
        if lines:
            data["line_items"] = [{k: v for k, v in l.items() if k in ("item_code", "description", "description_of_goods", "quantity",
                                                                      "unit_price", "total_amount", "hs_code")} for l in lines]
            data["description"] = data.get("description") or " / ".join(str(l.get("description") or l.get("description_of_goods") or "") for l in lines)
        links = {f"rel_{numbers[v].lower()}": [numbers[v], v] for v in refs.values() if v and v in numbers}
        hub = next((v for key in ("reference_po", "reference_invoice", "reference_bl", "reference_lc") for v in [refs.get(key)] if v), "")
        out.append({"id": f"seed-{d['stem']}", "kind": "seed2", "mode": "shipping", "host": TRADE_HOST, "type": d["code"],
                    "self": [d["code"], d["number"]], "lang": t["lang"], "detail": True, "url": f"local://{d['stem']}.pdf",
                    "hub": hub, "data": data, "links": links})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--langs", default="")
    ap.add_argument("--templates", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = yaml.safe_load((HERE / "records.yaml").read_text(encoding="utf-8"))
    keys = []
    for r in recs:
        for ty, rows in (("goods", r["goods"]), ("order", r["orders"]), ("tracking", r["shipments"]), ("review", r["reviews"]),
                         ("coupon", r["coupons"]), ("event", r["events"])):
            keys += [["commerce", r["domain"], ty, str(x["id"])] for x in rows]
        keys += [["shipping", TRADE_HOST, d["code"], d["number"]] for d in r["trade"]["docs"]]
    store = [{"id": "v2-keys", "kind": "keys", "hosts": [r["domain"] for r in recs] + [TRADE_HOST], "items": keys}]
    for r in recs:
        store += commerce_seeds(r)
    for r in recs:
        store += trade_seeds(r)
    store.append({"id": "v2-weblog", "kind": "analytic", "events": str((HERE / "weblog" / "events.jsonl").resolve()), "batch": 20})
    store.append({"id": "v2-dump-store", "kind": "dump"})
    (out / "plan_store.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in store), encoding="utf-8")

    queries = yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))
    langs = set(filter(None, a.langs.split(",")))
    tmpls = set(filter(None, a.templates.split(",")))
    plan = [{"id": q["id"], "kind": "search", "query": q["query"], "mode": q["domain"], "language": "korean", "host": q["host"]}
            for q in queries if (not langs or q["lang"] in langs) and (not tmpls or q["template"] in tmpls)]
    (out / "plan_search.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plan), encoding="utf-8")
    print(f"store {len(store)} lines (seed2 {sum(1 for x in store if x['kind'] == 'seed2')}) | search {len(plan)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
