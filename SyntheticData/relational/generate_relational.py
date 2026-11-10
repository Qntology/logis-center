"""
관계형 합성 데이터 (v2) 생성기
=============================

  python generate_relational.py            전체 생성 (reportlab 이 있는 파이썬)
  python generate_relational.py --check    다시 생성해 저장된 파일과 같은지만 확인

출력
  commerce/output_html/<page_id>.html   쇼핑몰 4곳 × 11장 (render.py)
  commerce/input_yamls/<page_id>.yaml   페이지 정답 (열의 truth_field)
  trade/output_pdfs/<stem>.pdf          거래 4건 × 7장 (trade.py)
  trade/input_yamls/<stem>.yaml         서류 정답
  weblog/events.jsonl                   행동 로그 원시 행 (weblog.py, Cloudflare D1 행 모양)
  weblog/sites.yaml                     도메인 · 웹 경로 정의
  records.yaml                          세계 레코드와 관계 (질의 정답 · 채점의 바탕)
  manifest.yaml                         입력 목록 (페이지 · 서류 · 로그, 언어 · URL · 경로)
  queries.yaml                          N:N 복합 질의 (12개 언어, queries.py)
"""
import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

import json

import render
import trade
import weblog
import world as W

HERE = Path(__file__).resolve().parent


def dump(path: Path, obj) -> None:
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")


def iso(d):
    return d.isoformat() if d else None


def world_records(sv: W.ShopV2, w: dict, deal: dict) -> dict:
    """세계를 채점용 레코드로. 키는 '<쇼핑몰>:<종류>:<식별자>' 입니다."""
    k = sv.shop.key
    key = lambda t, i: f"{k}:{t}:{i}"  # noqa: E731
    out = {"shop": k, "lang": sv.shop.lang, "name": sv.shop.name, "domain": sv.shop.domain, "company": sv.shop.company,
           "goods": [], "orders": [], "shipments": [], "reviews": [], "coupons": [], "events": [], "members": [], "trade": {}}
    for g in w["goods"]:
        out["goods"].append({"key": key("goods", g["id"]), "id": g["id"], "title": g["title"], "title_en": g["title_en"],
                             "title_ko": g["title_ko"], "concept": g["concept"], "status": g["status"],
                             "price": W.gcp.money_truth(g["sale_price"], sv.shop), "stock": g["stock"],
                             "registered": iso(g["registered"]), "imported": g["imported"]})
    for m in w["members"]:
        out["members"].append({"id": m["id"], "name": m["name"], "phone": m["phone"]})
    for o in w["orders"]:
        out["orders"].append({"key": key("order", o["id"]), "id": o["id"], "when": o["when"].isoformat(timespec="minutes"),
                              "member": o["member"], "buyer": o["buyer"]["name"], "receiver": o["receiver"]["name"],
                              "status": o["status"], "amount": W.gcp.money_truth(o["amount"], sv.shop),
                              "goods": [l["goods"] for l in o["lines"]], "qty": {l["goods"]: l["qty"] for l in o["lines"]},
                              "coupon": o["coupon"], "event": o["event"], "tracking": list(o["tracking"])})
    for s in w["shipments"]:
        out["shipments"].append({"key": key("tracking", s["id"]), "id": s["id"], "order": s["order"], "kind": s["kind"],
                                 "carrier": s["carrier"], "goods": s["goods"], "shipped": s["shipped"].isoformat(timespec="minutes"),
                                 "delivered": s["delivered"].isoformat(timespec="minutes") if s["delivered"] else None,
                                 "status": "Returned" if s["kind"] == "return" else s["status"], "receiver": s["receiver"]["name"]})
    for x in w["reviews"]:
        out["reviews"].append({"key": key("review", x["id"]), "id": x["id"], "order": x["order"], "goods": x["goods"],
                               "author": x["author"], "rating": x["rating"], "text": x["text"], "posted": iso(x["posted"]),
                               "status": x["status"]})
    for c in w["coupons"]:
        out["coupons"].append({"key": key("coupon", c["id"]), "id": c["id"], "code": c["code"], "title": c["title"],
                               "start": iso(c["start"]), "end": iso(c["end"]), "event": c["event"]})
    for e in w["events"]:
        out["events"].append({"key": key("event", e["id"]), "id": e["id"], "title": e["title"], "start": iso(e["start"]),
                              "end": iso(e["end"]), "goods": e["goods"], "coupon": e["coupon"]})
    out["trade"] = {
        "lang": deal["lang"], "exporter": deal["exporter"], "importer": deal["importer"], "currency": deal["currency"],
        "pol": deal["pol"], "pod": deal["pod"], "numbers": deal["numbers"],
        "ships": [{"n": s["n"], "ci": s["ci"], "pl": s["pl"] if s["n"] == 1 else None, "bl": s["bl"], "vessel": s["vessel"],
                   "etd": iso(s["etd"]), "eta": iso(s["eta"]), "container": s["container"],
                   "goods": [l["goods"] for l in s["lines"]], "descriptions": [l["description"] for l in s["lines"]]}
                  for s in deal["ships"]],
        "docs": [{"stem": d["_stem"], "code": d["document_code"], "ship": d["_ship"], "number": trade.doc_number(d),
                  "refs": trade.doc_refs(d)} for d in deal["docs"]],
    }
    return out


def generate(root: Path) -> dict:
    dirs = {name: root / name for name in ("commerce/output_html", "commerce/input_yamls", "trade/output_pdfs", "trade/input_yamls")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    (root / "weblog").mkdir(exist_ok=True)
    manifest, records, rows, truth = [], [], [], []
    for sv in W.SHOPS:
        w = W.build_world(sv)
        deal = trade.build_deal(sv, w)   # 수입 품목 표시(imported)가 세계에 남으므로 레코드보다 먼저
        for info, html in render.pages_for(sv, w):
            pid = info["page_id"]
            (dirs["commerce/output_html"] / f"{pid}.html").write_text(html, encoding="utf-8")
            dump(dirs["commerce/input_yamls"] / f"{pid}.yaml", {k: v for k, v in info.items() if k != "page_id"})
            manifest.append({"id": pid, "kind": "html", "shop": sv.shop.key, "lang": sv.shop.lang, "page_type": info["page_type"],
                             "layout": info["layout"], "url": info["url"], "path": f"commerce/output_html/{pid}.html"})
        trade.write_docs(deal, dirs["trade/input_yamls"], dirs["trade/output_pdfs"])
        for d in deal["docs"]:
            manifest.append({"id": f"v2-{d['_stem']}", "kind": "document", "shop": sv.shop.key, "lang": deal["lang"],
                             "doc_code": d["document_code"], "path": f"trade/output_pdfs/{d['_stem']}.pdf"})
        records.append(world_records(sv, w, deal))
        for s in weblog.build_sessions(sv, w):
            for e in s.events:
                rows.append(e["row"])
                truth.append(e["truth"])
    rows.sort(key=lambda r: r["created_at"])
    manifest.append({"id": "v2-weblog", "kind": "analytic", "path": "weblog/events.jsonl", "events": len(rows)})
    (root / "weblog" / "events.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dump(root / "weblog" / "truth.yaml", truth)
    dump(root / "weblog" / "sites.yaml", {"sites": [weblog.site_def(sv) for sv in W.SHOPS], "routes": weblog.ROUTES})
    dump(root / "manifest.yaml", manifest)
    dump(root / "records.yaml", records)
    return {"pages": sum(1 for m in manifest if m["kind"] == "html"), "docs": sum(1 for m in manifest if m["kind"] == "document"),
            "events": len(rows)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(generate(HERE))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        generate(t)
        diffs = []
        for sub in ("commerce/output_html", "commerce/input_yamls", "trade/output_pdfs", "trade/input_yamls"):
            c = filecmp.dircmp(t / sub, HERE / sub)
            diffs += c.diff_files + c.left_only + c.right_only
        for f in ("manifest.yaml", "records.yaml", "weblog/events.jsonl", "weblog/truth.yaml", "weblog/sites.yaml"):
            if not filecmp.cmp(t / f, HERE / f, shallow=False):
                diffs.append(f)
    print("같음" if not diffs else f"다름 {len(diffs)}건: {diffs[:10]}")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
