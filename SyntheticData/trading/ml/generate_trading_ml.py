"""
무역 서류 다국어판 (분할 ml)
==========================

기존 45건(../input_yamls, 영어)과 같은 모양의 서식 5종(BL · CI · CO · PL · IP)을 11개 언어(한국어 + 10)로 만듭니다.
언어마다 거래 하나(수출자 · 수입자 · 품목 · 번호)를 두고 그 거래의 서류 5장을 씁니다. 11 × 5 = 55건.

  input_yamls/<서식>_<언어>.yaml   정답. 키는 기존 영어 서식과 같은 이름, 값은 그 언어 (truth.py 가 그대로 읽음)
  output_pdfs/<서식>_<언어>.pdf    렌더 결과. 라벨 · 날짜 · 숫자가 그 언어의 꼴입니다
  queries.yaml                     검색 질의 (식별자 · 서식 이름 · 용도 · 당사자 · 품목, 그 언어로)

언어별 문구는 packs/<언어>.py 에 있습니다. reportlab · PyYAML 이 필요합니다.

  python generate_trading_ml.py            # 다시 생성
  python generate_trading_ml.py --check    # 임시 폴더에 다시 만들어 저장된 파일과 같은지만 확인

PDF 는 기존 서식과 같은 reportlab 로 씁니다. 앱은 PDF 글자 층(pdf-extract)을 읽는데, 브라우저 인쇄 PDF 는
낱말 안에 공백이 끼고 아랍어가 표시형 · 역순으로, 일부 한자가 부수 코드로 나와 쓰지 않았습니다.
reportlab 은 아랍어 글자 잇기(shaping)를 하지 않아 화면에서는 글자가 떨어져 보이지만 글자 층은 논리 순서입니다.
"""
import argparse
import filecmp
import random
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import yaml
from reportlab import rl_config
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate
from xml.sax.saxutils import escape

from packs import PACKS

ROOT = Path(__file__).resolve().parent
rl_config.invariant = 1  # 생성 시각 · 문서 id 를 고정해 같은 입력이면 같은 PDF

# (코드, 영어 서식 이름, 파일 이름 앞부분)
DOC_TYPES = [
    ("CI", "Commercial Invoice", "Commercial_Invoice"),
    ("PL", "Packing List", "Packing_List"),
    ("BL", "Bill of Lading", "Bill_of_Lading"),
    ("CO", "Certificate of Origin", "Certificate_of_Origin"),
    ("IP", "Cargo Insurance Policy", "Cargo_Insurance_Policy"),
]

# 글꼴: 언어 팩의 font 이름 → (보통, 굵게). TTC 는 (경로, 하위 글꼴 번호)
FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {
    "latin": (("arial.ttf", 0), ("arialbd.ttf", 0)),
    "ko": (("malgun.ttf", 0), ("malgunbd.ttf", 0)),
    "ja": (("YuGothM.ttc", 0), ("YuGothB.ttc", 0)),
    "zh": (("msyh.ttc", 0), ("msyhbd.ttc", 0)),
}

# 숫자 칸의 소수 자릿수 (통화가 JPY 면 금액은 0)
DECIMALS = {
    "quantity": 0, "total_quantity": 0, "packages": 0, "total_packages": 0,
    "unit_price": 2, "total_amount": 2, "total_items_amount": 2, "freight_charge": 2, "insurance": 2,
    "grand_total_amount": 2, "sum_insured": 2, "premium_amount": 2,
    "gross_weight_kg": 1, "net_weight_kg": 1, "total_net_weight_kg": 1, "total_gross_weight_kg": 1,
    "measurement_cbm": 2, "total_measurement_cbm": 2,
}
MONEY_KEYS = {"unit_price", "total_amount", "total_items_amount", "freight_charge", "insurance", "grand_total_amount",
              "sum_insured", "premium_amount"}
DATE_KEYS = {"issue_date", "effective_date"}
SKIP_KEYS = {"language", "document_title"}  # 정답에만 두고 PDF 에는 쓰지 않는 키


# ──────────────────────────────────────────────
# 1. 표기
# ──────────────────────────────────────────────
def fmt_number(v: float, style: str, decimals: int) -> str:
    s = f"{v:,.{decimals}f}"
    if style == "dot_comma":       # 1.234,50 (de es it pt nl)
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    if style == "space_comma":     # 1 234,50 (fr cs)
        return s.replace(",", " ").replace(".", ",")
    return s                       # 1,234.50


def fmt_date(iso: str, style: str) -> str:
    y, m, d = map(int, iso.split("-"))
    return {
        "dmy_dot": f"{d:02d}.{m:02d}.{y}", "dmy_slash": f"{d:02d}/{m:02d}/{y}", "dmy_dash": f"{d:02d}-{m:02d}-{y}",
        "cs_dot": f"{d}. {m}. {y}", "ymd_slash": f"{y}/{m:02d}/{d:02d}", "cjk_long": f"{y}年{m}月{d}日",
        "ko_dot": f"{y}.{m:02d}.{d:02d}",
    }[style]


# ──────────────────────────────────────────────
# 2. 거래 → 서류 5장 (정답 YAML 모양)
# ──────────────────────────────────────────────
def build_docs(pk: dict) -> list:
    """언어 팩 하나의 거래로 서류 5장을 만듭니다. 번호는 언어 이름에서 뽑은 난수라 언어를 더해도 다른 언어가 그대로입니다."""
    v, cc = pk["values"], pk["code"]
    r = random.Random(f"trading-ml-{pk['lang']}")
    base = date(2026, 9, 1) + timedelta(days=r.randint(0, 40))
    ym = f"{base.year % 100:02d}{base.month:02d}"
    day = lambda n: (base + timedelta(days=n)).isoformat()  # noqa: E731
    num = {
        "po": f"PO-{cc}-{r.randint(10000, 99999)}", "lc": f"LC-{cc}{r.randint(1000000, 9999999)}",
        "ci": f"CI-{cc}-{ym}-{r.randint(100, 999)}", "pl": f"PL-{cc}-{ym}-{r.randint(100, 999)}",
        "bl": f"BL-{cc}{r.randint(10000000, 99999999)}", "co": f"CO-{base.year}-{cc}-{r.randint(1000, 9999)}",
        "ip": f"IP-{cc}-{r.randint(100000, 999999)}",
    }
    container = f"{v['container_owner']}{r.randint(1000000, 9999999)}"
    seal = f"SL{r.randint(100000, 999999)}"
    jpy = v["currency"] == "JPY"
    money = (lambda x: float(round(x))) if jpy else (lambda x: round(x, 2))  # noqa: E731

    goods = v["goods"]
    lines = [{
        "item_code": g["code"], "description": g["desc"], "quantity": g["qty"], "unit": v["unit"],
        "unit_price": float(g["price"]), "total_amount": money(g["qty"] * g["price"]), "hs_code": g["hs"],
    } for g in goods]
    items_total = money(sum(l["total_amount"] for l in lines))
    freight = money(items_total * v["freight_rate"])
    insurance = money(items_total * v["insurance_rate"])
    grand = money(items_total + freight + insurance)
    gross = round(sum(g["gross"] for g in goods), 1)
    net = round(sum(g["net"] for g in goods), 1)
    cbm = round(sum(g["cbm"] for g in goods), 2)
    packages = sum(g["packages"] for g in goods)
    packages_text = v["packages_text"].format(n=packages)
    exp, imp = v["exporter"], v["importer"]
    route = {"vessel_flight": v["vessel"], "port_of_loading": v["pol"], "port_of_discharge": v["pod"]}

    def head(code: str) -> dict:
        title = next(t for c, t, _ in DOC_TYPES if c == code)
        return {"document_type": title, "document_code": code, "language": pk["lang"], "document_title": pk["titles"][code]}

    ci = {**head("CI"),
          "metadata": {"invoice_number": num["ci"], "issue_date": day(0), "reference_po_number": num["po"],
                       "related_bl_number": num["bl"], "related_lc_number": num["lc"], "currency": v["currency"]},
          "parties": {"shipper_exporter": {k: exp[k] for k in ("name", "address", "contact_number", "tax_id")},
                      "consignee": {k: imp[k] for k in ("name", "address", "contact_number", "tax_id")}},
          "shipping_details": {"port_of_loading": v["pol"], "port_of_discharge": v["pod"], "vessel_flight": v["vessel"],
                               "incoterms": v["incoterms"]},
          "line_items": lines,
          "summary": {"total_quantity": sum(g["qty"] for g in goods), "total_items_amount": items_total,
                      "freight_charge": freight, "insurance": insurance, "grand_total_amount": grand}}
    pl = {**head("PL"),
          "metadata": {"packing_list_number": num["pl"], "issue_date": day(0), "related_invoice_number": num["ci"],
                       "related_po_number": num["po"], "related_bl_number": num["bl"]},
          "parties": {"shipper_exporter": {"name": exp["name"], "address": exp["address"]},
                      "consignee": {"name": imp["name"], "address": imp["address"]}},
          "shipping_details": dict(route),
          "packing_details": [{
              "item_code": g["code"], "description": g["desc"], "quantity": g["qty"], "unit": v["unit"],
              "packages": g["packages"], "package_type": v["package_type"], "net_weight_kg": float(g["net"]),
              "gross_weight_kg": float(g["gross"]), "measurement_cbm": float(g["cbm"]),
          } for g in goods],
          "summary": {"total_packages": packages, "total_net_weight_kg": net, "total_gross_weight_kg": gross,
                      "total_measurement_cbm": cbm}}
    bl = {**head("BL"),
          "metadata": {"bl_number": num["bl"], "issue_date": day(2), "related_invoice_number": num["ci"],
                       "related_po_number": num["po"], "related_lc_number": num["lc"], "freight_term": v["freight_term"]},
          "parties": {"shipper": {"name": exp["name"], "address": exp["address"]},
                      "consignee": {"name": imp["name"], "address": imp["address"]},
                      "notify_party": {"name": imp["name"], "address": imp["address"]},
                      "carrier": {"name": v["carrier"]}},
          "shipping_details": {**route, "place_of_receipt": v["receipt"], "place_of_delivery": v["delivery"]},
          "cargo_details": {"container_number": container, "seal_number": seal, "marks_and_numbers": v["marks"],
                            "number_of_packages": packages_text, "description_of_goods": v["goods_all"],
                            "gross_weight_kg": gross, "measurement_cbm": cbm}}
    co = {**head("CO"),
          "metadata": {"certificate_number": num["co"], "issue_date": day(1), "related_invoice_number": num["ci"],
                       "related_bl_number": num["bl"], "related_lc_number": num["lc"]},
          "parties": {"exporter": {"name": exp["name"], "address": exp["address"]},
                      "importer_consignee": {"name": imp["name"], "address": imp["address"]},
                      "issuing_authority": dict(v["chamber"])},
          "transport_details": dict(route),
          "cargo_details": [{
              "marks_and_numbers": v["marks"],
              "description_of_goods": g["desc"] if g["code"] in g["desc"] else f"{g['desc']} ({g['code']})",
              "hs_code": g["hs"], "quantity": f"{fmt_number(g['qty'], pk['number'], 0)} {v['unit']}", "gross_weight_kg": float(g["gross"]),
              "origin_criterion": v["origin_criterion"],
          } for g in goods],
          "declaration": {"statement": v["statement"]}}
    sum_insured = money(grand * 1.1)
    # 보험은 CIF · CIP 면 수출자가, 나머지(FOB · FCA · CFR)면 수입자가 듭니다.
    insured = exp if v["incoterms"].split()[0] in ("CIF", "CIP") else imp
    ip = {**head("IP"),
          "metadata": {"policy_number": num["ip"], "issue_date": day(1), "effective_date": day(2),
                       "related_invoice_number": num["ci"], "related_bl_number": num["bl"], "related_lc_number": num["lc"],
                       "related_po_number": num["po"]},
          "parties": {"insurer": dict(v["insurer"]), "insured": {"name": insured["name"], "address": insured["address"]}},
          "coverage_details": {"sum_insured": sum_insured, "currency": v["currency"], "valuation_basis": v["valuation_basis"],
                               "coverage_conditions": v["coverage_conditions"], "premium_amount": money(sum_insured * 0.0025),
                               "claims_payable_at": v["claims_payable_at"]},
          "transport_details": dict(route),
          "cargo_details": [{"mark_and_numbers": v["marks"], "description_of_goods": v["goods_all"],
                             "number_of_packages": packages_text, "gross_weight_kg": gross}]}
    return [ci, pl, bl, co, ip]


def stem_of(doc: dict) -> str:
    prefix = next(p for c, _, p in DOC_TYPES if c == doc["document_code"])
    return f"{prefix}_{doc['language'].replace('-', '_')}"


# ──────────────────────────────────────────────
# 3. PDF (기존 batch_yaml_to_pdf.py 와 같은 '라벨: 값' 줄, 라벨은 그 언어)
# ──────────────────────────────────────────────
_registered = set()


def font_for(pk: dict) -> str:
    name = pk["font"]
    if name not in _registered:
        (reg, reg_i), (bold, bold_i) = FONTS[name]
        pdfmetrics.registerFont(TTFont(f"ml-{name}", str(FONT_DIR / reg), subfontIndex=reg_i))
        pdfmetrics.registerFont(TTFont(f"ml-{name}-b", str(FONT_DIR / bold), subfontIndex=bold_i))
        for b, i, face in ((0, 0, f"ml-{name}"), (1, 0, f"ml-{name}-b"), (0, 1, f"ml-{name}"), (1, 1, f"ml-{name}-b")):
            addMapping(f"ml-{name}", b, i, face)
        _registered.add(name)
    return f"ml-{name}"


def display(key: str, value, pk: dict, currency: str) -> str:
    if key in DATE_KEYS and isinstance(value, str):
        return fmt_date(value, pk["date"])
    if key in DECIMALS and isinstance(value, (int, float)) and not isinstance(value, bool):
        decimals = 0 if (key in MONEY_KEYS and currency == "JPY") else DECIMALS[key]
        return fmt_number(value, pk["number"], decimals)
    if key == "document_type":  # 정답은 영어 서식 이름, PDF 는 그 언어 이름
        return pk["titles"][next(c for c, t, _ in DOC_TYPES if t == value)]
    return str(value)


def story_lines(node, pk: dict, currency: str, level: int = 0) -> list:
    labels, out = pk["labels"], []
    indent = "&nbsp;" * (level * 4)
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SKIP_KEYS:
                continue
            label = escape(labels[key])
            if isinstance(value, (dict, list)):
                out.append(f"{indent}<b>{label}</b>:")
                out += story_lines(value, pk, currency, level + 1)
            else:
                out.append(f"{indent}<b>{label}</b>: {escape(display(key, value, pk, currency))}")
    elif isinstance(node, list):
        for i, item in enumerate(node, 1):
            out.append(f"{indent}<b>[ {escape(labels['item'])} {i} ]</b>")
            out += story_lines(item, pk, currency, level + 1)
    return out


def write_pdf(doc: dict, pk: dict, path: Path) -> None:
    font = font_for(pk)
    currency = (doc.get("metadata") or {}).get("currency") or (doc.get("coverage_details") or {}).get("currency") or ""
    # 서식 이름을 첫 줄 제목으로 (document_type 줄은 그 언어 이름으로 다시 나옵니다)
    title = ParagraphStyle("t", fontName=font, fontSize=14, leading=20, spaceAfter=10)
    body = ParagraphStyle("b", fontName=font, fontSize=10, leading=16, spaceAfter=6)
    if pk.get("rtl"):
        title.alignment = body.alignment = 2  # 오른쪽 맞춤 (글자 층 순서는 그대로)
    story = [Paragraph(escape(pk["titles"][doc["document_code"]]), title)]
    story += [Paragraph(line, body) for line in story_lines(doc, pk, currency)]
    out = SimpleDocTemplate(str(path), pagesize=letter, title=pk["titles"][doc["document_code"]], author="", subject="",
                            creator="", producer="")
    out.build(story)


# ──────────────────────────────────────────────
# 4. 질의
# ──────────────────────────────────────────────
def queries_for(pk: dict, docs: list, all_docs: list) -> list:
    """서식 이름 · 용도 질의의 정답은 모든 언어의 같은 서식입니다(질의가 언어를 고르지 않으므로)."""
    lang, q = pk["lang"], pk["queries"]
    same_code = lambda code: [stem_of(d) for d in all_docs if d["document_code"] == code]  # noqa: E731
    out = []
    for d in docs:
        number = next(iter(d["metadata"].values()))
        common = {"split": "ml", "domain": "shipping", "lang": lang}
        out.append({**common, "kind": "identifier", "query": q["identifier"].format(number=number), "relevant": [stem_of(d)]})
        out.append({**common, "kind": "doc_title", "query": pk["titles"][d["document_code"]],
                    "relevant": same_code(d["document_code"])})
        out.append({**common, "kind": "doc_purpose", "query": pk["purposes"][d["document_code"]],
                    "relevant": same_code(d["document_code"])})
    stems = [stem_of(d) for d in docs]
    out.append({"split": "ml", "domain": "shipping", "lang": lang, "kind": "party",
                "query": q["party"].format(name=pk["values"]["exporter"]["name"]), "relevant": stems})
    out.append({"split": "ml", "domain": "shipping", "lang": lang, "kind": "goods",
                "query": q["goods"].format(goods=pk["values"]["goods"][0]["desc"]), "relevant": stems})
    return out


# ──────────────────────────────────────────────
# 5. 실행
# ──────────────────────────────────────────────
def generate(out_root: Path) -> dict:
    yaml_dir, pdf_dir = out_root / "input_yamls", out_root / "output_pdfs"
    for d in (yaml_dir, pdf_dir):  # 폴더는 두고 안의 파일만 지웁니다 (Windows 에서 열린 폴더는 지울 수 없음)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    per_lang = {lang: build_docs(pk) for lang, pk in PACKS.items()}
    all_docs = [d for docs in per_lang.values() for d in docs]
    stems = [stem_of(d) for d in all_docs]
    assert len(set(stems)) == len(stems)
    numbers = [next(iter(d["metadata"].values())) for d in all_docs]
    assert len(set(numbers)) == len(numbers), "문서번호가 겹칩니다"
    queries = []
    for lang, docs in per_lang.items():
        pk = PACKS[lang]
        for d in docs:
            stem = stem_of(d)
            (yaml_dir / f"{stem}.yaml").write_text(
                yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
            write_pdf(d, pk, pdf_dir / f"{stem}.pdf")
        queries += queries_for(pk, docs, all_docs)
    (out_root / "queries.yaml").write_text(
        yaml.safe_dump(queries, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    return {"docs": len(all_docs), "queries": len(queries)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="임시 폴더에 다시 생성해 저장된 파일과 같은지 확인합니다")
    a = ap.parse_args()
    if not a.check:
        print(generate(ROOT))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        generate(tmp_root)
        diffs = []
        for sub in ("input_yamls", "output_pdfs"):
            cmp = filecmp.dircmp(tmp_root / sub, ROOT / sub)
            diffs += cmp.diff_files + cmp.left_only + cmp.right_only
        if not filecmp.cmp(tmp_root / "queries.yaml", ROOT / "queries.yaml", shallow=False):
            diffs.append("queries.yaml")
    print("같음" if not diffs else f"다름 {len(diffs)}건: {diffs[:10]}")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
