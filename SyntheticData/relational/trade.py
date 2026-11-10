"""
무역 거래 (v2)
=============

쇼핑몰마다 수입 거래 하나. 주문서(PO) 하나를 두 번에 나눠 싣고(분할 선적), 신용장(LC) 하나로 결제합니다.

  1차 선적: 상업송장 · 포장명세서 · 선하증권 · 원산지증명서 · 적하보험증권 (품목 2개)
  2차 선적: 상업송장 · 선하증권 (품목 1개)

서류는 서로 번호로 가리킵니다(CI → PO · BL · LC, PL → CI · PO · BL, BL → CI · PO · LC, CO → CI · BL · LC, IP → CI · BL · LC · PO).
수출자는 쇼핑몰과 다른 언어로 서류를 씁니다(소래마켓 ← 독일어, Copperleaf ← 일본어, こもれび ← 한국어, Kranichhof ← 영어).
품목 번호(item_code)는 쇼핑몰의 상품 번호라서 무역 서류와 커머스 상품이 같은 식별자로 이어집니다.

PDF 는 ../trading/ml 의 reportlab 렌더러를 그대로 씁니다(라벨은 그 언어, 생성 시각 고정).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trading" / "ml"))

import generate_trading_ml as gtm  # noqa: E402  (PDF 렌더러 · 서류 언어 팩)

import world as W  # noqa: E402

EN_PACK = {
    "lang": "en", "code": "EN", "font": "latin", "date": "en_long", "number": "comma_dot",
    "titles": {"CI": "Commercial Invoice", "PL": "Packing List", "BL": "Bill of Lading", "CO": "Certificate of Origin",
               "IP": "Cargo Insurance Policy"},
    "labels": {
        "document_type": "Document type", "document_code": "Document code", "item": "Item",
        "metadata": "Document details", "parties": "Parties", "shipping_details": "Shipping details",
        "transport_details": "Transport details", "cargo_details": "Cargo details", "line_items": "Invoice lines",
        "packing_details": "Packing details", "summary": "Totals", "declaration": "Declaration",
        "coverage_details": "Coverage",
        "bl_number": "B/L No.", "invoice_number": "Invoice No.", "certificate_number": "Certificate No.",
        "packing_list_number": "Packing List No.", "policy_number": "Policy No.", "issue_date": "Date of issue",
        "effective_date": "Cover commences", "related_invoice_number": "Related invoice", "related_po_number": "Purchase order",
        "reference_po_number": "Buyer's PO No.", "related_bl_number": "Related B/L", "related_lc_number": "L/C No.",
        "freight_term": "Freight", "currency": "Currency",
        "shipper": "Shipper", "consignee": "Consignee", "notify_party": "Notify party", "carrier": "Carrier",
        "shipper_exporter": "Shipper / Exporter", "exporter": "Exporter", "importer_consignee": "Importer / Consignee",
        "issuing_authority": "Issuing body", "insurer": "Insurer", "insured": "Assured",
        "name": "Name", "address": "Address", "contact_number": "Telephone", "tax_id": "VAT No.",
        "vessel_flight": "Vessel / Voyage", "port_of_loading": "Port of loading", "port_of_discharge": "Port of discharge",
        "place_of_receipt": "Place of receipt", "place_of_delivery": "Place of delivery", "incoterms": "Terms of delivery",
        "container_number": "Container No.", "seal_number": "Seal No.", "marks_and_numbers": "Marks and numbers",
        "mark_and_numbers": "Marks and numbers", "number_of_packages": "Number of packages",
        "description_of_goods": "Description of goods", "gross_weight_kg": "Gross weight (kg)",
        "net_weight_kg": "Net weight (kg)", "measurement_cbm": "Measurement (m³)", "item_code": "Item code",
        "description": "Description", "quantity": "Quantity", "unit": "Unit", "unit_price": "Unit price",
        "total_amount": "Amount", "hs_code": "HS code", "packages": "Packages", "package_type": "Package type",
        "origin_criterion": "Origin criterion", "total_quantity": "Total quantity", "total_items_amount": "Goods value",
        "freight_charge": "Freight", "insurance": "Insurance", "grand_total_amount": "Invoice total",
        "total_packages": "Total packages", "total_net_weight_kg": "Total net weight (kg)",
        "total_gross_weight_kg": "Total gross weight (kg)", "total_measurement_cbm": "Total measurement (m³)",
        "statement": "Statement", "sum_insured": "Sum insured", "valuation_basis": "Basis of valuation",
        "coverage_conditions": "Conditions", "premium_amount": "Premium", "claims_payable_at": "Claims payable at",
    },
}

# 쇼핑몰 → 수출자 쪽 거래 값 (서류 언어 그대로). 이름 · 주소 · 번호는 모두 가상입니다.
DEALS = {
    "sorae": {  # 독일 수출자 → 인천 소래마켓
        "exporter": {"name": "Wolkenfeld Haushaltswaren GmbH", "address": "Gutenbergstraße 8, 90461 Nürnberg, Deutschland",
                     "contact_number": "+49 911 555 0126", "tax_id": "DE 314 552 087"},
        "importer": {"name": "Sorae Sanghoe Co., Ltd.", "address": "12 Soraeyeok-ro, Namdong-gu, Incheon, Südkorea",
                     "contact_number": "+82 32 555 0412", "tax_id": "131-86-40912"},
        "carrier": "Elbstrom Reederei GmbH",
        "insurer": {"name": "Frankenland Transportversicherung AG", "address": "Am Plärrer 3, 90443 Nürnberg, Deutschland"},
        "chamber": {"name": "Handelskammer Pegnitzgrund", "address": "Hauptmarkt 25, 90403 Nürnberg, Deutschland"},
        "vessel": ["ELBSTERN V.207E", "ELBSTERN V.211E"], "container_owner": "EBRU",
        "pol": "Hamburg, Deutschland", "pod": "Incheon, Südkorea", "receipt": "Nürnberg, Deutschland",
        "delivery": "Incheon, Südkorea", "incoterms": "FOB Hamburg", "currency": "EUR", "rate": 1450,
        "freight_term": "Fracht zahlbar am Bestimmungsort", "unit": "Stk.", "package_type": "Kartons",
        "packages_text": "{n} Kartons", "marks": "SORAE / INCHEON / MADE IN GERMANY",
        "origin_criterion": "Vollständig in Deutschland hergestellt",
        "statement": "Es wird bescheinigt, dass die oben bezeichneten Waren ihren Ursprung in Deutschland haben.",
        "valuation_basis": "Rechnungswert zuzüglich 10 %",
        "coverage_conditions": "Institute Cargo Clauses (A) – Deckung gegen alle Gefahren",
        "claims_payable_at": "Incheon, Südkorea",
    },
    "copper": {  # 일본 수출자 → 보이시 Copperleaf
        "exporter": {"name": "株式会社ひなた工房", "address": "大阪府東大阪市高井田元町1-3-9",
                     "contact_number": "+81 6-5555-0193", "tax_id": "T4120001098765"},
        "importer": {"name": "Copperleaf Goods Inc.", "address": "1220 Larkspur Ave, Boise, ID 83702, アメリカ合衆国",
                     "contact_number": "+1-208-555-0163", "tax_id": "US 84-2210457"},
        "carrier": "みなと汽船株式会社",
        "insurer": {"name": "あかね海上保険株式会社", "address": "大阪府大阪市中央区北久宝寺町2-4-6"},
        "chamber": {"name": "河内国際商工会", "address": "大阪府東大阪市長堂1-8-37"},
        "vessel": ["MINATO MARU V.118E", "MINATO MARU V.122E"], "container_owner": "MNTU",
        "pol": "大阪港（日本）", "pod": "シアトル港（アメリカ合衆国）", "receipt": "東大阪（日本）",
        "delivery": "ボイシ（アメリカ合衆国）", "incoterms": "CIF Seattle", "currency": "JPY", "rate": 9.2,
        "freight_term": "運賃前払い", "unit": "個", "package_type": "カートン",
        "packages_text": "{n}カートン", "marks": "COPPERLEAF / BOISE / MADE IN JAPAN",
        "origin_criterion": "日本国内で完全に生産",
        "statement": "上記の貨物が日本を原産地とすることを証明する。",
        "valuation_basis": "送り状価額の110%",
        "coverage_conditions": "協会貨物約款（A）全危険担保",
        "claims_payable_at": "シアトル（アメリカ合衆国）",
    },
    "komorebi": {  # 한국 수출자 → 도쿄 こもれび
        "exporter": {"name": "(주)새벽공방", "address": "경기도 안산시 단원구 해봉로 77",
                     "contact_number": "+82 31-555-0148", "tax_id": "134-86-55210"},
        "importer": {"name": "Komorebi Co., Ltd.", "address": "일본 도쿄도 스기나미구 아사가야키타 2-14-7",
                     "contact_number": "+81 3-5555-0148", "tax_id": "T8011101076543"},
        "carrier": "한바다해운(주)",
        "insurer": {"name": "새솔해상보험(주)", "address": "서울특별시 중구 을지로 55"},
        "chamber": {"name": "서해안국제상공회", "address": "경기도 안산시 단원구 광덕대로 182"},
        "vessel": ["HANBADA STAR V.031E", "HANBADA STAR V.035E"], "container_owner": "HBDU",
        "pol": "부산, 대한민국", "pod": "도쿄, 일본", "receipt": "안산, 대한민국",
        "delivery": "도쿄, 일본", "incoterms": "CFR Tokyo", "currency": "USD", "rate": 1300,
        "freight_term": "운임 선불", "unit": "개", "package_type": "상자",
        "packages_text": "{n}상자", "marks": "KOMOREBI / TOKYO / MADE IN KOREA",
        "origin_criterion": "대한민국에서 완전 생산",
        "statement": "위 물품이 대한민국을 원산지로 함을 증명합니다.",
        "valuation_basis": "송장 금액의 110%",
        "coverage_conditions": "협회적하약관(A) 전위험담보",
        "claims_payable_at": "도쿄, 일본",
    },
    "kranich": {  # 영국 수출자 → 라이프치히 Kranichhof
        "exporter": {"name": "Bramblewood Outfitters Ltd.", "address": "14 Wharf Road, Felixstowe IP11 3AB, United Kingdom",
                     "contact_number": "+44 1394 555 0172", "tax_id": "GB 482 7731 05"},
        "importer": {"name": "Kranichhof Handels GmbH", "address": "Lindenallee 21, 04109 Leipzig, Germany",
                     "contact_number": "+49 341 5550187", "tax_id": "DE 297 664 310"},
        "carrier": "Northbridge Line Ltd.",
        "insurer": {"name": "Harrowgate Marine Assurance plc", "address": "22 Mincing Row, London EC3R 7AA, United Kingdom"},
        "chamber": {"name": "East Coast Chamber of Trade", "address": "3 Dock Street, Ipswich IP4 1AA, United Kingdom"},
        "vessel": ["NORTHBRIDGE TRADER V.55E", "NORTHBRIDGE TRADER V.58E"], "container_owner": "NBTU",
        "pol": "Felixstowe, United Kingdom", "pod": "Hamburg, Germany", "receipt": "Felixstowe, United Kingdom",
        "delivery": "Leipzig, Germany", "incoterms": "CIP Leipzig", "currency": "GBP", "rate": 1700,
        "freight_term": "Freight prepaid", "unit": "pcs", "package_type": "cartons",
        "packages_text": "{n} cartons", "marks": "KRANICHHOF / LEIPZIG / MADE IN UK",
        "origin_criterion": "Wholly obtained in the United Kingdom",
        "statement": "It is hereby certified that the goods described above originate in the United Kingdom.",
        "valuation_basis": "Invoice value plus 10%",
        "coverage_conditions": "Institute Cargo Clauses (A) - All Risks",
        "claims_payable_at": "Hamburg, Germany",
    },
}

HS = {"cold_weather": "6110.30", "hot_weather": "6205.20", "desk_work": "8473.30", "cooking": "7323.93", "kids": "9503.00",
      "pet": "4201.00", "outdoor": "6306.22", "skincare": "3304.99", "snack_food": "2008.19", "audio_power": "8518.30"}
CODE = {"de": "DE", "ja": "JP", "ko": "KR", "en": "GB"}


def doc_pack(lang: str) -> dict:
    return EN_PACK if lang == "en" else gtm.PACKS[lang]


def goods_name(g: dict, lang: str) -> str:
    """수출자 서류에 쓰는 품명: 서류 언어의 상품 이름."""
    if lang == "ko":
        return g["title_ko"]
    if lang == "en":
        return g["title_en"]
    return W.gcp.PACKS[lang]["products"][W.gcp.PRODUCT_KEYS[g["title_en"]]][0]


def fmt_date_en(iso: str) -> str:
    y, m, d = map(int, iso.split("-"))
    return f"{d} {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1]} {y}"


_orig_fmt_date = gtm.fmt_date


def _fmt_date(iso: str, style: str) -> str:
    return fmt_date_en(iso) if style == "en_long" else _orig_fmt_date(iso, style)


gtm.fmt_date = _fmt_date


def build_deal(sv: W.ShopV2, world: dict) -> dict:
    """거래 하나 → 서류 7장(정답 YAML 모양) + 거래 정답(번호 · 날짜 · 품목 · 연결)."""
    v = DEALS[sv.shop.key]
    lang = sv.trade_lang
    cc = CODE[lang]
    r = W.rng_for(sv.shop.key, "trade")
    # 수입 품목 3개: 서로 다른 개념의 상품
    pool, picked, seen = list(world["goods"]), [], set()
    r.shuffle(pool)
    for g in pool:
        if g["concept"] not in seen:
            picked.append(g)
            seen.add(g["concept"])
        if len(picked) == 3:
            break
    for g in picked:
        g["imported"] = True
    money = (lambda x: float(round(x))) if v["currency"] == "JPY" else (lambda x: round(x, 2))
    lines = []
    for k, g in enumerate(picked):
        base = W.gcp.PRODUCTS[g["pi"]][6]
        qty = r.choice([120, 180, 240, 300, 360, 480])
        unit_price = money(base * 0.42 / v["rate"])
        net = round(qty * r.uniform(0.25, 1.6), 1)
        lines.append({
            "goods": g["id"], "item_code": g["id"], "description": goods_name(g, lang), "quantity": qty, "unit": v["unit"],
            "unit_price": unit_price, "total_amount": money(qty * unit_price), "hs_code": HS[g["concept"]],
            "net": net, "gross": round(net * 1.12, 1), "packages": max(1, qty // 24), "cbm": round(qty * 0.011, 2),
        })
    num = {
        "po": f"PO-{cc}-{r.randint(10000, 99999)}", "lc": f"LC-{cc}{r.randint(1000000, 9999999)}",
        "co": f"CO-2026-{cc}-{r.randint(1000, 9999)}", "ip": f"IP-{cc}-{r.randint(100000, 999999)}",
    }
    etd = [date(2026, 6, 8) + timedelta(days=r.randint(0, 6)), date(2026, 8, 17) + timedelta(days=r.randint(0, 6))]
    ships = []
    for s, (part, when) in enumerate(((lines[:2], etd[0]), (lines[2:], etd[1]))):
        ym = f"{when.year % 100:02d}{when.month:02d}"
        ships.append({
            "n": s + 1, "lines": part, "etd": when, "eta": when + timedelta(days=r.randint(18, 26)),
            "ci": f"CI-{cc}-{ym}-{r.randint(100, 999)}", "pl": f"PL-{cc}-{ym}-{r.randint(100, 999)}",
            "bl": f"BL-{cc}{r.randint(10000000, 99999999)}", "vessel": v["vessel"][s],
            "container": f"{v['container_owner']}{r.randint(1000000, 9999999)}", "seal": f"SL{r.randint(100000, 999999)}",
        })
    pk = doc_pack(lang)
    exp, imp = v["exporter"], v["importer"]

    def head(code: str) -> dict:
        title = next(t for c, t, _ in gtm.DOC_TYPES if c == code)
        return {"document_type": title, "document_code": code, "language": lang, "document_title": pk["titles"][code]}

    def inv_lines(part):
        return [{k: l[k] for k in ("item_code", "description", "quantity", "unit", "unit_price", "total_amount", "hs_code")}
                for l in part]

    docs = []
    for sh in ships:
        part = sh["lines"]
        items_total = money(sum(l["total_amount"] for l in part))
        freight = money(items_total * (0.04 if v["incoterms"].split()[0] in ("CIF", "CFR", "CIP") else 0))
        insurance = money(items_total * (0.003 if v["incoterms"].split()[0] in ("CIF", "CIP") else 0))
        route = {"vessel_flight": sh["vessel"], "port_of_loading": v["pol"], "port_of_discharge": v["pod"]}
        gross = round(sum(l["gross"] for l in part), 1)
        cbm = round(sum(l["cbm"] for l in part), 2)
        packages = sum(l["packages"] for l in part)
        goods_all = " / ".join(l["description"] for l in part)
        day = lambda n: (sh["etd"] + timedelta(days=n)).isoformat()  # noqa: E731
        docs.append({**head("CI"), "_ship": sh["n"],
                     "metadata": {"invoice_number": sh["ci"], "issue_date": day(-3), "reference_po_number": num["po"],
                                  "related_bl_number": sh["bl"], "related_lc_number": num["lc"], "currency": v["currency"]},
                     "parties": {"shipper_exporter": dict(exp), "consignee": dict(imp)},
                     "shipping_details": {**route, "incoterms": v["incoterms"]},
                     "line_items": inv_lines(part),
                     "summary": {"total_quantity": sum(l["quantity"] for l in part), "total_items_amount": items_total,
                                 "freight_charge": freight, "insurance": insurance,
                                 "grand_total_amount": money(items_total + freight + insurance)}})
        if sh["n"] == 1:
            docs.append({**head("PL"), "_ship": 1,
                         "metadata": {"packing_list_number": sh["pl"], "issue_date": day(-3), "related_invoice_number": sh["ci"],
                                      "related_po_number": num["po"], "related_bl_number": sh["bl"]},
                         "parties": {"shipper_exporter": {"name": exp["name"], "address": exp["address"]},
                                     "consignee": {"name": imp["name"], "address": imp["address"]}},
                         "shipping_details": dict(route),
                         "packing_details": [{"item_code": l["item_code"], "description": l["description"], "quantity": l["quantity"],
                                              "unit": l["unit"], "packages": l["packages"], "package_type": v["package_type"],
                                              "net_weight_kg": float(l["net"]), "gross_weight_kg": float(l["gross"]),
                                              "measurement_cbm": float(l["cbm"])} for l in part],
                         "summary": {"total_packages": packages, "total_net_weight_kg": round(sum(l["net"] for l in part), 1),
                                     "total_gross_weight_kg": gross, "total_measurement_cbm": cbm}})
        docs.append({**head("BL"), "_ship": sh["n"],
                     "metadata": {"bl_number": sh["bl"], "issue_date": day(0), "related_invoice_number": sh["ci"],
                                  "related_po_number": num["po"], "related_lc_number": num["lc"], "freight_term": v["freight_term"]},
                     "parties": {"shipper": {"name": exp["name"], "address": exp["address"]},
                                 "consignee": {"name": imp["name"], "address": imp["address"]},
                                 "notify_party": {"name": imp["name"], "address": imp["address"]},
                                 "carrier": {"name": v["carrier"]}},
                     "shipping_details": {**route, "place_of_receipt": v["receipt"], "place_of_delivery": v["delivery"]},
                     "cargo_details": {"container_number": sh["container"], "seal_number": sh["seal"], "marks_and_numbers": v["marks"],
                                       "number_of_packages": v["packages_text"].format(n=packages), "description_of_goods": goods_all,
                                       "gross_weight_kg": gross, "measurement_cbm": cbm}})
        if sh["n"] == 1:
            docs.append({**head("CO"), "_ship": 1,
                         "metadata": {"certificate_number": num["co"], "issue_date": day(-1), "related_invoice_number": sh["ci"],
                                      "related_bl_number": sh["bl"], "related_lc_number": num["lc"]},
                         "parties": {"exporter": {"name": exp["name"], "address": exp["address"]},
                                     "importer_consignee": {"name": imp["name"], "address": imp["address"]},
                                     "issuing_authority": dict(v["chamber"])},
                         "transport_details": dict(route),
                         "cargo_details": [{"marks_and_numbers": v["marks"], "description_of_goods": f"{l['description']} ({l['item_code']})",
                                            "hs_code": l["hs_code"], "quantity": f"{l['quantity']} {l['unit']}",
                                            "gross_weight_kg": float(l["gross"]), "origin_criterion": v["origin_criterion"]} for l in part],
                         "declaration": {"statement": v["statement"]}})
            sum_insured = money((items_total + freight + insurance) * 1.1)
            insured = exp if v["incoterms"].split()[0] in ("CIF", "CIP") else imp
            docs.append({**head("IP"), "_ship": 1,
                         "metadata": {"policy_number": num["ip"], "issue_date": day(-1), "effective_date": day(0),
                                      "related_invoice_number": sh["ci"], "related_bl_number": sh["bl"],
                                      "related_lc_number": num["lc"], "related_po_number": num["po"]},
                         "parties": {"insurer": dict(v["insurer"]), "insured": {"name": insured["name"], "address": insured["address"]}},
                         "coverage_details": {"sum_insured": sum_insured, "currency": v["currency"], "valuation_basis": v["valuation_basis"],
                                              "coverage_conditions": v["coverage_conditions"], "premium_amount": money(sum_insured * 0.0025),
                                              "claims_payable_at": v["claims_payable_at"]},
                         "transport_details": dict(route),
                         "cargo_details": [{"mark_and_numbers": v["marks"], "description_of_goods": goods_all,
                                            "number_of_packages": v["packages_text"].format(n=packages), "gross_weight_kg": gross}]})
    for d in docs:
        d["_stem"] = f"{sv.shop.key}_{d['document_code']}{d['_ship']}"
    return {"shop": sv.shop.key, "lang": lang, "pack": pk, "numbers": num, "ships": ships, "lines": lines, "docs": docs,
            "exporter": exp["name"], "importer": imp["name"], "pol": v["pol"], "pod": v["pod"], "currency": v["currency"]}


def doc_number(d: dict) -> str:
    return next(iter(d["metadata"].values()))


def doc_refs(d: dict) -> list:
    """이 서류가 가리키는 다른 서류 번호들 (PO · LC 는 서류로 만들지 않으므로 번호만)."""
    m = d["metadata"]
    return [m[k] for k in ("reference_po_number", "related_po_number", "related_invoice_number", "related_bl_number", "related_lc_number")
            if k in m]


def write_docs(deal: dict, yaml_dir: Path, pdf_dir: Path) -> None:
    import yaml
    for d in deal["docs"]:
        truth = {k: v for k, v in d.items() if not k.startswith("_")}
        (yaml_dir / f"{d['_stem']}.yaml").write_text(yaml.safe_dump(truth, allow_unicode=True, sort_keys=False, width=1000),
                                                     encoding="utf-8")
        gtm.write_pdf(truth, deal["pack"], pdf_dir / f"{d['_stem']}.pdf")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for sv in W.SHOPS:
        deal = build_deal(sv, W.build_world(sv))
        print(sv.shop.key, deal["lang"], [(d["_stem"], doc_number(d), doc_refs(d)) for d in deal["docs"]][:3])
