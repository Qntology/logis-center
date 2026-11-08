"""
무역 서류 언어 팩 (generate_trading_ml.py)
=========================================

언어마다 모듈 하나가 PACK 을 둡니다.

  lang · code      언어 (Stanza 이름) · 문서번호에 넣는 두 글자
  font · rtl       PDF 글꼴 (latin · ko · ja · zh) · 오른쪽 맞춤
  date · number    날짜 · 숫자 꼴 (generate_trading_ml.fmt_date · fmt_number)
  titles · purposes  서식 5종의 이름 · 용도 (용도는 서식 이름을 쓰지 않는 설명, 검색 질의로 씀)
  labels           정답 YAML 키 → PDF 라벨
  values           거래 하나 (수출자 · 수입자 · 운송 · 품목 2개 · 보험 · 원산지 문구)
  queries          식별자 · 당사자 · 품목 질의 틀

이름 · 주소 · 번호는 가상입니다. 실제 회사 · 기관 이름을 쓰지 않습니다.
"""
from . import ar, cs, de, es, fr, it, ja, ko, nl, pt, zh_hans

CODES = ["CI", "PL", "BL", "CO", "IP"]
LABEL_KEYS = [
    "document_type", "document_code", "item",
    # 묶음
    "metadata", "parties", "shipping_details", "transport_details", "cargo_details", "line_items", "packing_details",
    "summary", "declaration", "coverage_details",
    # 문서 정보
    "bl_number", "invoice_number", "certificate_number", "packing_list_number", "policy_number", "issue_date",
    "effective_date", "related_invoice_number", "related_po_number", "reference_po_number", "related_bl_number",
    "related_lc_number", "freight_term", "currency",
    # 당사자
    "shipper", "consignee", "notify_party", "carrier", "shipper_exporter", "exporter", "importer_consignee",
    "issuing_authority", "insurer", "insured", "name", "address", "contact_number", "tax_id",
    # 운송
    "vessel_flight", "port_of_loading", "port_of_discharge", "place_of_receipt", "place_of_delivery", "incoterms",
    # 화물 · 품목
    "container_number", "seal_number", "marks_and_numbers", "mark_and_numbers", "number_of_packages",
    "description_of_goods", "gross_weight_kg", "net_weight_kg", "measurement_cbm", "item_code", "description",
    "quantity", "unit", "unit_price", "total_amount", "hs_code", "packages", "package_type", "origin_criterion",
    # 합계
    "total_quantity", "total_items_amount", "freight_charge", "insurance", "grand_total_amount", "total_packages",
    "total_net_weight_kg", "total_gross_weight_kg", "total_measurement_cbm",
    # 원산지 선언 · 보험
    "statement", "sum_insured", "valuation_basis", "coverage_conditions", "premium_amount", "claims_payable_at",
]
VALUE_KEYS = [
    "exporter", "importer", "carrier", "insurer", "chamber", "vessel", "container_owner", "pol", "pod", "receipt",
    "delivery", "incoterms", "currency", "freight_term", "freight_rate", "insurance_rate", "unit", "package_type",
    "packages_text", "marks", "goods", "goods_all", "origin_criterion", "statement", "valuation_basis",
    "coverage_conditions", "claims_payable_at",
]
GOODS_KEYS = ["code", "desc", "hs", "qty", "price", "net", "gross", "packages", "cbm"]


def _check(pk: dict) -> None:
    lang = pk["lang"]
    assert set(pk["titles"]) == set(CODES) and set(pk["purposes"]) == set(CODES), f"{lang}: titles · purposes"
    assert len(set(pk["titles"].values())) == len(CODES), f"{lang}: 서식 이름이 겹칩니다"
    missing = [k for k in LABEL_KEYS if k not in pk["labels"]]
    assert not missing, f"{lang}: labels 에 없는 키 {missing}"
    v = pk["values"]
    missing = [k for k in VALUE_KEYS if k not in v]
    assert not missing, f"{lang}: values 에 없는 키 {missing}"
    for who in ("exporter", "importer"):
        assert all(v[who].get(k) for k in ("name", "address", "contact_number", "tax_id")), f"{lang}: {who}"
    assert all(v[k].get("name") and v[k].get("address") for k in ("insurer", "chamber")), f"{lang}: insurer · chamber"
    assert len(v["goods"]) == 2 and all(set(g) == set(GOODS_KEYS) for g in v["goods"]), f"{lang}: goods"
    assert "{n}" in v["packages_text"], f"{lang}: packages_text"
    assert len(v["container_owner"]) == 4 and v["container_owner"].endswith("U"), f"{lang}: container_owner"
    assert "{number}" in pk["queries"]["identifier"] and "{name}" in pk["queries"]["party"] \
        and "{goods}" in pk["queries"]["goods"], f"{lang}: queries"
    assert pk["font"] in ("latin", "ko", "ja", "zh") and pk["number"] in ("dot_comma", "space_comma", "comma_dot")


PACKS = {}
for _mod in (ko, de, es, fr, it, pt, nl, cs, ar, ja, zh_hans):
    _check(_mod.PACK)
    PACKS[_mod.PACK["lang"]] = _mod.PACK
