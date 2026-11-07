"""
합성 데이터 정답 읽기
====================

커머스: commerce/input_yamls/*.yaml 이 이미 스키마 이름으로 되어 있습니다.
무역:   trading/input_yamls/*.yaml 의 키는 서식마다 달라서(거의 600종) 스키마 이름으로 옮깁니다.
        값이 어느 스키마 필드인지 애매한 키는 정답에 넣지 않습니다. 평가는 '옮길 수 있는 값' 만 봅니다.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMMERCE = ROOT / "commerce"
TRADING = ROOT / "trading"


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# 분할: 튜닝에 쓰지 않는 test 는 Part 번호로 고정합니다 (3 의 배수 15건).
# ──────────────────────────────────────────────
def trading_split(stem: str) -> str:
    part = int(re.search(r"Part(\d+)", stem).group(1))
    return "test" if part % 3 == 0 else "dev"


# ──────────────────────────────────────────────
# 무역 키 → 스키마 필드
# ──────────────────────────────────────────────
SENDER_ROLES = ("shipper", "exporter", "seller", "supplier", "beneficiary", "drawer")
RECIPIENT_ROLES = ("consignee", "buyer", "importer", "drawee")

REFERENCE_KEYS = {
    "po": "reference_po", "invoice": "reference_invoice", "bl": "reference_bl", "lc": "reference_lc",
    "booking": "reference_booking", "sc": "reference_contract", "contract": "reference_contract",
    "master_bl": "reference_master_bl", "sr": "reference_sr",
}
LEAF_FIELDS = {
    "port_of_loading": "pol", "port_of_discharge": "pod", "vessel_flight": "vessel", "vessel_name": "vessel",
    "vessel": "vessel", "incoterms": "incoterms", "place_of_receipt": "place_receipt",
    "place_of_delivery": "place_delivery", "payment_terms": "payment_terms", "currency": "currency",
    "container_number": "container_number", "seal_number": "seal_number",
    "gross_weight_kg": "weight_gross", "total_gross_weight_kg": "weight_gross",
    "total_net_weight_kg": "weight_net", "measurement_cbm": "volume", "total_measurement_cbm": "volume",
    "grand_total_amount": "amount", "total_amount": "amount", "grand_total_amount_krw": "amount",
    "expiry_date": "expiry_date", "due_date": "due_date",
    "estimated_time_of_arrival_eta": "eta", "estimated_time_of_departure_etd": "etd",
    "partial_shipments": "partial_shipments",
}
ITEM_SECTIONS = ("line_items", "purchased_items", "licensed_items", "goods_classification", "packing_details")
ITEM_FIELDS = {
    "item_code": "item_code", "description": "description", "quantity": "quantity", "unit": "unit",
    "unit_price": "unit_price", "total_amount": "total_price", "hs_code": "hs_code",
}


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for x in node:
            yield from _walk(x, path + ("[]",))
    else:
        yield path, node


def trade_truth(doc: dict) -> dict:
    """스키마 필드 → 값 목록. 같은 필드에 값이 여럿일 수 있습니다(표의 행)."""
    out = {"doc_type": [doc["document_code"]]}
    meta = doc.get("metadata") or {}
    keys = list(meta.keys())
    if keys:
        out.setdefault("doc_number", []).append(meta[keys[0]])
    for k, v in meta.items():
        if k == "issue_date":
            out.setdefault("issue_date", []).append(v)
        m = re.fullmatch(r"(?:related|reference)_(.+)_number", k)
        if m and m.group(1) in REFERENCE_KEYS:
            out.setdefault(REFERENCE_KEYS[m.group(1)], []).append(v)
        if k in LEAF_FIELDS:
            out.setdefault(LEAF_FIELDS[k], []).append(v)
    for role, info in (doc.get("parties") or {}).items():
        if not isinstance(info, dict):
            continue
        # 복합 역할(shipper_exporter · applicant_exporter · importer_consignee)은 단어 포함으로 봅니다.
        # applicant 단독처럼 서식마다 뜻이 갈리는 역할은 정답에 넣지 않습니다.
        kind = "sender" if any(r in role for r in SENDER_ROLES) else "recipient" if any(r in role for r in RECIPIENT_ROLES) else None
        if not kind:
            continue
        if info.get("name"):
            out.setdefault(f"{kind}_name", []).append(info["name"])
        if info.get("address"):
            out.setdefault(f"{kind}_address", []).append(info["address"])
    for path, v in _walk({k: x for k, x in doc.items() if k not in ("metadata", "parties", "document_type", "document_code")}):
        if v is None:
            continue
        leaf = path[-1]
        if path[0] in ITEM_SECTIONS and leaf in ITEM_FIELDS:
            out.setdefault(ITEM_FIELDS[leaf], []).append(v)
        elif leaf in LEAF_FIELDS:
            out.setdefault(LEAF_FIELDS[leaf], []).append(v)
    return out


def trading_docs() -> list:
    docs = []
    for p in sorted((TRADING / "input_yamls").glob("*.yaml")):
        y = read_yaml(p)
        docs.append({
            "stem": p.stem, "code": y["document_code"], "title": y["document_type"],
            "split": trading_split(p.stem), "truth": trade_truth(y), "raw": y,
            "pdf": str((TRADING / "output_pdfs" / f"{p.stem}.pdf").as_posix()),
        })
    return docs


# ──────────────────────────────────────────────
# 무역 검색 질의
# 45건이 한 거래의 서류라 PO · B/L · 당사자는 거의 모든 문서가 같습니다. 그래서 문서를 가르는 것은
# 문서번호 · 서식 이름 · 서식의 용도뿐이고, 질의도 이 셋으로 만듭니다. 용도 질의는 서식 이름을 쓰지 않습니다.
# ──────────────────────────────────────────────
TRADE_NAMES_KO = {
    "AWB": "항공화물운송장", "AN": "화물도착통지서", "BE": "환어음", "BL": "선하증권", "BK": "선복예약확인서",
    "IP": "적하보험증권", "CM": "적하목록", "CDR": "화물검정보고서", "COA": "성분분석증명서", "CNM": "비가공증명서",
    "CO": "원산지증명서", "CI": "상업송장", "CP": "구매확인서", "CSI": "영사송장", "CN": "대변표", "CCC": "통관증명서",
    "DGD": "위험물신고서", "DN": "차변표", "DO": "화물인도지시서", "ED": "수출신고서", "EL": "수출승인서",
    "FCR": "포워더화물수령증", "FI": "운임청구서", "FC": "훈증증명서", "HBL": "하우스 선하증권", "ID": "수입신고서",
    "IC": "검사증명서", "ICF": "보험금청구서", "LC": "신용장", "LG": "수입화물선취보증서", "LLC": "내국신용장",
    "PL": "포장명세서", "PC": "식물검역증명서", "PI": "견적송장", "POD": "화물인수증", "PO": "구매주문서",
    "SC": "매매계약서", "SWB": "해상화물운송장", "SA": "선적통지서", "SR": "선적요청서", "SOA": "거래잔액명세서",
    "TI": "세금계산서", "TR": "수입화물대도증", "WR": "창고증권", "WC": "중량증명서",
}
TRADE_PURPOSE_KO = {
    "AWB": "비행기로 보낸 화물의 운송 계약 서류", "AN": "배가 곧 들어온다고 수하인에게 알리는 서류",
    "BE": "은행을 통해 대금 지급을 요구하는 증권", "BL": "선사가 화물을 받고 내주는 소유권 증권",
    "BK": "선사가 배의 자리를 잡아 주었다고 확인한 서류", "IP": "운송 중 사고에 대비해 가입한 보험 서류",
    "CM": "한 배에 실린 화물 전체를 나열한 서류", "CDR": "화물 손상을 조사한 전문 기관의 결과 보고",
    "COA": "제품의 성분을 시험한 결과 서류", "CNM": "경유지에서 손대지 않았음을 세관이 증명한 서류",
    "CO": "물품이 어느 나라에서 만들어졌는지 증명하는 서류", "CI": "수입자에게 품목과 금액을 청구하는 서류",
    "CP": "국내 공급 물품이 수출용 원자재임을 확인하는 서류", "CSI": "수입국 영사가 공증한 인보이스",
    "CN": "금액을 깎아 주거나 돌려줄 때 보내는 정산 서류", "CCC": "세관 절차가 끝났음을 확인하는 서류",
    "DGD": "폭발성이나 인화성 화물을 운송할 때 내는 서류", "DN": "추가 비용을 청구할 때 보내는 정산 서류",
    "DO": "창고나 터미널에 화물을 내주라고 지시하는 서류", "ED": "물품을 내보낼 때 세관에 내는 서류",
    "EL": "수출 허가를 받았음을 보여 주는 서류", "FCR": "운송주선인이 화물을 받았다고 내주는 서류",
    "FI": "운송료를 청구하는 서류", "FC": "목재 포장의 해충을 약품으로 처리한 증명",
    "HBL": "운송주선인이 화주에게 발행하는 운송 증권", "ID": "물품을 들여올 때 세관에 내는 서류",
    "IC": "선적 전에 품질과 수량을 확인한 결과 서류", "ICF": "손해를 입어 보험사에 돈을 요구하는 서류",
    "LC": "은행이 대금 지급을 약속하는 서류", "LG": "원본 서류 없이 화물을 먼저 찾을 때 은행이 내주는 보증",
    "LLC": "국내 공급자에게 은행이 지급을 약속하는 서류", "PL": "상자별 수량과 무게를 적은 서류",
    "PC": "농산물에 병해충이 없음을 정부가 증명한 서류", "PI": "거래 전에 예상 금액을 제시하는 서류",
    "POD": "화물을 받았다고 서명한 서류", "PO": "구매자가 물품을 발주하는 서류",
    "SC": "매도인과 매수인이 조건을 합의한 서류", "SWB": "유통되지 않는 해상 운송 계약 서류",
    "SA": "수출자가 배에 실었다고 수입자에게 알리는 서류", "SR": "화주가 운송인에게 선적을 의뢰하는 서류",
    "SOA": "거래처별 입금과 미수 내역을 정리한 서류", "TI": "부가세가 표시된 국내 거래 증빙",
    "TR": "대금을 내기 전에 은행 담보 화물을 빌려 찾는 서류", "WR": "창고에 화물을 맡겼음을 증명하는 서류",
    "WC": "화물 무게를 공인 계량한 서류",
}


def trading_queries() -> list:
    out = []
    for d in trading_docs():
        code, number = d["code"], d["truth"]["doc_number"][0]
        common = {"split": d["split"], "domain": "shipping", "relevant": [d["stem"]]}
        out.append({**common, "kind": "identifier", "lang": "ko", "query": f"문서번호 {number} 찾아줘"})
        out.append({**common, "kind": "doc_title", "lang": "en", "query": d["title"]})
        out.append({**common, "kind": "doc_title", "lang": "ko", "query": TRADE_NAMES_KO[code]})
        out.append({**common, "kind": "doc_purpose", "lang": "ko", "query": TRADE_PURPOSE_KO[code]})
    return out


# ──────────────────────────────────────────────
# 커머스
# ──────────────────────────────────────────────
def commerce_pages() -> list:
    return [read_yaml(p) for p in sorted((COMMERCE / "input_yamls").glob("*.yaml"))]


def commerce_queries() -> list:
    return read_yaml(COMMERCE / "queries.yaml")


def commerce_records() -> list:
    return read_yaml(COMMERCE / "records.yaml")


# ──────────────────────────────────────────────
# 12개 언어 검색 시나리오 (eval/queries_12lang.yaml)
# Stanza 변환본 12개 언어가 검색 경로에서 도는지 보려고 같은 정답 집합의 질의를 12개 언어로 씁니다.
# 계획 id 는 이 목록의 순번입니다 (커머스 · 무역 따로).
# ──────────────────────────────────────────────
LANGS_12 = ["en", "ko", "de", "es", "fr", "it", "pt", "nl", "cs", "ar", "ja", "zh-hans"]


def multilingual_queries(domain: str) -> list:
    spec = read_yaml(Path(__file__).resolve().parent / "queries_12lang.yaml")
    out = []
    for entry in spec["commerce" if domain == "commerce" else "shipping"]:
        for lang in LANGS_12:
            out.append({
                "split": "12lang", "shop": entry.get("shop", ""), "kind": entry["kind"], "key": entry["key"],
                "lang": lang, "domain": domain, "query": entry["query"][lang], "relevant": list(entry["relevant"]),
            })
    return out


if __name__ == "__main__":
    import collections
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    docs = trading_docs()
    print("무역 문서", len(docs), collections.Counter(d["split"] for d in docs))
    fields = collections.Counter()
    for d in docs:
        fields.update(d["truth"].keys())
    print(fields.most_common())
    print(docs[11]["stem"], {k: v[:2] for k, v in docs[11]["truth"].items()})
    print("무역 질의", collections.Counter((q["split"], q["kind"]) for q in trading_queries()))
