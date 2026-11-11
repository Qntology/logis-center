"""
비전 게이트 합성 평가 세트
==========================

앱의 이미지(SigLIP2) 경로에서 '읽은 값을 받을지 · 어느 축으로 보낼지' 를 가르는 게이트를 실제 함수로 재기 위한 결정론적 데이터입니다.
전체 추출(크롭마다 LLM 판독)은 쪽당 5~25분이라, 게이트의 입력을 직접 만들어 게이트만 돌립니다
(commerce/src-tauri/tests/it/synthetic_eval.rs 의 vision_gates · label_gates).

  python make_vision_cases.py <영어 페이지 폴더> <ml 페이지 폴더> <출력 폴더>

페이지 폴더는 rasterize_trading.py 의 출력(PNG + 낱말 상자)입니다. 영어는 trading/output_pdfs, ml 은 trading/ml/output_pdfs.

1) 출처 판독성 — cases.jsonl + img/  (verify_claims_v2)
   정답 값마다 '주장'(필드 · 값 · 출처 상자)을 둡니다. 출처 상자는 앱처럼 크롭 상자입니다:
     block  값이 속한 YAML 묶음(카테고리 크롭)      line  값 줄 ±1줄(좁은 크롭)
   열화(값 글자만 건드리고 라벨은 그대로):
     clean   없음                                           → 유지
     light   한 묶음의 값 글자에 가우스 흐림 σ=1.0 (읽힘)     → 유지
     smudge  같은 묶음의 값 글자에 σ=6 (글자 높이 ~20px 이 뭉개짐) → 그 묶음의 주장은 폐기
     stamp   반투명 붉은 도장을 인명 값 위에 (아래 글자가 비침) → 유지
     blot    값 하나를 불투명 잉크 얼룩으로 덮음               → 그 주장은 폐기
     sign    인명 값 앞에서 들어오는 서명 획                   → 유지
   calib(열화 없음, 채점 안 함 — SDS 기준선이 쌓이는 평소 문서): 영어 dev 쪽 + ml 의 앞 6개 언어(ko · de · es · fr · it · pt) 첫 쪽.
   평가: 영어 test 쪽 + ml 의 나머지 언어(nl · cs · ja · zh-hans) 첫 쪽. 아랍어는 뺍니다(PDFium 이 아랍어 낱말을 표시 순서로
   돌려줘 값 위치를 못 찾음). calib 가 파일 앞에 옵니다. 하네스는 평가 사례를 사례마다 따로 된 SDS 하위 스코프에서 판정하므로
   평가 사례의 기준선은 같은 서식 코드의 calib 뿐입니다(같은 쪽의 다른 열화본이 서로의 기준선에 섞이지 않음).
2) 인명 잔해 — names.jsonl
   정답 인명 앞뒤에 서명 획이 글자로 읽힌 잔해('-Jane')를 붙인 값(→ 원래 이름)과, 양끝 구두점이 정당한 이름(→ 그대로).
3) 라벨 라우팅 — labels.jsonl
   언어 팩(trading/ml/packs)의 인쇄 라벨 → 스키마 필드. 서식(언어 × 코드) 하나가 문서 하나입니다.

열화는 Pillow 만 씁니다. 난수는 페이지 이름의 CRC32 로 고정합니다(같은 입력이면 같은 출력).
"""
import json
import random
import sys
import zlib
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from truth import ROOT, TRADING_ML, trading_docs

sys.path.insert(0, str(TRADING_ML))
from packs import PACKS  # noqa: E402

BIAS = ROOT.parent / "commerce" / "src-tauri" / "src" / "utils" / "bias.json"
KINDS = ["clean", "light", "smudge", "stamp", "blot", "sign"]
# SDS 기준선은 서식 코드 스코프에 쌓입니다. ml 은 언어마다 같은 서식 5종이라 앞 언어들의 깨끗한 쪽을 그 코드의 기준선(calib)으로,
# 뒤 언어들을 평가로 씁니다. 영어는 문서마다 코드가 달라(45종) 평가 쪽 코드에 기준선이 없습니다 — 표본 없음 경로를 잽니다.
ML_CALIB_LANGS = ("ko", "de", "es", "fr", "it", "pt")
HEAVY_SIGMA, LIGHT_SIGMA = 6.0, 1.0
PAD = 10
NAME_FIELDS = ("sender_name", "recipient_name")
RESIDUE_HEAD = ["-", "- ", "–", "—", "_", "/", "|", "~", "--"]
RESIDUE_TAIL = ["-", " -", "/", " /", "_", "|", "~"]
# 양끝 구두점이 정당한 이름 (잔해로 깎으면 안 되는 것)
LEGIT_EDGE_NAMES = [
    "ACME Co., Ltd.", "Smith & Sons Inc.", "(주)한빛물산", "Hans-Peter Müller", "Jean-Luc Moreau", "O'Brien Freight",
    "J. R. Smith Jr.", "Müller GmbH & Co. KG", "Société Générale S.A.", "A/S Nordfragt", "Kim, Min-jun",
    "\"Delta\" Trading", "Pacific Lines Ltd. (HK)", "大阪物産(株)", "Nordic Cargo A.S.", "Øresund Shipping ApS",
]

# 팩 라벨 키 → 스키마 필드 (truth.py 의 매핑과 같은 축)
LABEL_FIELDS = {
    "shipper": "sender_name", "shipper_exporter": "sender_name", "exporter": "sender_name",
    "consignee": "recipient_name", "importer_consignee": "recipient_name", "notify_party": "notify_party_name",
    "port_of_loading": "pol", "port_of_discharge": "pod", "place_of_receipt": "place_receipt",
    "place_of_delivery": "place_delivery", "vessel_flight": "vessel",
    "gross_weight_kg": "weight_gross", "total_gross_weight_kg": "weight_gross",
    "net_weight_kg": "weight_net", "total_net_weight_kg": "weight_net",
    "measurement_cbm": "volume", "total_measurement_cbm": "volume",
    "issue_date": "issue_date", "related_invoice_number": "reference_invoice", "related_po_number": "reference_po",
    "reference_po_number": "reference_po", "related_bl_number": "reference_bl", "related_lc_number": "reference_lc",
    "grand_total_amount": "amount", "incoterms": "incoterms", "currency": "currency",
    "container_number": "container_number", "seal_number": "seal_number",
    "marks_and_numbers": "marks_numbers", "mark_and_numbers": "marks_numbers",
    "number_of_packages": "package_count", "total_packages": "package_count",
    "bl_number": "doc_number", "invoice_number": "doc_number", "packing_list_number": "doc_number",
    "certificate_number": "doc_number", "policy_number": "doc_number",
}
PARTY_ROLES = {"shipper", "shipper_exporter", "exporter", "consignee", "importer_consignee", "notify_party"}


def field_categories() -> dict:
    base = json.loads(BIAS.read_text(encoding="utf-8"))["trade_schema"]["base"]
    return {f: cat for cat, fields in base.items() for f in fields}


def norm(t: str) -> str:
    return t.strip(",.:;()[]").lower()


def label_end(line_words: list) -> int:
    """줄에서 라벨이 끝나는 낱말 수. 라벨은 ':' 로 끝나는 첫 낱말까지입니다. 없으면 0(줄 전체가 값)."""
    for i, w in enumerate(line_words):
        if w["text"].endswith(":") or w["text"].endswith("："):
            return i + 1
    return 0


def page_layout(words: list) -> tuple:
    """줄 번호 → 낱말 목록, 줄 번호 → 묶음 번호, 값 낱말 집합(인덱스)."""
    lines = {}
    for i, w in enumerate(words):
        w["i"] = i
        lines.setdefault(w["line"], []).append(w)
    order = sorted(lines)
    left = min(lines[l][0]["x0"] for l in order)
    block_of, block, value_idx = {}, -1, set()
    for l in order:
        ws = lines[l]
        if block < 0 or ws[0]["x0"] < left + 12:  # 들여쓰기 0 단계 줄이 새 묶음을 엽니다
            block += 1
        block_of[l] = block
        for w in ws[label_end(ws):]:
            value_idx.add(w["i"])
    return lines, block_of, value_idx


def find_value(value, words: list, value_idx: set):
    """값 낱말이 연속으로 인쇄된 자리(값 낱말만). 없으면 None."""
    tokens = [norm(t) for t in str(value).split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    normed = [norm(w["text"]) for w in words]
    for i in range(len(normed) - len(tokens) + 1):
        if normed[i: i + len(tokens)] == tokens and all(j in value_idx for j in range(i, i + len(tokens))):
            return words[i: i + len(tokens)]
    return None


def union(ws: list, w: int, h: int, pad: int = PAD) -> list:
    return [max(0, int(min(x["x0"] for x in ws)) - pad), max(0, int(min(x["y0"] for x in ws)) - pad),
            min(w, int(max(x["x1"] for x in ws)) + pad), min(h, int(max(x["y1"] for x in ws)) + pad)]


def page_claims(doc: dict, words: list, size: tuple, cats: dict) -> list:
    lines, block_of, value_idx = page_layout(words)
    w, h = size
    out, seen = [], set()
    for field, values in doc["truth"].items():
        if field == "doc_type" or field not in cats:
            continue
        for v in values:
            if v is None or (field, str(v)) in seen:
                continue
            run = find_value(v, words, value_idx)
            if not run:
                continue
            seen.add((field, str(v)))
            line = run[0]["line"]
            block = block_of[line]
            block_words = [x for l, ws in lines.items() if block_of[l] == block for x in ws]
            near = [x for l, ws in lines.items() if block_of[l] == block and abs(l - line) <= 1 for x in ws]
            out.append({
                "category": cats[field], "field": field, "value": str(v), "block": block, "line": line,
                "words": [x["i"] for x in run],
                "bbox_block": union(block_words, w, h), "bbox_line": union(near, w, h),
            })
    return out


def value_spans(words: list, idxs: list, lines: dict) -> list:
    """값 낱말을 줄마다 하나의 가로 구간으로 묶습니다."""
    by_line = {}
    for i in idxs:
        by_line.setdefault(words[i]["line"], []).append(words[i])
    return [(min(x["x0"] for x in ws), min(x["y0"] for x in ws), max(x["x1"] for x in ws), max(x["y1"] for x in ws))
            for ws in by_line.values()]


def blur_spans(img: Image.Image, spans: list, sigma: float) -> None:
    pad = int(max(4, 2 * sigma))
    for x0, y0, x1, y1 in spans:
        box = (max(0, int(x0) - 4), max(0, int(y0) - pad), min(img.width, int(x1) + pad), min(img.height, int(y1) + pad))
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(sigma)), box)


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp(img: Image.Image, span: tuple, rng: random.Random) -> None:
    r = 95
    layer = Image.new("RGBA", (2 * r + 20, 2 * r + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    red = (196, 32, 44, 150)
    c = r + 10
    d.ellipse((c - r, c - r, c + r, c + r), outline=red, width=7)
    d.ellipse((c - r + 16, c - r + 16, c + r - 16, c + r - 16), outline=red, width=3)
    f = font(30)
    for text, dy in (("RECEIVED", -22), (f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", 18)):
        tw = d.textlength(text, font=f)
        d.text((c - tw / 2, c + dy - 15), text, font=f, fill=red)
    layer = layer.rotate(rng.uniform(-20, 20), resample=Image.BICUBIC)
    cx, cy = (span[0] + span[2]) / 2, (span[1] + span[3]) / 2
    img.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)


def blot(img: Image.Image, span: tuple) -> None:
    x0, y0, x1, y1 = (int(span[0]) - 8, int(span[1]) - 10, int(span[2]) + 8, int(span[3]) + 10)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((x0, y0, x1, y1), fill=245)
    mask = mask.filter(ImageFilter.GaussianBlur(3))
    img.paste(Image.new("RGB", img.size, (35, 30, 45)), (0, 0), mask)


def signature(img: Image.Image, span: tuple, rng: random.Random) -> None:
    """이름 첫 글자들 위를 지나는 서명 획(Catmull-Rom 곡선)."""
    x0, y0, _, y1 = span
    cy = (y0 + y1) / 2
    pts = [(x0 - 24 + i * 22 + rng.uniform(-6, 6), cy + rng.uniform(-26, 14)) for i in range(7)]
    curve = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(12):
            t = k / 12
            curve.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                                      + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t * t * t) for j in (0, 1)))
    ImageDraw.Draw(img).line(curve, fill=(25, 35, 110), width=4, joint="curve")


def degrade(img: Image.Image, kind: str, claims: list, words: list, layout: tuple, rng: random.Random) -> tuple:
    """열화를 입히고 (폐기되어야 하는 주장 번호 집합, 설명) 을 돌려줍니다. 해당 없으면 None."""
    lines, block_of, value_idx = layout
    if kind == "clean":
        return set(), "열화 없음"
    if kind in ("light", "smudge"):
        counts = {}
        for c in claims:
            counts[c["block"]] = counts.get(c["block"], 0) + 1
        block = max(sorted(counts), key=lambda b: counts[b])
        idxs = sorted(i for i, w in enumerate(words) if i in value_idx and block_of[w["line"]] == block)
        blur_spans(img, value_spans(words, idxs, lines), LIGHT_SIGMA if kind == "light" else HEAVY_SIGMA)
        hit = {k for k, c in enumerate(claims) if c["block"] == block} if kind == "smudge" else set()
        return hit, f"묶음 {block} 의 값 글자 σ={LIGHT_SIGMA if kind == 'light' else HEAVY_SIGMA}"
    names = [k for k, c in enumerate(claims) if c["field"] in NAME_FIELDS]
    if kind in ("stamp", "sign"):
        if not names:
            return None
        k = names[rng.randrange(len(names))]
        span = value_spans(words, claims[k]["words"], lines)[0]
        (stamp if kind == "stamp" else signature)(img, span, rng)
        return set(), f"{claims[k]['field']} 위 {'도장' if kind == 'stamp' else '서명 획'}"
    if kind == "blot":
        others = [k for k, c in enumerate(claims) if c["field"] not in NAME_FIELDS]
        if not others:
            return None
        k = others[rng.randrange(len(others))]
        for span in value_spans(words, claims[k]["words"], lines):
            blot(img, span)
        return {k}, f"{claims[k]['field']} 잉크 얼룩"
    raise ValueError(kind)


def name_cases(docs: list) -> list:
    names = sorted({str(v) for d in docs for f in NAME_FIELDS for v in d["truth"].get(f, []) if v})
    out = []
    for n in names:
        rng = random.Random(zlib.crc32(n.encode("utf-8")))
        out.append({"value": n, "expect": n, "kind": "legit"})
        for head in rng.sample(RESIDUE_HEAD, 2):
            out.append({"value": head + n, "expect": n, "kind": "residue_head"})
        out.append({"value": n + rng.choice(RESIDUE_TAIL), "expect": n, "kind": "residue_tail"})
    out += [{"value": n, "expect": n, "kind": "legit_edge"} for n in LEGIT_EDGE_NAMES]
    fields = ["sender_name", "recipient_name", "notify_party_name", "party_name", "signatory_name"]
    return [dict(c, field=fields[i % len(fields)], id=f"N{i:03d}") for i, c in enumerate(out)]


def label_cases() -> list:
    out = []
    for lang, pk in PACKS.items():
        labels = pk["labels"]
        for path in sorted((TRADING_ML / "input_yamls").glob(f"*_{lang.replace('-', '_')}.yaml")):  # zh-hans → *_zh_hans.yaml
            y = json.loads(json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")), default=str))
            code = y["document_code"]

            def walk(node, parent=""):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k in PARTY_ROLES and isinstance(v, dict):
                            role = labels[k]
                            if v.get("name"):
                                out.append((lang, code, path.stem, role, v["name"], LABEL_FIELDS[k]))
                                out.append((lang, code, path.stem, f"{role} {labels['name']}", v["name"], LABEL_FIELDS[k]))
                            if v.get("address") and k != "notify_party":
                                addr_field = LABEL_FIELDS[k].replace("_name", "_address")
                                out.append((lang, code, path.stem, f"{role} {labels['address']}", v["address"], addr_field))
                        elif k in LABEL_FIELDS and not isinstance(v, (dict, list)) and v not in (None, ""):
                            out.append((lang, code, path.stem, labels[k], str(v), LABEL_FIELDS[k]))
                        else:
                            walk(v, k)
                elif isinstance(node, list):
                    for x in node:
                        walk(x, parent)

            walk({k: v for k, v in y.items() if k not in ("document_type", "document_code", "language")})
    seen, rows = set(), []
    for lang, code, stem, label, value, field in out:
        if (stem, label, field) in seen:
            continue
        seen.add((stem, label, field))
        rows.append({"id": f"L{len(rows):04d}", "lang": lang, "code": code, "doc": stem, "label": label,
                     "value": value, "expect": field})
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    en_pages, ml_pages, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    (out / "img").mkdir(parents=True, exist_ok=True)
    cats = field_categories()
    docs = trading_docs()
    rows, stats = [], {}
    for doc in docs:
        lang = doc.get("lang", "en")
        if lang == "ar":
            continue
        folder = ml_pages if doc["split"] == "ml" else en_pages
        pages = sorted(folder.glob(f"{doc['stem']}_p*.png"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
        if doc["split"] == "ml":
            pages = pages[:1]
        for png in pages:
            meta = json.loads(png.with_suffix(".words.json").read_text(encoding="utf-8"))
            words = meta["words"]
            if not words:
                continue
            claims = page_claims(doc, words, (meta["width"], meta["height"]), cats)
            if not claims:
                continue
            layout = page_layout(words)
            base = {"page": png.stem, "doc": doc["stem"], "code": doc["code"], "lang": lang, "split": doc["split"]}
            calib = doc["split"] == "dev" or lang in ML_CALIB_LANGS
            kinds = ["clean"] if calib else KINDS
            for kind in kinds:
                rng = random.Random(zlib.crc32(f"{png.stem}|{kind}".encode("utf-8")))
                if kind == "clean":
                    image, hit, why = png, set(), "열화 없음"
                    rel = str(png.resolve())
                else:
                    img = Image.open(png).convert("RGB")
                    res = degrade(img, kind, claims, words, layout, rng)
                    if res is None:
                        continue
                    hit, why = res
                    rel = f"img/{png.stem}.{kind}.png"
                    img.save(out / rel, optimize=True)
                case_claims = []
                for k, c in enumerate(claims):
                    for bk in ("block", "line"):
                        case_claims.append({"category": c["category"], "field": c["field"], "value": c["value"],
                                            "bbox": c[f"bbox_{bk}"], "bbox_kind": bk,
                                            "expect": "reject" if k in hit else "keep"})
                rows.append(dict(base, case=f"{png.stem}.{kind}", kind="calib" if calib else kind,
                                 image=rel, note=why, claims=case_claims))
                stats[rows[-1]["kind"]] = stats.get(rows[-1]["kind"], 0) + 1
    rows.sort(key=lambda r: r["kind"] != "calib")  # 기준선 쪽을 먼저 (안정 정렬: 그 안의 순서는 그대로)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    names = name_cases(docs)
    with (out / "names.jsonl").open("w", encoding="utf-8") as f:
        for r in names:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = label_cases()
    with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    claims = sum(len(r["claims"]) for r in rows)
    reject = sum(c["expect"] == "reject" for r in rows for c in r["claims"])
    print(f"판독성 사례 {len(rows)}건 {stats} | 주장 {claims} (폐기 정답 {reject})")
    print(f"인명 {len(names)}건 | 라벨 {len(labels)}건 ({len(PACKS)}개 언어)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
