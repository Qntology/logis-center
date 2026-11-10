"""
관계형 합성 세계 (v2)
====================

쇼핑몰 네 곳(한국어 · 영어 · 일본어 · 독일어)의 세계를 한 번에 만듭니다. v1(../commerce)은 쇼핑몰마다 표가 따로 놀아
주문자 · 리뷰 작성자가 매번 새 사람이고 쿠폰 · 이벤트가 주문과 이어지지 않았습니다. 여기서는 같은 회원이 여러 번 사고,
주문이 쿠폰 · 이벤트 · 운송장 여러 개 · 리뷰 · 반품과 이어지며, 수입 상품은 무역 거래(PO → CI · PL · BL · CO · IP)와
행동 로그(상품 보기 → 장바구니 → 주문 완료)까지 같은 식별자로 엮입니다. 질의의 정답은 이 세계에서 계산합니다.

표기(날짜 · 금액 · 상태어 · 사람 이름 · 상품 이름)는 v1 생성기와 언어 팩을 그대로 씁니다. 네 쇼핑몰은 같은 테마
(admin_flat)라서 언어 사이의 차이가 레이아웃 차이와 섞이지 않습니다.
"""
import hashlib
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "commerce"))

import generate_commerce_pages as gcp  # noqa: E402  (v1 생성기: 표기 · 언어 팩 · 상품 사전)

SEED = 20261108


def rng_for(*parts) -> random.Random:
    digest = hashlib.sha256("|".join([str(SEED)] + [str(p) for p in parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class ShopV2:
    shop: gcp.Shop          # v1 의 Shop (표기 · 언어 팩이 이것을 받습니다)
    prefix: str             # 상품 번호 머리
    trade_lang: str         # 이 쇼핑몰이 수입하는 상대(수출자)의 서류 언어
    importer: str           # 무역 서류의 수입자 이름 (쇼핑몰 운영 회사, 그 언어 그대로)
    importer_address: str


def _shop(key, name, domain, lang, company, address, phone, currency, date_style, money_style) -> gcp.Shop:
    return gcp.Shop(key, name, domain, lang, "admin_flat", "v2", company, address, phone, currency, date_style, money_style)


SHOPS = [
    ShopV2(_shop("sorae", "소래마켓", "sorae-market.test", "ko", "(주)소래상회", "인천광역시 남동구 소래역로 12 3층", "1588-0412",
                 "KRW", "iso_time", "won_suffix"), "SR", "de", "(주)소래상회", "인천광역시 남동구 소래역로 12 3층"),
    ShopV2(_shop("copper", "Copperleaf Goods", "copperleaf-goods.test", "en", "Copperleaf Goods Inc.",
                 "1220 Larkspur Ave, Boise, ID 83702", "+1-208-555-0163", "USD", "us_slash", "dollar"),
           "CL", "ja", "Copperleaf Goods Inc.", "1220 Larkspur Ave, Boise, ID 83702, USA"),
    ShopV2(_shop("komorebi", "こもれび商店", "komorebi-shop.test", "ja", "株式会社こもれび", "東京都杉並区阿佐谷北2-14-7",
                 "03-5555-0148", "JPY", "ja_long", "yen"), "KM", "ko", "株式会社こもれび", "東京都杉並区阿佐谷北2-14-7"),
    ShopV2(_shop("kranich", "Kranichhof Markt", "kranichhof-markt.test", "de", "Kranichhof Handels GmbH",
                 "Lindenallee 21, 04109 Leipzig", "+49 341 5550187", "EUR", "dmy_dot", "eur_suffix"),
           "KR", "en", "Kranichhof Handels GmbH", "Lindenallee 21, 04109 Leipzig, Deutschland"),
]
BY_KEY = {s.shop.key: s for s in SHOPS}

N_GOODS, N_MEMBERS, N_ORDERS, N_REVIEWS = 18, 12, 32, 14
WINDOW = (datetime(2026, 4, 1, 9, 0), datetime(2026, 9, 30, 21, 0))

# 상태는 영어 상태어로 들고 다니고, 화면에 낼 때 언어로 바꿉니다(status_label).
ORDER_STATUS = ["Awaiting payment", "Paid", "Processing", "Shipped", "Delivered", "Cancelled", "Returned"]
STATUS_KO = {"Awaiting payment": "입금대기", "Paid": "결제완료", "Processing": "상품준비중", "Shipped": "배송중",
             "Delivered": "배송완료", "Cancelled": "취소", "Returned": "반품",
             "Approved": "승인", "Pending": "대기중", "Hidden": "숨김", "Live": "진행중", "Ended": "종료", "Scheduled": "대기중",
             "Active": "판매중", "Sold out": "품절", "Draft": "진열안함"}


def label(sv: ShopV2, en: str) -> str:
    """영어 화면 문구 · 상태어 → 쇼핑몰 언어."""
    lang = sv.shop.lang
    if lang == "en":
        return en
    if lang == "ko":
        return STATUS_KO.get(en, en)
    return gcp.tr(sv.shop, en)


def price(sv: ShopV2, won: int) -> int:
    return gcp.price_of(won, sv.shop)


# ──────────────────────────────────────────────
# 상품 · 회원 · 쿠폰 · 이벤트
# ──────────────────────────────────────────────
def build_goods(sv: ShopV2) -> list:
    r = rng_for(sv.shop.key, "goods")
    picked = r.sample(range(len(gcp.PRODUCTS)), N_GOODS)
    out = []
    for i, pi in enumerate(picked, start=1):
        name_ko, name_en, concept, cat_ko, desc_ko, desc_en, base = gcp.PRODUCTS[pi]
        gr = rng_for(sv.shop.key, "goods", i)
        lang = sv.shop.lang
        local = None if lang in ("ko", "en") else gcp.PACKS[lang]["products"][gcp.PRODUCT_KEYS[name_en]]
        title = name_ko if lang == "ko" else (name_en if lang == "en" else local[0])
        stock = gr.choice([0, 4, 9, 25, 60, 140])
        sale = price(sv, int(round(base * gr.uniform(0.93, 1.1) / 100)) * 100)
        out.append({
            "id": f"{sv.prefix}{20000 + i * 41}",
            "title": title,
            "title_en": name_en,
            "title_ko": name_ko,
            "pi": pi,
            "concept": concept,
            "category": label(sv, gcp.CATEGORY_EN[cat_ko]) if lang != "ko" else cat_ko,
            "status": "Sold out" if stock == 0 else "Active",
            "sale_price": sale,
            "stock": stock,
            "registered": date(2026, 1, 10) + timedelta(days=gr.randint(0, 70)),
            "brand": gr.choice(gcp.BRANDS),
            # 수입 상품이면 무역 거래가 이 상품을 싣습니다 (build_trade).
            "imported": False,
        })
    return out


def build_members(sv: ShopV2) -> list:
    out, names = [], set()
    i = 0
    while len(out) < N_MEMBERS:
        i += 1
        r = rng_for(sv.shop.key, "member", i)
        p = gcp.person(sv.shop, r)
        if p["name"] in names:
            continue
        names.add(p["name"])
        out.append({"id": f"M{sv.prefix}{3000 + len(out) * 7}", **p, "joined": date(2025, 6, 1) + timedelta(days=r.randint(0, 300))})
    return out


COUPON_NAMES_KO = ["첫 구매 환영 혜택", "여름 휴가 응원 쿠폰", "재구매 고객 감사", "가을 맞이 감사제"]
COUPON_NAMES_EN = ["First order welcome", "Summer getaway coupon", "Returning customer perk", "Autumn thank-you sale"]
EVENT_NAMES_KO = ["봄 산책 기획전", "여름 휴가 준비전", "추석 선물 모음"]
EVENT_NAMES_EN = ["Spring stroll showcase", "Summer getaway picks", "Harvest gift guide"]


def build_events_and_coupons(sv: ShopV2, goods: list) -> tuple:
    """이벤트 3개(봄 · 여름 · 가을)는 기간이 겹치지 않고, 각 이벤트가 상품 5개를 걸고 쿠폰 하나를 줍니다.
    쿠폰 하나(재구매 감사)는 이벤트와 무관하게 상시입니다."""
    lang = sv.shop.lang
    pk = None if lang in ("ko", "en") else gcp.PACKS[lang]
    ev_names = EVENT_NAMES_KO if lang == "ko" else (EVENT_NAMES_EN if lang == "en" else pk["event_names"][:3])
    cp_names = COUPON_NAMES_KO if lang == "ko" else (COUPON_NAMES_EN if lang == "en" else
                                                     [pk["coupon_names"][0], pk["coupon_names"][1], pk["coupon_names"][3], pk["coupon_names"][1]])
    if lang not in ("ko", "en"):
        # 팩의 쿠폰 이름 6개 중 '여름'과 '가을'에 맞는 것을 고릅니다: 0 첫 구매, 1 가을 감사, 2 주말, 3 재구매, 4 앱, 5 장바구니.
        cp_names = [pk["coupon_names"][0], pk["coupon_names"][4], pk["coupon_names"][3], pk["coupon_names"][1]]
    r = rng_for(sv.shop.key, "events")
    periods = [(date(2026, 4, 6), date(2026, 4, 26)), (date(2026, 6, 15), date(2026, 7, 12)), (date(2026, 9, 1), date(2026, 9, 20))]
    pool = list(goods)
    r.shuffle(pool)
    coupons = [
        {"id": f"CP{5000 + 1 * 13}", "code": f"{sv.prefix}WELCOME", "title": cp_names[0], "kind": "fixed",
         "value": price(sv, 3000), "start": date(2026, 4, 1), "end": date(2026, 9, 30), "min": price(sv, 20000), "event": None},
        {"id": f"CP{5000 + 2 * 13}", "code": f"{sv.prefix}SUMMER", "title": cp_names[1], "kind": "percent",
         "value": 15, "start": periods[1][0], "end": periods[1][1], "min": price(sv, 30000), "event": None},
        {"id": f"CP{5000 + 3 * 13}", "code": f"{sv.prefix}AGAIN", "title": cp_names[2], "kind": "percent",
         "value": 10, "start": date(2026, 4, 1), "end": date(2026, 9, 30), "min": price(sv, 10000), "event": None},
        {"id": f"CP{5000 + 4 * 13}", "code": f"{sv.prefix}HARVEST", "title": cp_names[3], "kind": "fixed",
         "value": price(sv, 5000), "start": periods[2][0], "end": periods[2][1], "min": price(sv, 40000), "event": None},
    ]
    events = []
    for k, (start, end) in enumerate(periods):
        featured = pool[k * 5:(k + 1) * 5]
        ev = {"id": f"EV{700 + (k + 1) * 9}", "title": ev_names[k], "start": start, "end": end,
              "goods": [g["id"] for g in featured],
              "coupon": coupons[1]["id"] if k == 1 else (coupons[3]["id"] if k == 2 else None)}
        events.append(ev)
    coupons[1]["event"] = events[1]["id"]
    coupons[3]["event"] = events[2]["id"]
    return events, coupons


def status_on(today: date, start: date, end: date) -> str:
    return "Scheduled" if today < start else ("Ended" if today > end else "Live")


# ──────────────────────────────────────────────
# 주문 · 운송장 · 리뷰
# ──────────────────────────────────────────────
TODAY = date(2026, 10, 5)   # 화면의 '오늘'. 이벤트 · 쿠폰 상태와 주문 상태 진행이 이 날짜 기준입니다.


def _carriers(sv: ShopV2) -> list:
    lang = sv.shop.lang
    return gcp.CARRIERS_KO if lang == "ko" else (gcp.CARRIERS_EN if lang == "en" else gcp.PACKS[lang]["carriers"])


def build_orders(sv: ShopV2, goods: list, members: list, events: list, coupons: list) -> tuple:
    by_id = {g["id"]: g for g in goods}
    weights = [6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1]   # 단골 몇 명이 여러 번 삽니다
    orders, shipments = [], []
    for i in range(1, N_ORDERS + 1):
        r = rng_for(sv.shop.key, "order", i)
        when = WINDOW[0] + timedelta(minutes=r.randint(0, int((WINDOW[1] - WINDOW[0]).total_seconds() // 60)))
        member = r.choices(members, weights=weights)[0]
        receiver = dict(member)
        if r.random() < 0.25:   # 선물: 받는 사람이 다른 회원
            receiver = dict(r.choice([m for m in members if m["id"] != member["id"]]))
        live = [e for e in events if e["start"] <= when.date() <= e["end"]]
        n_lines = r.choice([1, 1, 2, 2, 3])
        picks = []
        if live and r.random() < 0.75:   # 이벤트 기간이면 이벤트 상품을 하나 이상 담습니다
            picks.append(by_id[r.choice(live[0]["goods"])])
        for g in r.sample(goods, len(goods)):
            if len(picks) >= n_lines:
                break
            if g not in picks:
                picks.append(g)
        lines = [{"goods": g["id"], "qty": r.choice([1, 1, 1, 2, 3]), "unit": g["sale_price"]} for g in picks]
        subtotal = sum(l["qty"] * l["unit"] for l in lines)
        coupon = None
        usable = [c for c in coupons if c["start"] <= when.date() <= c["end"] and subtotal >= c["min"]]
        event_coupon = [c for c in usable if c["event"] and live and c["event"] == live[0]["id"]]
        if event_coupon and r.random() < 0.7:
            coupon = event_coupon[0]
        elif usable and r.random() < 0.25:
            coupon = r.choice(usable)
        discount = 0
        if coupon:
            discount = coupon["value"] if coupon["kind"] == "fixed" else int(subtotal * coupon["value"] / 100)
            if sv.shop.currency in ("KRW", "JPY", "CZK"):
                discount = int(round(discount, -1))
        fee = 0 if subtotal >= price(sv, 50000) else price(sv, 3000)
        age = (TODAY - when.date()).days
        # 오래된 주문일수록 진행이 끝나 있습니다. 취소 · 반품은 일부만.
        if age < 2:
            status = r.choice(["Awaiting payment", "Paid"])
        elif age < 5:
            status = r.choice(["Processing", "Shipped"])
        else:
            status = r.choices(["Delivered", "Cancelled", "Returned", "Shipped"], weights=[14, 2, 2, 1])[0]
        if when.date() < date(2026, 9, 25) and status in ("Awaiting payment", "Paid", "Processing"):
            status = "Delivered"
        order = {
            "id": f"{when:%Y%m%d}-{300000 + i * 71}",
            "seq": i,
            "when": when,
            "member": member["id"],
            "buyer": member,
            "receiver": receiver,
            "lines": lines,
            "subtotal": subtotal,
            "discount": discount,
            "fee": fee,
            "amount": subtotal - discount + fee,
            "coupon": coupon["id"] if coupon else None,
            "event": live[0]["id"] if live and any(l["goods"] in live[0]["goods"] for l in lines) else None,
            "status": status,
            "payment": r.choice(["Credit card", "Credit card", "Wire transfer", "Digital wallet"]),
            "tracking": [],
        }
        # 운송장: 상품이 2개 이상이면 나눠 보내기도 합니다(부분 배송). 반품은 회수 운송장이 하나 더 붙습니다.
        if status in ("Shipped", "Delivered", "Returned"):
            groups = [lines]
            if len(lines) >= 2 and r.random() < 0.45:
                cut = r.randint(1, len(lines) - 1)
                groups = [lines[:cut], lines[cut:]]
            carriers = _carriers(sv)
            for k, part in enumerate(groups):
                ship_at = when + timedelta(days=1 + k * r.randint(1, 3), hours=r.randint(2, 9))
                done = status in ("Delivered", "Returned") or (k == 0 and len(groups) > 1 and r.random() < 0.6)
                no = f"{r.randint(10 ** 11, 10 ** 12 - 1)}"
                shipments.append({
                    "id": no, "order": order["id"], "kind": "outbound", "carrier": r.choice(carriers),
                    "goods": [l["goods"] for l in part], "shipped": ship_at,
                    "delivered": (ship_at + timedelta(days=r.randint(1, 3))) if done else None,
                    "status": "Delivered" if done else "Shipped", "receiver": receiver,
                })
                order["tracking"].append(no)
            if status == "Returned":
                no = f"{r.randint(10 ** 11, 10 ** 12 - 1)}"
                back = shipments[-1]["delivered"] + timedelta(days=r.randint(2, 6))
                shipments.append({
                    "id": no, "order": order["id"], "kind": "return", "carrier": r.choice(carriers),
                    "goods": [lines[0]["goods"]], "shipped": back, "delivered": back + timedelta(days=2),
                    "status": "Returned", "receiver": {"name": sv.shop.company, "address": sv.shop.company_address,
                                                       "phone": sv.shop.company_phone},
                })
                order["tracking"].append(no)
        orders.append(order)
    orders.sort(key=lambda o: o["when"], reverse=True)
    return orders, shipments


def build_reviews(sv: ShopV2, orders: list) -> list:
    lang = sv.shop.lang
    if lang == "ko":
        tmpl = ["생각보다 %s 만족합니다.", "두 번째 구매인데 역시 %s 좋네요.", "포장이 꼼꼼했고 %s 괜찮습니다."]
        adj = ["튼튼해서", "가벼워서", "마감이 좋아서", "색이 예뻐서"]
        bad = ["배송 중에 파손되어 실망했습니다.", "설명과 달라서 반품했습니다.", "금방 고장 났어요."]
    elif lang == "en":
        tmpl = ["Better than expected, really %s.", "Second purchase and still %s.", "Packed with care and %s overall."]
        adj = ["sturdy", "lightweight", "well finished", "lovely in colour"]
        bad = ["Arrived damaged, very disappointed.", "Not as described, sent it back.", "Broke within a week."]
    else:
        pk = gcp.PACKS[lang]
        tmpl, adj = pk["review_tmpl"], pk["review_adj"]
        bad = BAD_REVIEWS[lang]
    done = [o for o in orders if o["status"] in ("Delivered", "Returned")]
    r = rng_for(sv.shop.key, "reviews")
    picks = r.sample(done, min(N_REVIEWS, len(done)))
    out = []
    for k, o in enumerate(sorted(picks, key=lambda o: o["when"])):
        rr = rng_for(sv.shop.key, "review", k)
        line = rr.choice(o["lines"]) if o["status"] == "Delivered" else o["lines"][0]
        low = o["status"] == "Returned" or rr.random() < 0.18
        rating = rr.choice([1, 2]) if low else rr.choice([4, 5, 5])
        out.append({
            "id": f"RV{8100 + k * 17}", "order": o["id"], "goods": line["goods"], "member": o["member"],
            "author": o["buyer"]["name"], "rating": rating,
            "text": rr.choice(bad) if low else rr.choice(tmpl) % rr.choice(adj),
            "posted": (o["when"] + timedelta(days=rr.randint(5, 14))).date(),
            "status": "Approved" if rr.random() < 0.85 else "Pending",
        })
    return out


BAD_REVIEWS = {
    "ja": ["配送中に破損していて残念でした。", "説明と違ったので返品しました。", "すぐに壊れました。"],
    "de": ["Beim Versand beschädigt, sehr enttäuschend.", "Nicht wie beschrieben, zurückgeschickt.", "Nach einer Woche kaputt."],
}


def build_world(sv: ShopV2) -> dict:
    goods = build_goods(sv)
    members = build_members(sv)
    events, coupons = build_events_and_coupons(sv, goods)
    orders, shipments = build_orders(sv, goods, members, events, coupons)
    reviews = build_reviews(sv, orders)
    return {"shop": sv, "goods": goods, "members": members, "events": events, "coupons": coupons,
            "orders": orders, "shipments": shipments, "reviews": reviews}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for sv in SHOPS:
        w = build_world(sv)
        split = sum(1 for o in w["orders"] if len([t for t in o["tracking"]]) >= 2)
        print(sv.shop.key, sv.shop.lang, {k: len(v) for k, v in w.items() if isinstance(v, list)},
              "coupon orders", sum(1 for o in w["orders"] if o["coupon"]), "event orders", sum(1 for o in w["orders"] if o["event"]),
              "multi-tracking", split, "statuses", sorted({o["status"] for o in w["orders"]}))
