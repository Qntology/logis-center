"""
행동 로그 (v2): Cloudflare D1 대신 쓰는 가짜 원시 행
=====================================================

앱은 console.logis.center Worker 가 D1 에서 꺼낸 원시 이벤트를 받아(modes/analytic.ts) 저장하고, Qwen3.5 로 문장으로
구조화한 뒤 임베딩합니다. 그 Worker 가 지금 연결되어 있지 않아, 같은 모양의 행을 세계에서 만듭니다.

도메인 · 웹 경로 정의 (sites.yaml)
  · 도메인 4개 = 쇼핑몰 4곳의 구매자 사이트 (host · 사이트 이름 · 언어)
  · 경로 10종 = 페이지 종류 (home · goods_list · goods_view · search · cart · checkout · order_complete · event_view ·
    mypage_orders · review_write). 상품 · 주문 · 이벤트 번호는 앱의 relay 별칭이 아는 매개변수(goodsNo · orderNo · eventNo)로
    URL 에 실려, 구조화 단계의 bind_url_relays 가 행동 → 상품 · 주문 연결을 새깁니다.

원시 행 (events.jsonl, 한 줄 = 이벤트 하나)
  {id, type(click · hover · change · touch), from(방문자 주소), host, path(쿼리 포함), created_at(ms),
   data: {origin, link, site, title, device, action:[대상 요소 outerHTML], relate:[옆 요소 outerHTML ...]}}
  cc · bcc · ref · to 는 하네스가 앱과 같은 식으로 채웁니다(tests/it/synthetic_eval.rs analytic_rows).

세션은 세계의 주문 · 리뷰 · 이벤트에서 나옵니다: 구매(이벤트 페이지 → 상품 → 장바구니 → 결제 → 주문 완료),
장바구니 이탈(결제 페이지에서 멈춤), 리뷰 작성, 검색, 이벤트 쿠폰 받기. 정답(weblog/truth.yaml)은 이벤트마다
페이지 종류 · 행동 종류 · 상품 · 주문 · 쿠폰 · 이벤트 · 기기 · 세션을 둡니다.
"""
import hashlib
import html
from datetime import datetime, timedelta

import world as W

E = html.escape

ROUTES = [
    {"kind": "home", "path": "/"},
    {"kind": "goods_list", "path": "/goods/list", "params": ["cate"]},
    {"kind": "goods_view", "path": "/goods/view", "params": ["goodsNo"]},
    {"kind": "search", "path": "/search", "params": ["q"]},
    {"kind": "cart", "path": "/order/cart"},
    {"kind": "checkout", "path": "/order/checkout"},
    {"kind": "order_complete", "path": "/order/complete", "params": ["orderNo"]},
    {"kind": "event_view", "path": "/event/view", "params": ["eventNo"]},
    {"kind": "mypage_orders", "path": "/mypage/orders", "params": ["orderNo"]},
    {"kind": "review_write", "path": "/mypage/review/write", "params": ["goodsNo", "orderNo"]},
]

UI = {
    "ko": {"cart": "장바구니 담기", "buy": "바로 구매", "wish": "찜하기", "coupon": "쿠폰 받기", "checkout": "주문하기",
           "keep": "쇼핑 계속하기", "pay": "결제하기", "detail": "주문 상세 보기", "color": "색상", "qty": "수량",
           "submit": "리뷰 등록", "search": "검색어 입력", "pay_label": "결제수단", "total": "총 결제금액", "rating": "별점",
           "review_ph": "상품은 어떠셨나요?", "event_more": "기획전 상품 더 보기"},
    "en": {"cart": "Add to cart", "buy": "Buy now", "wish": "Save to wishlist", "coupon": "Get coupon", "checkout": "Proceed to checkout",
           "keep": "Continue shopping", "pay": "Place order", "detail": "View order details", "color": "Colour", "qty": "Quantity",
           "submit": "Submit review", "search": "Search products", "pay_label": "Payment method", "total": "Order total",
           "rating": "Rating", "review_ph": "How was the product?", "event_more": "See more picks"},
    "ja": {"cart": "カートに入れる", "buy": "今すぐ購入", "wish": "お気に入り", "coupon": "クーポンを受け取る", "checkout": "レジに進む",
           "keep": "買い物を続ける", "pay": "注文を確定する", "detail": "注文詳細を見る", "color": "カラー", "qty": "数量",
           "submit": "レビューを投稿", "search": "商品を検索", "pay_label": "お支払い方法", "total": "お支払い合計", "rating": "評価",
           "review_ph": "商品はいかがでしたか？", "event_more": "特集商品をもっと見る"},
    "de": {"cart": "In den Warenkorb", "buy": "Jetzt kaufen", "wish": "Merken", "coupon": "Gutschein sichern", "checkout": "Zur Kasse",
           "keep": "Weiter einkaufen", "pay": "Zahlungspflichtig bestellen", "detail": "Bestelldetails ansehen", "color": "Farbe",
           "qty": "Menge", "submit": "Bewertung absenden", "search": "Artikel suchen", "pay_label": "Zahlungsart",
           "total": "Gesamtbetrag", "rating": "Bewertung", "review_ph": "Wie gefällt Ihnen der Artikel?", "event_more": "Mehr Aktionsartikel"},
}
TITLE = {
    "ko": {"home": "{site}", "goods_view": "{goods} | {site}", "search": "'{q}' 검색 결과 | {site}", "cart": "장바구니 | {site}",
           "checkout": "주문서 작성 | {site}", "order_complete": "주문 완료 | {site}", "event_view": "{event} | {site}",
           "review_write": "리뷰 쓰기 | {site}"},
    "en": {"home": "{site}", "goods_view": "{goods} | {site}", "search": "Results for '{q}' | {site}", "cart": "Your cart | {site}",
           "checkout": "Checkout | {site}", "order_complete": "Order confirmed | {site}", "event_view": "{event} | {site}",
           "review_write": "Write a review | {site}"},
    "ja": {"home": "{site}", "goods_view": "{goods} | {site}", "search": "「{q}」の検索結果 | {site}", "cart": "カート | {site}",
           "checkout": "ご注文手続き | {site}", "order_complete": "ご注文完了 | {site}", "event_view": "{event} | {site}",
           "review_write": "レビューを書く | {site}"},
    "de": {"home": "{site}", "goods_view": "{goods} | {site}", "search": "Ergebnisse für „{q}“ | {site}", "cart": "Warenkorb | {site}",
           "checkout": "Kasse | {site}", "order_complete": "Bestellung bestätigt | {site}", "event_view": "{event} | {site}",
           "review_write": "Bewertung schreiben | {site}"},
}
SEARCH_WORDS = {"ko": ["담요", "충전기", "텀블러"], "en": ["blanket", "charger", "tumbler"], "ja": ["ブランケット", "充電器", "タンブラー"],
                "de": ["Decke", "Ladegerät", "Thermobecher"]}


def visitor(seed: str) -> str:
    return "0x" + hashlib.sha256(seed.encode()).hexdigest()[:40]


def colors(sv: W.ShopV2) -> list:
    lang = sv.shop.lang
    return W.gcp.COLORS_KO if lang == "ko" else (W.gcp.COLORS_EN if lang == "en" else W.gcp.PACKS[lang]["colors"])


class Session:
    def __init__(self, sv: W.ShopV2, kind: str, sid: str, who: str, member, device: str, start: datetime):
        self.sv, self.kind, self.sid, self.who, self.member, self.device = sv, kind, sid, who, member, device
        self.t = start
        self.events = []

    def ev(self, page: str, path: str, title: str, action: str, target: str, relate: list, ev_type: str = "click",
           step_s: int = 40, **refs):
        if self.device == "mobile" and ev_type == "click":
            ev_type = "touch"
        self.t += timedelta(seconds=step_s)
        host = self.sv.shop.domain
        n = len(self.events) + 1
        self.events.append({
            "row": {
                "id": f"ev-{self.sid}-{n:02d}", "type": ev_type, "from": self.who, "host": host, "path": path,
                "created_at": int(self.t.timestamp() * 1000),
                "data": {"origin": f"https://{host}", "link": f"https://{host}{path}", "site": self.sv.shop.name, "title": title,
                         "device": self.device, "action": [target], "relate": relate},
            },
            "truth": {"id": f"ev-{self.sid}-{n:02d}", "shop": self.sv.shop.key, "session": self.sid, "session_kind": self.kind,
                      "type": ev_type, "page": page, "action": action, "device": self.device, "member": self.member,
                      "at": self.t.isoformat(timespec="seconds"), "path": path, **{k: v for k, v in refs.items() if v}},
        })


def button(cls: str, text: str, **attrs) -> str:
    extra = "".join(f' data-{k}="{E(str(v))}"' for k, v in attrs.items())
    return f'<button type="button" class="{cls}"{extra}>{E(text)}</button>'


def goods_page_events(s: Session, w: dict, g: dict, ui: dict, then_cart=True, coupon=None):
    sv = s.sv
    site = sv.shop.name
    path = f"/goods/view?goodsNo={g['id']}"
    title = TITLE[sv.shop.lang]["goods_view"].format(goods=g["title"], site=site)
    r = W.rng_for(s.sid, "goods", g["id"])
    cs = colors(sv)
    pick = r.choice(cs)
    opts = "".join(f"<option{' selected' if c == pick else ''}>{E(c)}</option>" for c in cs[:4] + ([pick] if pick not in cs[:4] else []))
    s.ev("goods_view", path, title, "option",
         f'<select name="opt_color" aria-label="{E(ui["color"])}">{opts}</select>',
         [f'<span class="goods-name">{E(g["title"])}</span>', f'<input type="number" name="qty" value="1" aria-label="{E(ui["qty"])}">'],
         ev_type="change", step_s=r.randint(20, 90), goods=g["id"])
    if coupon:
        s.ev("goods_view", path, title, "coupon", button("btn-coupon", ui["coupon"], cp=coupon["id"]),
             [f'<span class="coupon-name">{E(coupon["title"])}</span>'], step_s=r.randint(10, 40), goods=g["id"], coupon=coupon["id"])
    if then_cart:
        s.ev("goods_view", path, title, "add_to_cart", button("btn-cart", ui["cart"], goods=g["id"]),
             [button("btn-buy", ui["buy"]), button("btn-wish", ui["wish"]), f'<strong class="price">{E(W.gcp.fmt_money(g["sale_price"], sv.shop.money_style))}</strong>'],
             step_s=r.randint(15, 60), goods=g["id"])


def checkout_events(s: Session, ui: dict, total: int, pay_en: str, place=True, order=None):
    sv = s.sv
    lang = sv.shop.lang
    site = sv.shop.name
    s.ev("cart", "/order/cart", TITLE[lang]["cart"].format(site=site), "checkout", button("btn-order", ui["checkout"]),
         [button("btn-keep", ui["keep"])], step_s=35)
    methods = ["Credit card", "Wire transfer", "Digital wallet"]
    opts = "".join(f"<option{' selected' if m == pay_en else ''}>{E(W.label(sv, m))}</option>" for m in methods)
    s.ev("checkout", "/order/checkout", TITLE[lang]["checkout"].format(site=site), "payment_method",
         f'<select name="pay_method" aria-label="{E(ui["pay_label"])}">{opts}</select>',
         [f'<p class="total">{E(ui["total"])} <strong>{E(W.gcp.fmt_money(total, sv.shop.money_style))}</strong></p>'],
         ev_type="change", step_s=50)
    if place:
        s.ev("checkout", "/order/checkout", TITLE[lang]["checkout"].format(site=site), "pay", button("btn-pay", ui["pay"]),
             [f'<p class="total">{E(ui["total"])} <strong>{E(W.gcp.fmt_money(total, sv.shop.money_style))}</strong></p>'],
             step_s=25, order=order)


def build_sessions(sv: W.ShopV2, w: dict) -> list:
    lang = sv.shop.lang
    ui = UI[lang]
    site = sv.shop.name
    by_g = {g["id"]: g for g in w["goods"]}
    by_cp = {c["id"]: c for c in w["coupons"]}
    by_ev = {e["id"]: e for e in w["events"]}
    members = {m["id"]: m for m in w["members"]}
    r = W.rng_for(sv.shop.key, "weblog")
    sessions = []

    # 1) 구매 세션 7개: 이벤트 · 쿠폰 · 여러 상품 주문을 앞에 둡니다.
    paid = [o for o in w["orders"] if o["status"] not in ("Cancelled",)]
    paid.sort(key=lambda o: (-(o["event"] is not None) - (o["coupon"] is not None) - (len(o["lines"]) >= 2), o["seq"]))
    for k, o in enumerate(paid[:7]):
        device = "mobile" if k % 3 == 1 else "desktop"
        n_events_before = 6
        start = o["when"] - timedelta(minutes=n_events_before + r.randint(2, 9))
        s = Session(sv, "purchase", f"{sv.shop.key}-p{k + 1}", visitor(o["member"]), o["member"], device, start)
        coupon = by_cp.get(o["coupon"])
        if o["event"]:
            e = by_ev[o["event"]]
            g0 = by_g[next(l["goods"] for l in o["lines"] if l["goods"] in e["goods"])]
            others = [by_g[x] for x in e["goods"] if x != g0["id"]][:3]
            s.ev("event_view", f"/event/view?eventNo={e['id']}", TITLE[lang]["event_view"].format(event=e["title"], site=site),
                 "goods_link", f'<a href="/goods/view?goodsNo={g0["id"]}" class="pick">{E(g0["title"])}</a>',
                 [f'<a href="/goods/view?goodsNo={x["id"]}" class="pick">{E(x["title"])}</a>' for x in others] +
                 [f'<a class="more">{E(ui["event_more"])}</a>'], step_s=30, event=e["id"], goods=g0["id"])
        for i, line in enumerate(o["lines"]):
            g = by_g[line["goods"]]
            goods_page_events(s, w, g, ui, then_cart=True, coupon=coupon if (i == 0 and coupon) else None)
        checkout_events(s, ui, o["amount"], o["payment"], place=True, order=o["id"])
        s.ev("order_complete", f"/order/complete?orderNo={o['id']}", TITLE[lang]["order_complete"].format(site=site),
             "view_order", f'<a href="/mypage/orders?orderNo={o["id"]}">{E(ui["detail"])}</a>',
             [f'<p class="order-no">{E(o["id"])}</p>'], step_s=20, order=o["id"])
        sessions.append(s)

    # 2) 장바구니 이탈 3개: 결제 페이지에서 결제수단만 고르고 떠납니다.
    for k in range(3):
        m = r.choice(w["members"])
        g = r.choice(w["goods"])
        start = W.WINDOW[0] + timedelta(minutes=r.randint(0, 60 * 24 * 170))
        s = Session(sv, "abandon", f"{sv.shop.key}-a{k + 1}", visitor(m["id"]), m["id"], "mobile" if k == 1 else "desktop", start)
        goods_page_events(s, w, g, ui, then_cart=True)
        checkout_events(s, ui, g["sale_price"], r.choice(["Credit card", "Digital wallet"]), place=False)
        sessions.append(s)

    # 3) 리뷰 작성 2개
    for k, x in enumerate(w["reviews"][:2]):
        g = by_g[x["goods"]]
        posted = datetime(x["posted"].year, x["posted"].month, x["posted"].day, 20, r.randint(0, 59))
        s = Session(sv, "review", f"{sv.shop.key}-r{k + 1}", visitor(x["member"]), x["member"], "desktop", posted)
        path = f"/mypage/review/write?goodsNo={g['id']}&orderNo={x['order']}"
        title = TITLE[lang]["review_write"].format(site=site)
        stars = "★" * x["rating"] + "☆" * (5 - x["rating"])
        s.ev("review_write", path, title, "rating", f'<select name="rating" aria-label="{E(ui["rating"])}"><option selected>{stars}</option></select>',
             [f'<span class="goods-name">{E(g["title"])}</span>'], ev_type="change", step_s=30, goods=g["id"], order=x["order"])
        s.ev("review_write", path, title, "review_text", f'<textarea name="content" placeholder="{E(ui["review_ph"])}">{E(x["text"])}</textarea>',
             [f'<span class="goods-name">{E(g["title"])}</span>'], ev_type="change", step_s=60, goods=g["id"], order=x["order"])
        s.ev("review_write", path, title, "review_submit", button("btn-submit", ui["submit"]), [], step_s=15, goods=g["id"], order=x["order"])
        sessions.append(s)

    # 4) 검색 2개: 검색어를 넣고 결과에서 상품 하나를 엽니다(익명 방문자).
    for k in range(2):
        q = SEARCH_WORDS[lang][k]
        hit = next((g for g in w["goods"] if q.lower() in g["title"].lower()), r.choice(w["goods"]))
        start = W.WINDOW[0] + timedelta(minutes=r.randint(0, 60 * 24 * 170))
        s = Session(sv, "search", f"{sv.shop.key}-s{k + 1}", visitor(f"anon-{sv.shop.key}-{k}"), None, "mobile" if k else "desktop", start)
        s.ev("search", f"/search?q={q}", TITLE[lang]["search"].format(q=q, site=site), "search_input",
             f'<input type="search" name="q" value="{E(q)}" placeholder="{E(ui["search"])}">', [], ev_type="change", step_s=10)
        s.ev("search", f"/search?q={q}", TITLE[lang]["search"].format(q=q, site=site), "goods_link",
             f'<a href="/goods/view?goodsNo={hit["id"]}" class="result">{E(hit["title"])}</a>', [], step_s=15, goods=hit["id"])
        goods_page_events(s, w, hit, ui, then_cart=False)
        sessions.append(s)

    # 5) 이벤트 쿠폰만 받고 나가는 방문 1개 (여름 이벤트)
    e = w["events"][1]
    cp = by_cp[e["coupon"]]
    m = r.choice(w["members"])
    start = datetime(e["start"].year, e["start"].month, e["start"].day, 12, 0) + timedelta(days=3)
    s = Session(sv, "coupon", f"{sv.shop.key}-c1", visitor(m["id"]), m["id"], "mobile", start)
    s.ev("event_view", f"/event/view?eventNo={e['id']}", TITLE[lang]["event_view"].format(event=e["title"], site=site), "coupon",
         button("btn-coupon", ui["coupon"], cp=cp["id"]), [f'<span class="coupon-name">{E(cp["title"])}</span>'],
         step_s=20, event=e["id"], coupon=cp["id"])
    sessions.append(s)
    for s in sessions:
        if s.member:
            assert s.member in members
    return sessions


def site_def(sv: W.ShopV2) -> dict:
    return {"host": sv.shop.domain, "name": sv.shop.name, "lang": sv.shop.lang, "shop": sv.shop.key}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for sv in W.SHOPS:
        w = W.build_world(sv)
        ss = build_sessions(sv, w)
        evs = [e for s in ss for e in s.events]
        print(sv.shop.key, len(ss), "sessions", len(evs), "events", sorted({e["truth"]["action"] for e in evs}))
    print(evs[3]["row"])
