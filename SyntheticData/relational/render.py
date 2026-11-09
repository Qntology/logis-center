"""
커머스 페이지 (v2)
=================

world.py 의 세계를 관리자 화면(admin_flat)으로 렌더링합니다. 관계는 실제 관리자 화면처럼 링크로 드러납니다.

  주문 목록   주문번호 → 주문 보기(od_id), 대표 상품 → 상품 편집(it_id), 쿠폰 → 쿠폰 보기(cp_id), 운송장 → 배송 보기(invoice)
  주문 상세   품목 표의 상품마다 it_id 링크, 운송장마다 invoice 링크, 쿠폰 링크
  배송 목록   운송장 → 배송 보기(invoice), 주문번호 → 주문 보기(od_id), 상품 → 상품 편집(it_id)
  리뷰 목록   상품 → 상품 편집(it_id), 주문번호 → 주문 보기(od_id)
  쿠폰 목록   이벤트 쿠폰이면 이벤트 → 이벤트 보기(ev_id)
  이벤트 목록 · 상세   쿠폰 링크, 상세에는 기획 상품 표(상품마다 it_id)

링크 매개변수 이름은 앱의 relay_type_aliases 가 아는 이름(od_id · it_id · invoice · cp_id · ev_id)입니다.
정답 YAML 은 v1 과 같이 Col 의 truth_field 로 만들고, 관계 정답은 records.yaml 에 따로 둡니다.
"""
from datetime import datetime

import world as W

gcp = W.gcp
Col = gcp.Col
E = gcp.E

EXTRA = {  # v1 언어 팩에 없는 화면 문구
    "ja": {"Coupon": "クーポン", "Event": "イベント", "Delivered on": "配達日", "Featured products": "対象商品", "Return": "返品"},
    "de": {"Coupon": "Gutschein", "Event": "Aktion", "Delivered on": "Zugestellt am", "Featured products": "Aktionsartikel",
           "Return": "Rücksendung"},
}
KO = {"Coupon": "쿠폰", "Event": "이벤트", "Delivered on": "배송완료일", "Featured products": "기획 상품", "Return": "반품 회수"}


def T(sv: W.ShopV2, ko: str, en: str) -> str:
    lang = sv.shop.lang
    if lang == "ko":
        return ko
    if lang == "en":
        return en
    return EXTRA.get(lang, {}).get(en) or gcp.tr(sv.shop, en)


def money(sv: W.ShopV2, v: int) -> str:
    return gcp.fmt_money(v, sv.shop.money_style)


def mtruth(sv: W.ShopV2, v: int):
    return gcp.money_truth(v, sv.shop)


def dt(sv: W.ShopV2, d, with_time=False) -> str:
    return gcp.fmt_date(d, sv.shop.date_style, with_time)


def iso(d) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def U(path: str) -> str:
    return path


def goods_href(gid: str) -> str:
    return f"/admin/goods/edit?it_id={gid}"


def order_href(oid: str) -> str:
    return f"/admin/order/view?od_id={oid}"


def ship_href(no: str) -> str:
    return f"/admin/shipping/view?invoice={no}"


def coupon_href(cid: str) -> str:
    return f"/admin/coupon/view?cp_id={cid}"


def event_href(eid: str) -> str:
    return f"/admin/event/view?ev_id={eid}"


def goods_label(sv: W.ShopV2, w: dict, ids: list) -> tuple:
    by = {g["id"]: g for g in w["goods"]}
    first = by[ids[0]]["title"]
    extra = len(ids) - 1
    if not extra:
        return first, first
    lang = sv.shop.lang
    if lang == "ko":
        return f"{first} 외 {extra}건", first
    if lang == "en":
        return f"{first} + {extra} more", first
    return gcp.PACKS[lang]["more"].format(title=first, n=extra, total=extra + 1), first


# ──────────────────────────────────────────────
# 열 (목록 · 상세)
# ──────────────────────────────────────────────
def order_cols(sv, w, o, detail=False) -> list:
    by_cp = {c["id"]: c for c in w["coupons"]}
    label_text, main = goods_label(sv, w, [l["goods"] for l in o["lines"]])
    cols = [
        Col(T(sv, "주문번호", "Order #"), o["id"], "id,link", o["id"], href=order_href(o["id"])),
        Col(T(sv, "주문일시", "Ordered"), dt(sv, o["when"], True), "registration_date", iso(o["when"])),
        Col(T(sv, "주문상태", "Status"), W.label(sv, o["status"]), "status", W.label(sv, o["status"])),
        Col(T(sv, "주문상품", "Items"), label_text, "goods", main, href=goods_href(o["lines"][0]["goods"])),
        Col(T(sv, "주문자", "Customer"), o["buyer"]["name"], "sender_name", o["buyer"]["name"]),
        Col(T(sv, "수령인", "Ship to"), o["receiver"]["name"], "recipient_name", o["receiver"]["name"]),
        Col(T(sv, "수량", "Qty"), str(sum(l["qty"] for l in o["lines"])), "quantity", sum(l["qty"] for l in o["lines"])),
        Col(T(sv, "결제금액", "Total"), money(sv, o["amount"]), "amount", mtruth(sv, o["amount"])),
        Col(T(sv, "결제수단", "Payment"), W.label(sv, o["payment"]), "payment_method", W.label(sv, o["payment"])),
        Col(T(sv, "쿠폰할인", "Discount"), money(sv, o["discount"]), "", None),
    ]
    cp = by_cp.get(o["coupon"])
    cols.append(Col(T(sv, "쿠폰", "Coupon"), cp["title"] if cp else "-", "", None, href=coupon_href(cp["id"]) if cp else ""))
    outbound = [t for t in o["tracking"]]
    cols.insert(4, Col(T(sv, "운송장번호", "Tracking #"), outbound[0] if outbound else "-",
                       "tracking_number" if outbound else "", outbound[0] if outbound else None,
                       href=ship_href(outbound[0]) if outbound else ""))
    if not detail:
        return cols
    cols += [
        Col(T(sv, "주문자 연락처", "Customer phone"), o["buyer"]["phone"], "sender_phone", o["buyer"]["phone"]),
        Col(T(sv, "배송지", "Shipping address"), o["receiver"]["address"], "recipient_address", o["receiver"]["address"]),
        Col(T(sv, "수령인 연락처", "Recipient phone"), o["receiver"]["phone"], "recipient_phone", o["receiver"]["phone"]),
    ]
    return cols


def goods_cols(sv, w, g, detail=False) -> list:
    return [
        Col(T(sv, "상품번호", "Product ID"), g["id"], "id,link", g["id"], href=goods_href(g["id"])),
        Col(T(sv, "상품명", "Product"), g["title"], "title", g["title"]),
        Col(T(sv, "상태", "Status"), W.label(sv, g["status"]), "status", W.label(sv, g["status"])),
        Col(T(sv, "카테고리", "Category"), g["category"], "category", g["category"]),
        Col(T(sv, "판매가", "Price"), money(sv, g["sale_price"]), "sale_price", mtruth(sv, g["sale_price"])),
        Col(T(sv, "재고", "Stock"), str(g["stock"]), "quantity", g["stock"]),
        Col(T(sv, "등록일", "Created"), dt(sv, g["registered"]), "registration_date", iso(g["registered"])),
        Col(T(sv, "브랜드", "Brand"), g["brand"], "brand_name", g["brand"]),
    ]


def ship_cols(sv, w, s, detail=False) -> list:
    label_text, main = goods_label(sv, w, s["goods"])
    status = "Returned" if s["kind"] == "return" else s["status"]
    return [
        Col(T(sv, "운송장번호", "Tracking #"), s["id"], "id,link", s["id"], href=ship_href(s["id"])),
        Col(T(sv, "택배사", "Carrier"), s["carrier"], "carrier", s["carrier"]),
        Col(T(sv, "배송상태", "Status"), W.label(sv, status), "status", W.label(sv, status)),
        Col(T(sv, "상품", "Items"), label_text, "title", main, href=goods_href(s["goods"][0])),
        Col(T(sv, "발송일", "Shipped on"), dt(sv, s["shipped"]), "shipping_date", iso(s["shipped"])),
        Col(T(sv, "배송완료일", "Delivered on"), dt(sv, s["delivered"]) if s["delivered"] else "-", "", None),
        Col(T(sv, "받는분", "Recipient"), s["receiver"]["name"], "recipient_name", s["receiver"]["name"]),
        Col(T(sv, "주문번호", "Order #"), s["order"], "", None, href=order_href(s["order"])),
    ]


def review_cols(sv, w, x, detail=False) -> list:
    by = {g["id"]: g for g in w["goods"]}
    return [
        Col(T(sv, "번호", "ID"), x["id"], "id,link", x["id"], href=f"/admin/review/view?rv_id={x['id']}"),
        Col(T(sv, "상태", "Status"), W.label(sv, x["status"]), "status", W.label(sv, x["status"])),
        Col(T(sv, "내용", "Review"), x["text"], "title", x["text"]),
        Col(T(sv, "상품", "Product"), by[x["goods"]]["title"], "", None, href=goods_href(x["goods"])),
        Col(T(sv, "주문번호", "Order #"), x["order"], "", None, href=order_href(x["order"])),
        Col(T(sv, "작성자", "Author"), x["author"], "name", x["author"]),
        Col(T(sv, "평점", "Rating"), "★" * x["rating"] + "☆" * (5 - x["rating"]), "", None),
        Col(T(sv, "작성일", "Posted"), dt(sv, x["posted"]), "registration_date", iso(x["posted"])),
    ]


def coupon_cols(sv, w, c, detail=False) -> list:
    status = W.status_on(W.TODAY, c["start"], c["end"])
    value = f"{c['value']}%" if c["kind"] == "percent" else money(sv, c["value"])
    cols = [
        Col(T(sv, "쿠폰번호", "Discount ID"), c["id"], "id,link", c["id"], href=coupon_href(c["id"])),
        Col(T(sv, "코드", "Code"), c["code"], "code", c["code"]),
        Col(T(sv, "쿠폰명", "Name"), c["title"], "title", c["title"]),
        Col(T(sv, "종류", "Type"), T(sv, "정률" if c["kind"] == "percent" else "정액",
                                   "Percent off" if c["kind"] == "percent" else "Fixed amount"), "", None),
        Col(T(sv, "할인", "Value"), value, "discount", value),
        Col(T(sv, "시작일", "Starts"), dt(sv, c["start"]), "started_at", iso(c["start"])),
        Col(T(sv, "종료일", "Ends"), dt(sv, c["end"]), "expired_at", iso(c["end"])),
        Col(T(sv, "상태", "Status"), W.label(sv, status), "status", W.label(sv, status)),
        Col(T(sv, "최소주문금액", "Minimum spend"), money(sv, c["min"]), "min_order_amount", mtruth(sv, c["min"])),
    ]
    ev = next((e for e in w["events"] if e["id"] == c["event"]), None)
    cols.append(Col(T(sv, "이벤트", "Event"), ev["title"] if ev else "-", "", None, href=event_href(ev["id"]) if ev else ""))
    return cols


def event_cols(sv, w, e, detail=False) -> list:
    status = W.status_on(W.TODAY, e["start"], e["end"])
    cp = next((c for c in w["coupons"] if c["id"] == e["coupon"]), None)
    return [
        Col(T(sv, "번호", "ID"), e["id"], "id,link", e["id"], href=event_href(e["id"])),
        Col(T(sv, "기획전명", "Campaign"), e["title"], "title", e["title"]),
        Col(T(sv, "시작일", "Starts"), dt(sv, e["start"]), "started_at", iso(e["start"])),
        Col(T(sv, "종료일", "Ends"), dt(sv, e["end"]), "expired_at", iso(e["end"])),
        Col(T(sv, "상태", "Status"), W.label(sv, status), "status", W.label(sv, status)),
        Col(T(sv, "쿠폰", "Coupon"), cp["title"] if cp else "-", "", None, href=coupon_href(cp["id"]) if cp else ""),
    ]


# ──────────────────────────────────────────────
# 상세 페이지의 표 (주문 품목 · 운송장 · 기획 상품)
# ──────────────────────────────────────────────
def order_lines_table(sv, w, o) -> str:
    by = {g["id"]: g for g in w["goods"]}
    head = "".join(f"<th>{E(T(sv, a, b))}</th>" for a, b in (("상품번호", "Product ID"), ("상품", "Item"), ("수량", "Qty"),
                                                              ("단가", "Unit price"), ("소계", "Subtotal")))
    rows = "".join(
        f'<tr><td><a href="{E(goods_href(l["goods"]))}">{E(l["goods"])}</a></td><td>{E(by[l["goods"]]["title"])}</td>'
        f'<td>{l["qty"]}</td><td>{E(money(sv, l["unit"]))}</td><td>{E(money(sv, l["unit"] * l["qty"]))}</td></tr>'
        for l in o["lines"]
    )
    return (f'<div class="order-lines"><h3>{E(T(sv, "주문 상품", "Items ordered"))}</h3>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def order_ships_table(sv, w, o) -> str:
    ships = [s for s in w["shipments"] if s["order"] == o["id"]]
    if not ships:
        return ""
    by = {g["id"]: g for g in w["goods"]}
    head = "".join(f"<th>{E(T(sv, a, b))}</th>" for a, b in (("운송장번호", "Tracking #"), ("택배사", "Carrier"), ("상품", "Items"),
                                                              ("배송상태", "Status")))
    rows = "".join(
        f'<tr><td><a href="{E(ship_href(s["id"]))}">{E(s["id"])}</a></td><td>{E(s["carrier"])}</td>'
        f'<td>{E(", ".join(by[g]["title"] for g in s["goods"]))}</td>'
        f'<td>{E(W.label(sv, "Returned" if s["kind"] == "return" else s["status"]))}</td></tr>'
        for s in ships
    )
    return (f'<div class="order-shipments"><h3>{E(T(sv, "운송장번호", "Tracking #"))}</h3>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def event_goods_table(sv, w, e) -> str:
    by = {g["id"]: g for g in w["goods"]}
    head = "".join(f"<th>{E(T(sv, a, b))}</th>" for a, b in (("상품번호", "Product ID"), ("상품명", "Product"), ("판매가", "Price")))
    rows = "".join(
        f'<tr><td><a href="{E(goods_href(gid))}">{E(gid)}</a></td><td>{E(by[gid]["title"])}</td>'
        f'<td>{E(money(sv, by[gid]["sale_price"]))}</td></tr>'
        for gid in e["goods"]
    )
    return (f'<div class="event-goods"><h3>{E(T(sv, "기획 상품", "Featured products"))}</h3>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


# ──────────────────────────────────────────────
# 페이지 조립
# ──────────────────────────────────────────────
PTYPE_PATH = {"order": "order", "goods": "goods", "tracking": "shipping", "review": "review", "coupon": "coupon", "event": "event"}


def list_page(sv, w, ptype, rows_cols, page) -> tuple:
    labels = []
    for cols in rows_cols:
        for c in cols:
            if c.label not in labels:
                labels.append(c.label)
    aligned = [[{c.label: c for c in cols}.get(x, Col(x, "-")) for x in labels] for cols in rows_cols]
    title = gcp.page_title(sv.shop, ptype, "list")
    body = gcp.toolbar(sv.shop, ptype, len(rows_cols) * 3) + gcp.LIST_RENDERERS["admin_flat"](sv.shop, aligned, ptype) + gcp.pager(sv.shop, page)
    return gcp.chrome_for(sv.shop, ptype, body, title), title


def detail_page(sv, ptype, cols, extra_html: str) -> tuple:
    title = gcp.page_title(sv.shop, ptype, "detail")
    body = gcp.DETAIL_RENDERERS["admin_flat"](sv.shop, cols, ptype) + extra_html
    return gcp.chrome_for(sv.shop, ptype, body, title), title


def chunks(xs: list, n: int) -> list:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def pages_for(sv: W.ShopV2, w: dict):
    """(페이지 정보, HTML). 페이지 정보의 items/fields 는 열의 정답(truth_field)입니다."""
    base = f"https://{sv.shop.domain}"
    out = []

    def add_list(ptype, recs, cols_fn, per_page):
        for p, part in enumerate(chunks(recs, per_page), start=1):
            rows = [cols_fn(sv, w, r) for r in part]
            html, title = list_page(sv, w, ptype, rows, p)
            out.append(({"page_id": f"{sv.shop.key}-{ptype}-list-{p}", "url": f"{base}/admin/{PTYPE_PATH[ptype]}/list?page={p}",
                         "page_type": ptype, "layout": "list", "title": title,
                         "items": [gcp.truth_of(c) for c in rows]}, html))

    add_list("goods", w["goods"], goods_cols, 18)
    add_list("order", w["orders"], order_cols, 16)
    ships = sorted(w["shipments"], key=lambda s: s["shipped"], reverse=True)
    add_list("tracking", ships, ship_cols, (len(ships) + 1) // 2)
    add_list("review", w["reviews"], review_cols, 14)
    add_list("coupon", w["coupons"], coupon_cols, 10)
    add_list("event", w["events"], event_cols, 10)

    # 주문 상세 2장: 상품이 여럿이고 쿠폰을 썼으며 운송장이 둘 이상인 주문을 우선합니다.
    rich = sorted(w["orders"], key=lambda o: (-(len(o["lines"]) >= 2) - (o["coupon"] is not None) - (len(o["tracking"]) >= 2), o["seq"]))
    for o in rich[:2]:
        cols = order_cols(sv, w, o, detail=True)
        html, title = detail_page(sv, "order", cols, order_lines_table(sv, w, o) + order_ships_table(sv, w, o))
        out.append(({"page_id": f"{sv.shop.key}-order-detail-{o['id']}", "url": f"{base}/admin/order/view?od_id={o['id']}",
                     "page_type": "order", "layout": "detail", "title": title, "fields": gcp.truth_of(cols)}, html))
    e = w["events"][1]
    cols = event_cols(sv, w, e, detail=True)
    html, title = detail_page(sv, "event", cols, event_goods_table(sv, w, e))
    out.append(({"page_id": f"{sv.shop.key}-event-detail-{e['id']}", "url": f"{base}/admin/event/view?ev_id={e['id']}",
                 "page_type": "event", "layout": "detail", "title": title, "fields": gcp.truth_of(cols)}, html))
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for sv in W.SHOPS:
        w = W.build_world(sv)
        pages = pages_for(sv, w)
        print(sv.shop.key, len(pages), [p["page_id"] for p, _ in pages])
