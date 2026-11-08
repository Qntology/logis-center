"""
언어 팩: 한국어 · 영어 밖의 쇼핑몰 (generate_commerce_pages.py 의 분할 ml)
=======================================================================

언어마다 모듈 하나(de · es · fr · it · pt · nl · cs · ar · ja · zh_hans)가 PACK 을 둡니다.

  shop        쇼핑몰 (Shop 필드 + code_prefix). 이름 · 도메인 · 회사는 가상이고 도메인은 .test 만 씁니다
  tr          영어 화면 문구 · 열 이름 · 상태어 · 범주 → 그 언어 (생성기가 영어 문구로 찾음)
  chrome      관리자 · 구매자 화면 틀 (CHROME["en"] 과 같은 모양)
  products    PRODUCTS 와 같은 순서의 (이름, 설명) 60개
  origins · colors (5 · 7개, 영어 목록과 같은 순서) · carriers · banks · cards · pg (가상 이름)
  person      이름 · 주소 · 전화 만들기 (first · last · name · places · streets · address · phone)
              성이 성별로 갈리면 first_f(여성 이름) · last_f(last 와 같은 자리의 여성형)를 둡니다
  more · days 대표 상품 '외 N건' ({n} 나머지 수, {total} 전체 수) · 소요일 표기
              (days 는 {1: "1 Tag", "other": "{n} Tage"} 처럼 수마다 다른 꼴도 됩니다)
  review_tmpl (%s 하나) · review_adj · coupon_names (6) · event_names (6)
  concepts    의미 검색 질의 (개념 10개) · queries (식별자 · 상태 · 이름 · 금액 · 기간 · 상품) · months (12)

상표 · 실제 회사 이름을 쓰지 않습니다. bias.json 앵커 예시에 있는 브랜드 · 택배사 · 카드사 이름도 쓰지 않습니다.
"""
from . import ar, cs, de, es, fr, it, ja, nl, pt, zh_hans

# PRODUCTS (generate_commerce_pages.py) 의 영어 이름, 같은 순서. 생성기가 같은지 확인합니다.
PRODUCT_NAMES_EN = [
    "Fleece-lined zip hoodie", "Lightweight padded vest", "Sherpa fleece jacket", "Microfiber throw blanket",
    "Rechargeable hand warmer", "Plush house slippers", "Linen short-sleeve shirt", "Cooling neck band",
    "Handheld mini fan", "Ice gel cooling mat", "Ventilated sandals", "UV-blocking parasol",
    "Monitor riser stand", "Cable management box", "Ergonomic vertical mouse", "Silent mechanical keyboard",
    "Laptop stand", "Document file organizer", "Stainless steel frying pan", "Silicone utensil set",
    "Ceramic chef knife", "Temperature control kettle", "Airtight glass container set", "Immersion hand blender",
    "Wooden building blocks", "Toddler tableware set", "Kids kick scooter", "Colouring sketchbook",
    "Plush bunny doll", "Coding robot kit", "Dog harness", "Cat scratching post",
    "Automatic pet feeder", "Pet travel carrier", "Hypoallergenic kibble", "Pet grooming brush",
    "Lightweight camp chair", "Two-person dome tent", "Rechargeable LED lantern", "Portable cookware set",
    "Vacuum insulated tumbler", "Folding trekking poles", "Barrier moisture cream", "Vitamin C serum",
    "Low-pH cleansing foam", "Mineral sunscreen", "Repair hair essence", "Fragrance-free body lotion",
    "Roasted almonds", "Organic granola", "Cold brew concentrate", "Low-sugar protein bar",
    "Jeju tangerine juice", "Scorched brown rice crisps", "Noise-cancelling wireless earbuds", "Waterproof Bluetooth speaker",
    "High-capacity power bank", "65W fast charger", "Smart fitness band", "USB-C multiport hub",
]
PRODUCT_KEYS = {name: i for i, name in enumerate(PRODUCT_NAMES_EN)}

# 생성기가 영어 문구로 번역을 찾는 낱말 (열 이름 · 화면 제목 · 상태어 · 범주 · 값)
TR_KEYS = [
    # 열 이름
    "Order #", "Ordered", "Status", "Items", "Customer", "Ship to", "Qty", "Total", "Shipping", "Payment", "Rewards",
    "Discount", "Tracking #", "Customer phone", "Billing address", "Recipient phone", "Shipping address", "Payment gateway",
    "Paid at", "Bank", "Card", "Product ID", "Product", "Category", "Price", "Stock", "Created", "Brand", "List price",
    "Waived over threshold", "Internal code", "SKU", "Model", "Manufacturer", "Country of origin", "Color", "Barcode",
    "Unit cost", "Low stock alert", "Description", "Carrier", "Shipped on", "Recipient", "Registered", "Sender",
    "Sender address", "Sender phone", "Recipient address", "Weight", "Width", "Height", "Length", "Method",
    "Transit time", "Freight charge", "ID", "Review", "Author", "Rating", "Posted", "Discount ID", "Code", "Name",
    "Type", "Value", "Starts", "Ends", "Issued", "Minimum spend", "Campaign", "Item", "Unit price", "Subtotal",
    "Items ordered",
    # 화면 제목
    "Orders", "Order details", "Products", "Product details", "Shipments", "Shipment details", "Product reviews",
    "Discount codes", "Discount code", "Campaigns",
    # 상태어
    "All", "Awaiting payment", "Paid", "Processing", "Shipped", "Delivered", "Cancelled", "Returned", "Active",
    "Sold out", "Draft", "Approved", "Pending", "Hidden", "Live", "Scheduled", "Ended",
    # 값
    "New", "Credit card", "Wire transfer", "Digital wallet", "Standard", "Freight", "Store pickup", "Percent off",
    "Fixed amount", "Showcase", "Sweepstakes", "Tester",
    # 범주
    "Apparel", "Home", "Appliances", "Office", "Electronics", "Kitchen", "Toys", "Baby", "Pets", "Outdoor", "Beauty",
    "Grocery",
]
STATUS_EN = ["Awaiting payment", "Paid", "Processing", "Shipped", "Delivered", "Cancelled", "Returned", "Active",
             "Sold out", "Draft", "Approved", "Pending", "Hidden", "Live", "Scheduled", "Ended"]
CHROME_LABELS = ["copyright", "owner", "biz", "cs", "addr", "banner", "search", "reset", "excel", "edit", "view", "total",
                 "rows", "prev", "next", "recent", "chat", "top"]
CONCEPTS = ["cold_weather", "hot_weather", "desk_work", "cooking", "kids", "pet", "outdoor", "skincare", "snack_food",
            "audio_power"]
QUERY_KEYS = ["order", "tracking", "goods", "status", "person", "amount", "period", "goods_in_order"]
SHOP_FIELDS = ["key", "name", "domain", "lang", "theme", "split", "company", "company_address", "company_phone",
               "currency", "date_style", "money_style", "code_prefix"]


def _check(lang: str, pk: dict) -> None:
    """빠진 번역 · 개수 · 형식 문자열을 생성 전에 잡습니다."""
    missing = [k for k in TR_KEYS if k not in pk["tr"]]
    assert not missing, f"{lang}: tr 에 없는 낱말 {missing}"
    statuses = [pk["tr"][k] for k in STATUS_EN]
    assert len(set(statuses)) == len(statuses), f"{lang}: 상태어가 겹칩니다 {statuses}"
    assert len(pk["products"]) == len(PRODUCT_NAMES_EN), f"{lang}: 상품 {len(pk['products'])}개"
    assert all(len(p) == 2 and p[0] and p[1] for p in pk["products"]), f"{lang}: 상품은 (이름, 설명)"
    assert len({p[0] for p in pk["products"]}) == len(pk["products"]), f"{lang}: 상품 이름이 겹칩니다"
    assert len(pk["origins"]) == 5 and len(pk["colors"]) == 7, f"{lang}: 원산지 5 · 색 7"
    for key, n in (("carriers", 2), ("banks", 2), ("cards", 2), ("pg", 2)):
        assert len(pk[key]) >= n, f"{lang}: {key}"
    assert len(pk["coupon_names"]) == 6 and len(pk["event_names"]) == 6, f"{lang}: 쿠폰 · 이벤트 이름 6개"
    assert all(t.count("%s") == 1 for t in pk["review_tmpl"]), f"{lang}: 리뷰 틀에 %s 하나"
    assert [c for c in CONCEPTS if c not in pk["concepts"]] == [], f"{lang}: 개념 질의"
    assert [q for q in QUERY_KEYS if q not in pk["queries"]] == [], f"{lang}: 질의 틀"
    assert len(pk["months"]) == 12, f"{lang}: 달 이름 12개"
    ch = pk["chrome"]
    assert len(ch["nav"]) == 9 and len(ch["foot"]) == 5, f"{lang}: chrome nav 9 · foot 5"
    assert set(ch["side"]) == {"order", "goods", "tracking", "review", "coupon", "event"}, f"{lang}: chrome side"
    assert [k for k in CHROME_LABELS if k not in ch] == [], f"{lang}: chrome 문구"
    assert [k for k in SHOP_FIELDS if k not in pk["shop"]] == [], f"{lang}: shop 필드"
    assert pk["shop"]["lang"] == lang and pk["shop"]["domain"].endswith(".test"), f"{lang}: shop"
    pp = pk["person"]
    assert pp["first"] and pp["last"] and pp["places"] and "#" in pp["phone"], f"{lang}: person"
    if "first_f" in pp:
        assert set(pp["first_f"]) <= set(pp["first"]) and len(pp["last_f"]) == len(pp["last"]), f"{lang}: 성별 성씨"
    days = pk["days"] if isinstance(pk["days"], dict) else {"other": pk["days"]}
    more = pk["more"]
    assert "{n}" in days["other"] and "{title}" in more and ("{n}" in more or "{total}" in more), f"{lang}: more · days"


PACKS = {}
for _mod in (de, es, fr, it, pt, nl, cs, ar, ja, zh_hans):
    _lang = _mod.PACK["shop"]["lang"]
    _check(_lang, _mod.PACK)
    PACKS[_lang] = _mod.PACK
