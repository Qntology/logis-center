"""
커머스 합성 웹페이지 생성기
==========================

시드로 고정한 가상의 쇼핑몰 세계(상품 · 고객 · 주문 · 배송 · 리뷰 · 쿠폰 · 이벤트)를 만들고,
그 세계를 서로 구조가 다른 웹페이지로 렌더링합니다.

  input_yamls/<page_id>.yaml   페이지별 정답 (필드 이름은 bias.json 의 스키마 이름)
  output_html/<page_id>.html   브라우저가 받을 법한 렌더 결과
  queries.yaml                 자연어 검색 평가 질의와 정답 레코드

무역 문서 쪽(SyntheticData/trading)과 같은 짝을 이룹니다: YAML 이 정답이고 렌더 결과가 입력입니다.

만들 때 지킨 것
  · 앵커 어휘를 베끼지 않습니다. bias.json 예시에 있는 브랜드 · 택배사 · 카드사 이름은 쓰지 않습니다.
    앵커 예시가 그대로 페이지에 나오면 임베딩이 맞힌 것인지 문자열이 맞은 것인지 구분되지 않습니다.
  · 실제 회사 · 상표를 쓰지 않습니다. 도메인은 예약 TLD(.test)만 씁니다.
  · dev / test 분할은 쇼핑몰과 테마가 겹치지 않습니다. test 는 튜닝에 쓰지 않습니다.
  · 실제 페이지에 있는 함정을 넣습니다: 상태 필터 탭, 푸터 회사 주소, 추천 상품 위젯, 주문 상세의 상품 표,
    두 줄짜리 행, 라벨 없는 카드, 해시 CSS 클래스, input 안의 값, <br> 로 끊긴 주소, 적립금 · 할인 같은 스키마 밖 숫자 열.

다국어 (lang_packs)
  · 한국어 · 영어 쇼핑몰 6곳(dev · test) 밖에 독일어 · 스페인어 · 프랑스어 · 이탈리아어 · 포르투갈어 · 네덜란드어 · 체코어 ·
    아랍어 · 일본어 · 중국어 쇼핑몰을 하나씩 둡니다 (분할 ml). 화면 문구 · 상태어 · 상품명 · 사람 · 주소 · 금액 · 날짜 표기 ·
    질의를 lang_packs/<언어>.py 에서 가져오고, 한국어 · 영어 쇼핑몰의 출력은 바뀌지 않습니다 (--check).

사용법
  python generate_commerce_pages.py            # 전체 생성
  python generate_commerce_pages.py --check    # 다시 생성해 파일이 같은지만 확인
"""
import argparse
import hashlib
import html
import random
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from lang_packs import PACKS, PRODUCT_KEYS

ROOT = Path(__file__).resolve().parent
SEED = 20261103


# ──────────────────────────────────────────────
# 1. 상품 사전 (개념 키는 의미 검색 정답에 씁니다)
# ──────────────────────────────────────────────
# (한국어 이름, 영어 이름, 개념 키, 범주, 한국어 설명, 영어 설명, 기준가(원))
PRODUCTS = [
    ("기모 후드집업", "Fleece-lined zip hoodie", "cold_weather", "의류", "안감을 도톰하게 기모 처리한 집업입니다. 소매 시보리가 손목을 감쌉니다.", "A zip hoodie with a brushed inner lining and ribbed cuffs.", 49000),
    ("경량 패딩 조끼", "Lightweight padded vest", "cold_weather", "의류", "거위털을 얇게 충전해 겉옷 안에 겹쳐 입을 수 있습니다.", "Thin goose-down fill that layers under a coat.", 69000),
    ("양털 플리스 자켓", "Sherpa fleece jacket", "cold_weather", "의류", "뽀글이 원단으로 만든 자켓으로 몸의 열을 잡아 줍니다.", "A sherpa jacket that holds body heat.", 59000),
    ("극세사 담요", "Microfiber throw blanket", "cold_weather", "생활", "촘촘한 극세사 원단이라 덮으면 금방 포근해집니다.", "Dense microfiber that feels cosy within minutes.", 25000),
    ("충전식 손난로", "Rechargeable hand warmer", "cold_weather", "생활", "주머니에 넣고 쓰는 충전식 발열 제품입니다.", "A pocket-sized rechargeable heating pad.", 22000),
    ("털 슬리퍼", "Plush house slippers", "cold_weather", "생활", "발등까지 털로 덮인 실내용 슬리퍼입니다.", "Indoor slippers lined with plush up to the instep.", 18000),
    ("리넨 반팔 셔츠", "Linen short-sleeve shirt", "hot_weather", "의류", "통기성이 좋은 리넨 원단으로 땀이 나도 달라붙지 않습니다.", "Breathable linen that does not cling when you sweat.", 45000),
    ("냉감 넥밴드", "Cooling neck band", "hot_weather", "생활", "물에 적시면 체온을 낮춰 주는 목걸이형 밴드입니다.", "A wet-activated band that lowers neck temperature.", 12000),
    ("휴대용 미니 선풍기", "Handheld mini fan", "hot_weather", "가전", "손에 들고 쓰는 3단 풍량 선풍기입니다.", "A handheld fan with three speed settings.", 19000),
    ("아이스 쿨매트", "Ice gel cooling mat", "hot_weather", "생활", "젤이 열을 흡수해 누우면 시원합니다.", "Gel that absorbs heat when you lie on it.", 39000),
    ("통풍 샌들", "Ventilated sandals", "hot_weather", "의류", "구멍 뚫린 갑피로 발이 답답하지 않습니다.", "Perforated uppers keep your feet from feeling stuffy.", 33000),
    ("자외선 차단 양산", "UV-blocking parasol", "hot_weather", "생활", "뙤약볕을 막아 주는 경량 우양산입니다.", "A light parasol that shuts out harsh sunshine.", 28000),
    ("모니터 받침대", "Monitor riser stand", "desk_work", "사무", "화면 높이를 눈높이로 올리고 아래 공간에 키보드를 넣습니다.", "Lifts the screen to eye level with keyboard space beneath.", 35000),
    ("케이블 정리함", "Cable management box", "desk_work", "사무", "멀티탭과 전선을 상자 안에 숨겨 줍니다.", "Hides power strips and wires inside a box.", 16000),
    ("인체공학 버티컬 마우스", "Ergonomic vertical mouse", "desk_work", "디지털", "손목을 세운 자세로 쥐어 피로를 줄입니다.", "A handshake grip that eases wrist strain.", 42000),
    ("저소음 기계식 키보드", "Silent mechanical keyboard", "desk_work", "디지털", "타건음이 작은 스위치를 넣은 87키 키보드입니다.", "An 87-key board with quiet switches.", 89000),
    ("노트북 거치대", "Laptop stand", "desk_work", "사무", "알루미늄 재질로 각도를 여섯 단계로 조절합니다.", "An aluminium stand with six angle positions.", 31000),
    ("서류 정리함", "Document file organizer", "desk_work", "사무", "A4 서류를 칸별로 세워 보관합니다.", "Keeps A4 papers upright in labelled slots.", 14000),
    ("스테인리스 프라이팬", "Stainless steel frying pan", "cooking", "주방", "삼중 바닥이라 열이 고르게 퍼집니다.", "A tri-ply base spreads heat evenly.", 54000),
    ("실리콘 조리도구 세트", "Silicone utensil set", "cooking", "주방", "뒤집개와 국자 등 다섯 종으로 구성했습니다.", "Five pieces including a turner and a ladle.", 27000),
    ("세라믹 식칼", "Ceramic chef knife", "cooking", "주방", "녹슬지 않는 세라믹 날로 채소를 썰기 좋습니다.", "A rust-free ceramic blade ideal for slicing vegetables.", 36000),
    ("온도조절 전기포트", "Temperature control kettle", "cooking", "가전", "물을 원하는 온도로 끓여 보온합니다.", "Boils water to a chosen temperature and holds it.", 48000),
    ("밀폐 유리 용기 세트", "Airtight glass container set", "cooking", "주방", "남은 반찬을 냄새 없이 보관합니다.", "Stores leftovers without odours.", 32000),
    ("핸드 블렌더", "Immersion hand blender", "cooking", "가전", "냄비 안에서 바로 수프를 갈 수 있습니다.", "Purees soup right in the pot.", 44000),
    ("원목 블록 세트", "Wooden building blocks", "kids", "완구", "모서리를 둥글게 다듬은 백 피스 블록입니다.", "One hundred blocks with rounded corners.", 38000),
    ("유아 식기 세트", "Toddler tableware set", "kids", "유아", "흡착 바닥이라 잘 뒤집히지 않습니다.", "Suction bases keep plates from tipping over.", 26000),
    ("어린이 킥보드", "Kids kick scooter", "kids", "완구", "세 바퀴로 균형을 잡기 쉽습니다.", "Three wheels make it easy to balance.", 65000),
    ("색칠놀이 스케치북", "Colouring sketchbook", "kids", "완구", "동물 그림 마흔 장이 들어 있습니다.", "Forty animal outlines to fill in.", 9000),
    ("토끼 봉제 인형", "Plush bunny doll", "kids", "완구", "부드러운 원단으로 만든 애착 인형입니다.", "A soft comfort toy for bedtime.", 21000),
    ("코딩 로봇 키트", "Coding robot kit", "kids", "완구", "카드를 순서대로 놓아 로봇을 움직입니다.", "Line up cards to steer the robot.", 79000),
    ("강아지 하네스", "Dog harness", "pet", "반려", "가슴줄이 넓어 목에 부담이 없습니다.", "A wide chest strap that spares the neck.", 24000),
    ("고양이 스크래처", "Cat scratching post", "pet", "반려", "골판지로 만든 원통형 스크래처입니다.", "A cylindrical post made of corrugated card.", 19000),
    ("자동 급식기", "Automatic pet feeder", "pet", "반려", "정해진 시간에 사료를 내어 줍니다.", "Dispenses kibble at set times.", 72000),
    ("반려동물 이동장", "Pet travel carrier", "pet", "반려", "병원에 갈 때 쓰는 통기성 가방입니다.", "A breathable bag for vet visits.", 46000),
    ("저알러지 사료", "Hypoallergenic kibble", "pet", "반려", "단일 단백질로 만들어 피부가 예민한 아이에게 맞습니다.", "Single-protein food for sensitive skin.", 34000),
    ("펫 빗 브러시", "Pet grooming brush", "pet", "반려", "죽은 털을 빗어 내는 브러시입니다.", "Combs out loose fur.", 13000),
    ("경량 캠핑 의자", "Lightweight camp chair", "outdoor", "레저", "접으면 물병만 한 크기가 됩니다.", "Folds down to the size of a water bottle.", 37000),
    ("2인용 돔 텐트", "Two-person dome tent", "outdoor", "레저", "방수 코팅된 플라이를 씌운 텐트입니다.", "A tent with a waterproof-coated fly.", 129000),
    ("LED 충전 랜턴", "Rechargeable LED lantern", "outdoor", "레저", "밝기를 네 단계로 조절합니다.", "Adjustable across four brightness levels.", 29000),
    ("휴대용 코펠 세트", "Portable cookware set", "outdoor", "레저", "냄비와 팬을 겹쳐 수납합니다.", "A pot and pan that nest together.", 41000),
    ("진공 보온 텀블러", "Vacuum insulated tumbler", "outdoor", "주방", "이중 진공 벽으로 온도를 오래 지킵니다.", "A double wall keeps drinks at temperature for hours.", 23000),
    ("접이식 등산 스틱", "Folding trekking poles", "outdoor", "레저", "세 단으로 접혀 배낭에 들어갑니다.", "Three-section poles that fit in a backpack.", 52000),
    ("수분 장벽 크림", "Barrier moisture cream", "skincare", "뷰티", "세라마이드를 넣어 건조한 피부에 씁니다.", "Ceramide-rich cream for dry skin.", 32000),
    ("비타민 C 세럼", "Vitamin C serum", "skincare", "뷰티", "칙칙한 피부 톤을 정돈해 줍니다.", "Evens out a dull complexion.", 38000),
    ("약산성 클렌징 폼", "Low-pH cleansing foam", "skincare", "뷰티", "세안 후 당기지 않는 거품 세안제입니다.", "A foaming wash that leaves no tightness.", 15000),
    ("무기자차 선크림", "Mineral sunscreen", "skincare", "뷰티", "SPF50 의 무기 필터 제품입니다.", "SPF 50 with mineral filters.", 24000),
    ("손상모 헤어 에센스", "Repair hair essence", "skincare", "뷰티", "상한 머릿결을 부드럽게 정돈합니다.", "Smooths damaged strands.", 19000),
    ("무향 바디 로션", "Fragrance-free body lotion", "skincare", "뷰티", "향료 없이 보습만 담았습니다.", "Pure hydration without added scent.", 17000),
    ("구운 아몬드", "Roasted almonds", "snack_food", "식품", "기름 없이 볶아 고소합니다.", "Dry-roasted for a nutty flavour.", 12000),
    ("유기농 그래놀라", "Organic granola", "snack_food", "식품", "귀리와 꿀을 구워 만든 시리얼입니다.", "Baked oats and honey cereal.", 14000),
    ("콜드브루 원액", "Cold brew concentrate", "snack_food", "식품", "물에 희석해 마시는 커피 원액입니다.", "Coffee concentrate to dilute with water.", 16000),
    ("저당 단백질바", "Low-sugar protein bar", "snack_food", "식품", "한 개에 단백질 15그램이 들어 있습니다.", "Fifteen grams of protein per bar.", 22000),
    ("제주 감귤 주스", "Jeju tangerine juice", "snack_food", "식품", "감귤을 착즙해 병에 담았습니다.", "Cold-pressed tangerines, bottled.", 18000),
    ("현미 누룽지", "Scorched brown rice crisps", "snack_food", "식품", "끓는 물을 부어 먹는 간편식입니다.", "A quick meal: just add hot water.", 11000),
    ("노이즈캔슬링 무선 이어폰", "Noise-cancelling wireless earbuds", "audio_power", "디지털", "주변 소음을 줄여 주는 블루투스 이어폰입니다.", "Bluetooth earbuds that hush outside sound.", 119000),
    ("방수 블루투스 스피커", "Waterproof Bluetooth speaker", "audio_power", "디지털", "물가에서도 쓸 수 있는 소형 스피커입니다.", "A compact speaker safe near water.", 58000),
    ("대용량 보조배터리", "High-capacity power bank", "audio_power", "디지털", "20,000mAh 용량으로 휴대폰을 네 번 충전합니다.", "20,000 mAh, enough for four phone charges.", 39000),
    ("65W 고속 충전기", "65W fast charger", "audio_power", "디지털", "노트북까지 충전되는 GaN 충전기입니다.", "A GaN charger strong enough for laptops.", 34000),
    ("스마트 밴드", "Smart fitness band", "audio_power", "디지털", "걸음 수와 수면을 기록합니다.", "Tracks steps and sleep.", 49000),
    ("USB-C 멀티 허브", "USB-C multiport hub", "audio_power", "디지털", "HDMI 와 USB 포트를 늘려 줍니다.", "Adds HDMI and USB ports.", 37000),
]

assert [p[1] for p in PRODUCTS] == list(PRODUCT_KEYS), "lang_packs.PRODUCT_NAMES_EN 이 PRODUCTS 와 다릅니다"

CATEGORY_EN = {
    "의류": "Apparel", "생활": "Home", "가전": "Appliances", "사무": "Office", "디지털": "Electronics",
    "주방": "Kitchen", "완구": "Toys", "유아": "Baby", "반려": "Pets", "레저": "Outdoor",
    "뷰티": "Beauty", "식품": "Grocery",
}

# 앵커 예시와 겹치지 않는 가상 이름
BRANDS = ["누리온", "하람", "솔빛", "다온랩", "마루공방", "버들", "Kestrel Lane", "Oakfern", "Tidewell", "Marlow & Finch"]
MAKERS = ["누리온산업", "하람제작소", "솔빛리빙", "다온랩스", "Oakfern Works", "Tidewell Supply"]
ORIGINS_KO = ["국산", "말레이시아", "태국", "폴란드", "포르투갈"]
ORIGINS_EN = ["Korea", "Malaysia", "Thailand", "Poland", "Portugal"]
COLORS_KO = ["먹색", "연회색", "크림", "올리브", "버건디", "하늘색", "모래색"]
COLORS_EN = ["Charcoal", "Light grey", "Cream", "Olive", "Burgundy", "Sky blue", "Sand"]
CARRIERS_KO = ["바른택배", "한빛로지스", "새길운송", "온누리익스프레스"]
CARRIERS_EN = ["Swiftline Parcel", "BlueRoute Freight", "Harbor Courier", "Pinewood Express"]
BANKS_KO = ["가온은행", "누리은행", "한결은행"]
BANKS_EN = ["Harbor Savings Bank", "First Lantern Bank", "Pinecrest Credit Union"]
CARDS_KO = ["가온카드", "누리카드", "한결카드"]
CARDS_EN = ["Lantern Card", "Harbor Visa", "Pinecrest Rewards"]
PG_KO = ["바로결제", "온페이먼츠", "스마트PG"]
PG_EN = ["PayRelay", "CheckoutBridge", "SwiftTender"]
SURNAMES_KO = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권"]
GIVEN_KO = ["서연", "민준", "지우", "도윤", "하은", "시우", "수아", "예준", "유진", "주원", "다은", "건우", "소율", "현우", "채원", "은호"]
FIRST_EN = ["Avery", "Jordan", "Riley", "Morgan", "Casey", "Taylor", "Quinn", "Rowan", "Harper", "Emerson", "Finley", "Sawyer"]
LAST_EN = ["Whitfield", "Marsh", "Holloway", "Sinclair", "Prescott", "Langley", "Dorsey", "Kendrick", "Ashby", "Mercer"]
ROADS_KO = [
    ("서울특별시", "마포구", "합정로"), ("서울특별시", "성동구", "왕십리로"), ("부산광역시", "수영구", "광남로"),
    ("대전광역시", "유성구", "대학로"), ("광주광역시", "서구", "상무대로"), ("경기도 수원시", "영통구", "매탄로"),
    ("경기도 고양시", "일산동구", "중앙로"), ("강원특별자치도 춘천시", "", "공지로"), ("전북특별자치도 전주시", "완산구", "효자로"),
]
CITIES_EN = [
    ("Portland", "OR", "97205"), ("Madison", "WI", "53703"), ("Burlington", "VT", "05401"),
    ("Asheville", "NC", "28801"), ("Boise", "ID", "83702"), ("Santa Fe", "NM", "87501"),
]
STREETS_EN = ["Alder St", "Birch Ave", "Cedar Ln", "Maple Dr", "Willow Ct", "Juniper Rd"]


# ──────────────────────────────────────────────
# 2. 쇼핑몰 · 테마 · 분할
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Shop:
    key: str
    name: str
    domain: str
    lang: str
    theme: str
    split: str
    company: str
    company_address: str
    company_phone: str
    currency: str
    # 날짜 · 금액 표기는 쇼핑몰마다 다릅니다.
    date_style: str
    money_style: str


SHOPS = [
    Shop("haneul", "하늘상점", "haneul-market.test", "ko", "admin_twin", "dev", "(주)하늘커머스",
         "서울특별시 금천구 가산디지털2로 77 9층", "1800-0119", "KRW", "iso_time", "won_suffix"),
    Shop("moa", "모아몰", "moa-mall.test", "ko", "admin_grid", "dev", "모아리테일 주식회사",
         "경기도 성남시 수정구 창업로 42", "1811-0345", "KRW", "dot_short", "plain_comma"),
    Shop("soop", "숲속잡화", "soop-shop.test", "ko", "store_cards", "dev", "숲속잡화 협동조합",
         "제주특별자치도 제주시 첨단로 213", "1877-0221", "KRW", "korean_long", "won_suffix"),
    Shop("harbor", "Harbor & Pine", "harborandpine.test", "en", "admin_flat", "dev", "Harbor & Pine Trading LLC",
         "410 Foundry Way, Tacoma, WA 98402", "+1-253-555-0142", "USD", "us_slash", "dollar"),
    # ── test: 튜닝에 쓰지 않습니다. 쇼핑몰도 테마도 dev 와 겹치지 않습니다.
    Shop("ttukbaegi", "뚝배기마켓", "ttukbaegi.test", "ko", "mobile_dl", "test", "뚝배기유통 주식회사",
         "대구광역시 동구 동대구로 465", "1833-0763", "KRW", "dash_short", "won_prefix"),
    Shop("lantern", "Blue Lantern Supply", "bluelantern-supply.test", "en", "store_sections", "test", "Blue Lantern Supply Co.",
         "88 Quarry Road, Duluth, MN 55802", "+1-218-555-0177", "USD", "us_long", "dollar"),
]

# ── ml: 언어별 쇼핑몰 (lang_packs). dev · test 수치에 섞이지 않게 분할을 따로 둡니다.
SHOPS += [Shop(**{k: v for k, v in pk["shop"].items() if k != "code_prefix"}) for pk in PACKS.values()]
LOCAL_SHOP = {"ko", "en"}

THEME_DESC = {
    "admin_twin": "관리자 표. 레코드 하나가 두 개의 <tr> 로 나뉩니다.",
    "admin_grid": "ARIA 역할과 해시 CSS 클래스만 있는 div 그리드.",
    "store_cards": "구매자 화면의 카드. 대부분의 값에 라벨이 없습니다.",
    "admin_flat": "한 행에 한 레코드인 평범한 관리자 표.",
    "mobile_dl": "모바일 웹뷰의 dl/dt/dd 레이아웃.",
    "store_sections": "구매자 화면의 section/h3/strong 레이아웃.",
}


def rng_for(*parts) -> random.Random:
    """이름으로 파생한 난수기. 한 항목을 바꿔도 다른 항목의 값이 밀리지 않습니다."""
    digest = hashlib.sha256("|".join([str(SEED)] + [str(p) for p in parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ──────────────────────────────────────────────
# 3. 표기 함수
# ──────────────────────────────────────────────
def fmt_date(d, style: str, with_time: bool = False) -> str:
    dt = d if isinstance(d, datetime) else datetime(d.year, d.month, d.day)
    t = dt.strftime(" %H:%M") if with_time else ""
    if style == "iso_time":
        return dt.strftime("%Y-%m-%d") + t
    if style == "dot_short":
        return dt.strftime("%y.%m.%d") + t
    if style == "korean_long":
        return f"{dt.year}년 {dt.month}월 {dt.day}일" + t
    if style == "us_slash":
        return dt.strftime("%m/%d/%Y") + t
    if style == "dash_short":
        return dt.strftime("%y-%m-%d") + t
    if style == "us_long":
        return dt.strftime("%b ") + f"{dt.day}, {dt.year}" + t
    if style == "dmy_dot":
        return dt.strftime("%d.%m.%Y") + t
    if style == "dmy_slash":
        return dt.strftime("%d/%m/%Y") + t
    if style == "dmy_dash":
        return dt.strftime("%d-%m-%Y") + t
    if style == "cs_dot":
        return f"{dt.day}. {dt.month}. {dt.year}" + t
    if style == "ymd_slash":
        return dt.strftime("%Y/%m/%d") + t
    if style == "ja_long":
        return f"{dt.year}年{dt.month}月{dt.day}日" + t
    if style == "iso":
        return dt.strftime("%Y-%m-%d") + t
    raise ValueError(style)


def _eu(amount_minor: int) -> str:
    """1.234,56 (유럽식 천 단위 · 소수점)"""
    return f"{amount_minor / 100:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_money(amount, style: str) -> str:
    if style == "won_suffix":
        return f"{amount:,}원"
    if style == "plain_comma":
        return f"{amount:,}"
    if style == "won_prefix":
        return f"₩{amount:,}"
    if style == "dollar":
        return f"${amount / 100:,.2f}"
    if style == "eur_suffix":
        return f"{_eu(amount)} €"
    if style == "eur_fr":
        return f"{_eu(amount).replace('.', ' ')} €"
    if style == "eur_prefix":
        return f"€ {_eu(amount)}"
    if style == "czk":
        return f"{amount:,} Kč".replace(",", " ")
    if style == "sar":
        return f"{amount / 100:,.2f} ر.س"
    if style == "yen":
        return f"¥{amount:,}"
    if style == "yuan":
        return f"¥{amount / 100:,.2f}"
    raise ValueError(style)


# 원화 기준가를 쇼핑몰 통화로: (1 단위 = 원, 보조 단위 배수). USD · KRW 는 위 식 그대로입니다.
CURRENCY = {"EUR": (1450, 100), "JPY": (9.2, 1), "CNY": (190, 100), "CZK": (60, 1), "SAR": (365, 100)}


def price_of(base_won: int, shop: Shop) -> int:
    """KRW 는 원, USD 는 센트 단위 정수로 저장합니다. 다른 통화는 CURRENCY (EUR · CNY · SAR 은 보조 단위, JPY · CZK 는 1 단위)."""
    if shop.currency in CURRENCY:
        rate, minor = CURRENCY[shop.currency]
        return int(round(base_won / rate * 100 / 5)) * 5 if minor == 100 else int(round(base_won / rate / 10)) * 10
    return int(round(base_won / 1300 * 100 / 5)) * 5 if shop.currency == "USD" else base_won


def money_truth(amount: int, shop: Shop):
    if shop.currency in CURRENCY:
        return round(amount / 100, 2) if CURRENCY[shop.currency][1] == 100 else amount
    return round(amount / 100, 2) if shop.currency == "USD" else amount


# ──────────────────────────────────────────────
# 4. 세계 생성
# ──────────────────────────────────────────────
def pack(shop: Shop) -> dict:
    return PACKS[shop.lang]


def tr(shop: Shop, en: str) -> str:
    """영어 화면 문구 · 상태어 → 언어 팩 번역 (한국어 · 영어 쇼핑몰에서는 부르지 않습니다)."""
    return pack(shop)["tr"][en]


def pack_person(shop: Shop, r: random.Random) -> dict:
    pp = pack(shop)["person"]
    first, last = r.choice(pp["first"]), r.choice(pp["last"])
    if first in pp.get("first_f", ()):
        last = pp["last_f"][pp["last"].index(last)]
    place = r.choice(pp["places"])
    street = r.choice(pp["streets"]) if pp["streets"] else ""
    addr = pp["address"].format(street=street, num=r.randint(1, 180), a=r.randint(1, 9), b=r.randint(1, 30),
                                c=r.randint(1, 20), **place)
    phone = "".join(str(r.randint(0, 9)) if ch == "#" else ch for ch in pp["phone"])
    return {"name": pp["name"].format(first=first, last=last), "address": addr, "phone": phone}


def person(shop: Shop, r: random.Random) -> dict:
    if shop.lang not in LOCAL_SHOP:
        return pack_person(shop, r)
    if shop.lang == "ko":
        si, gu, road = r.choice(ROADS_KO)
        name = r.choice(SURNAMES_KO) + r.choice(GIVEN_KO)
        addr = " ".join(x for x in [si, gu, road, f"{r.randint(3, 480)}"] if x)
        if r.random() < 0.6:
            addr += f" {r.randint(1, 18)}층"
        phone = f"010-{r.randint(2000, 9899)}-{r.randint(1000, 9999)}"
    else:
        city, st, zip_code = r.choice(CITIES_EN)
        name = f"{r.choice(FIRST_EN)} {r.choice(LAST_EN)}"
        addr = f"{r.randint(101, 9800)} {r.choice(STREETS_EN)}, {city}, {st} {zip_code}"
        phone = f"+1-{r.randint(201, 989)}-555-01{r.randint(0, 99):02d}"
    return {"name": name, "address": addr, "phone": phone}


def build_goods(shop: Shop) -> list:
    r = rng_for(shop.key, "goods")
    picked = r.sample(PRODUCTS, 42)
    goods = []
    for i, p in enumerate(picked, start=1):
        name_ko, name_en, concept, cat_ko, desc_ko, desc_en, base = p
        gr = rng_for(shop.key, "goods", i)
        ko = shop.lang == "ko"
        reg = date(2026, 1, 5) + timedelta(days=gr.randint(0, 250))
        price = price_of(int(round(base * gr.uniform(0.92, 1.12) / 100)) * 100, shop)
        compare_at = price + (price // 5 if gr.random() < 0.45 else 0)
        stock = gr.choice([0, 0, 3, 7, 18, 42, 96, 120, 350])
        code_prefix = {"haneul": "HN", "moa": "MA", "soop": "SP", "harbor": "HP", "ttukbaegi": "TB", "lantern": "BL"}.get(shop.key) \
            or pack(shop)["shop"]["code_prefix"]
        loc = None if shop.lang in LOCAL_SHOP else pack(shop)
        local = loc["products"][PRODUCT_KEYS[name_en]] if loc else None
        status_ko = "품절" if stock == 0 else gr.choice(["판매중", "판매중", "판매중", "진열안함"])
        status_en = "Sold out" if stock == 0 else gr.choice(["Active", "Active", "Active", "Draft"])
        goods.append({
            "seq": i,
            "id": f"{code_prefix}{10000 + i * 37}",
            "code": f"{code_prefix}-{concept[:3].upper()}-{100 + i}",
            "status": status_ko if ko else (tr(shop, status_en) if loc else status_en),
            "title": name_ko if ko else (local[0] if loc else name_en),
            "title_alt": name_en if ko or loc else name_ko,
            "concept": concept,
            "category": cat_ko if ko else (tr(shop, CATEGORY_EN[cat_ko]) if loc else CATEGORY_EN[cat_ko]),
            "registration_date": reg,
            "brand_name": gr.choice(BRANDS),
            "manufacturer": gr.choice(MAKERS),
            "origin_country": (ORIGINS_KO if ko else (loc["origins"] if loc else ORIGINS_EN))[gr.randrange(5)],
            "color": (COLORS_KO if ko else (loc["colors"] if loc else COLORS_EN))[gr.randrange(7)],
            "sale_price": price,
            "supply_price": int(price * gr.uniform(0.55, 0.72)),
            "compare_at_price": compare_at,
            "quantity": stock,
            "low_stock_threshold": gr.choice([5, 10, 20]),
            "stock_keeping_unit": f"{code_prefix}{concept[:2].upper()}{1000 + i}",
            "barcode": f"88{gr.randint(10 ** 10, 10 ** 11 - 1)}",
            "model_name": f"{concept[:2].upper()}-{gr.randint(100, 999)}",
            "description": desc_ko if ko else (local[1] if loc else desc_en),
            "shipping_fee": 0 if price >= price_of(50000, shop) else price_of(3000, shop),
            "condition": "새상품" if ko else (tr(shop, "New") if loc else "New"),
        })
    return goods


def build_orders(shop: Shop, goods: list) -> list:
    orders = []
    ko = shop.lang == "ko"
    statuses = (
        ["입금대기", "결제완료", "상품준비중", "배송중", "배송중", "배송완료", "배송완료", "배송완료", "취소", "반품"]
        if ko else
        ["Awaiting payment", "Paid", "Processing", "Shipped", "Shipped", "Delivered", "Delivered", "Delivered", "Cancelled", "Returned"]
    )
    loc = None if shop.lang in LOCAL_SHOP else pack(shop)
    carriers = CARRIERS_KO if ko else (loc["carriers"] if loc else CARRIERS_EN)
    for i in range(1, 49):
        r = rng_for(shop.key, "order", i)
        when = datetime(2026, 4, 1, 8, 0) + timedelta(minutes=r.randint(0, 60 * 24 * 195))
        buyer = person(shop, r)
        receiver = person(shop, r) if r.random() < 0.35 else dict(buyer)
        lines = []
        for g in r.sample(goods, r.choice([1, 1, 1, 2, 2, 3])):
            lines.append({"goods": g, "qty": r.choice([1, 1, 1, 2, 2, 3, 5])})
        subtotal = sum(l["goods"]["sale_price"] * l["qty"] for l in lines)
        ship_fee = 0 if subtotal >= price_of(50000, shop) else price_of(3000, shop)
        discount = (subtotal // 10) if r.random() < 0.3 else 0
        status = r.choice(statuses)
        method = r.choice(["card", "card", "bank", "easy"])
        if ko:
            method_label = {"card": "신용카드", "bank": "무통장입금", "easy": "간편결제"}[method]
        else:
            method_label = {"card": "Credit card", "bank": "Wire transfer", "easy": "Digital wallet"}[method]
        pre_ship = status in ("입금대기", "결제완료", "상품준비중", "취소", "Awaiting payment", "Paid", "Processing", "Cancelled")
        tracking = None if pre_ship else f"{r.randint(10 ** 11, 10 ** 12 - 1)}"
        ship_at = None if pre_ship else when + timedelta(days=r.randint(1, 3), hours=r.randint(1, 9))
        title = lines[0]["goods"]["title"]
        extra = len(lines) - 1
        if extra:
            if loc:
                title_disp = loc["more"].format(title=title, n=extra, total=extra + 1)
            else:
                title_disp = f"{title} 외 {extra}건" if ko else f"{title} + {extra} more"
        else:
            title_disp = title
        orders.append({
            "seq": i,
            "id": f"{when:%Y%m%d}-{100000 + i * 53}",
            "status": tr(shop, status) if loc else status,
            "when": when,
            "paid_at": None if status in ("입금대기", "Awaiting payment") else when + timedelta(minutes=r.randint(1, 240)),
            "buyer": buyer,
            "receiver": receiver,
            "lines": lines,
            "goods_label": title_disp,
            "goods_main": title,
            "quantity": sum(l["qty"] for l in lines),
            "subtotal": subtotal,
            "discount": discount,
            "points": subtotal // 100,
            "shipping_fee": ship_fee,
            "amount": subtotal - discount + ship_fee,
            "method": method,
            "payment_method": tr(shop, method_label) if loc else method_label,
            "bank": r.choice(BANKS_KO if ko else (loc["banks"] if loc else BANKS_EN)) if method == "bank" else None,
            "card": r.choice(CARDS_KO if ko else (loc["cards"] if loc else CARDS_EN)) if method == "card" else None,
            "payment_origin": r.choice(PG_KO if ko else (loc["pg"] if loc else PG_EN)),
            "tracking": tracking,
            "carrier": r.choice(carriers) if tracking else None,
            "ship_at": ship_at,
            "weight": round(r.uniform(0.3, 7.5), 1),
            "dims": (r.randint(18, 60), r.randint(10, 45), r.randint(6, 40)),
            "shipping_method": (
                r.choice(["일반택배", "화물배송", "방문수령"]) if ko else
                (tr(shop, r.choice(["Standard", "Freight", "Store pickup"])) if loc else r.choice(["Standard", "Freight", "Store pickup"]))
            ),
            "duration_days": r.randint(1, 5),
        })
    return orders


def build_reviews(shop: Shop, goods: list) -> list:
    ko = shop.lang == "ko"
    tmpl_ko = ["생각보다 %s 만족합니다.", "두 번째 구매인데 역시 %s 좋네요.", "포장이 꼼꼼했고 %s 괜찮습니다."]
    tmpl_en = ["Better than expected, really %s.", "Second purchase and still %s.", "Packed with care and %s overall."]
    adj_ko = ["튼튼해서", "가벼워서", "마감이 좋아서", "색이 예뻐서"]
    adj_en = ["sturdy", "lightweight", "well finished", "lovely in colour"]
    loc = None if shop.lang in LOCAL_SHOP else pack(shop)
    if loc:
        tmpl_en, adj_en = loc["review_tmpl"], loc["review_adj"]
    out = []
    for i in range(1, 21):
        r = rng_for(shop.key, "review", i)
        g = r.choice(goods)
        out.append({
            "seq": i,
            "id": f"RV{7000 + i * 11}",
            "status": (lambda s: tr(shop, s) if loc else s)(r.choice(["승인", "승인", "대기중", "숨김"] if ko else ["Approved", "Approved", "Pending", "Hidden"])),
            "title": (r.choice(tmpl_ko) % r.choice(adj_ko)) if ko else (r.choice(tmpl_en) % r.choice(adj_en)),
            "goods": g["title"],
            "name": person(shop, r)["name"],
            "rating": r.randint(3, 5),
            "registration_date": date(2026, 5, 1) + timedelta(days=r.randint(0, 150)),
        })
    return out


def build_coupons(shop: Shop) -> list:
    ko = shop.lang == "ko"
    names_ko = ["첫 구매 환영 혜택", "가을 맞이 감사제", "주말 한정 할인", "재구매 고객 감사", "앱 전용 혜택", "장바구니 응원"]
    names_en = ["First order welcome", "Autumn thank-you sale", "Weekend-only offer", "Returning customer perk", "App-exclusive deal", "Cart rescue offer"]
    loc = None if shop.lang in LOCAL_SHOP else pack(shop)
    if loc:
        names_en = loc["coupon_names"]
    out = []
    for i in range(1, 11):
        r = rng_for(shop.key, "coupon", i)
        start = date(2026, 3, 1) + timedelta(days=r.randint(0, 200))
        pct = r.random() < 0.5
        out.append({
            "seq": i,
            "id": f"CP{4000 + i * 7}",
            "code": f"{shop.key[:3].upper()}{r.randint(100, 999)}",
            "title": r.choice(names_ko if ko else names_en) + f" {i}" + ("차" if ko else ""),
            "type": ("정률" if pct else "정액") if ko else (lambda s: tr(shop, s) if loc else s)("Percent off" if pct else "Fixed amount"),
            "discount": f"{r.choice([5, 10, 15, 20])}%" if pct else fmt_money(price_of(r.choice([2000, 3000, 5000]), shop), shop.money_style),
            "started_at": start,
            "expired_at": start + timedelta(days=r.choice([7, 14, 30])),
            "status": (lambda s: tr(shop, s) if loc else s)(r.choice(["진행중", "종료", "대기중"] if ko else ["Live", "Ended", "Scheduled"])),
            "quantity": r.choice([100, 300, 500, 1000]),
            "min_order_amount": price_of(r.choice([10000, 30000, 50000]), shop),
        })
    return out


def build_events(shop: Shop) -> list:
    ko = shop.lang == "ko"
    names_ko = ["봄 산책 기획전", "여름 휴가 준비전", "추석 선물 모음", "연말 감사 축제", "신상품 체험단 모집", "리뷰 왕 선발"]
    names_en = ["Spring stroll showcase", "Summer getaway picks", "Harvest gift guide", "Year-end celebration", "New arrivals tester program", "Top reviewer contest"]
    loc = None if shop.lang in LOCAL_SHOP else pack(shop)
    if loc:
        names_en = loc["event_names"]
    out = []
    for i in range(1, 7):
        r = rng_for(shop.key, "event", i)
        start = date(2026, 2, 1) + timedelta(days=r.randint(0, 240))
        out.append({
            "seq": i,
            "id": f"EV{300 + i * 3}",
            "title": (names_ko if ko else names_en)[i - 1],
            "type": (lambda s: tr(shop, s) if loc else s)(r.choice(["기획전", "응모", "체험단"] if ko else ["Showcase", "Sweepstakes", "Tester"])),
            "started_at": start,
            "expired_at": start + timedelta(days=r.choice([10, 20, 45])),
            "status": (lambda s: tr(shop, s) if loc else s)(r.choice(["진행중", "종료", "대기중"] if ko else ["Live", "Ended", "Scheduled"])),
        })
    return out


def build_world(shop: Shop) -> dict:
    goods = build_goods(shop)
    return {
        "goods": goods,
        "orders": build_orders(shop, goods),
        "reviews": build_reviews(shop, goods),
        "coupons": build_coupons(shop),
        "events": build_events(shop),
    }


# ──────────────────────────────────────────────
# 5. 페이지 공통 조각 (함정 포함)
# ──────────────────────────────────────────────
E = html.escape

CHROME = {
    "ko": {
        "nav": ["대시보드", "주문관리", "상품관리", "배송관리", "고객관리", "프로모션", "게시판", "통계", "환경설정"],
        "side": {
            "order": ["전체주문", "입금전관리", "배송준비중관리", "취소/환불관리", "주문서출력"],
            "goods": ["상품목록", "상품등록", "분류관리", "재고관리", "브랜드관리"],
            "tracking": ["배송현황", "송장출력", "일괄발송", "반품관리", "택배사설정"],
            "review": ["상품후기", "상품문의", "신고관리", "포인트설정"],
            "coupon": ["쿠폰목록", "쿠폰생성", "발급내역", "적립금관리"],
            "event": ["기획전목록", "기획전등록", "배너관리", "팝업관리"],
        },
        "foot": ["회사소개", "이용약관", "개인정보처리방침", "입점문의", "고객센터"],
        "copyright": "All rights reserved.",
        "owner": "대표자", "biz": "사업자등록번호", "cs": "고객센터", "addr": "주소",
        "banner": "추석 연휴 기간에는 출고가 지연될 수 있습니다.",
        "search": "검색", "reset": "초기화", "excel": "엑셀 다운로드", "edit": "수정", "view": "보기",
        "total": "총", "rows": "건", "prev": "이전", "next": "다음", "recent": "함께 본 상품",
        "chat": "채팅 상담", "top": "맨 위로",
    },
    "en": {
        "nav": ["Dashboard", "Orders", "Products", "Fulfillment", "Customers", "Marketing", "Content", "Analytics", "Settings"],
        "side": {
            "order": ["All orders", "Unpaid", "Ready to ship", "Cancellations", "Print packing slips"],
            "goods": ["All products", "Add product", "Collections", "Inventory", "Vendors"],
            "tracking": ["Shipments", "Print labels", "Bulk dispatch", "Returns", "Carrier settings"],
            "review": ["Product reviews", "Questions", "Reports", "Reward settings"],
            "coupon": ["Discount codes", "Create code", "Redemptions", "Store credit"],
            "event": ["Campaigns", "New campaign", "Banners", "Popups"],
        },
        "foot": ["About us", "Terms of service", "Privacy policy", "Wholesale", "Help center"],
        "copyright": "All rights reserved.",
        "owner": "CEO", "biz": "Business ID", "cs": "Support", "addr": "Address",
        "banner": "Dispatch may be delayed over the holiday weekend.",
        "search": "Search", "reset": "Reset", "excel": "Export CSV", "edit": "Edit", "view": "View",
        "total": "Total", "rows": "records", "prev": "Previous", "next": "Next", "recent": "Customers also viewed",
        "chat": "Live chat", "top": "Back to top",
    },
}

PAGE_TITLE = {
    ("order", "list", "ko"): "주문내역 관리", ("order", "detail", "ko"): "주문 상세내역",
    ("goods", "list", "ko"): "상품 관리", ("goods", "detail", "ko"): "상품 정보",
    ("tracking", "list", "ko"): "배송 조회", ("tracking", "detail", "ko"): "배송 상세 정보",
    ("review", "list", "ko"): "상품후기 관리", ("review", "detail", "ko"): "후기 보기",
    ("coupon", "list", "ko"): "쿠폰 관리", ("coupon", "detail", "ko"): "쿠폰 정보",
    ("event", "list", "ko"): "기획전 관리", ("event", "detail", "ko"): "기획전 정보",
    ("order", "list", "en"): "Orders", ("order", "detail", "en"): "Order details",
    ("goods", "list", "en"): "Products", ("goods", "detail", "en"): "Product details",
    ("tracking", "list", "en"): "Shipments", ("tracking", "detail", "en"): "Shipment details",
    ("review", "list", "en"): "Product reviews", ("review", "detail", "en"): "Review",
    ("coupon", "list", "en"): "Discount codes", ("coupon", "detail", "en"): "Discount code",
    ("event", "list", "en"): "Campaigns", ("event", "detail", "en"): "Campaign",
}

STATUS_TABS = {
    "order": {"ko": ["전체", "입금대기", "결제완료", "상품준비중", "배송중", "배송완료", "취소", "반품"],
              "en": ["All", "Awaiting payment", "Paid", "Processing", "Shipped", "Delivered", "Cancelled", "Returned"]},
    "goods": {"ko": ["전체", "판매중", "품절", "진열안함"], "en": ["All", "Active", "Sold out", "Draft"]},
    "tracking": {"ko": ["전체", "배송중", "배송완료", "반품"], "en": ["All", "Shipped", "Delivered", "Returned"]},
    "review": {"ko": ["전체", "승인", "대기중", "숨김"], "en": ["All", "Approved", "Pending", "Hidden"]},
    "coupon": {"ko": ["전체", "진행중", "대기중", "종료"], "en": ["All", "Live", "Scheduled", "Ended"]},
    "event": {"ko": ["전체", "진행중", "대기중", "종료"], "en": ["All", "Live", "Scheduled", "Ended"]},
}


def chrome(shop: Shop) -> dict:
    return CHROME.get(shop.lang) or pack(shop)["chrome"]


def page_title(shop: Shop, ptype: str, layout: str) -> str:
    if shop.lang in LOCAL_SHOP:
        return PAGE_TITLE[(ptype, layout, shop.lang)]
    return tr(shop, PAGE_TITLE[(ptype, layout, "en")])


def status_tabs(shop: Shop, ptype: str) -> list:
    if shop.lang in LOCAL_SHOP:
        return STATUS_TABS[ptype].get(shop.lang)
    return [tr(shop, x) for x in STATUS_TABS[ptype]["en"]]


def css_hash(*parts) -> str:
    return "css-" + hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:7]


def rtl_attr(shop: Shop) -> str:
    return ' dir="rtl"' if shop.lang == "ar" else ""


def admin_chrome(shop: Shop, ptype: str, body: str, title: str) -> str:
    c = chrome(shop)
    nav = "".join(f'<li class="gnb-item"><a href="/admin/{i}">{E(n)}</a></li>' for i, n in enumerate(c["nav"]))
    side = "".join(f'<li><a href="/admin/{ptype}/{i}">{E(n)}</a></li>' for i, n in enumerate(c["side"][ptype]))
    return (
        f'<!doctype html><html lang="{shop.lang}"{rtl_attr(shop)}><head><meta charset="utf-8"><title>{E(title)} | {E(shop.name)}</title></head>'
        f'<body class="admin"><header id="hd"><h1 class="logo"><a href="/admin">{E(shop.name)} ADMIN</a></h1>'
        f'<nav id="gnb"><ul>{nav}</ul></nav></header>'
        f'<div id="wrapper"><aside id="snb"><h2>{E(c["nav"][1])}</h2><ul>{side}</ul></aside>'
        f'<main id="container"><p class="notice">{E(c["banner"])}</p>'
        f'<div class="breadcrumb">{E(c["nav"][0])} &gt; {E(title)}</div><h2 class="page-title">{E(title)}</h2>'
        f'{body}</main></div>'
        f'<footer id="ft"><p>{E(shop.company)} · {E(c["addr"])} {E(shop.company_address)} · {E(c["cs"])} {E(shop.company_phone)}</p>'
        f'<p>{E(c["copyright"])}</p></footer></body></html>'
    )


def store_chrome(shop: Shop, ptype: str, body: str, title: str) -> str:
    c = chrome(shop)
    links = "".join(f'<a href="/p/{i}">{E(n)}</a>' for i, n in enumerate(c["foot"]))
    return (
        f'<!doctype html><html lang="{shop.lang}"{rtl_attr(shop)}><head><meta charset="utf-8"><title>{E(title)} - {E(shop.name)}</title></head>'
        f'<body><div class="top-banner">{E(c["banner"])}</div>'
        f'<header class="site-header"><a class="brand" href="/">{E(shop.name)}</a>'
        f'<form class="site-search" action="/search"><input name="q" placeholder="{E(c["search"])}"></form>'
        f'<a href="/cart">🛒</a><a href="/my">My</a></header>'
        f'<main class="page"><h1>{E(title)}</h1>{body}</main>'
        f'<aside class="floating"><button>{E(c["chat"])}</button><button>{E(c["top"])}</button></aside>'
        f'<footer class="site-footer"><nav>{links}</nav>'
        f'<address>{E(shop.company)} · {E(c["owner"])} · {E(c["biz"])} 123-45-67890 · {E(c["cs"])} {E(shop.company_phone)} · {E(shop.company_address)}</address>'
        f'</footer></body></html>'
    )


def chrome_for(shop: Shop, ptype: str, body: str, title: str) -> str:
    if shop.theme.startswith("admin"):
        return admin_chrome(shop, ptype, body, title)
    return store_chrome(shop, ptype, body, title)


def toolbar(shop: Shop, ptype: str, count: int) -> str:
    """상태 필터 탭과 건수. 상태 어휘가 표 밖에도 잔뜩 나타나는 실제 페이지의 함정입니다."""
    c = chrome(shop)
    tabs = "".join(
        f'<a class="tab" href="?status={i}">{E(t)}<em>({(count * (9 - i)) // 7})</em></a>'
        for i, t in enumerate(status_tabs(shop, ptype))
    )
    opts = "".join(f"<option>{E(t)}</option>" for t in status_tabs(shop, ptype))
    return (
        f'<div class="status-tabs">{tabs}</div>'
        f'<form class="filter"><select name="status">{opts}</select><input name="q">'
        f'<button>{E(c["search"])}</button><button type="reset">{E(c["reset"])}</button></form>'
        f'<div class="list-head"><span>{E(c["total"])} {count} {E(c["rows"])}</span><button>{E(c["excel"])}</button></div>'
    )


def pager(shop: Shop, page: int) -> str:
    c = chrome(shop)
    nums = "".join(f'<a href="?page={i}"{" class=on" if i == page else ""}>{i}</a>' for i in range(1, 6))
    return f'<nav class="pager"><a href="?page={max(1, page - 1)}">{E(c["prev"])}</a>{nums}<a href="?page={page + 1}">{E(c["next"])}</a></nav>'


# ──────────────────────────────────────────────
# 6. 레코드 → 표시값 / 정답
# ──────────────────────────────────────────────
@dataclass
class Col:
    """목록의 열 또는 상세의 항목 하나."""
    label: str
    text: str
    truth_field: str = ""       # 빈 문자열이면 스키마 밖 값(방해 요소)
    truth_value: object = None
    href: str = ""
    kind: str = "text"          # 입력 요소 렌더링 힌트: text / select / textarea / image
    options: list = field(default_factory=list)


def L(shop: Shop, ko: str, en: str) -> str:
    if shop.lang in LOCAL_SHOP:
        return ko if shop.lang == "ko" else en
    return tr(shop, en)


def days_text(shop: Shop, n: int) -> str:
    if shop.lang in LOCAL_SHOP:
        return L(shop, f"{n}일", f"{n} days")
    days = pack(shop)["days"]
    if isinstance(days, dict):
        days = days.get(n, days["other"])
    return days.format(n=n)


def iso(d) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def order_cols(shop: Shop, o: dict, detail: bool) -> list:
    ds, ms = shop.date_style, shop.money_style
    money = lambda v: fmt_money(v, ms)
    link = f"/admin/order/view?od_id={o['id']}" if shop.theme.startswith("admin") else f"/my/orders/{o['id']}"
    cols = [
        Col(L(shop, "주문번호", "Order #"), o["id"], "id,link", o["id"], href=link),
        Col(L(shop, "주문일시", "Ordered"), fmt_date(o["when"], ds, True), "registration_date", iso(o["when"])),
        Col(L(shop, "주문상태", "Status"), o["status"], "status", o["status"]),
        Col(L(shop, "주문상품", "Items"), o["goods_label"], "goods", o["goods_main"]),
        Col(L(shop, "주문자", "Customer"), o["buyer"]["name"], "sender_name", o["buyer"]["name"]),
        Col(L(shop, "수령인", "Ship to"), o["receiver"]["name"], "recipient_name", o["receiver"]["name"]),
        Col(L(shop, "수량", "Qty"), str(o["quantity"]), "quantity", o["quantity"]),
        Col(L(shop, "결제금액", "Total"), money(o["amount"]), "amount", money_truth(o["amount"], shop)),
        Col(L(shop, "배송비", "Shipping"), money(o["shipping_fee"]), "shipping_fee", money_truth(o["shipping_fee"], shop)),
        Col(L(shop, "결제수단", "Payment"), o["payment_method"], "payment_method", o["payment_method"]),
        # 스키마 밖 숫자 열: 금액 필드와 헷갈리는 자리입니다.
        Col(L(shop, "적립금", "Rewards"), money(o["points"]), "", None),
        Col(L(shop, "쿠폰할인", "Discount"), money(o["discount"]), "", None),
    ]
    if o["tracking"]:
        cols.insert(4, Col(L(shop, "운송장번호", "Tracking #"), o["tracking"], "tracking_number", o["tracking"]))
    if not detail:
        return cols
    cols += [
        Col(L(shop, "주문자 연락처", "Customer phone"), o["buyer"]["phone"], "sender_phone", o["buyer"]["phone"]),
        Col(L(shop, "주문자 주소", "Billing address"), o["buyer"]["address"], "sender_address", o["buyer"]["address"]),
        Col(L(shop, "수령인 연락처", "Recipient phone"), o["receiver"]["phone"], "recipient_phone", o["receiver"]["phone"]),
        Col(L(shop, "배송지", "Shipping address"), o["receiver"]["address"], "recipient_address", o["receiver"]["address"]),
        Col(L(shop, "결제대행사", "Payment gateway"), o["payment_origin"], "payment_origin", o["payment_origin"]),
    ]
    if o["paid_at"]:
        cols.append(Col(L(shop, "결제일시", "Paid at"), fmt_date(o["paid_at"], ds, True), "payment_date", iso(o["paid_at"])))
    if o["bank"]:
        cols.append(Col(L(shop, "입금은행", "Bank"), o["bank"], "bank", o["bank"]))
    if o["card"]:
        cols.append(Col(L(shop, "결제카드", "Card"), o["card"], "card", o["card"]))
    return cols


def goods_cols(shop: Shop, g: dict, detail: bool) -> list:
    ds, ms = shop.date_style, shop.money_style
    money = lambda v: fmt_money(v, ms)
    link = f"/admin/goods/edit?it_id={g['id']}" if shop.theme.startswith("admin") else f"/products/{g['id']}"
    cols = [
        Col(L(shop, "상품번호", "Product ID"), g["id"], "id,link", g["id"], href=link),
        Col(L(shop, "상품명", "Product"), g["title"], "title", g["title"]),
        Col(L(shop, "상태", "Status"), g["status"], "status", g["status"]),
        Col(L(shop, "카테고리", "Category"), g["category"], "category", g["category"]),
        Col(L(shop, "판매가", "Price"), money(g["sale_price"]), "sale_price", money_truth(g["sale_price"], shop)),
        Col(L(shop, "재고", "Stock"), str(g["quantity"]), "quantity", g["quantity"]),
        Col(L(shop, "등록일", "Created"), fmt_date(g["registration_date"], ds), "registration_date", iso(g["registration_date"])),
        Col(L(shop, "브랜드", "Brand"), g["brand_name"], "brand_name", g["brand_name"]),
    ]
    if g["compare_at_price"] != g["sale_price"]:
        cols.insert(5, Col(L(shop, "정상가", "List price"), money(g["compare_at_price"]), "compare_at_price", money_truth(g["compare_at_price"], shop)))
    if not detail:
        return cols
    fee_text = money(g["shipping_fee"]) if g["shipping_fee"] else L(shop, "조건부 면제", "Waived over threshold")
    cols += [
        Col(L(shop, "관리코드", "Internal code"), g["code"], "code", g["code"]),
        Col(L(shop, "재고관리번호", "SKU"), g["stock_keeping_unit"], "stock_keeping_unit", g["stock_keeping_unit"]),
        Col(L(shop, "모델", "Model"), g["model_name"], "model_name", g["model_name"]),
        Col(L(shop, "제조사", "Manufacturer"), g["manufacturer"], "manufacturer", g["manufacturer"]),
        Col(L(shop, "원산지", "Country of origin"), g["origin_country"], "origin_country", g["origin_country"]),
        Col(L(shop, "색상", "Color"), g["color"], "color", g["color"]),
        Col(L(shop, "바코드", "Barcode"), g["barcode"], "barcode", g["barcode"]),
        Col(L(shop, "공급가", "Unit cost"), money(g["supply_price"]), "supply_price", money_truth(g["supply_price"], shop)),
        Col(L(shop, "품절임박 기준", "Low stock alert"), str(g["low_stock_threshold"]), "low_stock_threshold", g["low_stock_threshold"]),
        Col(L(shop, "배송비", "Shipping"), fee_text, "shipping_fee", money_truth(g["shipping_fee"], shop) if g["shipping_fee"] else None),
        Col(L(shop, "상품설명", "Description"), g["description"], "description", g["description"], kind="textarea"),
    ]
    return cols


def tracking_cols(shop: Shop, o: dict, detail: bool) -> list:
    ds, ms = shop.date_style, shop.money_style
    link = f"/admin/shipping/view?invoice={o['tracking']}" if shop.theme.startswith("admin") else f"/track/{o['tracking']}"
    cols = [
        Col(L(shop, "운송장번호", "Tracking #"), o["tracking"], "id,link", o["tracking"], href=link),
        Col(L(shop, "택배사", "Carrier"), o["carrier"], "carrier", o["carrier"]),
        Col(L(shop, "배송상태", "Status"), o["status"], "status", o["status"]),
        Col(L(shop, "상품", "Items"), o["goods_label"], "title", o["goods_main"]),
        Col(L(shop, "발송일", "Shipped on"), fmt_date(o["ship_at"], ds), "shipping_date", iso(o["ship_at"])),
        Col(L(shop, "받는분", "Recipient"), o["receiver"]["name"], "recipient_name", o["receiver"]["name"]),
        # 주문번호는 배송 스키마에 없습니다. 식별자 열과 경쟁하는 방해 요소입니다.
        Col(L(shop, "주문번호", "Order #"), o["id"], "", None),
    ]
    if not detail:
        return cols
    w, h, l = o["dims"]
    cols += [
        Col(L(shop, "접수일", "Registered"), fmt_date(o["when"], ds), "registration_date", iso(o["when"])),
        Col(L(shop, "보내는분", "Sender"), shop.company, "sender_name", shop.company),
        Col(L(shop, "보내는분 주소", "Sender address"), shop.company_address, "sender_address", shop.company_address),
        Col(L(shop, "보내는분 연락처", "Sender phone"), shop.company_phone, "sender_phone", shop.company_phone),
        Col(L(shop, "받는분 주소", "Recipient address"), o["receiver"]["address"], "recipient_address", o["receiver"]["address"]),
        Col(L(shop, "받는분 연락처", "Recipient phone"), o["receiver"]["phone"], "recipient_phone", o["receiver"]["phone"]),
        Col(L(shop, "중량", "Weight"), f"{o['weight']}kg", "weight", o["weight"]),
        Col(L(shop, "가로", "Width"), f"{w}cm", "width", w),
        Col(L(shop, "세로", "Height"), f"{h}cm", "height", h),
        Col(L(shop, "길이", "Length"), f"{l}cm", "length", l),
        Col(L(shop, "배송방법", "Method"), o["shipping_method"], "shipping_method", o["shipping_method"]),
        Col(L(shop, "소요일", "Transit time"), days_text(shop, o['duration_days']), "shipping_duration", o["duration_days"]),
        Col(L(shop, "운임", "Freight charge"), fmt_money(o["shipping_fee"], ms), "shipping_fee", money_truth(o["shipping_fee"], shop)),
    ]
    return cols


def review_cols(shop: Shop, x: dict, detail: bool) -> list:
    ds = shop.date_style
    return [
        Col(L(shop, "번호", "ID"), x["id"], "id,link", x["id"], href=f"/admin/review/{x['id']}"),
        Col(L(shop, "상태", "Status"), x["status"], "status", x["status"]),
        Col(L(shop, "내용", "Review"), x["title"], "title", x["title"]),
        Col(L(shop, "상품", "Product"), x["goods"], "", None),
        Col(L(shop, "작성자", "Author"), x["name"], "name", x["name"]),
        Col(L(shop, "평점", "Rating"), "★" * x["rating"], "", None),
        Col(L(shop, "작성일", "Posted"), fmt_date(x["registration_date"], ds), "registration_date", iso(x["registration_date"])),
    ]


def coupon_cols(shop: Shop, x: dict, detail: bool) -> list:
    ds, ms = shop.date_style, shop.money_style
    cols = [
        Col(L(shop, "쿠폰번호", "Discount ID"), x["id"], "id,link", x["id"], href=f"/admin/coupon/{x['id']}"),
        Col(L(shop, "코드", "Code"), x["code"], "code", x["code"]),
        Col(L(shop, "쿠폰명", "Name"), x["title"], "title", x["title"]),
        Col(L(shop, "종류", "Type"), x["type"], "type", x["type"]),
        Col(L(shop, "할인", "Value"), x["discount"], "discount", x["discount"]),
        Col(L(shop, "시작일", "Starts"), fmt_date(x["started_at"], ds), "started_at", iso(x["started_at"])),
        Col(L(shop, "종료일", "Ends"), fmt_date(x["expired_at"], ds), "expired_at", iso(x["expired_at"])),
        Col(L(shop, "상태", "Status"), x["status"], "status", x["status"]),
    ]
    if detail:
        cols += [
            Col(L(shop, "발급수량", "Issued"), str(x["quantity"]), "quantity", x["quantity"]),
            Col(L(shop, "최소주문금액", "Minimum spend"), fmt_money(x["min_order_amount"], ms), "min_order_amount", money_truth(x["min_order_amount"], shop)),
        ]
    return cols


def event_cols(shop: Shop, x: dict, detail: bool) -> list:
    ds = shop.date_style
    return [
        Col(L(shop, "번호", "ID"), x["id"], "id,link", x["id"], href=f"/admin/event/{x['id']}"),
        Col(L(shop, "기획전명", "Campaign"), x["title"], "title", x["title"]),
        Col(L(shop, "유형", "Type"), x["type"], "type", x["type"]),
        Col(L(shop, "시작일", "Starts"), fmt_date(x["started_at"], ds), "started_at", iso(x["started_at"])),
        Col(L(shop, "종료일", "Ends"), fmt_date(x["expired_at"], ds), "expired_at", iso(x["expired_at"])),
        Col(L(shop, "상태", "Status"), x["status"], "status", x["status"]),
    ]


# ──────────────────────────────────────────────
# 7. 목록 렌더러 (테마별)
# ──────────────────────────────────────────────
def cell_html(c: Col) -> str:
    if c.href:
        return f'<a href="{E(c.href)}">{E(c.text)}</a>'
    # 긴 주소는 실제 페이지처럼 <br> 로 끊깁니다. 낱말 경계가 태그에만 있는 경우입니다.
    if c.truth_field.endswith("_address") and "," not in c.text:
        words = c.text.split(" ")
        return E(" ".join(words[:3])) + "<br>" + E(" ".join(words[3:]))
    return E(c.text)


def list_admin_twin(shop: Shop, rows: list, ptype: str) -> str:
    """레코드 하나가 <tr> 두 개입니다. 윗줄은 앞 절반, 아랫줄은 뒤 절반입니다."""
    labels = [c.label for c in rows[0]]
    mid = (len(labels) + 1) // 2
    head1 = "".join(f"<th>{E(x)}</th>" for x in labels[:mid])
    head2 = "".join(f"<th>{E(x)}</th>" for x in labels[mid:])
    out = [
        f'<div class="tbl_head01"><table id="sodr_list" class="tablef"><thead>'
        f'<tr><th rowspan="2"><input type="checkbox"></th><th rowspan="2">No</th>{head1}</tr>'
        f'<tr>{head2}<th>{E(chrome(shop)["edit"])}</th></tr></thead><tbody>'
    ]
    for i, cols in enumerate(rows):
        cl = "list" + str(i % 2)
        top = "".join(f"<td>{cell_html(c)}</td>" for c in cols[:mid])
        bottom = "".join(f"<td>{cell_html(c)}</td>" for c in cols[mid:])
        out.append(
            f'<tr class="{cl}"><td rowspan="2"><input type="checkbox" name="chk[]" value="{i}"></td>'
            f'<td rowspan="2">{len(rows) - i}</td>{top}</tr>'
            f'<tr class="{cl}">{bottom}<td><a class="btn" href="#">{E(chrome(shop)["edit"])}</a></td></tr>'
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def list_admin_grid(shop: Shop, rows: list, ptype: str) -> str:
    """시맨틱 태그 없이 ARIA 역할과 해시 클래스만 있습니다."""
    labels = [c.label for c in rows[0]]
    gcl, rcl, ccl = css_hash("g", ptype), css_hash("r", ptype), css_hash("c", ptype)
    head = "".join(f'<div role="columnheader" class="{ccl}"><span>{E(x)}</span></div>' for x in labels)
    out = [f'<div role="grid" class="{gcl}"><div role="row" class="{rcl} {css_hash("hd")}">{head}</div>']
    for cols in rows:
        cells = "".join(
            f'<div role="gridcell" class="{ccl}"><div class="{css_hash("v", c.label)}">{cell_html(c)}</div></div>'
            for c in cols
        )
        out.append(f'<div role="row" class="{rcl}">{cells}</div>')
    out.append("</div>")
    return "".join(out)


def list_admin_flat(shop: Shop, rows: list, ptype: str) -> str:
    labels = [c.label for c in rows[0]]
    head = "".join(f"<th>{E(x)}</th>" for x in labels)
    body = "".join("<tr>" + "".join(f"<td>{cell_html(c)}</td>" for c in cols) + "</tr>" for cols in rows)
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def list_store_cards(shop: Shop, rows: list, ptype: str) -> str:
    """구매자 화면 카드. 앞의 다섯 값에는 라벨이 없고, 나머지만 '라벨 값' 으로 붙습니다."""
    out = ['<ul class="order-cards">']
    for cols in rows:
        head = "".join(f'<span class="meta">{cell_html(c)}</span>' for c in cols[:2])
        title = f'<strong class="name">{cell_html(cols[2])}</strong>' if len(cols) > 2 else ""
        rest_plain = "".join(f'<p class="sub">{cell_html(c)}</p>' for c in cols[3:5])
        labelled = "".join(f'<li><span>{E(c.label)}</span> <b>{cell_html(c)}</b></li>' for c in cols[5:])
        out.append(
            f'<li class="card"><div class="card-head">{head}</div>'
            f'<div class="card-body"><img src="/img/thumb.jpg" alt="">{title}{rest_plain}</div>'
            f'<ul class="card-meta">{labelled}</ul><div class="card-foot"><button>{E(chrome(shop)["view"])}</button></div></li>'
        )
    out.append("</ul>")
    return "".join(out)


def list_mobile_dl(shop: Shop, rows: list, ptype: str) -> str:
    out = ['<div class="m-list">']
    for cols in rows:
        pairs = "".join(f"<dt>{E(c.label)}</dt><dd>{cell_html(c)}</dd>" for c in cols)
        out.append(f'<section class="m-item"><dl>{pairs}</dl></section>')
    out.append("</div>")
    return "".join(out)


def list_store_sections(shop: Shop, rows: list, ptype: str) -> str:
    """응답형 표. 셀 안에 data-label 이 있고 값은 div 로 한 겹 더 싸여 있습니다."""
    labels = [c.label for c in rows[0]]
    head = "".join(f'<th scope="col">{E(x)}</th>' for x in labels)
    body = "".join(
        "<tr>" + "".join(f'<td data-label="{E(c.label)}"><div class="cell">{cell_html(c)}</div></td>' for c in cols) + "</tr>"
        for cols in rows
    )
    return f'<section class="responsive"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>'


LIST_RENDERERS = {
    "admin_twin": list_admin_twin, "admin_grid": list_admin_grid, "admin_flat": list_admin_flat,
    "store_cards": list_store_cards, "mobile_dl": list_mobile_dl, "store_sections": list_store_sections,
}


# ──────────────────────────────────────────────
# 8. 상세 렌더러 (테마별)
# ──────────────────────────────────────────────
def input_html(c: Col, i: int) -> str:
    """관리자 폼: 값이 태그 텍스트가 아니라 input/textarea 의 속성 · 내용에 있습니다."""
    if c.kind == "textarea":
        return f'<textarea name="f{i}">{E(c.text)}</textarea>'
    if c.href:
        return f'<a href="{E(c.href)}">{E(c.text)}</a>'
    return f'<input type="text" name="f{i}" value="{E(c.text)}">'


def detail_admin_twin(shop: Shop, cols: list, ptype: str) -> str:
    rows = "".join(
        f'<tr><th scope="row"><label for="f{i}">{E(c.label)}</label></th><td>{input_html(c, i)}</td></tr>'
        for i, c in enumerate(cols)
    )
    return f'<form class="frm"><div class="tbl_frm01"><table>{rows}</table></div><button>{E(chrome(shop)["edit"])}</button></form>'


def detail_admin_grid(shop: Shop, cols: list, ptype: str) -> str:
    half = (len(cols) + 1) // 2
    def panel(part, n):
        rows = "".join(
            f'<div class="{css_hash("f", n)}"><div class="{css_hash("l", n)}">{E(c.label)}</div>'
            f'<div class="{css_hash("v", n)}">{cell_html(c)}</div></div>'
            for c in part
        )
        return f'<section class="{css_hash("p", n)}">{rows}</section>'
    return panel(cols[:half], 1) + panel(cols[half:], 2)


def detail_admin_flat(shop: Shop, cols: list, ptype: str) -> str:
    rows = "".join(f"<tr><td class=\"k\">{E(c.label)}</td><td>{cell_html(c)}</td></tr>" for c in cols)
    return f'<table class="kv">{rows}</table>'


def detail_store_cards(shop: Shop, cols: list, ptype: str) -> str:
    head = "".join(f"<p>{cell_html(c)}</p>" for c in cols[:3])
    rows = "".join(f'<li><span class="k">{E(c.label)}</span><span class="v">{cell_html(c)}</span></li>' for c in cols[3:])
    return f'<section class="hero"><img src="/img/hero.jpg" alt="">{head}</section><ul class="spec">{rows}</ul>'


def detail_mobile_dl(shop: Shop, cols: list, ptype: str) -> str:
    pairs = "".join(f"<dt>{E(c.label)}</dt><dd>{cell_html(c)}</dd>" for c in cols)
    return f'<article class="m-detail"><dl>{pairs}</dl></article>'


def detail_store_sections(shop: Shop, cols: list, ptype: str) -> str:
    chunks = []
    for start in range(0, len(cols), 5):
        ps = "".join(f"<p><strong>{E(c.label)}:</strong> {cell_html(c)}</p>" for c in cols[start:start + 5])
        chunks.append(f"<section>{ps}</section>")
    return "".join(chunks)


DETAIL_RENDERERS = {
    "admin_twin": detail_admin_twin, "admin_grid": detail_admin_grid, "admin_flat": detail_admin_flat,
    "store_cards": detail_store_cards, "mobile_dl": detail_mobile_dl, "store_sections": detail_store_sections,
}


def recommend_widget(shop: Shop, world: dict, exclude_title: str, seed_key) -> str:
    """구매자 화면 상세 아래의 추천 상품. 다른 상품 이름과 가격이 본문에 섞이고, 반복 구조라 목록처럼 보이는 함정입니다."""
    r = rng_for(shop.key, "recommend", seed_key)
    picks = r.sample([g for g in world["goods"] if g["title"] != exclude_title], 4)
    items = "".join(
        f'<li class="reco"><a href="/products/{g["id"]}"><img src="/img/t.jpg" alt=""></a>'
        f'<span class="reco-name">{E(g["title"])}</span><span class="reco-brand">{E(g["brand_name"])}</span>'
        f'<em>{E(fmt_money(g["sale_price"], shop.money_style))}</em></li>'
        for g in picks
    )
    return f'<section class="recommend"><h3>{E(chrome(shop)["recent"])}</h3><ul>{items}</ul></section>'


def line_items_table(shop: Shop, o: dict) -> str:
    """주문 상세의 상품 표. 상세 페이지에도 행이 반복되는 표가 있다는 것을 보여 주는 함정입니다."""
    money = lambda v: fmt_money(v, shop.money_style)
    head = "".join(f"<th>{E(L(shop, a, b))}</th>" for a, b in (("상품", "Item"), ("수량", "Qty"), ("단가", "Unit price"), ("소계", "Subtotal")))
    rows = "".join(
        f'<tr><td>{E(l["goods"]["title"])}</td><td>{l["qty"]}</td><td>{E(money(l["goods"]["sale_price"]))}</td>'
        f'<td>{E(money(l["goods"]["sale_price"] * l["qty"]))}</td></tr>'
        for l in o["lines"]
    )
    return f'<div class="order-lines"><h3>{E(L(shop, "주문 상품", "Items ordered"))}</h3><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


# ──────────────────────────────────────────────
# 9. 페이지 조립
# ──────────────────────────────────────────────
LIST_SIZE = 12
LIST_PAGES = {"order": 2, "goods": 2, "tracking": 2, "review": 1, "coupon": 1, "event": 1}
DETAIL_PAGES = {"order": 4, "goods": 4, "tracking": 4, "review": 1, "coupon": 1, "event": 1}
COLS_OF = {
    "order": order_cols, "goods": goods_cols, "tracking": tracking_cols,
    "review": review_cols, "coupon": coupon_cols, "event": event_cols,
}
ID_FIELD = "id,link"


def truth_of(cols: list) -> dict:
    return {c.truth_field: c.truth_value for c in cols if c.truth_field and c.truth_value is not None}


def page_url(shop: Shop, ptype: str, layout: str, key) -> str:
    admin = shop.theme.startswith("admin")
    base = f"https://{shop.domain}"
    if admin:
        path = {"order": "order", "goods": "goods", "tracking": "shipping", "review": "review", "coupon": "coupon", "event": "event"}[ptype]
        return f"{base}/admin/{path}/list?page={key}" if layout == "list" else f"{base}/admin/{path}/view?id={key}"
    path = {"order": "my/orders", "goods": "products", "tracking": "my/shipments", "review": "reviews", "coupon": "my/coupons", "event": "events"}[ptype]
    return f"{base}/{path}?page={key}" if layout == "list" else f"{base}/{path}/{key}"


def make_list_page(shop: Shop, world: dict, ptype: str, records: list, page: int) -> tuple:
    rows = [COLS_OF[ptype](shop, rec, False) for rec in records]
    # 열 구성은 표 하나 안에서 같아야 합니다(선택 열은 빈 값으로 맞춥니다).
    labels = []
    for cols in rows:
        for c in cols:
            if c.label not in labels:
                labels.append(c.label)
    aligned = []
    for cols in rows:
        by = {c.label: c for c in cols}
        aligned.append([by.get(x, Col(x, "-")) for x in labels])
    title = page_title(shop, ptype, "list")
    body = toolbar(shop, ptype, len(records) * 7) + LIST_RENDERERS[shop.theme](shop, aligned, ptype) + pager(shop, page)
    return chrome_for(shop, ptype, body, title), [truth_of(c) for c in rows], title


def make_detail_page(shop: Shop, world: dict, ptype: str, rec: dict) -> tuple:
    cols = COLS_OF[ptype](shop, rec, True)
    title = page_title(shop, ptype, "detail")
    body = DETAIL_RENDERERS[shop.theme](shop, cols, ptype)
    if ptype == "order":
        body += line_items_table(shop, rec)
    if ptype in ("goods", "order") and not shop.theme.startswith("admin"):
        exclude = rec["title"] if ptype == "goods" else rec["goods_main"]
        body += recommend_widget(shop, world, exclude, rec["id"])
    return chrome_for(shop, ptype, body, title), truth_of(cols), title


def pages_for(shop: Shop, world: dict):
    shipped = [o for o in world["orders"] if o["tracking"]]
    sources = {
        "order": world["orders"], "goods": world["goods"], "tracking": shipped,
        "review": world["reviews"], "coupon": world["coupons"], "event": world["events"],
    }
    for ptype, recs in sources.items():
        for page in range(1, LIST_PAGES[ptype] + 1):
            part = recs[(page - 1) * LIST_SIZE: page * LIST_SIZE]
            if not part:
                continue
            doc, items, title = make_list_page(shop, world, ptype, part, page)
            yield {
                "page_id": f"{shop.key}-{ptype}-list-{page}",
                "url": page_url(shop, ptype, "list", page),
                "page_type": ptype, "layout": "list", "items": items, "title": title,
            }, doc
        for rec in recs[:DETAIL_PAGES[ptype]]:
            key = rec["tracking"] if ptype == "tracking" else rec["id"]
            doc, fields, title = make_detail_page(shop, world, ptype, rec)
            yield {
                "page_id": f"{shop.key}-{ptype}-detail-{key}",
                "url": page_url(shop, ptype, "detail", key),
                "page_type": ptype, "layout": "detail", "fields": fields, "title": title,
            }, doc


# ──────────────────────────────────────────────
# 10. 자연어 질의 (검색 평가)
# ──────────────────────────────────────────────
CONCEPT_QUERIES = {
    "cold_weather": ("추운 날 몸을 따뜻하게 해 줄 물건", "something to keep me warm when it is freezing"),
    "hot_weather": ("더위를 식히는 데 도움이 되는 제품", "products that help me cool off on a sweltering day"),
    "desk_work": ("재택근무 책상을 정돈하고 편하게 만드는 용품", "gear for a tidier, more comfortable home office desk"),
    "cooking": ("요리할 때 쓰는 주방 도구", "tools I would use while cooking dinner"),
    "kids": ("어린아이에게 줄 선물", "a gift for a young child"),
    "pet": ("강아지나 고양이를 키울 때 필요한 것", "things I need for my dog or cat"),
    "outdoor": ("주말 야영에 챙겨 갈 장비", "equipment to pack for a weekend camping trip"),
    "skincare": ("건조하고 민감한 피부 관리 제품", "products for looking after dry skin"),
    "snack_food": ("출출할 때 먹을 간식이나 마실 것", "something to eat or drink for a quick bite"),
    "audio_power": ("이동 중에 음악을 듣거나 전자기기를 충전하는 물건", "gadgets for listening to music or charging devices on the go"),
}


def rec_key(shop: Shop, kind: str, rid: str) -> str:
    return f"{shop.key}:{kind}:{rid}"


def visible_records(world: dict) -> dict:
    """페이지에 실제로 나타나는 레코드. 페이지에 없는 레코드는 파이프라인이 볼 수 없으므로 검색 정답에서 뺍니다."""
    shipped = [o for o in world["orders"] if o["tracking"]]
    return {
        "goods": world["goods"][: LIST_SIZE * LIST_PAGES["goods"]],
        "orders": world["orders"][: LIST_SIZE * LIST_PAGES["order"]],
        "shipped": shipped[: LIST_SIZE * LIST_PAGES["tracking"]],
    }


def queries_for(shop: Shop, world: dict) -> list:
    """관련 레코드는 세계에서 계산합니다. 사람이 고른 정답이 아니라 조건을 만족하는 전체 집합입니다."""
    q = []
    ko = shop.lang == "ko"
    seen = visible_records(world)
    goods, orders, shipped = seen["goods"], seen["orders"], seen["shipped"]

    def add(kind, lang, text, relevant, domain):
        if relevant:
            q.append({
                "split": shop.split, "shop": shop.key, "kind": kind, "lang": lang, "domain": domain,
                "query": text, "relevant": sorted(set(relevant)),
            })

    loc = None if shop.lang in LOCAL_SHOP else pack(shop)
    if loc:
        queries_for_pack(shop, world, loc, add)
        return q

    # ① 의미 검색: 페이지에 없는 말로 묻습니다. 영어 질의는 언어를 건너가는 경우입니다.
    for concept, (q_ko, q_en) in CONCEPT_QUERIES.items():
        rel = [rec_key(shop, "goods", g["id"]) for g in goods if g["concept"] == concept]
        add("semantic", "ko", q_ko, rel, "goods")
        add("semantic", "en", q_en, rel, "goods")

    # ② 정확 식별자
    r = rng_for(shop.key, "queries")
    for o in r.sample(orders, 5):
        add("identifier", shop.lang, (f"주문번호 {o['id']} 내역 알려줘" if ko else f"look up order {o['id']}"),
            [rec_key(shop, "order", o["id"])], "order")
    for o in r.sample(shipped, min(5, len(shipped))):
        add("identifier", shop.lang, (f"운송장 {o['tracking']} 어디까지 왔어" if ko else f"where is tracking number {o['tracking']}"),
            [rec_key(shop, "tracking", o["tracking"])], "tracking")
    for g in r.sample(goods, 5):
        add("identifier", shop.lang, (f"상품번호 {g['id']} 재고 알려줘" if ko else f"stock for product {g['id']}"),
            [rec_key(shop, "goods", g["id"])], "goods")

    # ③ 상태 조건
    for s in sorted({o["status"] for o in orders}):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if o["status"] == s]
        add("status", shop.lang, (f"{s} 상태인 주문 목록" if ko else f"orders with status {s}"), rel, "order")

    # ④ 이름 · 금액 · 기간 조건
    for o in r.sample(orders, 5):
        name = o["buyer"]["name"]
        rel = [rec_key(shop, "order", x["id"]) for x in orders if x["buyer"]["name"] == name]
        add("person", shop.lang, (f"{name} 고객 주문 찾아줘" if ko else f"find orders from {name}"), rel, "order")
    amounts = sorted(o["amount"] for o in orders)
    threshold = amounts[int(len(amounts) * 0.85)]
    rel = [rec_key(shop, "order", o["id"]) for o in orders if o["amount"] >= threshold]
    shown = fmt_money(threshold, shop.money_style)
    add("amount", shop.lang, (f"결제금액 {shown} 이상 주문" if ko else f"orders of {shown} or more"), rel, "order")
    for month in (5, 7, 9):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if o["when"].month == month]
        add("period", shop.lang, (f"2026년 {month}월 주문" if ko else f"orders placed in {date(2026, month, 1):%B} 2026"), rel, "order")

    # ⑤ 상품 이름 → 그 상품이 들어간 주문 (다른 언어의 이름으로도 묻습니다)
    for g in r.sample(goods, 6):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if any(l["goods"]["id"] == g["id"] for l in o["lines"])]
        add("goods_in_order", shop.lang, (f"{g['title']} 들어간 주문" if ko else f"orders containing {g['title']}"), rel, "order")
        add("goods_exact", shop.lang, g["title"], [rec_key(shop, "goods", g["id"])], "goods")
        add("goods_crosslang", "en" if ko else "ko", g["title_alt"], [rec_key(shop, "goods", g["id"])], "goods")
    return q


def queries_for_pack(shop: Shop, world: dict, loc: dict, add) -> list:
    """언어 팩 쇼핑몰: 같은 종류의 질의를 그 언어로 씁니다. 의미 검색은 그 언어 + 영어(언어 건너기), 상품명 건너기는 영어 이름."""
    seen = visible_records(world)
    goods, orders, shipped = seen["goods"], seen["orders"], seen["shipped"]
    q = loc["queries"]
    lang = shop.lang
    for concept, (_, q_en) in CONCEPT_QUERIES.items():
        rel = [rec_key(shop, "goods", g["id"]) for g in goods if g["concept"] == concept]
        add("semantic", lang, loc["concepts"][concept], rel, "goods")
        add("semantic", "en", q_en, rel, "goods")
    r = rng_for(shop.key, "queries")
    for o in r.sample(orders, 5):
        add("identifier", lang, q["order"].format(id=o["id"]), [rec_key(shop, "order", o["id"])], "order")
    for o in r.sample(shipped, min(5, len(shipped))):
        add("identifier", lang, q["tracking"].format(tracking=o["tracking"]), [rec_key(shop, "tracking", o["tracking"])], "tracking")
    for g in r.sample(goods, 5):
        add("identifier", lang, q["goods"].format(id=g["id"]), [rec_key(shop, "goods", g["id"])], "goods")
    for s in sorted({o["status"] for o in orders}):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if o["status"] == s]
        add("status", lang, q["status"].format(status=s), rel, "order")
    for o in r.sample(orders, 5):
        name = o["buyer"]["name"]
        rel = [rec_key(shop, "order", x["id"]) for x in orders if x["buyer"]["name"] == name]
        add("person", lang, q["person"].format(name=name), rel, "order")
    amounts = sorted(o["amount"] for o in orders)
    threshold = amounts[int(len(amounts) * 0.85)]
    rel = [rec_key(shop, "order", o["id"]) for o in orders if o["amount"] >= threshold]
    add("amount", lang, q["amount"].format(amount=fmt_money(threshold, shop.money_style)), rel, "order")
    for month in (5, 7, 9):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if o["when"].month == month]
        add("period", lang, q["period"].format(month=loc["months"][month - 1], n=month), rel, "order")
    for g in r.sample(goods, 6):
        rel = [rec_key(shop, "order", o["id"]) for o in orders if any(l["goods"]["id"] == g["id"] for l in o["lines"])]
        add("goods_in_order", lang, q["goods_in_order"].format(title=g["title"]), rel, "order")
        add("goods_exact", lang, g["title"], [rec_key(shop, "goods", g["id"])], "goods")
        add("goods_crosslang", "en", g["title_alt"], [rec_key(shop, "goods", g["id"])], "goods")


def status_words() -> dict:
    """언어 팩 상태어 → 영어 상태어. 평가 도구(score_commerce.status_table)가 시드의 상태 코드를 정할 때 씁니다."""
    english = sorted({x for tabs in STATUS_TABS.values() for x in tabs["en"] if x != "All"})
    return {lang: {pk["tr"][e]: e for e in english} for lang, pk in PACKS.items()}


# ──────────────────────────────────────────────
# 11. 정답 파일 · 실행
# ──────────────────────────────────────────────
def world_records(shop: Shop, world: dict) -> list:
    """페이지에 나타나는 모든 레코드의 정답. 검색 평가에서 정답 키가 가리키는 대상입니다."""
    out = []
    seen = visible_records(world)
    for g in seen["goods"]:
        out.append({
            "key": rec_key(shop, "goods", g["id"]), "shop": shop.key, "type": "goods", "id": g["id"],
            "title": g["title"], "concept": g["concept"], "status": g["status"],
            "price": money_truth(g["sale_price"], shop), "date": iso(g["registration_date"]),
        })
    for o in seen["orders"]:
        out.append({
            "key": rec_key(shop, "order", o["id"]), "shop": shop.key, "type": "order", "id": o["id"],
            "title": o["goods_main"], "status": o["status"], "amount": money_truth(o["amount"], shop),
            "date": iso(o["when"]), "buyer": o["buyer"]["name"], "tracking": o["tracking"],
            "goods_ids": [l["goods"]["id"] for l in o["lines"]],
        })
    for o in seen["shipped"]:
        out.append({
            "key": rec_key(shop, "tracking", o["tracking"]), "shop": shop.key, "type": "tracking", "id": o["tracking"],
            "title": o["goods_main"], "status": o["status"], "date": iso(o["ship_at"]),
            "carrier": o["carrier"], "order_id": o["id"],
        })
    return out


def generate(out_root: Path) -> dict:
    yaml_dir = out_root / "input_yamls"
    html_dir = out_root / "output_html"
    for d in (yaml_dir, html_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    manifest, queries, records = [], [], []
    for shop in SHOPS:
        world = build_world(shop)
        for meta, doc in pages_for(shop, world):
            pid = meta["page_id"]
            meta = {
                "page_id": pid, "split": shop.split, "shop": shop.key, "shop_name": shop.name,
                "lang": shop.lang, "theme": shop.theme, **{k: v for k, v in meta.items() if k != "page_id"},
            }
            (html_dir / f"{pid}.html").write_text(doc, encoding="utf-8")
            (yaml_dir / f"{pid}.yaml").write_text(
                yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8"
            )
            manifest.append({k: meta[k] for k in ("page_id", "split", "shop", "lang", "theme", "page_type", "layout", "url")})
        queries += queries_for(shop, world)
        records += world_records(shop, world)

    (out_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    (out_root / "queries.yaml").write_text(
        yaml.safe_dump(queries, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    (out_root / "records.yaml").write_text(
        yaml.safe_dump(records, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    (out_root / "status_words.yaml").write_text(
        yaml.safe_dump(status_words(), allow_unicode=True, sort_keys=True, width=1000), encoding="utf-8")
    return {"pages": len(manifest), "queries": len(queries), "records": len(records)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="임시 폴더에 다시 생성해 저장된 파일과 같은지 확인합니다")
    args = ap.parse_args()
    if not args.check:
        print(generate(ROOT))
        return 0

    import filecmp
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        generate(tmp_root)
        diffs = []
        for sub in ("input_yamls", "output_html"):
            cmp = filecmp.dircmp(tmp_root / sub, ROOT / sub)
            diffs += cmp.diff_files + cmp.left_only + cmp.right_only
        for name in ("manifest.yaml", "queries.yaml", "records.yaml", "status_words.yaml"):
            if not filecmp.cmp(tmp_root / name, ROOT / name, shallow=False):
                diffs.append(name)
    print("같음" if not diffs else f"다름 {len(diffs)}건: {diffs[:10]}")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
