"""
N:N 복합 질의 (v2)
=================

질의 틀 23개를 12개 언어로 씁니다. 틀마다 정답은 세계(records.yaml · weblog/truth.yaml)에서 계산합니다.

  커머스 C1~C10   표 두 개 이상을 건너는 질의 (주문 → 운송장, 고객 → 주문 → 상품, 이벤트 기간 → 주문 → 상품 ...)
                  틀마다 쇼핑몰 2곳(데이터 언어 2개)에 하나씩, 모든 질의 언어가 같은 사례를 묻습니다
  무역   R1~R7    서류 사슬 (B/L → CI · PL · CO · IP, PO → B/L 2건, L/C → CI 2건, 품목 → B/L, 기간 · 선적항 → 서류)
  행동   A1~A6    도메인(사이트 이름) · 웹 경로(페이지 종류) · 기간 · 상품을 가리키는 행동 로그 질의

슬롯 규칙: 상품 이름은 질의 언어의 이름(번역, 언어 건너기를 재려고), 고객 · 쿠폰 · 이벤트 · 수출자 · 사이트 이름은
화면에 보이는 그대로(데이터 언어), 번호는 그대로, 달 · 날짜 · 항구 이름은 질의 언어의 꼴입니다.
"""
from datetime import date

import yaml

LANGS = ["ko", "en", "ja", "zh-hans", "de", "fr", "es", "it", "pt", "nl", "cs", "ar"]

MONTH = {  # '그 달에' 꼴
    "ko": lambda m: f"{m}월", "ja": lambda m: f"{m}月", "zh-hans": lambda m: f"{m}月",
    "en": lambda m: "in " + ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                            "October", "November", "December"][m - 1],
    "de": lambda m: "im " + ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                            "November", "Dezember"][m - 1],
    "fr": lambda m: "en " + ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                            "novembre", "décembre"][m - 1],
    "es": lambda m: "en " + ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                            "noviembre", "diciembre"][m - 1],
    "it": lambda m: ("ad " if m in (4, 8, 10) else "a ") + ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                                                           "agosto", "settembre", "ottobre", "novembre", "dicembre"][m - 1],
    "pt": lambda m: "em " + ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
                            "novembro", "dezembro"][m - 1],
    "nl": lambda m: "in " + ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
                            "november", "december"][m - 1],
    "cs": lambda m: ["v lednu", "v únoru", "v březnu", "v dubnu", "v květnu", "v červnu", "v červenci", "v srpnu", "v září",
                     "v říjnu", "v listopadu", "v prosinci"][m - 1],
    "ar": lambda m: "في " + ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
                            "نوفمبر", "ديسمبر"][m - 1],
}

_EN_M = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAY = {  # 날짜 하나
    "ko": lambda d: f"{d.year}년 {d.month}월 {d.day}일", "ja": lambda d: f"{d.year}年{d.month}月{d.day}日",
    "zh-hans": lambda d: f"{d.year}年{d.month}月{d.day}日", "en": lambda d: f"{_EN_M[d.month - 1]} {d.day}, {d.year}",
    "de": lambda d: f"{d.day}. " + ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                                     "November", "Dezember"][d.month - 1] + f" {d.year}",
    "fr": lambda d: ("1er" if d.day == 1 else str(d.day)) + " " + ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                                                                    "août", "septembre", "octobre", "novembre", "décembre"][d.month - 1] + f" {d.year}",
    "es": lambda d: f"{d.day} de " + ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                                       "octubre", "noviembre", "diciembre"][d.month - 1] + f" de {d.year}",
    "it": lambda d: f"{d.day} " + ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
                                    "ottobre", "novembre", "dicembre"][d.month - 1] + f" {d.year}",
    "pt": lambda d: f"{d.day} de " + ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
                                       "outubro", "novembro", "dezembro"][d.month - 1] + f" de {d.year}",
    "nl": lambda d: f"{d.day} " + ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september",
                                    "oktober", "november", "december"][d.month - 1] + f" {d.year}",
    "cs": lambda d: f"{d.day}. " + ["ledna", "února", "března", "dubna", "května", "června", "července", "srpna", "září", "října",
                                     "listopadu", "prosince"][d.month - 1] + f" {d.year}",
    "ar": lambda d: f"{d.day} " + ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
                                    "نوفمبر", "ديسمبر"][d.month - 1] + f" {d.year}",
}

PORT = {  # 선적항(도시) 이름, 질의 언어의 꼴. 체코어는 'z ...' 뒤의 2격.
    "Hamburg": {"ko": "함부르크", "ja": "ハンブルク", "zh-hans": "汉堡", "ar": "هامبورغ", "cs": "Hamburku"},
    "Osaka": {"ko": "오사카", "ja": "大阪", "zh-hans": "大阪", "ar": "أوساكا", "cs": "Ósaky", "en": "Osaka"},
    "Busan": {"ko": "부산", "ja": "釜山", "zh-hans": "釜山", "ar": "بوسان", "cs": "Pusanu"},
    "Felixstowe": {"ko": "펠릭스토우", "ja": "フェリクストウ", "zh-hans": "费利克斯托", "ar": "فيليكستو", "cs": "Felixstowe"},
}
POL_CITY = {"sorae": "Hamburg", "copper": "Osaka", "komorebi": "Busan", "kranich": "Felixstowe"}

T = {
    # ── 커머스 ──
    "C1": {"ko": "주문 {order}의 운송장 번호 알려줘", "en": "show the tracking numbers for order {order}",
           "ja": "注文{order}の追跡番号を教えて", "zh-hans": "订单{order}的快递单号是多少",
           "de": "Sendungsnummern zur Bestellung {order} anzeigen", "fr": "numéros de suivi de la commande {order}",
           "es": "números de seguimiento del pedido {order}", "it": "numeri di tracciamento dell'ordine {order}",
           "pt": "números de rastreio do pedido {order}", "nl": "trackingnummers van bestelling {order}",
           "cs": "čísla zásilek k objednávce {order}", "ar": "أرقام تتبع الطلب {order}"},
    "C2": {"ko": "운송장 번호가 {tracking}인 주문 찾아줘", "en": "which order was shipped with tracking number {tracking}?",
           "ja": "追跡番号{tracking}で発送した注文はどれ？", "zh-hans": "快递单号{tracking}对应的是哪个订单",
           "de": "Zu welcher Bestellung gehört die Sendungsnummer {tracking}?",
           "fr": "quelle commande a été expédiée avec le numéro de suivi {tracking} ?",
           "es": "¿qué pedido se envió con el número de seguimiento {tracking}?",
           "it": "quale ordine è stato spedito con il numero di tracciamento {tracking}?",
           "pt": "qual pedido foi enviado com o código de rastreio {tracking}?",
           "nl": "bij welke bestelling hoort trackingnummer {tracking}?",
           "cs": "ke které objednávce patří číslo zásilky {tracking}?", "ar": "ما الطلب الذي شُحن برقم التتبع {tracking}؟"},
    "C3": {"ko": "{customer} 고객이 주문한 상품", "en": "products ordered by {customer}", "ja": "{customer}さんが注文した商品",
           "zh-hans": "{customer}订购的商品", "de": "Artikel, die {customer} bestellt hat", "fr": "produits commandés par {customer}",
           "es": "productos que pidió {customer}", "it": "prodotti ordinati da {customer}", "pt": "produtos que {customer} encomendou",
           "nl": "producten die {customer} heeft besteld", "cs": "zboží z objednávek zákazníka {customer}",
           "ar": "المنتجات التي طلبها {customer}"},
    "C4": {"ko": "{month}에 {goods} 포함된 주문", "en": "orders {month} that included {goods}", "ja": "{month}に{goods}を含む注文",
           "zh-hans": "{month}包含{goods}的订单", "de": "Bestellungen {month} mit {goods}", "fr": "commandes {month} contenant {goods}",
           "es": "pedidos {month} que incluían {goods}", "it": "ordini {month} che contenevano {goods}",
           "pt": "pedidos {month} que incluíam {goods}", "nl": "bestellingen {month} met {goods}",
           "cs": "objednávky {month}, které obsahovaly {goods}", "ar": "الطلبات {month} التي تضمنت {goods}"},
    "C5": {"ko": "'{coupon}' 쿠폰을 사용한 주문", "en": "orders that used the '{coupon}' coupon", "ja": "「{coupon}」クーポンを使った注文",
           "zh-hans": "使用了“{coupon}”优惠券的订单", "de": "Bestellungen, bei denen der Gutschein „{coupon}“ eingelöst wurde",
           "fr": "commandes ayant utilisé le coupon « {coupon} »", "es": "pedidos que usaron el cupón «{coupon}»",
           "it": "ordini che hanno usato il coupon «{coupon}»", "pt": "pedidos que usaram o cupom “{coupon}”",
           "nl": "bestellingen waarbij de kortingsbon '{coupon}' is gebruikt", "cs": "objednávky, ve kterých byl použit kupón „{coupon}“",
           "ar": "الطلبات التي استخدمت قسيمة «{coupon}»"},
    "C6": {"ko": "'{event}' 기획전에 걸린 상품", "en": "products featured in the '{event}' promotion", "ja": "「{event}」の対象商品",
           "zh-hans": "“{event}”活动的参与商品", "de": "Artikel der Aktion „{event}“", "fr": "produits de l'opération « {event} »",
           "es": "productos de la promoción «{event}»", "it": "prodotti della promozione «{event}»", "pt": "produtos da promoção “{event}”",
           "nl": "producten in de actie '{event}'", "cs": "zboží v akci „{event}“", "ar": "المنتجات المشمولة في عرض «{event}»"},
    "C7": {"ko": "'{event}' 기간에 팔린 상품", "en": "products sold during the '{event}' promotion", "ja": "「{event}」の期間中に売れた商品",
           "zh-hans": "“{event}”活动期间卖出的商品", "de": "Artikel, die während der Aktion „{event}“ verkauft wurden",
           "fr": "produits vendus pendant l'opération « {event} »", "es": "productos vendidos durante la promoción «{event}»",
           "it": "prodotti venduti durante la promozione «{event}»", "pt": "produtos vendidos durante a promoção “{event}”",
           "nl": "producten die tijdens de actie '{event}' zijn verkocht", "cs": "zboží prodané během akce „{event}“",
           "ar": "المنتجات التي بيعت خلال عرض «{event}»"},
    "C8": {"ko": "{customer} 고객이 리뷰를 남긴 상품", "en": "products {customer} wrote a review for", "ja": "{customer}さんがレビューを書いた商品",
           "zh-hans": "{customer}写过评价的商品", "de": "Artikel, zu denen {customer} eine Bewertung geschrieben hat",
           "fr": "produits pour lesquels {customer} a laissé un avis", "es": "productos que {customer} reseñó",
           "it": "prodotti recensiti da {customer}", "pt": "produtos que {customer} avaliou", "nl": "producten waarover {customer} een review schreef",
           "cs": "zboží s recenzí od zákazníka {customer}", "ar": "المنتجات التي كتب {customer} مراجعة لها"},
    "C9": {"ko": "반품된 주문의 상품", "en": "products from returned orders", "ja": "返品された注文の商品", "zh-hans": "退货订单里的商品",
           "de": "Artikel aus zurückgesendeten Bestellungen", "fr": "produits des commandes retournées", "es": "productos de pedidos devueltos",
           "it": "prodotti degli ordini restituiti", "pt": "produtos de pedidos devolvidos", "nl": "producten uit geretourneerde bestellingen",
           "cs": "zboží z vrácených objednávek", "ar": "منتجات الطلبات المرتجعة"},
    "C10": {"ko": "{month}에 {customer} 고객에게 배송된 운송장", "en": "shipments delivered to {customer} {month}",
            "ja": "{month}に{customer}さんへ配送された荷物", "zh-hans": "{month}寄给{customer}的包裹",
            "de": "Sendungen, die {month} an {customer} zugestellt wurden", "fr": "colis livrés à {customer} {month}",
            "es": "envíos entregados a {customer} {month}", "it": "spedizioni consegnate a {customer} {month}",
            "pt": "envios entregues a {customer} {month}", "nl": "zendingen die {month} aan {customer} zijn bezorgd",
            "cs": "zásilky doručené zákazníkovi {customer} {month}", "ar": "الشحنات التي سُلّمت إلى {customer} {month}"},
    # ── 무역 ──
    "R1": {"ko": "B/L {bl} 관련 서류", "en": "documents related to B/L {bl}", "ja": "B/L {bl}に関連する書類", "zh-hans": "提单{bl}相关的单据",
           "de": "Unterlagen zum Konnossement {bl}", "fr": "documents liés au connaissement {bl}",
           "es": "documentos relacionados con el conocimiento de embarque {bl}", "it": "documenti collegati alla polizza di carico {bl}",
           "pt": "documentos relacionados ao conhecimento de embarque {bl}", "nl": "documenten bij cognossement {bl}",
           "cs": "dokumenty ke konosamentu {bl}", "ar": "المستندات المتعلقة ببوليصة الشحن {bl}"},
    "R2": {"ko": "발주번호 {po}의 선하증권", "en": "bills of lading for purchase order {po}", "ja": "注文書{po}の船荷証券",
           "zh-hans": "采购订单{po}的提单", "de": "Konnossemente zur Bestellung {po}", "fr": "connaissements de la commande {po}",
           "es": "conocimientos de embarque de la orden de compra {po}", "it": "polizze di carico dell'ordine d'acquisto {po}",
           "pt": "conhecimentos de embarque do pedido de compra {po}", "nl": "cognossementen bij inkooporder {po}",
           "cs": "konosamenty k objednávce {po}", "ar": "بوالص الشحن الخاصة بأمر الشراء {po}"},
    "R3": {"ko": "선하증권 {bl}의 상업송장", "en": "commercial invoice for bill of lading {bl}", "ja": "船荷証券{bl}の商業送り状",
           "zh-hans": "提单{bl}对应的商业发票", "de": "Handelsrechnung zum Konnossement {bl}", "fr": "facture commerciale du connaissement {bl}",
           "es": "factura comercial del conocimiento de embarque {bl}", "it": "fattura commerciale della polizza di carico {bl}",
           "pt": "fatura comercial do conhecimento de embarque {bl}", "nl": "handelsfactuur bij cognossement {bl}",
           "cs": "obchodní faktura ke konosamentu {bl}", "ar": "الفاتورة التجارية لبوليصة الشحن {bl}"},
    "R4": {"ko": "신용장 {lc} 결제 건의 상업송장", "en": "commercial invoices paid under L/C {lc}", "ja": "信用状{lc}で決済した商業送り状",
           "zh-hans": "用信用证{lc}结算的商业发票", "de": "Handelsrechnungen, die über das Akkreditiv {lc} bezahlt wurden",
           "fr": "factures commerciales réglées par la lettre de crédit {lc}", "es": "facturas comerciales pagadas con la carta de crédito {lc}",
           "it": "fatture commerciali pagate con la lettera di credito {lc}", "pt": "faturas comerciais pagas com a carta de crédito {lc}",
           "nl": "handelsfacturen betaald onder accreditief {lc}", "cs": "obchodní faktury placené akreditivem {lc}",
           "ar": "الفواتير التجارية المدفوعة بموجب خطاب الاعتماد {lc}"},
    "R5": {"ko": "{exporter}에서 보낸 선하증권", "en": "bills of lading shipped by {exporter}", "ja": "{exporter}が出荷した船荷証券",
           "zh-hans": "{exporter}发运的提单", "de": "Konnossemente, die {exporter} verschifft hat",
           "fr": "connaissements des envois de {exporter}", "es": "conocimientos de embarque de los envíos de {exporter}",
           "it": "polizze di carico delle spedizioni di {exporter}", "pt": "conhecimentos de embarque dos envios de {exporter}",
           "nl": "cognossementen van zendingen van {exporter}", "cs": "konosamenty zásilek od {exporter}",
           "ar": "بوالص الشحن لشحنات {exporter}"},
    "R6": {"ko": "{goods} 품목이 실린 선하증권", "en": "which bill of lading carried {goods}?", "ja": "{goods}を積んだ船荷証券",
           "zh-hans": "装运{goods}的提单", "de": "Mit welchem Konnossement wurde {goods} verschifft?",
           "fr": "quel connaissement a transporté {goods} ?", "es": "¿qué conocimiento de embarque llevó {goods}?",
           "it": "quale polizza di carico ha trasportato {goods}?", "pt": "qual conhecimento de embarque levou {goods}?",
           "nl": "met welk cognossement is {goods} verscheept?", "cs": "kterým konosamentem bylo odesláno zboží {goods}?",
           "ar": "ما بوليصة الشحن التي نقلت {goods}؟"},
    "R7": {"ko": "{month}에 {port}에서 선적된 서류", "en": "documents for the shipment that left {port} {month}",
           "ja": "{month}に{port}から出港した貨物の書類", "zh-hans": "{month}从{port}装船的单据",
           "de": "Unterlagen zur Verschiffung ab {port} {month}", "fr": "documents de l'expédition partie de {port} {month}",
           "es": "documentos del envío que salió de {port} {month}", "it": "documenti della spedizione partita da {port} {month}",
           "pt": "documentos do embarque que saiu de {port} {month}", "nl": "documenten van de zending die {month} uit {port} vertrok",
           "cs": "dokumenty k zásilce odeslané z {port} {month}", "ar": "مستندات الشحنة التي غادرت {port} {month}"},
    # ── 행동 로그 ──
    "A1": {"ko": "{site}에서 장바구니에 담은 기록", "en": "add-to-cart actions on {site}", "ja": "{site}でカートに入れた記録",
           "zh-hans": "{site}上加入购物车的记录", "de": "Wann wurde bei {site} etwas in den Warenkorb gelegt?",
           "fr": "ajouts au panier sur {site}", "es": "productos añadidos a la cesta en {site}", "it": "aggiunte al carrello su {site}",
           "pt": "itens adicionados ao carrinho na {site}", "nl": "toevoegingen aan de winkelwagen bij {site}",
           "cs": "vložení do košíku na {site}", "ar": "عمليات الإضافة إلى السلة على {site}"},
    "A2": {"ko": "{site} 상품 상세 페이지에서 옵션을 바꾼 기록", "en": "option changes on product pages of {site}",
           "ja": "{site}の商品詳細ページでオプションを変更した記録", "zh-hans": "在{site}商品详情页更改选项的记录",
           "de": "Optionsänderungen auf Produktseiten von {site}", "fr": "changements d'option sur les fiches produit de {site}",
           "es": "cambios de opción en las fichas de producto de {site}", "it": "modifiche delle opzioni nelle pagine prodotto di {site}",
           "pt": "alterações de opção nas páginas de produto da {site}", "nl": "optiewijzigingen op productpagina's van {site}",
           "cs": "změny možností na stránkách produktů {site}", "ar": "تغييرات الخيارات في صفحات المنتجات على {site}"},
    "A3": {"ko": "{goods} 장바구니 담기 기록", "en": "who added {goods} to the cart", "ja": "{goods}をカートに入れた記録",
           "zh-hans": "把{goods}加入购物车的记录", "de": "wer {goods} in den Warenkorb gelegt hat", "fr": "qui a ajouté {goods} au panier",
           "es": "quién añadió {goods} a la cesta", "it": "chi ha aggiunto {goods} al carrello", "pt": "quem adicionou {goods} ao carrinho",
           "nl": "wie {goods} in de winkelwagen heeft gelegd", "cs": "kdo vložil {goods} do košíku", "ar": "من أضاف {goods} إلى السلة"},
    "A4": {"ko": "{site}에서 결제 페이지까지 갔다가 결제하지 않은 기록", "en": "visits on {site} that reached checkout but never paid",
           "ja": "{site}で決済ページまで進んだのに支払わなかった記録", "zh-hans": "在{site}进入结算页面却没有付款的记录",
           "de": "Besuche bei {site}, die bis zur Kasse kamen, aber nicht bezahlt haben",
           "fr": "visites sur {site} arrivées au paiement sans commander", "es": "visitas en {site} que llegaron al pago pero no pagaron",
           "it": "visite su {site} arrivate al pagamento senza pagare", "pt": "visitas na {site} que chegaram ao pagamento mas não pagaram",
           "nl": "bezoeken bij {site} die tot de kassa kwamen maar niet betaalden",
           "cs": "návštěvy na {site}, které došly k pokladně, ale nezaplatily", "ar": "زيارات على {site} وصلت إلى صفحة الدفع دون أن تدفع"},
    "A5": {"ko": "'{event}' 이벤트 페이지에서 쿠폰 받은 기록", "en": "coupon downloads on the '{event}' event page",
           "ja": "「{event}」のイベントページでクーポンを受け取った記録", "zh-hans": "在“{event}”活动页面领取优惠券的记录",
           "de": "Gutschein-Downloads auf der Aktionsseite „{event}“", "fr": "coupons récupérés sur la page de l'opération « {event} »",
           "es": "cupones obtenidos en la página de la promoción «{event}»", "it": "coupon ritirati nella pagina della promozione «{event}»",
           "pt": "cupons resgatados na página da promoção “{event}”", "nl": "kortingsbonnen opgehaald op de actiepagina '{event}'",
           "cs": "stažené kupóny na stránce akce „{event}“", "ar": "القسائم التي تم الحصول عليها في صفحة عرض «{event}»"},
    "A6": {"ko": "{d1}부터 {d2}까지 {site} 모바일 터치 기록", "en": "mobile taps on {site} from {d1} to {d2}",
           "ja": "{d1}から{d2}までの{site}のモバイルタッチ記録", "zh-hans": "{d1}到{d2}期间{site}的手机触摸记录",
           "de": "mobile Touch-Eingaben bei {site} vom {d1} bis {d2}", "fr": "appuis mobiles sur {site} du {d1} au {d2}",
           "es": "toques en el móvil en {site} del {d1} al {d2}", "it": "tocchi da mobile su {site} dal {d1} al {d2}",
           "pt": "toques no celular na {site} de {d1} a {d2}", "nl": "mobiele tikken bij {site} van {d1} tot {d2}",
           "cs": "dotyky na mobilu na {site} od {d1} do {d2}", "ar": "لمسات الجوال على {site} من {d1} إلى {d2}"},
}
DOMAIN = {**{f"C{i}": "commerce" for i in range(1, 11)}, **{f"R{i}": "shipping" for i in range(1, 8)},
          **{f"A{i}": "analytic" for i in range(1, 7)}}
KIND = {"C1": "order→tracking", "C2": "tracking→order", "C3": "customer→order→goods", "C4": "goods+month→order",
        "C5": "coupon→order", "C6": "event→goods", "C7": "event period→order→goods", "C8": "customer→review→goods",
        "C9": "status→order→goods", "C10": "customer+month→tracking",
        "R1": "BL→CI·PL·CO·IP", "R2": "PO→BL×2", "R3": "BL→CI", "R4": "LC→CI×2", "R5": "exporter→BL×2", "R6": "goods→BL",
        "R7": "port+month→docs", "A1": "site+action", "A2": "site+page+action", "A3": "goods+action", "A4": "site+funnel",
        "A5": "event page+action", "A6": "site+period+device"}


def goods_name(rec: dict, lang: str) -> str:
    """상품 이름을 질의 언어로 (v1 상품 사전 · 언어 팩)."""
    import world as W
    if lang == "ko":
        return rec["title_ko"]
    if lang == "en":
        return rec["title_en"]
    key = {"zh-hans": "zh-hans"}.get(lang, lang)
    return W.gcp.PACKS[key]["products"][W.gcp.PRODUCT_KEYS[rec["title_en"]]][0]


def port_name(shop: str, lang: str) -> str:
    city = POL_CITY[shop]
    return PORT[city].get(lang, city)


# ──────────────────────────────────────────────
# 사례 고르기 (정답이 비지 않고, 슬롯이 질의를 실제로 좁히는 것)
# ──────────────────────────────────────────────
def _month(iso: str) -> int:
    return int(iso[5:7])


def commerce_cases(rec: dict, tmpl: str) -> dict:
    """쇼핑몰 하나에서 틀 하나의 사례: {slots, relevant(키 목록), note}."""
    k = rec["shop"]
    orders = rec["orders"]
    key = lambda t, i: f"{k}:{t}:{i}"  # noqa: E731
    goods = {g["id"]: g for g in rec["goods"]}
    live = [o for o in orders if o["status"] != "Cancelled"]
    if tmpl == "C1":
        o = next(o for o in sorted(orders, key=lambda o: o["id"]) if len(o["tracking"]) >= 2)
        return {"slots": {"order": o["id"]}, "relevant": [key("tracking", t) for t in o["tracking"]]}
    if tmpl == "C2":
        o = next(o for o in sorted(orders, key=lambda o: o["id"], reverse=True) if len(o["goods"]) >= 2 and o["tracking"])
        return {"slots": {"tracking": o["tracking"][0]}, "relevant": [key("order", o["id"])]}
    if tmpl == "C3":
        by = {}
        for o in live:
            by.setdefault(o["buyer"], []).append(o)
        name = sorted(by, key=lambda n: (-len(by[n]), n))[1]
        gs = sorted({g for o in by[name] for g in o["goods"]})
        return {"slots": {"customer": name}, "relevant": [key("goods", g) for g in gs]}
    if tmpl == "C4":
        cnt = {}
        for o in live:
            for g in o["goods"]:
                cnt.setdefault(g, []).append(o)
        gid = sorted(cnt, key=lambda g: (-len(cnt[g]), g))[0]
        months = sorted({_month(o["when"]) for o in cnt[gid]})
        m = months[0]
        hit = [o for o in cnt[gid] if _month(o["when"]) == m]
        return {"slots": {"goods_rec": goods[gid], "month": m}, "relevant": [key("order", o["id"]) for o in hit]}
    if tmpl == "C5":
        use = {}
        for o in orders:
            if o["coupon"]:
                use.setdefault(o["coupon"], []).append(o)
        cid = sorted(use, key=lambda c: (-len(use[c]), c))[0]
        cp = next(c for c in rec["coupons"] if c["id"] == cid)
        return {"slots": {"coupon": cp["title"]}, "relevant": [key("order", o["id"]) for o in use[cid]]}
    if tmpl == "C6":
        e = rec["events"][1]
        return {"slots": {"event": e["title"]}, "relevant": [key("goods", g) for g in e["goods"]]}
    if tmpl == "C7":
        e = rec["events"][1]
        sold = sorted({g for o in live if e["start"] <= o["when"][:10] <= e["end"] for g in o["goods"]})
        return {"slots": {"event": e["title"]}, "relevant": [key("goods", g) for g in sold]}
    if tmpl == "C8":
        by = {}
        for x in rec["reviews"]:
            by.setdefault(x["author"], []).append(x)
        name = sorted(by, key=lambda n: (-len(by[n]), n))[0]
        return {"slots": {"customer": name}, "relevant": [key("goods", g) for g in sorted({x["goods"] for x in by[name]})]}
    if tmpl == "C9":
        gs = sorted({g for o in orders if o["status"] == "Returned" for g in o["goods"]})
        return {"slots": {}, "relevant": [key("goods", g) for g in gs]}
    if tmpl == "C10":
        by = {}
        for s in rec["shipments"]:
            if s["kind"] == "outbound" and s["delivered"]:
                by.setdefault((s["receiver"], _month(s["delivered"])), []).append(s)
        (name, m) = sorted(by, key=lambda x: (-len(by[x]), x))[0]
        return {"slots": {"customer": name, "month": m}, "relevant": [key("tracking", s["id"]) for s in by[(name, m)]]}
    raise KeyError(tmpl)


def trade_cases(rec: dict, tmpl: str) -> dict:
    t = rec["trade"]
    k = rec["shop"]
    stem = lambda code, n: f"{k}_{code}{n}"  # noqa: E731
    ships = t["ships"]
    if tmpl == "R1":
        return {"slots": {"bl": ships[0]["bl"]}, "relevant": [stem(c, 1) for c in ("BL", "CI", "PL", "CO", "IP")]}
    if tmpl == "R2":
        return {"slots": {"po": t["numbers"]["po"]}, "relevant": [stem("BL", 1), stem("BL", 2)]}
    if tmpl == "R3":
        return {"slots": {"bl": ships[1]["bl"]}, "relevant": [stem("CI", 2)]}
    if tmpl == "R4":
        return {"slots": {"lc": t["numbers"]["lc"]}, "relevant": [stem("CI", 1), stem("CI", 2)]}
    if tmpl == "R5":
        return {"slots": {"exporter": t["exporter"]}, "relevant": [stem("BL", 1), stem("BL", 2)]}
    if tmpl == "R6":
        gid = ships[1]["goods"][0]
        g = next(x for x in rec["goods"] if x["id"] == gid)
        return {"slots": {"goods_rec": g}, "relevant": [stem("BL", 2)]}
    if tmpl == "R7":
        m = _month(ships[0]["etd"])
        return {"slots": {"port_shop": k, "month": m}, "relevant": [stem(c, 1) for c in ("CI", "PL", "BL", "CO", "IP")]}
    raise KeyError(tmpl)


def analytic_cases(rec: dict, events: list, tmpl: str) -> dict:
    k = rec["shop"]
    mine = [e for e in events if e["shop"] == k]
    site = rec["name"]
    if tmpl == "A1":
        return {"slots": {"site": site}, "relevant": [e["id"] for e in mine if e["action"] == "add_to_cart"]}
    if tmpl == "A2":
        return {"slots": {"site": site}, "relevant": [e["id"] for e in mine if e["action"] == "option" and e["page"] == "goods_view"]}
    if tmpl == "A3":
        cnt = {}
        for e in mine:
            if e["action"] == "add_to_cart":
                cnt.setdefault(e["goods"], []).append(e)
        gid = sorted(cnt, key=lambda g: (-len(cnt[g]), g))[0]
        g = next(x for x in rec["goods"] if x["id"] == gid)
        return {"slots": {"goods_rec": g}, "relevant": [e["id"] for e in cnt[gid]]}
    if tmpl == "A4":
        return {"slots": {"site": site}, "relevant": [e["id"] for e in mine if e["session_kind"] == "abandon" and e["page"] == "checkout"]}
    if tmpl == "A5":
        e = rec["events"][1]
        return {"slots": {"event": e["title"]},
                "relevant": [x["id"] for x in mine if x["action"] == "coupon" and x["page"] == "event_view" and x.get("event") == e["id"]]}
    if tmpl == "A6":
        taps = sorted((e for e in mine if e["type"] == "touch"), key=lambda e: e["at"])
        # 터치가 가장 많이 몰린 3주 창
        best = None
        for e in taps:
            d0 = date.fromisoformat(e["at"][:10])
            d1 = date.fromordinal(d0.toordinal() + 20)
            inside = [x for x in taps if d0.isoformat() <= x["at"][:10] <= d1.isoformat()]
            if best is None or len(inside) > len(best[2]):
                best = (d0, d1, inside)
        d0, d1, inside = best
        return {"slots": {"site": site, "d1": d0, "d2": d1}, "relevant": [e["id"] for e in inside]}
    raise KeyError(tmpl)


def fill(tmpl: str, lang: str, slots: dict) -> str:
    s = dict(slots)
    if "goods_rec" in s:
        s["goods"] = goods_name(s.pop("goods_rec"), lang)
    if "month" in s:
        s["month"] = MONTH[lang](s["month"])
    if "port_shop" in s:
        s["port"] = port_name(s.pop("port_shop"), lang)
    for d in ("d1", "d2"):
        if d in s:
            s[d] = DAY[lang](s[d])
    return T[tmpl][lang].format(**s)


def build(records: list, events: list) -> list:
    """사례 × 12개 언어 질의. 커머스 · 행동 로그는 틀마다 쇼핑몰 2곳, 무역은 틀마다 1곳(돌아가며).
    두 번째 쇼핑몰 사례는 진단(평가 축약)에 쓰지 않는 보류 집합입니다 — 행동 로그 쪽은 3차(2026-11-11)에 더했습니다:
    행동 로그 수정을 진단 72건의 로그에서 찾아 그 1.000 이 사례에 맞춘 것인지 가릴 질의가 없었습니다."""
    shops = [r["shop"] for r in records]
    by = {r["shop"]: r for r in records}
    out = []
    for i, tmpl in enumerate(sorted(T, key=lambda t: (t[0] != "C", t[0] != "R", int(t[1:])))):
        dom = DOMAIN[tmpl]
        n = int(tmpl[1:])
        if dom in ("commerce", "analytic"):
            picks = [shops[n % 4], shops[(n + 2) % 4]]
        else:
            picks = [shops[n % 4]]
        for shop in picks:
            rec = by[shop]
            case = (commerce_cases(rec, tmpl) if dom == "commerce" else trade_cases(rec, tmpl) if dom == "shipping"
                    else analytic_cases(rec, events, tmpl))
            assert case["relevant"], (tmpl, shop)
            for lang in LANGS:
                out.append({
                    "id": f"{tmpl}-{shop}-{lang}", "template": tmpl, "kind": KIND[tmpl], "domain": dom, "shop": shop,
                    "data_lang": rec["lang"] if dom != "shipping" else rec["trade"]["lang"], "lang": lang,
                    "host": rec["domain"] if dom == "commerce" else "", "query": fill(tmpl, lang, case["slots"]),
                    "relevant": case["relevant"],
                })
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")
    here = Path(__file__).resolve().parent
    records = yaml.safe_load((here / "records.yaml").read_text(encoding="utf-8"))
    events = yaml.safe_load((here / "weblog" / "truth.yaml").read_text(encoding="utf-8"))
    qs = build(records, events)
    print(len(qs), "queries", len({q["template"] + q["shop"] for q in qs}), "cases")
    for q in qs:
        if q["lang"] in ("ko", "de") and q["template"] in ("C4", "C10", "R6", "R7", "A6", "C3"):
            print(q["id"], "|", q["query"], "|", len(q["relevant"]))
